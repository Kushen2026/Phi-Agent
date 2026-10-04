// ACE-Step 1.5: the 32-layer DiT and the condition builder (encoder /
// tokenizer / detokenizer).
//
// ── the checkpoint ────────────────────────────────────────────────────────
//
// `acestep_v1.5_xl_sft_bf16.safetensors` (bf16, 9.97 GB) is one flat file:
//
//   decoder.layers.0..31    the DiT blocks (Qwen3-shaped: hidden 2560, 32 heads
//                           x 128 and 8 kv heads x 128, MLP 9728, q_norm/k_norm,
//                           RoPE theta 1e6, eps 1e-6)
//   decoder.proj_in         Conv1d(192 -> 2560, k=2, s=2) + bias
//   decoder.proj_out        ConvTranspose1d(2560 -> 64, k=2, s=2) + bias
//   decoder.time_embed[_r]  TimestepEmbedding(256 -> 2560) + a 2560 -> 6*2560
//                           modulation projection
//   decoder.condition_embedder  Linear(2048 -> 2560) + bias
//   encoder.text_projector  Linear(1024 -> 2048), no bias
//   encoder.lyric_encoder   embed_tokens(1024 -> 2048) + 8 blocks + norm
//   encoder.timbre_encoder  embed_tokens(64 -> 2048) + 4 blocks + norm +
//                           special_token
//   tokenizer.audio_acoustic_proj  Linear(64 -> 2048) + bias
//   tokenizer.attention_pooler     embed_tokens + special_token + 2 blocks
//   tokenizer.quantizer            FSQ (project_in 2048 -> 6, project_out 6 ->
//                                  2048, levels 8,8,8,5,5,5 -> 64000 codes)
//   detokenizer             embed_tokens + special_tokens[1,5,2048] + 2 blocks
//                           + norm + proj_out(2048 -> 64)
//
// A forward follows `comfy/ldm/ace/ace_step15.py` (the module shapes) and
// `comfy/text_encoders/ace15.py` (the conditioning) step for step. The places
// where this implementation deliberately differs are called out at the code
// that does it:
//
//   * the self-attention of the even-numbered blocks is a *symmetric* ±128 band
//     in the reference; the engine's flash kernels take one key range per query
//     block, so the band is implemented per 64-row query chunk with exact
//     per-row upper bounds and a lower edge that is the chunk's own start (see
//     `attn_band`). A dense dispatch is what the fp32 kernel would give instead.
//   * `time_embed_r` is evaluated at `timestep - timestep_r`, which is 0 in the
//     reference's own forward (it passes `timestep_r = timestep`).
//
// ── precision ─────────────────────────────────────────────────────────────
//
// Every projection below is an `AceLinear`: the weight is the checkpoint's own
// bf16 bytes and the GEMM reads them as bf16 with fp32 accumulation. The two
// paths that are *not* `AceLinear` are the ones whose shapes are not a plain
// [n, k] matrix (`proj_out`'s stacked taps, `proj_in`'s convolution) and the
// four `bf16_linear` calls, which were already bf16 - nothing here requantises.
#include "models/ace_models.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "kernels/gpu_ops.hpp"
#include "kernels/kernels.hpp"   // attn_query_tile()
#include "models/ace_text.hpp"
#include "models/video_vae_internal.hpp"   // dispatch_f32_to_f16
#include "runtime/sched.hpp"               // arena_chunk_cost / arena_chunk_bytes_for
#include "runtime/vram_budget.hpp"
#include "runtime/vram_window.hpp"

namespace phi::media {

namespace {

GpuAlloc sub(const GpuAlloc& a, u64 bytes) {
	GpuAlloc r = a;
	r.off += bytes;
	r.bytes = r.bytes > bytes ? r.bytes - bytes : 0;
	return r;
}
GpuAlloc slice(const GpuAlloc& a, u64 off, u64 bytes) {
	GpuAlloc r = sub(a, off);
	r.bytes = bytes;
	return r;
}

// y = x @ W^T + b against a weight the caller already uploaded as bf16. Used
// where the weight is not a plain [n, k] matrix (`proj_out`'s stacked taps, the
// timestep MLP) - the shape that is a matrix goes through `ace_linear`, which
// adds the dtype dispatch and the bias epilogue.
void bf16_linear(GpuCtx& g, const GpuAlloc& a, const GpuAlloc& b, const GpuAlloc& bias,
                 const GpuAlloc& c, i64 m, i64 n, i64 k, bool has_bias) {
	GemmF16Args ga;
	ga.a = a;
	ga.b = b;
	ga.bias = bias;
	ga.c = c;
	ga.m = m;
	ga.n = n;
	ga.k = k;
	ga.a_is_f32 = true;
	ga.b_is_bf16 = true;
	ga.has_bias = has_bias;
	dispatch_gemm_f16(*g.ctx, ga);
}

// y = a * (1 + b) + c, b and c broadcast over the rows: the DiT's adaLN pair
// `norm_hidden * (1 + scale) + shift`.
void modadd(GpuCtx& g, const GpuAlloc& y, const GpuAlloc& scale, const GpuAlloc& shift, i64 rows,
            i64 cols) {
	ElemArgs e;
	e.op = ElemOp::ModAdd;
	e.a = y;
	e.b = scale;
	e.c = shift;
	e.y = y;
	e.rows = rows;
	e.cols = cols;
	e.bMode = ElemMode::Row;
	e.cMode = ElemMode::Row;
	dispatch_elem(*g.ctx, e);
}

// y += a * gate, gate broadcast over the rows.
void gated_add(GpuCtx& g, const GpuAlloc& y, const GpuAlloc& a, const GpuAlloc& gate, i64 rows,
               i64 cols) {
	ElemArgs e;
	e.op = ElemOp::AddMul;
	e.a = y;
	e.b = a;
	e.c = gate;
	e.y = y;
	e.rows = rows;
	e.cols = cols;
	e.cMode = ElemMode::Row;
	dispatch_elem(*g.ctx, e);
}

void silu_(GpuCtx& g, const GpuAlloc& x, i64 n) {
	ElemArgs e;
	e.op = ElemOp::Silu;
	e.a = x;
	e.y = x;
	e.rows = 1;
	e.cols = n;
	dispatch_elem(*g.ctx, e);
}

// comfy/ldm/flux/layers.py :: timestep_embedding(t, dim, time_factor=1000):
// cos first, then sin, `max_period` 10000. `t` here is already the multiplier
// the engine was handed (`ModelSamplingDiscreteFlow.timestep(sigma)`, i.e.
// sigma for ACE 1.5), and the *embedding's* own scale of 1000 is applied here.
void timestep_embedding(float t, i64 dim, std::vector<float>& out) {
	out.assign((size_t)dim, 0.0f);
	const i64 half = dim / 2;
	const double tt = 1000.0 * (double)t;
	for (i64 j = 0; j < half; j++) {
		const double f = std::exp(-std::log(10000.0) * (double)j / (double)half);
		const double a = tt * f;
		out[(size_t)j] = (float)std::cos(a);
		out[(size_t)(half + j)] = (float)std::sin(a);
	}
}

// Upload a host fp32 buffer into the activations arena.
GpuAlloc upload_f32_into(GpuCtx& g, const GpuAlloc& a, const float* p, i64 n) {
	const bool own = !g.ctx->recording();
	if (own) g.ctx->begin();
	upload_range(*g.ctx, *g.ring, a.res, a.off, p, (u64)n * 4);
	if (own) {
		g.ctx->submit();
		g.ring->rewind();
	}
	return a;
}

// ── the silence latent ────────────────────────────────────────────────────
//
// `comfy/ldm/ace/ace_step15.py::get_silence_latent`: a [1, 64, L] tensor whose
// first four columns are fixed and whose remaining columns repeat one 64-vector.
// It is the "reference audio" the pipeline uses when the caller supplies none,
// and it is what the timbre encoder and (on the no-codes path) the tokenizer
// see. Stored as the five 64-wide columns (4 head columns + the repeated one).
const float kSilence[5][64] = {
	{0.570700f, 0.098200f, 0.690900f, -0.565800f, 0.626600f, 0.699600f, -0.136500f, -0.129100f,
	 -0.077600f, -0.117100f, -0.274300f, -0.842200f, -0.116800f, 1.553900f, -4.693600f, 0.743600f,
	 -1.184600f, -0.263700f, 0.693300f, -6.726600f, 0.096600f, -0.118700f, -0.350100f, -1.173600f,
	 0.058700f, -2.051700f, -1.365100f, 0.750800f, -0.249000f, -1.354800f, -0.129000f, -0.726100f,
	 1.113200f, -0.324900f, 0.233700f, 0.300400f, 0.660500f, -0.029800f, -0.198900f, -0.404100f,
	 0.284300f, -1.096300f, -0.551900f, 0.263900f, -1.043600f, -0.118300f, 0.064000f, 0.446000f,
	 -1.100100f, -0.617200f, -1.324100f, 1.137900f, 0.562300f, -0.150700f, -0.196300f, -0.474200f,
	 -2.469700f, 0.530200f, 0.538100f, 0.463600f, -0.178200f, -0.068700f, 1.033300f, 0.420200f},
	{0.304000f, -0.136700f, 0.620000f, 0.066500f, -0.064200f, 0.465500f, -0.118700f, -0.044000f,
	 0.294100f, -0.275300f, 0.017300f, -0.242100f, -0.014700f, 1.560300f, -2.702500f, 0.790700f,
	 -0.973600f, -0.068200f, 0.129400f, -5.070700f, -0.216700f, 0.330200f, -0.151300f, -0.810000f,
	 -0.389400f, -0.288400f, -0.314900f, 0.866000f, -0.381700f, -1.706100f, 0.582400f, -0.484000f,
	 0.693800f, 0.185900f, 0.175300f, 0.308100f, 0.019500f, 0.140300f, -0.075400f, -0.209100f,
	 0.125100f, -0.157800f, -0.496800f, -0.105200f, -0.455400f, -0.032000f, 0.128400f, 0.497400f,
	 -1.188900f, -0.034400f, -0.831300f, 0.295300f, 0.544500f, -0.624900f, -0.159500f, -0.068200f,
	 -3.141200f, 0.048400f, 0.415300f, 0.826000f, -0.152600f, -0.062500f, 0.536600f, 0.847300f},
	{0.053524f, -0.175340f, 0.544430f, -0.435010f, -0.002132f, 0.372000f, -0.004014f, -0.155160f,
	 -0.129680f, -0.153750f, -0.077107f, -0.205930f, -0.327800f, 1.514200f, -2.610100f, 0.586980f,
	 -1.271600f, -0.247730f, -0.027933f, -5.079900f, 0.116010f, 0.409870f, -0.022030f, -0.664950f,
	 -0.209950f, -0.634740f, -0.158930f, 0.827450f, -0.229920f, -1.681600f, 0.544400f, -0.495790f,
	 0.551280f, 0.304770f, 0.083052f, -0.061782f, 0.005904f, 0.295530f, -0.080645f, -0.100600f,
	 0.191440f, -0.381240f, -0.729490f, 0.024520f, -0.508140f, 0.239770f, 0.092943f, 0.392560f,
	 -1.199300f, -0.327520f, -0.727070f, 0.294760f, 0.435420f, -0.885970f, -0.416860f, -0.085390f,
	 -2.901800f, 0.064988f, 0.539450f, 0.919880f, 0.058762f, -0.070098f, 0.647720f, 0.891180f},
	{-0.032225f, -0.131950f, 0.564110f, -0.547660f, -0.005217f, 0.314250f, -0.054367f, -0.194190f,
	 -0.130590f, -0.136600f, -0.090984f, -0.195400f, -0.255900f, 1.544000f, -2.634900f, 0.682730f,
	 -1.253200f, -0.198100f, -0.022793f, -5.050600f, 0.188180f, 0.501090f, 0.007355f, -0.687710f,
	 -0.306760f, -0.732570f, -0.166870f, 0.922320f, -0.189870f, -1.726700f, 0.533550f, -0.531790f,
	 0.449530f, 0.288200f, 0.130120f, -0.209430f, -0.113480f, 0.339290f, -0.150690f, -0.129190f,
	 0.189290f, -0.361660f, -0.807560f, 0.066387f, -0.588670f, 0.169780f, 0.101340f, 0.338770f,
	 -1.213300f, -0.324920f, -0.812370f, 0.381010f, 0.437650f, -0.805960f, -0.445310f, -0.047513f,
	 -2.926600f, 0.001174f, 0.451230f, 0.930750f, 0.053688f, -0.196210f, 0.645300f, 0.938700f},
	{-0.136720f, -0.158200f, 0.585940f, -0.574220f, 0.030273f, 0.279300f, -0.002594f, -0.207030f,
	 -0.161130f, -0.147460f, -0.027710f, -0.180660f, -0.296880f, 1.601600f, -2.671900f, 0.777340f,
	 -1.351600f, -0.194340f, -0.071289f, -5.093800f, 0.243160f, 0.472660f, 0.046387f, -0.664060f,
	 -0.219730f, -0.675780f, -0.157230f, 0.953120f, -0.200200f, -1.710900f, 0.589840f, -0.574220f,
	 0.515620f, 0.283200f, 0.145510f, -0.187500f, -0.059814f, 0.367190f, -0.100590f, -0.157230f,
	 0.206050f, -0.433590f, -0.828120f, 0.045654f, -0.660160f, 0.148440f, 0.094727f, 0.384770f,
	 -1.257800f, -0.332030f, -0.855470f, 0.433590f, 0.423830f, -0.894530f, -0.503910f, -0.056152f,
	 -2.921900f, -0.024658f, 0.503910f, 0.984380f, 0.072754f, -0.215820f, 0.636720f, 1.000000f},
};

// [64, T] *channel-major* (the reference's [1, 64, T]).
void silence_latent(i64 T, std::vector<float>& out) {
	out.assign((size_t)64 * (size_t)T, 0.0f);
	for (i64 c = 0; c < 64; c++)
		for (i64 t = 0; t < T; t++)
			out[(size_t)c * (size_t)T + (size_t)t] = kSilence[(size_t)std::min<i64>(t, 4)][(size_t)c];
}

// ── FSQ (tokenizer.quantizer) ─────────────────────────────────────────────
//
// `ResidualFSQ(levels=[8,8,8,5,5,5], num_quantizers=1, dim=2048)`: one FSQ layer
// over the 6-wide projected code, scale 1, with the reference's soft clamp.
// Both entry points produce the same thing - a 6-vector in [-1, 1] - so
// `project_out` is the only place that has to know the formula.
const i32 kFsqLevels[6] = {8, 8, 8, 5, 5, 5};
const i32 kFsqBasis[6] = {1, 8, 64, 512, 2560, 12800};   // cumprod([1] + levels[:-1])
constexpr i32 kFsqSize = 64000;                          // 8*8*8*5*5*5
// The code a short `audio_codes` run is padded with
// (`F.pad(audio_codes, (0, ceil(T/5) - n), "constant", 35847)`).
constexpr i32 kFsqPadCode = 35847;

void fsq_index_to_vector(i32 code, float* v) {
	if (code < 0 || code >= kFsqSize) code = kFsqPadCode;
	for (int d = 0; d < 6; d++) {
		const i32 digit = (code / kFsqBasis[d]) % kFsqLevels[d];
		v[d] = (float)digit * (2.0f / (float)(kFsqLevels[d] - 1)) - 1.0f;
	}
}

// `FSQ.bound`: soft clamp -> tanh -> bracket -> floor -> rescale. (`codes` is
// the straight-through quantity, but in the forward direction the estimator is
// the identity, so the value is `2/(L-1) * floor(bracket) - 1`.)
void fsq_bound_vector(const float* z_in, float* v) {
	for (int d = 0; d < 6; d++) {
		const float lm1 = (float)(kFsqLevels[d] - 1);
		const float clampv = 1.0f + 1.0f / lm1;   // soft_clamp_input_value
		const float z = std::tanh(z_in[d] / clampv) * clampv;
		const float bracket = lm1 * (std::tanh(z) + 1.0f) / 2.0f + 0.5f;
		v[d] = 2.0f / lm1 * std::floor(bracket) - 1.0f;
	}
}

}  // namespace

// ── the debug dump (`PHI_DIT_DUMP=<prefix>`) ───────────────────────────────
//
// The same switch the H3 DiT uses (see av_dit.cpp): with it set, one forward
// writes the tensors a reference comparison needs - the model input, the two
// halves of the conditioning, the patched hidden stream at the two ends of the
// block loop, and the returned velocity - as raw fp32 files named
// `<prefix>ace_*.f32`. "The music is scrambled" has three very different causes
// (the conditioning, the block stack, the output head) and a clip cannot tell
// them apart; this can, in one run, and it is also what makes the 10 GB DiT
// checkable against `comfy/ldm/ace/ace_step15.py` at all - the reference can
// only be run one layer at a time, and only against files like these.
const char* ace_dump_prefix() {
	static int read = 0;
	static const char* v = nullptr;
	if (!read) {
		read = 1;
		v = getenv("PHI_DIT_DUMP");
		if (v && !*v) v = nullptr;
	}
	return v;
}

void ace_dump(const char* name, const float* p, size_t n) {
	const char* pre = ace_dump_prefix();
	if (!pre) return;
	const std::string path = std::string(pre) + name + ".f32";
	FILE* f = fopen(path.c_str(), "wb");
	if (!f) return;
	fwrite(p, 4, n, f);
	fclose(f);
}

void ace_dump_dev(GpuCtx& g, const GpuAlloc& a, i64 n_f32, const char* name) {
	if (!ace_dump_prefix()) return;
	g.ctx->submit_if_recording();
	std::vector<float> v = g.download_f32(a, (u64)n_f32);
	ace_dump(name, v.data(), v.size());
}

// ── the DiT ───────────────────────────────────────────────────────────────

namespace {

struct AceDitLayer {
	AceLinear q, k, v, o;        // self-attention
	AceLinear cq, ck, cv, co;    // cross-attention
	AceLinear gate, up, down;    // MLP
	GpuAlloc self_ln, cross_ln, mlp_ln, q_norm, k_norm, cq_norm, ck_norm;
	GpuAlloc sst;                // scale_shift_table, fp32 [6, hidden]
};

struct AceDitConfig {
	i64 hidden = 0, n_layers = 0, n_heads = 0, n_kv_heads = 0, head_dim = 0, intermediate = 0;
	i64 in_ch = 0, out_ch = 0, patch = 0, cond_dim = 0, t_dim = 0;
	// Not in the checkpoint: the reference's `sliding_window=128` and its
	// `layer_types` schedule (even layers sliding, odd layers full).
	i64 sliding_window = 128;
	float eps = 1e-6f, rope_theta = 1e6f;
};

}  // namespace

struct AceDiT::Impl {
	GpuCtx* g = nullptr;
	SafeTensors st_;
	AceDitConfig cfg;
	// Resident (keep arena): everything outside the 32 blocks.
	GpuAlloc proj_in_w, proj_in_b, norm_out_w, sst2;
	AceLinear cond_w;             // condition_embedder, at the bundle's precision
	GpuAlloc cond_b;              // its fp32 bias
	GpuAlloc out_w, out_b;        // the stacked proj_out form
	GpuAlloc te_w[2][3], te_b[2][3];   // linear_1 / linear_2 / time_proj, [t, r]
	GpuAlloc null_cond;

	// ── the resident weight window (`plan_residency`) ────────────────────
	// `res_arena` is the one arena `new_step()`/`new_layer()` never touch, so
	// the blocks uploaded into it survive the whole sampling loop. Its chunk
	// granularity is the weight arena's (the two plans are priced the same way),
	// and `res_layers[0..res_n)` are the blocks that live in it.
	GpuArena res_arena;
	std::vector<AceDitLayer> res_layers;
	i64 res_n = 0;
	u64 res_bytes = 0;
	u64 res_layer_bytes = 0;   // one block's tensor bytes, probed once
	u64 res_inflight = 0;      // what streaming one block charges the accountant
	u64 act_extra = 0;         // the activation frame beyond what `aa` already holds
	u64 plan_reserve = 0;      // act + inflight + held, the plan's own floor
	bool planned = false;

	void probe();
	void upload_layer(i64 index, AceDitLayer& L, GpuArena* into);
};

void AceDiT::Impl::probe() {
	const SafeTensors& st = st_;
	const StTensor& qp = st.require("decoder.layers.0.self_attn.q_proj.weight");
	cfg.hidden = qp.shape[1];
	cfg.head_dim = st.require("decoder.layers.0.self_attn.q_norm.weight").shape[0];
	if (cfg.head_dim != 128)
		throw MediaError("ace dit: head_dim " + std::to_string(cfg.head_dim) + " != 128");
	cfg.n_heads = qp.shape[0] / cfg.head_dim;
	cfg.n_kv_heads = st.require("decoder.layers.0.self_attn.k_proj.weight").shape[0] / cfg.head_dim;
	cfg.intermediate = st.require("decoder.layers.0.mlp.gate_proj.weight").shape[0];
	i64 l = 0;
	while (st.find("decoder.layers." + std::to_string(l) + ".self_attn.q_proj.weight")) l++;
	cfg.n_layers = l;
	if (cfg.n_layers <= 0) throw MediaError("ace dit: no decoder layers in the bundle");
	const StTensor& pin = st.require("decoder.proj_in.1.weight");
	if (pin.shape.size() != 3) throw MediaError("ace dit: proj_in is not [OC,IC,K]");
	cfg.in_ch = pin.shape[1];
	cfg.patch = pin.shape[2];
	const StTensor& pout = st.require("decoder.proj_out.1.weight");
	cfg.out_ch = pout.shape[1];
	cfg.cond_dim = st.require("decoder.condition_embedder.weight").shape[1];
	cfg.t_dim = st.require("decoder.time_embed.linear_1.weight").shape[1];

	// One block's own tensor bytes. The plan divides the card by this after the
	// activation frame and the streamed block have taken their share; the upload
	// loop below still measures the arena's *actual* growth per block, because a
	// block's eleven matrices do not tile the arena's chunk granularity.
	{
		const std::string p = "decoder.layers.0.";
		static const char* kMats[] = {"self_attn.q_proj",   "self_attn.k_proj",
		                              "self_attn.v_proj",   "self_attn.o_proj",
		                              "cross_attn.q_proj",  "cross_attn.k_proj",
		                              "cross_attn.v_proj",  "cross_attn.o_proj",
		                              "mlp.gate_proj",      "mlp.up_proj",
		                              "mlp.down_proj"};
		u64 b = 0;
		for (const char* n : kMats) {
			const StTensor& t = st.require(p + n + ".weight");
			// What `ace_upload_linear` actually allocates, which is not
			// `numel * dtype_size` for every family: a quantised weight lands on the
			// int8 tensorwise pair (one code per element plus one fp32 scale per
			// output row), and a float one keeps the file's own width. Pricing the
			// raw `numel * dtype_size` here made the resident window a block too
			// large for a quantised bundle, which is exactly the overrun the plan
			// exists to prevent.
			if (weight_is_dense_float(st, t)) {
				const DType as = t.dtype == DType::F32    ? DType::F32
				                 : t.dtype == DType::BF16 ? DType::BF16
				                                          : DType::F16;   // fp8 / nvfp4 -> f16
				b += (u64)t.shape[0] * (u64)t.shape[1] * dtype_size(as);
			} else {
				b += (u64)t.shape[0] * (u64)t.shape[1] + (u64)t.shape[0] * 4;
			}
		}
		// The seven RMSNorm vectors (3 x hidden + 2 x hidden + 2 x kv-dim) and the
		// scale_shift_table (6 x hidden) are widened to fp32 on upload.
		b += (u64)(3 * cfg.hidden + 6 * cfg.hidden + 2 * cfg.hidden +
		           2 * cfg.n_kv_heads * cfg.head_dim) *
		     4;
		res_layer_bytes = b;
	}
}

void AceDiT::Impl::upload_layer(i64 index, AceDitLayer& L, GpuArena* into) {
	const std::string p = "decoder.layers." + std::to_string(index) + ".";
	// The checkpoint's own bytes. With `into == nullptr` the weight arena is reset
	// by the next `new_layer()`, so that block is a per-block stream; with `into`
	// naming the resident arena the upload is a *keep* and the block is reused by
	// every sampling step (see `plan_residency`).
	auto lin = [&](const std::string& n) { return ace_upload_linear(*g, st_, p + n, into); };
	L.q = lin("self_attn.q_proj");
	L.k = lin("self_attn.k_proj");
	L.v = lin("self_attn.v_proj");
	L.o = lin("self_attn.o_proj");
	L.cq = lin("cross_attn.q_proj");
	L.ck = lin("cross_attn.k_proj");
	L.cv = lin("cross_attn.v_proj");
	L.co = lin("cross_attn.o_proj");
	L.gate = lin("mlp.gate_proj");
	L.up = lin("mlp.up_proj");
	L.down = lin("mlp.down_proj");
	L.self_ln = ace_upload_vector(*g, st_, p + "self_attn_norm.weight", into);
	L.cross_ln = ace_upload_vector(*g, st_, p + "cross_attn_norm.weight", into);
	L.mlp_ln = ace_upload_vector(*g, st_, p + "mlp_norm.weight", into);
	L.q_norm = ace_upload_vector(*g, st_, p + "self_attn.q_norm.weight", into);
	L.k_norm = ace_upload_vector(*g, st_, p + "self_attn.k_norm.weight", into);
	L.cq_norm = ace_upload_vector(*g, st_, p + "cross_attn.q_norm.weight", into);
	L.ck_norm = ace_upload_vector(*g, st_, p + "cross_attn.k_norm.weight", into);
	L.sst = ace_upload_vector(*g, st_, p + "scale_shift_table", into);   // [1,6,H] -> [6,H]
}

AceDiT::AceDiT() = default;
AceDiT::~AceDiT() = default;

void AceDiT::open(const std::string& path, GpuCtx* gpu) {
	impl_ = std::make_unique<Impl>();
	Impl& m = *impl_;
	m.g = gpu;
	m.st_.open(path);
	m.probe();
	const AceDitConfig& cfg = m.cfg;
	GpuCtx& g = *gpu;

	// The resident window's arena. Its chunk size is the weight arena's, so the
	// plan's charge and the accountant's are the same number (see
	// `ImageDiT::open`, which is the same decision).
	m.res_arena.init(g.ctx, arena_chunk_bytes_for(vram_budget().limit()));
	m.res_arena.set_tag("ace.dit_resident");

	g.new_step();
	g.ctx->begin();
	auto bf16 = [&](const std::string& n) {
		const StTensor& t = m.st_.require(n);
		// The ACE-Step 1.5 quantizer/decoder tables are bf16 in the checkpoint, and
		// the kernel that reads them (`conv2d`'s projection, and the `bf16_linear`
		// helpers' operands) is written for the file's own width - so this converts
		// only if a re-exported checkpoint stored a different one, and never the
		// reverse.
		GpuAlloc a = g.kalloc(t.numel * 2);
		g.upload_tensor_into(m.st_, t, DType::BF16, a);
		return a;
	};
	auto f32v = [&](const std::string& n) {
		std::vector<float> v = tensor_to_f32(m.st_, m.st_.require(n));
		GpuAlloc a = g.kalloc((u64)v.size() * 4);
		upload_range(*g.ctx, *g.ring, a.res, a.off, v.data(), v.size() * 4);
		return a;
	};
	m.proj_in_w = bf16("decoder.proj_in.1.weight");
	m.proj_in_b = f32v("decoder.proj_in.1.bias");
	m.norm_out_w = f32v("decoder.norm_out.weight");
	m.sst2 = f32v("decoder.scale_shift_table");   // [1,2,H] -> [2,H]
	m.cond_w = ace_upload_linear(g, m.st_, "decoder.condition_embedder", g.keep);
	if (m.cond_w.k != cfg.cond_dim || m.cond_w.n != cfg.hidden)
		throw MediaError("ace dit: condition_embedder shape");
	m.cond_b = f32v("decoder.condition_embedder.bias");
	m.null_cond = f32v("null_condition_emb");
	for (int i = 0; i < 2; i++) {
		const std::string p = i == 0 ? "decoder.time_embed." : "decoder.time_embed_r.";
		m.te_w[i][0] = bf16(p + "linear_1.weight");
		m.te_b[i][0] = f32v(p + "linear_1.bias");
		m.te_w[i][1] = bf16(p + "linear_2.weight");
		m.te_b[i][1] = f32v(p + "linear_2.bias");
		m.te_w[i][2] = bf16(p + "time_proj.weight");
		m.te_b[i][2] = f32v(p + "time_proj.bias");
	}
	// `proj_out` is a ConvTranspose1d(2560 -> 64, k=2, s=2, pad=0), i.e.
	// out[2i + k] = x[i] @ W[:, :, k] + b. Both taps are one GEMM against a
	// stacked [2*out_ch, hidden] weight whose row `k*out_ch + oc` is tap k's
	// output row, and the host interleaves the halves afterwards (which also
	// keeps the reference's epilogue order).
	{
		const StTensor& w = m.st_.require("decoder.proj_out.1.weight");   // [hidden, out_ch, k]
		if (w.shape.size() != 3 || w.shape[0] != cfg.hidden || w.shape[1] != cfg.out_ch ||
		    w.shape[2] != cfg.patch)
			throw MediaError("ace dit: proj_out shape");
		if (w.dtype != DType::BF16)
			throw MediaError("ace dit: proj_out is not bf16 (the tap restack is byte-wise)");
		const u16* src = (const u16*)m.st_.data_of(w);
		std::vector<u16> sw((size_t)(cfg.out_ch * cfg.patch) * (size_t)cfg.hidden);
		for (i64 ic = 0; ic < cfg.hidden; ic++)
			for (i64 oc = 0; oc < cfg.out_ch; oc++)
				for (i64 kk = 0; kk < cfg.patch; kk++)
					sw[(size_t)(kk * cfg.out_ch + oc) * (size_t)cfg.hidden + (size_t)ic] =
					    src[((size_t)ic * (size_t)cfg.out_ch + (size_t)oc) * (size_t)cfg.patch +
					        (size_t)kk];
		m.out_w = g.kalloc((u64)sw.size() * 2);
		upload_range(*g.ctx, *g.ring, m.out_w.res, m.out_w.off, sw.data(), sw.size() * 2);
		m.out_b = f32v("decoder.proj_out.1.bias");
	}
	g.ctx->submit();
	g.ring->rewind();
}

std::vector<float> AceDiT::forward(const float* x, i64 T, float sigma, const AceCondition& cond) {
	if (!impl_) throw MediaError("ace dit: not open");
	Impl& m = *impl_;
	const AceDitConfig& cfg = m.cfg;
	GpuCtx& g = *m.g;
	if (T <= 0) throw MediaError("ace dit: empty latent");
	if ((i64)cond.context.size() != cond.T * 2 * cfg.out_ch)
		throw MediaError("ace dit: context is not [T, 2*out_ch]");
	if (cond.T < T) throw MediaError("ace dit: the condition context is shorter than the latent");

	ace_dump("ace_in_x", x, (size_t)T * (size_t)cfg.out_ch);
	ace_dump("ace_in_ctx", cond.context.data(), cond.context.size());
	ace_dump("ace_in_enc", cond.encoder_hidden.data(), cond.encoder_hidden.size());
	if (ace_dump_prefix())
		fprintf(stderr, "[acedit] T=%lld S=%lld sigma=%.6f n_enc=%lld\n", (long long)T,
		        (long long)ceil_div(T, cfg.patch), (double)sigma, (long long)cond.n);
	const i64 H = cfg.hidden;
	const i64 HQ = cfg.n_heads * cfg.head_dim;
	const i64 HK = cfg.n_kv_heads * cfg.head_dim;
	const i64 I = cfg.intermediate;
	const i64 patch = cfg.patch;
	const i64 Tp = ceil_div(T, patch) * patch;
	const i64 S = Tp / patch;
	const i64 tile = std::max<i64>(64, (i64)attn_query_tile());
	const i64 W = cfg.sliding_window;
	const float scale = 1.0f / std::sqrt((float)cfg.head_dim);

	// The condition: `encoder_hidden` [n, 2048] with n = 0 meaning "the caller
	// wants the null embedding" (`replace_with_null_embeds` in the reference).
	const bool null_cond = cond.n <= 0;
	const i64 n_enc = null_cond ? 1 : cond.n;
	if (!null_cond && (i64)cond.encoder_hidden.size() < n_enc * cfg.cond_dim)
		throw MediaError("ace dit: encoder_hidden is not [n, cond_dim]");
	// The engine's flash kernel uses ONE sequence length for both the query rows
	// and the key range (`aa.s`): the key tile loop runs to ceil(s/BN) and the
	// per-row key bound is what clamps a row to its real key set. So the
	// cross-attention dispatch needs `s >= S` even when the condition is shorter
	// than the latent - which it usually is, since the timbre encoder contributes
	// one token per latent frame, but nothing guarantees it. The two K/V buffers
	// are therefore sized `n_kv`, and their rows past the real key count are
	// zero-filled: read by the tile loop, then excluded by `ub_cross`.
	const i64 n_kv = std::max(n_enc, S);

	g.new_step();
	const i64 rows_att = S + tile;   // the tiled kernel writes a whole query tile
	auto A = [&](i64 n) { return g.aalloc((u64)n * 4); };
	GpuAlloc inp = A(cfg.in_ch * Tp);
	GpuAlloc cin = A(H * S);
	GpuAlloc hin = A(H * S);
	GpuAlloc enc = A(H * n_kv);          // the raw condition encoder output
	GpuAlloc ench = A(H * n_kv);         // ... after condition_embedder
	GpuAlloc t = A(H * S);
	GpuAlloc mo = A(6 * H);
	GpuAlloc qr = A(HQ * rows_att);
	GpuAlloc qn = A(HQ * S);
	GpuAlloc qh = g.aalloc((u64)HQ * (u64)rows_att * 2);
	GpuAlloc kr = A(HK * S);
	GpuAlloc kn = A(HK * S);
	GpuAlloc vr = A(HK * S);
	GpuAlloc kh = g.aalloc((u64)HK * (u64)S * 2);
	GpuAlloc vh = g.aalloc((u64)HK * (u64)S * 2);
	GpuAlloc attn = A(HQ * rows_att);
	GpuAlloc proj = A(H * S);
	GpuAlloc ckr = A(HK * n_kv);
	GpuAlloc ckn = A(HK * n_kv);
	GpuAlloc ckh = g.aalloc((u64)HK * (u64)n_kv * 2);
	GpuAlloc cvh = g.aalloc((u64)HK * (u64)n_kv * 2);
	GpuAlloc gate = A(I * S);
	GpuAlloc up = A(I * S);
	GpuAlloc hh = A(I * S);
	GpuAlloc down = A(H * S);
	GpuAlloc outy = A(cfg.out_ch * patch * S);
	GpuAlloc te_in = A(cfg.t_dim);
	GpuAlloc te_in_r = A(cfg.t_dim);
	GpuAlloc te1 = A(H);
	GpuAlloc temb_t = A(H);
	GpuAlloc temb_r = A(H);
	GpuAlloc proj_t = A(6 * H);
	GpuAlloc proj_r = A(6 * H);
	GpuAlloc ub_self = g.aalloc((u64)std::max(S, tile) * 4);
	GpuAlloc ub_cross = g.aalloc((u64)std::max(std::max(n_enc, S), tile) * 4);
	// The int8 weights' activation scratch (see `AceLinear`): only touched when the
	// bundle is quantised, and sized here for the widest (rows, k) any projection of
	// this pass feeds - `rows` is the padded query tile for q/cross-q, `S` for the
	// rest, and `k` is one of H / I / cond_dim.
	AceQScratch qs;
	qs.need(g, rows_att, std::max(H, std::max(I, cfg.cond_dim)));

	// ── the model input: [context(T, 128) | x(T, 64)] padded to `patch` rows ──
	{
		std::vector<float> in((size_t)cfg.in_ch * (size_t)Tp, 0.0f);
		for (i64 tt = 0; tt < T; tt++) {
			for (i64 c = 0; c < 2 * cfg.out_ch; c++)
				in[(size_t)c * (size_t)Tp + (size_t)tt] =
				    cond.context[(size_t)tt * (size_t)(2 * cfg.out_ch) + (size_t)c];
			for (i64 c = 0; c < cfg.out_ch; c++)
				in[(size_t)(2 * cfg.out_ch + c) * (size_t)Tp + (size_t)tt] =
				    x[(size_t)tt * (size_t)cfg.out_ch + (size_t)c];
		}
		upload_f32_into(g, inp, in.data(), (i64)in.size());
	}
	{
		// Conv1d(192 -> 2560, k=2, s=2) as a kh=1 NCHW convolution over
		// [1, 192, 1, Tp], then channel-major -> token-major.
		Conv2dArgsG ca;
		ca.x = inp;
		ca.wgt = m.proj_in_w;
		ca.bias = m.proj_in_b;
		ca.y = cin;
		ca.n = 1;
		ca.ic = cfg.in_ch;
		ca.ih = 1;
		ca.iw = Tp;
		ca.oc = H;
		ca.kh = 1;
		ca.kw = patch;
		ca.stride = patch;
		ca.pad = 0;
		ca.has_bias = true;
		ca.w_dtype = 1;   // the ACE-Step quantizer's projection is bf16 in the checkpoint
		dispatch_conv2d(*g.ctx, ca);
		dispatch_transpose_cs(*g.ctx, cin, hin, H, S, true);
	}

	// ── the timestep heads ──
	//
	// `AceStepConditionGenerationModel.forward` passes `timestep_r = timestep`,
	// so the second head is evaluated at 0 - a constant. It is a real (trained)
	// head, so it is computed the same way the first one is.
	{
		std::vector<float> e;
		timestep_embedding(sigma, cfg.t_dim, e);
		upload_f32_into(g, te_in, e.data(), cfg.t_dim);
		timestep_embedding(0.0f, cfg.t_dim, e);
		upload_f32_into(g, te_in_r, e.data(), cfg.t_dim);
		auto head = [&](int which, const GpuAlloc& tin, const GpuAlloc& temb, const GpuAlloc& prj) {
			bf16_linear(g, tin, m.te_w[which][0], m.te_b[which][0], te1, 1, H, cfg.t_dim, true);
			silu_(g, te1, H);
			bf16_linear(g, te1, m.te_w[which][1], m.te_b[which][1], temb, 1, H, H, true);
			{
				ElemArgs ce;
				ce.op = ElemOp::Copy;
				ce.a = temb;
				ce.y = te1;
				ce.rows = 1;
				ce.cols = H;
				dispatch_elem(*g.ctx, ce);
			}
			silu_(g, te1, H);
			bf16_linear(g, te1, m.te_w[which][2], m.te_b[which][2], prj, 1, 6 * H, H, true);
		};
		head(0, te_in, temb_t, proj_t);
		head(1, te_in_r, temb_r, proj_r);
		ace_add_inplace(g, temb_t, temb_r, 1, H);       // temb = temb_t + temb_r
		ace_add_inplace(g, proj_t, proj_r, 1, 6 * H);   // timestep_proj = proj_t + proj_r
	}

	// ── the condition encoder output (condition_embedder + bias) ──
	//
	// The buffer is `n_kv` rows: the real keys, then zeros. The tail exists only
	// for the flash kernel's key-tile loop (see `n_kv` above) and those rows are
	// excluded by the mask from every query row, but they are still *read* by the
	// projections below, so they are written explicitly rather than left as
	// whatever the arena handed back. Reusing a block that still holds a NaN would
	// otherwise put NaN into the K/V rows that the mask multiplies by zero.
	{
		std::vector<float> host((size_t)n_kv * (size_t)cfg.cond_dim, 0.0f);
		if (null_cond) {
			std::vector<float> nc((size_t)cfg.cond_dim);
			g.ctx->download(m.null_cond.res, m.null_cond.off, nc.data(),
			                (size_t)cfg.cond_dim * 4);
			std::memcpy(host.data(), nc.data(), (size_t)cfg.cond_dim * 4);
		} else {
			std::memcpy(host.data(), cond.encoder_hidden.data(),
			            (size_t)(n_enc * cfg.cond_dim) * 4);
		}
		upload_f32_into(g, enc, host.data(), (i64)host.size());
	}
	// `ench` is a separate buffer: the bf16 GEMM's operand and its accumulator
	// are not the same layout, and aliasing them is not something this kernel
	// promises.
	ace_linear(g, m.cond_w, enc, n_kv, ench, m.cond_b, &qs);

	// ── the tiled-kernel key bounds ──
	//
	// `dispatch_attn_flash` reads one per-query-row *exclusive* key bound; every
	// dispatch below passes one, which is also what pins the fp16 tiled kernel
	// (the fp32 kernel has no mask and would read packed halves as fp32).
	{
		std::vector<u32> us((size_t)std::max(S, tile), (u32)S);
		std::vector<u32> uc((size_t)std::max(std::max(n_enc, S), tile), (u32)n_enc);
		g.ctx->begin();
		upload_range(*g.ctx, *g.ring, ub_self.res, ub_self.off, us.data(), us.size() * 4);
		upload_range(*g.ctx, *g.ring, ub_cross.res, ub_cross.off, uc.data(), uc.size() * 4);
		g.ctx->submit();
		g.ring->rewind();
	}

	ace_dump_dev(g, hin, S * H, "ace_h_proj");

	// ── the block loop ──
	//
	// Blocks [0, res_n) are already uploaded into the resident arena by
	// `plan_residency` and are used *in place* for every step - `new_layer()`
	// resets the streaming weight arena, not that one, so skipping the reset is
	// what keeps them live. Everything past the window streams through the weight
	// arena exactly as it did before the window existed.
	for (i64 li = 0; li < cfg.n_layers; li++) {
		AceDitLayer streamed;
		AceDitLayer* lp = nullptr;
		if (li < m.res_n && li < (i64)m.res_layers.size()) {
			lp = &m.res_layers[(size_t)li];
			// No arena reset: the previous block's `end_layer()` already submitted,
			// so this only opens the next command list.
			g.ctx->begin();
		} else {
			g.new_layer();
			m.upload_layer(li, streamed, nullptr);
			lp = &streamed;
		}
		AceDitLayer& L = *lp;

		// modulation = scale_shift_table + timestep_proj, chunked 6 ways:
		// shift, scale, gate (self-attention), c_shift, c_scale, c_gate (MLP).
		{
			ElemArgs e;
			e.op = ElemOp::Add;
			e.a = L.sst;
			e.b = proj_t;
			e.y = mo;
			e.rows = 6;
			e.cols = H;
			dispatch_elem(*g.ctx, e);
		}
		const GpuAlloc sh1 = slice(mo, (u64)H * 0, (u64)H * 4);
		const GpuAlloc sc1 = slice(mo, (u64)H * 4, (u64)H * 4);
		const GpuAlloc gt1 = slice(mo, (u64)H * 8, (u64)H * 4);
		const GpuAlloc sh2 = slice(mo, (u64)H * 12, (u64)H * 4);
		const GpuAlloc sc2 = slice(mo, (u64)H * 16, (u64)H * 4);
		const GpuAlloc gt2 = slice(mo, (u64)H * 20, (u64)H * 4);

		// ── self-attention ──
		ace_rms_norm(g, hin, S, H, L.self_ln, t, cfg.eps);
		modadd(g, t, sc1, sh1, S, H);
		ace_linear(g, L.q, t, S, qr, GpuAlloc{}, &qs);
		ace_linear(g, L.k, t, S, kr, GpuAlloc{}, &qs);
		ace_linear(g, L.v, t, S, vr, GpuAlloc{}, &qs);
		ace_rms_norm(g, qr, S * cfg.n_heads, cfg.head_dim, L.q_norm, qn, cfg.eps);
		ace_rms_norm(g, kr, S * cfg.n_kv_heads, cfg.head_dim, L.k_norm, kn, cfg.eps);
		{
			// HF half-split RoPE at absolute positions 0..S-1: the DiT's
			// RotaryEmbedding is `emb = cat((freqs, freqs))` + rotate_half, which
			// is `dispatch_rope_half`, not `qkv_prep`'s interleaved pairs.
			RopeHalfArgs ra;
			ra.head_dim = cfg.head_dim;
			ra.theta = cfg.rope_theta;
			ra.base = 0;
			ra.rows = S * cfg.n_heads;
			ra.heads = cfg.n_heads;
			ra.x = qn;
			ra.y = qr;
			dispatch_rope_half(*g.ctx, qn.res, ra);
			ra.rows = S * cfg.n_kv_heads;
			ra.heads = cfg.n_kv_heads;
			ra.x = kn;
			ra.y = kr;
			dispatch_rope_half(*g.ctx, kn.res, ra);
		}
		dispatch_f32_to_f16(*g.ctx, qr, qh, S * HQ);
		dispatch_f32_to_f16(*g.ctx, kr, kh, S * HK);
		dispatch_f32_to_f16(*g.ctx, vr, vh, S * HK);
		auto attn_dispatch = [&](const GpuAlloc& qs, const GpuAlloc& ks, const GpuAlloc& vs,
		                         const GpuAlloc& os, i64 nrows, i64 keys, const GpuAlloc& ub) {
			AttnFlashArgs aa;
			aa.q = qs;
			aa.k = ks;
			aa.v = vs;
			aa.o = os;
			aa.s = keys;
			aa.q_rows = nrows;
			aa.heads = cfg.n_heads;
			aa.kv_heads = cfg.n_kv_heads;
			aa.head_dim = cfg.head_dim;
			aa.f32_input = false;   // fp16 operands -> the tiled kernel (and `ub`)
			aa.causal = false;      // the mask is always `ub`
			aa.ub = ub;
			aa.scale = scale;
			dispatch_attn_flash(*g.ctx, aa);
		};
		if ((li % 2) != 0) {
			// `full_attention` (the reference's odd rows): everything in one
			// dispatch.
			attn_dispatch(qh, kh, vh, attn, S, S, ub_self);
		} else {
			// `sliding_attention` (even rows): `|i - j| <= 128`.
			//
			// The reference masks a *symmetric* band. The engine's flash kernels
			// take one key range per query block and only an *upper* bound per
			// query row, so the band is walked a query tile at a time: rows
			// [q0, q0 + tile) against the key range [q0 - W, q0 + tile + W), with
			// the exact per-row upper bound and a lower edge that is the tile's
			// own start. Every row therefore sees its true ±W keys plus at most
			// `tile - 1` (63) older ones - the alternative the kernels *can*
			// express, and much closer to the reference than the dense attention
			// the fp32 path would give.
			std::vector<u32> ubv((size_t)tile, 0);
			for (i64 q0 = 0; q0 < S; q0 += tile) {
				const i64 rows = std::min(tile, S - q0);
				const i64 k0 = std::max<i64>(0, q0 - W);
				const i64 k1 = std::min(S, q0 + tile + W);
				const i64 sl = k1 - k0;
				for (i64 i = 0; i < tile; i++) {
					const i64 bound = (q0 + i + W + 1) - k0;
					ubv[(size_t)i] = (u32)std::min<i64>(sl, std::max<i64>(0, bound));
				}
				GpuAlloc ubc = ace_upload_u32(g, ubv.data(), tile);
				attn_dispatch(sub(qh, (u64)q0 * (u64)HQ * 2), sub(kh, (u64)k0 * (u64)HK * 2),
				              sub(vh, (u64)k0 * (u64)HK * 2), sub(attn, (u64)q0 * (u64)HQ * 4), rows,
				              sl, ubc);
			}
		}
		ace_linear(g, L.o, attn, S, proj, GpuAlloc{}, &qs);
		gated_add(g, hin, proj, gt1, S, H);

		// ── cross-attention ──
		//
		// `AceStepAttention` with `is_cross_attention=True`: q from the latent,
		// k/v from the encoder output, `q_norm`/`k_norm` per head, and *no* RoPE
		// on either side (position_embeddings is only applied on the self path).
		ace_rms_norm(g, hin, S, H, L.cross_ln, t, cfg.eps);
		ace_linear(g, L.cq, t, S, qr, GpuAlloc{}, &qs);
		ace_rms_norm(g, qr, S * cfg.n_heads, cfg.head_dim, L.cq_norm, qn, cfg.eps);
		dispatch_f32_to_f16(*g.ctx, qn, qh, S * HQ);
		ace_linear(g, L.ck, ench, n_kv, ckr, GpuAlloc{}, &qs);
		ace_rms_norm(g, ckr, n_kv * cfg.n_kv_heads, cfg.head_dim, L.ck_norm, ckn, cfg.eps);
		ace_linear(g, L.cv, ench, n_kv, vr, GpuAlloc{}, &qs);
		dispatch_f32_to_f16(*g.ctx, ckn, ckh, n_kv * HK);
		dispatch_f32_to_f16(*g.ctx, vr, cvh, n_kv * HK);
		attn_dispatch(qh, ckh, cvh, attn, S, n_kv, ub_cross);
		ace_linear(g, L.co, attn, S, proj, GpuAlloc{}, &qs);
		ace_add_inplace(g, hin, proj, S, H);

		// ── MLP ──
		ace_rms_norm(g, hin, S, H, L.mlp_ln, t, cfg.eps);
		modadd(g, t, sc2, sh2, S, H);
		ace_linear(g, L.gate, t, S, gate, GpuAlloc{}, &qs);
		ace_linear(g, L.up, t, S, up, GpuAlloc{}, &qs);
		{
			ElemArgs e;
			e.op = ElemOp::SiluGate;
			e.a = gate;
			e.b = up;
			e.y = hh;
			e.rows = S;
			e.cols = I;
			dispatch_elem(*g.ctx, e);
		}
		ace_linear(g, L.down, hh, S, down, GpuAlloc{}, &qs);
		gated_add(g, hin, down, gt2, S, H);
		g.end_layer();
	}

	ace_dump_dev(g, hin, S * H, "ace_h_blocks");

	// ── the output head ──
	// x = norm_out(x) * (1 + scale) + shift, with (shift, scale) = sst2 + temb.
	g.new_layer();
	ace_rms_norm(g, hin, S, H, m.norm_out_w, t, cfg.eps);
	{
		ElemArgs e;
		e.op = ElemOp::Add;
		e.a = m.sst2;
		e.b = temb_t;
		e.y = mo;               // [2, H]: row 0 = shift, row 1 = scale
		e.rows = 2;
		e.cols = H;
		e.bMode = ElemMode::Row;
		dispatch_elem(*g.ctx, e);
	}
	modadd(g, t, slice(mo, (u64)H * 4, (u64)H * 4), slice(mo, 0, (u64)H * 4), S, H);
	bf16_linear(g, t, m.out_w, m.out_b, outy, S, cfg.out_ch * patch, H, false);
	g.end_layer();

	std::vector<float> flat = g.download_f32(outy, (u64)S * (u64)(cfg.out_ch * patch));
	std::vector<float> bias;
	{
		bias = g.download_f32(m.out_b, (u64)cfg.out_ch);
	}
	std::vector<float> out((size_t)T * (size_t)cfg.out_ch, 0.0f);
	for (i64 tt = 0; tt < T; tt++) {
		const i64 i = tt / patch, kk = tt % patch;
		for (i64 oc = 0; oc < cfg.out_ch; oc++)
			out[(size_t)tt * (size_t)cfg.out_ch + (size_t)oc] =
			    flat[(size_t)i * (size_t)(cfg.out_ch * patch) + (size_t)(kk * cfg.out_ch + oc)] +
			    bias[(size_t)oc];
	}
	ace_dump("ace_out", out.data(), out.size());
	return out;
}

// ── residency ─────────────────────────────────────────────────────────────
//
// The decision is the one `ImageDiT`/`AvDiT` make for their own stacks (see
// core/runtime/vram_window.hpp for the band and the arithmetic): keep as many
// blocks as fit beside everything else the step holds, stream the rest, and let
// the *accountant* rather than arithmetic on tensor sizes say how many. An arena
// books whole chunks and a block's eleven matrices do not tile them, so the loop
// below grows the window by each block's measured arena growth and stops when
// the next block would not fit.
//
// Three things are measured rather than named, because each of them is what a
// hand-written estimate gets wrong: what the process already holds (`local()`,
// read while the weight arena is empty so the streamed block is not counted
// twice), the activation frame (priced with the arena's own chunk rule, minus
// what that arena already holds from earlier phases - `reset()` keeps chunks, so
// that part is already in the ledger), and the charge one *streamed* block books
// (measured by actually streaming one, the way `ImageDiT` does).

namespace {

// The activations one `forward` allocates, in the order it allocates them - the
// arena is a bump allocator, so the order is part of the charge. One entry per
// `aalloc`; the fp16 attention scratch is booked at two bytes. The
// sliding-attention path allocates one small u32 key-bound mask per query tile
// of every even block, inside the loop; that is folded in as one entry.
std::vector<u64> ace_activation_sizes(const AceDitConfig& cfg, i64 T, i64 n_enc) {
	const i64 H = cfg.hidden;
	const i64 HQ = cfg.n_heads * cfg.head_dim;
	const i64 HK = cfg.n_kv_heads * cfg.head_dim;
	const i64 I = cfg.intermediate;
	const i64 patch = cfg.patch;
	const i64 Tp = ceil_div(T, patch) * patch;
	const i64 S = Tp / patch;
	const i64 tile = std::max<i64>(64, (i64)attn_query_tile());
	const i64 rows_att = S + tile;
	const i64 n_kv = std::max(n_enc, S);
	std::vector<u64> v;
	v.reserve(48);
	auto f32 = [&](i64 n) { v.push_back((u64)std::max<i64>(n, 0) * 4); };
	auto f16 = [&](i64 n) { v.push_back((u64)std::max<i64>(n, 0) * 2); };
	f32((i64)cfg.in_ch * Tp);          // inp
	f32(H * S);                        // cin
	f32(H * S);                        // hin
	f32(H * n_kv);                     // enc
	f32(H * n_kv);                     // ench
	f32(H * S);                        // t
	f32(6 * H);                        // mo
	f32(HQ * rows_att);                // qr
	f32(HQ * S);                       // qn
	f16(HQ * rows_att);                // qh
	f32(HK * S);                       // kr
	f32(HK * S);                       // kn
	f32(HK * S);                       // vr
	f16(HK * S);                       // kh
	f16(HK * S);                       // vh
	f32(HQ * rows_att);                // attn
	f32(H * S);                        // proj
	f32(HK * n_kv);                    // ckr
	f32(HK * n_kv);                    // ckn
	f16(HK * n_kv);                    // ckh
	f16(HK * n_kv);                    // cvh
	f32(I * S);                        // gate
	f32(I * S);                        // up
	f32(I * S);                        // hh
	f32(H * S);                        // down
	f32((i64)cfg.out_ch * patch * S);  // outy
	f32(cfg.t_dim);                    // te_in
	f32(cfg.t_dim);                    // te_in_r
	f32(H);                            // te1
	f32(H);                            // temb_t
	f32(H);                            // temb_r
	f32(6 * H);                        // proj_t
	f32(6 * H);                        // proj_r
	f32(std::max(S, tile));            // ub_self (u32 rows)
	f32(std::max(std::max(n_enc, S), tile));   // ub_cross
	v.push_back((u64)(cfg.n_layers / 2) * (u64)ceil_div(S, tile) * (u64)tile * 4);
	return v;
}

}  // namespace

i64 AceDiT::n_layers() const { return impl_ ? impl_->cfg.n_layers : 0; }

i64 AceDiT::resident_layers() const { return impl_ ? impl_->res_n : 0; }

u64 AceDiT::resident_bytes() const { return impl_ ? impl_->res_bytes : 0; }

void AceDiT::release_resident() {
	if (!impl_) return;
	Impl& m = *impl_;
	m.res_arena.release_chunks();
	m.res_layers.clear();
	m.res_n = 0;
	m.res_bytes = 0;
	m.planned = false;
}

void AceDiT::plan_residency(i64 T, i64 n_enc) {
	if (!impl_ || !impl_->g || !impl_->g->ok()) return;
	Impl& m = *impl_;
	GpuCtx& g = *m.g;
	const AceDitConfig& cfg = m.cfg;
	if (m.planned || T <= 0 || cfg.n_layers <= 0) return;
	const u64 budget = vram_budget().limit();
	if (budget == 0) return;
	const u64 chunk = arena_chunk_bytes_for(budget);
	m.res_arena.set_chunk_bytes(chunk);

	// ── what the step holds besides the window ──
	// Read with the weight arena empty so the streamed block is not counted here
	// *and* in the in-flight term below.
	if (g.wa) g.wa->release_chunks();
	const u64 resident_now = vram_budget().local();

	// The activation frame, priced the way the arena books it. What the arena
	// already holds (the condition builder's scratch, from the text phase) is
	// already charged to `resident_now` and `reset()` keeps those chunks, so only
	// the *extra* is claimed here.
	{
		const u64 ac = g.aa ? g.aa->chunk_bytes() : chunk;
		const u64 need = arena_chunk_cost(ace_activation_sizes(cfg, T, n_enc), ac);
		const u64 have = g.aa ? g.aa->capacity() : 0;
		m.act_extra = need > have ? need - have : 0;
	}

	// What streaming one block charges: the block is uploaded the way the loop
	// uploads it and the weight arena's capacity is read back. The header's
	// tensor sum is close but not exact - a block's eleven matrices land on
	// different chunk boundaries - and that difference is the whole margin when
	// the window is spending nearly the whole card.
	u64 inflight = window_charge_bytes(1, m.res_layer_bytes, g.wa ? g.wa->chunk_bytes() : chunk);
	{
		g.ctx->begin();
		try {
			AceDitLayer tmp;
			m.upload_layer(0, tmp, nullptr);
			g.ctx->submit();
			const u64 measured = g.wa ? g.wa->capacity() : 0;
			if (measured > inflight) inflight = measured;
		} catch (const std::exception& e) {
			// The probe is the *estimate* of what streaming one block costs; a card that
			// cannot even hold one block has no window to take, and the honest figure is
			// the header's own byte sum (which is in `inflight` already). Do not let a
			// failed probe abort the run - the loop's own refusal is the better error.
			g.ctx->submit_if_recording();
			if (getenv("PHI_ACE_PLAN"))
				fprintf(stderr, "[ace] plan: the streamed-block probe failed (%s)\n", e.what());
		}
		g.ring->rewind();
		if (g.wa) g.wa->release_chunks();
	}
	m.res_inflight = inflight;

	// The plan's own floor, then as much window as the card will still hand out.
	//
	// The bound the loop below tests is `window + est ≤ cap`, and `est` is one
	// streamed block's whole charge - the headroom the streamed block needs while
	// the window is up. So `cap` is what is left after *everything* else:
	// `resident_now + act + inflight` (that is `plan_reserve`) plus the one
	// streamed block `est` stands for plus the slack. This is `ImageDiT`'s own
	// convention (`cap = room`), deliberately: the tighter derivation (`cap =
	// room + inflight`, which does not price the streamed block twice) buys about
	// one more block and pays for it by leaving no room for an activation estimate
	// that came out short - a refusal in the middle of the sampling loop, which is
	// a lost generation rather than a slower one.
	const u64 kSlack = 96ull << 20;
	m.plan_reserve = resident_now + m.act_extra + inflight;
	u64 room = budget > m.plan_reserve + kSlack ? budget - m.plan_reserve - kSlack : 0;
	const u64 device_room = vram_budget().live_headroom();
	if (device_room < room) room = device_room;
	const u64 cap = room;

	// ── take the window ──
	// Uploaded here rather than at first use: the disk traffic is the same, but
	// the plan's invariant has to hold before the loop starts, and a block that
	// failed to upload mid-step would fail after the conditioning had already
	// been built.
	m.res_arena.release_chunks();
	m.res_layers.assign((size_t)cfg.n_layers, AceDitLayer{});
	i64 n = 0;
	u64 est = inflight;
	g.ctx->begin();
	if (const char* e = getenv("PHI_ACE_RES")) {
		const long vv = strtol(e, nullptr, 10);
		const i64 want = vv <= 0 ? 0 : std::min<i64>(cfg.n_layers, vv);
		for (; n < want; n++) {
			if (!g.ctx->recording()) g.ctx->begin();
			m.upload_layer(n, m.res_layers[(size_t)n], &m.res_arena);
		}
	} else {
		while (n < cfg.n_layers) {
			if (m.res_arena.capacity() + est > cap) break;
			const u64 have = m.res_arena.capacity();
			if (!g.ctx->recording()) g.ctx->begin();
			try {
				m.upload_layer(n, m.res_layers[(size_t)n], &m.res_arena);
			} catch (const std::exception& e) {
				if (getenv("PHI_ACE_PLAN"))
					fprintf(stderr, "[ace] plan: window stopped at %lld/%lld (%s)\n",
					        (long long)n, (long long)cfg.n_layers, e.what());
				break;
			}
			const u64 grew = m.res_arena.capacity() - have;
			if (grew > est) est = grew;
			n++;
		}
	}
	g.ctx->submit();
	g.ring->rewind();
	m.res_layers.resize((size_t)n);
	m.res_n = n;
	m.res_bytes = m.res_arena.capacity();
	m.planned = true;
	if (getenv("PHI_ACE_PLAN"))
		fprintf(stderr,
		        "[ace] plan: budget=%s held=%s act=%s stream=%s window=%lldx%s "
		        "(re-read %s/step)\n",
		        format_bytes(budget).c_str(), format_bytes(resident_now).c_str(),
		        format_bytes(m.act_extra).c_str(), format_bytes(inflight).c_str(), (long long)n,
		        format_bytes(m.res_bytes).c_str(),
		        format_bytes((u64)(cfg.n_layers - n) * m.res_layer_bytes).c_str());
}

// ── the condition builder ─────────────────────────────────────────────────
//
// `comfy/ldm/ace/ace_step15.py::AceStepConditionEncoder` + tokenizer +
// detokenizer, for the one case this engine drives: no reference audio (the
// pipeline's own silence latent stands in), so the packed sequence is
// [lyric, timbre(1 token), text] and `is_covers` follows the audio codes.
struct AceConditionBuilder::Impl {
	GpuCtx* g = nullptr;
	SafeTensors st_;
	i64 text_dim = 0, enc_dim = 0;
	// encoder
	AceLinear text_proj;                   // 1024 -> 2048, no bias
	AceLinear lyric_emb;                   // 1024 -> 2048
	GpuAlloc lyric_emb_b, lyric_norm_w;
	AceQwen3Config lyric_cfg;
	GpuAlloc timbre_emb_w, timbre_emb_b, timbre_norm_w, timbre_special;
	AceQwen3Config timbre_cfg;
	// tokenizer
	GpuAlloc acp_w, acp_b;                 // audio_acoustic_proj 64 -> 2048 (bf16)
	AceLinear pool_emb;
	GpuAlloc pool_emb_b, pool_special, pool_norm_w;
	AceQwen3Config pool_cfg;
	std::vector<float> qin_w, qin_b, qout_w, qout_b;   // FSQ project_in/out (host)
	i64 fsq_dim = 0;
	// detokenizer
	AceLinear det_emb, det_proj;
	GpuAlloc det_emb_b, det_special, det_norm_w;
	AceQwen3Config det_cfg;
	GpuAlloc det_proj_b;

	void load_tower(const std::string& prefix, AceQwen3Config* cfg, GpuAlloc* norm_w);
};

void AceConditionBuilder::Impl::load_tower(const std::string& prefix, AceQwen3Config* cfg,
                                           GpuAlloc* norm_w) {
	GpuCtx& gc = *g;
	ace_qwen3_probe(st_, prefix, cfg);
	cfg->rope_theta = 1e6f;   // the bundle's RotaryEmbedding default
	if (st_.find(prefix + "norm.weight")) *norm_w = ace_upload_vector(gc, st_, prefix + "norm.weight", gc.keep);
	// The `embed_tokens` projection of every one of these towers is loaded by the
	// caller (it is a plain [n, k] matrix, so it goes through `ace_upload_linear`).
}

AceConditionBuilder::AceConditionBuilder() = default;
AceConditionBuilder::~AceConditionBuilder() = default;

void AceConditionBuilder::open(const std::string& path, GpuCtx* gpu) {
	impl_ = std::make_unique<Impl>();
	Impl& m = *impl_;
	m.g = gpu;
	m.st_.open(path);
	GpuCtx& g = *gpu;
	const SafeTensors& st = m.st_;

	m.text_dim = st.require("encoder.text_projector.weight").shape[1];
	m.enc_dim = st.require("encoder.text_projector.weight").shape[0];
	if (m.enc_dim != 2048)
		throw MediaError("ace condition: the encoder hidden is " + std::to_string(m.enc_dim) +
		                 ", expected 2048");
	m.fsq_dim = st.require("tokenizer.quantizer.project_in.weight").shape[1];

	g.new_step();
	g.ctx->begin();
	auto vec = [&](const std::string& n) {
		return ace_upload_vector(g, st, n, g.keep);
	};
	// ── the encoder ──
	m.text_proj = ace_upload_linear(g, st, "encoder.text_projector", g.keep);
	if (m.text_proj.k != m.text_dim || m.text_proj.n != m.enc_dim)
		throw MediaError("ace condition: text_projector shape");
	m.load_tower("encoder.lyric_encoder.", &m.lyric_cfg, &m.lyric_norm_w);
	m.lyric_emb = ace_upload_linear(g, st, "encoder.lyric_encoder.embed_tokens", g.keep);
	m.lyric_emb_b = vec("encoder.lyric_encoder.embed_tokens.bias");
	m.load_tower("encoder.timbre_encoder.", &m.timbre_cfg, &m.timbre_norm_w);
	m.timbre_special = vec("encoder.timbre_encoder.special_token");
	// The timbre embedder takes a 64-wide input; the engine's GEMM reads it as a
	// plain bf16 matrix, which is what `ace_upload_linear` gives.
	{
		const StTensor& t = st.require("encoder.timbre_encoder.embed_tokens.weight");
		m.timbre_emb_w = g.kalloc(t.numel * dtype_size(t.dtype));
		g.upload_tensor_into(st, t, t.dtype, m.timbre_emb_w);
		m.timbre_emb_b = vec("encoder.timbre_encoder.embed_tokens.bias");
	}

	// ── the tokenizer ──
	{
		const StTensor& t = st.require("tokenizer.audio_acoustic_proj.weight");
		m.acp_w = g.kalloc(t.numel * dtype_size(t.dtype));
		g.upload_tensor_into(st, t, t.dtype, m.acp_w);
	}
	m.acp_b = vec("tokenizer.audio_acoustic_proj.bias");
	m.load_tower("tokenizer.attention_pooler.", &m.pool_cfg, &m.pool_norm_w);
	m.pool_emb = ace_upload_linear(g, st, "tokenizer.attention_pooler.embed_tokens", g.keep);
	m.pool_emb_b = vec("tokenizer.attention_pooler.embed_tokens.bias");
	m.pool_special = vec("tokenizer.attention_pooler.special_token");
	// The FSQ projections are 6-wide, which is host work by any measure: the
	// weights are 6x2048 and the values they act on are one row per pooled
	// frame.
	m.qin_w = tensor_to_f32(st, st.require("tokenizer.quantizer.project_in.weight"));
	m.qin_b = tensor_to_f32(st, st.require("tokenizer.quantizer.project_in.bias"));
	m.qout_w = tensor_to_f32(st, st.require("tokenizer.quantizer.project_out.weight"));
	m.qout_b = tensor_to_f32(st, st.require("tokenizer.quantizer.project_out.bias"));

	// ── the detokenizer ──
	m.load_tower("detokenizer.", &m.det_cfg, &m.det_norm_w);
	m.det_emb = ace_upload_linear(g, st, "detokenizer.embed_tokens", g.keep);
	m.det_emb_b = vec("detokenizer.embed_tokens.bias");
	m.det_special = vec("detokenizer.special_tokens");
	m.det_proj = ace_upload_linear(g, st, "detokenizer.proj_out", g.keep);
	m.det_proj_b = vec("detokenizer.proj_out.bias");
	g.ctx->submit();
	g.ring->rewind();

	if (m.lyric_cfg.hidden != m.enc_dim || m.timbre_cfg.hidden != m.enc_dim ||
	    m.pool_cfg.hidden != m.enc_dim || m.det_cfg.hidden != m.enc_dim)
		throw MediaError("ace condition: a tower's hidden is not the encoder width");
}

AceCondition AceConditionBuilder::build(const std::vector<float>& text_hidden, i64 n_text,
                                        const std::vector<float>& lyric_hidden, i64 n_lyric,
                                        const std::vector<i32>& codes, i64 T) {
	if (!impl_) throw MediaError("ace condition: not open");
	Impl& m = *impl_;
	GpuCtx& g = *m.g;
	const SafeTensors& st = m.st_;
	const i64 E = m.enc_dim;
	if (T <= 0) throw MediaError("ace condition: T must be positive");
	// The conditioning's own inputs, for the reference comparison (see
	// `ace_dump_prefix`): the two prompt towers' hidden states and the audio-code
	// ids the LM sampled.
	ace_dump("ace_cond_text", text_hidden.data(), text_hidden.size());
	ace_dump("ace_cond_lyric", lyric_hidden.data(), lyric_hidden.size());
	{
		const char* pre = ace_dump_prefix();
		if (pre) {
			FILE* fp = fopen((std::string(pre) + "ace_cond_codes.i32").c_str(), "wb");
			if (fp) {
				fwrite(codes.data(), 4, codes.size(), fp);
				fclose(fp);
			}
		}
	}
	if (n_text < 0 || (i64)text_hidden.size() != n_text * m.text_dim)
		throw MediaError("ace condition: text_hidden is not [n_text, text_dim]");
	if (n_lyric < 0 || (i64)lyric_hidden.size() != n_lyric * m.text_dim)
		throw MediaError("ace condition: lyric_hidden is not [n_lyric, text_dim]");

	// The pipeline's silence latent: the reference audio when the caller has
	// none, so it is what the timbre encoder and the no-codes tokenizer see.
	std::vector<float> sil;   // [64, T] channel-major
	silence_latent(T, sil);
	const i64 P = ceil_div(T, 5);            // pool windows of 5 latent frames
	const i64 tp5 = P * 5;
	const i64 rows_max = std::max<i64>(std::max(std::max(n_lyric, n_text), tp5 + 5), P * 6 + 1);

	g.new_step();
	auto A = [&](i64 n) { return g.aalloc((u64)n * 4); };
	GpuAlloc h1 = A(rows_max * E);
	GpuAlloc h2 = A(rows_max * E);
	// The staging buffer every raw input goes through: the text/lyric prompts are
	// [n, 1024] and the latent-side inputs are [n, 64], so it is sized for the
	// wider of the two.
	GpuAlloc hx = A(std::max<i64>(rows_max * 64, std::max<i64>(n_lyric, n_text) * m.text_dim));
	GpuAlloc det_out = A(rows_max * 64);
	AceStackScratch sc;
	sc.alloc(g, rows_max, m.lyric_cfg);

	auto ub_all = [&](i64 rows, i64 bound) {
		std::vector<u32> v((size_t)std::max<i64>(rows, 64), (u32)bound);
		GpuAlloc a = ace_upload_u32(g, v.data(), (i64)v.size());
		return a;
	};

	// ── 1. the lyric encoder: [n_lyric, 1024] -> 8 blocks -> norm ──
	std::vector<float> lyric_out;
	if (n_lyric > 0) {
		upload_f32_into(g, hx, lyric_hidden.data(), n_lyric * m.text_dim);
		// embed_tokens is a 1024 -> 2048 linear *with* a bias.
		ace_linear(g, m.lyric_emb, hx, n_lyric, h1, m.lyric_emb_b, &sc.qs);
		GpuAlloc u = ub_all(n_lyric, n_lyric);
		ace_qwen3_forward(g, st, "encoder.lyric_encoder.", m.lyric_cfg, m.lyric_cfg.n_layers, true,
		                  "encoder.lyric_encoder.norm.weight", h1, n_lyric, sc, nullptr, 0,
		                  n_lyric, u);
		lyric_out = g.download_f32(h1, n_lyric * E);
	}

	// ── 2. the timbre encoder: the whole silence sequence, first row out ──
	std::vector<float> timbre_out((size_t)E, 0.0f);
	{
		// [64, T] -> [T, 64] (the reference's `movedim(-1, -2)`).
		std::vector<float> tk((size_t)T * 64);
		for (i64 tt = 0; tt < T; tt++)
			for (i64 c = 0; c < 64; c++) tk[(size_t)tt * 64 + (size_t)c] = sil[(size_t)c * T + (size_t)tt];
		upload_f32_into(g, hx, tk.data(), T * 64);
		bf16_linear(g, hx, m.timbre_emb_w, m.timbre_emb_b, h1, T, E, 64, true);
		GpuAlloc u = ub_all(T, T);
		ace_qwen3_forward(g, st, "encoder.timbre_encoder.", m.timbre_cfg, m.timbre_cfg.n_layers,
		                  true, "encoder.timbre_encoder.norm.weight", h1, T, sc, nullptr, 0, T, u);
		std::vector<float> full = g.download_f32(h1, T * E);
		// `unpack_timbre_embeddings` with one batch item is the identity, and the
		// row it unpacks is the *first* (`hidden_states[:, 0, :]`).
		std::memcpy(timbre_out.data(), full.data(), (size_t)E * 4);
	}

	// ── 3. the text projector ──
	std::vector<float> text_out;
	if (n_text > 0) {
		upload_f32_into(g, hx, text_hidden.data(), n_text * m.text_dim);
		ace_linear(g, m.text_proj, hx, n_text, h1, GpuAlloc{}, &sc.qs);
		text_out = g.download_f32(h1, n_text * E);
	}

	// ── 4. the audio codes: the FSQ hints at 5 Hz ──
	// `get_output_from_indices` (the codes path) and `tokenize` (the silence
	// path) both end in `project_out(codes)`, so this is the one place the 6-vector
	// is formed.
	std::vector<float> hints5;   // [P, 2048]
	{
		std::vector<float> cvec((size_t)P * 6, 0.0f);
		if (!codes.empty()) {
			std::vector<i32> cs = codes;
			if ((i64)cs.size() < P) cs.resize((size_t)P, kFsqPadCode);   // the reference's pad
			if ((i64)cs.size() > P) cs.resize((size_t)P);
			for (i64 p = 0; p < P; p++) fsq_index_to_vector(cs[(size_t)p], &cvec[(size_t)p * 6]);
		} else {
			// `tokenizer.tokenize(silence)`: audio_acoustic_proj -> the attention
			// pooler -> FSQ.
			std::vector<float> x((size_t)tp5 * 64, 0.0f);
			for (i64 tt = 0; tt < std::min<i64>(T, tp5); tt++)
				for (i64 c = 0; c < 64; c++)
					x[(size_t)tt * 64 + (size_t)c] = sil[(size_t)c * T + (size_t)tt];
			upload_f32_into(g, hx, x.data(), tp5 * 64);
			bf16_linear(g, hx, m.acp_w, m.acp_b, h1, tp5, E, 64, true);
			std::vector<float> acc = g.download_f32(h1, tp5 * E);
			// The pooler's input is `cat([special, x], dim=2)` over 5-frame
			// windows, i.e. each window is a 6-row group whose row 0 is the
			// special token; the tower runs per group (see the stack runner's
			// `group`).
			std::vector<float> pool_in((size_t)P * 6 * E, 0.0f);
			std::vector<float> spec((size_t)E);
			g.ctx->download(m.pool_special.res, m.pool_special.off, spec.data(), (size_t)E * 4);
			for (i64 p = 0; p < P; p++) {
				std::memcpy(&pool_in[(size_t)p * 6 * E], spec.data(), (size_t)E * 4);
				for (i64 j = 0; j < 5; j++)
					std::memcpy(&pool_in[((size_t)p * 6 + 1 + (size_t)j) * E],
					            &acc[((size_t)p * 5 + (size_t)j) * E], (size_t)E * 4);
			}
			upload_f32_into(g, h1, pool_in.data(), P * 6 * E);
			ace_linear(g, m.pool_emb, h1, P * 6, h2, m.pool_emb_b, &sc.qs);
			GpuAlloc u = ub_all(6, 6);
			ace_qwen3_forward(g, st, "tokenizer.attention_pooler.", m.pool_cfg,
			                  m.pool_cfg.n_layers, true, "tokenizer.attention_pooler.norm.weight", h2,
			                  P * 6, sc, nullptr, 0, 0, u, 6);
			std::vector<float> pooled = g.download_f32(h2, P * 6 * E);
			// Row 0 of each group (`x[:, 0, :]`).
			std::vector<float> rows((size_t)P * E);
			for (i64 p = 0; p < P; p++)
				std::memcpy(&rows[(size_t)p * E], &pooled[(size_t)p * 6 * E], (size_t)E * 4);
			// project_in -> 6 numbers, then `FSQ.bound`.
			for (i64 p = 0; p < P; p++) {
				std::vector<float> proj((size_t)6, 0.0f);
				for (i64 o = 0; o < 6; o++) {
					double accd = m.qin_b[(size_t)o];
					for (i64 k = 0; k < m.fsq_dim; k++)
						accd += (double)m.qin_w[(size_t)o * m.fsq_dim + (size_t)k] *
						        (double)rows[(size_t)p * E + (size_t)k];
					proj[(size_t)o] = (float)accd;
				}
				fsq_bound_vector(proj.data(), &cvec[(size_t)p * 6]);
			}
		}
		// `project_out` (host: the weight is 6 x 2048).
		hints5.assign((size_t)P * E, 0.0f);
		for (i64 p = 0; p < P; p++)
			for (i64 o = 0; o < E; o++) {
				double accd = m.qout_b[(size_t)o];
				for (i64 k = 0; k < 6; k++)
					accd += (double)m.qout_w[(size_t)o * 6 + (size_t)k] *
					        (double)cvec[(size_t)p * 6 + (size_t)k];
				hints5[(size_t)p * E + (size_t)o] = (float)accd;
			}
	}

	// ── 5. the detokenizer: [P, 2048] -> [tp5, 64] ──
	std::vector<float> src_latents((size_t)64 * (size_t)T, 0.0f);   // [64, T] channel-major
	{
		upload_f32_into(g, h1, hints5.data(), P * E);
		ace_linear(g, m.det_emb, h1, P, h2, m.det_emb_b, &sc.qs);
		std::vector<float> emb = g.download_f32(h2, P * E);
		std::vector<float> spec((size_t)5 * E);
		g.ctx->download(m.det_special.res, m.det_special.off, spec.data(), (size_t)5 * E * 4);
		// `x.unsqueeze(2).repeat(1,1,pw,1) + special_tokens` -> [P*pw, E].
		std::vector<float> din((size_t)tp5 * E, 0.0f);
		for (i64 p = 0; p < P; p++)
			for (i64 j = 0; j < 5; j++)
				for (i64 o = 0; o < E; o++)
					din[((size_t)p * 5 + (size_t)j) * E + (size_t)o] =
					    emb[(size_t)p * E + (size_t)o] + spec[(size_t)j * E + (size_t)o];
		upload_f32_into(g, h1, din.data(), tp5 * E);
		GpuAlloc u = ub_all(5, 5);
		ace_qwen3_forward(g, st, "detokenizer.", m.det_cfg, m.det_cfg.n_layers, true,
		                  "detokenizer.norm.weight", h1, tp5, sc, nullptr, 0, 0, u, 5);
		ace_linear(g, m.det_proj, h1, tp5, det_out, m.det_proj_b, &sc.qs);
		std::vector<float> hints = g.download_f32(det_out, tp5 * 64);   // [tp5, 64]
		// `lm_hints[:, :src_latents.shape[1], :]`, transposed into the latent's
		// own channel-major layout.
		for (i64 tt = 0; tt < T; tt++)
			for (i64 c = 0; c < 64; c++)
				src_latents[(size_t)c * T + (size_t)tt] = hints[(size_t)tt * 64 + (size_t)c];
	}

	// ── 6. assemble ──
	// `prepare_condition`: with no reference audio the pipeline passes
	// `is_covers = audio_codes is not None`, so the source latents are the
	// detokenised hints when codes were given and the silence latent otherwise.
	AceCondition out;
	{
		std::vector<float> eh;
		eh.reserve((size_t)(n_lyric + 1 + n_text) * E);
		eh.insert(eh.end(), lyric_out.begin(), lyric_out.end());
		eh.insert(eh.end(), timbre_out.begin(), timbre_out.end());
		eh.insert(eh.end(), text_out.begin(), text_out.end());
		out.n = (i64)eh.size() / E;
		out.encoder_hidden = std::move(eh);
	}
	if (codes.empty()) {
		// `is_covers = False`: src_latents = refer_audio (the silence latent).
		out.context.assign((size_t)T * (size_t)(2 * 64), 0.0f);
		for (i64 tt = 0; tt < T; tt++)
			for (i64 c = 0; c < 64; c++) {
				out.context[(size_t)tt * 128 + (size_t)c] = sil[(size_t)c * T + (size_t)tt];
				out.context[(size_t)tt * 128 + 64 + (size_t)c] = 1.0f;   // chunk_masks
			}
	} else {
		out.context.assign((size_t)T * (size_t)(2 * 64), 0.0f);
		for (i64 tt = 0; tt < T; tt++)
			for (i64 c = 0; c < 64; c++) {
				out.context[(size_t)tt * 128 + (size_t)c] = src_latents[(size_t)c * T + (size_t)tt];
				out.context[(size_t)tt * 128 + 64 + (size_t)c] = 1.0f;
			}
	}
	out.T = T;
	return out;
}

}  // namespace phi::media
