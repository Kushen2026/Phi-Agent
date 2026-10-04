// Breeze-TTS-2 text-to-speech (the VoiceDesign path).
//
// Three networks, in the reference's own order (native.py / runtime.py):
//
//   text encoder   T5Gemma2 encoder, 26 layers x 1152 wide, bidirectional with a
//                  512-token sliding window on all but every sixth layer, Q/K
//                  RMSNorm over head_dim 256, GeGLU(gelu_pytorch_tanh), and the
//                  checkpoint's `embed_tokens.eoi_embedding` substituted at the
//                  EOI position. Host fp32 (see below).
//   backbone       Qwen3-1B (28 layers x 2048, 16 heads of 128, GQA 8 K/V heads,
//                  theta 1e6) consuming the projected text rows.
//   depth decoder  a 12-layer llama (1024 wide, 8 heads of 128, 2 K/V heads,
//                  mlp 8192) that turns one backbone token into the other 15
//                  codebooks of the same frame, plus the per-codebook
//                  `codebooks_head` [15, 1024, 2051].
//
// then the codec (core/models/mimi_codec.*) turns the frames into PCM.
//
// ── what runs where ────────────────────────────────────────────────────────
// The text encoder is a single prefill pass over a short prompt, so it runs on
// the host in fp32, reading the checkpoint a slab of rows at a time through
// `SafeTensors::dequant_rows` (the whole encoder is 2.8 GB as fp32 and never
// needs to be resident). Its sliding-window band mask is not expressible with any
// kernel the engine has - `attn_flash`'s per-row bound is an *upper* bound only,
// and the window needs both - and the host pass also keeps its (1 + w) norms and
// its two RoPE tables exact.
//
// The backbone and the depth decoder run on the GPU through the engine's int8
// tensorwise + convrot GEMM (`load_quant_linear` normalises the bf16 checkpoint
// onto it) and the fp16 tiled flash attention. Their weights are loaded *once*
// into the `keep` arena (~1.8 GB int8 for both) rather than streamed per layer:
// a TTS utterance is up to 2048 sequential single-token steps, so re-uploading a
// layer per step would cost more than the arithmetic it feeds.
//
// ── the sampling loop ─────────────────────────────────────────────────────
// runtime.py's `generate_codes`, faithfully: prefill the merged prompt, sample
// the first backbone token, then per frame (a) run the depth decoder to get the
// 15 remaining codebooks, (b) feed the frame's summed audio embedding back into
// the backbone, (c) sample the next token under the repetition penalty. CFG is
// the reference's `uncond + scale * (cond - uncond)` over the two prompt
// branches, and the reserved ids [2048, 2051) are suppressed on both sampling
// sites. The two branches are kept as two *unpadded* sequences: the reference
// left-pads the shorter (negative) branch to the cond branch's length, masks the
// pads out of every attention, and gives the continuation the RoPE positions of
// its own real tokens - which is exactly what a separate compact sequence
// computes, bit for bit.
//
// Numeric notes: the backbone and the depth decoder run in fp32 (int8 weights,
// fp32 activations and accumulators). The reference runs them in bf16 on a GPU
// that supports it; fp32 is the more accurate of the two and the *sampling
// order* - the part that decides which tokens come out - is reproduced exactly.
// The one-row decode steps cannot use the flash kernel's `causal` flag (its
// cut-off is the query *row index*, which is 0 for a single-row dispatch), so
// they pass `causal=false` with the engine's per-row key bound `ub` =
// "keys [0, row0+1)": the single query row is the newest position, so attending
// to every cached key is precisely its causal mask.
#include "models/breeze_tts.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "host/quant.hpp"
#include "kernels/gpu_ops.hpp"
#include "kernels/parallel_for.hpp"
#include "models/video_vae_internal.hpp"   // dispatch_f32_to_f16
#include "util/base.hpp"
#include "util/json.hpp"

namespace phi::media {

using kernels::parallel_for;

// The sampler's randomness. The reference draws from torch's global RNG after
// `fix_seed`; there is no torch here, so this is its own generator. What matters
// for reproducing the reference is the *distribution* (the same temperature /
// top_k / top_p / penalty pipeline and the same categorical draw), not the bit
// stream of a generator this engine does not share.
float BreezeTts::Rng::next01() {
	u64 z = (s += 0x9E3779B97F4A7C15ull);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	z ^= z >> 31;
	return (float)((z >> 40) * (1.0 / 16777216.0));
}

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// ── host kernels ───────────────────────────────────────────────────────────

// T5Gemma2's RMSNorm: fp32, then `output * (1 + w)` - the Gemma convention, and
// what the checkpoint's near-zero norm weights are the zero-init of.
void te_rms_rows(const float* x, i64 rows, i64 cols, const float* w, float eps, float* y) {
	parallel_for(rows, [&](i64 r) {
		const float* xr = x + r * cols;
		float* yr = y + r * cols;
		float sq = 0.0f;
		for (i64 i = 0; i < cols; i++) sq += xr[i] * xr[i];
		const float inv = 1.0f / std::sqrt(sq / (float)cols + eps);
		for (i64 i = 0; i < cols; i++) yr[i] = xr[i] * inv * (1.0f + w[i]);
	});
}

float gelu_tanh1(float x) {
	const float k = 0.79788456080286535588f;   // sqrt(2/pi)
	return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

std::vector<float> default_inv_freq(float theta, i64 dim) {
	std::vector<float> inv((size_t)(dim / 2));
	for (i64 i = 0; i < dim / 2; i++)
		inv[(size_t)i] = 1.0f / std::pow(theta, (float)(2 * i) / (float)dim);
	return inv;
}

// HF's `_compute_llama3_parameters`: a wavelength-dependent rescale of the
// default inverse frequencies. The depth decoder's rope_scaling is
// {"rope_type": "llama3", factor 32, original_max_position_embeddings 16,
//  low_freq_factor 0.001953125, high_freq_factor 0.0078125}.
std::vector<float> llama3_inv_freq(float theta, i64 dim, float factor, float low_factor,
                                   float high_factor, float original_max) {
	std::vector<float> inv((size_t)(dim / 2));
	const float low_wl = original_max / low_factor;
	const float high_wl = original_max / high_factor;
	const float pi2 = 6.28318530717958648f;
	for (i64 i = 0; i < dim / 2; i++) {
		const float f = 1.0f / std::pow(theta, (float)(2 * i) / (float)dim);
		const float wl = pi2 / f;
		if (wl < high_wl) {
			inv[(size_t)i] = f;
		} else if (wl > low_wl) {
			inv[(size_t)i] = f / factor;
		} else {
			const float smooth = (original_max / wl - low_factor) / (high_factor - low_factor);
			inv[(size_t)i] = (1.0f - smooth) * f / factor + smooth * f;
		}
	}
	return inv;
}

// A [rows, heads, head_dim] half-split RoPE with `pos0` as the first row's position.
void rope_rows(float* x, i64 rows, i64 heads, i64 head_dim, const std::vector<float>& inv,
               i64 pos0) {
	const i64 half = head_dim / 2;
	parallel_for(rows, [&](i64 r) {
		const i64 pos = pos0 + r;
		for (i64 h = 0; h < heads; h++) {
			float* p = x + (r * heads + h) * head_dim;
			for (i64 i = 0; i < half; i++) {
				const float ang = (float)pos * inv[(size_t)i];
				const float c = std::cos(ang), s = std::sin(ang);
				const float a = p[i], b = p[i + half];
				p[i] = a * c - b * s;
				p[i + half] = b * c + a * s;
			}
		}
	});
}

// ── sampling (runtime.py's sample_logits) ──────────────────────────────────

// The reference's order, exactly: repetition penalty -> suppress tokens ->
// (argmax when do_sample is off) -> temperature -> top_k -> top_p -> sample.
i32 sample_logits(std::vector<float>& logits, float temperature, i64 top_k, float top_p,
                  float repetition_penalty, const std::vector<i32>* history, i64 suppress_from,
                  i64 suppress_to, BreezeTts::Rng& rng) {
	const i64 n = (i64)logits.size();
	if (history && repetition_penalty != 1.0f && !history->empty()) {
		std::vector<i32> uniq = *history;
		std::sort(uniq.begin(), uniq.end());
		uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
		for (i32 t : uniq) {
			if (t < 0 || t >= n) continue;
			const float v = logits[(size_t)t];
			logits[(size_t)t] = v > 0 ? v / repetition_penalty : v * repetition_penalty;
		}
	}
	for (i64 i = std::max<i64>(0, suppress_from); i < std::min(suppress_to, n); i++)
		logits[(size_t)i] = kNegInf;

	i64 best = 0;
	for (i64 i = 1; i < n; i++)
		if (logits[(size_t)i] > logits[(size_t)best]) best = i;
	if (!(temperature > 0.0f)) return (i32)best;   // do_sample with temperature 0 = greedy

	for (i64 i = 0; i < n; i++) logits[(size_t)i] /= temperature;

	if (top_k > 0 && top_k < n) {
		std::vector<float> sorted(logits);
		std::sort(sorted.begin(), sorted.end(), std::greater<float>());
		const float thr = sorted[(size_t)(top_k - 1)];
		for (i64 i = 0; i < n; i++)
			if (logits[(size_t)i] < thr) logits[(size_t)i] = kNegInf;
	}

	float mx = kNegInf;
	for (i64 i = 0; i < n; i++) mx = std::max(mx, logits[(size_t)i]);
	if (mx == kNegInf) return (i32)best;
	std::vector<float> probs((size_t)n, 0.0f);
	double sum = 0.0;
	for (i64 i = 0; i < n; i++) {
		const float p = std::exp(logits[(size_t)i] - mx);
		probs[(size_t)i] = p;
		sum += p;
	}
	if (!(sum > 0.0)) return (i32)best;

	if (top_p < 1.0f) {
		std::vector<i32> order((size_t)n);
		for (i64 i = 0; i < n; i++) order[(size_t)i] = (i32)i;
		std::sort(order.begin(), order.end(),
		          [&](i32 a, i32 b) { return probs[(size_t)a] > probs[(size_t)b]; });
		double cum = 0.0;
		bool cut = false;
		for (i64 i = 0; i < n; i++) {
			const i32 id = order[(size_t)i];
			if (cut) {
				sum -= (double)probs[(size_t)id];
				probs[(size_t)id] = 0.0f;
				continue;
			}
			cum += (double)probs[(size_t)id];
			if (cum > (double)top_p) cut = true;   // this one is kept, the rest are dropped
		}
		if (!(sum > 0.0)) return (i32)order[0];
	}

	const double target = (double)rng.next01() * sum;
	double acc = 0.0;
	for (i64 i = 0; i < n; i++) {
		acc += (double)probs[(size_t)i];
		if (acc >= target) return (i32)i;
	}
	for (i64 i = n - 1; i >= 0; i--)
		if (probs[(size_t)i] > 0.0f) return (i32)i;
	return (i32)best;
}

// ── device plumbing ────────────────────────────────────────────────────────

// One projection, through whichever path the weight's own precision implies:
// int8 (verbatim tensorwise / a packed family) rotates + quantises the [m, K]
// fp32 activation into `q8`/`s8` and runs the integer product; a float source
// (f16 / bf16 / f32) runs the dense fp16 GEMM with no conversion.
void qgemm(GpuCtx& g, const QuantLinear& w, const GpuAlloc& x, i64 m, const GpuAlloc& q8,
           const GpuAlloc& s8, const GpuAlloc& out) {
	if (w.n <= 0 || w.k <= 0) throw MediaError("breeze: a resident weight is missing");
	linear_gemm(g, w, x, m, w.k, out, q8, s8);
}

void gpu_rms(GpuCtx& g, const GpuAlloc& x, i64 rows, i64 cols, const GpuAlloc& w,
             const GpuAlloc& y, float eps) {
	NormArgs na;
	na.x = x;
	na.w = w;
	na.y = y;
	na.rows = rows;
	na.cols = cols;
	na.eps = eps;
	na.affine = true;
	dispatch_norm(*g.ctx, na);
}

void gpu_add(GpuCtx& g, const GpuAlloc& a, const GpuAlloc& b, i64 rows, i64 cols) {
	ElemArgs ea;
	ea.op = ElemOp::Add;
	ea.a = a;
	ea.b = b;
	ea.y = a;
	ea.rows = rows;
	ea.cols = cols;
	dispatch_elem(*g.ctx, ea);
}

void gpu_silu_gate(GpuCtx& g, const GpuAlloc& gate, const GpuAlloc& up, const GpuAlloc& out,
                   i64 rows, i64 cols) {
	ElemArgs ea;
	ea.op = ElemOp::SiluGate;
	ea.a = gate;
	ea.b = up;
	ea.y = out;
	ea.rows = rows;
	ea.cols = cols;
	dispatch_elem(*g.ctx, ea);
}

}  // namespace

// ── the text encoder's streamed matrices ───────────────────────────────────
//
// `y[m, n] = x[m, k] @ W^T` with W streamed out of the checkpoint a slab of
// output rows at a time: the encoder is 26 layers x ~27 M parameters, 2.8 GB as
// fp32, and materialising it would be the largest allocation in the process. The
// slab is decoded into the task's own buffer (256 rows x k floats = 1.2 MB, which
// stays in L2) and the tasks split the *output* columns, so every output element
// keeps the plain serial reduction order over k.
void BreezeTts::TeMat::gemm(const float* x, i64 m, float* y) const {
	if (!ok()) throw MediaError("breeze: a text encoder matrix is missing");
	const SafeTensors& SEN = *st;
	const StTensor& TEN = *t;
	const i64 slab = std::min<i64>(256, n);
	const i64 blocks = (n + slab - 1) / slab;
	parallel_for(blocks, [&](i64 blk) {
		const i64 r0 = blk * slab;
		const i64 rows = std::min(slab, n - r0);
		std::vector<float> w((size_t)(rows * k));
		SEN.dequant_rows(TEN, r0, rows, w.data());
		for (i64 i = 0; i < m; i++) {
			const float* xi = x + i * k;
			float* yi = y + i * n + r0;
			for (i64 r = 0; r < rows; r++) {
				const float* wr = w.data() + r * k;
				float acc = 0.0f;
				for (i64 j = 0; j < k; j++) acc += wr[j] * xi[j];
				yi[r] = acc;
			}
		}
	});
}

// ── open ───────────────────────────────────────────────────────────────────

namespace {

// Reads the checkpoint's config.json - the scalars a shape cannot carry:
// epsilons, RoPE thetas, the sliding window, the RoPE scaling and the special
// token ids. `override_path` (a settings selection) wins; otherwise the file is
// read beside the weights. Returns Null when there is no file.
JsonValue read_config(const std::string& model_path, const std::string& override_path) {
	const size_t slash = model_path.find_last_of("/\\");
	const std::string cfg =
	    !override_path.empty()
	        ? override_path
	        : (slash == std::string::npos ? std::string() : model_path.substr(0, slash + 1)) +
	              "config.json";
	std::vector<u8> raw;
	if (!read_file_bytes(cfg, raw)) return JsonValue();
	std::string err;
	auto j = json_parse(std::string((const char*)raw.data(), raw.size()), &err);
	return j ? *j : JsonValue();
}

i64 cfg_i(const JsonValue& o, const char* key, i64 def) {
	const JsonValue* v = o.find(key);
	return v ? v->as_int(def) : def;
}
float cfg_f(const JsonValue& o, const char* key, float def) {
	const JsonValue* v = o.find(key);
	return v ? (float)v->as_number(def) : def;
}

}  // namespace

void BreezeTts::check(const char* what, bool ok) const {
	if (!ok) throw MediaError(std::string("breeze: ") + what);
}

void BreezeTts::open(const std::string& model_path, const std::string& codec_path,
                     const std::string& tokenizer_json, const std::string& config_json,
                     const std::string& codec_config_json, GpuCtx* gpu) {
	if (!gpu || !gpu->ok()) throw MediaError("breeze: no GPU context");
	g_ = gpu;
	const size_t slash = model_path.find_last_of("/\\");
	dir_ = slash == std::string::npos ? std::string() : model_path.substr(0, slash + 1);
	st_.open(model_path);
	tok_.load(tokenizer_json);
	codec_.open(codec_path, gpu, codec_config_json);

	const JsonValue cfg = read_config(model_path, config_json);

	// ── shapes off the checkpoint, scalars off config.json ──
	const StTensor& tq = st_.require("backbone_model.layers.0.self_attn.q_proj.weight");
	const StTensor& tk = st_.require("backbone_model.layers.0.self_attn.k_proj.weight");
	const StTensor& tg = st_.require("backbone_model.layers.0.mlp.gate_proj.weight");
	bb_hidden_ = tq.shape[1];
	bb_head_dim_ = st_.require("backbone_model.layers.0.self_attn.q_norm.weight").shape[0];
	bb_heads_ = tq.shape[0] / bb_head_dim_;
	bb_kv_ = tk.shape[0] / bb_head_dim_;
	bb_inter_ = tg.shape[0];
	i64 l = 0;
	while (st_.find("backbone_model.layers." + std::to_string(l) + ".input_layernorm.weight")) l++;
	bb_layers_ = l;
	check("the backbone has no layers", bb_layers_ > 0);

	const StTensor& dq = st_.require("depth_decoder.model.layers.0.self_attn.q_proj.weight");
	const StTensor& dk = st_.require("depth_decoder.model.layers.0.self_attn.k_proj.weight");
	const StTensor& dg = st_.require("depth_decoder.model.layers.0.mlp.gate_proj.weight");
	dp_hidden_ = dq.shape[1];
	dp_inter_ = dg.shape[0];
	l = 0;
	while (st_.find("depth_decoder.model.layers." + std::to_string(l) + ".input_layernorm.weight")) l++;
	dp_layers_ = l;
	check("the depth decoder has no layers", dp_layers_ > 0);
	const StTensor& dp_emb = st_.require("depth_decoder.model.embed_tokens.weight");
	dp_embed_t_ = &dp_emb;
	dp_embed_rows_ = dp_emb.shape[0];
	dp_embed_dim_ = dp_emb.shape[1];
	const StTensor& dp_proj = st_.require("depth_decoder.model.inputs_embeds_projector.weight");
	check("the depth projector is not hidden -> hidden",
	      dp_proj.shape[0] == dp_hidden_ && dp_proj.shape[1] == dp_embed_dim_);

	const StTensor& te_q = st_.require("text_encoder.layers.0.self_attn.q_proj.weight");
	const StTensor& te_k = st_.require("text_encoder.layers.0.self_attn.k_proj.weight");
	te_hidden_ = te_q.shape[1];
	te_head_dim_ = st_.require("text_encoder.layers.0.self_attn.q_norm.weight").shape[0];
	te_heads_ = te_q.shape[0] / te_head_dim_;
	te_kv_ = te_k.shape[0] / te_head_dim_;
	te_inter_ = st_.require("text_encoder.layers.0.mlp.gate_proj.weight").shape[0];
	l = 0;
	while (st_.find("text_encoder.layers." + std::to_string(l) + ".pre_self_attn_layernorm.weight")) l++;
	te_layers_ = l;
	check("the text encoder has no layers", te_layers_ > 0);
	const StTensor& te_emb = st_.require("text_encoder.embed_tokens.weight");
	te_vocab_ = te_emb.shape[0];
	check("the text encoder's embedding width does not match hidden_size",
	      te_emb.shape[1] == te_hidden_);
	te_eoi_embed_ = st_.data_f32(st_.require("text_encoder.embed_tokens.eoi_embedding"));
	te_embed_t_ = st_.data_f32(te_emb);

	const StTensor& lm = st_.require("lm_head.weight");
	check("lm_head's input width is not the backbone's hidden size", lm.shape[1] == bb_hidden_);
	lm_head_n_ = lm.shape[0];

	// The special ids: the checkpoint's config.json carries them, and `vocab_size`
	// + 1 (the backbone's own EOS) is the last row of lm_head.
	audio_vocab_ = cfg_i(cfg, "audio_vocab_size", lm_head_n_ - 1);
	num_codebooks_ = cfg_i(cfg, "num_codebooks", dp_embed_rows_ / std::max<i64>(1, audio_vocab_));
	codebook_pad_ = cfg_i(cfg, "codebook_pad_token_id", -1);
	codebook_eos_ = cfg_i(cfg, "codebook_eos_token_id", 0);
	backbone_eos_ = audio_vocab_;   // config.vocab_size: the token that stops the loop
	{
		const JsonValue* codec = cfg.find("codec_config");
		codec_codebook_size_ = codec ? cfg_i(*codec, "codebook_size", 2048) : 2048;
	}
	check("the embedding table is not num_codebooks * audio_vocab",
	      dp_embed_rows_ == num_codebooks_ * audio_vocab_);
	check("lm_head's width is not audio_vocab + 1", lm_head_n_ == audio_vocab_ + 1);
	check("the codec and the backbone disagree on the codebook size",
	      codec_codebook_size_ <= audio_vocab_);

	// Backbone: Qwen3 defaults, overridden by config.json's backbone_config.
	{
		const JsonValue* bc = cfg.find("backbone_config");
		bb_eps_ = bc ? cfg_f(*bc, "rms_norm_eps", 1e-6f) : 1e-6f;
		bb_theta_ = bc ? cfg_f(*bc, "rope_theta", 1e6f) : 1e6f;
		if (bc) {
			check("backbone_config disagrees with the checkpoint's hidden size",
			      cfg_i(*bc, "hidden_size", bb_hidden_) == bb_hidden_);
			check("backbone_config disagrees with the checkpoint's head count",
			      cfg_i(*bc, "num_attention_heads", bb_heads_) == bb_heads_);
			check("backbone_config disagrees with the checkpoint's layer count",
			      cfg_i(*bc, "num_hidden_layers", bb_layers_) == bb_layers_);
		}
	}
	// Depth decoder: its head_dim cannot be read off a shape (8 heads of 128 and
	// 16 of 64 look the same), so it comes from the config.
	{
		const JsonValue* dc = cfg.find("depth_decoder_config");
		dp_head_dim_ = dc ? cfg_i(*dc, "head_dim", 128) : 128;
		dp_heads_ = dq.shape[0] / dp_head_dim_;
		dp_kv_ = dk.shape[0] / dp_head_dim_;
		check("depth_decoder_config disagrees with the checkpoint's q width",
		      dc == nullptr || cfg_i(*dc, "num_attention_heads", dp_heads_) == dp_heads_);
		dp_vocab_ = dc ? cfg_i(*dc, "vocab_size", audio_vocab_) : audio_vocab_;
		dp_eps_ = dc ? cfg_f(*dc, "rms_norm_eps", 1e-5f) : 1e-5f;
		check("the depth decoder's vocab does not match audio_vocab_size", dp_vocab_ == audio_vocab_);
		const float theta = dc ? cfg_f(*dc, "rope_theta", 5e5f) : 5e5f;
		const JsonValue* rs = dc ? dc->find("rope_scaling") : nullptr;
		if (rs && rs->is_object() && rs->find("rope_type") &&
		    rs->find("rope_type")->as_string() == "llama3") {
			dp_inv_freq_ = llama3_inv_freq(theta, dp_head_dim_, cfg_f(*rs, "factor", 32.0f),
			                               cfg_f(*rs, "low_freq_factor", 0.125f),
			                               cfg_f(*rs, "high_freq_factor", 0.5f),
			                               cfg_f(*rs, "original_max_position_embeddings", 1024.0f));
		} else {
			dp_inv_freq_ = default_inv_freq(theta, dp_head_dim_);
		}
	}
	// Text encoder: two RoPE tables (sliding theta 1e4, full theta 1e6 scaled by
	// the linear factor) and the per-layer attention type.
	{
		const JsonValue* tc = cfg.find("text_encoder_config");
		te_eps_ = tc ? cfg_f(*tc, "rms_norm_eps", 1e-6f) : 1e-6f;
		te_eoi_ = tc ? cfg_i(*tc, "eoi_token_index", 256000) : 256000;
		te_sliding_ = tc ? cfg_i(*tc, "sliding_window", 512) : 512;
		te_scale_ = std::sqrt((float)te_hidden_);
		const float full_factor = [&] {
			if (!tc) return 1.0f;
			const JsonValue* rp = tc->find("rope_parameters");
			const JsonValue* f = rp ? rp->find("full_attention") : nullptr;
			return f ? cfg_f(*f, "factor", 1.0f) : 1.0f;
		}();
		te_inv_sliding_ = default_inv_freq(10000.0f, te_head_dim_);
		te_inv_full_ = default_inv_freq(1000000.0f, te_head_dim_);
		for (float& v : te_inv_full_) v /= full_factor;
		te_layer_types_.assign((size_t)te_layers_, 0);
		const JsonValue* lt = tc ? tc->find("layer_types") : nullptr;
		if (lt && lt->is_array()) {
			for (i64 i = 0; i < te_layers_ && i < (i64)lt->items().size(); i++)
				te_layer_types_[(size_t)i] = lt->items()[(size_t)i].as_string() == "full_attention" ? 1 : 0;
		} else {
			for (i64 i = 0; i < te_layers_; i++) te_layer_types_[(size_t)i] = ((i + 1) % 6 == 0) ? 1 : 0;
		}
	}
	max_seq_len_ = 2048;   // runtime.py's MAX_SEQ_LEN

	// ── the text encoder's matrices (streamed; only the handles are kept) ──
	te_.assign((size_t)te_layers_, TeLayer{});
	for (i64 i = 0; i < te_layers_; i++) {
		const std::string b = "text_encoder.layers." + std::to_string(i) + ".";
		TeLayer& L = te_[(size_t)i];
		auto mat = [&](const std::string& n, i64 rows, i64 cols) {
			TeMat m;
			m.st = &st_;
			m.t = &st_.require(b + n);
			m.n = m.t->shape[0];
			m.k = m.t->shape[1];
			if (m.n != rows || m.k != cols)
				throw MediaError("breeze: '" + b + n + "' has an unexpected shape");
			return m;
		};
		L.q = mat("self_attn.q_proj.weight", te_heads_ * te_head_dim_, te_hidden_);
		L.k = mat("self_attn.k_proj.weight", te_kv_ * te_head_dim_, te_hidden_);
		L.v = mat("self_attn.v_proj.weight", te_kv_ * te_head_dim_, te_hidden_);
		L.o = mat("self_attn.o_proj.weight", te_hidden_, te_heads_ * te_head_dim_);
		L.gate = mat("mlp.gate_proj.weight", te_inter_, te_hidden_);
		L.up = mat("mlp.up_proj.weight", te_inter_, te_hidden_);
		L.down = mat("mlp.down_proj.weight", te_hidden_, te_inter_);
		L.pre_attn = st_.data_f32(st_.require(b + "pre_self_attn_layernorm.weight"));
		L.post_attn = st_.data_f32(st_.require(b + "post_self_attn_layernorm.weight"));
		L.pre_ff = st_.data_f32(st_.require(b + "pre_feedforward_layernorm.weight"));
		L.post_ff = st_.data_f32(st_.require(b + "post_feedforward_layernorm.weight"));
	}
	te_norm_ = st_.data_f32(st_.require("text_encoder.norm.weight"));
	{
		TeMat m;
		m.st = &st_;
		m.t = &st_.require("text_encoder_proj.weight");
		m.n = m.t->shape[0];
		m.k = m.t->shape[1];
		check("text_encoder_proj is not [backbone hidden, text encoder hidden]",
		      m.n == bb_hidden_ && m.k == te_hidden_);
		te_proj_ = m;
	}

	// ── the resident device weights ──
	g_->ctx->begin();
	auto keep_lin = [&](const std::string& base) {
		QuantLinear q = load_quant_linear(*g_, st_, base, g_->keep, nullptr);
		if (q.n <= 0 || q.k <= 0) throw MediaError("breeze: '" + base + "' did not load");
		return q;
	};
	g_->ctx->begin();
	bb_.assign((size_t)bb_layers_, BackboneLayer{});
	for (i64 i = 0; i < bb_layers_; i++) {
		const std::string b = "backbone_model.layers." + std::to_string(i) + ".";
		BackboneLayer& L = bb_[(size_t)i];
		L.q = keep_lin(b + "self_attn.q_proj");
		L.k = keep_lin(b + "self_attn.k_proj");
		L.v = keep_lin(b + "self_attn.v_proj");
		L.o = keep_lin(b + "self_attn.o_proj");
		L.gate = keep_lin(b + "mlp.gate_proj");
		L.up = keep_lin(b + "mlp.up_proj");
		L.down = keep_lin(b + "mlp.down_proj");
		auto vec = [&](const std::string& n, i64 count) {
			const StTensor& t = st_.require(b + n);
			if (t.numel != count) throw MediaError("breeze: '" + b + n + "' has the wrong length");
			std::vector<float> v = tensor_to_f32(st_, t);
			GpuAlloc a = g_->keep->alloc((u64)count * 4);
			upload_range(*g_->ctx, *g_->ring, a.res, a.off, v.data(), (u64)count * 4);
			return a;
		};
		L.in_ln = vec("input_layernorm.weight", bb_hidden_);
		L.post_ln = vec("post_attention_layernorm.weight", bb_hidden_);
		L.q_norm = vec("self_attn.q_norm.weight", bb_head_dim_);
		L.k_norm = vec("self_attn.k_norm.weight", bb_head_dim_);
	}
	{
		std::vector<float> v = tensor_to_f32(st_, st_.require("backbone_model.norm.weight"));
		bb_norm_ = g_->keep->alloc((u64)bb_hidden_ * 4);
		upload_range(*g_->ctx, *g_->ring, bb_norm_.res, bb_norm_.off, v.data(),
		             (u64)bb_hidden_ * 4);
	}
	lm_head_ = keep_lin("lm_head");
	g_->ctx->submit();

	// ── the depth decoder ──
	g_->ctx->begin();
	dp_.assign((size_t)dp_layers_, DepthLayer{});
	for (i64 i = 0; i < dp_layers_; i++) {
		const std::string b = "depth_decoder.model.layers." + std::to_string(i) + ".";
		DepthLayer& L = dp_[(size_t)i];
		L.q = keep_lin(b + "self_attn.q_proj");
		L.k = keep_lin(b + "self_attn.k_proj");
		L.v = keep_lin(b + "self_attn.v_proj");
		L.o = keep_lin(b + "self_attn.o_proj");
		L.gate = keep_lin(b + "mlp.gate_proj");
		L.up = keep_lin(b + "mlp.up_proj");
		L.down = keep_lin(b + "mlp.down_proj");
		auto vec = [&](const std::string& n, i64 count) {
			std::vector<float> v = tensor_to_f32(st_, st_.require(b + n));
			if ((i64)v.size() != count)
				throw MediaError("breeze: '" + b + n + "' has the wrong length");
			GpuAlloc a = g_->keep->alloc((u64)count * 4);
			upload_range(*g_->ctx, *g_->ring, a.res, a.off, v.data(), (u64)count * 4);
			return a;
		};
		L.in_ln = vec("input_layernorm.weight", dp_hidden_);
		L.post_ln = vec("post_attention_layernorm.weight", dp_hidden_);
	}
	{
		std::vector<float> v = tensor_to_f32(st_, st_.require("depth_decoder.model.norm.weight"));
		dp_norm_ = g_->keep->alloc((u64)dp_hidden_ * 4);
		upload_range(*g_->ctx, *g_->ring, dp_norm_.res, dp_norm_.off, v.data(),
		             (u64)dp_hidden_ * 4);
	}
	dp_projector_ = keep_lin("depth_decoder.model.inputs_embeds_projector");
	g_->ctx->submit();

	// The per-codebook heads: [15, 1024, 2051] -> 15 int8 [2051, 1024] matrices.
	// The reference computes `hidden @ weight[cb].T`, so the int8 operand is the
	// *transpose* of the stored [K, N] slab - the engine's GEMM wants B as [N, K].
	{
		const StTensor& head = st_.require("depth_decoder.codebooks_head.weight");
		const i64 nheads = head.shape[0], hk = head.shape[1], hn = head.shape[2];
		check("codebooks_head is not [num_codebooks - 1, hidden, vocab]",
		      nheads == num_codebooks_ - 1 && hk == dp_hidden_ && hn == dp_vocab_);
		QuantSpec spec;
		spec.format = "int8_tensorwise";
		spec.convrot = true;
		spec.convrot_groupsize = 256;
		spec.valid = true;
		std::vector<float> slab((size_t)(hk * hn));
		std::vector<float> tr((size_t)(hn * hk));
		std::vector<float> scales;
		head_w_.assign((size_t)nheads, GpuAlloc{});
		head_scale_.assign((size_t)nheads, GpuAlloc{});
		g_->ctx->begin();
		for (i64 cb = 0; cb < nheads; cb++) {
			st_.dequant_rows(head, cb * hk, hk, slab.data());
			parallel_for(hn, [&](i64 n) {
				for (i64 k = 0; k < hk; k++) tr[(size_t)(n * hk + k)] = slab[(size_t)(k * hn + n)];
			});
			QuantInt8 qi = quantize_weight(tr.data(), hn, hk, spec);
			head_w_[(size_t)cb] = g_->keep->alloc((u64)qi.q.size());
			upload_range(*g_->ctx, *g_->ring, head_w_[(size_t)cb].res, head_w_[(size_t)cb].off,
			             qi.q.data(), qi.q.size());
			head_scale_[(size_t)cb] = g_->keep->alloc((u64)qi.scale.size() * 4);
			upload_range(*g_->ctx, *g_->ring, head_scale_[(size_t)cb].res,
			             head_scale_[(size_t)cb].off, qi.scale.data(), qi.scale.size() * 4);
		}
		g_->ctx->submit();
	}

	// ── the depth decoder's RoPE table ──
	{
		dp_inv_dev_ = g_->keep->alloc((u64)dp_inv_freq_.size() * 4);
		g_->ctx->begin();
		upload_range(*g_->ctx, *g_->ring, dp_inv_dev_.res, dp_inv_dev_.off, dp_inv_freq_.data(),
		             (u64)dp_inv_freq_.size() * 4);
		g_->ctx->submit();
		g_->ring->rewind();
	}

	// ── the KV caches ──
	//
	// One per branch, in the `keep` arena: a run appends one row per frame for up
	// to MAX_SEQ_LEN tokens, so they have to outlive every step. The backbone's are
	// fp16 (they are the flash kernel's operands) and dominate the model's VRAM
	// after the weights: 2 branches x 28 layers x 2048 rows x 2 KB = 470 MB.
	kv_rows_ = max_seq_len_;
	depth_rows_ = 2 + (num_codebooks_ - 1);
	{
		const i64 bb_kv_dim = bb_kv_ * bb_head_dim_;
		const i64 dp_kv_dim = dp_kv_ * dp_head_dim_;
		g_->ctx->begin();
		for (i64 b = 0; b < 2; b++) {
			Branch& br = branches_[b];
			br.k.assign((size_t)bb_layers_, GpuAlloc{});
			br.v.assign((size_t)bb_layers_, GpuAlloc{});
			for (i64 l = 0; l < bb_layers_; l++) {
				const u64 bytes = (u64)kv_rows_ * (u64)bb_kv_dim * 2;
				br.k[(size_t)l] = g_->keep->alloc(bytes);
				br.v[(size_t)l] = g_->keep->alloc(bytes);
			}
			br.dk.assign((size_t)dp_layers_, GpuAlloc{});
			br.dv.assign((size_t)dp_layers_, GpuAlloc{});
			for (i64 l = 0; l < dp_layers_; l++) {
				const u64 bytes = (u64)depth_rows_ * (u64)dp_kv_dim * 2;
				br.dk[(size_t)l] = g_->keep->alloc(bytes);
				br.dv[(size_t)l] = g_->keep->alloc(bytes);
			}
		}
		g_->ctx->submit();
		g_->ring->rewind();
	}

	sample_buf_.assign((size_t)std::max(lm_head_n_, dp_vocab_), 0.0f);
	loaded_ = true;
}

i64 BreezeTts::sample_rate() const { return codec_.sample_rate(); }

i64 BreezeTts::max_frames_for(i64 prefill_len, i64 max_new_tokens) const {
	const i64 cap = max_seq_len_ - 1 - prefill_len;
	if (cap <= 0) return 0;
	return std::min(max_new_tokens, cap);
}

// ── the prompt (runtime.py's design_segments / _prepare_one) ───────────────

void BreezeTts::build_prompt(const BreezeParams& p) {
	// `design_segments(text, instruction, speaker="S0")` renders
	// "[S0]<ins_bos>{instruction}<ins_eos>{text}" and the negative branch is
	// `design_negative_segments` = "[S0]{text}".
	const std::string cond = "[S0]<ins_bos>" + p.instruction + "<ins_eos>" + p.text;
	const std::string neg = "[S0]" + p.text;
	auto encode_segment = [&](const std::string& s) {
		// `tokenizer(text, add_special_tokens=True)` -> `decode` -> `tokenizer(...,
		// False)`: the reference renders and re-encodes each segment, which is what
		// puts the post-processor's <bos> into the mask as an ordinary token.
		std::vector<i32> ids = tok_.encode(s, true);
		const std::string rendered = tok_.decode(ids, false);
		return tok_.encode(rendered, false);
	};
	branches_[0].ids = encode_segment(cond);
	branches_[1].ids = encode_segment(neg);
}

// ── the text encoder (host fp32) ───────────────────────────────────────────

std::vector<float> BreezeTts::text_encoder(const std::vector<i32>& ids) const {
	const i64 S = (i64)ids.size();
	const i64 H = te_hidden_;
	const i64 HQ = te_heads_ * te_head_dim_;
	const i64 HK = te_kv_ * te_head_dim_;
	std::vector<float> x((size_t)(S * H));
	// `T5Gemma2TextScaledWordEmbedding`: rows scaled by sqrt(hidden), and the EOI
	// row replaced by `eoi_embedding`.
	for (i64 s = 0; s < S; s++) {
		const i32 id = ids[(size_t)s];
		if (id == (i32)te_eoi_) {
			std::memcpy(&x[(size_t)(s * H)], te_eoi_embed_, sizeof(float) * (size_t)H);
			continue;
		}
		if (id < 0 || id >= te_vocab_) throw MediaError("breeze: text token id out of range");
		const float* row = te_embed_t_ + (size_t)id * H;
		for (i64 i = 0; i < H; i++) x[(size_t)(s * H + i)] = row[i] * te_scale_;
	}

	std::vector<float> hn((size_t)(S * H)), q((size_t)(S * HQ)), k((size_t)(S * HK)),
	    v((size_t)(S * HK)), attn((size_t)(S * HQ)), proj((size_t)(S * H)),
	    gate((size_t)(S * te_inter_)), up((size_t)(S * te_inter_)),
	    hh((size_t)(S * te_inter_)), down((size_t)(S * H));

	const float scale = 1.0f / std::sqrt((float)te_hidden_);   // query_pre_attn_scalar^-0.5 == head_dim^-0.5
	for (i64 li = 0; li < te_layers_; li++) {
		const TeLayer& L = te_[(size_t)li];
		const bool full = te_layer_types_[(size_t)li] != 0;
		const std::vector<float>& inv = full ? te_inv_full_ : te_inv_sliding_;
		// ── attention (bidirectional; a 512-wide band on the sliding layers) ──
		te_rms_rows(x.data(), S, H, L.pre_attn, te_eps_, hn.data());
		L.q.gemm(hn.data(), S, q.data());
		L.k.gemm(hn.data(), S, k.data());
		L.v.gemm(hn.data(), S, v.data());
		// Q/K norm: T5Gemma2RMSNorm is (1 + w) over head_dim, per head.
		{
			const std::string b = "text_encoder.layers." + std::to_string(li) + ".";
			const float* qn = st_.data_f32(st_.require(b + "self_attn.q_norm.weight"));
			const float* kn = st_.data_f32(st_.require(b + "self_attn.k_norm.weight"));
			parallel_for(S * te_heads_, [&](i64 r) {
				float* p = q.data() + r * te_head_dim_;
				float sq = 0.0f;
				for (i64 i = 0; i < te_head_dim_; i++) sq += p[i] * p[i];
				const float inv_s = 1.0f / std::sqrt(sq / (float)te_head_dim_ + te_eps_);
				for (i64 i = 0; i < te_head_dim_; i++) p[i] = p[i] * inv_s * (1.0f + qn[i]);
			});
			parallel_for(S * te_kv_, [&](i64 r) {
				float* p = k.data() + r * te_head_dim_;
				float sq = 0.0f;
				for (i64 i = 0; i < te_head_dim_; i++) sq += p[i] * p[i];
				const float inv_s = 1.0f / std::sqrt(sq / (float)te_head_dim_ + te_eps_);
				for (i64 i = 0; i < te_head_dim_; i++) p[i] = p[i] * inv_s * (1.0f + kn[i]);
			});
		}
		rope_rows(q.data(), S, te_heads_, te_head_dim_, inv, 0);
		rope_rows(k.data(), S, te_kv_, te_head_dim_, inv, 0);
		{
			// `local = (0 <= q - kv < 256) | (-257 < q - kv < 0)`, i.e.
			// kv in [q - (sliding+1)/2 + 1, q + sliding/2 + 1).
			const i64 left = (te_sliding_ + 1) / 2;   // 256
			const i64 right = te_sliding_ / 2 + 1;    // 257
			const i64 kv_groups = te_heads_ / te_kv_;
			parallel_for(S, [&](i64 s) {
				std::vector<float> score((size_t)S);
				for (i64 h = 0; h < te_heads_; h++) {
					const float* qp = q.data() + (s * te_heads_ + h) * te_head_dim_;
					const i64 kh = h / kv_groups;
					i64 lo = 0, hi = S;
					if (!full) {
						lo = std::max<i64>(0, s - left + 1);
						hi = std::min<i64>(S, s + right);
					}
					float mx = kNegInf;
					for (i64 kk = lo; kk < hi; kk++) {
						const float* kp = k.data() + (kk * te_kv_ + kh) * te_head_dim_;
						float acc = 0.0f;
						for (i64 d = 0; d < te_head_dim_; d++) acc += qp[d] * kp[d];
						acc *= scale;
						score[(size_t)kk] = acc;
						mx = std::max(mx, acc);
					}
					float sum = 0.0f;
					for (i64 kk = lo; kk < hi; kk++) {
						const float e = std::exp(score[(size_t)kk] - mx);
						score[(size_t)kk] = e;
						sum += e;
					}
					const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
					float* op = attn.data() + (s * te_heads_ + h) * te_head_dim_;
					for (i64 d = 0; d < te_head_dim_; d++) op[d] = 0.0f;
					for (i64 kk = lo; kk < hi; kk++) {
						const float p2 = score[(size_t)kk] * inv_sum;
						const float* vp = v.data() + (kk * te_kv_ + kh) * te_head_dim_;
						for (i64 d = 0; d < te_head_dim_; d++) op[d] += p2 * vp[d];
					}
				}
			});
		}
		L.o.gemm(attn.data(), S, proj.data());
		// post_self_attn_layernorm is applied to the *sublayer output*, then added.
		te_rms_rows(proj.data(), S, H, L.post_attn, te_eps_, proj.data());
		for (i64 i = 0; i < S * H; i++) x[(size_t)i] += proj[(size_t)i];
		// ── mlp (GeGLU, gelu_pytorch_tanh) ──
		te_rms_rows(x.data(), S, H, L.pre_ff, te_eps_, hn.data());
		L.gate.gemm(hn.data(), S, gate.data());
		L.up.gemm(hn.data(), S, up.data());
		for (i64 i = 0; i < S * te_inter_; i++)
			hh[(size_t)i] = gelu_tanh1(gate[(size_t)i]) * up[(size_t)i];
		L.down.gemm(hh.data(), S, down.data());
		te_rms_rows(down.data(), S, H, L.post_ff, te_eps_, down.data());
		for (i64 i = 0; i < S * H; i++) x[(size_t)i] += down[(size_t)i];
	}
	te_rms_rows(x.data(), S, H, te_norm_, te_eps_, x.data());

	// `text_encoder_proj` into the backbone's width.
	std::vector<float> out((size_t)(S * bb_hidden_));
	te_proj_.gemm(x.data(), S, out.data());
	return out;
}


// ── the backbone's 28 layers ───────────────────────────────────────────────

// Runs the backbone over `rows` rows of one branch, with `embeds` (host,
// [rows, hidden]) as the input hidden states. `row0` is the KV row the first new
// key/value lands on, `pos0` the RoPE position of the first row and `keys` the
// total number of keys the attention may see (row0 + rows for a causal prefill).
// Returns the post-norm hidden states ([rows, hidden], device).
GpuAlloc BreezeTts::backbone_run(Branch& b, i64 row0, i64 rows, i64 pos0, i64 keys, bool causal,
                                 const float* embeds) {
	const i64 H = bb_hidden_;
	const i64 HQ = bb_heads_ * bb_head_dim_;
	const i64 HK = bb_kv_ * bb_head_dim_;
	// The fp16 tiled kernel rounds the query axis up to its own tile and reads
	// those rows, so q/o must be at least that tall.
	const i64 qrows = std::max<i64>(rows, 64);
	auto A = [&](u64 bytes) { return g_->aalloc(bytes); };
	GpuAlloc x = A((u64)rows * H * 4);
	GpuAlloc t = A((u64)rows * H * 4);
	GpuAlloc q = A((u64)rows * HQ * 4);
	GpuAlloc k = A((u64)rows * HK * 4);
	GpuAlloc v = A((u64)rows * HK * 4);
	GpuAlloc kn = A((u64)rows * HK * 4);
	GpuAlloc qn = A((u64)rows * HQ * 4);
	GpuAlloc qh = A((u64)qrows * HQ * 2);
	GpuAlloc o = A((u64)qrows * H * 4);
	GpuAlloc gate = A((u64)rows * bb_inter_ * 4);
	GpuAlloc up = A((u64)rows * bb_inter_ * 4);
	GpuAlloc hh = A((u64)rows * bb_inter_ * 4);
	GpuAlloc down = A((u64)rows * H * 4);
	GpuAlloc q8 = A((u64)rows * (u64)std::max(H, bb_inter_));
	GpuAlloc s8 = A((u64)rows * 4);
	GpuAlloc ub;
	if (!causal) {
		// The engine's per-row key bound: one entry per query row of *this*
		// dispatch. Rows past the first are never read back, but the kernel does
		// read their bounds (for rows below the key count), so the table is
		// padded to the tile width.
		ub = A(64 * 4);
		std::vector<u32> host(64, (u32)keys);
		upload_range(*g_->ctx, *g_->ring, ub.res, ub.off, host.data(), 64 * 4);
	}
	upload_range(*g_->ctx, *g_->ring, x.res, x.off, embeds, (u64)rows * H * 4);

	const float qscale = 1.0f / std::sqrt((float)bb_head_dim_);
	for (i64 li = 0; li < bb_layers_; li++) {
		const BackboneLayer& L = bb_[(size_t)li];
		gpu_rms(*g_, x, rows, H, L.in_ln, t, bb_eps_);
		qgemm(*g_, L.q, t, rows, q8, s8, q);
		qgemm(*g_, L.k, t, rows, q8, s8, k);
		qgemm(*g_, L.v, t, rows, q8, s8, v);
		gpu_rms(*g_, q, rows * bb_heads_, bb_head_dim_, L.q_norm, qn, bb_eps_);
		gpu_rms(*g_, k, rows * bb_kv_, bb_head_dim_, L.k_norm, kn, bb_eps_);
		{
			RopeHalfArgs ra;
			ra.head_dim = bb_head_dim_;
			ra.theta = bb_theta_;
			ra.base = pos0;
			ra.rows = rows * bb_heads_;
			ra.heads = bb_heads_;
			ra.x = qn;
			ra.y = q;
			dispatch_rope_half(*g_->ctx, qn.res, ra);
			ra.rows = rows * bb_kv_;
			ra.heads = bb_kv_;
			ra.x = kn;
			ra.y = k;
			dispatch_rope_half(*g_->ctx, kn.res, ra);
		}
		// fp16 operands for the tensor-core kernel; k/v go straight into the
		// branch's cache at row0.
		GpuAlloc kdst{b.k[(size_t)li].res, b.k[(size_t)li].off + (u64)row0 * (u64)HK * 2,
		              (u64)rows * HK * 2};
		GpuAlloc vdst{b.v[(size_t)li].res, b.v[(size_t)li].off + (u64)row0 * (u64)HK * 2,
		              (u64)rows * HK * 2};
		dispatch_f32_to_f16(*g_->ctx, q, qh, rows * HQ);
		dispatch_f32_to_f16(*g_->ctx, k, kdst, rows * HK);
		dispatch_f32_to_f16(*g_->ctx, v, vdst, rows * HK);
		{
			AttnFlashArgs aa;
			aa.q = qh;
			aa.k = b.k[(size_t)li];
			aa.v = b.v[(size_t)li];
			aa.o = o;
			aa.s = keys;
			aa.q_rows = rows;
			aa.heads = bb_heads_;
			aa.kv_heads = bb_kv_;
			aa.head_dim = bb_head_dim_;
			aa.f32_input = false;
			aa.causal = causal;
			aa.ub = ub;
			aa.scale = qscale;
			dispatch_attn_flash(*g_->ctx, aa);
		}
		qgemm(*g_, L.o, o, rows, q8, s8, t);
		gpu_add(*g_, x, t, rows, H);
		gpu_rms(*g_, x, rows, H, L.post_ln, t, bb_eps_);
		qgemm(*g_, L.gate, t, rows, q8, s8, gate);
		qgemm(*g_, L.up, t, rows, q8, s8, up);
		gpu_silu_gate(*g_, gate, up, hh, rows, bb_inter_);
		qgemm(*g_, L.down, hh, rows, q8, s8, down);
		gpu_add(*g_, x, down, rows, H);
	}
	GpuAlloc hout = A((u64)rows * H * 4);
	gpu_rms(*g_, x, rows, H, bb_norm_, hout, bb_eps_);
	return hout;
}

// ── the depth decoder's 12 layers ──────────────────────────────────────────

// A table-driven half-split RoPE, because the depth decoder's inverse
// frequencies are not `theta^(-2i/D)`: its rope_scaling is llama3, which rescales
// each frequency by wavelength (see llama3_inv_freq). `dispatch_rope_half` only
// takes a theta, so this is the one kernel this file compiles itself - the engine
// builds every kernel from source at runtime, so this is the same machinery the
// dispatchers use, not a new dependency.
static const char* kBreezeRopeTable = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};
// values: 0 rows, 1 heads, 2 head_dim, 3 base   srv: 0 = X, 1 = INV   uav: 0 = Y
// [numthreads(head_dim / 2, 1, 1)], grid (rows * heads)
extern "C" __global__ void breeze_rope_table(Args a) {
    const unsigned rows = a.v[0], heads = a.v[1], D = a.v[2], base = a.v[3];
    const unsigned half = D / 2u;
    const unsigned r = blockIdx.x;
    const unsigned t = threadIdx.x;
    if (r >= rows * heads || t >= half) return;
    const float* X = (const float*)a.s[0];
    float* Y = (float*)a.u[0];
    const float* INV = (const float*)a.s[1];
    const unsigned pos = r / heads + base;
    const float ang = (float)pos * INV[t];
    const float c = cosf(ang), s2 = sinf(ang);
    const unsigned b = r * D;
    const float x0 = X[b + t], x1 = X[b + t + half];
    Y[b + t] = x0 * c - x1 * s2;
    Y[b + t + half] = x1 * c + x0 * s2;
}
)CUDA";

void BreezeTts::rope_table(const GpuAlloc& x, const GpuAlloc& y, i64 rows, i64 heads, i64 head_dim,
                           i64 base) {
	GpuKernel* k = g_->ctx->pipeline("breeze_rope_table", kBreezeRopeTable, "breeze_rope_table",
	                                 ShaderModel::SM5_1, (u32)(head_dim / 2));
	KernelParams p{};
	p.values[0] = (u32)rows;
	p.values[1] = (u32)heads;
	p.values[2] = (u32)head_dim;
	p.values[3] = (u32)base;
	p.srv[0] = x.res;
	p.srv_offset[0] = x.off;
	p.srv[1] = dp_inv_dev_.res;
	p.srv_offset[1] = dp_inv_dev_.off;
	p.uav[0] = y.res;
	p.uav_offset[0] = y.off;
	g_->ctx->dispatch(k, p, (u32)(rows * heads), 1, 1);
}

GpuAlloc BreezeTts::depth_run(Branch& b, i64 row0, i64 rows, i64 pos0, i64 keys, bool causal,
                              const float* embeds) {
	const i64 H = dp_hidden_;
	const i64 HQ = dp_heads_ * dp_head_dim_;
	const i64 HK = dp_kv_ * dp_head_dim_;
	const i64 qrows = std::max<i64>(rows, 64);
	auto A = [&](u64 bytes) { return g_->aalloc(bytes); };
	GpuAlloc x = A((u64)rows * H * 4);
	GpuAlloc t = A((u64)rows * H * 4);
	GpuAlloc q = A((u64)rows * HQ * 4);
	GpuAlloc k = A((u64)rows * HK * 4);
	GpuAlloc v = A((u64)rows * HK * 4);
	GpuAlloc qh = A((u64)qrows * HQ * 2);
	GpuAlloc o = A((u64)qrows * H * 4);
	GpuAlloc gate = A((u64)rows * dp_inter_ * 4);
	GpuAlloc up = A((u64)rows * dp_inter_ * 4);
	GpuAlloc hh = A((u64)rows * dp_inter_ * 4);
	GpuAlloc down = A((u64)rows * H * 4);
	GpuAlloc q8 = A((u64)rows * (u64)std::max(H, dp_inter_));
	GpuAlloc s8 = A((u64)rows * 4);
	GpuAlloc ub;
	if (!causal) {
		ub = A(64 * 4);
		std::vector<u32> host(64, (u32)keys);
		upload_range(*g_->ctx, *g_->ring, ub.res, ub.off, host.data(), 64 * 4);
	}
	{
		// `inputs_embeds_projector(embed_tokens(ids))`: `embeds` (host) is the
		// gather's result, [rows, embed].
		GpuAlloc src = A((u64)rows * dp_embed_dim_ * 4);
		upload_range(*g_->ctx, *g_->ring, src.res, src.off, embeds,
		             (u64)rows * dp_embed_dim_ * 4);
		qgemm(*g_, dp_projector_, src, rows, q8, s8, x);
	}

	const float qscale = 1.0f / std::sqrt((float)dp_head_dim_);
	for (i64 li = 0; li < dp_layers_; li++) {
		const DepthLayer& L = dp_[(size_t)li];
		gpu_rms(*g_, x, rows, H, L.in_ln, t, dp_eps_);
		qgemm(*g_, L.q, t, rows, q8, s8, q);
		qgemm(*g_, L.k, t, rows, q8, s8, k);
		qgemm(*g_, L.v, t, rows, q8, s8, v);
		rope_table(q, q, rows, dp_heads_, dp_head_dim_, pos0);
		rope_table(k, k, rows, dp_kv_, dp_head_dim_, pos0);
		GpuAlloc kdst{b.dk[(size_t)li].res, b.dk[(size_t)li].off + (u64)row0 * (u64)HK * 2,
		              (u64)rows * HK * 2};
		GpuAlloc vdst{b.dv[(size_t)li].res, b.dv[(size_t)li].off + (u64)row0 * (u64)HK * 2,
		              (u64)rows * HK * 2};
		dispatch_f32_to_f16(*g_->ctx, q, qh, rows * HQ);
		dispatch_f32_to_f16(*g_->ctx, k, kdst, rows * HK);
		dispatch_f32_to_f16(*g_->ctx, v, vdst, rows * HK);
		{
			AttnFlashArgs aa;
			aa.q = qh;
			aa.k = b.dk[(size_t)li];
			aa.v = b.dv[(size_t)li];
			aa.o = o;
			aa.s = keys;
			aa.q_rows = rows;
			aa.heads = dp_heads_;
			aa.kv_heads = dp_kv_;
			aa.head_dim = dp_head_dim_;
			aa.f32_input = false;
			aa.causal = causal;
			aa.ub = ub;
			aa.scale = qscale;
			dispatch_attn_flash(*g_->ctx, aa);
		}
		qgemm(*g_, L.o, o, rows, q8, s8, t);
		gpu_add(*g_, x, t, rows, H);
		gpu_rms(*g_, x, rows, H, L.post_ln, t, dp_eps_);
		qgemm(*g_, L.gate, t, rows, q8, s8, gate);
		qgemm(*g_, L.up, t, rows, q8, s8, up);
		gpu_silu_gate(*g_, gate, up, hh, rows, dp_inter_);
		qgemm(*g_, L.down, hh, rows, q8, s8, down);
		gpu_add(*g_, x, down, rows, H);
	}
	GpuAlloc hout = A((u64)rows * H * 4);
	gpu_rms(*g_, x, rows, H, dp_norm_, hout, dp_eps_);
	return hout;
}

// `codebooks_head[cb]` applied to one row of the depth hidden state.
void BreezeTts::head_apply(const GpuAlloc& hidden, i64 row, i64 cb, const GpuAlloc& out) {
	QuantLinear w;
	w.w = head_w_[(size_t)cb];
	w.scale = head_scale_[(size_t)cb];
	w.n = dp_vocab_;
	w.k = dp_hidden_;
	GpuAlloc a{hidden.res, hidden.off + (u64)row * (u64)dp_hidden_ * 4, (u64)dp_hidden_ * 4};
	GpuAlloc q8 = g_->aalloc((u64)dp_hidden_);
	GpuAlloc s8 = g_->aalloc(4);
	qgemm(*g_, w, a, 1, q8, s8, out);
}

// The tied audio embedding table: `embed_tokens[cb * audio_vocab + code]`, a
// 2048-wide bf16 row. Read out of the mapping one row at a time.
void BreezeTts::embed_row(i64 row, float* out) const {
	st_.dequant_rows(*dp_embed_t_, row, 1, out);
}

void BreezeTts::frame_embed(const i32* frame, float* out) const {
	std::vector<float> row((size_t)dp_embed_dim_);
	std::memset(out, 0, sizeof(float) * (size_t)dp_embed_dim_);
	for (i64 cb = 0; cb < num_codebooks_; cb++) {
		i32 code = frame[cb];
		if (code < 0) code = 0;
		if (code >= audio_vocab_) code = (i32)audio_vocab_ - 1;
		embed_row(cb * audio_vocab_ + code, row.data());
		for (i64 i = 0; i < dp_embed_dim_; i++) out[i] += row[(size_t)i];
	}
}

void BreezeTts::depth_frame(const std::vector<float>& backbone_hidden, i64 nb, i32 first_token,
                            const BreezeParams& p, Rng& rng, std::vector<i32>& out15) {
	out15.clear();
	std::vector<float> emb((size_t)(2 * dp_embed_dim_), 0.0f);
	std::vector<float> row((size_t)dp_embed_dim_);
	std::vector<float> lg[2];
	std::vector<GpuAlloc> lg_dev(2);
	std::vector<i32> depth_hist;

	auto sample_depth = [&](std::vector<float> a, std::vector<float> c) {
		std::vector<float>& v = sample_buf_;
		if (nb >= 2) {
			for (i64 i = 0; i < dp_vocab_; i++)
				v[(size_t)i] = c[(size_t)i] + p.cfg_scale * (a[(size_t)i] - c[(size_t)i]);
		} else {
			for (i64 i = 0; i < dp_vocab_; i++) v[(size_t)i] = a[(size_t)i];
		}
		return sample_logits(v, p.depth_temperature, p.depth_top_k, p.depth_top_p, 1.0f, nullptr,
		                     codec_codebook_size_, dp_vocab_, rng);
	};

	// ── prefill: position 0 is the backbone's hidden state, position 1 the
	// first codebook's embedding ──
	g_->new_step();
	g_->ctx->begin();
	for (i64 b = 0; b < nb; b++) {
		std::memcpy(emb.data(), backbone_hidden.data() + (size_t)(b * dp_embed_dim_),
		            sizeof(float) * (size_t)dp_embed_dim_);
		embed_row(0 * audio_vocab_ + first_token, row.data());
		std::memcpy(emb.data() + dp_embed_dim_, row.data(), sizeof(float) * (size_t)dp_embed_dim_);
		GpuAlloc h = depth_run(branches_[(size_t)b], 0, 2, 0, 2, true, emb.data());
		lg_dev[(size_t)b] = g_->aalloc((u64)dp_vocab_ * 4);
		head_apply(h, 1, 0, lg_dev[(size_t)b]);
	}
	g_->ctx->submit();
	g_->ring->rewind();
	for (i64 b = 0; b < nb; b++) {
		lg[(size_t)b].assign((size_t)dp_vocab_, 0.0f);
		g_->ctx->download(lg_dev[(size_t)b].res, lg_dev[(size_t)b].off, lg[(size_t)b].data(),
		                  (u64)dp_vocab_ * 4);
	}
	i32 token = sample_depth(lg[0], lg[nb - 1]);
	out15.push_back(token);

	// ── 14 further positions: one token, one step each ──
	for (i64 cb = 1; cb < num_codebooks_ - 1; cb++) {
		const i64 pos = cb + 1;
		g_->new_step();
		g_->ctx->begin();
		for (i64 b = 0; b < nb; b++) {
			embed_row(cb * audio_vocab_ + token, row.data());
			std::memcpy(emb.data(), row.data(), sizeof(float) * (size_t)dp_embed_dim_);
			GpuAlloc h = depth_run(branches_[(size_t)b], pos, 1, pos, pos + 1, false, emb.data());
			lg_dev[(size_t)b] = g_->aalloc((u64)dp_vocab_ * 4);
			head_apply(h, 0, cb, lg_dev[(size_t)b]);
		}
		g_->ctx->submit();
		g_->ring->rewind();
		for (i64 b = 0; b < nb; b++) {
			lg[(size_t)b].assign((size_t)dp_vocab_, 0.0f);
			g_->ctx->download(lg_dev[(size_t)b].res, lg_dev[(size_t)b].off, lg[(size_t)b].data(),
			                  (u64)dp_vocab_ * 4);
		}
		token = sample_depth(lg[0], lg[nb - 1]);
		out15.push_back(token);
	}
}

// ── speak ──────────────────────────────────────────────────────────────────

i64 BreezeTts::estimate_frames(const std::string& text) const {
	// runtime.py's `estimate_speech_frames`: en ~4.1 frames per text token, zh
	// ~3.5, each inline vocal event ~5 frames - calibrated on the official
	// checkpoint. Used only when the caller leaves max_new_tokens at 0.
	static const char* kEvents[] = {"(laugh)",  "(laughs)", "(laughing)",      "(cough)",
	                                "(coughs)", "(sigh)",   "(sighs)",         "(sniff)",
	                                "(sneeze)", "(groan)",  "(gasp)",          "(hum)",
	                                "(clears throat)"};
	i64 events = 0;
	std::string stripped;
	for (size_t i = 0; i < text.size();) {
		bool hit = false;
		for (const char* e : kEvents) {
			const size_t n = std::strlen(e);
			if (i + n <= text.size() && text.compare(i, n, e) == 0) {
				events++;
				i += n;
				hit = true;
				break;
			}
		}
		if (hit) continue;
		if (text[i] == '[') {
			// [笑] [笑声] [咳嗽] [清嗓子] [叹气] [叹息] [抽泣] [哭] [喘息] [呼气]
			static const char* zh[] = {"\xE7\xAC\x91",                // 笑
			                           "\xE7\xAC\x91\xE5\xA3\xB0",     // 笑声
			                           "\xE5\x92\xB3\xE5\x97\xBD",     // 咳嗽
			                           "\xE6\xB8\x85\xE5\x97\x93\xE5\xAD\x90",  // 清嗓子
			                           "\xE5\x8F\xB9\xE6\xB0\x94",     // 叹气
			                           "\xE5\x8F\xB9\xE6\x81\xAF",     // 叹息
			                           "\xE6\x8A\xBD\xE6\xB3\xA3",     // 抽泣
			                           "\xE5\x93\xAD",                 // 哭
			                           "\xE5\x96\x98\xE6\x81\xAF",     // 喘息
			                           "\xE5\x91\xBC\xE6\xB0\x94"};    // 呼气
			bool z = false;
			for (const char* e : zh) {
				const size_t n = std::strlen(e);
				if (i + 1 + n < text.size() && text[i + 1 + n] == ']' &&
				    text.compare(i + 1, n, e) == 0) {
					events++;
					i += n + 2;
					z = true;
					break;
				}
			}
			if (z) continue;
		}
		stripped.push_back(text[i]);
		i++;
	}
	std::vector<i32> ids = tok_.encode(stripped, false);
	if (ids.empty()) return std::max<i64>(16, (i64)std::llround((double)events * 5.0 + 8.0));
	// `convert_ids_to_tokens` then "does the token hold a CJK codepoint?"
	i64 cjk = 0;
	for (i32 id : ids) {
		const std::string s = tok_.decode({id}, false);
		for (size_t i = 0; i < s.size();) {
			const unsigned char c = (unsigned char)s[i];
			if (c < 0x80) {
				i++;
				continue;
			}
			u32 cp = 0;
			size_t n = 1;
			if ((c & 0xE0) == 0xC0) {
				cp = c & 0x1F;
				n = 2;
			} else if ((c & 0xF0) == 0xE0) {
				cp = c & 0x0F;
				n = 3;
			} else {
				cp = c & 0x07;
				n = 4;
			}
			for (size_t k = 1; k < n && i + k < s.size(); k++)
				cp = (cp << 6) | ((u8)s[i + k] & 0x3F);
			if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF)) {
				cjk++;
				break;
			}
			i += n;
		}
	}
	const i64 other = (i64)ids.size() - cjk;
	const double est = (double)cjk * 3.5 + (double)other * 4.1 + (double)events * 5.0 + 4.0;
	return std::max<i64>(16, (i64)std::llround(est));
}

// ── stage 1: the prompt and the text encoder (host fp32) ────────────────────
std::shared_ptr<BreezeTts::Prompt> BreezeTts::prepare(const BreezeParams& p) {
	if (!loaded_) throw MediaError("breeze: speak before open");
	if (p.text.empty()) throw MediaError("breeze: the text is empty");
	build_prompt(p);
	const i64 nb = (p.cfg_scale != 1.0f) ? 2 : 1;
	i64 prefill = 0;
	for (i64 b = 0; b < nb; b++) prefill = std::max(prefill, (i64)branches_[(size_t)b].ids.size());
	if (prefill <= 0) throw MediaError("breeze: the prompt tokenised to nothing");
	// The frame budget. When the caller does not give one, take the reference
	// node's own default (1500 frames = 120 s, `_generation_controls`) and let
	// `max_frames_for` clip it to the context: the model stops at EOS by itself,
	// so a budget derived from the *text* would only ever cut a sentence off
	// mid-word (`estimate_frames` is a display estimate, not a bound).
	const i64 want = p.max_new_tokens > 0 ? p.max_new_tokens : 1500;
	const i64 max_frames = max_frames_for(prefill, want);
	if (max_frames <= 0)
		throw MediaError("breeze: the prompt is " + std::to_string(prefill) + " of the " +
		                 std::to_string(max_seq_len_) + " token context");

	for (i64 b = 0; b < nb; b++) {
		Branch& br = branches_[(size_t)b];
		br.prompt = (i64)br.ids.size();
		br.len = br.prompt;
		br.base = br.prompt;
		prefill_embeds_[(size_t)b] = text_encoder(br.ids);
	}


	auto out = std::make_shared<Prompt>();
	out->prefill = prefill;
	out->branches = nb;
	out->max_frames = max_frames;
	return out;
}

// ── stage 2: the backbone + depth-decoder token loop ─────────────────────────
std::vector<i32> BreezeTts::generate_codes(const std::shared_ptr<Prompt>& prompt, const BreezeParams& p,
                                           i64* frames_out,
                                           const std::function<void(i64, i64)>& progress,
                                           const std::function<bool()>& cancelled) {
	if (!loaded_) throw MediaError("breeze: speak before open");
	if (!prompt) throw MediaError("breeze: no prepared prompt");
	const i64 nb = prompt->branches;
	const i64 max_frames = prompt->max_frames;

	Rng rng;
	rng.s = p.seed ? p.seed : make_media_seed();

	std::vector<float> hidden_host((size_t)(nb * dp_embed_dim_), 0.0f);
	std::vector<std::vector<float>> host_logits((size_t)nb);
	std::vector<float> hrow((size_t)bb_hidden_, 0.0f);

	// ── prefill the backbone and take the first token ──
	i32 token = 0;
	{
		g_->new_step();
		g_->ctx->begin();
		std::vector<GpuAlloc> hid((size_t)nb), lg((size_t)nb);
		GpuAlloc q8 = g_->aalloc((u64)prompt->prefill * (u64)std::max(bb_hidden_, bb_inter_));
		GpuAlloc s8 = g_->aalloc((u64)prompt->prefill * 4);
		for (i64 b = 0; b < nb; b++) {
			Branch& br = branches_[(size_t)b];
			hid[(size_t)b] = backbone_run(br, 0, br.prompt, 0, br.prompt, true,
			                              prefill_embeds_[(size_t)b].data());
		}
		for (i64 b = 0; b < nb; b++) {
			Branch& br = branches_[(size_t)b];
			GpuAlloc row{hid[(size_t)b].res,
			             hid[(size_t)b].off + (u64)(br.prompt - 1) * (u64)bb_hidden_ * 4,
			             (u64)bb_hidden_ * 4};
			lg[(size_t)b] = g_->aalloc((u64)lm_head_n_ * 4);
			qgemm(*g_, lm_head_, row, 1, q8, s8, lg[(size_t)b]);
		}
		g_->ctx->submit();
		g_->ring->rewind();
		for (i64 b = 0; b < nb; b++) {
			Branch& br = branches_[(size_t)b];
			host_logits[(size_t)b].assign((size_t)lm_head_n_, 0.0f);
			g_->ctx->download(lg[(size_t)b].res, lg[(size_t)b].off, host_logits[(size_t)b].data(),
			                  (u64)lm_head_n_ * 4);
			// The depth decoder's position 0 is the post-norm hidden state.
			g_->ctx->download(hid[(size_t)b].res,
			                  hid[(size_t)b].off + (u64)(br.prompt - 1) * (u64)bb_hidden_ * 4,
			                  hrow.data(), (u64)bb_hidden_ * 4);
			std::memcpy(hidden_host.data() + (size_t)(b * dp_embed_dim_), hrow.data(),
			            sizeof(float) * (size_t)dp_embed_dim_);
		}
		std::vector<float>& v = sample_buf_;
		std::vector<float>& a = host_logits[0];
		std::vector<float>& c = host_logits[(size_t)nb - 1];
		for (i64 i = 0; i < lm_head_n_; i++)
			v[(size_t)i] =
			    nb >= 2 ? c[(size_t)i] + p.cfg_scale * (a[(size_t)i] - c[(size_t)i]) : a[(size_t)i];
		token = sample_logits(v, p.temperature, p.top_k, p.top_p, 1.0f, nullptr,
		                      codec_codebook_size_, audio_vocab_, rng);
	}

	// ── the frame loop (runtime.py's generate_codes) ──
	codes_.clear();
	std::vector<i32> history;
	history.reserve((size_t)max_frames);
	i64 frames = 0;
	std::vector<i32> d15;
	std::vector<float> frame_emb((size_t)bb_hidden_, 0.0f);
	i32 frame[16];
	for (i64 step = 0; step < max_frames; step++) {
		if (cancelled && cancelled()) break;
		if ((i64)token == backbone_eos_) break;   // config.vocab_size

		depth_frame(hidden_host, nb, token, p, rng, d15);
		frame[0] = token;
		for (i64 i = 1; i < num_codebooks_; i++)
			frame[i] = (i - 1) < (i64)d15.size() ? d15[(size_t)(i - 1)] : (i32)codebook_eos_;
		bool all_pad = true;
		for (i64 i = 0; i < num_codebooks_; i++)
			if (frame[i] != (i32)codebook_pad_) all_pad = false;
		if (!all_pad) {
			for (i64 i = 0; i < num_codebooks_; i++) codes_.push_back(frame[i]);
			frames++;
		}
		if (progress) progress(step + 1, max_frames);
		history.push_back(token);

		frame_embed(frame, frame_emb.data());
		g_->new_step();
		g_->ctx->begin();
		std::vector<GpuAlloc> hid((size_t)nb), lg((size_t)nb);
		GpuAlloc q8 = g_->aalloc((u64)std::max(bb_hidden_, bb_inter_));
		GpuAlloc s8 = g_->aalloc(4);
		for (i64 b = 0; b < nb; b++) {
			Branch& br = branches_[(size_t)b];
			hid[(size_t)b] =
			    backbone_run(br, br.len, 1, br.len, br.len + 1, false, frame_emb.data());
			lg[(size_t)b] = g_->aalloc((u64)lm_head_n_ * 4);
			qgemm(*g_, lm_head_, hid[(size_t)b], 1, q8, s8, lg[(size_t)b]);
		}
		g_->ctx->submit();
		g_->ring->rewind();
		for (i64 b = 0; b < nb; b++) {
			host_logits[(size_t)b].assign((size_t)lm_head_n_, 0.0f);
			g_->ctx->download(lg[(size_t)b].res, lg[(size_t)b].off, host_logits[(size_t)b].data(),
			                  (u64)lm_head_n_ * 4);
			g_->ctx->download(hid[(size_t)b].res, hid[(size_t)b].off, hrow.data(),
			                  (u64)bb_hidden_ * 4);
			std::memcpy(hidden_host.data() + (size_t)(b * dp_embed_dim_), hrow.data(),
			            sizeof(float) * (size_t)dp_embed_dim_);
			branches_[(size_t)b].len += 1;
		}
		std::vector<float>& v = sample_buf_;
		std::vector<float>& a = host_logits[0];
		std::vector<float>& c = host_logits[(size_t)nb - 1];
		for (i64 i = 0; i < lm_head_n_; i++)
			v[(size_t)i] =
			    nb >= 2 ? c[(size_t)i] + p.cfg_scale * (a[(size_t)i] - c[(size_t)i]) : a[(size_t)i];
		token = sample_logits(v, p.temperature, p.top_k, p.top_p, p.repetition_penalty, &history,
		                      codec_codebook_size_, audio_vocab_, rng);
	}
	if (frames <= 0) throw MediaError("breeze: no audio frames were produced");
	if (frames_out) *frames_out = frames;
	return codes_;
}

// ── stage 3: the codec ───────────────────────────────────────────────────────
std::vector<float> BreezeTts::decode_codes(const std::vector<i32>& codes, i64 frames) {
	if (frames <= 0 || (i64)codes.size() < frames * num_codebooks_)
		throw MediaError("breeze: no audio frames to decode");
	return codec_.decode(codes, frames);
}

// ── speak: the three stages, in order ────────────────────────────────────────
std::vector<float> BreezeTts::speak(const BreezeParams& p,
                                    const std::function<void(i64, i64)>& progress,
                                    const std::function<bool()>& cancelled) {
	auto prompt = prepare(p);
	i64 frames = 0;
	std::vector<i32> codes = generate_codes(prompt, p, &frames, progress, cancelled);
	return decode_codes(codes, frames);
}

}  // namespace phi::media
