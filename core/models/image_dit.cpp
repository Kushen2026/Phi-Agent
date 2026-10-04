// Qwen-Image-2.1 DiT implementation. See image_dit.hpp for the mapping to
// comfy/ldm/qwen_image21/model.py and for what each part mirrors.
//
// Shape of a step: the two bf16 "embedders" (img_in, the text projection's two
// linears) and the adaLN/out-projection run as ordinary `gemm_f16` calls against
// the checkpoint's bf16 weights; the 32 blocks run int8 convrot through the
// `quant_gemm` pair (`quant_convrot` + `int8_gemm`); the timestep MLP, the
// modulation and the final scale are small enough (4096-wide vectors) that they
// run on the host, where the weights are kept as fp32 once.
#include "models/image_dit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "kernels/gpu_ops.hpp"
#include "kernels/h3_kernels.hpp"
#include "models/lora.hpp"
#include "runtime/vram_budget.hpp"
#include "runtime/vram_window.hpp"

namespace phi::media {

namespace {

bool qi_debug() {
	static int v = -1;
	if (v < 0) {
		const char* e = getenv("PHI_QI_DEBUG");
		v = (e && *e && *e != '0') ? 1 : 0;
	}
	return v != 0;
}

void stat(ComputeContext& ctx, const char* name, const GpuAlloc& a, i64 n) {
	if (!qi_debug()) return;
	const bool was = ctx.recording();
	ctx.submit_if_recording();
	std::vector<float> v((size_t)n);
	ctx.download(a.res, a.off, v.data(), (size_t)n * 4);
	double mn = 1e30, mx = -1e30, sum = 0;
	long long nan = 0;
	for (float f : v) {
		if (std::isnan(f)) nan++;
		mn = std::min(mn, (double)f);
		mx = std::max(mx, (double)f);
		sum += f;
	}
	printf("[qi] %-14s [%.4g, %.4g] mean %+.4g nan %lld\n", name, mn, mx, sum / (double)v.size(),
	       nan);
	if (was) ctx.begin();
}

// The RoPE table `qkv_prep` reads: [omega(D/2) | axis(D/2)], built from
// `EmbedND`/`rope()`'s per-axis linspace.
void build_rope_table(const ImageDitConfig& cfg, std::vector<float>& out) {
	const int D = (int)cfg.head_dim, half = D / 2;
	out.assign((size_t)D, 0.0f);
	int pair = 0;
	for (size_t ax = 0; ax < cfg.axes_dims.size() && pair < half; ax++) {
		const int dim = cfg.axes_dims[ax];
		const int h = dim / 2;
		const double step = (h <= 1) ? 0.0 : ((double)(dim - 2) / (double)dim) / (double)(h - 1);
		for (int i = 0; i < h && pair < half; i++, pair++) {
			out[(size_t)pair] = (float)(1.0 / std::pow((double)cfg.rope_theta, step * (double)i));
			out[(size_t)(half + pair)] = (float)ax;
		}
	}
}

// A sub-range of a GpuAlloc, the byte-offset helper the block arithmetic is
// written with.
GpuAlloc sub(const GpuAlloc& a, u64 byte_off, u64 bytes) {
	GpuAlloc r = a;
	r.off += byte_off;
	r.bytes = bytes;
	return r;
}

std::vector<float> bf16_host(const SafeTensors& st, const std::string& name) {
	return tensor_to_f32(st, st.require(name));
}

// y = W x, W stored [out, in] row major fp32.
std::vector<float> host_linear(const std::vector<float>& w, i64 n, i64 k,
                               const std::vector<float>& x) {
	std::vector<float> y((size_t)n);
	for (i64 o = 0; o < n; o++) {
		const float* wr = &w[(size_t)o * k];
		double acc = 0;
		for (i64 i = 0; i < k; i++) acc += (double)wr[i] * (double)x[(size_t)i];
		y[(size_t)o] = (float)acc;
	}
	return y;
}

std::vector<float> silu_v(std::vector<float> v) {
	for (float& f : v) f = f / (1.0f + std::exp(-f));
	return v;
}

// comfy/ldm/flux/layers.py :: timestep_embedding(t, 256, time_factor=1000)
std::vector<float> timestep_embedding(float t, i64 dim) {
	std::vector<float> e((size_t)dim);
	const i64 half = dim / 2;
	const double tt = 1000.0 * (double)t;
	for (i64 j = 0; j < half; j++) {
		const double f = std::exp(-std::log(10000.0) * (double)j / (double)half);
		const double a = tt * f;
		e[(size_t)j] = (float)std::cos(a);
		e[(size_t)(half + j)] = (float)std::sin(a);
	}
	return e;
}

}  // namespace

void QIScratch::alloc(GpuCtx* g, i64 S, i64 dim, i64 mlp, i64 ch, i64 n_img, i64 n_ref_tok,
                      i64 ref_px) {
	auto A = [&](i64 n) { return g->aalloc((u64)n * 4); };
	hidden = A(S * dim);
	txt = A(((i64)((S > 0) ? S : 1)) * dim);
	normed = A(S * dim);
	qkv = A(S * 3 * dim);
	q = g->aalloc((u64)S * dim * 2);
	k = g->aalloc((u64)S * dim * 2);
	v = g->aalloc((u64)S * dim * 2);
	attn = A(S * dim);
	proj = A(S * dim);
	gate_up = A(S * 2 * mlp);
	hh = A(S * mlp);
	down = A(S * dim);
	outp = A(n_img * ch);
	ids = A(S * 3);
	ub = g->aalloc((u64)S * 4);
	ub_t = g->aalloc((u64)std::max<i64>(S, 1) * 4);
	(void)n_ref_tok;
	(void)ref_px;
}

// ── open / probe ──────────────────────────────────────────────────────────

void ImageDiT::probe() {
	st_.open(path_);
	const StTensor& im = st_.require("img_in.weight");
	cfg_.dim = im.shape[0];
	cfg_.in_channels = im.shape[1];
	cfg_.context_dim = st_.require("txt_in.in_layer.weight").shape[0];
	cfg_.mlp_hidden = st_.require("transformer_blocks.0.img_mlp.gate_up.weight").shape[0] / 2;
	cfg_.head_dim = st_.require("transformer_blocks.0.attn.norm_q.weight").shape[0];
	cfg_.n_heads = st_.require("transformer_blocks.0.attn.to_q.weight").shape[0] / cfg_.head_dim;
	i64 l = 0;
	while (st_.find("transformer_blocks." + std::to_string(l) + ".attn.to_q.weight")) l++;
	cfg_.n_layers = l;
	cfg_.t_freq = st_.require("time_text_embed.timestep_embedder.linear_1.weight").shape[1];
	if (st_.require("modulation.1.weight").shape[0] != 4 * cfg_.dim)
		throw MediaError("qwen_image: the modulation projection is not 4*dim wide");
	if (cfg_.head_dim != 128) throw MediaError("qwen_image: head_dim != 128");
	// One block's tensor bytes, summed from the header the same way
	// `layer_alloc_sizes` prices a streamed block. The runtime tuner needs the
	// *tensor* size (it rounds to the resident arena's chunks itself), not the
	// arena's measured growth for whichever block happened to be uploaded first.
	{
		u64 lb = 0;
		for (u64 b : layer_alloc_sizes(0)) lb += b;
		res_layer_bytes_ = lb;
	}
}

void ImageDiT::open(const std::string& path, GpuCtx* gpu) {
	g_ = gpu;
	path_ = path;
	probe();
	// The window `plan_residency()` fills. Its chunk granularity is the weight
	// arena's, so the plan's charge and the accountant's are the same number.
	res_arena_.init(g_->ctx, (g_->wa ? g_->wa->chunk_bytes() : (64ull << 20)));
	res_arena_.set_tag("image.dit_resident");
	g_->new_step();
	g_->ctx->begin();

	auto bf16 = [&](const std::string& name) {
		const StTensor& t = st_.require(name);
		// `t.numel` (not `t.nbytes`) so the same shape is charged whatever the
		// source precision; materialising it bf16 is what the embedders read.
		GpuAlloc a = g_->kalloc((u64)t.numel * 2);
		g_->upload_tensor_into(st_, t, DType::BF16, a);
		return a;
	};
	img_in_w_ = bf16("img_in.weight");
	txt_in_w_ = bf16("txt_in.in_layer.weight");
	txt_out_w_ = bf16("txt_in.out_layer.weight");
	ttl1_w_ = bf16("time_text_embed.timestep_embedder.linear_1.weight");
	ttl2_w_ = bf16("time_text_embed.timestep_embedder.linear_2.weight");
	mod_w_ = bf16("modulation.1.weight");
	nout_w_ = bf16("norm_out.linear.weight");
	pout_w_ = bf16("proj_out.weight");
	// ZeroCenteredRMSNorm stores `scale - 1`; fold the +1 in here so the row norm
	// can use the engine's plain affine RMSNorm.
	{
		std::vector<float> w = bf16_host(st_, "txt_in.text_norm.weight");
		for (float& f : w) f += 1.0f;
		txt_norm_w_ = g_->kalloc((u64)w.size() * 4);
		upload_range(*g_->ctx, *g_->ring, txt_norm_w_.res, txt_norm_w_.off, w.data(),
		             w.size() * 4);
	}
	g_->ctx->submit();
	g_->ring->rewind();
	open_ = true;
}

void ImageDiT::release_resident() {
	// The prefix cache goes back with the window: it is only useful while a
	// sampling loop is running, and the VAE decode that follows wants the card.
	free_prefix_cache();
	release_resident_window();
}

void ImageDiT::release_resident_window() {
	res_arena_.release_chunks();
	res_layers_.clear();
	res_n_ = 0;
	res_bytes_ = 0;
	res_planned_ = false;
}

// `into` picks the arena the layer's tensors land in: null streams it through the
// weight arena (reused by the next block), a pointer keeps it for the run.
void ImageDiT::upload_layer(i64 i, QILayer& L, GpuArena* into) {
	const std::string p = "transformer_blocks." + std::to_string(i) + ".";
	// q/k/v are concatenated into one [3*dim, dim] int8 matrix so a single
	// `quant_gemm` produces the per-token [q | k | v] layout `qkv_prep` reads.
	// Either source precision is handled inside the helper: the shipped int8 is
	// uploaded verbatim, a float weight is decoded and requantised.
	//
	// No `loras_` is passed to either helper: a LoRA is the runtime correction
	// below, never folded. Folding would make every adapted matrix of this
	// checkpoint - 7.2 GB of int8 - be decoded, adapted and requantised on every
	// sampling step, which is the one thing the streaming loop cannot afford.
	L.q = load_quant3_linear(*g_, st_, p + "attn.to_q", p + "attn.to_k", p + "attn.to_v",
	                         into, nullptr);
	L.out = load_quant_linear(*g_, st_, p + "attn.to_out.0", into, nullptr);
	L.gate_up = load_quant_linear(*g_, st_, p + "img_mlp.gate_up", into, nullptr);
	L.mlp_out = load_quant_linear(*g_, st_, p + "img_mlp.out", into, nullptr);
	// The matching factor pairs, over the same output rows: the fused matrix's
	// rows are q then k then v, and its correction has to be too.
	if (loras_ && !loras_->empty()) {
		const i64 dim = cfg_.dim;
		// A *reference*, not a copy: `GpuArena` is a bump allocator whose whole
		// point is that it owns a chunk list and reports what it charges, and a
		// tail allocated into a copy would be invisible to both - the accountant
		// would under-count it, and the copy's teardown would look at chunks the
		// real arena is still handing out.
		GpuArena& arena = into ? *into : *g_->wa;
		L.tq = load_lora_tail(*g_, arena,
		                      {{p + "attn.to_q", 0, dim, loras_},
		                       {p + "attn.to_k", dim, dim, loras_},
		                       {p + "attn.to_v", 2 * dim, dim, loras_}},
		                      3 * dim, dim);
		L.tout = load_lora_tail(*g_, arena, p + "attn.to_out.0", loras_, dim, dim);
		L.tgate_up = load_lora_tail(*g_, arena, p + "img_mlp.gate_up", loras_, 2 * cfg_.mlp_hidden,
		                            dim);
		L.tmlp_out = load_lora_tail(*g_, arena, p + "img_mlp.out", loras_, dim, cfg_.mlp_hidden);
	}
	auto vec = [&](const std::string& n) {
		std::vector<float> v = tensor_to_f32(st_, st_.require(p + n));
		GpuAlloc a = into ? into->alloc((u64)v.size() * 4) : g_->walloc((u64)v.size() * 4);
		upload_range(*g_->ctx, *g_->ring, a.res, a.off, v.data(), v.size() * 4);
		return a;
	};
	L.norm_q = vec("attn.norm_q.weight");
	L.norm_k = vec("attn.norm_k.weight");
}

// ── residency ─────────────────────────────────────────────────────────────

std::vector<u64> ImageDiT::activation_sizes(i64 S, i64 dim, i64 mlp_hidden, i64 in_ch,
                                               i64 n_img, i64 ctx_dim, i64 n_txt,
                                               const std::vector<i64>& ref_px) {
	// **In `forward`'s order, one entry per `aalloc`.** The order is not
	// cosmetic: the activation arena is a bump allocator that books whole chunks
	// and only ever moves forward, so the same tensors in a different order cost a
	// different amount - and a list that merges two allocations into one (the two
	// `embed_latent` staging buffers, say) under-counts by up to a chunk each time
	// the merge straddles a boundary. Getting this list wrong is invisible until
	// the plan is spending 98 % of the card, and then it is a refusal in the
	// middle of the first forward.
	//
	// One entry per reference block in `ref_px` (each reference's *own* pixel
	// count, in packing order) plus the target block of `n_img`.
	std::vector<u64> v;
	// ── QIScratch::alloc ──
	v.push_back((u64)S * dim * 4);                  // hidden
	v.push_back((u64)S * dim * 4);                  // txt
	v.push_back((u64)S * dim * 4);                  // normed
	v.push_back((u64)S * 3 * dim * 4);              // qkv
	v.push_back((u64)S * dim * 2);                  // q (fp16)
	v.push_back((u64)S * dim * 2);                  // k
	v.push_back((u64)S * dim * 2);                  // v
	v.push_back((u64)S * dim * 4);                  // attn
	v.push_back((u64)S * dim * 4);                  // proj
	v.push_back((u64)S * 2 * mlp_hidden * 4);       // gate_up
	v.push_back((u64)S * mlp_hidden * 4);           // hh
	v.push_back((u64)S * dim * 4);                  // down
	v.push_back((u64)n_img * in_ch * 4);            // outp
	v.push_back((u64)S * 3 * 4);                    // ids
	v.push_back((u64)S * 4);                        // key bound
	v.push_back((u64)S * 4);                        // the cached pass's key bound
	// ── the context projection ──
	v.push_back((u64)n_txt * ctx_dim * 4);          // cin
	v.push_back((u64)n_txt * dim * 4);              // cnorm
	// ── `embed_latent`, two allocations per block (the [C, px] latent and its
	// transposed copy), the references then the target ──
	for (i64 px : ref_px) {
		if (px <= 0) continue;
		v.push_back((u64)px * in_ch * 4);
		v.push_back((u64)px * in_ch * 4);
	}
	v.push_back((u64)n_img * in_ch * 4);
	v.push_back((u64)n_img * in_ch * 4);
	// ── the modulation vectors ──
	v.push_back((u64)4 * dim * 4);                  // mod_t
	v.push_back((u64)4 * dim * 4);                  // mod_p
	// ── the int8 activation quantiser and its per-row scales ──
	v.push_back((u64)S * std::max(dim, mlp_hidden));
	v.push_back((u64)S * 4);
	v.push_back((u64)S * (u64)kLoraRankBound * 4);  // lora_h_
	// ── the final layer ──
	v.push_back((u64)dim * 4);                      // the final scale
	v.push_back((u64)n_img * in_ch * 4);            // the velocity download
	return v;
}

// The same list, charged the way the arena books it: whole chunks, not bytes.
// This is the number the plan carries when it has not measured the arena itself.
u64 ImageDiT::activation_charge(i64 S, i64 dim, i64 mlp_hidden, i64 in_ch, i64 n_img,
                                    i64 ctx_dim, i64 n_txt, const std::vector<i64>& ref_px,
                                    u64 chunk_bytes) {
	return arena_chunk_cost(
	    activation_sizes(S, dim, mlp_hidden, in_ch, n_img, ctx_dim, n_txt, ref_px),
	    chunk_bytes);
}

u64 ImageDiT::estimate_dit_activation_bytes(i64 S, i64 dim, i64 mlp_hidden, i64 in_ch,
                                                i64 n_img) {
	u64 t = 0;
	for (u64 b : activation_sizes(S, dim, mlp_hidden, in_ch, n_img, dim, 0, {})) t += b;
	return t;
}

// The sizes one block's `upload_layer` allocates, in that order. Derived from
// the checkpoint's *shapes* rather than its stored byte counts, so the number is
// the same whatever the file stores: the upload is one byte per weight plus an
// fp32 scale per row.
std::vector<u64> ImageDiT::layer_alloc_sizes(i64 i) {
	const std::string p = "transformer_blocks." + std::to_string(i) + ".";
	auto shape = [&](const std::string& n) {
		return st_.require(p + n + ".weight").shape;
	};
	auto numel = [&](const std::string& n) -> u64 {
		u64 v = 1;
		for (i64 d : shape(n)) v *= (u64)d;
		return v;
	};
	auto rows = [&](const std::string& n) -> u64 { return (u64)shape(n)[0]; };
	auto vec_bytes = [&](const std::string& n) -> u64 {
		const StTensor& t = st_.require(p + n);
		u64 v = 1;
		for (i64 d : t.shape) v *= (u64)d;
		return v * 4;
	};
	// The operand width. An int8 / packed source is one byte per weight plus one
	// fp32 scale per row; a float source costs its own element size (f16 / bf16 /
	// fp8 / nvfp4 -> 2, f32 -> 4) and carries no scale. Pricing every source at one
	// byte over-counts a float checkpoint into far too few resident blocks; pricing
	// an int8 one at its tensor bytes (not its chunks) under-counts it.
	const u64 welem_num = 8u;
	const u64 welem_den = 8u;
	const u64 scale_row = 4u;
	const u64 slop = 4096u;
	std::vector<u64> v;
	auto dense_esz = [&](const std::string& n) -> u64 {
		const StTensor& t = st_.require(p + n + ".weight");
		if (!weight_is_dense_float(st_, t)) return 0;
		return weight_precision_of(st_, t) == WeightPrecision::F32 ? 4u : 2u;
	};
	auto push_mat = [&](const char* n, u64 rows_, u64 numel_) {
		const u64 esz = dense_esz(n);
		if (esz) {
			v.push_back(numel_ * esz + slop);
		} else {
			v.push_back(numel_ * welem_num / welem_den + slop);
			v.push_back(rows_ * scale_row);
		}
	};
	// The fused q|k|v is one allocation (q rows then k then v), whatever precision.
	{
		const u64 n_qkv = numel("attn.to_q") + numel("attn.to_k") + numel("attn.to_v");
		const u64 r_qkv = rows("attn.to_q") + rows("attn.to_k") + rows("attn.to_v");
		push_mat("attn.to_q", r_qkv, n_qkv);
	}
	push_mat("attn.to_out.0", rows("attn.to_out.0"), numel("attn.to_out.0"));
	push_mat("img_mlp.gate_up", rows("img_mlp.gate_up"), numel("img_mlp.gate_up"));
	push_mat("img_mlp.out", rows("img_mlp.out"), numel("img_mlp.out"));
	v.push_back(vec_bytes("attn.norm_q.weight"));
	v.push_back(vec_bytes("attn.norm_k.weight"));
	return v;
}

void ImageDiT::plan_residency(i64 H, i64 W, i64 n_txt, const std::vector<i64>& ref_px) {
	res_planned_ = true;
	const u64 budget = vram_budget().limit();
	const u64 chunk = arena_chunk_bytes_for(budget);
	i64 ref_tok = 0;
	for (i64 px : ref_px) ref_tok += std::max<i64>(px, 0);
	const i64 S = n_txt + ref_tok + H * W;
	// The rows the cache would keep: the prefix (prompt + references).
	const i64 n_prefix_rows = n_txt + ref_tok;
	// The floor the window is planned against: enough for one streamed block plus
	// the slack below, which is what keeps a plan from spending the last of the
	// card on either the cache or the window.
	const u64 kMinWindow = (u64)cfg_.n_layers > 0 ? (1ull << 30) : 0;
	act_bytes_ = activation_charge(S, cfg_.dim, cfg_.mlp_hidden, cfg_.in_channels, H * W,
	                               cfg_.context_dim, n_txt, ref_px, chunk);

	// One block's tensors **in the order `upload_layer` allocates them**, and what
	// streaming one of them costs the accountant (whole weight-arena chunks, not
	// its own bytes). The order matters: the arena is a bump allocator, so the same
	// ten tensors cost different amounts depending on where each one lands
	// relative to a chunk boundary.
	std::vector<u64> tensors = layer_alloc_sizes(0);
	u64 layer_bytes = 0;
	for (u64 b : tensors) layer_bytes += b;
	if (layer_bytes == 0) layer_bytes = 226ull << 20;
	const u64 wa_chunk = (g_ && g_->wa) ? g_->wa->chunk_bytes() : (64ull << 20);
	u64 in_flight = layer_bytes;
	if (!tensors.empty()) {
		const u64 cost = arena_chunk_cost(tensors, wa_chunk);
		if (cost > in_flight) in_flight = cost;
	}
	// What streaming one block really costs, measured: the block is uploaded the
	// way the loop uploads it and the weight arena's capacity is read back. The
	// simulation above is close but not exact - a block's ten tensors land on
	// different chunk boundaries depending on what the arena already holds - and
	// that difference is the whole margin when the plan is spending 98 % of the
	// card.
	if (g_ && g_->wa && !tensors.empty() && g_->wa->capacity() == 0) {
		g_->ctx->begin();
		QILayer tmp;
		upload_layer(0, tmp);
		g_->ctx->submit();
		g_->ring->rewind();
		const u64 measured = g_->wa->capacity();
		if (measured > in_flight) in_flight = measured;
		// The charge goes back: nothing is live in that arena any more, and the
		// streamed blocks will recreate exactly these chunks.
		g_->wa->release_chunks();
	}
	// ── what else is resident, before this plan commits anything ──
	// Measured, not named, so it cannot go stale: the `keep` arena's tables, the
	// caller's reference latents, whatever else this process holds. Taken while
	// both the weight streamer's chunks and the activation arena are empty, so
	// that neither is counted here *and* in the terms below.
	const u64 resident_now = vram_budget().local();

	// ── the activation arena, measured the same way as the streamed block ──
	// The sizes one forward allocates, in its own order, handed to the arena the
	// forward will use. A simulation of the allocator over those sizes gets the
	// shape of the charge right but not the number - 2.81 GB against a measured
	// 2.71 GB at 1080p - and the difference is exactly what the residency target is
	// short of the budget. The chunks are *kept*: `new_step()` resets the arena
	// without dropping them, so the first step finds its scratch already committed
	// and sized.
	//
	// The measurement is allowed to fail: a sequence long enough that its own frame
	// does not fit beside the long-lived tables has no residency to plan at all,
	// and the honest answer there is the *forward's* refusal with the sequence it
	// could not hold - not an allocator error from inside the planner.
	const u64 frame_sim = activation_charge(S, cfg_.dim, cfg_.mlp_hidden, cfg_.in_channels,
	                                       H * W, cfg_.context_dim, n_txt, ref_px, chunk);
	if (frame_sim + resident_now > budget) {
		// The sequence does not fit even with nothing resident, and no plan can help:
		// the frame is `S` rows of a 4096-wide stream, at 4 bytes each. Say so with
		// the numbers the caller can act on - what this shape wants, and what it
		// would have to shrink to - instead of letting the allocator fail in the
		// middle of a forward.
		const i64 canvas_px = H * W;
		const u64 room = budget > resident_now ? budget - resident_now : 0;
		const i64 fit_px = std::max<i64>(
		    256, (i64)((double)frame_sim > 0
		                   ? (double)(S - n_txt - ref_tok) * (double)room /
		                         (double)frame_sim
		                   : canvas_px));
		char buf[384];
		snprintf(buf, sizeof buf,
					         "qwen_image: %lld tokens need %s of activation frame (%lld reference "
					         "px + %lld canvas px) beside %s of resident tables, past "
					         "this machine's %s. Reduce the canvas or the reference resolution - the "
					         "frame is linear in the pixel count, so about %lld canvas px would fit.",
					         (long long)S, format_bytes(frame_sim).c_str(), (long long)ref_tok,
					         (long long)canvas_px, format_bytes(resident_now).c_str(),
					         format_bytes(budget).c_str(), (long long)fit_px);
		throw MediaError(buf);
	}
	if (g_ && g_->aa && g_->aa->capacity() == 0 && frame_sim + resident_now <= budget) {
		for (u64 bytes : activation_sizes(S, cfg_.dim, cfg_.mlp_hidden, cfg_.in_channels, H * W,
		                                  cfg_.context_dim, n_txt, ref_px))
			g_->aa->alloc(bytes);
		act_bytes_ = g_->aa->capacity();
	}
	if (g_ && g_->aa) g_->aa->reset();

	// ── the prefix K/V cache, if it was asked for and fits ──
	//
	// Taken *before* the window is sized, because the two compete for the same
	// card and the window is what this engine would rather spend it on: the cache
	// is a speed decision (it removes the prefix from every step's arithmetic),
	// while the window is a disk decision (a block that is not resident is 380 MB
	// re-read on every step). So the question is asked in that order - frame,
	// cache, then as much window as is left - and a card that cannot afford the
	// cache simply keeps recomputing the prefix, which is what it did before the
	// node existed.
	cache_ok_ = false;
	cache_reserve_ = 0;
	if (cache_want_ && n_prefix_rows > 0 && H * W > 0) {
		const u64 need = prefix_cache_charge(n_prefix_rows, H * W, cfg_);
		const u64 resident_with = resident_now + act_bytes_ + in_flight + need + kMinWindow;
		if (need > 0 && resident_with <= budget && alloc_prefix_cache(n_prefix_rows, H * W)) {
			cache_reserve_ = need;
			cache_ok_ = true;
		} else {
			free_prefix_cache();
		}
	}

	// The margin is the difference between a *model* of the allocator and the
	// allocator: the streaming arena's chunk list shifts a little as the window
	// fills the other one, and a plan that spends the last 50 MB of the card is a
	// plan that fails on the roundings - in the middle of the sampling loop.
	const u64 kSlack = 96ull << 20;
	plan_reserve_ = act_bytes_ + in_flight + resident_now + cache_reserve_;
	u64 room = budget > plan_reserve_ + kSlack ? budget - plan_reserve_ - kSlack : 0;
	const u64 device_room = vram_budget().live_headroom();
	if (device_room < room) room = device_room;
	const u64 cap = room;

	(void)device_room;

	// ── take the window ──
	// The blocks are uploaded here rather than at their first use: the disk
	// traffic is the same, but the plan's own invariant has to hold *before* the
	// loop starts, and an upload that fails mid-step would fail after the run had
	// already spent its text encoding.
	//
	// How many, and how much they charge, is decided by the accountant rather than
	// by arithmetic on tensor sizes: an arena books whole chunks, a block's ten
	// tensors do not tile its 64 MB granularity, and where they land depends on
	// what the arena already holds - so `res_arena_.capacity()` (which *is* the
	// accountant's charge for this arena) and each block's own growth are the
	// numbers the loop uses. One chunk of headroom is kept per block so the block
	// the loop starts never overshoots the cap it was admitted under.
	release_resident_window();
	res_layers_.resize((size_t)cfg_.n_layers);
	u64 est = in_flight;
	i64 n = 0;
	g_->ctx->begin();
	if (const char* e = getenv("PHI_DIT_RES")) {
		const int v = atoi(e);
		const i64 want = (v <= 0) ? 0 : std::min<i64>(cfg_.n_layers, v);
		for (; n < want; n++) {
			if (!g_->ctx->recording()) g_->ctx->begin();
			upload_layer(n, res_layers_[(size_t)n], &res_arena_);
		}
	} else {
		while (n < cfg_.n_layers) {
			if (res_arena_.capacity() + est > cap) break;
			const u64 have = res_arena_.capacity();
			if (!g_->ctx->recording()) g_->ctx->begin();
			upload_layer(n, res_layers_[(size_t)n], &res_arena_);
			const u64 grew = res_arena_.capacity() - have;
			if (grew > est) est = grew;
			n++;
		}
	}
	g_->ctx->submit();
	g_->ring->rewind();
	res_layers_.resize((size_t)n);
	res_n_ = n;
	res_bytes_ = res_arena_.capacity();
	res_percent_ = budget ? (int)(100ull * (res_bytes_ + plan_reserve_) / budget) : 0;
	// Set last, and only here: `release_resident_window()` (just above, and again
	// from `release_resident`) clears the flag, so a plan that ends by re-taking
	// the window has to mark the phase planned *after* that. The sampler node keys
	// "have I planned for this phase?" on it, and a stale false would re-run this
	// whole function - the upload probe included - on every sampling step.
	// Kept for the runtime tuner: at plan time the weight arena has been handed back,
	// so the *streamed* block is not yet charged to the ledger the tuner reads. It is
	// exactly the reserve a correction must leave room for.
	res_inflight_ = in_flight;
	// ...and what one more *resident* block really costs this arena: the growth the
	// upload loop above actually measured, or - for a window that ended at zero -
	// the one-block cost of the arena's own granularity. The tuner grows with this
	// number, never with `window_charge_bytes(1, ...)`, which rounds a single block
	// in isolation and therefore under-states what the accountant books.
	if (est > res_block_charge_) res_block_charge_ = est;
	if (res_block_charge_ == 0) res_block_charge_ = res_layer_bytes_;
	if (getenv("PHI_DIT_PLAN"))
		fprintf(stderr,
		        "[qi] plan: budget=%s act=%s (%s of tensors) window=%lldx%s (%d%% of the budget)"
		        " reserve=%s (inflight %s, held %s, slack %s, cap %s)\n",
		        format_bytes(budget).c_str(), format_bytes(act_bytes_).c_str(),
		        format_bytes(estimate_dit_activation_bytes(S, cfg_.dim, cfg_.mlp_hidden,
		                                                  cfg_.in_channels, H * W))
		            .c_str(),
		        (long long)res_n_, format_bytes(res_bytes_).c_str(), res_percent_,
		        format_bytes(plan_reserve_).c_str(), format_bytes(in_flight).c_str(),
		        format_bytes(resident_now).c_str(), format_bytes(kSlack).c_str(),
		        format_bytes(cap).c_str());
	res_planned_ = true;
}

// ── dynamic residency (core/runtime/vram_window.hpp) ────────────────
//
// The plan above decides the window once, from the request and from the card as
// it looked before the loop. The loop then holds that decision for the whole run
// while the driver's own figure moves and while the plan's terms turn out to be
// estimates - so the window is re-decided once per step from what the process
// actually holds:
//
//   * `other` is measured, not named: the ledger minus what the resident arena is
//     *charged*, so the activation frame, the streamed block's chunks and the
//     caller's own footprint are all in it without anyone having to price them;
//   * the band is the driver's limit (see VramBudget), and the live figure is the
//     driver's own usage floored by the ledger, so a driver that has not caught
//     up yet cannot read as "the card is empty";
//   * growing appends blocks to the arena (a bump allocator, so appending is what
//     it is for); shedding hands the window back and re-takes it smaller, which is
//     the only way to *free* resident chunks on this allocator.
//
// Either way the arithmetic is untouched: the window only decides which blocks
// are re-read from the checkpoint each step. So every failure here is swallowed
// and the run keeps going with whatever window it has.

i64 ImageDiT::grow_resident(i64 want) {
	if (want <= 0 || !g_) return 0;
	const i64 before = res_n_;
	const i64 target = std::min<i64>(cfg_.n_layers, before + want);
	if (target <= before) return 0;
	if (res_layers_.size() < (size_t)cfg_.n_layers)
		res_layers_.resize((size_t)cfg_.n_layers);
	g_->ctx->begin();
	for (i64 i = before; i < target; i++) {
		try {
			upload_layer(i, res_layers_[(size_t)i], &res_arena_);
			res_n_ = i + 1;
		} catch (const std::exception& e) {
			if (getenv("PHI_DIT_PLAN"))
				fprintf(stderr, "[qi] tune: window growth stopped at %lld/%lld (%s)\n",
				        (long long)i, (long long)target, e.what());
			break;
		}
	}
	g_->ctx->submit();
	g_->ring->rewind();
	res_bytes_ = res_arena_.capacity();
	return res_n_ - before;
}

i64 ImageDiT::shrink_resident(i64 drop) {
	if (drop <= 0 || !g_) return 0;
	const i64 before = res_n_;
	const i64 want = std::max<i64>(0, before - drop);
	if (want >= before) return 0;
	// A bump arena cannot give back one block: the only way to free resident
	// chunks is to drop the window and re-take it smaller. The *window* only: the
	// prefix K/V cache is not part of it and must survive the correction, or the
	// first shrink of a run would silently turn the cache off for every step after
	// it (and leave the sampler reporting a cache it is no longer using).
	release_resident_window();
	if (want > 0) {
		res_layers_.resize((size_t)cfg_.n_layers);
		g_->ctx->begin();
		for (i64 i = 0; i < want; i++) {
			try {
				upload_layer(i, res_layers_[(size_t)i], &res_arena_);
				res_n_ = i + 1;
			} catch (const std::exception&) {
				break;
			}
		}
		g_->ctx->submit();
		g_->ring->rewind();
		res_bytes_ = res_arena_.capacity();
		res_layers_.resize((size_t)res_n_);
	}
	return before - res_n_;
}

i64 ImageDiT::tune_residency() {
	if (!open_ || !g_ || cfg_.n_layers <= 0 || res_layer_bytes_ == 0) return res_n_;
	const u64 limit = vram_budget().limit();
	if (limit == 0) return res_n_;

	const u64 window = res_arena_.capacity();
	const u64 local = vram_budget().local();
	WindowLive l;
	l.n = res_n_;
	l.n_total = cfg_.n_layers;
	l.other = local > window ? local - window : 0;
	l.block_bytes = res_layer_bytes_;
	l.chunk_bytes = res_arena_.chunk_bytes();
	l.limit = limit;
	l.used = std::max(vram_budget().live_usage(), local);

	WindowPolicy pol;
	// The headroom the *next* step needs beyond what the ledger already holds.
	//
	// The layer the loop streams is charged when this runs (`new_layer()` rewinds the
	// weight arena's bump pointer and keeps its chunks), so the reservation is the
	// streamed block's cost *minus what the arena still holds* - only the difference
	// is uncovered by the ledger. Reserving the whole charge over-reserved and left
	// the window below the band; reserving a single chunk under-reserved and the
	// loop's own streamed layer was refused mid-run (measured: "96.00 MB for
	// image.weights would take the process to 4.86 GB of the 4.85 GB ceiling").
	const u64 wa_held = (g_ && g_->wa) ? g_->wa->capacity() : 0;
	const u64 next_step_charge = res_inflight_ > wa_held ? res_inflight_ - wa_held : 0;
	pol.safety_bytes =
	    std::max<i64>((i64)res_arena_.chunk_bytes(), (i64)next_step_charge);
	// Grow by what one more block *really* costs (measured, see res_block_charge_) -
	// the difference against the granularity rounding is a whole chunk, and it is
	// exactly the margin the last correction must not eat.
	pol.block_charge = res_block_charge_ ? res_block_charge_ : res_layer_bytes_;
	if (getenv("PHI_DIT_PLAN"))
		fprintf(stderr,
		        "[qi] tune in: n=%lld used=%.2f other=%.2f window=%.2f wa=%.2f safety=%.2f "
		        "per=%.2f limit=%.2f\n",
		        (long long)l.n, (double)l.used / 1073741824.0, (double)l.other / 1073741824.0,
		        (double)window / 1073741824.0, (double)wa_held / 1073741824.0,
		        (double)pol.safety_bytes / 1073741824.0, (double)pol.block_charge / 1048576.0,
		        (double)limit / 1073741824.0);
	i64 delta = 0;
	switch (tune_window(l, pol, &delta)) {
		case WindowMove::Grow: {
			const i64 moved = grow_resident(delta);
			if (moved > 0 && getenv("PHI_DIT_PLAN"))
				fprintf(stderr, "[qi] tune: +%lld block(s) -> %lld/%lld (%d%% of the limit)\n",
				        (long long)moved, (long long)res_n_, (long long)cfg_.n_layers,
				        (int)(100ull * (res_bytes_ + l.other) / limit));
			break;
		}
		case WindowMove::Shrink: {
			const i64 moved = shrink_resident(-delta);
			if (moved > 0 && getenv("PHI_DIT_PLAN"))
				fprintf(stderr, "[qi] tune: -%lld block(s) -> %lld/%lld (%d%% of the limit)\n",
				        (long long)moved, (long long)res_n_, (long long)cfg_.n_layers,
				        (int)(100ull * (res_bytes_ + l.other) / limit));
			break;
		}
		case WindowMove::Hold: break;
	}
	res_percent_ = (int)(100ull * (res_arena_.capacity() + l.other) / limit);
	return res_n_;
}

// ── the prefix K/V cache (ComfyUI's QwenImage21Cache) ─────────────────────
//
// The buffer holds one K and one V row set per block, `n_prefix + n_img` rows
// each: the kept prefix in the head and this step's target K/V in the tail, so the
// attention reads one contiguous key range and the kernels need no second source.
// The tail is what makes the buffer a little larger than the prefix alone, and it
// is also what saves the per-step copy `torch.cat` does in the reference.
u64 ImageDiT::prefix_cache_charge(i64 n_prefix, i64 n_img, const ImageDitConfig& cfg) {
	if (n_prefix <= 0 || n_img < 0 || cfg.n_layers <= 0) return 0;
	// One k and one v row set per block (see `alloc_prefix_cache`), fp16.
	const u64 row = (u64)cfg.n_heads * (u64)cfg.head_dim * 2;
	return 2 * (u64)cfg.n_layers * (u64)(n_prefix + n_img) * row;
}

bool ImageDiT::alloc_prefix_cache(i64 n_prefix, i64 n_img) {
	free_prefix_cache();
	if (!g_ || !g_->ctx || n_prefix <= 0) return false;
	const u64 rows = (u64)(n_prefix + n_img);
	const u64 row_bytes = (u64)cfg_.n_heads * (u64)cfg_.head_dim * 2;
	// **One slice per block.** Every block has its own k/v projections, so its
	// keys and values are its own; a single shared buffer would have the block 1
	// pass overwrite what block 0 kept (and the target pass would then attend into
	// another block's keys, which is wrong in a way that still produces a
	// plausible-looking picture). The buffer is [n_layers][rows][heads*head_dim]
	// fp16 per tensor, which is exactly what `prefix_cache_charge` prices.
	const u64 slice = rows * row_bytes;
	const u64 bytes = (u64)cfg_.n_layers * slice;
	// The accountant refuses a charge that would cross the limit, and a refusal is
	// an answer ("this card cannot afford the cache"), not a failure: the run then
	// recomputes the prefix every step, which is what it did before the node
	// existed. Nothing is allocated before the charge succeeds.
	try {
		vram_budget().charge(2 * bytes, "qwen.prefix_cache");
	} catch (const std::exception&) {
		return false;
	}
	try {
		cache_k_.res = g_->ctx->create_device_buffer(bytes, true);
		cache_v_.res = g_->ctx->create_device_buffer(bytes, true);
	} catch (const std::exception&) {
		if (cache_k_.res) g_->ctx->release_buffer(cache_k_.res);
		if (cache_v_.res) g_->ctx->release_buffer(cache_v_.res);
		cache_k_ = GpuAlloc{};
		cache_v_ = GpuAlloc{};
		vram_budget().release(2 * bytes);
		return false;
	}
	cache_k_.off = 0;
	cache_v_.off = 0;
	cache_rows_ = (i64)rows;
	cache_charged_ = 2 * bytes;
	cache_ready_ = false;
	cache_key_ = nullptr;
	cache_key_rows_ = 0;
	return true;
}

void ImageDiT::free_prefix_cache() {
	if (cache_k_.res && g_ && g_->ctx) g_->ctx->release_buffer(cache_k_.res);
	if (cache_v_.res && g_ && g_->ctx) g_->ctx->release_buffer(cache_v_.res);
	cache_k_ = GpuAlloc{};
	cache_v_ = GpuAlloc{};
	if (cache_charged_) vram_budget().release(cache_charged_);
	cache_charged_ = 0;
	cache_rows_ = 0;
	cache_ready_ = false;
	cache_key_ = nullptr;
	cache_key_rows_ = 0;
}

// ── ops ───────────────────────────────────────────────────────────────────

void ImageDiT::quant_gemm(const QuantLinear& w, const GpuAlloc& x, i64 m, i64 k,
                              const GpuAlloc& out) {
	// The whole block forward goes through this one call: int8 (verbatim tensorwise
	// or a packed family) gets the quantise+rotate kernel and the int8 GEMM; a float
	// source (f16 / bf16 / f32) gets the dense fp16 GEMM with no conversion.
	linear_gemm(*g_, w, x, m, k, out, q8_, s8_);
}

const char* ImageDiT::weight_precision() const {
	if (!open_) return "unknown";
	// The block projections share one precision (they come from one checkpoint), so
	// the first block's fused q/k/v speaks for the stack.
	const StTensor* t = st_.find("transformer_blocks.0.attn.to_q.weight");
	if (!t) return "unknown";
	return weight_precision_name(weight_precision_of(st_, *t));
}

void ImageDiT::block(const QILayer& L, const GpuAlloc& hidden, const BlockPass& bp,
                         const GpuAlloc& mod_t, const GpuAlloc& mod_p, const GpuAlloc& ids,
                         const GpuAlloc& rope, QIScratch& s) {
	auto& ctx = *g_->ctx;
	const i64 dim = cfg_.dim, D = cfg_.head_dim, H = cfg_.n_heads, mlp = cfg_.mlp_hidden;
	const i64 S = bp.rows;              // rows this pass runs
	const i64 n_prefix = bp.n_prefix;   // ... of which these are prefix rows
	const i64 n_tgt = S - n_prefix;
	const i64 seg = dim * 4;
	if (S <= 0) return;

	auto slice_off = [](const GpuAlloc& a, i64 off, i64 bytes) { return sub(a, (u64)off, (u64)bytes); };
	auto mod_rows = [&](const GpuAlloc& x, const GpuAlloc& scale_vec, i64 row0, i64 rows) {
		ElemArgs e;
		e.op = ElemOp::Modulate;
		e.a = slice_off(x, row0 * dim * 4, rows * dim * 4);
		e.b = scale_vec;
		e.c = scale_vec;
		e.y = e.a;
		e.rows = rows;
		e.cols = dim;
		e.bMode = ElemMode::Row;
		dispatch_elem(ctx, e);
	};
	auto gated_add = [&](const GpuAlloc& x, const GpuAlloc& y, const GpuAlloc& gate_vec,
	                     i64 row0, i64 rows) {
		ElemArgs e;
		e.op = ElemOp::AddMul;
		e.a = slice_off(x, row0 * dim * 4, rows * dim * 4);
		e.b = slice_off(y, row0 * dim * 4, rows * dim * 4);
		e.c = gate_vec;
		e.y = e.a;
		e.rows = rows;
		e.cols = dim;
		e.cMode = ElemMode::Row;
		dispatch_elem(ctx, e);
	};
	const GpuAlloc sc1t = mod_t, gt1t = slice_off(mod_t, seg, seg), sc2t = slice_off(mod_t, seg * 2, seg),
	              gt2t = slice_off(mod_t, seg * 3, seg);
	const GpuAlloc sc1p = mod_p, gt1p = slice_off(mod_p, seg, seg), sc2p = slice_off(mod_p, seg * 2, seg),
	              gt2p = slice_off(mod_p, seg * 3, seg);

	auto row_norm = [&](const GpuAlloc& src, const GpuAlloc& dst) {
		NormArgs na;
		na.x = src;
		na.y = dst;
		na.rows = S;
		na.cols = dim;
		na.eps = cfg_.eps;
		na.affine = false;
		na.layer_norm = true;
		dispatch_norm(ctx, na);
	};

	// ── attention ──
	row_norm(hidden, s.normed);
	if (n_prefix > 0) mod_rows(s.normed, sc1p, 0, n_prefix);
	if (n_tgt > 0) mod_rows(s.normed, sc1t, n_prefix, n_tgt);
	quant_gemm(L.q, s.normed, S, dim, s.qkv);
	apply_lora_tail(*g_, L.tq, s.normed, S, s.qkv, lora_h_);
	// The K/V of this pass land at `kv_row0` of the pass's K/V buffer (the scratch's
	// for an uncached run, the prefix cache's for a cached one, where the target
	// rows are written into the tail of the buffer holding the kept prefix).
	const i64 kv_stride = H * D * 2;   // fp16, heads * head_dim per row
	GpuAlloc kv_k = sub(bp.k, (u64)bp.kv_row0 * (u64)kv_stride,
	                    (u64)S * (u64)kv_stride);
	GpuAlloc kv_v = sub(bp.v, (u64)bp.kv_row0 * (u64)kv_stride,
	                    (u64)S * (u64)kv_stride);
	{
		QkvPrepArgs qp;
		qp.qkv = s.qkv;
		qp.wq = L.norm_q;
		qp.wk = L.norm_k;
		qp.rope = rope;
		qp.ids = ids;
		qp.q = s.q;
		qp.k = kv_k;
		qp.v = kv_v;
		qp.s = S;
		qp.heads = H;
		qp.kv_heads = H;
		qp.head_dim = D;
		qp.eps = cfg_.eps;
		qp.scale = 1.0f;
		qp.norm = true;
		qp.rope_on = true;
		qp.f32_out = false;
		// The pass's rows are a slice of the sequence, so the RoPE positions come
		// from its own window of the ids table.
		qp.ids_row = bp.ids_row;
		dispatch_qkv_prep(ctx, qp);
	}
	{
		AttnFlashArgs aa;
		aa.q = s.q;
		// The keys run over the whole buffer, not just this pass's rows: that is
		// what lets the target rows attend into the kept prefix.
		aa.k = bp.k;
		aa.v = bp.v;
		aa.o = s.attn;
		aa.s = bp.kv_rows > 0 ? bp.kv_rows : S;
		aa.q_rows = S;
		aa.heads = H;
		aa.kv_heads = H;
		aa.head_dim = D;
		aa.f32_input = false;
		aa.causal = false;
		aa.ub = bp.umask;
		aa.scale = 1.0f / std::sqrt((float)D);
		dispatch_attn_flash(ctx, aa);
	}
	quant_gemm(L.out, s.attn, S, dim, s.proj);
	apply_lora_tail(*g_, L.tout, s.attn, S, s.proj, lora_h_);
	if (n_prefix > 0) gated_add(hidden, s.proj, gt1p, 0, n_prefix);
	if (n_tgt > 0) gated_add(hidden, s.proj, gt1t, n_prefix, n_tgt);
	stat(ctx, "after_attn", hidden, S * dim);

	// ── MLP (fused [gate; up] then swiglu) ──
	row_norm(hidden, s.normed);
	if (n_prefix > 0) mod_rows(s.normed, sc2p, 0, n_prefix);
	if (n_tgt > 0) mod_rows(s.normed, sc2t, n_prefix, n_tgt);
	quant_gemm(L.gate_up, s.normed, S, dim, s.gate_up);
	apply_lora_tail(*g_, L.tgate_up, s.normed, S, s.gate_up, lora_h_);
	dispatch_silu_gate_fused(ctx, s.gate_up, s.hh, S, mlp);
	quant_gemm(L.mlp_out, s.hh, S, mlp, s.down);
	apply_lora_tail(*g_, L.tmlp_out, s.hh, S, s.down, lora_h_);
	if (n_prefix > 0) gated_add(hidden, s.down, gt2p, 0, n_prefix);
	if (n_tgt > 0) gated_add(hidden, s.down, gt2t, n_prefix, n_tgt);
	stat(ctx, "after_mlp", hidden, S * dim);
}

// ── forward ───────────────────────────────────────────────────────────────

std::vector<float> ImageDiT::forward(const std::vector<float>& x, i64 H, i64 W,
                                        const std::vector<float>& context, i64 n_txt,
                                        const std::vector<std::vector<float>>& refs,
                                        const std::vector<ImageRefGeom>& ref_geom, float sigma,
                                        std::vector<i32> slots, const void* cond_key) {
	if (!open_) throw MediaError("qwen_image: not open");
	auto& ctx = *g_->ctx;
	const i64 C = cfg_.in_channels, dim = cfg_.dim, mlp = cfg_.mlp_hidden;
	const i64 n_img = H * W;
	if ((i64)x.size() != C * n_img) throw MediaError("qwen_image: latent shape mismatch");
	if ((i64)context.size() != n_txt * cfg_.context_dim)
		throw MediaError("qwen_image: context shape mismatch");

	// ── the host half: timestep MLP, modulation, final scale ──
	// Kept on the host because every tensor here is a single 4096-wide vector and
	// the weights are cheap to hold as fp32 (see open()).
	const std::vector<float> w_t1 = bf16_host(st_, "time_text_embed.timestep_embedder.linear_1.weight");
	const std::vector<float> w_t2 = bf16_host(st_, "time_text_embed.timestep_embedder.linear_2.weight");
	const std::vector<float> w_mod = bf16_host(st_, "modulation.1.weight");
	std::vector<float> w_nout = bf16_host(st_, "norm_out.linear.weight");
	auto temb_of = [&](float t) {
		std::vector<float> e = timestep_embedding(t, cfg_.t_freq);
		std::vector<float> h = silu_v(host_linear(w_t1, dim, cfg_.t_freq, e));
		return host_linear(w_t2, dim, dim, h);
	};
	auto mod_of = [&](float t) {
		std::vector<float> temb = temb_of(t);
		std::vector<float> m = host_linear(w_mod, 4 * dim, dim, silu_v(temb));
		// `_split_rows(gate.tanh())`: the gate halves are tanh'd when the
		// modulation is built, the scale halves are not.
		for (i64 i = 0; i < dim; i++) m[(size_t)(dim + i)] = std::tanh(m[(size_t)(dim + i)]);
		for (i64 i = 0; i < dim; i++)
			m[(size_t)(3 * dim + i)] = std::tanh(m[(size_t)(3 * dim + i)]);
		return m;
	};
	std::vector<float> temb_t = temb_of(sigma);
	std::vector<float> mod_t = mod_of(sigma);
	std::vector<float> mod_p = mod_of(0.0f);
	std::vector<float> scale_final = host_linear(w_nout, dim, dim, silu_v(temb_t));

	// ── the sequence layout ──
	// With reference latents the model splices each image block into the text
	// stream at the position its vision rows occupied in the template
	// (`image_slots`), which is where the `<|image_pad|>` token sat - right after
	// the block's `<|vision_start|>`, before the prompt text. A text-to-image
	// request is the degenerate case with no blocks.
	const i64 n_ref = (i64)refs.size();
	if ((i64)slots.size() != n_ref) {
		// No slots handed in: put every block in front of the text (the layout
		// this chain used before it had slots).
		slots.assign((size_t)n_ref, 0);
		i64 at = 0;
		for (i64 i = 0; i < n_ref; i++) slots[(size_t)i] = (i32)(at += 0);
	}
	for (i64 i = 0; i < n_ref; i++) {
		if (slots[(size_t)i] < 0 || slots[(size_t)i] > n_txt)
			throw MediaError("qwen_image: reference slot outside the context");
		for (i64 j = 0; j < i; j++)
			if (slots[(size_t)j] > slots[(size_t)i])
				throw MediaError("qwen_image: reference slots must be ordered");
	}
	if ((i64)ref_geom.size() != n_ref)
		throw MediaError("qwen_image: " + std::to_string(n_ref) + " reference latent(s) but " +
		                 std::to_string(ref_geom.size()) + " geometry record(s)");
	// Each reference contributes *its own* grid (ComfyUI's build_sequence reads
	// h/w off the latent it is about to splice), so the token count is a sum and
	// not `n_ref * ref_h * ref_w`.
	i64 n_ref_tok = 0;
	for (i64 i = 0; i < n_ref; i++) {
		const ImageRefGeom& rge = ref_geom[(size_t)i];
		if (rge.h <= 0 || rge.w <= 0)
			throw MediaError("qwen_image: reference " + std::to_string(i + 1) +
			                 " has an empty latent grid");
		if ((i64)refs[(size_t)i].size() != C * rge.h * rge.w)
			throw MediaError("qwen_image: reference " + std::to_string(i + 1) +
			                 " latent shape mismatch");
		n_ref_tok += rge.h * rge.w;
	}
	const i64 S = n_txt + n_ref_tok + n_img;
	const i64 n_prefix = S - n_img;
	if (n_txt > cfg_.dim * 8) throw MediaError("qwen_image: prompt unexpectedly long");

	g_->new_step();
	g_->ctx->begin();
	QIScratch s;
	s.alloc(g_, S, dim, mlp, C, n_img, n_ref_tok, 0);

	// ── context projection (txt_in) ──
	GpuAlloc cin = g_->aalloc((u64)n_txt * cfg_.context_dim * 4);
	upload_range(ctx, *g_->ring, cin.res, cin.off, context.data(), context.size() * 4);
	GpuAlloc cnorm = g_->aalloc((u64)n_txt * dim * 4);
	{
		NormArgs na;
		na.x = cin;
		na.w = txt_norm_w_;
		na.y = cnorm;
		na.rows = n_txt;
		na.cols = cfg_.context_dim;
		na.eps = cfg_.eps;
		na.affine = true;
		dispatch_norm(ctx, na);
	}
	{
		GemmF16Args ga;
		ga.a = cnorm;
		ga.b = txt_in_w_;
		ga.c = s.hidden;
		ga.m = n_txt;
		ga.n = dim;
		ga.k = cfg_.context_dim;
		ga.a_is_f32 = true;
		ga.b_is_bf16 = true;
		dispatch_gemm_f16(ctx, ga);
	}
	{
		ElemArgs e;
		e.op = ElemOp::GeluTanh;
		e.a = s.hidden;
		e.b = s.hidden;
		e.c = s.hidden;
		e.y = s.hidden;
		e.rows = 1;
		e.cols = n_txt * dim;
		dispatch_elem(ctx, e);
	}
	{
		GemmF16Args ga;
		ga.a = s.hidden;
		ga.b = txt_out_w_;
		ga.c = s.txt;
		ga.m = n_txt;
		ga.n = dim;
		ga.k = dim;
		ga.a_is_f32 = true;
		ga.b_is_bf16 = true;
		dispatch_gemm_f16(ctx, ga);
	}

	// ── latent embedder (img_in) for the references and the target ──
	auto embed_latent = [&](const std::vector<float>& lat, i64 h, i64 w, const GpuAlloc& dst) {
		const i64 p = h * w;
		GpuAlloc lin = g_->aalloc((u64)C * p * 4);
		upload_range(ctx, *g_->ring, lin.res, lin.off, lat.data(), lat.size() * 4);
		GpuAlloc lt = g_->aalloc((u64)p * C * 4);
		dispatch_transpose_cs(ctx, lin, lt, C, p, true);
		GemmF16Args ga;
		ga.a = lt;
		ga.b = img_in_w_;
		ga.c = dst;
		ga.m = p;
		ga.n = dim;
		ga.k = C;
		ga.a_is_f32 = true;
		ga.b_is_bf16 = true;
		dispatch_gemm_f16(ctx, ga);
	};

	// Assemble the sequence: the prompt's rows with each reference block spliced
	// in at its slot, and the target block last.
	{
		i64 src = 0, row = 0;
		auto place_text = [&](i64 from, i64 n) {
			if (n <= 0) return;
			ElemArgs e;
			e.op = ElemOp::Copy;
			e.a = sub(s.txt, (u64)from * dim * 4, (u64)n * dim * 4);
			e.b = e.a;
			e.c = e.a;
			e.y = sub(s.hidden, (u64)row * dim * 4, (u64)n * dim * 4);
			e.rows = n;
			e.cols = dim;
			dispatch_elem(ctx, e);
			row += n;
		};
		for (i64 i = 0; i < n_ref; i++) {
			const ImageRefGeom& rge = ref_geom[(size_t)i];
			const i64 px = rge.h * rge.w;
			place_text(src, slots[(size_t)i] - src);
			src = slots[(size_t)i];
			embed_latent(refs[(size_t)i], rge.h, rge.w,
			             sub(s.hidden, row * dim * 4, (u64)px * dim * 4));
			row += px;
		}
		place_text(src, n_txt - src);
		embed_latent(x, H, W, sub(s.hidden, row * dim * 4, (u64)n_img * dim * 4));
	}
	stat(ctx, "emb_hidden", s.hidden, S * dim);
	// ── ids, key bounds and the RoPE table ──
	{
		std::vector<float> ids((size_t)S * 3, 0.0f);
		std::vector<u32> ub((size_t)S, 0);
		i64 row = 0, pos = 0;
		auto put_img = [&](i64 h, i64 w, i64 hh_par, i64 ww_par) {
			const i64 m = h * w;
			for (i64 y = 0; y < h; y++) {
				const float yv =
				    (float)(y - (h - h / 2)) + 0.5f * (float)(h % 2 - hh_par % 2);
				for (i64 xx = 0; xx < w; xx++) {
					const i64 r = row + y * w + xx;
					ids[(size_t)r * 3] = (float)pos;
					ids[(size_t)r * 3 + 1] = yv;
					ids[(size_t)r * 3 + 2] =
					    (float)(xx - (w - w / 2)) + 0.5f * (float)(w % 2 - ww_par % 2);
				}
			}
			for (i64 i = 0; i < m; i++) ub[(size_t)(row + i)] = (u32)(row + m);
			row += m;
			pos += std::max(h, w);
		};
		// The prompt's runs carry one causal run each, all three ids equal to a
		// single counter that the reference blocks advance by `max(h, w)` - the
		// `pos` bookkeeping of comfy/ldm/qwen_image21/model.py::build_sequence.
		auto put_text = [&](i64 n) {
			for (i64 i = 0; i < n; i++) {
				const i64 r = row + i;
				const float pv = (float)(pos + i);
				ids[(size_t)r * 3] = pv;
				ids[(size_t)r * 3 + 1] = pv;
				ids[(size_t)r * 3 + 2] = pv;
				ub[(size_t)r] = (u32)(r + 1);
			}
			row += n;
			pos += n;
		};
		{
			i64 src = 0;
			for (i64 i = 0; i < n_ref; i++) {
				put_text(slots[(size_t)i] - src);
				src = slots[(size_t)i];
				// `put_img(h, w, H, W)` centres each reference's grid on the
				// target's: the half-token term `0.5 * (h % 2 - H % 2)` is the
				// reference's `hh`/`ww` (build_sequence), and the *parity it is
				// centred against* is the target's, not the first reference's.
				put_img(ref_geom[(size_t)i].h, ref_geom[(size_t)i].w, H, W);
			}
			put_text(n_txt - src);
			put_img(H, W, H, W);
		}

		upload_range(ctx, *g_->ring, s.ids.res, s.ids.off, ids.data(), ids.size() * 4);
		upload_range(ctx, *g_->ring, s.ub.res, s.ub.off, ub.data(), ub.size() * 4);
		// The cached target pass queries only the target rows, and each of them
		// attends to the whole slice (the block-causal bound of the last block): a
		// mask of its own, indexed from query row 0, is what its dispatch needs.
		{
			std::vector<u32> ub_t((size_t)std::max<i64>(S, 1), (u32)S);
			upload_range(ctx, *g_->ring, s.ub_t.res, s.ub_t.off, ub_t.data(), ub_t.size() * 4);
		}
		std::vector<float> rope;
		build_rope_table(cfg_, rope);
		s.rope = g_->kalloc((u64)rope.size() * 4);
		upload_range(ctx, *g_->ring, s.rope.res, s.rope.off, rope.data(), rope.size() * 4);
	}
	{
		GpuAlloc mt = g_->aalloc((u64)mod_t.size() * 4);
		upload_range(ctx, *g_->ring, mt.res, mt.off, mod_t.data(), mod_t.size() * 4);
		GpuAlloc mp = g_->aalloc((u64)mod_p.size() * 4);
		upload_range(ctx, *g_->ring, mp.res, mp.off, mod_p.data(), mod_p.size() * 4);
		s.mod_t = mt;
		s.mod_p = mp;
	}
	// activation quantiser scratch: one layer's worth, reused by every block
	q8_ = g_->aalloc((u64)S * std::max(dim, mlp));
	s8_ = g_->aalloc((u64)S * 4);
	// The LoRA correction's hidden form, in the same slot `activation_sizes`
	// prices it in (right after the quantiser's buffers). `kLoraRankBound` rows
	// wide so the plan's number is the one `apply_lora_tail` needs, whatever the
	// loaded chain's rank turns out to be.
	lora_h_ = g_->aalloc((u64)S * (u64)kLoraRankBound * 4);
	ctx.submit();
	g_->ring->rewind();
	ctx.begin();

	// ── the prefix K/V cache (ComfyUI's QwenImage21Cache) ──
	//
	// `cache_k_` is the plan's (see `plan_residency`): it exists when the node asked
	// for the cache and the accountant had room for it. The prefix rows are then run
	// once per conditioning, their K/V kept, and every step after that runs the
	// target rows alone - the same arithmetic on the same values, which is why the
	// cached and uncached paths agree to the last bit.
	const bool cache_ok = cache_k_.res != nullptr && cond_key != nullptr && n_prefix > 0 &&
	                      n_img > 0 && cache_rows_ == n_prefix + n_img;
	if (cache_ok && (cache_key_ != cond_key || cache_key_rows_ != cache_rows_)) {
		// A different conditioning (or canvas): the kept prefix is not this one.
		cache_ready_ = false;
		cache_key_ = cond_key;
		cache_key_rows_ = cache_rows_;
	}
	const bool run_prefix_pass = cache_ok && !cache_ready_;

	// ── the blocks ──
	for (i64 i = 0; i < cfg_.n_layers; i++) {
		g_->new_layer();
		// A block the plan kept resident is already uploaded: it is read straight
		// out of `res_arena_` and the weight arena stays free for the ones that
		// stream through it.
		QILayer tmp;
		const bool resident = i < res_n_;
		if (!resident) upload_layer(i, tmp);
		const QILayer& L = resident ? res_layers_[(size_t)i] : tmp;
		// This block's slice of the cache: its own k/v, kept for the rest of the run.
		GpuAlloc blk_k, blk_v;
		if (cache_ok) {
			const u64 slice = (u64)cache_rows_ * (u64)cfg_.n_heads * (u64)cfg_.head_dim * 2;
			blk_k = sub(cache_k_, (u64)i * slice, slice);
			blk_v = sub(cache_v_, (u64)i * slice, slice);
		}
		if (run_prefix_pass) {
			// The prefix rows: t = 0 modulation, block-causal among themselves, and
			// their K/V land in this block's slice, where they stay for the run.
			BlockPass bp;
			bp.rows = n_prefix;
			bp.n_prefix = n_prefix;
			bp.k = blk_k;
			bp.v = blk_v;
			bp.kv_row0 = 0;
			bp.kv_rows = n_prefix;
			bp.ids_row = 0;
			bp.umask = s.ub;   // rows [0, n_prefix) of the sequence's own bounds
			block(L, s.hidden, bp, s.mod_t, s.mod_p, s.ids, s.rope, s);
		}
		{
			// The target rows. Uncached they are the tail of the same sequence the
			// prefix is part of (one pass over S); cached they are their own pass,
			// attending into the kept prefix (rows [0, n_prefix) of the block's
			// slice) with their own fresh K/V in that slice's tail.
			BlockPass bp;
			GpuAlloc hidden = sub(s.hidden, (u64)n_prefix * dim * 4, (u64)n_img * dim * 4);
			if (cache_ok) {
				bp.rows = n_img;
				bp.n_prefix = 0;
				bp.k = blk_k;
				bp.v = blk_v;
				bp.kv_row0 = n_prefix;
				bp.kv_rows = n_prefix + n_img;
				bp.ids_row = n_prefix;
				bp.umask = s.ub_t;
			} else {
				bp.rows = S;
				bp.n_prefix = n_prefix;
				bp.k = s.k;
				bp.v = s.v;
				bp.kv_row0 = 0;
				bp.kv_rows = S;
				bp.ids_row = 0;
				bp.umask = s.ub;
				hidden = s.hidden;
			}
			block(L, hidden, bp, s.mod_t, s.mod_p, s.ids, s.rope, s);
		}
		g_->end_layer();
	}
	if (run_prefix_pass) cache_ready_ = true;

	// ── final layer: the target rows only ──
	g_->ctx->begin();   // end_layer() closed the previous bracket
	GpuAlloc tgt = sub(s.hidden, (u64)n_prefix * dim * 4, (u64)n_img * dim * 4);
	GpuAlloc fscale = g_->aalloc((u64)dim * 4);
	upload_range(ctx, *g_->ring, fscale.res, fscale.off, scale_final.data(), dim * 4);
	{
		NormArgs na;
		na.x = tgt;
		na.y = s.normed;
		na.rows = n_img;
		na.cols = dim;
		na.eps = cfg_.eps;
		na.affine = false;
		na.layer_norm = true;
		dispatch_norm(ctx, na);
	}
	{
		ElemArgs e;
		e.op = ElemOp::Modulate;
		e.a = s.normed;
		e.b = fscale;
		e.c = fscale;
		e.y = s.normed;
		e.rows = n_img;
		e.cols = dim;
		e.bMode = ElemMode::Row;
		dispatch_elem(ctx, e);
	}
	{
		GemmF16Args ga;
		ga.a = s.normed;
		ga.b = pout_w_;
		ga.c = s.outp;
		ga.m = n_img;
		ga.n = C;
		ga.k = dim;
		ga.a_is_f32 = true;
		ga.b_is_bf16 = true;
		dispatch_gemm_f16(ctx, ga);
	}
	GpuAlloc lat = g_->aalloc((u64)C * n_img * 4);
	dispatch_transpose_cs(ctx, s.outp, lat, C, n_img, false);
	ctx.submit();
	std::vector<float> out((size_t)C * n_img);
	ctx.download(lat.res, lat.off, out.data(), out.size() * 4);
	g_->ring->rewind();
	return out;
}

}  // namespace phi::media
