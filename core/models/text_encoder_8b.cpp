#include "models/text_encoder_8b.hpp"

#include <cmath>
#include <cstring>

#include "host/quant.hpp"
#include "kernels/gpu_ops.hpp"
#include "runtime/vram_budget.hpp"
#include "runtime/vram_window.hpp"

namespace phi::media {

void TextEncoder8B::open(const std::string& path, GpuCtx* gpu) {
	g_ = gpu;
	path_ = path;
	st_.open(path);

	const StTensor& emb = st_.require("model.embed_tokens.weight");
	if (emb.shape.size() != 2) throw MediaError("qwen3vl: unexpected embedding shape");
	cfg_.vocab = emb.shape[0];
	cfg_.hidden = emb.shape[1];

	i64 l = 0;
	while (st_.find("model.layers." + std::to_string(l) + ".input_layernorm.weight")) l++;
	cfg_.n_layers = l;
	if (cfg_.n_layers <= 0) throw MediaError("qwen3vl: no layers in the checkpoint");

	const StTensor& qp = st_.require("model.layers.0.self_attn.q_proj.weight");
	const StTensor& kp = st_.require("model.layers.0.self_attn.k_proj.weight");
	cfg_.head_dim = st_.require("model.layers.0.self_attn.q_norm.weight").shape[0];
	cfg_.n_heads = qp.shape[0] / cfg_.head_dim;
	cfg_.n_kv_heads = kp.shape[0] / cfg_.head_dim;
	cfg_.intermediate = st_.require("model.layers.0.mlp.gate_proj.weight").shape[0];
	if (qp.shape[1] != cfg_.hidden)
		throw MediaError("qwen3vl: q_proj input width does not match the embedding width");
	// The tower may arrive as the shipped int8 tensorwise checkpoint or as any
	// f16 / bf16 tensorwise quantisation of the same weights; load_quant_linear()
	// normalises all of them onto the engine's int8 GEMM path, so no branch is
	// needed here beyond refusing a file that is neither.
	if (!(qp.dtype == DType::I8 || qp.dtype == DType::F16 || qp.dtype == DType::BF16 ||
	      qp.dtype == DType::F32 || qp.dtype == DType::F8_E4M3 || qp.dtype == DType::F8_E5M2))
		throw MediaError("qwen3vl: unsupported weight precision for q_proj");
	if (cfg_.out_layer < 0 || cfg_.out_layer >= cfg_.n_layers) cfg_.out_layer = cfg_.n_layers - 1;
}

void TextEncoder8B::upload_layer(i64 index, Layer& L, GpuArena* into) {
	const std::string p = "model.layers." + std::to_string(index) + ".";
	auto quant = [&](const std::string& n) {
		// Int8 shipped weights upload verbatim; float weights are decoded and
		// requantised, and a LoRA touching the module is folded in before that.
		return load_quant_linear(*g_, st_, p + n, into, loras_);
	};
	auto vec = [&](const std::string& n) {
		std::vector<float> v = tensor_to_f32(st_, st_.require(p + n));
		GpuArena& arena = into ? *into : *g_->wa;
		GpuAlloc a = arena.alloc((u64)v.size() * 4);
		upload_range(*g_->ctx, *g_->ring, a.res, a.off, v.data(), v.size() * 4);
		return a;
	};
	L.q = quant("self_attn.q_proj");
	L.k = quant("self_attn.k_proj");
	L.v = quant("self_attn.v_proj");
	L.o = quant("self_attn.o_proj");
	L.gate = quant("mlp.gate_proj");
	L.up = quant("mlp.up_proj");
	L.down = quant("mlp.down_proj");
	L.in_ln = vec("input_layernorm.weight");
	L.post_ln = vec("post_attention_layernorm.weight");
	L.q_norm = vec("self_attn.q_norm.weight");
	L.k_norm = vec("self_attn.k_norm.weight");
}

// ── the resident layer window ────────────────────────────────────────────────
//
// The same formula as the 32B tower's (see that file for the derivation): the
// window is the largest number of layers that fits inside
//
//   limit - (held + one_layer + slack)
//
// where `limit` is what the system grants this process, `held` is the ledger
// *after* the activation frame exists, `one_layer` is the arena growth this
// loader measured for a layer, and the charge is booked in arena chunks. The
// layer's own precision enters only through that measurement.
u64 TextEncoder8B::layer_bytes_estimate(i64 i) const {
	const std::string p = "model.layers." + std::to_string(i) + ".";
	u64 total = 0;
	auto add = [&](const std::string& base) {
		const StTensor& t = st_.require(base + ".weight");
		const i64 n = t.shape[0];
		const i64 K = st_.logical_cols(t);
		// `load_quant_linear`: a float source keeps its own dtype and runs the dense
		// GEMM; everything else lands on the engine's int8 tensorwise pair (codes +
		// one fp32 scale per row), whether it arrived int8 already or as a packed
		// family that the slab requantiser expanded.
		const u64 bytes = weight_is_dense_float(st_, t)
		                      ? (u64)n * (u64)K *
		                            (weight_precision_of(st_, t) == WeightPrecision::F32 ? 4ull : 2ull)
		                      : (u64)n * (u64)K + (u64)n * 4;
		total += (bytes + 255) & ~(u64)255;
	};
	add(p + "self_attn.q_proj");
	add(p + "self_attn.k_proj");
	add(p + "self_attn.v_proj");
	add(p + "self_attn.o_proj");
	add(p + "mlp.gate_proj");
	add(p + "mlp.up_proj");
	add(p + "mlp.down_proj");
	// The four per-layer norms (two hidden-wide, two head-wide) as fp32 rows.
	total += 2ull * (u64)cfg_.hidden * 4ull + 2ull * (u64)cfg_.head_dim * 4ull;
	return total;
}

void TextEncoder8B::release_resident() {
	res_arena_.release_chunks();
	res_layers_.clear();
	res_n_ = 0;
	res_bytes_ = 0;
	res_planned_ = false;
}

void TextEncoder8B::plan_residency(i64 S) {
	(void)S;   // the frame is already charged: this runs after the activations exist
	release_resident();
	res_planned_ = true;
	const i64 n_total = cfg_.out_layer + 1;
	if (!g_ || !g_->ctx || n_total <= 0) return;
	const u64 budget = vram_budget().limit();
	const u64 one_layer = layer_bytes_estimate(0);
	const u64 chunk = budget ? arena_chunk_bytes_for(budget) : (64ull << 20);
	res_arena_.init(g_->ctx, chunk);
	res_arena_.set_tag("qwen3vl8.res");
	res_layers_.resize((size_t)n_total);
	if (budget == 0) {
		// No device answered: stream every layer, exactly as before this window
		// existed. (The arena keeps no chunks, so nothing is charged.)
		res_layers_.clear();
		return;
	}
	if (const char* e = getenv("PHI_TE_RES")) {
		const int v = atoi(e);
		const i64 want = (v <= 0) ? 0 : std::min<i64>(n_total, v);
		if (want > 0) {
			g_->ctx->begin();
			for (i64 i = 0; i < want; i++) {
				if (!g_->ctx->recording()) g_->ctx->begin();
				upload_layer(i, res_layers_[(size_t)i], &res_arena_);
			}
			g_->ctx->submit();
			g_->ring->rewind();
			res_n_ = want;
			res_bytes_ = res_arena_.capacity();
		} else {
			res_arena_.release_chunks();
		}
		return;
	}
	const u64 held = vram_budget().local();
	const u64 slack = 96ull << 20;
	const u64 other = held + one_layer + slack;
	u64 room = budget > other ? budget - other : 0;
	const u64 device_room = vram_budget().live_headroom();
	if (device_room < room) room = device_room;
	if (room == 0) {
		// Nothing fits beside the frame this prompt needs; stream, as before.
		res_layers_.clear();
		return;
	}
	u64 est = one_layer;
	i64 n = 0;
	g_->ctx->begin();
	while (n < n_total) {
		if (res_arena_.capacity() + est > room) break;
		const u64 have = res_arena_.capacity();
		if (!g_->ctx->recording()) g_->ctx->begin();
		upload_layer(n, res_layers_[(size_t)n], &res_arena_);
		const u64 grew = res_arena_.capacity() - have;
		if (grew > est) est = grew;
		n++;
	}
	g_->ctx->submit();
	g_->ring->rewind();
	res_layers_.resize((size_t)n);
	res_n_ = n;
	res_bytes_ = res_arena_.capacity();
	if (n == 0) res_arena_.release_chunks();
	if (getenv("PHI_TE_PLAN")) {
		const u64 charged = held + res_bytes_ + one_layer + slack;
		fprintf(stderr,
		        "[te8] window %lld/%lld layers, %s resident (one layer %s); the process is "
		        "planned to use %s of %s (%d%%), leaving %s for the streamed layer\n",
		        (long long)res_n_, (long long)n_total, format_bytes(res_bytes_).c_str(),
		        format_bytes(res_n_ ? res_bytes_ / (u64)res_n_ : one_layer).c_str(),
		        format_bytes(charged).c_str(), format_bytes(budget).c_str(),
		        budget ? (int)(100ull * charged / budget) : 0, format_bytes(one_layer).c_str());
	}
}

std::vector<float> TextEncoder8B::encode(const std::vector<i32>& ids) {
	const i64 S = (i64)ids.size();
	if (S <= 0) throw MediaError("qwen3vl: empty prompt");
	const i64 H = cfg_.hidden;
	const i64 HQ = cfg_.n_heads * cfg_.head_dim;
	const i64 HK = cfg_.n_kv_heads * cfg_.head_dim;
	const i64 inter = cfg_.intermediate;

	// ── embedding gather (host side: one row per token, times its row scale) ──
	//
	// The table is int8 + convrot, exactly like every projection weight. A Linear
	// absorbs that rotation by rotating *its* input; a gather has no input to
	// rotate, so the stored rows have to be de-rotated here. ComfyUI does the
	// same thing for this format - `int8_tensorwise` is the one case where its
	// Embedding does not call F.embedding at all:
	//
	//     if self.quant_format == "int8_tensorwise":
	//         x = get_layout_class(self.layout_type).dequantize_embedding(qdata, params, input)
	//
	// (comfy/ops.py, MixedPrecisionOps.Embedding, "int8: per-row scale possible
	// ConvRot, so let the layout do the gather"). The Hadamard is an involution
	// (H^2 = I), so the inverse is the same convrot the quantiser applied.
	// Rotating the gathered rows once is what makes them the model's embeddings:
	// with the raw rows the tower's own tied lm_head predicts 5.13 nats of
	// entropy and nonsense tokens, with the de-rotated rows it predicts "Here" /
	// "**" / "Certainly" at 1.01 nats.
	std::vector<float> xf((size_t)S * H);
	{
		const StTensor& t = st_.require("model.embed_tokens.weight");
		const bool quant_like = t.dtype == DType::I8 || t.dtype == DType::U8 ||
		                        t.dtype == DType::I4 || t.dtype == DType::F8_E4M3 ||
		                        t.dtype == DType::F8_E5M2;
		if (quant_like && !st_.is_plain_int8(t)) {
			// Any other quantised family (a codebook, six-bit codes, nvfp4, a grouped
			// int8, fp8): one row per token through the reader, which is where the
			// family's codebook / group-scale tables live, then the same de-rotation.
			for (i64 s = 0; s < S; s++) {
				const i32 id = ids[(size_t)s];
				if (id < 0 || id >= cfg_.vocab) throw MediaError("qwen3vl: token id out of range");
				st_.dequant_rows(t, id, 1, &xf[(size_t)s * H]);
			}
			QuantSpec spec;
			if (const StTensor* q = st_.find("model.embed_tokens.comfy_quant")) {
				spec = parse_quant_spec(st_.data_of(*q), (size_t)q->nbytes);
			}
			if (spec.convrot) {
				std::vector<float> un((size_t)S * H);
				convrot_inverse(xf.data(), un.data(), S, H, spec.convrot_groupsize);
				xf.swap(un);
			}
		} else if (t.dtype == DType::I8) {
			// Shipped int8 + per-row scale + convrot(256): gather, scale, de-rotate.
			const StTensor& sc = st_.require("model.embed_tokens.weight_scale");
			std::vector<float> scales = tensor_to_f32(st_, sc);
			if ((i64)scales.size() != cfg_.vocab)
				throw MediaError("qwen3vl: embedding scale table does not match the vocabulary");
			const signed char* base = (const signed char*)st_.data_of(t);
			for (i64 s = 0; s < S; s++) {
				const i32 id = ids[(size_t)s];
				if (id < 0 || id >= cfg_.vocab) throw MediaError("qwen3vl: token id out of range");
				const signed char* row = base + (u64)id * H;
				const float scv = scales[(size_t)id];
				float* dst = &xf[(size_t)s * H];
				for (i64 i = 0; i < H; i++) dst[i] = (float)row[i] * scv;
			}
			QuantSpec spec;
			if (const StTensor* q = st_.find("model.embed_tokens.comfy_quant")) {
				spec = parse_quant_spec(st_.data_of(*q), (size_t)q->nbytes);
			}
			if (spec.convrot) {
				std::vector<float> un((size_t)S * H);
				convrot_inverse(xf.data(), un.data(), S, H, spec.convrot_groupsize);
				xf.swap(un);
			}
		} else {
			// f16 / bf16 / f32 table: gather with a plain conversion.
			const u64 row_bytes = (u64)H * dtype_size(t.dtype);
			const u8* base = (const u8*)st_.data_of(t);
			for (i64 s = 0; s < S; s++) {
				const i32 id = ids[(size_t)s];
				if (id < 0 || id >= cfg_.vocab) throw MediaError("qwen3vl: token id out of range");
				convert_to_f32(t.dtype, base + (u64)id * row_bytes, &xf[(size_t)s * H], (size_t)H);
			}
		}
	}

	g_->new_step();
	g_->ctx->begin();
	GpuAlloc x = g_->aalloc((u64)S * H * 4);
	upload_range(*g_->ctx, *g_->ring, x.res, x.off, xf.data(), (u64)xf.size() * 4);
	g_->ctx->submit();
	g_->ring->rewind();

	GpuAlloc t = g_->aalloc((u64)S * H * 4);
	GpuAlloc q = g_->aalloc((u64)S * HQ * 4);
	GpuAlloc k = g_->aalloc((u64)S * HK * 4);
	GpuAlloc v = g_->aalloc((u64)S * HK * 4);
	GpuAlloc qn = g_->aalloc((u64)S * HQ * 4);
	GpuAlloc kn = g_->aalloc((u64)S * HK * 4);
	GpuAlloc attn = g_->aalloc((u64)S * HQ * 4);
	GpuAlloc gate = g_->aalloc((u64)S * inter * 4);
	GpuAlloc up = g_->aalloc((u64)S * inter * 4);
	GpuAlloc hh = g_->aalloc((u64)S * inter * 4);
	GpuAlloc down = g_->aalloc((u64)S * H * 4);
	const i64 maxk = std::max(H, inter);
	GpuAlloc q8 = g_->aalloc((u64)S * maxk);
	GpuAlloc s8 = g_->aalloc((u64)S * 4);

	// The window is planned *after* the frame exists: `VramBudget::local()` is then
	// the honest "everything else", and what is left of the system's limit - less
	// one streamed layer and the margin - is what the window may spend.
	plan_residency(S);

	// One projection, through whichever path the weight's own precision implies:
	// int8 (verbatim tensorwise, or a packed family) rotates + quantises the
	// activation and runs the int8 GEMM; a float source (f16 / bf16 / f32) runs the
	// dense fp16 GEMM with no conversion.
	auto linear = [&](const QuantLinear& w, const GpuAlloc& in, i64 m, i64 kk,
	                  const GpuAlloc& out) {
		linear_gemm(*g_, w, in, m, kk, out, q8, s8);
	};
	auto rms = [&](const GpuAlloc& in, i64 rows, i64 cols, const GpuAlloc& w,
	               const GpuAlloc& out) {
		NormArgs na;
		na.x = in;
		na.w = w;
		na.y = out;
		na.rows = rows;
		na.cols = cols;
		na.eps = cfg_.eps;
		na.affine = true;
		dispatch_norm(*g_->ctx, na);
	};
	auto add = [&](const GpuAlloc& a, const GpuAlloc& b, i64 rows, i64 cols) {
		ElemArgs ea;
		ea.op = ElemOp::Add;
		ea.a = a;
		ea.b = b;
		ea.y = a;
		ea.rows = rows;
		ea.cols = cols;
		dispatch_elem(*g_->ctx, ea);
	};

	const float scale = 1.0f / std::sqrt((float)cfg_.head_dim);
	for (i64 li = 0; li <= cfg_.out_layer; li++) {
		g_->new_layer();
		Layer streamed;
		const bool resident = li < res_n_;
		if (!resident) upload_layer(li, streamed, nullptr);
		// A resident layer is used in place (see `plan_residency`).
		const Layer& L = resident ? res_layers_[(size_t)li] : streamed;

		// ── attention ──
		rms(x, S, H, L.in_ln, t);
		linear(L.q, t, S, H, q);
		linear(L.k, t, S, H, k);
		linear(L.v, t, S, H, v);
		rms(q, S * cfg_.n_heads, cfg_.head_dim, L.q_norm, qn);
		rms(k, S * cfg_.n_kv_heads, cfg_.head_dim, L.k_norm, kn);
		{
			RopeHalfArgs ra;
			ra.head_dim = cfg_.head_dim;
			ra.theta = cfg_.rope_theta;
			ra.rows = S * cfg_.n_heads;
			ra.heads = cfg_.n_heads;
			ra.x = qn;
			ra.y = q;
			dispatch_rope_half(*g_->ctx, qn.res, ra);
			ra.rows = S * cfg_.n_kv_heads;
			ra.heads = cfg_.n_kv_heads;
			ra.x = kn;
			ra.y = k;
			dispatch_rope_half(*g_->ctx, kn.res, ra);
		}
		AttnFlashArgs aa;
		aa.q = q;
		aa.k = k;
		aa.v = v;
		aa.o = attn;
		aa.s = S;
		aa.heads = cfg_.n_heads;
		aa.kv_heads = cfg_.n_kv_heads;
		aa.head_dim = cfg_.head_dim;
		aa.f32_input = true;
		aa.causal = true;
		aa.scale = scale;
		dispatch_attn_flash(*g_->ctx, aa);

		linear(L.o, attn, S, HQ, t);
		add(x, t, S, H);

		// ── MLP ──
		rms(x, S, H, L.post_ln, t);
		linear(L.gate, t, S, H, gate);
		linear(L.up, t, S, H, up);
		{
			ElemArgs ea;
			ea.op = ElemOp::SiluGate;
			ea.a = gate;
			ea.b = up;
			ea.y = hh;
			ea.rows = S;
			ea.cols = inter;
			dispatch_elem(*g_->ctx, ea);
		}
		linear(L.down, hh, S, inter, down);
		add(x, down, S, H);
		g_->end_layer();
	}

	std::vector<float> out = g_->download_f32(x, (u64)S * H);
	// Hand the window back: the DiT phase follows and plans against the ledger, so a
	// window still held here would be subtracted from its budget instead.
	release_resident();
	return out;
}

}  // namespace phi::media
