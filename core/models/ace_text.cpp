// ACE-Step 1.5 text side: the Qwen3-0.6B embedder tower, the Qwen3-4B audio-code
// LM, and the shared stack runner the DiT bundle's four small towers use.
//
// See the header for the precision contract: the weights are uploaded as the
// checkpoint stores them and the GEMM reads them at that precision. Nothing on
// this path runs on the CPU except the embedding gather (one row per token) and
// the sampling arithmetic over the code logits.
//
// The three things this file adds on top of the `text_encoder_8b.cpp` recipe:
//
//   * a **query window**: the LM decodes one token at a time, so the attention is
//     dispatched with `q_rows = 1` over a key range that grows by one row per
//     step. The fp32 `attn_flash` kernel takes `S` for both the query rows and
//     the key length and therefore cannot express that at all; the *tiled fp16*
//     kernel can, and it is also the only one that reads `AttnFlashArgs::ub`.
//     Since `ub` is a hard requirement of the code below (every dispatch passes
//     one), a machine forced onto the fp32 kernel gets a MediaError instead of
//     silently-numbered noise - `dispatch_attn_flash` refuses a mask without the
//     tiled kernel for exactly this reason.
//   * a **KV cache** in the `keep` arena, fp16 (what the tiled kernel reads),
//     appended one row per decode step.
//   * a **multi-pass stack**: the LM's two classifier-free-guidance branches are
//     run through each layer while that layer's weights are resident, so the
//     8.4 GB of bf16 weights are read once per decoded token instead of once per
//     branch. See `ace_stack_multi`.
//   * a **resident window** on top of that: reading them once *per token* is still
//     300 x 8.4 GB on a 60 s request, so `AceLm::plan_residency` keeps as many of
//     the 36 layers on the card as fit beside the K/V cache and the activations and
//     streams only the rest. See the residency comment at `plan_residency`.
#include "models/ace_text.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "kernels/gpu_ops.hpp"
#include "kernels/kernels.hpp"            // attn_query_tile()
#include "models/video_vae_internal.hpp"  // dispatch_f32_to_f16
#include "runtime/sched.hpp"              // arena_chunk_cost / arena_chunk_bytes_for
#include "runtime/vram_budget.hpp"
#include "runtime/vram_window.hpp"        // window_charge_bytes

namespace phi::media {

namespace {

// ── releasing the resident window on the way out of `generate` ─────────────
//
// `AceLm::generate` has several exit paths (the empty-prompt and zero-token
// returns, the decode loop's own break) and the loop body can throw - a refused
// allocation, a cancelled run, a launch failure. Every one of them has to hand the
// window back *before* it leaves, not because the LM needs the card again but
// because the DiT's `plan_residency` runs as soon as the codes are sampled and
// plans against `vram_budget().local()`: a window still held here would be
// subtracted from the DiT's budget and buy resident DiT blocks for the LM instead,
// which is the "too few layers stay resident" report this whole change is about.
//
// A guard, rather than a release at each return, is what makes that hold on the
// throwing paths too.
struct ResidentGuard {
	AceLm* lm = nullptr;
	explicit ResidentGuard(AceLm* l) : lm(l) {}
	~ResidentGuard() {
		if (!lm) return;
		// A destructor must not throw, and this one runs while an exception may be
		// unwinding (a refused allocation inside the decode loop is the ordinary
		// case for that). The hand-back is pure accounting - the arena's chunks are
		// the context's and the next plan re-takes them - so a failure here has
		// nowhere to be reported and nothing to undo.
		try {
			lm->release_resident();
		} catch (...) {
		}
	}
	ResidentGuard(const ResidentGuard&) = delete;
	ResidentGuard& operator=(const ResidentGuard&) = delete;
};

// The reference's own vocabulary layout: the audio codes live in
// [151669, 215669) (64000 codes = FSQ 8*8*8*5*5*5,
// `comfy/ldm/ace/ace_step15.py::FSQ`), and 151645 is the Qwen "end of turn" id
// the reference scores as the end-of-stream token.
constexpr i32 kQwenAudioStart = 151669;
constexpr i32 kQwenAudioEnd = 215669;
constexpr i32 kQwenEos = 151645;

GpuAlloc sub(const GpuAlloc& a, u64 bytes) {
	GpuAlloc r = a;
	r.off += bytes;
	r.bytes = r.bytes > bytes ? r.bytes - bytes : 0;
	return r;
}

// Upload `n` u32s into the activations arena, opening its own command-list
// bracket when the caller has none.
GpuAlloc upload_u32(GpuCtx& g, const u32* src, i64 n) {
	GpuAlloc a = g.aalloc((u64)n * 4);
	const bool own = !g.ctx->recording();
	if (own) g.ctx->begin();
	upload_range(*g.ctx, *g.ring, a.res, a.off, src, (u64)n * 4);
	if (own) {
		g.ctx->submit();
		g.ring->rewind();
	}
	return a;
}

// How `dispatch_gemm_f16` has to read a stored weight: bf16 is the checkpoint's
// own format, f16 and f32 are accepted so a hand-converted file is not silently
// misread as something else.
struct GemmOperand {
	bool b_is_bf16 = true;
	bool b_is_f32 = false;
};

GemmOperand gemm_operand_for(DType t) {
	switch (t) {
		case DType::BF16: return GemmOperand{true, false};
		case DType::F16: return GemmOperand{false, false};
		case DType::F32: return GemmOperand{false, true};
		default: break;
	}
	throw MediaError("ace: a weight is neither bf16, f16 nor f32");
}

// One sequence's view of a pass through the stack: where its hidden state lives,
// which KV-cache row it writes and how many keys it may read.
struct AcePass {
	GpuAlloc* hidden = nullptr;
	i64 rows = 0;
	i64 kv_row0 = 0;
	i64 kv_len = 0;
	AceKvCache* kv = nullptr;
	GpuAlloc ub;
	AceStackScratch* sc = nullptr;
};

// Runs the stack once for *every* pass, one layer at a time: the layer's weights
// are uploaded once and used by all of them. That is what makes the LM's two
// CFG branches cost one set of weights instead of two - the 4B checkpoint is
// 8.4 GB of bf16, and re-uploading it per branch would double that traffic for
// every decoded token.
//
// `res`/`res_arena`/`res_n` are the LM's resident window (`AceLm::plan_residency`):
// layers `[0, res_n)` are already in `res_arena` and are used straight out of it,
// every other layer streams through the weight arena the way the towers' one-shot
// pass does. The defaults - no window - are what `ace_qwen3_forward` keeps passing,
// so the DiT bundle's four small towers are unaffected.
void ace_stack_multi(GpuCtx& g, const SafeTensors& st, const std::string& prefix,
                     const AceQwen3Config& cfg, i64 n_layers, bool final_norm,
                     const std::string& norm_name, i64 group, AcePass* passes, i64 n_pass,
                     GpuArena* res_arena = nullptr, const std::vector<AceQwen3Layer>* res = nullptr,
                     i64 res_n = 0) {
	const i64 H = cfg.hidden;
	const i64 HQ = cfg.n_heads * cfg.head_dim;
	const i64 HK = cfg.n_kv_heads * cfg.head_dim;
	const i64 I = cfg.intermediate;
	const float scale = 1.0f / std::sqrt((float)cfg.head_dim);

	for (i64 li = 0; li < n_layers; li++) {
		// A layer the plan kept is already on the device, so nothing is streamed for
		// it and the streaming arena has nothing of its own to give back: the command
		// list is opened directly (`submit_if_recording` + `begin`, which is what
		// `new_layer()` does) instead of through `new_layer()`, which would also rewind
		// the weight arena and the shared staging ring for a block that uses neither.
		// The resident block's uploads happened once, in the plan, into `res_arena_` -
		// this loop must not touch that arena (`end_layer` below submits the block's
		// dispatches and the next pass reuses the same weights).
		AceQwen3Layer streamed;
		const bool resident = res && res_arena && li < res_n;
		const AceQwen3Layer& L = resident ? (*res)[(size_t)li] : streamed;
		if (resident) {
			g.ctx->submit_if_recording();
			g.ctx->begin();
		} else {
			g.new_layer();
			ace_qwen3_upload_layer(g, st, prefix, li, &streamed);
		}
		if (li == 0) {
			// The upload is where a wrong checkpoint would surface, and it is
			// silent otherwise: a mis-shaped weight would be read as whatever the
			// GEMM's strides say it is.
			if (L.q.k != H || L.q.n != HQ || L.k.n != HK || L.v.n != HK || L.o.k != HQ ||
			    L.o.n != H || L.gate.n != I || L.up.n != I || L.down.k != I || L.down.n != H)
				throw MediaError("ace: a layer's weight shapes do not match the detected config");
			for (i64 p = 0; p < n_pass; p++)
				if (passes[p].kv &&
				    (passes[p].kv_row0 < 0 ||
				     passes[p].kv_row0 + passes[p].rows > passes[p].kv->capacity))
					throw MediaError("ace: KV cache overrun");
		}

		for (i64 p = 0; p < n_pass; p++) {
			AcePass& P = passes[p];
			AceStackScratch& sc = *P.sc;
			const i64 S = P.rows;
			GpuAlloc& hidden = *P.hidden;

			// ── attention ──
			ace_rms_norm(g, hidden, S, H, L.in_ln, sc.t, cfg.eps);
			ace_linear(g, L.q, sc.t, S, sc.qr, GpuAlloc{}, &sc.qs);
			ace_linear(g, L.k, sc.t, S, sc.kr, GpuAlloc{}, &sc.qs);
			ace_linear(g, L.v, sc.t, S, sc.vr, GpuAlloc{}, &sc.qs);
			ace_rms_norm(g, sc.qr, S * cfg.n_heads, cfg.head_dim, L.q_norm, sc.qn, cfg.eps);
			ace_rms_norm(g, sc.kr, S * cfg.n_kv_heads, cfg.head_dim, L.k_norm, sc.kn, cfg.eps);
			{
				// Absolute positions `kv_row0 .. kv_row0 + S - 1`: a decode step is
				// a one-row pass whose row sits at the current sequence length, and
				// `RopeHalfArgs::base` is exactly that offset. The half-split ("HF
				// rotate_half") convention is the one ACE's RotaryEmbedding builds
				// - `emb = cat((freqs, freqs))` - and `rope_half` is that
				// convention, unlike `qkv_prep`'s interleaved pairs.
				//
				// A grouped pass restarts the positions per window (see the
				// header).
				auto rope_one = [&](const GpuAlloc& x, const GpuAlloc& y, i64 rows, i64 heads) {
					RopeHalfArgs ra;
					ra.head_dim = cfg.head_dim;
					ra.theta = cfg.rope_theta;
					ra.base = 0;
					ra.rows = rows;
					ra.heads = heads;
					ra.x = x;
					ra.y = y;
					dispatch_rope_half(*g.ctx, x.res, ra);
				};
				if (group <= 0) {
					RopeHalfArgs ra;
					ra.head_dim = cfg.head_dim;
					ra.theta = cfg.rope_theta;
					ra.base = P.kv_row0;
					ra.rows = S * cfg.n_heads;
					ra.heads = cfg.n_heads;
					ra.x = sc.qn;
					ra.y = sc.qr;
					dispatch_rope_half(*g.ctx, sc.qn.res, ra);
					ra.rows = S * cfg.n_kv_heads;
					ra.heads = cfg.n_kv_heads;
					ra.x = sc.kn;
					ra.y = sc.kr;
					dispatch_rope_half(*g.ctx, sc.kn.res, ra);
				} else {
					const i64 nwin = S / group;
					if (S % group != 0)
						throw MediaError("ace: grouped pass with a ragged window");
					for (i64 wi = 0; wi < nwin; wi++) {
						const u64 oq = (u64)wi * (u64)group * (u64)cfg.n_heads * (u64)cfg.head_dim;
						const u64 ok =
						    (u64)wi * (u64)group * (u64)cfg.n_kv_heads * (u64)cfg.head_dim;
						rope_one(sub(sc.qn, oq * 4), sub(sc.qr, oq * 4), group * cfg.n_heads,
						         cfg.n_heads);
						rope_one(sub(sc.kn, ok * 4), sub(sc.kr, ok * 4),
						         group * cfg.n_kv_heads, cfg.n_kv_heads);
					}
				}
			}
			// fp16 staging: q into a scratch row block (the tiled kernel reads
			// packed halves), k/v into the cache at this pass's rows.
			dispatch_f32_to_f16(*g.ctx, sc.qr, sc.qh, S * HQ);
			GpuAlloc kdst =
			    P.kv ? sub(P.kv->k[(size_t)li], (u64)P.kv_row0 * (u64)HK * 2) : sc.kh;
			GpuAlloc vdst =
			    P.kv ? sub(P.kv->v[(size_t)li], (u64)P.kv_row0 * (u64)HK * 2) : sc.vh;
			dispatch_f32_to_f16(*g.ctx, sc.kr, kdst, S * HK);
			dispatch_f32_to_f16(*g.ctx, sc.vr, vdst, S * HK);
			{
				auto attn_one = [&](const GpuAlloc& qs, const GpuAlloc& ks, const GpuAlloc& vs,
				                    const GpuAlloc& os, i64 nrows, i64 keys) {
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
					aa.f32_input = false;   // fp16 operands -> the tiled kernel
					aa.causal = false;      // the mask is `ub`
					aa.ub = P.ub;
					aa.scale = scale;
					dispatch_attn_flash(*g.ctx, aa);
				};
				if (group <= 0) {
					attn_one(sc.qh, P.kv ? P.kv->k[(size_t)li] : sc.kh,
					         P.kv ? P.kv->v[(size_t)li] : sc.vh, sc.attn, S,
					         P.kv_len > 0 ? P.kv_len : S);
				} else {
					const i64 nwin = S / group;
					for (i64 wi = 0; wi < nwin; wi++) {
						const u64 oq = (u64)wi * (u64)group * (u64)HQ;
						const u64 ok = (u64)wi * (u64)group * (u64)HK;
						attn_one(sub(sc.qh, oq * 2), sub(sc.kh, ok * 2), sub(sc.vh, ok * 2),
						         sub(sc.attn, oq * 4), group, group);
					}
				}
			}
			ace_linear(g, L.o, sc.attn, S, sc.proj, GpuAlloc{}, &sc.qs);
			ace_add_inplace(g, hidden, sc.proj, S, H);

			// ── MLP ──
			ace_rms_norm(g, hidden, S, H, L.post_ln, sc.t, cfg.eps);
			ace_linear(g, L.gate, sc.t, S, sc.gate, GpuAlloc{}, &sc.qs);
			ace_linear(g, L.up, sc.t, S, sc.up, GpuAlloc{}, &sc.qs);
			{
				ElemArgs ea;
				ea.op = ElemOp::SiluGate;
				ea.a = sc.gate;
				ea.b = sc.up;
				ea.y = sc.hh;
				ea.rows = S;
				ea.cols = I;
				dispatch_elem(*g.ctx, ea);
			}
			ace_linear(g, L.down, sc.hh, S, sc.down, GpuAlloc{}, &sc.qs);
			ace_add_inplace(g, hidden, sc.down, S, H);
		}
		g.end_layer();
	}

	if (final_norm && !norm_name.empty()) {
		// The norm lives outside the layer prefix. The loop above closed its last
		// bracket already (every `end_layer` is a submit), so this opens its own
		// rather than uploading into a command list that no longer exists.
		std::vector<float> w = tensor_to_f32(st, st.require(norm_name));
		g.ctx->begin();
		GpuAlloc nw = g.walloc((u64)w.size() * 4);
		upload_range(*g.ctx, *g.ring, nw.res, nw.off, w.data(), w.size() * 4);
		for (i64 p = 0; p < n_pass; p++)
			ace_rms_norm(g, *passes[p].hidden, passes[p].rows, H, nw, *passes[p].hidden, cfg.eps);
		g.ctx->submit();
		g.ring->rewind();
	}
}

// The reference's `_sample_audio_token`, on the host.
//
// `use_eos` is the reference's `use_eos_score`: when it is false the categorical
// is the `n_codes` code logits alone (`eos_token_id=None` in the reference, in
// which case it never concatenates an eos entry and never rewrites index 0).
// That branch is not cosmetic - see `generate` for why the reference disables it
// for the first `min_tokens` steps, and why that is the whole length of a
// text-to-music run.
//
// When it is true the eos score rides at index 0 and the code logits at
// [1, 1 + n_codes); the function returns the *raw* token id, or -1 for the eos
// branch.
i32 sample_audio(const std::vector<float>& cond, const std::vector<float>& uncond, float eos_cond,
                 float eos_uncond, bool use_eos, bool use_cfg, float cfg, float temperature,
                 float top_p, i64 top_k, float min_p, std::mt19937_64& rng, i32 audio_start) {
	const i64 n = (i64)cond.size();
	const i64 N = use_eos ? n + 1 : n;
	const i64 base = use_eos ? 1 : 0;
	const float neg_inf = -3.0e38f;
	std::vector<float> lg((size_t)N, neg_inf);
	for (i64 i = 0; i < n; i++) {
		const float u = (use_cfg && (i64)uncond.size() == n) ? uncond[(size_t)i] : 0.0f;
		lg[(size_t)(i + base)] = use_cfg ? (u + cfg * (cond[(size_t)i] - u)) : cond[(size_t)i];
	}
	if (use_eos) lg[0] = use_cfg ? (eos_uncond + cfg * (eos_cond - eos_uncond)) : eos_cond;

	// top-k: keep the k largest logits.
	if (top_k > 0 && top_k < N) {
		std::vector<float> s = lg;
		std::nth_element(s.begin(), s.begin() + (size_t)(N - top_k), s.end());
		const float thr = s[(size_t)(N - top_k)];
		for (i64 i = 0; i < N; i++)
			if (lg[(size_t)i] < thr) lg[(size_t)i] = neg_inf;
	}
	auto softmax = [&](std::vector<double>& p) {
		float mx = neg_inf;
		for (float v : lg) mx = std::max(mx, v);
		double sum = 0.0;
		for (i64 i = 0; i < N; i++) {
			p[(size_t)i] = std::exp((double)lg[(size_t)i] - (double)mx);
			sum += p[(size_t)i];
		}
		if (sum <= 0.0) {
			p.assign((size_t)N, 0.0);
			return;
		}
		for (i64 i = 0; i < N; i++) p[(size_t)i] /= sum;
	};
	// min-p: drop everything below min_p times the peak *probability*.
	if (min_p > 0.0f) {
		std::vector<double> p((size_t)N);
		softmax(p);
		double mx = 0.0;
		for (double v : p) mx = std::max(mx, v);
		for (i64 i = 0; i < N; i++)
			if (p[(size_t)i] < (double)min_p * mx) lg[(size_t)i] = neg_inf;
	}
	// top-p (nucleus): sort, then cut where the cumulative mass passes top_p.
	if (top_p < 1.0f) {
		std::vector<i32> idx((size_t)N);
		for (i64 i = 0; i < N; i++) idx[(size_t)i] = (i32)i;
		std::sort(idx.begin(), idx.end(),
		          [&](i32 a, i32 b) { return lg[(size_t)a] > lg[(size_t)b]; });
		std::vector<double> p((size_t)N);
		softmax(p);
		double cum = 0.0;
		for (i64 i = 0; i < N; i++) {
			const i32 j = idx[(size_t)i];
			cum += p[(size_t)j];
			if (i > 0 && cum > (double)top_p) lg[(size_t)j] = neg_inf;
		}
	}

	auto to_token = [&](i64 i) -> i32 {
		return use_eos ? (i == 0 ? -1 : audio_start + (i32)i - 1) : audio_start + (i32)i;
	};
	if (temperature > 0.0f) {
		for (i64 i = 0; i < N; i++) lg[(size_t)i] /= temperature;
		std::vector<double> p((size_t)N);
		softmax(p);
		std::uniform_real_distribution<double> u(0.0, 1.0);
		double r = u(rng);
		double cum = 0.0;
		for (i64 i = 0; i < N; i++) {
			cum += p[(size_t)i];
			if (r <= cum) return to_token(i);
		}
		return to_token(N - 1);
	}
	i64 best = 0;
	for (i64 i = 1; i < N; i++)
		if (lg[(size_t)i] > lg[(size_t)best]) best = i;
	return to_token(best);
}

}  // namespace

GpuAlloc ace_upload_u32(GpuCtx& g, const u32* src, i64 n) { return upload_u32(g, src, n); }

// ── linear layers at the checkpoint's precision ───────────────────────────

void AceQScratch::need(GpuCtx& g, i64 rows, i64 k) {
	if (rows <= 0 || k <= 0) return;
	const u64 want_q = (u64)rows * (u64)k;
	const u64 want_s = (u64)rows * 4;
	if (want_q > q8_cap) {
		q8 = g.aalloc(want_q);
		q8_cap = want_q;
	}
	if (want_s > s8_cap) {
		s8 = g.aalloc(want_s);
		s8_cap = want_s;
	}
}

AceLinear ace_upload_linear(GpuCtx& g, const SafeTensors& st, const std::string& base,
                            GpuArena* into) {
	const StTensor& t = st.require(base + ".weight");
	if (t.shape.size() != 2)
		throw MediaError("ace: '" + base + ".weight' is not a 2-D matrix");

	AceLinear L;
	L.n = t.shape[0];
	L.k = t.shape[1];

	// A quantised source goes onto the engine's int8 tensorwise + convrot bus - the
	// same pair every other chain runs - rather than being rescaled into a float.
	// `upload_quant_linear_i8` is that path: a plain int8 weight is uploaded
	// verbatim, a packed family (w4a8 / w6a8 / grouped int8) is expanded on device,
	// and a float source is requantised - but this branch only sees the first two,
	// because a float is routed below.
	//
	// The one thing that cannot be expressed this way is a K that is not a multiple
	// of the convrot group (256): the rotation has no grouping there, and the I8
	// GEMM's K tiles are 32 wide. `upload_quant_linear_i8` refuses with the shape
	// rather than approximating, which is the same behaviour the image DiT's loader
	// has.
	if (!weight_is_dense_float(st, t)) {
		GpuArena& arena = into ? *into : *g.wa;
		I8Upload up = upload_quant_linear_i8(g, st, base, arena, nullptr);
		L.w = up.w;
		L.s = up.s;
		L.n = up.n;
		L.k = up.k;
		L.i8 = true;
		L.dtype = DType::I8;
		return L;
	}

	// A float source: the file's own dtype for f16 / bf16 / f32, and fp16 for fp8
	// and nvfp4 (exact - both have fewer significand bits than fp16).
	const DType as = t.dtype == DType::F32    ? DType::F32
	                 : t.dtype == DType::BF16 ? DType::BF16
	                                          : DType::F16;
	L.dtype = as;
	const u64 bytes = (u64)t.numel * dtype_size(as);
	L.w = into ? into->alloc(bytes) : g.walloc(bytes);
	if (st.is_raw(t) && t.dtype == as) {
		// Verbatim: the file's bytes go straight into the ring and out to VRAM, so
		// there is nothing for the CPU to compute and nothing to round.
		g.upload_file_into(L.w, st, t);
	} else {
		// fp8 / nvfp4 decode to f16, exactly.
		std::vector<u8> b = st.materialize(t, as);
		if (b.size() != bytes)
			throw MediaError("ace: '" + base + ".weight' decoded to " + std::to_string(b.size()) +
			                 " bytes, expected " + std::to_string(bytes));
		g.upload_into(L.w, b.data(), b.size());
	}
	return L;
}

GpuAlloc ace_upload_vector(GpuCtx& g, const SafeTensors& st, const std::string& name,
                           GpuArena* into) {
	std::vector<float> v = tensor_to_f32(st, st.require(name));
	GpuAlloc a = into ? into->alloc((u64)v.size() * 4) : g.walloc((u64)v.size() * 4);
	upload_range(*g.ctx, *g.ring, a.res, a.off, v.data(), (u64)v.size() * 4);
	return a;
}

GpuAlloc ace_upload_vector_opt(GpuCtx& g, const SafeTensors& st, const std::string& name,
                               GpuArena* into) {
	if (!st.find(name)) return GpuAlloc{};
	return ace_upload_vector(g, st, name, into);
}

void ace_linear(GpuCtx& g, const AceLinear& w, const GpuAlloc& x, i64 m, const GpuAlloc& y,
                const GpuAlloc& bias, AceQScratch* qs) {
	if (w.n <= 0 || w.k <= 0) throw MediaError("ace: an empty linear layer");
	if (w.is_int8()) {
		// The engine's int8 path, the one every DiT / text tower runs: rotate and
		// quantise the fp32 activation with `quant_convrot`, then the int8 tensor-core
		// GEMM against the (already rotated) codes, scaled per row on both sides.
		if (qs == nullptr || !qs->ready())
			throw MediaError("ace: a quantised weight needs the int8 activation scratch "
			                 "(rows " + std::to_string(m) + ", K " + std::to_string(w.k) +
			                 "); the caller must pass an AceQScratch sized for this pass");
		qs->need(g, m, w.k);
		if (qs->q8_cap < (u64)m * (u64)w.k)
			throw MediaError("ace: the int8 activation scratch is too small for this shape");
		QuantConvrotArgs qa;
		qa.x = x.res;
		qa.x_offset = x.off;
		qa.q = qs->q8.res;
		qa.q_offset = qs->q8.off;
		qa.s = qs->s8.res;
		qa.s_offset = qs->s8.off;
		qa.rows = m;
		qa.K = w.k;
		qa.qmax = 127.0f;
		dispatch_quant_convrot(*g.ctx, qa);
		Int8GemmArgs ga;
		ga.a = qs->q8.res;
		ga.a_offset = qs->q8.off;
		ga.b = w.w.res;
		ga.b_offset = w.w.off;
		ga.sa = qs->s8.res;
		ga.sa_offset = qs->s8.off;
		ga.sb = w.s.res;
		ga.sb_offset = w.s.off;
		ga.c = y.res;
		ga.c_offset = y.off;
		ga.M = m;
		ga.N = w.n;
		ga.K = w.k;
		// The int8 kernel's epilogue takes an fp32 bias, so a biased layer is still
		// one dispatch (the DiTs have none, the ACE stack's conditioning heads do).
		if (bias.res != nullptr) {
			ga.bias = bias.res;
			ga.bias_offset = bias.off;
		}
		dispatch_int8_gemm(*g.ctx, ga);
		return;
	}
	const GemmOperand op = gemm_operand_for(w.dtype);
	GemmF16Args ga;
	ga.a = x;            // fp32 activations
	ga.b = w.w;          // the weight, at the precision the file stores it in
	ga.c = y;
	ga.m = m;
	ga.n = w.n;
	ga.k = w.k;
	ga.a_is_f32 = true;
	ga.b_is_bf16 = op.b_is_bf16;
	ga.b_is_f32 = op.b_is_f32;
	ga.has_bias = (bias.res != nullptr);
	if (ga.has_bias) ga.bias = bias;
	dispatch_gemm_f16(*g.ctx, ga);
}

// ── norms and the plain add ───────────────────────────────────────────────

void ace_rms_norm(GpuCtx& g, const GpuAlloc& in, i64 rows, i64 cols, const GpuAlloc& w,
                  const GpuAlloc& out, float eps) {
	NormArgs na;
	na.x = in;
	na.w = w;
	na.y = out;
	na.rows = rows;
	na.cols = cols;
	na.eps = eps;
	na.affine = true;
	dispatch_norm(*g.ctx, na);
}

void ace_add_inplace(GpuCtx& g, const GpuAlloc& acc, const GpuAlloc& b, i64 rows, i64 cols) {
	ElemArgs ea;
	ea.op = ElemOp::Add;
	ea.a = acc;
	ea.b = b;
	ea.y = acc;
	ea.rows = rows;
	ea.cols = cols;
	dispatch_elem(*g.ctx, ea);
}

void AceStackScratch::alloc(GpuCtx& g, i64 rows, const AceQwen3Config& cfg) {
	const i64 H = cfg.hidden;
	const i64 HQ = cfg.n_heads * cfg.head_dim;
	const i64 HK = cfg.n_kv_heads * cfg.head_dim;
	const i64 I = cfg.intermediate;
	auto A = [&](i64 n) { return g.aalloc((u64)n * 4); };
	t = A(rows * H);
	qr = A(rows * HQ);
	kr = A(rows * HK);
	vr = A(rows * HK);
	qn = A(rows * HQ);
	kn = A(rows * HK);
	qh = g.aalloc((u64)rows * (u64)HQ * 2);   // fp16 (the tiled kernel's operands)
	kh = g.aalloc((u64)rows * (u64)HK * 2);
	vh = g.aalloc((u64)rows * (u64)HK * 2);
	attn = A(rows * HQ);
	proj = A(rows * H);
	gate = A(rows * I);
	up = A(rows * I);
	hh = A(rows * I);
	down = A(rows * H);
	// The int8 activation scratch, for a quantised checkpoint's layers. `k` of a
	// projection here is either the hidden width (q/k/v/gate/up) or the intermediate
	// one (o/down), so max(H, I) covers every layer of this stack.
	qs.need(g, rows, std::max(H, I));
}

void AceKvCache::init(GpuCtx& g, i64 n_layers, i64 cap, i64 dim) {
	layers = n_layers;
	capacity = cap;
	kv_dim = dim;
	len = 0;
	k.assign((size_t)n_layers, GpuAlloc{});
	v.assign((size_t)n_layers, GpuAlloc{});
	for (i64 i = 0; i < n_layers; i++) {
		k[(size_t)i] = g.kalloc((u64)cap * (u64)dim * 2);
		v[(size_t)i] = g.kalloc((u64)cap * (u64)dim * 2);
	}
}

// ── config probing / layer upload ─────────────────────────────────────────

void ace_qwen3_probe(const SafeTensors& st, const std::string& prefix, AceQwen3Config* cfg) {
	const StTensor& qp = st.require(prefix + "layers.0.self_attn.q_proj.weight");
	if (qp.shape.size() != 2) throw MediaError("ace: q_proj is not a matrix");
	cfg->hidden = qp.shape[1];
	cfg->head_dim = st.require(prefix + "layers.0.self_attn.q_norm.weight").shape[0];
	if (cfg->head_dim != 128)
		throw MediaError("ace: head_dim " + std::to_string(cfg->head_dim) + " != 128");
	cfg->n_heads = qp.shape[0] / cfg->head_dim;
	cfg->n_kv_heads =
	    st.require(prefix + "layers.0.self_attn.k_proj.weight").shape[0] / cfg->head_dim;
	if ((cfg->n_heads % cfg->n_kv_heads) != 0)
		throw MediaError("ace: heads are not a multiple of kv heads");
	cfg->intermediate = st.require(prefix + "layers.0.mlp.gate_proj.weight").shape[0];
	i64 l = 0;
	while (st.find(prefix + "layers." + std::to_string(l) + ".input_layernorm.weight")) l++;
	cfg->n_layers = l;
	if (cfg->n_layers <= 0) throw MediaError("ace: no layers under '" + prefix + "'");
	if (const StTensor* e = st.find(prefix + "embed_tokens.weight")) cfg->vocab = e->shape[0];
}

void ace_qwen3_upload_layer(GpuCtx& g, const SafeTensors& st, const std::string& prefix, i64 index,
                            AceQwen3Layer* L, GpuArena* into) {
	const std::string p = prefix + "layers." + std::to_string(index) + ".";
	// The checkpoint's own bytes: no int8 requantisation, so nothing here runs on
	// the CPU (see the header). `into` is forwarded unchanged, the way
	// `AceDiT::upload_layer` forwards it: null streams the layer through the weight
	// arena, a pointer keeps it (the LM's resident window).
	L->q = ace_upload_linear(g, st, p + "self_attn.q_proj", into);
	L->k = ace_upload_linear(g, st, p + "self_attn.k_proj", into);
	L->v = ace_upload_linear(g, st, p + "self_attn.v_proj", into);
	L->o = ace_upload_linear(g, st, p + "self_attn.o_proj", into);
	L->gate = ace_upload_linear(g, st, p + "mlp.gate_proj", into);
	L->up = ace_upload_linear(g, st, p + "mlp.up_proj", into);
	L->down = ace_upload_linear(g, st, p + "mlp.down_proj", into);
	L->in_ln = ace_upload_vector(g, st, p + "input_layernorm.weight", into);
	L->post_ln = ace_upload_vector(g, st, p + "post_attention_layernorm.weight", into);
	L->q_norm = ace_upload_vector(g, st, p + "self_attn.q_norm.weight", into);
	L->k_norm = ace_upload_vector(g, st, p + "self_attn.k_norm.weight", into);
}

// ── the stack runner ──────────────────────────────────────────────────────

void ace_qwen3_forward(GpuCtx& g, const SafeTensors& st, const std::string& prefix,
                       const AceQwen3Config& cfg, i64 n_layers, bool final_norm,
                       const std::string& norm_name, const GpuAlloc& hidden, i64 S,
                       AceStackScratch& sc, AceKvCache* kv, i64 kv_row0, i64 kv_len,
                       const GpuAlloc& ub, i64 group) {
	AcePass P;
	P.hidden = const_cast<GpuAlloc*>(&hidden);
	P.rows = S;
	P.kv_row0 = kv_row0;
	P.kv_len = kv_len;
	P.kv = kv;
	P.ub = ub;
	P.sc = &sc;
	ace_stack_multi(g, st, prefix, cfg, n_layers, final_norm, norm_name, group, &P, 1,
	                /*res_arena=*/nullptr, /*res=*/nullptr, /*res_n=*/0);
}

// ── host-side embedding gather ────────────────────────────────────────────

void ace_gather_rows(const SafeTensors& st, const std::string& name, i64 hidden, const i32* ids,
                     i64 S, float* dst) {
	const StTensor& t = st.require(name);
	if (t.shape.size() != 2 || t.shape[1] != hidden)
		throw MediaError("ace: '" + name + "' is not [vocab, " + std::to_string(hidden) + "]");
	const i64 vocab = t.shape[0];
	if (t.dtype != DType::BF16 && t.dtype != DType::F16 && t.dtype != DType::F32)
		throw MediaError("ace: '" + name + "' has an unsupported dtype for a gather");
	const u64 row_bytes = (u64)hidden * dtype_size(t.dtype);
	const u8* base = (const u8*)st.data_of(t);
	for (i64 s = 0; s < S; s++) {
		const i32 id = ids[(size_t)s];
		if (id < 0 || (i64)id >= vocab)
			throw MediaError("ace: token id " + std::to_string(id) + " out of range");
		convert_to_f32(t.dtype, base + (u64)id * row_bytes, dst + (size_t)s * (size_t)hidden,
		               (size_t)hidden);
	}
}

// ── AceQwen3 ──────────────────────────────────────────────────────────────

AceQwen3::AceQwen3() = default;
AceQwen3::~AceQwen3() = default;

void AceQwen3::open(const std::string& path, GpuCtx* gpu) {
	g_ = gpu;
	st_.open(path);
	ace_qwen3_probe(st_, "model.", &cfg_);
	if (cfg_.vocab <= 0) throw MediaError("ace: '" + path + "' has no embedding table");
}

const AceQwen3Config& AceQwen3::config() const { return cfg_; }

std::vector<float> AceQwen3::encode(const std::vector<i32>& ids, i64 out_layer) {
	if (!g_ || !g_->ok()) throw MediaError("ace: the embedder tower is not open");
	const i64 S = (i64)ids.size();
	if (S <= 0) throw MediaError("ace: empty prompt");
	const i64 H = cfg_.hidden;

	std::vector<float> xf((size_t)S * (size_t)H);
	ace_gather_rows(st_, "model.embed_tokens.weight", H, ids.data(), S, xf.data());

	// `out_layer` is a `hidden_states` index, which is what the reference's
	// `{"layer": [...]}` *list* form names: the list form captures the state
	// *entering* the selected block (llama.py appends `x.unsqueeze(1)` at the top
	// of the block loop), so index 0 is the tower's embedding output and no block
	// runs at all. `ACE15TEModel` asks for exactly that one (`{"layer": [0]}`)
	// for the lyrics conditioning, and takes it without the final norm
	// (`layer_norm_hidden_state=False`). Index k >= 1 is the state after k blocks
	// and -1 the last block followed by the final norm.
	if (out_layer == 0) return xf;

	g_->new_step();
	GpuAlloc hidden = g_->aalloc((u64)S * (u64)H * 4);
	g_->ctx->begin();
	upload_range(*g_->ctx, *g_->ring, hidden.res, hidden.off, xf.data(), xf.size() * 4);
	g_->ctx->submit();
	g_->ring->rewind();

	// Causal: row r sees keys [0, r]. `ub` is per query row, which is exactly the
	// staircase the reference's `triu(1)` mask builds.
	std::vector<u32> ubv((size_t)S);
	for (i64 r = 0; r < S; r++) ubv[(size_t)r] = (u32)(r + 1);
	GpuAlloc ub = upload_u32(*g_, ubv.data(), S);

	AceStackScratch sc;
	sc.alloc(*g_, S, cfg_);
	const i64 nblk =
	    out_layer < 0 ? cfg_.n_layers : std::min<i64>(out_layer, cfg_.n_layers);
	// `layer 0` of the lyrics prompt is taken *without* the final norm
	// (ace15.py sets {"layer": [0]} and layer_norm_hidden_state stays False).
	const bool final_norm = out_layer < 0;
	ace_qwen3_forward(*g_, st_, "model.", cfg_, nblk, final_norm, "model.norm.weight", hidden, S,
	                  sc, nullptr, 0, S, ub);
	return g_->download_f32(hidden, (u64)S * (u64)H);
}

// ── AceLm ─────────────────────────────────────────────────────────────────

AceLm::AceLm() = default;
AceLm::~AceLm() = default;

void AceLm::open(const std::string& path, GpuCtx* gpu) {
	g_ = gpu;
	st_.open(path);
	ace_qwen3_probe(st_, "model.", &cfg_);
	if (cfg_.vocab <= 0) throw MediaError("ace: '" + path + "' has no embedding table");

	// The window `plan_residency()` fills. Its chunk granularity is derived from
	// the budget the same way the weight arena's is, so what the plan measures and
	// what the accountant books are the same number (`ImageDiT::open` is the same
	// decision, at the same place).
	if (g_ && g_->ctx) {
		res_arena_.init(g_->ctx, arena_chunk_bytes_for(vram_budget().limit()));
		res_arena_.set_tag("ace.lm_resident");
	}

	// One layer's own tensor bytes, summed from the header rather than taken from
	// the file's stored totals, the way `AceDiT::probe` prices a block: the plan
	// needs what a layer *weighs*, and rounds it to the resident arena's chunks
	// itself (it measures the upload's real growth as well - see `plan_residency`).
	{
		const std::string p = "model.layers.0.";
		static const char* kMats[] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
		                              "self_attn.o_proj",   "mlp.gate_proj",    "mlp.up_proj",
		                              "mlp.down_proj"};
		u64 b = 0;
		for (const char* n : kMats) {
			const StTensor& t = st_.require(p + n + ".weight");
			b += (u64)t.numel * dtype_size(t.dtype);
		}
		// The four RMSNorm vectors (input/post = hidden, q/k_norm = head_dim) are
		// widened to fp32 on upload (`ace_upload_vector`).
		b += (u64)(2 * cfg_.hidden + 2 * cfg_.head_dim) * 4;
		res_layer_bytes_ = b;
	}
}

i64 AceLm::vocab() const { return cfg_.vocab; }

// ── the resident window ───────────────────────────────────────────────────
//
// One `generate()` call is hundreds of decode passes (`ceil(duration) * 5` tokens -
// 300 on a 60 s request) and every pass runs all 36 layers of the 8.4 GB
// checkpoint, once per CFG branch. Streaming every layer on every pass is 300 x
// 8.4 GB of PCIe traffic and the same volume of checkpoint re-reads; a layer that
// stays on the card costs neither. So the plan keeps as many of them as fit
// beside everything else for the duration of the call and streams only the rest -
// the same decision `ImageDiT`/`AceDiT` take for their blocks.
//
// The decision is taken once, from **measurements** rather than arithmetic on
// tensor sizes, because none of the three terms is derivable from the shapes:
//
//   * `held` = `vram_budget().local()`, read with the weight arena empty: whatever
//     this process already holds (the embedder tower's leftover activation chunks,
//     the tied-head slice of an earlier call, another chain's tables). Measuring it
//     is what keeps the plan honest when it is not zero.
//
//   * `inflight`: one streamed layer's charge to the **weight** arena, read back
//     from the arena after uploading layer 0. The header's tensor sum
//     (`res_layer_bytes_`, 192.5 MB here) is not that charge: an arena books whole
//     chunks - 80 MB on the reference card - and a layer's seven matrices do not
//     tile them (q/o are 20 MB, gate/up/down 47.5 MB each, so gate and up cannot
//     share a chunk). The passes commit the chunk-rounded number, so that is what
//     the plan reserves.
//
//   * `act`: the frame the decode holds for the whole call. The fp16 K/V cache
//     (`n_layers x 2 x cap x HK x 2` per branch), the per-pass activation scratch
//     (the pass's `hid` plus the `rows_alloc`-row `AceStackScratch`, in `aa`), and
//     the tied-head slice the sampler scores with (64000 x 2560 bf16 = 312 MB,
//     taken on the first decode step and held to the end, in `keep`). Both parts
//     are priced with `arena_chunk_cost` in the granularity of the arena that will
//     take them, and only for the part that arena does not already hold - its
//     existing chunks are already counted in `held` - which is the same correction
//     `AceDiT::plan_residency` calls `act_extra`.
//
// The window is taken here, before the prefill, rather than at each layer's first
// use: the disk traffic is the same, but the invariant has to hold *before* the
// loop starts, and a layer that failed to upload mid-token would fail after the
// call had already spent its prompt prefill.
void AceLm::plan_residency() {
	if (!g_ || !g_->ok() || res_planned_ || cfg_.n_layers <= 0) return;
	const u64 budget = vram_budget().limit();
	if (budget == 0) return;

	const i64 H = cfg_.hidden;
	const i64 HK = cfg_.n_kv_heads * cfg_.head_dim;
	const u64 chunk = arena_chunk_bytes_for(budget);
	// The granularity today's budget implies: the tools re-derive the limit from the
	// driver on every call, so the number `open()` sized this arena with may be old.
	res_arena_.set_chunk_bytes(chunk);

	// ── what streaming one layer books ──
	u64 inflight = window_charge_bytes(1, res_layer_bytes_, g_->wa ? g_->wa->chunk_bytes() : chunk);
	if (g_->wa) {
		g_->wa->release_chunks();
		g_->ctx->begin();
		bool measured = false;
		try {
			AceQwen3Layer tmp;
			ace_qwen3_upload_layer(*g_, st_, "model.", 0, &tmp, nullptr);
			measured = true;
		} catch (const std::exception& e) {
			// A failed measurement costs precision, never the run: the derived figure
			// above stays and the plan is merely more conservative.
			if (getenv("PHI_ACE_LM_PLAN"))
				fprintf(stderr, "[acelm] plan: one streamed layer could not be measured (%s)\n",
				        e.what());
		}
		g_->ctx->submit();
		g_->ring->rewind();
		const u64 got = measured ? g_->wa->capacity() : 0;
		if (got > inflight) inflight = got;
		// Handed straight back: `held` below is read with the arena empty so the
		// streamed layer is not counted both there and here, and the passes recreate
		// exactly these chunks.
		g_->wa->release_chunks();
	}
	res_inflight_ = inflight;

	// ── what the process already holds ──
	const u64 held = vram_budget().local();

	// ── the frame the decode holds for the whole call ──
	const i64 nbr = plan_nbr_ > 0 ? plan_nbr_ : 1;
	const i64 rows_alloc = std::max<i64>(plan_rows_, 1);
	// The tied-head slice (`logits`) is allocated from the **window's** arena, not
	// from `keep`: it is 300 MB of embedding rows that only the code sampling
	// reads, and `keep` is never handed back - parked there it would still be
	// charged while the DiT plans its own window, i.e. one resident DiT block spent
	// on a phase that has already finished. So it is priced against the window's
	// own room below rather than in `act_kv` (which would subtract `keep`'s
	// existing chunks, and it does not use them).
	//
	// The `fits_window` flag is what `logits` reads: when this plan takes a window
	// the slice goes there, and when it does not (no device, no budget) the slice
	// falls back to `keep` and is priced the old way - in `act_kv`, below.
	const u64 head_bytes = plan_head_bytes_;
	bool window_planned = false;
	u64 act_kv = 0;
	if (plan_cap_ > 0) {
		// The K/V cache, in `keep` (`AceKvCache::init` uses `kalloc`), in the order
		// `init` allocates it.
		std::vector<u64> sizes;
		sizes.reserve((size_t)(2 * nbr * cfg_.n_layers) + 1);
		const u64 kv_row = (u64)plan_cap_ * (u64)HK * 2;
		for (i64 p = 0; p < nbr; p++)
			for (i64 l = 0; l < cfg_.n_layers; l++) {
				sizes.push_back(kv_row);
				sizes.push_back(kv_row);
			}
		if (!window_planned && plan_head_bytes_) sizes.push_back(plan_head_bytes_);
		// `kalloc` (which `AceKvCache::init` and `head_w_` use) is the `keep` arena
		// when the caller has one and `aa` otherwise, so the charge is priced in
		// whichever arena will really take it.
		GpuArena* ar = g_->keep ? g_->keep : g_->aa;
		const u64 kc = ar ? ar->chunk_bytes() : chunk;
		const u64 need = arena_chunk_cost(sizes, kc);
		// Only the part that arena does not already hold - its existing chunks are
		// already counted in `held`. In the `keep`-less fallback the scratch term
		// below subtracts `aa`'s capacity, so nothing is handed back twice.
		const u64 have = (ar && ar != g_->aa) ? ar->capacity() : 0;
		act_kv = need > have ? need - have : 0;
	}
	u64 act_scratch = 0;
	{
		// The per-pass scratch, in `generate`'s `run` allocation order (a bump
		// arena, so the order is part of the price). `hid` is the pass's own row
		// count - the prefill is the widest pass of the call - and everything else
		// is `rows_alloc` rows, which is what `AceStackScratch::alloc` is given.
		const i64 HQ = cfg_.n_heads * cfg_.head_dim;
		const i64 I = cfg_.intermediate;
		const u64 r = (u64)rows_alloc;
		std::vector<u64> sizes;
		sizes.reserve((size_t)(17 * nbr) + 4);
		for (i64 p = 0; p < nbr; p++) {
			const i64 pre = (size_t)p < plan_pre_rows_.size() ? plan_pre_rows_[(size_t)p] : 1;
			sizes.push_back((u64)std::max<i64>(pre, 1) * (u64)H * 4);   // hid
			sizes.push_back(r * (u64)H * 4);      // t
			sizes.push_back(r * (u64)HQ * 4);     // qr
			sizes.push_back(r * (u64)HK * 4);     // kr
			sizes.push_back(r * (u64)HK * 4);     // vr
			sizes.push_back(r * (u64)HQ * 4);     // qn
			sizes.push_back(r * (u64)HK * 4);     // kn
			sizes.push_back(r * (u64)HQ * 2);     // qh (fp16)
			sizes.push_back(r * (u64)HK * 2);     // kh
			sizes.push_back(r * (u64)HK * 2);     // vh
			sizes.push_back(r * (u64)HQ * 4);     // attn
			sizes.push_back(r * (u64)H * 4);      // proj
			sizes.push_back(r * (u64)I * 4);      // gate
			sizes.push_back(r * (u64)I * 4);      // up
			sizes.push_back(r * (u64)I * 4);      // hh
			sizes.push_back(r * (u64)H * 4);      // down
			sizes.push_back(r * 4);               // the per-row key bound
		}
		// ...and what the head allocates in the same step, after the pass: `logits`
		// stages one [hidden] row and reads back [n_codes], `eos_score` the same for
		// its single eos row.
		sizes.push_back((u64)H * 4);
		sizes.push_back((u64)std::max<i64>(plan_n_codes_, 1) * 4);
		sizes.push_back((u64)H * 4);
		sizes.push_back(4);
		const u64 ac = g_->aa ? g_->aa->chunk_bytes() : chunk;
		const u64 need = arena_chunk_cost(sizes, ac);
		const u64 have = g_->aa ? g_->aa->capacity() : 0;
		act_scratch = need > have ? need - have : 0;
	}
	const u64 act = act_kv + act_scratch;

	// ── the plan's floor, and the room left for the window ──
	// `held` is measured; `act` and `inflight` are what this call still adds to it.
	// The window may spend what is left after all three and the slack - `room`, the
	// same quantity `ImageDiT::plan_residency` calls `cap` (the contract: as many
	// layers as fit under `budget - (held + act + inflight) - slack`).
	//
	// The slack is the difference between a model of two bump arenas and the
	// allocators themselves; a plan that spends the last tens of megabytes of the
	// card is a plan that fails on the roundings, in the middle of the decode loop.
	const u64 kSlack = 96ull << 20;
	res_reserve_ = held + act + inflight;
	u64 room = budget > res_reserve_ + kSlack ? budget - res_reserve_ - kSlack : 0;
	// The tied-head slice is taken *out of the window's own room* (see `head_bytes`
	// above): the slice and the window share one arena, so a window that spent the
	// last of the room would refuse the slice on the first decode step. One chunk
	// of margin covers the arena rounding the slice adds (a 300 MB allocation does
	// not tile an 80 MB granularity cleanly).
	if (head_bytes) room = room > head_bytes + chunk ? room - head_bytes - chunk : 0;
	// ...and never more than the device will hand out: the ledger and the driver's
	// live figure are two different walls (see `VramBudget::live_headroom`).
	const u64 device_room = vram_budget().live_headroom();
	if (device_room < room) room = device_room;
	const u64 cap = room;

	// ── take the window ──
	// How many layers, and what they charge, is decided by the accountant rather
	// than by arithmetic on tensor sizes: `res_arena_.capacity()` *is* the charge
	// for the window, and each layer's own growth is measured as it is admitted.
	// One streamed layer's worth (`est`) is kept in hand so the first layer the loop
	// streams never overshoots the cap it was admitted under.
	res_arena_.release_chunks();
	res_layers_.assign((size_t)cfg_.n_layers, AceQwen3Layer{});
	i64 n = 0;
	u64 est = inflight;
	g_->ctx->begin();
	if (const char* e = getenv("PHI_ACE_LM_RES")) {
		const long v = strtol(e, nullptr, 10);
		const i64 want = v <= 0 ? 0 : std::min<i64>(cfg_.n_layers, v);
		for (; n < want; n++) {
			if (!g_->ctx->recording()) g_->ctx->begin();
			try {
				ace_qwen3_upload_layer(*g_, st_, "model.", n, &res_layers_[(size_t)n], &res_arena_);
			} catch (const std::exception& ex) {
				if (getenv("PHI_ACE_LM_PLAN"))
					fprintf(stderr, "[acelm] plan: forced window stopped at %lld/%lld (%s)\n",
					        (long long)n, (long long)want, ex.what());
				break;
			}
		}
	} else {
		while (n < cfg_.n_layers) {
			if (res_arena_.capacity() + est > cap) break;
			const u64 have = res_arena_.capacity();
			if (!g_->ctx->recording()) g_->ctx->begin();
			try {
				ace_qwen3_upload_layer(*g_, st_, "model.", n, &res_layers_[(size_t)n], &res_arena_);
			} catch (const std::exception& ex) {
				// A failed upload ends the window at the last layer that fit; it never
				// ends the run (the layers after it stream, which is what they did
				// before this existed).
				if (getenv("PHI_ACE_LM_PLAN"))
					fprintf(stderr, "[acelm] plan: window stopped at %lld/%lld (%s)\n",
					        (long long)n, (long long)cfg_.n_layers, ex.what());
				break;
			}
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
	res_planned_ = true;
	// Kept past `release_resident()` (see `planned_layers`): the window is gone by
	// the time `generate` returns, and this is the only record of what the code
	// sampling ran with.
	res_last_n_ = n;
	res_last_bytes_ = res_bytes_;
	// The slice is now priced against the window's room, so `logits` must put it
	// there rather than in `keep` - and only after the window really exists, which
	// is why the flag is set here and not at the top.
	head_in_window_ = head_bytes != 0;
	if (getenv("PHI_ACE_LM_PLAN"))
		fprintf(stderr,
		        "[acelm] plan: budget=%s held=%s act=%s (kv %s, scratch %s) head=%s(window) "
		        "stream=%s reserve=%s layer=%s window=%lldx%s (%s/token re-read, %lld/%lld layers)\n",
		        format_bytes(budget).c_str(), format_bytes(held).c_str(), format_bytes(act).c_str(),
		        format_bytes(act_kv).c_str(), format_bytes(act_scratch).c_str(),
		        format_bytes(head_in_window_ ? head_bytes : 0).c_str(),
		        format_bytes(inflight).c_str(), format_bytes(res_reserve_).c_str(),
		        format_bytes(res_layer_bytes_).c_str(), (long long)res_n_,
		        format_bytes(res_bytes_).c_str(),
		        format_bytes((u64)(cfg_.n_layers - res_n_) * res_layer_bytes_).c_str(),
		        (long long)res_n_, (long long)cfg_.n_layers);
}

void AceLm::release_resident() {
	// The window goes back to the device and to the accountant. Only legal because
	// nothing in it is live: `generate` holds it across the whole decode and this is
	// either its exit path or a re-plan.
	res_arena_.release_chunks();
	res_layers_.clear();
	res_n_ = 0;
	res_bytes_ = 0;
	res_planned_ = false;
	// The tied-head slice went into that arena when the plan succeeded (`logits`),
	// so the cache marker has to be dropped with it: the next `generate` must
	// re-upload rather than read rows that are no longer there.
	head_w_ = GpuAlloc{};
	head_lo_ = head_hi_ = -1;
	head_in_window_ = false;
}

void AceLm::logits(const std::vector<float>& h, i32 lo, i32 hi, std::vector<float>& out) {
	const i64 H = cfg_.hidden;
	const i64 n = (i64)hi - (i64)lo;
	if ((i64)h.size() != H) throw MediaError("ace lm: the logits row is not [hidden]");
	if (n <= 0) throw MediaError("ace lm: empty logit range");
	// The head is the tied embedding table (`config.lm_head = False`), so what the
	// sampler scores is a row slice of `model.embed_tokens.weight` - read at the
	// precision the checkpoint stores it in.
	const StTensor& t = st_.require("model.embed_tokens.weight");
	const GemmOperand op = gemm_operand_for(t.dtype);
	const u64 row_bytes = (u64)H * dtype_size(t.dtype);
	if (t.shape[1] != H) throw MediaError("ace lm: the tied head is not [vocab, hidden]");
	if (head_lo_ != lo || head_hi_ != hi) {
		// The slice goes into the **resident window's** arena, not the `keep` arena,
		// whenever a window was planned and priced for it (see `plan_residency`). It
		// is 300 MB of tied embedding rows that only the code sampling reads, and
		// `keep` is never handed back: parked there it would still be charged while
		// the DiT plans its own window - one resident DiT block spent on a phase that
		// has already finished (see `AceDiT::plan_residency`). In the window's arena
		// it is released with the window by `release_resident`, and the next
		// `generate` re-uploads it once.
		//
		// When no window was planned (`head_in_window_` false - no device, no budget
		// to plan against) the `keep` arena is what is left, and the upload is then a
		// one-off the way it always was - priced in the plan's `act_kv` term.
		head_w_ = head_in_window_ ? res_arena_.alloc((u64)n * row_bytes)
		                           : g_->kalloc((u64)n * row_bytes);
		const bool own = !g_->ctx->recording();
		if (own) g_->ctx->begin();
		g_->upload_into(head_w_, (const u8*)st_.data_of(t) + (u64)lo * row_bytes,
		                (u64)n * row_bytes);
		if (own) {
			g_->ctx->submit();
			g_->ring->rewind();
		}
		head_lo_ = lo;
		head_hi_ = hi;
	}

	GpuAlloc x = g_->aalloc((u64)H * 4);
	GpuAlloc y = g_->aalloc((u64)n * 4);
	g_->ctx->begin();
	upload_range(*g_->ctx, *g_->ring, x.res, x.off, h.data(), (u64)H * 4);
	GemmF16Args ga;
	ga.a = x;
	ga.b = head_w_;
	ga.c = y;
	ga.m = 1;
	ga.n = n;
	ga.k = H;
	ga.a_is_f32 = true;
	ga.b_is_bf16 = op.b_is_bf16;
	ga.b_is_f32 = op.b_is_f32;
	dispatch_gemm_f16(*g_->ctx, ga);
	g_->ctx->submit();
	g_->ring->rewind();
	out.assign((size_t)n, 0.0f);
	g_->ctx->download(y.res, y.off, out.data(), (size_t)n * 4);
}

std::vector<i32> AceLm::generate(const std::vector<i32>& pos_ids, const std::vector<i32>& neg_ids,
                                 i64 max_new, i64 min_new, float cfg, float temperature, float top_p,
                                 i64 top_k, float min_p, u64 seed, i32 audio_start, i32 audio_end,
                                 i32 eos_id) {
	if (!g_ || !g_->ok()) throw MediaError("ace lm: not open");
	// From here on the resident window `plan_residency` takes below is handed back on
	// *every* return, including a throw out of the decode loop: the DiT's own plan
	// runs as soon as the codes are sampled and plans against `vram_budget().local()`,
	// so a window still held here would be spent on LM layers it can no longer use
	// instead of on DiT blocks that still need it.
	ResidentGuard guard(this);
	if (pos_ids.empty()) return {};
	if (max_new <= 0) return {};
	if (min_new < 0) min_new = 0;
	if (audio_start < 0) audio_start = kQwenAudioStart;
	if (audio_end < 0) audio_end = kQwenAudioEnd;
	if (eos_id < 0) eos_id = kQwenEos;

	const i64 H = cfg_.hidden;
	const i64 HK = cfg_.n_kv_heads * cfg_.head_dim;
	const i64 n_codes = (i64)audio_end - (i64)audio_start;
	if (n_codes <= 0) throw MediaError("ace lm: empty code range");

	// Classifier-free guidance needs both prompts. The reference left-pads the
	// shorter one with the pad token and runs the pair as *one batch*
	// (`generate_audio_codes`). That padding is a batching artefact: here the two
	// prompts are independent sequences - each row of a causal LM attends only
	// its own - so unequal lengths need nothing, and the only thing the two
	// passes share is the token the combined logits pick.
	const bool use_cfg = (cfg != 1.0f) && !neg_ids.empty();
	const i64 nbr = use_cfg ? 2 : 1;
	const i64 npos = (i64)pos_ids.size();
	const i64 nneg = use_cfg ? (i64)neg_ids.size() : 0;
	const i64 cap = std::max(npos, nneg) + max_new + 8;
	const i64 tile = std::max<i64>(64, (i64)attn_query_tile());
	const i64 rows_alloc = std::max<i64>(std::max(npos, nneg), tile);

	// Plan the resident window for *this* request, before the K/V cache is
	// allocated: `plan_residency` takes no arguments (it is re-entrant and the
	// geometry lives in the `plan_*` members, the way `AvDiT` keeps the shape its
	// plan was taken for), so the numbers it prices are filled in here - the K/V
	// rows the decode will hold, the scratch's row count, the CFG branches and their
	// prefill lengths, the logit range, and the tied-head slice the sampler needs
	// (zero when that range is already on the device, because then the plan's
	// `held` term carries it instead of a new allocation).
	plan_cap_ = cap;
	plan_rows_ = rows_alloc;
	plan_nbr_ = nbr;
	plan_n_codes_ = n_codes;
	plan_pre_rows_.assign((size_t)nbr, 1);
	plan_pre_rows_[0] = npos;
	if (use_cfg) plan_pre_rows_[1] = nneg;
	{
		const StTensor& tie = st_.require("model.embed_tokens.weight");
		const u64 head_row = (u64)H * dtype_size(tie.dtype);
		plan_head_bytes_ =
		    (head_lo_ == audio_start && head_hi_ == audio_end) ? 0 : (u64)n_codes * head_row;
	}
	plan_residency();

	AceKvCache kv[2];
	kv[0].init(*g_, cfg_.n_layers, cap, HK);
	if (use_cfg) kv[1].init(*g_, cfg_.n_layers, cap, HK);

	// The eos row of the tied head, scored with the same GEMM.
	const GemmOperand head_op = gemm_operand_for(st_.require("model.embed_tokens.weight").dtype);
	auto eos_score = [&](const std::vector<float>& last) {
		GpuAlloc x = g_->aalloc((u64)H * 4);
		GpuAlloc y = g_->aalloc(4);
		g_->ctx->begin();
		upload_range(*g_->ctx, *g_->ring, x.res, x.off, last.data(), (u64)H * 4);
		GemmF16Args ga;
		ga.a = x;
		ga.b = eos_w_;
		ga.c = y;
		ga.m = 1;
		ga.n = 1;
		ga.k = H;
		ga.a_is_f32 = true;
		ga.b_is_bf16 = head_op.b_is_bf16;
		ga.b_is_f32 = head_op.b_is_f32;
		dispatch_gemm_f16(*g_->ctx, ga);
		g_->ctx->submit();
		g_->ring->rewind();
		std::vector<float> v = g_->download_f32(y, 1);
		return v[0];
	};
	if (eos_row_ != eos_id) {
		const StTensor& t = st_.require("model.embed_tokens.weight");
		const u64 row_bytes = (u64)H * dtype_size(t.dtype);
		if (eos_id < 0 || (i64)eos_id >= t.shape[0])
			throw MediaError("ace lm: eos id out of range");
		eos_w_ = g_->kalloc(row_bytes);
		const bool own = !g_->ctx->recording();
		if (own) g_->ctx->begin();
		g_->upload_into(eos_w_, (const u8*)st_.data_of(t) + (u64)eos_id * row_bytes, row_bytes);
		if (own) {
			g_->ctx->submit();
			g_->ring->rewind();
		}
		eos_row_ = eos_id;
	}

	// Per-pass device state. `hid` is the fp32 stream the stack runs over and
	// `sc` the activations; both live in the arenas `new_step()` resets, so their
	// handles are rebuilt on every pass rather than kept across steps.
	std::vector<GpuAlloc> hid((size_t)nbr);
	std::vector<AceStackScratch> sc((size_t)nbr);
	std::vector<AcePass> pass((size_t)nbr);
	std::vector<std::vector<float>> hp((size_t)nbr);
	std::vector<i64> len((size_t)nbr, 0);

	// One stack pass over `rows[p]` rows of every branch, writing the branch's K/V
	// at rows [pos0, pos0 + rows) and reading keys [0, kvlen).
	auto run = [&](const std::vector<std::vector<float>>& embeds,
	               const std::vector<i64>& rows, const std::vector<i64>& pos0,
	               const std::vector<i64>& kvlen) {
		g_->new_step();
		for (i64 p = 0; p < nbr; p++) {
			const i64 S = rows[(size_t)p];
			hid[(size_t)p] = g_->aalloc((u64)S * (u64)H * 4);
			sc[(size_t)p].alloc(*g_, rows_alloc, cfg_);
			// Row r of the pass sees keys [0, min(kv_len, pos0 + r + 1)): causal
			// during the prefill, and the whole written prefix for a decode row
			// (which is exactly `pos0 + 1`).
			std::vector<u32> u((size_t)rows_alloc, 0);
			for (i64 r = 0; r < rows_alloc; r++)
				u[(size_t)r] = (u32)std::min<i64>(kvlen[(size_t)p],
				                                  pos0[(size_t)p] + std::min<i64>(r + 1, S));
			AcePass& P = pass[(size_t)p];
			P.hidden = &hid[(size_t)p];
			P.rows = S;
			P.kv_row0 = pos0[(size_t)p];
			P.kv_len = kvlen[(size_t)p];
			P.kv = &kv[p];
			P.sc = &sc[(size_t)p];
			g_->ctx->begin();
			upload_range(*g_->ctx, *g_->ring, hid[(size_t)p].res, hid[(size_t)p].off,
			             embeds[(size_t)p].data(), (u64)embeds[(size_t)p].size() * 4);
			P.ub = upload_u32(*g_, u.data(), rows_alloc);
			g_->ctx->submit();
			g_->ring->rewind();
		}
		ace_stack_multi(*g_, st_, "model.", cfg_, cfg_.n_layers, true, "model.norm.weight", 0,
		                pass.data(), nbr, &res_arena_, &res_layers_, res_n_);
		for (i64 p = 0; p < nbr; p++)
			hp[(size_t)p] = g_->download_f32(hid[(size_t)p], (u64)rows[(size_t)p] * (u64)H);
	};

	// ── prefill ──
	{
		std::vector<std::vector<float>> pre((size_t)nbr);
		std::vector<i64> rows((size_t)nbr), pos0((size_t)nbr, 0), kvlen((size_t)nbr);
		for (i64 p = 0; p < nbr; p++) {
			const std::vector<i32>& ids = p == 0 ? pos_ids : neg_ids;
			const i64 S = (i64)ids.size();
			pre[(size_t)p].assign((size_t)S * (size_t)H, 0.0f);
			ace_gather_rows(st_, "model.embed_tokens.weight", H, ids.data(), S,
			                pre[(size_t)p].data());
			rows[(size_t)p] = S;
			kvlen[(size_t)p] = S;
		}
		run(pre, rows, pos0, kvlen);
		for (i64 p = 0; p < nbr; p++) len[(size_t)p] = rows[(size_t)p];
	}

	std::vector<float> last((size_t)H), lp_pos((size_t)n_codes), lp_neg((size_t)n_codes);
	std::mt19937_64 rng(seed);
	std::vector<i32> out;

	for (i64 step = 0; step < max_new; step++) {
		// The reference's `use_eos_score = eos_token_id is not None and eos_token_id
		// < audio_start_id and min_tokens < step`
		// (`comfy/text_encoders/ace15.py::sample_manual_loop_no_classes`). The eos
		// branch is *off* until `step` has passed `min_new`, and for text-to-music the
		// caller sets `min_new == max_new` (both are `duration * 5`, the tokenizer's
		// own `tokens_duration`), so eos is never offered: the LM always emits exactly
		// `duration * 5` codes.
		//
		// Scoring it from step 0 instead - which this loop used to do - lets the tied
		// embedding's eos row win the categorical on a step the reference never gives
		// it, the loop breaks, and the generation is short. The DiT then conditions
		// the tail of the latent on the *pad* code (`prepare_condition`'s
		// `constant 35847`), which is a constant row per pooled window - the detokenised
		// hint is flat there, so the diffusion step has nothing to sing and the second
		// half of the clip comes out silent.
		const bool use_eos = eos_id >= 0 && min_new < step;
		std::memcpy(last.data(), &hp[0][hp[0].size() - (size_t)H], sizeof(float) * (size_t)H);
		logits(last, audio_start, audio_end, lp_pos);
		float eos_pos = 0.0f;
		if (use_eos) eos_pos = eos_score(last);
		float eos_neg = 0.0f;
		if (use_cfg) {
			std::memcpy(last.data(), &hp[1][hp[1].size() - (size_t)H], sizeof(float) * (size_t)H);
			logits(last, audio_start, audio_end, lp_neg);
			if (use_eos) eos_neg = eos_score(last);
		}
		const i32 tok = sample_audio(lp_pos, lp_neg, eos_pos, eos_neg, use_eos, use_cfg, cfg,
		                             temperature, top_p, top_k, min_p, rng, audio_start);
		if (use_eos && tok < 0) break;
		out.push_back(tok - audio_start);

		// Feed the sampled token back: one more row through *both* sequences. The
		// two branches share the layer upload (see `ace_stack_multi`), which is
		// why they are advanced together rather than one after the other.
		const i32 one[1] = {tok};
		std::vector<float> row((size_t)H);
		ace_gather_rows(st_, "model.embed_tokens.weight", H, one, 1, row.data());
		std::vector<std::vector<float>> emb((size_t)nbr);
		std::vector<i64> rows((size_t)nbr, 1), pos0((size_t)nbr), kvlen((size_t)nbr);
		for (i64 p = 0; p < nbr; p++) {
			emb[(size_t)p] = row;
			pos0[(size_t)p] = len[(size_t)p];
			kvlen[(size_t)p] = len[(size_t)p] + 1;
		}
		run(emb, rows, pos0, kvlen);
		for (i64 p = 0; p < nbr; p++) len[(size_t)p] += 1;
	}
	return out;
}

}  // namespace phi::media
