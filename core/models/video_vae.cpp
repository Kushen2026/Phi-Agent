// MiniMax H3 video VAE — native implementation (owner: W2).
//
// Reference: C:/ComfyUI/comfy/ldm/minimax/vae.py (MiniMaxH3VideoVAE). The plan
// called this "a causal 3D-conv decoder"; the reference is unambiguous that the
// *encoder* is the causal 3D FCN and the *decoder* is a 36-layer ViT3D
// transformer. Both halves are
// implemented exactly as the reference does them, including the temporal
// chunking and the spatial tiling — in the reference those are part of the
// decode path, not an optimisation.
//
// Weight streaming: the fp16 decoder is 4.8 GB, more than the 6 GB card holds
// alongside a tile's activations, so one transformer block (135 MB) is uploaded
// into a dedicated arena at a time and that arena is reset between blocks. The
// encoder (360 MB) is lazily loaded once into the class's permanent arena.
//
// The frozen header may not grow members, so the state this implementation
// needs lives in a side table keyed by `this` (see `g_impl`).

#include "models/video_vae.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kernels/kernels.hpp"
#include "kernels/gpu_ops.hpp"
#include "models/video_vae_internal.hpp"
#include "runtime/vram_budget.hpp"

namespace phi::media {

// ── reference constants ───────────────────────────────────────────────────
//
// vae.py:327 — constructor defaults, confirmed against the checkpoint
// (encoder.down.0..5 = 128/256/256/512/512/1024 => ch_mult 1,2,2,4,4,8).

static const i64 kSpaceDown[6] = {2, 2, 2, 2, 1, 1};
static const i64 kTimeDown[6] = {1, 2, 2, 1, 1, 1};
static const i64 kClipLength = 17;
static const i64 kTokenDrop = 3;
static const i64 kTileSize = 256;
static const i64 kTileOverlapMin = 64;
static const i64 kNormGroups = 32;
static const i64 kEncLevels = 6;
static const float kGnEps = 1e-6f;

// decoder (ViT3DDecoder, vae.py:261)
static const i64 kDim = 2048;   // heads * dim_head
static const i64 kHeads = 32;
static const i64 kDimHead = 64;
static const i64 kLayers = 36;
static const i64 kPatch = 16;   // vae_ratio
static const i64 kPatchT = 4;   // vae_ratio_t
static const i64 kRegTokens = 4;
static const float kAttnEps = 1e-5f;
static const i64 kRopePairs = 24;   // int(dim_head * 0.75) / 2
static const i64 kRopeAxes = 3;
static const i64 kRopeFreqs = 8;    // len(arange(0, 1, 2*n_dim/dim))
static const float kRopeTheta = 100.0f;

// pixel normalisation, ImageNet statistics (vae.py:17)
static const float kPixelMean[3] = {0.485f, 0.456f, 0.406f};
static const float kPixelStd[3] = {0.229f, 0.224f, 0.225f};

static i64 pymod(i64 a, i64 b) {
	i64 r = a % b;
	return r < 0 ? r + b : r;
}

// ── plan helpers (declared in video_vae_internal.hpp) ──────────────────

H3Tiles h3_split_tiles(i64 input_len, i64 tile_size, i64 tile_overlap_min, i64 ratio) {
	H3Tiles t;
	if (tile_size >= input_len) {
		t.pos = {0};
		t.len = {input_len};
		return t;
	}
	i64 n = ceil_div(input_len, tile_size);
	i64 remaining = 0;
	for (;;) {
		const i64 overlaps = tile_overlap_min * (n - 1);
		remaining = tile_size * n - overlaps - input_len;
		if (remaining < 0) n++;
		else break;
	}
	std::vector<i64> overlaps((size_t)n - 1, tile_overlap_min);
	const i64 units = remaining / ratio;
	for (i64 i = 0; i < units; i++) overlaps[(size_t)(i % (n - 1))] += ratio;
	t.pos = {0};
	for (i64 i = 0; i < n - 1; i++)
		t.pos.push_back(t.pos.back() + tile_size - overlaps[(size_t)i]);
	t.len.assign((size_t)n, tile_size);
	t.overlap = overlaps;
	return t;
}

void h3_decode_temporal_chunks(i64 z_len, i64 tokens_chunk_size, i64 token_drop, i64* pad_tokens,
                               i64* num_chunks) {
	i64 pseudo = z_len + token_drop;
	i64 pad = pymod(-pseudo, tokens_chunk_size);
	pseudo += pad;
	i64 chunks = pseudo / tokens_chunk_size - (token_drop > 0 ? 1 : 0);
	if (chunks < 1) {
		pad += tokens_chunk_size;
		chunks += 1;
	}
	*pad_tokens = pad;
	*num_chunks = chunks;
}

i64 h3_decode_output_frames(i64 z_len, i64 pad_tokens, i64 num_chunks, i64 clip_length,
                            i64 tokens_chunk_size, i64 token_drop, i64 vae_ratio_t) {
	const i64 token_overlap = pymod(-token_drop, tokens_chunk_size);
	const i64 frame_pre_padding = pymod(-clip_length, vae_ratio_t);
	const i64 chunk_dec = tokens_chunk_size * vae_ratio_t;
	const i64 split_count = (token_drop > 0 ? 1 : 0) + 1;

	i64 total_frames = 0, final_overlap_frames = 0;
	for (i64 i = 0; i < num_chunks; i++) {
		const i64 t_start = i * tokens_chunk_size;
		const i64 t_end = t_start + tokens_chunk_size + token_overlap;
		const i64 clip_token_len = std::max<i64>(0, std::min(t_end, z_len) - std::min(t_start, z_len));
		const i64 clip_frame_len = clip_token_len * vae_ratio_t;
		for (i64 j = 0; j < split_count; j++) {
			const i64 f_start = j * chunk_dec;
			const i64 f_end = std::min(f_start + chunk_dec, clip_frame_len);
			const i64 chunk_frames = std::max<i64>(0, f_end - f_start - frame_pre_padding);
			if (j == 0) total_frames += chunk_frames;
			else final_overlap_frames = chunk_frames;
		}
	}
	total_frames += final_overlap_frames;

	i64 pad_frames = 0;
	if (pad_tokens > 0) {
		const i64 intra_tail = clip_length % vae_ratio_t;
		if (intra_tail == 0) {
			pad_frames = pad_tokens * vae_ratio_t;
		} else {
			const i64 before = z_len - pad_tokens;
			for (i64 k = 0; k < pad_tokens; k++)
				pad_frames += ((before + k) % tokens_chunk_size == 0) ? intra_tail : vae_ratio_t;
		}
	}
	return total_frames - pad_frames;
}

// ── internal state ────────────────────────────────────────────────────────

namespace {

// A causal 3-D convolution's weights (fp16, as stored in the checkpoint).
struct CConv {
	i64 ic = 0, oc = 0, kt = 3, kh = 3, kw = 3;
	i64 strT = 1, strS = 1;
	// Padding, in the reference's own terms (comfy/ldm/minimax/vae.py).
	//
	// `CausalConv3d.forward` (vae.py:47) pads H and W **reflect**-symmetric by
	// `causal_padding[1]`/`[2]`, zero-pads the *front* of the time axis by
	// `2 * causal_padding[0]`, and then runs a plain `Conv3d` with padding=0.
	// `Downsample3D` (vae.py:86) is the exception: it reflect-pads H and W by 1 on
	// the *right/bottom only* and then calls a CausalConv3d whose own spatial
	// padding is 0. So the two numbers are genuinely different and both are
	// recorded here (`padH` left/top, `padHR` right/bottom).
	//
	// Temporal: for a single input frame the reference truncates the weight to its
	// last tap instead of convolving against zero frames (vae.py:52,
	// autopad="causal_zero"), i.e. no temporal padding at all and only `ktl = kt-1`
	// contributes. `padT = kt - 1` reproduces that exactly: the tap index becomes
	// `ot + ktl - (kt-1)`, which is negative for every ktl < kt-1 (the kernel
	// zeroes negative time) and equals ot for ktl = kt-1.
	i64 padT = 0;   // zeros prepended on the time axis (causal front padding)
	i64 padH = 0;   // reflect pad on the left/top
	i64 padHR = 0;  // reflect pad on the right/bottom
	GpuAlloc w, b;
	bool has_bias = false;
	// The weight tensor's own storage dtype, as the kernel's `wDtype` code:
	// 0 = fp32, 1 = fp16, 2 = bf16. `load_conv_weight` sets it from the
	// checkpoint, so the kernel reads exactly the bytes the file holds instead of
	// one hard-coded width.
	int w_dtype = 1;
};

struct CGn {
	GpuAlloc w, b;
};

struct CRes {
	CGn n1, n2;
	CConv c1, c2, nin;
	bool has_nin = false;
};

struct Encoder {
	CConv conv_in;
	std::vector<std::vector<CRes>> blocks;   // [level][block]
	std::vector<CConv> downs;
	std::vector<bool> has_down;
	CGn norm_out;
	CConv conv_out;
};

// One decoder transformer block's weights. Defined up here so `Impl`'s resident
// cache can hold them (see `Impl::wcache`); `load_block` fills one.
//
// The four projections reach the device in one of two forms, and `i8` says
// which:
//
//   * fp16 - the fp16 checkpoint's stored precision, run through `gemm_f16_mma`.
//   * int8 + a per-output-row fp32 scale - an int8 (convrot) checkpoint's stored
//     precision, run through `dispatch_int8_gemm`.
//
// The int8 form is what makes an int8 checkpoint actually faster than an fp16
// one. Materialising the int8 matrices to fp16 instead (the old behaviour) threw
// away both of the things the quantisation buys: half the bytes have to be read
// off the file and pushed through the ring for every streamed block, and the
// projections lose the int8 GEMM's ~54 TOPS for the fp16 mma kernel's ~18
// TFLOP/s - at 320x320 x 5 frames the int8 checkpoint used to need 54 s against
// the fp16 checkpoint's 10.8 s, because every streamed block was decoded and
// rotated back to fp16 on every tile.
struct BlockW {
	GpuAlloc qkv_w, qkv_b, out_w, out_b, w1_w, w1_b, w2_w, w2_b, n1, n2, s1, s2;
	GpuAlloc qkv_q, qkv_s, out_q, out_s, w1_q, w1_s, w2_q, w2_s;
	bool i8 = false;
};

struct DecStatics {
	GpuAlloc x_embed_w, x_embed_b;
	GpuAlloc reg_tokens;   // [4, 2048] fp32
	GpuAlloc norm_out_w, norm_out_b;
	GpuAlloc proj_out_w, proj_out_b;
};

struct Impl {
	Encoder enc;
	DecStatics dec;
	CConv quant, post_quant;
	GpuAlloc latents_mean_dev, latents_std_dev;
	GpuAlloc inv_freq_dev;
	std::vector<float> latents_mean, latents_std;   // fp32, from the checkpoint
	std::vector<std::string> block_prefix;
	GpuArena stream;   // one transformer block at a time
	bool enc_loaded = false;

	// ── the decoder blocks held across tiles ──────────────────────────────
	//
	// The tiled decoder runs 36 transformer blocks per *tile*, and a 540P clip
	// is ~105 tiles, so the streaming window above re-reads (and re-uploads) the
	// same 3.6 GB of fp16 weights 105 times: ~380 GB through the ring for one
	// clip, which was 45 % of the decode. The decoder is only ~3.6 GB, the
	// budget is ~4.85 GB, and the tile activations are a few hundred MB, so the
	// whole stack fits: `wcache` holds it, and the per-tile loop reads the block
	// addresses out of `dec_w` instead of re-loading them.
	//
	// Filled greedily on the first tile (the budget is re-checked before every
	// block) and abandoned the moment the accountant refuses a chunk, so a
	// smaller card just keeps fewer blocks resident and streams the rest
	// exactly as it used to.
	GpuArena wcache;
	std::vector<BlockW> dec_w;
	i64 dec_n = 0;        // blocks held in `wcache`
	u64 dec_blk = 0;      // one block's bytes, for the budget check
	bool dec_pinned = false;
	bool dec_tried = false;
	i64 dec_logged = -1;  // last residency count reported, so the log is one line
	// The file ranges of one transformer block's four fp16 matrices, so the
	// blocks that do not fit the VRAM cache can be read *while the previous one
	// computes* (see SafeTensors::prefetch_ranges) instead of in front of it.
	std::vector<std::vector<std::pair<u64, u64>>> block_ranges;
};

std::map<const VideoVae*, std::unique_ptr<Impl>> g_impl;

Impl* impl_of(const VideoVae* v) {
	auto it = g_impl.find(v);
	return it == g_impl.end() ? nullptr : it->second.get();
}

// ── host-side [C,T,H,W] view ──────────────────────────────────────────────
//
// `ct` is the stride between channels, which is *not* T*H*W once the view is a
// temporal slice of a larger buffer, and `rw` is the row stride, which is *not*
// W once the view is an x-slice of a wider row. Both are needed by
// tiled_decode, whose strips are canvas-width buffers that get sliced at a
// tile's own x-range.
struct MatV {
	float* p = nullptr;
	i64 C = 0, T = 0, H = 0, W = 0;
	i64 ct = 0;
	i64 rw = 0;
	float* at(i64 c, i64 t, i64 h, i64 w) const {
		return p + (size_t)c * (size_t)ct + (size_t)(t * H + h) * (size_t)rw + (size_t)w;
	}
	static MatV whole(float* p, i64 C, i64 T, i64 H, i64 W) {
		return MatV{p, C, T, H, W, T * H * W, W};
	}
	// a [C, t0:t0+nt, H, W] slice of a [C, T, H, W] buffer
	static MatV tslice(float* p, i64 C, i64 T, i64 t0, i64 nt, i64 H, i64 W) {
		return MatV{p + (size_t)t0 * H * W, C, nt, H, W, T * H * W, W};
	}
	// the [C, T, H, x0:x0+w) columns of a [C, T, H, stride] buffer
	static MatV xslice(float* p, i64 C, i64 T, i64 H, i64 stride, i64 x0, i64 w) {
		return MatV{p + (size_t)x0, C, T, H, w, T * H * stride, stride};
	}
};

// vae.py:448 — `a`'s trailing `extent` slabs along `axis` fade into `b`'s head.
// axis: 1 = T, 2 = H, 3 = W.
void blend_axis(const MatV& a, const MatV& b, i64 extent, int axis) {
	const i64 bdim = (axis == 1) ? b.T : (axis == 2) ? b.H : b.W;
	const i64 adim = (axis == 1) ? a.T : (axis == 2) ? a.H : a.W;
	const i64 ne = std::min({adim, bdim, extent});
	if (ne <= 0) return;
	for (i64 c = 0; c < b.C; c++) {
		if (axis == 1) {
			for (i64 t = 0; t < ne; t++) {
				const float wb = (float)t / (float)extent, wa = 1.0f - wb;
				for (i64 h = 0; h < b.H; h++)
					for (i64 w = 0; w < b.W; w++) {
						const float av = *a.at(c, adim - extent + t, h, w);
						float* bv = b.at(c, t, h, w);
						*bv = av * wa + (*bv) * wb;
					}
			}
		} else if (axis == 2) {
			for (i64 t = 0; t < b.T; t++) {
				for (i64 y = 0; y < ne; y++) {
					const float wb = (float)y / (float)extent, wa = 1.0f - wb;
					for (i64 w = 0; w < b.W; w++) {
						const float av = *a.at(c, t, adim - extent + y, w);
						float* bv = b.at(c, t, y, w);
						*bv = av * wa + (*bv) * wb;
					}
				}
			}
		} else {
			for (i64 t = 0; t < b.T; t++) {
				for (i64 h = 0; h < b.H; h++) {
					for (i64 x = 0; x < ne; x++) {
						const float wb = (float)x / (float)extent, wa = 1.0f - wb;
						const float av = *a.at(c, t, h, adim - extent + x);
						float* bv = b.at(c, t, h, x);
						*bv = av * wa + (*bv) * wb;
					}
				}
			}
		}
	}
}

// ── dispatch wrappers (constant-buffer layouts are documented in the HLSL) ─

void dispatch_conv3d_causal(ComputeContext& ctx, const CConv& c, const GpuAlloc& x,
                            const GpuAlloc& y, i64 IT, i64 IH, i64 IW, i64 OT, i64 OH, i64 OW) {
	if (OT <= 0 || OH <= 0 || OW <= 0) throw MediaError("h3vae: empty conv3d output");
	GpuKernel* pso = ctx.pipeline("h3vae_conv3d_causal", conv3d_causal_hlsl(),
	                                        "conv3d_causal", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)c.ic;
	p.values[1] = (u32)IT;
	p.values[2] = (u32)IH;
	p.values[3] = (u32)IW;
	p.values[4] = (u32)c.oc;
	p.values[5] = (u32)OT;
	p.values[6] = (u32)OH;
	p.values[7] = (u32)OW;
	p.values[8] = (u32)x.off;
	p.values[9] = (u32)c.w.off;
	p.values[10] = c.has_bias ? (u32)c.b.off : 0u;
	p.values[11] = (u32)y.off;
	p.values[12] = (u32)(c.kt | (c.kh << 4) | (c.kw << 8));
	p.values[13] = (u32)(c.strT | (c.strS << 4));
	// pads: 0-3 padT, 4-7 padH (left/top), 8-11 padHR (right/bottom).
	//
	// There used to be a `single` special case here that set ktBase = kt-1 for a
	// one-frame input. That pushed the only tap `it` is allowed to read *past* the
	// input (`it = ot + (kt-1) + ktl`), so every tap was out of range and the
	// convolution degenerated to its bias - which is the whole of the ref2va
	// reference-image encode (T is always 1 there). `padT = kt - 1` gets the
	// reference's behaviour without a special case, because the kernel already
	// zeroes negative time.
	p.values[14] = (u32)(c.padT | (c.padH << 4) | (c.padHR << 8));
	p.values[15] = c.has_bias ? 1u : 0u;
	p.values[16] = (u32)std::clamp(c.w_dtype, 0, 2);
	p.srv[0] = x.res;
	p.srv[1] = c.w.res;
	p.srv[2] = c.has_bias ? c.b.res : x.res;
	p.uav[0] = y.res;
	ctx.dispatch(pso, p, (u32)ceil_div(c.oc, 64), (u32)ceil_div(OT * OH * OW, 64), (u32)OT);
}

// ── tensor-core GEMM wrappers (kernels/gemm_f16_mma.cpp) ──────────────────
//
// The mma kernel takes fp16 A, so the fp32 activation is cast into per-layer
// scratch first; that pass is a few hundred KB of traffic against a 200 MB
// weight sweep, and it is what lets the GEMM use cp.async for both operands.
//
// These two are defined *outside* the anonymous namespace (they are declared in
// video_vae_internal.hpp); the file-local helpers around them are not.
// Can this shape go through the mma kernel? K must be a whole number of
// 16-half mma k-steps (the staging zero-fills a partial tile, but the `K - 8`
// address clamp needs K >= 16) and every operand must be 16-byte aligned - the
// arena hands out 256-byte allocations, but a GpuAlloc is a base plus an offset
// and the sub-allocations this path is handed are not all at index 0.
bool mma_gemm_shape_ok(i64 m, i64 n, i64 k, const GpuAlloc& a, const GpuAlloc& w,
                       const GpuAlloc& c) {
	if (m <= 0 || n <= 0 || k < 16 || (k % 16) != 0) return false;
	if (((a.off | w.off | c.off) & 15u) != 0) return false;
	// The kernel's k-tile staging reads 16-byte chunks, so a row of A must start
	// on one: element stride k halfs = 2k bytes, a multiple of 16 iff k is even.
	return (k % 8) == 0;
}

void dispatch_causal_gn(ComputeContext& ctx, const CGn& g, const GpuAlloc& x, const GpuAlloc& y,
                        i64 C, i64 T, i64 H, i64 W) {
	GpuKernel* pso =
	    ctx.pipeline("h3vae_causal_gn", causal_gn_hlsl(), "causal_gn", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)C;
	p.values[1] = (u32)T;
	p.values[2] = (u32)(H * W);
	p.values[3] = (u32)kNormGroups;
	p.values[4] = (u32)(C / kNormGroups);
	p.values[5] = (u32)x.off;
	p.values[6] = (u32)g.w.off;
	p.values[7] = (u32)g.b.off;
	p.values[8] = (u32)y.off;
	float eps = kGnEps;
	memcpy(&p.values[9], &eps, 4);
	p.values[10] = g.w ? 1u : 0u;
	p.srv[0] = x.res;
	p.srv[1] = g.w.res ? g.w.res : x.res;
	p.srv[2] = g.b.res ? g.b.res : x.res;
	p.uav[0] = y.res;
	ctx.dispatch(pso, p, (u32)(T * kNormGroups), 1, 1);
}

void dispatch_qkv_split3d(ComputeContext& ctx, const GpuAlloc& qkv, const GpuAlloc& q,
                          const GpuAlloc& k, const GpuAlloc& v, i64 S) {
	GpuKernel* pso =
	    ctx.pipeline("h3vae_qkv_split3d", qkv_split3d_hlsl(), "qkv_split3d", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)S;
	p.values[1] = (u32)kHeads;
	p.values[2] = (u32)kDimHead;
	p.values[3] = (u32)qkv.off;
	p.values[4] = (u32)q.off;
	p.values[5] = (u32)k.off;
	p.values[6] = (u32)v.off;
	float eps = kAttnEps;
	memcpy(&p.values[7], &eps, 4);
	p.srv[0] = qkv.res;
	p.uav[0] = q.res;
	p.uav[1] = k.res;
	p.uav[2] = v.res;
	ctx.dispatch(pso, p, (u32)S, (u32)kHeads, 1);
}

void dispatch_rope3d(ComputeContext& ctx, const GpuAlloc& x, const GpuAlloc& ids,
                     const GpuAlloc& inv_freq, i64 S) {
	GpuKernel* pso =
	    ctx.pipeline("h3vae_rope3d", rope3d_hlsl(), "rope3d", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)(S * kHeads);
	p.values[1] = (u32)kHeads;
	p.values[2] = (u32)kDimHead;
	p.values[3] = (u32)x.off;
	p.values[4] = (u32)ids.off;
	p.values[5] = (u32)inv_freq.off;
	p.values[6] = (u32)kRopePairs;
	p.values[7] = (u32)kRopeAxes;
	p.values[8] = (u32)kRopeFreqs;
	p.srv[0] = ids.res;
	p.srv[1] = inv_freq.res;
	p.uav[0] = x.res;
	ctx.dispatch(pso, p, (u32)(S * kHeads), 1, 1);
}

void dispatch_unpatch3d(ComputeContext& ctx, const GpuAlloc& x, const GpuAlloc& y, i64 T, i64 H,
                        i64 W) {
	const i64 total = 3 * (T * kPatchT) * (H * kPatch) * (W * kPatch);
	GpuKernel* pso =
	    ctx.pipeline("h3vae_unpatch3d", unpatch3d_hlsl(), "unpatch3d", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = 3;
	p.values[1] = (u32)T;
	p.values[2] = (u32)H;
	p.values[3] = (u32)W;
	p.values[4] = (u32)kPatch;
	p.values[5] = (u32)kPatchT;
	p.values[6] = (u32)x.off;
	p.values[7] = (u32)y.off;
	p.values[8] = (u32)total;
	p.srv[0] = x.res;
	p.uav[0] = y.res;
	ctx.dispatch(pso, p, (u32)ceil_div(total, 256), 1, 1);
}

// ── weight loading ────────────────────────────────────────────────────────
//
// Goal #1 for this VAE: the conv weights are read in the checkpoint's own
// storage precision, not in one chosen by the engine. `load_conv` records that
// precision in `CConv::w_dtype` and the kernel decodes it as it loads (`ldWeight`
// in kernels/conv3d_causal.cpp), so an fp32 checkpoint keeps fp32 weights, an
// fp16 one keeps fp16, and a bf16 one keeps bf16 - no rounding anywhere.
//
// The one family with no direct path is a *quantised* source (int8 tensorwise, or
// a packed 4/6-bit family). Its shape is 5-D, which no packed family can hold, so
// the only legal decode is to float; fp16 is chosen because it is the widest
// format the kernel reads natively and it is exact for an int8 code times its
// fp32 scale only to fp16's 11 significand bits - which is exactly the precision
// of the conversion ComfyUI's own fp16 VAE runs at.

// The width code the kernel's `wDtype` takes, and the dtype a checkpoint of that
// precision must be uploaded as. -1 = "no native path" (see the note above).
int conv_weight_dtype(DType d) {
	switch (d) {
		case DType::F32: return 0;
		case DType::F16: return 1;
		case DType::BF16: return 2;
		default: return -1;
	}
}

// A conv weight in its own storage precision. Returns the device allocation and,
// in `*w_dtype`, the kernel code that decodes it. `upload_file_into` is used for
// the verbatim cases (see the note on the streaming path below).
GpuAlloc load_conv_weight(SafeTensors& st, const std::string& name, GpuCtx& g, GpuArena& arena,
                          int* w_dtype) {
	const StTensor& t = st.require(name);
	const int code = conv_weight_dtype(t.dtype);
	if (code < 0) {
		// A quantised / fp8 source: no native width, decode to fp16 (see the note at
		// the top of this section).
		*w_dtype = 1;
		std::vector<u8> b = st.materialize(t, DType::F16);
		GpuAlloc a = arena.alloc(b.size());
		g.upload_into(a, b.data(), b.size());
		return a;
	}
	*w_dtype = code;
	GpuAlloc a = arena.alloc(t.nbytes);
	g.upload_file_into(a, st, t);
	return a;
}

GpuAlloc load_f16(SafeTensors& st, const std::string& name, GpuCtx& g, GpuArena& arena) {
	const StTensor& t = st.require(name);
	// The fp16 checkpoint uploads verbatim; a source in any other precision
	// materialises to fp16 so the fp16 conv kernels read it unchanged.
	//
	// The verbatim case reads through `upload_file_into` rather than the mapping.
	// This is the decoder's streaming path - the blocks that did not fit the
	// resident cache are re-read once per 256 px tile - and a mapped read costs a
	// page fault per 4 KB against `read_file`'s 4 MB `ReadFile` served out of the
	// pinned host cache. Measured on this box: 2.5 GB/s mapped against ~10 GB/s
	// cached, on ~0.75 GB per tile, ~300 ms of a 1.5 s tile.
	if (st.is_raw(t) && t.dtype == DType::F16) {
		GpuAlloc a = arena.alloc(t.nbytes);
		g.upload_file_into(a, st, t);
		return a;
	}
	std::vector<u8> b = st.materialize(t, DType::F16);
	GpuAlloc a = arena.alloc(b.size());
	g.upload_into(a, b.data(), b.size());
	return a;
}

GpuAlloc load_f32(SafeTensors& st, const std::string& name, GpuCtx& g, GpuArena& arena) {
	const StTensor& t = st.require(name);
	std::vector<float> v = tensor_to_f32(st, t);
	GpuAlloc a = arena.alloc((u64)v.size() * 4);
	g.upload_into(a, v.data(), v.size() * 4);
	return a;
}

bool has_tensor(SafeTensors& st, const std::string& name) { return st.find(name) != nullptr; }

// PHI_VAE_I8=0 forces the int8 checkpoint's blocks back onto the fp16 path (each
// int8 matrix expanded to fp16 once per tile). It exists as the A/B that tells a
// precision change apart from a speed one, and as the fallback if a checkpoint's
// quantisation ever disagrees with the engine's.
bool i8_blocks_on() {
	static int v = -1;
	if (v < 0) {
		const char* e = getenv("PHI_VAE_I8");
		v = (e && *e && *e != '0') ? 0 : 1;
	}
	return v != 0;
}

// kind: 0 = plain conv3d, 1 = causal conv3d (pad on every axis),
// 2 = Downsample3D (temporal causal pad, reflect spatial pad, strides)
CConv load_conv(SafeTensors& st, const std::string& name, GpuCtx& g, GpuArena& arena, int kind,
                i64 time_stride, i64 space_stride) {
	CConv c;
	const StTensor& w = st.require(name + ".weight");
	// Any precision, as long as the shape is right. `load_conv_weight` uploads the
	// file's own storage width and hands back the kernel code that decodes it (see
	// the note on this section); a quantised source - which the 5-D shape proves
	// cannot be packed - decodes to fp16. The shape check is the real gate.
	if (w.shape.size() != 5) throw MediaError("h3vae: " + name + ".weight must be 5-D");
	c.oc = w.shape[0];
	c.ic = w.shape[1];
	c.kt = w.shape[2];
	c.kh = w.shape[3];
	c.kw = w.shape[4];
	c.w = load_conv_weight(st, name + ".weight", g, arena, &c.w_dtype);
	c.has_bias = has_tensor(st, name + ".bias");
	// The bias stays fp16: the kernel's `convBiasAt` reads 2-byte elements, and an
	// fp32 bias would need a second decode path in the epilogue for a few hundred
	// values. A bias is one value per output channel against kt*kh*kw*ic weights,
	// so this is the one place a fixed width costs nothing.
	if (c.has_bias) c.b = load_f16(st, name + ".bias", g, arena);
	if (kind == 0) {
		// A plain Conv3d (quant_conv / post_quant_conv / nin_shortcut): 1x1, no padding.
		c.strT = c.strS = 1;
		c.padT = c.padH = c.padHR = 0;
	} else if (kind == 1) {
		// CausalConv3d(kernel_size=3, padding=1): reflect-pad H and W by 1 on both
		// sides, zero-pad the front of time by kt-1, conv with padding=0.
		c.strT = c.strS = 1;
		c.padT = c.kt - 1;
		c.padH = c.padHR = (c.kh - 1) / 2;
	} else {
		// Downsample3D: reflect-pad H/W by 1 on the right/bottom only (space_stride
		// 2), then CausalConv3d(padding=(1,0,0), stride=(tstride, sstride, sstride)),
		// whose spatial padding is 0 - the pre-pad is the whole spatial pad.
		c.strT = time_stride;
		c.strS = space_stride;
		c.padT = c.kt - 1;
		c.padH = 0;
		c.padHR = (space_stride == 2) ? 1 : 0;
	}
	return c;
}

// The GEMMs go through the tensor-core kernel (gemm_f16_mma). PHI_VAE_MMA0=1
// forces the SIMT `gemm_f16` path instead - it exists so the two can be timed
// back to back on the same shapes, and is 8x slower. Same for PHI_VAE_ATTN0
// (the original attn3d).
bool mma_gemm_on() {
	static int v = -1;
	if (v < 0) {
		const char* e = getenv("PHI_VAE_MMA0");
		v = (e && *e && *e != '0') ? 0 : 1;
	}
	return v != 0;
}

// ── the runner ────────────────────────────────────────────────────────────

struct Runner {
	GpuCtx& g;
	ComputeContext& ctx;
	GpuArena& aa;
	u64* act_bytes;

	GpuAlloc alloc(i64 n_f32) {
		GpuAlloc a = aa.alloc((u64)n_f32 * 4);
		*act_bytes += (u64)n_f32 * 4;
		return a;
	}
	void elem(ElemOp op, const GpuAlloc& a, const GpuAlloc& b, const GpuAlloc& c, const GpuAlloc& y,
	          i64 rows, i64 cols, ElemMode bm = ElemMode::Flat, ElemMode cm = ElemMode::Flat) {
		ElemArgs e;
		e.op = op;
		e.a = a;
		e.b = b;
		e.c = c;
		e.y = y;
		e.rows = rows;
		e.cols = cols;
		e.bMode = bm;
		e.cMode = cm;
		dispatch_elem(ctx, e);
	}
	void silu_ip(const GpuAlloc& x, i64 n) { elem(ElemOp::Silu, x, x, x, x, 1, n); }
	void add(const GpuAlloc& a, const GpuAlloc& b, const GpuAlloc& y, i64 n) {
		elem(ElemOp::Add, a, b, b, y, 1, n);
	}
	void addmul(const GpuAlloc& a, const GpuAlloc& b, const GpuAlloc& scale, const GpuAlloc& y,
	            i64 rows, i64 cols) {
		elem(ElemOp::AddMul, a, b, scale, y, rows, cols, ElemMode::Flat, ElemMode::Row);
	}
	// The decoder's four per-block projections. These are 75% of a block's GPU
	// time, so they go through the tensor-core kernel whenever the shape allows
	// it (k a multiple of 16, 16-byte-aligned operands); everything else - the
	// 24-wide x_embedder above all - keeps the SIMT `gemm_f16`, whose fp32 A
	// operand is also what the one GEMM with k < 16 needs.
	void gemm(const GpuAlloc& a, const GpuAlloc& w, const GpuAlloc& bias, const GpuAlloc& c, i64 m,
	          i64 n, i64 k) {
		if (mma_gemm_on() && mma_gemm_shape_ok(m, n, k, a, w, c)) {
			// fp16 A for the mma path; the cast is one pass over the activation
			// (<= 30 MB) against a weight sweep a hundred times that.
			GpuAlloc ah = aa.alloc((u64)m * (u64)k * 2);
			*act_bytes += (u64)m * (u64)k * 2;
			dispatch_f32_to_f16(ctx, a, ah, m * k);
			dispatch_gemm_f16_mma(ctx, ah, w, bias, c, m, n, k);
			return;
		}
		GemmF16Args ga;
		ga.a = a;
		ga.b = w;
		ga.c = c;
		ga.bias = bias;
		ga.m = m;
		ga.n = n;
		ga.k = k;
		ga.a_is_f32 = true;
		ga.has_bias = (bool)bias;
		dispatch_gemm_f16(ctx, ga);
	}
	// The same projection against an int8 (convrot) weight: rotate + per-row
	// quantise the fp32 activation (`dispatch_quant_convrot`), contract it with the
	// int8 weight (`dispatch_int8_gemm`, which picks the tensor-core kernel), and
	// add the bias in the GEMM's own epilogue (`Int8GemmArgs::bias`). The bias is
	// fused rather than run as a second pass because the output is [S, N] fp32 and
	// at these shapes (N = 16384 for `w1`) re-reading and re-writing it is a
	// quarter of the stage. Kept apart from `gemm` rather than folded into it: the
	// fp16 path is what the fp16 checkpoint and every GEMM outside these 36 blocks
	// runs on, and the two do not share an operand layout.
	//
	// `k` must be a multiple of 256 (the convrot group) and of 32 (the int8 mma
	// k-step); the four decoder projections are 2048 and 8192, so they are.
	void gemm_i8(const GpuAlloc& a, const GpuAlloc& w, const GpuAlloc& ws, const GpuAlloc& bias,
	             const GpuAlloc& c, i64 m, i64 n, i64 k) {
		GpuAlloc q = aa.alloc((u64)m * (u64)k);
		*act_bytes += (u64)m * (u64)k;
		GpuAlloc s = aa.alloc((u64)m * 4);
		*act_bytes += (u64)m * 4;
		QuantConvrotArgs qa;
		qa.x = a.res;
		qa.x_offset = a.off;
		qa.q = q.res;
		qa.q_offset = q.off;
		qa.s = s.res;
		qa.s_offset = s.off;
		qa.rows = m;
		qa.K = k;
		dispatch_quant_convrot(ctx, qa);
		Int8GemmArgs ga;
		ga.a = q.res;
		ga.a_offset = q.off;
		ga.b = w.res;
		ga.b_offset = w.off;
		ga.sa = s.res;
		ga.sa_offset = s.off;
		ga.sb = ws.res;
		ga.sb_offset = ws.off;
		ga.c = c.res;
		ga.c_offset = c.off;
		ga.M = m;
		ga.N = n;
		ga.K = k;
		if (bias) {
			ga.bias = bias.res;
			ga.bias_offset = bias.off;
		}
		dispatch_int8_gemm(ctx, ga);
	}
	void norm(const GpuAlloc& x, const GpuAlloc& w, const GpuAlloc& b, const GpuAlloc& y, i64 rows,
	          i64 cols, bool layer, bool affine, bool has_bias, float eps) {
		NormArgs n;
		n.x = x;
		n.w = w;
		n.b = b;
		n.y = y;
		n.rows = rows;
		n.cols = cols;
		n.eps = eps;
		n.affine = affine;
		n.has_bias = has_bias;
		n.layer_norm = layer;
		dispatch_norm(ctx, n);
	}
};

// ── encoder ───────────────────────────────────────────────────────────────

struct EncOut {
	std::vector<float> v;
	i64 C = 0, T = 0, H = 0, W = 0;
};

// Temporal output length. With `padT = kt - 1` the padded length is t + kt - 1, so
// this is `(t - 1) / strT + 1` - which is also the single-frame answer (the
// reference's truncated weight convolves the unpadded frame: (t-1)/strT+1 = 1).
i64 conv_out_t(i64 t, const CConv& c) {
	return (t + c.padT - c.kt) / c.strT + 1;
}
// Spatial output length: `(padded length - kernel) / stride + 1` with the padded
// length being `s + padH + padHR`.
i64 conv_out_s(i64 s, const CConv& c) { return (s + c.padH + c.padHR - c.kh) / c.strS + 1; }

// EncoderFCN3D.forward (vae.py:156) on one clip. Returns the moments tensor.
// Forward declarations: the encode path is defined above the dump helpers it
// uses (the decode path has them the other way round).
bool vae_dump_on();
void dump_dev(GpuCtx& g, const GpuAlloc& a, i64 n_f32, const char* name);

// Defined with the tiled helpers below; declared here because the encode path needs the
// guarded copy (see its use in the input upload).
void upload_act(Runner& r, const GpuAlloc& a, const void* src, u64 bytes);

EncOut encoder_forward(Runner& r, const Encoder& e, const std::vector<float>& pix, i64 T, i64 H,
                       i64 W) {
	// The encoder's weights are loaded lazily on the first encode(), and that
	// load ends by submitting (upload_chunked) - so the dispatch bracket the
	// caller opened for the encode is closed again by the time the first
	// activation upload happens, and `upload_range` refuses to drop a copy with
	// no open bracket. Same guard `upload_act` uses; the decode path never needs
	// it because the decoder's weights are loaded at open().
	if (!r.ctx.recording()) r.ctx.begin();

	// ── one stage's scratch at a time, not the sum of the whole clip ────────
	//
	// Every stage below allocates its intermediates, consumes them once and is
	// done with them, so the arena is marked before the scratch and rewound
	// after it: only the stage's *result* outlives it, and that is allocated
	// before the mark. Without the rewind the bump allocator keeps the sum of
	// every stage of every residual block of every tile resident - 5.38 GB of
	// allocations to encode a 512x512 reference image at T=1, of which only
	// ~0.45 GB is live at any instant. A reference scaled up to a 540P canvas
	// walks twelve tiles of that, which is how the encode alone drove the VAE's
	// activation arena into the accountant's ceiling and killed the call with
	//
	//   VRAM exhausted: 256.00 MB for gpu_arena would take the process to
	//   5.08 GB of the 5.01 GB ceiling
	//
	// The kernels, their order and their operands are unchanged - only where
	// their scratch lives. The decode half has worked this way since the tiled
	// decoder rewrote the same failure (`decode_pixels`' per-tile mark and the
	// per-layer `loop_mark` in its 36-block loop).
	i64 t = conv_out_t(T, e.conv_in), h = conv_out_s(H, e.conv_in), w = conv_out_s(W, e.conv_in);
	GpuAlloc cur = r.alloc(e.conv_in.oc * t * h * w);
	const GpuArena::Mark in_mark = r.aa.mark();
	GpuAlloc xin = r.alloc(3 * T * H * W);
	// `upload_act`, not a bare `upload_range`: everything above this line can
	// *close* the bracket, because a chunk the arena has to take (or the encoder's
	// lazy weight load on the first call) submits the stream before it commits.
	// The T=1 case never allocates a new chunk for the tile it is about to upload,
	// so a bare copy looked safe - until the second encode() of a clip long enough
	// to need one, where it failed with "no open dispatch bracket". The guard is
	// the same one `upload_act` and the encoder's weight load already use.
	upload_act(r, xin, pix.data(), (u64)pix.size() * 4);
	// PHI_VAE_DUMP also covers the encode half: the ref2va path's reference
	// latent is the only consumer of it, and it had no oracle at all.
	if (vae_dump_on()) {
		r.ctx.submit();
		dump_dev(r.g, xin, 3 * T * H * W, "enc_in");
		dump_dev(r.g, cur, (i64)(e.conv_in.oc * t * h * w), "enc_conv_in");
		r.g.ring->rewind();
		if (!r.ctx.recording()) r.ctx.begin();
	}
	r.aa.rewind_to(in_mark);   // the input tile is dead as soon as conv_in has read it

	for (size_t lev = 0; lev < e.blocks.size(); lev++) {
		for (size_t bi = 0; bi < e.blocks[lev].size(); bi++) {
			const CRes& blk = e.blocks[lev][bi];
			const i64 icc = blk.c1.ic, occ = blk.c1.oc;
			const i64 o1t = conv_out_t(t, blk.c1);
			// This block's result is what the next block reads: it is allocated
			// first, everything the block needs to compute it goes into the
			// rewound region below it.
			GpuAlloc out = r.alloc(occ * o1t * h * w);
			const GpuArena::Mark blk_mark = r.aa.mark();
			GpuAlloc nrm = r.alloc(icc * t * h * w);
			dispatch_causal_gn(r.ctx, blk.n1, cur, nrm, icc, t, h, w);
			r.silu_ip(nrm, icc * t * h * w);
			GpuAlloc h1 = r.alloc(occ * o1t * h * w);
			dispatch_conv3d_causal(r.ctx, blk.c1, nrm, h1, t, h, w, o1t, h, w);

			GpuAlloc n2 = r.alloc(occ * o1t * h * w);
			dispatch_causal_gn(r.ctx, blk.n2, h1, n2, occ, o1t, h, w);
			r.silu_ip(n2, occ * o1t * h * w);
			GpuAlloc h2 = r.alloc(occ * o1t * h * w);
			dispatch_conv3d_causal(r.ctx, blk.c2, n2, h2, o1t, h, w, o1t, h, w);

			GpuAlloc res;
			if (icc != occ || o1t != t) {
				res = r.alloc(occ * o1t * h * w);
				if (icc != occ) dispatch_conv3d_causal(r.ctx, blk.nin, cur, res, t, h, w, o1t, h, w);
				else
					r.elem(ElemOp::Copy, cur, cur, cur, res, 1, occ * o1t * h * w);
			} else {
				res = cur;   // allocated above the mark: still live here
			}
			r.add(res, h2, out, occ * o1t * h * w);
			if (vae_dump_on()) {
				char nm[48];
				snprintf(nm, sizeof nm, "enc_L%zu_B%zu", lev, bi);
				r.ctx.submit();
				dump_dev(r.g, out, (i64)(occ * o1t * h * w), nm);
				r.g.ring->rewind();
				if (!r.ctx.recording()) r.ctx.begin();
			}
			r.aa.rewind_to(blk_mark);
			cur = out;
			t = o1t;
		}
		if (e.has_down[lev]) {
			const CConv& dc = e.downs[lev];
			const i64 ot = conv_out_t(t, dc), oh = conv_out_s(h, dc), ow = conv_out_s(w, dc);
			GpuAlloc dn = r.alloc(dc.oc * ot * oh * ow);
			dispatch_conv3d_causal(r.ctx, dc, cur, dn, t, h, w, ot, oh, ow);
			cur = dn;
			t = ot;
			h = oh;
			w = ow;
			if (vae_dump_on()) {
				char nm[48];
				snprintf(nm, sizeof nm, "enc_D%zu", lev);
				r.ctx.submit();
				dump_dev(r.g, dn, (i64)(dc.oc * ot * oh * ow), nm);
				r.g.ring->rewind();
				if (!r.ctx.recording()) r.ctx.begin();
			}
		}
	}

	const i64 oc = e.conv_out.ic;

	const i64 ot = conv_out_t(t, e.conv_out), oh = conv_out_s(h, e.conv_out),
	          ow = conv_out_s(w, e.conv_out);
	EncOut out;
	out.C = e.conv_out.oc;
	out.T = ot;
	out.H = oh;
	out.W = ow;
	out.v.resize((size_t)(ot * oh * ow * e.conv_out.oc));
	GpuAlloc dst = r.alloc(e.conv_out.oc * ot * oh * ow);
	const GpuArena::Mark tail_mark = r.aa.mark();
	GpuAlloc normed = r.alloc(oc * t * h * w);
	dispatch_causal_gn(r.ctx, e.norm_out, cur, normed, oc, t, h, w);
	r.silu_ip(normed, oc * t * h * w);
	dispatch_conv3d_causal(r.ctx, e.conv_out, normed, dst, t, h, w, ot, oh, ow);
	if (vae_dump_on()) {
		r.ctx.submit();
		dump_dev(r.g, normed, (i64)(oc * t * h * w), "enc_norm_out");
		dump_dev(r.g, dst, (i64)(e.conv_out.oc * ot * oh * ow), "enc_conv_out");
		r.g.ring->rewind();
		if (!r.ctx.recording()) r.ctx.begin();
	}
	r.ctx.submit();
	r.g.ring->rewind();
	r.ctx.download(dst.res, dst.off, out.v.data(), out.v.size() * 4);
	r.aa.rewind_to(tail_mark);   // the moments are on the host from here on
	return out;
}

// ── decoder ───────────────────────────────────────────────────────────────

BlockW load_block(SafeTensors& st, GpuCtx& g, GpuArena& arena, const std::string& p) {
	BlockW b;
	// The precision the checkpoint stores the block's first projection in decides
	// the whole block: a convrot int8 checkpoint has all four as int8 + a per-row
	// scale, an fp16 one has all four as fp16. `upload_quant_linear_i8` is the
	// engine's own int8 loader - it requantises whatever the file holds into the
	// rotated int8 bus the GEMM reads, caches the result, and is what the H3 DiT
	// already streams its blocks through.
	b.i8 = st.require(p + ".attn.to_qkv.weight").dtype == DType::I8 && i8_blocks_on();
	if (b.i8) {
		I8Upload qkv = upload_quant_linear_i8(g, st, p + ".attn.to_qkv", arena, nullptr);
		I8Upload out = upload_quant_linear_i8(g, st, p + ".attn.to_out", arena, nullptr);
		I8Upload w1 = upload_quant_linear_i8(g, st, p + ".ff.w1", arena, nullptr);
		I8Upload w2 = upload_quant_linear_i8(g, st, p + ".ff.w2", arena, nullptr);
		b.qkv_q = qkv.w;
		b.qkv_s = qkv.s;
		b.out_q = out.w;
		b.out_s = out.s;
		b.w1_q = w1.w;
		b.w1_s = w1.s;
		b.w2_q = w2.w;
		b.w2_s = w2.s;
	} else {
		b.qkv_w = load_f16(st, p + ".attn.to_qkv.weight", g, arena);
		b.out_w = load_f16(st, p + ".attn.to_out.weight", g, arena);
		b.w1_w = load_f16(st, p + ".ff.w1.weight", g, arena);
		b.w2_w = load_f16(st, p + ".ff.w2.weight", g, arena);
	}
	b.qkv_b = load_f32(st, p + ".attn.to_qkv.bias", g, arena);
	b.out_b = load_f32(st, p + ".attn.to_out.bias", g, arena);
	b.w1_b = load_f32(st, p + ".ff.w1.bias", g, arena);
	b.w2_b = load_f32(st, p + ".ff.w2.bias", g, arena);
	b.n1 = load_f32(st, p + ".norm1.weight", g, arena);
	b.n2 = load_f32(st, p + ".norm2.weight", g, arena);
	b.s1 = load_f32(st, p + ".scale1", g, arena);
	b.s2 = load_f32(st, p + ".scale2", g, arena);
	return b;
}

// `_decode_pixels` (vae.py:392) for one tile: post_quant_conv -> ViT3DDecoder ->
// unpatchify. Returns the raw (pre-finalize) pixels [3, T*4, H*16, W*16].
bool vae_dump_on() {
	static int v = -1;
	if (v < 0) {
		const char* e = getenv("PHI_VAE_DUMP");
		v = (e && *e && *e != '0') ? 1 : 0;
	}
	return v != 0;
}

// PHI_VAE_TIME: per-stage attribution inside the decoder's 36-block loop.
bool vae_time_on() {
	static int v = -1;
	if (v < 0) {
		const char* e = getenv("PHI_VAE_TIME");
		v = (e && *e && *e != '0') ? 1 : 0;
	}
	return v != 0;
}

// The decoder's attention, on the tensor cores (see `dispatch_attn3d_tc`).
// PHI_VAE_ATTN_TC=0 puts it back on the fp32 kernel - the A/B switch, and how a
// precision change is told apart from a speed one.
bool vae_attn_tc() {
	static int v = -1;
	if (v < 0) {
		const char* e = getenv("PHI_VAE_ATTN_TC");
		v = (e && *e && *e != '0') ? 0 : 1;
	}
	return v != 0;
}

// The decoder's attention kernel: 2 = attn3d_v2 (default), 1 = the original
// attn3d. PHI_VAE_ATTN=<n> picks one, which is how the two are held against
// each other on real tensors; 1 is not a supported setting.
int attn_impl() {
	static int v = -1;
	if (v < 0) {
		const char* e = getenv("PHI_VAE_ATTN");
		v = (e && *e) ? atoi(e) : 2;
		if (v != 1 && v != 2) v = 2;
	}
	return v;
}

// Writes one activation to build/phi_stage_<name>.f32 for the reference diff.
void dump_dev(GpuCtx& g, const GpuAlloc& a, i64 n_f32, const char* name) {
	if (!vae_dump_on()) return;
	const std::vector<float> v = g.download_f32(a, (u64)n_f32);
	const std::string path = std::string("C:/Phi/build/phi_stage_") + name + ".f32";
	if (FILE* f = fopen(path.c_str(), "wb")) {
		fwrite(v.data(), 4, v.size(), f);
		fclose(f);
	}
	double mn = 1e30, mx = -1e30;
	for (float x : v) {
		mn = std::min(mn, (double)x);
		mx = std::max(mx, (double)x);
	}
	fprintf(stderr, "[h3vae] dump %-10s %lld floats  [%.4f, %.4f]\n", name, (long long)v.size(), mn,
	        mx);
}

// silu(gate) * value for the fused [rows, 2*inner] buffer FeedForward.w1 emits.
//
// This is the one place in the decoder where an operand is a *stride* into a
// multi-row buffer instead of a contiguous slice: the shared `elem` op reads
// each operand flat from its own base, which pairs row 0 correctly and every
// other row with the wrong half (see kernels/conv3d_causal.cpp). It gets its
// own kernel, and it is the same pairing the DiT gets for free because there
// each row chunk is a separate buffer.

std::vector<float> decode_pixels(SafeTensors& st, Impl& im, Runner& r, const std::vector<float>& zt,
                                 i64 T, i64 H, i64 W) {
	const i64 nz = 24 * T * H * W;
	if ((i64)zt.size() != nz) throw MediaError("h3vae: bad tile latent size");

	// one tile at a time: the activation arena is reused, which keeps the peak
	// at one tile's worth of activations instead of the whole video's
	r.ctx.submit_if_recording();
	r.g.ring->rewind();
	r.aa.reset();
	*r.act_bytes = 0;
	r.ctx.begin();

	GpuAlloc zi = r.alloc(nz);
	upload_range(r.ctx, *r.g.ring, zi.res, zi.off, zt.data(), (u64)nz * 4);

	// post_quant_conv is Conv3d(24,24,1)
	GpuAlloc pq = r.alloc(nz);
	dispatch_conv3d_causal(r.ctx, im.post_quant, zi, pq, T, H, W, T, H, W);
	if (vae_dump_on()) {
		r.ctx.submit();
		dump_dev(r.g, pq, nz, "post_quant");
		r.ctx.begin();
	}

	const i64 S0 = T * H * W;
	const i64 S = S0 + 1 + kRegTokens;
	const i64 Sp = round_up(S, 32);

	GpuAlloc tok = r.alloc(S0 * 24);
	dispatch_transpose_cs(r.ctx, pq, tok, 24, S0, true);
	GpuAlloc h = r.alloc(Sp * kDim);
	r.gemm(tok, im.dec.x_embed_w, im.dec.x_embed_b, h, S0, kDim, 24);
	if (vae_dump_on()) {
		r.ctx.submit();
		dump_dev(r.g, tok, S0 * 24, "tokens");
		dump_dev(r.g, h, S0 * kDim, "x_embed");
		r.ctx.begin();
	}

	{
		// register tokens (4) followed by one zero row (vae.py:293)
		std::vector<float> suffix((size_t)((kRegTokens + 1) * kDim), 0.0f);
		std::vector<float> rv = tensor_to_f32(st, st.require("decoder.register_tokens"));
		if ((i64)rv.size() != kRegTokens * kDim)
			throw MediaError("h3vae: unexpected register_tokens shape");
		memcpy(suffix.data(), rv.data(), rv.size() * 4);
		upload_range(r.ctx, *r.g.ring, h.res, h.off + (u64)S0 * kDim * 4, suffix.data(),
		             suffix.size() * 4);
	}

	// 3-D ids: create_token_ids -> 2*(i+0.5)/n - 1, zero for the suffix rows
	std::vector<float> ids((size_t)S * 3, 0.0f);
	for (i64 t = 0; t < T; t++)
		for (i64 y = 0; y < H; y++)
			for (i64 x = 0; x < W; x++) {
				const i64 s = (t * H + y) * W + x;
				ids[(size_t)(s * 3 + 0)] = 2.0f * ((float)t + 0.5f) / (float)T - 1.0f;
				ids[(size_t)(s * 3 + 1)] = 2.0f * ((float)y + 0.5f) / (float)H - 1.0f;
				ids[(size_t)(s * 3 + 2)] = 2.0f * ((float)x + 0.5f) / (float)W - 1.0f;
			}
	GpuAlloc ids_dev = r.alloc(S * 3);
	upload_range(r.ctx, *r.g.ring, ids_dev.res, ids_dev.off, ids.data(), ids.size() * 4);

	GpuAlloc q = r.alloc(Sp * kDim), k = r.alloc(Sp * kDim), v = r.alloc(Sp * kDim);
	GpuAlloc attn = r.alloc(Sp * kDim), hn = r.alloc(Sp * kDim), o2 = r.alloc(Sp * kDim);
	const i64 inner = 4 * kDim;

	// Everything this loop allocates is per-block scratch: a block's result goes into
	// `h` (or into q/k/v above), and the fused qkv, the gated FFN intermediate and
	// the two GEMM outputs are dead as soon as the next layer starts. Without this
	// mark/rewind the activation arena grows to the *sum* of all 36 blocks instead
	// of their maximum — 4.2 GB at a 320x512 / 5-frame latent, which is what the
	// first real decode on this machine reported as "VRAM exhausted" (the same
	// failure the Flux AE had).
	const GpuArena::Mark loop_mark = r.aa.mark();   // PROF
	double prof[16] = {0};
	const bool pf = vae_time_on();
	double p_last = now_ms();
	int pi = 0;
	auto p_stage = [&]() {
		if (!pf) return;
		r.ctx.submit();
		double n = now_ms();
		prof[pi] += n - p_last;
		pi++;
		p_last = n;
		r.ctx.begin();
	};

	// ── the decoder stack, resident across tiles ────────────────────────────
	//
	// One greedy fill, on the first tile of the first decode; every later tile
	// (and every later decode, until `release_activation_memory` hands the
	// device back) reads its block addresses straight out of `dec_w`. The fill
	// itself happens at the *end* of `decode_pixels` below, where one tile's
	// activations are still standing: that is the exact footprint the stack has
	// to leave room for, so the budget it grows against is the real one rather
	// than an estimate. A refusal is a stop, not a failure - a card that cannot
	// hold the whole stack keeps what it can and streams the rest as before.
	if (!im.dec_tried) {
		im.dec_tried = true;
		const char* off = getenv("PHI_VAE_RES0");
		if (off && *off && *off != '0') {
			fprintf(stderr, "[h3vae] decoder residency disabled (PHI_VAE_RES0)\n");
		} else {
			im.dec_w.assign((size_t)kLayers, BlockW{});
			// What one block costs, measured off the checkpoint's shapes rather
			// than guessed: 3/4 of it is the four fp16 matrices.
			u64 blk = 0;
			for (const char* nm : {".attn.to_qkv.weight", ".attn.to_out.weight",
			                      ".ff.w1.weight", ".ff.w2.weight"}) {
				blk += st.require(im.block_prefix[0] + nm).nbytes;
			}
			blk += (u64)(3 * kDim + kDim + 2 * kDim) * 4 * 4;   // biases + norms + scales
			im.dec_blk = blk;
			// One block per chunk (plus a megabyte for its norms and biases, which
			// are interleaved with the matrices in `load_block`'s order). The chunk
			// is the *granularity at which the accountant is asked*, so a four-block
			// chunk made the growth stall half a gigabyte early: at 536 MB chunks
			// the stack stopped at 25 of the 36 blocks with 1.3 GB of the budget
			// unspent. At 135 MB it reaches ~28, and each block that stops streaming
			// is 134 MB of file per tile, 100+ tiles per clip.
			im.wcache.init(&r.ctx, blk + (1ull << 20));
			im.wcache.set_tag("h3vae.weights");
		}
	}

	// ── one block ahead on the disk (see SafeTensors::prefetch_ranges) ──────
	//
	// The blocks past `dec_n` are re-read off the file once per tile, and the
	// read used to sit between the previous block's compute and this block's -
	// 134 MB of dead time per block, ~300 ms of a 1.5 s tile. Issued here it runs
	// on a worker thread while the *previous* block computes, and `read_file`
	// serves the loader straight out of the slot. Only the streamed tail is
	// prefetched: a resident block is not read at all.
	auto prefetch_block = [&](i64 l) {
		if (l < im.dec_n || l >= kLayers || im.block_ranges.empty()) return;
		st.prefetch_ranges(im.block_ranges[(size_t)l]);
	};
	st.prefetch_wait();          // nothing of ours is in flight yet
	prefetch_block(im.dec_n);   // the first block that has to come off the disk

	for (i64 layer = 0; layer < kLayers; layer++) {
		r.aa.rewind_to(loop_mark);
		r.ctx.submit();
		if (pf) {
			double n = now_ms();
			prof[0] += n - p_last;
			p_last = n;
			pi = 1;
		}
		// A resident block costs nothing here: it was uploaded once on the first
		// tile, and the arena it lives in is never rewound.
		BlockW streamed_w;
		const bool cached = layer < im.dec_n;
		if (!cached) {
			// The read for this block was issued by the previous iteration (or by
			// the prologue above); take it before the loader asks for the bytes.
			st.prefetch_wait();
			r.g.ring->rewind();
			im.stream.reset();
			r.ctx.begin();
			streamed_w = load_block(st, r.g, im.stream, im.block_prefix[(size_t)layer]);
			// This block's bytes are on the device now, so the slot is free: start
			// the read for the next streamed block before this one's compute.
			prefetch_block(layer + 1);
		} else {
			r.ctx.begin();
		}
		const BlockW& bw = cached ? im.dec_w[(size_t)layer] : streamed_w;
		// The four per-block projections. Which kernel runs - the fp16 mma one or
		// the int8 one - is the checkpoint's own precision (`BlockW::i8`), so an
		// int8 file is decoded *as* int8 instead of being expanded back to fp16.
		auto proj = [&](const GpuAlloc& a, const GpuAlloc& w16, const GpuAlloc& wq,
		                const GpuAlloc& wsc, const GpuAlloc& bias, const GpuAlloc& c, i64 m,
		                i64 n, i64 k) {
			if (bw.i8) r.gemm_i8(a, wq, wsc, bias, c, m, n, k);
			else r.gemm(a, w16, bias, c, m, n, k);
		};
		if (pf) {
			double n = now_ms();
			prof[1] += n - p_last;
			p_last = n;
			pi = 2;
		}

		// x = x + attn(norm1(x)) * scale1
		r.norm(h, bw.n1, GpuAlloc{}, hn, S, kDim, false, true, false, kAttnEps);
		p_stage();
		if (vae_dump_on() && layer == 0) {
			r.ctx.submit();
			dump_dev(r.g, hn, S * kDim, "b_n1");
			r.ctx.begin();
		}
		GpuAlloc qkv = r.alloc(Sp * 3 * kDim);
		proj(hn, bw.qkv_w, bw.qkv_q, bw.qkv_s, bw.qkv_b, qkv, S, 3 * kDim, kDim);
		p_stage();
		if (vae_dump_on() && layer == 0) {
			r.ctx.submit();
			dump_dev(r.g, qkv, S * 3 * kDim, "b_qkv");
			r.ctx.begin();
		}
		dispatch_qkv_split3d(r.ctx, qkv, q, k, v, S);
		p_stage();
		if (vae_dump_on() && layer == 0) {
			r.ctx.submit();
			dump_dev(r.g, q, S * kDim, "b_q");
			dump_dev(r.g, k, S * kDim, "b_k");
			dump_dev(r.g, v, S * kDim, "b_v");
			r.ctx.begin();
		}
		dispatch_rope3d(r.ctx, q, ids_dev, im.inv_freq_dev, S);
		p_stage();
		if (vae_dump_on() && layer == 0) {
			r.ctx.submit();
			dump_dev(r.g, q, S * kDim, "b_qn");
			dump_dev(r.g, k, S * kDim, "b_kn");
			r.ctx.begin();
		}
		dispatch_rope3d(r.ctx, k, ids_dev, im.inv_freq_dev, S);
		p_stage();
		if (vae_dump_on() && layer == 0) {
			r.ctx.submit();
			dump_dev(r.g, q, S * kDim, "b_qr");
			dump_dev(r.g, k, S * kDim, "b_kr");
			r.ctx.begin();
		}
		// The tensor-core path needs packed fp16 operands; the cast is one pass
		// over S*2048 elements per tensor against 36 layers of attention.
		if (vae_attn_tc()) {
			GpuAlloc qh = r.alloc(Sp * kDim * 2), kh = r.alloc(Sp * kDim * 2),
			        vh = r.alloc(Sp * kDim * 2);
			dispatch_f32_to_f16(r.ctx, q, qh, Sp * kDim);
			dispatch_f32_to_f16(r.ctx, k, kh, Sp * kDim);
			dispatch_f32_to_f16(r.ctx, v, vh, Sp * kDim);
			dispatch_attn3d_tc(r.ctx, qh, kh, vh, attn, S);
		} else {
			dispatch_attn3d(r.ctx, q, k, v, attn, S, attn_impl());
		}
		p_stage();
		if (vae_dump_on() && layer == 0) {
			r.ctx.submit();
			dump_dev(r.g, attn, S * kDim, "b_attn");
			r.ctx.begin();
		}
		proj(attn, bw.out_w, bw.out_q, bw.out_s, bw.out_b, o2, S, kDim, kDim);
		p_stage();
		if (vae_dump_on() && layer == 0) {
			r.ctx.submit();
			dump_dev(r.g, o2, S * kDim, "b_o");
			r.ctx.begin();
		}
		r.addmul(h, o2, bw.s1, h, S, kDim);
		p_stage();
		if (vae_dump_on() && layer == 0) {
			r.ctx.submit();
			dump_dev(r.g, h, S * kDim, "b_x1");
			r.ctx.begin();
		}

		// x = x + ff(norm2(x)) * scale2 ; w1 emits [gate | value]
		r.norm(h, bw.n2, GpuAlloc{}, hn, S, kDim, false, true, false, kAttnEps);
		p_stage();
		if (vae_dump_on() && layer == 0) {
			r.ctx.submit();
			dump_dev(r.g, hn, S * kDim, "b_n2");
			r.ctx.begin();
		}
		GpuAlloc gx = r.alloc(Sp * 2 * inner);
		proj(hn, bw.w1_w, bw.w1_q, bw.w1_s, bw.w1_b, gx, S, 2 * inner, kDim);
		p_stage();
		if (vae_dump_on() && layer == 0) {
			r.ctx.submit();
			dump_dev(r.g, gx, S * 2 * inner, "b_w1");
			r.ctx.begin();
		}
		GpuAlloc gated = r.alloc(Sp * inner);
		dispatch_silu_gate_fused(r.ctx, gx, gated, S, inner);
		p_stage();
		if (vae_dump_on() && layer == 0) {
			r.ctx.submit();
			dump_dev(r.g, gated, S * inner, "b_gated");
			r.ctx.begin();
		}
		GpuAlloc ff = r.alloc(Sp * kDim);
		proj(gated, bw.w2_w, bw.w2_q, bw.w2_s, bw.w2_b, ff, S, kDim, inner);
		p_stage();
		if (vae_dump_on() && layer == 0) {
			r.ctx.submit();
			dump_dev(r.g, ff, S * kDim, "b_ff");
			r.ctx.begin();
		}
		r.addmul(h, ff, bw.s2, h, S, kDim);
		p_stage();
		if (vae_dump_on() && layer < 2) {
			r.ctx.submit();
			dump_dev(r.g, h, S * kDim, layer == 0 ? "block0" : "block1");
			r.ctx.begin();
		}
		// the next iteration submits, rewinds and resets `im.stream` at its top,
		// which is what makes the block's weight buffer reusable
	}

	if (pf) {
		r.ctx.submit();
		double tot = 0;
		for (int i = 0; i < 15; i++) tot += prof[i];
		static const char* nm[15] = {"prev-tail", "wload", "norm1", "qkv", "split", "ropeq",
		                             "ropek", "attn", "outproj", "add1", "norm2", "w1",
		                             "silu", "w2", "add2"};
		printf("[h3vae-time] tile S=%lld Sp=%lld layer=%.2f ms\n", (long long)S, (long long)Sp,
		       tot / (double)kLayers);
		for (int i = 0; i < 15; i++)
			printf("    %-10s %8.1f ms  %5.1f%%\n", nm[i], prof[i], 100.0 * prof[i] / tot);
		p_last = now_ms();
		r.ctx.begin();
	}

	// norm_out + proj_out, only for the real patches (both are row-wise, so this
	// is the same number as the reference's post-hoc slicing)
	GpuAlloc nh = r.alloc(S0 * kDim);
	r.norm(h, im.dec.norm_out_w, im.dec.norm_out_b, nh, S0, kDim, true, true, true, kAttnEps);
	GpuAlloc proj = r.alloc(S0 * 3 * kPatchT * kPatch * kPatch);
	r.gemm(nh, im.dec.proj_out_w, im.dec.proj_out_b, proj, S0, 3 * kPatchT * kPatch * kPatch, kDim);
	if (vae_dump_on()) {
		r.ctx.submit();
		dump_dev(r.g, nh, S0 * kDim, "norm_out");
		dump_dev(r.g, proj, S0 * 3 * kPatchT * kPatch * kPatch, "proj_out");
		r.ctx.begin();
	}

	const i64 OT = T * kPatchT, OH = H * kPatch, OW = W * kPatch;
	GpuAlloc rgb = r.alloc(3 * OT * OH * OW);
	dispatch_unpatch3d(r.ctx, proj, rgb, T, H, W);

	// ── grow the resident decoder stack (see `Impl::wcache`) ───────────────
	//
	// This runs with the whole tile still standing in the activation arena, so
	// the check is "would one more block fit *next to a tile*", which is the
	// only question that matters: the next tile is the same shape and its
	// allocations are the same ones. A refusal stops the growth (never the
	// decode), and the stack that did fit is reused by every remaining tile -
	// 100+ of them for a 540P clip.
	if (!im.dec_w.empty() && im.dec_n < kLayers) {
			const u64 limit = vram_budget().limit();
			// One more block, plus one `wcache` chunk, plus a little slack. A refusal
			// from the accountant below stops the growth too - this is only the
			// cheap check that keeps the common case out of the exception path.
			const u64 sanity = 192ull << 20;
			while (im.dec_n < kLayers) {
				if (limit && vram_budget().local() + (im.dec_blk + (1ull << 20)) + sanity > limit)
					break;
			try {
				im.dec_w[(size_t)im.dec_n] =
				    load_block(st, r.g, im.wcache, im.block_prefix[(size_t)im.dec_n]);
				im.dec_n++;
			} catch (const std::exception& e) {
				fprintf(stderr, "[h3vae] resident decoder stopped at %lld/%lld blocks (%s)\n",
				        (long long)im.dec_n, (long long)kLayers, e.what());
				break;
			}
		}
		if (im.dec_n > im.dec_logged) {
			im.dec_logged = im.dec_n;
			fprintf(stderr,
			        "[h3vae] decoder weights resident: %lld/%lld blocks, %.2f GB%s\n",
			        (long long)im.dec_n, (long long)kLayers,
			        (double)im.wcache.capacity() / 1073741824.0,
			        im.dec_n == kLayers ? " (every tile reuses them)" : "");
			if (getenv("PHI_VAE_PLAN"))
				fprintf(stderr,
				        "[h3vae] grow stop: n=%lld blk=%s local=%s limit=%s sanity=%s\n",
				        (long long)im.dec_n, format_bytes(im.dec_blk).c_str(),
				        format_bytes(vram_budget().local()).c_str(),
				        format_bytes(vram_budget().limit()).c_str(),
				        format_bytes(192ull << 20).c_str());
		}
	}
	// ── pin whatever did not fit, once ───────────────────────
	//
	// The blocks past `dec_n` are re-read off the file on *every* tile - 134 MB
	// each, 100+ tiles - and the file is 5 GB, so the OS page cache cannot hold
	// the working set and the reads land on an NVMe that delivers ~1.2 GB/s for
	// them. System memory delivers them at ~10 GB/s, and the same `pin_range`
	// policy the DiT uses applies: pin the fixed set that recurs, stream the
	// rest. The set is small precisely *because* the residency worked out - 9
	// blocks here, 1.2 GB - and it is what turns the per-tile re-read into a
	// memcpy. Skipped when the machine has no room for it; the reads then simply
	// stay cold.
	if (!im.dec_pinned && im.dec_n > 0 && im.dec_n < kLayers && !im.block_ranges.empty()) {
		im.dec_pinned = true;
		u64 want = 0;
		for (i64 l = im.dec_n; l < kLayers; l++)
			for (const auto& rr : im.block_ranges[(size_t)l]) want += rr.second;
		const u64 have = SafeTensors::host_free_bytes();
		if (have == 0 || want > have / 2) {
			fprintf(stderr, "[h3vae] %lld streamed blocks not pinned (%.2f GB wanted, %.2f GB free)\n",
			        (long long)(kLayers - im.dec_n), (double)want / 1073741824.0,
			        (double)have / 1073741824.0);
		} else {
			double t0 = now_ms();
			for (i64 l = im.dec_n; l < kLayers; l++) {
				for (const auto& rr : im.block_ranges[(size_t)l]) {
					try {
						st.pin_range(rr.first, rr.second);
					} catch (const std::exception&) {
						break;
					}
				}
			}
			fprintf(stderr,
			        "[h3vae] pinned %lld streamed blocks (%.2f GB) in %.1f s; the remaining tiles "
			        "read them out of RAM\n",
			        (long long)(kLayers - im.dec_n), (double)st.pinned_bytes() / 1073741824.0,
			        (now_ms() - t0) / 1000.0);
		}
	}

	std::vector<float> host((size_t)(3 * OT * OH * OW));
	r.ctx.submit();
	r.g.ring->rewind();
	r.ctx.download(rgb.res, rgb.off, host.data(), host.size() * 4);
	return host;
}

// ── host-side slicing helpers ─────────────────────────────────────────────

std::vector<float> slice_latent(const std::vector<float>& z, i64 T, i64 H, i64 W, i64 t0, i64 nt,
                                i64 h0, i64 nh, i64 w0, i64 nw) {
	std::vector<float> out((size_t)(24 * nt * nh * nw));
	for (i64 c = 0; c < 24; c++)
		for (i64 t = 0; t < nt; t++)
			for (i64 y = 0; y < nh; y++)
				for (i64 x = 0; x < nw; x++)
					out[(size_t)(((c * nt + t) * nh + y) * nw + x)] =
					    z[(size_t)(((c * T + t0 + t) * H + h0 + y) * W + w0 + x)];
	return out;
}

std::vector<float> slice_pixels(const std::vector<float>& src, i64 T, i64 H, i64 W, i64 h0,
                                i64 nh, i64 w0, i64 nw) {
	std::vector<float> out((size_t)(3 * T * nh * nw));
	for (i64 c = 0; c < 3; c++)
		for (i64 t = 0; t < T; t++)
			for (i64 y = 0; y < nh; y++)
				for (i64 x = 0; x < nw; x++)
					out[(size_t)(((c * T + t) * nh + y) * nw + x)] =
					    src[(size_t)(((c * T + t) * H + h0 + y) * W + w0 + x)];
	return out;
}

void upload_act(Runner& r, const GpuAlloc& a, const void* src, u64 bytes) {
	if (!r.ctx.recording()) r.ctx.begin();
	upload_range(r.ctx, *r.g.ring, a.res, a.off, src, bytes);
}

// ── tiled decode (vae.py:600) ─────────────────────────────────────────────
//
// Splits the *pixel* canvas into 256px tiles, decodes each (one tile = a full
// 36-layer transformer pass) and fades the seams. z is [24, T, H, W] and the
// result is [3, T*4, H*16, W*16] in raw decoder units.
//
// The order of the two blends and *what* each one is fed is the reference's and
// is load-bearing:
//
//   * the y blend takes the previous row's **assembled strip** (`strip`, a
//     canvas-width buffer already x-blended) sliced at this tile's own x-range,
//     not this column's own row tail. In the x-overlap band the strip holds the
//     *next* tile's blended contribution, because that is what was written to
//     the canvas there (vae.py:611).
//   * the x tail handed to the tile on the right is taken after the y blend and
//     after the x blend (vae.py:523).
//
// Taking both tails off the raw tile instead - which is what this did - leaves
// every tile-overlap band disagreeing with the reference: at 540P the overlap
// bands are 80 px wide (x) and 112 px tall (y) on a 256 px tile grid, i.e. >60 %
// of the frame, laid out exactly on the tile grid. That is the "grid-like"
// corruption the video tool produced on every frame.
std::vector<float> tiled_decode(SafeTensors& st, Impl& im, Runner& r, const std::vector<float>& z,
                                i64 T, i64 H, i64 W, i64 tile) {
	const i64 HP = H * kPatch, WP = W * kPatch, OT = T * kPatchT;
	const i64 tile_size = tile > 0 ? tile : kTileSize;
	const H3Tiles ty = h3_split_tiles(HP, tile_size, kTileOverlapMin, kPatch);
	const H3Tiles tx = h3_split_tiles(WP, tile_size, kTileOverlapMin, kPatch);
	const i64 nY = (i64)ty.len.size(), nX = (i64)tx.len.size();

	std::vector<float> canvas((size_t)(3 * OT * HP * WP), 0.0f);
	MatV cv = MatV::whole(canvas.data(), 3, OT, HP, WP);

	// The previous row's output assembled at canvas coordinates: [3, OT, strip_h, WP].
	// Every column of every row is written into it, so it is the canvas row *with*
	// the x blending already applied - which is what the next row's y blend reads.
	std::vector<float> strip;
	i64 strip_h = 0;
	// The x tail of the tile to this one's left: [3, OT, left_h, left_w].
	std::vector<float> left_tail;
	i64 left_h = 0, left_w = 0;

	i64 out_y = 0, keep_h_last = 0;
	for (i64 i = 0; i < nY; i++) {
		const i64 zi = ty.pos[i] / kPatch;
		const i64 zl = std::min(ty.len[i] / kPatch, H - zi);
		if (zl <= 0) throw MediaError("h3vae: empty tile row");
		std::vector<float> new_strip;
		i64 new_strip_h = 0;
		i64 out_x = 0;
		left_tail.clear();
		left_h = left_w = 0;
		for (i64 j = 0; j < nX; j++) {
			const i64 zj = tx.pos[j] / kPatch;
			const i64 zw = std::min(tx.len[j] / kPatch, W - zj);
			if (zw <= 0) throw MediaError("h3vae: empty tile column");
			std::vector<float> zt = slice_latent(z, T, H, W, 0, T, zi, zl, zj, zw);
			const GpuArena::Mark tile_mark = r.aa.mark();
			std::vector<float> tile = decode_pixels(st, im, r, zt, T, zl, zw);
			// Tiles share nothing but the host-side canvas, so a tile's activations are
			// given back before the next one starts; the per-layer rewind inside
			// decode_pixels bounds one tile, this bounds the cross-tile growth.
			r.aa.rewind_to(tile_mark);
			const i64 tH = zl * kPatch, tW = zw * kPatch;
			MatV tv = MatV::whole(tile.data(), 3, OT, tH, tW);

			// y: the strip's own x-range for this tile, i.e. [tx.pos[j], +tile width).
			if (i > 0) {
				const i64 x0 = tx.pos[j];
				const i64 xw = std::min(tx.len[j], WP - x0);
				MatV st = MatV::xslice(strip.data(), 3, OT, strip_h, WP, x0, xw);
				blend_axis(st, tv, ty.overlap[i - 1], 2);
			}
			// x: the left neighbour's tail.
			if (j > 0 && left_w > 0) {
				MatV lt = MatV::whole(left_tail.data(), 3, OT, left_h, left_w);
				blend_axis(lt, tv, tx.overlap[j - 1], 3);
			}
			// the tail the tile to the right will blend against (vae.py:523)
			if (j < nX - 1) {
				const i64 ov = tx.overlap[j];
				left_h = tH;
				left_w = ov;
				left_tail.assign((size_t)(3 * OT * tH * ov), 0.0f);
				MatV dst = MatV::whole(left_tail.data(), 3, OT, tH, ov);
				for (i64 c = 0; c < 3; c++)
					for (i64 t = 0; t < OT; t++)
						for (i64 y = 0; y < tH; y++)
							for (i64 x = 0; x < ov; x++)
								*dst.at(c, t, y, x) = *tv.at(c, t, y, tW - ov + x);
			} else {
				left_tail.clear();
				left_h = left_w = 0;
			}

			const i64 keepW = tW - ((j < nX - 1) ? tx.overlap[j] : 0);
			const i64 keepH = tH - ((i < nY - 1) ? ty.overlap[i] : 0);
			if (i < nY - 1) {
				// This row's strip: the tile's y tail (after both blends) at the canvas
				// position it will occupy, exactly like the reference's new_strip.
				const i64 ov = ty.overlap[i];
				if (new_strip_h == 0) {
					new_strip.assign((size_t)(3 * OT * ov * WP), 0.0f);
					new_strip_h = ov;
				}
				MatV ns = MatV::whole(new_strip.data(), 3, OT, ov, WP);
				for (i64 c = 0; c < 3; c++)
					for (i64 t = 0; t < OT; t++)
						for (i64 y = 0; y < ov; y++)
							for (i64 x = 0; x < keepW; x++)
								*ns.at(c, t, y, out_x + x) = *tv.at(c, t, tH - ov + y, x);
			}
			for (i64 c = 0; c < 3; c++)
				for (i64 t = 0; t < OT; t++)
					for (i64 y = 0; y < keepH; y++)
						for (i64 x = 0; x < keepW; x++)
							*cv.at(c, t, out_y + y, out_x + x) = *tv.at(c, t, y, x);
			out_x += keepW;
			keep_h_last = keepH;
		}
		strip = std::move(new_strip);
		strip_h = new_strip_h;
		out_y += keep_h_last;
	}
	return canvas;
}

// ── temporal chunking (vae.py:609) ────────────────────────────────────────
std::vector<float> decode_temporal(SafeTensors& st, Impl& im, Runner& r, const std::vector<float>& zn,
                                   i64 T, i64 H, i64 W, i64 tile) {
	const i64 tokens_chunk_size = ceil_div(kClipLength, kPatchT);   // 5
	const i64 token_overlap = pymod(-kTokenDrop, tokens_chunk_size);
	const i64 frame_pre_padding = pymod(-kClipLength, kPatchT);
	const i64 frame_overlap = std::max<i64>(token_overlap * kPatchT - frame_pre_padding, 0);
	const i64 chunk_dec = tokens_chunk_size * kPatchT;

	i64 pad_tokens = 0, num_chunks = 0;
	h3_decode_temporal_chunks(T, tokens_chunk_size, kTokenDrop, &pad_tokens, &num_chunks);

	const i64 Tp = T + pad_tokens;
	std::vector<float> zp((size_t)(24 * Tp * H * W));
	for (i64 c = 0; c < 24; c++) {
		memcpy(zp.data() + (size_t)c * Tp * H * W, zn.data() + (size_t)c * T * H * W,
		       (size_t)(T * H * W) * 4);
		for (i64 k = 0; k < pad_tokens; k++)
			memcpy(zp.data() + ((size_t)c * Tp + T + k) * H * W,
			       zn.data() + ((size_t)c * T + T - 1) * H * W, (size_t)H * W * 4);
	}

	const i64 HP = H * kPatch, WP = W * kPatch;
	const i64 frames = h3_decode_output_frames(Tp, pad_tokens, num_chunks, kClipLength,
	                                           tokens_chunk_size, kTokenDrop, kPatchT);
	std::vector<float> canvas((size_t)(3 * frames * HP * WP), 0.0f);
	MatV cv = MatV::whole(canvas.data(), 3, frames, HP, WP);

	i64 write_pos = 0;
	auto write_part = [&](const MatV& part) {
		const i64 pf = part.T;
		if (pf <= 0) return;
		const i64 copy_frames = std::min(pf, std::max<i64>(0, frames - write_pos));
		for (i64 c = 0; c < 3; c++)
			for (i64 t = 0; t < copy_frames; t++)
				for (i64 y = 0; y < HP; y++)
					for (i64 x = 0; x < WP; x++) {
						float v = *part.at(c, t, y, x) * kPixelStd[c] + kPixelMean[c];
						v = std::min(1.0f, std::max(0.0f, v));
						*cv.at(c, write_pos + t, y, x) = v;
					}
		write_pos += copy_frames;
	};

	std::vector<float> ov;
	i64 ov_T = 0;
	for (i64 i = 0; i < num_chunks; i++) {
		const i64 t_start = i * tokens_chunk_size;
		const i64 t_end = t_start + tokens_chunk_size + token_overlap;
		const i64 t_lo = std::min(t_start, Tp), t_hi = std::min(t_end, Tp);
		const i64 clip_n = std::max<i64>(0, t_hi - t_lo);
		std::vector<float> clip_z = slice_latent(zp, Tp, H, W, t_lo, clip_n, 0, H, 0, W);
		std::vector<float> clip = tiled_decode(st, im, r, clip_z, clip_n, H, W, tile);
		const i64 clip_frames = clip_n * kPatchT;

		for (i64 j = 0; j < 2; j++) {
			const i64 f_start = j * chunk_dec;
			const i64 f_end = std::min(f_start + chunk_dec, clip_frames);
			const i64 s = f_start + frame_pre_padding;
			const i64 pf = std::max<i64>(0, f_end - s);
			if (j == 0) {
				if (pf <= 0) continue;
				if (!ov.empty()) {
					// the overlap belongs to the previous chunk's tail; blend it into
					// this chunk's head and emit the merged frames
					std::vector<float> blended((size_t)(3 * pf * HP * WP));
					MatV pb = MatV::whole(blended.data(), 3, pf, HP, WP);
					MatV src = MatV::tslice(clip.data(), 3, clip_frames, s, pf, HP, WP);
					for (i64 c = 0; c < 3; c++)
						for (i64 t = 0; t < pf; t++)
							for (i64 y = 0; y < HP; y++)
								for (i64 x = 0; x < WP; x++)
									*pb.at(c, t, y, x) = *src.at(c, t, y, x);
					MatV oa = MatV::whole(ov.data(), 3, ov_T, HP, WP);
					blend_axis(oa, pb, frame_overlap, 1);
					write_part(pb);
					ov.clear();
					ov_T = 0;
				} else {
					MatV src = MatV::tslice(clip.data(), 3, clip_frames, s, pf, HP, WP);
					write_part(src);
				}
			} else {
				ov_T = pf;
				if (pf <= 0) {
					ov.clear();
					ov_T = 0;
				} else {
					ov.resize((size_t)(3 * pf * HP * WP));
					MatV dst = MatV::whole(ov.data(), 3, pf, HP, WP);
					MatV src = MatV::tslice(clip.data(), 3, clip_frames, s, pf, HP, WP);
					for (i64 c = 0; c < 3; c++)
						for (i64 t = 0; t < pf; t++)
							for (i64 y = 0; y < HP; y++)
								for (i64 x = 0; x < WP; x++)
									*dst.at(c, t, y, x) = *src.at(c, t, y, x);
				}
			}
		}
		if (i == num_chunks - 1 && !ov.empty()) {
			MatV oo = MatV::whole(ov.data(), 3, ov_T, HP, WP);
			write_part(oo);
			ov.clear();
			ov_T = 0;
		}
	}
	return canvas;
}

// ── encoder pipeline ─────────────────────────────────────────────────────

// _encode_moments: EncoderFCN3D then quant_conv (vae.py:389)
EncOut encode_moments(Impl& im, Runner& r, const std::vector<float>& pix, i64 T, i64 H, i64 W) {
	EncOut m = encoder_forward(r, im.enc, pix, T, H, W);
	GpuAlloc x = r.alloc(m.C * m.T * m.H * m.W);
	upload_act(r, x, m.v.data(), (u64)m.v.size() * 4);
	GpuAlloc y = r.alloc(im.quant.oc * m.T * m.H * m.W);
	dispatch_conv3d_causal(r.ctx, im.quant, x, y, m.T, m.H, m.W, m.T, m.H, m.W);
	EncOut out;
	out.C = im.quant.oc;
	out.T = m.T;
	out.H = m.H;
	out.W = m.W;
	out.v.resize((size_t)(out.C * out.T * out.H * out.W));
	r.ctx.submit();
	r.g.ring->rewind();
	r.ctx.download(y.res, y.off, out.v.data(), out.v.size() * 4);
	return out;
}

// tiled_encode (vae.py:473)
EncOut encode_tiled(Impl& im, Runner& r, const std::vector<float>& pix, i64 T, i64 H, i64 W) {
	const H3Tiles ty = h3_split_tiles(H, kTileSize, kTileOverlapMin, kPatch);
	const H3Tiles tx = h3_split_tiles(W, kTileSize, kTileOverlapMin, kPatch);
	const i64 nY = (i64)ty.len.size(), nX = (i64)tx.len.size();

	std::vector<std::vector<EncOut>> tiles((size_t)nY, std::vector<EncOut>((size_t)nX));
	for (i64 i = 0; i < nY; i++) {
		const i64 h0 = ty.pos[i], hn = std::min(ty.len[i], H - h0);
		for (i64 j = 0; j < nX; j++) {
			const i64 w0 = tx.pos[j], wn = std::min(tx.len[j], W - w0);
			std::vector<float> sub = slice_pixels(pix, T, H, W, h0, hn, w0, wn);
			// A tile is encoded and downloaded to the host before the next one
			// starts, so nothing it put in the arena is live afterwards: rewind
			// instead of letting the bump allocator book every tile's
			// intermediates (and, with `encode()`'s clip loop, every clip's)
			// until the arena runs out of the budget. This is the same
			// mark/rewind the tiled *decode* does per tile.
			const GpuArena::Mark tile_mark = r.aa.mark();
			tiles[(size_t)i][(size_t)j] = encode_moments(im, r, sub, T, hn, wn);
			r.aa.rewind_to(tile_mark);
		}
	}

	std::vector<i64> keepH((size_t)nY), keepW((size_t)nX);
	for (i64 i = 0; i < nY; i++)
		keepH[(size_t)i] = tiles[(size_t)i][0].H - ((i < nY - 1) ? ty.overlap[i] / kPatch : 0);
	for (i64 j = 0; j < nX; j++)
		keepW[(size_t)j] = tiles[0][(size_t)j].W - ((j < nX - 1) ? tx.overlap[j] / kPatch : 0);
	i64 TH = 0, TW = 0;
	for (i64 i = 0; i < nY; i++) TH += keepH[(size_t)i];
	for (i64 j = 0; j < nX; j++) TW += keepW[(size_t)j];

	EncOut out;
	out.C = tiles[0][0].C;
	out.T = tiles[0][0].T;
	out.H = TH;
	out.W = TW;
	out.v.assign((size_t)(out.C * out.T * TH * TW), 0.0f);
	MatV ov = MatV::whole(out.v.data(), out.C, out.T, TH, TW);

	i64 y0 = 0;
	for (i64 i = 0; i < nY; i++) {
		i64 x0 = 0;
		for (i64 j = 0; j < nX; j++) {
			EncOut& src = tiles[(size_t)i][(size_t)j];
			std::vector<float> work = src.v;   // blended in place, sources stay raw
			MatV tv = MatV::whole(work.data(), src.C, src.T, src.H, src.W);
			if (i > 0) {
				EncOut& up = tiles[(size_t)(i - 1)][(size_t)j];
				MatV u = MatV::whole(up.v.data(), up.C, up.T, up.H, up.W);
				blend_axis(u, tv, ty.overlap[i - 1] / kPatch, 2);
			}
			if (j > 0) {
				EncOut& lf = tiles[(size_t)i][(size_t)(j - 1)];
				MatV l = MatV::whole(lf.v.data(), lf.C, lf.T, lf.H, lf.W);
				blend_axis(l, tv, tx.overlap[j - 1] / kPatch, 3);
			}
			for (i64 c = 0; c < out.C; c++)
				for (i64 t = 0; t < out.T; t++)
					for (i64 y = 0; y < keepH[(size_t)i]; y++)
						for (i64 x = 0; x < keepW[(size_t)j]; x++)
							*ov.at(c, t, y0 + y, x0 + x) = *tv.at(c, t, y, x);
			x0 += keepW[(size_t)j];
		}
		y0 += keepH[(size_t)i];
	}
	return out;
}

// ── lazy encoder load (360 MB; only the ref2va path needs it) ─────────────
void load_encoder(SafeTensors& st, Impl& im, GpuCtx& g, GpuArena& arena) {
	Encoder& e = im.enc;
	e.conv_in = load_conv(st, "encoder.conv_in", g, arena, 1, 1, 1);
	e.blocks.resize((size_t)kEncLevels);
	e.downs.resize((size_t)kEncLevels);
	e.has_down.assign((size_t)kEncLevels, false);
	for (i64 lev = 0; lev < kEncLevels; lev++) {
		const std::string base = "encoder.down." + std::to_string(lev) + ".";
		i64 nb = 0;
		while (has_tensor(st, base + "block." + std::to_string(nb) + ".conv1.weight")) nb++;
		if (nb != 2) throw MediaError("h3vae: encoder level " + std::to_string(lev) + " has " +
		                              std::to_string(nb) + " res blocks, expected 2");
		for (i64 b = 0; b < nb; b++) {
			const std::string p = base + "block." + std::to_string(b);
			CRes r;
			r.n1 = CGn{load_f32(st, p + ".norm1.weight", g, arena),
			            load_f32(st, p + ".norm1.bias", g, arena)};
			r.n2 = CGn{load_f32(st, p + ".norm2.weight", g, arena),
			            load_f32(st, p + ".norm2.bias", g, arena)};
			r.c1 = load_conv(st, p + ".conv1", g, arena, 1, 1, 1);
			r.c2 = load_conv(st, p + ".conv2", g, arena, 1, 1, 1);
			r.has_nin = has_tensor(st, p + ".nin_shortcut.weight");
			if (r.has_nin) r.nin = load_conv(st, p + ".nin_shortcut", g, arena, 0, 1, 1);
			if ((r.c1.ic != r.c1.oc) != r.has_nin)
				throw MediaError("h3vae: " + p + " channel change does not match its shortcut");
			e.blocks[(size_t)lev].push_back(r);
		}
		const bool want_down = (kSpaceDown[lev] * kTimeDown[lev] > 1);
		const std::string dn = base + "downsample.conv";
		if (has_tensor(st, dn + ".weight") != want_down)
			throw MediaError("h3vae: encoder level " + std::to_string(lev) +
			                 " downsample presence does not match space_down/time_down");
		if (want_down) {
			e.has_down[(size_t)lev] = true;
			e.downs[(size_t)lev] = load_conv(st, dn, g, arena, 2, kTimeDown[lev], kSpaceDown[lev]);
		}
	}
	e.norm_out = CGn{load_f32(st, "encoder.norm_out.weight", g, arena),
	                 load_f32(st, "encoder.norm_out.bias", g, arena)};
	e.conv_out = load_conv(st, "encoder.conv_out", g, arena, 1, 1, 1);
	if (e.conv_out.oc != 48) throw MediaError("h3vae: encoder.conv_out is not 2*z_channels");
}

}  // namespace

// ── the decoder's attention (declared in video_vae_internal.hpp) ───────
// `impl` selects the implementation: 2 = attn3d_v2 (what the decoder runs),
// 1 = the original attn3d, which exists so the kernel test can hold the fast one
// against the slow one on the same tensors and is no longer on any engine path.
void dispatch_attn3d(ComputeContext& ctx, const GpuAlloc& q, const GpuAlloc& k, const GpuAlloc& v,
                     const GpuAlloc& o, i64 S, int impl) {
	const char* src = impl == 2 ? attn3d_v2_hlsl() : attn3d_hlsl();
	const char* key = impl == 2 ? "h3vae_attn3d_v2" : "h3vae_attn3d";
	const char* entry = impl == 2 ? "attn3d_v2" : "attn3d";
	GpuKernel* pso = ctx.pipeline(key, src, entry, ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)S;
	p.values[1] = (u32)kHeads;
	p.values[2] = (u32)kDimHead;
	p.values[3] = (u32)q.off;
	p.values[4] = (u32)k.off;
	p.values[5] = (u32)v.off;
	p.values[6] = (u32)o.off;
	float scale = 1.0f / std::sqrt((float)kDimHead);
	memcpy(&p.values[7], &scale, 4);
	p.values[8] = (u32)(kHeads * kDimHead);
	p.srv[0] = q.res;
	p.srv[1] = k.res;
	p.srv[2] = v.res;
	p.uav[0] = o.res;
	ctx.dispatch(pso, p, (u32)ceil_div(S, impl == 1 ? 32 : 64), (u32)kHeads, 1);
}

// ── the decoder's attention on the tensor cores ───────────────────────────
//
// `attn3d_v2` is fp32 SIMT and measured 542 ms of a 1.26 s tile - 43 % of the
// whole decode - at ~0.9 TMAC/s, 11 % of this part's fp32 FMA peak. The maths
// does not need fp32: the operands are a RoPE'd Q/K and a V straight out of a
// linear, the reference implementation itself runs the VAE in fp16, and the
// same kernel is already the DiT's hot path in fp16 with an fp16 *accumulator*
// per key tile and an fp32 flush across tiles (see attn_tiled's header).
//
// So this is `attn_tiled` with head_dim 64 (the source rewrite in
// `attn_tiled_src`), fed by an fp32 -> fp16 cast of q/k/v. The cast is
// S*2048*2 bytes per tensor against 36 layers of attention; the kernel reads
// the packed halves with ldmatrix and runs the two matmuls on
// `mma.m16n8k16.f16.f16.f16`.
void dispatch_attn3d_tc(ComputeContext& ctx, const GpuAlloc& q, const GpuAlloc& k, const GpuAlloc& v,
                        const GpuAlloc& o, i64 S) {
	const char* src = attn_tiled_src(64u, (unsigned)kDimHead);
	const std::string key = attn_tiled_name(64u, (unsigned)kDimHead);
	GpuKernel* pso =
	    ctx.pipeline("h3vae_" + key, src, "attn_tiled", ShaderModel::SM6_2);
	KernelParams p{};
	p.values[0] = (u32)S;
	p.values[1] = (u32)kHeads;
	p.values[2] = (u32)kDimHead;
	p.values[3] = (u32)q.off;
	p.values[4] = (u32)k.off;
	p.values[5] = (u32)v.off;
	p.values[6] = (u32)o.off;
	float scale = 1.0f / std::sqrt((float)kDimHead);
	memcpy(&p.values[7], &scale, 4);
	p.values[8] = (u32)(kHeads * kDimHead);   // seqStride
	p.values[9] = (u32)kHeads;                // no GQA
	p.values[10] = (u32)(kHeads * kDimHead);  // kvStride
	p.values[11] = 0u;                        // not causal, no row mask
	p.values[14] = 0u;                        // whole query axis in one dispatch
	p.srv[0] = q.res;
	p.srv[1] = k.res;
	p.srv[2] = v.res;
	p.uav[0] = o.res;
	// BM = 64 here (the source's own tile), so the grid is the row blocks per head.
	ctx.dispatch(pso, p, (u32)ceil_div(S, 64), (u32)kHeads, 1);
}


// ── tensor-core GEMM wrappers (kernels/gemm_f16_mma.cpp) ──────────────────
//
// The mma kernel takes fp16 A, so the fp32 activation is cast into per-layer
// scratch first; that pass is a few hundred KB of traffic against a 200 MB
// weight sweep, and it is what lets the GEMM use cp.async for both operands.
void dispatch_f32_to_f16(ComputeContext& ctx, const GpuAlloc& x, const GpuAlloc& y, i64 count) {
	if (count <= 0) return;
	GpuKernel* pso =
	    ctx.pipeline("f32_to_f16", f32_to_f16_hlsl(), "f32_to_f16", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)count;
	p.values[1] = (u32)x.off;
	p.values[2] = (u32)y.off;
	p.srv[0] = x.res;
	p.uav[0] = y.res;
	ctx.dispatch(pso, p, (u32)ceil_div(count, 4 * 256), 1, 1);
}

void dispatch_gemm_f16_mma(ComputeContext& ctx, const GpuAlloc& a, const GpuAlloc& b,
                           const GpuAlloc& bias, const GpuAlloc& c, i64 m, i64 n, i64 k) {
	GpuKernel* pso =
	    ctx.pipeline("gemm_f16_mma", gemm_f16_mma_hlsl(), "gemm_f16_mma", ShaderModel::SM6_4);
	KernelParams p{};
	p.values[0] = (u32)m;
	p.values[1] = (u32)n;
	p.values[2] = (u32)k;
	p.values[3] = bias ? 1u : 0u;
	p.values[4] = (u32)a.off;
	p.values[5] = (u32)b.off;
	p.values[6] = (u32)c.off;
	p.values[7] = (u32)bias.off;
	p.srv[0] = a.res;
	p.srv[1] = b.res;
	p.srv[2] = bias ? bias.res : a.res;
	p.uav[0] = c.res;
	// grid is (ceil(M/BM), ceil(N/BN)) and the kernel walks it column-major. BM is
	// the kernel's own tile (kernels/gemm_f16_mma.cpp) and is duplicated here in
	// the same way the int8 dispatcher duplicates its own: the kernel derives
	// `mTile` from the block's position, so the two numbers have to agree.
	ctx.dispatch(pso, p, (u32)ceil_div(m, 128), (u32)ceil_div(n, 128), 1);
}

// ── the frozen interface ──────────────────────────────────────────────────

// --- test-only kernel entry points ---------------------------------------

std::vector<float> h3vae_test_conv3d(GpuCtx& g, GpuArena& arena, const std::vector<u16>& w,
                                     const std::vector<float>& bias, const std::vector<float>& x,
                                     i64 ic, i64 IT, i64 IH, i64 IW, i64 oc, i64 kt, i64 kh,
                                     i64 kw, i64 strT, i64 strS, i64 padT, i64 padH, i64 padHR) {
	ComputeContext& ctx = *g.ctx;
	ctx.submit_if_recording();
	g.ring->rewind();
	arena.reset();
	ctx.begin();
	CConv c;
	c.ic = ic;
	c.oc = oc;
	c.kt = kt;
	c.kh = kh;
	c.kw = kw;
	c.strT = strT;
	c.strS = strS;
	c.padT = padT;
	c.padH = padH;
	c.padHR = padHR;
	c.has_bias = !bias.empty();
	c.w = arena.alloc(w.size() * 2);
	g.upload_into(c.w, w.data(), w.size() * 2);
	if (c.has_bias) {
		// fp16 on the device: that is what the checkpoint holds, and it is what
		// `load_conv` uploads, so the kernel reads a bias the same width here.
		std::vector<u16> bh(bias.size());
		for (size_t i = 0; i < bias.size(); i++) bh[i] = f32_to_f16(bias[i]);
		c.b = arena.alloc(bh.size() * 2);
		g.upload_into(c.b, bh.data(), bh.size() * 2);
	}
	GpuAlloc xa = arena.alloc(x.size() * 4);
	g.upload_into(xa, x.data(), x.size() * 4);
	// The two spatial pads are separate (left/top vs right/bottom), so the
	// padded length is `IH + padH + padHR`, not `IH + 2*padH`: the symmetric
	// form silently shortened the output by one row when a caller asked for
	// right-only reflect padding, which is exactly the Downsample3D case.
	const i64 OT = (IT == 1 && kt > 1) ? 1 : (IT + padT - kt) / strT + 1;
	const i64 OH = (IH + padH + padHR - kh) / strS + 1;
	const i64 OW = (IW + padH + padHR - kw) / strS + 1;
	GpuAlloc ya = arena.alloc((u64)(oc * OT * OH * OW) * 4);
	dispatch_conv3d_causal(ctx, c, xa, ya, IT, IH, IW, OT, OH, OW);
	ctx.submit();
	g.ring->rewind();
	std::vector<float> out((size_t)(oc * OT * OH * OW));
	ctx.download(ya.res, ya.off, out.data(), out.size() * 4);
	return out;
}

std::vector<float> h3vae_test_causal_gn(GpuCtx& g, GpuArena& arena, const std::vector<float>& x,
                                        const std::vector<float>& w, const std::vector<float>& b,
                                        i64 C, i64 T, i64 H, i64 W) {
	ComputeContext& ctx = *g.ctx;
	ctx.submit_if_recording();
	g.ring->rewind();
	arena.reset();
	ctx.begin();
	CGn gn;
	gn.w = arena.alloc(w.size() * 4);
	g.upload_into(gn.w, w.data(), w.size() * 4);
	gn.b = arena.alloc(b.size() * 4);
	g.upload_into(gn.b, b.data(), b.size() * 4);
	GpuAlloc xa = arena.alloc(x.size() * 4);
	g.upload_into(xa, x.data(), x.size() * 4);
	GpuAlloc ya = arena.alloc(x.size() * 4);
	dispatch_causal_gn(ctx, gn, xa, ya, C, T, H, W);
	ctx.submit();
	g.ring->rewind();
	std::vector<float> out(x.size());
	ctx.download(ya.res, ya.off, out.data(), out.size() * 4);
	return out;
}

std::vector<float> h3vae_test_rope3d(GpuCtx& g, GpuArena& arena, const std::vector<float>& x,
                                     const std::vector<float>& ids, i64 S, i64 heads) {
	ComputeContext& ctx = *g.ctx;
	ctx.submit_if_recording();
	g.ring->rewind();
	arena.reset();
	ctx.begin();
	GpuAlloc xa = arena.alloc(x.size() * 4);
	g.upload_into(xa, x.data(), x.size() * 4);
	GpuAlloc ia = arena.alloc(ids.size() * 4);
	g.upload_into(ia, ids.data(), ids.size() * 4);
	std::vector<float> f((size_t)kRopeFreqs);
	const float step = 2.0f * (float)kRopeAxes / (float)(2 * kRopePairs);
	for (i64 i = 0; i < kRopeFreqs; i++)
		f[(size_t)i] = (float)std::pow((double)kRopeTheta, -(double)i * (double)step);
	GpuAlloc fa = arena.alloc(f.size() * 4);
	g.upload_into(fa, f.data(), f.size() * 4);
	dispatch_rope3d(ctx, xa, ia, fa, S);
	ctx.submit();
	g.ring->rewind();
	std::vector<float> out(x.size());
	ctx.download(xa.res, xa.off, out.data(), out.size() * 4);
	return out;
}

std::vector<float> h3vae_test_attn3d(GpuCtx& g, GpuArena& arena, const std::vector<float>& q,
                                     const std::vector<float>& k, const std::vector<float>& v,
                                     i64 S, int impl) {
	ComputeContext& ctx = *g.ctx;
	ctx.submit_if_recording();
	g.ring->rewind();
	arena.reset();
	ctx.begin();
	// GpuArena::alloc counts *bytes*; these were element counts, so each buffer
	// was a quarter of the size it needed and the uploads below ran off the end
	// of them (into the next buffer, which is why the kernel still produced
	// self-consistent numbers while both it and the host reference were right).
	const i64 Sp = round_up(S, 32);
	const u64 bts = (u64)Sp * (u64)kDim * 4;
	GpuAlloc qa = arena.alloc(bts);
	GpuAlloc ka = arena.alloc(bts);
	GpuAlloc va = arena.alloc(bts);
	GpuAlloc oa = arena.alloc(bts);
	g.upload_into(qa, q.data(), q.size() * 4);
	g.upload_into(ka, k.data(), k.size() * 4);
	g.upload_into(va, v.data(), v.size() * 4);
	dispatch_attn3d(ctx, qa, ka, va, oa, S, impl);
	ctx.submit();
	g.ring->rewind();
	std::vector<float> out((size_t)(S * kDim));
	ctx.download(oa.res, oa.off, out.data(), out.size() * 4);
	return out;
}

std::vector<float> h3vae_test_unpatch3d(GpuCtx& g, GpuArena& arena, const std::vector<float>& tok,
                                        i64 T, i64 H, i64 W) {
	ComputeContext& ctx = *g.ctx;
	ctx.submit_if_recording();
	g.ring->rewind();
	arena.reset();
	ctx.begin();
	GpuAlloc xa = arena.alloc(tok.size() * 4);
	g.upload_into(xa, tok.data(), tok.size() * 4);
	const i64 total = 3 * (T * kPatchT) * (H * kPatch) * (W * kPatch);
	GpuAlloc ya = arena.alloc((u64)total * 4);
	dispatch_unpatch3d(ctx, xa, ya, T, H, W);
	ctx.submit();
	g.ring->rewind();
	std::vector<float> out((size_t)total);
	ctx.download(ya.res, ya.off, out.data(), out.size() * 4);
	return out;
}

void VideoVae::open(const std::string& path, GpuCtx* gpu) {
	if (!gpu || !gpu->ok()) throw MediaError("h3vae: open() needs a live GpuCtx");
	g_ = gpu;
	path_ = path;   // release_gpu_memory() may empty the arenas; this is how they come back
	st_.open(path);
	warena_.init(gpu->ctx, 192ull << 20);
	aarena_.init(gpu->ctx, 256ull << 20);

	auto im = std::make_unique<Impl>();
	im->stream.init(gpu->ctx, 256ull << 20);

	gpu->ctx->submit_if_recording();
	gpu->ring->rewind();
	gpu->ctx->begin();

	im->dec.x_embed_w = load_f16(st_, "decoder.x_embedder.weight", *gpu, warena_);
	im->dec.x_embed_b = load_f32(st_, "decoder.x_embedder.bias", *gpu, warena_);
	im->dec.reg_tokens = load_f32(st_, "decoder.register_tokens", *gpu, warena_);
	im->dec.norm_out_w = load_f32(st_, "decoder.norm_out.weight", *gpu, warena_);
	im->dec.norm_out_b = load_f32(st_, "decoder.norm_out.bias", *gpu, warena_);
	im->dec.proj_out_w = load_f16(st_, "decoder.proj_out.weight", *gpu, warena_);
	im->dec.proj_out_b = load_f32(st_, "decoder.proj_out.bias", *gpu, warena_);
	im->quant = load_conv(st_, "quant_conv", *gpu, warena_, 0, 1, 1);
	im->post_quant = load_conv(st_, "post_quant_conv", *gpu, warena_, 0, 1, 1);

	im->latents_mean = tensor_to_f32(st_, st_.require("latents_mean"));
	im->latents_std = tensor_to_f32(st_, st_.require("latents_std"));
	im->latents_mean_dev = load_f32(st_, "latents_mean", *gpu, warena_);
	im->latents_std_dev = load_f32(st_, "latents_std", *gpu, warena_);

	{
		std::vector<float> f((size_t)kRopeFreqs);
		const float step = 2.0f * (float)kRopeAxes / (float)(2 * kRopePairs);   // 6/48 = 0.125
		for (i64 i = 0; i < kRopeFreqs; i++)
			f[(size_t)i] = (float)std::pow((double)kRopeTheta, -(double)i * (double)step);
		im->inv_freq_dev = warena_.alloc(f.size() * 4);
		gpu->upload_into(im->inv_freq_dev, f.data(), f.size() * 4);
	}

	for (i64 i = 0; i < kLayers; i++) {
		const std::string p = "decoder.transformer_blocks." + std::to_string(i);
		const StTensor& w = st_.require(p + ".attn.to_qkv.weight");
		if (w.shape.size() != 2 || w.shape[0] != 3 * kDim || w.shape[1] != kDim)
			throw MediaError("h3vae: unexpected qkv shape at " + p);
		im->block_prefix.push_back(p);
		// The four fp16 matrices a block streams (the biases and the norms are
		// kilobytes each and are left to the ordinary path).
		std::vector<std::pair<u64, u64>> r;
		for (const char* nm : {".attn.to_qkv.weight", ".attn.to_out.weight",
		                      ".ff.w1.weight", ".ff.w2.weight"}) {
			const StTensor& t = st_.require(p + nm);
			r.push_back({t.offset, t.nbytes});
		}
		im->block_ranges.push_back(std::move(r));
	}

	gpu->ctx->submit();
	gpu->ring->rewind();

	// The frozen header's defaults describe the plan's (wrong) architecture; the
	// runtime config reports what the checkpoint and the reference actually are.
	cfg_.in_channels = 3;
	cfg_.z_channels = 24;
	cfg_.ch = 128;
	cfg_.ch_mult = {1, 2, 2, 4, 4, 8};
	cfg_.num_res_blocks = 2;
	cfg_.spatial_factor = kPatch;
	cfg_.temporal_factor = kPatchT;
	cfg_.norm_groups = kNormGroups;
	cfg_.norm_eps = kGnEps;

	g_impl[this] = std::move(im);
	open_ = true;
}

u64 VideoVae::estimate_tile_bytes(i64 tile) {
	// Calibrated on this checkpoint: a 256 px tile peaks at ~768 MB of decoder
	// activations, and the buffers inside decode_pixels scale with the tile area
	// (the tile's latent is tile/16 square). A 20 % margin is included, the same
	// one plan_residency adds.
	const double t = (double)std::max<i64>(tile, 16) / 256.0;
	const double bytes = 768.0 * 1024.0 * 1024.0 * t * t;
	return (u64)bytes;
}

i64 VideoVae::plan_tile(i64 H, i64 W) {
	const u64 budget = vram_budget().limit();
	// The decoder's ~192 MB of resident weights are part of the same phase.
	const u64 weights = 256ull << 20;
	// The default is the reference's own 256 px, and it is deliberately not
	// raised: the tiled decode is an *approximation* of the untiled one whose blend
	// depends on the tile size, so the 256 px result is the one this engine has
	// been validated against (a probe against ComfyUI's own decoder:
	// maxdiff 0.005419, cos 1.000000). A bigger tile would be both faster and
	// closer to the untiled model, and on this 6 GB card it fits - but it would
	// also silently change every frame of every video, which is not a trade a
	// budget guard should make on its own. PHI_VAE_TILE opts in explicitly.
	if (const char* e = getenv("PHI_VAE_TILE")) {
		const long v = strtol(e, nullptr, 10);
		if (v >= 64 && v <= 1024) return (i64)v;
	}
	if (budget == 0) return kTileSize;
	static const i64 kCand[] = {256, 192, 128};
	for (i64 t : kCand) {
		if (t == kTileSize) return t;   // the validated default always wins when it fits
		if (estimate_tile_bytes(t) + weights <= budget) return t;
	}
	return 128;
}

std::vector<float> VideoVae::decode(const std::vector<float>& z, i64 T, i64 H, i64 W, i64 tile) {
	Impl* im = impl_of(this);
	if (!open_) throw MediaError("h3vae: decode() before open()");
	// `release_gpu_memory()` hands the weights back and drops the Impl (see the
	// header); the next decode re-reads them - the file is in the page cache by
	// then, so it is a memcpy per tensor against minutes of decoding.
	if (!im) {
		open(path_, g_);
		im = impl_of(this);
		if (!im) throw MediaError("h3vae: decode() could not reload its weights");
	}
	if (T <= 0 || H <= 0 || W <= 0 || (i64)z.size() != 24 * T * H * W)
		throw MediaError("h3vae: decode latent shape mismatch");
	// The planner picks the largest tile the budget can hold; if the accountant
	// still refuses the allocation (a different checkpoint, a client that took
	// VRAM since the plan) the decode is retried one size down rather than
	// failing the whole video. Tile 0 means "plan it".
	i64 use_tile = tile > 0 ? tile : plan_tile(H, W);

	// z = z * latents_std + latents_mean (vae.py:700)
	std::vector<float> zn(z.size());
	for (i64 c = 0; c < 24; c++) {
		const float m = im->latents_mean[(size_t)c], s = im->latents_std[(size_t)c];
		for (i64 i = 0; i < T * H * W; i++)
			zn[(size_t)(c * T * H * W + i)] = z[(size_t)(c * T * H * W + i)] * s + m;
	}

	Runner r{*g_, *g_->ctx, aarena_, &act_bytes_};
	act_bytes_ = 0;

	last_tile_ = use_tile;
	for (;;) {
		try {
			if (T == 1) {
				std::vector<float> raw = tiled_decode(st_, *im, r, zn, 1, H, W, use_tile);
				const i64 HP = H * kPatch, WP = W * kPatch;
				std::vector<float> out((size_t)(3 * HP * WP));
				MatV src = MatV::whole(raw.data(), 3, kPatchT, HP, WP);
				for (i64 c = 0; c < 3; c++)
					for (i64 y = 0; y < HP; y++)
						for (i64 x = 0; x < WP; x++) {
							float v = *src.at(c, kPatchT - 1, y, x) * kPixelStd[c] + kPixelMean[c];
							out[(size_t)((c * HP + y) * WP + x)] =
							    std::min(1.0f, std::max(0.0f, v));
						}
				return out;
			}
			return decode_temporal(st_, *im, r, zn, T, H, W, use_tile);
		} catch (const std::exception& ex) {
			// Retry one size down: a decode that ran out of the accountant's budget
			// is a tile-size decision, not a broken request.
			if (use_tile <= 128) throw;
			const char* what = ex.what();
			if (!what || (!strstr(what, "VRAM") && !strstr(what, "vram"))) throw;
			const i64 next = use_tile > 256 ? use_tile - 128 : use_tile / 2;
			fprintf(stderr, "[h3vae] tile %lld refused (%s); retrying at %lld\n",
			        (long long)use_tile, what, (long long)next);
			use_tile = std::max<i64>(128, next);
			last_tile_ = use_tile;
			r.ctx.submit_if_recording();
			r.aa.reset();
			r.ctx.begin();
		}
	}
}

std::vector<float> VideoVae::encode(const std::vector<float>& x, i64 T, i64 H, i64 W) {
	Impl* im = impl_of(this);
	if (!open_) throw MediaError("h3vae: encode() before open()");
	if (!im) {   // see decode()
		open(path_, g_);
		im = impl_of(this);
		if (!im) throw MediaError("h3vae: encode() could not reload its weights");
	}
	if (T <= 0 || H <= 0 || W <= 0 || (i64)x.size() != 3 * T * H * W)
		throw MediaError("h3vae: encode input shape mismatch");

	Runner r{*g_, *g_->ctx, aarena_, &act_bytes_};
	r.ctx.submit_if_recording();
	r.g.ring->rewind();
	r.aa.reset();
	act_bytes_ = 0;
	r.ctx.begin();
	if (!im->enc_loaded) {
		load_encoder(st_, *im, *g_, warena_);
		im->enc_loaded = true;
	}

	auto normalize = [](std::vector<float>& p, i64 TT, i64 HH, i64 WW) {
		const i64 hw = HH * WW;
		for (i64 c = 0; c < 3; c++)
			for (i64 i = 0; i < TT * hw; i++) {
				float* v = &p[(size_t)(c * TT * hw + i)];
				*v = ((*v + 1.0f) * 0.5f - kPixelMean[c]) / kPixelStd[c];
			}
	};

	// The encoder's first 24 channels are the mean half of the moments
	// (`torch.chunk(moments, 2, dim=1)[0]`); concatenating them per clip is the
	// same number as concatenating first and splitting afterwards.
	std::vector<float> lat;
	i64 HL = 0, WL = 0, total_T = 0;

	if (T == 1) {
		std::vector<float> nx = x;
		normalize(nx, 1, H, W);
		EncOut m = encode_tiled(*im, r, nx, 1, H, W);
		if (m.C < 24) throw MediaError("h3vae: encoder produced too few channels");
		HL = m.H;
		WL = m.W;
		total_T = 1;
		lat.assign((size_t)(24 * HL * WL), 0.0f);
		for (i64 c = 0; c < 24; c++)
			for (i64 y = 0; y < HL; y++)
				for (i64 xx = 0; xx < WL; xx++)
					lat[(size_t)((c * HL + y) * WL + xx)] =
					    m.v[(size_t)(((c * m.T + (m.T - 1)) * m.H + y) * m.W + xx)];
	} else {
		const i64 nclips = ceil_div(T, kClipLength);
		std::vector<std::vector<float>> clips((size_t)nclips);
		for (i64 ci = 0; ci < nclips; ci++) {
			const i64 f0 = ci * kClipLength;
			const i64 fn = std::min(kClipLength, T - f0);
			std::vector<float> clip((size_t)(3 * kClipLength * H * W));
			for (i64 c = 0; c < 3; c++)
				for (i64 t = 0; t < kClipLength; t++) {
					const i64 src_t = f0 + std::min(t, fn - 1);   // repeat the last frame
					memcpy(clip.data() + ((size_t)c * kClipLength + t) * H * W,
					       x.data() + ((size_t)c * T + src_t) * H * W, (size_t)H * W * 4);
				}
			normalize(clip, kClipLength, H, W);
			EncOut m = encode_tiled(*im, r, clip, kClipLength, H, W);
			HL = m.H;
			WL = m.W;
			clips[(size_t)ci].assign((size_t)(24 * m.T * HL * WL), 0.0f);
			for (i64 c = 0; c < 24; c++)
				for (i64 t = 0; t < m.T; t++)
					for (i64 y = 0; y < HL; y++)
						for (i64 xx = 0; xx < WL; xx++)
							clips[(size_t)ci][(size_t)(((c * m.T + t) * HL + y) * WL + xx)] =
							    m.v[(size_t)(((c * m.T + t) * m.H + y) * m.W + xx)];
			total_T += m.T;
		}
		lat.assign((size_t)(24 * total_T * HL * WL), 0.0f);
		i64 off = 0;
		for (i64 ci = 0; ci < nclips; ci++) {
			const i64 ct = (i64)clips[(size_t)ci].size() / std::max<i64>(1, 24 * HL * WL);
			for (i64 c = 0; c < 24; c++)
				memcpy(lat.data() + (size_t)(c * total_T + off) * HL * WL + 0,
				       clips[(size_t)ci].data() + (size_t)(c * ct) * HL * WL, (size_t)(ct * HL * WL) * 4);
			off += ct;
		}
		if (kTokenDrop > 0) {
			if (total_T <= kTokenDrop) throw MediaError("h3vae: too few latent tokens to drop");
			total_T -= kTokenDrop;
		}
	}

	// (mean - latents_mean) / latents_std
	std::vector<float> out((size_t)(24 * total_T * HL * WL));
	for (i64 c = 0; c < 24; c++) {
		const float m = im->latents_mean[(size_t)c], s = im->latents_std[(size_t)c];
		for (i64 i = 0; i < total_T * HL * WL; i++)
			out[(size_t)(c * total_T * HL * WL + i)] =
			    (lat[(size_t)(c * total_T * HL * WL + i)] - m) / s;
	}
	return out;
}

void VideoVae::release_activation_memory() {
	if (!open_) return;
	// Both of these are pure scratch for one tile or one transformer block of one
	// pass, and both are re-claimed (and re-read from the checkpoint) by whatever
	// runs next: the encoder's tile scratch in `aarena_`, and the decoder's
	// streaming window for a block's fp16 weights in `stream`. They are
	// deliberately not kept "warm" across phases - the arenas' chunks are
	// committed device memory that the accountant keeps charging while the DiT
	// samples or the *next* call encodes its references, and a decode leaves
	// ~1-4 GB of them behind (16 of the 256 MB chunks for the 4.00 GB the
	// reference encode used to reserve, 4 after a 960x544 decode). That leftover
	// is what made this failure probabilistic: the same request succeeded on a
	// fresh process and died after a call that had left the arena full.
	aarena_.release_chunks();
	if (Impl* im = impl_of(this)) {
		im->stream.release_chunks();
		// The resident decoder stack goes back too, and for the same reason: it is
		// 3.6 GB of committed device memory the accountant keeps charging while the
		// DiT samples or the next reference encode runs. Dropping it here costs one
		// re-upload per `decode()` call (~1 s), and keeps every tile of that call
		// reading it out of VRAM instead of off the file.
		im->wcache.release_chunks();
		im->dec_w.clear();
		im->dec_n = 0;
		im->dec_tried = false;
		// The pinned file ranges stay: they are host memory holding bytes that
		// will be wanted again the moment the next decode re-fills the stack,
		// and re-reading 1.2 GB off the disk to get them back would cost more
		// than the RAM they occupy.
		im->dec_logged = -1;
	}
	act_bytes_ = 0;
}

// The cross-chain boundary (see the header): the fixed weights go too. The Impl
// is dropped with them because every GpuAlloc it holds (the decoder statics, the
// latents' mean/std, the rope table, the encoder's blocks) lives in `warena_` -
// keeping the alloc list after releasing the arena would leave it pointing at
// freed memory. `decode()`/`encode()` treat a missing Impl as "reload" and re-run
// `open(path_, g_)`, so the class is usable again immediately; the owner (the
// node cache's `release_media_cache`) drops the cached handle's weights, which is
// what makes the *next run* do the re-open at its own decode step.
void VideoVae::release_gpu_memory() {
	if (!open_) return;
	release_activation_memory();
	g_impl.erase(this);   // the allocs in `warena_` dangle from here on
	warena_.release_chunks();
}

// Test-only export of the fused-gate activation (see the internal header).
void h3vae_test_silu_gate_fused(GpuCtx& g, GpuArena& arena, const std::vector<float>& gx,
                                std::vector<float>& out, i64 rows, i64 cols) {
	GpuAlloc a = arena.alloc((u64)gx.size() * 4);
	GpuAlloc y = arena.alloc((u64)rows * cols * 4);
	g.ctx->submit_if_recording();
	g.ctx->begin();
	g.upload_into(a, gx.data(), (u64)gx.size() * 4);
	dispatch_silu_gate_fused(*g.ctx, a, y, rows, cols);
	g.ctx->submit();
	out = g.download_f32(y, (u64)rows * cols);
}

}  // namespace phi::media
