// Host-side wrappers for the M1 kernels: shapes in, dispatch out.
//
// Every entry point takes GpuAlloc ranges and the logical shape, derives the
// grid, and throws MediaError on anything it cannot honour (a K that is not
// half-precise, an attention head that does not fit shared memory, ...). The
// reference implementation of each operator lives in tensor.* and the kernel
// self-tests compare against it.
#pragma once

#include "runtime/sched.hpp"

namespace phi::media {

// ── elementwise ────────────────────────────────────────────────────────────

enum class ElemOp {
	Add = 0,
	Mul = 1,
	SiluGate = 2,   // silu(a) * b
	Tanh = 3,
	Silu = 4,
	Scale = 5,      // a * alpha
	AddMul = 6,     // a + b * c
	Copy = 7,
	Modulate = 8,   // a * (1 + b)
	Exp = 9,
	AddScale = 10,  // a + b * alpha
	GeluTanh = 11,  // gelu(a), tanh approximation (F.gelu approximate="tanh")
	// `a * (1 + b) + c` - the AdaLN pair every DiT block applies to its normed
	// input (normalise, scale-and-shift by the modulation row, project). The two
	// ops it replaces each read and write the whole [rows, cols] fp32 buffer, so
	// fusing them removes one full pass over the activation - which on this box is
	// bandwidth, not FLOPs, and it is the same result bit for bit: `av * (1+bv) +
	// cv` is evaluated in the same order, with the same two roundings.
	ModAdd = 12,
};

// Broadcast mode for the B and C operands.
enum class ElemMode { Flat = 0, Row = 1, Col = 2, Scalar = 3 };

struct ElemArgs {
	ElemOp op = ElemOp::Add;
	GpuAlloc a, b, c, y;
	i64 rows = 1, cols = 1;      // logical shape of Y (flat if rows*cols == N)
	ElemMode bMode = ElemMode::Flat;
	ElemMode cMode = ElemMode::Flat;
	float alpha = 0.0f;
	float beta = 0.0f;
};

void dispatch_elem(ComputeContext& ctx, const ElemArgs& a);

// ── normalisation ──────────────────────────────────────────────────────────

struct NormArgs {
	GpuAlloc x, w, b, residual, y;
	i64 rows = 0, cols = 0;
	float eps = 1e-6f;
	bool affine = false;
	bool has_bias = false;
	bool add_residual = false;
	bool layer_norm = false;   // false = RMSNorm
};

void dispatch_norm(ComputeContext& ctx, const NormArgs& a);

// Per-channel normalisation statistics, folded with the affine into the pair
// (A[c], Bc[c]) = (rstd*w[c], b[c] - mean*rstd*w[c]), so a consumer can
// reproduce silu(group_norm(x)) as silu(x*A + Bc) in its own operand load.
// The reduction is the same code in the same order as the normalising path,
// so the statistics are bit-identical to the ones it computes.
struct GroupNormStatsArgs {
	GpuAlloc x, wgt, bias, out;   // out is 2*C floats: [A(C) | Bc(C)]
	i64 n = 1, c = 0, h = 0, w = 0;
	int groups = 32;
	float eps = 1e-6f;
};

void dispatch_group_norm_stats(ComputeContext& ctx, const GroupNormStatsArgs& a);

struct GroupNormArgs {
	GpuAlloc x, wgt, bias, y;
	i64 n = 1, c = 0, h = 0, w = 0;
	int groups = 32;
	float eps = 1e-6f;
	bool affine = true;
};

void dispatch_group_norm(ComputeContext& ctx, const GroupNormArgs& a);

// ── convolution ────────────────────────────────────────────────────────────

struct Conv2dArgsG {
	GpuAlloc x, wgt, bias, y;               // x [N,C,H,W], w [OC,IC,KH,KW], y [N,OC,OH,OW]
	i64 n = 1, ic = 0, ih = 0, iw = 0;
	i64 oc = 0, kh = 0, kw = 0;
	i64 stride = 1, pad = 1;
	// Input is nearest-neighbour duplicated by this factor before the convolution
	// (1 = plain conv, 2 = the AE's upsample.conv). Reading the two output pixels
	// that share a source pixel is one gather instead of a separate 2x2 expansion
	// pass, which at 1080p removes a 2.14 GB intermediate - the single largest
	// allocation the decoder makes.
	int dup = 1;
	bool has_bias = false;
	// silu(group_norm(x)) folded into the convolution's operand load, using the
	// packed (A | Bc) pair `dispatch_group_norm_stats` produced for the *input*
	// tensor. This is what stops the decoder materialising the normalised copy
	// of a 2.14 GB activation at 1080p.
	GpuAlloc norm;
	bool fused_norm = false;
	i64 norm_channels = 0;   // input channels; also the stride from A[c] to Bc[c]
	// Zero-pad the bottom/right edge only (torch's ZeroPad2d((0,1,0,1))), which is
	// what the Wan VAEs' strided downsample convs are built on. With `pad` left at
	// 0 the kernel's own bounds check turns the missing row/column into the same
	// zero the reference materialises, so only the output size differs.
	bool pad_right_bottom = false;
	// The weight tensor's own storage width: 0 = fp32, 1 = bf16, 2 = fp16. The
	// conv kernel decodes each operand as it loads it, so a decoder that keeps
	// 675 MB of bf16 weights charges the accountant 675 MB rather than the 1.35 GB
	// the fp32 copy costs - which is what decides whether a 1024x1024 decode fits
	// on a 6 GB card at all. `image_vae` sets it from the checkpoint (goal #1: the
	// file's precision, not one the engine picked).
	int w_dtype = 0;
};

void dispatch_conv2d(ComputeContext& ctx, const Conv2dArgsG& a);

// ── causal 1-D convolution (the ACE-Step 1.5 audio VAE decoder) ─────────────
//
// torch Conv1d / ConvTranspose1d over one sample's [C, L] activation (no batch,
// channel-major, fp32). The two module references the engine already has
// host-side are `Conv1dRef::forward` / `ConvT1dRef::forward`
// (core/models/audio_vocoder.hpp): the dispatchers derive the output length from
// the same formula and refuse a mismatched one, and both kernels accumulate in
// the reference's own order (see the kernel documentation in kernels/ops.cpp).

struct Conv1dArgsG {
	GpuAlloc x, wgt, bias, y;   // x [IC,L], wgt [OC,IC,K], y [OC, LO], bias [OC] or empty
	i64 ic = 0, l = 0, oc = 0, k = 0;
	i64 stride = 1, pad = 0, dilation = 1;
	bool has_bias = false;
};

void dispatch_conv1d(ComputeContext& ctx, const Conv1dArgsG& a);

struct ConvT1dArgsG {
	GpuAlloc x, wgt, bias, y;   // x [IC,L], wgt [IC,OC,K], y [OC, LO], bias [OC] or empty
	i64 ic = 0, l = 0, oc = 0, k = 0;
	i64 stride = 1, pad = 0;
	bool has_bias = false;
};

// Requires k == 2*stride (the only shape this net has, and the only one the
// two-tap gather is exact for). Anything else is a MediaError, not a wrong
// tensor: the kernel cannot represent a third tap.
void dispatch_convt1d(ComputeContext& ctx, const ConvT1dArgsG& a);

// y[c,j] = x[c,j] + sin(a[c]*x[c,j])^2 / (b[c] + 1e-9), with a/b the *already
// exponentiated* SnakeBeta parameters ([C] fp32 each). x and y may alias.
struct SnakeBetaArgsG {
	GpuAlloc x, alpha, beta, y;
	i64 c = 0, l = 0;
};

void dispatch_snake_beta(ComputeContext& ctx, const SnakeBetaArgsG& a);

// conv2d with the input nearest-duplicated by `dup` and/or y += conv(x).

// [C,S] -> [S,C] (to_token = true) or [S,C] -> [C,S] (false).
void dispatch_transpose_cs(ComputeContext& ctx, const GpuAlloc& x, const GpuAlloc& y, i64 c,
                           i64 s, bool to_token);

// nearest 2x upsample, [N,C,H,W] -> [N,C,2H,2W]
void dispatch_upsample2x(ComputeContext& ctx, GpuBuffer* res, const GpuAlloc& x,
                         const GpuAlloc& y, i64 c, i64 h, i64 w);

// ── gemm ───────────────────────────────────────────────────────────────────

struct GemmF16Args {
	GpuAlloc a, b, c, bias;
	i64 m = 0, n = 0, k = 0;
	bool a_is_f32 = false;
	bool b_is_bf16 = false;
	bool b_is_f32 = false;
	bool has_bias = false;
};

void dispatch_gemm_f16(ComputeContext& ctx, const GemmF16Args& a);

// C[m, n] += H[m, rank] @ B^T. See `lora_rank_add_hlsl` for why this is not a
// `gemm_f16` with its bias aimed at C.
struct LoraRankAddArgs {
	GpuAlloc h, b, c;
	i64 m = 0, n = 0, rank = 0;
};
void dispatch_lora_rank_add(ComputeContext& ctx, const LoraRankAddArgs& a);

// ── layout ─────────────────────────────────────────────────────────────────

void dispatch_patchify(ComputeContext& ctx, GpuBuffer* res, const GpuAlloc& x,
                       const GpuAlloc& y, i64 c, i64 h, i64 w, int patch);
void dispatch_unpatchify(ComputeContext& ctx, GpuBuffer* res, const GpuAlloc& x,
                         const GpuAlloc& y, i64 c, i64 h, i64 w, int patch);

// ── rope ───────────────────────────────────────────────────────────────────

struct RopeHalfArgs {
	GpuAlloc x, y;
	i64 rows = 0, head_dim = 0;
	float theta = 1e6f;
	i64 heads = 1;
	i64 base = 0;
};

void dispatch_rope_half(ComputeContext& ctx, GpuBuffer* res, const RopeHalfArgs& a);

// ── DiT attention ──────────────────────────────────────────────────────────

struct QkvPrepArgs {
	GpuAlloc qkv, wq, wk, rope, ids;
	GpuAlloc q, k, v;
	i64 s = 0, heads = 0, kv_heads = 0, head_dim = 0;
	float eps = 1e-6f;
	float scale = 1.0f;
	bool norm = true;
	bool rope_on = true;
	// Store q/k/v as fp32. attn_flash reads one element per multiply-accumulate,
	// so the mask+shift+convert the fp16 form needs costs several times the
	// instructions of a bare load. Worth it whenever attention is the bottleneck.
	bool f32_out = false;
	// First row of `ids` that belongs to this sequence. The refiners run on a
	// slice of the concatenated stream (the cap tokens then the image tokens),
	// so the image tokens must address ids[cap_len + s], not ids[s].
	i64 ids_row = 0;
};

void dispatch_qkv_prep(ComputeContext& ctx, const QkvPrepArgs& a);

struct AttnFlashArgs {
	GpuAlloc q, k, v, o;
	i64 s = 0, heads = 0, head_dim = 0;
	i64 kv_heads = 0;   // 0 => same as heads (no GQA)
	bool f32_input = false;
	bool causal = true; // false = the DiT's bidirectional attention
	float scale = 1.0f;
	// Optional per-row key bound, [q_rows] u32, exclusive: query row r attends
	// keys [0, ub[r]). This is the Qwen-Image DiT's *block-causal* mask - the text
	// tokens attend causally inside their own run and every image block attends
	// to everything before its end - which a plain `causal` flag cannot express.
	// Only the tiled kernel implements it (see dispatch_attn_flash).
	//
	// Indexed by *this dispatch's* query row (0 = the first row of `q`), not by an
	// absolute sequence row: a caller that runs a later slice of a sequence hands
	// in the mask for that slice.
	GpuAlloc ub;
	// Query rows of this dispatch. 0 (the default) means "as many as `s`", which
	// is every caller but the prefix-cached one: there the keys span the whole
	// sequence (`s`) while only the target rows are queried, so the grid - and the
	// rows of `q`/`o` - are `q_rows` tall. `k`/`v` still run over [0, s).
	i64 q_rows = 0;

	i64 query_rows() const { return q_rows > 0 ? q_rows : s; }
};

void dispatch_attn_flash(ComputeContext& ctx, const AttnFlashArgs& a);

// Wan-VAE "DupUp3D" with factor_s 2 and an optional temporal factor_t:
//   out[c][2h + i][2w + j] = in[(ft*4*c + 4*(ft-1) + 2i + j) / repeats][h][w]
// with repeats = ft*4*oc/ic. `ft = 1` is the spatial-only form; `ft = 2` is the
// decoder stages whose `temperal_upsample` flag is set, where the single-frame
// path keeps the f_t = 1 slice (DupUp3D's `first_chunk`) and the 3D view folds
// the temporal slots back into the channel gather. The decoder's upsample
// shortcut: the resample conv's inverse on the channel axis. It is not a
// nearest upsample and not a conv, hence its own kernel.
struct DupUp2xArgs {
	GpuAlloc x, y;
	i64 ic = 0, oc = 0, h = 0, w = 0;
	i64 ft = 1;
};

void dispatch_dupup2x(ComputeContext& ctx, const DupUp2xArgs& a);

// Channel-wise RMSNorm over an NCHW activation: `cols` channels, `rows` pixels,
// y = x * w[c] / sqrt(mean_c x^2 + eps). The Wan VAEs' RMS_norm.
struct ChwNormArgs {
	GpuAlloc x, w, y;
	i64 rows = 0, cols = 0;
	float eps = 1e-12f;
};

void dispatch_chw_norm(ComputeContext& ctx, const ChwNormArgs& a);

// Zeroes everything of a [c,h,w] fp32 tensor outside rows [y0,y1) x columns
// [x0,x1), in place. The tiled VAE decode uses it to give each stage the
// reference's "defined only on the image" boundary (see mask_rect_hlsl).
void dispatch_mask_rect(ComputeContext& ctx, const GpuAlloc& x, i64 c, i64 h, i64 w, i64 y0,
                        i64 y1, i64 x0, i64 x1);

// Row-wise softmax over an fp32 [rows, cols] matrix, out of place.
void dispatch_row_softmax(ComputeContext& ctx, const GpuAlloc& x, const GpuAlloc& y, i64 rows,
                          i64 cols);


}  // namespace phi::media
