#include "kernels/gpu_ops.hpp"

#include <algorithm>
#include <cstdlib>

#include "util/base.hpp"

#include "kernels/kernels.hpp"

namespace phi::media {

// ── elementwise ────────────────────────────────────────────────────────────

void dispatch_elem(ComputeContext& ctx, const ElemArgs& a) {
	// Host-side gate: the kernel's chain of `if (op == ...) else if ... else`
	// treats any unknown opcode as OP_ADD_SCALE (the final `else`), so a
	// mis-numbered op would silently compute the wrong thing. Reject here.
	if ((unsigned)a.op > 12u) throw MediaError("elem: unknown opcode " +
	                                           std::to_string((int)a.op));
	i64 n = a.rows * a.cols;
	if (n <= 0) throw MediaError("elem: empty");
	// A rows*cols that does not match the buffer is the worst kind of bug: the
	// kernel happily writes past the allocation and corrupts whatever the arena
	// put next to it (that is how a mis-specified tanh() flattened a whole
	// scratch block). GpuAlloc carries the byte size, so check it.
	if (a.y.bytes && (u64)n * 4 > a.y.bytes)
		throw MediaError("elem: output is " + format_bytes((u64)n * 4) + " but the buffer is " +
		                 format_bytes(a.y.bytes));
	if (a.a.bytes && (u64)n * 4 > a.a.bytes)
		throw MediaError("elem: operand A is too small for " + std::to_string(n) + " elements");
	if (a.b.bytes && a.bMode == ElemMode::Flat && (u64)n * 4 > a.b.bytes)
		throw MediaError("elem: operand B is too small for " + std::to_string(n) + " elements");
	// A flat B that lives inside A's own buffer is the fused gate/value layout, and
	// pairing the two halves flat is right for row 0 and wrong for every other row:
	// the pairing needs a row stride, which `elem` does not have. The H3 DiT fed
	// exactly this shape into SiluGate for all 50 blocks (a [rows, 2*ffn] buffer with
	// B re-based by rows*ffn) and lost every block's MLP to it; use
	// `dispatch_silu_gate_fused` (h3_kernels.hpp) instead. Two *separate* buffers of
	// the same size are the ordinary case and stay supported.
	if (a.op == ElemOp::SiluGate && a.bMode == ElemMode::Flat && a.rows > 1 && a.a.res &&
	    a.a.res == a.b.res && a.a.bytes >= 2 * (u64)n * 4 && a.b.off == a.a.off + (u64)n * 4)
		throw MediaError(
		    "elem: SiluGate with a flat B inside A's buffer pairs row 0 correctly and every "
		    "other row with the wrong half; use dispatch_silu_gate_fused");
	if (a.c.bytes && a.cMode == ElemMode::Flat && a.op == ElemOp::AddMul &&
	    (u64)n * 4 > a.c.bytes)
		throw MediaError("elem: operand C is too small for " + std::to_string(n) + " elements");
	// Four elements per thread when the shape allows it: 16-byte accesses on both
	// the input and the output, and N has no tail to peel. See the `elem4` comment
	// in kernels/ops.cpp for why the scalar form was the DiT's third-largest cost.
	static const bool no_vec = [] { const char* e = getenv("PHI_ELEM_SCALAR"); return e && *e && *e != '0'; }();
	const bool vec = !no_vec && (n % 4 == 0) && (a.a.off % 16 == 0) && (a.y.off % 16 == 0);
	GpuKernel* pso = vec ? ctx.pipeline("elem4", ops_hlsl(), "elem4", ShaderModel::SM5_1)
	                     : ctx.pipeline("elem", ops_hlsl(), "elem", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)n;
	p.values[1] = (u32)a.rows;
	p.values[2] = (u32)a.cols;
	p.values[3] = (u32)a.op;
	p.values[4] = (u32)a.a.off;
	p.values[5] = (u32)a.b.off;
	p.values[6] = (u32)a.c.off;
	p.values[7] = (u32)a.y.off;
	p.values[8] = (u32)a.bMode;
	p.values[9] = (u32)a.cMode;
	float alpha = a.alpha, beta = a.beta;
	memcpy(&p.values[10], &alpha, 4);
	memcpy(&p.values[11], &beta, 4);
	p.srv[0] = a.a.res;
	p.srv[1] = a.b.res ? a.b.res : a.a.res;
	p.srv[2] = a.c.res ? a.c.res : a.a.res;
	p.uav[0] = a.y.res;
	if (vec) {
		ctx.dispatch(pso, p, (u32)ceil_div(n, 4 * 256), 1, 1);
		return;
	}
	ctx.dispatch(pso, p, (u32)ceil_div(n, 256), 1, 1);
}

// ── normalisation ──────────────────────────────────────────────────────────

void dispatch_norm(ComputeContext& ctx, const NormArgs& a) {
	if (a.rows <= 0 || a.cols <= 0) throw MediaError("norm: empty");
	// The 16-byte form when every operand a row touches is whole float4s and
	// 16-byte aligned. `cols % 4` covers the row stride; the offsets cover the
	// tensors themselves (arena allocations are 256-byte aligned, but a caller
	// may hand in a slice - see NormArgs::x). See the `norm_rows4` comment.
	static const bool no_vec4 = [] { const char* e = getenv("PHI_NORM_SCALAR"); return e && *e && *e != '0'; }();
	const bool vec4 = !no_vec4 && (a.cols % 4 == 0) && (a.x.off % 16 == 0) && (a.y.off % 16 == 0) &&
	                  (a.w.off % 16 == 0) && (a.b.off % 16 == 0) && (a.residual.off % 16 == 0);
	GpuKernel* pso = vec4 ? ctx.pipeline("norm_rows4", norm_hlsl(), "norm_rows4", ShaderModel::SM5_1)
	                      : ctx.pipeline("norm_rows", norm_hlsl(), "norm_rows", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.rows;
	p.values[1] = (u32)a.cols;
	p.values[2] = (u32)a.x.off;
	p.values[3] = (u32)a.w.off;
	p.values[4] = a.has_bias ? (u32)a.b.off : 0u;
	p.values[5] = (u32)a.y.off;
	p.values[6] = (u32)a.residual.off;
	u32 flags = (a.affine ? 1u : 0u) | (a.add_residual ? 2u : 0u);
	p.values[7] = flags;
	float eps = a.eps;
	memcpy(&p.values[8], &eps, 4);
	p.values[9] = a.layer_norm ? 1u : 0u;
	p.srv[0] = a.x.res;
	p.srv[1] = a.w.res ? a.w.res : a.x.res;
	p.srv[2] = a.b.res ? a.b.res : a.x.res;
	p.srv[3] = a.residual.res ? a.residual.res : a.x.res;
	p.uav[0] = a.y.res;
	ctx.dispatch(pso, p, (u32)a.rows, 1, 1);
	(void)vec4;
}

void dispatch_group_norm(ComputeContext& ctx, const GroupNormArgs& a) {
	if (a.c % a.groups) throw MediaError("group_norm: channels not divisible by groups");
	GpuKernel* pso =
	    ctx.pipeline("group_norm", group_norm_hlsl(), "group_norm", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.c;
	p.values[1] = (u32)a.h;
	p.values[2] = (u32)a.w;
	p.values[3] = (u32)a.groups;
	p.values[4] = (u32)a.x.off;
	p.values[5] = (u32)a.wgt.off;
	p.values[6] = (u32)a.bias.off;
	p.values[7] = (u32)a.y.off;
	float eps = a.eps;
	memcpy(&p.values[8], &eps, 4);
	p.values[9] = (u32)(a.c / a.groups);
	p.values[10] = (u32)(a.h * a.w);
	p.values[11] = a.affine ? 1u : 0u;
	p.values[12] = 0u;   // mode: normalise
	p.values[13] = (u32)a.c;
	p.srv[0] = a.x.res;
	p.srv[1] = a.wgt.res ? a.wgt.res : a.x.res;
	p.srv[2] = a.bias.res ? a.bias.res : a.x.res;
	p.uav[0] = a.y.res;
	ctx.dispatch(pso, p, (u32)(a.n * a.groups), 1, 1);
}

// ── convolution ────────────────────────────────────────────────────────────

void dispatch_group_norm_stats(ComputeContext& ctx, const GroupNormStatsArgs& a) {
	if (a.c % a.groups) throw MediaError("group_norm_stats: channels not divisible by groups");
	GpuKernel* pso =
	    ctx.pipeline("group_norm", group_norm_hlsl(), "group_norm", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.c;
	p.values[1] = (u32)a.h;
	p.values[2] = (u32)a.w;
	p.values[3] = (u32)a.groups;
	p.values[4] = (u32)a.x.off;
	p.values[5] = (u32)a.wgt.off;
	p.values[6] = (u32)a.bias.off;
	p.values[7] = (u32)a.out.off;
	float eps = a.eps;
	memcpy(&p.values[8], &eps, 4);
	p.values[9] = (u32)(a.c / a.groups);
	p.values[10] = (u32)(a.h * a.w);
	p.values[11] = 1u;   // affine
	p.values[12] = 1u;   // mode: statistics + fold
	p.values[13] = (u32)a.c;
	p.srv[0] = a.x.res;
	p.srv[1] = a.wgt.res ? a.wgt.res : a.x.res;
	p.srv[2] = a.bias.res ? a.bias.res : a.x.res;
	p.uav[0] = a.out.res;
	ctx.dispatch(pso, p, (u32)(a.n * a.groups), 1, 1);
}

void dispatch_conv2d(ComputeContext& ctx, const Conv2dArgsG& a) {
	// `dup` folds a nearest-neighbour upsample into the convolution: the operand is
	// IHxIW and the output is (IH*dup)x(IW*dup), so the padding and the window walk
	// happen on the duplicated grid while only the source pixels are touched.
	const i64 uh = a.ih * a.dup, uw = a.iw * a.dup;
	// `pad_right_bottom` is the Wan VAE's ZeroPad2d((0,1,0,1)): one zero row and
	// column past the tensor's far edge, and none before its origin. The kernel is
	// handed pad = 0 for it and its own `iy < IH*dup` bound drops that extra row,
	// which is bit-identical to convolving the materialised zeros.
	const i64 pad_eff = a.pad_right_bottom ? 0 : a.pad;
	const i64 extra = a.pad_right_bottom ? 1 : 0;
	i64 oh = (uh + extra + 2 * pad_eff - (a.kh - 1) - 1) / a.stride + 1;
	i64 ow = (uw + extra + 2 * pad_eff - (a.kw - 1) - 1) / a.stride + 1;
	if (oh <= 0 || ow <= 0) throw MediaError("conv2d: empty output");
	GpuKernel* pso = ctx.pipeline("conv2d", conv2d_hlsl(), "conv2d", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.ic;
	p.values[1] = (u32)a.ih;
	p.values[2] = (u32)a.iw;
	p.values[3] = (u32)a.oc;
	p.values[4] = (u32)oh;
	p.values[5] = (u32)ow;
	p.values[6] = (u32)a.kh;
	p.values[7] = (u32)a.kw;
	p.values[8] = (u32)a.x.off;
	p.values[9] = (u32)a.wgt.off;
	p.values[10] = (u32)a.bias.off;
	p.values[11] = (u32)a.y.off;
	p.values[12] = (u32)(a.ic * a.kh * a.kw);
	p.values[13] = (u32)a.stride;
	p.values[14] = (u32)(i32)pad_eff;   // signed
	p.values[15] =
	    (a.has_bias ? 1u : 0u) | (a.dup != 1 ? 4u : 0u) | (a.fused_norm ? 8u : 0u);
	p.values[17] = (u32)a.dup;
	p.values[18] = (u32)a.norm.off;
	p.values[19] = (u32)(a.norm_channels * 4);
	p.values[20] = (u32)std::clamp(a.w_dtype, 0, 2);
	p.srv[0] = a.x.res;
	p.srv[1] = a.wgt.res;
	p.srv[2] = a.bias.res ? a.bias.res : a.x.res;
	// Bound only in the fused-norm form; the kernel reads the packed (A | Bc)
	// pair for the input channel as it loads each operand tile.
	p.srv[4] = a.norm.res ? a.norm.res : a.x.res;
	p.uav[0] = a.y.res;
	const u32 gx = (u32)ceil_div(a.oc * a.n, 64);
	// One (oc, pixel) work item per thread and K multiply-accumulates per item,
	// so the dispatch is O(OC * pixels * IC*KH*KW) in a single command. At the
	// AE's 1080p top level (128ch, 1088x1920, 3x3) that is ~3e11 MACs, which is
	// several seconds of uninterrupted GPU work: long enough for the Windows
	// watchdog (TDR, 2 s) to reset the device. Output positions are independent,
	// so splitting the spatial axis changes nothing but the schedule.
	const i64 pixels = oh * ow;
	const u64 per_pixel = (u64)a.oc * (u64)a.ic * (u64)a.kh * (u64)a.kw;
	// Measured: this kernel sustains ~2e10 multiply-accumulates/s, so a chunk of
	// 5e9 MACs is ~0.25 s - comfortably inside the 2 s watchdog. (20000 was the
	// first guess and still tripped it: 1.9e10 MACs in one dispatch is ~1 s.)
	const i64 target = 5000000000ll;
	i64 chunk_px = per_pixel ? (i64)((u64)target / per_pixel) : pixels;
	if (chunk_px < 512) chunk_px = 512;
	if (chunk_px > (1 << 21)) chunk_px = 1 << 21;
	for (i64 p0 = 0; p0 < pixels; p0 += chunk_px) {
		i64 span = std::min(chunk_px, pixels - p0);
		p.values[16] = (u32)p0;
		ctx.dispatch(pso, p, gx, (u32)ceil_div(span, 64), 1);
		// Drain between chunks. Bounding each *dispatch* is not enough: they
		// all land in the caller's command list, and the watchdog measures the
		// list. The AE's 1080p top level is 122 chunks of ~18 ms, i.e. 2.2 s of
		// uninterrupted GPU work in one ExecuteCommandLists - over the 2 s TDR
		// limit, which resets the device and then reports itself as a failed
		// allocation in an unrelated place. A fence wait per chunk is ~50 us
		// against 18 ms of work.
		ctx.submit();
		ctx.begin();
	}
}

// ── causal 1-D convolution (the ACE-Step 1.5 audio VAE decoder) ─────────────

void dispatch_conv1d(ComputeContext& ctx, const Conv1dArgsG& a) {
	if (a.ic <= 0 || a.l <= 0 || a.oc <= 0 || a.k <= 0)
		throw MediaError("conv1d: empty shape");
	if (a.stride <= 0 || a.dilation <= 0) throw MediaError("conv1d: stride/dilation must be >= 1");
	// The output length comes from the reference's own formula rather than from
	// the caller, so a caller that got the padding arithmetic wrong gets a
	// MediaError instead of a kernel that silently writes a different tensor
	// (the kernel's only shape knowledge is the input bounds test).
	const i64 lo = (a.l + 2 * a.pad - a.dilation * (a.k - 1) - 1) / a.stride + 1;
	if (lo <= 0) throw MediaError("conv1d: empty output");
	if (a.x.bytes && (u64)a.ic * (u64)a.l * 4 > a.x.bytes)
		throw MediaError("conv1d: the input allocation is smaller than " + std::to_string(a.ic) +
		                 " x " + std::to_string(a.l));
	if (a.wgt.bytes && (u64)a.oc * (u64)a.ic * (u64)a.k * 4 > a.wgt.bytes)
		throw MediaError("conv1d: the weight allocation is smaller than the [OC,IC,K] tensor");
	if (a.has_bias && a.bias.bytes && (u64)a.oc * 4 > a.bias.bytes)
		throw MediaError("conv1d: the bias allocation is smaller than OC");
	if (a.y.bytes && (u64)a.oc * (u64)lo * 4 > a.y.bytes)
		throw MediaError("conv1d: the output allocation is smaller than " + std::to_string(a.oc) +
		                 " x " + std::to_string(lo));

	GpuKernel* pso =
	    ctx.pipeline("conv1d_gather", conv1d_gather_hlsl(), "conv1d_gather", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.ic;
	p.values[1] = (u32)a.l;
	p.values[2] = (u32)a.oc;
	p.values[3] = (u32)a.k;
	p.values[4] = (u32)lo;
	p.values[5] = (u32)a.stride;
	p.values[6] = (u32)(i32)a.pad;   // signed
	p.values[7] = (u32)a.dilation;
	p.values[8] = (u32)a.x.off;
	p.values[9] = (u32)a.wgt.off;
	p.values[10] = (u32)a.bias.off;
	p.values[11] = (u32)a.y.off;
	p.values[12] = a.has_bias ? 1u : 0u;
	p.srv[0] = a.x.res;
	p.srv[1] = a.wgt.res;
	p.srv[2] = a.bias.res ? a.bias.res : a.x.res;
	p.uav[0] = a.y.res;
	ctx.dispatch(pso, p, (u32)ceil_div(a.oc * lo, 256), 1, 1);
}

void dispatch_convt1d(ComputeContext& ctx, const ConvT1dArgsG& a) {
	if (a.ic <= 0 || a.l <= 0 || a.oc <= 0 || a.k <= 0)
		throw MediaError("convt1d: empty shape");
	if (a.stride <= 0) throw MediaError("convt1d: stride must be >= 1");
	// The two-tap gather below is only the transposed convolution when each output
	// position is reached by exactly two taps, i.e. k == 2*stride. Any other k
	// would need j = r + 2*stride, ... as well and the kernel would quietly return
	// a truncated sum - so this is a hard error (and `open()` asserts the same
	// thing where the weight shape is read).
	if (a.k != 2 * a.stride)
		throw MediaError("convt1d: k must be 2*stride for the two-tap gather, got k=" +
		                 std::to_string(a.k) + " stride=" + std::to_string(a.stride));
	const i64 lo = (a.l - 1) * a.stride - 2 * a.pad + a.k;   // ConvT1dRef::out_len
	if (lo <= 0) throw MediaError("convt1d: empty output");
	if (a.x.bytes && (u64)a.ic * (u64)a.l * 4 > a.x.bytes)
		throw MediaError("convt1d: the input allocation is smaller than " + std::to_string(a.ic) +
		                 " x " + std::to_string(a.l));
	if (a.wgt.bytes && (u64)a.ic * (u64)a.oc * (u64)a.k * 4 > a.wgt.bytes)
		throw MediaError("convt1d: the weight allocation is smaller than the [IC,OC,K] tensor");
	if (a.has_bias && a.bias.bytes && (u64)a.oc * 4 > a.bias.bytes)
		throw MediaError("convt1d: the bias allocation is smaller than OC");
	if (a.y.bytes && (u64)a.oc * (u64)lo * 4 > a.y.bytes)
		throw MediaError("convt1d: the output allocation is smaller than " + std::to_string(a.oc) +
		                 " x " + std::to_string(lo));

	GpuKernel* pso = ctx.pipeline("convt1d", convt1d_hlsl(), "convt1d", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.ic;
	p.values[1] = (u32)a.l;
	p.values[2] = (u32)a.oc;
	p.values[3] = (u32)a.k;
	p.values[4] = (u32)lo;
	p.values[5] = (u32)a.stride;
	p.values[6] = (u32)(i32)a.pad;   // signed
	p.values[7] = (u32)a.x.off;
	p.values[8] = (u32)a.wgt.off;
	p.values[9] = (u32)a.bias.off;
	p.values[10] = (u32)a.y.off;
	p.values[11] = a.has_bias ? 1u : 0u;
	p.srv[0] = a.x.res;
	p.srv[1] = a.wgt.res;
	p.srv[2] = a.bias.res ? a.bias.res : a.x.res;
	p.uav[0] = a.y.res;
	ctx.dispatch(pso, p, (u32)ceil_div(a.oc * lo, 256), 1, 1);
}

void dispatch_snake_beta(ComputeContext& ctx, const SnakeBetaArgsG& a) {
	if (a.c <= 0 || a.l <= 0) throw MediaError("snake_beta: empty shape");
	if (!a.alpha.res || !a.beta.res) throw MediaError("snake_beta: missing alpha/beta");
	const u64 n = (u64)a.c * (u64)a.l;
	if (a.x.bytes && n * 4 > a.x.bytes)
		throw MediaError("snake_beta: the input allocation is smaller than C*L elements");
	if (a.y.bytes && n * 4 > a.y.bytes)
		throw MediaError("snake_beta: the output allocation is smaller than C*L elements");
	if (a.alpha.bytes && (u64)a.c * 4 > a.alpha.bytes)
		throw MediaError("snake_beta: alpha is shorter than C");
	if (a.beta.bytes && (u64)a.c * 4 > a.beta.bytes)
		throw MediaError("snake_beta: beta is shorter than C");

	GpuKernel* pso = ctx.pipeline("snake_beta", snake_beta_hlsl(), "snake_beta",
	                              ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.c;
	p.values[1] = (u32)a.l;
	p.values[2] = (u32)a.x.off;
	p.values[3] = (u32)a.y.off;
	p.values[4] = (u32)a.alpha.off;
	p.values[5] = (u32)a.beta.off;
	p.srv[0] = a.x.res;
	p.srv[1] = a.alpha.res;
	p.srv[2] = a.beta.res;
	p.uav[0] = a.y.res;
	ctx.dispatch(pso, p, (u32)ceil_div((i64)n, 256), 1, 1);
}

void dispatch_transpose_cs(ComputeContext& ctx, const GpuAlloc& x, const GpuAlloc& y, i64 c,
                           i64 s, bool to_token) {
	const i64 total = c * s;
	if (total <= 0) throw MediaError("transpose_cs: empty");
	GpuKernel* pso =
	    ctx.pipeline("transpose_cs", transpose_cs_hlsl(), "transpose_cs", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)c;
	p.values[1] = (u32)s;
	p.values[2] = (u32)x.off;
	p.values[3] = (u32)y.off;
	p.values[4] = to_token ? 1u : 0u;
	p.srv[0] = x.res;
	p.uav[0] = y.res;
	ctx.dispatch(pso, p, (u32)ceil_div(total, 256), 1, 1);
}

void dispatch_dupup2x(ComputeContext& ctx, const DupUp2xArgs& a) {
	if (a.ic <= 0 || a.oc <= 0) throw MediaError("dupup2x: empty channel count");
	if (a.ft != 1 && a.ft != 2) throw MediaError("dupup2x: temporal factor must be 1 or 2");
	// repeats = oc*factor/ic with factor = ft*4; the checkpoint only ever uses 2 or 4.
	const i64 num = a.oc * a.ft * 4;
	if (num % a.ic) throw MediaError("dupup2x: oc*ft*4 is not a multiple of ic");
	const i64 repeats = num / a.ic;
	const i64 total = a.oc * (a.h * 2) * (a.w * 2);
	GpuKernel* pso = ctx.pipeline("dupup2x", dupup2x_hlsl(), "dupup2x", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.ic;
	p.values[1] = (u32)a.h;
	p.values[2] = (u32)a.w;
	p.values[3] = (u32)a.x.off;
	p.values[4] = (u32)a.y.off;
	p.values[5] = (u32)a.oc;
	p.values[6] = (u32)repeats;
	p.values[7] = (u32)(a.ft * 4);        // channel stride of one output channel
	p.values[8] = (u32)((a.ft - 1) * 4);  // first_chunk keeps the f_t = ft-1 slice
	p.srv[0] = a.x.res;
	p.uav[0] = a.y.res;
	ctx.dispatch(pso, p, (u32)ceil_div(total, 256), 1, 1);
}

void dispatch_mask_rect(ComputeContext& ctx, const GpuAlloc& x, i64 c, i64 h, i64 w, i64 y0,
                        i64 y1, i64 x0, i64 x1) {
	if (c <= 0 || h <= 0 || w <= 0) throw MediaError("mask_rect: empty shape");
	y0 = std::clamp<i64>(y0, 0, h);
	y1 = std::clamp<i64>(y1, 0, h);
	x0 = std::clamp<i64>(x0, 0, w);
	x1 = std::clamp<i64>(x1, 0, w);
	if (y0 == 0 && y1 == h && x0 == 0 && x1 == w) return;   // nothing outside
	GpuKernel* pso = ctx.pipeline("mask_rect", mask_rect_hlsl(), "mask_rect", ShaderModel::SM5_1);
	const u64 n = (u64)c * (u64)h * (u64)w;
	KernelParams p{};
	p.values[0] = (u32)n;
	p.values[1] = (u32)w;
	p.values[2] = (u32)h;
	p.values[3] = (u32)y0;
	p.values[4] = (u32)y1;
	p.values[5] = (u32)x0;
	p.values[6] = (u32)x1;
	p.values[7] = (u32)x.off;
	p.uav[0] = x.res;
	ctx.dispatch(pso, p, (u32)ceil_div((i64)n, 256), 1, 1);
}

void dispatch_chw_norm(ComputeContext& ctx, const ChwNormArgs& a) {
	if (a.cols <= 0 || a.rows <= 0) throw MediaError("chw_norm: empty shape");
	GpuKernel* pso =
	    ctx.pipeline("chw_rmsnorm", chw_rmsnorm_hlsl(), "chw_rmsnorm", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.rows;   // pixels
	p.values[1] = (u32)a.cols;   // channels
	p.values[2] = (u32)a.x.off;
	p.values[3] = (u32)a.y.off;
	p.values[4] = (u32)a.w.off;
	float e = a.eps;
	memcpy(&p.values[5], &e, 4);
	p.srv[0] = a.x.res;
	p.srv[1] = a.w.res;
	p.uav[0] = a.y.res;
	ctx.dispatch(pso, p, (u32)ceil_div(a.rows, 256), 1, 1);
}

void dispatch_row_softmax(ComputeContext& ctx, const GpuAlloc& x, const GpuAlloc& y, i64 rows,
                          i64 cols) {
	if (rows <= 0 || cols <= 0) throw MediaError("row_softmax: empty");
	GpuKernel* pso =
	    ctx.pipeline("row_softmax", row_softmax_hlsl(), "row_softmax", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)rows;
	p.values[1] = (u32)cols;
	p.values[2] = (u32)x.off;
	p.values[3] = (u32)y.off;
	p.srv[0] = x.res;
	p.uav[0] = y.res;
	ctx.dispatch(pso, p, (u32)rows, 1, 1);
}

void dispatch_upsample2x(ComputeContext& ctx, GpuBuffer* res, const GpuAlloc& x,
                         const GpuAlloc& y, i64 c, i64 h, i64 w) {
	(void)res;
	i64 total = c * h * 2 * w * 2;
	GpuKernel* pso =
	    ctx.pipeline("upsample2x", upsample_hlsl(), "upsample2x", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)total;
	p.values[1] = (u32)h;
	p.values[2] = (u32)w;
	p.values[3] = (u32)x.off;
	p.values[4] = (u32)y.off;
	p.srv[0] = x.res;
	p.uav[0] = y.res;
	ctx.dispatch(pso, p, (u32)ceil_div(total, 256), 1, 1);
}

// ── gemm ───────────────────────────────────────────────────────────────────

void dispatch_gemm_f16(ComputeContext& ctx, const GemmF16Args& a) {
	if (a.m <= 0 || a.n <= 0 || a.k <= 0) throw MediaError("gemm_f16: empty problem");
	GpuKernel* pso =
	    ctx.pipeline("gemm_f16", gemm_f16_hlsl(), "gemm_f16", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.m;
	p.values[1] = (u32)a.n;
	p.values[2] = (u32)a.k;
	p.values[3] = (a.a_is_f32 ? 1u : 0u) | (a.has_bias ? 2u : 0u) | (a.b_is_bf16 ? 4u : 0u) |
	             (a.b_is_f32 ? 8u : 0u);
	p.values[4] = (u32)a.a.off;
	p.values[5] = (u32)a.b.off;
	p.values[6] = (u32)a.c.off;
	p.values[7] = (u32)a.bias.off;
	p.srv[0] = a.a.res;
	p.srv[1] = a.b.res;
	p.srv[2] = a.bias.res ? a.bias.res : a.a.res;
	p.uav[0] = a.c.res;
	ctx.dispatch(pso, p, (u32)ceil_div(a.m, 64), (u32)ceil_div(a.n, 64), 1);
}

void dispatch_lora_rank_add(ComputeContext& ctx, const LoraRankAddArgs& a) {
	if (a.m <= 0 || a.n <= 0 || a.rank <= 0) throw MediaError("lora_rank_add: empty problem");
	GpuKernel* pso = ctx.pipeline("lora_rank_add", lora_rank_add_hlsl(), "lora_rank_add",
	                              ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.m;
	p.values[1] = (u32)a.n;
	p.values[2] = (u32)a.rank;
	p.values[3] = (u32)a.c.off;
	p.values[4] = (u32)a.h.off;
	p.values[5] = (u32)a.b.off;
	p.srv[0] = a.h.res;
	p.srv[1] = a.b.res;
	p.uav[0] = a.c.res;
	// One block covers a 64x64 output tile (see the kernel's tile note), and the
	// kernel holds C in registers across the whole rank.
	ctx.dispatch(pso, p, (u32)ceil_div(a.n, 64), (u32)ceil_div(a.m, 64), 1);
}

// ── layout ─────────────────────────────────────────────────────────────────

void dispatch_patchify(ComputeContext& ctx, GpuBuffer* res, const GpuAlloc& x,
                       const GpuAlloc& y, i64 c, i64 h, i64 w, int patch) {
	(void)res;
	i64 total = c * h * w;
	i64 wp = w / patch;
	GpuKernel* pso =
	    ctx.pipeline("patchify", patchify_hlsl(), "patchify", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)c;
	p.values[1] = (u32)h;
	p.values[2] = (u32)w;
	p.values[3] = (u32)patch;
	p.values[4] = (u32)x.off;
	p.values[5] = (u32)y.off;
	p.values[6] = (u32)total;
	p.values[7] = (u32)((h / patch) * wp);
	p.values[8] = (u32)wp;
	p.srv[0] = x.res;
	p.uav[0] = y.res;
	ctx.dispatch(pso, p, (u32)ceil_div(total, 256), 1, 1);
}

void dispatch_unpatchify(ComputeContext& ctx, GpuBuffer* res, const GpuAlloc& x,
                         const GpuAlloc& y, i64 c, i64 h, i64 w, int patch) {
	(void)res;
	i64 total = c * h * w;
	GpuKernel* pso =
	    ctx.pipeline("unpatchify", unpatchify_hlsl(), "unpatchify", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)c;
	p.values[1] = (u32)h;
	p.values[2] = (u32)w;
	p.values[3] = (u32)patch;
	p.values[4] = (u32)x.off;
	p.values[5] = (u32)y.off;
	p.values[6] = (u32)total;
	p.values[7] = (u32)((h / patch) * (w / patch));
	p.values[8] = (u32)(w / patch);
	p.srv[0] = x.res;
	p.uav[0] = y.res;
	ctx.dispatch(pso, p, (u32)ceil_div(total, 256), 1, 1);
}

// ── rope ───────────────────────────────────────────────────────────────────

void dispatch_rope_half(ComputeContext& ctx, GpuBuffer* res, const RopeHalfArgs& a) {
	(void)res;
	GpuKernel* pso =
	    ctx.pipeline("rope_half", rope_half_hlsl(), "rope_half", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.rows;
	p.values[1] = (u32)a.head_dim;
	p.values[2] = (u32)a.x.off;
	p.values[3] = (u32)a.y.off;
	float th = a.theta;
	memcpy(&p.values[4], &th, 4);
	p.values[5] = (u32)a.heads;
	p.values[6] = (u32)a.base;
	p.srv[0] = a.x.res;
	p.uav[0] = a.y.res;
	ctx.dispatch(pso, p, (u32)a.rows, 1, 1);
}

// ── attention ──────────────────────────────────────────────────────────────

void dispatch_qkv_prep(ComputeContext& ctx, const QkvPrepArgs& a) {
	if (a.head_dim != 128) throw MediaError("qkv_prep: head_dim must be 128");
	if (a.heads < a.kv_heads) throw MediaError("qkv_prep: kv heads > heads");
	GpuKernel* pso =
	    ctx.pipeline("qkv_prep", qkv_prep_hlsl(), "qkv_prep", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.s;
	p.values[1] = (u32)a.heads;
	p.values[2] = (u32)a.kv_heads;
	p.values[3] = (u32)a.head_dim;
	p.values[4] = (u32)a.qkv.off;
	p.values[5] = (u32)a.q.off;
	p.values[6] = (u32)a.k.off;
	p.values[7] = (u32)a.v.off;
	float eps = a.eps;
	memcpy(&p.values[8], &eps, 4);
	float sc = a.scale;
	memcpy(&p.values[9], &sc, 4);
	p.values[10] = (a.norm ? 1u : 0u) | (a.rope_on ? 2u : 0u) | (a.f32_out ? 4u : 0u);
	p.values[11] = (u32)a.ids.off;
	p.values[12] = (u32)a.rope.off;
	p.values[13] = (u32)a.ids_row;
	// qkv/q/k/v are addressed with the byte offsets in the constant buffer;
	// the small side tables (norms, rope table, ids) have no offset slot of
	// their own, so they ride the root descriptor instead.
	p.srv[0] = a.qkv.res;
	p.srv[1] = a.wq.res ? a.wq.res : a.qkv.res;
	p.srv[2] = a.wk.res ? a.wk.res : a.qkv.res;
	p.srv[3] = a.rope.res;
	p.srv[4] = a.ids.res;
	p.srv_offset[1] = a.wq.res ? a.wq.off : a.qkv.off;
	p.srv_offset[2] = a.wk.res ? a.wk.off : a.qkv.off;
	p.srv_offset[3] = a.rope.off;
	p.srv_offset[4] = a.ids.off;
	p.uav[0] = a.q.res;
	p.uav[1] = a.k.res;
	p.uav[2] = a.v.res;
	ctx.dispatch(pso, p, (u32)a.s, (u32)a.heads, 1);
}

namespace {

// PHI_ATTN=0 forces the original one-load-per-FMA kernel; PHI_ATTN=2 forces the
// tiled one. Unset picks the tiled kernel for every fp16 call (the DiT and the
// 4B text encoder), because that is the whole point of having it.
int attn_kernel_choice() {
	static int once = [] {
		if (auto e = get_env("PHI_ATTN")) return atoi(e->c_str());
		return 2;
	}();
	return once;
}

}  // namespace

void dispatch_attn_flash(ComputeContext& ctx, const AttnFlashArgs& a) {
	if (a.head_dim != 128) throw MediaError("attn_flash: head_dim must be 128");
	// The tiled kernel reads packed fp16 words; the fp32 q/k/v path stays on the
	// original kernel, which is the only one that understands interleaved fp32.
	const bool tiled = !a.f32_input && attn_kernel_choice() >= 2;
	// The per-row key bound lives in the tiled kernel only; the fp32 path would
	// silently ignore it, which is worse than refusing.
	if ((bool)a.ub && !tiled)
		throw MediaError("attn_flash: a per-row mask needs the tiled kernel (PHI_ATTN>=2 and fp16 q/k/v)");
	if (tiled) {
		// Same query tile as the DiT's own attention (see attn_tiled_src): the
		// image DiT runs the same kernel at S ~ 4-5k, where the K/V tile no longer
		// fits in L2 either.
		const unsigned bm = attn_query_tile();
		const std::string ak = attn_tiled_name(bm, 128u);
		GpuKernel* tpso =
		    ctx.pipeline(ak, attn_tiled_src(bm, 128u), "attn_tiled", ShaderModel::SM6_2);
		KernelParams tp{};
		tp.values[0] = (u32)a.s;
		tp.values[1] = (u32)a.heads;
		tp.values[2] = (u32)a.head_dim;
		tp.values[3] = (u32)a.q.off;
		tp.values[4] = (u32)a.k.off;
		tp.values[5] = (u32)a.v.off;
		tp.values[6] = (u32)a.o.off;
		float tsc = a.scale;
		memcpy(&tp.values[7], &tsc, 4);
		tp.values[8] = (u32)(a.heads * a.head_dim);
		i64 tkvh = a.kv_heads ? a.kv_heads : a.heads;
		tp.values[9] = (u32)tkvh;
		tp.values[10] = (u32)(tkvh * a.head_dim);
		tp.values[11] = (a.causal ? 2u : 0u) | (a.ub ? 4u : 0u);
		tp.srv[0] = a.q.res;
		tp.srv[1] = a.k.res;
		tp.srv[2] = a.v.res;
		// srv[3] is the per-row key bound when the mask is on; the kernel never
		// touches it otherwise, so an unset mask binds anything valid.
		tp.srv[3] = a.ub ? a.ub.res : a.q.res;
		tp.srv_offset[3] = a.ub.off;
		tp.uav[0] = a.o.res;
		ctx.dispatch(tpso, tp, (u32)ceil_div(a.query_rows(), (i64)bm), (u32)a.heads, 1);
		return;
	}
	GpuKernel* pso =
	    ctx.pipeline("attn_flash", attn_flash_hlsl(), "attn_flash", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.s;
	p.values[1] = (u32)a.heads;
	p.values[2] = (u32)a.head_dim;
	p.values[3] = (u32)a.q.off;
	p.values[4] = (u32)a.k.off;
	p.values[5] = (u32)a.v.off;
	p.values[6] = (u32)a.o.off;
	float sc = a.scale;
	memcpy(&p.values[7], &sc, 4);
	p.values[8] = (u32)(a.heads * a.head_dim);
	i64 kvh = a.kv_heads ? a.kv_heads : a.heads;
	p.values[9] = (u32)kvh;
	p.values[10] = (u32)(kvh * a.head_dim);
	p.values[11] = (a.f32_input ? 1u : 0u) | (a.causal ? 2u : 0u);
	p.srv[0] = a.q.res;
	p.srv[1] = a.k.res;
	p.srv[2] = a.v.res;
	p.uav[0] = a.o.res;
	ctx.dispatch(pso, p, (u32)ceil_div(a.s, 32), (u32)a.heads, 1);
}


}  // namespace phi::media
