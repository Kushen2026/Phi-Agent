// Qwen3-VL-32B text encoder — W5 implementation.
//
// Reference: comfy/text_encoders/llama.py (Llama2_ / TransformerBlock /
// Attention / MLP / precompute_freqs_cis / apply_rope) plus
// comfy/text_encoders/qwen3vl.py and minimax.py for the H3 presentation. The
// shapes come from the checkpoint header, read off tensor by tensor.
//
// The tower is 50 of the original 64 layers, truncated, with **no final norm**
// (`final_norm = False` in Qwen3VL_32BConfig) — the H3 DiT conditions on the raw
// last-layer hidden state.
//
// Per layer:
//   t = input_layernorm(x)                      RMSNorm, eps 1e-6, no bias
//   q = int4_linear(t, q_proj)   [S, 8192]      64 heads x 128
//   k = int4_linear(t, k_proj)   [S, 1024]      8 heads x 128   (GQA)
//   v = int4_linear(t, v_proj)   [S, 1024]
//   q = q_norm(q), k = k_norm(k)                per-head RMSNorm, eps 1e-6
//   q, k = rope_half(q, k)                      theta 5e6, pairs (i, i+64)
//   o = attn(q, k, v)                           causal, 64 heads, 8 kv
//   x = x + o_proj(o)
//   t = post_attention_layernorm(x)
//   x = x + down_proj(silu(gate(t)) * up(t))
//
// The int4 GEMM is a dedicated kernel (`h3_kernels.cpp`): the checkpoint stores
// two signed nibbles per byte along K and the activations were quantised over the
// same 4-bit range (absmax/7), so both sides must carry those codes for the
// integer product to be exact. `convrot` is applied to the activations at
// runtime by the same `quant_convrot` kernel the int8 path uses, with qmax = 7.
//
// The weights stream one layer at a time (244 MB per layer through the upload
// ring); the activation footprint at a few hundred tokens is ~40 MB, so this
// tower is bandwidth-bound and needs no residency plan of its own.
#include "models/text_encoder_32b.hpp"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "kernels/h3_kernels.hpp"
#include "host/quant.hpp"
#include "models/lora.hpp"
#include "models/video_vae.hpp"   // for nothing but the shared headers it pulls in
#include "runtime/vram_budget.hpp"
#include "runtime/vram_window.hpp"

namespace phi::media {

namespace {

// One projection of the 32B tower, in the checkpoint's own precision:
//   * the original int8 tensorwise pair (one code per element + per-row fp32
//     scale) - uploaded verbatim, on the int8 GEMM;
//   * the shipped convrot_w4a4 pair (packed int4 + per-row scale) - also uploaded
//     verbatim, on the int4 GEMM;
//   * a float source (f16 / bf16 / f32, fp8 -> f16) - kept at its own precision and
//     run through the dense fp16/bf16/f32 GEMM, *not* repacked to int4.
//
// `TeLinear32` / `TeLayerW` live in the header so the resident window can hold a
// `std::vector<TeLayerW>` (see `plan_residency`).

void up_pair(SafeTensors& st, GpuCtx& g, GpuArena& arena, const std::string& wname, TeLinear32& L,
             const LoraSet* lora = nullptr) {
	const StTensor& t = st.require(wname);
	if (t.shape.size() != 2) throw MediaError("qwen3vl: " + wname + " is not a 2-D matrix");
	const std::string base = wname.substr(0, wname.size() - 7);
	// The original int8 tensorwise form comes first: `is_plain_int8` is the only
	// test that tells it apart from the packed convrot_w4a4 pair (both are stored
	// as I8 with a `weight_scale` beside them; the comfy_quant blob is what says
	// which). Treating an int8 weight as int4 would feed the int4 GEMM half a row
	// of codes, so the order matters and the check is the layout's, not the dtype's.
	if (st.is_plain_int8(t)) {
		L.w = arena.alloc(t.nbytes);
		upload_file_range(*g.ctx, *g.ring, L.w.res, L.w.off, st, t);
		const StTensor& sc = st.require(base + ".weight_scale");
		std::vector<float> v = tensor_to_f32(st, sc);
		if ((i64)v.size() != t.shape[0])
			throw MediaError("qwen3vl: " + base + ".weight_scale length mismatch");
		L.s = arena.alloc((u64)v.size() * 4);
		upload_range(*g.ctx, *g.ring, L.s.res, L.s.off, v.data(), v.size() * 4);
		L.i8 = true;
		return;
	}
	if (t.dtype == DType::I8 && st.find(base + ".weight_scale")) {
		// Shipped convrot_w4a4: the packed int4 + per-row scale upload verbatim.
		L.w = arena.alloc(t.nbytes);
		// Off the file, not through the mapping: the tower's 13.9 GB is read once per
		// invocation and its pages are always cold, which is where the mapping's ~0.74
		// GB/s against ReadFile's 2.1 GB/s is the difference between a 14 s text phase
		// and a 7 s one.
		upload_file_range(*g.ctx, *g.ring, L.w.res, L.w.off, st, t);
		const StTensor& sc = st.require(base + ".weight_scale");
		std::vector<float> v = tensor_to_f32(st, sc);
		L.s = arena.alloc((u64)v.size() * 4);
		upload_range(*g.ctx, *g.ring, L.s.res, L.s.off, v.data(), v.size() * 4);
		return;
	}
	if (weight_is_dense_float(st, t)) {
		// A float checkpoint keeps its precision: the dense GEMM, no float -> int4
		// conversion (goal #1). A LoRA that touches the module is folded in.
		DenseUpload du = load_dense_linear(g, st, base, &arena, lora);
		L.d = du.w;
		L.dtype = du.dtype;
		return;
	}
	// Any other packed family: repacked onto the engine's convrot_w4a4 path.
	I4Upload up = upload_quant_linear_w4a4(g, st, base, arena, lora);
	L.w = up.w;
	L.s = up.s;
}

GpuAlloc up_f32(SafeTensors& st, GpuCtx& g, GpuArena& arena, const std::string& name) {
	const StTensor& t = st.require(name);
	std::vector<float> v = tensor_to_f32(st, t);
	GpuAlloc a = arena.alloc((u64)v.size() * 4);
	upload_range(*g.ctx, *g.ring, a.res, a.off, v.data(), v.size() * 4);
	return a;
}

void load_layer(SafeTensors& st, GpuCtx& g, GpuArena& arena, i64 i, TeLayerW& L,
                const LoraSet* lora) {
	const std::string p = "model.layers." + std::to_string(i) + ".";
	up_pair(st, g, arena, p + "self_attn.q_proj.weight", L.q, lora);
	up_pair(st, g, arena, p + "self_attn.k_proj.weight", L.k, lora);
	up_pair(st, g, arena, p + "self_attn.v_proj.weight", L.v, lora);
	up_pair(st, g, arena, p + "self_attn.o_proj.weight", L.o, lora);
	up_pair(st, g, arena, p + "mlp.gate_proj.weight", L.gate, lora);
	up_pair(st, g, arena, p + "mlp.up_proj.weight", L.up, lora);
	up_pair(st, g, arena, p + "mlp.down_proj.weight", L.down, lora);
	L.in_ln = up_f32(st, g, arena, p + "input_layernorm.weight");
	L.post_ln = up_f32(st, g, arena, p + "post_attention_layernorm.weight");
	L.q_norm = up_f32(st, g, arena, p + "self_attn.q_norm.weight");
	L.k_norm = up_f32(st, g, arena, p + "self_attn.k_norm.weight");
}

// The Qwen3-VL `convrot_w4a4` activation range: the packer used absmax/7.
constexpr float kInt4Qmax = 7.0f;

// Stage trace (PHI_TE_DEBUG=1). A 50-layer stream of 244 MB weights per layer is
// minutes long; when something stops, the only useful question is which layer and
// which phase, and this is what answers it.
bool te_debug() {
	static int v = -1;
	if (v < 0) {
		const char* e = getenv("PHI_TE_DEBUG");
		v = (e && *e && *e != '0') ? 1 : 0;
	}
	return v != 0;
}

#define TLOG(...)                        \
	do {                                 \
		if (te_debug()) {                \
			fprintf(stderr, __VA_ARGS__); \
			fputc(10, stderr);           \
			fflush(stderr);              \
		}                                \
	} while (0)

// ── vision tower kernels ───────────────────────────────────────────────────
//
// Three ops the ViT needs that the frozen dispatchers cannot express: the
// attention they expose requires head_dim 128 (the ViT's is 1152/16 = 72), the
// rope applies one position per token (the ViT needs a row position for its
// first frequency band and a column position for the second), and there is no
// GELU-tanh anywhere in `ElemOp`. They are compiled by NVRTC out of the sources
// below and keep the same frozen ABI as every ported kernel: `Args{v[24],
// s[8], u[4]}`, byte offsets in `a.v[]`, resources in `a.s[]`/`a.u[]`.
static const char* kVitRopeCuda = R"CUDA(
struct Args { unsigned v[24]; const char* s[8]; char* u[4]; };
__device__ __forceinline__ float ldF(const char* b, unsigned o) { return __uint_as_float(*(const unsigned*)(b + o)); }
__device__ __forceinline__ void stF(char* b, unsigned o, float v) { *(unsigned*)(b + o) = __float_as_uint(v); }

// cbuffer P: uint S, D, xOff, stride, heads, csOff;
//   S      tokens
//   D      head_dim (72)
//   xOff   byte offset of the section (q or k) inside the fused qkv buffer
//   stride elements between consecutive tokens (= 3 * heads * D)
//   heads  query heads
//   csOff  byte offset of the per-token cos|sin table ([S, D] fp32: D/2 cos
//          values then D/2 sin values)
// srv[0] = QKV   srv[1] = CS   uav[0] = QKV (in place)
//
// y[i] = x[i]*cos - x[i+D/2]*sin ; y[i+D/2] = x[i+D/2]*cos + x[i]*sin, for the
// D/2 pairs of one head, with the angle of pair i out of the table. The first
// half of the pairs uses the patch row and the second half the patch column
// (Qwen3-VL's vision rotary), which is why the table is per token and not
// derivable from the row index the way `rope_half` derives it.
//
// In place is safe: the pairs (i, i+D/2) are disjoint and each one is read and
// written by a single thread.
// numthreads(128, 1, 1) - grid (S, 1)
extern "C" __global__ void vit_rope(Args a) {
	const unsigned S = a.v[0], D = a.v[1], xOff = a.v[2], stride = a.v[3];
	const unsigned heads = a.v[4], csOff = a.v[5];
	const unsigned D2 = D / 2u;
	const char* X = a.s[0];
	const char* CS = a.s[1];
	char* Y = a.u[0];
	const unsigned tk = blockIdx.x;
	if (tk >= S) return;
	const unsigned base = xOff + tk * stride * 4u;
	const unsigned csb = csOff + tk * 2u * D2 * 4u;
	const unsigned npair = heads * D2;
	for (unsigned p = threadIdx.x; p < npair; p += blockDim.x) {
		const unsigned h = p / D2, i = p % D2;
		const float c = ldF(CS, csb + i * 4u);
		const float sn = ldF(CS, csb + (D2 + i) * 4u);
		const unsigned e = base + (h * D + i) * 4u;
		const float x0 = ldF(X, e), x1 = ldF(X, e + D2 * 4u);
		stF(Y, e, x0 * c - x1 * sn);
		stF(Y, e + D2 * 4u, x1 * c + x0 * sn);
	}
}
)CUDA";

static const char* kVitAttnCuda = R"CUDA(
struct Args { unsigned v[24]; const char* s[8]; char* u[4]; };
__device__ __forceinline__ float ldF(const char* b, unsigned o) { return __uint_as_float(*(const unsigned*)(b + o)); }
__device__ __forceinline__ void stF(char* b, unsigned o, float v) { *(unsigned*)(b + o) = __float_as_uint(v); }

// cbuffer P: uint S, H, D, qOff, kOff, vOff, oOff; float scale; uint stride,
//            token0, dgrp;
//   S      tokens in this attention sequence (one reference frame)
//   qOff   byte offset of q inside the fused qkv buffer (kOff/vOff likewise)
//   stride elements between consecutive tokens of the fused buffer
//   token0 first token of this sequence inside the buffers
//   dgrp   output dims per thread group of 4 (= D/4; D must be a multiple of 4)
// srv[0] = QKV (one buffer, three sections)   uav[0] = O ([S_total, H*D] fp32)
//
// Non-causal full attention for one (sequence, head), one-load-per-FMA flash
// structure: 32 query rows x 32 keys per step, online softmax, fp32 in and out.
// The ViT's head_dim is 72, which `attn_flash` refuses, and the sequence is one
// image frame - the ViT never attends across references.
// numthreads(128, 1, 1) - grid (ceil(S/32), H)
extern "C" __global__ void vit_attn(Args a) {
	const unsigned S = a.v[0], H = a.v[1], D = a.v[2];
	const unsigned qOff = a.v[3], kOff = a.v[4], vOff = a.v[5], oOff = a.v[6];
	const float scale = __uint_as_float(a.v[7]);
	const unsigned stride = a.v[8], token0 = a.v[9], dgrp = a.v[10];
	const char* QKV = a.s[0];
	char* O = a.u[0];

	const unsigned BM = 32u, BN = 32u, MAXD = 32u;
	__shared__ float sP[BM][BN];

	const unsigned h = blockIdx.y;
	const unsigned qBase = blockIdx.x * BM;
	const unsigned t = threadIdx.x;
	// The four threads of a query row are lanes 4r..4r+3 of one warp, which is
	// what lets the softmax reduction be a shuffle instead of a shared-memory
	// hand-off between a designated writer and everybody else.
	const unsigned r = t / 4u, g = t % 4u;
	const unsigned qIdx = qBase + r;
	// Grid rounds the query axis up to whole BM blocks, so the tail block can
	// name rows >= S. Rows are independent, so such a row reads row 0 (the
	// `attn_flash` guard) and is never stored.
	const bool qOk = qIdx < S;
	const unsigned grow = token0 + (qOk ? qIdx : 0u);
	const unsigned qRow = qOff + (grow * stride + h * D) * 4u;
	const unsigned kSec = kOff + (token0 * stride + h * D) * 4u;
	const unsigned vSec = vOff + (token0 * stride + h * D) * 4u;

	float outAcc[MAXD];
	#pragma unroll
	for (unsigned i = 0; i < MAXD; i++) outAcc[i] = 0.0f;
	// Running softmax state, held in registers and identical in all four threads
	// of the row (each one reduces over its own 8 keys and then over the four).
	float rowMax = -3.0e38f, rowSum = 0.0f;

	const unsigned kTiles = (S + BN - 1u) / BN;
	for (unsigned kt = 0; kt < kTiles; kt++) {
		const unsigned kBase = kt * BN;
		float acc[8];
		#pragma unroll
		for (unsigned j = 0; j < 8; j++) acc[j] = 0.0f;
		const unsigned kRow = kBase + g * 8u;
		if (kRow < S) {
			for (unsigned d = 0; d < D; d++) {
				const float qv = ldF(QKV, qRow + d * 4u);
				#pragma unroll
				for (unsigned j = 0; j < 8; j++) {
					const unsigned kj = kRow + j;
					if (kj < S) acc[j] += qv * ldF(QKV, kSec + (kj * stride + d) * 4u);
				}
			}
		}
		float lmax = -3.0e38f;
		#pragma unroll
		for (unsigned j = 0; j < 8; j++)
			if (kBase + g * 8u + j < S) lmax = fmaxf(lmax, acc[j] * scale);
		lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 1));
		lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 2));

		const float newmax = fmaxf(rowMax, lmax);
		const float corr = (rowMax <= -3.0e37f) ? 0.0f : expf(rowMax - newmax);
		float ps[8];
		float lsum = 0.0f;
		#pragma unroll
		for (unsigned j = 0; j < 8; j++) {
			const bool ok = (kBase + g * 8u + j) < S;
			ps[j] = ok ? expf(acc[j] * scale - newmax) : 0.0f;
			lsum += ps[j];
		}
		lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
		lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
		rowMax = newmax;
		rowSum = rowSum * corr + lsum;
		#pragma unroll
		for (unsigned j = 0; j < 8; j++) sP[r][g * 8u + j] = ps[j];
		__syncthreads();
		{
			const unsigned dBase = g * dgrp;
			#pragma unroll
			for (unsigned i = 0; i < MAXD; i++) outAcc[i] *= corr;
			for (unsigned j = 0; j < BN; j++) {
				if (kBase + j >= S) continue;
				const float p = sP[r][j];
				if (p == 0.0f) continue;
				const unsigned vRow = vSec + ((kBase + j) * stride + dBase) * 4u;
				#pragma unroll
				for (unsigned i = 0; i < MAXD; i++)
					if (i < dgrp) outAcc[i] += p * ldF(QKV, vRow + i * 4u);
			}
		}
		__syncthreads();
	}

	const float inv = 1.0f / fmaxf(rowSum, 1e-20f);
	if (qOk) {
		const unsigned dBase = g * dgrp;
		const unsigned oRow = oOff + ((token0 + qIdx) * (H * D) + h * D + dBase) * 4u;
		#pragma unroll
		for (unsigned i = 0; i < MAXD; i++)
			if (i < dgrp && dBase + i < D) stF(O, oRow + i * 4u, outAcc[i] * inv);
	}
}
)CUDA";

static const char* kVitGeluCuda = R"CUDA(
struct Args { unsigned v[24]; const char* s[8]; char* u[4]; };
__device__ __forceinline__ float ldF(const char* b, unsigned o) { return __uint_as_float(*(const unsigned*)(b + o)); }
__device__ __forceinline__ void stF(char* b, unsigned o, float v) { *(unsigned*)(b + o) = __float_as_uint(v); }

// cbuffer P: uint n, xOff, yOff;   srv[0] = X   uav[0] = Y (may be X)
// gelu(x, approximate="tanh"), the ViT MLP's activation (and the merger's).
// numthreads(256, 1, 1)
extern "C" __global__ void vit_gelu(Args a) {
	const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= a.v[0]) return;
	const float x = ldF(a.s[0], a.v[1] + i * 4u);
	const float u = 0.044715f * x * x * x;
	stF(a.u[0], a.v[2] + i * 4u, 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + u))));
}
)CUDA";

static const char* kVitLnCuda = R"CUDA(
struct Args { unsigned v[24]; const char* s[8]; char* u[4]; };
__device__ __forceinline__ float ldF(const char* b, unsigned o) { return __uint_as_float(*(const unsigned*)(b + o)); }
__device__ __forceinline__ void stF(char* b, unsigned o, float v) { *(unsigned*)(b + o) = __float_as_uint(v); }

// cbuffer P: uint rows, cols, xOff, wOff, bOff, yOff; float eps;
// srv[0] = X [rows, cols]   srv[1] = W [cols]   srv[2] = B [cols]   uav[0] = Y
//
// y[r, c] = (x[r, c] - mean(r)) * rsqrt(var(r) + eps) * w[c] + b[c], the ViT's
// (bias-having) LayerNorm, bias fused.
//
// This is the ViT's own kernel rather than `dispatch_norm` + a row-broadcast
// `elem` for two reasons:
//   * `norm_rows` reads bOff == 0 as "no bias", and an arena allocation that
//     lands at byte 0 of a chunk would silently lose its bias, so the shared
//     path adds the bias in a second pass;
//   * the shared kernel's LayerNorm path reuses its shared reduction slot for
//     the variance partials without a barrier between "every thread has read
//     the mean" and "thread 0 overwrites red[0]", which is a real read/write
//     race. The barrier below is the fix.
// The tree (halving strides over 256 partials, mean then variance, in that
// order) is the same one `norm_rows` used, so the numbers this replaced stay
// comparable.
// numthreads(256, 1, 1) - grid (rows, 1)
extern "C" __global__ void vit_ln(Args a) {
	const unsigned rows = a.v[0], cols = a.v[1];
	const unsigned xOff = a.v[2], wOff = a.v[3], bOff = a.v[4], yOff = a.v[5];
	const float eps = __uint_as_float(a.v[6]);
	const char* X = a.s[0];
	const char* W = a.s[1];
	const char* B = a.s[2];
	char* Y = a.u[0];

	__shared__ float red[256];
	const unsigned row = blockIdx.x;
	if (row >= rows) return;
	const unsigned t = threadIdx.x;
	const unsigned base = xOff + row * cols * 4u;
	const unsigned obase = yOff + row * cols * 4u;

	float part = 0.0f;
	for (unsigned c = t; c < cols; c += 256u) part += ldF(X, base + c * 4u);
	red[t] = part;
	__syncthreads();
	#pragma unroll
	for (unsigned s = 128u; s > 0u; s >>= 1) {
		if (t < s) red[t] += red[t + s];
		__syncthreads();
	}
	const float mean = red[0] / (float)cols;
	// Without this barrier every thread's read of red[0] above is unordered
	// against thread 0's `red[0] = part2` below: a warp whose load is issued
	// after that store reads a variance partial as the mean, and every column
	// that warp owns is normalised by the wrong constant.
	__syncthreads();

	float part2 = 0.0f;
	for (unsigned c = t; c < cols; c += 256u) {
		const float d = ldF(X, base + c * 4u) - mean;
		part2 += d * d;
	}
	red[t] = part2;
	__syncthreads();
	#pragma unroll
	for (unsigned s = 128u; s > 0u; s >>= 1) {
		if (t < s) red[t] += red[t + s];
		__syncthreads();
	}
	const float inv = rsqrtf(red[0] / (float)cols + eps);

	for (unsigned c = t; c < cols; c += 256u) {
		const float nv = (ldF(X, base + c * 4u) - mean) * inv;
		stF(Y, obase + c * 4u, nv * ldF(W, wOff + c * 4u) + ldF(B, bOff + c * 4u));
	}
}
)CUDA";

GpuKernel* vit_rope_kernel(ComputeContext& ctx) {
	return ctx.pipeline("vit_rope", kVitRopeCuda, "vit_rope", ShaderModel::SM5_1, 128, 1, 1);
}
GpuKernel* vit_attn_kernel(ComputeContext& ctx) {
	return ctx.pipeline("vit_attn", kVitAttnCuda, "vit_attn", ShaderModel::SM5_1, 128, 1, 1);
}
GpuKernel* vit_gelu_kernel(ComputeContext& ctx) {
	return ctx.pipeline("vit_gelu", kVitGeluCuda, "vit_gelu", ShaderModel::SM5_1, 256, 1, 1);
}
GpuKernel* vit_ln_kernel(ComputeContext& ctx) {
	return ctx.pipeline("vit_ln", kVitLnCuda, "vit_ln", ShaderModel::SM5_1, 256, 1, 1);
}

// bf16 tensor -> device as bf16, straight out of the checkpoint (the fp16 GEMM
// reads bf16 B operands directly, so there is nothing to convert).
GpuAlloc up_bf16(SafeTensors& st, GpuCtx& g, GpuArena& arena, const std::string& name) {
	const StTensor& t = st.require(name);
	if (t.dtype != DType::BF16)
		throw MediaError("qwen3vl(visual): " + name + " is not bf16");
	GpuAlloc a = arena.alloc(t.nbytes);
	upload_file_range(*g.ctx, *g.ring, a.res, a.off, st, t);
	return a;
}

// The ViT's LayerNorm (bias included) - see `vit_ln` above for why it is not
// `dispatch_norm` plus a row-broadcast `elem`.
void vit_layer_norm(GpuCtx& g, const GpuAlloc& x, const GpuAlloc& w, const GpuAlloc& b, i64 rows,
                    i64 cols, float eps, const GpuAlloc& y) {
	GpuKernel* k = vit_ln_kernel(*g.ctx);
	KernelParams p{};
	p.values[0] = (u32)rows;
	p.values[1] = (u32)cols;
	p.values[2] = (u32)x.off;
	p.values[3] = (u32)w.off;
	p.values[4] = (u32)b.off;
	p.values[5] = (u32)y.off;
	memcpy(&p.values[6], &eps, 4);
	p.srv[0] = x.res;
	p.srv[1] = w.res;
	p.srv[2] = b.res;
	p.uav[0] = y.res;
	g.ctx->dispatch(k, p, (u32)rows, 1, 1);
}

void vit_rope_run(ComputeContext& ctx, const GpuAlloc& qkv, const GpuAlloc& cs, i64 S, i64 heads,
                  i64 D, i64 stride, i64 section) {
	GpuKernel* k = vit_rope_kernel(ctx);
	KernelParams p{};
	p.values[0] = (u32)S;
	p.values[1] = (u32)D;
	p.values[2] = (u32)(qkv.off + (u64)section * (u64)heads * (u64)D * 4u);
	p.values[3] = (u32)stride;
	p.values[4] = (u32)heads;
	p.values[5] = (u32)cs.off;
	p.srv[0] = qkv.res;
	p.srv[1] = cs.res;
	p.uav[0] = qkv.res;
	ctx.dispatch(k, p, (u32)S, 1, 1);
}

void vit_attn_run(ComputeContext& ctx, const GpuAlloc& qkv, const GpuAlloc& o, i64 token0, i64 seq_len,
                  i64 heads, i64 D, i64 stride) {
	GpuKernel* k = vit_attn_kernel(ctx);
	KernelParams p{};
	const float scale = 1.0f / std::sqrt((float)D);
	p.values[0] = (u32)seq_len;
	p.values[1] = (u32)heads;
	p.values[2] = (u32)D;
	// Byte offsets into the fused qkv buffer, resource base included: the kernel
	// adds them to the bare resource pointer it is handed.
	p.values[3] = (u32)qkv.off;                                        // q
	p.values[4] = (u32)(qkv.off + (u64)heads * (u64)D * 4u);           // k
	p.values[5] = (u32)(qkv.off + 2ull * (u64)heads * (u64)D * 4u);    // v
	p.values[6] = (u32)o.off;
	memcpy(&p.values[7], &scale, 4);
	p.values[8] = (u32)stride;
	p.values[9] = (u32)token0;
	p.values[10] = (u32)(D / 4);
	p.srv[0] = qkv.res;
	p.uav[0] = o.res;
	const u32 gx = (u32)((seq_len + 31) / 32);
	ctx.dispatch(k, p, gx, (u32)heads, 1);
}

void vit_gelu_run(ComputeContext& ctx, const GpuAlloc& x, i64 n) {
	GpuKernel* k = vit_gelu_kernel(ctx);
	KernelParams p{};
	p.values[0] = (u32)n;
	p.values[1] = (u32)x.off;
	p.values[2] = (u32)x.off;
	p.srv[0] = x.res;
	p.uav[0] = x.res;
	ctx.dispatch(k, p, (u32)((n + 255) / 256), 1, 1);
}

void vit_gemm(ComputeContext& ctx, const GpuAlloc& a, const GpuAlloc& b, const GpuAlloc& bias,
              const GpuAlloc& c, i64 m, i64 n, i64 kk) {
	GemmF16Args ga;
	ga.a = a;
	ga.b = b;
	ga.c = c;
	ga.bias = bias;
	ga.m = m;
	ga.n = n;
	ga.k = kk;
	ga.a_is_f32 = true;    // activations stay fp32; only the weights are bf16
	ga.b_is_bf16 = true;
	ga.has_bias = bias.res != nullptr;
	dispatch_gemm_f16(ctx, ga);
}

// fp32 linear + fp32 bias
void vit_add(GpuCtx& g, const GpuAlloc& dst, const GpuAlloc& src, i64 rows, i64 cols) {
	ElemArgs ea;
	ea.op = ElemOp::Add;
	ea.a = dst;
	ea.b = src;
	ea.y = dst;
	ea.rows = rows;
	ea.cols = cols;
	dispatch_elem(*g.ctx, ea);
}

}  // namespace

void TextEncoder32B::open(const std::string& path, GpuCtx* gpu) {
	if (!gpu || !gpu->ok()) throw MediaError("qwen3vl: open() needs a live GpuCtx");
	g_ = gpu;
	path_ = path;
	st_.open(path);

	const StTensor& emb = st_.require("model.embed_tokens.weight");
	if (emb.shape.size() != 2) throw MediaError("qwen3vl: unexpected embedding shape");
	cfg_.vocab = emb.shape[0];
	cfg_.hidden = emb.shape[1];
	wdtype_ = emb.dtype;

	i64 layers = 0;
	while (st_.find("model.layers." + std::to_string(layers) + ".input_layernorm.weight")) layers++;
	cfg_.n_layers = layers;
	if (cfg_.out_layer >= cfg_.n_layers) cfg_.out_layer = cfg_.n_layers - 1;

	// Shapes, checked against the header constants the struct carries: a
	// mismatch here would otherwise surface as a wrong-looking video.
	const StTensor& q = st_.require("model.layers.0.self_attn.q_proj.weight");
	const StTensor& k = st_.require("model.layers.0.self_attn.k_proj.weight");
	const StTensor& g = st_.require("model.layers.0.mlp.gate_proj.weight");
	if (q.shape[0] != cfg_.n_heads * cfg_.head_dim)
		throw MediaError("qwen3vl: q_proj is not 64 x 128");
	if (k.shape[0] != cfg_.n_kv_heads * cfg_.head_dim)
		throw MediaError("qwen3vl: k_proj is not 8 x 128");
	if (g.shape[0] != cfg_.intermediate) throw MediaError("qwen3vl: gate_proj is not 25600 wide");
	// A packed int4 pair halves the stored K; a plain int8 tensorwise weight
	// stores one code per element, so its K is the full hidden width. The order
	// matters: both are I8 with a `weight_scale` beside them, and only the layout
	// (`is_plain_int8`) tells them apart - testing the dtype alone made an int8
	// checkpoint fail the halving check with a confusing message.
	const bool int4_ckpt = !st_.is_plain_int8(g) && g.dtype == DType::I8 &&
	                       st_.find("model.layers.0.mlp.gate_proj.weight_scale") != nullptr;
	if (int4_ckpt) {
		if (g.shape[1] * 2 != cfg_.hidden)
			throw MediaError("qwen3vl: the int4 K axis does not halve");
	} else if (g.shape[1] != cfg_.hidden) {
		throw MediaError("qwen3vl: gate_proj input width does not match the embedding width");
	}
	// The embedding table is gathered on the host (a
	// device copy is 1.5 GB of residency that one gather reads).
	open_ = true;
}

std::vector<float> TextEncoder32B::encode(const std::vector<i32>& ids) {
	if (!open_) throw MediaError("qwen3vl: encode() before open()");
	const i64 S = (i64)ids.size();
	if (S <= 0) throw MediaError("qwen3vl: empty token sequence");
	const i64 H = cfg_.hidden;

	// ── embedding gather (host side, straight out of the mapping) ──────────
	TLOG("[te] gathering %lld embedding rows", (long long)S);
	std::vector<float> xf((size_t)S * H);
	{
		const StTensor& t = st_.require("model.embed_tokens.weight");
		const size_t esz = (t.dtype == DType::BF16 || t.dtype == DType::F16) ? 2 : 4;
		std::vector<u8> row((size_t)H * esz);
		for (i64 s = 0; s < S; s++) {
			const i32 id = ids[(size_t)s];
			if (id < 0 || id >= cfg_.vocab) throw MediaError("qwen3vl: token id out of range");
			memcpy(row.data(), (const u8*)st_.data_of(t) + (u64)id * H * esz, row.size());
			for (i64 d = 0; d < H; d++) {
				if (wdtype_ == DType::BF16) {
					const u16 h = *(const u16*)&row[(size_t)d * 2];
					xf[(size_t)(s * H + d)] = bf16_to_f32(h);
				} else if (wdtype_ == DType::F16) {
					const u16 h = *(const u16*)&row[(size_t)d * 2];
					xf[(size_t)(s * H + d)] = f16_to_f32(h);
				} else {
					memcpy(&xf[(size_t)(s * H + d)], &row[(size_t)d * 4], 4);
				}
			}
		}
	}

	return encode_embeds(xf.data(), S);
}

// ── the resident layer window ────────────────────────────────────────────────
//
// See the header for the policy. The formula, in the terms vram_window.hpp takes:
//
//   limit = VramBudget::limit()           min(96 % x the driver's figure for
//                                         this process, the card's own ceiling)
//   other = held + one_layer + slack      everything the window must fit beside
//   window(n) = window_charge_bytes(n, one_layer, chunk)   booked in chunks
//   n = max{ k : window(k) <= limit - other }   capped at the layer count
//
// Every term is measured rather than assumed: `held` is the ledger *after* the
// activation frame is allocated (so the prompt length and the token count are
// already in it), `one_layer` is the arena growth the upload loop itself measures
// for a layer, and `chunk` is the granularity `arena_chunk_bytes_for` derives from
// this machine's budget. That is what makes the same code fill 90-98 % of a 6 GB
// laptop part and of a 48 GB workstation, with an int4, int8, fp8, fp16, bf16 or
// fp32 checkpoint - the layer's own precision only enters through the measured
// growth, so nothing here has to enumerate the formats.
u64 TextEncoder32B::layer_bytes_estimate(i64 i) const {
	const std::string p = "model.layers." + std::to_string(i) + ".";
	u64 total = 0;
	auto add = [&](const std::string& base) {
		const StTensor& t = st_.require(base + ".weight");
		const i64 n = t.shape[0];
		const i64 K = st_.logical_cols(t);
		u64 bytes = 0;
		// The three paths `up_pair` takes, priced the way they allocate:
		if (weight_is_dense_float(st_, t)) {
			// the file's own f16 / bf16 / f32 (fp8 and nvfp4 decode to f16, exactly)
			bytes = (u64)n * (u64)K * (weight_precision_of(st_, t) == WeightPrecision::F32 ? 4ull : 2ull);
		} else if (st_.is_plain_int8(t)) {
			bytes = (u64)n * (u64)K + (u64)n * 4;   // codes verbatim + fp32 per-row scale
		} else {
			bytes = (u64)n * (u64)K / 2 + (u64)n * 4;   // repacked onto convrot_w4a4
		}
		total += (bytes + 255) & ~(u64)255;
	};
	add(p + "self_attn.q_proj");
	add(p + "self_attn.k_proj");
	add(p + "self_attn.v_proj");
	add(p + "self_attn.o_proj");
	add(p + "mlp.gate_proj");
	add(p + "mlp.up_proj");
	add(p + "mlp.down_proj");
	// The four per-layer norms: two layer norms, q_norm and k_norm, as fp32 rows.
	total += 4ull * (u64)cfg_.hidden * 4ull;
	return total;
}

void TextEncoder32B::release_resident() {
	res_arena_.release_chunks();
	res_layers_.clear();
	res_n_ = 0;
	res_bytes_ = 0;
	res_planned_ = false;
}

void TextEncoder32B::plan_residency(i64 S) {
	(void)S;   // the frame is already charged: this runs after the activations exist
	release_resident();
	res_planned_ = true;
	const i64 n_total = cfg_.out_layer + 1;
	if (!g_ || !g_->ctx || n_total <= 0) return;
	const u64 budget = vram_budget().limit();
	// The streamed-layer reserve (and the loop's first estimate): the header's own
	// price for one layer, which the first upload replaces with the measured value.
	const u64 one_layer = layer_bytes_estimate(0);
	const u64 chunk = budget ? arena_chunk_bytes_for(budget) : (64ull << 20);
	res_arena_.init(g_->ctx, chunk);
	res_arena_.set_tag("qwen3vl.res");
	res_layers_.resize((size_t)n_total);
	if (budget == 0) {
		// No device answered: stream every layer, exactly as before this window
		// existed. (The arena keeps no chunks, so nothing is charged.)
		res_layers_.clear();
		return;
	}
	// A debug override, like the DiTs' PHI_DIT_RES: a caller comparing "no window"
	// against "the plan's window" needs to be able to pin the count.
	if (const char* e = getenv("PHI_TE_RES")) {
		const int v = atoi(e);
		const i64 want = (v <= 0) ? 0 : std::min<i64>(n_total, v);
		if (want > 0) {
			g_->ctx->begin();
			for (i64 i = 0; i < want; i++) {
				if (!g_->ctx->recording()) g_->ctx->begin();
				load_layer(st_, *g_, res_arena_, i, res_layers_[(size_t)i], loras_);
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
	// ── what the window has to fit beside ──
	const u64 held = vram_budget().local();   // activations, the last streamed layer, the caller
	const u64 slack = 96ull << 20;            // the DiTs' own margin term
	const u64 other = held + one_layer + slack;
	u64 room = budget > other ? budget - other : 0;
	// The driver's *live* headroom is the second wall (`charge()` refuses on both),
	// so the plan respects it too - a process that is already near the device's edge
	// must not be handed a window the driver will not back.
	const u64 device_room = vram_budget().live_headroom();
	if (device_room < room) room = device_room;
	if (room == 0) {
		// Nothing fits beside the frame this prompt needs; stream, as before.
		res_layers_.clear();
		return;
	}
	// ── take the window ──
	// Uploaded here rather than at first use, so the plan's invariant holds before
	// the layer loop starts (and so the reads happen once either way). The count is
	// decided by the accountant, not by arithmetic on tensor sizes: one upload's
	// growth is the charge that matters, and it includes the arena's chunk rounding.
	u64 est = one_layer;
	i64 n = 0;
	g_->ctx->begin();
	while (n < n_total) {
		if (res_arena_.capacity() + est > room) break;
		const u64 have = res_arena_.capacity();
		if (!g_->ctx->recording()) g_->ctx->begin();
		load_layer(st_, *g_, res_arena_, n, res_layers_[(size_t)n], loras_);
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
		        "[te32] window %lld/%lld layers, %s resident (one layer %s); the process is "
		        "planned to use %s of %s (%d%%), leaving %s for the streamed layer\n",
		        (long long)res_n_, (long long)n_total, format_bytes(res_bytes_).c_str(),
		        format_bytes(res_n_ ? res_bytes_ / (u64)res_n_ : one_layer).c_str(),
		        format_bytes(charged).c_str(), format_bytes(budget).c_str(),
		        budget ? (int)(100ull * charged / budget) : 0, format_bytes(one_layer).c_str());
	}
}

// ── the 50-layer int4 backbone over a caller-built [S, hidden] embedding matrix.
// Shared by encode() (text only: every row is out of `model.embed_tokens`) and
// encode_vl() (the same stream with the ViT's merged patches spliced into it).
std::vector<float> TextEncoder32B::encode_embeds(const float* xf, i64 S,
                                                 const std::vector<float>* mrope) {
	if (!open_) throw MediaError("qwen3vl: encode() before open()");
	if (S <= 0) throw MediaError("qwen3vl: empty token sequence");
	if (mrope && (i64)mrope->size() != S * cfg_.head_dim)
		throw MediaError("qwen3vl: the mrope table is " + std::to_string(mrope->size()) +
		                 " floats, a [" + std::to_string(S) + ", " + std::to_string(cfg_.head_dim) +
		                 "] one needs " + std::to_string(S * cfg_.head_dim));
	const i64 H = cfg_.hidden;
	const i64 HQ = cfg_.n_heads * cfg_.head_dim;      // 8192
	const i64 HK = cfg_.n_kv_heads * cfg_.head_dim;   // 1024
	const i64 inter = cfg_.intermediate;              // 25600

	g_->new_step();   // reset the activation arena; the weights live per layer
	TLOG("[te] step reset done; allocating activations");	auto& aa = *g_->aa;
	aa.set_tag("qwen3vl.acts");
	GpuAlloc x = aa.alloc((u64)S * H * 4);
	g_->upload_into(x, xf, (u64)S * H * 4);
	TLOG("[te] activations up (%.1f MB)", (double)((u64)S * H * 4) / 1048576.0);
	GpuAlloc t = aa.alloc((u64)S * H * 4);
	GpuAlloc q = aa.alloc((u64)S * HQ * 4);
	GpuAlloc k = aa.alloc((u64)S * HK * 4);
	GpuAlloc v = aa.alloc((u64)S * HK * 4);
	GpuAlloc o = aa.alloc((u64)S * HQ * 4);
	GpuAlloc proj = aa.alloc((u64)S * H * 4);
	GpuAlloc gate = aa.alloc((u64)S * inter * 4);
	GpuAlloc up = aa.alloc((u64)S * inter * 4);
	GpuAlloc sw = aa.alloc((u64)S * inter * 4);
	GpuAlloc down = aa.alloc((u64)S * H * 4);
	GpuAlloc q8 = aa.alloc((u64)S * inter);   // the widest K any layer quantises
	GpuAlloc s8 = aa.alloc((u64)S * 4);
	// The 3-D rotary table, on the device once for the whole backbone: every layer
	// reads the same angles, and a per-layer copy would be fifty uploads of it.
	GpuAlloc mt;
	if (mrope != nullptr) {
		mt = aa.alloc((u64)S * (u64)cfg_.head_dim * 4);
		g_->upload_into(mt, mrope->data(), (u64)S * (u64)cfg_.head_dim * 4);
	}

	// The plan is taken *after* the activation frame exists: `VramBudget::local()`
	// is then the honest "everything else", and the window is what is left of the
	// system's limit once the frame, one streamed layer and the margin are set
	// aside. Nothing here assumes a prompt length or a token count - both are
	// already charged to the ledger by the time it runs.
	plan_residency(S);

	const float scale = 1.0f / std::sqrt((float)cfg_.head_dim);

	// One projection, through whichever path its own precision implies: the shipped
	// convrot_w4a4 pair runs the int4 GEMM; a float source runs the dense fp16 /
	// bf16 / f32 GEMM with no conversion.
	auto gemm = [&](const GpuAlloc& a, i64 m, i64 n, i64 kk, const TeLinear32& L,
	               const GpuAlloc& out) {
		if (L.dense()) {
			GemmF16Args ga;
			ga.a = a;
			ga.b = L.d;
			ga.c = out;
			ga.m = m;
			ga.n = n;
			ga.k = kk;
			ga.a_is_f32 = true;
			ga.b_is_bf16 = (L.dtype == DType::BF16);
			ga.b_is_f32 = (L.dtype == DType::F32);
			dispatch_gemm_f16(*g_->ctx, ga);
			return;
		}
			QuantConvrotArgs qa;
			qa.x = a.res;
			qa.x_offset = a.off;
			qa.q = q8.res;
			qa.q_offset = q8.off;
			qa.s = s8.res;
			qa.s_offset = s8.off;
			qa.rows = m;
			qa.K = kk;
			qa.qmax = L.i8 ? 127.0f : kInt4Qmax;
			dispatch_quant_convrot(*g_->ctx, qa);
			if (L.i8) {
				Int8GemmArgs ga;
				ga.a = q8.res;
				ga.a_offset = q8.off;
				ga.b = L.w.res;
				ga.b_offset = L.w.off;
				ga.sa = s8.res;
				ga.sa_offset = s8.off;
				ga.sb = L.s.res;
				ga.sb_offset = L.s.off;
				ga.c = out.res;
				ga.c_offset = out.off;
				ga.M = m;
				ga.N = n;
				ga.K = kk;
				dispatch_int8_gemm(*g_->ctx, ga);
				return;
			}
			Int4GemmArgs ga;
		ga.a = q8;
		ga.b = L.w;
		ga.sa = s8;
		ga.sb = L.s;
		ga.c = out;
		ga.m = m;
		ga.n = n;
		ga.k = kk;
		dispatch_int4_gemm(*g_->ctx, ga);
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

	for (i64 li = 0; li <= cfg_.out_layer; li++) {
		g_->new_layer();
		TeLayerW streamed;
		const bool resident = li < res_n_;
		if (!resident) {
			TLOG("[te] layer %lld: loading weights", (long long)li);
			load_layer(st_, *g_, *g_->wa, li, streamed, loras_);
		}
		g_->ctx->submit();
		g_->ctx->begin();
		// A resident layer is used *in place*: `plan_residency` uploaded it into its
		// own arena, and the streaming arena was reset by `new_layer()` just above -
		// which is exactly why the window does not live in the streaming arena.
		const TeLayerW& L = resident ? res_layers_[(size_t)li] : streamed;
		TLOG("[te] layer %lld: weights up%s", (long long)li, resident ? " (resident)" : "");

		// ── attention ──
		rms(x, S, H, L.in_ln, t);
		gemm(t, S, HQ, H, L.q, q);
		gemm(t, S, HK, H, L.k, k);
		gemm(t, S, HK, H, L.v, v);

		// per-head RMSNorm (RFC-like: rows are (token, head) pairs)
		rms(q, S * cfg_.n_heads, cfg_.head_dim, L.q_norm, q);
		rms(k, S * cfg_.n_kv_heads, cfg_.head_dim, L.k_norm, k);
		// q/k_norm are in place: the row kernel reads the whole row before it
		// writes any of it, and each row belongs to exactly one thread block.
		if (mrope == nullptr) {
			RopeHalfArgs ra;
			ra.head_dim = cfg_.head_dim;
			ra.theta = cfg_.rope_theta;
			ra.rows = S * cfg_.n_heads;
			ra.heads = cfg_.n_heads;
			ra.x = q;
			ra.y = q;
			dispatch_rope_half(*g_->ctx, q.res, ra);
			ra.rows = S * cfg_.n_kv_heads;
			ra.heads = cfg_.n_kv_heads;
			ra.x = k;
			ra.y = k;
			dispatch_rope_half(*g_->ctx, k.res, ra);
		} else {
			// The 3-D rotary: the same split-half pairing, but the angle of pair
			// `i` comes from the *table* rather than from the row index, because
			// a vision span's tokens carry a (row, column) pair and the two are
			// interleaved across the frequency bands. `vit_rope` is exactly that
			// kernel (it was written for the tower's own rotary), so it is reused
			// here rather than duplicated - the two callers differ only in the
			// stride they pass. `mt` is claimed once, above the layer loop.
			vit_rope_run(*g_->ctx, q, mt, S, cfg_.n_heads, cfg_.head_dim,
			             cfg_.n_heads * cfg_.head_dim, 0);
			vit_rope_run(*g_->ctx, k, mt, S, cfg_.n_kv_heads, cfg_.head_dim,
			             cfg_.n_kv_heads * cfg_.head_dim, 0);
		}
		{
			AttnFlashArgs aa2;
			aa2.q = q;
			aa2.k = k;
			aa2.v = v;
			aa2.o = o;
			aa2.s = S;
			aa2.heads = cfg_.n_heads;
			aa2.kv_heads = cfg_.n_kv_heads;
			aa2.head_dim = cfg_.head_dim;
			aa2.f32_input = true;
			aa2.causal = true;
			aa2.scale = scale;
			dispatch_attn_flash(*g_->ctx, aa2);
		}
		gemm(o, S, H, HQ, L.o, proj);
		{
			ElemArgs ea;
			ea.op = ElemOp::Add;
			ea.a = x;
			ea.b = proj;
			ea.y = x;
			ea.rows = S;
			ea.cols = H;
			dispatch_elem(*g_->ctx, ea);
		}

		// ── MLP ──
		rms(x, S, H, L.post_ln, t);
		gemm(t, S, inter, H, L.gate, gate);
		gemm(t, S, inter, H, L.up, up);
		{
			ElemArgs ea;
			ea.op = ElemOp::SiluGate;
			ea.a = gate;
			ea.b = up;
			ea.y = sw;
			ea.rows = S;
			ea.cols = inter;
			dispatch_elem(*g_->ctx, ea);
		}
		gemm(sw, S, H, inter, L.down, down);
		{
			ElemArgs ea;
			ea.op = ElemOp::Add;
			ea.a = x;
			ea.b = down;
			ea.y = x;
			ea.rows = S;
			ea.cols = H;
			dispatch_elem(*g_->ctx, ea);
		}
		g_->end_layer();
		TLOG("[te] layer %lld: done", (long long)li);
	}

	// No final norm: the checkpoint is truncated at layer 50 and the DiT
	// conditions on the raw hidden state (Qwen3VL_32BConfig.final_norm = False).
	std::vector<float> out = g_->download_f32(x, (u64)S * H);
	g_->ctx->submit_if_recording();
	// The window goes back at the end of the call: the DiT's own plan runs next and
	// reads `VramBudget::local()`, so a window still held here would be subtracted
	// from its budget and buy resident DiT blocks for the tower instead.
	release_resident();
	return out;
}

// ── vision tower ───────────────────────────────────────────────────────────
//
// Qwen3-VL "Qwen35VisionModel" (comfy/text_encoders/qwen35.py): a pre-LN ViT over
// 16x16 patches with a temporal patch of 2, 27 blocks, a 2x2 spatial merge.
//
// Layout. vision_prep emits one 32x32 patch block per token, channel-planar
// ([C][32][32]). The reference orders its patch rows merged-block-major - four
// consecutive rows carry the 2x2 sub-patches of one 32x32 block in (mh, mw)
// order, each [C, T, p, p] - which is exactly the order the merger's 4-row view
// needs, so the patch matrix is assembled on the host in that order and nothing
// downstream has to be shuffled.
//
// Sequences. The frozen prep emits t_frames * grid_h * grid_w placeholders for a
// video block, i.e. one per 32x32 block *per frame*, and the splice must fill
// every one of them. The tower is therefore run with one attention sequence per
// frame: a still's single frame is duplicated into both temporal slots (which is
// what the reference does for an image), and every video frame is tokenised on
// its own, so the merged-token count equals the run vision_prep emitted.
//
// All sequences share one activation matrix, so the tower's 920 MB of weights are
// streamed once per call - one block (or the merger) at a time through the
// weights arena, exactly like the text backbone below streams its layers.
std::vector<float> TextEncoder32B::run_vision(const std::vector<VisionBlock>& blocks,
                                          std::vector<i64>* rows_per_block) {
	rows_per_block->clear();
	const i64 H = cfg_.vit_hidden;               // 1152
	const i64 heads = cfg_.vit_heads;            // 16
	const i64 inter = cfg_.vit_intermediate;     // 4304
	const i64 P = cfg_.patch;                    // 16
	const i64 TP = cfg_.vit_patch_t;             // 2 (temporal patch)
	const i64 mrg = cfg_.merge;                  // 2
	const i64 outH = cfg_.hidden;                // 5120 = merger fc2 width
	const i64 block_px = mrg * P;                // 32 = vision_prep's patch block
	const i64 kPatch = 3 * TP * P * P;           // 1536 = the conv's K
	const i64 kBlockFloats = 3 * block_px * block_px;   // 3072 = prep's row stride
	if (H <= 0 || heads <= 0 || H % heads != 0)
		throw MediaError("qwen3vl(visual): the ViT hidden size is not heads * head_dim");
	const i64 D = H / heads;                     // 72
	const i64 nband = D / 4;                     // 18 rotary bands per axis
	if (D % 4 != 0 || D / 2 != 2 * nband)
		throw MediaError("qwen3vl(visual): the ViT head_dim is not a multiple of 4");
	if (block_px != 32)
		throw MediaError("qwen3vl(visual): vision_prep blocks are 32x32, the config disagrees");

	// ── enumerate the attention sequences ─────────────────────────────────
	//
	// One attention sequence per *temporal patch*. A still contributes one
	// sequence (its single frame duplicated into both temporal slots, which is
	// what the reference does for an image); a temporally packed video block
	// contributes one sequence holding its frame pair; a legacy block whose
	// frames are not packed contributes one sequence per frame, which is the
	// presentation this path used before the H3 ref2video node moved to the
	// released 2 fps / pair-per-patch layout.
	struct Seq {
		const float* src;        // this sequence's first frame's [gh*gw, kBlockFloats] blocks
		i64 gh = 0, gw = 0;      // 32px block grid of one frame
		i64 frames = 1;          // frames filling the temporal patch (1 = duplicated)
		i64 tok0 = 0, ntok = 0;
	};
	std::vector<Seq> seqs;
	i64 S = 0;
	for (const VisionBlock& b : blocks) {
		if (b.n_patches <= 0) {
			rows_per_block->push_back(0);
			continue;
		}
		const i64 gh = b.grid_h, gw = b.grid_w;
		// t_frames == 0 is the prep's "no frame could be decoded" marker; that
		// block still owns one placeholder, so it still gets one merged token.
		const i64 frames = b.t_frames > 1 ? b.t_frames : 1;
		if (gh <= 0 || gw <= 0)
			throw MediaError("qwen3vl(visual): a vision block has an empty grid");
		if (gh * gw * frames != b.n_patches)
			throw MediaError("qwen3vl(visual): a vision block's patch count does not match its grid");
		if ((i64)b.patches.size() != b.n_patches * kBlockFloats)
			throw MediaError("qwen3vl(visual): a vision block's patch buffer has the wrong size");
		// `tokens()` is the merged-token count the prompt reserved for this block:
		// `n_patches` for a per-frame block, `n_patches / t_frames` when the
		// frames share one temporal patch.
		rows_per_block->push_back(b.tokens());
		const i64 per_seq = b.temporal_pack && frames > 1 ? frames : 1;
		for (i64 f = 0; f + per_seq <= frames; f += per_seq) {
			Seq sq;
			sq.src = b.patches.data() + (size_t)(f * gh * gw) * (size_t)kBlockFloats;
			sq.gh = gh;
			sq.gw = gw;
			sq.frames = per_seq;
			sq.tok0 = S;
			sq.ntok = 4 * gh * gw;   // 2x2 merge -> 16x16 patches after the conv
			seqs.push_back(sq);
			S += sq.ntok;
		}
	}
	if (S <= 0) return {};
	if (S % 4 != 0) throw MediaError("qwen3vl(visual): the token count is not a multiple of the merge");

	// ── host-side per-token tables ─────────────────────────────────────────
	//   patchmat [S, 3*TP*P*P]  the patch rows, temporal slot duplicated
	//   posmat   [S, H]         the interpolated pos_embed, same row order
	//   cs       [S, D]         per-token cos|sin of the vision rotary
	std::vector<float> patchmat((size_t)S * (size_t)kPatch);
	std::vector<float> posmat((size_t)S * (size_t)H);
	std::vector<float> cs((size_t)S * (size_t)D);
	const std::vector<float> pos_tab =
	    tensor_to_f32(st_, st_.require("visual.pos_embed.weight"));
	const i64 pos_rows = (i64)pos_tab.size() / H;
	const i64 side = (i64)std::llround(std::sqrt((double)pos_rows));
	if (side * side != pos_rows)
		throw MediaError("qwen3vl(visual): pos_embed is not a square grid");
	// inv_freq of the vision rotary: theta^( -2j / (D/2) ), j < D/4 (the reference
	// builds `dim = hidden // heads // 2` frequencies and splits them over the two
	// axes).
	std::vector<float> inv((size_t)nband);
	for (i64 j = 0; j < nband; j++)
		inv[(size_t)j] = std::pow(10000.0f, (float)(-2.0 * (double)j / (double)(D / 2)));

	for (const Seq& sq : seqs) {
		const i64 gh = sq.gh, gw = sq.gw;
		const i64 Hp = gh * 2, Wp = gw * 2;   // 16px patch grid of one frame
		// pos_embed: interpolate the side x side table to (Hp, Wp) exactly like
		// fast_pos_embed_interpolate (four corners, bilinear weights), then index it
		// in merged order. linspace(0, side-1, n).int() is a floor for n >= 1.
		std::vector<i64> hf((size_t)Hp), hc((size_t)Hp), wf((size_t)Wp), wc((size_t)Wp);
		std::vector<float> dh((size_t)Hp), dw((size_t)Wp);
		for (i64 py = 0; py < Hp; py++) {
			const double u = Hp > 1 ? (double)py * (double)(side - 1) / (double)(Hp - 1) : 0.0;
			hf[(size_t)py] = (i64)u;
			hc[(size_t)py] = std::min(hf[(size_t)py] + 1, side - 1);
			dh[(size_t)py] = (float)(u - (double)hf[(size_t)py]);
		}
		for (i64 px = 0; px < Wp; px++) {
			const double u = Wp > 1 ? (double)px * (double)(side - 1) / (double)(Wp - 1) : 0.0;
			wf[(size_t)px] = (i64)u;
			wc[(size_t)px] = std::min(wf[(size_t)px] + 1, side - 1);
			dw[(size_t)px] = (float)(u - (double)wf[(size_t)px]);
		}
		for (i64 r = 0; r < sq.ntok; r++) {
			const i64 b = r / 4, mh = (r % 4) / 2, mw = r % 2;
			const i64 by = b / gw, bx = b % gw;
			const i64 py = by * 2 + mh, px = bx * 2 + mw;
			// patch row: sub-patch (mh, mw) of 32x32 block b, [C, TP, P, P]. The
			// temporal slots take the sequence's frame 0 and frame 1 when the
			// block packs a pair (the H3 reference-video presentation), and the
			// same frame twice for a still (the reference's image behaviour).
			float* dst = patchmat.data() + (size_t)(sq.tok0 + r) * (size_t)kPatch;
			for (i64 c = 0; c < 3; c++)
				for (i64 tt = 0; tt < TP; tt++) {
					const i64 fi = sq.frames > 1 ? std::min<i64>(tt, sq.frames - 1) : 0;
					const float* src = sq.src +
					                   (size_t)(fi * sq.gh * sq.gw + b) * (size_t)kBlockFloats;
					for (i64 iy = 0; iy < P; iy++) {
						const float* srow = src + (size_t)(c * block_px * block_px +
						                                   (mh * P + iy) * block_px + mw * P);
						float* drow = dst + ((c * TP + tt) * P + iy) * P;
						for (i64 ix = 0; ix < P; ix++) drow[ix] = srow[ix];
					}
				}
			// interpolated pos_embed row
			{
				const float w00 = (1.0f - dh[(size_t)py]) * (1.0f - dw[(size_t)px]);
				const float w01 = (1.0f - dh[(size_t)py]) * dw[(size_t)px];
				const float w10 = dh[(size_t)py] * (1.0f - dw[(size_t)px]);
				const float w11 = dh[(size_t)py] * dw[(size_t)px];
				const float* t00 = pos_tab.data() + (size_t)(hf[(size_t)py] * side + wf[(size_t)px]) * H;
				const float* t01 = pos_tab.data() + (size_t)(hf[(size_t)py] * side + wc[(size_t)px]) * H;
				const float* t10 = pos_tab.data() + (size_t)(hc[(size_t)py] * side + wf[(size_t)px]) * H;
				const float* t11 = pos_tab.data() + (size_t)(hc[(size_t)py] * side + wc[(size_t)px]) * H;
				float* prow = posmat.data() + (size_t)(sq.tok0 + r) * (size_t)H;
				for (i64 d = 0; d < H; d++)
					prow[d] = w00 * t00[d] + w01 * t01[d] + w10 * t10[d] + w11 * t11[d];
			}
			// rotary: the first half of the pairs carries the patch row, the second
			// half the patch column (rot_pos_emb's stacked (row, col) coords).
			{
				float* crow = cs.data() + (size_t)(sq.tok0 + r) * (size_t)D;
				for (i64 j = 0; j < nband; j++) {
					const float ah = (float)py * inv[(size_t)j];
					const float aw = (float)px * inv[(size_t)j];
					crow[j] = std::cos(ah);
					crow[nband + j] = std::cos(aw);
					crow[D / 2 + j] = std::sin(ah);
					crow[D / 2 + nband + j] = std::sin(aw);
				}
			}
		}
	}

	if (const char* dp = std::getenv("PHI_VIT_DUMP")) {
		if (*dp) {
			const std::string pre(dp);
			if (FILE* f = std::fopen((pre + "_in.f32").c_str(), "wb")) {
				std::fwrite(patchmat.data(), 4, patchmat.size(), f);
				std::fclose(f);
			}
			if (FILE* f = std::fopen((pre + "_pos.f32").c_str(), "wb")) {
				std::fwrite(posmat.data(), 4, posmat.size(), f);
				std::fclose(f);
			}
			TLOG("[vit] dumped %lld tokens to %s_in.f32", (long long)S, pre.c_str());
		}
	}

	// ── device ─────────────────────────────────────────────────────────────
	GpuCtx& g = *g_;
	auto& aa = *g.aa;
	g.new_step();   // the caller's previous stage is dead; ours starts clean
	// Stage dumps for the numeric oracle (the reference writes the same
	// tensors). Off unless PHI_VIT_DUMP names a prefix.
	const char* dump_pre = std::getenv("PHI_VIT_DUMP");
	if (dump_pre && !*dump_pre) dump_pre = nullptr;
	auto dump_dev = [&](const char* what, const GpuAlloc& a, i64 count) {
		if (!dump_pre) return;
		// The DtoH helper is a synchronous copy, but this runs inside an open
		// dispatch bracket: drain the stream first so the dump is the completed
		// tensor and not whatever the copy engine raced the producer for.
		g.ctx->submit_if_recording();
		std::vector<float> v = g.download_f32(a, (u64)count);
		g.ctx->begin();
		const std::string fn = std::string(dump_pre) + "_" + what + ".f32";
		if (FILE* f = std::fopen(fn.c_str(), "wb")) {
			std::fwrite(v.data(), 4, v.size(), f);
			std::fclose(f);
		}
	};
	TLOG("[vit] %lld tokens over %zu sequences (%lld blocks)", (long long)S, seqs.size(),
	     (long long)blocks.size());
	GpuAlloc d_patch = aa.alloc((u64)S * (u64)kPatch * 4);
	g.upload_into(d_patch, patchmat.data(), (u64)patchmat.size() * 4);
	GpuAlloc d_cs = aa.alloc((u64)S * (u64)D * 4);
	g.upload_into(d_cs, cs.data(), (u64)cs.size() * 4);
	GpuAlloc d_pos = aa.alloc((u64)S * (u64)H * 4);
	g.upload_into(d_pos, posmat.data(), (u64)posmat.size() * 4);
	GpuAlloc x = aa.alloc((u64)S * (u64)H * 4);
	GpuAlloc t = aa.alloc((u64)S * (u64)H * 4);
	GpuAlloc qkv = aa.alloc((u64)S * (u64)(3 * H) * 4);
	GpuAlloc attn_o = aa.alloc((u64)S * (u64)H * 4);
	GpuAlloc down = aa.alloc((u64)S * (u64)H * 4);
	GpuAlloc gate = aa.alloc((u64)S * (u64)inter * 4);
	GpuAlloc merged = aa.alloc((u64)(S / 4) * (u64)outH * 4);

	// ── sync-free stage tracer (PHI_VIT_TRACE=<prefix>) ─────────────────────
	// The PHI_VIT_DUMP path has to submit+sync to read a stage back, and where
	// those syncs fall changes this tower's outcome (it is what hid the
	// LayerNorm race for so long), so a dump run
	// is not the run being diagnosed. This tracer instead snapshots each stage
	// with an *async*
	// device->host copy into one pinned buffer, ordered on the compute stream and
	// never synchronising; the whole trace is written once, at the end. Comparing
	// two processes' traces names the first stage that differs, with the sync
	// pattern of both runs identical (one sync, after everything).
	struct TrStage {
		std::string name;
		u64 off = 0, bytes = 0;
	};
	const char* trace_pre = std::getenv("PHI_VIT_TRACE");
	if (trace_pre && !*trace_pre) trace_pre = nullptr;
	std::vector<TrStage> tr;
	u8* tr_host = nullptr;
	u64 tr_used = 0;
	if (trace_pre) {
		auto want = [&](const std::string& n, u64 bytes) {
			tr.push_back(TrStage{n, tr_used, bytes});
			tr_used += bytes;
		};
		want("emb", (u64)S * H * 4);
		want("posadd", (u64)S * H * 4);
		// Four row-local stages per block: the LayerNorm output, the attention
		// output, the residual after the attention projection, and the MLP output.
		// A block whose output first differs is localised by which of the four
		// moved first.
		for (i64 li = 0; li < cfg_.vit_layers; li++) {
			const std::string k = std::to_string(li);
			want("t" + k, (u64)S * H * 4);
			want("attn" + k, (u64)S * H * 4);
			want("xmid" + k, (u64)S * H * 4);
			want("mp" + k, (u64)S * H * 4);
		}
		want("mnorm", (u64)S * H * 4);
		want("m1", (u64)S * H * 4);
		want("mgelu", (u64)S * H * 4);
		want("merged", (u64)(S / 4) * outH * 4);
		void* mapped = nullptr;
		g.ctx->device_context().alloc_host(tr_used, &mapped);
		tr_host = (u8*)mapped;
		TLOG("[vit] tracing %llu bytes to %s_trace.bin", (unsigned long long)tr_used, trace_pre);
	}
	auto tr_snap = [&](const std::string& name, const GpuAlloc& a) {
		if (!tr_host) return;
		for (const TrStage& s : tr)
			if (s.name == name) {
				cuda_api().MemcpyDtoHAsync(tr_host + s.off, a.res->base + a.off, (size_t)s.bytes,
				                           g.ctx->device_context().stream());
				return;
			}
	};

	// patch embed: stride == kernel, so the 3-D conv is one GEMM against the
	// weight flattened [1152, 3*TP*P*P] (the row layout is exactly [C, T, p, p]).
	g.new_layer();
	{
		GpuAlloc pw = up_bf16(st_, g, *g.wa, "visual.patch_embed.proj.weight");
		GpuAlloc pb = up_f32(st_, g, *g.wa, "visual.patch_embed.proj.bias");
		vit_gemm(*g.ctx, d_patch, pw, pb, x, S, H, kPatch);
		dump_dev("emb", x, S * H);
		tr_snap("emb", x);
		vit_add(g, x, d_pos, S, H);
		dump_dev("posadd", x, S * H);
		tr_snap("posadd", x);
	}
	g.end_layer();

	// Fixed: this loop used to move the vision rows between runs. The cause was
	// `norm_rows`'s LayerNorm path in core/kernels/ops.cpp: it reads the
	// reduction result out of shared memory (`total = red[0]`, in `reduce()`) and
	// then reuses the same array for the variance partials (`red[t] = part2`) with
	// no barrier in between, so a warp whose load lands after thread 0's store
	// normalises its columns with a variance partial for a mean. A layer's two
	// LayerNorms are where the tower forked, which is why the text backbone (RMS
	// only) never moved. `vit_layer_norm` now runs the ViT-local `vit_ln`, whose
	// WAR fence makes that read safe; the note above has the
	// evidence and tests/vit_tower_probe.cpp --repeat for the regression check.
	const i64 stride = 3 * H;   // elements between tokens in the fused qkv buffer
	// Debug: hold every block's weights resident instead of streaming one block
	// per phase, to separate "the tower" from "the weight streamer".
	for (i64 li = 0; li < cfg_.vit_layers; li++) {
		g.new_layer();
		const std::string p = "visual.blocks." + std::to_string(li) + ".";
		GpuAlloc n1w = up_f32(st_, g, *g.wa, p + "norm1.weight");
		GpuAlloc n1b = up_f32(st_, g, *g.wa, p + "norm1.bias");
		GpuAlloc n2w = up_f32(st_, g, *g.wa, p + "norm2.weight");
		GpuAlloc n2b = up_f32(st_, g, *g.wa, p + "norm2.bias");
		GpuAlloc qw = up_bf16(st_, g, *g.wa, p + "attn.qkv.weight");
		GpuAlloc qb = up_f32(st_, g, *g.wa, p + "attn.qkv.bias");
		GpuAlloc apw = up_bf16(st_, g, *g.wa, p + "attn.proj.weight");
		GpuAlloc apb = up_f32(st_, g, *g.wa, p + "attn.proj.bias");
		GpuAlloc f1w = up_bf16(st_, g, *g.wa, p + "mlp.linear_fc1.weight");
		GpuAlloc f1b = up_f32(st_, g, *g.wa, p + "mlp.linear_fc1.bias");
		GpuAlloc f2w = up_bf16(st_, g, *g.wa, p + "mlp.linear_fc2.weight");
		GpuAlloc f2b = up_f32(st_, g, *g.wa, p + "mlp.linear_fc2.bias");

		// x = x + proj(attn(norm1(x)))
		vit_layer_norm(g, x, n1w, n1b, S, H, 1e-6f, t);
		if (li == 0) dump_dev("t0", t, S * H);
		tr_snap("t" + std::to_string(li), t);
		vit_gemm(*g.ctx, t, qw, qb, qkv, S, 3 * H, H);
		if (li == 0) dump_dev("qkvpre0", qkv, S * 3 * H);
		vit_rope_run(*g.ctx, qkv, d_cs, S, heads, D, stride, 0);
		vit_rope_run(*g.ctx, qkv, d_cs, S, heads, D, stride, 1);

		for (const Seq& sq : seqs)
			vit_attn_run(*g.ctx, qkv, attn_o, sq.tok0, sq.ntok, heads, D, stride);
		if (li == 0) dump_dev("attnraw", attn_o, S * H);
		tr_snap("attn" + std::to_string(li), attn_o);
		if (li == 0) {
			dump_dev("qkv0", qkv, S * 3 * H);          // all three sections, post-rope
			dump_dev("attn0", attn_o, S * H);
		}
		vit_gemm(*g.ctx, attn_o, apw, apb, down, S, H, H);

		vit_add(g, x, down, S, H);
		tr_snap("xmid" + std::to_string(li), x);

		// x = x + fc2(gelu(fc1(norm2(x))))
		vit_layer_norm(g, x, n2w, n2b, S, H, 1e-6f, t);

		vit_gemm(*g.ctx, t, f1w, f1b, gate, S, inter, H);

		vit_gelu_run(*g.ctx, gate, S * inter);
		if (li == 0) dump_dev("gelu0", gate, S * inter);

		vit_gemm(*g.ctx, gate, f2w, f2b, down, S, H, inter);
		tr_snap("mp" + std::to_string(li), down);
		vit_add(g, x, down, S, H);

		if (li < 3) dump_dev(("block" + std::to_string(li)).c_str(), x, S * H);
		// `end_layer` submits and rewinds the upload ring (the weights stay put
		// either way: only `new_layer` resets the weights arena).
		g.end_layer();
		TLOG("[vit] block %lld done", (long long)li);
	}

	// merger: LayerNorm over the pre-merge width, then the 4-row view through
	// fc1 -> gelu -> fc2 into the backbone's hidden width.
	g.new_layer();
	{
		GpuAlloc mnw = up_f32(st_, g, *g.wa, "visual.merger.norm.weight");
		GpuAlloc mnb = up_f32(st_, g, *g.wa, "visual.merger.norm.bias");
		GpuAlloc m1w = up_bf16(st_, g, *g.wa, "visual.merger.linear_fc1.weight");
		GpuAlloc m1b = up_f32(st_, g, *g.wa, "visual.merger.linear_fc1.bias");
		GpuAlloc m2w = up_bf16(st_, g, *g.wa, "visual.merger.linear_fc2.weight");
		GpuAlloc m2b = up_f32(st_, g, *g.wa, "visual.merger.linear_fc2.bias");
		vit_layer_norm(g, x, mnw, mnb, S, H, 1e-6f, t);
		dump_dev("mnorm", t, S * H);
		tr_snap("mnorm", t);
		vit_gemm(*g.ctx, t, m1w, m1b, down, S / 4, 4 * H, 4 * H);
		tr_snap("m1", down);
		vit_gelu_run(*g.ctx, down, (S / 4) * 4 * H);
		tr_snap("mgelu", down);
		vit_gemm(*g.ctx, down, m2w, m2b, merged, S / 4, outH, 4 * H);
		tr_snap("merged", merged);
	}
	g.end_layer();

	std::vector<float> out = g.download_f32(merged, (u64)(S / 4) * (u64)outH);
	g.ctx->submit_if_recording();
	if (const char* dp = std::getenv("PHI_VIT_DUMP")) {
		if (*dp) {
			const std::string fn = std::string(dp) + "_merged.f32";
			if (FILE* f = std::fopen(fn.c_str(), "wb")) {
				std::fwrite(out.data(), 4, out.size(), f);
				std::fclose(f);
			}
		}
	}
	TLOG("[vit] done: %lld merged rows of %lld", (long long)(S / 4), (long long)outH);
	if (trace_pre && tr_host) {
		// Every snapshot above is already complete (the `merged` readback at the
		// top of this function is a synchronous copy), so this is a host write and
		// nothing else.
		const std::string fn = std::string(trace_pre) + "_trace.bin";
		if (FILE* f = std::fopen(fn.c_str(), "wb")) {
			const u32 nst = (u32)tr.size();
			fwrite(&nst, 4, 1, f);
			for (const TrStage& s : tr) {
				char nm[32] = {0};
				snprintf(nm, sizeof nm, "%s", s.name.c_str());
				fwrite(nm, 1, sizeof nm, f);
				fwrite(&s.bytes, 8, 1, f);
			}
			fwrite(tr_host, 1, (size_t)tr_used, f);
			fclose(f);
		}
	}
	return out;
}

// ── Qwen3-VL's 3-D rotary positions ───────────────────────────────────────
//
// A text token has one position; a vision token has three - (time, row, column)
// - and Qwen3-VL folds them into a single rotary table by giving different
// frequency bands different axes. The reference builds it in two steps
// (`qwen2vl_mrope_position_ids`, then `precompute_freqs_cis` with
// `interleaved_mrope`); this is both, in one pass, because the table is what the
// kernels consume.
//
// The position sequences, transcribed from the reference:
//
//   * everything before the first vision span is 0, 1, 2, ...
//   * a span of `n` merged tokens starting at token `start`:
//       time = start + offset                       (constant across the span)
//       row  = start + offset + j / ceil(n / gh)
//       col  = start + offset + j % gw
//     with gh/gw the span's *merged* grid - one token per 32-pixel block. (The
//     reference names the ViT's 16-pixel patch grid in `grid_thw` and halves it;
//     this stream is already merged, so it does not.)
//   * the text after a span continues from `max(gh, gw) + start + offset`
//   * `offset` accumulates `max(gh, gw) - n` per span, which is what keeps
//     consecutive spans' column axes from overlapping.
//
// The bands: with `rope_dims = {24, 20, 20}` and the interleaved form, band m
// reads the row axis when m % 3 == 1, the column axis when m % 3 == 2, and the
// time axis otherwise - each axis' slice stopping at `3 * dim`, so bands 60..63
// are time as well. 24 + 20 + 20 = 64 is head_dim / 2, hence
// `inv_freq[m] = theta^(-2m/head_dim)`, the same frequency ladder the plain
// rotary uses.
//
// The result is the [L, head_dim] table `vit_rope` consumes: head_dim/2 cos
// values then head_dim/2 sin values, per token.
std::vector<float> build_mrope_table(const PromptTokens& p, const TextEncoder32BConfig& cfg) {
	const i64 L = (i64)p.ids.size();
	const i64 D = cfg.head_dim;
	const i64 R = D / 2;
	if (L <= 0 || D <= 0 || (D % 2) != 0)
		throw MediaError("qwen3vl(visual): head_dim " + std::to_string(D) +
		                 " cannot carry a 3-D rotary");
	constexpr i64 kDims[3] = {24, 20, 20};
	if (kDims[0] + kDims[1] + kDims[2] != R)
		throw MediaError("qwen3vl(visual): rope_dims {24, 20, 20} do not sum to head_dim / 2 = " +
		                 std::to_string(R));

	std::vector<int> axis((size_t)R, 0);
	for (int a = 1; a < 3; a++)
		for (i64 m = a; m < kDims[a] * 3; m += 3) axis[(size_t)m] = a;

	std::vector<i64> pos((size_t)3 * (size_t)L, 0);
	i64 offset = 0;
	bool first = true;
	for (size_t k = 0; k < p.blocks.size(); k++) {
		const i64 at = k < p.block_at.size() ? p.block_at[k] : -1;
		if (at < 0) continue;
		const VisionBlock& b = p.blocks[k];
		// The *merged* token count: one per 32x32 block, and one per block for a
		// temporally packed pair as well (`n_patches / t_frames`).
		const i64 n = b.tokens();
		if (n <= 0) continue;
		if (at + n > L)
			throw MediaError("qwen3vl(visual): a vision span runs past the end of the prompt");
		if (first) {
			for (i64 s = 0; s < at; s++)
				pos[(size_t)s] = pos[(size_t)L + s] = pos[(size_t)2 * L + s] = s;
			first = false;
		}
		const i64 gh = b.grid_h, gw = b.grid_w;
		const i64 len_max = std::max<i64>(gh, gw);
		const i64 end = at + n;
		for (i64 i = end; i < L; i++)
			for (int a = 0; a < 3; a++)
				pos[(size_t)a * L + i] = len_max + at + offset + (i - end);
		const i64 rows_per_group = (n + gh - 1) / gh;
		for (i64 j = 0; j < n; j++) {
			pos[(size_t)at + j] = at + offset;
			pos[(size_t)L + at + j] = at + offset + j / rows_per_group;
			pos[(size_t)2 * L + at + j] = at + offset + j % gw;
		}
		offset += len_max - n;
	}

	std::vector<double> inv((size_t)R);
	for (i64 m = 0; m < R; m++)
		inv[(size_t)m] = std::pow((double)cfg.rope_theta, -2.0 * (double)m / (double)D);
	std::vector<float> table((size_t)L * (size_t)D);
	for (i64 s = 0; s < L; s++) {
		float* row = &table[(size_t)s * (size_t)D];
		for (i64 m = 0; m < R; m++) {
			const double ang = (double)pos[(size_t)axis[(size_t)m] * L + s] * inv[(size_t)m];
			row[m] = (float)std::cos(ang);
			row[R + m] = (float)std::sin(ang);
		}
	}
	return table;
}

// The vision half of encode_vl: text rows out of the embedding table, the ViT's
// merged rows at the placeholder positions, [L, hidden] with L == ids.size().
std::vector<float> TextEncoder32B::encode_vl_embeds(const PromptTokens& p) {
	if (!open_) throw MediaError("qwen3vl: encode_vl() before open()");
	const i64 L = (i64)p.ids.size();
	const i64 H = cfg_.hidden;
	if (L <= 0) throw MediaError("qwen3vl: empty token sequence");
	if ((i64)p.tags.size() != L)
		throw MediaError("qwen3vl: PromptTokens has mismatched ids/tags sizes");

	// The 3-D positions the backbone's rotary needs, built here because this is
	// where the vision spans are known (see `build_mrope_table`).
	std::vector<float> mrope;
	if (!p.blocks.empty()) mrope = build_mrope_table(p, cfg_);

	// ── text rows (host side, straight out of the mapping) ────────────────
	std::vector<float> xf((size_t)L * (size_t)H);
	{
		const StTensor& t = st_.require("model.embed_tokens.weight");
		const size_t esz = (t.dtype == DType::BF16 || t.dtype == DType::F16) ? 2 : 4;
		std::vector<u8> row((size_t)H * esz);
		for (i64 s = 0; s < L; s++) {
			const i32 id = p.ids[(size_t)s];
			if (id < 0 || id >= cfg_.vocab) throw MediaError("qwen3vl: token id out of range");
			memcpy(row.data(), (const u8*)st_.data_of(t) + (u64)id * (u64)H * esz, row.size());
			for (i64 d = 0; d < H; d++) {
				if (wdtype_ == DType::BF16) {
					const u16 h = *(const u16*)&row[(size_t)d * 2];
					xf[(size_t)(s * H + d)] = bf16_to_f32(h);
				} else if (wdtype_ == DType::F16) {
					const u16 h = *(const u16*)&row[(size_t)d * 2];
					xf[(size_t)(s * H + d)] = f16_to_f32(h);
				} else {
					memcpy(&xf[(size_t)(s * H + d)], &row[(size_t)d * 4], 4);
				}
			}
		}
	}

	// ── splice the vision rows in ─────────────────────────────────────────
	// A block with placeholders but no `block_at` entry has nowhere to go, so
	// this throws rather than silently leaving `model.embed_tokens` rows (the
	// image-pad id's own embedding) in an image slot.
	if (!p.blocks.empty()) {
		std::vector<i64> rows;
		std::vector<float> vis = run_vision(p.blocks, &rows);
		i64 off = 0;
		std::vector<char> covered((size_t)L, 0);
		std::vector<i32> pad_ids;
		for (size_t k = 0; k < p.blocks.size(); k++) {
			const i64 n = k < rows.size() ? rows[k] : 0;
			if (n <= 0) continue;
			const i64 at = k < p.block_at.size() ? p.block_at[k] : -1;
			if (at < 0 || at + n > L)
				throw MediaError(
				    "qwen3vl(visual): a vision block has no placeholder run to splice into");
			for (i64 r = 0; r < n; r++) {
				if (p.tags[(size_t)(at + r)] != (i32)Modality::Visual)
					throw MediaError(
					    "qwen3vl(visual): a vision block's placeholder run is not tagged "
					    "Visual");
				covered[(size_t)(at + r)] = 1;
			}
			pad_ids.push_back(p.ids[(size_t)at]);
			if ((i64)vis.size() < (off + n) * H)
				throw MediaError("qwen3vl(visual): the tower returned too few rows");
			memcpy(&xf[(size_t)at * (size_t)H], &vis[(size_t)off * (size_t)H],
			       (size_t)n * (size_t)H * sizeof(float));
			off += n;
		}
		// Every placeholder has to be covered by a run. Counting Visual tags no
		// longer measures that - the flanking `<|vision_start|>`/`<|vision_end|>`
		// are tagged Visual too, as the reference's modality tags are - so the
		// check is per placeholder id: an `image_pad`/`video_pad` id outside a run
		// would reach the backbone as the embedding table's row for that id, i.e.
		// an image slot filled with a constant, and nothing downstream would
		// notice.
		for (i64 s = 0; s < L; s++) {
			if (covered[(size_t)s]) continue;
			for (i32 id : pad_ids)
				if (p.ids[(size_t)s] == id)
					throw MediaError(
					    "qwen3vl(visual): a vision placeholder outside every vision run at "
					    "token " + std::to_string(s));
		}
		TLOG("[vit] spliced %lld vision rows into %lld tokens", (long long)off, (long long)L);
	}
	return xf;
}

std::vector<float> TextEncoder32B::encode_vl(const PromptTokens& p) {
	// A prompt with no vision block (an audio-only reference, or no reference at
	// all) is the same path as encode(): every row comes out of the embedding
	// table and the plain sequence-position rotary is exact for it, because
	// Qwen3-VL's three mrope axes are all equal when no vision span is present.
	// A prompt *with* a vision span gets the 3-D table `build_mrope_table`
	// derives, so the vision rows carry the grid positions the tower was trained
	// with rather than consecutive ones.
	const std::vector<float> emb = encode_vl_embeds(p);
	std::vector<float> mrope;
	if (!p.blocks.empty()) mrope = build_mrope_table(p, cfg_);
	return encode_embeds(emb.data(), (i64)p.ids.size(), mrope.empty() ? nullptr : &mrope);
}

}  // namespace phi::media
