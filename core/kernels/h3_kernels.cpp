// H3 DiT qkv-prep and bidirectional attention. See h3_kernels.hpp for why these
// two are separate from the image DiT's.
#include "kernels/h3_kernels.hpp"

#include "models/video_vae_internal.hpp"   // silu_gate_fused_hlsl()

#include <cmath>
#include <cstring>

#include "kernels/kernels.hpp"
#include "util/base.hpp"

namespace phi::media {

// ── qkv prep: split + per-head RMSNorm + split-half RoPE ───────────────────
//
// One thread block per (token, head) pair, 128 threads = head_dim. The head row
// lives in groupshared while it is normalised and rotated, the same layout the
// image DiT's qkv_prep uses.
//
// The rope table holds (cos, sin) per *pair* index: pair i rotates dims
// (i, i + rot/2). Angles come from rope.inv_freq (16 frequencies) and the
// three-axis position table, both already evaluated on the host — the reference
// builds them in fp64 and a device-side reimplementation would only be a chance
// to disagree.
static const char* kH3QkvPrepCuda = R"CUDA(
#include <cuda_fp16.h>   // __float2half_rn / __half_as_ushort, packed-fp16 output
struct Args {
    unsigned v[24];    // the old cbuffer, byte offsets included
    const char* s[8];  // srv[0..7] bases (srv_offset folded in by the host)
    char* u[4];        // uav[0..3] bases
};

// The old cbuffer arrives verbatim in a.v[]: S, H, D, F32OUT, qkvOff, qOff,
// kOff, vOff, eps(bits), scale(bits), ropeOff, ropeStride, rot, qkvRow0, outRow0.
__device__ __forceinline__ float ldF(const char* b, unsigned o) {
    return __uint_as_float(*(const unsigned*)(b + o));
}
__device__ __forceinline__ void stU(char* b, unsigned o, unsigned u) {
    *(unsigned*)(b + o) = u;
}

static const unsigned NT = 128;
// Three rows, not one: q, k and v are independent per-head rows held by the same
// (row, head) block, and running them one after the other under one barrier
// chain each made the kernel barrier-bound. `q` and `k` each cost nine
// `__syncthreads()` in `headNorm` alone (one to publish the partials, seven for
// the halving tree, one after the normalised write) against 128 threads of work,
// and the block re-used one `row[]`, so k could not start until q's store was
// out. Holding all three rows and reducing q and k together removes nothing
// from the arithmetic - the tree is still `red[t] += red[t + st]` over the same
// strides - and turns ~24 barriers into 11.
__shared__ float row[3 * NT];
__shared__ float red[3 * NT];

__device__ __forceinline__ void storeRow(char* dst, unsigned byteBase, const float* r,
                                         unsigned t, unsigned D, unsigned f32) {
    if (f32 != 0u) {
        if (t < D) stU(dst, byteBase + t * 4u, __float_as_uint(r[t]));
    } else {
        // packed fp16 for attn_tiled: two halves per 32-bit word, low first
        const unsigned D2 = D / 2u;
        if (t < D2) {
            const unsigned lo = (unsigned)__half_as_ushort(__float2half_rn(r[2u * t]));
            const unsigned hi =
                (unsigned)__half_as_ushort(__float2half_rn(r[2u * t + 1u]));
            stU(dst, byteBase + t * 4u, lo | (hi << 16));
        }
    }
}

// numthreads(128, 1, 1) -> one thread per head dim, one block per (s, head).
// The early return is block-uniform (blockIdx.x/y are the same for every
// thread), so it cannot strand a __syncthreads().
extern "C" __global__ void h3_qkv_prep(Args a) {
    const unsigned S = a.v[0], H = a.v[1], D = a.v[2], F32OUT = a.v[3];
    const unsigned qkvOff = a.v[4], qOff = a.v[5], kOff = a.v[6], vOff = a.v[7];
    const float eps = __uint_as_float(a.v[8]);
    const float scale = __uint_as_float(a.v[9]);
    const unsigned ropeStride = a.v[11], rot = a.v[12];
    const unsigned qkvRow0 = a.v[13], outRow0 = a.v[14];
    const char* QKV = a.s[0];
    const char* WQ = a.s[1];
    const char* WK = a.s[2];
    const char* ROPE = a.s[3];

    const unsigned s = blockIdx.x;
    const unsigned h = blockIdx.y;
    const unsigned t = threadIdx.x;
    if (s >= S) return;
    const unsigned stride = 3u * H * D;
    const unsigned half = rot / 2u;
    const unsigned esz = (F32OUT != 0u) ? 4u : 2u;
    // The input row lives in the caller's fused buffer, the output row (and
    // the rope row) in the caller's slice of the packed sequence: they differ
    // when the DiT runs the projection in chunks off a chunk-sized scratch.
    const unsigned sr = qkvRow0 + s;
    const unsigned so = outRow0 + s;

    // q, k, v for this (row, head), all three loads issued before the first
    // barrier. v needs neither the norm nor the rope, but it shares the load and
    // store phases, which is where the barrier count comes from.
    row[0u * NT + t] = ldF(QKV, qkvOff + (sr * stride + h * D + t) * 4u);
    row[1u * NT + t] = ldF(QKV, qkvOff + (sr * stride + (H + h) * D + t) * 4u);
    row[2u * NT + t] = ldF(QKV, qkvOff + (sr * stride + (2u * H + h) * D + t) * 4u);

    // q and k share one reduction tree: same strides, same order per row, three
    // rows' worth of work behind one barrier per step instead of three.
    red[0u * NT + t] = row[0u * NT + t] * row[0u * NT + t];
    red[1u * NT + t] = row[1u * NT + t] * row[1u * NT + t];
    __syncthreads();
    for (unsigned st = NT / 2u; st > 0u; st >>= 1) {
        if (t < st) {
            red[0u * NT + t] += red[0u * NT + t + st];
            red[1u * NT + t] += red[1u * NT + t + st];
        }
        __syncthreads();
    }
    const float invq = rsqrtf(red[0u * NT] / (float)D + eps);
    const float invk = rsqrtf(red[1u * NT] / (float)D + eps);
    row[0u * NT + t] = row[0u * NT + t] * invq * ldF(WQ, t * 4u);
    row[1u * NT + t] = row[1u * NT + t] * invk * ldF(WK, t * 4u);
    __syncthreads();

    // Split-half rotation over the first `rot` dims: pair i is (i, i + rot/2).
    // Thread t owns pair (t, t + half) for t < half, so it reads both halves
    // before writing either and no other thread touches them - one barrier in
    // front of the branch is all the synchronisation this needs, and the two
    // rows carry the same rope row, so they share it.
    if (t < half) {
        const unsigned base = (so * ropeStride + t) * 8u;
        const float c = ldF(ROPE, base);
        const float sn = ldF(ROPE, base + 4u);
        #pragma unroll
        for (unsigned m = 0u; m < 2u; m++) {
            float* r = row + m * NT;
            const float x0 = r[t];
            const float x1 = r[t + half];
            r[t] = x0 * c - x1 * sn;
            r[t + half] = x1 * c + x0 * sn;
        }
    }
    __syncthreads();
    row[0u * NT + t] *= scale;
    __syncthreads();
    storeRow(a.u[0], qOff + (so * H + h) * D * esz, row + 0u * NT, t, D, F32OUT);
    storeRow(a.u[1], kOff + (so * H + h) * D * esz, row + 1u * NT, t, D, F32OUT);
    storeRow(a.u[2], vOff + (so * H + h) * D * esz, row + 2u * NT, t, D, F32OUT);
}
)CUDA";

void dispatch_h3_qkv_prep(ComputeContext& ctx, const H3QkvPrepArgs& a) {
	if (a.head_dim != 128) throw MediaError("h3_qkv_prep: head_dim must be 128");
	// rot == 0 is legal and means "no rotation": that is what the token refiner's
	// attention needs (it passes rope_freqs = None).
	if (a.rot < 0 || a.rot > a.head_dim || (a.rot % 2) != 0)
		throw MediaError("h3_qkv_prep: rot must be even and <= head_dim");
	GpuKernel* pso =
	    ctx.pipeline("h3_qkv_prep", kH3QkvPrepCuda, "h3_qkv_prep", ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)a.s;
	p.values[1] = (u32)a.heads;
	p.values[2] = (u32)a.head_dim;
	p.values[3] = a.f32_out ? 1u : 0u;   // 0 = packed fp16 for attn_tiled
	p.values[4] = (u32)a.qkv.off;
	p.values[5] = (u32)a.q.off;
	p.values[6] = (u32)a.k.off;
	p.values[7] = (u32)a.v.off;
	float eps = a.eps;
	memcpy(&p.values[8], &eps, 4);
	float sc = a.scale;
	memcpy(&p.values[9], &sc, 4);
	p.values[10] = (u32)a.rope.off;
	// ropeStride is the number of *pairs* per row, not the number of floats: the
	// kernel indexes the table as (s * ropeStride + pair) * 8 bytes, and a pair is
	// two floats. Passing the float count here reads two rows ahead per row and
	// walks off the end of the last few tokens.
	p.values[11] = (u32)(a.rot / 2);
	p.values[12] = (u32)a.rot;
	p.values[13] = (u32)a.qkv_row0;
	p.values[14] = (u32)a.out_row0;
	p.srv[0] = a.qkv.res;
	p.srv[1] = a.wq.res;
	p.srv[2] = a.wk.res;
	p.srv[3] = a.rope.res;
	// Offset convention (the same one the image DiT's qkv_prep uses): the tensors whose
	// byte offset has a constant-buffer slot (qkv, q, k, v) are addressed *inside*
	// the shader and their root descriptor must point at the resource base, while
	// the three side tables (wq, wk, rope) have no slot and carry their offset in
	// the descriptor itself. Getting this backwards reads the right bytes from the
	// wrong place, or nothing at all.
	p.srv_offset[0] = 0;
	p.srv_offset[1] = a.wq.off;
	p.srv_offset[2] = a.wk.off;
	p.srv_offset[3] = a.rope.off;
	p.uav[0] = a.q.res;
	p.uav[1] = a.k.res;
	p.uav[2] = a.v.res;
	ctx.dispatch(pso, p, (u32)a.s, (u32)a.heads, 1);
}

// ── bidirectional attention over a query window ────────────────────────────
//
// Same shape as attn_flash (32 query rows and 32 keys per block, 128 threads,
// one online-softmax pass) minus the causal mask and the fp16 path, plus the
// explicit query window. The online softmax is not an optimisation here: at
// S = 20k a materialised score tile is 1.6 GB.
static const char* kH3AttnCuda = R"CUDA(
struct Args {
    unsigned v[24];    // the old cbuffer, byte offsets included
    const char* s[8];  // srv[0..7] bases (srv_offset folded in by the host)
    char* u[4];        // uav[0..3] bases
};

// v[0..10]: Sk, Q0, Sq, H, D, qOff, kOff, vOff, oOff, scale(bits), seqStride.
__device__ __forceinline__ float ldF(const char* b, unsigned o) {
    return __uint_as_float(*(const unsigned*)(b + o));
}
__device__ __forceinline__ void stF(char* b, unsigned o, float v) {
    *(unsigned*)(b + o) = __float_as_uint(v);
}

static const unsigned BM = 32;
static const unsigned BN = 32;
static const unsigned NT = 128;
static const unsigned DGRP = 32;

__shared__ float sP[BM][BN];
__shared__ float sMax[BM];
__shared__ float sSum[BM];
__shared__ float sCorr[BM];

// numthreads(128, 1, 1): 4 threads cooperate per query row (r = t/4), each
// owning DGRP = 32 of the head dims and 8 keys of the current tile.
extern "C" __global__ void h3_attn(Args a) {
    const unsigned Sk = a.v[0], Q0 = a.v[1];
    const unsigned H = a.v[3], D = a.v[4];
    const unsigned qOff = a.v[5], kOff = a.v[6], vOff = a.v[7], oOff = a.v[8];
    const float scale = __uint_as_float(a.v[9]);
    const unsigned seqStride = a.v[10];
    const char* Q = a.s[0];
    const char* K = a.s[1];
    const char* V = a.s[2];
    char* O = a.u[0];

    const unsigned t = threadIdx.x;
    const unsigned r = t / 4u;
    const unsigned g = t % 4u;
    const unsigned h = blockIdx.y;
    const unsigned qIdx = Q0 + blockIdx.x * BM + r;
    const bool qOk = qIdx < Sk;

    float outAcc[DGRP];
    #pragma unroll
    for (unsigned i = 0; i < DGRP; i++) outAcc[i] = 0.0f;

    if (t < BM) {
        sMax[t] = -3.0e38f;
        sSum[t] = 0.0f;
    }
    __syncthreads();

    const unsigned qRowBase = qOff + (qIdx * seqStride + h * D) * 4u;
    const unsigned kRowBase0 = kOff + h * D * 4u;
    const unsigned vRowBase0 = vOff + h * D * 4u;
    const unsigned kTiles = (Sk + BN - 1u) / BN;

    for (unsigned kt = 0; kt < kTiles; kt++) {
        const unsigned kBase = kt * BN;

        {
            float acc[8];
            #pragma unroll
            for (unsigned j = 0; j < 8; j++) acc[j] = 0.0f;
            const unsigned kRow = kBase + g * 8u;
            if (qOk && kRow < Sk) {
                for (unsigned d = 0; d < D; d++) {
                    const float qv = ldF(Q, qRowBase + d * 4u);
                    #pragma unroll
                    for (unsigned j = 0; j < 8; j++) {
                        const unsigned kj = kRow + j;
                        if (kj < Sk)
                            acc[j] += qv * ldF(K, kRowBase0 + (kj * seqStride + d) * 4u);
                    }
                }
            }
            #pragma unroll
            for (unsigned j2 = 0; j2 < 8; j2++) {
                const unsigned kj = kBase + g * 8u + j2;
                sP[r][g * 8u + j2] = (qOk && kj < Sk) ? acc[j2] * scale : -3.0e38f;
            }
        }
        __syncthreads();

        if (g == 0u) {
            float bmax = -3.0e38f;
            #pragma unroll
            for (unsigned j = 0; j < BN; j++) bmax = fmaxf(bmax, sP[r][j]);
            const float oldmax = sMax[r];
            const float newmax = fmaxf(oldmax, bmax);
            const float corr = (oldmax <= -3.0e37f) ? 0.0f : expf(oldmax - newmax);
            float sm = sSum[r] * corr;
            #pragma unroll
            for (unsigned j2 = 0; j2 < BN; j2++) {
                const float e = expf(sP[r][j2] - newmax);
                sP[r][j2] = e;
                sm += e;
            }
            sMax[r] = newmax;
            sSum[r] = sm;
            sCorr[r] = corr;
        }
        __syncthreads();

        const float corr = sCorr[r];
        #pragma unroll
        for (unsigned i = 0; i < DGRP; i++) outAcc[i] *= corr;
        {
            const unsigned dBase = g * DGRP;
            for (unsigned j = 0; j < BN; j++) {
                const unsigned kj = kBase + j;
                if (kj >= Sk) continue;
                const float p = sP[r][j];
                if (p == 0.0f) continue;
                const unsigned vRow = vRowBase0 + (kj * seqStride + dBase) * 4u;
                #pragma unroll
                for (unsigned i = 0; i < DGRP; i++)
                    outAcc[i] += p * ldF(V, vRow + i * 4u);
            }
        }
        __syncthreads();
    }

    if (!qOk) return;   // block-uniform: qIdx only depends on blockIdx/r
    const float inv = 1.0f / fmaxf(sSum[r], 1e-20f);
    const unsigned dBase = g * DGRP;
    #pragma unroll
    for (unsigned i = 0; i < DGRP; i++)
        stF(O, oOff + (qIdx * seqStride + h * D + dBase + i) * 4u, outAcc[i] * inv);
}
)CUDA";

namespace {

// PHI_H3_ATTN=fp32 forces the original kernel. Default: the tiled fp16 one.
bool h3_attn_fp16_ok() {
	static int once = [] {
		if (auto e = get_env("PHI_H3_ATTN")) {
			const std::string v = *e;
			if (v == "fp32" || v == "0") return 0;
		}
		return 1;
	}();
	return once != 0;
}

}  // namespace

void dispatch_h3_attn(ComputeContext& ctx, const H3AttnArgs& a) {
	if (a.head_dim != 128) throw MediaError("h3_attn: head_dim must be 128");
	if (a.sq <= 0 || a.q0 < 0 || a.q0 + a.sq > a.sk)
		throw MediaError("h3_attn: bad query window");

	// ── tiled fp16 path ────────────────────────────────────────────────────
	//
	// The DiT's attention is bidirectional over the whole packed sequence and at
	// 540P/243 frames S is ~37500, which is what decides the kernel: the fp32
	// one re-derives every address and converts one element per
	// multiply-accumulate (measured 0.25 TMAC/s, and a *single* 32-row block at
	// S > ~1200 already runs longer than the 2 s TDR watchdog tolerates).
	//
	// The tiled kernel reads packed fp16 words and register-blocks the score and
	// output phases. Its query axis is chunked here for the same reason the fp32
	// path chunks it, but the budget is computed from the work in the dispatch
	// instead of a fixed 1024 rows: past a few thousand keys one 64-row block is
	// already seconds, and a fixed row count is exactly how the device gets lost.
	if (a.fp16_qkv) {
		if (!h3_attn_fp16_ok())
			throw MediaError(
			    "h3_attn: fp16 q/k/v needs the tiled kernel (PHI_H3_ATTN=fp32 is a debug "
			    "switch, and the fp32 kernel would read packed halves as fp32)");
		// The query tile (see attn_tiled_src): 128 rows by default, which halves
		// the K/V traffic of the 64-row tile the kernel was tuned at. A dispatch
		// covers `rows` rows in ceil(rows/BM) blocks of BM, and `rows` never
		// exceeds the caller's chunk, so growing the tile can only leave the
		// output window shorter than the dispatch - never longer - and the
		// kernel's `row < S` store guard covers what is left.
		const unsigned bm = attn_query_tile();
		const std::string ak = attn_tiled_name(bm, 128u);
		GpuKernel* pso = ctx.pipeline(ak, attn_tiled_src(bm, 128u), "attn_tiled",
		                              ShaderModel::SM6_2);
		const float scale = a.scale != 0.0f ? a.scale : (float)(1.0 / std::sqrt((double)a.head_dim));
		// One query row costs heads * sk * 2 * head_dim MACs; the tiled kernel
		// sustains ~1.8 TMAC/s on this part. Keep a dispatch near 1 s of work,
		// but never below one query tile (the kernel's own block).
		const double row_macs = (double)a.heads * (double)a.sk * 2.0 * (double)a.head_dim;
		i64 chunk = (i64)(1.0e12 / std::max(row_macs, 1.0));
		chunk = std::max<i64>((i64)bm, std::min<i64>(chunk, 2048));
		for (i64 r0 = a.q0; r0 < a.q0 + a.sq; r0 += chunk) {
			const i64 rows = std::min<i64>(chunk, a.q0 + a.sq - r0);
			KernelParams p{};
			p.values[0] = (u32)a.sk;
			p.values[1] = (u32)a.heads;
			p.values[2] = 128;
			p.values[3] = (u32)a.q.off;
			p.values[4] = (u32)a.k.off;
			p.values[5] = (u32)a.v.off;
			// `o_row0` lets the caller drop this window's output at row 0 of a
			// buffer only `sq` rows tall (see H3AttnArgs::o_row0). The stride
			// the kernel indexes with is heads*head_dim elements per row.
			p.values[6] =
			    (u32)(a.o.off -
			          (u64)a.o_row0 * (u64)a.heads * (u64)a.head_dim * 4u);
			memcpy(&p.values[7], &scale, 4);
			p.values[8] = (u32)(a.heads * a.head_dim);
			p.values[9] = (u32)a.heads;                       // no GQA
			p.values[10] = (u32)(a.heads * a.head_dim);
			p.values[11] = 0u;                                // not causal
			p.values[14] = (u32)r0;                           // query-row window
			p.srv[0] = a.q.res;
			p.srv[1] = a.k.res;
			p.srv[2] = a.v.res;
			p.uav[0] = a.o.res;
			ctx.dispatch(pso, p, (u32)ceil_div(rows, (i64)bm), (u32)a.heads, 1);
			ctx.submit();
			ctx.begin();
		}
		return;
	}
	// The watchdog measures the command list, not the dispatch: one list holding
	// the whole 20k-token attention is tens of seconds of GPU work and takes the
	// device down (a TDR, i.e. a lost device). 512 query rows is ~1/40th of that,
	// and the drains are cheap (a fence wait against hundreds of ms of work).
	const i64 chunk = a.sq > 1024 ? 1024 : a.sq;
	GpuKernel* pso = ctx.pipeline("h3_attn", kH3AttnCuda, "h3_attn", ShaderModel::SM5_1);
	const float scale = a.scale != 0.0f ? a.scale : (float)(1.0 / std::sqrt((double)a.head_dim));
	for (i64 r0 = a.q0; r0 < a.q0 + a.sq; r0 += chunk) {
		const i64 rows = a.sq - (r0 - a.q0) < chunk ? a.sq - (r0 - a.q0) : chunk;
		KernelParams p{};
		p.values[0] = (u32)a.sk;
		p.values[1] = (u32)r0;
		p.values[2] = (u32)rows;
		p.values[3] = (u32)a.heads;
		p.values[4] = 128;
		p.values[5] = (u32)a.q.off;
		p.values[6] = (u32)a.k.off;
		p.values[7] = (u32)a.v.off;
		p.values[8] =
		    (u32)(a.o.off - (u64)a.o_row0 * (u64)a.heads * (u64)a.head_dim * 4u);
		memcpy(&p.values[9], &scale, 4);
		p.values[10] = (u32)(a.heads * a.head_dim);
		p.srv[0] = a.q.res;
		p.srv[1] = a.k.res;
		p.srv[2] = a.v.res;
		p.uav[0] = a.o.res;
		ctx.dispatch(pso, p, (u32)ceil_div(rows, 32), (u32)a.heads, 1);
		ctx.submit();
		ctx.begin();
	}
}

// ── fused gate/value activation over a [rows, 2*cols] buffer ───────────────
//
// Shared by the H3 DiT (every block's MLP) and the H3 video VAE. The HLSL lives
// in kernels/conv3d_causal.cpp, where the pairing is documented.
void dispatch_silu_gate_fused(ComputeContext& ctx, const GpuAlloc& gx, const GpuAlloc& out,
                              i64 rows, i64 cols) {
	const i64 total = rows * cols;
	if (rows <= 0 || cols <= 0) throw MediaError("silu_gate_fused: empty");
	GpuKernel* pso =
	    ctx.pipeline("h3vae_silu_gate_fused", silu_gate_fused_hlsl(), "silu_gate_fused",
	                 ShaderModel::SM5_1);
	KernelParams p{};
	p.values[0] = (u32)rows;
	p.values[1] = (u32)cols;
	p.values[2] = (u32)gx.off;
	p.values[3] = (u32)out.off;
	p.values[4] = (u32)total;
	p.srv[0] = gx.res;
	p.uav[0] = out.res;
	ctx.dispatch(pso, p, (u32)ceil_div(total, 256), 1, 1);
}
// ── int4 GEMM (the Qwen3-VL text encoder's `convrot_w4a4` weights) ─────────
//
// Same tile and accumulation order as the engine's int8 GEMM, so the two agree
// bit-for-bit when fed the same int8 operands; the only change is the B tile
// loader, which expands packed nibbles instead of loading bytes.
//
// B is [N, K/2]: byte j of a row holds column 2j in its low nibble and column
// 2j+1 in its high nibble (comfy_kitchen's `_pack_int4_row_major`), and the
// nibble is a *signed* 4-bit field.
static const char* kInt4GemmCuda = R"CUDA(
struct Args {
    unsigned v[24];    // the old cbuffer, byte offsets included
    const char* s[8];  // srv[0..7] bases (srv_offset folded in by the host)
    char* u[4];        // uav[0..3] bases
};

// v[0..7]: M, N, K, aByteOffset, bByteOffset, cByteOffset, saByteOffset,
// sbByteOffset.  srv: A(t0), B(t1), SA(t2), SB(t3).  uav: C(u0).
static const unsigned BM = 128;
static const unsigned BN = 64;
static const unsigned BK = 64;
static const unsigned TM = 4;
static const unsigned TN = 8;
static const unsigned NT = 256;

static const unsigned AS_STRIDE = BK / 4 + 1;
static const unsigned BS_STRIDE = BK / 4 + 1;

__shared__ int sA[BM][AS_STRIDE];
__shared__ int sB[BN][BS_STRIDE];

__device__ __forceinline__ float ldF(const char* b, unsigned o) {
    return __uint_as_float(*(const unsigned*)(b + o));
}
__device__ __forceinline__ void stF(char* b, unsigned o, float v) {
    *(unsigned*)(b + o) = __float_as_uint(v);
}
__device__ __forceinline__ unsigned ldU(const char* b, unsigned o) {
    return *(const unsigned*)(b + o);
}

// dot4add_i8packed has no sub-6.4 equivalent to fall back to; __dp4a is the only
// path, and it is what the whole engine targets.
__device__ __forceinline__ int dot4(int x, int y) {
    return __dp4a(x, y, 0);
}

// four consecutive int8 values, packed little-endian into one int32
__device__ __forceinline__ int load4(const char* buf, unsigned byteOffset,
                                     unsigned rowStride, unsigned k) {
    return (int)ldU(buf, byteOffset + rowStride + k);
}

// Two dp4a operands out of eight packed nibbles.
//
// The packing is column-major over *nibbles*: a 32-bit word holds columns
// 8u..8u+7 with nibble j (0 = least significant) being column 8u+j, and the
// nibble is a signed 4-bit field. The tile stores those as two int32s of four
// int8s each, so nibbles 0..3 become one operand and 4..7 the next, in order.
// (Interleaving them the other way round - low nibbles in one word, high
// nibbles in the other - is the mistake this comment exists to prevent: it
// produces an answer that is *nearly* right and a rel L2 of 1.1.)
// numthreads(256, 1, 1)
extern "C" __global__ void int4_gemm(Args a) {
    const unsigned M = a.v[0], N = a.v[1], K = a.v[2];
    const unsigned aByteOffset = a.v[3], bByteOffset = a.v[4];
    const unsigned cByteOffset = a.v[5];
    const unsigned saByteOffset = a.v[6], sbByteOffset = a.v[7];
    const char* A = a.s[0];
    const char* B = a.s[1];
    const char* SA = a.s[2];
    const char* SB = a.s[3];
    char* C = a.u[0];

    const unsigned tid = threadIdx.x;
    const unsigned tx = tid % (BN / TN);
    const unsigned ty = tid / (BN / TN);
    const unsigned mBase = blockIdx.x * BM + ty * TM;
    const unsigned nBase = blockIdx.y * BN + tx * TN;

    int acc[TM][TN];
    #pragma unroll
    for (unsigned i = 0; i < TM; i++)
        #pragma unroll
        for (unsigned j = 0; j < TN; j++) acc[i][j] = 0;

    const unsigned kTiles = (K + BK - 1) / BK;
    for (unsigned kt = 0; kt < kTiles; kt++) {
        const unsigned kBase = kt * BK;

        #pragma unroll
        for (unsigned s = 0; s < (BM * BK / 4) / NT; s++) {
            const unsigned li = tid + s * NT;
            const unsigned row = li / (BK / 4);
            const unsigned col = li % (BK / 4);
            const unsigned m = blockIdx.x * BM + row;
            const unsigned k = kBase + col * 4;
            sA[row][col] = (m < M) ? load4(A, aByteOffset, m * K, k) : 0;
        }
        // B rows are K/2 bytes: one word carries 8 columns, i.e. two tile words
        #pragma unroll
        for (unsigned s = 0; s < (BN * BK / 8) / NT; s++) {
            const unsigned li = tid + s * NT;
            const unsigned row = li / (BK / 8);
            const unsigned u = li % (BK / 8);
            const unsigned n = blockIdx.y * BN + row;
            unsigned w = 0u;
            if (n < N) w = ldU(B, bByteOffset + n * (K / 2u) + (kBase + u * 8u) / 2u);
            unsigned lo = 0u, hi = 0u;
            #pragma unroll
            for (unsigned i = 0; i < 4u; i++) {
                int va = (int)((w >> (4u * i)) & 0xFu);
                int vb = (int)((w >> (4u * i + 16u)) & 0xFu);
                if (va >= 8) va -= 16;
                if (vb >= 8) vb -= 16;
                lo |= ((unsigned)(va & 0xFF)) << (8u * i);
                hi |= ((unsigned)(vb & 0xFF)) << (8u * i);
            }
            sB[row][u * 2u] = (int)lo;
            sB[row][u * 2u + 1u] = (int)hi;
        }
        __syncthreads();

        #pragma unroll
        for (unsigned kk = 0; kk < BK / 4; kk++) {
            int av[TM], bv[TN];
            #pragma unroll
            for (unsigned i = 0; i < TM; i++) av[i] = sA[ty * TM + i][kk];
            #pragma unroll
            for (unsigned j = 0; j < TN; j++) bv[j] = sB[tx * TN + j][kk];
            #pragma unroll
            for (unsigned i = 0; i < TM; i++)
                #pragma unroll
                for (unsigned j = 0; j < TN; j++) acc[i][j] += dot4(av[i], bv[j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned i = 0; i < TM; i++) {
        const unsigned m = mBase + i;
        if (m >= M) continue;
        const float sa = ldF(SA, saByteOffset + m * 4);
        #pragma unroll
        for (unsigned j = 0; j < TN; j++) {
            const unsigned n = nBase + j;
            if (n >= N) continue;
            const float sb = ldF(SB, sbByteOffset + n * 4);
            stF(C, cByteOffset + (m * N + n) * 4, (float)acc[i][j] * sa * sb);
        }
    }
}
)CUDA";

void dispatch_int4_gemm(ComputeContext& ctx, const Int4GemmArgs& a) {
	if (a.m <= 0 || a.n <= 0 || a.k <= 0) throw MediaError("int4_gemm: empty problem");
	if ((a.k % 512) != 0)
		throw MediaError("int4_gemm: K=" + std::to_string(a.k) + " is not a multiple of 512");
	// SM 6.4 for dot4add_i8packed, exactly like the engine's int8 GEMM. The
	// emulation path is not just slower — compiling it for a kernel of this size
	// takes minutes and then fails, and that failure looks like a hung dispatch
	// rather than a compile error.
	GpuKernel* pso =
	    ctx.pipeline("int4_gemm", kInt4GemmCuda, "int4_gemm", ShaderModel::SM6_4);
	KernelParams p{};
	p.values[0] = (u32)a.m;
	p.values[1] = (u32)a.n;
	p.values[2] = (u32)a.k;
	p.values[3] = (u32)a.a.off;
	p.values[4] = (u32)a.b.off;
	p.values[5] = (u32)a.c.off;
	p.values[6] = (u32)a.sa.off;
	p.values[7] = (u32)a.sb.off;
	p.srv[0] = a.a.res;
	p.srv[1] = a.b.res;
	p.srv[2] = a.sa.res;
	p.srv[3] = a.sb.res;
	p.uav[0] = a.c.res;
	// Every operand is addressed inside the shader from its constant-buffer offset,
	// so all four descriptors point at their resource base.
	p.srv_offset[0] = 0;
	p.srv_offset[1] = 0;
	p.srv_offset[2] = 0;
	p.srv_offset[3] = 0;
	p.uav_offset[0] = 0;
	ctx.dispatch(pso, p, (u32)ceil_div(a.m, 128), (u32)ceil_div(a.n, 64), 1);
}

// ── host reference (same arithmetic order as the shader) ───────────────────
void h3_rope_ref(const float* qkv, float* q, float* k, float* v, const float* wq,
                 const float* wk, const float* rope, i64 s, i64 heads, i64 head_dim,
                 i64 rot, float eps, float scale) {
	const i64 stride = 3 * heads * head_dim;
	const i64 half = rot / 2;
	for (i64 t = 0; t < s; t++) {
		for (i64 h = 0; h < heads; h++) {
			for (int which = 0; which < 3; which++) {
				const float* src = qkv + t * stride + (which * heads + h) * head_dim;
				float* dst = (which == 0 ? q : which == 1 ? k : v) + (t * heads + h) * head_dim;
				float r[256];
				for (i64 d = 0; d < head_dim; d++) r[d] = src[d];
				if (which < 2) {
					const float* w = which == 0 ? wq : wk;
					double ss = 0;
					for (i64 d = 0; d < head_dim; d++) ss += (double)r[d] * r[d];
					const float inv = (float)(1.0 / std::sqrt(ss / (double)head_dim + (double)eps));
					for (i64 d = 0; d < head_dim; d++) r[d] = r[d] * inv * w[d];
					for (i64 i = 0; i < half; i++) {
						const float c = rope[(t * half + i) * 2];
						const float sn = rope[(t * half + i) * 2 + 1];
						const float x0 = r[i], x1 = r[i + half];
						r[i] = x0 * c - x1 * sn;
						r[i + half] = x1 * c + x0 * sn;
					}
					if (which == 0)
						for (i64 d = 0; d < head_dim; d++) r[d] *= scale;
				}
				for (i64 d = 0; d < head_dim; d++) dst[d] = r[d];
			}
		}
	}
}

}  // namespace phi::media
