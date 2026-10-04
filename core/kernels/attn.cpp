// Attention kernels.
//
// The shapes differ enormously, so there are three entry points:
//
//   * `qkv_prep` + `attn_flash` - the Qwen-Image-2.1 DiT and the H3 DiT:
//     30 heads x head_dim 128, sequence up to ~8k tokens.  A materialised score
//     matrix would be S^2 fp32 per head (266 MB at 1080p), so the softmax
//     streams: online rescaling over 32-key blocks, exactly like flash
//     attention.
//
//   * `rope_half` - the Qwen3 half-split (GPT-J style) rotation the text
//     encoder and the language-ish towers use.
//
//   * `attn_tiled` - the same tiled FA as `attn_flash` with fp16 operands and
//     the matmuls on the tensor cores.
//
// ── HLSL -> CUDA ───────────────────────────────────────────────────────────
//
// These are the same kernels emitted as CUDA C++ (NVRTC) instead of HLSL. The
// host-side contract did not move, so neither did the parameter layout: every
// entry keeps its original name, is `extern "C"` (the host looks the kernel up
// by that name) and takes the whole frozen block by value:
//
//   struct Args { unsigned v[24]; const char* s[8]; char* u[4]; };
//
// `a.v[]` is the old root constant buffer, **byte offsets included**, in the
// same order and with the same meaning - the `cbuffer P` layout is recorded in
// a comment at the top of each kernel so the two can be diffed by eye. `a.s[i]`
// / `a.u[i]` are the resource bases with the host-side `srv_offset` /
// `uav_offset` already folded in, exactly like the root descriptors were, so
// the tensors whose byte offset has a constant-buffer slot (qkv/q/k/v, X/Y, ...)
// are addressed by adding it here, while the side tables that rode their
// descriptor (wq/wk/rope/ids) must NOT have `a.v[]` added a second time. Nothing
// is reordered or renumbered.
//
// Group size is no longer part of the kernel: the host still launches with the
// original `[numthreads(...)]` value, which is recorded in a comment next to
// each entry (128 for rope_half / qkv_prep / attn_flash / attn_tiled). The
// thread-map arithmetic below (lane / warp / rr / kk / rr2 /
// dd) still assumes that block shape, so those two facts have to stay together.
// `attn_tiled` partitions its 64 query rows as 4 warps x 16 rows and every
// warp keeps its own online-softmax state, so it needs the full NT = 128: with
// 64 threads half the rows would never be computed. The grid is
// (ceil(rows/BM), heads) with BM = 64 (the query-rows-per-block tile, which the
// host computes as ceil_div(s, 64) and this kernel must therefore keep).
//
// Wave intrinsics: `attn_tiled` is the only one that used them, and only as the
// butterfly `WaveReadLaneAt(v, lane ^ off)`. The fp32 revision reduced over the
// eight key-group lanes with off in {4,8,16}; the tensor-core revision reduces
// the four lanes that share a row with off in {1,2}, because the wmma
// accumulator gives lane (lane>>2, lane&3) the columns 2c,2c+1,2c+8,2c+9 of
// rows lane>>2 and lane>>2 + 8. Both are `__shfl_xor_sync(0xffffffffu, v, off)`
// - the value the warp read from lane `laneid ^ off` - over lanes that all
// hold a piece of the same row, in the same tree order.
//
// Two things had to change because CUDA has none of a graphics API's
// out-of-range robustness (reads there clamp, stores are dropped; here they
// fault):
//
//   * `attn_flash` rounds its query block up to BM rows, so the last block
//     addresses query rows >= S. Those rows now read Q from row 0 (the guard
//     `attn_tiled` already used) and are not stored. Every row < S is computed
//     bit-identically to the HLSL. `attn_tiled` itself was already in bounds by
//     construction (qOk / kOk / min(key, S-1)), so it needed nothing.
//   * everywhere else the HLSL was already bounded, so nothing else moved.
//
// Everything else is a literal translation; the arithmetic order, the reduction
// structure and the shared-memory layout are unchanged.
#include "kernels/kernels.hpp"

#include <cctype>
#include <cstdlib>
#include <map>
#include <string>

#include "util/media_common.hpp"

namespace phi::media {

// ── half-split RoPE (HF / Qwen) ───────────────────────────────────────────
//
// pos depends only on the token, and each token owns `heads` rows, so
// pos = row / heads + base.  inv_freq[i] = theta^(-2i/D) and the rotation is
//   y[i]      = x[i]      * cos - x[i+D/2] * sin
//   y[i+D/2]  = x[i+D/2]  * cos + x[i]     * sin
//
// (The static const char* names keep their `...Hlsl` spelling: the C++
// structure of this file is unchanged, only the string bodies are CUDA now.)
static const char* kRopeHalfHlsl = R"CUDA(
// cbuffer P : register(b0) { uint rows, D, xOff, yOff; float theta; uint heads,
//                             base; uint extra0..4; }
struct Args {
    unsigned v[24];      // 0 rows  1 D  2 xOff  3 yOff  4 theta  5 heads  6 base
    const char* s[8];    // srv[0] = X
    char* u[4];          // uav[0] = Y
};

// [numthreads(128, 1, 1)] - the host launches `rows` blocks of 128 threads.
extern "C" __global__ void rope_half(Args a) {
    const unsigned rows = a.v[0], D = a.v[1];
    const unsigned xOff = a.v[2], yOff = a.v[3];
    const float theta = __uint_as_float(a.v[4]);
    const unsigned heads = a.v[5], base = a.v[6];
    const unsigned char* X = (const unsigned char*)a.s[0];
    unsigned char* Y = (unsigned char*)a.u[0];

    const unsigned row = blockIdx.x;    // gid.x
    const unsigned t = threadIdx.x;     // tid.x
    if (row >= rows || t >= D / 2u) return;
    const float pos = (float)(row / heads + base) + (float)(0);
    const float inv = powf(theta, (-2.0f * (float)t) / (float)D);
    const float ang = pos * inv;
    const float c = cosf(ang), sn = sinf(ang);
    // A ByteAddressBuffer.Load is a 32-bit load at a 4-byte-aligned offset;
    // every offset here is a multiple of 4.
    const unsigned off = xOff + row * D * 4u;
    const float x0 = __uint_as_float(*(const unsigned*)(X + off + t * 4u));
    const float x1 = __uint_as_float(*(const unsigned*)(X + off + (t + D / 2u) * 4u));
    const unsigned ob = yOff + row * D * 4u;
    *(unsigned*)(Y + ob + t * 4u) = __float_as_uint(x0 * c - x1 * sn);
    *(unsigned*)(Y + ob + (t + D / 2u) * 4u) = __float_as_uint(x1 * c + x0 * sn);
}
)CUDA";

const char* rope_half_hlsl() { return kRopeHalfHlsl; }

// ── DiT: q/k/v split, per-head RMSNorm + axial RoPE ───────────────────────
//
// Input  qkv: [S, (H + 2*Hk) * D] fp32, laid out [q | k | v] per the reference's
//             torch.split order.
// Output q, k, v: [S, Hk, D] fp16 (or fp32 when `flags` bit2 is set); q is
//             pre-scaled by 1/sqrt(D).
// ROPE packs [omega(0..D/2) | axis(0..D/2)] as fp32; IDS is [S,3] fp32, read
// from row `idsRow + s` so a block running on a slice of the sequence (the DiT
// noise refiner on the image half) sees its own positions.
static const char* kQkvPrepHlsl = R"CUDA(
#include <cuda_fp16.h>   // __float2half_rn / __half_as_ushort

// cbuffer P : register(b0) {
//     uint S, H, Hk, D;
//     uint qkvOff, qOff, kOff, vOff;
//     float eps;
//     float scale;
//     uint flags;          // bit0 rmsnorm on q/k, bit1 rope
//     uint idsOff;
//     uint ropeOff;
//     uint idsRow;         // first ids row of this sequence (cap tokens come first)
//     uint extra0;
// };
struct Args {
    unsigned v[24];     // 0 S  1 H  2 Hk  3 D  4 qkvOff  5 qOff  6 kOff  7 vOff
                        // 8 eps  9 scale  10 flags  11 idsOff  12 ropeOff  13 idsRow
    const char* s[8];   // srv[0] QKV  srv[1] WQ  srv[2] WK  srv[3] ROPE  srv[4] IDS
    char* u[4];         // uav[0] Q  uav[1] K  uav[2] V
};

static const unsigned NT = 128;

__shared__ float row[NT];
__shared__ float red[NT];

// Write one row of `row` out. `byteBase` is the row's byte offset in `dst`.
//
// fp16 packs two halves per 4-byte Store, so lanes t < D/2 write one word each;
// fp32 has every lane write its own word. A ByteAddressBuffer.Store is 4 bytes,
// which is why this is not a plain loop over D either way.
__device__ inline void storeRow(unsigned char* dst, unsigned byteBase, unsigned t, unsigned D,
                                unsigned f32) {
    if (f32 != 0u) {
        if (t < D) *(unsigned*)(dst + byteBase + t * 4u) = __float_as_uint(row[t]);
    } else {
        const unsigned D2 = D / 2u;
        if (t < D2) {
            const unsigned lo = (unsigned)__half_as_ushort(__float2half_rn(row[2u * t])) & 0xFFFFu;
            const unsigned hi =
			    (unsigned)__half_as_ushort(__float2half_rn(row[2u * t + 1u])) & 0xFFFFu;
            *(unsigned*)(dst + byteBase + t * 4u) = lo | (hi << 16);
        }
    }
}

// [numthreads(NT, 1, 1)] - the host launches grid (S, heads): one block per
// (token, head) pair, 128 threads = head_dim.
extern "C" __global__ void qkv_prep(Args a) {
    const unsigned S = a.v[0], H = a.v[1], Hk = a.v[2], D = a.v[3];
    const unsigned qkvOff = a.v[4], qOff = a.v[5], kOff = a.v[6], vOff = a.v[7];
    const float eps = __uint_as_float(a.v[8]);
    const float scale = __uint_as_float(a.v[9]);
    const unsigned flags = a.v[10];
    // a.v[11] (idsOff) and a.v[12] (ropeOff) are deliberately not read: those two
    // tables carry their byte offset in their root descriptor (the host sets
    // srv_offset for them), exactly as in the HLSL. Adding both reads them from
    // twice their offset.
    const unsigned idsRow = a.v[13];
    const unsigned char* QKV = (const unsigned char*)a.s[0];
    const float* WQ = (const float*)a.s[1];
    const float* WK = (const float*)a.s[2];
    const float* ROPE = (const float*)a.s[3];
    const float* IDS = (const float*)a.s[4];
    unsigned char* Q = (unsigned char*)a.u[0];
    unsigned char* K = (unsigned char*)a.u[1];
    unsigned char* V = (unsigned char*)a.u[2];

    const unsigned s = blockIdx.x;          // token
    const unsigned h = blockIdx.y;          // head
    const unsigned t = threadIdx.x;
    if (s >= S) return;
    const unsigned qkvStride = (H + 2u * Hk) * D;
    const unsigned D2 = D / 2u;
    // bit2: store q/k/v as fp32. The reader (attn_flash) pays a mask+shift+f16tof32
    // for every fp16 element it touches, which is once per multiply-accumulate in
    // its inner loops; fp32 is a bare load. See dispatch_qkv_prep.
    const unsigned f32 = (flags >> 2u) & 1u;
    const unsigned esz = f32 != 0u ? 4u : 2u;

    // ── q ──
    row[t] = __uint_as_float(*(const unsigned*)(QKV + qkvOff + (s * qkvStride + h * D + t) * 4u));
    if ((flags & 1u) != 0u) {
        red[t] = row[t] * row[t];
        __syncthreads();
        for (unsigned st = NT / 2u; st > 0u; st >>= 1) {
            if (t < st) red[t] += red[t + st];
            __syncthreads();
        }
        const float inv = rsqrtf(red[0] / (float)D + eps);
        row[t] = row[t] * inv * WQ[t];
    }
    if ((flags & 2u) != 0u) {
        __syncthreads();
        if (t < D2) {
            const float omega = ROPE[t];
            const unsigned axis = (unsigned)ROPE[D2 + t];
            const float id = IDS[(idsRow + s) * 3u + axis];
            const float ang = id * omega;
            const float c = cosf(ang), sn = sinf(ang);
            const float x0 = row[2u * t], x1 = row[2u * t + 1u];
            red[2u * t] = x0 * c - x1 * sn;
            red[2u * t + 1u] = x1 * c + x0 * sn;
        }
        __syncthreads();
        for (unsigned i = 0; i < D / NT; i++) row[t + i * NT] = red[t + i * NT];
    }
    __syncthreads();
    for (unsigned i = 0; i < D / NT; i++) row[t + i * NT] *= scale;
    __syncthreads();
    storeRow(Q, qOff + (s * H + h) * D * esz, t, D, f32);

    if (h >= Hk) return;

    // ── k ──
    __syncthreads();
    row[t] = __uint_as_float(*(const unsigned*)(QKV + qkvOff +
                                               (s * qkvStride + (H + h) * D + t) * 4u));
    if ((flags & 1u) != 0u) {
        red[t] = row[t] * row[t];
        __syncthreads();
        for (unsigned st = NT / 2u; st > 0u; st >>= 1) {
            if (t < st) red[t] += red[t + st];
            __syncthreads();
        }
        const float inv = rsqrtf(red[0] / (float)D + eps);
        row[t] = row[t] * inv * WK[t];
    }
    if ((flags & 2u) != 0u) {
        __syncthreads();
        if (t < D2) {
            const float omega = ROPE[t];
            const unsigned axis = (unsigned)ROPE[D2 + t];
            const float id = IDS[(idsRow + s) * 3u + axis];
            const float ang = id * omega;
            const float c = cosf(ang), sn = sinf(ang);
            const float x0 = row[2u * t], x1 = row[2u * t + 1u];
            red[2u * t] = x0 * c - x1 * sn;
            red[2u * t + 1u] = x1 * c + x0 * sn;
        }
        __syncthreads();
        for (unsigned i2 = 0; i2 < D / NT; i2++) row[t + i2 * NT] = red[t + i2 * NT];
    }
    __syncthreads();
    storeRow(K, kOff + (s * Hk + h) * D * esz, t, D, f32);

    // ── v (no norm, no rope) ──
    __syncthreads();
    row[t] = __uint_as_float(*(const unsigned*)(QKV + qkvOff +
                                               (s * qkvStride + (H + Hk + h) * D + t) * 4u));
    __syncthreads();
    storeRow(V, vOff + (s * Hk + h) * D * esz, t, D, f32);
}
)CUDA";

const char* qkv_prep_hlsl() { return kQkvPrepHlsl; }

// ── DiT: causal flash attention ───────────────────────────────────────────
static const char* kAttnFlashHlsl = R"CUDA(
#include <cuda_fp16.h>   // __half2float / __ushort_as_half

// cbuffer P : register(b0) {
//     uint S, H, D;
//     uint qOff, kOff, vOff, oOff;
//     float scale;
//     uint seqStride;      // elements between consecutive tokens (H*D)
//     uint kvHeads;
//     uint kvStride;       // elements between consecutive tokens in k/v (Hk*D)
//     uint flags;          // bit0: q/k/v are fp32, bit1: causal
//     uint extra1, extra2;
// };
struct Args {
    unsigned v[24];     // 0 S  1 H  2 D  3 qOff  4 kOff  5 vOff  6 oOff  7 scale
                        // 8 seqStride  9 kvHeads  10 kvStride  11 flags
    const char* s[8];   // srv[0] Q  srv[1] K  srv[2] V   ([S,H,D] fp16 or fp32)
    char* u[4];         // uav[0] O    [S,H,D] fp32
};

static const unsigned BM = 32;   // query rows per block
static const unsigned BN = 32;   // keys per block
static const unsigned DGRP = 32; // output dims per thread
// (the old `static const uint NT = 128;` only existed to feed [numthreads(NT,1,1)];
//  the launch geometry now lives on the host, see the comment below.)

__shared__ float sP[BM][BN];
__shared__ float sMax[BM];
__shared__ float sSum[BM];
__shared__ float sCorr[BM];

// ByteAddressBuffer.Load is 4-byte aligned; pick the half out of the word.
__device__ inline float halfAt(const unsigned char* buf, unsigned byteOff) {
    const unsigned u = *(const unsigned*)(buf + (byteOff & ~3u));
    const unsigned sh = (byteOff & 2u) ? 16u : 0u;
    return __half2float(__ushort_as_half((unsigned short)((u >> sh) & 0xFFFFu)));
}

__device__ inline float elemAt(const unsigned char* buf, unsigned byteOff, bool isF32) {
    return isF32 ? __uint_as_float(*(const unsigned*)(buf + byteOff)) : halfAt(buf, byteOff);
}

// [numthreads(NT, 1, 1)] - the host launches grid (ceil(S/BM), heads).
extern "C" __global__ void attn_flash(Args a) {
    const unsigned S = a.v[0], H = a.v[1], D = a.v[2];
    const unsigned qOff = a.v[3], kOff = a.v[4], vOff = a.v[5], oOff = a.v[6];
    const float scale = __uint_as_float(a.v[7]);
    const unsigned seqStride = a.v[8], kvHeads = a.v[9], kvStride = a.v[10];
    const unsigned flags = a.v[11];
    const unsigned char* Q = (const unsigned char*)a.s[0];
    const unsigned char* K = (const unsigned char*)a.s[1];
    const unsigned char* V = (const unsigned char*)a.s[2];
    unsigned char* O = (unsigned char*)a.u[0];

    const bool isF32 = (flags & 1u) != 0u;
    // The text encoder is causal; the image DiT attends over the whole
    // sequence (cap tokens then image tokens, no mask).
    const bool causal = (flags & 2u) != 0u;
    const unsigned esz = isF32 ? 4u : 2u;
    const unsigned h = blockIdx.y;
    const unsigned qBase = blockIdx.x * BM;
    const unsigned t = threadIdx.x;
    const unsigned r = t / 4u;
    const unsigned g = t % 4u;
    const unsigned qIdx = qBase + r;
    // The grid rounds the query axis up to whole BM-row blocks, so the last block
    // can hold rows >= S (S=70 with BM=32, say). An out-of-range load clamps to
    // zero and an out-of-range store is dropped by a bounds-checking backend;
    // CUDA would fault, so such rows read Q from row 0 - the same guard
    // attn_tiled already uses - and are not stored.
    // Rows are independent in this kernel (own score row, own softmax row, own
    // output row), so no row < S changes by a single bit.
    const bool qOk = qIdx < S;

    float outAcc[DGRP];
    #pragma unroll
    for (unsigned i = 0; i < DGRP; i++) outAcc[i] = 0.0f;

    if (t < BM) {
        sMax[t] = -3.0e38f;
        sSum[t] = 0.0f;
    }
    __syncthreads();

    // GQA: several query heads share one key/value head
    const unsigned kvHead = (kvHeads == H) ? h : (h / (H / kvHeads));
    const unsigned qRowBase = qOff + ((qOk ? qIdx : 0u) * seqStride + h * D) * esz;
    // Every thread in the block must agree on how many k-blocks to run, so the
    // causal cut-off is derived from the *block's* last query row, not from
    // this thread's. Anything past it is masked out inside.
    const unsigned kTilesAll = (S + BN - 1) / BN;
    const unsigned kTiles = causal ? min(kTilesAll, (qBase + BM + BN - 1u) / BN) : kTilesAll;
    const unsigned kRowBase0 = kOff + kvHead * D * esz;
    const unsigned vRowBase0 = vOff + kvHead * D * esz;

    for (unsigned kt = 0; kt < kTiles; kt++) {
        const unsigned kBase = kt * BN;

        {
            float acc[8];
            #pragma unroll
            for (unsigned j = 0; j < 8; j++) acc[j] = 0.0f;
            const unsigned kRow = kBase + g * 8u;
            if ((!causal || kRow <= qIdx) && kRow < S) {
                for (unsigned d = 0; d < D; d++) {
                    const float qv = elemAt(Q, qRowBase + d * esz, isF32);
                    #pragma unroll
                    for (unsigned j = 0; j < 8; j++) {
                        const unsigned kj = kRow + j;
                        if ((!causal || kj <= qIdx) && kj < S) {
                            const float kv = elemAt(K, kRowBase0 + (kj * kvStride + d) * esz, isF32);
                            acc[j] += qv * kv;
                        }
                    }
                }
            }
            #pragma unroll
            for (unsigned j2 = 0; j2 < 8; j2++) {
                const unsigned kj = kBase + g * 8u + j2;
                sP[r][g * 8u + j2] = ((!causal || kj <= qIdx) && kj < S) ? acc[j2] * scale : -3.0e38f;
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
            float s = sSum[r] * corr;
            #pragma unroll
            for (unsigned j2 = 0; j2 < BN; j2++) {
                const float e = expf(sP[r][j2] - newmax);
                sP[r][j2] = e;
                s += e;
            }
            sMax[r] = newmax;
            sSum[r] = s;
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
                if (kj >= S || (causal && kj > qIdx)) continue;
                const float p = sP[r][j];
                if (p == 0.0f) continue;
                const unsigned vRow = vRowBase0 + (kj * kvStride + dBase) * esz;
                #pragma unroll
                for (unsigned i = 0; i < DGRP; i++)
                    outAcc[i] += p * elemAt(V, vRow + i * esz, isF32);
            }
        }
        __syncthreads();
    }

    const float inv = 1.0f / fmaxf(sSum[r], 1e-20f);
    const unsigned dBase = g * DGRP;
    if (qOk) {
        #pragma unroll
        for (unsigned i = 0; i < DGRP; i++)
            *(unsigned*)(O + oOff + (qIdx * seqStride + h * D + dBase + i) * 4u) =
                __float_as_uint(outAcc[i] * inv);
    }
}
)CUDA";

const char* attn_flash_hlsl() { return kAttnFlashHlsl; }


// ── DiT: tiled flash attention (fp16 q/k/v, D = 128) ──────────────────────
//
// Same maths as `attn_flash`, same parameter block, same grid, but both matmuls
// run on the tensor cores and - since the second revision - through raw
// `mma.sync` PTX rather than `wmma`.  The fp32-FMA kernel reached ~4 TFLOP/s at
// the DiT's shape; the wmma/fp32-accumulator one 19.5 TFLOP/s; this one is bound
// by the fp16-accumulate issue rate (measured 61 TFLOP/s register-only on this
// part, vs 30.5 for fp32 accumulate).
//
// What the earlier revisions established, and this one keeps:
//
//   * the score phase must not read K straight from global - a single load
//     instruction then touches eight key rows `kvStride` bytes apart, so every
//     query block re-streams all of K and the amplification, not the FLOPs,
//     decides the runtime.  K (and V) are staged in shared with a fully
//     coalesced read and read back from there.
//   * the streaming (online) softmax: per-row running max / running sum, with
//     the previous partial output rescaled by exp2(m_old - m_new) whenever the
//     max moves.  Numerically identical to the fp32 kernel's: the max over a
//     tile is taken before any exp, the sums use the *unrounded* exps, and
//     only the probability tile itself is rounded to fp16 (the weights are in
//     [0,1] and the output is a convex combination of V rows, so that costs
//     ~1.7e-4 relative - two orders of magnitude under the checkpoint's own
//     int8 error).
//   * the thread map is block-uniform: no early return anywhere, so every
//     __syncthreads() is reached by all four warps.  Rows past S are loaded
//     from row 0 (the qOk guard) and kept out of the output by the mask and
//     the store guard; keys past S are masked out.
//   * 4 warps x 16 query rows.  A warp owns its 16 rows outright, so its
//     online-softmax state never leaves its registers and no cross-warp
//     reduction or shared state is needed at all.
//
// What this revision changes:
//
//   * `wmma` is gone.  Every mma is `mma.sync.aligned.m16n8k16.row.col.f16.
//     f16.f16.f16` - fp16 accumulate, which is the 2x-rate mode - issued from an
//     explicit register tile.  The fragment layouts were verified numerically on
//     this part before the rewrite:
//     A(16x16) lane (g,c) = {a0:(g,2c)(g,2c+1), a1:(g+8,2c)(g+8,2c+1),
//     a2:(g,2c+8)(g,2c+9), a3:(g+8,2c+8)(g+8,2c+9)}, B(16x8) = {b0:(2c,g)(2c+1,g),
//     b1:(2c+8,g)(2c+9,g)}, D(16x8) = {d0:(g,2c)(g,2c+1), d1:(g+8,2c)(g+8,2c+1)}.
//   * the probability tile never reaches shared memory.  For m16n8k16 the four
//     A registers of k-step `kk` are *exactly* the packed halves of score
//     n-fragments 2kk and 2kk+1, so the exp phase writes them straight into the
//     A fragment and the P·V mma consumes them from registers.  The old kernel
//     stored P to a per-warp shared tile (stride 72) and reloaded it as an
//     operand, which cost a __syncwarp, a whole barrier-visible round trip and
//     the 9 KB sP buffer.
//   * Q is not staged either.  Each element is touched once per block, so the
//     stage bought nothing but 17 KB of footprint and one more barrier; read
//     straight from global the A fragments need 8 rows x 16 B per instruction,
//     which wastes half of each 32-byte sector, but over the whole dispatch the
//     doubled Q traffic is ~0.2% of the runtime (measured
//     below).  The 17 KB is what buys the third block per SM.
//   * the score phase's B operand (K^T) is loaded with `ldmatrix.x4` *without*
//     .trans: the natural distribution, lane (g,c) = {M[g][2c], M[g][2c+1]} of
//     an 8x8 tile, is already the K fragment, so one instruction delivers the
//     b0/b1 pair of two adjacent n-fragments - 32 instructions per warp per
//     tile instead of 128 scalar 4-byte shared loads.
//   * the softmax runs in log2.  `scale` is folded with log2(e) once and the
//     exponentials are `ex2.approx.f32`, so the per-element cost is one FMA and
//     one SFU op instead of a `expf` polynomial.  A constant factor on a row is
//     exactly what the softmax normalisation removes, so this is not an
//     approximation of the result.
//   * the fp16 S accumulator is read for the max (packed __hmax2 butterfly over
//     the four lanes sharing a row, then one convert per row) and again for the
//     exps; only the *rounded* scores cross into fp32, as before.
//   * the P·V B operand (V) is loaded with `ldmatrix.x4.trans`.  Staged
//     row-major (the coalesced global read), V[key][dim] has the key-major pair
//     the fragment wants split across two rows, which no single shared word
//     holds; .trans gets the right distribution while keeping every address on
//     a 16-byte aligned whole row.
//   * the per-tile P·V accumulates in fp16 (fresh accumulator each tile) and is
//     flushed into a *fp32* output register tile every tile, so the cross-tile
//     chain keeps the fp32 accumulator's precision.  Carrying the fp16
//     accumulator across all S/16 k-steps instead would accumulate
//     eps*sqrt(S/16) ~= 4.9e-4*16 = 7.8e-3 relative at S=4096 - over the 5e-3
//     budget - because the running partial sums grow with the running sum,
//     while a per-tile flush keeps them at the tile's own (small) scale.
//
// Measured at the DiT's shape (S=4096, 56 heads, D=128):
// 19.65 TFLOP/s for the wmma/fp32-accumulator revision, 33.4 TFLOP/s for this
// one (best observed 36.2 under a favourable clock; the SM clock on this laptop
// wanders between 1.4 and 2.0 GHz and moves the number by ~10%).  The register-
// only ceiling under the same conditions is 60 TFLOP/s, so the kernel is at 56%
// of it, and the ablations say where the rest went (medians, S=4096, 30 heads,
// in the same harness):
//
//   full kernel                            7.97 ms
//   - softmax exp/shuffle chain            7.88   (+1%)
//   - the fp32 output flush                7.6-7.7 (+4%)
//   - the K/V staging                      7.11   (-11%)   -> no staging at all
//   - staging AND flush together           6.23   (-22%)   -> the two interact
//   - also the ldmatrix operand loads      6.16            (41.9 TFLOP/s)
//
// i.e. the remaining gap is the serialised staging phase (4 barriers per key
// tile) and its interaction with the fp32 flush, not the mma issue.  Things that
// were tried and did not pay: a bigger key tile (BN=96/128 - the score/P
// registers push the block count down and it nets out worse), 4 blocks/SM via
// __launch_bounds__(NT,4) (128 registers + 120 B of local spill per thread,
// -25%), a cp.async double-buffered two-stage pipeline (needs a second 17 KB
// buffer, so 2 blocks/SM, and measured 3% *slower*), cp.async for the existing
// staging (no overlap, so neutral), hoisting the operand loads ahead of the mma
// and gating the rescale on cr != 1 (both neutral - the compiler already
// schedules those).
static const char* kAttnTiledHlsl = R"CUDA(
#include <cuda_fp16.h>   // __half, __half2, __hmax2, __floats2half2_rn

struct Args {
    unsigned v[24];     // 0 S  1 H  2 D  3 qOff  4 kOff  5 vOff  6 oOff  7 scale
                        // 8 seqStride  9 kvHeads  10 kvStride  11 flags
                        // 12/13 extra (unused here)  14 row0
    const char* s[8];   // srv[0] Q  srv[1] K  srv[2] V   ([S,H,D] packed fp16)
    char* u[4];         // uav[0] O    [S,H,D] fp32
};

// cbuffer P : register(b0) {
//     uint S, H, D;
//     uint qOff, kOff, vOff, oOff;
//     float scale;
//     uint seqStride;      // elements between consecutive tokens (H*D)
//     uint kvHeads;
//     uint kvStride;       // elements between consecutive tokens in k/v (Hk*D)
//     uint flags;          // bit0: (unused here) bit1: causal
//     uint extra1, extra2;
//     uint row0;           // first query row of this dispatch; the callers that
//                          // can run for a long time chunk the query axis so one
//                          // dispatch stays under the TDR watchdog (0 = whole)
// };

static const unsigned BM    = 64u;   // query rows per block; the host launches
                                     // grid (ceil(rows/BM), heads) with BM = 64
static const unsigned BN    = 64u;   // keys per K/V tile
static const unsigned HD    = 128u;  // head dim; the host rejects head_dim != 128
static const unsigned NT    = 128u;  // [numthreads(NT, 1, 1)] - 4 warps, which the
                                     // thread maps below require (warp -> 16 rows)
static const unsigned LDK   = HD + 8u;   // 136: staged K/V row stride (halfs).
                                     // 136 is a multiple of 8, so every (row,
                                     // 8-half column) address ldmatrix needs is
                                     // 16-byte aligned, and 136/2 = 68 words per
                                     // row with 68 % 32 = 4 puts the eight rows
                                     // one ldmatrix reads on banks 0,4,...,28 -
                                     // all 32 banks, i.e. conflict free.  128
                                     // would alias every row onto the same bank.
static const unsigned WROWS = 16u;       // query rows owned by one warp
static const unsigned NFRAG = BN / 8u;   // 8 score n-fragments  (m16n8k16)
static const unsigned KSTP  = HD / 16u;  // 8 score k-steps
static const unsigned PFRAG = HD / 8u;   // 16 output n-fragments
static const unsigned PSTP  = BN / 16u;  // 4 output k-steps

__shared__ __align__(16) __half sKV[BN * LDK];   // K tile, then V tile
// There is deliberately no staged Q tile.  Every Q element is touched exactly
// once per block, so staging it bought nothing but 17 KB of the footprint and a
// barrier - and the footprint is what decides how many blocks fit on an SM.  At
// 17 KB the kernel runs 3 blocks/SM instead of 2, which measured +15% at S=4096
// (the two other blocks' mma covers this one's staging and softmax stalls).
// The A fragments are read straight from global instead; the 8-rows x 16-bytes
// access pattern wastes half of each 32-byte sector, but the whole tensor is
// read once per block so even the doubled traffic is 0.2% of the runtime.

// D(16x8) += A(16x16) * B(16x8), all fp16, explicit registers.
__device__ __forceinline__ void mma16(unsigned& d0, unsigned& d1, unsigned a0, unsigned a1,
                                      unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d0), "+r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The same product with an *fp32* accumulator: D(16x8) += A(16x16) * B(16x8).
//
// The score matmul needs it. An fp16 accumulator saturates at 65504 and one
// row's QK^T is a sum of `head_dim` products of fp16 operands: at this model's
// activations (|q| ~ 165, |k| ~ 110 after the per-head norm and the axial rope)
// a single well-aligned pair of vectors passes 65504, the score becomes +inf,
// the row max becomes +inf, and `s * scale - max` is then `inf - inf` = NaN -
// so that (row, head) of the attention output is NaN. It is a per-(row, head)
// lottery, which is why a handful of heads in a chunk went bad and the rest of
// the block was fine.
//
// For a long time nothing downstream noticed: the DiT consumes the attention
// output through an *int8* GEMM, whose quantiser takes the row's amax (which a
// NaN cannot exceed, so the amax stays finite) and then rounds `x * inv`, and
// the C cast of a NaN lands on 0 - a silently wrong but finite value. A LoRA
// correction applied on the fp32 activation instead of folded into the weight
// is what surfaced it: the NaN propagates through the correction into the
// hidden stream, and 32 rows of the video come out as NaN.
__device__ __forceinline__ void mma16f32(float& d0, float& d1, float& d2, float& d3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Four 8x8 b16 tiles out of shared, transposed on the way into the registers.
__device__ __forceinline__ void ldsm4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                      const void* p) {
    const unsigned addr = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// Same, without .trans: lane (g,c) gets {M[g][2c], M[g][2c+1]} of each 8x8 tile.
// That is exactly the K B-fragment (b0 = {K[n][2c], K[n][2c+1]}), so the whole
// score-phase operand comes out four tiles at a time - one instruction per two
// n-fragments instead of four 4-byte shared loads.
__device__ __forceinline__ void ldsm4n(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                       const void* p) {
    const unsigned addr = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// Row max of one packed pair.  The butterfly below only balances the four lanes
// that share a row; it does NOT make the two halves of a lane's register equal:
// each lane of an m16n8 fragment holds columns 2c and 2c+1 of the row, so after
// hmax2 + lane^1/^2 the low half is the max over the even-offset columns of every
// 8-column key group (j*8 + 0,2,4,6) and the high half the max over the odd-offset
// ones (j*8 + 1,3,5,7).  Both halves are the *same row*, so the row max is the max
// of the two.
__device__ __forceinline__ float hmax2f(unsigned a) {
    const __half2 h = *(const __half2*)&a;
    return fmaxf(__low2float(h), __high2float(h));
}

__device__ __forceinline__ unsigned hmax2u(unsigned a, unsigned b) {
    const __half2 r = __hmax2(*(const __half2*)&a, *(const __half2*)&b);
    return *(const unsigned*)&r;
}

__device__ __forceinline__ unsigned packh2(float lo, float hi) {
    const __half2 r = __floats2half2_rn(lo, hi);
    return *(const unsigned*)&r;
}

// 2^x with the hardware approximate unit: one SFU op instead of the `expf`
// polynomial.  2^-22 relative over the whole argument range a softmax can
// produce (the arguments are <= 0 because the running max is subtracted).
__device__ __forceinline__ float ex2(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(r) : "f"(x));
    return r;
}

// One [nrows][HD] fp16 tile out of a global tensor whose rows are `stride`
// halfs apart; rows past S read row 0 so nothing reads outside the tensor (the
// caller's mask keeps those rows out of the result).  A16 selects the 16-byte
// path: it needs the tensor's byte offset to be 16-byte aligned, which the
// host's 256-byte allocations give but which this kernel does not assume.
// The lane map divides the work by the number of units per *row*, which is the
// one place head-dim 64 differs from 128: a row is `HD/8` uint4 (16 bytes each)
// on the aligned path and `HD/2` words on the unaligned one. Both divisors are
// compile-time powers of two, so `idx / WR` and `idx % WR` stay shifts.
static const unsigned WR16 = HD / 8u;   // uint4 per staged row (16-byte path)
static const unsigned WR32 = HD / 2u;   // words per staged row (4-byte path)

template <bool A16>
__device__ __forceinline__ void stageTile(__half* dst, const unsigned char* src, unsigned byteBase,
                                          unsigned stride, unsigned kBase, unsigned nrows,
                                          unsigned S, unsigned t) {
    if (A16) {
        #pragma unroll
        for (unsigned i = 0u; i < (nrows * WR16) / NT; i++) {
            const unsigned idx = t + NT * i;
            const unsigned row = idx / WR16;
            const unsigned w = idx % WR16;
            const unsigned key = kBase + row;
            const unsigned r = (key < S) ? key : 0u;
            *(uint4*)(dst + row * LDK + w * 8u) =
                *(const uint4*)(src + byteBase + (r * stride + w * 8u) * 2u);
        }
    } else {
        #pragma unroll
        for (unsigned i = 0u; i < (nrows * WR32) / NT; i++) {
            const unsigned idx = t + NT * i;
            const unsigned row = idx / WR32;
            const unsigned w = idx % WR32;
            const unsigned key = kBase + row;
            const unsigned r = (key < S) ? key : 0u;
            *(unsigned*)(dst + row * LDK + w * 2u) =
                *(const unsigned*)(src + byteBase + (r * stride + w * 2u) * 2u);
        }
    }
}

// [numthreads(NT, 1, 1)] - grid (ceil(S/BM), heads).  Warp `w` owns the query
// rows [BM-block base + 16w, +16) and all of its own softmax state; the warps
// share only the staged K/V tiles.
// __launch_bounds__(NT, 3): the shared footprint admits five blocks but 168
// registers/thread only admit three (65536/3/128 = 170).  Saying so keeps the
// allocator from spending the third block's headroom on nothing; asking for a
// fourth instead costs 120 bytes of local spill per thread and measured -25%.
extern "C" __global__ void __launch_bounds__(NT, 3) attn_tiled(Args a) {
    const unsigned S = a.v[0], H = a.v[1];
    const unsigned qOff = a.v[3], kOff = a.v[4], vOff = a.v[5], oOff = a.v[6];
    const float scale = __uint_as_float(a.v[7]);
    const unsigned seqStride = a.v[8], kvHeads = a.v[9], kvStride = a.v[10];
    const bool causal = (a.v[11] & 2u) != 0u;
    // bit2: a per-row exclusive key bound ([S] u32 in srv[3]) instead of the
    // plain causal rule. The Qwen-Image DiT's block-causal mask is exactly this:
    // a text run's row r attends keys [0, r+1] (so its bound is r+1) and every
    // row of an image block attends [0, end_of_block) (so its bound is a constant
    // over the block). One u32 per row instead of an [S, S] matrix.
    const bool useMask = (a.v[11] & 4u) != 0u;
    const unsigned row0 = a.v[14];
    const unsigned char* Q = (const unsigned char*)a.s[0];
    const unsigned char* K = (const unsigned char*)a.s[1];
    const unsigned char* V = (const unsigned char*)a.s[2];
    const unsigned* UB = (const unsigned*)a.s[3];
    unsigned char* O = (unsigned char*)a.u[0];

    const unsigned t = threadIdx.x;
    const unsigned lane = t & 31u, warp = t >> 5u;
    const unsigned g = lane >> 2u, c = lane & 3u;
    const unsigned h = blockIdx.y;
    const unsigned qBase = row0 + blockIdx.x * BM;
    const bool a16 = ((qOff | kOff | vOff) & 15u) == 0u;

    const unsigned kvHead = (kvHeads == H) ? h : (h / (H / kvHeads));
    const unsigned kTilesAll = (S + BN - 1u) / BN;
    const unsigned kTiles = causal ? min(kTilesAll, (qBase + BM + BN - 1u) / BN) : kTilesAll;

    const unsigned qByte = qOff + h * HD * 2u;
    const unsigned kByte = kOff + kvHead * HD * 2u;
    const unsigned vByte = vOff + kvHead * HD * 2u;

    // Q's eight A fragments, hoisted: they are reused by every key tile.  A row
    // past S reads row 0 - the same guard the staged path used, and the mask
    // below keeps those rows out of the result.  Nothing here depends on a16:
    // the two-halves-at-a-time load needs only 4-byte alignment, which an even
    // half index always has.
    unsigned qa[KSTP][4];
    {
        const unsigned rowA = qBase + warp * WROWS + g;
        const unsigned rowB = rowA + 8u;
        const unsigned char* qp0 = Q + qByte + (size_t)((rowA < S) ? rowA : 0u) * seqStride * 2u;
        const unsigned char* qp1 = Q + qByte + (size_t)((rowB < S) ? rowB : 0u) * seqStride * 2u;
        #pragma unroll
        for (unsigned k = 0u; k < KSTP; k++) {
            const unsigned off = (k * 16u + c * 2u) * 2u;
            qa[k][0] = *(const unsigned*)(qp0 + off);
            qa[k][1] = *(const unsigned*)(qp1 + off);
            qa[k][2] = *(const unsigned*)(qp0 + off + 16u);
            qa[k][3] = *(const unsigned*)(qp1 + off + 16u);
        }
    }

    // The output tile is fp32 across key tiles (see the header): 16 n-fragments
    // x 4 registers, rows g / g+8 in the low / high pair.
    float of[PFRAG][4];
    #pragma unroll
    for (unsigned n = 0u; n < PFRAG; n++)
        #pragma unroll
        for (unsigned r = 0u; r < 4u; r++) of[n][r] = 0.0f;

    float m0 = -3.0e38f, m1 = -3.0e38f;
    float s0 = 0.0f, s1 = 0.0f;
    const float scale2 = scale * 1.44269504088896340736f;   // log2(e)
    const unsigned qTop = qBase + warp * WROWS + g;

    for (unsigned kt = 0u; kt < kTiles; kt++) {
        const unsigned kBase = kt * BN;

        __syncthreads();
        if (a16) stageTile<true>(sKV, K, kByte, kvStride, kBase, BN, S, t);
        else stageTile<false>(sKV, K, kByte, kvStride, kBase, BN, S, t);
        __syncthreads();

        // S = Q K^T in fp16.  One ldmatrix.x4 covers the b0/b1 pair of two
        // adjacent n-fragments for one k-step: lane (g,c) of lanes 0-7 supplies
        // row j*8+g of the k-block at column k*16, lanes 8-15 the same rows at
        // column k*16+8, and lanes 16-31 repeat both for n-fragment j+1.
        // fp32 accumulators (see mma16f32): the score range is the one thing in
        // this kernel that fp16 cannot hold.
        float sa[NFRAG][4];
        #pragma unroll
        for (unsigned j = 0u; j < NFRAG; j++) {
            sa[j][0] = 0.0f;
            sa[j][1] = 0.0f;
            sa[j][2] = 0.0f;
            sa[j][3] = 0.0f;
        }
        const unsigned krow = (lane >> 4u) * 8u + (lane & 7u);
        const unsigned kcol = ((lane >> 3u) & 1u) * 8u;
        #pragma unroll
        for (unsigned k = 0u; k < KSTP; k++) {
            #pragma unroll
            for (unsigned jp = 0u; jp < NFRAG / 2u; jp++) {
                unsigned b0, b1, b2, b3;
                ldsm4n(b0, b1, b2, b3, sKV + (krow + jp * 16u) * LDK + k * 16u + kcol);
                mma16f32(sa[2u * jp][0], sa[2u * jp][1], sa[2u * jp][2], sa[2u * jp][3],
                         qa[k][0], qa[k][1], qa[k][2], qa[k][3], b0, b1);
                mma16f32(sa[2u * jp + 1u][0], sa[2u * jp + 1u][1], sa[2u * jp + 1u][2],
                         sa[2u * jp + 1u][3], qa[k][0], qa[k][1], qa[k][2], qa[k][3], b2, b3);
            }
        }

        // causal / ragged-edge mask, written as -inf in fp16 so the max and the
        // exp below need no special case for a masked key.  (`!causal` is part of
        // the test, not just `allIn`: a fully in-range tile still needs the mask
        // when the mask is causal.)
        const bool allIn = (qBase + BM <= S) && (kBase + BN <= S) && !causal && !useMask;
        if (!allIn) {
            const unsigned rowB = qTop + 8u;
            const unsigned ubTop = useMask ? ((qTop < S) ? UB[qTop] : 0u) : 0xffffffffu;
            const unsigned ubBot = useMask ? ((rowB < S) ? UB[rowB] : 0u) : 0xffffffffu;
            #pragma unroll
            for (unsigned j = 0u; j < NFRAG; j++) {
                const unsigned key = kBase + j * 8u + 2u * c;
                // The m16n8k16 fragment lays c0/c1 on row g and c2/c3 on row
                // g+8, which is exactly the (qTop, rowB) pairing below.
                if (!((qTop < S) && (key < S) &&
                      (useMask ? (key < ubTop) : (!causal || key <= qTop))))
                    sa[j][0] = -3.0e38f;
                if (!((qTop < S) && (key + 1u < S) &&
                      (useMask ? (key + 1u < ubTop) : (!causal || key + 1u <= qTop))))
                    sa[j][1] = -3.0e38f;
                if (!((rowB < S) && (key < S) &&
                      (useMask ? (key < ubBot) : (!causal || key <= rowB))))
                    sa[j][2] = -3.0e38f;
                if (!((rowB < S) && (key + 1u < S) &&
                      (useMask ? (key + 1u < ubBot) : (!causal || key + 1u <= rowB))))
                    sa[j][3] = -3.0e38f;
            }
        }

        // The tile max, in fp32. The four registers of a fragment are c0..c3,
        // so the low row is (c0, c1) and the high row (c2, c3) - and with the
        // scores in fp32 there is nothing to unpack.
        //
        // This used to be a packed `__hmax2` butterfly over fp16 fragments whose
        // first version read only the low half of each register, silently
        // dropping half of a row's keys from the max; whatever the true max was,
        // `s * scale2 - nm` then stopped being <= 0 and the extra exponent
        // overflowed the fp16 P*V accumulator. The fp32 accumulator removes the
        // range limit that made both of those fatal in the first place.
        float um0 = sa[0][0], um1 = sa[0][2];
        #pragma unroll
        for (unsigned j = 0u; j < NFRAG; j++) {
            um0 = fmaxf(um0, fmaxf(sa[j][0], sa[j][1]));
            um1 = fmaxf(um1, fmaxf(sa[j][2], sa[j][3]));
        }
        um0 = fmaxf(um0, __shfl_xor_sync(0xffffffffu, um0, 1));
        um0 = fmaxf(um0, __shfl_xor_sync(0xffffffffu, um0, 2));
        um1 = fmaxf(um1, __shfl_xor_sync(0xffffffffu, um1, 1));
        um1 = fmaxf(um1, __shfl_xor_sync(0xffffffffu, um1, 2));
        // floor the running max at -1e30: a fully masked row makes the tile max
        // -inf, and (-inf) - (-inf) below would be a NaN.
        const float nm0 = fmaxf(fmaxf(m0, um0 * scale2), -1.0e30f);
        const float nm1 = fmaxf(fmaxf(m1, um1 * scale2), -1.0e30f);
        const float cr0 = (m0 <= -3.0e37f) ? 0.0f : ex2(m0 - nm0);
        const float cr1 = (m1 <= -3.0e37f) ? 0.0f : ex2(m1 - nm1);
        m0 = nm0;
        m1 = nm1;

        // P = 2^(scale2*s - nm) in fp32, packed straight into the A fragments of
        // the P*V mma: k-step kk's four A registers are the packed halves of
        // score fragments 2kk (a0,a1) and 2kk+1 (a2,a3).
        unsigned pa[PSTP][4];
        float e0s = 0.0f, e1s = 0.0f;
        // `- log2bn` folds a constant 1/BN into every probability, which is what
        // keeps the fp16 P*V accumulator (below) in range: a tile's probabilities
        // then sum to at most 1 instead of at most BN, so the tile's contribution
        // is bounded by |V|max rather than by BN * |V|max, and 65504 stops being
        // a limit the weights can reach. The constant cancels in the final
        // `of / s` - both are scaled by it - so nothing about the result changes.
        const float log2bn = log2f((float)BN);
        #pragma unroll
        for (unsigned j = 0u; j < NFRAG; j++) {
            const float e00 = ex2(fmaf(sa[j][0], scale2, -nm0) - log2bn);
            const float e01 = ex2(fmaf(sa[j][1], scale2, -nm0) - log2bn);
            const float e10 = ex2(fmaf(sa[j][2], scale2, -nm1) - log2bn);
            const float e11 = ex2(fmaf(sa[j][3], scale2, -nm1) - log2bn);
            e0s += e00 + e01;
            e1s += e10 + e11;
            const unsigned kk = j >> 1u, sel = (j & 1u) * 2u;
            pa[kk][sel + 0u] = packh2(e00, e01);
            pa[kk][sel + 1u] = packh2(e10, e11);
        }
        e0s += __shfl_xor_sync(0xffffffffu, e0s, 1);
        e0s += __shfl_xor_sync(0xffffffffu, e0s, 2);
        e1s += __shfl_xor_sync(0xffffffffu, e1s, 1);
        e1s += __shfl_xor_sync(0xffffffffu, e1s, 2);
        s0 = s0 * cr0 + e0s;
        s1 = s1 * cr1 + e1s;

        __syncthreads();
        if (a16) stageTile<true>(sKV, V, vByte, kvStride, kBase, BN, S, t);
        else stageTile<false>(sKV, V, vByte, kvStride, kBase, BN, S, t);
        __syncthreads();

        #pragma unroll
        for (unsigned n = 0u; n < PFRAG; n++) {
            of[n][0] *= cr0;
            of[n][1] *= cr0;
            of[n][2] *= cr1;
            of[n][3] *= cr1;
        }

        // O += P V, fp16 accumulate within the tile.  The B operand comes out of
        // the staged, row-major V with ldmatrix.x4.trans: one instruction per two
        // n-fragments, and the four registers it drops are already the
        // {b0(n0),b1(n0),b0(n1),b1(n1)} quadruple the two mma calls want.
        unsigned oa[PFRAG][2];
        #pragma unroll
        for (unsigned n = 0u; n < PFRAG; n++) { oa[n][0] = 0u; oa[n][1] = 0u; }
        const unsigned l16 = lane & 15u;
        const unsigned vrow = (l16 & 7u) + ((l16 >> 3u) * 8u);
        const unsigned vcol = (lane >> 4u) * 8u;
        const __half* vb = sKV + vrow * LDK + vcol;
        #pragma unroll
        for (unsigned k = 0u; k < PSTP; k++) {
            #pragma unroll
            for (unsigned np = 0u; np < PFRAG / 2u; np++) {
                unsigned r0, r1, r2, r3;
                ldsm4(r0, r1, r2, r3, vb + k * 16u * LDK + np * 16u);
                mma16(oa[2u * np][0], oa[2u * np][1], pa[k][0], pa[k][1], pa[k][2], pa[k][3],
                      r0, r1);
                mma16(oa[2u * np + 1u][0], oa[2u * np + 1u][1], pa[k][0], pa[k][1], pa[k][2],
                      pa[k][3], r2, r3);
            }
        }

        #pragma unroll
        for (unsigned n = 0u; n < PFRAG; n++) {
            const float2 q0 = __half22float2(*(const __half2*)&oa[n][0]);
            const float2 q1 = __half22float2(*(const __half2*)&oa[n][1]);
            of[n][0] += q0.x;
            of[n][1] += q0.y;
            of[n][2] += q1.x;
            of[n][3] += q1.y;
        }
    }

    const float inv0 = 1.0f / fmaxf(s0, 1e-20f);
    const float inv1 = 1.0f / fmaxf(s1, 1e-20f);
    #pragma unroll
    for (unsigned n = 0u; n < PFRAG; n++)
        #pragma unroll
        for (unsigned r = 0u; r < 4u; r++) {
            const unsigned row = qTop + ((r & 2u) ? 8u : 0u);
            if (row < S) {
                const unsigned col = n * 8u + 2u * c + (r & 1u);
                const float v = of[n][r] * ((r & 2u) ? inv1 : inv0);
                *(unsigned*)(O + oOff + (row * seqStride + h * HD + col) * 4u) =
                    __float_as_uint(v);
            }
        }
}
)CUDA";

const char* attn_tiled_hlsl() { return kAttnTiledHlsl; }

// ── the query tile, as a source rewrite ───────────────────────────────────
//
// `BM` and `NT` are one number: one warp owns 16 query rows, so 4 warps is 64
// rows and 8 warps is 128. Everything else in the kernel - the fragment
// plumbing, the softmax butterfly, the V ldmatrix, the store guard - is already
// written against `warp * WROWS` and works unchanged. Rewriting the two
// constants (and the launch bounds, which move with NT: 4 warps of 168
// registers is three blocks per SM, 8 warps is one) is the same trick
// autotune.cpp plays on the int8 GEMM's tile, and it is cheaper to read than a
// template would be.
//
// Whether the bigger tile is *faster* is a measurement, not an argument, so the
// choice is a runtime one (PHI_ATTN_BM) and the two variants are compiled to
// distinct pipeline names - the cache keys on the name, not the source, so
// sharing one name would silently serve the first tile's cubin for both.
namespace {

void sub_uint_const(std::string& s, const char* name, unsigned value) {
	// `static const unsigned BM    = 64u;` - the alignment spaces are part of the
	// source, so the match has to skip whitespace rather than expect one space.
	const std::string pat = std::string("unsigned ") + name;
	size_t p = s.find(pat);
	if (p == std::string::npos)
		throw MediaError(std::string("attn_tiled: constant ") + name + " not found");
	size_t d = p + pat.size();
	while (d < s.size() && (s[d] == ' ' || s[d] == '\t')) d++;
	if (d >= s.size() || s[d] != '=')
		throw MediaError(std::string("attn_tiled: constant ") + name + " not assigned");
	d++;
	while (d < s.size() && (s[d] == ' ' || s[d] == '\t')) d++;
	size_t e = d;
	while (e < s.size() && std::isdigit((unsigned char)s[e])) e++;
	if (e == d) throw MediaError(std::string("attn_tiled: constant ") + name + " has no value");
	s.replace(d, e - d, std::to_string(value));
}

void sub_str(std::string& s, const std::string& from, const std::string& to) {
	size_t p = 0;
	while ((p = s.find(from, p)) != std::string::npos) {
		s.replace(p, from.size(), to);
		p += to.size();
	}
}

}  // namespace

std::string attn_tiled_name(unsigned bm, unsigned hd) {
	std::string n = "attn_tiled";
	if (bm != 64u) n += "_bm" + std::to_string(bm);
	if (hd != 128u) n += "_hd" + std::to_string(hd);
	if (const char* e = getenv("PHI_ATTN_BN")) n += "_bn" + std::string(e);
	if (const char* e = getenv("PHI_ATTN_BLOCKS")) n += "_b" + std::string(e);
	return n;
}

const char* attn_tiled_src(unsigned bm, unsigned hd) {
	// 16 rows per warp; anything else is not a tile this kernel can take.
	if (bm < 16u || (bm % 16u) != 0u) throw MediaError("attn_tiled: bad query tile");
	// The head dim must be a whole number of 16-half steps (one mma k-step) and
	// leave `LDK = hd + 8` 16-byte aligned, which every multiple of 8 does.
	if (hd < 16u || (hd % 16u) != 0u) throw MediaError("attn_tiled: bad head dim");
	// The cache key must move with *every* input that changes the source,
	// including the two experiment knobs: `ComputeContext::pipeline` returns the
	// first kernel whose name matches and never looks at the source again, so a
	// name that ignored PHI_ATTN_BN would serve the first tile's cubin for both.
	static std::map<std::string, std::string> cache;
	const std::string key = attn_tiled_name(bm, hd);
	auto it = cache.find(key);
	if (it != cache.end()) return it->second.c_str();
	std::string s = kAttnTiledHlsl;
	const unsigned nt = 32u * (bm / 16u);
	if (hd != 128u) sub_uint_const(s, "HD", hd);
	// PHI_ATTN_BN / PHI_ATTN_BLOCKS: the two knobs that decide the kernel's
	// shape on a given part, exposed so they can be swept on the real workload
	// instead of argued about. BN is the key tile (elements per staged K/V row
	// block); NFRAG and PSTP are written as `BN / 8` and `BN / 16`, so rewriting
	// the one constant moves the whole fragment map with it. The second launch
	// bound is the occupancy target, which trades registers against resident
	// blocks: the kernel wants 168 registers, and 65536/(128*168) is exactly 3.
	if (const char* e = getenv("PHI_ATTN_BN")) {
		const long v = strtol(e, nullptr, 10);
		if (v == 16 || v == 32 || v == 64 || v == 128) sub_uint_const(s, "BN", (unsigned)v);
	}
	unsigned min_blocks = 0;
	if (const char* e = getenv("PHI_ATTN_BLOCKS")) {
		const long v = strtol(e, nullptr, 10);
		if (v >= 1 && v <= 6) min_blocks = (unsigned)v;
	}
	if (bm != 64u || min_blocks) {
		if (bm != 64u) {
			sub_uint_const(s, "BM", bm);
			sub_uint_const(s, "NT", nt);
		}
		// Registers are the constraint, not shared memory: at ~168 registers a
		// thread the register file holds 65536/(nt*168) blocks, and asking for
		// more costs local spill (measured -25% when the 4-warp kernel asked for
		// a fourth block).
		unsigned blocks = min_blocks ? min_blocks : 65536u / (nt * 168u);
		if (blocks < 1u) blocks = 1u;
		if (blocks > 6u) blocks = 6u;
		sub_str(s, "__global__ void __launch_bounds__(NT, 3) attn_tiled(",
		        "__global__ void __launch_bounds__(NT, " + std::to_string(blocks) +
		            ") attn_tiled(");
	}
	auto ins = cache.emplace(key, std::move(s));
	return ins.first->second.c_str();
}


unsigned attn_query_tile() {
	static unsigned v = [] {
		if (const char* e = getenv("PHI_ATTN_BM")) {
			const long n = strtol(e, nullptr, 10);
			if (n == 64 || n == 128 || n == 256) return (unsigned)n;
		}
		// 64 (the tile the kernel was tuned at) is the default, and that is a
		// measurement rather than a preference: at 540P/5s (S=19303, 56 heads) the
		// 128-row tile halves the K/V traffic but drops the block from three per
		// SM to one - 8 warps instead of 12 - and the DiT's attention went from
		// 16.6 s to 23.3 s per step. The traffic is not what that kernel is short
		// of; latency hiding is. The switch is kept because the trade is shape-
		// dependent (a shorter sequence has less K/V to re-read and more to gain
		// from the wider tile).
		return 64u;
	}();
	return v;
}

}  // namespace phi::media
