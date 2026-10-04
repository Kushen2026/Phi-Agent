// CUDA for the MiniMax H3 **video VAE** — the pieces the M1 kernel set does not
// have:
//
//   * `conv3d_causal`  causal 3-D convolution (the encoder's building block).
//   * `causal_gn`      GroupNorm with per-frame statistics ([C,T,H,W] layout).
//   * `qkv_split3d`    q/k/v split of the decoder's fused qkv projection with
//                      the per-head RMSNorm folded in.
//   * `rope3d`         the decoder's 3-D RoPE (folded in below from
//                      kernels/rope3d.hlsl; the standalone file is no longer used).
//   * `attn3d`         non-causal flash attention with head_dim 64. The DiT's
//                      `attn_flash` is hard-wired to head_dim 128 (it assigns
//                      4 x 32 output dims per block), so the VAE needs its own.
//   * `unpatch3d`      decoder token stream -> [3, T*4, H*16, W*16] pixels.
//
// Reference: comfy/ldm/minimax/vae.py. The host-side dispatchers live in
// video_vae.cpp; the constant-buffer layouts below are mirrored there.
//
// Precision: weights/biases stay fp16 (they are fp16 in the checkpoint and are
// upcast exactly), activations are fp32, accumulation is fp32 — the reference
// runs the VAE in fp32 with fp16 parameters cast on entry, so upcasting the
// fp16 operands reproduces it bit-for-bit at the operand level.
//
// ── HLSL -> CUDA ──────────────────────────────────────────────────────────
// Each entry below is now an NVRTC-compiled `extern "C" __global__` taking the
// frozen `Args` block (the old root constants in `v[]`, byte offsets included,
// plus the srv/uav bases in `s[]`/`u[]`). The maths is a literal translation:
//   * `[numthreads(N,1,1)]` is dropped (recorded in a comment); the host launches
//     with the same block shape.
//   * `ByteAddressBuffer.Load/Load4` -> a typed pointer load at `base + offset`;
//     `RWByteAddressBuffer.Store` -> `*(T*)(base + offset) = ...`.
//   * `GroupMemoryBarrierWithGroupSync()` -> `__syncthreads()`.
//   * `f16tof32(h)` -> `__half2float(__ushort_as_half(h))`.
//   * `asfloat/asuint` -> `__uint_as_float/__float_as_uint`.
//   * HLSL `uint` is not a CUDA type, so the leaf constants are `unsigned` here
//     (they are not text-substituted in this file; only int8_gemm.cpp is).
//
// The `rope3d` helpers used to live in the standalone kernels/rope3d.hlsl that
// this file `#include`d as two macros. That file is no longer touched: its
// content is folded in below as two CUDA raw strings, so `rope3d_helpers_hlsl()`
// (the shared rotation helper) and `rope3d_hlsl()` (helper + the standalone
// `rope3d` entry) keep their exact signatures and concatenation order.
#include "models/video_vae_internal.hpp"

#include <string>

namespace phi::media {

// ── causal 3-D convolution ────────────────────────────────────────────────
//
// CausalConv3d (vae.py:41) pads the two *spatial* axes with `reflect` and the
// temporal axis with zeros on the front only, then runs a plain conv3d with no
// padding. A single-frame input is special-cased there: the reference truncates
// the weight to the last temporal tap (`autopad="causal_zero"`, ops.py:611)
// instead of convolving against a zero frame. That is the same number as
// `padT = kt - 1` here (the only tap whose index lands inside the frame is the
// last one, and the kernel zeroes negative time), so no special case is needed.
//
// Folding the reference's own `F.pad(reflect)` into the gather makes one kernel
// cover every case: `padHL`/`padHR` reflect-pad H and W on the left/top and
// right/bottom respectively, and `padT` zero-pads the *front* of T.
//
// The two spatial pads are separate on purpose. CausalConv3d pads both sides by
// `(kernel-1)/2`, while Downsample3D (vae.py:87) reflect-pads by one on the
// right/bottom *only* before calling a CausalConv3d whose own padding is zero.
// Expressing that as a symmetric `padH = 1` shifts the stride-2 tap window one
// pixel relative to the reference, which is a silent numerical error (the output
// shape still comes out right). Until this was split, the H3 video VAE's
// *encoder* disagreed with comfy/ldm/minimax/vae.py at `cos 0.14` on its very
// first convolution - and nothing noticed, because the decoder is a transformer
// and every existing VAE test exercised the decoder.
//
// Weight is [OC, IC, KT, KH, KW] fp16 (contiguous = [OC, K], K = IC*KT*KH*KW),
// x is [IC, IT, IH, IW] fp32, y is [OC, OT, OH, OW] fp32.
//
// cbuffer (a.v[]):
//   0 IC  1 IT  2 IH  3 IW
//   4 OC  5 OT  6 OH  7 OW
//   8 xOff 9 wOff 10 bOff 11 yOff
//  12 kdim   = KT | KH<<4 | KW<<8
//  13 stride = strT | strS<<4
//  14 pads   = padT | padHL<<4 | padHR<<8
//  15 flags  = bit0 has_bias
//
// srv: s[0]=X s[1]=W s[2]=Bias   uav: u[0]=Y
static const char* kConv3dCausalHlsl = R"CUDA(
#include <cuda_fp16.h>   // __half, f16tof32 -> __half2float/__ushort_as_half

struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

// old [numthreads(NT, 1, 1)] with NT = 256
static const unsigned BM = 64;
static const unsigned BN = 64;
static const unsigned BK = 32;
static const unsigned TM = 4;
static const unsigned TN = 4;
static const unsigned NT = 256;

__device__ __forceinline__ float convBiasAt(const char* buf, unsigned byteOff) {
    unsigned u = *(const unsigned*)(buf + (byteOff & ~3u));
    unsigned sh = (byteOff & 2u) ? 16u : 0u;
    return __half2float(__ushort_as_half((unsigned short)((u >> sh) & 0xFFFFu)));
}

// One weight element, in the checkpoint's own storage dtype. 0 = fp32, 1 = fp16,
// 2 = bf16 (see the `wDtype` note in the kernel). fp16 has 10 significand bits and
// bf16 has 8, both far fewer than the f32 the accumulator runs in, so widening
// here is exact whatever the checkpoint held.
__device__ __forceinline__ float ldWeight(const char* W, unsigned wOff, unsigned idx,
                                         unsigned wDtype) {
    if (wDtype == 0u) return *(const float*)(W + wOff + idx * 4u);
    const unsigned short h = *(const unsigned short*)(W + wOff + idx * 2u);
    return (wDtype == 2u) ? __uint_as_float((unsigned)h << 16) : __half2float(__ushort_as_half(h));
}

extern "C" __global__ void conv3d_causal(Args a) {
    const unsigned IC = a.v[0], IT = a.v[1], IH = a.v[2], IW = a.v[3];
    const unsigned OC = a.v[4], OT = a.v[5], OH = a.v[6], OW = a.v[7];
    const unsigned xOff = a.v[8], wOff = a.v[9], bOff = a.v[10], yOff = a.v[11];
    const unsigned kdim = a.v[12], stride = a.v[13], pads = a.v[14], flags = a.v[15];
    // The weight tensor's own storage dtype: 0 = fp32, 1 = fp16, 2 = bf16. The
    // checkpoint decides it (`load_conv_weight` uploads the file's precision
    // verbatim); the kernel converts each operand as it loads it, so the only
    // thing that changes is how many bytes one element is and how it is decoded.
    // fp16 has 10 significand bits and bf16 8, both fewer than the f32 the
    // accumulator runs in, so no decode here can round a value that arrived
    // exact.
    const unsigned wDtype = a.v[16];

    const char* X = a.s[0];
    const char* W = a.s[1];
    const char* Bias = a.s[2];
    char* Y = a.u[0];

    const unsigned KT = kdim & 15u;
    const unsigned KH = (kdim >> 4) & 15u;
    const unsigned KW = (kdim >> 8) & 15u;
    const unsigned strT = stride & 15u;
    const unsigned strS = (stride >> 4) & 15u;
    // pads: 0-3 = zero padding prepended on the time axis (causal),
    //       4-7 = reflect padding on the left/top,
    //       8-11 = reflect padding on the right/bottom.
    //
    // Left and right are separate because the two callers differ:
    // CausalConv3d reflect-pads H/W symmetrically, while Downsample3D pads the
    // right/bottom only. A single shared "padH" made the stride-2 downsample read
    // one pixel off (symmetric-by-1 shifts the tap window by one relative to
    // right-only-by-1), and a stale `ktBase` made the one-frame path read out of
    // range entirely.
    const unsigned padT = pads & 15u;
    const unsigned padHL = (pads >> 4) & 15u;
    const unsigned padHR = (pads >> 8) & 15u;
    const bool hasBias = (flags & 1u) != 0u;

    const unsigned tidx = threadIdx.x;
    const unsigned tx = tidx % (BN / TN);
    const unsigned ty = tidx / (BN / TN);
    const unsigned ocBase = blockIdx.x * BM;
    const unsigned pBase = blockIdx.y * BN;
    const unsigned ot = blockIdx.z;

    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    float acc[TM][TN];
    #pragma unroll
    for (unsigned i = 0; i < TM; i++)
        #pragma unroll
        for (unsigned j = 0; j < TN; j++) acc[i][j] = 0.0f;

    const unsigned kHW = KH * KW;
    const unsigned kTKW = KT * kHW;
    const unsigned K = IC * kTKW;
    const unsigned spatial = OH * OW;
    const unsigned kTiles = (K + BK - 1) / BK;

    for (unsigned kt = 0; kt < kTiles; kt++) {
        const unsigned kBase = kt * BK;

        // weights, two per lane, in the checkpoint's own storage dtype.
        #pragma unroll
        for (unsigned s = 0; s < (BM * BK / 2) / NT; s++) {
            const unsigned li = tidx + s * NT;
            const unsigned r = li / (BK / 2);
            const unsigned c2 = li % (BK / 2);
            const unsigned oc = ocBase + r;
            const unsigned k = kBase + c2 * 2;
            float lo = 0.0f, hi = 0.0f;
            if (oc < OC && k < K) {
                // One element at a time. `k` is even but K is not always: the
                // encoder's first conv has 3 input channels, so K = IC*KT*KH*KW is
                // odd and `(oc*K + k)` is odd for odd `oc`, which a vector load
                // cannot express.
                lo = ldWeight(W, wOff, oc * K + k, wDtype);
                if (k + 1 < K)   // also keeps the pair load inside the tensor
                    hi = ldWeight(W, wOff, oc * K + k + 1, wDtype);
            }
            sA[r][c2 * 2] = lo;
            sA[r][c2 * 2 + 1] = (k + 1 < K) ? hi : 0.0f;
        }

        // activations, gathered with the reflect/zero padding folded in
        #pragma unroll
        for (unsigned s2 = 0; s2 < (BK * BN) / NT; s2++) {
            const unsigned li = tidx + s2 * NT;
            const unsigned kk = li / BN;
            const unsigned pp = li % BN;
            const unsigned k = kBase + kk;
            float v = 0.0f;
            if (k < K && (pBase + pp) < spatial) {
                const unsigned ic = k / kTKW;
                const unsigned r0 = k - ic * kTKW;
                const unsigned ktl = r0 / kHW;
                const unsigned r1 = r0 - ktl * kHW;
                const unsigned kh = r1 / KW;
                const unsigned kw = r1 - kh * KW;
                const unsigned p = pBase + pp;
                const unsigned oh = p / OW;
                const unsigned ow = p - oh * OW;
                const int it = (int)(ot * strT + ktl) - (int)padT;
                int ih = (int)(oh * strS + kh) - (int)padHL;
                int iw = (int)(ow * strS + kw) - (int)padHL;
                if (padHL > 0u) {
                    if (ih < 0) ih = -ih;
                    if (iw < 0) iw = -iw;
                }
                if (padHR > 0u) {
                    if (ih >= (int)IH) ih = 2 * (int)IH - 2 - ih;
                    if (iw >= (int)IW) iw = 2 * (int)IW - 2 - iw;
                }
                if (it >= 0 && it < (int)IT && ih >= 0 && ih < (int)IH && iw >= 0 && iw < (int)IW)
                    v = *(const float*)(X + xOff + (((ic * IT + (unsigned)it) * IH + (unsigned)ih) * IW + (unsigned)iw) * 4u);
            }
            sB[kk][pp] = v;
        }
        __syncthreads();

        #pragma unroll
        for (unsigned kk2 = 0; kk2 < BK; kk2++) {
            float av[TM], bv[TN];
            #pragma unroll
            for (unsigned i = 0; i < TM; i++) av[i] = sA[ty * TM + i][kk2];
            #pragma unroll
            for (unsigned j = 0; j < TN; j++) bv[j] = sB[kk2][tx * TN + j];
            #pragma unroll
            for (unsigned i2 = 0; i2 < TM; i2++)
                #pragma unroll
                for (unsigned j2 = 0; j2 < TN; j2++) acc[i2][j2] += av[i2] * bv[j2];
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned i3 = 0; i3 < TM; i3++) {
        const unsigned oc = ocBase + ty * TM + i3;
        if (oc >= OC) continue;
        float bv = hasBias ? convBiasAt(Bias, bOff + oc * 2u) : 0.0f;
        #pragma unroll
        for (unsigned j3 = 0; j3 < TN; j3++) {
            const unsigned p = pBase + tx * TN + j3;
            if (p >= spatial) continue;
            *(unsigned*)(Y + yOff + ((oc * OT + ot) * spatial + p) * 4u) =
                __float_as_uint(acc[i3][j3] + bv);
        }
    }
}
)CUDA";

const char* conv3d_causal_hlsl() { return kConv3dCausalHlsl; }

// ── per-frame GroupNorm ───────────────────────────────────────────────────
//
// TemporalIsolatedGroupNorm (vae.py:59): the statistics are computed per frame,
// i.e. the (C/G, H, W) slab of one time step. Layout here is [C, T, H, W] (the
// engine's batch is folded into C for the VAE), so the element of channel c and
// spatial index hw at time t lives at (c*T + t)*HW + hw.
//
// cbuffer (a.v[]): 0 C 1 T 2 HW 3 G 4 cg 5 xOff 6 wOff 7 bOff 8 yOff 9 eps 10 affine
// srv: s[0]=X s[1]=Wg s[2]=Bg   uav: u[0]=Y
//
// Reduction note: the HLSL reduces over the whole thread group with a shared
// `red[NT]` tree (`gnReduce`, `red[t] += red[t+s]` for s = 128,64,...,1), *not*
// with wave intrinsics. It spans the full 256-thread block (8 warps), so it is
// kept as an explicit shared-memory tree here, barrier-for-barrier identical to
// the HLSL — a warp-shuffle rewrite would change the addition order and the
// result's last bits.
static const char* kCausalGnHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

// old [numthreads(NT, 1, 1)] with NT = 256
static const unsigned NT = 256;

extern "C" __global__ void causal_gn(Args a) {
    const unsigned C = a.v[0], T = a.v[1], HW = a.v[2], G = a.v[3], cg = a.v[4];
    const unsigned xOff = a.v[5], wOff = a.v[6], bOff = a.v[7], yOff = a.v[8];
    const float eps = __uint_as_float(a.v[9]);
    const unsigned affine = a.v[10];

    const char* X = a.s[0];
    const char* Wg = a.s[1];
    const char* Bg = a.s[2];
    char* Y = a.u[0];

    __shared__ float red[NT];

    const unsigned t = threadIdx.x;
    const unsigned tt = blockIdx.x / G;
    const unsigned g = blockIdx.x % G;
    const unsigned count = cg * HW;

    float sum = 0.0f;
    for (unsigned i = t; i < count; i += NT) {
        const unsigned c = g * cg + i / HW;
        sum += *(const float*)(X + xOff + ((c * T + tt) * HW + (i % HW)) * 4u);
    }
    red[t] = sum;
    __syncthreads();
    for (unsigned s = NT / 2u; s > 0u; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    const float total = red[0];
    const float mean = total / (float)count;

    float v2 = 0.0f;
    for (unsigned i2 = t; i2 < count; i2 += NT) {
        const unsigned c = g * cg + i2 / HW;
        float d = *(const float*)(X + xOff + ((c * T + tt) * HW + (i2 % HW)) * 4u) - mean;
        v2 += d * d;
    }
    red[t] = v2;
    __syncthreads();
    for (unsigned s = NT / 2u; s > 0u; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    const float total2 = red[0];
    const float inv = rsqrtf(total2 / (float)count + eps);

    for (unsigned i3 = t; i3 < count; i3 += NT) {
        const unsigned ch = g * cg + i3 / HW;
        float v = (*(const float*)(X + xOff + ((ch * T + tt) * HW + (i3 % HW)) * 4u) - mean) * inv;
        if (affine != 0u) {
            v *= *(const float*)(Wg + wOff + ch * 4u);
            v += *(const float*)(Bg + bOff + ch * 4u);
        }
        *(unsigned*)(Y + yOff + ((ch * T + tt) * HW + (i3 % HW)) * 4u) = __float_as_uint(v);
    }
}
)CUDA";

const char* causal_gn_hlsl() { return kCausalGnHlsl; }

// ── q/k/v split + per-head RMSNorm ────────────────────────────────────────
//
// Attention.forward (vae.py:225) views the fused projection as
// [S, heads, 3*dim_head] and chunks the last axis, so head h of token s owns
// qkv[s][h*192 .. h*192+191] = [q(64) | k(64) | v(64)]. q and k then get an
// RMSNorm *without* affine (elementwise_affine=False -> weight is None) at
// eps 1e-5, applied per head. v is passed through.
//
// One block per (token, head), 64 threads, one per head dim.
//
// cbuffer (a.v[]): 0 S 1 H 2 D 3 qkvOff 4 qOff 5 kOff 6 vOff 7 eps
// srv: s[0]=QKV   uav: u[0]=Q u[1]=K u[2]=V
static const char* kQkvSplit3dHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

// old [numthreads(NT, 1, 1)] with NT = 64
static const unsigned NT = 64;

extern "C" __global__ void qkv_split3d(Args a) {
    const unsigned S = a.v[0], H = a.v[1], D = a.v[2];
    const unsigned qkvOff = a.v[3], qOff = a.v[4], kOff = a.v[5], vOff = a.v[6];
    const float eps = __uint_as_float(a.v[7]);

    const char* QKV = a.s[0];
    char* Q = a.u[0];
    char* K = a.u[1];
    char* V = a.u[2];

    __shared__ float red[64];

    const unsigned s = blockIdx.x;
    const unsigned h = blockIdx.y;
    const unsigned t = threadIdx.x;
    // No early return: a thread-group sync is not allowed inside varying flow
    // control ("thread sync operation must be in non-varying flow control"), and
    // the two RMS reductions below are full-block tree reductions. The dispatch
    // grid is exactly (S, H) so `inside` is always true in practice, but the
    // guard has to be a predicate rather than a branch either way. (The only
    // return is after every barrier, so it can never strand a block at a sync.)
    const bool inside = (s < S) && (t < D);
    const unsigned rowBytes = qkvOff + (s * H + h) * (3u * D) * 4u;
    const float q = inside ? *(const float*)(QKV + rowBytes + t * 4u) : 0.0f;
    const float k = inside ? *(const float*)(QKV + rowBytes + (D + t) * 4u) : 0.0f;
    const float v = inside ? *(const float*)(QKV + rowBytes + (2u * D + t) * 4u) : 0.0f;

    red[t] = q * q;
    __syncthreads();
    for (unsigned s2 = NT / 2u; s2 > 0u; s2 >>= 1) {
        if (t < s2) red[t] += red[t + s2];
        __syncthreads();
    }
    const float tq = red[0];
    const float invq = rsqrtf(tq / (float)D + eps);

    __syncthreads();
    red[t] = k * k;
    __syncthreads();
    for (unsigned s2 = NT / 2u; s2 > 0u; s2 >>= 1) {
        if (t < s2) red[t] += red[t + s2];
        __syncthreads();
    }
    const float tk = red[0];
    const float invk = rsqrtf(tk / (float)D + eps);

    if (!inside) return;
    const unsigned outBytes = (s * H + h) * D * 4u;
    *(unsigned*)(Q + qOff + outBytes + t * 4u) = __float_as_uint(q * invq);
    *(unsigned*)(K + kOff + outBytes + t * 4u) = __float_as_uint(k * invk);
    *(unsigned*)(V + vOff + outBytes + t * 4u) = __float_as_uint(v);
}
)CUDA";

const char* qkv_split3d_hlsl() { return kQkvSplit3dHlsl; }

// The 3-D RoPE used to live in its own file (kernels/rope3d.hlsl), included as
// two HLSL raw-string macros. It is folded in here as CUDA raw strings so the
// standalone file is no longer needed; the helpers are shared with the
// standalone shader below so both rotate identically.
static const char* kRope3dHelpers = R"CUDA(
// Shared helper: the angle of pair `i` of token `s`.
__device__ __forceinline__ float phi_rope3d_angle(const char* IDS, const char* INV, unsigned idsOff,
                                                  unsigned invOff, unsigned s, unsigned i,
                                                  unsigned n_axis, unsigned freqs_per_axis) {
    unsigned axis = i / freqs_per_axis;      // 0 = t, 1 = h, 2 = w
    if (axis >= n_axis) axis = n_axis - 1u;
    unsigned f = i % freqs_per_axis;
    float id = __uint_as_float(*(const unsigned*)(IDS + idsOff + (s * n_axis + axis) * 4u));
    float inv = __uint_as_float(*(const unsigned*)(INV + invOff + f * 4u));
    return 6.2831853071795864f * id * inv;
}
)CUDA";

const char* rope3d_helpers_hlsl() { return kRope3dHelpers; }

// The standalone shader needs the shared angle helper; the two live in separate
// raw strings (a helper has no entry point, a shader has no place for one), so
// they are concatenated once here. Without this the compile fails with
// "the call to phi_rope3d_angle is undeclared" — which is what the video VAE's
// first real decode on this machine reported.
static const char* kRope3dMain = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

// old [numthreads(NT, 1, 1)] with NT = 64
static const unsigned NT = 64;

extern "C" __global__ void rope3d(Args a) {
    // cbuffer (a.v[]): 0 rows 1 heads 2 D 3 xOff 4 idsOff 5 invOff
    //                  6 nPair 7 nAxis 8 freqsPerAxis
    // srv: s[0]=IDS s[1]=INV   uav: u[0]=X
    const unsigned rows = a.v[0], heads = a.v[1], D = a.v[2], xOff = a.v[3];
    const unsigned idsOff = a.v[4], invOff = a.v[5];
    const unsigned nPair = a.v[6], nAxis = a.v[7], freqsPerAxis = a.v[8];

    char* X = a.u[0];
    const char* IDS = a.s[0];
    const char* INV = a.s[1];

    __shared__ float xs[64];

    const unsigned row = blockIdx.x;
    const unsigned t = threadIdx.x;
    // Predicates, not early returns: a thread-group sync is not allowed inside
    // varying flow control, and the whole block takes part in the barrier below
    // even when its row is out of range. (The dispatch grid already makes both
    // conditions false, but a guard that cannot compile is not a guard.)
    const bool inside = (row < rows) && (t < D);
    const unsigned rows_ = row < rows ? row : 0u;
    const unsigned s = rows_ / heads;
    const unsigned base = xOff + rows_ * D * 4u;

    xs[t] = inside ? *(const float*)(X + base + t * 4u) : 0.0f;
    __syncthreads();

    if (inside && t < nPair) {
        float ang = phi_rope3d_angle(IDS, INV, idsOff, invOff, s, t, nAxis, freqsPerAxis);
        float c = cosf(ang), sn = sinf(ang);
        float x0 = xs[t];
        float x1 = xs[t + nPair];
        *(unsigned*)(X + base + t * 4u) = __float_as_uint(x0 * c - x1 * sn);
        *(unsigned*)(X + base + (t + nPair) * 4u) = __float_as_uint(x1 * c + x0 * sn);
    }
}
)CUDA";

const char* rope3d_hlsl() {
	static const std::string combined = std::string(kRope3dHelpers) + "\n" + kRope3dMain;
	return combined.c_str();
}

// ── non-causal flash attention, head_dim 64 ───────────────────────────────
//
// 32 heads x 64 dims over the whole token stream (no mask: the ViT3D decoder
// attends bidirectionally). Same online-softmax structure as the DiT's
// `attn_flash`; the only differences are the 4x16 output split (head_dim 64
// instead of 128) and the absence of causal masking / GQA.
//
// cbuffer (a.v[]): 0 S 1 H 2 D 3 qOff 4 kOff 5 vOff 6 oOff 7 scale 8 seqStride
// srv: s[0]=Q s[1]=K s[2]=V   uav: u[0]=O
//
// Reduction note: this kernel does not use wave intrinsics. Each query row r is
// softmaxed by its own group of 4 threads (g = 0 carries it) through the shared
// `sP`/`sMax`/`sSum`/`sCorr` arrays, synchronised by full-block barriers. That
// structure is translated literally.
static const char* kAttn3dHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

// old [numthreads(NT, 1, 1)] with NT = 128
static const unsigned BM = 32;
static const unsigned BN = 32;
static const unsigned NT = 128;
static const unsigned DGRP = 16;

extern "C" __global__ void attn3d(Args a) {
    const unsigned S = a.v[0], H = a.v[1], D = a.v[2];
    const unsigned qOff = a.v[3], kOff = a.v[4], vOff = a.v[5], oOff = a.v[6];
    const float scale = __uint_as_float(a.v[7]);
    const unsigned seqStride = a.v[8];      // elements between consecutive tokens (H*D)

    const char* Q = a.s[0];
    const char* K = a.s[1];
    const char* V = a.s[2];
    char* O = a.u[0];

    __shared__ float sP[BM][BN];
    __shared__ float sMax[BM];
    __shared__ float sSum[BM];
    __shared__ float sCorr[BM];

    const unsigned h = blockIdx.y;
    const unsigned qBase = blockIdx.x * BM;
    const unsigned t = threadIdx.x;
    const unsigned r = t / 4u;
    const unsigned g = t % 4u;
    const unsigned qIdx = qBase + r;
    // The grid rounds the query axis up to whole BM-row blocks, so the last
    // block can hold rows >= S (S = 70 with BM = 32, say). Such rows read Q
    // from row 0 - the same guard attn_flash and attn3d_v2 already carry -
    // and the store at the bottom drops them (its own `if (qIdx < S)` test).
    // Reading them straight made the load address land past the end of Q,
    // which is a fault on CUDA (unlike the HLSL original, whose out-of-range
    // reads were clamped).
    const bool qOk = qIdx < S;

    float outAcc[DGRP];
    #pragma unroll
    for (unsigned i = 0; i < DGRP; i++) outAcc[i] = 0.0f;

    if (t < BM) {
        sMax[t] = -3.0e38f;
        sSum[t] = 0.0f;
    }
    __syncthreads();

    const unsigned qRowBytes = qOff + ((qOk ? qIdx : 0u) * seqStride + h * D) * 4u;
    const unsigned kRowBase = kOff + h * D * 4u;
    const unsigned vRowBase = vOff + h * D * 4u;
    const unsigned kTiles = (S + BN - 1) / BN;

    for (unsigned kt = 0; kt < kTiles; kt++) {
        const unsigned kBase = kt * BN;

        {
            float acc[8];
            #pragma unroll
            for (unsigned j = 0; j < 8; j++) acc[j] = 0.0f;
            const unsigned kRow = kBase + g * 8u;
            if (kRow < S) {
                for (unsigned d = 0; d < D; d++) {
                    float qv = *(const float*)(Q + qRowBytes + d * 4u);
                    #pragma unroll
                    for (unsigned j2 = 0; j2 < 8; j2++) {
                        unsigned kj = kRow + j2;
                        if (kj < S) {
                            float kv = *(const float*)(K + kRowBase + (kj * seqStride + d) * 4u);
                            acc[j2] += qv * kv;
                        }
                    }
                }
            }
            #pragma unroll
            for (unsigned j3 = 0; j3 < 8; j3++) {
                unsigned kj = kBase + g * 8u + j3;
                sP[r][g * 8u + j3] = (kj < S) ? acc[j3] * scale : -3.0e38f;
            }
        }
        __syncthreads();

        if (g == 0u) {
            float bmax = -3.0e38f;
            #pragma unroll
            for (unsigned j = 0; j < BN; j++) bmax = fmaxf(bmax, sP[r][j]);
            float oldmax = sMax[r];
            float newmax = fmaxf(oldmax, bmax);
            float corr = (oldmax <= -3.0e37f) ? 0.0f : expf(oldmax - newmax);
            float s = sSum[r] * corr;
            #pragma unroll
            for (unsigned j2 = 0; j2 < BN; j2++) {
                float e = expf(sP[r][j2] - newmax);
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
                unsigned kj = kBase + j;
                if (kj >= S) continue;
                float p = sP[r][j];
                if (p == 0.0f) continue;
                const unsigned vRow = vRowBase + (kj * seqStride + dBase) * 4u;
                #pragma unroll
                for (unsigned i = 0; i < DGRP; i++)
                    outAcc[i] += p * *(const float*)(V + vRow + i * 4u);
            }
        }
        __syncthreads();
    }

    if (qIdx < S) {
        const float inv = 1.0f / fmaxf(sSum[r], 1e-20f);
        const unsigned dBase = g * DGRP;
        #pragma unroll
        for (unsigned i = 0; i < DGRP; i++)
            *(unsigned*)(O + oOff + (qIdx * seqStride + h * D + dBase + i) * 4u) =
                __float_as_uint(outAcc[i] * inv);
    }
}
)CUDA";

const char* attn3d_hlsl() { return kAttn3dHlsl; }

// ── the same attention, staged and register-blocked (attn3d_v2) ───────────
//
// `attn3d` above is correct and was the decoder's only attention kernel, but it
// is written as if global memory were free: for every (query row, key) pair it
// reads the whole 64-wide head from *global* with scalar loads, and the 32
// threads of a warp land on 32 different rows (`seqStride` floats apart), so
// every load instruction touches 32 cache lines to use 128 bytes of them. At a
// 256 px tile that measured 0.87 TFLOP/s - 13% of this card's fp32 FMA peak,
// and 45% of the decoder block's GPU time once the GEMMs moved to tensor cores.
//
// v2 keeps the maths *bit for bit* (same per-(row,key) accumulation order over
// d, same per-(row,dim) accumulation order over keys, same online softmax with
// the same guard) and fixes the memory behaviour instead:
//
//   * K is staged into shared with coalesced 16-byte loads and read back as
//     `float4`, so the tile is fetched once per block instead of once per row.
//   * Q and V are read straight from global but *coalesced*: Q's 16 dims per
//     instruction come from 16 distinct rows and are served out of L1 after the
//     first k-tile, and V's 64 dims are one contiguous 256-byte run per key, so
//     a warp's lane 0..15 (all four dim groups) covers it exactly.
//   * each thread owns a 4x4 block of the score matrix and a 4x4 block of the
//     output, so the inner loop is 16 FMAs per 8 operand words instead of 1 FMA
//     per operand word.
//
// BM = BN = 64 with 256 threads: tr = t & 15 owns rows tr*4.., tc = t >> 4 owns
// keys (scores) / dims (output) tc*4... Only K is staged, which keeps the
// footprint at 34 KB - two blocks per SM - where staging Q and V as well would
// have pushed it past 48 KB and cost the second block.
//
// cbuffer (a.v[]): 0 S 1 H 2 D 3 qOff 4 kOff 5 vOff 6 oOff 7 scale 8 seqStride
// srv: s[0]=Q s[1]=K s[2]=V   uav: u[0]=O
// ── the same attention, staged and register-blocked (attn3d_v2) ───────────
//
// `attn3d` above is correct and was the decoder's only attention kernel, but it
// is written as if global memory were free: for every (query row, key) pair it
// reads the whole 64-wide head from *global* with scalar loads, and the 32
// threads of a warp land on 32 different rows (`seqStride` floats apart), so
// every load instruction touches 32 cache lines to use 128 bytes of them.
//
// v2 keeps the maths *bit for bit* - same per-(row,key) accumulation order over
// d, same per-(row,dim) order over keys, same online softmax with the same
// guard - and fixes the memory behaviour:
//
//   * K is staged into shared with coalesced 16-byte loads, so a key tile is
//     fetched once per block instead of once per query row;
//   * Q and V are read from global but *coalesced*: V's 16 dims per instruction
//     are one contiguous run and Q's are served out of L1 (a block's Q tile is
//     16 KB and is re-read by every k-tile);
//   * each thread owns a 4x4 block of the score matrix and a 4x4 block of the
//     output, so the inner loop is 16 FMAs per 8 operand words.
//
// The two phases deliberately use different thread maps, which is what keeps
// the shared traffic affordable:
//
//   scores: row = tr + 16*i, key = tc + 16*j   (both warp spans *consecutive*
//           rows/keys, so the sK reads are a broadcast and the sP writes are
//           conflict-free at stride 68)
//   output: row = tr + 16*i, dim = 4*tc + j    (so the 4 probabilities one
//           thread needs for 4 consecutive keys are one `float4`, and the V
//           operand is one `float4` per key)
//
// With rows 4 apart in both phases the 16 lanes of a half-warp would land on
// two banks each (stride 68 words is 4 words per row modulo 32) - a 4-way
// conflict on every sP access, which is exactly what made the first version of
// this kernel only 1.3x faster than the one it replaces.
//
// Only K is staged: staging Q as well would take the footprint past 48 KB and
// cost the second block per SM.
//
// cbuffer (a.v[]): 0 S 1 H 2 D 3 qOff 4 kOff 5 vOff 6 oOff 7 scale 8 seqStride
// srv: s[0]=Q s[1]=K s[2]=V   uav: u[0]=O
static const char* kAttn3dV2Cuda = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

typedef unsigned int uint;

static const uint BM = 64;    // query rows per block
static const uint BN = 64;    // keys per tile
static const uint D  = 64;    // head dim (the host rejects anything else)
static const uint NT = 256;   // 8 warps
static const uint SP = 68;    // shared row stride, floats (64 + 4 of padding)
static const uint TQ = 4;     // rows, and keys/dims, per thread

// numthreads(256, 1, 1)
extern "C" __global__ void attn3d_v2(Args a) {
    const uint S = a.v[0], H = a.v[1];
    const uint qOff = a.v[3], kOff = a.v[4], vOff = a.v[5], oOff = a.v[6];
    const float scale = __uint_as_float(a.v[7]);
    const uint seqStride = a.v[8];      // elements between consecutive tokens (H*D)

    const char* Q = a.s[0];
    const char* K = a.s[1];
    const char* V = a.s[2];
    char* O = a.u[0];

    __shared__ __align__(16) float sK[BN * SP];
    __shared__ __align__(16) float sP[BM * SP];
    __shared__ float sMax[BM];
    __shared__ float sSum[BM];
    __shared__ float sCorr[BM];

    const uint t = threadIdx.x;
    const uint tr = t & 15u;    // row lane: rows tr, tr+16, tr+32, tr+48
    const uint tc = t >> 4u;    // key/dim lane
    const uint h = blockIdx.y;
    const uint qBase = blockIdx.x * BM;

    const uint qHead = qOff + h * D * 4u;
    const uint kHead = kOff + h * D * 4u;
    const uint vHead = vOff + h * D * 4u;

    if (t < BM) {
        sMax[t] = -3.0e38f;
        sSum[t] = 0.0f;
    }

    float outAcc[TQ][TQ];
    #pragma unroll
    for (uint i = 0; i < TQ; i++)
        #pragma unroll
        for (uint j = 0; j < TQ; j++) outAcc[i][j] = 0.0f;

    const uint kTiles = (S + BN - 1u) / BN;

    for (uint kt = 0; kt < kTiles; kt++) {
        const uint kBase = kt * BN;

        // ── stage K[BN, D], 16 bytes at a time ─────────────────────────────
        #pragma unroll
        for (uint st = 0; st < (BN * D / 4u) / NT; st++) {
            const uint idx = t + NT * st;
            const uint row = idx >> 4u, c4 = idx & 15u;
            const uint key = kBase + row;
            const float4 kv = (key < S)
                                  ? *(const float4*)(K + kHead + (key * seqStride + c4 * 4u) * 4u)
                                  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            *(float4*)(&sK[row * SP + c4 * 4u]) = kv;
        }
        __syncthreads();

        // ── scores: 4 rows x 4 keys per thread, d in the original's order ──
        float acc[TQ][TQ];
        #pragma unroll
        for (uint i = 0; i < TQ; i++)
            #pragma unroll
            for (uint j = 0; j < TQ; j++) acc[i][j] = 0.0f;

        #pragma unroll
        for (uint d4 = 0; d4 < D / 4u; d4++) {
            float4 qv[TQ], kv[TQ];
            #pragma unroll
            for (uint i = 0; i < TQ; i++) {
                const uint row = qBase + tr + 16u * i;
                const uint rr = (row < S) ? row : 0u;
                qv[i] = *(const float4*)(Q + qHead + (rr * seqStride + d4 * 4u) * 4u);
            }
            #pragma unroll
            for (uint j = 0; j < TQ; j++)
                kv[j] = *(const float4*)(&sK[(tc + 16u * j) * SP + d4 * 4u]);
            #pragma unroll
            for (uint i = 0; i < TQ; i++)
                #pragma unroll
                for (uint j = 0; j < TQ; j++) {
                    acc[i][j] += qv[i].x * kv[j].x;
                    acc[i][j] += qv[i].y * kv[j].y;
                    acc[i][j] += qv[i].z * kv[j].z;
                    acc[i][j] += qv[i].w * kv[j].w;
                }
        }

        #pragma unroll
        for (uint i = 0; i < TQ; i++) {
            const uint row = qBase + tr + 16u * i;
            #pragma unroll
            for (uint j = 0; j < TQ; j++) {
                const uint key = kBase + tc + 16u * j;
                sP[(tr + 16u * i) * SP + tc + 16u * j] =
                    (key < S && row < S) ? acc[i][j] * scale : -3.0e38f;
            }
        }
        __syncthreads();

        // ── softmax, one row per thread (the original's structure) ─────────
        if (t < BM) {
            float bmax = -3.0e38f;
            #pragma unroll 4
            for (uint j = 0; j < BN; j++) bmax = fmaxf(bmax, sP[t * SP + j]);
            const float oldmax = sMax[t];
            const float newmax = fmaxf(oldmax, bmax);
            const float corr = (oldmax <= -3.0e37f) ? 0.0f : expf(oldmax - newmax);
            float sum = sSum[t] * corr;
            #pragma unroll 4
            for (uint j = 0; j < BN; j++) {
                const float e = expf(sP[t * SP + j] - newmax);
                sP[t * SP + j] = e;
                sum += e;
            }
            sMax[t] = newmax;
            sSum[t] = sum;
            sCorr[t] = corr;
        }
        __syncthreads();

        // ── output: rescale this thread's rows, accumulate over keys ───────
        #pragma unroll
        for (uint i = 0; i < TQ; i++) {
            const float c = sCorr[tr + 16u * i];
            #pragma unroll
            for (uint j = 0; j < TQ; j++) outAcc[i][j] *= c;
        }

        #pragma unroll 2
        for (uint kg = 0; kg < BN / 4u; kg++) {
            float4 pv[TQ];
            #pragma unroll
            for (uint i = 0; i < TQ; i++)
                pv[i] = *(const float4*)(&sP[(tr + 16u * i) * SP + kg * 4u]);
            #pragma unroll
            for (uint jj = 0; jj < 4u; jj++) {
                // A key past S has probability exactly 0 (the score was -inf
                // and the row has at least one valid key), so the row only has
                // to be clamped to keep the load in bounds.
                const uint key = kBase + kg * 4u + jj;
                const uint kk = (key < S) ? key : 0u;
                const float4 vv = *(const float4*)(V + vHead + (kk * seqStride + tc * 4u) * 4u);
                #pragma unroll
                for (uint i = 0; i < TQ; i++) {
                    const float4 pi = pv[i];
                    const float p = (jj == 0u) ? pi.x : (jj == 1u) ? pi.y : (jj == 2u) ? pi.z : pi.w;
                    outAcc[i][0] += p * vv.x;
                    outAcc[i][1] += p * vv.y;
                    outAcc[i][2] += p * vv.z;
                    outAcc[i][3] += p * vv.w;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint i = 0; i < TQ; i++) {
        const uint row = qBase + tr + 16u * i;
        if (row >= S) continue;
        const float inv = 1.0f / fmaxf(sSum[tr + 16u * i], 1e-20f);
        const float4 ov = make_float4(outAcc[i][0] * inv, outAcc[i][1] * inv, outAcc[i][2] * inv,
                                      outAcc[i][3] * inv);
        *(float4*)(O + oOff + (row * seqStride + h * D + tc * 4u) * 4u) = ov;
    }
}
)CUDA";

const char* attn3d_v2_hlsl() { return kAttn3dV2Cuda; }



// ── decoder token stream -> pixels ────────────────────────────────────────
//
// ViT3DDecoder.forward (vae.py:304): the token stream is cut back to the real
// patches, viewed as [.., out_channels, pt, p, p] and permuted so the output is
// [3, T*4, H*16, W*16]. Token index = t*(H*W) + h*W + w and the projection's
// 3072 outputs are [oc, pt, ph, pw].
//
// cbuffer (a.v[]): 0 C 1 T 2 H 3 W 4 P 5 PT 6 xOff 7 yOff 8 total
// srv: s[0]=X   uav: u[0]=Y
static const char* kUnpatch3dHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

extern "C" __global__ void unpatch3d(Args a) {
    // old [numthreads(256, 1, 1)]
    const unsigned C = a.v[0], T = a.v[1], H = a.v[2], W = a.v[3], P = a.v[4], PT = a.v[5];
    const unsigned xOff = a.v[6], yOff = a.v[7], total = a.v[8];

    const char* X = a.s[0];
    char* Y = a.u[0];

    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;   // SV_DispatchThreadID
    if (i >= total) return;
    const unsigned OW = W * P;
    const unsigned OH = H * P;
    const unsigned OT = T * PT;
    const unsigned ow = i % OW;
    const unsigned oh = (i / OW) % OH;
    const unsigned ot = (i / (OW * OH)) % OT;
    const unsigned oc = i / (OW * OH * OT);

    const unsigned token = (ot / PT) * (H * W) + (oh / P) * W + (ow / P);
    const unsigned k = oc * (PT * P * P) + (ot % PT) * (P * P) + (oh % P) * P + (ow % P);
    *(unsigned*)(Y + yOff + i * 4u) =
        __float_as_uint(*(const float*)(X + xOff + (token * (C * PT * P * P) + k) * 4u));
}
)CUDA";

const char* unpatch3d_hlsl() { return kUnpatch3dHlsl; }

// --- fused-gate activation: out = silu(gx[row][col]) * gx[row][cols + col] ---
//
// FeedForward.w1 emits the gate and the value side by side (vae.py:208:
// `gate, x = self.w1(x).chunk(2, dim=-1)`), and the decoder runs it on a
// fused [rows, 2*cols] buffer.
//
// The shared `elem` op cannot express this: it reads each operand *flat* from
// that operand's base offset, so one dispatch would pair a[i] with b[cols + i]
// -- right for row 0, wrong for every other row (the first reference-matched
// decode: w1 matched ComfyUI at cos 1.000000, the gated result at 0.05).
// Re-basing the offset per row block does not work either, because A needs
// re-basing as much as B does. A row stride is what the pairing actually
// needs, so the pairing gets its own kernel.
//
// cbuffer (a.v[]): 0 rows 1 cols 2 gxOff 3 outOff 4 total
// srv: s[0]=GX   uav: u[0]=OUT
static const char* kSiluGateFusedHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

extern "C" __global__ void silu_gate_fused(Args a) {
    // old [numthreads(256, 1, 1)]
    const unsigned rows = a.v[0], cols = a.v[1];
    const unsigned gxOff = a.v[2], outOff = a.v[3], total = a.v[4];

    const char* GX = a.s[0];
    char* OUT = a.u[0];

    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;   // SV_DispatchThreadID
    if (i >= total) return;
    const unsigned row = i / cols;
    const unsigned col = i % cols;
    const unsigned aOff = gxOff + (row * 2u * cols + col) * 4u;
    const float ga = *(const float*)(GX + aOff);
    const float gb = *(const float*)(GX + aOff + cols * 4u);
    *(unsigned*)(OUT + outOff + i * 4u) = __float_as_uint((ga / (1.0f + expf(-ga))) * gb);
}
)CUDA";

const char* silu_gate_fused_hlsl() { return kSiluGateFusedHlsl; }

}  // namespace phi::media
