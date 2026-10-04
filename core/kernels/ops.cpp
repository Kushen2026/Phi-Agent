// CUDA sources for the M1 image path: fp16 GEMM, row/group normalisation,
// elementwise algebra, 2-D convolution, attention and the layout kernels.
//
// Ported from HLSL to CUDA C++. The host-side contract is frozen: every entry
// point keeps its HLSL name (hence `extern "C"`), the old root constants arrive
// verbatim in `Args::v[24]` (byte offsets stay there and the kernel adds them),
// and the resources arrive in `s[0..7]` / `u[0..3]` in the same binding order
// the HLSL used. The old `[numthreads(x, y, z)]` group size is kept as a
// `numthreads(...)` comment, because CUDA has no such attribute and the host
// parses it back out to size the launch (see compute.cpp's BlockShapeParser).
//
// Every kernel documents the host-side reference it must agree with, because
// running the same maths on the CPU is the only correctness instrument the
// engine has.
//
// Layout conventions used throughout:
//   * row-major, contiguous, fp32 unless the name says otherwise
//   * a "row" is the last dimension
//   * weight matrices are stored [out, in] (safetensors/HF convention), so a
//     linear layer is C[m,n] = sum_k x[m,k] * w[n,k]
#include "kernels/kernels.hpp"

namespace phi::media {

// ── elementwise algebra ───────────────────────────────────────────────────
//
// One kernel, an opcode and three optional operands; each operand can be a
// flat same-shaped buffer, a row vector [cols], a column vector [rows] or a
// scalar. That covers every fusion in the DiT and the AE without needing a
// kernel per expression.
static const char* kOpsHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

// ByteAddressBuffer.Load<uint> / .Store equivalents: the buffers are byte
// addressed and every offset below is a byte offset that stays in a.v[].
__device__ __forceinline__ float ldF(const char* buf, unsigned byteOff) {
    return __uint_as_float(*(const unsigned*)(buf + byteOff));
}
__device__ __forceinline__ void stF(char* buf, unsigned byteOff, float v) {
    *(unsigned*)(buf + byteOff) = __float_as_uint(v);
}

__device__ __forceinline__ float loadMod(const char* buf, unsigned off, unsigned i, unsigned mode, float scalar, unsigned cols) {
    if (mode == 0u) return ldF(buf, off + i * 4);
    if (mode == 1u) return ldF(buf, off + (i % cols) * 4);
    if (mode == 2u) return ldF(buf, off + (i / cols) * 4);
    return scalar;
}

// opcodes
#define OP_ADD        0u   // y = a + b
#define OP_MUL        1u   // y = a * b
#define OP_SILU_GATE  2u   // y = silu(a) * b
#define OP_TANH       3u   // y = tanh(a)
#define OP_SILU       4u   // y = silu(a)
#define OP_SCALE      5u   // y = a * alpha
#define OP_ADD_MUL    6u   // y = a + b * c
#define OP_COPY       7u   // y = a
#define OP_MODULATE   8u   // y = a * (1 + b)
#define OP_EXP        9u   // y = exp(a)
#define OP_ADD_SCALE 10u   // y = a + b * alpha
#define OP_GELU_TANH 11u   // y = gelu(a), tanh approximation
#define OP_MOD_ADD   12u   // y = a * (1 + b) + c

// The op table, shared by the scalar and the vector kernel so the two cannot
// drift apart on an opcode.
__device__ __forceinline__ float applyElem(unsigned op, float av, float bv, float cv,
                                           float alpha) {
    if (op == OP_ADD)            return av + bv;
    if (op == OP_MUL)            return av * bv;
    if (op == OP_SILU_GATE)      return (av / (1.0f + expf(-av))) * bv;
    if (op == OP_TANH)           return tanhf(av);
    if (op == OP_SILU)           return av / (1.0f + expf(-av));
    if (op == OP_SCALE)          return av * alpha;
    if (op == OP_ADD_MUL)        return av + bv * cv;
    if (op == OP_COPY)           return av;
    if (op == OP_MODULATE)       return av * (1.0f + bv);
    if (op == OP_MOD_ADD)        return av * (1.0f + bv) + cv;
    if (op == OP_GELU_TANH) {
        // 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))) - torch's
        // F.gelu(..., approximate="tanh"), which is what Qwen-Image's text
        // projection uses.
        const float t = 0.7978845608028654f * (av + 0.044715f * av * av * av);
        return 0.5f * av * (1.0f + tanhf(t));
    }
    if (op == OP_EXP)            return expf(av);
    return av + bv * alpha;
}

// numthreads(256, 1, 1)
extern "C" __global__ void elem(Args a) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned N = a.v[0];
    if (i >= N) return;
    const unsigned cols = a.v[2];
    const unsigned op = a.v[3];
    // alpha/beta were the cbuffer floats at byte offsets 40/44; the host memcpy'd
    // the raw bits into values[10]/values[11], so reinterpret them back.
    const float alpha = __uint_as_float(a.v[10]);
    const float beta  = __uint_as_float(a.v[11]);
    const char* A = a.s[0];
    const char* B = a.s[1];
    const char* C = a.s[2];
    char* Y = a.u[0];

    float av = ldF(A, a.v[4] + i * 4);
    float bv = loadMod(B, a.v[5], i, a.v[8], alpha, cols);
    float cv = loadMod(C, a.v[6], i, a.v[9], beta, cols);
    stF(Y, a.v[7] + i * 4, applyElem(op, av, bv, cv, alpha));
}

// The same, four elements per thread. The scalar form is one 4-byte load and one
// 4-byte store per thread, so a 400 MB tensor is 100 M threads and the kernel is
// bound by how many of those loads a warp scheduler can keep in flight, not by
// bandwidth: measured ~25 GB/s of the ~300 this part can do, and the DiT spends
// 3.4 s of a 40 s sampling step in these ops (the modulates, the gated adds, the
// residual adds and the silu gates around its four GEMMs).
//
// Four elements per thread turns every access into one 16-byte one, which is what
// the memory system wants, and it is *bit-identical* to the scalar kernel: the
// ops are per-element functions, so nothing is reassociated. The vector form is
// only dispatched when N and both byte offsets are whole multiples of the access
// size (see dispatch_elem), and the operand modes are still resolved per element,
// so a row/column-vector operand behaves exactly as it did.
// numthreads(256, 1, 1)
extern "C" __global__ void elem4(Args a) {
    const unsigned N = a.v[0];
    const unsigned i = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (i >= N) return;
    const unsigned cols = a.v[2];
    const unsigned op = a.v[3];
    const float alpha = __uint_as_float(a.v[10]);
    const float beta  = __uint_as_float(a.v[11]);
    const char* A = a.s[0];
    const char* B = a.s[1];
    const char* C = a.s[2];
    char* Y = a.u[0];

    const float4 av = *(const float4*)(A + a.v[4] + i * 4u);
    const float a4[4] = {av.x, av.y, av.z, av.w};
    float y4[4];
    #pragma unroll
    for (unsigned k = 0u; k < 4u; k++) {
        const unsigned j = i + k;
        const float bv = loadMod(B, a.v[5], j, a.v[8], alpha, cols);
        const float cv = loadMod(C, a.v[6], j, a.v[9], beta, cols);
        y4[k] = applyElem(op, a4[k], bv, cv, alpha);
    }
    float4 out;
    out.x = y4[0]; out.y = y4[1]; out.z = y4[2]; out.w = y4[3];
    *(float4*)(Y + a.v[7] + i * 4u) = out;
}
)CUDA";

const char* ops_hlsl() { return kOpsHlsl; }

// ── row-wise normalisation ────────────────────────────────────────────────
//
// RMSNorm:   y[r,:] = x[r,:] * rsqrt(mean(x^2) + eps) * w
// LayerNorm: y[r,:] = (x[r,:] - mean) * rsqrt(var + eps) * w + b
// `flags` bit0: affine, bit1: add the residual `A` to the result.
static const char* kNormHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

__device__ __forceinline__ float ldF(const char* buf, unsigned byteOff) {
    return __uint_as_float(*(const unsigned*)(buf + byteOff));
}
__device__ __forceinline__ void stF(char* buf, unsigned byteOff, float v) {
    *(unsigned*)(buf + byteOff) = __float_as_uint(v);
}

static const unsigned NT = 256;

// Block-wide tree reduction over NT = 256 threads. This is NOT a wave op: it
// spans 8 warps, so it stays exactly the shared-memory + barrier reduction the
// HLSL `groupshared` form used, and the reduction tree (halving strides) is
// preserved, so the last-bit result matches the host reference.
__device__ __forceinline__ void reduce(unsigned t, float* red, float& total) {
    __syncthreads();
    for (unsigned s = NT / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    total = red[0];
}

// numthreads(256, 1, 1)   (the old HLSL group size was NT = 256)
extern "C" __global__ void norm_rows(Args a) {
    const unsigned rows = a.v[0];
    const unsigned cols = a.v[1];
    const unsigned xOff = a.v[2], wOff = a.v[3], bOff = a.v[4];
    const unsigned yOff = a.v[5], aOff = a.v[6];
    const unsigned flags = a.v[7];
    const float eps = __uint_as_float(a.v[8]);
    const unsigned mode = a.v[9];
    const char* X = a.s[0];
    const char* W = a.s[1];
    const char* B = a.s[2];
    const char* A = a.s[3];
    char* Y = a.u[0];

    __shared__ float red[NT];

    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    const unsigned t = threadIdx.x;
    const unsigned base = xOff + row * cols * 4;
    const unsigned obase = yOff + row * cols * 4;

    float mean = 0.0f;
    float inv = 0.0f;

    if (mode == 0u) {
        float part = 0.0f;
        for (unsigned c = t; c < cols; c += NT) {
            float v = ldF(X, base + c * 4);
            part += v * v;
        }
        red[t] = part;
        float total = 0.0f;
        reduce(t, red, total);
        inv = rsqrtf(total / (float)cols + eps);
    } else {
        float part = 0.0f;
        for (unsigned c = t; c < cols; c += NT) part += ldF(X, base + c * 4);
        red[t] = part;
        float total = 0.0f;
        reduce(t, red, total);
        mean = total / (float)cols;
        // `total` is red[0], and the variance pass below re-uses red[] - thread 0
        // writes its own partial into red[0] one statement later. Without this
        // barrier a warp whose `total = red[0]` load is issued after that store
        // reads the *variance partial* as the mean and normalises its columns by
        // the wrong constant: a race that only shows up as run-to-run drift (it
        // is what made the Qwen3-VL vision tower's LayerNorms non-deterministic;
        // the RMS path above has a single reduction and never reuses red[], which
        // is why only the LayerNorm users - the ViT and the text towers that
        // take this path - ever moved).
        __syncthreads();
        float part2 = 0.0f;
        for (unsigned c2 = t; c2 < cols; c2 += NT) {
            float d = ldF(X, base + c2 * 4) - mean;
            part2 += d * d;
        }
        red[t] = part2;
        float total2 = 0.0f;
        reduce(t, red, total2);
        inv = rsqrtf(total2 / (float)cols + eps);
    }

    for (unsigned c = t; c < cols; c += NT) {
        float v = ldF(X, base + c * 4);
        if (mode == 1u) v -= mean;
        v *= inv;
        if ((flags & 1u) != 0u) {
            v *= ldF(W, wOff + c * 4);
            if (bOff != 0u) v += ldF(B, bOff + c * 4);
        }
        if ((flags & 2u) != 0u) v += ldF(A, aOff + row * cols * 4 + c * 4);
        stF(Y, obase + c * 4, v);
    }
}

// ── the same, four elements per thread ────────────────────────────────
//
// `norm_rows` moves 4 bytes per thread per instruction, and its reduction is
// eight dependent `__syncthreads()` steps - so it measures ~9 GB/s on this part
// against the ~300 the memory system can do, i.e. it is bound by how many of
// those 4-byte accesses a warp can keep in flight and by the barrier chain, not
// by bandwidth. That made it the second largest item of the video VAE's tile
// (113 ms of a 774 ms tile, 36 calls) and the largest non-GEMM item of the
// image DiT's block (two 4096-wide norms of a 4500-row activation, 32 blocks a
// step).
//
// Four elements per thread makes every access 16 bytes, and the reduction is
// done in two steps - warp shuffles, then one cross-warp pass through shared -
// which is 2 barriers instead of 8.
//
// The last bits *do* move: the per-thread partial now sums four consecutive
// elements instead of a strided sequence, and the tree is wider at the leaf.
// Neither is a correctness issue for a mean (the reference implementations sum
// in whatever order their own blocking implies) and the fp32 error stays ~1e-7
// relative; what it is not is bit-identical to `norm_rows`, so the scalar kernel
// stays as the fallback for shapes this one cannot take.
// numthreads(256, 1, 1)
extern "C" __global__ void norm_rows4(Args a) {
    const unsigned rows = a.v[0];
    const unsigned cols = a.v[1];
    const unsigned xOff = a.v[2], wOff = a.v[3], bOff = a.v[4];
    const unsigned yOff = a.v[5], aOff = a.v[6];
    const unsigned flags = a.v[7];
    const float eps = __uint_as_float(a.v[8]);
    const unsigned mode = a.v[9];
    const char* X = a.s[0];
    const char* W = a.s[1];
    const char* B = a.s[2];
    const char* A = a.s[3];
    char* Y = a.u[0];

    __shared__ float red[NT];

    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    const unsigned t = threadIdx.x;
    const unsigned nq = cols / 4u;                 // float4 per row
    const unsigned base = xOff + row * cols * 4;
    const unsigned obase = yOff + row * cols * 4;

    float mean = 0.0f;
    float inv = 0.0f;

    // One warp-level total: the 32 lanes' partials, then the 8 warps' via shared.
    auto crossWarp = [&](float part) -> float {
        #pragma unroll
        for (unsigned off = 16u; off > 0u; off >>= 1u)
            part += __shfl_xor_sync(0xffffffffu, part, off);
        // A warp-level total is only warp-level: without this barrier the lane-0
        // writes of the *next* call's partials can land while another warp is
        // still reading red[] at the end of this one (mode 1 calls this twice),
        // which is the same race the scalar kernel documents below.
        __syncthreads();
        if ((t & 31u) == 0u) red[t >> 5u] = part;
        __syncthreads();
        if (t < (NT / 32u)) {
            float w = red[t];
            // Only the first NT/32 = 8 lanes enter this branch, so the mask is
            // the low byte, not 0xffffffff: a `_sync` shuffle whose mask names a
            // lane that never reaches the instruction is a convergence wait for
            // a lane that will never arrive, and on Volta+ (independent thread
            // scheduling) that hangs the block instead of returning garbage.
            // That is what an earlier 0xffffffff mask here did to every kernel
            // that took this path.
            #pragma unroll
            for (unsigned off = (NT / 64u); off > 0u; off >>= 1u)
                w += __shfl_xor_sync((1u << (NT / 32u)) - 1u, w, off);
            if (t == 0u) red[0] = w;
        }
        __syncthreads();
        return red[0];
    };

    if (mode == 0u) {
        float part = 0.0f;
        for (unsigned i = t; i < nq; i += NT) {
            const float4 v = *(const float4*)(X + base + i * 16u);
            part += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        inv = rsqrtf(crossWarp(part) / (float)cols + eps);
    } else {
        float part = 0.0f;
        for (unsigned i = t; i < nq; i += NT) {
            const float4 v = *(const float4*)(X + base + i * 16u);
            part += (v.x + v.y) + (v.z + v.w);
        }
        mean = crossWarp(part) / (float)cols;
        float part2 = 0.0f;
        for (unsigned i = t; i < nq; i += NT) {
            const float4 v = *(const float4*)(X + base + i * 16u);
            const float d0 = v.x - mean, d1 = v.y - mean, d2 = v.z - mean, d3 = v.w - mean;
            part2 += (d0 * d0 + d1 * d1) + (d2 * d2 + d3 * d3);
        }
        inv = rsqrtf(crossWarp(part2) / (float)cols + eps);
    }

    const bool affine = (flags & 1u) != 0u;
    const bool add_res = (flags & 2u) != 0u;
    for (unsigned i = t; i < nq; i += NT) {
        const unsigned c = i * 4u;
        float4 v = *(const float4*)(X + base + c * 4u);
        if (mode == 1u) {
            v.x -= mean; v.y -= mean; v.z -= mean; v.w -= mean;
        }
        v.x *= inv; v.y *= inv; v.z *= inv; v.w *= inv;
        if (affine) {
            const float4 w = *(const float4*)(W + wOff + c * 4u);
            v.x *= w.x; v.y *= w.y; v.z *= w.z; v.w *= w.w;
            if (bOff != 0u) {
                const float4 b = *(const float4*)(B + bOff + c * 4u);
                v.x += b.x; v.y += b.y; v.z += b.z; v.w += b.w;
            }
        }
        if (add_res) {
            const float4 r = *(const float4*)(A + aOff + row * cols * 4 + c * 4u);
            v.x += r.x; v.y += r.y; v.z += r.z; v.w += r.w;
        }
        *(float4*)(Y + obase + c * 4u) = v;
    }
}
)CUDA";

const char* norm_hlsl() { return kNormHlsl; }

// ── GroupNorm over NCHW ───────────────────────────────────────────────────
//
// One block per (image, group); the group's (C/G) * H * W values are reduced
// and then normalised with the per-channel affine parameters.
static const char* kGroupNormHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

__device__ __forceinline__ float ldF(const char* buf, unsigned byteOff) {
    return __uint_as_float(*(const unsigned*)(buf + byteOff));
}
__device__ __forceinline__ void stF(char* buf, unsigned byteOff, float v) {
    *(unsigned*)(buf + byteOff) = __float_as_uint(v);
}

static const unsigned NT = 256;

// Same block-wide (8-warp) tree reduction as norm_rows - it is not a wave op,
// so it stays a shared-memory reduction with the original tree structure.
__device__ __forceinline__ void reduceGN(unsigned t, float* red, float& total) {
    __syncthreads();
    for (unsigned s = NT / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    total = red[0];
}

// numthreads(256, 1, 1)   (the old HLSL group size was NT = 256)
extern "C" __global__ void group_norm(Args a) {
    const unsigned C = a.v[0];
    const unsigned G = a.v[3];
    const unsigned xOff = a.v[4], wOff = a.v[5], bOff = a.v[6], yOff = a.v[7];
    const float eps = __uint_as_float(a.v[8]);
    const unsigned cg = a.v[9];     // channels per group
    const unsigned hw = a.v[10];    // H*W
    const unsigned flags = a.v[11];
    const unsigned mode = a.v[12];
    const unsigned cTotal = a.v[13];  // channels per image (mode 1)
    const char* X = a.s[0];
    const char* WG = a.s[1];
    const char* BG = a.s[2];
    char* Y = a.u[0];

    __shared__ float red[NT];

    const unsigned t = threadIdx.x;
    const unsigned g = blockIdx.x % G;
    const unsigned img = blockIdx.x / G;
    const unsigned count = cg * hw;
    const unsigned base = xOff + (img * C + g * cg) * hw * 4;

    float s = 0.0f;
    for (unsigned i = t; i < count; i += NT) s += ldF(X, base + i * 4);
    red[t] = s;
    float total = 0.0f;
    reduceGN(t, red, total);
    float mean = total / (float)count;

    float v2 = 0.0f;
    for (unsigned i2 = t; i2 < count; i2 += NT) {
        float d = ldF(X, base + i2 * 4) - mean;
        v2 += d * d;
    }
    red[t] = v2;
    float total2 = 0.0f;
    reduceGN(t, red, total2);
    float inv = rsqrtf(total2 / (float)count + eps);

    if (mode == 1u) {
        // Fold the statistics and the affine into x * A + Bc, one pair per channel,
        // so a following convolution reproduces silu(group_norm(x)) exactly in the
        // space where it used to need a full-size intermediate tensor (2.14 GB at
        // 1080p - the single largest allocation the decoder makes).
        for (unsigned i3 = t; i3 < cg; i3 += NT) {
            const unsigned ch = img * C + g * cg + i3;
            const float na = inv * ldF(WG, wOff + ch * 4);
            const float bc = ldF(BG, bOff + ch * 4) - mean * na;
            stF(Y, yOff + ch * 4, na);
            stF(Y, yOff + (cTotal + ch) * 4, bc);
        }
        return;
    }

    for (unsigned i3 = t; i3 < count; i3 += NT) {
        unsigned ch = (i3 / hw) % cg;
        float y = (ldF(X, base + i3 * 4) - mean) * inv;
        if ((flags & 1u) != 0u) {
            y *= ldF(WG, wOff + (g * cg + ch) * 4);
            y += ldF(BG, bOff + (g * cg + ch) * 4);
        }
        stF(Y, yOff + (img * C + g * cg) * hw * 4 + i3 * 4, y);
    }
}
)CUDA";

const char* group_norm_hlsl() { return kGroupNormHlsl; }

// ── 2-D convolution (implicit GEMM) ───────────────────────────────────────
//
// C[oc, oh, ow] = bias[oc] + sum_{ic,kh,kw} W[oc,ic,kh,kw] * X[ic, oh*s-pad+kh, ow*s-pad+kw]
//
// Tiled like a GEMM: BM output channels x BN spatial positions, K walked in BK
// chunks where k = (ic*KH + kh)*KW + kw. The weight tile is contiguous; the
// activation tile is gathered with the padding test folded in.
static const char* kConv2dHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

__device__ __forceinline__ float ldF(const char* buf, unsigned byteOff) {
    return __uint_as_float(*(const unsigned*)(buf + byteOff));
}
__device__ __forceinline__ void stF(char* buf, unsigned byteOff, float v) {
    *(unsigned*)(buf + byteOff) = __float_as_uint(v);
}

static const unsigned BM = 64;   // output channels per block
static const unsigned BN = 64;   // spatial positions per block
static const unsigned BK = 32;
static const unsigned TM = 4;
static const unsigned TN = 4;
static const unsigned NT = 256;

// numthreads(256, 1, 1)   (the old HLSL group size was NT = 256)
// bf16 -> fp32: a bf16's value is the high half of the float with the same bits,
// so the decode is a shift of the packed halfword. Used when a weight tensor is
// kept in the checkpoint's own storage precision (Conv2dArgsG::w_dtype); see
// `ldW` above for the fp16 / fp32 cases.
__device__ __forceinline__ float ldBf16(const char* buf, unsigned byteOff) {
    return __uint_as_float((unsigned)*(const unsigned short*)(buf + byteOff) << 16);
}

// fp16 -> fp32, bit-exact and header-free: the conv2d translation unit includes no
// cuda_fp16.h, and a 10-bit significand widened into 23 bits loses nothing.
__device__ __forceinline__ float ldHalf(const char* buf, unsigned byteOff) {
    const unsigned short h = *(const unsigned short*)(buf + byteOff);
    const unsigned sign = (unsigned)(h & 0x8000u) << 16;
    const unsigned exp = (h >> 10) & 0x1Fu;
    unsigned mant = h & 0x3FFu;
    if (exp == 0u) {
        if (mant == 0u) return __uint_as_float(sign);   // +-0
        // Subnormal: normalise into the fp32 range. `m * 2^-24` becomes
        // `(m' / 1024) * 2^(-14 - shift)` once bit 10 is set.
        int shift = -1;
        do {
            shift++;
            mant <<= 1;
        } while ((mant & 0x400u) == 0u);
        mant &= 0x3FFu;
        return __uint_as_float(sign | ((113u - (unsigned)shift) << 23) | (mant << 13));
    }
    if (exp == 31u) return __uint_as_float(sign | 0x7F800000u | (mant << 13));   // inf / nan
    return __uint_as_float(sign | ((exp + 112u) << 23) | (mant << 13));
}

// One conv weight element in the checkpoint's own storage width.
// `code`: 0 = fp32, 1 = bf16, 2 = fp16. Every one of them is exact in fp32, which
// is what the accumulator runs in, so no decode here rounds a value that arrived
// exact - the only thing the code changes is how many bytes an element is.
__device__ __forceinline__ float ldW(const char* buf, unsigned byteOff, unsigned code) {
    if (code == 1u) return ldBf16(buf, byteOff);
    if (code == 2u) return ldHalf(buf, byteOff);
    return ldF(buf, byteOff);
}

extern "C" __global__ void conv2d(Args a) {
    const unsigned IH = a.v[1], IW = a.v[2], OC = a.v[3];
    const unsigned OH = a.v[4], OW = a.v[5], KH = a.v[6], KW = a.v[7];
    const unsigned xOff = a.v[8], wOff = a.v[9], bOff = a.v[10], yOff = a.v[11];
    const unsigned K = a.v[12];         // IC*KH*KW
    const unsigned stride = a.v[13];
    const int pad = (int)a.v[14];       // signed
    const unsigned flags = a.v[15];     // bit0 has bias, bit2 input duplicated, bit3 fused norm
    const unsigned pBase0 = a.v[16];    // first spatial position of this dispatch
    const unsigned dup = a.v[17];       // input nearest-duplication factor (1 or 2)
    const unsigned normOff = a.v[18];   // byte offset of A[] in the fused-norm buffer
    const unsigned normStride = a.v[19];// bytes from A[ic] to Bc[ic]
    const unsigned wBf16 = a.v[20];     // 1 = the weight tensor is bf16
    const char* X = a.s[0];
    const char* Wt = a.s[1];
    const char* Bias = a.s[2];
    const char* NormAB = a.s[4];        // read only when flags bit3 (fused norm)
    char* Y = a.u[0];

    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    const unsigned tx = threadIdx.x % (BN / TN);
    const unsigned ty = threadIdx.x / (BN / TN);
    const unsigned ocBase = blockIdx.x * BM;
    const unsigned pBase = pBase0 + blockIdx.y * BN;

    float acc[TM][TN];
    #pragma unroll
    for (unsigned i = 0; i < TM; i++)
        #pragma unroll
        for (unsigned j = 0; j < TN; j++) acc[i][j] = 0.0f;

    const unsigned kHW = KH * KW;
    const unsigned kTiles = (K + BK - 1) / BK;
    const unsigned spatial = OH * OW;

    for (unsigned kt = 0; kt < kTiles; kt++) {
        const unsigned kBase = kt * BK;
        #pragma unroll
        for (unsigned s = 0; s < (BM * BK) / NT; s++) {
            const unsigned li = threadIdx.x + s * NT;
            const unsigned r = li / BK;
            const unsigned c = li % BK;
            const unsigned oc = ocBase + r;
            const unsigned k = kBase + c;
            sA[r][c] = (oc < OC && k < K) ? ldW(Wt, wOff + (oc * K + k) * (wBf16 == 0u ? 4u : 2u), wBf16) : 0.0f;
        }
        #pragma unroll
        for (unsigned s2 = 0; s2 < (BK * BN) / NT; s2++) {
            const unsigned li = threadIdx.x + s2 * NT;
            const unsigned kk = li / BN;
            const unsigned pp = li % BN;
            const unsigned k = kBase + kk;
            float v = 0.0f;
            const unsigned p = pBase + pp;
            if (k < K && p < spatial) {
                const unsigned ic = k / kHW;
                const unsigned r = k - ic * kHW;
                const unsigned kh = r / KW;
                const unsigned kw = r - kh * KW;
                const unsigned oh = p / OW;
                const unsigned ow = p - oh * OW;
                // The window walks the (possibly nearest-duplicated) input grid:
                // `dup` expands X in the index domain, so no 2x2 copy of it has to
                // exist in memory.
                const int iy = (int)(oh * stride) - pad + (int)kh;
                const int ix = (int)(ow * stride) - pad + (int)kw;
                if (iy >= 0 && iy < (int)(IH * dup) && ix >= 0 && ix < (int)(IW * dup)) {
                    const unsigned sy = (unsigned)iy / dup;
                    const unsigned sx = (unsigned)ix / dup;
                    v = ldF(X, xOff + ((ic * IH + sy) * IW + sx) * 4);
                    if ((flags & 8u) != 0u) {
                        // silu(group_norm(x)) folded into x*A + Bc, per input channel.
                        // Outside the tensor the *normalised* form is what gets
                        // zero-padded, so the padded case leaves v at 0.0f.
                        const float na = ldF(NormAB, normOff + ic * 4);
                        const float nb = ldF(NormAB, normOff + normStride + ic * 4);
                        const float tt = v * na + nb;
                        v = tt / (1.0f + expf(-tt));
                    }
                }
            }
            sB[kk][pp] = v;
        }
        __syncthreads();

        #pragma unroll
        for (unsigned kkn = 0; kkn < BK; kkn++) {
            float av[TM], bv[TN];
            #pragma unroll
            for (unsigned i = 0; i < TM; i++) av[i] = sA[ty * TM + i][kkn];
            #pragma unroll
            for (unsigned j = 0; j < TN; j++) bv[j] = sB[kkn][tx * TN + j];
            #pragma unroll
            for (unsigned i2 = 0; i2 < TM; i2++)
                #pragma unroll
                for (unsigned j2 = 0; j2 < TN; j2++) acc[i2][j2] += av[i2] * bv[j2];
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned i = 0; i < TM; i++) {
        const unsigned oc = ocBase + ty * TM + i;
        if (oc >= OC) continue;
        float bvn = ((flags & 1u) != 0u) ? ldF(Bias, bOff + oc * 4) : 0.0f;
        #pragma unroll
        for (unsigned j = 0; j < TN; j++) {
            const unsigned p = pBase + tx * TN + j;
            if (p >= spatial) continue;
            stF(Y, yOff + (oc * spatial + p) * 4, acc[i][j] + bvn);
        }
    }
}
)CUDA";

const char* conv2d_hlsl() { return kConv2dHlsl; }

// ── nearest 2x upsample ───────────────────────────────────────────────────
//
// PyTorch's interpolate(mode="nearest") maps a destination index d to
// src = floor(d * in / out); for an exact 2x scale that is d / 2.
static const char* kUpsampleHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

__device__ __forceinline__ float ldF(const char* buf, unsigned byteOff) {
    return __uint_as_float(*(const unsigned*)(buf + byteOff));
}
__device__ __forceinline__ void stF(char* buf, unsigned byteOff, float v) {
    *(unsigned*)(buf + byteOff) = __float_as_uint(v);
}

// numthreads(256, 1, 1)
extern "C" __global__ void upsample2x(Args a) {
    const unsigned total = a.v[0];
    const unsigned H = a.v[1], W = a.v[2];
    const unsigned xOff = a.v[3], yOff = a.v[4];
    const char* X = a.s[0];
    char* Y = a.u[0];

    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const unsigned OW = W * 2u;
    const unsigned OH = H * 2u;
    const unsigned ow = i % OW;
    const unsigned oh = (i / OW) % OH;
    const unsigned c = i / (OW * OH);
    const float v = ldF(X, xOff + ((c * H + oh / 2u) * W + ow / 2u) * 4);
    stF(Y, yOff + i * 4, v);
}
)CUDA";

const char* upsample_hlsl() { return kUpsampleHlsl; }

// ── [C,S] <-> [S,C] transpose ─────────────────────────────────────────────
//
// The AE's 1x1 convolutions are channel-major ([C,H,W]) but the attention
// kernel wants token-major ([S,C]). A dedicated transpose is cheaper than
// teaching conv2d a second output layout, and it is the only layout change the
// decoder needs.
static const char* kTransposeCsHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

__device__ __forceinline__ float ldF(const char* buf, unsigned byteOff) {
    return __uint_as_float(*(const unsigned*)(buf + byteOff));
}
__device__ __forceinline__ void stF(char* buf, unsigned byteOff, float v) {
    *(unsigned*)(buf + byteOff) = __float_as_uint(v);
}

// numthreads(256, 1, 1)
extern "C" __global__ void transpose_cs(Args a) {
    const unsigned C = a.v[0], S = a.v[1];
    const unsigned xOff = a.v[2], yOff = a.v[3];
    const unsigned toToken = a.v[4];
    const char* X = a.s[0];
    char* Y = a.u[0];

    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned total = C * S;
    if (i >= total) return;
    unsigned src;
    if (toToken != 0u) {
        // out index i = s*C + c, source is x[c*S + s]
        const unsigned ss = i / C, cc = i % C;
        src = cc * S + ss;
    } else {
        // out index i = c*S + s, source is x[s*C + c]
        const unsigned cc = i / S, ss = i % S;
        src = ss * C + cc;
    }
    stF(Y, yOff + i * 4, ldF(X, xOff + src * 4));
}
)CUDA";

const char* transpose_cs_hlsl() { return kTransposeCsHlsl; }

// ── Wan-VAE upsample shortcut ("DupUp3D", factor_t 1 / factor_s 2) ───────────
//
// The channel-to-pixel shuffle a strided conv's residual path uses in the Wan
// VAEs. comfy/ldm/wan/vae2_2.py :: DupUp3D.forward, for T == 1:
//
//   y = x.repeat_interleave(repeats, 1)                 repeats = oc*4/ic
//   y = y.view(B, oc, 1, 2, 2, T, H, W).permute(0,1,5,2,6,3,7,4)
//   out[c][2h + i][2w + j] = x[(4c + 2i + j) / repeats][h][w]
//
// so the output width is the *caller's* oc, which is not ic/2 in general: the
// channel-preserving levels of the decoder (1152 -> 1152) use repeats = 4 and
// reduce to a plain nearest 2x upsample, while 1152 -> 576 uses repeats = 2 and
// folds the row parity into the channel index.
static const char* kDupUp2xHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

__device__ __forceinline__ float ldF(const char* buf, unsigned byteOff) {
    return __uint_as_float(*(const unsigned*)(buf + byteOff));
}
__device__ __forceinline__ void stF(char* buf, unsigned byteOff, float v) {
    *(unsigned*)(buf + byteOff) = __float_as_uint(v);
}

// cbuffer P: uint C_in, H, W, xOff, yOff, C_out, repeats;
//   H,W  input spatial size
// numthreads(256, 1, 1)
extern "C" __global__ void dupup2x(Args a) {
    const unsigned H = a.v[1], W = a.v[2];
    const unsigned xOff = a.v[3], yOff = a.v[4];
    const unsigned OC = a.v[5], repeats = a.v[6];
    const unsigned stride = a.v[7], shift = a.v[8];
    const char* X = a.s[0];
    char* Y = a.u[0];

    const unsigned OH = H * 2u, OW = W * 2u;
    const unsigned total = OC * OH * OW;
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total || repeats == 0u) return;

    const unsigned ow = i % OW;
    const unsigned oh = (i / OW) % OH;
    const unsigned oc = i / (OW * OH);
    const unsigned w = ow >> 1u, h = oh >> 1u;
    const unsigned ic = (stride * oc + shift + 2u * (oh & 1u) + (ow & 1u)) / repeats;
    stF(Y, yOff + i * 4u, ldF(X, xOff + ((ic * H + h) * W + w) * 4u));
}
)CUDA";

const char* dupup2x_hlsl() { return kDupUp2xHlsl; }

// ── channel-wise RMSNorm for the Wan VAEs (NCHW activations) ─────────────
//
// comfy/ldm/wan/vae.py :: RMS_norm normalises over dim = 1 (the channel axis)
// and multiplies by sqrt(C) * gamma, i.e. y = x * gamma / sqrt(mean_c x^2).
// Every other norm in the engine reduces over the *last* dimension, which in
// NCHW is a pixel, so this one needs its own kernel: one thread owns one pixel
// and walks the channel axis (consecutive threads touch consecutive pixels, so
// every load is coalesced) twice - once for the sum of squares, once for the
// scale.
//
// eps sits on the mean of the squares rather than on the norm the reference
// clamps, which differs only when a pixel's whole channel vector is ~0 (then
// both sides return 0).
static const char* kChwRmsNormHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

__device__ __forceinline__ float rldF(const char* buf, unsigned byteOff) {
    return __uint_as_float(*(const unsigned*)(buf + byteOff));
}
__device__ __forceinline__ void rstF(char* buf, unsigned byteOff, float v) {
    *(unsigned*)(buf + byteOff) = __float_as_uint(v);
}

// cbuffer P: uint S, C, xOff, yOff, gOff; float eps;
//   S    pixels of one sample (H*W)
//   C    channels
// numthreads(256, 1, 1)
extern "C" __global__ void chw_rmsnorm(Args a) {
    const unsigned S = a.v[0], C = a.v[1];
    const unsigned xOff = a.v[2], yOff = a.v[3], gOff = a.v[4];
    const float eps = __uint_as_float(a.v[5]);
    const unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= S) return;
    const char* X = a.s[0];
    const char* G = a.s[1];
    char* Y = a.u[0];

    float acc = 0.0f;
    for (unsigned c = 0; c < C; c++) {
        const float v = rldF(X, xOff + ((size_t)c * S + p) * 4u);
        acc += v * v;
    }
    const float inv = rsqrtf(acc / (float)C + eps);
    for (unsigned c = 0; c < C; c++) {
        const float v = rldF(X, xOff + ((size_t)c * S + p) * 4u);
        const float g = rldF(G, gOff + c * 4u);
        rstF(Y, yOff + ((size_t)c * S + p) * 4u, v * inv * g);
    }
}
)CUDA";

const char* chw_rmsnorm_hlsl() { return kChwRmsNormHlsl; }

// ── rectangle mask (fp32) ────────────────────────────────────────────────
//
// Zeroes everything of a [C,H,W] fp32 tensor outside rows [y0,y1) x columns
// [x0,x1), in place. The tiled VAE decode computes each tile with a halo of
// context around it, and that context has to behave exactly like the untiled
// pass does outside the image: every convolution in the reference zero-pads its
// input, so a feature map is *defined* only on the image and reads as 0 beyond
// it. A crop cannot express that on its own - its halo holds computed values -
// so each stage's output is masked back to the image footprint before the next
// one reads it. That is what makes the tiled decode bit-identical rather than
// merely seamless.
static const char* kMaskRectHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

__device__ __forceinline__ void mstF(char* buf, unsigned byteOff, float v) {
    *(unsigned*)(buf + byteOff) = __float_as_uint(v);
}

// cbuffer P: uint N (C*H*W), W, H, y0, y1, x0, x1, off;
// numthreads(256, 1, 1)
extern "C" __global__ void mask_rect(Args a) {
    const unsigned N = a.v[0];
    const unsigned W = a.v[1], H = a.v[2];
    const unsigned y0 = a.v[3], y1 = a.v[4], x0 = a.v[5], x1 = a.v[6];
    const unsigned off = a.v[7];
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const unsigned x = i % W;
    const unsigned y = (i / W) % H;
    if (y >= y0 && y < y1 && x >= x0 && x < x1) return;
    mstF(a.u[0], off + i * 4u, 0.0f);
}
)CUDA";

const char* mask_rect_hlsl() { return kMaskRectHlsl; }

// ── row-wise softmax (fp32) ────────────────────────────────────────────────
//
// The Wan VAE's middle attention does not fit the engine's flash-style attention
// kernels (its feature width is the channel count, 768/1152, where those assume
// head_dim 128), so it is run as two `gemm_f16` calls with this in between. One
// block per row, 256 threads: row max, sum of exp, then the normalised write.
static const char* kRowSoftmaxHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

__device__ __forceinline__ float rldF2(const char* buf, unsigned byteOff) {
    return __uint_as_float(*(const unsigned*)(buf + byteOff));
}
__device__ __forceinline__ void rstF2(char* buf, unsigned byteOff, float v) {
    *(unsigned*)(buf + byteOff) = __float_as_uint(v);
}

static const unsigned SNT = 256;
__shared__ float sred[SNT];

// cbuffer P: uint rows, cols, xOff, yOff
// numthreads(256, 1, 1)
extern "C" __global__ void row_softmax(Args a) {
    const unsigned rows = a.v[0], cols = a.v[1];
    const unsigned xOff = a.v[2], yOff = a.v[3];
    const unsigned r = blockIdx.x;
    const unsigned t = threadIdx.x;
    if (r >= rows) return;
    const char* X = a.s[0];
    char* Y = a.u[0];
    const unsigned base = xOff + r * cols * 4u;

    float m = -3.0e38f;
    for (unsigned i = t; i < cols; i += SNT) m = fmaxf(m, rldF2(X, base + i * 4u));
    sred[t] = m;
    __syncthreads();
    for (unsigned st = SNT / 2u; st > 0u; st >>= 1) {
        if (t < st) sred[t] = fmaxf(sred[t], sred[t + st]);
        __syncthreads();
    }
    const float mx = sred[0];
    __syncthreads();

    float s = 0.0f;
    for (unsigned i = t; i < cols; i += SNT) s += expf(rldF2(X, base + i * 4u) - mx);
    sred[t] = s;
    __syncthreads();
    for (unsigned st2 = SNT / 2u; st2 > 0u; st2 >>= 1) {
        if (t < st2) sred[t] += sred[t + st2];
        __syncthreads();
    }
    const float inv = 1.0f / fmaxf(sred[0], 1e-20f);

    const unsigned ob = yOff + r * cols * 4u;
    for (unsigned i = t; i < cols; i += SNT)
        rstF2(Y, ob + i * 4u, expf(rldF2(X, base + i * 4u) - mx) * inv);
}
)CUDA";

const char* row_softmax_hlsl() { return kRowSoftmaxHlsl; }

// ── causal 1-D convolution (the ACE-Step 1.5 audio VAE decoder) ────────────
//
// The three kernels the DAC-lineage `AudioOobleckVAE` decoder needs, one per
// module: a dilated 1-D convolution (every WNConv1d), a transposed 1-D
// convolution (every WNConvTranspose1d) and the SnakeBeta activation. The host
// references are `Conv1dRef::forward` / `ConvT1dRef::forward`
// (core/models/audio_vocoder.cpp) and `kernels::snake_beta_f32`
// (core/kernels/snake.cpp); the accumulation order of each is spelled out below
// because that - not the formula - is what decides whether the two agree.
//
// All three are one thread per output element and one float per thread: the
// decoder is ~5e12 FLOP for a 60 s clip at 48 kHz but its *layout* is what makes
// the naive form the right one here: the two convolutions' operands are shared
// across neighbouring threads (x through the overlap of neighbouring windows, w
// across every thread that owns the same output channel), so the traffic is
// served by L1/L2 and the warp's gathers coalesce along the position axis.
// A GEMM-shaped variant would need the im2col this VAE has never had.

// PyTorch Conv1d, fp32, `w` fp32 in the checkpoint's [OC, IC, K] layout:
//
//   y[oc, t] = bias[oc] + sum_ic sum_{j<K} x[ic, t*stride - pad + j*dilation] * w[oc, ic, j]
//
// An out-of-range input index *is* the reference's zero padding, so the bounds
// test is on the input index and there is no materialised border.
//
// Accumulation order: `Conv1dRef::forward` seeds the output with the bias and
// then walks ic outer, j inner (`out[j] += wk * xs[j]`), which is the same order
// this kernel evaluates - the two agree bit for bit when the host reference is
// compiled with -ffp-contract=off, and with contraction on the only difference
// is which multiply-add the compiler fuses. `dilation` is free: the residual
// units of this net use 1, 3 and 9.
static const char* kConv1dHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

__device__ __forceinline__ float ldF(const char* buf, unsigned byteOff) {
    return __uint_as_float(*(const unsigned*)(buf + byteOff));
}
__device__ __forceinline__ void stF(char* buf, unsigned byteOff, float v) {
    *(unsigned*)(buf + byteOff) = __float_as_uint(v);
}

// cbuffer P: uint IC, L, OC, K, LO, stride, pad (signed), dilation,
//              xOff, wOff, bOff, yOff, flags (bit0 = has bias)
// numthreads(256, 1, 1)
extern "C" __global__ void conv1d_gather(Args a) {
    const unsigned IC = a.v[0], L = a.v[1];
    const unsigned OC = a.v[2], K = a.v[3], LO = a.v[4];
    const int stride = (int)a.v[5];
    const int pad = (int)a.v[6];        // signed: the left border is a negative index
    const int dil = (int)a.v[7];
    const unsigned xOff = a.v[8], wOff = a.v[9], bOff = a.v[10], yOff = a.v[11];
    const unsigned flags = a.v[12];     // bit0 = has bias
    const char* X = a.s[0];
    const char* W = a.s[1];
    const char* Bias = a.s[2];
    char* Y = a.u[0];

    const unsigned total = OC * LO;
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const unsigned oc = i / LO;
    const unsigned t = i - oc * LO;

    float acc = ((flags & 1u) != 0u) ? ldF(Bias, bOff + oc * 4u) : 0.0f;
    const unsigned wRow = wOff + oc * IC * K * 4u;   // w[oc, 0, 0]
    const int base = (int)(t * (unsigned)stride) - pad;
    for (unsigned ic = 0; ic < IC; ic++) {
        const unsigned xRow = xOff + ic * L * 4u;
        const unsigned wr = wRow + ic * K * 4u;
        for (unsigned j = 0; j < K; j++) {
            const int xi = base + (int)j * dil;
            if (xi >= 0 && xi < (int)L) acc += ldF(W, wr + j * 4u) * ldF(X, xRow + (unsigned)xi * 4u);
        }
    }
    stF(Y, yOff + i * 4u, acc);
}
)CUDA";

const char* conv1d_gather_hlsl() { return kConv1dHlsl; }

// PyTorch ConvTranspose1d, fp32, `w` fp32 in the checkpoint's [IC, OC, K]
// layout:
//
//   y[oc, t] = bias[oc] + sum_ic sum_{j<K} x[ic, i] * w[ic, oc, j]   where t = i*stride - pad + j
//
// Every upsampler of this net has k == 2*stride (asserted at load time and
// again in dispatch_convt1d), which is what makes the gather *direct*: for an
// output position t exactly two taps can reach it, and only from the two input
// positions
//
//   r  = (t + pad) mod stride          (a floor-division remainder, in [0, stride))
//   i0 = (t + pad - r) / stride        ->  tap j = r
//   i1 = i0 - 1                        ->  tap j = r + stride
//
// so y[oc,t] = bias[oc] + sum_ic ( x[ic,i0]*w[ic,oc,r] + x[ic,i1]*w[ic,oc,r+stride] ),
// with an in-range test on both input indices. Accumulation order: the
// reference scatters in (ic ascending, j ascending) order, i.e. per input
// channel the j = r tap (position i0) is added before the j = r + stride one
// (position i1). Measured against `ConvT1dRef`: that order is bit-exact, the
// reverse costs ~7e-7 absolute on the same data.
static const char* kConvT1dHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

__device__ __forceinline__ float ldF(const char* buf, unsigned byteOff) {
    return __uint_as_float(*(const unsigned*)(buf + byteOff));
}
__device__ __forceinline__ void stF(char* buf, unsigned byteOff, float v) {
    *(unsigned*)(buf + byteOff) = __float_as_uint(v);
}

// cbuffer P: uint IC, L, OC, K, LO, stride, pad (signed),
//              xOff, wOff, bOff, yOff, flags (bit0 = has bias)
// numthreads(256, 1, 1)
extern "C" __global__ void convt1d(Args a) {
    const unsigned IC = a.v[0], L = a.v[1];
    const unsigned OC = a.v[2], K = a.v[3], LO = a.v[4];
    const int stride = (int)a.v[5];
    const int pad = (int)a.v[6];
    const unsigned xOff = a.v[7], wOff = a.v[8], bOff = a.v[9], yOff = a.v[10];
    const unsigned flags = a.v[11];
    const char* X = a.s[0];
    const char* W = a.s[1];
    const char* Bias = a.s[2];
    char* Y = a.u[0];

    const unsigned total = OC * LO;
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const unsigned oc = i / LO;
    const unsigned t = i - oc * LO;

    // t + pad can be negative: C's % truncates towards zero, so the remainder is
    // corrected into [0, stride) and i0 becomes floor((t + pad) / stride).
    const int up = (int)t + pad;
    int r = up % stride;
    if (r < 0) r += stride;
    const int i0 = (up - r) / stride;
    const int i1 = i0 - 1;

    float acc = ((flags & 1u) != 0u) ? ldF(Bias, bOff + oc * 4u) : 0.0f;
    for (unsigned ic = 0; ic < IC; ic++) {
        const unsigned wRow = wOff + (ic * OC + oc) * K * 4u;   // w[ic, oc, 0]
        const unsigned xRow = xOff + ic * L * 4u;
        // j = r (position i0) first, then j = r + stride (position i1): the
        // reference's own (ic, j) scatter order.
        if (i0 >= 0 && i0 < (int)L)
            acc += ldF(W, wRow + (unsigned)r * 4u) * ldF(X, xRow + (unsigned)i0 * 4u);
        if (i1 >= 0 && i1 < (int)L)
            acc += ldF(W, wRow + (unsigned)(r + stride) * 4u) * ldF(X, xRow + (unsigned)i1 * 4u);
    }
    stF(Y, yOff + i * 4u, acc);
}
)CUDA";

const char* convt1d_hlsl() { return kConvT1dHlsl; }

// SnakeBeta, the log-parameter form the checkpoint stores:
//
//   y[c, j] = x[c, j] + sin(a * x[c, j])^2 / (b + 1e-9)   with a = exp(alpha_log[c]),
//                                                              b = exp(beta_log[c])
//
// `kernels::snake_beta_f32` is the reference and its evaluation order is
// reproduced literally: r = 1/(b + 1e-9) rounded to fp32 first, then
// t = sin(a*x) and (t*t)*r + x. `a`/`b` arrive already exponentiated (open()
// uploads exp() of the stored logs, the same expression the host applies), so
// this kernel and the host path share the parameters bit for bit. x and y may
// alias: one thread reads and writes only its own element.
static const char* kSnakeBetaHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

__device__ __forceinline__ float ldF(const char* buf, unsigned byteOff) {
    return __uint_as_float(*(const unsigned*)(buf + byteOff));
}
__device__ __forceinline__ void stF(char* buf, unsigned byteOff, float v) {
    *(unsigned*)(buf + byteOff) = __float_as_uint(v);
}

// cbuffer P: uint C, L, xOff, yOff, aOff, bOff
//   aOff/bOff are the byte offsets of the per-channel arrays inside their own
//   buffers (plain arena offsets - the kernel adds them to `a.s[1]`/`a.s[2]`,
//   exactly like every other kernel here).
// numthreads(256, 1, 1)
extern "C" __global__ void snake_beta(Args a) {
    const unsigned C = a.v[0], L = a.v[1];
    const unsigned xOff = a.v[2], yOff = a.v[3], aOff = a.v[4], bOff = a.v[5];
    const char* X = a.s[0];
    const char* Alpha = a.s[1];
    const char* Beta = a.s[2];
    char* Y = a.u[0];

    const unsigned total = C * L;
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const unsigned c = i / L;
    const float av = ldF(Alpha, aOff + c * 4u);
    const float bv = ldF(Beta, bOff + c * 4u);
    const float r = 1.0f / (bv + 1e-9f);
    const float xv = ldF(X, xOff + i * 4u);
    const float t = sinf(av * xv);
    stF(Y, yOff + i * 4u, (t * t) * r + xv);
}
)CUDA";

const char* snake_beta_hlsl() { return kSnakeBetaHlsl; }

}  // namespace phi::media
