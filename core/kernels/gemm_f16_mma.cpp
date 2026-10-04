// fp16 tensor-core GEMM (`mma.m16n8k16`), plus the fp32 -> fp16 activation
// cast it needs.
//
// ── why this exists ───────────────────────────────────────────────────────
//
// The H3 video VAE's decoder is a 36-layer ViT3D: four GEMMs per block, and at
// a 256 px tile (S ~ 1.8 k tokens) those four are ~75% of the block's GPU time.
// The shared `gemm_f16` they used to run through is a 64x64x32 SIMT kernel with
// a 4x4 register tile - 2 FMAs per shared-memory load - and it measures 1.4-1.5
// TFLOP/s on this part (an RTX 3060 laptop: ~13 TFLOP/s fp32 FMA, ~27 TFLOP/s
// fp16 tensor). So the decoder spent its time at 5% of the arithmetic the card
// has, which is why one 320x512 x 5-frame decode took 53 s.
//
// This kernel is the same product on the tensor cores. It is deliberately a
// *separate* entry point rather than a rewrite of `gemm_f16`: that kernel is on
// the text encoder's, the embedders' and the Flux AE's paths, several of which
// are pinned to its exact accumulation order by tests, and it takes fp32
// activations (this one takes fp16, which is also what the reference model
// itself computes in: the checkpoint is fp16 throughout, so ComfyUI's own
// `ops.Linear` rounds both operands to fp16 and accumulates in fp32 - exactly
// this kernel's arithmetic).
//
// ── layout ────────────────────────────────────────────────────────────────
//
// 128x128 block tile, 32-deep K, 8 warps as 4 (M) x 2 (N): one warp owns 32
// rows x 64 cols, i.e. 2 m-fragments x 8 n-fragments = 16 accumulators = 64
// fp32 registers. Two-deep `cp.async` pipeline on both operands, so the K tile
// for kt+1 is in flight while kt multiplies. Shared row stride 80 bytes rather
// than the 64 the data needs, for the same reason `int8_gemm_tc` uses it: with
// a 64-byte pitch the eight rows one `ldmatrix` reads land on two 32-byte
// windows and every load replays 4x; at 80 they tile all 32 banks.
//
// A is [M,K] fp16, B is [N,K] fp16 (the checkpoint's own layout - a row of B is
// one output channel's weights), C is [M,N] fp32 plus an optional fp32 bias.
//
// cbuffer (a.v[]): 0 M 1 N 2 K 3 flags(bit0 has_bias) 4 aOff 5 bOff 6 cOff
//                  7 biasOff
// srv: s[0]=A s[1]=B s[2]=Bias   uav: u[0]=C
#include "kernels/kernels.hpp"

namespace phi::media {

// ── fp32 -> fp16 ──────────────────────────────────────────────────────────
//
// One pass over the activation so the GEMM above can read it with `cp.async`
// (which cannot convert). 4 elements a thread, 16-byte loads and 8-byte stores.
//
// cbuffer (a.v[]): 0 count 1 xOff 2 yOff   srv: s[0]=X   uav: u[0]=Y
static const char* kF32ToF16Cuda = R"CUDA(
#include <cuda_fp16.h>

struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

typedef unsigned int uint;

// numthreads(256, 1, 1)
extern "C" __global__ void f32_to_f16(Args a) {
    const uint n = a.v[0];
    const uint xOff = a.v[1], yOff = a.v[2];
    const char* X = a.s[0];
    char* Y = a.u[0];

    const uint i = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (i + 4u <= n) {
        const float4 v = *(const float4*)(X + xOff + (size_t)i * 4u);
        const __half2 h0 = __floats2half2_rn(v.x, v.y);
        const __half2 h1 = __floats2half2_rn(v.z, v.w);
        *(unsigned*)(Y + yOff + (size_t)i * 2u) = *(const unsigned*)&h0;
        *(unsigned*)(Y + yOff + ((size_t)i + 2u) * 2u) = *(const unsigned*)&h1;
    } else if (i < n) {
        for (uint k = i; k < n; k++)
            *(unsigned short*)(Y + yOff + (size_t)k * 2u) =
                __half_as_ushort(__float2half_rn(*(const float*)(X + xOff + (size_t)k * 4u)));
    }
}
)CUDA";

const char* f32_to_f16_hlsl() { return kF32ToF16Cuda; }

// ── the GEMM ──────────────────────────────────────────────────────────────
static const char* kGemmF16MmaCuda = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

typedef unsigned int uint;

static const uint BM = 128;   // rows of C per block
static const uint BN = 128;   // cols of C per block
static const uint BK = 32;    // K per shared tile (= 2 mma k-steps of 16)
static const uint NT = 256;   // 8 warps, 4 (M) x 2 (N)
static const uint SA = 80;    // shared row stride, bytes (64 of data + 16 pad)
static const uint SB = 80;
static const uint SAB = BM * SA;
static const uint SBB = BN * SB;

// One warp owns 32 rows x 64 cols: 2 m-fragments x 8 n-fragments.
//
// ── the tile is 128x128, and 256x128 was built and measured slower ───────
//
// A block re-reads its own BM-row strip of A and its own BN-wide strip of B, so
// over a whole GEMM A moves ceil(N/BN) times and B ceil(M/BM) times, and bytes
// per FLOP is (1/BM + 1/BN): 0.0156 here against 0.0117 at 256x128. On this
// part that is 21.5 TFLOP/s against 28.7 at the 336 GB/s the memory system
// delivers, and the VAE's decoder GEMMs (3.4 TMAC per tile, 105 tiles a clip)
// measure 18.5 - i.e. they look bandwidth-bound, and the wider tile looks like
// free speed.
//
// It is not, and the reason is occupancy rather than traffic. 256x128 is 16
// accumulator fragments a warp (64 registers) with 512 threads: exactly half
// the register file, so one block per SM and the SM's warps all reach the same
// `__syncthreads()` together. At 128x128 with 256 threads a second block fits,
// and its work covers the first block's barrier stalls. Widening the tile cost
// 17-28% on every decoder GEMM (w1 220 -> 257 ms, w2 117 -> 149, out_proj
// 31 -> 39, the tile 746 -> 854 ms) and the whole clip 86 -> 95 s. The bytes
// are not what this kernel is short of.
static const uint WF_M = 2;
static const uint WF_N = 8;

__shared__ __align__(16) char sA[2 * SAB];
__shared__ __align__(16) char sB[2 * SBB];

// D(16x8) += A(16x16) * B(16x8); fp16 operands, fp32 accumulate.
__device__ __forceinline__ void mma_f16(float (&d)[4], const unsigned (&a)[4],
                                        const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

// global -> shared, 16 bytes, asynchronous. `nbytes` is 0 for the K/M tail rows
// that must read as zero: the cp-size stays 16 so the destination is always a
// whole chunk, and src-size 0 makes the hardware write the padding.
//
// The source address is clamped to the last valid row/column so a tail block
// never hands the copy engine an address outside the tensor.
__device__ __forceinline__ void cp16(void* dst, const void* src, unsigned nbytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)),
                 "l"(src), "r"(nbytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Stage one BK-wide (32-half) slice of A and B: four 16-byte chunks per row,
// consecutive threads taking consecutive chunks.
//
// K is assumed to be a multiple of 8. cp.async.cg requires the SOURCE address
// to be 16-byte aligned, and the source is `A + aOff + m*K*2 + kh*2` with
// kh a multiple of 8: the offset is 16-byte aligned iff K is a multiple of 8.
// With K a multiple of 8 every BK/8-wide chunk is either fully inside [0,K)
// (ok=16) or fully past it (ok=0, hardware zero-fills the destination), so the
// binary `ok` is exact. The dispatch MUST reject K % 8 != 0 (or round K up and
// zero-pad A and B) — silently accepting such a K would clobber the last valid
// chunk's data with zeros.
//
// The old `(kh + 8u <= K ? kh : K - 8u)` source clamp was removed: when ok==16
// the chunk is entirely in-range by construction, and when ok==0 the copy is a
// no-op (src-size 0) so the source address is never dereferenced.
__device__ __forceinline__ void stage_mma(char* dsA, char* dsB, const char* A, const char* B,
                                          uint aOff, uint bOff, uint M, uint N, uint K, uint m0,
                                          uint n0, uint k0, uint t) {
    #pragma unroll
    for (uint i = 0; i < (BM * (BK / 8u)) / NT; i++) {
        const uint idx = t + NT * i;
        const uint row = idx >> 2u, cc = idx & 3u;
        const uint m = m0 + row, kh = k0 + cc * 8u;
        const uint ok = (m < M && kh + 8u <= K) ? 16u : 0u;
        cp16(dsA + row * SA + cc * 16u,
             A + aOff + (size_t)m * K * 2u + (size_t)kh * 2u,
             ok);
    }
    #pragma unroll
    for (uint i = 0; i < (BN * (BK / 8u)) / NT; i++) {
        const uint idx = t + NT * i;
        const uint row = idx >> 2u, cc = idx & 3u;
        const uint n = n0 + row, kh = k0 + cc * 8u;
        const uint ok = (n < N && kh + 8u <= K) ? 16u : 0u;
        cp16(dsB + row * SB + cc * 16u,
             B + bOff + (size_t)n * K * 2u + (size_t)kh * 2u,
             ok);
    }
}

// numthreads(256, 1, 1)
extern "C" __global__ void __launch_bounds__(NT, 2) gemm_f16_mma(Args a) {
    const uint M = a.v[0];
    const uint N = a.v[1];
    const uint K = a.v[2];
    const uint flags = a.v[3];
    const uint aOff = a.v[4];
    const uint bOff = a.v[5];
    const uint cOff = a.v[6];
    const uint biasOff = a.v[7];

    const char* A = a.s[0];
    const char* B = a.s[1];
    const char* Bias = a.s[2];
    char* C = a.u[0];

    const uint t = threadIdx.x;
    const uint lane = t & 31u;
    const uint w = t >> 5u;
    const uint gg = lane >> 2u;
    const uint tg = lane & 3u;
    const uint wm = w >> 1u;   // warp row group: 0..3
    const uint wn = w & 1u;    // warp col group: 0..1

    // Column-major tile walk: consecutive blocks share one 128-row A strip, so
    // it stays in L2 while the (much larger) B sweeps past.
    const uint gx = (M + BM - 1u) / BM;
    const uint gy = (N + BN - 1u) / BN;
    const uint bid = blockIdx.x + blockIdx.y * gx;
    const uint mTile = bid / gy;
    const uint nTile = bid - mTile * gy;
    const uint m0 = mTile * BM;
    const uint n0 = nTile * BN;

    float acc[WF_M][WF_N][4];
    #pragma unroll
    for (uint i = 0; i < WF_M; i++)
        #pragma unroll
        for (uint j = 0; j < WF_N; j++)
            #pragma unroll
            for (uint r = 0; r < 4; r++) acc[i][j][r] = 0.0f;

    // ldmatrix address decode: lane group `g8` supplies the eight row addresses
    // of matrix `g8`, `l8` is the row inside that group. For `m16n8k16` the four
    // 8x8 tiles of a 16x16 A operand are (rows 0-7,k0-7) (rows 8-15,k0-7)
    // (rows 0-7,k8-15) (rows 8-15,k8-15) - which is exactly the order ldmatrix
    // hands back as a0..a3 - and for B they are nested the other way,
    // (n0-7,k0-7) (n0-7,k8-15) (n8-15,k0-7) (n8-15,k8-15), because b0/b1 are the
    // two k halves of the *same* n row.
    const uint g8 = lane >> 3u;
    const uint l8 = lane & 7u;
    const uint aRow = l8 + ((g8 & 1u) ? 8u : 0u);
    const uint aKb = (g8 >= 2u) ? 16u : 0u;
    const uint bRow = l8 + ((g8 >= 2u) ? 8u : 0u);
    const uint bKb = (g8 & 1u) ? 16u : 0u;

    const uint kTiles = (K + BK - 1u) / BK;

    // ── two-deep pipeline prologue ─────────────────────────────────────────
    stage_mma(sA, sB, A, B, aOff, bOff, M, N, K, m0, n0, 0u, t);
    cp_commit();
    if (kTiles > 1u) {
        stage_mma(sA + SAB, sB + SBB, A, B, aOff, bOff, M, N, K, m0, n0, BK, t);
        cp_commit();
    }

    char* curA = sA;
    char* curB = sB;
    for (uint kt = 0; kt < kTiles; kt++) {
        // Leave one group in flight when there is a kt+1 to wait for, so the
        // copy for the tile after this one keeps running.
        if (kt + 1u < kTiles) cp_wait<1>();
        else cp_wait<0>();
        __syncthreads();

        #pragma unroll
        for (uint ks = 0; ks < BK / 16u; ks++) {
            const uint ka = ks * 32u;   // bytes: 16 halfs
            unsigned af[WF_M][4];
            unsigned bf[WF_N][2];

            #pragma unroll
            for (uint i = 0; i < WF_M; i++) {
                const unsigned ad =
                    smem_u32(curA + (wm * 32u + i * 16u + aRow) * SA + ka + aKb);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(af[i][0]), "=r"(af[i][1]), "=r"(af[i][2]), "=r"(af[i][3])
                             : "r"(ad));
            }
            #pragma unroll
            for (uint jj = 0; jj < WF_N / 2u; jj++) {
                const unsigned bd =
                    smem_u32(curB + (wn * 64u + jj * 16u + bRow) * SB + ka + bKb);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(bf[jj * 2u][0]), "=r"(bf[jj * 2u][1]), "=r"(bf[jj * 2u + 1u][0]),
                               "=r"(bf[jj * 2u + 1u][1])
                             : "r"(bd));
            }

            #pragma unroll
            for (uint i = 0; i < WF_M; i++)
                #pragma unroll
                for (uint j = 0; j < WF_N; j++) mma_f16(acc[i][j], af[i], bf[j]);
        }
        __syncthreads();

        // Recycle the buffer this tile just used for the tile two ahead.
        if (kt + 2u < kTiles) {
            stage_mma(curA, curB, A, B, aOff, bOff, M, N, K, m0, n0, (kt + 2u) * BK, t);
            cp_commit();
        }
        curA = (curA == sA) ? sA + SAB : sA;
        curB = (curB == sB) ? sB + SBB : sB;
    }

    // ── epilogue: C[m,n] = acc + bias[n] ───────────────────────────────────
    // The m16n8 fragment puts (row gg, col 2*tg) and (row gg, col 2*tg+1) in
    // d[0]/d[1] and their +8-row twins in d[2]/d[3].
    const bool hasBias = (flags & 1u) != 0u;
    #pragma unroll
    for (uint i = 0; i < WF_M; i++) {
        const uint rowA = wm * 32u + i * 16u + gg;
        const bool okA = (m0 + rowA) < M;
        const bool okA8 = (m0 + rowA + 8u) < M;
        #pragma unroll
        for (uint j = 0; j < WF_N; j++) {
            const uint colB = wn * 64u + j * 8u + tg * 2u;
            const uint n = n0 + colB;
            if (n >= N) continue;
            const bool okN1 = (n + 1u) < N;
            float b0 = 0.0f, b1 = 0.0f;
            if (hasBias) {
                b0 = *(const float*)(Bias + biasOff + n * 4u);
                if (okN1) b1 = *(const float*)(Bias + biasOff + (n + 1u) * 4u);
            }
            if (okA) {
                float* d0 = (float*)(C + cOff + ((size_t)(m0 + rowA) * N + n) * 4u);
                d0[0] = acc[i][j][0] + b0;
                if (okN1) d0[1] = acc[i][j][1] + b1;
            }
            if (okA8) {
                float* d1 = (float*)(C + cOff + ((size_t)(m0 + rowA + 8u) * N + n) * 4u);
                d1[0] = acc[i][j][2] + b0;
                if (okN1) d1[1] = acc[i][j][3] + b1;
            }
        }
    }
}
)CUDA";

const char* gemm_f16_mma_hlsl() { return kGemmF16MmaCuda; }

}  // namespace phi::media
