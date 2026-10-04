// int8 GEMM with dp4a (SM 6.4 dot4add_i8packed -> CUDA `__dp4a`), plus the
// convrot activation quantiser that feeds it.
//
// Math implemented here, matching comfy_kitchen's eager reference
// (quantize_and_rotate_rowwise = _rotate_activation then quantize_int8_rowwise):
//
//   A' = rot(A)                256-wide Hadamard groups along K, activations
//   sA[m] = amax(A'[m,:])/127  per *row* (per token) — the whole row, not a group
//   qA   = round(A'/sA)        int8
//   W' = rot(W)                already done offline: the checkpoint stores W' int8
//   C[m,n] = (qA[m,:] . qW[n,:]) * sA[m] * sW[n]
//
// The per-row scale means the activation kernel has to see the entire row before
// it can quantise any of it, so it walks the row twice. Rotation is repeated on
// the second pass rather than buffering K floats of rotated row in shared
// memory: the second pass hits L2, and it keeps the kernel correct for every K
// (the H3 DiT reaches K=14336, which would not fit one shared tile).
//
// ── HLSL -> CUDA ──────────────────────────────────────────────────────────
// Both entries are now `extern "C" __global__` taking the frozen `Args` block.
// The six tile constants (BM/BN/BK/TM/TN/NT) keep their exact `uint NAME = `
// spelling: `autotune.cpp::sub_const` text-substitutes the digits after
// "uint NAME = " to build tile-shape variants, so the token must stay `uint`.
// CUDA has no `uint`, so the emitted source defines it (`typedef unsigned int
// uint;`) at the top. `dot4add_i8packed(a,b,c)` is `__dp4a(a,b,c)`;
// `ByteAddressBuffer.Load/Load4` are typed pointer loads at `base + offset` and
// `RWByteAddressBuffer.Store` is `*(T*)(base + offset) = ...`. The maths,
// accumulation order and the software pipeline are unchanged.

#include <algorithm>
#include <cstdio>

#include "runtime/cuda_api.hpp"
#include "kernels/kernels.hpp"

namespace phi::media {

// ── the int8 dp4a GEMM ─────────────────────────────────────────────────────
//
// Tile 128 (M) x 64 (N) x 64 (K), 256 threads, 4x8 fragment per thread.
// A is [M,K] int8, B is [N,K] int8 (safetensors stores weights transposed: row n
// is one output channel), C is [M,N] fp32.
//
// K should be a multiple of 4 so every int8 row starts on a 4-byte boundary,
// which is what lets a thread fetch one dp4a operand with a single aligned load.
// The loader still handles a ragged K tail byte-by-byte, so any K stays correct
// (just slower).
//
// Measured on the RTX 3060 Laptop, the kernel was ~11.9 / 26.7 / 84.2 ms on
// 4096x3840x3840 / 2048x14336x5376 / 4096x21504x5376 (≈10.1 / 11.8 / 11.3 TOPS),
// against a pure-dp4a issue ceiling of ~27 TOPS and an inner-loop-only rate of
// ~20.5 TOPS. Three things were costing the difference, and each is addressed
// below; the accumulation order is untouched throughout, so the fp32 result stays
// bit-identical (the engine's self-test asserts rel L2 0.00e+00).
//
// 1. The fragment is *column-interleaved*: thread tx owns output columns
//    tx, tx+8, tx+16, ... instead of the 8 consecutive columns tx*8..tx*8+7.
//    With the consecutive mapping lane tx reads sB rows tx*8+j, so the eight
//    lanes of a warp sit 8 rows apart and 8*stride mod 32 can only take four
//    values -> a 2-way conflict on every B operand load.
//
// 2. The global loads were serialised against the compute: each k-tile read 12 KB
//    and the loop issued those loads, then waited on them at the barrier, exposing
//    the whole DRAM time. The loop now software-pipelines - the next k-tile is
//    prefetched into registers *before* the current tile's dp4a loop, so the
//    memory system works while the INT pipe runs.
//
// 3. Every operand moved 4 bytes at a time. Both tiles are now held in shared
//    memory as vec4 (four dp4a operands = 16 bytes per element) and fetched from
//    global memory with 128-bit loads, so a k-tile costs 3 global loads, 3 shared
//    stores and 12 shared loads per thread instead of 12, 12 and 48.
//
// The tile grid is also walked column-major: the hardware hands blocks out in
// linear id order (x fastest), and with the natural mapping the gx blocks that run
// together each read a different 128-row A strip — a 15.7 MB working set at
// M=4096 against a 3 MB L2 — so A was re-read from DRAM once per N-tile.
//
// cbuffer (a.v[]): 0 M 1 N 2 K 3 flags 4 aOffset 5 bOffset 6 cOffset 7 saOffset
//                  8 sbOffset 9 biasOffset
//   flags bit0: add the [N] fp32 bias to the scaled result
// srv: s[0]=A s[1]=B s[2]=SA s[3]=SB s[4]=Bias   uav: u[0]=C
static const char* kInt8GemmHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

// CUDA has no `uint`. `autotune.cpp::sub_const` substitutes the digits after the
// literal "uint NAME = ", so these names must keep that spelling; the typedef is
// what makes them compile.
typedef unsigned int uint;

static const uint BM = 128;
static const uint BN = 64;
static const uint BK = 64;
static const uint TM = 4;    // rows of C per thread
static const uint TN = 8;    // cols of C per thread
static const uint NT = 256;  // (BM/TM) * (BN/TN) = 32 * 8

// Interleaved column groups: thread tx owns columns tx + j*CX, and its TM rows
// are TY apart. Neighbouring lanes then read shared rows one apart, which the odd
// row stride (below) turns into distinct banks.
static const uint CX = BN / TN;      // 8 column groups
static const uint TY = NT / CX;      // 32 row groups

// 16-byte chunk geometry for the loader: CPC chunks per tile row, NT threads
// cover whole rows, each thread takes A_CHUNKS / B_CHUNKS chunks from rows CROW
// apart. The loads stride by CROW*K bytes and do not depend on the k-tile index.
static const uint CPC = BK / 16;                // 4
static const uint CROW = NT / CPC;              // 64
static const uint A_CHUNKS = BM / CROW;         // 2
static const uint B_CHUNKS = BN / CROW;         // 1

// Shared tiles in vec4 form; the +1 int4 of padding moves consecutive rows onto
// different banks (an odd row stride walks eight lanes onto eight banks).
static const uint AS4 = BK / 16 + 1;            // 5 int4 per row

// dot4add_i8packed is SM 6.4; on CUDA it is the `__dp4a` builtin (sm_61+). The
// old four-mad emulation existed only for the DXBC 5.1 fallback, which CUDA does
// not have, so it is dropped - `__dp4a` is the same arithmetic.
__device__ __forceinline__ int dot4acc(int a, int b, int c) {
    return __dp4a(a, b, c);
}

// One 4-byte operand of a row: a direct load, or a byte-wise assembly for the
// tail when K is not a multiple of 4 or the row runs past the end.
__device__ __forceinline__ int load4(const char* buf, uint byteOffset, uint rowBase, uint k,
                                     uint availK) {
    if (k + 3 < availK) return *(const int*)(buf + byteOffset + rowBase + k);
    uint b0 = (k + 0 < availK) ? (*(const unsigned*)(buf + byteOffset + rowBase + k + 0) & 0xFF) : 0u;
    uint b1 = (k + 1 < availK) ? (*(const unsigned*)(buf + byteOffset + rowBase + k + 1) & 0xFF) : 0u;
    uint b2 = (k + 2 < availK) ? (*(const unsigned*)(buf + byteOffset + rowBase + k + 2) & 0xFF) : 0u;
    return (int)(b0 | (b1 << 8) | (b2 << 16));
}

// Four consecutive 4-byte operands of one row. `vec` says the row start and K are
// 16-byte aligned, in which case a full chunk is one 128-bit load; anything else
// (a ragged K tail, an unaligned offset) goes through load4 four times and stays
// correct for any K.
__device__ __forceinline__ int4 ld16(const char* buf, uint byteOffset, uint rowBase, uint k,
                                     uint availK, bool vec) {
    if (vec && k + 16 <= availK) return *(const int4*)(buf + byteOffset + rowBase + k);
    int4 r;
    r.x = load4(buf, byteOffset, rowBase, k + 0, availK);
    r.y = load4(buf, byteOffset, rowBase, k + 4, availK);
    r.z = load4(buf, byteOffset, rowBase, k + 8, availK);
    r.w = load4(buf, byteOffset, rowBase, k + 12, availK);
    return r;
}

extern "C" __global__ void int8_gemm_dp4a(Args a) {
    // old [numthreads(NT, 1, 1)]
    const uint M = a.v[0];
    const uint N = a.v[1];
    const uint K = a.v[2];
    // a.v[3] bit0: add the [N] fp32 bias in s[4] (0 for every caller but the
    // video VAE's projections, whose linears are not bias-free).
    const uint flags = a.v[3];
    const uint aByteOffset = a.v[4];
    const uint bByteOffset = a.v[5];
    const uint cByteOffset = a.v[6];
    const uint saByteOffset = a.v[7];
    const uint sbByteOffset = a.v[8];
    const uint biasByteOffset = a.v[9];
    const bool hasBias = (flags & 1u) != 0u;

    const char* A = a.s[0];
    const char* B = a.s[1];
    const char* SA = a.s[2];
    const char* SB = a.s[3];
    const char* Bias = a.s[4];
    char* C = a.u[0];

    __shared__ int4 sA[BM][AS4];
    __shared__ int4 sB[BN][AS4];

    const uint tidx = threadIdx.x;

    // Column-major tile walk (see the note on the source above).
    const uint gx = (M + BM - 1) / BM;
    const uint gy = (N + BN - 1) / BN;
    const uint bid = blockIdx.x + blockIdx.y * gx;
    const uint mTile = bid / gy;
    const uint nTile = bid - mTile * gy;

    const uint tx = tidx % CX;              // 0..7  -> interleaved column sub-index
    const uint ty = tidx / CX;              // 0..31 -> row sub-index
    const uint mRow = mTile * BM + ty;      // this thread's rows are mRow + TY*i
    const uint nCol = nTile * BN + tx;      // ... and its columns nCol + CX*j

    int acc[TM][TN];
    #pragma unroll
    for (uint i = 0; i < TM; i++)
        #pragma unroll
        for (uint j = 0; j < TN; j++) acc[i][j] = 0;

    // loader coordinates: the 16-byte chunk this thread owns inside one k-tile
    const uint crow = tidx / CPC;                    // 0..CROW-1
    const uint ccol = tidx % CPC;                    // chunk within the row
    const uint aRowBase = (mTile * BM + crow) * K;    // bytes to this thread's first row
    const uint bRowBase = (nTile * BN + crow) * K;
    const uint kOwn = ccol * 16;
    const bool vec = ((K & 15u) == 0u) && ((aByteOffset & 15u) == 0u) && ((bByteOffset & 15u) == 0u);

    // A_CHUNKS / B_CHUNKS can be zero for some autotune tile shapes (CROW > BM or
    // > BN); HLSL tolerates a zero-length array, CUDA C++ does not. The loops
    // below are sized by the real value, so a dead operand array stays dead -- the
    // clamp only keeps the declaration legal and does not change any maths.
    int4 pfA[A_CHUNKS ? A_CHUNKS : 1];
    int4 pfB[B_CHUNKS ? B_CHUNKS : 1];

    // prefetch k-tile 0 (ld16 keeps the ragged-K tail correct, so any K works)
    #pragma unroll
    for (uint s = 0; s < A_CHUNKS; s++) {
        const uint m = mTile * BM + crow + CROW * s;
        pfA[s] = (m < M) ? ld16(A, aByteOffset, aRowBase + s * CROW * K, kOwn, K, vec)
                         : make_int4(0, 0, 0, 0);
    }
    #pragma unroll
    for (uint s = 0; s < B_CHUNKS; s++) {
        const uint n = nTile * BN + crow + CROW * s;
        pfB[s] = (n < N) ? ld16(B, bByteOffset, bRowBase + s * CROW * K, kOwn, K, vec)
                         : make_int4(0, 0, 0, 0);
    }

    const uint kTiles = (K + BK - 1) / BK;
    for (uint kt = 0; kt < kTiles; kt++) {
        // publish the tile that was prefetched during the previous iteration
        #pragma unroll
        for (uint s = 0; s < A_CHUNKS; s++) sA[crow + CROW * s][ccol] = pfA[s];
        #pragma unroll
        for (uint s = 0; s < B_CHUNKS; s++) sB[crow + CROW * s][ccol] = pfB[s];
        __syncthreads();

        // start the next k-tile's loads now: they overlap the dp4a loop below
        if (kt + 1 < kTiles) {
            const uint kNext = (kt + 1) * BK + kOwn;
            #pragma unroll
            for (uint s = 0; s < A_CHUNKS; s++) {
                const uint m = mTile * BM + crow + CROW * s;
                pfA[s] = (m < M) ? ld16(A, aByteOffset, aRowBase + s * CROW * K, kNext, K, vec)
                                 : make_int4(0, 0, 0, 0);
            }
            #pragma unroll
            for (uint s = 0; s < B_CHUNKS; s++) {
                const uint n = nTile * BN + crow + CROW * s;
                pfB[s] = (n < N) ? ld16(B, bByteOffset, bRowBase + s * CROW * K, kNext, K, vec)
                                 : make_int4(0, 0, 0, 0);
            }
        }

        // chunk cc holds kk = 4cc..4cc+3, consumed .x .y .z .w in that order, so
        // the per-output accumulation order is the plain k order.
        #pragma unroll
        for (uint cc = 0; cc < BK / 16; cc++) {
            int4 av[TM], bv[TN];
            #pragma unroll
            for (uint i = 0; i < TM; i++) av[i] = sA[ty + TY * i][cc];
            #pragma unroll
            for (uint j = 0; j < TN; j++) bv[j] = sB[tx + CX * j][cc];
            #pragma unroll
            for (uint i = 0; i < TM; i++)
                #pragma unroll
                for (uint j = 0; j < TN; j++) {
                    acc[i][j] = dot4acc(av[i].x, bv[j].x, acc[i][j]);
                    acc[i][j] = dot4acc(av[i].y, bv[j].y, acc[i][j]);
                    acc[i][j] = dot4acc(av[i].z, bv[j].z, acc[i][j]);
                    acc[i][j] = dot4acc(av[i].w, bv[j].w, acc[i][j]);
                }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint i = 0; i < TM; i++) {
        const uint m = mRow + TY * i;
        if (m >= M) continue;
        const float sa = *(const float*)(SA + saByteOffset + m * 4);
        #pragma unroll
        for (uint j = 0; j < TN; j++) {
            const uint n = nCol + CX * j;
            if (n >= N) continue;
            const float sb = *(const float*)(SB + sbByteOffset + n * 4);
            const float bv = hasBias ? *(const float*)(Bias + biasByteOffset + n * 4) : 0.0f;
            *(unsigned*)(C + cByteOffset + (m * N + n) * 4) =
                __float_as_uint((float)acc[i][j] * sa * sb + bv);
        }
    }
}
)CUDA";

const char* int8_gemm_hlsl() { return kInt8GemmHlsl; }

// ── int8 tensor-core GEMM ─────────────────────────────────────────────────
//
// The dp4a kernel above is the correctness reference (the engine's self-test
// asserts its output is bit-identical to the host `gemm_int8`), but it runs on
// the INT pipe. Ampere's int8 throughput is in the tensor cores, and the gap is
// an order of magnitude rather than a few percent:
//
//   measured on this RTX 3060 Laptop
//     mma.m16n8k32.row.col.s32.s8.s8.s32, register-only loop
//                                                                      121.7 TOP/s
//     dp4a pure issue (tests/int8_gemm_bench.cpp)                       27.2 TOP/s
//     the dp4a kernel above, in situ                                   ~18-20 TOP/s
//
// It matters because at 540P/10s the DiT's GEMM shapes are 1.44e15 FLOP per
// sampling step: 72 s of the step at 20 TOP/s, 12 s at 120.
//
// ── what the mma wants, and why the checkpoint already has it ─────────────
//
// `mma...m16n8k32.row.col.s32.s8.s8.s32` contracts A[16,32] (row-major) with
// B[32,8] where B is *column*-major, i.e. B[k][n] with k contiguous. Our weights
// are stored [N,K] with K contiguous, which is exactly that: no transpose, no
// repacking. Only the fragment plumbing changes, because the ISA fixes the map
// (groupID = lane>>2, tg = lane&3):
//
//   a0 = A[g][4tg .. 4tg+3]      a1 = A[g+8][4tg .. 4tg+3]
//   a2 = A[g][4tg+16 .. 4tg+19]  a3 = A[g+8][4tg+16 .. +19]
//   b0 = B[4tg .. 4tg+3][g]      b1 = B[4tg+16 .. 4tg+19][g]
//   d0 = D[g][2tg]  d1 = D[g][2tg+1]  d2 = D[g+8][2tg]  d3 = D[g+8][2tg+1]
//
// ── how the shape of this kernel was arrived at ───────────────────────────
//
// Three measurements, in order, and each one closed off an explanation:
//
//   1. The first version was 128x64x64 with 128 threads: 34 TOP/s. Replacing
//      the shared staging with a bank-conflict-free form bought 8%, and doubling
//      the K per tile bought nothing - so neither shared bandwidth nor the
//      barrier count was the limit.
//   2. Deleting *only* the global loads, leaving everything else in place, took
//      the same kernel to 92 TOP/s. So the limit was the load path.
//   3. At the DiT's shapes there are only 3 LDG.128 per thread per k-tile, so it
//      is not instruction count or byte count - it is that one block per SM
//      (512 threads at ~118 registers is the whole register file) has nothing to
//      run while it waits ~700 cycles for the tile, and a `__syncthreads()`
//      right behind the load makes every warp wait for it.
//
// Hence: a 256x128 block tile (which also raises the flops-per-byte the block
// reads from L2 from 85 to 171) and a **two-deep cp.async pipeline**, so tile
// kt+1 is already in flight while tile kt is being multiplied. cp.async is what
// makes that affordable: the copies go global->shared without occupying the
// registers the accumulators need.
//
// ── shared memory layout ─────────────────────────────────────────────────
//
// Each a/b fragment is 8 rows x 16 bytes, which is what `ldmatrix.m8n8.x4.b16`
// loads natively (an 8x8 matrix of b16 is 8 rows of 16 bytes, and lane L
// receives row L>>2, byte 4*(L&3)), so every fragment is one instruction.
//
// The shared row stride is 80 bytes rather than the 64 the data needs. That is
// not padding for its own sake: with a 64-byte stride rows 0..7 start on only
// two distinct 32-byte windows and every ldmatrix replays 4x. At 80 bytes
// (20 words) the eight row addresses land on the eight disjoint 4-bank windows
// ({0,20,8,28,16,4,24,12} + {0..3}), so the loads are conflict-free.
//
// The 16-byte staging stores do replay 4x under that stride (eight rows x four
// chunks cannot tile 32 banks when the row pitch is a multiple of 4 words), and
// they are left that way deliberately: measurements (1) above show shared write
// bandwidth is not what limits this kernel.
//
// cbuffer (a.v[]): 0 M 1 N 2 K 3 flags 4 aOffset 5 bOffset 6 cOffset 7 saOffset
//                  8 sbOffset 9 biasOffset
//   flags bit0: add the [N] fp32 bias to the scaled result
// srv: s[0]=A s[1]=B s[2]=SA s[3]=SB s[4]=Bias   uav: u[0]=C
static const char* kInt8GemmTcCuda = R"CUDA(
// [numthreads(NT, 1, 1)] - 16 warps, laid out 4 (M) x 4 (N) by the map below.
// smem 61440
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

typedef unsigned int uint;

static const uint BM = 256;   // rows of C per block
static const uint BN = 128;   // cols of C per block
static const uint BK = 64;    // K per shared tile (= 2 mma k-steps)
static const uint NT = 512;   // 16 warps, 4 (M) x 4 (N)
static const uint SA = 80;    // shared row stride, bytes (64 + 16 of padding)
static const uint SB = 80;
static const uint SAB = BM * SA;
static const uint SBB = BN * SB;

// One warp owns 64 rows x 32 cols: 4 m-fragments x 4 n-fragments = 16
// `m16n8k32` accumulators = 64 registers. That is what fixes the block at 512
// threads: a 64x64 warp tile would need 128 registers for accumulators alone,
// and 512 threads x 128 registers is the entire register file.
static const uint WF_M = 4;
static const uint WF_N = 4;

// Two tiles of each operand, so the copy for kt+1 runs while kt multiplies.
// That is 61 KB, past the 48 KB static limit, so it is dynamic shared memory
// (the `// smem 61440` marker above is what makes the host opt in).
extern __shared__ __align__(16) char sAB[];
#define sA (sAB)
#define sB (sAB + 2 * SAB)

__device__ __forceinline__ void mma_i8(int (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

// global -> shared, 16 bytes, asynchronous. `nbytes` is 0 for the K/N tail rows
// that must read as zero: the cp-size stays 16 so the destination is always a
// whole chunk, and src-size 0 makes the hardware write the padding.
//
// A note on what is *not* optimised here. A 32-lane 16-byte store can never be
// bank-conflict-free on this part: every row pitch that keeps `ldmatrix`'s
// 16-byte addresses aligned is a multiple of 4 words, so the row-to-row bank
// offset is a multiple of 4, and the 4-word window a lane touches therefore
// starts on one of only 8 admissible boundaries - 32 lanes over 8 boundaries is
// a guaranteed 4x replay. A 4-byte cp.async with a conflict-free (row, word)
// map does remove that replay, and it was built and measured: it lands a warp
// instruction on 8 rows x 16 bytes instead of 8 rows x 64, so it fetches twice
// the cache lines, and the two effects cancel - 52.6 TOP/s either way on
// 4096x3840x3840, but the 4-byte form lost 20% on the 4096x21504x5376 shape
// where B is 115 MB and the extra sectors are real DRAM traffic. The 16-byte
// form is kept because its cost is a shared-memory replay that the kernel has
// headroom for, not a DRAM read it does not.
__device__ __forceinline__ void cp16(void* dst, const void* src, unsigned nbytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)),
                 "l"(src), "r"(nbytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Stage one BK-wide slice of A and B. The source addresses are clamped to the
// last valid row so that a tail block never hands the copy engine an address
// outside the tensor; the byte count, not the address, is what makes it zero.
__device__ __forceinline__ void stage_tc(char* dsA, char* dsB, const char* A, const char* B,
                                         uint aOff, uint bOff, uint M, uint N, uint K, uint m0,
                                         uint n0, uint k0, uint t) {
    #pragma unroll
    for (uint i = 0; i < (BM * (BK / 16u)) / NT; i++) {
        const uint idx = t + NT * i;
        const uint row = idx >> 2u, c = idx & 3u;
        const uint m = m0 + row, kb = k0 + c * 16u;
        const uint ok = (m < M && kb + 16u <= K) ? 16u : 0u;
        cp16(dsA + row * SA + c * 16u,
             A + aOff + (size_t)(m < M ? m : M - 1u) * K + (kb + 16u <= K ? kb : K - 16u), ok);
    }
    #pragma unroll
    for (uint i = 0; i < (BN * (BK / 16u)) / NT; i++) {
        const uint idx = t + NT * i;
        const uint row = idx >> 2u, c = idx & 3u;
        const uint n = n0 + row, kb = k0 + c * 16u;
        const uint ok = (n < N && kb + 16u <= K) ? 16u : 0u;
        cp16(dsB + row * SB + c * 16u,
             B + bOff + (size_t)(n < N ? n : N - 1u) * K + (kb + 16u <= K ? kb : K - 16u), ok);
    }
}

extern "C" __global__ void __launch_bounds__(NT, 1) int8_gemm_tc(Args a) {
    const uint M = a.v[0];
    const uint N = a.v[1];
    const uint K = a.v[2];
    // a.v[3] bit0: add the [N] fp32 bias in s[4] (see the dp4a entry).
    const uint flags = a.v[3];
    const uint aByteOffset = a.v[4];
    const uint bByteOffset = a.v[5];
    const uint cByteOffset = a.v[6];
    const uint saByteOffset = a.v[7];
    const uint sbByteOffset = a.v[8];
    const uint biasByteOffset = a.v[9];
    const bool hasBias = (flags & 1u) != 0u;

    const char* A = a.s[0];
    const char* B = a.s[1];
    const char* SAp = a.s[2];
    const char* SBp = a.s[3];
    const char* Bias = a.s[4];
    char* C = a.u[0];

    const uint t = threadIdx.x;
    const uint lane = t & 31u;
    const uint w = t >> 5u;
    const uint gg = lane >> 2u;
    const uint tg = lane & 3u;
    const uint wm = w >> 2u;   // warp row group: 0..3
    const uint wn = w & 3u;    // warp col group: 0..3

    // Column-major tile walk: consecutive blocks share one 256-row A strip so it
    // stays in L2 while the (much larger) B sweeps past. Same argument as the
    // dp4a kernel's and worth the same here.
    const uint gx = (M + BM - 1u) / BM;
    const uint gy = (N + BN - 1u) / BN;
    const uint bid = blockIdx.x + blockIdx.y * gx;
    const uint mTile = bid / gy;
    const uint nTile = bid - mTile * gy;
    const uint m0 = mTile * BM;
    const uint n0 = nTile * BN;

    int acc[WF_M][WF_N][4];
    #pragma unroll
    for (uint i = 0; i < WF_M; i++)
        #pragma unroll
        for (uint j = 0; j < WF_N; j++)
            #pragma unroll
            for (uint r = 0; r < 4; r++) acc[i][j][r] = 0;

    // ldmatrix address decode: lane group `g8` supplies the eight row addresses of
    // matrix `g8`, and `l8` is the row inside that group.
    //
    // A needs (rows 0-7,k0-15) (rows 8-15,k0-15) (rows 0-7,k16-31) (rows 8-15,k16-31)
    // in that order to land as a0 a1 a2 a3. B needs the *other* nesting -
    // (n0-7,k0-15) (n0-7,k16-31) (n8-15,k0-15) (n8-15,k16-31) - because b0 and b1
    // are the two k halves of the *same* n row, not two n rows of one k half.
    const uint g8 = lane >> 3u;
    const uint l8 = lane & 7u;
    const uint aRowInFrag = l8 + ((g8 & 1u) ? 8u : 0u);
    const uint aKol = (g8 >= 2u) ? 16u : 0u;
    const uint bRowInFrag = l8 + ((g8 >= 2u) ? 8u : 0u);
    const uint bKol = (g8 & 1u) ? 16u : 0u;

    const uint kTiles = (K + BK - 1u) / BK;

    // ── two-deep pipeline prologue ─────────────────────────────────────────
    stage_tc(sA, sB, A, B, aByteOffset, bByteOffset, M, N, K, m0, n0, 0u, t);
    cp_commit();
    if (kTiles > 1u) {
        stage_tc(sA + SAB, sB + SBB, A, B, aByteOffset, bByteOffset, M, N, K, m0, n0, BK, t);
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
        for (uint ks = 0; ks < BK / 32u; ks++) {
            const uint ka = ks * 32u;
            unsigned af[WF_M][4];
            unsigned bf[WF_N][2];

            #pragma unroll
            for (uint i = 0; i < WF_M; i++) {
                const unsigned ad =
                    smem_u32(curA + (wm * 64u + i * 16u + aRowInFrag) * SA + ka + aKol);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(af[i][0]), "=r"(af[i][1]), "=r"(af[i][2]), "=r"(af[i][3])
                             : "r"(ad));
            }
            #pragma unroll
            for (uint jj = 0; jj < WF_N / 2u; jj++) {
                const unsigned bd =
                    smem_u32(curB + (wn * 32u + jj * 16u + bRowInFrag) * SB + ka + bKol);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(bf[jj * 2u][0]), "=r"(bf[jj * 2u][1]), "=r"(bf[jj * 2u + 1u][0]),
                               "=r"(bf[jj * 2u + 1u][1])
                             : "r"(bd));
            }

            #pragma unroll
            for (uint i = 0; i < WF_M; i++)
                #pragma unroll
                for (uint j = 0; j < WF_N; j++) mma_i8(acc[i][j], af[i], bf[j]);
        }
        __syncthreads();

        // Recycle the buffer this tile just used for the tile two ahead.
        if (kt + 2u < kTiles) {
            stage_tc(curA, curB, A, B, aByteOffset, bByteOffset, M, N, K, m0, n0,
                     (kt + 2u) * BK, t);
            cp_commit();
        }
        curA = (curA == sA) ? sA + SAB : sA;
        curB = (curB == sB) ? sB + SBB : sB;
    }

    // ── epilogue: C[m,n] = acc * sa[m] * sb[n] ─────────────────────────────
    #pragma unroll
    for (uint i = 0; i < WF_M; i++) {
        const uint rowA = wm * 64u + i * 16u + gg;
        const bool okA = (m0 + rowA) < M;
        const bool okA8 = (m0 + rowA + 8u) < M;
        const float sa0 = okA ? *(const float*)(SAp + saByteOffset + (m0 + rowA) * 4u) : 0.0f;
        const float sa1 = okA8 ? *(const float*)(SAp + saByteOffset + (m0 + rowA + 8u) * 4u) : 0.0f;
        #pragma unroll
        for (uint j = 0; j < WF_N; j++) {
            const uint colB = wn * 32u + j * 8u + tg * 2u;
            const uint n = n0 + colB;
            if (n >= N) continue;
            const float sb0 = *(const float*)(SBp + sbByteOffset + n * 4u);
            const float sb1 = (n + 1u < N) ? *(const float*)(SBp + sbByteOffset + (n + 1u) * 4u) : 0.0f;
            // The bias is [N] fp32, so it rides with the sb loads that are already
            // in the epilogue's registers rather than costing its own pass.
            const float b0 = hasBias ? *(const float*)(Bias + biasByteOffset + n * 4u) : 0.0f;
            const float b1 = (hasBias && n + 1u < N)
                                 ? *(const float*)(Bias + biasByteOffset + (n + 1u) * 4u)
                                 : 0.0f;
            if (okA) {
                float* d0 = (float*)(C + cByteOffset + ((size_t)(m0 + rowA) * N + n) * 4u);
                d0[0] = (float)acc[i][j][0] * sa0 * sb0 + b0;
                if (n + 1u < N) d0[1] = (float)acc[i][j][1] * sa0 * sb1 + b1;
            }
            if (okA8) {
                float* d1 = (float*)(C + cByteOffset + ((size_t)(m0 + rowA + 8u) * N + n) * 4u);
                d1[0] = (float)acc[i][j][2] * sa1 * sb0 + b0;
                if (n + 1u < N) d1[1] = (float)acc[i][j][3] * sa1 * sb1 + b1;
            }
        }
    }
}
)CUDA";

const char* int8_gemm_tc_hlsl() { return kInt8GemmTcCuda; }

// ── activation rotate + per-row quantise ───────────────────────────────────
//
// One block per row of the [M,K] fp32 activation, 256 threads; K must be a
// multiple of 256 (every K in both models is: 2560/3840/5120/5376/10240/11520/
// 14336).
//
// Pass 1 rotates every group and reduces a row-wide amax.
// Pass 2 rotates again (cheap, and by then in L2) and writes int8 + the row
// scale, matching quantize_and_rotate_rowwise exactly.
//
// The rotation is H = H4^{x4} with H4 = J - 2R, normalised by 16; stage d works
// on the 4 values that differ only in base-4 digit d, so it is a ping-pong
// butterfly: 4 stages, one shared-memory barrier per stage, reads and writes
// always in different halves of the tile. A single-buffer version needs two
// barriers per stage and races across warps — which is exactly how a byte-exact
// reference comparison catches it.
//
// H4 = J - 2R, not 2I - J: both are Hadamard matrices but the checkpoint was
// rotated with the first, so the digit pass is y[d] = sum(x) - 2*x[3-d].
//
// cbuffer (a.v[]): 0 M 1 K 2 xOffset 3 qOffset 4 sOffset 5 qmax
// srv: s[0]=X   uav: u[0]=Q u[1]=S
//
// Reduction note: the row amax uses the same full-block shared-memory tree as
// the HLSL (no wave intrinsics), kept barrier-for-barrier identical.
static const char* kQuantConvrotHlsl = R"CUDA(
struct Args {
    unsigned v[24];
    const char* s[8];
    char* u[4];
};

typedef unsigned int uint;

static const uint GRP = 256;
static const uint NT = 256;
// Groups rotated between two barriers (`bfly_stage` below carries the barrier of
// its own stage, so RP groups are rotated between two of them).
//
// The butterfly is independent per group, so ROTATING MORE OF THEM BETWEEN
// BARRIERS IS FREE ARITHMETICALLY - the stages, their order and the per-row max
// are unchanged, and the exactness check below (`convrot activation kernel ...
// mismatches 0`) is the gate that says so. The win is small but real and
// measured with PHI_QUANT_BENCH at [2048, 14336]: 2.07 ms at RP=1 against 1.91 ms
// at RP=8, i.e. 114 -> 123 GB/s. It is *not* the large win one might expect from
// the barrier count, because this kernel is memory-bound: it reads the row twice
// (once for the rotation and the amax, once to rotate and quantise) and that is
// 8 bytes of DRAM per element, which is where the time goes - not in the
// 5 barriers per 256-element group that the first version had.
static const uint RP = 8;
static const uint SB = 2 * GRP;   // floats per group: ping-pong halves

// One radix-4 stage over a GROUP-sized slice, in place with a ping-pong half.
// Identical arithmetic to the serial form (`butterfly`); the caller owns the
// barriers so it can run RP slices between them.
__device__ __forceinline__ void bfly_stage(float* buf, uint t, uint stage) {
    const uint rBase = (stage & 1u) ? GRP : 0u;
    const uint wBase = (stage & 1u) ? 0u : GRP;
    const uint stride = 1u << (2u * stage);
    const uint blk = 4u * stride;
    const uint b0 = (t / blk) * blk;
    const uint off = t % stride;
    const uint d = (t / stride) % 4u;
    float x0 = buf[rBase + b0 + off + 0u * stride];
    float x1 = buf[rBase + b0 + off + 1u * stride];
    float x2 = buf[rBase + b0 + off + 2u * stride];
    float x3 = buf[rBase + b0 + off + 3u * stride];
    float s = x0 + x1 + x2 + x3;
    float mirror = (d == 0u) ? x3 : (d == 1u) ? x2 : (d == 2u) ? x1 : x0;
    buf[wBase + b0 + off + d * stride] = s - 2.0f * mirror;
    // Four stages with alternating halves leave the result in buf[0..GRP-1].
}

extern "C" __global__ void quant_convrot_activation(Args a) {
    // old [numthreads(NT, 1, 1)]
    const uint M = a.v[0];
    const uint K = a.v[1];              // multiple of 256
    const uint xByteOffset = a.v[2];
    const uint qByteOffset = a.v[3];
    const uint sByteOffset = a.v[4];
    const float qmax = __uint_as_float(a.v[5]);   // 127 (int8) or 7 (the w4a4 head)

    const char* X = a.s[0];
    char* Q = a.u[0];   // int8 activations, [M,K]
    char* S = a.u[1];   // per-row scale, [M] fp32

    __shared__ float buf[RP * SB];
    __shared__ float red[NT];
    __shared__ int qbuf[8 * GRP];   // quantised rows, packed 4-wide on store

    // gid.x is uniform across the block, so this whole-block return cannot
    // strand a thread at a barrier below.
    if (blockIdx.x >= M) return;
    const uint row = blockIdx.x;
    const uint t = threadIdx.x;
    const uint groups = K / GRP;
    const uint xRow = xByteOffset + row * K * 4;

    // ── pass 1: rotate everything, track the row amax ──
    float amax = 0.0f;
    for (uint g0 = 0; g0 < groups; g0 += RP) {
        const uint n = (g0 + RP <= groups) ? RP : (groups - g0);
        #pragma unroll
        for (uint j = 0; j < RP; j++)
            buf[j * SB + t] =
                (j < n) ? *(const float*)(X + xRow + ((g0 + j) * GRP + t) * 4) : 0.0f;
        __syncthreads();
        #pragma unroll
        for (uint st = 0; st < 4; st++) {
            #pragma unroll
            for (uint j = 0; j < RP; j++) bfly_stage(buf + j * SB, t, st);
            __syncthreads();
        }
        #pragma unroll
        for (uint j = 0; j < RP; j++)
            if (j < n) amax = fmaxf(amax, fabsf(buf[j * SB + t] * (1.0f / 16.0f)));
    }

    // tree reduction over the block (explicit, full 256-thread block). A max is
    // order-independent, so running RP groups between the barriers above does not
    // change the result.
    red[t] = amax;
    __syncthreads();
    #pragma unroll
    for (uint step = NT / 2; step > 0; step >>= 1) {
        if (t < step) red[t] = fmaxf(red[t], red[t + step]);
        __syncthreads();
    }
    const float rowAmax = red[0];
    const float scale = (rowAmax > 1e-30f) ? (rowAmax / qmax) : 1.0f;
    const float inv = 1.0f / scale;

    // ── pass 2: rotate again and quantise with the final row scale ──
    for (uint g0 = 0; g0 < groups; g0 += RP) {
        const uint n = (g0 + RP <= groups) ? RP : (groups - g0);
        #pragma unroll
        for (uint j = 0; j < RP; j++)
            buf[j * SB + t] =
                (j < n) ? *(const float*)(X + xRow + ((g0 + j) * GRP + t) * 4) : 0.0f;
        __syncthreads();
        #pragma unroll
        for (uint st = 0; st < 4; st++) {
            #pragma unroll
            for (uint j = 0; j < RP; j++) bfly_stage(buf + j * SB, t, st);
            __syncthreads();
        }
        // Round half away from zero, exactly like the host reference
        // ((long)(d + 0.5) for d >= 0). HLSL round() is half-to-even, which
        // would differ on the .5 boundary.
        #pragma unroll
        for (uint j = 0; j < RP; j++) {
            if (j >= n) continue;
            float d = buf[j * SB + t] * (1.0f / 16.0f) * inv;
            d = fminf(fmaxf(d, -qmax), qmax);
            qbuf[j * GRP + t] = (int)(d >= 0.0f ? floorf(d + 0.5f) : ceilf(d - 0.5f));
        }
        __syncthreads();
        // one 32-bit store per four int8 outputs, little-endian byte order.
        // RP * GRP/4 = 512 slots cover `RP` groups, so every thread has work.
        #pragma unroll
        for (uint slot = 0; slot < RP * (GRP / 4); slot += NT) {
            const uint i = t + slot;
            if (i >= RP * (GRP / 4)) break;
            const uint j = i / (GRP / 4);
            if (j >= n) continue;
            const uint p = (i % (GRP / 4)) * 4u;
            uint packed = ((uint)qbuf[j * GRP + p + 0u] & 0xFFu) |
                          (((uint)qbuf[j * GRP + p + 1u] & 0xFFu) << 8) |
                          (((uint)qbuf[j * GRP + p + 2u] & 0xFFu) << 16) |
                          (((uint)qbuf[j * GRP + p + 3u] & 0xFFu) << 24);
            *(unsigned*)(Q + qByteOffset + row * K + (g0 + j) * GRP + p) = packed;
        }
        __syncthreads();
    }

    if (t == 0) *(unsigned*)(S + sByteOffset + row * 4) = __float_as_uint(scale);
}
)CUDA";

const char* quant_convrot_hlsl() { return kQuantConvrotHlsl; }

}  // namespace phi::media
