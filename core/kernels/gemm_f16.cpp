// fp16 GEMM (fp32 accumulate) and the layout kernels: patchify / unpatchify.
//
// Ported from HLSL to CUDA C++. The host-side contract is frozen: the entry
// names are unchanged (hence `extern "C"`), the old root constants arrive
// verbatim in `Args::v[24]` (byte offsets included) and the resources in
// `s[0..7]` / `u[0..3]` in the original binding order. The old
// `[numthreads(x, y, z)]` group size is kept as a `numthreads(...)` comment,
// since the host parses it back out to size the launch.
//
// Both DiTs are int8 for every big matmul, but the text encoders, the
// embedders and the output projections are bf16/fp32 weights, so those go
// through here. Weights are converted to fp16 once at load time; activations stay fp32
// and are converted on the fly (the operand is small - S tokens - so the extra
// bandwidth is irrelevant next to the weights).
#include "kernels/kernels.hpp"

namespace phi::media {

static const char* kGemmF16Hlsl = R"CUDA(
#include <cuda_fp16.h>

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

static const unsigned BM = 64;
static const unsigned BN = 64;
static const unsigned BK = 32;
static const unsigned TM = 4;
static const unsigned TN = 4;
static const unsigned NT = 256;

// Two packed fp16 halves at `byteOff`. Read as two 16-bit words so the access
// stays valid even when byteOff is only 2-byte aligned (K need not be even);
// on the little-endian device this is byte-identical to a 32-bit Load.
__device__ __forceinline__ float2 loadHalf2(const char* buf, unsigned byteOff) {
    unsigned u = (unsigned)*(const unsigned short*)(buf + byteOff) |
                 ((unsigned)*(const unsigned short*)(buf + byteOff + 2) << 16);
    __half2 h = *reinterpret_cast<const __half2*>(&u);
    return make_float2(__low2float(h), __high2float(h));
}

// bfloat16 is just the top half of an fp32, so the conversion is a shift.
__device__ __forceinline__ float2 loadBf16x2(const char* buf, unsigned byteOff) {
    unsigned u = (unsigned)*(const unsigned short*)(buf + byteOff) |
                 ((unsigned)*(const unsigned short*)(buf + byteOff + 2) << 16);
    return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xFFFF0000u));
}

// numthreads(256, 1, 1)   (the old HLSL group size was NT = 256)
extern "C" __global__ void gemm_f16(Args a) {
    const unsigned M = a.v[0], N = a.v[1], K = a.v[2];
    const unsigned flags = a.v[3];   // bit0: A is fp32, bit1: has bias, bit2: B is bf16, bit3: B is fp32
    const unsigned aOff = a.v[4], bOff = a.v[5], cOff = a.v[6], biasOff = a.v[7];
    const char* A = a.s[0];          // [M,K] fp16 (or fp32 when flags&1)
    const char* B = a.s[1];          // [N,K] fp16
    const char* Bias = a.s[2];
    char* C = a.u[0];                // [M,N] fp32

    __shared__ float sA[BM][BK];
    __shared__ float sB[BN][BK];

    const unsigned tx = threadIdx.x % (BN / TN);
    const unsigned ty = threadIdx.x / (BN / TN);
    const unsigned mBase = blockIdx.x * BM;
    const unsigned nBase = blockIdx.y * BN;

    float acc[TM][TN];
    #pragma unroll
    for (unsigned i = 0; i < TM; i++)
        #pragma unroll
        for (unsigned j = 0; j < TN; j++) acc[i][j] = 0.0f;

    const bool aIsF32 = (flags & 1u) != 0u;
    const bool bIsBf16 = (flags & 4u) != 0u;
    const bool bIsF32 = (flags & 8u) != 0u;
    const unsigned kTiles = (K + BK - 1) / BK;

    for (unsigned kt = 0; kt < kTiles; kt++) {
        const unsigned kBase = kt * BK;
        if (aIsF32) {
            #pragma unroll
            for (unsigned s = 0; s < (BM * BK) / NT; s++) {
                const unsigned li = threadIdx.x + s * NT;
                const unsigned r = li / BK;
                const unsigned c = li % BK;
                const unsigned m = mBase + r;
                const unsigned k = kBase + c;
                sA[r][c] = (m < M && k < K) ? ldF(A, aOff + (m * K + k) * 4) : 0.0f;
            }
        } else {
            #pragma unroll
            for (unsigned s = 0; s < (BM * BK / 2) / NT; s++) {
                const unsigned li = threadIdx.x + s * NT;
                const unsigned r = li / (BK / 2);
                const unsigned c2 = li % (BK / 2);
                const unsigned m = mBase + r;
                const unsigned k = kBase + c2 * 2;
                float2 v = (m < M && (k + 1) < K) ? loadHalf2(A, aOff + (m * K + k) * 2)
                                                  : make_float2(0.0f, 0.0f);
                if (m < M && k < K) sA[r][c2 * 2] = v.x;
                if (m < M && (k + 1) < K) sA[r][c2 * 2 + 1] = v.y;
            }
        }
        if (bIsF32) {
            // Unquantised fp32 weights (embedders, final layer): kept in fp32
            // rather than rounded to fp16, so a difference against the
            // reference is a bug and never a rounding question.
            #pragma unroll
            for (unsigned s3 = 0; s3 < (BN * BK) / NT; s3++) {
                const unsigned li = threadIdx.x + s3 * NT;
                const unsigned r = li / BK;
                const unsigned c = li % BK;
                const unsigned n = nBase + r;
                const unsigned k = kBase + c;
                sB[r][c] = (n < N && k < K) ? ldF(B, bOff + (n * K + k) * 4) : 0.0f;
            }
        } else {
            #pragma unroll
            for (unsigned s2 = 0; s2 < (BN * BK / 2) / NT; s2++) {
                const unsigned li = threadIdx.x + s2 * NT;
                const unsigned r = li / (BK / 2);
                const unsigned c2 = li % (BK / 2);
                const unsigned n = nBase + r;
                const unsigned k = kBase + c2 * 2;
                float2 v = make_float2(0.0f, 0.0f);
                if (n < N && (k + 1) < K) {
                    v = bIsBf16 ? loadBf16x2(B, bOff + (n * K + k) * 2)
                                : loadHalf2(B, bOff + (n * K + k) * 2);
                }
                if (n < N && k < K) sB[r][c2 * 2] = v.x;
                if (n < N && (k + 1) < K) sB[r][c2 * 2 + 1] = v.y;
            }
        }
        __syncthreads();

        #pragma unroll
        for (unsigned kkn = 0; kkn < BK; kkn++) {
            float av[TM], bv[TN];
            #pragma unroll
            for (unsigned i = 0; i < TM; i++) av[i] = sA[ty * TM + i][kkn];
            #pragma unroll
            for (unsigned j = 0; j < TN; j++) bv[j] = sB[tx * TN + j][kkn];
            #pragma unroll
            for (unsigned i2 = 0; i2 < TM; i2++)
                #pragma unroll
                for (unsigned j2 = 0; j2 < TN; j2++) acc[i2][j2] += av[i2] * bv[j2];
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned i = 0; i < TM; i++) {
        const unsigned m = mBase + ty * TM + i;
        if (m >= M) continue;
        #pragma unroll
        for (unsigned j = 0; j < TN; j++) {
            const unsigned n = nBase + tx * TN + j;
            if (n >= N) continue;
            float v = acc[i][j];
            if ((flags & 2u) != 0u) v += ldF(Bias, biasOff + n * 4);
            stF(C, cOff + (m * N + n) * 4, v);
        }
    }
}
)CUDA";

const char* gemm_f16_hlsl() { return kGemmF16Hlsl; }

// ── the LoRA correction's epilogue ─────────────────────────────────────────
//
// C[m, n] += sum_r H[m, r] * B[n, r], the second half of the runtime LoRA
// correction (see `LoraTail`). Written as its own kernel rather than a second
// `gemm_f16` plus an `elem` add because the accumulator is C itself: the
// rank-r product is
//
//     out += (x @ A^T) @ B^T
//
// and `gemm_f16` has only a per-column bias, so pointing its bias at `out`
// would add out's *first row* to every row - which is what the first version of
// this did, and it turned a 0.05 % delta into a completely different model.
//
// ── why it is tiled ───────────────────────────────────────────────────────
//
// The first version was one thread per output element, reading its whole H row
// and its whole B row from global memory. That is `M * N * R` reads of B and
// `M * N * R` of H - 15 GB per call for a 28672-column module at 2048 rows, and
// B's row is 256 bytes read 4 bytes at a time (a 32-byte sector per float, 8x
// amplification), so it measured 320 ms per call. With four adapted modules per
// block over 46 streamed blocks that was 15 s of a 23 s sampling step - the LoRA
// chain's whole cost, for a correction worth 1.2 % of the block's FLOPs.
//
// This version is the ordinary GEMM tiling: a 64x64 output tile per block, each
// operand staged in shared memory once per rank chunk. The traffic becomes
// `(M/64) * N * R + (N/64) * M * R` floats for H and B - 235 MB each at the
// shape above instead of 15 GB - and C is held in registers across the whole
// rank so it is read and written exactly once.
//
// cbuffer P: uint M, N, R, cOff, hOff, bOff;
//   M, N   C's shape (row stride N)
//   R      the rank (the correction's inner dimension)
//   cOff   byte offset of C, hOff of H ([M, R]), bOff of B ([N, R])
// srv[0] = H   srv[1] = B   uav[0] = C (read-modify-write)
// numthreads(256, 1, 1) - grid (ceil(N/64), ceil(M/64))
static const char* kLoraRankAdd = R"CUDA(
struct Args { unsigned v[24]; const char* s[8]; char* u[4]; };
__device__ __forceinline__ float ldF(const char* b, unsigned o) { return __uint_as_float(*(const unsigned*)(b + o)); }
__device__ __forceinline__ void stF(char* b, unsigned o, float v) { *(unsigned*)(b + o) = __float_as_uint(v); }

static const unsigned LBM = 64;   // rows of C per block
static const unsigned LBN = 64;   // cols of C per block
static const unsigned LBR = 32;   // rank per shared stage (R is 16-128 in practice)

// Each thread owns a 4 (m) x 4 (n) patch of the tile: 16 accumulators, so the
// whole rank is accumulated in registers and C is touched twice in total.
//
// numthreads(256, 1, 1)
extern "C" __global__ void lora_rank_add(Args a) {
	const unsigned M = a.v[0], N = a.v[1], R = a.v[2];
	const unsigned cOff = a.v[3], hOff = a.v[4], bOff = a.v[5];
	const char* H = a.s[0];
	const char* B = a.s[1];
	char* C = a.u[0];

	const unsigned t = threadIdx.x;
	const unsigned ty = t >> 4u, tx = t & 15u;      // 16 x 16 thread grid
	const unsigned m0 = blockIdx.y * LBM, n0 = blockIdx.x * LBN;
	const unsigned mrow = m0 + ty * 4u, ncol = n0 + tx * 4u;

	// The rank stride is padded by one float: the inner loop reads column `r` of
	// sixteen different rows at once, and with a 32-float row pitch every one of
	// those rows lands on the same bank (row * 32 + r mod 32 == r) - a 16-way
	// conflict on every shared load. 33 floats moves each row one bank over.
	static const unsigned LRP = LBR + 1u;
	__shared__ float sH[LBM][LRP];
	__shared__ float sB[LBN][LRP];

	float acc[4][4];
	#pragma unroll
	for (unsigned i = 0; i < 4; i++)
		#pragma unroll
		for (unsigned j = 0; j < 4; j++) {
			const unsigned m = mrow + i, n = ncol + j;
			acc[i][j] = (m < M && n < N) ? ldF(C, cOff + (m * N + n) * 4u) : 0.0f;
		}

	for (unsigned r0 = 0; r0 < R; r0 += LBR) {
		__syncthreads();
		// Cooperative staging: 64 x 32 floats each, 8 per thread, rows outer so
		// a warp reads 32 consecutive floats of one row (a 128-byte transaction).
		#pragma unroll
		for (unsigned i = 0; i < (LBM * LBR) / 256u; i++) {
			const unsigned idx = t + 256u * i;
			const unsigned row = idx >> 5u, col = idx & 31u;
			const unsigned m = m0 + row, r = r0 + col;
			sH[row][col] = (m < M && r < R) ? ldF(H, hOff + (m * R + r) * 4u) : 0.0f;
		}
		#pragma unroll
		for (unsigned i = 0; i < (LBN * LBR) / 256u; i++) {
			const unsigned idx = t + 256u * i;
			const unsigned row = idx >> 5u, col = idx & 31u;
			const unsigned n = n0 + row, r = r0 + col;
			sB[row][col] = (n < N && r < R) ? ldF(B, bOff + (n * R + r) * 4u) : 0.0f;
		}
		__syncthreads();

		const unsigned rows = min(LBR, R - r0);
		for (unsigned r = 0; r < rows; r++) {
			float h[4], bb[4];
			#pragma unroll
			for (unsigned i = 0; i < 4; i++) h[i] = sH[ty * 4u + i][r];
			#pragma unroll
			for (unsigned j = 0; j < 4; j++) bb[j] = sB[tx * 4u + j][r];
			#pragma unroll
			for (unsigned i = 0; i < 4; i++)
				#pragma unroll
				for (unsigned j = 0; j < 4; j++) acc[i][j] += h[i] * bb[j];
		}
	}

	#pragma unroll
	for (unsigned i = 0; i < 4; i++)
		#pragma unroll
		for (unsigned j = 0; j < 4; j++) {
			const unsigned m = mrow + i, n = ncol + j;
			if (m < M && n < N) stF(C, cOff + (m * N + n) * 4u, acc[i][j]);
		}
}
)CUDA";

const char* lora_rank_add_hlsl() { return kLoraRankAdd; }

// ── patchify / unpatchify ─────────────────────────────────────────────────
//
// x is [C,H,W]; the token stream is [(H/p)*(W/p), p*p*C] with the last axis
// ordered (pi, pj, c) - exactly the reference's
//   x.view(C,H/p,p,W/p,p).permute(0,1,3,2,4).flatten(2)
static const char* kPatchHlsl = R"CUDA(
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
extern "C" __global__ void patchify(Args a) {
    const unsigned C = a.v[0], H = a.v[1], W = a.v[2], P = a.v[3];  // P = patch size
    const unsigned xOff = a.v[4], yOff = a.v[5];
    const unsigned total = a.v[6];
    const unsigned Wp = a.v[8];
    const char* X = a.s[0];
    char* Y = a.u[0];

    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const unsigned c = i % C;
    const unsigned rest = i / C;
    const unsigned pj = rest % P;
    const unsigned rest2 = rest / P;
    const unsigned pi = rest2 % P;
    const unsigned tok = rest2 / P;
    const unsigned wi = tok % Wp;
    const unsigned hi = tok / Wp;
    const float v = ldF(X, xOff + ((c * H + hi * P + pi) * W + wi * P + pj) * 4);
    stF(Y, yOff + i * 4, v);
}
)CUDA";

const char* patchify_hlsl() { return kPatchHlsl; }

static const char* kUnpatchHlsl = R"CUDA(
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
extern "C" __global__ void unpatchify(Args a) {
    const unsigned C = a.v[0], H = a.v[1], W = a.v[2], P = a.v[3];
    const unsigned xOff = a.v[4], yOff = a.v[5];
    const unsigned total = a.v[6];
    const unsigned Wp = a.v[8];
    const char* X = a.s[0];   // [rows, P*P*C]
    char* Y = a.u[0];         // [C,H,W]

    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;   // index into Y
    if (i >= total) return;
    const unsigned wi = i % W;
    const unsigned hi = (i / W) % H;
    const unsigned c = i / (W * H);
    const unsigned tok = (hi / P) * Wp + (wi / P);
    const unsigned pi = hi % P;
    const unsigned pj = wi % P;
    const unsigned k = (pi * P + pj) * C + c;
    stF(Y, yOff + i * 4, ldF(X, xOff + (tok * P * P * C + k) * 4));
}
)CUDA";

const char* unpatchify_hlsl() { return kUnpatchHlsl; }

}  // namespace phi::media
