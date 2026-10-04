// CUDA kernel sources, embedded as string literals so the exe needs no data
// files next to it. Each returns one kernel; its thread-group size travels as a
// `numthreads(x,y,z)` comment, which compute.cpp parses back out (CUDA has no
// such attribute, and every kernel with a `__syncthreads()` reduction depends
// on it).
//
// Every kernel documents the exact maths it implements and the host-side
// reference it must agree with (core/host/quant.*), because that agreement is
// the only correctness instrument the engine has.
#pragma once

#include <string>

namespace phi::media {

// C[M,N] = (A[M,K] int8 . B[N,K] int8) * sA[m] * sB[n].
//
// Two implementations of the one product. The dp4a one is the correctness
// reference (the engine's self-test asserts it is bit-identical to the host
// `gemm_int8`) and the fallback for shapes the tensor cores cannot take. The tc
// one is the same maths on `mma.m16n8k32.s8.s8.s32`, which on this part is
// 121 TOP/s against dp4a's 27 ceiling / 20 in situ - i.e. the difference between
// the DiT's 1.44e15 FLOP/step costing 72 s and costing 12 s at 540P/10s.
const char* int8_gemm_hlsl();
// Requires K % 32 == 0 and 16-byte-aligned operands; dispatch_int8_gemm picks.
const char* int8_gemm_tc_hlsl();

// fp32 activations [M,K] -> int8 + per-row fp32 scale, with the 256-wide
// convrot Hadamard rotation applied first.
const char* quant_convrot_hlsl();


// ── M1 image path ─────────────────────────────────────────────────────────

// one opcode-driven elementwise kernel (see kernels/ops.cpp for the op list)
const char* ops_hlsl();
// row-wise RMSNorm / LayerNorm
const char* norm_hlsl();
// GroupNorm over NCHW
const char* group_norm_hlsl();
// 2-D convolution, implicit GEMM
const char* conv2d_hlsl();
// nearest-neighbour 2x upsample
const char* upsample_hlsl();
// Wan-VAE channel-to-pixel upsample shortcut (DupUp3D, factor 2)
const char* dupup2x_hlsl();
const char* transpose_cs_hlsl();
// fp16 GEMM with fp32 accumulation
const char* gemm_f16_hlsl();
// C[m, n] += sum_r H[m, r] * B[n, r] - the low-rank LoRA correction's epilogue
const char* lora_rank_add_hlsl();
// [C,H,W] <-> [tokens, p*p*C]
const char* patchify_hlsl();
const char* unpatchify_hlsl();
// half-split RoPE (Qwen3 text encoder)
const char* rope_half_hlsl();
// DiT q/k/v split with per-head RMSNorm + axial RoPE
const char* qkv_prep_hlsl();
// DiT causal flash attention
const char* attn_flash_hlsl();
// DiT tiled attention (fp16 only): register-blocked scores, groupshared
// probability tile, wave-shuffle softmax reduction. Hot path for the DiT.
const char* attn_tiled_hlsl();
// The same kernel with a different query tile: `bm` rows per block, which the
// source rewrites as (BM, NT) together (NT = 32 * bm/16, one 16-row tile per
// warp). The kernel's K/V traffic is one pass over the whole key sequence per
// query *block*, so the bytes it moves per flop are proportional to `bm` and the
// tile is the lever when the sequence is long enough that K/V no longer fits in
// L2 - at 540P/5s it is 9.9 MB per head against a 3 MB L2. The arithmetic does
// not depend on the tile (each query row owns its online-softmax state and walks
// the key tiles in the same order), so two tiles produce the same bits.
const char* attn_tiled_src(unsigned bm, unsigned hd);
// The pipeline name that goes with `attn_tiled_src`. It carries the tile, the
// head dim *and* the two experiment overrides (PHI_ATTN_BN, PHI_ATTN_BLOCKS),
// because `ComputeContext::pipeline` keys on the name alone and would otherwise
// hand back the first cubin it compiled for any of them.
std::string attn_tiled_name(unsigned bm, unsigned hd);
// The query tile every tiled-attention caller uses (PHI_ATTN_BM overrides).
unsigned attn_query_tile();
// channel-wise RMSNorm over NCHW activations (the Wan VAEs' RMS_norm)
const char* chw_rmsnorm_hlsl();
// row-wise fp32 softmax (the Wan VAEs' spatial attention)
const char* row_softmax_hlsl();
// zero a [C,H,W] fp32 tensor outside a rectangle (the tiled VAE decode's
// "defined only on the image" boundary)
const char* mask_rect_hlsl();

// ── causal 1-D convolution (the ACE-Step 1.5 audio VAE decoder) ─────────────

// torch Conv1d as a gather: y[oc,t] = bias + sum_ic sum_j x[ic, t*stride - pad +
// j*dilation] * w[oc,ic,j]. The host reference is `Conv1dRef::forward`.
const char* conv1d_gather_hlsl();
// torch ConvTranspose1d with k == 2*stride, as two gathered taps per output:
// y[oc,t] = bias + sum_ic ( x[ic,i0]*w[ic,oc,r] + x[ic,i0-1]*w[ic,oc,r+stride] ),
// r = (t+pad) mod stride. The host reference is `ConvT1dRef::forward`.
const char* convt1d_hlsl();
// SnakeBeta with pre-exp()'d per-channel alpha/beta. The host reference is
// `kernels::snake_beta_f32` (core/kernels/snake.cpp).
const char* snake_beta_hlsl();

}  // namespace phi::media
