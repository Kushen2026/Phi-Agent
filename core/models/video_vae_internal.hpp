// Internal (non-frozen) declarations for the H3 video VAE kernels and helpers.
//
// `core/models/video_vae.hpp` is a frozen interface: its `VideoVae` class may
// not gain members or change signatures until G1. Everything W2 needed to add
// (extra kernels, weight-streaming state, the temporal/space tiling helpers)
// lives here instead. Nothing in this file is consumed by another agent today;
// W0 decides at G1 whether any of it graduates into `gpu_ops.hpp`.
#pragma once

#include <string>
#include <vector>

#include "util/media_common.hpp"
#include "runtime/sched.hpp"   // ComputeContext, GpuAlloc

namespace phi::media {

// HLSL entry points implemented in core/kernels/conv3d_causal.cpp.
const char* conv3d_causal_hlsl();   // entry "conv3d_causal"
const char* causal_gn_hlsl();       // entry "causal_gn"
const char* qkv_split3d_hlsl();     // entry "qkv_split3d"
const char* rope3d_hlsl();          // entry "rope3d" (source: kernels/rope3d.hlsl)
const char* attn3d_hlsl();          // entry "attn3d" (the staged v1, kept for the test)
const char* attn3d_v2_hlsl();       // entry "attn3d_v2" (staged, register-blocked)
const char* unpatch3d_hlsl();       // entry "unpatch3d"
const char* silu_gate_fused_hlsl();  // entry "silu_gate_fused"

// Non-causal attention over q/k/v [S, heads, 64] fp32, `impl` selecting the
// kernel (2 = attn3d_v2, the default; 1 = the original attn3d - see the doc
// comments in kernels/conv3d_causal.cpp). Exposed so tests/vae_attn_bench.cpp can
// time it without the host-side upload the test wrapper does.
void dispatch_attn3d(ComputeContext& ctx, const GpuAlloc& q, const GpuAlloc& k, const GpuAlloc& v,
                     const GpuAlloc& o, i64 S, int impl = 2);

// Tensor-core fp16 GEMM (core/kernels/gemm_f16_mma.cpp). The decoder's
// four per-block projections are ~75% of a block's GPU time and ran at 1.4
// TFLOP/s through the shared SIMT `gemm_f16`; these two entries are the same
// product on `mma.m16n8k16` (~10x on this part). They are separate entry points
// so `gemm_f16`'s accumulation order stays pinned for the text encoder and the
// image paths.
const char* gemm_f16_mma_hlsl();   // entry "gemm_f16_mma"
const char* f32_to_f16_hlsl();     // entry "f32_to_f16"

// A [m,k] fp32 -> fp16 into `y` (the mma kernel's A operand).
void dispatch_f32_to_f16(ComputeContext& ctx, const GpuAlloc& x, const GpuAlloc& y, i64 count);
void dispatch_attn3d_tc(ComputeContext& ctx, const GpuAlloc& q, const GpuAlloc& k, const GpuAlloc& v,
                        const GpuAlloc& o, i64 S);
// C[m,n] = A[m,k] fp16 . B[n,k] fp16 + bias[n]. Requires k % 16 == 0, k >= 16
// and 16-byte-aligned operands (all three are checked by the caller).
void dispatch_gemm_f16_mma(ComputeContext& ctx, const GpuAlloc& a, const GpuAlloc& b,
                           const GpuAlloc& bias, const GpuAlloc& c, i64 m, i64 n, i64 k);

// The shared 3-D RoPE HLSL helpers (kernels/rope3d.hlsl); prefixed to any source
// that needs the same rotation maths as the standalone `rope3d` shader.
const char* rope3d_helpers_hlsl();

// ── temporal / spatial tiling plan (reference semantics) ──────────────────
//
// These mirror comfy/ldm/minimax/vae.py exactly: `split_tiles` (vae.py:424),
// `_decode_temporal_chunks` (vae.py:597) and `_decode_temporal_frame_plan`
// (vae.py:573). They are exposed so the test can check the plan against the
// reference's numbers without running a decode.
struct H3Tiles {
	std::vector<i64> pos;      // tile start offsets, in pixels
	std::vector<i64> len;      // tile extents, in pixels
	std::vector<i64> overlap;  // overlap with the *next* tile, in pixels
};

// tile_size 256 / overlap_min 64 / ratio 16 are the reference's own constants.
H3Tiles h3_split_tiles(i64 input_len, i64 tile_size = 256, i64 tile_overlap_min = 64,
                       i64 ratio = 16);

// (pad_tokens, num_chunks) for a latent sequence of `z_len` tokens.
void h3_decode_temporal_chunks(i64 z_len, i64 tokens_chunk_size, i64 token_drop, i64* pad_tokens,
                               i64* num_chunks);

// Number of output frames `decode_temporal` will produce (and therefore the
// canvas size), i.e. `_decode_temporal_frame_plan` minus the dropped pad frames.
i64 h3_decode_output_frames(i64 z_len, i64 pad_tokens, i64 num_chunks, i64 clip_length,
                            i64 tokens_chunk_size, i64 token_drop, i64 vae_ratio_t);

// [rows, 2*cols] fused-gate activation: out[r][c] = silu(gx[r][c]) * gx[r][cols + c].
// The HLSL (kernels/conv3d_causal.cpp) and this dispatch are shared with the H3
// DiT, whose per-block MLP has the same fused gate/value layout.
void dispatch_silu_gate_fused(ComputeContext& ctx, const GpuAlloc& gx, const GpuAlloc& out,
                              i64 rows, i64 cols);

// ── test-only entry points ────────────────────────────────────────────────
//
// The kernels are compiled from HLSL strings owned by video_vae.cpp; these
// wrappers run each one on host tensors and compare it with an independent host
// reference (the engine's usual kernel conformance pattern). They are not used
// by the model code.
struct GpuCtx;
class GpuArena;

// Fused-gate activation: out[r][c] = silu(gx[r][2c]) * gx[r][2c+1] over [rows, 2*cols].
void h3vae_test_silu_gate_fused(GpuCtx& g, GpuArena& arena, const std::vector<float>& gx,
                                std::vector<float>& out, i64 rows, i64 cols);

// Causal 3-D convolution. w is [oc, ic, kt, kh, kw] fp16, x is [ic,IT,IH,IW] fp32,
// y is [oc,OT,OH,OW] fp32. OT/OH/OW follow the reference's padding arithmetic.
//
// `padH` is the reflect pad on the left/top and `padHR` the one on the
// right/bottom: CausalConv3d pads both sides, Downsample3D only the right (that
// asymmetry is a bug this engine has already had once, so the test needs to be
// able to express both).
std::vector<float> h3vae_test_conv3d(GpuCtx& g, GpuArena& arena, const std::vector<u16>& w,
                                     const std::vector<float>& bias, const std::vector<float>& x,
                                     i64 ic, i64 IT, i64 IH, i64 IW, i64 oc, i64 kt, i64 kh,
                                     i64 kw, i64 strT, i64 strS, i64 padT, i64 padH,
                                     i64 padHR = 0);

// Temporal-isolated GroupNorm over x [C,T,H,W].
std::vector<float> h3vae_test_causal_gn(GpuCtx& g, GpuArena& arena, const std::vector<float>& x,
                                        const std::vector<float>& w, const std::vector<float>& b,
                                        i64 C, i64 T, i64 H, i64 W);

// 3-D RoPE in place on x [S*heads, D].
std::vector<float> h3vae_test_rope3d(GpuCtx& g, GpuArena& arena, const std::vector<float>& x,
                                     const std::vector<float>& ids, i64 S, i64 heads);

// Non-causal attention over q/k/v [S, heads, 64] fp32. `impl` picks between
// `attn3d_v2` (2, the decoder's kernel) and the original `attn3d` (1), so a test
// can hold the fast kernel against the slow one on the same tensors.

std::vector<float> h3vae_test_attn3d(GpuCtx& g, GpuArena& arena, const std::vector<float>& q,
                                     const std::vector<float>& k, const std::vector<float>& v,
                                     i64 S, int impl = 2);

// Decoder token stream -> [3, T*4, H*16, W*16].
std::vector<float> h3vae_test_unpatch3d(GpuCtx& g, GpuArena& arena, const std::vector<float>& tok,
                                        i64 T, i64 H, i64 W);

}  // namespace phi::media
