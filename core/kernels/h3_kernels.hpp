// Two kernels the H3 DiT needs and the image DiT does not.
//
// Both exist because the H3 attention differs from the image DiT's in ways its
// `qkv_prep` / `attn_flash` pair cannot express, and rewriting one of them with
// another flag is cheaper than pretending the shapes agree:
//
//   * the H3 RoPE is not axial-interleaved. `rope.inv_freq` is 16 wide, the head
//     is 128 wide of which the first 96 dims rotate as three 16-frequency axes
//     (t/h/w, 48 angles), and the rotation pairs (i, i+48) — the HF "split
//     half" convention, not (2i, 2i+1). The angles were already computed on the
//     host (`layout.position_ids` is fp64 in the reference, so the table is
//     built there too) and arrive as (cos, sin) pairs.
//   * the H3 attention is *bidirectional* over the packed sequence and its S can
//     be 20k+ tokens. `attn_flash` uses S for both the query rows and the key
//     length, so a query-range chunk (which the TDR watchdog requires: one
//     command list may not hold tens of seconds of work) cannot be expressed.
//     These kernels take an explicit [q0, q0+Sq) query window over an Sk-long
//     key/value sequence; the host walks the window.
//
// Not frozen; new file. `core/kernels/gpu_ops.hpp` is untouched so the image path
// keeps its own (unchanged) kernels.
#pragma once

#include "runtime/sched.hpp"

namespace phi::media {

// Fused qkv [S, 3*H*D] fp32 -> q/k/v [S, H*D] fp32, with the per-head RMSNorm
// (separate q/k weights), the split-half rotation over the first `rot` dims, and
// q scaled by 1/sqrt(head_dim). v is copied unchanged.
struct H3QkvPrepArgs {
	GpuAlloc qkv;       // [S, (3*H)*head_dim] fp32
	GpuAlloc wq, wk;    // [head_dim] fp32
	GpuAlloc rope;      // [S, rot/2, 2] fp32 — (cos, sin) per pair
	GpuAlloc q, k, v;   // [S, H, head_dim]; fp32, or packed fp16 when f32_out=false
	i64 s = 0;
	// Row windows. The DiT runs its qkv projection in row chunks so the fused
	// fp32 [rows, 3*H*D] intermediate never scales with the sequence length:
	// `qkv_row0` is this chunk's first row inside the fused buffer (always 0 for a
	// chunk-local scratch), `out_row0` its first row in q/k/v (and therefore the
	// first row of the rope table this chunk reads). Both are 0 for a whole-buffer
	// call, which is what the token refiner and the tests use.
	i64 qkv_row0 = 0;
	i64 out_row0 = 0;
	i64 heads = 0;
	i64 head_dim = 128;
	i64 rot = 96;       // rotated dims, even; rot/2 must be <= head_dim/2
	float eps = 1e-5f;
	float scale = 1.0f;   // 1/sqrt(head_dim) for q; 1.0 for k
	// Store q/k/v as packed fp16. The H3 attention is the tiled fp16 kernel
	// (attn_tiled), which reads a word per two elements instead of one element
	// per multiply-accumulate; fp32 output is only needed by the old
	// one-load-per-FMA H3 kernel.
	bool f32_out = true;
};

void dispatch_h3_qkv_prep(ComputeContext& ctx, const H3QkvPrepArgs& a);

// Bidirectional attention over [sk] keys for query rows [q0, q0+sq).
//
// Two kernels: the original fp32 one (one load per multiply-accumulate, kept as
// the reference and as the fallback) and the tiled fp16 one the DiT uses, which
// needs `fp16_qkv` and reads q/k/v as packed halves.
struct H3AttnArgs {
	GpuAlloc q, k, v;      // [sk, H, head_dim] fp32, or packed fp16 when fp16_qkv
	GpuAlloc o;            // [sk, H, head_dim] fp32 (both kernels)
	i64 sk = 0;            // total sequence length (query and key rows)
	i64 q0 = 0;            // first query row of this dispatch
	i64 sq = 0;            // query rows in this dispatch
	i64 heads = 0;
	i64 head_dim = 128;
	float scale = 0.0f;    // 0 => 1/sqrt(head_dim)
	bool fp16_qkv = false; // q/k/v are packed fp16 (uses the tiled kernel)
	// First *output* row this dispatch writes, when it is not the same as `q0`.
	//
	// The kernel writes O at the absolute query row, which is what lets one
	// dispatch cover a query window. When the caller wants a window's output in a
	// buffer that is only `sq` rows tall (the DiT does: the attention output is
	// consumed by out_proj one chunk at a time, and an S-tall buffer is 1.08 GB
	// at 540P/10s), it passes o_row0 = q0 and the dispatch subtracts that many
	// rows' stride from the output byte offset. The arithmetic is unsigned so the
	// wrap is exact, and the kernel's `row < S` guard still refers to the input.
	i64 o_row0 = 0;
};

void dispatch_h3_attn(ComputeContext& ctx, const H3AttnArgs& a);

// ── int4 GEMM for the Qwen3-VL text encoder ────────────────────────────────
//
// C[M,N] = (A[M,K] int8 . B[N,K/2] packed-int4) * sA[m] * sB[n].
//
// The text encoder's weights are `convrot_w4a4`: two signed int4 per byte, low
// nibble first, along K, and the activations were quantised over the same 4-bit
// range (absmax/7). Unpacking to int8 in the kernel (rather than on the host, or
// once at load time) is what keeps 15 GB of checkpoint from being expanded to
// 30 GB of traffic: the tile is unpacked in shared memory, four dp4a operands at
// a time, exactly like the int8 kernel's B tile.
struct Int4GemmArgs {
	GpuAlloc a;       // [M,K] int8 (rotated + quantised activations)
	GpuAlloc b;       // [N,K/2] packed int4
	GpuAlloc sa;      // [M] fp32
	GpuAlloc sb;      // [N] fp32
	GpuAlloc c;       // [M,N] fp32
	i64 m = 0, n = 0, k = 0;
};

void dispatch_int4_gemm(ComputeContext& ctx, const Int4GemmArgs& a);

// Head-dim RMSNorm + split-half rotation, as a *reference* the kernels can be
// checked against on the host (the kernel suite's --self-test). Mirrors the HLSL
// arithmetic in the same order so a mismatch is a real one.
void h3_rope_ref(const float* qkv, float* q, float* k, float* v, const float* wq,
                 const float* wk, const float* rope, i64 s, i64 heads, i64 head_dim,
                 i64 rot, float eps, float scale);

// ── fused-gate activation over a [rows, 2*cols] buffer ─────────────────────
//
// out[r][c] = silu(gx[r][c]) * gx[r][cols + c]. Both H3 models emit the FFN's
// gate and value side by side in one buffer (`gate, x = w1(x).chunk(2, dim=-1)`),
// and the generic `elem` op cannot express that pairing: it reads each operand
// *flat* from its own base offset, so one dispatch pairs a[i] with b[cols + i],
// which is right for row 0 and wrong for every other row. For the DiT that is
// every block's MLP (and the token refiner's), i.e. the whole network.
//
// The HLSL lives in kernels/conv3d_causal.cpp (`silu_gate_fused`, shared with the
// H3 video VAE, which hit the same trap) and the dispatch is defined in
// video_vae.cpp.
void dispatch_silu_gate_fused(ComputeContext& ctx, const GpuAlloc& gx, const GpuAlloc& out,
                              i64 rows, i64 cols);

}  // namespace phi::media
