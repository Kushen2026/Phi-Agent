// Shared plumbing for the model implementations: device arenas, weight upload
// and the small conversion helpers.
//
// Memory model (M1): two bump arenas. `weights` is reset once per layer, so a
// 30-layer DiT streams its 6 GB through a window of one layer; `acts` holds the
// activations and is reset once per sampling step. Everything is uploaded
// through one pinned ring so a layer's weights cost a memcpy plus one copy
// command rather than a committed resource per tensor.
#pragma once

#include <string>
#include <vector>

#include "kernels/gpu_ops.hpp"
#include "runtime/sched.hpp"
#include "host/st.hpp"

namespace phi::media {

class LoraSet;

// One linear weight, in whatever precision the checkpoint stores it, together
// with the execution path that precision implies.
//
// The engine no longer re-quantises a float checkpoint into int8: the file's own
// precision decides the path (goal #1). Two shapes live here:
//
//   * the int8 form - [N,K] int8 plus [N] fp32 scales - which is byte-for-byte
//     the long-standing tensor-core path. It carries the shipped int8 tensorwise
//     checkpoints *and* the packed families (w4a8 / w6a8 / nvfp4), because those
//     are expanded onto the device into exactly these two tensors (see
//     `w4a8_pack_rows`), which is the model's own format and not a conversion;
//   * the dense float form - `d` holding the raw [N,K] f16 / bf16 / fp32 matrix
//     the file stores (fp8 is decoded to f16, which is exact) - consumed by the
//     dense `dispatch_gemm_f16` path. `w`/`scale` are empty in that case.
struct QuantLinear {
	GpuAlloc w;      // int8 matrix (the int8 GEMM's operand)
	GpuAlloc scale;  // int8 per-row scale
	// Dense float storage: set when the checkpoint's own precision is
	// fp16 / bf16 / fp32 (or fp8, decoded to fp16). Mutually exclusive with `w`.
	GpuAlloc d;
	DType dtype = DType::Unknown;   // F16 / BF16 / F32 when dense, else Unknown
	i64 n = 0, k = 0;

	bool dense() const {
		return dtype == DType::F16 || dtype == DType::BF16 || dtype == DType::F32;
	}
};

struct GpuCtx {
	CudaContext* device = nullptr;
	ComputeContext* ctx = nullptr;
	GpuArena* wa = nullptr;    // weights, reset per layer
	GpuArena* aa = nullptr;    // activations, reset per step
	GpuArena* keep = nullptr;  // long-lived buffers (embedding tables, ...)
	UploadRing* ring = nullptr;

	bool ok() const { return ctx && wa && aa && ring; }

	GpuAlloc walloc(u64 bytes) { return wa->alloc(bytes); }
	GpuAlloc aalloc(u64 bytes) { return aa->alloc(bytes); }
	GpuAlloc kalloc(u64 bytes) { return (keep ? keep : aa)->alloc(bytes); }

	// Upload raw bytes (already in the format the kernel wants).
	GpuAlloc upload_raw(const void* src, u64 bytes, bool to_weights_arena = true);
	// Same, into a caller-owned allocation (the `keep` arena's long-lived tables).
	void upload_into(const GpuAlloc& a, const void* src, u64 bytes);
	// A tensor uploaded straight off the checkpoint file rather than out of its
	// mapping: the bytes come through the pinned host cache / the look-ahead slot
	// and are DMA'd from the ring. Returns the byte count. `SafeTensors` is not
	// named here to keep this header light; see the definition for why the
	// difference matters (~3-10x on a cached range).
	u64 upload_file_into(const GpuAlloc& a, const SafeTensors& st, const StTensor& t);
	// Upload fp32 values.
	GpuAlloc upload_f32(const float* src, u64 count, bool to_weights_arena = true);
	// bf16 source -> the device keeps bf16 (the GEMM reads it directly)
	GpuAlloc upload_bf16(const void* src, u64 count, bool to_weights_arena = true);

	std::vector<float> download_f32(const GpuAlloc& a, u64 count);
	std::vector<float> download_f16(const GpuAlloc& a, u64 count);

	// Copy a safetensors tensor to the device, converting bf16/f16/f32 to the
	// dtype the kernels expect (`as`), or leaving int8/uint8 alone.
	GpuAlloc upload_tensor(const SafeTensors& st, const StTensor& t, DType as);
	// Same, into a caller-owned allocation (a named table in the `keep` arena).
	void upload_tensor_into(const SafeTensors& st, const StTensor& t, DType as, const GpuAlloc& dst);

	void begin();                 // begin a command list
	void submit();                // close, execute, wait
	void new_layer();             // begin() + weights arena reset + ring rewind
	void end_layer();             // submit() + ring rewind
	void new_step();
};

// fp32 -> fp16 conversion of a whole buffer (host side).
std::vector<u16> to_f16(const float* src, size_t count);

// bf16/f16/f32 tensor -> fp32 host vector
std::vector<float> tensor_to_f32(const SafeTensors& st, const StTensor& t);

// ── unified weight loading (any stored precision) ───────────────────────────
//
// One quantised linear, whatever the checkpoint stores and however it is packed:
//
//   * safetensors int8_tensorwise (+ optional convrot): uploaded verbatim, so the
//     shipped int8 checkpoints cost exactly what they did before this helper;
//   * an f16/bf16/f32 matrix: decoded to fp32 and re-quantised to the same
//     int8_tensorwise + convrot form, so every consumer in the engine keeps
//     running the one int8 tensor-core GEMM it was written against;
//   * a source whose rows are not a multiple of the convrot group (256) is
//     refused with the shape, because the rotator cannot be applied to it - the
//     few matrices in these models with such a K (the latent/embed projections)
//     are loaded as bf16 by the callers that already do so.
//
// `lora`, when non-null and touching the module, is merged into the fp32 weight
// before it is quantised - for an int8 checkpoint that means dequantise, rotate
// back to the original basis, add the delta, and re-quantise.
//
// The spec to requantise a *decoded* weight with, which decides whether the
// requantiser rotates. Exposed because getting it wrong is silent: rot() is an
// involution, so rotating a weight that the file already stored rotated (every
// `"convrot": true` checkpoint, i.e. all the shipped ones) hands the GEMM a
// different matrix of the same size - the model then answers with noise and
// nothing anywhere reports an error. `media_packed_quant_test` pins it down.
QuantSpec requant_spec_for(const SafeTensors& st, const std::string& base);

// ── the dense float loader (the path that replaces the float->int8 requantise) ──
//
// fp16 / bf16 / f32 are uploaded in their own precision (a raw tensor is streamed
// straight off the file, byte for byte); fp8 is decoded to f16, which is exact
// because fp8 has fewer significand bits than f16. The upload the same slab-at-a-
// time discipline the int8 streamer uses, so a streamed DiT block keeps a bounded
// host working set instead of materialising the whole matrix.
struct DenseUpload {
	GpuAlloc w;     // [n, k] in `dtype`'s element size
	i64 n = 0, k = 0;
	DType dtype = DType::F16;
};

DenseUpload upload_dense_linear(GpuCtx& g, const SafeTensors& st, const std::string& base,
                                GpuArena& arena);
DenseUpload load_dense_linear(GpuCtx& g, const SafeTensors& st, const std::string& base,
                              GpuArena* into, const LoraSet* lora);
// The fused q/k/v dense form: one [3*N, K] matrix (q rows then k then v), the same
// row layout `qkv_prep` reads from the int8 path.
DenseUpload load_dense3_linear(GpuCtx& g, const SafeTensors& st, const std::string& base_q,
                               const std::string& base_k, const std::string& base_v,
                               GpuArena* into, const LoraSet* lora);

// The precision-preserving streaming upload one DiT block uses: int8 tensorwise
// and the packed families come back as the int8 pair (unchanged); a float source
// comes back as its own f16/bf16/f32.
QuantLinear upload_linear_any(GpuCtx& g, const SafeTensors& st, const std::string& base,
                              GpuArena& arena, const LoraSet* lora);

// The one GEMM the engine runs for a loaded linear, whatever its precision:
//
//   int8 tensorwise / a packed family -> rotate + quantise the activation, the
//       int8 tensor-core GEMM (`q8`/`s8` are the activation scratch);
//   dense f16 / bf16 / f32 -> the dense `dispatch_gemm_f16`.
//
// `x` is the fp32 activation either way (the runtime LoRA correction binds to
// this same `x`), `out` is [m, n] fp32.
void linear_gemm(GpuCtx& g, const QuantLinear& w, const GpuAlloc& x, i64 m, i64 k,
                 const GpuAlloc& out, const GpuAlloc& q8, const GpuAlloc& s8);

// `base` is the tensor name without the `.weight` suffix.
QuantLinear load_quant_linear(GpuCtx& g, const SafeTensors& st, const std::string& base,
                              GpuArena* into = nullptr, const LoraSet* lora = nullptr);

// The fused q/k/v form: one [3*N,K] int8 matrix whose rows are q then k then v, so
// a single GEMM produces the interleaved layout `qkv_prep` reads. Each of the
// three may be int8 (raw) or any other precision (quantised on load); the
// combined scales are the concatenation of the three.
QuantLinear load_quant3_linear(GpuCtx& g, const SafeTensors& st, const std::string& base_q,
                               const std::string& base_k, const std::string& base_v,
                               GpuArena* into = nullptr, const LoraSet* lora = nullptr);

// ── streaming weight upload ─────────────────────────────────────────────────
//
// The native way to bring a checkpoint matrix onto the engine's quantised GEMM
// paths. Both helpers read the matrix out of the file a slab of rows at a time
// (`SafeTensors::dequant_rows`), convert exactly that slab, and upload exactly
// that slab - so the host working set is a bounded buffer instead of the whole
// fp32 expansion of the matrix.
//
// That matters most for the big files: a 21 GB block-quantised checkpoint read
// through the old path materialised its largest matrix as fp32 first (4 bytes per
// element, 490 MB -> 2.0 GB at this model's widths) and then requantised it,
// while every other precision pays the same extra copy. The slab loop is also the
// faster of the two: the slab is in cache when it is rotated, instead of being
// written out and read back.
//
// Both are bit-identical to the whole-matrix calls they wrap, because both
// quantisers are per-output-row by construction (per-row amax; convrot along the
// row in 256-wide groups).

// int8 tensorwise + convrot, into `arena`. `lora`, when it touches `base`, forces
// the whole-matrix path: a LoRA delta is dense over [n,k] and cannot be folded in a
// row slab.
struct I8Upload {
	GpuAlloc w, s;
	i64 n = 0, k = 0;
};

I8Upload upload_quant_linear_i8(GpuCtx& g, const SafeTensors& st, const std::string& base,
                                GpuArena& arena, const LoraSet* lora);

// convrot_w4a4 (signed int4, two per byte, plus one fp32 scale per output row) -
// the Qwen3-VL text tower's format. Same slab discipline.
struct I4Upload {
	GpuAlloc w, s;
	i64 n = 0, k = 0;
};
I4Upload upload_quant_linear_w4a4(GpuCtx& g, const SafeTensors& st, const std::string& base,
                                  GpuArena& arena, const LoraSet* lora);

// ── the runtime LoRA correction ────────────────────────────────────────────
//
// Folding a LoRA into the weight is fine for a matrix that is loaded once. It is
// the wrong shape for a *streamed* one: the H3 DiT keeps a handful of its 50
// blocks resident, so every step re-derives the other blocks' fp32 form, adds the
// delta and requantises - a decode of the whole 21 GB checkpoint per sampling
// step. Measured on this machine that was 330 s a step against 17 s without the
// LoRA, i.e. the chain could not finish inside a tool call at all.
//
// The delta does not have to go through the weight. `y = x @ (W + B@A)^T` is the
// same product as `x @ W^T + (x @ A^T) @ B^T`, and the second term needs the
// base weight in its *original* form - which for every shipped int8 checkpoint is
// exactly the bytes the file already holds, uploaded verbatim. The correction is
// two ordinary GEMMs whose inner dimension is the rank (64 here), so it costs a
// percent or two of the base GEMM, and it is *more* accurate than folding: the
// delta stays fp32 instead of being rounded into the int8 weight.
//
// `at` is every factor's A stacked along the rank axis, `bt` every factor's B
// with its scale folded in, so N factors compose as one rank-sum correction.
struct LoraTail {
	bool active = false;
	i64 rank = 0;
	i64 n = 0, k = 0;
	GpuAlloc at;   // [rank, k] fp32: the factors' A rows, stacked
	GpuAlloc bt;   // [n, rank] fp32: the factors' B columns, scaled, stacked
};

// Empty when `lora` is null or holds no factor whose (n, k) is exactly the
// module's, so an untouched module is the graph it was.
//
// One module of a fused projection: `base` (a tensor name without `.weight`) is
// corrected over the output rows [row0, row0+n) of the layer's [total_n, k]
// weight. The image DiT's q|k|v is three of these stacked into one matrix and
// one GEMM, so the correction has to be assembled the same way.
struct LoraModule {
	std::string base;
	i64 row0 = 0;
	i64 n = 0;
	const LoraSet* lora = nullptr;
};

LoraTail load_lora_tail(GpuCtx& g, GpuArena& arena, const std::vector<LoraModule>& mods, i64 n,
                        i64 k);
// The one-module form, which is every non-fused projection.
LoraTail load_lora_tail(GpuCtx& g, GpuArena& arena, const std::string& base, const LoraSet* lora,
                        i64 n, i64 k);

// `out[m, n] += (x @ A^T) @ B^T`. `x` is the same fp32 [m, k] activation the
// int8 GEMM was given and `out` is that GEMM's [m, n] result, so this is the
// epilogue of that GEMM and not a second pipeline. `h` is the [., rank] scratch
// the caller owns (see `Fwd::lora_h`); its capacity is checked, because a short
// scratch here would be a silent overwrite of whatever follows it.
void apply_lora_tail(GpuCtx& g, const LoraTail& t, const GpuAlloc& x, i64 m, const GpuAlloc& out,
                     const GpuAlloc& h);

// Host memory this process may spend to avoid re-deriving a weight: the pinned
// file-range cache and the requantised-block cache share it. Two thirds of what
// is free right now, capped at 9 GB; PHI_DIT_HOSTCACHE_MB overrides it, and 0
// disables both caches (the switch for "what does the disk alone do"). Defined in
// av_dit.cpp, where the measurement is documented.
u64 host_cache_budget_bytes();
// The RAM the pin loops must leave alone (see the note in model_common.cpp).
u64 host_cache_reserve_bytes();

}  // namespace phi::media
