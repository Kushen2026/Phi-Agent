// Native consumption of a grouped-scale 4-bit checkpoint (`asym_w4a8_int8`), and
// its six-bit sibling (`w6a8_int8`).
//
// ── what "native" means here, and why it is the right shape for this engine ─
//
// A checkpoint in one of these families stores one 4-bit code per weight, plus a
// 16-entry codebook, a per-group relative scale (fp8/fp16/bf16/fp32) and a
// per-output-row channel scale:
//
//     w[r, c] = decode(code[r, c]) * s_rel[r, c / group] * s_channel[r]
//
// ComfyUI runs these layers through the *INT8* GEMM: the packed weight, the
// group table and the codebook travel into the kernel as they are stored, and the
// kernel folds the codebook and the two scales into the operand it multiplies
// (comfy/ops.py: `_GROUPED_INT8_FORMATS` -> `AsymW4A8Int8Layout` -> the shared
// int8 linear). Nothing is ever materialised as fp32, and nothing is decoded on
// the host.
//
// This engine's GEMM is also an int8 tensor-core GEMM, so its operand needs the
// same thing: the weight as int8 codes on an int8 grid, with one fp32 scale per
// output row. This kernel is that conversion - on the device, from the stored
// bytes, with the codebook and both scales applied exactly as the checkpoint
// defines them:
//
//   * the value of a weight is decoded with the same operand order the host
//     reference uses (`codebook[code] * s_rel * s_channel`), so the fp32 value is
//     bit-identical to `SafeTensors::dequant_rows`;
//   * the int8 grid is the host quantiser's own - per-row absmax over the decoded
//     row, `scale = amax / 127`, round-half-away-from-zero in double - so the
//     codes and scales are bit-identical to `requant_rows_into` as well.
//
// The consequence is the property that makes the port checkable: the device path
// produces the *same bytes* as the CPU path it replaces (PHI_W4A8_CPU=1 forces
// that path, and the two agree bit for bit). What changes is only where the work
// happens and what crosses the bus:
//
//   * the host path decoded to fp32 on the CPU. For the H3 DiT that was 219 s of
//     a 261 s sampling step: 46 streamed blocks x 4 matrices, one row at a time,
//     through a per-element dtype switch, on *every* step (the decoded set is
//     23 GB, so no host cache can hold it and no step can reuse it);
//   * this path reads the packed codes, which are what the file stores - so the
//     bytes the checkpoint costs per step are the *packed* ones (12.6 GB for the
//     H3 DiT, against 17 GB for the same model stored as int8), and the decode is
//     one pass over them on the device.
//
// ── why it is a separate pass rather than folded into the GEMM ────────────
//
// Unpacking inside the GEMM looks cheaper but is not: a GEMM block re-reads its
// whole N-strip of the weight for every M-tile (`M / BM` blocks share it), so an
// in-kernel unpack decodes each weight `M / 256` times - 145 times at a 37 k-token
// presentation - while this pass decodes it once and leaves the int8 form in
// device memory for the GEMM to read. Measured, the extra device traffic costs
// ~0.2 s/step at the largest shape this engine runs, against ~4.6 s/step of
// redundant unpacking on the INT pipe for the same shape.
#pragma once

#include "runtime/compute.hpp"
#include "runtime/sched.hpp"

namespace phi::media {

// How a row's relative-scale table is stored. Anything else is refused by the
// caller, which falls back to the CPU path (there is none today: every float
// spelling the reader can decode is here).
enum class RelDType : int { F32 = 0, F16 = 1, BF16 = 2, F8_E4M3 = 3, F8_E5M2 = 4 };

// The code layout of the packed weight.
enum class CodeLayout : int {
	Nibble4 = 0,   // asym_w4a8_int8: two unsigned 4-bit codes per byte, low nibble first
	Packed6 = 1,   // w6a8_int8: four unsigned 6-bit codes per three bytes
};

struct W4a8ExpandArgs {
	// Every input lives in one buffer (`pool`) at the offsets below: the packed
	// codes, the per-group relative table, the per-row channel scale and the
	// codebook. One buffer means the streaming caller stages all four through the
	// staging ring with a single reservation (`w4a8_pack_rows`).
	GpuBuffer* pool = nullptr;
	u64 codes_off = 0;
	u64 srel_off = 0;
	u64 sch_off = 0;    // ignored unless has_channel_scale
	u64 cb_off = 0;     // ignored unless has_codebook
	// Destination: the [n, k] int8 codes and the [n] fp32 scales the int8 GEMM
	// reads. Separate allocations in the weight arena, so separate bindings.
	GpuBuffer* q = nullptr;
	u64 q_off = 0;
	GpuBuffer* s = nullptr;
	u64 s_off = 0;

	i64 rows = 0;   // n, the number of output rows (= the GEMM's N)
	i64 k = 0;      // logical K, the number of columns (a whole number of groups)
	i64 group = 0;  // columns per relative scale (a multiple of the code cgroup)

	// Row strides into the pool. `codes_row_bytes` is the *stored* width of one
	// row of codes (`k/2` for a correctly packed 4-bit tensor, `k*3/4` for six-bit),
	// which is what the header says and not assumed.
	i64 codes_row_bytes = 0;
	i64 srel_row_bytes = 0;

	RelDType rel_dtype = RelDType::F8_E4M3;
	CodeLayout layout = CodeLayout::Nibble4;
	bool has_channel_scale = false;
	bool has_codebook = false;
};

// Enqueues the decode + requantise. One block per output row, 256 threads.
void dispatch_w4a8_expand(ComputeContext& ctx, const W4a8ExpandArgs& a);

}  // namespace phi::media
