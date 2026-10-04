// Quantisation: int8 tensorwise weights with the convrot (Hadamard) rotation,
// plus int4 unpacking for the Qwen3-VL text encoder.
//
// convrot maths, exactly as the checkpoints were produced:
//   H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]  (= 2I - J)
//   H  = H4^{x4} / sqrt(256)                             (orthonormal, 256x256)
//   weights  (offline): W' = rot(W) along the input dim, in 256-wide groups
//   activations (online): x' = rot(x) along the last dim, same grouping
//   y = dequant(x'_int8 . W'_int8) * (s_w (*) s_x)
// Because H is orthonormal, rot(x) . rot(W)^T == x . W^T; the rotation exists so
// the int8 quantisation error is spread instead of concentrated in a few
// outlier channels.
#pragma once

#include "util/media_common.hpp"
#include "host/mtx.hpp"

namespace phi::media {

// ── metadata ───────────────────────────────────────────────────────────────

// Parsed from the "<layer>.comfy_quant" sidecar tensor (72-byte UTF-8 JSON).
//
// The shipped checkpoints use five families, all distinguished by this blob:
//
//   {"format": "int8_tensorwise",   "convrot": true, "convrot_groupsize": 256}
//        the original int8 weight: one code per element, one fp32 scale per row
//   {"format": "asym_w4a8_int8", "group_size": 16, ...}
//        4-bit *codebook indices* (two per byte, low nibble first), a 16-entry
//        fp32 codebook, a per-row fp32 scale and a per-16-group relative scale
//   {"format": "w6a8_int8",      "group_size": 32, ...}
//        six-bit codes, four per three bytes, per-row + per-32-group scales
//   {"format": "nvfp4",          "group_size": 16, ...}
//        E2M1 4-bit floats, two per byte, per-16-block E4M3 scales, one global
//        fp32 scale
//   {"format": "fp8" / "float8_e4m3fn"}  — fp8 (or a bare fp8 tensor)
//        one fp8 value per element, optional per-row or per-tensor scale
//
// Nothing else in the engine has to know: `SafeTensors::dequant_rows` turns any
// of them into fp32 rows, and the loaders requantise that to the one int8
// tensorwise + convrot form the GEMMs were written against.
struct QuantSpec {
	std::string format;        // "int8_tensorwise" | "asym_w4a8_int8" | ...
	bool convrot = false;
	int convrot_groupsize = 256;
	int group_size = 0;        // width of the *relative* scale groups (16, 32, ...)
	bool valid = false;
};

// Bits per weight on disk for a format family (8 = one code per byte).
int quant_bits_per_weight(const std::string& format);

// Parse the comfy_quant JSON blob (no exceptions: an unparsable blob yields an
// invalid spec, and the loader then treats the layer as unquantised).
QuantSpec parse_quant_spec(const void* json_bytes, size_t len);

// ── convrot ────────────────────────────────────────────────────────────────

// out = rot(x) with x shaped [rows, cols]; cols must be a multiple of `group`.
// `group` must be 256 (the only grouping used by the checkpoints).
void convrot_forward(const float* x, float* out, i64 rows, i64 cols, int group);
// rot() is an involution up to scale: inverse == forward for orthonormal H.
inline void convrot_inverse(const float* x, float* out, i64 rows, i64 cols, int group) {
	convrot_forward(x, out, rows, cols, group);
}

// ── quantisation ───────────────────────────────────────────────────────────

// Int8 tensor, either d (weights, per-output-channel scale) or dynamic
// (activations, per-row scale).
struct QuantInt8 {
	std::vector<int8_t> q;
	std::vector<float> scale;  // one entry per row (weights: per output channel)
	i64 rows = 0;
	i64 cols = 0;
	bool convrot = false;
};

// Weight quantisation: per-output-channel amax/127 scale, optional convrot.
QuantInt8 quantize_weight(const float* w, i64 rows, i64 cols, const QuantSpec& spec);
// The same maths, writing into caller-owned `q` (rows*cols) and `scale` (rows)
// instead of allocating. The streaming requantiser wants this: it decodes a
// checkpoint matrix a range of rows at a time and writes each range straight into
// its place in one destination, without a second copy of the codes. Because both
// quantisers are per-output-row by construction, the rows can also be produced by
// different threads and land in the same buffer - which is what makes a
// block-quantised checkpoint's first step cost the CPU pool instead of one core.
void quantize_weight_into(const float* w, i64 rows, i64 cols, const QuantSpec& spec,
                          int8_t* q, float* scale);
// Activation quantisation: per-row (per-token) dynamic scale, optional convrot.
QuantInt8 quantize_activation(const float* x, i64 rows, i64 cols, const QuantSpec& spec);

// Quantise an already-rotated weight (used when the checkpoint stores W' int8 +
// scale and we need the fp32 reference back).
void dequantize_int8(const QuantInt8& t, float* dst);

// ── reference GEMM ─────────────────────────────────────────────────────────

// y[m,n] = sum_k x[m,k] * w[n,k], fp32. x: [M,K], w: [N,K], y: [M,N].
void gemm_f32(const float* x, const float* w, i64 M, i64 N, i64 K, float* y);

// y[m,n] = sum_k qx[m,k]*qw[n,k] * sx[m]*sw[n]; int32 accumulation.
void gemm_int8(const int8_t* qx, const float* sx, const int8_t* qw, const float* sw,
               i64 M, i64 N, i64 K, float* y);

// ── convrot reference path (used by the self-test and the CPU fallback) ────
//
// Rotates and quantises both operands, runs the int8 GEMM and applies the
// scales: the exact maths the GPU kernels implement.
void gemm_int8_convrot(const float* x, const float* w, i64 M, i64 N, i64 K,
                       const QuantSpec& spec, float* y);

// ── int4 (Qwen3-VL text encoder) ───────────────────────────────────────────
// [W5 BEGIN] — Qwen3-VL-32B int4 extraction (format "convrot_w4a4").
// The int8 section above is W1's; nothing in this block touches it.
//
// On disk: `weight` is I8 [N, K/2] — two signed int4 per byte, low nibble is
// column 2j, high nibble is column 2j+1 (comfy_kitchen eager
// `_unpack_int4_row_major`) — and `weight_scale` is per-OUTPUT-ROW fp32 [N]
// (`quantize_signed_int4_rowwise`: absmax/7 over the last dim). The rotations are
// Hadamard: the stored nibbles are `rot(W)` with rot() applied along K in
// 256-wide groups, so the weight the GEMM wants is the stored one. The only two
// operations the engine needs are the packer below (bring a float tower onto the
// int4 path) and the GEMM itself (`gemm_rot_int4`, in the kernels).

// The packer: rotate W along K in 256-wide groups, per-row absmax/7 scale, low
// nibble = column 2j. `packed` holds rows*cols/2 bytes, `scales` rows floats.
void quantize_convrot_w4a4(const float* w, i64 rows, i64 cols, void* packed, float* scales,
                           int group = 256);
// [W5 END]

// ── packed weight families ────────────────────────────────────────────────
//
// One row of a packed weight, decoded to fp32. The callers —
// `SafeTensors::dequant_rows` and the streaming requantiser — already know which
// header/scale tensors belong to the row, so these take raw pointers and never
// look at a file.

// A scale that may be stored in any of the floats the checkpoints use.
float scale_at(const void* p, DType dtype, i64 i);

// "asym_w4a8_int8": `codes` is k/2 bytes per row, low nibble = column 2j. Each
// nibble indexes `codebook` (16 fp32 entries, index 0..15 as stored, unsigned);
// the value is codebook[idx] * s_channel * s_rel[g] with g = column / group.
//
// `codebook` is the raw 16-entry fp32 table from `weight_codebook`.
void dequant_codebook_int4_row(const u8* codes, i64 k, const float* codebook,
                               float s_channel, const void* s_rel_row, DType s_rel_dtype,
                               i64 group, float* dst);

// One three-byte group of a six-bit pack -> its four codes (code 0 in bits 0..5
// of the first byte). The layout is shared by the decoder and the payload check.
void unpack_6bit_triple(const u8* triple, u8 out[4]);

// "w6a8_int8": six-bit codes, four per three bytes, little-endian inside the
// triple (code 0 in bits 0..5, code 1 in bits 6..11, ...), so a row of k codes is
// k*3/4 bytes. value = code * s_channel * s_rel[column / group].
void dequant_packed6_row(const u8* codes, i64 k, float s_channel, const void* s_rel_row,
                         DType s_rel_dtype, i64 group, float* dst);

// "nvfp4": E2M1 4-bit floats, two per byte (low nibble = column 2j), a per-16
// block scale in fp8 and one global fp32 scale.
void dequant_nvfp4_row(const u8* codes, i64 k, const void* block_scale, DType scale_dtype,
                       float global_scale, i64 block, float* dst);

// fp8 weights: one value per element, optional per-row or per-tensor scale.
void dequant_fp8_row(const u8* codes, i64 k, DType dtype, float scale, float* dst);

}  // namespace phi::media
