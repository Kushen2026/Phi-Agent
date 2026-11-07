#include "host/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>

#ifndef _WIN32
#define _aligned_free(p) free(p)
#endif

#include "util/json.hpp"

namespace phi::media {

// ── comfy_quant metadata ───────────────────────────────────────────────────

QuantSpec parse_quant_spec(const void* json_bytes, size_t len) {
	QuantSpec spec;
	if (!json_bytes || len == 0) return spec;
	std::string text((const char*)json_bytes, len);
	// the blob is a fixed-size padded buffer; cut at the closing brace
	size_t end = text.find_last_of('}');
	if (end == std::string::npos) return spec;
	text.resize(end + 1);
	std::string err;
	auto parsed = json_parse(text, &err);
	if (!parsed) return spec;
	const JsonValue& j = *parsed;
	spec.format = j["format"].as_string("int8_tensorwise");
	spec.convrot = j["convrot"].as_bool(false);
		spec.convrot_groupsize = (int)j["convrot_groupsize"].as_int(256);
	if (spec.convrot_groupsize <= 0) spec.convrot_groupsize = 256;
	spec.group_size = (int)j["group_size"].as_int(0);
	if (spec.group_size < 0) spec.group_size = 0;
	spec.valid = true;
	return spec;
}

// The four bit-packed families (4/6 vs 8 bits per weight). Only the *payload*
// width is derived from this; the per-group scale width comes from
// `group_size`, which the packer writes next to it.
int quant_bits_per_weight(const std::string& format) {
	// NB: "convrot_w4a4"/"int4" are deliberately *not* here. The Qwen3-VL-32B
	// tower's packed int4 has its own loader and its own payload width (two values
	// per byte along K, which its reader already walks), so it is not one of the
	// families this generic packed path decodes.
	if (format == "asym_w4a8_int8" || format == "nvfp4" || format == "mxfp4") return 4;
	if (format == "w6a8_int8" || format == "w6a8") return 6;
	return 8;
}

// ── convrot ────────────────────────────────────────────────────────────────

// H = H4^{x4} / 16 with H4 = J - 2R (R the 4-point reversal), i.e.
// H4[x][y] = -1 when y == 3 - x and +1 otherwise. That is the Hadamard the
// checkpoint was rotated with: a 4-wide "digit" of the 256-element index carries
// one H4 factor, so the transform factorises into four 4-point butterfly passes
// (y[j] = sum(x) - 2*x[3-j]), which is what convrot_forward runs. Nothing here
// builds the 256x256 matrix: the factorised form is the same linear map with 64x
// fewer multiplies and no 256 KB lookup table.
//
// The 2I - J variant is also a Hadamard matrix, but a *different* one: rotating
// the activation with it and then contracting against a weight that was rotated
// with J - 2R gives an unrelated answer, not a rounding difference.
void convrot_forward(const float* x, float* out, i64 rows, i64 cols, int group) {
	if (group != 256) {
		throw MediaError("convrot: unsupported group size " + std::to_string(group));
	}
	if (cols % group != 0) {
		throw MediaError("convrot: cols=" + std::to_string(cols) + " not a multiple of " +
		                 std::to_string(group));
	}
	const i64 groups = cols / group;
	for (i64 r = 0; r < rows; r++) {
		const float* src = x + r * cols;
		float* dst = out + r * cols;
		for (i64 g = 0; g < groups; g++) {
			const float* in = src + g * group;
			float* o = dst + g * group;
			memcpy(o, in, (size_t)group * sizeof(float));
			// four butterfly passes over index digits d = 0..3 (stride 4^d)
			for (int d = 0, stride = 1; d < 4; d++, stride *= 4) {
				for (int base = 0; base < group; base += stride * 4) {
					for (int off = 0; off < stride; off++) {
						float* p = o + base + off;
						float x0 = p[0], x1 = p[stride], x2 = p[stride * 2], x3 = p[stride * 3];
						float s = x0 + x1 + x2 + x3;
						p[0] = s - 2 * x3;
						p[stride] = s - 2 * x2;
						p[stride * 2] = s - 2 * x1;
						p[stride * 3] = s - 2 * x0;
					}
				}
			}
			const float inv = 1.0f / 16.0f;
			for (int i = 0; i < group; i++) o[i] *= inv;
		}
	}
}

// ── quantisation ───────────────────────────────────────────────────────────

// Rotate one row of `cols` floats (a whole number of `group`-wide groups) into
// `out`. Doing it per row rather than per matrix is what keeps the streaming
// requantiser's working set at one row: the matrix-sized `rotated` buffer the
// whole-matrix form needs is another rows*cols fp32 written and read back, which
// on a 6.7 GB checkpoint is tens of gigabytes of memory traffic for nothing.
static void convrot_row_forward(const float* in, float* out, i64 cols, int group) {
	for (i64 g = 0; g < cols; g += group) {
		const float* src = in + g;
		float* o = out + g;
		memcpy(o, src, (size_t)group * sizeof(float));
		for (int d = 0, stride = 1; d < 4; d++, stride *= 4) {
			for (int base = 0; base < group; base += stride * 4) {
				for (int off = 0; off < stride; off++) {
					float* p = o + base + off;
					const float x0 = p[0], x1 = p[stride], x2 = p[stride * 2],
					            x3 = p[stride * 3];
					const float s = x0 + x1 + x2 + x3;
					p[0] = s - 2 * x3;
					p[stride] = s - 2 * x2;
					p[stride * 2] = s - 2 * x1;
					p[stride * 3] = s - 2 * x0;
				}
			}
		}
		const float inv = 1.0f / 16.0f;
		for (int i = 0; i < group; i++) o[i] *= inv;
	}
}

// The shared body: rotate and quantise [rows, cols] into caller-owned buffers.
static void quantize_impl_into(const float* src, i64 rows, i64 cols, const QuantSpec& spec,
                               int8_t* out_q, float* out_s) {
	const bool rot = spec.convrot;
	if (rot && (cols % spec.convrot_groupsize) != 0)
		throw MediaError("quantize: cols=" + std::to_string(cols) +
		                 " not divisible by convrot group " +
		                 std::to_string(spec.convrot_groupsize));
	std::vector<float> scratch;
	if (rot) scratch.resize((size_t)cols);

	// A weight matrix is scaled per output channel (one amax per row of [N,K]);
	// an activation matrix is scaled per token (one amax per row as well, but
	// recomputed at runtime). Same loop either way.
	for (i64 r = 0; r < rows; r++) {
		const float* row = src + r * cols;
		if (rot) {
			convrot_row_forward(row, scratch.data(), cols, spec.convrot_groupsize);
			row = scratch.data();
		}
		float amax = 0.0f;
		for (i64 c = 0; c < cols; c++) {
			float a = fabsf(row[c]);
			if (a > amax) amax = a;
		}
		const float scale = amax > 1e-30f ? amax / 127.0f : 1.0f;
		const float inv = 1.0f / scale;
		out_s[(size_t)r] = scale;
		int8_t* qrow = out_q + r * cols;
		for (i64 c = 0; c < cols; c++) {
			double d = (double)(row[c] * inv);
			long v = (long)(d >= 0 ? d + 0.5 : d - 0.5);
			v = v < -127 ? -127 : (v > 127 ? 127 : v);
			qrow[c] = (int8_t)v;
		}
	}
}

void quantize_weight_into(const float* w, i64 rows, i64 cols, const QuantSpec& spec, int8_t* q,
                          float* scale) {
	quantize_impl_into(w, rows, cols, spec, q, scale);
}

static QuantInt8 quantize_impl(const float* src, i64 rows, i64 cols, const QuantSpec& spec,
                              bool per_row) {
	(void)per_row;
	QuantInt8 out;
	out.rows = rows;
	out.cols = cols;
	out.convrot = spec.convrot;
	out.q.assign((size_t)(rows * cols), 0);
	out.scale.assign((size_t)rows, 1.0f);
	quantize_impl_into(src, rows, cols, spec, out.q.data(), out.scale.data());
	return out;
}

QuantInt8 quantize_weight(const float* w, i64 rows, i64 cols, const QuantSpec& spec) {
	return quantize_impl(w, rows, cols, spec, false);
}

QuantInt8 quantize_activation(const float* x, i64 rows, i64 cols, const QuantSpec& spec) {
	return quantize_impl(x, rows, cols, spec, true);
}

void dequantize_int8(const QuantInt8& t, float* dst) {
	const i64 n = t.rows * t.cols;
	for (i64 r = 0; r < t.rows; r++) {
		const float s = t.scale[(size_t)r];
		const int8_t* q = t.q.data() + r * t.cols;
		float* d = dst + r * t.cols;
		for (i64 c = 0; c < t.cols; c++) d[c] = (float)q[c] * s;
	}
	(void)n;
}

// ── reference GEMM ─────────────────────────────────────────────────────────

void gemm_f32(const float* x, const float* w, i64 M, i64 N, i64 K, float* y) {
	for (i64 m = 0; m < M; m++) {
		const float* xr = x + m * K;
		float* yr = y + m * N;
		for (i64 n = 0; n < N; n++) {
			const float* wr = w + n * K;
			float acc = 0.0f;
			for (i64 k = 0; k < K; k++) acc += xr[k] * wr[k];
			yr[n] = acc;
		}
	}
}

void gemm_int8(const int8_t* qx, const float* sx, const int8_t* qw, const float* sw,
               i64 M, i64 N, i64 K, float* y) {
	for (i64 m = 0; m < M; m++) {
		const int8_t* xr = qx + m * K;
		float* yr = y + m * N;
		const float smx = sx ? sx[m] : 1.0f;
		for (i64 n = 0; n < N; n++) {
			const int8_t* wr = qw + n * K;
			int32_t acc = 0;
			for (i64 k = 0; k < K; k++) acc += (int32_t)xr[k] * (int32_t)wr[k];
			yr[n] = (float)acc * smx * (sw ? sw[n] : 1.0f);
		}
	}
}

void gemm_int8_convrot(const float* x, const float* w, i64 M, i64 N, i64 K,
                       const QuantSpec& spec, float* y) {
	QuantInt8 qw = quantize_weight(w, N, K, spec);
	QuantInt8 qx = quantize_activation(x, M, K, spec);
	gemm_int8(qx.q.data(), qx.scale.data(), qw.q.data(), qw.scale.data(), M, N, K, y);
}

// ── int4 ───────────────────────────────────────────────────────────────────

void quantize_convrot_w4a4(const float* w, i64 rows, i64 cols, void* packed, float* scales,
                           int group) {
	if (group != 256 || (cols % group) != 0)
		throw MediaError("quantize_convrot_w4a4: cols=" + std::to_string(cols) +
		                 " is not a multiple of " + std::to_string(group));
	std::vector<float> rot((size_t)(rows * cols));
	convrot_forward(w, rot.data(), rows, cols, group);
	u8* out = (u8*)packed;
	for (i64 r = 0; r < rows; r++) {
		const float* row = &rot[(size_t)r * cols];
		float amax = 0.0f;
		for (i64 c = 0; c < cols; c++) amax = std::max(amax, std::fabs(row[c]));
		// quantize_signed_int4_rowwise: absmax/7, so qmax = 7.
		const float s = amax > 0.0f ? amax / 7.0f : 1.0f;
		scales[r] = s;
		u8* o = out + (size_t)r * (cols / 2);
		for (i64 c = 0; c < cols; c += 2) {
			long lo = (long)std::lround(row[c] / s);
			lo = lo < -7 ? -7 : (lo > 7 ? 7 : lo);
			long hi = (long)std::lround(row[c + 1] / s);
			hi = hi < -7 ? -7 : (hi > 7 ? 7 : hi);
			o[c / 2] = (u8)((lo & 0x0F) | ((hi & 0x0F) << 4));
		}
	}
}


// ── packed weight families ─────────────────────────────────────────────────
//
// The three families below (asym_w4a8_int8, w6a8_int8, nvfp4) all store their
// weights as bit-packed codes plus a *relative* scale per group of columns, and
// they all decode as
//
//     value = decode(code) * scale_relative(group) * scale_channel(row)
//
// with the two scale tensors stored beside the weight. The difference between
// them is only how a code maps to a number and how many codes fit in a byte.

float scale_at(const void* p, DType dtype, i64 i) {
	switch (dtype) {
		case DType::F32: return ((const float*)p)[(size_t)i];
		case DType::F16: return f16_to_f32(((const u16*)p)[(size_t)i]);
		case DType::BF16: return bf16_to_f32(((const u16*)p)[(size_t)i]);
		case DType::F8_E4M3: return f8_e4m3_to_f32(((const u8*)p)[(size_t)i]);
		case DType::F8_E5M2: return f8_e5m2_to_f32(((const u8*)p)[(size_t)i]);
		// A byte container holding fp8: the on-disk spelling every packer that wrote
		// before safetensors had an fp8 dtype used, and what ComfyUI's loader still
		// accepts ("v.view(torch.float8_e4m3fn)"). E4M3 is the only fp8 these tables
		// are ever written in.
		case DType::U8: return f8_e4m3_to_f32(((const u8*)p)[(size_t)i]);
		default: throw MediaError("scale_at: unreadable scale dtype " + std::string(dtype_name(dtype)));
	}
}

// The packer's rounding onto the INT8 grid: **round half to even**, clamped to
// [-127, 127].
//
// This is the eager reference's "values.round().clamp_(-127, 127)" and its device
// counterpart's "clamp(rint(v * s), -127, 127)": torch's `round` and IEEE `rint`
// both resolve a .5 tie towards the even integer, and a level times a group scale
// lands on .5 often enough (the levels are small integers and the scales are fp8)
// that a half-away-from-zero rule shows up as a wrong code - one grid step, which
// is one `s_channel`, on the affected elements only.
//
// Note this is *not* the rule the engine's own quantiser uses (`quantize_weight`,
// which rounds half away from zero and whose output the shipped int8 checkpoints'
// requantisation is pinned against): this one decodes what a packer wrote, that one
// decides what the engine writes.
static inline float int8_grid(float v) {
	const double r0 = std::nearbyint((double)v);   // FE_TONEAREST: half to even
	double r = r0;
	if (r < -127.0) r = -127.0;
	if (r > 127.0) r = 127.0;
	return (float)r;
}

// One element of the grouped families: the code's *level*, scaled by its group's
// relative scale onto the int8 grid, times the row's channel scale. `corr` is this
// row's correction entry for the element's group (0 when the file has none).
//
// The level is a float, not an integer: a codebook holds real levels (the packer's
// ALS fits them, and the fixed LUT's entries are fractional too), so truncating it
// to a whole number throws the codebook away. Only the *uniform* form's level is
// an integer, and it arrives here as one.
static inline float grouped_value(float level, float s_rel, float s_channel, float corr) {
	return int8_grid(level * s_rel) * s_channel + corr;
}

void dequant_codebook_int4_row(const u8* codes, i64 k, const float* codebook,
                               float s_channel, const void* s_rel_row, DType s_rel_dtype,
                               i64 group, const float* corr_row, float* dst) {
	if (group <= 0) group = k;
	for (i64 c = 0; c < k; c += 2) {
		const u8 b = codes[c / 2];
		const i64 i0 = (i64)(b & 0x0F);
		const i64 i1 = (i64)((b >> 4) & 0x0F);
		// A codebook stores the levels themselves; without one the family is the
		// uniform symmetric form, whose zero point is the middle of the 4-bit range.
		const float l0 = codebook ? codebook[i0] : (float)(i0 - 8);
		const float l1 = codebook ? codebook[i1] : (float)(i1 - 8);
		const i64 g0 = c / group;
		const float rel0 = s_rel_row ? scale_at(s_rel_row, s_rel_dtype, g0) : 1.0f;
		dst[c] = grouped_value(l0, rel0, s_channel, corr_row ? corr_row[g0] : 0.0f);
		if (c + 1 < k) {
			const i64 g1 = (c + 1) / group;
			const float rel1 = s_rel_row ? scale_at(s_rel_row, s_rel_dtype, g1) : 1.0f;
			dst[c + 1] = grouped_value(l1, rel1, s_channel, corr_row ? corr_row[g1] : 0.0f);
		}
	}
}

void dequant_packed6_row(const u8* codes, i64 k, float s_channel, const void* s_rel_row,
                         DType s_rel_dtype, i64 group, const float* corr_row, float* dst) {
	if ((k & 1) != 0 || (k % 4) != 0)
		throw MediaError("w6a8_int8: K=" + std::to_string(k) +
		                 " is not a multiple of 4 (the six-bit pack has no row for it)");
	if (group <= 0) group = k;
	// The row is two planes: k/2 bytes of low nibbles, then k/4 bytes of top-2-bit
	// codes (four per byte, at bit positions 0, 2, 4 and 6).
	const u8* low_plane = codes;
	const u8* hi_plane = codes + k / 2;
	for (i64 c = 0; c < k; c += 4) {
		const u8 lo0 = low_plane[c / 2];
		const u8 lo1 = low_plane[c / 2 + 1];
		const u8 hb = hi_plane[c / 4];
		const long code[4] = {
		    (long)((u8)(lo0 & 0x0F) | (u8)(((hb >> 0) & 0x03) << 4)),
		    (long)((u8)((lo0 >> 4) & 0x0F) | (u8)(((hb >> 2) & 0x03) << 4)),
		    (long)((u8)(lo1 & 0x0F) | (u8)(((hb >> 4) & 0x03) << 4)),
		    (long)((u8)((lo1 >> 4) & 0x0F) | (u8)(((hb >> 6) & 0x03) << 4)),
		};
		for (int j = 0; j < 4; j++) {
			const i64 col = c + j;
			if (col >= k) break;
			const i64 g = col / group;
			const float rel = s_rel_row ? scale_at(s_rel_row, s_rel_dtype, g) : 1.0f;
			// Six-bit storage is uniform and symmetric: the level is the code minus
			// the middle of the 6-bit range.
			dst[col] =
			    grouped_value((float)(code[j] - 32), rel, s_channel, corr_row ? corr_row[g] : 0.0f);
		}
	}
}

// The inverse of the cuBLAS "blocked" (tiled) scale layout, for the (row, block)
// pair a weight element belongs to.
//
// The packer pads the [rows, blocks] table to (roundup(rows,128), roundup(blocks,4)),
// views it as (n_row_blocks, 128, n_col_blocks, 4), permutes to (n_row_blocks,
// n_col_blocks, 128, 4) and re-groups each 128x4 tile as 32x(4x4). The element at
// (row, block) therefore lands at flat index "chunk*512 + b*16 + a*4 + w" of the
// stored 2-D table, with the variables below. ComfyUI's "from_blocked" is that walk
// backwards, and tests/precision_coverage_test.cpp checks this against it.
static inline i64 blocked_scale_index(i64 row, i64 block_col, i64 nb_c, i64 blocked_cols) {
	const i64 i_r = row / 128;
	const i64 h = row % 128;
	const i64 i_c = block_col / 4;
	const i64 w = block_col % 4;
	const i64 a = h / 32;
	const i64 b = h % 32;
	const i64 chunk = i_r * nb_c + i_c;
	const i64 flat = chunk * 512 + b * 16 + a * 4 + w;
	return (flat / blocked_cols) * blocked_cols + (flat % blocked_cols);
}

void dequant_nvfp4_row(const u8* codes, i64 row, i64 k, const u8* blocked, i64 blocked_rows,
                       i64 blocked_cols, DType scale_dtype, float global_scale, i64 block,
                       float* dst) {
	if (block <= 0) block = 16;
	const i64 nb_c = blocked_cols > 0 ? blocked_cols / 4 : 0;
	const bool have_scales = blocked != nullptr && nb_c > 0 && blocked_rows > 0;
	const size_t esz = have_scales ? dtype_size(scale_dtype) : 0;
	for (i64 c = 0; c < k; c++) {
		const u8 b = codes[c / 2];
		// The even column is in the *high* nibble (the packer's hi_first order), the
		// opposite of the int4 families above.
		const u8 nib = (u8)((c & 1) ? (b & 0x0F) : (b >> 4));
		float s = 1.0f;
		if (have_scales) {
			const i64 idx = blocked_scale_index(row, c / block, nb_c, blocked_cols);
			s = scale_at(blocked + (size_t)idx * esz, scale_dtype, 0);
		}
		dst[c] = e2m1_to_f32(nib) * s * global_scale;
	}
}

void dequant_fp8_row(const u8* codes, i64 k, DType dtype, float scale, float* dst) {
	if (dtype == DType::F8_E5M2) {
		for (i64 c = 0; c < k; c++) dst[c] = f8_e5m2_to_f32(codes[c]) * scale;
	} else {
		for (i64 c = 0; c < k; c++) dst[c] = f8_e4m3_to_f32(codes[c]) * scale;
	}
}

}  // namespace phi::media
