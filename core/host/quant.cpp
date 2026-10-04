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
		default: throw MediaError("scale_at: unreadable scale dtype " + std::string(dtype_name(dtype)));
	}
}

void dequant_codebook_int4_row(const u8* codes, i64 k, const float* codebook,
                               float s_channel, const void* s_rel_row, DType s_rel_dtype,
                               i64 group, float* dst) {
	if (!codebook) throw MediaError("asym_w4a8_int8: the weight has no codebook");
	if (group <= 0) group = k;
	for (i64 c = 0; c < k; c += 2) {
		const u8 b = codes[c / 2];
		const float g0 = s_rel_row ? scale_at(s_rel_row, s_rel_dtype, (c / group)) : 1.0f;
		const long i0 = (long)(b & 0x0F);
		const long i1 = (long)((b >> 4) & 0x0F);
		dst[c] = codebook[i0] * g0 * s_channel;
		if (c + 1 < k) {
			const float g1 = s_rel_row ? scale_at(s_rel_row, s_rel_dtype, ((c + 1) / group)) : 1.0f;
			dst[c + 1] = codebook[i1] * g1 * s_channel;
		}
	}
}

// Six-bit codes, four per three bytes, little-endian inside the triple.
void unpack_6bit_triple(const u8* t, u8 out[4]) {
	out[0] = (u8)(t[0] & 0x3F);
	out[1] = (u8)(((t[0] >> 6) | (t[1] << 2)) & 0x3F);
	out[2] = (u8)(((t[1] >> 4) | (t[2] << 4)) & 0x3F);
	out[3] = (u8)(t[2] >> 2);
}

void dequant_packed6_row(const u8* codes, i64 k, float s_channel, const void* s_rel_row,
                         DType s_rel_dtype, i64 group, float* dst) {
	if ((k & 3) != 0)
		throw MediaError("w6a8_int8: K=" + std::to_string(k) + " is not a multiple of 4");
	if (group <= 0) group = k;
	for (i64 c = 0; c < k; c += 4) {
		u8 v[4];
		unpack_6bit_triple(codes + (c / 4) * 3, v);
		for (int j = 0; j < 4; j++) {
			const i64 col = c + j;
			const float g = s_rel_row ? scale_at(s_rel_row, s_rel_dtype, col / group) : 1.0f;
			dst[col] = (float)v[j] * g * s_channel;
		}
	}
}

void dequant_nvfp4_row(const u8* codes, i64 k, const void* block_scale, DType scale_dtype,
                       float global_scale, i64 block, float* dst) {
	if (block <= 0) block = 16;
	for (i64 c = 0; c < k; c++) {
		const u8 b = codes[c / 2];
		const u8 nib = (u8)((c & 1) ? (b >> 4) : (b & 0x0F));
		const float s = block_scale ? scale_at(block_scale, scale_dtype, c / block) : 1.0f;
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
