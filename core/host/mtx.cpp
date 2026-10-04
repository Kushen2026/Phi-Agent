#include "host/mtx.hpp"

#include <cstdlib>
#include <cstring>

#ifndef _WIN32
#define _aligned_malloc(s, a) aligned_alloc(a, s)
#define _aligned_free(p) free(p)
#endif

namespace phi::media {

// round-to-nearest, away from zero at .5 — matches numpy's rint closely enough
// for int8 quantisation and avoids lrintf's feature-test gating on MinGW
static inline long round_i(float v) {
	double d = (double)v;
	return (long)(d >= 0 ? d + 0.5 : d - 0.5);
}

// ── HostBuffer ─────────────────────────────────────────────────────────────

void HostBuffer::resize(size_t bytes) {
	if (bytes == size_) return;
	if (ptr_) {
		_aligned_free(ptr_);
		ptr_ = nullptr;
		size_ = 0;
	}
	if (bytes == 0) return;
	// 64-byte aligned: one cache line, and a legal VRAM upload granularity
	void* p = _aligned_malloc(bytes, 64);
	if (!p) {
		throw MediaError("HostBuffer: allocation of " + format_bytes(bytes) + " failed");
	}
	ptr_ = p;
	size_ = bytes;
}

// ── dtype conversion ───────────────────────────────────────────────────────

void convert_to_f32(DType src, const void* src_data, float* dst, size_t count) {
	const u8* p = (const u8*)src_data;
	switch (src) {
		case DType::F32:
			memcpy(dst, p, count * 4);
			break;
		case DType::F16:
			for (size_t i = 0; i < count; i++) dst[i] = f16_to_f32(((const u16*)p)[i]);
			break;
		case DType::BF16:
			for (size_t i = 0; i < count; i++) dst[i] = bf16_to_f32(((const u16*)p)[i]);
			break;
		case DType::I8:
			for (size_t i = 0; i < count; i++) dst[i] = (float)((const int8_t*)p)[i];
			break;
		case DType::U8:
			for (size_t i = 0; i < count; i++) dst[i] = (float)p[i];
			break;
		case DType::I4:
			for (size_t i = 0; i < count; i++) {
				u8 b = p[i / 2];
				dst[i] = (float)((i & 1) ? unpack_i4_high(b) : unpack_i4_low(b));
			}
			break;
		case DType::F8_E4M3:
			for (size_t i = 0; i < count; i++) dst[i] = f8_e4m3_to_f32(p[i]);
			break;
		case DType::F8_E5M2:
			for (size_t i = 0; i < count; i++) dst[i] = f8_e5m2_to_f32(p[i]);
			break;
		default:
			throw MediaError("convert_to_f32: unsupported dtype");
	}
}

void convert_from_f32(const float* src, DType dst_dtype, void* dst, size_t count) {
	u8* p = (u8*)dst;
	switch (dst_dtype) {
		case DType::F32:
			memcpy(p, src, count * 4);
			break;
		case DType::F16:
			for (size_t i = 0; i < count; i++) ((u16*)p)[i] = f32_to_f16(src[i]);
			break;
		case DType::BF16:
			for (size_t i = 0; i < count; i++) ((u16*)p)[i] = f32_to_bf16(src[i]);
			break;
		case DType::I8:
			for (size_t i = 0; i < count; i++) {
				long v = round_i(src[i]);
				v = v < -127 ? -127 : (v > 127 ? 127 : v);
				((int8_t*)p)[i] = (int8_t)v;
			}
			break;
		case DType::I4:
			for (size_t i = 0; i < count; i += 2) {
				long lo = round_i(src[i]);
				lo = lo < -8 ? -8 : (lo > 7 ? 7 : lo);
				long hi = 0;
				if (i + 1 < count) {
					hi = round_i(src[i + 1]);
					hi = hi < -8 ? -8 : (hi > 7 ? 7 : hi);
				}
				p[i / 2] = (u8)((lo & 0x0F) | ((hi & 0x0F) << 4));
			}
			break;
		default:
			throw MediaError("convert_from_f32: unsupported dtype");
	}
}

}  // namespace phi::media
