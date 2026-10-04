#include "util/media_common.hpp"

#include <windows.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>
#include <thread>

namespace phi::media {

// ── dtypes ─────────────────────────────────────────────────────────────────

const char* dtype_name(DType t) {
	switch (t) {
		case DType::F32: return "F32";
		case DType::F16: return "F16";
		case DType::BF16: return "BF16";
		case DType::I8: return "I8";
		case DType::U8: return "U8";
		case DType::I4: return "I4";
		case DType::F8_E4M3: return "F8_E4M3";
		case DType::F8_E5M2: return "F8_E5M2";
		default: return "?";
	}
}

DType dtype_from_name(std::string_view name) {
	if (name == "F32" || name == "float32" || name == "float") return DType::F32;
	if (name == "F16" || name == "float16") return DType::F16;
	if (name == "BF16" || name == "bfloat16") return DType::BF16;
	if (name == "I8" || name == "int8") return DType::I8;
	if (name == "U8" || name == "uint8") return DType::U8;
	if (name == "I4" || name == "int4" || name == "U4") return DType::I4;
	if (name == "F8_E4M3" || name == "float8_e4m3fn" || name == "float8_e4m3" ||
	    name == "F8E4M3" || name == "e4m3")
		return DType::F8_E4M3;
	if (name == "F8_E5M2" || name == "float8_e5m2" || name == "F8E5M2" || name == "e5m2")
		return DType::F8_E5M2;
	return DType::Unknown;
}

size_t dtype_size(DType t) {
	switch (t) {
		case DType::F32: return 4;
		case DType::F16: return 2;
		case DType::BF16: return 2;
		case DType::I8: return 1;
		case DType::U8: return 1;
		case DType::I4: return 1;  // packed: two elements per byte
		case DType::F8_E4M3: return 1;
		case DType::F8_E5M2: return 1;
		default: return 0;
	}
}

// ── half / bfloat16 ────────────────────────────────────────────────────────

float f16_to_f32(u16 h) {
	u32 sign = (u32)(h & 0x8000) << 16;
	u32 exp = (h >> 10) & 0x1F;
	u32 man = h & 0x3FF;
	u32 bits;
	if (exp == 0) {
		if (man == 0) {
			bits = sign;  // +-0
		} else {
			// subnormal: normalize into a float
			int shift = 0;
			while ((man & 0x400) == 0) {
				man <<= 1;
				shift++;
			}
			man &= 0x3FF;
			bits = sign | (u32)(127 - 15 - shift) << 23 | man << 13;
		}
	} else if (exp == 0x1F) {
		bits = sign | 0x7F800000u | (man << 13);  // inf / nan
	} else {
		bits = sign | (exp + 127 - 15) << 23 | (man << 13);
	}
	float f;
	memcpy(&f, &bits, 4);
	return f;
}

u16 f32_to_f16(float f) {
	u32 bits;
	memcpy(&bits, &f, 4);
	u32 sign = (bits >> 16) & 0x8000;
	u32 exp = (bits >> 23) & 0xFF;
	u32 man = bits & 0x7FFFFF;
	if (exp == 0xFF) return (u16)(sign | 0x7C00 | (man ? 0x200 : 0));
	int e = (int)exp - 127 + 15;
	if (e >= 0x1F) return (u16)(sign | 0x7C00);  // overflow -> inf
	if (e <= 0) {
		if (e < -10) return (u16)sign;
		// subnormal, with round-to-nearest-even
		man |= 0x800000;
		u32 shift = (u32)(14 - e);
		u32 v = man >> shift;
		u32 rest = man & ((1u << shift) - 1);   // low `shift` bits, not shift-1
		u32 halfway = 1u << (shift - 1);
		if (rest > halfway || (rest == halfway && (v & 1))) v++;
		return (u16)(sign | v);
	}
	u32 v = (u32)e << 10 | (man >> 13);
	u32 rem = man & 0x1FFF;
	if (rem > 0x1000 || (rem == 0x1000 && (v & 1))) v++;
	return (u16)(sign | v);  // carries into the exponent are correct here
}

float bf16_to_f32(u16 h) {
	u32 bits = (u32)h << 16;
	float f;
	memcpy(&f, &bits, 4);
	return f;
}

// ── fp8 / fp4 ──────────────────────────────────────────────────────────────
//
// Both fp8 spellings are built from a table lookup: 256 entries, generated once.
// That keeps the bit twiddling (subnormals, inf, NaN) in one place and makes the
// hot loops a gather. E4M3 has no inf (0x7F/0xFF are NaN) and 3 mantissa bits;
// E5M2 is IEEE-shaped with 2 mantissa bits.

static float f8_lut(bool e4m3, u8 b) {
	const int sign = (b >> 7) & 1;
	const int ebits = e4m3 ? 4 : 5;
	const int mbits = e4m3 ? 3 : 2;
	const int emax = (1 << ebits) - 1;
	const int exp = (b >> mbits) & emax;
	const int man = b & ((1 << mbits) - 1);
	const int bias = e4m3 ? 7 : 15;
	float v;
	if (exp == emax) {
		// all-ones exponent: E4M3 makes 0x7F/0xFF NaN, everything else inf
		v = (e4m3 && man == (1 << mbits) - 1) ? std::numeric_limits<float>::quiet_NaN()
		                                      : std::numeric_limits<float>::infinity();
	} else if (exp == 0) {
		v = (float)man * std::ldexp(1.0f, 1 - bias - mbits);  // subnormal
	} else {
		v = (float)((1 << mbits) + man) * std::ldexp(1.0f, exp - bias - mbits);
	}
	return sign ? -v : v;
}

float f8_e4m3_to_f32(u8 b) {
	static float lut[256];
	static std::once_flag once;
	std::call_once(once, [] {
		for (int i = 0; i < 256; i++) lut[i] = f8_lut(true, (u8)i);
	});
	return lut[b];
}

float f8_e5m2_to_f32(u8 b) {
	static float lut[256];
	static std::once_flag once;
	std::call_once(once, [] {
		for (int i = 0; i < 256; i++) lut[i] = f8_lut(false, (u8)i);
	});
	return lut[b];
}

// nvfp4's 4-bit float: 1 sign + 2 exponent + 1 mantissa, values
// 0, .5, 1, 1.5, 2, 3, 4, 6 (subnormals are .5 and 1).
float e2m1_to_f32(u8 nibble) {
	static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
	const u8 m = nibble & 0x07;
	return (nibble & 0x08) ? -mag[m] : mag[m];
}

u16 f32_to_bf16(float f) {
	u32 bits;
	memcpy(&bits, &f, 4);
	u32 lsb = (bits >> 16) & 1;
	bits += 0x7FFF + lsb;  // round-to-nearest-even
	return (u16)(bits >> 16);
}

// ── misc ───────────────────────────────────────────────────────────────────

double now_ms() {
	using clock = std::chrono::steady_clock;
	static const clock::time_point origin = clock::now();
	return std::chrono::duration<double, std::milli>(clock::now() - origin).count();
}

u64 mix_seed(u64 a) {
	a ^= a >> 33;
	a *= 0xFF51AFD7ED558CCDull;
	a ^= a >> 33;
	a *= 0xC4CEB9FE1A85EC53ull;
	a ^= a >> 33;
	return a;
}

u64 make_media_seed() {
	// random_device (+) high_resolution_clock (+) pid (+) thread id, finalised
	// with splitmix64: a plain xor of four weakly-distributed streams leaves
	// visible structure in the low bits, and the first thing the sampler does
	// with the seed is index a Box-Muller pair.
	std::random_device rd;
	u64 a = ((u64)rd() << 32) ^ (u64)rd();
	u64 b = (u64)std::chrono::high_resolution_clock::now().time_since_epoch().count();
#ifdef _WIN32
	u64 c = (u64)GetCurrentProcessId();
#else
	u64 c = 0;
#endif
	u64 d = (u64)std::hash<std::thread::id>{}(std::this_thread::get_id());
	u64 x = a ^ (b * 0x9E3779B97F4A7C15ull) ^ (c << 32) ^ (d * 0xBF58476D1CE4E5B9ull);
	x += 0x9E3779B97F4A7C15ull;
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
	x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
	return x ^ (x >> 31);
}

std::string file_version_of(const std::string& path) {
	std::wstring w;
	int need = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), (int)path.size(), nullptr, 0);
	w.resize((size_t)need);
	MultiByteToWideChar(CP_UTF8, 0, path.c_str(), (int)path.size(), w.data(), need);
	DWORD dummy = 0;
	DWORD size = GetFileVersionInfoSizeW(w.c_str(), &dummy);
	if (!size) return "";
	std::vector<u8> buf(size);
	if (!GetFileVersionInfoW(w.c_str(), 0, size, buf.data())) return "";
	VS_FIXEDFILEINFO* ffi = nullptr;
	UINT len = 0;
	if (!VerQueryValueW(buf.data(), L"\\", (void**)&ffi, &len) || !ffi) return "";
	char out[64];
	snprintf(out, sizeof out, "%u.%u.%u.%u", (unsigned)(ffi->dwFileVersionMS >> 16),
	         (unsigned)(ffi->dwFileVersionMS & 0xFFFF), (unsigned)(ffi->dwFileVersionLS >> 16),
	         (unsigned)(ffi->dwFileVersionLS & 0xFFFF));
	return out;
}

std::string format_bytes(u64 n) {
	char buf[64];
	double v = (double)n;
	const char* unit = "B";
	if (v >= 1024.0 * 1024 * 1024) {
		v /= 1024.0 * 1024 * 1024;
		unit = "GB";
	} else if (v >= 1024.0 * 1024) {
		v /= 1024.0 * 1024;
		unit = "MB";
	} else if (v >= 1024.0) {
		v /= 1024.0;
		unit = "KB";
	}
	snprintf(buf, sizeof buf, "%.2f %s", v, unit);
	return buf;
}

std::string format_ms(double ms) {
	char buf[64];
	if (ms >= 1000.0) {
		snprintf(buf, sizeof buf, "%.2f s", ms / 1000.0);
	} else {
		snprintf(buf, sizeof buf, "%.1f ms", ms);
	}
	return buf;
}

}  // namespace phi::media
