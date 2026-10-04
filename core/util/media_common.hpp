// Media inference engine — shared primitives.
//
// The engine is built out of Windows system components only: CUDA (the driver
// API + NVRTC, loaded at runtime) for compute, Media Foundation for
// container/codec work, WIC for still images. Nothing in here links against a
// third-party runtime and nothing needs Python.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace phi::media {

using i32 = int32_t;
using i64 = int64_t;
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

// Every failure inside the engine is reported as MediaError so the tool layer
// can turn it into a readable tool result instead of taking the app down.
struct MediaError : std::runtime_error {
	explicit MediaError(const std::string& what) : std::runtime_error(what) {}
};

// ── dtypes ─────────────────────────────────────────────────────────────────

// F8_E4M3 / F8_E5M2 are storage dtypes only: nothing computes in them, but the
// new checkpoints carry them (fp8 weights, and the block scales of the 4-bit
// formats), so every reader converts them on the way in.
enum class DType { Unknown, F32, F16, BF16, I8, U8, I4, F8_E4M3, F8_E5M2 };

const char* dtype_name(DType t);
DType dtype_from_name(std::string_view name);  // safetensors spelling
size_t dtype_size(DType t);                    // bytes per element (I4: 1 byte holds 2)

// fp8 / fp4 element decode. `e4m3` and `e5m2` are the two safetensors fp8
// spellings; `e2m1` is the 4-bit float nvfp4 stores its weights in.
float f8_e4m3_to_f32(u8 b);
float f8_e5m2_to_f32(u8 b);
float e2m1_to_f32(u8 nibble);

// IEEE half / bfloat16 <-> float (round-to-nearest-even on the way down)
float f16_to_f32(u16 h);
u16 f32_to_f16(float f);
float bf16_to_f32(u16 h);
u16 f32_to_bf16(float f);

// int4 nibble decode (low nibble is the first element — comfy_quant layout)
inline int8_t unpack_i4_low(u8 b) {
	int8_t v = (int8_t)(b & 0x0F);
	return (v & 0x08) ? (int8_t)(v - 16) : v;
}
inline int8_t unpack_i4_high(u8 b) {
	int8_t v = (int8_t)((b >> 4) & 0x0F);
	return (v & 0x08) ? (int8_t)(v - 16) : v;
}

// ── misc ───────────────────────────────────────────────────────────────────

double now_ms();      // monotonic milliseconds
std::string file_version_of(const std::string& path);  // PE fixed file version ("" on failure)
u64 mix_seed(u64 a);  // 64-bit mixer used to derive the per-run seed

// 64-bit seed from the OS entropy pool mixed with the clock, the process id and
// the calling thread, so two generations asked for in the same millisecond still
// differ. Shared by image_generate and video_generate; the value is echoed back
// to the caller so a run can be reproduced.
u64 make_media_seed();

inline i64 ceil_div(i64 a, i64 b) { return (a + b - 1) / b; }
inline i64 round_up(i64 a, i64 b) { return ceil_div(a, b) * b; }
inline i64 round_down(i64 a, i64 b) { return (a / b) * b; }

std::string format_bytes(u64 n);   // "6.20 GB"
std::string format_ms(double ms);  // "12.3 s"

}  // namespace phi::media
