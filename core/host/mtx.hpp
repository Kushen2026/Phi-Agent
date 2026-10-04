// Host-side buffers and dtype conversion: everything the engine moves between
// the safetensors files, RAM and VRAM passes through here.
#pragma once

#include <utility>

#include "util/media_common.hpp"

namespace phi::media {

// Owned, aligned (64B) fp32/byte buffer used for host-side staging. Aligned so
// the same memory can back an upload to VRAM without a copy.
class HostBuffer {
public:
	HostBuffer() = default;
	explicit HostBuffer(size_t bytes) { resize(bytes); }
	~HostBuffer() { release(); }
	HostBuffer(const HostBuffer&) = delete;
	HostBuffer& operator=(const HostBuffer&) = delete;
	HostBuffer(HostBuffer&& o) noexcept { swap(o); }
	HostBuffer& operator=(HostBuffer&& o) noexcept {
		if (this != &o) {
			release();
			swap(o);
		}
		return *this;
	}

	void resize(size_t bytes);
	void release() { resize(0); }
	void* data() { return ptr_; }
	const void* data() const { return ptr_; }
	size_t size() const { return size_; }

private:
	void swap(HostBuffer& o) noexcept {
		std::swap(ptr_, o.ptr_);
		std::swap(size_, o.size_);
	}
	void* ptr_ = nullptr;
	size_t size_ = 0;
};

// Convert `count` elements of `src` (dtype) into fp32 at dst.
void convert_to_f32(DType src, const void* src_data, float* dst, size_t count);
// Convert fp32 into `dst_dtype` at dst.
void convert_from_f32(const float* src, DType dst_dtype, void* dst, size_t count);

// Dequantise a tensor to fp32 using a per-output-channel scale vector
// (scale.size() == rows or 1). Handles I8/U8/I4/F16/BF16/F32.

}  // namespace phi::media
