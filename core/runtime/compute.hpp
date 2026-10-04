// Compute layer: NVRTC compilation, kernel caching, and dispatch.
//
// What the rest of the engine sees is deliberately narrow: the same
// `KernelParams` (20 scalars + 8 SRV + 4 UAV), the same
// `begin()` / `dispatch()` / `submit()` cycle, and the same
// `create_device_buffer` / `upload` / `download` helpers. A compiled kernel is a
// `GpuKernel*` — the same handle the kernels themselves are launched through.
//
// The one new thing is `KernelArgs`: the struct the host fills and the device
// reads. Every CUDA kernel takes exactly `(KernelArgs a)` by value, so the
// per-kernel launch code is now one generic marshal step in `dispatch()`.
//
// `KernelArgs` MUST stay layout-identical to the `Args` struct the kernels
// declare. It is 176 bytes: 20 x u32, 8 x pointer, 4 x pointer.
#pragma once

#include <string>
#include <vector>

#include "runtime/cuda_device.hpp"
#include "util/media_common.hpp"

namespace phi::media {

// The kernel ABI. Byte offsets stay in `v[]` exactly as they were in the HLSL
// root constants; `s[]`/`u[]` are the resource bases and the kernel adds the
// offsets itself. That is what kept this port mechanical.
struct KernelArgs {
	u32 v[24] = {0};
	const char* s[8] = {nullptr};
	char* u[4] = {nullptr};
};

// Root-signature equivalent: 24 scalars, 8 read-only buffers, 4 writable.
struct KernelParams {
	u32 values[24] = {0};
	GpuBuffer* srv[8] = {nullptr};
	u64 srv_offset[8] = {0};
	u64 srv_size[8] = {0};
	GpuBuffer* uav[4] = {nullptr};
	u64 uav_offset[4] = {0};
	u64 uav_size[4] = {0};
};

// How NVRTC is told to compile. The values are kept as the old shader-model
// names because the call sites pass them; on CUDA they only select a warning
// level, since the arch is picked from the device.
enum class ShaderModel { SM5_1, SM6_0, SM6_2, SM6_4, SM6_6 };

struct ShaderCompilerStatus {
	bool nvrtc = false;        // nvrtc64_*.dll usable
	int compiles = 0;
	int cache_hits = 0;
	std::string last_error;
	std::string describe() const;
};
const ShaderCompilerStatus& shader_compiler_status();
// Removes every cached cubin under `cache_dir`; returns how many files went.
int clear_shader_cache(const std::string& cache_dir);

// A compiled kernel: the module that owns it plus the function handle and the
// block shape the source declared.
class GpuKernel {
public:
	std::string key;
	std::string entry;
	CUmodule module = nullptr;
	CUfunction fn = nullptr;
	u32 block_x = 256, block_y = 1, block_z = 1;
	// Dynamic shared memory the kernel asks for at launch (0 = none). Set from
	// the source's `// smem <bytes>` marker when present.
	u32 dynamic_smem = 0;
};

class ComputeContext {
public:
	ComputeContext() = default;
	~ComputeContext();
	ComputeContext(const ComputeContext&) = delete;
	ComputeContext& operator=(const ComputeContext&) = delete;

	// `cuda` must outlive this object. `cache_dir` enables the on-disk kernel
	// cache (compiling ~25 kernels through NVRTC costs tens of seconds).
	void create(CudaContext& cuda, const std::string& cache_dir);
	void destroy();
	bool valid() const { return cuda_ != nullptr; }

	// Compiles (or loads from cache) `source` and returns the kernel for `entry`.
	// `block` overrides the block shape parsed out of the source (autotune uses
	// this, because its tile sizes are text-substituted into the source and the
	// block size follows them).
	// Throws MediaError with NVRTC's diagnostics on failure.
	GpuKernel* pipeline(const std::string& key, const std::string& source, const std::string& entry,
	                    ShaderModel min_model = ShaderModel::SM6_4, u32 block_x = 0,
	                    u32 block_y = 0, u32 block_z = 0);
	bool has_pipeline(const std::string& key) const;

	// Records one dispatch: the launch is enqueued immediately, and begin()/
	// submit() only bracket it. Buffers are bound as device addresses, so no
	// descriptor heap churn is possible: byte offsets are folded into the pointer.
	void begin();
	void dispatch(GpuKernel* k, const KernelParams& params, u32 gx, u32 gy, u32 gz);
	void submit();  // stream synchronise

	// Whether a dispatch bracket is currently open. Kept because the scheduler
	// uses it to decide whether to re-open one; it no longer means "commands are
	// recorded but not executed".
	bool recording() const { return recording_; }
	void submit_if_recording();

	// Convenience: begin + dispatch + submit.
	void run(GpuKernel* k, const KernelParams& params, u32 gx, u32 gy, u32 gz);

	// GPU buffer helpers. `uav` is accepted for source compatibility and ignored:
	// CUDA has no resource states.
	GpuBuffer* create_device_buffer(u64 bytes, bool uav);
	GpuBuffer* create_upload_buffer(u64 bytes);
	GpuBuffer* create_readback_buffer(u64 bytes);
	void release_buffer(GpuBuffer* res);

	// Copies `bytes` from `src` (host memory) into a device buffer.
	void upload(GpuBuffer* dst, const void* src, u64 bytes);
	// Same, at a byte offset into the destination. A page-locked `src` is a true
	// DMA; a pageable one is staged by the driver.
	void upload(GpuBuffer* dst, u64 dst_off, const void* src, u64 bytes);
	void download(GpuBuffer* src, void* dst, u64 bytes);
	void download(GpuBuffer* src, u64 src_off, void* dst, u64 bytes);

	const ShaderCompilerStatus& compiler_status() const { return shader_compiler_status(); }
	CudaContext& device_context() { return *cuda_; }
	CUdevice device() const { return cuda_ ? cuda_->device() : 0; }
	CUstream stream() const { return cuda_ ? cuda_->stream() : nullptr; }

private:
	CudaContext* cuda_ = nullptr;
	std::vector<GpuKernel*> kernels_;
	std::vector<GpuBuffer*> owned_;
	bool recording_ = false;
	std::string cache_dir_;
	// Current dispatch bracket. Nothing needs it, but keeping the flag preserves
	// the begin()/submit() contract.
	int depth_ = 0;
};

// ── the two int8 kernels ───────────────────────────────────────────────────

struct Int8GemmArgs {
	GpuBuffer* a = nullptr;       // [M,K] int8
	u64 a_offset = 0;
	GpuBuffer* b = nullptr;       // [N,K] int8
	u64 b_offset = 0;
	GpuBuffer* sa = nullptr;      // [M] fp32
	u64 sa_offset = 0;
	GpuBuffer* sb = nullptr;      // [N] fp32
	u64 sb_offset = 0;
	GpuBuffer* c = nullptr;       // [M,N] fp32
	u64 c_offset = 0;
	// Optional [N] fp32 bias, added to the scaled result in the epilogue. The
	// DiTs and the text towers have bias-free linears and leave this null; the
	// video VAE's four projections do have biases, and folding them in here is
	// what keeps the int8 path from paying a second read+write of the whole
	// [M,N] fp32 output (a quarter of the w1 stage at the decoder's shapes).
	GpuBuffer* bias = nullptr;
	u64 bias_offset = 0;
	i64 M = 0, N = 0, K = 0;
};

// Enqueues the GEMM.
void dispatch_int8_gemm(ComputeContext& ctx, const Int8GemmArgs& args);

struct QuantConvrotArgs {
	GpuBuffer* x = nullptr;   // [rows,K] fp32
	u64 x_offset = 0;
	GpuBuffer* q = nullptr;   // [rows,K] int8 out
	u64 q_offset = 0;
	GpuBuffer* s = nullptr;   // [rows] fp32 out
	u64 s_offset = 0;
	i64 rows = 0, K = 0;
	// Quantiser range. 127 for the int8 checkpoints; 7 for the Qwen3-VL text
	// encoder, whose `convrot_w4a4` weights were packed against a 4-bit
	// activation range (absmax/7) and whose integer product is therefore exact
	// only if the activations carry the same codes.
	float qmax = 127.0f;
};

// Enqueues rotate+quantise for `rows` rows of fp32 [rows,K].
// K must be a multiple of 256.
void dispatch_quant_convrot(ComputeContext& ctx, const QuantConvrotArgs& args);

}  // namespace phi::media
