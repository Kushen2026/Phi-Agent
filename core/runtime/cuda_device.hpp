// CUDA device/context plumbing: the one context, the one compute stream, the
// event-based fence helpers, and the VRAM budget that drives weight streaming.
//
// The parts the rest of the engine relies on:
//
//   * There is no command list. CUDA enqueues work the moment it is asked to, so
//     `ComputeContext::begin()` records nothing and `dispatch()` launches
//     straight onto the stream. `submit()` is a stream synchronise. The
//     recording/submit *contract* is preserved (so callers did not have to
//     change), but it is no longer a batching mechanism.
//   * A device allocation is a plain 64-bit address, so `GpuBuffer` is a thin
//     value rather than a COM object with refcounts. It exists only so the
//     engine's "resource + byte offset" idiom (GpuAlloc) survives unchanged.
//   * `double` on the GPU is emulated, not native; no kernel in the engine
//     uses it, and none should start.
#pragma once

#include <string>
#include <vector>

#include "runtime/cuda_api.hpp"
#include "util/media_common.hpp"

namespace phi::media {

// A device allocation. Not refcounted: `CudaContext` owns it and frees it in
// destroy() or on an explicit release.
struct GpuBuffer {
	CUdeviceptr base = 0;
	u64 bytes = 0;
	bool host_visible = false;   // pinned staging / readback
	void* mapped = nullptr;      // valid for readback and pinned buffers
};

struct AdapterInfo {
	int index = 0;
	std::string name;
	int cc_major = 0, cc_minor = 0;
	u64 dedicated_vram = 0;
	u64 shared_system = 0;
	bool software = false;
	std::string describe() const;
};

// Every CUDA device, best (most VRAM, discrete) first.
std::vector<AdapterInfo> enumerate_adapters();

struct VideoMemoryBudget {
	u64 budget = 0;             // bytes we may use before the OS starts evicting
	u64 current_usage = 0;
	u64 available_reservation = 0;
	u64 process_usage = 0;
};

// ── the driver's own budget for this process ────────────────────────────────
//
// `cuMemGetInfo` answers a question about the *device*: how many bytes are not
// committed. That is not the number a planner needs, and on WDDM it is actively
// misleading in two ways:
//
//   * it never reports less than the card's total minus what is committed, so a
//     process that runs past the driver's *local-memory budget* still sees "free"
//     falling to 0 and then simply keeps allocating - the driver silently backs
//     the overflow with system memory, every touch of a demoted page costs
//     hundreds of microseconds, and a two-second kernel becomes a TDR;
//   * it counts the CUDA context's own reservation (~1 GB on this box) as
//     committed, but WDDM charges the same context only ~96 MB, so the two
//     accounting systems do not agree on what "used" means.
//
// DXGI's `IDXGIAdapter3::QueryVideoMemoryInfo(LOCAL)` is the driver's own answer:
// the budget is what WDDM will let this process commit in local memory before it
// demotes, and the usage is what the process (CUDA context included) really
// holds. The accountant plans and refuses against *that* pair; the CUDA figures
// stay for what only they can answer (the device's total, the context baseline).
struct DriverVram {
	u64 budget = 0;        // local video memory this process may commit
	u64 usage = 0;         // of which already committed
	u64 reservation = 0;   // still available for reservation
	bool ok = false;       // false => no DXGI answer; callers fall back to CUDA
};

// Queries the DXGI adapter that belongs to `cuda_adapter` (matched by dedicated
// VRAM, preferring a hardware adapter) for the driver's live local-memory
// figures for this process. Creates and caches the DXGI factory once; every call
// after that is a single `QueryVideoMemoryInfo`, which is cheap enough for the
// accountant's live probe to run on each charge. Never throws: `ok = false`
// means "no DXGI", not "no memory".
DriverVram query_driver_vram(const AdapterInfo& cuda_adapter);

class CudaContext {
public:
	CudaContext() = default;
	~CudaContext();
	CudaContext(const CudaContext&) = delete;
	CudaContext& operator=(const CudaContext&) = delete;

	// Creates the context on `device_index` (-1 = pick automatically: the discrete
	// device with the most dedicated VRAM). Throws MediaError when the CUDA
	// runtime is absent, naming what was missing.
	void create(int device_index = -1);
	void destroy();
	bool valid() const { return ctx_ != nullptr; }

	CUdevice device() const { return dev_; }
	CUcontext context() const { return ctx_; }
	CUstream stream() const { return stream_; }
	const AdapterInfo& adapter() const { return adapter_; }

	// Blocks until everything enqueued on the compute stream has finished.
	void sync();

	// Fence helpers, preserved for the scheduler. A fence value is a slot index;
	// `signal_fence()` records an event and returns its slot.
	u64 signal_fence();
	void wait_fence(u64 value);
	bool fence_reached(u64 value) const;

	// ── memory ─────────────────────────────────────────────────────────────
	// Device memory. Throws MediaError when the allocation fails (the caller is
	// expected to have consulted the VRAM accountant first).
	GpuBuffer* alloc(u64 bytes);
	// Pinned host memory, for the upload ring. `*mapped` gets the stable host
	// pointer.
	GpuBuffer* alloc_host(u64 bytes, void** mapped);
	void free(GpuBuffer* b);

	void copy_h2d(GpuBuffer* dst, u64 dst_off, const void* src, u64 bytes);
	// Async form, ordered on the compute stream. Requires `src` to be pinned when
	// the copy is to be genuinely asynchronous; the upload ring is.
	void copy_h2d_async(GpuBuffer* dst, u64 dst_off, const void* src, u64 bytes);
	void copy_d2h(GpuBuffer* src, u64 src_off, void* dst, u64 bytes);
	void copy_d2d(GpuBuffer* dst, u64 dst_off, GpuBuffer* src, u64 src_off, u64 bytes);
	void memset_d8(GpuBuffer* dst, u64 dst_off, u8 value, u64 bytes);

	VideoMemoryBudget query_budget() const;

	// True once the context has been lost (a fault took the device down).
	// Everything after that point is undefined, so callers bail out and report it.
	bool device_removed() const { return lost_; }
	void note_device_lost() { lost_ = true; }

	// Adapter capability probes, filled once at create().
	bool supports_int8_dot() const;      // sm_61+ __dp4a
	bool supports_fp16_pack() const;     // always true on the archs we support
	bool supports_tensor_cores() const;  // sm_70+ wmma
	std::string capabilities() const;
	// What the CUDA context alone costs this process, in device bytes.
	// "sm_86" - what NVRTC is told to compile for.
	const std::string& arch() const { return arch_; }
	int shared_mem_per_block_optin() const { return smem_optin_; }

private:
	CUdevice dev_ = 0;
	CUcontext ctx_ = nullptr;
	CUstream stream_ = nullptr;
	std::vector<GpuBuffer*> owned_;
	std::vector<CUevent> events_;
	std::vector<char> event_signalled_;
	AdapterInfo adapter_;
	std::string arch_ = "sm_86";
	int smem_optin_ = 48 * 1024;
	int sm_count_ = 0;
	// Device bytes the CUDA context itself holds, sampled once right after
	// cuCtxCreate and before this engine allocates anything. On WDDM it is about
	// 1 GB, and it is invisible to the VRAM ledger - which is why the accountant
	// has to subtract it before comparing device usage against a budget, or every
	// check is ~1 GB too strict. Measured, not assumed: see
	// It was measured, not assumed.
	u64 baseline_bytes_ = 0;
	// Bytes DXGI attributes to this process before it holds anything of the
	// engine's own (the WDDM charge for the CUDA context). Subtracted from every
	// later driver figure so `usage`/`budget` come back in ledger units.
	u64 driver_baseline_usage_ = 0;
	bool lost_ = false;
};

}  // namespace phi::media
