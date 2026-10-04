// CUDA device/context implementation.
#include "runtime/cuda_device.hpp"

#include <windows.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <cstring>
#include <mutex>

#include "host/st.hpp"
#include "runtime/vram_budget.hpp"

namespace phi::media {

namespace {

// Turns a driver failure into a MediaError that names the call and the reason,
// so a refusal reads like a diagnosis instead of a number.
void check(CUresult r, const char* what) {
	if (r == CUDA_SUCCESS) return;
	throw MediaError(std::string("CUDA: ") + what + " failed: " + cuda_api().error_text(r));
}

}  // namespace

// ── the driver's per-process video-memory budget (DXGI) ─────────────────────
//
// One factory, one adapter object, cached for the life of the process: creating
// a factory costs milliseconds and the accountant's live probe runs on every
// charge, so the only per-call work must be `QueryVideoMemoryInfo` itself (a few
// microseconds, straight out of the driver's accounting). The adapter is picked
// once, by dedicated VRAM against the CUDA device the engine opened, so a laptop
// with a discrete NVIDIA part and an integrated GPU never reads the wrong one's
// budget.
namespace {

IDXGIAdapter3* dxgi_adapter_for(u64 dedicated_vram) {
	struct Cache {
		std::mutex mu;
		IDXGIAdapter3* adapter = nullptr;
		u64 matched = 0;
		bool tried = false;
	};
	static Cache c;
	std::lock_guard<std::mutex> lk(c.mu);
	if (!c.tried) {
		c.tried = true;
		IDXGIFactory1* factory = nullptr;
		if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory)) && factory) {
			// Best match first: the adapter whose dedicated memory is closest to the
			// one CUDA reported. A software adapter reports 0 and is never picked while
			// a hardware one has more.
			u64 best_delta = ~0ull;
			for (UINT i = 0;; i++) {
				IDXGIAdapter1* a1 = nullptr;
				if (factory->EnumAdapters1(i, &a1) == DXGI_ERROR_NOT_FOUND || !a1) break;
				DXGI_ADAPTER_DESC1 d{};
				if (SUCCEEDED(a1->GetDesc1(&d)) && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
					IDXGIAdapter3* a3 = nullptr;
					if (SUCCEEDED(a1->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&a3)) && a3) {
						const u64 have = (u64)d.DedicatedVideoMemory;
						const u64 delta = have > dedicated_vram ? have - dedicated_vram
						                                        : dedicated_vram - have;
						if (delta < best_delta) {
							best_delta = delta;
							if (c.adapter) c.adapter->Release();
							c.adapter = a3;
							c.matched = have;
						} else {
							a3->Release();
						}
					}
				}
				a1->Release();
			}
			factory->Release();
		}
	}
	return c.adapter;
}

}  // namespace

DriverVram query_driver_vram(const AdapterInfo& cuda_adapter) {
	DriverVram v;
	IDXGIAdapter3* a3 = dxgi_adapter_for(cuda_adapter.dedicated_vram);
	if (!a3) return v;
	DXGI_QUERY_VIDEO_MEMORY_INFO info{};
	if (FAILED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) return v;
	v.budget = (u64)info.Budget;
	v.usage = (u64)info.CurrentUsage;
	v.reservation = (u64)info.AvailableForReservation;
	v.ok = v.budget != 0;
	return v;
}

std::string AdapterInfo::describe() const {
	std::string s = name.empty() ? "CUDA device" : name;
	s += " (sm_" + std::to_string(cc_major) + std::to_string(cc_minor);
	s += ", " + format_bytes(dedicated_vram) + ")";
	return s;
}

std::vector<AdapterInfo> enumerate_adapters() {
	std::vector<AdapterInfo> out;
	CudaApi& api = cuda_api();
	if (!api.driver_ok()) return out;
	if (api.Init(0) != CUDA_SUCCESS) return out;
	int count = 0;
	if (api.DeviceGetCount(&count) != CUDA_SUCCESS) return out;
	for (int i = 0; i < count; i++) {
		CUdevice dev = 0;
		if (api.DeviceGet(&dev, i) != CUDA_SUCCESS) continue;
		AdapterInfo a;
		a.index = i;
		char name[256] = {0};
		if (api.DeviceGetName(name, sizeof(name), dev) == CUDA_SUCCESS) a.name = name;
		int maj = 0, min = 0;
		api.DeviceGetAttribute(&maj, kAttrComputeCapabilityMajor, dev);
		api.DeviceGetAttribute(&min, kAttrComputeCapabilityMinor, dev);
		a.cc_major = maj;
		a.cc_minor = min;
		size_t total = 0;
		if (api.DeviceTotalMem && api.DeviceTotalMem(&total, dev) == CUDA_SUCCESS)
			a.dedicated_vram = total;
		// A device with no unified addressing cannot run the engine at all; treat it
		// as software so the automatic pick skips it.
		a.software = (maj == 0);
		out.push_back(a);
	}
	std::stable_sort(out.begin(), out.end(), [](const AdapterInfo& a, const AdapterInfo& b) {
		if (a.software != b.software) return !a.software;
		return a.dedicated_vram > b.dedicated_vram;
	});
	return out;
}

CudaContext::~CudaContext() { destroy(); }

void CudaContext::create(int device_index) {
	destroy();
	std::string why;
	CudaApi& api = cuda_api();
	if (!api.load(&why)) throw MediaError(why);

	std::vector<AdapterInfo> adapters = enumerate_adapters();
	if (adapters.empty()) throw MediaError("CUDA: no device found");

	if (device_index < 0) {
		adapter_ = adapters.front();
	} else {
		bool found = false;
		for (const auto& a : adapters) {
			if (a.index == device_index) {
				adapter_ = a;
				found = true;
				break;
			}
		}
		if (!found)
			throw MediaError("CUDA: device index " + std::to_string(device_index) +
			                 " not present");
	}
	vram_budget().note_dedicated_vram(adapter_.dedicated_vram);

	check(api.DeviceGet(&dev_, adapter_.index), "cuDeviceGet");
	check(api.CtxCreate(&ctx_, 0, dev_), "cuCtxCreate");
	// The primary context is what the driver hands out; make it current on this
	// thread so every later driver call (and NVRTC's, indirectly) sees it.
	check(api.CtxSetCurrent(ctx_), "cuCtxSetCurrent");
	check(api.StreamCreate(&stream_, 0), "cuStreamCreate");

	// Hand the host-side weight caches a page-locked allocator (see
	// SafeTensors::set_host_allocator). The streamer moves ~17 GB per sampling
	// step host -> device, and when the source buffer is locked the driver can
	// DMA straight out of it instead of copying it into the ring first. The cap
	// is deliberately a *fraction of free RAM*: page-locked memory cannot be
	// paged out, and locking the ~7 GB the pinned set wants on a 16 GB machine
	// would leave the OS none. Past the cap the caches fall back to malloc and
	// the streamer to the ring - slower, never wrong.
	//
	// The fraction is a *reservation*, not a target: whatever is left past the
	// cache's own share is what the look-ahead slot gets, and the slot is the
	// more valuable of the two (it covers the block the streamer has not asked
	// for yet, and the ring copy it avoids is a full pass over every streamed
	// weight). At free/3 with a 4 GB ceiling the cache's 7.9 GB request consumed
	// the whole cap first and the slot fell back to malloc on every block - the
	// measured cost was a host memcpy of ~11 GB per sampling step.
	{
		const u64 free_ram = SafeTensors::host_free_bytes();
		u64 cap = free_ram / 2;
		const u64 hard = 7ull << 30;
		if (cap > hard) cap = hard;
		SafeTensors::set_host_allocator(
		    [](size_t n) -> void* {
			    void* p = nullptr;
			    if (cuda_api().MemAllocHost(&p, n, kMemHostAllocPortable) != CUDA_SUCCESS) return nullptr;
			    return p;
		    },
		    [](void* p) { cuda_api().MemFreeHost(p); }, cap);
	}

	// Sample what the context costs before we allocate anything of our own. The
	// accountant needs this to translate between "device bytes committed" and
	// "bytes this engine is responsible for": without it the ~1 GB the WDDM
	// context reserves looks like our usage and every budget check is that much
	// too strict (it is what made 1080p and even 512x320 video start refusing).
	{
		size_t free_b = 0, total_b = 0;
		if (api.MemGetInfo && api.MemGetInfo(&free_b, &total_b) == CUDA_SUCCESS)
			baseline_bytes_ = (u64)(total_b - free_b);
	}

	// The driver's own budget for this process, sampled before this engine holds a
	// single byte of its own: DXGI's figure is what WDDM will really let the
	// process commit in local video memory, and the usage measured here is the part
	// of it that exists whether the engine works or not (the CUDA context's WDDM
	// charge). The accountant plans against the difference - the process's own
	// allowance - instead of deriving a ceiling from the card's total, which is the
	// number that let a 6 GB part be planned as if it had 5.76 GB of spendable
	// local memory when the driver was only willing to back ~5.0 GB.
	{
		const DriverVram drv = query_driver_vram(adapter_);
		if (drv.ok) {
			driver_baseline_usage_ = drv.usage;
			vram_budget().note_driver_vram(drv.budget, drv.usage, drv.reservation);
		}
	}

	int smem = 0;
	if (api.DeviceGetAttribute(&smem, kAttrMaxSharedMemoryPerBlockOptin, dev_) == CUDA_SUCCESS)
		smem_optin_ = smem;
	int smc = 0;
	if (api.DeviceGetAttribute(&smc, kAttrMultiProcessorCount, dev_) == CUDA_SUCCESS)
		sm_count_ = smc;
	arch_ = "sm_" + std::to_string(adapter_.cc_major) + std::to_string(adapter_.cc_minor);

	// Keep the accountant's view of the driver budget live: free device memory
	// moves every time another process claims or releases VRAM, and planning
	// against a stale figure is how a run ends up demoting pages and taking a
	// TDR.
	vram_budget().set_live_probe([this] {
		VramBudget::LiveVram v;
		size_t free_b = 0, total_b = 0;
		const bool cuda_ok = cuda_api().MemGetInfo &&
		                     cuda_api().MemGetInfo(&free_b, &total_b) == CUDA_SUCCESS;
		if (cuda_ok) v.total = (u64)total_b;
		// The driver's own accounting first: it is the wall, and it is the only
		// figure that keeps falling once the process is past its local budget
		// (`cuMemGetInfo` pins at 0 while the driver quietly spills to system
		// memory, which is a TDR rather than a slowdown). `usage`/`budget` are
		// reported in ledger units - the process's allowance and what it holds of
		// it - by subtracting the non-engine baseline measured at create().
		const DriverVram drv = query_driver_vram(adapter_);
		if (drv.ok) {
			const u64 base = driver_baseline_usage_;
			const u64 ours = drv.usage > base ? drv.usage - base : 0;
			const u64 budget = drv.budget > base ? drv.budget - base : 0;
			v.usage = ours;
			v.budget = budget ? budget : ours;
			v.free = drv.budget > drv.usage ? drv.budget - drv.usage : 0;
			if (cuda_ok && (u64)free_b < v.free) v.free = (u64)free_b;
			return v;
		}
		if (cuda_ok) {
			// No DXGI (an exotic driver, or a machine whose adapter list does not
			// match): fall back to the CUDA figure, which is the old behaviour.
			const u64 committed = (u64)(total_b - free_b);
			const u64 ours = committed > baseline_bytes_ ? committed - baseline_bytes_ : 0;
			v.usage = ours;
			v.budget = ours + (u64)free_b;
			v.free = (u64)free_b;
		}
		return v;
	});
}

void CudaContext::destroy() {
	CudaApi& api = cuda_api();
	if (ctx_ && api.valid()) {
		// Drop every allocation we still own. A failed run may leave the ladder
		// half-built; freeing here keeps the next run's budget honest.
		for (GpuBuffer* b : owned_) {
			if (!b) continue;
			if (b->mapped && b->host_visible && api.MemFreeHost) api.MemFreeHost(b->mapped);
			else if (b->base && api.MemFree) api.MemFree(b->base);
			delete b;
		}
		owned_.clear();
		for (CUevent e : events_)
			if (e && api.EventDestroy) api.EventDestroy(e);
		events_.clear();
		event_signalled_.clear();
		if (stream_ && api.StreamDestroy) api.StreamDestroy(stream_);
		stream_ = nullptr;
		if (api.CtxSynchronize) api.CtxSynchronize();
		if (api.CtxDestroy) api.CtxDestroy(ctx_);
	}
	ctx_ = nullptr;
	stream_ = nullptr;
	dev_ = 0;
	lost_ = false;
}

void CudaContext::sync() {
	if (!stream_) return;
	CUresult r = cuda_api().StreamSynchronize(stream_);
	if (r == 0) return;
	// Only errors that actually take the context down mark the device lost
	// (same filter as dispatch). Marking every stream error lost made a
	// transient launch-timeout (701/702) look like a dead card until restart.
	if (r == 700 || r == 719) lost_ = true;
	throw MediaError("CUDA: stream synchronise failed: " + cuda_api().error_text(r));
}

u64 CudaContext::signal_fence() {
	CudaApi& api = cuda_api();
	CUevent e = nullptr;
	check(api.EventCreate(&e, 0), "cuEventCreate");
	CUresult r = api.EventRecord(e, stream_);
	if (r != CUDA_SUCCESS) {
		if (api.EventDestroy) api.EventDestroy(e);
		throw MediaError(std::string("CUDA: cuEventRecord failed: ") + api.error_text(r));
	}
	events_.push_back(e);
	event_signalled_.push_back(0);
	return events_.size();   // 1-based slot index
}

void CudaContext::wait_fence(u64 value) {
	CudaApi& api = cuda_api();
	if (value == 0 || value > events_.size()) return;
	check(api.EventSynchronize(events_[(size_t)value - 1]), "cuEventSynchronize");
}

bool CudaContext::fence_reached(u64 value) const {
	// CUDA has no non-blocking "is this event done"; the honest answer is to wait.
	// The scheduler only uses this on a path where waiting is acceptable, so a
	// blocking answer is correct, just not lazy.
	if (value == 0 || value > events_.size()) return true;
	return cuda_api().EventSynchronize(events_[(size_t)value - 1]) == CUDA_SUCCESS;
}

GpuBuffer* CudaContext::alloc(u64 bytes) {
	if (bytes == 0) bytes = 256;
	auto* b = new GpuBuffer();
	b->bytes = bytes;
	// Above 2 MB the driver must not silently demote; a failure here is a real
	// "does not fit" and the accountant's refusal should have caught it first.
	check(cuda_api().MemAlloc(&b->base, (size_t)bytes), "cuMemAlloc");
	owned_.push_back(b);
	return b;
}

GpuBuffer* CudaContext::alloc_host(u64 bytes, void** mapped) {
	if (bytes == 0) bytes = 256;
	auto* b = new GpuBuffer();
	b->bytes = bytes;
	b->host_visible = true;
	check(cuda_api().MemAllocHost(&b->mapped, (size_t)bytes, kMemHostAllocPortable),
	      "cuMemAllocHost");
	b->base = (CUdeviceptr)b->mapped;
	if (mapped) *mapped = b->mapped;
	owned_.push_back(b);
	return b;
}

void CudaContext::free(GpuBuffer* b) {
	if (!b) return;
	CudaApi& api = cuda_api();
	for (size_t i = 0; i < owned_.size(); i++) {
		if (owned_[i] != b) continue;
		if (b->mapped && b->host_visible && api.MemFreeHost) api.MemFreeHost(b->mapped);
		else if (b->base && api.MemFree) api.MemFree(b->base);
		owned_.erase(owned_.begin() + (long)i);
		delete b;
		return;
	}
}

void CudaContext::copy_h2d(GpuBuffer* dst, u64 dst_off, const void* src, u64 bytes) {
	if (!bytes) return;
	check(cuda_api().MemcpyHtoD(dst->base + dst_off, src, (size_t)bytes), "cuMemcpyHtoD");
}

void CudaContext::copy_h2d_async(GpuBuffer* dst, u64 dst_off, const void* src, u64 bytes) {
	if (!bytes) return;
	check(cuda_api().MemcpyHtoDAsync(dst->base + dst_off, src, (size_t)bytes, stream_),
	      "cuMemcpyHtoDAsync");
}

void CudaContext::copy_d2h(GpuBuffer* src, u64 src_off, void* dst, u64 bytes) {
	if (!bytes) return;
	check(cuda_api().MemcpyDtoH(dst, src->base + src_off, (size_t)bytes), "cuMemcpyDtoH");
}

void CudaContext::copy_d2d(GpuBuffer* dst, u64 dst_off, GpuBuffer* src, u64 src_off, u64 bytes) {
	if (!bytes) return;
	check(cuda_api().MemcpyDtoD(dst->base + dst_off, src->base + src_off, (size_t)bytes),
	      "cuMemcpyDtoD");
}

void CudaContext::memset_d8(GpuBuffer* dst, u64 dst_off, u8 value, u64 bytes) {
	if (!bytes) return;
	check(cuda_api().MemsetD8(dst->base + dst_off, value, (size_t)bytes), "cuMemsetD8");
}

VideoMemoryBudget CudaContext::query_budget() const {
	VideoMemoryBudget b;
	if (!ctx_) return b;
	size_t free_b = 0, total_b = 0;
	if (!cuda_api().MemGetInfo || cuda_api().MemGetInfo(&free_b, &total_b) != CUDA_SUCCESS) return b;
	// Same conversion as the live probe: CUDA's `free` excludes what this
	// engine already holds, so it has to be added back to get the ceiling.
	const u64 ours = vram_budget().local() + vram_budget().staging();
	b.budget = (u64)free_b + ours;
	b.current_usage = ours;
	b.process_usage = ours;   // the driver API has no per-process breakdown
	b.available_reservation = 0;
	vram_budget().note_os_budget(b.budget);
	return b;
}

bool CudaContext::supports_int8_dot() const { return adapter_.cc_major >= 6; }
bool CudaContext::supports_fp16_pack() const { return adapter_.cc_major >= 6; }
bool CudaContext::supports_tensor_cores() const { return adapter_.cc_major >= 7; }

std::string CudaContext::capabilities() const {
	if (!ctx_) return "no device";
	std::string s = "sm_" + std::to_string(adapter_.cc_major) + std::to_string(adapter_.cc_minor);
	s += ", " + std::to_string(sm_count_) + " SMs";
	s += supports_int8_dot() ? ", int8 dp4a" : ", no int8 dp4a";
	s += supports_tensor_cores() ? ", tensor cores" : ", no tensor cores";
	s += ", " + format_bytes((u64)smem_optin_) + " shared/block";
	s += ", driver " + cuda_api().driver_version() + ", nvrtc " + cuda_api().nvrtc_version();
	return s;
}

}  // namespace phi::media
