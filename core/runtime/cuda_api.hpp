// CUDA driver API + NVRTC, loaded at runtime.
//
// The two DLLs are nvrtc64_*.dll + nvcuda.dll. Nothing here links against the
// CUDA toolkit, so phi.exe stays a self-contained binary and - crucially -
// **needs no MSVC**: `nvcc` on Windows refuses to run without cl.exe, but NVRTC
// is a plain DLL that compiles CUDA C++ to PTX/cubin in-process. That is the
// whole reason a mingw build can drive an NVIDIA card at all.
//
// If the DLLs are absent the media engine reports a readable error and stays up,
// exactly like the missing-NVRTC path.
#pragma once

#include <string>
#include <vector>

#include "util/media_common.hpp"

namespace phi::media {

// ── minimal CUDA ABI ───────────────────────────────────────────────────────
// Declared here rather than pulled from cuda.h so the build needs no CUDA SDK
// on the include path. These are the stable driver-ABI shapes: every handle is
// an opaque pointer and every scalar is 32/64-bit.
using CUdeviceptr = unsigned long long;
using CUresult = int;
using CUdevice = int;
using CUcontext = struct CUctx_st*;
using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;
using CUstream = struct CUstream_st*;
using CUevent = struct CUevent_st*;
using nvrtcProgram = struct _nvrtcProgram*;

inline constexpr CUresult CUDA_SUCCESS = 0;

// cuDeviceGetAttribute codes we actually ask for.
enum CudaDevAttr : int {
	kAttrComputeCapabilityMajor = 75,
	kAttrComputeCapabilityMinor = 76,
	kAttrMultiProcessorCount = 16,
	kAttrMaxSharedMemoryPerBlockOptin = 97,
};

// cuFuncSetAttribute code for "shared memory this kernel may opt into".
inline constexpr int kFuncAttrMaxDynamicSharedBytes = 8;

// cuMemHostAlloc flags.
inline constexpr unsigned kMemHostAllocPortable = 1;
inline constexpr unsigned kMemHostAllocMapped = 2;

struct CudaApi {
	// ── driver ─────────────────────────────────────────────────────────────
	CUresult (*Init)(unsigned) = nullptr;
	CUresult (*DriverGetVersion)(int*) = nullptr;
	CUresult (*DeviceGetCount)(int*) = nullptr;
	CUresult (*DeviceGet)(CUdevice*, int) = nullptr;
	CUresult (*DeviceGetName)(char*, int, CUdevice) = nullptr;
	CUresult (*DeviceGetAttribute)(int*, int, CUdevice) = nullptr;
	CUresult (*DeviceTotalMem)(size_t*, CUdevice) = nullptr;

	CUresult (*CtxCreate)(CUcontext*, unsigned, CUdevice) = nullptr;
	CUresult (*CtxDestroy)(CUcontext) = nullptr;
	CUresult (*CtxSetCurrent)(CUcontext) = nullptr;
	CUresult (*CtxPushCurrent)(CUcontext) = nullptr;
	CUresult (*CtxPopCurrent)(CUcontext*) = nullptr;
	CUresult (*CtxSynchronize)() = nullptr;

	CUresult (*MemGetInfo)(size_t*, size_t*) = nullptr;
	CUresult (*MemAlloc)(CUdeviceptr*, size_t) = nullptr;
	CUresult (*MemFree)(CUdeviceptr) = nullptr;
	CUresult (*MemAllocHost)(void**, size_t, unsigned) = nullptr;
	CUresult (*MemFreeHost)(void*) = nullptr;

	CUresult (*MemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
	CUresult (*MemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
	CUresult (*MemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t) = nullptr;
	CUresult (*MemsetD8)(CUdeviceptr, unsigned char, size_t) = nullptr;
	CUresult (*MemsetD32)(CUdeviceptr, unsigned, size_t) = nullptr;
	// Async copies on an explicit stream. The weight-upload path uses these rather
	// than the blocking forms so the copy is *ordered* against the kernels the
	// same stream carries; from pinned memory they are a true DMA and return
	// immediately.
	CUresult (*MemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream) = nullptr;
	CUresult (*MemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream) = nullptr;

	CUresult (*ModuleLoadData)(CUmodule*, const void*) = nullptr;
	CUresult (*ModuleUnload)(CUmodule) = nullptr;
	CUresult (*ModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
	CUresult (*FuncSetAttribute)(CUfunction, int, int) = nullptr;
	CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
	                         unsigned, unsigned, CUstream, void**, void**) = nullptr;

	CUresult (*StreamCreate)(CUstream*, unsigned) = nullptr;
	CUresult (*StreamDestroy)(CUstream) = nullptr;
	CUresult (*StreamSynchronize)(CUstream) = nullptr;

	CUresult (*EventCreate)(CUevent*, unsigned) = nullptr;
	CUresult (*EventRecord)(CUevent, CUstream) = nullptr;
	CUresult (*EventSynchronize)(CUevent) = nullptr;
	CUresult (*EventElapsedTime)(float*, CUevent, CUevent) = nullptr;
	CUresult (*EventDestroy)(CUevent) = nullptr;

	CUresult (*GetErrorName)(CUresult, const char**) = nullptr;
	CUresult (*GetErrorString)(CUresult, const char**) = nullptr;

	// ── nvrtc ──────────────────────────────────────────────────────────────
	int (*NvrtcCreateProgram)(nvrtcProgram*, const char*, const char*, int, const char**,
	                          const char**) = nullptr;
	int (*NvrtcDestroyProgram)(nvrtcProgram*) = nullptr;
	int (*NvrtcCompileProgram)(nvrtcProgram, int, const char**) = nullptr;
	int (*NvrtcGetPTXSize)(nvrtcProgram, size_t*) = nullptr;
	int (*NvrtcGetPTX)(nvrtcProgram, char*) = nullptr;
	int (*NvrtcGetCUBINSize)(nvrtcProgram, size_t*) = nullptr;
	int (*NvrtcGetCUBIN)(nvrtcProgram, char*) = nullptr;
	int (*NvrtcGetProgramLogSize)(nvrtcProgram, size_t*) = nullptr;
	int (*NvrtcGetProgramLog)(nvrtcProgram, char*) = nullptr;
	int (*NvrtcVersion)(int*, int*) = nullptr;

	// Loads both DLLs and resolves every symbol above. Idempotent. Returns false
	// (with `why` filled) when the machine has no usable CUDA runtime; callers
	// turn that into a MediaError naming the missing piece.
	bool load(std::string* why);
	bool valid() const { return device_ok && nvrtc_ok; }
	bool driver_ok() const { return device_ok; }
	bool compiler_ok() const { return nvrtc_ok; }

	// "13.2" / "12.4" - what the driver reports, for the diagnostics block.
	const std::string& driver_version() const { return driver_version_; }
	const std::string& nvrtc_version() const { return nvrtc_version_; }
	// The nvrtc DLL actually loaded ("" when loaded by bare name / none).

	// Human-readable text for a CUresult.
	std::string error_text(CUresult r) const;

	// Where the CUDA headers live (for the NVRTC include path). Set by load():
	// $CUDA_PATH, then the registry, then the usual install dirs. Empty when the
	// headers could not be found, which is only fatal for kernels that #include
	// <cuda_fp16.h> - the engine's own kernels are self-contained.
	const std::string& include_dir() const { return include_dir_; }
	// Same list as a NVRTC option vector ("-I<dir>"), with the space-safe form.
	const std::vector<std::string>& include_opts() const { return include_opts_; }

private:
	void* driver_dll_ = nullptr;
	void* nvrtc_dll_ = nullptr;
	void* builtins_dll_ = nullptr;
	bool device_ok = false;
	bool nvrtc_ok = false;
	bool tried_ = false;
	std::string driver_version_;
	std::string nvrtc_version_;
	std::string include_dir_;
	std::vector<std::string> include_opts_;
	std::string why_;
};

// Process-wide CUDA API table. Safe to call from any thread; the DLL load is
// guarded by a function-local static.
CudaApi& cuda_api();

}  // namespace phi::media
