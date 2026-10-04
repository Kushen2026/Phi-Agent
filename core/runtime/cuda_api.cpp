// CUDA driver API + NVRTC loader.
//
// Two jobs:
//   * find and load nvrtc64_*.dll / nvrtc-builtins64_*.dll / nvcuda.dll, and
//   * resolve the symbol table in cuda_api.hpp, tolerating the _v2 suffixes the
//     driver API grew over the years.
//
// Why this exists at all: nvcc cannot run on a machine without MSVC's cl.exe, so
// the engine never invokes it. NVRTC compiles CUDA C++ in-process to PTX, and the
// driver JITs that. The build therefore stays a plain mingw one and phi.exe keeps
// needing nothing but DLLs next to it.
#include "runtime/cuda_api.hpp"

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace phi::media {

namespace {

const char* const kCudaRegRoot = "SOFTWARE\\NVIDIA GPU Computing Toolkit\\CUDA";

// Candidate directories that may hold the CUDA runtime DLLs, best first.
// The vendored install is preferred so a machine with a *different* CUDA
// version still runs the build it was compiled against; failing that, every
// CUDA toolkit the machine actually has is tried (see cuda_roots).
std::vector<std::string> runtime_dirs() {
	std::vector<std::string> out;

	char exe[MAX_PATH] = {0};
	if (GetModuleFileNameA(nullptr, exe, MAX_PATH)) {
		std::string dir(exe);
		size_t slash = dir.find_last_of("\\/");
		if (slash != std::string::npos) {
			dir.resize(slash);
			out.push_back(dir + "\\cuda\\bin");
			out.push_back(dir + "\\..\\vendor\\cuda\\bin");
		}
	}
	if (const char* p = getenv("PHI_CUDA_BIN")) out.push_back(p);
	if (const char* p = getenv("CUDA_PATH")) {
		out.push_back(std::string(p) + "\\bin\\x64");
		out.push_back(std::string(p) + "\\bin");
	}
	// pip-installed runtime packages (`pip install nvidia-cuda-nvrtc-cu12`): the
	// DLL lives under site-packages/nvidia/cuda_nvrtc/bin and is the exact copy a
	// user with a 12.8/12.9 driver would already have on disk.
	for (const char* base : {"LOCALAPPDATA", "APPDATA", "USERPROFILE"}) {
		if (const char* p = getenv(base)) {
			out.push_back(std::string(p) +
			               "\\Programs\\Python\\Python311\\Lib\\site-packages\\nvidia\\cuda_nvrtc\\bin");
			out.push_back(std::string(p) +
			               "\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\nvidia\\cuda_nvrtc\\bin");
		}
	}
	return out;
}

// Every CUDA install root we can find, newest first: the registry, the standard
// install tree (every `v*` version present, not a hard-coded pair), and the pip
// nvrtc package under the user profile. Used for both the DLLs and the headers.
std::vector<std::string> cuda_roots() {
	std::vector<std::string> out;
	HKEY key = nullptr;
	if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, kCudaRegRoot, 0, KEY_READ, &key) == ERROR_SUCCESS) {
		char name[256];
		for (DWORD i = 0;; i++) {
			DWORD n = sizeof(name);
			if (RegEnumKeyExA(key, i, name, &n, nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS)
				break;
			HKEY sub = nullptr;
			if (RegOpenKeyExA(key, name, 0, KEY_READ, &sub) == ERROR_SUCCESS) {
				char val[MAX_PATH];
				DWORD sz = sizeof(val), type = 0;
				if (RegQueryValueExA(sub, "InstallDir", nullptr, &type, (LPBYTE)val, &sz) ==
				    ERROR_SUCCESS) {
					std::string d(val);
					while (!d.empty() && (d.back() == '\\' || d.back() == '/')) d.pop_back();
					out.push_back(d);
				}
				RegCloseKey(sub);
			}
		}
		RegCloseKey(key);
	}
	// The standard install tree: enumerate every version directory actually present
	// (v12.8, v12.9, v13.0, ...) so a machine whose toolkit is not one of the two
	// hard-coded defaults is still found.
	{
		const char* root = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA";
		WIN32_FIND_DATAA fd{};
		std::string glob = std::string(root) + "\\v*";
		HANDLE h = FindFirstFileA(glob.c_str(), &fd);
		if (h != INVALID_HANDLE_VALUE) {
			do {
				if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
					out.push_back(std::string(root) + "\\" + fd.cFileName);
			} while (FindNextFileA(h, &fd));
			FindClose(h);
		}
	}
	// Newest first: v12.0 > v9.0 lexically is wrong, so sort on the numeric tail.
	std::sort(out.begin(), out.end(), [](const std::string& a, const std::string& b) {
		auto ver = [](const std::string& s) {
			size_t v = s.find_last_of("\\/");
			std::string t = (v == std::string::npos) ? s : s.substr(v + 1);
			if (t.size() > 1 && (t[0] == 'v' || t[0] == 'V')) t = t.substr(1);
			int maj = 0, min = 0;
			sscanf(t.c_str(), "%d.%d", &maj, &min);
			return maj * 100 + min;
		};
		if (ver(a) != ver(b)) return ver(a) > ver(b);
		return a > b;
	});
	return out;
}

// Numeric "12.8" of a directory name / a version string; -1 when unparsable.
int version_key(const std::string& s) {
	size_t v = s.find_last_of("\\/");
	std::string t = (v == std::string::npos) ? s : s.substr(v + 1);
	if (t.size() > 1 && (t[0] == 'v' || t[0] == 'V')) t = t.substr(1);
	int maj = 0, min = 0;
	if (sscanf(t.c_str(), "%d.%d", &maj, &min) != 2) return -1;
	return maj * 100 + min;
}

// First file in `dir` matching `pattern` ("nvrtc64_*.dll"), full path.
std::string find_in(const std::string& dir, const char* pattern) {
	std::string glob = dir + "\\" + pattern;
	WIN32_FIND_DATAA fd{};
	HANDLE h = FindFirstFileA(glob.c_str(), &fd);
	if (h == INVALID_HANDLE_VALUE) return {};
	std::string found = dir + "\\" + fd.cFileName;
	FindClose(h);
	return found;
}

// Loads `pattern` from the first directory that has it, then by bare name (PATH).
[[maybe_unused]] void* load_first(const char* pattern, const std::vector<std::string>& dirs, std::string* where) {
	for (const std::string& d : dirs) {
		std::string full = find_in(d, pattern);
		if (full.empty()) continue;
		// nvrtc-builtins must be loaded before nvrtc asks for it, and with a full
		// path so the loader does not go hunting through PATH for a mismatched copy.
		void* h = (void*)LoadLibraryExA(full.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
		if (h) {
			if (where) *where = full;
			return h;
		}
	}
	void* h = (void*)LoadLibraryA(pattern);
	return h;
}

template <class T>
void bind(void* dll, T& fn, const char* name, const char* alt = nullptr) {
	if (!dll) return;
	fn = (T)(void*)GetProcAddress((HMODULE)dll, name);
	if (!fn && alt) fn = (T)(void*)GetProcAddress((HMODULE)dll, alt);
}

// Loads a *matched* nvrtc64_*.dll + nvrtc-builtins64_*.dll pair from the same
// directory. Both must come from the same toolkit build: nvrtc resolves its
// builtins by bare name, and a mismatched pair fails at the first kernel. The
// directories are tried in the order the caller hands them (vendored first, then
// the machine's own toolkits sorted by how close their version is to the driver's
// - see load()), so this never needs a hard-coded CUDA version.
void* load_nvrtc_pair(const std::vector<std::string>& dirs, std::string* builtins_path,
                      void** builtins_handle) {
	if (builtins_path) builtins_path->clear();
	if (builtins_handle) *builtins_handle = nullptr;
	for (const std::string& d : dirs) {
		std::string nv = find_in(d, "nvrtc64_*.dll");
		if (nv.empty()) continue;
		std::string bi = find_in(d, "nvrtc-builtins64_*.dll");
		if (bi.empty()) continue;
		void* hb = (void*)LoadLibraryExA(bi.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
		if (!hb) continue;
		void* hn = (void*)LoadLibraryExA(nv.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
		if (!hn) {
			FreeLibrary((HMODULE)hb);
			continue;
		}
		if (builtins_path) *builtins_path = bi;
		if (builtins_handle) *builtins_handle = hb;
		return hn;
	}
	// Last resort: a bare name from PATH (the driver-installed runtime, or a
	// user's own copy). Ordered newest-first; the loader only needs one to work.
	static const char* kNames[] = {"nvrtc64_130_0.dll", "nvrtc64_129_0.dll", "nvrtc64_128_0.dll",
	                               "nvrtc64_120_0.dll", "nvrtc64_0.dll"};
	for (const char* n : kNames) {
		void* hn = (void*)LoadLibraryA(n);
		if (hn) return hn;
	}
	return nullptr;
}

}  // namespace

bool CudaApi::load(std::string* why) {
	if (tried_) {
		if (!valid() && why) *why = why_;
		return valid();
	}
	tried_ = true;

	// ── driver first: its version steers which toolkit's nvrtc is preferred ──
	driver_dll_ = (void*)LoadLibraryA("nvcuda.dll");
	if (driver_dll_) {
		void* h = driver_dll_;
		bind(h, Init, "cuInit");
		bind(h, DriverGetVersion, "cuDriverGetVersion");
		bind(h, DeviceGetCount, "cuDeviceGetCount");
		bind(h, DeviceGet, "cuDeviceGet");
		bind(h, DeviceGetName, "cuDeviceGetName");
		bind(h, DeviceGetAttribute, "cuDeviceGetAttribute");
		bind(h, DeviceTotalMem, "cuDeviceTotalMem_v2", "cuDeviceTotalMem");
		bind(h, CtxCreate, "cuCtxCreate_v2", "cuCtxCreate");
		bind(h, CtxDestroy, "cuCtxDestroy_v2", "cuCtxDestroy");
		bind(h, CtxSetCurrent, "cuCtxSetCurrent");
		bind(h, CtxPushCurrent, "cuCtxPushCurrent_v2", "cuCtxPushCurrent");
		bind(h, CtxPopCurrent, "cuCtxPopCurrent_v2", "cuCtxPopCurrent");
		bind(h, CtxSynchronize, "cuCtxSynchronize");
		bind(h, MemGetInfo, "cuMemGetInfo_v2", "cuMemGetInfo");
		bind(h, MemAlloc, "cuMemAlloc_v2", "cuMemAlloc");
		bind(h, MemFree, "cuMemFree_v2", "cuMemFree");
		bind(h, MemAllocHost, "cuMemAllocHost_v2", "cuMemAllocHost");
		bind(h, MemFreeHost, "cuMemFreeHost");
		bind(h, MemcpyHtoD, "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
		bind(h, MemcpyDtoH, "cuMemcpyDtoH_v2", "cuMemcpyDtoH");
		bind(h, MemcpyDtoD, "cuMemcpyDtoD_v2", "cuMemcpyDtoD");
		bind(h, MemsetD8, "cuMemsetD8_v2", "cuMemsetD8");
		bind(h, MemsetD32, "cuMemsetD32_v2", "cuMemsetD32");
		bind(h, MemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync");
		bind(h, MemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync");
		bind(h, ModuleLoadData, "cuModuleLoadData");
		bind(h, ModuleUnload, "cuModuleUnload");
		bind(h, ModuleGetFunction, "cuModuleGetFunction");
		bind(h, FuncSetAttribute, "cuFuncSetAttribute");
		bind(h, LaunchKernel, "cuLaunchKernel");
		bind(h, StreamCreate, "cuStreamCreate");
		bind(h, StreamDestroy, "cuStreamDestroy_v2", "cuStreamDestroy");
		bind(h, StreamSynchronize, "cuStreamSynchronize");
		bind(h, EventCreate, "cuEventCreate");
		bind(h, EventRecord, "cuEventRecord");
		bind(h, EventSynchronize, "cuEventSynchronize");
		bind(h, EventElapsedTime, "cuEventElapsedTime");
		bind(h, EventDestroy, "cuEventDestroy_v2", "cuEventDestroy");
		bind(h, GetErrorName, "cuGetErrorName");
		bind(h, GetErrorString, "cuGetErrorString");
		device_ok = Init && DeviceGetCount && CtxCreate && MemAlloc && LaunchKernel &&
		            ModuleLoadData && ModuleGetFunction;
		if (device_ok && DriverGetVersion) {
			int v = 0;
			if (DriverGetVersion(&v) == CUDA_SUCCESS)
				driver_version_ = std::to_string(v / 1000) + "." +
				                  std::to_string((v % 1000) / 10);
		}
	}

	// ── nvrtc ──────────────────────────────────────────────────────────────
	// The vendored copy next to the exe is tried first (it is the build the engine
	// was validated against); then every CUDA toolkit the machine actually has,
	// ordered by how close its version is to the driver's, so a 12.8 driver with
	// only a 12.8 toolkit picks 12.8's nvrtc and never a 13.0 one it cannot use.
	// Nothing is hard-coded to a single CUDA version, and no DLL is compiled at
	// startup.
	std::vector<std::string> dirs = runtime_dirs();
	{
		const int driver_key = driver_version_.empty() ? -1 : version_key(driver_version_);
		std::vector<std::string> roots = cuda_roots();
		if (driver_key >= 0) {
			std::stable_sort(roots.begin(), roots.end(),
			                 [&](const std::string& a, const std::string& b) {
				                 const int ka = version_key(a), kb = version_key(b);
				                 const int da = ka < 0 ? (1 << 20) : std::abs(ka - driver_key);
				                 const int db = kb < 0 ? (1 << 20) : std::abs(kb - driver_key);
				                 return da < db;
				                 });
		}
		for (const std::string& r : roots) {
			dirs.push_back(r + "\\bin\\x64");
			dirs.push_back(r + "\\bin");
		}
	}
	std::string builtins_path;
	void* builtins_handle = nullptr;
	nvrtc_dll_ = load_nvrtc_pair(dirs, &builtins_path, &builtins_handle);
	builtins_dll_ = builtins_handle;
	if (nvrtc_dll_) {
		void* h = (void*)nvrtc_dll_;
		bind(h, NvrtcCreateProgram, "nvrtcCreateProgram");
		bind(h, NvrtcDestroyProgram, "nvrtcDestroyProgram");
		bind(h, NvrtcCompileProgram, "nvrtcCompileProgram");
		bind(h, NvrtcGetPTXSize, "nvrtcGetPTXSize");
		bind(h, NvrtcGetPTX, "nvrtcGetPTX");
		bind(h, NvrtcGetCUBINSize, "nvrtcGetCUBINSize");
		bind(h, NvrtcGetCUBIN, "nvrtcGetCUBIN");
		bind(h, NvrtcGetProgramLogSize, "nvrtcGetProgramLogSize");
		bind(h, NvrtcGetProgramLog, "nvrtcGetProgramLog");
		bind(h, NvrtcVersion, "nvrtcVersion");
		nvrtc_ok = NvrtcCreateProgram && NvrtcCompileProgram && NvrtcGetProgramLog;
		if (nvrtc_ok) {
			int maj = 0, min = 0;
			if (NvrtcVersion) NvrtcVersion(&maj, &min);
			nvrtc_version_ = std::to_string(maj) + "." + std::to_string(min);
		}
	}

	// ── headers ────────────────────────────────────────────────────────────
	// NVRTC's -I does not reliably accept a path with spaces, so a kernel that
	// needs <cuda_fp16.h> must be given a space-free directory. Prefer the copy
	// vendored next to the exe; fall back to the toolkit install only if it has
	// no spaces.
	{
		char exe[MAX_PATH] = {0};
		std::vector<std::string> cands;
		if (GetModuleFileNameA(nullptr, exe, MAX_PATH)) {
			std::string dir(exe);
			size_t slash = dir.find_last_of("\\/");
			if (slash != std::string::npos) {
				dir.resize(slash);
				cands.push_back(dir + "\\cuda\\include");
				cands.push_back(dir + "\\..\\vendor\\cuda\\include");
			}
		}
		if (const char* p = getenv("PHI_CUDA_INCLUDE")) cands.push_back(p);
		for (const std::string& r : cuda_roots()) cands.push_back(r + "\\include");
		for (const std::string& c : cands) {
			WIN32_FILE_ATTRIBUTE_DATA fad{};
			if (!GetFileAttributesExA((c + "\\cuda_runtime_api.h").c_str(), GetFileExInfoStandard,
			                          &fad) &&
			    !GetFileAttributesExA((c + "\\cuda_fp16.h").c_str(), GetFileExInfoStandard, &fad))
				continue;
			if (c.find(' ') != std::string::npos) continue;  // NVRTC cannot take this
			include_dir_ = c;
			break;
		}
		if (!include_dir_.empty()) {
			std::string opt = "-I" + include_dir_;
			include_opts_.push_back(opt);
		}
	}

	if (!valid()) {
		why_ = "CUDA backend unavailable: ";
		if (!driver_dll_|| !device_ok) why_ += "nvcuda.dll/driver symbols missing; ";
		if (!nvrtc_ok) why_ += "nvrtc64_*.dll/NVRTC symbols missing; ";
		why_ += "(looked in the exe's cuda\\bin, ../vendor/cuda/bin, $CUDA_PATH and the registry)";
		if (why) *why = why_;
	}
	return valid();
}

std::string CudaApi::error_text(CUresult r) const {
	if (r == CUDA_SUCCESS) return "CUDA_SUCCESS";
	std::string out = "CUDA error " + std::to_string(r);
	if (GetErrorName) {
		const char* n = nullptr;
		if (GetErrorName(r, &n) == CUDA_SUCCESS && n) out = n;
	}
	if (GetErrorString) {
		const char* s = nullptr;
		if (GetErrorString(r, &s) == CUDA_SUCCESS && s) {
			out += " (";
			out += s;
			out += ")";
		}
	}
	return out;
}

CudaApi& cuda_api() {
	static CudaApi api;
	static bool once = (api.load(nullptr), true);
	(void)once;
	return api;
}

}  // namespace phi::media
