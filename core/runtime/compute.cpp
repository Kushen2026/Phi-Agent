// Compute layer implementation: NVRTC compilation, the on-disk kernel cache, and
// dispatch.
#include "runtime/compute.hpp"

#include <windows.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <mutex>

#include "kernels/kernels.hpp"
#include "util/base.hpp"

namespace phi::media {

namespace {

std::mutex g_status_mutex;
ShaderCompilerStatus& mutable_status() {
	static ShaderCompilerStatus s;
	return s;
}

// FNV-1a over the source + the compile options. The cache key has to change when
// the kernel text changes, or an edited kernel silently keeps running the old
// cubin - which is exactly how a shader cache turns into a ghost.
u64 fnv1a(const std::string& a, const std::string& b = {}) {
	u64 h = 1469598103934665603ull;
	auto step = [&h](const char* p, size_t n) {
		for (size_t i = 0; i < n; i++) {
			h ^= (u8)p[i];
			h *= 1099511628211ull;
		}
	};
	step(a.data(), a.size());
	step(b.data(), b.size());
	return h;
}

std::string hex16(u64 v) {
	char buf[24];
	snprintf(buf, sizeof buf, "%016llx", (unsigned long long)v);
	return buf;
}

// The cache lives under the install's data directory, and that path is UTF-8
// everywhere else in this engine (it holds non-ASCII characters on a machine
// whose user name or install folder is not ASCII). The C runtime's `fopen`
// interprets the path in the process' ANSI codepage, so on such a machine every
// cubin lookup silently missed and every launch recompiled through NVRTC - a
// slow engine rather than a broken one, which is exactly the kind of failure
// that never gets reported. `_wfopen` with the UTF-8 -> wide conversion the rest
// of the tree uses is the same call the model loaders make.
FILE* open_u8(const std::string& path, const wchar_t* mode) {
	return _wfopen(utf8_to_wide(path).c_str(), mode);
}

bool read_file(const std::string& path, std::vector<u8>* out) {
	FILE* f = open_u8(path, L"rb");
	if (!f) return false;
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (n <= 0) {
		fclose(f);
		return false;
	}
	out->resize((size_t)n);
	size_t got = fread(out->data(), 1, (size_t)n, f);
	fclose(f);
	out->resize(got);
	return got > 0;
}

void write_file(const std::string& path, const void* data, size_t n) {
	// atomic: write a temp file and rename over the target. A crash mid-write
	// would otherwise leave a partial cubin that reads as "valid cache"
	// (> 64 bytes) and then fails ModuleLoadData on every later run, with no
	// self-heal path.
	std::string tmp = path + ".tmp";
	FILE* f = open_u8(tmp, L"wb");
	if (!f) return;
	fwrite(data, 1, n, f);
	fclose(f);
	std::wstring wfrom = utf8_to_wide(path_backslash(tmp));
	std::wstring wto = utf8_to_wide(path_backslash(path));
	if (!MoveFileExW(wfrom.c_str(), wto.c_str(), MOVEFILE_REPLACE_EXISTING)) {
		DeleteFileW(wfrom.c_str());
	}
}

// ── block-shape parsing ────────────────────────────────────────────────────
//
// The thread-group size is not a CUDA attribute, so the ported sources carry it
// as a `numthreads(x,y,z)` comment and the host parses it back out. The value
// matters: every kernel with a `__syncthreads()` reduction assumes it.
//
// Tokens may be symbolic - int8_gemm writes `[numthreads(NT, 1, 1)]` with
// `static const uint NT = 256;`, and autotune rewrites NT - so a non-numeric
// token is resolved against `static const ... TOKEN = <number>;` in the same
// source.
class BlockShapeParser {
public:
	explicit BlockShapeParser(const std::string& s) : src_(s) {}

	bool parse(u32* bx, u32* by, u32* bz) {
		size_t pos = src_.find("numthreads");
		if (pos == std::string::npos) return false;
		pos += 10;
		if (pos < src_.size() && src_[pos] == '(') pos++;
		u32 got[3] = {0, 0, 0};
		for (int i = 0; i < 3; i++) {
			while (pos < src_.size() &&
			       (isspace((unsigned char)src_[pos]) || src_[pos] == ',' || src_[pos] == ')'))
				pos++;
			size_t start = pos;
			while (pos < src_.size() && (isalnum((unsigned char)src_[pos]) || src_[pos] == '_'))
				pos++;
			if (pos == start) break;
			std::string tok = src_.substr(start, pos - start);
			got[i] = (u32)constant_value(tok);
		}
		if (got[0] == 0) return false;
		*bx = got[0];
		*by = got[1] ? got[1] : 1;
		*bz = got[2] ? got[2] : 1;
		return true;
	}

	// `// smem <bytes>` - dynamic shared memory to request at launch.
	u32 smem() {
		size_t p = src_.find("smem ");
		if (p == std::string::npos) return 0;
		return (u32)strtoul(src_.c_str() + p + 5, nullptr, 10);
	}

private:
	// Finds `tok` as a whole identifier (so NT does not match inside CNT).
	size_t find_ident(const std::string& tok, size_t from) {
		size_t p = from;
		while ((p = src_.find(tok, p)) != std::string::npos) {
			const bool left_ok =
			    (p == 0) || !(isalnum((unsigned char)src_[p - 1]) || src_[p - 1] == '_');
			const size_t end = p + tok.size();
			const bool right_ok =
			    (end >= src_.size()) ||
			    !(isalnum((unsigned char)src_[end]) || src_[end] == '_');
			if (left_ok && right_ok) return p;
			p = end;
		}
		return std::string::npos;
	}

	long constant_value(const std::string& tok) {
		if (!tok.empty() && isdigit((unsigned char)tok[0])) return strtol(tok.c_str(), nullptr, 10);
		// A symbolic token resolves against its declaration, e.g.
		// `static const unsigned NT   = 128u;` (autotune's spelling, and the one the
		// ported attention kernels use). The whitespace before '=' is arbitrary and
		// the literal may carry a `u` suffix, so neither can be assumed.
		size_t p = 0;
		while ((p = find_ident(tok, p)) != std::string::npos) {
			size_t r = p + tok.size();
			size_t ws = r;
			while (ws < src_.size() && isspace((unsigned char)src_[ws])) ws++;
			if (ws > r && ws < src_.size() && src_[ws] == '=') {
				r = ws + 1;
				while (r < src_.size() && isspace((unsigned char)src_[r])) r++;
				if (r < src_.size() && isdigit((unsigned char)src_[r]))
					return strtol(src_.c_str() + r, nullptr, 10);
			}
			p += tok.size();
		}
		return 0;
	}

	const std::string& src_;
};

}  // namespace

// ── compiler status ────────────────────────────────────────────────────────

std::string ShaderCompilerStatus::describe() const {
	std::string s = nvrtc ? "nvrtc" : "no nvrtc";
	s += " (" + std::to_string(compiles) + " compiled, " + std::to_string(cache_hits) +
	     " cached)";
	if (!last_error.empty()) s += " last error: " + last_error;
	return s;
}

const ShaderCompilerStatus& shader_compiler_status() { return mutable_status(); }

int clear_shader_cache(const std::string& cache_dir) {
	if (cache_dir.empty()) return 0;
	std::string glob = path_backslash(cache_dir) + "\\*.cubin";
	WIN32_FIND_DATAW fd{};
	HANDLE h = FindFirstFileW(utf8_to_wide(glob).c_str(), &fd);
	if (h == INVALID_HANDLE_VALUE) return 0;
	const std::wstring dir = utf8_to_wide(path_backslash(cache_dir));
	int n = 0;
	do {
		std::wstring full = dir + L"\\" + fd.cFileName;
		if (DeleteFileW(full.c_str())) n++;
	} while (FindNextFileW(h, &fd));
	FindClose(h);
	return n;
}

// ── ComputeContext ─────────────────────────────────────────────────────────

ComputeContext::~ComputeContext() { destroy(); }

void ComputeContext::destroy() {
	CudaApi& api = cuda_api();
	for (GpuKernel* k : kernels_) {
		if (!k) continue;
		if (k->module && api.ModuleUnload) api.ModuleUnload(k->module);
		delete k;
	}
	kernels_.clear();
	for (GpuBuffer* b : owned_) {
		if (b && cuda_) cuda_->free(b);
	}
	owned_.clear();
	cuda_ = nullptr;
	recording_ = false;
	depth_ = 0;
}

void ComputeContext::create(CudaContext& cuda, const std::string& cache_dir) {
	destroy();
	if (!cuda.valid()) throw MediaError("ComputeContext: CUDA context is not valid");
	cuda_ = &cuda;
	cache_dir_ = cache_dir;
	if (!cache_dir_.empty()) {
		std::wstring w = utf8_to_wide(path_backslash(cache_dir_));
		if (!CreateDirectoryW(w.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) {
			// No writable cache directory: compiling in memory per kernel is a
			// working engine (just a slower first frame), so this is not fatal.
			cache_dir_.clear();
		}
	}
	mutable_status().nvrtc = cuda_api().compiler_ok();
}

bool ComputeContext::has_pipeline(const std::string& key) const {
	for (const GpuKernel* k : kernels_)
		if (k && k->key == key) return true;
	return false;
}

GpuKernel* ComputeContext::pipeline(const std::string& key, const std::string& source,
                                    const std::string& entry, ShaderModel min_model, u32 block_x,
                                    u32 block_y, u32 block_z) {
	if (!cuda_) throw MediaError("ComputeContext::pipeline: no CUDA context");
	for (GpuKernel* k : kernels_)
		if (k && k->key == key) return k;
	(void)min_model;   // CUDA picks the arch from the device, not from a model level

	CudaApi& api = cuda_api();
	if (!api.compiler_ok())
		throw MediaError("shader '" + key + "' cannot compile: NVRTC is unavailable");

	BlockShapeParser shape(source);
	u32 bx = 256, by = 1, bz = 1;
	if (!shape.parse(&bx, &by, &bz) && block_x == 0)
		throw MediaError("shader '" + key +
		                 "': no block shape in the source (add a `numthreads(x,y,z)` "
		                 "comment) and none was supplied");
	if (block_x) {
		bx = block_x;
		by = block_y ? block_y : 1;
		bz = block_z ? block_z : 1;
	}

	// ── disk cache ─────────────────────────────────────────────────────────
	// Compiling ~25 kernels through NVRTC costs tens of seconds, so the cubin is
	// kept. The key includes a hash of the source, so editing a kernel misses the
	// cache instead of silently reusing the old binary.
	// The cache key must change with EVERY input that affects the compiled
	// cubin, not just the source: a different NVRTC version, a different
	// include dir (cuda_fp16.h changed between CUDA versions), or the
	// PHI_CUDA_FAST_MATH switch all produce a different binary. Hashing only
	// (source, entry, arch) meant fast-math could be silently reused for a
	// non-fast-math run (or the reverse) — a precision change the user cannot
	// see.
	std::string compile_inputs = entry + "|" + cuda_->arch() + "|" +
	                             api.nvrtc_version() + "|" +
	                             (getenv("PHI_CUDA_FAST_MATH") ? "fastmath" : "nofastmath") + "|" +
	                             api.include_dir();
	std::string tag = key + "-" + hex16(fnv1a(source, compile_inputs));
	std::string bin_path = cache_dir_.empty() ? "" : cache_dir_ + "\\" + tag + ".cubin";
	std::vector<u8> binary;
	bool from_cache = false;
	if (!bin_path.empty() && read_file(bin_path, &binary) && binary.size() > 64) {
		from_cache = true;
	}

	if (!from_cache) {
		std::vector<std::string> opt = {"-arch=" + cuda_->arch(), "--std=c++17",
		                                "-default-device"};
		for (const std::string& inc : api.include_opts()) opt.push_back(inc);
		if (getenv("PHI_CUDA_FAST_MATH")) opt.push_back("--use_fast_math");
		std::vector<const char*> opts;
		for (std::string& s : opt) opts.push_back(s.c_str());

		nvrtcProgram prog = nullptr;
		int rc = api.NvrtcCreateProgram(&prog, source.c_str(), (key + ".cu").c_str(), 0, nullptr,
		                                nullptr);
		if (rc != 0) throw MediaError("nvrtcCreateProgram failed for '" + key + "'");
		rc = api.NvrtcCompileProgram(prog, (int)opts.size(), opts.data());
		std::string log;
		size_t lsz = 0;
		if (api.NvrtcGetProgramLogSize && api.NvrtcGetProgramLogSize(prog, &lsz) == 0 && lsz > 1) {
			log.resize(lsz);
			api.NvrtcGetProgramLog(prog, &log[0]);
			while (!log.empty() && (log.back() == '\0' || log.back() == '\n')) log.pop_back();
		}
		if (rc != 0) {
			mutable_status().last_error = log;
			if (api.NvrtcDestroyProgram) api.NvrtcDestroyProgram(&prog);
			throw MediaError("shader '" + key + "' failed to compile:\n" + log);
		}
		// CUBIN when the toolkit can emit it (the driver then loads it directly);
		// PTX otherwise, which the driver JITs on first use.
		size_t bsz = 0;
		bool got = false;
		if (api.NvrtcGetCUBINSize && api.NvrtcGetCUBIN && api.NvrtcGetCUBINSize(prog, &bsz) == 0 &&
		    bsz > 0) {
			binary.resize(bsz);
			got = api.NvrtcGetCUBIN(prog, (char*)binary.data()) == 0;
		}
		if (!got) {
			size_t psz = 0;
			if (api.NvrtcGetPTXSize(prog, &psz) != 0) {
				if (api.NvrtcDestroyProgram) api.NvrtcDestroyProgram(&prog);
				throw MediaError("nvrtc: no cubin and no PTX for '" + key + "'");
			}
			binary.resize(psz);
			if (api.NvrtcGetPTX(prog, (char*)binary.data()) != 0) {
				if (api.NvrtcDestroyProgram) api.NvrtcDestroyProgram(&prog);
				throw MediaError("nvrtcGetPTX failed for '" + key + "'");
			}
		}
		if (api.NvrtcDestroyProgram) api.NvrtcDestroyProgram(&prog);
		if (!bin_path.empty() && binary.size() > 64) write_file(bin_path, binary.data(),
		                                                        binary.size());
		mutable_status().compiles++;
	} else {
		mutable_status().cache_hits++;
	}

	// cuModuleLoadData sniffs cubin vs PTX.
	auto* k = new GpuKernel();
	k->key = key;
	k->entry = entry;
	k->block_x = bx;
	k->block_y = by;
	k->block_z = bz;
	k->dynamic_smem = shape.smem();

	CUresult r = api.ModuleLoadData(&k->module, binary.data());
	if (r != CUDA_SUCCESS) {
		delete k;
		throw MediaError("module load failed for '" + key + "': " + api.error_text(r));
	}
	r = api.ModuleGetFunction(&k->fn, k->module, entry.c_str());
	if (r != CUDA_SUCCESS || !k->fn) {
		std::string detail = api.error_text(r);
		api.ModuleUnload(k->module);
		delete k;
		throw MediaError("kernel '" + entry + "' not found in '" + key + "' (extern \"C\" missing?): " +
		                 detail);
	}
	// Let the kernel opt into shared memory past the 48 KB default. Harmless for
	// kernels that do not ask; required for the attention tiles that do.
	if (api.FuncSetAttribute && cuda_->shared_mem_per_block_optin() > 48 * 1024) {
		api.FuncSetAttribute(k->fn, kFuncAttrMaxDynamicSharedBytes,
		                     cuda_->shared_mem_per_block_optin());
	}
	kernels_.push_back(k);
	return k;
}

void ComputeContext::begin() {
	recording_ = true;
	// Clamped on purpose. A bracket is nothing but this flag: a second begin()
	// without an intervening submit() adds no state and - because submit() used to
	// `--depth_` and return while it was > 0 - used to make every *later* submit()
	// a silent no-op. That unbalances the counter for the rest of the process, and
	// from then on submit() no longer waited for the GPU even though the arenas,
	// the staging ring and the tests all assume it does (see submit()).
	depth_ = 1;
}

void ComputeContext::submit_if_recording() {
	if (recording_) submit();
}

void ComputeContext::dispatch(GpuKernel* k, const KernelParams& params, u32 gx, u32 gy, u32 gz) {
	if (!k || !k->fn) throw MediaError("dispatch: null kernel");
	if (!cuda_) throw MediaError("dispatch: no CUDA context");

	// Marshal the root-signature shape into the kernel ABI. Offsets are folded
	// into the pointers so the kernel reads `a.s[i] + a.v[j]`, exactly like the
	// HLSL read `srv[i]` at the byte offset in `values[j]`.
	KernelArgs a;
	memcpy(a.v, params.values, sizeof a.v);
	for (int i = 0; i < 8; i++) {
		if (!params.srv[i]) {
			a.s[i] = nullptr;
			continue;
		}
		a.s[i] = (const char*)(params.srv[i]->base + params.srv_offset[i]);
	}
	for (int i = 0; i < 4; i++) {
		if (!params.uav[i]) {
			a.u[i] = nullptr;
			continue;
		}
		a.u[i] = (char*)(params.uav[i]->base + params.uav_offset[i]);
	}

	void* args[] = {&a};
	CUresult r = cuda_api().LaunchKernel(k->fn, gx, gy, gz, k->block_x, k->block_y, k->block_z,
	                                     k->dynamic_smem, cuda_->stream(), args, nullptr);
	if (r != CUDA_SUCCESS) {
		// 700/719 mean the context is gone; record it so callers stop rather than
		// queueing work onto a dead stream.
		if (r == 700 || r == 719) cuda_->note_device_lost();
		throw MediaError("launch of '" + k->key + "' failed (" + std::to_string(gx) + "," +
		                 std::to_string(gy) + "," + std::to_string(gz) + "): " +
		                 cuda_api().error_text(r));
	}
}

void ComputeContext::submit() {
	// The whole engine is written against the "the queue is idle when submit()
	// returns" contract: `GpuCtx::end_layer()` rewinds the staging ring right
	// after it, the DiT resets its weight arena right after it, `media_bench`
	// reads results right after it. So this must *always* wait - never only when a
	// bracket happens to be open at depth 1.
	//
	// It used to be `if (!recording_) return; if (--depth_ > 0) return; ...`, which
	// meant (a) a submit() with no bracket open did not wait at all, although
	// callers use it as a fence, and (b) any unbalanced begin() (the debug dump
	// helpers deliberately close a bracket, take a readback and leave one open)
	// pushed depth_ above 1 permanently, after which no submit() ever synced
	// again. The GPU then ran arbitrarily far behind the CPU, which reuses both the
	// pinned staging ring and the reset arenas - the result is a timing-dependent
	// corruption: bit-identical runs diverge, and a run with stage dumps can turn
	// into NaN where the same run without them is clean.
	recording_ = false;
	depth_ = 0;
	cuda_->sync();
}

void ComputeContext::run(GpuKernel* k, const KernelParams& params, u32 gx, u32 gy, u32 gz) {
	begin();
	dispatch(k, params, gx, gy, gz);
	submit();
}

// ── buffers ────────────────────────────────────────────────────────────────

GpuBuffer* ComputeContext::create_device_buffer(u64 bytes, bool uav) {
	(void)uav;
	GpuBuffer* b = cuda_->alloc(bytes);
	owned_.push_back(b);
	return b;
}

GpuBuffer* ComputeContext::create_upload_buffer(u64 bytes) {
	void* mapped = nullptr;
	GpuBuffer* b = cuda_->alloc_host(bytes, &mapped);
	owned_.push_back(b);
	return b;
}

GpuBuffer* ComputeContext::create_readback_buffer(u64 bytes) {
	void* mapped = nullptr;
	GpuBuffer* b = cuda_->alloc_host(bytes, &mapped);
	owned_.push_back(b);
	return b;
}

void ComputeContext::release_buffer(GpuBuffer* res) {
	if (!res) return;
	for (size_t i = 0; i < owned_.size(); i++) {
		if (owned_[i] != res) continue;
		owned_.erase(owned_.begin() + (long)i);
		cuda_->free(res);
		return;
	}
}

void ComputeContext::upload(GpuBuffer* dst, const void* src, u64 bytes) {
	cuda_->copy_h2d(dst, 0, src, bytes);
}

void ComputeContext::upload(GpuBuffer* dst, u64 dst_off, const void* src, u64 bytes) {
	cuda_->copy_h2d(dst, dst_off, src, bytes);
}

void ComputeContext::download(GpuBuffer* src, void* dst, u64 bytes) {
	cuda_->copy_d2h(src, 0, dst, bytes);
}

void ComputeContext::download(GpuBuffer* src, u64 src_off, void* dst, u64 bytes) {
	cuda_->copy_d2h(src, src_off, dst, bytes);
}

// ── the two int8 kernels ────────────────────────────────────────────────
// The values[] layout and the binding order are fixed by the kernels' `Args`
// struct; the host side only has to fill it.

void dispatch_int8_gemm(ComputeContext& ctx, const Int8GemmArgs& a) {
	if (a.M <= 0 || a.N <= 0 || a.K <= 0) throw MediaError("int8_gemm: empty problem");

	// Tensor cores when the shape allows, dp4a otherwise. The gap is 6x on this
	// part (121 vs 20 TOP/s, see kernels/int8_gemm.cpp), which is the difference
	// between the DiT's GEMMs costing 72 s per sampling step at 540P/10s and
	// costing 12 s - so the fast path is the default and the constraints are
	// checked rather than assumed.
	//
	// `mma.m16n8k32` contracts k in 32-wide steps and the staging reads 16 bytes
	// at a time, so K must be a multiple of 32 and the operand bases must be
	// 16-byte aligned. The engine's own K values are 5376 / 7168 / 14336 / 21504 /
	// 3840 / 10240 (all multiples of 64) and its arena allocations are
	// 256-byte aligned, so this is always the path the models take; the check
	// exists for the tests and for any future shape.
	// PHI_INT8_DP4A=1 forces the old kernel: an A/B switch for the port, and the
	// way to tell a tensor-core regression from anything else.
	static const bool force_dp4a = [] {
		const char* e = getenv("PHI_INT8_DP4A");
		return e && *e && *e != '0';
	}();
	const bool tc_ok = !force_dp4a && (a.K % 32 == 0) && (a.a_offset % 16 == 0) &&
	                   (a.b_offset % 16 == 0);
	if (tc_ok) {
		GpuKernel* k =
		    ctx.pipeline("int8_gemm_tc", int8_gemm_tc_hlsl(), "int8_gemm_tc", ShaderModel::SM6_4);

		// ── one launch, or one per m-strip (PHI_TC_STRIP) ──────────────────
		//
		// The block tile is 256x128 because that is the largest output tile the
		// register file holds (`BM*BN` fp32 accumulators is half of it), and the
		// tile is what fixes the DRAM traffic the GEMM must move: every block
		// re-reads its own BM-row strip of A and its own BN-wide strip of B, so
		// over the whole matrix A is fetched ceil(N/BN) times and B ceil(M/BM)
		// times. Bytes per FLOP is then (1/BN + 1/BM)/2, which at 256x128 on this
		// part is 54 TOPS - exactly what the kernel measures (49 in the bench,
		// 45 in the DiT), so these GEMMs are bandwidth rather than arithmetic.
		//
		// The obvious next step is to remove A's share of that: inside one
		// m-strip every block reads the *same* 256 rows, so the strip only has to
		// arrive from DRAM once if it is still in L2 when the strip's blocks run.
		// Issuing one launch per strip is the way to make that true, and it is
		// implemented and measurably *worthless* here: 47.6 TOPS against 49.2 for
		// the single launch on 2048x3840x3840, and worse at small M where the
		// extra launches dominate. The reason is that a single launch already
		// walks m-strips in order (`bid / gy` increases with bid), so the
		// scheduler's ~30 in-flight blocks are already one strip. L2 does the
		// reuse; the block tile is the thing that would have to grow, and the
		// register file is what stops it. The switch is kept as the A/B that
		// establishes this.
		const u32 kStrip = 256u;
		static const bool strip_on = [] {
			const char* e = getenv("PHI_TC_STRIP");
			return e && *e && *e != '0';
		}();
		const i64 strip = (strip_on && a.M > (i64)kStrip) ? (i64)kStrip : (i64)a.M;
		for (i64 m0 = 0; m0 < a.M; m0 += strip) {
			const i64 rows = std::min<i64>(strip, a.M - m0);
			KernelParams p{};
			p.values[0] = (u32)rows;
			p.values[1] = (u32)a.N;
			p.values[2] = (u32)a.K;
			p.values[3] = a.bias ? 1u : 0u;   // flags: bit0 = add the bias
			p.values[4] = (u32)(a.a_offset + (u64)m0 * (u64)a.K);
			p.values[5] = (u32)a.b_offset;
			p.values[6] = (u32)(a.c_offset + (u64)m0 * (u64)a.N * 4u);
			p.values[7] = (u32)(a.sa_offset + (u64)m0 * 4u);
			p.values[8] = (u32)a.sb_offset;
			p.values[9] = (u32)a.bias_offset;
			p.srv[0] = a.a;
			p.srv[1] = a.b;
			p.srv[2] = a.sa;
			p.srv[3] = a.sb;
			p.srv[4] = a.bias ? a.bias : a.a;
			p.uav[0] = a.c;

			// One block covers 256 rows x 128 cols (see the tile-size note in
			// kernels/int8_gemm.cpp - the 128x64 first version was L2-bound); the
			// kernel walks the same (row-slow, col-fast) order the dp4a one does.
			const u32 gx = (u32)((rows + (i64)kStrip - 1) / (i64)kStrip);
			const u32 gy = (u32)((a.N + 127) / 128);
			ctx.dispatch(k, p, gx, gy, 1);
		}
		return;
	}

	GpuKernel* k =
	    ctx.pipeline("int8_gemm_dp4a", int8_gemm_hlsl(), "int8_gemm_dp4a", ShaderModel::SM6_4);

	KernelParams p{};
	p.values[0] = (u32)a.M;
	p.values[1] = (u32)a.N;
	p.values[2] = (u32)a.K;
	p.values[3] = a.bias ? 1u : 0u;   // flags: bit0 = add the bias
	p.values[4] = (u32)a.a_offset;
	p.values[5] = (u32)a.b_offset;
	p.values[6] = (u32)a.c_offset;
	p.values[7] = (u32)a.sa_offset;
	p.values[8] = (u32)a.sb_offset;
	p.values[9] = (u32)a.bias_offset;
	p.srv[0] = a.a;
	p.srv[1] = a.b;
	p.srv[2] = a.sa;
	p.srv[3] = a.sb;
	p.srv[4] = a.bias ? a.bias : a.a;
	p.uav[0] = a.c;

	const u32 gx = (u32)((a.M + 127) / 128);
	const u32 gy = (u32)((a.N + 63) / 64);
	ctx.dispatch(k, p, gx, gy, 1);
}

void dispatch_quant_convrot(ComputeContext& ctx, const QuantConvrotArgs& a) {
	if (a.rows <= 0) return;
	if (a.K % 256 != 0) {
		throw MediaError("quant_convrot: K=" + std::to_string(a.K) +
		                 " is not a multiple of 256 (convrot group size)");
	}
	GpuKernel* k = ctx.pipeline("quant_convrot_activation", quant_convrot_hlsl(),
	                            "quant_convrot_activation", ShaderModel::SM6_0);

	KernelParams p{};
	p.values[0] = (u32)a.rows;
	p.values[1] = (u32)a.K;
	p.values[2] = (u32)a.x_offset;
	p.values[3] = (u32)a.q_offset;
	p.values[4] = (u32)a.s_offset;
	float qmax = a.qmax;
	memcpy(&p.values[5], &qmax, 4);
	p.srv[0] = a.x;
	p.uav[0] = a.q;
	p.uav[1] = a.s;
	ctx.dispatch(k, p, (u32)a.rows, 1, 1);
}

}  // namespace phi::media
