// Kernel autotuning + persistence (M3 · W1).
//
// The int8 GEMM's tile shape is baked into its HLSL as `static const uint BM/BN/
// BK/TM/TN/NT`. The fastest shape is machine- and shape-dependent, so instead of
// guessing, `autotune_int8_gemm` compiles a handful of variants, measures them on
// the real device and writes the winner to `data/autotune.json`. The next start
// loads that file and dispatches with the stored config, so the sweep costs one
// launch per machine, not one per run.
//
// The dispatcher has a `_v2` name on purpose: `compute.hpp`'s `dispatch_int8_gemm`
// is what the M1 image path already calls and it is frozen, so the autotuned path
// is a new entry point that a caller opts into.
//
// Correctness note: every variant sums the same integers in the same (ascending
// k) order, so all of them must produce *bit-identical* C. The sweep checks that
// against a CPU reference (`gemm_int8`) and against the baseline variant, which
// turns "is this tile allowed to change the result?" from an assumption into a
// measured fact.
#pragma once

#include <string>
#include <vector>

#include "runtime/compute.hpp"
#include "util/media_common.hpp"

namespace phi::media {

// One launch configuration for the int8 dp4a GEMM.
struct KernelConfig {
	std::string kernel = "int8_gemm_dp4a";
	int bm = 128, bn = 64, bk = 64;   // C/A/B tiles (elements)
	int tm = 4, tn = 8;               // per-thread fragment
	int nt = 256;                     // thread-group size
	double measured_ms = 0.0;
	double tops = 0.0;

	// Stable cache key (also the PSO cache tag, appended to the kernel name).
	std::string key() const;
	// Divisibility / shared-memory / thread-group checks. Returns false and fills
	// `why` when the HLSL would not compile or would be out of bounds.
	bool valid(std::string* why = nullptr) const;
	bool operator==(const KernelConfig& o) const {
		return bm == o.bm && bn == o.bn && bk == o.bk && tm == o.tm && tn == o.tn && nt == o.nt &&
		       kernel == o.kernel;
	}
};

// Persistent store, one JSON object per (kernel) in `data/autotune.json`.
class AutotuneStore {
public:
	static std::string default_path(const std::string& base_dir);
	bool load(const std::string& path);
	bool save(const std::string& path) const;

	// Best stored config for `kernel`, else the built-in default.
	KernelConfig get(const std::string& kernel) const;
	void put(const KernelConfig& c);   // replaces the entry for c.kernel
	size_t size() const { return items_.size(); }
	const std::vector<KernelConfig>& items() const { return items_; }

	static KernelConfig default_for(const std::string& kernel);
	// The candidate sweep for the int8 GEMM (filtered by `valid()`).
	static std::vector<KernelConfig> int8_gemm_candidates();

private:
	std::vector<KernelConfig> items_;
};

// Compile the int8 GEMM HLSL with `c`'s constants substituted. The entry point
// name is unchanged, so `pipeline(c.key(), ...)` caches each variant separately.
std::string int8_gemm_hlsl_variant(const KernelConfig& c);

// Autotuned dispatch (records into the open command list; no submit).
void dispatch_int8_gemm_v2(ComputeContext& ctx, const Int8GemmArgs& a, const KernelConfig& c);

struct AutotuneResult {
	bool ok = false;
	std::string error;
	KernelConfig best;
	std::vector<KernelConfig> tried;   // with measured_ms / tops filled
	int exact_matches = 0;             // variants bit-identical to the CPU ref
};

// Sweep the int8 GEMM candidates for an M x N x K problem, `iters` launches each.
// Verifies every variant is bit-identical to a CPU `gemm_int8` reference.
AutotuneResult autotune_int8_gemm(ComputeContext& ctx, i64 M, i64 N, i64 K, int iters = 3);

}  // namespace phi::media
