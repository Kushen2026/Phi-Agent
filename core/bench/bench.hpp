// Benchmark plumbing: per-stage timing and VRAM high-water tracking.
//
// phi.exe --media-bench prints exactly this table; it is the only acceptance
// instrument for the media tools (no ComfyUI cross-check).
#pragma once

#include <string>
#include <vector>

#include "util/media_common.hpp"

namespace phi::media {

class VramBudget;

struct BenchStage {
	std::string name;
	double ms = 0.0;
	int calls = 0;
};

class StageTimer {
public:
	// RAII scope: Scope s(timer, "sampling.step"); adds its wall time on exit.
	struct Scope {
		StageTimer* t;
		std::string name;
		double start;
		Scope(StageTimer* timer, std::string n)
			: t(timer), name(std::move(n)), start(now_ms()) {}
		~Scope() { t->add(name, now_ms() - start); }
		Scope(const Scope&) = delete;
		Scope& operator=(const Scope&) = delete;
	};
	Scope scope(const std::string& name) { return Scope(this, name); }

	void add(const std::string& name, double ms);
	void reset();
	const std::vector<BenchStage>& stages() const { return stages_; }
	double total_ms() const;

	std::string report(const std::string& title) const;

private:
	std::vector<BenchStage> stages_;
};

// Kernel-launch wall time + VRAM watermark. The VRAM number is sampled by the
// allocator (see CudaContext / vram_budget) and pushed here; the peak is what
// matters for the "<= 5.9 GB" hard budget.
class VramTracker {
public:
	void sample(u64 bytes) {
		current_ = bytes;
		if (bytes > peak_) peak_ = bytes;
	}
	void note_limit(u64 bytes) { limit_ = bytes; }
	// Mirror the unified VRAM accountant (M3 · W1) into this tracker: the budget
	// limit and the local-VRAM current/peak. The acceptance number stays the one
	// in this tracker, so both sources agree by construction.
	void observe(const VramBudget& b);
	u64 current() const { return current_; }
	u64 peak() const { return peak_; }
	u64 limit() const { return limit_; }
	void reset() {
		current_ = peak_ = 0;
	}
	std::string report() const;

private:
	u64 current_ = 0;
	u64 peak_ = 0;
	u64 limit_ = 0;
};

}  // namespace phi::media
