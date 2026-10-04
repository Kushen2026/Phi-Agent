#include "bench/bench.hpp"

#include <algorithm>
#include <cstdio>

#include "runtime/vram_budget.hpp"

namespace phi::media {

void StageTimer::add(const std::string& name, double ms) {
	for (auto& s : stages_) {
		if (s.name == name) {
			s.ms += ms;
			s.calls++;
			return;
		}
	}
	stages_.push_back(BenchStage{name, ms, 1});
}

void StageTimer::reset() { stages_.clear(); }

double StageTimer::total_ms() const {
	double t = 0;
	for (const auto& s : stages_) t += s.ms;
	return t;
}

std::string StageTimer::report(const std::string& title) const {
	std::string out = title + "\n";
	// widest stage name, so the columns line up in a console
	size_t w = 8;
	for (const auto& s : stages_) w = std::max(w, s.name.size());
	char line[256];
	for (const auto& s : stages_) {
		double pct = total_ms() > 0 ? 100.0 * s.ms / total_ms() : 0.0;
		snprintf(line, sizeof line, "  %-*s  %10.1f ms  %6.1f%%  x%d\n", (int)w, s.name.c_str(),
		         s.ms, pct, s.calls);
		out += line;
	}
	snprintf(line, sizeof line, "  %-*s  %10.1f ms\n", (int)w, "TOTAL", total_ms());
	out += line;
	return out;
}

std::string VramTracker::report() const {
	char buf[160];
	snprintf(buf, sizeof buf, "VRAM peak %s / limit %s (current %s)", format_bytes(peak_).c_str(),
	         limit_ ? format_bytes(limit_).c_str() : "unset", format_bytes(current_).c_str());
	return buf;
}

void VramTracker::observe(const VramBudget& b) {
	if (b.limit()) limit_ = b.limit();
	current_ = b.local();
	if (current_ > peak_) peak_ = current_;
}

}  // namespace phi::media
