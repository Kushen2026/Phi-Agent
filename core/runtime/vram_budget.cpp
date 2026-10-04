#include "runtime/vram_budget.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace phi::media {

// PHI_VRAM_TRACE=1 prints every charge/release with the running total.
static bool vram_trace() {
	static int v = -1;
	if (v < 0) {
		const char* e = getenv("PHI_VRAM_TRACE");
		v = e ? atoi(e) : 0;
	}
	return v != 0;
}

u64 VramBudget::compute_hard_cap(u64 dedicated_vram) {
	// 0 = the card has not reported itself yet. Answer for the reference card so
	// the caller gets the number the engine was calibrated against rather than a
	// fraction of it.
	if (dedicated_vram == 0) dedicated_vram = kHardCap;

	// Reserved for the OS, the compositor and the driver. This has to scale with
	// the card: a flat 200 MB is right for a 6 GB laptop part and would waste
	// 22 GB on a 24 GB one, while a flat percentage would eat the small card (a
	// 10 % haircut on 6 GB is 600 MB, most of a resident DiT block). One
	// sixteenth of the card, bounded at both ends, does neither: 384 MB at 6 GB
	// (still below that card's own driver budget, so the calibrated machine plans
	// exactly as it did), 1.5 GB at 24 GB, 2 GB at 48 GB.
	u64 reserve = dedicated_vram / 16;
	if (reserve < kMinReserve) reserve = kMinReserve;
	if (reserve > kMaxReserve) reserve = kMaxReserve;
	u64 cap = (dedicated_vram > reserve) ? (dedicated_vram - reserve) : 0;

	// Never *below* the reference card's authorised ceiling: on that card the
	// driver's budget binds first (it is ~0.7 GB below this), so this keeps the
	// numbers the engine was calibrated against - and the tests that pin them -
	// exactly where they were, while a bigger card is free to be used.
	if (dedicated_vram >= kHardCap && cap < kHardCap) cap = kHardCap;

	// PHI_VRAM_BUDGET_MB overrides everything (highest priority). It may raise the
	// ceiling as well as lower it - that is what makes "try it with 7 GB" a
	// one-line experiment on a 12 GB card - but it can never claim more than the
	// card physically has, and the reserve is never given up.
	if (const char* e = getenv("PHI_VRAM_BUDGET_MB")) {
		long long mb = atoll(e);
		if (mb >= 256) {
			u64 want = (u64)mb * 1024 * 1024;
			const u64 phys = dedicated_vram > kMinReserve ? dedicated_vram - kMinReserve : 0;
			if (want > phys) want = phys;
			cap = want;
		}
	}

	if (cap > dedicated_vram) cap = dedicated_vram;
	return cap;
}

void VramBudget::note_dedicated_vram(u64 dedicated_vram) {
	if (dedicated_vram == 0) return;
	std::lock_guard<std::mutex> lk(m_);
	dedicated_vram_ = dedicated_vram;
	hard_cap_ = compute_hard_cap(dedicated_vram_);
	if (driver_noted_) {
		// The driver's own allowance is the binding wall when it is smaller than
		// the hardware-derived ceiling (see note_driver_vram).
		const u64 drv = driver_budget_ > driver_baseline_ ? driver_budget_ - driver_baseline_ : 0;
		if (drv != 0 && drv < hard_cap_) hard_cap_ = drv;
	}
}

void VramBudget::note_driver_vram(u64 budget, u64 usage, u64 reservation) {
	if (budget == 0) return;
	std::lock_guard<std::mutex> lk(m_);
	// Only the first note sets the baseline: it is the process's non-engine
	// footprint (the CUDA context's WDDM charge) and it must not be re-read once
	// the engine's own allocations are what the driver is counting.
	if (!driver_noted_) {
		driver_budget_ = budget;
		driver_baseline_ = usage;
		driver_noted_ = true;
	} else if (budget < driver_budget_) {
		// The driver can shrink the budget live (another client claiming local
		// memory). Honour the smaller figure; never grow it back here - the live
		// probe is what re-reads the current one on every charge.
		driver_budget_ = budget;
	}
	driver_reservation_ = reservation;
	hard_cap_ = compute_hard_cap(dedicated_vram_);
	const u64 drv = driver_budget_ > driver_baseline_ ? driver_budget_ - driver_baseline_ : 0;
	if (drv != 0 && drv < hard_cap_) hard_cap_ = drv;
}

u64 VramBudget::driver_budget() const {
	std::lock_guard<std::mutex> lk(m_);
	return driver_budget_ > driver_baseline_ ? driver_budget_ - driver_baseline_ : 0;
}

// The bytes every plan must leave unspent, so that the allocation the loop asks
// for next is always affordable. Two terms, and both are real:
//
//   * one arena chunk - the accountant's own granularity (`GpuArena::alloc`
//     charges a whole chunk the moment one tensor crosses a boundary), so the
//     chunk a streamed block is about to ask for is never the one that crosses
//     the ceiling;
//   * a fraction of the budget for everything no estimate prices - the chunk
//     boundaries a block's tensors do not tile, the CUDA context's own growth,
//     and the drift of a card the desktop compositor and the app's WebView2 UI
//     are also drawing on.
//
// It is deliberately subtracted inside `budget_limit_for`, i.e. from the number
// *both* residency planners and the accountant's own enforcement use: one term,
// one place, so the plan and the refusal can never disagree about how much room
// the residual took.
u64 VramBudget::residual_reserve(u64 driver_budget) {
	if (driver_budget == 0) return 0;
	// Same shape as `arena_chunk_bytes_for`, so the two cannot drift apart.
	u64 chunk = driver_budget / 80;
	if (chunk < (64ull << 20)) chunk = 64ull << 20;
	if (chunk > (256ull << 20)) chunk = 256ull << 20;
	chunk = (chunk + (16ull << 20) - 1) & ~((16ull << 20) - 1);
	u64 fraction = driver_budget / 64;
	if (fraction < (32ull << 20)) fraction = 32ull << 20;
	if (fraction > (384ull << 20)) fraction = 384ull << 20;
	return chunk + fraction;
}

u64 VramBudget::hard_cap_bytes() const {
	std::lock_guard<std::mutex> lk(m_);
	return hard_cap_;
}

u64 VramBudget::hard_cap() { return vram_budget().hard_cap_bytes(); }

u64 VramBudget::budget_limit_for(u64 driver_budget) const {
	if (driver_budget == 0) return hard_cap_;   // nothing reported yet: only the ceiling
	// Multiply before dividing: `(budget / 100) * percent` truncates a byte or two
	// off the budget, so at 100% the limit came out *below* the budget it was
	// derived from (1073741824 -> 1073741800), which is exactly the kind of
	// off-by-a-hair limit a residency plan is sensitive to.
	u64 limit = driver_budget * (u64)budget_percent() / 100;
	// Integer division on a tiny budget would round the allowance away entirely.
	if (limit == 0) limit = driver_budget;
	if (limit > hard_cap_) limit = hard_cap_;
	// The residual: what a planner must leave unspent so the chunk the sampling
	// loop asks for next is still affordable (see residual_reserve). Subtracting
	// it here means both planners and the accountant's own enforcement leave the
	// same room without either of them having to remember.
	const u64 residual = residual_reserve(limit);
	return limit > residual ? limit - residual : limit / 2;
}

VramBudget::Environment VramBudget::refresh() {
	// Ask the driver outside the lock: the probe talks to CUDA and must never run
	// under it (see charge()).
	std::function<LiveVram()> probe;
	{
		std::lock_guard<std::mutex> lk(m_);
		probe = live_probe_;
	}
	LiveVram live;
	if (probe) live = probe();

	Environment env;
	{
		std::lock_guard<std::mutex> lk(m_);
		if (live.total) {
			// The device's own figure is authoritative and it is what the ceiling
			// should be derived from: `note_dedicated_vram` may have run before the
			// context existed, or on a machine whose adapter has since changed.
			dedicated_vram_ = live.total;
			env.live = true;
		}
		hard_cap_ = compute_hard_cap(dedicated_vram_);
		if (driver_noted_) {
			const u64 drv =
			    driver_budget_ > driver_baseline_ ? driver_budget_ - driver_baseline_ : 0;
			if (drv != 0 && drv < hard_cap_) hard_cap_ = drv;
		}
		if (live.budget) os_budget_ = live.budget;
		limit_ = budget_limit_for(os_budget_);

		env.device_total = live.total ? live.total : dedicated_vram_;
		env.device_free = live.free;
		env.driver_budget = os_budget_;
		env.hard_cap = hard_cap_;
		env.limit = limit_;
		env.percent = budget_percent();
		env.driver_local_budget = driver_budget_;
		env.driver_local_usage = driver_baseline_;
		env.residual = residual_reserve(limit_);
	}
	return env;
}

VramBudget::Environment VramBudget::environment() const {
	std::lock_guard<std::mutex> lk(m_);
	Environment env;
	env.device_total = dedicated_vram_;
	env.driver_budget = os_budget_;
	env.hard_cap = hard_cap_;
	env.limit = limit_;
	env.percent = budget_percent();
	env.residual = VramBudget::residual_reserve(limit_);
	env.driver_local_budget = driver_budget_;
	env.driver_local_usage = driver_baseline_;
	// `environment()` deliberately does not call the probe: it is the cheap
	// accessor the run's report uses, and the probe is a driver call.
	env.live = dedicated_vram_ != 0;
	return env;
}

int VramBudget::budget_percent() {
	static int pct = [] {
		int p = kDefaultBudgetPercent;
		if (const char* e = getenv("PHI_VRAM_MARGIN_PERCENT")) {
			int v = atoi(e);
			if (v >= 50 && v <= 100) p = v;
		}
		return p;
	}();
	return pct;
}

u64 VramBudget::budget_limit(u64 driver_budget) {
	VramBudget& b = vram_budget();
	std::lock_guard<std::mutex> lk(b.m_);
	return b.budget_limit_for(driver_budget);
}

void VramBudget::set_limit(u64 bytes) {
	std::lock_guard<std::mutex> lk(m_);
	limit_ = bytes;
}

void VramBudget::note_os_budget(u64 driver_budget) {
	std::lock_guard<std::mutex> lk(m_);
	os_budget_ = driver_budget;	// Two numbers, and the smaller one is the enforced limit:
	//  * the ceiling derived from the card's own VRAM (compute_hard_cap) - the
	//    card's memory is the wall, and going past it fails for real;
	//  * the driver's Budget for this process *less the safety margin* - the amount
	//    it can hand out without demoting pages to system memory. Exceeding it does
	//    not fail, it *thrashes*: a full-tensor reduction that normally takes 20 ms
	//    took 13 s and tripped the 2 s TDR watchdog when the decoder was allowed to
	//    run 5.7 GB into a 4.5 GB budget inside the app.
	//
	// The margin is what makes this correct *inside the app*. A 1080p generation
	// wants ~5.06 GB of a 5.10 GB budget (the DiT keeps as many layers resident as
	// the accountant allows) - 99% of it, which a bare console process that owns
	// the GPU survives and the app does not: there the compositor and the WebView2
	// UI claim and release VRAM while the generation runs (every progress update
	// repaints). The first client to ask after we took the last byte wins, our
	// pages are demoted, and the watchdog takes the device down - which is exactly
	// what the user saw: "download: Map failed" at 1080p, then every later
	// allocation failing against a device that had already been taken down.
	// Two numbers, and the smaller one is the enforced limit: the ceiling this
	// card supports (recomputed here as well, because this call is the first thing
	// `refresh()` does once a device answers) and the driver's budget for this
	// process.
	hard_cap_ = compute_hard_cap(dedicated_vram_);
	if (driver_noted_) {
		const u64 drv = driver_budget_ > driver_baseline_ ? driver_budget_ - driver_baseline_ : 0;
		if (drv != 0 && drv < hard_cap_) hard_cap_ = drv;
	}
	limit_ = budget_limit_for(driver_budget);
}

u64 VramBudget::limit() const {
	std::lock_guard<std::mutex> lk(m_);
	return limit_;
}

u64 VramBudget::os_budget() const {
	std::lock_guard<std::mutex> lk(m_);
	return os_budget_;
}

u64 VramBudget::local() const {
	std::lock_guard<std::mutex> lk(m_);
	return local_;
}

u64 VramBudget::peak() const {
	std::lock_guard<std::mutex> lk(m_);
	return peak_;
}

u64 VramBudget::headroom() const {
	std::lock_guard<std::mutex> lk(m_);
	return limit_ ? (local_ < limit_ ? limit_ - local_ : 0) : 0;
}

bool VramBudget::would_exceed(u64 bytes) const {
	std::lock_guard<std::mutex> lk(m_);
	return limit_ != 0 && local_ + bytes > limit_;
}

void VramBudget::bump_peak_locked(u64 bytes) {
	if (bytes > peak_) peak_ = bytes;
}

void VramBudget::set_live_probe(std::function<LiveVram()> probe) {
	std::lock_guard<std::mutex> lk(m_);
	live_probe_ = std::move(probe);
}

u64 VramBudget::live_usage() const {
	std::function<LiveVram()> probe;
	{
		std::lock_guard<std::mutex> lk(m_);
		probe = live_probe_;
	}
	// Called outside the lock: the probe talks to the driver, and holding the
	// accountant's mutex across a driver call is how you get a deadlock the day
	// someone charges from a callback.
	return probe ? probe().usage : 0;
}

u64 VramBudget::refused_live() const {
	std::lock_guard<std::mutex> lk(m_);
	return refused_live_;
}

u64 VramBudget::live_headroom() const {
	// The probe talks to the driver, so it runs outside the lock (see charge).
	std::function<LiveVram()> probe;
	{
		std::lock_guard<std::mutex> lk(m_);
		probe = live_probe_;
	}
	if (!probe) return ~0ull;
	const LiveVram live = probe();
	if (live.total == 0) return ~0ull;
	std::lock_guard<std::mutex> lk(m_);
	// Exactly the ceiling `charge()` will apply: min(percent x the driver's budget
	// for this process, the ceiling the card supports), less what we already hold.
	const u64 ceiling = budget_limit_for(live.budget);
	u64 room = ceiling > live.usage ? ceiling - live.usage : 0;
	if (live.free < room) room = live.free;
	return room;
}

void VramBudget::charge(u64 bytes, const char* tag) {
	if (bytes == 0) return;
	// Ask the driver *before* taking the lock. `local_` is this engine's ledger;
	// the driver also counts the upload ring, the staging heaps and every other
	// process on the box, and it is the driver that decides when a commit fails.
	// Checking here turns a lost device into a plain MediaError while there is
	// still a device to report it from.
	{
		std::function<LiveVram()> probe;
		u64 ceiling = 0;
		{
			std::lock_guard<std::mutex> lk(m_);
			probe = live_probe_;
			ceiling = limit_;
		}
		if (probe && ceiling != 0) {
			const LiveVram live = probe();   // talks to the driver: never under the lock
			// There are two independent walls and they must not be confused, because
			// one is denominated in *ledger* bytes and the other in *device* bytes:
			//
			//   * `limit_` caps what this engine may allocate. It is checked against
			//     the ledger (local_ + bytes) further down.
			//   * the device wall is what the card can actually hold before the
			//     driver starts demoting pages, and it must be compared against what
			//     the device reports as committed - NOT against the ledger, which
			//     under-reports by whatever the CUDA context, the loaded cubins and
			//     any released-but-not-yet-freed arena chunks hold.
			//
			// Mixing the two is how the check went wrong once before: the
			// ledger said 4.30 GB while the device was actually holding 5.49 GB, so
			// "free + ledger" under-estimated the capacity and the accountant refused
			// allocations while the card still had room. Keeping them separate also
			// preserves the original intent - noticing that another process took
			// VRAM while we were working - because that shows up as the device's
			// committed bytes rising.
			// `live.usage` excludes the driver context (CudaContext subtracts its
			// baseline), so both sides of this comparison are in the same units as
			// `limit_`: bytes this engine is responsible for.
			if (live.budget) ceiling = std::min(ceiling, budget_limit(live.budget));
				if (live.usage > ceiling || bytes > ceiling - live.usage) {
				std::lock_guard<std::mutex> lk(m_);
				refused_live_++;
				char buf[420];
				snprintf(buf, sizeof buf,
				         "VRAM exhausted: %s for %s would take the process to %s of the "
				         "%s ceiling (ledger %s of %s, driver budget %s, device free %s of %s)",
				         format_bytes(bytes).c_str(), tag ? tag : "allocation",
				         format_bytes(live.usage + bytes).c_str(), format_bytes(ceiling).c_str(),
				         format_bytes(local_).c_str(),
				         limit_ ? format_bytes(limit_).c_str() : "unset",
				         format_bytes(live.budget).c_str(), format_bytes(live.free).c_str(),
				         format_bytes(live.total).c_str());
				throw MediaError(buf);
			}
		}
	}
	std::function<void(u64)> announce;
	u64 newpeak = 0;
	bool trace = false;
	u64 traced_total = 0;
	{
		std::lock_guard<std::mutex> lk(m_);
		if (limit_ != 0 && local_ + bytes > limit_) {
			denied_++;
			char buf[224];
			snprintf(buf, sizeof buf, "VRAM budget exceeded: requesting %s for %s would take "
			                          "resident %s past the limit %s",
			         format_bytes(bytes).c_str(), tag ? tag : "allocation",
			         format_bytes(local_ + bytes).c_str(), format_bytes(limit_).c_str());
			throw MediaError(buf);
		}
		local_ += bytes;
		bump_peak_locked(local_);
		newpeak = peak_;
		announce = on_peak;   // copied under the lock; invoked outside it
		trace = vram_trace();
		traced_total = local_;
	}
	if (trace)
		fprintf(stderr, "[vram] +%s %-14s -> %s\n", format_bytes(bytes).c_str(),
		        tag ? tag : "?", format_bytes(traced_total).c_str());
	if (announce) announce(newpeak);
}

void VramBudget::release(u64 bytes) {
	if (bytes == 0) return;
	std::lock_guard<std::mutex> lk(m_);
	local_ = (local_ > bytes) ? local_ - bytes : 0;
	if (vram_trace())
		fprintf(stderr, "[vram] -%s %-14s -> %s\n", format_bytes(bytes).c_str(), "release",
		        format_bytes(local_).c_str());
}

u64 VramBudget::staging() const {
	std::lock_guard<std::mutex> lk(m_);
	return staging_;
}

void VramBudget::note_staging(i64 delta) {
	std::lock_guard<std::mutex> lk(m_);
	if (delta < 0) {
		u64 d = (u64)(-delta);
		staging_ = (staging_ > d) ? staging_ - d : 0;
	} else {
		staging_ += (u64)delta;
	}
}

void VramBudget::reset_staging() {
	std::lock_guard<std::mutex> lk(m_);
	staging_ = 0;
}

void VramBudget::reset() {
	std::lock_guard<std::mutex> lk(m_);
	local_ = peak_ = staging_ = denied_ = 0;
}

u64 VramBudget::denied() const {
	std::lock_guard<std::mutex> lk(m_);
	return denied_;
}

std::string VramBudget::report() const {
	std::lock_guard<std::mutex> lk(m_);
	char buf[256];
	snprintf(buf, sizeof buf,
	         "VRAM local %s / peak %s / limit %s (os %s) + staging %s, denied %llu"
	         " (live %llu)",
	         format_bytes(local_).c_str(), format_bytes(peak_).c_str(),
	         limit_ ? format_bytes(limit_).c_str() : "unset",
	         os_budget_ ? format_bytes(os_budget_).c_str() : "unset",
	         format_bytes(staging_).c_str(), (unsigned long long)denied_,
	         (unsigned long long)refused_live_);
	return buf;
}

VramBudget& vram_budget() {
	// Function-local static: the scheduler is initialised lazily, and the
	// accountant must outlive every arena (it is read during static teardown in
	// some test binaries).
	static VramBudget instance;
	return instance;
}

}  // namespace phi::media