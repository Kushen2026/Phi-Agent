// Unified VRAM accounting (M3 · W1).
//
// One accountant for the whole process. Every byte that stays resident in
// *local* video memory is charged here before it is committed; if the charge
// would push the resident total past the limit the allocation is **refused**
// (MediaError), never silently truncated. That matters: a scheduler that
// quietly shrinks a window produces a model that runs but is wrong, which is
// strictly worse than one that stops.
//
// Where the charges come from:
//   * `GpuArena` chunk creation (sched.cpp) — the weight / activation / keep
//     arenas are the model's entire resident footprint;
//   * `CudaContext::alloc_host` (via `create_upload_buffer` /
//     `create_readback_buffer`, sched.cpp) notes its transient staging bytes
//     against a *separate* counter, because pinned host memory is system memory
//     and must not be charged against the local budget.
//
// The limiter policy is min(kDefaultBudgetPercent% of the driver's budget for
// this process, the ceiling derived from the card's own VRAM), measured in
// bytes that are *resident*, never transient. The percentage is a safety
// margin, not a rounding fudge - see the policy comment below. `note_os_budget()`
// applies it, so any process that creates a CudaContext is protected without the
// caller having to remember.
//
// The ceiling is a function of the card, not a constant: the engine has to run
// on a 6 GB laptop part and on a 48 GB workstation, and a hard-coded ceiling is
// the one thing that makes the *same* binary plan 22 % of a big card and 100 % of
// a small one. `compute_hard_cap()` derives it per device (see the policy there),
// and `refresh()` re-derives it - together with the driver's current budget -
// every time a media tool is invoked, so a card that was busy a minute ago and
// is free now is planned against now.
#pragma once

#include <functional>
#include <mutex>
#include <string>

#include "util/media_common.hpp"

namespace phi::media {

class VramBudget {
public:
	// The ceiling the engine derives from the card it found. Two walls are
	// involved:
	//
	//   * the card's own memory (5.85 GB reported, 6 GB nominal on the reference
	//     laptop part) is the real one - past it a commit genuinely fails, so the
	//     ceiling must always stay below it;
	//   * the driver's budget for this process is what it can hand out *without*
	//     demoting pages to system memory. Past it a commit still succeeds, but
	//     every touch of a demoted page costs orders of magnitude more: a
	//     full-tensor reduction went from 20 ms to 13 s that way, which is a TDR,
	//     not a slowdown.
	//
	// Both are enforced (see note_os_budget). The driver's budget is a *live*
	// figure - it shrinks the moment another GPU client (the desktop compositor,
	// the WebView2 UI redrawing the generation's progress, a browser) claims
	// memory - so it is re-read on every charge (the live probe) and on every tool
	// invocation (`refresh()`), never cached across a run.
	//
	// `kHardCap` is the ceiling the owner authorised for the *reference* 6 GB
	// card, and it is kept as a floor rather than as the answer: on that card the
	// driver's own budget is smaller than it anyway, so the effective limit there
	// is exactly what it always was. `compute_hard_cap()` is what makes a bigger
	// card usable at all - a flat 5.2 GB ceiling left 78 % of a 24 GB card idle.
	static constexpr u64 kHardCap = 5220ull * 1024 * 1024;
	// Reserve policy of `compute_hard_cap`: bytes held back from the card's own
	// total for the OS, the compositor and the driver. A flat number is wrong at
	// both ends (200 MB is right for 6 GB and absurd on 48 GB; a flat percentage
	// wastes the small card), so it scales with the card between these bounds.
	static constexpr u64 kMinReserve = 192ull * 1024 * 1024;
	static constexpr u64 kMaxReserve = 2048ull * 1024 * 1024;

	// The driver's own budget for this process, as DXGI reports it (`Budget`)
	// together with the part of it that exists before the engine allocates
	// anything (`Usage` at create time, the context's WDDM charge). The ceiling is
	// `budget - usage`: what the driver will really back for this engine, which on
	// a 6 GB laptop part is ~5.0 GB against a 5.76 GB hardware-derived ceiling.
	// The hardware total stays as the absolute bound; the driver's figure is the
	// one that binds, because past it a commit does not fail - it is demoted to
	// system memory and the run trips the watchdog.
	void note_driver_vram(u64 budget, u64 usage, u64 reservation);
	u64 driver_budget() const;

	// Held back from the limit so a plan that spends "all of it" still has room
	// for the allocation that arrives next: one arena chunk (the granularity the
	// accountant refuses on) plus a fraction of the budget for the residuals no
	// estimate prices. Without it the last chunk of a window is refused in the
	// middle of a sampling loop, which is precisely the failure the driver limit
	// is supposed to prevent.
	static u64 residual_reserve(u64 driver_budget);

	// The ceiling this card supports: `dedicated_vram` less a reserve that scales
	// with the card (total/16, clamped to [kMinReserve, kMaxReserve]), never below
	// the reference card's authorised kHardCap, and never past the card itself.
	// PHI_VRAM_BUDGET_MB overrides it (it may raise as well as lower, bounded by the
	// physical card). 0 means "unknown card": the reference ceiling is used.
	static u64 compute_hard_cap(u64 dedicated_vram);

	// Note the adapter's dedicated VRAM for dynamic budget calculation.
	// Called by CudaContext during device creation.
	void note_dedicated_vram(u64 dedicated_vram);

	// The ceiling currently in force (recomputed by `refresh()`).
	u64 hard_cap_bytes() const;

	// Legacy: kept for compatibility. Forwards to the singleton's current ceiling,
	// which `refresh()` keeps in step with the device that is actually present.
	// (It used to be a function-local `static` initialised on first call, so a
	// caller that asked before the adapter was noted - or after a second device
	// appeared - got a stale answer for the life of the process.)
	static u64 hard_cap();

	// Fraction of the driver's budget the engine may plan against: the `safety` term
	// the plan's window formula always carried (§3.3). It was dropped for a while
	// because the 1080p decode looked like the 0.9 factor was refusing memory the
	// card could serve - but the decode's real problem was its own 6.4 GB footprint,
	// which the fused-norm / fused-upsample rewrites fixed (4.30 GB peak). With that
	// fixed the margin costs the image chain nothing, and it is the only thing
	// keeping the process off the driver's edge while the app is on screen.
	// PHI_VRAM_MARGIN_PERCENT overrides it (clamped to [50, 100]).
	//
	// 100: the limit *is* the system's own figure for this process, and the 90-98 %
	// band (vram_window.hpp) is what a plan then lands inside. Lowering this number
	// as well would take the margin twice - a 96 % limit under a 98 % band is really
	// a 94 % target, and on a 6 GB card the difference is a 1024x1024 VAE decode
	// that no longer fits (measured: it needs 4.77 GB of a 5.01 GB driver Budget).
	// PHI_VRAM_MARGIN_PERCENT lowers it for a caller that wants the extra headroom
	// back, and the residual reserve below is the margin the next allocation needs.
	static constexpr int kDefaultBudgetPercent = 100;
	static int budget_percent();
	// What the policy yields for a driver budget: min(percent x budget, the
	// ceiling the card supports). Exposed so the report and the enforcement cannot
	// drift apart.
	static u64 budget_limit(u64 driver_budget);

	// Everything the machine says about itself in one struct, so a tool can report
	// what it planned against instead of only what it asked for. Filled by
	// `refresh()`; `device_*` stay 0 when no device has answered yet.
	struct Environment {
		u64 device_total = 0;    // the card's own VRAM
		u64 device_free = 0;     // free right now, as the driver reports it
		u64 driver_budget = 0;   // free + what this engine holds (ledger units)
		u64 hard_cap = 0;        // ceiling derived from `device_total`
		u64 limit = 0;           // what the planner may spend
		int percent = kDefaultBudgetPercent;
		bool live = false;       // a device answered this refresh
		// The driver's own DXGI figures, for the report: what WDDM will let this
		// process commit in local video memory and what it already holds.
		u64 driver_local_budget = 0;
		u64 driver_local_usage = 0;
		u64 residual = 0;        // held back from `limit` (see residual_reserve)
	};

	// Re-read the device's live state and re-derive every limit from it: the card's
	// total refreshes the ceiling, the driver's current free bytes refresh the
	// budget, and the process limit is the smaller of the two. Called at the start
	// of every media tool invocation (the whole point of "adapt to the machine":
	// a card that another job was holding a minute ago is planned against now, and
	// one it has since filled is planned against honestly). Cheap: one
	// cuMemGetInfo plus arithmetic. Never throws.
	Environment refresh();
	Environment environment() const;

	// 0 = unlimited (the default until a device reports its budget).
	void set_limit(u64 bytes);
	// Records the driver's own figure and sets the process limit to
	// min(budget_percent() x that figure, the current ceiling). Called by
	// CudaContext, and again by `refresh()` on every tool invocation.
	void note_os_budget(u64 driver_budget);
	u64 limit() const;
	u64 os_budget() const;

	// Resident local VRAM. `charge` throws if it would cross the limit.
	u64 local() const;
	u64 peak() const;
	u64 headroom() const;
	bool would_exceed(u64 bytes) const;
	void charge(u64 bytes, const char* tag = nullptr);
	void release(u64 bytes);

	// Host-visible staging heaps (informational; never triggers the cap).
	u64 staging() const;
	void note_staging(i64 delta);
	void reset_staging();

	// Zero every counter (between independent runs).
	void reset();

	// Fired whenever a new local peak is reached, so a VramTracker can mirror it.
	std::function<void(u64)> on_peak;

	// What the driver says about this process right now: how much local memory it
	// is holding (`usage`) and how much it may still hand out without demoting
	// pages (`budget` - the figure that moves while we run).
	struct LiveVram {
		u64 usage = 0;
		u64 budget = 0;
		// Raw device figures, carried for the refusal message. `usage`/`budget`
		// are the ledger-comparable pair the check uses; these two are what the
		// driver actually reported, and when a refusal looks wrong they are the
		// numbers that say why.
		u64 free = 0;
		u64 total = 0;
	};

	// Set by CudaContext. The ledger below counts what *this engine* committed;
	// the driver also counts the upload ring, the staging heaps and every other
	// process on the box, and it is the driver that decides when a commit fails.
	// With the probe installed a charge that would
	// exhaust the card is refused as a MediaError while the device is still alive.
	//
	// The comparison is min(ledger limit, budget_percent() x the *live* budget),
	// not the limit sampled at device creation: the driver lowers the budget while
	// we run (another client claiming memory is the usual reason), and a ledger
	// working off the stale snapshot would plan itself straight into the paging
	// that ends in a TDR.
	void set_live_probe(std::function<LiveVram()> probe);
	u64 live_usage() const;
	u64 refused_live() const;   // refusals caused by the probe, not the ledger

	// The room this process may still commit *on the device*: the probe's live
	// limit less what the device says we already hold, and never more than what it
	// will actually hand out.
	//
	// The ledger limit and this number are **two different walls** - `charge()`
	// refuses on both - and they do not have to agree: the ledger counts what this
	// engine asked for, while the driver counts what the process really holds
	// (the WDDM context, anything another client took, and any chunk this process
	// released in its accounting while the device kept the pages). A plan sized
	// only from the ledger can therefore reserve room the card will not back,
	// which is not a demotion but a refusal in the middle of a run.
	//
	// `~0ull` when no device has answered (nothing to cap against), so a caller
	// can take it as a plain `min()` term.
	u64 live_headroom() const;

	u64 denied() const;       // charges refused so far
	std::string report() const;

private:
	void bump_peak_locked(u64 bytes);

	mutable std::mutex m_;
	u64 limit_ = 0;
	u64 os_budget_ = 0;
	u64 local_ = 0;
	u64 peak_ = 0;
	u64 staging_ = 0;
	u64 denied_ = 0;
	u64 refused_live_ = 0;
	// `budget_limit()` without the lock, for callers that already hold it.
	u64 budget_limit_for(u64 driver_budget) const;

	std::function<LiveVram()> live_probe_;
	u64 dedicated_vram_ = 0;                  // the card's own VRAM
	u64 hard_cap_ = kHardCap;                 // ceiling derived from it
	// DXGI's per-process figures, noted once at device creation.
	u64 driver_budget_ = 0;                   // raw Budget
	u64 driver_baseline_ = 0;                 // raw Usage before any engine bytes
	u64 driver_reservation_ = 0;              // raw AvailableForReservation
	bool driver_noted_ = false;
};

VramBudget& vram_budget();

// RAII: charges on construction, releases on destruction. Used by the upload
// pipeline, where the bytes are transient (a chunk is overwritten as soon as
// the copy lands) and the accounting must follow the ring head, not a bump.
class VramScope {
public:
	explicit VramScope(u64 bytes, const char* tag = nullptr) : bytes_(bytes) {
		vram_budget().charge(bytes, tag);
		ok_ = true;
	}
	~VramScope() {
		if (ok_) vram_budget().release(bytes_);
	}
	VramScope(const VramScope&) = delete;
	VramScope& operator=(const VramScope&) = delete;

private:
	u64 bytes_;
	bool ok_ = false;
};

}  // namespace phi::media