// Where each part of a checkpoint lives, and where a reader gets it from.
//
// Every generation chain in this engine loads weights the same way and, until
// now, decided the *host* half of it for itself: three models each computed the
// machine's budget (`host_cache_budget_bytes`), each pinned up to that much, and
// each picked its own order and its own stop condition. Three of them (`ImageDiT`,
// both text towers) pinned nothing at all, so the largest chain on a card that
// cannot hold it - 7.25 GB of DiT re-read on every one of 25 steps - paid the disk
// for bytes it had already read, while a *smaller* model lower in the same run was
// holding gigabytes of system memory in a cache it would never touch again.
//
// ── the tiers ─────────────────────────────────────────────────────────────
//
//   Vram  the model's own resident window (`plan_residency` in every chain).
//         Read by *not* being read: the block is already on the device and the
//         loader never opens the file for it. That decision - how many blocks the
//         card keeps - is the engine's existing VRAM policy and is deliberately
//         **not** part of this header: the formula must not move, because on the
//         reference card the margin between the plan and the driver's ceiling is
//         two percent.
//   Ram   the units that did not fit, held in system memory once and served
//         from there for the rest of the run (`SafeTensors::pin_range`). A read
//         from this tier is a memcpy out of system memory, and when the bytes are
//         page-locked it is a direct DMA with no copy at all.
//   Disk  everything else: the checkpoint's own mapping or its ReadFile path.
//
// Between Ram and Disk sits the OS: a working set that fits the machine's page
// cache is served out of it (by ReadFile, or by the mapping) without this code
// doing anything. That is a *tier* in its own right, and the tier plan has to
// respect it - a set the page cache already keeps costs nothing, and pinning it
// buys nothing while costing a serial pass over it. Measured on the reference box
// (64 GB, H3 DiT as bf16, 37.5 GB on disk, 30.2 GB streamed per step): 87.9 s for
// a 6-step clip with the streamed set left to the page cache against 103.7 s with
// it pinned. So the plan pins only what the page cache cannot keep (see
// `WeightResidency::plan`), which is exactly the small-memory / large-checkpoint
// case the tier exists for.
//
// ── precision is a size, and the size is the file's ────────────────────────
//
// Nothing here assumes a checkpoint's size: every figure is `t.nbytes` summed over
// the ranges the loader actually reads, and the precision enters only through
// that. It has to, because the same architecture is a very different problem at
// each width - the H3 DiT is 20 GB as int8 and 40 GB as bf16, the 32B tower 14 GB
// at int4 and four times that at bf16 - and the three answers (how many blocks fit
// the card, how much RAM the tier may hold, how much must stream) all move with it:
//
//   * the *resident window* is priced by the block's own bytes, so a bf16
//     checkpoint keeps half the blocks an int8 one does on the same card (measured
//     here: 10 of the H3 DiT's 50 blocks, 736 MB each, in a 10.68 GB budget);
//   * the *RAM tier* pins per tensor from the file's own ranges, so a block that is
//     736 MB as bf16 gets 736 MB of grant and the same block at 370 MB gets 370 -
//     and a grant that stops inside a block still pins its leading matrices rather
//     than leaving a third of the RAM unspent;
//   * the *page-lock* ceiling and the look-ahead slot's headroom scale with the
//     machine and with the largest unit the checkpoint streams, so a bf16 run
//     keeps its DMA path where a fixed 768 MB of headroom would have handed the
//     slot to the ring;
//   * the *report* prints the precision beside the counts (see
//     `weight_plan_line`), because "12 of 50 blocks in RAM" means two different
//     runs at int8 and at bf16 and only the pair says which one this was.
//
// The tier is resolved *inside* `SafeTensors`, per read, so every loader in the
// engine - the DiTs, the text towers, the VAEs, the int8 streamers and the slab
// decoders - gets the best tier that holds the bytes without knowing the tiers
// exist. `read_file` has served the pinned set and the look-ahead slot for a
// while; `pinned_span` (host/st.hpp) is the same cache offered to the readers
// that copy out of the *mapping*, which is every decode path.
//
// What this header adds is (a) the decision of which unit goes into which tier,
// (b) one process-wide ledger that decision is made against, and (c) the
// reporting that makes it checkable.
//
// ── why one ledger ────────────────────────────────────────────────────────
//
// A generation runs several checkpoints through one process: an image chain is
// text tower (9 GB) -> DiT (7.25 GB) -> VAE (0.25 GB); the video chain is 32B
// tower (15 GB) -> H3 DiT (19.5 GB) -> video VAE (5 GB). Each model computing
// "two thirds of the machine, capped at 9 GB" for itself and spending it means
// the *first* one to load takes the whole budget, and the one that is about to
// re-read 15 GB per sampling step gets nothing. So the RAM tier is granted from
// one pool, in two classes:
//
//   Primary    the model the run is actually sampling (a DiT). Its bytes are
//              re-read once per step for the whole loop.
//   Secondary  a model that runs one pass per call (a text tower, the ACE LM, a
//              VAE). Cheap to stream, and its pins are revocable: when a Primary
//              asks for room the pool hands back Secondary grants, because the
//              Secondary phase is over by the time the Primary plans (the text
//              encode and the VAE encode both finish before the sampling loop
//              starts).
//
// Revocation is the point of the ledger. Without it the ordering of the chain -
// which is fixed, encode first, sample last - would hand the machine's RAM to
// the phase that needs it least.
//
// ── the roll call, so "every model" is checkable ──────────────────────────
//
//   Vram window + Ram tier (a Primary, i.e. it re-reads per sampling step)
//     ImageDiT     qwen_image   - was pinning *nothing* before this
//     AvDiT        h3dit        - had its own pin loop and budget
//     AceDiT       acedit       - had its own pin loop and budget
//   Vram window + Ram tier (a Secondary: one pass per call)
//     TextEncoder8B  te8        - was pinning nothing
//     TextEncoder32B te32       - was pinning nothing
//     AceLm          acelm      - had its own pin loop, budget and a crossover
//                                  heuristic that the pool's classes now enforce
//     VideoVae       h3vae      - had its own pin loop with a "half of free RAM"
//                                  rule
//   Whole model in VRAM, no tier needed
//     ImageVae, AceVae, BreezeTts' backbone, the audio VAE / vocoder / codec
//     These load every weight once and read each of them once per call: there is
//     no recurring read for a pin to serve, and their checkpoints fit beside the
//     DiT's window by construction (the plans subtract what the ledger already
//     holds). A model that later grows past the card wants a *streaming* loop
//     inside its own forward first; the tier only helps what is re-read.
//
// Every one of them reads its weights through `SafeTensors`, which is where the
// tier is resolved, so none of them needed to learn what a tier is.
#pragma once

#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "host/st.hpp"
#include "util/media_common.hpp"

namespace phi::media {

// ── the machine's host-side weight budget ──────────────────────────────────
//
// Moved here from `model_common.cpp` when the caches stopped being per-model:
// the policy (two thirds of installed memory, capped, and never more than what is
// free minus a reserve) is a property of the *machine*, not of a model, and every
// caller must read the same number or the tiers are planned against fiction.
// PHI_DIT_HOSTCACHE_MB still overrides it; 0 disables the RAM tier everywhere.
u64 host_weight_budget_bytes();
// True when that budget came from PHI_DIT_HOSTCACHE_MB rather than from the
// machine. An explicit budget is an instruction ("pin this much"), so the
// page-cache gate below - which exists to stop the tier spending RAM on a set the
// OS already keeps - stands aside for it; that is also what makes the A/B
// ("what does pinning cost on this box") runnable.
bool host_weight_budget_forced();
// RAM the process must leave alone (the engine's own buffers, the UI, the file
// pages the streaming reads still need). A pin loop stops when free memory falls
// to it, whatever the ledger says.
u64 host_weight_reserve_bytes();

// The checkpoint's own label, for a report: the precision of its first weight
// tensor ("int8", "w4a8", "bf16", "fp32", ...), or "?" when the file has none.
//
// One call, because every model would otherwise spell "which tensor do I ask about"
// for itself - and the answer must be the same everywhere, since the label is what
// tells a 20 GB int8 H3 checkpoint from its 40 GB bf16 twin in a run log.
std::string checkpoint_precision_label(const SafeTensors& st);

// Every byte this process has read out of the RAM tier so far - through
// `read_file` (the streamers), through `pinned_span` (the decoders) and straight
// off a page-locked buffer by the driver. It is the only figure that says the tier
// is *working*: a pin nobody reads costs system memory and buys nothing, and the
// planned bytes alone cannot tell the two cases apart. Reset with the generation
// (`release_generation_state`), printed by `PHI_MEDIA_STATE`.
u64 host_weight_served_bytes();
void reset_host_weight_served();

// ── one streamable unit of a checkpoint ────────────────────────────────────
//
// The granularity the tiers are decided at: a DiT block, a text-tower layer, a
// VAE decoder block, a token refiner. `ranges` are the (file offset, bytes) the
// unit's loader reads, in that loader's own order.
struct WeightUnit {
	std::vector<std::pair<u64, u64>> ranges;
	u64 bytes = 0;
};

// The file ranges of every tensor whose name starts with `prefix`, in header
// order. The one-liner a loader uses to describe a unit it reads by name.
WeightUnit weight_unit_of(const SafeTensors& st, const std::string& prefix);

// ── the process-wide host (RAM) weight ledger ──────────────────────────────
//
// Grants are in bytes, tracked per owner (a `WeightResidency`, keyed by its
// address). A grant is a *promise*: the owner pins up to that many bytes and
// calls `shrink` with what it could not use. Nothing here touches a file - the
// pool decides how much RAM a checkpoint may hold, `WeightResidency` spends it.
class HostWeightPool {
public:
	enum class Rank { Primary = 0, Secondary = 1 };

	static HostWeightPool& instance();

	// The budget this process plans against, decided once per process (and
	// re-decided by `reset`, so a long-lived app can follow the machine).
	u64 budget() const;
	u64 held() const;
	u64 available() const;

	// Reserve up to `want` bytes for `owner`. `drop` is called - with the lock
	// released, and only from a Primary's request - when a Secondary's grant has to
	// be reclaimed to make room; it must free the owner's pins and release its
	// grant. `name` is what the report calls the owner ("h3dit", "te32", ...): the
	// owners are opaque addresses, and a ledger nobody can read is not a ledger.
	// Returns the bytes granted, 0..want.
	u64 acquire(const void* owner, Rank rank, u64 want, const char* name,
	            std::function<void()> drop = nullptr);
	// Give the unused part of a grant back (the pin loop stopped early).
	void shrink(const void* owner, u64 bytes);
	void release(const void* owner);
	// Drop every owner's pins and zero the ledger: the end of a generation.
	// (Nothing survives one - see `release_generation_state`.)
	void reset();

	// "5.12 GB of 9.00 GB in system memory (h3dit 4.20, te32 0.91)"; empty when
	// nothing is granted.
	std::string report() const;

private:
	// Decides the budget on first use and after every `reset`, so a long-lived
	// process keeps following the machine it is running on.
	u64 budget_locked() const;

	struct Owner {
		Rank rank = Rank::Primary;
		u64 want = 0;
		u64 granted = 0;
		std::string name;
		std::function<void()> drop;
	};
	mutable std::mutex m_;
	mutable std::map<const void*, Owner> owners_;
	u64 held_ = 0;
	mutable u64 budget_ = 0;
	mutable bool budget_set_ = false;
};

// ── the ledger's two convenience faces ─────────────────────────────────────
HostWeightPool& host_weight_pool();
// The ledger as one line, for a model's own diagnostics: "5.12 GB of 9.00 GB in
// system memory (h3dit 4.20, te32 0.92)". Empty when nothing is granted.
std::string host_weight_pool_report();

// ── one checkpoint's tier plan ─────────────────────────────────────────────
//
// Registered by the loader (which owns the reader and knows the units), planned
// once the model's VRAM window is fixed, and dropped at the phase boundary with
// it.
//
//   residency.attach(&st_, "h3dit", HostWeightPool::Rank::Primary);
//   for (i) residency.add_unit(block_ranges[i]);
//   residency.add_unit(refiner_ranges);          // always streamed -> last
//   ...
//   const auto& p = residency.plan(res_n);       // blocks [0, res_n) are on the card
//   ...
//   residency.drop();                            // phase boundary / teardown
//
// `plan(n)` means: units [0, n) are already in the **VRAM** tier (the model
// uploaded them itself), and units [n, count) are candidates for the **RAM**
// tier, in registration order - which is the order the loop touches them, so a
// budget that covers only part of the run covers the front of it rather than a
// scattering.
class WeightResidency {
public:
	struct Plan {
		bool planned = false;
		i64 units = 0;          // units registered
		i64 vram_units = 0;     // [0, vram_units) are resident on the device
		i64 ram_units = 0;      // of the rest, how many are pinned end to end
		i64 ram_partial_units = 0;   // units with only some tensors pinned
		i64 disk_units = 0;     // the remainder: read off the checkpoint
		u64 vram_bytes = 0;
		u64 ram_bytes = 0;      // what the pinned units weigh (= the grant used)
		u64 disk_bytes = 0;
		u64 want_bytes = 0;     // what the RAM tier asked for
		// The streamed set fits what the OS can keep in its page cache, so the tier
		// deliberately pinned nothing (see `plan`): the bytes are served out of the
		// cache, overlapped with the GPU, and a pin would be a serial read of the same
		// bytes. Set for the sets a big box can hold whole - which is where a bf16
		// checkpoint minus its resident window still fits, and an int8 one does on any
		// machine at all.
		bool page_cache_covers = false;

		// True when unit `i` (in registration order) is whole in the pinned (RAM)
		// tier. Units are pinned from `vram_units` upward, so a caller whose units are
		// a block loop can ask "is this block already in system memory?" - the one
		// question a prefetch has to answer, because reading a pinned block ahead of
		// time is a memcpy nobody will use.
		//
		// A *partially* pinned unit answers false: its unpinned tensors still come off
		// the checkpoint, so the look-ahead is still worth issuing. That case only
		// arises for the last unit the grant covered, and it is the one that proves
		// the tier is priced per tensor rather than per block (see `plan`).
		bool unit_pinned(i64 i) const { return i >= vram_units && i < vram_units + ram_units; }
	};

	WeightResidency() = default;
	// Releases the grant. The *pins* are the reader's (see `drop`), so a caller
	// that wants them freed must call `drop()` first - which every model does, from
	// the same place it hands its VRAM window back.
	~WeightResidency();
	WeightResidency(const WeightResidency&) = delete;
	WeightResidency& operator=(const WeightResidency&) = delete;

	// `st` must outlive this object (declare the member after the reader).
	void attach(SafeTensors* st, std::string owner, HostWeightPool::Rank rank);
	bool attached() const { return st_ != nullptr; }
	const std::string& owner() const { return owner_; }

	void add_unit(WeightUnit unit);
	void add_unit(const std::vector<std::pair<u64, u64>>& ranges);
	void clear_units();

	// Decide the tiers and pin what fits. Idempotent for an unchanged
	// `vram_units`; a *moved* window re-plans from scratch, because the units it
	// now holds are no longer the ones to pin (the pins are dropped and rebuilt,
	// which costs the same read the first plan did - and only happens when the
	// window really moved).
	const Plan& plan(i64 vram_units);

	// Release the grant and the reader's pinned set. Safe to call any number of
	// times, and from the pool's revocation callback.
	void drop();

	const Plan& state() const { return plan_; }
	// "9/50 on the card (3.42 GB), 12 in RAM (4.21 GB), 29 off the checkpoint"
	std::string describe() const;

private:
	void release_grant();

	SafeTensors* st_ = nullptr;
	std::string owner_;
	HostWeightPool::Rank rank_ = HostWeightPool::Rank::Primary;
	std::vector<WeightUnit> units_;
	Plan plan_;
};

// The one line a model prints when its plan is decided: the unit counts per tier
// plus the process ledger, so a run's log says where every chain's weights are.
//   "qwen_image: 7/30 units on the card (3.42 GB), 18 in RAM (4.21 of 5.48 GB
//    asked), 5 off the checkpoint (1.10 GB) | 4.21 GB of 9.00 GB in system memory"
// `precision` is the checkpoint's own ("int8", "w4a8", "bf16", ...) and is
// printed with the counts: the same model is twice the bytes at bf16, so the two
// numbers only mean something together.
std::string weight_plan_line(const WeightResidency& wr, const char* tag,
                             const char* precision = nullptr);

// Plans the RAM tier when the enclosing scope ends, from a window count that is
// only final on the way out. The two text towers decide their window inside a
// function with half a dozen early returns (no device, no budget, a debug pin, no
// room beside the frame); this is how each of them gets *one* plan call that runs
// whatever path was taken, without the plan leaking into every branch.
class TierPlanOnExit {
public:
	TierPlanOnExit(WeightResidency& wr, const i64& vram_units)
	    : wr_(&wr), units_(&vram_units) {}
	~TierPlanOnExit() {
		if (wr_) wr_->plan(*units_);
	}
	TierPlanOnExit(const TierPlanOnExit&) = delete;
	TierPlanOnExit& operator=(const TierPlanOnExit&) = delete;

private:
	WeightResidency* wr_ = nullptr;
	const i64* units_ = nullptr;
};

}  // namespace phi::media
