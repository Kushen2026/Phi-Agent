#include "host/residency.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace phi::media {

// ── the machine's budget ───────────────────────────────────────────────────
//
// The policy, literally moved: two thirds of *physical* memory, capped at 9 GB,
// and never more than what is free right now minus a reserve for the process.
//
// The free-memory term is the one that keeps a 16 GB box alive. `pin_range`
// commits RAM the process cannot give back (a pinned range is not reclaimable),
// while a checkpoint's own pages are clean and evictable, so a budget derived
// from installed memory alone can pin the machine out of memory: measured here,
// a 16 GB box with 9 GB of physical-memory budget pinned 8.97 GB for the H3
// chain and the *next* allocation was answered with an OOM kill - the process
// died mid-run with no error at all. The reserve keeps room for the engine's
// own host buffers, the WebView2 UI and the file pages the streaming reads
// still need.
//
// Basing this on installed memory rather than on the momentary free count is
// deliberate. A block format has no raw form the streamer can upload: every
// non-resident matrix is decoded, rotated and requantised, and the only thing
// that keeps that off the critical path of *every* sampling step is this cache.
// But the decode itself reads the entire file, so by the time the first block has
// been cached the page cache has eaten the machine - and a free-memory budget
// then collapses to a fraction of a gigabyte exactly when the cache is needed
// most, so the loop re-derives the weights it derived one step ago. Physical
// memory does not move, so this budget does not either; the file pages a large
// budget displaces are clean and reclaimable, and the OS evicts them to make
// room for the cache that is actually re-read.
bool host_weight_budget_forced() {
	const char* e = getenv("PHI_DIT_HOSTCACHE_MB");
	return e && *e;
}

u64 host_weight_budget_bytes() {
	if (const char* e = getenv("PHI_DIT_HOSTCACHE_MB")) {
		const long v = strtol(e, nullptr, 10);
		return v <= 0 ? 0 : (u64)v << 20;
	}
	const u64 total = SafeTensors::host_total_bytes();
	u64 want = total / 3 * 2;
	// The ceiling used to be a flat 9 GB, sized for the 16 GB box the policy was
	// measured on. That is the wrong number for the *checkpoint* as well as for the
	// machine: the H3 DiT is 20 GB as int8 and 40 GB as bf16, so the same card on
	// the same box has twice as much to pin or twice as much to re-read per step.
	// The cap now tracks installed memory - half of it, no lower than the old 9 GB
	// and no higher than 32 GB - which leaves a 16 GB box exactly where it was (9 GB,
	// still cut down by the free-memory term below), and lets a 64 GB box use the RAM
	// it has for a bf16 checkpoint instead of spending the run on the disk.
	u64 cap = total / 2;
	if (cap < (9ull << 30)) cap = 9ull << 30;
	if (cap > (32ull << 30)) cap = 32ull << 30;
	if (cap > total) cap = total;
	if (want > cap) want = cap;
	const u64 reserve = host_weight_reserve_bytes();
	const u64 free_now = SafeTensors::host_free_bytes();
	if (free_now > reserve && want > free_now - reserve) want = free_now - reserve;
	return want & ~((u64)4095);
}

// The reserve scales with the machine the way the VRAM policy's does: a small box
// needs a meaningful fraction back, a large one a bounded amount.
u64 host_weight_reserve_bytes() {
	const u64 total = SafeTensors::host_total_bytes();
	u64 reserve = total / 8;
	if (reserve < (1ull << 30)) reserve = 1ull << 30;
	if (reserve > (4ull << 30)) reserve = 4ull << 30;
	return reserve;
}

// ── units ─────────────────────────────────────────────────────────────────

std::string checkpoint_precision_label(const SafeTensors& st) {
	// The precision that *dominates the file by bytes*, not the first weight's.
	// Every one of these checkpoints mixes families: the H3 bundle keeps its patch
	// and output projections in fp32 beside a bf16 backbone, and the 32B tower's
	// fp32 head sits in front of 50 int4 layers. Reporting the first one labelled a
	// 40 GB bf16 H3 as "f32" and a 15 GB int4 tower as "bf16" - the two facts a
	// reader needs to tell those runs apart, both wrong.
	std::map<int, u64> bytes;
	for (const StTensor& t : st.tensors()) {
		// Only weights: the sidecars (`.weight_scale`, codebooks) are not, and a
		// `comfy_quant` blob is not either.
		if (t.name.size() < 7 || t.name.compare(t.name.size() - 7, 7, ".weight") != 0) continue;
		const WeightPrecision p = weight_precision_of(st, t);
		if (p == WeightPrecision::Other) continue;
		bytes[(int)p] += t.nbytes;
	}
	int best = -1;
	u64 best_bytes = 0;
	for (const auto& kv : bytes) {
		if (kv.second > best_bytes) {
			best = kv.first;
			best_bytes = kv.second;
		}
	}
	if (best < 0) return "?";
	const char* nm = weight_precision_name((WeightPrecision)best);
	if (!nm || !*nm) return "?";
	// Packed families say their bits, which is what makes the *size* legible: w4a4
	// is a quarter of bf16 per weight, and that is why the 32B tower takes a tenth
	// of the RAM its bf16 twin would.
	return nm;
}

WeightUnit weight_unit_of(const SafeTensors& st, const std::string& prefix) {
	WeightUnit u;
	for (const StTensor& t : st.tensors()) {
		if (t.name.size() <= prefix.size()) continue;
		if (t.name.compare(0, prefix.size(), prefix) != 0) continue;
		if (t.nbytes == 0) continue;
		u.ranges.push_back({t.offset, t.nbytes});
		u.bytes += t.nbytes;
	}
	return u;
}

// ── the pool ──────────────────────────────────────────────────────────────

HostWeightPool& HostWeightPool::instance() {
	static HostWeightPool pool;
	return pool;
}

u64 HostWeightPool::budget_locked() const {
	if (!budget_set_) {
		budget_ = host_weight_budget_bytes();
		budget_set_ = true;
	}
	return budget_;
}

u64 HostWeightPool::budget() const {
	std::lock_guard<std::mutex> lk(m_);
	return budget_locked();
}

u64 HostWeightPool::held() const {
	std::lock_guard<std::mutex> lk(m_);
	return held_;
}

u64 HostWeightPool::available() const {
	std::lock_guard<std::mutex> lk(m_);
	const u64 b = budget_locked();
	return b > held_ ? b - held_ : 0;
}

HostWeightPool& host_weight_pool() { return HostWeightPool::instance(); }

std::string host_weight_pool_report() { return HostWeightPool::instance().report(); }

// The grant loop. The only subtle part is the revocation: a Primary's request may
// take back every Secondary grant, and dropping one means calling the owner's
// callback - which will call `release` on this object. So the victims are removed
// from the table under the lock and their callbacks are run *after* it is
// released, then the loop re-reads the ledger it just changed.
u64 HostWeightPool::acquire(const void* owner, Rank rank, u64 want, const char* name,
                            std::function<void()> drop) {
	if (!owner) return 0;
	for (;;) {
		std::vector<std::function<void()>> victims;
		{
			std::lock_guard<std::mutex> lk(m_);
			// A re-grant for an owner that already holds one replaces it: the callers
			// release first, but a ledger that double-counts its own owner is the kind
			// of bug that only shows up as a mysterious partial grant.
			if (auto old = owners_.find(owner); old != owners_.end()) {
				held_ -= std::min(held_, old->second.granted);
				owners_.erase(old);
			}
			const u64 budget = budget_locked();
			const u64 avail = budget > held_ ? budget - held_ : 0;
			const bool reclaimable =
			    rank == Rank::Primary && avail < want && !owners_.empty();
			if (reclaimable) {
				bool any = false;
				for (auto it = owners_.begin(); it != owners_.end();) {
					if (it->second.rank != Rank::Secondary) {
						++it;
						continue;
					}
					any = true;
					held_ -= std::min(held_, it->second.granted);
					if (it->second.drop) victims.push_back(std::move(it->second.drop));
					it = owners_.erase(it);
				}
				if (!any) {
					// Nothing left to reclaim: grant what is there.
					const u64 granted = std::min(want, avail);
					Owner& o = owners_[owner];
					o.rank = rank;
					o.want = want;
					o.granted = granted;
					o.name = name ? name : "?";
					o.drop = std::move(drop);
					held_ += granted;
					return granted;
				}
			} else {
				const u64 granted = std::min(want, avail);
				Owner& o = owners_[owner];
				o.rank = rank;
				o.want = want;
				o.granted = granted;
				o.name = name ? name : "?";
				o.drop = std::move(drop);
				held_ += granted;
				return granted;
			}
		}
		// Outside the lock: a victim frees its pins and releases its own grant.
		for (auto& v : victims)
			if (v) v();
	}
}

void HostWeightPool::shrink(const void* owner, u64 granted) {
	std::lock_guard<std::mutex> lk(m_);
	auto it = owners_.find(owner);
	if (it == owners_.end()) return;
	const u64 was = it->second.granted;
	it->second.granted = std::min(was, granted);
	held_ -= std::min(held_, was - it->second.granted);
}

void HostWeightPool::release(const void* owner) {
	std::lock_guard<std::mutex> lk(m_);
	auto it = owners_.find(owner);
	if (it == owners_.end()) return;
	held_ -= std::min(held_, it->second.granted);
	owners_.erase(it);
}

void HostWeightPool::reset() {
	std::vector<std::function<void()>> drops;
	{
		std::lock_guard<std::mutex> lk(m_);
		for (auto& kv : owners_)
			if (kv.second.drop) drops.push_back(std::move(kv.second.drop));
		owners_.clear();
		held_ = 0;
		budget_ = 0;
		budget_set_ = false;
	}
	for (auto& d : drops)
		if (d) d();
}

std::string HostWeightPool::report() const {
	std::lock_guard<std::mutex> lk(m_);
	if (owners_.empty()) return std::string();
	std::string out = format_bytes(held_) + " of " + format_bytes(budget_locked()) +
	                  " in system memory";
	out += " (";
	bool first = true;
	for (const auto& kv : owners_) {
		if (!first) out += ", ";
		first = false;
		char buf[64];
		snprintf(buf, sizeof buf, "%s %.2f", kv.second.name.c_str(),
		         (double)kv.second.granted / 1073741824.0);
		out += buf;
	}
	out += ")";
	return out;
}

// ── the per-checkpoint plan ───────────────────────────────────────────────

WeightResidency::~WeightResidency() { release_grant(); }

void WeightResidency::attach(SafeTensors* st, std::string owner,
                             HostWeightPool::Rank rank) {
	release_grant();
	st_ = st;
	owner_ = std::move(owner);
	rank_ = rank;
	// A fresh attachment is a fresh checkpoint: `open()` may run twice on one object
	// (a VAE whose weights were handed back re-opens), and keeping the old unit list
	// would register every block twice.
	units_.clear();
	plan_ = Plan{};
}

void WeightResidency::add_unit(WeightUnit unit) {
	if (unit.bytes == 0) return;
	units_.push_back(std::move(unit));
}

void WeightResidency::add_unit(const std::vector<std::pair<u64, u64>>& ranges) {
	WeightUnit u;
	for (const auto& r : ranges) {
		if (r.second == 0) continue;
		u.ranges.push_back(r);
		u.bytes += r.second;
	}
	add_unit(std::move(u));
}

void WeightResidency::clear_units() {
	units_.clear();
	plan_ = Plan{};
}

void WeightResidency::release_grant() {
	HostWeightPool::instance().release(this);
}

void WeightResidency::drop() {
	if (st_) st_->drop_cache();
	release_grant();
	// The tier is unplanned: the bytes are back on the checkpoint, and a later
	// `plan()` re-pins them (see the header on why the cache is all-or-nothing).
	const i64 units = plan_.units;
	const u64 vram_bytes = plan_.vram_bytes;
	plan_ = Plan{};
	plan_.units = units;
	plan_.vram_bytes = vram_bytes;
}

const WeightResidency::Plan& WeightResidency::plan(i64 vram_units) {
	const i64 units = (i64)units_.size();
	vram_units = std::clamp<i64>(vram_units, 0, units);
	if (plan_.planned && plan_.vram_units == vram_units) return plan_;
	// A moved window invalidates which units are the streamed ones, and the pinned
	// set cannot be given back a unit at a time (it is `SafeTensors::drop_cache`,
	// all or nothing). Re-plan from a clean set rather than leaving the window's
	// bytes pinned and never read.
	if (plan_.planned) {
		if (st_) st_->drop_cache();
		release_grant();
	}

	Plan p;
	p.planned = true;
	p.units = units;
	p.vram_units = vram_units;
	for (i64 i = 0; i < units; i++) {
		if (i < vram_units) {
			p.vram_bytes += units_[(size_t)i].bytes;
		} else {
			p.disk_units++;
			p.disk_bytes += units_[(size_t)i].bytes;
			p.want_bytes += units_[(size_t)i].bytes;
		}
	}
	plan_ = p;
	if (!st_ || p.disk_units == 0 || p.want_bytes == 0) return plan_;

	HostWeightPool& pool = HostWeightPool::instance();
	if (pool.budget() == 0) return plan_;

	// ── the page cache is a tier too, and it is the one that scales with the file ──
	//
	// A working set the OS can keep costs nothing extra: after the first step the
	// bytes are in the page cache, and reads out of it are already overlapped with
	// the GPU by the look-ahead slot (`prefetch_ranges`). Pinning such a set buys
	// nothing and costs a *serial* pass over it - the pin loop reads the bytes with no
	// compute to hide the read behind.
	//
	// Measured on this box (64 GB, H3 DiT as **bf16 - 37.5 GB on disk**, 30.2 GB
	// streamed per step, 6-step clip): 89.2 s with the tier off against 103.7 s with
	// 30.2 GB pinned, both runs with the file already in the page cache; the
	// difference is the pin's own read. The same policy on a 16 GB box would pin 6 GB
	// of a set it cannot hold, which is the case the tier exists for.
	//
	// So the tier only spends RAM on what the page cache *cannot* keep: the streamed
	// set is compared against the memory the OS reports as available (which counts
	// what it is already caching) less the reserve. Nothing here names a precision or
	// a size - an int8 checkpoint is half the bytes, so it fits where its bf16 twin
	// does not, and the rule answers each case from the file it was handed.
	const u64 reserve = host_weight_reserve_bytes();
	const u64 avail = SafeTensors::host_free_bytes();
	const u64 page_cache_room = avail > reserve ? avail - reserve : 0;
	if (!host_weight_budget_forced() && p.want_bytes <= page_cache_room) {
		plan_.page_cache_covers = true;
		return plan_;
	}

	// What the RAM tier *asks* for: every streamed unit, bounded by the pool's whole
	// budget - not by what is free right now. A Primary asking after a Secondary has
	// taken the machine's RAM has to be able to ask for more than "what is left", or
	// the pool's revocation class would never fire and the phase that needs the RAM
	// most would get its leftovers. `acquire` reclaims; the pin loop below stops at
	// the machine's own floor (one free-memory reading per unit) and hands the
	// unused part of the grant back.
	const u64 ask = std::min(p.want_bytes, pool.budget());
	if (ask == 0) return plan_;

	// The delegate is a plain `this`: the object owns the pins, and the pool only
	// ever calls it from `acquire`/`reset`, never while it holds the lock.
	const u64 granted = pool.acquire(this, rank_, ask, owner_.c_str(), [this] { drop(); });
	if (granted == 0) return plan_;

	// Pin in registration order - the order the loop touches the units - so a
	// budget that covers part of the run covers its front. A failure is a reason to
	// pin less, never to fail a run: the same bytes still come off the checkpoint.
	//
	// The pin unit is a *range* (one tensor), not a whole unit, and that is what
	// makes the tier adapt to the checkpoint's precision rather than to one
	// checkpoint. A DiT block is 226 MB of int8 and 440 MB of bf16; the token
	// refiner's fc1 is 294 MB against 588 MB; the 32B tower's layer is 300 MB at
	// int4 and 1.2 GB at bf16. Pinning whole units meant a grant that stopped just
	// short of one block left the leftover unspent - a few hundred MB on the int8
	// file and up to a gigabyte on the bf16 one, where the run needs every byte it
	// can get. Per range, the leftover is under one tensor instead of under one
	// block. A unit counts as *pinned* only when every one of its ranges is, which
	// is what the prefetch gate asks (`unit_pinned`), because a half-pinned block
	// still has to come off the disk for the rest.
	// The look-ahead slot's headroom: the largest unit this checkpoint streams, which
	// is what the slot will have to hold. Taken from the file's own ranges, so it is
	// 740 MB for a bf16 DiT block and 370 MB for the same block as int8.
	u64 slot = 0;
	for (i64 i = vram_units; i < units; i++)
		if (units_[(size_t)i].bytes > slot) slot = units_[(size_t)i].bytes;
	if (slot == 0) slot = 768ull << 20;

	u64 pinned = 0;
	i64 full = 0;       // units pinned end to end
	i64 partial = 0;    // units with some, not all, of their ranges pinned
	bool room = true;
	for (i64 i = vram_units; i < units && room; i++) {
		const WeightUnit& u = units_[(size_t)i];
		if (u.bytes == 0) {
			full++;
			continue;
		}
		i64 got = 0;
		for (const auto& r : u.ranges) {
			if (pinned + r.second > granted || SafeTensors::host_free_bytes() <= reserve) {
				room = false;
				break;
			}
			try {
				st_->pin_range(r.first, r.second, slot);
			} catch (const std::exception&) {
				room = false;
				break;
			}
			pinned += r.second;
			got++;
		}
		if (got == (i64)u.ranges.size()) {
			full++;
		} else if (got > 0) {
			partial++;
		}
	}
	if (pinned < granted) pool.shrink(this, pinned);
	plan_.ram_units = full;
	plan_.ram_partial_units = partial;
	plan_.ram_bytes = pinned;
	plan_.disk_units = units - vram_units - full;
	plan_.disk_bytes = plan_.want_bytes - pinned;
	return plan_;
}

std::string weight_plan_line(const WeightResidency& wr, const char* tag, const char* precision) {
	std::string out;
	if (tag && *tag) out = std::string(tag) + ": ";
	if (precision && *precision) {
		// The precision is part of the report, not a detail of it: the same
		// architecture streams twice the bytes as bf16 that it does as int8 (a 20 GB
		// H3 file against 40 GB), so "12/50 blocks in RAM" means two different things
		// and only the pair says which run this was.
		out += std::string(precision) + ", ";
	}
	out += wr.describe();
	const std::string ledger = host_weight_pool_report();
	if (!ledger.empty()) out += " | " + ledger;
	return out;
}

std::string WeightResidency::describe() const {
	const Plan& p = plan_;
	if (!p.planned) return std::string();
	char buf[320];
	if (p.page_cache_covers) {
		snprintf(buf, sizeof buf,
		         "%lld/%lld units on the card (%.2f GB), the other %lld (%.2f GB) fit the "
		         "page cache, so they are not pinned",
		         (long long)p.vram_units, (long long)p.units,
		         (double)p.vram_bytes / 1073741824.0, (long long)p.disk_units,
		         (double)p.disk_bytes / 1073741824.0);
		return buf;
	}
	snprintf(buf, sizeof buf,
	         "%lld/%lld units on the card (%.2f GB), %lld in RAM%s (%.2f of %.2f GB asked), "
	         "%lld off the checkpoint (%.2f GB)",
	         (long long)p.vram_units, (long long)p.units,
	         (double)p.vram_bytes / 1073741824.0, (long long)p.ram_units,
	         p.ram_partial_units > 0
	             ? (std::string(" + ") + std::to_string(p.ram_partial_units) + " partial").c_str()
	             : "",
	         (double)p.ram_bytes / 1073741824.0, (double)p.want_bytes / 1073741824.0,
	         (long long)p.disk_units, (double)p.disk_bytes / 1073741824.0);
	return buf;
}

}  // namespace phi::media
