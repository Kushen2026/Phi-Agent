#include "runtime/vram_window.hpp"

#include <algorithm>

namespace phi::media {

u64 window_charge_bytes(i64 n, u64 block_bytes, u64 chunk_bytes) {
	if (n <= 0 || block_bytes == 0) return 0;
	const u64 bytes = (u64)n * block_bytes;
	if (chunk_bytes == 0) return bytes;
	return (bytes + chunk_bytes - 1) / chunk_bytes * chunk_bytes;
}

WindowPlan plan_window_layers(u64 limit, u64 other, u64 block_bytes, u64 chunk_bytes,
                              i64 n_total) {
	WindowPlan p;
	p.window = 0;
	p.usage = other;
	if (n_total > 0 && block_bytes != 0 && limit != 0 && other < limit) {
		i64 n = 0;
		for (;;) {
			if (n >= n_total) break;
			const u64 w = window_charge_bytes(n + 1, block_bytes, chunk_bytes);
			if (other + w > limit) break;
			n++;
			p.window = w;
		}
		p.n = n;
		p.usage = other + p.window;
		p.fills = n == n_total ||
		          other + window_charge_bytes(n + 1, block_bytes, chunk_bytes) > limit;
	} else {
		p.fills = n_total <= 0 || block_bytes == 0;
	}
	p.percent = limit ? (int)(100ull * p.usage / limit) : 0;
	p.in_band = p.percent >= kWindowBandLowPercent && p.percent <= kWindowBandHighPercent;
	return p;
}

u64 request_pixel_reserve_bytes(u64 video_pixels, u64 ref_pixels) {
	// The total pixel budget, then the weight. Both addends are counts the
	// request's own geometry supplies, so a caller that over-counts (a reference
	// audio row read as a pixel) only makes the plan more conservative.
	const u64 total = video_pixels > ~0ull - ref_pixels ? ~0ull : video_pixels + ref_pixels;
	if (total > ~0ull / kRequestPixelWeightBytes) return ~0ull;
	return total * kRequestPixelWeightBytes;
}

WindowMove tune_window(const WindowLive& l, const WindowPolicy& p, i64* delta) {
	if (delta) *delta = 0;
	// Nothing to decide against: no device answered, no stack, or nothing measured.
	if (l.limit == 0 || l.n_total <= 0 || l.block_bytes == 0) return WindowMove::Hold;

	const u64 low = l.limit * (u64)std::max(0, p.low_percent) / 100;
	const u64 high = l.limit * (u64)std::min(100, p.high_percent) / 100;
	// The charge one more block costs the arena: the caller's measurement when it
	// has one (see WindowPolicy::block_charge), the granularity rounding otherwise.
	const u64 per = p.block_charge
	                    ? p.block_charge
	                    : window_charge_bytes(1, l.block_bytes, l.chunk_bytes);
	// The caller hands the live driver figure, floored by the ledger so a driver
	// that has not caught up yet (the first step, before the window's pages are
	// touched) cannot read as "the card is empty" and grow into the ceiling.
	const u64 used = l.used;

	// Over the top. This is the safety half of the controller: the live figure
	// moved past the ceiling the window was planned against - another GPU client
	// claimed memory while we sampled, or the plan's own reserve was optimistic -
	// and every further allocation in the loop can only fail. Give back enough
	// blocks to bring the live figure back under the ceiling, one block buying back
	// its own chunk charge. It is deliberately not "the first time the driver
	// hiccups": the correction converges instead of thrashing.
	if (used > high && l.n > 0) {
		const u64 over = used - high;
		i64 drop = per ? (i64)((over + per - 1) / per) : l.n;
		if (drop < 1) drop = 1;
		const i64 cap = std::max<i64>(1, p.max_shrink_per_tune);
		if (drop > cap) drop = cap;
		if (drop > l.n) drop = l.n;
		if (delta) *delta = -drop;
		return WindowMove::Shrink;
	}

	// Under the floor: the card is idle enough that blocks which stream on every
	// step could stay instead. Grow by as many as the band still has room for
	// (capped by `max_grow_per_tune`), each one charged a whole chunk, and only
	// while that room survives the reserve the correction must not eat into.
	if (used < low && l.n < l.n_total) {
		// The room left under the band's top measured *from where the process is*:
		// `used`, not `other`. `other` is the ledger minus the window's charge, so
		// taking the band from it would hand the correction the room the window
		// already occupies - and then grow straight past the ceiling (a window that
		// reached 99 % of the limit with nothing left for the block the sampling
		// loop streams next is exactly this arithmetic). `used` already contains the
		// window, the activations, the resident tables and the device's own figure,
		// so `room` really is "how many more blocks fit". `safety_bytes` is then the
		// reserve the streamed block needs, which `used` cannot carry because the
		// loop hands the weight arena back before the tuner runs.
		const u64 band_room = high > used ? high - used : 0;
		const u64 room = band_room > p.safety_bytes ? band_room - p.safety_bytes : 0;
		if (per > 0 && per <= room) {
			i64 fits = (i64)(room / per);
			// Each block is charged whole chunks and the fit is rounded down, so this
			// is the count whose *total* charge the band can still cover; the loop
			// still verifies against the arena's real capacity before it uploads.
			const i64 remaining = l.n_total - l.n;
			if (fits > remaining) fits = remaining;
			const i64 cap = std::max<i64>(1, p.max_grow_per_tune);
			if (fits > cap) fits = cap;
			if (fits >= 1) {
				if (delta) *delta = fits;
				return WindowMove::Grow;
			}
		}
	}
	return WindowMove::Hold;
}

}  // namespace phi::media
