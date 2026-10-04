// The resident weight window: how many DiT blocks stay on the card for the whole
// sampling loop, and the band both media tools are held to.
//
// A text-to-image request and a 540P video request plan the *same* quantity in
// the same way - how many of the DiT's blocks fit beside everything else the step
// already holds - but they answer it from different inputs: the image chain from
// the canvas and the reference latents, the video chain from the resolution, the
// clip length, the reference rows and their VAE footprint. What this header adds
// is the piece that was duplicated (and drifted) between them:
//
//   * the *band* the sampling phase has to sit in. The requirement is that a
//     generation spend 90-98 % of the driver's limit for the process while it
//     samples. Below 90 % the card is idle and the run is paying for it in disk
//     traffic (every non-resident block is re-read on every step); at 98 % the
//     window still leaves room for the next streamed block, while spending more
//     would put the process on the driver's edge - which on this hardware is a
//     lost device, not a slowdown.
//   * the *arithmetic* of "how many blocks fit", priced the way the accountant
//     books it: whole arena chunks, not tensor bytes.
//   * the *runtime correction*. The plan is computed once, from estimates and
//     from the card as it looked before the run; the sampling loop then holds
//     that window for minutes while the driver's own figure moves (the WebView2
//     UI repainting, the desktop compositor, another engine phase handing its
//     arenas back). `tune_window` is the once-per-step decision that grows or
//     sheds the window to put the process back inside the band.
//
// It is deliberately a plain function of plain numbers: the residency decision is
// the one thing the two chains must not disagree about, and a host test can pin
// it without a device (tests/media_residency_test.cpp).
#pragma once

#include "util/media_common.hpp"

namespace phi::media {

// The band, as percentages of `VramBudget::limit()` (which is itself
// min(96 % x the driver's budget for this process, the ceiling the card
// supports)). Both ends are inclusive. The high end is 98, not 100: the last
// two percent are what the streamed block the sampling loop asks for *next*
// lands in, and a window that spends the whole limit is a window that refuses
// its own fallback path on the roundings (the failure mode `block_charge` below
// describes). At 98 the process still pays the disk for at most a couple of
// blocks while staying clear of the driver's edge.
inline constexpr int kWindowBandLowPercent = 90;
inline constexpr int kWindowBandHighPercent = 98;

// What an arena charges for a window of `n` blocks: the charge is per chunk, so
// `n` blocks of `block_bytes` are booked as whole `chunk_bytes` chunks. This is
// the same rule `GpuArena::alloc` applies and the one both planners round with.
u64 window_charge_bytes(i64 n, u64 block_bytes, u64 chunk_bytes);

struct WindowPlan {
	i64 n = 0;         // blocks to keep resident
	u64 window = 0;    // window_charge_bytes(n, ...)
	u64 usage = 0;     // other + window: what the process is planned to hold
	int percent = 0;   // usage as a percentage of the limit
	bool fills = false;   // n is the largest count that fits under the limit
	bool in_band = false; // percent >= low and <= high
};

// The largest window that still fits under `limit` beside `other` bytes of
// everything else the loop holds (activations, the streamed block's chunks, the
// token refiner, the caller's own footprint). Never more than `n_total`.
//
// `in_band` is answered with the *same* n: a window that cannot reach the band
// because one more block would leave the budget is reported as such rather than
// rounded up into a refusal.
WindowPlan plan_window_layers(u64 limit, u64 other, u64 block_bytes, u64 chunk_bytes,
                              i64 n_total);

// ── the request's own pixel weight ─────────────────────────────────────────
//
// The activation estimate a chain plans with is linear in its packed row count
// `S`, and `S` already carries the geometry - one video token per 2x2 latent
// patch (a 32x32-pixel cell), one reference row per packed patch or audio
// frame. Sizing the window from that alone is not enough, because the sampling
// loop also carries a set of *per-pixel* buffers that do not scale with `S`:
// the patch-projection staging rows, the gather/scatter rows the reference
// blocks and the output heads walk, and the VAE encode arenas that are still
// settling when the plan runs. Each is small per pixel, but they multiply by the
// whole request, and the term the plan is missing is exactly the one that grows
// with a long clip or a big reference image re-injected on every step - which is
// the request whose sampling then crosses the driver's ceiling and is refused
// mid-run rather than planned short.
//
// So a request is priced twice: once by its rows (`S`, the estimate above) and
// once by its **total pixel budget** - the generated frame area times the clip's
// length, plus every reference block's own area (a reference image is scaled to
// the generation's area, a reference video to its own canvas). The second charge
// is `kRequestPixelWeightBytes` per pixel; at a 2x2-patch token of 1024 pixels
// the row estimate already books ~63 bytes a pixel, so this weight adds ~13 % on
// top for the per-pixel buffers. It is deliberately a *reserve*, not a second
// estimate: the residency decision subtracts it before dividing by the block
// size, so a heavier request keeps fewer blocks - the same direction the request
// geometry already moves the window, just further.
inline constexpr u64 kRequestPixelWeightBytes = 8;

// The reserve, in bytes, the request's own pixels contribute on top of the row
// estimate: `video_pixels` (frame area x length) plus `ref_pixels` (every
// reference block's area), times the weight above. Saturates rather than wraps.
// Pure, so `tests/media_residency_test.cpp` can pin it with no device.
u64 request_pixel_reserve_bytes(u64 video_pixels, u64 ref_pixels);

enum class WindowMove { Hold, Grow, Shrink };

struct WindowPolicy {
	// The band. Kept as a struct rather than globals so a test can pin the edges
	// without touching the environment. 90-98 (see the header): the top 2 % are the
	// room the loop's next streamed block needs, not idle margin.
	int low_percent = kWindowBandLowPercent;
	int high_percent = kWindowBandHighPercent;
	// How many blocks one correction may move. A correction costs a checkpoint read
	// per block it adds, so it is bounded - but not to one: after the first step the
	// window can be far below the band (a fresh run, another GPU client that just
	// released memory, a card whose driver budget grew), and reaching the 90-100 %
	// band one block per step leaves the card idle - and re-reads the rest of the
	// stack off disk - for the whole climb. Growing by as much as the band actually
	// has room for (capped here) reaches the band on the first step that can, which
	// is the difference between an 8 s step and a 19 s one on the reference card.
	i64 max_grow_per_tune = 4;
	i64 max_shrink_per_tune = 1;
	// The charge a correction is not allowed to eat into: the *next* streamed
	// block has to be able to land, or the correction turns the run's own
	// fallback path into a refusal. One chunk is the granularity the accountant
	// actually refuses on.
	u64 safety_bytes = 0;
	// The charge one *more* resident block really costs the arena, measured rather
	// than derived from its tensor bytes. `window_charge_bytes(1, ...)` rounds a
	// single block in isolation, but a block's tensors do not tile the arena's
	// chunks, so the charge the accountant books for the (n+1)-th block is larger -
	// 315 MB against a 288 MB block on the reference card's 64 MB granularity. A
	// controller that grows by the derived figure overshoots the band by exactly
	// that difference, and the overshoot is how a window ended at 99 % of the limit
	// with nothing left for the block the sampling loop streams next. 0 keeps the
	// derived figure, which is what a caller with nothing to measure (and the host
	// tests) gets.
	u64 block_charge = 0;
};

struct WindowLive {
	i64 n = 0;           // blocks resident now
	i64 n_total = 0;     // blocks in the stack
	u64 other = 0;       // resident besides the window (ledger minus the window)
	u64 block_bytes = 0; // one block's tensor bytes
	u64 chunk_bytes = 0; // the resident arena's granularity
	u64 limit = 0;       // VramBudget::limit() (0 = no device answered: hold)
	// The live figure the band is defined on: what the driver attributes to this
	// process, never *less* than the ledger (see `tune_window`).
	u64 used = 0;
};

// One correction, from the numbers as they are *now*. `*delta` is signed and is
// only meaningful for Grow/Shrink. Pure: it decides, the caller performs the
// upload or the shed.
WindowMove tune_window(const WindowLive& live, const WindowPolicy& p, i64* delta);

}  // namespace phi::media
