// Minimal fork/join for the fp32 audio kernels (W3).
//
// The H3 audio path runs on the host: BigVGAN's 21 resblocks are ~200 GFLOP per
// 10 s of audio at 32 kHz, which is 20-60 s on a single core and ~3 s on this
// box's 16. The reference evaluates the whole thing in fp32 (the audio VAE
// checkpoint is fp32 throughout, and the SnakeBeta exponents plus the
// anti-aliasing filter coefficients make fp16 unacceptable against the 1e-4
// rel-L2 budget), so the CPU is where it belongs.
//
// Tasks always split the *output* space: each output element is produced by
// exactly one task, with the accumulation order the serial loop would use. The
// result is therefore bit-identical to a serial run, whatever the thread count
// or the scheduling — which matters, because otherwise "the test passed" would
// depend on the machine.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>

#include "util/media_common.hpp"

namespace phi::media::kernels {

// Number of worker threads. PHI_AUDIO_THREADS overrides it (tests pin it to 1
// to prove the parallel and serial paths agree bit for bit).
inline unsigned thread_count() {
	static unsigned n = [] {
		if (const char* s = std::getenv("PHI_AUDIO_THREADS")) {
			int v = std::atoi(s);
			if (v > 0) return (unsigned)v;
		}
		unsigned c = std::thread::hardware_concurrency();
		return c ? c : 1u;
	}();
	return n;
}

// fn(i) for i in [0, n). fn must not throw (a throwing task would std::terminate
// the process from the worker threads).
template <typename Fn>
void parallel_for(i64 n, Fn&& fn, unsigned threads = 0) {
	if (n <= 0) return;
	if (threads == 0) threads = thread_count();
	const unsigned t = (unsigned)std::min<i64>(n, (i64)threads);
	if (t <= 1) {
		for (i64 i = 0; i < n; i++) fn(i);
		return;
	}
	std::atomic<i64> next{0};
	std::vector<std::thread> pool;
	pool.reserve(t - 1);
	auto worker = [&] {
		for (;;) {
			const i64 i = next.fetch_add(1, std::memory_order_relaxed);
			if (i >= n) break;
			fn(i);
		}
	};
	for (unsigned k = 0; k + 1 < t; k++) pool.emplace_back(worker);
	worker();
	for (auto& th : pool) th.join();
}

}  // namespace phi::media::kernels
