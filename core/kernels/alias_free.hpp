// Anti-aliased resampling for the H3 audio VAE / BigVGAN (W3).
//
// Ported from comfy/ldm/minimax/audio_vae.py:
//
//   kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
//   UpSample1d(ratio, kernel_size)        -> replicate pad, conv_transpose1d, * ratio, crop
//   LowPassFilter1d(cutoff, half_width, stride, kernel_size)
//   DownSample1d(ratio, kernel_size)      -> LowPassFilter1d(0.5/ratio, 0.6/ratio, ratio)
//   Activation1d(act, 2, 2, 12, 12)       -> up x2, act, down x2
//
// This is the numerically delicate part of the vocoder: every filter is 12 taps
// of a Kaiser-windowed sinc whose coefficients are stored *in the checkpoint*
// (`*.upsample.filter` / `*.downsample.lowpass.filter`). The kernels here load
// those coefficients (see `filter()`); kaiser_sinc_filter1d() reproduces them
// from the reference formula to 2.6e-8 max abs (fp32 storage rounding), which is
// what the conformance test asserts. `set_filter()` accepts the checkpoint's own
// 12 floats so the runtime never depends on that agreement.
//
// Layout: [C, L] row major; a stereo channel is a separate call. The 1-D padding
// is torch's `mode="replicate"` (edge value).
#pragma once

#include <vector>

#include "util/media_common.hpp"
#include "kernels/snake.hpp"

namespace phi::media::kernels {

// Reference formula. Evaluated in double and stored fp32: the reference's own
// coefficients (checkpoint) match that to one fp32 ulp.
std::vector<float> kaiser_sinc_filter1d(float cutoff, float half_width, i64 kernel_size);

// ── UpSample1d ─────────────────────────────────────────────────────────────

struct Upsample1d {
	i64 ratio = 2;
	i64 kernel_size = 12;
	i64 pad = 0;        // F.pad(x, (pad, pad)) on both sides
	i64 pad_left = 0;   // crop after the transposed conv
	i64 pad_right = 0;
	std::vector<float> filter;   // [kernel_size]

	// Derives pad/pad_left/pad_right exactly as the reference __init__ does and
	// fills the filter from the formula.
	void init(i64 ratio_, i64 kernel_size_);
	// Swap in the checkpoint's coefficients (must be kernel_size long).
	void set_filter(const float* f, i64 n);
	const float* filter_data() const { return filter.data(); }

	i64 out_len(i64 L) const;
	// y: [C, out_len(L)]; x may not alias y.
	void forward(const float* x, i64 C, i64 L, float* y) const;
};

// ── LowPassFilter1d / DownSample1d ─────────────────────────────────────────

struct LowPassFilter1d {
	float cutoff = 0.5f, half_width = 0.6f;
	i64 stride = 1;
	i64 kernel_size = 12;
	i64 pad_left = 0, pad_right = 0;
	std::vector<float> filter;

	void init(float cutoff_, float half_width_, i64 stride_, i64 kernel_size_);
	void set_filter(const float* f, i64 n);
	const float* filter_data() const { return filter.data(); }

	i64 out_len(i64 L) const;
	void forward(const float* x, i64 C, i64 L, float* y) const;
};

struct DownSample1d {
	i64 ratio = 2;
	i64 kernel_size = 12;
	LowPassFilter1d lowpass;

	void init(i64 ratio_, i64 kernel_size_) {
		ratio = ratio_;
		kernel_size = kernel_size_;
		lowpass.init(0.5f / (float)ratio_, 0.6f / (float)ratio_, ratio_, kernel_size_);
	}
	void set_filter(const float* f, i64 n) { lowpass.set_filter(f, n); }
	const float* filter_data() const { return lowpass.filter_data(); }
	i64 out_len(i64 L) const { return lowpass.out_len(L); }
	void forward(const float* x, i64 C, i64 L, float* y) const { lowpass.forward(x, C, L, y); }
};

// ── Activation1d ───────────────────────────────────────────────────────────

struct Activation1d {
	Upsample1d up;
	DownSample1d down;
	SnakeBetaAct snake;

	void init(i64 channels) {
		(void)channels;
		up.init(2, 12);
		down.init(2, 12);
	}
	// The reference always uses 2/2 with kernel 12 on both sides, so the output
	// length equals the input length.
	i64 out_len(i64 L) const { return down.out_len(up.out_len(L)); }
	// `scratch` must hold scratch_len(C, L) floats (the x2 intermediate).
	static i64 scratch_len(i64 C, i64 L) { return 2 * C * L; }
	void forward(const float* x, i64 C, i64 L, float* y, float* scratch) const;
};

}  // namespace phi::media::kernels
