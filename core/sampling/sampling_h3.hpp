// MiniMax H3 joint audio+video sampler (euler / beta, flow matching).
//
// A frozen interface: the video_generate chain is the only consumer.
//
// The video and audio latents are denoised *together*: one DiT evaluation
// produces both velocity fields, and the same sigma schedule drives both, but
// the audio branch is rescaled by `audio_scale = shift / audio_shift`. With the
// plan's shift 12 / audio_shift 3 that is exactly 4.0.
//
// This is the host-side latent walk, mirroring sampling.* for the image path:
// the arithmetic is a handful of scalars plus one elementwise pass per step, and
// "the sigmas must match the reference bit for bit" is far easier to guarantee
// on the host than in a GPU reduction.
#pragma once

#include <cstddef>
#include <vector>

#include "util/media_common.hpp"

namespace phi::media {

struct H3SamplingParams {
	i64 steps = 32;
	float shift = 12.0f;        // video flow shift
	float audio_shift = 3.0f;   // audio flow shift (audio_scale = shift/audio_shift)
	float alpha = 0.6f;         // beta scheduler shape
	float beta = 0.6f;
	bool use_beta = false;      // false = plain euler on the shifted grid
};

// shift / audio_shift, the factor applied to the audio sigma grid. Exposed so
// the tests and the model code agree to the last bit.
float h3_audio_scale(const H3SamplingParams& p);

// comfy/ldm/minimax/model.py::time_shift_sigma — invert the flow shift, then
// re-apply the other. Carries the audio latent onto the video schedule; the
// sampler's own state is the *carried* stream (z * sigma_v / sigma_a, which is
// 1x at sigma_v = 1), and what it divides back out at the end is
// h3_audio_scale() - see h3_joint_av.hpp. Added in Phase 0 after reading the
// reference (pure addition, no frozen line changed).
float h3_time_shift_sigma(float sigma, float from_shift, float to_shift);

// The descending sigma grid, `steps + 1` entries (last is 0). The video grid is
// time_snr_shift(shift, .); the audio grid is that rescaled by h3_audio_scale().
std::vector<float> h3_sigmas(const H3SamplingParams& p);
std::vector<float> h3_audio_sigmas(const H3SamplingParams& p);

// alpha/beta-shaped grid used when `use_beta` is set (alpha = beta = 0.6).
std::vector<float> h3_beta_sigmas(const H3SamplingParams& p);

// One flow-matching euler step: x[i] += (sigma_next - sigma) * v[i].
void h3_euler_step(float* x, const float* v, i64 n, float sigma, float sigma_next);

}  // namespace phi::media
