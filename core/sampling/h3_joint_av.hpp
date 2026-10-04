// MiniMax H3 joint A/V carry transforms — reference-faithful.
//
// NOT FROZEN (new file, added by W8). `core/sampling/sampling_h3.hpp` is a frozen
// header and is *not* touched; the symbols below live here instead.
//
// The audio stream is carried on the *video* flow grid. ComfyUI's reference does
// this in three places, and this header mirrors them in the reference's own
// order — no "equivalent rewrite", because the float32 evaluation order matters
// for the last bit:
//
//   1. comfy/model_base.py::MiniMaxH3._scale_audio_slice
//        process_latent_in : x_a_sampler = audio_scale * x_a_latent
//        process_latent_out: x_a_latent  = x_a_sampler / audio_scale
//      with audio_scale = ModelSamplingAV.audio_scale = shift / audio_shift
//      (= 12 / 3 = 4.0 for the reference workflow).
//
//      Only the *out* half applies to a text-to-video run: ComfyUI's sampler
//      skips process_latent_in when the latent it was handed is empty
//      (`if latent_image is not None and torch.count_nonzero(latent_image) > 0`
//      in CFGGuider.inner_sample), and t2v hands it the zero latent, so the
//      sampler's audio variable starts at the plain unit noise like the video
//      one. The 4x is not a property of the sampler variable either - it is the
//      ratio sigma_v/sigma_a, 1 at sigma_v = 1 and 4 at sigma_v = 0 - so it
//      belongs to the carry (2) and to the closing process_latent_out, nowhere
//      else. Scaling the initial noise here put the audio stream 4x out of
//      distribution from step 0, and the first generation that did it came back
//      with distorted audio; the streams are both unit-scaled at step 0 and only
//      the carry divides them.
//
//   2. comfy/ldm/minimax/model.py::MiniMaxH3Model.forward (the carry):
//        sigma_v = clamp(timestep / 1000, min = 1e-6)
//        sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
//        carry   = sigma_a / sigma_v
//        x_net_a = x_a_sampler * carry            # what the DiT actually sees
//      (timestep is the flow timestep; with our interface `t` is already the
//      flow sigma in [0, 1], so sigma_v == clamp(t, 1e-6).)
//
//   3. the same forward, unscaling the velocity on the way out:
//        out_a = (1 - scale) * (x_a_sampler * carry)
//              + (1 + (scale - 1) * sigma_a) * net_out_a
//      `out_a` is d(x_a_sampler) / d(sigma_v), i.e. the velocity the sampler
//      multiplies by (sigma_next - sigma) on the *video* grid.
//
// Consequence: the sampler walks x_v and x_a_sampler with the *same* sigma grid
// (the video grid). There is no separate per-stream audio sigma grid in the
// reference; `H3JointSamplerV2` below implements exactly that walk, with both
// streams on the one grid.
//
// The walk itself is `res_multistep` (comfy/k_diffusion/sampling.py, the body
// `sample_res_multistep` runs with eta = 0): the same multistep integrator the
// image path uses, on the beta grid. It is not plain Euler - from the second
// step on it carries the previous step's denoised tensor and the lag
// (t_prev - t_old) into the phi1/phi2 coefficients, which is what makes it
// second order. Both streams share one grid, so the per-element recursion is
// run twice (video, audio) with the same sigma pair and the two stay in step.
// `sampling.hpp`'s `ResMultistep` is the same implementation the image chain
// calls, so the two paths cannot drift apart.
#pragma once

#include <vector>

#include "util/media_common.hpp"
#include "sampling/sampling.hpp"
#include "sampling/sampling_h3.hpp"

namespace phi::media {

// sigma_a / sigma_v, as computed in model.py::forward. `sigma_v` is the flow
// sigma (noise level) in [0, 1]; shift_v/shift_a are 12 / 3 for the workflow.
struct H3AVCarry {
	float sigma_v = 0.0f;   // clamped, >= 1e-6
	float sigma_a = 0.0f;   // time_shift_sigma(sigma_v, shift_v, shift_a)
	float carry = 1.0f;     // sigma_a / sigma_v
};

H3AVCarry h3_av_carry(float sigma_v, float shift_v, float shift_a);

// Reference step 1: `_scale_audio_slice`. `scale` is audio_scale
// (h3_audio_scale(p)); a scale of exactly 1 is a no-op, mirroring the early
// return. Used on the *output* side (process_latent_out, scale = 1/audio_scale),
// which is what video_gen calls before the audio VAE; the input side does not
// apply to an empty latent (see the note at the top of this file).
void h3_av_scale_audio(float* x_a, i64 n, float scale);

// Reference step 2: what the DiT sees given the sampler-space audio latent.
//   x_net[k] = x_a_sampler[k] * carry
void h3_av_audio_to_dit(const float* x_a_sampler, float* x_net, i64 n, float carry);

// Reference step 3: turn the DiT's raw audio velocity into the sampler-space
// velocity (see (3) above). A scale of exactly 1 copies `net_out` through.
void h3_av_audio_from_dit(const float* net_out, const float* x_a_sampler, float* out_sampler,
                          i64 n, float carry, float sigma_a, float scale);

// Reference-faithful joint walk: both streams advance on the video sigma grid
// with the same res_multistep update. `step` expects the *unscaled* audio
// velocity produced by h3_av_audio_from_dit(), i.e. the sampler-space address
// of the stream, and `init` takes both latents as the sampler holds them (unit
// noise times sigmas[0] at the start of a run). Dividing the finished audio
// latent by h3_av_scale_audio()'s factor is the caller's job (that is
// process_latent_out) - not scaling the noise on the way in.
class H3JointSamplerV2 {
public:
	explicit H3JointSamplerV2(const H3SamplingParams& p) : p_(p) {}

	// The integrator. The reference workflow runs res_multistep, which is what
	// `true` selects; `false` is the plain Euler update `x += v * (s1 - s0)` on
	// both streams - the same schedule, one model evaluation per step instead of
	// the multistep carry. Exposed so the tool's sampler setting is real for the
	// video chain too (the image/music chains choose through `run_sampler`).
	void set_multistep(bool on) { multistep_ = on; }
	bool multistep() const { return multistep_; }

	// `sigmas`, when non-empty, is the descending grid the caller already built
	// (the SCHEDULER node's, so the sigma fed to the DiT and the sigma integrated
	// here are the same numbers). Empty keeps this class's own derivation from
	// `p_`, which is the beta/simple default the released video workflow uses.
	void init(std::vector<float> x_v, std::vector<float> x_a_sampler,
	          std::vector<float> sigmas = {});

	i64 steps() const { return p_.steps; }
	const std::vector<float>& sigmas() const { return sigmas_; }   // video grid
	float audio_scale() const { return h3_audio_scale(p_); }

	// Advance one step. `v_*` are the DiT's velocity fields; the sampler turns
	// them into the denoised estimate itself (ModelSamplingDiscreteFlow:
	// x0 = x - sigma * v, which is exactly what res_multistep's `denoised` is)
	// so neither caller nor cache has to know the integrator.
	void step(i64 i, const std::vector<float>& v_v, const std::vector<float>& v_a_sampler);

	const std::vector<float>& x_v() const { return x_v_; }
	const std::vector<float>& x_a() const { return x_a_; }   // sampler-space audio

private:
	H3SamplingParams p_;
	std::vector<float> sigmas_, x_v_, x_a_;
	bool multistep_ = true;
	// One res_multistep state per stream (same sigma grid, so they move together).
	ResMultistep rm_v_, rm_a_;
	std::vector<float> den_v_, den_a_;   // this step's x0 estimate, per stream
};

}  // namespace phi::media
