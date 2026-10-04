// Host-side sampler maths: the ComfyUI "simple" scheduler + res_multistep.
//
// This is the CPU half of the image pipeline. The DiT lives on the GPU but the
// latent walk (sigmas, x, denoised) is a handful of scalars and one elementwise
// pass per step, so it is cheaper and far easier to verify on the host — and it
// has to be bit-for-bit reproducible, which a GPU reduction is not.
//
// Everything below mirrors a specific ComfyUI revision:
//
//   comfy/samplers.py            simple_scheduler, KSampler.set_steps
//   comfy/model_sampling.py      ModelSamplingDiscreteFlow, time_snr_shift
//   comfy_extras/nodes_model_advanced.py
//                                ModelSamplingAuraFlow.patch_aura -> patch(shift, multiplier=1.0)
//   comfy/k_diffusion/sampling.py
//                                res_multistep
//
// The workflow it reproduces is
// ModelSamplingAuraFlow(shift = 3.0) + KSampler(sampler="res_multistep",
// scheduler="simple", steps=8, denoise=1.0).
#pragma once

#include <cstddef>
#include <vector>

#include "util/media_common.hpp"

namespace phi::media {

// ── ModelSamplingDiscreteFlow (+ the AuraFlow patch) ───────────────────────
//
// ModelSamplingAuraFlow is a subclass of ModelSamplingSD3 whose patch_aura()
// calls patch(model, shift, multiplier=1.0), so the workflow's model_sampling is
// exactly ModelSamplingDiscreteFlow(shift=3.0, multiplier=1.0):
//
//     def timestep(self, sigma):    return sigma * self.multiplier
//     def sigma(self, timestep):    return time_snr_shift(self.shift, timestep / self.multiplier)
//
// Note that these two are NOT inverses of each other. `sigma(timestep(s))` is
// `time_snr_shift(shift, s)`; only the pair (shift == 1) collapses them to the
// identity. The multiplier is a pure rescaling of the value handed to the DiT
// as its "timestep" input, while the KSampler's sigmas are the *unrescaled*
// time_snr_shift output. With multiplier == 1 (this workflow) sigma == timestep
// and the sigma axis *is* the model's flow interpolation parameter t
// (x_t = (1 - t) * x0 + t * noise).
struct SamplingFlow {
	// comfy/model_base.py::model_sampling picks the *class* from the model's
	// `model_type`. `ModelType.FLOW` (AuraFlow/SD3, the H3 chain)
	// gets ModelSamplingDiscreteFlow; `ModelType.FLUX` (Qwen-Image, and Qwen
	// Image 2.1 -> QwenImage21 -> ModelType.FLUX) gets ModelSamplingFlux. Both
	// take a `shift` in sampling_settings and both build a sigma table, but the
	// sigma *function* is different: `time_snr_shift(alpha, t)` against
	// `flux_time_shift(mu, t)`. Reading Qwen's `shift = 0.69` through the
	// discrete-flow formula (what this struct did before) walks a different,
	// much more aggressive schedule: 0.9430 where the checkpoint's own is
	// 0.9795 at the first step.
	enum class Kind { DiscreteFlow, Flux };

	Kind kind = Kind::DiscreteFlow;
	float shift = 3.0f;
	float multiplier = 1.0f;

	float timestep(float sigma) const;  // sigma -> model timestep (sigma * multiplier)
	float sigma(float timestep) const;  // forward: t -> sigma
	float inverse(float sigma) const;   // inverse: sigma -> t
};

// comfy/model_sampling.py:
//     def time_snr_shift(alpha, t):
//         if alpha == 1.0:
//             return t
//         return alpha * t / (1 + (alpha - 1) * t)
// The alpha == 1.0 shortcut is kept: it is exactly what makes multistep/aura
// flows with shift 1.0 reproduce the raw t instead of a rounded copy of it.
float time_snr_shift(float alpha, float t);

// The algebraic inverse of the above (t = s / (alpha - (alpha - 1) * s)). Used to
// place the discretisation grid of the model's sigma table and by the tests.
float inverse_time_snr_shift(float alpha, float sigma);

// comfy/model_sampling.py:
//     def flux_time_shift(mu, sigma=1.0, t):
//         return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
// ModelSamplingFlux.sigma(). Its inverse is 1 / (1 + e^mu (1 - s) / s).
float flux_time_shift(float mu, float t);
float inverse_flux_time_shift(float mu, float sigma);

// ModelSamplingDiscreteFlow.set_parameters() discretises the sigma axis into
// `timesteps` entries (its own default is 1000). simple_scheduler below indexes
// that table, so the number matters and is part of the reference behaviour.
inline constexpr int kSigmaTableSteps = 1000;

// ModelSamplingFlux registers its table over t = k / timesteps with its own
// default of 10000 entries. Both are uniform in t, so a sigma table is
// equivalent to a linear t grid either way; the count only shows up in
// simple_scheduler's `int(x * len/steps)` truncation, which lands on the same
// t for any count that is a multiple of the step count.
inline constexpr int kFluxSigmaTableSteps = 10000;

// The `sigmas` buffer ModelSamplingDiscreteFlow registers, ascending:
//     sigma(k / timesteps * multiplier), k = 1 .. timesteps
std::vector<float> model_sigma_table(const SamplingFlow& flow, int timesteps = kSigmaTableSteps);

// ComfyUI's simple scheduler, as selected by KSampler(scheduler="simple").
//
// comfy/samplers.py:
//     def simple_scheduler(model_sampling, steps):
//         s = model_sampling
//         sigs = []
//         ss = len(s.sigmas) / steps
//         for x in range(steps):
//             sigs += [float(s.sigmas[-(1 + int(x * ss))])]
//         sigs += [0.0]
//         return torch.FloatTensor(sigs)
//
// `model_sigma_min` / `model_sigma_max` are that buffer's first/last entry; the
// returned vector has `steps + 1` entries, sigmas[0] == sigma_max .. sigmas[steps] == 0.
// `denoise` mirrors KSampler.set_steps (see the .cpp).
std::vector<float> simple_scheduler(float model_sigma_min, float model_sigma_max, int steps,
                                    float denoise, const SamplingFlow& flow = SamplingFlow{});

// ── res_multistep ─────────────────────────────────────────────────────────
//
// comfy/k_diffusion/sampling.py :: res_multistep (via sample_res_multistep,
// which passes eta = 0). The sampler is NOT plain Euler: from the second step
// on it evaluates the phi1/phi2 exponential-integrator coefficients with a
// lagged (t_prev - t_old) term, i.e. the second order multistep method of
// https://arxiv.org/pdf/2308.02157. Only the first step — and any step whose
// sigma_down is 0 — degenerates to Euler.
//
// Cross-step state (all of it): the previous step's denoised tensor, the
// previous step's sigma_down, and the sigma the previous step started from.
class ResMultistep {
public:
	void reset();

	// One step. `x` and `denoised` are flat row-major tensors of the same size
	// (a 4-D image latent: [C,H,W] for batch 1). Returns the new x, i.e. what
	// ComfyUI assigns to `x` at the end of iteration i, and advances the state.
	std::vector<float> step(const std::vector<float>& x, float sigma, float sigma_next,
	                        const std::vector<float>& denoised);

private:
	bool has_old_ = false;
	float prev_sigma_ = 0.0f;       // sigmas[i - 1] of the current iteration
	float old_sigma_down_ = 0.0f;   // sigma_down of the previous iteration
	std::vector<float> old_denoised_;
};

}  // namespace phi::media
