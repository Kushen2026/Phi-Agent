// ComfyUI samplers, ported to the host.
//
// The engine's sampling loops are host-side walks over small tensors (see
// sampling.hpp / sampling_h3.hpp): the DiT runs on the GPU, but the integrator is
// a handful of scalars and one elementwise pass per step, and keeping it on the
// host is what makes "the sigmas and the update agree with ComfyUI bit for bit"
// checkable. This header is the one generic integrator, so one sampler name means
// the same update in the image and music chains.
//
// The interface is ComfyUI's own, with the wrapping already done: ComfyUI's
// `KSAMPLER` hands the integrator a `model(x, sigma)` that returns the *denoised*
// estimate (x0), and k-diffusion's `to_d(x, sigma, denoised) = (x - denoised)/sigma`
// turns that back into a derivative. `DenoiseFn` below is exactly that
// `model(x, sigma)`: the caller does the DiT forward(s), combines them for CFG,
// and returns x0. The stochastic samplers draw their own noise from the unit
// field the latent was scaled from (`noise`), seeded by `seed`.
//
// Ported (the common half of ComfyUI's list - the exotic SDE/DEIS/sa_solver
// families are deliberately left out):
//
//   euler, euler_ancestral, heun, dpmpp_2m, dpmpp_2s_ancestral, dpmpp_2m_sde,
//   dpmpp_sde, res_multistep, lcm
//
// `res_multistep` is the one the image chain already used (sampling.hpp's
// `ResMultistep`) and the H3 joint sampler uses; exposing it here means the two
// paths share one implementation.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "util/media_common.hpp"

namespace phi::media {

// ComfyUI's `model(x, sigma)` -> x0 estimate (denoised). Must be deterministic
// for the deterministic samplers and must return a tensor the same size as `x`.
using DenoiseFn = std::function<std::vector<float>(const std::vector<float>& x, float sigma)>;

enum class SamplerKind {
	Euler,
	EulerAncestral,
	Heun,
	DPMpp2M,
	DPMpp2SAncestral,
	DPMpp2MSDE,
	DPMppSDE,
	ResMultistep,
	LCM,
};

// Parse a ComfyUI sampler name. Returns false for an unknown name / for the
// samplers this engine does not port, so a caller keeps its own default.
bool sampler_from_name(const std::string& name, SamplerKind* out);
const char* sampler_name(SamplerKind k);
// Every sampler, in the order a settings dropdown should show them.
std::vector<std::string> sampler_names();
// The name a chain offers by default (the released workflow's own sampler).
const char* default_sampler_name(const std::string& arch);
// True for a sampler that is stochastic: it needs the unit noise field and a
// seed. A caller that has no noise still gets a valid run (the stochastic term
// is skipped), but the result is the deterministic member of the family.
bool sampler_is_stochastic(SamplerKind k);

// Run `kind` over the sigma grid.
//
//   * `sigmas` is descending and `sigmas.back() == 0` (what `build_sigmas`
//     returns); `sigmas[0]` is the noise level `x` was drawn at.
//   * `x` is the initial latent (already scaled by `sigmas[0]`); the result is
//     the final latent.
//   * `noise` is the unit noise field `x` was scaled from, same size; used only
//     by the stochastic samplers.
//   * `seed` seeds those samplers' per-step draws (deterministic given a seed).
//   * `on_step(i, total)` is called once per step, before the model is asked for
//     it; it may throw to cancel (the caller's `MediaError` propagates).
//
// `sigmas`/`noise` may be empty/invalid only in ways the caller is expected to
// have rejected already; the function throws MediaError for a shape mismatch.
std::vector<float> run_sampler(SamplerKind kind, const std::vector<float>& sigmas,
                               std::vector<float> x, const DenoiseFn& denoise,
                               const std::vector<float>& noise, u64 seed,
                               const std::function<void(i64, i64)>& on_step);

}  // namespace phi::media
