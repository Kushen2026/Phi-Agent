// ComfyUI schedulers, ported to the host.
//
// ComfyUI's `comfy/samplers.py` exposes nine schedulers through one table:
//
//     simple, normal, sgm_uniform, karras, exponential, ddim_uniform, beta,
//     linear_quadratic, kl_optimal
//
// Each turns the model's own `model_sampling` object into a descending sigma
// grid of `steps + 1` entries whose last entry is exactly 0. This header is the
// one place that mapping lives: the image, music and video chains all build their
// grid through `build_sigmas`, so a scheduler name means the same thing in every
// chain and there is exactly one implementation to compare against the Python.
//
// What a scheduler needs from the model is small and already abstracted by
// `SamplingFlow`:
//
//   * `sigma(t)` / `timestep(sigma)` - the model's noise-schedule axis;
//   * the sigma table (`model_sigma_table`) and therefore its endpoints
//     `sigma_min` / `sigma_max`, which several of the schedulers index directly.
//
// The two flow classes are not interchangeable here (see sampling.hpp): the
// DiscreteFlow table has 1000 entries and a multiplier, the Flux table 10000
// entries and none, and `normal` in particular reads `timestep(sigma_max)` -
// which is `sigma * multiplier` for one and `sigma` for the other. `build_sigmas`
// derives both the table length and the endpoints from `flow.kind`, so the caller
// only ever hands in the flow.
#pragma once

#include <string>
#include <vector>

#include "sampling/sampling.hpp"

namespace phi::media {

enum class SchedulerKind {
	Simple,           // simple_scheduler: the model's table, evenly sub-sampled
	Normal,           // linspace in timestep space
	SGMUniform,       // normal with the +1-then-drop linspace (sgm=True)
	Karras,           // karras sigmas, rho 7
	Exponential,      // exponential in log-sigma
	DDIMUniform,      // ddim_scheduler: the table, evenly sub-sampled with a 0 head
	Beta,             // beta(0.6, 0.6) quantiles on the table
	LinearQuadratic,  // linear head then quadratic tail
	KLOptimal,        // atan-spaced sigmas (kl_optimal)
};

// Parse a ComfyUI scheduler name. The canonical spelling is what
// `scheduler_names()` returns; a couple of well-known aliases
// ("ddim_uniform"/"ddim", "sgm_uniform"/"sgm") are accepted. Returns false for
// an unknown name (the caller keeps its own default rather than failing).
bool scheduler_from_name(const std::string& name, SchedulerKind* out);
// The canonical name ("simple", "karras", ...).
const char* scheduler_name(SchedulerKind k);
// Every scheduler, in the order a settings dropdown should show them.
std::vector<std::string> scheduler_names();

// The model's own `sigma_min` / `sigma_max`: the first and last entries of its
// sigma table. `SamplingFlow` already gives the axis; this is the two numbers
// the Karras / exponential / kl_optimal schedulers take instead of the table.
void model_sigma_bounds(const SamplingFlow& flow, float* sigma_min, float* sigma_max);

// Build the descending sigma grid: `steps + 1` entries, the last exactly 0.
//
// `denoise` reproduces `KSampler.set_steps`: > 0.9999 builds the full schedule;
// <= 0 returns empty; anything else builds `int(steps / denoise)` steps and keeps
// the tail `steps + 1` entries (a partial denoise is the tail of a longer
// schedule, not a rescaled one). This is the same trim the old `simple_scheduler`
// did internally, lifted here so every scheduler gets it.
std::vector<float> build_sigmas(SchedulerKind kind, const SamplingFlow& flow, int steps,
                                float denoise = 1.0f);

}  // namespace phi::media
