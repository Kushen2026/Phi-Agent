#include "sampling/sampling.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

namespace phi::media {

// ── ModelSamplingDiscreteFlow ──────────────────────────────────────────────
//
// The Python does all of this in float64 when `t` is a Python float and in
// float32 when it is a tensor element. We keep the intermediate arithmetic in
// double and round once on the way out: that is the correctly-rounded float32
// result, which is what `float(t)` of a float32 tensor plus `torch.FloatTensor`
// storage gives back anyway.

float time_snr_shift(float alpha, float t) {
	if (alpha == 1.0f) return t;  // exact shortcut, see the header
	const double a = (double)alpha;
	const double td = (double)t;
	return (float)(a * td / (1.0 + (a - 1.0) * td));
}

float inverse_time_snr_shift(float alpha, float sigma) {
	if (alpha == 1.0f) return sigma;
	const double a = (double)alpha;
	const double s = (double)sigma;
	return (float)(s / (a - (a - 1.0) * s));
}

float flux_time_shift(float mu, float t) {
	if (!(t > 0.0f)) return 0.0f;
	const double m = std::exp((double)mu);
	const double td = (double)t;
	return (float)(m / (m + (1.0 / td - 1.0)));
}

float inverse_flux_time_shift(float mu, float sigma) {
	if (!(sigma > 0.0f)) return 0.0f;
	const double m = std::exp((double)mu);
	const double s = (double)sigma;
	return (float)(1.0 / (1.0 + m * (1.0 - s) / s));
}

float SamplingFlow::timestep(float sigma) const { return sigma * multiplier; }

float SamplingFlow::sigma(float timestep) const {
	const float t = (float)((double)timestep / (double)multiplier);
	return kind == Kind::Flux ? flux_time_shift(shift, t) : time_snr_shift(shift, t);
}

float SamplingFlow::inverse(float sigma) const {
	return kind == Kind::Flux ? inverse_flux_time_shift(shift, sigma)
	                          : inverse_time_snr_shift(shift, sigma);
}

std::vector<float> model_sigma_table(const SamplingFlow& flow, int timesteps) {
	if (timesteps <= 0) throw MediaError("model_sigma_table: timesteps must be > 0");
	std::vector<float> table((size_t)timesteps);
	// Python: self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier)
	for (int k = 1; k <= timesteps; k++) {
		const double t = ((double)k / (double)timesteps) * (double)flow.multiplier;
		table[(size_t)(k - 1)] = flow.sigma((float)t);
	}
	return table;
}

// ── simple_scheduler ───────────────────────────────────────────────────────
//
// comfy/samplers.py :: simple_scheduler — see the header for the four lines we
// are porting. Two things there are easy to get wrong:
//
//  1. `ss = len(s.sigmas) / steps` is a *float* division and the index is
//     `int(x * ss)`, i.e. truncated. For steps that divide the table length
//     (1000 / 8 = 125 exactly) the picks sit exactly on the table; for other
//     step counts they land one entry low or high of a true linspace, and that
//     off-by-one is part of the reference schedule. So we index the table the
//     same way instead of "simplifying" into a linspace over sigmas.
//
//  2. The table is built by ModelSamplingDiscreteFlow.set_parameters from the
//     *shifted* axis: entry k holds sigma(k / N * multiplier), k = 1..N. The
//     top of that grid is therefore t = multiplier (= 1.0 here), not
//     t = timestep(sigma_max) in general — `timestep` and `sigma` are not
//     inverses, which is the trap this module is meant to close. We recover the
//     grid top from sigma_max through the true inverse of time_snr_shift, and
//     the grid is anchored at t = 0 like the Python one.
//
// Consequence of (2) for the shift-3 workflow: t_max = 1, the grid is
// t = k/1000, and simple_scheduler picks k = 1000, 875, 750, ... 125, i.e.
// t = 1, 0.875, ... 0.125 -> sigma = 1.0, 0.9545454, 0.9, ... 0.3, then 0.0.
// Note that the bottom entries of the grid — including the floor
// sigma_min = sigma(1/1000) — are never read: the deepest pick is
// N - 1 - int((steps - 1) * ss) = 124 for 8 steps, and the trailing 0.0 is
// appended rather than taken from the table.
std::vector<float> simple_scheduler(float model_sigma_min, float model_sigma_max, int steps,
                                    float denoise, const SamplingFlow& flow) {
	if (steps <= 0) throw MediaError("simple_scheduler: steps must be > 0");
	if (!(model_sigma_min < model_sigma_max))
		throw MediaError("simple_scheduler: model_sigma_min must be < model_sigma_max");

	// KSampler.set_steps(steps, denoise):
	//     if denoise is None or denoise > 0.9999: self.sigmas = calculate_sigmas(steps)
	//     elif denoise <= 0.0:                    self.sigmas = empty
	//     else: new_steps = int(steps / denoise)
	//           self.sigmas = calculate_sigmas(new_steps)[-(steps + 1):]
	// The "partial denoise" case is not a scaled schedule, it is the tail of a
	// *longer* schedule — exactly like the Python.
	if (denoise <= 0.0f) return {};

	int build_steps = steps;
	bool trim = false;
	if (!(denoise > 0.9999f)) {
		build_steps = (int)((double)steps / (double)denoise);  // Python int() truncates
		if (build_steps <= 0) return {};
		trim = true;
	}

	const int N = flow.kind == SamplingFlow::Kind::Flux ? kFluxSigmaTableSteps : kSigmaTableSteps;
	// Grid top in shifted-axis units. sigma_max == sigma(multiplier), so this is
	// `multiplier` again; taking the inverse of sigma_max keeps the function
	// correct for callers that hand in a differently-derived endpoint.
	const double t_top = (double)flow.inverse(model_sigma_max);
	std::vector<float> table((size_t)N);  // ascending in sigma, like model_sampling.sigmas
	for (int k = 1; k <= N; k++) table[(size_t)(k - 1)] = flow.sigma((float)(t_top * (double)k / (double)N));

	const double ss = (double)N / (double)build_steps;
	std::vector<float> sigs;
	sigs.reserve((size_t)build_steps + 1);
	for (int x = 0; x < build_steps; x++) {
		int idx = N - 1 - (int)((double)x * ss);  // Python: s.sigmas[-(1 + int(x * ss))]
		if (idx < 0) idx = 0;                     // unreachable for x < build_steps
		sigs.push_back(table[(size_t)idx]);
	}
	sigs.push_back(0.0f);  // the terminal 0.0 is appended, never read from the table

	if (trim) {
		// [-(steps + 1):] — the tail of the longer schedule, not a rescale of it.
		const size_t keep = (size_t)steps + 1;
		if (sigs.size() > keep) sigs.erase(sigs.begin(), sigs.end() - (std::ptrdiff_t)keep);
	}
	return sigs;
}

// ── res_multistep ──────────────────────────────────────────────────────────
//
// comfy/k_diffusion/sampling.py, the non-cfg_pp body (which is the one
// `sample_res_multistep` runs):
//
//     for i in range(len(sigmas) - 1):
//         denoised = model(x, sigmas[i] * s_in, ...)
//         sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
//         if sigma_down == 0 or old_denoised is None:
//             d = to_d(x, sigmas[i], denoised); dt = sigma_down - sigmas[i]; x = x + d * dt
//         else:
//             t, t_old, t_next, t_prev = t_fn(sigmas[i]), t_fn(old_sigma_down), t_fn(sigma_down), t_fn(sigmas[i - 1])
//             h = t_next - t
//             c2 = (t_prev - t_old) / h
//             phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
//             b1 = nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
//             b2 = nan_to_num(phi2_val / c2, nan=0.0)
//             x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)
//         if sigma_up > 0: x = x + noise_sampler(...) * s_noise * sigma_up
//         old_denoised = denoised
//         old_sigma_down = sigma_down
//
// with sigma_fn(t) = exp(-t), t_fn(sigma) = -log(sigma), phi1(t) = expm1(t)/t,
// phi2(t) = (phi1(t) - 1)/t and to_d(x, sigma, denoised) = (x - denoised)/sigma.
//
// History handling — the places that are easy to get subtly wrong:
//
//  * `old_sigma_down` is the sigma_down of the PREVIOUS iteration, and with
//    eta = 0 (sample_res_multistep) sigma_down == sigmas[i + 1], so at
//    iteration i it is exactly sigmas[i] == t, and t_old == t. The lag that
//    really enters c2 is t_prev - t = t_fn(sigmas[i - 1]) - t_fn(sigmas[i]),
//    i.e. the previous *step size* in t space (negative, because t grows as
//    sigma shrinks).
//  * The first step has no lag (`old_denoised is None`) and the last step has
//    sigma_down == 0, so BOTH ends of the schedule take the Euler branch: the
//    first because the history is empty, the last because the update is the
//    exact `x -> denoised` collapse. 8 steps therefore execute 1 + 6 + 1.
//  * `sigma_up` is 0 for eta = 0, so no noise tensor is ever added; the
//    sampler is fully deterministic and this host port needs no RNG.
//  * `nan_to_num(..., nan=0.0)` leaves the inf case to torch's default, which
//    is the dtype max/min — reproduced below, because c2 == 0 (two steps of
//    equal length in t space) is reachable and phi2/c2 would be inf there.

namespace {

// torch.nan_to_num(x, nan=0.0): nan -> 0, +-inf -> +-FLT_MAX (torch's default
// for posinf/neginf is the largest/smallest finite value of the dtype).
inline float nan_to_num(float v) {
	if (std::isnan(v)) return 0.0f;
	if (std::isinf(v)) return v > 0.0f ? 3.4028234663852886e38f : -3.4028234663852886e38f;
	return v;
}

inline double t_fn(double sigma) { return -std::log(sigma); }

// phi1_fn / phi2_fn. The Python evaluates them with plain divisions, so
// phi1(0) is 0/0 = nan and phi2(0) = nan; that nan is what nan_to_num clears.
inline double phi1_fn(double t) {
	if (t == 0.0) return std::numeric_limits<double>::quiet_NaN();
	return std::expm1(t) / t;
}
inline double phi2_fn(double t) { return (phi1_fn(t) - 1.0) / t; }

}  // namespace

void ResMultistep::reset() {
	has_old_ = false;
	prev_sigma_ = 0.0f;
	old_sigma_down_ = 0.0f;
	old_denoised_.clear();
}

std::vector<float> ResMultistep::step(const std::vector<float>& x, float sigma, float sigma_next,
                                      const std::vector<float>& denoised) {
	if (denoised.size() != x.size())
		throw MediaError("res_multistep: denoised and x must have the same size");

	// get_ancestral_step(sigma, sigma_next, eta=0.) -> (sigma_next, 0.)
	const float sigma_down = sigma_next;
	// const float sigma_up = 0.0f;   // eta == 0, so the noise term is skipped

	std::vector<float> out(x.size());

	if (sigma_down == 0.0f || !has_old_) {
		// Euler branch: d = to_d(x, sigma, denoised) = (x - denoised) / sigma,
		// dt = sigma_down - sigma (both in sigma space, not t space).
		const float inv_sigma = 1.0f / sigma;
		const float dt = sigma_down - sigma;
		for (size_t i = 0; i < x.size(); i++) out[i] = x[i] + (x[i] - denoised[i]) * inv_sigma * dt;
	} else {
		const double t = t_fn((double)sigma);
		const double t_old = t_fn((double)old_sigma_down_);
		const double t_next = t_fn((double)sigma_down);
		const double t_prev = t_fn((double)prev_sigma_);
		const double h = t_next - t;
		const double c2 = (t_prev - t_old) / h;

		const float hf = (float)h;
		const float pf1 = (float)phi1_fn(-h);
		const float pf2 = (float)phi2_fn(-h);
		// The b1/b2 split is the only place the lag enters; b1 + b2 == phi1(-h)
		// for any c2, which is why a constant-denoiser test alone cannot catch a
		// wrong c2 — the gold test checks the coefficients directly instead.
		const float q = pf2 / (float)c2;
		const float b1 = nan_to_num(pf1 - q);
		const float b2 = nan_to_num(q);
		const float exp_neg_h = std::exp(-hf);  // sigma_fn(h)

		for (size_t i = 0; i < x.size(); i++)
			out[i] = exp_neg_h * x[i] + hf * (b1 * denoised[i] + b2 * old_denoised_[i]);
	}

	// State advance (unconditional, exactly like the Python loop tail).
	old_denoised_ = denoised;
	old_sigma_down_ = sigma_down;
	prev_sigma_ = sigma;
	has_old_ = true;
	return out;
}

}  // namespace phi::media
