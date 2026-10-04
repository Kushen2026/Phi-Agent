// ComfyUI samplers — the integrators behind one `KSAMPLER` node.
//
// The interface is ComfyUI's own (see samplers.hpp): the caller's `denoise(x,
// sigma)` returns the *denoised* estimate x0 — exactly what ComfyUI's model
// wrapper hands k-diffusion — and `to_d(x, sigma, denoised) = (x - denoised) /
// sigma` turns that back into the ODE derivative each update consumes.
//
// Ported bodies (comfy/k_diffusion/sampling.py, at the revision quoted by the
// task):
//
//   sample_euler, sample_euler_ancestral, sample_heun, sample_dpmpp_2m,
//   sample_dpmpp_2s_ancestral, sample_dpmpp_2m_sde, sample_dpmpp_sde,
//   sample_lcm  (+ ResMultistep from sampling.hpp for sample_res_multistep)
//
// Conventions and traps:
//
//  * The sigma grid is descending and ends at 0; the loop runs over
//    `i in [0, sigmas.size() - 1)` with `sigma = sigmas[i]`, `sigma_next =
//    sigmas[i+1]`. The last iteration is the collapse to denoised.
//  * The newer sampler bodies spell the log-sigma (VP) change of variables
//    through `sigma_to_half_log_snr` / `half_log_snr_to_sigma`. For every model
//    this engine carries (ModelSamplingDiscreteFlow / ModelSamplingFlux, neither
//    of which is a CONST) those are exactly `-log(sigma)` and `exp(-t)`, and
//    `offset_first_sigma_for_snr` is the identity — so the classic k-diffusion
//    t_fn/sigma_fn form is the same float32 arithmetic, not a simplification.
//  * Scalar intermediates are evaluated in double and rounded to float once
//    (the project's standing policy, see sampling.cpp); the elementwise passes
//    stay in float, in the reference's own operation order. The `alpha_*`
//    prefactors of the SDE solvers are kept even though they are algebraically 1
//    for log-sigma models, because the reference computes them explicitly and
//    "equivalent" simplifications change the last bit.
//  * RNG: ComfyUI's default_noise_sampler draws a fresh `torch.randn` per call
//    from a torch.Generator seeded with the run seed. The engine has no torch
//    RNG, so each draw is answered from a std::mt19937_64 reseeded from
//    (seed, step) — the one documented deviation. The noise field handed to
//    run_sampler is *not* consumed by the stochastic terms, exactly as in
//    ComfyUI (whose noise sampler ignores the latent's own noise too); it is
//    only shape-checked.
#include "sampling/samplers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <utility>

#include "sampling/sampling.hpp"

namespace phi::media {

namespace {

// ── k-diffusion's change of variables ──────────────────────────────────────
//
// t_fn(sigma) = -log(sigma), sigma_fn(t) = exp(-t). See the file header: for the
// flow models this engine carries these are exactly ComfyUI's
// sigma_to_half_log_snr / half_log_snr_to_sigma.
double t_fn(double sigma) { return -std::log(sigma); }
double sigma_fn(double t) { return std::exp(-t); }

// get_ancestral_step(sigma_from, sigma_to, eta): the sigma to step down to and
// the amount of noise to add. `eta == 0` (or sigma_to == 0) is the
// deterministic step: sigma_up = 0, sigma_down = sigma_to.
void ancestral_step(float sigma_from, float sigma_to, double eta, float* sigma_down, float* sigma_up) {
	const double from = (double)sigma_from, to = (double)sigma_to;
	if (eta == 0.0 || to == 0.0) {
		*sigma_down = sigma_to;
		*sigma_up = 0.0f;
		return;
	}
	const double up = eta * std::sqrt(to * to * (from * from - to * to) / (from * from));
	const float su = (float)std::min(to, up);   // min(sigma_to, ...)
	*sigma_up = su;
	*sigma_down = (float)std::sqrt(to * to - (double)su * (double)su);
}

// ── the per-step gaussian draw ─────────────────────────────────────────────
//
// ComfyUI's default_noise_sampler returns `torch.randn(x.size())` from a
// generator seeded with the run seed; every call is a fresh draw. torch's RNG
// cannot be reproduced here, so a step's draw is a std::normal_distribution over
// a std::mt19937_64 seeded from (seed, step) — documented deviation. `sub`
// separates the two draws sample_dpmpp_sde makes inside one step; every other
// sampler uses 0. (0x9E3779B97F4A7C15 / 0xBF58476D1CE4E5B9 are the golden-ratio
// and splitmix64 constants, so neighbouring steps do not share a stream.)
std::vector<float> step_noise(u64 seed, i64 step, int sub, size_t n) {
	std::mt19937_64 rng(seed + (u64)step * 0x9E3779B97F4A7C15ULL + (u64)sub * 0xBF58476D1CE4E5B9ULL);
	std::normal_distribution<float> normal(0.0f, 1.0f);
	std::vector<float> out(n);
	for (size_t i = 0; i < n; i++) out[i] = normal(rng);
	return out;
}

// `model(x, sigma)`, with the shape check every sampler needs: a denoiser that
// returns the wrong number of floats would otherwise silently corrupt the walk.
std::vector<float> model_x0(const DenoiseFn& denoise, const std::vector<float>& x, float sigma,
                            size_t n) {
	if (!denoise) throw MediaError("run_sampler: the denoise callback is empty");
	std::vector<float> den = denoise(x, sigma);
	if (den.size() != n)
		throw MediaError("run_sampler: denoise() returned " + std::to_string(den.size()) +
		                 " floats, expected " + std::to_string(n));
	return den;
}

}  // namespace

// ── euler ──────────────────────────────────────────────────────────────────
//
// sample_euler with s_churn = 0, so sigma_hat == sigmas[i]:
//     d = to_d(x, sigma, denoised); x = x + d * (sigmas[i + 1] - sigma)
// With `denoised = x - sigma * v` this is exactly the image/music chain's
// `x += v * (sigma_next - sigma)`.
static std::vector<float> run_euler(const std::vector<float>& sigmas, std::vector<float> x,
                                    const DenoiseFn& denoise, u64, i64 total,
                                    const std::function<void(i64, i64)>& on_step) {
	const size_t n = x.size();
	for (i64 i = 0; i < total; i++) {
		if (on_step) on_step(i, total);
		const float sigma = sigmas[(size_t)i];
		const float sigma_next = sigmas[(size_t)(i + 1)];
		const std::vector<float> den = model_x0(denoise, x, sigma, n);
		const float dt = sigma_next - sigma;
		for (size_t j = 0; j < n; j++) {
			const float d = (x[j] - den[j]) / sigma;
			x[j] = x[j] + d * dt;
		}
	}
	return x;
}

// ── euler_ancestral (eta = 1) ──────────────────────────────────────────────
//
// sample_euler_ancestral. The last step (sigma_next == 0 -> sigma_down == 0)
// collapses to x = denoised; every other step is Euler to sigma_down plus
// `noise * s_noise * sigma_up`.
static std::vector<float> run_euler_ancestral(const std::vector<float>& sigmas, std::vector<float> x,
                                              const DenoiseFn& denoise, u64 seed, i64 total,
                                              const std::function<void(i64, i64)>& on_step) {
	const size_t n = x.size();
	constexpr float s_noise = 1.0f;
	for (i64 i = 0; i < total; i++) {
		if (on_step) on_step(i, total);
		const float sigma = sigmas[(size_t)i];
		const float sigma_next = sigmas[(size_t)(i + 1)];
		const std::vector<float> den = model_x0(denoise, x, sigma, n);
		float sigma_down = 0.0f, sigma_up = 0.0f;
		ancestral_step(sigma, sigma_next, /*eta=*/1.0, &sigma_down, &sigma_up);
		if (sigma_down == 0.0f) {
			x = den;   // the reference's explicit collapse
			continue;
		}
		const float dt = sigma_down - sigma;
		for (size_t j = 0; j < n; j++) {
			const float d = (x[j] - den[j]) / sigma;
			x[j] = x[j] + d * dt;
		}
		if (sigma_up > 0.0f) {
			const std::vector<float> noise = step_noise(seed, i, 0, n);
			for (size_t j = 0; j < n; j++) x[j] = x[j] + noise[j] * s_noise * sigma_up;
		}
	}
	return x;
}

// ── heun (s_churn = 0) ─────────────────────────────────────────────────────
//
// sample_heun. The intermediate point is the Euler step; the correction
// re-evaluates the model at sigma_next and averages the two derivatives. The
// last step (sigma_next == 0) is the plain Euler branch, never the correction.
static std::vector<float> run_heun(const std::vector<float>& sigmas, std::vector<float> x,
                                   const DenoiseFn& denoise, u64, i64 total,
                                   const std::function<void(i64, i64)>& on_step) {
	const size_t n = x.size();
	for (i64 i = 0; i < total; i++) {
		if (on_step) on_step(i, total);
		const float sigma = sigmas[(size_t)i];
		const float sigma_next = sigmas[(size_t)(i + 1)];
		const std::vector<float> den = model_x0(denoise, x, sigma, n);
		const float dt = sigma_next - sigma;
		if (sigma_next == 0.0f) {
			for (size_t j = 0; j < n; j++) {
				const float d = (x[j] - den[j]) / sigma;
				x[j] = x[j] + d * dt;
			}
			continue;
		}
		std::vector<float> x2(n);
		for (size_t j = 0; j < n; j++) {
			const float d = (x[j] - den[j]) / sigma;
			x2[j] = x[j] + d * dt;
		}
		const std::vector<float> den2 = model_x0(denoise, x2, sigma_next, n);
		for (size_t j = 0; j < n; j++) {
			const float d = (x[j] - den[j]) / sigma;
			const float d2 = (x2[j] - den2[j]) / sigma_next;
			const float d_prime = (d + d2) / 2.0f;
			x[j] = x[j] + d_prime * dt;
		}
	}
	return x;
}

// ── dpmpp_2m ───────────────────────────────────────────────────────────────
//
// sample_dpmpp_2m: DPM-Solver++(2M) in log-sigma space. The first step has no
// history and the last has sigma_next == 0, so both take the same first-order
// closed form `exp(-h) * x - expm1(-h) * denoised`; the interior steps blend the
// previous denoised estimate with r = h_last / h.
static std::vector<float> run_dpmpp_2m(const std::vector<float>& sigmas, std::vector<float> x,
                                       const DenoiseFn& denoise, u64, i64 total,
                                       const std::function<void(i64, i64)>& on_step) {
	const size_t n = x.size();
	std::vector<float> old_den;
	bool has_old = false;
	for (i64 i = 0; i < total; i++) {
		if (on_step) on_step(i, total);
		const float sigma = sigmas[(size_t)i];
		const float sigma_next = sigmas[(size_t)(i + 1)];
		const std::vector<float> den = model_x0(denoise, x, sigma, n);
		const double t = t_fn((double)sigma), t_next = t_fn((double)sigma_next);
		const double h = t_next - t;
		// sigma_fn(t_next) / sigma_fn(t) and the denoised coefficient
		// `-(-h).expm1()` of `x = ... * x - (-h).expm1() * denoised`. The
		// coefficient is positive: the reference's unary minus applies to the
		// (negative) expm1, so the update is a convex blend toward denoised.
		const float ratio = (float)sigma_fn(t_next) / (float)sigma_fn(t);
		const float ncoef = (float)(-std::expm1(-h));
		if (!has_old || sigma_next == 0.0f) {
			// Last step: h -> inf, ratio -> 0, ncoef -> 1, i.e. x = denoised.
			for (size_t j = 0; j < n; j++) x[j] = ratio * x[j] + ncoef * den[j];
		} else {
			const double h_last = t - t_fn((double)sigmas[(size_t)(i - 1)]);
			const double r = h_last / h;
			const float q = (float)(1.0 / (2.0 * r));   // 1 / (2 * r)
			const float k1 = 1.0f + q;
			for (size_t j = 0; j < n; j++) {
				const float dd = k1 * den[j] - q * old_den[j];
				x[j] = ratio * x[j] + ncoef * dd;
			}
		}
		old_den = den;
		has_old = true;
	}
	return x;
}

// ── dpmpp_2s_ancestral (eta = 1) ───────────────────────────────────────────
//
// sample_dpmpp_2s_ancestral: one model evaluation at the midpoint sigma_s =
// sigma_fn(t + h/2) drives the second-order correction; the ancestral noise is
// added on every step whose *next* sigma is non-zero (the reference adds it even
// when sigma_up is 0, so the draw is made unconditionally there).
static std::vector<float> run_dpmpp_2s_ancestral(const std::vector<float>& sigmas,
                                                 std::vector<float> x, const DenoiseFn& denoise,
                                                 u64 seed, i64 total,
                                                 const std::function<void(i64, i64)>& on_step) {
	const size_t n = x.size();
	constexpr double eta = 1.0;
	constexpr float s_noise = 1.0f;
	constexpr double r = 1.0 / 2.0;
	for (i64 i = 0; i < total; i++) {
		if (on_step) on_step(i, total);
		const float sigma = sigmas[(size_t)i];
		const float sigma_next = sigmas[(size_t)(i + 1)];
		const std::vector<float> den = model_x0(denoise, x, sigma, n);
		float sigma_down = 0.0f, sigma_up = 0.0f;
		ancestral_step(sigma, sigma_next, eta, &sigma_down, &sigma_up);
		if (sigma_down == 0.0f) {
			// Euler branch (the last step).
			const float dt = sigma_down - sigma;
			for (size_t j = 0; j < n; j++) {
				const float d = (x[j] - den[j]) / sigma;
				x[j] = x[j] + d * dt;
			}
		} else {
			// DPM-Solver++(2S)
			const double t = t_fn((double)sigma), t_next = t_fn((double)sigma_down);
			const double h = t_next - t;
			const double s = t + r * h;
			const float sigma_s = (float)sigma_fn(s);
			const float ratio_s = (float)sigma_fn(s) / (float)sigma_fn(t);
			// Coefficient on denoised: `-(-h * r).expm1()`, positive.
			const float n_s = (float)(-std::expm1(-h * r));
			std::vector<float> x2(n);
			for (size_t j = 0; j < n; j++) x2[j] = ratio_s * x[j] + n_s * den[j];
			const std::vector<float> den2 = model_x0(denoise, x2, sigma_s, n);
			const float ratio_n = (float)sigma_fn(t_next) / (float)sigma_fn(t);
			const float n_n = (float)(-std::expm1(-h));
			for (size_t j = 0; j < n; j++) x[j] = ratio_n * x[j] + n_n * den2[j];
		}
		if (sigma_next > 0.0f) {
			const std::vector<float> noise = step_noise(seed, i, 0, n);
			for (size_t j = 0; j < n; j++) x[j] = x[j] + noise[j] * s_noise * sigma_up;
		}
	}
	return x;
}

// ── dpmpp_2m_sde (solver_type = 'midpoint', eta = 1) ───────────────────────
//
// sample_dpmpp_2m_sde. The update is the SDE form
//     x = sigmas[i+1]/sigmas[i] * exp(-h*eta) * x + alpha_t * (1 - exp(-h_eta)) * denoised
// plus the midpoint correction over the previous denoised estimate, plus a
// `noise * sigma_next * sqrt(1 - exp(-2*h*eta))` kick. `alpha_t` is
// `sigmas[i+1] * exp(lambda_t)` — algebraically 1 for log-sigma models, but kept
// because the reference computes it.
static std::vector<float> run_dpmpp_2m_sde(const std::vector<float>& sigmas, std::vector<float> x,
                                           const DenoiseFn& denoise, u64 seed, i64 total,
                                           const std::function<void(i64, i64)>& on_step) {
	const size_t n = x.size();
	constexpr double eta = 1.0;
	constexpr double s_noise = 1.0;
	std::vector<float> old_den;
	bool has_old = false;
	double h = 0.0, h_last = 0.0;
	for (i64 i = 0; i < total; i++) {
		if (on_step) on_step(i, total);
		const float sigma = sigmas[(size_t)i];
		const float sigma_next = sigmas[(size_t)(i + 1)];
		const std::vector<float> den = model_x0(denoise, x, sigma, n);
		if (sigma_next == 0.0f) {
			x = den;   // denoising step
		} else {
			const double ls = t_fn((double)sigma), lt = t_fn((double)sigma_next);
			h = lt - ls;
			const double h_eta = h * (eta + 1.0);
			const float alpha_t = (float)((double)sigma_next * std::exp(lt));
			const float coef_x = sigma_next / sigma * (float)std::exp(-h * eta);
			const float coef_d = alpha_t * (float)(-std::expm1(-h_eta));
			for (size_t j = 0; j < n; j++) x[j] = coef_x * x[j] + coef_d * den[j];
			if (has_old) {
				const double r = h_last / h;
				// 0.5 * alpha_t * (-h_eta).expm1().neg() * (1 / r), the midpoint
				// coefficient, multiplied left-to-right exactly like the reference.
				const float q = 0.5f * alpha_t * (float)(-std::expm1(-h_eta)) * (float)(1.0 / r);
				for (size_t j = 0; j < n; j++) x[j] = x[j] + q * (den[j] - old_den[j]);
			}
			if (eta > 0.0 && s_noise > 0.0) {
				const float su = (float)std::sqrt(-std::expm1(-2.0 * h * eta));
				const std::vector<float> noise = step_noise(seed, i, 0, n);
				for (size_t j = 0; j < n; j++)
					x[j] = x[j] + noise[j] * sigma_next * su * (float)s_noise;
			}
		}
		old_den = den;
		has_old = true;
		h_last = h;
	}
	return x;
}

// ── dpmpp_sde (r = 1/2, eta = 1) ───────────────────────────────────────────
//
// sample_dpmpp_sde: the stochastic DPM-Solver++. Each step makes two model
// evaluations — the first at sigma_s_1 = sigma_fn(lambda_s + r*h) — and adds an
// ancestral noise kick after each. `fac = 1/(2r)` is 1 here, so `denoised_d`
// collapses to denoised_2, but the blend is written out as the reference has it.
static std::vector<float> run_dpmpp_sde(const std::vector<float>& sigmas, std::vector<float> x,
                                        const DenoiseFn& denoise, u64 seed, i64 total,
                                        const std::function<void(i64, i64)>& on_step) {
	const size_t n = x.size();
	constexpr double eta = 1.0;
	constexpr double s_noise = 1.0;
	constexpr double r = 1.0 / 2.0;
	const float fac = (float)(1.0 / (2.0 * r));
	for (i64 i = 0; i < total; i++) {
		if (on_step) on_step(i, total);
		const float sigma = sigmas[(size_t)i];
		const float sigma_next = sigmas[(size_t)(i + 1)];
		const std::vector<float> den = model_x0(denoise, x, sigma, n);
		if (sigma_next == 0.0f) {
			x = den;   // denoising step
			continue;
		}
		const double ls = t_fn((double)sigma), lt = t_fn((double)sigma_next);
		const double h = lt - ls;
		const double ls1 = ls + r * h;
		const float sigma_s_1 = (float)sigma_fn(ls1);
		// alpha_* = sigma * exp(lambda): algebraically 1 for log-sigma models.
		const float alpha_s = (float)((double)sigma * std::exp(ls));
		const float alpha_s_1 = (float)((double)sigma_s_1 * std::exp(ls1));
		const float alpha_t = (float)((double)sigma_next * std::exp(lt));

		// Step 1: an ancestral collapse from sigma to sigma_s_1.
		float sd = 0.0f, su = 0.0f;
		ancestral_step((float)std::exp(-ls), sigma_s_1, eta, &sd, &su);
		const double h1 = t_fn((double)sd) - ls;
		const float coef_x1 = alpha_s_1 / alpha_s * (float)std::exp(-h1);
		const float m1 = alpha_s_1 * (float)std::expm1(-h1);   // negative
		std::vector<float> x2(n);
		for (size_t j = 0; j < n; j++) x2[j] = coef_x1 * x[j] - m1 * den[j];
		if (eta > 0.0 && s_noise > 0.0) {
			const std::vector<float> noise = step_noise(seed, i, 0, n);
			for (size_t j = 0; j < n; j++)
				x2[j] = x2[j] + alpha_s_1 * noise[j] * (float)s_noise * su;
		}
		const std::vector<float> den2 = model_x0(denoise, x2, sigma_s_1, n);

		// Step 2: the same from sigma to sigma_next, blended with the two
		// denoised estimates.
		ancestral_step((float)std::exp(-ls), (float)std::exp(-lt), eta, &sd, &su);
		const double h2 = t_fn((double)sd) - ls;
		const float coef_x2 = alpha_t / alpha_s * (float)std::exp(-h2);
		const float m2 = alpha_t * (float)std::expm1(-h2);   // negative
		const float keep = 1.0f - fac;
		for (size_t j = 0; j < n; j++) {
			const float dd = keep * den[j] + fac * den2[j];
			x[j] = coef_x2 * x[j] - m2 * dd;
		}
		if (eta > 0.0 && s_noise > 0.0) {
			const std::vector<float> noise = step_noise(seed, i, 1, n);
			for (size_t j = 0; j < n; j++) x[j] = x[j] + alpha_t * noise[j] * (float)s_noise * su;
		}
	}
	return x;
}

// ── res_multistep (eta = 0) ────────────────────────────────────────────────
//
// sample_res_multistep / sampling.hpp's ResMultistep. The class already carries
// the cross-step state (old denoised, old sigma_down, previous sigma) and the
// nan_to_num / phi1 / phi2 arithmetic; feeding it `denoised = model(x, sigma)`
// per step is the whole port.
static std::vector<float> run_res_multistep(const std::vector<float>& sigmas, std::vector<float> x,
                                            const DenoiseFn& denoise, u64, i64 total,
                                            const std::function<void(i64, i64)>& on_step) {
	const size_t n = x.size();
	ResMultistep rm;
	rm.reset();
	for (i64 i = 0; i < total; i++) {
		if (on_step) on_step(i, total);
		const float sigma = sigmas[(size_t)i];
		const float sigma_next = sigmas[(size_t)(i + 1)];
		const std::vector<float> den = model_x0(denoise, x, sigma, n);
		x = rm.step(x, sigma, sigma_next, den);
	}
	return x;
}

// ── lcm ────────────────────────────────────────────────────────────────────
//
// sample_lcm: every step is `x = denoised`, and while the next sigma is still
// non-zero the sample is re-noised to that level. The reference calls
// `model_sampling.noise_scaling(sigma_next, noise, x)`; the LCM path patches the
// model with comfy_extras' `LCM(EPS)` sampling, and that class is where
// sigma_data = 0.5 lives — in its calculate_denoised x0 preconditioning, which
// is the caller's denoise callback's job, not the sampler's. The noise scaling
// the sampler does is the EPS one it inherits, with max_denoise = False: i.e.
// `latent + noise * sigma`. The default s_noise = 1 leaves the draw unscaled.
static std::vector<float> run_lcm(const std::vector<float>& sigmas, std::vector<float> x,
                                  const DenoiseFn& denoise, u64 seed, i64 total,
                                  const std::function<void(i64, i64)>& on_step) {
	const size_t n = x.size();
	for (i64 i = 0; i < total; i++) {
		if (on_step) on_step(i, total);
		const float sigma = sigmas[(size_t)i];
		const float sigma_next = sigmas[(size_t)(i + 1)];
		const std::vector<float> den = model_x0(denoise, x, sigma, n);
		x = den;
		if (sigma_next > 0.0f) {
			const std::vector<float> noise = step_noise(seed, i, 0, n);
			for (size_t j = 0; j < n; j++) x[j] = x[j] + noise[j] * sigma_next;
		}
	}
	return x;
}

// ── name table ─────────────────────────────────────────────────────────────

namespace {
constexpr const char* kSamplerNames[] = {
    "euler",       "euler_ancestral", "heun",          "dpmpp_2m", "dpmpp_2s_ancestral",
    "dpmpp_2m_sde", "dpmpp_sde",      "res_multistep", "lcm",
};
constexpr int kSamplerCount = (int)(sizeof(kSamplerNames) / sizeof(kSamplerNames[0]));
}  // namespace

bool sampler_from_name(const std::string& name, SamplerKind* out) {
	for (int i = 0; i < kSamplerCount; i++) {
		if (name == kSamplerNames[i]) {
			if (out) *out = (SamplerKind)i;
			return true;
		}
	}
	// The rest of KSAMPLER_NAMES (the cfg_pp / gpu / heunpp2 / DEIS families) is
	// deliberately not ported: the caller keeps its own default.
	return false;
}

const char* sampler_name(SamplerKind k) {
	const int i = (int)k;
	if (i < 0 || i >= kSamplerCount) return "euler";
	return kSamplerNames[i];
}

std::vector<std::string> sampler_names() {
	std::vector<std::string> out;
	out.reserve((size_t)kSamplerCount);
	for (int i = 0; i < kSamplerCount; i++) out.emplace_back(kSamplerNames[i]);
	return out;
}

// The sampler a chain runs when its settings name none: the released workflow's
// own. Qwen-Image's image chain and ACE-Step 1.5's music chain both run euler;
// MiniMax H3's video workflow runs res_multistep (which is what its joint A/V
// walk implements, see sampling/h3_joint_av.hpp).
const char* default_sampler_name(const std::string& arch) {
	return arch == "minimax_h3" ? "res_multistep" : "euler";
}

bool sampler_is_stochastic(SamplerKind k) {
	switch (k) {
	case SamplerKind::Euler: return false;
	case SamplerKind::EulerAncestral: return true;
	case SamplerKind::Heun: return false;
	case SamplerKind::DPMpp2M: return false;
	case SamplerKind::DPMpp2SAncestral: return true;
	case SamplerKind::DPMpp2MSDE: return true;
	case SamplerKind::DPMppSDE: return true;
	case SamplerKind::ResMultistep: return false;
	case SamplerKind::LCM: return true;
	}
	return false;
}

// ── dispatch ───────────────────────────────────────────────────────────────

std::vector<float> run_sampler(SamplerKind kind, const std::vector<float>& sigmas,
                               std::vector<float> x, const DenoiseFn& denoise,
                               const std::vector<float>& noise, u64 seed,
                               const std::function<void(i64, i64)>& on_step) {
	if (sigmas.size() < 2)
		throw MediaError("run_sampler: the sigma grid must have at least two entries");
	if (!noise.empty() && noise.size() != x.size())
		throw MediaError("run_sampler: the noise field must have the same size as x");
	const i64 total = (i64)sigmas.size() - 1;
	switch (kind) {
	case SamplerKind::Euler:
		return run_euler(sigmas, std::move(x), denoise, seed, total, on_step);
	case SamplerKind::EulerAncestral:
		return run_euler_ancestral(sigmas, std::move(x), denoise, seed, total, on_step);
	case SamplerKind::Heun:
		return run_heun(sigmas, std::move(x), denoise, seed, total, on_step);
	case SamplerKind::DPMpp2M:
		return run_dpmpp_2m(sigmas, std::move(x), denoise, seed, total, on_step);
	case SamplerKind::DPMpp2SAncestral:
		return run_dpmpp_2s_ancestral(sigmas, std::move(x), denoise, seed, total, on_step);
	case SamplerKind::DPMpp2MSDE:
		return run_dpmpp_2m_sde(sigmas, std::move(x), denoise, seed, total, on_step);
	case SamplerKind::DPMppSDE:
		return run_dpmpp_sde(sigmas, std::move(x), denoise, seed, total, on_step);
	case SamplerKind::ResMultistep:
		return run_res_multistep(sigmas, std::move(x), denoise, seed, total, on_step);
	case SamplerKind::LCM:
		return run_lcm(sigmas, std::move(x), denoise, seed, total, on_step);
	}
	throw MediaError("run_sampler: unknown sampler kind");
}

}  // namespace phi::media
