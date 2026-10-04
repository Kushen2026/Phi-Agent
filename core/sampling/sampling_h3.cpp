// MiniMax H3 joint sampler — Phase 0 implementation of the schedule maths.
//
// The sigma grid is *not* guessed: it is reproduced from the reference that owns
// the workflow (ComfyUI `comfy/samplers.py::beta_scheduler` +
// `comfy/model_sampling.py::ModelSamplingDiscreteFlow/AV`), and checked bit for
// bit against a dump of the same code. The workflow is
//
//     BasicScheduler(scheduler="beta", steps=20)  shift=12  audio_shift=3
//     KSampler(sampler="res_multistep")
//
// so:
//   * the model's sigma table is  time_snr_shift(shift, (k+1)/1000), k=0..999
//     (the `multiplier` cancels out of the table)
//   * beta_scheduler draws `steps` quantiles of Beta(0.6, 0.6), rounds them onto
//     that table and appends 0.0
//   * ModelSamplingAV.audio_scale = shift / audio_shift
#include "sampling/sampling_h3.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "sampling/h3_joint_av.hpp"
#include "sampling/sampling.hpp"

namespace phi::media {

namespace {

constexpr int kTableSteps = 1000;   // ModelSamplingDiscreteFlow default timesteps

// ── regularised incomplete beta, inverted ──────────────────────────────────
//
// beta_scheduler needs Beta(alpha, beta).ppf at alpha = beta = 0.6. There is no
// closed form, so this is the standard continued fraction for I_x(a,b)
// (Numerical Recipes `betacf`/`betai`) inverted by bisection. Bisection is used
// rather than Newton because ppf only has to be accurate to well under half a
// table step (1/1000) and bisection cannot diverge near the tails, which is
// exactly where the beta quantiles for a small `steps` land.
double betacf(double a, double b, double x) {
	constexpr int kMaxIt = 300;
	constexpr double kEps = 3e-16;
	constexpr double kFpmim = 1e-300;
	const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
	double c = 1.0;
	double d = 1.0 - qab * x / qap;
	if (std::fabs(d) < kFpmim) d = kFpmim;
	d = 1.0 / d;
	double h = d;
	for (int m = 1; m <= kMaxIt; m++) {
		const int m2 = 2 * m;
		double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
		d = 1.0 + aa * d;
		if (std::fabs(d) < kFpmim) d = kFpmim;
		c = 1.0 + aa / c;
		if (std::fabs(c) < kFpmim) c = kFpmim;
		d = 1.0 / d;
		h *= d * c;
		aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
		d = 1.0 + aa * d;
		if (std::fabs(d) < kFpmim) d = kFpmim;
		c = 1.0 + aa / c;
		if (std::fabs(c) < kFpmim) c = kFpmim;
		d = 1.0 / d;
		const double del = d * c;
		h *= del;
		if (std::fabs(del - 1.0) < kEps) break;
	}
	return h;
}

// I_x(a, b), the regularised incomplete beta function.
double betai(double a, double b, double x) {
	if (x <= 0.0) return 0.0;
	if (x >= 1.0) return 1.0;
	const double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
	                           a * std::log(x) + b * std::log(1.0 - x));
	if (x < (a + 1.0) / (a + b + 2.0)) return bt * betacf(a, b, x) / a;
	return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

// Inverse of betai: the p-quantile of Beta(a, b).
double beta_ppf(double p, double a, double b) {
	if (p <= 0.0) return 0.0;
	if (p >= 1.0) return 1.0;
	double lo = 0.0, hi = 1.0;
	// 200 halvings put the bracket below double resolution, so the result is
	// limited only by betai itself.
	for (int i = 0; i < 200; i++) {
		const double mid = 0.5 * (lo + hi);
		if (betai(a, b, mid) < p)
			lo = mid;
		else
			hi = mid;
	}
	return 0.5 * (lo + hi);
}

// numpy's np.rint: round half to even. std::nearbyint under the default
// FE_TONEAREST mode is the same function (and already returns an integral value).
inline i64 rint_i64(double v) { return (i64)std::nearbyint(v); }

// float32 time_snr_shift, matching how the reference actually evaluates it.
//
// This is deliberately NOT the shared double-precision `time_snr_shift()` from
// sampling.*: the reference computes the table with torch float32 tensor ops
// (a python-float scalar does not promote the tensor), so doing the same algebra
// in double and rounding once at the end lands 1 ulp off on ~1/3 of the entries.
// The H3 sigma grid is compared bit for bit, so the rounding has to match.
inline float tss_f32(float alpha, float t) {
	if (alpha == 1.0f) return t;
	return (float)(alpha * t / (1.0f + (alpha - 1.0f) * t));
}

// ModelSamplingDiscreteFlow's table, in the reference's own evaluation order:
//     ts = sigma((arange(1, N+1) / N) * multiplier)
//     sigma(t) = time_snr_shift(shift, t / multiplier)
// multiplier cancels algebraically but NOT in float32, hence the round trip.
std::vector<float> sigma_table(float shift, int timesteps = kTableSteps,
                              float multiplier = 1000.0f) {
	std::vector<float> t((size_t)timesteps);
	for (int k = 0; k < timesteps; k++) {
		const float a = (float)(k + 1) / (float)timesteps;
		const float b = a * multiplier;
		const float c = b / multiplier;
		t[(size_t)k] = tss_f32(shift, c);
	}
	return t;
}

}  // namespace

float h3_time_shift_sigma(float sigma, float from_shift, float to_shift) {
	// comfy/ldm/minimax/model.py::time_shift_sigma — invert the flow shift, then
	// re-apply the other. Used to carry the audio latent onto the video schedule.
	const float base = sigma / (from_shift + sigma * (1.0f - from_shift));
	return to_shift * base / (1.0f + (to_shift - 1.0f) * base);
}

float h3_audio_scale(const H3SamplingParams& p) {
	return p.audio_shift != 0.0f ? p.shift / p.audio_shift : 1.0f;
}

std::vector<float> h3_sigmas(const H3SamplingParams& p) {
	SamplingFlow flow;
	flow.shift = p.shift;
	flow.multiplier = 1.0f;
	return simple_scheduler(0.0f, 1.0f, (int)p.steps, 1.0f, flow);
}

std::vector<float> h3_audio_sigmas(const H3SamplingParams& p) {
	// The audio stream rides the video grid through time_shift_sigma; the sampler
	// carries it scaled by audio_scale (model.py::forward).
	std::vector<float> v = p.use_beta ? h3_beta_sigmas(p) : h3_sigmas(p);
	for (float& s : v) s = h3_time_shift_sigma(s, p.shift, p.audio_shift);
	return v;
}

std::vector<float> h3_beta_sigmas(const H3SamplingParams& p) {
	const std::vector<float> table = sigma_table(p.shift);
	const i64 total = (i64)table.size() - 1;   // 999
	const i64 steps = std::max<i64>(1, p.steps);

	std::vector<float> sigs;
	sigs.reserve((size_t)steps + 1);
	i64 last = -1;
	for (i64 i = 0; i < steps; i++) {
		// ts = 1 - linspace(0, 1, steps, endpoint=False)  =>  1 - i/steps
		const double u = 1.0 - (double)i / (double)steps;
		const double q = beta_ppf(u, (double)p.alpha, (double)p.beta);
		i64 t = rint_i64(q * (double)total);
		t = std::clamp<i64>(t, 0, total);
		if (t != last) sigs.push_back(table[(size_t)t]);
		last = t;
	}
	sigs.push_back(0.0f);
	return sigs;
}

void h3_euler_step(float* x, const float* v, i64 n, float sigma, float sigma_next) {
	const float d = sigma_next - sigma;   // < 0: sigma decreases
	for (i64 i = 0; i < n; i++) x[i] += d * v[i];
}

// ── carry / anti-carry ─────────────────────────────────────────────────────
//
// Mirrors comfy/ldm/minimax/model.py::MiniMaxH3Model.forward exactly, including
// the evaluation order (the workflow passes `scale = audio_scale`):
//
//   sigma_v = clamp(timestep / 1000, min=1e-6)          # here t == sigma_v
//   sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
//   carry   = sigma_a / sigma_v
//   x_net   = x_a_sampler * carry
//   out     = (1 - scale) * (x_a_sampler * carry)
//           + (1 + (scale - 1) * sigma_a) * net_out
//
// Everything is float32 because that is what torch runs when the operands are
// float32 tensors (python-float scalars do not promote a tensor).

H3AVCarry h3_av_carry(float sigma_v, float shift_v, float shift_a) {
	H3AVCarry c;
	c.sigma_v = sigma_v < 1e-6f ? 1e-6f : sigma_v;
	c.sigma_a = h3_time_shift_sigma(c.sigma_v, shift_v, shift_a);
	c.carry = c.sigma_a / c.sigma_v;
	return c;
}

void h3_av_scale_audio(float* x_a, i64 n, float scale) {
	// _scale_audio_slice returns the slice untouched when the scale is 1.
	if (scale == 1.0f) return;
	for (i64 i = 0; i < n; i++) x_a[i] *= scale;
}

void h3_av_audio_to_dit(const float* x_a_sampler, float* x_net, i64 n, float carry) {
	for (i64 i = 0; i < n; i++) x_net[i] = x_a_sampler[i] * carry;
}

void h3_av_audio_from_dit(const float* net_out, const float* x_a_sampler, float* out_sampler,
                          i64 n, float carry, float sigma_a, float scale) {
	if (scale == 1.0f) {
		// forward() skips the whole block when scale == 1.0.
		for (i64 i = 0; i < n; i++) out_sampler[i] = net_out[i];
		return;
	}
	const float one_minus_scale = 1.0f - scale;
	const float scale_minus_one = scale - 1.0f;
	const float t2 = 1.0f + scale_minus_one * sigma_a;   // (1 + (scale-1)*sigma_a)
	for (i64 i = 0; i < n; i++) {
		const float carried = x_a_sampler[i] * carry;
		const float t1 = one_minus_scale * carried;
		out_sampler[i] = t1 + t2 * net_out[i];
	}
}

void H3JointSamplerV2::init(std::vector<float> x_v, std::vector<float> x_a_sampler,
                            std::vector<float> sigmas) {
	// The whole packed latent rides the *video* schedule; the audio stream is
	// carried onto it by the model's carry/unscale (see h3_joint_av.hpp).
	//
	// A caller-supplied grid is used verbatim: the video sampler node takes the
	// SCHEDULER node's sigmas, so one schedule decides both the sigma the DiT is
	// told and the sigma the integrator advances on (the two used to be derived
	// independently here, which is only harmless while the one scheduler name both
	// sides understand is "beta").
	if (!sigmas.empty()) {
		sigmas_ = std::move(sigmas);
		p_.steps = (i64)sigmas_.size() - 1;
	} else {
		sigmas_ = p_.use_beta ? h3_beta_sigmas(p_) : h3_sigmas(p_);
	}
	x_v_ = std::move(x_v);
	x_a_ = std::move(x_a_sampler);
	den_v_.assign(x_v_.size(), 0.0f);
	den_a_.assign(x_a_.size(), 0.0f);
	rm_v_.reset();
	rm_a_.reset();
}

void H3JointSamplerV2::step(i64 i, const std::vector<float>& v_v,
                            const std::vector<float>& v_a_sampler) {
	if (i < 0 || (size_t)(i + 1) >= sigmas_.size())
		throw MediaError("H3JointSamplerV2::step: index out of range");
	const float s0 = sigmas_[(size_t)i], s1 = sigmas_[(size_t)i + 1];
	if (v_v.size() != x_v_.size() || v_a_sampler.size() != x_a_.size())
		throw MediaError("H3JointSamplerV2::step: velocity size does not match latent");
	if (!multistep_) {
		// Plain Euler, both streams on the same video grid: the DiT's velocity is
		// already d(x)/d(sigma), so the update is x += v * (s1 - s0).
		const float ds = s1 - s0;
		for (size_t k = 0; k < x_v_.size(); k++) x_v_[k] += ds * v_v[k];
		for (size_t k = 0; k < x_a_.size(); k++) x_a_[k] += ds * v_a_sampler[k];
		return;
	}
	// res_multistep integrates *denoised* estimates, and the flow model's
	// denoise step is x0 = x - sigma * v (ModelSamplingDiscreteFlow), so the
	// velocity the DiT produces is turned into `denoised` here. Both streams
	// take the same sigma pair: the
	// reference samples the flat AV pack on the video grid (comfy/samplers.py),
	// so the two res_multistep states stay in lockstep.
	for (size_t k = 0; k < x_v_.size(); k++) den_v_[k] = x_v_[k] - s0 * v_v[k];
	for (size_t k = 0; k < x_a_.size(); k++) den_a_[k] = x_a_[k] - s0 * v_a_sampler[k];
	x_v_ = rm_v_.step(x_v_, s0, s1, den_v_);
	x_a_ = rm_a_.step(x_a_, s0, s1, den_a_);
}

}  // namespace phi::media
