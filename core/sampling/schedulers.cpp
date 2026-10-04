// ComfyUI schedulers — the C++ half of SCHEDULER_HANDLERS.
//
// One scheduler = one descending sigma grid of `steps + 1` entries whose last
// entry is 0. `comfy/samplers.py` dispatches them through a small table:
//
//     simple          simple_scheduler            (model sigma table)
//     sgm_uniform     normal_scheduler(sgm=True)
//     karras          get_sigmas_karras(n, sigma_min, sigma_max)
//     exponential     get_sigmas_exponential(n, sigma_min, sigma_max)
//     ddim_uniform    ddim_scheduler              (model sigma table)
//     beta            beta_scheduler              (model sigma table)
//     normal          normal_scheduler
//     linear_quadratic linear_quadratic_schedule
//     kl_optimal      kl_optimal_scheduler(n, sigma_min, sigma_max)
//
// Three traps, all handled here:
//
//  1. `simple`, `ddim_uniform` and `beta` do not build a linspace — they index
//     the model's own sigma table (`model_sampling.sigmas`, ascending). That
//     table is `model_sigma_table(flow, N)` and lives in sampling.cpp, so the
//     schedulers and the rest of the engine cannot disagree about it.
//  2. `normal` has a rare branch (`steps += 1; append_zero = False`) that triggers
//     only when `sigma(timestep(sigma_min))` is already ~0. It is dead for the
//     DiscreteFlow / Flux shifts this engine carries, but it is part of the
//     reference and is reproduced faithfully.
//  3. KSampler.set_steps repeats *once*, at the bottom of build_sigmas: a partial
//     denoise is the tail of a longer schedule (`[-(steps + 1):]`), never a
//     rescaled copy of a short one.
//
// Scalar arithmetic follows the project's standing policy (see sampling.cpp):
// intermediates are evaluated in double and rounded to float once on the way
// out, which is the correctly-rounded float32 result the Python tensors hold.
#include "sampling/schedulers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace phi::media {

namespace {

// ── small utilities ────────────────────────────────────────────────────────

// numpy's np.rint: round half to even. std::nearbyint under the default
// FE_TONEAREST mode is exactly that.
i64 rint_i64(double v) { return (i64)std::nearbyint(v); }

// math.isclose(x, 0, abs_tol=atol): with numpy's default rel_tol the comparison
// collapses to |x| <= atol for every x that is not itself infinitesimal.
bool isclose_zero(float v, double atol) { return std::fabs((double)v) <= atol; }

// torch.linspace(a, b, n): n points, both ends inclusive; n == 1 is [a].
// Values are accumulated as `a + step * i` like ATen's CPU kernel and rounded
// once to the float32 the tensor would hold.
std::vector<float> linspace(double a, double b, int n) {
	std::vector<float> out((size_t)std::max(n, 0));
	if (n <= 0) return out;
	if (n == 1) {
		out[0] = (float)a;
		return out;
	}
	const double step = (b - a) / (double)(n - 1);
	for (int i = 0; i < n; i++) out[(size_t)i] = (float)(a + step * (double)i);
	return out;
}

// ── regularised incomplete beta, inverted (for beta_scheduler) ─────────────
//
// beta_scheduler needs the Beta(0.6, 0.6) ppf. There is no closed form, so this
// is the standard continued fraction for I_x(a, b) (Numerical Recipes
// `betacf`/`betai`) inverted by bisection: 200 halvings drive the bracket below
// double resolution, so the result is limited only by betai itself. Bisection is
// used rather than Newton because the ppf only has to be accurate to well under
// half a table step and cannot diverge near the tails, which is exactly where
// the quantiles for a small `steps` land. (Same approach as sampling_h3.cpp; it
// is re-implemented here rather than shared, because that file's helpers are
// file-local.)
double betacf(double a, double b, double x) {
	constexpr int kMaxIt = 300;
	constexpr double kEps = 3e-16;
	constexpr double kFpmin = 1e-300;
	const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
	double c = 1.0;
	double d = 1.0 - qab * x / qap;
	if (std::fabs(d) < kFpmin) d = kFpmin;
	d = 1.0 / d;
	double h = d;
	for (int m = 1; m <= kMaxIt; m++) {
		const int m2 = 2 * m;
		double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
		d = 1.0 + aa * d;
		if (std::fabs(d) < kFpmin) d = kFpmin;
		c = 1.0 + aa / c;
		if (std::fabs(c) < kFpmin) c = kFpmin;
		d = 1.0 / d;
		h *= d * c;
		aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
		d = 1.0 + aa * d;
		if (std::fabs(d) < kFpmin) d = kFpmin;
		c = 1.0 + aa / c;
		if (std::fabs(c) < kFpmin) c = kFpmin;
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
	for (int i = 0; i < 200; i++) {
		const double mid = 0.5 * (lo + hi);
		if (betai(a, b, mid) < p)
			lo = mid;
		else
			hi = mid;
	}
	return 0.5 * (lo + hi);
}

}  // namespace

// ── name table ─────────────────────────────────────────────────────────────
//
// The order is the enum order (it is what a settings dropdown should show) and
// the canonical spelling is what the Python table uses. `ddim` (and `sgm`) are
// well-known aliases ComfyUI accepts elsewhere; everything else is rejected so
// the caller can keep its own default.

namespace {
constexpr const char* kSchedulerNames[] = {
    "simple", "normal", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta",
    "linear_quadratic", "kl_optimal",
};
constexpr int kSchedulerCount = (int)(sizeof(kSchedulerNames) / sizeof(kSchedulerNames[0]));
}  // namespace

bool scheduler_from_name(const std::string& name, SchedulerKind* out) {
	for (int i = 0; i < kSchedulerCount; i++) {
		if (name == kSchedulerNames[i]) {
			if (out) *out = (SchedulerKind)i;
			return true;
		}
	}
	// The two aliases ComfyUI's own UI understands.
	if (name == "ddim") {
		if (out) *out = SchedulerKind::DDIMUniform;
		return true;
	}
	if (name == "sgm") {
		if (out) *out = SchedulerKind::SGMUniform;
		return true;
	}
	return false;
}

const char* scheduler_name(SchedulerKind k) {
	const int i = (int)k;
	if (i < 0 || i >= kSchedulerCount) return "simple";
	return kSchedulerNames[i];
}

std::vector<std::string> scheduler_names() {
	std::vector<std::string> out;
	out.reserve((size_t)kSchedulerCount);
	for (int i = 0; i < kSchedulerCount; i++) out.emplace_back(kSchedulerNames[i]);
	return out;
}

// ── model endpoints ────────────────────────────────────────────────────────

// `model_sampling.sigma_min` / `sigma_max` are just the ends of the registered
// table, and the table length is a property of the flow class (1000 for
// ModelSamplingDiscreteFlow, 10000 for ModelSamplingFlux). Both are ascending.
void model_sigma_bounds(const SamplingFlow& flow, float* sigma_min, float* sigma_max) {
	const int N = flow.kind == SamplingFlow::Kind::Flux ? kFluxSigmaTableSteps : kSigmaTableSteps;
	const std::vector<float> table = model_sigma_table(flow, N);
	if (sigma_min) *sigma_min = table.front();
	if (sigma_max) *sigma_max = table.back();
}

// ── build_sigmas ───────────────────────────────────────────────────────────

std::vector<float> build_sigmas(SchedulerKind kind, const SamplingFlow& flow, int steps,
                                float denoise) {
	if (steps <= 0) throw MediaError("build_sigmas: steps must be > 0");

	// KSampler.set_steps(steps, denoise):
	//     > 0.9999          -> the full `steps`-step schedule
	//     <= 0.0            -> empty
	//     otherwise         -> calculate_sigmas(int(steps / denoise))[-(steps + 1):]
	// The partial case is the *tail* of a longer schedule, so the tail trim is
	// applied once, at the end, to whatever this branch builds.
	if (denoise <= 0.0f) return {};
	int build_steps = steps;
	bool trim = false;
	if (!(denoise > 0.9999f)) {
		build_steps = (int)((double)steps / (double)denoise);  // Python int() truncates
		if (build_steps <= 0) return {};
		trim = true;
	}

	// Every table-based scheduler (and the endpoints the three closed-form ones
	// take) uses the flow's own table length.
	const int N = flow.kind == SamplingFlow::Kind::Flux ? kFluxSigmaTableSteps : kSigmaTableSteps;

	std::vector<float> sigs;
	switch (kind) {
	case SchedulerKind::Simple: {
		// The existing bit-exact port (sampling.cpp) already reproduces
		// simple_scheduler. Pass denoise = 1 so the trim stays centralised here.
		float sigma_min = 0.0f, sigma_max = 0.0f;
		model_sigma_bounds(flow, &sigma_min, &sigma_max);
		sigs = simple_scheduler(sigma_min, sigma_max, build_steps, 1.0f, flow);
		break;
	}
	case SchedulerKind::Normal:
	case SchedulerKind::SGMUniform: {
		// normal_scheduler: linspace on the *timestep* axis, then sigma().
		// `start`/`end` are timestep(sigma_max)/timestep(sigma_min); they are not
		// the inverses of sigma(), which is why this cannot be a sigma linspace.
		float sigma_min = 0.0f, sigma_max = 0.0f;
		model_sigma_bounds(flow, &sigma_min, &sigma_max);
		const float start = flow.timestep(sigma_max);
		const float end = flow.timestep(sigma_min);
		bool append_zero = true;
		if (kind == SchedulerKind::SGMUniform) {
			// sgm=True: linspace(start, end, steps + 1)[:-1] — one extra point,
			// then the last is dropped, which moves every interior point.
			std::vector<float> ts = linspace((double)start, (double)end, build_steps + 1);
			ts.pop_back();
			sigs.reserve((size_t)build_steps + 1);
			for (float t : ts) sigs.push_back(flow.sigma(t));
		} else {
			int n = build_steps;
			// The rare branch: only reachable when sigma(timestep(sigma_min)) is
			// already ~0. For DiscreteFlow/Flux it never is, but it is kept
			// verbatim so the port stays comparable to the reference: it grows the
			// grid by one and skips the trailing 0 (the last sigma is the ~0 one).
			if (isclose_zero(flow.sigma(end), 1e-5)) {
				n += 1;
				append_zero = false;
			}
			std::vector<float> ts = linspace((double)start, (double)end, n);
			sigs.reserve(ts.size() + 1);
			for (float t : ts) sigs.push_back(flow.sigma(t));
		}
		if (append_zero) sigs.push_back(0.0f);
		break;
	}
	case SchedulerKind::Karras: {
		// get_sigmas_karras(n, sigma_min, sigma_max, rho=7):
		//   ramp = linspace(0, 1, n)
		//   (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** 7
		float sigma_min = 0.0f, sigma_max = 0.0f;
		model_sigma_bounds(flow, &sigma_min, &sigma_max);
		constexpr double rho = 7.0;
		const double min_inv_rho = std::pow((double)sigma_min, 1.0 / rho);
		const double max_inv_rho = std::pow((double)sigma_max, 1.0 / rho);
		const std::vector<float> ramp = linspace(0.0, 1.0, build_steps);
		sigs.reserve((size_t)build_steps + 1);
		for (float r : ramp) {
			const double v = max_inv_rho + (double)r * (min_inv_rho - max_inv_rho);
			sigs.push_back((float)std::pow(v, rho));
		}
		sigs.push_back(0.0f);
		break;
	}
	case SchedulerKind::Exponential: {
		// get_sigmas_exponential(n, sigma_min, sigma_max):
		//   linspace(log(sigma_max), log(sigma_min), n).exp()
		float sigma_min = 0.0f, sigma_max = 0.0f;
		model_sigma_bounds(flow, &sigma_min, &sigma_max);
		const std::vector<float> ts = linspace(std::log((double)sigma_max),
		                                       std::log((double)sigma_min), build_steps);
		sigs.reserve((size_t)build_steps + 1);
		for (float t : ts) sigs.push_back((float)std::exp((double)t));
		sigs.push_back(0.0f);
		break;
	}
	case SchedulerKind::DDIMUniform: {
		// ddim_scheduler: walk the *ascending* table in steps of len//steps,
		// collect, then reverse — the reversed list is the descending grid.
		// The head is a literal 0.0 unless the table's second entry is already
		// ~0, in which case there is no head and `steps` grows by one.
		const std::vector<float> table = model_sigma_table(flow, N);
		int nsteps = build_steps;
		std::vector<float> out;
		const int x0 = 1;  // the Python starts at index 1
		if (isclose_zero(table[x0], 1e-5)) {
			nsteps += 1;  // no 0.0 head: the reverse already ends near 0
		} else {
			out.push_back(0.0f);
		}
		const int ss = std::max((int)table.size() / nsteps, 1);
		for (int x = x0; x < (int)table.size(); x += ss) out.push_back(table[(size_t)x]);
		std::reverse(out.begin(), out.end());
		sigs = std::move(out);
		break;
	}
	case SchedulerKind::Beta: {
		// beta_scheduler(alpha=0.6, beta=0.6):
		//   ts = 1 - linspace(0, 1, steps, endpoint=False)
		//   t  = rint(beta.ppf(ts, .6, .6) * (len(table) - 1))
		//   keep table[t] only when t changes (consecutive dedup), then 0.0.
		// `total` is len(table) - 1, i.e. one below the table length: the ppf's
		// 1.0 maps onto the last *index*, not past the end.
		const std::vector<float> table = model_sigma_table(flow, N);
		const i64 total = (i64)table.size() - 1;
		sigs.reserve((size_t)build_steps + 1);
		i64 last = -1;
		for (int i = 0; i < build_steps; i++) {
			const double u = 1.0 - (double)i / (double)build_steps;
			const double q = beta_ppf(u, 0.6, 0.6);
			i64 t = rint_i64(q * (double)total);
			t = std::clamp<i64>(t, 0, total);
			if (t != last) sigs.push_back(table[(size_t)t]);
			last = t;
		}
		sigs.push_back(0.0f);
		break;
	}
	case SchedulerKind::LinearQuadratic: {
		// linear_quadratic_schedule(threshold_noise=0.025, linear_steps=steps//2):
		// a linear head, a quadratic tail and a closing 1.0, all mirrored by
		// (1 - x) and scaled by sigma_max. steps == 1 is its own [1.0, 0.0].
		float sigma_max = 0.0f;
		model_sigma_bounds(flow, nullptr, &sigma_max);
		constexpr double threshold_noise = 0.025;
		if (build_steps == 1) {
			sigs = {1.0f, 0.0f};
		} else {
			const int linear_steps = build_steps / 2;
			const int quadratic_steps = build_steps - linear_steps;
			const double diff =
			    (double)linear_steps - threshold_noise * (double)build_steps;
			const double quadratic_coef =
			    diff / ((double)linear_steps * (double)quadratic_steps * (double)quadratic_steps);
			const double linear_coef = threshold_noise / (double)linear_steps -
			                           2.0 * diff / ((double)quadratic_steps * (double)quadratic_steps);
			const double cst = quadratic_coef * ((double)linear_steps * (double)linear_steps);
			std::vector<double> sched;
			sched.reserve((size_t)build_steps + 1);
			for (int i = 0; i < linear_steps; i++)
				sched.push_back((double)i * threshold_noise / (double)linear_steps);
			for (int i = linear_steps; i < build_steps; i++)
				sched.push_back(quadratic_coef * ((double)i * (double)i) + linear_coef * (double)i +
				                cst);
			sched.push_back(1.0);
			sigs.resize(sched.size());
			for (size_t i = 0; i < sched.size(); i++) sigs[i] = (float)(1.0 - sched[i]);
		}
		// torch.FloatTensor(schedule) * sigma_max: the float32 product, not a
		// repeated multiply in double.
		for (float& s : sigs) s = s * sigma_max;
		break;
	}
	case SchedulerKind::KLOptimal: {
		// kl_optimal_scheduler(n, sigma_min, sigma_max):
		//   adj = arange(n) / (n - 1)
		//   tan(adj * atan(sigma_min) + (1 - adj) * atan(sigma_max))
		//   last entry 0.
		// n == 1 divides by zero in the reference (0/0 -> nan); the engine keeps
		// the grid valid instead and returns [sigma_max, 0] there.
		float sigma_min = 0.0f, sigma_max = 0.0f;
		model_sigma_bounds(flow, &sigma_min, &sigma_max);
		const std::vector<float> adj = linspace(0.0, 1.0, build_steps);
		const double a_min = std::atan((double)sigma_min);
		const double a_max = std::atan((double)sigma_max);
		sigs.assign((size_t)build_steps + 1, 0.0f);
		for (int i = 0; i < build_steps; i++) {
			const double a = (double)adj[(size_t)i];
			sigs[(size_t)i] = (float)std::tan(a * a_min + (1.0 - a) * a_max);
		}
		sigs[(size_t)build_steps] = 0.0f;
		break;
	}
	}

	// KSampler.set_steps' `sigmas[-(steps + 1):]` tail.
	if (trim) {
		const size_t keep = (size_t)steps + 1;
		if (sigs.size() > keep) sigs.erase(sigs.begin(), sigs.end() - (std::ptrdiff_t)keep);
	}
	return sigs;
}

}  // namespace phi::media
