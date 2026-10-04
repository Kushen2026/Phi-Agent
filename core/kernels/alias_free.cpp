// Anti-aliased resampling — see alias_free.hpp for the reference correspondence.
#include "kernels/alias_free.hpp"

#include <cmath>
#include <cstring>

#include "kernels/parallel_for.hpp"

namespace phi::media::kernels {

namespace {

constexpr double kPi = 3.14159265358979323846;

// I0(x) — modified Bessel function of the first kind, order 0. Cephes rational
// approximations (the same branch points numpy/torch use), evaluated in double.
double bessel_i0(double x) {
	x = std::fabs(x);
	if (x <= 15.0) {
		const double y = x * x;
		return 1.0 + y * (3.5156229 +
		                  y * (3.0899424 +
		                       y * (1.2067492 +
		                            y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
	}
	const double y = 3.75 / x;
	return (std::exp(x) / std::sqrt(x)) *
	       (0.39894228 +
	        y * (0.1328592e-1 +
	             y * (0.225319e-2 +
	                  y * (-0.157565e-2 +
	                       y * (0.916281e-2 +
	                            y * (-0.2057706e-1 +
	                                 y * (0.2635537e-1 +
	                                      y * (-0.1647633e-1 + y * 0.392377e-2))))))));
}

// torch.sinc(x) = sin(pi x) / (pi x)
double sinc(double x) {
	if (x == 0.0) return 1.0;
	const double p = kPi * x;
	return std::sin(p) / p;
}

}  // namespace

std::vector<float> kaiser_sinc_filter1d(float cutoff, float half_width, i64 kernel_size) {
	const bool even = (kernel_size % 2) == 0;
	const i64 half_size = kernel_size / 2;

	// kaiser window design (reference constants, in order)
	const double delta_f = 4.0 * (double)half_width;
	const double A = 2.285 * (double)(half_size - 1) * kPi * delta_f + 7.95;
	double beta = 0.0;
	if (A > 50.0) {
		beta = 0.1102 * (A - 8.7);
	} else if (A >= 21.0) {
		beta = 0.5842 * std::pow(A - 21.0, 0.4) + 0.07886 * (A - 21.0);
	}

	// torch.kaiser_window(n, beta, periodic=False):
	//   i0(beta*sqrt(1 - (2t/(n-1) - 1)^2)) / i0(beta),  t = 0..n-1
	const double denom = bessel_i0(beta);
	std::vector<double> window((size_t)kernel_size);
	for (i64 t = 0; t < kernel_size; t++) {
		double u = (kernel_size == 1) ? 0.0 : (2.0 * (double)t / (double)(kernel_size - 1) - 1.0);
		double arg = 1.0 - u * u;
		if (arg < 0.0) arg = 0.0;
		window[(size_t)t] = bessel_i0(beta * std::sqrt(arg)) / denom;
	}

	std::vector<float> filter((size_t)kernel_size);
	double sum = 0.0;
	for (i64 t = 0; t < kernel_size; t++) {
		const double time = even ? ((double)(t - half_size) + 0.5) : (double)(t - half_size);
		const double v = 2.0 * (double)cutoff * window[(size_t)t] * sinc(2.0 * (double)cutoff * time);
		filter[(size_t)t] = (float)v;
		sum += v;
	}
	const float inv = (float)(1.0 / sum);
	for (auto& v : filter) v *= inv;
	return filter;
}

// ── UpSample1d ─────────────────────────────────────────────────────────────

void Upsample1d::init(i64 ratio_, i64 kernel_size_) {
	ratio = ratio_;
	kernel_size = kernel_size_;
	pad = kernel_size / ratio - 1;
	pad_left = pad * ratio + (kernel_size - ratio) / 2;
	pad_right = pad * ratio + (kernel_size - ratio + 1) / 2;
	filter = kaiser_sinc_filter1d(0.5f / (float)ratio, 0.6f / (float)ratio, kernel_size);
}

void Upsample1d::set_filter(const float* f, i64 n) {
	if (n != kernel_size) {
		throw MediaError("Upsample1d::set_filter: expected " + std::to_string(kernel_size) +
		                 " coefficients, got " + std::to_string(n));
	}
	std::memcpy(filter.data(), f, sizeof(float) * (size_t)n);
}

i64 Upsample1d::out_len(i64 L) const {
	const i64 lp = L + 2 * pad;
	const i64 full = (lp - 1) * ratio + kernel_size;
	return full - pad_left - pad_right;
}

void Upsample1d::forward(const float* x, i64 C, i64 L, float* y) const {
	const i64 lp = L + 2 * pad;
	const i64 full = (lp - 1) * ratio + kernel_size;
	const i64 Lout = full - pad_left - pad_right;
	// x padded with the edge value (torch F.pad(mode="replicate"))
	auto xin = [&](const float* xp, i64 i) -> float {
		i64 s = i - pad;
		if (s < 0) s = 0;
		if (s > L - 1) s = L - 1;
		return xp[s];
	};
	parallel_for(C, [&](i64 c) {
		const float* xp = x + c * L;
		float* yp = y + c * Lout;
		for (i64 jj = 0; jj < Lout; jj++) {
			const i64 j = jj + pad_left;   // index into the (uncropped) conv output
			float acc = 0.0f;
			// y[j] = sum_k filter[k] * xpad[(j - k)/ratio], (j-k) % ratio == 0
			for (i64 k = 0; k < kernel_size; k++) {
				const i64 t = j - k;
				if (t < 0) break;                 // t decreases with k
				const i64 r = t % ratio;
				if (r != 0) continue;
				const i64 i = t / ratio;
				if (i >= lp) continue;
				acc += filter[(size_t)k] * xin(xp, i);
			}
			yp[jj] = acc * (float)ratio;
		}
	});
}

// ── LowPassFilter1d ────────────────────────────────────────────────────────

void LowPassFilter1d::init(float cutoff_, float half_width_, i64 stride_, i64 kernel_size_) {
	cutoff = cutoff_;
	half_width = half_width_;
	stride = stride_;
	kernel_size = kernel_size_;
	pad_left = kernel_size / 2 - (i64)(kernel_size % 2 == 0);
	pad_right = kernel_size / 2;
	filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size);
}

void LowPassFilter1d::set_filter(const float* f, i64 n) {
	if (n != kernel_size) {
		throw MediaError("LowPassFilter1d::set_filter: expected " + std::to_string(kernel_size) +
		                 " coefficients, got " + std::to_string(n));
	}
	std::memcpy(filter.data(), f, sizeof(float) * (size_t)n);
}

i64 LowPassFilter1d::out_len(i64 L) const {
	const i64 lp = L + pad_left + pad_right;
	return (lp - kernel_size) / stride + 1;
}

void LowPassFilter1d::forward(const float* x, i64 C, i64 L, float* y) const {
	const i64 Lout = out_len(L);
	parallel_for(C, [&](i64 c) {
		const float* xp = x + c * L;
		float* yp = y + c * Lout;
		for (i64 j = 0; j < Lout; j++) {
			float acc = 0.0f;
			for (i64 k = 0; k < kernel_size; k++) {
				i64 s = j * stride + k - pad_left;
				if (s < 0) s = 0;
				if (s > L - 1) s = L - 1;
				acc += filter[(size_t)k] * xp[s];
			}
			yp[j] = acc;
		}
	});
}

// ── Activation1d ───────────────────────────────────────────────────────────

void Activation1d::forward(const float* x, i64 C, i64 L, float* y, float* scratch) const {
	const i64 L2 = up.out_len(L);
	up.forward(x, C, L, scratch);
	snake.apply(scratch, scratch, C, L2);
	down.forward(scratch, C, L2, y);
}

}  // namespace phi::media::kernels
