// Snake activations — see snake.hpp for the reference correspondence.
#include "kernels/snake.hpp"

#include <cmath>

#include "kernels/parallel_for.hpp"

namespace phi::media::kernels {

void snake_f32(const float* x, float* y, i64 C, i64 L, const float* alpha, const float* beta) {
	if (C <= 0 || L <= 0) return;
	parallel_for(C, [&](i64 c) {
		const float a = alpha ? alpha[c] : 1.0f;
		const float b = beta ? beta[c] : 1.0f;
		// torch: (beta + 1e-9).reciprocal() — a rounded fp32 reciprocal
		const float r = 1.0f / (b + 1e-9f);
		const float* xp = x + c * L;
		float* yp = y + c * L;
		for (i64 j = 0; j < L; j++) {
			const float t = std::sin(a * xp[j]);
			yp[j] = (t * t) * r + xp[j];
		}
	});
}

void snake_beta_f32(const float* x, float* y, i64 C, i64 L, const float* alpha_log,
                    const float* beta_log) {
	if (C <= 0 || L <= 0) return;
	parallel_for(C, [&](i64 c) {
		const float a = std::exp(alpha_log ? alpha_log[c] : 0.0f);
		const float b = std::exp(beta_log ? beta_log[c] : 0.0f);
		const float r = 1.0f / (b + 1e-9f);
		const float* xp = x + c * L;
		float* yp = y + c * L;
		for (i64 j = 0; j < L; j++) {
			const float t = std::sin(a * xp[j]);
			yp[j] = (t * t) * r + xp[j];
		}
	});
}

void gelu_tanh_f32(float* y, i64 n) {
	// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))  — only ever used
	// on the encoder's 2048-wide `w0(x)` product, so a scalar loop is fine.
	const float k = 0.79788456080286535588f;  // sqrt(2/pi)
	parallel_for(n, [&](i64 i) {
		const float x = y[i];
		const float inner = k * (x + 0.044715f * x * x * x);
		y[i] = 0.5f * x * (1.0f + std::tanh(inner));
	});
}

}  // namespace phi::media::kernels
