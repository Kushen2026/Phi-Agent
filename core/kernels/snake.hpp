// Snake activations for the H3 audio VAE (W3).
//
// Ported from comfy/ldm/minimax/audio_vae.py:
//
//   def snake(x, alpha, beta):
//       t = torch.sin(alpha * x)
//       return t.mul_(t).mul_((beta + 1e-9).reciprocal()).add_(x)
//
// i.e.  y = x + sin(alpha*x)^2 / (beta + 1e-9).  The evaluation order
// ((t*t)*r + x, r = 1/(beta+1e-9) rounded to fp32) is reproduced literally: the
// 1e-9 is part of the reference definition, not a guard, and `Snake1d` reuses
// alpha as beta, which for small alpha makes it numerically visible.
//
// Layout: x/y are [C, L] row major (torch's [1, C, L] flattened, the batch is
// always 1 = one stereo channel). alpha/beta are [C].
#pragma once

#include "util/media_common.hpp"

namespace phi::media::kernels {

// y = x + sin(alpha*x)^2 / (beta + 1e-9).  `alpha`/`beta` may be null (= 1) and
// may alias each other (Snake1d).  x and y may alias.
void snake_f32(const float* x, float* y, i64 C, i64 L, const float* alpha, const float* beta);

// SnakeBeta: the stored parameters are in log space (reference applies exp()).
void snake_beta_f32(const float* x, float* y, i64 C, i64 L, const float* alpha_log,
                    const float* beta_log);

// nn.GELU(approximate="tanh"), used by the encoder's GeGluMlp.
void gelu_tanh_f32(float* y, i64 n);

// ── the two activation modules ─────────────────────────────────────────────

struct Snake1dAct {           // encoder side: alpha is [1, C, 1]
	const float* alpha = nullptr;
	void apply(const float* x, float* y, i64 C, i64 L) const {
		snake_f32(x, y, C, L, alpha, alpha);
	}
};

struct SnakeBetaAct {         // decoder side: log-scale alpha/beta, both [C]
	const float* alpha_log = nullptr;
	const float* beta_log = nullptr;
	void apply(const float* x, float* y, i64 C, i64 L) const {
		snake_beta_f32(x, y, C, L, alpha_log, beta_log);
	}
};

}  // namespace phi::media::kernels
