// BigVGAN decoder + DAC encoder + attention posterior head (W3).
//
// Line-by-line port of comfy/ldm/minimax/audio_vae.py. Every module below names
// the python it mirrors; the evaluation order that matters for fp32 agreement is
// spelled out at each site (the reference's `t*t*r + x`, `xs.div_(3)`,
// `add_()` chains and `clamp_` are all reproduced literally rather than
// algebraically simplified).
//
// ── correspondence ─────────────────────────────────────────────────────────
//   snake / Snake1d / SnakeBeta        kernels/snake.cpp
//   kaiser_sinc_filter1d / UpSample1d
//   LowPassFilter1d / DownSample1d
//   Activation1d                       kernels/alias_free.cpp
//   Conv1dRef / ConvT1dRef             this file (torch conv1d / conv_transpose1d)
//   AmpBlock1Ref                       this file (AMPBlock1)
//   VocoderNet                         this file (BigVGAN)
//   ResidualUnitRef / EncoderBlockRef
//   DacEncoderNet                      this file (ResidualUnit / EncoderBlock / Encoder)
//   AttnProjectionNet                  this file (GeGluMlp / CausalAttention / AttnProjection)
//
// ── numeric notes ──────────────────────────────────────────────────────────
// * Weights are read out of the mmap'd checkpoint, which is fp32 throughout: no
//   conversion, no copies, no quantisation.
// * The reference's convs go through cuBLAS/MKL im2col; ours accumulate in fp32
//   with the (ic, k) order the weight layout implies. That is a ~1e-7 relative
//   difference per conv, i.e. far inside the 1e-4 rel-L2 budget, and it is
//   deterministic (see kernels/parallel_for.hpp).
// * `F.adaptive_avg_pool1d(256 -> 32)` divides exactly (window 8), so it is a
//   plain mean of 8 consecutive values; the code asserts that instead of
//   implementing the general adaptive rule.
#include "models/audio_vocoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "kernels/parallel_for.hpp"

namespace phi::media {

using kernels::parallel_for;

// ── loading helpers ────────────────────────────────────────────────────────

// A host fp32 view of a tensor, whatever the checkpoint stores: `data_f32` decodes
// a quantised source (int8 with its per-row scale, or one of the packed families)
// and converts a half/bfloat one, keeping the result alive in the reader's cache.
const float* borrow_f32(const SafeTensors& st, const std::string& name, i64 count) {
	const StTensor& t = st.require(name);
	if (t.numel != count) {
		throw MediaError("audio vae: '" + name + "' has " + std::to_string(t.numel) +
		                 " elements, expected " + std::to_string(count));
	}
	return st.data_f32(t);
}

namespace {

const float* optional_f32(const SafeTensors& st, const std::string& name, i64 count) {
	const StTensor* t = st.find(name);
	if (!t) return nullptr;
	if (t->numel != count) return nullptr;
	return st.data_f32(*t);
}

// y[m, n] = sum_k x[m, k] * w[n, k] (+ b[n]);  x [M, K], w [N, K] row major.
void linear_bias(const float* x, i64 m, i64 k, const float* w, const float* b, float* y, i64 n) {
	parallel_for(m, [&](i64 r) {
		const float* xr = x + r * k;
		float* yr = y + r * n;
		for (i64 o = 0; o < n; o++) {
			const float* wr = w + o * k;
			float acc = b ? b[o] : 0.0f;
			for (i64 i = 0; i < k; i++) acc += wr[i] * xr[i];
			yr[o] = acc;
		}
	});
}

// torch.nn.functional.layer_norm over the last dimension (biased variance).
void layer_norm_rows(const float* x, i64 rows, i64 cols, const float* w, const float* b, float eps,
                     float* y) {
	parallel_for(rows, [&](i64 r) {
		const float* xr = x + r * cols;
		float* yr = y + r * cols;
		float sum = 0.0f;
		for (i64 i = 0; i < cols; i++) sum += xr[i];
		const float mean = sum / (float)cols;
		float sq = 0.0f;
		for (i64 i = 0; i < cols; i++) {
			const float d = xr[i] - mean;
			sq += d * d;
		}
		const float inv = 1.0f / std::sqrt(sq / (float)cols + eps);
		for (i64 i = 0; i < cols; i++) {
			float v = (xr[i] - mean) * inv;
			yr[i] = w ? v * w[i] + (b ? b[i] : 0.0f) : v;
		}
	});
}

float gelu_tanh(float x) {
	const float k = 0.79788456080286535588f;  // sqrt(2/pi)
	const float inner = k * (x + 0.044715f * x * x * x);
	return 0.5f * x * (1.0f + std::tanh(inner));
}

}  // namespace

// ── Conv1d / ConvTranspose1d ───────────────────────────────────────────────

void Conv1dRef::forward(const float* x, i64 L, float* y, i64 Lout) const {
	// out[o][j] = b[o] + sum_{i,k} w[o][i][k] * x[i][j*stride + k*dilation - pad]
	const i64 jblk = 256;
	const i64 nj = std::max<i64>(1, (Lout + jblk - 1) / jblk);
	parallel_for(oc * nj, [&](i64 task) {
		const i64 o = task / nj;
		const i64 j0 = (task % nj) * jblk;
		const i64 j1 = std::min(Lout, j0 + jblk);
		if (j0 >= j1) return;
		float* out = y + o * Lout;
		const float bv = b ? b[o] : 0.0f;
		for (i64 j = j0; j < j1; j++) out[j] = bv;
		if (stride == 1) {
			for (i64 i = 0; i < ic; i++) {
				const float* xp = x + i * L;
				const float* wp = w + (o * ic + i) * k;
				for (i64 kk = 0; kk < k; kk++) {
					const float wk = wp[kk];
					if (wk == 0.0f) continue;
					const i64 base = kk * dilation - pad;   // x index at j = 0
					i64 lo = j0, hi = j1;
					if (lo < -base) lo = -base;
					if (hi > L - base) hi = L - base;
					const float* xs = xp + base;
					for (i64 j = lo; j < hi; j++) out[j] += wk * xs[j];
				}
			}
		} else {
			for (i64 j = j0; j < j1; j++) {
				float acc = out[j];
				for (i64 i = 0; i < ic; i++) {
					const float* xp = x + i * L;
					const float* wp = w + (o * ic + i) * k;
					for (i64 kk = 0; kk < k; kk++) {
						const i64 xi = j * stride + kk * dilation - pad;
						if (xi >= 0 && xi < L) acc += wp[kk] * xp[xi];
					}
				}
				out[j] = acc;
			}
		}
	});
}

void ConvT1dRef::forward(const float* x, i64 L, float* y, i64 Lout) const {
	// y[o][i*stride + k - pad] += w[i][o][k] * x[i][idx]   (torch ConvTranspose1d)
	parallel_for(oc, [&](i64 o) {
		float* out = y + o * Lout;
		const float bv = b ? b[o] : 0.0f;
		for (i64 j = 0; j < Lout; j++) out[j] = bv;
		for (i64 i = 0; i < ic; i++) {
			const float* xp = x + i * L;
			const float* wp = w + (i * oc + o) * k;
			for (i64 kk = 0; kk < k; kk++) {
				const float wk = wp[kk];
				if (wk == 0.0f) continue;
				const i64 base = kk - pad;   // output index at input 0
				// 0 <= idx*stride + base < Lout
				// The first input index that lands inside the output is ceil(-base/stride),
				// and getting that sign wrong (negating the ceil, which leaves `lo` at 0)
				// sends the store loop to `out + base` — up to `stride` floats *before*
				// the output buffer. That is a silent out-of-bounds write into the heap;
				// the allocator is what notices it, several frames later and in a
				// completely different function.
				i64 lo = 0, hi = L;
				if (base < 0) lo = std::max<i64>(lo, (-base + stride - 1) / stride);
				const i64 room = Lout - base;
				if (room <= 0) continue;
				hi = std::min<i64>(hi, (room + stride - 1) / stride);
				if (hi <= lo) continue;
				const i64 j0 = lo * stride + base;
				float* yp = out + j0;
				const float* xq = xp + lo;
				for (i64 n = 0; n < hi - lo; n++) yp[n * stride] += wk * xq[n];
			}
		}
	});
}

// ── AMPBlock1 ──────────────────────────────────────────────────────────────

void AmpBlock1Ref::forward(const float* x, i64 L, float* y, Scratch& sc) const {
	const i64 n = channels * L;
	std::memcpy(y, x, sizeof(float) * (size_t)n);
	const i64 na = (i64)acts.size();
	for (i64 it = 0; it * 2 + 1 < na; it++) {
		float* a = sc.get(0, n);
		float* b = sc.get(1, n);
		float* up = sc.get(2, 2 * n);
		acts[(size_t)(2 * it)].forward(y, channels, L, a, up);
		convs1[(size_t)it].forward(a, L, b, L);
		acts[(size_t)(2 * it + 1)].forward(b, channels, L, a, up);
		convs2[(size_t)it].forward(a, L, b, L);
		// reference: x = xt.add_(x)
		for (i64 i = 0; i < n; i++) y[i] += b[i];
	}
}

// ── BigVGAN ────────────────────────────────────────────────────────────────

void VocoderNet::load_act(const std::string& base, i64 channels,
                          kernels::Activation1d& a) const {
	a.init(channels);
	// The anti-aliasing coefficients live in the checkpoint; if a checkpoint
	// ships without them the formula's own output is used instead (the
	// conformance test proves the two agree to 2.6e-8).
	if (const float* f = optional_f32(*st_, base + ".upsample.filter", 12))
		a.up.set_filter(f, 12);
	if (const float* f = optional_f32(*st_, base + ".downsample.lowpass.filter", 12))
		a.down.set_filter(f, 12);
	a.snake.alpha_log = borrow_f32(*st_, base + ".act.alpha", channels);
	a.snake.beta_log = borrow_f32(*st_, base + ".act.beta", channels);
}

void VocoderNet::open(const SafeTensors& st, const std::string& prefix) {
	st_ = &st;
	prefix_ = prefix;
	// The H3 32 kHz configuration (BigVGAN defaults in the reference signature).
	rates_ = {5, 5, 2, 2, 2, 2, 2};
	kernels_ = {9, 9, 4, 4, 4, 4, 4};
	resblocks_.clear();
	ups_.clear();
	sc_.reset();

	const StTensor& cw = st.require(prefix_ + "conv_pre.weight");
	if (cw.shape.size() != 3) throw MediaError("audio vae: conv_pre.weight is not [OC,IC,K]");
	initial_ch_ = cw.shape[0];
	num_mels_ = cw.shape[1];
	conv_pre_.oc = cw.shape[0];
	conv_pre_.ic = cw.shape[1];
	conv_pre_.k = cw.shape[2];
	conv_pre_.stride = 1;
	conv_pre_.pad = (conv_pre_.k - 1) / 2;
	conv_pre_.w = borrow_f32(st, prefix_ + "conv_pre.weight", cw.numel);
	conv_pre_.b = borrow_f32(st, prefix_ + "conv_pre.bias", initial_ch_);

	if ((i64)rates_.size() != 7) throw MediaError("audio vae: internal rate table");
	i64 ch = initial_ch_;
	for (i64 i = 0; i < 7; i++) {
		const std::string base = prefix_ + "ups." + std::to_string(i) + ".0.";
		const StTensor& w = st.require(base + "weight");
		if (w.shape.size() != 3) throw MediaError("audio vae: " + base + "weight is not [IC,OC,K]");
		ConvT1dRef u;
		u.ic = w.shape[0];
		u.oc = w.shape[1];
		u.k = w.shape[2];
		u.stride = rates_[(size_t)i];
		u.pad = (u.k - u.stride) / 2;
		if (u.ic != ch || u.oc != ch / 2 || u.k != kernels_[(size_t)i]) {
			throw MediaError("audio vae: upsampler " + std::to_string(i) + " is [" +
			                 std::to_string(u.ic) + "," + std::to_string(u.oc) + "," +
			                 std::to_string(u.k) + "], expected [" + std::to_string(ch) + "," +
			                 std::to_string(ch / 2) + "," + std::to_string(kernels_[(size_t)i]) + "]");
		}
		u.w = st.data_f32(w);
		u.b = borrow_f32(st, base + "bias", ch / 2);
		ups_.push_back(u);
		ch = ch / 2;
	}

	// BigVGAN's two-level indexing, read off the checkpoint rather than assumed:
	// `resblocks.<level*3 + j>` is the j-th kernel variant of that level, and all
	// three convolutions inside one block share that kernel — only their *dilation*
	// differs. The dilation is not visible in the weights at all (it lives in the
	// padding), so the schedule is the reference's (1,3,5) and `get_padding()` below
	// is what turns it into the right receptive field. Getting the nesting backwards
	// (kernel per inner index instead of per block) is what the first run of this
	// loader failed on: "resblock 1 convs1.0 shape mismatch".
	const i64 res_k[3] = {3, 7, 11};
	const i64 res_d[3] = {1, 3, 5};
	for (i64 level = 0; level < 7; level++) {
		for (i64 j = 0; j < 3; j++) {
			const std::string base =
			    prefix_ + "resblocks." + std::to_string(level * 3 + j) + ".";
			const StTensor& w1 = st.require(base + "convs1.0.weight");
			const i64 cch = w1.shape[0];
			AmpBlock1Ref blk;
			blk.channels = cch;
			for (i64 t = 0; t < 3; t++) {
				const std::string cb = base + "convs1." + std::to_string(t) + ".";
				const StTensor& w = st.require(cb + "weight");
				Conv1dRef c;
				c.oc = w.shape[0];
				c.ic = w.shape[1];
				c.k = w.shape[2];
				c.stride = 1;
				c.dilation = res_d[t];
				c.pad = (c.k * c.dilation - c.dilation) / 2;   // get_padding()
				if (c.oc != cch || c.ic != cch || c.k != res_k[j]) {
					throw MediaError("audio vae: resblock " + std::to_string(level * 3 + j) +
					                 " convs1." + std::to_string(t) + " shape mismatch");
				}
				c.w = st.data_f32(w);
				c.b = borrow_f32(st, cb + "bias", cch);
				blk.convs1.push_back(c);

				const std::string c2b = base + "convs2." + std::to_string(t) + ".";
				const StTensor& w2 = st.require(c2b + "weight");
				Conv1dRef c2;
				c2.oc = w2.shape[0];
				c2.ic = w2.shape[1];
				c2.k = w2.shape[2];
				c2.stride = 1;
				c2.dilation = 1;
				c2.pad = (c2.k - 1) / 2;                        // get_padding(k, 1)
				if (c2.oc != cch || c2.ic != cch || c2.k != res_k[j]) {
					throw MediaError("audio vae: resblock " + std::to_string(level * 3 + j) +
					                 " convs2." + std::to_string(t) + " shape mismatch");
				}
				c2.w = st.data_f32(w2);
				c2.b = borrow_f32(st, c2b + "bias", cch);
				blk.convs2.push_back(c2);

				kernels::Activation1d a1, a2;
				load_act(base + "activations." + std::to_string(2 * t), cch, a1);
				load_act(base + "activations." + std::to_string(2 * t + 1), cch, a2);
				blk.acts.push_back(a1);
				blk.acts.push_back(a2);
			}
			resblocks_.push_back(blk);
		}
	}

	// load_act() appends ".act.alpha" / ".upsample.filter" itself, so the prefix
	// carries no trailing dot (the resblock call sites pass "activations.N").
	load_act(prefix_ + "activation_post", ch, act_post_);
	if (ch != 8) {
		throw MediaError("audio vae: conv_post input width is " + std::to_string(ch) +
		                 ", expected 8 (1024 / 2^7)");
	}
	const StTensor& pw = st.require(prefix_ + "conv_post.weight");
	conv_post_.oc = pw.shape[0];
	conv_post_.ic = pw.shape[1];
	conv_post_.k = pw.shape[2];
	conv_post_.stride = 1;
	conv_post_.pad = (conv_post_.k - 1) / 2;
	conv_post_.w = st.data_f32(pw);
	conv_post_.b = nullptr;   // bias=False
	if (conv_post_.ic != 8 || conv_post_.oc != 1) {
		throw MediaError("audio vae: conv_post is not [1,8,K]");
	}
}

i64 VocoderNet::out_len(i64 L) const {
	i64 n = L;
	for (i64 r : rates_) n *= r;
	return n;
}

void VocoderNet::forward(const float* mel, i64 L, std::vector<float>& pcm, StageSink* sink) const {
	std::vector<float> cur(mel, mel + (size_t)(num_mels_ * L));
	i64 cl = L, cc = num_mels_;

	// conv_pre
	{
		const i64 lo = conv_pre_.out_len(cl);
		std::vector<float> y((size_t)(conv_pre_.oc * lo));
		conv_pre_.forward(cur.data(), cl, y.data(), lo);
		cur.swap(y);
		cl = lo;
		cc = conv_pre_.oc;
		if (sink) sink->stage("@dec_conv_pre", cur.data(), cc, cl);
	}

	for (i64 level = 0; level < (i64)ups_.size(); level++) {
		const ConvT1dRef& u = ups_[(size_t)level];
		const i64 lu = u.out_len(cl);
		std::vector<float> xu((size_t)(u.oc * lu));
		u.forward(cur.data(), cl, xu.data(), lu);
		if (sink) sink->stage("@dec_up" + std::to_string(level), xu.data(), u.oc, lu);

		// xs = blk0(x); xs += blk1(x); xs += blk2(x); x = xs / 3
		const i64 ch = u.oc;
		std::vector<float> acc((size_t)(ch * lu));
		resblocks_[(size_t)(level * 3)].forward(xu.data(), lu, acc.data(), sc_);
		std::vector<float> tmp((size_t)(ch * lu));
		for (i64 j = 1; j < 3; j++) {
			resblocks_[(size_t)(level * 3 + j)].forward(xu.data(), lu, tmp.data(), sc_);
			for (i64 i = 0; i < ch * lu; i++) acc[(size_t)i] += tmp[(size_t)i];
		}
		for (i64 i = 0; i < ch * lu; i++) acc[(size_t)i] = acc[(size_t)i] / 3.0f;   // torch div_
		cur.swap(acc);
		cl = lu;
		cc = ch;
	}

	// activation_post
	{
		const i64 lo = act_post_.out_len(cl);
		std::vector<float> y((size_t)(cc * lo));
		std::vector<float> up((size_t)kernels::Activation1d::scratch_len(cc, cl));
		act_post_.forward(cur.data(), cc, cl, y.data(), up.data());
		cur.swap(y);
		cl = lo;
		if (sink) sink->stage("@dec_act_post", cur.data(), cc, cl);
	}

	// conv_post + clamp
	{
		const i64 lo = conv_post_.out_len(cl);
		pcm.assign((size_t)(conv_post_.oc * lo), 0.0f);
		conv_post_.forward(cur.data(), cl, pcm.data(), lo);
		if (sink) sink->stage("@dec_conv_post", pcm.data(), conv_post_.oc, lo);
		for (auto& v : pcm) v = std::min(1.0f, std::max(-1.0f, v));
		if (sink) sink->stage("@dec_pcm", pcm.data(), conv_post_.oc, lo);
	}
}

// ── encoder ────────────────────────────────────────────────────────────────

void ResidualUnitRef::forward(const float* x, i64 L, float* y, Scratch& sc) const {
	const i64 c = conv7.ic;
	const i64 n = c * L;
	float* s = sc.get(0, n);
	float* t = sc.get(1, n);
	float* u = sc.get(2, n);
	snake1.apply(x, s, c, L);
	conv7.forward(s, L, t, L);
	snake2.apply(t, u, c, L);
	conv1.forward(u, L, y, L);
	// reference: y.add_(x) (the dilated k=7 conv keeps the length, so the
	// centre-crop branch of ResidualUnit.forward never triggers here)
	for (i64 i = 0; i < n; i++) y[i] += x[i];
}

void EncoderBlockRef::forward(const float* x, i64 L, std::vector<float>& out, i64& out_L,
                              Scratch* scp) const {
	Scratch local;
	Scratch& sc = scp ? *scp : local;
	const i64 c = units[0].conv7.ic;
	std::vector<float> a((size_t)(c * L)), b((size_t)(c * L));
	units[0].forward(x, L, a.data(), sc);
	units[1].forward(a.data(), L, b.data(), sc);
	units[2].forward(b.data(), L, a.data(), sc);
	// sc.get(0..2) are reused by the next unit, so the snake needs its own copy
	std::vector<float> s((size_t)(c * L));
	snake.apply(a.data(), s.data(), c, L);
	const i64 lo = down.out_len(L);
	out.assign((size_t)(down.oc * lo), 0.0f);
	down.forward(s.data(), L, out.data(), lo);
	out_L = lo;
}

void DacEncoderNet::open(const SafeTensors& st, const std::string& prefix) {
	blocks_.clear();
	const StTensor& w0 = st.require(prefix + "block.0.weight");
	conv0_.oc = w0.shape[0];
	conv0_.ic = w0.shape[1];
	conv0_.k = w0.shape[2];
	conv0_.stride = 1;
	conv0_.pad = (conv0_.k - 1) / 2;
	conv0_.w = st.data_f32(w0);
	conv0_.b = borrow_f32(st, prefix + "block.0.bias", conv0_.oc);
	if (conv0_.ic != 1) throw MediaError("audio vae: encoder conv_in is not 1-channel");

	const i64 strides[5] = {2, 4, 4, 5, 5};
	i64 dim = conv0_.oc;
	for (i64 bi = 0; bi < 5; bi++) {
		const std::string base =
		    prefix + "block." + std::to_string(bi + 1) + ".block.";
		EncoderBlockRef blk;
		const i64 half = dim;   // ResidualUnit dim // 2 == the running width
		const i64 dil[3] = {1, 3, 9};
		for (i64 u = 0; u < 3; u++) {
			const std::string ub = base + std::to_string(u) + ".block.";
			ResidualUnitRef ru;
			ru.snake1.alpha = borrow_f32(st, ub + "0.alpha", half);
			const StTensor& c7 = st.require(ub + "1.weight");
			ru.conv7.oc = c7.shape[0];
			ru.conv7.ic = c7.shape[1];
			ru.conv7.k = c7.shape[2];
			ru.conv7.stride = 1;
			ru.conv7.dilation = dil[u];
			ru.conv7.pad = (ru.conv7.k * ru.conv7.dilation - ru.conv7.dilation) / 2;
			ru.conv7.w = st.data_f32(c7);
			ru.conv7.b = borrow_f32(st, ub + "1.bias", half);
			ru.snake2.alpha = borrow_f32(st, ub + "2.alpha", half);
			const StTensor& c1 = st.require(ub + "3.weight");
			ru.conv1.oc = c1.shape[0];
			ru.conv1.ic = c1.shape[1];
			ru.conv1.k = c1.shape[2];
			ru.conv1.stride = 1;
			ru.conv1.dilation = 1;
			ru.conv1.pad = 0;
			ru.conv1.w = st.data_f32(c1);
			ru.conv1.b = borrow_f32(st, ub + "3.bias", half);
			if (ru.conv7.ic != half || ru.conv7.oc != half || ru.conv1.ic != half) {
				throw MediaError("audio vae: encoder residual unit " + std::to_string(bi) + "/" +
				                 std::to_string(u) + " width mismatch (expected " +
				                 std::to_string(half) + ")");
			}
			blk.units.push_back(ru);
		}
		blk.snake.alpha = borrow_f32(st, base + "3.alpha", half);
		const i64 stride = strides[bi];
		const StTensor& dw = st.require(base + "4.weight");
		blk.down.oc = dw.shape[0];
		blk.down.ic = dw.shape[1];
		blk.down.k = dw.shape[2];
		blk.down.stride = stride;
		blk.down.pad = (stride + 1) / 2;   // math.ceil(stride / 2)
		blk.down.w = st.data_f32(dw);
		blk.down.b = borrow_f32(st, base + "4.bias", blk.down.oc);
		if (blk.down.ic != half || blk.down.oc != half * 2 || blk.down.k != 2 * stride) {
			throw MediaError("audio vae: encoder block " + std::to_string(bi) +
			                 " downsample shape mismatch");
		}
		blocks_.push_back(std::move(blk));
		dim = dim * 2;
	}

	snake_.alpha = borrow_f32(st, prefix + "block.6.alpha", dim);
	const StTensor& w7 = st.require(prefix + "block.7.weight");
	conv7_.oc = w7.shape[0];
	conv7_.ic = w7.shape[1];
	conv7_.k = w7.shape[2];
	conv7_.stride = 1;
	conv7_.pad = (conv7_.k - 1) / 2;
	conv7_.w = st.data_f32(w7);
	conv7_.b = borrow_f32(st, prefix + "block.7.bias", conv7_.oc);
	if (conv7_.ic != dim || conv7_.oc != dim) {
		throw MediaError("audio vae: encoder latent width mismatch");
	}
}

i64 DacEncoderNet::out_len(i64 L) const {
	i64 n = L;
	for (const auto& b : blocks_) n = b.down.out_len(n);
	return n;
}

void DacEncoderNet::forward(const float* x, i64 L, std::vector<float>& out, i64& Tout,
                            StageSink* sink) const {
	Scratch sc;
	const i64 lo = conv0_.out_len(L);
	std::vector<float> cur((size_t)(conv0_.oc * lo), 0.0f);
	conv0_.forward(x, L, cur.data(), lo);
	i64 cl = lo, cc = conv0_.oc;
	if (sink) sink->stage("@enc_block0", cur.data(), cc, cl);
	for (size_t bi = 0; bi < blocks_.size(); bi++) {
		std::vector<float> y;
		i64 yl = 0;
		blocks_[bi].forward(cur.data(), cl, y, yl, &sc);
		cur.swap(y);
		cl = yl;
		cc = blocks_[bi].down.oc;
		if (sink) sink->stage("@enc_block" + std::to_string(bi + 1), cur.data(), cc, cl);
	}
	{
		std::vector<float> y((size_t)(cc * cl));
		snake_.apply(cur.data(), y.data(), cc, cl);
		cur.swap(y);
		if (sink) sink->stage("@enc_block6", cur.data(), cc, cl);
	}
	{
		const i64 l7 = conv7_.out_len(cl);
		std::vector<float> y((size_t)(conv7_.oc * l7), 0.0f);
		conv7_.forward(cur.data(), cl, y.data(), l7);
		cur.swap(y);
		cl = l7;
		cc = conv7_.oc;
		if (sink) sink->stage("@enc_block7", cur.data(), cc, cl);
	}
	out.swap(cur);
	Tout = cl;
}

// ── CausalAttention / AttnProjection ───────────────────────────────────────

void AttnProjectionNet::open(const SafeTensors& st, const std::string& prefix) {
	st_ = &st;
	prefix_ = prefix;
	const std::string p = prefix;
	const StTensor& qkv = st.require(p + "attn.qkv.weight");
	in_dim_ = qkv.shape[1];
	if (qkv.shape[0] != 3 * in_dim_) throw MediaError("audio vae: pre_block qkv is not [3C, C]");
	const StTensor& proj = st.require(p + "proj.weight");
	out_dim_ = proj.shape[0];
	if (proj.shape[1] != in_dim_) throw MediaError("audio vae: pre_block proj shape mismatch");
	heads_ = 8;   // AttnProjection(latent_dim, vae_latent_channels, num_heads=8)
	if (in_dim_ % heads_) throw MediaError("audio vae: pre_block heads do not divide in_dim");
	const StTensor& w0 = st.require(p + "mlp.w0.weight");
	hidden_ = w0.shape[0];
	if (w0.shape[1] != out_dim_ || hidden_ != out_dim_ * 2) {
		throw MediaError("audio vae: pre_block mlp is not GeGluMlp(out_dim, 2*out_dim)");
	}
	n1w_ = borrow_f32(st, p + "norm1.weight", in_dim_);
	n1b_ = borrow_f32(st, p + "norm1.bias", in_dim_);
	n3w_ = borrow_f32(st, p + "norm3.weight", in_dim_);
	n3b_ = borrow_f32(st, p + "norm3.bias", in_dim_);
	proj_w_ = st.data_f32(proj);
	proj_b_ = borrow_f32(st, p + "proj.bias", out_dim_);
	qkv_w_ = st.data_f32(qkv);
	q_bias_ = borrow_f32(st, p + "attn.q_bias", in_dim_);
	v_bias_ = borrow_f32(st, p + "attn.v_bias", in_dim_);
	// zero_k_bias is a zero buffer in the reference (the k projection is biasless)
	const StTensor& zb = st.require(p + "attn.zero_k_bias");
	for (i64 i = 0; i < zb.numel; i++) {
		const float v = (st.data_f32(zb))[i];
		if (v != 0.0f) throw MediaError("audio vae: pre_block zero_k_bias is not zero");
	}
	const StTensor& apw = st.require(p + "attn.proj.weight");
	if (apw.shape[0] != out_dim_ || apw.shape[1] != out_dim_) {
		throw MediaError("audio vae: pre_block attn.proj shape mismatch");
	}
	attn_proj_w_ = st.data_f32(apw);
	attn_proj_b_ = borrow_f32(st, p + "attn.proj.bias", out_dim_);
	n2w_ = borrow_f32(st, p + "norm2.weight", out_dim_);
	n2b_ = borrow_f32(st, p + "norm2.bias", out_dim_);
	mnorm_w_ = borrow_f32(st, p + "mlp.norm.weight", out_dim_);
	mnorm_b_ = borrow_f32(st, p + "mlp.norm.bias", out_dim_);
	w0_ = st.data_f32(w0);
	w0b_ = borrow_f32(st, p + "mlp.w0.bias", hidden_);
	const StTensor& w1 = st.require(p + "mlp.w1.weight");
	w1_ = st.data_f32(w1);
	w1b_ = borrow_f32(st, p + "mlp.w1.bias", hidden_);
	const StTensor& w2 = st.require(p + "mlp.w2.weight");
	w2_ = st.data_f32(w2);
	w2b_ = borrow_f32(st, p + "mlp.w2.bias", out_dim_);
}

void AttnProjectionNet::forward(const float* x, i64 T, std::vector<float>& y) const {
	const i64 ci = in_dim_, co = out_dim_, hd = in_dim_ / heads_;
	// x = proj(norm3(x)) + attn(norm1(x))
	std::vector<float> n3((size_t)(T * ci)), base((size_t)(T * co));
	layer_norm_rows(x, T, ci, n3w_, n3b_, ln_eps_, n3.data());
	linear_bias(n3.data(), T, ci, proj_w_, proj_b_, base.data(), co);

	std::vector<float> n1((size_t)(T * ci));
	layer_norm_rows(x, T, ci, n1w_, n1b_, ln_eps_, n1.data());

	// qkv = F.linear(n1, qkv_w, cat(q_bias, zero_k_bias, v_bias))
	std::vector<float> qkv((size_t)(T * 3 * ci));
	parallel_for(T, [&](i64 r) {
		const float* xr = n1.data() + r * ci;
		float* yr = qkv.data() + r * 3 * ci;
		for (i64 o = 0; o < 3 * ci; o++) {
			const float* wr = qkv_w_ + o * ci;
			float acc = (o < ci) ? q_bias_[o] : ((o >= 2 * ci) ? v_bias_[o - 2 * ci] : 0.0f);
			for (i64 i = 0; i < ci; i++) acc += wr[i] * xr[i];
			yr[o] = acc;
		}
	});

	// scaled_dot_product_attention(q, k, v, is_causal=True), then the mean over
	// heads and the pool of head_dim -> out_dim (adaptive_avg_pool1d).
	const float scale = 1.0f / std::sqrt((float)hd);
	std::vector<float> att((size_t)(T * ci));
	parallel_for(T, [&](i64 i) {
		std::vector<float> scores((size_t)T);
		std::vector<float> row((size_t)ci, 0.0f);
		for (i64 h = 0; h < heads_; h++) {
			const float* q = qkv.data() + (size_t)i * 3 * ci + (size_t)h * hd;
			float mx = -INFINITY;
			for (i64 j = 0; j <= i; j++) {
				const float* k = qkv.data() + (size_t)j * 3 * ci + ci + (size_t)h * hd;
				float s = 0.0f;
				for (i64 d = 0; d < hd; d++) s += q[d] * k[d];
				s *= scale;
				scores[(size_t)j] = s;
				mx = std::max(mx, s);
			}
			float sum = 0.0f;
			for (i64 j = 0; j <= i; j++) {
				const float e = std::exp(scores[(size_t)j] - mx);
				scores[(size_t)j] = e;
				sum += e;
			}
			const float inv = 1.0f / sum;
			for (i64 d = 0; d < hd; d++) {
				float acc = 0.0f;
				for (i64 j = 0; j <= i; j++) {
					const float* v = qkv.data() + (size_t)j * 3 * ci + 2 * ci + (size_t)h * hd;
					acc += scores[(size_t)j] * v[d];
				}
				row[(size_t)(h * hd + d)] = acc * inv;
			}
		}
		// mean over heads (torch.mean(dim=1)) then the pool over head_dim
		float* ar = att.data() + (size_t)i * ci;
		const i64 win = hd / co;
		if (win * co != hd) throw MediaError("audio vae: pre_block pool is not exact");
		for (i64 o = 0; o < co; o++) {
			float acc = 0.0f;
			for (i64 w01 = 0; w01 < win; w01++) {
				for (i64 h = 0; h < heads_; h++)
					acc += row[(size_t)(h * hd + o * win + w01)];
			}
			ar[o] = acc / (float)(win * heads_);
		}
	});

	std::vector<float> ap((size_t)(T * co));
	linear_bias(att.data(), T, co, attn_proj_w_, attn_proj_b_, ap.data(), co);
	for (size_t i = 0; i < base.size(); i++) base[i] += ap[i];

	// x += mlp(norm2(x))
	std::vector<float> nm((size_t)(T * co));
	layer_norm_rows(base.data(), T, co, n2w_, n2b_, ln_eps_, nm.data());
	std::vector<float> h0((size_t)(T * hidden_)), h1((size_t)(T * hidden_));
	linear_bias(nm.data(), T, co, w0_, w0b_, h0.data(), hidden_);
	for (auto& v : h0) v = gelu_tanh(v);
	linear_bias(nm.data(), T, co, w1_, w1b_, h1.data(), hidden_);
	for (size_t i = 0; i < h0.size(); i++) h0[i] *= h1[i];
	std::vector<float> mo((size_t)(T * co));
	linear_bias(h0.data(), T, hidden_, w2_, w2b_, mo.data(), co);
	for (size_t i = 0; i < base.size(); i++) base[i] += mo[i];
	y.swap(base);
}

// ── the whole audio VAE ────────────────────────────────────────────────────

void AudioVaeNet::open(const SafeTensors& st) {
	st_ = &st;
	dec_.open(st, "decoder.");
	latent_dim_ = dec_.num_mels();   // 2048 == dec_in_proj output == conv_pre input

	const StTensor& dw = st.require("dec_in_proj.weight");
	z_ch_ = dw.shape[1];
	dec_in_proj_.oc = dw.shape[0];
	dec_in_proj_.ic = dw.shape[1];
	dec_in_proj_.k = dw.shape[2];
	dec_in_proj_.stride = 1;
	dec_in_proj_.pad = (dec_in_proj_.k - 1) / 2;
	dec_in_proj_.w = st.data_f32(dw);
	dec_in_proj_.b = borrow_f32(st, "dec_in_proj.bias", dec_in_proj_.oc);
	if (dec_in_proj_.oc != latent_dim_) {
		throw MediaError("audio vae: dec_in_proj does not produce the decoder's input width");
	}

	lat_mean_ = std::vector<float>(borrow_f32(st, "latents_mean", z_ch_),
	                               borrow_f32(st, "latents_mean", z_ch_) + z_ch_);
	lat_std_ = std::vector<float>(borrow_f32(st, "latents_std", z_ch_),
	                              borrow_f32(st, "latents_std", z_ch_) + z_ch_);
	for (float v : lat_std_) {
		if (!(v > 0.0f)) throw MediaError("audio vae: latents_std must be positive");
	}

	auto load_1x1 = [&](const std::string& name, i64 ch, Conv1dRef& c) {
		const StTensor& w = st.require(name);
		if (w.shape.size() != 3 || w.shape[0] != ch || w.shape[1] != ch || w.shape[2] != 1) {
			throw MediaError("audio vae: " + name + " is not a 1x1 conv of width " +
			                 std::to_string(ch));
		}
		c.oc = c.ic = ch;
		c.k = 1;
		c.stride = 1;
		c.pad = 0;
		c.w = st.data_f32(w);
		c.b = borrow_f32(st, name.substr(0, name.size() - 7) + ".bias", ch);
	};
	load_1x1("mean_proj.weight", z_ch_, mean_proj_);
	load_1x1("logs_proj.weight", z_ch_, logs_proj_);   // unused at inference

	enc_.open(st, "encoder.");
	pre_.open(st, "pre_block.");
	if (pre_.in_dim() != latent_dim_ || pre_.out_dim() != z_ch_) {
		throw MediaError("audio vae: pre_block is not [" + std::to_string(latent_dim_) + " -> " +
		                 std::to_string(z_ch_) + "]");
	}
}

void AudioVaeNet::decode_channel(const float* z, i64 T, std::vector<float>& pcm,
                                 StageSink* sink) const {
	const i64 c = z_ch_;
	// reference: z = z * std + mean  (broadcast [1, C, 1] over T)
	std::vector<float> x((size_t)(c * T));
	for (i64 ci = 0; ci < c; ci++) {
		const float s = lat_std_[(size_t)ci], m = lat_mean_[(size_t)ci];
		for (i64 t = 0; t < T; t++) x[(size_t)(ci * T + t)] = z[(size_t)(ci * T + t)] * s + m;
	}
	const i64 l1 = dec_in_proj_.out_len(T);
	std::vector<float> y((size_t)(dec_in_proj_.oc * l1), 0.0f);
	dec_in_proj_.forward(x.data(), T, y.data(), l1);
	if (sink) sink->stage("@dec_in_proj", y.data(), dec_in_proj_.oc, l1);
	dec_.forward(y.data(), l1, pcm, sink);
}

std::vector<float> AudioVaeNet::decode_stereo(const float* z, i64 T, StageSink* sink) const {
	std::vector<float> left, right;
	decode_channel(z, T, left, sink);
	decode_channel(z + (size_t)z_ch_ * (size_t)T, T, right, sink);
	std::vector<float> out((size_t)(2 * (i64)left.size()));
	std::copy(left.begin(), left.end(), out.begin());
	std::copy(right.begin(), right.end(), out.begin() + (i64)left.size());
	return out;
}

std::vector<float> AudioVaeNet::encode_channel(const float* pcm, i64 n, StageSink* sink) const {
	// reference: right-pad with zeros to a multiple of hop_length, then encode
	if (hop_ <= 0) throw MediaError("audio vae: hop must be positive");
	const i64 t_frames = (n + hop_ - 1) / hop_;
	const i64 padded = t_frames * hop_;
	std::vector<float> w((size_t)std::max<i64>(padded, 1), 0.0f);
	if (n > 0) std::memcpy(w.data(), pcm, sizeof(float) * (size_t)n);

	std::vector<float> feat;
	i64 t_out = 0;
	enc_.forward(w.data(), padded, feat, t_out, sink);
	if (t_out != t_frames) {
		throw MediaError("audio vae: encoder produced " + std::to_string(t_out) +
		                 " frames for " + std::to_string(padded) + " samples (expected " +
		                 std::to_string(t_frames) + ")");
	}
	// pre_block works on [T, C]
	std::vector<float> tc((size_t)(t_frames * latent_dim_));
	for (i64 ci = 0; ci < latent_dim_; ci++)
		for (i64 t = 0; t < t_frames; t++)
			tc[(size_t)(t * latent_dim_ + ci)] = feat[(size_t)(ci * t_frames + t)];
	std::vector<float> head;
	pre_.forward(tc.data(), t_frames, head);
	if (sink) sink->stage("@enc_pre_block", head.data(), t_frames, pre_.out_dim());

	// back to [C, T] then mean_proj then the (z - mean) / std normalization
	std::vector<float> ct((size_t)(z_ch_ * t_frames));
	for (i64 t = 0; t < t_frames; t++)
		for (i64 ci = 0; ci < z_ch_; ci++)
			ct[(size_t)(ci * t_frames + t)] = head[(size_t)(t * z_ch_ + ci)];
	std::vector<float> z((size_t)(z_ch_ * t_frames), 0.0f);
	mean_proj_.forward(ct.data(), t_frames, z.data(), t_frames);
	if (sink) sink->stage("@enc_mean_proj", z.data(), z_ch_, t_frames);
	for (i64 ci = 0; ci < z_ch_; ci++) {
		const float m = lat_mean_[(size_t)ci], s = lat_std_[(size_t)ci];
		for (i64 t = 0; t < t_frames; t++) {
			size_t k = (size_t)(ci * t_frames + t);
			z[k] = (z[k] - m) / s;
		}
	}
	if (sink) sink->stage("@enc_z", z.data(), z_ch_, t_frames);
	return z;
}

std::vector<float> AudioVaeNet::encode_stereo(const float* pcm, i64 n) const {
	std::vector<float> l = encode_channel(pcm, n, nullptr);
	std::vector<float> r = encode_channel(pcm + n, n, nullptr);
	const i64 t = (i64)(l.size() / (size_t)z_ch_);
	std::vector<float> out((size_t)(z_ch_ * 2 * t));
	std::copy(l.begin(), l.end(), out.begin());
	std::copy(r.begin(), r.end(), out.begin() + (i64)l.size());
	return out;
}

}  // namespace phi::media
