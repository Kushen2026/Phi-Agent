// ACE-Step 1.5 audio VAE decoder.
//
// ── what the checkpoint is ────────────────────────────────────────────────
//
// `ace_1.5_vae.safetensors` (bf16, 337 MB) holds an encoder/decoder pair whose
// tensor names are the stable-audio-tools `OobleckVAE` (`comfy/ldm/audio/
// autoencoder.py::AudioOobleckVAE`), reached from ComfyUI through the
// `"decoder.layers.1.layers.0.beta" in sd` branch of `comfy/sd.py`:
//
//     AudioOobleckVAE(in_channels=2, channels=128, latent_dim=64,
//                     c_mults=[1,2,4,8,16], strides=[2,4,4,6,10],
//                     use_snake=True, final_tanh=False)
//
// which `comfy/sd.py` selects from the *kernel width of the second upsampler*
// (`decoder.layers.2.layers.1.weight_v` is [1024, 512, 12] -> stride 6). The
// strides are what make the 25 Hz latent of `comfy.latent_formats.ACEAudio15`
// (latent_channels 64) turn into 1920 samples per frame, i.e. 48 kHz.
//
//   decoder.layers.0            WNConv1d(64 -> 2048, k=7, pad=3)
//   decoder.layers.1..5         DecoderBlock: SnakeBeta(in),
//                               WNConvTranspose1d(in, out, k=2r, stride=r,
//                               pad=ceil(r/2)), then 3 residual units
//                               (dilation 1, 3, 9)
//   decoder.layers.6            SnakeBeta(128)
//   decoder.layers.7            WNConv1d(128 -> 2, k=7, pad=3, no bias)
//
// `DecoderBlock`/`ResidualUnit` are DAC's (`comfy/ldm/minimax/audio_vae.py` has
// the same shapes): a residual unit is SnakeBeta, WNConv1d(k=7, dilated, pad
// 3*d), SnakeBeta, WNConv1d(k=1), plus the identity. The activation is
// `SnakeBeta` (log-scale alpha/beta in the file, exp() at use), not plain
// `Snake1d`; `kernels::snake_beta_f32` is the reference port of it.
//
// ── memory: why the decode is chunked ─────────────────────────────────────
//
// The decoder is *expansive*: a stage's activation is
// `channels * frames * hop`, which grows 100x from the latent to the output
// (2048xT -> 128x1920T). Decoding a 60 s clip in one pass would hold ~2 GB per
// buffer at the last two stages. So the latent is decoded in chunks of
// `chunk_frames_` frames, each with a halo of `halo_frames_` latent frames on
// both sides taken from the real neighbours, and the output chunk is cropped
// back out.
//
// The halo is not a heuristic: the decoder's receptive field per side, summed
// over the stages and expressed in latent frames (the local field in output
// samples divided by the stride that follows it), is
//
//     conv_in 3 + block1 (19 + 39)/10 + block2 (12 + 39)/6 ...
//     = 3 + 5.8 + 8.5*... ~= 9.1 latent frames
//
// so 16 frames of halo carry every output sample's *whole* input set - which is
// what makes the chunked result identical to the unchunked one, element for
// element: `Conv1dRef` (and the `conv1d_gather` kernel that mirrors it) produces
// each output element from its own window, in the same accumulation order, and
// the window is complete. Nothing in the net carries state across a chunk, so
// the property is the same on both paths. Only the true sequence ends see the
// reference's zero padding.
#include "models/ace_vae.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "kernels/parallel_for.hpp"

namespace phi::media {

namespace {

std::vector<float> f32_of(const SafeTensors& st, const std::string& name) {
	return tensor_to_f32(st, st.require(name));
}

// torch.utils.parametrizations.weight_norm (dim=0) folding:
//     w[o] = v[o] * (g[o] / ||v[o]||_2)
// The multiply-by-(g/norm) order is torch's own `_weight_norm`, and it matters
// at the last ulp: `(v * g) / norm` rounds twice on a different path.
std::vector<float> fold_weight_norm(const SafeTensors& st, const std::string& base, i64 oc,
                                    i64 rest) {
	const StTensor& g = st.require(base + ".weight_g");
	const StTensor& v = st.require(base + ".weight_v");
	if (g.numel != oc || v.numel != oc * rest)
		throw MediaError("ace vae: '" + base + "' is not weight_norm-shaped for this net");
	std::vector<float> gv = f32_of(st, base + ".weight_g");
	std::vector<float> vv = f32_of(st, base + ".weight_v");
	std::vector<float> w(vv.size());
	kernels::parallel_for(oc, [&](i64 o) {
		const float* vr = vv.data() + (size_t)o * (size_t)rest;
		float* wr = w.data() + (size_t)o * (size_t)rest;
		double acc = 0.0;
		for (i64 i = 0; i < rest; i++) acc += (double)vr[i] * (double)vr[i];
		const double nrm = std::sqrt(acc);
		const double s = nrm > 0.0 ? (double)gv[(size_t)o] / nrm : 0.0;
		for (i64 i = 0; i < rest; i++) wr[i] = (float)((double)vr[i] * s);
	});
	return w;
}

bool has(const SafeTensors& st, const std::string& n) { return st.find(n) != nullptr; }

}  // namespace

// ── the module tree ────────────────────────────────────────────────────────
//
// Every module carries its folded weights twice over: the host fp32 vectors in
// `store_` (the reference and the fallback) and, when the net was opened with a
// device, `GpuAlloc`s filled by `upload_device_weights` at decode time. The
// device handles live next to the host ones here so the decode has no lookup to
// do.

struct AceVae::ResidualUnit {          // SnakeBeta, Conv7(dilated), SnakeBeta, Conv1
	kernels::SnakeBetaAct s1, s2;
	Conv1dRef c7, c1;
	GpuAlloc s1_a, s1_b, s2_a, s2_b;   // exp()'d alpha/beta, [oc] fp32
	GpuAlloc c7_w, c7_b, c1_w, c1_b;
};

struct AceVae::Block {                 // SnakeBeta, ConvTranspose1d, 3 units
	kernels::SnakeBetaAct snake;
	ConvT1dRef up;
	GpuAlloc snake_a, snake_b;
	GpuAlloc up_w, up_b;
	std::vector<ResidualUnit> units;
};

AceVae::AceVae() = default;
AceVae::~AceVae() = default;

void AceVae::open(const std::string& path, GpuCtx* gpu) {
	// The folded weights are built on the host either way: they are the reference
	// the device kernels are checked against, and `store_` is where all of them
	// live (both paths read the same fp32 values).
	path_ = path;
	st_owned_.open(path);
	st_ = &st_owned_;
	const SafeTensors& st = *st_;
	// The device is not touched until the whole tree has been loaded and the
	// folded weights exist - `dev_ready_` is what `decode_raw_gpu` needs, and
	// `upload_device_weights` is what puts them on the card (see there for why the
	// upload is not done here).
	const bool have_gpu = gpu && gpu->ok() && gpu->keep != nullptr;
	g_ = have_gpu ? gpu : nullptr;
	dev_ready_ = false;

	// ── structure, read off the names ──
	const StTensor& cin = st.require("decoder.layers.0.weight_v");
	latent_ = cin.shape[1];
	i64 ch = cin.shape[0];

	store_.clear();
	store_.reserve(256);

	auto conv = [&](const std::string& base, Conv1dRef& c, i64 oc, i64 ic, i64 k, i64 stride,
	                i64 pad, i64 dilation) {
		std::vector<float> w = fold_weight_norm(st, base, oc, ic * k);
		const StTensor* bt = st.find(base + ".bias");
		std::vector<float> b;
		if (bt) {
			b = f32_of(st, base + ".bias");
			if ((i64)b.size() != oc)
				throw MediaError("ace vae: '" + base + ".bias' is not " + std::to_string(oc));
		}
		c.ic = ic;
		c.oc = oc;
		c.k = k;
		c.stride = stride;
		c.pad = pad;
		c.dilation = dilation;
		c.w = store_.emplace_back(std::move(w)).data();
		c.b = b.empty() ? nullptr : store_.emplace_back(std::move(b)).data();
		if (!c.b && bt)
			throw MediaError("ace vae: bias storage went missing for '" + base + "'");
	};
	auto snake = [&](const std::string& base, kernels::SnakeBetaAct& a, i64 c) {
		std::vector<float> al = f32_of(st, base + ".alpha");
		std::vector<float> be = f32_of(st, base + ".beta");
		if ((i64)al.size() != c || (i64)be.size() != c)
			throw MediaError("ace vae: '" + base + "' alpha/beta is not " + std::to_string(c));
		a.alpha_log = store_.emplace_back(std::move(al)).data();
		a.beta_log = store_.emplace_back(std::move(be)).data();
	};
	// ── decoder.layers.0: the latent projection ──
	if (cin.shape.size() != 3) throw MediaError("ace vae: decoder.layers.0 is not [OC,IC,K]");
	conv("decoder.layers.0", conv_in_, (i64)cin.shape[0], (i64)cin.shape[1], (i64)cin.shape[2], 1,
	     ((i64)cin.shape[2] - 1) / 2, 1);

	// ── the upsampling blocks ──
	i64 li = 1;
	hop_ = 1;
	for (; has(st, "decoder.layers." + std::to_string(li) + ".layers.0.alpha"); li++) {
		Block blk;
		const std::string p = "decoder.layers." + std::to_string(li) + ".";
		const StTensor& uv = st.require(p + "layers.1.weight_v");   // [IC, OC, K]
		const i64 ic = uv.shape[0], oc = uv.shape[1], k = uv.shape[2];
		if (ic != ch) throw MediaError("ace vae: upsampler " + std::to_string(li) +
		                               " takes " + std::to_string(ic) + " channels, have " +
		                               std::to_string(ch));
		snake(p + "layers.0", blk.snake, ic);
		blk.up.ic = ic;
		blk.up.oc = oc;
		blk.up.k = k;
		blk.up.stride = k / 2;
		blk.up.pad = (k - blk.up.stride) / 2;   // ceil(stride/2), the reference's padding
		{
			// ConvTranspose1d stores [IC, OC, K] and torch's weight_norm uses dim=0,
			// so the normalisation runs over (OC, K) for each *input* channel and
			// `weight_g` is [IC, 1, 1] - not per output channel, which is what a
			// Conv1d's [OC, IC, K] would give. Folding it the wrong way still
			// produces a same-shaped matrix, i.e. audio of the right length and the
			// wrong content.
			const StTensor& g = st.require(p + "layers.1.weight_g");
			if (g.numel != ic)
				throw MediaError("ace vae: upsampler weight_g is not " + std::to_string(ic) +
				                 " long (dim-0 weight_norm), got " + std::to_string(g.numel));
			std::vector<float> gv = f32_of(st, p + "layers.1.weight_g");
			std::vector<float> vv = f32_of(st, p + "layers.1.weight_v");
			std::vector<float> w(vv.size());
			kernels::parallel_for(ic, [&](i64 i) {
				double acc = 0.0;
				for (i64 o = 0; o < oc; o++)
					for (i64 kk = 0; kk < k; kk++) {
						const double x = vv[(size_t)((i * oc + o) * k + kk)];
						acc += x * x;
					}
				const double nrm = std::sqrt(acc);
				const double sc = nrm > 0.0 ? (double)gv[(size_t)i] / nrm : 0.0;
				for (i64 o = 0; o < oc; o++)
					for (i64 kk = 0; kk < k; kk++) {
						const size_t idx = (size_t)((i * oc + o) * k + kk);
						w[idx] = (float)((double)vv[idx] * sc);
					}
			});
			blk.up.w = store_.emplace_back(std::move(w)).data();
			std::vector<float> b = f32_of(st, p + "layers.1.bias");
			if ((i64)b.size() != oc) throw MediaError("ace vae: upsampler bias size");
			blk.up.b = store_.emplace_back(std::move(b)).data();
		}
		hop_ *= blk.up.stride;

		// three residual units, dilation 1/3/9, in order
		const i64 dil[3] = {1, 3, 9};
		for (i64 u = 0; u < 3; u++) {
			const std::string up = p + "layers." + std::to_string(u + 2) + ".";
			if (!has(st, up + "layers.0.alpha"))
				throw MediaError("ace vae: block " + std::to_string(li) + " has only " +
				                 std::to_string(u) + " residual units");
			ResidualUnit ru;
			snake(up + "layers.0", ru.s1, oc);
			conv(up + "layers.1", ru.c7, oc, oc, 7, 1, 3 * dil[u], dil[u]);
			snake(up + "layers.2", ru.s2, oc);
			conv(up + "layers.3", ru.c1, oc, oc, 1, 1, 0, 1);
			blk.units.push_back(std::move(ru));
		}
		blocks_.push_back(std::move(blk));
		ch = oc;
	}
	if (blocks_.empty()) throw MediaError("ace vae: no decoder blocks in " + path);

	// ── the output head ──
	snake("decoder.layers." + std::to_string(li), snake_out_, ch);
	li++;
	const StTensor& cout = st.require("decoder.layers." + std::to_string(li) + ".weight_v");
	out_ch_ = cout.shape[0];
	if (cout.shape[1] != ch) throw MediaError("ace vae: the output conv does not take the last block's channels");
	conv("decoder.layers." + std::to_string(li), conv_out_, out_ch_, ch, (i64)cout.shape[2], 1,
	     ((i64)cout.shape[2] - 1) / 2, 1);

	latent_ = conv_in_.ic;
	if (latent_ != 64)
		throw MediaError("ace vae: the latent is " + std::to_string(latent_) +
		                 " channels, ACE 1.5's latent format is 64");
	if (const char* e = getenv("PHI_ACE_VAE_CHUNK")) {
		const long v = strtol(e, nullptr, 10);
		if (v > 0) chunk_frames_ = v;
	}

	if (!have_gpu) return;
	// ── the device path is available; the weights go up at decode time ──
	//
	// The upload is *not* done here, and that is deliberate. This loader runs
	// before the sampler in every ready-made chain, so anything parked on the card
	// now is subtracted from the DiT's residency plan (`AceDiT::plan_residency`
	// reads `vram_budget().local()`), and the decoder's folded fp32 weights - about
	// 330 MB - would cost the DiT a resident block for the whole sampling loop, to
	// be used once at the very end. `upload_device_weights` moves them into the
	// streaming weight arena at the first decode instead: that arena is not reset
	// by anything between the sampler's last block and the decode, and the next
	// phase's `new_layer()` - or `release_shared_arenas` - reclaims it, so the same
	// 330 MB are spent on the decode and then handed back rather than held
	// throughout the DiT's 50 steps.
	//
	// What still has to be checked here (and only here) is that the checkpoint's
	// upsamplers really have the shape the two-tap kernel is exact for: a file that
	// does not must fail at load, not silently two-thirds of the way through a
	// decode.
	for (const Block& blk : blocks_) {
		if (blk.up.k != 2 * blk.up.stride)
			throw MediaError("ace vae: upsampler k=" + std::to_string(blk.up.k) +
			                 " is not 2*stride=" + std::to_string(2 * blk.up.stride) +
			                 "; the GPU path needs the two-tap form (PHI_ACE_VAE_CPU=1 decodes "
			                 "this file on the host)");
	}
	dev_ready_ = true;
}

// ── the decode ─────────────────────────────────────────────────────────────

namespace {

inline float* buf(std::vector<float>& v, i64 n) {
	if ((i64)v.size() != n) v.assign((size_t)n, 0.0f);
	return v.data();
}

}  // namespace

void AceVae::decode_raw(const float* z, i64 T, std::vector<float>& out) const {
	// ── decoder.layers.0 ──
	i64 C = conv_in_.oc, L = T;
	float* a = buf(a_, C * L);
	conv_in_.forward(z, L, a, L);

	for (const Block& blk : blocks_) {
		const i64 Lr = blk.up.out_len(L);
		const i64 Co = blk.up.oc;
		// SnakeBeta, then the transposed convolution (which writes its own bias).
		float* s0 = buf(b_, C * L);
		blk.snake.apply(a, s0, C, L);
		float* cur = buf(c_, Co * Lr);
		blk.up.forward(s0, L, cur, Lr);

		for (const ResidualUnit& ru : blk.units) {
			float* t1 = buf(d_, Co * Lr);
			ru.s1.apply(cur, t1, Co, Lr);
			float* t2 = buf(a_, Co * Lr);
			ru.c7.forward(t1, Lr, t2, Lr);
			ru.s2.apply(t2, t1, Co, Lr);
			float* t3 = buf(b_, Co * Lr);
			ru.c1.forward(t1, Lr, t3, Lr);
			// reference: x = xt.add_(x)
			const i64 n = Co * Lr;
			kernels::parallel_for((n + 4095) / 4096, [&](i64 blkIdx) {
				const i64 r0 = blkIdx * 4096;
				const i64 r1 = std::min<i64>(n, r0 + 4096);
				for (i64 i = r0; i < r1; i++) cur[i] += t3[i];
			});
		}
		// `cur` is this stage's activation; the old `a` buffer is now free, so
		// hand the roles over instead of copying.
		a_.swap(c_);
		a = a_.data();
		C = Co;
		L = Lr;
	}

	// ── the output head ──
	float* h = buf(b_, C * L);
	snake_out_.apply(a, h, C, L);
	out.assign((size_t)(out_ch_ * L), 0.0f);
	conv_out_.forward(h, L, out.data(), L);
}

// ── the GPU decode ─────────────────────────────────────────────────────────

// PHI_ACE_VAE_CPU=1 forces the host reference (this is the conformance seam: the
// device kernels are what has to be checked against it, so the switch has to be
// able to select either side on a machine that has both).
static bool ace_vae_cpu_forced() {
	static const bool v = [] {
		const char* e = getenv("PHI_ACE_VAE_CPU");
		return e && *e && *e != '0';
	}();
	return v;
}

void AceVae::upload_device_weights() {
	// Every folded weight, every bias and the exp()'d SnakeBeta parameters, into
	// the *streaming* weight arena - see `open` for why not the `keep` arena, and
	// why this runs at decode time rather than at load time.
	//
	// Re-uploaded on every `decode_pair` call: the arena is reset by whatever GPU
	// phase ran before this one (the DiT's per-block `new_layer()`, the LM's, or
	// `release_shared_arenas` between tool calls), so the handles cannot be cached
	// across calls - and the cost (330 MB of H2D, tens of milliseconds) is nothing
	// against the decode that follows it.
	g_->ctx->begin();
	auto upload = [&](const float* p, u64 n) {
		GpuAlloc a = g_->walloc(n * 4);
		if (n) g_->upload_into(a, p, n * 4);
		return a;
	};
	// A conv's weights, plus its bias when the checkpoint has one. An absent bias
	// leaves the handle empty: the dispatchers read a bias only when `has_bias`
	// says so, and bind the input buffer otherwise.
	auto upload_conv = [&](const Conv1dRef& c, GpuAlloc& w, GpuAlloc& b) {
		w = upload(c.w, (u64)c.oc * (u64)c.ic * (u64)c.k);
		b = c.b ? upload(c.b, (u64)c.oc) : GpuAlloc{};
	};
	// SnakeBeta: the reference applies exp() to the stored log parameters
	// (`kernels::snake_beta_f32`), and doing it here means the device and the host
	// share the *same* per-channel a/b values bit for bit.
	auto upload_snake = [&](const kernels::SnakeBetaAct& s, i64 c, GpuAlloc& a, GpuAlloc& b) {
		std::vector<float> al((size_t)c), be((size_t)c);
		for (i64 i = 0; i < c; i++) {
			al[(size_t)i] = std::exp(s.alpha_log[i]);
			be[(size_t)i] = std::exp(s.beta_log[i]);
		}
		a = upload(al.data(), (u64)c);
		b = upload(be.data(), (u64)c);
	};

	upload_conv(conv_in_, d_conv_in_w_, d_conv_in_b_);
	for (Block& blk : blocks_) {
		// `open` already refused a k != 2*stride upsampler (see there).
		blk.up_w = upload(blk.up.w, (u64)blk.up.ic * (u64)blk.up.oc * (u64)blk.up.k);
		blk.up_b = blk.up.b ? upload(blk.up.b, (u64)blk.up.oc) : GpuAlloc{};
		upload_snake(blk.snake, blk.up.ic, blk.snake_a, blk.snake_b);
		for (ResidualUnit& ru : blk.units) {
			upload_snake(ru.s1, blk.up.oc, ru.s1_a, ru.s1_b);
			upload_conv(ru.c7, ru.c7_w, ru.c7_b);
			upload_snake(ru.s2, blk.up.oc, ru.s2_a, ru.s2_b);
			upload_conv(ru.c1, ru.c1_w, ru.c1_b);
		}
	}
	upload_snake(snake_out_, conv_out_.ic, d_out_a_, d_out_b_);
	upload_conv(conv_out_, d_conv_out_w_, d_conv_out_b_);
	// The uploads are copies recorded on the stream; they have to land before the
	// weights are read, which is why the ring is not rewound until `submit()` has
	// waited for them.
	g_->ctx->submit();
	g_->ring->rewind();
}

bool AceVae::dev_decode() const {
	if (!dev_ready_ || !g_ || !g_->ok()) return false;
	return !ace_vae_cpu_forced();
}

void AceVae::decode_raw_gpu(const float* z, i64 T, std::vector<float>& out) const {
	// One chunk of `decode_raw`, module for module and in the same order, with
	// every activation in the activations arena. The arithmetic is the kernels'
	// (see kernels/ops.cpp, which documents each one against the host reference
	// this file also runs), so the two paths differ only in their rounding.
	//
	// ── the footprint ──
	//
	// The decoder is expansive: the last block's activation is
	// 128 x 1920*T samples. Two of those are live at once - the stage being
	// computed and the one before it, which the upsampler is still reading - and
	// they alternate between two fixed slots, because a bump arena cannot give
	// the older of two allocations back without giving back the newer one too.
	// The residual units' three temporaries are the only other working set and they
	// are rewound per unit, so the peak is 2 slots + 3 temporaries (the reward for
	// the whole dance is that the *peak* is what the accountant sees, instead of
	// the sum of five blocks' upsampled activations).
	ComputeContext& ctx = *g_->ctx;
	GpuArena& aa = *g_->aa;
	i64 slot = conv_in_.oc * T;
	{
		i64 L = T;
		for (const Block& blk : blocks_) {
			L = blk.up.out_len(L);
			slot = std::max<i64>(slot, blk.up.oc * L);
		}
	}
	GpuAlloc zin = aa.alloc((u64)latent_ * (u64)T * 4);
	GpuAlloc sa = aa.alloc((u64)slot * 4);
	GpuAlloc sb = aa.alloc((u64)slot * 4);
	// A view of a slot at the size the current stage actually needs: `bytes` is
	// what the dispatchers validate and what makes the two slots reusable across
	// stages of different widths.
	auto view = [](const GpuAlloc& base, i64 n) {
		GpuAlloc v = base;
		v.bytes = (u64)n * 4;
		return v;
	};

	ctx.begin();
	g_->upload_into(zin, z, (u64)latent_ * (u64)T * 4);

	auto conv1d = [&](const Conv1dRef& c, const GpuAlloc& w, const GpuAlloc& b, const GpuAlloc& x,
	                  i64 l, const GpuAlloc& y) {
		Conv1dArgsG a;
		a.x = x;
		a.wgt = w;
		a.bias = b;
		a.y = y;
		a.ic = c.ic;
		a.l = l;
		a.oc = c.oc;
		a.k = c.k;
		a.stride = c.stride;
		a.pad = c.pad;
		a.dilation = c.dilation;
		a.has_bias = c.b != nullptr;
		dispatch_conv1d(ctx, a);
	};
	// `y` may alias `x` (one thread reads and writes only its own element).
	auto snake_into = [&](const GpuAlloc& x, const GpuAlloc& al, const GpuAlloc& be, i64 c, i64 l,
	                     const GpuAlloc& y) {
		SnakeBetaArgsG a;
		a.x = x;
		a.alpha = al;
		a.beta = be;
		a.y = y;
		a.c = c;
		a.l = l;
		dispatch_snake_beta(ctx, a);
	};
	auto snake = [&](const GpuAlloc& x, const GpuAlloc& al, const GpuAlloc& be, i64 c, i64 l) {
		snake_into(x, al, be, c, l, x);
	};

	// ── decoder.layers.0 ──
	i64 C = conv_in_.oc, L = T;
	GpuAlloc cur = view(sa, C * L);
	conv1d(conv_in_, d_conv_in_w_, d_conv_in_b_, zin, T, cur);
	bool in_sa = true;

	for (const Block& blk : blocks_) {
		const i64 Lr = blk.up.out_len(L);
		const i64 Co = blk.up.oc;
		snake(cur, blk.snake_a, blk.snake_b, C, L);
		GpuAlloc up = view(in_sa ? sb : sa, Co * Lr);
		{
			ConvT1dArgsG a;
			a.x = cur;
			a.wgt = blk.up_w;
			a.bias = blk.up_b;
			a.y = up;
			a.ic = blk.up.ic;
			a.l = L;
			a.oc = blk.up.oc;
			a.k = blk.up.k;
			a.stride = blk.up.stride;
			a.pad = blk.up.pad;
			a.has_bias = blk.up.b != nullptr;
			dispatch_convt1d(ctx, a);
		}
		cur = up;
		in_sa = !in_sa;
		C = Co;
		L = Lr;

		for (const ResidualUnit& ru : blk.units) {
			// The reference's residual unit is `x + conv2(snake2(conv1(snake1(x))))`
			// - the *skip* keeps the unit's input, so `snake1` must not be applied in
			// place here, unlike the block-level one above (whose input is dead once
			// the upsampler has read it). Doing it in place adds the snake'd value
			// back onto itself instead of onto `x`, which is a different, louder
			// signal: measured on the host/device A/B of this file, block 0's first
			// unit already came out 4x off in range and the clip clipped at full
			// scale.
			//
			// The unit's three temporaries are the only allocations that can be
			// rewound inside a chunk: `t3` is read by the add that is already on the
			// stream, and the next unit's first write is recorded *after* it, so the
			// reuse is ordered.
			GpuArena::Mark m = aa.mark();
			GpuAlloc t1 = aa.alloc((u64)C * (u64)L * 4);
			snake_into(cur, ru.s1_a, ru.s1_b, C, L, t1);
			GpuAlloc t2 = aa.alloc((u64)C * (u64)L * 4);
			conv1d(ru.c7, ru.c7_w, ru.c7_b, t1, L, t2);
			snake_into(t2, ru.s2_a, ru.s2_b, C, L, t1);
			GpuAlloc t3 = aa.alloc((u64)C * (u64)L * 4);
			conv1d(ru.c1, ru.c1_w, ru.c1_b, t1, L, t3);
			// reference: x = xt.add_(x)
			ElemArgs e;
			e.op = ElemOp::Add;
			e.a = cur;
			e.b = t3;
			e.c = t3;
			e.y = cur;
			e.rows = 1;
			e.cols = C * L;
			dispatch_elem(ctx, e);
			aa.rewind_to(m);
		}
		// Drain between blocks: the whole chunk is ~5e11 FLOP of naive gather
		// kernels in one command list, and the Windows watchdog measures the
		// *list*, not the dispatch (TDR is 2 s). A fence per block is ~50 us against
		// tens of milliseconds of work, and it changes nothing about the result
		// (nothing crosses a block boundary).
		ctx.submit();
		ctx.begin();
	}

	// ── the output head ──
	snake(cur, d_out_a_, d_out_b_, C, L);
	GpuAlloc yo = aa.alloc((u64)out_ch_ * (u64)L * 4);
	conv1d(conv_out_, d_conv_out_w_, GpuAlloc{}, cur, L, yo);
	ctx.submit();
	out.assign((size_t)(out_ch_ * L), 0.0f);
	ctx.download(yo.res, yo.off, out.data(), out.size() * 4);
}

std::vector<float> AceVae::decode_pair(const std::vector<float>& z, i64 T) {
	if (T <= 0) throw MediaError("ace vae: empty latent");
	if ((i64)z.size() != latent_ * T)
		throw MediaError("ace vae: latent is " + std::to_string(z.size()) + " floats, expected " +
		                 std::to_string(latent_ * T) + " (" + std::to_string(latent_) + " x " +
		                 std::to_string(T) + ")");
	a_.clear();
	b_.clear();
	c_.clear();
	d_.clear();
	const bool dev = dev_decode();
	if (dev) {
		// One reset for the whole decode: every chunk's temporaries are rewound at
		// the end of the chunk, so this only has to drop whatever the previous
		// stage of the graph left behind (new_step() submits anything still
		// recorded before it reuses the memory).
		g_->new_step();
		// The device weights, re-uploaded into the streaming arena (see
		// `upload_device_weights`). Done once per call, not per chunk.
		upload_device_weights();
	}

	const i64 Lout = T * hop_;
	std::vector<float> pcm((size_t)(2 * Lout), 0.0f);
	std::vector<float> chunk_z, chunk_out;
	const i64 step = std::max<i64>(1, chunk_frames_);
	const i64 halo = std::max<i64>(0, halo_frames_);
	for (i64 c0 = 0; c0 < T; c0 += step) {
		const i64 c1 = std::min(T, c0 + step);
		const i64 h0 = std::max<i64>(0, c0 - halo);
		const i64 h1 = std::min(T, c1 + halo);
		const i64 th = h1 - h0;
		chunk_z.resize((size_t)(latent_ * th));
		for (i64 ch = 0; ch < latent_; ch++)
			std::memcpy(&chunk_z[(size_t)ch * (size_t)th], &z[(size_t)ch * (size_t)T + (size_t)h0],
			            sizeof(float) * (size_t)th);
		if (dev) {
			// The chunk's memory comes back the moment its result has been
			// downloaded (the download waits for the queue), so a long clip
			// decodes in one chunk's footprint rather than in the clip's.
			GpuArena::Mark m = g_->aa->mark();
			decode_raw_gpu(chunk_z.data(), th, chunk_out);
			g_->aa->rewind_to(m);
		} else {
			decode_raw(chunk_z.data(), th, chunk_out);
		}
		// Crop: keep columns [c0, c1) of this chunk's output.
		const i64 off = (c0 - h0) * hop_;
		const i64 n = (c1 - c0) * hop_;
		for (i64 ch = 0; ch < out_ch_; ch++) {
			// interleaved output: channel `ch` of output sample `c0*hop + i`
			const float* src = &chunk_out[(size_t)ch * (size_t)(th * hop_) + (size_t)off];
			for (i64 i = 0; i < n; i++)
				pcm[(size_t)(c0 * hop_ + i) * (size_t)out_ch_ + (size_t)ch] = src[i];
		}
	}
	return pcm;
}

std::vector<float> AceVae::decode(const std::vector<float>& z, i64 T) {
	std::vector<float> pair = decode_pair(z, T);
	const i64 n = (i64)pair.size() / out_ch_;
	std::vector<float> mono((size_t)n, 0.0f);
	for (i64 i = 0; i < n; i++) {
		float s = 0.0f;
		for (i64 c = 0; c < out_ch_; c++) s += pair[(size_t)i * (size_t)out_ch_ + (size_t)c];
		mono[(size_t)i] = s / (float)out_ch_;
	}
	return mono;
}

std::vector<float> AceVae::decode_stereo(const std::vector<float>& z, i64 T) {
	if (T <= 0) throw MediaError("ace vae: empty latent");
	if ((i64)z.size() != latent_ * 2 * T)
		throw MediaError("ace vae: [64,2,T] latent expected, got " + std::to_string(z.size()));
	// `z` is [latent, 2, T] (the H3 audio VAE's own `[z_channels, 2, T]` shape and
	// the same slicing): stereo channel `s`'s mono latent is z[c][s][:].
	std::vector<float> slice((size_t)(latent_ * T));
	std::vector<float> pcm((size_t)(2 * T * hop_), 0.0f);
	for (i64 s = 0; s < 2; s++) {
		for (i64 c = 0; c < latent_; c++)
			std::memcpy(&slice[(size_t)c * (size_t)T],
			            &z[((size_t)c * 2 + (size_t)s) * (size_t)T], sizeof(float) * (size_t)T);
		std::vector<float> mono = decode(slice, T);
		for (size_t i = 0; i < mono.size(); i++) pcm[i * 2 + (size_t)s] = mono[i];
	}
	return pcm;
}

}  // namespace phi::media
