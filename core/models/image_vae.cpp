// Qwen-Image-2.1 VAE implementation (see image_vae.hpp for the mapping to
// ComfyUI's Wan 2.2 VAE code).
//
// The whole chain is four primitives: a 2-D convolution (`dispatch_conv2d`),
// a channel-wise RMSNorm (`chw_rmsnorm`), an elementwise op (`dispatch_elem`)
// and single-head spatial attention (`dispatch_vae_attn_out`). The two ops that
// are specific to this VAE family are covered without new device code:
//
//   * `AvgDown3D` is a 2x2 stride-2 convolution whose kernel is generated on the
//     host from the group rule (see `build_avg_kernel`), and
//   * `DupUp3D` is the `dupup2x` gather kernel.
//
// Memory. The chain is carried in two ping-pong arenas (`carry_[2]`), each
// holding exactly one activation tensor: the stage that is running reads one and
// writes the other, and the arena it is about to write into is reset first,
// which is safe because its contents are two stages old. Everything a stage
// needs besides its input (the normalised copy, the shortcut, the q/k/v of an
// attention) lives in `act_`, which every block rewinds to its own mark. Peak
// footprint is therefore about two activations plus one block's scratch, instead
// of the sum of every tensor the pass ever touched.
#include "models/image_vae.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include "io/image_io.hpp"
#include "kernels/gpu_ops.hpp"
#include "runtime/vram_budget.hpp"

namespace phi::media {

namespace {

// comfy/sd.py's Qwen-Image-2.1 VAE config: temperal_downsample [F,T,T,T] and its
// reversal for the decoder. Kept here (rather than in the config) because they
// are properties of the architecture, not of the file.
constexpr int kTemporalStages = 4;
constexpr bool kTemporalDown[kTemporalStages] = {false, true, true, true};
constexpr bool kTemporalUp[kTemporalStages] = {true, true, true, false};

u64 bytes_of(i64 c, i64 h, i64 w) { return (u64)c * (u64)h * (u64)w * 4u; }

GpuAlloc slice(const GpuAlloc& a, u64 byte_off, u64 bytes) {
	GpuAlloc r = a;
	r.off += byte_off;
	r.bytes = bytes;
	return r;
}

// PHI_QVAE_DUMP=<dir> writes every stage as raw f32 so the port can be diffed
// against a stage-by-stage reference instead of one cosine for the
// whole decoder.
void qdump(ComputeContext& ctx, const char* name, const GpuAlloc& a, i64 n) {
	const char* dir = getenv("PHI_QVAE_DUMP");
	if (!dir || !*dir) return;
	ctx.submit();
	std::vector<float> v((size_t)n);
	ctx.download(a.res, a.off, v.data(), (size_t)n * 4);
	std::string path = std::string(dir) + "/" + name + ".f32";
	FILE* f = fopen(path.c_str(), "wb");
	if (f) {
		fwrite(v.data(), 4, (size_t)n, f);
		fclose(f);
	}
}

// PHI_QVAE_TRACE=1 prints each decode stage's arena capacities: the decoder's
// peak is the *sum of the chunks each arena has ever grown* (rewind_to reuses
// them), not the largest tensor it holds, so this is the instrument that says
// which stage sets the high-water mark.
bool qvae_trace() {
	static int v = -1;
	if (v < 0) {
		const char* e = getenv("PHI_QVAE_TRACE");
		v = (e && *e && *e != '0') ? 1 : 0;
	}
	return v != 0;
}

}  // namespace

// ── weights ───────────────────────────────────────────────────────────────

// The latent (de)normalisation tables, from comfy/latent_formats.py ::
// QwenImage21. They are NOT in the checkpoint (its containers carry no
// latents_mean/latents_std), so they are transcribed here - and `probe()` still
// prefers the checkpoint's own tables when a future container ships them.
static const float kLatentMean[64] = {
	    0.512600f, 0.772100f, -0.063100f, 1.350600f,
	    -0.785500f, -2.102500f, -0.345800f, 1.372200f,
	    1.887300f, -1.717700f, -0.651000f, 0.273200f,
	    0.756200f, -0.616300f, -1.027700f, 3.836300f,
	    2.021000f, 0.047200f, 0.932000f, 2.008700f,
	    2.495400f, -0.139100f, -1.424900f, 1.846400f,
	    -0.523600f, 1.282600f, 3.704600f, -1.303500f,
	    2.728600f, -1.451800f, -1.903600f, -1.995500f,
	    -0.034200f, -1.026500f, -0.763600f, 3.055500f,
	    0.074600f, -3.075100f, -0.107600f, 1.737600f,
	    -1.091400f, -1.943500f, -0.278400f, -1.368000f,
	    0.480900f, -0.443300f, 0.376400f, 0.572900f,
	    -2.059500f, 1.096000f, -1.326000f, -2.021100f,
	    -5.017900f, 0.527500f, 4.016200f, 1.850500f,
	    0.302600f, 1.937300f, 1.493700f, 0.263200f,
	    0.554700f, -1.712100f, -0.156200f, 0.030400f
};
static const float kLatentStd[64] = {
	    3.200100f, 3.293600f, 3.432100f, 3.009100f,
	    3.106100f, 4.037900f, 4.070500f, 3.791000f,
	    3.078500f, 3.650000f, 3.930800f, 3.090400f,
	    2.877800f, 3.767500f, 3.732000f, 5.075600f,
	    3.286400f, 4.039700f, 3.131700f, 4.044300f,
	    2.924900f, 3.945400f, 3.098800f, 4.248900f,
	    3.489600f, 3.851300f, 3.932300f, 3.471900f,
	    3.749800f, 4.283000f, 3.569400f, 4.246700f,
	    3.903700f, 3.294700f, 5.077000f, 3.507500f,
	    3.270000f, 3.476700f, 2.806300f, 5.112500f,
	    3.532700f, 4.783300f, 3.128600f, 4.181900f,
	    3.852700f, 3.831200f, 3.560500f, 4.387500f,
	    3.962400f, 4.016800f, 3.564300f, 4.055000f,
	    5.561400f, 4.296300f, 4.408000f, 3.495900f,
	    3.874700f, 3.760800f, 3.573500f, 3.149000f,
	    3.766200f, 3.674600f, 3.456300f, 3.816100f
};

void ImageVae::probe() {
	st_.open(path_);
	const StTensor& c1 = st_.require("encoder.conv1.weight");
	if (c1.shape.size() != 5 || c1.shape[2] != 1)
		throw MediaError("qwen_vae: encoder.conv1 is not a temporal-kernel-1 Conv3d");
	cfg_.enc_dim = c1.shape[0];
	cfg_.image_channels = c1.shape[1];   // 4: the encoder eats the alpha plane
	const StTensor& hg = st_.require("decoder.head.0.gamma");
	cfg_.dec_dim = hg.shape[0];
	const StTensor& hw = st_.require("decoder.head.2.weight");
	if (hw.shape[0] != cfg_.image_channels)
		throw MediaError("qwen_vae: decoder head channels != encoder input channels");
	const StTensor& zc = st_.require("conv2.weight");
	cfg_.z2 = st_.require("conv1.weight").shape[0];
	cfg_.latent_channels = zc.shape[0];
	if (cfg_.z2 != 2 * cfg_.latent_channels)
		throw MediaError("qwen_vae: conv1/conv2 disagree about the latent width");
	// Spatial compression: 2^(len(dim_mult)-1) with the last level not striding.
	i64 levels = 0;
	while (st_.find("decoder.upsamples." + std::to_string(levels) + ".upsamples.0.residual.0.gamma"))
		levels++;
	if (levels == 0) throw MediaError("qwen_vae: no decoder upsample levels");
	if ((i64)cfg_.dim_mult.size() != levels)
		throw MediaError("qwen_vae: the checkpoint has " + std::to_string(levels) +
		                 " upsample levels, the config assumes " +
		                 std::to_string(cfg_.dim_mult.size()));
	i64 s = 1;
	for (i64 i = 0; i + 1 < levels; i++) s *= 2;
	cfg_.spatial = s;
	// The number of residual blocks per level is read off the first level.
	i64 blocks = 0;
	while (st_.find("decoder.upsamples.0.upsamples." + std::to_string(blocks) +
	                ".residual.0.gamma"))
		blocks++;
	cfg_.num_res_blocks = std::max<i64>(1, blocks - 1);

	// The latent normalisation, straight off the latent format's tables.
	cfg_.mean.clear();
	cfg_.std_.clear();
	if (const StTensor* m = st_.find("latents_mean"))
		cfg_.mean = tensor_to_f32(st_, *m);
	if (const StTensor* stt = st_.find("latents_std"))
		cfg_.std_ = tensor_to_f32(st_, *stt);
	if (cfg_.mean.size() != (size_t)cfg_.latent_channels) {
		cfg_.mean.assign(kLatentMean, kLatentMean + 64);
		cfg_.std_.assign(kLatentStd, kLatentStd + 64);
	}
}

void ImageVae::load_conv(const std::string& name, QVaeT& c) {
	const StTensor& t = st_.require(name + ".weight");
	// Conv3d everywhere except the attention block's projections, which are plain
	// Conv2d. Every temporal kernel in this checkpoint is 1, so a 5-D tensor is a
	// 2-D convolution wearing a 5-D shape.
	if (t.shape.size() == 5) {
		if (t.shape[2] != 1)
			throw MediaError("qwen_vae: " + name + " is not a temporal-kernel-1 Conv3d");
		c.oc = t.shape[0];
		c.ic = t.shape[1];
		c.kh = t.shape[3];
		c.kw = t.shape[4];
	} else if (t.shape.size() == 4) {
		c.oc = t.shape[0];
		c.ic = t.shape[1];
		c.kh = t.shape[2];
		c.kw = t.shape[3];
	} else {
		throw MediaError("qwen_vae: " + name + " is not a convolution weight");
	}
	// The weights stay in the checkpoint's own storage precision, and the conv
	// kernel decodes whatever width that is (`Conv2dArgsG::w_dtype`):
	//
	//   * bf16 (this checkpoint's own format) - the kernel reads bf16 operands
	//     directly, which halves both the charge and the weight traffic per conv
	//     against an fp32 copy (675 MB against 1.35 GB, the difference between
	//     decoding and not decoding a 1024x1024);
	//   * fp16 - read at its own width, same argument again;
	//   * fp32 - read as fp32, which is what a non-bf16 source used to be
	//     materialised into anyway;
	//   * anything else (a quantised or fp8 source: no 4-D conv can be packed, and
	//     the conv has no fp8 path) - decoded to fp32, which is the widest and is
	//     exact for every one of them.
	//
	// `PHI_QVAE_W32` pins the fp32 operand path, which is the A/B for "is this a
	// precision change or a speed change".
	GpuArena& warena = warena_;
	const DType wd = t.dtype;
	const bool w32 = getenv("PHI_QVAE_W32") != nullptr;
	if (!w32 && (wd == DType::BF16 || wd == DType::F16)) {
		c.w_dtype = (wd == DType::BF16) ? 1 : 2;
		std::vector<u8> wp = st_.materialize(t, wd);
		c.w = warena.alloc(wp.size());
		upload_range(*g_->ctx, *g_->ring, c.w.res, c.w.off, wp.data(), wp.size());
	} else {
		c.w_dtype = 0;
		std::vector<float> wp = tensor_to_f32(st_, t);
		c.w = warena.alloc((u64)wp.size() * 4);
		upload_range(*g_->ctx, *g_->ring, c.w.res, c.w.off, wp.data(), wp.size() * 4);
	}
	if (const StTensor* b = st_.find(name + ".bias")) {
		std::vector<float> bp = tensor_to_f32(st_, *b);
		if ((i64)bp.size() != c.oc) throw MediaError("qwen_vae: " + name + " bias shape");
		c.b = warena_.alloc((u64)bp.size() * 4);
		upload_range(*g_->ctx, *g_->ring, c.b.res, c.b.off, bp.data(), bp.size() * 4);
		c.has_bias = true;
	}
}

void ImageVae::load_gamma(const std::string& name, GpuAlloc& g) {
	const StTensor& t = st_.require(name);
	std::vector<float> v = tensor_to_f32(st_, t);
	g = warena_.alloc((u64)v.size() * 4);
	upload_range(*g_->ctx, *g_->ring, g.res, g.off, v.data(), v.size() * 4);
}

void ImageVae::load_res(const std::string& prefix, QVaeRes& r, i64 ic, i64 oc) {
	load_gamma(prefix + ".residual.0.gamma", r.g0);
	load_gamma(prefix + ".residual.3.gamma", r.g3);
	load_conv(prefix + ".residual.2", r.c2);
	load_conv(prefix + ".residual.6", r.c6);
	if (ic != oc) {
		load_conv(prefix + ".shortcut", r.shortcut);
		r.has_shortcut = true;
	}
}

void ImageVae::load_attn(const std::string& prefix, QVaeAttn& a, i64 c) {
	load_gamma(prefix + ".norm.gamma", a.norm_g);
	load_conv(prefix + ".to_qkv", a.qkv);
	load_conv(prefix + ".proj", a.proj);
	if (a.qkv.oc != 3 * c || a.proj.oc != c)
		throw MediaError("qwen_vae: attention shape mismatch at " + prefix);
}

// The `AvgDown3D` shortcut as a 2x2 stride-2 convolution.
//
// PyTorch view/permute/mean: the flat axis is (channel, f_t, row, column) with
// `ft*4` entries per input channel; `out` channels split it into groups of
// `in*ft*4/out` and the mean runs over one group. The frame sits in the f_t =
// ft-1 slot of the padded stack (AvgDown3D pads at the *front*), and the other
// slots are zeros - so only the f_t = ft-1 slice of a group contributes, and the
// mean still divides by the full group size.
void ImageVae::load_avg_kernel(QVaeT& c, i64 ic, i64 oc, i64 ft) {
	const i64 per = ft * 4;
	const i64 g = ic * per / oc;
	std::vector<float> k((size_t)oc * ic * 4, 0.0f);
	for (i64 o = 0; o < oc; o++) {
		for (i64 t = 0; t < g; t++) {
			const i64 idx = o * g + t;
			const i64 ci = idx / per, rem = idx % per;
			if (rem / 4 != ft - 1) continue;   // the f_t padding slots are zeros
			const i64 j = (rem % 4) >> 1, kk = rem & 1;
			k[(size_t)((o * ic + ci) * 4 + j * 2 + kk)] += 1.0f / (float)g;
		}
	}
	c.oc = oc;
	c.ic = ic;
	c.kh = 2;
	c.kw = 2;
	c.w = warena_.alloc((u64)k.size() * 4);
	upload_range(*g_->ctx, *g_->ring, c.w.res, c.w.off, k.data(), k.size() * 4);
	c.has_bias = false;
}

void ImageVae::load_weights() {
	if (weights_loaded_) return;
	g_->ctx->begin();

	const i64 levels = (i64)cfg_.dim_mult.size();
	// decoder dims: [dec_dim*mult[-1], dec_dim*mult[-1], dec_dim*mult[::-1]...]
	std::vector<i64> dd;
	dd.push_back(cfg_.dec_dim * cfg_.dim_mult.back());
	for (i64 i = levels - 1; i >= 0; i--) dd.push_back(cfg_.dec_dim * cfg_.dim_mult[(size_t)i]);
	dims_ = dd;
	load_gamma("decoder.head.0.gamma", dec_head_g_);
	load_conv("decoder.head.2", dec_head_);
	load_conv("decoder.conv1", dec_conv1_);
	load_res("decoder.middle.0", dec_mid_[0], dd[0], dd[0]);
	load_attn("decoder.middle.1", dec_attn_, dd[0]);
	load_res("decoder.middle.2", dec_mid_[1], dd[0], dd[0]);
	up_.assign((size_t)(dd.size() - 1), QVaeUp{});
	for (size_t i = 0; i + 1 < dd.size(); i++) {
		const i64 ic = dd[i], oc = dd[i + 1];
		const bool up_flag = (i64)i != levels - 1;
		const std::string p = "decoder.upsamples." + std::to_string(i);
		QVaeUp& u = up_[i];
		u.ic = ic;
		u.oc = oc;
		// temperal_upsample is temperal_downsample reversed; stages past its end
		// (the last one) are spatial-only.
		u.dup_ft = ((i64)i < kTemporalStages && kTemporalUp[i]) ? 2 : 1;
		for (i64 b = 0; b <= cfg_.num_res_blocks; b++)
			u.blocks.emplace_back(), load_res(p + ".upsamples." + std::to_string(b),
			                                  u.blocks.back(), b == 0 ? ic : oc, oc);
		if (up_flag) {
			const std::string rp = p + ".upsamples." + std::to_string(cfg_.num_res_blocks + 1);
			load_conv(rp + ".resample.1", u.resample);
			u.has_resample = true;
		}
	}

	// encoder dims: [enc_dim * 1, enc_dim * mult...]
	std::vector<i64> ed{cfg_.enc_dim};
	for (i64 m : cfg_.dim_mult) ed.push_back(cfg_.enc_dim * m);
	load_gamma("encoder.head.0.gamma", enc_head_g_);
	load_conv("encoder.head.2", enc_head_);
	load_conv("encoder.conv1", enc_conv1_);
	load_res("encoder.middle.0", enc_mid_[0], ed.back(), ed.back());
	load_attn("encoder.middle.1", enc_attn_, ed.back());
	load_res("encoder.middle.2", enc_mid_[1], ed.back(), ed.back());
	down_.assign((size_t)(ed.size() - 1), QVaeDown{});
	for (size_t i = 0; i + 1 < ed.size(); i++) {
		const i64 ic = ed[i], oc = ed[i + 1];
		const bool down_flag = (i64)i != levels - 1;
		const std::string p = "encoder.downsamples." + std::to_string(i);
		QVaeDown& d = down_[i];
		d.ic = ic;
		d.oc = oc;
		d.avg_ft = ((i64)i < kTemporalStages && kTemporalDown[i]) ? 2 : 1;
		for (i64 b = 0; b < cfg_.num_res_blocks; b++)
			d.blocks.emplace_back(), load_res(p + ".downsamples." + std::to_string(b),
			                                  d.blocks.back(), b == 0 ? ic : oc, oc);
		if (down_flag) {
			load_conv(p + ".downsamples." + std::to_string(cfg_.num_res_blocks) + ".resample.1",
			          d.resample);
			d.has_resample = true;
		}
		// Down_ResidualBlock always adds the average shortcut, even where both of
		// its factors are 1 and it is the identity.
		load_avg_kernel(d.avg, ic, oc, d.avg_ft);
	}
	load_conv("conv1", root_conv1_);
	load_conv("conv2", root_conv2_);

	g_->ctx->submit();
	g_->ring->rewind();
	weights_loaded_ = true;
}

void ImageVae::open(const std::string& path, GpuCtx* gpu) {
	g_ = gpu;
	path_ = path;
	probe();
	// The arenas: weights are one flat block, the carries are single tensors, and
	// the scratch is rewound per block. 64 MB chunks keep the accountant's
	// rounding waste small at every shape this VAE runs (the alternative, a chunk
	// per large tensor, over-charges the small ones).
	warena_.init(g_->ctx, 64ull << 20);
	warena_.set_tag("vae.weights");
	act_.init(g_->ctx, 64ull << 20);
	act_.set_tag("vae.acts");
	in_.init(g_->ctx, 64ull << 20);
	in_.set_tag("vae.tile");
	for (int i = 0; i < 2; i++) {
		carry_[i].init(g_->ctx, 64ull << 20);
		carry_[i].set_tag("vae.slot");
	}
	open_ = true;
}

void ImageVae::release_activation_memory() {
	act_.release_chunks();
	in_.release_chunks();
	carry_[0].release_chunks();
	carry_[1].release_chunks();
}

void ImageVae::release_gpu_memory() {
	release_activation_memory();
	warena_.release_chunks();
	weights_loaded_ = false;
}

u64 ImageVae::estimate_vae_activation_bytes(i64 h, i64 w, const ImageVaeConfig& cfg) {
	const i64 levels = (i64)cfg.dim_mult.size();
	std::vector<i64> dd;
	dd.push_back(cfg.dec_dim * cfg.dim_mult.back());
	for (i64 i = levels - 1; i >= 0; i--) dd.push_back(cfg.dec_dim * cfg.dim_mult[(size_t)i]);
	// The untiled peak, which is what the decoder costs when it has the room for
	// it: per level, the two carries (input at h x w and output at h x w) plus the
	// block's own normalised copy, and at full resolution the head's copy as well.
	u64 peak = 0;
	i64 hh = h, ww = w;
	for (size_t i = 0; i + 1 < dd.size(); i++) {
		const u64 two = bytes_of(dd[i], hh, ww) + bytes_of(dd[i + 1], hh, ww);
		peak = std::max(peak, two * 2 + bytes_of(dd[i + 1], hh, ww));
		if ((i64)i != levels - 1) {
			hh *= 2;
			ww *= 2;
		}
	}
	peak = std::max(peak, bytes_of(cfg.image_channels, hh, ww) * 2ull);
	return peak;
}

// ── the tiled decode's geometry ───────────────────────────────────────────

// Every residual block is two 3x3 convs (radius 2 at its own resolution), each
// resample is one 3x3 conv with the nearest-2x upsample folded in (radius 1 at
// the *input* resolution, which is what `dup` reads), and the head is one more
// 3x3 conv at full resolution. Summing those radii down the level stack and
// scaling each by the ratio between its resolution and the split level's gives
// the distance an output pixel has to be from the edge of its tile for its value
// to be the one the untiled decode computes.
i64 ImageVae::suffix_halo(i64 split) const {
	const i64 nlevels = (i64)dims_.size() - 1;
	double r = 0.0;
	for (i64 i = split; i < nlevels; i++) {
		const double sc = std::pow(0.5, (double)(i - split));
		r += (double)(2 * (cfg_.num_res_blocks + 1)) * sc;   // the level's blocks
		if (i != nlevels - 1) r += sc;                        // its resample
	}
	r += std::pow(0.5, (double)(nlevels - 1 - split));        // the head's conv
	return (i64)std::ceil(r) + 1;
}

// The untiled decode's own footprint (two full-resolution carries plus the head's
// copy) against what is left of the budget once the weights, the upload ring and
// a working margin are out of the way. Above it, the decoder is split: levels
// [0, split) run once over the whole canvas and the rest runs per tile, so the
// peak stops following the canvas and starts following the tile.
// The per-level cost of running upsample levels [from, to) over a canvas that is
// hh x ww at level `from`'s resolution, as the two things that actually decide
// the footprint:
//
//   * `act`  - the activation arena's high-water mark. Each level allocates its
//     dup-up tensor once (where it has one) and one block's scratch, and the
//     arena is rewound per block, so a level costs dup + max(block scratch).
//   * `carry`- the two carry slots. A block ping-pongs them, so two tensors of
//     the level's output width are live, and a level that upsamples holds a
//     third at twice the resolution until the resample has added it in.
struct VaeLevelCost {
	u64 act = 0, carry = 0;
};

static VaeLevelCost vae_level_cost(const std::vector<i64>& dims, i64 from, i64 to, i64 hh, i64 ww) {
	const i64 nlevels = (i64)dims.size() - 1;
	VaeLevelCost c;
	for (i64 i = from; i < to; i++) {
		const i64 ic = dims[(size_t)i], oc = dims[(size_t)(i + 1)];
		const bool res = (i != nlevels - 1);
		u64 a = bytes_of(std::max(ic, oc), hh, ww);
		if (res) a += bytes_of(oc, hh * 2, ww * 2);
		c.act = std::max(c.act, a);
		c.carry = std::max(c.carry, 2 * bytes_of(oc, hh, ww));
		if (res) {
			// The upsample flips to the other slot and holds the result at twice
			// the resolution while the block output it is computed from is still
			// live: 1 + 4 of the level-resolution tensor.
			c.carry = std::max(c.carry, 5 * bytes_of(oc, hh, ww));
			hh *= 2;
			ww *= 2;
		}
	}
	return c;
}

// The tile size and split level to decode a canvas of H x W latent pixels with,
// or 0 for "run it whole". The decision is made from the machine's budget rather
// than from the canvas: the untiled decoder holds three of the last level's
// tensors at full resolution, which at 1080p is 3.6 GB before the weights, so
// past a shape the card can back the pass is split - levels [0, split) run once
// over the whole canvas and the rest runs one overlapping tile at a time. Both
// halves are costed with `vae_level_cost` and the split/tile pair whose peak the
// budget can hold is the one that runs.
i64 ImageVae::plan_decode(i64 H, i64 W, i64* split) const {
	const i64 nlevels = (i64)dims_.size() - 1;
	if (nlevels < 2) {
		*split = 0;
		return 0;
	}
	*split = 0;
	i64 want = 0;
	if (const char* e = getenv("PHI_QVAE_TILE")) want = atoll(e);
	const u64 budget = vram_budget().limit();
	const u64 held = vram_budget().local();
	const u64 room = (budget > held ? budget - held : 0);
	// `held` is what the accountant has already booked - the weights included -
	// and the upload ring is pinned *host* memory, so it is not part of the local
	// budget and must not be subtracted twice. The margin is for the small
	// long-lived buffers the phase carries (the latent, the readbacks).
	const u64 margin = 96ull << 20;
	const u64 act = room > margin ? room - margin : 0;
	const i64 H4 = H * cfg_.spatial, W4 = W * cfg_.spatial;

	// The whole-canvas pass: levels [0, nlevels) plus the head.
	VaeLevelCost whole = vae_level_cost(dims_, 0, nlevels, H, W);
	whole.act = std::max<u64>(whole.act, bytes_of(dims_.back(), H4, W4) +
	                                           bytes_of(cfg_.image_channels, H4, W4));
	const u64 whole_peak = whole.act + whole.carry;
	if (want <= 0 && whole_peak <= act) {
		last_tile_ = 0;
		return 0;
	}

	i64 best_tile = 0, best_split = 0;
	for (i64 k = nlevels - 1; k >= 1; k--) {
		const i64 sh = nlevels - 1 - k;
		const i64 ph = H4 >> sh, pw = W4 >> sh;
		const VaeLevelCost pre = vae_level_cost(dims_, 0, k, H, W);
		const u64 pre_tensor = bytes_of(dims_[(size_t)k], ph, pw);
		const i64 halo = suffix_halo(k);
		const i64 cap = want > 0 ? want : std::min<i64>(1024, std::max(H4, W4));
		for (i64 T = cap; T >= 64; T -= 64) {
			if (T < 64) break;
			const i64 lh = T / (1 << sh) + 2 * halo;
			VaeLevelCost suf = vae_level_cost(dims_, k, nlevels, lh, lh);
			const i64 fh = lh << sh;
			suf.act = std::max<u64>(suf.act, bytes_of(dims_.back(), fh, fh) +
			                                    bytes_of(cfg_.image_channels, fh, fh));
			const u64 carry = std::max(pre.carry, pre_tensor + suf.carry);
			// The crop the tile is fed from, and the safety factor: the estimate
			// is a model of an allocator whose chunks are rounded up, and a plan
			// that is *just* inside the budget is a plan that fails on the
			// rounding.
			const u64 crop = bytes_of(dims_[(size_t)k], lh, lh);
			const u64 peak = (std::max(pre.act, suf.act) + carry + crop) * 11 / 10 + (32ull << 20);
			if (qvae_trace())
				fprintf(stderr, "[qvae]  cand split %lld tile %lld lh %lld peak %s (act %s) pre %s/%s/%s suf %s/%s\n",
				        (long long)k, (long long)T, (long long)lh, format_bytes(peak).c_str(),
				        format_bytes(act).c_str(), format_bytes(pre.act).c_str(),
				        format_bytes(pre.carry).c_str(), format_bytes(pre_tensor).c_str(),
				        format_bytes(suf.act).c_str(), format_bytes(suf.carry).c_str());
			if (peak <= act) {
				if (T > best_tile) {
					best_tile = T;
					best_split = k;
				}
				break;   // the largest tile at this split is the cheapest run
			}
		}
	}
	if (best_tile <= 0) {
		// Nothing fits the accounting: take the cheapest shape anyway and let the
		// accountant's refusal name the shape that did not. A silent downgrade to
		// an untiled pass would only move the same failure later.
		best_split = std::max<i64>(1, nlevels - 2);
		best_tile = want > 0 ? want : 256;
	}
	if (qvae_trace())
		fprintf(stderr,
		        "[qvae] plan: budget %s, held %s, room %s, act %s | whole_peak %s -> "
		        "split %lld tile %lld\n",
		        format_bytes(budget).c_str(), format_bytes(held).c_str(), format_bytes(room).c_str(),
		        format_bytes(act).c_str(), format_bytes(whole_peak).c_str(),
		        (long long)best_split, (long long)best_tile);
	*split = best_split;
	if (const char* e = getenv("PHI_QVAE_SPLIT")) {
		const i64 v = atoll(e);
		if (v > 0) *split = std::clamp<i64>(v, 1, nlevels - 1);
	}
	if (want > 0) best_tile = want;
	best_tile = std::min(best_tile, std::max<i64>(H4, W4));
	best_tile = std::max<i64>(32, (best_tile / 16) * 16);
	last_tile_ = best_tile;
	return best_tile;
}

// ── ops ───────────────────────────────────────────────────────────────────

// `dup` folds the nearest 2x upsample into the convolution: the operand stays at
// h x w and the output is (2h)x(2w), so `Resample.forward`'s
// `interpolate(scale_factor=2, mode="nearest-exact")` followed by a 3x3 conv
// needs no materialised 2x tensor. Those two steps agree exactly for an integer
// scale factor of 2 (`nearest-exact`'s `floor((dst + 0.5) * scale_inv)` and the
// plain `floor(dst * scale_inv)` differ only for non-integer ratios), and the
// kernel's `iy < IH*dup` bound is the zero row/column the interpolation pads
// with. At a 1024x1024 canvas the tensor it removes is 144 channels of
// 1024x1024 fp32 - 576 MiB, and the decoder's arena keeps every chunk it grows,
// so the peak drops by about 4/3 of that.
void ImageVae::conv(const QVaeT& c, const GpuAlloc& x, const GpuAlloc& y, i64 h, i64 w,
                        i64 stride, bool br_pad, int dup) {
	Conv2dArgsG a;
	a.x = x;
	a.wgt = c.w;
	a.bias = c.b;
	a.y = y;
	a.n = 1;
	a.ic = c.ic;
	a.ih = h;
	a.iw = w;
	a.oc = c.oc;
	a.kh = c.kh;
	a.kw = c.kw;
	a.stride = stride;
	a.pad = (c.kh == 3 && stride == 1) ? 1 : 0;
	a.has_bias = c.has_bias;
	a.dup = dup;
	a.w_dtype = c.w_dtype;
	a.pad_right_bottom = br_pad;
	dispatch_conv2d(*g_->ctx, a);
}

void ImageVae::rms(const GpuAlloc& g, const GpuAlloc& x, const GpuAlloc& y, i64 c, i64 s) {
	ChwNormArgs a;
	a.x = x;
	a.w = g;
	a.y = y;
	a.rows = s;
	a.cols = c;
	a.eps = cfg_.norm_eps;
	dispatch_chw_norm(*g_->ctx, a);
}

void ImageVae::silu(const GpuAlloc& x, i64 n) {
	ElemArgs e;
	e.op = ElemOp::Silu;
	e.a = x;
	e.b = x;
	e.c = x;
	e.y = x;
	e.rows = 1;
	e.cols = n;
	dispatch_elem(*g_->ctx, e);
}

// Zero everything outside a tile's image footprint; a no-op for the whole-tensor
// pass (`valid` is null there).
void ImageVae::mask_rect(const GpuAlloc& x, i64 c, i64 h, i64 w, const i64* valid) {
	if (!valid) return;
	dispatch_mask_rect(*g_->ctx, x, c, h, w, valid[0], valid[1], valid[2], valid[3]);
}

void ImageVae::add(const GpuAlloc& a, const GpuAlloc& b, const GpuAlloc& y, i64 n) {
	ElemArgs e;
	e.op = ElemOp::Add;
	e.a = a;
	e.b = b;
	e.c = b;
	e.y = y;
	e.rows = 1;
	e.cols = n;
	dispatch_elem(*g_->ctx, e);
}

GpuAlloc ImageVae::alloc_carry(i64 c, i64 h, i64 w) {
	cur_ ^= 1;
	carry_[cur_].reset();
	return carry_[cur_].alloc(bytes_of(c, h, w));
}

GpuAlloc ImageVae::run_res(const QVaeRes& r, const GpuAlloc& x, const GpuAlloc& out, i64 ic,
                               i64 oc, i64 h, i64 w, GpuArena& act, const i64* valid) {
	const i64 s = h * w;
	const i64 big = std::max(ic, oc);
	GpuAlloc t = act.alloc(bytes_of(big, h, w));
	rms(r.g0, x, t, ic, s);
	silu(t, ic * s);
	conv(r.c2, t, out, h, w);          // ic -> oc, written straight into the output
	// The intermediate output is read back through `rms -> silu -> conv`, and a
	// convolution reads its input's padding as zero. Inside a tile that region is
	// the crop's halo, whose values are computed and therefore *not* zero: it has
	// to be masked here as well, not only on the block's final result.
	mask_rect(out, oc, h, w, valid);
	rms(r.g3, out, t, oc, s);
	silu(t, oc * s);
	conv(r.c6, t, out, h, w);          // oc -> oc, in place over the residual sum
	if (r.has_shortcut) {
		GpuAlloc sc = act.alloc(bytes_of(oc, h, w));
		conv(r.shortcut, x, sc, h, w);
		add(out, sc, out, oc * s);
	} else {
		add(out, x, out, oc * s);
	}
	mask_rect(out, oc, h, w, valid);
	return out;
}

GpuAlloc ImageVae::run_attn(const QVaeAttn& a, const GpuAlloc& x, const GpuAlloc& out, i64 c,
                                i64 h, i64 w, GpuArena& act) {
	auto& ctx = *g_->ctx;
	const i64 s = h * w;
	GpuAlloc t = act.alloc(bytes_of(c, h, w));
	rms(a.norm_g, x, t, c, s);
	qdump(ctx, "a_norm", t, c * s);
	GpuAlloc qkv_c = act.alloc(bytes_of(3 * c, h, w));
	conv(a.qkv, t, qkv_c, h, w);
	qdump(ctx, "a_qkv", qkv_c, 3 * c * s);

	// The attention is single-head over the `s` pixels with `c` features - the one
	// shape the engine's flash kernels cannot take (they assume head_dim 128 and
	// `c` here is 768 or 1152), so it runs as two GEMMs with a softmax between
	// them. `c` and `s` are both multiples of 32, which is what gemm_f16 needs.
	const u64 sec = (u64)c * s * 4;
	GpuAlloc q = act.alloc(sec), k = act.alloc(sec);
	dispatch_transpose_cs(ctx, slice(qkv_c, 0, sec), q, c, s, true);
	dispatch_transpose_cs(ctx, slice(qkv_c, sec, sec), k, c, s, true);
	// V stays in its conv layout: [C, S] is exactly the [n, k] `gemm_f16` wants
	// for the second product, so it is the one operand that needs no transpose.
	GpuAlloc v = slice(qkv_c, sec * 2, sec);

	// q *= 1/sqrt(c) so the softmax kernel needs no scale of its own
	{
		ElemArgs e;
		e.op = ElemOp::Scale;
		e.a = q;
		e.b = q;
		e.c = q;
		e.y = q;
		e.rows = 1;
		e.cols = s * c;
		e.alpha = 1.0f / std::sqrt((float)c);
		dispatch_elem(ctx, e);
	}

	GpuAlloc sc = act.alloc((u64)s * s * 4);
	GemmF16Args g1;
	g1.a = q;
	g1.b = k;
	g1.c = sc;
	g1.m = s;
	g1.n = s;
	g1.k = c;
	g1.a_is_f32 = true;
	g1.b_is_f32 = true;
	dispatch_gemm_f16(ctx, g1);
	qdump(ctx, "a_scores", sc, s * s);

	GpuAlloc pr = act.alloc((u64)s * s * 4);
	dispatch_row_softmax(ctx, sc, pr, s, s);
	qdump(ctx, "a_probs", pr, s * s);

	GpuAlloc o = act.alloc(sec);
	GemmF16Args g2;
	g2.a = pr;
	g2.b = v;   // [c, s] = [n, k] for this product
	g2.c = o;
	g2.m = s;
	g2.n = c;
	g2.k = s;
	g2.a_is_f32 = true;
	g2.b_is_f32 = true;
	dispatch_gemm_f16(ctx, g2);
	qdump(ctx, "a_o", o, s * c);

	GpuAlloc oc2 = act.alloc(bytes_of(c, h, w));
	dispatch_transpose_cs(ctx, o, oc2, c, s, false);
	qdump(ctx, "a_oc2", oc2, c * s);
	conv(a.proj, oc2, out, h, w);
	add(out, x, out, c * s);
	return out;
}

// ── decode ────────────────────────────────────────────────────────────────

// One pass of the decoder's upsample stack. Everything the tile logic needs to
// know about a level is here, so the untiled decode and a tile's decode run the
// exact same code over different rectangles.
GpuAlloc ImageVae::run_up_levels(const GpuAlloc& cur_in, i64 from, i64 to, i64 h, i64 w,
                                     i64* out_h, i64* out_w, const i64* valid) {
	auto& ctx = *g_->ctx;
	GpuAlloc cur = cur_in;
	// `valid` is the image footprint at the resolution level `from` runs at; the
	// halo around it holds computed values the reference does not have (every
	// conv there reads a zero), so each stage's output is masked back to it.
	i64 vy0 = 0, vy1 = 0, vx0 = 0, vx1 = 0;
	if (valid) {
		vy0 = valid[0];
		vy1 = valid[1];
		vx0 = valid[2];
		vx1 = valid[3];
	}
	auto mask = [&](const GpuAlloc& x, i64 c, i64 hh, i64 ww, i64 y0, i64 y1, i64 x0, i64 x1) {
		if (!valid) return;
		dispatch_mask_rect(ctx, x, c, hh, ww, y0, y1, x0, x1);
	};
	(void)mask;
	for (i64 i = from; i < to; i++) {
		const QVaeUp& u = up_[(size_t)i];
		GpuArena::Mark m = act_.mark();
		if (qvae_trace())
			fprintf(stderr,
			        "[qvae] level %lld in %lldx%lld ic=%lld oc=%lld | weights %s act %s carry %s+%s\n",
			        (long long)i, (long long)h, (long long)w, (long long)u.ic, (long long)u.oc,
			        format_bytes(warena_.capacity()).c_str(), format_bytes(act_.capacity()).c_str(),
			        format_bytes(carry_[0].capacity()).c_str(),
			        format_bytes(carry_[1].capacity()).c_str());
		{
			char nm[32];
			snprintf(nm, sizeof(nm), "d_up%lld_pre", (long long)i);
			qdump(ctx, nm, cur, u.ic * h * w);
		}
		// The resample's shortcut is the *block input*'s channel-to-pixel shuffle
		// (`Up_ResidualBlock.forward` adds `avg_shortcut(x)` to the main path's
		// result), so it has to be taken from the input **before** the residual
		// loop: the loop ping-pongs the two carry arenas, and its second block
		// lands back in the one the stage input occupies.
		GpuAlloc dup;
		if (u.has_resample) {
			dup = act_.alloc(bytes_of(u.oc, h * 2, w * 2));
			DupUp2xArgs da;
			da.x = cur;
			da.y = dup;
			da.ic = u.ic;
			da.oc = u.oc;
			da.h = h;
			da.w = w;
			da.ft = u.dup_ft;
			dispatch_dupup2x(ctx, da);
			qdump(ctx, "d_rs_dup0", dup, u.oc * (h * 2) * (w * 2));
		}
		i64 ch = u.ic;
		for (size_t b = 0; b < u.blocks.size(); b++) {
			// A block's scratch is dead as soon as it returns (its input and its
			// result live in the carry arenas), so the arena is rewound per block
			// rather than per level: the high-water mark becomes one block's
			// scratch instead of every block's, which at 1080p is the difference
			// between 1.2 GB and 302 MB of `vae.acts`.
			GpuArena::Mark bm = act_.mark();
			GpuAlloc out = alloc_carry(u.oc, h, w);
			const i64 vr[4] = {vy0, vy1, vx0, vx1};
			run_res(u.blocks[b], cur, out, ch, u.oc, h, w, act_, valid ? vr : nullptr);
			act_.rewind_to(bm);
			cur = out;
			ch = u.oc;
		}
		if (u.has_resample) {
			// resample = nearest-2x upsample then a 3x3 conv, with the upsample
			// folded into the convolution's window walk (`dup`).
			GpuAlloc out = alloc_carry(u.oc, h * 2, w * 2);
			conv(u.resample, cur, out, h, w, 1, false, 2);
			add(out, dup, out, u.oc * (h * 2) * (w * 2));
			cur = out;
			h *= 2;
			w *= 2;
			vy0 *= 2; vy1 *= 2; vx0 *= 2; vx1 *= 2;
			mask(out, u.oc, h, w, vy0, vy1, vx0, vx1);
		}
		act_.rewind_to(m);
	}
	*out_h = h;
	*out_w = w;
	return cur;
}

// The decoder head: `rms -> silu -> 3x3 conv to RGBA`, downloaded to the host.
void ImageVae::run_head(const GpuAlloc& cur, i64 h, i64 w, std::vector<float>& out_rgba) {
	auto& ctx = *g_->ctx;
	GpuArena::Mark m = act_.mark();
	const i64 c = dec_head_.ic;
	GpuAlloc t = act_.alloc(bytes_of(c, h, w));
	rms(dec_head_g_, cur, t, c, h * w);
	silu(t, c * h * w);
	GpuAlloc rgba = act_.alloc(bytes_of(cfg_.image_channels, h, w));
	conv(dec_head_, t, rgba, h, w);
	ctx.submit();
	out_rgba.resize((size_t)cfg_.image_channels * h * w);
	ctx.download(rgba.res, rgba.off, out_rgba.data(), out_rgba.size() * 4);
	act_.rewind_to(m);
}

std::vector<float> ImageVae::decode(const std::vector<float>& z, i64 H, i64 W) {
	if (!open_) throw MediaError("qwen_vae: not open");
	load_weights();
	auto& ctx = *g_->ctx;
	const i64 Z = cfg_.latent_channels;
	if ((i64)z.size() != Z * H * W) throw MediaError("qwen_vae: latent shape mismatch");
	const i64 nlevels = (i64)dims_.size() - 1;
	i64 split = 0;
	const i64 tile = plan_decode(H, W, &split);

	// The latent and the head's output are the only full-size host buffers; they
	// are uploaded/downloaded through the staging ring.
	act_.reset();
	carry_[0].reset();
	carry_[1].reset();
	cur_ = 1;   // the first alloc_carry() flips to 0

	ctx.begin();
	// The latent is only read by `conv2`, and it is given up again straight after
	// so that the activation arena's every later phase starts at its origin (see
	// the note on `in_`).
	GpuArena::Mark zm = act_.mark();
	GpuAlloc zi = act_.alloc(bytes_of(Z, H, W));
	upload_range(ctx, *g_->ring, zi.res, zi.off, z.data(), z.size() * 4);
	ctx.submit();
	g_->ring->rewind();
	ctx.begin();

	GpuAlloc cur = alloc_carry(Z, H, W);
	conv(root_conv2_, zi, cur, H, W);
	qdump(ctx, "d_conv2", cur, Z * H * W);
	act_.rewind_to(zm);

	i64 h = H, w = W;
	// decoder.conv1 widens to the first level's channel count
	{
		GpuArena::Mark m = act_.mark();
		GpuAlloc out = alloc_carry(dec_conv1_.oc, h, w);
		conv(dec_conv1_, cur, out, h, w);
		qdump(ctx, "d_conv1", out, dec_conv1_.oc * h * w);
		act_.rewind_to(m);
		cur = out;
	}
	// middle: residual, attention, residual — all at the same shape
	for (int pass = 0; pass < 3; pass++) {
		GpuArena::Mark m = act_.mark();
		GpuAlloc out = alloc_carry(dec_conv1_.oc, h, w);
		if (pass == 1) run_attn(dec_attn_, cur, out, dec_conv1_.oc, h, w, act_);
		else run_res(dec_mid_[pass == 0 ? 0 : 1], cur, out, dec_conv1_.oc, dec_conv1_.oc, h, w,
		             act_);
		{
			char nm[32];
			snprintf(nm, sizeof(nm), "d_mid%d", pass);
			qdump(ctx, nm, out, dec_conv1_.oc * h * w);
		}
		act_.rewind_to(m);
		cur = out;
	}

	// A full-resolution latent canvas costs three of the last level's tensors at
	// (H*16)x(W*16) fp32 - two carries and the head's normalised copy - which at
	// 1080p is 3.6 GB, on top of the 832 MB of weights. Above the size the
	// accountant can back, the decode is therefore split: the prefix (levels
	// [0, split)) runs once over the whole canvas and comes back to the host, and
	// the rest runs one overlapping tile at a time. Each tile is cropped with
	// `suffix_halo()` pixels of context - the receptive field of the levels it
	// runs - and only its interior is kept, so every kept pixel is the value the
	// untiled decode computes. The tiling is exact, not blended.
	if (tile > 0) {
		GpuAlloc pre = run_up_levels(cur, 0, split, h, w, &h, &w);
		const i64 pre_c = dims_[(size_t)split];
		ctx.submit();
		std::vector<float> mid((size_t)pre_c * h * w);
		ctx.download(pre.res, pre.off, mid.data(), mid.size() * 4);
		// the prefix is on the host now: every arena it lived in can go back
		carry_[0].reset();
		carry_[1].reset();
		cur_ = 1;
		act_.reset();

		const i64 shift = 1 << (nlevels - 1 - split);   // full res / prefix res
		const i64 H4 = h * shift, W4 = w * shift;
		const i64 halo = suffix_halo(split);
		const i64 keep0 = shift * halo;                  // the kept window's origin
		const i64 C4 = cfg_.image_channels;
		std::vector<float> canvas((size_t)C4 * H4 * W4, 0.0f);
		i64 tiles = 0;
		for (i64 y0 = 0; y0 < H4; y0 += tile) {
			const i64 ty = std::min(tile, H4 - y0);
			for (i64 x0 = 0; x0 < W4; x0 += tile) {
				const i64 tx = std::min(tile, W4 - x0);
				// run_head() below closes the bracket, so every tile opens its own
				ctx.begin();
				// the prefix-space rectangle this tile reads: the tile's own
				// footprint (ty/shift, tx/shift) grown by the halo on every side
				const i64 lh = ty / shift + 2 * halo, lw = tx / shift + 2 * halo;
				const i64 oy = y0 / shift - halo, ox = x0 / shift - halo;
				std::vector<float> crop((size_t)pre_c * (size_t)lh * (size_t)lw, 0.0f);
				for (i64 c = 0; c < pre_c; c++) {
					const float* src = &mid[(size_t)c * h * w];
					float* dst = &crop[(size_t)c * lh * lw];
					for (i64 a = 0; a < lh; a++) {
						const i64 sy = oy + a;
						if (sy < 0 || sy >= h) continue;   // outside the canvas: zero
						for (i64 b = 0; b < lw; b++) {
							const i64 sx = ox + b;
							if (sx < 0 || sx >= w) continue;
							dst[a * lw + b] = src[sy * w + sx];
						}
					}
				}
				in_.reset();
				GpuAlloc cin = in_.alloc(bytes_of(pre_c, lh, lw));
				// Growing the arena can drain the queue (GpuArena::alloc drops the
				// chunk list when it is fully rewound), which closes the bracket the
				// upload below needs.
				if (!ctx.recording()) ctx.begin();
				upload_range(ctx, *g_->ring, cin.res, cin.off, crop.data(), crop.size() * 4);
				// The tile's image footprint, in the crop's own coordinates: what
				// the reference defines and what it reads as zero outside.
				const i64 rect[4] = {std::max<i64>(0, -oy), std::min<i64>(lh, h - oy),
				                     std::max<i64>(0, -ox), std::min<i64>(lw, w - ox)};
				i64 th = 0, tw = 0;
				GpuAlloc tcur = run_up_levels(cin, split, nlevels, lh, lw, &th, &tw, rect);
				std::vector<float> raw;
				run_head(tcur, th, tw, raw);
				g_->ring->rewind();
				act_.reset();
				if (th < keep0 + ty || tw < keep0 + tx)
					throw MediaError("qwen_vae: decode tile came back too small");
				for (i64 c = 0; c < C4; c++) {
					const float* src = &raw[(size_t)c * th * tw];
					float* dst = &canvas[(size_t)c * H4 * W4];
					for (i64 yy = 0; yy < ty; yy++)
						memcpy(&dst[(y0 + yy) * W4 + x0],
						       &src[(keep0 + yy) * tw + keep0], (size_t)tx * 4);
				}
				tiles++;
			}
		}
		last_tile_ = tile;
		last_split_ = split;
		last_tiles_ = tiles;
		if (qvae_trace())
			fprintf(stderr, "[qvae] tiled decode: split %lld, tile %lld px, %lld tiles, "
			                "weights %s act %s carry %s+%s\n",
			        (long long)split, (long long)tile, (long long)tiles,
			        format_bytes(warena_.capacity()).c_str(), format_bytes(act_.capacity()).c_str(),
			        format_bytes(carry_[0].capacity()).c_str(),
			        format_bytes(carry_[1].capacity()).c_str());
		std::vector<float> chw((size_t)3 * H4 * W4);
		for (i64 cix = 0; cix < 3; cix++)
			memcpy(&chw[(size_t)cix * H4 * W4], &canvas[(size_t)cix * H4 * W4],
			       (size_t)H4 * W4 * 4);
		std::vector<float> hwc;
		chw_to_hwc(chw.data(), 3, (int)H4, (int)W4, hwc);
		release_activation_memory();
		g_->ring->rewind();
		return hwc;
	}

	last_tile_ = 0;
	last_split_ = 0;
	last_tiles_ = 0;
	cur = run_up_levels(cur, 0, nlevels, h, w, &h, &w, nullptr);
	if (qvae_trace())
		fprintf(stderr, "[qvae] decode end of levels: weights %s act %s carry %s+%s\n",
		        format_bytes(warena_.capacity()).c_str(), format_bytes(act_.capacity()).c_str(),
		        format_bytes(carry_[0].capacity()).c_str(), format_bytes(carry_[1].capacity()).c_str());
	std::vector<float> raw;
	run_head(cur, h, w, raw);

	std::vector<float> chw((size_t)3 * h * w);
	for (i64 cix = 0; cix < 3; cix++)
		memcpy(&chw[(size_t)cix * h * w], &raw[(size_t)cix * h * w], (size_t)h * w * 4);
	std::vector<float> hwc;
	chw_to_hwc(chw.data(), 3, (int)h, (int)w, hwc);
	release_activation_memory();
	g_->ring->rewind();
	return hwc;
}

// ── encode ────────────────────────────────────────────────────────────────

std::vector<float> ImageVae::encode(const std::vector<float>& img, i64 H, i64 W) {
	if (!open_) throw MediaError("qwen_vae: not open");
	load_weights();
	auto& ctx = *g_->ctx;
	const i64 C = cfg_.image_channels;
	if ((i64)img.size() != C * H * W) throw MediaError("qwen_vae: image shape mismatch");

	act_.reset();
	carry_[0].reset();
	carry_[1].reset();
	cur_ = 1;

	ctx.begin();
	GpuAlloc xi = act_.alloc(bytes_of(C, H, W));
	upload_range(ctx, *g_->ring, xi.res, xi.off, img.data(), img.size() * 4);
	ctx.submit();
	g_->ring->rewind();
	ctx.begin();

	i64 h = H, w = W;
	GpuAlloc cur = alloc_carry(cfg_.enc_dim, h, w);
	{
		GpuArena::Mark m = act_.mark();
		conv(enc_conv1_, xi, cur, h, w);
		qdump(ctx, "e_conv1", cur, cfg_.enc_dim * h * w);
		act_.rewind_to(m);
	}
	for (size_t i = 0; i < down_.size(); i++) {
		const QVaeDown& d = down_[i];
		GpuArena::Mark m = act_.mark();
		// The shortcut is the average of the *block's* input, computed before the
		// path strides it down (and, at the last stage, at the input's own
		// resolution: there it is just `x`).
		const i64 oh = d.has_resample ? h / 2 : h;
		const i64 ow = d.has_resample ? w / 2 : w;
		GpuAlloc avg = act_.alloc(bytes_of(d.oc, oh, ow));
		if (d.has_resample) {
			conv(d.avg, cur, avg, h, w, 2);
		} else {
			ElemArgs e;
			e.op = ElemOp::Copy;
			e.a = cur;
			e.b = cur;
			e.c = cur;
			e.y = avg;
			e.rows = d.oc;
			e.cols = oh * ow;
			dispatch_elem(ctx, e);
		}
		{
			char nm[32];
			snprintf(nm, sizeof(nm), "e_avg%zu", i);
			qdump(ctx, nm, avg, d.oc * oh * ow);
		}
		i64 ch = d.ic;
		for (size_t b = 0; b < d.blocks.size(); b++) {
			GpuAlloc out = alloc_carry(d.oc, h, w);
			run_res(d.blocks[b], cur, out, ch, d.oc, h, w, act_);
			cur = out;
			ch = d.oc;
		}
		{
			char nm[32];
			snprintf(nm, sizeof(nm), "e_res%zu", i);
			qdump(ctx, nm, cur, d.oc * h * w);
		}
		if (d.has_resample) {
			GpuAlloc out = alloc_carry(d.oc, h / 2, w / 2);
			conv(d.resample, cur, out, h, w, 2, /*br_pad=*/true);
			{
				char nm[32];
				snprintf(nm, sizeof(nm), "e_conv%zu", i);
				qdump(ctx, nm, out, d.oc * (h / 2) * (w / 2));
			}
			add(out, avg, out, d.oc * (h / 2) * (w / 2));
			cur = out;
			h /= 2;
			w /= 2;
		} else {
			GpuAlloc out = alloc_carry(d.oc, h, w);
			add(cur, avg, out, d.oc * h * w);
			cur = out;
		}
		act_.rewind_to(m);
	}
	const i64 c = enc_head_.ic;
	for (int pass = 0; pass < 3; pass++) {
		GpuArena::Mark m = act_.mark();
		GpuAlloc out = alloc_carry(c, h, w);
		if (pass == 1) run_attn(enc_attn_, cur, out, c, h, w, act_);
		else run_res(enc_mid_[pass == 0 ? 0 : 1], cur, out, c, c, h, w, act_);
		{
			char nm[32];
			snprintf(nm, sizeof(nm), "e_mid%d", pass);
			qdump(ctx, nm, out, c * h * w);
		}
		act_.rewind_to(m);
		cur = out;
	}
	GpuAlloc mu;
	{
		GpuArena::Mark m = act_.mark();
		GpuAlloc t = act_.alloc(bytes_of(c, h, w));
		rms(enc_head_g_, cur, t, c, h * w);
		silu(t, c * h * w);
		GpuAlloc z2 = act_.alloc(bytes_of(cfg_.z2, h, w));
		conv(enc_head_, t, z2, h, w);
		qdump(ctx, "e_z2", z2, cfg_.z2 * h * w);
		mu = act_.alloc(bytes_of(cfg_.latent_channels, h, w));
		conv(root_conv1_, z2, mu, h, w);   // 1x1, 128 -> 128; the first half is mu
		qdump(ctx, "e_mu", mu, cfg_.z2 * h * w);
		ctx.submit();
		std::vector<float> raw((size_t)cfg_.z2 * h * w);
		ctx.download(mu.res, mu.off, raw.data(), raw.size() * 4);
		act_.rewind_to(m);

		if (cfg_.mean.size() == (size_t)cfg_.latent_channels &&
		    cfg_.std_.size() == (size_t)cfg_.latent_channels) {
			std::vector<float> out((size_t)cfg_.latent_channels * h * w);
			for (i64 ci = 0; ci < cfg_.latent_channels; ci++) {
				const float m = cfg_.mean[(size_t)ci], sd = cfg_.std_[(size_t)ci];
				const float* src = &raw[(size_t)ci * h * w];
				float* dst = &out[(size_t)ci * h * w];
				for (i64 p = 0; p < h * w; p++) dst[p] = (src[p] - m) / sd;
			}
			release_activation_memory();
			g_->ring->rewind();
			return out;
		}
		std::vector<float> out(raw.begin(), raw.begin() + (size_t)cfg_.latent_channels * h * w);
		release_activation_memory();
		g_->ring->rewind();
		return out;
	}
}

}  // namespace phi::media
