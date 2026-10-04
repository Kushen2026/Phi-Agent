// Qwen-Image-2.1 VAE (the Wan 2.2 conv layout, spatial only for stills).
//
// Mirrors C:/ComfyUI/comfy/ldm/wan/vae2_2.py for the configuration ComfyUI's
// VAELoader picks for `qwen_image_2.1_vae_bf16.safetensors` (comfy/sd.py, the
// `decoder.middle.0.residual.0.gamma` branch):
//
//   dim 96 / dec_dim 144 / z_dim 64 / dim_mult [1,2,4,8,8] / num_res_blocks 2
//   image_channels 4 (RGBA), patch_size 1, temporal_kernel 1
//   16x spatial compression, 64 latent channels
//
// Two things follow from `temporal_kernel 1` and the image path's T == 1 input,
// and they are what make this a plain 2D VAE:
//
//   * every Conv3d in the checkpoint has a temporal kernel of 1, so it *is* a
//     2D conv (the 1x1x1 convs are the `conv1` / `conv2` / `shortcut` layers);
//   * `Resample.forward` only touches the temporal axis inside its
//     `if feat_cache is not None` branch, and `WanVAE.decode` passes no cache
//     for a 4-D latent, so `time_conv` never runs. Same for the encoder.
//
// Everything else is transcribed literally, including the two ops that are
// peculiar to this VAE family:
//
//   * `AvgDown3D` - the strided downsample's shortcut: a mean over a block of
//     `in*ft*4/out` flat (channel, f_t, row, column) indices, with the frame
//     occupying a single f_t slot and the other one padding zeros. It is a plain
//     2x2 stride-2 conv with a sparse, precomputed kernel, so it needs no device
//     code of its own.
//   * `DupUp3D` - the upsample's shortcut: a channel-to-pixel shuffle
//     (`out[c][2h+i][2w+j] = in[(ft*4*c + 4*(ft-1) + 2i + j)/repeats][h][w]`,
//     kernel `dupup2x`) added to the main path.
//
// The normalisation is `RMS_norm` (comfy/ldm/wan/vae.py): L2 over the *channel*
// axis times sqrt(C) times a learned gamma, which no other norm in the engine
// implements, hence `chw_rmsnorm`.
#pragma once

#include <string>
#include <vector>

#include "models/model_common.hpp"

namespace phi::media {

struct ImageVaeConfig {
	i64 latent_channels = 64;
	i64 enc_dim = 96;    // encoder.conv1's output width
	i64 dec_dim = 144;   // decoder.head.0.gamma's width
	i64 z2 = 128;        // encoder head output (2 * latent_channels)
	i64 image_channels = 4;   // RGBA in and out (the VAE is trained on alpha)
	i64 spatial = 16;         // encoder downsamples 2^4
	std::vector<i64> dim_mult{1, 2, 4, 8, 8};
	i64 num_res_blocks = 2;
	float norm_eps = 1e-12f;  // RMS_norm normalises over L2, no eps on the mean
	// Latent (de)normalisation, read off the latent format (`process_in` /
	// `process_out`): encode() returns (mu - mean)/std, and the DiT's reference
	// latents go through the same map *inside the model*, so decode() expects a
	// latent already in that space.
	std::vector<float> mean, std_;
};

// One weight tensor plus its shape, or the per-channel gamma of an RMS_norm.
struct QVaeT {
	GpuAlloc w, b;
	i64 oc = 0, ic = 0, kh = 1, kw = 1;
	bool has_bias = false;
	// `w` holds the checkpoint's own storage width (bf16 for the shipped file, and
	// fp16 or fp32 for a re-exported one); the conv kernel decodes the operands as
	// it loads them. See `Conv2dArgsG::w_dtype`.
	int w_dtype = 0;
};

struct QVaeRes {
	QVaeT c2, c6, shortcut;
	GpuAlloc g0, g3;       // residual.0.gamma / residual.3.gamma
	bool has_shortcut = false;
};

struct QVaeAttn {
	QVaeT qkv, proj;
	GpuAlloc norm_g;
};

struct QVaeUp {
	std::vector<QVaeRes> blocks;   // num_res_blocks + 1 of them
	QVaeT resample;                // resample.1, only when has_resample
	QVaeT shortcut;                // the dup-up shortcut of the same stage
	bool has_resample = false;
	// `Up_ResidualBlock`'s DupUp3D carries a temporal factor as well
	// (`temperal_upsample`): 2 for the first three decoder stages, 1 for the last
	// one. With a single frame it only changes the channel gather.
	i64 dup_ft = 1;
	i64 ic = 0, oc = 0;
};

struct QVaeDown {
	std::vector<QVaeRes> blocks;   // num_res_blocks of them
	QVaeT resample;
	QVaeT avg;                     // the sparse strided average kernel
	bool has_resample = false;
	// `Down_ResidualBlock` always adds `AvgDown3D(x)`, even at the last stage
	// where its factors are both 1 and the op is the identity - that extra `+ x`
	// is part of the reference. Its own temporal factor (`temperal_downsample`)
	// is 2 from the second stage on.
	i64 avg_ft = 1;
	i64 ic = 0, oc = 0;
};

class ImageVae {
public:
	void open(const std::string& path, GpuCtx* gpu);
	const ImageVaeConfig& config() const { return cfg_; }

	// z: [latent_channels, H, W] fp32, H and W in latent pixels.
	// Returns the decoded image as interleaved HWC [spatial*H, spatial*W, 3] fp32
	// in [-1, 1] (the checkpoint is RGBA; the alpha plane is dropped).
	std::vector<float> decode(const std::vector<float>& z, i64 H, i64 W);

	// img: [4, H, W] fp32 in [-1, 1] (RGBA; the caller pads RGB with alpha = 1).
	// Returns the *normalised* latent (mu - mean)/std, [latent_channels, H/16, W/16].
	std::vector<float> encode(const std::vector<float>& img, i64 H, i64 W);

	// Hands the activation scratch back to the device (the decoder's own boundary
	// before the next phase, same argument as ImageVae::release_activation_memory).
	void release_activation_memory();
	// ... and the weights as well: the two media chains share one budget and this
	// one cannot use the DiT's window. `decode`/`encode` reload them (out of the
	// page cache, ~0.2 s of the multi-second decode).
	void release_gpu_memory();

	// Activation peak for a latent of H x W, for the tool's budget report.
	static u64 estimate_vae_activation_bytes(i64 H, i64 W, const ImageVaeConfig& cfg);

	// The output-pixel tile the last decode() ran with (0 = untiled), and the
	// upsample level it started tiling at. For the report.
	i64 last_tile() const { return last_tile_; }
	i64 last_split() const { return last_split_; }
	// How many tiles the last decode() walked.
	i64 last_tiles() const { return last_tiles_; }
	// What the last decode()'s plan charged for: the tile plane's activations plus
	// the carries. This is the number the tile/split decision was made against, so
	// the report and the decision cannot drift apart.

private:
	void probe();
	void load_weights();
	void load_conv(const std::string& name, QVaeT& c);
	void load_gamma(const std::string& name, GpuAlloc& g);
	void load_res(const std::string& prefix, QVaeRes& r, i64 ic, i64 oc);
	void load_attn(const std::string& prefix, QVaeAttn& a, i64 c);
	void load_avg_kernel(QVaeT& c, i64 ic, i64 oc, i64 ft);

	// ops
	// `dup` = 2 folds a nearest 2x upsample into the convolution (the operand is
	// h x w, the output 2h x 2w); see the note on the definition.
	void conv(const QVaeT& c, const GpuAlloc& x, const GpuAlloc& y, i64 h, i64 w,
	          i64 stride = 1, bool br_pad = false, int dup = 1);

	// ── the tiled decode ──
	// Decoder dims, `dd[i]` being the channel count the i-th upsample level reads
	// (and `dd[i+1]` what it writes). Filled by load_weights().
	std::vector<i64> dims_;
	// Runs upsample levels [from, to) over `cur_in` (dims_[from] channels, h x w)
	// and returns the tensor the chain ends in, with its resolution in *out_h/*out_w.
	// `valid` (4 values: y0,y1,x0,x1, in this level's own resolution units)
	// is the image footprint inside a tile's crop, or null for the whole tensor.
	GpuAlloc run_up_levels(const GpuAlloc& cur_in, i64 from, i64 to, i64 h, i64 w, i64* out_h,
	                       i64* out_w, const i64* valid = nullptr);
	// The decoder head (RMS -> SiLU -> 3x3 conv to RGBA) over `cur`, downloaded.
	void run_head(const GpuAlloc& cur, i64 h, i64 w, std::vector<float>& out_rgba);
	// The receptive-field radius of levels [split, dims_.size()-1) plus the head, in
	// units of the resolution level `split` runs at. Every output pixel the tiled
	// pass keeps is at least this far from its tile's edge, which is what makes the
	// tiled decode bit-identical to the untiled one.
	i64 suffix_halo(i64 split) const;
	// The output-pixel tile to decode (0 = untiled) and the level to start tiling
	// at, for a canvas of H x W latent pixels.
	i64 plan_decode(i64 H, i64 W, i64* split) const;

	mutable i64 last_tile_ = 0;
	mutable i64 last_split_ = 0;
	mutable i64 last_tiles_ = 0;
	void rms(const GpuAlloc& g, const GpuAlloc& x, const GpuAlloc& y, i64 c, i64 s);
	void silu(const GpuAlloc& x, i64 n);
	void add(const GpuAlloc& a, const GpuAlloc& b, const GpuAlloc& y, i64 n);
	// Zero everything of `x` outside the image footprint `valid` (y0,y1,x0,x1);
	// a no-op when `valid` is null, which is the whole-tensor pass.
	void mask_rect(const GpuAlloc& x, i64 c, i64 h, i64 w, const i64* valid);
	// The next carry slot: flips the ping-pong and reuses the arena that has been
	// free since two stages ago.
	GpuAlloc alloc_carry(i64 c, i64 h, i64 w);
	GpuAlloc run_res(const QVaeRes& r, const GpuAlloc& x, const GpuAlloc& out, i64 ic, i64 oc,
	                 i64 h, i64 w, GpuArena& act, const i64* valid = nullptr);
	GpuAlloc run_attn(const QVaeAttn& a, const GpuAlloc& x, const GpuAlloc& out, i64 c, i64 h,
	                  i64 w, GpuArena& act);

	GpuCtx* g_ = nullptr;
	ImageVaeConfig cfg_;
	SafeTensors st_;
	std::string path_;
	bool open_ = false;
	bool weights_loaded_ = false;

	GpuArena warena_;    // every conv weight, fp32
	GpuArena act_;       // per-op scratch, rewound around each op
	GpuArena carry_[2];  // the ping-pong the chain is carried in
	// The tiled decode's input crop. It lives outside `act_` so that every
	// upsample level starts its own scratch at the arena's origin: a bump
	// allocator only hands back chunks when it grows *from the start*, which is
	// what turns the activation high-water mark from the sum of the levels into
	// their maximum.
	GpuArena in_;
	int cur_ = 0;

	// decoder
	QVaeT dec_conv1_, dec_head_;
	QVaeRes dec_mid_[2];
	QVaeAttn dec_attn_;
	std::vector<QVaeUp> up_;
	GpuAlloc dec_head_g_;

	// encoder
	QVaeT enc_conv1_, enc_head_;
	std::vector<QVaeDown> down_;
	QVaeRes enc_mid_[2];
	QVaeAttn enc_attn_;
	GpuAlloc enc_head_g_;

	// the two top-level 1x1 convs: `conv1` (z2 -> z2, the encoder's latent
	// projection) and `conv2` (z -> z, the decoder's)
	QVaeT root_conv1_, root_conv2_;
};

}  // namespace phi::media
