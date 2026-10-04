// MiniMax H3 video VAE (causal 3D convolutional, 24-channel latent).
//
// A frozen interface: the chain consumes `decode()`/`encode()` only.
//
// The decoder is a causal 3D-conv autoencoder: 16x spatial compression and a
// temporal compression of 16 frames per latent step. "Causal" means the temporal
// axis never peeks at future frames, so the first frame is special-cased (the
// padding is zero on the left of the time axis, not symmetric). The encoder is
// the mirror image; the H3 pipeline only needs the decoder for generation and
// the encode path for the ref2va reference latents.
#pragma once

#include <string>
#include <vector>

#include "models/model_common.hpp"

namespace phi::media {

struct VideoVaeConfig {
	i64 in_channels = 3;
	i64 z_channels = 24;
	i64 ch = 128;
	std::vector<i64> ch_mult{1, 2, 4, 4};
	i64 num_res_blocks = 2;
	i64 spatial_factor = 16;    // 2^4
	i64 temporal_factor = 16;
	i64 norm_groups = 32;
	float norm_eps = 1e-6f;
};

class VideoVae {
public:
	void open(const std::string& path, GpuCtx* gpu);
	const VideoVaeConfig& config() const { return cfg_; }

	// z: [z_channels, T, H, W] fp32 row major (latent space).
	// -> planar [3, T*temporal_factor, H*spatial_factor, W*spatial_factor] fp32 in
	//    [0, 1]: the decode ends in `pixel * std + mean` followed by a clamp to
	//    [0,1] (the reference does the same, which is why the byte conversion on
	//    the consumer side is `float01_to_srgb8`, *not* the image path's [-1,1]
	//    `float_to_srgb8`).
	// `tile` is the pixel size of the spatial decode tile. 0 (the default) asks
	// the planner for the largest tile this machine's VRAM budget can hold: a
	// bigger tile is both faster (fewer overlapping passes, less duplicated work)
	// and closer to the untiled reference, and the planner falls back to a smaller
	// one when the accountant refuses the allocation.
	std::vector<float> decode(const std::vector<float>& z, i64 T, i64 H, i64 W, i64 tile = 0);

	// The tile size a decode will use: the reference's 256 px when its 768 MB peak
	// fits this machine's budget (see VramBudget), a smaller tile when it does not.
	// `H`/`W` are latent dims; the canvas is 16x that. PHI_VAE_TILE overrides it,
	// which is how a larger (faster, closer to untiled) tile is opted into - see
	// the comment in video_vae.cpp for why that is not the default.
	static i64 plan_tile(i64 H, i64 W);
	// Bytes one tile's decode peak is estimated to need, calibrated at 256 px
	// (768 MB on this checkpoint) and scaled with the tile area.
	static u64 estimate_tile_bytes(i64 tile);
	// The tile the last decode() ran with (for the report).
	i64 last_tile() const { return last_tile_; }

	// x: planar [3, T, H, W] fp32 in [-1, 1] -> [z_channels, T/temporal_factor,
	//    H/16, W/16]. (The encode side *does* take the centred range: it maps
	//    the picture to [0,1] before normalising by the same pixel statistics.)
	std::vector<float> encode(const std::vector<float>& x, i64 T, i64 H, i64 W);

	// Hands the *encode* phase's activation arena back to the device.
	//
	// `encode()` allocates its buffers without reuse (`Runner::alloc` only bumps
	// the arena), so a 416x416 reference image leaves ~2.6 GB of chunks charged
	// to the accountant for the life of the process. The ref2va path is the one
	// caller that runs the encoder *before* the DiT samples, and there the live
	// ledger is what the DiT's residency plan is decided against: without this the
	// plan sees a card that is already half full and refuses the request. No
	// signature changed - this is an addition, and the two exported entry points
	// (`decode`, `encode`) are untouched. Safe because every GpuAlloc into that
	// arena is dead once encode() returns, and the next call allocates fresh ones.
	void release_activation_memory();

	// Everything else this VAE holds: the activation arenas above (the encoder's
	// tile scratch and the decoder's per-block streaming window) *and* the fixed
	// weight block `open()` uploaded into `warena_` - the decoder's projections, the
	// quant convs and the constants: 0.38 GB charged on the reference card, out of
	// the ~0.5 GB of chunks they occupy.
	//
	// This is the one to watch across *calls*. The weights are loaded lazily, by
	// the first `decode()` - i.e. after the first run's DiT has finished - so no
	// plan made before that sees them and every plan after it does: two identical
	// video requests planned 9 and then 8 of their 50 resident blocks (70.8 s
	// against 95.1 s). The owner (the node cache's release path,
// `release_media_cache`) marks the VAE's weights gone, so the next run re-opens it
	// it, so the next run re-runs `open()` and re-reads them out of the page cache.
	void release_gpu_memory();

	u64 last_activation_bytes() const { return act_bytes_; }

private:
	GpuCtx* g_ = nullptr;
	VideoVaeConfig cfg_;
	SafeTensors st_;
	std::string path_;   // kept so the next call can undo release_gpu_memory()
	GpuArena warena_, aarena_;
	u64 act_bytes_ = 0;
	i64 last_tile_ = 0;
	bool open_ = false;
};

}  // namespace phi::media
