// ACE-Step 1.5 audio VAE (the DAC-lineage `AudioOobleckVAE` decoder).
//
// NOT a frozen interface yet; `core/models/ace_vae.cpp` carries the mapping to
// the tensor names in `ace_1.5_vae.safetensors`.
//
// The net is a stack of weight-normed convolutions, SnakeBeta activations and
// residual units. `open()` folds the weight-norm weights into host fp32 through
// the helpers `core/models/audio_vocoder.hpp` already provides (Conv1dRef,
// ConvT1dRef, `kernels::SnakeBetaAct`) and, when a working `GpuCtx` was handed
// to it, runs the decode with the engine's device op set (`audio_vocoder`'s
// kernels: `conv1d_gather`, `convt1d`, `snake_beta` - see `core/kernels/ops.cpp`).
// The device copy of the folded weights is uploaded per decode rather than kept:
// this loader runs *before* the DiT sampler, so anything parked on the card at
// `open()` time is subtracted from the DiT's residency plan, and 330 MB of fp32
// decoder weights held through a 50-step sampling loop to be used once at the end
// costs the DiT a whole resident block. See `upload_device_weights`.
//
// The host implementation is kept as the reference the device kernels are
// checked against *and* as a fallback:
//
//   PHI_ACE_VAE_CPU=1    decode on the host even when a device is available
//                        (unset, empty or 0 -> the GPU path, whenever `open()`
//                        was given a `GpuCtx` whose `ok()` and `keep` are set)
//
// Both paths are fp32 end to end: the checkpoint is fp32-quality and the
// SnakeBeta exponents plus the dilated residual stack do not survive fp16 inside
// a 1e-4 rel-L2 budget. They are not bit-identical to each other (the host and
// the device evaluate `sinf`/`expf` and the multiply-add chains with different
// rounding), but they agree to ~1e-7 and they produce the *same* tensor by
// construction: the same module tree, the same accumulation order and the same
// chunking (see the chunk/halo note in ace_vae.cpp).
//
// ── precision of the checkpoints this reads ────────────────────────────────
//
// Every weight here is read through `tensor_to_f32`, which asks the safetensors
// reader for the tensor's *values*: a raw f32 / f16 / bf16 tensor is converted,
// and a quantised one (int8 tensorwise, asym_w4a8_int8, w6a8_int8, nvfp4, fp8, a
// grouped int8) is decoded by its own family decoder first. So an fp16, bf16, fp8
// or int8 re-export of this VAE loads and runs - widening a small format into the
// fp32 the net computes in is exact, and a quantised checkpoint decodes to the
// values it was quantised from. What this file does *not* do is keep a narrow
// format's bytes on the device, because there is no device operand path for it:
// the compute precision is the reference's, and it is fp32.
#pragma once

#include <string>
#include <vector>

#include "kernels/snake.hpp"
#include "models/audio_vocoder.hpp"
#include "models/model_common.hpp"

namespace phi::media {

class AceVae {
public:
	AceVae();
	~AceVae();

	void open(const std::string& path, GpuCtx* gpu);
	i64 latent_channels() const { return latent_; }   // 64
	i64 sample_rate() const { return sr_; }           // 48000
	i64 output_channels() const { return out_ch_; }   // 2 - see the note on `decode`
	// Decoded samples per latent frame (strides 10*6*4*4*2 = 1920).
	i64 hop() const { return hop_; }

	// z: [64, T] *normalised* latent (the sampler's output) -> pcm in [-1, 1].
	//
	// This checkpoint's decoder is **2-channel out** (`decoder.layers.7` is
	// [2, 128, 7]): one [64, T] latent decodes to a stereo clip, exactly like the
	// stable-audio-tools `OobleckVAE` it is taken from. The frozen API here
	// promises *mono* PCM, so `decode` returns the mean of those two channels and
	// `decode_pair` returns the pair interleaved. A caller that wants what the
	// model actually produces wants `decode_pair`.
	std::vector<float> decode(const std::vector<float>& z, i64 T);
	// z: [64, 2, T] channel-major -> interleaved stereo PCM: each of the two
	// latents is decoded and the two mono results are interleaved. (The native
	// stereo form of this checkpoint is `decode_pair`; this one exists because
	// the frozen API asks for it.)
	std::vector<float> decode_stereo(const std::vector<float>& z, i64 T);
	// z: [64, T] -> interleaved stereo [2*L], the decoder's own two channels.
	std::vector<float> decode_pair(const std::vector<float>& z, i64 T);

private:
	// One decode of `T` latent frames: [out_ch, T * hop] into `out`.
	void decode_raw(const float* z, i64 T, std::vector<float>& out) const;
	// The same module tree on the device (one dispatch per module, the chunk's
	// temporaries out of `g_->aa`, the last stage downloaded once).
	void decode_raw_gpu(const float* z, i64 T, std::vector<float>& out) const;
	// True when `decode_pair` should take the device path.
	bool dev_decode() const;
	// Puts the folded weights (and the exp()'d SnakeBeta parameters) on the card,
	// into the streaming weight arena. Called once at the top of every device
	// decode; see `open` for why they are not uploaded once and kept.
	void upload_device_weights();

	struct ResidualUnit;
	struct Block;

	SafeTensors st_owned_;   // the opened checkpoint (the refs borrow from it)
	const SafeTensors* st_ = nullptr;
	std::string path_;
	i64 latent_ = 0;   // 64
	i64 out_ch_ = 0;   // 2
	i64 sr_ = 48000;
	i64 hop_ = 1;      // product of the upsampling strides

	// Host storage for the folded weight-norm weights. `Conv1dRef` borrows a
	// `const float*`, so every vector here has to outlive the net; the outer
	// vector may grow (its elements move, their heap buffers do not).
	std::vector<std::vector<float>> store_;
	std::vector<float> conv_in_w_, conv_in_b_;
	std::vector<float> conv_out_w_;
	std::vector<float> in_alpha_, in_beta_;     // SnakeBeta log parameters
	std::vector<float> out_alpha_, out_beta_;
	Conv1dRef conv_in_;
	Conv1dRef conv_out_;
	kernels::SnakeBetaAct snake_out_;
	std::vector<Block> blocks_;

	// Reused temporaries, sized per call (see the chunking note in decode_raw).
	mutable std::vector<float> a_, b_, c_, d_;
	i64 chunk_frames_ = 128;
	i64 halo_frames_ = 16;

	// The device path. `g_` is borrowed from the loader (the graph's GpuCtx
	// outlives every VAE); `dev_ready_` is set by `open()` when a working `GpuCtx`
	// was handed to it, and `upload_device_weights()` (at decode time) fills the
	// handles below. Those live in the **weight** arena, which is why they are
	// re-uploaded per call rather than kept: see `open` for the accounting reason.
	GpuCtx* g_ = nullptr;
	bool dev_ready_ = false;
	GpuAlloc d_conv_in_w_, d_conv_in_b_;
	GpuAlloc d_conv_out_w_, d_conv_out_b_;   // the head is bias-free: d_conv_out_b_ is empty
	GpuAlloc d_out_a_, d_out_b_;   // exp()'d SnakeBeta alpha/beta of the output head
};

}  // namespace phi::media
