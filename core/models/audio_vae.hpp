// MiniMax H3 audio VAE + BigVGAN vocoder.
//
// A frozen interface: the chain consumes `decode()`, and `encode()` is used for
// reference-audio latents.
//
// Two stages:
//   1. AudioVae   latent [32, T40] @ 40 Hz  <->  mel/f0 features at a multiple
//      of the frame rate.
//   2. AudioVocoder  mel -> 32 kHz PCM. The generator upsamples by
//      (5,5,2,2,2,2,2) = 800, matching 40 Hz * 800 = 32 kHz exactly.
//
// The "alias-free" activation (Snake / SnakeBeta) and the anti-aliased resample
// are the numerically delicate parts; they live in the kernels, not here.
#pragma once

#include <string>
#include <vector>

#include "models/model_common.hpp"

namespace phi::media {

struct AudioVaeConfig {
	i64 z_channels = 32;
	i64 sample_rate = 32000;
	i64 frame_rate = 40;        // latent frames per second
	i64 mel_bins = 128;
	i64 hop = 800;              // samples per latent frame @ 32 kHz
	float norm_eps = 1e-6f;
};

struct AudioVocoderConfig {
	std::vector<i64> upsample_rates{5, 5, 2, 2, 2, 2, 2};   // product 800
	std::vector<i64> upsample_kernels{10, 10, 4, 4, 4, 4, 4};
	i64 num_mels = 128;
	i64 upsample_initial_channel = 1536;
	std::string activation = "snakebeta";
};

class AudioVae {
public:
	void open(const std::string& path, GpuCtx* gpu);
	const AudioVaeConfig& config() const { return cfg_; }

	// z: [z_channels, T40] fp32 -> mel/feature frames, then through BigVGAN.
	// Returns mono PCM in [-1, 1] at `sample_rate`, length ~= T40 * hop.
	std::vector<float> decode(const std::vector<float>& z, i64 T40);

	// pcm mono @ sample_rate -> [z_channels, T40].
	std::vector<float> encode(const std::vector<float>& pcm, i64 n_samples);

private:
	GpuCtx* g_ = nullptr;
	AudioVaeConfig cfg_;
	AudioVocoderConfig vcfg_;
	SafeTensors st_;
	GpuArena warena_, aarena_;
	bool open_ = false;
};

// The vocoder is exposed separately so the audio VAE conformance test can feed
// it a mel shaped by hand without a round trip through the latent.
class AudioVocoder {
public:
	void open(const std::string& path, GpuCtx* gpu);
	const AudioVocoderConfig& config() const { return cfg_; }
	// mel: [num_mels, frames] fp32 -> mono PCM at the audio VAE's sample rate.
	std::vector<float> synthesize(const std::vector<float>& mel, i64 frames);

private:
	GpuCtx* g_ = nullptr;
	AudioVocoderConfig cfg_;
	SafeTensors st_;
	GpuArena warena_, aarena_;
	bool open_ = false;
};

}  // namespace phi::media
