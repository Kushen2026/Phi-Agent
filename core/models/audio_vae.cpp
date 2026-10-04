// MiniMax H3 audio VAE + BigVGAN — implementation (W3).
//
// The frozen header declares the two entry points the rest of the engine uses;
// the arithmetic lives in models/audio_vocoder.cpp (decoder, DAC encoder, posterior
// head) and kernels/{snake,alias_free}.cpp. This file is the binding: it loads
// the checkpoint, fills in the configuration the frozen structs carry, and
// validates shapes at the boundary.
//
// ── why no GPU ─────────────────────────────────────────────────────────────
// `GpuCtx*` is accepted and stored, never dereferenced: this path is host fp32.
// Three reasons, in order of weight:
//   1. the reference is fp32 (the H3 audio VAE checkpoint is fp32 throughout,
//      working_dtypes = [float32] in comfy/sd.py) and the acceptance budget is
//      rel L2 < 1e-4 on the PCM. fp16 activation/weight stages do not survive
//      the 21-resblock SnakeBeta chain, and neither do the anti-aliasing
//      filters (12-tap Kaiser sinc, coefficients ~2e-3..4.4e-1);
//   2. there is no conv1d/conv_transpose1d dispatcher in gpu_ops.hpp; a GPU
//      version means two new kernels plus a resample kernel, for a stage that
//      is ~200 GFLOP per 10 s of audio;
//   3. measure it: this decode is seconds on the box's 16 cores, while the
//      video VAE's ViT3D is where the GPU time actually goes. The GPU token
//      stays free for W2/W7/W8.
// If the profile ever says otherwise, the operators here are already separable
// (one call per module) and can be ported one at a time behind the same API.
#include "models/audio_vae.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "models/audio_vocoder.hpp"

namespace phi::media {

namespace {

struct Entry {
	AudioVaeNet vae;                     // AudioVae
	std::shared_ptr<VocoderNet> voc;     // AudioVocoder
	const SafeTensors* st = nullptr;     // which mapping the weights point into
};

std::mutex& reg_mu() {
	static std::mutex m;
	return m;
}
std::map<const void*, Entry>& registry() {
	static std::map<const void*, Entry> r;
	return r;
}

// The frozen headers have no destructor hook, so the per-object state cannot be
// a member; a registry keyed by `this` is the only option. Every use re-checks
// that the entry was built from *this* object's safetensors mapping, which makes
// a stale entry (address reuse after destruction) fail loudly instead of reading
// a closed mapping.
Entry& entry_for(const void* key, const SafeTensors* st, const char* what) {
	std::lock_guard<std::mutex> lk(reg_mu());
	auto it = registry().find(key);
	if (it == registry().end() || it->second.st != st) {
		throw MediaError(std::string(what) + ": open() must be called first");
	}
	return it->second;
}

void store_entry(const void* key, Entry&& e) {
	std::lock_guard<std::mutex> lk(reg_mu());
	registry()[key] = std::move(e);
}

}  // namespace

// ── AudioVae ─────────────────────────────────────────────────────────────

void AudioVae::open(const std::string& path, GpuCtx* gpu) {
	g_ = gpu;   // accepted for interface symmetry; this path never dereferences it
	st_.open(path);

	Entry e;
	e.st = &st_;
	e.vae.open(st_);

	cfg_.z_channels = e.vae.z_channels();
	cfg_.sample_rate = e.vae.sample_rate();
	cfg_.hop = e.vae.hop();
	cfg_.frame_rate = cfg_.hop > 0 ? cfg_.sample_rate / cfg_.hop : 0;
	// The BigVGAN decoder consumes latent_dim-wide features (2048), not a
	// 128-bin mel: `mel_bins` is reported as the width the vocoder actually
	// takes, which is what a consumer of config() needs to size its buffer.
	cfg_.mel_bins = e.vae.latent_dim();

	vcfg_.upsample_rates = e.vae.decoder().rates();
	vcfg_.upsample_kernels = e.vae.decoder().kernels();
	vcfg_.num_mels = e.vae.decoder().num_mels();
	vcfg_.upsample_initial_channel = e.vae.decoder().initial_ch();
	vcfg_.activation = "snakebeta";

	store_entry(this, std::move(e));
	open_ = true;
}

std::vector<float> AudioVae::decode(const std::vector<float>& z, i64 T40) {
	Entry& e = entry_for(this, &st_, "AudioVae::decode");
	const i64 c = e.vae.z_channels();
	if (T40 <= 0) T40 = c > 0 ? (i64)z.size() / c : 0;
	if (T40 <= 0 || (i64)z.size() != c * T40) {
		throw MediaError("AudioVae::decode: latent is " + std::to_string(z.size()) +
		                 " floats, expected " + std::to_string(c) + " x T40=" + std::to_string(T40));
	}
	// One stereo channel per call: the reference reshapes [B, 32, 2, T] to
	// [B*2, 32, T] and runs the pair independently (see decode_stereo()).
	std::vector<float> pcm;
	e.vae.decode_channel(z.data(), T40, pcm, nullptr);
	return pcm;
}

std::vector<float> AudioVae::encode(const std::vector<float>& pcm, i64 n_samples) {
	Entry& e = entry_for(this, &st_, "AudioVae::encode");
	if (n_samples < 0 || (i64)pcm.size() < n_samples) n_samples = (i64)pcm.size();
	return e.vae.encode_channel(pcm.data(), n_samples, nullptr);
}

// ── AudioVocoder ─────────────────────────────────────────────────────────

void AudioVocoder::open(const std::string& path, GpuCtx* gpu) {
	g_ = gpu;
	st_.open(path);

	Entry e;
	e.st = &st_;
	e.voc = std::make_shared<VocoderNet>();
	e.voc->open(st_, "decoder.");

	cfg_.upsample_rates = e.voc->rates();
	cfg_.upsample_kernels = e.voc->kernels();
	cfg_.num_mels = e.voc->num_mels();
	cfg_.upsample_initial_channel = e.voc->initial_ch();
	cfg_.activation = "snakebeta";

	store_entry(this, std::move(e));
	open_ = true;
}

std::vector<float> AudioVocoder::synthesize(const std::vector<float>& mel, i64 frames) {
	Entry& e = entry_for(this, &st_, "AudioVocoder::synthesize");
	if (!e.voc) throw MediaError("AudioVocoder::synthesize: not a vocoder instance");
	const i64 c = e.voc->num_mels();
	if (frames <= 0) frames = c > 0 ? (i64)mel.size() / c : 0;
	if (frames <= 0 || (i64)mel.size() != c * frames) {
		throw MediaError("AudioVocoder::synthesize: mel is " + std::to_string(mel.size()) +
		                 " floats, expected " + std::to_string(c) + " x frames=" +
		                 std::to_string(frames));
	}
	std::vector<float> pcm;
	e.voc->forward(mel.data(), frames, pcm, nullptr);
	return pcm;
}

}  // namespace phi::media
