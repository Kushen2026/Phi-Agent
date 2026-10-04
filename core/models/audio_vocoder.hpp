// MiniMax H3 audio VAE internals: BigVGAN decoder, DAC-lineage encoder and the
// encoder's attention projection head (W3).
//
// NOT a frozen interface (`core/models/audio_vae.hpp` is). This header
// exists so the pieces can be built and diffed one at a time — the net is a set
// of separable modules, each callable on its own.
//
// The arithmetic is a line-by-line port of comfy/ldm/minimax/audio_vae.py;
// audio_vocoder.cpp carries the correspondence table and the evaluation-order notes.
//
// Everything here is fp32 host code operating on one sample ([C, L] row major,
// no batch): the stereo pair is two calls. That is not a shortcut — the
// reference is fp32 throughout (BigVGAN's SnakeBeta exponents and the Kaiser
// anti-aliasing coefficients do not survive fp16 within the 1e-4 rel-L2
// budget), the H3 checkpoint is fp32, and the whole 10 s decode is ~200 GFLOP,
// i.e. seconds on this box's 16 cores. GPU time goes to the video path.
//
// Weights are *borrowed* from the safetensors mapping (offset: nothing is
// copied, and the file is mmapped), so the SafeTensors object must outlive the
// net built from it.
#pragma once

#include <string>
#include <vector>

#include "models/model_common.hpp"
#include "kernels/alias_free.hpp"
#include "kernels/snake.hpp"

namespace phi::media {

// Receives the named [C, L] activations of a forward pass, in order. The
// conformance test uses it to diff the chain stage by stage against the
// reference dumps, so a wrong tensor is *named* instead of
// showing up as a slightly wrong waveform.
struct StageSink {
	virtual ~StageSink() = default;
	virtual void stage(const std::string& name, const float* data, i64 C, i64 L) = 0;
};

// Reusable temporaries: one audio decode allocates ~10 buffers of up to
// 2*C*L floats, and the levels run at a nearly constant C*L product, so a small
// pool removes essentially all of the allocation traffic.
struct Scratch {
	std::vector<float> buf[6];
	float* get(int slot, i64 n) {
		auto& v = buf[slot];
		if ((i64)v.size() < n) v.resize((size_t)n);
		return v.data();
	}
	void reset() {
		for (auto& v : buf) v.clear();
	}
};

// ── reference modules ──────────────────────────────────────────────────────

// ops.Conv1d / ops.ConvTranspose1d: the checkpoint stores already-folded
// weight-norm weights in torch's [OC, IC, K] (resp. [IC, OC, K]) layout.
struct Conv1dRef {
	const float* w = nullptr;
	const float* b = nullptr;
	i64 ic = 0, oc = 0, k = 1, stride = 1, pad = 0, dilation = 1;

	i64 out_len(i64 L) const { return (L + 2 * pad - dilation * (k - 1) - 1) / stride + 1; }
	// y: [oc, out_len(L)]; must not alias x.
	void forward(const float* x, i64 L, float* y, i64 Lout) const;
};

struct ConvT1dRef {
	const float* w = nullptr;
	const float* b = nullptr;
	i64 ic = 0, oc = 0, k = 1, stride = 1, pad = 0;

	i64 out_len(i64 L) const { return (L - 1) * stride - 2 * pad + k; }
	void forward(const float* x, i64 L, float* y, i64 Lout) const;
};

// AMPBlock1: 3 x [alias-free activation, dilated conv, alias-free activation,
// conv] with a residual add. `dilation` is implicit in convs1[k].dilation.
struct AmpBlock1Ref {
	i64 channels = 0;
	std::vector<Conv1dRef> convs1;
	std::vector<Conv1dRef> convs2;
	std::vector<kernels::Activation1d> acts;   // 2 per (conv1, conv2) pair

	i64 out_len(i64 L) const { return L; }   // kernel/dilation/padding keep L
	void forward(const float* x, i64 L, float* y, Scratch& sc) const;
};

class VocoderNet {
public:
	// `st` must outlive the net. prefix = "decoder." for the H3 audio VAE.
	void open(const SafeTensors& st, const std::string& prefix);

	const std::vector<i64>& rates() const { return rates_; }
	const std::vector<i64>& kernels() const { return kernels_; }
	i64 num_mels() const { return num_mels_; }
	i64 initial_ch() const { return initial_ch_; }
	i64 out_len(i64 L) const;

	// mel: [num_mels, L] (the H3 "mel" is the 2048-wide post-`dec_in_proj`
	// latent) -> pcm [out_len(L)] clamped to [-1, 1], like the reference.
	void forward(const float* mel, i64 L, std::vector<float>& pcm, StageSink* sink = nullptr) const;


private:
	void load_act(const std::string& base, i64 channels, kernels::Activation1d& a) const;

	const SafeTensors* st_ = nullptr;
	std::string prefix_;
	i64 num_mels_ = 0, initial_ch_ = 0;
	std::vector<i64> rates_, kernels_;
	Conv1dRef conv_pre_;
	std::vector<ConvT1dRef> ups_;
	std::vector<AmpBlock1Ref> resblocks_;   // 3 per level, 21 total
	kernels::Activation1d act_post_;
	Conv1dRef conv_post_;
	mutable Scratch sc_;   // reused temporaries (one level's activations at a time)
};

// ── DAC-lineage encoder ────────────────────────────────────────────────────

struct ResidualUnitRef {          // Snake1d, Conv1d(7, dilated), Snake1d, Conv1d(1)
	kernels::Snake1dAct snake1, snake2;
	Conv1dRef conv7, conv1;
	void forward(const float* x, i64 L, float* y, Scratch& sc) const;
};

struct EncoderBlockRef {          // 3 residual units, Snake1d, strided Conv1d
	std::vector<ResidualUnitRef> units;   // dilations 1, 3, 9
	kernels::Snake1dAct snake;
	Conv1dRef down;
	void forward(const float* x, i64 L, std::vector<float>& out, i64& out_L, Scratch* sc) const;
};

class DacEncoderNet {
public:
	void open(const SafeTensors& st, const std::string& prefix);   // "encoder."
	i64 out_len(i64 L) const;
	// x: [1, L] mono waveform in [-1, 1] -> [latent_dim, out_len(L)]
	void forward(const float* x, i64 L, std::vector<float>& out, i64& Tout,
	             StageSink* sink = nullptr) const;


private:
	Conv1dRef conv0_;
	std::vector<EncoderBlockRef> blocks_;   // one per stride
	kernels::Snake1dAct snake_;
	Conv1dRef conv7_;
};

// ── encoder posterior head (AttnProjection + GeGluMlp + CausalAttention) ────

class AttnProjectionNet {
public:
	void open(const SafeTensors& st, const std::string& prefix);   // "pre_block."
	i64 in_dim() const { return in_dim_; }
	i64 out_dim() const { return out_dim_; }
	i64 heads() const { return heads_; }
	// x: [T, in_dim] (the reference feeds the transposed encoder output) ->
	// y: [T, out_dim]
	void forward(const float* x, i64 T, std::vector<float>& y) const;

private:
	const SafeTensors* st_ = nullptr;
	std::string prefix_;
	i64 in_dim_ = 0, out_dim_ = 0, heads_ = 0, hidden_ = 0;
	float ln_eps_ = 1e-5f;
	const float* n1w_ = nullptr;
	const float* n1b_ = nullptr;
	const float* n3w_ = nullptr;
	const float* n3b_ = nullptr;
	const float* proj_w_ = nullptr;
	const float* proj_b_ = nullptr;
	const float* qkv_w_ = nullptr;
	const float* q_bias_ = nullptr;
	const float* v_bias_ = nullptr;
	const float* attn_proj_w_ = nullptr;
	const float* attn_proj_b_ = nullptr;
	const float* n2w_ = nullptr;
	const float* n2b_ = nullptr;
	const float* mnorm_w_ = nullptr;
	const float* mnorm_b_ = nullptr;
	const float* w0_ = nullptr;
	const float* w0b_ = nullptr;
	const float* w1_ = nullptr;
	const float* w1b_ = nullptr;
	const float* w2_ = nullptr;
	const float* w2b_ = nullptr;
};

// ── the whole audio VAE (decoder + encoder + posterior head) ────────────────
//
// audio_vae.cpp binds this to the frozen AudioVae / AudioVocoder. It
// lives here so the chain can be driven without the frozen header (tests), and
// so the stereo pair — which the reference processes as two independent mono
// passes over a [32, 2, T] latent — has one obvious entry point.
class AudioVaeNet {
public:
	// `st` (the 605 MB fp32 audio VAE checkpoint) must outlive this object.
	void open(const SafeTensors& st);

	i64 z_channels() const { return z_ch_; }
	i64 latent_dim() const { return latent_dim_; }
	i64 hop() const { return hop_; }
	i64 sample_rate() const { return sample_rate_; }
	const VocoderNet& decoder() const { return dec_; }
	const DacEncoderNet& encoder() const { return enc_; }
	const AttnProjectionNet& pre_block() const { return pre_; }
	const Conv1dRef& dec_in_proj() const { return dec_in_proj_; }
	const std::vector<float>& latents_mean() const { return lat_mean_; }
	const std::vector<float>& latents_std() const { return lat_std_; }

	// z: [z_channels, T] *normalized* (what the sampler produces, one stereo
	// channel) -> pcm [T * hop] in [-1, 1].
	void decode_channel(const float* z, i64 T, std::vector<float>& pcm,
	                    StageSink* sink = nullptr) const;
	// latent [z_channels, 2, T] -> stereo pcm [2, T * hop] (channel major).
	std::vector<float> decode_stereo(const float* z, i64 T, StageSink* sink = nullptr) const;
	// mono waveform [n] in [-1, 1] (zero-padded to a multiple of hop, exactly as
	// the reference does) -> [z_channels, T] normalized posterior mean.
	std::vector<float> encode_channel(const float* pcm, i64 n, StageSink* sink = nullptr) const;
	// waveform [2, n] -> latent [z_channels, 2, T].
	std::vector<float> encode_stereo(const float* pcm, i64 n) const;

private:
	const SafeTensors* st_ = nullptr;
	VocoderNet dec_;
	DacEncoderNet enc_;
	AttnProjectionNet pre_;
	Conv1dRef dec_in_proj_, mean_proj_, logs_proj_;
	std::vector<float> lat_mean_, lat_std_;
	i64 z_ch_ = 0, latent_dim_ = 0, hop_ = 800, sample_rate_ = 32000;
};

// ── fp32 tensor helpers used by the loader ─────────────────────────────────

// Borrows a tensor from the mapping as [count] fp32. Throws MediaError when the
// tensor is missing, not fp32, or has the wrong element count.
const float* borrow_f32(const SafeTensors& st, const std::string& name, i64 count);

}  // namespace phi::media
