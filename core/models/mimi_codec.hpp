// Breeze-TTS-2's audio codec: the Qwen3-TTS 12 Hz tokenizer (a Mimi-family
// RVQ + a conv decoder, dumped from qwen-tts 0.1.1's
// Qwen3TTSTokenizerV2Model). Codes -> PCM only: the encoder half (voice
// cloning) is not on this path.
//
// ── where the weights come from ────────────────────────────────────────────
// `audio_tokenizer/model.safetensors` (682 MB, fp32 throughout). The Breeze
// checkpoint carries the same `codec_model.*` tree, but the reference loads the
// codec from this separate file (loader.py: `_build_codec`), so that is what
// `open()` reads. Tensors are *borrowed* from the mmap (`borrow_f32`), so the
// 682 MB costs no host copy and the SafeTensors object must outlive this one.
//
// ── why host fp32 ──────────────────────────────────────────────────────────
// Same reason as core/models/audio_vocoder.*: the checkpoint is fp32, the net is
// a handful of hundred-MFLOP conv stages, and the whole decode of a 20 s
// utterance is a couple of seconds on this box's cores. The GPU is where the
// per-frame transformer loop runs; this is not.
//
// ── the module tree (names are the checkpoint's) ───────────────────────────
//   decoder.quantizer.rvq_first  1 codebook  -> [256,T] -> output_proj [512,T]
//   decoder.quantizer.rvq_rest  15 codebooks -> [256,T] -> output_proj [512,T]
//   decoder.pre_conv            causal conv 512 -> 1024, k=3
//   decoder.pre_transformer     8 layers, 512-wide, 16 heads x 64, layer_scale,
//                               causal sliding-window (72) attention, + in/out proj
//   decoder.upsample[0..1]      ConvTranspose1d(1024,1024,2,2) + ConvNeXtBlock(1024)
//   decoder.decoder[0]          causal conv 1024 -> 1536, k=7
//   decoder.decoder[1..4]       4 x DecoderBlock: SnakeBeta, ConvT(2*rate, stride=rate),
//                               3 x ResidualUnit(dilation 1,3,9)
//   decoder.decoder[5]          SnakeBeta(96)
//   decoder.decoder[6]          causal conv 96 -> 1, k=7
//
// The forward is a line-by-line port of Qwen3TTSTokenizerV2Decoder.forward
// (vendor/codec_model.py): quantizer -> pre_conv -> pre_transformer -> upsample
// -> decoder, with `clamp(-1, 1)` at the end, over `chunked_decode`'s 300-frame
// chunks with 25 frames of left context (the reference's own decode entry
// point). Every causal conv pads on the *left only* and the transposed convs
// trim `k - stride` on the right, exactly as Qwen3TTSTokenizerV2CausalConvNet /
// Qwen3TTSTokenizerV2CausalTransConvNet do.
#pragma once

#include <string>
#include <vector>

#include "models/audio_vocoder.hpp"   // Conv1dRef, ConvT1dRef, borrow_f32
#include "models/model_common.hpp"

namespace phi::media {

// y = x + sin(exp(alpha_log)*x)^2 / (exp(beta_log) + 1e-9), per channel: the
// checkpoint's SnakeBeta (which stores the *log* of alpha and beta).
struct MimiSnake {
	const float* alpha = nullptr;
	const float* beta = nullptr;
	i64 channels = 0;
	void apply(const float* x, float* y, i64 L) const;
};

// Qwen3TTSTokenizerV2CausalConvNet with stride 1: `kernel_size = (k-1)*dilation
// + 1`, `padding = kernel_size - stride` zeros on the left, and the reference's
// `_get_extra_padding_for_conv1d` on the right - which for stride 1 is exactly
// zero. c.pad is preset to the left-only padding at load time, so `forward`
// reads the first `L` columns of a `L + pad`-wide window.
struct MimiCausalConv {
	Conv1dRef c;
	void forward(const float* x, i64 L, float* y) const { c.forward(x, L, y, L); }
};

// Qwen3TTSTokenizerV2CausalTransConvNet: ConvTranspose1d(k, stride) followed by
// the trim of the last `k - stride` output columns. `raw_len` is what the conv
// itself produces (the caller compacts it down to out_len).
struct MimiCausalTConv {
	ConvT1dRef c;
	i64 right_pad = 0;
	i64 out_len(i64 L) const { return L * c.stride; }
	i64 raw_len(i64 L) const { return (L - 1) * c.stride + c.k; }
	// y must be [oc, raw_len(L)]; after this call the first out_len(L) columns of
	// each row hold the result.
	void forward(const float* x, i64 L, float* y) const;
};

// Qwen3TTSTokenizerV2DecoderDecoderResidualUnit: SnakeConvSnakeConv + residual.
struct MimiResidualUnit {
	MimiSnake act1, act2;
	MimiCausalConv conv1;   // k=7, dilation d
	MimiCausalConv conv2;   // k=1
	// x, y: [C, L]; y must not alias x; tmp is [C, L] scratch.
	void forward(const float* x, i64 L, float* y, float* tmp) const;
};

// Qwen3TTSTokenizerV2DecoderDecoderBlock: [SnakeBeta, ConvT, 3 x ResidualUnit].
struct MimiDecoderBlock {
	MimiSnake act;
	MimiCausalTConv up;
	std::vector<MimiResidualUnit> units;   // dilations 1, 3, 9
	i64 in_ch = 0, out_ch = 0, rate = 1;
	// y must be [out_ch, raw_len(L)] (see MimiCausalTConv); `a` and `b` are [., .]
	// scratch of at least max(in_ch*L, out_ch*L*rate) floats. Returns the buffer
	// holding the result (y or b, whichever the 3-unit ping-pong ended in).
	const float* forward(const float* x, i64 L, float* y, float* a, float* b) const;
};

// Qwen3TTSTokenizerV2ConvNeXtBlock: depthwise causal conv 7, LayerNorm over the
// channels, dim -> 4*dim -> dim with an exact (erf) GELU, then gamma.
struct MimiConvNeXt {
	i64 dim = 0;
	Conv1dRef dw;   // groups = dim: one 1x7 tap set per channel
	const float* dw_bias = nullptr;
	const float* gamma = nullptr;
	const float* norm_w = nullptr;
	const float* norm_b = nullptr;
	const float* pw1_w = nullptr;   // [4*dim, dim]
	const float* pw1_b = nullptr;   // [4*dim]
	const float* pw2_w = nullptr;   // [dim, 4*dim]
	const float* pw2_b = nullptr;   // [dim]
	// x -> y, both [dim, L]. `dw_out` is [dim, L], `tc` is [L, 4*dim] and `hc` is
	// [L, dim] scratch (the block norms over the channel axis, so it transposes
	// between the depthwise conv and the pointwise pair, like the reference does).
	void forward(const float* x, i64 L, float* y, float* dw_out, float* tc, float* hc) const;
};

// One Qwen3TTSTokenizerV2DecoderTransformerLayer (512-wide, 16 heads x 64).
struct MimiPreLayer {
	const float* in_ln = nullptr;    // [512]
	const float* post_ln = nullptr;  // [512]
	const float* q_w = nullptr;      // [1024, 512]
	const float* k_w = nullptr;      // [1024, 512]
	const float* v_w = nullptr;      // [1024, 512]
	const float* o_w = nullptr;      // [512, 1024]
	const float* gate_w = nullptr;   // [1024, 512]
	const float* up_w = nullptr;     // [1024, 512]
	const float* down_w = nullptr;   // [512, 1024]
	const float* attn_scale = nullptr;   // [512] layer scale
	const float* mlp_scale = nullptr;    // [512] layer scale
};

// The 12 Hz codec: RVQ codebook indices -> PCM at 24 kHz.
class MimiCodec {
public:
	// `config_json` overrides the sibling `config.json` the codec reads for the
	// scalars a shape cannot carry; empty keeps the sibling lookup.
	void open(const std::string& path, GpuCtx* gpu, const std::string& config_json = std::string());
	i64 sample_rate() const { return sample_rate_; }
	// codes: [frames, 16] (frame-major, 16 codebooks) int32 -> mono PCM in [-1,1].
	std::vector<float> decode(const std::vector<i32>& codes, i64 frames);
	// frames of the codec's own rate (12.5 Hz).
	static double frames_per_second() { return 12.5; }

	i64 codebooks() const { return codebooks_; }
	i64 upsample_rate() const { return total_upsample_; }
	bool loaded() const { return loaded_; }

private:
	// One chunk of the reference's `chunked_decode`: frames x 1920 samples.
	void decode_chunk(const i32* codes, i64 frames, std::vector<float>& out) const;
	void quantize_frames(const i32* codes, i64 frames, float* out /*[512, frames]*/) const;
	void pre_transformer(float* x /*[frames, 1024]*/, i64 frames) const;

	SafeTensors st_;
	std::string prefix_ = "decoder.";
	bool loaded_ = false;
	i64 sample_rate_ = 24000;
	i64 codebooks_ = 16;
	i64 codebook_size_ = 2048;
	i64 codebook_dim_ = 512;    // the quantizer's dimension (2 * 256)
	i64 emb_dim_ = 256;         // one codebook entry's width
	i64 latent_ = 1024;
	i64 hidden_ = 512;
	i64 heads_ = 16, head_dim_ = 64, kv_heads_ = 16;
	i64 num_layers_ = 8;
	i64 sliding_window_ = 72;
	i64 total_upsample_ = 1920;
	std::vector<i64> up_rates_;   // [8,5,4,3]

	// Quantizer: the two halves' codebooks and their 1x1 output projections.
	// The codebooks are derived at load time (embedding_sum / cluster_usage).
	std::vector<float> emb_first_;   // [codebook_size, 256]
	std::vector<float> emb_rest_;    // [(codebooks-1) * codebook_size, 256]
	const float* outp_first_ = nullptr;   // [512, 256]
	const float* outp_rest_ = nullptr;    // [512, 256]

	MimiCausalConv pre_conv_;
	const float* in_proj_w_ = nullptr;    // [512, 1024]
	const float* in_proj_b_ = nullptr;    // [512]
	const float* out_proj_w_ = nullptr;   // [1024, 512]
	const float* out_proj_b_ = nullptr;   // [1024]
	const float* final_norm_ = nullptr;   // [512]
	std::vector<MimiPreLayer> layers_;

	ConvT1dRef up_t_[2];
	MimiConvNeXt convnext_[2];

	MimiCausalConv head_conv_;   // decoder.0: latent -> decoder_dim
	std::vector<MimiDecoderBlock> blocks_;
	MimiSnake final_snake_;      // decoder.5
	MimiCausalConv final_conv_;  // decoder.6: output_dim -> 1
	i64 decoder_dim_ = 1536;
	i64 inter_ = 0;             // the bottleneck mlp's inner width
	i64 n_up_ = 0;              // upsample stages (2 in every shipped config)
	float rms_eps_ = 1e-5f;
	float rope_theta_ = 10000.0f;
};

}  // namespace phi::media
