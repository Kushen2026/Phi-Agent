// Breeze-TTS-2 text-to-speech (the VoiceDesign path).
//
// Three networks, in the reference's own order (native.py / runtime.py):
//
//   text encoder   T5Gemma2 encoder, 26 layers x 1152 wide, bidirectional with a
//                  512-token sliding window on all but every sixth layer, Q/K
//                  RMSNorm over head_dim 256, GeGLU(gelu_pytorch_tanh), and the
//                  checkpoint's `embed_tokens.eoi_embedding` substituted at the
//                  EOI position. Host fp32.
//   backbone       Qwen3-1B (28 layers x 2048, 16 heads of 128, GQA 8 K/V heads,
//                  theta 1e6) consuming the projected text rows.
//   depth decoder  a 12-layer llama (1024 wide, 8 heads of 128, 2 K/V heads,
//                  mlp 8192) that turns one backbone token into the other 15
//                  codebooks of the same frame, plus the per-codebook
//                  `codebooks_head` [15, 1024, 2051].
//
// then the codec (core/models/mimi_codec.*) turns the frames into PCM.
//
// ── what runs where ────────────────────────────────────────────────────────
// The text encoder is a single prefill pass over a short prompt, so it runs on
// the host in fp32, reading the checkpoint a slab of rows at a time through
// `SafeTensors::dequant_rows` (the whole encoder is 2.8 GB as fp32 and never
// needs to be resident). Its sliding-window band mask is not expressible with any
// kernel the engine has - `attn_flash`'s per-row bound is an *upper* bound only,
// and the window needs both - and the host pass also keeps its (1 + w) norms and
// its two RoPE tables exact.
//
// The backbone and the depth decoder run on the GPU through whichever GEMM the
// checkpoint's own precision implies (`load_quant_linear` / `linear_gemm`: int8
// tensorwise and the packed families keep their int8 path, a bf16/f16/f32 checkpoint
// runs the dense GEMM - the file's precision is never converted) and the fp16 tiled
// flash attention. Their weights are loaded *once* into the `keep` arena (~1.8 GB
// for both at bf16) rather than streamed per layer, and that is the one place this
// chain differs from every other one in the engine: a TTS utterance is up to 2048
// sequential single-token steps, each of which walks *every* layer, so re-uploading
// a layer per step would cost far more than the arithmetic it feeds (500 MB of PCIe
// traffic per step for the backbone alone, i.e. tens of gigabytes per sentence).
//
// The price is that the weights must be *resident*, which on the 6 GB reference
// card is most of the accountant's ceiling. There is no hard-coded size here: the
// resident set is sized by the checkpoint and charged against the *live* budget
// (`vram_budget().limit()`, which itself scales with the card), so a bigger card is
// used fully and a small one refuses up front with the accountant's own message
// rather than being demoted to system memory mid-utterance.
//
// ── the sampling loop ─────────────────────────────────────────────────────
// runtime.py's `generate_codes`, faithfully: prefill the merged prompt, sample
// the first backbone token, then per frame run the depth decoder for the 15
// remaining codebooks, feed the frame's summed audio embedding back into the
// backbone, and sample the next token under the repetition penalty. CFG is the
// reference's `uncond + scale * (cond - uncond)` over the two prompt branches,
// and the reserved ids [2048, 2051) are suppressed on both sampling sites. The
// two branches are kept as two *unpadded* sequences: the reference left-pads the
// shorter (negative) branch to the cond branch's length, masks the pads out of
// every attention, and gives the continuation the RoPE positions of its own real
// tokens - which is exactly what a separate compact sequence computes.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "models/mimi_codec.hpp"
#include "models/model_common.hpp"
#include "text/breeze_tokenizer.hpp"

namespace phi::media {

// Everything one TTS run is configured with.
//
// ── the two strings ────────────────────────────────────────────────────────
// `instruction` describes the *voice*, `text` is what is *spoken* - and the two
// are not interchangeable. runtime.py builds exactly one prompt out of them,
// `design_segments(text, instruction)` ->
//
//     "[S0]<ins_bos>" + instruction + "<ins_eos>" + text
//
// so the timbre description is what sits inside the checkpoint's instruction
// tokens (config.json: `"<ins_bos>": "<instruction>"`) and the words to speak are
// what follows them. `build_prompt` below is the only place that lays them out,
// and it must stay in that order: swapping the two is not a parameter
// convention, it is a different request (the model reads the description aloud).
//
// A build that *does* read the description aloud is therefore almost never this
// order being wrong - it is the prompt arriving at the backbone unusable, and the
// one thing that does that here is the text encoder's attention scale: it must be
// the encoder config's `query_pre_attn_scalar^-0.5`, not `head_dim^-0.5` and not
// `hidden_size^-0.5` (see `BreezeTts::open`). Scaling it wrongly flattens every
// attention distribution in the encoder, which is heard as the model having no
// idea which span of the prompt is the voice and which is the sentence.
//
// ── the sampling parameters ────────────────────────────────────────────────
// The defaults are the released Voice-Design node's own `_generation_controls`
// (nodes.py) - the values its example workflow ships - so a caller that passes
// nothing gets the reference's behaviour: cfg 4 for voice design, backbone
// 0.9/50/1.0 with a 1.1 repetition penalty, depth decoder 0.9/50/1.0. Sampling
// either loop looser than this (top-k off, depth temperature 1.0) makes the
// frames drift: the utterance ends early and the tail decodes as noise.
struct BreezeParams {
	std::string text;             // the words to speak (the part that is read aloud)
	std::string instruction;      // the voice/timbre description (never spoken)
	float cfg_scale = 4.0f;       // voice-design default (the reference node's 4)
	// The cap on generated *acoustic code frames*, which is the sampler's unit
	// (12.5 frames = 1 s, see `MimiCodec::frames_per_second`). 0 = let the *text*
	// decide: `prepare` sizes the budget from `estimate_frames(text)`, so a long
	// sentence is not cut at some fixed duration. Either way the model stops at
	// its own EOS token, usually well before the budget.
	i64 max_new_tokens = 0;
	float temperature = 0.9f;      // backbone
	i64 top_k = 50;
	float top_p = 1.0f;
	float repetition_penalty = 1.1f;
	float depth_temperature = 0.9f;   // depth decoder (the 15 remaining codebooks)
	i64 depth_top_k = 50;
	float depth_top_p = 1.0f;
	u64 seed = 0;
};

class BreezeTts {
public:
	// `model_path` = the bf16 safetensors; `codec_path` = audio_tokenizer/model.safetensors;
	// `tokenizer_json` = the tokenizer.json file. `config_json` / `codec_config_json`
	// override the config.json each loader would otherwise read beside its weights;
	// empty keeps that sibling lookup, which is what every shipped layout uses.
	void open(const std::string& model_path, const std::string& codec_path,
	          const std::string& tokenizer_json, const std::string& config_json,
	          const std::string& codec_config_json, GpuCtx* gpu);
	i64 sample_rate() const;   // 24000

	// Voice-design path only (no reference audio / no voice cloning):
	// runs the text encoder, the backbone + depth-decoder sampling loop, and the
	// codec decoder. `progress(current, total)` reports generation frames.
	// Returns mono PCM in [-1,1] at sample_rate.
	std::vector<float> speak(const BreezeParams& p,
	                         const std::function<void(i64, i64)>& progress,
	                         const std::function<bool()>& cancelled);

	// ── the three stages, exposed so a node graph can run them separately ──
	//
	// `speak` is exactly prepare -> generate_codes -> decode_codes, and the split
	// is what lets the TTS chain be the same three-node shape the image, video and
	// music chains are: a conditioning node, a sampler node and a decode node, with
	// the prompt travelling on a socket instead of inside one call.
	//
	// The prepared prompt is a handle, not a buffer: the *branches* (their token
	// ids, the text encoder's embeddings and the KV caches) live in this object,
	// because they are the model's own state and a generation is one-at-a-time
	// under the media engine's lock. So the object holds one live prompt; the
	// shared_ptr exists only to give the node graph a value to carry (and to keep
	// the frame budget the sampler needs).
	struct Prompt {
		i64 prefill = 0;      // the longest branch's prompt length, in tokens
		i64 branches = 0;     // 1 = no CFG, 2 = conditional + negative
		i64 max_frames = 0;   // frames this prompt leaves room for
	};
	std::shared_ptr<Prompt> prepare(const BreezeParams& p);

	// The backbone + depth-decoder token loop. `frames` receives the number of
	// acoustic frames actually generated. Returns `frames * num_codebooks` code
	// ids, frame-major. Honours `cancelled` (stops early, keeping what it has)
	// and reports `progress(current, total)`.
	std::vector<i32> generate_codes(const std::shared_ptr<Prompt>& prompt, const BreezeParams& p,
	                                i64* frames,
	                                const std::function<void(i64, i64)>& progress,
	                                const std::function<bool()>& cancelled);

	// The codec: `frames * num_codebooks` frame-major ids -> PCM.
	std::vector<float> decode_codes(const std::vector<i32>& codes, i64 frames);

	// The sampler's randomness: a splitmix64 stream seeded from `BreezeParams::seed`
	// (0 = a fresh media seed). Public because the sampling helpers in the
	// implementation are free functions.
	struct Rng {
		u64 s = 0;
		float next01();
	};

	bool loaded() const { return loaded_; }
	// Frames a prompt of `prefill_len` tokens can still generate: runtime.py's
	// `min(max_new_tokens, MAX_SEQ_LEN - 1 - prefill_len)`.
	i64 max_frames_for(i64 prefill_len, i64 max_new_tokens) const;

private:
	struct BackboneLayer {
		QuantLinear q, k, v, o, gate, up, down;
		GpuAlloc in_ln, post_ln, q_norm, k_norm;
	};
	struct DepthLayer {
		QuantLinear q, k, v, o, gate, up, down;
		GpuAlloc in_ln, post_ln;
	};

	// One prompt branch: its token ids, its own (unpadded) KV caches - the
	// backbone's in the `keep` arena for the whole utterance, the depth decoder's
	// for the 17 positions of one frame - and how far it has run.
	struct Branch {
		std::vector<i32> ids;
		i64 len = 0;       // tokens in the backbone cache (prompt + frames)
		i64 prompt = 0;    // the prompt's length
		i64 base = 0;      // RoPE position of the first generated token
		std::vector<GpuAlloc> k, v;     // [n_layers]
		std::vector<GpuAlloc> dk, dv;   // [n_layers]
	};

	// A [n, k] encodeder matrix streamed out of the checkpoint, one slab of
	// output rows at a time.
	struct TeMat {
		const SafeTensors* st = nullptr;
		const StTensor* t = nullptr;
		i64 n = 0, k = 0;
		bool ok() const { return st && t && n > 0 && k > 0; }
		// y[m, n] = x[m, k] @ W^T
		void gemm(const float* x, i64 m, float* y) const;
	};
	struct TeLayer {
		TeMat q, k, v, o, gate, up, down;
		const float* pre_attn = nullptr;
		const float* post_attn = nullptr;
		const float* pre_ff = nullptr;
		const float* post_ff = nullptr;
	};

	void check(const char* what, bool ok) const;
	// [S, backbone hidden] fp32: the projected text encoder output of one branch.
	std::vector<float> text_encoder(const std::vector<i32>& ids) const;
	// runtime.py's design_segments / _prepare_one for both branches. This is the
	// ONE place the request's two strings meet, and the reference's order is fixed:
	// `design_segments(text, instruction)` renders
	// "[S0]<ins_bos>{instruction}<ins_eos>{text}" - the timbre description inside
	// the instruction tokens, the words to speak after them.
	void build_prompt(const BreezeParams& p);
	// runtime.py's `estimate_speech_frames`: how many acoustic frames the text
	// usually takes (en ~4.1 / zh ~3.5 per token, ~5 per inline event), calibrated
	// on the official checkpoint. `prepare` sizes an unspecified frame budget from
	// it, so "how long should this be" follows the *text* rather than a fixed
	// duration.
	i64 estimate_frames(const std::string& text) const;

	// The backbone's 28 layers over `rows` rows of one branch. `row0` is the KV
	// row the first new key/value lands on, `pos0` the first row's RoPE position
	// and `keys` how many keys the attention sees. Returns the post-norm hidden
	// states ([rows, hidden], device).
	GpuAlloc backbone_run(Branch& b, i64 row0, i64 rows, i64 pos0, i64 keys, bool causal,
	                      const float* embeds);
	// The depth decoder's 12 layers, same conventions.
	GpuAlloc depth_run(Branch& b, i64 row0, i64 rows, i64 pos0, i64 keys, bool causal,
	                   const float* embeds);
	// `codebooks_head[cb]` applied to one row of a depth hidden state.
	void head_apply(const GpuAlloc& hidden, i64 row, i64 cb, const GpuAlloc& out);
	// The table-driven RoPE the depth decoder needs (llama3 scaling).
	void rope_table(const GpuAlloc& x, const GpuAlloc& y, i64 rows, i64 heads, i64 head_dim,
	                i64 base);
	// The tied audio embedding table, one bf16 row at a time.
	void embed_row(i64 row, float* out) const;
	// sum over the 16 codebooks of the frame's embedding rows.
	void frame_embed(const i32* frame, float* out) const;
	// One frame's 15 remaining codebook tokens.
	void depth_frame(const std::vector<float>& backbone_hidden, i64 n_branch, i32 first_token,
	                 const BreezeParams& p, Rng& rng, std::vector<i32>& out);

	SafeTensors st_;
	std::string dir_;
	GpuCtx* g_ = nullptr;
	bool loaded_ = false;
	BreezeTokenizer tok_;
	MimiCodec codec_;

	// ── shapes (the checkpoints') and scalars (config.json's) ──
	i64 bb_hidden_ = 0, bb_layers_ = 0, bb_heads_ = 0, bb_kv_ = 0, bb_head_dim_ = 0, bb_inter_ = 0;
	float bb_eps_ = 1e-6f, bb_theta_ = 1e6f;
	i64 dp_hidden_ = 0, dp_layers_ = 0, dp_heads_ = 0, dp_kv_ = 0, dp_head_dim_ = 0, dp_inter_ = 0;
	i64 dp_vocab_ = 0, dp_embed_dim_ = 0, dp_embed_rows_ = 0;
	float dp_eps_ = 1e-5f;
	i64 te_hidden_ = 0, te_layers_ = 0, te_heads_ = 0, te_kv_ = 0, te_head_dim_ = 0, te_inter_ = 0;
	i64 te_vocab_ = 0, te_sliding_ = 512, te_eoi_ = 256000;
	// The attention scale is the *checkpoint's* `query_pre_attn_scalar^-0.5`, not
	// `head_dim^-0.5`: the two happen to agree on the shipped config (256), but
	// they are different fields and a checkpoint may set them apart. Getting this
	// wrong leaves the prompt's embeddings off-distribution, which is heard as the
	// model reading the wrong span of the prompt (e.g. the timbre instruction
	// instead of the text to speak).
	float te_eps_ = 1e-6f, te_scale_ = 1.0f, te_qpas_ = 256.0f;
	std::vector<i64> te_layer_types_;   // 0 = sliding, 1 = full attention
	std::vector<float> te_inv_sliding_, te_inv_full_;
	const float* te_eoi_embed_ = nullptr;
	const float* te_embed_t_ = nullptr;   // borrowed fp32 [te_vocab, te_hidden]

	i64 audio_vocab_ = 0, num_codebooks_ = 16, lm_head_n_ = 0;
	i64 codebook_pad_ = -1, codebook_eos_ = 0, backbone_eos_ = -1;
	i64 codec_codebook_size_ = 2048, max_seq_len_ = 2048;

	// ── resident device weights (`keep`) ──
	std::vector<BackboneLayer> bb_;
	GpuAlloc bb_norm_;
	QuantLinear lm_head_;
	std::vector<DepthLayer> dp_;
	GpuAlloc dp_norm_;
	QuantLinear dp_projector_;
	std::vector<GpuAlloc> head_w_, head_scale_;   // [num_codebooks - 1]
	GpuAlloc dp_inv_dev_;                         // the depth RoPE's inverse frequencies
	std::vector<float> dp_inv_freq_;              // ... their host form
	const StTensor* dp_embed_t_ = nullptr;

	// ── the text encoder's streamed matrices ──
	std::vector<TeLayer> te_;
	TeMat te_proj_;
	const float* te_norm_ = nullptr;

	// ── per-run state ──
	Branch branches_[2];
	i64 kv_rows_ = 0, depth_rows_ = 17;
	std::vector<float> prefill_embeds_[2];
	// One scratch row for both sampling sites, sized for the wider of the two
	// heads (`lm_head`, 2052 rows) and never for the narrower one. The *width* is
	// not this vector's size but the argument the caller passes to
	// `sample_logits`; see the note there for why the distinction is load-bearing.
	std::vector<float> sample_buf_;
	std::vector<i32> codes_;
};

}  // namespace phi::media
