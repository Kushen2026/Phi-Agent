// Qwen3-VL-8B text encoder (the Qwen-Image-2.1 conditioning tower).
//
// Source checkpoint: `qwen3vl_8b_int8_convrot.safetensors` - the Qwen3-VL-8B
// language tower with int8 tensorwise + convrot(256) weights, the same
// quantisation format the image DiT uses, so the tower rides the same
// `quant_convrot` + `int8_gemm` pair.
//
// Presentation, straight from comfy/text_encoders/qwen_image21.py:
//
//   * the prompt is wrapped in
//       <|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n
//       <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
//     and everything before the *second* `<|im_start|>` (the system turn) is
//     dropped, so the DiT conditions on the user turn and the assistant prefix
//     (`QwenImage21TEModel.encode_token_weights`);
//   * `layer_norm_hidden_state = False`: the output is the last layer's hidden
//     state, **without** `model.norm` - the transformers 4.57 `hidden_states[-1]`
//     the Qwen results are tuned to.
//
// Per layer (Qwen3-VL language block):
//   t = rms(x)                                   (input_layernorm, eps 1e-6)
//   q = int8_linear(t, q_proj)  [S, 4096]        32 heads x 128
//   k = int8_linear(t, k_proj)  [S, 1024]        8 heads x 128  (GQA)
//   v = int8_linear(t, v_proj)  [S, 1024]
//   q = rms_per_head(q), k = rms_per_head(k)     (q_norm / k_norm, eps 1e-6)
//   q, k = rope_half(q, k)                       theta 5e6, pairs (i, i+64)
//   o = causal_attn(q, k, v) 32 heads / 8 kv
//   x = x + o_proj(o)
//   t = rms(x)                                   (post_attention_layernorm)
//   x = x + down_proj(silu(gate(t)) * up(t))
//
// The text-only presentation makes the multimodal RoPE degenerate: its three
// position sections all carry the same token index, so the rotation is the
// ordinary Qwen one over the full head - which is what `rope_half` implements.
//
// ── the ViT is deliberately not ported, and why that is exact ───────────────
//
// ComfyUI's `QwenImage21Qwen3VLClipModel` can run this tower's *vision* path and
// splice the picture's rows into the sequence; `TextEncodeQwenImage21` uses it
// exactly when no VAE is connected (`keep_vision = len(ref_latents) == 0`), so
// that a reference still conditions the model through the tower when there is no
// way to VAE-encode it.
//
// phi's chain has no such case: `MediaQwenImageEditEncode` *requires* the image
// VAE whenever a reference is given (it throws otherwise), builds one
// reference latent per image from the same pixels the tower is shown, and hands
// them to the DiT - which is the reference's own `keep_vision = false` branch,
// rows and all. So the two paths agree whenever phi can run at all, and the
// vision path would be dead code that could never be exercised, in a tower that
// streams 8.7 GB of weights per call.
//
// The invariant is enforced where it matters: the encode node refuses a
// reference without a VAE (rather than silently conditioning on a picture the
// DiT has no latent for), and the DiT sizes each reference block from the latent
// the VAE actually produced.
#pragma once

#include <string>
#include <vector>

#include "models/model_common.hpp"

namespace phi::media {

struct TextEncoder8BConfig {
	i64 hidden = 4096;
	i64 n_layers = 36;
	i64 n_heads = 32;
	i64 n_kv_heads = 8;
	i64 head_dim = 128;
	i64 intermediate = 12288;
	i64 vocab = 151936;
	float eps = 1e-6f;
	float rope_theta = 5000000.0f;
	// `encode()` stops after this layer (the conformance hook); the shipped value
	// is the last layer, whose raw output is what the presentation returns.
	i64 out_layer = -1;   // -1 = n_layers - 1
};

class TextEncoder8B {
public:
	void open(const std::string& path, GpuCtx* gpu);
	// LoRA chain applied at load time (may be null). Set before open().
	void set_loras(const LoraSet* loras) { loras_ = loras; }
	const TextEncoder8BConfig& config() const { return cfg_; }

	// Text-only: ids -> [seq, hidden] fp32 (row major), the tower's last-layer
	// hidden state.
	std::vector<float> encode(const std::vector<i32>& ids);

	// ── weight residency (goal #2) ──────────────────────────────────────────
	//
	// The same shape the 32B tower uses: the 36-layer stack is 9 GB of int8 and is
	// re-read in full on every call, so on anything but the reference 6 GB card a
	// window of resident layers removes most of that traffic. `plan_residency(S)`
	// runs once per call, after the activation frame exists, and sizes the window
	// from the system's own limit (core/runtime/vram_window.hpp), pricing each
	// layer by the arena growth its own loader produced - so int8, a packed family,
	// fp8, f16, bf16 and fp32 all size themselves without a format list here.
	void plan_residency(i64 S);
	void release_resident();
	i64 resident_layers() const { return res_n_; }
	u64 resident_bytes() const { return res_bytes_; }

private:
	struct Layer {
		QuantLinear q, k, v, o, gate, up, down;
		GpuAlloc in_ln, post_ln, q_norm, k_norm;
	};
	// `into` picks the arena the layer lands in: null streams it through the weight
	// arena (reused by the next layer), a pointer keeps it for the whole call - the
	// resident window's arena.
	void upload_layer(i64 index, Layer& L, GpuArena* into = nullptr);
	// The header-driven price of one layer, used to seed - and to reserve room for -
	// the window. See the 32B tower's equivalent for the policy.
	u64 layer_bytes_estimate(i64 index) const;

	GpuCtx* g_ = nullptr;
	TextEncoder8BConfig cfg_;
	SafeTensors st_;
	std::string path_;
	const LoraSet* loras_ = nullptr;

	// The embedding table stays on the host: it is 151936 x 4096 int8 = 622 MB and
	// `encode()` gathers one row per prompt token out of the mapping, exactly as
	// the Qwen3-4B tower does (its own note has the measurement) - but here the
	// rows are int8 and carry a per-row scale, so the gather dequantises too.

	// The resident layer window (see `plan_residency`). A separate arena from the
	// streaming one on purpose: `GpuCtx::new_layer()` resets the streaming arena
	// before every layer, and a window living in it would be thrown away 36 times
	// per call.
	GpuArena res_arena_;
	std::vector<Layer> res_layers_;
	i64 res_n_ = 0;
	u64 res_bytes_ = 0;
	bool res_planned_ = false;
};

}  // namespace phi::media
