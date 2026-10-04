// Qwen3-VL-32B text encoder (the MiniMax H3 conditioning tower).
//
// A frozen interface: the DiT and the sampler consume only the hidden
// vector and its width; the ViT path is private. Signature changes via `_v2`.
//
// Source checkpoint: `minimax_h3_qwen3vl_32b_convrot_int4.safetensors` (15 GB,
// int4 + convrot). The backbone is the Qwen3-VL language tower; the vision tower
// is ViT-based and consumes the patch blocks produced by vision_prep.
//
// Two entry points matter to the rest of the pipeline:
//   * `encode()`   text-only: ids -> hidden [L, hidden]
//   * `encode_vl()` text + vision: the same, with visual patch tokens spliced in
//     at the positions vision_prep marked.
#pragma once

#include <string>
#include <vector>

#include "models/model_common.hpp"
#include "text/vision_prep.hpp"

namespace phi::media {

// Every constant below is read off the checkpoint header (see
// the architecture note for that dump). The earlier
// n_heads=40 / intermediate=13824 values were guesses and are wrong:
//
//   model.embed_tokens.weight        BF16 [151936, 5120]
//   layers.N.self_attn.q_proj.weight I8   [8192, 2560]   8192 = 64 heads * 128
//   layers.N.self_attn.k_proj.weight I8   [1024, 2560]   1024 =  8 heads * 128 (GQA)
//   layers.N.self_attn.v_proj.weight I8   [1024, 2560]
//   layers.N.self_attn.o_proj.weight I8   [5120, 4096]   4096 = 8192 K / 2 (int4 pack)
//   layers.N.mlp.gate_proj.weight    I8   [25600, 2560]  intermediate = 25600
//   layers.N.mlp.up_proj.weight      I8   [25600, 2560]
//   layers.N.mlp.down_proj.weight    I8   [5120, 12800]  12800 = 25600 / 2 (int4 pack)
//
// The K axis of every quantised weight is *halved* because `convrot_w4a4` stores
// two int4 nibbles per byte along the input dimension; the loader must expand it
// before the GEMM (see quant.* and the notes in host/st.hpp).
struct TextEncoder32BConfig {
	i64 hidden = 5120;
	i64 n_layers = 50;
	i64 n_heads = 64;
	i64 n_kv_heads = 8;    // GQA: k_proj/v_proj are 8 heads wide
	i64 head_dim = 128;
	i64 intermediate = 25600;
	i64 vocab = 151936;
	float eps = 1e-6f;
	float rope_theta = 5000000.0f;
	i64 out_layer = 49;   // run all 50 layers, return the last hidden
	// ViT (vision tower): hidden 1152, 27 blocks, self-attention (no GQA),
	// LayerNorm (weight *and* bias), mlp intermediate 4304, patch 16x16 spatial
	// with a temporal patch of 2.
	i64 vit_hidden = 1152;
	i64 vit_layers = 27;
	i64 vit_heads = 16;
	i64 vit_intermediate = 4304;
	i64 vit_patch_t = 2;
	i64 patch = 16;
	i64 merge = 2;        // spatial merge, so one ViT token covers 32x32 px
};

// One projection of the 32B tower, in the checkpoint's own precision:
//   * the original int8 tensorwise pair (one code per element + per-row fp32
//     scale) - uploaded verbatim, on the int8 GEMM;
//   * the shipped convrot_w4a4 pair (packed int4 + per-row scale) - also
//     uploaded verbatim, on the int4 GEMM;
//   * a float source (f16 / bf16 / f32, fp8 -> f16) - kept at its own precision
//     and run through the dense fp16/bf16/f32 GEMM, *not* repacked to int4.
//
// Namespace scope rather than class members so the resident window can hold a
// `std::vector<LayerW>` and the loaders can spell them unqualified.
struct TeLinear32 {
	GpuAlloc w, s;              // weight + per-row scale (see `i8`)
	GpuAlloc d;                 // dense float weight
	DType dtype = DType::Unknown;   // F16 / BF16 / F32 when dense
	bool i8 = false;            // `w` is int8 tensorwise (else packed int4)
	bool dense() const { return d.res != nullptr; }
};

struct TeLayerW {
	TeLinear32 q, k, v, o, gate, up, down;
	GpuAlloc in_ln, post_ln, q_norm, k_norm;
};

class TextEncoder32B {
public:
	void open(const std::string& path, GpuCtx* gpu);
	// LoRA chain applied at load time (may be null). Set before open().
	void set_loras(const LoraSet* loras) { loras_ = loras; }
	const TextEncoder32BConfig& config() const { return cfg_; }

	// Text-only: ids -> [seq, hidden] fp32 (row major).
	std::vector<float> encode(const std::vector<i32>& ids);

	// Text + vision: runs the ViT over every block and splices the merged patch
	// embeddings into the stream at `p.block_at`. Returns [L, hidden] where L is
	// the *expanded* sequence length (text tokens + merged visual tokens), which
	// is what the DiT's `embed` argument expects.
	std::vector<float> encode_vl(const PromptTokens& p);

	// The vision half of encode_vl: the merged visual rows spliced into the
	// *input-embedding* stream (visual rows from the ViT merger, text rows out of
	// `model.embed_tokens`), [L, hidden] with L == p.ids.size(). encode_vl() runs
	// the 50-layer backbone over exactly this matrix, so this entry point is what
	// lets the vision path be validated without paying for 15 GB of int4 weight
	// streaming. No frozen signature changed.
	std::vector<float> encode_vl_embeds(const PromptTokens& p);

	// ── weight residency (goal #2) ──────────────────────────────────────────
	//
	// The 50-layer tower is 15 GB of int4, re-read in full on every call, and on
	// any card bigger than the reference 6 GB the same is true of most of it: the
	// activation footprint at a prompt of a few hundred tokens is ~40 MB, so the
	// question is never the frame, it is how much of the stack the card can hold.
	//
	// `plan_residency(S)` answers it with the engine's one window formula
	// (core/runtime/vram_window.hpp): the limit is what the *system* grants this
	// process, the terms are what the loop really holds (the activation arena is
	// already allocated when this runs, and VramBudget::local() is the honest
	// measure of it), one streamed layer's own charge is reserved so a
	// non-resident layer still has somewhere to land, and the window is then the
	// largest number of layers that fits beside all of that - priced in arena
	// chunks and corrected by the upload loop's own measurement, exactly like the
	// DiTs' plans. The layer's *precision* is part of the term rather than an
	// assumption: what the plan charges is the growth of the arena while a layer is
	// uploaded through the same loader the loop uses, so int4, int8, fp8, fp16,
	// bf16 and fp32 all size themselves correctly.
	//
	// `encode_embeds` calls this before its layer loop; `release_resident` hands
	// the window back (the phase boundary the DiT's own plan reads).
	void plan_residency(i64 S);
	void release_resident();
	i64 resident_layers() const { return res_n_; }
	u64 resident_bytes() const { return res_bytes_; }

	// Debug aid: stop after `n` backbone layers (layer-by-layer conformance).

private:
	// The 50-layer int4 backbone over a caller-built [S, hidden] input-embedding
	// matrix. encode() and encode_vl() differ only in what they hand it.
	//
	// `mrope` is the [S, head_dim] per-token rotary table the vision path builds
	// (see `build_mrope_table`): half cos, half sin, indexed by token. Null means
	// the ordinary sequence-position rotary, which is what a text-only prompt
	// uses - and what it *must* use, since Qwen3-VL's 3-D positions degenerate to
	// the sequence index exactly when no vision span is present.
	std::vector<float> encode_embeds(const float* xf, i64 S,
	                                 const std::vector<float>* mrope = nullptr);

	// The header-driven price of one backbone layer, in the engine's own upload
	// layout (see the .cpp). Used to seed - and to reserve room for - the window.
	u64 layer_bytes_estimate(i64 index) const;

private:

	// ViT + merger over every vision block: row-major [rows, hidden] merged
	// patches plus the per-block merged-token count (which must equal the block's
	// placeholder run, or the splice has nowhere to go).
	std::vector<float> run_vision(const std::vector<VisionBlock>& blocks,
	                              std::vector<i64>* rows_per_block);

	GpuCtx* g_ = nullptr;
	TextEncoder32BConfig cfg_;
	SafeTensors st_;
	std::string path_;
	bool open_ = false;
	const LoraSet* loras_ = nullptr;
	// Dtype of the stored embedding table (bf16 in this checkpoint). One added
	// private member, the same shape as the image tower's header grew for
	// the same reason: the gather in encode() needs to know how wide a row is
	// before it can pick the conversion. No public signature changes.
	DType wdtype_ = DType::BF16;

	// The resident layer window (see `plan_residency`). `res_arena_` is a separate
	// arena from the streaming one on purpose: `GpuCtx::new_layer()` resets the
	// streaming arena before every layer, and a window living inside it would be
	// thrown away 50 times per call. `res_layers_[i]` is layer `i`'s upload, valid
	// for the whole call once the plan has taken it.
	GpuArena res_arena_;
	std::vector<TeLayerW> res_layers_;
	i64 res_n_ = 0;
	u64 res_bytes_ = 0;
	bool res_planned_ = false;
};

}  // namespace phi::media
