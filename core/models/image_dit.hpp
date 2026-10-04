// Qwen-Image-2.1 DiT (comfy/ldm/qwen_image21/model.py).
//
//   * 32 blocks, inner dim 4096 (32 heads x 128), no GQA
//   * in_channels 64 (the 16x-compressed VAE latent), **no patchify**: `img_in`
//     is Linear(64 -> 4096) applied to the flattened latent
//   * text and image tokens share one sequence, laid out
//       [text chunk][ref-1 image][text chunk][ref-2 image]...[text chunk][target]
//     and attended with a *block-causal* mask: a text row attends causally
//     inside its own run, every image row attends to everything up to the end of
//     its own block (`block_causal_attention` in the reference)
//   * **every reference keeps its own latent grid.** The reference's
//     `build_sequence` walks `zip(bounds, ref_latents + [x])` and reads `h, w`
//     off *each* latent, so a portrait and a landscape reference are NOT
//     stretched onto one shared grid - each contributes `h * w` tokens on its own
//     geometry, with its RoPE ids centred on the target grid (`hh = arange(h) -
//     (h - h//2) + 0.5 * (h % 2 - H % 2)`, and the same for `w`). Reference
//     *sizes* therefore have to travel with the reference latents, which is what
//     `ImageRefGeom` carries.
//   * one shared adaLN: `modulation = SiLU -> Linear(4096, 4*4096)`, and the
//     prefix rows (text + references) are modulated from **t = 0** while the
//     target row uses the sampled t
//   * LayerNorm (no affine) * (1 + scale), tanh-gated residuals
//   * SwiGLU FFN with the fused [gate; up] projection
//   * axial RoPE, axes (16, 56, 56), theta 10000, applied to *interleaved* pairs
//     (the flux/Lumina convention `apply_rope1` implements, which is what
//     `qkv_prep` does), with per-token (pos, h, w) ids
//   * int8 tensorwise + convrot on every big matmul, so the same
//     `quant_convrot` / `int8_gemm` pair the H3 DiT uses runs them
//   * the final layer scales the **target rows only** and the module does NOT
//     negate its output (unlike Flux/Lumina)
#pragma once

#include <string>
#include <vector>

#include "models/model_common.hpp"

namespace phi::media {

struct ImageDitConfig {
	i64 dim = 4096;
	i64 n_layers = 32;
	i64 n_heads = 32;
	i64 head_dim = 128;
	i64 in_channels = 64;
	i64 context_dim = 4096;
	i64 mlp_hidden = 12288;   // gate_up is [2*mlp_hidden, dim]
	i64 t_freq = 256;
	float eps = 1e-6f;
	float rope_theta = 10000.0f;
	std::vector<int> axes_dims{16, 56, 56};
};

// One reference image's latent geometry: the grid its VAE encode produced, in
// latent pixels (canvas / 16). The reference's `build_sequence` takes one of
// these per reference block, so the DiT never has to assume that every reference
// shares the first one's grid - which is what used to stretch a reference whose
// aspect differed from its neighbour's, and to price every reference at the
// first one's pixel count.
struct ImageRefGeom {
	i64 h = 0, w = 0;
};

// Everything one forward needs besides the weights. Allocated once per forward so
// the block loop does no allocation at all.
struct QIScratch {
	GpuAlloc hidden, txt, normed, qkv, q, k, v, attn, proj, gate_up, hh, down, outp;
	// `ub` is the sequence's own block-causal key bound; `ub_t` is the
	// prefix-cached target pass's (one entry per target row, all of them the whole
	// sequence - see `forward`).
	GpuAlloc ids, ub, ub_t, rope, mod_t, mod_p;
	void alloc(GpuCtx* g, i64 S, i64 dim, i64 mlp, i64 ch, i64 n_img, i64 n_ref_tok,
	           i64 ref_px);
};

struct QILayer {
	QuantLinear q, k, v, out, gate_up, mlp_out;
	GpuAlloc norm_q, norm_k;   // [head_dim] fp32
	// The runtime LoRA correction of the four projections the block runs (the
	// fused q|k|v counts as one). Empty when no LoRA touches the layer.
	LoraTail tq, tout, tgate_up, tmlp_out;
};

class ImageDiT {
public:
	void open(const std::string& path, GpuCtx* gpu);
	// LoRA chain applied at load time (may be null). Set before open().
	void set_loras(const LoraSet* loras) { loras_ = loras; }
	const ImageDitConfig& config() const { return cfg_; }

	// x: [in_channels, H, W] fp32 (H, W = canvas/16) - the noisy latent.
	// context: [n_txt, dim] fp32, the text tower's output.
	// refs: reference latents, each [in_channels, h, w] fp32 in the same
	// normalised space as x (the model's `process_latent_in` has been applied),
	// with `ref_geom[i]` its own latent grid - references are NOT assumed to share
	// one grid (ComfyUI's `build_sequence` reads each latent's own h/w).
	// `slots` says how many context rows precede each reference block in the
	// sequence (ComfyUI's `image_slots`); with no slots every reference goes in
	// front of the text.
	// Returns [in_channels, H, W] fp32, the model's velocity prediction.
	// `cond_key` is the identity of the conditioning (the caller's ConditioningData
	// address). It is what the prefix K/V cache is keyed on: the same key with the
	// same canvas means the prefix rows are bit-identical, so the kept K/V are
	// still valid. nullptr disables the cache for the call (see set_prefix_cache).
	std::vector<float> forward(const std::vector<float>& x, i64 H, i64 W,
	                           const std::vector<float>& context, i64 n_txt,
	                           const std::vector<std::vector<float>>& refs,
	                           const std::vector<ImageRefGeom>& ref_geom, float sigma,
	                           std::vector<i32> slots = {}, const void* cond_key = nullptr);

	// Weight residency. The checkpoint is 7.25 GB and this card's budget ~5 GB, so
	// a plan that streams *every* block leaves a third of the card idle while
	// paying for it in disk traffic on every step. `plan_residency` therefore
	// sizes a window from the machine and *keeps* that many blocks in
	// `res_arena_` for the whole sampling loop; only the rest stream through the
	// weights arena. It is called once, between the encode and the sampling loop,
	// with the shape the run will use - and it uploads the window there, so the
	// first step does not pay for it.
	//
	// `ref_px` is one entry per reference block, in packing order: the pixel count
	// of *that* reference's own latent grid. Each one adds two `in_ch x px`
	// staging tensors to the activation arena and `px` tokens to the sequence, so
	// the list is what makes the plan's frame the frame the forward really uses
	// (heterogeneous references used to be priced at the first one's grid).
	void plan_residency(i64 H, i64 W, i64 n_txt, const std::vector<i64>& ref_px = {});

	// The checkpoint's own weight precision, as a short label (int8 / w4a8 / w6a8 /
	// nvfp4 / fp8 / f16 / bf16 / f32). Goal #1: it is the file's, not a converted
	// one. The media tool report prints it.
	const char* weight_precision() const;

	i64 resident_layers() const { return res_n_; }
	u64 resident_bytes() const { return res_bytes_; }
	u64 act_bytes() const { return act_bytes_; }
	// True once `plan_residency` has run for the current phase, even when it
	// decided to keep *zero* blocks (a card too small for one, or a shape whose
	// frame fills it). The sampler node asks this instead of `resident_layers()!=0`,
	// which would re-run the whole plan - the upload probe included - on every
	// sampling step of a zero-window run.
	bool residency_planned() const { return res_planned_; }
	void release_resident();

	// ── the prefix K/V cache (ComfyUI's `QwenImage21Cache`) ────────────────
	//
	// The prefix rows - the prompt and every reference block - are processed with
	// the *t = 0* modulation and a causal/block-causal mask, so their keys and
	// values do not depend on the sampling step at all. Keeping them turns each
	// step from "run text + references + target through 32 blocks" into "run the
	// target through 32 blocks, attending into the kept prefix", which is the whole
	// of the reference node's trick.
	//
	// What it costs is VRAM, and that is why it is *opt-in* here rather than
	// automatic as in ComfyUI: this engine's binding resource is the room the
	// streamed weights need (a 19.5 GB checkpoint on a 6 GB card means every
	// megabyte the cache takes is a block that must come off the disk on every
	// step). The cache pays off when the weights are resident - a small checkpoint
	// or a large card - and loses when they are not. `plan_residency` therefore
	// asks the accountant whether the cache fits *after* the window it would
	// otherwise take, and silently leaves it out when it does not.
	//
	//   * the K/V of the prefix rows, per block, fp16 (the precision the attention
	//     itself runs at, so a cached step is bit-identical to an uncached one);
	//   * plus the target rows' own K/V, which are rewritten every step in the tail
	//     of the same buffer - the attention reads one contiguous key range, which
	//     is what keeps the kernels unchanged;
	//   * dropped by `release_resident` (the sampler's phase boundary), so it never
	//     outlives the run that planned it.
	void set_prefix_cache(bool want) { cache_want_ = want; }
	bool prefix_cache_wanted() const { return cache_want_; }
	// True when a plan accepted the cache and it fits this run's shape.
	bool prefix_cache_planned() const { return cache_ok_; }
	// True once the prefix pass has run for the current conditioning, i.e. from the
	// second step on: the run reports this as "the cache is doing something".
	bool prefix_cache_ready() const { return cache_ready_; }
	// Rows the cache holds per block, and the bytes both of its tensors take.
	i64 prefix_cache_rows() const { return cache_rows_; }
	u64 prefix_cache_bytes() const { return cache_charged_; }
	// Bytes a cache for this shape would take (both tensors, every block). Static
	// and cheap so a caller can report what the cache *would* cost.
	static u64 prefix_cache_charge(i64 n_prefix, i64 n_img, const ImageDitConfig& cfg);

	// ── dynamic residency (core/runtime/vram_window.hpp) ──────────────────
	//
	// `plan_residency` decides the window once, from the request and the card as it
	// looked before the sampling loop. The loop then holds that decision for up to
	// a few minutes, while the driver's own figure for this process moves (the
	// WebView2 UI repainting, the desktop compositor, another chain handing its
	// arenas back) and while the plan's own terms turn out to be estimates. This is
	// the once-per-step correction: it measures what the window and the rest of the
	// loop actually hold, and grows or sheds blocks to put the process back inside
	// the 90-100 % band of the driver's limit. Returns the count resident after it.
	//
	// Growing is an upload of the next block(s) out of the checkpoint; shedding is
	// the window handed back and re-taken at the smaller count. Both are speed
	// decisions - the arithmetic and the pixels do not depend on the window, only
	// the disk traffic does - so every path here is allowed to fail quietly and
	// leave the run streaming.
	i64 tune_residency();

	// How the last `plan_residency`/`tune_residency` left the window, as a
	// percentage of `VramBudget::limit()` (for the report and the tests).

	// One block's tensor bytes, summed from the checkpoint header in `open()`; the
	// window's charge is a whole number of these rounded up to the arena's chunks.
	u64 layer_bytes() const { return res_layer_bytes_; }

	static constexpr i64 kTotalBlocks = 32;

	// Every allocation one forward makes in the activation arena, in order. The
	// arena is a bump allocator charged per *chunk*, so the honest cost of a
	// shape is the simulated chunk list, not the sum of the tensors - see
	// `estimate_dit_activation_bytes` and `arena_chunk_cost`.
	static std::vector<u64> activation_sizes(i64 S, i64 dim, i64 mlp_hidden, i64 in_ch, i64 n_img,
	                                         i64 ctx_dim, i64 n_txt,
	                                         const std::vector<i64>& ref_px);
	static u64 estimate_dit_activation_bytes(i64 S, i64 dim, i64 mlp_hidden, i64 in_ch, i64 n_img);
	// The same list, charged the way `GpuArena` books it at this granularity.
	static u64 activation_charge(i64 S, i64 dim, i64 mlp_hidden, i64 in_ch, i64 n_img,
	                             i64 ctx_dim, i64 n_txt, const std::vector<i64>& ref_px,
	                             u64 chunk_bytes);

private:
	void probe();
	void upload_layer(i64 i, QILayer& L, GpuArena* into = nullptr);
	// The sizes `upload_layer` allocates for block `i`, in that order.
	std::vector<u64> layer_alloc_sizes(i64 i);
	// `tune_residency`'s two halves: append `want` blocks to the window, or hand it
	// back and re-take it `drop` blocks smaller. Both return how many moved.
	i64 grow_resident(i64 want);
	i64 shrink_resident(i64 drop);
	void quant_gemm(const QuantLinear& w, const GpuAlloc& x, i64 m, i64 k, const GpuAlloc& out);

	// One block's pass over one set of rows.
	//
	// The uncached forward runs a single pass over the whole sequence; the cached
	// one runs two - the prefix rows, whose K/V are kept for the rest of the run,
	// and then the target rows, which attend into them. Everything that differs
	// between those passes is in here, so `block()` stays one implementation of the
	// block and the three callers cannot drift apart.
	struct BlockPass {
		i64 rows = 0;       // rows of `hidden` this pass runs
		i64 n_prefix = 0;   // of those, the leading ones modulated from t = 0
		GpuAlloc k, v;      // where the K/V go (the scratch's, or the cache's)
		i64 kv_row0 = 0;    // first row of that buffer this pass writes
		i64 kv_rows = 0;    // key rows the attention reads (0 => this pass's rows)
		i64 ids_row = 0;    // first row of the ids / RoPE table this pass owns
		GpuAlloc umask;     // per-query-row key bound (indexed from row 0 of `q`)
	};
	void block(const QILayer& L, const GpuAlloc& hidden, const BlockPass& bp,
	           const GpuAlloc& mod_t, const GpuAlloc& mod_p, const GpuAlloc& ids,
	           const GpuAlloc& rope, QIScratch& scratch);

	// The prefix cache: allocate (charging the accountant), use, drop.
	bool alloc_prefix_cache(i64 n_prefix, i64 n_img);
	void free_prefix_cache();
	// The window half of `release_resident`: the plan takes the cache before it
	// sizes the window and must not drop it again in the same call.
	void release_resident_window();

	GpuCtx* g_ = nullptr;
	ImageDitConfig cfg_;
	SafeTensors st_;
	std::string path_;
	bool open_ = false;
	const LoraSet* loras_ = nullptr;

	// activation quantiser scratch, sized by the forward
	GpuAlloc q8_, s8_;
	// The LoRA correction's [S, rank] hidden form; `kLoraRankBound` rows wide.
	GpuAlloc lora_h_;

	// The prefix cache (see set_prefix_cache). `cache_reserve_` is what the plan
	// held back for it, so the window it takes is planned for the card the cache is
	// also living on.
	bool cache_want_ = false;
	bool cache_ok_ = false;
	bool cache_ready_ = false;
	GpuAlloc cache_k_, cache_v_;
	i64 cache_rows_ = 0;
	u64 cache_charged_ = 0;
	u64 cache_reserve_ = 0;
	// What the kept prefix belongs to: the conditioning identity and the canvas.
	const void* cache_key_ = nullptr;
	i64 cache_key_rows_ = 0;
	GpuArena res_arena_;
	std::vector<QILayer> res_layers_;
	i64 res_n_ = 0;
	u64 res_bytes_ = 0;
	bool res_planned_ = false;
	u64 plan_reserve_ = 0;
	u64 act_bytes_ = 0;
	u64 res_layer_bytes_ = 0;   // one block's tensor bytes (probe())
	u64 res_inflight_ = 0;      // what one streamed block costs the weight arena
	// What one *more resident* block really costs `res_arena_`, measured by the
	// plan's own upload loop (whole chunks, not tensor bytes). The runtime tuner
	// sizes its corrections with this rather than with the granularity rounding of
	// a single block, because the two differ by up to a chunk - and growing by the
	// smaller number is what let a window reach 99 % of the limit with nothing left
	// for the block the sampling loop streams next.
	u64 res_block_charge_ = 0;
	int res_percent_ = 0;       // last planned/measured usage, % of the limit

	// resident (bf16, read straight out of the checkpoint layout)
	GpuAlloc img_in_w_, txt_in_w_, txt_out_w_, ttl1_w_, ttl2_w_, mod_w_, nout_w_, pout_w_;
	GpuAlloc txt_norm_w_;   // text_norm.weight + 1 (ZeroCenteredRMSNorm)
};

}  // namespace phi::media
