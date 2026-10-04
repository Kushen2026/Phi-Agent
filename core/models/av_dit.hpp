// MiniMax H3 diffusion transformer (the audio+video backbone).
//
// A frozen interface: the sampler depends only on `forward()`'s signature
// below; the internal block layout is private and may evolve without touching the
// ABI, and a signature change lands as a `_v2` symbol.
//
// Architecture, read off the pruned int8 checkpoint
// (`minimax_h3_hybrid_ref2va_b25-49_pruned_int8_convrot`) tensor by tensor:
//
//   blocks.N.attn.qkv_proj.weight  I8 [21504, 5376]  21504 = 3 * 7168, so it is a
//                                                    fused MHA (56 heads x 128),
//                                                    not GQA: n_kv_heads == n_heads
//   blocks.N.attn.out_proj.weight  I8 [5376, 7168]
//   blocks.N.mlp.fc1.weight        I8 [28672, 5376]  28672 = 2 * 14336: fc1 carries
//                                                    gate and up in one matrix
//   blocks.N.mlp.fc2.weight        I8 [5376, 14336]
//   blocks.N.adaln_proj.linear     F16 [96768, 8]    96768 = 18 * 5376; the input is
//                                                    the 8-wide adaln_t_table row
//   blocks.N.norm1/norm2           BF16 [5376]
//   blocks.N.attn.q_norm/k_norm    BF16 [128]        per-head RMSNorm
//
// Every quantised matrix is int8_tensorwise + convrot(256) (a sibling
// `*.comfy_quant` tensor spells that out). There is no time_embedder: the
// timestep enters purely through `adaln_t_table` [1025, 8]. 3D patchify on the
// video latent is (1, 2, 2); the audio latent is consumed as a parallel token
// stream that shares the attention; conditioning is re-injected every step.
//
// A hybrid range splits the stack into two halves:
//   [0, hybrid_lo)  video-only blocks
//   [hybrid_lo, hybrid_hi)  joint audio+video blocks
//   [hybrid_hi, n_layers)   audio-only blocks
// The exact split is read from the checkpoint (`hybrid_range = 25..49`);
// forward() is agnostic to it.
#pragma once

#include <string>
#include <vector>

#include "models/model_common.hpp"

namespace phi::media {

struct AvDitConfig {
	i64 hidden = 5376;
	// Flow shifts. The DiT itself only needs them to carry the audio stream onto
	// the video schedule (comfy/ldm/minimax/model.py::time_shift_sigma, with
	// shift 12 / audio_shift 3 for the released workflow => audio_scale 4).
	float sigma_shift_video = 12.0f;
	float sigma_shift_audio = 3.0f;
	i64 n_layers = 50;
	i64 n_heads = 56;
	i64 head_dim = 128;
	i64 n_kv_heads = 56;   // == n_heads: the checkpoint fuses QKV as 3*56*128
	i64 ffn_hidden = 14336;   // fc1 emits 2 * 14336 (gate and up fused)
	i64 v_channels = 24;   // video latent channels
	i64 a_channels = 32;   // audio latent channels
	i64 patch_t = 1, patch_h = 2, patch_w = 2;
	i64 hybrid_lo = 25;    // first joint block (video-only before it)
	i64 hybrid_hi = 50;    // first audio-only block
	i64 text_dim = 5120;   // width of the TE hidden the DiT conditions on
	float norm_eps = 1e-5f;
	// MiniMaxH3Model's own defaults: norm_eps = qk_norm_eps = final_norm_eps =
	// 1e-5 (comfy/ldm/minimax/model.py). The 1e-6 that stood here was copied from
	// the Qwen3 text tower, which uses a different constant.
	float qk_norm_eps = 1.0e-5f;
};

// `plan_residency` diagnoses itself into this struct (plan §4): every number the
// residency decision was made from, so `--media-bench --video` can print the
// budget, the activation estimate, the weight space left and the layer count.
struct H3Residency {
	bool planned = false;
	bool degraded = false;      // fewer layers resident than the plan asked for
	u64 budget = 0;             // limit = min(budget_percent x the driver's figure, the
	                            // ceiling the card supports) - see VramBudget
	u64 act_dit = 0;            // estimated activation peak of one DiT evaluation
	u64 act_vae = 0;            // estimated activation peak of the video VAE
	u64 overhead = 0;           // everything resident besides the window and the DiT's
	                            // activations: what the accountant was already holding at
	                            // plan time (measured) plus the refiner / streamed-block
	                            // maximum (priced in arena chunks)
	u64 available = 0;          // budget - act_dit - overhead, capped by the device
	u64 request_reserve = 0;    // the request's own pixel weight (generated frame area plus
	                            // every reference block's area, times the per-pixel weight).
	                            // Its *reference* share is charged to the loop (see
	                            // request_pixel_reserve_bytes / plan_residency); the generated
	                            // frame's share is already inside `act_dit`. Reported whole.
	u64 layer_bytes = 0;        // one block's int8 weights, measured at open()
	u64 block_charge = 0;       // what one *more resident* block costs the resident
	                            // arena, priced in that arena's own chunks
	                            // (`arena_chunk_cost`), which is more than the block
	                            // weighs: the runtime tuner grows by this number
	u64 stream_charge = 0;      // what one streamed block costs the weight arena, in
	                            // its chunks - the reservation a correction must leave
	                            // for the block the sampling loop streams next
	u64 resident_bytes = 0;
	i64 res_main_n = 0;         // resident blocks (0 => every block streams)
	i64 n_layers = 0;
	i64 s_tokens = 0;           // S = text + refs + audio + video tokens
	i64 dit_chunk = 0;          // activation rows per dispatch the plan settled on
	i64 s_max = 0;              // largest S this budget would hold (0 = unknown)
	std::string reason;         // why it degraded, for the error path
};

// Where each modality lands in the concatenated token sequence. The DiT runs one
// attention over [video tokens | audio tokens]; this records the split so the
// sampler and the debug dumps agree on ordering without re-deriving it from T.
struct PackedLayout {
	i64 n_video = 0;   // *target* video tokens (after patchify)
	i64 n_audio = 0;   // *target* audio tokens
	i64 n_text = 0;    // text/conditioning tokens
	// ref2va only: the reference blocks pack between the text span and the
	// targets, so they add rows to the packed sequence that belong to neither the
	// text nor the two target streams (see the reference's PackedLayout.segments:
	// kinds ref_img / ref_audio). Both are 0 on the t2va path.
	i64 n_ref_video = 0;   // reference video rows (patchified ref latents)
	i64 n_ref_audio = 0;   // reference audio rows
	i64 total() const { return n_video + n_audio + n_text + n_ref_video + n_ref_audio; }
};

// ── the ref2va reference payload ───────────────────────────────────────────
//
// One entry per reference block, in the order the reference implementation packs
// them (`payload["refs"]`, i.e. the order the ComfyUI node builds: images, then
// each video's (soundtrack, frames), then standalone audio). The geometry is the
// *reference's* geometry: a reference image is resized to its own canvas, so its
// latent grid is independent of the target's.
//
//   kind Image : video rows only (`latent`, `latent_t` == 1)
//   kind Video : video rows, and audio rows when `audio_latent` != null
//   kind Audio : audio rows only
//
// The pointers are borrowed for the duration of the forward call.
struct H3Ref {
	enum Kind : i32 { Image = 0, Video = 1, Audio = 2 };
	i32 kind = Image;
	// Video/image rows: [24, latent_t, latent_h, latent_w] row-major fp32, the
	// VAE-encoded reference (VideoVae::encode). Null => the block has no video
	// rows. latent_h/latent_w are the *ref's* latent grid (canvas/16); both must
	// be even so the 2x2 patchify is exact.
	const float* latent = nullptr;
	i64 latent_t = 0, latent_h = 0, latent_w = 0;
	// Audio rows: [32, 2, ref_audio_t] channel-major fp32 (AudioVae::encode).
	// Null => no audio rows.
	const float* audio_latent = nullptr;
	i64 ref_audio_t = 0;
};

// The packed row count the reference blocks of `refs` contribute: patchified
// video rows (one per 2x2 patch of each ref latent frame) plus 2*ref_audio_t per
// reference audio. Exposed so `video_gen` can ask for a residency plan for the
// right S before it runs a forward. Never throws.
i64 h3_ref_row_count(const H3Ref* refs, i64 n_refs);

// The references' own pixel area (each reference video/image's latent grid times
// the VAE's 16x spatial scale, over its frame count). The companion of
// `h3_ref_row_count`: rows move the packed sequence `S`, pixels move the plan's
// per-pixel reference charge. Never throws.
u64 h3_ref_pixel_area(const H3Ref* refs, i64 n_refs);

// `torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=float32)`
// written into `dst` (n floats): the noise field the reference's
// `_cond_video_rows` augments a reference's patch rows with. It is a bit-level
// contract with torch's CPU generator (see the derivation in av_dit.cpp), so
// it is exposed as a free function: `tests/t_cond_noise.cpp` is what holds it
// against torch's own output, and nothing else can.
void h3_torch_randn_fill(float* dst, i64 n, u64 seed);

// Estimate of the video VAE's decode peak, in the same units as the DiT's own
// estimate (plan §4 asks for both so the phase with the larger activation is the
// one the residency plan reserves space for). The VAE is a 36-block ViT3D over
// 16x16x4-pixel patches plus a causal 3D-conv front end; calibrated against
// VideoVae::last_activation_bytes() on this machine.
u64 estimate_h3_vae_activation_bytes(i64 H, i64 W, i64 T_frames);

class AvDiT {
public:
	// `dit_path` is the DiT checkpoint; the token refiner,
	// patch projections and final head live in the same file.
	void open(const std::string& dit_path, GpuCtx* gpu);
	// LoRA chain applied at load time (may be null). Set before open().
	void set_loras(const LoraSet* loras) { loras_ = loras; }
	const AvDitConfig& config() const { return cfg_; }

	// One denoising evaluation. All pointers are host fp32 buffers.
	//
	//   x_v      [B, 24, T, H/16, W/16]   noisy video latent
	//   x_a      [B, 32, 2, T40]          noisy audio latent
	//   embed    [L, text_dim]            TE hidden (conditioning)
	//   tags     [L]                      per-token modality (0 visual, 1 text)
	//   t        scalar                   flow timestep in [0, 1]
	//   ref_v_lat [ref_T, ...]            optional reference video latent (may
	//   ref_a_lat [...]                   be null); re-injected as extra cond
	//   out_v / out_a                     caller-allocated, same shapes as x_v/x_a
	//
	// The function is pure: no allocation leaks to the caller, and the arenas it
	// uses are reset internally per call.
	void forward(const float* x_v, const float* x_a, i64 T, i64 H, i64 W,
	             const float* embed, const i32* tags, i64 L, float t,
	             const float* ref_v_lat, i64 ref_T, const float* ref_a_lat,
	             float* out_v, float* out_a);

	// The ref2va entry point: the same evaluation as forward(), with `refs`
	// reference blocks packed between the text span and the targets
	// (PackedLayout's refs list). forward(x, ..., nullptr, 0, nullptr, ...) is
	// exactly this call with n_refs == 0, and is verified bit-identical to the
	// t2va path that predates it.
	//
	// `cond_noise_aug` is the reference's `visual_cond_noise_aug`
	// (VISUAL_COND_TIMESTEP = 0.999 in the released workflow): reference video rows
	// are `aug*r + (1-aug)*noise` with `noise = torch.randn(r.shape,
	// generator=torch.Generator().manual_seed(noise_seed))` restarted *per
	// condition*. `noise_seed` is the reference's `payload["seed"]` (0 when the
	// caller does not set one). Reference *audio* rows are not augmented: the
	// reference's AUDIO_COND_TIMESTEP is 1.0, which makes the same expression a
	// no-op, so the code path is skipped here rather than asserted.
	void forward_refs(const float* x_v, const float* x_a, i64 T, i64 H, i64 W,
	                  const float* embed, const i32* tags, i64 L, float t,
	                  const H3Ref* refs, i64 n_refs, float cond_noise_aug,
	                  u64 noise_seed, float* out_v, float* out_a);

	// Layout of the last forward() call (for the contract dumps / tests).

	// --- dynamic budget (plan §4, W9) ───────────────────────────────────────
	//
	// One denoising evaluation's activation peak for this shape, computed from
	// the buffers this implementation actually allocates (see the allocation
	// list in av_dit.cpp) rather than from a generic transformer formula:
	//
	//   hidden [S,hidden]            + fused qkv/attn [S,3*7168]
	// + q/k/v [S,7168] x3           + chunked FFN scratch
	// + rope table [S,48,2]         + token-refiner scratch
	//
	// `plan_residency` adds the 20% safety margin the plan asks for.
	//
	// `kDefaultOverheadBytes` is now only a *fallback*: what the loop carries
	// besides the window and the DiT's activations is **measured** at plan time
	// (`vram_budget().local()`, minus the streamer's own chunks) and what the
	// refiner / a streamed block costs is priced by `arena_chunk_cost` from the
	// checkpoint's own tensor sizes, so neither term has to be guessed on another
	// machine. The constant is used when no device has answered the accountant
	// yet, and by the video tool's request fitter.
	static constexpr u64 kDefaultOverheadBytes = (256ull + 770ull + 256ull) << 20;

	// What one token-refiner block costs the weight arena. The refiner re-streams it
	// on *every* step (`embed_rows`), so it is live beside the resident window in
	// the steady state; `plan_residency` prices it with `arena_chunk_cost` over the
	// checkpoint's own tensors (~800 MB of chunks for 770 MB of bf16 weights) and
	// takes `max(refiner, one streamed DiT block)` - the two never overlap inside a
	// step, since the refiner runs in the embedding phase and the block loop after
	// it. This constant is the fallback for a checkpoint-less plan.
	static constexpr u64 kRefinerArenaBytes = 770ull << 20;

	// `chunk` is the activation row-chunk the run will actually use (the fused
	// qkv scratch and the FFN working set scale with it, not with S); it defaults
	// to the engine's preferred 2048 rows. `plan_residency` lowers it when that is
	// what makes a shape fit.
	static constexpr i64 kDefaultChunk = 2048;
	static u64 estimate_h3_dit_activation_bytes(i64 S, i64 n_video, i64 n_audio, i64 T_latent,
	                                            i64 H, i64 W, i64 chunk = kDefaultChunk);

	// What the activation arena really books for this shape, priced the way the
	// accountant books it: the arena's own chunk arithmetic over `forward()`'s
	// allocations, in `forward()`'s order.
	//
	// The estimate above is a closed form with a 20 % margin, which is right for a
	// report and wrong for a plan: the charge is per *chunk*, so a sum of tensor
	// bytes is systematically below what `GpuArena::alloc` reserves, and a planner
	// that spends the last of the card on the estimate is refused mid-run. This is
	// the number `plan_residency` sizes the window against, and it is a function of
	// the shape alone - resolution, clip length and the packed row count - so a
	// checkpoint of any precision prices the same (the activation frame is fp16 /
	// fp32 scratch, never weights).
	//
	// `phase` is the row count the token refiner runs over. It is >= the text span
	// and <= S, and it only appears in the refiner's scratch, which shares a region
	// with the block loop's buffers; passing S is a safe upper bound.
	// `ref_patch_rows` is the packed row count the reference blocks add (patchified
	// reference-video rows plus 2*ref_audio_t). The reference rows are staged through
	// `F.prow` on *every* step (`embed_rows`), which is a per-pixel buffer that does
	// not scale with `n_video`, so it is priced here rather than left to the row
	// count: a long reference video or a big reference image is exactly the case the
	// old estimate called comfortable and the driver refused mid-run.
	static u64 dit_activation_cost(i64 S, i64 n_video, i64 n_audio, i64 text_dim, i64 phase,
	                               i64 chunk, i64 ref_patch_rows = 0);

	// The activation estimate this shape will *actually* run with: the default
	// chunk when that fits `budget - overhead`, otherwise the largest smaller chunk
	// on the halving ladder that does. `plan_residency` and the video tool's
	// request fitter both go through here, so "does this fit" has exactly one
	// answer in the process.
	static u64 plan_activation(i64 S, i64 n_video, i64 n_audio, i64 T_latent, i64 H, i64 W,
	                           u64 budget, u64 overhead, i64* chunk_out = nullptr,
	                           i64 refiner_rows = 0, i64 ref_patch_rows = 0);

	// The VAE half of the same decision (the free function above).

	// Decides how many of the 50 blocks stay resident for this request, from the
	// real VRAM budget (`VramBudget`) and the estimate above. Idempotent for a
	// given (H, W, T, text, ref rows) signature; the numbers land in
	// `residency()`.
	//
	// `ref_rows` is the number of packed rows the reference blocks add between the
	// text span and the targets (H3Ref rows, i.e. patchified ref-video rows plus
	// 2*ref_audio_t). It is part of S and therefore of the residency decision; 0
	// is the t2va path and keeps the numbers byte-for-byte what they were.
	//
	// `extra_resident` is the footprint the *caller* is holding on the card for
	// the whole DiT phase and that the fixed reserve does not know about: on the
	// ref2va path that is the reference build's VAE weights and the vision
	// tower's arenas, which are live before the first DiT block and stay live
	// through the sampling loop. The reserve is a fixed, calibrated number (the
	// upload ring, the weight streamer and the keep arena), so anything else that
	// is already resident has to come out of the same budget as the resident
	// window - otherwise the window is sized for a card the run does not have.
	// 0 (the t2va path) leaves every number unchanged.
	// `ref_pixels` is the reference material's own *pixel* area (the sum over the
	// reference blocks of their latent grid times the VAE's 16x spatial scale; a
	// reference image is scaled to the generation's canvas, a reference video to
	// its own). It is not part of `S` - `ref_rows` covers the packed rows - but it
	// is what the plan's pixel weight prices (see `request_pixel_reserve_bytes`),
	// so a large reference image or a long reference video leaves fewer blocks
	// resident. 0 leaves the t2va numbers unchanged, exactly as `ref_rows` does.
	//
	// Since the activation term became the arena's *exact* cost
	// (`dit_activation_cost`), the generated frame's pixels are already inside it; the
	// references' *own* pixels - which do not scale with the packed row count - are
	// charged on top (the reference share of `request_pixel_reserve_bytes`). `ref_rows`
	// still moves the packed row count `S` and therefore the frame, which is where a
	// reference's row cost lives.
	void plan_residency(i64 H, i64 W, i64 T_frames, i64 text_tokens, i64 ref_rows = 0,
	                    u64 extra_resident = 0, u64 ref_pixels = 0);
	const H3Residency& residency() const { return res_; }

	// The plan decides the window and records it in `residency()`; the window's
	// tensors are *uploaded* lazily by the first `forward()`, right after the token
	// refiner releases the weight arena. That ordering is what keeps the window from
	// competing with the refiner's ~800 MB on the *first* step; from the second step
	// on the refiner streams while the window is resident, which is why the plan
	// charges the refiner to the loop as well (`reserve`).
	//
	// The upload stops at the first block the accountant refuses and keeps the
	// blocks that did land (`forward()`), instead of throwing the whole window away:
	// a window is a speed decision, and the blocks already resident cost nothing to
	// keep.

	// Hands the resident window and the streaming arena back to the device. The
	// video VAE needs the whole budget to itself and the two phases never overlap
	// (plan §4 "阶段边界 release"); the next plan_residency() re-reads them.
	void release_weights();

	// ── dynamic residency (core/runtime/vram_window.hpp) ───────────────
	//
	// `plan_residency` answers the request's own inputs - resolution, clip length,
	// the references' packed rows and their VAE footprint - once, from the card as
	// it looked before sampling. The loop then runs for minutes while the driver's
	// figure for this process moves, so the window is re-decided once per step from
	// what the process is actually holding: `other` (the activation frame, the
	// refiner's chunks, the caller's footprint) is measured as the ledger minus the
	// resident arena's own charge, the live figure is the driver's usage floored by
	// the ledger, and the target is the 90-100 % band of the driver's limit.
	//
	// Growing appends blocks to the resident arena; shedding hands the window back
	// and re-takes it smaller (a bump arena cannot return one block). The window is
	// a disk-traffic decision only - the pixels do not depend on it - so every
	// failure here leaves the run streaming instead of failing it. Returns the
	// number of resident blocks after the correction.
	i64 tune_residency();

	// Blocks that will never be re-read for the current plan (the resident head
	// of the stack). Reported by the bench; 0 before the first plan.

	// One block's tensor bytes, measured from the checkpoint header in open() (0
	// before that). The request fitter reports the resident window a shape would
	// get, and the window is a count of *these* bytes - the number is exposed so
	// the pre-check cannot quote a different block size than the planner uses.
	u64 layer_bytes() const;

	// The checkpoint's own weight precision, as a short label (int8 / w4a8 / w6a8 /
	// nvfp4 / fp8 / f16 / bf16 / f32). Goal #1: the file's, never a converted one.
	const char* weight_precision() const;

private:
	GpuCtx* g_ = nullptr;
	AvDitConfig cfg_;
	std::string dir_;
	PackedLayout layout_;
	bool open_ = false;
	const LoraSet* loras_ = nullptr;
	// The one piece of new state this header carries: it is what residency()
	// reports, so it belongs to the class rather than the side table. Everything
	// else the implementation needs lives in `av_dit.cpp`'s registry, keyed
	// by `this` (the same pattern video_vae.cpp uses for the same reason).
	H3Residency res_;
};

}  // namespace phi::media
