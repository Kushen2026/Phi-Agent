// ACE-Step 1.5 text side: the Qwen3-0.6B embedder tower, the Qwen3-4B audio-code
// LM, and the shared stack runner the DiT bundle's four small towers use.
//
//   * `AceQwen3` - the qwen3-0.6B **embedder** (`qwen_0.6b_ace15.safetensors`,
//     bf16, 1.19 GB). It produces the two host-side feature tensors the
//     conditioning builder consumes: the last hidden state of the Qwen3 prompt
//     (the "caption" prompt) and the *layer 0* hidden state of the lyrics prompt
//     (`comfy/text_encoders/ace15.py::ACE15TEModel.encode_token_weights`), which
//     is the whole lyrics-prompt sequence of the tower's embedding output.
//
//   * `AceLm` - the qwen3-4B **audio-code LM** (`qwen_4b_ace15.safetensors`,
//     bf16, 8.38 GB), run autoregressively with classifier-free guidance to
//     sample the semantic audio codes, mirroring
//     `comfy/text_encoders/ace15.py::sample_manual_loop_no_classes`.
//
// ── the weights keep the checkpoint's own precision ────────────────────────
//
// Every ACE checkpoint is bf16 and ComfyUI runs it at bf16 (its
// `supported_inference_dtypes` are [bfloat16, float32]), so nothing here
// converts a weight: `AceLinear` uploads the file's bytes verbatim and the GEMM
// reads them at that precision (`dispatch_gemm_f16`'s half-precision B operand,
// fp32 accumulate). That is the one path in the engine that takes a
// half-precision weight without going through the int8 requantiser.
//
// This is a deliberate change from the first cut, which routed these layers
// through `load_quant_linear` (bf16 -> int8 tensorwise + convrot). Two things
// were wrong with it. Numerically it ran a bf16 model at int8 weights - a lossy
// conversion the reference never makes and the released checkpoints were never
// calibrated for. Structurally the requantiser is *host* work, one fp32 row at a
// time: 10 GB of DiT plus 8.4 GB of LM is what pegged the CPU at 95% and made a
// load take minutes. Uploading the file's own bytes removes both - the weights
// go from the mapping to VRAM through the pinned ring and the CPU does no
// arithmetic at all.
//
// The activations stay fp32 on the host side of each dispatch, and the attention
// operands are staged as fp16 (`dispatch_f32_to_f16`) because the engine's tiled
// flash kernel reads packed halves - the same arrangement every other
// transformer in this engine uses.
//
// The four small towers inside the DiT bundle (`encoder.lyric_encoder`,
// `encoder.timbre_encoder`, `tokenizer.attention_pooler`, `detokenizer`) have
// exactly this shape too, so `ace_models.cpp` runs them through the same stack
// runner; that is what the `ace_qwen3_*` entry points below are for. They are
// internal (this header is new and not frozen).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "models/model_common.hpp"
#include "runtime/sched.hpp"   // GpuArena: the LM's resident-weight window

namespace phi::media {

// A plain Qwen3 tower (no vision tower) used as ACE 1.5's embedder / LM.
struct AceQwen3Config {
	i64 hidden = 0, n_layers = 0, n_heads = 0, n_kv_heads = 0, head_dim = 0,
	    intermediate = 0, vocab = 0;
	float eps = 1e-6f, rope_theta = 1e6f;
};

// ── a linear layer at the checkpoint's own precision ───────────────────────

// One linear of the ACE stack (the DiT's 11 projections, the encoders' 7), in the
// precision the checkpoint stores it in - the same policy as every other loader:
//
//   * bf16 / f16 / f32 -> `w` at that dtype, which `dispatch_gemm_f16` reads
//     directly (`b_is_bf16` / `b_is_f32`);
//   * fp8 (E4M3/E5M2) and nvfp4 -> `w` in f16, which is exact (both have fewer
//     significand bits than fp16, so this is a widening and not a lossy step);
//   * int8 tensorwise and the packed families (w4a8 / w6a8 / grouped int8) -> the
//     engine's int8 tensorwise + convrot pair (`i8`), the same bus the DiTs and the
//     text towers run: the codes stay codes and the `quant_convrot` /
//     `int8_gemm` pair that consumes them is the one every other chain uses.
//
// The int8 form needs an activation scratch (the rotated int8 codes and their
// per-row scales); the caller supplies it (see `AceQScratch`).
struct AceLinear {
	GpuAlloc w;                 // the weight: int8 codes when `i8`, else the float dtype
	GpuAlloc s;                 // its per-row fp32 scale (int8 only)
	i64 n = 0, k = 0;
	DType dtype = DType::BF16;  // F16 / BF16 / F32 when `!i8`
	bool i8 = false;
	bool is_int8() const { return i8; }
};

// The activation scratch an int8 `AceLinear` needs: `q8` is [rows, k] int8 and `s8`
// is [rows] fp32, exactly what the int8 GEMM takes. One instance per pass, sized
// for the widest (rows, k) that pass runs, so a quantised checkpoint pays one
// allocation per forward rather than one per layer.
struct AceQScratch {
	GpuAlloc q8, s8;
	u64 q8_cap = 0, s8_cap = 0;
	// Grows (never shrinks) to hold `rows * k` codes; a no-op when it already does.
	void need(GpuCtx& g, i64 rows, i64 k);
	bool ready() const { return q8.res != nullptr && s8.res != nullptr; }
};

// Uploads `base + ".weight"`. `into` selects the arena (null = the streaming
// weight arena, which the next `new_layer()` releases).
AceLinear ace_upload_linear(GpuCtx& g, const SafeTensors& st, const std::string& base,
                            GpuArena* into);

// A 1-D tensor (an RMSNorm weight, a bias) as fp32. Norms and biases are the
// one place this file widens: the reference casts them to the compute dtype and
// accumulates in fp32, and the engine's `dispatch_norm` and bias epilogue are
// fp32, so fp32 here is the *more* accurate of the two, never a lossy step.
GpuAlloc ace_upload_vector(GpuCtx& g, const SafeTensors& st, const std::string& name,
                           GpuArena* into);

// y[m, n] = x[m, k] @ w[n, k]^T (+ bias[n]). `x`/`y` are fp32; the weight is
// read at its own precision. A default-constructed `bias` means "no bias".
// `qs` is only consulted when `w` is int8 (see `AceLinear`); a null one there is a
// refusal with the shape, because an int8 weight without its activation scratch is
// a call that cannot be satisfied rather than one to silently approximate.
void ace_linear(GpuCtx& g, const AceLinear& w, const GpuAlloc& x, i64 m, const GpuAlloc& y,
                const GpuAlloc& bias = GpuAlloc{}, AceQScratch* qs = nullptr);

// One Qwen3 block's weights, streamed through the weight arena. The four small
// towers in the DiT bundle have the same layout, so all call sites share it.
struct AceQwen3Layer {
	AceLinear q, k, v, o, gate, up, down;
	GpuAlloc in_ln, post_ln, q_norm, k_norm;
};

// Activations for one stack pass over `rows` rows (row-major, fp32).
//
// `kh`/`vh` are only used when the pass has no KV cache (a one-shot tower
// pass): the tiled attention still needs the pass's own K/V in fp16.
struct AceStackScratch {
	GpuAlloc t, qr, kr, vr, qn, kn, qh, kh, vh, attn, proj, gate, up, hh, down;
	// The int8 activation scratch, only touched when a layer's weight is quantised
	// (see `AceLinear`). Sized here for the widest (rows, k) this pass runs, so a
	// quantised checkpoint does not allocate inside the layer loop.
	AceQScratch qs;
	void alloc(GpuCtx& g, i64 rows, const AceQwen3Config& cfg);
};

// An fp16 K/V cache: one [capacity, n_kv_heads * head_dim] pair per layer.
struct AceKvCache {
	i64 layers = 0, capacity = 0, len = 0, kv_dim = 0;
	std::vector<GpuAlloc> k, v;
	void init(GpuCtx& g, i64 layers, i64 capacity, i64 kv_dim);
};

// Config detection and per-layer upload for the Qwen3 shape.
void ace_qwen3_probe(const SafeTensors& st, const std::string& prefix, AceQwen3Config* cfg);
// `into` selects the arena the layer's tensors land in (see `ace_upload_linear`):
// null streams the layer through the weight arena, where the next `new_layer()`
// releases it; a pointer keeps it in the caller's own arena for as long as the
// caller wants it (that is what `AceLm::plan_residency` builds its window with).
void ace_qwen3_upload_layer(GpuCtx& g, const SafeTensors& st, const std::string& prefix, i64 index,
                            AceQwen3Layer* L, GpuArena* into = nullptr);

// Affine RMSNorm and the plain add, exactly as `text_encoder_8b.cpp` writes
// them (the norm runs in fp32 regardless of the weight's stored precision).
void ace_rms_norm(GpuCtx& g, const GpuAlloc& in, i64 rows, i64 cols, const GpuAlloc& w,
                  const GpuAlloc& out, float eps);
void ace_add_inplace(GpuCtx& g, const GpuAlloc& acc, const GpuAlloc& b, i64 rows, i64 cols);

// Runs `n_layers` blocks starting at 0 over `hidden` [S, hidden] fp32 in place,
// then `norm_name` when `final_norm`. Streaming: each layer's weights are
// uploaded into the weight arena and released by the next layer's `new_layer`.
//
// `kv` (may be null): the K/V of this pass are written at rows
// [kv_row0, kv_row0 + S) and the attention reads keys [0, kv_len). `ub` is the
// per-query-row exclusive key bound of `dispatch_attn_flash` - causal
// (ub[r] = r+1), full (ub[r] = kv_len) or decoded (ub[r] = current length) -
// which is also what pins the fp16 tiled kernel: the fp32 kernel would read
// packed halves as fp32 and silently produce noise.
//
// `group` is 0 for one flat sequence. A positive value says the S rows are a
// *batch* of S/group independent windows of that many rows - the DiT bundle's
// attention pooler (6 rows: one special token + a 5-frame window) and
// detokenizer (5 rows) are like that. Then both the RoPE positions and the
// attention's key range restart per window, exactly as the reference's
// `x.view(B*T, P, D)` does, and each window is dispatched on its own (the flash
// kernels take one key range per query block, which a block-diagonal mask is
// not).
void ace_qwen3_forward(GpuCtx& g, const SafeTensors& st, const std::string& prefix,
                       const AceQwen3Config& cfg, i64 n_layers, bool final_norm,
                       const std::string& norm_name, const GpuAlloc& hidden, i64 S,
                       AceStackScratch& sc, AceKvCache* kv, i64 kv_row0, i64 kv_len,
                       const GpuAlloc& ub, i64 group = 0);

// Host-side embedding gather out of a bf16/f16/f32 table (one row per id).
void ace_gather_rows(const SafeTensors& st, const std::string& name, i64 hidden, const i32* ids,
                     i64 S, float* dst);

// Upload `n` u32s into an activations-arena allocation (opening its own
// command-list bracket when the caller has none).
GpuAlloc ace_upload_u32(GpuCtx& g, const u32* src, i64 n);

// ── the two towers ─────────────────────────────────────────────────────────

class AceQwen3 {
public:
	AceQwen3();
	~AceQwen3();

	void open(const std::string& path, GpuCtx* gpu);   // detects config from the header
	const AceQwen3Config& config() const;
	// ids -> hidden states of `hidden_states` index `out_layer`, [S, hidden]
	// row-major. That is the index the reference's `{"layer": [...]}` list form
	// names: 0 is the embedding output (no block runs, no final norm - the
	// lyrics conditioning), k >= 1 the state *after* k blocks, and -1 the last
	// block followed by the tower's final norm (the caption conditioning).
	std::vector<float> encode(const std::vector<i32>& ids, i64 out_layer = -1);

private:
	GpuCtx* g_ = nullptr;
	SafeTensors st_;
	AceQwen3Config cfg_;
};

// ── the Qwen3-4B audio-code LM ──────────────────────────────────────────────
//
// `generate()` decodes one token at a time and, for each token, runs the whole
// 36-layer stack once per CFG branch. Streaming every layer through the weight
// arena therefore re-reads all 8.4 GB of bf16 from the checkpoint and pushes it
// over PCIe **per decoded token** - `ceil(duration) * 5` times, so 300 x 8.4 GB
// on a 60 s request. The decode is a pure streaming workload otherwise: the
// token's own arithmetic is one row wide.
//
// So the LM keeps a **window** of layers resident for the duration of one
// `generate()` call, exactly like `ImageDiT`/`AceDiT` do for their blocks and for
// the same reason: a layer that is not resident is 192 MB re-read on every one of
// those 300 passes. The window is decided once, by `plan_residency()`, before the
// prefill, and handed back by `release_resident()` when `generate()` returns -
// including when it throws, because the DiT's own plan runs immediately after the
// codes are sampled and must see the card free (see `generate`).
class AceLm {
public:
	AceLm();
	~AceLm();

	void open(const std::string& path, GpuCtx* gpu);
	i64 vocab() const;

	// Decides the resident window for the **next** `generate()` call and uploads
	// it. The geometry the decision is priced for lives in the `plan_*` members and
	// is filled by `generate()` before the call (the signature takes no arguments,
	// the way `AvDiT` keeps the shape its plan was taken for). Idempotent: a second
	// call without an intervening `release_resident()` does nothing, and the window
	// can be re-taken after the release. A plan that cannot be measured (no device,
	// no budget) leaves the run fully streamed.
	void plan_residency();
	// Hands the window back to the device and the accountant. Called by the guard in
	// `generate()` on every exit path; safe to call when nothing was planned.
	void release_resident();
	i64 resident_layers() const { return res_n_; }
	u64 resident_bytes() const { return res_bytes_; }
	// The window the **last** plan took, kept across `release_resident()`. The
	// window itself is gone by the time `generate()` returns (the DiT's plan needs
	// the card), so a caller that wants to report what the code sampling ran with
	// has nothing left to read: these two are the record of it.
	i64 planned_layers() const { return res_last_n_; }
	u64 planned_bytes() const { return res_last_bytes_; }
	i64 total_layers() const { return cfg_.n_layers; }
	// Batched-prompt autoregressive sampling of audio codes with classifier-free
	// guidance, mirroring comfy/text_encoders/ace15.py::sample_manual_loop_no_classes.
	// `pos_ids` is the conditional prompt, `neg_ids` the unconditional one (used
	// only when cfg != 1). Returns the generated code ids (length <= max_new).
	// audio_start/audio_end bound the legal code id range; the LM samples among
	// [audio_start, audio_end) plus one extra "eos" logit scored with id eos_id.
	//
	// `min_new` is the reference's `min_tokens`: the eos branch is not offered
	// until `min_new < step`, and text-to-music passes `min_new == max_new` (both
	// `duration * 5`), so eos never fires and the call returns exactly `max_new`
	// codes. See `generate` for why that is load-bearing.
	std::vector<i32> generate(const std::vector<i32>& pos_ids, const std::vector<i32>& neg_ids,
	                          i64 max_new, i64 min_new, float cfg, float temperature, float top_p,
	                          i64 top_k, float min_p, u64 seed,
	                          i32 audio_start, i32 audio_end, i32 eos_id);

private:
	// `h` [hidden] one row -> `logits` [n] over the tied embedding rows [lo, hi).
	void logits(const std::vector<float>& h, i32 lo, i32 hi, std::vector<float>& out);

	GpuCtx* g_ = nullptr;
	SafeTensors st_;
	AceQwen3Config cfg_;
	// The tied embedding rows the head reads, uploaded once per range. These are
	// the checkpoint's own bf16 bytes (the head *is* the embedding table).
	i32 head_lo_ = -1, head_hi_ = -1;
	GpuAlloc head_w_;
	GpuAlloc eos_w_;
	i32 eos_row_ = -1;

	// ── the resident window (`plan_residency`) ──
	// The arena the window lives in. Its chunk granularity is the weight arena's,
	// so what the plan measures and what the accountant books are the same number
	// (see `ImageDiT::open`).
	GpuArena res_arena_;
	// `res_layers_[0..res_n_)` are the layers resident in it; the sampler reads
	// them instead of streaming them. Kept as a vector of the same struct the
	// streamed path builds one of, so the stack loop indexes both the same way.
	std::vector<AceQwen3Layer> res_layers_;
	i64 res_n_ = 0;
	u64 res_bytes_ = 0;     // the window's charge (`res_arena_.capacity()`)
	u64 res_layer_bytes_ = 0;   // one layer's tensor bytes, from the header (open())
	u64 res_inflight_ = 0;      // what streaming one layer costs the weight arena
	u64 res_reserve_ = 0;       // held + act + inflight, the plan's own floor
	bool res_planned_ = false;
	// True when the plan priced the tied-head slice against the window's room, i.e.
	// `logits` must allocate it from `res_arena_` rather than from `keep` (see
	// `plan_residency` / `logits`). Reset by `release_resident` through the
	// `head_lo_` cache marker being dropped.
	bool head_in_window_ = false;
	// What the last plan decided, for reporting (see `planned_layers`).
	i64 res_last_n_ = 0;
	u64 res_last_bytes_ = 0;

	// The geometry `plan_residency()` prices its frame from, filled by `generate()`
	// before it plans: the K/V cache rows the decode will hold, the scratch's row
	// count, the number of CFG branches, the branches' prefill lengths, the logit
	// range, and the tied-head slice the sampler will need (0 when the same range
	// is already uploaded - it is part of `held` then).
	i64 plan_cap_ = 0;
	i64 plan_rows_ = 0;
	i64 plan_nbr_ = 0;
	i64 plan_n_codes_ = 0;
	std::vector<i64> plan_pre_rows_;
	u64 plan_head_bytes_ = 0;
};

}  // namespace phi::media
