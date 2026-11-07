// LoRA loader.
//
// A LoRA checkpoint is a safetensors file holding, per adapted module, a rank-r
// factorisation of a weight delta:
//
//     W_eff = W_base + strength * (alpha / r) * B @ A   (A: [r, k], B: [n, r])
//
// `strength` is the user's own scaling for that file: ComfyUI's LoraLoader
// `strength_model` / `strength_clip` (nodes.py: default 1.0, range -100..100,
// step 0.01). Its `calculate_weight` multiplies the *delta* by it
// (`weight += (strength * alpha) * lora_diff`, comfy/weight_adapter/lora.py,
// with `alpha = lora_alpha / rank`), so the base weight is never touched, a
// value above 1 exaggerates the LoRA, a value between 0 and 1 damps it, and a
// negative one subtracts its effect. Exactly 0 means "leave this file out":
// ComfyUI returns `(model, clip)` untouched in that case, and so does this
// loader (the file is not loaded at all - see MediaLoraLoader).
//
// One strength per file, shared by every consumer of the chain. A chain here is
// one node feeding the DiT *and* the text encoder, so they see the same value;
// ComfyUI keeps two widgets because it patches two models separately, and a
// file that adapts only one of them behaves identically under either scheme.
//
// Three naming conventions are in the wild and all three appear in files people
// drop next to these checkpoints, so the parser normalises them to one internal
// form (the base weight name with the `.weight` suffix stripped, e.g.
// `transformer_blocks.0.attn.to_q`):
//
//   * diffusers / peft:  `<base>.lora_A.weight` / `<base>.lora_B.weight`
//                        (optionally `<base>.lora_A.default.weight`),
//   * kohya sd-scripts:  `<base>.lora_down.weight` / `<base>.lora_up.weight`,
//   * ComfyUI native:    the same as diffusers, with a display-name prefix that
//                        this loader strips (`diffusion_model.`, `model.`,
//                        `unet.`, `transformer.`).
//
// Up to `kMaxLoras` files are supported and their order matters: each file's
// delta is applied in turn, so two LoRAs that touch the same module compose as
// `W + d1 + d2` (not `W + d2 + d1`). The order is the user's, held in
// settings.json's tool settings.
//
// Application is on the *host fp32* weight, before the loader quantises it:
//   * for a safetensors int8 checkpoint the stored weight is dequantised to
//     fp32, the delta is added, and the result is re-quantised with the same
//     convrot the checkpoint used, so the LoRA lands inside the int8 weight the
//     kernels already consume;
//   * for a float checkpoint the weight is decoded to fp32, the delta is
//     added, and the result is stored as the load path's bf16 tensor.
// Both cases are handled by `GpuCtx::load_checkpoint_linear`, which is the one
// place a weight is turned into a device tensor.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "host/st.hpp"

namespace phi::media {

// The most files a recipe may chain. Two is the requirement; the constant lives
// here so the tool schema, the UI and the loader agree on the number.
inline constexpr int kMaxLoras = 2;

// The widest rank a runtime correction may have - i.e. the width of the [rows,
// rank] hidden form both DiT activation plans reserve for it. It is a bound and
// not a measurement because the plans are static: they are asked what a shape
// costs before any LoRA is loaded, and the answer has to be the same one the
// forward pass allocates. A chain whose factors exceed it is refused at load
// with its rank, rather than over-committing the card by a scratch nobody
// priced. 256 = `kMaxLoras` files of rank 128.
inline constexpr i64 kLoraRankBound = 256;

// The strength widget's own contract, copied from ComfyUI's LoraLoader so a
// value means here what it means there: default 1.0 (the delta at the scale its
// file was trained at), -100..100 with a 0.01 step. The range is a limit on the
// UI, not on the maths - `LoraSet::add` takes whatever number it is given - but
// the two agree on the default, because that is the value a chain that has no
// stored setting gets.
inline constexpr float kLoraStrengthDefault = 1.0f;
inline constexpr float kLoraStrengthMin = -100.0f;
inline constexpr float kLoraStrengthMax = 100.0f;

// One file of a chain, as the settings and the workflow parameters carry it: the
// resolved path plus the user's strength for that slot. It exists so a strength
// cannot be separated from the file it belongs to - the settings store two
// parallel lists per chain, and a list of these keeps them in step from the
// settings file all the way to `LoraSet::add`.
struct LoraRef {
	std::string path;
	float strength = kLoraStrengthDefault;
};

// Two chains are the same chain only when every file *and* its strength match:
// the tool compares the resolved spec against the engine's cached one to decide
// whether the loaded handles are stale, and a strength change alone rewrites
// every adapted weight - so it has to count as a change, or the next run would
// silently reuse the previous value's weights.
inline bool operator==(const LoraRef& a, const LoraRef& b) {
	return a.path == b.path && a.strength == b.strength;
}
inline bool operator!=(const LoraRef& a, const LoraRef& b) { return !(a == b); }

class LoraSet {
public:
	// Loads `path` (a safetensors LoRA) and appends it to the chain. Throws
	// MediaError with a readable reason when the file is not a LoRA this loader
	// understands. Files are applied in the order added.
	//
	// `strength` scales that file's delta (see the header's formula): it is
	// folded into every factor's `scale` here, once, so the folding path
	// (`apply` / `apply_rows`) and the runtime correction path (`deltas_for`)
	// cannot disagree about it. A strength of 0 adds nothing and is the caller's
	// cue not to load the file at all; a negative one is legal (it subtracts).
	void add(const std::string& path, float strength = kLoraStrengthDefault);
	void clear() { files_.clear(); }
	bool empty() const { return files_.empty(); }
	size_t size() const { return files_.size(); }
	const std::vector<std::string>& paths() const { return paths_; }

	// A short human description ("name (42 tensors, r<=16)") for the tool report.
	// A file loaded at a strength other than 1.0 says so ("name @0.60 (...)"):
	// the report is how a user sees which slider value a run actually used.
	std::string describe() const;

	// Applies every delta registered for `target` (the base weight name with the
	// `.weight` suffix removed) onto the fp32 weight `w` [n, k], in file order.
	// Returns how many factors matched. A factor whose (n,k) disagrees with the
	// weight is skipped rather than applied wrongly.
	int apply(const std::string& target, float* w, i64 n, i64 k) const;

	// The row-slice form of `apply`: adds the deltas to rows [row0, row0+rows) of a
	// [n, k] weight, using the matching rows of each factor's B. This is what lets
	// the caller fold a LoRA a slab of rows at a time, which is the only way to
	// keep the fp32 working set bounded - the whole-matrix form materialises the
	// entire [n, k] plane (4 bytes per element) just to add a delta to it.
	int apply_rows(const std::string& target, float* w, i64 row0, i64 rows, i64 k) const;

	// True if any file registers a factor that could match `target` (used to
	// decide whether a weight needs the fp32 round trip at all).
	bool touches(const std::string& target) const;

	// One factor as the *runtime* correction needs it: the raw rank-r product,
	// un-applied. `W_eff = W + scale * b @ a`, a is [rank, k] and b is [n, rank],
	// with `scale` already carrying the file's strength (`strength * alpha/rank`).
	//
	// This is what lets a streamed layer keep its base weight verbatim and add the
	// delta to the GEMM's output instead of folding it into the weight: folding a
	// 21 GB checkpoint's adapted matrices means decode + adapt + requantise on
	// *every* sampling step, which is the one thing a LoRA chain cannot afford.
	struct Delta {
		i64 n = 0, k = 0, rank = 0;
		float scale = 1.0f;         // strength * alpha / rank
		const float* a = nullptr;   // [rank, k]
		const float* b = nullptr;   // [n, rank]
	};
	// Appends every factor of `target` whose (n, k) is exactly the weight's, in
	// file order. Returns how many were appended; a factor whose shape disagrees
	// is skipped rather than applied wrongly, exactly as `apply` does.
	int deltas_for(const std::string& target, i64 n, i64 k, std::vector<Delta>* out) const;


private:
	struct Factor {
		std::string target;      // normalised base name without ".weight"
		i64 n = 0, k = 0, rank = 0;
		float scale = 1.0f;      // strength * alpha / rank
		std::vector<float> A;    // [rank, k]
		std::vector<float> B;    // [n, rank]
	};
	// w[o][i] += scale * sum_r B[row0 + o][r] * A[r][i] for o in [0, rows).
	static void apply_rows_of(const Factor& f, float* w, i64 row0, i64 rows, i64 k);
	struct File {
		std::string path;
		std::string name;        // file basename
		float strength = kLoraStrengthDefault;   // the user's widget value
		std::vector<Factor> factors;
	};
	std::vector<File> files_;
	std::vector<std::string> paths_;
};

// Normalises a LoRA tensor key to its target module name (without `.weight`).
// Returns an empty string when the key is not a LoRA factor key.
std::string lora_normalize_target(const std::string& key, bool* is_a, bool* is_b);

}  // namespace phi::media
