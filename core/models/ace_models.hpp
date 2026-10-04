// ACE-Step 1.5 conditioning + DiT.
//
// `ace_models.cpp` carries the mapping to the checkpoint:
// `acestep_v1.5_xl_sft_bf16.safetensors` (bf16, 9.97 GB) holds the 32-layer
// Qwen3-shaped DiT (`decoder.`), the three towers that turn text/lyrics/codes
// into its cross-attention context (`encoder.` = text projector + lyric encoder
// + timbre encoder) and the audio-code tokenizer/detokenizer pair
// (`tokenizer.`, `detokenizer.`).
//
// ── precision ──────────────────────────────────────────────────────────────
//
// The bundle is bf16 and ComfyUI runs it at bf16 (`supported_inference_dtypes =
// [bfloat16, float32]`), so no weight here is converted: `AceLinear` uploads
// the file's own bytes and the GEMM reads them at that precision. The first cut
// of this file routed every projection through `load_quant_linear` (bf16 ->
// int8 tensorwise + convrot); that is both a lossy conversion the reference
// never makes and host-side work over 10 GB of weights, which is what pegged the
// CPU. See `ace_text.hpp` for the same contract on the two towers.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "models/model_common.hpp"

namespace phi::media {

// What the text-encode node hands the DiT, computed once per generation.
struct AceCondition {
	std::vector<float> encoder_hidden;   // [n, 2048] row-major, the condition encoder output
	i64 n = 0;
	std::vector<float> context;          // [T, 128] row-major (src_latents || chunk_mask)
	i64 T = 0;
	// What the audio-code LM's own residency plan kept (see `AceLm::plan_residency`),
	// carried through here because the LM's window is handed back before the DiT's
	// plan runs and nothing else would be left to read. Purely informational: the
	// sampler reports it so a run can be told apart from one that re-read all 8.4 GB
	// of the LM on every decoded token.
	i64 lm_resident_layers = 0;
	i64 lm_total_layers = 0;
	u64 lm_resident_bytes = 0;
};

class AceDiT {
public:
	AceDiT();
	~AceDiT();

	void open(const std::string& path, GpuCtx* gpu);
	// x: [T,64] row-major (the noisy latent). sigma in [0,1]; the DiT forms
	// timestep = sigma * 1000 internally (TimestepEmbedding scale=1000).
	// Returns [T,64] row-major (the model velocity / output).
	std::vector<float> forward(const float* x, i64 T, float sigma, const AceCondition& cond);

	// ── weight residency ────────────────────────────────────────────────────
	//
	// The bundle is 9.97 GB of bf16 and the card this engine plans against has a
	// ~5 GB ceiling, so the 32 decoder blocks (about 223 MB each) cannot all be
	// resident. Without a plan every block is re-read from the checkpoint and
	// re-uploaded *on every sampling step* - 7.1 GB of PCIe traffic per step, 50
	// steps of it for one clip. `plan_residency` keeps as many of the blocks as
	// fit beside the activation frame for the whole loop and streams only the
	// rest, which is the same decision `ImageDiT`/`AvDiT` make for their own
	// stacks (see core/runtime/vram_window.hpp).
	//
	// Called once by the sampler, before the loop, with the latent length and the
	// number of conditioning rows the forward will use (the activation frame is
	// sized from both); `release_resident` hands the card back for the VAE decode
	// that follows.
	void plan_residency(i64 T, i64 n_enc);
	void release_resident();
	i64 resident_layers() const;
	u64 resident_bytes() const;
	i64 n_layers() const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

// What the text-encode node needs from the audio-code LM.
class AceConditionBuilder {
public:
	AceConditionBuilder();
	~AceConditionBuilder();

	// opens the DiT bundle (encoder/tokenizer/detokenizer live inside it)
	void open(const std::string& dit_bundle_path, GpuCtx* gpu);
	// text_hidden: [n_text,1024] (qwen3-0.6b last hidden of the qwen3_06b prompt)
	// lyric_hidden: [n_lyric,1024] (qwen3-0.6b **layer-0 / embedding** rows of the
	// whole lyrics prompt: `qwen3_06b(lyrics, layer=[0])[:, 0]` in ace15.py)
	// codes: [n_codes] semantic audio codes, or empty (silence / no codes path).
	// frames: the DiT latent length T (the detokeniser output is truncated to T).
	AceCondition build(const std::vector<float>& text_hidden, i64 n_text,
	                   const std::vector<float>& lyric_hidden, i64 n_lyric,
	                   const std::vector<i32>& codes, i64 T);

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

}  // namespace phi::media
