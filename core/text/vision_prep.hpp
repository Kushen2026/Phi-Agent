// Reference preprocessing for the MiniMax H3 prompt: text + image/audio/video
// references -> the token stream + patch blocks the Qwen3-VL text encoder eats.
//
// A frozen interface, and the one with the heaviest semantics in the chain: the
// text encoder and the DiT are written against exactly this, so a change lands as
// a new symbol rather than an edit. Its consumers are `MediaH3Ref2VaEncode`
// (core/models/text_encoder_32b.*) and the 32B tower that reads the layout it
// emits.
//
// The upstream template is the MiniMax-H3 "ref2va" prompt format:
//
//   <Picture 1> <Audio 1> <Video 1> <T.T seconds> ... <prompt text>
//
// One placeholder token is emitted per visual *patch-block*, not per reference;
// the ViT patch grid decides how many. `tags` marks, per emitted token id,
// whether that position is a visual token (tag 0) or a text token (tag 1) so the
// DiT can re-inject the right modality conditioning each step. The two
// delimiters `<|vision_start|>` / `<|vision_end|>` are tagged *visual* as well,
// because the reference widens a vision span by one token on each side when it
// derives those tags (`token_tags_from_embeds_info`).
#pragma once

#include <string>
#include <vector>

#include "util/media_common.hpp"

namespace phi::media {

// ── modality tags ──────────────────────────────────────────────────────────
//
// Kept as plain i32 in the structs (the plan's ABI) so the tag column can be
// uploaded straight to the device without a conversion pass. The enum is only a
// readability aid at call sites.
enum class Modality : i32 {
	Visual = 0,   // image / video patch token
	Text = 1,     // text token
};

// One preprocessed visual reference (an image, or one pair of video frames).
//
// `patches` is row-major [n_patches, 3 * patch * patch], already scaled to the
// ViT's input range and laid out the way the patch-embed conv expects. The
// consumer only reads `patches`; `grid_*`/`t_frames` exist so the positional
// encoding and the temporal axis can be rebuilt without re-deriving them.
struct VisionBlock {
	std::vector<float> patches;
	i64 n_patches = 0;
	i64 grid_h = 0;
	i64 grid_w = 0;
	i64 t_frames = 1;   // 1 for a still image, >1 for a video reference
	// True when `t_frames` frames fill *one* temporal patch instead of one patch
	// per frame. The MiniMax H3 reference-video presentation samples at 2 fps and
	// packs each pair of frames into a single temporal patch (the reference's
	// `process_video_block`, grid_thw = [1, grid_h, grid_w]), which is what the
	// released checkpoint was trained on: the Qwen3-VL ViT's temporal patch is 2
	// frames, so a pair contributes `grid_h * grid_w` merged tokens - not
	// `2 * grid_h * grid_w`. `patches` still holds one 32x32 patch block per frame
	// (frame index outermost), so `n_patches` stays the buffer's row count while
	// `tokens()` is what the prompt and the DiT count.
	bool temporal_pack = false;

	// The merged-token count this block contributes to the prompt.
	i64 tokens() const {
		if (n_patches <= 0) return 0;
		if (temporal_pack && t_frames > 1) return n_patches / t_frames;
		return n_patches;
	}
};

// A reference clip's frames, at the DiT's own rate (24 fps), already resized to
// the canvas the video VAE encodes and the text tower samples from. `rgb` is
// tightly packed HWC: canvas_w * canvas_h * 3 bytes per frame.
struct RefVideoFrames {
	i64 canvas_w = 0, canvas_h = 0, frames = 0;
	std::vector<unsigned char> rgb;
};

// A reference image's pixels, already resized to the rule's size
// (`ref2va_image_size` in the node layer), RGB HWC 8-bit. The text tower's vision
// run and the video VAE's encode are given the *same* pixels, which is what makes
// every vision slot cover exactly as many latents as the DiT splices in.
struct RefImagePixels {
	i64 w = 0, h = 0;
	std::vector<unsigned char> rgb;
};

// The whole preprocessed prompt.
//
//   ids[i]   = token id fed to the tokenizer/embedding
//   tags[i]  = Modality::Visual / Modality::Text for ids[i]
//   blocks[k] = the k-th visual reference, in reference order
//   block_at[k] = first index into ids that belongs to blocks[k] (its first
//                 placeholder token). -1 if the block contributes no tokens.
//
// Invariant: every index in block_at is < ids.size() and the visual run is
// contiguous. The `append_ref_*` functions below are responsible for keeping it.
struct PromptTokens {
	std::vector<i32> ids;
	std::vector<i32> tags;
	std::vector<VisionBlock> blocks;
	std::vector<i64> block_at;
};

// Point the prompt tokenizer at *this chain's* three tokenizer files (the token ->
// id table, the BPE merge ranks, the special-token ids). Call before building a
// prompt. Empty strings restore the fallback search. Returns true when a usable
// vocabulary was found, false when the byte fallback will be used.
bool set_text_tokenizer_files(const std::string& vocab, const std::string& merges,
                              const std::string& config);

// "<Picture i>: " followed by the image's vision run. `ordinal` is the 1-based
// `<Picture i>` number; it only counts images.
void append_ref_image_block(PromptTokens& out, i64 ordinal, const RefImagePixels& px);
// "<Audio j>: " on its own: an audio reference reaches the model as a reference
// *latent*, never through the text tower, so its label is all it contributes.
void append_ref_audio_label(PromptTokens& out, i64 ordinal);
// The prompt itself, last in the presentation, with no separator of its own
// (every label above already ends in one).
void append_prompt_text(PromptTokens& out, const std::string& prompt);

// The H3 reference-video presentation, exactly as
// comfy/text_encoders/minimax.py builds it (`MiniMaxH3Tokenizer`):
//
//   "<Video k>: " [ "<T.T seconds>" <vision block(2 frames)> ] ...
//
// with the frames sampled at 2 fps (`range(0, n, 12)` on the 24 fps clip) and the
// timestamp of each pair the mean of its two frames' (i/2.0), an odd trailing
// frame repeat-padded into the last pair. Each block is `temporal_pack`, so its
// placeholder run is `grid_h * grid_w`. `ordinal` is the 1-based `<Video k>`
// number; it only counts videos.
void append_ref_video_blocks(PromptTokens& out, i64 ordinal, const RefVideoFrames& v);

// The reference-image resize rule: every edge is snapped up to the next multiple
// of 32 (ref_image_size = "match"). Exposed so the ViT path and the tests agree
// on the exact grid arithmetic instead of each doing its own ceil.
i64 snap_ref_edge(i64 v);

// The reference-image resize rule of the *released* ref2va node, which is a
// down-only scale rather than a snap:
//
//   match: scale = min(1, sqrt((target_w * target_h) / (w * h)))   (the target is
//          the generation's canvas)
//   max:   scale = min(1, 2048 / min(w, h))                        (the reference
//          pipeline's short edge, for the best identity fidelity)
//
// Both then round each edge to the nearest multiple of 32 (round, not ceil) with
// a floor of 32, and never upscale.
void ref2va_image_size(i64 src_w, i64 src_h, i64 target_w, i64 target_h, bool use_max,
                       i64* out_w, i64* out_h);

}  // namespace phi::media
