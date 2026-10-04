// vision_prep — the MiniMax-H3 "ref2va" reference presentation.
//
// This is one half of ComfyUI's `MiniMaxH3ReferenceToVideo`: the *presentation*
// (labels, vision runs, timestamps, tags) that comfy/text_encoders/minimax.py's
// `MiniMaxH3Tokenizer` builds, kept apart from the reference *latents* (a VAE
// encode, which needs the GPU and therefore lives in the node -
// `MediaH3Ref2VaEncode`). The two are one contract: the released checkpoint was
// trained on this exact token stream, so a change here is a change to the chain.
//
// The presentation order, transcribed from the reference node's `ref_items`:
//
//    <Picture 1>: <vision run>            one per reference image, in order
//    <Video 1>:   <T.T seconds> <vision run(2 frames)> ...   one per clip
//    <Audio 1>:                           label only (the latent carries it)
//    <prompt>                             last, with no separator of its own
//
// with a soundtrack's `<Audio j>` label emitted *before* its `<Video k>`, which
// is the order the DiT packs that block in (audio rows first). Ordinals are
// 1-based per type.
//
// The API is a set of `append_ref_*` builders rather than one "prep everything"
// call, because the caller has to interleave them with the VAE work: it decodes
// and resizes each reference once and hands the *same* pixels to the text tower
// and to the VAE (a vision slot covers exactly as many latents as the DiT
// splices in - `RefImagePixels` / `RefVideoFrames` are that shared payload).
//
// Invariants held by the builders:
//   ids.size() == tags.size();  every tag is 0 or 1;
//   every block_at is in [0, ids.size()) and strictly ascending;
//   the Visual run of a block starts exactly at block_at[k] and is contiguous;
//   patches.size() == n_patches * 3*32*32.
// ============================================================================

#include "text/vision_prep.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "io/image_io.hpp"
#include "text/tokenizer.hpp"
#include "io/video_io.hpp"

namespace phi::media {

i64 snap_ref_edge(i64 v) {
	// Pure arithmetic: snap up to the next multiple of 32. Implemented in the
	// freeze so the ViT grid maths is fixed before either side is written.
	return ((v + 31) / 32) * 32;
}

namespace {

// ── scheme constants ──────────────────────────────────────────────────────

constexpr i64 kBlockPx = 32;      // emitted patch-block edge in pixels (2x2 ViT patches)
constexpr i64 kViTPatchPx = 16;   // the ViT's own patch edge, documented for W5
constexpr i64 kBlockPlane = kBlockPx * kBlockPx;          // 1024
constexpr i64 kBlockFloats = 3 * kBlockPx * kBlockPx;     // 3072 floats / block
constexpr i64 kVideoFrames = 8;   // frames sampled per video reference

// Canonical Qwen2.5-VL special-token ids, used only when the loaded vocabulary
// does not carry the token itself (it does for the shipped qwen25 vocab).
constexpr i32 kIdVisionStart = 151652;
constexpr i32 kIdVisionEnd = 151653;
constexpr i32 kIdImagePad = 151655;
constexpr i32 kIdVideoPad = 151656;

static_assert(kViTPatchPx * 2 == kBlockPx, "one emitted block is a 2x2 merge of ViT patches");

// ── tokenizer access ──────────────────────────────────────────────────────

Qwen2Tokenizer& tokenizer_slot() {
	static Qwen2Tokenizer t;
	return t;
}

// The three files the prompt tokenizer reads, set by the chain (see
// `set_text_tokenizer_files`); empty until a chain names them.
struct TextTokenizerFiles {
	std::string vocab, merges, config;
	bool operator==(const TextTokenizerFiles& o) const {
		return vocab == o.vocab && merges == o.merges && config == o.config;
	}
};
TextTokenizerFiles& text_tokenizer_files() {
	static TextTokenizerFiles f;
	return f;
}

// Loads the Qwen2.5 vocabulary once per process; nullptr when unavailable.
// Never throws: a missing/malformed vocabulary degrades to the byte fallback.
//
// The chain's own three files (set by `set_text_tokenizer_files`) are the only
// source. There is deliberately no "shared vocabulary" fallback any more: each
// chain names its own, so a missing one is a configuration error to report rather
// than a reason to silently tokenise another model's prompt.
const Qwen2Tokenizer* tokenizer_or_null() {
	static const Qwen2Tokenizer* cached = nullptr;
	static TextTokenizerFiles cached_files;
	static bool cached_set = false;
	const TextTokenizerFiles f = text_tokenizer_files();
	if (cached_set && f == cached_files) return cached;
	cached_set = true;
	cached_files = f;
	cached = nullptr;
	if (f.vocab.empty() || f.merges.empty()) return nullptr;
	Qwen2Tokenizer& t = tokenizer_slot();
	try {
		t.load(f.vocab, f.merges, f.config);
	} catch (const std::exception&) {
		return nullptr;
	}
	if (!t.loaded()) return nullptr;
	cached = &t;
	return cached;
}

std::vector<i32> encode_text(const std::string& s) {
	if (const Qwen2Tokenizer* t = tokenizer_or_null()) return t->encode(s);
	// Deterministic byte-level fallback: the structure is what matters.
	std::vector<i32> ids;
	ids.reserve(s.size());
	for (unsigned char c : s) ids.push_back((i32)c);
	return ids;
}

i32 special_id(const char* token, i32 fallback) {
	if (const Qwen2Tokenizer* t = tokenizer_or_null()) {
		const i32 id = t->token_id(token);
		if (id >= 0) return id;
	}
	return fallback;
}

// ── token emission ────────────────────────────────────────────────────────

void push_tagged(PromptTokens& out, i32 id, Modality m) {
	out.ids.push_back(id);
	out.tags.push_back((i32)m);
}

void push_text(PromptTokens& out, const std::string& s) {
	for (i32 id : encode_text(s)) push_tagged(out, id, Modality::Text);
}

// Appends a block and its placeholder run. block_at = first Visual index.
//
// The two delimiter tokens are Visual as well, not Text: the reference widens a
// vision span by one token on each side when it derives the DiT's modality tags
// (`token_tags_from_embeds_info`: `tags[index - 1 : index + size + 1] = 0`), so
// `<|vision_start|>`, the placeholders and `<|vision_end|>` are one video-modality
// run and only the *labels* between the runs are text. Tagging them Text made the
// DiT modulate those two rows as text and mis-colour the block's boundaries.
// `block_at` still names the first *placeholder*, which is what the splice into
// the ViT's output needs.
void emit_visual_run(PromptTokens& out, VisionBlock&& blk, i32 pad_id) {
	out.blocks.push_back(std::move(blk));
	const VisionBlock& b = out.blocks.back();
	// `tokens()`, not `n_patches`: a temporally packed block holds two frames'
	// patch rows but contributes one merged token per 32x32 block.
	const i64 n = b.tokens();
	if (n <= 0) {  // no token belongs to this block
		out.block_at.push_back(-1);
		return;
	}
	push_tagged(out, special_id("<|vision_start|>", kIdVisionStart), Modality::Visual);
	out.block_at.push_back((i64)out.ids.size());
	for (i64 p = 0; p < n; ++p) push_tagged(out, pad_id, Modality::Visual);
	push_tagged(out, special_id("<|vision_end|>", kIdVisionEnd), Modality::Visual);
}

// One 32x32 patch block per frame, packed the way the ViT's patch-embed conv
// wants a *temporal* patch: row `p` of the block holds [3, frames, 32, 32] read
// from the frames in order (the reference's `process_video_block`).
void patchify_pair(const RefVideoFrames& v, i64 f0, i64 f1, i64 tw, i64 th, i64 gw, i64 gh,
                   std::vector<float>* out) {
	// Always two temporal slots: an odd trailing frame is *repeat-padded* into the
	// last pair (the reference does `frames = cat([frames, frames[-1:]])`), so the
	// pair whose two indices are equal still fills a full temporal patch - which is
	// what keeps one placeholder run per pair, and the patch buffer's row count
	// equal to `n_patches`.
	const i64 frames = 2;
	out->assign((size_t)(gw * gh * frames * kBlockFloats), 0.0f);
	for (i64 fi = 0; fi < frames; ++fi) {
		const i64 f = fi == 0 ? f0 : std::max<i64>(f0, f1);
		const unsigned char* frame = v.rgb.data() + (size_t)f * (size_t)tw * (size_t)th * 3;
		for (i64 gy = 0; gy < gh; ++gy)
			for (i64 gx = 0; gx < gw; ++gx) {
				float* blk = out->data() + (size_t)(fi * gw * gh + gy * gw + gx) * kBlockFloats;
				for (i64 c = 0; c < 3; ++c)
					for (i64 py = 0; py < kBlockPx; ++py)
						for (i64 px = 0; px < kBlockPx; ++px) {
							const i64 sx = gx * kBlockPx + px;
							const i64 sy = gy * kBlockPx + py;
							const unsigned ch =
							    frame[((size_t)sy * (size_t)tw + (size_t)sx) * 3 + (size_t)c];
							blk[c * kBlockPlane + py * kBlockPx + px] =
							    2.0f * ((float)ch / 255.0f) - 1.0f;
						}
			}
	}
}

// ── image -> patch block ──────────────────────────────────────────────────

// Snaps both edges to a multiple of 32, resamples with the Lanczos-3 kernel the
// image path uses, then cuts the result into 32x32x3 channel-planar blocks.
// `dst` must hold gw*gh*kBlockFloats floats; `dst` is NOT resized here so the
// video path can stack frames into one buffer.
void patchify_into(const unsigned char* rgb, i64 w, i64 h, std::vector<unsigned char>& scratch,
                   i64 tw, i64 th, i64 gw, i64 gh, float* dst) {
	resize_lanczos3(rgb, (int)w, (int)h, (int)tw, (int)th, scratch);
	const unsigned char* src = scratch.data();
	for (i64 gy = 0; gy < gh; ++gy)
		for (i64 gx = 0; gx < gw; ++gx) {
			float* blk = dst + ((gy * gw) + gx) * kBlockFloats;
			for (i64 c = 0; c < 3; ++c)
				for (i64 py = 0; py < kBlockPx; ++py)
					for (i64 px = 0; px < kBlockPx; ++px) {
						const i64 sx = gx * kBlockPx + px;
						const i64 sy = gy * kBlockPx + py;
						const unsigned v = src[((size_t)sy * (size_t)tw + (size_t)sx) * 3 + (size_t)c];
						blk[c * kBlockPlane + py * kBlockPx + px] = 2.0f * ((float)v / 255.0f) - 1.0f;
					}
		}
}


}  // namespace

// Points the prompt tokenizer at a directory - the video chain's own tokenizer
// role. Called by `MediaH3Ref2VaEncode` before it builds the presentation, so the
// ids it emits come from the same vocabulary the text tower was loaded with.
// An empty `dir` leaves the search to the environment / legacy path. Returns
// whether a usable vocabulary was found.
bool set_text_tokenizer_files(const std::string& vocab, const std::string& merges,
                              const std::string& config) {
	TextTokenizerFiles& f = text_tokenizer_files();
	f.vocab = vocab;
	f.merges = merges;
	f.config = config;
	return tokenizer_or_null() != nullptr;
}

void append_ref_image_block(PromptTokens& out, i64 ordinal, const RefImagePixels& px) {
	const i32 image_pad = special_id("<|image_pad|>", kIdImagePad);
	if (px.w <= 0 || px.h <= 0 || px.rgb.empty()) return;
	const i64 tw = snap_ref_edge(px.w), th = snap_ref_edge(px.h);
	const i64 gw = tw / kBlockPx, gh = th / kBlockPx;
	VisionBlock blk;
	blk.grid_w = gw;
	blk.grid_h = gh;
	blk.t_frames = 1;
	blk.n_patches = gw * gh;
	blk.patches.resize((size_t)(blk.n_patches * kBlockFloats));
	std::vector<unsigned char> scratch;
	patchify_into(px.rgb.data(), px.w, px.h, scratch, tw, th, gw, gh, blk.patches.data());
	push_text(out, "<Picture " + std::to_string(ordinal) + ">: ");
	emit_visual_run(out, std::move(blk), image_pad);
}

void append_ref_audio_label(PromptTokens& out, i64 ordinal) {
	push_text(out, "<Audio " + std::to_string(ordinal) + ": ");
}

void append_prompt_text(PromptTokens& out, const std::string& prompt) {
	if (!prompt.empty()) push_text(out, prompt);
}

void append_ref_video_blocks(PromptTokens& out, i64 ordinal, const RefVideoFrames& v) {
	const i32 video_pad = special_id("<|video_pad|>", kIdVideoPad);
	if (v.canvas_w <= 0 || v.canvas_h <= 0 || v.frames <= 0) return;
	// 2 fps over the 24 fps clip, exactly the reference's `range(0, n, FPS // 2)`;
	// an odd trailing frame is repeat-padded so every temporal patch holds two.
	std::vector<i64> idx;
	for (i64 i = 0; i < v.frames; i += 12) idx.push_back(i);
	if (idx.empty()) idx.push_back(0);
	const bool odd = (idx.size() % 2) != 0;
	if (odd) idx.push_back(idx.back());

	const i64 gw = snap_ref_edge(v.canvas_w) / kBlockPx;
	const i64 gh = snap_ref_edge(v.canvas_h) / kBlockPx;
	push_text(out, "<Video " + std::to_string(ordinal) + ">: ");
	for (size_t i = 0; i + 1 < idx.size(); i += 2) {
		// the pair's timestamp is the mean of its two frames' (i / 2.0 seconds),
		// written with one decimal: "<%.1f seconds>"
		const double ts = ((double)i / 2.0 + (double)(i + 1) / 2.0) / 2.0;
		char buf[48];
		snprintf(buf, sizeof buf, "<%.1f seconds>", ts);
		push_text(out, buf);
		VisionBlock blk;
		blk.grid_w = gw;
		blk.grid_h = gh;
		blk.t_frames = 2;
		blk.temporal_pack = true;
		blk.n_patches = gw * gh * 2;
		patchify_pair(v, idx[i], idx[i + 1], v.canvas_w, v.canvas_h, gw, gh, &blk.patches);
		emit_visual_run(out, std::move(blk), video_pad);
	}
}

void ref2va_image_size(i64 src_w, i64 src_h, i64 target_w, i64 target_h, bool use_max,
                       i64* out_w, i64* out_h) {
	const double area = (double)std::max<i64>(target_w, 1) * (double)std::max<i64>(target_h, 1);
	const double scale =
	    use_max ? std::min(1.0, 2048.0 / (double)std::min(src_w, src_h))
	            : std::min(1.0, std::sqrt(area / ((double)src_w * (double)src_h)));
	*out_w = std::max<i64>(32, (i64)std::llround((double)src_w * scale / 32.0) * 32);
	*out_h = std::max<i64>(32, (i64)std::llround((double)src_h * scale / 32.0) * 32);
}

}  // namespace phi::media
