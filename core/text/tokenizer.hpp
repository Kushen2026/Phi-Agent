// Qwen2 byte-level BPE tokenizer.
//
// Both text encoders (Qwen3-VL-8B for the image path, Qwen3-VL-32B for the video
// path) use the same Qwen2.5 vocabulary: 151,643 merge-table tokens plus 26
// added tokens, the GPT-2 byte alphabet, and a leftmost-first pre-tokenisation
// regex. The regex is reproduced by hand (see unicode_cat.*) because there is no
// third-party regex engine here and <regex> has no \p{L}.
//
// Everything the pipelines need is expressed as: encode(text) -> ids,
// decode(ids) -> text (round-trips byte-for-byte), and the chat-template
// helpers.
#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "util/media_common.hpp"

namespace phi::media {

// The three files a Qwen2 tokenizer is built from, as one value. They are
// independent *selections* - the vocab table, the BPE merge ranks and the
// special-token ids - so nothing is derived from anything else, and the same
// struct is what the settings roles resolve to and what the loader receives.
struct TokenizerFiles {
	std::string vocab;    // token -> id table
	std::string merges;   // BPE merge ranks
	std::string config;   // special-token ids (optional)
	bool complete() const { return !vocab.empty() && !merges.empty(); }
};

class Qwen2Tokenizer {
public:
	// vocab.json + merges.txt are the two files ComfyUI ships; tokenizer_config
	// is optional and only supplies the added tokens. Throws MediaError on a
	// malformed or missing file.
	void load(const std::string& vocab_json, const std::string& merges_txt,
	          const std::string& tokenizer_config_json = std::string());

	bool loaded() const { return !id_to_token_.empty(); }
	size_t vocab_size() const { return id_to_token_.size(); }
	size_t merge_count() const { return merge_rank_.size(); }
	size_t added_count() const { return added_.size(); }

	// -1 when the token is not in the vocabulary.
	i32 token_id(const std::string& token) const;
	const std::string& token_text(i32 id) const;

	// `allow_special` controls whether added/special tokens embedded in the text
	// are recognised as single tokens (the reference tokenizer's default).
	std::vector<i32> encode(const std::string& text, bool allow_special = true) const;
	std::string decode(const std::vector<i32>& ids) const;

	// The pre-tokenisation step on its own, exposed because it is the part most
	// likely to drift from the reference and the conformance test compares it
	// directly. Returns raw (not byte-mapped) pre-token substrings.
	std::vector<std::string> pre_tokenize(const std::string& text) const;

	// ── chat templates ─────────────────────────────────────────────────────
	//
	// A plain single-turn prompt with no system turn and the generation prompt
	// enabled (the shape a text-to-image workflow's encoder expects).
	std::string chat_prompt_no_system(const std::string& user) const;
	// Qwen2.5's default template with the standard system turn.
	std::string chat_prompt_with_system(const std::string& user) const;

	// ── Qwen-Image-2.1 ───────────────────────────────────────────────────
	//
	// comfy/text_encoders/qwen_image21.py's T2I_TEMPLATE. With reference images
	// the template grows a block per image at the head of the user turn,
	// `"<image1><|vision_start|><|image_pad|><|vision_end|> "` (QwenImage21Tokenizer
	// builds it: `" ".join("<image{}>{}".format(i + 1, VISION_BLOCK))`) and the
	// tower's vision rows are removed from the encoder output afterwards; the
	// DiT puts a reference *latent block* in their place.
	std::string qwen_image_prompt(const std::string& user, int n_refs = 0) const;

	// The prompt as a run of token ids plus the positions the reference blocks
	// take in it. `slots` receives, per reference, how many kept tokens precede
	// its block - i.e. exactly the `image_slots` that
	// comfy/text_encoders/qwen_image21.py's encode_token_weights hands the DiT.
	struct QwenImageIds {
		std::vector<i32> all;            // the whole template, exactly as encoded
		std::vector<i32> ids;            // the kept tokens, vision rows removed
		std::vector<i32> slots;          // one entry per reference image
		i64 drop_rows = 0;               // leading rows this shortened (the system turn)
	};
	QwenImageIds encode_qwen_image_refs(const std::string& user, int n_refs) const;

	// The *whole* template, system turn included, exactly what the text encoder
	// is run over.
	std::vector<i32> encode_qwen_image(const std::string& user, i64* drop_rows = nullptr) const;
	// How many leading rows of that encoding's output the DiT keeps in its
	// sequence: `encode_token_weights` drops everything before the *second*
	// `<|im_start|>` **after** the encoder has run, so the system turn still
	// attends into every kept token (and still shifts their RoPE positions).
	i64 qwen_image_drop_rows(const std::string& user) const;

	// ── special token ids (Qwen2.5 fixed layout) ───────────────────────────
	i32 id_im_start() const { return id_im_start_; }
	i32 id_im_end() const { return id_im_end_; }
	i32 id_endoftext() const { return id_endoftext_; }
	// Qwen3-VL's vision marker: one per reference block, inside the user turn.
	i32 id_image_pad() const { return id_image_pad_; }

	// Byte-level alphabet, exposed so the video tokenizer can assemble vision
	// blocks out of raw ids without round-tripping through text.
	static const std::array<u32, 256>& byte_to_codepoint();
	static const std::unordered_map<u32, u8>& codepoint_to_byte();

private:
	// Merges every occurrence of the lowest-ranked adjacent pair until no pair
	// is in the merge table. `symbols` holds byte-level UTF-8 pieces.
	void bpe_merge_all(std::vector<std::string>& symbols) const;

	std::unordered_map<std::string, i32> token_to_id_;
	std::vector<std::string> id_to_token_;
	std::unordered_map<std::string, i32> merge_rank_;  // "a\x01b" -> rank
	// Added tokens sorted longest-first so the scan takes the longest match at
	// each position, like the reference added-vocabulary split.
	std::vector<std::pair<std::string, i32>> added_;
	size_t longest_added_ = 0;
	i32 id_im_start_ = -1;
	i32 id_im_end_ = -1;
	i32 id_endoftext_ = -1;
	i32 id_image_pad_ = -1;
};

}  // namespace phi::media
