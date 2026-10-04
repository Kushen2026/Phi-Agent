// Breeze-TTS-2 tokenizer (HuggingFace fast-tokenizer dump, tokenizer.json).
//
// The model's prompt is a plain string that mixes ordinary text with the
// checkpoint's inline control tokens (`<|AUDIO|>`, `<|audio_eos|>`, `<ins_bos>`,
// `<ins_eos>`, `[S0]`..`[S9]`), so the two things that have to be exact are the
// added-vocabulary pass and the BPE itself. Everything else about this
// vocabulary is deliberately simple:
//
//   * the alphabet is *not* the GPT-2 byte alphabet. The tokens are stored as
//     real UTF-8 ("▁Hello" is five codepoints), the way the SentencePiece
//     lineage keeps them, and a character the vocabulary does not hold falls
//     back to the 256 `<0xXX>` byte tokens (`byte_fallback: true`);
//   * the normalizer is `Replace(" " -> "▁")`, so every space becomes U+2581
//     *before* pre-tokenisation;
//   * the pre-tokeniser is `Split(" ")`, whose pattern cannot match after that
//     replacement - so the pre-tokenisation is a no-op and each run between two
//     added tokens is one BPE sequence (verified against `tokenizers` 0.23 on
//     mixed Chinese/English/emoji/control-token input: encode() agrees with the
//     reference for every case tried, and decode() round-trips);
//   * the post-processor prepends `<bos>` (id 2) when `add_special_tokens` is
//     set, which is what the reference's `tokenizer(text, add_special_tokens=True)`
//     does.
//
// `decode` reproduces the checkpoint's decoder chain
// [Replace("▁"->" "), ByteFallback, Fuse]: a `<0xXX>` token becomes its raw
// byte, everything else is appended with U+2581 turned back into a space. The
// Fuse stage only glues the byte tokens back into valid UTF-8, which plain
// concatenation already does.
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "util/media_common.hpp"

namespace phi::media {

// Breeze-TTS-2's tokenizer.json (a HuggingFace BPE: "▁" normalizer +
// Split(" ") merged-with-previous pre-tokeniser + ByteFallback decoder), with
// the 6428 added/special tokens (<|AUDIO|>, <|audio_eos|>, <ins_bos>,
// <ins_eos>, [S0]..).
class BreezeTokenizer {
public:
	// Throws MediaError on a missing or malformed file.
	void load(const std::string& tokenizer_json);

	bool loaded() const { return !id_to_token_.empty(); }
	size_t vocab_size() const { return id_to_token_.size(); }
	size_t added_count() const { return added_.size(); }

	std::vector<i32> encode(const std::string& text, bool add_special_tokens) const;
	std::string decode(const std::vector<i32>& ids, bool skip_special) const;
	i32 token_id(const std::string& token) const;   // -1 when absent
	i32 bos_id() const;

	// The added-vocabulary pass on its own: the tokens that encode() will pull
	// out of `text` before any BPE runs, in order. Exposed because this is the
	// step whose details (leftmost match, longest at that position, matched
	// against the *unnormalised* text) decide whether `<ins_bos>` survives as one
	// token or is spelled out by the BPE.
	struct AddedHit {
		i64 pos = 0;
		i32 id = -1;
	};
	std::vector<AddedHit> find_added(const std::string& text) const;

private:
	// Merges every occurrence of the lowest-ranked adjacent pair until no pair is
	// in the merge table (the reference's single-pass-per-rank loop).
	void bpe_merge_all(std::vector<std::string>& symbols) const;
	void emit_bpe(const std::string& normalized, std::vector<i32>& out) const;

	std::unordered_map<std::string, i32> token_to_id_;
	std::vector<std::string> id_to_token_;
	std::unordered_map<std::string, i32> merge_rank_;  // "a\x01b" -> rank
	// Added tokens sorted longest-first, so the scan takes the longest match at
	// each position, exactly like the reference added-vocabulary split.
	std::vector<std::pair<std::string, i32>> added_;
	std::unordered_set<i32> special_;
	std::vector<i32> byte_id_;   // 256 entries: the `<0xXX>` token of each byte
	i32 bos_ = -1, eos_ = -1, pad_ = -1, unk_ = -1;
};

}  // namespace phi::media
