#include "text/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

#include "util/base.hpp"
#include "util/json.hpp"
#include "text/unicode_cat.hpp"

namespace phi::media {

namespace {

// ── UTF-8 ──────────────────────────────────────────────────────────────────

size_t utf8_decode(const std::string& s, size_t i, u32* cp) {
	const u8 c = (u8)s[i];
	if (c < 0x80) {
		*cp = c;
		return 1;
	}
	size_t need;
	u32 v;
	if ((c & 0xE0) == 0xC0) {
		need = 1;
		v = c & 0x1F;
	} else if ((c & 0xF0) == 0xE0) {
		need = 2;
		v = c & 0x0F;
	} else if ((c & 0xF8) == 0xF0) {
		need = 3;
		v = c & 0x07;
	} else {
		// Invalid lead byte: treat it as a single raw byte so nothing is lost
		// (the byte alphabet covers every possible byte, so the tokenizer can
		// still round-trip arbitrary input).
		*cp = 0x110000u + c;
		return 1;
	}
	if (i + need >= s.size()) {
		*cp = 0x110000u + c;
		return 1;
	}
	for (size_t k = 1; k <= need; k++) {
		u8 cc = (u8)s[i + k];
		if ((cc & 0xC0) != 0x80) {
			*cp = 0x110000u + c;
			return 1;
		}
		v = (v << 6) | (cc & 0x3F);
	}
	*cp = v;
	return need + 1;
}

void utf8_append(std::string& out, u32 cp) {
	if (cp < 0x80) {
		out.push_back((char)cp);
	} else if (cp < 0x800) {
		out.push_back((char)(0xC0 | (cp >> 6)));
		out.push_back((char)(0x80 | (cp & 0x3F)));
	} else if (cp < 0x10000) {
		out.push_back((char)(0xE0 | (cp >> 12)));
		out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
		out.push_back((char)(0x80 | (cp & 0x3F)));
	} else {
		out.push_back((char)(0xF0 | (cp >> 18)));
		out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
		out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
		out.push_back((char)(0x80 | (cp & 0x3F)));
	}
}

// Decoded view of a string: codepoints plus the byte offset each starts at, so
// the pre-tokenizer can slice the original bytes exactly.
struct CpView {
	std::vector<u32> cp;
	std::vector<size_t> off;  // off[i] = byte offset of cp[i]; off[n] = size
};

CpView decode_all(const std::string& s) {
	CpView v;
	size_t i = 0;
	while (i < s.size()) {
		u32 c = 0;
		size_t n = utf8_decode(s, i, &c);
		v.cp.push_back(c);
		v.off.push_back(i);
		i += n;
	}
	v.off.push_back(s.size());
	return v;
}

// ── the Qwen2 pre-tokenisation regex ──────────────────────────────────────
//
//   1  (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   2  [^\r\n\p{L}\p{N}]?\p{L}+
//   3  \p{N}
//   4  ' ?[^\s\p{L}\p{N}]+[\r\n]*'      (the leading space is a literal 0x20)
//   5  \s*[\r\n]+
//   6  \s+(?!\S)
//   7  \s+
//
// Python's re picks the FIRST alternative that matches at a position (not the
// longest), and only then applies greedy quantifiers inside it, so the order
// below matters.

bool is_newline(u32 c) { return c == '\n' || c == '\r'; }

// alternative 1
size_t try_contraction(const CpView& v, size_t i) {
	size_t n = v.cp.size();
	if (i + 1 >= n || v.cp[i] != '\'') return i;
	u32 a = v.cp[i + 1] | 0x20;  // ASCII-fold the letter (all candidates are ASCII)
	switch (a) {
		case 's':
		case 't':
		case 'm':
		case 'd':
			return i + 2;
		case 'r':
			if (i + 2 < n && (v.cp[i + 2] | 0x20) == 'e') return i + 3;
			return i;
		case 'v':
			if (i + 2 < n && (v.cp[i + 2] | 0x20) == 'e') return i + 3;
			return i;
		case 'l':
			if (i + 2 < n && (v.cp[i + 2] | 0x20) == 'l') return i + 3;
			return i;
		default:
			return i;
	}
}

// alternative 2
size_t try_letter_run(const CpView& v, size_t i) {
	size_t n = v.cp.size();
	size_t start = i;
	if (!(i < n && unicode_is_letter(v.cp[i]))) {
		// one optional non-newline, non-letter, non-number codepoint
		if (i + 1 < n && !is_newline(v.cp[i]) && !unicode_is_letter(v.cp[i]) &&
		    !unicode_is_number(v.cp[i]) && unicode_is_letter(v.cp[i + 1])) {
			start = i + 1;
		} else {
			return i;
		}
	}
	size_t j = start;
	while (j < n && unicode_is_letter(v.cp[j])) j++;
	return j > start ? j : i;
}

// alternative 4
size_t try_other_run(const CpView& v, size_t i) {
	size_t n = v.cp.size();
	size_t j = i;
	if (j < n && v.cp[j] == ' ') j++;
	size_t k = j;
	while (k < n && !unicode_is_space(v.cp[k]) && !unicode_is_letter(v.cp[k]) &&
	       !unicode_is_number(v.cp[k])) {
		k++;
	}
	if (k == j) return i;  // needs at least one
	while (k < n && is_newline(v.cp[k])) k++;
	return k;
}

// alternative 5: \s*[\r\n]+  (greedy \s* backtracking to the last \r or \n)
size_t try_ws_newline(const CpView& v, size_t i) {
	size_t n = v.cp.size();
	size_t run = i;
	while (run < n && unicode_is_space(v.cp[run])) run++;
	if (run == i) return i;
	// largest k such that the codepoint after k whitespace chars is \r or \n
	size_t k = run;
	while (k > i && !is_newline(v.cp[k - 1])) k--;
	if (k == i) return i;
	size_t j = k - 1;  // position of the last newline inside the run
	while (j < n && is_newline(v.cp[j])) j++;
	return j;
}

// alternatives 6 and 7
size_t try_trailing_ws(const CpView& v, size_t i, bool lookahead) {
	size_t n = v.cp.size();
	size_t run = i;
	while (run < n && unicode_is_space(v.cp[run])) run++;
	if (run == i) return i;
	if (!lookahead) return run;
	// \s+(?!\S): the match must not be followed by a non-whitespace codepoint
	if (run == n) return run;
	if (unicode_is_space(v.cp[run])) return run;
	return run - 1 > i ? run - 1 : i;
}

size_t match_alternative(const CpView& v, size_t i) {
	size_t r = try_contraction(v, i);
	if (r > i) return r;
	r = try_letter_run(v, i);
	if (r > i) return r;
	if (unicode_is_number(v.cp[i])) return i + 1;  // alternative 3
	r = try_other_run(v, i);
	if (r > i) return r;
	r = try_ws_newline(v, i);
	if (r > i) return r;
	r = try_trailing_ws(v, i, true);
	if (r > i) return r;
	return try_trailing_ws(v, i, false);
}

}  // namespace

// ── byte alphabet ──────────────────────────────────────────────────────────

const std::array<u32, 256>& Qwen2Tokenizer::byte_to_codepoint() {
	// GPT-2's printability remap. The vocab stores every byte as one of these
	// printable codepoints, which is what makes the BPE lossless on any input.
	static const std::array<u32, 256> table = [] {
		std::array<u32, 256> t{};
		int next = 0;
		for (int b = 0; b < 256; b++) {
			bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
			                 (b >= 0xAE && b <= 0xFF);
			t[b] = printable ? (u32)b : (u32)(256 + next++);
		}
		return t;
	}();
	return table;
}

const std::unordered_map<u32, u8>& Qwen2Tokenizer::codepoint_to_byte() {
	static const std::unordered_map<u32, u8> inv = [] {
		std::unordered_map<u32, u8> m;
		const auto& fwd = byte_to_codepoint();
		for (int b = 0; b < 256; b++) m[fwd[b]] = (u8)b;
		return m;
	}();
	return inv;
}

// ── loading ────────────────────────────────────────────────────────────────

void Qwen2Tokenizer::load(const std::string& vocab_json, const std::string& merges_txt,
                          const std::string& tokenizer_config_json) {
	token_to_id_.clear();
	id_to_token_.clear();
	merge_rank_.clear();
	added_.clear();
	longest_added_ = 0;
	id_im_start_ = id_im_end_ = id_endoftext_ = -1;
	id_image_pad_ = -1;

	// ── vocab.json ──
	std::vector<u8> raw;
	if (!read_file_bytes(vocab_json, raw)) {
		throw MediaError("tokenizer: cannot read " + vocab_json);
	}
	std::string text((const char*)raw.data(), raw.size());
	std::string err;
	auto parsed = json_parse(text, &err);
	if (!parsed || !parsed->is_object()) {
		throw MediaError("tokenizer: " + vocab_json + " is not a JSON object" +
		                 (err.empty() ? "" : " (" + err + ")"));
	}
	const auto& entries = parsed->entries();
	token_to_id_.reserve(entries.size() * 2);
	id_to_token_.reserve(entries.size() + 64);
	for (const auto& kv : entries) {
		if (!kv.second.is_number()) continue;
		i32 id = (i32)kv.second.as_int();
		token_to_id_.emplace(kv.first, id);
	}
	i64 max_id = 0;
	for (const auto& kv : token_to_id_)
		if (kv.second > max_id) max_id = kv.second;
	id_to_token_.assign((size_t)max_id + 1, std::string());
	for (const auto& kv : token_to_id_) id_to_token_[(size_t)kv.second] = kv.first;

	// ── merges.txt ──
	std::vector<u8> mraw;
	if (!read_file_bytes(merges_txt, mraw)) {
		throw MediaError("tokenizer: cannot read " + merges_txt);
	}
	{
		std::string s((const char*)mraw.data(), mraw.size());
		size_t pos = 0;
		i32 rank = 0;
		while (pos < s.size()) {
			size_t nl = s.find('\n', pos);
			if (nl == std::string::npos) nl = s.size();
			std::string line = s.substr(pos, nl - pos);
			pos = nl + 1;
			if (!line.empty() && line.back() == '\r') line.pop_back();
			if (line.empty() || line[0] == '#') continue;  // "#version: 0.2"
			size_t sp = line.find(' ');
			if (sp == std::string::npos) continue;
			std::string key = line.substr(0, sp);
			key.push_back('\x01');
			key += line.substr(sp + 1);
			merge_rank_.emplace(std::move(key), rank++);
		}
	}

	// ── added tokens ──
	if (!tokenizer_config_json.empty()) {
		std::vector<u8> craw;
		if (read_file_bytes(tokenizer_config_json, craw)) {
			std::string cs((const char*)craw.data(), craw.size());
			std::string cerr;
			auto cfg = json_parse(cs, &cerr);
			if (cfg && cfg->is_object()) {
				const JsonValue* dec = cfg->find("added_tokens_decoder");
				if (dec && dec->is_object()) {
					for (const auto& kv : dec->entries()) {
						i32 id = (i32)strtol(kv.first.c_str(), nullptr, 10);
						const JsonValue* content = kv.second.find("content");
						if (!content) continue;
						std::string tok = content->as_string();
						if (tok.empty()) continue;
						added_.emplace_back(tok, id);
						// Keep the id map consistent: added tokens live *above*
						// the base vocab, so they must be addressable both ways.
						token_to_id_.emplace(tok, id);
						if ((size_t)id >= id_to_token_.size())
							id_to_token_.resize((size_t)id + 1);
						id_to_token_[(size_t)id] = tok;
					}
				}
			}
		}
	}
	std::sort(added_.begin(), added_.end(), [](const auto& a, const auto& b) {
		if (a.first.size() != b.first.size()) return a.first.size() > b.first.size();
		return a.first < b.first;
	});
	for (const auto& a : added_)
		if (a.first.size() > longest_added_) longest_added_ = a.first.size();

	id_endoftext_ = token_id("<|endoftext|>");
	id_im_start_ = token_id("<|im_start|>");
	id_im_end_ = token_id("<|im_end|>");
	id_image_pad_ = token_id("<|image_pad|>");
}

i32 Qwen2Tokenizer::token_id(const std::string& token) const {
	auto it = token_to_id_.find(token);
	return it == token_to_id_.end() ? -1 : it->second;
}

const std::string& Qwen2Tokenizer::token_text(i32 id) const {
	static const std::string empty;
	if (id < 0 || (size_t)id >= id_to_token_.size()) return empty;
	return id_to_token_[(size_t)id];
}

// ── pre-tokenisation ───────────────────────────────────────────────────────

std::vector<std::string> Qwen2Tokenizer::pre_tokenize(const std::string& text) const {
	std::vector<std::string> out;
	if (text.empty()) return out;
	CpView v = decode_all(text);
	size_t i = 0;
	while (i < v.cp.size()) {
		size_t j = match_alternative(v, i);
		if (j <= i) j = i + 1;  // cannot happen, but never loop forever
		out.push_back(text.substr(v.off[i], v.off[j] - v.off[i]));
		i = j;
	}
	return out;
}

// ── BPE ────────────────────────────────────────────────────────────────────

void Qwen2Tokenizer::bpe_merge_all(std::vector<std::string>& symbols) const {
	while (symbols.size() > 1) {
		i32 best_rank = std::numeric_limits<i32>::max();
		size_t best_i = 0;
		for (size_t i = 0; i + 1 < symbols.size(); i++) {
			std::string key = symbols[i];
			key.push_back('\x01');
			key += symbols[i + 1];
			auto it = merge_rank_.find(key);
			if (it != merge_rank_.end() && it->second < best_rank) {
				best_rank = it->second;
				best_i = i;
			}
		}
		if (best_rank == std::numeric_limits<i32>::max()) break;
		// Merge every non-overlapping occurrence of that pair, left to right —
		// the same single pass the reference implementation performs.
		const std::string& a = symbols[best_i];
		const std::string& b = symbols[best_i + 1];
		std::vector<std::string> merged;
		merged.reserve(symbols.size());
		for (size_t i = 0; i < symbols.size();) {
			if (i + 1 < symbols.size() && symbols[i] == a && symbols[i + 1] == b) {
				merged.push_back(a + b);
				i += 2;
			} else {
				merged.push_back(symbols[i]);
				i += 1;
			}
		}
		symbols.swap(merged);
	}
}

// ── encode ─────────────────────────────────────────────────────────────────

std::vector<i32> Qwen2Tokenizer::encode(const std::string& text, bool allow_special) const {
	if (!loaded()) throw MediaError("tokenizer: not loaded");
	const auto& b2c = byte_to_codepoint();
	std::vector<i32> out;

	size_t i = 0;
	while (i < text.size()) {
		// At each position the added vocabulary gets first refusal, longest
		// match first — that is how <|im_start|> survives pre-tokenisation.
		i32 hit = -1;
		size_t hit_len = 0;
		if (allow_special && !added_.empty() && i + 1 <= text.size()) {
			for (const auto& a : added_) {
				if (a.first.size() > text.size() - i) continue;
				if (memcmp(text.data() + i, a.first.data(), a.first.size()) != 0) continue;
				hit = a.second;
				hit_len = a.first.size();
				break;
			}
		}
		if (hit >= 0) {
			out.push_back(hit);
			i += hit_len;
			continue;
		}
		// Otherwise: find the end of the current pre-token and encode that.
		size_t next = text.size();
		if (allow_special && !added_.empty()) {
			for (const auto& a : added_) {
				if (a.first.size() > text.size() - i) continue;
				size_t p = text.find(a.first, i);
				if (p != std::string::npos && p < next) next = p;
			}
		}
		size_t seg_end = next;
		// Pre-tokenise the segment [i, seg_end) with the regex, on its own, so
		// an added token in the middle never merges into a neighbouring word.
		std::string seg = text.substr(i, seg_end - i);
		if (!seg.empty()) {
			std::vector<std::string> pieces = pre_tokenize(seg);
			for (const std::string& piece : pieces) {
				// byte-map, then BPE on the mapped codepoints
				std::string mapped;
				mapped.reserve(piece.size() * 2);
				for (unsigned char c : piece) utf8_append(mapped, b2c[c]);
				CpView mv = decode_all(mapped);
				std::vector<std::string> symbols;
				symbols.reserve(mv.cp.size());
				for (size_t k = 0; k < mv.cp.size(); k++) {
					std::string one;
					utf8_append(one, mv.cp[k]);
					symbols.push_back(std::move(one));
				}
				bpe_merge_all(symbols);
				for (const std::string& s : symbols) {
					auto it = token_to_id_.find(s);
					if (it != token_to_id_.end()) {
						out.push_back(it->second);
					} else {
						// Cannot happen for well-formed input (the alphabet is
						// complete), but never silently drop bytes.
						for (unsigned char c : s) {
							std::string one;
							utf8_append(one, b2c[c]);
							auto it2 = token_to_id_.find(one);
							if (it2 != token_to_id_.end()) out.push_back(it2->second);
						}
					}
				}
			}
		}
		i = seg_end;
	}
	return out;
}

std::string Qwen2Tokenizer::decode(const std::vector<i32>& ids) const {
	const auto& c2b = codepoint_to_byte();
	std::string out;
	for (i32 id : ids) {
		const std::string& tok = token_text(id);
		// added tokens are stored verbatim and have no byte-level encoding
		bool is_added = false;
		for (const auto& a : added_) {
			if (a.first == tok) {
				is_added = true;
				break;
			}
		}
		if (is_added) {
			out += tok;
			continue;
		}
		CpView v = decode_all(tok);
		for (u32 c : v.cp) {
			auto it = c2b.find(c);
			if (it != c2b.end()) out.push_back((char)it->second);
		}
	}
	return out;
}

// ── chat templates ─────────────────────────────────────────────────────────

std::string Qwen2Tokenizer::chat_prompt_no_system(const std::string& user) const {
	return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
}

std::string Qwen2Tokenizer::qwen_image_prompt(const std::string& user, int n_refs) const {
	// comfy/text_encoders/qwen_image21.py: SYSTEM_PROMPT + T2I_TEMPLATE, with the
	// reference blocks QwenImage21Tokenizer splices into the placeholder when the
	// node hands it images. Each block is a *textual* "<imageN>" marker (four
	// ordinary tokens) followed by the three vision tokens; the tower turns the
	// middle one into the picture's rows.
	std::string user_turn;
	if (n_refs > 0) {
		const std::string block = "<|vision_start|><|image_pad|><|vision_end|>";
		for (int i = 0; i < n_refs; i++) {
			if (i) user_turn += " ";
			user_turn += "<image" + std::to_string(i + 1) + ">" + block;
		}
		// the refs are the head of the user turn, the prompt follows with no
		// separator of its own (the template inserts `refs + "{}"`)
	}
	return "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
	       "<|im_start|>user\n" +
	       user_turn + user +
	       "<|im_end|>\n<|im_start|>assistant\n";
}

std::vector<i32> Qwen2Tokenizer::encode_qwen_image(const std::string& user, i64* drop_rows) const {
	const std::vector<i32> ids = encode(qwen_image_prompt(user));
	// Everything before the second <|im_start|> is dropped *after* the text
	// encoder has run, not before: the system turn is still causal context for
	// the user turn, and it still occupies the first RoPE positions. Callers
	// need the count to slice the encoder's output.
	i32 seen = 0;
	size_t cut = 0;
	for (size_t i = 0; i < ids.size(); i++) {
		if (ids[i] == id_im_start_) {
			seen++;
			if (seen == 2) {
				cut = i;
				break;
			}
		}
	}
	if (drop_rows) *drop_rows = (i64)cut;
	return ids;
}

Qwen2Tokenizer::QwenImageIds Qwen2Tokenizer::encode_qwen_image_refs(const std::string& user,
                                                                  int n_refs) const {
	QwenImageIds out;
	const std::vector<i32> ids = encode(qwen_image_prompt(user, n_refs));
	out.all = ids;
	// The same slice encode_qwen_image reports, applied here.
	i32 seen = 0;
	size_t cut = 0;
	for (size_t i = 0; i < ids.size(); i++) {
		if (ids[i] == id_im_start_) {
			seen++;
			if (seen == 2) {
				cut = i;
				break;
			}
		}
	}
	out.drop_rows = (i64)cut;
	// Walk the kept tail: the picture's rows are exactly its <|image_pad|> token
	// (the tower expands that one token into a whole grid of rows, and
	// encode_token_weights removes them again), and the block's slot is the
	// number of surviving rows before it - `<|vision_start|>` stays, so the
	// block lands between it and `<|vision_end|>`.
	out.ids.reserve(ids.size() - cut);
	for (size_t i = cut; i < ids.size(); i++) {
		if (ids[i] == id_image_pad_ && (i32)out.slots.size() < n_refs) {
			out.slots.push_back((i32)out.ids.size());
			continue;
		}
		out.ids.push_back(ids[i]);
	}
	return out;
}

i64 Qwen2Tokenizer::qwen_image_drop_rows(const std::string& user) const {
	i64 drop = 0;
	encode_qwen_image(user, &drop);
	return drop;
}

std::string Qwen2Tokenizer::chat_prompt_with_system(const std::string& user) const {
	return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
	       "<|im_start|>user\n" +
	       user + "<|im_end|>\n<|im_start|>assistant\n";
}

}  // namespace phi::media
