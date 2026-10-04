#include "text/breeze_tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

#include "util/base.hpp"
#include "util/json.hpp"

namespace phi::media {

namespace {

// U+2581 LOWER ONE EIGHTH BLOCK, the space marker the normalizer substitutes.
const char kBlock[] = "\xE2\x96\x81";

// A character the vocabulary does not hold falls back to its bytes: HF's
// `byte_fallback` emits `format!("<0x{:02X}>", byte)`, i.e. uppercase hex.
i32 byte_token_id(const std::unordered_map<std::string, i32>& vocab, u8 b) {
	char buf[8];
	std::snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned)b);
	auto it = vocab.find(buf);
	return it == vocab.end() ? -1 : it->second;
}

// Space -> U+2581 (the normalizer's Replace, applied to every byte of the text
// that is not part of an added token).
std::string normalize_spaces(const std::string& s) {
	if (s.find(' ') == std::string::npos) return s;
	std::string out;
	out.reserve(s.size() + 8);
	for (char c : s) {
		if (c == ' ') out += kBlock;
		else out.push_back(c);
	}
	return out;
}

}  // namespace

// ── loading ────────────────────────────────────────────────────────────────

void BreezeTokenizer::load(const std::string& tokenizer_json) {
	token_to_id_.clear();
	id_to_token_.clear();
	merge_rank_.clear();
	added_.clear();
	special_.clear();
	byte_id_.assign(256, -1);
	bos_ = eos_ = pad_ = unk_ = -1;

	std::vector<u8> raw;
	if (!read_file_bytes(tokenizer_json, raw)) {
		throw MediaError("breeze tokenizer: cannot read " + tokenizer_json);
	}
	std::string text((const char*)raw.data(), raw.size());
	std::string err;
	auto parsed = json_parse(text, &err);
	if (!parsed || !parsed->is_object()) {
		throw MediaError("breeze tokenizer: " + tokenizer_json + " is not a JSON object" +
		                 (err.empty() ? "" : " (" + err + ")"));
	}
	const JsonValue* model = parsed->find("model");
	if (!model || !model->is_object()) {
		throw MediaError("breeze tokenizer: no \"model\" object in " + tokenizer_json);
	}
	const JsonValue* vocab = model->find("vocab");
	if (!vocab || !vocab->is_object()) {
		throw MediaError("breeze tokenizer: no \"model.vocab\" in " + tokenizer_json);
	}

	const auto& entries = vocab->entries();
	token_to_id_.reserve(entries.size() * 2);
	id_to_token_.reserve(entries.size() + 64);
	i64 max_id = -1;
	for (const auto& kv : entries) {
		if (!kv.second.is_number()) continue;
		const i64 id = kv.second.as_int(-1);
		if (id < 0) continue;
		token_to_id_.emplace(kv.first, (i32)id);
		if (id > max_id) max_id = id;
	}
	id_to_token_.assign((size_t)max_id + 1, std::string());
	for (const auto& kv : token_to_id_) id_to_token_[(size_t)kv.second] = kv.first;

	// ── merges ──
	// The field is `[[a, b], ...]` in the current tokenizers dump and the
	// historical `"a b"` string in older ones; both are accepted.
	const JsonValue* merges = model->find("merges");
	if (merges && merges->is_array()) {
		const auto& arr = merges->items();
		merge_rank_.reserve(arr.size() * 2);
		i32 rank = 0;
		for (const JsonValue& m : arr) {
			std::string a, b;
			if (m.is_array() && m.items().size() >= 2) {
				a = m.items()[0].as_string();
				b = m.items()[1].as_string();
			} else if (m.is_string()) {
				const std::string& s = m.as_string();
				const size_t sp = s.find(' ');
				if (sp == std::string::npos) continue;
				a = s.substr(0, sp);
				b = s.substr(sp + 1);
			} else {
				continue;
			}
			if (a.empty() || b.empty()) continue;
			std::string key = a;
			key.push_back('\x01');
			key += b;
			merge_rank_.emplace(std::move(key), rank++);
		}
	}

	// ── added tokens ──
	const JsonValue* at = parsed->find("added_tokens");
	if (at && at->is_array()) {
		for (const JsonValue& t : at->items()) {
			const JsonValue* content = t.find("content");
			const JsonValue* idv = t.find("id");
			if (!content || !idv) continue;
			const std::string tok = content->as_string();
			if (tok.empty()) continue;
			const i32 id = (i32)idv->as_int(-1);
			if (id < 0) continue;
			added_.emplace_back(tok, id);
			const JsonValue* sp = t.find("special");
			if (sp && sp->as_bool()) special_.insert(id);
			// Keep the id map consistent: added tokens live above the base vocab
			// and must be addressable both ways.
			token_to_id_.emplace(tok, id);
			if ((size_t)id >= id_to_token_.size()) id_to_token_.resize((size_t)id + 1);
			id_to_token_[(size_t)id] = tok;
		}
	}
	std::sort(added_.begin(), added_.end(), [](const auto& a, const auto& b) {
		if (a.first.size() != b.first.size()) return a.first.size() > b.first.size();
		return a.first < b.first;
	});

	for (i32 b = 0; b < 256; b++) byte_id_[(size_t)b] = byte_token_id(token_to_id_, (u8)b);

	bos_ = token_id("<bos>");
	eos_ = token_id("<eos>");
	pad_ = token_id("<pad>");
	unk_ = token_id("<unk>");
	if (id_to_token_.empty()) throw MediaError("breeze tokenizer: empty vocabulary");
}

i32 BreezeTokenizer::token_id(const std::string& token) const {
	auto it = token_to_id_.find(token);
	return it == token_to_id_.end() ? -1 : it->second;
}

i32 BreezeTokenizer::bos_id() const { return bos_; }

std::vector<BreezeTokenizer::AddedHit> BreezeTokenizer::find_added(const std::string& text) const {
	std::vector<AddedHit> out;
	size_t i = 0;
	while (i < text.size()) {
		i32 hit = -1;
		size_t hit_len = 0;
		for (const auto& a : added_) {
			if (a.first.size() > text.size() - i) continue;
			if (std::memcmp(text.data() + i, a.first.data(), a.first.size()) != 0) continue;
			hit = a.second;
			hit_len = a.first.size();
			break;   // longest-first ordering
		}
		if (hit >= 0) {
			out.push_back(AddedHit{(i64)i, hit});
			i += hit_len;
			continue;
		}
		// No match here: skip to the next position that starts one.
		size_t next = text.size();
		for (const auto& a : added_) {
			if (a.first.size() > text.size() - i) continue;
			const size_t p = text.find(a.first, i);
			if (p != std::string::npos && p < next) next = p;
		}
		i = next > i ? next : i + 1;
	}
	return out;
}

// ── BPE ────────────────────────────────────────────────────────────────────

void BreezeTokenizer::bpe_merge_all(std::vector<std::string>& symbols) const {
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
		// Merge every non-overlapping occurrence of that pair, left to right -
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

// One BPE sequence: the symbols are the UTF-8 characters of the (already
// normalized) run, merged by rank, then mapped to ids with the byte fallback for
// anything the vocabulary does not hold.
void BreezeTokenizer::emit_bpe(const std::string& normalized, std::vector<i32>& out) const {
	if (normalized.empty()) return;
	std::vector<std::string> symbols;
	symbols.reserve(normalized.size());
	for (size_t i = 0; i < normalized.size();) {
		const unsigned char c = (unsigned char)normalized[i];
		size_t n = 1;
		if (c >= 0xF0) n = 4;
		else if (c >= 0xE0) n = 3;
		else if (c >= 0xC0) n = 2;
		if (i + n > normalized.size()) n = 1;   // malformed tail: one byte at a time
		symbols.push_back(normalized.substr(i, n));
		i += n;
	}
	bpe_merge_all(symbols);
	for (const std::string& s : symbols) {
		auto it = token_to_id_.find(s);
		if (it != token_to_id_.end()) {
			out.push_back(it->second);
			continue;
		}
		// Byte fallback: the whole symbol becomes its bytes' `<0xXX>` tokens.
		bool any = false;
		for (unsigned char c : s) {
			const i32 id = byte_id_[(size_t)c];
			if (id >= 0) {
				out.push_back(id);
				any = true;
			}
		}
		if (!any && unk_ >= 0) out.push_back(unk_);
	}
}

// ── encode / decode ────────────────────────────────────────────────────────

std::vector<i32> BreezeTokenizer::encode(const std::string& text, bool add_special_tokens) const {
	if (id_to_token_.empty()) throw MediaError("breeze tokenizer: not loaded");
	std::vector<i32> out;
	if (add_special_tokens && bos_ >= 0) out.push_back(bos_);

	size_t i = 0;
	while (i < text.size()) {
		// The added vocabulary gets first refusal, longest match first - that is
		// how <ins_bos> / [S0] / <|AUDIO|> survive as single tokens.
		i32 hit = -1;
		size_t hit_len = 0;
		for (const auto& a : added_) {
			if (a.first.size() > text.size() - i) continue;
			if (std::memcmp(text.data() + i, a.first.data(), a.first.size()) != 0) continue;
			hit = a.second;
			hit_len = a.first.size();
			break;
		}
		if (hit >= 0) {
			out.push_back(hit);
			i += hit_len;
			continue;
		}
		size_t next = text.size();
		for (const auto& a : added_) {
			if (a.first.size() > text.size() - i) continue;
			const size_t p = text.find(a.first, i);
			if (p != std::string::npos && p < next) next = p;
		}
		emit_bpe(normalize_spaces(text.substr(i, next - i)), out);
		i = next > i ? next : i + 1;
	}
	return out;
}

std::string BreezeTokenizer::decode(const std::vector<i32>& ids, bool skip_special) const {
	std::string out;
	for (i32 id : ids) {
		if (skip_special && special_.count(id)) continue;
		if (id < 0 || (size_t)id >= id_to_token_.size()) continue;
		const std::string& tok = id_to_token_[(size_t)id];
		// ByteFallback: "<0xXX>" is one raw byte, not text.
		if (tok.size() == 6 && tok.compare(0, 3, "<0x") == 0 && tok[5] == '>') {
			auto hex = [](char c) -> int {
				if (c >= '0' && c <= '9') return c - '0';
				if (c >= 'A' && c <= 'F') return c - 'A' + 10;
				if (c >= 'a' && c <= 'f') return c - 'a' + 10;
				return -1;
			};
			const int hi = hex(tok[3]), lo = hex(tok[4]);
			if (hi >= 0 && lo >= 0) {
				out.push_back((char)((hi << 4) | lo));
				continue;
			}
		}
		// Replace("▁" -> " ")
		size_t p = 0;
		while (p < tok.size()) {
			if (tok.compare(p, sizeof(kBlock) - 1, kBlock) == 0) {
				out.push_back(' ');
				p += sizeof(kBlock) - 1;
			} else {
				out.push_back(tok[p]);
				p++;
			}
		}
	}
	return out;
}

}  // namespace phi::media
