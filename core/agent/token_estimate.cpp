#include "agent/token_estimate.hpp"

#include <string>
#include <string_view>

namespace phi {
namespace tokens {

namespace {

// One code point, decoded permissively: the estimator runs on tool output and
// file contents as well as on text Phi itself produced, so a stray byte must
// cost *something* rather than desynchronise the whole scan.
struct Cursor {
	std::string_view text;
	size_t i = 0;

	bool done() const { return i >= text.size(); }

	// Returns the code point and advances past it. Invalid or truncated
	// sequences yield U+FFFD (counted as "other") and consume one byte.
	unsigned next() {
		unsigned char c = (unsigned char)text[i];
		if (c < 0x80) {
			i += 1;
			return c;
		}
		size_t need = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
		if (need == 0 || i + need > text.size()) {
			i += 1;
			return 0xFFFD;
		}
		unsigned cp = need == 2 ? (unsigned)(c & 0x1Fu) : need == 3 ? (unsigned)(c & 0x0Fu) : (unsigned)(c & 0x07u);
		for (size_t k = 1; k < need; k++) {
			unsigned char cc = (unsigned char)text[i + k];
			if ((cc & 0xC0) != 0x80) {
				i += 1;
				return 0xFFFD;
			}
			cp = (cp << 6) | (unsigned)(cc & 0x3Fu);
		}
		i += need;
		return cp;
	}
};

// Structural classes, exactly the ones a byte-level BPE distinguishes.
struct Counts {
	double words = 0;
	double extra_letters = 0;  // letters past kWordKnee
	double long_letters = 0;   // letters past kLongWordKnee
	double digits = 0;
	double spaces = 0;
	double punct = 0;
	double cjk = 0;
	double cjk_punct = 0;
	double other = 0;
};

// A word's first few letters are covered by its word price; past this length a
// word tends to split into a stem plus one or more sub-word pieces.
constexpr size_t kWordKnee = 6;
// Past this length even a common stem plus suffixes will not fit in one or two
// tokens, so the per-letter price goes up again (long identifiers, rare words).
constexpr size_t kLongWordKnee = 10;

// Han, kana, hangul and the CJK compatibility blocks: one token per character
// is the ceiling, and frequent characters merge below it.
bool is_cjk(unsigned cp) {
	return (cp >= 0x4E00 && cp <= 0x9FFF) ||   // CJK unified ideographs
		(cp >= 0x3400 && cp <= 0x4DBF) ||      // extension A
		(cp >= 0xF900 && cp <= 0xFAFF) ||      // compatibility ideographs
		(cp >= 0x3040 && cp <= 0x30FF) ||      // hiragana + katakana
		(cp >= 0x31F0 && cp <= 0x31FF) ||      // katakana phonetic extensions
		(cp >= 0xAC00 && cp <= 0xD7AF) ||      // hangul syllables
		(cp >= 0x1100 && cp <= 0x11FF) ||      // hangul jamo
		(cp >= 0x20000 && cp <= 0x2FA1F);      // supplementary + compatibility supplement
}

// Full-width forms and CJK punctuation: counted apart from ideographs because
// the merge table treats them differently (、。，：； are cheap, brackets are not).
bool is_cjk_punct(unsigned cp) {
	return (cp >= 0x3000 && cp <= 0x303F) ||   // CJK symbols and punctuation
		(cp >= 0xFF00 && cp <= 0xFFEF);        // half-width and full-width forms
}

bool is_letter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

bool is_digit(char c) { return c >= '0' && c <= '9'; }

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

Counts classify(std::string_view text) {
	Counts n;
	Cursor cur{text, 0};
	size_t word_len = 0;

	// Close the word being accumulated, charging it as a unit plus the letters
	// that put it past the knees above.
	auto end_word = [&]() {
		if (word_len == 0) return;
		n.words += 1;
		if (word_len > kWordKnee) n.extra_letters += (double)(word_len - kWordKnee);
		if (word_len > kLongWordKnee) n.long_letters += (double)(word_len - kLongWordKnee);
		word_len = 0;
	};

	while (!cur.done()) {
		unsigned cp = cur.next();
		if (cp < 0x80) {
			char c = (char)cp;
			if (is_letter(c)) {
				word_len++;
				continue;
			}
			end_word();
			if (is_digit(c)) n.digits += 1;
			else if (is_space(c)) n.spaces += 1;
			else n.punct += 1;
		} else {
			end_word();
			if (is_cjk(cp)) n.cjk += 1;
			else if (is_cjk_punct(cp)) n.cjk_punct += 1;
			else n.other += 1;
		}
	}
	end_word();
	return n;
}

int64_t price(const Counts& n) {
	const double total = n.words * kWordRank + n.extra_letters * kExtraLetterRank +
		n.long_letters * kLongLetterRank + n.digits * kDigitRank + n.spaces * kSpaceRank +
		n.punct * kPunctRank + n.cjk * kCjkRank + n.cjk_punct * kCjkPunctRank +
		n.other * kOtherRank;
	// Round to the nearest token rather than flooring: a long history is off by
	// a fraction of a token per call at most, and the estimate is reported to
	// the user as one number rather than accumulated.
	return total <= 0 ? 0 : (int64_t)(total + 0.5);
}

// A picture or a video is billed by vision pricing, not by its payload. The
// alternative — pricing the base64 — would report a single screenshot as tens
// of thousands of tokens, which is further from the truth than any of the
// per-class numbers above.
int64_t media_price(const ContentPart& part) {
	const std::string& mime = part.image_mime;
	bool video = mime.size() >= 6 && mime.compare(0, 6, "video/") == 0;
	return video ? kVideoTokens : kImageTokens;
}

}  // namespace

int64_t estimate_text(std::string_view text) {
	if (text.empty()) return 0;
	return price(classify(text));
}

int64_t estimate_message(const AgentMessage& message) {
	int64_t total = kMessageOverhead;
	// A compaction summary is not a content part: the model sees it as a user
	// turn built from the summary text plus a fixed wrapper
	// (AgentSessionHost::transform_context), so it is priced as text.
	if (message.role == "compactionSummary") {
		total += price(classify(message.summary)) + kBlockOverhead;
	}
	for (const auto& part : message.content) {
		switch (part.type) {
			case ContentPart::Type::Text:
			case ContentPart::Type::Thinking:
				total += price(classify(part.text)) + kBlockOverhead;
				break;
			case ContentPart::Type::ToolCall:
				// The call is sent as name + serialized arguments; both are
				// JSON-shaped, so the text density applies to the payload.
				total += price(classify(part.tool_name));
				total += part.arguments_text.empty()
					? (part.arguments.is_null() ? 0 : price(classify(part.arguments.dump())))
					: price(classify(part.arguments_text));
				total += kBlockOverhead;
				break;
			case ContentPart::Type::Image:
				total += media_price(part) + kBlockOverhead;
				break;
		}
	}
	return total;
}

int64_t estimate_tools_json(std::string_view tools_json) {
	if (tools_json.empty()) return 0;
	return price(classify(tools_json)) + kBlockOverhead;
}

int64_t estimate_request(std::string_view system_prompt, std::string_view tools_json,
	const std::vector<AgentMessage>& messages) {
	int64_t total = 0;
	if (!system_prompt.empty()) total += price(classify(system_prompt)) + kMessageOverhead;
	total += estimate_tools_json(tools_json);
	for (const auto& message : messages) total += estimate_message(message);
	return total;
}

}  // namespace tokens
}  // namespace phi
