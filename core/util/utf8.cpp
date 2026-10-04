#include "util/utf8.hpp"

#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace phi {

size_t utf8_sequence_len(std::string_view s, size_t i, unsigned& cp) {
	unsigned char c = (unsigned char)s[i];
	size_t n;
	if (c < 0x80) { cp = c; return 1; }
	if ((c & 0xE0) == 0xC0) { n = 2; cp = c & 0x1Fu; }
	else if ((c & 0xF0) == 0xE0) { n = 3; cp = c & 0x0Fu; }
	else if ((c & 0xF8) == 0xF0) { n = 4; cp = c & 0x07u; }
	else return 0;  // stray continuation byte or invalid lead byte
	if (i + n > s.size()) return 0;  // truncated sequence
	for (size_t k = 1; k < n; k++) {
		unsigned char cc = (unsigned char)s[i + k];
		if ((cc & 0xC0) != 0x80) return 0;  // missing continuation byte
		cp = (cp << 6) | (cc & 0x3Fu);
	}
	// overlong form / surrogate half / beyond U+10FFFF are all invalid
	if ((n == 2 && cp < 0x80u) || (n == 3 && cp < 0x800u) || (n == 4 && cp < 0x10000u)) return 0;
	if (cp >= 0xD800u && cp <= 0xDFFFu) return 0;
	if (cp > 0x10FFFFu) return 0;
	return n;
}

bool utf8_valid(std::string_view s) {
	for (size_t i = 0; i < s.size();) {
		unsigned cp = 0;
		size_t n = utf8_sequence_len(s, i, cp);
		if (n == 0) return false;
		i += n;
	}
	return true;
}

static const char kReplacement[] = "\xEF\xBF\xBD";  // U+FFFD

std::string utf8_sanitize(std::string_view s) {
	// valid input is the common case: return it untouched, no rewriting cost and
	// no risk of altering good data
	if (utf8_valid(s)) return std::string(s);
	std::string out;
	out.reserve(s.size() + 8);
	for (size_t i = 0; i < s.size();) {
		unsigned cp = 0;
		size_t n = utf8_sequence_len(s, i, cp);
		if (n == 0) {
			out += kReplacement;
			i++;  // resync on the next byte (what a decoder's recovery does)
		} else {
			out.append(s, i, n);
			i += n;
		}
	}
	return out;
}

std::string decode_external_text(const std::string& bytes) {
	if (bytes.empty() || utf8_valid(bytes)) return bytes;
#ifdef _WIN32
	// Not UTF-8, so it is most likely the system ANSI code page (GBK/CP936 on
	// Chinese Windows) — the encoding of legacy .txt files and of whatever a
	// console program printed. Latin-1 is deliberately NOT tried: it accepts
	// almost any byte and would "succeed" with mojibake.
	if (bytes.size() <= (size_t)INT_MAX) {
		const UINT code_pages[] = {CP_ACP, 936u /* GBK */};
		for (UINT cp : code_pages) {
			int wlen = MultiByteToWideChar(cp, MB_ERR_INVALID_CHARS,
				bytes.data(), (int)bytes.size(), nullptr, 0);
			if (wlen <= 0) continue;
			std::wstring wide((size_t)wlen, L'\0');
			if (MultiByteToWideChar(cp, MB_ERR_INVALID_CHARS, bytes.data(), (int)bytes.size(),
					wide.data(), wlen) != wlen) {
				continue;
			}
			int len = WideCharToMultiByte(CP_UTF8, 0, wide.data(), wlen, nullptr, 0, nullptr, nullptr);
			if (len <= 0) continue;
			std::string out((size_t)len, '\0');
			if (WideCharToMultiByte(CP_UTF8, 0, wide.data(), wlen, out.data(), len, nullptr, nullptr) != len) {
				continue;
			}
			return out;
		}
	}
#endif
	return utf8_sanitize(bytes);  // undecodable: keep the text, mark bad bytes
}

}  // namespace phi
