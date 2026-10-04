// Strict UTF-8 validation/conversion shared by every layer that touches bytes
// of unknown origin: tool output, file contents, and the JSON on the wire.
//
// One rule, one implementation. When each layer rolled its own "is this
// UTF-8?" check, some only looked for a lead byte followed by continuation
// bytes and therefore accepted CESU-8 surrogates (ED A0 80 = U+D800) and
// overlong forms (C0 A1 = U+0021). Those bytes travelled on into:
//   * the JSON broadcast to the UI — Chromium closes a websocket whose text
//     frame is not valid UTF-8 (close code 1007), which the user experienced as
//     the app "randomly losing the connection", and
//   * the upstream chat request — rejected with
//     "HTTP 400 ... invalid unicode code point".
// Keeping the primitive in one place stops the checks from diverging again.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace phi {

// Strict RFC 3629 decode of the sequence starting at s[i]. Returns its length
// (1..4) and stores the code point in `cp`, or returns 0 when the bytes at that
// position are not valid UTF-8: a stray continuation byte, a truncated
// sequence, a missing continuation byte, an overlong encoding, a surrogate half
// (U+D800..U+DFFF), or a value beyond U+10FFFF.
size_t utf8_sequence_len(std::string_view s, size_t i, unsigned& cp);

// True when the whole buffer is valid UTF-8 under the rule above.
bool utf8_valid(std::string_view s);

// A copy of `s` in which every invalid byte is replaced by U+FFFD, so the
// result is always valid UTF-8 — and therefore always safe to serialize and
// send over a websocket.
std::string utf8_sanitize(std::string_view s);

// Decode bytes of unknown origin into valid UTF-8 text:
//   * already valid UTF-8 -> returned unchanged (the common case, zero cost)
//   * otherwise the system ANSI code page is tried, then GBK/CP936 explicitly
//     (a legacy .txt file or a console command's output on Chinese Windows
//     lives there); a lossless decode wins
//   * still undecodable -> utf8_sanitize, i.e. the offending bytes become
//     U+FFFD instead of poisoning the JSON downstream
// Used for tool output and file contents so a GBK file reads as 中文 rather than
// a wall of replacement boxes, without ever emitting invalid UTF-8.
std::string decode_external_text(const std::string& bytes);

}  // namespace phi
