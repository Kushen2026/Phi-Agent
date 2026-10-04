#include "util/json.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "util/utf8.hpp"

namespace phi {

JsonValue& JsonValue::operator[](std::string_view key) {
	if (type_ != Type::Object) {
		type_ = Type::Object;
		obj_ = std::make_shared<Object>();
	}
	for (auto& [k, v] : *obj_) {
		if (k == key) return v;
	}
	obj_->emplace_back(std::string(key), JsonValue());
	return obj_->back().second;
}

const JsonValue& JsonValue::at(std::string_view key) const {
	static const JsonValue null_v;
	const JsonValue* v = find(key);
	return v ? *v : null_v;
}

void JsonValue::push_back(JsonValue v) {
	if (type_ != Type::Array) {
		type_ = Type::Array;
		arr_ = std::make_shared<Array>();
	}
	arr_->push_back(std::move(v));
}

// Strict UTF-8 decoding lives in core/utf8.*, shared with the tool-output and
// file-reading paths: a per-layer "is this UTF-8?" check is exactly how CESU-8
// surrogates (ED A0 80) and overlong forms (C0 A1) used to slip through and
// break the websocket / the upstream request. Here we only need the "escape a
// code point that may not appear in a valid stream" rule.

// a code point that may not appear in a valid UTF-8 stream becomes U+FFFD
static unsigned san_cp(unsigned cp) {
	if (cp >= 0xD800u && cp <= 0xDFFFu) return 0xFFFDu;
	if (cp > 0x10FFFFu) return 0xFFFDu;
	return cp;
}

static void dump_string(const std::string& s, std::string& out) {
	out.reserve(out.size() + s.size() + 2);
	out.push_back('"');
	for (size_t i = 0; i < s.size();) {
		unsigned char c = (unsigned char)s[i];
		if (c < 0x80) {
			switch (c) {
				case '"': out += "\\\""; break;
				case '\\': out += "\\\\"; break;
				case '\b': out += "\\b"; break;
				case '\f': out += "\\f"; break;
				case '\n': out += "\\n"; break;
				case '\r': out += "\\r"; break;
				case '\t': out += "\\t"; break;
				default:
					if (c < 0x20) {
						char buf[8];
						snprintf(buf, sizeof(buf), "\\u%04x", c);
						out += buf;
					} else {
						out.push_back((char)c);
					}
			}
			i++;
			continue;
		}
		unsigned cp = 0;
		size_t n = utf8_sequence_len(s, i, cp);
		if (n == 0) {
			// not valid UTF-8: emit U+FFFD and resync on the next byte so the
			// serialized text is well-formed no matter what the source held
			out += "\\uFFFD";
			i++;
			continue;
		}
		out.append(s, i, n);  // valid sequence (incl. astral) passes through
		i += n;
	}
	out.push_back('"');
}

// python repr(float): shortest round-trip digits; positional when
// -4 < decpt <= 16, else exponential; integral positional gets ".0".
static void py_repr_double(double n, std::string& out) {
	out.reserve(32);  // reserve space for the formatted number
	char buf[40];
	int prec = 0;
	for (; prec <= 16; prec++) {
		snprintf(buf, sizeof(buf), "%.*e", prec, n);
		if (strtod(buf, nullptr) == n) break;
	}
	// buf looks like [-]d[.ddd]e±XX; extract sign, digits, exponent
	const char* s = buf;
	bool neg = (*s == '-');
	if (neg) s++;
	std::string digits;
	digits.push_back(*s++);
	if (*s == '.') {
		s++;
		while (*s && *s != 'e' && *s != 'E') digits.push_back(*s++);
	}
	int exp10 = atoi(s + 1);  // after 'e'
	int decpt = exp10 + 1;
	if (neg) out.push_back('-');
	if (decpt <= -4 || decpt > 16) {
		// exponential: d[.ddd]e±XX (min 2 exponent digits)
		out.push_back(digits[0]);
		if (digits.size() > 1) {
			out.push_back('.');
			out += digits.substr(1);
		}
		char eb[16];
		snprintf(eb, sizeof(eb), "e%+03d", exp10);
		out += eb;
	} else if (decpt <= 0) {
		out += "0.";
		out.append((size_t)(-decpt), '0');
		out += digits;
	} else if ((size_t)decpt >= digits.size()) {
		out += digits;
		out.append((size_t)decpt - digits.size(), '0');
		out += ".0";
	} else {
		out += digits.substr(0, (size_t)decpt);
		out.push_back('.');
		out += digits.substr((size_t)decpt);
	}
}

static void dump_value(const JsonValue& v, std::string& out, int indent, int depth) {
	auto nl = [&](int d) {
		if (indent >= 0) {
			out.push_back('\n');
			out.append((size_t)indent * d, ' ');
		}
	};
	switch (v.type()) {
		case JsonValue::Type::Null: out += "null"; break;
		case JsonValue::Type::Bool: out += v.as_bool() ? "true" : "false"; break;
		case JsonValue::Type::Number: {
			// An exact integer is printed as one, whatever its magnitude: a 20-digit
			// seed has to come back as the same 20 digits it was written with.
			if (v.int_repr()) {
				char buf[32];
				if (v.int_unsigned())
					snprintf(buf, sizeof(buf), "%llu", (unsigned long long)v.int_value());
				else
					snprintf(buf, sizeof(buf), "%lld", (long long)(int64_t)v.int_value());
				out += buf;
				break;
			}
			double n = v.as_number();
			if (std::isfinite(n)) {
				py_repr_double(n, out);
			} else {
				// python json.dumps default: Infinity / -Infinity / NaN
				if (std::isnan(n)) out += "NaN";
				else out += (n > 0) ? "Infinity" : "-Infinity";
			}
			break;
		}
		case JsonValue::Type::String: dump_string(v.as_string(), out); break;
		case JsonValue::Type::Array: {
			out.push_back('[');
			bool first = true;
			for (const auto& item : v.items()) {
				if (!first) {
					out.push_back(',');
					if (indent < 0) out.push_back(' ');  // python compact: ", "
				}
				first = false;
				nl(depth + 1);
				dump_value(item, out, indent, depth + 1);
			}
			if (!v.items().empty()) nl(depth);
			out.push_back(']');
			break;
		}
		case JsonValue::Type::Object: {
			out.push_back('{');
			bool first = true;
			for (const auto& [k, item] : v.entries()) {
				if (!first) {
					out.push_back(',');
					if (indent < 0) out.push_back(' ');  // python compact: ", "
				}
				first = false;
				nl(depth + 1);
				dump_string(k, out);
				out.push_back(':');
				out.push_back(' ');  // python always ": "
				dump_value(item, out, indent, depth + 1);
			}
			if (!v.entries().empty()) nl(depth);
			out.push_back('}');
			break;
		}
	}
}

std::string JsonValue::dump(int indent) const {
	std::string out;
	dump_value(*this, out, indent, 0);
	return out;
}

// ── parser ─────────────────────────────────────────────────────────────────

struct JsonParser {
	const char* p;
	const char* end;
	std::string error;

	bool fail(const std::string& msg) {
		if (error.empty()) error = msg;
		return false;
	}

	void skip_ws() {
		while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
	}

	bool parse_value(JsonValue& out, int depth) {
		if (depth > 128) return fail("nesting too deep");
		skip_ws();
		if (p >= end) return fail("unexpected end");
		char c = *p;
		if (c == '{') return parse_object(out, depth);
		if (c == '[') return parse_array(out, depth);
		if (c == '"') {
			std::string s;
			if (!parse_string(s)) return false;
			out = JsonValue(std::move(s));
			return true;
		}
		if (c == 't') {
			if (end - p >= 4 && strncmp(p, "true", 4) == 0) { p += 4; out = JsonValue(true); return true; }
			return fail("bad literal");
		}
		if (c == 'f') {
			if (end - p >= 5 && strncmp(p, "false", 5) == 0) { p += 5; out = JsonValue(false); return true; }
			return fail("bad literal");
		}
		if (c == 'n') {
			if (end - p >= 4 && strncmp(p, "null", 4) == 0) { p += 4; out = JsonValue(nullptr); return true; }
			return fail("bad literal");
		}
		return parse_number(out);
	}

	bool parse_number(JsonValue& out) {
		const char* start = p;
		if (p < end && *p == '-') p++;
		// must have at least one digit before decimal point or exponent
		bool has_digits = false;
		while (p < end && *p >= '0' && *p <= '9') { p++; has_digits = true; }
		if (!has_digits) return fail("bad number");
		bool has_decimal = false;
		if (p < end && *p == '.') {
			p++;
			has_decimal = true;
			// trailing dot without fractional digits is not valid JSON (e.g. "1.")
			bool has_frac_digits = false;
			while (p < end && *p >= '0' && *p <= '9') { p++; has_frac_digits = true; }
			if (!has_frac_digits) return fail("bad number: trailing dot");
		}
		if (p < end && (*p == 'e' || *p == 'E')) {
			p++;
			if (p < end && (*p == '+' || *p == '-')) p++;
			bool has_exp_digits = false;
			while (p < end && *p >= '0' && *p <= '9') { p++; has_exp_digits = true; }
			if (!has_exp_digits) return fail("bad number: expected exponent digits");
		}
		std::string text(start, p);
		// python parses 2 as int (dumps "2") but 2.0 as float (dumps "2.0"), and an
		// int keeps every digit it was written with. Storing only the double
		// silently rewrote a number that needs all 64 bits - 18208026114954838000
		// came back as 18208026114954838016 - and that number is exactly a seed the
		// caller asked the run to be reproducible from (the tool then read it
		// through as_int(), which refused the value outright and drew a fresh one).
		if (!has_decimal && text.find('e') == std::string::npos && text.find('E') == std::string::npos) {
			JsonValue exact;
			if (parse_int_exact(text, exact)) {
				out = exact;
				return true;
			}
			// more than 64 bits: nothing here can hold it exactly, keep the double
		}
		out = JsonValue(strtod(text.c_str(), nullptr));
		return true;
	}

	// Exact 64-bit read of an integer token (sign handled here, so strtoull's own
	// sign wrapping never sees one). False - and the caller falls back to the
	// double - when the magnitude does not fit.
	static bool parse_int_exact(const std::string& text, JsonValue& out) {
		const bool neg = !text.empty() && text[0] == '-';
		const char* digits = text.c_str() + (neg ? 1 : 0);
		if (*digits == '\0') return false;
		errno = 0;
		char* stop = nullptr;
		const unsigned long long mag = strtoull(digits, &stop, 10);
		// ERANGE catches a magnitude above 2^64-1; `stop` catches anything strtoull
		// would not consume.
		if (errno == ERANGE || stop != digits + strlen(digits)) return false;
		if (neg) {
			if (mag > 9223372036854775808ull) return false;  // below INT64_MIN
			out = JsonValue::make_int((uint64_t)(0 - mag), false);
		} else {
			out = JsonValue::make_int((uint64_t)mag, mag > 9223372036854775807ull);
		}
		return true;
	}

	bool parse_hex4(unsigned& out) {
		out = 0;
		for (int i = 0; i < 4; i++) {
			if (p >= end) return fail("bad \\u escape");
			char c = *p++;
			out <<= 4;
			if (c >= '0' && c <= '9') out |= (unsigned)(c - '0');
			else if (c >= 'a' && c <= 'f') out |= (unsigned)(c - 'a' + 10);
			else if (c >= 'A' && c <= 'F') out |= (unsigned)(c - 'A' + 10);
			else return fail("bad \\u escape");
		}
		return true;
	}

	static void append_utf8(std::string& s, unsigned cp) {
		if (cp < 0x80) {
			s.push_back((char)cp);
		} else if (cp < 0x800) {
			s.push_back((char)(0xC0 | (cp >> 6)));
			s.push_back((char)(0x80 | (cp & 0x3F)));
		} else if (cp < 0x10000) {
			s.push_back((char)(0xE0 | (cp >> 12)));
			s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
			s.push_back((char)(0x80 | (cp & 0x3F)));
		} else {
			s.push_back((char)(0xF0 | (cp >> 18)));
			s.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
			s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
			s.push_back((char)(0x80 | (cp & 0x3F)));
		}
	}

	bool parse_string(std::string& out) {
		p++;  // opening quote
		out.clear();
		while (p < end) {
			char c = *p++;
			if (c == '"') return true;
			if (c == '\\') {
				if (p >= end) return fail("bad escape");
				char e = *p++;
				switch (e) {
					case '"': out.push_back('"'); break;
					case '\\': out.push_back('\\'); break;
					case '/': out.push_back('/'); break;
					case 'b': out.push_back('\b'); break;
					case 'f': out.push_back('\f'); break;
					case 'n': out.push_back('\n'); break;
					case 'r': out.push_back('\r'); break;
					case 't': out.push_back('\t'); break;
					case 'u': {
						unsigned cp;
						if (!parse_hex4(cp)) return false;
						if (cp >= 0xD800 && cp <= 0xDBFF) {
							// high surrogate: only a following low surrogate forms a real
							// code point. Storing the raw value would write CESU-8
							// (ED A0 80) — invalid UTF-8 that poisons every later dump
							// (websocket close 1007 / "invalid unicode code point").
							// Mirror a browser decoder: an unpaired half becomes U+FFFD.
							unsigned lo = 0;
							bool have_low = (end - p >= 6 && p[0] == '\\' && p[1] == 'u');
							if (have_low) {
								p += 2;
								if (!parse_hex4(lo)) return false;
							}
							if (have_low && lo >= 0xDC00 && lo <= 0xDFFF) {
								cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
								append_utf8(out, cp);
							} else {
								append_utf8(out, 0xFFFDu);
								// the escape already consumed: keep its value (sanitized too)
								if (have_low) append_utf8(out, san_cp(lo));
							}
						} else if (cp >= 0xDC00 && cp <= 0xDFFF) {
							append_utf8(out, 0xFFFDu);  // lone low surrogate
						} else {
							append_utf8(out, san_cp(cp));
						}
						break;
					}
					default: return fail("bad escape");
				}
			} else {
				// raw (non-escaped) bytes are untrusted input as well: a stray
				// continuation / truncated / overlong byte must not survive as
				// invalid UTF-8 inside the parsed value
				if ((unsigned char)c < 0x80) {
					out.push_back(c);
				} else {
					unsigned cp = 0;
					size_t n = utf8_sequence_len(std::string_view(p - 1, (size_t)(end - (p - 1))), 0, cp);
					if (n == 0) {
						append_utf8(out, 0xFFFDu);  // resync on the next byte
					} else {
						out.append(p - 1, n);
						p += n - 1;
					}
				}
			}
		}
		return fail("unterminated string");
	}

	bool parse_array(JsonValue& out, int depth) {
		p++;  // [
		out = JsonValue::array();
		skip_ws();
		if (p < end && *p == ']') { p++; return true; }
		while (true) {
			JsonValue item;
			if (!parse_value(item, depth + 1)) return false;
			out.push_back(std::move(item));
			skip_ws();
			if (p >= end) return fail("unterminated array");
			if (*p == ',') { p++; continue; }
			if (*p == ']') { p++; return true; }
			return fail("expected , or ]");
		}
	}

	bool parse_object(JsonValue& out, int depth) {
		p++;  // {
		out = JsonValue::object();
		skip_ws();
		if (p < end && *p == '}') { p++; return true; }
		while (true) {
			skip_ws();
			if (p >= end || *p != '"') return fail("expected object key");
			std::string key;
			if (!parse_string(key)) return false;
			skip_ws();
			if (p >= end || *p != ':') return fail("expected :");
			p++;
			JsonValue value;
			if (!parse_value(value, depth + 1)) return false;
			// keep duplicates: python object_pairs_hook needs to see them
			out.obj_->emplace_back(std::move(key), std::move(value));
			skip_ws();
			if (p >= end) return fail("unterminated object");
			if (*p == ',') { p++; continue; }
			if (*p == '}') { p++; return true; }
			return fail("expected , or }");
		}
	}
};

std::optional<JsonValue> json_parse(std::string_view text, std::string* error) {
	JsonParser parser{text.data(), text.data() + text.size(), ""};
	JsonValue out;
	if (!parser.parse_value(out, 0)) {
		if (error) *error = parser.error;
		return std::nullopt;
	}
	return out;
}

}  // namespace phi
