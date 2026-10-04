#include "tools/tools_helpers.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <windows.h>

#include "io/image_io.hpp"
#include "io/video_io.hpp"
#include "util/base.hpp"
#include "util/i18n.hpp"
#include "util/utf8.hpp"

namespace phi {

// ── paths ──────────────────────────────────────────────────────────────────

namespace {
// windows absolute: drive letter (C:\), UNC (\\), or root slash
bool looks_absolute(const std::string& p) {
	if (p.size() >= 2 && ((p[0] >= 'a' && p[0] <= 'z') || (p[0] >= 'A' && p[0] <= 'Z')) && p[1] == ':')
		return true;
	return !p.empty() && (p[0] == '/' || p[0] == '\\');
}
}  // namespace

std::string tool_resolve_path(const std::string& cwd, const std::string& path) {
	std::string p = trim(path);
	if (p.empty()) return path_absolute(cwd);
	// tolerate file:// urls and quoted paths
	if (starts_with(p, "file:///")) p = p.substr(8);
	else if (starts_with(p, "file://")) p = p.substr(7);
	while (!p.empty() && (p.front() == '"' || p.front() == '\'')) p.erase(p.begin());
	while (!p.empty() && (p.back() == '"' || p.back() == '\'')) p.pop_back();
	if (p.empty()) return path_absolute(cwd);
	std::string joined = looks_absolute(p) ? p : path_join(cwd, p);
	return path_normalize(joined);
}

std::string tool_display_path(const std::string& cwd, const std::string& abs_path) {
	// case-insensitive comparison, same as tool_path_in_cwd: Windows paths are
	// case-insensitive and path_normalize keeps case, so a model-supplied
	// lowercase spelling of the cwd used to fail the prefix test and the tool
	// reported an absolute path where a relative one was intended.
	std::string base = to_lower(path_normalize(cwd));
	std::string p_lower = to_lower(path_normalize(abs_path));
	std::string base_slash = base;
	if (!base_slash.empty() && base_slash.back() != '/' && base_slash.back() != '\\')
		base_slash += "/";
	if (starts_with(p_lower, base_slash)) {
		// slice the ORIGINAL normalized path, not the lowered copy: the display
		// form must keep the case the caller typed
		std::string p = path_normalize(abs_path);
		std::string rel = p.substr(base_slash.size());
		return rel.empty() ? "." : rel;
	}
	return path_normalize(abs_path);
}

// ── text ───────────────────────────────────────────────────────────────────

std::vector<std::string> tool_split_lines(const std::string& text) {
	std::vector<std::string> lines;
	size_t start = 0;
	while (start <= text.size()) {
		size_t nl = text.find('\n', start);
		if (nl == std::string::npos) {
			lines.push_back(text.substr(start));
			break;
		}
		lines.push_back(text.substr(start, nl - start));
		start = nl + 1;
	}
	while (!lines.empty() && lines.back().empty()) lines.pop_back();
	return lines;
}

std::string tool_strip_cr(const std::string& text) {
	std::string out;
	out.reserve(text.size());
	for (char c : text) {
		if (c != '\r') out.push_back(c);
	}
	return out;
}

bool tool_looks_binary(const std::string& sample) {
	if (sample.find('\0') != std::string::npos) return true;
	if (sample.empty()) return false;
	size_t controls = 0;
	size_t n = std::min<size_t>(sample.size(), 8192);
	for (size_t i = 0; i < n; i++) {
		unsigned char c = (unsigned char)sample[i];
		if (c < 9 || (c > 13 && c < 32)) controls++;
	}
	return controls * 100 / n > 10;
}

std::string tool_human_size(int64_t bytes) {
	char buf[32];
	if (bytes < 1024) {
		snprintf(buf, sizeof(buf), "%lld B", (long long)bytes);
	} else if (bytes < 1024 * 1024) {
		snprintf(buf, sizeof(buf), "%.1f KB", (double)bytes / 1024.0);
	} else {
		snprintf(buf, sizeof(buf), "%.1f MB", (double)bytes / (1024.0 * 1024.0));
	}
	return buf;
}

std::string tool_truncate_middle(const std::string& text, size_t max_chars,
	const std::string& language) {
	if (text.size() <= max_chars) return text;
	if (max_chars < 64) return text.substr(0, max_chars);
	size_t head = max_chars * 2 / 3;
	size_t tail = max_chars - head;
	long long hidden = (long long)text.size() - (long long)max_chars;
	std::string out = text.substr(0, head);
	out += tr(language, "\n... [已省略 " + std::to_string(hidden) + " 个字符] ...\n",
		"\n... [" + std::to_string(hidden) + " characters omitted] ...\n");
	out += text.substr(text.size() - tail);
	return out;
}

// ── tool settings ──────────────────────────────────────────────────────────

size_t tool_output_limit(const ToolContext& ctx, size_t fallback) {
	auto it = ctx.output_limits.find(ctx.current_tool);
	if (it == ctx.output_limits.end() || it->second == 0) return fallback;
	return it->second;
}

bool tool_path_in_cwd(const std::string& cwd, const std::string& abs_path) {
	std::string base = to_lower(path_normalize(cwd));
	std::string p = to_lower(path_normalize(abs_path));
	if (!base.empty() && base.back() != '/' && base.back() != '\\') base += '/';
	if (!p.empty() && p.back() != '/' && p.back() != '\\') p += '/';
	return starts_with(p, base);
}

std::string tool_check_path_permission(const ToolContext& ctx, const std::string& abs_path) {
	auto it = ctx.permissions.find(ctx.current_tool);
	if (it == ctx.permissions.end() || it->second != "in-cwd") return "";
	if (tool_path_in_cwd(ctx.cwd, abs_path)) return "";
	return tr(ctx.language,
		"拒绝访问: " + ctx.current_tool + " 被限制在工作目录内",
		"Access denied: " + ctx.current_tool + " is restricted to the working directory");
}

std::string tool_check_command_permission(const ToolContext& ctx, const std::string& command) {
	auto it = ctx.permissions.find("bash");
	if (it == ctx.permissions.end() || it->second != "in-cwd") return "";
	// best-effort scan: flag windows-style drive paths and unix-style /c/ roots
	// that resolve outside the working directory
	for (size_t i = 0; i + 1 < command.size(); i++) {
		char c = command[i];
		bool drive = ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) &&
			command[i + 1] == ':';
		bool unix_root = c == '/' && i + 2 < command.size() &&
			((command[i + 1] >= 'a' && command[i + 1] <= 'z') ||
			 (command[i + 1] >= 'A' && command[i + 1] <= 'Z')) &&
			command[i + 2] == '/';
		if (!drive && !unix_root) continue;
		size_t start = i;
		size_t end = command.find_first_of(" \t\"'|&;<>()\n", i + (drive ? 2 : 3));
		if (end == std::string::npos) end = command.size();
		std::string token = command.substr(start, end - start);
		// resolve and test: drive paths directly, unix-style via the cwd drive
		std::string candidate;
		if (drive) {
			candidate = token;
		} else {
			// /c/foo -> C:/foo when the cwd is on C:
			std::string cwd_norm = path_normalize(ctx.cwd);
			if (cwd_norm.size() < 2 || cwd_norm[1] != ':') continue;
			candidate = std::string(1, cwd_norm[0]) + ":" + token.substr(2);
		}
		std::string abs = path_normalize(path_absolute(candidate));
		if (!tool_path_in_cwd(ctx.cwd, abs)) {
			return tr(ctx.language,
				"拒绝访问: bash 被限制在工作目录内",
				"Access denied: bash is restricted to the working directory");
		}
		i = end - 1;
	}
	return "";
}

bool tool_glob_match(std::string_view glob, std::string_view name) {
	// iterative matcher: '*' any run (greedy with backtrack), '?' one char
	size_t g = 0, n = 0, star_g = std::string_view::npos, star_n = 0;
	auto eq = [](char a, char b) {
		a = (char)std::tolower((unsigned char)a);
		b = (char)std::tolower((unsigned char)b);
		return a == b;
	};
	while (n < name.size()) {
		if (g < glob.size() && (glob[g] == '?' || eq(glob[g], name[n]))) {
			g++;
			n++;
		} else if (g < glob.size() && glob[g] == '*') {
			star_g = g++;
			star_n = n;
		} else if (star_g != std::string_view::npos) {
			g = star_g + 1;
			n = ++star_n;
		} else {
			return false;
		}
	}
	while (g < glob.size() && glob[g] == '*') g++;
	return g == glob.size();
}

// ── images handed to the model ─────────────────────────────────────────────

// The long side of the picture the model is actually given. Past this the
// provider resamples it away anyway (Anthropic's own guidance is 1568 px), while
// every extra pixel is still paid for twice: in the request body and in the
// transcript that carries it again on each turn.
namespace {
constexpr int kModelImageMaxSide = 1568;
// Below ~80 the artefacts start showing on line art and screenshots — which is
// most of what gets read here — so this is the usual photo/screenshot
// compromise, not the smallest file that still looks like a file.
constexpr int kModelImageQuality = 85;

// Scale a decoded RGB buffer to what a model can resolve and encode it as JPEG.
// Shared by the still-image path and the video-frame path: a frame out of a
// video arrives here as the same tightly packed HWC the still path produces, so
// both get exactly one re-encode and one set of numbers.
std::vector<unsigned char> encode_model_jpeg(const unsigned char* rgb, int w, int h) {
	std::vector<unsigned char> scaled;
	int ow = w, oh = h;
	const int long_side = std::max(w, h);
	if (long_side > kModelImageMaxSide) {
		const double s = (double)kModelImageMaxSide / (double)long_side;
		ow = std::max(1, (int)std::lround((double)w * s));
		oh = std::max(1, (int)std::lround((double)h * s));
		media::resize_lanczos3(rgb, w, h, ow, oh, scaled);
	} else {
		scaled.assign(rgb, rgb + (size_t)w * (size_t)h * 3);
	}
	return media::encode_jpeg_memory(scaled.data(), ow, oh, kModelImageQuality);
}
}  // namespace

ModelImage tool_model_image(const std::vector<uint8_t>& raw, const std::string& mime) {
	ModelImage out;
	out.mime = mime.empty() ? "image/png" : mime;
	out.bytes = (int64_t)raw.size();
	// The pass-through form is built first, not as an afterthought: the codec
	// below is allowed to fail at every step, and the picture the caller asked
	// for is a better answer than an error in all of them.
	out.part = ContentPart::image(base64_encode_bytes(raw), out.mime);
	if (raw.empty()) return out;

	std::vector<unsigned char> rgb;
	int w = 0, h = 0;
	try {
		rgb = media::decode_image_memory(raw.data(), raw.size(), &w, &h);
	} catch (const std::exception&) {
		// not a picture this machine's codecs know (webp, svg, a truncated
		// file): hand the bytes over as they are
		return out;
	}
	if (w <= 0 || h <= 0 || rgb.size() < (size_t)w * (size_t)h * 3) return out;

	try {
		std::vector<unsigned char> jpeg;
		try {
			jpeg = encode_model_jpeg(rgb.data(), w, h);
		} catch (const std::exception&) {
			// encoder unavailable, or the resize refused the shape: the fallback in
			// `out` is still the original payload
		}
		// Only take the re-encode when it actually pays for itself: a diagram
		// that is already 6 KB smaller as a PNG stays a PNG.
		if (!jpeg.empty() && jpeg.size() < raw.size()) {
			out.part = ContentPart::image(base64_encode_bytes(jpeg), "image/jpeg");
			out.mime = "image/jpeg";
			out.bytes = (int64_t)jpeg.size();
			out.recompressed = true;
		}
	} catch (const std::exception&) {
		// the fallback in `out` is still the original payload
	}
	return out;
}

VideoProbe tool_read_video_frames(const std::string& abs_path, const std::vector<int64_t>& frames) {
	VideoProbe probe;

	// open() throws MediaError when the container/stream cannot be read at all;
	// the caller turns that into a tool error rather than a text read attempt.
	media::VideoReader reader;
	reader.open(abs_path);
	probe.width = reader.width();
	probe.height = reader.height();
	probe.fps = reader.fps();
	probe.duration = reader.duration_seconds();
	if (probe.fps > 0.0 && probe.duration > 0.0)
		probe.frame_count = (int64_t)std::llround(probe.duration * probe.fps);

	// Decoding walks the file in frame order — a forward seek is the only one MF
	// does cheaply — while the report keeps the order the caller asked in.
	std::vector<size_t> order(frames.size());
	for (size_t i = 0; i < order.size(); i++) order[i] = i;
	std::stable_sort(order.begin(), order.end(),
		[&](size_t a, size_t b) { return frames[a] < frames[b]; });

	const double fps = probe.fps > 0.0 ? probe.fps : 30.0;
	std::vector<VideoFrameImage> decoded(frames.size());
	for (size_t i : order) {
		const int64_t f = frames[i];
		VideoFrameImage& out = decoded[i];
		out.frame = f;
		const int64_t target = f - 1;  // 0-based
		double seconds = (double)target / fps;
		if (probe.duration > 0.0 && seconds > probe.duration) seconds = probe.duration;
		try {
			reader.seek(seconds);
			media::VideoFrame vf;
			if (reader.read(vf)) {
				// The frame number is not quoted from `VideoFrame::index`: that
				// counter counts reads since the last seek(), and a seek here is a
				// time. The presentation time is what was asked about, so it is what
				// the caller reports on.
				out.seconds = vf.seconds;

				// A decoder's surface may be padded past the geometry the file
				// declares (this H.264 stream decodes to 1920x1088 for a 1920x1080
				// picture, the extra rows being alignment, not image), and the
				// reader reports the padded height once it has produced one. The
				// crop is to the geometry recorded at open time, top-left aligned,
				// so the picture keeps the aspect the file declares.
				int64_t cw = vf.w, ch = vf.h;
				if (probe.width > 0 && probe.height > 0) {
					cw = std::min(cw, probe.width);
					ch = std::min(ch, probe.height);
				}
				const unsigned char* px = vf.rgb.data();
				std::vector<unsigned char> cropped;
				if (cw > 0 && ch > 0 && (cw != vf.w || ch != vf.h)) {
					cropped.resize((size_t)cw * (size_t)ch * 3);
					for (int64_t y = 0; y < ch; y++)
						std::memcpy(&cropped[(size_t)y * (size_t)cw * 3],
							&vf.rgb[(size_t)y * (size_t)vf.w * 3], (size_t)cw * 3);
					px = cropped.data();
				}
				if (px && cw > 0 && ch > 0) {
					out.image.mime = "image/jpeg";
					try {
						std::vector<unsigned char> jpeg = encode_model_jpeg(px, (int)cw, (int)ch);
						if (!jpeg.empty()) {
							out.image.part = ContentPart::image(base64_encode_bytes(jpeg), "image/jpeg");
							out.image.bytes = (int64_t)jpeg.size();
							out.image.recompressed = true;
						}
					} catch (const std::exception&) {
						// a frame the encoder refused: reported with an empty image
					}
				}
			}
		} catch (const std::exception&) {
			// a frame that will not decode does not abort the frames that will
		}
	}
	probe.frames = std::move(decoded);
	return probe;
}

// ── json schema builders ───────────────────────────────────────────────────

// Each description comes as a (Chinese, English) pair; tr() picks the one
// matching the session language, so a parameter description is never
// hard-coded to a single language.
JsonValue schema_string(const std::string& language, const char* zh, const char* en) {
	JsonValue v = JsonValue::object();
	v["type"] = "string";
	std::string description = tr(language, zh, en);
	if (!description.empty()) v["description"] = description;
	return v;
}

JsonValue schema_number(const std::string& language, const char* zh, const char* en) {
	JsonValue v = JsonValue::object();
	v["type"] = "number";
	std::string description = tr(language, zh, en);
	if (!description.empty()) v["description"] = description;
	return v;
}

JsonValue schema_boolean(const std::string& language, const char* zh, const char* en) {
	JsonValue v = JsonValue::object();
	v["type"] = "boolean";
	std::string description = tr(language, zh, en);
	if (!description.empty()) v["description"] = description;
	return v;
}

JsonValue schema_array(const std::string& language, JsonValue items, const char* zh, const char* en) {
	JsonValue v = JsonValue::object();
	v["type"] = "array";
	std::string description = tr(language, zh, en);
	if (!description.empty()) v["description"] = description;
	v["items"] = std::move(items);
	return v;
}

JsonValue schema_object(std::vector<std::pair<std::string, JsonValue>> properties,
	std::vector<std::string> required) {
	JsonValue v = JsonValue::object();
	v["type"] = "object";
	JsonValue props = JsonValue::object();
	for (auto& [name, schema] : properties) props[name] = std::move(schema);
	v["properties"] = std::move(props);
	if (!required.empty()) {
		JsonValue req = JsonValue::array();
		for (auto& r : required) req.push_back(r);
		v["required"] = std::move(req);
	}
	return v;
}

int64_t tool_file_size(const std::string& path) {
	WIN32_FILE_ATTRIBUTE_DATA fad;
	std::wstring w = utf8_to_wide(path_backslash(path));
	if (!GetFileAttributesExW(w.c_str(), GetFileExInfoStandard, &fad)) return -1;
	ULARGE_INTEGER ul;
	ul.HighPart = fad.nFileSizeHigh;
	ul.LowPart = fad.nFileSizeLow;
	return (int64_t)ul.QuadPart;
}

// ── args access ─────────────────────────────────────────────────────────────

std::string arg_string(const JsonValue& args, const char* key, const std::string& def) {
	const JsonValue* v = args.find(key);
	if (!v || v->is_null()) return def;
	if (v->is_string()) return v->as_string();
	return def;
}

int64_t arg_int(const JsonValue& args, const char* key, int64_t def) {
	const JsonValue* v = args.find(key);
	if (!v || v->is_null()) return def;
	if (v->is_number()) return v->as_int(def);
	if (v->is_string()) {
		try {
			return std::stoll(v->as_string());
		} catch (...) {
			return def;
		}
	}
	return def;
}

namespace {
// The parse behind arg_u64_present: an exact integer (JSON integer tokens are kept
// as their 64 bits, so a seed above INT64_MAX arrives whole), a non-negative
// double, or a quoted decimal. False when the value is not a seed at all -
// missing, null, negative, empty, unparsable - which is what keeps a stray
// spelling from silently seeding a run.
bool parse_u64_arg(const JsonValue& args, const char* key, uint64_t& out) {
	const JsonValue* v = args.find(key);
	if (!v || v->is_null()) return false;
	if (v->is_number()) {
		// An exact integer needs no double round-trip: only its sign is in question
		// (int_unsigned() is the value carried above INT64_MAX, which is a seed).
		if (v->int_repr()) {
			if (!v->int_unsigned() && (int64_t)v->int_value() < 0) return false;
			out = v->int_value();
			return true;
		}
		const double d = v->as_number();
		if (!(d >= 0.0 && d < 18446744073709551616.0)) return false;
		out = (uint64_t)d;
		return true;
	}
	if (v->is_string()) {
		// A quoted seed ("1234") is a valid spelling of the same id; a negative one
		// is not a seed, so it is rejected rather than wrapped around
		const std::string& s = v->as_string();
		if (s.empty() || s[0] == '-') return false;
		errno = 0;
		char* stop = nullptr;
		const unsigned long long n = strtoull(s.c_str(), &stop, 10);
		if (errno == ERANGE || stop == s.c_str() || *stop != '\0') return false;
		out = (uint64_t)n;
		return true;
	}
	return false;
}
}  // namespace

bool arg_u64_present(const JsonValue& args, const char* key, uint64_t* out) {
	uint64_t v = 0;
	if (!parse_u64_arg(args, key, v)) return false;
	if (out) *out = v;
	return true;
}

double arg_number(const JsonValue& args, const char* key, double def) {
	const JsonValue* v = args.find(key);
	if (!v || v->is_null()) return def;
	if (v->is_number()) return v->as_number(def);
	if (v->is_string()) {
		try {
			return std::stod(v->as_string());
		} catch (...) {
			return def;
		}
	}
	return def;
}

// ── process polling ────────────────────────────────────────────────────────

// A text chunk read from a pipe is a raw byte stream: it may start or end in
// the middle of a multi-byte utf-8 sequence, so partial sequences are tracked
// across polls so the UI never renders a half-sequence mid-stream.
// Validation is STRICT (see core/util/utf8.hpp): "lead byte followed by
// continuation bytes" is not enough, because CESU-8 surrogates (ED A0 80) and
// overlong forms (C0 A1) satisfy it while being invalid UTF-8. Forwarding those
// put invalid bytes into the JSON sent to the UI, where Chromium closes the
// websocket (close code 1007) — the app "randomly losing the connection".
// Anything not valid UTF-8 becomes U+FFFD instead.
// (A legacy-codepage stream, e.g. GBK console output, therefore reads as
// replacement marks. That is deliberate: re-decoding it here would require
// buffering the whole — up to 10MB — stream, and wire correctness comes first.
// File contents go through decode_external_text(), which is lossless.)
namespace {

// U+FFFD in UTF-8
constexpr const char* kReplacementChar = "\xEF\xBF\xBD";

struct Utf8Sanitizer {
	std::string pending;

	void feed(const std::string& chunk, std::string& out) {
		std::string all = pending + chunk;
		pending.clear();
		size_t i = 0;
		while (i < all.size()) {
			unsigned char c = (unsigned char)all[i];
			if (c < 0x80) {
				out.push_back((char)c);
				i++;
				continue;
			}
			// how many bytes this lead byte promises (0 = not a lead byte at all)
			size_t need = 0;
			if ((c & 0xE0) == 0xC0) need = 2;
			else if ((c & 0xF0) == 0xE0) need = 3;
			else if ((c & 0xF8) == 0xF0) need = 4;
			if (need == 0) {
				out += kReplacementChar;
				i++;
				continue;
			}
			if (i + need > all.size()) {
				// the sequence may simply be split across two pipe reads: hold it
				// back until the rest arrives (flush() settles it at stream end)
				pending.assign(all, i, std::string::npos);
				break;
			}
			// enough bytes are buffered, so a zero result means genuinely invalid:
			// missing continuation, overlong form, surrogate half, out of range
			unsigned cp = 0;
			size_t n = utf8_sequence_len(all, i, cp);
			if (n == 0) {
				out += kReplacementChar;
				i++;
				continue;
			}
			out.append(all, i, n);
			i += n;
		}
	}

	void flush(std::string& out) {
		if (!pending.empty()) {
			out += kReplacementChar;
			pending.clear();
		}
	}
};

}  // namespace

ToolExecResult tool_run_shell(const std::string& cmdline, const std::string& cwd,
	int timeout_ms, const std::function<bool()>& cancelled,
	const std::string& language,
	const std::function<void(const std::string&)>& on_update) {
	ToolExecResult res;
	ChildProcess* child = spawn_shell(cmdline, cwd);
	if (!child) {
		res.spawn_failed = true;
		res.exit_code = -1;
		res.output = tr(language, "启动失败: 找不到 shell", "spawn failed: shell not found");
		return res;
	}

	auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
	Utf8Sanitizer sanitizer;
	std::string raw;
	auto last_update = std::chrono::steady_clock::now();
	// hard output cap: a runaway command (cat on a multi-GB log, yes-loop, a
	// build spewing endless warnings) used to accumulate unbounded memory and
	// could OOM the whole app. 10MB is far above every real output need.
	static constexpr size_t kMaxRawOutput = 10ull * 1024 * 1024;
	bool truncated = false;

	while (true) {
		if (cancelled && cancelled()) {
			child_kill(child);
			res.aborted = true;
			break;
		}
		// deadline is checked every iteration, not only when the pipe is idle:
		// a command that keeps producing output ("while true; do echo x; done")
		// would otherwise never reach the branch below, so the timeout never
		// fired on exactly the commands it exists for.
		if (std::chrono::steady_clock::now() > deadline) {
			child_kill(child);
			res.timed_out = true;
			break;
		}
		std::string chunk;
		bool got = child_poll(child, chunk);
		if (got) {
			std::string clean;
			sanitizer.feed(chunk, clean);
			if (raw.size() + clean.size() >= kMaxRawOutput) {
				raw.append(clean, 0,
				   kMaxRawOutput > raw.size() ? kMaxRawOutput - raw.size() : 0);
				truncated = true;
				child_kill(child);
				res.output_truncated = true;
				break;
			}
			raw += clean;
			if (on_update) {
				auto now = std::chrono::steady_clock::now();
				if (now - last_update > std::chrono::milliseconds(200)) {
					last_update = now;
					std::string tail = raw.size() > 2000 ? raw.substr(raw.size() - 2000) : raw;
					on_update(tail);
				}
			}
			continue;
		}
		if (child_alive(child)) {
			std::this_thread::sleep_for(std::chrono::milliseconds(40));
			continue;
		}
		// exited: drain the leftovers (child_poll returns them first)
		std::string extra;
		while (child_poll(child, extra)) {
			std::string clean;
			sanitizer.feed(extra, clean);
			raw += clean;
		}
		break;
	}
	sanitizer.flush(raw);
	res.output = std::move(raw);
	res.exit_code = child_wait(child);
	if (truncated) {
		res.exit_code = 0;   // override the kill code from truncated output
	}
	child_free(child);
	return res;
}

}  // namespace phi
