// Shared helpers for the built-in tools: path resolution against the session
// cwd, JSON schema builders, and small text utilities.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "agent/agent.hpp"
#include "util/base.hpp"
#include "util/json.hpp"

namespace phi {

// ── paths ──────────────────────────────────────────────────────────────────

// resolve a tool path argument (absolute kept, relative joined with cwd),
// normalized with forward slashes for display and model consumption
std::string tool_resolve_path(const std::string& cwd, const std::string& path);
// display form: relative to cwd when the path lives under it, else absolute
std::string tool_display_path(const std::string& cwd, const std::string& abs_path);

// ── text ───────────────────────────────────────────────────────────────────

// split on '\n', dropping a single trailing empty line (CRLF stays; callers
// that need it stripped do so per line)
std::vector<std::string> tool_split_lines(const std::string& text);
std::string tool_strip_cr(const std::string& text);
// heuristics: NUL byte in the sample or a high control-char ratio
bool tool_looks_binary(const std::string& sample);
std::string tool_human_size(int64_t bytes);
// hard cap: keep head + tail with a marker in between (`language` localizes
// that marker, which lands in the tool output)
std::string tool_truncate_middle(const std::string& text, size_t max_chars,
	const std::string& language);

// ── tool settings (settings.json "tools") ─────────────────────────────────

// max output chars for the tool currently executing; falls back to `fallback`
// when the session carries no limit (or 0) for it
size_t tool_output_limit(const ToolContext& ctx, size_t fallback);

// true when abs_path lives under cwd (case-insensitive: windows paths)
bool tool_path_in_cwd(const std::string& cwd, const std::string& abs_path);

// permission gate for file tools: returns an empty string when allowed,
// otherwise a user-facing error explaining the denial
std::string tool_check_path_permission(const ToolContext& ctx, const std::string& abs_path);

// permission gate for bash: best-effort scan of the command line for absolute
// paths pointing outside the working directory ("in-cwd" mode); empty = allowed
std::string tool_check_command_permission(const ToolContext& ctx, const std::string& command);
// fnmatch-style glob: '*' any run, '?' single char (case-insensitive on ascii)
bool tool_glob_match(std::string_view glob, std::string_view name);

// file size in bytes, -1 when unavailable
int64_t tool_file_size(const std::string& path);

// ── images handed to the model ─────────────────────────────────────────────

// A picture sent to a model is not the file on disk but a re-encoded copy:
// decoded, scaled down to what a model can actually resolve, and written as
// JPEG so one screenshot cannot add megabytes of base64 to the request body and
// to the stored transcript (which carries the image again on every turn).
//
// `mime` / `bytes` / `recompressed` describe what was actually produced, so a
// tool can tell the model what it is looking at — they report the ORIGINAL bytes
// when the codec could not read the picture or the re-encode would not have made
// it smaller (a small PNG stays a PNG: lossy is a downgrade below a few KB).
// Re-encoding is lossy and JPEG has no alpha channel, so a transparent PNG is
// flattened onto black; that is the price of not carrying the file's full bytes
// through every later request.
struct ModelImage {
	ContentPart part;
	std::string mime;
	int64_t bytes = 0;
	bool recompressed = false;
};
ModelImage tool_model_image(const std::vector<uint8_t>& raw, const std::string& mime);

// ── video frames handed to the model ───────────────────────────────────────

// One frame of a video, decoded and re-encoded for the model exactly like a
// still read from disk: scaled to what a model can resolve and written as JPEG,
// so a handful of 1080p frames cannot add tens of megabytes of base64 to the
// request body and to the stored transcript.
//
// `frame` is the 1-based number the caller asked for; `seconds` is the
// presentation time the decode actually landed on. A request past the end of the
// file is answered with the last frame and `seconds` says so.
struct VideoFrameImage {
	int64_t frame = 0;
	double seconds = 0.0;
	ModelImage image;
};

// What a video read found: the stream's geometry plus one entry per requested
// frame, in the order they were asked for with duplicates collapsed. Throws
// MediaError when the container/stream cannot be opened (so a caller can tell
// "not a video this machine can decode" from "a frame the file does not have");
// a frame that cannot be decoded is reported with an empty image rather than
// aborting the whole read.
struct VideoProbe {
	int64_t width = 0;
	int64_t height = 0;
	double fps = 0.0;
	double duration = 0.0;
	int64_t frame_count = 0;  // round(duration * fps), 0 when unknown
	std::vector<VideoFrameImage> frames;
};
VideoProbe tool_read_video_frames(const std::string& abs_path, const std::vector<int64_t>& frames);

// ── json schema builders ───────────────────────────────────────────────────
// Every parameter description is a (Chinese, English) pair: the session
// language picks one, so a schema built for an English session never carries a
// Chinese description (and vice versa).
JsonValue schema_string(const std::string& language, const char* zh, const char* en);
JsonValue schema_number(const std::string& language, const char* zh, const char* en);
JsonValue schema_boolean(const std::string& language, const char* zh, const char* en);
JsonValue schema_array(const std::string& language, JsonValue items, const char* zh, const char* en);
JsonValue schema_object(std::vector<std::pair<std::string, JsonValue>> properties,
	std::vector<std::string> required);

// ── args access ────────────────────────────────────────────────────────────
std::string arg_string(const JsonValue& args, const char* key, const std::string& def = "");
int64_t arg_int(const JsonValue& args, const char* key, int64_t def);
// The read of a 64-bit unsigned id (the media tools' `seed`): the whole uint64
// range is legal there, so int64_t is the wrong width for it. This is the
// *presence* form - true and *out set when the call did give one - rather than a
// value-with-default, because the ids it reads have a meaningful 0: a seed the
// caller spelled out (0 included) has to come back as itself, while only an
// argument that is absent altogether leaves the caller free to draw its own.
bool arg_u64_present(const JsonValue& args, const char* key, uint64_t* out);
// JSON numbers arrive as doubles (the schema's "number" type); this is the
// accessor for a fractional one. Strings are accepted too, since some callers
// quote their numerics.
double arg_number(const JsonValue& args, const char* key, double def);

// ── process polling loop shared by bash / grep / fetch-driven spawn ────────
struct ToolExecResult {
	int exit_code = 0;
	std::string output;       // merged stdout+stderr (utf-8 best effort)
	bool timed_out = false;
	bool aborted = false;
	bool spawn_failed = false;
	bool output_truncated = false;
};
// run `cmdline` through the shell (spawn_shell), polling until exit/timeout;
// on_update receives the live tail of the output (~every poll tick). `language`
// localizes the built-in messages it appends (spawn failure, output cap).
ToolExecResult tool_run_shell(const std::string& cmdline, const std::string& cwd,
	int timeout_ms, const std::function<bool()>& cancelled,
	const std::string& language,
	const std::function<void(const std::string&)>& on_update = {});

}  // namespace phi
