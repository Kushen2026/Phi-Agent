#include "tools/tools.hpp"
#include "util/rtk.hpp"
#include "tools/tools_helpers.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <regex>
#include <string>
#include <vector>

#include "util/base.hpp"
#include "util/i18n.hpp"
#include "util/utf8.hpp"

namespace phi {

namespace {

// ── schema shorthand ───────────────────────────────────────────────────────

// Every built-in prompt string has two versions: a Chinese one and a plain,
// easy-to-understand English one. make_tool registers both, plus an input-schema
// builder that materializes the parameter descriptions for the session language
// (see build_tools_json).
ToolDef make_tool(const char* name, const char* description, const char* description_en,
	const char* snippet, const char* snippet_en,
	std::function<JsonValue(const std::string& language)> parameters_fn,
	std::function<ToolResult(const JsonValue&, ToolContext&)> run) {
	ToolDef t;
	t.name = name;
	t.description = description;
	t.description_en = description_en;
	t.prompt_snippet = snippet;
	t.prompt_snippet_en = snippet_en;
	t.parameters_fn = std::move(parameters_fn);
	t.run = std::move(run);
	return t;
}

// ── bash ───────────────────────────────────────────────────────────────────

ToolDef bash_tool() {
	return make_tool(
		"bash",
		"执行 shell 命令并查看输出。通过真实 shell 运行（支持管道、&&、重定向和 Unix coreutils。输出会自动压缩（去除 ANSI 代码、精简 git/构建输出、硬截断）。文件操作请优先使用专用工具（read/edit/write/ls/grep）。",
		"Execute a shell command and see its output. Runs through a real shell (pipes, &&, redirects and Unix coreutils work). Output is automatically compacted (ANSI stripped, git/build output condensed, hard truncation). Prefer dedicated tools (read/edit/write/ls/grep) for file work.",
		"执行 shell 命令，实时输出",
		"Run a shell command and stream its output",
		[](const std::string& lang) {
			return schema_object({
				{"command", schema_string(lang, "要运行的 shell 命令", "The shell command to run")},
				{"timeout", schema_number(lang, "超时时间（毫秒）", "Timeout in milliseconds")},
			}, {"command"});
		},
		[](const JsonValue& args, ToolContext& ctx) -> ToolResult {
			std::string command = trim(arg_string(args, "command"));
			if (command.empty()) {
				return error_result(tr(ctx.language, "命令不能为空", "command must not be empty"));
			}
			if (std::string deny = tool_check_command_permission(ctx, command); !deny.empty()) {
				return error_result(deny);
			}
			int64_t timeout = arg_int(args, "timeout", 120000);
			timeout = std::clamp<int64_t>(timeout, 1000, 600000);

			// windows niceties ported from the rtk optimizer: python needs utf-8 io
			std::string effective = command;
			if (starts_with(to_lower(trim(command)), "python")) {
				effective = "export PYTHONIOENCODING=utf-8; " + command;
			}

			ToolExecResult r = tool_run_shell(effective, ctx.cwd, (int)timeout, ctx.cancelled, ctx.language,
				[&](const std::string& partial) {
					if (ctx.on_update) ctx.on_update(partial);
				});
			if (r.spawn_failed) return error_result(r.output);

			std::string out = rtk::compact_output(r.output, command, ctx.language);

				std::string footer;
				if (r.aborted) {
					footer = tr(ctx.language, "\n[已被用户中止]", "\n[aborted by user]");
				} else if (r.timed_out) {
					footer = tr(ctx.language,
						"\n[超过 " + std::to_string(timeout) + " 毫秒未完成 — 进程已被终止]",
						"\n[timed out after " + std::to_string(timeout) + " ms — process killed]");
				} else if (r.output_truncated) {
					footer = tr(ctx.language,
						"\n[输出超过 10MB 已截断 — 命令已被终止]",
						"\n[output truncated at 10MB — command killed]");
				}
				// A clean exit is the normal case and needs no line of its own: only a
				// non-zero code is worth reporting.
				if (r.exit_code != 0)
					footer += tr(ctx.language, "\n(退出码 " + std::to_string(r.exit_code) + ")",
						"\n(exit code " + std::to_string(r.exit_code) + ")");

				std::string text = out.empty() ? tr(ctx.language, "(无输出)", "(no output)") : out;
				text += footer;
				return text_result(tool_truncate_middle(text, tool_output_limit(ctx, 120000), ctx.language),
							   r.exit_code != 0 || r.timed_out || r.aborted);
		});
}

// ── read ───────────────────────────────────────────────────────────────────

// extension -> image mime, for the multimodal read path (.jpg / .jpeg / .png)
bool image_mime_for_path(const std::string& path, std::string& mime) {
	size_t dot = path.find_last_of('.');
	if (dot == std::string::npos) return false;
	std::string ext = to_lower(path.substr(dot));
	if (ext == ".png") { mime = "image/png"; return true; }
	if (ext == ".jpg" || ext == ".jpeg") { mime = "image/jpeg"; return true; }
	return false;
}

// ── video frames ───────────────────────────────────────────────────────────

// Extension -> "this is a video container", for the frame read path. mp4 is the
// format the tool is documented for; the others share the same MF reader and
// cost nothing to accept (a container whose decoder is missing fails open() and
// is reported as such).
bool video_ext_for_path(const std::string& path) {
	size_t dot = path.find_last_of('.');
	if (dot == std::string::npos) return false;
	const std::string ext = to_lower(path.substr(dot));
	return ext == ".mp4" || ext == ".m4v" || ext == ".mov" || ext == ".mkv" ||
	       ext == ".webm" || ext == ".avi";
}

// `frames` arrives either as a JSON array of numbers (the schema's shape) or as
// a "1, 30, 60" string; both spellings are accepted because the parameter is
// written by hand as often as by a model.
std::vector<int64_t> parse_frame_args(const JsonValue& args) {
	std::vector<int64_t> frames;
	auto take = [&](const std::string& token) {
		std::string s = trim(token);
		if (s.empty()) return;
		try {
			size_t used = 0;
			long long v = std::stoll(s, &used);
			if (used == s.size()) frames.push_back((int64_t)v);
		} catch (const std::exception&) {
			// a token that is not a number is dropped, not an error
		}
	};
	const JsonValue& fv = args["frames"];
	if (fv.is_string()) {
		std::string cur;
		for (char c : fv.as_string()) {
			if (c == ',' || c == ';' || c == ' ' || c == '\t' || c == '\n' || c == '\r') {
				take(cur);
				cur.clear();
			} else {
				cur += c;
			}
		}
		take(cur);
	} else {
		if (fv.is_number()) {
			frames.push_back((int64_t)fv.as_number());
			return frames;
		}
		for (const JsonValue& v : fv.items()) {
			if (v.is_number()) frames.push_back((int64_t)v.as_number());
			else if (v.is_string()) take(v.as_string());
		}
	}
	return frames;
}

// Read the requested frames and build the tool result: one text part describing
// the video and which frames follow, then one jpeg image part per frame. Mimics
// the still-image read path, down to the re-encode and the size report.
ToolResult read_video_result(const std::vector<int64_t>& frames, const std::string& abs,
	ToolContext& ctx) {
	const std::string& lang = ctx.language;
	if (frames.empty()) {
		return error_result(tr(lang,
			"读取视频需要 frames：要看的帧号（从 1 开始，最多 4 个）",
			"reading a video needs `frames`: the frame numbers to look at (1-based, at most 4)"));
	}
	for (int64_t f : frames) {
		if (f < 1) {
			return error_result(tr(lang,
				"帧号从 1 开始（第 1 帧就是第一帧）",
				"frame numbers start at 1 (frame 1 is the first frame)"));
		}
	}
	// The caller's order is kept (that is the order the pictures come back in),
	// but repeats collapse to one frame and the list is capped at 4: at most four
	// images per call keeps one read from ballooning the transcript.
	std::vector<int64_t> wanted;
	bool truncated = false;
	for (int64_t f : frames) {
		if (std::find(wanted.begin(), wanted.end(), f) != wanted.end()) continue;
		if (wanted.size() == 4) {
			truncated = true;
			continue;
		}
		wanted.push_back(f);
	}

	VideoProbe probe;
	try {
		probe = tool_read_video_frames(abs, wanted);
	} catch (const std::exception& e) {
		return error_result(tr(lang, "无法读取视频: ", "cannot read video: ") + abs + " — " + e.what());
	}

	// geometry + rate + duration, the way the still path reports mime + size
	char dims[64];
	std::snprintf(dims, sizeof dims, "%lldx%lld", (long long)probe.width, (long long)probe.height);
	std::string meta = dims;
	if (probe.fps > 0.0) {
		char buf[64];
		std::snprintf(buf, sizeof buf, "%.4g", probe.fps);
		meta += std::string(", ") + buf + " fps";
	}
	if (probe.duration > 0.0) {
		char buf[64];
		std::snprintf(buf, sizeof buf, "%.4g", probe.duration);
		meta += std::string(", ") + buf + tr(lang, " 秒", " s");
	}
	if (probe.frame_count > 0)
		meta += ", " + std::to_string(probe.frame_count) + tr(lang, " 帧", " frames");

	std::string display = tool_display_path(ctx.cwd, abs);

	ToolResult r;
	std::string text = tr(lang, "视频: ", "Video: ") + display +
		tr(lang, "（" + meta + "）", " (" + meta + ")");

	std::vector<ContentPart> images;
	std::string sent;
	std::string missed;
	const bool have_count = probe.frame_count > 0;
	for (const VideoFrameImage& f : probe.frames) {
		const std::string label = tr(lang, "第 ", "frame ") + std::to_string(f.frame) + tr(lang, " 帧", "");
		// A frame number past the end of the file has no picture of its own.
		const bool past_end = have_count && f.frame > probe.frame_count;
		const std::string past_end_note = tr(lang,
			"（视频只有 " + std::to_string(probe.frame_count) + " 帧）",
			" (the video has only " + std::to_string(probe.frame_count) + " frames)");
		if (f.image.part.image_data.empty()) {
			if (!missed.empty()) missed += tr(lang, "、", ", ");
			missed += label + (past_end ? past_end_note : "");
			continue;
		}
		if (!sent.empty()) sent += tr(lang, "、", ", ");
		char tbuf[64];
		std::snprintf(tbuf, sizeof tbuf, "%.4g", f.seconds);
		sent += label + " (t=" + tbuf + tr(lang, " 秒", " s") + ")";
		// The request was answered with the last frame; say so, otherwise the
		// model would take it for the frame it asked for.
		if (past_end)
			sent += past_end_note + tr(lang, "，取到末帧", " — the last frame was taken");
		images.push_back(std::move(f.image.part));
	}

	if (images.empty()) {
		return error_result(tr(lang,
			"没有取到任何帧（视频共 " + std::to_string(probe.frame_count) + " 帧）: ",
			"could not decode any of the requested frames (video has " +
				std::to_string(probe.frame_count) + " frames): ") + display);
	}

	text += tr(lang, "\n已作为多模态 jpeg 图片提供给模型：",
		"\nsent to the model as multimodal jpeg images: ") + sent +
		tr(lang, "（共 " + std::to_string(images.size()) + " 张）",
			" (" + std::to_string(images.size()) + " in total)");
	if (truncated)
		text += tr(lang, "\n（一次最多 4 帧，只取了前 4 个帧号）",
			"\n(at most 4 frames per read — only the first 4 frame numbers were used)");
	if (!missed.empty())
		text += tr(lang, "\n未取到:", "\nnot decoded:") + " " + missed;

	r.content.push_back(ContentPart::text_part(text));
	for (ContentPart& img : images) r.content.push_back(std::move(img));
	return r;
}

ToolDef read_tool() {
	return make_tool(
		"read",
		"读取文本文件并显示行号，支持 offset/limit 读取大文件。也可读取 jpg jpeg png 图片与 mp4 视频的若干帧：图片和视频帧都会重新编码为 jpeg，再作为多模态内容提供给模型。读视频时用 frames 指定要看的帧号（从 1 开始，最多 4 帧）。请勿读取超出所需的内容。",
		"Read a text file with line numbers (offset/limit for large files). Also reads jpg / jpeg / png images and frames of an mp4 video: images and video frames are re-encoded as jpeg and sent to the model as multimodal content. For a video, pass `frames` with the 1-based frame numbers to look at (at most 4). Do not read more than needed",
		"读取文件、图片或视频帧",
		"Read a file, an image, or video frames",
		[](const std::string& lang) {
			return schema_object({
				{"path", schema_string(lang, "文件路径（绝对路径，或相对于工作目录）", "File path (absolute, or relative to the working directory)")},
				{"offset", schema_number(lang, "起始行号（从 1 开始）", "First line to read (1-based)")},
				{"limit", schema_number(lang, "最多返回的行数", "Maximum number of lines to return")},
				{"frames", schema_array(lang, schema_number(lang, "帧号", "frame number"),
					"读取视频时要看的帧号（从 1 开始，最多 4 帧；仅视频路径使用）",
					"Frame numbers to read from a video (1-based, at most 4; video paths only)")},
			}, {"path"});
		},
		[](const JsonValue& args, ToolContext& ctx) -> ToolResult {
			std::string path = arg_string(args, "path");
			if (path.empty()) {
				return error_result(tr(ctx.language, "path 不能为空", "path must not be empty"));
			}
			std::string abs = tool_resolve_path(ctx.cwd, path);
			if (std::string deny = tool_check_path_permission(ctx, abs); !deny.empty()) {
				return error_result(deny);
			}
			if (!path_exists(abs)) {
				return error_result(tr(ctx.language, "文件不存在: ", "file does not exist: ") + abs);
			}
			if (path_is_dir(abs)) {
				return error_result(tr(ctx.language,
					"这是目录，请用 ls 查看列表: ", "this is a directory, use ls to list it: ") + abs);
			}

			// images: hand the picture to the model as multimodal content instead of
			// trying to read it as text (which would only fail the binary check)
			std::string image_mime;
			if (image_mime_for_path(abs, image_mime)) {
				std::vector<uint8_t> raw;
				if (!read_file_bytes(abs, raw)) {
					return error_result(tr(ctx.language, "无法读取图片: ", "cannot read image: ") + abs);
				}
				if (raw.empty()) {
					return error_result(tr(ctx.language, "图片文件为空: ", "image file is empty: ") + abs);
				}
				std::string display = tool_display_path(ctx.cwd, abs);
				// What the model gets is a re-encoded copy, and the report quotes the
				// copy: the file is named (so a later edit can find it again), the
				// payload is described by what was actually sent. Saying that the
				// picture was compressed is what stops the model from reading the same
				// file over and over looking for the "full" version.
				ModelImage img = tool_model_image(raw, image_mime);
				const std::string info = img.mime + ", " + tool_human_size(img.bytes);
				const std::string tail = img.recompressed
					? tr(ctx.language,
						"图片已压缩后作为多模态内容提供给模型（原文件 " +
							tool_human_size((int64_t)raw.size()) + "）。",
						"sent to the model as multimodal image content, recompressed (the file on "
						"disk is " + tool_human_size((int64_t)raw.size()) + ").")
					: tr(ctx.language, "已作为多模态图片内容提供给模型。",
						"sent to the model as multimodal image content.");
				ToolResult r;
				r.content.push_back(ContentPart::text_part(tr(ctx.language,
					"图片: " + display + "（" + info + "）\n" + tail,
					"Image: " + display + " (" + info + ")\n" + tail)));
				r.content.push_back(std::move(img.part));
				return r;
			}

			// videos: decode the requested frames and hand them back as jpeg images,
			// the same way the still path above does for a picture
			if (video_ext_for_path(abs)) {
				return read_video_result(parse_frame_args(args), abs, ctx);
			}

			std::string bytes;
			if (!read_file(abs, bytes)) {
				return error_result(tr(ctx.language, "无法读取文件: ", "cannot read file: ") + abs);
			}
			if (tool_looks_binary(bytes.substr(0, std::min<size_t>(bytes.size(), 8192)))) {
				return error_result(tr(ctx.language,
					"二进制文件，无法以文本方式读取: ", "binary file, cannot be read as text: ") + abs);
			}
			// a file on disk is not guaranteed to be UTF-8 (a legacy GBK .txt reads
			// as a wall of replacement boxes otherwise) and whatever it holds must
			// not reach the JSON as invalid bytes — that used to close the UI
			// websocket the moment such a file was read
			bytes = decode_external_text(bytes);

			int64_t offset = std::max<int64_t>(1, arg_int(args, "offset", 1));
			int64_t limit = std::clamp<int64_t>(arg_int(args, "limit", 2000), 1, 5000);

			std::vector<std::string> lines = tool_split_lines(tool_strip_cr(bytes));
			int64_t total = (int64_t)lines.size();
			if (lines.empty()) {
				return text_result(tr(ctx.language, "(空文件)\n", "(empty file)\n"));
			}
			if (offset > total) {
				return error_result(tr(ctx.language,
					"offset " + std::to_string(offset) + " 超出文件行数（共 " + std::to_string(total) + " 行）",
					"offset " + std::to_string(offset) + " is past the end of the file (" +
						std::to_string(total) + " lines)"));
			}
			int64_t end = std::min(total, offset - 1 + limit);

			size_t width = std::max(std::to_string(total).size(),
								 std::to_string(end).size());
			std::string out;
			for (int64_t i = offset; i <= end; i++) {
				std::string line = lines[(size_t)(i - 1)];
				if (line.size() > 2000) {
					line = line.substr(0, 2000) +
						tr(ctx.language, " ... (单行已在 2000 字符处截断)", " ... (line truncated at 2000 chars)");
				}
				std::string num = std::to_string(i);
				out += std::string(width - num.size(), ' ') + num + "\t" + line + "\n";
			}
			if (end < total) {
				out += tr(ctx.language,
					"... (显示第 " + std::to_string(offset) + "-" + std::to_string(end) + " 行，共 " +
						std::to_string(total) + " 行；用 offset/limit 继续读取)\n",
					"... (showing lines " + std::to_string(offset) + "-" + std::to_string(end) +
						" of " + std::to_string(total) + "; use offset/limit to read more)\n");
			}
			return text_result(tool_truncate_middle(out, tool_output_limit(ctx, 120000), ctx.language));
		});
}

// ── write ──────────────────────────────────────────────────────────────────

ToolDef write_tool() {
	return make_tool(
		"write",
		"创建新文件或覆盖现有文件。父目录会自动创建。修改现有文件请优先使用 edit。",
		"Create a new file or overwrite an existing one with the given content. Parent directories are created automatically. Prefer edit for changing existing files.",
		"创建或覆盖文件",
		"Create or overwrite a file",
		[](const std::string& lang) {
			return schema_object({
				{"path", schema_string(lang, "文件路径（绝对路径，或相对于工作目录）", "File path (absolute, or relative to the working directory)")},
				{"content", schema_string(lang, "要写入的文件内容", "The file content to write")},
			}, {"path", "content"});
		},
		[](const JsonValue& args, ToolContext& ctx) -> ToolResult {
			std::string path = arg_string(args, "path");
			std::string content = arg_string(args, "content");
			if (path.empty()) {
				return error_result(tr(ctx.language, "path 不能为空", "path must not be empty"));
			}
			std::string abs = tool_resolve_path(ctx.cwd, path);
			if (std::string deny = tool_check_path_permission(ctx, abs); !deny.empty()) {
				return error_result(deny);
			}
			if (path_is_dir(abs)) {
				return error_result(tr(ctx.language, "目标路径是目录: ", "target path is a directory: ") + abs);
			}
			std::string dir = path_dirname(abs);
			if (!dir.empty() && !mkdirs(dir)) {
				return error_result(tr(ctx.language, "无法创建目录: ", "cannot create directory: ") + dir);
			}
			if (!write_file_atomic(abs, content)) {
				return error_result(tr(ctx.language, "写入失败: ", "write failed: ") + abs);
			}
			return text_result(tr(ctx.language, "已写入 ", "wrote ") +
				tool_display_path(ctx.cwd, abs));
		});
}

// ── edit ───────────────────────────────────────────────────────────────────

// count non-overlapping occurrences of `needle` in `text`
size_t count_occurrences(const std::string& text, const std::string& needle) {
	if (needle.empty()) return 0;
	size_t count = 0;
	for (size_t pos = text.find(needle); pos != std::string::npos; pos = text.find(needle, pos + needle.size())) {
		count++;
	}
	return count;
}

// count non-overlapping occurrences of `needle` in `text`, case-insensitively
size_t count_occurrences_ci(const std::string& text, const std::string& needle) {
	if (needle.empty()) return 0;
	std::string lt = to_lower(text), ln = to_lower(needle);
	size_t count = 0;
	for (size_t pos = lt.find(ln); pos != std::string::npos; pos = lt.find(ln, pos + ln.size())) {
		count++;
	}
	return count;
}

// ── tolerant matching for edit ──────────────────────────────────────────
// An exact-match-only edit fails on the trivial mismatches a model keeps
// producing: a CRLF checkout against LF in the snippet, a stray trailing space,
// or indentation it reproduced slightly wrong. These helpers retry the match
// after normalising those, and report the span in the ORIGINAL text so the
// replacement is still applied in place.

struct EditSpan {
	size_t pos = 0;
	size_t len = 0;
	bool found = false;
};

// normalise `text`; src[i] is the index in `text` of the i-th result byte.
// mode 1: CRLF -> LF and per-line trailing whitespace dropped.
// mode 2: leading whitespace dropped too (indentation-tolerant).
std::string normalize_for_match(const std::string& text, int mode, std::vector<size_t>& src) {
	std::string out;
	src.clear();
	size_t i = 0, n = text.size();
	while (i < n) {
		size_t nl = text.find('\n', i);
		bool has_nl = nl != std::string::npos;
		size_t line_end = has_nl ? nl : n;
		size_t b = i, e = line_end;
		while (e > b && (text[e - 1] == ' ' || text[e - 1] == '\t' || text[e - 1] == '\r')) e--;
		if (mode >= 2) {
			while (b < e && (text[b] == ' ' || text[b] == '\t')) b++;
		}
		for (size_t k = b; k < e; k++) {
			out.push_back(text[k]);
			src.push_back(k);
		}
		if (has_nl) {
			out.push_back('\n');
			src.push_back(line_end);
		}
		i = has_nl ? line_end + 1 : n;
	}
	return out;
}

size_t count_in(const std::string& hay, const std::string& needle) {
	if (needle.empty()) return 0;
	size_t c = 0;
	for (size_t p = hay.find(needle); p != std::string::npos; p = hay.find(needle, p + needle.size())) c++;
	return c;
}

// retry an exact-match failure tolerantly. The needle is tried as given and with
// trailing newlines trimmed; only a UNIQUE match counts (0 = miss, >1 = too
// ambiguous to edit safely). Returns the span in `content`.
bool tolerant_match(const std::string& content, const std::string& old_text, bool ci,
	int mode, EditSpan& out) {
	std::vector<std::string> needles{old_text};
	std::string trimmed = old_text;
	while (!trimmed.empty() && (trimmed.back() == '\n' || trimmed.back() == '\r')) trimmed.pop_back();
	if (trimmed != old_text) needles.push_back(trimmed);

	std::vector<size_t> cs;
	std::string nc = normalize_for_match(content, mode, cs);
	std::string hc = ci ? to_lower(nc) : nc;
	for (const auto& needle : needles) {
		std::vector<size_t> os;
		std::string no = normalize_for_match(needle, mode, os);
		if (no.empty()) continue;
		std::string ho = ci ? to_lower(no) : no;
		if (count_in(hc, ho) != 1) continue;
		size_t p = hc.find(ho);
		size_t start = cs[p];
		size_t end = (p + ho.size() < cs.size()) ? cs[p + ho.size()] : content.size();
		if (mode >= 2) {
			// include the matched line's leading whitespace so the replacement is not
			// duplicated after it
			while (start > 0 && content[start - 1] != '\n') start--;
		}
		out = {start, end - start, true};
		return true;
	}
	return false;
}

ToolDef edit_tool() {
	return make_tool(
		"edit",
		"对文件进行精确的字符串替换。传入一组编辑操作；每个 oldText 必须在文件中精确匹配唯一一处（包含足够的上下文）。所有编辑会原子性地一次性写入；如果任何编辑未匹配，则不会进行任何更改。",
		"Make exact string replacements in a file. Pass a list of edits; each oldText must match exactly ONE unique location in the file (include enough surrounding context). All edits are applied in one atomic write; if any edit fails to match, nothing is changed.",
		"精确替换文件中的文本",
		"Replace exact text in a file",
		[](const std::string& lang) {
			return schema_object({
				{"path", schema_string(lang, "文件路径（绝对路径，或相对于工作目录）", "File path (absolute, or relative to the working directory)")},
				{"edits", schema_array(lang, schema_object({
					{"oldText", schema_string(lang, "要替换的精确文本（必须在文件中唯一）", "The exact text to replace (must be unique in the file)")},
					{"newText", schema_string(lang, "替换的文本", "The replacement text")},
				}, {"oldText", "newText"}), "一组目标替换操作", "A list of targeted replacements")},
			}, {"path", "edits"});
		},
		[](const JsonValue& args, ToolContext& ctx) -> ToolResult {
			std::string path = arg_string(args, "path");
			if (path.empty()) {
				return error_result(tr(ctx.language, "path 不能为空", "path must not be empty"));
			}
			std::string abs = tool_resolve_path(ctx.cwd, path);
			if (std::string deny = tool_check_path_permission(ctx, abs); !deny.empty()) {
				return error_result(deny);
			}
			if (!path_exists(abs)) {
				return error_result(tr(ctx.language, "文件不存在: ", "file does not exist: ") + abs);
			}
			if (path_is_dir(abs)) {
				return error_result(tr(ctx.language, "目标路径是目录: ", "target path is a directory: ") + abs);
			}
			const JsonValue* edits = args.find("edits");
			if (!edits || !edits->is_array() || edits->size() == 0) {
				return error_result(tr(ctx.language,
					"edits 必须是非空数组（每项含 oldText/newText）",
					"edits must be a non-empty array of objects with oldText/newText"));
			}
			bool ci = args.find("case_insensitive") && args.find("case_insensitive")->as_bool(false);

			std::string content;
			if (!read_file(abs, content)) {
				return error_result(tr(ctx.language, "无法读取文件: ", "cannot read file: ") + abs);
			}
			if (tool_looks_binary(content.substr(0, std::min<size_t>(content.size(), 8192)))) {
				return error_result(tr(ctx.language,
					"二进制文件不支持编辑: ", "binary files cannot be edited: ") + abs);
			}

			// validate + apply in memory; any failure aborts the whole call
			std::string updated = content;
			for (size_t i = 0; i < edits->size(); i++) {
				const JsonValue& e = (*edits)[i];
				std::string old_text = e["oldText"].as_string("");
				std::string new_text = e["newText"].as_string("");
				if (old_text.empty()) {
					return error_result(tr(ctx.language,
						"edit " + std::to_string(i) + ": oldText 不能为空",
						"edit " + std::to_string(i) + ": oldText must not be empty"));
				}
				size_t n = ci ? count_occurrences_ci(updated, old_text) : count_occurrences(updated, old_text);
				EditSpan span;
				if (n == 1) {
					size_t pos = ci ? to_lower(updated).find(to_lower(old_text)) : updated.find(old_text);
					span = {pos, old_text.size(), true};
				} else if (n > 1) {
					return error_result(tr(ctx.language,
						"edit " + std::to_string(i) + ": oldText 匹配到 " + std::to_string(n) +
							" 处，必须唯一 — 请补充更多上下文使其唯一，或拆分多次调用",
						"edit " + std::to_string(i) + ": oldText matched " + std::to_string(n) +
							" places but must be unique — add more surrounding context to make it unique, or split it into several calls"));
				} else if (!tolerant_match(updated, old_text, ci, 1, span) &&
				           !tolerant_match(updated, old_text, ci, 2, span)) {
					// exact match missed and no tolerant fallback: report it with the hint
					// that CRLF / whitespace / indentation differences are forgiven
					std::string preview = old_text.substr(0, 120);
					return error_result(tr(ctx.language,
						"edit " + std::to_string(i) + ": oldText 在文件中未找到（已尝试忽略换行符 / 行尾空白 / 缩进差异）。片段: \"" +
							preview + (old_text.size() > 120 ? "\"..." : "\""),
						"edit " + std::to_string(i) + ": oldText was not found in the file (already retried ignoring line endings / trailing whitespace / indentation). Snippet: \"" +
							preview + (old_text.size() > 120 ? "\"..." : "\"")));
				}
				// keep the file's newline style when the replacement is multi-line
				if (contains(updated, "\r\n") && !contains(new_text, "\r\n")) {
					new_text = replace_all(new_text, "\n", "\r\n");
				}
				updated = updated.substr(0, span.pos) + new_text + updated.substr(span.pos + span.len);
			}

			if (!write_file_atomic(abs, updated)) {
				return error_result(tr(ctx.language, "写入失败: ", "write failed: ") + abs);
			}
			return text_result(tr(ctx.language,
				"已应用 " + std::to_string(edits->size()) + " 处编辑：",
				"applied " + std::to_string(edits->size()) + " edit(s): ") + tool_display_path(ctx.cwd, abs));
		});
}

// ── ls ─────────────────────────────────────────────────────────────────────

struct LsEntry {
	std::string rel;    // path relative to the root (with / separators)
	bool is_dir = false;
	int64_t size = 0;
};

void walk_listing(const std::string& root, const std::string& rel_prefix, bool recursive,
	size_t max_entries, std::vector<LsEntry>& out, bool& truncated) {
	if (out.size() >= max_entries) {
		truncated = true;
		return;
	}
	std::string dir = rel_prefix.empty() ? root : root + "/" + rel_prefix;
	auto names = list_dir(dir);
	std::sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) {
		std::string la = to_lower(a), lb = to_lower(b);
		return la < lb;
	});
	for (const auto& name : names) {
		if (out.size() >= max_entries) {
			truncated = true;
			return;
		}
		std::string full = dir + "/" + name;
		std::string rel = rel_prefix.empty() ? name : rel_prefix + "/" + name;
		bool is_dir = path_is_dir(full);
		// skip VCS / dependency noise in recursive mode
		if (recursive && is_dir && (name == ".git" || name == "node_modules")) continue;
		LsEntry e;
		e.rel = std::move(rel);
		e.is_dir = is_dir;
		out.push_back(std::move(e));
		if (is_dir && recursive) {
			walk_listing(root, out.back().rel, recursive, max_entries, out, truncated);
		}
	}
}

ToolDef ls_tool() {
	return make_tool(
		"ls",
		"列出目录内容：文件名、类型和文件大小。使用 recursive 递归遍历子目录（跳过 .git / node_modules）。",
		"List the contents of a directory: names, types and file sizes. Use recursive to walk subdirectories (skips .git / node_modules).",
		"列出目录内容",
		"List the contents of a directory",
		[](const std::string& lang) {
			return schema_object({
				{"path", schema_string(lang, "目录路径", "Directory path")},
				{"recursive", schema_boolean(lang, "是否递归列出子目录", "Whether to list subdirectories recursively")},
			}, {});
		},
		[](const JsonValue& args, ToolContext& ctx) -> ToolResult {
			std::string path = arg_string(args, "path");
			bool recursive = args.find("recursive") && args.find("recursive")->as_bool(false);
			std::string abs = path.empty() ? path_absolute(ctx.cwd) : tool_resolve_path(ctx.cwd, path);
			if (std::string deny = tool_check_path_permission(ctx, abs); !deny.empty()) {
				return error_result(deny);
			}
			if (!path_exists(abs)) {
				return error_result(tr(ctx.language, "路径不存在: ", "path does not exist: ") + abs);
			}
			if (!path_is_dir(abs)) {
				return error_result(tr(ctx.language, "不是目录: ", "not a directory: ") + abs);
			}

			std::vector<LsEntry> entries;
			bool truncated = false;
			const size_t kMax = recursive ? 2000 : 500;
			walk_listing(abs, "", recursive, kMax, entries, truncated);

			if (entries.empty()) {
				return text_result(tr(ctx.language, "(空目录) ", "(empty directory) ") +
					tool_display_path(ctx.cwd, abs));
			}
			std::string out = tr(ctx.language, "目录: ", "Listing: ") + tool_display_path(ctx.cwd, abs) +
				(recursive ? tr(ctx.language, "（递归）", " (recursive)") : "") + "\n";
			for (const auto& e : entries) {
				if (e.is_dir) {
					out += e.rel + "/\n";
				} else {
					int64_t size = tool_file_size(abs + "/" + e.rel);
					out += e.rel + (size >= 0 ? "  (" + tool_human_size(size) + ")" : "") + "\n";
				}
			}
			out += tr(ctx.language,
				std::to_string(entries.size()) + " 项",
				std::to_string(entries.size()) + " entries") +
				(truncated ? tr(ctx.language, "（已截断）", " (truncated)") : "");
			return text_result(tool_truncate_middle(out, tool_output_limit(ctx, 60000), ctx.language));
		});
}

// ── grep ───────────────────────────────────────────────────────────────────

// native fallback search (used when the vendored rg.exe is unavailable)
std::string grep_native(const std::string& language, const std::string& pattern,
	const std::string& root, bool is_dir,
	const std::string& include_glob, bool ignore_case, bool literal,
	const std::string& output_mode, int context_lines, bool& matched) {
	std::regex re;
	bool regex_ok = true;
	if (!literal) {
		try {
			auto flags = std::regex::ECMAScript;
			if (ignore_case) flags |= std::regex::icase;
			re.assign(pattern, flags);
		} catch (const std::exception&) {
			regex_ok = false;
		}
	} else if (ignore_case) {
		try {
			re.assign(std::regex_replace(pattern, std::regex("[-[\\]{}()*+?.,\\^$|#\\s]"), "\\$&"),
				std::regex::icase);
		} catch (const std::exception&) {
			regex_ok = false;
		}
	}
	if (!regex_ok) return tr(language, "无效的正则表达式: ", "invalid regex pattern: ") + pattern;

	std::string lower_pattern = to_lower(pattern);
	auto line_matches = [&](const std::string& line) {
		if (literal && !ignore_case) return contains(line, pattern);
		if (literal) return contains(to_lower(line), lower_pattern);
		return std::regex_search(line, re);
	};

	std::vector<std::string> files;
	if (!is_dir) {
		files.push_back(root);
	} else {
		// walk (shallow-bound), skipping .git / node_modules and binary files
		std::vector<std::string> stack{root};
		while (!stack.empty() && files.size() < 4000) {
			std::string dir = stack.back();
			stack.pop_back();
			for (const auto& name : list_dir(dir)) {
				std::string full = dir + "/" + name;
				if (path_is_dir(full)) {
					if (name == ".git" || name == "node_modules") continue;
					stack.push_back(full);
					continue;
				}
				if (!include_glob.empty() && !tool_glob_match(include_glob, name)) continue;
				files.push_back(full);
			}
		}
	}

	std::string out;
	long match_count = 0;
	const long kMaxMatches = 2000;
	for (const auto& file : files) {
		if (match_count >= kMaxMatches) break;
		std::string bytes;
		if (!read_file(file, bytes)) continue;
		if (bytes.empty()) continue;
		if (tool_looks_binary(bytes.substr(0, std::min<size_t>(bytes.size(), 8192)))) continue;
		bytes = decode_external_text(bytes);  // legacy encodings -> valid UTF-8
		std::vector<std::string> lines = tool_split_lines(tool_strip_cr(bytes));
		bool file_matched = false;
		long file_count = 0;
		std::string file_out;
		for (size_t i = 0; i < lines.size(); i++) {
			if (!line_matches(lines[i])) continue;
			file_matched = true;
			match_count++;
			file_count++;
			if (output_mode == "files_with_matches") break;
			size_t from = i > (size_t)context_lines ? i - context_lines : 0;
			size_t to = std::min(lines.size() - 1, i + (size_t)context_lines);
			for (size_t k = from; k <= to; k++) {
				file_out += file + ":" + std::to_string(k + 1) + ": " + lines[k] + "\n";
			}
			if (context_lines > 0 && to < lines.size() - 1) file_out += "---\n";
			if (match_count >= kMaxMatches) break;
		}
		if (file_matched) {
			matched = true;
			if (output_mode == "files_with_matches") {
				out += file + "\n";
			} else if (output_mode == "count") {
				out += file + ":" + std::to_string(file_count) + "\n";
			} else {
				out += file_out;
			}
		}
	}
	if (match_count >= kMaxMatches) {
		out += tr(language,
			"(已达到 " + std::to_string(kMaxMatches) + " 条匹配，停止搜索)\n",
			"(stopped at " + std::to_string(kMaxMatches) + " matches)\n");
	}
	return out;
}

// quote one argument for the shell that spawn_shell uses (bash or cmd):
// wraps in double quotes and escapes \, " and $ ` ! so a pattern coming from
// the model can never terminate the quoting and inject commands
static std::string win_quote_shell_arg(const std::string& arg) {
	std::string out = "\"";
	for (char c : arg) {
		switch (c) {
			case '\\': out += "\\\\"; break;
			case '"':  out += "\\\""; break;
			case '$':  out += "\\$"; break;
			case '`':  out += "\\`"; break;
			case '!':  out += "\\!"; break;
			default:   out.push_back(c); break;
		}
	}
	out.push_back('"');
	return out;
}

ToolDef grep_tool() {
	return make_tool(
		"grep",
		"在文件内容中搜索文本（默认正则表达式，literal:true 时作为纯文本处理）。可通过 include glob 过滤文件名（如 \"*.cpp\"）。输出模式：content（默认，路径:行号: 文本）、files_with_matches、count。",
		"Search file contents for a pattern (regex by default, literal with literal:true). Narrow with include globs like \"*.cpp\". Output modes: content (default, path:line: text), files_with_matches, count.",
		"搜索文件内容（正则、glob 过滤）",
		"Search file contents (regex, glob filter)",
		[](const std::string& lang) {
			return schema_object({
				{"pattern", schema_string(lang, "要搜索的文本或正则表达式", "The text or regular expression to search for")},
				{"path", schema_string(lang, "要搜索的文件或目录", "The file or directory to search")},
				{"include", schema_string(lang, "文件名 glob 过滤，例如 \"*.ts\" 或 \"*.{h,cpp}\"", "Filename glob filter, e.g. \"*.ts\" or \"*.{h,cpp}\"")},
				{"literal", schema_boolean(lang, "将模式视为纯文本", "Treat the pattern as plain text")},
				{"case_insensitive", schema_boolean(lang, "忽略大小写", "Ignore case")},
				{"output_mode", schema_string(lang, "输出模式：content、files_with_matches 或 count", "Output mode: content, files_with_matches or count")},
				{"context", schema_number(lang, "每个匹配点前后的上下文行数", "Number of context lines before and after each match")},
			}, {"pattern"});
		},
		[](const JsonValue& args, ToolContext& ctx) -> ToolResult {
			std::string pattern = arg_string(args, "pattern");
			if (pattern.empty()) {
				return error_result(tr(ctx.language, "pattern 不能为空", "pattern must not be empty"));
			}
			std::string path = arg_string(args, "path");
			std::string include = arg_string(args, "include");
			bool literal = args.find("literal") && args.find("literal")->as_bool(false);
			bool ignore_case = args.find("case_insensitive") && args.find("case_insensitive")->as_bool(false);
			std::string output_mode = arg_string(args, "output_mode", "content");
			int context = std::clamp<int>(std::max<int64_t>(0, arg_int(args, "context", 0)), 0, 5);

			std::string abs = path.empty() ? path_absolute(ctx.cwd) : tool_resolve_path(ctx.cwd, path);
			if (std::string deny = tool_check_path_permission(ctx, abs); !deny.empty()) {
				return error_result(deny);
			}
			if (!path_exists(abs)) {
				return error_result(tr(ctx.language, "路径不存在: ", "path does not exist: ") + abs);
			}
			bool is_dir = path_is_dir(abs);

			// prefer the vendored ripgrep; fall back to the native walker
			std::string rg = path_join(ctx.bin_dir, "rg.exe");
			if (path_exists(rg)) {
				std::string cmdline = "\"" + path_backslash(rg) + "\" --no-heading --path-separator / -n";
				if (ignore_case) cmdline += " -i";
				if (literal) cmdline += " -F";
				if (output_mode == "files_with_matches") cmdline += " --files-with-matches";
				else if (output_mode == "count") cmdline += " -c";
				else if (context > 0) cmdline += " -C " + std::to_string(context);
				if (!include.empty()) cmdline += " -g " + win_quote_shell_arg(include);
				// pattern comes from the MODEL and may contain quotes/$/backticks:
				// naive \"...\" wrapping let a pattern like `"; rm -rf ~; "` escape
				// the quoting and run arbitrary commands inside the shell
				cmdline += " -- " + win_quote_shell_arg(pattern) + " " + win_quote_shell_arg(path_backslash(abs));
				ToolExecResult r = tool_run_shell(cmdline, ctx.cwd, 30000, ctx.cancelled, ctx.language);
				if (!r.spawn_failed && r.exit_code <= 1) {
					std::string out = tool_strip_cr(r.output);
					if (trim(out).empty()) {
						return text_result(tr(ctx.language,
							"没有找到匹配项: ", "No matches found for: ") + pattern);
					}
					return text_result(tool_truncate_middle(out, tool_output_limit(ctx, 60000), ctx.language));
				}
				// rg missing/failed at runtime → fall through to native search
			}

			bool matched = false;
			std::string out = grep_native(ctx.language, pattern, abs, is_dir, include, ignore_case, literal,
				output_mode, context, matched);
			// the native walker prefixes a bad pattern with this localized marker
			std::string invalid = tr(ctx.language, "无效的正则表达式", "invalid regex");
			if (!matched && out.find(invalid) == std::string::npos) {
				return text_result(tr(ctx.language,
					"没有找到匹配项: ", "No matches found for: ") + pattern);
			}
			return text_result(tool_truncate_middle(out, tool_output_limit(ctx, 60000), ctx.language),
				out.find(invalid) == 0);
		});
}

}  // namespace

std::vector<ToolDef> build_file_tools() {
	return {
		bash_tool(),
		read_tool(),
		write_tool(),
		edit_tool(),
		ls_tool(),
		grep_tool(),
	};
}

}  // namespace phi
