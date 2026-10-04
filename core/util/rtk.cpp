#include "util/rtk.hpp"

#include <algorithm>
#include <cctype>
#include <set>

#include "util/base.hpp"
#include "util/i18n.hpp"
#include "tools/tools_helpers.hpp"

namespace phi {
namespace rtk {

namespace {

// first tokens of a command, lowercased (env assignments skipped)
std::vector<std::string> command_head(const std::string& command) {
	std::vector<std::string> out;
	std::string line = tool_strip_cr(command);
	for (auto& c : line) c = (char)std::tolower((unsigned char)c);
	size_t start = 0;
	while (start < line.size() && (int)out.size() < 4) {
		size_t end = line.find_first_of(" \t|;&", start);
		std::string tok = line.substr(start, end == std::string::npos ? std::string::npos : end - start);
		if (!tok.empty() && tok != "sudo") out.push_back(tok);
		if (end == std::string::npos) break;
		start = end + 1;
	}
	return out;
}

bool is_build_command(const std::vector<std::string>& head) {
	if (head.empty()) return false;
	const std::string& c0 = head[0];
	auto starts = [&](const char* p) { return starts_with(c0, p); };
	if (c0 == "cargo" || c0 == "make" || c0 == "cmake" || c0 == "gradle" || c0 == "mvn" ||
		c0 == "tsc" || c0 == "g++" || c0 == "gcc" || c0 == "clang" || c0 == "clang++" ||
		c0 == "go" || c0 == "rustc" || c0 == "dotnet" || c0 == "mingw32-make")
		return true;
	if (starts("npm") || starts("pnpm") || starts("yarn") || starts("bun")) return true;
	return false;
}

bool is_git_status(const std::vector<std::string>& head) {
	return head.size() >= 2 && head[0] == "git" && (head[1] == "status" || head[1] == "stash");
}

bool is_git_log(const std::vector<std::string>& head) {
	return head.size() >= 2 && head[0] == "git" && head[1] == "log";
}

bool is_diff_output(const std::string& text) {
	return contains(text, "diff --git ") || contains(text, "\ndiff --git ");
}

}  // namespace

// ── ANSI stripping ─────────────────────────────────────────────────────────

std::string strip_ansi(const std::string& text) {
	if (text.find('\x1b') == std::string::npos) return text;
	std::string out;
	out.reserve(text.size());
	size_t i = 0;
	while (i < text.size()) {
		char c = text[i];
		if (c != '\x1b') {
			out.push_back(c);
			i++;
			continue;
		}
		// ESC [ ... letter  (CSI)
		if (i + 1 < text.size() && text[i + 1] == '[') {
			size_t j = i + 2;
			while (j < text.size() && ((unsigned char)text[j] < 0x40 || (unsigned char)text[j] > 0x7E)) j++;
			if (j < text.size()) {
				i = j + 1;
				continue;
			}
			i = text.size();
			continue;
		}
		// ESC ] ... BEL / ESC\  (OSC)
		if (i + 1 < text.size() && text[i + 1] == ']') {
			size_t j = i + 2;
			while (j < text.size() && text[j] != '\x07' && !(text[j] == '\x1b' && j + 1 < text.size() && text[j + 1] == '\\')) j++;
			if (j < text.size()) {
				i = (text[j] == '\x07') ? j + 1 : j + 2;
				continue;
			}
			i = text.size();
			continue;
		}
		// other single-char escapes
		i += (i + 1 < text.size()) ? 2 : 1;
	}
	return out;
}

// ── git compaction ─────────────────────────────────────────────────────────

namespace {

// diff → per-file summary: header, hunk heads, +/- lines (capped), totals
std::string compact_diff(const std::string& output, const std::string& language) {
	std::vector<std::string> lines = tool_split_lines(output);
	std::vector<std::string> result;
	std::string current_file;
	long added = 0, removed = 0;
	bool in_hunk = false;
	int hunk_lines = 0;
	const size_t kMaxResultLines = 60;
	const int kMaxHunkLines = 10;
	// once the output cap is hit, counting stops — the totals are then a
	// lower bound, so they are flagged "~" to avoid presenting them as exact
	bool truncated = false;

	auto flush_file = [&]() {
		if (!current_file.empty() && (added > 0 || removed > 0)) {
			result.push_back("  +" + std::to_string(added) + (truncated ? "~" : "") + " -" +
				std::to_string(removed) + (truncated ? "~" : ""));
		}
	};

	for (const auto& line : lines) {
		if (result.size() >= kMaxResultLines) {
			// stop counting here: the flushed totals are then a lower bound,
			// flagged "~" below instead of being presented as exact
			truncated = true;
			break;
		}
		if (starts_with(line, "diff --git")) {
			flush_file();
			size_t b = line.find(" b/");
			current_file = b == std::string::npos ? "unknown" : line.substr(b + 3);
			result.push_back("");
			result.push_back("> " + current_file);
			added = removed = 0;
			in_hunk = false;
			continue;
		}
		if (starts_with(line, "@@")) {
			in_hunk = true;
			hunk_lines = 0;
			size_t end = line.find("@@", 2);
			result.push_back("  " + (end == std::string::npos ? "@@" : line.substr(0, end + 2)));
			continue;
		}
		if (!in_hunk) continue;
		if (starts_with(line, "+") && !starts_with(line, "+++")) {
			if (!truncated) added++;
			if (hunk_lines < kMaxHunkLines) {
				result.push_back("  " + line);
				hunk_lines++;
			}
		} else if (starts_with(line, "-") && !starts_with(line, "---")) {
			if (!truncated) removed++;
			if (hunk_lines < kMaxHunkLines) {
				result.push_back("  " + line);
				hunk_lines++;
			}
		} else if (!starts_with(line, "\\") && hunk_lines < kMaxHunkLines && hunk_lines > 0) {
			result.push_back("  " + line);
			hunk_lines++;
		}
		if (hunk_lines == kMaxHunkLines) {
			result.push_back(tr(language, "  ...（已省略）", "  ... (truncated)"));
			hunk_lines++;
		}
	}
	flush_file();
	if (truncated)
		result.push_back(tr(language, "\n... （更多改动已省略，计数为下限）",
			"\n... (more changes omitted; counts are lower bounds)"));
	return join(result, "\n");
}

// status → branch line + staged/modified/untracked counts with samples
std::string compact_status(const std::string& output, const std::string& language) {
	std::vector<std::string> lines = tool_split_lines(output);
	if (lines.empty() || (lines.size() == 1 && trim(lines[0]).empty())) {
		return tr(language, "工作区干净", "Clean working tree");
	}
	std::vector<std::string> staged, modified, untracked, conflicts;
	std::string branch;
	for (const auto& line : lines) {
		if (starts_with(line, "##")) {
			std::string rest = trim(line.substr(2));
			size_t dots = rest.find("...");
			branch = dots == std::string::npos ? rest : rest.substr(0, dots);
			continue;
		}
		if (line.size() < 2) continue;
		char x = line[0], y = line[1];
		std::string file = trim(line.substr(2));
		if (file.empty()) continue;
		if (x == '?' && y == '?') {
			untracked.push_back(file);
		} else if (x == 'U' || y == 'U' || (x == 'A' && y == 'A') || (x == 'D' && y == 'D')) {
			conflicts.push_back(file);
		} else if (x != ' ') {
			staged.push_back(file);
		} else if (y != ' ') {
			modified.push_back(file);
		}
	}
	std::vector<std::string> out;
	if (!branch.empty()) out.push_back(tr(language, "分支 ", "Branch: ") + branch);
	auto section = [&](const char* label_zh, const char* label_en, std::vector<std::string>& files) {
		if (files.empty()) return;
		out.push_back(tr(language, std::string(label_zh) + " " + std::to_string(files.size()) + " 个：",
			std::string(label_en) + " (" + std::to_string(files.size()) + "):"));
		for (size_t i = 0; i < files.size() && i < 10; i++) out.push_back("  " + files[i]);
		if (files.size() > 10)
			out.push_back(tr(language, "  ... 另有 " + std::to_string(files.size() - 10) + " 个",
				"  ... and " + std::to_string(files.size() - 10) + " more"));
	};
	section("已暂存", "Staged", staged);
	section("已修改", "Modified", modified);
	section("未跟踪", "Untracked", untracked);
	section("冲突", "Conflicts", conflicts);
	return join(out, "\n");
}

}  // namespace

std::string compact_git(const std::string& output, const std::string& command,
	const std::string& language) {
	auto head = command_head(command);
	if (output.empty()) return output;
	if (is_git_status(head)) return compact_status(output, language);
	if (is_git_log(head)) {
		std::vector<std::string> lines = tool_split_lines(output);
		if (lines.size() > 50) {
			std::vector<std::string> kept(lines.begin(), lines.begin() + 50);
			kept.push_back(tr(language,
				"... （还有 " + std::to_string(lines.size() - 50) + " 行）",
				"... (" + std::to_string(lines.size() - 50) + " more lines)"));
			return join(kept, "\n");
		}
		return output;
	}
	// diff / show / stash show: compact when it looks like a diff
	if (is_diff_output(output)) return compact_diff(output, language);
	return output;
}

// ── build output filtering ─────────────────────────────────────────────────

namespace {

bool is_skip_line(const std::string& line) {
	static const char* prefixes[] = {
		"Compiling ", "Checking ", "Downloading ", "Downloaded ", "Fetching ", "Fetched ",
		"Updating ", "Updated ", "Building ", "Generated ", "Creating ", "Running ",
		"Finished ", "Packaging ", "Installing "};
	std::string t = trim(line);
	if (t.empty()) return false;
	for (const char* p : prefixes) {
		if (starts_with(t, p)) return true;
	}
	return false;
}

bool is_error_line(const std::string& line) {
	static const char* prefixes[] = {"error[", "error:", "ERROR:", "[ERROR]", "FAILED", "fatal:", "Fatal:"};
	std::string t = trim(line);
	for (const char* p : prefixes) {
		if (starts_with(t, p)) return true;
	}
	if (contains(line, ": error:") || contains(line, ": fatal error") ||
		contains(line, "undefined reference")) return true;
	// MSVC linker errors look like "... : error LNK2019: ..." — require the
	// "error LNK<digits>" token so ordinary prose mentioning LNK or error
	// (e.g. "LNK error") is not flagged
	size_t pos = 0;
	while ((pos = line.find("error LNK", pos)) != std::string::npos) {
		size_t after = pos + 9;
		if (after < line.size() && isdigit((unsigned char)line[after])) return true;
		pos = after;
	}
	return false;
}

bool is_warning_line(const std::string& line) {
	static const char* prefixes[] = {"warning:", "WARNING:", "[WARNING]", "warn:"};
	std::string t = trim(line);
	for (const char* p : prefixes) {
		if (starts_with(t, p)) return true;
	}
	return contains(line, ": warning:");
}

}  // namespace

std::string filter_build_output(const std::string& output, const std::string& command,
	const std::string& language) {
	auto head = command_head(command);
	if (!is_build_command(head)) return output;
	std::vector<std::string> lines = tool_split_lines(output);
	std::vector<std::string> out;
	long skipped = 0;
	long compiled = 0;
	for (size_t i = 0; i < lines.size(); i++) {
		const std::string& line = lines[i];
		if (contains(line, "Compiling ") || contains(line, "Building ") || contains(line, "Checking ")) {
			compiled++;
			continue;
		}
		if (is_skip_line(line)) {
			skipped++;
			continue;
		}
		if (is_error_line(line)) {
			out.push_back(line);
			// keep one line of context after an error (often the source snippet)
			if (i + 1 < lines.size() && !trim(lines[i + 1]).empty()) {
				out.push_back(lines[i + 1]);
				i++;
			}
			continue;
		}
		if (is_warning_line(line)) {
			out.push_back(line);
			continue;
		}
		// keep short progress summaries like "Build succeeded." / "0 errors"
		if (contains(line, "error") || contains(line, "succeeded") || contains(line, "failed")) {
			out.push_back(line);
			continue;
		}
		skipped++;
	}
	if (out.empty()) {
		return tr(language,
			"（构建输出：" + std::to_string(compiled) + " 步，已过滤 " + std::to_string(skipped) +
				" 行进度输出，无错误）",
			"(build output: " + std::to_string(compiled) + " steps, " + std::to_string(skipped) +
				" progress lines filtered, no errors)");
	}
	std::string result = join(out, "\n");
	if (skipped > 0) {
		result += tr(language, "\n（已过滤 " + std::to_string(skipped) + " 行进度输出）",
			"\n(" + std::to_string(skipped) + " progress lines filtered)");
	}
	return result;
}

// ── misc stages ────────────────────────────────────────────────────────────

std::string collapse_blank_lines(const std::string& text) {
	std::string out;
	out.reserve(text.size());
	int blanks = 0;
	std::vector<std::string> lines = tool_split_lines(text);
	for (size_t i = 0; i < lines.size(); i++) {
		if (trim(lines[i]).empty()) {
			blanks++;
			if (blanks > 1) continue;
		} else {
			blanks = 0;
		}
		out += lines[i];
		out += "\n";
	}
	while (!out.empty() && out.back() == '\n') out.pop_back();
	return out;
}

std::string hard_truncate(const std::string& text, size_t max_chars, const std::string& language) {
	return tool_truncate_middle(text, max_chars, language);
}

// ── pipeline ───────────────────────────────────────────────────────────────

std::string compact_output(const std::string& raw, const std::string& command,
	const std::string& language, const Options& options) {
	std::string text = tool_strip_cr(raw);
	if (options.strip_ansi) text = strip_ansi(text);
	if (options.compact_git) text = compact_git(text, command, language);
	if (options.filter_build) text = filter_build_output(text, command, language);
	if (options.collapse_blank_lines) text = collapse_blank_lines(text);
	return hard_truncate(text, options.max_chars, language);
}

}  // namespace rtk
}  // namespace phi
