// Built-in rtk: tool-output compaction, in-process (the old external rtk
// module is gone — the bash tool compacts its own output here).
//
// Pipeline (inspired by pi-rtk-optimizer, ported natively):
//   1. CRLF + ANSI escape stripping
//   2. command-aware compaction: git (status/diff/log), build output filtering
//   3. blank-line collapsing
//   4. hard truncation (head + tail preserved, middle replaced with a marker)
//
// Every prose line the pipeline inserts is a (Chinese, English) pair picked by
// the session language (see i18n.hpp), so compacted output stays in the same
// language as the tool that produced it.
#pragma once

#include <string>

namespace phi {
namespace rtk {

struct Options {
	bool strip_ansi = true;
	bool compact_git = true;
	bool filter_build = true;
	bool collapse_blank_lines = true;
	size_t max_chars = 30000;   // hard truncation bound
};

// full pipeline; `command` is the shell command the output came from (used to
// pick the git/build compaction strategies) and `language` localizes the few
// prose lines this pipeline inserts into the tool output.
std::string compact_output(const std::string& raw, const std::string& command,
	const std::string& language, const Options& options = {});

// individual stages (exposed for tests)
std::string strip_ansi(const std::string& text);
std::string compact_git(const std::string& output, const std::string& command,
	const std::string& language);
std::string filter_build_output(const std::string& output, const std::string& command,
	const std::string& language);
std::string collapse_blank_lines(const std::string& text);
std::string hard_truncate(const std::string& text, size_t max_chars, const std::string& language);

}  // namespace rtk
}  // namespace phi
