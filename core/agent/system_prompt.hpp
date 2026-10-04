// System prompt construction (zh/en), ported from system-prompt.ts.
#pragma once

#include <string>
#include <vector>

#include "store/resources.hpp"

namespace phi {

struct SystemPromptOptions {
	// tool names / one-line snippets: no longer rendered into the prompt text
	// (the 可用工具 / Available tools list was removed); kept for callers
	std::vector<std::string> selected_tools;
	std::vector<std::pair<std::string, std::string>> tool_snippets;  // name -> one-line
	std::string append_system_prompt;
	std::string language;  // "zh" | "en"
	std::string cwd;
	std::vector<std::string> cwd_files;  // files in current working directory
	std::string docs_readme_path;
	std::string docs_dir;
	std::string docs_examples_dir;
};

std::string build_system_prompt(const SystemPromptOptions& options);

}  // namespace phi
