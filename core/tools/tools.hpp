// Built-in file/command tools: bash, read, write, edit, ls, grep.
// bash has the rtk output compactor built in (core/util/rtk.hpp) — no external
// rtk binary or separate rtk tool.
#pragma once

#include <vector>

#include "agent/agent.hpp"

namespace phi {

// all built-in tools in registration order; `agent` (subagent) is appended by
// tools_subagent.cpp via build_agent_tool() when the caller assembles the set,
// and image_generate/video_generate come from tools_media.cpp.
std::vector<ToolDef> build_file_tools();
std::vector<ToolDef> build_media_tools();

}  // namespace phi
