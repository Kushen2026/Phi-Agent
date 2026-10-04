// Subagent tool: delegate a task to a background child agent that runs its own
// agent loop (think → batch tools → … → summary) with a filtered tool set and a
// separate context window, using the same system prompt as the main model.
// The child's log is rendered in the session's message format; its final output
// is delivered back to the main model when it finishes.
#pragma once

#include <vector>

#include "agent/agent.hpp"
#include "util/json.hpp"

namespace phi {

ToolDef build_agent_tool();

}  // namespace phi
