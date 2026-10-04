// Background subagents. The `agent` tool hands a task to a SubagentPool, which
// runs a child Agent on its own thread and returns immediately, so the main
// model keeps streaming while subagents work in parallel.
//
// The child's log is its transcript rendered in the session's own message format
// (session_store.cpp message_to_json — the exact serialization the session uses
// for both the UI protocol and the JSONL store), and its final assistant text is
// the report the parent delivers back to the main model.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "agent/agent.hpp"

namespace phi {

// Everything needed to run one child agent. The parent Agent fills in the whole
// config (model, tools, system prompt, language); the agent tool only names the
// task.
struct SubagentSpec {
	std::string id;
	std::string description;
	std::string task;
	Agent::Config config;
	ProviderStreamFn stream_fn;
};

struct SubagentOutcome {
	std::string id;
	std::string description;
	std::string output;      // the child's last assistant text (最后一段输出)
	std::string transcript;  // the child's log, in the session's message format
	bool is_error = false;
};

// Render messages exactly like a session stores them: one JSON object per line,
// each produced by message_to_json (the session's canonical message format).
std::string render_session_transcript(const std::vector<AgentMessage>& messages);

class SubagentPool {
public:
	using UpdateFn = std::function<void(const std::string& id, const std::string& transcript)>;
	using DoneFn = std::function<void(const SubagentOutcome& outcome)>;

	SubagentPool() = default;
	~SubagentPool();
	SubagentPool(const SubagentPool&) = delete;
	SubagentPool& operator=(const SubagentPool&) = delete;

	// Runs `spec` on a background thread and returns immediately.
	void start(SubagentSpec spec, UpdateFn on_update, DoneFn on_done);
	// Ask every running subagent to stop (they poll this between steps and
	// between provider reads).
	void cancel_all();
	size_t running() const;

private:
	struct Task;
	std::vector<std::shared_ptr<Task>> tasks_;
	mutable std::mutex mutex_;
};

}  // namespace phi
