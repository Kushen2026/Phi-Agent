// Agent core contract: content parts, messages, stream events, tools, loop.
#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "util/base.hpp"
#include "util/json.hpp"
#include "llm/models.hpp"

namespace phi {

// ── usage ──────────────────────────────────────────────────────────────────

struct Usage {
	int64_t input = 0;
	int64_t output = 0;
	int64_t cache_read = 0;
	int64_t cache_write = 0;
	int64_t total_tokens = 0;
	double cost = 0;  // usd

	static Usage compute(const Usage& raw, const ModelDef& model) {
		Usage u = raw;
		u.total_tokens = raw.input + raw.output + raw.cache_read + raw.cache_write;
		u.cost = (raw.input * model.cost.input + raw.output * model.cost.output +
			raw.cache_read * model.cost.cache_read + raw.cache_write * model.cost.cache_write) / 1e6;
		return u;
	}
};

// ── content parts ──────────────────────────────────────────────────────────

struct ContentPart {
	enum class Type { Text, Thinking, ToolCall, Image };
	Type type = Type::Text;
	std::string text;              // Text / Thinking
	bool redacted = false;         // Thinking
	std::string tool_call_id;      // ToolCall
	std::string tool_name;
	JsonValue arguments;           // ToolCall
	std::string arguments_text;    // ToolCall: raw json accumulation during streaming
	std::string image_data;        // Image (base64)
	std::string image_mime;

	static ContentPart text_part(std::string t) {
		ContentPart p;
		p.type = Type::Text;
		p.text = std::move(t);
		return p;
	}
	static ContentPart thinking(std::string t, bool redacted = false) {
		ContentPart p;
		p.type = Type::Thinking;
		p.text = std::move(t);
		p.redacted = redacted;
		return p;
	}
	static ContentPart tool_call(std::string id, std::string name, JsonValue args) {
		ContentPart p;
		p.type = Type::ToolCall;
		p.tool_call_id = std::move(id);
		p.tool_name = std::move(name);
		p.arguments = std::move(args);
		return p;
	}
	static ContentPart image(std::string data, std::string mime) {
		ContentPart p;
		p.type = Type::Image;
		p.image_data = std::move(data);
		p.image_mime = std::move(mime);
		return p;
	}
};

// ── messages ───────────────────────────────────────────────────────────────

struct AgentMessage {
	std::string role;  // "user" | "assistant" | "toolResult" | "compactionSummary"
	std::vector<ContentPart> content;
	std::string model_provider;
	std::string model_id;
	std::string model_name;
	int64_t timestamp = 0;
	std::string stop_reason;
	std::string error_message;
	Usage usage;
	bool has_usage = false;
	// toolResult
	std::string tool_call_id;
	std::string tool_name;
	bool is_error = false;
	// compactionSummary
	std::string summary;
	int64_t tokens_before = 0;
	int64_t tokens_after = 0;
	// subagentSummary: the short description of the child that produced it
	std::string subagent_description;

	static AgentMessage user_text(const std::string& text) {
		AgentMessage m;
		m.role = "user";
		m.timestamp = now_ms();
		if (!text.empty()) m.content.push_back(ContentPart::text_part(text));
		return m;
	}
	static AgentMessage tool_result(const std::string& call_id, const std::string& name,
		std::vector<ContentPart> parts, bool is_error) {
		AgentMessage m;
		m.role = "toolResult";
		m.tool_call_id = std::move(call_id);
		m.tool_name = std::move(name);
		m.content = std::move(parts);
		m.is_error = is_error;
		m.timestamp = now_ms();
		return m;
	}
	// a background subagent's report: rendered as its own box in the UI and
	// sent to the main model as context (never as a tool result)
	static AgentMessage subagent_summary(std::string description, std::string output, bool is_error) {
		AgentMessage m;
		m.role = "subagentSummary";
		m.subagent_description = std::move(description);
		if (!output.empty()) m.content.push_back(ContentPart::text_part(std::move(output)));
		m.is_error = is_error;
		m.timestamp = now_ms();
		return m;
	}
};

// ── provider stream events ─────────────────────────────────────────────────

struct StreamEvent {
	enum class Type { Start, TextDelta, ThinkingDelta, ToolCallStart, ToolCallArgsDelta, Done, Error };
	Type type = Type::Start;
	std::string text;          // TextDelta / ThinkingDelta / Error
	int tool_index = 0;        // ToolCallStart / ToolCallArgsDelta
	std::string tool_id;       // ToolCallStart
	std::string tool_name;     // ToolCallStart
	std::string args_delta;    // ToolCallArgsDelta
	std::string stop_reason;   // Done: "endTurn" | "toolUse" | "aborted" | "error" | "maxTokens" | ...
	std::string error_message; // Done (when stop_reason == "error")
	Usage usage;               // Done
};

struct StreamOptions {
	std::string system_prompt;
	std::string thinking_level;  // canonical level, provider maps it
	std::string session_id;
	// tool schemas as a JSON array string: [{"name","description","input_schema"}]
	std::string tools_json;
	// resolved api key for the model's provider
	std::string api_key;
	// language: "zh" or "en", used to select tool descriptions
	std::string language;
	// polled by the transport between reads; return true to stop the stream
	// (abort that works even when the provider is silent for a while)
	std::function<bool()> cancelled;
};

// callback returns false to request abort
using StreamCallback = std::function<bool(const StreamEvent&)>;
using ProviderStreamFn =
	std::function<void(const ModelDef& model, const std::vector<AgentMessage>& context,
		const StreamOptions& options, const StreamCallback& on_event)>;

// implemented in providers.cpp
void stream_llm(const ModelDef& model, const std::vector<AgentMessage>& context,
	const StreamOptions& options, const StreamCallback& on_event);

// ── tools ──────────────────────────────────────────────────────────────────

struct ToolResult {
	std::vector<ContentPart> content;
	bool is_error = false;
};

// tool-settings snapshot handed to the agent at session start (from settings.json)
struct ToolPolicies {
	std::map<std::string, size_t> output_limits;
	std::map<std::string, std::string> permissions;
	std::set<std::string> subagent_tools;
};

struct ToolDef;  // fwd: ToolContext carries a snapshot of the tool set
class SubagentPool;
class Agent;

// A background subagent request. The agent tool only names the task; the parent
// Agent owns the child's model/tools/prompt and runs it off-thread.
struct SubagentRequest {
	std::string description;
	std::string task;
};

struct ToolContext {
	std::string cwd;
	std::string agent_dir;   // data dir
	std::string bin_dir;     // vendored tool binaries (rg.exe, fd.exe)
	std::string session_id;
	std::string system_prompt; // same as parent session
	std::function<void(const std::string& partial_text)> on_update;
	// name of the tool currently executing (set by Agent::execute_tool) —
	// keys into output_limits / permissions below
	std::string current_tool;
	// tool name -> max output chars (missing / 0 = tool default)
	std::map<std::string, size_t> output_limits;
	// tool name -> "everywhere" | "in-cwd" (missing = everywhere)
	std::map<std::string, std::string> permissions;
	// subagent tool gating: empty = all parent tools, else the allowed subset
	std::set<std::string> subagent_tools;
	// thinking level for spawned subagents: always the parent session's
	// current level (set by Agent::execute_tool; the old settings override was
	// removed — subagents follow the main agent)
	std::string subagent_thinking;
	// session language: "zh" or "en" — every built-in prompt string follows it
	std::string language = "zh";
	// background subagent launch (agent tool): starts a child agent off-thread
	// and returns immediately with its id; throws std::string when it cannot start
	std::function<std::string(const SubagentRequest&)> launch_subagent;
	// polled by long-running tools (bash, subagents); true = user requested abort
	std::function<bool()> cancelled;
	// subagent plumbing (used by the agent tool)
	std::optional<ModelDef> model;   // model the parent session is running on
	ProviderStreamFn stream_fn;      // provider seam (test overrides flow through)
	std::function<std::vector<ToolDef>()> get_tools;  // parent tool set snapshot
	std::function<std::optional<ModelDef>(const std::string& provider, const std::string& id)> resolve_model;
	std::function<std::string(const ModelDef&)> resolve_api_key;
};

// Every built-in prompt string has two forms — Chinese and English — picked by
// the session language (settings.json → systemPromptLanguage, surfaced as
// Config::language; see i18n.hpp).
struct ToolDef {
	std::string name;
	std::string description;   // full description for the model (Chinese)
	std::string description_en; // English description
	std::string prompt_snippet;  // one-line for the system prompt tools list (Chinese)
	std::string prompt_snippet_en;  // English one-line snippet
	// input-schema builder: parameter descriptions follow the session language
	// too, so the schema is materialized on demand (Agent::build_tools_json)
	std::function<JsonValue(const std::string& language)> parameters_fn;
	std::function<ToolResult(const JsonValue& args, ToolContext& ctx)> run;
};

// ── agent events (consumed by AgentSession) ────────────────────────────────

struct AgentEvent {
	enum class Type {
		AgentStart,
		TurnStart,
		MessageStart,
		MessageUpdate,
		MessageEnd,
		ToolExecutionStart,
		ToolExecutionUpdate,
		ToolExecutionEnd,
		TurnEnd,
		AgentEnd,
		AutoRetryStart,
		AutoRetryEnd,
		CompactionStart,
		CompactionEnd,
		Error,
	};
	Type type;
	std::string tool_call_id;
	std::string tool_name;
	JsonValue args;
	std::string partial_output;
	std::optional<ToolResult> tool_result;
	std::string message;  // Error text
};

using AgentEventCallback = std::function<void(const AgentEvent&)>;

// ── tool result helpers ────────────────────────────────────────────────────
inline ToolResult text_result(std::string text, bool is_error = false) {
	ToolResult r;
	r.content.push_back(ContentPart::text_part(std::move(text)));
	r.is_error = is_error;
	return r;
}
inline ToolResult error_result(std::string text) {
	return text_result(std::move(text), true);
}

// ── the agent loop ─────────────────────────────────────────────────────────

class Agent {
public:
	// atomic view of the message list + streaming message (both under one lock;
	// separate messages()/streaming_message() calls can straddle a finalize)
	struct Projection {
		std::vector<AgentMessage> messages;
		std::optional<AgentMessage> streaming;
		bool streaming_active = false;
	};
public:
	struct Config {
		std::string system_prompt;
		std::optional<ModelDef> model;
		std::string thinking_level = "medium";
		// tool language: "zh" or "en"
		std::string language = "zh";
		std::vector<ToolDef> tools;   // enabled tools
		std::string session_id;
		std::string session_cwd;
		std::string agent_dir;        // data dir (passed to tools)
		std::string bin_dir;          // vendored tool binaries
		// context filtering (compaction): applied to the message history before each call
		std::function<std::vector<AgentMessage>(const std::vector<AgentMessage>&)> transform_context;
		// api key resolution for the current model
		std::function<std::string(const ModelDef&)> resolve_api_key;
		// model catalog lookup for subagents ("provider/id" -> ModelDef)
		std::function<std::optional<ModelDef>(const std::string& provider, const std::string& id)> resolve_model;
		// external cancel (e.g. parent agent aborting a subagent); polled before
		// and during each stream turn
		std::function<bool()> cancelled;
		// tool settings snapshot (truncation limits, permissions, subagent gating)
		ToolPolicies tool_policies;
		// results handed back by background subagents while this agent is idle are
		// re-entered here so the host can run them on its own worker/queue
		// (empty = run inline, e.g. in tests)
		std::function<void(std::string)> background_prompt;
	};

	Agent(Config config, ProviderStreamFn stream_fn);
	~Agent();

	// queues and drives the loop on a worker thread; safe while streaming
	// (queues as follow-up)
	void prompt(std::string text, std::vector<ContentPart> images = {});
	// inject into the current turn
	void steer(std::string text, std::vector<ContentPart> images = {});
	void abort();
	bool is_streaming() const { return streaming_; }
	size_t steer_queue_size() const;
	size_t follow_up_queue_size() const;

	void set_event_callback(AgentEventCallback cb) { on_event_ = std::move(cb); }

	// atomic snapshot of messages + streaming message (single state_mutex_ hold)
	Projection projection() const;

	// how many background subagents this agent started that are still running
	size_t subagents_running() const;

	// a finished background subagent's report, waiting to be shown to the model
	struct SubagentReport {
		std::string description;
		std::string output;
		bool is_error = false;
	};

	// state access (caller holds state_mutex when reading while streaming)
	std::vector<AgentMessage> messages() const;
	std::optional<AgentMessage> streaming_message() const;
	void set_messages(std::vector<AgentMessage> msgs);  // session restore

	void set_model(std::optional<ModelDef> model);
	void set_thinking_level(const std::string& level);
	std::optional<ModelDef> model() const;
	std::string thinking_level() const;

private:
	// one turn = one user input driven to completion by repeated steps;
	// one step = one THINK (streamed model call) + one batch EXECUTE (all tool
	// calls from that response). The loop alternates steps until a step makes
	// no tool calls — that final step's text is the closing summary.
	enum class StepOutcome { Continue, Done, Aborted, Error };

	void run_chain(std::string first_text, std::vector<ContentPart> first_images);
	void run_turn(const std::string& text, const std::vector<ContentPart>& images);
	StepOutcome run_step();
	void emit(AgentEvent::Type type);
	void emit_tool(AgentEvent::Type type, const std::string& call_id, const std::string& name,
		const JsonValue& args, const std::string& partial, const std::optional<ToolResult>& result);
	bool drain_steering(std::vector<AgentMessage>& out);
	bool drain_follow_up(std::vector<AgentMessage>& out);
	std::string build_tools_json() const;
	ToolResult execute_tool(const ContentPart& call, bool emit_updates);
	bool cancelled() const;

	// ── background subagents ───────────────────────────────────────
	// The agent tool delegates here: validate the child config, register the
	// subagent and run it off-thread. Throws std::string when it cannot start.
	std::string launch_subagent(const SubagentRequest& request, const std::string& tool_call_id);
	// live log: the child transcript rides the tool run that launched it
	void on_subagent_update(const std::string& id, const std::string& transcript);
	void on_subagent_done(const std::string& id, const std::string& description,
		const std::string& output, bool is_error);
	// append pending finished-subagent reports to the history as
	// subagentSummary messages. require_all: only deliver once EVERY
	// background subagent has finished (used when the main model is paused).
	// returns true when at least one message was appended.
	bool deliver_subagent_summaries(bool require_all);

	Config config_;
	ProviderStreamFn stream_fn_;
	AgentEventCallback on_event_;

	mutable std::mutex state_mutex_;
	mutable std::mutex queue_mutex_;
	std::vector<AgentMessage> messages_;
	std::optional<AgentMessage> streaming_message_;
	std::atomic<bool> streaming_{false};
	std::atomic<bool> abort_requested_{false};
	std::thread worker_;
	std::vector<AgentMessage> steering_;
	std::vector<AgentMessage> follow_up_;

	// ── background subagent state ──────────────────────────────────────
	struct SubagentState {
		std::string description;
		std::string tool_call_id;
		bool finished = false;
		bool is_error = false;
	};
	mutable std::mutex subagent_mutex_;
	std::map<std::string, SubagentState> subagents_;
	std::vector<SubagentReport> subagent_reports_;
	// declared last: its destructor cancels and joins the background subagents
	// while every other member is still alive
	std::unique_ptr<SubagentPool> subagent_pool_;
};

}  // namespace phi
