// phi agent loop — think → batch tools → think → batch tools → … → summary.
//
// The loop is built around a strict two-phase step:
//   THINK    stream one model response (reasoning + tool calls + optional text)
//   EXECUTE  run every tool call from that response as a batch, append results
#include "agent/agent.hpp"
#include "util/trace.hpp"

#include <algorithm>
#include <unordered_set>

#include "util/i18n.hpp"
#include "agent/subagent.hpp"

namespace phi {

namespace {

std::string extract_text(const std::vector<ContentPart>& parts) {
	std::string out;
	for (const auto& p : parts) {
		if (p.type == ContentPart::Type::Text) out += p.text;
	}
	return out;
}
std::vector<ContentPart> extract_images(const std::vector<ContentPart>& parts) {
	std::vector<ContentPart> out;
	for (const auto& p : parts) {
		if (p.type == ContentPart::Type::Image) out.push_back(p);
	}
	return out;
}

// providers split usage across events (anthropic: input in message_start,
// output in message_delta; openai: usage chunks + final chunk). Each field
// takes the max seen so partial reports accumulate instead of clobbering.
void merge_usage(Usage& acc, const Usage& u) {
	if (u.input > 0)       acc.input       = std::max(acc.input,       u.input);
	if (u.output > 0)      acc.output      = std::max(acc.output,      u.output);
	if (u.cache_read > 0)  acc.cache_read  = std::max(acc.cache_read,  u.cache_read);
	if (u.cache_write > 0) acc.cache_write = std::max(acc.cache_write, u.cache_write);
}

void append_text_part(std::vector<ContentPart>& content, const std::string& text) {
	if (!content.empty() && content.back().type == ContentPart::Type::Text) {
		content.back().text += text;
	} else {
		content.push_back(ContentPart::text_part(text));
	}
}

void append_thinking_part(std::vector<ContentPart>& content, const std::string& text) {
	if (!content.empty() && content.back().type == ContentPart::Type::Thinking) {
		content.back().text += text;
	} else {
		content.push_back(ContentPart::thinking(text));
	}
}

// publish a live snapshot of the in-flight assistant message; tool call
// arguments are re-parsed from the raw accumulation so the UI can render
// calls as they form
AgentMessage make_streaming_view(const AgentMessage& assistant,
	const std::vector<ContentPart>& calls) {
	AgentMessage view = assistant;
	size_t call_idx = 0;
	for (auto& part : view.content) {
		if (part.type != ContentPart::Type::ToolCall) continue;
		if (call_idx >= calls.size()) break;
		const ContentPart& live = calls[call_idx++];
		auto parsed = json_parse(live.arguments_text.empty() ? "{}" : live.arguments_text);
		part.arguments = parsed ? *parsed : JsonValue::object();
		part.arguments_text = live.arguments_text;
	}
	return view;
}

}  // namespace

// ── lifecycle ──────────────────────────────────────────────────────────────

Agent::Agent(Config config, ProviderStreamFn stream_fn)
	: config_(std::move(config)), stream_fn_(std::move(stream_fn)),
	  subagent_pool_(std::make_unique<SubagentPool>()) {}

Agent::~Agent() = default;

// ── background subagents ───────────────────────────────────────────────────

size_t Agent::subagents_running() const {
	return subagent_pool_ ? subagent_pool_->running() : 0;
}

std::string Agent::launch_subagent(const SubagentRequest& request, const std::string& tool_call_id) {
	const bool zh = lang_is_zh(config_.language);
	std::string task = trim(request.task);
	if (task.empty()) {
		throw std::string(zh ? "prompt 不能为空" : "prompt must not be empty");
	}

	// child tool set: the parent's tools minus the agent tool (subagents never
	// spawn subagents), still gated by 设置 → 工具设置 → 子代理
	std::vector<ToolDef> tools;
	{
		std::lock_guard<std::mutex> lock(state_mutex_);
		for (const auto& tool : config_.tools) {
			if (tool.name == "agent") continue;
			if (!config_.tool_policies.subagent_tools.empty() &&
				!config_.tool_policies.subagent_tools.count(tool.name)) {
				continue;
			}
			tools.push_back(tool);
		}
	}
	if (tools.empty()) {
		throw std::string(zh
			? "子代理没有可用工具（检查 设置 → 工具设置 → 子代理 的工具开关）"
			: "no tools available for the subagent (check Settings → Tool settings → Subagent)");
	}

	// subagent model: the gw-subagent gateway when configured, else the parent's
	std::optional<ModelDef> model;
	if (config_.resolve_model) model = config_.resolve_model("gw-subagent", "");
	if (!model) model = config_.model;
	if (!model) {
		throw std::string(zh ? "没有可用模型（请在 API 配置中设置模型）"
		                     : "no model available (configure a model in API settings)");
	}

	std::string id = "sub-" + uuid4().substr(0, 8);
	std::string description = trim(request.description);
	if (description.empty()) description = zh ? "未命名子代理" : "unnamed subagent";

	SubagentSpec spec;
	spec.id = id;
	spec.description = description;
	spec.task = task;
	spec.stream_fn = stream_fn_;

	Agent::Config child;
	// the child runs on the main model's system prompt, verbatim
	child.system_prompt = config_.system_prompt;
	child.model = *model;
	child.thinking_level = config_.thinking_level;
	// built-in prompts inside the child follow the session language too
	child.language = config_.language;
	child.tools = std::move(tools);
	child.session_id = config_.session_id + "-" + id;
	child.session_cwd = config_.session_cwd;
	child.agent_dir = config_.agent_dir;
	child.bin_dir = config_.bin_dir;
	child.resolve_api_key = config_.resolve_api_key;
	child.resolve_model = config_.resolve_model;
	child.cancelled = [this] { return cancelled(); };
	child.tool_policies = config_.tool_policies;
	// note: background_prompt is deliberately not inherited — a child can never
	// spawn subagents, so it has nothing to deliver
	spec.config = std::move(child);

	{
		std::lock_guard<std::mutex> lock(subagent_mutex_);
		SubagentState state;
		state.description = description;
		state.tool_call_id = tool_call_id;
		subagents_[id] = std::move(state);
	}

	subagent_pool_->start(std::move(spec),
		[this](const std::string& sid, const std::string& transcript) {
			on_subagent_update(sid, transcript);
		},
		[this](const SubagentOutcome& outcome) {
			on_subagent_done(outcome.id, outcome.description, outcome.output, outcome.is_error);
		});
	return id;
}

void Agent::on_subagent_update(const std::string& id, const std::string& transcript) {
	std::string call_id;
	{
		std::lock_guard<std::mutex> lock(subagent_mutex_);
		auto it = subagents_.find(id);
		if (it == subagents_.end()) return;
		call_id = it->second.tool_call_id;
	}
	if (call_id.empty()) return;
	// the subagent's log rides the tool run of the `agent` call that started it,
	// in the session's own message format
	AgentEvent ev;
	ev.type = AgentEvent::Type::ToolExecutionUpdate;
	ev.tool_call_id = call_id;
	ev.tool_name = "agent";
	ev.partial_output = transcript;
	if (on_event_) on_event_(ev);
}

void Agent::on_subagent_done(const std::string& id, const std::string& description,
	const std::string& output, bool is_error) {
	{
		std::lock_guard<std::mutex> lock(subagent_mutex_);
		auto it = subagents_.find(id);
		if (it != subagents_.end()) {
			it->second.finished = true;
			it->second.is_error = is_error;
		}
		SubagentReport report;
		report.description = description;
		report.output = output;
		report.is_error = is_error;
		subagent_reports_.push_back(std::move(report));
	}
	// main model still running: the report is held and delivered at the tail of a
	// round (run_step / run_chain) instead of interrupting the stream
	if (is_streaming()) return;
	// main model paused: append the summary after the last output and start a
	// fresh turn so the model actually receives it as context. If siblings are
	// still running, wait — the last one to finish drives the turn.
	if (!deliver_subagent_summaries(true)) return;
	if (config_.background_prompt) config_.background_prompt("");
	else prompt("");
}

bool Agent::deliver_subagent_summaries(bool require_all) {
	std::vector<SubagentReport> reports;
	{
		std::lock_guard<std::mutex> lock(subagent_mutex_);
		if (subagent_reports_.empty()) return false;
		bool any_running = false;
		for (const auto& [id, state] : subagents_) {
			if (!state.finished) any_running = true;
		}
		if (require_all && any_running) return false;
		reports.swap(subagent_reports_);
		// housekeeping: once nothing is still running the batch is over, so the
		// finished entries can go (otherwise a long session accumulates them)
		if (!any_running) subagents_.clear();
	}
	std::lock_guard<std::mutex> lock(state_mutex_);
	for (const auto& report : reports) {
		messages_.push_back(AgentMessage::subagent_summary(report.description, report.output, report.is_error));
	}
	return true;
}

bool Agent::cancelled() const {
	if (abort_requested_.load()) return true;
	return config_.cancelled && config_.cancelled();
}

void Agent::emit(AgentEvent::Type type) {
	AgentEvent ev;
	ev.type = type;
	if (on_event_) on_event_(ev);
}

void Agent::emit_tool(AgentEvent::Type type, const std::string& call_id, const std::string& name,
	const JsonValue& args, const std::string& partial, const std::optional<ToolResult>& result) {
	AgentEvent ev;
	ev.type = type;
	ev.tool_call_id = call_id;
	ev.tool_name = name;
	ev.args = args;
	ev.partial_output = partial;
	ev.tool_result = result;
	if (on_event_) on_event_(ev);
}

// ── state accessors (unchanged public contract) ────────────────────────────

std::vector<AgentMessage> Agent::messages() const {
	std::lock_guard<std::mutex> lock(state_mutex_);
	return messages_;
}

std::optional<AgentMessage> Agent::streaming_message() const {
	std::lock_guard<std::mutex> lock(state_mutex_);
	return streaming_message_;
}

Agent::Projection Agent::projection() const {
	std::lock_guard<std::mutex> lock(state_mutex_);
	Projection p;
	p.messages = messages_;
	p.streaming = streaming_message_;
	p.streaming_active = streaming_.load();
	return p;
}

void Agent::set_messages(std::vector<AgentMessage> msgs) {
	std::lock_guard<std::mutex> lock(state_mutex_);
	messages_ = std::move(msgs);
}

void Agent::set_model(std::optional<ModelDef> model) {
	std::lock_guard<std::mutex> lock(state_mutex_);
	config_.model = std::move(model);
}

void Agent::set_thinking_level(const std::string& level) {
	std::lock_guard<std::mutex> lock(state_mutex_);
	config_.thinking_level = level;
}

std::optional<ModelDef> Agent::model() const {
	std::lock_guard<std::mutex> lock(state_mutex_);
	return config_.model;
}

std::string Agent::thinking_level() const {
	std::lock_guard<std::mutex> lock(state_mutex_);
	return config_.thinking_level;
}

// ── entry: own the streaming slot and drive the turn chain ─────────────────

void Agent::prompt(std::string text, std::vector<ContentPart> images) {
	bool expected = false;
	if (!streaming_.compare_exchange_strong(expected, true)) {
		// already running: queue as a follow-up turn
		std::lock_guard<std::mutex> lock(queue_mutex_);
		AgentMessage m = AgentMessage::user_text(text);
		for (auto& img : images) m.content.push_back(std::move(img));
		follow_up_.push_back(std::move(m));
		return;
	}
	abort_requested_ = false;
	emit(AgentEvent::Type::AgentStart);
	run_chain(std::move(text), std::move(images));
	streaming_ = false;
	// a subagent can finish in the gap between the chain's final delivery check
	// and this flag clearing: drain that last report here and drive one more turn
	// so it is never stranded (the summary would otherwise wait for a user input).
	// A loop, not a recursive prompt(): a recursive call emitted a second
	// AgentStart *before* this frame's AgentEnd, so the pair the UI keys its
	// state on came out as Start,Start,End,End.
	while (!abort_requested_.load() && !cancelled() && deliver_subagent_summaries(true)) {
		if (config_.background_prompt) {
			config_.background_prompt("");
			break;  // the host owns the next turn; it will re-enter prompt() itself
		}
		run_chain("", {});
	}
	emit(AgentEvent::Type::AgentEnd);
}

void Agent::run_chain(std::string first_text, std::vector<ContentPart> first_images) {
	std::string text = std::move(first_text);
	std::vector<ContentPart> images = std::move(first_images);

	while (true) {
		bool failed = false;
		try {
			run_turn(text, images);
		} catch (const std::string& e) {
			failed = true;
			AgentEvent ev;
			ev.type = AgentEvent::Type::Error;
			ev.message = e;
			if (on_event_) on_event_(ev);
		} catch (const std::exception& e) {
			failed = true;
			AgentEvent ev;
			ev.type = AgentEvent::Type::Error;
			ev.message = e.what();
			if (on_event_) on_event_(ev);
		} catch (...) {
			failed = true;
			AgentEvent ev;
			ev.type = AgentEvent::Type::Error;
			ev.message = "unknown error";
			if (on_event_) on_event_(ev);
		}
		text.clear();
		images.clear();
		// external cancel (config_.cancelled, e.g. the parent aborting a subagent)
		// must end the chain just like abort(): polling only inside run_turn let
		// the chain resume from the follow-up queue after the parent gave up
		if (failed || abort_requested_.load() || cancelled()) break;

		// The model has paused. Append any subagent reports that landed during the
		// final step after the last output, then drive one more turn (with no new
		// user text) so they are actually sent to the model as context.
		if (deliver_subagent_summaries(true)) {
			std::lock_guard<std::mutex> lock(queue_mutex_);
			follow_up_.push_back(AgentMessage::user_text(""));
		}

		// next queued input starts its own turn
		std::vector<AgentMessage> follow;
		if (!drain_follow_up(follow) || follow.empty()) break;
		// NOTE: abort_requested_ is checked again AFTER acquiring queue_mutex_
		// to avoid a TOCTOU race where abort fires between the drain and the lock.
		// In the worst case the follow-up messages get re-enqueued on the queue
		// (drained again on the next loop iteration) instead of being silently lost.
		// should_requeue also covers external cancel via cancelled(): a parent
		// aborting a subagent must stop the queue from being consumed.
		bool should_requeue = abort_requested_.load() || cancelled();
		{
			std::lock_guard<std::mutex> lock(queue_mutex_);
			if (should_requeue) {
				for (auto& m : follow) follow_up_.push_back(std::move(m));
			}
		}
		if (should_requeue) break;
		bool requeued = false;
		for (size_t i = 0; i < follow.size(); i++) {
			if (abort_requested_.load() || cancelled()) {
				std::lock_guard<std::mutex> lock(queue_mutex_);
				for (size_t j = i; j < follow.size(); j++) follow_up_.push_back(std::move(follow[j]));
				requeued = true;
				break;
			}
			try {
				run_turn(extract_text(follow[i].content), extract_images(follow[i].content));
			} catch (const std::string& e) {
				AgentEvent ev;
				ev.type = AgentEvent::Type::Error;
				ev.message = e;
				if (on_event_) on_event_(ev);
				break;
			} catch (const std::exception& e) {
				AgentEvent ev;
				ev.type = AgentEvent::Type::Error;
				ev.message = e.what();
				if (on_event_) on_event_(ev);
				break;
			} catch (...) {
				AgentEvent ev;
				ev.type = AgentEvent::Type::Error;
				ev.message = "unknown error";
				if (on_event_) on_event_(ev);
				break;
			}
		}
		if (requeued || abort_requested_.load() || cancelled()) break;
	}
}

// ── steering / queues ──────────────────────────────────────────────────────

void Agent::steer(std::string text, std::vector<ContentPart> images) {
	std::lock_guard<std::mutex> lock(queue_mutex_);
	AgentMessage m = AgentMessage::user_text(text);
	for (auto& img : images) m.content.push_back(std::move(img));
	steering_.push_back(std::move(m));
}

void Agent::abort() {
	abort_requested_ = true;
}

size_t Agent::steer_queue_size() const {
	std::lock_guard<std::mutex> lock(queue_mutex_);
	return steering_.size();
}

size_t Agent::follow_up_queue_size() const {
	std::lock_guard<std::mutex> lock(queue_mutex_);
	return follow_up_.size();
}

bool Agent::drain_steering(std::vector<AgentMessage>& out) {
	std::lock_guard<std::mutex> lock(queue_mutex_);
	out = std::move(steering_);
	steering_.clear();
	return !out.empty();
}

bool Agent::drain_follow_up(std::vector<AgentMessage>& out) {
	std::lock_guard<std::mutex> lock(queue_mutex_);
	out = std::move(follow_up_);
	follow_up_.clear();
	return !out.empty();
}

// ── turn: repeat steps until the model stops calling tools ─────────────────

void Agent::run_turn(const std::string& text, const std::vector<ContentPart>& images) {
	// seed the history with the user input driving this turn
	if (!text.empty() || !images.empty()) {
		std::lock_guard<std::mutex> lock(state_mutex_);
		AgentMessage m = AgentMessage::user_text(text);
		for (auto& img : images) m.content.push_back(img);
		messages_.push_back(std::move(m));
	}

	emit(AgentEvent::Type::TurnStart);
	try {
		while (true) {
			if (cancelled()) break;  // also covers external cancel (parent aborting a subagent)
			StepOutcome outcome = run_step();
			if (outcome != StepOutcome::Continue) break;
			// Continue: the step issued tool calls, results are in history —
			// loop straight into the next THINK.
		}
	} catch (...) {
		// a throw out of run_step must not leave TurnStart without its
		// TurnEnd: the UI keys its "generating" state on the pair
		emit(AgentEvent::Type::TurnEnd);
		throw;
	}
	emit(AgentEvent::Type::TurnEnd);
}

// ── one step: THINK (stream) + EXECUTE (batch tools) ───────────────────────

Agent::StepOutcome Agent::run_step() {
	// steering arrives between steps: inject before the model call so the
	// next THINK sees it
	{
		std::vector<AgentMessage> steering;
		if (drain_steering(steering)) {
			std::lock_guard<std::mutex> lock(state_mutex_);
			for (auto& m : steering) messages_.push_back(std::move(m));
		}
	}

	// snapshot everything the stream call needs
	std::vector<AgentMessage> context;
	std::string system_prompt;
	std::string thinking_level;
	std::optional<ModelDef> model_now;
	std::string tools_json;
	{
		std::lock_guard<std::mutex> lock(state_mutex_);
		context = messages_;
		system_prompt = config_.system_prompt;
		thinking_level = config_.thinking_level;
		model_now = config_.model;
		tools_json = build_tools_json();
	}
	if (config_.transform_context) context = config_.transform_context(context);
	if (!model_now) {
		AgentEvent ev;
		ev.type = AgentEvent::Type::Error;
		ev.message = "no model configured";
		if (on_event_) on_event_(ev);
		return StepOutcome::Error;
	}

	// ═══ THINK ═══════════════════════════════════════════════════════════
	AgentMessage assistant;
	assistant.role = "assistant";
	assistant.timestamp = now_ms();
	assistant.model_provider = model_now->provider;
	assistant.model_id = model_now->id;
	assistant.model_name = model_now->name;

	emit(AgentEvent::Type::MessageStart);

	// tool calls accumulate out-of-band; slots map stream tool_index →
	// position in calls_, and the corresponding ContentPart lives in
	// assistant.content at call_slots_[i]
	std::vector<ContentPart> calls;
	std::map<int, size_t> index_to_call;
	std::string stop_reason;
	std::string error_message;
	bool stream_error = false;
	Usage raw_usage;

	StreamOptions opts;
	opts.system_prompt = system_prompt;
	opts.language = config_.language;
	opts.thinking_level = thinking_level;
	opts.session_id = config_.session_id;
	opts.tools_json = tools_json;
	if (config_.resolve_api_key) opts.api_key = config_.resolve_api_key(*model_now);
	opts.cancelled = [this] { return cancelled(); };

	auto publish_view = [&]() {
		std::lock_guard<std::mutex> lock(state_mutex_);
		streaming_message_ = make_streaming_view(assistant, calls);
	};

	if (stream_fn_) {
		StreamCallback on_event = [&](const StreamEvent& ev) -> bool {
			if (cancelled()) return false;
			switch (ev.type) {
				case StreamEvent::Type::Start:
					merge_usage(raw_usage, ev.usage);
					break;
				case StreamEvent::Type::TextDelta:
					// streaming prose is allowed here ONLY so the final step's
					// summary streams naturally; if tool calls show up this
					// text is dropped (below and at finalize)
					append_text_part(assistant.content, ev.text);
					break;
				case StreamEvent::Type::ThinkingDelta:
					append_thinking_part(assistant.content, ev.text);
					break;
				case StreamEvent::Type::ToolCallStart:
					index_to_call[ev.tool_index] = calls.size();
					calls.push_back(ContentPart::tool_call(ev.tool_id, ev.tool_name, JsonValue::object()));
					assistant.content.push_back(calls.back());
					break;
				case StreamEvent::Type::ToolCallArgsDelta: {
					auto it = index_to_call.find(ev.tool_index);
					if (it != index_to_call.end() && !ev.args_delta.empty()) {
						calls[it->second].arguments_text += ev.args_delta;
					}
					break;
				}
				case StreamEvent::Type::Done:
					stop_reason = ev.stop_reason;
					merge_usage(raw_usage, ev.usage);
					if (raw_usage.input > 0 || raw_usage.output > 0 || raw_usage.cache_read > 0 ||
						raw_usage.cache_write > 0) {
						assistant.has_usage = true;
						assistant.usage = Usage::compute(raw_usage, *model_now);
					}
					if (!ev.error_message.empty()) {
						error_message = ev.error_message;
						stream_error = true;
					}
					break;
				case StreamEvent::Type::Error:
					stream_error = true;
					error_message = ev.text;
					break;
			}
			publish_view();
			emit(AgentEvent::Type::MessageUpdate);
			return true;
		};
		stream_fn_(*model_now, context, opts, on_event);
		// usage may land after Done (openai-completions appends a terminal
		// usage-only chunk once include_usage is set) — recompute from the
		// merged totals so those tokens are not lost
		if (raw_usage.input > 0 || raw_usage.output > 0 || raw_usage.cache_read > 0 ||
			raw_usage.cache_write > 0) {
			assistant.has_usage = true;
			assistant.usage = Usage::compute(raw_usage, *model_now);
		}
	} else {
		stream_error = true;
		error_message = "no provider stream function configured";
	}

	// finalize tool call arguments from the raw JSON accumulations
	for (auto& call : calls) {
		if (call.arguments_text.empty()) continue;
		auto parsed = json_parse(call.arguments_text);
		if (parsed) call.arguments = *parsed;
	}
	// An id the provider never sent (a gateway that streams the id in a later
	// chunk than the name leaves the call nameless by the time it is recorded
	// here) has to be synthesized. Every wire format requires one, and the
	// history must carry the SAME id on the assistant's tool_calls and on the
	// tool message answering it: an empty id is dropped on the way out, and the
	// next request then dies with "an assistant message with 'tool_calls' must be
	// followed by tool messages responding to each 'tool_call_id'". Filling it in
	// once, here, keeps both sides — and every later re-send of this history —
	// consistent.
	//
	// The ids also have to be DISTINCT: both wire formats pair a tool message
	// with a tool call *by id*, so a provider that repeats one id across the
	// parallel calls of a turn (some gateways stream the same placeholder id for
	// every call) makes the batch look like it answers fewer calls than it made —
	// the same 400. Renaming the repeat here is what makes the batch answerable.
	{
		std::unordered_set<std::string> seen;
		seen.reserve(calls.size() * 2);
		for (auto& call : calls) {
			std::string& id = call.tool_call_id;
			if (id.empty() || seen.count(id)) {
				// Recheck after generation: uuid collision is rare but possible
				do { id = "call_" + uuid4(); } while (seen.count(id));
			}
			seen.insert(id);
		}
	}
	{
		size_t idx = 0;
		for (auto& part : assistant.content) {
			if (part.type == ContentPart::Type::ToolCall && idx < calls.size()) {
				part.arguments = calls[idx].arguments;
				part.tool_call_id = calls[idx].tool_call_id;
				idx++;
			}
		}
	}

	if (cancelled() && !stream_error) {
		stop_reason = "aborted";
	}
	if (stream_error) {
		stop_reason = "error";
		assistant.error_message = error_message;
	}
	// a step that produced tool calls is a toolUse step (regardless of what
	// the provider called the finish reason) — unless abort/error won first
	if (stop_reason.empty() && !calls.empty()) {
		stop_reason = "toolUse";
	} else if (stop_reason.empty() && !stream_error) {
		stop_reason = "endTurn";
	}
	assistant.stop_reason = stop_reason;

	{
		std::lock_guard<std::mutex> lock(state_mutex_);
		streaming_message_ = std::nullopt;
		messages_.push_back(assistant);
	}
	emit(AgentEvent::Type::MessageEnd);

	// A failed stream still has to *answer* every tool call it already recorded:
	// the assistant message (with them in it) is in the history and already on
	// disk, and a later request that carries `tool_calls` with fewer tool
	// messages is rejected by every provider — "an assistant message with
	// 'tool_calls' must be followed by tool messages responding to each
	// 'tool_call_id'" — for the rest of the session, because the dangling calls
	// are no longer at the tail where session replay repairs them. The tools are
	// deliberately NOT run here (the response that asked for them was truncated),
	// so this is the abort path's placeholder, per call.
	if (stream_error) {
		if (!calls.empty()) {
			for (const auto& call : calls) {
				ToolResult failed;
				failed.is_error = true;
				failed.content.push_back(ContentPart::text_part(tr(config_.language,
					"[响应中断，该工具未执行]", "[the response failed; this tool did not run]")));
				// recorded first, then reported: the event callback can throw (see
				// the note in EXECUTE) and the pairing must not depend on it
				{
					std::lock_guard<std::mutex> lock(state_mutex_);
					messages_.push_back(AgentMessage::tool_result(call.tool_call_id, call.tool_name,
					                                             failed.content, true));
				}
				try {
					emit_tool(AgentEvent::Type::ToolExecutionStart, call.tool_call_id, call.tool_name,
					          call.arguments, "", std::nullopt);
					emit_tool(AgentEvent::Type::ToolExecutionEnd, call.tool_call_id, call.tool_name,
					          call.arguments, "", failed);
				} catch (...) {
				}
			}
		}
		return StepOutcome::Error;
	}
	bool aborted_now = cancelled();
	if (calls.empty()) {
		// no tools → this step's text is the closing summary (or plain abort)
		return aborted_now ? StepOutcome::Aborted : StepOutcome::Done;
	}

	// ═══ EXECUTE ═════════════════════════════════════════════════════════
	// run the whole batch; every call gets a result message even when aborted
	// mid-batch (a dangling tool_use without a result breaks the next request)
	//
	// Each result is appended as soon as its call is answered instead of being
	// collected in a local vector, because the one thing that must survive this
	// function is the *pairing*: `emit_tool` runs the host's event callback, which
	// builds a full snapshot of the history (images and all) and hands it to the
	// WebSocket — an allocation failure or a throw in there used to unwind the
	// whole batch before anything was appended, leaving the assistant message's
	// tool calls unanswered on disk and in memory, and every later request of that
	// session then dying with "an assistant message with 'tool_calls' must be
	// followed by tool messages responding to each 'tool_call_id'". A UI failure
	// is not a reason for the conversation to become unsendable.
	auto append_result = [&](const ContentPart& call, const ToolResult& result) {
		std::lock_guard<std::mutex> lock(state_mutex_);
		messages_.push_back(AgentMessage::tool_result(call.tool_call_id, call.tool_name,
		                                             result.content, result.is_error));
	};
	// an event callback that throws is swallowed here: the tool has already run,
	// and the transcript (not the UI) is what the next request is built from
	auto emit_quiet = [&](AgentEvent::Type type, const ContentPart& call, const ToolResult* r) {
		try {
			emit_tool(type, call.tool_call_id, call.tool_name, call.arguments, "",
			          r ? std::optional<ToolResult>(*r) : std::optional<ToolResult>());
		} catch (...) {
		}
	};
	for (const auto& call : calls) {
		ToolResult result;
		if (cancelled()) {
			result.is_error = true;
			result.content.push_back(ContentPart::text_part("[aborted before execution]"));
			emit_quiet(AgentEvent::Type::ToolExecutionStart, call, nullptr);
			emit_quiet(AgentEvent::Type::ToolExecutionEnd, call, &result);
			append_result(call, result);
			continue;
		}
		emit_quiet(AgentEvent::Type::ToolExecutionStart, call, nullptr);
		result = execute_tool(call, true);
		emit_quiet(AgentEvent::Type::ToolExecutionEnd, call, &result);
		append_result(call, result);
	}
	// background subagents that finished while this batch ran: their reports go
	// in as their own messages at the tail of this round, so the next THINK sees
	// them as context without any of them being folded into a tool result
	deliver_subagent_summaries(false);
	// abort may have landed during the batch: the results above keep the
	// history consistent either way
	return aborted_now || cancelled() ? StepOutcome::Aborted : StepOutcome::Continue;
}

// ── tool execution ─────────────────────────────────────────────────────────

std::string Agent::build_tools_json() const {
	JsonValue arr = JsonValue::array();
	const bool is_zh = (config_.language != "en");
	for (const auto& tool : config_.tools) {
		JsonValue t = JsonValue::object();
		t["name"] = tool.name;
		// two versions of every built-in prompt: pick by the session language,
		// falling back to Chinese when no English translation is registered
		t["description"] = is_zh ? tool.description : (tool.description_en.empty() ? tool.description : tool.description_en);
		// the parameter descriptions are localized the same way, so the schema is
		// materialized for the current language on each request
		t["input_schema"] = tool.parameters_fn ? tool.parameters_fn(config_.language) : JsonValue::object();
		arr.push_back(t);
	}
	return arr.dump();
}

ToolResult Agent::execute_tool(const ContentPart& call, bool emit_updates) {
	ToolResult result;
	std::vector<ToolDef> tools_snapshot;
	{
		std::lock_guard<std::mutex> lock(state_mutex_);
		tools_snapshot = config_.tools;
	}
	const ToolDef* def = nullptr;
	for (const auto& tool : tools_snapshot) {
		if (tool.name == call.tool_name) {
			def = &tool;
			break;
		}
	}
	if (!def) {
		result.is_error = true;
		result.content.push_back(ContentPart::text_part(tr(config_.language,
			"未知工具: " + call.tool_name,
			"Unknown tool: " + call.tool_name)));
		return result;
	}

	ToolContext ctx;
	ctx.cwd = config_.session_cwd;
	ctx.agent_dir = config_.agent_dir;
	ctx.bin_dir = config_.bin_dir;
	ctx.session_id = config_.session_id;
	ctx.system_prompt = config_.system_prompt;
	ctx.language = config_.language;
	ctx.current_tool = def->name;
	ctx.output_limits = config_.tool_policies.output_limits;
	ctx.permissions = config_.tool_policies.permissions;
	ctx.subagent_tools = config_.tool_policies.subagent_tools;
	// 子代理思考强度跟随主会话当前思考级别（设置项已移除）
	ctx.subagent_thinking = config_.thinking_level;
	ctx.cancelled = [this] { return cancelled(); };
	ctx.model = config_.model;
	ctx.stream_fn = stream_fn_;
	ctx.get_tools = [tools_snapshot]() { return tools_snapshot; };
	ctx.resolve_model = config_.resolve_model;
	ctx.resolve_api_key = config_.resolve_api_key;
	// background subagent seam: the agent tool supplies the task, the agent owns
	// the child config and runs it off-thread
	ctx.launch_subagent = [this, call_id = call.tool_call_id](const SubagentRequest& request) {
		return launch_subagent(request, call_id);
	};
	if (emit_updates) {
		ContentPart call_copy = call;
		ctx.on_update = [this, call_id = call.tool_call_id, name = call.tool_name, call_copy](const std::string& partial) {
			emit_tool(AgentEvent::Type::ToolExecutionUpdate, call_id, name, call_copy.arguments, partial, std::nullopt);
		};
	}

	try {
		result = def->run(call.arguments, ctx);
	} catch (const std::exception& e) {
		result = ToolResult();
		result.is_error = true;
		result.content.push_back(ContentPart::text_part(tr(config_.language,
			std::string("工具错误: ") + e.what(),
			std::string("tool error: ") + e.what())));
	} catch (...) {
		result = ToolResult();
		result.is_error = true;
		result.content.push_back(ContentPart::text_part(tr(config_.language,
			"工具错误: 未知", "tool error: unknown")));
	}
	return result;
}

}  // namespace phi
