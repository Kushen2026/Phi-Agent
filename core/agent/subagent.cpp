#include "agent/subagent.hpp"

#include <algorithm>

#include "util/base.hpp"
#include "store/session_store.hpp"

namespace phi {

std::string render_session_transcript(const std::vector<AgentMessage>& messages) {
	std::string out;
	for (const auto& m : messages) {
		out += message_to_json(m).dump(-1);
		out += "\n";
	}
	return out;
}

namespace {

// the child's closing summary = its last assistant message that carried text
std::string last_assistant_text(const std::vector<AgentMessage>& msgs) {
	for (size_t i = msgs.size(); i-- > 0;) {
		if (msgs[i].role != "assistant") continue;
		std::string out;
		for (const auto& part : msgs[i].content) {
			if (part.type == ContentPart::Type::Text) out += part.text;
		}
		if (!out.empty()) return out;
	}
	return "";
}

void default_stream(const ModelDef& model, const std::vector<AgentMessage>& context,
	const StreamOptions& options, const StreamCallback& on_event) {
	stream_llm(model, context, options, on_event);
}

}  // namespace

struct SubagentPool::Task {
	SubagentSpec spec;
	UpdateFn on_update;
	DoneFn on_done;
	std::atomic<bool> alive{true};     // cleared when the pool goes away
	std::atomic<bool> cancel{false};   // set by cancel_all() / ~SubagentPool
	std::atomic<bool> finished{false};
	std::thread thread;
};

SubagentPool::~SubagentPool() {
	cancel_all();
	std::vector<std::shared_ptr<Task>> tasks;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		tasks.swap(tasks_);
	}
	// join every child before returning: the callbacks reach back into the
	// owning agent, so a detached thread would outlive it
	for (auto& task : tasks) {
		task->alive = false;
		if (task->thread.joinable()) task->thread.join();
	}
}

void SubagentPool::cancel_all() {
	std::lock_guard<std::mutex> lock(mutex_);
	for (auto& task : tasks_) task->cancel = true;
}

size_t SubagentPool::running() const {
	std::lock_guard<std::mutex> lock(mutex_);
	size_t n = 0;
	for (const auto& task : tasks_) {
		if (!task->finished.load()) n++;
	}
	return n;
}

void SubagentPool::start(SubagentSpec spec, UpdateFn on_update, DoneFn on_done) {
	auto task = std::make_shared<Task>();
	task->spec = std::move(spec);
	task->on_update = std::move(on_update);
	task->on_done = std::move(on_done);

	// the child polls this: its own abort plus "the pool is shutting down".
	// weak_ptr (not shared_ptr) so the callback never keeps the task alive.
	std::weak_ptr<Task> weak = task;
	std::function<bool()> upstream = task->spec.config.cancelled;
	task->spec.config.cancelled = [weak, upstream]() {
		auto self = weak.lock();
		if (!self) return true;  // pool gone
		if (self->cancel.load()) return true;
		return upstream ? upstream() : false;
	};

	std::shared_ptr<Task> shared = task;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		// drop finished tasks so a long session does not accumulate them
		tasks_.erase(std::remove_if(tasks_.begin(), tasks_.end(),
			[](const std::shared_ptr<Task>& t) { return t->finished.load(); }), tasks_.end());
		tasks_.push_back(task);
		shared->thread = std::thread([shared]() {
			Agent::Config config = shared->spec.config;
			ProviderStreamFn stream = shared->spec.stream_fn ? shared->spec.stream_fn : default_stream;
			Agent agent(std::move(config), stream);

			std::string transcript;
			agent.set_event_callback([&](const AgentEvent& ev) {
				if (ev.type != AgentEvent::Type::MessageEnd &&
					ev.type != AgentEvent::Type::ToolExecutionEnd) {
					return;
				}
				// the log is the child's transcript in the session's message
				// format, rebuilt whenever a message is finalized
				transcript = render_session_transcript(agent.messages());
				if (shared->alive.load() && shared->on_update) {
					shared->on_update(shared->spec.id, transcript);
				}
			});

			try {
				agent.prompt(shared->spec.task);
			} catch (...) {
				// a throw here still produces whatever the child managed to record
			}

			auto messages = agent.messages();
			if (transcript.empty()) transcript = render_session_transcript(messages);

			SubagentOutcome outcome;
			outcome.id = shared->spec.id;
			outcome.description = shared->spec.description;
			outcome.transcript = std::move(transcript);
			outcome.output = last_assistant_text(messages);
			outcome.is_error = shared->cancel.load() || outcome.output.empty();

			shared->finished = true;
			if (shared->alive.load() && shared->on_done) shared->on_done(outcome);
		});
	}
}

}  // namespace phi
