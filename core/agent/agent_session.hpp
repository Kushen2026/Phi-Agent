// AgentSessionHost: session facade + snapshot projection for the UI protocol.
// Combines the coding-agent AgentSession and the web-ui SessionHost behavior.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "agent/agent.hpp"
#include "util/json.hpp"
#include "llm/models.hpp"
#include "store/resources.hpp"
#include "store/session_store.hpp"
#include "store/stores.hpp"

namespace phi {

// tool execution projection entry (per tool call, current session lifetime)
struct ToolRun {
	std::string tool_call_id;
	std::string name;
	JsonValue args;
	std::string status;  // running | complete | error
	std::string output;
	bool is_error = false;
};

struct HostLog {
	std::string level;  // info | warn | error
	std::string message;
};

struct HostEvent {
	bool is_snapshot = false;  // full projection (structural changes)
	bool is_patch = false;     // incremental stream tick (streaming msg + changed tool runs)
	JsonValue snapshot;
	JsonValue patch;
	HostLog log;
};

class AgentSessionHost {
public:
	AgentSessionHost(std::string base_dir, std::string cwd);
	~AgentSessionHost();

	void set_event_callback(std::function<void(const HostEvent&)> cb) { on_event_ = std::move(cb); }
	void set_folder_picker(std::function<std::optional<std::string>(const std::string& initial)> picker) {
		folder_picker_ = std::move(picker);
	}
	// test seam: override the provider stream function (defaults to stream_llm)
	void set_stream_fn(ProviderStreamFn fn) { stream_fn_ = std::move(fn); }

	// ── lifecycle ──────────────────────────────────────────────────────────
	void create_session(const std::string& name, const JsonValue* model_ref, const std::string& thinking_level,
		bool continue_recent);  // throws std::string
	void open_session(const std::string& session_id);  // empty = continue recent
	void teardown();
	bool is_active() const { return active_; }
	const std::string& session_id() const { return session_id_; }
	const std::string& cwd() const { return cwd_; }
	// switch working directory: tears down and re-opens (applies active profile)
	void change_cwd(const std::string& new_cwd);  // throws std::string
	// set cwd before any session is opened (startup path; no session churn)
	void set_initial_cwd(const std::string& new_cwd);
	const std::string& agent_dir() const { return agent_dir_; }
	void apply_profile_after_start();  // re-applies active profile model/thinking

	// ── commands (throw std::string) ──────────────────────────────────────
	void prompt(const std::string& text, const std::string& mode, const JsonValue& attachments);
	void abort();
	void compact();
	void set_model(const std::string& provider, const std::string& id);
	void set_thinking(const std::string& level);
	void set_system_prompt_language(const std::string& language);
	bool isChinese() const { return settings_->system_prompt_language() != "en"; }
	void set_session_name(const std::string& name);
	void reload_resources();

	// ── listings ───────────────────────────────────────────────────────────
	JsonValue list_models();
	JsonValue list_skills();  // drives the skills panel (and its delete button)
	JsonValue list_commands();
	JsonValue get_snapshot();  // throws when inactive
	std::vector<SessionSummary> list_sessions();
	void delete_session(const std::string& session_id);
	JsonValue settings_json() const;

	// accessors for the protocol layer (config panel endpoints)
	ConfigStore& configs() { return *configs_; }
	SettingsStore& settings() { return *settings_; }
	Resources& resources() { return *resources_; }

	static constexpr int kProtocolVersion = 1;

private:
	void require_active() const;
	void sync_archive();
	// 从回放内容重建会话；model_ref/thinking_level 为显式覆盖（新建路径）。
	void init_session(std::unique_ptr<SessionArchive> archive, SessionContent* restored,
		const std::string& name, const JsonValue* model_ref, const std::string& thinking_level);
	std::vector<ToolDef> build_tools();
	ToolPolicies tool_policies() const;
	std::string build_system_prompt_text();
	void rebuild_system_prompt();
	JsonValue build_snapshot();

	std::string base_dir_;
	std::string agent_dir_;
	std::string bin_dir_;
	std::string cwd_;
	std::string docs_dir_;

	std::unique_ptr<ModelRuntime> runtime_;
	std::unique_ptr<SettingsStore> settings_;
	std::unique_ptr<ConfigStore> configs_;
	std::unique_ptr<Resources> resources_;
	std::unique_ptr<SessionArchive> archive_;
	std::unique_ptr<Agent> agent_;
	std::string system_prompt_;
	std::string session_name_;
	std::string session_id_;
	std::string session_file_;
	std::atomic<bool> active_{false};

	// projection state
	mutable std::mutex projection_mutex_;
	std::vector<std::string> message_ids_;
	int64_t id_counter_ = 0;
	std::string streaming_id_;
	std::map<std::string, ToolRun> tool_runs_;
	// change tracking for stream patches: each run mutation bumps the counter
	// and records the version; patches carry runs whose version exceeds
	// patch_sent_version_ (set to the counter whenever a full snapshot or patch
	// is emitted). Keeps per-tick payload O(change), not O(history).
	std::map<std::string, int64_t> run_versions_;
	int64_t run_version_counter_ = 0;
	int64_t patch_sent_version_ = 0;
	std::string phase_ = "idle";
	std::atomic<int64_t> generation_{0};

	std::function<void(const HostEvent&)> on_event_;
	std::function<std::optional<std::string>(const std::string&)> folder_picker_;
	ProviderStreamFn stream_fn_;  // test seam override; empty = stream_llm

	// worker thread for agent runs
	std::thread worker_;
	std::mutex worker_mutex_;
	std::condition_variable worker_cv_;
	// guards active_/agent_ between worker jobs and teardown (a job enqueued
	// after wait_worker_idle() cleared the queue could otherwise run while
	// teardown resets agent_ — use-after-free during session switch)
	std::mutex lifecycle_mutex_;
	// serializes teardown() itself (open/create/change_cwd can race each other
	// from multiple WS clients); the early abort() must read agent_ safely
	std::mutex teardown_mutex_;
	std::deque<std::function<void()>> worker_jobs_;
	std::atomic<bool> worker_quit_{false};
	std::atomic<bool> job_running_{false};
	// snapshot timer thread: debounced snapshot emission independent of the
	// worker queue (which is busy while the agent streams). Mirrors the pi
	// platform's scheduleSnapshot() (web-ui session-host.ts): an independent
	// timer emits directly, never queued behind the agent run.
	std::thread snapshot_thread_;
	std::mutex snapshot_mutex_;
	std::condition_variable snapshot_cv_;
	bool snapshot_scheduled_ = false;  // full snapshot pending
	bool patch_scheduled_ = false;     // incremental stream patch pending
	bool snapshot_quit_ = false;
	void worker_loop();
	void run_on_worker(std::function<void()> job);
	void wait_worker_idle();
	std::unique_lock<std::mutex> acquire_agent();
	void set_phase(const std::string& phase);
	void schedule_snapshot(bool structural);
	void snapshot_timer_loop();
	void emit_patch_now();
	void emit_snapshot_now();
	void emit_log(const std::string& level, const std::string& message);
	void dispatch(const HostEvent& ev);
	JsonValue serialize_tool_run(const ToolRun& run) const;
	bool build_stream_patch(JsonValue& patch);
	void on_agent_event(const AgentEvent& ev);
	std::vector<AgentMessage> transform_context(const std::vector<AgentMessage>& messages) const;
};

}  // namespace phi
