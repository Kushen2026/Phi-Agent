#include "agent/agent_session.hpp"
#include "util/trace.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <set>

#include "util/base.hpp"
#include "agent/system_prompt.hpp"
#include "tools/tools.hpp"
#include "tools/tools_helpers.hpp"
#include "tools/tools_subagent.hpp"
#include "util/i18n.hpp"
#include "tools/tools_web.hpp"

namespace phi {

// optional heap-guard scan hook (defined by tests/probe_guard.cpp when linked)
extern "C" void phi_guard_scan(const char*) __attribute__((weak));
#define PHI_GUARD_SCAN(tag) do { if (phi_guard_scan) phi_guard_scan(tag); } while (0)

// collect only top-level files (one level deep) under a directory
static void collect_top_level_files(const std::string& dir, std::vector<std::string>& out) {
	auto names = list_dir(dir);
	for (const auto& name : names) {
		std::string full = path_join(dir, name);
		if (!path_is_dir(full)) {
			out.push_back(path_backslash(full));
		}
	}
}

static std::string flatten_text(const std::vector<ContentPart>& content) {
	std::string out;
	for (const auto& part : content) {
		if (part.type == ContentPart::Type::Text) out += part.text;
	}
	return out;
}

// rough token estimate: ~1 token per CJK codepoint, ~4 chars per token for
// ASCII text (bytes/4 badly underestimates Chinese text)
static int64_t estimate_tokens(const std::string& text) {
	int64_t tokens = 0, ascii = 0;
	for (size_t i = 0; i < text.size();) {
		unsigned char c = (unsigned char)text[i];
		if (c < 0x80) {
			ascii++;
			i++;
		} else if ((c & 0xC0) == 0x80) {
			i++;  // stray continuation byte
		} else {
			tokens++;
			i += c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : 2;
		}
	}
	return tokens + ascii / 4;
}

static void validate_dir(const std::string& path) {
	if (!path_is_dir(path)) throw std::string("不是有效的目录: ") + path;
}

// ── lifecycle ──────────────────────────────────────────────────────────────

AgentSessionHost::AgentSessionHost(std::string base_dir, std::string cwd)
	: base_dir_(std::move(base_dir)), cwd_(std::move(cwd)) {
	validate_dir(cwd_);
	agent_dir_ = path_join(base_dir_, "data");
	bin_dir_ = path_join(agent_dir_, "bin");
	if (!path_is_dir(bin_dir_)) bin_dir_ = path_join(base_dir_, "vendor/bin");
	docs_dir_ = path_join(base_dir_, "docs");

	PHI_GUARD_SCAN("host-paths");
	runtime_ = std::make_unique<ModelRuntime>(agent_dir_);
	settings_ = std::make_unique<SettingsStore>(agent_dir_);
	configs_ = std::make_unique<ConfigStore>(agent_dir_, *runtime_, *settings_);
	// wire key provider: resolve API keys from ConfigStore profiles (config.json).
	// Resolution order (for each profile):
	//   1. Direct match: p.provider == provider (e.g. catalog providers like "anthropic")
	//   2. Gateway profile: gw_id == provider, return p.api_key (main profile key)
	//   NOTE: gw-subagent is registered directly in user_providers_ with its own api_key,
	//         so it's resolved by ModelRuntime::resolve_api_key after this callback returns.
	runtime_->set_key_provider_callback([this](const std::string& provider) -> std::optional<std::string> {
		auto listing = configs_->list();
		for (const auto& p : listing.profiles) {
			// 1) direct provider match (catalog entries have provider == id)
			if (p.provider == provider && !p.api_key.empty()) return p.api_key;
			// 2) gateway profile match: use the main profile's api_key
			std::string gw_id = configs_->gateway_provider_id(p.name);
			if (gw_id == provider && !p.api_key.empty()) return p.api_key;
		}
		return std::nullopt;
	});
	resources_ = std::make_unique<Resources>(agent_dir_, cwd_);




	worker_ = std::thread(&AgentSessionHost::worker_loop, this);
	snapshot_thread_ = std::thread(&AgentSessionHost::snapshot_timer_loop, this);

}

AgentSessionHost::~AgentSessionHost() {
	teardown();
	{
		std::lock_guard<std::mutex> lock(snapshot_mutex_);
		snapshot_quit_ = true;
	}
	snapshot_cv_.notify_all();
	if (snapshot_thread_.joinable()) snapshot_thread_.join();
	worker_quit_ = true;
	worker_cv_.notify_all();
	if (worker_.joinable()) worker_.join();  // join (not detach): detached worker would touch freed state
}

void AgentSessionHost::worker_loop() {
	while (!worker_quit_) {
		std::function<void()> job;
		{
			std::unique_lock<std::mutex> lock(worker_mutex_);
			worker_cv_.wait(lock, [&] { return worker_quit_ || !worker_jobs_.empty(); });
			if (worker_quit_ && worker_jobs_.empty()) return;
			job = std::move(worker_jobs_.front());
			worker_jobs_.pop_front();
			// inside the critical section: wait_worker_idle() clears the queue and
			// then polls job_running_. Setting it here closes the window where the
			// job was popped but not yet marked running — otherwise teardown() could
			// see "idle", destroy the agent, and then this job would run on freed
			// state (use-after-free during session switch / cwd change).
			job_running_ = true;
		}
		job();
		job_running_ = false;
	}
}

void AgentSessionHost::run_on_worker(std::function<void()> job) {
	{
		std::lock_guard<std::mutex> lock(worker_mutex_);
		worker_jobs_.push_back(std::move(job));
	}
	worker_cv_.notify_one();
}

void AgentSessionHost::wait_worker_idle() {
	// drop pending jobs, then wait for the running one to finish
	{
		std::lock_guard<std::mutex> lock(worker_mutex_);
		worker_jobs_.clear();
	}
	while (job_running_) std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

// worker jobs hold this for their whole runtime: teardown() takes it after
// wait_worker_idle(), so the lock both validates and pins agent_ for the job
std::unique_lock<std::mutex> AgentSessionHost::acquire_agent() {
	return std::unique_lock<std::mutex>(lifecycle_mutex_);
}

void AgentSessionHost::set_phase(const std::string& phase) {
	// phase_ is read by build_snapshot under projection_mutex_; unsynchronized
	// writes from WS threads were UB (and could be clobbered by agent events)
	std::lock_guard<std::mutex> lock(projection_mutex_);
	phase_ = phase;
}

JsonValue AgentSessionHost::settings_json() const {
	JsonValue s = JsonValue::object();
	s["cwd"] = cwd_;
	s["agentDir"] = agent_dir_;
	s["version"] = std::to_string(kProtocolVersion);
	return s;
}

void AgentSessionHost::teardown() {
	// serialize teardowns (multiple WS clients can trigger open/create/cwd
	// concurrently); the early abort() reads agent_ which only teardown mutates
	std::lock_guard<std::mutex> tc(teardown_mutex_);
	// abort FIRST, then wait: a session switch / cwd change issued from the WS
	// reader thread must not block for the remaining runtime of a long agent
	// run — with abort set, the stream tears down, running bash is killed and
	// subagents cancel, so the wait below is bounded (was: unbounded)
	if (agent_) agent_->abort();
	wait_worker_idle();
	{
		// Drop pending snapshot emissions and tear the agent down under the same
		// mutex the snapshot timer thread holds while building a snapshot, so an
		// in-flight emission has finished before agent_ disappears. active_ goes
		// false first so the timer thread never observes a live session with a
		// null agent.
		// lifecycle_mutex_: a job enqueued after wait_worker_idle() cleared the
		// queue could otherwise run while agent_ is reset here (use-after-free).
		std::lock_guard<std::mutex> lc(lifecycle_mutex_);
		std::lock_guard<std::mutex> lock(snapshot_mutex_);
		snapshot_scheduled_ = false;
		patch_scheduled_ = false;
		active_ = false;
		if (agent_) agent_->abort();
		agent_.reset();
	}
	// 新建但从未写入过任何消息的会话文件直接丢弃，避免空文件堆积。
	if (archive_) archive_->discard_if_empty();
	archive_.reset();
	{
		std::lock_guard<std::mutex> lock(projection_mutex_);
		message_ids_.clear();
		id_counter_ = 0;
		streaming_id_.clear();
		tool_runs_.clear();
		run_versions_.clear();
		run_version_counter_ = 0;
		patch_sent_version_ = 0;
	}
	set_phase("idle");
	active_ = false;
	session_id_.clear();
	session_file_.clear();
}

void AgentSessionHost::change_cwd(const std::string& new_cwd) {
	std::string resolved = path_absolute(new_cwd);
	validate_dir(resolved);
	if (resolved == cwd_) return;
	// Abort FIRST (before acquiring teardown_mutex_) so the worker sees the flag
	// immediately and can exit its current tool run rather than polling until a
	// timeout. Then acquire teardown_mutex_ only for the state mutations that
	// follow — this avoids the classic deadlock where teardown() needs
	// lifecycle_mutex_ while a running job holds it and checks abort_requested_.
	if (agent_) agent_->abort();
	// declared outside the block below: the open_session() rollback path also
	// needs to restore the previous cwd
	std::string old_cwd = cwd_;
	{
		// Call teardown() directly without holding teardown_mutex_ first.
		// teardown() acquires teardown_mutex_ internally to serialize against
		// concurrent teardowns.  Holding it here and then calling teardown()
		// would deadlock because std::mutex is non-recursive.
		teardown();
		cwd_ = resolved;
		try {
			resources_ = std::make_unique<Resources>(agent_dir_, cwd_);
			configs_->save_cwd(cwd_);
		} catch (...) {
			// Resources construction / cwd persistence must not leave the host
			// pointing at a directory whose session never opened: roll back so
			// the object stays consistent (the old session was already torn down,
			// but state remains valid and the caller sees the error).
			cwd_ = old_cwd;
			resources_ = std::make_unique<Resources>(agent_dir_, cwd_);
			throw;
		}
	}
	try {
		open_session("");
	} catch (...) {
		// open_session failed after the cwd was already switched: roll the cwd
		// back and rebuild resources so a retry operates on a consistent state
		// instead of a host whose cwd_/resources_/archive_ disagree.
		cwd_ = old_cwd;
		try {
			resources_ = std::make_unique<Resources>(agent_dir_, cwd_);
		} catch (...) {
			emit_log("error", "切换目录失败后恢复资源失败，请重启应用");
		}
		throw;
	}
	apply_profile_after_start();
}

void AgentSessionHost::set_initial_cwd(const std::string& new_cwd) {
	std::string resolved = path_absolute(new_cwd);
	validate_dir(resolved);
	cwd_ = resolved;
	resources_ = std::make_unique<Resources>(agent_dir_, cwd_);
	configs_->save_cwd(cwd_);
}

// ── session construction ───────────────────────────────────────────────────

void AgentSessionHost::open_session(const std::string& session_id) {
	teardown();
	// 定位目标文件：指定 id -> 精确匹配；否则取最近更新的会话。
	SessionArchive probe(agent_dir_, cwd_);
	std::optional<std::string> path = session_id.empty() ? probe.latest() : probe.locate(session_id);
	if (session_id.empty() && !path) {
		create_session("", nullptr, "", false);
		return;
	}
	if (!path) {
		// 指定 ID 找不到。常见原因是当前就是空会话：teardown() 已经把它的
		// 空文件 discard_if_empty() 掉了，此时 locate(id) 必然失败。
		// 必须新建空会话，绝不能 fallback 到 probe.latest()——否则会打开
		// 上一个有内容的旧对话，表现就是"切换配置/重启会话后跳到了别的对话"。
		emit_log("warn", "Session not found: " + session_id + ", creating a new session");
		create_session("", nullptr, "", false);
		return;
	}
	auto content = std::make_unique<SessionContent>(SessionArchive::replay(*path));
	if (content->session_id.empty()) {
		throw std::string("Session file is invalid: ") + *path;
	}
	auto archive = std::make_unique<SessionArchive>(agent_dir_, cwd_);
	archive->attach(*path);
	// 回放得到的元数据回填内存，随第一次 sync 以新格式重新落盘。
	if (!content->provider.empty()) archive->set_model(content->provider, content->model_id);
	if (!content->thinking_level.empty()) archive->set_thinking(content->thinking_level);
	if (!content->title.empty()) archive->set_title(content->title);
	init_session(std::move(archive), content.get(), "", nullptr, "");
	for (const auto& fix : content->repairs) emit_log("warn", fix);
	apply_profile_after_start();
}

void AgentSessionHost::create_session(const std::string& name, const JsonValue* model_ref,
	const std::string& thinking_level, bool continue_recent) {
	(void)continue_recent;
	teardown();
	// 创建新文件（SessionArchive::start 只写头部）；SessionContent 为空。
	auto content = std::make_unique<SessionContent>();
	auto archive = std::make_unique<SessionArchive>(agent_dir_, cwd_);
	archive->start();
	init_session(std::move(archive), content.get(), name, model_ref, thinking_level);
	apply_profile_after_start();
}

void AgentSessionHost::init_session(std::unique_ptr<SessionArchive> archive, SessionContent* restored,
	const std::string& name, const JsonValue* model_ref, const std::string& thinking_level) {
	resources_->reload();
	system_prompt_ = build_system_prompt_text();

	// initial model: explicit -> restored -> settings default -> first with auth
	std::optional<ModelDef> model;
	std::string fallback_message;
	if (model_ref != nullptr) {
		std::string provider = (*model_ref)["provider"].as_string("");
		std::string id = (*model_ref)["id"].as_string("");
		model = runtime_->get_model(provider, id);
		if (!model) throw std::string("Unknown model ") + provider + "/" + id;
	}
	if (!model && restored && !restored->provider.empty()) {
		model = runtime_->get_model(restored->provider, restored->model_id);
		if (model && !runtime_->has_configured_auth(model->provider)) {
			model = std::nullopt;
		}
		if (!model) {
			fallback_message = "Could not restore model " + restored->provider + "/" + restored->model_id;
		}
	}
	if (!model) {
		auto dp = settings_->default_provider();
		auto dm = settings_->default_model();
		if (dp && dm) model = runtime_->get_model(*dp, *dm);
	}
	if (!model) {
		for (const auto& m : runtime_->get_models()) {
			if (runtime_->has_configured_auth(m.provider)) {
				model = m;
				break;
			}
		}
	}

	std::string level = thinking_level.empty()
		? (restored && !restored->thinking_level.empty() ? restored->thinking_level
			: settings_->default_thinking_level().value_or("medium"))
		: thinking_level;
	if (level.empty()) level = "medium";
	if (model) {
		if (level != "off" && !model->supports_thinking(level)) {
			auto supported = model->supported_thinking_levels();
			if (std::find(supported.begin(), supported.end(), level) == supported.end()) level = "medium";
		}
	} else {
		level = "off";
		if (fallback_message.empty()) {
			fallback_message = "没有可用的 API 配置";
		}
	}

	archive_ = std::move(archive);
	session_id_ = archive_ ? archive_->id() : std::string();
	session_file_ = archive_ ? archive_->file() : std::string();
	session_name_ = name;
	if (restored && !restored->title.empty() && session_name_.empty()) {
		session_name_ = restored->title;
	}

	Agent::Config config;
	config.system_prompt = system_prompt_;
	config.model = model;
	config.thinking_level = level;
	config.language = settings_->system_prompt_language();
	config.session_id = session_id_;
	config.session_cwd = cwd_;
	config.agent_dir = agent_dir_;
	config.bin_dir = bin_dir_;
	config.tools = build_tools();
	config.tool_policies = tool_policies();
	config.transform_context = [this](const std::vector<AgentMessage>& msgs) {
		return transform_context(msgs);
	};
	config.resolve_api_key = [this](const ModelDef& m) {
		auto key = runtime_->resolve_api_key(m.provider);
		return key.value_or("");
	};
	config.resolve_model = [this](const std::string& provider, const std::string& id) {
		return runtime_->get_model(provider, id);
	};
	// Background subagents deliver their results through the session worker, so
	// the delivery turn runs on the same thread as every other agent run (the
	// projection/archive bookkeeping is not safe to touch from a child thread).
	config.background_prompt = [this](std::string message) {
		run_on_worker([this, message]() {
			try {
				auto lc = acquire_agent();
				if (!active_ || !agent_) return;  // session switched while queued
				agent_->prompt(message);
			} catch (const std::string& e) {
				emit_log("error", e);
			} catch (const std::exception& e) {
				emit_log("error", e.what());
			}
		});
	};

	ProviderStreamFn stream = stream_fn_;
	if (!stream) {
		stream = [](const ModelDef& m, const std::vector<AgentMessage>& ctx, const StreamOptions& opts,
			const StreamCallback& cb) { stream_llm(m, ctx, opts, cb); };
	}
	agent_ = std::make_unique<Agent>(std::move(config), std::move(stream));
	agent_->set_event_callback([this](const AgentEvent& ev) { on_agent_event(ev); });

	if (restored) {
		// Fix empty tool_call_ids before setting messages
		for (auto& m : restored->messages) {
			if (m.role != "assistant") continue;
			for (auto& p : m.content) {
				if (p.type == ContentPart::Type::ToolCall && p.tool_call_id.empty()) {
					p.tool_call_id = "call_recovered_" + uuid4().substr(0, 12);
				}
			}
		}
		agent_->set_messages(restored->messages);
	} else {
		// 新会话：把初始元数据记入内存，随第一次 sync 一并落盘。
		if (model) archive_->set_model(model->provider, model->id);
		archive_->set_thinking(level);
		if (!name.empty()) archive_->set_title(name);
	}

	set_phase("idle");
	active_ = true;
	{
		std::lock_guard<std::mutex> lock(projection_mutex_);
		message_ids_.clear();
		id_counter_ = 0;
		streaming_id_.clear();
		tool_runs_.clear();
		run_versions_.clear();
		run_version_counter_ = 0;
		patch_sent_version_ = 0;
		if (restored) {
			// rebuild the tool-run projection from the restored history: tool
			// status lives only in tool_runs_ (populated by live ToolExecution
			// events, which never replay), and an empty map made every restored
			// tool card fall back to the UI's "complete" default — failed calls
			// lost their red state after a restart
			for (const auto& m : restored->messages) {
				if (m.role == "assistant") {
					for (const auto& p : m.content) {
						if (p.type != ContentPart::Type::ToolCall) continue;
						ToolRun run;
						run.tool_call_id = p.tool_call_id;
						run.name = p.tool_name;
						run.args = p.arguments;
						// placeholder for a dangling call (no result message in the
						// file); corrected below when the result exists
						run.status = "error";
						run.is_error = true;
						run.output = "(no result recorded)";
						tool_runs_[p.tool_call_id] = std::move(run);
					}
				} else if (m.role == "toolResult") {
					auto it = tool_runs_.find(m.tool_call_id);
					if (it == tool_runs_.end()) continue;
					std::string text;
					for (const auto& p : m.content) {
						if (p.type == ContentPart::Type::Text) text += p.text;
					}
					it->second.status = m.is_error ? "error" : "complete";
					it->second.is_error = m.is_error;
					it->second.output = std::move(text);
				}
			}
		}
	}
	emit_snapshot_now();
	if (!fallback_message.empty()) emit_log("warn", fallback_message);
}

// ── system prompt / tools ──────────────────────────────────────────────────

std::vector<ToolDef> AgentSessionHost::build_tools() {
	// the full built-in set: file/command tools + web tools + subagent tool
	std::vector<ToolDef> tools = build_file_tools();
	tools.push_back(web_search_tool());
	tools.push_back(fetch_tool());
	// the native media engine: Qwen-Image-2.1 text-to-image / reference edit,
	// and the H3 video path. The weights are only read on the first call.
	for (ToolDef& mt : build_media_tools()) tools.push_back(std::move(mt));
	tools.push_back(build_agent_tool());

	// user tool settings: drop disabled tools (settings → 工具设置)
	JsonValue ts = settings_->tool_settings();
	const JsonValue* enabled = ts.find("enabled");
	std::vector<ToolDef> filtered;
	for (auto& t : tools) {
		bool on = enabled ? (*enabled)[t.name].as_bool(true) : true;
		if (on) filtered.push_back(std::move(t));
	}
	return filtered;
}

ToolPolicies AgentSessionHost::tool_policies() const {
	JsonValue ts = settings_->tool_settings();
	ToolPolicies p;
	if (const JsonValue* limits = ts.find("outputLimits")) {
		if (limits->is_object()) {
			for (const auto& [name, v] : limits->entries()) {
				int64_t n = v.as_int(0);
				if (n > 0) p.output_limits[name] = (size_t)n;
			}
		}
	}
	if (const JsonValue* perms = ts.find("permissions")) {
		if (perms->is_object()) {
			for (const auto& [name, v] : perms->entries()) {
				std::string mode = v.as_string("everywhere");
				if (mode == "in-cwd") p.permissions[name] = mode;
			}
		}
	}
	if (const JsonValue* sub = ts.find("subagent")) {
		const JsonValue* tools_arr = sub->find("tools");
		if (tools_arr && tools_arr->is_array()) {
			for (const auto& v : tools_arr->items()) {
				std::string name = v.as_string("");
				if (!name.empty()) p.subagent_tools.insert(name);
			}
		}
		// note: subagent thinking is no longer user-configurable — subagents
		// follow the parent session's current thinking level
	}
	return p;
}

void AgentSessionHost::rebuild_system_prompt() {
	system_prompt_ = build_system_prompt_text();
	// note: the prompt is fixed per session in this port; it refreshes on the
	// next session create/open or cwd change
}

std::string AgentSessionHost::build_system_prompt_text() {
	SystemPromptOptions opts;
	opts.language = settings_->system_prompt_language();
	opts.cwd = cwd_;
	opts.cwd_files.clear();
	collect_top_level_files(cwd_, opts.cwd_files);
	const bool zh = lang_is_zh(opts.language);
	// actual tool names + one-line snippets from the live tool set (localized to
	// the session's system-prompt language)
	{
		for (const auto& t : build_tools()) {
			opts.selected_tools.push_back(t.name);
			std::string snippet = zh || t.prompt_snippet_en.empty() ? t.prompt_snippet : t.prompt_snippet_en;
			opts.tool_snippets.push_back({t.name, std::move(snippet)});
		}
	}
	opts.docs_readme_path = path_join(docs_dir_, "readme.md");
	opts.docs_dir = docs_dir_;
	opts.docs_examples_dir = path_join(docs_dir_, "examples");
	return build_system_prompt(opts);
}

// ── commands ───────────────────────────────────────────────────────────────

void AgentSessionHost::require_active() const {
	if (!active_ || !agent_) throw std::string("No active session. Create one first.");
}

void AgentSessionHost::sync_archive() {
	if (!archive_ || !agent_) return;
	auto msgs = agent_->messages();
	if (msgs.empty()) return;  // 没有历史不落盘（新会话保持只有头部的空态）
	archive_->sync(msgs);
}

void AgentSessionHost::prompt(const std::string& text, const std::string& mode, const JsonValue& attachments) {
	require_active();
	std::string prompt_text = text;
	std::vector<ContentPart> images;
	std::vector<std::string> file_notes;
	if (attachments.is_array()) {
		for (const auto& att : attachments.items()) {
			std::string path = att["path"].as_string("");
			std::string name = att["name"].as_string("file");
			std::string mime = att["mimeType"].as_string("application/octet-stream");
			if (path.empty()) continue;
			if (starts_with(mime, "image/")) {
				std::vector<uint8_t> bytes;
				if (read_file_bytes(path, bytes)) {
					// the same re-encode the read tool applies: an attachment is a
					// full-size photo far more often than not, and it is kept in the
					// transcript and re-sent with every later request
					images.push_back(tool_model_image(bytes, mime).part);
				} else {
					emit_log("warn", "无法读取图片附件「" + name + "」");
				}
			} else {
				file_notes.push_back(name + " — " + path);
			}
		}
	}
	if (!file_notes.empty()) {
		// attachment block attached to the user turn: two versions, picked by the
		// session language (the model should see it in the language it is running in)
		prompt_text += isChinese()
			? "\n\n[附件文件]\n" + join(file_notes, "\n")
			: "\n\n[Attached files]\n" + join(file_notes, "\n");
	}

	std::string final_text = prompt_text;
	std::vector<ContentPart> final_images = std::move(images);
	bool steer_mode = mode == "steer";
	run_on_worker([this, final_text, final_images, steer_mode]() {
		try {
			auto lc = acquire_agent();
			if (!active_ || !agent_) return;  // session switched while queued
			if (steer_mode) {
				agent_->steer(final_text, final_images);
			} else {
				agent_->prompt(final_text, final_images);
			}
		} catch (const std::string& e) {
			emit_log("error", e);
		} catch (const std::exception& e) {
			emit_log("error", e.what());
		}
	});
}

void AgentSessionHost::abort() {
	require_active();
	{
		std::lock_guard<std::mutex> lock(snapshot_mutex_);
		if (agent_) agent_->abort();
	}
	emit_snapshot_now();
}

void AgentSessionHost::compact() {
	require_active();
	set_phase("compacting");
	emit_snapshot_now();
	run_on_worker([this]() {
		try {
			auto lc = acquire_agent();
			if (!active_ || !agent_) { set_phase("idle"); return; }
			auto messages = agent_->messages();
			size_t start = 0;
			for (size_t i = messages.size(); i > 0; i--) {
				if (messages[i - 1].role == "compactionSummary") {
					start = i - 1;  // include the previous summary: re-compaction must keep its information
					break;
				}
			}
			std::vector<AgentMessage> to_summarize(messages.begin() + start, messages.end());
			// context size before compaction: the most recent request's usage is
			// the true figure (its input already includes all earlier turns);
			// summing per-turn totals would double-count the shared history
			int64_t tokens_before = 0;
			for (auto it = to_summarize.rbegin(); it != to_summarize.rend(); ++it) {
				if (it->role == "assistant" && it->has_usage && it->usage.total_tokens > 0) {
					tokens_before = it->usage.total_tokens;
					break;
				}
			}
			if (tokens_before <= 0) {
				// provider never reported usage — estimate from the text itself
				std::string flat;
				for (const auto& m : to_summarize) {
					flat += flatten_text(m.content);
					if (m.role == "compactionSummary") flat += m.summary;
					flat += "\n";
				}
				tokens_before = estimate_tokens(flat);
			}

			auto L = [&](const std::string& zh, const std::string& en) { return isChinese() ? zh : en; };

			std::string transcript;
			for (const auto& m : to_summarize) {
				if (m.role == "compactionSummary") {
					transcript += L("上一条摘要：", "PREVIOUS SUMMARY: ") + m.summary + "\n\n";
				} else if (m.role == "user") {
					transcript += L("用户：", "USER: ") + flatten_text(m.content) + "\n\n";
				} else if (m.role == "assistant") {
					transcript += L("助手：", "ASSISTANT: ") + flatten_text(m.content) + "\n\n";
				} else if (m.role == "toolResult") {
					std::string text = flatten_text(m.content);
					if (text.size() > 2000) text = text.substr(0, 2000);
					transcript += L("工具(", "TOOL(") + m.tool_name + L("): ", "): ") + text + "\n\n";
				} else if (m.role == "subagentSummary") {
					transcript += L("子代理摘要(", "SUBAGENT SUMMARY(") + m.subagent_description +
						L("): ", "): ") + flatten_text(m.content) + "\n\n";
				}
			}
			std::string summarizer_system = isChinese()
				? "暂停前面的任务，将之前的对话历史总结为摘要。\n\n1.  总结用户提出的每一个要求或约束。\n2. 总结已经完成的内容。\n3. 总结没有完成的工作内容。\n4. 其他重要信息。"
				: "Pause the previous task and summarize the previous conversation history into a summary.\n\n1. Summarize every request or constraint the user made.\n2. Summarize what has already been completed.\n3. Summarize what has not been completed.\n4. Other important information.";
			std::string summary;
			std::string error;
			std::optional<ModelDef> model = agent_->model();
			if (!model) throw std::string("No model configured; cannot compact.");
			StreamOptions opts;
			opts.system_prompt = summarizer_system;
			opts.thinking_level = "off";
			opts.session_id = session_id_;
			if (auto key = runtime_->resolve_api_key(model->provider)) opts.api_key = *key;
			// respect the stream-function seam (same override the agent loop uses)
			ProviderStreamFn stream = stream_fn_;
			if (!stream) {
				stream = [](const ModelDef& m, const std::vector<AgentMessage>& ctx, const StreamOptions& o,
					const StreamCallback& cb) { stream_llm(m, ctx, o, cb); };
			}
			// transcript must be appended to BOTH language branches: without
			// parens the ternary binds looser than +, so the Chinese branch sent
			// only the header line and the model saw no history at all (compaction
			// produced "无对话历史" summaries while tokensBefore showed 200k+)
			std::string compact_request = isChinese()
				? ("对话记录（请压缩）：\n\n" + transcript)
				: ("Conversation transcript to summarize:\n\n" + transcript);
			stream(*model, {AgentMessage::user_text(compact_request)}, opts,
				[&](const StreamEvent& ev) -> bool {
					if (ev.type == StreamEvent::Type::TextDelta) summary += ev.text;
					if (ev.type == StreamEvent::Type::Error) {
						error = ev.text;
						return false;
					}
					return true;
				});
			if (!error.empty()) throw std::string("compaction failed: ") + error;
			if (trim(summary).empty()) throw std::string("compaction failed: empty summary");

			AgentMessage compaction;
			compaction.role = "compactionSummary";
			compaction.summary = summary;
			compaction.tokens_before = tokens_before;
			// estimate post-compaction tokens from the summary text itself
			compaction.tokens_after = estimate_tokens(summary);
			compaction.timestamp = now_ms();
			// 压缩后的历史整体替换：置脏后由 sync 以完整新内容重写文件，
			// 磁盘上不再残留被压缩掉的旧历史。
			std::vector<AgentMessage> compacted;
			compacted.push_back(std::move(compaction));
			agent_->set_messages(std::move(compacted));
			if (archive_) {
				archive_->mark_dirty();
				sync_archive();
			}
			emit_log("info", "会话已压缩：压缩前 " + std::to_string(tokens_before) + " tokens → 压缩后约 " + std::to_string(compaction.tokens_after) + " tokens");
		} catch (const std::string& e) {
			emit_log("error", e);
		} catch (const std::exception& e) {
			emit_log("error", e.what());
		}
		set_phase("idle");
		emit_snapshot_now();
	});
}

void AgentSessionHost::set_model(const std::string& provider, const std::string& id) {
	require_active();
	auto model = runtime_->get_model(provider, id);
	if (!model) throw std::string("Unknown model ") + provider + "/" + id;
	{
		std::lock_guard<std::mutex> lock(snapshot_mutex_);
		if (!active_ || !agent_) throw std::string("No active session. Create one first.");
		agent_->set_model(*model);
		if (archive_) {
			archive_->set_model(provider, id);
			// inline of sync_archive(): the pointers must stay pinned by
			// snapshot_mutex_ across the archive_->sync() call too
			auto msgs = agent_->messages();
			if (!msgs.empty()) archive_->sync(msgs);
		}
	}
	emit_snapshot_now();
}

void AgentSessionHost::set_thinking(const std::string& level) {
	require_active();
	{
		std::lock_guard<std::mutex> lock(snapshot_mutex_);
		if (!active_ || !agent_) throw std::string("No active session. Create one first.");
		auto model = agent_->model();
		if (model && level != "off" && !model->supports_thinking(level)) {
			throw std::string("思考级别 ") + level + " 不被当前模型支持";
		}
		agent_->set_thinking_level(level);
		if (archive_) {
			archive_->set_thinking(level);
			auto msgs = agent_->messages();
			if (!msgs.empty()) archive_->sync(msgs);
		}
	}
	// 同步为全局默认：新会话/重启后沿用用户最后选择的思考强度
	settings_->set_string("defaultThinkingLevel", level);
	emit_snapshot_now();
}

void AgentSessionHost::set_system_prompt_language(const std::string& language) {
	settings_->set_system_prompt_language(language);
	rebuild_system_prompt();
}

void AgentSessionHost::set_session_name(const std::string& name) {
	require_active();
	{
		std::lock_guard<std::mutex> lock(snapshot_mutex_);
		if (!active_ || !agent_) throw std::string("No active session. Create one first.");
		session_name_ = name;
		if (archive_) {
			archive_->set_title(name);
			auto msgs = agent_->messages();
			if (!msgs.empty()) archive_->sync(msgs);
		}
	}
	emit_snapshot_now();
}

void AgentSessionHost::reload_resources() {
	require_active();
	{
		std::lock_guard<std::mutex> lock(snapshot_mutex_);
		if (!active_ || !agent_) throw std::string("No active session. Create one first.");
		resources_->reload();
		rebuild_system_prompt();
	}
	emit_snapshot_now();
}

// ── listings ───────────────────────────────────────────────────────────────

JsonValue AgentSessionHost::get_snapshot() {
	std::lock_guard<std::mutex> lock(snapshot_mutex_);
	return build_snapshot();
}

void AgentSessionHost::apply_profile_after_start() {
	auto applied = configs_->apply_active();
	if (!applied || !is_active()) return;
	const StoredProfile& profile = *applied;
	std::string provider =
		profile.base_url.empty() ? profile.provider : ConfigStore::gateway_provider_id(profile.name);
	try {
		set_model(provider, profile.model);
	} catch (const std::string& e) {
		emit_log("warn", e);
	}
	if (!profile.system_prompt_language.empty()) {
		try {
			set_system_prompt_language(profile.system_prompt_language);
		} catch (const std::string& e) {
			emit_log("warn", e);
		}
	}
}

JsonValue AgentSessionHost::list_models() {
	JsonValue arr = JsonValue::array();
	for (const auto& m : runtime_->get_models()) {
		JsonValue j = JsonValue::object();
		j["provider"] = m.provider;
		j["id"] = m.id;
		j["name"] = m.name;
		j["reasoning"] = m.reasoning;
		j["contextWindow"] = (int64_t)m.context_window;
		j["maxTokens"] = (int64_t)m.max_tokens;
		JsonValue cost = JsonValue::object();
		cost["input"] = m.cost.input;
		cost["output"] = m.cost.output;
		cost["cacheRead"] = m.cost.cache_read;
		cost["cacheWrite"] = m.cost.cache_write;
		j["cost"] = cost;
		JsonValue levels = JsonValue::array();
		for (const auto& level : m.supported_thinking_levels()) levels.push_back(level);
		j["supportedThinkingLevels"] = levels;
		JsonValue auth = JsonValue::array();
		if (runtime_->resolve_api_key(m.provider)) auth.push_back("api_key");
		j["auth"] = auth;
		arr.push_back(j);
	}
	return arr;
}

JsonValue AgentSessionHost::list_skills() {
	JsonValue arr = JsonValue::array();
	for (const auto& skill : resources_->skills()) {
		JsonValue j = JsonValue::object();
		j["name"] = skill.name;
		j["description"] = skill.description;
		j["baseDir"] = skill.base_dir;
		j["filePath"] = skill.file_path;
		j["disableModelInvocation"] = skill.disable_model_invocation;
		arr.push_back(j);
	}
	return arr;
}

JsonValue AgentSessionHost::list_commands() {
	struct Cmd {
		std::string name, description, argument_hint;
	};
	auto L = [&](const std::string& zh, const std::string& en) { return isChinese() ? zh : en; };
	// Commands whose action is already a visible window button (settings/model/thinking/new)
	// are intentionally omitted from the list — the buttons cover them.
	std::vector<Cmd> cmds = {
		{"copy", L("复制最后一条助手消息到剪贴板", "Copy the last assistant message to the clipboard"), ""},
		{"session", L("显示会话信息和统计", "Show session info and stats"), ""},
		{"compact", L("手动压缩会话上下文", "Manually compact the conversation context"), ""},
	};
	for (const auto& t : resources_->templates()) {
		cmds.push_back({t.name, t.description.empty() ? "Prompt template" : t.description, t.argument_hint});
	}
	std::sort(cmds.begin(), cmds.end(), [](const Cmd& a, const Cmd& b) { return a.name < b.name; });
	JsonValue arr = JsonValue::array();
	for (const auto& c : cmds) {
		JsonValue j = JsonValue::object();
		j["name"] = c.name;
		j["description"] = c.description;
		j["source"] = "builtin";
		if (!c.argument_hint.empty()) j["argumentHint"] = c.argument_hint;
		arr.push_back(j);
	}
	return arr;
}

std::vector<SessionSummary> AgentSessionHost::list_sessions() {
	SessionArchive temp(agent_dir_, cwd_);
	return temp.index();
}

void AgentSessionHost::delete_session(const std::string& session_id) {
	if (session_id.empty()) throw std::string("会话 ID 不能为空");
	if (session_id_ == session_id) throw std::string("不能删除正在使用的会话");
	SessionArchive temp(agent_dir_, cwd_);
	auto path = temp.locate(session_id);
	if (!path) throw std::string("会话不存在: ") + session_id;
	std::string detail;
	if (recycle_path(*path, &detail) != 0)
		throw std::string("删除会话失败: ") + *path + (detail.empty() ? "" : " (" + detail + ")");
}

// ── projection ─────────────────────────────────────────────────────────────

JsonValue AgentSessionHost::serialize_tool_run(const ToolRun& run) const {
	JsonValue j = JsonValue::object();
	j["toolCallId"] = run.tool_call_id;
	j["name"] = run.name;
	j["args"] = run.args;
	j["status"] = run.status;
	j["output"] = run.output;
	j["isError"] = run.is_error;
	return j;
}

JsonValue AgentSessionHost::build_snapshot() {
	require_active();
	std::lock_guard<std::mutex> lock(projection_mutex_);

	// atomic view: separate messages()/streaming_message() calls could straddle
	// a finalize (streaming id dropped → client DOM node churn)
	Agent::Projection proj = agent_->projection();
	const std::vector<AgentMessage>& messages = proj.messages;
	bool streaming = proj.streaming_active;
	const std::optional<AgentMessage>& streaming_message = proj.streaming;

	std::vector<AgentMessage> snap_messages = messages;

	// Trim stale IDs left over from prior compaction: after compaction the
	// live message list shrinks but message_ids_ keeps growing, leaving
	// dangling entries that waste memory and could index out of bounds if we
	// ever accessed them by message index.
	if (message_ids_.size() > snap_messages.size()) {
			const size_t drop = message_ids_.size() - snap_messages.size();
			message_ids_.erase(message_ids_.begin(), message_ids_.begin() + (std::ptrdiff_t)drop);
		}
	while (message_ids_.size() < snap_messages.size()) {
		// the just-finalized assistant message inherits the streaming id so the
		// client keeps the same DOM node across the streaming→complete transition
		// (assistant messages always append before their tool results, and a full
		// snapshot always precedes patches, so the first un-assigned message is
		// the streamed one whenever streaming_id_ is set)
		if (!streaming_id_.empty()) {
			message_ids_.push_back(streaming_id_);
			streaming_id_.clear();
		} else {
			message_ids_.push_back("m" + std::to_string(++id_counter_));
		}
	}

	std::set<std::string> referenced_tool_ids;
	JsonValue arr = JsonValue::array();
	for (size_t i = 0; i < snap_messages.size(); i++) {
		const AgentMessage& m = snap_messages[i];
		if (m.role == "toolResult") continue;
		JsonValue jm = message_to_json(m);
		jm["id"] = message_ids_[i];
		// surface error/aborted stop reasons as message status: the UI only
		// renders the error tag and error text for message.status — without this
		// a failed turn (HTTP 4xx/5xx, rate limit) ends with an empty assistant
		// message and the agent just "goes idle" with no visible reason
		std::string status;
		if (m.role == "assistant") {
			if (m.stop_reason == "error") status = "error";
			else if (m.stop_reason == "aborted") status = "aborted";
		}
		// completed message inherits the streaming id (client DOM node stability)
		if (status.empty() && streaming_id_ == message_ids_[i] && !streaming) {
			status = "complete";
		}
		if (!status.empty()) jm["status"] = status;
		arr.push_back(jm);
		for (const auto& part : m.content) {
			if (part.type == ContentPart::Type::ToolCall) referenced_tool_ids.insert(part.tool_call_id);
		}
	}

	if (streaming && streaming_message && streaming_message->role == "assistant") {
		if (streaming_id_.empty()) streaming_id_ = "m" + std::to_string(++id_counter_);
		JsonValue jm = message_to_json(*streaming_message);
		jm["id"] = streaming_id_;
		jm["status"] = "streaming";
		arr.push_back(jm);
		for (const auto& part : streaming_message->content) {
			if (part.type == ContentPart::Type::ToolCall) referenced_tool_ids.insert(part.tool_call_id);
		}
	} else {
		streaming_id_.clear();
	}

	// Prune tool_runs_ entries whose tool calls are no longer referenced in
	// the current message list. This prevents unbounded growth during long
	// sessions where completed tool runs accumulate forever.
	for (auto it = tool_runs_.begin(); it != tool_runs_.end();) {
		if (!referenced_tool_ids.count(it->first)) it = tool_runs_.erase(it);
		else ++it;
	}
	JsonValue runs = JsonValue::array();
	for (const auto& [id, run] : tool_runs_) {
		runs.push_back(serialize_tool_run(run));
	}

	JsonValue snap = JsonValue::object();
	snap["id"] = session_id_;
	if (!session_name_.empty()) snap["name"] = session_name_;
	snap["cwd"] = cwd_;
	if (auto model = agent_->model()) {
		JsonValue ref = JsonValue::object();
		ref["provider"] = model->provider;
		ref["id"] = model->id;
		snap["model"] = ref;
		snap["modelName"] = model->name;
	} else {
		snap["model"] = JsonValue(nullptr);
	}
	snap["thinkingLevel"] = agent_->thinking_level();
	snap["activeConfig"] = configs_.get() ? configs_->list().active : "";
	snap["phase"] = phase_;
	snap["isStreaming"] = streaming;
	snap["generation"] = (int64_t)generation_;
	snap["messages"] = arr;
	snap["toolRuns"] = runs;
	if (!session_file_.empty()) snap["sessionFile"] = session_file_;
	// a full snapshot carries every run, so patches only need to report runs
	// mutated AFTER this point
	patch_sent_version_ = run_version_counter_;
	return snap;
}

bool AgentSessionHost::build_stream_patch(JsonValue& patch) {
	require_active();
	std::lock_guard<std::mutex> lock(projection_mutex_);

	// runs changed since the last full snapshot / patch
	JsonValue runs = JsonValue::array();
	for (const auto& [id, run] : tool_runs_) {
		auto it = run_versions_.find(id);
		if (it != run_versions_.end() && it->second > patch_sent_version_) {
			runs.push_back(serialize_tool_run(run));
		}
	}

	Agent::Projection proj = agent_->projection();
	bool streaming = proj.streaming_active;
	auto streaming_message = std::move(proj.streaming);
	bool has_streaming = streaming && streaming_message && streaming_message->role == "assistant";
	if (!has_streaming && runs.size() == 0) return false;  // nothing changed

	if (has_streaming) {
		if (streaming_id_.empty()) streaming_id_ = "m" + std::to_string(++id_counter_);
		JsonValue jm = message_to_json(*streaming_message);
		jm["id"] = streaming_id_;
		jm["status"] = "streaming";
		patch["streaming"] = std::move(jm);
	} else {
		patch["streaming"] = JsonValue(nullptr);
	}
	patch["sessionId"] = session_id_;
	if (runs.size() > 0) patch["toolRuns"] = std::move(runs);
	patch_sent_version_ = run_version_counter_;
	return true;
}

void AgentSessionHost::on_agent_event(const AgentEvent& ev) {
	switch (ev.type) {
		case AgentEvent::Type::AgentStart:
		case AgentEvent::Type::TurnStart:
		case AgentEvent::Type::MessageStart:
			{
				std::lock_guard<std::mutex> lock(projection_mutex_);
				phase_ = "streaming";
			}
			break;
		case AgentEvent::Type::ToolExecutionStart:
			{
				std::lock_guard<std::mutex> lock(projection_mutex_);
				if (!tool_runs_.count(ev.tool_call_id)) {
					ToolRun run;
					run.tool_call_id = ev.tool_call_id;
					run.name = ev.tool_name;
					run.args = ev.args;
					run.status = "running";
					tool_runs_[ev.tool_call_id] = std::move(run);
					run_versions_[ev.tool_call_id] = ++run_version_counter_;
				}
				phase_ = "streaming";
			}
			break;
		case AgentEvent::Type::ToolExecutionUpdate: {
			std::lock_guard<std::mutex> lock(projection_mutex_);
			auto it = tool_runs_.find(ev.tool_call_id);
			if (it != tool_runs_.end()) {
				it->second.output = ev.partial_output;
				run_versions_[ev.tool_call_id] = ++run_version_counter_;
			}
			break;
		}
		case AgentEvent::Type::ToolExecutionEnd: {
			std::lock_guard<std::mutex> lock(projection_mutex_);
			auto it = tool_runs_.find(ev.tool_call_id);
			if (it != tool_runs_.end()) {
				bool is_error = ev.tool_result && ev.tool_result->is_error;
				it->second.status = is_error ? "error" : "complete";
				it->second.is_error = is_error;
				std::string text;
				if (ev.tool_result) {
					for (const auto& part : ev.tool_result->content) {
						if (part.type == ContentPart::Type::Text) text += part.text;
					}
				}
				it->second.output = text;
				run_versions_[ev.tool_call_id] = ++run_version_counter_;
			}
			break;
		}
		case AgentEvent::Type::Error:
			// exception-path errors (run_chain catch / no model) never become an
			// assistant message — without a log they vanish and the turn just ends
			emit_log("error", ev.message);
			break;
		case AgentEvent::Type::TurnEnd:
		case AgentEvent::Type::AgentEnd:
			{
				std::lock_guard<std::mutex> lock(projection_mutex_);
				phase_ = "idle";
			}
			break;
	default:
			break;
	}
	// MessageUpdate / ToolExecutionUpdate are per-delta events: they only mutate
	// the streaming view / tool-run output, never messages_. Persisting here
	// would deep-copy the whole history on EVERY delta (agent_->messages()),
	// pegging the worker thread on long sessions. Messages only grow on the
	// other events (assistant finalized / tool results appended), which cover
	// every growth point.
	bool delta_tick = ev.type == AgentEvent::Type::MessageUpdate ||
		ev.type == AgentEvent::Type::ToolExecutionUpdate;
	if (!delta_tick) {
		sync_archive();
	}
	schedule_snapshot(!delta_tick);  // structural events → full snapshot; deltas → patch
}

void AgentSessionHost::snapshot_timer_loop() {
	std::unique_lock<std::mutex> lock(snapshot_mutex_);
	int64_t debounce_ms = 30;  // owned by this thread only
	while (true) {
		snapshot_cv_.wait(lock, [&] { return snapshot_quit_ || snapshot_scheduled_ || patch_scheduled_; });
		if (snapshot_quit_) break;
		lock.unlock();
		std::this_thread::sleep_for(std::chrono::milliseconds(debounce_ms));  // debounce window
		lock.lock();
		// a pending full snapshot subsumes any pending patch (it carries the
		// streaming message and all runs); clear both flags together
		bool fire_full = snapshot_scheduled_;
		bool fire_patch = patch_scheduled_;
		snapshot_scheduled_ = false;
		patch_scheduled_ = false;
		lock.unlock();
		if (fire_full || fire_patch) {
			try {
				auto t0 = std::chrono::steady_clock::now();
				if (fire_full) {
					emit_snapshot_now();
				} else {
					emit_patch_now();
				}
				auto cost_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
					std::chrono::steady_clock::now() - t0).count();
				// adaptive backoff: the payload now scales with the DELTA (patch) or
				// the history (full snapshot). If a cycle costs more than ~1/3 of the
				// debounce, the consumer (renderer) can't keep up — slow the cadence
				// down instead of saturating it, and recover once cheap again.
				debounce_ms = cost_ms > 10 ? std::min<int64_t>(cost_ms * 3, 250) : 30;
			} catch (...) {
				// must never kill the timer thread
			}
		}
		lock.lock();
	}
}

void AgentSessionHost::schedule_snapshot(bool structural) {
	{
		std::lock_guard<std::mutex> lock(snapshot_mutex_);
		if (structural) {
			snapshot_scheduled_ = true;
		} else {
			patch_scheduled_ = true;
		}
	}
	snapshot_cv_.notify_one();
}

void AgentSessionHost::emit_patch_now() {
	if (!active_) return;
	JsonValue patch;
	auto t_build0 = phi::trace::now_ms();
	try {
		// snapshot_mutex_ is held for the WHOLE build: teardown() resets
		// agent_ under the same mutex, and build_stream_patch dereferences
		// agent_ — releasing the lock before the build was a use-after-free
		// whenever a session switch landed in the gap.
		std::lock_guard<std::mutex> lock(snapshot_mutex_);
		generation_++;
		patch["generation"] = (int64_t)generation_.load();
		if (!build_stream_patch(patch)) return;
	} catch (const std::string&) {
		return;
	}
	auto t_build1 = phi::trace::now_ms();
	HostEvent ev;
	ev.is_patch = true;
	ev.patch = std::move(patch);
	auto t_send0 = phi::trace::now_ms();
	dispatch(ev);
	PHI_TRACE("PATCH build_ms=%lld send_ms=%lld",
		(long long)(t_build1 - t_build0),
		(long long)(phi::trace::now_ms() - t_send0));
}

void AgentSessionHost::emit_snapshot_now() {
	if (!active_) return;
	HostEvent ev;
	ev.is_snapshot = true;
	{
		// serialize snapshot building (also against teardown() resetting agent_)
		std::lock_guard<std::mutex> lock(snapshot_mutex_);
		generation_++;
		try {
			ev.snapshot = build_snapshot();
		} catch (const std::string&) {
			return;
		}
	}
	dispatch(ev);
}

void AgentSessionHost::emit_log(const std::string& level, const std::string& message) {
	HostEvent ev;
	ev.log = {level, message};
	dispatch(ev);
}

void AgentSessionHost::dispatch(const HostEvent& ev) {
	if (on_event_) on_event_(ev);
}

// ── context transform (compaction) ─────────────────────────────────────────

std::vector<AgentMessage> AgentSessionHost::transform_context(const std::vector<AgentMessage>& messages) const {
	std::vector<AgentMessage> out;
	size_t start = 0;
	bool has_summary = false;
	for (size_t i = 0; i < messages.size(); i++) {
		if (messages[i].role == "compactionSummary") {
			start = i;
			has_summary = true;
		}
	}
	if (!has_summary) return messages;
	for (size_t i = start; i < messages.size(); i++) {
		const AgentMessage& m = messages[i];
		if (m.role == "compactionSummary") {
			out.push_back(AgentMessage::user_text(
				"[Previous conversation summary]\n" + m.summary + "\n[End of summary. Continue the task.]"));
			continue;
		}
		out.push_back(m);
	}
	return out;
}

}  // namespace phi
