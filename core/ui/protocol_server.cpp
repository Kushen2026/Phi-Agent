// ProtocolServer implementation: static UI + upload/pick endpoints + WS protocol.
#include "ui/protocol_server.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <cctype>
#include <thread>
#include <vector>

#include "agent/agent_session.hpp"
#include "util/base.hpp"
#include "net/http_client.hpp"
#include "net/http_server.hpp"
#include "ui/ui_theme.hpp"
#include "models/media_models.hpp"

namespace phi {

struct ProtocolServer::Impl {
	std::string base_dir;
	std::string ui_dir;
	HttpServer http;
	std::unique_ptr<AgentSessionHost> host;
	std::atomic<bool> auto_session{false};
	std::function<std::optional<std::string>(const std::string&)> folder_picker;
	// Media model downloads run on their own thread and stream progress over the
	// socket; this latches the one-in-flight rule so two downloads cannot race the
	// same file.
	std::atomic<bool> media_downloading{false};

	// The models directory (models/ next to the app, else the same upward search
	// the media tools use).
	std::string models_dir() const {
		std::string p = path_join(base_dir, "models");
		if (path_is_dir(p)) return p;
		return path_join(path_dirname(base_dir), "models");
	}

	void broadcast(const std::string& text) { http.broadcast(text); }
	void broadcast_latest(const std::string& text) { http.broadcast_latest(text); }

	void send_response(WebSocketConnection& ws, const std::string& id, bool ok, JsonValue result,
		const std::string& error) {
		JsonValue msg = JsonValue::object();
		msg["type"] = "response";
		msg["id"] = id;
		if (ok) {
			msg["ok"] = true;
			msg["result"] = std::move(result);
		} else {
			msg["ok"] = false;
			JsonValue err = JsonValue::object();
			err["message"] = error;
			msg["error"] = err;
		}
		ws.send_text(msg.dump());
	}

	void send_session(WebSocketConnection& ws, const std::string& id) {
		try {
			JsonValue result = JsonValue::object();
			result["kind"] = "session";
			result["session"] = host->get_snapshot();
			send_response(ws, id, true, result, "");
		} catch (const std::string& e) {
			send_response(ws, id, false, JsonValue(), e);
		}
	}

	void broadcast_snapshot() {
		try {
			JsonValue msg = JsonValue::object();
			msg["type"] = "session_updated";
			msg["sessionId"] = host->session_id();
			msg["snapshot"] = host->get_snapshot();
			broadcast(msg.dump());
		} catch (...) {
		}
	}

	void broadcast_skills() {
		try {
			JsonValue msg = JsonValue::object();
			msg["type"] = "skills";
			msg["skills"] = host->list_skills();
			broadcast(msg.dump());
		} catch (...) {
		}
	}

		void broadcast_models() {
		try {
			JsonValue msg = JsonValue::object();
			msg["type"] = "models";
			msg["models"] = host->list_models();
			broadcast(msg.dump());
		} catch (...) {
		}
	}
};

namespace {

std::string mime_for(const std::string& path) {
	if (ends_with(path, ".html")) return "text/html; charset=utf-8";
	if (ends_with(path, ".js")) return "text/javascript; charset=utf-8";
	if (ends_with(path, ".css")) return "text/css; charset=utf-8";
	if (ends_with(path, ".json")) return "application/json; charset=utf-8";
	if (ends_with(path, ".svg")) return "image/svg+xml";
	if (ends_with(path, ".png")) return "image/png";
	if (ends_with(path, ".ico")) return "image/x-icon";
	return "application/octet-stream";
}

// FNV-1a 64 over the served bytes: a cheap content validator (no stat/read of
	// the source file needed, and it stays correct after the theme is injected into
	// index.html — a size/mtime tag would go stale on a theme change)
static std::string content_etag(const std::string& body) {
	uint64_t h = 1469598103934665603ull;
	for (unsigned char c : body) {
		h ^= c;
		h *= 1099511628211ull;
	}
	char buf[32];
	snprintf(buf, sizeof buf, "\"%016llx\"", (unsigned long long)h);
	return std::string(buf);
}

	HttpResponse static_file(const std::string& ui_dir, const std::string& url_path) {
	std::string rel = url_path == "/" ? "index.html" : url_path.substr(1);
	std::string path = path_join(ui_dir, rel);
	if (contains(rel, "..") || !path_exists(path) || path_is_dir(path)) {
		HttpResponse res;
		res.status = 404;
		res.body = "Not found";
		return res;
	}
	HttpResponse res;
	res.content_type = mime_for(path);
	res.body = read_file(path).value_or("");
	// serve with a validator (not no-store): an unchanged asset answers 304 and
	// the renderer reuses its cached — and already parsed — copy, while an
	// edited file simply gets a fresh tag
	res.cache_no_store = false;
	res.etag = content_etag(res.body);
	return res;
}

JsonValue profile_to_client(const StoredProfile& p) {
	JsonValue j = JsonValue::object();
	j["name"] = p.name;
	j["provider"] = p.provider;
	j["model"] = p.model;
	j["apiKeyMasked"] = ConfigStore::mask_key(p.api_key);
	j["apiKey"] = p.api_key;
	if (!p.base_url.empty()) j["baseUrl"] = p.base_url;
	if (!p.api_format.empty()) j["apiFormat"] = p.api_format;
	if (p.subagent) {
		JsonValue sub = JsonValue::object();
		sub["apiKeyMasked"] = ConfigStore::mask_key(p.subagent->api_key);
		sub["apiKey"] = p.subagent->api_key;
		sub["baseUrl"] = p.subagent->base_url;
		sub["model"] = p.subagent->model;
		sub["apiFormat"] = p.subagent->api_format;
		j["subagent"] = sub;
	}
	if (!p.system_prompt_language.empty()) j["systemPromptLanguage"] = p.system_prompt_language;
	return j;
}

JsonValue configs_result(AgentSessionHost& host) {
	ConfigStore::Listing listing = host.configs().list();
	JsonValue result = JsonValue::object();
	result["kind"] = "configs";
	result["active"] = listing.active.empty() ? JsonValue(nullptr) : JsonValue(listing.active);
	JsonValue profiles = JsonValue::array();
	for (const auto& p : listing.profiles) profiles.push_back(profile_to_client(p));
	result["profiles"] = profiles;
	if (listing.subagent) {
		JsonValue sub = JsonValue::object();
		sub["apiKeyMasked"] = ConfigStore::mask_key(listing.subagent->api_key);
		sub["baseUrl"] = listing.subagent->base_url;
		sub["model"] = listing.subagent->model;
		sub["apiFormat"] = listing.subagent->api_format;
		result["subagent"] = sub;
	}
	result["systemPromptLanguage"] = listing.system_prompt_language;
	return result;
}

// ── remote model listing (设置 → API 配置 → 获取模型列表) ────────────────────

// extract model ids from the many list-shapes providers use:
// {data:[{id}|"s"]} | {models:[{id|name}|"s"]} | [{id|name}|"s"]
void collect_model_ids(const JsonValue& j, std::vector<std::string>& out) {
	auto push_id = [&](const JsonValue& v) {
		if (v.is_string()) {
			std::string s = trim(v.as_string());
			if (!s.empty()) out.push_back(s);
		} else if (v.is_object()) {
			std::string s = trim(v["id"].as_string(v["name"].as_string("")));
			if (!s.empty()) out.push_back(s);
		}
	};
	auto scan = [&](const JsonValue& v) {
		if (!v.is_array()) return false;
		for (const auto& item : v.items()) push_id(item);
		return true;
	};
	if (scan(j)) return;
	if (j.is_object()) {
		if (scan(j["data"])) return;
		if (scan(j["models"])) return;
	}
}

// GET {base}/models in both wire formats (header styles differ); returns the
// union of ids parsed from any successful response

// best-effort: pull a human-readable reason out of an error response body
// ("{"error":{"message":"invalid api key"}}" / "{"error":"..."}" / raw snippet)
std::string error_body_reason(const std::string& body) {
	if (body.empty()) return "";
	std::string reason;
	if (auto parsed = json_parse(body)) {
		if (parsed->is_object()) {
			const JsonValue& err = (*parsed)["error"];
			if (err.is_object()) {
				reason = trim(err["message"].as_string(""));
			} else if (err.is_string()) {
				reason = trim(err.as_string());
			}
		}
	}
	if (reason.empty()) {
		reason = trim(body);
		if (reason.size() > 120) reason = reason.substr(0, 120) + "...";
	}
	// keep it one-line
	for (auto& c : reason) {
		if (c == '\n' || c == '\r' || c == '\t') c = ' ';
	}
	return reason;
}

std::vector<std::string> fetch_remote_models(const std::string& base_url, const std::string& api_key,
	const std::string& api_format, std::string& error) {
	std::string base = trim(base_url);
	while (!base.empty() && base.back() == '/') base.pop_back();
	if (base.empty()) {
		error = "请先填写请求地址";
		return {};
	}
	std::vector<std::string> urls;
	bool looks_v1 = ends_with(to_lower(base), "/v1");
	if (api_format == "openai-completions") {
		urls.push_back(base + "/models");
		if (!looks_v1) urls.push_back(base + "/v1/models");
	} else {
		urls.push_back(base + (looks_v1 ? "/models" : "/v1/models"));
		if (!looks_v1) urls.push_back(base + "/models");
	}

	std::vector<std::string> ids;
	bool got_response = false;
	std::string last_error;
	for (const auto& url : urls) {
		for (int auth = 0; auth < 2; auth++) {
			std::vector<std::pair<std::string, std::string>> headers;
			if (!api_key.empty()) {
				if (auth == 0) {
					headers.push_back({"Authorization", "Bearer " + api_key});
					headers.push_back({"x-api-key", api_key});
				} else {
					headers.push_back({"x-api-key", api_key});
					headers.push_back({"anthropic-version", "2023-06-01"});
				}
			}
			HttpStreamResult res = http_request("GET", url, headers, "", 20000);
			if (res.status == 200 && !res.body.empty()) {
				got_response = true;
				if (auto parsed = json_parse(res.body)) {
					collect_model_ids(*parsed, ids);
				}
			}
			if (res.status != 0) {
				got_response = true;
				std::string reason = error_body_reason(res.body);
				last_error = "HTTP " + std::to_string(res.status) + " @ " + url +
					(reason.empty() ? "" : " (" + reason + ")");
			}
		}
		if (!ids.empty()) break;  // one URL answered with a usable list
	}
	if (!got_response && ids.empty()) {
		error = "无法连接到 " + base + (last_error.empty() ? "" : " (" + last_error + ")");
	} else if (ids.empty()) {
		error = "服务器未返回模型列表" + (last_error.empty() ? "" : " (" + last_error + ")");
	}
	// dedupe + sort
	std::sort(ids.begin(), ids.end());
	ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
	return ids;
}

}  // namespace

// ── request dispatch ───────────────────────────────────────────────────────

static void handle_ws_message(ProtocolServer::Impl& impl, WebSocketConnection& ws, const std::string& text) {
	auto parsed = json_parse(text);
	if (!parsed) {
		impl.send_response(ws, "", false, JsonValue(), "Invalid JSON");
		return;
	}
	const JsonValue& req = *parsed;
	std::string type = req["type"].as_string("");
	std::string id = req["id"].as_string("");
	AgentSessionHost& host = *impl.host;

	try {
		if (type == "create_session") {
			JsonValue model_ref;
			bool has_model = false;
			if (const JsonValue* m = req.find("model")) {
				if (!m->is_null()) {
					model_ref = *m;
					has_model = true;
				}
			}
			host.create_session(req["name"].as_string(""), has_model ? &model_ref : nullptr,
				req["thinkingLevel"].as_string(""), false);
			impl.send_session(ws, id);
		} else if (type == "open_session") {
			host.open_session(req["sessionId"].as_string(""));
			impl.send_session(ws, id);
		} else if (type == "list_sessions") {
			JsonValue result = JsonValue::object();
			result["kind"] = "sessions";
			JsonValue arr = JsonValue::array();
			for (const auto& s : host.list_sessions()) {
				JsonValue j = JsonValue::object();
				j["id"] = s.id;
				j["name"] = s.title;
				j["cwd"] = s.cwd;
				j["updatedAt"] = (int64_t)s.updated_at;
				j["messageCount"] = s.message_count;
				arr.push_back(j);
			}
			result["sessions"] = arr;
			impl.send_response(ws, id, true, result, "");
		} else if (type == "delete_session") {
			host.delete_session(req["sessionId"].as_string(""));
			JsonValue result = JsonValue::object();
			result["kind"] = "ok";
			impl.send_response(ws, id, true, result, "");
		} else if (type == "list_models") {
			JsonValue result = JsonValue::object();
			result["kind"] = "models";
			result["models"] = host.list_models();
			impl.send_response(ws, id, true, result, "");
		} else if (type == "list_skills") {
			JsonValue result = JsonValue::object();
			result["kind"] = "skills";
			result["skills"] = host.list_skills();
			impl.send_response(ws, id, true, result, "");
		} else if (type == "delete_skill") {
			// the panel sends the file_path it was given by list_skills; only a
			// path that is currently a known skill may be recycled, so this can
			// never be talked into deleting an arbitrary directory
			std::string file_path = req["filePath"].as_string("");
			bool known = false;
			for (const auto& skill : host.resources().skills()) {
				if (skill.file_path == file_path) known = true;
			}
			if (!known) throw std::string("Skill not found: ") + file_path;
			std::string target = path_basename(file_path) == "SKILL.md" ? path_dirname(file_path) : file_path;
			{
				std::string detail;
				if (recycle_path(target, &detail) != 0)
					throw std::string("Failed to delete: ") + target + (detail.empty() ? "" : " (" + detail + ")");
			}
			host.reload_resources();
			impl.broadcast_skills();
			JsonValue result = JsonValue::object();
			result["kind"] = "ok";
			impl.send_response(ws, id, true, result, "");
		} else if (type == "list_commands") {
			JsonValue result = JsonValue::object();
			result["kind"] = "commands";
			result["commands"] = host.list_commands();
			impl.send_response(ws, id, true, result, "");
		} else if (type == "prompt") {
			JsonValue no_attachments = JsonValue::array();
			const JsonValue* atts = req.find("attachments");
			host.prompt(req["text"].as_string(""), req["mode"].as_string(""),
				atts ? *atts : no_attachments);
			impl.send_session(ws, id);
		} else if (type == "abort") {
			host.abort();
			impl.send_session(ws, id);
		} else if (type == "compact") {
			host.compact();
			impl.send_session(ws, id);
		} else if (type == "set_model") {
			host.set_model(req["model"]["provider"].as_string(""), req["model"]["id"].as_string(""));
			impl.send_session(ws, id);
		} else if (type == "set_thinking") {
			host.set_thinking(req["thinkingLevel"].as_string(""));
			impl.send_session(ws, id);
		} else if (type == "set_system_prompt_language") {
			std::string language = req["language"].as_string("zh");
			host.settings().set_system_prompt_language(language);
			host.set_system_prompt_language(language);
			impl.send_response(ws, id, true, configs_result(host), "");
		} else if (type == "set_session_name") {
			host.set_session_name(req["name"].as_string(""));
			impl.send_session(ws, id);
		} else if (type == "get_snapshot") {
			impl.send_session(ws, id);
		} else if (type == "set_cwd") {
			host.change_cwd(req["cwd"].as_string(""));
			JsonValue result = JsonValue::object();
			result["kind"] = "settings";
			result["settings"] = host.settings_json();
			impl.send_response(ws, id, true, result, "");
			impl.broadcast_skills();
			impl.broadcast_models();
		} else if (type == "reload_resources") {
			host.reload_resources();
			impl.broadcast_skills();
			JsonValue result = JsonValue::object();
			result["kind"] = "ok";
			impl.send_response(ws, id, true, result, "");
		} else if (type == "list_configs") {
			impl.send_response(ws, id, true, configs_result(host), "");
		} else if (type == "save_config") {
			const JsonValue& draft = req["profile"];
			ProfileDraft p;
			p.name = draft["name"].as_string("");
			p.provider = draft["provider"].as_string("");
			p.model = draft["model"].as_string("");
			p.api_key = draft["apiKey"].as_string("");
			p.base_url = draft["baseUrl"].as_string("");
			p.api_format = draft["apiFormat"].as_string("");
			p.system_prompt_language = draft["systemPromptLanguage"].as_string("");
			if (const JsonValue* sub = draft.find("subagent")) {
				if (!sub->is_null()) {
					SubagentConfig sc;
					sc.api_key = (*sub)["apiKey"].as_string("");
					sc.base_url = (*sub)["baseUrl"].as_string("");
					sc.model = (*sub)["model"].as_string("");
					sc.api_format = (*sub)["apiFormat"].as_string("anthropic-messages");
					p.subagent = sc;
				}
			}
			host.configs().save(p);
			// language takes effect on the running session for the active profile
			if (!p.system_prompt_language.empty() && host.configs().list().active == p.name) {
				host.set_system_prompt_language(p.system_prompt_language);
			}
			impl.send_response(ws, id, true, configs_result(host), "");
		} else if (type == "delete_config") {
			host.configs().remove(req["name"].as_string(""));
			impl.send_response(ws, id, true, configs_result(host), "");
		} else if (type == "activate_config") {
			std::string name = req["name"].as_string("");
			host.configs().activate(name);
			// system prompt language follows the activated profile immediately
			host.set_system_prompt_language(host.settings().system_prompt_language());
			// gateway profiles register at session start: resume the most recent session
			bool requires_restart = false;
			auto listing = host.configs().list();
			for (const auto& p : listing.profiles) {
				if (p.name == name && (!p.base_url.empty() || p.subagent)) requires_restart = true;
			}
			if (requires_restart) {
				// 保留当前会话（如果有），而不是跳到最近的旧会话
				host.open_session(host.is_active() ? host.session_id() : "");
			}
			impl.broadcast_models();


			JsonValue result = JsonValue::object();
			result["kind"] = "configs";
			result["active"] = listing.active.empty() ? JsonValue(nullptr) : JsonValue(listing.active);
			JsonValue profiles = JsonValue::array();
			for (const auto& p : listing.profiles) profiles.push_back(profile_to_client(p));
			result["profiles"] = profiles;
			impl.send_response(ws, id, true, result, "");
		} else if (type == "save_subagent_config") {
			const JsonValue& draft = req["config"];
			SubagentConfig sc;
			sc.api_key = draft["apiKey"].as_string("");
			sc.base_url = draft["baseUrl"].as_string("");
			sc.model = draft["model"].as_string("");
			sc.api_format = draft["apiFormat"].as_string("anthropic-messages");
			host.configs().save_subagent(sc);
			impl.send_response(ws, id, true, configs_result(host), "");
		} else if (type == "get_tool_settings") {
			JsonValue result = JsonValue::object();
			result["kind"] = "tool_settings";
			result["settings"] = host.settings().tool_settings();
			impl.send_response(ws, id, true, result, "");
		} else if (type == "save_tool_settings") {
			const JsonValue& s = req["settings"];
			if (!s.is_object()) throw std::string("settings 必须是对象");
			host.settings().set_tool_settings(s);
			JsonValue result = JsonValue::object();
			result["kind"] = "tool_settings";
			result["settings"] = host.settings().tool_settings();
			impl.send_response(ws, id, true, result, "");
		} else if (type == "list_media_models") {
			// 设置 → 工具设置 → 媒体模型：models/ 下的每一个文件都可以按角色选择。
			// 这里就是「实时」的数据源——面板每次打开都会再问一次，所以手动放进
			// models/ 的文件立刻可选；没有内置清单，也没有「默认」项。
			JsonValue result = JsonValue::object();
			result["kind"] = "media_models";
			const std::string mdir = impl.models_dir();
			JsonValue files = JsonValue::array();
			for (const std::string& rel : media::list_media_files(mdir)) {
				JsonValue j = JsonValue::object();
				j["name"] = rel;
				j["path"] = path_join(mdir, rel);
				files.push_back(j);
			}
			result["files"] = files;
			result["modelsDir"] = mdir;
			impl.send_response(ws, id, true, result, "");
		} else if (type == "save_media_settings") {
			const JsonValue& media = req["media"];
			if (!media.is_object()) throw std::string("media 必须是对象");
			JsonValue tools = host.settings().tool_settings();
			tools["media"] = media;
			host.settings().set_tool_settings(tools);
			JsonValue result = JsonValue::object();
			result["kind"] = "tool_settings";
			result["settings"] = host.settings().tool_settings();
			impl.send_response(ws, id, true, result, "");
		} else if (type == "download_media_model") {
			const std::string mid = req["id"].as_string("");
			const media::MediaModelEntry* entry = media::media_model_entry(mid);
			if (!entry) throw std::string("未知模型: ") + mid;
			std::atomic<bool>* flag = &impl.media_downloading;
			if (flag->exchange(true)) throw std::string("已有下载在进行");
			const media::MediaModelEntry e = *entry;
			const std::string mdir = impl.models_dir();
			HttpServer* http = &impl.http;
			auto send = [http](const JsonValue& j) { http->broadcast(j.dump()); };
			std::thread([e, mdir, send, flag]() mutable {
				auto emit = [&](const char* status, double p, const std::string& err) {
					JsonValue j = JsonValue::object();
					j["type"] = "media_download";
					j["id"] = e.id;
					j["status"] = status;
					j["progress"] = p;
					if (!err.empty()) j["error"] = err;
					send(j);
				};
				emit("start", 0.0, "");
				try {
					media::media_model_download(
					    e, mdir, [&](double p) { emit("progress", p, ""); }, [] { return false; });
					emit("done", 1.0, "");
				} catch (const std::exception& ex) {
					emit("error", 0.0, ex.what());
				}
				flag->store(false);
			}).detach();
			JsonValue result = JsonValue::object();
			result["kind"] = "ok";
			impl.send_response(ws, id, true, result, "");
		} else if (type == "get_ui_settings") {
			{
				JsonValue result = JsonValue::object();
				result["kind"] = "ui_settings";
				result["settings"] = JsonValue::object();
				result["settings"]["hotkeys"] = host.settings().hotkeys();
				result["settings"]["appearance"] = host.settings().appearance();
				result["settings"]["prefs"] = host.settings().prefs();
				impl.send_response(ws, id, true, result, "");
			}
		} else if (type == "save_ui_settings") {
			const JsonValue& s = req["settings"];
			if (!s.is_object()) throw std::string("settings 必须是对象");
			// save hotkeys / appearance / prefs to settings.json
			for (const auto& [k, v] : s.entries()) {
				if (k == "hotkeys")       host.settings().set_hotkeys(v);
				else if (k == "appearance") host.settings().set_appearance(v);
				else if (k == "prefs")    host.settings().set_prefs(v);
			}
			JsonValue result = JsonValue::object();
			result["kind"] = "ui_settings";
			result["settings"] = s;
			impl.send_response(ws, id, true, result, "");
		} else if (type == "restart_session") {
			// reopen the current session so new tool settings take effect;
			// history is preserved (same path as activate_config)
			host.open_session(host.is_active() ? host.session_id() : "");
			impl.send_session(ws, id);
			impl.broadcast_models();
		} else {
			throw std::string("Unhandled request type: ") + type;
		}
	} catch (const std::string& e) {
		impl.send_response(ws, id, false, JsonValue(), e);
	} catch (const std::exception& e) {
		impl.send_response(ws, id, false, JsonValue(), e.what());
	}
}

// ── lifecycle ──────────────────────────────────────────────────────────────

ProtocolServer::ProtocolServer() : impl_(std::make_unique<Impl>()) {}

ProtocolServer::~ProtocolServer() {
	shutdown();
}

void ProtocolServer::set_stream_fn(ProviderStreamFn fn) {
	if (impl_->host) impl_->host->set_stream_fn(std::move(fn));
}

void ProtocolServer::set_folder_picker(std::function<std::optional<std::string>(const std::string&)> picker) {
	impl_->folder_picker = std::move(picker);
}

bool ProtocolServer::start(const std::string& base_dir) {
	impl_->base_dir = base_dir;
	impl_->ui_dir = path_join(base_dir, "ui");

	fprintf(stderr, "[start] creating host\n");
	impl_->host = std::make_unique<AgentSessionHost>(base_dir, path_absolute("."));
	fprintf(stderr, "[start] host ok\n");
	// resume saved cwd from the config panel when present
	if (auto saved = impl_->host->configs().saved_cwd()) {
		try {
			if (path_is_dir(*saved)) impl_->host->set_initial_cwd(*saved);
		} catch (...) {
		}
	}

	// host events -> broadcast
	impl_->host->set_event_callback([impl = impl_.get()](const HostEvent& ev) {
		if (ev.is_snapshot) {
			try {
				JsonValue msg = JsonValue::object();
				msg["type"] = "session_updated";
				msg["sessionId"] = ev.snapshot["id"].as_string("");
				msg["snapshot"] = ev.snapshot;
				impl->broadcast(msg.dump());
			} catch (...) {
			}
		} else if (ev.is_patch) {
			// incremental stream tick: streaming message + changed tool runs only.
			// latest-wins broadcast: a busy renderer only needs the newest patch
			// (each carries full streaming state) — prevents unbounded send queues
			// and the multi-second broadcast stalls behind it
			try {
				JsonValue msg = JsonValue::object();
				msg["type"] = "stream_patch";
				msg["sessionId"] = ev.patch["sessionId"].as_string("");
				msg["generation"] = ev.patch["generation"];
				msg["streaming"] = ev.patch["streaming"];
				if (const JsonValue* runs = ev.patch.find("toolRuns")) msg["toolRuns"] = *runs;
				impl->broadcast_latest(msg.dump());
			} catch (...) {
			}
		} else {
			JsonValue msg = JsonValue::object();
			msg["type"] = "log";
			msg["level"] = ev.log.level;
			msg["message"] = ev.log.message;
			impl->broadcast(msg.dump());
		}
	});

	impl_->http.set_http_handler([impl = impl_.get()](const HttpRequest& req) -> HttpResponse {
		// Helper to check Host header for loopback only
		auto check_host = [](const HttpRequest& req) -> std::pair<int, std::string> {
			auto it = req.headers.find("host");
			std::string host_hdr = it == req.headers.end() ? "" : it->second;
			std::string host_name;
			if (!host_hdr.empty() && host_hdr[0] == '[') {
				// IPv6 literal: compare up to the closing bracket
				size_t close = host_hdr.find(']');
				host_name = close == std::string::npos ? host_hdr : host_hdr.substr(0, close + 1);
			} else {
				size_t colon = host_hdr.find(':');
				host_name = colon == std::string::npos ? host_hdr : host_hdr.substr(0, colon);
			}
			if (host_name != "127.0.0.1" && host_name != "localhost" && host_name != "[::1]") {
				return {403, "forbidden"};
			}
			return {0, ""};
		};
		if (req.method == "POST" && req.path == "/api/upload") {
			if (auto [code, body] = check_host(req); code != 0) {
				HttpResponse r; r.status = code; r.body = body; return r;
			}
			HttpResponse res;
			auto body = json_parse(req.body);
			if (!body) {
				res.status = 400;
				res.body = "Invalid JSON body";
				return res;
			}
			std::string name = (*body)["name"].as_string("file");
			std::string mime = (*body)["mimeType"].as_string("application/octet-stream");
			auto data = base64_decode((*body)["data"].as_string(""));
			if (!data || data->empty()) {
				res.status = 400;
				res.body = "Empty file";
				return res;
			}
			std::string safe = name;
			std::string clean;
			for (char c : safe) {
				clean.push_back((std::isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_' || c == ' ')
					? c : '_');
				if (clean.size() >= 100) break;
			}
			if (clean.empty()) clean = "file";
			std::string uploads = path_join(impl->base_dir, "data/uploads");
			mkdirs(uploads);
			std::string path = path_join(uploads, uuid4() + "-" + clean);
			write_file(path, std::string_view((const char*)data->data(), data->size()));
			JsonValue out = JsonValue::object();
			out["path"] = path;
			out["name"] = clean;
			out["mimeType"] = mime;
			res.content_type = "application/json; charset=utf-8";
			res.body = out.dump();
			return res;
		}
		if (req.method == "GET" && req.path == "/api/pick-directory") {
			if (auto [code, body] = check_host(req); code != 0) {
				HttpResponse r; r.status = code; r.body = body; return r;
			}
			HttpResponse res;
			res.content_type = "application/json; charset=utf-8";
			JsonValue out = JsonValue::object();
			if (impl->folder_picker) {
				auto path = impl->folder_picker(impl->host ? impl->host->cwd() : std::string("."));
				out["path"] = path ? JsonValue(*path) : JsonValue(nullptr);
			} else {
				out["path"] = JsonValue(nullptr);
			}
			res.body = out.dump();
			return res;
		}
		if (req.method == "GET" && req.path == "/api/settings") {
			if (auto [code, body] = check_host(req); code != 0) {
				HttpResponse r; r.status = code; r.body = body; return r;
			}
			HttpResponse res;
			res.content_type = "application/json; charset=utf-8";
			res.body = impl->host ? impl->host->settings_json().dump() : "{}";
			return res;
		}
		if (req.method == "POST" && req.path == "/api/fetch-models") {
			// 设置 → API 配置 → 获取模型列表：从请求地址拉取可用模型
			if (auto [code, body] = check_host(req); code != 0) {
				HttpResponse r; r.status = code; r.body = body; return r;
			}
			HttpResponse res;
			res.content_type = "application/json; charset=utf-8";
			auto body = json_parse(req.body);
			if (!body) {
				res.status = 400;
				res.body = "{\"error\":\"Invalid JSON body\"}";
				return res;
			}
			std::string base_url = (*body)["baseUrl"].as_string("");
			std::string api_key = (*body)["apiKey"].as_string("");
			std::string api_format = (*body)["apiFormat"].as_string("anthropic-messages");
			std::string error;
			std::vector<std::string> models = fetch_remote_models(base_url, api_key, api_format, error);
			JsonValue out = JsonValue::object();
			JsonValue arr = JsonValue::array();
			for (const auto& m : models) arr.push_back(m);
			out["models"] = arr;
			if (!error.empty()) out["error"] = error;
			res.body = out.dump();
			return res;
		}
		if (req.method == "POST" && req.path == "/api/perf") {
			// perf instrumentation dump from the UI (?perf=1): append to data/perf.log.
			// write-only endpoint with no auth; the server binds loopback only, so
			// the remaining risk is DNS-rebinding (attacker page resolving one of
			// its names to 127.0.0.1): rejecting non-loopback Host headers closes it
			HttpResponse res;
			{
				auto it = req.headers.find("host");
				std::string host_hdr = it == req.headers.end() ? "" : it->second;
				std::string host_name;
				if (!host_hdr.empty() && host_hdr[0] == '[') {
					// IPv6 literal: compare up to the closing bracket
					size_t close = host_hdr.find(']');
					host_name = close == std::string::npos ? host_hdr : host_hdr.substr(0, close + 1);
				} else {
					size_t colon = host_hdr.find(':');
					host_name = colon == std::string::npos ? host_hdr : host_hdr.substr(0, colon);
				}
				if (host_name != "127.0.0.1" && host_name != "localhost" && host_name != "[::1]") {
					res.status = 403;
					res.body = "forbidden";
					return res;
				}
			}
			res.status = 200;
			res.content_type = "text/plain";
			if (!req.body.empty()) {
				std::string dir = path_join(impl->base_dir, "data");
				mkdirs(dir);
				std::string log_path = path_join(dir, "perf.log");
				FILE* f = _wfopen(utf8_to_wide(log_path).c_str(), L"ab");
				if (f) {
					fwrite(req.body.data(), 1, req.body.size(), f);
					fclose(f);
				}
			}
			res.body = "ok";
			return res;
		}
		if (req.method == "GET" || req.method == "HEAD") {
			auto res = static_file(impl->ui_dir, req.path);
			// Theme must be in place for the *first* paint: stamping data-theme on
			// <html> (instead of letting an inline script set it on <body>) makes the
			// :root-level variables valid as soon as the document starts painting, so
			// a light theme no longer flashes the dark defaults for half a second.
			if (req.path == "/" && res.status == 200 && impl->host) {
				// as_string(def) returns a temporary std::string — keep it alive in a
				// named local; .c_str() on the temporary would dangle immediately.
				const std::string theme =
					impl->host->settings().appearance()["theme"].as_string("midnight");
				res.body = apply_first_paint_theme(std::move(res.body), theme,
					css_canvas_theme(impl->base_dir, theme));
				// the injected theme is part of the bytes: refresh the validator so a
				// theme switch invalidates the cached copy of the shell
				res.etag = content_etag(res.body);
			}
			return res;
		}
		HttpResponse res;
		res.status = 405;
		res.body = "Method not allowed";
		return res;
	});

	impl_->http.set_ws_handlers(
		[impl = impl_.get()](WebSocketConnection& ws) {
			JsonValue ready = JsonValue::object();
			ready["type"] = "ready";
			ready["version"] = 1;
			ready["settings"] = impl->host->settings_json();
			ws.send_text(ready.dump());

			// auto-resume the most recent session on first client only if no session
			// is already active — avoids creating an empty session that gets torn down
			// the moment the frontend sends its own create_session (wasteful + confuses
			// the frontend's firstSnapshot tracking)
			bool expected = false;
			if (impl->auto_session.compare_exchange_strong(expected, true)) {
				if (!impl->host->is_active()) {
					try {
						impl->host->open_session("");
					} catch (const std::string& e) {
						JsonValue log = JsonValue::object();
						log["type"] = "log";
						log["level"] = "error";
						log["message"] = "Failed to open session: " + e;
						ws.send_text(log.dump());
					}
				}
			}
			if (impl->host->is_active()) {
				// full sync for late joiners
				try {
					JsonValue msg = JsonValue::object();
					msg["type"] = "session_updated";
					msg["sessionId"] = impl->host->session_id();
					msg["snapshot"] = impl->host->get_snapshot();
					ws.send_text(msg.dump());
				} catch (...) {
				}
				impl->broadcast_models();
				impl->broadcast_skills();
			}
		},
		[impl = impl_.get()](WebSocketConnection& ws, const std::string& text) {
			handle_ws_message(*impl, ws, text);
		},
		[](uint64_t) {});

	if (!impl_->http.start(0)) {
		fprintf(stderr, "[start] http bind failed\n");
		return false;
	}
	fprintf(stderr, "[start] ok\n");
	return true;
}

int ProtocolServer::port() const {
	return impl_->http.port();
}

void ProtocolServer::shutdown() {
	if (impl_->host) {
		impl_->host->teardown();
	}
	impl_->http.stop();
}

void ProtocolServer::wait_idle() const {
	// headless mode: park forever
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(3600));
	}
}

}  // namespace phi
