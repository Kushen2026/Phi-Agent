#include "store/session_store.hpp"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <map>
#include <set>

#include "util/base.hpp"

namespace phi {

// ── 时间工具 ────────────────────────────────────────────────────────────────

static std::string iso_stamp(int64_t ms) {
	// epoch ms -> "YYYY-MM-DDTHH:MM:SS.mmmZ"（UTC）
	time_t secs = (time_t)(ms / 1000);
	int millis = (int)(ms % 1000);
	struct tm t;
#ifdef _WIN32
	gmtime_s(&t, &secs);
#else
	gmtime_r(&secs, &t);
#endif
	char buf[40];
	strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &t);
	char out[48];
	snprintf(out, sizeof(out), "%s.%03dZ", buf, millis);
	return out;
}

// ISO 串 -> 文件系统安全的主干（':' '.' 均替换为 '-'）。
static std::string fs_safe_stamp(int64_t ms) {
	return replace_all(replace_all(iso_stamp(ms), ":", "-"), ".", "-");
}

// ── 桶路径 ──────────────────────────────────────────────────────────────────

std::string SessionArchive::bucket_for(const std::string& cwd) {
	// "C:\a\b" -> "--C--a--b--"
	std::string s = cwd;
	while (!s.empty() && (s[0] == '/' || s[0] == '\\')) s = s.substr(1);
	s = replace_all(s, "\\", "-");
	s = replace_all(s, "/", "-");
	s = replace_all(s, ":", "-");
	return "--" + s + "--";
}

SessionArchive::SessionArchive(std::string data_dir, std::string cwd)
	: data_dir_(std::move(data_dir)), cwd_(std::move(cwd)) {}

std::string SessionArchive::bucket_dir() const {
	return path_join(data_dir_ + "/sessions", bucket_for(cwd_));
}

// ── 文件名约定 ──────────────────────────────────────────────────────────────
//
// "<fs-safe ISO 时间>_<会话id>.jsonl"。会话 id 是唯一权威标识：定位、去重、
// 打开都只认它。文件名里的时间戳仅用于人类浏览，不参与任何逻辑。

static std::string session_id_from_name(const std::string& name) {
	std::string stem = name;
	if (ends_with(stem, ".jsonl")) stem = stem.substr(0, stem.size() - 6);
	size_t cut = stem.find('_');
	return cut == std::string::npos ? stem : stem.substr(cut + 1);
}

// ── 生命周期 ────────────────────────────────────────────────────────────────

void SessionArchive::start() {
	detach();
	mkdirs(bucket_dir());
	id_ = uuid4();
	file_ = path_join(bucket_dir(), fs_safe_stamp(now_ms()) + "_" + id_ + ".jsonl");
	created_fresh_ = true;
	dirty_ = true;
	last_synced_size_ = 0;
	write_header_locked();
}

void SessionArchive::attach(const std::string& file) {
	detach();
	file_ = file;
	id_ = session_id_from_name(path_basename(file));
	created_fresh_ = false;
	dirty_ = true;              // 元数据未知，待回放方通过 set_* 补全
	last_synced_size_ = 0;
}

void SessionArchive::detach() {
	std::lock_guard<std::mutex> lock(write_mutex_);
	file_.clear();
	id_.clear();
	provider_.clear();
	model_id_.clear();
	thinking_.clear();
	title_.clear();
	dirty_ = false;
	last_synced_size_ = 0;
	created_fresh_ = false;
}

void SessionArchive::discard_if_empty() {
	std::string file;
	{
		std::lock_guard<std::mutex> lock(write_mutex_);
		if (!created_fresh_ || file_.empty()) return;
		// 有过消息（已经 sync 过真实内容）的文件不是空文件。
		if (last_synced_size_ > 0) return;
		// 兜底：文件里若已有消息行（例如接管后的 sync 已经写入），不删。
		if (auto raw = read_file(file_)) {
			if (raw->find("\"type\": \"message\"") != std::string::npos ||
				raw->find("\"type\":\"message\"") != std::string::npos) {
				return;
			}
		}
		file = file_;
		file_.clear();
		id_.clear();
		created_fresh_ = false;
	}
	remove_file(file);
}

// ── 元数据 ──────────────────────────────────────────────────────────────────

void SessionArchive::set_model(const std::string& provider, const std::string& model_id) {
	std::lock_guard<std::mutex> lock(write_mutex_);
	if (provider_ == provider && model_id_ == model_id) return;
	provider_ = provider;
	model_id_ = model_id;
	dirty_ = true;
}

void SessionArchive::set_thinking(const std::string& level) {
	std::lock_guard<std::mutex> lock(write_mutex_);
	if (thinking_ == level) return;
	thinking_ = level;
	dirty_ = true;
}

void SessionArchive::set_title(const std::string& title) {
	std::lock_guard<std::mutex> lock(write_mutex_);
	if (title_ == title) return;
	title_ = title;
	dirty_ = true;
}

void SessionArchive::mark_dirty() {
	std::lock_guard<std::mutex> lock(write_mutex_);
	dirty_ = true;
}

// ── 落盘（整体重写）─────────────────────────────────────────────────────────



void SessionArchive::write_header_locked() {
	if (file_.empty()) return;
	JsonValue head = JsonValue::object();
	head["type"] = "session";
	head["version"] = 3;
	head["id"] = id_;
	head["cwd"] = cwd_;
	head["timestamp"] = iso_stamp(now_ms());
	// atomic tmp+rename: a plain write here truncated the target FIRST, so a
	// crash mid-write left a 0-byte file — the session then replayed as empty
	// and every message in it was unrecoverable.
	if (!write_file_atomic(file_, head.dump() + "\n")) {
		// 写失败（磁盘满/权限）：保留旧文件内容不动，等待下次 sync 重试。
		return;
	}
}

void SessionArchive::sync(const std::vector<AgentMessage>& messages) {
	std::lock_guard<std::mutex> lock(write_mutex_);
	if (file_.empty()) return;
	if (!dirty_ && messages.size() == last_synced_size_) return;
	if (messages.empty()) return;  // 没有历史就不落盘（新会话只有头部）

	// 组装完整文件内容：头部 + 元数据 + 全部消息。
	std::string content;
	content.reserve(messages.size() * 512 + 256);
	{
		JsonValue head = JsonValue::object();
		head["type"] = "session";
		head["version"] = 3;
		head["id"] = id_;
		head["cwd"] = cwd_;
		head["timestamp"] = iso_stamp(now_ms());
		content += head.dump();
		content += "\n";
	}
	auto push_entry = [&](const char* type, JsonValue entry) {
		entry["type"] = type;
		entry["timestamp"] = iso_stamp(now_ms());
		content += entry.dump();
		content += "\n";
	};
	if (!provider_.empty() || !model_id_.empty()) {
		JsonValue e = JsonValue::object();
		e["provider"] = provider_;
		e["modelId"] = model_id_;
		push_entry("model", std::move(e));
	}
	if (!thinking_.empty()) {
		JsonValue e = JsonValue::object();
		e["level"] = thinking_;
		push_entry("thinking", std::move(e));
	}
	if (!title_.empty()) {
		JsonValue e = JsonValue::object();
		e["title"] = title_;
		push_entry("title", std::move(e));
	}
	for (const auto& m : messages) {
		JsonValue e = JsonValue::object();
		e["message"] = message_to_json(m);
		push_entry("message", std::move(e));
	}

	// 原子替换：写临时文件 + rename。崩溃后磁盘上要么是旧完整内容，要么是
	// 新完整内容，不存在残行或半截 JSON。
	if (!write_file_atomic(file_, content)) return;

	dirty_ = false;
	last_synced_size_ = messages.size();
	created_fresh_ = false;  // 已经是"有内容"的文件了
}

// ── 索引与定位 ──────────────────────────────────────────────────────────────

	std::vector<SessionSummary> SessionArchive::index() const {
	std::vector<SessionSummary> out;
	std::string dir = bucket_dir();
	if (!path_is_dir(dir)) return out;
	std::map<std::string, SessionSummary> by_id;
	for (const auto& name : list_dir(dir)) {
		if (!ends_with(name, ".jsonl")) continue;
		// 跳过原子写留下的临时文件
		if (name.find(".tmp") != std::string::npos) continue;
		std::string path = path_join(dir, name);
		SessionSummary s;
		s.id = session_id_from_name(name);
		if (s.id.empty()) continue;
		s.cwd = cwd_;
		s.file_path = path;
		s.updated_at = file_mtime_ms(path);
		// 快速启发式统计消息条数；同时探测"空会话"（无消息行）并跳过——
		// 历史遗留的空文件不该出现在列表里。
		s.message_count = 0;
		std::string stored_title;
		if (auto raw = read_file(path)) {
			// count via real JSON parsing: string-searching for '"type": "message"'
			// can match the same text inside a quoted string of another entry
			for (const auto& line : split(*raw, '\n')) {
				std::string t = trim(line);
				if (t.empty()) continue;
				auto entry = json_parse(t);
				if (!entry) continue;
				const std::string type = (*entry)["type"].as_string("");
				if (type == "message") s.message_count++;
				else if (type == "title") stored_title = (*entry)["title"].as_string("");
				else if (type == "session_name") {
					if (stored_title.empty())
						stored_title = (*entry)["name"].as_string("");
				}
			}
			if (s.message_count == 0) continue;  // 空会话（只有头部）不进索引
		}
		if (!stored_title.empty()) s.title = stored_title;
		auto it = by_id.find(s.id);
		if (it == by_id.end() || it->second.updated_at < s.updated_at) {
			by_id[s.id] = std::move(s);
		}
	}
	out.reserve(by_id.size());
	for (auto& [id, s] : by_id) {
		if (s.title.empty()) {
			// No title found in file: derive from filename
			s.title = path_basename(s.file_path);
			if (ends_with(s.title, ".jsonl")) s.title = s.title.substr(0, s.title.size() - 6);
			size_t cut = s.title.find('_');
			s.title = cut == std::string::npos ? s.title : s.title.substr(cut + 1);
		}
		out.push_back(std::move(s));
	}
	std::sort(out.begin(), out.end(),
		[](const SessionSummary& a, const SessionSummary& b) { return a.updated_at > b.updated_at; });
	return out;
}

std::optional<std::string> SessionArchive::locate(const std::string& id) const {
	if (id.empty()) return std::nullopt;
	std::string dir = bucket_dir();
	if (!path_is_dir(dir)) return std::nullopt;
	// 精确匹配文件名尾部的 id（id 是 uuid，不存在子串误命中）。
	std::string suffix = "_" + id + ".jsonl";
	std::string best;
	int64_t best_mtime = -1;
	for (const auto& name : list_dir(dir)) {
		if (!ends_with(name, suffix)) continue;
		std::string path = path_join(dir, name);
		int64_t mtime = file_mtime_ms(path);
		if (mtime > best_mtime) {
			best_mtime = mtime;
			best = path;
		}
	}
	if (best.empty()) return std::nullopt;
	return best;
}

std::optional<std::string> SessionArchive::latest() const {
	auto entries = index();
	if (entries.empty()) return std::nullopt;
	return entries.front().file_path;
}

// ── 回放（加载 + 自愈）──────────────────────────────────────────────────────

SessionContent SessionArchive::replay(const std::string& file) {
	SessionContent out;
	auto raw = read_file(file);
	if (!raw) return out;
	out.session_id = session_id_from_name(path_basename(file));

	// ── 阶段一：按序应用入口流，同时做行级修复 ────────────────────────────
	//  * 跳过残行（旧版追加式写入的崩溃产物）
	//  * compaction 之后旧历史作废（兼容旧格式文件）
	bool saw_compaction = false;
	for (const auto& line : split(*raw, '\n')) {
		std::string t = trim(line);
		if (t.empty()) continue;
		auto entry = json_parse(t);
		if (!entry) continue;  // 残行：跳过
		std::string type = (*entry)["type"].as_string("");
		if (type == "message") {
			if (saw_compaction) continue;  // compaction 截断旧历史（旧格式）
			const JsonValue* msg = (*entry).find("message");
			if (!msg || !msg->is_object()) continue;  // malformed entry: skip it
			out.messages.push_back(message_from_json(*msg));
		} else if (type == "model" || type == "model_change") {
			if (saw_compaction) continue;
			out.provider = (*entry)["provider"].as_string("");
			out.model_id = (*entry)["modelId"].as_string("");
		} else if (type == "thinking" || type == "thinking_level_change") {
			out.thinking_level = (*entry)["level"].as_string("");
			if (out.thinking_level.empty())
				out.thinking_level = (*entry)["thinkingLevel"].as_string("");
		} else if (type == "title" || type == "session_name") {
			out.title = (*entry)["title"].as_string("");
			if (out.title.empty()) out.title = (*entry)["name"].as_string("");
		} else if (type == "compaction") {
			// 压缩：历史被摘要替代。摘要本身作为一条 compactionSummary 消息
			// 进入历史（与 live 会话的结构一致）。
			out.messages.clear();
			AgentMessage m;
			m.role = "compactionSummary";
			m.summary = (*entry)["summary"].as_string("");
			m.tokens_before = (*entry)["tokensBefore"].as_int(0);
			m.tokens_after = (*entry)["tokensAfter"].as_int(0);
			m.timestamp = now_ms();
			out.messages.push_back(std::move(m));
			saw_compaction = true;
		} else if (type == "session") {
			// 头部：优先采用文件里记录的 id 与 cwd
			std::string hid = (*entry)["id"].as_string("");
			if (!hid.empty()) out.session_id = hid;
		}
	}
	// 收集工具调用 id 与已应答 id
	std::set<std::string> answered;
	for (const auto& m : out.messages) {
		if (m.role == "toolResult" && !m.tool_call_id.empty()) {
			answered.insert(m.tool_call_id);
		}
	}

	// 修复 2：为悬空的工具调用补上错误占位结果，**紧跟在发起它的助手消息之后**。
	// 悬空调用不只是在末尾：会话在流式过程中报错/中断时写下的那一条助手消息，
	// 可能早就被后面的正常回合推到中间去了，而这样的历史一旦发给提供方就会被
	// 拒绝（"insufficient tool messages following tool_calls message"），整段会话
	// 从此发不出去。旧实现只修尾部连续的悬空调用，中间的洞修不掉。
	std::vector<AgentMessage> fixed;
	fixed.reserve(out.messages.size() + 4);
	size_t dangling = 0;
	for (size_t i = 0; i < out.messages.size(); i++) {
		const AgentMessage& m = out.messages[i];
		const size_t asst_index = fixed.size();   // key: only store index
		fixed.push_back(m);
		if (m.role != "assistant") continue;
		// 这条助手消息后面已经跟着的结果（通常就是全部）
		size_t j = i + 1;
		while (j < out.messages.size() && out.messages[j].role == "toolResult") {
			fixed.push_back(out.messages[j]);
			j++;
		}
		// 缺的那些按调用顺序补在结果串的末尾，因此结果与调用的顺序和真实
		// 回合一致（提供方按 id 配对，但乱序的中途补位会让人看不出发生了什么）。
		// 没有 id 的调用（旧格式/网关不发 id 时写下的历史）在这里补一个：两种线格式
		// 都按 id 配对，空 id 的一对在线上一律被丢弃（assistant 侧留下一个没人应答的
		// tool_call，工具消息则被跳过）——那就是这条会话再也发不出去的原因。
		const size_t part_count = fixed[asst_index].content.size();
		for (size_t k = 0; k < part_count; k++) {
			ContentPart& p = fixed[asst_index].content[k];   // re-index each iteration
			if (p.type != ContentPart::Type::ToolCall) continue;
			if (p.tool_call_id.empty()) p.tool_call_id = "call_recovered_" + uuid4().substr(0, 12);
			if (answered.count(p.tool_call_id)) continue;
			// the id/name are copied out first: `fixed.push_back` below can reallocate
			// `fixed`, which invalidates the reference this loop walks
			const std::string id = p.tool_call_id;
			const std::string name = p.tool_name;
			AgentMessage tr;
			tr.role = "toolResult";
			tr.tool_call_id = id;
			tr.tool_name = name;
			tr.is_error = true;
			tr.timestamp = now_ms();
			tr.content.push_back(ContentPart::text_part("[会话中断，该工具未记录到结果]"));
			fixed.push_back(std::move(tr));
			answered.insert(id);
			dangling++;
		}
		i = j - 1;   // 上面已经把这条助手消息的结果串搬过去了
	}
	if (dangling > 0) {
		out.messages.swap(fixed);
		out.repairs.push_back("已为 " + std::to_string(dangling) + " 个未完成的工具调用补占位结果");
	}

	// 修复 3：删除尾部空助手消息（响应被中断、未产出任何内容）。
	size_t empties = 0;
	while (out.messages.size() >= 2) {
		const AgentMessage& last = out.messages.back();
		if (last.role != "assistant" || !last.content.empty()) break;
		out.messages.pop_back();
		empties++;
	}
	if (empties > 0) out.repairs.push_back("已移除 " + std::to_string(empties) + " 条空助手消息");

	return out;
}

// ── 消息编解码 ──────────────────────────────────────────────────────────────

JsonValue message_to_json(const AgentMessage& m) {
	JsonValue j = JsonValue::object();
	j["role"] = m.role;
	JsonValue content = JsonValue::array();
	for (const auto& part : m.content) {
		switch (part.type) {
			case ContentPart::Type::Text: {
				JsonValue p = JsonValue::object();
				p["type"] = "text";
				p["text"] = part.text;
				content.push_back(std::move(p));
				break;
			}
			case ContentPart::Type::Thinking: {
				JsonValue p = JsonValue::object();
				p["type"] = "thinking";
				p["thinking"] = part.text;
				if (part.redacted) p["redacted"] = true;
				content.push_back(std::move(p));
				break;
			}
			case ContentPart::Type::ToolCall: {
				JsonValue p = JsonValue::object();
				p["type"] = "toolCall";
				p["id"] = part.tool_call_id;
				p["name"] = part.tool_name;
				p["arguments"] = part.arguments;
				content.push_back(std::move(p));
				break;
			}
			case ContentPart::Type::Image: {
				JsonValue p = JsonValue::object();
				p["type"] = "image";
				p["data"] = part.image_data;
				p["mimeType"] = part.image_mime;
				content.push_back(std::move(p));
				break;
			}
		}
	}
	j["content"] = std::move(content);
	j["timestamp"] = (int64_t)m.timestamp;
	if (m.role == "assistant") {
		if (!m.model_provider.empty()) {
			JsonValue model = JsonValue::object();
			model["provider"] = m.model_provider;
			model["id"] = m.model_id;
			model["name"] = m.model_name;
			j["model"] = std::move(model);
		}
		if (!m.stop_reason.empty()) j["stopReason"] = m.stop_reason;
		if (!m.error_message.empty()) j["errorMessage"] = m.error_message;
		if (m.has_usage) {
			JsonValue usage = JsonValue::object();
			usage["input"] = (int64_t)m.usage.input;
			usage["output"] = (int64_t)m.usage.output;
			usage["cacheRead"] = (int64_t)m.usage.cache_read;
			usage["cacheWrite"] = (int64_t)m.usage.cache_write;
			usage["totalTokens"] = (int64_t)m.usage.total_tokens;
			JsonValue cost = JsonValue::object();
			cost["total"] = m.usage.cost;
			usage["cost"] = std::move(cost);
			j["usage"] = std::move(usage);
		}
	}
	if (m.role == "toolResult") {
		j["toolCallId"] = m.tool_call_id;
		j["toolName"] = m.tool_name;
		j["isError"] = m.is_error;
	}
	if (m.role == "compactionSummary") {
		j["summary"] = m.summary;
		j["tokensBefore"] = (int64_t)m.tokens_before;
		j["tokensAfter"] = (int64_t)m.tokens_after;
	}
	if (m.role == "subagentSummary") {
		j["description"] = m.subagent_description;
		j["isError"] = m.is_error;
	}
	return j;
}

AgentMessage message_from_json(const JsonValue& j) {
	AgentMessage m;
	m.role = j["role"].as_string("user");
	m.timestamp = j["timestamp"].as_int(now_ms());
	for (const auto& part : j["content"].items()) {
		std::string type = part["type"].as_string("");
		if (type == "text") {
			m.content.push_back(ContentPart::text_part(part["text"].as_string("")));
		} else if (type == "thinking") {
			m.content.push_back(ContentPart::thinking(part["thinking"].as_string(""), part["redacted"].as_bool(false)));
		} else if (type == "toolCall") {
			m.content.push_back(
				ContentPart::tool_call(part["id"].as_string(""), part["name"].as_string("tool"), part["arguments"]));
		} else if (type == "image") {
			m.content.push_back(ContentPart::image(part["data"].as_string(""), part["mimeType"].as_string("image/png")));
		}
	}
	if (m.role == "assistant") {
		if (const JsonValue* model = j.find("model")) {
			m.model_provider = (*model)["provider"].as_string("");
			m.model_id = (*model)["id"].as_string("");
			m.model_name = (*model)["name"].as_string("");
		}
		m.stop_reason = j["stopReason"].as_string("");
		m.error_message = j["errorMessage"].as_string("");
		if (const JsonValue* usage = j.find("usage")) {
			m.has_usage = true;
			m.usage.input = (*usage)["input"].as_int(0);
			m.usage.output = (*usage)["output"].as_int(0);
			m.usage.cache_read = (*usage)["cacheRead"].as_int(0);
			m.usage.cache_write = (*usage)["cacheWrite"].as_int(0);
			m.usage.total_tokens = (*usage)["totalTokens"].as_int(0);
			m.usage.cost = (*usage)["cost"]["total"].as_number(0);
		}
	} else if (m.role == "toolResult") {
		m.tool_call_id = j["toolCallId"].as_string("");
		m.tool_name = j["toolName"].as_string("tool");
		m.is_error = j["isError"].as_bool(false);
	} else if (m.role == "compactionSummary") {
		m.summary = j["summary"].as_string("");
		m.tokens_before = j["tokensBefore"].as_int(0);
		m.tokens_after = j["tokensAfter"].as_int(0);
	} else if (m.role == "subagentSummary") {
		m.subagent_description = j["description"].as_string("");
		m.is_error = j["isError"].as_bool(false);
	}
	return m;
}

}  // namespace phi
