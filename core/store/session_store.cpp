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

// 会话文件名的可读时间戳：本地时间 "YYYY-MM-DD-HH-MM-SS"。小时不补零，和界面
// 上看到的一致。只给人看，不参与任何逻辑——会话 id 才是唯一标识。
// 文件名里必须用 '-' 而不是 ':'：'/' 与 ':' 在 NTFS 上是非法字符，带冒号的
// 时间戳会让文件根本写不出来（整个会话无法落盘）。显示名再把时分秒还原成 ':'。
static std::string fs_safe_stamp(int64_t ms) {
	time_t secs = (time_t)(ms / 1000);
	struct tm t;
#ifdef _WIN32
	localtime_s(&t, &secs);
#else
	localtime_r(&secs, &t);
#endif
	char buf[48];
	snprintf(buf, sizeof(buf), "%04d-%02d-%02d-%d-%02d-%02d",
		 t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
	return std::string(buf);
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

// 文件名 -> 给人看的会话名。默认会话的文件名是 "<YYYY-MM-DD-H-MM-SS>_<uuid>.jsonl"，
// 显示名取时间戳那一半，并把时间部分的 '-' 还原成 ':'（文件名不能用冒号，
// 界面上看起来应该是 2026-11-07-9:23:40）。命名过的会话另有 title 条目，不走这里。
std::string session_file_display_name(const std::string& file) {
	std::string stem = path_basename(file);
	if (ends_with(stem, ".jsonl")) stem = stem.substr(0, stem.size() - 6);
	size_t cut = stem.find('_');
	if (cut == std::string::npos) return stem;
	const std::string stamp = stem.substr(0, cut);
	// 只认 "YYYY-MM-DD-H-MM-SS" 这个形状（日期一段 + 时/分/秒三段），避免把
	// 用户自己起的名字里的 '-' 也改掉。
	if (stamp.size() < 16 || stamp[4] != '-' || stamp[7] != '-' || stamp[10] != '-')
		return stamp;
	const size_t h = 11;                       // "YYYY-MM-DD-H" 之后
	const size_t c1 = stamp.find('-', h);
	if (c1 == std::string::npos) return stamp;
	const size_t c2 = stamp.find('-', c1 + 1);
	if (c2 == std::string::npos) return stamp;
	// "2026-11-07-9-23-40" -> "2026-11-07-9:23:40"
	return stamp.substr(0, c1) + ":" + stamp.substr(c1 + 1, c2 - c1 - 1) + ":" +
	       stamp.substr(c2 + 1);
}

static std::string session_id_from_name(const std::string& name) {
	std::string stem = name;
	if (ends_with(stem, ".jsonl")) stem = stem.substr(0, stem.size() - 6);
	size_t cut = stem.find('_');
	return cut == std::string::npos ? stem : stem.substr(cut + 1);
}

// 数一个会话文件里已经落盘的消息条数（只看 type 字段，不回放消息正文）。
// attach() 用它把 last_synced_size_ 对齐到文件的真实状态。
static size_t count_message_entries(const std::string& file) {
	auto raw = read_file(file);
	if (!raw) return 0;
	size_t n = 0;
	for (const auto& line : split(*raw, '\n')) {
		std::string t = trim(line);
		if (t.empty()) continue;
		if (t.find("\"type\": \"message\"") != std::string::npos ||
		    t.find("\"type\":\"message\"") != std::string::npos)
			n++;
	}
	return n;
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
	// 不置 dirty：仅仅打开一个会话不该改写磁盘上的文件，否则「最近修改」时间
	// 会被打开动作本身刷成「刚刚」，即使一个字都没输入。回放方随后用 set_title /
	// set_model / set_thinking 回填元数据，只有值真的与文件不同时才置 dirty。
	dirty_ = false;
	// 「已同步的消息条数」取文件里现有的条数，而不是 0：sync() 用它判断历史是否
	// 增长过。置 0 会让第一次 sync 误以为「从 0 条变成 N 条」，于是立刻重写整个
	// 文件——正是「打开就变刚刚」的根因。这里只数行，不回放消息。
	last_synced_size_ = count_message_entries(file);
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

void SessionArchive::mark_clean() {
	std::lock_guard<std::mutex> lock(write_mutex_);
	dirty_ = false;
}

void SessionArchive::sync(const std::vector<AgentMessage>& messages) {
	std::lock_guard<std::mutex> lock(write_mutex_);
	if (file_.empty()) return;
	// 只有「消息条数变了」或「元数据被显式改过」才考虑落盘。打开一个旧会话
	// 什么都不输入时，这两个条件都不成立，因此不会碰磁盘、也不会刷新
	// 「最近修改」时间。
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

	// 内容没变就不写：每次 sync 都会重新盖一个 timestamp，直接写下去会把
	// 「最近修改」时间刷成现在，哪怕这次调用没有任何实际变化（典型场景：
	// 打开一个旧会话，apply_profile_after_start() 重新应用配置时触发一次
	// sync）。只比较内容而不比较时间戳，正是为了让这种无变化的调用不留痕迹。
	if (auto raw = read_file(file_); raw && *raw == content) {
		dirty_ = false;
		last_synced_size_ = messages.size();
		return;
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
			// 没有命名：显示文件名里的时间戳部分（"YYYY-MM-DD-H:MM:SS"），
			// 也就是列表上「这个会话是什么时候开的」。文件名同样是
			// "<时间戳>_<uuid>.jsonl"，所以这里取 _ 之前的一半，
			// 而不是之前那样把 uuid 当标题（一串 8c14f6cf 对谁都没有意义）。
			s.title = session_file_display_name(s.file_path);
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

JsonValue message_to_json(const AgentMessage& m, bool inline_media) {
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
				p["mimeType"] = part.image_mime;
				// 落盘只写路径：一个几 MB 的图片或视频整段 base64 进 jsonl，会话文件
				// 会大出一个数量级，而文件本来就一直在磁盘上。重放时按路径读回同样
				// 的字节（文件没了就只是少这一张图，历史文本仍然完整）。界面快照
				// （inline_media）则要 base64——聊天窗口得把图片直接画出来。
				if (!part.source_path.empty()) p["path"] = part.source_path;
				if (inline_media || part.source_path.empty() || !path_exists(part.source_path)) {
					p["data"] = part.image_data;
				}
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
			ContentPart p = ContentPart::image(part["data"].as_string(""), part["mimeType"].as_string("image/png"));
			// 落盘时只留了路径（见 message_to_json）：把文件读回来，恢复成当时发送的
			// 那一段字节。文件被删/被换掉时这张图就取不到了——历史里的文字与工具
			// 输出仍然完整，模型只是少了那张图。
			std::string src = part["path"].as_string("");
			if (!src.empty()) {
				std::vector<uint8_t> raw;
				if (p.image_data.empty() && read_file_bytes(src, raw) && !raw.empty()) {
					p.image_data = base64_encode_bytes(raw);
					p.source_path = src;
				}
			}
			m.content.push_back(std::move(p));
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
