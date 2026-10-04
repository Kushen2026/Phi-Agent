// 会话持久化（重写版）。
//
// 存储模型
// ────────
// 每个工作目录对应一个"桶"目录 <data_dir>/sessions/<bucket>/，桶内每个会话
// 一个 .jsonl 文件。文件内容是一个完整的、自洽的会话快照：
//
//   {"type":"session","version":3,"id","cwd","timestamp"}     ← 头部，恒为第一行
//   {"type":"model","provider","modelId","timestamp"}         ← 最近一次的模型
//   {"type":"thinking","level","timestamp"}                   ← 最近一次的思考级别
//   {"type":"title","title","timestamp"}                      ← 用户命名的标题
//   {"type":"message","message":{...},"timestamp"}            ← 一条历史消息
//   {"type":"compaction","summary","tokensBefore","tokensAfter","timestamp"}
//
// 写入方式是"整体重写"：每次落盘都把 头部+元数据+完整历史 一次性写入临时
// 文件，然后原子 rename 覆盖目标。磁盘上任何时刻只有"旧完整内容"或"新完整
// 内容"，因此：
//   * 崩溃不可能产生残行/半截 JSON（没有追加，就没有写一半的行）；
//   * 消息不可能重复（文件是内存历史的投影，重写即覆盖）；
//   * 压缩历史直接体现为重写后的文件只含摘要，不需要回放时截断。
//
// 回放（replay）把入口流还原为消息历史，并对旧版追加式格式产生的遗留问题
// 做一次性自愈（残行跳过、相邻重复行去重、孤立工具结果剔除、悬空工具调用
// 补占位结果、尾部空助手消息删除）。修复结果会在下一次落盘时被固化。
//
// 元数据（模型/思考级别/标题）保存在内存，随下一次 sync 一起写入。
#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "agent/agent.hpp"
#include "util/json.hpp"

namespace phi {

// 一次回放（加载）的结果：消息历史 + 会话环境 + 自愈记录。
struct SessionContent {
	std::vector<AgentMessage> messages;
	std::string provider;        // 最近一次记录的模型提供方
	std::string model_id;        // 最近一次记录的模型 id
	std::string thinking_level;  // 最近一次记录的思考级别
	std::string title;           // 用户命名的会话标题
	std::string session_id;      // 头部记录的会话 id（缺失时取自文件名）
	// 回放时自动修复的问题（人可读，逐条一条）
	std::vector<std::string> repairs;
};

// 会话索引条目（目录扫描产物，不含消息正文）。
struct SessionSummary {
	std::string id;
	std::string title;      // 用户命名，未命名时为文件名主干
	std::string cwd;
	std::string file_path;
	int64_t updated_at = 0;  // 文件最后修改时刻（epoch ms）
	int message_count = 0;   // 已落盘的消息条数
};

class SessionArchive {
public:
	SessionArchive(std::string data_dir, std::string cwd);

	static std::string bucket_for(const std::string& cwd);  // "--C--x--y--"
	std::string bucket_dir() const;                         // <data_dir>/sessions/<bucket>

	// 空串表示尚未 start()/attach()
	const std::string& id() const { return id_; }
	const std::string& file() const { return file_; }

	// 新建会话文件（只写头部）。真正的元数据/消息随第一次 sync 落盘。
	void start();
	// 接管已有文件（继续使用）。
	void attach(const std::string& file);
	// 停止关联（会话切换/销毁时）。
	void detach();
	// 从未产生过任何消息的会话文件直接删除（避免空文件堆积）。
	void discard_if_empty();

	// ── 会话元数据（内存态，随下次 sync 落盘）─────────────────────────────
	void set_model(const std::string& provider, const std::string& model_id);
	void set_thinking(const std::string& level);
	void set_title(const std::string& title);
	// 强制下一次 sync 重写文件（历史被整体替换时使用，如压缩）。
	void mark_dirty();

	// 把当前历史整体重写到文件；内容无变化时内部直接跳过。
	void sync(const std::vector<AgentMessage>& messages);

	// 按最近更新排序的会话索引；从未有过消息的文件不会出现。
	std::vector<SessionSummary> index() const;
	// 按会话 id 精确定位文件。
	std::optional<std::string> locate(const std::string& id) const;
	std::optional<std::string> latest() const;

	// 回放一个会话文件；总是返回可用内容（文件缺失/不可解析时返回空）。
	static SessionContent replay(const std::string& file);

private:
	void write_header_locked();  // 仅写头部（新会话的初始状态）

	std::string data_dir_;
	std::string cwd_;
	std::string file_;
	std::string id_;
	std::mutex write_mutex_;

	// 内存元数据 + 落盘状态
	std::string provider_;
	std::string model_id_;
	std::string thinking_;
	std::string title_;
	bool dirty_ = false;             // 元数据变化，待落盘
	size_t last_synced_size_ = 0;     // 上次落盘时的消息条数
	bool created_fresh_ = false;      // 本进程新建（用于丢弃从未使用的文件）
};

// 消息 <-> JSON（持久化与 UI 快照投影共用同一套编解码）。
JsonValue message_to_json(const AgentMessage& m);
AgentMessage message_from_json(const JsonValue& j);

}  // namespace phi
