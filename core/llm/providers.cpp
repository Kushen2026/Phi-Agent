// LLM stream provider: anthropic-messages + openai-completions (SSE).
#include "agent/agent.hpp"
#include "net/http_client.hpp"
#include "util/i18n.hpp"

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace phi {

// ── stop-reason mappings (ported from TS) ───────────────────────────────────
// C++ StreamEvent::stop_reason contract: "endTurn" | "toolUse" | "aborted"
// | "error" | "maxTokens".  The agent loop checks == "toolUse" to drive
// the next tool-execution turn.

static std::string mapAnthropicStop(std::string_view r) {
	if (r == "end_turn" || r == "pause_turn" || r == "stop_sequence") return "endTurn";
	if (r == "max_tokens" || r == "max_output_tokens") return "maxTokens";
	if (r == "tool_use")   return "toolUse";
	if (r == "refusal" || r == "sensitive") return "error";
	return std::string(r);
}

static std::string mapOpenAIStop(std::string_view r) {
	if (r == "stop" || r == "end" || r.empty()) return "endTurn";
	if (r == "length")    return "maxTokens";
	if (r == "tool_calls" || r == "function_call") return "toolUse";
	if (r == "content_filter" || r == "network_error") return "error";
	return "error";
}

// ── thinking budget defaults (TS: DEFAULT_THINKING_BUDGETS in simple-options.ts)
static const struct { const char* level; int64_t tokens; } kThinkingBuckets[] = {
	{"minimal", 1024},
	{"low",     2048},
	{"medium",  8192},
	{"high",    16384},
};
static int64_t thinkingBudgetForLevel(const std::string& level) {
	for (auto& e : kThinkingBuckets)
		if (e.level == level) return e.tokens;
	return 8192;  // default = medium
}

// ── URL helpers ─────────────────────────────────────────────────────────────

static void urlAppendPath(std::string& base, std::string_view path) {
	// avoid double "/v1" when the configured baseUrl already ends with /v1
	if (ends_with(base, "/v1") && starts_with(path, "/v1/")) path.remove_prefix(4);
	if (base.empty() || base.back() == '/') base += std::string(path);
	else base += "/" + std::string(path);
}

// ── Anthropic request builder ───────────────────────────────────────────────

static JsonValue buildAnthropicBody(
	const ModelDef& model,
	const std::vector<AgentMessage>& context,
	const StreamOptions& options) {

	JsonValue body = JsonValue::object();
	body["model"] = model.id;
	int64_t max_tok = model.max_tokens;
	if (max_tok <= 0) max_tok = 32000;
	// integer: a double would dump "64000.0" which Go gateways reject
	// (cannot unmarshal number into uint -> HTTP 500)
	body["max_tokens"] = max_tok;
	body["stream"] = true;

	// system prompt
	if (!options.system_prompt.empty()) {
		body["system"] = options.system_prompt;
	}

	// messages
	//
	// The tool results of one assistant turn all ride ONE user message: the
	// assistant's `tool_use` blocks must be answered by `tool_result` blocks in
	// the message that immediately follows, and Anthropic itself treats
	// consecutive user turns as one. Emitting one user message per result (what
	// this did) leaves the *first* message answering only the first call, and a
	// gateway — or any hop that speaks the OpenAI wire format, which validates
	// "an assistant message with 'tool_calls' must be followed by tool messages
	// responding to each 'tool_call_id'" — rejects the request with exactly that
	// error as soon as one turn issues several tool calls (which is what reading
	// several pictures at once does). Same shape the openai-completions builder
	// already produces for `tool` messages; see the note on tool_images below.
	JsonValue msgs = JsonValue::array();
	JsonValue tool_results = JsonValue::array();
	auto flush_tool_results = [&]() {
		if (tool_results.size() == 0) return;
		JsonValue um = JsonValue::object();
		um["role"] = "user";
		um["content"] = tool_results;
		msgs.push_back(std::move(um));
		tool_results = JsonValue::array();
	};
	for (const auto& msg : context) {
		// anything that is not a tool result ends the batch: the buffered blocks
		// are emitted in front of it, never split across the batch
		if (msg.role != "toolResult") flush_tool_results();
		if (msg.role == "user" || msg.role == "compactionSummary") {
			if (msg.role == "compactionSummary" && !msg.summary.empty()) {
				JsonValue m = JsonValue::object();
				m["role"] = "user";
				m["content"] = msg.summary;
				msgs.push_back(m);
			} else if (!msg.content.empty()) {
				JsonValue m = JsonValue::object();
				m["role"] = "user";
				if (msg.content.size() == 1u
				        && msg.content[0].type == ContentPart::Type::Text
				        && msg.content[0].image_data.empty()) {
					m["content"] = msg.content[0].text;
				} else {
					JsonValue parts = JsonValue::array();
					for (const auto& p : msg.content) {
						if (p.type == ContentPart::Type::Text && !p.text.empty()) {
							JsonValue tb = JsonValue::object();
							tb["type"] = "text";
							tb["text"] = p.text;
							parts.push_back(tb);
						} else if (p.type == ContentPart::Type::Image && !p.image_data.empty()) {
							JsonValue ib = JsonValue::object();
							ib["type"] = "image";
							JsonValue src = JsonValue::object();
							src["type"] = "base64";
							src["media_type"] =
								p.image_mime.empty() ? "image/png" : p.image_mime;
							src["data"] = p.image_data;
							ib["source"] = src;
							parts.push_back(ib);
						}
					}
					if (parts.size() == 1u && ((const JsonValue&)parts)[(size_t)0].is_string())
						m["content"] = ((const JsonValue&)parts)[(size_t)0].as_string();
					else
						m["content"] = parts;
				}
				msgs.push_back(m);
			}
		} else if (msg.role == "assistant") {
			if (msg.content.empty()) continue;
			JsonValue m = JsonValue::object();
			m["role"] = "assistant";
			JsonValue parts = JsonValue::array();
			for (const auto& p : msg.content) {
				if (p.type == ContentPart::Type::Text && !p.text.empty()) {
					JsonValue tb = JsonValue::object();
					tb["type"] = "text";
					tb["text"] = p.text;
					parts.push_back(tb);
				} else if (p.type == ContentPart::Type::Thinking) {
					if (p.redacted) {
						JsonValue rb = JsonValue::object();
						rb["type"] = "redacted_thinking";
						rb["data"] = p.text;
						parts.push_back(rb);
					} else {
						JsonValue tb = JsonValue::object();
						tb["type"] = "thinking";
						tb["thinking"] = p.text;
						parts.push_back(tb);
					}
				} else if (p.type == ContentPart::Type::ToolCall) {
					JsonValue tb = JsonValue::object();
					tb["type"] = "tool_use";
					tb["id"] = p.tool_call_id;
					tb["name"] = p.tool_name;
					tb["input"] = p.arguments.is_null() ? JsonValue::object() : p.arguments;
					parts.push_back(tb);
				}
			}
			m["content"] = parts;
			msgs.push_back(m);
		} else if (msg.role == "subagentSummary") {
			// a finished background subagent's report: its own user turn so the main
			// model reads it as context (never folded into a tool result)
			JsonValue m = JsonValue::object();
			m["role"] = "user";
			std::string txt;
			for (const auto& p : msg.content)
				if (p.type == ContentPart::Type::Text) txt += p.text;
			std::string text = msg.subagent_description.empty()
				? std::string("A subagent has finished:")
				: ("Subagent \"" + msg.subagent_description + "\" has finished:");
			text += "\n" + (txt.empty() ? std::string("(no output)") : txt);
			m["content"] = text;
			msgs.push_back(m);
		} else if (msg.role == "toolResult") {
			if (msg.content.empty() && msg.tool_call_id.empty()) continue;
			JsonValue tr = JsonValue::object();
			tr["type"] = "tool_result";
			tr["tool_use_id"] = msg.tool_call_id;
			if (msg.is_error) tr["is_error"] = true;
			std::string txt;
			std::vector<const ContentPart*> imgs;
			for (const auto& p : msg.content) {
				if (p.type == ContentPart::Type::Text) txt += p.text;
				else if (p.type == ContentPart::Type::Image && !p.image_data.empty()) imgs.push_back(&p);
			}
			if (imgs.empty()) {
				tr["content"] = txt.empty() ? "(no output)" : txt;
			} else {
				// multimodal tool result: image I/O lets a tool (read) hand the model a
				// picture. Anthropic accepts an array of blocks inside tool_result.
				JsonValue blocks = JsonValue::array();
				if (!txt.empty()) {
					JsonValue tb = JsonValue::object();
					tb["type"] = "text";
					tb["text"] = txt;
					blocks.push_back(tb);
				}
				for (const ContentPart* p : imgs) {
					JsonValue ib = JsonValue::object();
					ib["type"] = "image";
					JsonValue src = JsonValue::object();
					src["type"] = "base64";
					src["media_type"] = p->image_mime.empty() ? "image/png" : p->image_mime;
					src["data"] = p->image_data;
					ib["source"] = src;
					blocks.push_back(ib);
				}
				tr["content"] = blocks;
			}
			// (an empty tool_call_id would be an invalid block; the agent always
			// fills one in, and a result that lost its call is skipped - same rule
			// the openai-completions builder applies below)
			if (msg.tool_call_id.empty()) continue;
			tool_results.push_back(std::move(tr));
		}
	}
	// trailing tool results: a history that ends in a tool result still has to
	// hand them over
	flush_tool_results();
	if (!msgs.is_null()) body["messages"] = msgs;

	// tools (from options.tools_json)
	if (!options.tools_json.empty()) {
		auto jp = json_parse(options.tools_json);
		if (jp && jp->is_array() && jp->size() > 0)
			body["tools"] = *jp;
	}

	// thinking (budget-based, ported from TS)
	if (model.reasoning && !options.thinking_level.empty()) {
		std::string mapped = model.map_thinking(options.thinking_level);
		if (mapped != "off" && !mapped.empty()) {
			JsonValue think = JsonValue::object();
			think["type"] = "enabled";
			think["budget_tokens"] = (double)thinkingBudgetForLevel(options.thinking_level);
			body["thinking"] = think;
		}
	}
	return body;
}

// ── OpenAI request builder ──────────────────────────────────────────────────

static JsonValue buildOpenAIBody(
	const ModelDef& model,
	const std::vector<AgentMessage>& context,
	const StreamOptions& options) {

	JsonValue body = JsonValue::object();
	body["model"] = model.id;
	body["stream"] = true;
	JsonValue so = JsonValue::object();
	so["include_usage"] = true;
	body["stream_options"] = so;

	int64_t max_tok = model.max_tokens;
	// integer: a double would dump "64000.0" which Go gateways reject
	// (cannot unmarshal number into uint -> HTTP 500)
	if (max_tok > 0) body["max_tokens"] = max_tok;

	// reasoning_effort (clamped: xhigh/max -> high, off -> omit)
	if (model.reasoning && !options.thinking_level.empty()) {
		std::string mapped = model.map_thinking(options.thinking_level);
		if (mapped != "off" && !mapped.empty()) {
			if (mapped == "xhigh" || mapped == "max") mapped = "high";
			body["reasoning_effort"] = mapped;
		}
	}

	JsonValue msgs = JsonValue::array();
	// system prompt: openai-completions has no top-level system field — it is a
	// leading system message. Without this the request silently loses ALL
	// instructions (e.g. the /compact summarizer prompt: the model then just
	// continues the raw transcript it was handed).
	if (!options.system_prompt.empty()) {
		JsonValue m = JsonValue::object();
		m["role"] = "system";
		m["content"] = options.system_prompt;
		msgs.push_back(m);
	}
	// ── tool-result images ────────────────────────────────────────────────────
	// chat-completions `tool` messages are text-only, so a picture a tool
	// returned has to travel in a *user* message. It cannot travel in the one
	// that follows its own tool result, though: an assistant turn that issued N
	// tool calls must be answered by N tool messages, back to back, and a user
	// message wedged in between makes OpenAI reject the whole request with
	// "an assistant message with 'tool_calls' must be followed by tool messages
	// responding to each 'tool_call_id'" — which is what happened as soon as one
	// turn read two pictures. The pictures are buffered here and flushed after
	// the whole batch of tool messages instead.
	JsonValue tool_images = JsonValue::array();
	auto flush_tool_images = [&]() {
		if (tool_images.size() == 0) return;
		JsonValue um = JsonValue::object();
		um["role"] = "user";
		um["content"] = tool_images;
		msgs.push_back(std::move(um));
		tool_images = JsonValue::array();
	};
	for (const auto& msg : context) {
		// anything that is not a tool result ends the batch: buffered pictures
		// are emitted in front of it, never inside the batch
		if (msg.role != "toolResult") flush_tool_images();
		if (msg.role == "user") {
			JsonValue m = JsonValue::object();
			m["role"] = "user";
			if (msg.content.size() == 1u
			        && msg.content[0].type == ContentPart::Type::Text
			        && msg.content[0].image_data.empty()) {
				m["content"] = msg.content[0].text;
			} else {
				JsonValue parts = JsonValue::array();
				for (const auto& p : msg.content) {
					if (p.type == ContentPart::Type::Text && !p.text.empty()) {
						JsonValue tb = JsonValue::object();
						tb["type"] = "text";
						tb["text"] = p.text;
						parts.push_back(tb);
					} else if (p.type == ContentPart::Type::Image && !p.image_data.empty()) {
						JsonValue ib = JsonValue::object();
						ib["type"] = "image_url";
						JsonValue iu = JsonValue::object();
						std::string mime = p.image_mime.empty() ? "image/png" : p.image_mime;
						iu["url"] = "data:" + mime + ";base64," + p.image_data;
						ib["image_url"] = iu;
						parts.push_back(ib);
					}
				}
				if (!parts.is_null()) m["content"] = parts;
			}
			msgs.push_back(m);
		} else if (msg.role == "assistant") {
			if (msg.content.empty()) continue;
			JsonValue m = JsonValue::object();
			m["role"] = "assistant";
			std::string txt;
			bool has_tc = false;
			for (const auto& p : msg.content) {
				if (p.type == ContentPart::Type::Text) txt += p.text;
				if (p.type == ContentPart::Type::ToolCall) has_tc = true;
			}
			m["content"] = txt.empty() ? "" : txt;
			if (has_tc) {
				JsonValue tcs = JsonValue::array();
				for (const auto& p : msg.content) {
					if (p.type == ContentPart::Type::ToolCall) {
						JsonValue tc = JsonValue::object();
						tc["id"] = p.tool_call_id;
						tc["type"] = "function";
						JsonValue fn = JsonValue::object();
						fn["name"] = p.tool_name;
						// OpenAI spec: function.arguments is a JSON-encoded STRING, not an
						// object. Serializing the parsed object gets HTTP 400 on the next
						// round ("invalid type: map, expected a string"), which killed the
						// loop after exactly one tool round.
						fn["arguments"] = p.arguments.is_null() ? std::string("{}") : p.arguments.dump();
						tc["function"] = fn;
						tcs.push_back(tc);
					}
				}
				m["tool_calls"] = tcs;
			}
			msgs.push_back(m);
		} else if (msg.role == "subagentSummary") {
			// a finished background subagent's report: its own user turn so the main
			// model reads it as context (never folded into a tool result)
			JsonValue m = JsonValue::object();
			m["role"] = "user";
			std::string txt;
			for (const auto& p : msg.content)
				if (p.type == ContentPart::Type::Text) txt += p.text;
			std::string text = msg.subagent_description.empty()
				? std::string("A subagent has finished:")
				: ("Subagent \"" + msg.subagent_description + "\" has finished:");
			text += "\n" + (txt.empty() ? std::string("(no output)") : txt);
			m["content"] = text;
			msgs.push_back(m);
		} else if (msg.role == "toolResult") {
			if (msg.tool_call_id.empty()) continue;
			JsonValue m = JsonValue::object();
			m["role"] = "tool";
			std::string txt;
			std::vector<const ContentPart*> imgs;
			for (const auto& p : msg.content) {
				if (p.type == ContentPart::Type::Text) txt += p.text;
				else if (p.type == ContentPart::Type::Image && !p.image_data.empty()) imgs.push_back(&p);
			}
			m["content"] = txt.empty() ? "(no output)" : txt;
			m["tool_call_id"] = msg.tool_call_id;
			msgs.push_back(m);
			// the images ride the buffered user message (see flush_tool_images):
			// each block is prefixed with the tool it came from, so the batch of
			// pictures still reads as one picture per tool result
			if (!imgs.empty()) {
				JsonValue tb = JsonValue::object();
				tb["type"] = "text";
				const std::string tool = msg.tool_name.empty() ? std::string("tool") : msg.tool_name;
				tb["text"] = tr(options.language,
					"以下是工具 " + tool + " 返回的图片：",
					"Image(s) returned by tool " + tool + ":");
				tool_images.push_back(std::move(tb));
				for (const ContentPart* p : imgs) {
					JsonValue ib = JsonValue::object();
					ib["type"] = "image_url";
					JsonValue iu = JsonValue::object();
					std::string mime = p->image_mime.empty() ? "image/png" : p->image_mime;
					iu["url"] = "data:" + mime + ";base64," + p->image_data;
					ib["image_url"] = iu;
					tool_images.push_back(std::move(ib));
				}
			}
		} else if (msg.role == "compactionSummary" && !msg.summary.empty()) {
			JsonValue m = JsonValue::object();
			m["role"] = "user";
			m["content"] = msg.summary;
			msgs.push_back(m);
		}
	}
	// trailing tool images: a history that ends in a tool result still has to
	// hand them over
	flush_tool_images();
	body["messages"] = msgs;

	// tools (from options.tools_json; Anthropic schema -> OpenAI function format).
	// Without this, openai-completions models never learn the tools exist and
	// cannot call any of them.
	if (!options.tools_json.empty()) {
		auto jp = json_parse(options.tools_json);
		if (jp && jp->is_array() && jp->size() > 0) {
			JsonValue tools = JsonValue::array();
			for (const auto& t : jp->items()) {
				JsonValue fn = JsonValue::object();
				fn["name"] = t["name"].as_string("");
				fn["description"] = t["description"].as_string("");
				fn["parameters"] = t["input_schema"].is_null() ? JsonValue::object() : t["input_schema"];
				JsonValue ot = JsonValue::object();
				ot["type"] = "function";
				ot["function"] = std::move(fn);
				tools.push_back(std::move(ot));
			}
			body["tools"] = std::move(tools);
		}
	}
	return body;
}

// ── lightweight SSE decoder ─────────────────────────────────────────────────

struct SseDec {
	std::string event;
	std::string data;
};

// Extract one LF-terminated line from buf; returns false when none available.
static bool takeSseLine(std::string& buf, std::string& out) {
	size_t nl = buf.find('\n');
	size_t cr = buf.find('\r');
	size_t pos;
	if (nl == std::string::npos && cr == std::string::npos) return false;
	if (nl == std::string::npos) pos = cr;
	else if (cr == std::string::npos) pos = nl;
	else pos = (cr < nl ? cr : nl);
	out = buf.substr(0, pos);
	buf.erase(0, pos + 1);
	if (!out.empty() && out.back() == '\r') out.pop_back();
	return true;
}

static void feedSse(SseDec& d, std::string_view line) {
	if (line.empty()) return;  // blank separator: state flushed by caller
	size_t col = line.find(':');
	std::string_view key = col == std::string::npos ? line : line.substr(0, col);
	std::string val;
	if (col != std::string::npos) {
		val = std::string(line.substr(col + 1));
		if (!val.empty() && val[0] == ' ') val = val.substr(1);
	}
	if (key == "event") d.event = val;
	else if (key == "data") d.data += val + "\n";
}

// ── stream_llm ───────────────────────────────────────────────────────────────

void stream_llm(const ModelDef& model, const std::vector<AgentMessage>& context,
	const StreamOptions& options, const StreamCallback& on_event) {

	const std::string& api_key = options.api_key;
	if (api_key.empty()) {
		on_event(StreamEvent{StreamEvent::Type::Start});
		StreamEvent err;
		err.type = StreamEvent::Type::Error;
		err.text = "no api key configured";
		err.error_message = "no api key configured";
		on_event(err);
		return;
	}

	// ── anthropic-messages ──────────────────────────────────────────────────
	if (model.api == "anthropic-messages") {
		std::string url = model.base_url;
		urlAppendPath(url, "/v1/messages");
		JsonValue bodyVal = buildAnthropicBody(model, context, options);
		std::string bodyStr = bodyVal.dump();
		std::vector<std::pair<std::string, std::string>> headers;
		headers.emplace_back("x-api-key", api_key);
		headers.emplace_back("anthropic-version", "2023-06-01");
		headers.emplace_back("content-type", "application/json");
		headers.emplace_back("accept", "application/json");

		on_event(StreamEvent{StreamEvent::Type::Start});

		SseDec d;
		std::string raw;
		// full response body kept for non-SSE error payloads (HTTP >= 400);
		// `raw` gets consumed by the SSE line splitter, leaving nothing to parse
		std::string raw_body;
		bool stop_emitted = false;
		bool consumer_stop = false;

		// forward an event to the consumer; a false return means "stop the stream"
		auto fire = [&](const StreamEvent& ev) {
			if (consumer_stop) return;
			if (!on_event(ev)) consumer_stop = true;
		};

		auto on_chunk = [&](const char* data, size_t len) -> bool {
			if (consumer_stop) return false;
			raw.append(data, len);
			if (raw_body.size() < 65536) raw_body.append(data, len);
			std::string line;
			while (takeSseLine(raw, line)) {
				if (!line.empty()) {
					feedSse(d, line);
					continue;
				}
				if (d.event.empty() && d.data.empty()) continue;
				std::string dat = d.data;
				if (!dat.empty() && dat.back() == '\n') dat.pop_back();
				std::string ev = d.event;
				d.event.clear(); d.data.clear();

				if (ev == "error") {
					StreamEvent err;
					err.type = StreamEvent::Type::Error;
					err.text = "stream error: " + dat;
					err.error_message = "stream error: " + dat;
					fire(err);
					stop_emitted = true;
					return !consumer_stop;
				}

				auto jp = json_parse(dat);
				if (!jp) continue;
				const JsonValue& j = *jp;
				// field-presence-checked parsing: a provider omitting "type" (or
				// any nested field) must not crash the stream — skip instead
				const JsonValue* type_v = j.find("type");
				if (!type_v || !type_v->is_string()) continue;
				std::string type = type_v->as_string();

				if (type == "message_start") {
					const JsonValue* u = j.find("message");
					if (!u || !u->is_object()) continue;
					u = u->find("usage");
					if (!u || !u->is_object()) continue;
					StreamEvent ue;
					ue.type = StreamEvent::Type::Start;
					ue.usage.input      = (int64_t)u->at("input_tokens").as_number();
					ue.usage.output     = (int64_t)u->at("output_tokens").as_number();
					ue.usage.cache_read  = (int64_t)u->at("cache_read_input_tokens").as_number();
					ue.usage.cache_write = (int64_t)u->at("cache_creation_input_tokens").as_number();
					fire(ue);
				} else if (type == "content_block_start") {
					const JsonValue* block = j.find("content_block");
					if (!block || !block->is_object()) continue;
					std::string bt = block->at("type").as_string();
					int idx = (int)j.at("index").as_number();
					if (bt == "tool_use") {
						StreamEvent te;
						te.type = StreamEvent::Type::ToolCallStart;
						te.tool_index = idx;
						te.tool_id = block->at("id").as_string();
						te.tool_name = block->at("name").as_string();
						fire(te);
					}
				} else if (type == "content_block_delta") {
					const JsonValue* delta = j.find("delta");
					if (!delta || !delta->is_object()) continue;
					std::string dt = delta->at("type").as_string();
					int idx = (int)j.at("index").as_number();
					if (dt == "text_delta") {
						StreamEvent te;
						te.type = StreamEvent::Type::TextDelta;
						te.text = delta->at("text").as_string();
						fire(te);
					} else if (dt == "thinking_delta") {
						StreamEvent te;
						te.type = StreamEvent::Type::ThinkingDelta;
						te.text = delta->at("thinking").as_string();
						fire(te);
					} else if (dt == "input_json_delta") {
						StreamEvent te;
						te.type = StreamEvent::Type::ToolCallArgsDelta;
						te.tool_index = idx;
						te.args_delta = delta->at("partial_json").as_string();
						fire(te);
					}
				} else if (type == "message_delta") {
					const JsonValue* delta = j.find("delta");
					if (!delta || !delta->is_object()) continue;
					std::string sr = delta->at("stop_reason").as_string();
					if (!sr.empty() && !stop_emitted) {
						StreamEvent de;
						de.type = StreamEvent::Type::Done;
						de.stop_reason = mapAnthropicStop(sr);
						if (const JsonValue* u = j.find("usage")) {
							de.usage.output     = (int64_t)u->at("output_tokens").as_number();
							de.usage.cache_read  = (int64_t)u->at("cache_read_input_tokens").as_number();
							de.usage.cache_write = (int64_t)u->at("cache_creation_input_tokens").as_number();
						}
						fire(de);
						stop_emitted = true;
					}
				}
			}
			return !consumer_stop;
		};

		HttpStreamResult hres = http_post_stream(url, headers, bodyStr, on_chunk, 0, options.cancelled);
		if (!hres.ok) {
			if (!stop_emitted) {
				StreamEvent err;
				err.type = StreamEvent::Type::Error;
				err.text = hres.error;
				err.error_message = hres.error;
				on_event(err);
			}
			return;
		}
		if (hres.status >= 400) {
			// error payload is JSON, not SSE; surface it. Field-presence checked:
			// gateways wrap errors in arbitrary shapes
			std::string msg = "HTTP " + std::to_string(hres.status);
			auto parsed = json_parse(raw_body);
			if (parsed) {
				std::string em;
				if (const JsonValue* e = parsed->find("error"))
					em = e->at("message").as_string("");
				if (em.empty()) em = parsed->at("message").as_string("");
				if (!em.empty()) msg += ": " + em;
				else msg += ": " + parsed->dump().substr(0, 400);
			} else if (!raw_body.empty()) {
				msg += ": " + raw_body.substr(0, 300);
			}
			if (!stop_emitted) {
				StreamEvent err;
				err.type = StreamEvent::Type::Error;
				err.text = msg;
				err.error_message = msg;
				on_event(err);
			}
			return;
		}

		// flush trailing data (incomplete final SSE frame) — field-presence checked
		if (!stop_emitted && (!d.event.empty() || !d.data.empty())) {
			std::string dat = d.data;
			if (!dat.empty() && dat.back() == '\n') dat.pop_back();
			auto jp = json_parse(dat);
			if (jp) {
				const JsonValue* delta = jp->find("delta");
				if (jp->at("type").as_string() == "message_delta" && delta) {
					std::string sr = delta->at("stop_reason").as_string();
					if (!sr.empty()) {
						StreamEvent de;
						de.type = StreamEvent::Type::Done;
						de.stop_reason = mapAnthropicStop(sr);
						on_event(de);
						stop_emitted = true;
					}
				}
			}
		}
		return;
	}

	// ── openai-completions ──────────────────────────────────────────────────
	if (model.api == "openai-completions") {
		std::string url = model.base_url;
		urlAppendPath(url, "/chat/completions");
		JsonValue bodyVal = buildOpenAIBody(model, context, options);
		std::string bodyStr = bodyVal.dump();
		std::vector<std::pair<std::string, std::string>> headers;
		headers.emplace_back("Authorization", "Bearer " + api_key);
		headers.emplace_back("content-type", "application/json");
		headers.emplace_back("accept", "application/json");

		on_event(StreamEvent{StreamEvent::Type::Start});

		SseDec d;
		std::string raw;
		// full response body kept for non-SSE error payloads (HTTP >= 400);
		// `raw` gets consumed by the SSE line splitter, leaving nothing to parse
		std::string raw_body;
		bool stop_emitted = false;
		bool consumer_stop = false;

		// forward an event to the consumer; a false return means "stop the stream"
		auto fire = [&](const StreamEvent& ev) {
			if (consumer_stop) return;
			if (!on_event(ev)) consumer_stop = true;
		};

		struct ToolAcc {
			int index = -1;
			std::string id;
			std::string name;  // arrives with the first chunk that carries it
			std::string args;  // pre-name argument chunks
			bool started = false;
		};
		std::vector<ToolAcc> toolAccs;

		auto on_chunk = [&](const char* data, size_t len) -> bool {
			if (consumer_stop) return false;
			raw.append(data, len);
			if (raw_body.size() < 65536) raw_body.append(data, len);
			std::string line;
			while (takeSseLine(raw, line)) {
				if (!line.empty()) {
					feedSse(d, line);
					continue;
				}
				if (d.event.empty() && d.data.empty()) continue;
				std::string dat = d.data;
				if (!dat.empty() && dat.back() == '\n') dat.pop_back();
				std::string ev = d.event;
				d.event.clear(); d.data.clear();

				if (ev == "error") {
					StreamEvent err;
					err.type = StreamEvent::Type::Error;
					err.text = "stream error: " + dat;
					err.error_message = "stream error: " + dat;
					fire(err);
					stop_emitted = true;
					return !consumer_stop;
				}
				if (ev != "message" && !ev.empty()) continue;

				auto jp = json_parse(dat);
				if (!jp) continue;
				const JsonValue& j = *jp;

				// usage — parse from any chunk that carries it. With
				// stream_options.include_usage the terminal usage-only chunk
				// (choices: []) arrives AFTER the finish chunk, so this must sit
				// before the choices check and must not be gated on stop_emitted:
				// otherwise usage is never seen and assistant messages end up with
				// no token counts at all. Field-presence checked throughout.
				if (const JsonValue* u = j.find("usage")) {
					if (u->is_object()) {
						StreamEvent ue;
						ue.type = StreamEvent::Type::Start;  // usage rides Start (same as Anthropic message_start)
						ue.usage.input = (int64_t)u->at("prompt_tokens").as_number();
						ue.usage.output = (int64_t)u->at("completion_tokens").as_number();
						// cached prompt tokens: OpenAI nests them in prompt_tokens_details,
						// DeepSeek sends prompt_cache_hit_tokens at top level
						if (u->find("prompt_cache_hit_tokens"))
							ue.usage.cache_read = (int64_t)u->at("prompt_cache_hit_tokens").as_number();
						else if (const JsonValue* ptd = u->find("prompt_tokens_details")) {
							if (ptd->is_object())
								ue.usage.cache_read = (int64_t)ptd->at("cached_tokens").as_number();
						}
						fire(ue);
					}
				}

				const JsonValue* choices_v = j.find("choices");
				if (!choices_v || !choices_v->is_array() || choices_v->size() == 0)
					continue;  // terminal usage-only chunk
				const JsonValue& choice = (*choices_v)[(size_t)0];
				const JsonValue* delta_v = choice.find("delta");
				if (!delta_v || !delta_v->is_object()) continue;
				const JsonValue& delta = *delta_v;

				// text delta
				if (delta.find("content")) {
					std::string txt = delta.at("content").as_string();
					if (!txt.empty()) {
						StreamEvent te;
						te.type = StreamEvent::Type::TextDelta;
						te.text = txt;
						fire(te);
					}
				}
				// reasoning / thinking delta (field name varies per provider)
				std::string rfield;
				if (delta.find("reasoning_content")) rfield = "reasoning_content";
				else if (delta.find("reasoning"))    rfield = "reasoning";
				if (!rfield.empty()) {
					std::string txt = delta.at(rfield).as_string();
					if (!txt.empty()) {
						StreamEvent te;
						te.type = StreamEvent::Type::ThinkingDelta;
						te.text = txt;
						fire(te);
					}
				}
				// tool call deltas
				if (const JsonValue* tcs_v = delta.find("tool_calls")) {
					if (tcs_v->is_array()) {
						const JsonValue& tcs = *tcs_v;
						for (size_t ti = 0; ti < tcs.size(); ti++) {
							const JsonValue& tc = tcs[ti];
							int idx = (int)tc.at("index").as_number();
							while ((int)toolAccs.size() <= idx) toolAccs.emplace_back();
							ToolAcc& acc = toolAccs[(size_t)idx];
							acc.index = idx;
							if (tc.find("id") && acc.id.empty())
								acc.id = tc.at("id").as_string();

							std::string fname;
							if (const JsonValue* fn = tc.find("function"))
								fname = fn->at("name").as_string();
							if (!fname.empty() && acc.name.empty())
								acc.name = fname;

							std::string arg_delta;
							if (const JsonValue* fn = tc.find("function")) {
								if (fn->find("arguments"))
									arg_delta = fn->at("arguments").as_string();
							}

							if (!arg_delta.empty()) {
								if (!acc.started) acc.args += arg_delta;
								else {
									StreamEvent te;
									te.type = StreamEvent::Type::ToolCallArgsDelta;
									te.tool_index = idx;
									te.args_delta = arg_delta;
									fire(te);
								}
							}

							// wait until id and name are both known before firing ToolCallStart
							if (!acc.started && !acc.name.empty() && !acc.id.empty()) {
								StreamEvent te;
								te.type = StreamEvent::Type::ToolCallStart;
								te.tool_index = idx;
								te.tool_id = acc.id;
								te.tool_name = acc.name;
								fire(te);
								acc.started = true;

								if (!acc.args.empty()) {
									StreamEvent ae;
									ae.type = StreamEvent::Type::ToolCallArgsDelta;
									ae.tool_index = idx;
									ae.args_delta = acc.args;
									fire(ae);
									acc.args.clear();
								}
							}
						}
					}
				}
				// finish reason -> Done. Delta chunks carry "finish_reason": null;
					// only a NON-EMPTY string value actually ends the stream (matching the
					// key alone fired Done on the first chunk with an empty reason, which
					// mapped to "endTurn" and swallowed the real "tool_calls" reason —
					// the agent then never executed the tool calls).
					if (choice.find("finish_reason") && !stop_emitted) {
						// drain any unfinished tool calls before emitting Done.
						// Condition: not started AND (name OR id is known).
						// We fire with whatever we have; agent.cpp will fill missing id.
						for (size_t ai = 0; ai < toolAccs.size(); ++ai) {
							ToolAcc& acc = toolAccs[ai];
							if (!acc.started && !acc.name.empty()) {
								StreamEvent te;
								te.type = StreamEvent::Type::ToolCallStart;
								te.tool_index = (int)ai;
								te.tool_id = acc.id;          // may be empty; agent will fill
								te.tool_name = acc.name;      // may be empty
								fire(te);
								acc.started = true;

								if (!acc.args.empty()) {
									StreamEvent ae;
									ae.type = StreamEvent::Type::ToolCallArgsDelta;
									ae.tool_index = (int)ai;
									ae.args_delta = acc.args;
									fire(ae);
									acc.args.clear();
								}
							}
						}
						std::string fr = choice.at("finish_reason").as_string();
						if (!fr.empty()) {
							StreamEvent de;
							de.type = StreamEvent::Type::Done;
							de.stop_reason = mapOpenAIStop(fr);
							if (const JsonValue* u = j.find("usage")) {
								if (u->is_object()) {
									de.usage.input  = (int64_t)u->at("prompt_tokens").as_number();
									de.usage.output = (int64_t)u->at("completion_tokens").as_number();
								}
							}
							fire(de);
							stop_emitted = true;
						}
					}
			}
			return !consumer_stop;
		};

		HttpStreamResult hres = http_post_stream(url, headers, bodyStr, on_chunk, 0, options.cancelled);
		if (!hres.ok) {
			if (!stop_emitted) {
				StreamEvent err;
				err.type = StreamEvent::Type::Error;
				err.text = hres.error;
				err.error_message = hres.error;
				on_event(err);
			}
			return;
		}
		if (hres.status >= 400) {
			// error payload is JSON, not SSE; surface it. Field-presence checked:
			// gateways wrap errors in arbitrary shapes
			std::string msg = "HTTP " + std::to_string(hres.status);
			auto parsed = json_parse(raw_body);
			if (parsed) {
				std::string em;
				if (const JsonValue* e = parsed->find("error"))
					em = e->at("message").as_string("");
				if (em.empty()) em = parsed->at("message").as_string("");
				if (!em.empty()) msg += ": " + em;
				else msg += ": " + parsed->dump().substr(0, 400);
			} else if (!raw_body.empty()) {
				msg += ": " + raw_body.substr(0, 300);
			}
			if (!stop_emitted) {
				StreamEvent err;
				err.type = StreamEvent::Type::Error;
				err.text = msg;
				err.error_message = msg;
				on_event(err);
			}
			return;
		}
		// No finish_reason emitted but stream ended normally: send Done(stop)
		if (!stop_emitted) {
			StreamEvent de;
			de.type = StreamEvent::Type::Done;
			de.stop_reason = "endTurn";
			on_event(de);
		}
		return;
	}

	// Unknown api
	on_event(StreamEvent{StreamEvent::Type::Start});
	StreamEvent err;
	err.type = StreamEvent::Type::Error;
	err.text = "unknown api: " + model.api;
	err.error_message = "unknown api: " + model.api;
	on_event(err);
}

}  // namespace phi
