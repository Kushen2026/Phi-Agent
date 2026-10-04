#include "tools/tools_subagent.hpp"
#include "tools/tools_helpers.hpp"

#include "util/i18n.hpp"

namespace phi {

namespace {

ToolResult run_subagent(const JsonValue& args, ToolContext& ctx) {
	const bool zh = lang_is_zh(ctx.language);
	SubagentRequest request;
	request.description = trim(arg_string(args, "description"));
	request.task = trim(arg_string(args, "prompt"));
	if (request.task.empty()) {
		return error_result(zh ? "prompt 不能为空" : "prompt must not be empty");
	}
	if (!ctx.launch_subagent) {
		return error_result(zh ? "当前环境不支持子代理" : "subagents are not available in this environment");
	}
	std::string id;
	try {
		// returns as soon as the subagent is running in the background
		id = ctx.launch_subagent(request);
	} catch (const std::string& e) {
		return error_result(e);
	} catch (const std::exception& e) {
		return error_result(e.what());
	}
	std::string description = request.description.empty()
		? (zh ? "未命名子代理" : "unnamed subagent")
		: request.description;
	return text_result(tr(ctx.language,
		"已启动子代理「" + description + "」（" + id + "），在后台运行，完成后自动返回结果。",
		"Started subagent \"" + description + "\" (" + id +
			") — running in the background; its result is returned automatically."));
}

}  // namespace

ToolDef build_agent_tool() {
	ToolDef t;
	t.name = "agent";
	t.description = "启动一个全新的子代理在后台运行，子代理的任务完成后会自动返回摘要";
	t.description_en =
	    "Launch a brand new subagent to run in the background, automatically returning a summary when the task is completed.";
	t.prompt_snippet = "把任务交给后台子代理并行处理";
	t.prompt_snippet_en = "Delegate tasks to background subagents";
	t.parameters_fn = [](const std::string& lang) {
		return schema_object({
			{"description", schema_string(lang, "任务的简短描述（3-5 个词）", "A short description of the task (3-5 words)")},
			{"prompt", schema_string(lang, "给子代理的任务", "The task for the subagent")},
		}, {"prompt"});
	};
	t.run = run_subagent;
	return t;
}

}  // namespace phi
