// `--media-tool <name> '<json args>'` — invoke one of the two media tools the way
// the agent does, and print its result text.
//
// This exists because the media tools are the *only* surface a caller has for the
// two chains (core/tools/tools_media.cpp), and until now the only way to run one
// was through a live agent session. That left the tool's own layer - the argument
// schema, the path/limit checks, the report and the graph it builds - reachable
// only from the app, while `--media-bench` exercised the *graphs* directly. The
// two chains drifted apart exactly there (the bench was passing while a tool run
// would have failed), so this is the missing seam: same ToolDef::run, same
// ToolContext, no model in the loop.
//
//   phi.exe --media-tool image_generate "{\"prompt\":\"...\",\"width\":1024,...}"
//   phi.exe --media-tool video_generate "{...}"
// The tool's own schema is printed with `--media-tool <name> schema`.
#include "agent/agent.hpp"
#include "tools/tools_media.hpp"
#include "graph/graph.hpp"
#include "models/media_models.hpp"
#include "graph/graph_nodes.hpp"
#include "graph/graph_workflows.hpp"
#include "util/base.hpp"
#include "util/json.hpp"

#include <cstdio>
#include <string>

namespace phi {
namespace {

// `--media-tool <name> graph`: build the chain's graph and validate it against
// the node registry, without a GPU, a checkpoint or a model in the loop. This is
// the check that the *wiring* is sound - every node type registered, every link
// naming a real output/input port of a compatible socket type, every required
// input fed - and it is the one that catches a node renamed in core/nodes/
// without its chain in core/workflows/ following. It is also what makes the
// "one node drives any chain" claim checkable from a shell.
void print_tool_graph(const std::string& name) {
	media::Graph g;
	if (name == "image_generate") {
		media::ImageWorkflowParams p;
		p.prompt = "x";
		p.output = "out.png";
		g = media::build_image_workflow(p);
	} else if (name == "video_generate") {
		media::VideoWorkflowParams p;
		p.prompt = "x";
		p.output = "out.mp4";
		g = media::build_video_workflow(p);
	} else if (name == "music_generate") {
		media::MusicWorkflowParams p;
		p.tags = "x";
		p.output = "out.wav";
		g = media::build_music_workflow(p);
	} else if (name == "tts_speak") {
		media::TtsWorkflowParams p;
		p.text = "x";
		p.instruction = "x";
		p.output = "out.wav";
		g = media::build_tts_workflow(p);
	} else {
		fprintf(stderr, "no graph for media tool: %s\n", name.c_str());
		return;
	}
	const media::NodeRegistry& reg = media::media_registry();
	printf("graph %s (%zu nodes, %zu links)\n", g.name.c_str(), g.nodes.size(), g.links.size());
	for (const media::GraphNode& n : g.nodes) {
		const media::NodeDef* d = reg.find(n.type);
		printf("  %3d  %-24s %s\n", n.id, n.type.c_str(), d ? "" : "<NOT REGISTERED>");
	}
	const std::string err = media::validate_graph(g, reg);
	if (err.empty()) {
		printf("OK: the graph validates against the %zu-node registry\n", reg.size());
	} else {
		printf("INVALID: %s\n", err.c_str());
	}
}

// `--media-tool <name> resolve`: print which file each role resolves to for the
// current settings.json, and whether it exists. This is the "which config will
// this chain actually read?" check - the per-model JSON roles are only useful if
// you can see what they landed on, and a missing tokenizer config is a silent
// wrong answer otherwise.
void print_tool_resolve(const std::string& base_dir) {
	const std::string agent_dir = path_join(base_dir, "data");
	const std::string models_dir = path_join(base_dir, "models");
	JsonValue media = JsonValue::object();
	auto raw = read_file(path_join(agent_dir, "settings.json"));
	if (raw) {
		auto parsed = json_parse(*raw);
		const JsonValue* tools = parsed && parsed->is_object() ? parsed->find("tools") : nullptr;
		const JsonValue* m = tools && tools->is_object() ? tools->find("media") : nullptr;
		if (m && m->is_object()) media = *m;
	}
	media::MediaModelSpec spec = media::resolve_media_spec(media, models_dir);
	struct Row { const char* role; const std::string* path; };
	const Row rows[] = {
	    {"image_dit", &spec.image_dit},
	    {"image_te", &spec.image_te},
	    {"image_vae", &spec.image_vae},
	    {"image_tokenizer_vocab", &spec.image_tokenizer_vocab},
	    {"image_tokenizer_merges", &spec.image_tokenizer_merges},
	    {"image_tokenizer_config", &spec.image_tokenizer_config},
	    {"video_dit", &spec.video_dit},
	    {"video_te", &spec.video_te},
	    {"video_vae", &spec.video_vae},
	    {"video_avae", &spec.video_avae},
	    {"video_tokenizer_vocab", &spec.video_tokenizer_vocab},
	    {"video_tokenizer_merges", &spec.video_tokenizer_merges},
	    {"video_tokenizer_config", &spec.video_tokenizer_config},
	    {"music_dit", &spec.music_dit},
	    {"music_te", &spec.music_te},
	    {"music_lm", &spec.music_lm},
	    {"music_vae", &spec.music_vae},
	    {"music_tokenizer_vocab", &spec.music_tokenizer_vocab},
	    {"music_tokenizer_merges", &spec.music_tokenizer_merges},
	    {"music_tokenizer_config", &spec.music_tokenizer_config},
	    {"tts_model", &spec.tts_model},
	    {"tts_codec", &spec.tts_codec},
	    {"tts_tokenizer", &spec.tts_tokenizer},
	    {"tts_config", &spec.tts_config},
	    {"tts_codec_config", &spec.tts_codec_config},
	};
	auto trim_base = [&](const std::string& p) {
		const std::string pre = models_dir + "/";
		return p.rfind(pre, 0) == 0 ? p.substr(pre.size()) : p;
	};
	int missing = 0;
	printf("models dir: %s\n", models_dir.c_str());
	for (const Row& r : rows) {
		const bool ok = !r.path->empty() && path_exists(*r.path);
		if (!ok) missing++;
		printf("  %-24s %-58s %s\n", r.role, r.path->empty() ? "(none)" : trim_base(*r.path).c_str(),
		       ok ? "ok" : "MISSING");
	}
	printf("%s\n", missing ? "some roles are missing a file" : "every role resolves to a file");
}

void print_tool_schema(const ToolDef& t) {
	JsonValue schema = JsonValue::object();
	schema["name"] = t.name;
	schema["description"] = t.description;
	schema["parameters"] = t.parameters_fn ? t.parameters_fn("en") : JsonValue::object();
	(void)0;
	printf("%s\n", schema.dump(2).c_str());
}

}  // namespace

int media_tool_main(const std::vector<std::string>& args, const std::string& base_dir) {
	// The catalogue owns the ToolDef objects, so keep the vector alive: a pointer
	// into the temporary a range-for makes would dangle the moment the loop ends.
	const std::vector<ToolDef> tools = build_media_tools();
	if (args.empty()) {
		fprintf(stderr, "usage: phi.exe --media-tool <image_generate|video_generate|music_generate|tts_speak> "
		                "[<json args>|schema]\n");
		for (const ToolDef& t : tools) fprintf(stderr, "       %s\n", t.name.c_str());
		return 2;
	}
	const std::string name = args[0];
	const ToolDef* found = nullptr;
	for (const ToolDef& t : tools)
		if (t.name == name) found = &t;
	if (!found) {
		fprintf(stderr, "unknown media tool: %s\n", name.c_str());
		return 2;
	}
	if (args.size() > 1 && args[1] == "schema") {
		print_tool_schema(*found);
		return 0;
	}
	if (args.size() > 1 && args[1] == "graph") {
		print_tool_graph(name);
		return 0;
	}
	if (args.size() > 1 && args[1] == "resolve") {
		print_tool_resolve(base_dir);
		return 0;
	}

	JsonValue in;
	if (args.size() > 1) {
		auto parsed = json_parse(args[1]);
		if (!parsed || !parsed->is_object()) {
			fprintf(stderr, "the arguments must be one JSON object\n");
			return 2;
		}
		in = *parsed;
	} else {
		in = JsonValue::object();
	}

	ToolContext ctx;
	ctx.cwd = base_dir;
	ctx.agent_dir = path_join(base_dir, "data");
	ctx.bin_dir = path_join(base_dir, "bin");
	ctx.language = "zh";
	ctx.current_tool = name;
	ctx.on_update = [](const std::string& s) {
		printf("  .. %s\n", s.c_str());
		fflush(stdout);
	};
	ctx.cancelled = [] { return false; };

	ToolResult r;
	try {
		r = found->run(in, ctx);
	} catch (const std::exception& ex) {
		fprintf(stderr, "tool threw: %s\n", ex.what());
		return 1;
	}
	for (const ContentPart& part : r.content) {
		if (part.type == ContentPart::Type::Text) printf("%s\n", part.text.c_str());
	}
	fflush(stdout);
	return r.is_error ? 1 : 0;
}

}  // namespace phi
