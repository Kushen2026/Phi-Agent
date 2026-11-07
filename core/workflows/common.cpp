#include "workflows/common.hpp"

#include <cmath>

namespace phi::media {
namespace workflows {

GraphNode gn(int id, const char* type) {
	GraphNode n;
	n.id = id;
	n.type = type;
	n.params = JsonValue::object();
	return n;
}

void put_str(JsonValue& p, const char* key, const std::string& v) {
	if (!v.empty()) p[key] = v;
}


// A LoRA chain is one node in front of the loaders (the same node a user would
// wire by hand). Returns 0 when there is nothing to apply, which keeps a plain
// graph exactly the graph it was before LoRAs existed.
//
// The strength is folded into the same node as a parallel list
// (`lora_strengths`), index-matched to `loras`: the node's own parameters are how
// a hand-wired graph sets a strength, the way ComfyUI's LoraLoader takes
// `strength_model` beside `lora_name`.
int add_lora_node(Graph& g, int id, const std::vector<LoraRef>& loras) {
	if (loras.empty()) return 0;
	GraphNode n = gn(id, "MediaLoraLoader");
	{
		JsonValue arr = JsonValue::array();
		JsonValue strengths = JsonValue::array();
		for (const LoraRef& r : loras) {
			arr.push_back(r.path);
			// Rounded on the way out: a strength is a `float` in the chain, and
			// widening it to a double would print `0.6000000238418579` for a value
			// the user typed as `0.6`. The widget's step is 0.01 and four decimals
			// is far finer than anything a run can distinguish, so this only
			// removes the widening noise.
			strengths.push_back(std::round((double)r.strength * 10000.0) / 10000.0);
		}
		n.params["loras"] = arr;
		n.params["lora_strengths"] = strengths;
	}
	g.nodes.push_back(std::move(n));
	return id;
}

void link(Graph& g, int from, const char* from_port, int to, const char* to_port) {
	GraphLink l;
	l.from = from;
	l.from_port = from_port;
	l.to = to;
	l.to_port = to_port;
	g.links.push_back(std::move(l));
}

// Wire a LoRA chain into a chain's two loaders: one MediaLoraLoader feeds both,
// so the DiT and the text encoder see the same chain (each applies it where it
// belongs). No loras, no node - a plain chain keeps the exact graph it had.
void apply_loras(Graph& g, const std::vector<LoraRef>& loras, int model, int clip) {
	if (!add_lora_node(g, 20, loras)) return;
	link(g, 20, "LORA", model, "lora");
	link(g, 20, "LORA", clip, "lora");
}

}  // namespace workflows
}  // namespace phi::media
