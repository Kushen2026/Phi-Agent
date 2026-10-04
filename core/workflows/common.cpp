#include "workflows/common.hpp"

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
int add_lora_node(Graph& g, int id, const std::vector<std::string>& loras) {
	if (loras.empty()) return 0;
	GraphNode n = gn(id, "MediaLoraLoader");
	{
		JsonValue arr = JsonValue::array();
		for (const std::string& s : loras) arr.push_back(s);
		n.params["loras"] = arr;
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
void apply_loras(Graph& g, const std::vector<std::string>& loras, int model, int clip) {
	if (!add_lora_node(g, 20, loras)) return;
	link(g, 20, "LORA", model, "lora");
	link(g, 20, "LORA", clip, "lora");
}

}  // namespace workflows
}  // namespace phi::media
