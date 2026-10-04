#include "graph/graph.hpp"

#include <algorithm>
#include <set>

namespace phi::media {

// ── socket types ───────────────────────────────────────────────────────────

const char* socket_type_name(SocketType t) {
	switch (t) {
		case SocketType::None: return "NONE";
		case SocketType::Model: return "MODEL";
		case SocketType::Clip: return "CLIP";
		case SocketType::Vae: return "VAE";
		case SocketType::Lora: return "LORA";
		case SocketType::Conditioning: return "CONDITIONING";
		case SocketType::Latent: return "LATENT";
		case SocketType::Sigmas: return "SIGMAS";
		case SocketType::Image: return "IMAGE";
		case SocketType::Audio: return "AUDIO";
		case SocketType::Video: return "VIDEO";
		case SocketType::Int: return "INT";
		case SocketType::Float: return "FLOAT";
		case SocketType::Bool: return "BOOL";
		case SocketType::String: return "STRING";
		case SocketType::Any: return "ANY";
	}
	return "NONE";
}

SocketType socket_type_from_name(const std::string& name) {
	static const std::pair<const char*, SocketType> kTable[] = {
	    {"MODEL", SocketType::Model},   {"CLIP", SocketType::Clip},
	    {"VAE", SocketType::Vae},       {"LORA", SocketType::Lora},
	    {"CONDITIONING", SocketType::Conditioning}, {"LATENT", SocketType::Latent},
	    {"SIGMAS", SocketType::Sigmas}, {"IMAGE", SocketType::Image},
	    {"AUDIO", SocketType::Audio},   {"VIDEO", SocketType::Video},
	    {"INT", SocketType::Int},       {"FLOAT", SocketType::Float},
	    {"BOOL", SocketType::Bool},     {"STRING", SocketType::String},
	    {"ANY", SocketType::Any},
	};
	for (const auto& e : kTable) {
		if (name == e.first) return e.second;
	}
	return SocketType::None;
}

// ── registry ───────────────────────────────────────────────────────────────

void NodeRegistry::add(NodeDef def) {
	if (def.type.empty()) return;
	if (def.params.is_null()) def.params = JsonValue::object();
	if (defs_.find(def.type) == defs_.end()) order_.push_back(def.type);
	defs_[def.type] = std::move(def);
}

const NodeDef* NodeRegistry::find(const std::string& type) const {
	auto it = defs_.find(type);
	return it == defs_.end() ? nullptr : &it->second;
}

std::vector<const NodeDef*> NodeRegistry::all() const {
	std::vector<const NodeDef*> out;
	out.reserve(order_.size());
	for (const std::string& t : order_) {
		auto it = defs_.find(t);
		if (it != defs_.end()) out.push_back(&it->second);
	}
	return out;
}

// ── graph (de)serialisation ────────────────────────────────────────────────

const GraphNode* Graph::find_node(int id) const {
	for (const GraphNode& n : nodes)
		if (n.id == id) return &n;
	return nullptr;
}

std::optional<Graph> Graph::from_json(const JsonValue& v, std::string* error) {
	auto fail = [&](const std::string& msg) -> std::optional<Graph> {
		if (error) *error = msg;
		return std::nullopt;
	};
	if (!v.is_object()) return fail("graph must be a JSON object");

	Graph g;
	if (const JsonValue* name = v.find("name"); name && name->is_string())
		g.name = name->as_string();

	const JsonValue* nodes = v.find("nodes");
	if (!nodes || !nodes->is_array()) return fail("graph.nodes must be an array");
	std::set<int> ids;
	for (const JsonValue& nj : nodes->items()) {
		if (!nj.is_object()) return fail("each graph node must be an object");
		GraphNode n;
		const JsonValue* id = nj.find("id");
		if (!id || !id->is_number()) return fail("graph node is missing a numeric id");
		n.id = (int)id->as_int(0);
		if (ids.count(n.id)) return fail("duplicate node id " + std::to_string(n.id));
		ids.insert(n.id);
		const JsonValue* type = nj.find("type");
		if (!type || !type->is_string() || type->as_string().empty())
			return fail("node " + std::to_string(n.id) + " is missing a type");
		n.type = type->as_string();
		if (const JsonValue* p = nj.find("params")) n.params = *p;
		if (const JsonValue* x = nj.find("x"); x && x->is_number()) n.x = x->as_number();
		if (const JsonValue* y = nj.find("y"); y && y->is_number()) n.y = y->as_number();
		g.nodes.push_back(std::move(n));
	}

	const JsonValue* links = v.find("links");
	if (links && links->is_array()) {
		for (const JsonValue& lj : links->items()) {
			if (!lj.is_object()) return fail("each link must be an object");
			GraphLink l;
			const JsonValue* from = lj.find("from");
			const JsonValue* to = lj.find("to");
			if (!from || !to) return fail("a link is missing 'from' or 'to'");
			l.from = (int)from->as_int(0);
			l.to = (int)to->as_int(0);
			const JsonValue* fp = lj.find("from_port");
			const JsonValue* tp = lj.find("to_port");
			l.from_port = fp && fp->is_string() ? fp->as_string() : std::string();
			l.to_port = tp && tp->is_string() ? tp->as_string() : std::string();
			if (!ids.count(l.from) || !ids.count(l.to))
				return fail("link " + std::to_string(l.from) + "->" + std::to_string(l.to) +
				            " points at a node that does not exist");
			g.links.push_back(std::move(l));
		}
	} else if (links) {
		return fail("graph.links must be an array");
	}

	const JsonValue* outs = v.find("outputs");
	if (outs && outs->is_array()) {
		for (const JsonValue& oj : outs->items()) {
			if (oj.is_string()) {
				// "12:IMAGE" shorthand
				std::string s = oj.as_string();
				size_t colon = s.find(':');
				if (colon == std::string::npos) return fail("output string must be 'id:PORT'");
				GraphOutput o;
				o.node = atoi(s.substr(0, colon).c_str());
				o.port = s.substr(colon + 1);
				if (!ids.count(o.node)) return fail("output points at an unknown node");
				g.outputs.push_back(std::move(o));
			} else if (oj.is_object()) {
				GraphOutput o;
				const JsonValue* n = oj.find("node");
				const JsonValue* p = oj.find("port");
				if (!n || !p) return fail("an output needs 'node' and 'port'");
				o.node = (int)n->as_int(0);
				o.port = p->as_string();
				if (!ids.count(o.node)) return fail("output points at an unknown node");
				g.outputs.push_back(std::move(o));
			} else {
				return fail("each output must be a string or an object");
			}
		}
	}
	return g;
}

JsonValue Graph::to_json() const {
	JsonValue v = JsonValue::object();
	if (!name.empty()) v["name"] = name;
	JsonValue ns = JsonValue::array();
	for (const GraphNode& n : nodes) {
		JsonValue j = JsonValue::object();
		j["id"] = (int64_t)n.id;
		j["type"] = n.type;
		j["params"] = n.params.is_null() ? JsonValue::object() : n.params;
		if (n.x != 0 || n.y != 0) {
			j["x"] = n.x;
			j["y"] = n.y;
		}
		ns.push_back(std::move(j));
	}
	v["nodes"] = std::move(ns);
	JsonValue ls = JsonValue::array();
	for (const GraphLink& l : links) {
		JsonValue j = JsonValue::object();
		j["from"] = (int64_t)l.from;
		j["from_port"] = l.from_port;
		j["to"] = (int64_t)l.to;
		j["to_port"] = l.to_port;
		ls.push_back(std::move(j));
	}
	v["links"] = std::move(ls);
	JsonValue os = JsonValue::array();
	for (const GraphOutput& o : outputs) {
		JsonValue j = JsonValue::object();
		j["node"] = (int64_t)o.node;
		j["port"] = o.port;
		os.push_back(std::move(j));
	}
	v["outputs"] = std::move(os);
	return v;
}

// ── the executor ───────────────────────────────────────────────────────────

namespace {

// A port is compatible when the types match, or either side is ANY/None.
bool type_ok(SocketType want, SocketType got) {
	if (want == SocketType::Any || got == SocketType::Any) return true;
	return want == got;
}

std::string key_of(int id, const std::string& port) {
	return std::to_string(id) + ":" + port;
}

}  // namespace

GraphResult run_graph(const Graph& g, GraphContext& ctx, const NodeRegistry& reg) {
	GraphResult res;

	// ── resolve node definitions and validate up front ─────────────────────
	std::map<int, const NodeDef*> defs;
	std::map<int, const GraphNode*> nodes;
	for (const GraphNode& n : g.nodes) {
		nodes[n.id] = &n;
		const NodeDef* def = reg.find(n.type);
		if (!def) {
			res.error = "unknown node type '" + n.type + "' (id " + std::to_string(n.id) + ")";
			return res;
		}
		defs[n.id] = def;
	}

	// incoming[to_id][to_port] = (from_id, from_port); also a fan-in check.
	//
	// Both ends of a link are checked against the definitions, because a typo in a
	// port name is *silent* otherwise: an unknown destination port is never read,
	// so an optional input stays unconnected and the chain runs with a piece
	// missing. Every workflow here names its ports as string literals, which is
	// exactly where that kind of typo lives.
	std::map<int, std::map<std::string, std::pair<int, std::string>>> incoming;
	for (const GraphLink& l : g.links) {
		auto to = nodes.find(l.to);
		if (to == nodes.end()) {
			res.error = "link destination node " + std::to_string(l.to) + " does not exist";
			return res;
		}
		{
			bool found = false;
			for (const Port& p : defs[l.from]->outputs)
				if (p.name == l.from_port) found = true;
			if (!found) {
				res.error = "node " + std::to_string(l.from) + " (" + defs[l.from]->type +
				            ") has no output '" + l.from_port + "'";
				return res;
			}
		}
		{
			bool found = false;
			for (const Port& p : defs[l.to]->inputs)
				if (p.name == l.to_port) found = true;
			if (!found) {
				res.error = "node " + std::to_string(l.to) + " (" + defs[l.to]->type +
				            ") has no input '" + l.to_port + "'";
				return res;
			}
		}
		auto& slot = incoming[l.to][l.to_port];
		if (!slot.second.empty()) {
			res.error = "node " + std::to_string(l.to) + " input '" + l.to_port +
			            "' has more than one link";
			return res;
		}
		slot = {l.from, l.from_port};
	}

	// ── topological order (Kahn) ───────────────────────────────────────────
	std::map<int, int> indegree;
	std::map<int, std::vector<int>> edges;
	for (const GraphNode& n : g.nodes) indegree[n.id] = 0;
	for (const GraphLink& l : g.links) {
		edges[l.from].push_back(l.to);
		indegree[l.to] += 1;
	}
	std::vector<int> ready;
	for (const GraphNode& n : g.nodes)
		if (indegree[n.id] == 0) ready.push_back(n.id);
	// Deterministic: process by id.
	std::sort(ready.begin(), ready.end());
	std::vector<int> order;
	while (!ready.empty()) {
		int id = ready.front();
		ready.erase(ready.begin());
		order.push_back(id);
		std::vector<int> next;
		for (int to : edges[id]) {
			if (--indegree[to] == 0) next.push_back(to);
		}
		std::sort(next.begin(), next.end());
		for (int n : next) ready.push_back(n);
	}
	if (order.size() != g.nodes.size()) {
		res.error = "graph has a cycle";
		return res;
	}

	// ── walk ───────────────────────────────────────────────────────────────
	std::map<std::string, Value> produced;
	auto exec_start = now_ms();

	for (int id : order) {
		const GraphNode& node = *nodes[id];
		const NodeDef& def = *defs[id];
		ctx.note(ctx.language == "en" ? ("node " + def.title_en + "…")
		                                 : ("节点 " + def.title + "…"));

		// Gather inputs in the definition's own order.
		std::vector<Value> inputs;
		inputs.reserve(def.inputs.size());
		bool missing = false;
		std::string why;
		for (const Port& p : def.inputs) {
			Value v;
			auto it = incoming[id].find(p.name);
			if (it != incoming[id].end()) {
				const auto& src = it->second;
				const std::string k = key_of(src.first, src.second);
				auto pit = produced.find(k);
				if (pit == produced.end()) {
					why = "input '" + p.name + "' is linked from " + k +
					      " which produced no value";
					missing = true;
					break;
				}
				v = pit->second;
				if (!type_ok(p.type, v.type)) {
					why = "input '" + p.name + "' expects " + socket_type_name(p.type) +
					      " but " + k + " carries " + socket_type_name(v.type);
					missing = true;
					break;
				}
			} else if (!p.optional) {
				why = "input '" + p.name + "' (" + socket_type_name(p.type) + ") is not connected";
				missing = true;
				break;
			}
			inputs.push_back(std::move(v));
		}
		if (missing) {
			res.error = "node " + std::to_string(id) + " (" + def.type + "): " + why;
			NodeTrace t;
			t.id = id;
			t.type = def.type;
			t.error = why;
			res.trace.push_back(std::move(t));
			return res;
		}

		if (ctx.is_cancelled()) {
			res.error = "cancelled before node " + std::to_string(id);
			return res;
		}

		NodeTrace t;
		t.id = id;
		t.type = def.type;
		double t0 = now_ms();
		std::vector<Value> outs;
		try {
			outs = def.run(ctx, node.params, inputs);
		} catch (const std::exception& ex) {
			t.ms = now_ms() - t0;
			t.error = ex.what();
			res.trace.push_back(t);
			res.error = "node " + std::to_string(id) + " (" + def.type + ") failed: " + ex.what();
			return res;
		}
		t.ms = now_ms() - t0;
		res.trace.push_back(std::move(t));
		(void)exec_start;

		if (outs.size() != def.outputs.size()) {
			res.error = "node " + std::to_string(id) + " (" + def.type + ") returned " +
			            std::to_string(outs.size()) + " values, expected " +
			            std::to_string(def.outputs.size());
			return res;
		}
		for (size_t i = 0; i < outs.size(); i++) {
			if (!outs[i].valid()) {
				// A node may legitimately produce an empty optional; expose it as
				// a typed-but-empty value so consumers can test `.valid()`.
				outs[i].type = def.outputs[i].type;
			}
			produced[key_of(id, def.outputs[i].name)] = outs[i];
		}
	}

	res.produced = produced;

	// ── requested outputs ──────────────────────────────────────────────────
	for (const GraphOutput& o : g.outputs) {
		const std::string k = key_of(o.node, o.port);
		auto it = produced.find(k);
		if (it == produced.end()) {
			res.error = "requested output " + k + " was not produced";
			return res;
		}
		res.outputs.push_back(it->second);
	}
	// No explicit outputs: hand back every sink (a node nothing links from),
	// in id order — the friendly default for a "run the whole graph" call.
	if (g.outputs.empty()) {
		std::set<int> has_consumer;
		for (const GraphLink& l : g.links) has_consumer.insert(l.from);
		for (int id : order) {
			if (has_consumer.count(id)) continue;
			for (const Port& p : defs[id]->outputs) {
				auto it = produced.find(key_of(id, p.name));
				if (it != produced.end()) res.outputs.push_back(it->second);
			}
		}
	}

	res.ok = true;
	return res;
}

std::string validate_graph(const Graph& g, const NodeRegistry& reg) {
	std::map<int, const NodeDef*> defs;
	std::map<int, const GraphNode*> nodes;
	for (const GraphNode& n : g.nodes) {
		nodes[n.id] = &n;
		const NodeDef* def = reg.find(n.type);
		if (!def) return "unknown node type '" + n.type + "' (id " + std::to_string(n.id) + ")";
		defs[n.id] = def;
	}
	std::map<int, std::map<std::string, std::string>> filled;
	for (const GraphLink& l : g.links) {
		auto fn = nodes.find(l.from);
		auto tn = nodes.find(l.to);
		if (fn == nodes.end()) return "link source node " + std::to_string(l.from) + " does not exist";
		if (tn == nodes.end()) return "link target node " + std::to_string(l.to) + " does not exist";
		const NodeDef& f = *defs[l.from];
		const NodeDef& t = *defs[l.to];
		const Port* op = nullptr;
		for (const Port& p : f.outputs)
			if (p.name == l.from_port) op = &p;
		if (!op)
			return "node " + std::to_string(l.from) + " (" + f.type + ") has no output '" +
			       l.from_port + "'";
		const Port* ip = nullptr;
		for (const Port& p : t.inputs)
			if (p.name == l.to_port) ip = &p;
		if (!ip)
			return "node " + std::to_string(l.to) + " (" + t.type + ") has no input '" + l.to_port +
			       "'";
		if (!type_ok(ip->type, op->type))
			return "link " + std::to_string(l.from) + ":" + l.from_port + " -> " +
			       std::to_string(l.to) + ":" + l.to_port + " type mismatch (" +
			       socket_type_name(op->type) + " into " + socket_type_name(ip->type) + ")";
		auto& slot = filled[l.to][l.to_port];
		if (!slot.empty())
			return "node " + std::to_string(l.to) + " input '" + l.to_port + "' has more than one link";
		slot = "1";
	}
	for (const GraphNode& n : g.nodes) {
		const NodeDef& d = *defs[n.id];
		for (const Port& p : d.inputs) {
			if (p.optional) continue;
			if (filled[n.id].find(p.name) == filled[n.id].end())
				return "node " + std::to_string(n.id) + " (" + d.type + ") input '" + p.name +
				       "' (" + socket_type_name(p.type) + ") is not connected";
		}
	}
	return std::string();
}

// ── parameter helpers ──────────────────────────────────────────────────────

std::string param_string(const JsonValue& p, const char* key, const std::string& def) {
	const JsonValue* v = p.find(key);
	if (!v) return def;
	if (v->is_string()) return v->as_string();
	return def;
}

int64_t param_int(const JsonValue& p, const char* key, int64_t def) {
	const JsonValue* v = p.find(key);
	if (!v || !v->is_number()) return def;
	return v->as_int(def);
}

double param_number(const JsonValue& p, const char* key, double def) {
	const JsonValue* v = p.find(key);
	if (!v || !v->is_number()) return def;
	return v->as_number(def);
}

bool param_bool(const JsonValue& p, const char* key, bool def) {
	const JsonValue* v = p.find(key);
	if (!v) return def;
	if (v->is_bool()) return v->as_bool(def);
	if (v->is_number()) return v->as_int(0) != 0;
	return def;
}

std::vector<std::string> param_string_list(const JsonValue& p, const char* key) {
	std::vector<std::string> out;
	const JsonValue* v = p.find(key);
	if (!v) return out;
	if (v->is_string()) {
		if (!v->as_string().empty()) out.push_back(v->as_string());
		return out;
	}
	if (!v->is_array()) return out;
	for (const JsonValue& e : v->items()) {
		if (e.is_string()) {
			if (!e.as_string().empty()) out.push_back(e.as_string());
		} else if (e.is_object()) {
			const JsonValue* val = e.find("value");
			if (val && val->is_string() && !val->as_string().empty()) out.push_back(val->as_string());
		}
	}
	return out;
}

}  // namespace phi::media
