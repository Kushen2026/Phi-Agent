// A small, ComfyUI-style node graph for the media engine.
//
// WHY THIS EXISTS
// ---------------
// The two media chains (image_gen, video_gen) used to be monolithic: each stage
// — load the DiT, load the text encoder, load the VAEs, apply LoRAs, encode the
// prompt, plan a latent, run the sampler, decode the latent, mux the file — was
// only reachable *through the pipeline that owned it*. Adding a feature meant
// editing the whole chain, and a new backbone meant re-writing everything around
// its forward call.
//
// This module turns every one of those stages into an independently addressable
// **node** with typed sockets. A graph is a list of nodes plus the links between
// their ports; the executor walks it in dependency order and hands each node the
// values its inputs asked for. Adding a capability is then "write the one missing
// node and wire it in", not "rewrite the chain":
//
//   ModelLoader ─┐
//   ClipLoader ──┼─ TextEncode ─┐
//   LoraLoader ──┘              ├─ KSampler ─ VaeDecode ─ SaveImage
//   EmptyLatent ────────────────┘
//
// The framework here knows nothing about GPUs, DiTs or prompts: it is the socket
// type system, the registry, the graph model and the executor. The media wiring
// (which concrete classes back MODEL / CLIP / VAE, and what each node computes)
// lives under `core/nodes/`; the ready-made chains under `core/workflows/`.
//
// DESIGN NOTES
// ------------
//   * A port carries a `Value`. Object sockets (MODEL, CLIP, VAE, LATENT, ...)
//     hold a `shared_ptr<NodeData>`; primitive sockets (INT / FLOAT / BOOL /
//     STRING) hold a scalar. Type mismatches are caught before a node runs.
//   * The executor is a single-shot topological walk with per-run memoisation:
//     a node whose output several consumers read runs once. There is no lazy
//     re-evaluation and no cross-run cache in the framework — a persistent cache
//     (reusing loaded weights between runs) is the job of the node that owns the
//     resource, keyed on its parameters.
//   * Errors are values, not exceptions: `run_graph` never throws. A node that
//     cannot run reports why, named after its type and id.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "util/media_common.hpp"
#include "util/json.hpp"

namespace phi::media {

// ── socket types ───────────────────────────────────────────────────────────

// The value a port carries. Object sockets are the engine's module handles;
// the scalar sockets let a graph compute a parameter instead of hard-coding it.
enum class SocketType {
	None,
	Model,         // a DiT backbone (Qwen-Image / MiniMax H3) + its LoRA chain
	Clip,          // a text encoder + tokenizer (+ its LoRA chain)
	Vae,           // an image / video / audio VAE
	Lora,          // a LoRA chain (a set of delta files, applied at load time)
	Conditioning,  // an encoded prompt (positive / negative / references / tags)
	Latent,        // a latent tensor plus the geometry it was planned at
	Sigmas,        // a sigma schedule (image: one grid; video: video + audio)
	Image,         // an 8-bit RGB still
	Audio,         // interleaved float PCM
	Video,         // a decoded clip (planar frames + geometry)
	Int,
	Float,
	Bool,
	String,
	Any,           // accepts anything (used by generic sinks)
};

const char* socket_type_name(SocketType t);
SocketType socket_type_from_name(const std::string& name);

// ── node payloads ──────────────────────────────────────────────────────────

// The base of every object a socket can carry. Concrete payloads live in
// graph_media.hpp so this header stays free of the model classes.
class NodeData {
public:
	virtual ~NodeData() = default;
	virtual SocketType socket() const = 0;
	virtual const char* kind() const = 0;   // "MODEL", "LATENT", ...
};

// One value on a wire. Object sockets use `obj`; scalars use i/f/b/s. `type`
// is always authoritative — a node reads it (or the executor resolved it from
// the port) before touching a field.
struct Value {
	SocketType type = SocketType::None;
	std::shared_ptr<NodeData> obj;
	int64_t i = 0;
	double f = 0.0;
	bool b = false;
	std::string s;

	bool valid() const { return type != SocketType::None; }

	static Value of_object(std::shared_ptr<NodeData> d) {
		Value v;
		v.type = d ? d->socket() : SocketType::None;
		v.obj = std::move(d);
		return v;
	}
	static Value of_int(int64_t n) {
		Value v;
		v.type = SocketType::Int;
		v.i = n;
		return v;
	}
	static Value of_float(double x) {
		Value v;
		v.type = SocketType::Float;
		v.f = x;
		return v;
	}
	static Value of_bool(bool x) {
		Value v;
		v.type = SocketType::Bool;
		v.b = x;
		return v;
	}
	static Value of_string(std::string x) {
		Value v;
		v.type = SocketType::String;
		v.s = std::move(x);
		return v;
	}

	// Scalar reads with a fallback; used by nodes that accept a linked value.
	int64_t as_int(int64_t def) const { return type == SocketType::Int ? i : def; }
	double as_float(double def) const {
		if (type == SocketType::Float) return f;
		if (type == SocketType::Int) return (double)i;
		return def;
	}
	bool as_bool(bool def) const {
		if (type == SocketType::Bool) return b;
		if (type == SocketType::Int) return i != 0;
		return def;
	}
};

// ── node definition ────────────────────────────────────────────────────────

struct Port {
	std::string name;         // "MODEL", "model", "positive", ...
	SocketType type = SocketType::None;
	bool optional = false;    // an unlinked optional input arrives as a default
	std::string doc;          // one line for the UI / the node catalogue
};

// What a node's run() gets. `inputs` is parallel to NodeDef::inputs; an
// unlinked optional port is an invalid Value (caller checks `.valid()`).
struct GraphContext;

struct NodeDef {
	std::string type;         // stable id, e.g. "MediaSampler"
	std::string category;     // "loaders" | "conditioning" | "latent" | "sampling" |
	                          // "vae" | "image" | "audio" | "video" | "utils"
	std::string title;        // display name (zh)
	std::string title_en;
	std::string description;
	std::string description_en;
	std::vector<Port> inputs;
	std::vector<Port> outputs;
	// A JSON-Schema-ish object describing the node's inline parameters, so a UI
	// can render widgets and a model can discover what to fill in. Not enforced
	// by the executor; each run() reads its own params.
	JsonValue params;

	std::function<std::vector<Value>(GraphContext&, const JsonValue& params,
	                                 const std::vector<Value>& inputs)> run;
};

// ── registry ───────────────────────────────────────────────────────────────

class NodeRegistry {
public:
	void add(NodeDef def);
	const NodeDef* find(const std::string& type) const;
	// Every registered node, in registration order.
	std::vector<const NodeDef*> all() const;
	size_t size() const { return order_.size(); }

private:
	std::map<std::string, NodeDef> defs_;
	std::vector<std::string> order_;
};

// The media node catalogue lives in graph_nodes.hpp (`media_registry()`): the
// framework here stays free of the engine so its unit test links nothing else.

// ── the graph ──────────────────────────────────────────────────────────────

struct GraphLink {
	int from = 0;
	std::string from_port;
	int to = 0;
	std::string to_port;
};

struct GraphNode {
	int id = 0;
	std::string type;
	JsonValue params;   // inline parameters (widget values)
	// UI-only placement; ignored by the executor but carried through dump/parse
	// so a node editor round-trips its layout.
	double x = 0, y = 0;
};

struct GraphOutput {
	int node = 0;
	std::string port;
};

struct Graph {
	std::vector<GraphNode> nodes;
	std::vector<GraphLink> links;
	std::vector<GraphOutput> outputs;   // what the caller wants back
	std::string name;

	// JSON form. `from_json` validates structure (ids, links point at nodes that
	// exist, no duplicate node ids) but not node types — the executor resolves
	// those against a registry, so a graph may be parsed before its nodes are
	// registered.
	static std::optional<Graph> from_json(const JsonValue& v, std::string* error);
	JsonValue to_json() const;

	const GraphNode* find_node(int id) const;
};

// ── execution ──────────────────────────────────────────────────────────────

// Everything a node needs that is not one of its inputs. The framework only
// promises the strings and the two callbacks; the media nodes use `gpu` (a
// GpuCtx*) and `user` (a shared state bag, see graph_nodes.hpp).
struct GraphContext {
	std::string cwd;
	std::string agent_dir;
	std::string models_dir;
	std::string language = "zh";
	void* gpu = nullptr;                  // media::GpuCtx*, opaque here
	std::shared_ptr<void> user;           // engine-side state (model cache, ...)
	std::function<void(const std::string&)> progress;
	std::function<bool()> cancelled;
	// Optional per-sampler-step observation, called at the top of each step with
	// the step index. The framework itself does not use it; it exists so a
	// measurement harness (the media benchmark) or a UI can sample the machine
	// while the sampling loop runs, exactly as the pipelines' `on_step` hook did.
	std::function<void(int64_t)> on_step;

	bool is_cancelled() const { return cancelled && cancelled(); }
	void note(const std::string& text) const {
		if (progress) progress(text);
	}
};

struct NodeTrace {
	int id = 0;
	std::string type;
	double ms = 0.0;
	std::string error;
};

struct GraphResult {
	bool ok = false;
	std::string error;                       // first failure, human readable
	std::vector<Value> outputs;              // in Graph::outputs order
	// Every value the run produced, keyed "id:PORT", for debugging / a UI.
	std::map<std::string, Value> produced;
	std::vector<NodeTrace> trace;
};

// Runs `g` against `reg`. Never throws. On a node failure the result carries the
// failing node's type/id and its message.
GraphResult run_graph(const Graph& g, GraphContext& ctx, const NodeRegistry& reg);

// Structural validation against a registry, without executing anything: every
// node type exists, every link names a real output/input port of the right
// socket type, no input receives two links, and every non-optional input is
// connected. Returns "" when the graph is runnable, otherwise the first problem
// (named by node id and port). This is what a node editor / agent calls before
// an expensive run; `run_graph` re-checks the same things as it walks.
std::string validate_graph(const Graph& g, const NodeRegistry& reg);

// ── parameter helpers (shared by the node implementations) ─────────────────

std::string param_string(const JsonValue& p, const char* key, const std::string& def = "");
int64_t param_int(const JsonValue& p, const char* key, int64_t def = 0);
double param_number(const JsonValue& p, const char* key, double def = 0.0);
bool param_bool(const JsonValue& p, const char* key, bool def = false);
// A list of strings from a JSON array (strings, or objects with a "value" key —
// the shape a node editor stores a loaded file list in).
std::vector<std::string> param_string_list(const JsonValue& p, const char* key);

}  // namespace phi::media
