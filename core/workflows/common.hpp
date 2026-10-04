#pragma once

// The bits every workflow builder shares: how a node and a link are spelled,
// and the LoRA-chain wiring (one node in front of both loaders, or nothing at
// all so a plain chain keeps the exact graph it had).

#include <string>
#include <vector>

#include "graph/graph.hpp"
#include "util/json.hpp"

namespace phi::media {
namespace workflows {

GraphNode gn(int id, const char* type);
void put_str(JsonValue& p, const char* key, const std::string& v);
int add_lora_node(Graph& g, int id, const std::vector<std::string>& loras);
void link(Graph& g, int from, const char* from_port, int to, const char* to_port);
void apply_loras(Graph& g, const std::vector<std::string>& loras, int model, int clip);

}  // namespace workflows
}  // namespace phi::media
