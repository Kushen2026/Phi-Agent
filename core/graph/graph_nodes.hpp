// Registration entry point for the media node catalogue.
//
// The framework (`graph.hpp`) is registered-agnostic: it never links the engine.
// This is the one symbol the rest of the app calls to populate a registry with
// every media node (loaders, conditioning, latents, samplers, VAEs, IO).
//
// The catalogue is one file per node under `core/nodes/` (plus `service.hpp`
// for the shared helpers and `registry.cpp` for this entry point), so a node is
// added or edited in exactly one place.
#pragma once

#include "graph/graph.hpp"

namespace phi::media {

// The process-wide registry with every media node registered (built once). This
// is the registry the media tools, the workflow tool and the node-graph runner
// execute against. graph.hpp's registry type is engine-free; this is the media
// catalogue's instance of it. Safe to call from any thread (built on first use).
const NodeRegistry& media_registry();

// Adds every media node to `reg`. Registering twice is harmless (a type is
// replaced in place, so the registry keeps its order).
void register_media_nodes(NodeRegistry& reg);

}  // namespace phi::media
