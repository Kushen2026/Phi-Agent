// The two media tools, and only two: `image_generate` (Qwen-Image-2.1 text to image /
// reference edit, PNG) and `video_generate` (MiniMax H3 ref2va, mp4).
//
// The node graph underneath them (`core/graph/*`) is the engine's internal
// structure, not a tool surface: the shipped chains are two constructors
// (`build_image_workflow` / `build_video_workflow`) and a new capability is a new
// wiring, not a new tool. Everything a caller needs to reach the graph - loading a
// model of any precision, encoding a prompt, sampling, decoding, writing - is a
// parameter of these two tools or a setting.
//
// The engine is process-wide and lazily created: the image checkpoint is 6.8 GB of
// int8 and its text encoder another 8.7 GB, so opening it per call would dominate
// the runtime, and opening it at startup would make every session pay for a tool
// that may never be used. The first `image_generate` call pays for it; later calls
// reuse the loaded weights.
//
// Everything runs on the caller's thread (Agent::execute_tool already runs tool
// bodies off the UI thread), and `ctx.cancelled` is polled between sampler steps so
// a 30-second run can be stopped without touching the process.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "agent/agent.hpp"

namespace phi::media {
struct GpuCtx;   // core/models/model_common.hpp
}

namespace phi {

// image_generate + video_generate
std::vector<ToolDef> build_media_tools();

// `phi.exe --media-tool <name> [<json args>|schema]`: runs one media tool exactly
// as the agent would (same ToolDef::run and ToolContext) and prints its result.
// Returns the process exit code. Defined in tools_media_cli.cpp.
int media_tool_main(const std::vector<std::string>& args, const std::string& base_dir);

// Runs `fn` with the process-wide media engine open and locked (one generation
// at a time), handing it the GPU context, the models directory and a persistent
// node-graph cache slot. Returns "" on success, otherwise a user-facing message
// (already localised). Both generators go through it; see tools_media.cpp for
// the phase-boundary handling it wraps.
std::string with_media_engine(
    const std::string& agent_dir, const std::string& cwd, const std::string& language,
    const std::function<std::string(media::GpuCtx*, const std::string&, std::shared_ptr<void>&)>& fn);

}  // namespace phi
