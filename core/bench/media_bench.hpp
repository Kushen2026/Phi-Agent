// `phi.exe --media-bench`: the engine's acceptance instrument.
//
// It prints, in one pass: the CUDA adapter + NVRTC environment, the VRAM budget
// the scheduler will be given, a real weight-streaming throughput measurement
// against the on-disk checkpoints, the int8-convrot correctness self-test, and
// the two native kernels' throughput with a CPU cross-check.
#pragma once

#include <string>
#include <vector>

#include "util/media_common.hpp"

namespace phi::media {

struct BenchOptions {
	std::vector<std::string> model_paths;  // defaults to the installed checkpoints
	bool quick = false;                    // skip the multi-GB streaming pass
	bool verbose = false;
	std::string out_json;                  // optional: write the report as JSON
	// --image: run the real Qwen-Image-2.1 chain once and report the segmented
	// timings and the VRAM peak. This is the M1 acceptance instrument.
	bool image = false;
	std::string image_prompt = "a red fox sitting in tall grass, golden hour";
	std::string image_negative;            // only used when --image-cfg > 1
	i64 image_w = 1024, image_h = 1024;
	std::string image_out;                 // optional: write the PNG here
	u64 image_seed = 0;                    // 0 = pick one (drawn here, printed below)
	// --image-ref path (repeatable): the workflow's reference images, i.e. the
	// reference-latent edit path ("put this shirt on the character in image 1").
	std::vector<std::string> image_refs;
	float image_cfg = 1.0f;
	// The pixel area each reference is resized to (0 = the canvas area).
	i64 image_ref_area = 0;

	// --video: run the real H3 chain once (text -> 16 joint steps -> VAEs -> mp4)
	// and report the segment timings, the VRAM peak and the residency plan. This
	// is the plan's §2 acceptance instrument.
	bool video = false;
	std::string video_prompt = "a cinematic shot of a red fox walking through tall golden grass at sunset, camera slowly pushing in";
	i64 video_w = 960, video_h = 540;
	i64 video_frames = 243;                 // 10.125 s @24fps (17k+5)
	std::string video_out = "video_out.mp4";
	u64 video_seed = 0;                    // 0 = pick one (drawn here, printed below)
	// --video-ref path (repeatable) / --video-ref-audio path (repeatable): the
	// ref2va reference blocks. A reference image is scaled to the generation's own
	// pixel area and a reference audio is resampled to the VAE's 32 kHz, exactly as
	// the tool does (`MiniMaxH3ReferenceToVideo`), so the residency plan these input
	// sizes drive is the one a tool call gets.
	std::vector<std::string> video_refs;
	std::vector<std::string> video_ref_audios;
	// Reference *videos* (the ref2va chain conditions on clips as well as stills)
	// and the soundtracks paired with them by index.
	std::vector<std::string> video_ref_clips;
	std::vector<std::string> video_ref_clip_audios;
	std::string video_ref_image_size = "match";
	// --video-lora path (repeatable): the video chain's LoRA files, in order, exactly
	// as the settings' tools.media.video_loras reach the tool. Without this the bench
	// could not reproduce a user's run: a LoRA rewrites every adapted weight, so a
	// chain measured without it is not the chain the tool builds.
	std::vector<std::string> video_loras;
	// 0 = the chain's own default (kVideoSteps). A small number is how a shape
	// that costs minutes is measured for its *segments* rather than its result:
	// the sampling loop is the only part that scales with the step count, so one
	// step plus the VAEs is the whole shape's cost minus (steps-1) samples.
	i64 video_steps = 0;
};

// Runs the benchmark and prints the report. Returns 0 on success, non-zero when
// a hard check failed (so CI/scripts can gate on it).
int media_bench_main(const BenchOptions& opts);

// Parses --media-bench arguments (everything after the flag). Returns false when
// the arguments are invalid (message in *error).
bool parse_media_bench_args(const std::vector<std::string>& args, BenchOptions* opts,
                            std::string* error);

}  // namespace phi::media
