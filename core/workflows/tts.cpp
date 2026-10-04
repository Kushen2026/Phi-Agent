// The TTS chain: Breeze-TTS-2, voice design (a timbre description + text).
//
// Three stages, the same shape the image and video chains use:
//
//   BreezeTTS2LoadModel  -> MediaModelLoader      (arch "breeze_tts")
//   BreezeTTS2VoiceDesign-> MediaBreezeTextEncode
//                          MediaBreezeSampler
//                          MediaBreezeDecode
//   SaveAudio            -> MediaSaveAudio        (16-bit WAV)
//
// Only the design path is wired: there is no reference clip and therefore no
// `BreezeTTS2VoiceClone` / `BreezeTTS2WhisperTranscribe` node, which is the one
// thing this engine deliberately does not ship (a voice is described, never
// cloned).
#include "graph/graph_workflows.hpp"
#include "workflows/common.hpp"

#include <cmath>

namespace phi::media {

using namespace workflows;

// The two unit bridges (see the note on `kTtsFramesPerSecond`). The rate is the
// *codec's*: Breeze-TTS-2's audio tokenizer emits 12.5 acoustic frames per
// second, so this is a property of the checkpoint family and not a tuning knob.
i64 tts_frames_for_seconds(double seconds) {
	if (!(seconds > 0.0)) return 0;
	const double f = seconds * kTtsFramesPerSecond;
	// The sampler's own ceiling; a request past it is clamped rather than refused,
	// because "as long as the model will go" is a legal thing to ask for.
	if (f >= 2047.0) return 2047;
	return std::max<i64>(1, (i64)std::llround(f));
}

double tts_seconds_for_frames(i64 frames) {
	if (frames <= 0) return 0.0;
	return (double)frames / kTtsFramesPerSecond;
}

Graph build_tts_workflow(const TtsWorkflowParams& p) {
	Graph g;
	g.name = "breeze_tts_2";

	GraphNode model = gn(1, "MediaModelLoader");
	model.params["arch"] = "breeze_tts";
	put_str(model.params, "model", p.model);
	// The codec and the tokenizer sit beside the weights by default; the JSON
	// configs are selectable in the settings panel (blank = read the sibling
	// config.json, the loaders' historical behaviour).
	put_str(model.params, "codec", p.codec);
	put_str(model.params, "tokenizer", p.tokenizer);
	put_str(model.params, "config", p.config);
	put_str(model.params, "codec_config", p.codec_config);

	// The voice-design prompt: the instruction describes the timbre, `text` is
	// what is spoken. `cfg_scale` 4 is the released node's default for design.
	GraphNode enc = gn(2, "MediaBreezeTextEncode");
	enc.params["text"] = p.text;
	enc.params["instruction"] = p.instruction;
	enc.params["cfg_scale"] = p.cfg_scale;

	GraphNode sampler = gn(3, "MediaBreezeSampler");
	// The sampler counts frames; a caller speaks seconds. `duration_seconds` is the
	// documented input and the frame budget is derived from it here, so the two
	// units cannot drift apart (the tool reports the cap back in seconds).
	sampler.params["max_new_tokens"] = (int64_t)(p.max_new_tokens > 0
	                                                ? p.max_new_tokens
	                                                : tts_frames_for_seconds(p.duration_seconds));
	sampler.params["temperature"] = p.temperature;
	sampler.params["top_k"] = (int64_t)p.top_k;
	sampler.params["top_p"] = p.top_p;
	sampler.params["repetition_penalty"] = p.repetition_penalty;
	sampler.params["depth_temperature"] = p.depth_temperature;
	sampler.params["depth_top_k"] = (int64_t)p.depth_top_k;
	sampler.params["depth_top_p"] = p.depth_top_p;
	sampler.params["seed"] = (int64_t)p.seed;

	GraphNode decode = gn(4, "MediaBreezeDecode");

	GraphNode save = gn(5, "MediaSaveAudio");
	save.params["path"] = p.output.empty() ? "speech.wav" : p.output;

	g.nodes = {model, enc, sampler, decode, save};
	g.links = {
	    {1, "MODEL", 2, "model"},
	    {2, "CONDITIONING", 3, "conditioning"},
	    {1, "MODEL", 3, "model"},
	    {1, "MODEL", 4, "model"},
	    {3, "LATENT", 4, "latent"},
	    {4, "AUDIO", 5, "audio"},
	};
	g.outputs = {{5, "path"}};
	return g;
}

}  // namespace phi::media
