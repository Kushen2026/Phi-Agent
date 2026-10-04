// Ready-made graphs that reproduce the two shipped media tools.
//
// These are the "workflow templates" a caller can run as-is, edit, or use as the
// starting point for a new chain. They are built out of the same registered
// nodes a user would drag onto a canvas — nothing here is a special path — so
// they double as worked examples of how the modules compose:
//
//   image:  ModelLoader ─┐
//           ClipLoader ──┼─ QwenImageEditEncode ─┬─ CONDITIONING ─┐
//           VaeLoader ───┘                      └─ LATENT ───────┼─ KSampler ─ VaeDecodeImage ─ SaveImage
//           EmptyLatentImage (text-to-image) ───────────────────┘      ↑
//           Scheduler(model) ──────────────────────────────────────────┘
//
//   video:  ModelLoader ─┬─ H3SigmaShift ───────────┐
//           ClipLoader ──┴─ H3Ref2VaEncode ─┐      ├─ H3Sampler ─┬─ VaeDecodeVideo ─┐
//           VaeLoader(video) ───────────┘           │             └─ VaeDecodeAudio ─┼─ SaveVideo
//           VaeLoader(audio) ───────────────────────┘                              │
//           EmptyLatentVideo ───────────────────────┘                              │
//           Scheduler(model) ───────────────────────┘                              │
// One chain, one file: `core/workflows/image.cpp`, `video.cpp` and `audio.cpp`
// each wire the same shared nodes; `workflows/common.cpp` holds the small bits
// they share (node/link spelling, the LoRA-chain wiring).
#pragma once

#include <string>
#include <vector>

#include "graph/graph.hpp"

namespace phi::media {

struct ImageWorkflowParams {
	std::string prompt;
	std::string negative_prompt;
	i64 width = 1024;
	i64 height = 1024;
	i64 steps = 0;                 // 0 = the chain default
	// ComfyUI's KSampler sampler/scheduler names. Empty keeps the released
	// workflow's own pair (image: euler + simple), so a caller that does not care
	// gets exactly the schedule it got before this setting existed. The nodes
	// validate the names and fall back to their defaults for an unknown one.
	std::string sampler;
	std::string scheduler;
	// The initial-noise seed, used exactly as given - 0 included, since being
	// reproducible from the number is the point of it. A caller that wants a fresh
	// one draws it (make_media_seed()) rather than leaving 0 here.
	u64 seed = 0;
	float cfg = 1.0f;
	std::string output;            // required
	std::vector<std::string> ref_images;
	// The reference sizing edge (ComfyUI's `resolution` widget): every reference is
	// resized to about ref_resolution^2 px at multiples of 32, aspect preserved,
	// independently of the canvas being generated. 0 keeps each reference at its
	// own size. This is the knob that bounds an edit's cost - reference tokens
	// ride through every sampling step - so it is deliberately decoupled from
	// `width`/`height`.
	i64 ref_resolution = 1024;
	// ComfyUI's `QwenImage21Cache`: keep the prompt's and the references' K/V and
	// run only the target rows from the second step on. Off by default - it trades
	// VRAM for step time, and this engine would rather spend VRAM on the streamed
	// weights (see `MediaQwenImage21Cache`).
	bool prefix_cache = false;
	std::string model;             // optional explicit file overrides
	std::string clip;
	std::string vae;
	// The tokenizer files this chain reads (vocab.json / merges.txt /
	// tokenizer_config.json). Empty = the chain's own role default.
	std::string tokenizer_vocab;
	std::string tokenizer_merges;
	std::string tokenizer_config;
	// LoRA files (names under models/, or paths), applied in order through one
	// MediaLoraLoader node wired into the model and CLIP loaders. Empty = no LoRA
	// node at all, so a plain chain is byte-for-byte the graph it was.
	std::vector<std::string> loras;
};

struct VideoWorkflowParams {
	std::string prompt;
	i64 frames = 121;
	i64 width = 960;
	i64 height = 540;
	i64 fps = 24;
	i64 steps = 0;
	// Sampler/scheduler names (ComfyUI KSampler). Empty keeps the released video
	// workflow's pair (res_multistep + beta).
	std::string sampler;
	std::string scheduler;
	u64 seed = 0;                  // as in ImageWorkflowParams: used verbatim
	std::string output;            // required
	std::vector<std::string> ref_images;
	std::vector<std::string> ref_audios;
	std::vector<std::string> ref_videos;
	// Soundtracks of the same-numbered reference videos (`ref_video_audios[k]`
	// belongs to `ref_videos[k]`); empty entries mean "no soundtrack".
	std::vector<std::string> ref_video_audios;
	// "match" (down-only scale to the generation's pixel area) or "max" (the
	// reference pipeline's 2048 px short edge).
	std::string ref_image_size = "match";
	// The H3 flow shifts (ComfyUI's MiniMaxH3SigmaShift). 0 lets the chain default
	// (12 video / 3 audio) apply, which is what the released model was trained at.
	float h3_shift_video = 0.0f;
	float h3_shift_audio = 0.0f;
	std::string model;
	std::string clip;
	std::string vae;
	std::string vae_audio;
	// The tokenizer files this chain reads (vocab.json / merges.txt /
	// tokenizer_config.json). Empty = the chain's own role default.
	std::string tokenizer_vocab;
	std::string tokenizer_merges;
	std::string tokenizer_config;
	std::vector<std::string> loras;  // see ImageWorkflowParams::loras
};

// Builds the default image graph (prompt -> PNG) from the parameters.
Graph build_image_workflow(const ImageWorkflowParams& p);
// Builds the default video graph (prompt (+ refs) -> mp4).
Graph build_video_workflow(const VideoWorkflowParams& p);
// Builds an audio-only graph (prompt -> wav): the H3 joint audio stream, decoded
// through the audio VAE and written as PCM without any video decode or mux.
//
// This exists to show the point of the node system: it reuses the *same*
// MediaModelLoader / MediaClipLoader / MediaH3Ref2VaEncode / MediaEmptyLatent /
// MediaScheduler / MediaSampler nodes the video chain uses, and only swaps the
// tail (MediaVaeDecode(role=audio) + MediaSaveAudio for MediaVaeDecode(role=video)
// + MediaSaveVideo). A new chain is a new wiring, not a new set of modules — the
// parameters are the video ones because the audio stream is part of the same
// joint model.
Graph build_audio_workflow(const VideoWorkflowParams& p);

// ── the music chain (ACE-Step 1.5) ──────────────────────────────────────────
//
// One text-to-music request: tags/lyrics/metadata in, a 48 kHz WAV out. The
// wiring is the node-for-node transcription of `ACE-Step-音乐生成.json`:
//
//   ModelLoader(ace_step15) ─┬― AceTextEncode ─┐
//   ClipLoader(ace15) ───────┘                │
//   VaeLoader(ace_audio) ─────────────┐       ├― AceSampler ─ AceVaeDecode ─ SaveAudio
//   AceModelSampling ─ Scheduler ──────┴──────―┘
//   AceLatent(seconds) ───────────────────────―┘
//
// The clip loader carries *two* towers (the Qwen3-0.6B embedder and the Qwen3-4B
// audio-code LM), exactly as ComfyUI's DualCLIPLoader does, and the text encode
// node is the one that runs the LM's guided sampling to produce the semantic
// audio codes the DiT conditions on.
struct MusicWorkflowParams {
	std::string tags;              // style / instrument / mood tags
	std::string lyrics;            // the lyrics, `[instrumental]` when none
	i64 bpm = 72;
	double duration = 60.0;        // seconds of music
	i64 steps = 0;                 // 0 = the chain default (kMusicSteps)
	// Sampler/scheduler names; empty keeps the released ACE-Step workflow's
	// pair (euler + simple).
	std::string sampler;
	std::string scheduler;
	u64 seed = 0;                  // used verbatim, 0 included
	std::string timesignature = "4";
	std::string language = "en";
	std::string keyscale = "D minor";
	// The LM that writes the semantic audio codes. On by default (the released
	// workflow's `generate_audio_codes`), with its own guidance knobs.
	bool generate_audio_codes = true;
	double lm_cfg_scale = 7.0;
	double lm_temperature = 0.85;
	double lm_top_p = 0.9;
	i64 lm_top_k = 0;
	double lm_min_p = 0.0;
	std::string output;            // required
	std::string model;             // explicit file overrides (optional)
	std::string clip;
	std::string clip_lm;
	std::string vae;
	// The tokenizer files this chain reads (vocab.json / merges.txt /
	// tokenizer_config.json). Empty = the chain's own role default.
	std::string tokenizer_vocab;
	std::string tokenizer_merges;
	std::string tokenizer_config;
};

// Builds the default music graph (tags + lyrics -> wav).
Graph build_music_workflow(const MusicWorkflowParams& p);

// ── the tts chain (Breeze-TTS-2, voice design) ────────────────────────────
//
// The one TTS request this engine ships: a timbre *description* plus the words
// to speak, never a reference clip (no voice cloning). The graph is the same
// three-stage shape the other chains use —
//
//   ModelLoader(breeze_tts) ─ BreezeTextEncode ─ BreezeSampler ─ BreezeDecode ─ SaveAudio
//
// — where the encode node turns the instruction and the text into the prompt the
// backbone prefills on, the sampler runs the backbone + depth-decoder loop, and
// the decode node turns the 12 Hz acoustic codes into 24 kHz PCM.
//
// ── the two units of "length" ────────────────────────────────────────────────
//
// The autoregressive loop counts *acoustic code frames*, and Breeze-TTS-2's codec
// runs at a fixed rate (12.5 frames per second), so "how long should this be" -
// which is what a caller actually knows - is seconds, and the frame count is a
// derived quantity. The two helpers below are the only bridge between the two, so
// the tool, the workflow and the report cannot disagree about the rate. A TTS
// length is a *cap*: the model stops on its own EOS token, which is usually well
// before the cap, and the report gives the length that was actually produced.
inline constexpr double kTtsFramesPerSecond = 12.5;

// Seconds -> the sampler's frame budget (rounded, at least one frame).
i64 tts_frames_for_seconds(double seconds);
// The frame budget -> the seconds it corresponds to (the cap, as a caller sees it).
double tts_seconds_for_frames(i64 frames);

struct TtsWorkflowParams {
	std::string text;              // the words to speak (required)
	std::string instruction;       // the timbre description (required for voice design)
	double cfg_scale = 4.0;        // the voice-design default
	// The cap on generated *acoustic code frames*, which is the sampler's own unit.
	// A caller that thinks in seconds sets `duration_seconds` instead and leaves
	// this 0; `build_tts_workflow` derives the budget with `tts_frames_for_seconds`.
	// 0 (with `duration_seconds` 0 too) means the model's own default of 1500
	// frames, i.e. 120 s, still subject to EOS.
	i64 max_new_tokens = 0;
	// The requested length in seconds. 0 = let the model decide from the text.
	double duration_seconds = 0.0;
	double temperature = 0.7;
	i64 top_k = 0;
	double top_p = 1.0;
	double repetition_penalty = 1.0;
	double depth_temperature = 1.0;
	i64 depth_top_k = 0;
	double depth_top_p = 1.0;
	u64 seed = 0;
	std::string output;            // required
	// Explicit file overrides (optional; empty = the role's shipped default).
	std::string model;
	std::string codec;             // audio_tokenizer/model.safetensors
	std::string tokenizer;         // tokenizer.json
	// The two JSON configs the Breeze-TTS-2 loaders read. Empty keeps the
	// loaders' sibling lookup (config.json beside the weights / the codec).
	std::string config;            // Breeze-TTS-2 config.json
	std::string codec_config;      // codec config.json
};

// Builds the default TTS graph (instruction + text -> wav).
Graph build_tts_workflow(const TtsWorkflowParams& p);

}  // namespace phi::media
