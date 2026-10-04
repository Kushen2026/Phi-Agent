// The media payloads a node socket can carry, and the engine-side state a graph
// run keeps alive.
//
// `graph.hpp` is the transport; this is what travels on it. Each struct below is
// one socket type's concrete value:
//
//   MODEL        ModelData        a DiT backbone + its LoRA chain
//   CLIP         ClipData         a text encoder + tokenizer + its LoRA chain
//   VAE          VaeData          an image / video / audio VAE
//   LORA         LoraData         a LoRA chain (delta files, applied at load)
//   CONDITIONING ConditioningData an encoded prompt (+ references, + tags, +
//                              the reference-grid geometry the DiT needs)
//   LATENT       LatentData       a latent tensor + the geometry it was planned at
//   SIGMAS       SigmasData       a sigma schedule
//   IMAGE        ImageData        8-bit RGB still
//   AUDIO        AudioData        interleaved float PCM
//   VIDEO        VideoData        a decoded clip (planar frames + geometry)
//
// The structs hold `shared_ptr`s to the *same* engine classes the two pipelines
// use, so a node is a thin adapter rather than a re-implementation. `ModelData`
// in particular wraps either `ImageDiT` or `AvDiT`, which is what
// lets one `MediaSampler` node serve the image chain and the video chain without
// either knowing about the other's forward signature - and one `MediaVaeDecode`
// serve every VAE.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "graph/graph.hpp"
#include "models/lora.hpp"
#include "models/av_dit.hpp"   // H3Ref
#include "models/media_geometry.hpp"
#include "sampling/sampling.hpp"   // SamplingFlow
#include "text/tokenizer.hpp"      // TokenizerFiles

namespace phi::media {

struct GpuCtx;
class ImageDiT;
class AvDiT;
class TextEncoder8B;
class TextEncoder32B;
class ImageVae;
class VideoVae;
class AudioVae;
class Qwen2Tokenizer;
class AceDiT;
class AceQwen3;
class AceLm;
class AceVae;
class BreezeTts;

// ── payloads ───────────────────────────────────────────────────────────────

struct ModelData : NodeData {
	SocketType socket() const override { return SocketType::Model; }
	const char* kind() const override { return "MODEL"; }
	std::string arch;                 // "qwen_image" | "minimax_h3" | "ace_step15" |
	                                  // "breeze_tts"
	std::string path;
	std::shared_ptr<ImageDiT> image;
	SamplingFlow image_flow;          // the image chain's ModelSamplingFlux(0.69)
	std::shared_ptr<AvDiT> h3;
	// ── music chain (ACE-Step 1.5) ──
	// The DiT bundle plus its condition encoder/tokenizer/detokenizer; the
	// audio-code LM and the embedder live on the CLIP socket (they are text
	// encoders), exactly as ComfyUI wires DualCLIPLoader -> TextEncodeAceStepAudio1.5.
	std::shared_ptr<AceDiT> ace;
	// `MediaAceModelSampling` (ComfyUI's ModelSamplingAuraFlow): the flow shift the
	// sigma grid is built with. The released 3.0 is the loader's default.
	float ace_shift = 3.0f;
	// ── tts chain (Breeze-TTS-2) ──
	std::shared_ptr<BreezeTts> tts;
	// The H3 flow shifts (ComfyUI's `MiniMaxH3SigmaShift`, i.e. ModelSamplingAV):
	// `h3_shift_video` drives the sampler's sigma grid and both are handed to the
	// DiT, which inverts the video schedule to the shared base grid and derives the
	// audio schedule from it (`audio_scale = shift_video / shift_audio`). They live
	// on the MODEL socket so a `MediaH3SigmaShift` node can patch them the way the
	// reference node patches its model - which is what a distilled / few-step LoRA
	// needs, and what used to be hard-coded here.
	float h3_shift_video = 12.0f;
	float h3_shift_audio = 3.0f;
	// The Qwen-Image-2.1 prefix K/V cache (ComfyUI's `QwenImage21Cache`): keep the
	// prompt's and the references' keys/values, which do not depend on the sampling
	// step, so every step after the first runs the target rows alone. Set by a
	// `MediaQwenImage21Cache` node in front of the sampler. It trades VRAM for step
	// time (see `ImageDiT::set_prefix_cache`), so it is off unless a graph asks.
	bool image_prefix_cache = false;
	std::shared_ptr<LoraSet> loras;   // applied at load time; may be null
	std::vector<std::string> lora_paths;
};

struct ClipData : NodeData {
	SocketType socket() const override { return SocketType::Clip; }
	const char* kind() const override { return "CLIP"; }
	std::string arch;                 // "qwen3vl_8b" | "qwen3vl_32b" | "ace15"
	std::string path;
	TokenizerFiles tokenizer_files;   // vocab / merges / config, resolved
	std::shared_ptr<TextEncoder8B> te8;
	std::shared_ptr<TextEncoder32B> te32;
	// ACE-Step 1.5: the Qwen3-0.6B embedder and the Qwen3-4B audio-code LM.
	std::shared_ptr<AceQwen3> ace_te;
	std::shared_ptr<AceLm> ace_lm;
	std::shared_ptr<Qwen2Tokenizer> tokenizer;   // always present
	std::shared_ptr<LoraSet> loras;
	i64 hidden = 0;
};

struct VaeData : NodeData {
	SocketType socket() const override { return SocketType::Vae; }
	const char* kind() const override { return "VAE"; }
	std::string role;                 // "image" | "video" | "audio" | "ace_audio"
	std::string path;
	std::shared_ptr<ImageVae> image;
	std::shared_ptr<VideoVae> video;
	std::shared_ptr<AudioVae> audio;
	std::shared_ptr<AceVae> ace;
};

struct LoraData : NodeData {
	SocketType socket() const override { return SocketType::Lora; }
	const char* kind() const override { return "LORA"; }
	std::shared_ptr<LoraSet> set;
	std::vector<std::string> paths;
};

struct ConditioningData : NodeData {
	SocketType socket() const override { return SocketType::Conditioning; }
	const char* kind() const override { return "CONDITIONING"; }
	std::string arch;

	// ── image chain (Qwen-Image-2.1) ──
	std::vector<float> positive;      // [n_pos, dim]
	std::vector<float> negative;      // [n_neg, dim] (empty unless cfg > 1)
	i64 n_pos = 0, n_neg = 0;
	i64 dim = 0;
	std::vector<i32> slots;           // image_slots, one per reference
	i64 n_refs = 0;
	// Reference latents (the image VAE encode of the reference images). The DiT
	// splices them into its sequence; they are built by the TextEncode node when a
	// VAE is wired in, exactly as `image_gen` builds them before its sampler.
	std::vector<std::vector<float>> ref_latents;
	// One entry per reference latent: *that* reference's own latent grid (latent
	// pixels = canvas / 16), plus its pixel count. ComfyUI's
	// `TextEncodeQwenImage21` sizes every reference on its own and the DiT's
	// `build_sequence` reads each latent's own h/w, so the geometry travels as a
	// list rather than as one shared grid - and the residency plan is priced off
	// that list (a heterogeneous set used to be charged at the first block's
	// grid, and a small second reference used to be stretched onto it).
	std::vector<i64> ref_hs, ref_ws, ref_px;

	// ── video chain (MiniMax H3) ──
	std::vector<float> embed;         // [n_ids, text_dim]
	i64 n_ids = 0;
	std::vector<i32> tags;            // per-token modality (0 visual, 1 text)
	std::vector<H3Ref> refs;          // reference blocks, in packing order
	std::vector<std::vector<float>> ref_vlat, ref_alat;
	std::vector<double> ref_audio_seconds;
	i64 text_dim = 0;
	// What the reference build left resident on the card (the video VAE's weights
	// and the vision arenas that survive the release above). The H3 sampler hands
	// this to plan_residency as `extra_resident`, the way video_gen does: a plan
	// that ignores it sizes the resident window for a card the run does not have.
	u64 ref_footprint = 0;

	// Re-point every H3Ref::latent / audio_latent at this object's own storage.
	// Call after the vectors are final (a reallocation would dangle them).
	void link_refs() {
		for (size_t i = 0; i < refs.size(); i++) {
			refs[i].latent = ref_vlat[i].empty() ? nullptr : ref_vlat[i].data();
			refs[i].audio_latent = ref_alat[i].empty() ? nullptr : ref_alat[i].data();
		}
	}

	// ── music chain (ACE-Step 1.5) ──
	// The precomputed DiT conditioning (the condition encoder's output plus the
	// detokenised audio-code context). Built by MediaAceTextEncode.
	std::shared_ptr<void> ace_cond;

	// ── tts chain (Breeze-TTS-2) ──
	// The merged prompt the sampler prefills on (text-encoder rows, audio
	// placeholder rows and the position/mask bookkeeping). Built by
	// MediaBreezeTextEncode; opaque here so this header stays free of the model.
	std::shared_ptr<void> breeze_prompt;
};

struct LatentData : NodeData {
	SocketType socket() const override { return SocketType::Latent; }
	const char* kind() const override { return "LATENT"; }
	std::string arch;                 // "image" | "video" | "music" | "tts"

	// image latent: [c, h, w]
	i64 c = 0, h = 0, w = 0;
	std::vector<float> x;
	// image geometry: the request, its 16-aligned canvas, and the crop between them
	i64 req_w = 0, req_h = 0, canvas_w = 0, canvas_h = 0;

	// video latents: video [24, vt, lat_h, lat_w], audio [32, 2, at]
	i64 vt = 0, lat_h = 0, lat_w = 0, at = 0;
	std::vector<float> xv, xa;
	VideoPlan plan;                   // frames / canvas / output geometry
	bool audio_carried = false;       // xa is sampler-space (video schedule)
	float audio_scale = 4.0f;         // shift / audio_shift for the audio VAE
	u64 seed = 0;

	// Weight residency the image sampler planned and finished with. The sampler
	// hands its window back at the phase boundary (before the VAE decode), so these
	// are captured here rather than read off the DiT afterwards - a reader that
	// looks at the model after the run only ever sees the released state.
	i64 resident_layers = 0;      // blocks the plan kept
	i64 resident_effective = 0;   // blocks after the loop's last tune
	u64 resident_bytes = 0;       // what those blocks charge
	// The prefix K/V cache, as the run left it (same reason as above: the sampler
	// releases it with the window).
	bool prefix_cache = false;    // it ran with the kept prefix
	i64 prefix_cache_rows = 0;    // rows kept per block (prefix + target)
	u64 prefix_cache_bytes = 0;   // both tensors, all blocks
	// What the cache would have taken for this shape. Reported when the plan
	// declined it (no room beside the weight window), which is the one case where
	// the caller can act on the number.
	u64 prefix_cache_wanted = 0;

	// Same for the H3 sampler: `release_weights()` clears the plan (`planned =
	// false`), so the residency it ran with is snapshotted here for the report.
	H3Residency h3_residency;

	// ── music chain (ACE-Step 1.5) ──
	// The latent is [64, mt] channel-major (the DiT's own layout); `seconds` is
	// the requested duration, from which mt = round(seconds * 48000 / 1920).
	std::vector<float> xm;
	i64 mt = 0;
	i64 seconds = 0;

	// ── tts chain (Breeze-TTS-2) ──
	// `codes` is [frames, codebooks] frame-major acoustic tokens out of the
	// backbone + depth decoder; the codec turns them into PCM.
	std::vector<i32> codes;
	i64 code_frames = 0;
	i64 codebooks = 0;
};

struct SigmasData : NodeData {
	SocketType socket() const override { return SocketType::Sigmas; }
	const char* kind() const override { return "SIGMAS"; }
	std::string arch;                 // "image" | "video" | "music"
	std::string scheduler;            // a ComfyUI scheduler name (simple | karras | beta | ...)
	std::vector<float> video;         // descending, last == 0
	std::vector<float> audio;         // H3 audio grid (video path); unused for image
	i64 steps() const { return (i64)video.size() - 1; }
};

struct ImageData : NodeData {
	SocketType socket() const override { return SocketType::Image; }
	const char* kind() const override { return "IMAGE"; }
	std::vector<unsigned char> rgb;   // tightly packed HWC, 3 bytes/pixel
	i64 w = 0, h = 0;
};

struct AudioData : NodeData {
	SocketType socket() const override { return SocketType::Audio; }
	const char* kind() const override { return "AUDIO"; }
	std::vector<float> pcm;           // interleaved
	i64 sample_rate = 32000;
	i64 channels = 2;
};

struct VideoData : NodeData {
	SocketType socket() const override { return SocketType::Video; }
	const char* kind() const override { return "VIDEO"; }
	std::vector<float> planes;        // [3, frames, h, w]
	i64 frames = 0, h = 0, w = 0;
	i64 out_w = 0, out_h = 0;         // effective output (symmetric centre crop)
	i64 fps = 24;
};

// ── typed accessors (null on a wrong socket) ───────────────────────────────
std::shared_ptr<ModelData> as_model(const Value& v);
std::shared_ptr<ClipData> as_clip(const Value& v);
std::shared_ptr<VaeData> as_vae(const Value& v);
std::shared_ptr<LoraData> as_lora(const Value& v);
std::shared_ptr<ConditioningData> as_conditioning(const Value& v);
std::shared_ptr<LatentData> as_latent(const Value& v);
std::shared_ptr<SigmasData> as_sigmas(const Value& v);
std::shared_ptr<ImageData> as_image(const Value& v);
std::shared_ptr<AudioData> as_audio(const Value& v);
std::shared_ptr<VideoData> as_video(const Value& v);

// ── engine-side state a run keeps ──────────────────────────────────────────

// Loaded modules are cached for the life of a `GraphContext`, keyed on
// (arch, path, loras): a graph that loads the same DiT twice — or a caller that
// runs the same workflow again on the same context — reuses the weights instead
// of re-reading the checkpoint. This is the node world's equivalent of the
// process-wide `MediaEngine`, minus the pipelines.
struct GraphModelCache {
	std::map<std::string, std::shared_ptr<ModelData>> models;
	std::map<std::string, std::shared_ptr<ClipData>> clips;
	std::map<std::string, std::shared_ptr<VaeData>> vaes;
	// tokenizers are keyed by directory (the two towers share the vocabulary)
	std::map<std::string, std::shared_ptr<Qwen2Tokenizer>> tokenizers;
	// Everything a node needs to keep alive between runs that is *not* a socket
	// payload, keyed on whatever identifies it. The music chain's condition
	// builder (a second view of the ACE DiT bundle) is the entry this exists for:
	// it is a device-side object, so it has to die with the context, and the state
	// bag is the one thing the engine already tears down on a lost device.
	std::map<std::string, std::shared_ptr<void>> aux;
};

// Gets (creating on first use) the media state bag attached to `ctx`.
std::shared_ptr<GraphModelCache> graph_state(GraphContext& ctx);

// Hands back every byte the cached modules are holding in VRAM: the image / video
// / audio VAEs' weights and scratch, and the two DiTs' resident windows and
// streaming arenas. The *loaded* modules stay cached (their safetensors mappings
// are host-side), so the next run re-streams from the page cache - which is what
// the two media tools always did between calls (they shared one card and could
// not use the other chain's weights). The caller is expected to hand the shared
// wa/aa arenas back as well; see the media engine's release path.
void release_media_cache(GraphModelCache& cache);

// The GpuCtx a run was handed, or a MediaError naming the missing context.
GpuCtx* graph_gpu(GraphContext& ctx);

}  // namespace phi::media
