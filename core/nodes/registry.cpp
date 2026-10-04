// The media node catalogue entry point: builds the process-wide registry
// and fills it from every node's own translation unit.
//
// The registration order is the catalogue's display order.
//
// ── the catalogue, and the ComfyUI node each entry transcribes ──────────────
//
// The two chains are ComfyUI graphs, and every node below is a port of the node
// the released workflow runs, so the chains can be read side by side:
//
//   loaders        MediaModelLoader      UNETLoader / CheckpointLoaderSimple
//                  MediaClipLoader       CLIPLoader
//                  MediaVaeLoader        VAELoader
//                  MediaLoraLoader       LoraLoader (chain: one node feeds both
//                                        loaders, exactly as the t2v template
//                                        wires LoraLoaderModelOnly + the clip one)
//   conditioning   MediaQwenImageEditEncode  TextEncodeQwenImage21
//                  MediaH3Ref2VaEncode      MiniMaxH3ReferenceToVideo
//                                           (+ MiniMaxH3Tokenizer's presentation)
//   latent         MediaEmptyLatent         EmptySD3LatentImage / EmptyMiniMaxH3LatentAV
//                                           (arch chooses the geometry)
//   sampling       MediaScheduler            BasicScheduler / KSampler's sigmas
//                  MediaH3SigmaShift        MiniMaxH3SigmaShift (ModelSamplingAV)
//                  MediaQwenImage21Cache    QwenImage21Cache (prefix K/V)
//                  MediaSampler             KSampler (the MODEL's arch chooses the flow)
//   vae            MediaVaeEncodeImage       VAEEncode
//                  MediaVaeDecode           VAEDecode / VAEDecodeAudio (the VAE's
//                                           role chooses which)
//   io             MediaLoadImage           LoadImage
//                  MediaSaveImage           SaveImage
//                  MediaSaveAudio           SaveAudio
//                  MediaSaveVideo           CreateVideo + SaveVideo (muxed mp4)
//   utils          MediaSeed / MediaConstInt  (the widget-value nodes a hand-wired
//                                           graph needs; the tool path writes the
//                                           same values as parameters)
//
// ── the music chain (ACE-Step 1.5) ─────────────────────────────────────────
//
//   loaders        MediaModelLoader          UNETLoader (arch "ace_step15": the DiT
//                                            bundle carries the decoder and the
//                                            condition encoder)
//                  MediaClipLoader           DualCLIPLoader (arch "ace15": the
//                                            Qwen3-0.6B embedder + the Qwen3-4B
//                                            audio-code LM)
//                  MediaVaeLoader            VAELoader (role "ace_audio")
//   conditioning   MediaAceTextEncode        TextEncodeAceStepAudio1.5 (+ the
//                                            Qwen3-4B guided code sampling)
//   latent         MediaEmptyLatent          EmptyAceStep1.5LatentAudio (arch "music")
//   sampling       MediaAceModelSampling     ModelSamplingAuraFlow
//                  MediaScheduler            KSampler's `simple` sigmas
//                  MediaSampler              KSampler(sampler="euler", cfg=1)
//   vae            MediaVaeDecode            VAEDecodeAudio
//
// ── the tts chain (Breeze-TTS-2, voice design) ─────────────────────────────
//
//   loaders        MediaModelLoader          BreezeTTS2LoadModel (arch
//                                            "breeze_tts": the checkpoint, the
//                                            codec and the tokenizer)
//   conditioning   MediaBreezeTextEncode     BreezeTTS2VoiceDesign (the timbre
//                                            instruction + the text)
//   sampling       MediaBreezeSampler        (the backbone + depth-decoder loop)
//   audio          MediaBreezeDecode         (the 12 Hz codec)
//
// The two new chains are *the same kind of object* as the first two: a graph over
// the nodes above, built by core/workflows/music.cpp and core/workflows/tts.cpp.
// Nothing in them is a special path - which is what let the TTS chain reuse the
// existing loaders and SaveAudio unchanged.
//
// ── one node per *stage*, not one node per model ───────────────────────────
//
// ComfyUI's catalogue has a single `KSampler`, a single `VAEDecode` and a single
// `EmptyLatent*` per geometry, and the model on the socket decides what each of
// them does. This catalogue now follows the same rule: `MediaSampler` dispatches
// on the MODEL's arch, `MediaVaeDecode` on the VAE's role, `MediaEmptyLatent` on
// its own `arch` parameter (a latent node has no model socket to infer from).
// That replaced ten nodes (`MediaKSampler` / `MediaH3Sampler` / `MediaAceSampler`,
// the four `MediaVaeDecode*` / `MediaAceVaeDecode`, and the three `Media*Latent`)
// with three, so a new backbone is a new *arch branch* rather than a new set of
// nodes - and the ready-made chains that used to name a different node per chain
// now name the same three. `MediaBreezeSampler` stays separate on purpose: TTS is
// an autoregressive token loop, not a sigma-scheduled diffusion step.
//
// What is deliberately *not* a node: the model-patch wrappers (a LoRA chain is
// folded in at load time), the Qwen-Image-2.1 prefix KV cache, the Fun
// ControlNet and Breeze-TTS-2's voice *clone* / whisper-transcribe pair (this
// engine takes a timbre description, never a reference clip), none of which the
// shipped chains ask for - the engine streams its weights, so a prefix cache
// would buy memory it does not have. A node is added here the same way as any
// other: one file under core/nodes/, one registration line below, and one wiring
// in core/workflows/.
#include "graph/graph_nodes.hpp"

#include "nodes/service.hpp"

namespace phi::media {

const NodeRegistry& media_registry() {
	static const NodeRegistry* reg = [] {
		auto* r = new NodeRegistry();
		register_media_nodes(*r);
		return r;
	}();
	return *reg;
}

// ── node registration ──────────────────────────────────────────────────────

void register_media_nodes(NodeRegistry& reg) {
	register_model_loader(reg);
	register_clip_loader(reg);
	register_vae_loader(reg);
	register_lora_loader(reg);
	register_qwen_image_edit_encode(reg);
	register_h3_ref2va_encode(reg);
	register_h3_sigma_shift(reg);
	register_qwen_image_cache(reg);
	register_latent(reg);
	register_scheduler(reg);
	register_sampler(reg);
	register_vae_decode(reg);
	register_vae_encode_image(reg);
	register_load_image(reg);
	register_save_image(reg);
	register_save_audio(reg);
	register_save_video(reg);
	register_seed(reg);
	register_const_int(reg);
	// music (ACE-Step 1.5)
	register_ace_text_encode(reg);
	register_ace_model_sampling(reg);
	// tts (Breeze-TTS-2)
	register_breeze_text_encode(reg);
	register_breeze_sampler(reg);
	register_breeze_decode(reg);
}

}  // namespace phi::media
