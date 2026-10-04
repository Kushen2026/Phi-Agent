// The music chain: ACE-Step 1.5, text-to-music.
//
// Node-for-node the graph `ACE-Step-音乐生成.json` wires:
//
//   UNETLoader            -> MediaModelLoader      (arch "ace_step15")
//   DualCLIPLoader        -> MediaClipLoader       (arch "ace15": 0.6B + 4B)
//   VAELoader             -> MediaVaeLoader        (role "ace_audio")
//   TextEncodeAceStep1.5  -> MediaAceTextEncode
//   ModelSamplingAuraFlow -> MediaAceModelSampling
//   (KSampler's sigmas)   -> MediaScheduler        (scheduler "simple")
//   EmptyAceStep1.5Latent -> MediaEmptyLatent   (arch "music")
//   KSampler(euler)       -> MediaSampler       (the ACE arch's own loop)
//   VAEDecodeAudio        -> MediaVaeDecode     (role "ace_audio")
//   SaveAudioMP3          -> MediaSaveAudio        (16-bit WAV, like every chain)
//
// The KSampler's two widgets are the tool settings (settings.json ->
// tools.media.music_sampler / music_scheduler): the released workflow runs
// euler + simple, which is what an empty setting keeps.
//
// Nothing here is a special path: every stage is a registered node, so editing
// the chain (another step count, a LoRA, a distilled shift) is editing this
// wiring rather than the engine.
#include "graph/graph_workflows.hpp"
#include "workflows/common.hpp"

namespace phi::media {

using namespace workflows;

Graph build_music_workflow(const MusicWorkflowParams& p) {
	Graph g;
	g.name = "ace_step_1.5";

	GraphNode model = gn(1, "MediaModelLoader");
	model.params["arch"] = "ace_step15";
	put_str(model.params, "model", p.model);

	// The two towers the DualCLIPLoader loads: the 0.6B embedder and the 4B
	// audio-code LM. They are one CLIP socket because the text encode node needs
	// both to produce the conditioning.
	GraphNode clip = gn(2, "MediaClipLoader");
	clip.params["arch"] = "ace15";
	put_str(clip.params, "model", p.clip);
	put_str(clip.params, "model_lm", p.clip_lm);
	put_str(clip.params, "vocab", p.tokenizer_vocab);
	put_str(clip.params, "merges", p.tokenizer_merges);
	put_str(clip.params, "tokenizer_config", p.tokenizer_config);

	GraphNode vae = gn(3, "MediaVaeLoader");
	vae.params["role"] = "ace_audio";
	put_str(vae.params, "model", p.vae);

	GraphNode enc = gn(4, "MediaAceTextEncode");
	enc.params["tags"] = p.tags;
	enc.params["lyrics"] = p.lyrics;
	enc.params["bpm"] = (int64_t)p.bpm;
	enc.params["duration"] = p.duration;
	// The latent length uses the same duration unless the caller overrides it, so
	// the DiT's context and the sampler's noise agree by construction.
	enc.params["seconds"] = p.duration;
	enc.params["timesignature"] = p.timesignature;
	enc.params["language"] = p.language;
	enc.params["keyscale"] = p.keyscale;
	enc.params["seed"] = (int64_t)p.seed;
	enc.params["generate_audio_codes"] = p.generate_audio_codes;
	enc.params["cfg_scale"] = p.lm_cfg_scale;
	enc.params["temperature"] = p.lm_temperature;
	enc.params["top_p"] = p.lm_top_p;
	enc.params["top_k"] = (int64_t)p.lm_top_k;
	enc.params["min_p"] = p.lm_min_p;

	// ModelSamplingAuraFlow: it patches the MODEL, so everything that reads the
	// backbone (the scheduler and the sampler) sits downstream of it.
	GraphNode shift = gn(5, "MediaAceModelSampling");

	GraphNode sched = gn(6, "MediaScheduler");
	sched.params["scheduler"] = p.scheduler.empty() ? std::string("simple") : p.scheduler;
	sched.params["steps"] = (int64_t)p.steps;

	GraphNode latent = gn(7, "MediaEmptyLatent");
	latent.params["arch"] = "music";
	latent.params["seconds"] = p.duration;
	latent.params["seed"] = (int64_t)p.seed;

	GraphNode sampler = gn(8, "MediaSampler");
	sampler.params["cfg"] = 1.0;
	sampler.params["sampler"] = p.sampler.empty() ? std::string("euler") : p.sampler;

	GraphNode decode = gn(9, "MediaVaeDecode");

	GraphNode save = gn(10, "MediaSaveAudio");
	save.params["path"] = p.output.empty() ? "music.wav" : p.output;

	g.nodes = {model, clip, vae, enc, shift, sched, latent, sampler, decode, save};
	g.links = {
	    {2, "CLIP", 4, "clip"},
	    {1, "MODEL", 4, "model"},
	    {1, "MODEL", 5, "model"},
	    {5, "MODEL", 6, "model"},
	    {6, "SIGMAS", 8, "sigmas"},
	    {5, "MODEL", 8, "model"},
	    {4, "CONDITIONING", 8, "conditioning"},
	    {7, "LATENT", 8, "latent"},
	    {3, "VAE", 9, "vae"},
	    {8, "LATENT", 9, "latent"},
	    {9, "out", 10, "audio"},
	};
	g.outputs = {{10, "path"}};
	return g;
}

}  // namespace phi::media
