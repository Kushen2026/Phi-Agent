// The audio-only chain: the same H3 front half as the video chain, only the tail
// swapped to VaeDecodeAudio + SaveAudio.
#include "graph/graph_workflows.hpp"
#include "workflows/common.hpp"

namespace phi::media {

using namespace workflows;

Graph build_audio_workflow(const VideoWorkflowParams& p) {
	Graph g;
	g.name = "minimax_h3_audio";

	// ── the same front half every H3 chain uses ────────────────────────────
	GraphNode model = gn(1, "MediaModelLoader");
	model.params["arch"] = "minimax_h3";
	put_str(model.params, "model", p.model);

	GraphNode clip = gn(2, "MediaClipLoader");
	clip.params["arch"] = "qwen3vl_32b";
	put_str(clip.params, "model", p.clip);
	put_str(clip.params, "vocab", p.tokenizer_vocab);
	put_str(clip.params, "merges", p.tokenizer_merges);
	put_str(clip.params, "tokenizer_config", p.tokenizer_config);

	GraphNode enc = gn(3, "MediaH3Ref2VaEncode");
	enc.params["prompt"] = p.prompt;
	enc.params["width"] = (int64_t)p.width;
	enc.params["height"] = (int64_t)p.height;
	enc.params["frames"] = (int64_t)p.frames;

	GraphNode latent = gn(4, "MediaEmptyLatent");
	latent.params["arch"] = "video";
	latent.params["frames"] = (int64_t)p.frames;
	latent.params["width"] = (int64_t)p.width;
	latent.params["height"] = (int64_t)p.height;
	latent.params["fps"] = (int64_t)p.fps;
	// As in the video chain: the seed is the caller's, verbatim.
	latent.params["seed"] = (int64_t)p.seed;

	// The same ModelSamplingMiniMaxH3 node the video chain uses: the audio stream
	// rides the video schedule, so the shift has to be set the same way here.
	GraphNode shift = gn(10, "MediaH3SigmaShift");
	shift.params["shift_video"] = p.h3_shift_video > 0 ? (double)p.h3_shift_video : 12.0;
	shift.params["shift_audio"] = p.h3_shift_audio > 0 ? (double)p.h3_shift_audio : 3.0;

	GraphNode sched = gn(5, "MediaScheduler");
	sched.params["scheduler"] = p.scheduler.empty() ? std::string("beta") : p.scheduler;
	sched.params["steps"] = (int64_t)p.steps;

	GraphNode sampler = gn(6, "MediaSampler");
	sampler.params["sampler"] = p.sampler.empty() ? std::string("res_multistep") : p.sampler;

	// ── the audio-only tail (this is the *only* difference from the video chain) ──
	GraphNode avae = gn(7, "MediaVaeLoader");
	avae.params["role"] = "audio";
	put_str(avae.params, "model", p.vae_audio);

	GraphNode adec = gn(8, "MediaVaeDecode");

	GraphNode save = gn(9, "MediaSaveAudio");
	save.params["path"] = p.output.empty() ? "out.wav" : p.output;

	g.nodes = {model, clip, enc, latent, sched, sampler, avae, adec, save, shift};
	g.links = {
	    {2, "CLIP", 3, "clip"},
	    {1, "MODEL", 10, "model"},
	    {10, "MODEL", 5, "model"},
	    {10, "MODEL", 6, "model"},
	    {3, "CONDITIONING", 6, "conditioning"},
	    {4, "LATENT", 6, "latent"},
	    {5, "SIGMAS", 6, "sigmas"},
	    {6, "LATENT", 8, "latent"},
	    {7, "VAE", 8, "vae"},
	    {8, "out", 9, "audio"},
	};
	g.outputs = {{9, "path"}};
	apply_loras(g, p.loras, 1, 2);
	return g;
}

}  // namespace phi::media
