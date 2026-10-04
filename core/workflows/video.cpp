// The video chain: the H3 joint A/V model -> H3Sampler ->
// VaeDecodeVideo + VaeDecodeAudio -> SaveVideo.
// The video chain: the H3 joint A/V model -> H3Sampler ->
// VaeDecodeVideo + VaeDecodeAudio -> SaveVideo.
//
// The SCHEDULER node's grid is the only schedule in this chain: the sampler is
// handed it verbatim, so the sigma the DiT is told and the sigma the integrator
// advances on are the same numbers. The pair comes from the tool settings
// (settings.json -> tools.media.video_sampler / video_scheduler); empty keeps the
// released workflow's res_multistep + beta. The H3 joint walk integrates both
// streams at once, so its sampler setting selects between the reference's
// res_multistep and plain euler rather than a generic per-tensor loop.
#include "graph/graph_workflows.hpp"
#include "workflows/common.hpp"

namespace phi::media {

using namespace workflows;

Graph build_video_workflow(const VideoWorkflowParams& p) {
	Graph g;
	g.name = "minimax_h3_ref2va";

	GraphNode model = gn(1, "MediaModelLoader");
	model.params["arch"] = "minimax_h3";
	put_str(model.params, "model", p.model);

	GraphNode clip = gn(2, "MediaClipLoader");
	clip.params["arch"] = "qwen3vl_32b";
	put_str(clip.params, "model", p.clip);
	put_str(clip.params, "vocab", p.tokenizer_vocab);
	put_str(clip.params, "merges", p.tokenizer_merges);
	put_str(clip.params, "tokenizer_config", p.tokenizer_config);

	GraphNode vvae = gn(3, "MediaVaeLoader");
	vvae.params["role"] = "video";
	put_str(vvae.params, "model", p.vae);

	GraphNode avae = gn(4, "MediaVaeLoader");
	avae.params["role"] = "audio";
	put_str(avae.params, "model", p.vae_audio);

	// ComfyUI's `MiniMaxH3ReferenceToVideo`: labels, the 2 fps video presentation,
	// the reference latents and the caps all live in this one node.
	GraphNode enc = gn(5, "MediaH3Ref2VaEncode");
	enc.params["prompt"] = p.prompt;
	enc.params["width"] = (int64_t)p.width;
	enc.params["height"] = (int64_t)p.height;
	enc.params["frames"] = (int64_t)p.frames;
	enc.params["ref_image_size"] = p.ref_image_size.empty() ? "match" : p.ref_image_size;
	auto put_list = [](JsonValue& obj, const char* key, const std::vector<std::string>& v) {
		if (v.empty()) return;
		JsonValue arr = JsonValue::array();
		for (const std::string& s : v) arr.push_back(s);
		obj[key] = arr;
	};
	put_list(enc.params, "ref_images", p.ref_images);
	put_list(enc.params, "ref_audios", p.ref_audios);
	put_list(enc.params, "ref_videos", p.ref_videos);
	put_list(enc.params, "ref_video_audios", p.ref_video_audios);

	GraphNode latent = gn(6, "MediaEmptyLatent");
	latent.params["arch"] = "video";
	latent.params["frames"] = (int64_t)p.frames;
	latent.params["width"] = (int64_t)p.width;
	latent.params["height"] = (int64_t)p.height;
	latent.params["fps"] = (int64_t)p.fps;
	// As in the image chain: the seed is the caller's, verbatim.
	latent.params["seed"] = (int64_t)p.seed;

	// ModelSamplingMiniMaxH3: the flow shifts the grid is built from and the DiT
	// applies. The released 12 / 3 is the loader's default; a distilled LoRA's own
	// shift is what this node exists for.
	GraphNode shift = gn(12, "MediaH3SigmaShift");
	shift.params["shift_video"] = p.h3_shift_video > 0 ? (double)p.h3_shift_video : 12.0;
	shift.params["shift_audio"] = p.h3_shift_audio > 0 ? (double)p.h3_shift_audio : 3.0;

	GraphNode sched = gn(7, "MediaScheduler");
	sched.params["scheduler"] = p.scheduler.empty() ? std::string("beta") : p.scheduler;
	sched.params["steps"] = (int64_t)p.steps;

	GraphNode sampler = gn(8, "MediaSampler");
	sampler.params["sampler"] = p.sampler.empty() ? std::string("res_multistep") : p.sampler;

	GraphNode vdec = gn(9, "MediaVaeDecode");
	GraphNode adec = gn(10, "MediaVaeDecode");

	GraphNode save = gn(11, "MediaSaveVideo");
	save.params["path"] = p.output;

	g.nodes = {model, clip, vvae, avae, enc, latent, shift, sched, sampler, vdec, adec, save};
	g.links = {
	    {2, "CLIP", 5, "clip"},
	    {3, "VAE", 5, "vae"},
	    {4, "VAE", 5, "vae_audio"},
	    {1, "MODEL", 12, "model"},
	    {12, "MODEL", 7, "model"},
	    {12, "MODEL", 8, "model"},
	    {5, "CONDITIONING", 8, "conditioning"},
	    {6, "LATENT", 8, "latent"},
	    {7, "SIGMAS", 8, "sigmas"},
	    {3, "VAE", 9, "vae"},
	    {8, "LATENT", 9, "latent"},
	    {4, "VAE", 10, "vae"},
	    {8, "LATENT", 10, "latent"},
	    {9, "out", 11, "video"},
	    {10, "out", 11, "audio"},
	};
	g.outputs = {{11, "path"}};
	apply_loras(g, p.loras, 1, 2);
	return g;
}

}  // namespace phi::media
