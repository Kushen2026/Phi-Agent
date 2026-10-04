// The image chain: Model / Clip / Vae loaders -> TextEncode -> Sampler ->
// VaeDecode(role=image) -> SaveImage. One chain, one file; the nodes are shared.
//
// The sampler and the scheduler are the tool-settings pair
// (settings.json -> tools.media.image_sampler / image_scheduler). Empty strings
// keep the released workflow's own "euler" + "simple"; the nodes validate any
// other name against the ported ComfyUI tables and refuse what they do not
// implement, so a typo is an error and not a silently different picture.
#include "graph/graph_workflows.hpp"
#include "workflows/common.hpp"

namespace phi::media {

using namespace workflows;

Graph build_image_workflow(const ImageWorkflowParams& p) {
	Graph g;
	g.name = "qwen_image_2.1";

	GraphNode model = gn(1, "MediaModelLoader");
	model.params["arch"] = "qwen_image";
	put_str(model.params, "model", p.model);

	GraphNode clip = gn(2, "MediaClipLoader");
	clip.params["arch"] = "qwen3vl_8b";
	put_str(clip.params, "model", p.clip);
	put_str(clip.params, "vocab", p.tokenizer_vocab);
	put_str(clip.params, "merges", p.tokenizer_merges);
	put_str(clip.params, "tokenizer_config", p.tokenizer_config);

	GraphNode vae = gn(3, "MediaVaeLoader");
	vae.params["role"] = "image";
	put_str(vae.params, "model", p.vae);

	// The encode node is ComfyUI's `TextEncodeQwenImage21`: it sizes every
	// reference on its own (`ref_resolution`), builds their latents, and hands
	// back the latent the sampler denoises. Its geometry is the caller's
	// `width`/`height` - the single source of the output size in this chain,
	// references or not - so the node's LATENT is what the sampler reads and the
	// graph needs no separate latent node: one place decides the canvas, and the
	// request is passed through to it verbatim.
	GraphNode enc = gn(4, "MediaQwenImageEditEncode");
	enc.params["prompt"] = p.prompt;
	put_str(enc.params, "negative_prompt", p.negative_prompt);
	enc.params["width"] = (int64_t)p.width;
	enc.params["height"] = (int64_t)p.height;
	// `resolution` sizes the *references* only; the output is width x height. It is
	// passed through as-is, including 0 - the node's "keep each reference at its own
	// size, rounded to 32" setting, which defaulting it to 1024 here would silently
	// turn into an upscale of every small reference.
	enc.params["resolution"] = (int64_t)p.ref_resolution;
	enc.params["seed"] = (int64_t)p.seed;
	{
		JsonValue arr = JsonValue::array();
		for (const std::string& s : p.ref_images) arr.push_back(s);
		if (!p.ref_images.empty()) enc.params["ref_images"] = arr;
	}

	// ModelSamplingDual / QwenImage21Cache: the prefix K/V cache, when asked for.
	// Like the H3 chain's shift node it patches the MODEL socket, so it sits
	// between the loader and everything that reads the backbone - and a chain that
	// does not ask for it has no node at all.
	GraphNode cache = gn(13, "MediaQwenImage21Cache");

	GraphNode sched = gn(6, "MediaScheduler");
	sched.params["scheduler"] = p.scheduler.empty() ? std::string("simple") : p.scheduler;
	sched.params["steps"] = (int64_t)p.steps;

	GraphNode sampler = gn(7, "MediaSampler");
	sampler.params["cfg"] = (double)p.cfg;
	sampler.params["sampler"] = p.sampler.empty() ? std::string("euler") : p.sampler;

	GraphNode decode = gn(8, "MediaVaeDecode");

	GraphNode save = gn(9, "MediaSaveImage");
	save.params["path"] = p.output;

	const int model_src = p.prefix_cache ? 13 : 1;
	g.nodes = {model, clip, vae, enc, sched, sampler, decode, save};
	// The cache node, when asked for, sits between the loader and everything that
	// reads the backbone: one node in front of the two consumers, exactly like the
	// LoRA chain. Not asked for, not in the graph at all.
	if (p.prefix_cache) g.nodes.push_back(cache);
	g.links = {
	    {2, "CLIP", 4, "clip"},
	    {3, "VAE", 4, "vae"},
	    {model_src, "MODEL", 6, "model"},
	    {model_src, "MODEL", 7, "model"},
	    {4, "CONDITIONING", 7, "conditioning"},
	    {4, "LATENT", 7, "latent"},
	    {6, "SIGMAS", 7, "sigmas"},
	    {3, "VAE", 8, "vae"},
	    {7, "LATENT", 8, "latent"},
	    {8, "out", 9, "image"},
	};
	if (p.prefix_cache) link(g, 1, "MODEL", 13, "model");
	g.outputs = {{9, "path"}};
	apply_loras(g, p.loras, 1, 2);
	return g;
}

}  // namespace phi::media
