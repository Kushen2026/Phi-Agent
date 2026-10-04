// Node "MediaH3Ref2VaEncode" — MiniMax H3 Reference to Video (ref2va).
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
//
// This is ComfyUI's `MiniMaxH3ReferenceToVideo` (comfy_extras/nodes_minimax_h3.py)
// together with the presentation its tokenizer owns (comfy/text_encoders/minimax.py),
// because the two are one contract: what the released checkpoint was trained on is
// the *pair* of them. The rules, in the reference's own order:
//
//   order        references enter the presentation as images, then videos (each
//                soundtrack's `<Audio j>` label immediately before its
//                `<Video k>`), then standalone audio; ordinals are 1-based per
//                type, so the prompt refers to `<Picture i>` / `<Video k>` /
//                `<Audio j>`.
//   images       down-only scaled: "match" to the generation's pixel area, or
//                "max" to the reference pipeline's 2048 px short edge (slower -
//                reference tokens ride through every sampling step - but that is
//                what the identity-fidelity workflow asks for).
//   videos       the clip is sampled at 2 fps for the text tower with one
//                timestamp per frame *pair* (`<T.T seconds>`), the pair filling
//                one temporal patch; the same clip is encoded by the video VAE at
//                24 fps for the DiT's reference rows, cut to the generation's
//                length and snapped to the legal 17k+5 frame count.
//   audio        contributes its label to the tower and its latent (through the
//                audio VAE) to the DiT; a soundtrack attached to a reference
//                video joins *that* block, so the audio rows share its timeline
//                origin - which is why the label order puts it first.
//
// The reference latents are built here because the VAEs are wired inputs; that is
// what makes this node the whole ref2va chain's front half rather than a widget.
//
// Cost note: the presentation holds one patch matrix per visual reference until
// the text tower runs (the prompt has to be assembled whole before the tower sees
// it), which is ~50 MB per 2048-px reference image and a few MB per sampled video
// pair. The references are capped at the released node's maxima (9 / 3 / 3) and
// `ref_image_size = "match"` - the tool's default - keeps an image near the
// generation's own pixel count, so the ordinary case is a few tens of megabytes
// rather than the "max" case's hundreds.
#include <algorithm>

#include "nodes/service.hpp"
#include "models/audio_vae.hpp"
#include "models/text_encoder_32b.hpp"
#include "models/video_vae.hpp"

namespace phi::media {

void register_h3_ref2va_encode(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaH3Ref2VaEncode";
	n.category = "conditioning";
	n.title = "文本编码（MiniMax H3 参考生视频）";
	n.title_en = "MiniMax H3 Reference to Video";
	n.description =
	    "把提示词 + 参考图/参考视频/参考音频编码成 H3 的条件。参考图按 ref_image_size 缩放，"
	    "参考视频按 2 fps 采样并逐对帧写时间戳（一对帧填一个时间 patch），参考音频只写标签；"
	    "参考潜变量由视频/音频 VAE 编码。提示词里用 <Picture i> / <Video k> / <Audio j> 引用。";
	n.description_en =
	    "Encodes a prompt with reference images / videos / audio into H3 conditioning. Images are "
	    "sized by ref_image_size, videos are sampled at 2 fps with one timestamp per frame pair "
	    "(a pair fills one temporal patch), audio contributes its label; the reference latents "
	    "come from the video/audio VAEs. Refer to them as <Picture i> / <Video k> / <Audio j>.";
	n.inputs = {port("clip", SocketType::Clip, false, "text encoder (Qwen3-VL-32B)"),
	            port("vae", SocketType::Vae, true, "video VAE (needed for image/video refs)"),
	            port("vae_audio", SocketType::Vae, true, "audio VAE (needed for audio refs)")};
	n.outputs = {port("CONDITIONING", SocketType::Conditioning, false, "the reference conditioning")};
	{
		JsonValue p = JsonValue::object();
		p["prompt"] = "the prompt (<Picture i> / <Video k> / <Audio j> tags)";
		p["width"] = "generation width (the canvas references are sized against)";
		p["height"] = "generation height";
		p["frames"] = "generation frames at 24 fps";
		p["ref_image_size"] = "match | max";
		p["ref_images"] = "reference image paths, in <Picture i> order";
		p["ref_videos"] = "reference video paths, in <Video k> order";
		p["ref_video_audios"] =
		    "soundtracks of the same-numbered reference videos (ref_video_audio_N for ref_video_N)";
		p["ref_audios"] = "standalone reference audio paths, in <Audio j> order";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>& in) {
		auto clip = as_clip(in[0]);
		if (!clip) throw MediaError("H3Ref2VaEncode: clip input is not a CLIP");
		if (!clip->te32) throw MediaError("H3Ref2VaEncode: needs the Qwen3-VL-32B text encoder");
		auto vae_v = as_vae(in[1]);
		auto vae_a = as_vae(in[2]);
		const std::string prompt = param_string(p, "prompt", "");
		if (prompt.empty()) throw MediaError("H3Ref2VaEncode: prompt must not be empty");

		// The generation's own geometry. It comes from the same planner the latent
		// node uses, so the canvas the references are sized against and the frame
		// count they are cut to cannot disagree with the clip being made.
		const i64 width = param_int(p, "width", 960);
		const i64 height = param_int(p, "height", 540);
		const i64 frames = param_int(p, "frames", 121);
		const VideoPlan plan = plan_video(width, height, frames, 24);
		const bool use_max = param_string(p, "ref_image_size", "match") == "max";

		std::vector<std::string> ref_images, ref_videos, ref_video_audios, ref_audios;
		for (const std::string& s : param_string_list(p, "ref_images"))
			if (!s.empty()) ref_images.push_back(resolve_user_path(ctx, s));
		for (const std::string& s : param_string_list(p, "ref_videos"))
			if (!s.empty()) ref_videos.push_back(resolve_user_path(ctx, s));
		// Index-paired soundtracks: `ref_video_audio_N` belongs to `ref_video_N`,
		// which is how the released node pairs its two Autogrow inputs.
		for (const std::string& s : param_string_list(p, "ref_video_audios"))
			ref_video_audios.push_back(s.empty() ? std::string() : resolve_user_path(ctx, s));
		for (const std::string& s : param_string_list(p, "ref_audios"))
			if (!s.empty()) ref_audios.push_back(resolve_user_path(ctx, s));
		// The released node's caps (Autogrow maxima): 9 images, 3 videos, 3 audios.
		if ((i64)ref_images.size() > 9)
			throw MediaError("H3Ref2VaEncode: at most 9 reference images");
		if ((i64)ref_videos.size() > 3)
			throw MediaError("H3Ref2VaEncode: at most 3 reference videos");
		if (ref_video_audios.size() > ref_videos.size())
			throw MediaError("H3Ref2VaEncode: a reference-video soundtrack has no matching video");
		if ((i64)ref_audios.size() > 3)
			throw MediaError("H3Ref2VaEncode: at most 3 reference audio clips");

		auto c = std::make_shared<ConditioningData>();
		c->arch = clip->arch;
		const bool has_refs = !ref_images.empty() || !ref_videos.empty() || !ref_audios.empty();

		PromptTokens pt;
		if (has_refs) {
			// Each VAE is required only by the references that need it, exactly as the
			// released node keeps both optional: with no image/video reference a chain
			// needs no video VAE, and with no audio one it needs no audio VAE.
			const bool need_video = !ref_images.empty() || !ref_videos.empty();
			const bool need_audio = !ref_audios.empty() ||
			                        std::any_of(ref_video_audios.begin(), ref_video_audios.end(),
			                                    [](const std::string& s) { return !s.empty(); });
			if (need_video && (!vae_v || vae_v->role != "video"))
				throw MediaError(
				    "H3Ref2VaEncode: reference images/videos need the video VAE on 'vae'");
			if (need_audio && (!vae_a || vae_a->role != "audio"))
				throw MediaError("H3Ref2VaEncode: reference audio needs the audio VAE on "
				                 "'vae_audio'");

			// What the reference build leaves resident on the card (the video VAE's
			// weights and the vision arenas that survive the release below). The H3
			// sampler hands this to plan_residency as `extra_resident`, the way
			// video_gen did: a plan that ignores it sizes the window for a card the
			// run does not have.
			const u64 ledger_before = vram_budget().local();
			i32 pic = 0, aud = 0;
			for (const std::string& path : ref_images) {
				const RefImagePixels px =
				    load_ref_image(path, plan.frame.canvas_w, plan.frame.canvas_h, use_max);
				append_ref_image_block(pt, ++pic, px);
				build_image_ref(*c, px, *vae_v->video);
			}
			i32 vid = 0;
			for (size_t k = 0; k < ref_videos.size(); k++) {
				const RefVideoFrames vf = load_ref_clip(ref_videos[k], plan.frames);
				const std::string track =
				    k < ref_video_audios.size() ? ref_video_audios[k] : std::string();
				if (!track.empty()) append_ref_audio_label(pt, ++aud);
				append_ref_video_blocks(pt, ++vid, vf);
				build_video_ref(*c, vf, track, vae_a ? vae_a->audio.get() : nullptr,
				                *vae_v->video);
			}
			for (const std::string& path : ref_audios) {
				append_ref_audio_label(pt, ++aud);
				build_audio_ref(*c, path, *vae_a->audio);
			}
			c->link_refs();
			if (vae_v && vae_v->video) vae_v->video->release_activation_memory();
			c->ref_footprint =
			    vram_budget().local() > ledger_before ? vram_budget().local() - ledger_before : 0;
		}

		// The prompt tokenizer is *this chain's*: `append_prompt_text` /
		// `append_ref_*` tokenise through vision_prep, which used to read a single
		// hard-coded vocabulary directory. Point it at the vocabulary the CLIP
		// socket was loaded with, so the ids the 32B tower embeds are the ids this
		// chain's tokenizer role produced.
		if (!clip->tokenizer_files.vocab.empty())
			set_text_tokenizer_files(clip->tokenizer_files.vocab, clip->tokenizer_files.merges,
			                         clip->tokenizer_files.config);

		// The prompt closes the presentation. With no references at all there is no
		// wrapper either: the ids are the plain prompt (ComfyUI's t2va path).
		append_prompt_text(pt, prompt);
		std::vector<i32> ids = pt.ids;
		if (ids.empty()) ids.push_back(clip->tokenizer->id_endoftext());
		c->tags = pt.tags;
		if (c->tags.empty()) c->tags.assign(ids.size(), 1);
		c->embed = pt.blocks.empty() ? clip->te32->encode(ids) : clip->te32->encode_vl(pt);
		if ((i64)c->embed.size() != (i64)ids.size() * clip->te32->config().hidden)
			throw MediaError("H3Ref2VaEncode: the text encoder returned an unexpected shape");
		c->n_ids = (i64)ids.size();
		c->text_dim = clip->te32->config().hidden;
		// Phase boundary: the 32B tower and the vision tower streamed through the
		// shared arenas, so they go back before the DiT plans its window.
		release_shared_arenas(static_cast<GpuCtx*>(ctx.gpu));
		return std::vector<Value>{Value::of_object(c)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
