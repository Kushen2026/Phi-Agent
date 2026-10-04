// Node "MediaQwenImageEditEncode" — Text Encode Qwen Image 2.1 (edit).
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
//
// This is ComfyUI's `TextEncodeQwenImage21` (comfy_extras/nodes_qwen.py),
// transcribed node-for-node, because it is the node an *edit* actually depends
// on: it decides how big every reference is, what the text tower is shown, where
// each reference latent is spliced into the DiT's sequence, and what latent the
// sampler is handed. The reference's own steps are:
//
//   resolution        every reference is resized to about `resolution` x
//                     `resolution` pixels - aspect preserved, every edge snapped
//                     to a multiple of 32 - *independently of the canvas the
//                     caller is generating*. `0` keeps each reference at its own
//                     size, rounded to 32. This is what keeps a 4-megapixel
//                     canvas from charging 4 megapixels for each of three
//                     references: the reference cost is the `resolution` knob,
//                     not the output size.
//   resize            lanczos, aspect preserved (comfy.utils.common_upscale with
//                     "lanczos" / "disabled"), never upscaled beyond the rule
//                     above and never stretched to a neighbour's grid.
//   vision rows       the text tower sees RGB over white
//                     (`rgb = rgb * a + (1 - a)`), while the VAE keeps all four
//                     channels. `read_image` hands out three channels, so the
//                     composite is already the identity here - the note is kept
//                     because it is what the reference does.
//   reference latents one VAE encode per reference, at *that* reference's size;
//                     the DiT splices each block into the text stream at the
//                     slot its `<image_i>` marker occupied (`image_slots`).
//   latent output     the node returns conditioning *and* the latent to denoise,
//                     sized by the caller's `width`/`height` - the *only* source
//                     of the output geometry in this chain, references or not.
//                     (The reference node instead derives that latent from the
//                     first reference image, which makes an edit's output size a
//                     function of the material attached to it; phi's two tools
//                     take an explicit width/height and must honour it, so the
//                     geometry stays the caller's.) The noise itself is drawn by
//                     this node (seed), because phi's KSampler multiplies the
//                     latent it is handed by sigma0 and adds nothing of its own;
//                     ComfyUI's KSampler adds `noise * sigma0` to a zero latent,
//                     which is the same field.
//
// The prompt is presented with one `<imageN><|vision_start|><|image_pad|><|vision_end|>`
// block per reference at the head of the user turn (QwenImage21Tokenizer), which
// is `Qwen2Tokenizer::qwen_image_prompt`; the vision rows are dropped from the
// conditioning and replaced by the reference latents (keep_vision is false when
// a VAE is wired, which is exactly this chain's case).
#include "nodes/service.hpp"
#include "models/image_vae.hpp"
#include "models/text_encoder_8b.hpp"

namespace phi::media {

namespace {

// ComfyUI's resize rule for a reference (`TextEncodeQwenImage21.execute`), which
// is *not* the H3 chain's `ref_image_size`: the ratio comes from the source, the
// area from `resolution`, and both edges snap to 32 with a round (not a ceiling).
void qwen_edit_ref_size(i64 src_w, i64 src_h, i64 resolution, i64* out_w, i64* out_h) {
	if (resolution > 0) {
		const double ratio = (double)src_w / (double)src_h;
		const double r2 = (double)resolution * (double)resolution;
		*out_w = (i64)std::llround(std::sqrt(r2 * ratio) / 32.0) * 32;
		*out_h = (i64)std::llround(std::sqrt(r2 / ratio) / 32.0) * 32;
	} else {
		*out_w = (i64)std::llround((double)src_w / 32.0) * 32;
		*out_h = (i64)std::llround((double)src_h / 32.0) * 32;
	}
	*out_w = std::max<i64>(32, *out_w);
	*out_h = std::max<i64>(32, *out_h);
}

// 8-bit RGB HWC -> the VAE encoder's [4, H, W] fp32 in [-1, 1], alpha = 1 (the
// caller pads RGB with white, as `ImageVae::encode` documents).
void rgb_to_vae_input(const std::vector<unsigned char>& rgb, i64 w, i64 h,
                      std::vector<float>* out) {
	out->assign((size_t)(4 * w * h), 1.0f);
	for (i64 y = 0; y < h; y++)
		for (i64 x = 0; x < w; x++)
			for (i64 c = 0; c < 3; c++) {
				const float v = (float)rgb[(size_t)((y * w + x) * 3 + c)] / 255.0f;
				(*out)[(size_t)((c * h + y) * w + x)] = v * 2.0f - 1.0f;
			}
}

}  // namespace

void register_qwen_image_edit_encode(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaQwenImageEditEncode";
	n.category = "conditioning";
	n.title = "文本编码（Qwen-Image 2.1 编辑）";
	n.title_en = "Text Encode Qwen Image 2.1";
	n.description =
	    "把提示词 + 参考图编码成条件，并给出 width × height 的潜变量。"
	    "每张参考图按 resolution 独立缩放（32 对齐），各自以自身尺寸编码进 VAE，"
	    "并按 image_slots 插进 DiT 序列；参考图不影响输出尺寸（输出尺寸只看 width/height）。";
	n.description_en =
	    "Encodes a prompt plus reference images into conditioning, and hands back a latent at "
	    "width x height. Every reference is sized by `resolution` on its own (32 aligned), "
	    "encoded at its own size, and spliced into the DiT sequence at its slot; a reference "
	    "never influences the output size (that is width/height alone).";
	// Four wired image inputs cover the graph case (MediaLoadImage -> here); the
	// tool path passes absolute paths in `ref_images`, which follow the wired
	// slots in reference order. ComfyUI's Autogrow allows sixteen; phi's socket
	// list is static, so the params carry the tail.
	n.inputs = {port("clip", SocketType::Clip, false, "text encoder (Qwen3-VL-8B)"),
	            port("vae", SocketType::Vae, true, "image VAE: needed for the reference latents"),
	            port("image1", SocketType::Image, true, "reference 1 (optional; params fill the rest)"),
	            port("image2", SocketType::Image, true, "reference 2"),
	            port("image3", SocketType::Image, true, "reference 3"),
	            port("image4", SocketType::Image, true, "reference 4"),
	            port("seed", SocketType::Int, true, "override the latent's noise seed")};
	n.outputs = {port("CONDITIONING", SocketType::Conditioning, false, "positive + negative"),
	             port("LATENT", SocketType::Latent, false,
	                  "empty latent at the first reference's size")};
	{
		JsonValue p = JsonValue::object();
		p["prompt"] = "positive prompt";
		p["negative_prompt"] = "negative prompt (used when the sampler's cfg > 1)";
		p["ref_images"] = "list of reference image paths (after any wired image1..image4)";
		p["resolution"] =
		    "reference pixel area edge: each reference is resized to about resolution^2 px at "
		    "multiples of 32, aspect preserved (0 = keep each reference's own size, rounded to 32)";
		p["width"] = "output width (the latent's), independent of the references";
		p["height"] = "output height (the latent's), independent of the references";
		p["seed"] = "the noise seed of the latent output, used as given";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>& in) {
		auto clip = as_clip(in[0]);
		if (!clip) throw MediaError("QwenImageEditEncode: clip input is not a CLIP");
		if (clip->arch != "qwen3vl_8b")
			throw MediaError("QwenImageEditEncode: needs the Qwen3-VL-8B text encoder (got " +
			                 clip->arch + ")");
		auto vae = as_vae(in[1]);
		const std::string prompt = param_string(p, "prompt", "");
		if (prompt.empty()) throw MediaError("QwenImageEditEncode: prompt must not be empty");
		const i64 resolution = param_int(p, "resolution", 1024);
		if (resolution < 0) throw MediaError("QwenImageEditEncode: resolution must be >= 0");

		// ── the reference sources: wired images first, then the param paths ───
		//
		// A source is *named*, not decoded: the loop below reads one reference at
		// a time, resizes it, encodes it and lets the pixels go, so a nine-image
		// edit whose sources are 8000-px photographs does not hold nine decoded
		// copies (144 MB each) at once.
		struct Source {
			int wired = -1;        // index into `wired`, -1 for a path
			std::string path;
		};
		std::vector<std::shared_ptr<ImageData>> wired;
		std::vector<Source> srcs;
		for (size_t i = 2; i < in.size(); i++) {
			auto img = as_image(in[i]);
			if (!img) continue;
			Source src;
			src.wired = (int)wired.size();
			wired.push_back(std::move(img));
			srcs.push_back(std::move(src));
		}
		for (const std::string& rp : param_string_list(p, "ref_images")) {
			Source src;
			src.path = resolve_user_path(ctx, rp);
			srcs.push_back(std::move(src));
		}
		const i32 n_refs = (i32)srcs.size();

		// ── the text tower ────────────────────────────────────────────────────
		auto c = std::make_shared<ConditioningData>();
		c->arch = clip->arch;
		c->n_refs = n_refs;
		auto encode_prompt = [&](const std::string& text, std::vector<float>* out, i64* n_out) {
			const Qwen2Tokenizer::QwenImageIds q =
			    clip->tokenizer->encode_qwen_image_refs(text, n_refs);
			const std::vector<i32> tmpl = q.all;
			std::vector<float> h = clip->te8->encode(tmpl);
			const i64 hid = clip->te8->config().hidden;
			const i64 all = (i64)(h.size() / (size_t)hid);
			const i64 drop = q.drop_rows;
			if (drop >= all) throw MediaError("QwenImageEditEncode: the template lost every token");
			*n_out = (i64)q.ids.size();
			out->assign((size_t)(*n_out) * hid, 0.0f);
			i64 r = 0;
			for (i64 i = drop; i < all; i++) {
				// The vision rows are dropped: with a VAE wired the DiT carries the
				// references as latents (ComfyUI's keep_vision = false), and this
				// tower is text-only.
				if (tmpl[(size_t)i] == clip->tokenizer->id_image_pad()) continue;
				memcpy(&(*out)[(size_t)r * hid], &h[(size_t)i * hid], (size_t)hid * 4);
				r++;
			}
			if (r != *n_out)
				throw MediaError("QwenImageEditEncode: the template did not carry one vision "
				                 "block per reference");
		};
		encode_prompt(prompt, &c->positive, &c->n_pos);
		c->dim = clip->te8->config().hidden;
		const std::string neg = param_string(p, "negative_prompt", "");
		if (!neg.empty()) encode_prompt(neg, &c->negative, &c->n_neg);
		// The slots the DiT splices the reference blocks at. Taken from the
		// prompt's own template: the shape depends only on the reference count.
		c->slots = clip->tokenizer->encode_qwen_image_refs(prompt, n_refs).slots;

		// The text tower streamed its layers through the shared weights arena; hand
		// them back *before* the reference encodes, which need the arena for their
		// own chunks. On a small card this is the difference between the encode
		// fitting beside the text tower's last layer and not fitting at all (the
		// 8B tower's chunk is ~250 MB and the reference encode's is ~100 MB).
		release_shared_arenas(static_cast<GpuCtx*>(ctx.gpu));

		// ── the reference latents, each at its own resized size ──────────────
		//
		// These are conditioning only: they shape the sequence and the DiT's
		// reference blocks, never the output geometry (see the latent's note below).
		if (n_refs > 0) {
			// Required, not optional: the reference's own node can fall back to
			// conditioning through the text tower's *vision* path when no VAE is
			// wired (its `keep_vision = len(ref_latents) == 0` branch), but this
			// tower is text-only (see the note in text_encoder_8b.hpp). Refusing is
			// the honest answer - the alternative would be a picture the DiT has no
			// latent for, conditioned on nothing but its label.
			if (!vae || vae->role != "image")
				throw MediaError("QwenImageEditEncode: reference images need the image VAE on "
				                 "'vae' (this tower is text-only, so the references reach the "
				                 "DiT as VAE latents)");
			std::vector<unsigned char> decoded, resized;
			for (const Source& src : srcs) {
				const unsigned char* pix = nullptr;
				i64 sw = 0, sh = 0;
				if (src.wired >= 0) {
					const ImageData& img = *wired[(size_t)src.wired];
					pix = img.rgb.data();
					sw = img.w;
					sh = img.h;
				} else {
					int rw = 0, rh = 0;
					decoded = read_image(src.path, &rw, &rh);
					if (rw <= 0 || rh <= 0)
						throw MediaError("QwenImageEditEncode: reference image did not "
						                 "decode: " + src.path);
					pix = decoded.data();
					sw = rw;
					sh = rh;
				}
				i64 tw = 0, th = 0;
				qwen_edit_ref_size(sw, sh, resolution, &tw, &th);
				if (tw != sw || th != sh)
					resize_lanczos3(pix, (int)sw, (int)sh, (int)tw, (int)th, resized);
				else if (decoded.empty())
					resized.assign(pix, pix + (size_t)(sw * sh * 3));
				else
					resized = decoded;
				pix = resized.data();
				std::vector<float> rgba;
				rgb_to_vae_input(resized, tw, th, &rgba);
				c->ref_latents.push_back(vae->image->encode(rgba, th, tw));
				c->ref_hs.push_back(th / 16);
				c->ref_ws.push_back(tw / 16);
				c->ref_px.push_back((th / 16) * (tw / 16));
			}
			vae->image->release_activation_memory();
		}

		// ── the latent: the caller's `width` / `height`, and nothing else ─────
		//
		// The output size has exactly one source of truth in this chain, and it is
		// the request: with references, without them, editing or not, the canvas is
		// `width` x `height`. A reference's own size never decides it - a reference
		// is conditioning, and letting the first one's aspect pick the canvas would
		// make the same request produce a different picture depending on which
		// photograph was attached first (and would silently override a caller that
		// asked for 640x640). `resolution` sizes the *references* only; the fallback
		// below exists purely so a hand-wired graph that forgets `width`/`height`
		// still runs, and it is the same square `resolution` defaults to.
		i64 lat_w = param_int(p, "width", 0);
		i64 lat_h = param_int(p, "height", 0);
		if (lat_w <= 0 || lat_h <= 0) {
			lat_w = resolution > 0 ? resolution : 1024;
			lat_h = resolution > 0 ? resolution : 1024;
		}
		if (lat_w < 16 || lat_h < 16)
			throw MediaError("QwenImageEditEncode: width/height must be at least 16 px (got " +
			                 std::to_string(lat_w) + "x" + std::to_string(lat_h) + ")");
		auto lat = std::make_shared<LatentData>();
		lat->arch = "image";
		lat->seed = latent_seed(p, in.size() > 6 ? in[6] : Value{});   // `seed` is the 7th input
		lat->req_w = lat_w;
		lat->req_h = lat_h;
		lat->canvas_w = ceil_div(lat_w, 16) * 16;
		lat->canvas_h = ceil_div(lat_h, 16) * 16;
		lat->h = lat->canvas_h / 16;
		lat->w = lat->canvas_w / 16;
		lat->c = 64;   // Qwen-Image in_channels
		lat->x.assign((size_t)(lat->c * lat->h * lat->w), 0.0f);
		fill_normal32(lat->x.data(), lat->x.size(), lat->seed);

		// the text tower (and the reference encodes) streamed through the shared
		// arenas; hand them back so the sampler's residency plan reads the real card
		release_shared_arenas(static_cast<GpuCtx*>(ctx.gpu));
		return std::vector<Value>{Value::of_object(c), Value::of_object(lat)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
