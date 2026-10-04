// Node "MediaVaeDecode" — one VAE decode for every chain.
//
// ComfyUI has exactly one `VAEDecode` node and one `VAEDecodeAudio`; the model
// on the VAE socket decides what the decode *is*, not the node type. This engine
// used to carry one node per model (`MediaVaeDecodeImage`, `MediaVaeDecodeVideo`,
// `MediaVaeDecodeAudio`, `MediaAceVaeDecode`) even though every one of them is
// the same two-input, one-output stage: a VAE and a LATENT in, a decoded medium
// out. That is the duplication the node layer is meant to remove, so this file
// is now the only VAE-decode node - and it dispatches on `VaeData::role`, exactly
// the way `MediaScheduler` dispatches on `ModelData::arch`.
//
// The output socket is `Any`: the concrete type is the payload's (IMAGE, VIDEO
// or AUDIO), which is what the executor actually puts on the wire (`Value`'s
// type comes from the object), so a downstream IMAGE/VIDEO/AUDIO input is
// wired correctly whatever the role.
//
// One node, one file. Registered here; the shared helpers live in service.hpp,
// the catalogue entry point in registry.cpp.
#include "nodes/service.hpp"

#include "models/ace_vae.hpp"
#include "models/audio_vae.hpp"
#include "models/image_vae.hpp"
#include "models/video_vae.hpp"

namespace phi::media {

namespace {

// ── image (Qwen-Image 2.1) ─────────────────────────────────────────────────
std::shared_ptr<ImageData> decode_image(GraphContext& ctx, const std::shared_ptr<VaeData>& vae,
                                        const std::shared_ptr<LatentData>& lat) {
	const i64 H = lat->h, W = lat->w, C = lat->c;
	std::vector<float> x = lat->x;
	const ImageVaeConfig& vc = vae->image->config();
	if (vc.std_.size() == (size_t)C && vc.mean.size() == (size_t)C) {
		for (i64 c = 0; c < C; c++) {
			const float sd = vc.std_[(size_t)c], mn = vc.mean[(size_t)c];
			float* q = &x[(size_t)c * H * W];
			for (i64 k = 0; k < H * W; k++) q[k] = q[k] * sd + mn;
		}
	}
	std::vector<float> img = vae->image->decode(x, H, W);
	const i64 cw = lat->canvas_w, chh = lat->canvas_h;
	std::vector<unsigned char> full;
	float_to_srgb8(img.data(), (int)cw, (int)chh, 3, full);
	auto out = std::make_shared<ImageData>();
	crop_centre(full.data(), (int)cw, (int)chh, (int)lat->req_w, (int)lat->req_h, out->rgb);
	out->w = lat->req_w;
	out->h = lat->req_h;
	vae->image->release_activation_memory();
	release_shared_arenas(static_cast<GpuCtx*>(ctx.gpu));
	return out;
}

// ── video (MiniMax H3) ─────────────────────────────────────────────────────
std::shared_ptr<VideoData> decode_video(GraphContext& ctx, const std::shared_ptr<VaeData>& vae,
                                        const std::shared_ptr<LatentData>& lat) {
	std::vector<float> planes = vae->video->decode(lat->xv, lat->vt, lat->lat_h, lat->lat_w);
	auto out = std::make_shared<VideoData>();
	out->planes = std::move(planes);
	out->frames = lat->plan.frames;
	out->h = lat->plan.frame.canvas_h;
	out->w = lat->plan.frame.canvas_w;
	out->out_w = lat->plan.frame.out_w;
	out->out_h = lat->plan.frame.out_h;
	out->fps = lat->plan.fps;
	vae->video->release_activation_memory();
	release_shared_arenas(static_cast<GpuCtx*>(ctx.gpu));
	return out;
}

// ── audio (H3 joint A/V) ───────────────────────────────────────────────────
std::shared_ptr<AudioData> decode_audio(const std::shared_ptr<VaeData>& vae,
                                        const std::shared_ptr<LatentData>& lat) {
	const i64 at = lat->at;
	const float scale = lat->audio_scale > 0 ? lat->audio_scale : 4.0f;
	std::vector<float> chan((size_t)32 * at);
	std::vector<float> pl, pr;
	for (i64 ch = 0; ch < 2; ch++) {
		for (i64 c = 0; c < 32; c++)
			for (i64 t = 0; t < at; t++)
				chan[(size_t)(c * at + t)] = lat->xa[(size_t)((c * 2 + ch) * at + t)];
		h3_av_scale_audio(chan.data(), 32 * at, 1.0f / scale);
		std::vector<float> pcm = vae->audio->decode(chan, at);
		(ch == 0 ? pl : pr) = std::move(pcm);
	}
	const i64 n = (i64)std::max(pl.size(), pr.size());
	auto out = std::make_shared<AudioData>();
	out->sample_rate = vae->audio->config().sample_rate > 0 ? vae->audio->config().sample_rate
	                                                       : 32000;
	out->channels = 2;
	out->pcm.assign((size_t)n * 2, 0.0f);
	for (i64 i = 0; i < n; i++) {
		out->pcm[(size_t)(i * 2 + 0)] = i < (i64)pl.size() ? pl[(size_t)i] : 0.0f;
		out->pcm[(size_t)(i * 2 + 1)] = i < (i64)pr.size() ? pr[(size_t)i] : 0.0f;
	}
	return out;
}

// ── music (ACE-Step 1.5 audio VAE) ─────────────────────────────────────────
//
// The music chain's tail: the [64, T] latent the sampler produced becomes 48 kHz
// PCM. `PHI_ACE_LATENT_IN=<file>` still replaces the latent with 64*T raw
// little-endian float32s (channel-major), the conformance seam the VAE is
// checked through.
std::shared_ptr<AudioData> decode_music(const std::shared_ptr<VaeData>& vae,
                                        const std::shared_ptr<LatentData>& lat) {
	i64 T = lat->mt;
	if (T <= 0 || (i64)lat->xm.size() != 64 * T)
		throw MediaError("VaeDecode: the music latent is not [64, T]");
	std::vector<float> over;
	if (const char* path = getenv("PHI_ACE_LATENT_IN"); path && *path) {
		FILE* f = fopen(path, "rb");
		if (!f) throw MediaError(std::string("VaeDecode: cannot open ") + path);
		float buf[4096];
		size_t n = 0;
		while ((n = fread(buf, sizeof(float), 4096, f)) > 0) over.insert(over.end(), buf, buf + n);
		fclose(f);
		if (over.empty() || over.size() % 64 != 0)
			throw MediaError("VaeDecode: the override latent is not a multiple of 64 floats");
		T = (i64)(over.size() / 64);
	}
	const std::vector<float>& z = over.empty() ? lat->xm : over;
	std::vector<float> pcm = vae->ace->decode_pair(z, T);
	auto out = std::make_shared<AudioData>();
	out->sample_rate = vae->ace->sample_rate();
	// This checkpoint's decoder is 2-channel out, so one [64, T] latent is the
	// whole stereo clip: `decode_pair` is the model's own output (a mono decoder
	// would report 1 channel here).
	out->channels = std::max<i64>(1, vae->ace->output_channels());
	out->pcm = std::move(pcm);
	return out;
}

}  // namespace

void register_vae_decode(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaVaeDecode";
	n.category = "vae";
	n.title = "VAE 解码";
	n.title_en = "VAE Decode";
	n.description =
	    "把潜变量解码成图像 / 视频 / 音频。接在 VAE 上的模型决定解码成什么，节点本身只有一个。";
	n.description_en =
	    "Decodes a latent into an image, a video or audio. The VAE on the socket decides which; "
	    "there is one node.";
	n.inputs = {port("vae", SocketType::Vae, false, "the VAE that decodes this latent"),
	            port("latent", SocketType::Latent, false, "the latent to decode")};
	// `Any`: the payload carries the concrete type (IMAGE / VIDEO / AUDIO), which
	// is what the executor puts on the wire.
	n.outputs = {port("out", SocketType::Any, false, "the decoded image / video / audio")};
	n.run = [](GraphContext& ctx, const JsonValue&, const std::vector<Value>& in) {
		auto vae = as_vae(in[0]);
		auto lat = as_latent(in[1]);
		if (!vae) throw MediaError("VaeDecode: the vae input is not a VAE");
		if (!lat) throw MediaError("VaeDecode: missing a latent");
		if (vae->role == "image") {
			if (!vae->image) throw MediaError("VaeDecode: the image VAE is not loaded");
			return std::vector<Value>{Value::of_object(decode_image(ctx, vae, lat))};
		}
		if (vae->role == "video") {
			if (!vae->video) throw MediaError("VaeDecode: the video VAE is not loaded");
			return std::vector<Value>{Value::of_object(decode_video(ctx, vae, lat))};
		}
		if (vae->role == "audio") {
			if (!vae->audio) throw MediaError("VaeDecode: the audio VAE is not loaded");
			return std::vector<Value>{Value::of_object(decode_audio(vae, lat))};
		}
		if (vae->role == "ace_audio") {
			if (!vae->ace) throw MediaError("VaeDecode: the ACE VAE is not loaded");
			return std::vector<Value>{Value::of_object(decode_music(vae, lat))};
		}
		throw MediaError("VaeDecode: unknown VAE role '" + vae->role + "'");
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
