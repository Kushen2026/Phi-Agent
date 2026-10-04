#include "nodes/service.hpp"

#include "models/ace_models.hpp"   // AceDiT::release_resident (see release_media_cache)
#include "models/audio_vae.hpp"
#include "models/av_dit.hpp"
#include "models/image_dit.hpp"
#include "models/image_vae.hpp"
#include "models/video_vae.hpp"

namespace phi::media {

// ── payload accessors ──────────────────────────────────────────────────────

std::shared_ptr<ModelData> as_model(const Value& v) {
	return v.type == SocketType::Model ? std::dynamic_pointer_cast<ModelData>(v.obj) : nullptr;
}
std::shared_ptr<ClipData> as_clip(const Value& v) {
	return v.type == SocketType::Clip ? std::dynamic_pointer_cast<ClipData>(v.obj) : nullptr;
}
std::shared_ptr<VaeData> as_vae(const Value& v) {
	return v.type == SocketType::Vae ? std::dynamic_pointer_cast<VaeData>(v.obj) : nullptr;
}
std::shared_ptr<LoraData> as_lora(const Value& v) {
	return v.type == SocketType::Lora ? std::dynamic_pointer_cast<LoraData>(v.obj) : nullptr;
}
std::shared_ptr<ConditioningData> as_conditioning(const Value& v) {
	return v.type == SocketType::Conditioning ? std::dynamic_pointer_cast<ConditioningData>(v.obj)
	                                          : nullptr;
}
std::shared_ptr<LatentData> as_latent(const Value& v) {
	return v.type == SocketType::Latent ? std::dynamic_pointer_cast<LatentData>(v.obj) : nullptr;
}
std::shared_ptr<SigmasData> as_sigmas(const Value& v) {
	return v.type == SocketType::Sigmas ? std::dynamic_pointer_cast<SigmasData>(v.obj) : nullptr;
}
std::shared_ptr<ImageData> as_image(const Value& v) {
	return v.type == SocketType::Image ? std::dynamic_pointer_cast<ImageData>(v.obj) : nullptr;
}
std::shared_ptr<AudioData> as_audio(const Value& v) {
	return v.type == SocketType::Audio ? std::dynamic_pointer_cast<AudioData>(v.obj) : nullptr;
}
std::shared_ptr<VideoData> as_video(const Value& v) {
	return v.type == SocketType::Video ? std::dynamic_pointer_cast<VideoData>(v.obj) : nullptr;
}

std::shared_ptr<GraphModelCache> graph_state(GraphContext& ctx) {
	if (!ctx.user) ctx.user = std::make_shared<GraphModelCache>();
	return std::static_pointer_cast<GraphModelCache>(ctx.user);
}

void release_media_cache(GraphModelCache& cache) {
	for (auto& kv : cache.vaes) {
		VaeData* v = kv.second.get();
		if (!v) continue;
		if (v->image) v->image->release_gpu_memory();
		if (v->video) v->video->release_gpu_memory();
		// the audio VAE (and its vocoder) is host fp32 code - nothing in VRAM
	}
	cache.aux.clear();
	for (auto& kv : cache.models) {
		ModelData* m = kv.second.get();
		if (!m) continue;
		if (m->image) m->image->release_resident();
		if (m->h3) m->h3->release_weights();
		// The ACE DiT's weight window is handed back by its sampler at the end of
		// every run, so this is the *second* chance - the one that matters after a
		// cancelled or failed run, where the sampler's own release never ran. It is
		// the same phase boundary the other two chains get here.
		if (m->ace) m->ace->release_resident();
	}
}

GpuCtx* graph_gpu(GraphContext& ctx) {
	if (!ctx.gpu) throw MediaError("node graph: no GPU context (the media engine is not open)");
	return static_cast<GpuCtx*>(ctx.gpu);
}

// ── small helpers ──────────────────────────────────────────────────────────


// The phase boundary the two pipelines drew by hand (`release_phase_arenas`):
// hand the shared streaming/activation arenas back to the device so the next
// phase's residency plan reads the card this process really has. It is called at
// the end of each node that streamed through them (text encode, a sampler, a VAE
// decode) - never inside a sampling loop, where the chunks are deliberately kept
// so the loop's accounting stabilises.
void release_shared_arenas(GpuCtx* g) {
	if (!g) return;
	if (g->wa) g->wa->release_chunks();
	if (g->aa) g->aa->release_chunks();
}

bool looks_absolute(const std::string& p) {
	if (p.empty()) return false;
	if (p[0] == '/' || p[0] == '\\') return true;
	return p.size() >= 2 && p[1] == ':';
}

// A caller-supplied path: absolute is taken as-is, relative is joined with the
// session cwd. Used for reference images/audio/video, which are files the user
// points at.
std::string resolve_user_path(GraphContext& ctx, const std::string& p) {
	if (looks_absolute(p)) return path_normalize(p);
	return path_normalize(path_join(ctx.cwd, p));
}

// A model file: an absolute path, a path relative to the models directory, or a
// bare filename matched anywhere under it. Empty => the shipped default for the
// role.
std::string resolve_model(GraphContext& ctx, const std::string& name, const char* role) {
	if (!name.empty()) {
		if (looks_absolute(name) && path_exists(name)) return path_normalize(name);
		std::string r = resolve_media_file(ctx.models_dir, name);
		if (!r.empty()) return r;
		throw MediaError(std::string("model file not found under the models directory: ") + name);
	}
	const std::string id = default_media_model_id(role);
	if (const MediaModelEntry* e = media_model_entry(id)) {
		std::string r = resolve_media_file(ctx.models_dir, e->file);
		if (!r.empty()) return r;
	}
	throw MediaError(std::string("no model selected for role '") + role + "'");
}

TokenizerFiles tokenizer_files_of(GraphContext& ctx, const std::string& arch,
                                  const std::string& override_vocab,
                                  const std::string& override_merges,
                                  const std::string& override_config) {
	const char* pv = arch == "qwen3vl_32b" ? "video_tokenizer_vocab"
	                 : arch == "ace15"      ? "music_tokenizer_vocab"
	                                        : "image_tokenizer_vocab";
	const char* pm = arch == "qwen3vl_32b" ? "video_tokenizer_merges"
	                 : arch == "ace15"      ? "music_tokenizer_merges"
	                                        : "image_tokenizer_merges";
	const char* pc = arch == "qwen3vl_32b" ? "video_tokenizer_config"
	                 : arch == "ace15"      ? "music_tokenizer_config"
	                                        : "image_tokenizer_config";
	// A node parameter wins; otherwise the chain's own role (resolved through the
	// same default table the settings panel uses). Absolute stays absolute, a bare
	// name is looked up under the models directory.
	auto pick = [&](const char* role, const std::string& given) -> std::string {
		if (!given.empty()) return path_normalize(resolve_user_path(ctx, given));
		const std::string v = resolve_media_role(JsonValue::object(), role, ctx.models_dir);
		return v.empty() ? std::string() : path_normalize(v);
	};
	TokenizerFiles f;
	f.vocab = pick(pv, override_vocab);
	f.merges = pick(pm, override_merges);
	f.config = pick(pc, override_config);
	if (!f.complete())
		throw MediaError(std::string("no tokenizer configured for '") + arch +
		                 "': select vocab.json / merges.txt in 工具设置 → 媒体模型 (roles " + pv +
		                 ", " + pm + ")");
	return f;
}

std::string resolve_config_file(GraphContext& ctx, const std::string& name) {
	if (name.empty()) return std::string();
	if (looks_absolute(name) && path_exists(name)) return path_normalize(name);
	std::string r = resolve_media_file(ctx.models_dir, name);
	if (!r.empty()) return r;
	return resolve_user_path(ctx, name);
}

std::shared_ptr<Qwen2Tokenizer> load_tokenizer(GraphContext& ctx, const TokenizerFiles& f) {
	// Keyed on the three paths: two chains with different vocabularies must not
	// share an entry, and a chain whose selection changed must reload.
	const std::string key = f.vocab + "|" + f.merges + "|" + f.config;
	auto cache = graph_state(ctx);
	auto it = cache->tokenizers.find(key);
	if (it != cache->tokenizers.end()) return it->second;
	auto tok = std::make_shared<Qwen2Tokenizer>();
	tok->load(f.vocab, f.merges, f.config);
	cache->tokenizers[key] = tok;
	return tok;
}

// Box-Muller over a mixed seed. The image chain seeds a 32-bit mt19937; the
// video chain a 64-bit one. Both consume the pair one spare at a time so an odd
// latent size does not shift the field.
void fill_normal32(float* dst, size_t n, u64 seed) {
	std::mt19937 rng((u32)(seed ^ (seed >> 32)));
	std::uniform_real_distribution<double> uni(0.0, 1.0);
	bool have_spare = false;
	double spare = 0.0;
	for (size_t i = 0; i < n; i++) {
		if (have_spare) {
			dst[i] = (float)spare;
			have_spare = false;
			continue;
		}
		double u1 = uni(rng), u2 = uni(rng);
		if (u1 <= 1e-12) u1 = 1e-12;
		const double rr = std::sqrt(-2.0 * std::log(u1));
		const double th = 6.283185307179586 * u2;
		dst[i] = (float)(rr * std::cos(th));
		spare = rr * std::sin(th);
		have_spare = true;
	}
}

void fill_normal64(float* dst, size_t n, u64 seed) {
	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<double> uni(0.0, 1.0);
	bool have_spare = false;
	double spare = 0.0;
	for (size_t i = 0; i < n; i++) {
		if (have_spare) {
			dst[i] = (float)spare;
			have_spare = false;
			continue;
		}
		double u1 = uni(rng), u2 = uni(rng);
		if (u1 <= 1e-12) u1 = 1e-12;
		const double rr = std::sqrt(-2.0 * std::log(u1));
		const double th = 6.283185307179586 * u2;
		dst[i] = (float)(rr * std::cos(th));
		spare = rr * std::sin(th);
		have_spare = true;
	}
}

// See the header for the rule. Presence, not the value, is what separates a seed
// from "none", so the parameter is read as the 64 bits it carries rather than as
// a signed number: a chain writes the seed with the whole u64 range in it, and
// int_value() hands that id back exactly (as_int() would refuse the half of the
// range above INT64_MAX, answering 0 - which used to be read as "draw one" and
// silently replaced the id the caller asked for). 0 comes back as 0.
u64 latent_seed(const JsonValue& p, const Value& wired) {
	if (wired.valid()) return (u64)wired.as_int(0);
	const JsonValue* v = p.find("seed");
	if (v && v->int_repr()) return v->int_value();
	if (v && v->is_number()) return v->as_u64(0);
	return make_media_seed();
}

// ── the ref2va reference blocks (shared with the video chain) ──────────────
//
// The rule lives here, next to the node that needs it, because that is what lets
// TextEncode be a node: it needs the reference latents, which are a VAE encode,
// and the VAE is a wired input rather than an owned member.

void ref_image_size(i64 w, i64 h, i64 canvas_w, i64 canvas_h, i64* tw, i64* th) {
	// ref_image_size = "match": the reference's own rule lives in vision_prep
	// (shared with the tokenizer side, which has to resize the same pixels for the
	// text tower), so this is a forward rather than a second copy of the maths.
	ref2va_image_size(w, h, canvas_w, canvas_h, false, tw, th);
}

void hwc_bytes_to_planar(const std::vector<unsigned char>& rgb, i64 w, i64 h, i64 t, i64 f,
                         std::vector<float>& out) {
	const i64 hw = w * h;
	out.assign((size_t)(3 * t * hw), 0.0f);
	for (i64 y = 0; y < h; y++)
		for (i64 x = 0; x < w; x++)
			for (i64 c = 0; c < 3; c++)
				out[(size_t)((c * t + f) * hw + y * w + x)] =
				    2.0f * ((float)rgb[(size_t)((y * w + x) * 3 + c)] / 255.0f) - 1.0f;
}

RefImagePixels load_ref_image(const std::string& path, i64 canvas_w, i64 canvas_h, bool use_max) {
	int w = 0, h = 0;
	std::vector<unsigned char> rgb = read_image(path, &w, &h);
	if (w <= 0 || h <= 0) throw MediaError("reference image has no pixels: " + path);
	RefImagePixels out;
	ref2va_image_size(w, h, canvas_w, canvas_h, use_max, &out.w, &out.h);
	if (out.w == w && out.h == h) {
		out.rgb = std::move(rgb);
	} else {
		resize_lanczos3(rgb.data(), w, h, (int)out.w, (int)out.h, out.rgb);
	}
	return out;
}

void build_image_ref(ConditioningData& c, const RefImagePixels& px, VideoVae& vvae) {
	if (px.w <= 0 || px.h <= 0 || px.rgb.empty())
		throw MediaError("reference image did not resize to a usable size");
	const i64 tw = px.w, th = px.h;
	std::vector<float> planar;
	hwc_bytes_to_planar(px.rgb, tw, th, 1, 0, planar);
	H3Ref r;
	r.kind = H3Ref::Image;
	r.latent_t = 1;
	r.latent_h = th / 16;
	r.latent_w = tw / 16;
	if (r.latent_h <= 0 || r.latent_w <= 0 || (r.latent_h % 2) || (r.latent_w % 2))
		throw MediaError("reference image did not resize to a multiple of 32");
	c.ref_vlat.push_back(vvae.encode(planar, 1, th, tw));
	c.ref_alat.emplace_back();
	c.refs.push_back(r);
}

void adapted_ref_video_canvas(i64 vw, i64 vh, i64* cw, i64* ch) {
	const double short_edge = 768.0, max_pixels = 768.0 * 1344.0;
	const double ratio = (double)vw / (double)vh;
	double nw = ratio >= 1.0 ? short_edge * ratio : short_edge;
	double nh = ratio >= 1.0 ? short_edge : short_edge / ratio;
	if (nw * nh > max_pixels) {
		const double s = std::sqrt(max_pixels / (nw * nh));
		nw *= s;
		nh *= s;
	}
	*cw = std::max<i64>(32, (i64)std::llround(nw / 32.0) * 32);
	*ch = std::max<i64>(32, (i64)std::llround(nh / 32.0) * 32);
	if ((double)vw * (double)vh < (double)*cw * (double)*ch) {
		*cw = std::max<i64>(32, (i64)std::llround((double)vw / 32.0) * 32);
		*ch = std::max<i64>(32, (i64)std::llround((double)vh / 32.0) * 32);
	}
}

RefVideoFrames load_ref_clip(const std::string& path, i64 target_frames) {
	VideoReader rd;
	rd.open(path);
	const i64 vw = rd.width(), vh = rd.height();
	if (vw <= 0 || vh <= 0) throw MediaError("reference video has no usable size: " + path);
	RefVideoFrames out;
	adapted_ref_video_canvas(vw, vh, &out.canvas_w, &out.canvas_h);
	const i64 cw = out.canvas_w, ch = out.canvas_h;
	// The reference's own duration, cut down to the generation's length and then
	// snapped *down* to the temporal VAE's legal 17k+5 grid (a longer reference
	// than the target cannot be anchored to it).
	i64 want = (i64)std::llround(rd.duration_seconds() * 24.0);
	if (want <= 0) want = target_frames;
	want = std::min(want, target_frames);
	if (want < 5) want = 5;
	while (want % 17 != 5) want--;
	std::vector<VideoFrame> frames = rd.sample_frames(want);
	if ((i64)frames.size() < 5) throw MediaError("reference video has too few frames: " + path);
	i64 n = (i64)frames.size();
	if (n > target_frames) n = target_frames;
	while (n % 17 != 5) n--;
	if (n < 5) n = 5;
	n = std::min<i64>(n, (i64)frames.size());
	out.frames = n;
	out.rgb.assign((size_t)(n * cw * ch * 3), 0);
	std::vector<unsigned char> res;
	for (i64 f = 0; f < n; f++) {
		const VideoFrame& fr = frames[(size_t)f];
		resize_lanczos3(fr.rgb.data(), (int)fr.w, (int)fr.h, (int)cw, (int)ch, res);
		memcpy(&out.rgb[(size_t)f * (size_t)cw * (size_t)ch * 3], res.data(),
		       (size_t)cw * (size_t)ch * 3);
	}
	return out;
}

void build_video_ref(ConditioningData& c, const RefVideoFrames& fr,
                     const std::string& audio_path, AudioVae* avae, VideoVae& vvae) {
	const i64 cw = fr.canvas_w, ch = fr.canvas_h, n = fr.frames;
	if (cw <= 0 || ch <= 0 || n <= 0) throw MediaError("reference video canvas is invalid");
	// The video VAE wants planar [-1, 1]; the text tower samples the *same* bytes
	// at 2 fps (`append_ref_video_blocks`), so both come from one decode+resize.
	std::vector<float> clip((size_t)(3 * n * cw * ch), 0.0f);
	for (i64 f = 0; f < n; f++) {
		const unsigned char* src = &fr.rgb[(size_t)f * (size_t)cw * (size_t)ch * 3];
		for (i64 y = 0; y < ch; y++)
			for (i64 x = 0; x < cw; x++)
				for (i64 c3 = 0; c3 < 3; c3++)
					clip[(size_t)((c3 * n + f) * cw * ch + y * cw + x)] =
					    2.0f * ((float)src[(size_t)((y * cw + x) * 3 + c3)] / 255.0f) - 1.0f;
	}
	H3Ref r;
	r.kind = H3Ref::Video;
	r.latent_h = ch / 16;
	r.latent_w = cw / 16;
	if (r.latent_h <= 0 || r.latent_w <= 0) throw MediaError("reference video canvas is invalid");
	std::vector<float> z = vvae.encode(clip, n, ch, cw);
	r.latent_t = (i64)z.size() / (24 * r.latent_h * r.latent_w);
	c.ref_vlat.push_back(std::move(z));
	c.ref_alat.emplace_back();
	// The soundtrack rides the same block (the DiT's `video_audio` packing: its
	// audio rows are injected *before* its video rows, which is why the reference
	// emits the `<Audio j>` label ahead of the `<Video k>` one).
	if (!audio_path.empty() && avae) {
		AudioClip clip_a = read_audio(audio_path);
		if (clip_a.sample_rate <= 0 || clip_a.samples.empty())
			throw MediaError("reference video soundtrack is empty: " + audio_path);
		const i64 sr = avae->config().sample_rate > 0 ? avae->config().sample_rate : 32000;
		std::vector<float> st =
		    clip_a.sample_rate == sr
		        ? clip_a.samples
		        : resample(clip_a.samples, clip_a.channels, clip_a.sample_rate, sr);
		st = to_channels(st, clip_a.channels, 2);
		const i64 ns = (i64)st.size() / 2;
		std::vector<float> wl((size_t)ns), wr((size_t)ns);
		for (i64 i = 0; i < ns; i++) {
			wl[(size_t)i] = st[(size_t)(i * 2 + 0)];
			wr[(size_t)i] = st[(size_t)(i * 2 + 1)];
		}
		std::vector<float> zl = avae->encode(wl, ns);
		std::vector<float> zr = avae->encode(wr, ns);
		const i64 t40 = (i64)zl.size() / 32;
		if (t40 <= 0) throw MediaError("reference video soundtrack is too short: " + audio_path);
		std::vector<float> za((size_t)(32 * 2 * t40), 0.0f);
		for (i64 ach = 0; ach < 32; ach++)
			for (i64 t = 0; t < t40; t++) {
				za[(size_t)((ach * 2 + 0) * t40 + t)] = zl[(size_t)(ach * t40 + t)];
				za[(size_t)((ach * 2 + 1) * t40 + t)] = zr[(size_t)(ach * t40 + t)];
			}
		r.ref_audio_t = t40;
		c.ref_alat.back() = std::move(za);
	}
	c.refs.push_back(r);
}

void build_audio_ref(ConditioningData& c, const std::string& path, AudioVae& avae) {
	AudioClip clip = read_audio(path);
	if (clip.sample_rate <= 0 || clip.samples.empty())
		throw MediaError("reference audio is empty: " + path);
	const i64 sr = avae.config().sample_rate > 0 ? avae.config().sample_rate : 32000;
	std::vector<float> st =
	    clip.sample_rate == sr ? clip.samples
	                           : resample(clip.samples, clip.channels, clip.sample_rate, sr);
	st = to_channels(st, clip.channels, 2);
	const i64 n = (i64)st.size() / 2;
	std::vector<float> wl((size_t)n), wr((size_t)n);
	for (i64 i = 0; i < n; i++) {
		wl[(size_t)i] = st[(size_t)(i * 2 + 0)];
		wr[(size_t)i] = st[(size_t)(i * 2 + 1)];
	}
	std::vector<float> zl = avae.encode(wl, n);
	std::vector<float> zr = avae.encode(wr, n);
	const i64 t40 = (i64)zl.size() / 32;
	if (t40 <= 0) throw MediaError("reference audio is too short");
	c.ref_audio_seconds.push_back((double)n / (double)sr);
	std::vector<float> z((size_t)(32 * 2 * t40), 0.0f);
	for (i64 ch = 0; ch < 32; ch++)
		for (i64 t = 0; t < t40; t++) {
			z[(size_t)((ch * 2 + 0) * t40 + t)] = zl[(size_t)(ch * t40 + t)];
			z[(size_t)((ch * 2 + 1) * t40 + t)] = zr[(size_t)(ch * t40 + t)];
		}
	H3Ref r;
	r.kind = H3Ref::Audio;
	r.ref_audio_t = t40;
	c.ref_alat.push_back(std::move(z));
	c.ref_vlat.emplace_back();
	c.refs.push_back(r);
}

Port port(const char* name, SocketType t, bool optional, const char* doc) {
	return Port{name, t, optional, doc};
}

}  // namespace phi::media
