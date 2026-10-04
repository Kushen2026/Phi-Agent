// Node "MediaEmptyLatent" — one initial-noise latent node for every chain.
//
// ComfyUI's `EmptySD3LatentImage` / `EmptyMiniMaxH3LatentAV` /
// `EmptyAceStep1.5LatentAudio` differ only in the geometry formula, and this
// engine carried one node per formula (`MediaEmptyLatentImage`,
// `MediaEmptyLatentVideo`, `MediaAceLatent`). Unlike a VAE or a sampler a latent
// node has no model socket to infer the chain from, so the chain is an explicit
// `arch` parameter - and everything else (the 16-aligned canvas, the 17k+5 frame
// snap, the 25 Hz music grid, the seed rule) is the one place each formula lives.
//
//   arch "image":  [64, H/16, W/16]   flux latent, 16-aligned canvas + crop geometry
//   arch "video":  [24, vt, lh, lw] + [32, 2, at]  the H3 joint A/V pair
//   arch "music":  [64, round(seconds * 25)]       ACE-Step 1.5 (48 kHz / hop 1920)
//
// One node, one file. Registered here; the shared helpers live in service.hpp,
// the catalogue entry point in registry.cpp.
#include "nodes/service.hpp"

namespace phi::media {

namespace {

std::shared_ptr<LatentData> latent_image(const JsonValue& p, const std::vector<Value>& in) {
	const i64 w = in[1].valid() ? in[1].as_int(1024) : param_int(p, "width", 1024);
	const i64 h = in[2].valid() ? in[2].as_int(1024) : param_int(p, "height", 1024);
	const u64 seed = latent_seed(p, in[0]);
	auto l = std::make_shared<LatentData>();
	l->arch = "image";
	l->seed = seed;
	l->req_w = w;
	l->req_h = h;
	l->canvas_w = ceil_div(w, 16) * 16;
	l->canvas_h = ceil_div(h, 16) * 16;
	l->h = l->canvas_h / 16;
	l->w = l->canvas_w / 16;
	l->c = 64;   // Qwen-Image in_channels
	l->x.assign((size_t)(l->c * l->h * l->w), 0.0f);
	fill_normal32(l->x.data(), l->x.size(), seed);
	return l;
}

std::shared_ptr<LatentData> latent_video(const JsonValue& p, const std::vector<Value>& in) {
	const i64 frames = in[1].valid() ? in[1].as_int(121) : param_int(p, "frames", 121);
	const i64 w = in[2].valid() ? in[2].as_int(960) : param_int(p, "width", 960);
	const i64 h = in[3].valid() ? in[3].as_int(540) : param_int(p, "height", 540);
	const i64 fps = param_int(p, "fps", 24);
	const u64 seed = latent_seed(p, in[0]);
	auto l = std::make_shared<LatentData>();
	l->arch = "video";
	l->seed = seed;
	l->plan = plan_video(w, h, frames, fps);
	l->vt = l->plan.video_t;
	l->lat_h = l->plan.latent_h;
	l->lat_w = l->plan.latent_w;
	l->at = l->plan.audio_t;
	l->c = 24;
	const size_t n_v = (size_t)(24 * l->vt * l->lat_h * l->lat_w);
	const size_t n_a = (size_t)(32 * 2 * l->at);
	l->xv.assign(n_v, 0.0f);
	l->xa.assign(n_a, 0.0f);
	fill_normal64(l->xv.data(), n_v, seed);
	fill_normal64(l->xa.data(), n_a, seed ^ 0x9E3779B97F4A7C15ull);
	return l;
}

std::shared_ptr<LatentData> latent_music(const JsonValue& p, const std::vector<Value>& in) {
	const double seconds = param_number(p, "seconds", 60.0);
	const i64 T = std::max<i64>(1, (i64)std::llround(seconds * 25.0));
	const u64 seed = latent_seed(p, in[0]);
	auto l = std::make_shared<LatentData>();
	l->arch = "music";
	l->seed = seed;
	l->mt = T;
	l->seconds = (i64)std::llround(seconds);
	l->xm.assign((size_t)64 * (size_t)T, 0.0f);
	fill_normal64(l->xm.data(), l->xm.size(), seed);
	return l;
}

}  // namespace

void register_latent(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaEmptyLatent";
	n.category = "latent";
	n.title = "空潜变量";
	n.title_en = "Empty Latent";
	n.description =
	    "按 arch 生成图像 / 视频 / 音乐链的初始噪声潜变量。图像按 width/height（16 对齐，含裁剪几何），"
	    "视频按 frames（吸附到 17k+5）与 width/height，音乐按 seconds（48kHz/hop1920 → 25 帧每秒）。";
	n.description_en =
	    "Creates the initial-noise latent for the image / video / music chain, chosen by `arch`. "
	    "Image: width/height (16-aligned, with the crop geometry). Video: frames (snapped to "
	    "17k+5) and width/height. Music: seconds (48 kHz / hop 1920 = 25 frames per second).";
	n.inputs = {port("seed", SocketType::Int, true, "override the seed"),
	            port("frames", SocketType::Int, true, "override the frame count (video)"),
	            port("width", SocketType::Int, true, "override the width"),
	            port("height", SocketType::Int, true, "override the height")};
	n.outputs = {port("LATENT", SocketType::Latent, false, "the latent")};
	{
		JsonValue p = JsonValue::object();
		p["arch"] = "image | video | music";
		p["width"] = "canvas width (image, video)";
		p["height"] = "canvas height (image, video)";
		p["frames"] = "requested frames, snapped to 17k+5 (video)";
		p["fps"] = "frames per second (24)";
		p["seconds"] = "duration in seconds (music)";
		p["seed"] = "the noise seed, used as given (leave the field out to draw one)";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>& in) {
		const std::string arch = param_string(p, "arch", "image");
		(void)ctx;
		if (arch == "image") return std::vector<Value>{Value::of_object(latent_image(p, in))};
		if (arch == "video") return std::vector<Value>{Value::of_object(latent_video(p, in))};
		if (arch == "music") return std::vector<Value>{Value::of_object(latent_music(p, in))};
		throw MediaError("EmptyLatent: unknown arch '" + arch + "' (image | video | music)");
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
