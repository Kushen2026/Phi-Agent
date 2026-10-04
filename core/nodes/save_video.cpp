// Node "MediaSaveVideo" — Save Video (mux mp4).
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
#include "nodes/service.hpp"

namespace phi::media {

void register_save_video(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaSaveVideo";
	n.category = "video";
	n.title = "保存视频（封装 mp4）";
	n.title_en = "Save Video (mux mp4)";
	n.description = "把视频帧（+ 可选音轨）封装成 H.264 + AAC 的 mp4。";
	n.description_en = "Muxes video frames (+ an optional audio track) into an H.264 + AAC mp4.";
	n.inputs = {port("video", SocketType::Video, false, "the frames"),
	            port("audio", SocketType::Audio, true, "the soundtrack")};
	n.outputs = {port("path", SocketType::String, false, "the written path"),
	             port("encoders", SocketType::String, true, "the MFTs the muxer chose")};
	{
		JsonValue p = JsonValue::object();
		p["path"] = "output mp4 path";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>& in) {
		auto v = as_video(in[0]);
		if (!v) throw MediaError("SaveVideo: missing the video input");
		auto a = as_audio(in[1]);
		const std::string path = resolve_user_path(ctx, param_string(p, "path", "out.mp4"));
		const size_t slash = path.find_last_of("/\\");
		if (slash != std::string::npos && slash > 0) mkdirs(path.substr(0, slash));
		const i64 fps = v->fps > 0 ? v->fps : 24;
		const i64 sr = a ? a->sample_rate : 32000;
		Mp4Muxer mux;
		mux.open(path, v->out_w, v->out_h, fps, a ? sr : 0, a ? a->channels : 0);
		std::vector<unsigned char> full, rgb;
		std::vector<float> fimg;
		i64 audio_off = 0;
		const i64 atotal = a ? (i64)(a->pcm.size() / (size_t)std::max<i64>(a->channels, 1)) : 0;
		for (i64 f = 0; f < v->frames; f++) {
			const double seconds = (double)f / (double)fps;
			if (a) {
				const i64 want =
				    std::min<i64>(atotal, (i64)std::llround(seconds * (double)sr));
				if (want > audio_off) {
					mux.add_audio(a->pcm.data() + (size_t)audio_off * a->channels,
					              (want - audio_off) * a->channels);
					audio_off = want;
				}
			}
			video_frame_to_hwc(v->planes.data(), 3, (int)v->frames, (int)f, (int)v->h,
			                   (int)v->w, fimg);
			float01_to_srgb8(fimg.data(), (int)v->w, (int)v->h, 3, full);
			rgb.clear();
			crop_centre(full.data(), (int)v->w, (int)v->h, (int)v->out_w, (int)v->out_h, rgb);
			mux.add_video(rgb.data(), v->out_w, v->out_h, seconds);
		}
		if (a && audio_off < atotal)
			mux.add_audio(a->pcm.data() + (size_t)audio_off * a->channels,
			              (atotal - audio_off) * a->channels);
		mux.close();
		std::string enc = mux.video_encoder();
		if (!mux.audio_encoder().empty()) enc += (enc.empty() ? "" : " + ") + mux.audio_encoder();
		return std::vector<Value>{Value::of_string(path), Value::of_string(enc)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
