// Node "MediaSaveAudio" — Save Audio.
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
#include "nodes/service.hpp"

namespace phi::media {

void register_save_audio(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaSaveAudio";
	n.category = "audio";
	n.title = "保存音频";
	n.title_en = "Save Audio";
	n.description = "把 PCM 写成 16 位 WAV。";
	n.description_en = "Writes PCM as a 16-bit WAV.";
	n.inputs = {port("audio", SocketType::Audio, false, "the PCM")};
	n.outputs = {port("path", SocketType::String, false, "the written path")};
	{
		JsonValue p = JsonValue::object();
		p["path"] = "output WAV path";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>& in) {
		auto a = as_audio(in[0]);
		if (!a || a->pcm.empty()) throw MediaError("SaveAudio: nothing to save");
		const std::string path = resolve_user_path(ctx, param_string(p, "path", "out.wav"));
		const size_t slash = path.find_last_of("/\\");
		if (slash != std::string::npos && slash > 0) mkdirs(path.substr(0, slash));
		AudioClip clip;
		clip.samples = a->pcm;
		clip.sample_rate = a->sample_rate;
		clip.channels = a->channels;
		write_wav(path, clip);
		return std::vector<Value>{Value::of_string(path)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
