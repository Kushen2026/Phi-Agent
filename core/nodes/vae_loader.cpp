// Node "MediaVaeLoader" — VAE Loader.
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
#include "nodes/service.hpp"
#include "models/image_vae.hpp"
#include "models/video_vae.hpp"
#include "models/audio_vae.hpp"
#include "models/ace_vae.hpp"

namespace phi::media {

void register_vae_loader(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaVaeLoader";
	n.category = "loaders";
	n.title = "VAE 加载器";
	n.title_en = "VAE Loader";
	n.description = "加载图像 VAE、视频 VAE、音频 VAE 或 ACE 音乐 VAE。";
	n.description_en = "Loads an image VAE, video VAE, audio VAE, or the ACE music VAE.";
	n.outputs = {port("VAE", SocketType::Vae, false, "the loaded VAE")};
	{
		JsonValue p = JsonValue::object();
		p["role"] = "image | video | audio | ace_audio";
		p["model"] = "file name (empty = the shipped default)";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>&) {
		const std::string role = param_string(p, "role", "image");
		const char* catalog_role = role == "video"      ? "video_vae"
		                           : role == "audio"    ? "video_avae"
		                           : role == "ace_audio" ? "music_vae"
		                                                 : "image_vae";
		const std::string path =
		    resolve_model(ctx, param_string(p, "model", ""), catalog_role);
		const std::string key = role + "|" + path;
		auto state = graph_state(ctx);
		auto it = state->vaes.find(key);
		if (it != state->vaes.end()) return std::vector<Value>{Value::of_object(it->second)};

		GpuCtx* g = graph_gpu(ctx);
		auto v = std::make_shared<VaeData>();
		v->role = role;
		v->path = path;
		if (role == "image") {
			v->image = std::make_shared<ImageVae>();
			v->image->open(path, g);
		} else if (role == "video") {
			v->video = std::make_shared<VideoVae>();
			v->video->open(path, g);
		} else if (role == "audio") {
			v->audio = std::make_shared<AudioVae>();
			v->audio->open(path, g);
		} else if (role == "ace_audio") {
			v->ace = std::make_shared<AceVae>();
			v->ace->open(path, g);
		} else {
			throw MediaError("unknown VAE role '" + role + "'");
		}
		state->vaes[key] = v;
		return std::vector<Value>{Value::of_object(v)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
