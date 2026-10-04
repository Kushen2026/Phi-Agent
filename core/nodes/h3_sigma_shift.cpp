// Node "MediaH3SigmaShift" — MiniMax H3 sigma shift (ModelSamplingAV).
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
//
// ComfyUI's `MiniMaxH3SigmaShift` (displayed as ModelSamplingMiniMaxH3): it
// patches a MODEL with the two flow shifts the H3 chain runs on.
//
//   * `shift_video` is the shift of the *sampler's* sigma schedule
//     (ModelSamplingAV, i.e. the beta grid's `time_snr_shift` factor);
//   * both values travel to the DiT in `transformer_options`, where the forward
//     pass inverts the video schedule onto the shared base grid and derives the
//     audio schedule from it (`sigma_a = time_shift_sigma(sigma_v, shift_v,
//     shift_a)`, `audio_scale = shift_v / shift_a`).
//
// The released workflow uses 12 / 3, which is what the loaders default to. A
// distilled / few-step LoRA (the turbo checkpoints) is trained at a *different*
// shift, and a chain that cannot say so is stuck with the base one - which is why
// this is a node and not a constant.
//
// The node returns a *copy* of the model handle: the engine's loaders cache the
// loaded weights per (arch, path, loras) in the run's state bag, and a patch that
// edited the cached handle would leak into every other chain sharing it. The copy
// shares the loaded DiT (and its LoRA chain) and only the two floats differ, so
// patching is free.
#include "nodes/service.hpp"

namespace phi::media {

void register_h3_sigma_shift(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaH3SigmaShift";
	n.category = "sampling";
	n.title = "模型采样（MiniMax H3 位移）";
	n.title_en = "ModelSamplingMiniMaxH3";
	n.description =
	    "设置 H3 的视频/音频 flow shift：视频 shift 决定采样 sigma 网格，两者都交给 DiT "
	    "（它把视频网格反变换回共享基准网格，再推导音频网格）。蒸馏/少步 LoRA 常用不同于 12/3 的值。";
	n.description_en =
	    "Sets the H3 video/audio flow shifts: the video shift drives the sampler's sigma grid and "
	    "both are handed to the DiT (which inverts the video schedule onto the shared base grid "
	    "and derives the audio one from it). A distilled / few-step LoRA usually wants different "
	    "values than the released 12 / 3.";
	n.inputs = {port("model", SocketType::Model, false, "the H3 backbone")};
	n.outputs = {port("MODEL", SocketType::Model, false, "the same backbone, shifts patched")};
	{
		JsonValue p = JsonValue::object();
		p["shift_video"] = "video flow shift (12.0 released)";
		p["shift_audio"] = "audio flow shift (3.0 released)";
		n.params = p;
	}
	n.run = [](GraphContext&, const JsonValue& p, const std::vector<Value>& in) {
		auto m = as_model(in[0]);
		if (!m) throw MediaError("H3SigmaShift: model input is not a MODEL");
		if (m->arch != "minimax_h3")
			throw MediaError("H3SigmaShift: the model is not MiniMax H3 (arch '" + m->arch + "')");
		auto out = std::make_shared<ModelData>(*m);   // shares the loaded DiT
		out->h3_shift_video = (float)param_number(p, "shift_video", (double)m->h3_shift_video);
		out->h3_shift_audio = (float)param_number(p, "shift_audio", (double)m->h3_shift_audio);
		if (!(out->h3_shift_video > 0.0f) || !(out->h3_shift_audio > 0.0f))
			throw MediaError("H3SigmaShift: shifts must be positive");
		return std::vector<Value>{Value::of_object(out)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
