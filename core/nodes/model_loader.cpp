// Node "MediaModelLoader" — Model Loader (DiT).
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
#include "nodes/service.hpp"
#include "models/image_dit.hpp"
#include "models/av_dit.hpp"
#include "models/ace_models.hpp"
#include "models/breeze_tts.hpp"

namespace phi::media {

void register_model_loader(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaModelLoader";
	n.category = "loaders";
	n.title = "模型加载器";
	n.title_en = "Model Loader (DiT)";
	n.description = "加载扩散主干（Qwen-Image 2.1 或 MiniMax H3）。可接 LoRA 链。";
	n.description_en =
	    "Loads a diffusion backbone (Qwen-Image 2.1 or MiniMax H3). Accepts a LoRA chain.";
	n.inputs = {port("lora", SocketType::Lora, true, "LoRA chain applied at load")};
	n.outputs = {port("MODEL", SocketType::Model, false, "the loaded backbone")};
	{
		JsonValue p = JsonValue::object();
		p["arch"] = "qwen_image | minimax_h3 | ace_step15 | breeze_tts";
		p["model"] = "file name (empty = the shipped default for the arch)";
		p["codec"] = "breeze_tts: the codec checkpoint (empty = the role default)";
		p["tokenizer"] = "breeze_tts: tokenizer.json (empty = the role default)";
		p["config"] = "breeze_tts: config.json (empty = the sibling config.json)";
		p["codec_config"] = "breeze_tts: codec config.json (empty = the sibling config.json)";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>& in) {
		const std::string arch = param_string(p, "arch", "qwen_image");
		const char* role = arch == "minimax_h3" ? "video_dit"
		                   : arch == "ace_step15" ? "music_dit"
		                   : arch == "breeze_tts" ? "tts_model"
		                                          : "image_dit";
		const std::string path = resolve_model(ctx, param_string(p, "model", ""), role);
		auto lora = as_lora(in[0]);
		const std::string lora_key = lora ? std::to_string((uintptr_t)lora->set.get()) : "-";
		const std::string key = arch + "|" + path + "|" + lora_key;
		auto state = graph_state(ctx);
		auto it = state->models.find(key);
		if (it != state->models.end()) return std::vector<Value>{Value::of_object(it->second)};

		GpuCtx* g = graph_gpu(ctx);
		auto m = std::make_shared<ModelData>();
		m->arch = arch;
		m->path = path;
		if (lora) {
			m->loras = lora->set;
			m->lora_paths = lora->paths;
		}
		if (arch == "qwen_image") {
			m->image = std::make_shared<ImageDiT>();
			m->image->set_loras(m->loras.get());
			m->image->open(path, g);
			m->image_flow.kind = SamplingFlow::Kind::Flux;
			m->image_flow.shift = 0.69f;
			m->image_flow.multiplier = 1.0f;
			if (const char* e = getenv("PHI_IMG_SHIFT")) m->image_flow.shift = (float)atof(e);
		} else if (arch == "minimax_h3") {
			m->h3 = std::make_shared<AvDiT>();
			m->h3->set_loras(m->loras.get());
			m->h3->open(path, g);
		} else if (arch == "ace_step15") {
			// The DiT bundle carries the decoder, the condition encoder and the
			// audio-code tokenizer/detokenizer; the embedder and the LM are the
			// CLIP socket (MediaClipLoader arch "ace15"), exactly as ComfyUI
			// wires DualCLIPLoader -> TextEncodeAceStepAudio1.5.
			m->ace = std::make_shared<AceDiT>();
			m->ace->open(path, g);
		} else if (arch == "breeze_tts") {
			// The TTS model is one checkpoint plus the codec, the tokenizer and
			// the two JSON configs that sit beside them; all of them resolve here
			// so the sampler nodes only ever see the MODEL socket. A blank
			// `config` / `codec_config` keeps the loaders' sibling `config.json`
			// lookup, which is what every shipped layout uses.
			const std::string codec =
			    resolve_model(ctx, param_string(p, "codec", ""), "tts_codec");
			const std::string tokenizer =
			    resolve_model(ctx, param_string(p, "tokenizer", ""), "tts_tokenizer");
			// The JSON roles resolve a *selection*; empty stays empty (the
			// loader's sibling lookup), so this must not go through
			// `resolve_model` (which would substitute the role default).
			const std::string config = resolve_config_file(ctx, param_string(p, "config", ""));
			const std::string codec_config =
			    resolve_config_file(ctx, param_string(p, "codec_config", ""));
			auto t = std::make_shared<BreezeTts>();
			t->open(path, codec, tokenizer, config, codec_config, g);
			m->tts = t;
		} else {
			throw MediaError("unknown arch '" + arch +
			                 "' (qwen_image | minimax_h3 | ace_step15 | breeze_tts)");
		}
		state->models[key] = m;
		return std::vector<Value>{Value::of_object(m)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
