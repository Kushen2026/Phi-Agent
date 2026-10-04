// Node "MediaClipLoader" — CLIP Loader (text encoder).
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
#include "nodes/service.hpp"
#include "models/text_encoder_8b.hpp"
#include "models/text_encoder_32b.hpp"
#include "models/ace_text.hpp"

namespace phi::media {

void register_clip_loader(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaClipLoader";
	n.category = "loaders";
	n.title = "文本编码器加载器";
	n.title_en = "CLIP Loader (text encoder)";
	n.description = "加载文本编码器（Qwen3-VL-8B 或 Qwen3-VL-32B）连同分词器。";
	n.description_en = "Loads a text encoder (Qwen3-VL-8B or Qwen3-VL-32B) plus its tokenizer.";
	n.inputs = {port("lora", SocketType::Lora, true, "LoRA chain applied at load")};
	n.outputs = {port("CLIP", SocketType::Clip, false, "text encoder + tokenizer")};
	{
		JsonValue p = JsonValue::object();
		p["arch"] = "qwen3vl_8b | qwen3vl_32b | ace15";
		p["model"] = "file name (empty = the shipped default)";
		p["vocab"] = "vocab.json (empty = this arch's own tokenizer role)";
		p["merges"] = "merges.txt (empty = this arch's own tokenizer role)";
		p["tokenizer_config"] = "tokenizer_config.json (empty = this arch's own role)";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>& in) {
		const std::string arch = param_string(p, "arch", "qwen3vl_8b");
		const char* role = arch == "qwen3vl_32b" ? "video_te"
		                   : arch == "ace15"      ? "music_te"
		                                          : "image_te";
		const std::string path = resolve_model(ctx, param_string(p, "model", ""), role);
		// ACE ships two towers: the 0.6B embedder and the 4B audio-code LM, so
		// its CLIP entry carries a second path (ComfyUI's DualCLIPLoader).
		const std::string lm_path =
		    arch == "ace15" ? resolve_model(ctx, param_string(p, "model_lm", ""), "music_lm")
		                    : std::string();
		// The three tokenizer files this chain reads. A node parameter wins (a
		// hand-wired graph), otherwise the chain's own roles.
		const TokenizerFiles tf = tokenizer_files_of(ctx, arch,
		                                            param_string(p, "vocab", ""),
		                                            param_string(p, "merges", ""),
		                                            param_string(p, "tokenizer_config", ""));
		auto lora = as_lora(in[0]);
		const std::string lora_key = lora ? std::to_string((uintptr_t)lora->set.get()) : "-";
		const std::string key = arch + "|" + path + "|" + lm_path + "|" + tf.vocab + "|" +
		                        tf.merges + "|" + tf.config + "|" + lora_key;
		auto state = graph_state(ctx);
		auto it = state->clips.find(key);
		if (it != state->clips.end()) return std::vector<Value>{Value::of_object(it->second)};

		GpuCtx* g = graph_gpu(ctx);
		auto c = std::make_shared<ClipData>();
		c->arch = arch;
		c->path = path;
		c->tokenizer_files = tf;
		c->tokenizer = load_tokenizer(ctx, tf);
		if (lora) c->loras = lora->set;
		if (arch == "qwen3vl_8b") {
			c->te8 = std::make_shared<TextEncoder8B>();
			c->te8->set_loras(c->loras.get());
			c->te8->open(path, g);
			c->hidden = c->te8->config().hidden;
		} else if (arch == "qwen3vl_32b") {
			c->te32 = std::make_shared<TextEncoder32B>();
			c->te32->set_loras(c->loras.get());
			c->te32->open(path, g);
			c->hidden = c->te32->config().hidden;
		} else if (arch == "ace15") {
			c->ace_te = std::make_shared<AceQwen3>();
			c->ace_te->open(path, g);
			c->ace_lm = std::make_shared<AceLm>();
			c->ace_lm->open(lm_path, g);
			c->hidden = c->ace_te->config().hidden;
		} else {
			throw MediaError("unknown text encoder arch '" + arch + "'");
		}
		state->clips[key] = c;
		return std::vector<Value>{Value::of_object(c)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
