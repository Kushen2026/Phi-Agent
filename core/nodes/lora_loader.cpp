// Node "MediaLoraLoader" — LoRA Loader.
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
#include "nodes/service.hpp"

namespace phi::media {

void register_lora_loader(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaLoraLoader";
	n.category = "loaders";
	n.title = "LoRA 加载器";
	n.title_en = "LoRA Loader";
	n.description =
	    "把一组 LoRA 文件读成一条链（按顺序叠加），接到模型/文本编码器加载器上生效。";
	n.description_en =
	    "Reads LoRA files into one chain (applied in order) for a model / CLIP loader.";
	n.outputs = {port("LORA", SocketType::Lora, false, "the LoRA chain")};
	{
		JsonValue p = JsonValue::object();
		p["loras"] = "list of LoRA file names under the models directory";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>&) {
		auto l = std::make_shared<LoraData>();
		l->set = std::make_shared<LoraSet>();
		for (const std::string& raw : param_string_list(p, "loras")) {
			const std::string path = resolve_model(ctx, raw, "image_dit");
			l->set->add(path);
			l->paths.push_back(path);
		}
		return std::vector<Value>{Value::of_object(l)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
