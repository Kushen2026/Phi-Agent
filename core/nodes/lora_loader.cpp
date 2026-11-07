// Node "MediaLoraLoader" — LoRA Loader.
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
//
// The two parameters are parallel lists, exactly like ComfyUI's LoraLoader
// widgets: `loras` names the files, `lora_strengths` scales each one (default
// 1.0, ComfyUI's `strength_model` / `strength_clip`). A missing / short / non
// numeric strength list is filled with the default, so a graph written before
// the strength existed loads the same weights it always did.
#include "nodes/service.hpp"

namespace phi::media {

void register_lora_loader(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaLoraLoader";
	n.category = "loaders";
	n.title = "LoRA 加载器";
	n.title_en = "LoRA Loader";
	n.description =
	    "把一组 LoRA 文件读成一条链（按顺序叠加），接到模型/文本编码器加载器上生效。"
	    "每个文件可单独设定强度（默认 1.0，负值反向、0 表示该文件不参与）。";
	n.description_en =
	    "Reads LoRA files into one chain (applied in order) for a model / CLIP loader. "
	    "Each file has its own strength (default 1.0; negative inverts, 0 drops the file).";
	n.outputs = {port("LORA", SocketType::Lora, false, "the LoRA chain")};
	{
		JsonValue p = JsonValue::object();
		p["loras"] = "list of LoRA file names under the models directory";
		p["lora_strengths"] =
		    "parallel list of strengths (default 1.0); index-matched to `loras`";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>&) {
		auto l = std::make_shared<LoraData>();
		l->set = std::make_shared<LoraSet>();
		// The strength list is read as numbers, not through param_string_list: the
		// values are floats, and a list entry may be a bare number or an object's
		// `value`. Anything unreadable is the default - a strength is a scaler, and
		// guessing one would be worse than doing what the file was trained for.
		const JsonValue* strengths = p.find("lora_strengths");
		const std::vector<std::string> names = param_string_list(p, "loras");
		for (size_t i = 0; i < names.size(); i++) {
			float strength = kLoraStrengthDefault;
			if (strengths && strengths->is_array() && i < strengths->size()) {
				const JsonValue& s = (*strengths)[i];
				const JsonValue* num = s.is_object() ? s.find("value") : &s;
				if (num && num->is_number()) strength = (float)num->as_number(kLoraStrengthDefault);
			}
			// ComfyUI's LoraLoader returns its inputs untouched when every strength
			// is 0, i.e. the file is not part of the model at all. Doing the same
			// here keeps a zeroed slot from costing a checkpoint read.
			if (strength == 0.0f) continue;
			const std::string path = resolve_model(ctx, names[i], "image_dit");
			l->set->add(path, strength);
			l->paths.push_back(path);
			l->strengths.push_back(strength);
		}
		return std::vector<Value>{Value::of_object(l)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
