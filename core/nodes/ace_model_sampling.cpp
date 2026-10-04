// Node "MediaAceModelSampling" — Model Sampling (AuraFlow shift).
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
//
// ComfyUI's `ModelSamplingAuraFlow` (comfy_extras/nodes_model_advanced.py) is
// `ModelSamplingDiscreteFlow(shift = ...)` with multiplier 1.0, applied to the
// MODEL. The sigma grid every consumer builds then reads that shift, so a
// distilled / few-step ACE LoRA — which ships its own shift — is one node in
// front of the scheduler instead of a rebuilt chain. The released 3.0 is the
// loader's default and is what `ACE-Step-音乐生成.json` wires.
#include "nodes/service.hpp"

namespace phi::media {

void register_ace_model_sampling(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaAceModelSampling";
	n.category = "sampling";
	n.title = "模型采样（AuraFlow shift）";
	n.title_en = "Model Sampling (AuraFlow shift)";
	n.description = "给 ACE-Step 1.5 的 MODEL 打上 flow shift（ModelSamplingAuraFlow，默认 3.0）。";
	n.description_en =
	    "Patches the ACE-Step 1.5 MODEL with its flow shift (ModelSamplingAuraFlow, default 3.0).";
	n.inputs = {port("model", SocketType::Model, false, "the ACE DiT")};
	n.outputs = {port("MODEL", SocketType::Model, false, "the patched model")};
	{
		JsonValue p = JsonValue::object();
		p["shift"] = "the flow shift (3.0)";
		n.params = p;
	}
	n.run = [](GraphContext&, const JsonValue& p, const std::vector<Value>& in) {
		auto m = as_model(in[0]);
		if (!m || !m->ace) throw MediaError("AceModelSampling: model is not an ACE DiT");
		const double shift = param_number(p, "shift", 3.0);
		if (!(shift > 0.0)) throw MediaError("AceModelSampling: shift must be > 0");
		m->ace_shift = (float)shift;
		return std::vector<Value>{Value::of_object(m)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
