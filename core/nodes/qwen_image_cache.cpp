// Node "MediaQwenImage21Cache" — Qwen Image 2.1 Cache (prefix K/V).
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
//
// ComfyUI's `QwenImage21Cache` enables the DiT's prefix K/V cache: the prompt and
// the reference blocks are processed under the *t = 0* modulation with a causal /
// block-causal mask, so their keys and values are the same on every sampling step.
// Keeping them turns each step from "run prompt + references + target through 32
// blocks" into "run the target through 32 blocks, attending into the kept prefix".
//
// The node is a *patch* on the MODEL socket, exactly like `MediaH3SigmaShift`: it
// returns the same loaded backbone with the flag set, so it can be dropped into a
// graph in front of the sampler (or left out, and the chain is what it was).
//
// Two of the reference node's knobs are deliberately not offered here, because in
// this engine they have no meaning:
//
//   * `device` (auto / gpu / cpu): the cache lives in device memory or not at all.
//     `auto` in the reference means "use spare VRAM, else host RAM, else
//     recompute"; here the VRAM accountant answers the same question - a charge
//     that would cross the limit is refused and the run recomputes the prefix -
//     and a host-resident cache would have to be re-uploaded through PCIe on every
//     block of every step, which costs more than the arithmetic it saves.
//   * `dtype` (default / int8 / int4): the cache is kept in the same fp16 the
//     attention itself runs at, which is what makes a cached step *bit-identical*
//     to an uncached one (the engine's correctness contract for this switch), and
//     the accountant already decides whether the cache fits at all.
//
// The trade it makes is real and worth saying out loud: the cache costs VRAM, and
// this engine's scarce resource on a small card is the room the streamed weights
// need (a block that is not resident is re-read on every step). It pays off when
// the weights fit the card, and the plan quietly leaves it out when they do not.
#include "nodes/service.hpp"
#include "models/image_dit.hpp"

namespace phi::media {

void register_qwen_image_cache(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaQwenImage21Cache";
	n.category = "sampling";
	n.title = "Qwen Image 2.1 前缀 KV 缓存";
	n.title_en = "Qwen Image 2.1 Cache";
	n.description =
	    "开启 Qwen-Image 2.1 DiT 的前缀 KV 缓存：提示词与参考块的 K/V 在每步都相同，"
	    "缓存一次后每步只跑目标行，注意力读取缓存的前缀（t=0 调制、块因果掩码，因此结果与不缓存逐位相同）。"
	    "缓存以 fp16 常驻显存，由显存账本决定是否放得下：放不下就自动退回每步重算。";
	n.description_en =
	    "Enables the Qwen-Image 2.1 DiT's prefix K/V cache: the prompt's and the references' "
	    "keys/values are the same on every step, so after the first they are kept and each step "
	    "runs the target rows alone, attending into the kept prefix (t = 0 modulation and the "
	    "block-causal mask, which is why a cached step is bit-identical to an uncached one). The "
	    "cache is fp16 resident VRAM, admitted by the VRAM accountant - when it does not fit the "
	    "run falls back to recomputing the prefix every step.";
	n.inputs = {port("model", SocketType::Model, false, "the Qwen-Image backbone")};
	n.outputs = {port("MODEL", SocketType::Model, false, "the same backbone, cache enabled")};
	{
		JsonValue p = JsonValue::object();
		p["enabled"] = "false disables the cache (the node is then a pass-through)";
		n.params = p;
	}
	n.run = [](GraphContext&, const JsonValue& p, const std::vector<Value>& in) {
		auto m = as_model(in[0]);
		if (!m) throw MediaError("QwenImage21Cache: model input is not a MODEL");
		if (m->arch != "qwen_image")
			throw MediaError("QwenImage21Cache: the model is not Qwen-Image (arch '" + m->arch +
			                 "')");
		auto out = std::make_shared<ModelData>(*m);   // shares the loaded DiT
		out->image_prefix_cache = param_bool(p, "enabled", true);
		return std::vector<Value>{Value::of_object(out)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
