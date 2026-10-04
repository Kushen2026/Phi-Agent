// Node "MediaConstInt" — Int.
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
#include "nodes/service.hpp"

namespace phi::media {

void register_const_int(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaConstInt";
	n.category = "utils";
	n.title = "整数";
	n.title_en = "Int";
	n.description = "一个整数常量，用来喂给另一个节点的整数输入。";
	n.description_en = "An integer constant to feed another node's integer input.";
	n.outputs = {port("INT", SocketType::Int, false, "the value")};
	{
		JsonValue p = JsonValue::object();
		p["value"] = "the value";
		n.params = p;
	}
	n.run = [](GraphContext&, const JsonValue& p, const std::vector<Value>&) {
		return std::vector<Value>{Value::of_int(param_int(p, "value", 0))};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
