// Node "MediaSeed" — Seed.
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
#include "nodes/service.hpp"

namespace phi::media {

void register_seed(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaSeed";
	n.category = "utils";
	n.title = "随机种子";
	n.title_en = "Seed";
	n.description = "产生一个种子（0 = 每次随机抽一个）。";
	n.description_en = "Produces a seed (0 = draw a fresh one).";
	n.outputs = {port("INT", SocketType::Int, false, "the seed")};
	{
		JsonValue p = JsonValue::object();
		p["seed"] = "0 = draw one";
		n.params = p;
	}
	n.run = [](GraphContext&, const JsonValue& p, const std::vector<Value>&) {
		u64 s = (u64)param_int(p, "seed", 0);
		if (s == 0) s = make_media_seed();
		return std::vector<Value>{Value::of_int((int64_t)s)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
