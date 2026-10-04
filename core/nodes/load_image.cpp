// Node "MediaLoadImage" — Load Image.
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
#include "nodes/service.hpp"

namespace phi::media {

void register_load_image(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaLoadImage";
	n.category = "image";
	n.title = "加载图像";
	n.title_en = "Load Image";
	n.description = "从磁盘读一张 PNG/JPEG。";
	n.description_en = "Reads a PNG/JPEG from disk.";
	n.outputs = {port("IMAGE", SocketType::Image, false, "the image")};
	{
		JsonValue p = JsonValue::object();
		p["path"] = "absolute path or one relative to the session cwd";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>&) {
		const std::string path = resolve_user_path(ctx, param_string(p, "path", ""));
		auto out = std::make_shared<ImageData>();
		int w = 0, h = 0;
		out->rgb = read_image(path, &w, &h);
		out->w = w;
		out->h = h;
		return std::vector<Value>{Value::of_object(out)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
