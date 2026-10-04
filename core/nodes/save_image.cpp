// Node "MediaSaveImage" — Save Image.
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
#include "nodes/service.hpp"

namespace phi::media {

void register_save_image(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaSaveImage";
	n.category = "image";
	n.title = "保存图像";
	n.title_en = "Save Image";
	n.description = "把一张图像写成 PNG。";
	n.description_en = "Writes an image as a PNG.";
	n.inputs = {port("image", SocketType::Image, false, "the image")};
	n.outputs = {port("path", SocketType::String, false, "the written path")};
	{
		JsonValue p = JsonValue::object();
		p["path"] = "output PNG path";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>& in) {
		auto img = as_image(in[0]);
		if (!img || img->rgb.empty()) throw MediaError("SaveImage: nothing to save");
		const std::string path = resolve_user_path(ctx, param_string(p, "path", "out.png"));
		const size_t slash = path.find_last_of("/\\");
		if (slash != std::string::npos && slash > 0) mkdirs(path.substr(0, slash));
		write_png(path, img->rgb, (int)img->w, (int)img->h);
		return std::vector<Value>{Value::of_string(path)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
