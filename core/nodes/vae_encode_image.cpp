// Node "MediaVaeEncodeImage" — VAE Encode (image).
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
#include "nodes/service.hpp"
#include "models/image_vae.hpp"

namespace phi::media {

void register_vae_encode_image(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaVaeEncodeImage";
	n.category = "vae";
	n.title = "VAE 编码（图像）";
	n.title_en = "VAE Encode (image)";
	n.description = "把一张 8 位图像编码成 Qwen-Image 潜变量（用于参考图/重绘）。";
	n.description_en = "Encodes an 8-bit image into a Qwen-Image latent (for references/edit).";
	n.inputs = {port("vae", SocketType::Vae, false, "the image VAE"),
	            port("image", SocketType::Image, false, "the image")};
	n.outputs = {port("LATENT", SocketType::Latent, false, "the latent")};
	n.run = [](GraphContext& ctx, const JsonValue&, const std::vector<Value>& in) {
		auto vae = as_vae(in[0]);
		auto img = as_image(in[1]);
		if (!vae || vae->role != "image") throw MediaError("VaeEncodeImage: need an image VAE");
		if (!img) throw MediaError("VaeEncodeImage: missing an image");
		const i64 W = (img->w / 16) * 16, H = (img->h / 16) * 16;
		if (W <= 0 || H <= 0) throw MediaError("VaeEncodeImage: image is too small");
		std::vector<float> rgba((size_t)4 * W * H, 1.0f);
		for (i64 y = 0; y < H; y++)
			for (i64 x = 0; x < W; x++)
				for (i64 c = 0; c < 3; c++) {
					const float f = (float)img->rgb[(size_t)((y * img->w + x) * 3 + c)] / 255.0f;
					rgba[(size_t)((c * H + y) * W + x)] = f * 2.0f - 1.0f;
				}
		auto out = std::make_shared<LatentData>();
		out->arch = "image";
		out->x = vae->image->encode(rgba, H, W);
		out->c = 64;
		out->h = H / 16;
		out->w = W / 16;
		out->req_w = img->w;
		out->req_h = img->h;
		out->canvas_w = W;
		out->canvas_h = H;
		vae->image->release_activation_memory();
		release_shared_arenas(static_cast<GpuCtx*>(ctx.gpu));
		return std::vector<Value>{Value::of_object(out)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
