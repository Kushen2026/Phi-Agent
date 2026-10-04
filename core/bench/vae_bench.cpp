// Standalone video-VAE decode driver: `phi_vae --vae <file>[,W,H,frames,tile]>`.
//
// The whole-app binary cannot always be linked while sibling work is in flight,
// and a decode measurement does not need the app anyway: this pulls in the VAE,
// the CUDA runtime and the weight readers, and nothing else. Same instrument as
// `--media-bench`'s PHI_BENCH_VAE block, so the two numbers are comparable.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "bench/bench.hpp"
#include "kernels/kernels.hpp"
#include "models/media_geometry.hpp"
#include "models/media_models.hpp"
#include "models/video_vae.hpp"
#include "runtime/compute.hpp"
#include "runtime/cuda_device.hpp"
#include "runtime/sched.hpp"
#include "runtime/vram_budget.hpp"
#include "util/base.hpp"

using namespace phi;
using namespace phi::media;

int main(int argc, char** argv) {
	std::string spec = "minimax_h3_video_vae_fp16.safetensors,960,540,243";
	std::string base = path_dirname(exe_dir_path());
	if (base.empty()) base = "C:/Phi";
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--vae") && i + 1 < argc) spec = argv[++i];
		else if (!strcmp(argv[i], "--base") && i + 1 < argc) base = argv[++i];
	}

	std::vector<std::string> parts;
	{
		size_t p = 0;
		for (;;) {
			size_t q = spec.find(',', p);
			parts.push_back(spec.substr(p, q == std::string::npos ? std::string::npos : q - p));
			if (q == std::string::npos) break;
			p = q + 1;
		}
	}
	const std::string name = parts[0];
	i64 W = 960, H = 540, frames = 243, tile = 0;
	if (parts.size() > 1 && !parts[1].empty()) W = atoll(parts[1].c_str());
	if (parts.size() > 2 && !parts[2].empty()) H = atoll(parts[2].c_str());
	if (parts.size() > 3 && !parts[3].empty()) frames = atoll(parts[3].c_str());
	if (parts.size() > 4 && !parts[4].empty()) tile = atoll(parts[4].c_str());

	CudaContext cuda;
	try {
		cuda.create(-1);
	} catch (const std::exception& e) {
		fprintf(stderr, "no CUDA device: %s\n", e.what());
		return 1;
	}
	if (!cuda.valid()) {
		fprintf(stderr, "no CUDA device\n");
		return 1;
	}
	ComputeContext compute;
	compute.create(cuda, path_join(base, "data/media_shader_cache"));
	// The same budget the tools and the bench derive, so the residency the decode
	// plans against is the one a real call gets.
	{
		const VideoMemoryBudget b = cuda.query_budget();
		vram_budget().note_os_budget(b.budget);
		vram_budget().refresh();
	}
	printf("device: %s\n", cuda.capabilities().c_str());
	printf("budget: %s\n", format_bytes(vram_budget().limit()).c_str());

	GpuArena wa, aa, keep;
	UploadRing ring;
	wa.init(&compute, arena_chunk_bytes_for(vram_budget().limit()));
	aa.init(&compute, keep_chunk_bytes_for(vram_budget().limit()));
	keep.init(&compute, 16ull << 20);
	ring.init(&compute, default_upload_ring_bytes());
	wa.set_tag("vae.weights");
	aa.set_tag("vae.acts");
	keep.set_tag("vae.keep");
	GpuCtx g{&cuda, &compute, &wa, &aa, &keep, &ring};

	const std::string models_dir = path_join(base, "models");
	std::string path = resolve_media_file(models_dir, name);
	if (path.empty()) path = resolve_media_role(JsonValue::object(), "video_vae", models_dir);
	printf("file  : %s\n", path.c_str());

	const VideoPlan vp = plan_video(W, H, frames);
	const i64 T = vp.video_t, LH = vp.latent_h, LW = vp.latent_w;
	printf("canvas: %lldx%lld -> latent %lldx%lld x %lld, tile %lld\n",
	       (long long)vp.frame.canvas_w, (long long)vp.frame.canvas_h, (long long)LH,
	       (long long)LW, (long long)T, (long long)tile);

	VideoVae vv;
	vv.open(path, &g);
	std::vector<float> z((size_t)(24 * T * LH * LW));
	std::mt19937 rng(1234);
	std::normal_distribution<float> nd(0.0f, 1.0f);
	for (auto& x : z) x = nd(rng);

	const double t0 = media::now_ms();
	std::vector<float> out = vv.decode(z, T, LH, LW, tile);
	const double t1 = media::now_ms();

	double mn = 1e30, mx = -1e30;
	i64 nan = 0;
	for (float v : out) {
		if (!(v == v)) {
			nan++;
			continue;
		}
		mn = std::min(mn, (double)v);
		mx = std::max(mx, (double)v);
	}
	printf("out   : %zu floats, %lld frames, tile %lld, activation %s\n", out.size(),
	       (long long)vp.frames, (long long)vv.last_tile(),
	       format_bytes(vv.last_activation_bytes()).c_str());
	printf("range : [%.4f, %.4f]  nan %lld\n", nan == (i64)out.size() ? 0.0 : mn,
	       nan == (i64)out.size() ? 0.0 : mx, (long long)nan);
	printf("TIME  : %8.1f s\n", (t1 - t0) / 1000.0);
	if (const char* o = getenv("VAE_OUT")) {
		write_file(o, std::string((const char*)out.data(), out.size() * 4));
		printf("wrote : %s\n", o);
	}
	return 0;
}
