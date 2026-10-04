#include "bench/media_bench.hpp"

#include <windows.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>

#include "util/base.hpp"
#include "bench/bench.hpp"
#include "runtime/compute.hpp"
#include "graph/graph.hpp"
#include "graph/graph_media.hpp"
#include "graph/graph_nodes.hpp"
#include "graph/graph_workflows.hpp"
#include "models/media_models.hpp"
#include "models/media_geometry.hpp"
#include "models/image_dit.hpp"
#include "models/image_vae.hpp"
#include "models/video_vae.hpp"
#include "util/json.hpp"
#include "runtime/cuda_api.hpp"
#include "runtime/cuda_device.hpp"
#include "host/mtx.hpp"
#include "host/quant.hpp"
#include "runtime/sched.hpp"   // GpuArena/UploadRing + default_upload_ring_bytes
#include "host/st.hpp"
#include "text/tokenizer.hpp"
#include "runtime/vram_budget.hpp"
#include "kernels/kernels.hpp"
#include "models/model_common.hpp"
#include "kernels/h3_kernels.hpp"

namespace phi::media {

namespace {

// Default checkpoint locations (after the M0 resource move they live in
// <base>/models; before it they are still under ComfyUI/models).
//
// The list is the *installed* set: the image chain moved from Z-Image Turbo
// (`z_image_turbo_int8_convrot` + the Qwen3-4B tower + the Flux AE) to
// Qwen-Image-2.1 (`qwen_image_2.1_int8_convrot` + the Qwen3-VL-8B tower +
// `qwen_image_2.1_vae_bf16`). The retired file names are kept here as a
// reminder of what was replaced; they do not exist on disk anymore.
// Naming them caused the streaming pass to report a shorter list than the
// model files the app really loads - silently, because a path that does not
// exist is skipped.
std::vector<std::string> default_model_paths(const std::string& base) {
	std::vector<std::string> out;
	const char* rel[] = {
	    "models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
	    "models/diffusion_models/minimax_h3_hybrid_ref2va_b25-49_pruned_int8_convrot.safetensors",
	    "models/text_encoders/qwen3vl_8b_int8_convrot.safetensors",
	    "models/text_encoders/minimax_h3_qwen3vl_32b_convrot_int4.safetensors",
	    "models/vae/qwen_image_2.1_vae_bf16.safetensors",
	    "models/vae/minimax_h3_video_vae_fp16.safetensors",
	    "models/vae/minimax_h3_audio_vae_fp32.safetensors",
	};
	for (const char* r : rel) {
		std::string p = path_join(base, r);
		if (path_exists(p)) out.push_back(p);
	}
	if (!out.empty()) return out;
	// fall back to the live ComfyUI tree (pre-move layout)
	const char* old[] = {
	    "C:/ComfyUI/models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
	    "C:/ComfyUI/models/diffusion_models/"
	    "minimax_h3_hybrid_ref2va_b25-49_pruned_int8_convrot.safetensors",
	    "C:/ComfyUI/models/text_encoders/qwen3vl_8b_int8_convrot.safetensors",
	    "C:/ComfyUI/models/text_encoders/minimax_h3_qwen3vl_32b_convrot_int4.safetensors",
	    "C:/ComfyUI/models/vae/qwen_image_2.1_vae_bf16.safetensors",
	    "C:/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors",
	    "C:/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors",
	};
	for (const char* r : old) {
		if (path_exists(r)) out.push_back(r);
	}
	return out;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b);

// ── native kernel gates ────────────────────────────────────────────────────
//
// These live here rather than in compute.* because they are verification
// scaffolding: they build random operands, run the CPU reference on the *same*
// quantised bytes the GPU gets, and report the disagreement.

struct NativeQuantResult {
	bool ok = false;
	std::string error;
	i64 rows = 0, cols = 0;
	int mismatches = 0;
	int max_diff = 0;
	double scale_rel = 0;
	double ms = 0;   // native_quant_bench only
};

struct NativeGemmResult {
	bool ok = false;
	std::string error;
	i64 m = 0, n = 0, k = 0;
	double ms = 0, tops = 0, rel_l2 = 0;
};

NativeQuantResult native_quant_check(ComputeContext& ctx, i64 rows, i64 k) {
	NativeQuantResult r;
	r.rows = rows;
	r.cols = k;
	try {
		QuantSpec spec;
		spec.format = "int8_tensorwise";
		spec.convrot = true;
		spec.convrot_groupsize = 256;
		spec.valid = true;

		std::mt19937 rng(7);
		std::normal_distribution<float> nd(0.0f, 2.0f);
		std::vector<float> x((size_t)rows * k);
		for (auto& v : x) v = nd(rng);

		QuantInt8 want = quantize_activation(x.data(), rows, k, spec);

		GpuBuffer* dx = ctx.create_device_buffer((u64)rows * k * 4, true);
		GpuBuffer* dq = ctx.create_device_buffer((u64)rows * k, true);
		GpuBuffer* ds = ctx.create_device_buffer((u64)rows * 4, true);
		ctx.upload(dx, x.data(), (u64)rows * k * 4);
		QuantConvrotArgs qa;
		qa.x = dx;
		qa.q = dq;
		qa.s = ds;
		qa.rows = rows;
		qa.K = k;
		ctx.begin();
		dispatch_quant_convrot(ctx, qa);
		ctx.submit();

		std::vector<int8_t> got_q((size_t)rows * k);
		std::vector<float> got_s((size_t)rows);
		ctx.download(dq, got_q.data(), (u64)rows * k);
		ctx.download(ds, got_s.data(), (u64)rows * 4);
		ctx.release_buffer(dx);
		ctx.release_buffer(dq);
		ctx.release_buffer(ds);

		for (size_t i = 0; i < got_q.size(); i++) {
			int d = std::abs((int)got_q[i] - (int)want.q[i]);
			if (d) r.mismatches++;
			if (d > r.max_diff) r.max_diff = d;
		}
		double num = 0, den = 0;
		for (i64 i = 0; i < rows; i++) {
			double d = (double)got_s[i] - want.scale[i];
			num += d * d;
			den += (double)want.scale[i] * want.scale[i];
		}
		r.scale_rel = den > 0 ? std::sqrt(num / den) : 0.0;
		r.ok = true;
	} catch (const std::exception& e) {
		r.error = e.what();
	}
	return r;
}

// Timed form of the quantiser, same operands, no correctness check: the number
// the DiT's per-call split attributes to it.
NativeQuantResult native_quant_bench(ComputeContext& ctx, i64 rows, i64 k, int iters) {
	NativeQuantResult r;
	r.rows = rows;
	r.cols = k;
	try {
		std::mt19937 rng(7);
		std::normal_distribution<float> nd(0.0f, 2.0f);
		std::vector<float> x((size_t)rows * k);
		for (auto& v : x) v = nd(rng);

		GpuBuffer* dx = ctx.create_device_buffer((u64)rows * k * 4, true);
		GpuBuffer* dq = ctx.create_device_buffer((u64)rows * k, true);
		GpuBuffer* ds = ctx.create_device_buffer((u64)rows * 4, true);
		ctx.upload(dx, x.data(), (u64)rows * k * 4);
		QuantConvrotArgs qa;
		qa.x = dx;
		qa.q = dq;
		qa.s = ds;
		qa.rows = rows;
		qa.K = k;
		ctx.begin();
		dispatch_quant_convrot(ctx, qa);   // warm-up (compiles the PSO)
		ctx.submit();
		double t0 = now_ms();
		for (int i = 0; i < iters; i++) {
			ctx.begin();
			dispatch_quant_convrot(ctx, qa);
			ctx.submit();
		}
		double t1 = now_ms();
		r.ms = (t1 - t0) / iters;
		ctx.release_buffer(dx);
		ctx.release_buffer(dq);
		ctx.release_buffer(ds);
		r.ok = true;
	} catch (const std::exception& e) {
		r.error = e.what();
	}
	return r;
}

NativeGemmResult native_int8_gemm(ComputeContext& ctx, i64 M, i64 N, i64 K, int iters) {
	NativeGemmResult r;
	r.m = M;
	r.n = N;
	r.k = K;
	try {
		QuantSpec spec;
		spec.format = "int8_tensorwise";
		spec.convrot = true;
		spec.convrot_groupsize = 256;
		spec.valid = true;

		std::mt19937 rng(11);
		std::normal_distribution<float> nd(0.0f, 1.0f);
		std::vector<float> x((size_t)M * K), w((size_t)N * K);
		for (auto& v : x) v = nd(rng);
		for (auto& v : w) v = nd(rng);

		QuantInt8 qa = quantize_activation(x.data(), M, K, spec);
		QuantInt8 qw = quantize_weight(w.data(), N, K, spec);
		std::vector<float> ref((size_t)M * N, 0.0f);
		gemm_int8(qa.q.data(), qa.scale.data(), qw.q.data(), qw.scale.data(), M, N, K,
		          ref.data());

		Int8GemmArgs args;
		args.a = ctx.create_device_buffer((u64)M * K, true);
		args.b = ctx.create_device_buffer((u64)N * K, true);
		args.sa = ctx.create_device_buffer((u64)M * 4, true);
		args.sb = ctx.create_device_buffer((u64)N * 4, true);
		args.c = ctx.create_device_buffer((u64)M * N * 4, true);
		args.M = M;
		args.N = N;
		args.K = K;
		ctx.upload(args.a, qa.q.data(), (u64)M * K);
		ctx.upload(args.b, qw.q.data(), (u64)N * K);
		ctx.upload(args.sa, qa.scale.data(), (u64)M * 4);
		ctx.upload(args.sb, qw.scale.data(), (u64)N * 4);

		ctx.begin();
		dispatch_int8_gemm(ctx, args);  // warm-up (compiles the PSO on first use)
		ctx.submit();

		double t0 = now_ms();
		for (int i = 0; i < iters; i++) {
			ctx.begin();
			dispatch_int8_gemm(ctx, args);
			ctx.submit();
		}
		double t1 = now_ms();
		r.ms = (t1 - t0) / iters;
		r.tops = 2.0 * (double)M * (double)N * (double)K / (r.ms / 1000.0) / 1e12;

		std::vector<float> got((size_t)M * N);
		ctx.download(args.c, got.data(), (u64)M * N * 4);
		// Free before the next shape: three shapes at 4096x3840 alone strand
		// ~190 MB that the image chain later cannot allocate.
		ctx.release_buffer(args.a);
		ctx.release_buffer(args.b);
		ctx.release_buffer(args.sa);
		ctx.release_buffer(args.sb);
		ctx.release_buffer(args.c);
		r.rel_l2 = rel_l2(ref, got);
		r.ok = true;
	} catch (const std::exception& e) {
		r.error = e.what();
	}
	return r;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
	double num = 0, den = 0;
	for (size_t i = 0; i < a.size(); i++) {
		double d = (double)a[i] - (double)b[i];
		num += d * d;
		den += (double)b[i] * (double)b[i];
	}
	return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

// ── streaming throughput ───────────────────────────────────────────────────

struct StreamResult {
	bool ok = false;
	std::string path;
	u64 bytes = 0;
	double ms = 0;
	double gbps = 0;
	u64 tensor_count = 0;
	// The same window through `read_file` (ReadFile at 4 MB granularity) instead
	// of the mapping. The streamer picks the faster one for block weights, and
	// printing both is what makes that choice auditable rather than folklore.
	u64 file_bytes = 0;
	double file_ms = 0;
	double file_gbps = 0;
};

// Maps a checkpoint and reads a window of it the way the weight streamer will
// (sequential, chunked), which is what the scheduler's window size is derived
// from. The OS page cache makes the second pass free, so this reports the cold
// number on the first window only.
StreamResult measure_stream(const std::string& path, u64 window_bytes) {
	StreamResult r;
	r.path = path;
	try {
		SafeTensors st;
		double t0 = now_ms();
		st.open(path);
		double t1 = now_ms();
		r.tensor_count = st.tensors().size();
		printf("      open+map %8.1f ms  (%zu tensors)\n", t1 - t0, st.tensors().size());

		u64 total = std::min<u64>(window_bytes, st.file_size());
		HostBuffer buf(std::min<u64>(total, 8ull * 1024 * 1024));
		const u64 chunk = buf.size();
		u64 done = 0;
		double s0 = now_ms();
		while (done < total) {
			u64 n = std::min<u64>(chunk, total - done);
			st.read(done, (size_t)n, buf.data());
			// touch the data so the compiler cannot elide the copy
			volatile u8 sink = ((const u8*)buf.data())[0];
			(void)sink;
			done += n;
		}
		double s1 = now_ms();
		r.bytes = done;
		r.ms = s1 - s0;
		r.gbps = (double)done / (r.ms / 1000.0) / 1e9;

		// Second pass, then the same window through ReadFile. The *first* pass of
		// each shape is the cold one, so run the file path first and the mapping
		// path second: an mmap page fault that finds the page already resident
		// (because read_file just read it) would otherwise flatter the mapping.
		st.reset_file_read_stats();
		const double f0 = now_ms();
		u64 fdone = 0;
		while (fdone < total) {
			const u64 n = std::min<u64>(chunk, total - fdone);
			st.read_file(fdone, (size_t)n, buf.data());
			volatile u8 sink = ((const u8*)buf.data())[0];
			(void)sink;
			fdone += n;
		}
		const double f1 = now_ms();
		r.file_bytes = fdone;
		r.file_ms = f1 - f0;
		r.file_gbps = (double)fdone / (std::max(0.001, r.file_ms) / 1000.0) / 1e9;
		r.ok = true;
	} catch (const std::exception& e) {
		printf("      error: %s\n", e.what());
	}
	return r;
}

// ── quantisation self-test (mirrors tests/media_quant_test.cpp) ────────────

struct QuantCheck {
	double rot_identity_rel = 1.0;
	double plain_int8_rel = 1.0;
	double convrot_int8_rel = 1.0;
	double weight_roundtrip_rel = 1.0;
	bool pass = false;
};

QuantCheck quant_selftest() {
	QuantCheck c;
	std::mt19937 rng(20240912);
	std::normal_distribution<float> nd(0.f, 1.f);
	const i64 M = 16, N = 256, K = 3840;
	std::vector<float> x((size_t)(M * K)), w((size_t)(N * K));
	for (auto& v : x) v = nd(rng);
	for (i64 m = 0; m < M; m++) {
		for (int o = 0; o < 24; o++) x[(size_t)(m * K + o * 137)] *= 20.0f;
	}
	for (auto& v : w) v = nd(rng) * 0.05f;

	std::vector<float> xr(x.size()), wr(w.size());
	convrot_forward(x.data(), xr.data(), M, K, 256);
	convrot_forward(w.data(), wr.data(), N, K, 256);
	std::vector<float> ref((size_t)(M * N)), rotated((size_t)(M * N));
	gemm_f32(x.data(), w.data(), M, N, K, ref.data());
	gemm_f32(xr.data(), wr.data(), M, N, K, rotated.data());
	c.rot_identity_rel = rel_l2(rotated, ref);

	std::vector<float> plain((size_t)(M * N)), rot((size_t)(M * N));
	QuantSpec spec;
	spec.format = "int8_tensorwise";
	spec.valid = true;
	gemm_int8_convrot(x.data(), w.data(), M, N, K, spec, plain.data());
	spec.convrot = true;
	gemm_int8_convrot(x.data(), w.data(), M, N, K, spec, rot.data());
	c.plain_int8_rel = rel_l2(plain, ref);
	c.convrot_int8_rel = rel_l2(rot, ref);

	QuantInt8 qw = quantize_weight(w.data(), N, K, spec);
	std::vector<float> back((size_t)(N * K)), unrot((size_t)(N * K));
	dequantize_int8(qw, back.data());
	convrot_inverse(back.data(), unrot.data(), N, K, 256);
	double num = 0, den = 0;
	for (size_t i = 0; i < w.size(); i++) {
		double d = (double)unrot[i] - (double)w[i];
		num += d * d;
		den += (double)w[i] * (double)w[i];
	}
	c.weight_roundtrip_rel = std::sqrt(num / den);
	// rotation must be algebraically exact; quantisation error is expected but
	// must not be worse than the unrotated path
	c.pass = c.rot_identity_rel < 1e-4 && c.convrot_int8_rel <= c.plain_int8_rel * 1.25 &&
	         c.weight_roundtrip_rel < 0.02;
	return c;
}

// ── real checkpoint inspection ─────────────────────────────────────────────

void inspect_checkpoint(const std::string& path) {
	try {
		SafeTensors st;
		st.open(path);
		printf("  %s\n", st.describe().c_str());
		size_t quantised = 0;
		u64 quant_bytes = 0;
		u64 total_bytes = 0;
		bool convrot_seen = false;
		int groupsize = 0;
		std::string sample;
		for (const auto& t : st.tensors()) {
			total_bytes += t.nbytes;
			if (!ends_with(t.name, ".weight")) continue;
			QuantSpec spec = st.quant_spec_of(t.name);
			if (!spec.valid) continue;
			quantised++;
			quant_bytes += t.nbytes;
			if (spec.convrot) {
				convrot_seen = true;
				groupsize = spec.convrot_groupsize;
				if (sample.empty()) sample = t.name;
			}
		}
		printf("      quantised weights: %zu (%s of %s) convrot=%s group=%d\n", quantised,
		       format_bytes(quant_bytes).c_str(), format_bytes(total_bytes).c_str(),
		       convrot_seen ? "yes" : "no", groupsize);
		if (!sample.empty()) printf("      sample: %s\n", sample.c_str());
	} catch (const std::exception& e) {
		printf("      error: %s\n", e.what());
	}
}

}  // namespace

int media_bench_main(const BenchOptions& opts) {
	// A video run takes minutes and is watched from a shell that redirects
	// stdout; block buffering would hide the per-step lines until the process
	// exits, which is exactly the information the run exists to produce.
	setvbuf(stdout, nullptr, _IONBF, 0);
	setvbuf(stderr, nullptr, _IONBF, 0);
	bool image_ok = true;
	bool video_ok = true;
	printf("=== phi media engine benchmark ===\n");
	std::string base = path_dirname(exe_dir_path());
	printf("base dir: %s\n\n", base.c_str());

	StageTimer timer;
	VramTracker vram;

	// ── environment ──
	printf("-- environment --\n");
	{
		auto s = timer.scope("env.adapters");
		auto adapters = enumerate_adapters();
		for (const auto& a : adapters) printf("  %s\n", a.describe().c_str());
		if (adapters.empty()) printf("  (no CUDA adapters!)\n");
	}
	{
		auto s = timer.scope("env.nvrtc");
		printf("  nvrtc %s, driver API %s\n", cuda_api().nvrtc_version().c_str(),
		       cuda_api().driver_version().c_str());
		printf("  headers %s\n",
		       cuda_api().include_dir().empty() ? "(not found)"
		                                         : cuda_api().include_dir().c_str());
	}

	CudaContext cuda;
	
	bool have_cuda = false;
	double best_int8_tops = 0;
	int failures = 0;
	{
		auto s = timer.scope("env.cuda_create");
		try {
			cuda.create(-1);
			have_cuda = true;
			printf("  CUDA device: %s\n", cuda.adapter().name.c_str());
			printf("  capabilities: %s\n", cuda.capabilities().c_str());
			VideoMemoryBudget b = cuda.query_budget();
			printf("  driver budget %s, current usage %s, reservation %s\n",
			       format_bytes(b.budget).c_str(), format_bytes(b.current_usage).c_str(),
			       format_bytes(b.available_reservation).c_str());
			// The number the accountant actually applies (see
			// VramBudget::note_os_budget), so the report and the enforcement agree:
			// the driver's budget for this process less the safety margin, capped by
			// the ceiling. Inside the app that margin is what keeps the compositor and
			// the UI's repaints from demoting our pages - a demoted page read is a
			// 13 s reduction, i.e. a TDR, not a slowdown.
			const u64 budget = VramBudget::budget_limit(b.budget);
			printf("  scheduler VRAM budget: %s (%d%% of this process' %s driver budget, "
			       "capped at the %llu MB ceiling)\n",
			       format_bytes(budget).c_str(), VramBudget::budget_percent(),
			       format_bytes(b.budget).c_str(),
			       (unsigned long long)(VramBudget::hard_cap() >> 20));
			vram.note_limit(budget);
			// The same re-read a media tool does on every invocation: the card's own
			// total (what the ceiling is derived from), the driver's current budget and
			// the limit the two produce. The tools and the bench have to agree here or
			// the acceptance numbers describe a different machine.
			const VramBudget::Environment env = vram_budget().refresh();
			printf("  environment: %s, VRAM %s (%s free), ceiling %s, limit %s\n",
			       cuda.adapter().name.c_str(), format_bytes(env.device_total).c_str(),
			       format_bytes(env.device_free).c_str(), format_bytes(env.hard_cap).c_str(),
			       format_bytes(env.limit).c_str());
		} catch (const std::exception& e) {
			printf("  CUDA unavailable: %s\n", e.what());
		}
	}

	// ── checkpoints ──
	std::vector<std::string> paths = opts.model_paths;
	if (paths.empty()) paths = default_model_paths(base);
	printf("\n-- checkpoints (%zu) --\n", paths.size());
	for (const auto& p : paths) {
		auto s = timer.scope("checkpoint.inspect");
		inspect_checkpoint(p);
	}

	// ── weight streaming ──
	printf("\n-- weight streaming --\n");
	if (paths.empty()) {
		printf("  no checkpoint found; skipping\n");
	} else {
		const std::string& big = paths.size() > 1 ? paths[1] : paths[0];
		u64 window = opts.quick ? 256ull * 1024 * 1024 : 1024ull * 1024 * 1024;
		printf("  reading %s of %s\n", format_bytes(window).c_str(),
		       path_basename(big).c_str());
		auto s = timer.scope("weights.stream");
		StreamResult r = measure_stream(big, window);
		if (r.ok) {
			printf("      mapping (memcpy)  %s in %.1f ms -> %.2f GB/s\n",
			       format_bytes(r.bytes).c_str(), r.ms, r.gbps);
			printf("      file    (ReadFile) %s in %.1f ms -> %.2f GB/s\n",
			       format_bytes(r.file_bytes).c_str(), r.file_ms, r.file_gbps);
			printf("      -> the block streamer reads weights with %s\n",
			       r.file_gbps >= r.gbps ? "ReadFile" : "the mapping");
		}
	}

	// ── quantisation math ──
	printf("\n-- int8-convrot self-test --\n");
	QuantCheck qc;
	{
		auto s = timer.scope("quant.selftest");
		qc = quant_selftest();
	}
	printf("  rotation identity (rot(x).rot(W)^T vs x.W^T) : %.3e  %s\n", qc.rot_identity_rel,
	       qc.rot_identity_rel < 1e-4 ? "PASS" : "FAIL");
	printf("  int8 GEMM rel L2, plain                       : %.4f\n", qc.plain_int8_rel);
	printf("  int8 GEMM rel L2, convrot                     : %.4f  %s\n", qc.convrot_int8_rel,
	       qc.convrot_int8_rel <= qc.plain_int8_rel * 1.25 ? "PASS" : "FAIL");
	printf("  weight int8 round trip rel L2                 : %.4f  %s\n",
	       qc.weight_roundtrip_rel, qc.weight_roundtrip_rel < 0.02 ? "PASS" : "FAIL");

	// ── tokenizer ──
	//
	// Both text encoders share the Qwen2.5 vocabulary, so the tokenizer is the
	// one component every pipeline depends on and the one that can be checked
	// exactly: the gold file is generated by running the reference (HuggingFace
	// `tokenizers`, which is what ComfyUI uses) over a corpus and the C++ side
	// has to reproduce every id. Conformance itself lives in
	// tests/media_tokenizer_test.cpp; this section is the quick smoke check.
	printf("\n-- tokenizer --\n");
	{
		auto s = timer.scope("tokenizer.load");
		try {
			Qwen2Tokenizer tk;
			// The image chain's own tokenizer role (each model carries its
			// vocabulary now; there is no shared tokenizers/qwen25 any more).
			const std::string models_dir = path_join(base, "models");
			const std::string vocab = resolve_media_role(JsonValue::object(),
			                                             "image_tokenizer_vocab", models_dir);
			const std::string merges = resolve_media_role(JsonValue::object(),
			                                              "image_tokenizer_merges", models_dir);
			const std::string cfg = resolve_media_role(JsonValue::object(),
			                                           "image_tokenizer_config", models_dir);
			if (vocab.empty() || !path_exists(vocab))
				throw MediaError("no image vocab.json configured under " + models_dir);
			tk.load(vocab, merges, cfg);
			printf("  vocab %zu, merges %zu, added %zu\n", tk.vocab_size(), tk.merge_count(),
			       tk.added_count());
			printf("  special ids: im_start=%d im_end=%d endoftext=%d\n", tk.id_im_start(),
			       tk.id_im_end(), tk.id_endoftext());
			const char* probe = "a photo of a cat, ultra detailed";
			std::vector<i32> ids = tk.encode(probe);
			bool round = tk.decode(ids) == probe;
			printf("  \"%s\" -> %zu tokens, round-trip %s\n", probe, ids.size(),
			       round ? "ok" : "FAILED");
			std::string prompt = tk.chat_prompt_no_system("hi");
			std::vector<i32> pids = tk.encode(prompt);
			bool tmpl = pids.size() >= 3 && pids[0] == tk.id_im_start() && pids[1] == 872 &&
			            pids[2] == 198;
			printf("  chat template (no system) -> %zu tokens, prefix %s\n", pids.size(),
			       tmpl ? "ok" : "FAILED");
			if (!round || !tmpl) failures++;
		} catch (const std::exception& e) {
			printf("  tokenizer unavailable: %s\n", e.what());
			failures++;
		}
	}

	// ── native CUDA kernels ──
	//
	// The engine's own GEMM: int8 in, int8 out with per-row (activations) and
	// per-output-channel (weights) scales, plus the convrot Hadamard rotation on
	// both sides. This is the only GEMM path the checkpoint format admits
	// (rotated int8 with two scale vectors), and the one every image/video run
	// goes through.
	//
	// The gate is numeric: the GPU result is compared against the CPU reference
	// fed with the *same* quantised operands, so any disagreement is a kernel bug
	// rather than a quantisation difference.
	printf("\n-- native CUDA kernels --\n");
	ComputeContext compute;
	bool have_compute = false;
	if (have_cuda) {
		auto s = timer.scope("env.compute_create");
		try {
			compute.create(cuda, path_join(base, "data/media_shader_cache"));
			have_compute = true;
			printf("  shader compiler: %s\n", compute.compiler_status().describe().c_str());
		} catch (const std::exception& e) {
			printf("  compute context unavailable: %s\n", e.what());
		}
	}

	if (have_compute) {
		// PHI_QUANT_BENCH="M,K[;M,K...]": time the convrot activation quantiser at
		// the shapes a model actually feeds it, because it is now the DiT's
		// largest single cost and a table with only one small shape in it cannot
		// say whether that is the shape or the kernel.
		if (const char* e = getenv("PHI_QUANT_BENCH")) {
			std::string spec(e);
			size_t pos = 0;
			while (pos < spec.size()) {
				size_t q = spec.find(';', pos);
				std::string one = spec.substr(pos, q == std::string::npos ? std::string::npos : q - pos);
				i64 m = 0, k = 0;
				if (sscanf(one.c_str(), "%lld,%lld", (long long*)&m, (long long*)&k) == 2 && m > 0 && k > 0) {
					NativeQuantResult r = native_quant_bench(compute, m, k, 5);
					if (r.ok) {
						printf("  convrot activation        %5lld rows x %5lld  %8.3f ms  %7.2f GB/s",
						       (long long)m, (long long)k, r.ms,
						       // bytes the kernel must move per element: the row is read
						       // twice (rotation + amax, then rotate + quantise), written
						       // once as int8, plus one fp32 scale per row.
						       r.ms > 0 ? (double)(m * k) * 9.0 / (r.ms / 1000.0) / 1e9 : 0.0);
					} else {
						printf("  convrot activation        %5lld x %5lld unavailable: %s",
						       (long long)m, (long long)k, r.error.c_str());
					}
					putchar(10);
				}
				if (q == std::string::npos) break;
				pos = q + 1;
			}
		}
		{
			auto s = timer.scope("kernel.quant_check");
			NativeQuantResult q = native_quant_check(compute, 512, 2560);
			if (q.ok) {
				// A .5 value sitting exactly on a rounding boundary can still
				// land on either side of the two implementations, so the gate is
				// "at most one ULP off, and essentially never". Anything larger
				// (a wrong rotation, a clobbered byte) shows up as ~75% of the
				// row disagreeing, which is what this check is really for.
				const double total = (double)q.rows * (double)q.cols;
				bool pass = q.max_diff <= 1 && (double)q.mismatches <= total / 100000.0 &&
				            q.scale_rel < 1e-6;
				printf("  convrot activation kernel  %5lld rows x %5lld  mismatches %d/%lld"
				       "  max|dq| %d  scale rel %.2e  %s\n",
				       (long long)q.rows, (long long)q.cols, q.mismatches,
				       (long long)(q.rows * q.cols), q.max_diff, q.scale_rel,
				       pass ? "PASS" : "FAIL");
			} else {
				printf("  convrot activation kernel  unavailable: %s\n", q.error.c_str());
			}
		}
		// ── the native grouped-scale decode, against the host reference ────
		//
		// `dispatch_w4a8_expand` replaces a CPU path this engine still has
		// (`w4a8_pack_rows` with PHI_W4A8_CPU=1), and the two are supposed to
		// produce the *same bytes*: the device kernel reproduces the host decoder
		// and the host quantiser exactly. Running a whole generation both ways is
		// the honest check and costs half an hour (the CPU decode is 4.8 s per
		// 368 MB of weights), so this compares the same thing on real tensors from
		// the checkpoint: decode + requantise on both sides, then compare the int8
		// codes and the fp32 scales byte for byte.
		if (const char* e = getenv("PHI_BENCH_W4A8")) {
			auto s = timer.scope("kernel.w4a8_decode");
			try {
				SafeTensors st;
				st.open(e);
				GpuArena wa, keep;
				wa.init(&compute, arena_chunk_bytes_for(vram_budget().limit()));
				keep.init(&compute, keep_chunk_bytes_for(vram_budget().limit()));
				UploadRing ring;
				ring.init(&compute, default_upload_ring_bytes());
				GpuCtx g;
				g.device = &cuda;
				g.ctx = &compute;
				g.wa = &wa;
				g.aa = &keep;
				g.keep = &keep;
				g.ring = &ring;

				i64 checked = 0, bad = 0, worst_codes = 0;
				double worst_scale = 0;
				for (i64 bi : {0, 12, 25, 37, 49}) {
					for (const char* nm : {"attn.qkv_proj", "attn.out_proj", "mlp.fc1",
					                       "mlp.fc2"}) {
						const std::string stem =
						    "blocks." + std::to_string(bi) + "." + nm;
						const std::string name = stem + ".weight";
						const StTensor* t = st.find(name);
						if (!t) continue;
						// The host reference, built the way the slow path builds
						// it: decode the rows, then the engine's own per-row
						// absmax quantiser. (`quantize_weight_into` is per-output-row
						// by construction, so this whole-matrix form and the
						// slab-wise form the loader uses produce the same bytes.)
						const QuantSpec host_spec = requant_spec_for(st, stem);
						const i64 n = t->shape[0], k = t->shape[1];
						std::vector<float> hv((size_t)(n * k));
						std::vector<int8_t> hq((size_t)(n * k));
						std::vector<float> hs((size_t)n);
						st.dequant_rows(*t, 0, n, hv.data());
						quantize_weight_into(hv.data(), n, k, host_spec, hq.data(), hs.data());

						// The device path, through the loader both chains use.
						wa.reset();
						I8Upload up = upload_quant_linear_i8(g, st, stem, wa, nullptr);
						std::vector<int8_t> dq((size_t)(up.n * up.k));
						std::vector<float> ds((size_t)up.n);
						compute.download(up.w.res, up.w.off, dq.data(), dq.size());
						compute.download(up.s.res, up.s.off, ds.data(), ds.size() * 4);
						compute.submit_if_recording();

						i64 misc = 0;
						for (size_t i = 0; i < dq.size(); i++)
							if (dq[i] != hq[i]) misc++;
						double sdiff = 0;
						for (size_t i = 0; i < ds.size(); i++)
							sdiff = std::max(sdiff, (double)std::fabs(ds[i] - hs[i]) /
							                            std::max(1e-30, (double)hs[i]));
						checked++;
						worst_codes = std::max<i64>(worst_codes, misc);
						worst_scale = std::max(worst_scale, sdiff);
						if (misc != 0 || sdiff != 0.0) {
							bad++;
							printf("  w4a8 %-26s %5lldx%5lld: %lld/%lld codes differ, "
							       "scale rel %.2e\n",
							       stem.c_str(), (long long)n, (long long)k, (long long)misc,
							       (long long)dq.size(), sdiff);
						}
					}
				}
				printf("  w4a8 native decode: %lld tensors, %lld differ "
				       "(worst %lld code(s), scale rel %.2e)  %s\n",
				       (long long)checked, (long long)bad, (long long)worst_codes, worst_scale,
				       bad == 0 ? "PASS" : "FAIL");
				if (checked == 0) printf("  w4a8 native decode: no grouped-scale tensor found\n");
				if (bad) failures++;
				ring.destroy();
				wa.destroy();
				keep.destroy();
			} catch (const std::exception& ex) {
				printf("  w4a8 native decode: unavailable: %s\n", ex.what());
			}
		}

		// PHI_BENCH_ATTN="S[,H[,iters]]": the H3 attention at the shape the DiT
		// runs it, on packed fp16 q/k/v of a whole sequence but only one query
		// chunk's worth of output.
		//
		// The instrument exists because the attention is half the sampling step at
		// 540P/10s (S=37500, 56 heads x 128) and the kernel's own tuning knobs
		// (PHI_ATTN_BM / BN / BLOCKS) can only be judged on that shape. Judging
		// them through a full generation costs six minutes a point; here it is a
		// second, and the numbers are the dispatch's own.
		if (const char* e = getenv("PHI_BENCH_ATTN")) {
			i64 S = 37549, H = 56, iters = 5;
			{
				std::string s(e);
				size_t p = s.find(',');
				S = atoll(s.c_str());
				if (p != std::string::npos) {
					H = atoll(s.c_str() + p + 1);
					size_t q = s.find(',', p + 1);
					if (q != std::string::npos) iters = atoll(s.c_str() + q + 1);
				}
			}
			if (S > 0 && H > 0 && iters > 0) {
				const i64 rows = std::min<i64>(S, 2048);
				// The kernel pads its last query block up to the tile, so the output
				// buffer has to be the *padded* height (the DiT's is a whole chunk,
				// which is a multiple of the tile; a bench that sized it at `rows`
				// would write up to 63 rows past it, which is an illegal access and
				// not a measurement).
				const i64 orows = ((rows + 63) / 64) * 64;
				auto s = timer.scope("kernel.h3_attn");
				try {
					const u64 qkv_bytes = (u64)S * (u64)H * 128ull * 2ull;
					GpuBuffer* qb = compute.create_device_buffer(qkv_bytes, true);
					GpuBuffer* kb = compute.create_device_buffer(qkv_bytes, true);
					GpuBuffer* vb = compute.create_device_buffer(qkv_bytes, true);
					GpuBuffer* ob =
					    compute.create_device_buffer((u64)orows * (u64)H * 128ull * 4ull, true);
					H3AttnArgs aa;
					aa.q = GpuAlloc{qb, 0, qkv_bytes};
					aa.k = GpuAlloc{kb, 0, qkv_bytes};
					aa.v = GpuAlloc{vb, 0, qkv_bytes};
					aa.o = GpuAlloc{ob, 0, (u64)orows * (u64)H * 128ull * 4ull};
					aa.sk = S;
					aa.q0 = 0;
					aa.sq = rows;
					aa.heads = H;
					aa.head_dim = 128;
					aa.scale = 1.0f / std::sqrt(128.0f);
					aa.fp16_qkv = true;
					// q/k/v read as fp16 constants: the kernel's address arithmetic
					// is what is under test, not the values.
					compute.begin();
					dispatch_h3_attn(compute, aa);
					compute.submit();
					const double t0 = now_ms();
					for (i64 i = 0; i < iters; i++) {
						compute.begin();
						dispatch_h3_attn(compute, aa);
						compute.submit();
					}
					const double t1 = now_ms();
					const double ms = (t1 - t0) / (double)iters;
					const double flops = 2.0 * 2.0 * (double)rows * (double)S * (double)H * 128.0;
					printf("  h3 attn  S=%lld H=%lld rows=%lld  %8.3f ms  %7.2f TFLOP/s"
					       "  (query tile %u, bn %s)\n",
					       (long long)S, (long long)H, (long long)rows, ms, flops / (ms / 1000.0) / 1e12,
					       attn_query_tile(), getenv("PHI_ATTN_BN") ? getenv("PHI_ATTN_BN") : "64");
					compute.release_buffer(qb);
					compute.release_buffer(kb);
					compute.release_buffer(vb);
					compute.release_buffer(ob);
				} catch (const std::exception& ex) {
					printf("  h3 attn  unavailable: %s\n", ex.what());
				}
			}
		}

		// PHI_BENCH_GEMM="M,N,K[;M,N,K...]" measures the int8 GEMM at the exact
		// shapes a model runs, which is the only way to tell a slow kernel from a
		// slow caller: the fixed shapes above (all square-ish) benchmark at ~49
		// TOPS, and the DiT's own stages measure 30x below that, so the shape has
		// to be the thing under test.
		std::vector<std::array<i64, 3>> shape_list;
		if (const char* e = getenv("PHI_BENCH_GEMM")) {
			std::string s(e);
			size_t p = 0;
			while (p < s.size()) {
				size_t q = s.find(';', p);
				std::string one = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
				i64 v[3] = {0, 0, 0};
				int got = sscanf(one.c_str(), "%lld,%lld,%lld", (long long*)&v[0], (long long*)&v[1],
				                 (long long*)&v[2]);
				if (got == 3 && v[0] > 0 && v[1] > 0 && v[2] > 0)
					shape_list.push_back({v[0], v[1], v[2]});
				if (q == std::string::npos) break;
				p = q + 1;
			}
		}
		const i64 shapes[][3] = {{1024, 1024, 1024}, {2048, 3840, 3840}, {4096, 3840, 3840}};
		std::vector<std::array<i64, 3>> all;
		for (const auto& sh : shapes) all.push_back({sh[0], sh[1], sh[2]});
		for (const auto& sh : shape_list) all.push_back(sh);
		for (const auto& sh : all) {
			auto s = timer.scope("kernel.int8_gemm");
			NativeGemmResult g = native_int8_gemm(compute, sh[0], sh[1], sh[2], 5);
			if (g.ok) {
				if (g.tops > best_int8_tops) best_int8_tops = g.tops;
				printf("  int8 dp4a GEMM  %5lldx%5lldx%5lld  %8.3f ms  %7.2f TOPS"
				       "  rel L2 vs CPU %.2e  %s\n",
				       (long long)g.m, (long long)g.n, (long long)g.k, g.ms, g.tops,
				       g.rel_l2, g.rel_l2 < 1e-5 ? "PASS" : "FAIL");
			} else {
				printf("  int8 dp4a GEMM  %5lldx%5lldx%5lld  unavailable: %s\n",
				       (long long)sh[0], (long long)sh[1], (long long)sh[2],
				       g.error.c_str());
			}
			if (cuda.device_removed()) {
				printf("  !! device removed inside the native kernel; stopping\n");
				break;
			}
		}
	}

		// ── CUDA runtime status ──
	//
	// What the CUDA path is running on: the arch NVRTC compiled for, the compiler,
	// and the kernel cache.
	printf("\n-- CUDA --\n");
	if (have_cuda) {
		printf("  %s\n", cuda.capabilities().c_str());
		printf("  nvrtc: %s\n", shader_compiler_status().describe().c_str());
	} else {
		printf("  not available\n");
	}

	// The backend choice is cached so later runs (and the scheduler) do not have
	// to repeat the probe.
	{
		std::string cache = path_join(base, "data/media_backend_cache.json");
		std::string json = "{\n";
		json += "  \"schema\": 1,\n";
		json += std::string("  \"gpu\": \"") + (have_cuda ? cuda.adapter().name : "") +
		        "\",\n";
		json += "  \"backend\": \"cuda-nvrtc\",\n";
		json += std::string("  \"arch\": \"") + (have_cuda ? cuda.arch() : "") + "\",\n";
		json += "  \"int8_gemm\": \"cuda-dp4a\",\n";
		json += std::string("  \"int8_gemm_tops\": ") + std::to_string(best_int8_tops) + ",\n";
		json += std::string("  \"notes\": \"") +
		        (have_cuda ? "NVRTC-compiled CUDA kernels; int8 dp4a verified against the "
		                     "host reference"
		                   : "no CUDA device");
		json += "\"\n";
		json += "}\n";
		{
			FILE* f = fopen(cache.c_str(), "wb");
			if (f) {
				fwrite(json.data(), 1, json.size(), f);
				fclose(f);
			}
		}
		printf("\n  backend cache -> %s\n", cache.c_str());
	}

	// PHI_BENCH_VAE="<file>[,<W>,<H>,<frames>[,<tile>]]": one video VAE decode,
	// with no DiT and no text encoder in the way - the decode's own wall clock at
	// the canvas the video chain would hand it. Default 960x540 x 243 frames.
	if (const char* e = getenv("PHI_BENCH_VAE")) {
		std::string name = e;
		i64 W = 960, Hh = 540, frames = 243, tile = 0;
		{
			std::vector<std::string> parts;
			size_t p = 0;
			for (;;) {
				size_t q = name.find(',', p);
				parts.push_back(name.substr(p, q == std::string::npos ? std::string::npos : q - p));
				if (q == std::string::npos) break;
				p = q + 1;
			}
			if (!parts.empty()) name = parts[0];
			if (parts.size() > 1 && !parts[1].empty()) W = atoll(parts[1].c_str());
			if (parts.size() > 2 && !parts[2].empty()) Hh = atoll(parts[2].c_str());
			if (parts.size() > 3 && !parts[3].empty()) frames = atoll(parts[3].c_str());
			if (parts.size() > 4 && !parts[4].empty()) tile = atoll(parts[4].c_str());
		}
		printf("\n-- video VAE decode (%lldx%lld, %lld frames) --\n", (long long)W, (long long)Hh,
		       (long long)frames);
		try {
			const std::string models_dir = path_join(base, "models");
			MediaModelSpec spec_v = resolve_media_spec(JsonValue::object(), models_dir);
			std::string path = name.empty() ? spec_v.video_vae : resolve_media_file(models_dir, name);
			if (path.empty()) path = spec_v.video_vae;

			GpuArena vwa, vaa, vkeep;
			UploadRing vring;
			vwa.init(&compute, 64ull << 20);
			vaa.init(&compute, 64ull << 20);
			vkeep.init(&compute, 16ull << 20);
			vring.init(&compute, default_upload_ring_bytes());
			vwa.set_tag("vae.weights");
			vaa.set_tag("vae.acts");
			vkeep.set_tag("vae.keep");
			GpuCtx vg{&cuda, &compute, &vwa, &vaa, &vkeep, &vring};

			VideoPlan vp = plan_video(W, Hh, frames);
			const i64 T = vp.video_t, LH = vp.latent_h, LW = vp.latent_w;
			printf("  file %s\n", path.c_str());
			printf("  canvas %lldx%lld, latent %lldx%lld x %lld, tile %lld\n",
			       (long long)vp.frame.canvas_w, (long long)vp.frame.canvas_h, (long long)LH,
			       (long long)LW, (long long)T, (long long)tile);

			VideoVae vv;
			vv.open(path, &vg);
			std::vector<float> z((size_t)(24 * T * LH * LW));
			std::mt19937 rng(1234);
			std::normal_distribution<float> nd(0.0f, 1.0f);
			for (auto& x : z) x = nd(rng);

			const double t0 = now_ms();
			std::vector<float> out = vv.decode(z, T, LH, LW, tile);
			const double t1 = now_ms();
			const double sec = (t1 - t0) / 1000.0;
			// A decode that collapsed to zeros or NaNs is a weights/arithmetic bug, and
			// it reads as a *fast* run if only the clock is printed - so the range is
			// part of the line. PHI_BENCH_VAE_OUT dumps the raw planes too, which is
			// how two checkpoints (or the two block precisions) are compared.
			double mn = 1e30, mx = -1e30;
			for (float v : out) {
				if (!(v == v)) continue;
				mn = std::min(mn, (double)v);
				mx = std::max(mx, (double)v);
			}
			if (const char* o = getenv("PHI_BENCH_VAE_OUT"))
				write_file(o, std::string((const char*)out.data(), out.size() * 4));
			printf("  output [%.4f, %.4f] over %zu floats %s\n", mn == 1e30 ? 0.0 : mn,
			       mn == 1e30 ? 0.0 : mx, out.size(), mn == 1e30 ? "(all NaN)" : "");
			printf("  out %zu floats, %lld px canvas frames\n", out.size(),
			       (long long)(vp.frames));
			printf("  vvae  %8.1f s  (tile %lld, activation %s)\n", sec,
			       (long long)vv.last_tile(),
			       format_bytes(vv.last_activation_bytes()).c_str());
			vv.release_gpu_memory();
		} catch (const std::exception& ex) {
			printf("  video VAE decode FAILED: %s\n", ex.what());
		}
	}

	// ── summary ──
	// ── the real image chain ──
	// Everything above measures the pieces. This runs the whole thing once with
	// the segmentation the plan asks for (weights / text / per-step sampling /
	// VAE / encode): a per-kernel number cannot tell you whether the tool meets
	// its latency budget, only whether a kernel is fast.
	if (opts.image && have_cuda) {
		printf("\n-- image chain (%lldx%lld) --\n", (long long)opts.image_w,
		       (long long)opts.image_h);
		try {
			GpuArena iwa, iaa, ikeep;
			UploadRing iring;
			iwa.init(&compute, 64ull << 20);
			iaa.init(&compute, 64ull << 20);
			ikeep.init(&compute, 16ull << 20);
			// The image chain shares the engine's ring in the app, so the bench sizes it
			// the same way (default_upload_ring_bytes) instead of with its own number.
			iring.init(&compute, default_upload_ring_bytes());
			iwa.set_tag("image.weights");
			iaa.set_tag("image.acts");
			ikeep.set_tag("image.keep");
			GpuCtx ig{&cuda, &compute, &iwa, &iaa, &ikeep, &iring};

			const std::string models_dir = path_join(base, "models");
			MediaModelSpec spec_i = resolve_media_spec(JsonValue::object(), models_dir);

			// The chain is the graph the tool builds (core/graph): image_generate
			// runs exactly this graph, so the numbers below are the tool's numbers, not
			// a private harness's. The three checkpoints are the settings' selection
			// (any precision - the loaders detect it).
			ImageWorkflowParams wp;
			wp.prompt = opts.image_prompt;
			wp.negative_prompt = opts.image_negative;
			wp.ref_images = opts.image_refs;
			wp.cfg = opts.image_cfg;
			// The reference sizing edge (ComfyUI's `resolution` widget); 0 keeps
			// each reference's own size. `--image-ref-area` is kept as the square
			// root of the old area spelling so an existing invocation still means
			// what it did.
			if (opts.image_ref_area > 0)
				wp.ref_resolution = (i64)std::llround(std::sqrt((double)opts.image_ref_area));
			wp.width = opts.image_w;
			wp.height = opts.image_h;
			// The chain uses a seed verbatim - 0 is a seed like any other - so "pick one"
			// is this draw, and the value the run used is read back off the sampler's
			// LATENT below.
			const u64 image_seed = opts.image_seed ? opts.image_seed : make_media_seed();
			wp.seed = image_seed;
			wp.output = opts.image_out.empty() ? "build/_bench_image.png" : opts.image_out;
			wp.model = spec_i.image_dit;
			wp.clip = spec_i.image_te;
			wp.vae = spec_i.image_vae;
			wp.tokenizer_vocab = spec_i.image_tokenizer_vocab;
			wp.tokenizer_merges = spec_i.image_tokenizer_merges;
			wp.tokenizer_config = spec_i.image_tokenizer_config;
			const Graph graph = build_image_workflow(wp);

			auto cache = std::make_shared<GraphModelCache>();
			GraphContext gc;
			gc.cwd = base;
			gc.agent_dir = base;
			gc.models_dir = models_dir;
			gc.gpu = &ig;
			gc.user = cache;

			double it0 = now_ms();
			// Per-step wall clock. The DiT's own profile (PHI_DIT_TIME) covers the
			// 30-block loop only; the gap between these deltas and that figure is
			// whatever the step spends outside it, which is exactly what a
			// "3 s/step but 1.9 s of blocks" report leaves open. The same hook samples
			// the sampling phase's footprint, which is what the chain is held to
			// (90-100 % of the driver's limit, or the model fits whole).
			double step_t0 = it0;
			u64 sample_proc = 0, sample_ledger = 0;
			std::vector<std::pair<i64, int>> iband;
			gc.on_step = [&](int64_t i) {
				const double t = now_ms();
				const VideoMemoryBudget vb = cuda.query_budget();
				if (vb.process_usage > sample_proc) sample_proc = vb.process_usage;
				const u64 led = vram_budget().local();
				if (led > sample_ledger) sample_ledger = led;
				const u64 lim = vram_budget().limit();
				if (lim) iband.push_back({i, (int)(100ull * std::max(vb.process_usage, led) / lim)});
				printf("    step %lld at %7.2f s (previous step %.2f s)  vram %s (ledger %s)\n",
				       (long long)i, (t - it0) / 1000.0, i ? (t - step_t0) / 1000.0 : 0.0,
				       format_bytes(vb.process_usage).c_str(), format_bytes(led).c_str());
				step_t0 = t;
			};

			VramTracker iv;
			GraphResult gr = run_graph(graph, gc, media_registry());
			if (!gr.ok) throw MediaError(gr.error);

			// Segmented timings come from the run's per-node trace (one line per node,
			// which is finer than the pipeline's stage list was).
			double ms_text = 0, ms_sample = 0, ms_vae = 0;
			for (const NodeTrace& nt : gr.trace) {
				if (nt.type == "MediaQwenImageEditEncode" || nt.type == "MediaH3Ref2VaEncode") ms_text += nt.ms;
				else if (nt.type == "MediaSampler") ms_sample += nt.ms;
				else if (nt.type == "MediaVaeDecode") ms_vae += nt.ms;
			}

			// The run's geometry + seed are on the sampler's LATENT.
			u64 out_seed = image_seed;
			i64 canvas_w = 0, canvas_h = 0, out_w = opts.image_w, out_h = opts.image_h, steps = 0;
			const GraphNode* sam = nullptr;
			for (const GraphNode& n : graph.nodes)
				if (n.type == "MediaSampler") sam = &n;
			if (sam) {
				auto it = gr.produced.find(std::to_string(sam->id) + ":LATENT");
				if (it != gr.produced.end())
					if (auto lat = as_latent(it->second)) {
						out_seed = lat->seed;
						canvas_w = lat->canvas_w;
						canvas_h = lat->canvas_h;
						out_w = lat->req_w;
						out_h = lat->req_h;
					}
			}
			const GraphNode* sch = nullptr;
			for (const GraphNode& n : graph.nodes)
				if (n.type == "MediaScheduler") sch = &n;
			if (sch) {
				auto it = gr.produced.find(std::to_string(sch->id) + ":SIGMAS");
				if (it != gr.produced.end())
					if (auto s = as_sigmas(it->second)) steps = s->steps();
			}

			// The plan's activation estimate survives on the cached DiT (a member,
			// not an allocation); the residency numbers come from the sampler's latent
			// - the sampler hands its window back at the phase boundary, so the DiT is
			// a *released* window by the time the run is over and reading it there
			// would always say 0. The VAE's tile is on the cached VAE.
			u64 act_dit_bytes = 0;
			for (auto& kv : cache->models)
				if (kv.second->arch == "qwen_image" && kv.second->image)
					act_dit_bytes = kv.second->image->act_bytes();
			LatentData lat_out;
			if (sam) {
				auto it = gr.produced.find(std::to_string(sam->id) + ":LATENT");
				if (it != gr.produced.end())
					if (auto l = as_latent(it->second)) lat_out = *l;
			}
			std::shared_ptr<VaeData> vae;
			for (auto& kv : cache->vaes)
				if (kv.second->role == "image") vae = kv.second;
			const u64 budget = vram_budget().limit();
			const VramBudget::Environment env = vram_budget().environment();

			{
				VideoMemoryBudget vb = cuda.query_budget();
				iv.sample(vb.current_usage);
			}
			printf("  canvas %lldx%lld -> %lldx%lld, seed %llu\n", (long long)canvas_w,
			       (long long)canvas_h, (long long)out_w, (long long)out_h,
			       (unsigned long long)out_seed);
			int iband_min = 0, iband_max = 0;
			for (const auto& s : iband) {
				if (s.first < 1 || s.first >= steps) continue;
				if (iband_min == 0 || s.second < iband_min) iband_min = s.second;
				if (s.second > iband_max) iband_max = s.second;
			}
			if (budget > 0) {
				printf("  plan: budget %s, dit_act %s, resident %lld/%lld = %s\n",
				       format_bytes(budget).c_str(), format_bytes(act_dit_bytes).c_str(),
				       (long long)lat_out.resident_layers, (long long)ImageDiT::kTotalBlocks,
				       format_bytes(lat_out.resident_bytes).c_str());
				printf("        effective residency after the loop: %lld/%lld\n",
				       (long long)lat_out.resident_effective, (long long)ImageDiT::kTotalBlocks);
				printf("        device %s (%s free)\n", format_bytes(env.device_total).c_str(),
				       format_bytes(env.device_free).c_str());
				printf("        sampling peak: %s by the driver = %.0f%% of the %s limit"
				       " (ledger %s); band %d-%d%% across the loop\n",
				       format_bytes(sample_proc).c_str(),
				       budget ? 100.0 * (double)sample_proc / (double)budget : 0.0,
				       format_bytes(budget).c_str(), format_bytes(sample_ledger).c_str(), iband_min,
				       iband_max);
			}
			if (vae && vae->image) {
				if (vae->image->last_tile() > 0)
					printf("        vae decode: tiles of %lld px from upsample level %lld (%lld tiles)\n",
					       (long long)vae->image->last_tile(),
					       (long long)vae->image->last_split(),
					       (long long)vae->image->last_tiles());
				else
					printf("        vae decode: one whole-canvas pass\n");
			}
			printf("  text   %8.1f s\n", ms_text / 1000.0);
			printf("  sample %8.1f s  (%.2f s per step, %d steps)\n", ms_sample / 1000.0,
				steps ? ms_sample / 1000.0 / (double)steps : 0.0, (int)steps);
			printf("  vae    %8.1f s\n", ms_vae / 1000.0);
			printf("  total  %8.1f s\n", (ms_text + ms_sample + ms_vae) / 1000.0);
			printf("  %s\n", iv.report().c_str());
			if (!opts.image_out.empty()) printf("  wrote %s\n", opts.image_out.c_str());
			else std::remove(wp.output.c_str());
		} catch (const std::exception& ex) {
			printf("  image chain FAILED: %s\n", ex.what());
			image_ok = false;
		}
	}

	// ── the real video chain (plan §2: the acceptance instrument) ──
	//
	// One full request, segmented: text encoding, the 16 joint steps, the video
	// VAE, the audio VAE + BigVGAN and the mux, plus the residency plan the DiT
	// actually ran with. The image block above is skipped when --video is asked
	// for on its own, because the two chains would otherwise both want the whole
	// VRAM budget.
	if (opts.video && have_cuda) {
		printf("\n-- video chain (%lldx%lld, %lld frames) --\n", (long long)opts.video_w,
		       (long long)opts.video_h, (long long)opts.video_frames);
		try {
			GpuArena vwa, vaa, vkeep;
			UploadRing vring;
			vwa.init(&compute, 64ull << 20);
			vaa.init(&compute, 64ull << 20);
			vkeep.init(&compute, 16ull << 20);
			// The same derivation the tool uses (tools_media.cpp -> sched.cpp): the ring
			// is host-visible staging and its size is also the granularity the weight
			// streamer moves per submit, so it scales with what the card reported
			// instead of being a fixed number. One function, so the two cannot drift
			// apart: a bench running a different ring is not running the tool.
			vring.init(&compute, default_upload_ring_bytes());
			printf("  upload ring %s\n", format_bytes(vring.size()).c_str());
			vwa.set_tag("video.weights");
			vaa.set_tag("video.acts");
			vkeep.set_tag("video.keep");
			GpuCtx vg{&cuda, &compute, &vwa, &vaa, &vkeep, &vring};

			const std::string models_dir = path_join(base, "models");
			MediaModelSpec spec_v = resolve_media_spec(JsonValue::object(), models_dir);

			// The graph the video_generate tool builds, on the tool's own arenas: a
			// number produced by a private harness would not be the acceptance number.
			VideoWorkflowParams wp;
			wp.prompt = opts.video_prompt;
			wp.width = opts.video_w;
			wp.height = opts.video_h;
			wp.frames = opts.video_frames;
			wp.output = opts.video_out;
			// As in the image bench: "pick one" is drawn here, since the chain uses a
			// seed verbatim.
			const u64 video_seed = opts.video_seed ? opts.video_seed : make_media_seed();
			wp.seed = video_seed;
			wp.ref_images = opts.video_refs;
			wp.ref_audios = opts.video_ref_audios;
			wp.ref_videos = opts.video_ref_clips;
			wp.ref_video_audios = opts.video_ref_clip_audios;
			wp.ref_image_size = opts.video_ref_image_size;
			wp.loras = opts.video_loras;
			if (opts.video_steps > 0) wp.steps = opts.video_steps;
			wp.model = spec_v.video_dit;
			wp.clip = spec_v.video_te;
			wp.vae = spec_v.video_vae;
			wp.vae_audio = spec_v.video_avae;
			wp.tokenizer_vocab = spec_v.video_tokenizer_vocab;
			wp.tokenizer_merges = spec_v.video_tokenizer_merges;
			wp.tokenizer_config = spec_v.video_tokenizer_config;
			const Graph graph = build_video_workflow(wp);

			auto cache = std::make_shared<GraphModelCache>();
			GraphContext gc;
			gc.cwd = base;
			gc.agent_dir = base;
			gc.models_dir = models_dir;
			gc.gpu = &vg;
			gc.user = cache;

			double vt0 = now_ms();
			double step_t0 = vt0;
			u64 vsample_proc = 0, vsample_ledger = 0;
			std::vector<std::pair<i64, int>> vband;
			gc.on_step = [&](int64_t i) {
				const double t = now_ms();
				const VideoMemoryBudget vb = cuda.query_budget();
				if (vb.process_usage > vsample_proc) vsample_proc = vb.process_usage;
				const u64 led = vram_budget().local();
				if (led > vsample_ledger) vsample_ledger = led;
				const u64 lim = vram_budget().limit();
				if (lim) vband.push_back({i, (int)(100ull * std::max(vb.process_usage, led) / lim)});
				printf("    step %2lld at %8.2f s (previous step %.2f s)  vram %s (ledger %s)\n",
				       (long long)i, (t - vt0) / 1000.0, i ? (t - step_t0) / 1000.0 : 0.0,
				       format_bytes(vb.process_usage).c_str(), format_bytes(led).c_str());
				step_t0 = t;
			};

			GraphResult gr = run_graph(graph, gc, media_registry());
			if (!gr.ok) throw MediaError(gr.error);

			// Segmented timings from the run's per-node trace. The two VAE decodes
			// are the *same* node type now (one `MediaVaeDecode` drives the video
			// VAE and the audio VAE, dispatched on the VAE's role), so they are
			// told apart by node id: the video chain wires its video decode first
			// and its audio decode second.
			double ms_text = 0, ms_sample = 0, ms_vva = 0, ms_ava = 0, ms_mux = 0;
			std::vector<const GraphNode*> decs;
			for (const GraphNode& n : graph.nodes)
				if (n.type == "MediaVaeDecode") decs.push_back(&n);
			const int vid_dec = decs.size() > 0 ? decs[0]->id : -1;
			const int aud_dec = decs.size() > 1 ? decs[1]->id : -1;
			for (const NodeTrace& nt : gr.trace) {
				if (nt.type == "MediaQwenImageEditEncode" || nt.type == "MediaH3Ref2VaEncode") ms_text += nt.ms;
				else if (nt.type == "MediaSampler") ms_sample += nt.ms;
				else if (nt.id == vid_dec) ms_vva += nt.ms;
				else if (nt.id == aud_dec) ms_ava += nt.ms;
				else if (nt.type == "MediaSaveVideo") ms_mux += nt.ms;
			}

			// Geometry + seed from the sampler's LATENT; the audio length from the
			// audio VAE's PCM.
			u64 out_seed = video_seed;
			VideoPlan plan;
			i64 steps = 0, audio_samples = 0;
			double audio_seconds = 0;
			const GraphNode* sam = nullptr;
			for (const GraphNode& n : graph.nodes)
				if (n.type == "MediaSampler") sam = &n;
			if (sam) {
				auto it = gr.produced.find(std::to_string(sam->id) + ":LATENT");
				if (it != gr.produced.end())
					if (auto lat = as_latent(it->second)) {
						out_seed = lat->seed;
						plan = lat->plan;
					}
			}
			const GraphNode* sch = nullptr;
			for (const GraphNode& n : graph.nodes)
				if (n.type == "MediaScheduler") sch = &n;
			if (sch) {
				auto it = gr.produced.find(std::to_string(sch->id) + ":SIGMAS");
				if (it != gr.produced.end())
					if (auto s = as_sigmas(it->second)) steps = s->steps();
			}
			if (aud_dec >= 0) {
				auto it = gr.produced.find(std::to_string(aud_dec) + ":out");
				if (it != gr.produced.end())
					if (auto a = as_audio(it->second)) {
						audio_samples = a->channels > 0 ? (i64)(a->pcm.size() / (size_t)a->channels) : 0;
						if (a->sample_rate > 0) audio_seconds = (double)audio_samples / (double)a->sample_rate;
					}
			}
			std::string encoders;
			for (const GraphNode& n : graph.nodes)
				if (n.type == "MediaSaveVideo") {
					auto it = gr.produced.find(std::to_string(n.id) + ":encoders");
					if (it != gr.produced.end() && it->second.type == SocketType::String)
						encoders = it->second.s;
				}

			// The residency plan the DiT ran with: `release_weights()` clears it, so it
			// is read from the sampler's latent (captured before the release).
			LatentData vlat_out;
			if (sam) {
				auto it = gr.produced.find(std::to_string(sam->id) + ":LATENT");
				if (it != gr.produced.end())
					if (auto l = as_latent(it->second)) vlat_out = *l;
			}
			H3Residency residency = vlat_out.h3_residency;

			int vband_min = 0, vband_max = 0;
			for (const auto& s : vband) {
				if (s.first < 1 || s.first >= steps) continue;
				if (vband_min == 0 || s.second < vband_min) vband_min = s.second;
				if (s.second > vband_max) vband_max = s.second;
			}
			{
				VideoMemoryBudget vb = cuda.query_budget();
				vram.sample(vb.current_usage);
			}
			printf("  request %lldx%lld x %lld frames -> canvas %lldx%lld, output %lldx%lld\n",
			       (long long)opts.video_w, (long long)opts.video_h,
			       (long long)opts.video_frames, (long long)plan.frame.canvas_w,
			       (long long)plan.frame.canvas_h, (long long)plan.frame.out_w,
			       (long long)plan.frame.out_h);
			printf("  latent %lldx%lld x %lld (video) / %lld (audio), %lld steps, seed %llu\n",
			       (long long)plan.latent_w, (long long)plan.latent_h, (long long)plan.video_t,
			       (long long)plan.audio_t, (long long)steps, (unsigned long long)out_seed);
			printf("  text   %8.1f s\n", ms_text / 1000.0);
			printf("  sample %8.1f s  (%.2f s per step)\n", ms_sample / 1000.0,
			       ms_sample / 1000.0 / (double)std::max<i64>(steps, 1));
			printf("  vvae   %8.1f s\n", ms_vva / 1000.0);
			printf("  avae   %8.1f s  (%lld PCM frames = %.2f s @32kHz)\n", ms_ava / 1000.0,
			       (long long)audio_samples, audio_seconds);
			printf("  mux    %8.1f s  (%s)\n", ms_mux / 1000.0,
			       encoders.empty() ? "no encoder reported" : encoders.c_str());
			printf("  total  %8.1f s\n", (ms_text + ms_sample + ms_vva + ms_ava + ms_mux) / 1000.0);
			printf("  wrote %s\n", wp.output.c_str());
			if (residency.planned) {
				const H3Residency& p = residency;
				printf("  plan_residency: S=%lld budget %s, dit_act %s, vae_act %s, overhead %s "
				       "(pixel reserve %s)\n",
				       (long long)p.s_tokens, format_bytes(p.budget).c_str(),
				       format_bytes(p.act_dit).c_str(), format_bytes(p.act_vae).c_str(),
				       format_bytes(p.overhead).c_str(), format_bytes(p.request_reserve).c_str());
				printf("              available %s, block %s -> resident %lld/%lld%s%s\n",
				       format_bytes(p.available).c_str(), format_bytes(p.layer_bytes).c_str(),
				       (long long)p.res_main_n, (long long)p.n_layers,
				       p.reason.empty() ? "" : " — ", p.reason.c_str());
				const u64 planned = p.resident_bytes + p.act_dit + p.overhead;
				printf("              planned %s = %.0f%% of the budget\n",
				       format_bytes(planned).c_str(),
				       p.budget ? 100.0 * (double)planned / (double)p.budget : 0.0);
				if (vram_budget().limit())
					printf("              sampling: %d-%d%% of the %s limit (peak %s by the driver, "
					       "ledger %s); resident window ended at %lld/%lld\n",
					       vband_min, vband_max, format_bytes(vram_budget().limit()).c_str(),
					       format_bytes(vsample_proc).c_str(),
					       format_bytes(vsample_ledger).c_str(), (long long)p.res_main_n,
					       (long long)p.n_layers);
			}
		} catch (const std::exception& ex) {
			printf("  video chain FAILED: %s\n", ex.what());
			video_ok = false;
		}
	}
	printf("\n-- summary --\n");
	printf("  %s\n", vram.report().c_str());
	printf("  CUDA: %s\n", have_cuda ? cuda.capabilities().c_str() : "not available");
	printf("  int8-convrot self-test: %s\n", qc.pass ? "PASS" : "FAIL");
	printf("%s", timer.report("\nstage timings").c_str());

	if (!opts.out_json.empty()) {
		std::string j = "{\n";
		j += "  \"rotation_identity_rel\": " + std::to_string(qc.rot_identity_rel) + ",\n";
		j += "  \"int8_plain_rel\": " + std::to_string(qc.plain_int8_rel) + ",\n";
		j += "  \"int8_convrot_rel\": " + std::to_string(qc.convrot_int8_rel) + ",\n";
		j += "  \"weight_roundtrip_rel\": " + std::to_string(qc.weight_roundtrip_rel) + ",\n";
		j += std::string("  \"pass\": ") + (qc.pass ? "true" : "false") + "\n";
		j += "}\n";
		write_file(opts.out_json, j);
		printf("wrote %s\n", opts.out_json.c_str());
	}
	// `failures` counts the tokenizer section's checks: a round-trip that does not
	// survive, a chat template whose prefix is not the expected one, or a tokenizer
	// that would not load at all. Any of those is a real failure of this build, not
	// a warning, so it is part of the exit status.
	return (qc.pass && image_ok && video_ok && failures == 0) ? 0 : 1;
}

bool parse_media_bench_args(const std::vector<std::string>& args, BenchOptions* opts,
                            std::string* error) {
	for (size_t i = 0; i < args.size(); i++) {
		const std::string& a = args[i];
		if (a == "--quick") {
			opts->quick = true;
		} else if (a == "--verbose" || a == "-v") {
			opts->verbose = true;
		} else if (a == "--json" && i + 1 < args.size()) {
			opts->out_json = args[++i];
		} else if (a == "--model" && i + 1 < args.size()) {
			opts->model_paths.push_back(args[++i]);
		} else if (starts_with(a, "--model=")) {
			opts->model_paths.push_back(a.substr(8));
		} else if (a == "--image") {
			opts->image = true;
		} else if (a == "--image-size" && i + 1 < args.size()) {
			const std::string& v = args[++i];
			const size_t x = v.find('x');
			if (x == std::string::npos) {
				if (error) *error = "--image-size wants WxH";
				return false;
			}
			opts->image_w = atoll(v.substr(0, x).c_str());
			opts->image_h = atoll(v.substr(x + 1).c_str());
		} else if (a == "--image-prompt" && i + 1 < args.size()) {
			opts->image_prompt = args[++i];
		} else if (a == "--image-negative" && i + 1 < args.size()) {
			opts->image_negative = args[++i];
		} else if (a == "--image-ref" && i + 1 < args.size()) {
			opts->image_refs.push_back(args[++i]);
		} else if (a == "--image-ref-area" && i + 1 < args.size()) {
			opts->image_ref_area = atoll(args[++i].c_str());
		} else if (a == "--image-cfg" && i + 1 < args.size()) {
			opts->image_cfg = (float)atof(args[++i].c_str());
		} else if (a == "--image-out" && i + 1 < args.size()) {
			opts->image_out = args[++i];
		} else if (a == "--image-seed" && i + 1 < args.size()) {
			opts->image_seed = strtoull(args[++i].c_str(), nullptr, 10);
		} else if (a == "--video") {
			opts->video = true;
		} else if (a == "--video-size" && i + 1 < args.size()) {
			const std::string& v = args[++i];
			const size_t x = v.find('x');
			if (x == std::string::npos) {
				if (error) *error = "--video-size wants WxH";
				return false;
			}
			opts->video_w = atoll(v.substr(0, x).c_str());
			opts->video_h = atoll(v.substr(x + 1).c_str());
		} else if (a == "--video-frames" && i + 1 < args.size()) {
			opts->video_frames = atoll(args[++i].c_str());
		} else if (a == "--video-prompt" && i + 1 < args.size()) {
			opts->video_prompt = args[++i];
		} else if (a == "--video-out" && i + 1 < args.size()) {
			opts->video_out = args[++i];
		} else if (a == "--video-seed" && i + 1 < args.size()) {
			opts->video_seed = strtoull(args[++i].c_str(), nullptr, 10);
		} else if (a == "--video-ref" && i + 1 < args.size()) {
			opts->video_refs.push_back(args[++i]);
		} else if (a == "--video-ref-clip" && i + 1 < args.size()) {
			opts->video_ref_clips.push_back(args[++i]);
		} else if (a == "--video-ref-clip-audio" && i + 1 < args.size()) {
			opts->video_ref_clip_audios.push_back(args[++i]);
		} else if (a == "--video-ref-image-size" && i + 1 < args.size()) {
			opts->video_ref_image_size = args[++i];
		} else if (a == "--video-ref-audio" && i + 1 < args.size()) {
			opts->video_ref_audios.push_back(args[++i]);
		} else if (a == "--video-steps" && i + 1 < args.size()) {
			opts->video_steps = atoll(args[++i].c_str());
		} else if (a == "--video-lora" && i + 1 < args.size()) {
			opts->video_loras.push_back(args[++i]);
		} else if (a == "--help" || a == "-h") {
						printf("usage: phi.exe --media-bench [--quick] [--verbose] [--json out.json] [--model path]...\n"
				       "       [--image] [--image-size WxH] [--image-prompt text] [--image-out out.png] [--image-seed N]\n"
				       "       [--image-ref path]... [--image-ref-area px2]\n"
				       "       [--image-negative text] [--image-cfg F]\n"
				       "       [--video] [--video-size WxH] [--video-frames N] [--video-prompt text] [--video-out out.mp4] [--video-seed N]\n"
				       "       [--video-ref path]... [--video-ref-clip path]... [--video-ref-clip-audio path]...\n"
				       "       [--video-ref-audio path]... [--video-ref-image-size match|max] [--video-lora path]...\n");
			return false;
		} else {
			if (error) *error = "unknown --media-bench option: " + a;
			return false;
		}
	}
	return true;
}

}  // namespace phi::media
