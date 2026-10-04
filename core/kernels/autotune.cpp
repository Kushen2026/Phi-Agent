#include "kernels/autotune.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include "util/base.hpp"
#include "kernels/kernels.hpp"
#include "host/quant.hpp"

namespace phi::media {

// ── config ─────────────────────────────────────────────────────────────────

std::string KernelConfig::key() const {
	char buf[128];
	snprintf(buf, sizeof buf, "%s/bm%d_bn%d_bk%d_tm%d_tn%d_nt%d", kernel.c_str(), bm, bn, bk, tm,
	         tn, nt);
	return buf;
}

bool KernelConfig::valid(std::string* why) const {
	auto fail = [&](const std::string& m) {
		if (why) *why = m;
		return false;
	};
	if (bm <= 0 || bn <= 0 || bk <= 0 || tm <= 0 || tn <= 0 || nt <= 0)
		return fail("non-positive dimension");
	if (nt % 32 != 0 || nt > 1024) return fail("nt must be a multiple of 32, <= 1024");
	if (bm % tm != 0) return fail("bm % tm != 0");
	if (bn % tn != 0) return fail("bn % tn != 0");
	if (nt != (bm / tm) * (bn / tn)) return fail("nt != (bm/tm)*(bn/tn)");
	if (bk % 4 != 0) return fail("bk must be a multiple of 4 (dp4a operand)");
	if ((bm * bk / 4) % nt != 0) return fail("A tile is not a whole number of per-thread loads");
	if ((bn * bk / 4) % nt != 0) return fail("B tile is not a whole number of per-thread loads");
	// __shared__ int sA[bm][bk/4+1] + sB[bn][bk/4+1], 4 bytes each. A thread
	// block gets 32 KB of shared memory by default.
	u64 shmem = (u64)(bm + bn) * (u64)(bk / 4 + 1) * 4;
	if (shmem > 32 * 1024) return fail("groupshared usage exceeds 32 KB");
	return true;
}

// ── HLSL variant ───────────────────────────────────────────────────────────

namespace {

void sub_const(std::string& s, const std::string& name, int value) {
	const std::string pat = "uint " + name + " = ";
	size_t p = s.find(pat);
	if (p == std::string::npos) throw MediaError("autotune: HLSL constant " + name + " not found");
	size_t d = p + pat.size();
	size_t e = d;
	while (e < s.size() && std::isdigit((unsigned char)s[e])) e++;
	s.replace(d, e - d, std::to_string(value));
}

// Minimal "find a key in a JSON-ish blob" helpers. The store writes the file
// itself, so the parser only has to read back its own fixed key order.
bool find_key(const std::string& s, size_t from, const std::string& key, size_t* at, size_t* end) {
	std::string needle = "\"" + key + "\"";
	size_t p = s.find(needle, from);
	if (p == std::string::npos) return false;
	p = s.find(':', p + needle.size());
	if (p == std::string::npos) return false;
	p++;
	while (p < s.size() && std::isspace((unsigned char)s[p])) p++;
	if (at) *at = p;
	if (end) {
		size_t e = p;
		while (e < s.size() && s[e] != ',' && s[e] != '}' && s[e] != '\n') e++;
		*end = e;
	}
	return true;
}

bool json_string(const std::string& s, size_t from, const std::string& key, std::string* out,
                 size_t* end) {
	size_t a = 0, e = 0;
	if (!find_key(s, from, key, &a, &e)) return false;
	if (a >= s.size() || s[a] != '"') return false;
	size_t close = s.find('"', a + 1);
	if (close == std::string::npos) return false;
	*out = s.substr(a + 1, close - a - 1);
	if (end) *end = close + 1;
	return true;
}

bool json_number(const std::string& s, size_t from, const std::string& key, double* out,
                 size_t* end) {
	size_t a = 0, e = 0;
	if (!find_key(s, from, key, &a, &e)) return false;
	*out = atof(s.c_str() + a);
	if (end) *end = e;
	return true;
}

}  // namespace

std::string int8_gemm_hlsl_variant(const KernelConfig& c) {
	std::string hlsl = int8_gemm_hlsl();
	sub_const(hlsl, "BM", c.bm);
	sub_const(hlsl, "BN", c.bn);
	sub_const(hlsl, "BK", c.bk);
	sub_const(hlsl, "TM", c.tm);
	sub_const(hlsl, "TN", c.tn);
	sub_const(hlsl, "NT", c.nt);
	return hlsl;
}

// ── store ──────────────────────────────────────────────────────────────────

KernelConfig AutotuneStore::default_for(const std::string& kernel) {
	KernelConfig c;
	c.kernel = kernel;
	// The M0 baseline: 10.3-10.5 TOPS measured at 2048x3840x3840.
	c.bm = 128;
	c.bn = 64;
	c.bk = 64;
	c.tm = 4;
	c.tn = 8;
	c.nt = 256;
	return c;
}

std::vector<KernelConfig> AutotuneStore::int8_gemm_candidates() {
	struct Shape {
		int bm, bn, bk, tm, tn, nt;
	};
	static const Shape shapes[] = {
	    {128, 64, 64, 4, 8, 256},    // baseline
	    {64, 64, 64, 4, 4, 256},
	    {128, 64, 32, 4, 8, 256},
	    {64, 128, 64, 4, 8, 256},
	    {128, 32, 64, 4, 4, 256},
	    {128, 128, 32, 4, 8, 512},
	    {64, 64, 128, 4, 4, 256},
	    {128, 64, 128, 4, 8, 512},
	};
	std::vector<KernelConfig> out;
	for (const Shape& s : shapes) {
		KernelConfig c;
		c.kernel = "int8_gemm_dp4a";
		c.bm = s.bm;
		c.bn = s.bn;
		c.bk = s.bk;
		c.tm = s.tm;
		c.tn = s.tn;
		c.nt = s.nt;
		std::string why;
		if (c.valid(&why)) out.push_back(c);
	}
	return out;
}

std::string AutotuneStore::default_path(const std::string& base_dir) {
	return path_join(base_dir, "data/autotune.json");
}

bool AutotuneStore::load(const std::string& path) {
	std::optional<std::string> data = read_file(path);
	if (!data) return false;
	const std::string& s = *data;
	items_.clear();
	size_t pos = s.find("\"items\"");
	if (pos == std::string::npos) return true;   // valid but empty
	// Each entry is a flat object; walk them.
	size_t cur = pos;
	std::string kernel;
	while (json_string(s, cur, "kernel", &kernel, &cur)) {
		KernelConfig c;
		c.kernel = kernel;
		double d = 0;
		if (json_number(s, cur, "bm", &d, &cur)) c.bm = (int)d;
		if (json_number(s, cur, "bn", &d, &cur)) c.bn = (int)d;
		if (json_number(s, cur, "bk", &d, &cur)) c.bk = (int)d;
		if (json_number(s, cur, "tm", &d, &cur)) c.tm = (int)d;
		if (json_number(s, cur, "tn", &d, &cur)) c.tn = (int)d;
		if (json_number(s, cur, "nt", &d, &cur)) c.nt = (int)d;
		if (json_number(s, cur, "measured_ms", &d, &cur)) c.measured_ms = d;
		if (json_number(s, cur, "tops", &d, &cur)) c.tops = d;
		std::string why;
		if (c.valid(&why) && c.kernel.size()) items_.push_back(c);
		// Advance past this object's closing brace so the next scan does not
		// re-find the same keys.
		size_t brace = s.find('}', cur);
		if (brace == std::string::npos) break;
		cur = brace + 1;
	}
	return true;
}

bool AutotuneStore::save(const std::string& path) const {
	std::string j = "{\n  \"schema\": 1,\n  \"items\": [\n";
	for (size_t i = 0; i < items_.size(); i++) {
		const KernelConfig& c = items_[i];
		char buf[512];
		snprintf(buf, sizeof buf,
		         "    {\"kernel\": \"%s\", \"bm\": %d, \"bn\": %d, \"bk\": %d, \"tm\": %d, "
		         "\"tn\": %d, \"nt\": %d, \"measured_ms\": %.4f, \"tops\": %.3f}%s\n",
		         c.kernel.c_str(), c.bm, c.bn, c.bk, c.tm, c.tn, c.nt, c.measured_ms, c.tops,
		         i + 1 < items_.size() ? "," : "");
		j += buf;
	}
	j += "  ]\n}\n";
	return write_file_atomic(path, j);
}

KernelConfig AutotuneStore::get(const std::string& kernel) const {
	for (const auto& c : items_) {
		if (c.kernel == kernel) return c;
	}
	return default_for(kernel);
}

void AutotuneStore::put(const KernelConfig& c) {
	for (auto& e : items_) {
		if (e.kernel == c.kernel) {
			e = c;
			return;
		}
	}
	items_.push_back(c);
}

// ── dispatch ───────────────────────────────────────────────────────────────

void dispatch_int8_gemm_v2(ComputeContext& ctx, const Int8GemmArgs& a, const KernelConfig& c) {
	if (a.M <= 0 || a.N <= 0 || a.K <= 0) throw MediaError("int8_gemm_v2: empty problem");
	std::string why;
	if (!c.valid(&why)) throw MediaError("int8_gemm_v2: invalid config (" + why + ")");

	// Variants live under their own PSO-cache key, so the shader cache keeps the
	// winner across runs instead of recompiling every candidate.
	GpuKernel* pso =
	    ctx.pipeline("int8_gemm_dp4a_" + c.key(), int8_gemm_hlsl_variant(c), "int8_gemm_dp4a",
	                 ShaderModel::SM6_4);

	KernelParams p{};
	p.values[0] = (u32)a.M;
	p.values[1] = (u32)a.N;
	p.values[2] = (u32)a.K;
	p.values[3] = 0;
	p.values[4] = (u32)a.a_offset;
	p.values[5] = (u32)a.b_offset;
	p.values[6] = (u32)a.c_offset;
	p.values[7] = (u32)a.sa_offset;
	p.values[8] = (u32)a.sb_offset;
	p.srv[0] = a.a;
	p.srv[1] = a.b;
	p.srv[2] = a.sa;
	p.srv[3] = a.sb;
	p.uav[0] = a.c;

	const u32 gx = (u32)((a.M + c.bm - 1) / c.bm);
	const u32 gy = (u32)((a.N + c.bn - 1) / c.bn);
	ctx.dispatch(pso, p, gx, gy, 1);
}

// ── sweep ──────────────────────────────────────────────────────────────────

AutotuneResult autotune_int8_gemm(ComputeContext& ctx, i64 M, i64 N, i64 K, int iters) {
	AutotuneResult r;
	try {
		std::mt19937 rng(20240913);
		std::uniform_int_distribution<int> qi(-127, 127);
		std::uniform_real_distribution<float> sf(0.002f, 0.05f);
		std::vector<int8_t> a((size_t)M * K), b((size_t)N * K);
		std::vector<float> sa((size_t)M), sb((size_t)N);
		for (auto& v : a) v = (int8_t)qi(rng);
		for (auto& v : b) v = (int8_t)qi(rng);
		for (auto& v : sa) v = sf(rng);
		for (auto& v : sb) v = sf(rng);

		std::vector<float> ref((size_t)M * N);
		gemm_int8(a.data(), sa.data(), b.data(), sb.data(), M, N, K, ref.data());

		Int8GemmArgs args;
		args.a = ctx.create_device_buffer((u64)M * K, true);
		args.b = ctx.create_device_buffer((u64)N * K, true);
		args.sa = ctx.create_device_buffer((u64)M * 4, true);
		args.sb = ctx.create_device_buffer((u64)N * 4, true);
		args.c = ctx.create_device_buffer((u64)M * N * 4, true);
		args.M = M;
		args.N = N;
		args.K = K;
		ctx.upload(args.a, a.data(), (u64)M * K);
		ctx.upload(args.b, b.data(), (u64)N * K);
		ctx.upload(args.sa, sa.data(), (u64)M * 4);
		ctx.upload(args.sb, sb.data(), (u64)N * 4);

		std::vector<KernelConfig> cands = AutotuneStore::int8_gemm_candidates();
		if (cands.empty()) throw MediaError("autotune: no valid candidate");
		std::vector<float> got((size_t)M * N);
		double best_ms = 1e30;
		for (KernelConfig c : cands) {
			c.measured_ms = 0;
			ctx.begin();
			dispatch_int8_gemm_v2(ctx, args, c);   // warm-up (compiles the PSO)
			ctx.submit();

			double t2 = now_ms();
			for (int i = 0; i < iters; i++) {
				ctx.begin();
				dispatch_int8_gemm_v2(ctx, args, c);
				ctx.submit();
			}
			double t3 = now_ms();
			c.measured_ms = (t3 - t2) / std::max(1, iters);
			c.tops = 2.0 * (double)M * (double)N * (double)K / (c.measured_ms / 1000.0) / 1e12;

			ctx.download(args.c, got.data(), (u64)M * N * 4);
			bool exact = true;
			for (size_t i = 0; i < got.size(); i++) {
				if (memcmp(&got[i], &ref[i], 4) != 0) {
					exact = false;
					break;
				}
			}
			if (exact) r.exact_matches++;
			r.tried.push_back(c);
			if (c.measured_ms < best_ms) {
				best_ms = c.measured_ms;
				r.best = c;
			}
		}
		r.ok = true;
	} catch (const std::exception& e) {
		r.error = e.what();
	}
	return r;
}

}  // namespace phi::media
