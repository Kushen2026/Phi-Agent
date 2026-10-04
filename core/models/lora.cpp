// LoRA loader implementation. See lora.hpp for the format contract.
#include "models/lora.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "models/model_common.hpp"
#include "kernels/parallel_for.hpp"
#include "util/base.hpp"

namespace phi::media {

namespace {

// Strips the display-name prefixes ComfyUI / kohya put in front of the module
// path, so `diffusion_model.transformer_blocks.0.attn.to_q` and
// `transformer_blocks.0.attn.to_q` normalise to the same target.
std::string strip_display_prefix(std::string s) {
	static const char* kPrefixes[] = {
	    "model.diffusion_model.", "diffusion_model.", "model.", "unet.", "transformer.",
	};
	bool changed = true;
	while (changed) {
		changed = false;
		for (const char* p : kPrefixes) {
			const size_t n = strlen(p);
			if (s.size() > n && s.compare(0, n, p) == 0) {
				s.erase(0, n);
				changed = true;
				break;
			}
		}
	}
	// kohya's `lora_unet_...` form: underscores stand for dots.
	if (s.compare(0, 10, "lora_unet_") == 0) {
		s = s.substr(10);
		for (char& c : s) if (c == '_') c = '.';
	} else if (s.compare(0, 8, "lora_te_") == 0) {
		s = s.substr(8);
		for (char& c : s) if (c == '_') c = '.';
	}
	// A trailing ".weight" can survive on keys that carried an alpha-like suffix.
	if (ends_with(s, ".weight")) s.erase(s.size() - 7);
	return s;
}

bool target_matches(const std::string& factor, const std::string& want) {
	if (factor == want) return true;
	// Forgiving suffix match: a LoRA may name the module with a different
	// container prefix than the checkpoint (`transformer_blocks.0...` vs
	// `diffusion_model.transformer_blocks.0...`).
	if (ends_with(want, "." + factor)) return true;
	if (ends_with(factor, "." + want)) return true;
	return false;
}

}  // namespace

std::string lora_normalize_target(const std::string& key, bool* is_a, bool* is_b) {
	if (is_a) *is_a = false;
	if (is_b) *is_b = false;
	// Per-module alpha scalar: "<base>.alpha".
	if (ends_with(key, ".alpha")) {
		return strip_display_prefix(key.substr(0, key.size() - 6));
	}
	static const char* kA[] = {".lora_A", ".lora_down"};
	static const char* kB[] = {".lora_B", ".lora_up"};
	for (const char* s : kA) {
		const size_t p = key.find(s);
		if (p != std::string::npos) {
			if (is_a) *is_a = true;
			return strip_display_prefix(key.substr(0, p));
		}
	}
	for (const char* s : kB) {
		const size_t p = key.find(s);
		if (p != std::string::npos) {
			if (is_b) *is_b = true;
			return strip_display_prefix(key.substr(0, p));
		}
	}
	return "";
}

void LoraSet::add(const std::string& path) {
	if ((int)files_.size() >= kMaxLoras)
		throw MediaError("LoRA: at most " + std::to_string(kMaxLoras) + " files may be chained");

	SafeTensors st;
	st.open(path);

	File file;
	file.path = path;
	file.name = path_basename(path);

	// Default alpha from the file metadata when present (kohya writes
	// `ss_network_alpha`; some ComfyUI exports write `lora_alpha`). A per-module
	// `<base>.alpha` tensor overrides it.
	float file_alpha = 0.0f;
	{
		const auto& meta = st.metadata();
		for (const char* k : {"ss_network_alpha", "lora_alpha", "network_alpha"}) {
			auto it = meta.find(k);
			if (it == meta.end()) continue;
			try {
				file_alpha = (float)std::atof(it->second.c_str());
			} catch (...) {
				file_alpha = 0.0f;
			}
			if (file_alpha > 0) break;
		}
	}

	// Group A/B/alpha keys by target.
	struct Pending {
		const StTensor* a = nullptr;
		const StTensor* b = nullptr;
		float alpha = 0.0f;
		bool has_alpha = false;
	};
	std::map<std::string, Pending> pending;
	std::vector<std::string> order;
	for (const StTensor& t : st.tensors()) {
		bool is_a = false, is_b = false;
		const std::string target = lora_normalize_target(t.name, &is_a, &is_b);
		if (target.empty()) continue;
		auto it = pending.find(target);
		if (it == pending.end()) {
			order.push_back(target);
			it = pending.emplace(target, Pending{}).first;
		}
		Pending& p = it->second;
		if (is_a) {
			p.a = &t;
		} else if (is_b) {
			p.b = &t;
		} else if (ends_with(t.name, ".alpha")) {
			std::vector<float> v = tensor_to_f32(st, t);
			if (!v.empty()) {
				p.alpha = v[0];
				p.has_alpha = true;
			}
		}
	}

	for (const std::string& target : order) {
		const Pending& p = pending[target];
		if (!p.a || !p.b) continue;   // incomplete factor
		const StTensor& ta = *p.a;
		const StTensor& tb = *p.b;
		if (ta.shape.size() != 2 || tb.shape.size() != 2) continue;
		Factor f;
		f.target = target;
		// A: [rank, k], B: [n, rank].
		f.rank = ta.shape[0];
		f.k = ta.shape[1];
		f.n = tb.shape[0];
		if (tb.shape[1] != f.rank || f.rank <= 0 || f.n <= 0 || f.k <= 0) continue;
		const float alpha = p.has_alpha ? p.alpha : (file_alpha > 0 ? file_alpha : (float)f.rank);
		f.scale = alpha / (float)f.rank;
		f.A = tensor_to_f32(st, ta);
		f.B = tensor_to_f32(st, tb);
		if ((i64)f.A.size() != f.rank * f.k || (i64)f.B.size() != f.n * f.rank) continue;
		file.factors.push_back(std::move(f));
	}

	if (file.factors.empty())
		throw MediaError("LoRA: " + file.name + " holds no A/B factor pairs this engine can map");

	files_.push_back(std::move(file));
	paths_.push_back(path);
}

void LoraSet::apply_rows_of(const Factor& f, float* w, i64 row0, i64 rows, i64 k) {
	// W[o][i] += scale * sum_r B[row0 + o][r] * A[r][i]
	//
	// One task per output row: a row is written by exactly one task and the rank
	// accumulation stays in its original order, so this is bit-identical to a
	// serial loop. It matters that it is spread over the cores - a rank-64 delta on
	// the H3 stack's four matrices is ~1.5 TFLOP per full pass, and a streamed block
	// makes that pass once per sampling step.
	kernels::parallel_for(rows, [&](i64 o) {
		float* wrow = w + (size_t)o * (size_t)k;
		const float* brow = &f.B[(size_t)(row0 + o) * (size_t)f.rank];
		for (i64 r = 0; r < f.rank; r++) {
			const float br = brow[r];
			if (br == 0.0f) continue;
			const float s = f.scale * br;
			const float* arow = &f.A[(size_t)r * (size_t)k];
			for (i64 i = 0; i < k; i++) wrow[i] += s * arow[i];
		}
	});
}

int LoraSet::apply(const std::string& target, float* w, i64 n, i64 k) const {
	if (w == nullptr || n <= 0 || k <= 0) return 0;
	int applied = 0;
	for (const File& file : files_) {
		for (const Factor& f : file.factors) {
			if (!target_matches(f.target, target)) continue;
			if (f.n != n || f.k != k) continue;   // shape disagrees: skip, never guess
			apply_rows_of(f, w, 0, n, k);
			applied++;
		}
	}
	return applied;
}

int LoraSet::apply_rows(const std::string& target, float* w, i64 row0, i64 rows, i64 k) const {
	if (w == nullptr || rows <= 0 || k <= 0 || row0 < 0) return 0;
	int applied = 0;
	for (const File& file : files_) {
		for (const Factor& f : file.factors) {
			if (!target_matches(f.target, target)) continue;
			if (f.k != k || row0 + rows > f.n) continue;   // slice outside the delta
			apply_rows_of(f, w, row0, rows, k);
			applied++;
		}
	}
	return applied;
}

bool LoraSet::touches(const std::string& target) const {
	for (const File& file : files_) {
		for (const Factor& f : file.factors) {
			if (target_matches(f.target, target)) return true;
		}
	}
	return false;
}
int LoraSet::deltas_for(const std::string& target, i64 n, i64 k, std::vector<Delta>* out) const {
	if (out == nullptr || n <= 0 || k <= 0) return 0;
	int added = 0;
	for (const File& file : files_) {
		for (const Factor& f : file.factors) {
			if (!target_matches(f.target, target)) continue;
			if (f.n != n || f.k != k) continue;   // shape disagrees: skip, never guess
			Delta d;
			d.n = f.n;
			d.k = f.k;
			d.rank = f.rank;
			d.scale = f.scale;
			d.a = f.A.data();
			d.b = f.B.data();
			out->push_back(d);
			added++;
		}
	}
	return added;
}

std::string LoraSet::describe() const {
	if (files_.empty()) return "";
	std::string s;
	for (size_t i = 0; i < files_.size(); i++) {
		if (i) s += ", ";
		i64 max_rank = 0;
		for (const Factor& f : files_[i].factors) max_rank = std::max(max_rank, f.rank);
		s += files_[i].name + " (" + std::to_string(files_[i].factors.size()) + " factors, r<=" +
		     std::to_string(max_rank) + ")";
	}
	return s;
}

}  // namespace phi::media
