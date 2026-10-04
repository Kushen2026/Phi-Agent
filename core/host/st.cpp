#include "host/st.hpp"

#include <windows.h>
#include <memoryapi.h>

#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#include "util/base.hpp"
#include "util/json.hpp"

namespace phi::media {

// ── HostPin ──────────────────────────────────────────────────────────────
//
// `g_host_alloc` is null until the CUDA layer registers one, and the lock budget
// is charged under its own mutex: `cuMemAllocHost` is a global resource and two
// caches (the pinned set and the look-ahead slot) allocate from it concurrently.
namespace {

SafeTensors::HostAllocFn g_host_alloc = nullptr;
SafeTensors::HostFreeFn g_host_free = nullptr;
u64 g_lock_cap = 0;
u64 g_lock_used = 0;
std::mutex g_lock_mutex;

}  // namespace

void SafeTensors::set_host_allocator(HostAllocFn alloc, HostFreeFn free, u64 lock_cap) {
	g_host_alloc = alloc;
	g_host_free = free;
	g_lock_cap = lock_cap;
}

HostPin::~HostPin() { reset(); }

HostPin::HostPin(HostPin&& o) noexcept : p_(o.p_), n_(o.n_), locked_(o.locked_) {
	o.p_ = nullptr;
	o.n_ = 0;
	o.locked_ = false;
}

HostPin& HostPin::operator=(HostPin&& o) noexcept {
	if (this != &o) {
		reset();
		p_ = o.p_;
		n_ = o.n_;
		locked_ = o.locked_;
		o.p_ = nullptr;
		o.n_ = 0;
		o.locked_ = false;
	}
	return *this;
}

void HostPin::reset() {
	if (!p_) return;
	if (locked_) {
		std::lock_guard<std::mutex> lk(g_lock_mutex);
		g_lock_used -= n_;
		if (g_host_free) g_host_free(p_);
		else free(p_);
	} else {
		free(p_);
	}
	p_ = nullptr;
	n_ = 0;
	locked_ = false;
}

void HostPin::resize(u64 bytes, u64 headroom) {
	if (bytes == 0) {
		reset();
		return;
	}
	if (n_ >= bytes) return;
	reset();
	if (g_host_alloc) {
		bool room = false;
		{
			std::lock_guard<std::mutex> lk(g_lock_mutex);
			room = (g_lock_cap == 0 || g_lock_used + bytes + headroom <= g_lock_cap);
			if (room) g_lock_used += bytes;
		}
		if (room) {
			void* p = g_host_alloc((size_t)bytes);
			if (p) {
				p_ = p;
				n_ = bytes;
				locked_ = true;
				return;
			}
			std::lock_guard<std::mutex> lk(g_lock_mutex);
			g_lock_used -= bytes;
		}
	}
	// Pageable fallback: the cache still works, it just goes through the ring.
	p_ = malloc((size_t)bytes);
	if (!p_) throw MediaError("HostPin: out of memory for " + std::to_string(bytes) + " bytes");
	n_ = bytes;
	locked_ = false;
}

u64 SafeTensors::host_locked_bytes() const {
	std::lock_guard<std::mutex> lk(g_lock_mutex);
	return g_lock_used;
}

SafeTensors::~SafeTensors() { close(); }

void SafeTensors::close() {
	// The look-ahead worker reads the mapping and the file handle, so it has to be
	// joined before either goes away (and before the slot is dropped).
	prefetch_wait();
	prefetch_clear();
	if (mapping_) {
		UnmapViewOfFile(mapping_);
		mapping_ = nullptr;
	}
	if (seq_handle_) {
		CloseHandle((HANDLE)seq_handle_);
		seq_handle_ = nullptr;
	}
	if (file_handle_) {
		CloseHandle((HANDLE)file_handle_);
		file_handle_ = nullptr;
	}
	tensors_.clear();
	by_name_.clear();
	metadata_.clear();
	drop_cache();
	{
		std::lock_guard<std::mutex> lk(cache_mutex_);
		host_cache_.clear();
	}
	path_.clear();
	file_size_ = 0;
	// The pinned set is only meaningful for the file it was read from.
}

void SafeTensors::open(const std::string& path) {
	close();
	path_ = path;
	std::wstring wpath = utf8_to_wide(path);
	HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
	                       FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
	if (h == INVALID_HANDLE_VALUE) {
		throw MediaError("cannot open safetensors: " + path + " (err " +
		                 std::to_string(GetLastError()) + ")");
	}
	file_handle_ = (void*)h;

	LARGE_INTEGER li{};
	if (!GetFileSizeEx(h, &li)) {
		close();
		throw MediaError("GetFileSizeEx failed: " + path);
	}
	file_size_ = (u64)li.QuadPart;
	if (file_size_ < 8) {
		close();
		throw MediaError("safetensors too small: " + path);
	}

	HANDLE m = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
	if (!m) {
		close();
		throw MediaError("CreateFileMapping failed: " + path + " (err " +
		                 std::to_string(GetLastError()) + ")");
	}
	mapping_ = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
	CloseHandle(m);
	if (!mapping_) {
		close();
		throw MediaError("MapViewOfFile failed for " + path + " (is the file larger than free "
		                 "address space? err " + std::to_string(GetLastError()) + ")");
	}

	parse_header();
}

// Parses the safetensors header.
void SafeTensors::parse_header() {
	// layout: u64 header_len (LE) | header JSON | payload
	const u8* base = (const u8*)mapping_;
	// gguf first: a gguf file starts with an ASCII magic, so its first eight bytes
	// read as a header length in the billions and the range check below would reject
	// it with "header length out of range" - true but useless. Name the container so
	// the chain says "convert it" instead of "corrupt file".
	if (file_size_ >= 4) {
		char magic[4];
		memcpy(magic, base, 4);
		if (memcmp(magic, "GGUF", 4) == 0 || memcmp(magic, "GGML", 4) == 0 ||
		    memcmp(magic, "GGJT", 4) == 0 || memcmp(magic, "GGLA", 4) == 0) {
			throw MediaError("gguf is not supported; the engine reads safetensors (" + path_ +
			                 " is a gguf container - convert the model to safetensors and retry)");
		}
	}
	u64 header_len = 0;
	memcpy(&header_len, base, 8);
	// Subtraction form (file_size_ >= 8 was checked in open()): `header_len + 8`
		// could wrap to 0 on a crafted 0xFFFF...F8 length and let the check pass.
		if (header_len == 0 || header_len > file_size_ - 8) {
		throw MediaError("safetensors header length out of range");
	}
	std::string header((const char*)base + 8, (size_t)header_len);
	std::string err;
	auto parsed = json_parse(header, &err);
	if (!parsed) {
		throw MediaError("safetensors header JSON parse failed: " + err);
	}
	const JsonValue& j = *parsed;
	const u64 data_base = 8 + header_len;
	for (const auto& [name, entry] : j.entries()) {
		if (name == "__metadata__") {
			if (entry.is_object()) {
				for (const auto& [k, v] : entry.entries()) {
					if (v.is_string()) metadata_[k] = v.as_string();
				}
			}
			continue;
		}
		if (!entry.is_object()) continue;
		StTensor t;
		t.name = name;
		t.dtype = dtype_from_name(entry["dtype"].as_string());
		if (t.dtype == DType::Unknown) continue;  // e.g. BOOL bookkeeping tensors
		const JsonValue& shp = entry["shape"];
		t.numel = 1;
		for (const auto& d : shp.items()) {
			i64 v = d.as_int(0);
			t.shape.push_back(v);
			t.numel *= v;
		}
		if (t.shape.empty()) t.numel = 0;
		const JsonValue& off = entry["data_offsets"];
		if (!off.is_array() || off.size() != 2) continue;
		u64 start = (u64)off[(size_t)0].as_int(0);
		u64 stop = (u64)off[(size_t)1].as_int(0);
		if (start > file_size_ - data_base)
			throw MediaError("safetensors tensor '" + name + "' data_offsets start out of range");
		if (stop < start)
			throw MediaError("safetensors tensor '" + name + "' data_offsets reversed");
		if (stop > file_size_ - data_base)
			throw MediaError("safetensors tensor '" + name + "' data_offsets stop out of range");
		t.offset = data_base + start;
		t.nbytes = stop - start;
		by_name_[name] = tensors_.size();
		tensors_.push_back(std::move(t));
	}
	if (tensors_.empty()) throw MediaError("safetensors contains no readable tensors");
	rewrite_packed_shapes();
}

// Packed quantised weights (asym_w4a8_int8, w6a8_int8, nvfp4) store `bits/8` of
// their logical K, and the *architecture*, not the file, is what says how wide a
// layer really is: every loader in the engine reads `shape[1]` as K. So the shape
// is rewritten to the logical one here, once, and the stored width is kept in
// `stored_cols` (with the stored row stride) for the decode path.
void SafeTensors::rewrite_packed_shapes() {
	for (StTensor& t : tensors_) {
		if (t.shape.size() < 2) continue;
		if (t.dtype != DType::I8 && t.dtype != DType::U8 && t.dtype != DType::I4) continue;
		const WeightLayout L = weight_layout(t.name);
		if (!L.packed || L.bits <= 0 || L.bits >= 8) continue;
		const i64 stored = t.shape.back();
		if (stored <= 0 || (stored * 8) % L.bits != 0) continue;
		const i64 rows = t.numel / stored;
		if (rows <= 0 || (u64)rows > t.nbytes) continue;
		t.stored_cols = stored;
		t.row_bytes_stored = t.nbytes / (u64)rows;
		t.shape.back() = stored * 8 / L.bits;
		t.numel = rows * t.shape.back();
	}
}

const StTensor* SafeTensors::find(std::string_view name) const {
	auto it = by_name_.find(std::string(name));
	if (it == by_name_.end()) return nullptr;
	return &tensors_[it->second];
}

// ── precision ───────────────────────────────────────────────────────────────

bool SafeTensors::is_quantized(const StTensor& t) const {
	return t.dtype == DType::I8 || t.dtype == DType::I4;
}

bool SafeTensors::is_raw(const StTensor& t) const {
	return t.dtype == DType::F32 || t.dtype == DType::F16 || t.dtype == DType::BF16;
}

u64 SafeTensors::row_bytes(const StTensor& t) const {
	if (t.row_bytes_stored) return t.row_bytes_stored;
	const i64 cols = t.shape.empty() ? 0 : t.shape.back();
	if (cols <= 0) return 0;
	// rows are dense, at the dtype's own element size. int4 is packed two per byte
	// along the row and is not readable through this path.
	if (t.dtype == DType::I4)
		throw MediaError("row access: '" + t.name + "' is int4-packed; use the int4 loader");
	return (u64)cols * (u64)dtype_size(t.dtype);
}

static std::string weight_stem(const std::string& name);  // defined with the layout helpers below

// Decode a row range of a *packed* weight family into fp32.//
//     value = decode(code) * weight_s_channel[row] * weight_s_rel[row, column/group]
//
// with the family deciding what a code means (a codebook index, a six-bit code,
// an E2M1 float). The scale tensors are sliced per row here, so the decoders in
// quant.cpp stay file-agnostic and every model in the engine gets the new
// formats through the one `dequant_rows` call it already makes.
// ── the signed int4 nibble family (convrot_w4a4) ────────────────────────────
//
// The Qwen3-VL-32B tower's own format: an I8 tensor whose stored row is K/2
// bytes, two signed nibbles per byte along K (low nibble = column 2j), plus one
// fp32 scale per output row. Its consumer (`upload_quant_linear_w4a4`) uploads the
// packed bytes verbatim to the int4 GEMM, but the family is *also* readable as
// numbers - a LoRA that folds into it, or any caller of `materialize` - and until
// this branch existed that read fell through to the plain int8 case and returned
// half a row of codes instead of K values. Same family, one decode.
bool is_convrot_w4a4(const SafeTensors::WeightLayout& L) {
	return L.spec.valid && L.spec.format == "convrot_w4a4";
}

void SafeTensors::dequant_int4_rows(const StTensor& t, i64 row0, i64 rows, float* dst,
                                    const WeightLayout& L) const {
	const i64 stored = t.shape.empty() ? 0 : t.shape.back();
	const i64 k = stored * 2;
	const i64 stride = row_bytes(t);
	const u8* base = (const u8*)mapping_ + t.offset + (u64)row0 * (u64)stride;
	for (i64 r = 0; r < rows; r++) {
		const float s = L.row_scale ? row_scale_value(*L.row_scale, row0 + r) : 1.0f;
		const u8* codes = base + (u64)r * (u64)stride;
		float* out = dst + (size_t)r * (size_t)k;
		for (i64 c = 0; c < stored; c++) {
			const u8 b = codes[c];
			out[2 * c] = (float)unpack_i4_low(b) * s;
			out[2 * c + 1] = (float)unpack_i4_high(b) * s;
		}
	}
}

void SafeTensors::dequant_packed_rows(const StTensor& t, i64 row0, i64 rows, float* dst) const {
	const i64 stored_cols = t.stored_cols > 0 ? t.stored_cols
	                                         : (t.shape.empty() ? 0 : t.shape.back());
	const WeightLayout L = weight_layout(t.name);
	const i64 k = stored_cols * 8 / L.bits;
	const u64 stride = (u64)stored_cols;  // packed payloads are byte-aligned
	const u8* rows_base = (const u8*)mapping_ + t.offset + (u64)row0 * stride;
	const i64 rel_cols = (L.rel_scale && !L.rel_scale->shape.empty()) ? L.rel_scale->shape.back() : 0;
	const DType rel_dtype = L.rel_scale ? L.rel_scale->dtype : DType::F32;
	const void* rel_base = L.rel_scale ? data_of(*L.rel_scale) : nullptr;
	// A one-entry table is a scalar and is read once; a per-row table is indexed
	// inside the loop.
	const bool rel_scalar = L.rel_scale && L.rel_scale->numel == 1;
	if (L.rel_scale && !rel_scalar && L.rel_scale->numel < (row0 + rows) * rel_cols)
		throw MediaError("'" + L.rel_scale->name + "' does not cover rows " +
		                 std::to_string(row0) + ".." + std::to_string(row0 + rows - 1) + " of '" +
		                 t.name + "'");

	// A six-bit pack must *look* like one: the codes of a rotated weight are a
	// bell around their group's scale, so the histogram of the codes carries far
	// less than six bits. A payload whose bits are independent (an encrypted or
	// scrambled file) decodes to uniform codes instead, and this is the only place
	// that difference can be caught - otherwise a wrong payload would be
	// requantised into a wrong model and only show up as a bad generation minutes
	// later. Checked once per tensor.
	if (L.spec.format == "w6a8_int8" || L.spec.format == "w6a8") validate_packed6_payload(t);

	std::vector<float> codebook;
	if (L.codebook) {
		codebook.resize((size_t)L.codebook->numel);
		convert_to_f32(L.codebook->dtype, data_of(*L.codebook), codebook.data(),
		               (size_t)L.codebook->numel);
	}
	const std::string& f = L.spec.format;
	for (i64 r = 0; r < rows; r++) {
		const u8* codes = rows_base + (u64)r * stride;
		float* out = dst + (size_t)r * (size_t)k;
		const float s_channel = L.row_scale ? row_scale_value(*L.row_scale, row0 + r) : 1.0f;
		const void* rel_row = nullptr;
		if (rel_base && !rel_scalar)
			rel_row = (const u8*)rel_base + (u64)(row0 + r) * (u64)rel_cols * dtype_size(rel_dtype);
		else if (rel_base)
			rel_row = rel_base;
		if (f == "asym_w4a8_int8") {
			dequant_codebook_int4_row(codes, k, codebook.data(), s_channel, rel_row, rel_dtype,
			                          L.group, out);
		} else if (f == "w6a8_int8" || f == "w6a8") {
			dequant_packed6_row(codes, k, s_channel, rel_row, rel_dtype, L.group, out);
		} else if (f == "nvfp4" || f == "mxfp4") {
			// `rel_scale` is the per-block fp8 table here, `global_scale` the
			// per-tensor fp32 one.
			const float global = L.global_scale ? row_scale_value(*L.global_scale, row0 + r) : 1.0f;
			dequant_nvfp4_row(codes, k, rel_row, rel_dtype, global, L.group, out);
		} else if (L.bits == 8) {
			// Grouped int8: signed codes with a relative scale per group of columns.
			const i64 g = L.group > 0 ? L.group : k;
			for (i64 c = 0; c < k; c++) {
				const float rel = rel_row ? scale_at(rel_row, rel_dtype, c / g) : 1.0f;
				out[c] = (float)((const int8_t*)codes)[c] * s_channel * rel;
			}
		} else {
			throw MediaError("'" + t.name + "': the packed layout \"" + f +
			                 "\" is not one this engine can decode");
		}
	}
}

// A six-bit payload has to *look* like one.
//
// The codes of a rotated, per-group quantised weight are a bell around zero, so
// their 64-bin histogram carries far fewer than the six bits a uniform one would.
// A payload whose bits are independent - an encrypted, scrambled or wrongly
// written file - decodes to uniform codes instead, and decoding it would feed the
// sampler a random model that only shows up as a bad generation minutes later.
// So the histogram is measured once per tensor and the tensor is refused by name.
//
// The numbers for the shipped `minimax_h3_ref2va_pruned_w6a8.safetensors`: its
// six-bit histogram entropy is 6.00 of a possible 6.00 bits, its bytes hold 7.88
// of a possible 8.0 bits, the byte autocorrelation is <0.005 at every lag and the
// values correlate with the same weights in another precision at |r|<0.002 -
// while a genuine six-bit pack of those same weights measures 3.48 bits of byte
// entropy. That file is not a quantisation of the weights it claims to hold.
void SafeTensors::validate_packed6_payload(const StTensor& t) const {
	static std::mutex mu;
	static std::set<std::string> checked;
	{
		std::lock_guard<std::mutex> lk(mu);
		if (!checked.insert(t.name).second) return;
	}
	const WeightLayout L = weight_layout(t.name);
	const i64 stored_cols = t.stored_cols > 0 ? t.stored_cols : (t.shape.empty() ? 0 : t.shape.back());
	if (stored_cols <= 0 || t.shape.empty()) return;
	const i64 rows_total = t.numel / t.shape.back();
	const i64 rows = std::min<i64>(std::max<i64>(rows_total, 1), 1024);
	const i64 group = L.group > 0 ? L.group : 32;
	const u8* base = (const u8*)mapping_ + t.offset;
	i64 hist[64] = {0};
	i64 seen = 0, groups = 0, extreme = 0;
	const i64 per_row = stored_cols * 8 / 6;  // codes in one row
	for (i64 r = 0; r < rows; r++) {
		const u8* codes = base + (u64)r * (u64)stored_cols;
		i64 in_group = 0;
		bool lo = false, hi = false;
		for (i64 c = 0; c + 4 <= per_row; c += 4) {
			u8 v[4];
			unpack_6bit_triple(codes + (c / 4) * 3, v);
			for (int j = 0; j < 4; j++) {
				hist[v[j]]++;
				seen++;
				lo = lo || v[j] <= 1;
				hi = hi || v[j] >= 62;
				if (++in_group == group) {
					groups++;
					if (lo || hi) extreme++;
					in_group = 0;
					lo = hi = false;
				}
			}
		}
		if (in_group) {
			groups++;
			if (lo || hi) extreme++;
		}
	}
	if (seen == 0 || groups == 0) return;
	double h = 0.0;
	for (int i = 0; i < 64; i++) {
		if (!hist[i]) continue;
		const double p = (double)hist[i] / (double)seen;
		h -= p * std::log2(p);
	}
	const double frac = (double)extreme / (double)groups;
	// Two independent signatures of "not a quantisation", both measured against a
	// genuine six-bit pack of the same weights (`media_packed_quant_test` builds
	// one, and the int8 twin of this very tensor was measured in Python):
	//
	//   * code-histogram entropy:  genuine 5.4-5.8 bits, uniform 6.00;
	//   * group extremes:          genuine 1.00 (a per-group normalisation puts
	//                              each group's maximum at a top code),
	//                              independent uniform codes ~0.64-0.70.
	//
	// Refusing needs *both*, so a file that is merely packed by a producer with an
	// unusual scale policy (high entropy, but extremes present) still loads.
	if (h > 5.9 && frac < 0.9) {
		char buf[160];
		snprintf(buf, sizeof(buf),
		         "%.2f of 6.00 bits and only %.0f%% of its %lld code groups reach an extreme "
		         "code (a genuine pack: ~100%%)",
		         h, frac * 100.0, (long long)groups);
		throw MediaError("'" + t.name +
		                 "': this file declares a six-bit quantisation of its weights, but its "
		                 "payload does not carry one - the codes are uniformly distributed (" +
		                 buf +
		                 "), i.e. its bits are independent and it is most likely scrambled, "
		                 "encrypted or wrongly written. The decoder for the declared layout is "
		                 "in the engine (media_packed_quant_test packs a file the way the format "
		                 "declares and reads it back); re-export this checkpoint, or hand over "
		                 "the packer that wrote it, and it will load.");
	}
}

void SafeTensors::dequant_rows(const StTensor& t, i64 row0, i64 rows, float* dst) const {
	const i64 stored_cols = t.shape.empty() ? 0 : t.shape.back();
	if (stored_cols <= 0 || rows <= 0) return;
	const WeightLayout layout = weight_layout(t.name);

	// fp8 payloads: one value per element, plus an optional per-row (or
	// per-tensor) scale, which is how the fp8 checkpoints are published.
	if (t.dtype == DType::F8_E4M3 || t.dtype == DType::F8_E5M2) {
		const u64 stride = (u64)stored_cols;
		const u8* base = (const u8*)mapping_ + t.offset + (u64)row0 * stride;
		for (i64 r = 0; r < rows; r++) {
			const float s = layout.row_scale ? row_scale_value(*layout.row_scale, row0 + r) : 1.0f;
			dequant_fp8_row(base + (u64)r * stride, stored_cols, t.dtype, s,
			                dst + (size_t)r * (size_t)stored_cols);
		}
		return;
	}

	const bool int_like = (t.dtype == DType::I8 || t.dtype == DType::U8 || t.dtype == DType::I4);
	if (int_like && (layout.packed || layout.rel_scale)) {
		dequant_packed_rows(t, row0, rows, dst);
		return;
	}
	// The signed int4 nibble family is packed but keeps its *stored* row width in
	// `shape` (its consumer reads `shape[1]` as the packed K), so it is not one of
	// the `packed` families above and needs its own decode. Without this it fell
	// through to the plain int8 case and produced K/2 codes.
	if (int_like && is_convrot_w4a4(layout)) {
		dequant_int4_rows(t, row0, rows, dst, layout);
		return;
	}
	if (int_like && layout.spec.valid && !layout.known_family)
		throw MediaError("'" + t.name + "': the quantisation format \"" + layout.spec.format +
		                 "\" is not one this engine can decode");

	const i64 cols = stored_cols;
	const u8* base = (const u8*)mapping_ + t.offset + (u64)row0 * row_bytes(t);
	switch (t.dtype) {
		case DType::F32:
			memcpy(dst, base, (size_t)(rows * cols) * 4);
			break;
		case DType::F16:
			for (i64 i = 0; i < rows * cols; i++) dst[i] = f16_to_f32(((const u16*)base)[(size_t)i]);
			break;
		case DType::BF16:
			for (i64 i = 0; i < rows * cols; i++) dst[i] = bf16_to_f32(((const u16*)base)[(size_t)i]);
			break;
		case DType::I8: {
			// An int8 tensor's *values* are the codes times the per-row scale stored
			// beside it (`<base>.weight_scale`). Reading the codes alone gives numbers
			// that are off by that factor - which is exactly what this helper did
			// before anything consumed it: the probe's first run reported a 0.99
			// relative L2 against the same weights in another precision, because the
			// reference side was being read unscaled.
			const StTensor* sp = find(weight_stem(t.name) + ".weight_scale");
			if (!sp)
				throw MediaError("'" + t.name +
				                 "': an int8 weight without a scale table (or a packed format this "
				                 "engine does not know) cannot be read");
			for (i64 r = 0; r < rows; r++) {
				const float s = row_scale_value(*sp, row0 + r);
				const int8_t* src = (const int8_t*)base + (u64)r * (u64)cols;
				for (i64 c = 0; c < cols; c++)
					dst[(size_t)r * (size_t)cols + (size_t)c] = (float)src[c] * s;
			}
			break;
		}
		case DType::U8:
			for (i64 i = 0; i < rows * cols; i++) dst[i] = (float)base[i];
			break;
		default:
			throw MediaError("row access: unsupported dtype on '" + t.name + "'");
	}
}

std::vector<u8> SafeTensors::materialize(const StTensor& t, DType target) const {
	if (target != DType::F32 && target != DType::F16 && target != DType::BF16)
		throw MediaError("materialize: unsupported target dtype");
	// A quantised weight is decoded through `dequant_rows`, which is the one place
	// that knows every family (int8 tensorwise, asym_w4a8_int8, w6a8_int8, nvfp4,
	// fp8, grouped int8) - including the bit-packed ones, whose stored row is
	// `bits/8` of their logical K. Slabs of rows, so the fp32 form is the only
	// large temporary - which is what the caller asked for by materialising it.
	// Nothing here de-rotates: a `convrot` weight comes back in the rotated basis
	// that every consumer of this call has always received.
	if (is_quantized(t) || t.dtype == DType::F8_E4M3 || t.dtype == DType::F8_E5M2) {
		const i64 k = logical_cols(t);
		const i64 total = t.numel;
		if (k <= 0 || total <= 0) return {};
		const i64 rows = total / k;
		std::vector<u8> out((size_t)(total * (i64)dtype_size(target)));
		const i64 slab = std::max<i64>(1, (16ll << 20) / std::max<i64>(k * 4, 1));
		// `convrot` weights are stored rotated (the int8 GEMMs want that basis, so
		// `dequant_rows` hands it back untouched), but a *float* consumer is asking
		// for the weight itself - so the rotation is undone here, in whole 256-wide
		// groups per row slab, which is exactly what `dequant_weight_f32` does.
		const bool unrot = weight_layout(t.name).spec.convrot;
		const int group = weight_layout(t.name).spec.convrot_groupsize > 0
		                      ? weight_layout(t.name).spec.convrot_groupsize
		                      : 256;
		if (unrot && (k % group) != 0)
			throw MediaError("'" + t.name + "': K=" + std::to_string(k) +
			                 " is not a multiple of the convrot group " + std::to_string(group) +
			                 ", so the rotation cannot be undone");
		const i64 srow = std::max<i64>(slab - slab % group, group);
		std::vector<float> f((size_t)(std::min<i64>(srow, rows) * k));
		std::vector<float> un;
		for (i64 r0 = 0; r0 < rows; r0 += srow) {
			const i64 n = std::min<i64>(srow, rows - r0);
			dequant_rows(t, r0, n, f.data());
			const float* src = f.data();
			if (unrot) {
				un.resize((size_t)(n * k));
				convrot_inverse(f.data(), un.data(), n, k, group);
				src = un.data();
			}
			convert_from_f32(src, target,
			                 out.data() + (size_t)r0 * (size_t)k * dtype_size(target),
			                 (size_t)(n * k));
		}
		return out;
	}
	std::vector<u8> out((size_t)(t.numel * (i64)dtype_size(target)));
	if (out.empty()) return out;
	// convert the stored dtype (int4 handled as its packed bytes are not what a
	// caller wants, so it is unpacked element-wise).
	switch (t.dtype) {
		case DType::F32:
		case DType::F16:
		case DType::BF16: {
			// Fast path: convert straight from the mapping to the target.
			const void* p = data_of(t);
			if (t.dtype == target) {
				memcpy(out.data(), p, out.size());
			} else if (t.dtype == DType::F32) {
				convert_from_f32((const float*)p, target, out.data(), (size_t)t.numel);
			} else {
				std::vector<float> f((size_t)t.numel);
				convert_to_f32(t.dtype, p, f.data(), (size_t)t.numel);
				convert_from_f32(f.data(), target, out.data(), (size_t)t.numel);
			}
			break;
		}
		default: {
			std::vector<float> f((size_t)t.numel);
			convert_to_f32(t.dtype, data_of(t), f.data(), (size_t)t.numel);
			convert_from_f32(f.data(), target, out.data(), (size_t)t.numel);
			break;
		}
	}
	return out;
}

const void* SafeTensors::data_as(const StTensor& t, DType target) const {
	if (is_raw(t) && t.dtype == target) return data_of(t);
	std::lock_guard<std::mutex> lk(cache_mutex_);
	auto it = host_cache_.find(t.name);
	if (it != host_cache_.end()) return it->second.data();
	std::vector<u8> bytes = materialize(t, target);
	auto ins = host_cache_.emplace(t.name, std::move(bytes));
	return ins.first->second.data();
}

const StTensor& SafeTensors::require(std::string_view name) const {
	const StTensor* t = find(name);
	if (!t) throw MediaError("tensor not found: " + std::string(name));
	return *t;
}

QuantSpec SafeTensors::quant_spec_of(std::string_view weight_name) const {
	// "layer.weight" -> "layer.comfy_quant"; the bare name also works for
	// callers that already stripped the suffix.
	std::vector<std::string> candidates;
	std::string name(weight_name);
	if (ends_with(name, ".weight")) {
		candidates.push_back(name.substr(0, name.size() - 7) + ".comfy_quant");
	}
	if (ends_with(name, ".weight_scale")) {
		candidates.push_back(name.substr(0, name.size() - 13) + ".comfy_quant");
	}
	candidates.push_back(name + ".comfy_quant");
	for (const auto& c : candidates) {
		const StTensor* q = find(c);
		if (q) return parse_quant_spec(data_of(*q), (size_t)q->nbytes);
	}
	return QuantSpec{};
}

// ── quantised weight layouts ───────────────────────────────────────────────

// The tensor's own name without the ".weight" suffix (the stem every scale and
// codebook tensor is named after).
static std::string weight_stem(const std::string& name) {
	if (ends_with(name, ".weight")) return name.substr(0, name.size() - 7);
	return name;
}

SafeTensors::WeightLayout SafeTensors::weight_layout(std::string_view weight_name) const {
	WeightLayout L;
	const std::string name(weight_name);
	const std::string base = weight_stem(name);
	L.spec = quant_spec_of(name);
	L.codebook = find(base + ".weight_codebook");
	L.rel_scale = find(base + ".weight_s_rel");
	L.global_scale = find(base + ".weight_scale_2");
	L.row_scale = find(base + ".weight_s_channel");
	if (!L.row_scale) L.row_scale = find(base + ".weight_scale");
	if (L.spec.format == "nvfp4" && !L.rel_scale && L.row_scale) {
		// An nvfp4 weight's scale lives in two places: a per-16-block E4M3 table and
		// one global fp32. Which tensor name holds which is a packer's choice, so the
		// table is identified by its *shape*: it has one entry per block, so its last
		// dim times its row count is not its element count. `weight_scale` is looked
		// at first (the name comfy's own packer uses) and `weight_scale_2` second.
		//
		// What this must NOT do is drop the global scale: `dequant_packed_rows`
		// passes `L.global_scale` through to `dequant_nvfp4_row`, and leaving it null
		// there reads 1.0 instead of the real factor, i.e. every nvfp4 weight an
		// order of magnitude out. The earlier form of this branch moved the global
		// into `row_scale` (where the nvfp4 decode ignores it) and nulled
		// `global_scale`, which is exactly that loss.
		const i64 n = L.row_scale->shape.empty() ? 0 : L.row_scale->shape.back();
		if (L.row_scale->shape.size() > 1 && n > 0 && n != L.row_scale->numel) {
			L.rel_scale = L.row_scale;   // the per-block table
			L.row_scale = nullptr;       // nvfp4 has no per-row scale
			// `global_scale` (weight_scale_2) is already where the decoder wants it.
		}
	}
	L.bits = quant_bits_per_weight(L.spec.format);
	const std::string& f = L.spec.format;
	if (f == "asym_w4a8_int8") {
		L.packed = true;
		L.group = L.spec.group_size > 0 ? L.spec.group_size : 16;
		L.known_family = true;
	} else if (f == "w6a8_int8" || f == "w6a8") {
		L.packed = true;
		L.group = L.spec.group_size > 0 ? L.spec.group_size : 32;
		L.known_family = true;
	} else if (f == "nvfp4" || f == "mxfp4") {
		L.packed = true;
		L.group = L.spec.group_size > 0 ? L.spec.group_size : 16;
		L.known_family = true;
	} else if (f == "int8_tensorwise" || f == "int8") {
		// The original form; a group table beside it makes this the grouped variant
		// of the same decode (value = code * s_channel * s_rel[group]).
		L.group = L.rel_scale ? (L.spec.group_size > 0 ? L.spec.group_size : 32) : 0;
		L.known_family = true;
	} else if (f == "fp8" || f == "float8_e4m3fn" || f == "float8_e4m3" || f == "float8_e5m2") {
		L.group = 0;
		L.known_family = true;
	}
	return L;
}

i64 SafeTensors::logical_cols(const StTensor& t) const {
	if (t.shape.empty()) return 0;
	if (t.stored_cols > 0) return t.shape.back();  // already the unpacked width
	const i64 stored = t.shape.back();
	const WeightLayout L = weight_layout(t.name);
	if (L.packed && L.bits > 0) return stored * 8 / L.bits;
	// The signed int4 nibble family: two values per byte along K, but `shape` keeps
	// the packed width (see `dequant_int4_rows`), so the unpacked K is twice it.
	if (is_convrot_w4a4(L)) return stored * 2;
	return stored;
}

// ── the file's own precision, and the path it implies ────────────────────────
//
// The policy is documented on the enum in st.hpp. Two things about the order of
// the tests below are load-bearing:
//
//  * the dtype answer comes first for the *unambiguous* cases (a float tensor is a
//    float tensor; I4 and fp8 have no other reading), and only an integer tensor
//    is asked what layout it carries;
//  * `is_plain_int8` is the one test that tells the original int8 tensorwise form
//    apart from the packed ones, because both arrive as I8 with a `weight_scale`
//    beside them - the comfy_quant sidecar is what says which. Asking "I8?" first
//    is what made an int8 checkpoint be read as int4 (half a row of codes into the
//    int4 GEMM) in the Qwen3-VL-32B loader.
WeightPrecision weight_precision_of(const SafeTensors& st, const StTensor& t) {
	switch (t.dtype) {
		case DType::F16: return WeightPrecision::F16;
		case DType::BF16: return WeightPrecision::Bf16;
		case DType::F32: return WeightPrecision::F32;
		case DType::F8_E4M3:
		case DType::F8_E5M2: return WeightPrecision::Fp8;
		// An I4 *dtype* is the signed-int4 family by construction: the reader only
		// accepts it when the header says so.
		case DType::I4: return WeightPrecision::W4A4;
		case DType::I8:
		case DType::U8: break;
		default: return WeightPrecision::Other;
	}
	const SafeTensors::WeightLayout L = st.weight_layout(t.name);
	const std::string& f = L.spec.format;
	if (st.is_plain_int8(t)) return WeightPrecision::Int8Tensorwise;
	if (f == "nvfp4" || f == "mxfp4") return WeightPrecision::Nvfp4;
	// The signed-int4 nibble family. It keeps its *packed* row width in `shape`
	// (see `dequant_int4_rows`), so `L.packed` is false for it - the format name is
	// the only thing that identifies it, and `is_convrot_w4a4` is that test.
	if (is_convrot_w4a4(L)) return WeightPrecision::W4A4;
	if (L.bits == 6) return WeightPrecision::W6A8;
	if (L.bits == 4) return WeightPrecision::W4A8;
	// Everything else that reached here is an integer tensor with a per-row scale and
	// no relative scale: the tensorwise int8 form, whatever the sidecar chose to call
	// itself (a missing/unparsable sidecar lands here too, which is the intent - the
	// bytes are int8 tensorwise).
	return WeightPrecision::Int8Tensorwise;
}

bool weight_is_dense_float(const SafeTensors& st, const StTensor& t) {
	switch (weight_precision_of(st, t)) {
		case WeightPrecision::F16:
		case WeightPrecision::Bf16:
		case WeightPrecision::F32:
		case WeightPrecision::Fp8:
		case WeightPrecision::Nvfp4: return true;
		default: return false;
	}
}

const char* weight_precision_name(WeightPrecision p) {
	switch (p) {
		case WeightPrecision::Int8Tensorwise: return "int8";
		case WeightPrecision::W4A8: return "w4a8";
		case WeightPrecision::W6A8: return "w6a8";
		case WeightPrecision::W4A4: return "w4a4";
		case WeightPrecision::Nvfp4: return "nvfp4";
		case WeightPrecision::F16: return "f16";
		case WeightPrecision::Bf16: return "bf16";
		case WeightPrecision::F32: return "f32";
		case WeightPrecision::Fp8: return "fp8";
		default: return "unknown";
	}
}

bool SafeTensors::is_plain_int8(const StTensor& t) const {
	if (t.dtype != DType::I8) return false;
	const WeightLayout L = weight_layout(t.name);
	if (L.packed || L.rel_scale) return false;
	if (L.spec.valid && !L.spec.format.empty() && L.spec.format != "int8_tensorwise") return false;
	return find(weight_stem(t.name) + ".weight_scale") != nullptr;
}

float SafeTensors::row_scale_value(const StTensor& scale, i64 row) const {
	if (scale.numel == 0) return 1.0f;
	if (scale.numel == 1) return scale_at(data_of(scale), scale.dtype, 0);
	if (row < 0 || row >= scale.numel)
		throw MediaError("'" + scale.name + "' has " + std::to_string(scale.numel) +
		                 " scales but row " + std::to_string(row) + " was asked for");
	return scale_at(data_of(scale), scale.dtype, row);
}

void SafeTensors::read(u64 offset, size_t len, void* dst) const {
	// Subtraction form: `offset + len` can wrap on a caller-supplied 64-bit offset.
	if (offset > file_size_ || len > file_size_ - offset)
		throw MediaError("SafeTensors::read out of range");
	memcpy(dst, (const u8*)mapping_ + offset, len);
}

std::mutex& SafeTensors::io_mutex() {
	static std::mutex m;
	return m;
}

void SafeTensors::read_file(u64 offset, size_t len, void* dst) const {
	if (len == 0) return;
	// Subtraction form: `offset + len` can wrap on a caller-supplied 64-bit
	// offset, silently defeating the check.
	if (offset > file_size_ || len > file_size_ - offset)
		throw MediaError("SafeTensors::read_file out of range");
	// The pinned set first: a hit is a memcpy out of system memory, which is what
	// turns a 15 GB-per-step disk read into a 15 GB-per-step RAM read for whatever
	// share of it was pinned (see the header).
	if (cache_serve(offset, len, dst)) return;
	// Then the look-ahead slot: the block the streaming loop read while the GPU
	// was computing the previous one. Same idea, but for the *next* 368 MB that
	// no fixed pinning could cover.
	if (prefetch_serve(offset, len, dst)) return;
	read_file_direct(offset, len, dst);
}

void SafeTensors::read_file_direct(u64 offset, size_t len, void* dst) const {
	// The seq handle is created under the I/O lock: two concurrent first reads
	// would otherwise both see null and the loser's HANDLE would leak. The file
	// position and the read itself are also under the lock (they share one handle).
	const double t0 = now_ms();
	std::lock_guard<std::mutex> lock(io_mutex());
	if (!seq_handle_) {
		// A second handle, opened for sequential access: FILE_FLAG_SEQUENTIAL_SCAN
		// turns on the OS read-ahead (which the mapping's FILE_FLAG_RANDOM_ACCESS
		// deliberately disables), and not touching the mapping means the read never
		// faults pages into the working set only to be dropped again.
		HANDLE h = CreateFileW(utf8_to_wide(path_).c_str(), GENERIC_READ,
		                       FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
		                       OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
		                       nullptr);
		if (h == INVALID_HANDLE_VALUE)
			throw MediaError("SafeTensors::read_file: cannot reopen " + path_ + " (err " +
		                 std::to_string(GetLastError()) + ")");
		seq_handle_ = (void*)h;
	}
	LARGE_INTEGER li{};
	li.QuadPart = (LONGLONG)offset;
	if (!SetFilePointerEx((HANDLE)seq_handle_, li, nullptr, FILE_BEGIN))
		throw MediaError("SafeTensors::read_file: seek failed (err " +
		                 std::to_string(GetLastError()) + ")");
	u8* p = (u8*)dst;
	u64 done = 0;
	while (done < len) {
		DWORD got = 0;
		const DWORD want = (DWORD)std::min<u64>(len - done, 1ull << 30);
		if (!ReadFile((HANDLE)seq_handle_, p + done, want, &got, nullptr)) {
			// A failure here would leave a half-filled weight buffer behind, which
			// computes garbage silently, so it is a hard error rather than a retry.
			throw MediaError("SafeTensors::read_file: ReadFile failed (err " +
			                 std::to_string(GetLastError()) + ")");
		}
		if (got == 0)
			throw MediaError("SafeTensors::read_file: short read at " +
			                 std::to_string(offset + done));
		done += got;
	}
	file_read_bytes_ += done;
	file_read_ms_ += now_ms() - t0;
}

// ── host-side pinned range cache ──────────────────────────────────────────

bool SafeTensors::cache_serve(u64 offset, size_t len, void* dst) const {
	std::lock_guard<std::mutex> lk(cache_mutex_);
	if (cache_.empty()) return false;
	// The set is keyed by the range's own offset, so the only candidate is the
	// greatest key not past `offset`.
	auto it = cache_.upper_bound(offset);
	if (it == cache_.begin()) return false;
	--it;
	const u64 rel = offset - it->first;
	if (rel + len > it->second.size()) return false;
	const double t0 = now_ms();
	memcpy(dst, (const u8*)it->second.data() + rel, len);
	cache_ms_ += now_ms() - t0;
	cache_served_bytes_ += len;
	return true;
}

// ── the look-ahead slot ───────────────────────────────────────────────────

void SafeTensors::prefetch_reset_locked() {
	pf_ranges_.clear();
	pf_total_ = 0;
	pf_state_ = 0;
}

void SafeTensors::prefetch_clear() {
	prefetch_wait();
	std::lock_guard<std::mutex> lk(pf_mutex_);
	prefetch_reset_locked();
	pf_buf_.reset();
}

const void* SafeTensors::pinned_dma_at(u64 offset, size_t len) const {
	if (len == 0) return nullptr;
	// The look-ahead slot first: it holds the block the streamer is about to ask
	// for, and it is the one buffer that is worth keeping locked even when the
	// pinned set is not.
	{
		std::lock_guard<std::mutex> lk(pf_mutex_);
		if (pf_state_ == 2 && pf_buf_.locked()) {
			for (const auto& r : pf_ranges_) {
				if (offset < r.file_off) continue;
				const u64 rel = offset - r.file_off;
				if (rel + len > r.len) continue;
				if (r.dest + rel + len > pf_buf_.size()) continue;
				return (const u8*)pf_buf_.data() + r.dest + rel;
			}
		}
	}
	{
		std::lock_guard<std::mutex> lk(cache_mutex_);
		if (cache_.empty()) return nullptr;
		auto it = cache_.upper_bound(offset);
		if (it == cache_.begin()) return nullptr;
		--it;
		if (!it->second.locked()) return nullptr;
		const u64 rel = offset - it->first;
		if (rel + len > it->second.size()) return nullptr;
		return (const u8*)it->second.data() + rel;
	}
}

void SafeTensors::prefetch_wait() const {
	std::lock_guard<std::mutex> lk(pf_join_mutex_);
	if (pf_joinable_) {
		if (pf_thread_.joinable()) pf_thread_.join();
		pf_joinable_ = false;
	}
}

bool SafeTensors::prefetch_ready() const {
	std::lock_guard<std::mutex> lk(pf_mutex_);
	return pf_state_ == 2;
}

void SafeTensors::prefetch_worker(std::vector<PfRange> ranges, u64 total) {
	// Each range is read with the same seek-free ReadFile path the synchronous
	// loader uses, so a prefetched block costs exactly what it would have cost in
	// the loop - it just costs it while the GPU is busy. A failure is silent: the
	// slot is left empty and the loop falls back to reading the block itself.
	u8* base = nullptr;
	try {
		std::lock_guard<std::mutex> lk(pf_mutex_);
		if (pf_buf_.size() < (size_t)total) pf_buf_.resize((size_t)total);
		base = (u8*)pf_buf_.data();
	} catch (...) {
		std::lock_guard<std::mutex> lk(pf_mutex_);
		prefetch_reset_locked();
		return;
	}
	for (const auto& r : ranges) {
		// A range the pinned set already holds is copied out of it rather than
		// re-read: the two caches describe the same file, and a block that is
		// half-pinned would otherwise pay for its pinned half twice.
		if (cache_serve(r.file_off, (size_t)r.len, base + r.dest)) continue;
		try {
			read_file_direct(r.file_off, (size_t)r.len, base + r.dest);
		} catch (...) {
			std::lock_guard<std::mutex> lk(pf_mutex_);
			prefetch_reset_locked();
			return;
		}
	}
	std::lock_guard<std::mutex> lk(pf_mutex_);
	pf_state_ = 2;
}

void SafeTensors::prefetch_ranges(const std::vector<std::pair<u64, u64>>& ranges) {
	u64 total = 0;
	std::vector<PfRange> laid;
	laid.reserve(ranges.size());
	for (const auto& r : ranges) {
		if (r.second == 0) continue;
		if (r.first > file_size_ || r.second > file_size_ - r.first) continue;
		PfRange pr;
		pr.file_off = r.first;
		pr.dest = total;
		pr.len = r.second;
		laid.push_back(pr);
		total += r.second;
	}
	std::lock_guard<std::mutex> lk(pf_join_mutex_);
	if (pf_joinable_) {
		if (pf_thread_.joinable()) pf_thread_.join();
		pf_joinable_ = false;
	}
	{
		std::lock_guard<std::mutex> pl(pf_mutex_);
		prefetch_reset_locked();
		pf_ranges_ = laid;
		pf_total_ = total;
		pf_state_ = laid.empty() ? 0 : 1;
	}
	if (laid.empty()) return;
	pf_thread_ = std::thread([this, laid, total]() { prefetch_worker(laid, total); });
	pf_joinable_ = true;
}

bool SafeTensors::prefetch_serve(u64 offset, size_t len, void* dst) const {
	std::lock_guard<std::mutex> lk(pf_mutex_);
	if (pf_state_ != 2) return false;
	for (const auto& r : pf_ranges_) {
		if (offset < r.file_off) continue;
		const u64 rel = offset - r.file_off;
		if (rel + len > r.len) continue;
		if (r.dest + rel + len > pf_buf_.size()) continue;
		memcpy(dst, (const u8*)pf_buf_.data() + r.dest + rel, len);
		return true;
	}
	return false;
}

void SafeTensors::pin_range(u64 offset, u64 len) {
	if (len == 0) return;
	if (offset > file_size_ || len > file_size_ - offset)
		throw MediaError("SafeTensors::pin_range out of range");
	{
		std::lock_guard<std::mutex> lk(cache_mutex_);
		if (cache_.count(offset)) return;   // already pinned
	}
	// Read through the normal (uncached) path first, then publish: taking the
	// cache lock around a 300 MB disk read would serialise every other reader
	// behind it for a second, and nothing needs to see the entry before it is
	// complete.
	// A `HostPin`: page-locked while the lock budget allows (so the streamer can
	// DMA straight out of it - see `pinned_dma_at`), plain malloc after that.
	// The headroom keeps the look-ahead slot lockable: this set is filled first
	// and would otherwise spend the whole budget, and the slot is the better
	// place for the last few hundred megabytes because it holds the block the
	// streamer has *not* asked for yet.
	HostPin buf;
	buf.resize(len, 768ull << 20);
	read_file(offset, (size_t)len, buf.data());
	std::lock_guard<std::mutex> lk(cache_mutex_);
	if (cache_.count(offset)) return;
	pinned_bytes_ += len;
	cache_.emplace(offset, std::move(buf));
}

u64 SafeTensors::pinned_bytes() const {
	std::lock_guard<std::mutex> lk(cache_mutex_);
	return pinned_bytes_;
}

void SafeTensors::drop_cache() {
	std::lock_guard<std::mutex> lk(cache_mutex_);
	cache_.clear();
	pinned_bytes_ = 0;
}

u64 SafeTensors::host_free_bytes() {
	MEMORYSTATUSEX ms{};
	ms.dwLength = sizeof(ms);
	if (!GlobalMemoryStatusEx(&ms)) return 0;
	return (u64)ms.ullAvailPhys;
}

// Physical memory, which unlike `host_free_bytes` does not move while a run reads a
// checkpoint through its mapping. The weight caches want a number that is stable
// across the run (see `host_cache_budget_bytes` in model_common.cpp).
u64 SafeTensors::host_total_bytes() {
	MEMORYSTATUSEX ms{};
	ms.dwLength = sizeof(ms);
	if (!GlobalMemoryStatusEx(&ms)) return 0;
	return (u64)ms.ullTotalPhys;
}

void SafeTensors::prefetch(u64 offset, size_t len) const {
	if (!mapping_ || len == 0) return;
	if (offset > file_size_ || len > file_size_ - offset)
		throw MediaError("SafeTensors::prefetch out of range");
	if (len > file_size_ - offset)
		len = (size_t)(file_size_ - offset);
	// PrefetchVirtualMemory is the cheap way to fault a whole range in up to
	// 64 entries at once; a failure here is a hint loss, never fatal.
	constexpr size_t kMaxEntries = 64;
	WIN32_MEMORY_RANGE_ENTRY entries[kMaxEntries];
	size_t count = 0;
	const u64 chunk = 16ull * 1024 * 1024;  // one entry per 16 MB keeps the list short
	for (u64 pos = offset; pos < offset + len && count < kMaxEntries; pos += chunk) {
		entries[count].VirtualAddress = (PVOID)((const u8*)mapping_ + pos);
		entries[count].NumberOfBytes = (SIZE_T)std::min<u64>(chunk, offset + len - pos);
		count++;
	}
	if (count) PrefetchVirtualMemory(GetCurrentProcess(), count, entries, 0);
}

void SafeTensors::release_pages(u64 offset, size_t len) const {
	if (!mapping_ || len == 0) return;
	if (offset > file_size_ || len > file_size_ - offset)
		throw MediaError("SafeTensors::release_pages out of range");
	if (len > file_size_ - offset)
		len = (size_t)(file_size_ - offset);
	// Align to 64 KiB (the smallest granularity the OS will honour) and discard.
	SYSTEM_INFO si{};
	GetSystemInfo(&si);
	const u64 gran = si.dwAllocationGranularity ? si.dwAllocationGranularity : 65536;
	u64 start = offset - (offset % gran);
	u64 end = offset + len;
	if (end % gran) end += gran - (end % gran);
	DiscardVirtualMemory((PVOID)((u8*)mapping_ + start), (SIZE_T)std::min<u64>(end - start, len));
}

std::string SafeTensors::describe() const {
	std::string s = path_basename(path_) + " (" + format_bytes(file_size_) + ", " +
	                std::to_string(tensors_.size()) + " tensors)";
	return s;
}

}  // namespace phi::media
