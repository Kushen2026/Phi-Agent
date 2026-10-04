#include "models/model_common.hpp"

#include <algorithm>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "host/quant.hpp"
#include "host/st.hpp"
#include "models/lora.hpp"
#include <cstdio>
#include "kernels/parallel_for.hpp"
#include "kernels/gpu_ops.hpp"
#include "kernels/w4a8_expand.hpp"

namespace phi::media {

std::vector<u16> to_f16(const float* src, size_t count) {
	std::vector<u16> out(count);
	for (size_t i = 0; i < count; i++) out[i] = f32_to_f16(src[i]);
	return out;
}

std::vector<float> tensor_to_f32(const SafeTensors& st, const StTensor& t) {
	// A quantised tensor goes through the reader: it knows the family (a codebook, a
	// six-bit pack, nvfp4, fp8, a per-row / per-group scale table) and hands back the
	// weight itself, with any convrot undone. Reading the raw codes here would return
	// unscaled integers, which is what this helper used to do for int8 - and the
	// callers are the norm / bias / scale / embedding paths, i.e. the ones that are
	// supposed to be plain floats.
	if (st.is_quantized(t) || t.dtype == DType::F8_E4M3 || t.dtype == DType::F8_E5M2) {
		std::vector<u8> bytes = st.materialize(t, DType::F32);
		std::vector<float> out(bytes.size() / 4);
		if (!out.empty()) memcpy(out.data(), bytes.data(), out.size() * 4);
		return out;
	}
	std::vector<float> out((size_t)t.numel);
	const void* p = st.data_of(t);
	switch (t.dtype) {
		case DType::F32:
			memcpy(out.data(), p, (size_t)t.numel * 4);
			break;
		case DType::F16:
			for (i64 i = 0; i < t.numel; i++) out[(size_t)i] = f16_to_f32(((const u16*)p)[(size_t)i]);
			break;
		case DType::BF16:
			for (i64 i = 0; i < t.numel; i++) out[(size_t)i] = bf16_to_f32(((const u16*)p)[(size_t)i]);
			break;
		default:
			throw MediaError("tensor_to_f32: unsupported dtype for " + t.name);
	}
	return out;
}

void GpuCtx::upload_into(const GpuAlloc& a, const void* src, u64 bytes) {
	// Big tables (the 741 MB embedding matrix) do not fit the staging ring, so
	// they are staged in chunks. `upload_into` *records* into whatever command
	// list is open and only closes/executes one when the ring has to wrap - if it
	// called begin() unconditionally it would Reset() the caller's list and throw
	// away every dispatch already recorded on it.
	//
	// Contract: the list is left OPEN. Callers that need the copies complete
	// (e.g. before reusing the weights arena) submit themselves.
	const u64 kChunk = 24ull << 20;
	u64 done = 0;
	while (done < bytes) {
		u64 n = std::min(kChunk, bytes - done);
		if (ring->free() < n) {
			ctx->submit();       // executes everything recorded so far
			ring->rewind();
		}
		if (!ctx->recording()) ctx->begin();
		upload_range(*ctx, *ring, a.res, a.off + done, (const u8*)src + done, n);
		done += n;
	}
}

// The same upload, sourced from the *file* rather than from the mapping.
//
// `upload_into` reads whatever pointer it is given, and for a checkpoint tensor
// that pointer is inside the file's `MapViewOfFile` view - so every 4 KB page is
// a page fault, which measures ~0.6-0.9 GB/s cold. `SafeTensors::read_file`
// goes through `ReadFile` instead (the OS read-ahead the mapping disables) and
// is served out of the pinned host cache / the look-ahead slot when either
// holds the range, which is worth 10 GB/s on this box. The bytes land in the
// staging ring and are DMA'd from there, so no intermediate copy is added.
//
// This is the path the H3 DiT's block streamer has always used (see
// `upload_chunked_file` in av_dit.cpp, which it now shares); the video
// VAE's decoder blocks used the mapping and paid for it - 299 ms per 256 px
// tile re-reading the 11 blocks that do not fit the VRAM cache.
u64 GpuCtx::upload_file_into(const GpuAlloc& a, const SafeTensors& st, const StTensor& t) {
	const u64 bytes = t.nbytes;
	if (bytes == 0) return 0;
	// The zero-copy case first: when either host cache holds this tensor in
	// *page-locked* memory, the driver can read the source directly and the copy
	// into the staging ring - which is what makes this function the DiT's second
	// largest cost after the attention, at 1.6 s of a 40 s step - disappears.
	// The DMA is issued blocking on purpose: the loop synchronises per block
	// anyway, and the file read in front of it is what overlaps the GPU.
	if (const void* locked = st.pinned_dma_at(t.offset, (size_t)bytes)) {
		if (!ctx->recording()) ctx->begin();
		ctx->upload(a.res, a.off, locked, bytes);
		return bytes;
	}
	// The ring never wraps a single copy, so a tensor larger than the ring is a
	// hard error by design (the same contract `upload_into` has).
	const u64 max_chunk = std::max<u64>((ring->size() / 2) & ~255ull, 1ull << 20);
	const bool own_bracket = !ctx->recording();
	if (own_bracket) ctx->begin();
	u64 done = 0;
	while (done < bytes) {
		const u64 n = std::min(max_chunk, bytes - done);
		// `reserve` drains (submits, waits, rewinds) only when this piece would not
		// fit what is left of the ring, and it leaves the caller's dispatch bracket
		// exactly as it found it.
		u64 ring_off = 0;
		void* host = ring->reserve(n, &ring_off);
		st.read_file(t.offset + done, (size_t)n, host);
		upload_range_staged(*ctx, *ring, a.res, a.off + done, ring_off, n);
		done += n;
	}
	if (own_bracket) {
		ctx->submit();
		ring->rewind();
	}
	return bytes;
}

GpuAlloc GpuCtx::upload_raw(const void* src, u64 bytes, bool to_weights_arena) {
	GpuAlloc a = to_weights_arena ? walloc(bytes) : aalloc(bytes);
	upload_into(a, src, bytes);
	return a;
}

GpuAlloc GpuCtx::upload_f32(const float* src, u64 count, bool to_weights_arena) {
	return upload_raw(src, count * 4, to_weights_arena);
}

GpuAlloc GpuCtx::upload_bf16(const void* src, u64 count, bool to_weights_arena) {
	return upload_raw(src, count * 2, to_weights_arena);
}

std::vector<float> GpuCtx::download_f32(const GpuAlloc& a, u64 count) {
	std::vector<float> out((size_t)count);
	ctx->download(a.res, a.off, out.data(), count * 4);
	return out;
}

std::vector<float> GpuCtx::download_f16(const GpuAlloc& a, u64 count) {
	std::vector<u16> h((size_t)count);
	ctx->download(a.res, a.off, h.data(), count * 2);
	std::vector<float> out((size_t)count);
	for (u64 i = 0; i < count; i++) out[(size_t)i] = f16_to_f32(h[(size_t)i]);
	return out;
}

void GpuCtx::begin() { ctx->begin(); }
void GpuCtx::submit() { ctx->submit(); }

void GpuCtx::new_layer() {
	// The weights arena is reused, so any pending copy into it must land first -
	// `begin()` on an open list would throw (it used to Reset it, silently
	// dropping the copies and the previous stages' dispatches with them).
	ctx->submit_if_recording();
	wa->reset();
	ring->rewind();
	ctx->begin();
}

void GpuCtx::end_layer() {
	ctx->submit();
	ring->rewind();
}

void GpuCtx::new_step() {
	ctx->submit_if_recording();
	aa->reset();
	ring->rewind();
}

// ── unified weight loading ──────────────────────────────────────────────────

void GpuCtx::upload_tensor_into(const SafeTensors& st, const StTensor& t, DType as,
                                const GpuAlloc& dst) {
	std::vector<u8> bytes = st.materialize(t, as);
	if (bytes.size() != (size_t)dst.bytes)
		throw MediaError("upload_tensor_into: '" + t.name + "' needs " +
		                 std::to_string(bytes.size()) + " bytes, allocation holds " +
		                 std::to_string(dst.bytes));
	upload_into(dst, bytes.data(), bytes.size());
}

GpuAlloc GpuCtx::upload_tensor(const SafeTensors& st, const StTensor& t, DType as) {
	// A raw tensor already in the target dtype is uploaded straight out of the
	// mapping - the same zero-copy path the int8 streamers use.
	//
	// Not `upload_file_into`, and the difference is measured rather than assumed:
	// reading the checkpoint through `read_file` (ReadFile on a shared handle,
	// serialised, plus the pinned-set / look-ahead lookups) beats the mapping only
	// for the *video* streamer's pattern - cold scattered reads of a file larger
	// than RAM. The image chain's file fits in the page cache after its first
	// pass, and there the mapping is a plain memcpy out of cached pages with no
	// syscall and no mutex, while `read_file` still pays both: swapping the two
	// measured 192 s for a 25-step 1024x1024 image against 98 s (7.08 s a step
	// against 3.23 s).
	if (st.is_raw(t) && t.dtype == as) {
		GpuAlloc a = walloc(t.nbytes);
		upload_range(*ctx, *ring, a.res, a.off, st.data_of(t), t.nbytes);
		return a;
	}
	std::vector<u8> bytes = st.materialize(t, as);
	GpuAlloc a = walloc(bytes.size());
	upload_range(*ctx, *ring, a.res, a.off, bytes.data(), bytes.size());
	return a;
}

namespace {

// The fp32 weight in its *original basis* (a convrot checkpoint stores the rotated
// matrix, so the rotation is undone) - whatever the file holds and however it is
// packed. This is `materialize`'s contract, so it is one call: a second, inlined
// copy of "which family is this and how is it scaled" is exactly the kind of
// duplication that lets one of the two drift.
std::vector<float> dequant_weight_f32(const SafeTensors& st, const std::string& base, i64 n,
                                      i64 k) {
	const StTensor& t = st.require(base + ".weight");
	if (t.dtype == DType::I4 && !st.weight_layout(t.name).known_family)
		throw MediaError("'" + base + "' is int4; it needs the int4 loader");
	std::vector<u8> bytes = st.materialize(t, DType::F32);
	std::vector<float> out((size_t)(n * k));
	const size_t have = bytes.size() / 4;
	if (have != out.size())
		throw MediaError("'" + base + ".weight' decoded to " + std::to_string(have) +
		                 " values, expected " + std::to_string(out.size()) + " (" +
		                 std::to_string(n) + " x " + std::to_string(k) + ")");
	if (!out.empty()) memcpy(out.data(), bytes.data(), out.size() * 4);
	return out;
}

QuantSpec default_int8_spec() {
	QuantSpec s;
	s.format = "int8_tensorwise";
	s.convrot = true;
	s.convrot_groupsize = 256;
	s.valid = true;
	return s;
}

// ── the streaming requantiser ─────────────────────────────────────────────
//
// The engine's quantised GEMMs all read int8 tensorwise + convrot, so a weight
// that arrives in any other precision has to be decoded and re-quantised. Doing
// that *natively* means doing it a slab of rows at a time: read the rows out of
// the checkpoint with `SafeTensors::dequant_rows`, rotate/quantise exactly the
// slab, upload exactly the slab. `quantize_weight` is per-output-row by
// construction (per-row amax, convrot along the row in 256-wide groups), so a
// slab-wise pass produces bit-identical int8 and scales to a whole-matrix pass.
//
// What it buys is host memory: the old path materialised the entire matrix as
// fp32 (4 bytes per element - a 490 MB Q4_K layer became a 2.0 GB fp32 vector
// before it could be quantised) and for a 20 GB checkpoint that is the difference
// between a bounded buffer and the machine's RAM. It is also the faster path:
// the slab is already in cache when it is rotated, instead of being written out
// and read back.
//
// `row0 == 0` is where the whole-matrix result is pinned; a LoRA that touches the
// module still needs the full fp32 matrix (the delta is dense over [n,k]), and
// its caller falls back to `dequant_weight_f32` for that case.
struct QuantUpload {
	u64 rows = 0;
	u64 cols = 0;
};

// The slab size: ~32 MB of fp32 working set, rounded down to whole rows. At 5376
// columns that is ~1500 rows; for a narrow matrix it is every row there is.
u64 slab_rows(i64 n, i64 k) {
	const i64 bytes_per_row = std::max<i64>(k, 1) * 4;
	const i64 rows = (32ll << 20) / bytes_per_row;
	return (u64)std::clamp<i64>(rows, 1, std::max<i64>(n, 1));
}

// Decode + rotate + quantise a whole [n, k] matrix of `t` into caller-owned
// buffers, splitting the row space across the CPU pool.
//
// Every stage is per-output-row by construction - `dequant_rows` decodes a row
// range as one contiguous block range, and the quantiser takes its amax per row
// and rotates along the row in whole 256-wide groups - so a task that owns a
// disjoint row range writes exactly the bytes the serial slab loop would, into
// exactly the place they belong. The output is therefore bit-identical whatever
// the thread count; the only thing threads change is that the first sampling step
// of a block-quantised checkpoint (the one that must decode and requantise the
// whole file) stops being one core's problem.
void requant_rows_into(const SafeTensors& st, const StTensor& t, i64 n, i64 k,
                       const QuantSpec& spec, int8_t* q, float* scale) {
	if (n <= 0 || k <= 0) return;
	const i64 slab = (i64)slab_rows(n, k);
	const unsigned threads = kernels::thread_count();
	// A few chunks per core, so a matrix whose rows differ in cost still balances;
	// never wider than one slab, so each task's fp32 working set stays the size the
	// serial loop's was.
	i64 per = (n + (i64)threads * 2 - 1) / ((i64)threads * 2);
	per = std::max<i64>(1, std::min<i64>(per, slab));
	const i64 ntask = (n + per - 1) / per;
	std::vector<std::exception_ptr> errs((size_t)ntask);
	kernels::parallel_for(ntask, [&](i64 task) {
		try {
			std::vector<float> buf;
			const i64 r1 = std::min<i64>(n, (task + 1) * per);
			for (i64 r0 = task * per; r0 < r1; r0 += per) {
				const i64 rows = std::min<i64>(per, r1 - r0);
				buf.resize((size_t)(rows * k));
				st.dequant_rows(t, r0, rows, buf.data());
				quantize_weight_into(buf.data(), rows, k, spec, q + r0 * k, scale + r0);
			}
		} catch (...) {
			errs[(size_t)task] = std::current_exception();
		}
	});
	for (const std::exception_ptr& e : errs)
		if (e) std::rethrow_exception(e);
}

// ── the requantised-block cache ──────────────────────────────────────────
//
// A streamed DiT block is decoded and requantised *on every sampling step*: the
// weights are the same bytes each time, and re-deriving a float form of it is
// CPU-side work. The cache keeps the result - the int8
// codes and their per-row scales, which is exactly what the arena receives - so
// from the second step on a block costs a copy.
//
// Bounded by the same host budget the pinned file-range cache uses
// (`host_cache_budget_bytes`), and filled in the order the loop streams
// (evicting the oldest entry first, which is the block the loop will need last).
// The key carries the *reader pointer* as well as the tensor name: two chains can
// hold a "blocks.0..." of their own, and a cache that conflated them would serve
// one model's weights to the other.
namespace {

struct RequantCache {
	struct Entry {
		std::vector<int8_t> q;
		std::vector<float> scale;
		i64 rows = 0, k = 0;
	};
	std::mutex mu;
	std::map<std::pair<const void*, std::string>, Entry> entries;
	std::deque<std::pair<const void*, std::string>> order;
	u64 bytes = 0;
	u64 hits = 0, misses = 0;   // reported under PHI_DIT_PLAN

	~RequantCache() {
		if (getenv("PHI_DIT_PLAN"))
			fprintf(stderr,
			        "[dit] requantised-weight cache: %llu hit(s), %llu miss(es), %zu entr(ies), "
			        "%.2f GB held\n",
			        (unsigned long long)hits, (unsigned long long)misses, entries.size(),
			        (double)bytes / 1073741824.0);
	}

	bool get(const void* owner, const std::string& name, i64* rows, i64* k, const int8_t** q,
	         const float** scale) {
		std::lock_guard<std::mutex> lk(mu);
		auto it = entries.find({owner, name});
		if (it == entries.end()) {
			misses++;
			return false;
		}
		hits++;
		*rows = it->second.rows;
		*k = it->second.k;
		*q = it->second.q.data();
		*scale = it->second.scale.data();
		return true;
	}
	void put(const void* owner, const std::string& name, i64 rows, i64 k, const int8_t* q,
	         u64 qbytes, const float* scale, u64 sbytes, u64 budget) {
		std::lock_guard<std::mutex> lk(mu);
		if (budget == 0 || qbytes + sbytes > budget) return;
		const auto key = std::make_pair(owner, name);
		if (entries.count(key)) return;
		Entry e;
		e.q.assign(q, q + qbytes);
		e.scale.assign(scale, scale + sbytes / sizeof(float));
		e.rows = rows;
		e.k = k;
		entries.emplace(key, std::move(e));
		order.push_back(key);
		bytes += qbytes + sbytes;
		while (bytes > budget && !order.empty()) {
			auto it = entries.find(order.front());
			if (it != entries.end()) {
				bytes -= it->second.q.size() + it->second.scale.size() * 4;
				entries.erase(it);
			}
			order.pop_front();
		}
	}
};

RequantCache& requant_cache() {
	static RequantCache c;
	return c;
}

// ── host-side weight cache ───────────────────────────────────────────────
//
// The pinned file-range cache and the requantised-block cache below both answer
// the same question - what may this process spend out of system RAM to avoid
// re-deriving a weight - so the two do not each spend the machine's memory.
//
// How much system memory the pinned weight set may take.
//
// The default is two thirds of what is free right now, capped at 9 GB.
//
// Measured on this box (15.4 GB of RAM, ~11 GB free): pinning 7.5 GB cut the
// sampling step from 18.3 s to 11.4 s, and left 3.7 GB free, which was enough for
// the engine's own pinned ring, the page cache the other pipelines run on and the
// browser the app is fronted by. Half of free memory was the first guess and it
// left value on the table; all of it would make the machine page, and a machine
// that pages is far slower than one that reads.
//
// PHI_DIT_HOSTCACHE_MB overrides it (0 disables the cache entirely, which is the
// switch to use when measuring what the disk alone does).

}  // namespace

// `upload_range` needs an open dispatch list, and the callers of these helpers
// differ: a model's `open()` has none, while the sampling loop's block loader is
// already inside one opened by `GpuCtx::new_layer`. Opening only when there is
// none, and closing only its own, is the discipline `upload_chunked` uses - and
// doing it once around the whole slab loop (not per slab) is what keeps a large
// matrix's upload from turning into one stream synchronise per 32 MB.
class UploadBracket {
public:
	explicit UploadBracket(GpuCtx& g) : g_(&g), own_(!g.ctx->recording()) {
		if (own_) g_->ctx->begin();
	}
	~UploadBracket() {
		if (own_ && g_) {
			g_->ctx->submit();
			g_->ring->rewind();
		}
	}
	UploadBracket(const UploadBracket&) = delete;
	UploadBracket& operator=(const UploadBracket&) = delete;

private:
	GpuCtx* g_ = nullptr;
	bool own_ = false;
};

}  // namespace

// Identifies a LoRA chain in a folded-weight cache key. A folded matrix is a
// function of (checkpoint, module, chain), and the chain is immutable for the
// life of a run, so its address is the identity - the same thing the node layer
// already keys its loaded-model cache on.
std::string lora_cache_tag(const LoraSet* lora) {
	if (!lora) return std::string();
	char buf[32];
	snprintf(buf, sizeof(buf), "|lora@%llx", (unsigned long long)(uintptr_t)lora);
	return std::string(buf);
}

u64 host_cache_budget_bytes() {
	if (const char* e = getenv("PHI_DIT_HOSTCACHE_MB")) {
		const long v = strtol(e, nullptr, 10);
		return v <= 0 ? 0 : (u64)v << 20;
	}
	// Two thirds of *physical* memory, capped at 9 GB - and never more than what is
	// *free* right now minus a reserve for the process that is asking.
	//
	// The free-memory term is the one that keeps a 16 GB box alive. `pin_range`
	// commits RAM the process cannot give back (a pinned range is not reclaimable),
	// while a checkpoint's own pages are clean and evictable, so a budget derived
	// from installed memory alone can pin the machine out of memory: measured here,
	// a 16 GB box with 9 GB of physical-memory budget pinned 8.97 GB for the H3
	// chain and the *next* allocation was answered with an OOM kill - the process
	// died mid-run with no error at all. The reserve keeps room for the engine's
	// own host buffers, the WebView2 UI and the file pages the streaming reads
	// still need.
	//
	// Basing this on installed memory rather than on the momentary free count is
	// deliberate. A block format has no raw form the streamer can upload: every
	// non-resident matrix is decoded, rotated and requantised, and the only thing
	// that keeps that off the critical path of *every* sampling step is this cache.
	// But the decode itself reads the entire file, so by the time the first block has
	// been cached the page cache has eaten the machine - and a free-memory budget
	// then collapses to a fraction of a gigabyte exactly when the cache is needed
	// most, so the loop re-derives the weights it derived one step ago. Physical
	// memory does not move, so this budget does not either; the file pages a large
	// budget displaces are clean and reclaimable, and the OS evicts them to make
	// room for the cache that is actually re-read.
	const u64 total = SafeTensors::host_total_bytes();
	u64 want = total / 3 * 2;
	const u64 cap = 9ull << 30;
	if (want > cap) want = cap;
	// The reserve scales with the machine the way the VRAM policy's does: a small
	// box needs a meaningful fraction back, a large one a bounded amount.
	u64 reserve = total / 8;
	if (reserve < (1ull << 30)) reserve = 1ull << 30;
	if (reserve > (4ull << 30)) reserve = 4ull << 30;
	const u64 free_now = SafeTensors::host_free_bytes();
	if (free_now > reserve && want > free_now - reserve) want = free_now - reserve;
	return want & ~((u64)4095);
}

// The same reserve, exposed so a *pin loop* can stop while it still has room:
// `host_cache_budget_bytes()` is decided once per weight set and stays put (see
// the note above), while the pinning itself proceeds block by block and must not
// spend the last free page on the machine.
u64 host_cache_reserve_bytes() {
	const u64 total = SafeTensors::host_total_bytes();
	u64 reserve = total / 8;
	if (reserve < (1ull << 30)) reserve = 1ull << 30;
	if (reserve > (4ull << 30)) reserve = 4ull << 30;
	return reserve;
}

// ── the native grouped-scale decode ───────────────────────────────────────
//
// A checkpoint in the `asym_w4a8_int8` / `w6a8_int8` family keeps its weights
// packed: 4 or 6 bits per weight, a 16-entry codebook (4-bit only), a per-group
// relative scale and a per-output-row channel scale. ComfyUI runs those layers
// through its int8 GEMM with the *packed* weight as the operand, folding the
// codebook and the two scales into the operand inside the kernel
// (`_GROUPED_INT8_FORMATS` -> AsymW4A8Int8Layout in comfy/ops.py). This is the
// same arrangement for this engine's int8 GEMM, in one place, so the image chain
// and the video chain take exactly the same route: `w4a8_pack_rows` is called by
// `load_quant_linear` (the image DiT and the VAEs) and by
// `upload_quant_linear_i8` (the H3 DiT), and both hand the kernel the same four
// staged tensors.
//
// Why it has to be on the device at all: the decode is elementwise, but it is
// 11 GB of packed codes into 23 GB of int8 *per sampling step* for the H3 DiT,
// because the streamed blocks are re-read on every step and 23 GB does not fit a
// 16 GB box - so no host cache can hold the result and no step can reuse it. On
// the CPU that measured 219 s of a 261 s step; on the device it is one pass over
// the packed bytes, which are also the bytes the checkpoint costs over the bus
// (12.6 GB per step for the H3 DiT, against 17 GB for the same model as int8).
//
// The kernel is bit-exact with the host decoder and the host quantiser
// (`core/kernels/w4a8_expand.cpp` carries both references), so this moves work
// rather than changing arithmetic: `PHI_W4A8_CPU=1` forces the old path, and the
// two produce identical bytes.
struct W4a8Plan {
	CodeLayout layout = CodeLayout::Nibble4;
	const StTensor* codes = nullptr;
	const StTensor* srel = nullptr;
	const StTensor* sch = nullptr;        // null: no channel scale
	const StTensor* codebook = nullptr;   // null: the code is the value (six-bit)
	i64 group = 0;
	i64 codes_row_bytes = 0;
	i64 srel_row_bytes = 0;
	RelDType rel_dtype = RelDType::F32;
};

// True when `t` is a weight this path can decode, filling `out` with where its
// parts live. The conditions are exactly the ones the kernel assumes, and each is
// checked rather than trusted: a family that does not match falls back to the
// generic requantiser below, which costs speed and nothing else.
bool w4a8_plan_of(const SafeTensors& st, const StTensor& t, W4a8Plan* out) {
	const SafeTensors::WeightLayout L = st.weight_layout(t.name);
	const std::string& f = L.spec.format;
	const bool six = (f == "w6a8_int8" || f == "w6a8");
	if (f != "asym_w4a8_int8" && !six) return false;
	if (t.shape.size() != 2 || t.dtype != DType::I8) return false;
	// The decode is a per-element lookup, which is only the right answer when the
	// codes are already in the basis the activations are rotated into. A file that
	// is *not* convrot needs a 256-wide Hadamard rotation in the middle of it and
	// goes through the generic path, which owns that rotation.
	if (!L.spec.convrot) return false;
	if (!L.rel_scale || L.rel_scale->shape.size() != 2) return false;
	switch (L.rel_scale->dtype) {
		case DType::F32:
		case DType::F16:
		case DType::BF16:
		case DType::F8_E4M3:
		case DType::F8_E5M2: break;
		default: return false;
	}

	const i64 n = t.shape[0], k = t.shape[1];
	const i64 group = L.group > 0 ? L.group : (i64)L.spec.group_size;
	const i64 codes_per_group = six ? 4 : 2;
	if (group <= 0 || group % codes_per_group != 0 || k % group != 0) return false;
	const i64 need = (group / codes_per_group) * (k / group);
	if (need <= 0 || t.row_bytes_stored < (u64)need) return false;
	if (L.rel_scale->shape[0] != n || L.rel_scale->shape[1] != k / group) return false;

	if (six) {
		// The six-bit family has no codebook: the code *is* the value.
		if (L.codebook) return false;
	} else {
		if (!L.codebook || L.codebook->dtype != DType::F32 || L.codebook->numel != 16)
			return false;
	}
	// The channel scale is read as one fp32 per row. A one-entry table is
	// broadcast below; any other dtype or length is the generic path's problem.
	if (L.row_scale) {
		if (L.row_scale->dtype != DType::F32) return false;
		const i64 rows = L.row_scale->numel;
		if (rows != 1 && rows != n) return false;
	}

	RelDType rdt = RelDType::F32;
	switch (L.rel_scale->dtype) {
		case DType::F32: rdt = RelDType::F32; break;
		case DType::F16: rdt = RelDType::F16; break;
		case DType::BF16: rdt = RelDType::BF16; break;
		case DType::F8_E4M3: rdt = RelDType::F8_E4M3; break;
		case DType::F8_E5M2: rdt = RelDType::F8_E5M2; break;
		default: break;
	}
	out->layout = six ? CodeLayout::Packed6 : CodeLayout::Nibble4;
	out->codes = &t;
	out->srel = L.rel_scale;
	out->sch = L.row_scale;
	out->codebook = six ? nullptr : L.codebook;
	out->group = group;
	out->codes_row_bytes = (i64)t.row_bytes_stored;
	out->srel_row_bytes = (i64)st.row_bytes(*L.rel_scale);
	out->rel_dtype = rdt;
	return true;
}

// Stage the four input tensors into the ring as one region and run the kernel.
//
// One reservation for the whole tensor rather than one per part: the ring is a
// bump allocator whose "drain" is a full stream synchronise, so four separate
// reservations would be four separate chances to drain. The parts are 256-byte
// aligned inside the region because the kernel reads them as typed arrays.
//
// The bytes come through `read_file`, i.e. the pinned host cache and the
// look-ahead slot the streaming loop maintains - the same path every other
// streamed tensor takes, so a 12.6 GB-per-step checkpoint gets the same benefit
// from the cache an int8 one does.
void w4a8_pack_rows(GpuCtx& g, const SafeTensors& st, const W4a8Plan& pl, const GpuAlloc& w,
                    const GpuAlloc& s, i64 n, i64 k) {
	const StTensor& codes = *pl.codes;
	const StTensor& srel = *pl.srel;
	// A one-entry channel table covers every row; materialise it per row in the
	// ring rather than making the kernel branch per row for it.
	const bool sch_scalar = pl.sch && pl.sch->numel == 1 && pl.sch->nbytes > 0;
	const u64 codes_bytes = (u64)n * (u64)pl.codes_row_bytes;
	const u64 srel_bytes = srel.nbytes;
	const u64 sch_bytes = !pl.sch ? 0 : (sch_scalar ? (u64)n * 4 : pl.sch->nbytes);
	const u64 cb_bytes = pl.codebook ? pl.codebook->nbytes : 0;
	auto up256 = [](u64 v) { return (v + 255ull) & ~255ull; };
	const u64 total = up256(codes_bytes) + up256(srel_bytes) + up256(sch_bytes) + up256(cb_bytes);

	u64 ring_off = 0;
	u8* host = (u8*)g.ring->reserve(total, &ring_off);
	u64 at = 0;
	const u64 off_codes = at;
	st.read_file(codes.offset, (size_t)codes_bytes, host + at);
	at += up256(codes_bytes);
	const u64 off_srel = at;
	st.read_file(srel.offset, (size_t)srel_bytes, host + at);
	at += up256(srel_bytes);
	const u64 off_sch = at;
	if (sch_bytes > 0) {
		if (sch_scalar) {
			const float v = st.row_scale_value(*pl.sch, 0);
			for (i64 r = 0; r < n; r++) memcpy(host + at + (u64)r * 4, &v, 4);
		} else {
			st.read_file(pl.sch->offset, (size_t)sch_bytes, host + at);
		}
		at += up256(sch_bytes);
	}
	const u64 off_cb = at;
	if (cb_bytes > 0) st.read_file(pl.codebook->offset, (size_t)cb_bytes, host + at);

	W4a8ExpandArgs a;
	a.pool = g.ring->buffer();
	a.codes_off = ring_off + off_codes;
	a.srel_off = ring_off + off_srel;
	a.sch_off = ring_off + off_sch;
	a.cb_off = ring_off + off_cb;
	a.q = w.res;
	a.q_off = w.off;
	a.s = s.res;
	a.s_off = s.off;
	a.rows = n;
	a.k = k;
	a.group = pl.group;
	a.codes_row_bytes = pl.codes_row_bytes;
	a.srel_row_bytes = pl.srel_row_bytes;
	a.rel_dtype = pl.rel_dtype;
	a.layout = pl.layout;
	a.has_channel_scale = sch_bytes > 0;
	a.has_codebook = pl.codebook != nullptr;
	dispatch_w4a8_expand(*g.ctx, a);
}

// `PHI_W4A8_CPU=1` forces the generic requantiser. That is the A/B which
// establishes that the two paths agree byte for byte: the device decode
// reproduces the host decoder *and* the host quantiser exactly, so a checkpoint
// produces the same output either way and the switch is only a speed difference.
bool w4a8_native_enabled() {
	static const bool on = [] {
		const char* e = getenv("PHI_W4A8_CPU");
		return !(e && *e && *e != '0');
	}();
	return on;
}

// ── the file's own precision ────────────────────────────────────────────────

namespace {

// The device dtype a dense source is stored in: the file's own precision for
// f16 / bf16 / f32, fp16 for fp8 (and nvfp4), which is exact.
DType dense_target_dtype(const SafeTensors& st, const StTensor& t) {
	switch (t.dtype) {
		case DType::F32: return DType::F32;
		case DType::BF16: return DType::BF16;
		case DType::F16: return DType::F16;
		default: break;
	}
	// fp8 (both spellings) and nvfp4: decode to fp16.
	(void)st;
	return DType::F16;
}

// Decode + convert one dense source into its slice of `dst` (a [n,k] matrix at
// `row0` of the destination), a slab of rows at a time. A raw f16/bf16/f32
// tensor is uploaded straight off the file in its own precision; fp8 / nvfp4 pay
// a slab decode through `dequant_rows`.
void dense_upload_into(GpuCtx& g, const SafeTensors& st, const StTensor& t, const GpuAlloc& dst,
                       i64 row0, i64 n, i64 k, DType target) {
	const u64 esz = (u64)dtype_size(target);
	const GpuAlloc slice = [&] {
		GpuAlloc a = dst;
		a.off += (u64)row0 * (u64)k * esz;
		a.bytes = (u64)n * (u64)k * esz;
		return a;
	}();
	if (st.is_raw(t) && t.dtype == target) {
		g.upload_file_into(slice, st, t);
		return;
	}
	const u64 slab = slab_rows(n, k);
	std::vector<float> buf;
	std::vector<u8> cvt;
	const bool own = !g.ctx->recording();
	if (own) g.ctx->begin();
	for (i64 r = 0; r < n; r += (i64)slab) {
		const i64 rows = std::min<i64>((i64)slab, n - r);
		buf.resize((size_t)(rows * k));
		st.dequant_rows(t, r, rows, buf.data());
		cvt.resize((size_t)(rows * k) * esz);
		convert_from_f32(buf.data(), target, cvt.data(), (size_t)(rows * k));
		upload_range(*g.ctx, *g.ring, slice.res, slice.off + (u64)r * (u64)k * esz, cvt.data(),
		             cvt.size());
	}
	if (own) {
		g.ctx->submit();
		g.ring->rewind();
	}
}

}  // namespace

DenseUpload upload_dense_linear(GpuCtx& g, const SafeTensors& st, const std::string& base,
                                GpuArena& arena) {
	const StTensor& t = st.require(base + ".weight");
	if (t.shape.size() != 2)
		throw MediaError("dense linear: '" + base + ".weight' is not a 2-D matrix");
	DenseUpload out;
	out.n = t.shape[0];
	out.k = t.shape[1];
	out.dtype = dense_target_dtype(st, t);
	const u64 esz = (u64)dtype_size(out.dtype);
	out.w = arena.alloc((u64)out.n * (u64)out.k * esz);
	dense_upload_into(g, st, t, out.w, 0, out.n, out.k, out.dtype);
	return out;
}

DenseUpload load_dense_linear(GpuCtx& g, const SafeTensors& st, const std::string& base,
                              GpuArena* into, const LoraSet* lora) {
	const StTensor& t = st.require(base + ".weight");
	if (t.shape.size() != 2)
		throw MediaError("dense linear: '" + base + ".weight' is not a 2-D matrix");
	const i64 n = t.shape[0], k = t.shape[1];
	const DType dt = dense_target_dtype(st, t);
	GpuArena& arena = into ? *into : *g.wa;
	if (!lora || !lora->touches(base)) return upload_dense_linear(g, st, base, arena);
	// A LoRA delta is dense over [n,k], so it is folded into the fp32 weight and the
	// result is stored in the file's own target dtype. Rare - the block loops apply
	// the delta at run time instead (see LoraTail), so this is the folded, single-
	// load path only.
	std::vector<float> w = dequant_weight_f32(st, base, n, k);
	lora->apply(base, w.data(), n, k);
	const u64 esz = (u64)dtype_size(dt);
	std::vector<u8> cvt((size_t)(n * k) * esz);
	convert_from_f32(w.data(), dt, cvt.data(), (size_t)(n * k));
	DenseUpload out;
	out.n = n;
	out.k = k;
	out.dtype = dt;
	out.w = arena.alloc((u64)n * (u64)k * esz);
	// A folding call is a resident, single-load layer, so the whole (small) fp32
	// matrix is in hand; the upload is bracketed the same way the int8 path's is, so
	// it works whether or not the caller already has a dispatch list open.
	const bool own = !g.ctx->recording();
	if (own) g.ctx->begin();
	g.upload_into(out.w, cvt.data(), cvt.size());
	if (own) {
		g.ctx->submit();
		g.ring->rewind();
	}
	return out;
}

DenseUpload load_dense3_linear(GpuCtx& g, const SafeTensors& st, const std::string& base_q,
                               const std::string& base_k, const std::string& base_v,
                               GpuArena* into, const LoraSet* lora) {
	const char* bases[3] = {base_q.c_str(), base_k.c_str(), base_v.c_str()};
	const StTensor* ts[3] = {&st.require(base_q + ".weight"), &st.require(base_k + ".weight"),
	                         &st.require(base_v + ".weight")};
	const i64 k = ts[0]->shape[1];
	for (int i = 0; i < 3; i++)
		if (ts[i]->shape.size() != 2 || ts[i]->shape[1] != k)
			throw MediaError(std::string("dense3 linear: shape mismatch at ") + bases[i]);
	const DType dt = dense_target_dtype(st, *ts[0]);
	for (int i = 1; i < 3; i++)
		if (dense_target_dtype(st, *ts[i]) != dt)
			throw MediaError("dense3 linear: q/k/v store different precisions for " +
			                 std::string(bases[i]));
	i64 N = 0;
	for (int i = 0; i < 3; i++) N += ts[i]->shape[0];
	DenseUpload out;
	out.n = N;
	out.k = k;
	out.dtype = dt;
	const u64 esz = (u64)dtype_size(dt);
	GpuArena& arena = into ? *into : *g.wa;
	out.w = arena.alloc((u64)N * (u64)k * esz);
	const bool own = !g.ctx->recording();
	if (own) g.ctx->begin();
	i64 row0 = 0;
	for (int i = 0; i < 3; i++) {
		const StTensor& t = *ts[i];
		const i64 ni = t.shape[0];
		if (lora && lora->touches(bases[i])) {
			std::vector<float> w = dequant_weight_f32(st, bases[i], ni, k);
			lora->apply(bases[i], w.data(), ni, k);
			std::vector<u8> cvt((size_t)(ni * k) * esz);
			convert_from_f32(w.data(), dt, cvt.data(), (size_t)(ni * k));
			GpuAlloc slice = out.w;
			slice.off += (u64)row0 * (u64)k * esz;
			slice.bytes = (u64)ni * (u64)k * esz;
			g.upload_into(slice, cvt.data(), cvt.size());
		} else {
			dense_upload_into(g, st, t, out.w, row0, ni, k, dt);
		}
		row0 += ni;
	}
	if (own) {
		g.ctx->submit();
		g.ring->rewind();
	}
	return out;
}

QuantLinear load_quant_linear(GpuCtx& g, const SafeTensors& st, const std::string& base,
                              GpuArena* into, const LoraSet* lora) {
	auto alloc = [&](u64 bytes) { return into ? into->alloc(bytes) : g.walloc(bytes); };
	const StTensor& t = st.require(base + ".weight");
	if (t.shape.size() != 2)
		throw MediaError("load_quant_linear: '" + base + ".weight' is not a 2-D matrix");
	const i64 n = t.shape[0], k = t.shape[1];
	// A float source keeps its own precision and runs the dense GEMM: this is the
	// conversion that used to happen here (float -> int8 requantise) and must not.
	if (weight_is_dense_float(st, t)) {
		DenseUpload du = load_dense_linear(g, st, base, into, lora);
		QuantLinear qd;
		qd.d = du.w;
		qd.dtype = du.dtype;
		qd.n = du.n;
		qd.k = du.k;
		return qd;
	}
	// Verbatim upload is only legal for the *original* int8 tensorwise form: one
	// code per element with one fp32 scale per row. A packed family (a codebook,
	// a six-bit pack, nvfp4) or a grouped int8 has to be decoded first.
	const bool raw_int8 = st.is_plain_int8(t);
	const bool touched = lora && lora->touches(base);

	QuantLinear q;
	q.n = n;
	q.k = k;
	if (raw_int8 && !touched) {
		// Shipped int8 tensorwise: upload verbatim (the caller may also have a
		// `.comfy_quant` spelling the convrot, which is already baked into the
		// bytes - the kernels assume convrot regardless).
		//
		// Out of the mapping, which for this chain is the fast path: the image
		// checkpoint fits the page cache after its first pass, so this is a memcpy
		// from cached pages, while routing it through `read_file` measured 7.08 s a
		// step instead of 3.23 s (see `GpuCtx::upload_tensor`).
		q.w = alloc(t.nbytes);
		upload_range(*g.ctx, *g.ring, q.w.res, q.w.off, st.data_of(t), t.nbytes);
		std::vector<float> s = tensor_to_f32(st, st.require(base + ".weight_scale"));
		if ((i64)s.size() != n) throw MediaError("weight_scale length mismatch: " + base);
		q.scale = alloc((u64)s.size() * 4);
		upload_range(*g.ctx, *g.ring, q.scale.res, q.scale.off, s.data(), s.size() * 4);
		return q;
	}

	// f16 / bf16 / f32 / a packed family: decode → adapt → requantise, in slabs of
	// rows.
	//
	// A grouped-scale checkpoint (asym_w4a8_int8, w6a8_int8) is the exception, and
	// it is the same exception in both chains: its codes, group table, codebook and
	// channel scale go to the device as they are stored and the int8 form is built
	// there (`w4a8_pack_rows`), which the video DiT's loader does too. A LoRA that
	// touches the module cannot take it - a dense delta belongs to the *values*,
	// and this path is the one loader that folds - so it falls through to the
	// slab-wise path below, which is the only one that materialises fp32.
	if (w4a8_native_enabled() && !touched) {
		W4a8Plan plan;
		if (w4a8_plan_of(st, t, &plan)) {
			q.w = alloc((u64)n * (u64)k);
			q.scale = alloc((u64)n * 4);
			w4a8_pack_rows(g, st, plan, q.w, q.scale, n, k);
			return q;
		}
	}

	// f16 / bf16 / f32 / a packed family: decode → adapt → requantise, in slabs of
	// rows.
	//
	// The destination is sized first and uploaded slab by slab, so the matrix is
	// never whole in host memory. A LoRA that touches the module is the one case
	// that needs the full fp32 matrix (its delta is dense over [n,k]); it takes the
	// old whole-matrix path, which is also what it did before.
	QuantSpec spec = requant_spec_for(st, base);
	if (spec.convrot && (k % spec.convrot_groupsize) != 0)
		throw MediaError("'" + base + "' has K=" + std::to_string(k) +
		                 ", not a multiple of the convrot group " +
		                 std::to_string(spec.convrot_groupsize) +
		                 "; load it on the bf16 path instead");
	if (touched) {
		// dequant_weight_f32 undoes the file's rotation, so the delta is applied in
		// the original basis and the result *does* have to be rotated on the way
		// back in - whatever the file declared.
		std::vector<float> w = dequant_weight_f32(st, base, n, k);
		lora->apply(base, w.data(), n, k);
		QuantInt8 qi = quantize_weight(w.data(), n, k, default_int8_spec());
		q.w = alloc((u64)qi.q.size());
		upload_range(*g.ctx, *g.ring, q.w.res, q.w.off, qi.q.data(), qi.q.size());
		q.scale = alloc((u64)qi.scale.size() * 4);
		upload_range(*g.ctx, *g.ring, q.scale.res, q.scale.off, qi.scale.data(),
		             qi.scale.size() * 4);
		return q;
	}
	q.w = alloc((u64)n * (u64)k);
	q.scale = alloc((u64)n * 4);
	// A layer that was requantised on an earlier step is a copy this time - the
	// decode + rotate + quantise pass is skipped entirely. This is what a streamed
	// layer costs per step: the sampling loop re-reads every non-resident layer
	// on every step, and for a float format that means re-deriving the int8 form
	// each time.
	{
		const int8_t* cq = nullptr;
		const float* cs = nullptr;
		i64 crows = 0, ck = 0;
		if (requant_cache().get(&st, base, &crows, &ck, &cq, &cs) && crows == n && ck == k) {
			const bool own = !g.ctx->recording();
			if (own) g.ctx->begin();
			upload_range(*g.ctx, *g.ring, q.w.res, q.w.off, cq, (u64)n * (u64)k);
			upload_range(*g.ctx, *g.ring, q.scale.res, q.scale.off, cs, (u64)n * 4);
			if (own) {
				g.ctx->submit();
				g.ring->rewind();
			}
			return q;
		}
	}
	const bool cacheable =
	    host_cache_budget_bytes() > 0 && (u64)n * (u64)k + (u64)n * 4 <= host_cache_budget_bytes();
	if (cacheable) {
		// The whole matrix fits the host budget - the common case, and the one every
		// streamed block takes. Requantise it once, in parallel, keep it, then upload.
		std::vector<int8_t> whole_q((size_t)(n * k));
		std::vector<float> whole_s((size_t)n);
		requant_rows_into(st, t, n, k, spec, whole_q.data(), whole_s.data());
		const u64 slab = slab_rows(n, k);
		const bool own_bracket = !g.ctx->recording();
		if (own_bracket) g.ctx->begin();
		for (i64 r0 = 0; r0 < n; r0 += (i64)slab) {
			const i64 rows = std::min<i64>((i64)slab, n - r0);
			upload_range(*g.ctx, *g.ring, q.w.res, q.w.off + (u64)r0 * (u64)k,
			             whole_q.data() + (size_t)r0 * (size_t)k, (u64)rows * (u64)k);
			upload_range(*g.ctx, *g.ring, q.scale.res, q.scale.off + (u64)r0 * 4,
			             whole_s.data() + (size_t)r0, (u64)rows * 4);
		}
		if (own_bracket) {
			g.ctx->submit();
			g.ring->rewind();
		}
		requant_cache().put(&st, base, n, k, whole_q.data(), whole_q.size(), whole_s.data(),
		                    whole_s.size() * 4, host_cache_budget_bytes());
		return q;
	}
	// Larger than the host budget: stream it a slab at a time and keep nothing, so
	// the working set stays bounded instead of the whole int8 form resident.
	{
		const u64 slab = slab_rows(n, k);
		std::vector<float> buf;
		// The calls above may already be inside someone else's bracket (a model's
		// `open()`), and each `upload_range` needs one open.
		const bool own_bracket = !g.ctx->recording();
		if (own_bracket) g.ctx->begin();
		for (i64 r0 = 0; r0 < n; r0 += (i64)slab) {
			const i64 rows = std::min<i64>((i64)slab, n - r0);
			buf.resize((size_t)(rows * k));
			st.dequant_rows(t, r0, rows, buf.data());
			QuantInt8 qi = quantize_weight(buf.data(), rows, k, spec);
			upload_range(*g.ctx, *g.ring, q.w.res, q.w.off + (u64)r0 * (u64)k, qi.q.data(),
			             qi.q.size());
			upload_range(*g.ctx, *g.ring, q.scale.res, q.scale.off + (u64)r0 * 4, qi.scale.data(),
			             qi.scale.size() * 4);
		}
		if (own_bracket) {
			g.ctx->submit();
			g.ring->rewind();
		}
	}
	return q;
}

I8Upload upload_quant_linear_i8(GpuCtx& g, const SafeTensors& st, const std::string& base,
                                GpuArena& arena, const LoraSet* lora) {
	UploadBracket bracket(g);
	const StTensor& t = st.require(base + ".weight");
	if (t.shape.size() != 2)
		throw MediaError("upload_quant_linear_i8: '" + base + ".weight' is not a 2-D matrix");
	I8Upload out;
	out.n = t.shape[0];
	out.k = t.shape[1];
	// A grouped-scale checkpoint is decoded on the device (see w4a8_plan_of).
	//
	// The `lora` argument does not divert this path: a delta is never folded into a
	// streamed weight here (the block loop re-reads it on every step, so folding
	// would mean decoding, adapting and requantising the whole checkpoint per step -
	// measured at 660 s a step against 6.8 s without). The correction is applied to
	// the GEMM's *output* instead, by `apply_lora_tail`, which is what the base
	// weight's own comment in `load_linear_pair` (av_dit.cpp) describes. It is accepted and ignored
	// so that a caller which does fold (a resident, single-load layer) is not
	// silently given the wrong bytes; those callers use `load_quant_linear`.
	(void)lora;
	if (w4a8_native_enabled()) {
		W4a8Plan plan;
		if (w4a8_plan_of(st, t, &plan)) {
			out.w = arena.alloc((u64)out.n * (u64)out.k);
			out.s = arena.alloc((u64)out.n * 4);
			w4a8_pack_rows(g, st, plan, out.w, out.s, out.n, out.k);
			return out;
		}
	}
	// A shipped int8 tensorwise weight whose file declares convrot is uploaded
	// *verbatim*: the block loop (and the video VAE / the text tower) was written
	// against exactly these bytes, and re-deriving them through fp32 would be both
	// slower and a (tiny) conversion - the opposite of goal #1. A LoRA that folds
	// needs the fp32 form, and a file that is *not* convrot needs the rotation the
	// quantiser applies, so both skip this.
	if (st.is_plain_int8(t) && st.weight_layout(t.name).spec.convrot &&
	    !(lora && lora->touches(base))) {
		out.w = arena.alloc(t.nbytes);
		g.upload_file_into(out.w, st, t);
		std::vector<float> sc = tensor_to_f32(st, st.require(base + ".weight_scale"));
		if ((i64)sc.size() != out.n) throw MediaError("weight_scale length mismatch: " + base);
		out.s = arena.alloc((u64)sc.size() * 4);
		upload_range(*g.ctx, *g.ring, out.s.res, out.s.off, sc.data(), sc.size() * 4);
		return out;
	}
	QuantSpec spec = requant_spec_for(st, base);
	if (spec.convrot && (out.k % spec.convrot_groupsize) != 0)
		throw MediaError("'" + base + "' has K=" + std::to_string(out.k) +
		                 ", not a multiple of the convrot group " +
		                 std::to_string(spec.convrot_groupsize) +
		                 "; load it on the bf16 path instead");
	if (lora && lora->touches(base)) {
		// A LoRA's delta is dense over the whole [n,k] plane, so it has to be folded
		// into the fp32 weight before that weight is quantised.
		//
		// Crucially it is folded a *slab* of rows at a time, exactly like the plain
		// requantiser below. Materialising the whole matrix instead (4 bytes per
		// element - 463 MB for one qkv of this stack, for each of the four matrices of
		// every block, on every sampling step) is not merely slower: its footprint is
		// what evicts the checkpoint's own mapped pages, so the next matrix comes back
		// from disk and the chain falls from ~850 MB/s to ~150 MB/s. That is what made
		// a LoRA chain an order of magnitude slower than the same chain without one.
		const std::string key = base + lora_cache_tag(lora);
		{
			const int8_t* cq = nullptr;
			const float* cs = nullptr;
			i64 crows = 0, ck = 0;
			if (requant_cache().get(&st, key, &crows, &ck, &cq, &cs) && crows == out.n &&
			    ck == out.k) {
				UploadBracket upload(g);
				upload_range(*g.ctx, *g.ring, out.w.res, out.w.off, cq, (u64)out.n * (u64)out.k);
				upload_range(*g.ctx, *g.ring, out.s.res, out.s.off, cs, (u64)out.n * 4);
				return out;
			}
		}
		out.w = arena.alloc((u64)out.n * (u64)out.k);
		out.s = arena.alloc((u64)out.n * 4);
		// The delta is added in the basis `dequant_rows` hands back, which is the one
		// the file stores: for a convrot checkpoint those rows are rotated, so the
		// rotation is undone first (rot() is an involution) and re-applied by the
		// quantiser. This is the order the whole-matrix path used, done a slab at a
		// time so the fp32 working set stays the size one slab needs.
		const bool stored_rotated = !spec.convrot;
		const u64 slab = slab_rows(out.n, out.k);
		const u64 nk = (u64)out.n * (u64)out.k;
		const bool cacheable =
		    host_cache_budget_bytes() > 0 && nk + (u64)out.n * 4 <= host_cache_budget_bytes();
		std::vector<int8_t> whole_q;
		std::vector<float> whole_s;
		if (cacheable) {
			whole_q.assign((size_t)nk, 0);
			whole_s.assign((size_t)out.n, 0.0f);
		}
		std::vector<float> buf;
		const bool own_bracket = !g.ctx->recording();
		if (own_bracket) g.ctx->begin();
		for (i64 r0 = 0; r0 < out.n; r0 += (i64)slab) {
			const i64 rows = std::min<i64>((i64)slab, out.n - r0);
			buf.resize((size_t)(rows * out.k));
			st.dequant_rows(t, r0, rows, buf.data());
			if (stored_rotated)
				convrot_forward(buf.data(), buf.data(), rows, out.k, spec.convrot_groupsize);
			lora->apply_rows(base, buf.data(), r0, rows, out.k);
			QuantInt8 qi = quantize_weight(buf.data(), rows, out.k, default_int8_spec());
			if (cacheable) {
				memcpy(whole_q.data() + (size_t)r0 * (size_t)out.k, qi.q.data(), qi.q.size());
				memcpy(whole_s.data() + (size_t)r0, qi.scale.data(), qi.scale.size() * 4);
			}
			upload_range(*g.ctx, *g.ring, out.w.res, out.w.off + (u64)r0 * (u64)out.k,
			             qi.q.data(), qi.q.size());
			upload_range(*g.ctx, *g.ring, out.s.res, out.s.off + (u64)r0 * 4, qi.scale.data(),
			             qi.scale.size() * 4);
		}
		if (own_bracket) {
			g.ctx->submit();
			g.ring->rewind();
		}
		if (cacheable)
			requant_cache().put(&st, key, out.n, out.k, whole_q.data(), whole_q.size(),
			                    whole_s.data(), whole_s.size() * 4, host_cache_budget_bytes());
		return out;
	}
	out.w = arena.alloc((u64)out.n * (u64)out.k);
	out.s = arena.alloc((u64)out.n * 4);
	// A matrix that was requantised for an earlier step is a copy this time. The
	// check is by (checkpoint, tensor name) and the entry holds exactly what the
	// arena receives, so a hit is the whole decode + rotate + quantise pass skipped.
	{
		const int8_t* cq = nullptr;
		const float* cs = nullptr;
		i64 crows = 0, ck = 0;
		if (requant_cache().get(&st, base, &crows, &ck, &cq, &cs) && crows == out.n &&
		    ck == out.k) {
			UploadBracket upload(g);
			upload_range(*g.ctx, *g.ring, out.w.res, out.w.off, cq, (u64)out.n * (u64)out.k);
			upload_range(*g.ctx, *g.ring, out.s.res, out.s.off, cs, (u64)out.n * 4);
			return out;
		}
	}
	// A matrix that fits the host budget is requantised once (in parallel) and
	// cached, so a streamed block's second step is a copy; one that does not is
	// streamed a slab at a time and kept nowhere, because it would not fit anyway.
	const bool cacheable =
	    host_cache_budget_bytes() > 0 &&
	    (u64)out.n * (u64)out.k + (u64)out.n * 4 <= host_cache_budget_bytes();
	if (cacheable) {
		std::vector<int8_t> whole_q((size_t)(out.n * out.k));
		std::vector<float> whole_s((size_t)out.n);
		requant_rows_into(st, t, out.n, out.k, spec, whole_q.data(), whole_s.data());
		const u64 slab = slab_rows(out.n, out.k);
		const bool own_bracket = !g.ctx->recording();
		if (own_bracket) g.ctx->begin();
		for (i64 r0 = 0; r0 < out.n; r0 += (i64)slab) {
			const i64 rows = std::min<i64>((i64)slab, out.n - r0);
			upload_range(*g.ctx, *g.ring, out.w.res, out.w.off + (u64)r0 * (u64)out.k,
			             whole_q.data() + (size_t)r0 * (size_t)out.k, (u64)rows * (u64)out.k);
			upload_range(*g.ctx, *g.ring, out.s.res, out.s.off + (u64)r0 * 4,
			             whole_s.data() + (size_t)r0, (u64)rows * 4);
		}
		if (own_bracket) {
			g.ctx->submit();
			g.ring->rewind();
		}
		requant_cache().put(&st, base, out.n, out.k, whole_q.data(), whole_q.size(),
		                    whole_s.data(), whole_s.size() * 4, host_cache_budget_bytes());
		return out;
	}
	{
		const u64 slab = slab_rows(out.n, out.k);
		std::vector<float> buf;
		const bool own_bracket = !g.ctx->recording();
		if (own_bracket) g.ctx->begin();
		for (i64 r0 = 0; r0 < out.n; r0 += (i64)slab) {
			const i64 rows = std::min<i64>((i64)slab, out.n - r0);
			buf.resize((size_t)(rows * out.k));
			st.dequant_rows(t, r0, rows, buf.data());
			QuantInt8 qi = quantize_weight(buf.data(), rows, out.k, spec);
			upload_range(*g.ctx, *g.ring, out.w.res, out.w.off + (u64)r0 * (u64)out.k, qi.q.data(),
			             qi.q.size());
			upload_range(*g.ctx, *g.ring, out.s.res, out.s.off + (u64)r0 * 4, qi.scale.data(),
			             qi.scale.size() * 4);
		}
		if (own_bracket) {
			g.ctx->submit();
			g.ring->rewind();
		}
	}
	return out;
}

I4Upload upload_quant_linear_w4a4(GpuCtx& g, const SafeTensors& st, const std::string& base,
                                  GpuArena& arena, const LoraSet* lora) {
	UploadBracket bracket(g);
	const StTensor& t = st.require(base + ".weight");
	if (t.shape.size() != 2)
		throw MediaError("upload_quant_linear_w4a4: '" + base + ".weight' is not a 2-D matrix");
	I4Upload out;
	out.n = t.shape[0];
	// `t.shape[1]` is the *packed* row width (K/2), which is what the caller reads
	// as the weight's K for the int4 GEMM (see `TeLinear32`), and what the device
	// allocation below is sized from. The logical K - twice it - is what the *values*
	// have to be decoded to, and the two must not be confused: before `logical_cols`
	// understood this family the decode produced K/2 int8 codes, i.e. a matrix of
	// half the width whose numbers were the packed bytes.
	const i64 packed_k = t.shape[1];
	const i64 logical_k = st.logical_cols(t);
	out.k = packed_k;
	if ((packed_k % 2) != 0 || (packed_k % 256) != 0)
		throw MediaError("qwen3vl: " + base + " has a packed width of " + std::to_string(packed_k) +
		                 ", not packable to convrot_w4a4");
	if (logical_k != packed_k * 2 || (logical_k % 256) != 0)
		throw MediaError("qwen3vl: " + base + " has K=" + std::to_string(logical_k) +
		                 ", not a convrot_w4a4 matrix (it must be twice the packed width)");
	out.w = arena.alloc((u64)out.n * (u64)packed_k);
	out.s = arena.alloc((u64)out.n * 4);
	if (lora && lora->touches(base)) {
		// The delta belongs to the *values*, so the weight is decoded to fp32 at its
		// logical K, adapted, and re-packed. Exactly the same maths as the no-LoRA
		// slab loop below, in one matrix because a dense delta cannot be slabbed.
		std::vector<float> w = dequant_weight_f32(st, base, out.n, logical_k);
		lora->apply(base, w.data(), out.n, logical_k);
		std::vector<int8_t> packed((size_t)out.n * (size_t)packed_k);
		std::vector<float> scales((size_t)out.n, 1.0f);
		quantize_convrot_w4a4(w.data(), out.n, logical_k, packed.data(), scales.data(), 256);
		upload_range(*g.ctx, *g.ring, out.w.res, out.w.off, packed.data(), packed.size());
		upload_range(*g.ctx, *g.ring, out.s.res, out.s.off, scales.data(), scales.size() * 4);
		return out;
	}
	const u64 slab = slab_rows(out.n, logical_k);
	std::vector<float> buf;
	for (i64 r0 = 0; r0 < out.n; r0 += (i64)slab) {
		const i64 rows = std::min<i64>((i64)slab, out.n - r0);
		buf.resize((size_t)(rows * logical_k));
		st.dequant_rows(t, r0, rows, buf.data());
		std::vector<int8_t> packed((size_t)rows * (size_t)packed_k);
		std::vector<float> scales((size_t)rows, 1.0f);
		quantize_convrot_w4a4(buf.data(), rows, logical_k, packed.data(), scales.data(), 256);
		upload_range(*g.ctx, *g.ring, out.w.res, out.w.off + (u64)r0 * (u64)packed_k,
		             packed.data(), packed.size());
		upload_range(*g.ctx, *g.ring, out.s.res, out.s.off + (u64)r0 * 4, scales.data(),
		             scales.size() * 4);
	}
	return out;
}

// ── the precision-preserving streaming upload + the one GEMM ────────────────

QuantLinear upload_linear_any(GpuCtx& g, const SafeTensors& st, const std::string& base,
                              GpuArena& arena, const LoraSet* lora) {
	const StTensor& t = st.require(base + ".weight");
	if (weight_is_dense_float(st, t)) {
		DenseUpload du = upload_dense_linear(g, st, base, arena);
		QuantLinear q;
		q.d = du.w;
		q.dtype = du.dtype;
		q.n = du.n;
		q.k = du.k;
		return q;
	}
	I8Upload up = upload_quant_linear_i8(g, st, base, arena, lora);
	QuantLinear q;
	q.w = up.w;
	q.scale = up.s;
	q.n = up.n;
	q.k = up.k;
	return q;
}

void linear_gemm(GpuCtx& g, const QuantLinear& w, const GpuAlloc& x, i64 m, i64 k,
                 const GpuAlloc& out, const GpuAlloc& q8, const GpuAlloc& s8) {
	if (w.k != k)
		throw MediaError("linear_gemm: K mismatch (weight " + std::to_string(w.k) + ", shape " +
		                 std::to_string(k) + ")");
	if (w.dense()) {
		// The dense float path: x is the fp32 activation, the weight is the file's
		// own f16 / bf16 / f32. No requantisation, no rotation.
		GemmF16Args ga;
		ga.a = x;
		ga.b = w.d;
		ga.c = out;
		ga.m = m;
		ga.n = w.n;
		ga.k = k;
		ga.a_is_f32 = true;
		ga.b_is_bf16 = (w.dtype == DType::BF16);
		ga.b_is_f32 = (w.dtype == DType::F32);
		dispatch_gemm_f16(*g.ctx, ga);
		return;
	}
	if (!w.w || !w.scale) throw MediaError("linear_gemm: the int8 weight is empty");
	if (!q8 || !s8) throw MediaError("linear_gemm: the int8 path needs activation scratch");
	QuantConvrotArgs qa;
	qa.x = x.res;
	qa.x_offset = x.off;
	qa.q = q8.res;
	qa.q_offset = q8.off;
	qa.s = s8.res;
	qa.s_offset = s8.off;
	qa.rows = m;
	qa.K = k;
	dispatch_quant_convrot(*g.ctx, qa);

	Int8GemmArgs ga;
	ga.a = q8.res;
	ga.a_offset = q8.off;
	ga.b = w.w.res;
	ga.b_offset = w.w.off;
	ga.sa = s8.res;
	ga.sa_offset = s8.off;
	ga.sb = w.scale.res;
	ga.sb_offset = w.scale.off;
	ga.c = out.res;
	ga.c_offset = out.off;
	ga.M = m;
	ga.N = w.n;
	ga.K = k;
	dispatch_int8_gemm(*g.ctx, ga);
}

QuantLinear load_quant3_linear(GpuCtx& g, const SafeTensors& st, const std::string& base_q,
                               const std::string& base_k, const std::string& base_v,
                               GpuArena* into, const LoraSet* lora) {
	auto alloc = [&](u64 bytes) { return into ? into->alloc(bytes) : g.walloc(bytes); };
	const char* bases[3] = {base_q.c_str(), base_k.c_str(), base_v.c_str()};
	const StTensor* ts[3] = {&st.require(base_q + ".weight"), &st.require(base_k + ".weight"),
	                         &st.require(base_v + ".weight")};
	const i64 k = ts[0]->shape[1];
	for (int i = 0; i < 3; i++) {
		if (ts[i]->shape.size() != 2 || ts[i]->shape[1] != k)
			throw MediaError(std::string("load_quant3_linear: shape mismatch at ") + bases[i]);
	}
	bool raw_int8 = true;
	bool touched = false;
	for (int i = 0; i < 3; i++) {
		raw_int8 = raw_int8 && st.is_plain_int8(*ts[i]);
		if (lora && lora->touches(bases[i])) touched = true;
	}

	QuantLinear q;
	q.k = k;
	q.n = ts[0]->shape[0] + ts[1]->shape[0] + ts[2]->shape[0];

	// A float source keeps its own precision (the dense q|k|v matrix).
	{
		bool any_dense = false, all_dense = true;
		for (int i = 0; i < 3; i++) {
			const bool d = weight_is_dense_float(st, *ts[i]);
			any_dense = any_dense || d;
			all_dense = all_dense && d;
		}
		if (any_dense) {
			if (!all_dense)
				throw MediaError("load_quant3_linear: q/k/v mix a float and a quantised precision");
			DenseUpload du = load_dense3_linear(g, st, base_q, base_k, base_v, into, lora);
			q.d = du.w;
			q.dtype = du.dtype;
			q.n = du.n;
			q.k = du.k;
			return q;
		}
	}

	if (raw_int8 && !touched) {
		// Concatenate the raw int8 rows and the scale vectors, exactly as the
		// pre-LoRA loaders did.
		u64 wbytes = 0;
		for (int i = 0; i < 3; i++) {
			wbytes += ts[i]->nbytes;
		}
		q.w = alloc(wbytes);
		u64 off = 0;
		for (int i = 0; i < 3; i++) {
			upload_range(*g.ctx, *g.ring, q.w.res, q.w.off + off, st.data_of(*ts[i]), ts[i]->nbytes);
			off += ts[i]->nbytes;
		}
		std::vector<float> scales;
		for (int i = 0; i < 3; i++) {
			std::vector<float> s = tensor_to_f32(st, st.require(std::string(bases[i]) + ".weight_scale"));
			if ((i64)s.size() != ts[i]->shape[0])
				throw MediaError(std::string("weight_scale length mismatch: ") + bases[i]);
			scales.insert(scales.end(), s.begin(), s.end());
		}
		q.scale = alloc((u64)scales.size() * 4);
		upload_range(*g.ctx, *g.ring, q.scale.res, q.scale.off, scales.data(), scales.size() * 4);
		return q;
	}

	QuantSpec spec = requant_spec_for(st, base_q);
	if (spec.convrot && (k % spec.convrot_groupsize) != 0)
		throw MediaError("load_quant3_linear: K=" + std::to_string(k) +
		                 " is not a multiple of the convrot group " +
		                 std::to_string(spec.convrot_groupsize));
	if (touched) {
		// A LoRA's delta is dense over the whole [n,k] plane, so it is the one case
		// that needs the fp32 matrix whole; fold and quantise once, as before. Rare
		// (the shipped image settings carry no image LoRA).
		std::vector<float> w;
		w.reserve((size_t)(q.n * k));
		for (int i = 0; i < 3; i++) {
			const i64 ni = ts[i]->shape[0];
			std::vector<float> wi = dequant_weight_f32(st, bases[i], ni, k);
			lora->apply(bases[i], wi.data(), ni, k);
			w.insert(w.end(), wi.begin(), wi.end());
		}
		QuantInt8 qi = quantize_weight(w.data(), q.n, k, spec);
		q.w = alloc((u64)qi.q.size());
		upload_range(*g.ctx, *g.ring, q.w.res, q.w.off, qi.q.data(), qi.q.size());
		q.scale = alloc((u64)qi.scale.size() * 4);
		upload_range(*g.ctx, *g.ring, q.scale.res, q.scale.off, qi.scale.data(), qi.scale.size() * 4);
		return q;
	}

	// Native path (f16 / bf16 / f32): decode each source a slab of rows at a
	// time straight into its slice of the concatenated [3N,K] matrix, so the fp32
	// form never exists whole in host RAM, and keep the finished int8 matrix in the
	// shared requantised-weight cache under one key covering the three. This is the
	// fused-qkv analogue of `load_quant_linear`'s slab loop: without it a streamed
	// DiT block re-decoded and re-rotated its qkv out of the file (through the old
	// whole-matrix `dequant_weight_f32`) on *every* sampling step.
	q.w = alloc((u64)q.n * (u64)k);
	q.scale = alloc((u64)q.n * 4);
	const std::string key = base_q + "|" + base_k + "|" + base_v + "|qkv";
	{
		const int8_t* cq = nullptr;
		const float* cs = nullptr;
		i64 crows = 0, ck = 0;
		if (requant_cache().get(&st, key, &crows, &ck, &cq, &cs) && crows == q.n && ck == k) {
			UploadBracket upload(g);
			upload_range(*g.ctx, *g.ring, q.w.res, q.w.off, cq, (u64)q.n * (u64)k);
			upload_range(*g.ctx, *g.ring, q.scale.res, q.scale.off, cs, (u64)q.n * 4);
			return q;
		}
	}
	const bool cacheable = host_cache_budget_bytes() > 0 &&
	                       (u64)q.n * (u64)k + (u64)q.n * 4 <= host_cache_budget_bytes();
	if (cacheable) {
		std::vector<int8_t> whole_q((size_t)(q.n * k));
		std::vector<float> whole_s((size_t)q.n);
		i64 row0 = 0;
		for (int i = 0; i < 3; i++) {
			const StTensor& ti = *ts[i];
			const i64 ni = ti.shape[0];
			requant_rows_into(st, ti, ni, k, spec, whole_q.data() + (size_t)row0 * (size_t)k,
			                  whole_s.data() + (size_t)row0);
			row0 += ni;
		}
		const u64 slab = slab_rows(q.n, k);
		{
			UploadBracket upload(g);
			for (i64 r0 = 0; r0 < q.n; r0 += (i64)slab) {
				const i64 rows = std::min<i64>((i64)slab, q.n - r0);
				upload_range(*g.ctx, *g.ring, q.w.res, q.w.off + (u64)r0 * (u64)k,
				             whole_q.data() + (size_t)r0 * (size_t)k, (u64)rows * (u64)k);
				upload_range(*g.ctx, *g.ring, q.scale.res, q.scale.off + (u64)r0 * 4,
				             whole_s.data() + (size_t)r0, (u64)rows * 4);
			}
		}
		requant_cache().put(&st, key, q.n, k, whole_q.data(), whole_q.size(), whole_s.data(),
		                    whole_s.size() * 4, host_cache_budget_bytes());
		return q;
	}
	{
		UploadBracket upload(g);
		std::vector<float> buf;
		i64 row0 = 0;
		for (int i = 0; i < 3; i++) {
			const StTensor& ti = *ts[i];
			const i64 ni = ti.shape[0];
			const u64 slab = slab_rows(ni, k);
			for (i64 r = 0; r < ni; r += (i64)slab) {
				const i64 rows = std::min<i64>((i64)slab, ni - r);
				buf.resize((size_t)(rows * k));
				st.dequant_rows(ti, r, rows, buf.data());
				QuantInt8 qi = quantize_weight(buf.data(), rows, k, spec);
				upload_range(*g.ctx, *g.ring, q.w.res, q.w.off + (u64)row0 * (u64)k, qi.q.data(),
				             qi.q.size());
				upload_range(*g.ctx, *g.ring, q.scale.res, q.scale.off + (u64)row0 * 4,
				             qi.scale.data(), qi.scale.size() * 4);
				row0 += rows;
			}
		}
	}
	return q;
}

// The spec to requantise a *decoded* weight with.
//
// The int8 GEMMs were written against the rotated basis: on disk a weight is
// `rot(W)` and the activation is rotated at run time, so the contraction comes out
// in the original basis. A checkpoint whose sidecar says `"convrot": true`
// (every shipped `*_int8_convrot`, `*_w4a8`, `*_w6a8`) therefore already stores
// exactly the bytes the kernels want, and the decode hands them back in that same
// basis - so requantising them must only rescale, not rotate. Rotating again is
// not a rounding difference: rot() is an involution, so the second application
// returns the matrix to its *original* basis, i.e. it produces a different
// matrix (energy-preserving, so it looks harmless, and then the model answers
// with noise). Only a source that is *not* rotated - a plain float checkpoint, or
// an int8 one without the flag - has to be rotated on the way in.
QuantSpec requant_spec_for(const SafeTensors& st, const std::string& base) {
	QuantSpec spec = default_int8_spec();
	const StTensor* t = st.find(base + ".weight");
	if (t) {
		const SafeTensors::WeightLayout L = st.weight_layout(t->name);
		if (L.spec.valid && L.spec.convrot) spec.convrot = false;
	}
	return spec;
}

// ── the runtime LoRA correction ────────────────────────────────────────────
//
// See the header for why a streamed module keeps its base weight and adds the
// delta to the GEMM's output. The two sizes worth keeping in mind here: `bt` is
// [n, rank] - one row of it per output row of the weight - and the correction's
// inner dimension is the rank, so both of its GEMMs are a couple of percent of
// the int8 GEMM they trail.
LoraTail load_lora_tail(GpuCtx& g, GpuArena& arena, const std::vector<LoraModule>& mods, i64 n,
                        i64 k) {
	LoraTail t;
	if (mods.empty() || n <= 0 || k <= 0) return t;

	// Every matching factor over every module, in module order.
	std::vector<std::pair<const LoraModule*, LoraSet::Delta>> found;
	i64 rank = 0;
	for (const LoraModule& m : mods) {
		if (!m.lora || m.n <= 0) continue;
		if (m.row0 < 0 || m.row0 + m.n > n)
			throw MediaError("LoRA correction: module " + m.base + " covers rows [" +
			                 std::to_string(m.row0) + ", " + std::to_string(m.row0 + m.n) +
			                 ") of a " + std::to_string(n) + "-row weight");
		std::vector<LoraSet::Delta> d;
		m.lora->deltas_for(m.base, m.n, k, &d);
		for (const LoraSet::Delta& f : d) {
			found.push_back({&m, f});
			rank += f.rank;
		}
	}
	if (rank <= 0) return t;
	// Refuse a chain the activation plans did not price. They reserve a
	// `kLoraRankBound`-wide scratch for this and cannot be asked anything else -
	// they run before a LoRA is loaded - so a wider one is refused here, with the
	// rank it asked for, instead of at the first correction with a message about
	// buffer sizes.
	if (rank > kLoraRankBound)
		throw MediaError("LoRA correction: the factors matching this module need a " +
		                 std::to_string(rank) + "-wide correction, past the " +
		                 std::to_string(kLoraRankBound) + " the activation plans reserve "
		                 "(kLoraRankBound)");

	// A stacked along the rank axis, B stacked the same way with each factor's
	// scale folded into its own columns, and each module's block placed at the
	// output rows it owns. The zero entries outside a factor's slice read A rows
	// that are zero too, so one rank-`rank` product is the sum of the factors'
	// products - i.e. exactly `apply()`'s order, in one GEMM.
	std::vector<float> a((size_t)rank * (size_t)k, 0.0f);
	std::vector<float> b((size_t)n * (size_t)rank, 0.0f);
	i64 r0 = 0;
	for (const auto& [mod, f] : found) {
		memcpy(&a[(size_t)r0 * (size_t)k], f.a, (size_t)f.rank * (size_t)k * 4);
		for (i64 o = 0; o < f.n; o++) {
			float* brow = &b[(size_t)(mod->row0 + o) * (size_t)rank + (size_t)r0];
			const float* src = &f.b[(size_t)o * (size_t)f.rank];
			for (i64 r = 0; r < f.rank; r++) brow[r] = f.scale * src[r];
		}
		r0 += f.rank;
	}

	t.at = arena.alloc((u64)a.size() * 4);
	t.bt = arena.alloc((u64)b.size() * 4);
	// Same bracket discipline as the other streamers: `load_block` calls this from
	// inside the bracket `new_layer()` opened, while the resident-window pass calls
	// it with none. Open one only when there is none, and close only our own - an
	// upload dropped for want of a recording context is a silent zero weight.
	{
		const bool own_bracket = !g.ctx->recording();
		if (own_bracket) g.ctx->begin();
		upload_range(*g.ctx, *g.ring, t.at.res, t.at.off, a.data(), a.size() * 4);
		upload_range(*g.ctx, *g.ring, t.bt.res, t.bt.off, b.data(), b.size() * 4);
		if (own_bracket) {
			g.ctx->submit();
			g.ring->rewind();
		}
	}
	t.rank = rank;
	t.n = n;
	t.k = k;
	t.active = true;
	return t;
}

LoraTail load_lora_tail(GpuCtx& g, GpuArena& arena, const std::string& base, const LoraSet* lora,
                        i64 n, i64 k) {
	if (!lora) return LoraTail{};
	return load_lora_tail(g, arena, {LoraModule{base, 0, n, lora}}, n, k);
}

void apply_lora_tail(GpuCtx& g, const LoraTail& t, const GpuAlloc& x, i64 m, const GpuAlloc& out,
                     const GpuAlloc& h) {
	if (!t.active || m <= 0) return;
	// The caller owns the [., rank] hidden form (its rank is a property of the
	// loaded LoRA files, not of the module). A short scratch here would be an
	// overwrite of whatever the arena put next, so it is checked rather than
	// trusted.
	const u64 need = (u64)m * (u64)t.rank * 4;
	if (h.bytes < need)
		throw MediaError("LoRA correction: the hidden-form scratch is " + std::to_string(h.bytes) +
		                 " bytes; a [" + std::to_string(m) + ", " + std::to_string(t.rank) +
		                 "] fp32 one needs " + std::to_string(need));

	GemmF16Args g1;
	g1.a = x;
	g1.b = t.at;
	g1.c = h;
	g1.m = m;
	g1.n = t.rank;
	g1.k = t.k;
	g1.a_is_f32 = true;
	g1.b_is_f32 = true;
	dispatch_gemm_f16(*g.ctx, g1);

	// `out += h @ B^T`, in one read-modify-write pass over `out`.
	LoraRankAddArgs ra;
	ra.h = h;
	ra.b = t.bt;
	ra.c = out;
	ra.m = m;
	ra.n = t.n;
	ra.rank = t.rank;
	dispatch_lora_rank_add(*g.ctx, ra);
}

}  // namespace phi::media
