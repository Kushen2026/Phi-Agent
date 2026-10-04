// safetensors reader: header parse + read-only file mapping.
//
// A 21 GB checkpoint must never be slurped into RAM: the engine maps the file
// and lets the OS page cache act as the shared second-level cache between the
// weight streamer and the disk. Tensor payloads are addressed by absolute file
// offset, so a block can be DMA'd to VRAM straight from the mapping.
#pragma once

#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "util/media_common.hpp"
#include "host/mtx.hpp"
#include "host/quant.hpp"

namespace phi::media {

// ── a host buffer the driver can DMA out of directly ──────────────────────
//
// The weight streamer moves ~17 GB per sampling step from host memory into the
// device. `upload_file_into` stages each piece into the upload ring - which is
// page-locked by construction, so the copy out of it is a real DMA - and issues
// the transfer from there. When the *source* buffer is page-locked as well that
// first copy is pure overhead, and the overhead is measurable: this box's
// `cache_serve` memcpy out of the pinned set alone reports 1.05 s of a 40 s
// step, and the look-ahead slot's share adds ~0.6 s.
//
// So the two host-side caches - the pinned file ranges and the look-ahead slot -
// can hold their bytes in `cuMemAllocHost` memory. The CUDA layer registers the
// allocator at context creation (`set_host_allocator`); without one this is
// plain malloc and nothing changes except that the streamer keeps using the
// ring. The cap matters: page-locked memory cannot be paged out, so only a
// bounded share of the machine's RAM is ever locked, and every allocation past
// the cap falls back to malloc.
class HostPin {
public:
	HostPin() = default;
	~HostPin();
	HostPin(const HostPin&) = delete;
	HostPin& operator=(const HostPin&) = delete;
	HostPin(HostPin&& o) noexcept;
	HostPin& operator=(HostPin&& o) noexcept;

	void reset();
	// At least `bytes`; the contents are undefined.
	// At least `bytes`; the contents are undefined. `headroom` is an amount the
	// caller wants kept *unlocked* for another cache - the pinned range set
	// leaves room for the look-ahead slot - and once the budget cannot cover
	// `bytes + headroom` the buffer falls back to pageable memory.
	void resize(u64 bytes, u64 headroom = 0);
	void* data() { return p_; }
	const void* data() const { return p_; }
	u64 size() const { return n_; }
	bool locked() const { return locked_; }

private:
	void* p_ = nullptr;
	u64 n_ = 0;
	bool locked_ = false;
};

struct StTensor {
	std::string name;
	DType dtype = DType::Unknown;
	std::vector<i64> shape;
	u64 offset = 0;  // absolute byte offset of the first element
	u64 nbytes = 0;
	i64 numel = 0;
	// For a *packed* quantised weight (4- or 6-bit codes) the header's last dim is
	// the packed width, not K. The reader rewrites `shape` to the logical K and
	// keeps the stored width and row stride here, so every consumer that reads
	// `shape[1]` as K - which is all of the model loaders - is right by
	// construction. Zero for an unpacked tensor.
	i64 stored_cols = 0;
	u64 row_bytes_stored = 0;
};

class SafeTensors {
public:
	SafeTensors() = default;
	~SafeTensors();
	SafeTensors(const SafeTensors&) = delete;
	SafeTensors& operator=(const SafeTensors&) = delete;

	// Opens and maps the file, parses the JSON header. Throws MediaError.
	void open(const std::string& path);
	void close();

	const std::string& path() const { return path_; }
	u64 file_size() const { return file_size_; }
	const std::vector<StTensor>& tensors() const { return tensors_; }
	// safetensors `__metadata__` (string -> string).
	const std::map<std::string, std::string>& metadata() const { return metadata_; }

	const StTensor* find(std::string_view name) const;
	// throws MediaError when the tensor is missing
	const StTensor& require(std::string_view name) const;

	// ── precision ──────────────────────────────────────────────────────────
	// True when the payload must be dequantised before any consumer can read it as
	// numbers (an int8 / int4 tensor).
	bool is_quantized(const StTensor& t) const;
	// True when `t` can be handed to a kernel verbatim (its `dtype` is the storage
	// dtype the consumers expect). int4 is not.
	bool is_raw(const StTensor& t) const;

	// ── row-wise access ────────────────────────────────────────────────────
	//
	// Safetensors stores rows densely, so a *row range* of a tensor is a contiguous
	// byte range of the payload. A consumer can therefore walk a matrix a slab of
	// rows at a time and convert each slab straight into the layout its kernel
	// reads, instead of materialising the whole tensor in host RAM first.
	u64 row_bytes(const StTensor& t) const;
	// Decodes `rows` rows starting at `row0` into fp32 `dst` (which must hold
	// rows * shape.back() floats). Handles every dtype: a raw float tensor is
	// converted, an int8 row is dequantised with its scale table, and the packed
	// families (asym_w4a8_int8, w6a8_int8, nvfp4, fp8) are unpacked by their own
	// decoders.
	void dequant_rows(const StTensor& t, i64 row0, i64 rows, float* dst) const;

	// The packed-weight path of `dequant_rows`: the logical K is `bits/8` wider
	// than the stored row, and the value comes from a codebook / six-bit code /
	// E2M1 code together with the per-row and per-group scales.
	void dequant_packed_rows(const StTensor& t, i64 row0, i64 rows, float* dst) const;

	// Rewrites the last dim of every packed quantised weight to its logical K (see
	// StTensor::stored_cols). Called once, from the header parse.
	void rewrite_packed_shapes();

	// Refuses a six-bit payload that does not carry a code distribution (see the
	// definition): the shipped w6a8 files are like that.
	void validate_packed6_payload(const StTensor& t) const;

	// Dequantise `t` into `target` (F32/F16/BF16) or, for a raw tensor, convert it.
	// Returns the host buffer; `nbytes` in the returned vector is
	// `numel * dtype_size(target)`. Throws MediaError for a layout it cannot read.
	std::vector<u8> materialize(const StTensor& t, DType target) const;
	std::vector<u8> materialize(std::string_view name, DType target) const {
		return materialize(require(name), target);
	}

	// A *stable* host pointer to the tensor's values in `target` dtype, valid for
	// the lifetime of the reader. A raw tensor already in `target` points straight
	// into the mapping; anything else (f16/bf16) is materialised once
	// and kept alive here. This is the accessor the host-side fp32 reference
	// weights several models hold for the whole run use.
	const void* data_as(const StTensor& t, DType target) const;
	const float* data_f32(const StTensor& t) const {
		return (const float*)data_as(t, DType::F32);
	}

	// Pointer into the mapping (valid until close()).
	const void* data_of(const StTensor& t) const {
		return (const u8*)mapping_ + t.offset;
	}
	const void* data_of(std::string_view name) const { return data_of(require(name)); }

	// comfy_quant sidecar for a weight tensor. The metadata is stored under the
	// layer name with the ".weight" suffix replaced by ".comfy_quant" (e.g.
	// "…attention.qkv.weight" -> "…attention.qkv.comfy_quant"). Returns an
	// invalid spec when the layer is not quantised.
	QuantSpec quant_spec_of(std::string_view weight_name) const;

	// ── quantised weight layouts ────────────────────────────────────────────
	//
	// What a weight actually is on disk. The family comes from the comfy_quant
	// blob and the scale/codebook tensors are found by name beside it, so a new
	// family is a new case here plus a decoder in quant.cpp - no consumer of
	// `dequant_rows` (which is every model in the engine) has to know.
	struct WeightLayout {
		QuantSpec spec;
		int bits = 8;                            // bits per weight on disk
		i64 group = 0;                           // width of the relative-scale groups
		const StTensor* row_scale = nullptr;     // weight_scale / weight_s_channel
		const StTensor* rel_scale = nullptr;     // weight_s_rel
		const StTensor* codebook = nullptr;      // weight_codebook
		const StTensor* global_scale = nullptr;  // weight_scale_2 (nvfp4)
		bool packed = false;                     // bits < 8: the last dim is not K
		bool known_family = false;               // one of the families the engine decodes
	};
	WeightLayout weight_layout(std::string_view weight_name) const;

	// The signed int4 nibble family (`convrot_w4a4`): two signed nibbles per byte
	// along K, one fp32 scale per output row, `shape` still holding the packed
	// width. Its own entry point because it is neither a `packed` family above
	// (logical_cols would then disagree with the consumers) nor a plain int8 one.
	void dequant_int4_rows(const StTensor& t, i64 row0, i64 rows, float* dst,
	                       const WeightLayout& layout) const;

	// Logical (unpacked) K of a weight: the stored last dim times 8/bits.
	i64 logical_cols(const StTensor& t) const;

	// The original int8 tensorwise form: one code per element, one fp32 scale per
	// row, no relative scale. Only this one may be uploaded to the int8 GEMM
	// verbatim; everything else must go through `dequant_rows` first.
	bool is_plain_int8(const StTensor& t) const;

	// One scale value for one row (scalar tables broadcast).
	float row_scale_value(const StTensor& scale, i64 row) const;

	// Copy [offset, offset+len) out of the mapping (sequential read; the OS
	// faults the pages in). Used by the block streamer.
	void read(u64 offset, size_t len, void* dst) const;
	// This is not a stylistic variant of read(): a cold read of a *mapping* costs
	// one 4 KB page fault per page, which measures ~0.6-0.9 GB/s on this box's
	// NVMe, while ReadFile at 4 MB granularity measures ~2.1 GB/s. The DiT's
	// scheduler moves ~16 GB of weights per sampling step and the file is larger
	// than RAM, so those pages are always cold: the choice of path is the
	// difference between a ~19 s step and a ~8 s one. Called under the I/O mutex,
	// so a worker thread may use it while another reads a different range.
	//
	// Ranges pinned by `pin_range` are served out of system memory instead.
	void read_file(u64 offset, size_t len, void* dst) const;

	// ── host-side pinned range cache ───────────────────────────────────────
	//
	// `pin_range` reads [offset, offset+len) into system memory once and serves
	// every later `read_file` that falls inside it from RAM. This is the only
	// lever that moves the DiT's sampling step at all, and the measurements say
	// why: the 19.5 GB checkpoint is re-read on every step (9 of its 50 blocks fit
	// in 6 GB of VRAM), and this NVMe delivers ~1.2 GB/s for that access pattern
	// *no matter how it is driven* - 1.13 GB/s for a whole-file sequential read,
	// 1.25 GB/s for the engine's own 12-scattered-reads-per-block pattern, 1.00-
	// 1.18 GB/s when a block's tensors are merged into one contiguous read, and
	// 1.02-1.24 GB/s with 2, 3, 4, 6 or 8 concurrent reader threads. Read size,
	// read granularity, queue depth and overlap are all already at the device's
	// limit; the only remaining way to shorten a step is to stop asking the disk
	// for bytes it has already handed over. System memory moves them ~10x faster.
	//
	// The cache is a set of *pinned* ranges rather than an LRU, and that is a
	// deliberate policy: the streamer walks a cyclic working set (15 GB of block
	// weights per step, the same 15 GB every step) that is larger than any sane
	// cache, and an LRU scores zero on that pattern because every entry is evicted
	// by the time the loop comes back to it. Pinning a fixed subset and streaming
	// the rest is the policy that actually hits.
	void pin_range(u64 offset, u64 len);
	// How many bytes the pinned set holds.
	u64 pinned_bytes() const;
	// Host time spent copying out of the pinned set, so "the cache hit" can be
	// told apart from "the cache hit was free". The two are what the
	// `PHI_DIT_PLAN` diagnostics print; a served-bytes count with no time next to
	// it cannot tell a fast cache from a slow one.
	double cache_ms() const { return cache_ms_; }
	u64 cache_served_bytes() const { return cache_served_bytes_; }
	void reset_cache_stats() {
		cache_ms_ = 0;
		cache_served_bytes_ = 0;
	}
	void drop_cache();
	// System memory that is free right now, for a caller deciding how much to pin.
	// 0 when the query is unavailable.
	static u64 host_free_bytes();
	// Installed physical memory. Stable across a run, unlike the free count, which a
	// checkpoint read through the mapping drives down as it fills the page cache.
	static u64 host_total_bytes();

	// Bytes/milliseconds moved by read_file so far, for the diagnostics that have
	// to prove which path the streamer actually took (`PHI_DIT_PLAN`).
	u64 file_read_bytes() const { return file_read_bytes_; }
	double file_read_ms() const { return file_read_ms_; }
	void reset_file_read_stats() const { file_read_bytes_ = 0; file_read_ms_ = 0; }

	// Serialises the seek-free OVERLAPPED reads below (one shared handle, many
	// callers).
	static std::mutex& io_mutex();

	// Hint the OS that [offset, len) will be read soon (prefetch + keep).
	void prefetch(u64 offset, size_t len) const;

	// Warn the OS to drop the pages for [offset, len) — called after a weight
	// block has been consumed so the streaming window keeps the cache hot for
	// the *next* block rather than caching a whole 21 GB file.
	void release_pages(u64 offset, size_t len) const;

	// ── one-block look-ahead slot ──────────────────────────────────────────
	//
	// The streaming loop is a strict alternation of "read the next block off the
	// disk" and "compute on the block just read", and on this box the two are
	// the same size (a block is 368 MB = ~0.3 s of disk at 1.2 GB/s, a block's
	// forward is ~0.8 s of GPU). Running them one after the other therefore
	// spends a third of every sampling step waiting on the disk for bytes the
	// GPU is not yet able to use.
	//
	// `prefetch_ranges` starts a *worker thread* that reads the given file
	// ranges into a slot owned by this object; the sampling loop issues it for
	// block i+1 the moment block i's weights have been copied to the device, so
	// the read overlaps block i's forward. `read_file` serves anything that
	// falls inside the slot, so the loader itself does not change: it just stops
	// paying for the disk.
	//
	// One slot, replaced wholesale: the working set is a cyclic 15 GB walk that
	// no cache fits (see `pin_range`), and what the loop needs is not history but
	// *the next 368 MB*. The ranges are stored with the byte offset they land at,
	// so a `read_file` that does not fall inside the current set falls through to
	// the disk exactly as before.
	void prefetch_ranges(const std::vector<std::pair<u64, u64>>& ranges);   // (file offset, bytes)

	// ── the direct-DMA query ──────────────────────────────────────────────
	//
	// A *page-locked* pointer covering [offset, offset+len), taken from either
	// host cache, or null. When it returns non-null the streamer can hand the
	// address straight to `cuMemcpyHtoD` and skip the ring - which is the whole
	// point of holding these bytes in locked memory. A consumer that copies
	// instead of DMA'ing keeps working either way; this only says "you may".
	const void* pinned_dma_at(u64 offset, size_t len) const;
	// The page-locked bytes currently held, for the diagnostics.
	u64 host_locked_bytes() const;

	// The CUDA layer's page-locked allocator. Registered once, before any cache
	// exists; `lock_cap` bounds the total this process will lock.
	using HostAllocFn = void* (*)(size_t);
	using HostFreeFn = void (*)(void*);
	static void set_host_allocator(HostAllocFn alloc, HostFreeFn free, u64 lock_cap);
	// Join the worker (idempotent). Call before the first `read_file` that is
	// meant to hit the slot, and before destroying the object.
	void prefetch_wait() const;
	// Whether the slot currently holds a completed range set.
	bool prefetch_ready() const;
	void prefetch_clear();

	std::string describe() const;

private:
	// One entry per prefetched range: where it is in the file, where it landed in
	// the slot, and how many bytes it is (so a `read_file` can be tested against
	// it without consulting the next entry).
	struct PfRange {
		u64 file_off = 0;
		u64 dest = 0;
		u64 len = 0;
	};
	void prefetch_worker(std::vector<PfRange> ranges, u64 total);
	// The raw disk read, with neither the pinned set nor the slot consulted.
	void read_file_direct(u64 offset, size_t len, void* dst) const;

	void parse_header();
	// Serves [offset, offset+len) out of the pinned set when it is fully covered.
	bool cache_serve(u64 offset, size_t len, void* dst) const;
	// Serves [offset, offset+len) out of the look-ahead slot when it is covered.
	bool prefetch_serve(u64 offset, size_t len, void* dst) const;
	void prefetch_reset_locked();

	std::string path_;
	u64 file_size_ = 0;
	void* file_handle_ = nullptr;  // HANDLE, FILE_FLAG_RANDOM_ACCESS (mapping's image)
	// Opened lazily by the first read_file(); mutable so the const readers can
	// create it (the class is otherwise a read-only view of an immutable file).
	mutable void* seq_handle_ = nullptr;
	void* mapping_ = nullptr;      // LPVOID base of the mapped view
	std::vector<StTensor> tensors_;
	std::map<std::string, size_t> by_name_;
	std::map<std::string, std::string> metadata_;
	// Lazily materialised host tensors (see data_as).
	mutable std::map<std::string, std::vector<u8>> host_cache_;
	// read_file is called from the streaming loop and (potentially) from a
	// prefetch thread; the stats are mutable so a const reader can account for
	// its own I/O.
	mutable u64 file_read_bytes_ = 0;
	mutable double file_read_ms_ = 0;

	// The pinned set: byte ranges held in system memory, keyed by file offset.
	// `cache_mutex_` guards the map and the counters; nothing else is held while it
	// is taken, and no lock is ever taken *while* holding it, so the I/O mutex and
	// this one can never deadlock against each other.
	mutable std::mutex cache_mutex_;
	mutable std::map<u64, HostPin> cache_;
	mutable u64 pinned_bytes_ = 0;
	mutable double cache_ms_ = 0;
	mutable u64 cache_served_bytes_ = 0;

	// The look-ahead slot (see `prefetch_ranges`). `pf_mutex_` guards the range
	// table and `pf_state_`; the worker writes `pf_buf_` and only flips
	// `pf_state_` to Ready when the whole set has landed, so a reader that sees
	// Ready knows every byte it can match is valid. `pf_join_mutex_` serialises
	// starting/joining the worker itself.
	mutable std::mutex pf_join_mutex_;
	mutable std::mutex pf_mutex_;
	mutable std::thread pf_thread_;
	mutable bool pf_joinable_ = false;
	mutable HostPin pf_buf_;
	mutable std::vector<PfRange> pf_ranges_;
	mutable u64 pf_total_ = 0;
	// 0 = empty, 1 = reading, 2 = ready.
	mutable int pf_state_ = 0;
};

// ── the file's own precision, and the path it implies ───────────────────────
//
// Goal #1: a loader must be able to read *any* precision the safetensors file
// stores (gguf is refused - see SafeTensors::open) and must not convert it. The
// mapping below is the whole policy in one place, and it lives here - next to
// the layout knowledge that answers it - rather than in a model loader, because
// every loader (main model, LoRA, text encoder, VAE) has to make the same
// decision about the same tensor:
//
//   int8_tensorwise       -> Int8Tensorwise : the verbatim int8 tensor-core path
//   asym_w4a8_int8        -> W4A8           : packed, expanded on device
//   w6a8_int8             -> W6A8           : packed, expanded on device
//   convrot_w4a4          -> W4A4           : packed signed int4 (own loader)
//   nvfp4 / mxfp4         -> Nvfp4          : packed E2M1, decoded to f16
//   f16 / bf16 / f32      -> F16/Bf16/Fp32 : the dense fp16/bf16/f32 GEMM
//   fp8 (E4M3 / E5M2)     -> Fp8            : decoded to f16 (exact), dense GEMM
//
// `Other` is a layout no path claims (an I4 tensor with no sidecar, say): the
// loaders refuse it by name instead of guessing a width.
enum class WeightPrecision {
	Int8Tensorwise,
	W4A8,
	W6A8,
	W4A4,
	Nvfp4,
	F16,
	Bf16,
	F32,
	Fp8,
	Other,
};

// The precision of a weight in a file, and the two questions every loader asks it:
// which family is this, and does the dense float GEMM read it directly. Free
// functions over the `SafeTensors` members, because a caller that already has the
// reader in hand should not have to spell the class to ask.
WeightPrecision weight_precision_of(const SafeTensors& st, const StTensor& t);
const char* weight_precision_name(WeightPrecision p);
bool weight_is_dense_float(const SafeTensors& st, const StTensor& t);

}  // namespace phi::media
