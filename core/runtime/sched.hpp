// GPU memory + weight-streaming layer.
//
// A bump allocator that never hands back committed resources mid-run, and a
// staging ring so uploading a layer's weights is one memcpy + one device copy
// instead of an allocation per tensor.
//
// The structure is what the engine is written against: a `GpuBuffer` is a plain
// device address rather than a COM resource, and the "ring" is pinned host
// memory the driver DMAs straight out of, so `stage()` is a memcpy into memory
// the GPU can read without a bounce buffer.
#pragma once

#include "runtime/compute.hpp"
#include "runtime/cuda_device.hpp"
#include "util/media_common.hpp"

namespace phi::media {

// Forward declarations for `upload_file_range` below: the safetensors reader lives
// in media/host/st.hpp, which includes this header's siblings, so naming the two
// types here keeps the runtime layer free of a host-layer include.
class SafeTensors;
struct StTensor;

// A sub-range of one arena chunk. Every caller in the engine is written against
// `res` + `off`.
struct GpuAlloc {
	GpuBuffer* res = nullptr;
	u64 off = 0;
	u64 bytes = 0;
	explicit operator bool() const { return res != nullptr; }
};

// Bump allocator over large chunks. `reset()` invalidates every allocation made
// since the previous reset but keeps the chunks, so a sampling loop stabilises
// after the first step.
//
// `mark()` / `rewind_to()` are the fine-grained version: a stage that needs
// scratch for one operation marks before it allocates and rewinds afterwards,
// which turns the arena high-water mark from the *sum* of the stages into their
// *maximum*.
class GpuArena {
public:
	struct Mark {
		size_t chunk = 0;
		u64 off = 0;
		u64 used = 0;
	};
	~GpuArena();
	// Copying an arena is never right: the copy would hand out memory the original
	// does not account for (its `used_`/`capacity_` are what the VRAM accountant
	// reads), and its teardown would look at chunks the original still owns. A
	// caller that wants to allocate into an arena passes a reference.
	GpuArena(const GpuArena&) = delete;
	GpuArena& operator=(const GpuArena&) = delete;
	GpuArena() = default;
	void init(ComputeContext* ctx, u64 chunk_bytes = 192ull << 20);
	// Changes the size of *future* chunks. Existing chunks are untouched (they are
	// committed device memory and may still be live), so this is safe to call
	// between phases - which is what the media tools do when the budget they were
	// planned against moved between two calls. 0 is ignored.
	void set_chunk_bytes(u64 chunk_bytes) {
		if (chunk_bytes) chunk_bytes_ = chunk_bytes;
	}
	u64 chunk_bytes() const { return chunk_bytes_; }
	void destroy();
	// Returns the chunks to the device and the charge to the accountant. Only
	// legal when nothing in the arena is live.
	void release_chunks();
	GpuAlloc alloc(u64 bytes, u64 align = 256);
	void reset();
	u64 used() const { return used_; }
	u64 capacity() const { return capacity_; }
	int chunks() const { return (int)chunks_.size(); }
	void rewind_to(const Mark& m);
	Mark mark() const { return Mark{cur_, cur_off_, used_}; }
	void set_tag(const char* t) { tag_ = (t && *t) ? t : "gpu_arena"; }
	const char* tag() const { return tag_; }

private:
	ComputeContext* ctx_ = nullptr;
	std::vector<GpuBuffer*> chunks_;
	std::vector<u64> chunk_size_;
	size_t cur_ = 0;
	u64 cur_off_ = 0;
	u64 used_ = 0;
	u64 capacity_ = 0;
	u64 chunk_bytes_ = 0;
	const char* tag_ = "gpu_arena";
};

// RAII form of mark()/rewind_to().
class ArenaScope {
public:
	explicit ArenaScope(GpuArena& a) : arena_(&a), mark_(a.mark()) {}
	~ArenaScope() { pop(); }
	ArenaScope(const ArenaScope&) = delete;
	ArenaScope& operator=(const ArenaScope&) = delete;
	void pop() {
		if (arena_) {
			arena_->rewind_to(mark_);
			arena_ = nullptr;
		}
	}

private:
	GpuArena* arena_ = nullptr;
	GpuArena::Mark mark_{};
};

// Pinned host memory used as a ring. Callers stage host bytes here and the
// driver copies straight out of it, so a weight upload costs one memcpy (host)
// plus one DMA (device) rather than a bounce through pageable memory. Rewound
// only once the queue has drained, which the engine guarantees by submitting
// synchronously - either the caller submits itself (the streaming helpers do)
// or stage() drains when it runs out of room (see below).
//
// The ring is a *shared* resource: several callers stage into it inside one open
// bracket (the text encoder stages a whole 244 MB layer as ~15 uploads, the DiT
// streams one block per submit), and stage()/reserve() can only be sized for one
// of them. So the rule is:
//   * a tensor that fits an *empty* ring is always staged - if the space left is
//     too small the ring submits what is recorded, waits for the GPU and rewinds
//     ("drains") before handing out the space;
//   * a tensor larger than the whole ring is a hard error, because no amount of
//     draining can make room for it.
class UploadRing {
public:
	~UploadRing();
	void init(ComputeContext* ctx, u64 bytes = 64ull << 20);
	// Re-creates the ring at a new size, releasing the old pinned buffer through
	// the context that owns it (a plain init() would leave it to the context's
	// teardown - these are hundreds of megabytes of pinned host memory).
	void reinit(u64 bytes);
	void destroy();
	// Returns the byte offset inside the ring; drains the queue when full.
	u64 stage(const void* src, u64 bytes, u64 align = 256);
	// Reserves `bytes` and hands back the mapped host pointer for that region, so
	// a caller can fill it in place (a file read, a decode). Same lifetime rules
	// as stage().
	//
	// The pointer is valid until the next staging call: a drain invalidates every
	// region handed out before it, so fill it and copy it out (upload_range_staged)
	// before staging anything else.
	void* reserve(u64 bytes, u64* out_off, u64 align = 256);
	void rewind();
	u64 size() const { return size_; }
	u64 free() const { return size_ - head_; }
	GpuBuffer* buffer() const { return res_; }
private:
	// Submit what is recorded and rewind, leaving the caller's bracket exactly as
	// it was (a caller that had one open still has one open).
	void drain();
	// The offset the next `bytes`-sized staging can start at, draining if needed.
	// Throws only when the tensor cannot fit an empty ring (checked by the caller).
	u64 ensure_room(u64 bytes, u64 align);

	ComputeContext* ctx_ = nullptr;
	GpuBuffer* res_ = nullptr;
	void* mapped_ = nullptr;
	u64 size_ = 0;
	u64 head_ = 0;
};

// The size the upload ring is created with, in one place because the tool
// (tools_media.cpp) and the acceptance bench (media_bench.cpp) must agree on it:
// the ring's size is also the granularity the weight streamers move per submit,
// so a bench that sizes it differently is not measuring what the tool runs.
//
// Derived from what the card can hold (an eighth of the budget, clamped to
// [128 MB, 2 GB]) and *not* from the accountant's limit alone: that limit is 0
// until something asks the driver for its budget, and reading "unset" as "tiny"
// gave the smallest ring to the largest models - which is how a 244 MB text
// encoder layer ended up streaming through a 128 MB ring.
// The upper bound scales with the card now: the old flat 1 GB cap left a 24 GB
// box streaming 20 GB checkpoints through the same pipe as a 6 GB one.
// PHI_RING_MB pins it, so the small-ring paths can be exercised on purpose.
u64 default_upload_ring_bytes();

// The same derivation for an explicit budget, so a caller that has just refreshed
// the accountant's view of the card can re-derive the ring from *that* number
// (`refresh_engine_run` in tools_media.cpp does, on every tool invocation).
u64 upload_ring_bytes_for(u64 limit_bytes);

// Accounting granularity of the weight/activation arenas: how much a chunk
// reserves the moment one allocation in it wants space. Small granularity means
// little rounding waste per arena, large means fewer allocations on a big card.
// 64 MB on the reference 6 GB card (the value the accounting was tuned for),
// growing to 256 MB at 20 GB and above, so the number of chunks an activation
// tensor needs does not grow with the card.
u64 arena_chunk_bytes_for(u64 limit_bytes);

// Same, for the small `keep` arena (embeddings, heads, tables): it holds hundreds
// of small tensors that are never rewound, so it wants finer granularity.
u64 keep_chunk_bytes_for(u64 limit_bytes);

// What the accountant will actually be charged for allocating `sizes` (bytes, in
// that order) through a GpuArena whose chunk granularity is `chunk_bytes`.
//
// This exists because the charge is per *chunk*, not per byte: a 386 MB DiT block
// whose tensors are 220 / 73 / 294 / 147 MB is booked as the *sum of the chunks*
// the bump allocator needs for it, which on a 64 MB granularity is 594 MB - 60 %
// more than the block's own weight. A residency plan that reserves "one block"
// therefore under-reserves by that much, sits at ~100 % of the budget, and turns
// any squeeze (another process claiming VRAM, the image chain's areans in the
// same process) into an `image.weights` refusal in the middle of a run.
//
// The simulation mirrors `GpuArena::alloc` step for step, including the
// chunk-list reuse across `reset()` and the release-on-grow when the arena is
// rewound to its start, and iterates to the fixed point a sampling loop reaches
// after its first block.
u64 arena_chunk_cost(const std::vector<u64>& sizes, u64 chunk_bytes);

// The same accounting as a state machine over a *sequence* of allocations, for
// plans whose arena is not one flat list: the H3 DiT allocates a persistent head
// (the packed stream, the rope and position tables), then a scratch block under a
// mark (the token refiner's), rewinds to that mark, and allocates the block
// loop's own buffers into the same region.
//
// `arena_chunk_cost` cannot price that: the two scratch sets are never live at
// once, and a flat sum would charge the run for both - which is exactly the kind
// of over-reservation that leaves a plan with no room for a resident block. The
// peak of the simulated arena over the whole sequence is what the accountant will
// really book, so a plan that uses this prices its activation frame exactly
// instead of estimating it with a safety margin and then reserving a second,
// separate term for "everything the estimate missed".
class GpuArenaModel {
public:
	explicit GpuArenaModel(u64 chunk_bytes);
	void alloc(u64 bytes, u64 align = 256);
	// `GpuArena::reset`: rewind the bump pointer, keep the chunks (and their
	// charge - that is the point of a bump allocator in a sampling loop).
	void reset();
	struct Mark {
		size_t chunk = 0;
		u64 off = 0;
		u64 used = 0;
	};
	Mark mark() const { return Mark{cur_, cur_off_, used_}; }
	void rewind_to(const Mark& m);
	u64 capacity() const { return capacity_; }
	u64 used() const { return used_; }

private:
	u64 chunk_bytes_ = 64ull << 20;
	std::vector<u64> chunks_;
	size_t cur_ = 0;
	u64 cur_off_ = 0;
	u64 used_ = 0;
	u64 capacity_ = 0;
};

// Upload `bytes` from host memory into (res, off) using the ring. Must be called
// between begin() and submit().
void upload_range(ComputeContext& ctx, UploadRing& ring, GpuBuffer* res, u64 off, const void* src,
                  u64 bytes);

// The copy half of upload_range for a region the caller already staged with
// reserve(): no host copy, just the device-to-device move.
void upload_range_staged(ComputeContext& ctx, UploadRing& ring, GpuBuffer* res, u64 off,
                         u64 ring_off, u64 bytes);

// The file-backed form of `upload_range`: `upload_range` copies from host memory
// the caller already holds, this one reads the tensor straight out of the
// checkpoint with `SafeTensors::read_file` (ReadFile at a 4 MB granule) and stages
// it in the caller's ring on the way through.
//
// This is not a stylistic variant of reading `data_of()` and passing the pointer:
// a cold read through the mapping costs one 4 KB page fault per page and measures
// ~0.74 GB/s on this box's NVMe, while the ReadFile path measures 2.1 GB/s cold
// and ~11 GB/s once the page cache holds the range. Same bytes either way, so the
// choice only moves time - see `SafeTensors::read_file` for the measurements.
//
// The tensor is staged in the ring in one piece, because these are the tens of
// megabytes a layer/block loader asks for (the 244 MB tables that need chunking go
// through `upload_into`). A tensor larger than the ring is a hard error, exactly as
// in `upload_range`. Must be called between begin() and submit().
void upload_file_range(ComputeContext& ctx, UploadRing& ring, GpuBuffer* res, u64 off,
                       const SafeTensors& st, const StTensor& t);

}  // namespace phi::media
