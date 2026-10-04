// GPU memory + weight-streaming implementation (CUDA).
#include "runtime/sched.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "host/st.hpp"
#include "runtime/vram_budget.hpp"

namespace phi::media {

// ── GpuArena ───────────────────────────────────────────────────────────────

GpuArena::~GpuArena() { destroy(); }

void GpuArena::init(ComputeContext* ctx, u64 chunk_bytes) {
	if (ctx_ == ctx && !chunks_.empty()) return;
	destroy();
	ctx_ = ctx;
	chunk_bytes_ = chunk_bytes;
}

void GpuArena::release_chunks() {
	// Drop the chunk list: the device memory (owned by ComputeContext) and the
	// accountant's charge for it. Only legal when nothing in this arena is live.
	// The drain is only about not rewinding memory the GPU is still reading -
	// which the caller's own submit() has already guaranteed, since CUDA has no
	// open command lists to invalidate.
	if (ctx_ && !chunks_.empty()) {
		ctx_->submit_if_recording();
		for (GpuBuffer* r : chunks_) ctx_->release_buffer(r);
		vram_budget().release(capacity_);
		chunks_.clear();
		chunk_size_.clear();
		cur_ = 0;
		cur_off_ = 0;
		used_ = 0;
		capacity_ = 0;
		return;
	}
	vram_budget().release(capacity_);
	chunks_.clear();
	chunk_size_.clear();
	cur_ = 0;
	cur_off_ = 0;
	used_ = 0;
	capacity_ = 0;
}

void GpuArena::destroy() {
	// Teardown, not "give the memory back". The chunks are owned by the
	// ComputeContext that created them and are freed in its destroy(); dropping
	// the list here is what stops this arena being part of the working set, and
	// the accountant is released with it.
	vram_budget().release(capacity_);
	chunks_.clear();
	chunk_size_.clear();
	cur_ = 0;
	cur_off_ = 0;
	used_ = 0;
	capacity_ = 0;
}

GpuAlloc GpuArena::alloc(u64 bytes, u64 align) {
	if (bytes == 0) return {};
	if (!ctx_) throw MediaError("GpuArena: not initialised");
	u64 need = round_up((i64)bytes, (i64)align);
	const bool fully_rewound = (cur_ == 0 && cur_off_ == 0);
	for (;;) {
		if (cur_ < chunks_.size()) {
			u64 left = chunk_size_[cur_] - cur_off_;
			if (left >= need) {
				GpuAlloc a{chunks_[cur_], cur_off_, bytes};
				cur_off_ += need;
				used_ += need;
				return a;
			}
		}
		if (cur_ < chunks_.size()) {
			cur_++;
			cur_off_ = 0;
			continue;
		}
		// Grow. The charge happens *before* the allocation and throws if it would
		// cross the budget, so the arena can never commit past the cap - that is
		// the "hard error, no silent degrade" rule.
		//
		// A bump allocator can only hand out the *sum* of its chunks, but a
		// mark/rewind caller only needs their maximum. When we are about to grow
		// at the very start of the arena, every allocation in it has already been
		// rewound, so the existing chunks are pure overhead from sizes that were
		// each needed once and can be dropped in favour of one chunk that fits
		// what is being asked for now.
		u64 want = std::max(chunk_bytes_, need);
		if (getenv("PHI_ARENA_TRACE")) {
			fprintf(stderr, "[arena %s] grow need=%llu cur=%zu/%llu rewound=%d chunks=%zu\n",
			        tag_, (unsigned long long)need, cur_, (unsigned long long)cur_off_,
			        (int)fully_rewound, chunks_.size());
		}
		if (fully_rewound && !chunks_.empty()) release_chunks();
		vram_budget().charge(want, tag_);
		GpuBuffer* res = ctx_->create_device_buffer(want, /*uav=*/true);
		chunks_.push_back(res);
		chunk_size_.push_back(want);
		capacity_ += want;
		cur_off_ = 0;
	}
}

void GpuArena::reset() {
	cur_ = 0;
	cur_off_ = 0;
	used_ = 0;
}

void GpuArena::rewind_to(const Mark& m) {
	if (m.chunk > chunks_.size()) return;   // stale mark; nothing to undo
	cur_ = m.chunk;
	cur_off_ = m.off;
	used_ = m.used;
}

// ── UploadRing ─────────────────────────────────────────────────────────────

// One derivation for the tool, the bench and the tests - see the header for why
// "the accountant's limit is 0" must not be read as "a tiny ring".
u64 upload_ring_bytes_for(u64 limit_bytes) {
	if (const char* e = getenv("PHI_RING_MB")) {
		const long v = strtol(e, nullptr, 10);
		if (v >= 1) return (u64)v << 20;
	}
	u64 base = limit_bytes;
	// 0 means "no device has reported a budget yet" (only CudaContext::query_budget()
	// sets it), not "no memory": fall back to the ceiling the owner authorised for
	// this card instead of letting the floor below become the answer.
	if (base == 0) base = vram_budget().hard_cap();
	u64 ring = base / 8;
	if (ring < (128ull << 20)) ring = 128ull << 20;
	if (ring > (2048ull << 20)) ring = 2048ull << 20;
	return (ring + (16ull << 20) - 1) & ~((16ull << 20) - 1);
}

u64 default_upload_ring_bytes() { return upload_ring_bytes_for(vram_budget().limit()); }

// The arena granularity. Both bounds are real: below 64 MB an arena that holds
// one 181 MB DiT block in 48 MB pieces charges for four chunks where two would
// do, and above 256 MB an arena that only ever wants 40 MB (the `keep` arena, a
// VAE tile) over-charges the accountant by hundreds of megabytes before its
// first dispatch runs. In between, the granularity scales with the card so the
// number of chunks holding a fixed-size activation tensor does not grow with it.
u64 arena_chunk_bytes_for(u64 limit_bytes) {
	u64 base = limit_bytes ? limit_bytes : vram_budget().hard_cap();
	u64 chunk = base / 80;
	if (chunk < (64ull << 20)) chunk = 64ull << 20;
	if (chunk > (256ull << 20)) chunk = 256ull << 20;
	return (chunk + (16ull << 20) - 1) & ~((16ull << 20) - 1);
}

u64 keep_chunk_bytes_for(u64 limit_bytes) {
	u64 base = limit_bytes ? limit_bytes : vram_budget().hard_cap();
	u64 chunk = base / 512;
	if (chunk < (16ull << 20)) chunk = 16ull << 20;
	if (chunk > (64ull << 20)) chunk = 64ull << 20;
	return (chunk + (16ull << 20) - 1) & ~((16ull << 20) - 1);
}

// The accountant's view of a sequence of allocations: the chunk list a GpuArena
// ends up holding. Mirrors GpuArena::alloc's walk (chunk -> chunk, then grow by
// max(chunk_bytes, need)) and its release-on-grow-when-fully-rewound rule, and
// runs the sequence to a fixed point because that is what a sampling loop does:
// every block allocates the same tensors, `new_layer()` rewinds the arena, and
// from the second block on the chunk list only grows where a tensor no longer
// fits.
u64 arena_chunk_cost(const std::vector<u64>& sizes, u64 chunk_bytes) {
	if (sizes.empty()) return 0;
	if (chunk_bytes == 0) chunk_bytes = 64ull << 20;
	std::vector<u64> chunks;
	u64 capacity = 0;
	for (int pass = 0; pass < 4; pass++) {
		size_t cur = 0;
		u64 off = 0;
		for (u64 raw : sizes) {
			const u64 need = (raw + 255ull) & ~255ull;   // alloc()'s 256-byte alignment
			for (;;) {
				if (cur < chunks.size()) {
					if (chunks[cur] - off >= need) {
						off += need;
						break;
					}
					cur++;
					off = 0;
					continue;
				}
				// Grow: the arena is about to add a chunk. Fully rewound (a fresh
				// phase), it drops the old list first - which is exactly the
				// release/new-chunk pair the ledger trace shows at a block boundary.
				if (cur == 0 && off == 0 && !chunks.empty()) {
					chunks.clear();
					capacity = 0;
				}
				const u64 want = std::max(chunk_bytes, need);
				chunks.push_back(want);
				capacity += want;
				off = need;
				break;
			}
		}
	}
	return capacity;
}

GpuArenaModel::GpuArenaModel(u64 chunk_bytes) : chunk_bytes_(chunk_bytes ? chunk_bytes : (64ull << 20)) {}

void GpuArenaModel::alloc(u64 bytes, u64 align) {
	if (bytes == 0) return;
	const u64 need = round_up((i64)bytes, (i64)align);
	const bool fully_rewound = (cur_ == 0 && cur_off_ == 0);
	for (;;) {
		if (cur_ < chunks_.size()) {
			if (chunks_[cur_] - cur_off_ >= need) {
				used_ += need;
				cur_off_ += need;
				return;
			}
			cur_++;
			cur_off_ = 0;
			continue;
		}
		// `GpuArena::alloc`'s rule, in the same words: a bump allocator can only
		// hand out the sum of its chunks, but a mark/rewind caller only needs their
		// maximum - so a grow that starts from a fully rewound arena drops the old
		// list first.
		if (fully_rewound && !chunks_.empty()) {
			chunks_.clear();
			capacity_ = 0;
		}
		const u64 want = std::max(chunk_bytes_, need);
		chunks_.push_back(want);
		capacity_ += want;
		cur_ = chunks_.size() - 1;
		cur_off_ = need;
		used_ += need;
		return;
	}
}

void GpuArenaModel::reset() {
	cur_ = 0;
	cur_off_ = 0;
	used_ = 0;
}

void GpuArenaModel::rewind_to(const Mark& m) {
	if (m.chunk > chunks_.size()) return;
	cur_ = m.chunk;
	cur_off_ = m.off;
	used_ = m.used;
}

UploadRing::~UploadRing() { destroy(); }

void UploadRing::reinit(u64 bytes) {
	if (!ctx_) {
		init(nullptr, bytes);
		return;
	}
	if (res_ && bytes == size_) return;
	ComputeContext* c = ctx_;
	GpuBuffer* old = res_;
	// Drain first: the region handed out last may still be being DMA'd from. The
	// caller is a tool-invocation boundary (nothing is in flight by contract), but
	// the ring is also reachable from a cancelled run, and freeing a buffer the GPU
	// is still reading takes the device down.
	c->submit_if_recording();
	destroy();   // forget the old pointers; the buffer belongs to the context
	if (old) c->release_buffer(old);
	init(c, bytes);
}

void UploadRing::init(ComputeContext* ctx, u64 bytes) {
	// If we were already initialized, release the old pinned buffer through its
	// owner before forgetting it — otherwise a repeated init() leaks hundreds of
	// MB of pinned host memory until the context dies.
	GpuBuffer* old = res_;
	ComputeContext* old_ctx = ctx_;
	destroy();
	if (old && old_ctx) old_ctx->release_buffer(old);
	ctx_ = ctx;
	size_ = bytes;
	// Pinned host memory: the GPU DMAs straight out of it, so a weight upload has
	// no bounce buffer and the copy is a real asynchronous transfer.
	res_ = ctx->create_upload_buffer(bytes);
	mapped_ = res_ ? res_->mapped : nullptr;
	if (!mapped_) throw MediaError("UploadRing: pinned allocation failed");
}

void UploadRing::destroy() {
	// Ownership of res_ stays with ComputeContext; just forget it.
	res_ = nullptr;
	mapped_ = nullptr;
	head_ = 0;
}

void UploadRing::drain() {
	// `submit()` waits for the GPU (see ComputeContext::submit), which is the whole
	// point: the region handed out next is one the GPU has finished reading. The
	// bracket is preserved, so a caller midway through staging a layer keeps
	// recording into the same context.
	const bool was_recording = ctx_ && ctx_->recording();
	if (ctx_) {
		ctx_->submit_if_recording();
		if (was_recording) ctx_->begin();
	}
	head_ = 0;
}

// The claim check stage() and reserve() share: is there room left, and if not can
// there be? Draining is what the chunked streamers used to do by hand around every
// piece (`submit_if_recording(); rewind(); begin();`); doing it here as well means
// a caller that does *not* chunk its uploads - the text encoder stages a whole
// 244 MB layer as one bracket - cannot turn a small ring into a hard failure.
//
// Wrapping *without* draining is the thing that must never happen: the GPU may
// still be reading the region being overwritten (that is the copy-corruption mode
// the synchronous submit contract exists to prevent).
u64 UploadRing::ensure_room(u64 bytes, u64 align) {
	const u64 off = round_up((i64)head_, (i64)align);
	if (off + bytes <= size_) return off;
	drain();
	return 0;
}

// The one case draining cannot fix, and the one the caller has to answer: the ring
// is sized for the largest single tensor it uploads.
[[noreturn]] static void ring_too_small(u64 bytes, u64 align, u64 size) {
	char buf[256];
	snprintf(buf, sizeof buf,
	         "UploadRing: staging buffer too small (a %s tensor does not fit the %s ring even "
	         "empty, align %llu)",
	         format_bytes((u64)round_up((i64)bytes, (i64)align)).c_str(),
	         format_bytes(size).c_str(), (unsigned long long)align);
	throw MediaError(buf);
}

u64 UploadRing::stage(const void* src, u64 bytes, u64 align) {
	if (!res_ || !mapped_) throw MediaError("UploadRing: not initialised");
	if (!align) align = 1;
	if ((u64)round_up((i64)bytes, (i64)align) > size_) ring_too_small(bytes, align, size_);
	const u64 off = ensure_room(bytes, align);
	memcpy((u8*)mapped_ + off, src, (size_t)bytes);
	head_ = off + bytes;
	return off;
}

void* UploadRing::reserve(u64 bytes, u64* out_off, u64 align) {
	if (!res_ || !mapped_) throw MediaError("UploadRing: not initialised");
	if (!align) align = 1;
	if ((u64)round_up((i64)bytes, (i64)align) > size_) ring_too_small(bytes, align, size_);
	const u64 off = ensure_room(bytes, align);
	head_ = off + bytes;
	if (out_off) *out_off = off;
	return (u8*)mapped_ + off;
}

void UploadRing::rewind() { head_ = 0; }

// ── upload helper ──────────────────────────────────────────────────────────

void upload_range(ComputeContext& ctx, UploadRing& ring, GpuBuffer* res, u64 off, const void* src,
                  u64 bytes) {
	if (bytes == 0) return;
	if (!ctx.recording())
		throw MediaError("upload_range: no open dispatch bracket (the copy would be dropped)");
	const u64 roff = ring.stage(src, bytes);
	upload_range_staged(ctx, ring, res, off, roff, bytes);
}

// The copy half, for a region the caller staged itself with reserve().
//
// A copy needs no resource-state transitions on CUDA, so the whole thing is one
// asynchronous copy, ordered against the stream's other work and therefore
// unable to race a kernel that was enqueued earlier.
void upload_range_staged(ComputeContext& ctx, UploadRing& ring, GpuBuffer* res, u64 off,
                         u64 ring_off, u64 bytes) {
	if (bytes == 0) return;
	if (!ctx.recording())
		throw MediaError("upload_range_staged: no open dispatch bracket (the copy would be dropped)");
	const void* src = (const u8*)ring.buffer()->mapped + ring_off;
	ctx.device_context().copy_h2d_async(res, off, src, bytes);
}

// The file-backed form of `upload_range` (see the header for why it exists). The
// tensor is staged in the ring in one piece, so a tensor larger than an empty ring
// is a hard error - which is the same contract `upload_range` has, and the same one
// the callers are written against.
void upload_file_range(ComputeContext& ctx, UploadRing& ring, GpuBuffer* res, u64 off,
                       const SafeTensors& st, const StTensor& t) {
	if (t.nbytes == 0) return;
	if (!ctx.recording())
		throw MediaError(
		    "upload_file_range: no open dispatch bracket (the copy would be dropped)");
	u64 ring_off = 0;
	void* host = ring.reserve(t.nbytes, &ring_off);
	st.read_file(t.offset, (size_t)t.nbytes, host);
	upload_range_staged(ctx, ring, res, off, ring_off, t.nbytes);
}

}  // namespace phi::media
