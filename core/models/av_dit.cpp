// MiniMax H3 DiT — W7 implementation.
//
// Reference: comfy/ldm/minimax/model.py (MiniMaxH3Model), read line by line; the
// architecture dump - a tensor-by-tensor read of that header - is the shape
// authority.
//
// What this file does, in the reference's own order:
//
//   1. PackedLayout — text | (refs) | audio | video token positions, and the rope
//      grid over them. The reference computes the grid in fp64 because the video
//      time axis accumulates 5/3-pixel-frame spans; replicated here in double.
//   2. embedding    — video rows patchified (1,2,2), audio rows packed
//      channel-major (both through fp32 patch projections); the text rows through
//      condition_proj + the 2-layer token_refiner (bf16 weights, fp16 GEMM).
//   3. adaLN        — the checkpoint has no time embedder: `adaln_t_table`
//      [1025, 8] is looked up with a lerp and feeds a rank-8 projection whose
//      output is 18 (6 modulations x 3 modalities) hidden-sized vectors. Eight
//      inputs against 96768 outputs is 1.5 MFLOP per block, so it is evaluated on
//      the host in fp32 and uploaded — the same arithmetic the reference's
//      `adaln_dtype=float32` linear does.
//   4. 50 blocks    — int8 convrot qkv/out/fc1/fc2, per-head RMSNorm, the 3-axis
//      split-half rope, bidirectional attention, SwiGLU.
//   5. final layer  — norm, adaLN, the fp32 video (96) and audio (32) heads, and
//      the audio carry/unscale that maps the packed latent's velocity back onto
//      the sampler's (`h3_joint_av.hpp`).
//
// Weight residency (plan §4): the checkpoint is 20.97 GB of int8 and the card has
// 6 GB. `plan_residency` keeps as many blocks resident as the activation peak
// leaves room for and streams the rest through the per-block weight arena; one
// block is ~386 MB.
//
// Memory policy: every buffer this forward needs is allocated once per call from
// the activation arena in a fixed order, so step-to-step the addresses do not
// move and nothing is allocated after the first step. The fused qkv buffer
// doubles as the attention output buffer (the attention reads q/k/v, never qkv):
// 1.6 GB of peak saved at 20k tokens.
#include "models/av_dit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "kernels/h3_kernels.hpp"
#include "kernels/parallel_for.hpp"
#include "kernels/gpu_ops.hpp"
#include "models/media_geometry.hpp"
#include "host/quant.hpp"
#include "models/lora.hpp"
#include "sampling/sampling_h3.hpp"
#include "runtime/vram_budget.hpp"
#include "runtime/vram_window.hpp"
#include "util/base.hpp"

namespace phi::media {

namespace {

// ── constants read off the checkpoint header ───────────────────────────────
constexpr i64 kRopeFreqs = 16;   // rope.inv_freq
constexpr i64 kRopePairs = 48;   // rot/2 = 96/2
constexpr i64 kRot = 96;         // 3 axes x 16 frequencies x 2
constexpr i64 kInner = 7168;     // heads * head_dim
constexpr i64 kQkvOut = 3 * kInner;
constexpr i64 kVidPatch = 96;   // 24 channels x 1 x 2 x 2
constexpr i64 kAudPatch = 32;
constexpr i64 kFfn = 14336;
// The hidden width the DiT's activation buffers are sized from (`config_.hidden`
// reads it back off `video_patch_proj.weight`, and every buffer in `forward()`
// is a multiple of it). Named here so the activation-cost model can price the
// frame without a config instance - the RNG kernels all run on the 5376-wide
// stream.
constexpr i64 kHidden = 5376;
// The text-encoder width the packed conditioning rows arrive at (`text_dim` in
// the model config). Same reason: the model prices its own frame in a pure
// function.
constexpr i64 kTextDim = 5120;
// FFN chunk: 2048 rows of 28672 fp32 is 235 MB, and the int8 GEMM tiles M by 128,
// so above that the chunk size is purely a memory decision.
constexpr i64 kChunk = 2048;      // default activation chunk (rows per dispatch)
constexpr i64 kChunkMin = 256;    // floor the residency planner will shrink it to

struct BlockW {
	// One precision-preserving projection each: int8 tensorwise / a packed family as
	// the int8 pair the tensor-core GEMM reads, a float source as its own dense
	// f16 / bf16 / f32. `linear_gemm` runs whichever the file stored.
	QuantLinear qkv, out, fc1, fc2;
	GpuAlloc norm1, norm2, q_norm, k_norm;
	// A LoRA that touches this block's projections is *not* folded into the
	// weights (see `LoraTail`): the base pair above is uploaded untouched and the
	// delta rides along as this epilogue.
	LoraTail tqkv, tout, tfc1, tfc2;
};

struct DenseW {
	GpuAlloc cond_w, cond_b;
	GpuAlloc vid_patch_w, vid_patch_b, aud_patch_w, aud_patch_b;
	GpuAlloc ref_norm;   // token_refiner.final_norm
	GpuAlloc final_norm, final_w, final_b;
	GpuAlloc vid_out_w, vid_out_b, aud_out_w, aud_out_b;
};

struct RefinerW {
	GpuAlloc qkv, out, fc1, fc2;   // bf16, raw
	GpuAlloc norm1, norm2, q_norm, k_norm;
	// The refiner's four projections are in the LoRA file as well, so they get the
	// same runtime correction the blocks do.
	LoraTail tqkv, tout, tfc1, tfc2;
};

struct Seg {
	i64 a = 0, b = 0;   // token range
	i64 chunk0 = 0;     // first modulation chunk row: (m*3 + modality) * 6
	i32 kind = 0;       // which packed-stream span this is (Kseg below)
};

// The packed-stream kinds of the reference's PackedLayout (`segments`).
enum Kseg : i32 { KText = 0, KCond, KRefImg, KRefAudio, KAudio, KVideo };

// One "video patch row" source for the embedding step: the rows of one packed
// segment that are filled from the video patch projection. On the t2va path
// there is exactly one (the target video segment) and it lands where the
// pre-ref2va code wrote it, so that projection call is byte-for-byte unchanged.
struct VRowSrc {
	i32 kind = KVideo;            // KRefImg / KCond (reference) or KVideo (target)
	const float* lat = nullptr;   // [24, vt, lh, lw]
	i64 vt = 0, lh = 0, lw = 0;
	i64 dst = 0, n = 0;           // first destination row, row count
};

// Same for the audio patch rows ([32, 2, rt] channel-major source).
struct ARowSrc {
	i32 kind = KAudio;
	const float* lat = nullptr;
	i64 rt = 0;
	i64 dst = 0, n = 0;
};

struct Layout {
	i64 text = 0, audio_t = 0, video_t = 0;
	i64 n_video = 0, n_audio = 0, s = 0;
	i64 tokens_h = 0, tokens_w = 0;
	std::vector<Seg> segs;  // packed spans, in sequence order (refs included)
	// ref2va: the reference rows, in packing order, and where they land.
	std::vector<VRowSrc> vrows;
	std::vector<ARowSrc> arows;
	i64 n_ref_video = 0, n_ref_audio = 0;
	// staging floats one patch-projection group needs
	i64 stage_floats = 0;
	std::vector<float> rope;   // [s, kRopePairs, 2]
	std::vector<float> pos;    // [s, 3]
	i64 m_count = 0;   // rows of the per-timestep modulation table
	std::vector<float> t_vals;
	std::vector<float> t_emb;   // [m_count, 8]
	i64 video_m = 1, audio_m = 0;
	i64 video_seg = -1, audio_seg = -1;
};

struct Impl {
	SafeTensors st;
	DenseW dense;
	std::vector<BlockW> res;   // resident blocks [0, res_n)
	i64 res_n = 0;
	// The window `plan_residency` decided on but has not uploaded yet: the upload is
	// deferred to the first forward(), *after* the token refiner has released the
	// weight arena, so the window is never live at the same time as the refiner's
	// 770 MB (which is what let the window grow by two blocks at 540P/124).
	bool res_pending = false;
	GpuArena res_arena;
	RefinerW ref0, ref1;   // token_refiner, streamed per call
	i64 planned_frames = 0, planned_audio_t = 0, planned_lat_h = 0, planned_lat_w = 0;
	i64 planned_text = 0;
	i64 planned_ref_rows = 0;
	u64 planned_ref_pixels = 0;
	u64 planned_extra = 0;
	i64 planned_chunk = kChunk;
	bool planned = false;
	u64 layer_bytes = 0;
	// The byte size of every tensor of one main block, *in the order load_block
	// allocates them*. The residency plan turns this into what the accountant will
	// really book for a streamed block (`arena_chunk_cost`): the charge is per
	// chunk, not per byte, so a 386 MB block costs ~594 MB on a 64 MB granularity.
	std::vector<u64> layer_tensors;
	// The same for the token refiner's block, which streams through the same arena
	// on every step (`load_refiner`'s order).
	std::vector<u64> refiner_tensors;
	// [offset, offset+nbytes) of every tensor of block i, in the checkpoint. The
	// streamer pages the next block's range in while this one computes: reading
	// the mapping cold costs a 4 KB fault per page (~0.6 GB/s measured on this
	// NVMe, against 2.1 GB/s through a 4 MB ReadFile), and `run_block` is seconds
	// of GPU work per block, which is exactly the window the paging I/O needs.
	// The (offset, nbytes) of every tensor `load_block` reads for a block, in that
	// call's own order - the list the host-side weight cache pins (see
	// SafeTensors::pin_range). `layer_range` below is the min/max envelope, which
	// is useless for this: a block's 18 tensors are scattered across the whole
	// 20 GB file (the four big matrices are one contiguous ~345 MB run, the scales
	// and norms live in kind-grouped regions elsewhere).
	std::vector<std::vector<std::pair<u64, u64>>> block_ranges;
	// The same for the token refiner's two blocks, which re-stream on every step.
	std::vector<std::pair<u64, u64>> refiner_ranges;
	// The host cache is filled once per weight set, on the first forward().
	bool host_cache_ready = false;
	// LoRA chain to fold into the block matrices at load time (may be null).
	const LoraSet* lora = nullptr;
	std::vector<std::pair<u64, u64>> layer_range;

};

std::map<const AvDiT*, std::unique_ptr<Impl>>& registry() {
	static std::map<const AvDiT*, std::unique_ptr<Impl>> r;
	return r;
}

Impl* impl_of(const AvDiT* d) {
	auto it = registry().find(d);
	return it == registry().end() ? nullptr : it->second.get();
}

// ── per-section profile (PHI_DIT_TIME) ─────────────────────────────────────
//
// The same instrument the image chain uses, for the same reason: "62 s per
// step" cannot say whether the time is the disk, the GEMMs or the attention,
// and each of those three answers implies a different change. Every
// `ComputeContext::submit()` in this engine is a submit-and-wait, so putting one
// at both ends of a section turns "recorded" into "executed" and the wall clock
// around the section is its time on the GPU.
struct H3Prof {
	double stream = 0;      // weight reads + uploads for the blocks that are not resident
	double embed = 0;       // patch projections + condition_proj + token refiner
	double qkv_gemm = 0;    // the chunked int8 qkv projections
	double qkv_prep = 0;    // split + per-head RMSNorm + 3-axis rope
	double attn = 0;        // the attention kernel itself
	double out_mlp = 0;     // out_proj + SwiGLU MLP (int8)
	double final = 0;       // final layer + heads
	double unpack = 0;      // device -> host readback of the velocity rows
	double total = 0;       // whole forward()
	u64 stream_bytes = 0;
	u64 io_bytes = 0;       // heavy bytes pulled off the file by read_file
	i64 resident = 0, blocks = 0;
	bool on = false;
};

H3Prof& h3_prof() {
	static H3Prof p;
	static bool init = false;
	if (!init) {
		init = true;
		const char* e = getenv("PHI_DIT_TIME");
		p.on = e && *e && *e != '0';
	}
	return p;
}

// Charge a section to `sink` (null when profiling is off, which makes the whole
// thing free). The queue is drained at both ends, so the delta is GPU time.
struct ProfScope {
	GpuCtx* g = nullptr;
	double* sink = nullptr;
	double t0 = 0;
	ProfScope(GpuCtx* gg, double* s) : g(gg), sink(s) {
		if (!sink) return;
		g->ctx->submit_if_recording();
		g->ctx->begin();
		t0 = now_ms();
	}
	~ProfScope() {
		if (!sink) return;
		g->ctx->submit();
		g->ctx->begin();
		*sink += now_ms() - t0;
	}
	ProfScope(const ProfScope&) = delete;
	ProfScope& operator=(const ProfScope&) = delete;
};

struct ProfSink {
	double* s = nullptr;
};

double* prof_sink(double H3Prof::*m) {
	H3Prof& p = h3_prof();
	return p.on ? &(p.*m) : nullptr;
}

// Debug-only hidden-stream dump (PHI_DIT_DUMP=<prefix>): the hidden stream right
// after the embedding, i.e. what the 50 blocks actually start from. "The sampled
// latent is noise" has two very different causes - the latent never reaches the
// model, or it does and the model fails to remove it - and this tells them apart
// in a single run (with no dump, both look like a white latent).
const char* dit_dump_prefix() {
	static int read = 0;
	static const char* v = nullptr;
	if (!read) {
		read = 1;
		v = getenv("PHI_DIT_DUMP");
		if (v && !*v) v = nullptr;
	}
	return v;
}

// PHI_H3_DBG_BLOCK=<i>: per-chunk stage dumps for block `i` only (default -1,
// i.e. off). The whole-stream `block%02d` dump says *that* a row went bad, not
// which stage produced it; this one writes every intermediate of one block so
// the first non-finite tensor can be named. Names are
// <prefix>dbg_b<i>_<stage>_<row0>.f32 and the sizes are printed on stderr.
i64 h3_dbg_block() {
	static i64 v = -2;
	if (v == -2) {
		const char* e = getenv("PHI_H3_DBG_BLOCK");
		v = (e && *e) ? strtoll(e, nullptr, 10) : -1;
	}
	return v;
}

void dbg_dump(GpuCtx& g, const GpuAlloc& a, i64 n_f32, i64 block, const char* stage, i64 row0) {
	const char* pre = dit_dump_prefix();
	if (!pre) return;
	char nm[160];
	snprintf(nm, sizeof nm, "dbg_b%lld_%s_%lld", (long long)block, stage, (long long)row0);
	g.ctx->submit_if_recording();
	g.ctx->begin();
	const std::vector<float> v = g.download_f32(a, (u64)n_f32);
	double mn = 1e30, mx = -1e30;
	i64 nf = 0;
	for (float x : v) {
		if (!std::isfinite(x)) {
			nf++;
			continue;
		}
		mn = std::min(mn, (double)x);
		mx = std::max(mx, (double)x);
	}
	fprintf(stderr, "[h3dbg] b%lld %-14s row0=%-6lld n=%-8lld [%.4g, %.4g] nonfinite %lld%c",
	        (long long)block, stage, (long long)row0, (long long)n_f32, mn, mx, (long long)nf, 10);
	const std::string path = std::string(pre) + nm + ".f32";
	if (FILE* f = fopen(path.c_str(), "wb")) {
		fwrite(v.data(), 4, v.size(), f);
		fclose(f);
	}
	g.ctx->begin();
}

// Same, for the packed-fp16 q/k/v: they are halves, so the fp32 reader would
// read twice the buffer (and print nonsense). Reports inf/NaN explicitly,
// because "a half went infinite" is exactly the failure this hunts.
void dbg_dump_f16(GpuCtx& g, const GpuAlloc& a, i64 n_half, i64 block, const char* stage,
                  i64 row0) {
	const char* pre = dit_dump_prefix();
	if (!pre) return;
	g.ctx->submit_if_recording();
	g.ctx->begin();
	std::vector<u16> h((size_t)n_half);
	g.ctx->download(a.res, a.off, h.data(), (size_t)n_half * 2);
	float mx = 0.0f;
	i64 nf = 0;
	for (u16 u : h) {
		const float x = f16_to_f32(u);
		if (!std::isfinite(x)) {
			nf++;
			continue;
		}
		mx = std::max(mx, std::fabs(x));
	}
	fprintf(stderr, "[h3dbg] b%lld %-14s row0=%-6lld n=%-8lld absmax %.4g nonfinite %lld%c",
	        (long long)block, stage, (long long)row0, (long long)n_half, mx, (long long)nf, 10);
	char nm[160];
	snprintf(nm, sizeof nm, "dbg_b%lld_%s_%lld.f16", (long long)block, stage, (long long)row0);
	const std::string path = std::string(pre) + nm;
	if (FILE* f = fopen(path.c_str(), "wb")) {
		fwrite(h.data(), 2, h.size(), f);
		fclose(f);
	}
	g.ctx->begin();
}

bool dit_res_debug() {
	static int v = -1;
	if (v < 0) {
		// PHI_DIT_PLAN is the documented "print the residency plan" switch (see
		// core/README.md) and PHI_DIT_RES is the count pin; both turn the trace on.
		// Reading only the pin made "log what the tuner decided" impossible without
		// also overriding the planner.
		const char* e = getenv("PHI_DIT_PLAN");
		if (!e || !*e || *e == '0') e = getenv("PHI_DIT_RES");
		v = (e && *e && *e != '0') ? 1 : 0;
	}
	return v != 0;
}

// ── upload helpers ─────────────────────────────────────────────────────────

// Weight uploads, in pieces that fit the staging ring.
//
// `UploadRing::stage` never wraps: a copy must not overwrite a region the GPU has
// not read yet, so a tensor larger than the ring is a hard error by design ("the
// caller sizes the ring for the largest layer it uploads"). One H3 block is
// 108 + 30 + 147 + 15 MB of int8, and the token refiner's fc1 is 294 MB of bf16,
// so "size the ring for the largest tensor" would mean a 384 MB host buffer that
// exists for one weight. Splitting the copy into ring/2 pieces and submitting
// between them costs a ~50 us fence wait per 128 MB and keeps the ring at the
// size the image chain wants.
void upload_chunked(GpuCtx& g, const GpuAlloc& dst, const void* src, u64 bytes) {
	const u64 max_chunk = std::max<u64>((g.ring->size() / 2) & ~255ull, 1ull << 20);
	// This function is called from two places with different bracket state:
	// `load_block` runs inside the bracket `GpuCtx::new_layer()` opened, while
	// `open()`'s weight uploads have none (and `begin()` on an already-open list
	// throws). So it opens one only when it has to, and closes only its own - which
	// is what keeps the "no per-piece submit" change below safe in both.
	const bool own_bracket = !g.ctx->recording();
	if (own_bracket) g.ctx->begin();
	u64 done = 0;
	while (done < bytes) {
		const u64 n = std::min(max_chunk, bytes - done);
		// No per-piece submit here any more. `UploadRing::reserve`/`stage` already
		// drains (submits, waits, rewinds) exactly when the *next* piece would not
		// fit, and every caller that rewinds the ring around these uploads
		// (`GpuCtx::new_layer`, `end_layer`, `new_step`) submits first. The old
		// `submit_if_recording(); rewind(); begin(); ... submit();` bracket around
		// each piece therefore did nothing the ring did not already guarantee, and
		// it cost one full stream synchronise per piece - see `upload_chunked_file`
		// for what that adds up to.
		upload_range(*g.ctx, *g.ring, dst.res, dst.off + done, (const u8*)src + done, n);
		done += n;
	}
	if (own_bracket) {
		g.ctx->submit();
		g.ring->rewind();
	}
}

// The same upload, sourced from the *file* instead of the mapping.
//
// This is the path every block weight takes, and it is the DiT's hot path: the
// sampling loop reads all 50 blocks per step (≈18 GB, the file is larger than
// RAM, so it is always cold). Reading through the mapping costs a page fault per
// 4 KB — measured here at ~0.85 GB/s — while `read_file` sustains ~1.2 GB/s, which
// is this NVMe's own ceiling for the pattern (see the host weight cache in
// `forward`). The bytes land straight in the staging ring (reserve()), so there is
// no intermediate copy either.
void upload_chunked_file(GpuCtx& g, const GpuAlloc& dst, SafeTensors& st, const StTensor& t) {
	// One implementation, shared with the video VAE's decoder blocks and with the
	// image/text chains: `upload_file_into` is the same chunked read-into-the-ring
	// this used to be, plus the zero-copy branch for a tensor one of the host
	// caches holds in page-locked memory (see SafeTensors::pinned_dma_at). This
	// is the DiT's hot path - 17 GB per sampling step - so the branch is worth
	// 1.6 s of a 40 s step whenever it fires.
	g.upload_file_into(dst, st, t);
}

[[maybe_unused]] void upload_chunked_file_ring(GpuCtx& g, const GpuAlloc& dst, SafeTensors& st,
                                              const StTensor& t) {
	const u64 bytes = t.nbytes;
	if (bytes == 0) return;
	const u64 max_chunk = std::max<u64>((g.ring->size() / 2) & ~255ull, 1ull << 20);
	// Same two callers, same two bracket states as `upload_chunked` above.
	const bool own_bracket = !g.ctx->recording();
	if (own_bracket) g.ctx->begin();
	u64 done = 0;
	while (done < bytes) {
		const u64 n = std::min(max_chunk, bytes - done);
		// The ring decides when to drain: `reserve` submits, waits and rewinds only
		// when this piece would not fit in what is left, and it leaves the caller's
		// dispatch bracket exactly as it found it. That is the whole safety
		// argument, because the three places that rewind the ring around these
		// uploads - `GpuCtx::new_layer`, `end_layer` and `new_step` - all submit
		// first, and `GpuArena::reset` for the weight arena runs after those.
		//
		// This used to be a hand-written bracket around *every* piece:
		// `submit_if_recording(); rewind(); begin(); ...; submit();`, which on the
		// DiT's hot path is 12 tensors x 41 streamed blocks = ~490 full stream
		// synchronises per sampling step. On this box that is worth 1.1 s of a
		// step - measured by flipping the old bracket back on behind an environment
		// switch and running both cold, four times each: 19.79 s/step with it,
		// 18.65 s/step without, and the gap held in every one of the four
		// alternating runs.
		u64 ring_off = 0;
		void* host = g.ring->reserve(n, &ring_off);
		st.read_file(t.offset + done, (size_t)n, host);
		upload_range_staged(*g.ctx, *g.ring, dst.res, dst.off + done, ring_off, n);
		done += n;
	}
	if (own_bracket) {
		g.ctx->submit();
		g.ring->rewind();
	}
}

[[maybe_unused]] GpuAlloc up_i8(SafeTensors& st, const std::string& name, GpuCtx& g, GpuArena& arena) {
	const StTensor& t = st.require(name);
	if (t.dtype != DType::I8) throw MediaError("h3dit: " + name + " is not int8");
	GpuAlloc a = arena.alloc(t.nbytes);
	upload_chunked_file(g, a, st, t);
	return a;
}

GpuAlloc up_scale(SafeTensors& st, const std::string& name, GpuCtx& g, GpuArena& arena) {
	const StTensor& t = st.require(name);
	std::vector<float> v = tensor_to_f32(st, t);   // [N,1] -> [N]
	GpuAlloc a = arena.alloc((u64)v.size() * 4);
	upload_chunked(g, a, v.data(), v.size() * 4);
	return a;
}

GpuAlloc up_f32(SafeTensors& st, const std::string& name, GpuCtx& g, GpuArena& arena) {
	const StTensor& t = st.require(name);
	std::vector<float> v = tensor_to_f32(st, t);
	GpuAlloc a = arena.alloc((u64)v.size() * 4);
	upload_chunked(g, a, v.data(), v.size() * 4);
	return a;
}

// raw bytes: bf16/f16/f32 weights the fp16 GEMM reads directly. When a helper
// tensor arrives in another precision it is materialised to `as` (the
// dtype its consumer reads), which never changes the safetensors behaviour: a
// raw tensor is uploaded verbatim whatever `as` says.
GpuAlloc up_raw(SafeTensors& st, const std::string& name, GpuCtx& g, GpuArena& arena,
                DType as = DType::F16) {
	const StTensor& t = st.require(name);
	if (st.is_raw(t)) {
		GpuAlloc a = arena.alloc((u64)t.numel * (u64)dtype_size(t.dtype));
		upload_chunked_file(g, a, st, t);
		return a;
	}
	std::vector<u8> bytes = st.materialize(t, as);
	GpuAlloc a = arena.alloc(bytes.size());
	upload_chunked(g, a, bytes.data(), bytes.size());
	return a;
}

// One quantised linear plus its per-output-row fp32 scale, whatever the
// checkpoint holds:
//   * int8 tensorwise (+ optional convrot): streamed off the file verbatim, so
//     the shipped int8 checkpoints cost exactly what they did before;
//   * a packed family (w4a8 / w6a8): its codes go to the device as stored and the
//     int8 form is built there - the model's own format, not a conversion;
//   * a float matrix (f16 / bf16 / f32, fp8 -> f16): uploaded in its own precision
//     for the dense GEMM. It is *not* requantised to int8 (goal #1).
QuantLinear load_linear_pair(SafeTensors& st, GpuCtx& g, GpuArena& arena,
                             const std::string& base) {
	const StTensor& t = st.require(base + ".weight");
	// A float source keeps its own precision: no requantisation, the dense GEMM.
	if (weight_is_dense_float(st, t)) {
		DenseUpload du = upload_dense_linear(g, st, base, arena);
		QuantLinear q;
		q.d = du.w;
		q.dtype = du.dtype;
		q.n = du.n;
		q.k = du.k;
		return q;
	}
	// Shipped int8 tensorwise: streamed verbatim (the block loop was written
	// against exactly these bytes).
	if (st.is_plain_int8(t)) {
		QuantLinear q;
		q.n = t.shape[0];
		q.k = t.shape[1];
		q.w = arena.alloc(t.nbytes);
		upload_chunked_file(g, q.w, st, t);
		q.scale = up_scale(st, base + ".weight_scale", g, arena);
		return q;
	}
	// A packed family (w4a8 / w6a8): expanded to int8 on the device.
	return upload_linear_any(g, st, base, arena, nullptr);
}

// Debug-only activation statistics (PHI_DIT_DUMP=<prefix>). One line per call:
// the extrema of one buffer, which is what turns "the residual stream grows by
// 300x in block 0" into "this term is the one that does it". A no-op unless
// the environment variable is set.
// Writes one buffer to <prefix><name>.f32 (fp32) or .f16 (packed halves as u16).
// Used by the block-0 diagnosis: comparing two runs separates "the attention
// already ignores its input" from "the blocks do".
void dump_file(GpuCtx* g, const GpuAlloc& a, i64 n, const char* name, bool as_f16) {
	const char* pre = dit_dump_prefix();
	if (!pre) return;
	g->ctx->submit_if_recording();
	g->ctx->begin();
	const std::string path = std::string(pre) + name + (as_f16 ? ".f16" : ".f32");
	FILE* f = fopen(path.c_str(), "wb");
	if (f) {
		if (as_f16) {
			std::vector<u16> h((size_t)n);
			g->ctx->download(a.res, a.off, h.data(), (size_t)n * 2);
			fwrite(h.data(), 2, h.size(), f);
		} else {
			const std::vector<float> v = g->download_f32(a, (u64)n);
			fwrite(v.data(), 4, v.size(), f);
		}
		fclose(f);
	}
	g->ctx->begin();
}

void dump_stat(GpuCtx* g, const GpuAlloc& a, i64 n_f32, const char* name, i64 block) {
	if (!dit_dump_prefix()) return;
	g->ctx->submit_if_recording();
	g->ctx->begin();
	const std::vector<float> v = g->download_f32(a, (u64)n_f32);
	double mn = 1e30, mx = -1e30, sum = 0, sum2 = 0;
	for (float x : v) {
		if (std::isnan(x)) continue;
		mn = std::min(mn, (double)x);
		mx = std::max(mx, (double)x);
		sum += x;
		sum2 += (double)x * x;
	}
	const double n = (double)std::max<size_t>(v.size(), 1);
	const double mean = sum / n;
	const double var = std::max(0.0, sum2 / n - mean * mean);
	fprintf(stderr, "[h3dit] b%lld %-12s [%.4f, %.4f] mean %.4f std %.4f%c", (long long)block,
	        name, mn, mx, mean, std::sqrt(var), 10);
	g->ctx->begin();
}

// PHI_DIT_DUMP=<prefix>: write the hidden stream after block `i` as
// <prefix>block%02d.f32. A handful of blocks by default; PHI_DIT_DUMP_ALL=1
// writes all 50, which is what the block-by-block reference comparison needs
// - 7.4 MB per block at 512x320x5, 415 MB at 540P/5s.
//
// PHI_DIT_DUMP_BLOCKS="48,49" picks exactly the blocks to write, comma
// separated. The default set is spread over the stack, which says nothing about
// *which* block corrupts a stream: bisecting a NaN that first appears after
// block 49 (at 415 MB a dump) by flipping ALL on and off is not a bisection.
// This switch is what made "the video is black" a five-run answer instead of a
// fifty-run one.
void dump_hidden(GpuCtx& g, const GpuAlloc& buf, i64 i, i64 S, i64 hidden) {
	const char* dp = dit_dump_prefix();
	if (!dp) return;
	static const i64 kDumpAt[] = {0, 1, 2, 4, 8, 16, 32, 49};
	static const std::set<i64>* only = []() -> const std::set<i64>* {
		const char* e = getenv("PHI_DIT_DUMP_BLOCKS");
		if (!e || !*e) return nullptr;
		auto* s = new std::set<i64>();
		std::string t(e);
		size_t p = 0;
		while (p <= t.size()) {
			const size_t q = t.find(',', p);
			const std::string one = t.substr(p, q == std::string::npos ? std::string::npos : q - p);
			if (!one.empty()) s->insert((i64)strtoll(one.c_str(), nullptr, 10));
			if (q == std::string::npos) break;
			p = q + 1;
		}
		return s;
	}();
	bool want = false;
	const char* all = getenv("PHI_DIT_DUMP_ALL");
	if (only) want = only->count(i) > 0;
	else if (all && *all && *all != '0') want = true;
	else {
		for (i64 kd : kDumpAt)
			if (kd == i) want = true;
	}
	if (!want) return;
	g.ctx->submit_if_recording();
	g.ctx->begin();
	const std::vector<float> hh = g.download_f32(buf, (u64)S * hidden);
	char path[512];
	snprintf(path, sizeof path, "%sblock%02lld.f32", dp, (long long)i);
	if (FILE* hf = fopen(path, "wb")) {
		fwrite(hh.data(), 4, hh.size(), hf);
		fclose(hf);
	}
	g.ctx->begin();
}

void load_block(SafeTensors& st, GpuCtx& g, GpuArena& arena, i64 i, BlockW& L,
                const LoraSet* lora) {
	const std::string p = "blocks." + std::to_string(i) + ".";
	// The base weights load *unadapted* - a LoRA is the runtime correction below,
	// so the streaming path pays nothing for it - and each adapted projection gets
	// its [rank, k] / [n, rank] factor pair in the same arena, i.e. streamed with
	// the block it belongs to.
	QuantLinear a = load_linear_pair(st, g, arena, p + "attn.qkv_proj");
	L.qkv = a;
	L.tqkv = load_lora_tail(g, arena, p + "attn.qkv_proj", lora, a.n, a.k);
	a = load_linear_pair(st, g, arena, p + "attn.out_proj");
	L.out = a;
	L.tout = load_lora_tail(g, arena, p + "attn.out_proj", lora, a.n, a.k);
	a = load_linear_pair(st, g, arena, p + "mlp.fc1");
	L.fc1 = a;
	L.tfc1 = load_lora_tail(g, arena, p + "mlp.fc1", lora, a.n, a.k);
	a = load_linear_pair(st, g, arena, p + "mlp.fc2");
	L.fc2 = a;
	L.tfc2 = load_lora_tail(g, arena, p + "mlp.fc2", lora, a.n, a.k);
	L.norm1 = up_f32(st, p + "norm1.weight", g, arena);
	L.norm2 = up_f32(st, p + "norm2.weight", g, arena);
	L.q_norm = up_f32(st, p + "attn.q_norm.weight", g, arena);
	L.k_norm = up_f32(st, p + "attn.k_norm.weight", g, arena);
}

void load_refiner(SafeTensors& st, GpuCtx& g, GpuArena& arena, i64 i, RefinerW& L,
                  const LoraSet* lora) {
	const std::string p = "token_refiner.blocks." + std::to_string(i) + ".";
	// The refiner's matrices are bf16 (fp16 GEMM); a source in another precision
	// is materialised back to bf16 so the same GEMM reads them.
	L.qkv = up_raw(st, p + "attn.qkv_proj.weight", g, arena, DType::BF16);
	L.out = up_raw(st, p + "attn.out_proj.weight", g, arena, DType::BF16);
	L.fc1 = up_raw(st, p + "mlp.fc1.weight", g, arena, DType::BF16);
	L.fc2 = up_raw(st, p + "mlp.fc2.weight", g, arena, DType::BF16);
	L.norm1 = up_f32(st, p + "norm1.weight", g, arena);
	L.norm2 = up_f32(st, p + "norm2.weight", g, arena);
	L.q_norm = up_f32(st, p + "attn.q_norm.weight", g, arena);
	L.k_norm = up_f32(st, p + "attn.k_norm.weight", g, arena);
	L.tqkv = load_lora_tail(g, arena, p + "attn.qkv_proj", lora, kQkvOut, kHidden);
	L.tout = load_lora_tail(g, arena, p + "attn.out_proj", lora, kHidden, kInner);
	L.tfc1 = load_lora_tail(g, arena, p + "mlp.fc1", lora, 2 * kFfn, kHidden);
	L.tfc2 = load_lora_tail(g, arena, p + "mlp.fc2", lora, kHidden, kFfn);
}

// ── the reference's position grid (PackedLayout) ───────────────────────────

std::vector<double> axis_from_sqrt_area(i64 dim, i64 patch, double sqrt_area) {
	const double ratio = (double)dim / sqrt_area;
	const i64 n = dim / patch;
	std::vector<double> out((size_t)n);
	for (i64 i = 0; i < n; i++)
		out[(size_t)i] = ((double)i * (ratio / (double)n) + (1.0 - ratio) * 0.5) * 32.0;
	return out;
}

const i64 kFramePerToken[5] = {1, 4, 4, 4, 4};
const double kFrameRescale = 5.0 / 3.0;

std::vector<double> video_t_grid(i64 n, double origin) {
	std::vector<double> out((size_t)n);
	double acc = origin;
	for (i64 k = 0; k < n; k++) {
		out[(size_t)k] = acc;
		acc += kFrameRescale * (double)kFramePerToken[k % 5];
	}
	return out;
}

// sum(_video_t_spans(n)): the time-axis length an n-frame reference video spans.
double video_t_spans_sum(i64 n) {
	double acc = 0.0;
	for (i64 k = 0; k < n; k++) acc += kFrameRescale * (double)kFramePerToken[k % 5];
	return acc;
}

// The frame grid of one latent frame at (h, w): the (h/2) x (w/2) patch rows,
// row-major, area-normalised. `w_axis` receives the width axis itself, which the
// audio grid of a reference video needs (`_frame_grid` returns it).
std::vector<double> frame_grid(i64 h, i64 w, std::vector<double>* w_axis) {
	const double sqrt_area = std::sqrt((double)h * (double)w);
	const std::vector<double> ah = axis_from_sqrt_area(h, 2, sqrt_area);
	const std::vector<double> aw = axis_from_sqrt_area(w, 2, sqrt_area);
	if (w_axis) *w_axis = aw;
	std::vector<double> out((size_t)ah.size() * aw.size() * 2);
	for (size_t y = 0; y < ah.size(); y++)
		for (size_t x = 0; x < aw.size(); x++) {
			out[(y * aw.size() + x) * 2 + 0] = ah[y];
			out[(y * aw.size() + x) * 2 + 1] = aw[x];
		}
	return out;
}

GpuAlloc sl(const GpuAlloc& a, u64 off_bytes, u64 bytes) {
	GpuAlloc r = a;
	r.off += off_bytes;
	r.bytes = bytes;
	return r;
}

// ── torch's CPU normal distribution (the reference's condition noise) ──────
//
// `_cond_video_rows` augments every reference video row with
//
//     r = aug * r + (1 - aug) * noise,  noise = torch.randn(r.shape, generator=g)
//
// where `g = torch.Generator("cpu").manual_seed(seed)` is restarted for *every*
// condition (so the field is a deterministic function of the row shape). Matching
// it needs torch's own arithmetic, not "a" normal generator: ATen's
// `normal_kernel` (aten/src/ATen/native/cpu/DistributionTemplates.h) fills the
// tensor with `uniform_real_distribution<float>` draws first and then Box-Mullers
// it in place 16 values at a time, with `theta = 2*pi*u2` from the *second* half
// and `radius = sqrt(-2*log(1-u1))` from the first. The uniform draw itself is
// `(mt19937_word % 2^24) / 2^24` (uniform_int_from_to_distribution's modulo
// path), *not* the word over 2^32. All of that is reproduced below, in float32,
// so the engine's reference rows are bit-comparable with ComfyUI's; see
// A ULP check against torch is what pinned it.
struct TorchRandn {
	std::mt19937 eng;
	bool have_spare = false;
	double spare = 0.0;
	explicit TorchRandn(u64 seed) : eng((u32)seed) {}

	// torch's `uniform_real_distribution<float>`: a 32-bit mt19937 draw reduced
	// modulo 2^24 (uniform_int_from_to_distribution's small-range branch), *not*
	// the word over 2^32.
	float uniform() { return (float)(eng() % (1u << 24)) * (1.0f / (float)(1u << 24)); }
	// torch's `random64()`, and the double uniform that rides on it.
	u64 random64() { return ((u64)eng() << 32) | (u64)eng(); }
	double uniform_double() { return (double)(random64() % (1ull << 53)) * (1.0 / (double)(1ull << 53)); }

	// `normal_fill_16`: Box-Muller over a block of 16 uniforms. The first eight
	// carry `1-u` (the radius), the last eight the angle.
	static void box16(float* d) {
		for (i64 j = 0; j < 8; j++) {
			const float u1 = 1.0f - d[j];   // [0,1) -> (0,1] for the log
			const float u2 = d[j + 8];
			const float radius = std::sqrt(-2.0f * std::log(u1));
			const float theta = (float)(2.0 * 3.14159265358979323846 * (double)u2);
			d[j] = radius * std::cos(theta);
			d[j + 8] = radius * std::sin(theta);
		}
	}

	// The `cpu_serial_kernel` path torch takes for fewer than 16 values: one
	// `at::normal_distribution<double>` per element, whose spare is cached in the
	// generator. Kept so the helper is total; the DiT never asks for < 16.
	float normal_small() {
		if (have_spare) {
			have_spare = false;
			return (float)spare;
		}
		const double u1 = uniform_double(), u2 = uniform_double();
		const double r = std::sqrt(-2.0 * std::log1p(-u2));
		const double theta = 2.0 * 3.14159265358979323846 * u1;
		spare = r * std::sin(theta);
		have_spare = true;
		return (float)(r * std::cos(theta));
	}

	void fill(float* data, i64 n) {
		if (n <= 0) return;
		if (n < 16) {
			for (i64 i = 0; i < n; i++) data[i] = normal_small();
			return;
		}
		for (i64 i = 0; i < n; i++) data[i] = uniform();
		for (i64 i = 0; i + 16 <= n; i += 16) box16(data + i);
		if (n % 16) {   // torch recomputes the last 16 values from fresh draws
			const i64 s0 = n - 16;
			for (i64 i = 0; i < 16; i++) data[s0 + i] = uniform();
			box16(data + s0);
		}
	}
};

// ── the buffers one forward needs ──────────────────────────────────────────

struct Fwd {
	Impl* im = nullptr;
	GpuCtx* g = nullptr;
	const AvDitConfig* cfg = nullptr;
	Layout ly;
	i64 chunk = 0;
	GpuAlloc hidden, attnout, qkv_out, q, k, v, rope, pos, norm, q8, s8, ffn, sw, down, proj,
	    mod, prow;
	GpuAlloc vrows, arows;
	GpuAlloc tqkv, tattn, tproj, tffn, tsw;
	// The runtime LoRA correction's [chunk, rank] hidden form. One buffer for the
	// whole run, because its rank is a property of the loaded LoRA files and the
	// chunk, not of the module being corrected - and a per-call allocation would
	// be one per adapted projection per segment per chunk (hundreds of MB a
	// block, all of it activation budget).
	GpuAlloc lora_h;
	// The token refiner keeps its own fp32 q/k/v: its own attention runs on the
	// fp32 kernel (it is text-length, a few hundred rows, where the tiled kernel's
	// setup is not worth it), while the DiT blocks run fp16 through attn_tiled.
	GpuAlloc tq, tk, tv;
	// Called when a charge is refused with the resident window up: it hands the
	// window back and returns true, so the refused stream can be retried against the
	// memory it freed. Set by forward_refs; empty on the paths that never stream.
	std::function<bool()> shed_resident;

	GpuAlloc hrow(i64 a, i64 n) const {
		return sl(hidden, (u64)a * cfg->hidden * 4, (u64)n * cfg->hidden * 4);
	}
	// modulation vector `c` of the segment's row
	GpuAlloc modv(const Seg& sg, i64 c) const {
		return sl(mod, (u64)((sg.chunk0 + c) * cfg->hidden) * 4, (u64)cfg->hidden * 4);
	}
};

void elem(GpuCtx& g, ElemOp op, const GpuAlloc& a, const GpuAlloc& b, const GpuAlloc& c,
          const GpuAlloc& y, i64 rows, i64 cols, ElemMode bm = ElemMode::Flat,
          ElemMode cm = ElemMode::Flat) {
	ElemArgs e;
	e.op = op;
	e.a = a;
	e.b = b;
	e.c = c;
	e.y = y;
	e.rows = rows;
	e.cols = cols;
	e.bMode = bm;
	e.cMode = cm;
	dispatch_elem(*g.ctx, e);
}

void rms(GpuCtx& g, const GpuAlloc& x, const GpuAlloc& w, const GpuAlloc& y, i64 rows, i64 cols,
         float eps) {
	NormArgs n;
	n.x = x;
	n.w = w;
	n.y = y;
	n.rows = rows;
	n.cols = cols;
	n.eps = eps;
	n.affine = true;
	dispatch_norm(*g.ctx, n);
}

// One block projection, through whichever path the weight's own precision
// implies: int8 (verbatim tensorwise or a packed family expanded on device)
// rotates + quantises the activation and runs the dp4a / tensor-core int8 GEMM;
// a float source (f16 / bf16 / f32) runs the dense fp16 GEMM with no conversion.
void quant_gemm(GpuCtx& g, const Fwd& F, const GpuAlloc& x, i64 m, i64 k, const QuantLinear& w,
                const GpuAlloc& out) {
	linear_gemm(g, w, x, m, k, out, F.q8, F.s8);
}

void gemm_dense(GpuCtx& g, const GpuAlloc& a, const GpuAlloc& b, const GpuAlloc& c, i64 m, i64 n,
                i64 k, bool b_bf16, bool b_f32, const GpuAlloc& bias) {
	GemmF16Args ga;
	ga.a = a;
	ga.b = b;
	ga.c = c;
	ga.bias = bias;
	ga.m = m;
	ga.n = n;
	ga.k = k;
	ga.a_is_f32 = true;
	ga.b_is_bf16 = b_bf16;
	ga.b_is_f32 = b_f32;
	ga.has_bias = (bool)bias;
	dispatch_gemm_f16(*g.ctx, ga);
}

// ── the modulation table, evaluated on the host ────────────────────────────
//
// out[(m*modalities + j)*chunks + c] is a hidden-wide vector: modality j,
// modulation c, timestep t_vals[m]. The weight rows are laid out (j, c, d) — the
// reference's `view(M*modalities, expand*hidden).chunk(expand)` — hence the
// modality stride of chunks*hidden and the chunk stride of hidden.
void build_mod(SafeTensors& st, const std::string& prefix, i64 modalities, const Layout& ly,
               i64 hidden, std::vector<float>& out) {
	// ── the rank-8 operand pair, decoded once per checkpoint ────────────
	//
	// `build_mod` runs once per block per step - fifty times a step, plus the
	// final layer - and the two tensors behind it are 1.7 MB of fp16 sitting at
	// an arbitrary offset in a 20 GB file that is being streamed through at the
	// same time. Reading them through the mapping on every call is a page fault
	// per 4 KB (0.6-0.9 GB/s measured, and it evicts the streaming window), and
	// converting 193k halves to fp32 each time buys nothing: it is the same
	// 1.7 MB all run. Decoded once and kept, it is 3.5 MB per block - ~175 MB for
	// the whole 50-block stack, against the ~7 GB of streamed weights the same
	// run already holds in host RAM.
	//
	// Keyed on the reader as well as the name: two models can each have a
	// `blocks.0.adaln_proj.linear`, and a cache that conflated them would feed one
	// model the other's modulation.
	static std::map<std::pair<const void*, std::string>, std::pair<std::vector<float>, std::vector<float>>> cache;
	static std::mutex cache_mu;
	const std::vector<float>* Wp = nullptr;
	const std::vector<float>* Bp = nullptr;
	{
		std::lock_guard<std::mutex> lk(cache_mu);
		auto& e = cache[{&st, prefix}];
		if (e.first.empty()) {
			e.first = tensor_to_f32(st, st.require(prefix + ".weight"));
			e.second = tensor_to_f32(st, st.require(prefix + ".bias"));
		}
		Wp = &e.first;
		Bp = &e.second;
	}
	const std::vector<float>& W = *Wp;
	const std::vector<float>& B = *Bp;
	const i64 chunks = (i64)(W.size() / 8) / (modalities * hidden);
	out.assign((size_t)(ly.m_count * modalities * chunks * hidden), 0.0f);
	// One task per (modality, chunk) row: `m` only picks which of the few 8-wide
	// timestep embeddings the row is dotted with, so the rows are independent and
	// the output space is split exactly as the serial loop split it. Each row is
	// `hidden` independent 8-term dots, which is what makes this worth
	// parallelising at all - it is 4.6 MFLOP a block in scalar host code, and the
	// GPU has nothing to do while it runs.
	const i64 row_count = ly.m_count * modalities * chunks;
	kernels::parallel_for(row_count, [&](i64 idx) {
		const i64 m = idx / (modalities * chunks);
		const i64 rem = idx % (modalities * chunks);
		const i64 j = rem / chunks;
		const i64 c = rem % chunks;
		const float* te = &ly.t_emb[(size_t)(m * 8)];
		const i64 r0 = (j * chunks + c) * hidden;
		const float* Wr = W.data() + (size_t)r0 * 8;
		const float* Br = B.data() + r0;
		float* o = out.data() + (size_t)idx * hidden;
		for (i64 d = 0; d < hidden; d++) {
			const float* wr = Wr + (size_t)d * 8;
			float a = 0.0f;
			for (i64 k = 0; k < 8; k++) a += wr[k] * te[k];
			o[d] = a + Br[d];
		}
	});
}

// ── embedding: text (condition_proj + token_refiner), audio, video ─────────

void refiner_block(Fwd& F, const RefinerW& R, i64 L, i64 which) {
	GpuCtx& g = *F.g;
	const i64 hidden = F.cfg->hidden;
	GpuAlloc x = F.hrow(0, L);
	GpuAlloc nn = sl(F.norm, 0, (u64)L * hidden * 4);

	rms(g, x, R.norm1, nn, L, hidden, F.cfg->norm_eps);
	dump_file(&g, nn, L * hidden, ("rf" + std::to_string(which) + "_n1").c_str(), false);
	gemm_dense(g, nn, R.qkv, F.tqkv, L, kQkvOut, hidden, true, false, GpuAlloc{});
	apply_lora_tail(g, R.tqkv, nn, L, F.tqkv, F.lora_h);
	dump_file(&g, F.tqkv, L * kQkvOut, ("rf" + std::to_string(which) + "_qkv").c_str(), false);
	{
		// RefinerBlock's attention has no rope (`rope_freqs=None`), only the
		// per-head q/k RMSNorm; rot=0 turns the rope off in the shared kernel.
		H3QkvPrepArgs qp;
		qp.qkv = F.tqkv;
		qp.wq = R.q_norm;
		qp.wk = R.k_norm;
		qp.rope = R.q_norm;   // never read when rot == 0
		qp.q = F.tq;
		qp.k = F.tk;
		qp.v = F.tv;
		qp.f32_out = true;
		qp.s = L;
		qp.heads = F.cfg->n_heads;
		qp.head_dim = F.cfg->head_dim;
		qp.rot = 0;
		qp.eps = F.cfg->qk_norm_eps;
		qp.scale = 1.0f;
		dispatch_h3_qkv_prep(*g.ctx, qp);
		std::string t = "rf" + std::to_string(which);
		dump_file(&g, F.tq, L * kInner, (t + "_q").c_str(), false);
		dump_file(&g, F.tk, L * kInner, (t + "_k").c_str(), false);
		dump_file(&g, F.tv, L * kInner, (t + "_v").c_str(), false);
	}
	{
		H3AttnArgs aa;
		aa.q = F.tq;
		aa.k = F.tk;
		aa.v = F.tv;
		aa.o = F.tattn;
		aa.sk = L;
		aa.q0 = 0;
		aa.sq = L;
		aa.heads = F.cfg->n_heads;
		aa.head_dim = F.cfg->head_dim;
		aa.scale = 1.0f / std::sqrt((float)F.cfg->head_dim);
		dispatch_h3_attn(*g.ctx, aa);
		std::string t2 = "rf" + std::to_string(which);
		dump_file(&g, F.tattn, L * kInner, (t2 + "_attn").c_str(), false);
	}
	gemm_dense(g, F.tattn, R.out, F.tproj, L, hidden, kInner, true, false, GpuAlloc{});
	apply_lora_tail(g, R.tout, F.tattn, L, F.tproj, F.lora_h);
	std::string t3 = "rf" + std::to_string(which);
	dump_file(&g, F.tproj, L * hidden, (t3 + "_o").c_str(), false);
	elem(g, ElemOp::Add, x, F.tproj, GpuAlloc{}, x, L, hidden);
	dump_file(&g, x, L * hidden, (t3 + "_x1").c_str(), false);

	rms(g, x, R.norm2, nn, L, hidden, F.cfg->norm_eps);
	gemm_dense(g, nn, R.fc1, F.tffn, L, 2 * kFfn, hidden, true, false, GpuAlloc{});
	apply_lora_tail(g, R.tfc1, nn, L, F.tffn, F.lora_h);
	dump_file(&g, F.tffn, L * 2 * kFfn, (t3 + "_ff").c_str(), false);
	// [L, 2*ffn] fused gate: the pairing needs a row stride (see h3_kernels.hpp).
	dispatch_silu_gate_fused(*g.ctx, F.tffn, F.tsw, L, kFfn);
	gemm_dense(g, F.tsw, R.fc2, F.tproj, L, hidden, kFfn, true, false, GpuAlloc{});
	apply_lora_tail(g, R.tfc2, F.tsw, L, F.tproj, F.lora_h);
	dump_file(&g, F.tproj, L * hidden, (t3 + "_down").c_str(), false);
	elem(g, ElemOp::Add, x, F.tproj, GpuAlloc{}, x, L, hidden);
}

// patchify_video on the host: [c, t, h/2, w/2, 2, 2] -> row (t*th + h)*tw + w,
// lanes c*4 + p*2 + q. `dst` receives n_rows*kVidPatch floats.
void patchify_video_into(float* dst, const float* lat, i64 vt, i64 lh, i64 lw) {
	const i64 th = lh / 2, tw = lw / 2;
	for (i64 tt = 0; tt < vt; tt++)
		for (i64 hh = 0; hh < th; hh++)
			for (i64 ww = 0; ww < tw; ww++) {
				float* row = dst + (size_t)(((tt * th) + hh) * tw + ww) * kVidPatch;
				for (i64 c = 0; c < 24; c++)
					for (i64 p = 0; p < 2; p++)
						for (i64 q = 0; q < 2; q++)
							row[c * 4 + p * 2 + q] =
							    lat[(size_t)(((c * vt + tt) * lh + (hh * 2 + p)) * lw + (ww * 2 + q))];
			}
}

// pack_audio on the host: [c, ch, t] -> row (ch*t40 + t), lanes c.
void pack_audio_into(float* dst, const float* lat, i64 t40) {
	for (i64 ch = 0; ch < 2; ch++)
		for (i64 tt = 0; tt < t40; tt++)
			for (i64 c = 0; c < 32; c++)
				dst[(size_t)((ch * t40 + tt) * kAudPatch + c)] =
				    lat[(size_t)((c * 2 + ch) * t40 + tt)];
}

void embed_rows(Fwd& F, const float* x_v, const float* x_a, const float* embed, i64 L,
                i64 text_dim, float cond_noise_aug, u64 noise_seed) {
	GpuCtx& g = *F.g;
	Layout& ly = F.ly;
	const i64 hidden = F.cfg->hidden;

	// video rows, one patch-projection group per packed segment that carries them.
	// With no references this is a single group: the target video segment, at the
	// same destination and with the same operands as before ref2va existed.
	for (const VRowSrc& vs : ly.vrows) {
		std::vector<float> vp((size_t)vs.n * kVidPatch);
		patchify_video_into(vp.data(), vs.lat, vs.vt, vs.lh, vs.lw);
		if (vs.kind != KVideo && cond_noise_aug < 1.0f) {
			// The reference's _cond_video_rows: `r = aug*r + (1-aug)*noise` with a
			// *fresh* generator per condition (the same stream every time).
			std::vector<float> noise(vp.size());
			TorchRandn rng(noise_seed);
			rng.fill(noise.data(), (i64)noise.size());
			const float a = cond_noise_aug, b = 1.0f - cond_noise_aug;
			for (size_t i = 0; i < vp.size(); i++) vp[i] = a * vp[i] + b * noise[i];
		}
		g.upload_into(F.prow, vp.data(), (u64)vp.size() * 4);
		gemm_dense(g, F.prow, F.im->dense.vid_patch_w, F.hrow(vs.dst, vs.n), vs.n, hidden,
		           kVidPatch, false, true, F.im->dense.vid_patch_b);
	}
	// audio rows: same shape of loop (the reference's AUDIO_COND_TIMESTEP is 1.0,
	// which makes its augmentation a no-op, so there is none here either).
	for (const ARowSrc& as : ly.arows) {
		std::vector<float> ap((size_t)as.n * kAudPatch);
		pack_audio_into(ap.data(), as.lat, as.rt);
		g.upload_into(F.prow, ap.data(), (u64)ap.size() * 4);
		gemm_dense(g, F.prow, F.im->dense.aud_patch_w, F.hrow(as.dst, as.n), as.n, hidden,
		           kAudPatch, false, true, F.im->dense.aud_patch_b);
	}
	// text rows: condition_proj (bf16) then the token refiner
	{
		std::vector<float> tx((size_t)L * text_dim);
		memcpy(tx.data(), embed, tx.size() * sizeof(float));
		g.upload_into(F.prow, tx.data(), (u64)tx.size() * 4);
		gemm_dense(g, F.prow, F.im->dense.cond_w, F.hrow(0, L), L, hidden, text_dim, true, false,
		           F.im->dense.cond_b);
		dump_file(&g, F.hrow(0, L), L * hidden, "r_cond", false);
	}
	// The two refiner blocks are 1.5 GB of bf16, so they stream through the
	// per-layer arena and only the last block gets the final norm.
	//
	// They stream on *every* step, which is what the residency plan has to charge
	// for: `loop_fixed` in plan_residency used to hand the window their 770 MB back
	// ("the refiner is released before the window goes up"), and that is true for
	// the first step only. From the second step on, the refiner's chunks are
	// claimed while the window is resident - which is where a run died with
	// "VRAM exhausted: 147.00 MB for image.weights" (147 MB = the refiner's bf16
	// fc2, not a DiT block's 73.5 MB int8 one, i.e. not a charge the mid-loop
	// recovery below could see).
	g.new_layer();
	try {
		load_refiner(F.im->st, g, *g.wa, 0, F.im->ref0, F.im->lora);
	} catch (const std::exception& e) {
		// The refiner's 770 MB of bf16 chunks are the biggest single claim on the
		// weight arena, and they are claimed with the resident window already up
		// (every step but the first). If the live budget has shrunk under us - the
		// WebView2 UI repainting, the desktop compositor, another client - this is
		// where it bites first, and the window is the thing to give back.
		if (!F.shed_resident || !F.shed_resident()) throw;
		fprintf(stderr, "[h3dit] refiner refused (%s); streaming every block instead\n",
		        e.what());
		g.new_layer();
		load_refiner(F.im->st, g, *g.wa, 0, F.im->ref0, F.im->lora);
	}
	g.ctx->submit_if_recording();
	g.ctx->begin();
	refiner_block(F, F.im->ref0, L, 0);
	dump_file(&g, F.hrow(0, L), L * hidden, "r_b0", false);
	g.end_layer();
	g.new_layer();
	try {
		load_refiner(F.im->st, g, *g.wa, 1, F.im->ref1, F.im->lora);
	} catch (const std::exception& e) {
		if (!F.shed_resident || !F.shed_resident()) throw;
		fprintf(stderr, "[h3dit] refiner refused (%s); streaming every block instead\n",
		        e.what());
		g.new_layer();
		load_refiner(F.im->st, g, *g.wa, 1, F.im->ref1, F.im->lora);
	}
	g.ctx->submit_if_recording();
	g.ctx->begin();
	refiner_block(F, F.im->ref1, L, 1);
	dump_file(&g, F.hrow(0, L), L * hidden, "r_b1", false);
	rms(g, F.hrow(0, L), F.im->dense.ref_norm, F.hrow(0, L), L, hidden, F.cfg->norm_eps);
}

// ── one DiT block ─────────────────────────────────────────────────────────

void run_block(Fwd& F, const BlockW& W, i64 block_index) {
	GpuCtx& g = *F.g;
	Layout& ly = F.ly;
	const i64 hidden = F.cfg->hidden;
	const i64 S = ly.s;

	{
		std::vector<float> modh;
		build_mod(F.im->st, "blocks." + std::to_string(block_index) + ".adaln_proj.linear", 3, ly,
		          hidden, modh);
		if ((u64)modh.size() * 4 > F.mod.bytes)
			throw MediaError("h3dit: modulation table larger than its buffer");
		g.upload_into(F.mod, modh.data(), (u64)modh.size() * 4);
	}

	// ── qkv projection + per-head prep, one chunk at a time ────────────────
	//
	// The fused qkv intermediate is one *chunk* tall, not S tall. At 540P/243
	// frames the old [S, 3*7168] fp32 buffer was 3.2 GB of a 4.6 GB budget, so
	// the whole shape was refused before a single weight was read; the packed
	// fp16 q/k/v the projection is consumed into are still S-proportional but
	// only two bytes an element. `dispatch_h3_qkv_prep` therefore takes a row
	// window (qkv_row0 / out_row0) and runs on each chunk as soon as that
	// chunk's projection lands.
	for (const Seg& sg : ly.segs) {
		for (i64 a = sg.a; a < sg.b; a += F.chunk) {
			const i64 n = std::min(F.chunk, sg.b - a);
			{
				ProfScope ps(&g, prof_sink(&H3Prof::qkv_gemm));
				GpuAlloc h = F.hrow(a, n);
				GpuAlloc nn = sl(F.norm, 0, (u64)n * hidden * 4);
				rms(g, h, W.norm1, nn, n, hidden, F.cfg->norm_eps);
				elem(g, ElemOp::ModAdd, nn, F.modv(sg, 1), F.modv(sg, 0), nn, n, hidden,
				     ElemMode::Row, ElemMode::Row);
				quant_gemm(g, F, nn, n, hidden, W.qkv,
				           sl(F.qkv_out, 0, (u64)n * kQkvOut * 4));
				apply_lora_tail(g, W.tqkv, nn, n, sl(F.qkv_out, 0, (u64)n * kQkvOut * 4),
				                F.lora_h);
			}
			{
				ProfScope ps(&g, prof_sink(&H3Prof::qkv_prep));
				H3QkvPrepArgs qp;
				qp.qkv = F.qkv_out;
				qp.wq = W.q_norm;
				qp.wk = W.k_norm;
				qp.rope = F.rope;
				qp.q = F.q;
				qp.k = F.k;
				qp.v = F.v;
				qp.f32_out = false;   // packed fp16 for the tiled attention
				qp.s = n;
				qp.qkv_row0 = 0;
				qp.out_row0 = a;
				qp.heads = F.cfg->n_heads;
				qp.head_dim = F.cfg->head_dim;
				qp.rot = kRot;
				qp.eps = F.cfg->qk_norm_eps;
				qp.scale = 1.0f;
				dispatch_h3_qkv_prep(*g.ctx, qp);
				if (block_index == h3_dbg_block()) {
					dbg_dump(g, sl(F.norm, 0, (u64)n * hidden * 4), n * hidden, block_index,
					         "nn", a);
					dbg_dump(g, sl(F.qkv_out, 0, (u64)n * kQkvOut * 4), n * kQkvOut, block_index,
					         "qkvout", a);
					dbg_dump_f16(g, sl(F.q, (u64)a * kInner * 2, (u64)n * kInner * 2), n * kInner,
					             block_index, "q", a);
					dbg_dump_f16(g, sl(F.k, (u64)a * kInner * 2, (u64)n * kInner * 2), n * kInner,
					             block_index, "k", a);
					dbg_dump_f16(g, sl(F.v, (u64)a * kInner * 2, (u64)n * kInner * 2), n * kInner,
					             block_index, "v", a);
				}
			}
		}
	}
	if (block_index == h3_dbg_block())
		dbg_dump(g, F.mod, (i64)(ly.m_count * 3 * 6 * hidden), block_index, "mod", 0);
	if (block_index == 0)
		dump_stat(F.g, F.qkv_out, (i64)std::min<i64>(S, F.chunk) * kQkvOut, "qkv", 0);
	if (block_index == 0)
		dump_stat(F.g, F.mod, (i64)(ly.m_count * 3 * 6 * hidden), "mod", 0);
	if (block_index == h3_dbg_block()) {
		// Whole-sequence packed q/k/v: the full S rows are what the attention
		// kernel actually reads (every query sees every key), so a per-chunk
		// dump cannot rule the inputs in or out.
		dbg_dump_f16(g, F.q, (i64)S * kInner, block_index, "Qfull", 0);
		dbg_dump_f16(g, F.k, (i64)S * kInner, block_index, "Kfull", 0);
		dbg_dump_f16(g, F.v, (i64)S * kInner, block_index, "Vfull", 0);
	}
}

// ── attention + out_proj + MLP, one query chunk at a time ────────────────
//
// The attention output is a *chunk*-tall buffer, not an S-tall one. Attention
// is per-query-row independent, and the only consumer of its output is this
// block's out_proj, which already ran one chunk at a time - so nothing needs
// more than one chunk of it at once. At 540P/10s the old [S, 7168] fp32 buffer
// was 1.08 GB, which together with q/k/v (1.61 GB) and the residual stream
// (0.81 GB) is what stopped 243 frames from fitting at all.
//
// No kernel change is needed for this: `dispatch_h3_attn` writes its output at
// the *absolute* query row, so the caller passes an `o_row0` and the dispatch
// subtracts that many rows' stride from the output byte offset. The arithmetic
// is unsigned, so the subtraction wraps exactly as intended and the row guard
// (`row < S`) still refers to the input rows.
void run_attn_mlp(Fwd& F, const BlockW& W, i64 block_index) {
	GpuCtx& g = *F.g;
	Layout& ly = F.ly;
	const i64 hidden = F.cfg->hidden;
	const i64 S = ly.s;
	// out_proj + gate_msa, then the MLP - with this chunk's attention computed
	// immediately before it, so no S-tall attention-output buffer exists.
	{
		ProfScope ps(&g, prof_sink(&H3Prof::out_mlp));
		for (const Seg& sg : ly.segs) {
			for (i64 a = sg.a; a < sg.b; a += F.chunk) {
				const i64 n = std::min(F.chunk, sg.b - a);
				GpuAlloc h = F.hrow(a, n);
				GpuAlloc nn = sl(F.norm, 0, (u64)n * hidden * 4);
				{
					ProfScope pa(&g, prof_sink(&H3Prof::attn));
					H3AttnArgs aa;
					// Q is passed as a *row slice* starting at this chunk, and the
					// dispatch keeps q0 = 0 / sq = n. The kernel then indexes
					// queries 0..n-1 of the slice and writes rows 0..n-1 of the
					// output - which is exactly what a chunk-tall output buffer
					// needs. The alternative (q0 = a, and a row origin for the
					// output) does not work here: the kernel adds its byte
					// offsets to a *64-bit* pointer, so a negative output offset
					// cannot wrap, and the write would land a gigabyte past the
					// buffer. `sk` stays the full sequence length, which is what
					// makes the keys and the `row < S` guard correct.
					aa.q = sl(F.q, (u64)a * kInner * 2, (u64)n * kInner * 2);
					aa.k = F.k;
					aa.v = F.v;
					aa.o = F.attnout;
					aa.sk = S;
					aa.q0 = 0;
					aa.sq = n;
					aa.heads = F.cfg->n_heads;
					aa.head_dim = F.cfg->head_dim;
					aa.scale = 1.0f / std::sqrt((float)F.cfg->head_dim);
					aa.fp16_qkv = true;
					dispatch_h3_attn(*g.ctx, aa);
					if (block_index == h3_dbg_block())
						dbg_dump(g, F.attnout, n * kInner, block_index, "attnout", a);
					if (block_index == 0 && sg.a == ly.segs.back().a) {
						dump_stat(F.g, F.attnout, n * kInner, "attn_out", 0);
						dump_file(F.g, F.attnout, n * kInner, "attn", false);
					}
				}
				quant_gemm(g, F, F.attnout, n, kInner, W.out, F.proj);
				apply_lora_tail(g, W.tout, F.attnout, n, F.proj, F.lora_h);
				if (block_index == 0 && sg.a == ly.segs.back().a) {
					dump_stat(F.g, F.proj, (i64)hidden, "out_proj_vid0", 0);
					dump_stat(F.g, F.attnout, (i64)kInner, "attn_vid0_in", 0);
					dump_stat(F.g, F.s8, 4, "s8_row0", 0);
				}
				elem(g, ElemOp::AddMul, h, F.proj, F.modv(sg, 2), h, n, hidden, ElemMode::Flat,
				     ElemMode::Row);
				if (block_index == 0 && sg.a == ly.segs.back().a)
					dump_stat(F.g, F.hrow(sg.a, 1), (i64)hidden, "h_after_attn", 0);

				if (dit_dump_prefix() && block_index == 0 && sg.a == ly.segs.back().a) {
					F.g->ctx->submit_if_recording(); F.g->ctx->begin();
					const std::vector<float> dv = F.g->download_f32(F.hrow(sg.a, 1), (u64)hidden);
					double mn = 1e30, mx = -1e30; for (float x : dv) { mn = std::min(mn, (double)x); mx = std::max(mx, (double)x); }
					fprintf(stderr, "[h3dit] block0 after-attn h[first video row] [%.4f, %.4f]%c", mn, mx, 10);
					// download() leaves the command list closed; the next dispatch needs one.
					F.g->ctx->begin();
				}
				rms(g, h, W.norm2, nn, n, hidden, F.cfg->norm_eps);
				elem(g, ElemOp::ModAdd, nn, F.modv(sg, 4), F.modv(sg, 3), nn, n, hidden,
				     ElemMode::Row, ElemMode::Row);
				quant_gemm(g, F, nn, n, hidden, W.fc1, F.ffn);
				apply_lora_tail(g, W.tfc1, nn, n, F.ffn, F.lora_h);

				// [n, 2*ffn] fused gate; `elem` would pair the halves flat and be
				// right only for the first row (see h3_kernels.hpp).
				dispatch_silu_gate_fused(*g.ctx, F.ffn, F.sw, n, kFfn);
				quant_gemm(g, F, F.sw, n, kFfn, W.fc2, F.down);
				apply_lora_tail(g, W.tfc2, F.sw, n, F.down, F.lora_h);
				elem(g, ElemOp::AddMul, h, F.down, F.modv(sg, 5), h, n, hidden, ElemMode::Flat,
				     ElemMode::Row);
				if (dit_dump_prefix() && block_index == 0 && sg.a == ly.segs.back().a) {
					F.g->ctx->submit_if_recording();
					F.g->ctx->begin();
					const std::vector<float> dv = F.g->download_f32(F.hrow(sg.a, 1), (u64)hidden);
					double mn = 1e30, mx = -1e30;
					for (float x : dv) {
						mn = std::min(mn, (double)x);
						mx = std::max(mx, (double)x);
					}
					fprintf(stderr, "[h3dit] block0 after-mlp  h[first video row] [%.4f, %.4f]%c", mn,
					        mx, 10);
					F.g->ctx->begin();
				}
			}
		}
	}
}

// ── final layer ───────────────────────────────────────────────────────────

void run_final(Fwd& F) {
	GpuCtx& g = *F.g;
	Layout& ly = F.ly;
	const i64 hidden = F.cfg->hidden;
	std::vector<float> fmod;
	build_mod(F.im->st, "final_layer.adaln_proj.linear", 1, ly, hidden, fmod);
	if ((u64)fmod.size() * 4 > F.mod.bytes) throw MediaError("h3dit: final modulation overflow");
	g.upload_into(F.mod, fmod.data(), (u64)fmod.size() * 4);

	struct Part {
		const Seg* sg;
		i64 m;
		GpuAlloc rows;
		i64 width;
		GpuAlloc w;
		GpuAlloc bias;
	};
	const Seg& vs = ly.segs[(size_t)ly.video_seg];
	const Seg& as = ly.segs[(size_t)ly.audio_seg];
	Part parts[2] = {
	    {&vs, ly.video_m, F.vrows, kVidPatch, F.im->dense.vid_out_w, F.im->dense.vid_out_b},
	    {&as, ly.audio_m, F.arows, kAudPatch, F.im->dense.aud_out_w, F.im->dense.aud_out_b},
	};
	for (const Part& p : parts) {
		for (i64 a = p.sg->a; a < p.sg->b; a += F.chunk) {
			const i64 n = std::min(F.chunk, p.sg->b - a);
			GpuAlloc x = F.hrow(a, n);
			GpuAlloc nn = sl(F.norm, 0, (u64)n * hidden * 4);
			rms(g, x, F.im->dense.final_norm, nn, n, hidden, F.cfg->norm_eps);
			// FinalLayer chunks (shift, scale) with modalities = 1
			GpuAlloc sc = sl(F.mod, (u64)((p.m * 2 + 1) * hidden) * 4, (u64)hidden * 4);
			GpuAlloc sh = sl(F.mod, (u64)((p.m * 2 + 0) * hidden) * 4, (u64)hidden * 4);
			elem(g, ElemOp::ModAdd, nn, sc, sh, nn, n, hidden, ElemMode::Row, ElemMode::Row);
			gemm_dense(g, nn, p.w, sl(p.rows, (u64)(a - p.sg->a) * p.width * 4,
			                          (u64)n * p.width * 4),
			           n, p.width, hidden, false, true, p.bias);
		}
	}
}


// The checkpoint's timestep embedding: `adaln_t_table` [1025, 8] linearly
// interpolated at `t` in [0, 1]. Used by the per-step layout builder to feed
// the modulation table for each timestep.
void adaln_table_row(const float* table, float t, float* out8) {
	constexpr i64 grid = 1025;
	const float tv = std::min(1.0f, std::max(0.0f, t));
	const float pos = tv * (float)(grid - 1);
	const i64 i0 = std::clamp<i64>((i64)std::floor(pos), 0, grid - 2);
	const float frac = pos - (float)i0;
	for (i64 k = 0; k < 8; k++) {
		const float a = table[(size_t)(i0 * 8 + k)];
		const float b = table[(size_t)((i0 + 1) * 8 + k)];
		out8[k] = a + frac * (b - a);
	}
}


}  // namespace

// ── open ──────────────────────────────────────────────────────────────────

void AvDiT::open(const std::string& dit_path, GpuCtx* gpu) {
	if (!gpu || !gpu->ok()) throw MediaError("h3dit: open() needs a live GpuCtx");
	g_ = gpu;
	dir_ = path_dirname(dit_path);

	auto im = std::make_unique<Impl>();
	const std::string path = dit_path;
	im->st.open(path);
	SafeTensors& st = im->st;
	im->lora = loras_;

	const StTensor& vpp = st.require("video_patch_proj.weight");
	const StTensor& qkv = st.require("blocks.0.attn.qkv_proj.weight");
	const StTensor& fc1 = st.require("blocks.0.mlp.fc1.weight");
	const StTensor& cond = st.require("condition_proj.weight");
	const StTensor& table = st.require("adaln_t_table");
	const StTensor& vhead = st.require("final_layer.video_out.weight");

	cfg_.hidden = vpp.shape[0];
	cfg_.v_channels = 24;
	cfg_.patch_t = 1;
	cfg_.patch_h = 2;
	cfg_.patch_w = 2;
	cfg_.n_heads = 56;
	cfg_.head_dim = 128;
	cfg_.n_kv_heads = 56;
	cfg_.ffn_hidden = fc1.shape[0] / 2;
	cfg_.text_dim = cond.shape[1];
	cfg_.n_layers = 0;
	while (st.find("blocks." + std::to_string(cfg_.n_layers) + ".norm1.weight")) cfg_.n_layers++;
	if (cfg_.n_layers != 50) throw MediaError("h3dit: expected 50 blocks");
	if (qkv.shape[0] != 3 * cfg_.n_heads * cfg_.head_dim)
		throw MediaError("h3dit: qkv_proj is not 3 x 56 x 128");
	if (table.shape[0] != 1025 || table.shape[1] != 8)
		throw MediaError("h3dit: unexpected adaln_t_table shape");
	if (vhead.shape[0] != kVidPatch) throw MediaError("h3dit: video head is not 24x1x2x2");

	{
		// One block's weights, *in load_block's allocation order*, priced from the
		// checkpoint's shapes rather than its stored byte counts: the upload is
		// always an int8 matrix (1 byte/element) plus one fp32 scale per row.
		//
		// The order is the order `load_block` calls `load_linear_pair` and `up_f32` in,
		// because the charge is per arena *chunk* and a bump allocator books a
		// different total for the same tensors in a different order.
		//
		// The price is the *file's own precision*: an int8 / packed matrix costs one
		// byte per element plus one fp32 scale per row; a float matrix costs its own
		// element size (f16 / bf16 / fp8 / nvfp4 -> 2, f32 -> 4) and no scale. A
		// checkpoint streamed at a wider precision therefore keeps fewer blocks
		// resident - the honest number.
		static const char* kMatrixOrder[] = {"attn.qkv_proj", "attn.out_proj", "mlp.fc1",
		                                     "mlp.fc2"};
		static const char* kVecOrder[] = {"norm1.weight", "norm2.weight", "attn.q_norm.weight",
		                                  "attn.k_norm.weight"};
		im->layer_tensors.clear();
		u64 b = 0;
		for (const char* name : kMatrixOrder) {
			const StTensor& t = st.require(std::string("blocks.0.") + name + ".weight");
			if (t.shape.size() != 2)
				throw MediaError(std::string("h3dit: blocks.0.") + name + " is not 2-D");
			const u64 rows = (u64)t.shape[0];
			const u64 cols = (u64)t.shape[1];
			if (weight_is_dense_float(st, t)) {
				const u64 esz = weight_precision_of(st, t) == WeightPrecision::F32 ? 4u : 2u;
				im->layer_tensors.push_back(rows * cols * esz);
				b += rows * cols * esz;
			} else {
				im->layer_tensors.push_back(rows * cols);   // the int8 upload
				im->layer_tensors.push_back(rows * 4);      // its per-row fp32 scale
				b += rows * cols + rows * 4;
			}
		}
		for (const char* name : kVecOrder) {
			const StTensor& t = st.require(std::string("blocks.0.") + name);
			const u64 nb = (u64)t.numel * 4;
			im->layer_tensors.push_back(nb);
			b += nb;
		}
		im->layer_bytes = b;

		// The token refiner's block, the other thing that streams through the same
		// arena while the resident window is up (`load_refiner`'s order).
		static const char* kRefOrder[] = {"attn.qkv_proj.weight", "attn.out_proj.weight",
		                                  "mlp.fc1.weight",      "mlp.fc2.weight",
		                                  "norm1.weight",        "norm2.weight",
		                                  "attn.q_norm.weight",  "attn.k_norm.weight"};
		im->refiner_tensors.clear();
		for (const char* n : kRefOrder) {
			const StTensor& t = st.require(std::string("token_refiner.blocks.0.") + n);
			const bool matrix = (strstr(n, "norm") == nullptr);
			im->refiner_tensors.push_back(matrix ? (u64)t.numel * 2 : (u64)t.numel * 4);
		}

		// File ranges (what the host-side weight cache pins / the streamer prefetches).
		im->block_ranges.assign((size_t)cfg_.n_layers, {});
		for (i64 i = 0; i < cfg_.n_layers; i++) {
			const std::string p = "blocks." + std::to_string(i) + ".";
			for (const char* name : kMatrixOrder) {
				const StTensor& t = st.require(p + name + ".weight");
				im->block_ranges[(size_t)i].push_back({t.offset, t.nbytes});
			}
			for (const char* name : kVecOrder) {
				const StTensor& t = st.require(p + name);
				im->block_ranges[(size_t)i].push_back({t.offset, t.nbytes});
			}
		}
		im->refiner_ranges.clear();
		for (i64 bb = 0; bb < 2; bb++) {
			const std::string p = "token_refiner.blocks." + std::to_string(bb) + ".";
			for (const char* n : kRefOrder) {
				const StTensor& t = st.require(p + n);
				im->refiner_ranges.push_back({t.offset, t.nbytes});
			}
		}
	}

	// Per-block byte ranges, for the prefetch the streaming loop issues one block
	// ahead of itself.
	{
		im->layer_range.resize((size_t)cfg_.n_layers);
		for (i64 i = 0; i < cfg_.n_layers; i++) {
			const std::string pre = "blocks." + std::to_string(i) + ".";
			u64 lo = ~0ull, hi = 0;
			for (const StTensor& t : st.tensors())
				if (t.name.size() > pre.size() && t.name.compare(0, pre.size(), pre) == 0) {
					lo = std::min(lo, t.offset);
					hi = std::max(hi, t.offset + t.nbytes);
				}
			im->layer_range[(size_t)i] = {lo == ~0ull ? 0 : lo, hi};
		}
	}


	auto& K = *gpu->keep;
	im->dense.cond_w = up_raw(st, "condition_proj.weight", *gpu, K, DType::BF16);
	im->dense.cond_b = up_f32(st, "condition_proj.bias", *gpu, K);     // the GEMM reads fp32
	im->dense.vid_patch_w = up_raw(st, "video_patch_proj.weight", *gpu, K, DType::F32);   // fp32
	im->dense.vid_patch_b = up_raw(st, "video_patch_proj.bias", *gpu, K, DType::F32);
	im->dense.aud_patch_w = up_raw(st, "audio_patch_proj.weight", *gpu, K, DType::F32);
	im->dense.aud_patch_b = up_raw(st, "audio_patch_proj.bias", *gpu, K, DType::F32);
	im->dense.ref_norm = up_f32(st, "token_refiner.final_norm.weight", *gpu, K);
	im->dense.final_norm = up_f32(st, "final_layer.norm.weight", *gpu, K);
	im->dense.final_w = up_raw(st, "final_layer.adaln_proj.linear.weight", *gpu, K, DType::F16);  // f16
	im->dense.final_b = up_raw(st, "final_layer.adaln_proj.linear.bias", *gpu, K, DType::F16);
	im->dense.vid_out_w = up_raw(st, "final_layer.video_out.weight", *gpu, K, DType::F32);   // fp32
	im->dense.vid_out_b = up_raw(st, "final_layer.video_out.bias", *gpu, K, DType::F32);
	im->dense.aud_out_w = up_raw(st, "final_layer.audio_out.weight", *gpu, K, DType::F32);
	im->dense.aud_out_b = up_raw(st, "final_layer.audio_out.bias", *gpu, K, DType::F32);

	im->res_arena.init(gpu->ctx, 192ull << 20);
	im->res_arena.set_tag("h3dit.resident");


	registry()[this] = std::move(im);
	open_ = true;
}

void AvDiT::release_weights() {
	Impl* im = impl_of(this);
	if (!im) return;
	im->res.clear();
	im->res_arena.release_chunks();
	im->res_n = 0;
	im->res_pending = false;
	im->planned = false;
	res_.res_main_n = 0;
	res_.planned = false;
	if (g_ && g_->wa) g_->wa->release_chunks();
	if (g_ && g_->aa) g_->aa->release_chunks();
	// The host-side weight cache goes with the device side, and for the same
	// reason: the sampling loop is over, the phase that follows (the video VAE's
	// 100-odd tiles) re-reads its own 5 GB checkpoint, and holding ~7 GB of system
	// RAM pinned for weights nobody will ask for again is what starved that phase.
	// Measured: the VAE's tile loop went from 774 ms to 1.55 s per tile because
	// `pin_range` refused the 1.25 GB of decoder blocks it wanted for want of a
	// free half of system memory - 70 s of an 8 minute run. A second generation
	// call re-pins: `forward` fills the cache on the first step either way, and
	// the file pages it reads are already in the OS cache.
	if (im->st.pinned_bytes()) im->st.drop_cache();
}

// ── estimates (plan §4) ───────────────────────────────────────────────────

u64 AvDiT::estimate_h3_dit_activation_bytes(i64 S, i64 n_video, i64 n_audio, i64 T_latent,
                                                   i64 H, i64 W, i64 chunk_hint) {
	(void)n_audio;
	(void)T_latent;
	(void)H;
	(void)W;
	if (S <= 0) return 0;
	const u64 f = 4;
	const u64 chunk = (u64)std::min<i64>(chunk_hint, std::max<i64>(S, 1));
	u64 b = 0;
	b += (u64)S * 5376 * f;                              // hidden stream (fp32)
	b += chunk * (u64)kQkvOut * f;                       // fused qkv: ONE CHUNK tall
	b += 3ull * (u64)S * kInner * 2;                     // q, k, v: packed fp16
	b += chunk * (u64)kInner * f;                        // attention output (fp32), one chunk
	b += (u64)S * kRopePairs * 2 * f + (u64)S * 3 * f;   // rope table + positions
	b += chunk * (2 * kFfn + kFfn + 3 * 5376) * f;       // fc1 out + swiglu + norm/down/proj
	b += chunk * kFfn + chunk * 4;                       // int8 activation + row scale
	b += (u64)std::max<i64>(n_video, 1) * 128 * f;       // patchified rows
	b += 64ull * 1024 * 1024;                            // norms, modulation, misc
	// 20% safety margin (plan §4: "先留 20% 安全余量").
	return b + b / 5;
}

u64 AvDiT::plan_activation(i64 S, i64 n_video, i64 n_audio, i64 T_latent, i64 H, i64 W,
                                  u64 budget, u64 overhead, i64* chunk_out, i64 refiner_rows,
                                  i64 ref_patch_rows) {
	(void)T_latent;
	(void)H;
	(void)W;	// The *exact* activation cost: the arena's own chunk arithmetic over the
	// allocations `forward()` makes, in the order it makes them, with the
	// refiner's scratch claimed under a mark and handed back before the block
	// loop's buffers - see `dit_activation_cost`.
	//
	// It replaces `estimate_h3_dit_activation_bytes` here (that function is kept
	// for the report, where a smooth closed form is what the table wants). The
	// estimate was a formula with a 20 % margin, and the plan carried a *second*,
	// independent reserve on top of it for "the per-pixel buffers the estimate does
	// not name" - two conservative terms over the same arena. At 540P/243 frames
	// they added up to 4.61 GB against a 4.85 GB limit and left the window at zero
	// blocks even though the arena's real frame is ~3.1 GB: the run streamed every
	// one of its 50 blocks for a reservation nobody was using. One exact term
	// prices the same arena without either fudge.
	const i64 rows =
	    refiner_rows > 0 ? std::min<i64>(refiner_rows, (i64)std::max<i64>(S, 1))
	                     : (i64)std::max<i64>(S, 1);
	i64 chunk = kChunk;
	u64 act = dit_activation_cost(S, n_video, n_audio, kTextDim, rows, chunk, ref_patch_rows);
	while (act + overhead > budget && chunk > kChunkMin) {
		const i64 next = std::max<i64>(kChunkMin, chunk / 2);
		if (next == chunk) break;
		chunk = next;
		act = dit_activation_cost(S, n_video, n_audio, kTextDim, rows, chunk, ref_patch_rows);
	}
	if (chunk_out) *chunk_out = chunk;
	return act;
}

// `phase` is the number of rows the refiner runs over (the text span plus the
// packed reference rows). It only appears in the refiner's scratch, which shares
// its region with the block loop's buffers, so any value >= the real one is a
// safe upper bound.
u64 AvDiT::dit_activation_cost(i64 S, i64 n_video, i64 n_audio, i64 text_dim, i64 phase,
                                      i64 chunk_hint, i64 ref_patch_rows) {
	if (S <= 0) return 0;
	const i64 chunk = std::clamp<i64>(chunk_hint, 1, std::max<i64>(S, 1));
	const i64 L = std::clamp<i64>(phase, 1, std::max<i64>(S, 1));
	const i64 hidden = kHidden;

	// `forward()`'s allocation order, read off the forward itself. Three groups:
	//
	//   * the head - the packed stream, the rope/position tables, the patch
	//     staging buffer and the gather rows. Live for the whole step.
	//   * the refiner's scratch, claimed under a mark taken *after* the head and
	//     rewound before the loop's buffers.
	//   * the block loop's buffers, claimed from that mark.
	const i64 stage_floats =
	    std::max<i64>(std::max<i64>(L * text_dim, std::max<i64>(n_video, ref_patch_rows) * kVidPatch),
	                  std::max<i64>(n_audio * kAudPatch, 1));
	// The modulation table's row count is the number of distinct timesteps in the
	// presentation: 2 (video, audio) plus one per reference modality. It is a few
	// hundred KB at any of those counts, so the bound is generous on purpose.
	const i64 m_count = 8;

	GpuArenaModel sim(arena_chunk_bytes_for(vram_budget().limit()));
	sim.alloc((u64)S * hidden * 4);                              // F.hidden
	sim.alloc((u64)S * kRopePairs * 2 * 4);                      // F.rope
	sim.alloc((u64)S * 3 * 4);                                   // F.pos
	sim.alloc((u64)std::max<i64>(chunk, L) * hidden * 4);        // F.norm
	sim.alloc((u64)stage_floats * 4);                            // F.prow
	sim.alloc((u64)std::max<i64>(n_video, 1) * kVidPatch * 4);  // F.vrows
	sim.alloc((u64)std::max<i64>(n_audio, 1) * kAudPatch * 4);  // F.arows
	sim.alloc((u64)std::max<i64>(chunk, L) * (u64)kLoraRankBound * 4);   // F.lora_h
	const GpuArenaModel::Mark m = sim.mark();
	// The refiner's scratch, live only inside `embed_rows`.
	sim.alloc((u64)L * kInner * 4);        // F.tq
	sim.alloc((u64)L * kInner * 4);        // F.tk
	sim.alloc((u64)L * kInner * 4);        // F.tv
	sim.alloc((u64)L * kQkvOut * 4);       // F.tqkv
	sim.alloc((u64)L * kInner * 4);        // F.tattn
	sim.alloc((u64)L * hidden * 4);        // F.tproj
	sim.alloc((u64)L * 2 * kFfn * 4);      // F.tffn
	sim.alloc((u64)L * kFfn * 4);          // F.tsw
	const u64 with_refiner = sim.capacity();
	sim.rewind_to(m);
	// The block loop's buffers, sharing the refiner's region.
	sim.alloc((u64)chunk * kQkvOut * 4);   // F.qkv_out
	sim.alloc((u64)chunk * kInner * 4);    // F.attnout
	sim.alloc((u64)S * kInner * 2);        // F.q
	sim.alloc((u64)S * kInner * 2);        // F.k
	sim.alloc((u64)S * kInner * 2);        // F.v
	sim.alloc((u64)chunk * kFfn);          // F.q8
	sim.alloc((u64)chunk * 4);             // F.s8
	sim.alloc((u64)chunk * 2 * kFfn * 4);  // F.ffn
	sim.alloc((u64)chunk * kFfn * 4);      // F.sw
	sim.alloc((u64)chunk * hidden * 4);    // F.down
	sim.alloc((u64)chunk * hidden * 4);    // F.proj
	sim.alloc((u64)m_count * 3 * 6 * hidden * 4);   // F.mod
	return std::max(with_refiner, sim.capacity());
}

u64 estimate_h3_vae_activation_bytes(i64 H, i64 W, i64 T_frames) {
	// Calibrated against a measured VRAM peak after one decode, minus the
	// 192 MB of resident decoder weights this checkpoint loads at open):
	//
	//   latent 32x20x2  ( 320x512 px, 5 frames) -> 768 MB
	//   latent 60x34x2  ( 960x544 px, 5 frames) -> 768 MB
	//
	// Identical at three times the pixel count, because the decoder is *tiled* at
	// 256 px (kTileSize): the activation peak is set by one tile plus the block
	// buffers that go with it, not by the frame size, and `tiled_decode` now rewinds
	// the activation arena per tile and per block (before that the arena grew to the
	// sum of all 36 blocks — 4.2 GB at the smaller of the two shapes above).
	//
	// The estimate is therefore a tile constant plus this shape's own output clip,
	// with the plan's 20% margin on top.
	const double frames = (double)std::max<i64>(T_frames, 1);
	const double tile_out = 3.0 * 256.0 * 256.0 * frames * 4.0;   // one tile's pixels
	const double out = 3.0 * (double)H * (double)W * frames * 4.0 / 4.0;
	const double fixed = 768.0 * 1024.0 * 1024.0;
	return (u64)(1.2 * std::max(fixed + tile_out, out));
}

u64 AvDiT::layer_bytes() const {
	Impl* im = impl_of(this);
	if (!open_ || !im) return 0;
	return im->layer_bytes;
}

const char* AvDiT::weight_precision() const {
	Impl* im = impl_of(this);
	if (!open_ || !im) return "unknown";
	// The block projections share one precision; the first block's fused q/k/v
	// speaks for the stack (the token refiner streams bf16 regardless).
	const StTensor* t = im->st.find("blocks.0.attn.qkv_proj.weight");
	if (!t) return "unknown";
	return weight_precision_name(weight_precision_of(im->st, *t));
}

void AvDiT::plan_residency(i64 H, i64 W, i64 T_frames, i64 text_tokens, i64 ref_rows,
                                 u64 extra_resident, u64 ref_pixels) {
	Impl* im = impl_of(this);
	if (!open_ || !im) {
		res_.planned = false;
		return;
	}
	const i64 lat_h = H, lat_w = W;
	const i64 video_t = h3_video_latent_t(T_frames);
	const i64 audio_t = h3_audio_latent_t(T_frames);
	const i64 n_video = video_t * (lat_h / cfg_.patch_h) * (lat_w / cfg_.patch_w);
	const i64 n_audio = 2 * audio_t;
	// The reference blocks pack between the text span and the targets, so they add
	// rows to S (and to nothing else: the activation arena is sized by S).
	const i64 S = std::max<i64>(text_tokens, 1) + n_video + n_audio + std::max<i64>(ref_rows, 0);

	res_.planned = true;
	res_.n_layers = cfg_.n_layers;
	res_.s_tokens = S;
	res_.layer_bytes = im->layer_bytes;
	res_.budget = vram_budget().limit();
	res_.act_vae = estimate_h3_vae_activation_bytes(lat_h * 16, lat_w * 16, T_frames);

	// ── what the DiT phase is allowed to keep resident ──────────────────────
	//
	// The plan's formula subtracts `max(dit_act, vae_act)` and calls the two
	// phases non-overlapping. They are — but that is an argument about *time*
	// (the VAEs run after the DiT weights are handed back), and subtracting the
	// larger of the two from the DiT's own budget is the opposite of what it
	// implies: it reserves the video VAE's 0.9 GB while the DiT is sampling,
	// where the number that matters is the DiT's own activation peak. Charging
	// the DiT that 0.9 GB cost two whole resident blocks (7 instead of 9), i.e.
	// 0.7 GB re-read from disk on each of the 16 steps.
	//
	// So: residency is sized against `dit_act`, and the VAE phase gets its own
	// check (its weights stream through the same arenas, and the DiT's resident
	// window is already gone by then).
	//
	// The fixed term is the *peak* of everything that is resident besides the
	// window and the DiT activations, and it has three parts plus one maximum:
	//
	//   keep arena + misc      what the accountant is already holding when the
	//                          plan runs - the patch projections, the heads, the
	//                          modulation table, the caller's reference footprint
	//                          and anything else this process shares with the
	//                          chain. **Measured** (`vram_budget().local()`), not
	//                          named, so it cannot go stale the way a constant
	//                          does;
	//   refiner stream  ~770 MB  the token refiner's bf16 block. `embed_rows`
	//                          re-streams it on *every* step while the window is
	//                          up (not only the first - see above), so it is live
	//                          beside the window in the steady state;
	//   streamed block  ~520 MB  one DiT block flowing through the weight arena,
	//                          priced the way the accountant books it
	//                          (`arena_chunk_cost`, below).
	//
	// The refiner and the streamed block are **sequential inside a step** (the
	// refiner runs in `embed_rows`, the block loop after it), so the larger of the
	// two is the reservation - charging both, as the old `loop_fixed` + `reserve`
	// pair did, reserved a whole extra block-sized arena.
	//
	// The old flat 1.28 GB ("ring + refiner + keep") was wrong in both directions:
	// the upload ring's device side is not charged to this ledger at all, and the
	// `keep` share was the figure from before the phase boundaries started handing
	// the text encoder's arenas back before the DiT ran. Measured on the reference
	// 6 GB card at 512x320x5, the plan spent 3.07 GB of a 5.01 GB budget (61 %) and
	// kept 7 of 50 blocks while the accountant's real peak was 3.85 GB - two more
	// blocks were affordable and were being paid for by nothing at all.
	//
	// PHI_DIT_OVERHEAD_MB still pins the resident term, for an experiment.
	const u64 kResChunk = 192ull << 20;   // Impl::res_arena's chunk size
	// ── what streaming one block actually costs ─────────────────────────────
	//
	// Not `im->layer_bytes`: the accountant books whole arena chunks, and a block's
	// tensors do not tile the chunk size, so the real cost is the sum of the chunks
	// the bump allocator needs - 522 MB for a 386 MB block on this card's 64 MB
	// weight-arena granularity (110 + 64 + 147 + 64 + 73.5 + 64, i.e. 35 % more than
	// the block weighs), which is the charge the `PHI_VRAM_TRACE` ledger shows.
	//
	// `arena_chunk_cost` simulates the allocator over the block's tensor sizes
	// instead of guessing, so the plan's own invariant (act + window + fixed <=
	// budget) is the one the enforcement will check.
	u64 wa_chunk = (g_ && g_->wa) ? g_->wa->chunk_bytes() : (64ull << 20);
	u64 in_flight = im->layer_bytes;
	if (!im->layer_tensors.empty()) {
		u64 cost = arena_chunk_cost(im->layer_tensors, wa_chunk);
		if (cost > in_flight) in_flight = cost;
	}
	// The refiner streams through the same arena, in the same step, and the two
	// halves of a step never overlap: the refiner runs in `embed_rows`, the block
	// loop after it. So the reservation is the larger of the two measured chunk
	// costs, not their sum - the plan used to reserve both (`loop_fixed` + the
	// streamed block), which is one arena more than the run ever holds.
	u64 refiner_cost = kRefinerArenaBytes;
	if (!im->refiner_tensors.empty()) {
		const u64 cost = arena_chunk_cost(im->refiner_tensors, wa_chunk);
		if (cost > refiner_cost) refiner_cost = cost;
	}
	const u64 reserve = std::max(refiner_cost, in_flight);

	// Everything the accountant already holds, less the weight streamer's own
	// chunks: those are the ones the refiner and the streamed block reuse, and
	// `reserve` has just paid for them. (At plan time the streamer still holds the
	// text encoder's last chunks in this process - they are handed back before the
	// refiner runs, and counting them *and* the refiner would reserve the same
	// arena twice.)
	u64 resident_now = vram_budget().local();
	if (g_ && g_->wa) {
		const u64 wa = g_->wa->capacity();
		resident_now = resident_now > wa ? resident_now - wa : 0;
	}
	// The caller's reference footprint is part of that measurement - `video_gen`
	// derives it from this same ledger - so it is a floor, not a second charge.
	if (extra_resident > resident_now) resident_now = extra_resident;
	// No device answered: fall back to the calibrated constant rather than 0.
	if (resident_now == 0) resident_now = kDefaultOverheadBytes;
	res_.overhead = resident_now;
	if (const char* e = getenv("PHI_DIT_OVERHEAD_MB")) {
		const long v = strtol(e, nullptr, 10);
		if (v >= 0 && v <= 8192) res_.overhead = (u64)v << 20;
	}
	// Everything resident besides the window and the DiT's own activations.
	//
	// ── the request's pixel weight ──────────────────────────────────────────
	//
	// The estimate below is linear in the row count `S`, which already carries the
	// resolution, the clip length and the reference rows - but the loop's real
	// per-step peak also holds buffers that scale with the request's *pixels*, not
	// its rows (the patch-projection staging rows, the un/patch gather rows, the
	// VAE-encode arenas still settling at plan time). Those are the terms that grow
	// with a long clip or a big reference image re-injected on every step, and the
	// reason a request the row estimate calls comfortable can cross the driver's
	// ceiling mid-sampling. They are charged here, once, as a function of the
	// request's total pixel budget: the generated frame area times `T_frames`, plus
	// every reference block's own area (`ref_pixels`, which `video_gen` derives from
	// the reference latents' geometry).
	//
	// ── the request's own pixel weight ───────────────────────────────────────
	//
	// It is charged as the *reference* share only. The generated frame's per-pixel
	// buffers are already inside `dit_activation_cost` (it simulates the arena's own
	// chunks over `forward()`'s allocations), so adding the whole figure back would
	// price the same arena twice - the failure recorded below. The references' pixels
	// are the part the activation term does *not* name: their latents' VAE encode and
	// the gather rows scale with the reference area, not with the packed row count,
	// so a big reference image or a long reference video is charged here, once, and
	// the window keeps fewer blocks.
	const u64 video_pixels =
	    (u64)std::max<i64>(lat_h, 0) * 16ull * (u64)std::max<i64>(lat_w, 0) * 16ull *
	    (u64)std::max<i64>(T_frames, 0);
	const u64 pixel_reserve = request_pixel_reserve_bytes(video_pixels, ref_pixels);
	res_.request_reserve = pixel_reserve;
	const u64 target_only_reserve = request_pixel_reserve_bytes(video_pixels, 0);
	const u64 ref_only_reserve =
	    pixel_reserve > target_only_reserve ? pixel_reserve - target_only_reserve : 0;
	const u64 fixed = res_.overhead + reserve + ref_only_reserve;
	res_.overhead = fixed;
	// ── the overhead the *loop* actually carries ────────────────────────────
	//
	// The refiner's share is charged exactly once, through `reserve`: it re-streams
	// on every step, so it really is live beside the window in the steady state.
	// Nothing is credited back for the arena `forward()` releases before the block
	// loop - the window is sized against the loop, and the loop keeps the refiner
	// (that is the whole point of the note above).
	const u64 loop_fixed = fixed;
	// ── the activation chunk ────────────────────────────────────────────────
	//
	// The fused projection scratch and the FFN working set scale with the *chunk*,
	// not with S, so the chunk is the one knob that trades dispatch count for
	// footprint. Start at the default and halve until the arena's real chunk cost
	// fits; the part that does not scale with the chunk is untouched by this, so
	// the search stops helping almost immediately and bails out at the floor.
	//
	// This test uses the *full* `fixed`, not `loop_fixed`: the activation arena is
	// allocated up front and is therefore fully live during the embedding phase too,
	// where the refiner's 770 MB is also resident. Under-counting it here allocates
	// its way past the card during the refiner, where the failure is the accountant's
	// refusal rather than a clean number.
	//
	// The refiner's scratch is priced over `max(text_tokens, chunk)` rows, the same
	// bound `forward()` allocates for it (`F.norm` is sized for the text span, and
	// the rest for the refiner's own row count, which is at most S).
	i64 chunk = kChunk;
	// The refiner's scratch is sized by the *conditioning* span (the text tokens plus
	// the packed reference rows), not by the video grid: `embed_rows` runs over the
	// conditioning rows, and `forward()` allocates its scratch as `L * ...` with
	// that L. Pricing it at `n_video + n_audio` would reserve ~3.5 GB of scratch the
	// run never asks for, which is the opposite failure to the one this model exists
	// to fix.
	const i64 phase_rows = std::max<i64>(text_tokens, 1);
	res_.act_dit = plan_activation(S, n_video, n_audio, video_t, lat_h, lat_w, res_.budget, fixed,
	                               &chunk, phase_rows, std::max<i64>(ref_rows, 0));
	res_.dit_chunk = chunk;

	// The largest sequence this budget would hold, from the same estimate. Every
	// term that scales with S is S * kActPerToken bytes, everything else is the
	// chunk scratch plus the fixed reserve, and the estimate carries a 20 %
	// margin - so the inverse is closed-form. An actionable number in the refusal
	// is worth more than the refusal.
	{
		const u64 per_token = (u64)5376 * 4 + 3ull * kInner * 2 + (u64)kInner * 4 +
		                      (u64)kRopePairs * 2 * 4 + 12 + 128 * 4;
		const u64 room = res_.budget > fixed ? res_.budget - fixed : 0;
		const u64 usable = room / 6 * 5;   // undo the estimate's 1/5 margin
		const u64 one = estimate_h3_dit_activation_bytes(1, 1, 0, 0, 0, 0, chunk);
		const u64 scaled = per_token + per_token / 5;   // the same 1/5 margin, per token
		const u64 fixed = one > scaled ? one - scaled : 0;
		res_.s_max = usable > fixed ? (i64)((usable - fixed) / per_token) : 0;
	}

	const u64 phase = res_.act_dit;
	// ── what the window is charged, not what its tensors weigh ─────────────────
	//
	// The plan books the window at `res_main_n * layer_bytes`, but the accountant
	// books whole 192 MB chunks (that is GpuArena's granularity, and the charge is
	// per chunk - see GpuArena::alloc). `window_charge` below rounds the count the
	// same way, so the plan's own invariant
	//
	//     act + window(chunks) + fixed <= budget
	//
	// is the one the enforcement will check. Measured, 5 frames at 256x128 on a
	// 5.01 GB budget: a plan that booked the window at its tensor bytes allowed 12
	// blocks (3.68 GB of tensors, 3.84 GB of chunks) and the run still died 147 MB
	// short of the ceiling - "VRAM exhausted: 147.00 MB for image.weights" - after
	// the text encoder had already run.
	//
	// `fixed` already carries `reserve` (the refiner / streamed-block maximum), so
	// the window's room is simply what is left of the budget after the activations,
	// the loop's own resident set and the request's pixel reserve - capped by what
	// the *device* will still hand out (`live_headroom()`: the ledger and the driver
	// are two walls, and the second one is the one a charge is refused on). The cap
	// does not bind in a normal run, where the two agree to within the ring; it binds
	// when they do not - another client on the card, a chunk this process released
	// in its accounting while the device kept the pages - and then the honest answer
	// is fewer resident blocks, not a refusal in the middle of the sampling loop.
	res_.available =
	    res_.budget > phase + loop_fixed ? res_.budget - phase - loop_fixed : 0;
	const u64 device_room = vram_budget().live_headroom();
	const bool device_capped = device_room < res_.available;
	if (device_capped) res_.available = device_room;
	// `res.block_charge` (below) is what the window is really charged - the sum of
	// the resident arena's chunks for one block - so the window count and the
	// charge are derived from the same number the accountant will apply.
	res_.res_main_n = im->layer_bytes ? (i64)(res_.available / im->layer_bytes) : 0;
	// What the arena really charges for one more block: the sum of the chunks its
	// bump allocator needs for this block's tensors, which is what the accountant
	// books and therefore what the tuner must grow by. Overwritten by the *measured*
	// growth as soon as a block is uploaded (the first upload happens in forward(),
	// so the first tune already uses the real number), because the simulation prices
	// an *empty* arena and a block appended to a populated one reuses its leftovers.
	res_.block_charge = im->layer_bytes;
	if (!im->layer_tensors.empty()) {
		const u64 cost = arena_chunk_cost(im->layer_tensors, kResChunk);
		if (cost > res_.block_charge) res_.block_charge = cost;
	}
	res_.stream_charge = reserve;
	while (res_.res_main_n > 0 &&
	       (u64)res_.res_main_n * res_.block_charge > res_.available)
		res_.res_main_n--;
	res_.res_main_n = std::clamp<i64>(res_.res_main_n, 0, cfg_.n_layers);
	// PHI_DIT_RES pins the count by hand and skips the sizing, exactly as it does
	// on the image chain: the instrument for "what does the runtime tuner do when
	// the plan under-fills the card". The window is still uploaded lazily, so the
	// pin is what the first forward takes.
	if (const char* e = getenv("PHI_DIT_RES")) {
		const int v = atoi(e);
		res_.res_main_n = std::clamp<i64>(v, 0, cfg_.n_layers);
	}
	res_.resident_bytes = (u64)res_.res_main_n * im->layer_bytes;
	res_.degraded = res_.available < im->layer_bytes;	// The VAE phase's own constraint: nothing is resident then, so its only claim
	// on the budget is its activation peak plus the fixed reserve.
	const bool vae_fits = res_.act_vae + loop_fixed <= res_.budget;
	res_.reason.clear();
	if (res_.degraded)
		res_.reason = "no room for a resident block beside the activation frame; every block "
		              "streams (the request's own pixels are the reason: the frame is linear "
		              "in them)";
	else if (!vae_fits)
		res_.reason = "the video VAE's decode peak does not fit the budget; it will stream-allocate";
	else if (device_capped)
		res_.reason = "resident window capped by what the device will still hand out";
	else if (res_.res_main_n < cfg_.n_layers)
		res_.reason = "resident window capped by the activation estimate";
	if (dit_res_debug())
		fprintf(stderr,
		        "[h3dit] resident window: %lld x %.1f MB = %.2f GB, + dit_act %.2f + resident "
		        "now %.2f + stream %.2f = %.2f <= budget %.2f GB (live %.2f/%.2f GB "
		        "at plan time)\n",
		        (long long)res_.res_main_n, (double)im->layer_bytes / 1048576.0,
		        (double)res_.resident_bytes / 1073741824.0, (double)res_.act_dit / 1073741824.0,
		        (double)(res_.overhead - reserve) / 1073741824.0,
		        (double)reserve / 1073741824.0,
		        (double)(res_.resident_bytes + loop_fixed) / 1073741824.0,
		        (double)res_.budget / 1073741824.0,
		        (double)vram_budget().live_usage() / 1073741824.0,
		        (double)vram_budget().os_budget() / 1073741824.0);

	// `im->planned` is part of the test, not just the shape: `release_weights()`
	// hands the window back at the end of a run and clears the flag, so the next
	// call - even one with the *same* geometry - has to arm `res_pending` again.
	// Without it the second `video_generate` of a session died in forward() with
	// "h3dit: plan_residency() must be called for this shape first", because the
	// shape still matched and the re-plan that sets the flag was skipped.
	const bool same = im->planned && im->planned_frames == T_frames &&
	                  im->planned_audio_t == audio_t && im->planned_lat_h == lat_h &&
	                  im->planned_lat_w == lat_w && im->planned_text == text_tokens &&
	                  im->planned_ref_rows == ref_rows && im->planned_ref_pixels == ref_pixels &&
	                  im->planned_extra == extra_resident;
	if (!same) {
		// Drop any window from the previous plan. The new one is *not* uploaded
		// here: it is deferred to the next forward(), which uploads it right after
		// the token refiner has released the weight arena (see `res_pending` and
		// loop_fixed above). Uploading it here would put the window and the refiner
		// on the card at once, which is what capped the window at 2 blocks.
		im->res.clear();
		im->res_arena.release_chunks();
		im->res_n = 0;
		// Armed even when the plan decided on *no* resident blocks: `res_pending` is
		// also what tells the runtime tuner that the first forward has not run yet,
		// and on that first step the DiT's activation frame - the arena sized by S,
		// which at 540P/243 frames is several gigabytes and is the single largest
		// consumer in the whole run - does not exist in the ledger yet. A tuner that
		// measured the process there would see an empty card, grow the window into
		// the room the frame is about to need, and the frame's own allocations would
		// then be refused mid-forward (measured: a 540P/243 plan that correctly
		// streamed every block still died with "64.00 MB for h3dit.acts would take
		// the process to 4.88 GB of the 4.85 GB ceiling"). From step 1 on the arena
		// is charged - `new_step()` rewinds the bump pointer and keeps the chunks -
		// so the band arithmetic is honest and the tuner is free to correct.
		im->res_pending = true;
		im->planned_frames = T_frames;
		im->planned_audio_t = audio_t;
		im->planned_lat_h = lat_h;
		im->planned_lat_w = lat_w;
		im->planned_text = text_tokens;
		im->planned_ref_rows = ref_rows;
		im->planned_ref_pixels = ref_pixels;
		im->planned_extra = extra_resident;
		im->planned_chunk = res_.dit_chunk;
		im->planned = true;
	}
	if (dit_res_debug())
		fprintf(stderr,
		        "[h3dit] plan_residency S=%lld frames=%lld: budget %.2f GB, dit_act %.2f GB, "
		        "vae_act %.2f GB, overhead %.2f GB (request pixels %.2f GB, reported only) -> "
		        "available %.2f GB, "
		        "block %.1f MB, resident %lld/%lld (%d%% of the budget)%s%s\n",
		        (long long)S, (long long)T_frames, (double)res_.budget / 1073741824.0,
		        (double)res_.act_dit / 1073741824.0, (double)res_.act_vae / 1073741824.0,
		        (double)res_.overhead / 1073741824.0, (double)pixel_reserve / 1073741824.0,
		        (double)res_.available / 1073741824.0,
		        (double)im->layer_bytes / 1048576.0, (long long)res_.res_main_n,
		        (long long)res_.n_layers,
		        res_.budget ? (int)(100ull * (res_.resident_bytes + res_.act_dit + res_.overhead) /
		                            res_.budget)
		                    : 0,
		        res_.reason.empty() ? "" : " — ", res_.reason.c_str());
}

// ── dynamic residency (core/runtime/vram_window.hpp) ────────────────
//
// See the header for why this exists. The one thing worth repeating here is the
// choice of numbers: none of the plan's estimates are reused. `other` is the
// ledger minus what the resident arena is *charged*, so the activation frame,
// the refiner's chunks and the caller's own footprint are all in it because they
// are really there; the live figure is the driver's usage floored by the ledger.
// The window is a disk-traffic decision and no arithmetic depends on it, so the
// corrections below swallow their own failures.

i64 AvDiT::tune_residency() {
	Impl* im = impl_of(this);
	if (!open_ || !im || !g_) return res_.res_main_n;
	// Before the first forward() the window the plan decided on has not been
	// uploaded yet (`res_pending`); there is nothing to measure or correct.
	if (im->res_pending) return im->res_n;
	const u64 limit = vram_budget().limit();
	if (limit == 0 || im->layer_bytes == 0) return im->res_n;

	const u64 window = im->res_arena.capacity();
	const u64 local = vram_budget().local();
	WindowLive l;
	l.n = im->res_n;
	l.n_total = cfg_.n_layers;
	l.other = local > window ? local - window : 0;
	l.block_bytes = im->layer_bytes;
	l.chunk_bytes = im->res_arena.chunk_bytes();
	l.limit = limit;
	l.used = std::max(vram_budget().live_usage(), local);

	WindowPolicy pol;
	// The headroom the *next* step needs beyond what the ledger already holds.
	//
	// Two sets of weights stream through the same arena, one after the other: the
	// token refiner's (~0.78 GB, at the start of every step) and the block loop's
	// (one block at a time, after it). `forward()` hands the arena back when the
	// refiner is done, so when this runs the ledger carries the *block loop's*
	// chunks and not the refiner's - and the refiner's load at the start of the next
	// step is what the window must not price out. The reservation is therefore the
	// refiner's charge *minus what the arena already holds*: the refiner's load
	// grows that chunk list rather than starting a new one, so only the difference
	// is uncovered.
	//
	// Reserving the whole refiner charge instead over-reserved by exactly the loop's
	// share and left the window below the band (~79 % measured); reserving a single
	// chunk under-reserved and the refiner's first load was refused, which shed the
	// window and re-grew it - an oscillation at ~20 s a step instead of steady
	// sampling.
	const u64 wa_held = (g_ && g_->wa) ? g_->wa->capacity() : 0;
	const u64 next_step_charge = res_.stream_charge > wa_held ? res_.stream_charge - wa_held : 0;
	pol.safety_bytes = std::max<i64>((i64)im->res_arena.chunk_bytes(), (i64)next_step_charge);
	pol.block_charge = res_.block_charge;
	if (dit_res_debug())
		fprintf(stderr,
		        "[h3dit] tune in: n=%lld used=%.2f other=%.2f window=%.2f wa=%.2f "
		        "safety=%.2f per=%.2f limit=%.2f\n",
		        (long long)l.n, (double)l.used / 1073741824.0, (double)l.other / 1073741824.0,
		        (double)window / 1073741824.0, (double)wa_held / 1073741824.0,
		        (double)pol.safety_bytes / 1073741824.0, (double)pol.block_charge / 1048576.0,
		        (double)limit / 1073741824.0);
	i64 delta = 0;
	const WindowMove mv = tune_window(l, pol, &delta);
	if (mv == WindowMove::Grow) {
		const i64 before = im->res_n;
		const i64 target = std::min<i64>(cfg_.n_layers, before + delta);
		if (im->res.size() < (size_t)cfg_.n_layers) im->res.resize((size_t)cfg_.n_layers);
		g_->ctx->begin();
		for (i64 i = before; i < target; i++) {
			const u64 before_cap = im->res_arena.capacity();
			try {
				load_block(im->st, *g_, im->res_arena, i, im->res[(size_t)i], im->lora);
				im->res_n = i + 1;
			} catch (const std::exception& e) {
				fprintf(stderr, "[h3dit] tune: window growth stopped at %lld/%lld (%s)\n",
				        (long long)i, (long long)target, e.what());
				break;
			}
			// One more block's real marginal charge, measured - see the same
			// measurement in forward()'s deferred upload.
			const u64 grew = im->res_arena.capacity() - before_cap;
			if (grew > 0) res_.block_charge = grew;
		}
		g_->ctx->submit();
		g_->ring->rewind();
	} else if (mv == WindowMove::Shrink) {
		const i64 before = im->res_n;
		const i64 want = std::max<i64>(0, before - (-delta));
		if (want < before) {
			// A bump arena cannot give back one block: the only way to free resident
			// chunks is to drop the window and re-take it smaller.
			im->res.clear();
			im->res_arena.release_chunks();
			im->res_n = 0;
			if (want > 0) {
				im->res.resize((size_t)want);
				g_->ctx->begin();
				for (i64 i = 0; i < want; i++) {
					try {
						load_block(im->st, *g_, im->res_arena, i, im->res[(size_t)i], im->lora);
						im->res_n = i + 1;
					} catch (const std::exception&) {
						break;
					}
				}
				g_->ctx->submit();
				g_->ring->rewind();
				im->res.resize((size_t)im->res_n);
			}
		}
	}

	if (mv != WindowMove::Hold) {
		res_.res_main_n = im->res_n;
		res_.resident_bytes = (u64)im->res_n * im->layer_bytes;
		res_.degraded = im->res_n < cfg_.n_layers;
		res_.reason = im->res_n == 0
		                  ? "live tune: the window was shed; every block streams"
		                  : "live tune: the window tracks the 90-100% sampling band";
		if (dit_res_debug())
			fprintf(stderr, "[h3dit] tune: window %lld/%lld, %d%% of the %s limit\n",
			        (long long)im->res_n, (long long)cfg_.n_layers,
			        (int)(100ull * (im->res_arena.capacity() + l.other) / limit),
			        format_bytes(limit).c_str());
	}
	return im->res_n;
}

// ── forward ───────────────────────────────────────────────────────────────

void h3_torch_randn_fill(float* dst, i64 n, u64 seed) {
	TorchRandn r(seed);   // the struct is in the anonymous namespace above
	r.fill(dst, n);
}

// The packed row count the reference blocks contribute (see the header).
i64 h3_ref_row_count(const H3Ref* refs, i64 n_refs) {
	i64 n = 0;
	for (i64 k = 0; k < n_refs; k++) {
		const H3Ref& r = refs[k];
		if (r.kind != H3Ref::Audio && r.latent && r.latent_t > 0)
			n += r.latent_t * (r.latent_h / 2) * (r.latent_w / 2);
		if (r.kind != H3Ref::Image && r.audio_latent && r.ref_audio_t > 0)
			n += 2 * r.ref_audio_t;
	}
	return n;
}

// The references' own *pixel* area, for `plan_residency`'s per-pixel charge: each
// reference video/image contributes its latent grid times the video VAE's 16x
// spatial scale, over its frame count. Audio-only blocks have no spatial extent and
// contribute nothing here (their cost is already in `h3_ref_row_count`).
u64 h3_ref_pixel_area(const H3Ref* refs, i64 n_refs) {
	u64 px = 0;
	for (i64 k = 0; k < n_refs; k++) {
		const H3Ref& r = refs[k];
		if (r.kind == H3Ref::Audio) continue;
		if (!r.latent || r.latent_t <= 0 || r.latent_h <= 0 || r.latent_w <= 0) continue;
		px += (u64)r.latent_t * (u64)(r.latent_h * 16) * (u64)(r.latent_w * 16);
	}
	return px;
}

// The legacy geometry-free entry point: no reference blocks at all. The old
// `ref_v_lat` / `ref_T` / `ref_a_lat` arguments are still accepted; they can only
// describe a *single* block that shares the target's spatial grid and audio
// length, which is what a caller that has no other geometry to hand could ever
// have meant by them, so that is what they are turned into. Anything richer goes
// through forward_refs().
void AvDiT::forward(const float* x_v, const float* x_a, i64 T, i64 H, i64 W,
                           const float* embed, const i32* tags, i64 L, float t,
                           const float* ref_v_lat, i64 ref_T, const float* ref_a_lat, float* out_v,
                           float* out_a) {
	Impl* im = impl_of(this);
	if (!open_ || !im) throw MediaError("h3dit: forward() before open()");
	H3Ref one;
	i64 n_refs = 0;
	if (ref_v_lat && ref_T > 0) {
		one.kind = H3Ref::Video;
		one.latent = ref_v_lat;
		one.latent_t = ref_T;
		one.latent_h = H;   // the only geometry these arguments can express
		one.latent_w = W;
		one.audio_latent = ref_a_lat;
		one.ref_audio_t = ref_a_lat ? im->planned_audio_t : 0;
		n_refs = 1;
	}
	forward_refs(x_v, x_a, T, H, W, embed, tags, L, t, n_refs ? &one : nullptr, n_refs, 1.0f, 0,
	             out_v, out_a);
}

void AvDiT::forward_refs(const float* x_v, const float* x_a, i64 T, i64 H, i64 W,
                                const float* embed, const i32* tags, i64 L, float t,
                                const H3Ref* refs, i64 n_refs, float cond_noise_aug,
                                u64 noise_seed, float* out_v, float* out_a) {
	Impl* im = impl_of(this);
	if (!open_ || !im) throw MediaError("h3dit: forward_refs() before open()");
	if (L <= 0) throw MediaError("h3dit: the text conditioning is empty");
	const i64 vt = T, lat_h = H, lat_w = W;
	const i64 ref_rows = h3_ref_row_count(refs, n_refs);
	if (!im->planned || im->planned_lat_h != lat_h || im->planned_lat_w != lat_w ||
	    im->planned_text != L || im->planned_ref_rows != ref_rows)
		throw MediaError("h3dit: plan_residency() must be called for this shape first");
	const i64 audio_t = im->planned_audio_t;
	const i64 hidden = cfg_.hidden;
	const i64 S = L + 2 * audio_t + vt * (lat_h / cfg_.patch_h) * (lat_w / cfg_.patch_w) + ref_rows;

	Fwd F;
	F.im = im;
	F.g = g_;
	F.cfg = &cfg_;
	// Chunk height comes from the residency plan, which lowers it when the
	// chunk-proportional scratch is what stands between this shape and the
	// budget (see plan_residency). It only ever bounds *scratch*: the packed
	// q/k/v and the attention output stay sequence-proportional.
	F.chunk = std::min<i64>(im->planned_chunk > 0 ? im->planned_chunk : kChunk, S);
	Layout& ly = F.ly;

	// 0. the audio carry (comfy MiniMaxH3Model.forward steps 1-2)
	//
	// The sampler walks the audio latent *carried* onto the video schedule: its
	// state is the stream's own latent z scaled by sigma_v/sigma_a (= 1 at
	// sigma_v = 1, the ratio video_gen's own /audio_scale undoes at the end), and
	// what the DiT sees is z itself (the stream's own latent at its own sigma):
	//
	//     sigma_v = clamp(t, 1e-6)
	//     sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
	//     carry   = sigma_a / sigma_v
	//     x_net_a = x_a_sampler * carry
	//
	// This has to be computed *before* the packed layout is built, because the
	// layout's row sources are pointers: the target audio rows must point at the
	// carried values, and this module has to hand the layout the very tensor the
	// embedding step will read. It used to be computed after the layout and passed
	// to `embed_rows` separately, which the audio rows ignored (they read the
	// layout's pointer), so the model saw the un-carried latent - a 20 % error on
	// every audio row of the packed sequence, at every step.
	const float audio_scale =
	    cfg_.sigma_shift_audio != 0.0f ? cfg_.sigma_shift_video / cfg_.sigma_shift_audio : 1.0f;
	const float sigma_v = std::max(t, 1e-6f);
	const float sigma_a =
	    h3_time_shift_sigma(sigma_v, cfg_.sigma_shift_video, cfg_.sigma_shift_audio);
	const float carry = sigma_a / sigma_v;
	const size_t a_elems = (size_t)32 * 2 * audio_t;
	std::vector<float> xa_net(a_elems);
	for (size_t i = 0; i < a_elems; i++) xa_net[i] = x_a[i] * carry;
	if (const char* dp = dit_dump_prefix()) {
		fprintf(stderr, "[h3dit] carry %.6f sigma_v %.6f sigma_a %.6f audio_scale %.3f\n",
		        carry, sigma_v, sigma_a, audio_scale);
		if (FILE* af = fopen((std::string(dp) + "xa_net.f32").c_str(), "wb")) {
			fwrite(xa_net.data(), 4, xa_net.size(), af);
			fclose(af);
		}
	}

	// 1. packed positions, rope angles, timesteps.
	//
	// The packed sequence is [text | reference blocks | audio | video]
	// (comfy/ldm/minimax/model.py::PackedLayout). On the t2va path there are no
	// reference blocks, and every number below is the one the pre-ref2va code
	// produced: one text span, the target audio span, the target video span, two
	// timesteps, and the same three projection calls in the same order.
	{
		ly.text = L;
		ly.audio_t = audio_t;
		ly.video_t = vt;
		ly.tokens_h = lat_h / cfg_.patch_h;
		ly.tokens_w = lat_w / cfg_.patch_w;
		ly.n_audio = 2 * audio_t;
		ly.n_video = vt * ly.tokens_h * ly.tokens_w;
		ly.s = S;

		std::vector<double> target_w_axis;
		const std::vector<double> target_frame = frame_grid(lat_h, lat_w, &target_w_axis);

		// --- the reference blocks: row counts and time spans ---------------
		//
		// `_ref_t_span` decides how far the target timeline is pushed out; the
		// reference rows themselves pack from cursor = text_len and each block
		// advances it by its own span, so the target timeline starts *after*
		// every reference span. A video block's audio rows and video rows share
		// that origin (and the audio rows come first).
		struct RefPack {
			const H3Ref* src = nullptr;
			i64 n_video = 0, n_audio = 0;
			double span = 0.0;
		};
		std::vector<RefPack> rp((size_t)std::max<i64>(n_refs, 0));
		double target_cursor = (double)L;
		for (i64 k = 0; k < n_refs; k++) {
			const H3Ref& r = refs[k];
			RefPack& q = rp[(size_t)k];
			q.src = &r;
			if (r.kind != H3Ref::Audio && r.latent && r.latent_t > 0)
				q.n_video = r.latent_t * (r.latent_h / 2) * (r.latent_w / 2);
			if (r.kind != H3Ref::Image && r.audio_latent && r.ref_audio_t > 0)
				q.n_audio = 2 * r.ref_audio_t;
			if (r.kind == H3Ref::Image) q.span = 1.0;
			else if (r.kind == H3Ref::Audio) q.span = (double)r.ref_audio_t;
			else q.span = std::max((double)r.ref_audio_t, video_t_spans_sum(r.latent_t));
			target_cursor += q.span;
		}

		// --- timesteps ----------------------------------------------------
		//
		// The reference's `seg_t` / `seg_tag`: reference rows are never denoised,
		// they run at the conditioning timestep (max(t, VISUAL_COND_TIMESTEP) for
		// video rows, max(t, AUDIO_COND_TIMESTEP) for audio ones).
		const float sigma_v0 = std::max(t, 1e-6f);
		const float sigma_a0 =
		    h3_time_shift_sigma(sigma_v0, cfg_.sigma_shift_video, cfg_.sigma_shift_audio);
		const float t_v = 1.0f - sigma_v0, t_a = 1.0f - sigma_a0;
		const float vis_aug = cond_noise_aug;   // VISUAL_COND_TIMESTEP (0.999 released)
		const float aud_aug = 1.0f;             // AUDIO_COND_TIMESTEP
		auto seg_t_of = [&](i32 kind) -> float {
			switch (kind) {
				case KText: case KVideo: return t_v;
				case KAudio: return t_a;
				case KCond: case KRefImg: return std::max(t_v, vis_aug);
				default: return std::max(t_a, aud_aug);   // KRefAudio
			}
		};
		auto seg_tag_of = [](i32 kind) -> i64 {
			switch (kind) {
				case KText: return 1;
				case KAudio: case KRefAudio: return 2;
				default: return 0;   // video / cond / ref_img
			}
		};

		// Text rows carry their token index on the *time* axis only; the h/w axes
		// stay 0. `PackedLayout.__init__` in comfy/ldm/minimax/model.py builds the
		// text span as `g = zeros(text_len, 3); g[:, 0] = arange(text_len)`, and
		// the per-axis rope reads the three axes separately (`rope_freqs` cats
		// t_f/h_f/w_f into the 48 pair angles), so putting the index on all three
		// axes rotates each text key by two extra angles per frequency - i.e. the
		// dialog conditioning the video attends to is scrambled. Measured against
		// the reference on the same dumped step: with the
		// index on all three axes the text rows' q/k miss the reference by ~70%
		// L2, with it on t alone they agree to fp16 rounding.
		ly.pos.assign((size_t)S * 3, 0.0f);
		for (i64 i = 0; i < L; i++) ly.pos[(size_t)(i * 3 + 0)] = (float)i;
		// channel-major stereo audio rows: t advances per latent frame, w is
		// pinned to the two grid extremes, h stays 0 (`_audio_grid`).
		auto put_audio = [&](i64 row0, i64 rt, double origin, double w_lo, double w_hi) {
			for (i64 ch = 0; ch < 2; ch++)
				for (i64 tt = 0; tt < rt; tt++) {
					const i64 rw = row0 + ch * rt + tt;
					ly.pos[(size_t)(rw * 3 + 0)] = (float)(origin + (double)tt);
					ly.pos[(size_t)(rw * 3 + 1)] = 0.0f;
					ly.pos[(size_t)(rw * 3 + 2)] = (float)(ch ? w_hi : w_lo);
				}
		};
		// `_video_grid`: the t axis walks FRAME_RESCALE * FRAME_PER_TOKEN; an image
		// reference instead pins every row at its cursor (`g[:, 0] = cursor`).
		auto put_video = [&](i64 row0, i64 nvt, i64 rh, i64 rw, const std::vector<double>& frame,
		                     double origin, bool const_t) {
			const i64 fh = rh / 2, fw = rw / 2;
			const std::vector<double> tg = video_t_grid(nvt, origin);
			for (i64 tt = 0; tt < nvt; tt++)
				for (i64 hh = 0; hh < fh; hh++)
					for (i64 ww = 0; ww < fw; ww++) {
						const i64 rw_ = row0 + (tt * fh + hh) * fw + ww;
						ly.pos[(size_t)(rw_ * 3 + 0)] =
						    (float)(const_t ? origin : tg[(size_t)tt]);
						ly.pos[(size_t)(rw_ * 3 + 1)] =
						    (float)frame[(size_t)((hh * fw + ww) * 2 + 0)];
						ly.pos[(size_t)(rw_ * 3 + 2)] =
						    (float)frame[(size_t)((hh * fw + ww) * 2 + 1)];
					}
		};

		// --- the segment table, in packed order ---------------------------
		struct Pending {
			i64 a = 0, b = 0;
			i32 kind = KText;
			i64 tag = 1;
		};
		std::vector<Pending> pend;
		i64 row = L;
		double cursor = (double)L;

		// text span: one segment per modality run (the presentation tags decide)
		if (tags) {
			i64 run = 0;
			for (i64 i = 1; i <= L; i++) {
				if (i == L || tags[i] != tags[run]) {
					Pending pd;
					pd.a = run;
					pd.b = i;
					pd.kind = KText;
					pd.tag = seg_tag_of(tags[run] == 0 ? KCond : KText);
					pend.push_back(pd);
					run = i;
				}
			}
		} else {
			Pending pd;
			pd.a = 0;
			pd.b = L;
			pd.kind = KText;
			pd.tag = seg_tag_of(KText);
			pend.push_back(pd);
		}

		// reference blocks, in packing order
		for (i64 k = 0; k < n_refs; k++) {
			const H3Ref& r = refs[k];
			const RefPack& q = rp[(size_t)k];
			if (r.kind == H3Ref::Video && q.n_audio > 0) {
				std::vector<double> rw_axis;
				frame_grid(r.latent_h, r.latent_w, &rw_axis);
				put_audio(row, r.ref_audio_t, cursor, rw_axis.front(), rw_axis.back());
				Pending pd;
				pd.a = row;
				pd.b = row + q.n_audio;
				pd.kind = KRefAudio;
				pd.tag = seg_tag_of(KRefAudio);
				pend.push_back(pd);
				ly.arows.push_back({KRefAudio, r.audio_latent, r.ref_audio_t, row, q.n_audio});
				row += q.n_audio;
			}
			if (r.kind != H3Ref::Audio && q.n_video > 0) {
				const std::vector<double> fr = frame_grid(r.latent_h, r.latent_w, nullptr);
				put_video(row, r.latent_t, r.latent_h, r.latent_w, fr, cursor,
				          r.kind == H3Ref::Image);
				Pending pd;
				pd.a = row;
				pd.b = row + q.n_video;
				pd.kind = KRefImg;
				pd.tag = seg_tag_of(KRefImg);
				pend.push_back(pd);
				ly.vrows.push_back(
				    {KRefImg, r.latent, r.latent_t, r.latent_h, r.latent_w, row, q.n_video});
				row += q.n_video;
			}
			if (r.kind == H3Ref::Audio && q.n_audio > 0) {
				put_audio(row, r.ref_audio_t, cursor, target_w_axis.front(),
				          target_w_axis.back());
				Pending pd;
				pd.a = row;
				pd.b = row + q.n_audio;
				pd.kind = KRefAudio;
				pd.tag = seg_tag_of(KRefAudio);
				pend.push_back(pd);
				ly.arows.push_back({KRefAudio, r.audio_latent, r.ref_audio_t, row, q.n_audio});
				row += q.n_audio;
			}
			if (r.kind == H3Ref::Image) cursor += 1.0;
			else if (r.kind == H3Ref::Audio) cursor += (double)r.ref_audio_t;
			else cursor += std::max((double)r.ref_audio_t, video_t_spans_sum(r.latent_t));
		}
		ly.n_ref_video = 0;
		ly.n_ref_audio = 0;
		for (const RefPack& q : rp) {
			ly.n_ref_video += q.n_video;
			ly.n_ref_audio += q.n_audio;
		}

		// target audio then target video, always the last two segments
		put_audio(row, audio_t, target_cursor, target_w_axis.front(), target_w_axis.back());
		{
			Pending pd;
			pd.a = row;
			pd.b = row + ly.n_audio;
			pd.kind = KAudio;
			pd.tag = seg_tag_of(KAudio);
			pend.push_back(pd);
			ly.arows.push_back({KAudio, xa_net.data(), audio_t, row, ly.n_audio});
			row += ly.n_audio;
		}
		put_video(row, vt, lat_h, lat_w, target_frame, target_cursor, false);
		{
			Pending pd;
			pd.a = row;
			pd.b = row + ly.n_video;
			pd.kind = KVideo;
			pd.tag = seg_tag_of(KVideo);
			pend.push_back(pd);
			ly.vrows.push_back({KVideo, x_v, vt, lat_h, lat_w, row, ly.n_video});
			row += ly.n_video;
		}
		if (row != S)
			throw MediaError("h3dit: packed layout is " + std::to_string(row) +
			                 " rows but S is " + std::to_string(S));

		// the distinct timesteps, in the reference's order (a sorted set)
		{
			std::vector<float> ut = {t_v, t_a};
			for (const Pending& pd : pend) ut.push_back(seg_t_of(pd.kind));
			std::sort(ut.begin(), ut.end());
			ut.erase(std::unique(ut.begin(), ut.end()), ut.end());
			ly.t_vals = ut;
			ly.m_count = (i64)ut.size();
			auto row_of = [&](float v) {
				for (size_t z = 0; z < ut.size(); z++)
					if (ut[z] == v) return (i64)z;
				throw MediaError("h3dit: timestep not in the unique set");
			};
			for (const Pending& pd : pend) {
				Seg s2;
				s2.a = pd.a;
				s2.b = pd.b;
				s2.kind = pd.kind;
				s2.chunk0 = (row_of(seg_t_of(pd.kind)) * 3 + pd.tag) * 6;
				ly.segs.push_back(s2);
			}
			ly.audio_seg = (i64)ly.segs.size() - 2;
			ly.video_seg = (i64)ly.segs.size() - 1;
			ly.video_m = row_of(t_v);
			ly.audio_m = row_of(t_a);
		}

		const std::vector<float> inv = tensor_to_f32(im->st, im->st.require("rope.inv_freq"));
		if ((i64)inv.size() != kRopeFreqs) throw MediaError("h3dit: rope.inv_freq is not 16 wide");
		ly.rope.assign((size_t)S * kRopePairs * 2, 0.0f);
		for (i64 i = 0; i < S; i++)
			for (i64 k = 0; k < kRopePairs; k++) {
				const i64 axis = k / kRopeFreqs;
				const double a = (double)ly.pos[(size_t)(i * 3 + axis)] *
				                 (double)inv[(size_t)(k % kRopeFreqs)];
				ly.rope[(size_t)((i * kRopePairs + k) * 2 + 0)] = (float)std::cos(a);
				ly.rope[(size_t)((i * kRopePairs + k) * 2 + 1)] = (float)std::sin(a);
			}

		// the staging buffer one patch-projection group needs
		ly.stage_floats = (i64)L * cfg_.text_dim;
		for (const VRowSrc& vs : ly.vrows)
			ly.stage_floats = std::max(ly.stage_floats, vs.n * kVidPatch);
		for (const ARowSrc& as : ly.arows)
			ly.stage_floats = std::max(ly.stage_floats, as.n * kAudPatch);

		const std::vector<float> table = tensor_to_f32(im->st, im->st.require("adaln_t_table"));
		ly.t_emb.assign((size_t)ly.m_count * 8, 0.0f);
		for (i64 m = 0; m < ly.m_count; m++)
			adaln_table_row(table.data(), ly.t_vals[(size_t)m], &ly.t_emb[(size_t)m * 8]);
	}

	// ── 3. allocations (fixed order: identical addresses every step) ───────
	//
	// The residency shed hook first: the embedding step below already streams the
	// refiner's 770 MB, and it is the one streaming step that runs with the window
	// up, so it must be able to give the window back the same way the block loop
	// does (see embed_rows and the loop's own recovery).
	F.shed_resident = [&]() -> bool {
		if (im->res_n == 0 && !im->res_pending) return false;
		g_->ctx->submit_if_recording();
		im->res.clear();
		im->res_arena.release_chunks();
		im->res_n = 0;
		im->res_pending = false;
		res_.res_main_n = 0;
		res_.resident_bytes = 0;
		res_.degraded = true;
		res_.reason = "resident window dropped mid-run (the accountant refused a weight "
		              "chunk); every block streams";
		return true;
	};
	// The order is fixed so every step lands on the same addresses, and it is now
	// *phase* order as well: the bump allocator's high-water mark is the sum of
	// everything live at once, and the refiner's scratch and the block loop's
	// chunk buffers are never live together. They used to be allocated side by
	// side for the whole step - 630 MB of refiner scratch (F.t* are all L rows
	// tall) plus 680 MB of loop buffers, 1.48 GB at a 2051-row presentation - which
	// is what pushed a 540P ref2va run past the plan's own activation estimate
	// (1.17 GB). The accountant then refused the last chunk of the resident
	// window, and a window the plan had sized for 4 blocks was shed and streamed
	// block by block: the plan was not wrong about the budget, the *arena* was
	// holding scratch nobody was using.
	auto& aa = *g_->aa;
	g_->new_step();
	aa.set_tag("h3dit.acts");
	const u64 chunk = (u64)F.chunk;
	F.hidden = aa.alloc((u64)S * hidden * 4);
	F.rope = aa.alloc((u64)S * kRopePairs * 2 * 4);
	F.pos = aa.alloc((u64)S * 3 * 4);
	// The refiner's own RMSNorm target is `F.norm` (`refiner_block` asks for
	// L * hidden floats), so it is sized for the text span as well as for one
	// chunk: L > chunk is the normal case for a reference presentation, and the
	// loop below never asks for more than `chunk` rows.
	F.norm = aa.alloc((u64)std::max<i64>(chunk, L) * hidden * 4);
	F.prow = aa.alloc((u64)ly.stage_floats * 4);
	F.vrows = aa.alloc((u64)ly.n_video * kVidPatch * 4);
	F.arows = aa.alloc((u64)ly.n_audio * kAudPatch * 4);
	// The LoRA correction's hidden form: as many rows as the largest correction
	// can have (the refiner runs the whole text span at once, the blocks run one
	// chunk - both bounded by `max(chunk, L)`, as `F.norm` is) and
	// `kLoraRankBound` wide, the bound `dit_activation_cost` prices it at. It is
	// claimed *outside* `embed_mark` below because the block loop needs it too,
	// and a chain with no LoRA still gets it (a few hundred KB): sizing it to
	// zero would turn a later `apply_lora_tail` into an error instead of a no-op.
	F.lora_h = aa.alloc((u64)std::max<i64>(chunk, L) * (u64)kLoraRankBound * 4);

	// The refiner's scratch: live inside `embed_rows` and dead the moment it
	// returns (the block loop reads none of it), so it is claimed under a mark
	// that is rewound before the loop's own buffers are allocated.
	const GpuArena::Mark embed_mark = aa.mark();
	const u64 tl = (u64)L;
	F.tq = aa.alloc(tl * kInner * 4);
	F.tk = aa.alloc(tl * kInner * 4);
	F.tv = aa.alloc(tl * kInner * 4);
	F.tqkv = aa.alloc(tl * kQkvOut * 4);
	F.tattn = aa.alloc(tl * kInner * 4);
	F.tproj = aa.alloc(tl * hidden * 4);
	F.tffn = aa.alloc(tl * 2 * kFfn * 4);
	F.tsw = aa.alloc(tl * kFfn * 4);

	g_->begin();
	g_->upload_into(F.rope, ly.rope.data(), (u64)ly.rope.size() * 4);
	g_->upload_into(F.pos, ly.pos.data(), (u64)ly.pos.size() * 4);

	// ── 4. embeddings ──────────────────────────────────────────────────────
	const double t_fwd0 = now_ms();
	{
		ProfScope ps(g_, prof_sink(&H3Prof::embed));
		embed_rows(F, x_v, xa_net.data(), embed, L, cfg_.text_dim, cond_noise_aug,
		           noise_seed);
	}
	if (const char* dp = dit_dump_prefix()) {
		const std::vector<float> hh = g_->download_f32(F.hidden, (u64)S * hidden);
		if (FILE* hf = fopen((std::string(dp) + "embed.f32").c_str(), "wb")) {
			fwrite(hh.data(), 4, hh.size(), hf);
			fclose(hf);
		}
		double mn = 1e30, mx = -1e30;
		for (float x : hh) { mn = std::min(mn, (double)x); mx = std::max(mx, (double)x); }
		fprintf(stderr, "[h3dit] dump embed [%lld x %lld] [%.4f, %.4f]\n",
		        (long long)S, (long long)hidden, mn, mx);
	}
	// The refiner weights (1.5 GB of bf16) are dead now; the streaming blocks
	// reuse the same arena anyway, but the resident path would otherwise keep
	// them for the whole sampling run.
	g_->wa->release_chunks();

	// Its *activation* scratch is dead too, so the activation arena goes back to
	// the mark taken before it (see the allocation order above). This has to
	// happen before the window is uploaded below: the refiner's scratch and the
	// block loop's buffers are never live at the same time, and the region they
	// share is 630 MB of the resident window's budget at a 2051-row
	// presentation.
	aa.rewind_to(embed_mark);

	// The resident window is uploaded here - after the refiner has let go of the
	// weight arena, not at plan time - so the two are never on the card at once.
	// `plan_residency` sized the window against the loop's own overhead for exactly
	// this reason (see loop_fixed there). Called once: the first forward() checks the
	// pending flag, every later step sees `res_n` already set.
	//
	// A window the accountant refuses is *not* fatal: the same blocks can be streamed
	// one at a time through the weight arena, which is what the loop falls back to
	// anyway, so the run gets slower and produces exactly the same video. Residency
	// is a speed decision, and the engine must never trade a completed run for it.
	if (im->res_pending) {
		// Blocks are loaded one at a time and the first refusal stops the window
		// rather than throwing it away. Residency is a speed decision and the loop
		// streams whatever is not resident (same arithmetic, same pixels, just more
		// disk traffic), so the honest answer to "the accountant said no at block k"
		// is "then k is the window", not "then there is no window": shedding the
		// whole arena over one chunk is what turned an optimistic *estimate* into a
		// run with zero resident blocks.
		//
		// The blocks that did land stay loaded: the arena is a bump allocator, the
		// loop only ever reads `res[0..res_n)`, and the chunks the refused block left
		// behind are the same chunks a resident block would have occupied anyway.
		i64 loaded = 0;
		if (res_.res_main_n > 0) {
			im->res.resize((size_t)res_.res_main_n);
			for (i64 i = 0; i < res_.res_main_n; i++) {
				// The window's own marginal charge, measured block by block: the arena
				// books whole chunks, so what one more block really costs is the growth
				// the upload causes, not the block's tensor bytes (and not the chunks a
				// simulation of the *empty* arena would need either - the leftovers of
				// the previous block's chunks are reused). The runtime tuner sizes its
				// corrections with this number, so a plan that keeps five blocks and a
				// tuner trying to add a sixth agree on what the sixth costs.
				const u64 before_cap = im->res_arena.capacity();
				try {
					load_block(im->st, *g_, im->res_arena, i, im->res[(size_t)i], im->lora);
					loaded = i + 1;
				} catch (const std::exception& e) {
					if (dit_res_debug() || i == 0)
						fprintf(stderr,
						        "[h3dit] resident window stopped at %lld/%lld blocks (%s); the "
						        "rest stream\n",
						        (long long)i, (long long)res_.res_main_n, e.what());
					break;
				}
				const u64 grew = im->res_arena.capacity() - before_cap;
				if (grew > 0) res_.block_charge = grew;
			}
		}
		im->res_n = loaded;
		res_.res_main_n = loaded;
		res_.resident_bytes = (u64)loaded * im->layer_bytes;
		if (loaded == 0)
			res_.reason =
			    "resident window refused by the accountant; every block streams";
		else if (loaded < cfg_.n_layers)
			res_.degraded = true;
		im->res_pending = false;
	}

	// ── the block loop's own buffers ────────────────────────────────────────
	//
	// Claimed here, after the refiner's scratch has been handed back, so the two
	// sets share one region instead of adding up (see the allocation order in
	// §3). One chunk of the fused qkv projection, not S of them; and a separate
	// attention-output buffer (which is also the out_proj input).
	F.qkv_out = aa.alloc(chunk * kQkvOut * 4);
	// Attention output: ONE CHUNK, not S. Attention is per-query-row independent
	// and its only consumer (this block's out_proj) already ran a chunk at a
	// time, so an S-tall buffer here was pure waste - 1.08 GB at 540P/10s, which
	// is a sixth of the whole activation budget. `run_attn_mlp` dispatches
	// attention for one chunk and consumes it before moving on.
	F.attnout = aa.alloc(chunk * kInner * 4);
	// q/k/v are packed fp16 (two elements per word): the tiled attention kernel
	// reads them that way, and at 540P/243 frames the fp32 form alone would be
	// 3.2 GB of the 4.6 GB budget.
	F.q = aa.alloc((u64)S * kInner * 2);
	F.k = aa.alloc((u64)S * kInner * 2);
	F.v = aa.alloc((u64)S * kInner * 2);
	F.q8 = aa.alloc(chunk * kFfn);
	F.s8 = aa.alloc(chunk * 4);
	F.ffn = aa.alloc(chunk * 2 * kFfn * 4);
	F.sw = aa.alloc(chunk * kFfn * 4);
	F.down = aa.alloc(chunk * hidden * 4);
	F.proj = aa.alloc(chunk * hidden * 4);
	F.mod = aa.alloc((u64)ly.m_count * 3 * 6 * hidden * 4);

	// ── 5. the 50 blocks ───────────────────────────────────────────────────
	H3Prof& PR = h3_prof();
	if (PR.on) {
		PR.resident = im->res_n;
		PR.blocks = cfg_.n_layers;
	}
	// ── the host-side weight cache ─────────────────────────────────────────
	//
	// Filled once per weight set, here, because this is the first point where the
	// residency decision is final (`res_n` is what the window really took) and the
	// refiner has already let go of its arena.
	//
	// Why it exists: 41 of the 50 blocks are streamed off the checkpoint on *every*
	// step - 15.1 GB per step, plus 1.5 GB for the refiner - and this box's NVMe
	// delivers ~1.2 GB/s for that pattern no matter how it is driven. Measured
	// on this box: 1.13 GB/s for a whole-file sequential read of the 19.5 GB
	// checkpoint, 1.25 GB/s for the engine's own 12-scattered-reads-per-block
	// pattern, 1.00-1.18 GB/s when a block's tensors are merged into fewer, larger
	// reads, 1.02-1.24 GB/s with 2/3/4/6/8 concurrent reader threads, and a
	// perfectly additive 99.8 s when the reads are interleaved with compute versus
	// 85.3 s when a worker thread overlaps them. Read granularity, queue depth,
	// thread count and overlap are all at the device's limit already; the disk time
	// is 14.8 s of a 18.9 s step and nothing in the read path can shorten it.
	//
	// What can shorten it is not reading those bytes again. System memory moves them
	// ~10x faster than the disk, so the streamed blocks that fit are pinned in RAM
	// once and served out of it for the rest of the session.
	//
	// The refiner goes first: 1.5 GB re-read on every step is the best
	// bytes-saved-per-byte-pinned in the whole chain. Blocks follow in index order,
	// which is also the order the loop touches them, so a partial budget pins the
	// front of the run rather than a scattering of it.
	if (!im->host_cache_ready) {
		im->host_cache_ready = true;
		const u64 budget = host_cache_budget_bytes();
		u64 pinned = 0;
		i64 pinned_blocks = 0;
		// A pin is RAM the process cannot give back, so the loop stops at whichever
		// comes first: the budget, or the free-memory reserve. The second term is
		// what makes the cache safe on a machine that is already busy - pinning
		// "two thirds of the box" is fine arithmetic and an OOM kill in practice
		// when the box is 16 GB and the UI, the text tower and the file pages are
		// using the rest.
		const u64 reserve = host_cache_reserve_bytes();
		auto room_left = [&]() {
			const u64 free_now = SafeTensors::host_free_bytes();
			return free_now > reserve;
		};
		if (budget > 0) {
			try {
				for (const auto& r : im->refiner_ranges) {
					if (pinned + r.second > budget || !room_left()) break;
					im->st.pin_range(r.first, r.second);
					pinned += r.second;
				}
				for (i64 i = im->res_n; i < cfg_.n_layers; i++) {
					u64 blk = 0;
					for (const auto& r : im->block_ranges[(size_t)i]) blk += r.second;
					if (pinned + blk > budget || !room_left()) break;
					for (const auto& r : im->block_ranges[(size_t)i])
						im->st.pin_range(r.first, r.second);
					pinned += blk;
					pinned_blocks++;
				}
			} catch (const std::exception& e) {
				// An allocation failure is a reason to pin less, never a reason to
				// fail a run: the same bytes still come off the disk.
				fprintf(stderr, "[h3dit] host weight cache stopped early: %s\n", e.what());
			}
		}
		fprintf(stderr,
		        "[h3dit] host weight cache: %.2f GB pinned (%lld streamed blocks of %lld, "
		        "budget %.2f GB, %.2f GB free, %.2f GB held back) - the rest streams off "
		        "the device\n",
		        (double)pinned / 1073741824.0, (long long)pinned_blocks,
		        (long long)(cfg_.n_layers - im->res_n),
		        (double)budget / 1073741824.0,
		        (double)SafeTensors::host_free_bytes() / 1073741824.0,
		        (double)reserve / 1073741824.0);
	}

	for (i64 i = 0; i < cfg_.n_layers; i++) {
		if (i < im->res_n) {
			g_->ctx->submit_if_recording();
			g_->ctx->begin();
			run_block(F, im->res[(size_t)i], i);
			run_attn_mlp(F, im->res[(size_t)i], i);
			g_->ctx->submit();
			dump_hidden(*g_, F.hidden, i, S, hidden);
		} else {
			// new_layer() resets the weight arena (the previous block's weights are
			// unreferenced once its compute has been submitted) and the ring; each
			// load inside load_block submits on its own list.
			//
			// A refusal here is the one place the plan can be wrong *at run time*: the
			// driver's budget for this process shrinks while we sample (the WebView2 UI
			// repainting, the desktop compositor, the image chain's arenas), and the
			// check is against the live budget. The window is the only thing that can
			// be given back mid-run, and streaming needs strictly less than it, so the
			// answer is to shed it and retry rather than to fail a run whose text
			// encoder has already finished: the weights, the arithmetic and the output
			// are identical, only the disk traffic changes.
			BlockW W;
			// ── one block ahead on the disk (see SafeTensors::prefetch_ranges) ──
			//
			// The loader and the GPU used to take turns: read 368 MB, compute on it,
			// read the next 368 MB, ... At 540P/5s that is ~11 s of a 51 s step spent
			// with the card idle waiting for a disk the GPU cannot help with. So the
			// read for block i+1 is issued here - after block i's bytes are already on
			// the device, so the slot can be overwritten - and it runs on a worker
			// thread while block i's forward uses the GPU. The `read_file` calls the
			// loader makes then come out of that slot instead of the disk, so
			// `load_block` itself does not change.
			//
			// `res_n` cannot move inside this loop, so "is the next block streamed"
			// is a stable decision.
			auto prefetch_next = [&](i64 j) {
				if (j < im->res_n || j >= cfg_.n_layers) return;   // resident or past the end
				im->st.prefetch_ranges(im->block_ranges[(size_t)j]);
			};
			// Block i's bytes are in the slot iff the previous iteration issued the read
			// for i; joining an already-finished worker is free.
			im->st.prefetch_wait();
			auto load = [&] {
				ProfScope ps(g_, prof_sink(&H3Prof::stream));
				PR.stream_bytes += im->layer_bytes;
				g_->new_layer();
				load_block(im->st, *g_, *g_->wa, i, W, im->lora);
			};
			try {
				load();
			} catch (const std::exception& e) {
				const i64 had = im->res_n;
				if (!F.shed_resident || !F.shed_resident()) throw;
				fprintf(stderr,
				        "[h3dit] block %lld refused (%s); %lld-block resident window dropped, "
				        "every remaining block streams\n",
				        (long long)i, e.what(), (long long)had);
				g_->wa->release_chunks();   // whatever the refused load left behind
				load();
			}
			// Block i's bytes have been copied into the ring; the slot is free again.
			// Kick off block i+1's read before block i's forward is submitted.
			prefetch_next(i + 1);
			// No PrefetchVirtualMemory here any more: it only helps the *mapping*
			// path, and the block weights now come off the file with ReadFile
			// (upload_chunked_file), whose own read-ahead covers the next range.
			// Hinting the mapping as well would fault 368 MB into the working set
			// and then read the same bytes a second time.
			g_->ctx->submit_if_recording();
			g_->ctx->begin();
			run_block(F, W, i);
			run_attn_mlp(F, W, i);
			g_->end_layer();
			dump_hidden(*g_, F.hidden, i, S, hidden);
		}
	}

	// ── 6. final layer ─────────────────────────────────────────────────────
	g_->ctx->submit_if_recording();
	g_->ctx->begin();
	{
		ProfScope ps(g_, prof_sink(&H3Prof::final));
		run_final(F);
	}
	g_->ctx->submit();

	// ── 7. unpack ──────────────────────────────────────────────────────────
	const double t_unpack = now_ms();
	const std::vector<float> vr = g_->download_f32(F.vrows, (u64)ly.n_video * kVidPatch);
	const std::vector<float> ar = g_->download_f32(F.arows, (u64)ly.n_audio * kAudPatch);
	if (PR.on) PR.unpack += now_ms() - t_unpack;   // the two readbacks, not the host loop
	const i64 th = ly.tokens_h, tw = ly.tokens_w;
	for (i64 tt = 0; tt < vt; tt++)
		for (i64 hh = 0; hh < th; hh++)
			for (i64 ww = 0; ww < tw; ww++) {
				const float* row = &vr[(size_t)(((tt * th) + hh) * tw + ww) * kVidPatch];
				for (i64 c = 0; c < 24; c++) {
					const i64 co = (c * vt + tt) * lat_h * lat_w;
					for (i64 p = 0; p < 2; p++)
						for (i64 q = 0; q < 2; q++)
							out_v[(size_t)(co + (hh * 2 + p) * lat_w + ww * 2 + q)] =
							    -row[c * 4 + p * 2 + q];
				}
			}
	{
		std::vector<float> tmp(a_elems);
		for (i64 ch = 0; ch < 2; ch++)
			for (i64 tt = 0; tt < audio_t; tt++)
				for (i64 c = 0; c < 32; c++)
					tmp[(size_t)((c * 2 + ch) * audio_t + tt)] =
					    ar[(size_t)((ch * audio_t + tt) * kAudPatch + c)];
		// The model returns -[video_velocity, audio_velocity], and the audio velocity
		// is then unscaled in the reference's own order:
		//
		//     out[1] = (1 - scale) * (x_a_sampler * carry)
		//            + (1 + (scale - 1) * sigma_a) * net_out_a      # net_out_a = -head
		//
		// i.e. the carry term keeps its sign and only the head term is negated. The
		// first version negated the whole expression, which flips the carry term on a
		// 4x-scaled stream: the audio direction was wrong for every step it was
		// (audio_scale = shift / audio_shift = 12 / 3 = 4), not just a rounding.
		const float om = 1.0f - audio_scale;
		const float t2 = 1.0f + (audio_scale - 1.0f) * sigma_a;
		for (size_t i = 0; i < a_elems; i++)
			out_a[i] = om * (x_a[i] * carry) - t2 * tmp[i];
	}
	layout_.n_video = ly.n_video;
	layout_.n_audio = ly.n_audio;
	layout_.n_text = L;
	layout_.n_ref_video = ly.n_ref_video;
	layout_.n_ref_audio = ly.n_ref_audio;

	if (PR.on) {
		const double tail = now_ms() - t_fwd0;   // final layer scope + readback + unpack
		PR.total += tail;
		const double mb = (double)PR.stream_bytes / 1048576.0;
		const double sec = PR.stream / 1000.0;
		const double io_mb = (double)im->st.file_read_bytes() / 1048576.0;
		const double io_s = im->st.file_read_ms() / 1000.0;
		// `out_mlp` encloses the attention scope, because attention and out_proj now
		// run in the same per-chunk loop (the attention output is a chunk-tall buffer
		// that out_proj consumes immediately). Printing the raw total would count
		// attention twice against the sum of the parts, so the attention time is
		// subtracted back out here.
		const double out_mlp_only = PR.out_mlp - PR.attn;
		fprintf(stderr,
		        "[h3dit] S=%lld total %.1f ms | stream %.1f ms (%.0f MB, %.2f GB/s) | io %.0f MB "
		        "at %.2f GB/s | embed %.1f "
		        "| qkv %.1f | prep %.1f | attn %.1f | out+mlp %.1f | final %.1f | unpack %.1f "
		        "| resident %lld/%lld\n",
		        (long long)S, PR.total, PR.stream, mb, sec > 0.001 ? mb / 1024.0 / sec : 0.0, io_mb,
		        io_s > 0.001 ? io_mb / 1024.0 / io_s : 0.0, PR.embed, PR.qkv_gemm, PR.qkv_prep,
		        PR.attn, out_mlp_only, PR.final, PR.unpack, (long long)PR.resident,
		        (long long)PR.blocks);
		// What the *driver* says we are holding, next to what the ledger thinks: the
		// residency decision is only ever as good as the gap between those two, so
		// the instrument has to show both.
		const double ch_ms = im->st.cache_ms();
		fprintf(stderr,
		        "[h3dit] vram live %.2f/%.2f GB (ledger %.2f GB, limit %.2f GB) | host cache: "
		        "%.2f GB cached (%.2f GB page-locked), %.2f GB served from RAM in %.0f ms "
		        "(%.2f GB/s), %.2f GB off the device\n",
		        (double)vram_budget().live_usage() / 1073741824.0,
		        (double)vram_budget().os_budget() / 1073741824.0,
		        (double)vram_budget().local() / 1073741824.0,
		        (double)vram_budget().limit() / 1073741824.0,
		        (double)im->st.pinned_bytes() / 1073741824.0,
		        (double)im->st.host_locked_bytes() / 1073741824.0,
		        (double)im->st.cache_served_bytes() / 1073741824.0,
		        ch_ms,
		        ch_ms > 1.0 ? (double)im->st.cache_served_bytes() / 1073741824.0 / (ch_ms / 1000.0)
		                     : 0.0,
		        io_mb / 1024.0);
		im->st.reset_file_read_stats();
		im->st.reset_cache_stats();
		PR.stream = PR.embed = PR.qkv_gemm = PR.qkv_prep = PR.attn = PR.out_mlp = PR.final =
		    PR.unpack = PR.total = 0;
		PR.stream_bytes = 0;
		PR.io_bytes = 0;
	}
}
}  // namespace phi::media
