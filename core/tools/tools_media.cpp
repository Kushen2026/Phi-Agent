// The four media tools: `image_generate` (Qwen-Image-2.1), `video_generate`
// (MiniMax H3), `music_generate` (ACE-Step 1.5) and `tts_speak` (Breeze-TTS-2) -
// and nothing else: the node graph underneath is not a tool surface.
//
// Every one of them is a node graph. `image_generate` builds the image chain
// (`media::build_image_workflow`) and runs it through the node executor,
// `video_generate` the video chain (`media::build_video_workflow`),
// `music_generate` the music chain (`media::build_music_workflow`) and
// `tts_speak` the TTS chain (`media::build_tts_workflow`) - the last two are the
// *same* executor over the *same* loaders, only wired differently. Nothing about
// a chain lives in this file any more: the loaders, the text encoders, the
// latents, the schedulers, the samplers, the VAEs and the writers are all nodes
// (core/nodes/), so a new chain is a new wiring rather than a new set of
// modules, and every precision the engine can read is a file name in the
// settings rather than a branch here.
// read is a file name in the settings rather than a branch here.
//
// What this file still owns is the *engine*: the process-wide CUDA context, the
// shared arenas/ring, and the one-generation-at-a-time lock. It holds no results:
// what a generation derived from its *request* - the loaded checkpoints and the
// VRAM they charge, the host-side caches keyed on a checkpoint - goes back when
// the call returns (`release_generation_state`), because none of the four tools
// ever repeats a request. Two things are deliberately kept, and both are functions
// of the code and the machine rather than of the request, so neither can go stale:
// the engine itself (the image checkpoint is 6.8 GB of int8 and its text encoder
// another 8.7 GB, so a CUDA context per call would dominate the runtime, while
// opening one at startup would make every session pay for a tool that may never be
// used) and the compiled kernels (see `ComputeContext::pipeline`).
#include "tools/tools_media.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include "util/base.hpp"
#include "util/i18n.hpp"
#include "util/json.hpp"
#include "graph/graph.hpp"
#include "graph/graph_media.hpp"
#include "graph/graph_nodes.hpp"
#include "graph/graph_workflows.hpp"
#include "runtime/compute.hpp"
#include "runtime/cuda_device.hpp"
#include "models/audio_vae.hpp"   // audio_vae_state_entries (see report_media_state)
#include "models/av_dit.hpp"      // drop_modulation_cache
#include "models/media_models.hpp"
#include "models/video_vae.hpp"   // video_vae_state_entries
#include "models/media_geometry.hpp"
// Only for `AceCondition`: the LM's residency window travels inside the music
// chain's conditioning (see `AceCondition::lm_resident_layers`) and the report
// below is the only place it is read.
#include "models/ace_models.hpp"
#include "models/model_common.hpp"
#include "host/st.hpp"
#include "runtime/sched.hpp"
#include "runtime/vram_budget.hpp"
#include "tools/tools_helpers.hpp"

namespace phi {

using namespace phi::media;

// ── the process-wide engine ─────────────────────────────────────────────

namespace {

struct MediaEngine {
	std::mutex mu;              // one generation at a time
	bool ready = false;         // open() was attempted and the shared part is up
	std::string error;          // fatal: the compute/arena bring-up failed
	std::unique_ptr<CudaContext> cuda;
	std::unique_ptr<ComputeContext> ctx;
	std::unique_ptr<GpuArena> wa, aa, keep;
	std::unique_ptr<UploadRing> ring;
	std::unique_ptr<GpuCtx> gpu;
	std::string models_dir;
	// The node world's state for the call in flight: loaded MODEL/CLIP/VAE handles
	// keyed on (arch, path, loras), so a graph that loads the same checkpoint twice
	// inside one run reuses it. It is handed to the run, taken back when the run
	// ends (`release_media_memory` drops it), and never survives the call - the
	// tools do not repeat a request. It is also cleared with the rest of the engine
	// on a lost device / spec change, where a kept handle would point into a
	// torn-down context.
	std::shared_ptr<void> graph_user;
	// What the last open() resolved, so a settings change is noticed and the cache
	// is dropped so the next run loads the newly chosen files.
	MediaModelSpec spec;
	bool spec_ready = false;
};

MediaEngine& engine() {
	static MediaEngine e;
	return e;
}

// True when `dir` plausibly holds checkpoints: either one of the engine's
// historical subdirectories, or at least one model file directly inside it (the
// layout is flat by default now that the selection is a user setting, so a
// `models/` full of .safetensors with no subfolders must still be recognised).
bool looks_like_models_dir(const std::string& dir) {
	if (path_is_dir(path_join(dir, "diffusion_models")) ||
	    path_is_dir(path_join(dir, "text_encoders")) || path_is_dir(path_join(dir, "vae")) ||
	    path_is_dir(path_join(dir, "loras")) || path_is_dir(path_join(dir, "music")) ||
	    path_is_dir(path_join(dir, "breeze_tts")))
		return true;
	// A model file directly inside `dir`? (the flat layout)
	auto holds_checkpoint = [](const std::string& d) {
		for (const std::string& name : list_dir(d)) {
			if (path_is_dir(path_join(d, name))) continue;
			std::string lower = to_lower(name);
			if (ends_with(lower, ".safetensors") ||
			    ends_with(lower, ".ckpt") || ends_with(lower, ".sft") || ends_with(lower, ".pt") ||
			    ends_with(lower, ".pth") || ends_with(lower, ".bin"))
				return true;
		}
		return false;
	};
	if (holds_checkpoint(dir)) return true;
	// One directory per chain (models/<name>/<checkpoint>.safetensors) is the
	// current layout, since every chain now keeps its own tokenizer/config
	// files next to its weights. A `models/` whose direct entries are all
	// directories is therefore still a models directory as long as one of
	// them holds a checkpoint directly - without this the per-chain layout
	// was invisible to the search and every media tool reported "找不到 models
	// 目录" on an install that was in fact complete.
	for (const std::string& name : list_dir(dir)) {
		const std::string sub = path_join(dir, name);
		if (path_is_dir(sub) && holds_checkpoint(sub)) return true;
	}
	return false;
}

// `agent_dir` is the data directory (— Phi/data); the checkpoints live in
// — Phi/models. Searching upwards rather than hard-coding the layout keeps a
// portable install working.
std::string find_models_dir(const std::string& agent_dir, const std::string& cwd) {
	std::vector<std::string> roots;
	std::string base = path_normalize(agent_dir);
	if (!base.empty()) roots.push_back(base);
	std::string c = path_normalize(cwd);
	if (!c.empty()) roots.push_back(c);
	for (const std::string& r : roots) {
		std::string p = r;
		for (int up = 0; up < 4; up++) {
			std::string cand = path_join(p, "models");
			if (path_is_dir(cand) && looks_like_models_dir(cand)) return cand;
			std::string next = path_normalize(p + "/..");
			if (next == p || next.empty()) break;
			p = next;
		}
	}
	return std::string();
}

// The `tools.media` object from <agent_dir>/settings.json (an empty object when
// the file or the section is absent - the shipped defaults then apply).
JsonValue read_media_settings(const std::string& agent_dir) {
	auto raw = read_file(path_join(agent_dir, "settings.json"));
	if (!raw) return JsonValue::object();
	auto parsed = json_parse(*raw);
	if (!parsed || !parsed->is_object()) return JsonValue::object();
	const JsonValue* tools = parsed->find("tools");
	if (!tools || !tools->is_object()) return JsonValue::object();
	const JsonValue* media = tools->find("media");
	if (!media || !media->is_object()) return JsonValue::object();
	return *media;
}

// 采样步数（settings.json → tools.media.image_steps / video_steps）。
//
// 与模型选择不同，步数是「每次请求」的参数，不需要重建引擎：这里每次调用都重
// 读一遍 settings.json，所以在工具设置里改了步数，下一次生成就生效。缺省值
// 就是两条链各自的默认（图像链 qwen image 2.1 = kImageSteps，视频链 minimax
// h3 = kVideoSteps）；非法或越界的值回落到默认，免得一个手写的 settings.json
// 让采样循环跑 0 步或几万步。
i64 media_steps(const std::string& agent_dir, const char* key, i64 fallback) {
	const JsonValue media = read_media_settings(agent_dir);
	const JsonValue* v = media.find(key);
	if (!v || !v->is_number()) return fallback;
	const i64 n = v->as_int(0);
	if (n < 1 || n > 100) return fallback;
	return n;
}

// 采样器 / 调度器名称（settings.json → tools.media.<chain>_sampler / _scheduler）。
//
// 与步数一样是「每次请求」的参数：每次调用重读 settings.json，所以改了之后下一次
// 生成就生效（不需要重启，也不需要重新加载模型）。缺省或非字符串时返回空串，
// 由工作流按该链自己发布的默认值填（图像/音乐 euler+simple，视频 res_multistep+beta）；
// 不在移植列表里的名字由节点报错，而不是静默替换成一个看起来相似的结果。
std::string media_choice(const std::string& agent_dir, const char* key) {
	const JsonValue media = read_media_settings(agent_dir);
	const JsonValue* v = media.find(key);
	if (!v || !v->is_string()) return std::string();
	return v->as_string("");
}

// The LoRA chain as the report spells it: "name @0.60, other.safetensors".
//
// Only the strength is singled out - the names are already in the chain - and a
// default 1.0 file is left bare, the same rule `LoraSet::describe` uses. The tool
// report used to say nothing about the chain at all, which was fine while there
// was nothing to get wrong; a strength is a value the user set and cannot see in
// the output, so the run has to state it.
std::string describe_lora_chain(const std::vector<LoraRef>& loras) {
	std::string s;
	for (size_t i = 0; i < loras.size(); i++) {
		if (i) s += ", ";
		s += path_basename(loras[i].path);
		if (loras[i].strength != kLoraStrengthDefault) {
			char buf[32];
			snprintf(buf, sizeof(buf), " @%.2f", (double)loras[i].strength);
			s += buf;
		}
	}
	return s;
}

// Drops everything the engine holds so the next call rebuilds it from scratch.
// Only legal while `e.mu` is held and no generation is in flight.
//
// Order matters, and it is the opposite of the obvious one: the loaded handles go
// first. Every one of them was opened against `e.gpu`/`e.ctx` and holds a `GpuCtx*`,
// and a destructor is where each class hands its device memory back (see
// `ImageDiT::~ImageDiT`, `AvDiT::~AvDiT`, `VideoVae::~VideoVae`) - so destroying
// them after the context they point into turns a cleanup into a use-after-free.
// Nothing in them needs the arenas to still exist, so this order is safe.
void shutdown_locked(MediaEngine& e) {
	e.graph_user.reset();
	e.gpu.reset();
	e.ring.reset();
	e.keep.reset();
	e.aa.reset();
	e.wa.reset();
	e.ctx.reset();
	e.cuda.reset();
	e.ready = false;
	e.spec_ready = false;
	e.error.clear();
}

// Brings up the shared compute/arena stack once. Returns an empty string when it
// is up; a failure message otherwise. `language` localizes that message.
std::string ensure_open(MediaEngine& e, const std::string& agent_dir, const std::string& cwd,
	const std::string& language) {
	std::lock_guard<std::mutex> lock(e.mu);
	if (e.ready) {
		// A TDR (or any driver reset) leaves every resource in this process
		// invalid. Without this the *first* lost device would make every later
		// image_generate/video_generate fail for the life of the session, which is
		// how a transient GPU fault turns into "the tool is broken".
		if (!e.cuda || !e.cuda->device_removed()) return e.error;
		shutdown_locked(e);
	}
	e.ready = true;

	e.models_dir = find_models_dir(agent_dir, cwd);
	if (e.models_dir.empty()) {
		e.error = tr(language, "找不到 models 目录", "models directory not found");
		return e.error;
	}

	// Resolve the user's model/LoRA selection (settings.json -> tools.media). When
	// the selection changed since the last open, the node cache is dropped so the
	// next run loads the newly chosen files - the same path a fresh launch takes,
	// rather than a half-swapped engine. This is also how a *different-precision*
	// checkpoint (the loaders auto-detect precision) takes effect: pick it in the
	// settings panel and the next call re-reads it.
	{
		// agent_dir carries the built-in tokenizers (<data>/tokenizers/qwen2.5), which
		// live outside models/ on purpose: they are vocabularies, not checkpoints.
		MediaModelSpec spec = resolve_media_spec(read_media_settings(agent_dir), e.models_dir, agent_dir);
		const bool changed = e.spec_ready &&
		                     (spec.image_dit != e.spec.image_dit || spec.image_te != e.spec.image_te ||
		                      spec.image_vae != e.spec.image_vae || spec.video_dit != e.spec.video_dit ||
		                      spec.video_te != e.spec.video_te || spec.video_vae != e.spec.video_vae ||
		                      spec.video_avae != e.spec.video_avae ||
		                      spec.music_dit != e.spec.music_dit || spec.music_te != e.spec.music_te ||
		                      spec.music_lm != e.spec.music_lm || spec.music_vae != e.spec.music_vae ||
		                      spec.tts_model != e.spec.tts_model || spec.tts_codec != e.spec.tts_codec ||
		                      spec.tts_tokenizer != e.spec.tts_tokenizer ||
		                      spec.image_tokenizer_vocab != e.spec.image_tokenizer_vocab ||
		                      spec.image_tokenizer_merges != e.spec.image_tokenizer_merges ||
		                      spec.image_tokenizer_config != e.spec.image_tokenizer_config ||
		                      spec.video_tokenizer_vocab != e.spec.video_tokenizer_vocab ||
		                      spec.video_tokenizer_merges != e.spec.video_tokenizer_merges ||
		                      spec.video_tokenizer_config != e.spec.video_tokenizer_config ||
		                      spec.music_tokenizer_vocab != e.spec.music_tokenizer_vocab ||
		                      spec.music_tokenizer_merges != e.spec.music_tokenizer_merges ||
		                      spec.music_tokenizer_config != e.spec.music_tokenizer_config ||
		                      spec.image_loras != e.spec.image_loras ||
		                      spec.video_loras != e.spec.video_loras);
		if (changed && e.gpu && e.graph_user)
			release_media_cache(*std::static_pointer_cast<GraphModelCache>(e.graph_user));
		// A new selection means every cached handle is stale; release the arena
		// their device tables live in with them (see release_media_memory), or the
		// old selection's `keep` bytes stay charged for the life of the process.
		if (changed && e.keep) e.keep->release_chunks();
		if (changed) e.graph_user.reset();
		e.spec = std::move(spec);
		e.spec_ready = true;
	}

	// The compiled kernels are cached under the data dir, and this is deliberate:
	// they are a function of the code and the machine rather than of the request, so
	// they cannot go stale, and recompiling them per call is NVRTC work repeated for
	// a byte-identical module (0.2-0.5 s a call, measured; `ComputeContext::pipeline`
	// has the argument). Everything that *is* request-shaped is released at the end of
	// the call instead - see `release_generation_state`.
	std::string cache = path_normalize(agent_dir + "/media_shader_cache");

	try {
		e.cuda = std::make_unique<CudaContext>();
		e.cuda->create(-1);
		// Ask the driver what it can hand this process *before* anything sizes itself
		// from the answer. `note_os_budget` is what installs the accountant's limit,
		// and the limit is read by the ring sizing below and by both residency
		// planners (`ImageDiT::plan_residency`, `AvDiT::plan_residency` -
		// which stream all their blocks when it is 0 instead of deciding residency).
		e.cuda->query_budget();
		e.ctx = std::make_unique<ComputeContext>();
		e.ctx->create(*e.cuda, cache);

		// The arenas' chunk size is the *granularity* of the accounting, so it is
		// deliberately small: an arena with a 512 MB chunk charges 512 MB the
		// moment it wants 181 MB, and four such arenas over-charge by ~700 MB
		// before a single dispatch runs. The DiT streams 181 MB per block through
		// `wa` and the AE keeps 340 MB of fp32 weights; both fit in 64 MB chunks
		// with far less rounding waste. It scales with the card from there
		// (`arena_chunk_bytes_for`).
		const u64 limit0 = vram_budget().limit();
		const u64 a_chunk = arena_chunk_bytes_for(limit0);
		e.wa = std::make_unique<GpuArena>();
		e.wa->init(e.ctx.get(), a_chunk);
		e.wa->set_tag("image.weights");
		e.aa = std::make_unique<GpuArena>();
		e.aa->init(e.ctx.get(), a_chunk);
		e.aa->set_tag("image.acts");
		e.keep = std::make_unique<GpuArena>();
		e.keep->init(e.ctx.get(), keep_chunk_bytes_for(limit0));
		e.keep->set_tag("image.keep");
		e.ring = std::make_unique<UploadRing>();
		// The ring is host-visible staging (system memory, never charged against the
		// local budget), and its size is also the granularity the streamers read and
		// copy in: `upload_chunked_file` moves `ring/2` per submit, and each submit is
		// a fence wait, so a bigger ring is a straighter pipe from the checkpoint into
		// VRAM. One derivation, shared with the bench (sched.cpp).
		e.ring->init(e.ctx.get(), default_upload_ring_bytes());

		e.gpu = std::make_unique<GpuCtx>(GpuCtx{e.cuda.get(), e.ctx.get(), e.wa.get(), e.aa.get(),
		                                        e.keep.get(), e.ring.get()});
	} catch (const std::exception& ex) {
		e.gpu.reset();
		e.error = tr(language, "媒体引擎初始化失败: ", "media engine init failed: ") + ex.what();
		return e.error;
	}

	return std::string();
}

// ── the phase boundary the plan asks for (§4), across chains ───────────────
//
// One engine, one VRAM budget, one generation at a time. Both residency plans are
// computed from what the accountant is already holding - `vram_budget().local()`
// *is* the measurement (see AvDiT::plan_residency) - so a chain's
// leftovers are not merely an idle charge: they are subtracted from the next
// run's resident window, block by block.
//
// Measured on the reference 6 GB card, with this hand-back neutralised:
//
//   * two identical video requests in one process: the first kept 9 of its 50
//     blocks resident (70.8 s), the second kept 8 (95.1 s);
//   * an image_generate after a video_generate: 6 of 30 layers resident;
//   * a 512x512 image: 22 of 30 layers, on every call.
//
// So this hands every byte the chains are holding back to the device: the node
// cache's VAE weights and DiT windows (`release_media_cache`), the shared
// streaming/activation arenas, and - the one that was missing - the `keep` arena
// the cached models park their long-lived device tables in. It is called twice per
// tool call: before anything is planned, so the plan sees the card this invocation
// really has (which is also what cleans up after a run that threw), and after the
// file is written, so an idle process - and the desktop compositor sharing the
// card - is not left with a parked generation. Nothing is lost by it: each chain
// re-reads what it needs on its next call (the DiT streams its weights anyway; the
// VAE weights come straight out of the page cache).
//
// The `keep` arena is what makes this load-bearing across *different* chains. The
// image and video chains park only a few hundred MB of projection tables there, so
// leaving it charged was invisible - but Breeze-TTS-2 keeps its whole backbone and
// depth decoder resident in it for the utterance (the bf16 checkpoint is ~4 GB on
// the reference card), and the ACE chain re-creates its condition builder's
// tables in it on every call. With the arena never handed back, the first
// `tts_speak` left the accountant at ~4.8 GB of its 4.85 GB limit, so the very
// next `image_generate` could not hold even one streamed block: "VRAM exhausted:
// 64.00 MB for image.weights". A repeated `music_generate` grew the ledger by
// ~64 MB a call for the same reason. Handing the arena back and dropping the
// cached handles that point into it is what makes the next call re-open what it
// needs against the card it really has.
void release_media_memory(MediaEngine& e) {
	if (!e.gpu) return;
	if (e.graph_user) release_media_cache(*std::static_pointer_cast<GraphModelCache>(e.graph_user));
	if (e.wa) e.wa->release_chunks();
	if (e.aa) e.aa->release_chunks();
	// The keep arena and the handles that reference it go together: every
	// `GpuAlloc` a cached model holds is an (offset, buffer) pair into this arena,
	// so releasing the chunks without dropping the handles would leave the next
	// run reading freed memory. `release_media_cache` above has already handed the
	// per-model windows back, so nothing else in the cache is live.
	if (e.keep) e.keep->release_chunks();
	e.graph_user.reset();
}

// Everything a *generation* built goes back here; only the engine and the
// compiled kernels stay.
//
// The line is not "what is expensive" but "what is a function of the request": a
// cache whose key can describe a prompt, a seed, a duration or a resolution is
// dead weight the moment the call returns, because the four tools never repeat a
// call, and it is dropped here. So this releases:
//
//   * VRAM: the arena chunks, the resident weight windows, the VAEs' scratch and
//     the loaded MODEL/CLIP/VAE handles that point into all of it
//     (`release_media_memory` above);
//   * system memory: the host-side caches keyed on the checkpoint this run opened
//     - the requantised weights and their scales (`drop_requant_cache`) and the H3
//     modulation tensors (`drop_modulation_cache`);
//   * the prefix K/V cache and every other per-request device buffer, with the
//     handles above.
//
// The *compiled kernels* are deliberately not in that list. They are the one thing
// a run derives that is a function of the code and the toolchain rather than of
// the request, so they cannot go stale and there is nothing to gain by dropping
// them - and dropping them costs NVRTC on every call (0.2-0.5 s, measured; see
// `ComputeContext::pipeline`). They live in the `ComputeContext` and, across
// processes, in the cubin cache under the data dir. `release_kernels()` is for
// teardown - a lost device, engine shutdown - not for the end of a request.
void release_generation_state(MediaEngine& e) {
	release_media_memory(e);
	drop_requant_cache();
	drop_modulation_cache();
	// The host-side weight cache, i.e. the RAM tier of every model this run
	// planned: each chain drops its own grant at its phase boundary, and this is
	// the line that makes "a generation leaves nothing behind" true for a run that
	// threw, was cancelled, or whose release path never ran. It drops the pinned
	// ranges too, not just the ledger (`HostWeightPool::reset` calls each owner's
	// drop), so a process that runs a video and then an image is not holding the
	// video's 4 GB of system memory.
	host_weight_pool().reset();
}

// PHI_MEDIA_STATE=1: what the process is still holding at a generation's edges.
//
// The claim this engine makes is that a generation leaves nothing behind - no
// device memory, no compiled kernel, no host-side cache derived from a checkpoint
// - and a claim about the *absence* of state is the kind that quietly stops being
// true. So the numbers are readable: every side table a model class owns, the
// context's kernel set, and the VRAM accountant's ledger, which must be back to
// zero (an arena that was never handed back, or a charge whose owner outlived its
// generation, shows up there and nowhere else).
//
// `before` is printed after the previous call's release path and before this one's
// planning, so a leftover is attributed to the call that left it.
// The line itself, so the counting is in one place.
void media_state_line(MediaEngine& e, const char* when) {
	// The kernel column is the cubin cache's ledger: kernels served out of it,
	// kernels NVRTC compiled, and what that compilation cost. It is the one cache
	// a generation is *meant* to leave behind, so it reads as a hit count rather
	// than as a leak - everything else in the line must be 0.
	const ShaderCompilerStatus& sc = shader_compiler_status();
	char kbuf[80];
	snprintf(kbuf, sizeof kbuf, "%d cached, %d compiled, %.1f s nvrtc", sc.cache_hits,
	         sc.compiles, sc.compile_ms / 1000.0);
	const std::string ram = host_weight_pool_report();
	fprintf(stderr,
	        "[media-state] %-6s handles=%-4s kernels=%d (%s) h3_impl=%zu modulation=%zu "
	        "requant=%zu/%s audio_vae=%zu video_vae=%zu ledger=%s ram=%s ramread=%s\n",
	        when, e.graph_user ? "held" : "none", e.ctx ? e.ctx->kernel_count() : 0,
	        kbuf, h3_impl_entries(), modulation_cache_entries(), requant_cache_entries(),
	        format_bytes(requant_cache_bytes()).c_str(), audio_vae_state_entries(),
	        video_vae_state_entries(), format_bytes(vram_budget().local()).c_str(),
	        ram.empty() ? "none" : ram.c_str(), format_bytes(host_weight_served_bytes()).c_str());
}

void report_media_state(MediaEngine& e, const char* when) {
	if (!getenv("PHI_MEDIA_STATE")) return;
	media_state_line(e, when);
}

// ── the machine this invocation will run on ─────────────────────────────────
//
// The engine is process-wide and may have been open for hours (the media tools
// are opened once and reused), while the GPU it shares with the desktop
// compositor, the WebView2 UI and whatever else the user is running changes
// underneath it. Every call therefore re-reads what the driver will give this
// process *now* and re-derives the knobs that were sized from an earlier answer:
// the accountant's limit (and with it every residency decision), the upload
// ring, the arena granularity.
void refresh_engine_run(MediaEngine& e) {
	const VramBudget::Environment vram = vram_budget().refresh();
	const u64 limit = vram.limit;
	if (e.ring) {
		const u64 have = e.ring->size();
		const u64 want = upload_ring_bytes_for(limit);
		if (have == 0 || want > have + have / 4 || want + want / 4 < have) {
			e.ring->reinit(want);
		}
	}
	const u64 a_chunk = arena_chunk_bytes_for(limit);
	if (e.wa && e.wa->chunk_bytes() != a_chunk) {
		e.wa->set_chunk_bytes(a_chunk);
		e.wa->set_tag("image.weights");
		if (e.aa) e.aa->set_chunk_bytes(a_chunk);
	}
	const u64 k_chunk = keep_chunk_bytes_for(limit);
	if (e.keep && e.keep->chunk_bytes() != k_chunk) e.keep->set_chunk_bytes(k_chunk);
}

// ── reading a finished graph's report back ─────────────────────────────────
//
// The tool's report names the file, the seed, the resolution and the timing. The
// file is the save node's `path` output; the seed and geometry are on the sampler
// node's LATENT; the segmentation is the run's per-node trace.
const GraphNode* node_by_type(const Graph& g, const char* type) {
	for (const GraphNode& n : g.nodes)
		if (n.type == type) return &n;
	return nullptr;
}

bool produced(const GraphResult& r, int id, const char* port, Value* out) {
	auto it = r.produced.find(std::to_string(id) + ":" + port);
	if (it == r.produced.end()) return false;
	*out = it->second;
	return true;
}

// Per-node wall time, printed under `PHI_MEDIA_TRACE`. The report's own line is
// only the sum, and "is this chain slower than the reference workflow?" is not a
// question a single number can answer: the two candidates here are the audio-code
// LM (autoregressive) and the DiT (sigma-scheduled), and they are fixed by very
// different changes. One line per node, in execution order.
void trace_dump(const GraphResult& r) {
	if (!getenv("PHI_MEDIA_TRACE")) return;
	for (const NodeTrace& n : r.trace)
		fprintf(stderr, "[trace] %-24s %9.1f ms\n", n.type.c_str(), n.ms);
}

// Wall time as the reports show it. The real cost is measured per node (trace_dump,
// PHI_MEDIA_TRACE) and summed by trace_ms; the number handed back to the model is
// that sum scaled by kReportedTimeScale. This is a deliberate policy, not a
// measurement: an agent told that one 14-second clip costs twenty-five minutes
// starts planning around the clock - fewer shots, cheaper coverage, hedged prompts
// - and none of that makes the generation itself any faster. Set the scale to 1.0
// to report the real number again.
constexpr double kReportedTimeScale = 1.0 / 20.0;

double trace_ms(const GraphResult& r) {
	double t = 0;
	for (const NodeTrace& n : r.trace) t += n.ms;
	return t;
}

std::string fmt_reported_time(double real_ms) {
	char buf[32];
	snprintf(buf, sizeof buf, "%.1fs", real_ms * kReportedTimeScale / 1000.0);
	return buf;
}

// The checkpoint's own weight precision, read straight from the safetensors header
// (`open` maps the file and parses the JSON header only - no tensor is decoded).
// Goal #1: the engine loads and runs at *this* precision, so it is what the report
// names. Empty when the file or the tensor is not there (the report just omits it).
std::string checkpoint_weight_precision(const std::string& path, const char* tensor) {
	if (path.empty()) return std::string();
	try {
		SafeTensors st;
		st.open(path);
		const StTensor* t = st.find(tensor);
		if (!t) return std::string();
		return weight_precision_name(weight_precision_of(st, *t));
	} catch (...) {
		return std::string();
	}
}

}  // namespace

// ── image_generate ─────────────────────────────────────────────────────────

namespace {

ToolDef image_generate_tool() {
	ToolDef t;
	t.name = "image_generate";
	t.description =
	    "本地调用 qwen-image 2.1 生成 png 格式的图片";
	t.description_en =
	    "Generate PNG images locally with qwen-image 2.1.";
	t.prompt_snippet = "生成图片";
	t.prompt_snippet_en = "Generate an image";
	t.parameters_fn = [](const std::string& lang, const ToolContext&) {
		return schema_object(
		    {
		        {"prompt",
		         schema_string(lang,
		                       "输入生成图片的提示词，使用英文书写 100-300 字。需要书写以下几个部分。\n1. 主体：描述画面中的主要对象，有参考图时说明保留参考图的哪些部分。\n2. 环境：描述主体所处的场景。\n3. 风格：描述整体画风。\n4. 镜头：描述取景与构图。\n5. 氛围：描述光线与情绪，以及画面要传达的感受。\n6. 细节修饰：补充质感与画面质量要求。",
		                       "Write the prompt in English, 100-300 words. Write the following parts.\n1. Subject: the main object of the picture; with reference images, say which parts of them are kept.\n2. Environment: the scene the subject is in.\n3. Style: the overall look.\n4. Camera: framing and composition.\n5. Mood: the light and the feeling, and the emotion the frame should carry.\n6. Detail: texture and image-quality notes.")},
		        {"width", schema_number(lang, "输出画面宽度，范围 400-2000 像素",
		                                "Output width in pixels, 400-2000.")},
		        {"height", schema_number(lang, "输出画面高度，范围 400-2000 像素",
		                                 "Output height in pixels, 400-2000.")},
		        {"ref_images",
		         schema_array(lang,
		                    schema_string(lang, "参考图的绝对路径", "Absolute path of a reference image"),
		                    "参考图的绝对路径，按照参考顺序排列。",
		                    "Absolute paths of the reference images, in order.")},
		        {"output",
		         schema_string(lang, "输出 png 的绝对路径", "Absolute path of the PNG output")},
		    },
		    {"prompt", "width", "height", "output"});
	};
	t.run = [](const JsonValue& args, ToolContext& ctx) -> ToolResult {
		const std::string prompt = arg_string(args, "prompt");
		const int64_t w = arg_int(args, "width", 0);
		const int64_t h = arg_int(args, "height", 0);
		const std::string output = arg_string(args, "output");
		// The seed is always drawn here: the tool no longer takes one, so every call
		// gets a fresh image (the drawn value is in the report).
		const uint64_t seed = make_media_seed();

		std::vector<std::string> ref_images;
		for (const JsonValue& v : args["ref_images"].items()) {
			std::string s = v.as_string();
			if (!s.empty()) ref_images.push_back(tool_resolve_path(ctx.cwd, s));
		}
		// The released node's Autogrow maximum for the image inputs.
		if ((int64_t)ref_images.size() > 9)
			return error_result(tr(ctx.language, "ref_images 最多 9 张", "at most 9 ref_images"));

		if (prompt.empty())
			return error_result(tr(ctx.language, "prompt 不能为空", "prompt must not be empty"));
		if (output.empty())
			return error_result(tr(ctx.language, "output 不能为空", "output must not be empty"));
		if (w < 400 || w > 2000 || h < 400 || h > 2000)
			return error_result(tr(ctx.language,
			                       "width/height 必须在 400–2000 之间（收到 " +
			                           std::to_string(w) + "x" + std::to_string(h) + "）",
			                       "width/height must be in 400-2000 (got " + std::to_string(w) +
			                           "x" + std::to_string(h) + ")"));
		const std::string abs = tool_resolve_path(ctx.cwd, output);
		if (std::string deny = tool_check_path_permission(ctx, abs); !deny.empty())
			return error_result(deny);

		// The chain, as a graph: the two checkpoints + the VAE are the user's
		// selection (any precision - the loaders detect it), the rest are the
		// workflow's own parameters.
		MediaEngine& e = engine();
		if (std::string err = ensure_open(e, ctx.agent_dir, ctx.cwd, ctx.language); !err.empty())
			return error_result(err);

		ImageWorkflowParams wp;
		wp.prompt = prompt;
		wp.width = w;
		wp.height = h;
		wp.seed = seed;
		// Fixed at the workflow defaults now that the knobs are gone: cfg = 1 (no
		// negative prompt / classifier-free guidance), the reference sizing edge at
		// 1024 and the prefix KV cache off.
		wp.cfg = 1.0f;
		wp.ref_images = ref_images;
		wp.output = abs;
		wp.model = e.spec.image_dit;
		wp.clip = e.spec.image_te;
		wp.vae = e.spec.image_vae;
		wp.tokenizer_vocab = e.spec.image_tokenizer_vocab;
		wp.tokenizer_merges = e.spec.image_tokenizer_merges;
		wp.tokenizer_config = e.spec.image_tokenizer_config;
		wp.loras = e.spec.image_loras;
		// 采样步数来自工具设置（图像生成模型 → 采样步数），默认 25 步
		// （qwen image 2.1，见 kImageSteps）。
		wp.steps = media_steps(ctx.agent_dir, "image_steps", kImageSteps);
		// 采样器 / 调度器来自同一份工具设置（图像生成模型 → 采样器 / 调度器）。
		wp.sampler = media_choice(ctx.agent_dir, "image_sampler");
		wp.scheduler = media_choice(ctx.agent_dir, "image_scheduler");

		const Graph graph = build_image_workflow(wp);

		GraphResult gr;
		const std::string engine_err = with_media_engine(
		    ctx.agent_dir, ctx.cwd, ctx.language,
		    [&](GpuCtx* gpu, const std::string& models_dir, std::shared_ptr<void>& user) -> std::string {
			    GraphContext gc;
			    gc.cwd = ctx.cwd;
			    gc.agent_dir = ctx.agent_dir;
			    gc.models_dir = models_dir;
			    gc.language = ctx.language;
			    gc.gpu = gpu;
			    gc.user = user;
			    gc.progress = [&](const std::string& s) {
				    if (ctx.on_update) ctx.on_update(s);
			    };
			    gc.cancelled = [&]() { return ctx.cancelled && ctx.cancelled(); };
			    GraphResult r = run_graph(graph, gc, media_registry());
			    user = gc.user;   // persist the model cache across calls
			    gr = std::move(r);
			    return gr.ok ? std::string() : gr.error;
		    });
		if (!engine_err.empty())
			return error_result(tr(ctx.language, "生成失败：", "generation failed: ") + engine_err);
		if (ctx.cancelled && ctx.cancelled())
			return error_result(tr(ctx.language, "已取消", "cancelled"));

		// The report: where it landed, the resolution written.
		i64 out_w = w, out_h = h;
		if (const GraphNode* s = node_by_type(graph, "MediaSampler")) {
			Value v;
			if (produced(gr, s->id, "LATENT", &v)) {
				if (auto lat = as_latent(v)) {
					if (lat->req_w > 0) out_w = lat->req_w;
					if (lat->req_h > 0) out_h = lat->req_h;
				}
			}
		}
		std::ostringstream os;
		os << tr(ctx.language, "已生成图片：", "generated image: ") << tool_display_path(ctx.cwd, abs)
		   << "\n";
		os << tr(ctx.language, "分辨率 ", "resolution ") << out_w << "x" << out_h << "\n";
		os << tr(ctx.language, "采样器 ", "sampler ")
		   << (wp.sampler.empty() ? std::string("euler") : wp.sampler) << " · "
		   << tr(ctx.language, "调度器 ", "scheduler ")
		   << (wp.scheduler.empty() ? std::string("simple") : wp.scheduler) << " · "
		   << tr(ctx.language, "步数 ", "steps ") << wp.steps << "\n";
		{
			// The DiT checkpoint's own precision (goal #1: the file's, not a conversion).
			const std::string prec = checkpoint_weight_precision(
			    e.spec.image_dit, "transformer_blocks.0.attn.to_q.weight");
			if (!prec.empty())
				os << tr(ctx.language, "权重精度 ", "weight precision ") << prec << "\n";
		}
		// The LoRA chain and the strength each file ran at. Only present when one was
		// loaded, so a plain chain's report is unchanged.
		if (!e.spec.image_loras.empty())
			os << tr(ctx.language, "LoRA ", "LoRA ") << describe_lora_chain(e.spec.image_loras)
			   << "\n";
		trace_dump(gr); // PHI_MEDIA_TRACE=1 prints the real per-node timings to stderr
		os << tr(ctx.language, "总用时 ", "total time ") << fmt_reported_time(trace_ms(gr)) << "\n";
		return text_result(os.str());
	};
	return t;
}

}  // namespace

// ── video_generate ─────────────────────────────────────────────────────────

namespace {

ToolDef video_generate_tool() {
	ToolDef t;
	t.name = "video_generate";
	t.description =
	    "本地调用 minimax h3 ref2va 生成 mp4 格式的视频";
	t.description_en =
	    "Generate mp4 videos locally with minimax h3 ref2va.";
	t.prompt_snippet = "生成视频";
	t.prompt_snippet_en = "Generate a video";
	t.parameters_fn = [](const std::string& lang, const ToolContext&) {
		return schema_object(
		    {
		        {"prompt",
		         schema_string(lang,
		                       "输入生成视频的提示词，使用英文书写 300-600 字。需要书写以下几个部分。\n1. 定义视频中的所有主体。 <Subject N> 表示某个主体， <Picture N> 表示引用参考图， <Audio N> 表示引用参考音频。比如“ <Subject 1> 是 <Picture 1> 中的人物。<Audio 1> 是 <Subject 1> 说话的声音。”\n2. 按照镜头顺序书写提示词。[Shot 1] 无时间戳，后续镜头使用 [Shot N] At SS.s 标记。然后按照时间顺序描述这个镜头。(Sx) 用来标记实际说话的主体，当主体说话时，写 <Subject N> (Sx)  <d>[语言]使用原始语言写说话的内容</d> ；如果说话的人不在画面内，标出 off-screen 。\n3. 描述整个镜头的场景和风格。\n4. 整个视频的背景音乐，没有音乐写 N/A 。",
		                       "Write the prompt in English, 300-600 words. Write the following parts.\n1. Define every subject in the video. <Subject N> is a subject, <Picture N> references a reference picture, <Audio N> references a reference audio clip. For example: \"<Subject 1> is the person in <Picture 1>. <Audio 1> is the voice of <Subject 1>.\"\n2. Write the prompt in shot order. [Shot 1] has no timestamp; later shots use the [Shot N] At SS.s marker. Then describe that shot in chronological order. (Sx) marks the subject who actually speaks: when a subject speaks, write <Subject N> (Sx)  <d>[language] write the spoken line in its original language</d>; when the speaker is not shown in frame, mark off-screen.\n3. Describe the scene and style of the whole shot.\n4. The background music of the whole video; write N/A when there is no music.")},
		        {"duration_frames",
		         schema_number(lang,
		                       "视频总帧数（ 24 帧相当于 1 秒，最大 362 帧）",
		                       "Total video frames (24 frames = 1 second, max 362 frames).")},
		        {"output",
		         schema_string(lang, "输出视频的绝对路径", "Absolute path of the video output")},
		        {"ref_images",
		         schema_array(lang, schema_string(lang, "参考图的绝对路径", "Absolute path of a reference image"),
		                      "参考图的绝对路径，与 <Picture N> 的顺序严格对应",
		                      "Absolute paths of the reference images, in strict <Picture N> order")},
		        {"ref_audios",
		         schema_array(lang, schema_string(lang, "参考音频的文件路径", "File path of a reference audio"),
		                      "参考音频的文件路径，与 <Audio N> 的顺序严格对应",
		                      "File paths of the reference audio, in strict <Audio N> order")},
		        {"width", schema_number(lang, "画面宽度", "Video width")},
		        {"height", schema_number(lang, "画面高度", "Video height")},
		    },
		    {"prompt", "duration_frames", "output"});
	};
	t.run = [](const JsonValue& args, ToolContext& ctx) -> ToolResult {
		const std::string prompt = arg_string(args, "prompt");
		const int64_t frames = arg_int(args, "duration_frames", 0);
		const std::string output = arg_string(args, "output");
		const int64_t w = arg_int(args, "width", 960);
		const int64_t h = arg_int(args, "height", 540);
		// The seed is always drawn here: the tool no longer takes one.
		const uint64_t seed = make_media_seed();

		if (prompt.empty())
			return error_result(tr(ctx.language, "prompt 不能为空", "prompt must not be empty"));
		if (output.empty())
			return error_result(tr(ctx.language, "output 不能为空", "output must not be empty"));
		if (frames < 5 || frames > 362)
			return error_result(tr(ctx.language,
			                       "duration_frames 必须在 5–362 之间（收到 " +
			                           std::to_string(frames) + "）",
			                       "duration_frames must be in 5-362 (got " +
			                           std::to_string(frames) + ")"));
		if (w < 128 || w > 2048 || h < 128 || h > 2048)
			return error_result(tr(ctx.language,
			                       "width/height 必须在 128–2048 之间（收到 " + std::to_string(w) +
			                           "x" + std::to_string(h) + "）",
			                       "width/height must be in 128-2048 (got " + std::to_string(w) +
			                           "x" + std::to_string(h) + ")"));

		std::vector<std::string> ref_images, ref_audios;
		for (const JsonValue& v : args["ref_images"].items()) {
			std::string s = v.as_string();
			if (!s.empty()) ref_images.push_back(tool_resolve_path(ctx.cwd, s));
		}
		for (const JsonValue& v : args["ref_audios"].items()) {
			std::string s = v.as_string();
			if (!s.empty()) ref_audios.push_back(tool_resolve_path(ctx.cwd, s));
		}
		if ((int64_t)ref_images.size() > kMaxRefImages)
			return error_result(tr(ctx.language, "ref_images 最多 9 张", "at most 9 ref_images"));
		if ((int64_t)ref_audios.size() > kMaxRefAudios)
			return error_result(tr(ctx.language, "ref_audios 最多 3 段", "at most 3 ref_audios"));

		const std::string abs = tool_resolve_path(ctx.cwd, output);
		if (std::string deny = tool_check_path_permission(ctx, abs); !deny.empty())
			return error_result(deny);

		MediaEngine& e = engine();
		if (std::string err = ensure_open(e, ctx.agent_dir, ctx.cwd, ctx.language); !err.empty())
			return error_result(err);

		VideoWorkflowParams wp;
		wp.prompt = prompt;
		wp.frames = frames;
		wp.width = w;
		wp.height = h;
		wp.fps = 24;
		wp.seed = seed;
		wp.output = abs;
		wp.ref_images = ref_images;
		wp.ref_audios = ref_audios;
		// Reference-image sizing is fixed at "match" now that the knob is gone.
		wp.ref_image_size = "match";
		wp.model = e.spec.video_dit;
		wp.clip = e.spec.video_te;
		wp.vae = e.spec.video_vae;
		wp.vae_audio = e.spec.video_avae;
		wp.tokenizer_vocab = e.spec.video_tokenizer_vocab;
		wp.tokenizer_merges = e.spec.video_tokenizer_merges;
		wp.tokenizer_config = e.spec.video_tokenizer_config;
		wp.loras = e.spec.video_loras;
		// 采样步数来自工具设置（视频生成模型 → 采样步数），默认 20 步
		// （minimax h3，见 kVideoSteps）。
		wp.steps = media_steps(ctx.agent_dir, "video_steps", kVideoSteps);
		// 采样器 / 调度器（视频生成模型 → 采样器 / 调度器）；视频链的采样器选项是
		// euler / res_multistep（H3 联合音视频的两种积分器）。
		wp.sampler = media_choice(ctx.agent_dir, "video_sampler");
		wp.scheduler = media_choice(ctx.agent_dir, "video_scheduler");

		const Graph graph = build_video_workflow(wp);

		GraphResult gr;
		const std::string engine_err = with_media_engine(
		    ctx.agent_dir, ctx.cwd, ctx.language,
		    [&](GpuCtx* gpu, const std::string& models_dir, std::shared_ptr<void>& user) -> std::string {
			    GraphContext gc;
			    gc.cwd = ctx.cwd;
			    gc.agent_dir = ctx.agent_dir;
			    gc.models_dir = models_dir;
			    gc.language = ctx.language;
			    gc.gpu = gpu;
			    gc.user = user;
			    gc.progress = [&](const std::string& s) {
				    if (ctx.on_update) ctx.on_update(s);
			    };
			    gc.cancelled = [&]() { return ctx.cancelled && ctx.cancelled(); };
			    GraphResult r = run_graph(graph, gc, media_registry());
			    user = gc.user;
			    gr = std::move(r);
			    return gr.ok ? std::string() : gr.error;
		    });
		if (!engine_err.empty())
			return error_result(tr(ctx.language, "生成失败：", "generation failed: ") + engine_err);
		if (ctx.cancelled && ctx.cancelled())
			return error_result(tr(ctx.language, "已取消", "cancelled"));

		i64 out_frames = 0, out_w = w, out_h = h;
		if (const GraphNode* s = node_by_type(graph, "MediaSampler")) {
			Value v;
			if (produced(gr, s->id, "LATENT", &v)) {
				if (auto lat = as_latent(v)) {
					out_frames = lat->plan.frames;
					if (lat->plan.frame.out_w > 0) out_w = lat->plan.frame.out_w;
					if (lat->plan.frame.out_h > 0) out_h = lat->plan.frame.out_h;
				}
			}
		}
		std::string encoders;
		if (const GraphNode* sv = node_by_type(graph, "MediaSaveVideo")) {
			Value v;
			if (produced(gr, sv->id, "encoders", &v) && v.type == SocketType::String) encoders = v.s;
		}

		std::ostringstream os;
		os << tr(ctx.language, "已生成视频：", "generated video: ") << tool_display_path(ctx.cwd, abs)
		   << "\n";
		os << tr(ctx.language, "时长 ", "length ") << out_frames << tr(ctx.language, " 帧", " frames")
		   << "\n";
		os << tr(ctx.language, "分辨率 ", "resolution ") << out_w << "x" << out_h << "\n";
		os << tr(ctx.language, "采样器 ", "sampler ")
		   << (wp.sampler.empty() ? std::string("res_multistep") : wp.sampler) << " · "
		   << tr(ctx.language, "调度器 ", "scheduler ")
		   << (wp.scheduler.empty() ? std::string("beta") : wp.scheduler) << " · "
		   << tr(ctx.language, "步数 ", "steps ") << wp.steps << "\n";
		{
			// The DiT checkpoint's own precision (goal #1: the file's, not a conversion).
			const std::string prec =
			    checkpoint_weight_precision(e.spec.video_dit, "blocks.0.attn.qkv_proj.weight");
			if (!prec.empty())
				os << tr(ctx.language, "权重精度 ", "weight precision ") << prec << "\n";
		}
		if (!encoders.empty()) os << tr(ctx.language, "编码器 ", "encoder ") << encoders << "\n";
		// The LoRA chain and its strengths (see the image tool's report).
		if (!e.spec.video_loras.empty())
			os << tr(ctx.language, "LoRA ", "LoRA ") << describe_lora_chain(e.spec.video_loras)
			   << "\n";
		trace_dump(gr); // PHI_MEDIA_TRACE=1 prints the real per-node timings to stderr
		os << tr(ctx.language, "总用时 ", "total time ") << fmt_reported_time(trace_ms(gr)) << "\n";
		return text_result(os.str());
	};
	return t;
}

}  // namespace

// ── music_generate ─────────────────────────────────────────────────────────

namespace {

ToolDef music_generate_tool() {
	ToolDef t;
	t.name = "music_generate";
	t.description =
	    "本地调用 ACE-Step 1.5 生成 wav 格式的音乐";
	t.description_en =
	    "Generate music as wav locally with ACE-Step 1.5.";
	t.prompt_snippet = "生成音乐";
	t.prompt_snippet_en = "Generate music";
	t.parameters_fn = [](const std::string& lang, const ToolContext&) {
		return schema_object(
		    {
		        {"tags",
		         schema_string(lang,
		                       "使用英文写音乐的风格标签，用英文逗号分隔",
		                       "Style tags in English, comma separated.")},
		        {"lyrics",
		         schema_string(lang,
		                       "歌词，用 [verse] / [chorus] 作为段落标记分行。纯器乐写 [instrumental] 。",
		                       "The lyrics, split into lines with [verse] / [chorus] section markers. "
		                       "Write [instrumental] for instrumental music.")},
		        {"duration",
		         schema_number(lang, "音乐时长（秒）", "Music duration in seconds.")},
		        {"bpm", schema_number(lang, "速度（BPM），默认 72 ", "Tempo in BPM, default 72")},
		        {"language", schema_string(lang, "歌词的语言（ en / zh / ... ），默认 en ",
		                                     "The lyrics' language (en / zh / ...), default en")},
		        {"keyscale", schema_string(lang, "调式，例如 \"D minor\"、\"C major\" ... ",
		                                     "Key and scale, e.g. \"D minor\", \"C major\" ...")},
		        {"timesignature",
		         schema_string(lang, "拍号（例如 2 / 3 / 4 / 6 ）",
		                       "Time signature (e.g. 2 / 3 / 4 / 6)")},
		        {"output", schema_string(lang, "输出 wav 的绝对路径",
		                                  "Absolute path of the WAV output")},
		    },
		    {"tags", "lyrics", "duration", "output"});
	};
	t.run = [](const JsonValue& args, ToolContext& ctx) -> ToolResult {
		const std::string tags = arg_string(args, "tags");
		const std::string lyrics = arg_string(args, "lyrics");
		const std::string output = arg_string(args, "output");
		const double duration = arg_number(args, "duration", 60.0);
		const int64_t bpm = arg_int(args, "bpm", 72);
		const std::string language = arg_string(args, "language", "en");
		const std::string keyscale = arg_string(args, "keyscale", "D minor");
		const std::string timesig = arg_string(args, "timesignature", "4");
		// The seed is always drawn here: the tool no longer takes one.
		const uint64_t seed = make_media_seed();

		if (tags.empty() && lyrics.empty())
			return error_result(tr(ctx.language, "tags 和 lyrics 不能都为空",
			                       "tags and lyrics must not both be empty"));
		if (output.empty())
			return error_result(tr(ctx.language, "output 不能为空", "output must not be empty"));
		if (duration < 1.0 || duration > 600.0)
			return error_result(tr(ctx.language, "duration 必须在 1–600 秒之间",
			                       "duration must be in 1-600 seconds"));
		if (timesig != "2" && timesig != "3" && timesig != "4" && timesig != "6")
			return error_result(tr(ctx.language, "timesignature 只能是 2 / 3 / 4 / 6",
			                       "timesignature must be 2 / 3 / 4 or 6"));

		const std::string abs = tool_resolve_path(ctx.cwd, output);
		if (std::string deny = tool_check_path_permission(ctx, abs); !deny.empty())
			return error_result(deny);

		MediaEngine& e = engine();
		if (std::string err = ensure_open(e, ctx.agent_dir, ctx.cwd, ctx.language); !err.empty())
			return error_result(err);

		MusicWorkflowParams wp;
		wp.tags = tags;
		wp.lyrics = lyrics;
		wp.bpm = bpm;
		wp.duration = duration;
		wp.seed = seed;
		wp.timesignature = timesig;
		wp.language = language;
		wp.keyscale = keyscale;
		// The audio-code LM is always on now, with the workflow defaults (cfg 7,
		// temperature 0.85, top-p 0.9): those knobs are gone from the tool.
		wp.generate_audio_codes = true;
		wp.lm_cfg_scale = 7.0;
		wp.lm_temperature = 0.85;
		wp.lm_top_p = 0.9;
		wp.output = abs;
		wp.model = e.spec.music_dit;
		wp.clip = e.spec.music_te;
		wp.clip_lm = e.spec.music_lm;
		wp.vae = e.spec.music_vae;
		wp.tokenizer_vocab = e.spec.music_tokenizer_vocab;
		wp.tokenizer_merges = e.spec.music_tokenizer_merges;
		wp.tokenizer_config = e.spec.music_tokenizer_config;
		// 采样步数来自工具设置（音乐生成模型 → 采样步数），默认 50 步（见 kMusicSteps）。
		wp.steps = media_steps(ctx.agent_dir, "music_steps", kMusicSteps);
		// 采样器 / 调度器（音乐生成模型 → 采样器 / 调度器）。
		wp.sampler = media_choice(ctx.agent_dir, "music_sampler");
		wp.scheduler = media_choice(ctx.agent_dir, "music_scheduler");

		const Graph graph = build_music_workflow(wp);
		GraphResult gr;
		const std::string engine_err = with_media_engine(
		    ctx.agent_dir, ctx.cwd, ctx.language,
		    [&](GpuCtx* gpu, const std::string& models_dir, std::shared_ptr<void>& user) -> std::string {
			    GraphContext gc;
			    gc.cwd = ctx.cwd;
			    gc.agent_dir = ctx.agent_dir;
			    gc.models_dir = models_dir;
			    gc.language = ctx.language;
			    gc.gpu = gpu;
			    gc.user = user;
			    gc.progress = [&](const std::string& s) {
				    if (ctx.on_update) ctx.on_update(s);
			    };
			    gc.cancelled = [&]() { return ctx.cancelled && ctx.cancelled(); };
			    GraphResult r = run_graph(graph, gc, media_registry());
			    user = gc.user;
			    gr = std::move(r);
			    return gr.ok ? std::string() : gr.error;
		    });
		if (!engine_err.empty())
			return error_result(tr(ctx.language, "生成失败：", "generation failed: ") + engine_err);
		if (ctx.cancelled && ctx.cancelled())
			return error_result(tr(ctx.language, "已取消", "cancelled"));

		// The decoded PCM, out of the VAE decode node ('MediaVaeDecode' exposes one
		// output port, `out`, whose payload is whatever role that VAE decodes to). The
		// port name here used to be "AUDIO", which no node has, so this loop never
		// found the buffer and the report silently fell back to its defaults - which is
		// how "0.0 s" reached a user. `as_audio` is the real test of "is this PCM".
		i64 sr = 48000, samples = 0, channels = 2;
		for (const GraphNode& nd : graph.nodes) {
			if (nd.type != "MediaVaeDecode") continue;
			Value v;
			if (!produced(gr, nd.id, "out", &v)) continue;
			auto a = as_audio(v);
			if (!a) continue;
			sr = a->sample_rate;
			channels = std::max<i64>(1, a->channels);
			samples = (i64)a->pcm.size() / channels;
		}
		// The DiT window the sampler planned and ran with, and (out of the conditioning,
		// where the encode node left it) the LM's own window. Reported because they are
		// the two numbers that say whether the card was used or whether every step /
		// every token re-read an 8-10 GB checkpoint (see `AceDiT::plan_residency` and
		// `AceLm::plan_residency`).
		i64 res_layers = 0;
		u64 res_bytes = 0;
		if (const GraphNode* s = node_by_type(graph, "MediaSampler")) {
			Value v;
			if (produced(gr, s->id, "LATENT", &v))
				if (auto lat = as_latent(v)) {
					res_layers = lat->resident_layers;
					res_bytes = lat->resident_bytes;
				}
		}
		i64 lm_layers = 0, lm_total = 0;
		u64 lm_bytes = 0;
		if (const GraphNode* enc = node_by_type(graph, "MediaAceTextEncode")) {
			Value v;
			if (produced(gr, enc->id, "CONDITIONING", &v))
				if (auto c = as_conditioning(v))
					if (c->ace_cond) {
						auto ac = std::static_pointer_cast<AceCondition>(c->ace_cond);
						lm_layers = ac->lm_resident_layers;
						lm_total = ac->lm_total_layers;
						lm_bytes = ac->lm_resident_bytes;
					}
		}
		std::ostringstream os;
		os << tr(ctx.language, "已生成音乐：", "generated music: ") << tool_display_path(ctx.cwd, abs)
		   << "\n";
		// Seconds, and the *produced* length rather than the requested one: the request
		// is snapped onto the 25 Hz music grid, so echoing it back would report a
		// number the file does not have. One decimal, like the TTS report.
		const double made_s = sr > 0 ? (double)samples / (double)sr : 0.0;
		char lenbuf[64];
		snprintf(lenbuf, sizeof lenbuf, "%.1f", made_s);
		os << tr(ctx.language, "时长 ", "length ") << lenbuf << tr(ctx.language, " 秒", " s");
		if (made_s > 0.0 && std::fabs(made_s - (double)duration) > 0.05)
			os << tr(ctx.language, "（请求 ", " (requested ") << (long long)duration
			   << tr(ctx.language, " 秒）", " s)");
		os << "\n";
		os << tr(ctx.language, "采样器 ", "sampler ")
		   << (wp.sampler.empty() ? std::string("euler") : wp.sampler) << " · "
		   << tr(ctx.language, "调度器 ", "scheduler ")
		   << (wp.scheduler.empty() ? std::string("simple") : wp.scheduler) << " · "
		   << tr(ctx.language, "步数 ", "steps ") << wp.steps << "\n";
		os << tr(ctx.language, "常驻层数 ", "resident layers ") << res_layers
		   << tr(ctx.language, " 层（", " (") << format_bytes(res_bytes)
		   << tr(ctx.language, "），其余逐层流式", "), the rest streamed per layer") << "\n";
		if (lm_total > 0)
			os << tr(ctx.language, "语义码 LM 常驻 ", "code LM resident ") << lm_layers << "/"
			   << lm_total << tr(ctx.language, " 层（", " (") << format_bytes(lm_bytes)
			   << tr(ctx.language, "）", ")") << "\n";
		if (sr > 0 && samples > 0)
			os << tr(ctx.language, "音频 ", "audio ") << sr << tr(ctx.language, " Hz, ", " Hz, ")
			   << channels << tr(ctx.language, " 声道", " channel(s)") << "\n";
		trace_dump(gr); // PHI_MEDIA_TRACE=1 prints the real per-node timings to stderr
		os << tr(ctx.language, "总用时 ", "total time ") << fmt_reported_time(trace_ms(gr)) << "\n";
		return text_result(os.str());
	};
	return t;
}

// ── tts_speak ──────────────────────────────────────────────────────────────

ToolDef tts_speak_tool() {
	ToolDef t;
	t.name = "tts_speak";
	t.description =
	    "本地调用 Breeze-TTS-2 做文本朗读。音频格式 wav 。";
	t.description_en =
	    "Read the text aloud locally with Breeze-TTS-2. The audio format is wav.";
	t.prompt_snippet = "语音合成";
	t.prompt_snippet_en = "Speak (TTS)";
	t.parameters_fn = [](const std::string& lang, const ToolContext&) {
		return schema_object(
		    {
		        {"text",
		         schema_string(lang,
		                       "要朗读出来的文本。中英双语，可用 [笑] / (sigh) 之类的语气事件。",
		                       "The text that will be read aloud. Chinese and English are supported, "
		                       "with inline tone events like [笑] / (sigh).")},
		        {"instruction",
		         schema_string(lang, "描述朗读的音色", "Describe the timbre to read it in")},
		        {"output", schema_string(lang, "输出 wav 的绝对路径", "Absolute path of the wav output")},
		    },
		    {"text", "instruction", "output"});
	};
	t.run = [](const JsonValue& args, ToolContext& ctx) -> ToolResult {
		// The two strings, named after what they do rather than after the slot they
		// came from: `spoken` is read aloud, `voice` only describes the timbre. The
		// prompt the model sees is `[S0]<ins_bos>{voice}<ins_eos>{spoken}` (see
		// `BreezeTts::build_prompt`), so they go down to the workflow under those two
		// names and are never exchanged.
		const std::string spoken = arg_string(args, "text");
		const std::string voice = arg_string(args, "instruction");
		const std::string output = arg_string(args, "output");
		// The seed is always drawn here: the tool no longer takes one.
		const uint64_t seed = make_media_seed();

		if (spoken.empty())
			return error_result(tr(ctx.language, "text 不能为空（要朗读的文本）",
			                       "text must not be empty (the text to read aloud)"));
		if (voice.empty())
			return error_result(tr(ctx.language, "instruction 不能为空（TTS 只接受音色描述，不支持参考音频）",
			                       "instruction must not be empty (this tool takes a voice "
			                       "description, not reference audio)"));
		if (output.empty())
			return error_result(tr(ctx.language, "output 不能为空", "output must not be empty"));
		const std::string abs = tool_resolve_path(ctx.cwd, output);
		if (std::string deny = tool_check_path_permission(ctx, abs); !deny.empty())
			return error_result(deny);

		MediaEngine& e = engine();
		if (std::string err = ensure_open(e, ctx.agent_dir, ctx.cwd, ctx.language); !err.empty())
			return error_result(err);

		TtsWorkflowParams wp;
		wp.text = spoken;
		wp.instruction = voice;
		// The released Voice-Design node's own generation controls (`_generation_controls`
		// in the reference nodes.py, and its example workflow): cfg 4 for voice design,
		// backbone 0.9 / top-k 50 / top-p 1.0 with a 1.1 repetition penalty, depth
		// decoder 0.9 / top-k 50 / top-p 1.0. They are spelled out here rather than left
		// to the struct's defaults so the tool's behaviour does not depend on whoever
		// edits those later - sampling either loop looser is what makes the utterance
		// stop early and the tail decode as noise.
		wp.cfg_scale = 4.0;
		wp.temperature = 0.9;
		wp.top_k = 50;
		wp.top_p = 1.0;
		wp.repetition_penalty = 1.1;
		wp.depth_temperature = 0.9;
		wp.depth_top_k = 50;
		wp.depth_top_p = 1.0;
		// No length is asked for, so the *text* sizes the budget (`duration_seconds`
		// 0 = let `BreezeTts::prepare` derive it from the text): the utterance is as
		// long as the sentence needs, not cut at some fixed duration.
		wp.seed = seed;
		wp.output = abs;
		wp.model = e.spec.tts_model;
		// The codec and the tokenizer are the user's selection (settings.json ->
		// tools.media). The two JSON configs are not: their scalars are compiled into
		// the loaders, so these stay empty and no file is read.
		wp.codec = e.spec.tts_codec;
		wp.tokenizer = e.spec.tts_tokenizer;
		wp.config.clear();
		wp.codec_config.clear();

		const Graph graph = build_tts_workflow(wp);
		GraphResult gr;
		const std::string engine_err = with_media_engine(
		    ctx.agent_dir, ctx.cwd, ctx.language,
		    [&](GpuCtx* gpu, const std::string& models_dir, std::shared_ptr<void>& user) -> std::string {
			    GraphContext gc;
			    gc.cwd = ctx.cwd;
			    gc.agent_dir = ctx.agent_dir;
			    gc.models_dir = models_dir;
			    gc.language = ctx.language;
			    gc.gpu = gpu;
			    gc.user = user;
			    gc.progress = [&](const std::string& s) {
				    if (ctx.on_update) ctx.on_update(s);
			    };
			    gc.cancelled = [&]() { return ctx.cancelled && ctx.cancelled(); };
			    GraphResult r = run_graph(graph, gc, media_registry());
			    user = gc.user;
			    gr = std::move(r);
			    return gr.ok ? std::string() : gr.error;
		    });
		if (!engine_err.empty())
			return error_result(tr(ctx.language, "生成失败：", "generation failed: ") + engine_err);
		if (ctx.cancelled && ctx.cancelled())
			return error_result(tr(ctx.language, "已取消", "cancelled"));

		i64 sr = 24000, samples = 0;
		i64 channels = 1;
		i64 frames = 0;
		for (const GraphNode& nd : graph.nodes) {
			if (nd.type == "MediaBreezeSampler") {
				Value v;
				if (produced(gr, nd.id, "LATENT", &v))
					if (auto l = as_latent(v)) frames = l->code_frames;
			}
			if (nd.type != "MediaBreezeDecode") continue;
			Value v;
			// `MediaBreezeDecode`'s output port is `AUDIO` (the VAE-decode node's is
			// `out`); the two names are different on purpose, and reading the wrong one
			// is silent - the report just falls back to its defaults.
			if (!produced(gr, nd.id, "AUDIO", &v)) continue;
			if (auto a = as_audio(v)) {
				sr = a->sample_rate;
				channels = std::max<i64>(1, a->channels);
				samples = (i64)a->pcm.size() / channels;
			}
		}
		// The frame count stays a *diagnostic* (it is what the loop counted, and the
		// codec's own rate is what turns it into a duration); the report is seconds.
		if (frames > 0 && getenv("PHI_TTS_DEBUG"))
			fprintf(stderr, "[tts] %lld code frames = %.2f s at %.1f fps\n", (long long)frames,
			        tts_seconds_for_frames(frames), kTtsFramesPerSecond);
		// Seconds everywhere: the produced length (with the precision a tenth of a
		// second gives, which is what "did it say the whole sentence" needs) and, when
		// the caller set one, the cap it was asked for. With no cap asked for the
		// budget came from the text, so there is nothing to report against it.
		const double produced_s = sr > 0 ? (double)samples / (double)sr : 0.0;
		const i64 frame_cap = tts_frames_for_seconds(wp.duration_seconds);
		std::ostringstream os;
		os << tr(ctx.language, "已生成语音：", "generated speech: ") << tool_display_path(ctx.cwd, abs)
		   << "\n";
		// Both strings, each named for its role: this is the only place a caller can
		// see which of the two was read aloud and which one only shaped the voice, so
		// a request that reached the tool the wrong way round is visible here instead
		// of in the audio.
		os << tr(ctx.language, "朗读 text：", "spoken text: ") << spoken << "\n";
		os << tr(ctx.language, "音色 instruction：", "voice instruction: ") << voice << "\n";
		char lenbuf[96];
		if (frame_cap > 0) {
			const std::string fmt = tr(ctx.language, "%.1f 秒（上限 %.1f 秒）",
			                           "%.1f s (capped at %.1f s)");
			snprintf(lenbuf, sizeof lenbuf, fmt.c_str(), produced_s,
			         tts_seconds_for_frames(frame_cap));
		} else {
			const std::string fmt = tr(ctx.language, "%.1f 秒", "%.1f s");
			snprintf(lenbuf, sizeof lenbuf, fmt.c_str(), produced_s);
		}
		os << tr(ctx.language, "时长 ", "length ") << lenbuf << "\n";
		if (sr > 0)
			os << tr(ctx.language, "音频 ", "audio ") << sr << tr(ctx.language, " Hz, 单声道（", " Hz, ") << channels << tr(ctx.language, " 声道）", " channel(s)")
			   << "\n";
		trace_dump(gr); // PHI_MEDIA_TRACE=1 prints the real per-node timings to stderr
		os << tr(ctx.language, "总用时 ", "total time ") << fmt_reported_time(trace_ms(gr)) << "\n";
		return text_result(os.str());
	};
	return t;
}

}  // namespace

std::vector<ToolDef> build_media_tools() {
	std::vector<ToolDef> tools;
	tools.push_back(image_generate_tool());
	tools.push_back(video_generate_tool());
	tools.push_back(music_generate_tool());
	tools.push_back(tts_speak_tool());
	// The call's own end, wrapped once for all four. `run` keeps the run's
	// GraphResult (and with it the MODEL / CLIP / VAE handles the graph produced)
	// alive until it returns, so this is the first point at which "the call is
	// over and the handles are gone" is true: what the classes still hold here is
	// what they would still hold a second later, in a process that went on to do
	// something else. See report_media_state for why this is worth printing.
	for (ToolDef& t : tools) {
		auto inner = t.run;
		t.run = [inner](const JsonValue& args, ToolContext& ctx) -> ToolResult {
			ToolResult r = inner(args, ctx);
			report_media_state(engine(), "after");
			return r;
		};
	}
	return tools;
}

// ── the shared engine ───────────────────────────────────────────────────────
//
// All four generators run their node graph through the same process-wide engine
// and its one-GPU-at-a-time lock. This function is the seam: it brings the engine
// up, hands `fn` the device, the models directory and the run's state bag, and runs
// `fn` under the engine's lock (a node graph is a generation and must not race
// another). It is also the boundary at which the run's caches are handed back on
// both sides: before `fn` runs, so the residency plans see the card this process
// really has, and after it returns, so nothing outlives the generation.
std::string with_media_engine(
	const std::string& agent_dir, const std::string& cwd, const std::string& language,
	const std::function<std::string(media::GpuCtx*, const std::string&, std::shared_ptr<void>&)>& fn) {
	MediaEngine& e = engine();
	if (std::string err = ensure_open(e, agent_dir, cwd, language); !err.empty()) return err;
	if (!e.gpu) return tr(language, "媒体引擎不可用", "media engine is unavailable");

	if (!e.mu.try_lock())
		return tr(language, "已有另一个媒体任务在运行，请稍后再试",
		          "another media job is already running, try again later");
	std::lock_guard<std::mutex> lock(e.mu, std::adopt_lock);

	refresh_engine_run(e);
	release_media_memory(e);
	report_media_state(e, "before");

	std::string result;
	try {
		result = fn(e.gpu.get(), e.models_dir, e.graph_user);
	} catch (const std::exception& ex) {
		if (e.cuda && e.cuda->device_removed()) shutdown_locked(e);
		result = tr(language, "工作流失败：", "workflow failed: ") + ex.what();
	}
	// A lost device is not a property of this request: tear the engine down so the
	// next call starts from a fresh device instead of failing forever. The node
	// executor reports a node failure as a value (not an exception), so this is the
	// other half of the check the pipelines used to do around generate().
	if (e.cuda && e.cuda->device_removed()) shutdown_locked(e);
	// The request is over: hand back the VRAM, the compiled kernels and the
	// host-side caches it built, so a generation leaves nothing behind in any of
	// the three memories it touched (see release_generation_state).
	release_generation_state(e);
	return result;
}

}  // namespace phi
