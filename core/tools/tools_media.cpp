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
// shared arenas/ring, the one-generation-at-a-time lock, and the persistent node
// cache (so the checkpoints are opened once per selection, not once per call).
// The engine is lazily created: the image checkpoint is 6.8 GB of int8 and its
// text encoder another 8.7 GB, so opening it per call would dominate the runtime
// and opening it at startup would make every session pay for a tool that may
// never be used.
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
#include "models/media_models.hpp"
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
	// The node world's persistent state: loaded MODEL/CLIP/VAE handles keyed on
	// (arch, path, loras), so a run reuses the weights the previous run loaded.
	// Cleared with the rest of the engine on a lost device / spec change, because
	// a cached handle would otherwise point into a torn-down context.
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

// Drops everything the engine holds so the next call rebuilds it from scratch.
// Only legal while `e.mu` is held and no generation is in flight.
void shutdown_locked(MediaEngine& e) {
	e.gpu.reset();
	e.graph_user.reset();
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
		MediaModelSpec spec = resolve_media_spec(read_media_settings(agent_dir), e.models_dir);
		const bool changed = e.spec_ready &&
		                     (spec.image_dit != e.spec.image_dit || spec.image_te != e.spec.image_te ||
		                      spec.image_vae != e.spec.image_vae || spec.video_dit != e.spec.video_dit ||
		                      spec.video_te != e.spec.video_te || spec.video_vae != e.spec.video_vae ||
		                      spec.video_avae != e.spec.video_avae ||
		                      spec.music_dit != e.spec.music_dit || spec.music_te != e.spec.music_te ||
		                      spec.music_lm != e.spec.music_lm || spec.music_vae != e.spec.music_vae ||
		                      spec.tts_model != e.spec.tts_model || spec.tts_codec != e.spec.tts_codec ||
		                      spec.tts_tokenizer != e.spec.tts_tokenizer ||
		                      spec.tts_config != e.spec.tts_config ||
		                      spec.tts_codec_config != e.spec.tts_codec_config ||
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
		if (changed && e.gpu) release_media_cache(*std::static_pointer_cast<GraphModelCache>(e.graph_user));
		if (changed) e.graph_user.reset();
		e.spec = std::move(spec);
		e.spec_ready = true;
	}

	try {
		e.cuda = std::make_unique<CudaContext>();
		e.cuda->create(-1);
		// Ask the driver what it can hand this process *before* anything sizes itself
		// from the answer. `note_os_budget` is what installs the accountant's limit,
		// and the limit is read by the ring sizing below and by both residency
		// planners (`ImageDiT::plan_residency`, `AvDiT::plan_residency` -
		// which stream all their blocks when it is 0 instead of deciding residency).
		e.cuda->query_budget();
		// The shader cache lives under the data dir so a read-only install
		// still works (ComputeContext falls back to compiling in memory).
		std::string cache = path_normalize(agent_dir + "/media_shader_cache");
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
// So this hands every byte both chains are holding back to the device: the node
// cache's VAE weights and DiT windows (`release_media_cache`) and the shared
// streaming/activation arenas. It is called twice per tool call: before anything
// is planned, so the plan sees the card this invocation really has (which is also
// what cleans up after a run that threw), and after the file is written, so an
// idle process - and the desktop compositor sharing the card - is not left with a
// parked generation. Nothing is lost by it: each chain re-reads what it needs on
// its next call (the DiT streams its weights anyway; the VAE weights come straight
// out of the page cache).
void release_media_memory(MediaEngine& e) {
	if (!e.gpu) return;
	if (e.graph_user) release_media_cache(*std::static_pointer_cast<GraphModelCache>(e.graph_user));
	if (e.wa) e.wa->release_chunks();
	if (e.aa) e.aa->release_chunks();
}

std::string fmt_ms(double ms) {
	char buf[32];
	snprintf(buf, sizeof buf, "%.1fs", ms / 1000.0);
	return buf;
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

double trace_ms(const GraphResult& r) {
	double t = 0;
	for (const NodeTrace& n : r.trace) t += n.ms;
	return t;
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
	    "本地调用 qwen-image 2.1 生成 png 格式的图片（文生图或参考图编辑）。输出尺寸只由 width/height 决定，参考图不参与决定画布；参考图只按 ref_resolution 独立缩放，参考图多/大导致放不下时把它调小即可。";
	t.description_en =
	    "Call qwen-image 2.1 locally to generate PNG images (text-to-image or "
	    "reference-conditioned editing). The output size is width x height alone - references "
	    "never decide the canvas; they are only sized by ref_resolution, so lowering that is what "
	    "makes a multi-reference edit fit a smaller card.";
	t.prompt_snippet = "生成图片";
	t.prompt_snippet_en = "Generate an image";
	t.parameters_fn = [](const std::string& lang) {
		return schema_object(
		    {
		        {"prompt",
		         schema_string(lang,
		                       "提示词使用英文写 100-200 单词。描述图片内容和画面风格，不写分辨率和长宽比。",
		                       "Write the prompt in English, 100-200 words. Describe the image content and art style, do not mention resolution or aspect ratio.")},
		        {"width", schema_number(lang, "输出画面宽度，范围 400-2000 像素（必须输入）。输出尺寸只由 width/height 决定，参考图不参与决定画布。", "Output image width, range 400-2000 pixels (required). The output size comes from width/height alone: references never decide the canvas.")},
		        {"height", schema_number(lang, "输出画面高度，范围 400-2000 像素（必须输入）。输出尺寸只由 width/height 决定，参考图不参与决定画布。", "Output image height, range 400-2000 pixels (required). The output size comes from width/height alone: references never decide the canvas.")},
		        {"negative_prompt",
		         schema_string(lang,
		                       "负面提示词（可选）。描述画面中不希望出现的内容；工作流的 KSampler cfg=1，因此仅在 cfg>1 时参与引导。",
		                       "Negative prompt (optional). Content the picture should avoid; the workflow's KSampler runs cfg=1, so it only steers the result when cfg>1.")},
		        {"cfg",
		         schema_number(lang,
		                       "分类器无关引导强度（可选，默认 1 = 工作流设置）。>1 时同时运行正/负两路条件（每步耗时约翻倍）。",
		                       "Classifier-free guidance scale (optional, default 1 = the workflow's setting). Above 1 the positive and negative conditionings are both evaluated (about twice the per-step cost).")},
		        {"ref_images",
		         schema_array(lang,
		                    schema_string(lang, "参考图的绝对路径", "Absolute path of a reference image"),
		                    "参考图的绝对路径，按照参考顺序排列（最多 9 张）。给定时为参考生图，否则为文生图。",
		                    "Absolute paths of the reference images, in order (up to 9). With references this is a reference-conditioned edit, without them plain text-to-image.")},
		        {"ref_resolution",
		         schema_number(lang,
		                       "每张参考图缩放到的边长（可选，默认 1024）：参考图会等比缩放到约 ref_resolution² 像素、32 对齐。它只影响参考图，不影响输出尺寸（输出只看 width/height）。参考图太多/太大导致放不下时，把它改小（例如 512）就能让工程跑起来；0 = 保持每张参考图的原始尺寸（32 对齐）。",
		                       "Edge every reference image is resized to (optional, default 1024): each reference is scaled to about ref_resolution^2 px, 32-aligned. This sizes the references only, never the output (that is width/height). If the references make the request too large, lower it (e.g. 512). 0 keeps each reference's own size, rounded to 32.")},
		        {"prefix_cache",
		         schema_boolean(lang,
		                     "是否启用前缀 KV 缓存（可选，默认 false）。开启后提示词/参考图的 K/V 只算一次，"
		                     "之后每步只跑目标行（结果与不启用逐位相同），代价是常驻显存：显存不足时自动退回。"
		                     "显存紧张时保持关闭通常更快，因为少了常驻的权重窗口就要每步重读权重。",
		                     "Enable the prefix K/V cache (optional, default false). The prompt's and "
		                     "the references' K/V are computed once and every later step runs only the "
		                     "target rows (bit-identical to leaving it off); it costs resident VRAM and "
		                     "falls back automatically when there is no room. On a small card leaving it "
		                     "off is usually faster, because one resident weight block fewer means "
		                     "re-reading a block on every step.")},
		        {"output",
		         schema_string(lang, "输出 PNG 的绝对路径", "Absolute path of the PNG output")},
		        {"seed",
		         schema_number(lang,
		                       "随机种子（可选）：一个 64 位无符号整数（0 – 18446744073709551615），填什么就用什么（包括 0）；不填才由引擎随机抽一个（抽到的值会在结果里报出）。相同的种子得到相同的图。",
		                       "Random seed (optional): an unsigned 64-bit integer (0 - 18446744073709551615), used exactly as given (0 included); only when it is left out does the engine draw one, and the value it drew is reported back. The same seed gives the same picture.")},
		    },
		    {"prompt", "width", "height", "output"});
	};
	t.run = [](const JsonValue& args, ToolContext& ctx) -> ToolResult {
		const std::string prompt = arg_string(args, "prompt");
		const std::string negative_prompt = arg_string(args, "negative_prompt");
		const int64_t w = arg_int(args, "width", 0);
		const int64_t h = arg_int(args, "height", 0);
		const std::string output = arg_string(args, "output");
		// The seed: what the call spells out is what runs — 0 included, the whole
		// point of a seed being that the same one gives the same picture — and only a
		// call that leaves the argument out gets a fresh one. Drawing it here, rather
		// than passing 0 along and letting the graph draw, is what keeps "omitted"
		// and "seed 0" apart; it also means the number is in hand for the report even
		// if the run never reaches a sampler.
		uint64_t seed = 0;
		if (!arg_u64_present(args, "seed", &seed)) seed = make_media_seed();
		const double cfg_arg = arg_number(args, "cfg", 1.0);
		const int64_t ref_resolution = arg_int(args, "ref_resolution", 1024);
		const bool prefix_cache = args["prefix_cache"].as_bool(false);

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
		wp.negative_prompt = negative_prompt;
		wp.width = w;
		wp.height = h;
		wp.seed = seed;
		wp.cfg = (float)std::clamp(cfg_arg, 0.0, 20.0);
		wp.ref_images = ref_images;
		// ComfyUI's `resolution` widget, verbatim: the reference sizing edge. It is
		// what keeps an edit's cost a property of the *references* rather than of
		// the output size, and 0 keeps each reference at its own size (rounded to
		// 32), which is the node's other legal setting.
		wp.ref_resolution = std::clamp<int64_t>(ref_resolution, 0, 4096);
		wp.prefix_cache = prefix_cache;
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

		// The report: where it landed, the seed to reproduce it with, the resolution
		// written and how long the run took.
		u64 out_seed = seed;
		i64 out_w = w, out_h = h;
		bool cache_used = false;
		i64 cache_rows = 0;
		u64 cache_bytes = 0;
		u64 cache_wanted = 0;
		if (const GraphNode* s = node_by_type(graph, "MediaSampler")) {
			Value v;
			if (produced(gr, s->id, "LATENT", &v)) {
				if (auto lat = as_latent(v)) {
					out_seed = lat->seed;
					if (lat->req_w > 0) out_w = lat->req_w;
					if (lat->req_h > 0) out_h = lat->req_h;
					cache_used = lat->prefix_cache;
					cache_rows = lat->prefix_cache_rows;
					cache_bytes = lat->prefix_cache_bytes;
					cache_wanted = lat->prefix_cache_wanted;
				}
			}
		}
		std::ostringstream os;
		os << tr(ctx.language, "已生成图片：", "generated image: ") << tool_display_path(ctx.cwd, abs)
		   << "\n";
		os << tr(ctx.language, "种子 ", "seed ") << (unsigned long long)out_seed << "\n";
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
		if (prefix_cache) {
			os << tr(ctx.language, "前缀缓存 ", "prefix cache ");
			if (cache_used) {
				os << tr(ctx.language, "已启用 ", "on ") << cache_rows
				   << tr(ctx.language, " 行 / ", " rows / ") << format_bytes(cache_bytes);
			} else {
				os << tr(ctx.language, "未启用（显存不足，需要 ", "off (no room; it needs ")
				   << format_bytes(cache_wanted)
				   << tr(ctx.language, "，已退回每步重算）",
				        ", so the prefix is recomputed each step)");
			}
			os << "\n";
		}
		os << tr(ctx.language, "总用时 ", "total time ") << fmt_ms(trace_ms(gr)) << "\n";
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
	    "本地调用 minimax h3 ref2va 生成 mp4 格式的视频（可带参考图 / 参考视频 / 参考音频）";
	t.description_en =
	    "Call minimax h3 ref2va locally to generate mp4 videos.";
	t.prompt_snippet = "生成视频";
	t.prompt_snippet_en = "Generate a video";
	t.parameters_fn = [](const std::string& lang) {
		return schema_object(
		    {
		        {"prompt",
		         schema_string(lang,
		                       "提示词使用英文写 200-500 单词，不写分辨率和长宽比。使用 <Subject N> 可以定义多次出现的可见内容，引用参考图片/音频使用标签 <Picture N> <Audio N> 。按顺序逐镜头写，[Shot 1] 无时间戳，后续镜头写 [Shot N] At MM:SS.mmm；分别描述每个镜头的运镜方式、镜头的画面、环境音效和物理音效。对话的主体发言写 <Subject N> (Sx) ，按照说话实际顺序标出 (S1)(S2)... ；说话/歌词使用 <d>[ 语言 ] 说话内容 </d> ；画外音要标出 off-screen。观众听到的背景音乐写无（后期剪辑再加）。",
		                       "Write the prompt in English, 200-500 words, do not mention resolution or aspect ratio. Use <Subject N> to define visible content that appears multiple times, and reference pictures/audio with <Picture N> / <Audio N> tags. Write shot by shot in order: [Shot 1] has no timestamp, later shots use [Shot N] At MM:SS.mmm; describe each shot's camera movement, its visuals, ambient sound and physical sound effects. For dialogue, write the speaker as <Subject N> (Sx) and mark all lines in the actual speaking order (S1)(S2)...; spoken lines/lyrics use <d>[ language ] spoken text </d>; mark off-screen voices as off-screen. For background music the audience hears, write none (added later in editing).}")},
		        {"duration_frames",
		         schema_number(lang,
		                       "视频总帧数（24帧 相当于 1秒 ，最大 362。会自动吸附到 17k+5）",
		                       "Total video frames (24 frames equals 1 second, max 362. Will auto-snap to 17k+5)")},
		        {"output",
		         schema_string(lang, "输出视频的绝对路径", "Absolute path of the video output")},
		        {"ref_images",
		         schema_array(lang, schema_string(lang, "参考图的绝对路径，与 <Picture N> 的顺序对应（图片数量 0-9 张）", "Absolute path of the reference images, in <Picture N> order (0-9 images)"),
		                      "参考图的绝对路径，与 <Picture N> 的顺序对应（图片数量 0-9 张）",
		                      "Absolute paths of the reference images, in <Picture N> order (0-9 images)")},
		        {"ref_videos",
		         schema_array(lang, schema_string(lang, "参考视频的绝对路径，与 <Video N> 的顺序对应（数量 0-3 段）。视频按 2 fps 采样、逐对帧写 <T.T seconds> 时间戳；提示词里用 <Video N> 引用。",
		                                          "Absolute path of a reference video, in <Video N> order (0-3 clips). The clip is sampled at 2 fps with a <T.T seconds> timestamp per frame pair; refer to it as <Video N>."),
		                      "参考视频的绝对路径，与 <Video N> 的顺序对应（数量 0-3 段）",
		                      "Absolute paths of the reference videos, in <Video N> order (0-3 clips)")},
		        {"ref_video_audios",
		         schema_array(lang, schema_string(lang, "参考视频自带的声音轨：第 N 个对应第 N 个参考视频（不需要的就写空字符串）",
		                                          "Soundtrack of the same-numbered reference video (entry N belongs to ref_videos[N]; use an empty string to skip)"),
		                      "参考视频的声音轨，按序号与 ref_videos 一一对应（不需要的留空字符串）",
		                      "Soundtracks of the reference videos, index-paired with ref_videos (empty string = none)")},
		        {"ref_audios",
		         schema_array(lang, schema_string(lang, "参考音频的决对路径，与 <Audio N> 的顺序对应（音频数量 0-3 段）", "Absolute path of the reference audio, in <Audio N> order (0-3 clips)"),
		                      "参考音频的决对路径，与 <Audio N> 的顺序对应（音频数量 0-3 段）",
		                      "Absolute paths of the reference audio, in <Audio N> order (0-3 clips)")},
		        {"ref_image_size",
		         schema_string(lang, "参考图的缩放方式：match（默认，按生成画面的像素面积等比缩小）或 max（缩到 2048 短边：人物一致性更好，但明显更慢）。",
		                       "Reference image sizing: match (default: down-only scale to the generation's pixel area) or max (2048 px short edge: better identity fidelity, several times slower).")},
		        {"width", schema_number(lang, "画面宽度，默认 960 像素（除非特别要求，否则不用填）",
		                        "Image width, default 960 pixels (leave blank unless specifically requested)")},
		        {"height", schema_number(lang, "画面高度，默认 540 像素（除非特别要求，否则不用填）",
		                                 "Image height, default 540 pixels (leave blank unless specifically requested)")},
		        {"seed",
		         schema_number(lang,
		                       "随机种子（可选）：一个 64 位无符号整数（0 – 18446744073709551615），填什么就用什么（包括 0）；不填才由引擎随机抽一个（抽到的值会在结果里报出）。相同的种子得到相同的视频。",
		                       "Random seed (optional): an unsigned 64-bit integer (0 - 18446744073709551615), used exactly as given (0 included); only when it is left out does the engine draw one, and the value it drew is reported back. The same seed gives the same video.")},
		    },
		    {"prompt", "duration_frames", "output"});
	};
	t.run = [](const JsonValue& args, ToolContext& ctx) -> ToolResult {
		const std::string prompt = arg_string(args, "prompt");
		const int64_t frames = arg_int(args, "duration_frames", 0);
		const std::string output = arg_string(args, "output");
		const int64_t w = arg_int(args, "width", 960);
		const int64_t h = arg_int(args, "height", 540);
		// As in image_generate: a seed that was given is used verbatim, and one that
		// was not is drawn here (see the comment there).
		uint64_t seed = 0;
		if (!arg_u64_present(args, "seed", &seed)) seed = make_media_seed();

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

		std::vector<std::string> ref_images, ref_videos, ref_video_audios, ref_audios;
		for (const JsonValue& v : args["ref_images"].items()) {
			std::string s = v.as_string();
			if (!s.empty()) ref_images.push_back(tool_resolve_path(ctx.cwd, s));
		}
		for (const JsonValue& v : args["ref_videos"].items()) {
			std::string s = v.as_string();
			if (!s.empty()) ref_videos.push_back(tool_resolve_path(ctx.cwd, s));
		}
		// An empty soundtrack entry is the "this video has none" spelling, so the
		// list stays index-aligned with ref_videos instead of being compacted.
		for (const JsonValue& v : args["ref_video_audios"].items()) {
			std::string s = v.as_string();
			ref_video_audios.push_back(s.empty() ? std::string() : tool_resolve_path(ctx.cwd, s));
		}
		for (const JsonValue& v : args["ref_audios"].items()) {
			std::string s = v.as_string();
			if (!s.empty()) ref_audios.push_back(tool_resolve_path(ctx.cwd, s));
		}
		if ((int64_t)ref_images.size() > kMaxRefImages)
			return error_result(tr(ctx.language, "ref_images 最多 9 张", "at most 9 ref_images"));
		if ((int64_t)ref_videos.size() > kMaxRefVideos)
			return error_result(tr(ctx.language, "ref_videos 最多 3 段", "at most 3 ref_videos"));
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
		wp.ref_videos = ref_videos;
		wp.ref_video_audios = ref_video_audios;
		{
			const std::string size = arg_string(args, "ref_image_size");
			if (!size.empty() && size != "match" && size != "max")
				return error_result(tr(ctx.language, "ref_image_size 只能是 match 或 max",
				                       "ref_image_size must be match or max"));
			wp.ref_image_size = size.empty() ? "match" : size;
		}
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

		u64 out_seed = seed;
		i64 out_frames = 0, out_w = w, out_h = h;
		if (const GraphNode* s = node_by_type(graph, "MediaSampler")) {
			Value v;
			if (produced(gr, s->id, "LATENT", &v)) {
				if (auto lat = as_latent(v)) {
					out_seed = lat->seed;
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
		os << tr(ctx.language, "种子 ", "seed ") << (unsigned long long)out_seed << "\n";
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
		os << tr(ctx.language, "总用时 ", "total time ") << fmt_ms(trace_ms(gr)) << "\n";
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
	    "本地调用 ACE-Step 1.5 生成音乐（wav）。输入风格标签 tags、歌词 lyrics 和音乐元信息"
	    "（bpm / 时长 / 拍号 / 调式 / 语言）；时长决定生成长度，纯器乐写 lyrics=\"[instrumental]\"。";
	t.description_en =
	    "Call ACE-Step 1.5 locally to generate music (wav). Give style tags, the lyrics, and the "
	    "music metadata (bpm / duration / time signature / key / language); the duration decides "
	    "how much is generated, and instrumental music uses lyrics=\"[instrumental]\".";
	t.prompt_snippet = "生成音乐";
	t.prompt_snippet_en = "Generate music";
	t.parameters_fn = [](const std::string& lang) {
		return schema_object(
		    {
		        {"tags",
		         schema_string(lang,
		                       "风格标签：乐器、情绪、流派、制作感，用英文逗号分隔（例如 \"pop, female vocal, "
		                       "warm, acoustic guitar, 90s\"）。不要在这里写歌词。",
		                       "Style tags: instruments, mood, genre, production, comma separated (e.g. "
		                       "\"pop, female vocal, warm, acoustic guitar, 90s\"). Do not put the lyrics here.")},
		        {"lyrics",
		         schema_string(lang,
		                       "歌词，用 [verse] / [chorus] 之类的段落标记分行；纯器乐写 [instrumental]。",
		                       "The lyrics, with [verse] / [chorus] section markers on their own lines; "
		                       "write [instrumental] for instrumental music.")},
		        {"duration",
		         schema_number(lang, "音乐时长（秒），1–600，默认 60。它同时决定音频语义码数量和潜变量长度。",
		                       "Music duration in seconds, 1-600, default 60. It decides both the number "
		                       "of audio semantic codes and the latent length.")},
		        {"bpm", schema_number(lang, "速度（BPM），默认 72。", "Tempo in BPM, default 72.")},
		        {"language", schema_string(lang, "歌词语言代码（en / zh / ja ...），默认 en。",
		                                     "Lyrics language code (en / zh / ja ...), default en.")},
		        {"keyscale", schema_string(lang, "调式，例如 \"D minor\"、\"C major\"。",
		                                     "Key and scale, e.g. \"D minor\", \"C major\".")},
		        {"timesignature",
		         schema_string(lang, "拍号：2 / 3 / 4 / 6，默认 4。",
		                       "Time signature: 2 / 3 / 4 / 6, default 4.")},
		        {"generate_audio_codes",
		         schema_boolean(lang,
		                        "是否用 Qwen3-4B 生成音频语义码（默认 true）。关闭更快、质量更低，且需要参考音频才能得到合理结果；文生音乐应保持开启。",
		                        "Run the Qwen3-4B audio-code LM (default true). Turning it off is faster "
		                        "and lower quality, and only makes sense with a reference clip; "
		                        "text-to-music should leave it on.")},
		        {"cfg_scale",
		         schema_number(lang,
		                       "音频语义码 LM 的引导强度（默认 7）。它只影响歌词/标签到语义码的采样，不影响扩散采样（扩散链固定 cfg=1，与工作流一致）。",
		                       "The audio-code LM's guidance scale (default 7). It steers only the "
		                       "tags/lyrics -> code sampling, not the diffusion loop, which runs cfg=1 "
		                       "exactly as the released workflow does.")},
		        {"temperature", schema_number(lang, "语义码 LM 采样温度（默认 0.85）。",
		                                       "The code LM's sampling temperature (default 0.85).")},
		        {"top_p", schema_number(lang, "语义码 LM 的 top-p（默认 0.9）。",
		                                 "The code LM's top-p (default 0.9).")},
		        {"output", schema_string(lang, "输出 WAV 的绝对路径",
		                                  "Absolute path of the WAV output")},
		        {"seed",
		         schema_number(lang,
		                       "随机种子（可选）：64 位无符号整数，填什么就用什么（含 0）；不填则随机抽一个并在结果里报出。相同种子得到相同音乐。",
		                       "Random seed (optional): a 64-bit unsigned integer, used exactly as given "
		                       "(0 included); omitted means the engine draws one and reports it. The same "
		                       "seed gives the same music.")},
		    },
		    {"tags", "output"});
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
		const bool gen_codes = args["generate_audio_codes"].as_bool(true);
		uint64_t seed = 0;
		if (!arg_u64_present(args, "seed", &seed)) seed = make_media_seed();

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
		wp.generate_audio_codes = gen_codes;
		wp.lm_cfg_scale = arg_number(args, "cfg_scale", 7.0);
		wp.lm_temperature = arg_number(args, "temperature", 0.85);
		wp.lm_top_p = arg_number(args, "top_p", 0.9);
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
		os << tr(ctx.language, "种子 ", "seed ") << (unsigned long long)seed << "\n";
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
		os << tr(ctx.language, "总用时 ", "total time ") << fmt_ms(trace_ms(gr)) << "\n";
		return text_result(os.str());
	};
	return t;
}

// ── tts_speak ──────────────────────────────────────────────────────────────

ToolDef tts_speak_tool() {
	ToolDef t;
	t.name = "tts_speak";
	t.description =
	    "本地调用 Breeze-TTS-2 做语音合成（wav）：输入「要说的文本」和「描述音色的指令」即可，"
	    "不需要任何参考音频（不支持声音克隆）。指令与文本语言保持一致。";
	t.description_en =
	    "Call Breeze-TTS-2 locally for speech synthesis (wav): give the text to speak and an "
	    "instruction describing the voice, with no reference audio of any kind (voice cloning is "
	    "deliberately not supported). The instruction's language should match the text's.";
	t.prompt_snippet = "语音合成";
	t.prompt_snippet_en = "Speak (TTS)";
	t.parameters_fn = [](const std::string& lang) {
		return schema_object(
		    {
		        {"text", schema_string(lang, "要合成的文本（必填）。中英双语，可用 [笑] / (sigh) 之类的语气事件。",
		                                "The text to synthesise (required). Chinese and English are "
		                                "supported, with inline tone events like [笑] / (sigh).")},
		        {"instruction",
		         schema_string(lang,
		                       "描述音色的自然语言指令（必填），例如「一个温柔的女声，语速偏慢，气息轻柔」"
		                       "或 \"a warm middle-aged male voice, calm and slightly hoarse\"。语言应与 text 一致。",
		                       "A natural-language instruction describing the voice (required), e.g. "
		                       "\"a warm middle-aged male voice, calm and slightly hoarse\". Its "
		                       "language should match the text's.")},
		        {"cfg_scale",
		         schema_number(lang,
		                       "引导强度（默认 4，音色设计节点的默认值）：越大越贴近指令，过大会发闷。",
		                       "Guidance scale (default 4, the voice-design node's default): higher "
		                       "follows the instruction more closely, too high sounds strained.")},
		        {"temperature", schema_number(lang, "采样温度（默认 0.7）。", "Sampling temperature (default 0.7).")},
		        {"repetition_penalty", schema_number(lang, "重复惩罚（默认 1 = 关闭）。",
		                                                 "Repetition penalty (default 1 = off).")},
	        {"duration_seconds",
	         schema_number(lang,
	                       "生成语音的长度上限（秒），默认 0 = 由文本长度决定，上限 163 秒。"
	                       "语音靠模型自己的 EOS 结束，实际长度通常短于这个上限（结果里会报出实际长度）。",
	                       "Upper bound on the spoken length in seconds; 0 (the default) lets the "
	                       "text length decide, and the ceiling is 163 s. The model stops on its own "
	                       "EOS token, so the audio is usually shorter than this cap - the result "
	                       "reports the length actually produced.")},
		        {"output", schema_string(lang, "输出 WAV 的绝对路径", "Absolute path of the WAV output")},
		        {"seed",
		         schema_number(lang,
		                       "随机种子（可选）：64 位无符号整数，填什么就用什么（含 0）；不填则随机抽一个并在结果里报出。",
		                       "Random seed (optional): a 64-bit unsigned integer, used exactly as "
		                       "given (0 included); omitted means the engine draws one and reports it.")},
		    },
		    {"text", "instruction", "output"});
	};
	t.run = [](const JsonValue& args, ToolContext& ctx) -> ToolResult {
		const std::string text = arg_string(args, "text");
		const std::string instruction = arg_string(args, "instruction");
		const std::string output = arg_string(args, "output");
		// Length is requested in *seconds* (the unit a caller knows). A caller written
		// against the older schema passed `max_new_tokens`, i.e. the sampler's own
		// frame count; that spelling is still accepted and converted, so nothing
		// silently changes meaning.
		const double want_seconds = arg_number(args, "duration_seconds", 0.0);
		const double legacy_frames = arg_number(args, "max_new_tokens", 0.0);
		double duration_seconds = want_seconds > 0.0 ? want_seconds : tts_seconds_for_frames((i64)legacy_frames);
		uint64_t seed = 0;
		if (!arg_u64_present(args, "seed", &seed)) seed = make_media_seed();

		if (text.empty())
			return error_result(tr(ctx.language, "text 不能为空", "text must not be empty"));
		if (instruction.empty())
			return error_result(tr(ctx.language, "instruction 不能为空（TTS 只接受音色描述，不支持参考音频）",
			                       "instruction must not be empty (this tool takes a voice "
			                       "description, not reference audio)"));
		if (output.empty())
			return error_result(tr(ctx.language, "output 不能为空", "output must not be empty"));
		if (duration_seconds < 0.0 || duration_seconds > 163.0)
			return error_result(tr(ctx.language, "duration_seconds 必须在 0–163 秒之间",
			                       "duration_seconds must be in 0-163"));

		const std::string abs = tool_resolve_path(ctx.cwd, output);
		if (std::string deny = tool_check_path_permission(ctx, abs); !deny.empty())
			return error_result(deny);

		MediaEngine& e = engine();
		if (std::string err = ensure_open(e, ctx.agent_dir, ctx.cwd, ctx.language); !err.empty())
			return error_result(err);

		TtsWorkflowParams wp;
		wp.text = text;
		wp.instruction = instruction;
		wp.cfg_scale = arg_number(args, "cfg_scale", 4.0);
		// The request is in seconds; `build_tts_workflow` turns it into the sampler's
		// frame budget with the codec's own rate, so the two units are one number here.
		wp.duration_seconds = duration_seconds;
		wp.temperature = arg_number(args, "temperature", 0.7);
		wp.repetition_penalty = arg_number(args, "repetition_penalty", 1.0);
		wp.seed = seed;
		wp.output = abs;
		wp.model = e.spec.tts_model;
		// The codec, the tokenizer and the two JSON configs are the user's
		// selection (settings.json -> tools.media); empty keeps the loaders'
		// sibling `config.json` lookup.
		wp.codec = e.spec.tts_codec;
		wp.tokenizer = e.spec.tts_tokenizer;
		wp.config = e.spec.tts_config;
		wp.codec_config = e.spec.tts_codec_config;

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
		// the caller set one, the cap it was asked for.
		const double produced_s = sr > 0 ? (double)samples / (double)sr : 0.0;
		const i64 frame_cap =
		    wp.max_new_tokens > 0 ? wp.max_new_tokens : tts_frames_for_seconds(wp.duration_seconds);
		std::ostringstream os;
		os << tr(ctx.language, "已生成语音：", "generated speech: ") << tool_display_path(ctx.cwd, abs)
		   << "\n";
		os << tr(ctx.language, "种子 ", "seed ") << (unsigned long long)seed << "\n";
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
		os << tr(ctx.language, "总用时 ", "total time ") << fmt_ms(trace_ms(gr)) << "\n";
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
	return tools;
}

// ── the shared engine ───────────────────────────────────────────────────────
//
// Both generators run their node graph through the same process-wide engine and
// its one-GPU-at-a-time lock. This function is the seam: it brings the engine
// up, hands `fn` the device, the models directory and the persistent node cache,
// and runs `fn` under the engine's lock (a node graph is a generation and must
// not race another).
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
	release_media_memory(e);
	return result;
}

}  // namespace phi
