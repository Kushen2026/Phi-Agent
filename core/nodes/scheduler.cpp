// Node "MediaScheduler" — Scheduler (sigmas).
//
// One node, one file. Registered here; the shared helpers live in service.hpp,
// the catalogue entry point in registry.cpp.
//
// ComfyUI has one `BasicScheduler` node and a scheduler name widget; the model on
// the socket carries its own `model_sampling`, so the node never learns which
// network it is building a grid for. This engine keeps that shape: the `arch`
// decides which `SamplingFlow` the grid is built from, and `sampling/schedulers.*`
// is the one port of ComfyUI's scheduler table. A name the engine does not port
// is refused by name (the caller keeps its own default) rather than silently
// replaced.
#include "nodes/service.hpp"

#include "sampling/schedulers.hpp"

namespace phi::media {

namespace {

// The scheduler a chain runs when its settings left the name empty, and the one
// `MediaScheduler` still falls back to for an unknown/garbled name. They are the
// released workflows' own: the ACE-Step JSON runs KSampler(scheduler="simple"),
// the H3 workflow BasicScheduler(scheduler="beta").
const char* default_scheduler_name(const std::string& arch) {
	return arch == "minimax_h3" ? "beta" : "simple";
}

}  // namespace

void register_scheduler(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaScheduler";
	n.category = "sampling";
	n.title = "采样调度器";
	n.title_en = "Scheduler (sigmas)";
	n.description =
	    "从模型生成 sigma 网格（移植自 ComfyUI 的调度器：simple / normal / sgm_uniform / "
	    "karras / exponential / ddim_uniform / beta / linear_quadratic / kl_optimal）。"
	    "图像链默认 simple（Flux shift），视频链默认 beta（H3 shift 12/3），音乐链默认 simple。";
	n.description_en =
	    "Builds the sigma grid from the model (ComfyUI schedulers ported: simple / normal / "
	    "sgm_uniform / karras / exponential / ddim_uniform / beta / linear_quadratic / "
	    "kl_optimal). Image defaults to simple (Flux shift), video to beta (H3 shift 12/3), "
	    "music to simple.";
	n.inputs = {port("model", SocketType::Model, false, "the backbone (decides the flow)"),
	            port("steps", SocketType::Int, true, "override the step count")};
	n.outputs = {port("SIGMAS", SocketType::Sigmas, false, "the sigma grid")};
	{
		JsonValue p = JsonValue::object();
		p["scheduler"] = "simple | normal | sgm_uniform | karras | exponential | ddim_uniform | "
		                 "beta | linear_quadratic | kl_optimal";
		p["steps"] = "step count (0 = the chain default)";
		p["denoise"] = "start step fraction (1.0 = the whole schedule)";
		n.params = p;
	}
	n.run = [](GraphContext&, const JsonValue& p, const std::vector<Value>& in) {
		auto m = as_model(in[0]);
		if (!m) throw MediaError("Scheduler: model input is not a MODEL");
		i64 steps = in[1].valid() ? in[1].as_int(0) : param_int(p, "steps", 0);
		std::string name = param_string(p, "scheduler", default_scheduler_name(m->arch));
		SchedulerKind kind = SchedulerKind::Simple;
		if (!scheduler_from_name(name, &kind))
			throw MediaError("Scheduler: unknown scheduler '" + name + "' (known: " +
			                 [&] {
				                 std::string s;
				                 for (const std::string& x : scheduler_names()) {
					                 if (!s.empty()) s += " | ";
					                 s += x;
				                 }
				                 return s;
			                 }() +
			                 ")");
		const float denoise = (float)param_number(p, "denoise", 1.0);
		auto s = std::make_shared<SigmasData>();
		s->scheduler = name;
		if (m->arch == "qwen_image") {
			s->arch = "image";
			if (steps <= 0) steps = kImageSteps;
			if (const char* e = getenv("PHI_IMG_STEPS")) {
				const long long v = atoll(e);
				if (v > 0) steps = (i64)v;
			}
			// The image backbone is ModelSamplingFlux(shift 0.69), carried on the
			// MODEL socket, so the scheduler reads the flow off the model exactly
			// like ComfyUI reads it off `model.model_sampling`.
			s->video = build_sigmas(kind, m->image_flow, (int)steps, denoise);
		} else if (m->arch == "minimax_h3") {
			s->arch = "video";
			if (steps <= 0) steps = kVideoSteps;
			if (const char* e = getenv("PHI_VIDEO_STEPS")) {
				const long v = strtol(e, nullptr, 10);
				if (v > 0 && v <= 64) steps = (i64)v;
			}
			H3SamplingParams sp;
			sp.steps = steps;
			// The shifts come off the MODEL, so a `MediaH3SigmaShift` node in front
			// of the sampler decides the grid: the released 12/3 is the loader's
			// default, and a distilled LoRA's own shift is one node away.
			sp.shift = m->h3_shift_video;
			sp.audio_shift = m->h3_shift_audio;
			sp.alpha = 0.6f;
			sp.beta = 0.6f;
			sp.use_beta = kind == SchedulerKind::Beta;
			if (kind == SchedulerKind::Beta) {
				// Bit-exact beta grid (the float32 evaluation order matters here;
				// see sampling_h3.cpp) and the audio grid the video path reports.
				s->video = h3_beta_sigmas(sp);
				s->audio = h3_audio_sigmas(sp);
			} else {
				// Every other scheduler is built generically from the model's own
				// flow (ModelSamplingAV is a ModelSamplingDiscreteFlow with an audio
				// scale), and the audio grid is the video grid carried onto the audio
				// shift - the same relation the joint sampler relies on.
				SamplingFlow flow;
				flow.kind = SamplingFlow::Kind::DiscreteFlow;
				flow.shift = m->h3_shift_video;
				flow.multiplier = 1.0f;
				s->video = build_sigmas(kind, flow, (int)steps, denoise);
				s->audio.reserve(s->video.size());
				for (float v : s->video)
					s->audio.push_back(h3_time_shift_sigma(v, m->h3_shift_video, m->h3_shift_audio));
			}
		} else if (m->arch == "ace_step15") {
			// ACE-Step 1.5: ModelSamplingAuraFlow == ModelSamplingDiscreteFlow with
			// the shift the MediaAceModelSampling node patched onto the MODEL, and
			// the workflow's `simple` scheduler (KSampler at 50 steps, cfg 1).
			s->arch = "music";
			if (steps <= 0) steps = kMusicSteps;
			SamplingFlow flow;
			flow.kind = SamplingFlow::Kind::DiscreteFlow;
			flow.shift = m->ace_shift;
			flow.multiplier = 1.0f;
			s->video = build_sigmas(kind, flow, (int)steps, denoise);
		} else {
			throw MediaError("Scheduler: unknown model arch '" + m->arch + "'");
		}
		if (s->steps() < 1) throw MediaError("Scheduler: the sigma grid is empty");
		return std::vector<Value>{Value::of_object(s)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
