// Node "MediaSampler" — one diffusion sampler for every chain.
//
// ComfyUI has one `KSampler`; the model on the socket carries its own sampling
// class (`model.model_sampling`), so the node never learns which network it is
// denoising. This engine used to carry one sampler node per model
// (`MediaKSampler`, `MediaH3Sampler`, `MediaAceSampler`) even though all three
// are the same stage: MODEL + CONDITIONING + LATENT + SIGMAS in, a sampled
// LATENT out. This file is now the only *diffusion* sampler node, and it
// dispatches on `ModelData::arch` — the same shape `MediaScheduler` already uses.
//
// The sampler *name* is now a real setting: the image and music chains hand the
// integrator to `run_sampler` (core/sampling/samplers.*), the H3 joint sampler
// takes the euler / res_multistep choice the video workflow's KSampler implies.
// The default of every chain is its released workflow's own sampler (euler for
// image/music, res_multistep for H3), so an empty setting is byte-for-byte the
// schedule this engine always ran.
//
// `MediaBreezeSampler` is deliberately NOT merged in: TTS is an autoregressive
// token loop, not a sigma-scheduled diffusion step (it has no SIGMAS input at
// all), so it is a different node in ComfyUI too.
//
// One node, one file. Registered here; the shared helpers live in service.hpp,
// the catalogue entry point in registry.cpp.
#include "nodes/service.hpp"

#include "models/ace_models.hpp"
#include "models/av_dit.hpp"
#include "models/image_dit.hpp"
#include "sampling/samplers.hpp"

namespace phi::media {

namespace {

// The sampler name this node uses when its parameter is empty is
// `default_sampler_name(arch)` (core/sampling/samplers.cpp): the released
// workflow's own sampler for that chain.
SamplerKind resolve_sampler(const GraphContext& ctx, const JsonValue& p,
                            const std::string& arch) {
	const std::string name = param_string(p, "sampler", default_sampler_name(arch));
	SamplerKind kind = SamplerKind::Euler;
	if (!sampler_from_name(name, &kind)) {
		std::string known;
		for (const std::string& s : sampler_names()) {
			if (!known.empty()) known += " | ";
			known += s;
		}
		throw MediaError("Sampler: unknown sampler '" + name + "' (known: " + known + ")");
	}
	(void)ctx;
	return kind;
}

// ── image (Qwen-Image 2.1, flux grid) ──────────────────────────────────────
std::shared_ptr<LatentData> sample_image(GraphContext& ctx, const JsonValue& p,
                                         const std::shared_ptr<ModelData>& m,
                                         const std::shared_ptr<ConditioningData>& pos,
                                         const std::shared_ptr<LatentData>& lat,
                                         const std::shared_ptr<SigmasData>& sig) {
	const i64 H = lat->h, W = lat->w, C = lat->c;
	const float cfg = (float)param_number(p, "cfg", 1.0);
	const bool use_neg = pos->n_neg > 0 && cfg > 1.0f;
	const SamplerKind kind = resolve_sampler(ctx, p, m->arch);
	m->image->set_prefix_cache(m->image_prefix_cache && !use_neg);
	if (!m->image->residency_planned())
		m->image->plan_residency(H, W, pos->n_pos, pos->ref_px);
	const i64 planned_layers = m->image->resident_layers();
	const u64 planned_bytes = m->image->resident_bytes();
	const i64 steps = sig->steps();
	if (steps < 1) throw MediaError("Sampler: the sigma grid is empty");
	// The unit noise the latent was drawn from, kept for the stochastic samplers
	// (a flow latent is `noise * sigmas[0]`, and every flow scheduler's first
	// sigma is 1.0, so this is the latent itself for the shipped workflows).
	std::vector<float> unit_noise = lat->x;
	std::vector<float> x = lat->x;
	const std::vector<i64>& ref_px = pos->ref_px;
	std::vector<ImageRefGeom> ref_geom;
	ref_geom.reserve(ref_px.size());
	for (size_t i = 0; i < ref_px.size(); i++) {
		ImageRefGeom rge;
		rge.h = i < pos->ref_hs.size() ? pos->ref_hs[i] : 0;
		rge.w = i < pos->ref_ws.size() ? pos->ref_ws[i] : 0;
		ref_geom.push_back(rge);
	}
	const float s0 = sig->video.empty() ? 1.0f : sig->video[0];
	for (float& v : x) v *= s0;   // sigma0 == 1 for the flux grid, so a no-op
	std::vector<i32> slots = pos->slots;

	// `model(x, sigma)` -> x0: run the backbone (twice when CFG is on), combine
	// the velocities, and convert to the denoised estimate the integrator wants
	// (ModelSamplingFlux: x0 = x - sigma * v).
	auto denoise = [&](const std::vector<float>& xin, float sigma) -> std::vector<float> {
		std::vector<float> v = m->image->forward(xin, H, W, pos->positive, pos->n_pos,
		                                         pos->ref_latents, ref_geom, sigma, slots,
		                                         pos.get());
		if (use_neg) {
			std::vector<float> vu = m->image->forward(xin, H, W, pos->negative, pos->n_neg,
			                                          pos->ref_latents, ref_geom, sigma, slots,
			                                          pos.get());
			for (size_t j = 0; j < v.size(); j++) v[j] = vu[j] + cfg * (v[j] - vu[j]);
		}
		std::vector<float> out(v.size());
		for (size_t j = 0; j < v.size(); j++) out[j] = xin[j] - sigma * v[j];
		return out;
	};
	auto on_step = [&](i64 i, i64 total) {
		if (ctx.is_cancelled()) throw MediaError("Sampler: cancelled");
		if (ctx.on_step) ctx.on_step(i);
		ctx.note(ctx.language == "en"
		             ? ("sampling " + std::to_string(i) + "/" + std::to_string(total) + "…")
		             : ("采样 " + std::to_string(i) + "/" + std::to_string(total) + "…"));
		m->image->tune_residency();
	};
	x = run_sampler(kind, sig->video, std::move(x), denoise, unit_noise, lat->seed, on_step);

	const i64 effective_layers = m->image->resident_layers();
	const bool cache_used = m->image->prefix_cache_ready() && m->image->prefix_cache_planned();
	const i64 cache_rows = m->image->prefix_cache_rows();
	const u64 cache_bytes = m->image->prefix_cache_bytes();
	i64 cache_prefix_rows = pos->n_pos;
	for (i64 px : pos->ref_px) cache_prefix_rows += std::max<i64>(px, 0);
	const u64 cache_wanted = ImageDiT::prefix_cache_charge(cache_prefix_rows, H * W,
	                                                      m->image->config());
	m->image->release_resident();
	release_shared_arenas(static_cast<GpuCtx*>(ctx.gpu));
	auto out = std::make_shared<LatentData>();
	out->arch = "image";
	out->c = C;
	out->h = H;
	out->w = W;
	out->x = std::move(x);
	out->req_w = lat->req_w;
	out->req_h = lat->req_h;
	out->canvas_w = lat->canvas_w;
	out->canvas_h = lat->canvas_h;
	out->seed = lat->seed;
	out->resident_layers = planned_layers;
	out->resident_effective = effective_layers;
	out->resident_bytes = planned_bytes;
	out->prefix_cache = cache_used;
	out->prefix_cache_rows = cache_rows;
	out->prefix_cache_bytes = cache_bytes;
	out->prefix_cache_wanted = cache_wanted;
	return out;
}

// ── video (MiniMax H3, joint A/V) ──────────────────────────────────────────
std::shared_ptr<LatentData> sample_h3(GraphContext& ctx, const JsonValue& p,
                                      const std::shared_ptr<ModelData>& m,
                                      const std::shared_ptr<ConditioningData>& cond,
                                      const std::shared_ptr<LatentData>& lat,
                                      const std::shared_ptr<SigmasData>& sig) {
	const float aug = (float)param_number(p, "visual_cond_timestep", 0.999);
	const i64 vt = lat->vt, lat_h = lat->lat_h, lat_w = lat->lat_w, at = lat->at;
	const i64 frames = lat->plan.frames;
	// The joint A/V walk is the H3 model's own (both streams on one video grid,
	// see h3_joint_av.hpp), so the sampler setting selects the integrator it uses
	// - res_multistep (the reference workflow) or plain euler - rather than a
	// generic per-tensor loop.
	const std::string sname = param_string(p, "sampler", default_sampler_name(m->arch));
	const bool multistep = sname == "res_multistep";
	if (!multistep && sname != "euler")
		throw MediaError("Sampler: the H3 joint A/V walk supports sampler 'euler' or "
		                 "'res_multistep' (got '" + sname + "')");
	const i64 ref_rows = h3_ref_row_count(cond->refs.data(), (i64)cond->refs.size());
	// The reference material's own *pixel* area (each reference image / video
	// contributes its latent grid at the VAE's 16x spatial scale). It is the
	// term that makes the residency plan scale with how big the references are,
	// not only with how many rows they pack - a large reference re-injected on
	// every step costs more than its row count says.
	const u64 ref_pixels = h3_ref_pixel_area(cond->refs.data(), (i64)cond->refs.size());
	m->h3->plan_residency(lat_h, lat_w, frames, cond->n_ids, ref_rows, cond->ref_footprint,
	                      ref_pixels);
	std::vector<float> x_v = lat->xv, x_a = lat->xa;
	const float s0 = sig->video.empty() ? 1.0f : sig->video[0];
	for (float& v : x_v) v *= s0;
	for (float& v : x_a) v *= s0;
	H3SamplingParams sp;
	sp.steps = sig->steps();
	sp.shift = m->h3_shift_video;
	sp.audio_shift = m->h3_shift_audio;
	sp.alpha = 0.6f;
	sp.beta = 0.6f;
	sp.use_beta = sig->scheduler == "beta";
	H3JointSamplerV2 sampler(sp);
	sampler.set_multistep(multistep);
	// The SCHEDULER node's grid is used verbatim: one schedule decides both the
	// sigma the DiT is told and the sigma the integrator advances on.
	sampler.init(std::move(x_v), std::move(x_a), sig->video);
	const size_t n_v = (size_t)(24 * vt * lat_h * lat_w);
	const size_t n_a = (size_t)(32 * 2 * at);
	// The DiT writes straight into these buffers, so their size has to be the
	// latent's, not an assumption: a mismatch here used to be a heap overwrite in
	// forward_refs rather than an error.
	if (sampler.x_v().size() != n_v || sampler.x_a().size() != n_a)
		throw MediaError("Sampler: H3 latent shape mismatch (video " +
		                 std::to_string(sampler.x_v().size()) + " vs " + std::to_string(n_v) +
		                 ", audio " + std::to_string(sampler.x_a().size()) + " vs " +
		                 std::to_string(n_a) + ")");
	std::vector<float> out_v(n_v), out_a(n_a);
	const i64 steps = sampler.steps();
	for (i64 i = 0; i < steps; i++) {
		if (ctx.is_cancelled()) throw MediaError("Sampler: cancelled");
		if (ctx.on_step) ctx.on_step(i);
		ctx.note(ctx.language == "en"
		             ? ("sampling " + std::to_string(i) + "/" + std::to_string(steps) + "…")
		             : ("采样 " + std::to_string(i) + "/" + std::to_string(steps) + "…"));
		m->h3->tune_residency();
		m->h3->forward_refs(sampler.x_v().data(), sampler.x_a().data(), vt, lat_h, lat_w,
		                    cond->embed.data(), cond->tags.data(), cond->n_ids,
		                    sig->video[(size_t)i], cond->refs.data(),
		                    (i64)cond->refs.size(), aug, 0, out_v.data(), out_a.data());
		sampler.step(i, out_v, out_a);
	}
	const H3Residency res_snapshot = m->h3->residency();
	m->h3->release_weights();
	release_shared_arenas(static_cast<GpuCtx*>(ctx.gpu));
	auto out = std::make_shared<LatentData>();
	out->arch = "video";
	out->vt = vt;
	out->lat_h = lat_h;
	out->lat_w = lat_w;
	out->at = at;
	out->plan = lat->plan;
	out->xv = sampler.x_v();
	out->xa = sampler.x_a();
	out->audio_carried = true;
	out->audio_scale = h3_audio_scale(sp);
	out->seed = lat->seed;
	out->h3_residency = res_snapshot;
	return out;
}

// ── music (ACE-Step 1.5, flow, cfg=1) ──────────────────────────────────────
std::shared_ptr<LatentData> sample_music(GraphContext& ctx, const JsonValue& p,
                                         const std::shared_ptr<ModelData>& m,
                                         const std::shared_ptr<ConditioningData>& cond,
                                         const std::shared_ptr<LatentData>& lat,
                                         const std::shared_ptr<SigmasData>& sig) {
	if (!m->ace) throw MediaError("Sampler: model is not an ACE DiT");
	if (!cond || !cond->ace_cond) throw MediaError("Sampler: missing the ACE conditioning");
	if (lat->arch != "music") throw MediaError("Sampler: latent is not a music latent");
	const double cfg = param_number(p, "cfg", 1.0);
	if (cfg != 1.0)
		throw MediaError("Sampler: the released ACE-Step 1.5 workflow runs cfg=1 (got " +
		                 std::to_string(cfg) + ")");
	const SamplerKind kind = resolve_sampler(ctx, p, m->arch);
	const AceCondition& ac = *std::static_pointer_cast<AceCondition>(cond->ace_cond);
	const i64 T = lat->mt;
	if (ac.T != T)
		throw MediaError("Sampler: the conditioning was built for T=" + std::to_string(ac.T) +
		                 " but the latent is T=" + std::to_string(T));
	// The latent lives channel-major ([64, T]); the DiT takes token rows
	// ([T, 64]), exactly the `movedim(-1, -2)` the reference does on entry.
	std::vector<float> xt((size_t)T * 64);
	for (i64 c = 0; c < 64; c++)
		for (i64 t = 0; t < T; t++) xt[(size_t)(t * 64 + c)] = lat->xm[(size_t)(c * T + t)];
	const i64 steps = sig->steps();
	if (steps < 1) throw MediaError("Sampler: the sigma grid is empty");
	// Weight residency: as many of the 32 decoder blocks as fit stay on the card
	// for the whole loop (see AceDiT::plan_residency). The conditioning has been
	// built by now and the block loop has not run yet, which is the one moment
	// both halves of the plan - the activation frame and what is already held -
	// are knowable.
	m->ace->plan_residency(T, ac.n);
	// The window goes back at the phase boundary - before the VAE decode, which is
	// the next thing to want the card - and on *every* exit path, including a
	// cancelled or failed run: a leaked window would still be charged to this
	// process by the accountant, and the next run's plan (which reads the ledger)
	// would then keep that many fewer blocks for itself. `release_resident` is
	// idempotent and safe when nothing was planned.
	struct WindowGuard {
		AceDiT* dit = nullptr;
		~WindowGuard() {
			if (dit) dit->release_resident();
		}
	} window{m->ace.get()};
	// Snapshotted here, not read off the DiT after the loop: `release_resident()`
	// clears the plan at the phase boundary, so a reader that looks at the model
	// afterwards only ever sees the released state (the image and H3 samplers do
	// the same).
	const i64 resident_layers = m->ace->resident_layers();
	const u64 resident_bytes = m->ace->resident_bytes();
	// The latent itself is the unit noise: every flow scheduler starts at 1.0, so
	// the raw draw and the sampler's x are the same tensor here.
	std::vector<float> unit_noise = xt;
	const float s0 = sig->video.empty() ? 1.0f : sig->video[0];
	for (float& v : xt) v *= s0;
	auto denoise = [&](const std::vector<float>& xin, float sigma) -> std::vector<float> {
		std::vector<float> v = m->ace->forward(xin.data(), T, sigma, ac);
		if ((i64)v.size() != T * 64)
			throw MediaError("Sampler: the DiT returned " + std::to_string(v.size()) +
			                 " floats, expected " + std::to_string(T * 64));
		std::vector<float> out(v.size());
		for (size_t k = 0; k < v.size(); k++) out[k] = xin[k] - sigma * v[k];
		return out;
	};
	auto on_step = [&](i64 i, i64 total) {
		if (ctx.is_cancelled()) throw MediaError("Sampler: cancelled");
		if (ctx.on_step) ctx.on_step(i);
		ctx.note(ctx.language == "en"
		             ? ("sampling " + std::to_string(i) + "/" + std::to_string(total) + "…")
		             : ("采样 " + std::to_string(i) + "/" + std::to_string(total) + "…"));
	};
	xt = run_sampler(kind, sig->video, std::move(xt), denoise, unit_noise, lat->seed, on_step);
	release_shared_arenas(static_cast<GpuCtx*>(ctx.gpu));
	auto out = std::make_shared<LatentData>();
	out->arch = "music";
	out->mt = T;
	out->seconds = lat->seconds;
	out->seed = lat->seed;
	out->resident_layers = resident_layers;
	out->resident_bytes = resident_bytes;
	out->xm.assign((size_t)64 * (size_t)T, 0.0f);
	for (i64 t = 0; t < T; t++)
		for (i64 c = 0; c < 64; c++) out->xm[(size_t)(c * T + t)] = xt[(size_t)(t * 64 + c)];
	return out;
}

}  // namespace

void register_sampler(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaSampler";
	n.category = "sampling";
	n.title = "采样器";
	n.title_en = "Sampler";
	n.description =
	    "扩散采样循环（移植自 ComfyUI 的采样器：euler / euler_ancestral / heun / dpmpp_2m / "
	    "dpmpp_2s_ancestral / dpmpp_2m_sde / dpmpp_sde / res_multistep / lcm）。"
	    "接在 model 上的主干决定用哪套流程（Qwen-Image、MiniMax H3 联合音视频、ACE-Step 1.5）；"
	    "节点本身只有一个。";
	n.description_en =
	    "The diffusion sampling loop (ComfyUI samplers ported: euler / euler_ancestral / heun / "
	    "dpmpp_2m / dpmpp_2s_ancestral / dpmpp_2m_sde / dpmpp_sde / res_multistep / lcm). The "
	    "backbone on the model socket decides the flow (Qwen-Image, MiniMax H3 joint A/V, "
	    "ACE-Step 1.5); there is one node.";
	n.inputs = {port("model", SocketType::Model, false, "the backbone (decides the flow)"),
	            port("conditioning", SocketType::Conditioning, false, "the encoded prompt(s)"),
	            port("latent", SocketType::Latent, false, "the noisy latent"),
	            port("sigmas", SocketType::Sigmas, false, "the sigma grid")};
	n.outputs = {port("LATENT", SocketType::Latent, false, "the sampled latent")};
	{
		JsonValue p = JsonValue::object();
		p["sampler"] = "euler | euler_ancestral | heun | dpmpp_2m | dpmpp_2s_ancestral | "
		               "dpmpp_2m_sde | dpmpp_sde | res_multistep | lcm "
		               "(H3: euler | res_multistep)";
		p["cfg"] = "classifier-free guidance scale (image chain); the music chain runs 1";
		p["visual_cond_timestep"] = "H3 reference visual noise augmentation (0.999)";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>& in) {
		auto m = as_model(in[0]);
		auto cond = as_conditioning(in[1]);
		auto lat = as_latent(in[2]);
		auto sig = as_sigmas(in[3]);
		if (!m) throw MediaError("Sampler: model is not a MODEL");
		if (!cond || !lat || !sig) throw MediaError("Sampler: missing an input");
		if (m->arch == "qwen_image") {
			if (!m->image) throw MediaError("Sampler: the Qwen-Image backbone is not loaded");
			return std::vector<Value>{Value::of_object(sample_image(ctx, p, m, cond, lat, sig))};
		}
		if (m->arch == "minimax_h3") {
			if (!m->h3) throw MediaError("Sampler: the MiniMax H3 backbone is not loaded");
			return std::vector<Value>{Value::of_object(sample_h3(ctx, p, m, cond, lat, sig))};
		}
		if (m->arch == "ace_step15")
			return std::vector<Value>{Value::of_object(sample_music(ctx, p, m, cond, lat, sig))};
		throw MediaError("Sampler: unknown model arch '" + m->arch +
		                 "' (qwen_image | minimax_h3 | ace_step15)");
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
