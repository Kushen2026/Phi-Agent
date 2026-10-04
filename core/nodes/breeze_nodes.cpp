// Nodes "MediaBreezeTextEncode" / "MediaBreezeSampler" / "MediaBreezeDecode" —
// the Breeze-TTS-2 chain's three middle stages.
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
//
// The chain mirrors the other two: a *conditioning* node (the voice-design
// prompt: the instruction that describes the timbre, the words to speak, and
// the negative branch CFG needs), a *sampler* node (the backbone + depth-decoder
// token loop) and a *decode* node (the 12 Hz codec). Nothing here reaches into
// the model: each stage is one call whose input and output travel on a socket.
//
//   MediaModelLoader(breeze_tts) ─ MediaBreezeTextEncode ─ MediaBreezeSampler ─
//       MediaBreezeDecode ─ MediaSaveAudio
#include "nodes/service.hpp"

#include "models/breeze_tts.hpp"

namespace phi::media {

namespace {

BreezeParams read_params(const JsonValue& p) {
	BreezeParams bp;
	bp.text = param_string(p, "text", "");
	bp.instruction = param_string(p, "instruction", "");
	bp.cfg_scale = (float)param_number(p, "cfg_scale", 4.0);
	bp.max_new_tokens = param_int(p, "max_new_tokens", 0);
	bp.temperature = (float)param_number(p, "temperature", 0.7);
	bp.top_k = param_int(p, "top_k", 0);
	bp.top_p = (float)param_number(p, "top_p", 1.0);
	bp.repetition_penalty = (float)param_number(p, "repetition_penalty", 1.0);
	bp.depth_temperature = (float)param_number(p, "depth_temperature", 1.0);
	bp.depth_top_k = param_int(p, "depth_top_k", 0);
	bp.depth_top_p = (float)param_number(p, "depth_top_p", 1.0);
	return bp;
}

std::shared_ptr<BreezeTts> breeze_of(const std::shared_ptr<ModelData>& m, const char* who) {
	if (!m || !m->tts) throw MediaError(std::string(who) + ": model is not a Breeze-TTS-2 model");
	return m->tts;
}

}  // namespace

void register_breeze_text_encode(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaBreezeTextEncode";
	n.category = "conditioning";
	n.title = "文本编码（TTS / 音色描述）";
	n.title_en = "Text Encode (TTS voice design)";
	n.description =
	    "把「要说的文本 + 描述音色的指令」编码成 Breeze-TTS-2 的条件（含 CFG 的负向分支）。";
	n.description_en =
	    "Encodes the spoken text plus the timbre instruction into the Breeze-TTS-2 conditioning "
	    "(including the negative branch CFG needs).";
	n.inputs = {port("model", SocketType::Model, false, "the Breeze-TTS-2 model")};
	n.outputs = {port("CONDITIONING", SocketType::Conditioning, false, "the prompt")};
	{
		JsonValue p = JsonValue::object();
		p["text"] = "the words to speak";
		p["instruction"] = "the timbre / style description";
		p["cfg_scale"] = "classifier-free guidance (4 for voice design)";
		n.params = p;
	}
	n.run = [](GraphContext&, const JsonValue& p, const std::vector<Value>& in) {
		auto model = as_model(in[0]);
		auto tts = breeze_of(model, "BreezeTextEncode");
		const BreezeParams bp = read_params(p);
		if (bp.text.empty()) throw MediaError("BreezeTextEncode: text must not be empty");
		auto pr = tts->prepare(bp);
		auto out = std::make_shared<ConditioningData>();
		out->arch = "tts";
		out->breeze_prompt = pr;
		return std::vector<Value>{Value::of_object(out)};
	};
	reg.add(std::move(n));
}

void register_breeze_sampler(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaBreezeSampler";
	n.category = "sampling";
	n.title = "采样器（TTS）";
	n.title_en = "Sampler (TTS)";
	n.description = "Breeze-TTS-2 的自回归 token 循环（backbone + depth decoder）。";
	n.description_en = "The Breeze-TTS-2 autoregressive token loop (backbone + depth decoder).";
	n.inputs = {port("model", SocketType::Model, false, "the Breeze-TTS-2 model"),
	            port("conditioning", SocketType::Conditioning, false, "the prompt"),
	            port("seed", SocketType::Int, true, "override the seed")};
	n.outputs = {port("LATENT", SocketType::Latent, false, "the acoustic codes")};
	{
		JsonValue p = JsonValue::object();
		p["max_new_tokens"] = "cap on generated frames (0 = the model's own 1500, subject to EOS)";
		p["temperature"] = "backbone sampling temperature";
		p["top_k"] = "backbone top-k (0 = off)";
		p["top_p"] = "backbone nucleus threshold";
		p["repetition_penalty"] = "backbone repetition penalty";
		p["depth_temperature"] = "depth-decoder temperature";
		p["depth_top_k"] = "depth-decoder top-k (0 = off)";
		p["depth_top_p"] = "depth-decoder nucleus threshold";
		p["seed"] = "the sampling seed, used as given (leave the field out to draw one)";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>& in) {
		auto model = as_model(in[0]);
		auto cond = as_conditioning(in[1]);
		auto tts = breeze_of(model, "BreezeSampler");
		if (!cond || !cond->breeze_prompt)
			throw MediaError("BreezeSampler: conditioning is not a TTS prompt");
		BreezeParams bp = read_params(p);
		bp.seed = latent_seed(p, in[2]);
		i64 frames = 0;
		std::vector<i32> codes = tts->generate_codes(
		    std::static_pointer_cast<BreezeTts::Prompt>(cond->breeze_prompt), bp, &frames,
		    [&](i64 cur, i64 total) {
			    ctx.note(ctx.language == "en"
			                 ? ("speaking " + std::to_string(cur) + "/" + std::to_string(total))
			                 : ("生成语音 " + std::to_string(cur) + "/" + std::to_string(total)));
		    },
		    [&]() { return ctx.is_cancelled(); });
		auto out = std::make_shared<LatentData>();
		out->arch = "tts";
		out->codes = std::move(codes);
		out->code_frames = frames;
		out->codebooks = 16;
		return std::vector<Value>{Value::of_object(out)};
	};
	reg.add(std::move(n));
}

void register_breeze_decode(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaBreezeDecode";
	n.category = "audio";
	n.title = "解码（TTS 声码器）";
	n.title_en = "Decode (TTS codec)";
	n.description = "把 Breeze-TTS-2 的 12Hz 声学码解码成 24kHz PCM。";
	n.description_en = "Decodes the Breeze-TTS-2 12 Hz acoustic codes into 24 kHz PCM.";
	n.inputs = {port("model", SocketType::Model, false, "the Breeze-TTS-2 model"),
	            port("latent", SocketType::Latent, false, "the codes")};
	n.outputs = {port("AUDIO", SocketType::Audio, false, "the decoded PCM")};
	n.run = [](GraphContext&, const JsonValue&, const std::vector<Value>& in) {
		auto model = as_model(in[0]);
		auto lat = as_latent(in[1]);
		auto tts = breeze_of(model, "BreezeDecode");
		if (!lat || lat->arch != "tts") throw MediaError("BreezeDecode: latent is not TTS codes");
		auto out = std::make_shared<AudioData>();
		out->sample_rate = tts->sample_rate();
		out->channels = 1;
		out->pcm = tts->decode_codes(lat->codes, lat->code_frames);
		return std::vector<Value>{Value::of_object(out)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
