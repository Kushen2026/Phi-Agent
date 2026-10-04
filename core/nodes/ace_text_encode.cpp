// Node "MediaAceTextEncode" — Text Encode (ACE-Step 1.5).
//
// One node, one file. Registered here; the shared helpers live in
// service.hpp, the catalogue entry point in registry.cpp.
//
// Transcribes ComfyUI's `TextEncodeAceStepAudio1.5`
// (comfy_extras/nodes_ace.py) and the model behind it
// (comfy/text_encoders/ace15.py::ACE15TEModel.encode_token_weights):
//
//   * the Qwen3-0.6B tower runs over the "qwen3_06b" prompt (caption + meta
//     caption) and its **last-layer** hidden state becomes the DiT's `context`
//     (the text projector takes it), while its **layer-0** hidden state over the
//     "lyrics" prompt is the learned lyric conditioning - the *whole* prompt
//     sequence, which at index 0 is the tower's token embeddings (`{"layer":
//     [0]}` selects the state *entering* block 0);
//   * when `generate_audio_codes` is on, the Qwen3-4B LM samples the semantic
//     audio-code stream from the "lm_prompt" with classifier-free guidance
//     (`cfg_scale`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`) — that
//     stream is what ACE-Step 1.5 conditions the DiT on instead of raw audio;
//   * the DiT bundle's own condition encoder / tokenizer / detokenizer
//     (`MediaAceConditionBuilder`) turn the three pieces into the encoder hidden
//     states and the `context_latents` the decoder runs on.
//
// The prompt templates are copied verbatim from `ACE15Tokenizer`, including the
// yaml meta block (`_metas_to_cot` / `_metas_to_cap`) and the `ceil(duration)`
// rounding: the LM was trained on those exact strings.
#include "nodes/service.hpp"

#include <memory>

#include "models/ace_models.hpp"
#include "models/ace_text.hpp"

namespace phi::media {

namespace {

// The DiT bundle takes a second pass to open for its condition encoder (the
// tokenizer, the detokenizer and the FSQ), so the builder is cached on the bundle
// path: a graph opens the same bundle once, but the encode node may run again
// with another prompt, and on a hand-wired graph it may sit in a different
// MediaAceTextEncode node.
//
// It is cached in the run's state bag rather than in a file-static map because it
// holds *device* objects: a lost device tears the bag down (and, with it, this),
// while a static map would hand the next run a builder that points into a
// context that no longer exists.
std::shared_ptr<AceConditionBuilder> condition_builder(GraphContext& ctx, const std::string& path,
                                                       GpuCtx* gpu) {
	auto state = graph_state(ctx);
	const std::string key = "ace_condition|" + path;
	auto it = state->aux.find(key);
	if (it != state->aux.end())
		return std::static_pointer_cast<AceConditionBuilder>(it->second);
	auto b = std::make_shared<AceConditionBuilder>();
	b->open(path, gpu);
	state->aux[key] = b;
	return b;
}

// yaml.dump({...}, sort_keys=True, default_flow_style=False).strip() — the keys
// are always bpm, duration, keyscale, timesignature (sorted) and the values are
// either an int or the bare string, exactly as PyYAML emits them.
// Trim ASCII whitespace from both ends (the reference's Python `str.strip()`).
std::string trim(const std::string& s) {
	size_t b = 0, e = s.size();
	auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
	while (b < e && ws(s[b])) b++;
	while (e > b && ws(s[e - 1])) e--;
	return s.substr(b, e - b);
}

std::string metas_cot(i64 bpm, i64 duration_seconds, const std::string& timesignature,
                      const std::string& keyscale) {
	std::string y = "bpm: " + std::to_string(bpm) + "\n";
	y += "duration: " + std::to_string(duration_seconds) + "\n";
	y += "keyscale: " + keyscale + "\n";
	y += "timesignature: " + timesignature;
	return "<think>\n" + y + "\n</think>";
}

// `_metas_to_cap`: the fixed bpm/timesignature/keyscale/duration order with the
// duration spelled "<n> seconds".
std::string metas_cap(i64 bpm, i64 duration_seconds, const std::string& timesignature,
                      const std::string& keyscale) {
	std::string c = "- bpm: " + std::to_string(bpm) + "\n";
	c += "- timesignature: " + timesignature + "\n";
	c += "- keyscale: " + keyscale + "\n";
	c += "- duration: " + std::to_string(duration_seconds) + " seconds";
	return c;
}

}  // namespace

void register_ace_text_encode(NodeRegistry& reg) {
	NodeDef n;
	n.type = "MediaAceTextEncode";
	n.category = "conditioning";
	n.title = "文本编码（ACE-Step 1.5）";
	n.title_en = "Text Encode (ACE-Step 1.5)";
	n.description =
	    "把 tags / lyrics / 音乐元信息编码成 ACE-Step 1.5 DiT 的条件（含 Qwen3-4B 生成的音频语义码）。";
	n.description_en =
	    "Encodes tags / lyrics / music metadata into the ACE-Step 1.5 DiT conditioning (including "
	    "the audio semantic codes the Qwen3-4B LM samples).";
	n.inputs = {port("clip", SocketType::Clip, false, "the ACE dual text encoder"),
	            port("model", SocketType::Model, false, "the ACE DiT bundle (condition encoder)")};
	n.outputs = {port("CONDITIONING", SocketType::Conditioning, false, "the DiT conditioning")};
	{
		JsonValue p = JsonValue::object();
		p["tags"] = "style / instrument / mood tags (the ComfyUI `tags` widget)";
		p["lyrics"] = "the lyrics, `[instrumental]` for an instrumental";
		p["bpm"] = "tempo";
		p["duration"] = "seconds of music";
		p["seconds"] = "the latent length driver (defaults to duration)";
		p["timesignature"] = "2 | 3 | 4 | 6";
		p["language"] = "lyrics language code (en, zh, ...)";
		p["keyscale"] = "e.g. \"D minor\"";
		p["seed"] = "the LM sampling seed";
		p["generate_audio_codes"] = "run the Qwen3-4B audio-code LM (on by default)";
		p["cfg_scale"] = "the LM's classifier-free guidance scale (7)";
		p["temperature"] = "the LM's sampling temperature";
		p["top_p"] = "the LM's nucleus threshold";
		p["top_k"] = "the LM's top-k (0 = off)";
		p["min_p"] = "the LM's min-p (0 = off)";
		n.params = p;
	}
	n.run = [](GraphContext& ctx, const JsonValue& p, const std::vector<Value>& in) {
		auto clip = as_clip(in[0]);
		auto model = as_model(in[1]);
		if (!clip || !clip->ace_te || !clip->ace_lm)
			throw MediaError("AceTextEncode: clip is not an ACE text encoder");
		if (!model || !model->ace) throw MediaError("AceTextEncode: model is not an ACE DiT");

		// ACE15Tokenizer strips the caption and, inside the LM template, the
		// lyrics (`text = text.strip()` / `lyrics.strip()`).
		const std::string tags = trim(param_string(p, "tags", ""));
		const std::string lyrics = param_string(p, "lyrics", "");
		// `lyrics.strip()` (ace15.py) is what the two LM prompts carry; the lyrics
		// *prompt* keeps the text as given.
		const std::string lyrics_stripped = trim(lyrics);
		const i64 bpm = param_int(p, "bpm", 72);
		const double duration_d = param_number(p, "duration", 60.0);
		const i64 duration = (i64)std::ceil(duration_d);
		const double seconds_d = param_number(p, "seconds", (double)duration);
		// 48 kHz / hop 1920 = 25 latent frames per second (EmptyAceStep1.5LatentAudio).
		const i64 T = std::max<i64>(1, (i64)std::llround(seconds_d * 25.0));
		std::string ts = param_string(p, "timesignature", "4");
		if (ts.size() > 2 && ts.compare(ts.size() - 2, 2, "/4") == 0) ts = ts.substr(0, ts.size() - 2);
		const std::string language = param_string(p, "language", "en");
		const std::string keyscale = param_string(p, "keyscale", "D minor");
		const bool gen_codes = param_bool(p, "generate_audio_codes", true);
		const float cfg = (float)param_number(p, "cfg_scale", 7.0);
		const float temp = (float)param_number(p, "temperature", 0.85);
		const float top_p = (float)param_number(p, "top_p", 0.9);
		const i64 top_k = param_int(p, "top_k", 0);
		const float min_p = (float)param_number(p, "min_p", 0.0);
		const u64 seed = latent_seed(p, Value());

		// ── the three prompt strings (ACE15Tokenizer::tokenize_with_weights) ──
		const std::string cot = metas_cot(bpm, duration, ts, keyscale);
		const std::string cap = metas_cap(bpm, duration, ts, keyscale);
		const std::string lm_template =
		    "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based on the "
		    "given conditions:\n\n<|im_end|>\n<|im_start|>user\n# Caption\n{}\n\n# Lyric\n{}\n"
		    "<|im_end|>\n<|im_start|>assistant\n{}\n\n<|im_end|>\n";
		auto fmt3 = [&](const std::string& a, const std::string& b, const std::string& c) {
			std::string s = lm_template;
			size_t pos = s.find("{}");
			s.replace(pos, 2, a);
			pos = s.find("{}");
			s.replace(pos, 2, b);
			pos = s.find("{}");
			s.replace(pos, 2, c);
			return s;
		};
		const std::string lm_prompt = fmt3(tags, lyrics_stripped, cot);
		const std::string lm_prompt_neg =
		    fmt3(tags, lyrics_stripped, "<think>\n\n</think>");
		const std::string lyrics_prompt =
		    "# Languages\n" + language + "\n\n# Lyric\n" + lyrics + "<|endoftext|><|endoftext|>";
		const std::string te_prompt =
		    "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"
		    "# Caption\n" + tags + "\n\n# Metas\n" + cap +
		    "\n<|endoftext|>\n<|endoftext|>";

		auto tok = clip->tokenizer;
		if (!tok) throw MediaError("AceTextEncode: the clip has no tokenizer");
		const std::vector<i32> te_ids = tok->encode(te_prompt, true);
		const std::vector<i32> lyr_ids = tok->encode(lyrics_prompt, true);
		const std::vector<i32> lm_ids = tok->encode(lm_prompt, true);
		const std::vector<i32> lm_ids_neg = tok->encode(lm_prompt_neg, true);

		// ── the Qwen3-0.6B embedder ──
		std::vector<float> text_hidden = clip->ace_te->encode(te_ids, -1);
		const i64 n_text = (i64)te_ids.size();
		// `conditioning_lyrics` is `qwen3_06b(lyrics_tokens, layer=[0])[:, 0]`
		// (ace15.py::ACE15TEModel.encode_token_weights). With a *list* of layers the
		// tower returns [batch, n_selected, seq, hidden] - the list form captures
		// the state entering the selected block - so `[:, 0]` picks the selected
		// *layer*, not a token: the conditioning is the layer-0 hidden state of the
		// **whole** lyrics prompt. The DiT's own `encoder.lyric_encoder` (8 blocks
		// with RoPE, then packed ahead of the timbre and text blocks) is what turns
		// that sequence into the cross-attention context, so keeping one row here
		// is the same as dropping every lyric token but the first.
		std::vector<float> lyric_hidden = clip->ace_te->encode(lyr_ids, 0);   // [L, 1024]
		const i64 n_lyric = (i64)lyr_ids.size();

		// ── the Qwen3-4B audio-code LM (optional) ──
		std::vector<i32> codes;
		if (gen_codes) {
			const i64 audio_start = 151669;   // the LM's code range (ace15.py)
			const i64 audio_end = 215669;
			const i32 eos_id = 151645;
			// The tokenizer sets both bounds to `tokens_duration = ceil(duration) * 5`
			// (`ace15.py::ACE15Tokenizer.tokenize_with_weights`), so the eos branch is
			// never offered and the LM emits exactly this many codes - which is what
			// `EmptyAceStep1.5LatentAudio` decodes: 5 codes per latent frame, 25 frames
			// per second, i.e. the same `duration` seconds. A shorter run leaves the
			// latent's tail on the FSQ pad code and the clip's second half silent.
			const i64 max_new = std::max<i64>(1, duration * 5);
			ctx.note(ctx.language == "en"
			             ? "generating audio semantic codes (" + std::to_string(max_new) + ")…"
			             : "生成音频语义码（" + std::to_string(max_new) + "）…");
			codes = clip->ace_lm->generate(lm_ids, cfg != 1.0f ? lm_ids_neg : std::vector<i32>(),
			                              max_new, max_new, cfg, temp, top_p, top_k, min_p, seed,
			                              (i32)audio_start, (i32)audio_end, eos_id);
		}
		if (ctx.is_cancelled()) throw MediaError("AceTextEncode: cancelled");

		// ── the DiT bundle's condition builder ──
		AceCondition cond =
		    condition_builder(ctx, model->path, graph_gpu(ctx))
		        ->build(text_hidden, n_text, lyric_hidden, n_lyric, codes, T);
		// The LM's window is already back on the card (its guard released it when
		// `generate` returned), so this is the only place its size can be carried out
		// to the report; the LP's own residency decision is what says whether "too few
		// layers resident" applied to the LM or only to the DiT.
		if (clip->ace_lm) {
			cond.lm_resident_layers = clip->ace_lm->planned_layers();
			cond.lm_total_layers = clip->ace_lm->total_layers();
			cond.lm_resident_bytes = clip->ace_lm->planned_bytes();
		}

		auto out = std::make_shared<ConditioningData>();
		out->arch = "music";
		out->ace_cond = std::make_shared<AceCondition>(cond);
		out->n_ids = cond.n;
		out->dim = 2048;
		return std::vector<Value>{Value::of_object(out)};
	};
	reg.add(std::move(n));
}

}  // namespace phi::media
