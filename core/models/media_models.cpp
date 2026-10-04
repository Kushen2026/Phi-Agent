// Media model registry implementation. See media_models.hpp for the contract.
#include "models/media_models.hpp"

#include <windows.h>
#include <winhttp.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>

#include "util/base.hpp"

namespace phi::media {

namespace {

// The shipped default file per role (the filenames the loaders used to hard-code).
// An empty `file` means the role has no shipped file: a blank selection stays
// blank so the loader keeps its own fallback (e.g. the sibling config.json).
struct DefaultRole {
	const char* role;
	const char* id;
	const char* subdir;
	const char* file;
};
const DefaultRole kDefaults[] = {
    // ── image (Qwen-Image 2.1) ─────────────────────────────────────────────
    {"image_dit", "qwen_image_2.1_int8_convrot", "qwen_image_2.1",
     "qwen_image_2.1_int8_convrot.safetensors"},
    {"image_te", "qwen3vl_8b_int8_convrot", "qwen_image_2.1",
     "qwen3vl_8b_int8_convrot.safetensors"},
    {"image_vae", "qwen_image_2.1_vae_bf16", "qwen_image_2.1",
     "qwen_image_2.1_vae_bf16.safetensors"},
    // ── video (MiniMax H3) ─────────────────────────────────────────────────
    {"video_dit", "minimax_h3_ref2va_int8_convrot", "minimax_h3",
     "minimax_h3_hybrid_ref2va_b25-49_pruned_int8_convrot.safetensors"},
    {"video_te", "minimax_h3_qwen3vl_32b_int4", "minimax_h3",
     "minimax_h3_qwen3vl_32b_convrot_int4.safetensors"},
    {"video_vae", "minimax_h3_video_vae_fp16", "minimax_h3",
     "minimax_h3_video_vae_fp16.safetensors"},
    {"video_avae", "minimax_h3_audio_vae_fp32", "minimax_h3",
     "minimax_h3_audio_vae_fp32.safetensors"},
    // ── music (ACE-Step 1.5) ───────────────────────────────────────────────
    {"music_dit", "ace_step15_xl_sft_bf16", "ace_step_1.5",
     "acestep_v1.5_xl_sft_bf16.safetensors"},
    {"music_te", "ace15_qwen3_0.6b", "ace_step_1.5", "qwen_0.6b_ace15.safetensors"},
    {"music_lm", "ace15_qwen3_4b", "ace_step_1.5", "qwen_4b_ace15.safetensors"},
    {"music_vae", "ace15_vae_bf16", "ace_step_1.5", "ace_1.5_vae.safetensors"},
    // ── tts (Breeze-TTS-2) ─────────────────────────────────────────────────
	{"tts_model", "breeze_tts2_bf16", "breeze_tts_2", "breeze_tts_2_bf16.safetensors"},
	{"tts_codec", "breeze_tts2_codec", "breeze_tts_2",
	 "breeze_tts_2_audio_tokenizer.safetensors"},
	{"tts_tokenizer", "breeze_tts2_tokenizer", "breeze_tts_2", "tokenizer.json"},
	// ── tokenizer files: three per chain, no chain shares another's ────────
	// The three files the Qwen2 tokenizer reads, each its own role: the token ->
	// id table, the BPE merge ranks, and the special-token ids. (The three chains
	// read the same Qwen2.5 vocabulary today, but the *selection* is per-model.)
	{"image_tokenizer_vocab", "qwen_image_tokenizer_vocab", "qwen_image_2.1", "vocab.json"},
	{"image_tokenizer_merges", "qwen_image_tokenizer_merges", "qwen_image_2.1", "merges.txt"},
	{"image_tokenizer_config", "qwen_image_tokenizer_config", "qwen_image_2.1",
	 "tokenizer_config.json"},
	{"video_tokenizer_vocab", "minimax_h3_tokenizer_vocab", "minimax_h3", "vocab.json"},
	{"video_tokenizer_merges", "minimax_h3_tokenizer_merges", "minimax_h3", "merges.txt"},
	{"video_tokenizer_config", "minimax_h3_tokenizer_config", "minimax_h3",
	 "tokenizer_config.json"},
	{"music_tokenizer_vocab", "ace15_tokenizer_vocab", "ace_step_1.5", "vocab.json"},
	{"music_tokenizer_merges", "ace15_tokenizer_merges", "ace_step_1.5", "merges.txt"},
	{"music_tokenizer_config", "ace15_tokenizer_config", "ace_step_1.5",
	 "tokenizer_config.json"},
	// Breeze-TTS-2's own config.json and the codec's own config.json. They are two
	// different files with two different formats (the model's scalars at the top
	// level, the codec's under `decoder_config`), which is why the codec is no
	// longer pointed at the model's.
	{"tts_config", "breeze_tts2_config", "breeze_tts_2", "config.json"},
	{"tts_codec_config", "breeze_tts2_codec_config", "breeze_tts_2",
	 "audio_tokenizer_config.json"},
};

const DefaultRole* default_role(const std::string& role) {
	for (const DefaultRole& d : kDefaults)
		if (role == d.role) return &d;
	return nullptr;
}

}  // namespace

std::vector<MediaModelEntry> media_model_catalog() {
	std::vector<MediaModelEntry> c;
	auto add = [&](const char* id, const char* name, const char* role, const char* format,
	               const char* subdir, const char* file, const char* url, u64 size) {
		MediaModelEntry e;
		e.id = id;
		e.name = name;
		e.role = role;
		e.format = format;
		e.subdir = subdir;
		e.file = file;
		e.url = url ? url : "";
		e.size = size;
		c.push_back(std::move(e));
	};
	// The shipped int8 checkpoints (the engine's calibrated defaults).
	add("qwen_image_2.1_int8_convrot", "Qwen-Image-2.1 DiT (int8 convrot)", "image_dit",
	    "safetensors", "qwen_image_2.1", "qwen_image_2.1_int8_convrot.safetensors", "", 0);
	add("qwen3vl_8b_int8_convrot", "Qwen3-VL-8B TE (int8 convrot)", "image_te", "safetensors",
	    "qwen_image_2.1", "qwen3vl_8b_int8_convrot.safetensors", "", 0);
	add("qwen3vl_8b_w4a8", "Qwen3-VL-8B TE (w4a8, codebook)", "image_te", "safetensors",
	    "qwen_image_2.1", "qwen3vl_8b_w4a8.safetensors", "", 0);
	add("qwen_image_2.1_vae_bf16", "Qwen-Image-2.1 VAE (bf16)", "image_vae", "safetensors",
	    "qwen_image_2.1",
	    "qwen_image_2.1_vae_bf16.safetensors", "", 0);
	add("minimax_h3_ref2va_int8_convrot", "MiniMax H3 ref2va DiT (int8 convrot)", "video_dit",
	    "safetensors", "minimax_h3",
	    "minimax_h3_hybrid_ref2va_b25-49_pruned_int8_convrot.safetensors", "", 0);
	add("minimax_h3_qwen3vl_32b_int4", "Qwen3-VL-32B TE (int4 convrot)", "video_te",
	    "safetensors", "minimax_h3", "minimax_h3_qwen3vl_32b_convrot_int4.safetensors", "", 0);
	add("minimax_h3_video_vae_fp16", "H3 video VAE (fp16)", "video_vae", "safetensors",
	    "minimax_h3",
	    "minimax_h3_video_vae_fp16.safetensors", "", 0);
	add("minimax_h3_video_vae_int8_convrot", "H3 video VAE (int8 convrot)", "video_vae",
	    "safetensors", "minimax_h3", "minimax_h3_video_vae_int8_convrot.safetensors", "", 0);
	add("minimax_h3_audio_vae_fp32", "H3 audio VAE + vocoder (fp32)", "video_avae", "safetensors",
	    "minimax_h3", "minimax_h3_audio_vae_fp32.safetensors", "", 0);
	// ── music (ACE-Step 1.5): a DiT bundle, two Qwen3 towers and the audio VAE ──
	add("ace_step15_xl_sft_bf16", "ACE-Step 1.5 XL DiT (bf16)", "music_dit", "safetensors",
	    "ace_step_1.5", "acestep_v1.5_xl_sft_bf16.safetensors", "", 0);
	add("ace15_qwen3_0.6b", "ACE-Step 1.5 embedder Qwen3-0.6B (bf16)", "music_te",
	    "safetensors", "ace_step_1.5", "qwen_0.6b_ace15.safetensors", "", 0);
	add("ace15_qwen3_4b", "ACE-Step 1.5 audio-code LM Qwen3-4B (bf16)", "music_lm",
	    "safetensors", "ace_step_1.5", "qwen_4b_ace15.safetensors", "", 0);
	add("ace15_vae_bf16", "ACE-Step 1.5 audio VAE (bf16)", "music_vae", "safetensors",
	    "ace_step_1.5", "ace_1.5_vae.safetensors", "", 0);
	// ── tts (Breeze-TTS-2): one checkpoint + the codec + the tokenizer ──
	add("breeze_tts2_bf16", "Breeze-TTS-2 (bf16)", "tts_model", "safetensors", "breeze_tts_2",
	    "breeze_tts_2_bf16.safetensors", "", 0);
	add("breeze_tts2_codec", "Breeze-TTS-2 codec (Qwen3-TTS 12 Hz)", "tts_codec",
	    "safetensors", "breeze_tts_2", "breeze_tts_2_audio_tokenizer.safetensors", "", 0);
	add("breeze_tts2_tokenizer", "Breeze-TTS-2 tokenizer.json", "tts_tokenizer", "json",
	    "breeze_tts_2", "tokenizer.json", "", 0);
	// ── the JSON configs (selectable, never downloaded) ─────────────────────
	// One per model: each chain's tokenizer config lives beside that chain's
	// weights, and the Breeze model / codec configs are two different files.
	add("qwen_image_tokenizer_vocab", "Qwen-Image tokenizer vocab.json",
	    "image_tokenizer_vocab", "json", "qwen_image_2.1", "vocab.json", "", 0);
	add("qwen_image_tokenizer_merges", "Qwen-Image tokenizer merges.txt",
	    "image_tokenizer_merges", "json", "qwen_image_2.1", "merges.txt", "", 0);
	add("qwen_image_tokenizer_config", "Qwen-Image tokenizer tokenizer_config.json",
	    "image_tokenizer_config", "json", "qwen_image_2.1", "tokenizer_config.json", "", 0);
	add("minimax_h3_tokenizer_vocab", "MiniMax H3 tokenizer vocab.json",
	    "video_tokenizer_vocab", "json", "minimax_h3", "vocab.json", "", 0);
	add("minimax_h3_tokenizer_merges", "MiniMax H3 tokenizer merges.txt",
	    "video_tokenizer_merges", "json", "minimax_h3", "merges.txt", "", 0);
	add("minimax_h3_tokenizer_config", "MiniMax H3 tokenizer tokenizer_config.json",
	    "video_tokenizer_config", "json", "minimax_h3", "tokenizer_config.json", "", 0);
	add("ace15_tokenizer_vocab", "ACE-Step 1.5 tokenizer vocab.json",
	    "music_tokenizer_vocab", "json", "ace_step_1.5", "vocab.json", "", 0);
	add("ace15_tokenizer_merges", "ACE-Step 1.5 tokenizer merges.txt",
	    "music_tokenizer_merges", "json", "ace_step_1.5", "merges.txt", "", 0);
	add("ace15_tokenizer_config", "ACE-Step 1.5 tokenizer tokenizer_config.json",
	    "music_tokenizer_config", "json", "ace_step_1.5", "tokenizer_config.json", "", 0);
	add("breeze_tts2_config", "Breeze-TTS-2 config.json", "tts_config", "json", "breeze_tts_2",
	    "config.json", "", 0);
	add("breeze_tts2_codec_config", "Breeze-TTS-2 codec config.json", "tts_codec_config",
	    "json", "breeze_tts_2", "audio_tokenizer_config.json", "", 0);
	return c;
}

std::vector<MediaModelEntry> media_model_catalog_for(const std::string& role) {
	std::vector<MediaModelEntry> out;
	for (const MediaModelEntry& e : media_model_catalog())
		if (e.role == role) out.push_back(e);
	return out;
}

const MediaModelEntry* media_model_entry(const std::string& id) {
	static const std::vector<MediaModelEntry> c = media_model_catalog();
	for (const MediaModelEntry& e : c)
		if (e.id == id) return &e;
	return nullptr;
}

std::string default_media_model_id(const std::string& role) {
	const DefaultRole* d = default_role(role);
	return d ? d->id : std::string();
}

bool media_model_present(const MediaModelEntry& entry, const std::string& models_dir) {
	return path_exists(path_join(path_join(models_dir, entry.subdir), entry.file));
}

namespace {

// Depth-first walk that records every file relative to the root. Every regular
// file is listed, including the `.json` configs: the settings panel offers them
// so each chain's tokenizer / model / codec config can be chosen, and the same
// list feeds resolve_media_file()'s bare-filename scan. Each model directory is
// self-contained now (its vocabulary sits beside its weights), so nothing has to
// be skipped for a "shared" file to keep its precedence.
void collect_files(const std::string& dir, const std::string& prefix,
                   std::vector<std::string>* out) {
	for (const std::string& name : list_dir(dir)) {
		std::string full = path_join(dir, name);
		std::string rel = prefix.empty() ? name : prefix + "/" + name;
		if (path_is_dir(full)) collect_files(full, rel, out);
		else out->push_back(std::move(rel));
	}
}

}  // namespace

std::vector<std::string> list_media_files(const std::string& models_dir) {
	std::vector<std::string> out;
	if (path_is_dir(models_dir)) collect_files(models_dir, "", &out);
	std::sort(out.begin(), out.end());
	return out;
}

std::string resolve_media_file(const std::string& models_dir, const std::string& name) {
	if (name.empty()) return "";
	// 1. already a usable path (absolute, or relative to the working directory).
	if (path_exists(name)) return path_normalize(name);
	// 2. exactly this name directly inside models/.
	std::string p = path_join(models_dir, name);
	if (path_exists(p)) return p;
	// 3. the same name somewhere under models/ (given without its subfolder).
	const std::string base = path_basename(name);
	for (const std::string& rel : list_media_files(models_dir))
		if (rel == name || path_basename(rel) == base) return path_join(models_dir, rel);
	return "";
}

std::string resolve_media_role(const JsonValue& media_settings, const std::string& role,
                               const std::string& models_dir) {
	const DefaultRole* d = default_role(role);
	if (!d) return "";
	const JsonValue* sel = media_settings.find(role);
	const std::string id = sel ? sel->as_string("") : "";

	// 1. the explicit selection: a file the user picked from the models listing
	//    (or, for a settings file written by an older build, a catalogue id).
	if (!id.empty()) {
		std::string p = resolve_media_file(models_dir, id);
		if (!p.empty()) return p;
		if (const MediaModelEntry* e = media_model_entry(id)) {
			p = path_join(path_join(models_dir, e->subdir), e->file);
			if (path_exists(p)) return p;
		}
		// Not on disk: name the file the loader was asked for so the error is
		// actionable instead of an empty string.
		return path_join(models_dir, id);
	}
	// 2. an empty selection falls back to the shipped default. The role's own
	//    subdirectory is tried first and by full path: several roles share a
	//    filename now (`tokenizer_config.json` exists in every chain's directory),
	//    so a bare-basename search would hand the image chain the music chain's
	//    vocabulary - alphabetically first - and be wrong without ever saying so.
	if (d->file && d->file[0]) {
		if (d->subdir && d->subdir[0]) {
			std::string exact = path_join(path_join(models_dir, d->subdir), d->file);
			if (path_exists(exact)) return exact;
		}
		// 2b. the file was hand-placed elsewhere under models/ (the layout is not
		//     fixed): accept it, but only when the name is unambiguous.
		std::vector<std::string> hits;
		for (const std::string& rel : list_media_files(models_dir))
			if (path_basename(rel) == d->file) hits.push_back(path_join(models_dir, rel));
		if (hits.size() == 1) return hits[0];
		if (!hits.empty()) return hits[0];
		// 3. nothing on disk: the default path, again for a useful error message.
		return path_join(path_join(models_dir, d->subdir ? d->subdir : ""), d->file);
	}
	return "";
}

MediaModelSpec resolve_media_spec(const JsonValue& media_settings, const std::string& models_dir) {
	MediaModelSpec s;
	s.image_dit = resolve_media_role(media_settings, "image_dit", models_dir);
	s.image_te = resolve_media_role(media_settings, "image_te", models_dir);
	s.image_vae = resolve_media_role(media_settings, "image_vae", models_dir);
	s.video_dit = resolve_media_role(media_settings, "video_dit", models_dir);
	s.video_te = resolve_media_role(media_settings, "video_te", models_dir);
	s.video_vae = resolve_media_role(media_settings, "video_vae", models_dir);
	s.video_avae = resolve_media_role(media_settings, "video_avae", models_dir);
	s.music_dit = resolve_media_role(media_settings, "music_dit", models_dir);
	s.music_te = resolve_media_role(media_settings, "music_te", models_dir);
	s.music_lm = resolve_media_role(media_settings, "music_lm", models_dir);
	s.music_vae = resolve_media_role(media_settings, "music_vae", models_dir);
	s.tts_model = resolve_media_role(media_settings, "tts_model", models_dir);
	s.tts_codec = resolve_media_role(media_settings, "tts_codec", models_dir);
	s.tts_tokenizer = resolve_media_role(media_settings, "tts_tokenizer", models_dir);
	// The tokenizer files: three roles per chain, resolved independently.
	//
	// Migration, in order of precedence:
	//   1. an older settings.json carried a single `tokenizer_config` used by all
	//      three chains (its *directory* supplied the vocabulary). A chain with no
	//      selection of its own inherits that directory's files, so an install
	//      predating the per-file roles keeps working;
	//   2. `PHI_TOKENIZER_DIR` (the benchmark's way) still overrides all three at
	//      once;
	//   3. otherwise each role falls back to its own shipped default, inside that
	//      chain's model directory.
	// Read as a plain string, not through `resolve_media_role`: the legacy shared
	// role is gone from the default table on purpose.
	std::string legacy_dir;
	if (const JsonValue* leg = media_settings.find("tokenizer_config"); leg) {
		const std::string v = leg->as_string("");
		if (!v.empty()) {
			const std::string p = resolve_media_file(models_dir, v);
			const std::string full = p.empty() ? v : p;
			legacy_dir = path_dirname(full);
		}
	}
	if (const char* v = getenv("PHI_TOKENIZER_DIR"); v && *v) {
		const std::string p = resolve_media_file(models_dir, v);
		legacy_dir = p.empty() ? std::string(v) : p;
	}
	auto tok_role = [&](const char* role, const char* file) -> std::string {
		// An explicit per-file selection wins (it can be a path or a bare name).
		if (const JsonValue* sel = media_settings.find(role); sel && !sel->as_string("").empty()) {
			const std::string p = resolve_media_file(models_dir, sel->as_string(""));
			if (!p.empty()) return p;
		}
		// The legacy / environment directory, then the chain's own default.
		if (!legacy_dir.empty()) {
			const std::string p = path_join(legacy_dir, file);
			if (path_exists(p)) return p;
		}
		return resolve_media_role(media_settings, role, models_dir);
	};
	s.image_tokenizer_vocab = tok_role("image_tokenizer_vocab", "vocab.json");
	s.image_tokenizer_merges = tok_role("image_tokenizer_merges", "merges.txt");
	s.image_tokenizer_config = tok_role("image_tokenizer_config", "tokenizer_config.json");
	s.video_tokenizer_vocab = tok_role("video_tokenizer_vocab", "vocab.json");
	s.video_tokenizer_merges = tok_role("video_tokenizer_merges", "merges.txt");
	s.video_tokenizer_config = tok_role("video_tokenizer_config", "tokenizer_config.json");
	s.music_tokenizer_vocab = tok_role("music_tokenizer_vocab", "vocab.json");
	s.music_tokenizer_merges = tok_role("music_tokenizer_merges", "merges.txt");
	s.music_tokenizer_config = tok_role("music_tokenizer_config", "tokenizer_config.json");
	s.tts_config = resolve_media_role(media_settings, "tts_config", models_dir);
	s.tts_codec_config = resolve_media_role(media_settings, "tts_codec_config", models_dir);

	// `PHI_IMAGE_DIT` / `PHI_VIDEO_DIT` (and the same for the other roles)
	// replace the resolved path outright. The settings file is the app's, and the
	// benchmark has none (`resolve_media_spec(JsonValue::object(), ...)`), so this
	// is the only way to put the *benchmark* on a different checkpoint. A bare
	// filename is looked up under models/ like a settings entry, so it does not
	// have to be a full path.
	{
		static const char* kRoles[] = {"image_dit",      "image_te",     "image_vae",
		                               "image_tokenizer_vocab",  "image_tokenizer_merges",
		                               "image_tokenizer_config", "video_dit",    "video_te",
		                               "video_vae",    "video_avae",   "video_tokenizer_vocab",
		                               "video_tokenizer_merges", "video_tokenizer_config",
		                               "music_dit",    "music_te",     "music_lm",
		                               "music_vae",    "music_tokenizer_vocab",
		                               "music_tokenizer_merges", "music_tokenizer_config",
		                               "tts_model",    "tts_codec",    "tts_tokenizer",
		                               "tts_config",   "tts_codec_config"};
		std::string* slots[] = {&s.image_dit,    &s.image_te,   &s.image_vae,
		                        &s.image_tokenizer_vocab,   &s.image_tokenizer_merges,
		                        &s.image_tokenizer_config,  &s.video_dit,  &s.video_te,
		                        &s.video_vae,    &s.video_avae, &s.video_tokenizer_vocab,
		                        &s.video_tokenizer_merges,  &s.video_tokenizer_config,
		                        &s.music_dit,    &s.music_te,   &s.music_lm,
		                        &s.music_vae,    &s.music_tokenizer_vocab,
		                        &s.music_tokenizer_merges,  &s.music_tokenizer_config,
		                        &s.tts_model,    &s.tts_codec,  &s.tts_tokenizer,
		                        &s.tts_config,   &s.tts_codec_config};
		const size_t n_roles = sizeof(kRoles) / sizeof(kRoles[0]);
		for (size_t i = 0; i < n_roles; i++) {
			std::string key = "PHI_";
			for (const char* c = kRoles[i]; *c; c++) key += (char)toupper((unsigned char)*c);
			const char* v = getenv(key.c_str());
			if (!v || !*v) continue;
			std::string p = resolve_media_file(models_dir, v);
			*slots[i] = p.empty() ? std::string(v) : p;
		}
	}


	// LoRAs: the settings name up to two files per chain (by path or by bare
	// name). Their order is the order they are stored in, which is the order the
	// loader applies them. The two chains are never merged.
	// The settings' LoRA lists travel to the chain as the MediaLoraLoader node's
	// input (see `core/workflows/`), and they are *resolved here*: the node's
	// parameter is a file name, and a bare name has to become the path under
	// models/ before it reaches the engine - the same lookup every other role uses.
	auto read_loras = [&](const char* key, std::vector<std::string>* out) {
		for (const JsonValue& v : media_settings[key].items()) {
			std::string name = v.as_string("");
			if (name.empty()) continue;
			// A settings entry may be a list ("a.safetensors, b.safetensors") or a
			// single name; both spellings are in the wild.
			size_t start = 0;
			while (start <= name.size()) {
				const size_t comma = name.find(',', start);
				std::string one = name.substr(start, comma == std::string::npos
				                                      ? std::string::npos
				                                      : comma - start);
				one = trim(one);
				start = comma == std::string::npos ? name.size() + 1 : comma + 1;
				if (one.empty()) continue;
				std::string p = resolve_media_file(models_dir, one);
				out->push_back(p.empty() ? one : p);
			}
		}
	};
	read_loras("image_loras", &s.image_loras);
	read_loras("video_loras", &s.video_loras);
	// Legacy: a settings file from a build that stored one shared list. Apply it
	// to both chains, which is what that build did.
	if (s.image_loras.empty() && s.video_loras.empty() && media_settings.find("loras")) {
		read_loras("loras", &s.image_loras);
		read_loras("loras", &s.video_loras);
	}
	return s;
}

// ── download ────────────────────────────────────────────────────────────────
//
// A plain WinHTTP GET streamed to a temporary file, then an atomic rename. The
// engine already links winhttp for the web tools, so this adds no dependency.
// The partial file is written as "<file>.part" and only renamed on success, so a
// cancelled or failed download never leaves something the loader would try to
// open.
namespace {

struct Url {
	std::wstring host, path;
	INTERNET_PORT port = 0;
	bool https = true;
};

bool parse_url(const std::string& url, Url* out, std::string* err) {
	std::wstring w = utf8_to_wide(url);
	URL_COMPONENTS uc{};
	uc.dwStructSize = sizeof(uc);
	wchar_t host[256] = {0}, path[2048] = {0};
	uc.lpszHostName = host;
	uc.dwHostNameLength = 256;
	uc.lpszUrlPath = path;
	uc.dwUrlPathLength = 2048;
	if (!WinHttpCrackUrl(w.c_str(), (DWORD)w.size(), 0, &uc)) {
		if (err) *err = "invalid URL: " + url;
		return false;
	}
	out->host.assign(uc.lpszHostName, uc.dwHostNameLength);
	out->path.assign(uc.lpszUrlPath, uc.dwUrlPathLength);
	out->port = uc.nPort;
	out->https = uc.nScheme == INTERNET_SCHEME_HTTPS;
	return true;
}

}  // namespace

void media_model_download(const MediaModelEntry& entry, const std::string& models_dir,
                          const std::function<void(double)>& progress,
                          const std::function<bool()>& cancel) {
	if (entry.url.empty())
		throw MediaError("model '" + entry.id + "' has no download URL; place " + entry.file +
		                 " in models/" + entry.subdir + "/ by hand");
	Url url;
	std::string err;
	if (!parse_url(entry.url, &url, &err)) throw MediaError(err);

	const std::string dir = path_join(models_dir, entry.subdir);
	mkdirs(dir);
	const std::string target = path_join(dir, entry.file);
	const std::string part = target + ".part";

	HINTERNET session = WinHttpOpen(L"phi/1.0", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
	                                WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
	if (!session) throw MediaError("WinHttpOpen failed");
	HINTERNET connect = WinHttpConnect(session, url.host.c_str(), url.port, 0);
	HINTERNET request = connect ? WinHttpOpenRequest(connect, L"GET", url.path.c_str(), nullptr,
	                                                 WINHTTP_NO_REFERER,
	                                                 WINHTTP_DEFAULT_ACCEPT_TYPES,
	                                                 url.https ? WINHTTP_FLAG_SECURE : 0)
	                            : nullptr;
	FILE* f = nullptr;
	HINTERNET conn_guard = connect, req_guard = request;
	auto cleanup = [&]() {
		if (f) fclose(f);
		if (req_guard) WinHttpCloseHandle(req_guard);
		if (conn_guard) WinHttpCloseHandle(conn_guard);
		if (session) WinHttpCloseHandle(session);
	};
	if (!request || !WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
	                                    WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
	    !WinHttpReceiveResponse(request, nullptr)) {
		cleanup();
		throw MediaError("download failed (connect) for " + entry.url);
	}
	DWORD status = 0, slen = sizeof(status);
	WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
	                    WINHTTP_HEADER_NAME_BY_INDEX, &status, &slen, WINHTTP_NO_HEADER_INDEX);
	if (status != 200) {
		cleanup();
		throw MediaError("download failed: HTTP " + std::to_string(status) + " for " + entry.url);
	}
	u64 total = entry.size;
	{
		wchar_t len[64] = {0};
		DWORD llen = sizeof(len);
		if (WinHttpQueryHeaders(request, WINHTTP_QUERY_CONTENT_LENGTH, WINHTTP_HEADER_NAME_BY_INDEX,
		                        len, &llen, WINHTTP_NO_HEADER_INDEX)) {
			try {
				total = std::stoull(wide_to_utf8(len));
			} catch (...) {
			}
		}
	}
	f = _wfopen(utf8_to_wide(part).c_str(), L"wb");
	if (!f) {
		cleanup();
		throw MediaError("cannot write " + part);
	}
	u64 done = 0;
	std::vector<u8> buf(1 << 20);
	for (;;) {
		if (cancel && cancel()) {
			cleanup();
			DeleteFileW(utf8_to_wide(part).c_str());
			throw MediaError("download cancelled");
		}
		DWORD got = 0;
		if (!WinHttpReadData(request, buf.data(), (DWORD)buf.size(), &got)) {
			cleanup();
			DeleteFileW(utf8_to_wide(part).c_str());
			throw MediaError("download failed (read) for " + entry.url);
		}
		if (got == 0) break;
		if (fwrite(buf.data(), 1, got, f) != got) {
			cleanup();
			DeleteFileW(utf8_to_wide(part).c_str());
			throw MediaError("download failed (write) for " + entry.file);
		}
		done += got;
		if (progress) progress(total ? (double)done / (double)total : -1.0);
	}
	fclose(f);
	f = nullptr;

	// Atomic publish.
	if (!MoveFileExW(utf8_to_wide(part).c_str(), utf8_to_wide(target).c_str(),
	                 MOVEFILE_REPLACE_EXISTING)) {
		cleanup();
		DeleteFileW(utf8_to_wide(part).c_str());
		throw MediaError("download failed (rename) for " + entry.file);
	}
	cleanup();
	if (progress) progress(1.0);
}

}  // namespace phi::media
