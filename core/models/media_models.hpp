// Media model registry: what the two media tools load, which file each role
// resolves to, and how a missing file is fetched.
//
// The engine reads these roles per chain:
//
//   image:  dit (Qwen-Image-2.1) | text encoder (Qwen3-VL-8B) | vae |
//           tokenizer files (vocab.json + merges.txt + tokenizer_config.json)
//   video:  dit (MiniMax H3)     | text encoder (Qwen3-VL-32B) | video vae |
//           audio vae + vocoder | tokenizer files
//   music:  dit (ACE-Step 1.5)   | embedder (Qwen3-0.6B) + audio-code LM
//           (Qwen3-4B) | audio vae | tokenizer files
//   tts:    model (Breeze-TTS-2) | codec (Qwen3-TTS 12 Hz) | tokenizer.json |
//           config.json + codec config.json
//
// ── one config per model, never one config for several ─────────────────────
//
// Every chain owns its own selection, down to the tokenizer *file*: each chain
// has its own `vocab.json`, `merges.txt` and `tokenizer_config.json` dropdowns
// (nine roles in all), each defaulting to a file inside that chain's model
// directory. The role that used to be shared - one `tokenizer_config` feeding the
// image, video and music encoders at once - is gone.
//
// The three files are separate roles rather than "a directory" because they are
// genuinely separate resources: the loaders take three independent paths, and the
// settings panel offers every file under models/, so naming the directory behind
// the user's back would hide which vocabulary a chain is really reading. The three
// chains happen to read byte-identical Qwen2.5 vocabularies today (ComfyUI ships
// the same `qwen25_tokenizer` for all of them), so the *content* repeats - but the
// *selection* does not, which is what makes a chain self-contained.
//
// The Breeze chain follows the same rule: the model's `config.json` and the
// codec's `config.json` are separate roles with separate files (they are
// different formats - the model config's scalars sit at the top level, the
// codec's under `decoder_config`; see core/models/mimi_codec.cpp), and the
// Breeze `tokenizer.json` is the chain's own.
//
// Plus up to two LoRA files per chain, applied in order. Historically each
// `open()` had the filename baked in; now the *selection* is a user setting
// (settings.json `tools.media`) and this module is the one place that turns it
// into paths.
//
// The catalogue is the filesystem: the settings UI offers every file under the
// models directory, so a user selects a checkpoint by dropping it in `models/`
// and picking its name. There is no built-in list to keep in sync, and no
// format flag to get wrong - the loaders detect the precision themselves (see
// host/st.hpp).
//
// The image and video chains each carry their own LoRA list: a LoRA trained for
// one network must never be applied to the other, so they are resolved and
// stored separately rather than as one shared list.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "util/media_common.hpp"
#include "util/json.hpp"

namespace phi::media {

// One file the engine loads, by role.
struct MediaModelSpec {
	// image chain
	std::string image_dit;
	std::string image_te;
	std::string image_vae;
	// video chain
	std::string video_dit;
	std::string video_te;
	std::string video_vae;
	std::string video_avae;      // H3 audio VAE + BigVGAN (one checkpoint)
	// music chain (ACE-Step 1.5)
	std::string music_dit;       // the DiT bundle (decoder + condition encoder + tokenizer)
	std::string music_te;        // Qwen3-0.6B embedding tower
	std::string music_lm;        // Qwen3-4B audio-code LM
	std::string music_vae;       // ACE-Step 1.5 audio VAE
	// tts chain (Breeze-TTS-2): the whole model in one checkpoint, plus the
	// codec (a separate file) and the tokenizer.json that sits beside them.
	std::string tts_model;
	std::string tts_codec;
	std::string tts_tokenizer;
	// The TTS config.json (beside the weights) and the codec's own config.json.
	// Empty means "read the sibling config.json", i.e. the pre-role behaviour.
	// They are deliberately *not* the same file: the model's config carries the
	// backbone / text-encoder scalars at the top level, the codec's carries its own
	// under `decoder_config`, and the two used to be conflated because the codec
	// checkpoint sits in the model's directory.
	std::string tts_config;
	std::string tts_codec_config;
	// ── per-chain tokenizer files ──────────────────────────────────────────
	// Three independent selections per chain, because the tokenizer needs three
	// files and they do *not* have to sit together: `vocab.json` is the token ->
	// id table, `merges.txt` the BPE merge ranks, `tokenizer_config.json` the
	// special-token ids. Each has its own dropdown and its own default, and none
	// is derived from another.
	std::string image_tokenizer_vocab;
	std::string image_tokenizer_merges;
	std::string image_tokenizer_config;
	std::string video_tokenizer_vocab;
	std::string video_tokenizer_merges;
	std::string video_tokenizer_config;
	std::string music_tokenizer_vocab;
	std::string music_tokenizer_merges;
	std::string music_tokenizer_config;
	// LoRA chains, in application order (0..1 each). Kept apart so an image LoRA
	// never leaks into the video DiT (or vice versa).
	std::vector<std::string> image_loras;
	std::vector<std::string> video_loras;

	bool empty() const {
		return image_dit.empty() && video_dit.empty() && music_dit.empty() && tts_model.empty() &&
		       image_tokenizer_vocab.empty() && video_tokenizer_vocab.empty() &&
		       music_tokenizer_vocab.empty() && tts_config.empty() && tts_codec_config.empty();
	}
};

// A model the user can pick in the settings UI.
struct MediaModelEntry {
	std::string id;         // stable id ("qwen_image_2.1_int8_convrot")
	std::string name;       // display name
	std::string role;       // "image_dit" | "image_te" | "image_vae" | "video_dit" |
		                        // "video_te" | "video_vae" | "video_avae" |
		                        // "music_dit" | "music_te" | "music_lm" | "music_vae" |
		                        // "tts_model" | "tts_codec" | "tts_tokenizer" |
		                        // "image_tokenizer_config" | "video_tokenizer_config" |
		                        // "music_tokenizer_config" | "tts_config" | "tts_codec_config"
	std::string format;     // "safetensors" | "json"
	std::string subdir;     // models/<subdir>/<file>
	std::string file;       // target filename
	std::string url;        // download URL ("" = place it by hand)
	std::string sha256;     // integrity check; "" = skip
	u64 size = 0;           // bytes (0 = unknown)
};

// The built-in catalogue: the shipped checkpoints.
std::vector<MediaModelEntry> media_model_catalog();

// The catalogue filtered to one role, for the settings UI.
std::vector<MediaModelEntry> media_model_catalog_for(const std::string& role);

// Finds a catalogue entry by id (nullptr when unknown).
const MediaModelEntry* media_model_entry(const std::string& id);

// Every regular file under `models_dir`, recursively, as paths relative to it
// and sorted. This is the settings UI's option list for every role: the
// filesystem *is* the catalogue, so a hand-placed checkpoint is selectable the
// moment it appears (the UI re-reads this list each time the panel opens).
std::vector<std::string> list_media_files(const std::string& models_dir);

// Resolves a user-entered model name to an existing file under `models_dir`.
// Accepts a full path, a path relative to `models_dir`, or a bare filename
// (matched anywhere under `models_dir`). Returns "" when nothing matches.
std::string resolve_media_file(const std::string& models_dir, const std::string& name);

// Resolves a role to a concrete path inside `models_dir`:
//   * the settings' selection when it names a file that is present;
//   * otherwise the shipped default filename, looked up anywhere under
//     `models_dir` (so a hand-placed file still works, whatever its subfolder);
// Returns a best-guess path when nothing exists, so the caller's error names a
// file the user can act on.
std::string resolve_media_role(const JsonValue& media_settings, const std::string& role,
                               const std::string& models_dir);

// Builds the full spec from settings.json's `tools.media` object and the models
// directory (which the caller located with find_models_dir).
MediaModelSpec resolve_media_spec(const JsonValue& media_settings, const std::string& models_dir);

// The default (shipped) id for a role, "" for an unknown role.
std::string default_media_model_id(const std::string& role);

// True when the file for `entry` already exists under `models_dir`.
bool media_model_present(const MediaModelEntry& entry, const std::string& models_dir);

// Downloads an entry into models_dir/<subdir>/<file> (creating directories),
// calling `progress(0..1)` as it goes (progress may be null; a negative value
// means "unknown size"). Honours `cancel` (polled between blocks) and throws
// MediaError on any failure, leaving no partial file behind. Blocks the caller.
void media_model_download(const MediaModelEntry& entry, const std::string& models_dir,
                          const std::function<void(double)>& progress,
                          const std::function<bool()>& cancel);

}  // namespace phi::media
