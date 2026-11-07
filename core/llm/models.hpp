// Model catalog + user providers (models.json).
#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "util/json.hpp"

namespace phi {

struct ModelCost {
	double input = 0;
	double output = 0;
	double cache_read = 0;
	double cache_write = 0;
};

// ── multimodal capability (API 配置 → 多模态) ─────────────────────────────
//
// What the configured model can actually look at. The setting is per profile
// (webui-configs.json "vision") and turns into ModelDef::input, which is the
// list the providers and the agent loop read. Three tiers, matching what an
// endpoint can do:
//
//   none        text only — a picture or a video cannot be read at all
//   image       text + pictures (the behaviour every profile had before this
//               setting existed, hence the default)
//   image_video pictures + mp4 video, sent whole
enum class VisionMode { None = 0, Image = 1, ImageVideo = 2 };

// "none" | "image" | "image_video". An unknown or empty key reads as Image: a
// profile written before this setting existed was served as an image-capable
// one, and keeping that is what makes old configs behave unchanged.
VisionMode vision_mode_from_key(const std::string& key);
std::string vision_mode_key(VisionMode mode);
// the model.input list a mode implies: {"text"} / {"text","image"} /
// {"text","image","video"} — in exactly this order
std::vector<std::string> vision_mode_inputs(VisionMode mode);

struct ModelDef {
	std::string provider;
	std::string id;
	std::string name;
	std::string api;         // "anthropic-messages" | "openai-completions"
	std::string base_url;    // api root (no trailing slash)
	bool reasoning = false;
	std::map<std::string, std::optional<std::string>> thinking_level_map;  // level -> mapped or unsupported
	std::vector<std::string> input;  // ["text","image"] (+ "video")
	ModelCost cost;
	int64_t context_window = 0;
	int64_t max_tokens = 0;

	// -- derived --
	bool supports_thinking(const std::string& level) const;
	std::string map_thinking(const std::string& level) const;
	std::vector<std::string> supported_thinking_levels() const;
	bool supports_image() const;
	bool supports_video() const;
};


struct ProviderDef {
	std::string id;
	std::string name;
	std::string base_url;
	std::string api;        // wire format
	std::string api_key;    // embedded (gateway) key; may be empty
	std::vector<ModelDef> models;
};

// Levels in canonical order.
extern const char* const THINKING_LEVELS[];
extern const int THINKING_LEVEL_COUNT;

class ModelRuntime {
public:
	// data_dir = agent dir (holds models.json)
	explicit ModelRuntime(const std::string& data_dir);

	// all models: user providers (models.json) + catalog providers, sorted
	std::vector<ModelDef> get_models() const;
	std::optional<ModelDef> get_model(const std::string& provider, const std::string& id) const;
	bool has_configured_auth(const std::string& provider) const;
	// api key resolution: callback -> models.json provider key -> catalog key -> env var
	std::optional<std::string> resolve_api_key(const std::string& provider) const;

	// set an external key provider callback (e.g. from ConfigStore profiles)
	using KeyProvider = std::function<std::optional<std::string>(const std::string&)>;
	void set_key_provider_callback(KeyProvider cb) { key_provider_callback_ = std::move(cb); }

	// raw access for config store
	const std::map<std::string, ProviderDef>& user_providers() const { return user_providers_; }
	void set_user_provider(const std::string& id, const ProviderDef& def);
	void save_models_json() const;

private:
	std::string data_dir_;
	std::map<std::string, ProviderDef> catalog_;        // built-in providers
	std::map<std::string, ProviderDef> user_providers_; // models.json
	KeyProvider key_provider_callback_;                  // external key resolver

	void load_catalog();
	void load_models_json();
};

ModelDef model_def_from_json(const std::string& provider, const JsonValue& j);

}  // namespace phi
