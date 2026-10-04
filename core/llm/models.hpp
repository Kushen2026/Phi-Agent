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

struct ModelDef {
	std::string provider;
	std::string id;
	std::string name;
	std::string api;         // "anthropic-messages" | "openai-completions"
	std::string base_url;    // api root (no trailing slash)
	bool reasoning = false;
	std::map<std::string, std::optional<std::string>> thinking_level_map;  // level -> mapped or unsupported
	std::vector<std::string> input;  // ["text","image"]
	ModelCost cost;
	int64_t context_window = 0;
	int64_t max_tokens = 0;

	// -- derived --
	bool supports_thinking(const std::string& level) const;
	std::string map_thinking(const std::string& level) const;
	std::vector<std::string> supported_thinking_levels() const;
	bool supports_image() const;
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
