// settings.json + webui-configs.json (profiles), ported from settings-manager
// and config-store.ts.
#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "util/json.hpp"
#include "llm/models.hpp"

namespace phi {

class SettingsStore {
public:
	explicit SettingsStore(const std::string& agent_dir);

	std::optional<std::string> get_string(const std::string& key) const;
	void set_string(const std::string& key, const std::string& value);

	std::optional<std::string> default_provider() const;
	std::optional<std::string> default_model() const;
	std::optional<std::string> default_thinking_level() const;
	std::string system_prompt_language() const;  // "zh" | "en"
	void set_system_prompt_language(const std::string& language);

	void set_defaults(const std::string& provider, const std::string& model, const std::string& thinking);

	// generic json value access (tool settings and future sections)
	JsonValue get_json(const std::string& key) const;  // null when missing
	void set_json(const std::string& key, const JsonValue& value);

	// tool settings (settings.json "tools" object), merged over defaults.
	// Shape:
	// {
	//   "enabled":     { tool: bool },
	//   "permissions": { tool: "everywhere" | "in-cwd" },
	//   "outputLimits": { tool: max chars },
	//   "subagent":    { "tools": [names] (empty = all), "thinking": level }
	// }
	JsonValue tool_settings() const;
	void set_tool_settings(const JsonValue& settings);

	// UI settings (persisted in settings.json for cross-session consistency)
	JsonValue hotkeys() const;      // { action: combo } — empty returns defaults
	void set_hotkeys(const JsonValue& h);
	JsonValue appearance() const;   // { theme, radius, density, motion }
	void set_appearance(const JsonValue& a);
	JsonValue prefs() const;        // { sendKey, statusTokens }
	void set_prefs(const JsonValue& p);

private:
	// settings.json is read lazily and touched from several threads (the HTTP
	// thread serving the UI shell, the websocket reader, the agent worker).
	// Without a lock, a thread entering ensure_loaded() while another was still
	// parsing saw loaded_ == true with an empty doc_ and got defaults instead of
	// the stored values — that read the appearance theme as the dark default and
	// made the first paint flash dark before the UI corrected itself.
	mutable std::recursive_mutex mutex_;
	mutable bool loaded_ = false;
	mutable JsonValue doc_;
	std::string path_;
	void ensure_loaded() const;
	void save() const;
};

// ── web-ui config profiles ─────────────────────────────────────────────────

struct SubagentConfig {
	std::string api_key;
	std::string base_url;
	std::string model;
	std::string api_format;
};

struct StoredProfile {
	std::string name;
	std::string provider;
	std::string model;
	std::string api_key;
	std::string base_url;
	std::string api_format;
	std::optional<SubagentConfig> subagent;
	std::string system_prompt_language;  // empty = unset
};

struct ProfileDraft {
	std::string name;
	std::string provider;
	std::string model;
	std::string api_key;
	std::string base_url;
	std::string api_format;
	std::optional<SubagentConfig> subagent;
	std::string system_prompt_language;
};

class ConfigStore {
public:
	ConfigStore(const std::string& agent_dir, ModelRuntime& runtime, SettingsStore& settings);

	std::string file_path() const;

	struct Listing {
		std::string active;
		std::vector<StoredProfile> profiles;
		std::optional<SubagentConfig> subagent;
		std::string system_prompt_language;
	};
	Listing list() const;

	// throws std::string on validation errors
	void save(const ProfileDraft& draft);
	void remove(const std::string& name);
	// throws std::string on error
	void activate(const std::string& name);
	// apply persisted active profile at startup; returns applied profile or nullopt
	std::optional<StoredProfile> apply_active();
	void save_subagent(const SubagentConfig& config);

	// saved cwd (webui-cwd.json)
	std::optional<std::string> saved_cwd() const;
	void save_cwd(const std::string& cwd);

	static std::string gateway_provider_id(const std::string& profile_name);
	static std::string mask_key(const std::string& key);

private:
	std::string agent_dir_;
	std::string configs_path_;
	ModelRuntime* runtime_;
	SettingsStore* settings_;

	JsonValue load_configs() const;
	void save_configs(const JsonValue& configs);
	void apply_subagent(const StoredProfile& profile);
	void activate_gateway(const StoredProfile& profile);
	void activate_direct(const StoredProfile& profile);
};

// json <-> struct helpers
JsonValue profile_to_json(const StoredProfile& p);
StoredProfile profile_from_json(const JsonValue& j);
JsonValue subagent_to_json(const SubagentConfig& s);
SubagentConfig subagent_from_json(const JsonValue& j);

}  // namespace phi
