#include "store/stores.hpp"

#include <functional>

#include "util/base.hpp"

namespace phi {

// ── settings.json ──────────────────────────────────────────────────────────

SettingsStore::SettingsStore(const std::string& agent_dir) : path_(path_join(agent_dir, "settings.json")) {}

void SettingsStore::ensure_loaded() const {
	std::lock_guard<std::recursive_mutex> lock(mutex_);
	if (loaded_) return;
	loaded_ = true;
	doc_ = JsonValue::object();
	auto raw = read_file(path_);
	if (!raw) return;  // first run: no file yet, defaults are correct
	auto parsed = json_parse(*raw);
	if (parsed && parsed->is_object()) {
		doc_ = *parsed;
		return;
	}
	// file exists but is not a valid JSON object: keep a copy so the next
	// save() cannot silently overwrite it with an empty document
	write_file_atomic(path_ + ".bak", *raw);
	fprintf(stderr, "[settings] %s is not a valid JSON object; backup written to %s.bak\n",
		path_.c_str(), path_.c_str());
}

void SettingsStore::save() const {
	std::lock_guard<std::recursive_mutex> lock(mutex_);
	mkdirs(path_dirname(path_));
	write_file_atomic(path_, doc_.dump(2) + "\n");
}

std::optional<std::string> SettingsStore::get_string(const std::string& key) const {
	std::lock_guard<std::recursive_mutex> lock(mutex_);
	ensure_loaded();
	const JsonValue* v = doc_.find(key);
	if (!v || !v->is_string()) return std::nullopt;
	return v->as_string();
}

void SettingsStore::set_string(const std::string& key, const std::string& value) {
	std::lock_guard<std::recursive_mutex> lock(mutex_);
	ensure_loaded();
	doc_[key] = value;
	save();
}

std::optional<std::string> SettingsStore::default_provider() const {
	return get_string("defaultProvider");
}
std::optional<std::string> SettingsStore::default_model() const {
	return get_string("defaultModel");
}
std::optional<std::string> SettingsStore::default_thinking_level() const {
	return get_string("defaultThinkingLevel");
}

std::string SettingsStore::system_prompt_language() const {
	auto lang = get_string("systemPromptLanguage");
	return lang && *lang == "en" ? "en" : "zh";
}

void SettingsStore::set_system_prompt_language(const std::string& language) {
	set_string("systemPromptLanguage", language == "en" ? "en" : "zh");
}

void SettingsStore::set_defaults(const std::string& provider, const std::string& model,
	const std::string& thinking) {
	std::lock_guard<std::recursive_mutex> lock(mutex_);
	ensure_loaded();
	doc_["defaultProvider"] = provider;
	doc_["defaultModel"] = model;
	doc_["defaultThinkingLevel"] = thinking;
	save();
}

JsonValue SettingsStore::get_json(const std::string& key) const {
	std::lock_guard<std::recursive_mutex> lock(mutex_);
	ensure_loaded();
	const JsonValue* v = doc_.find(key);
	return v ? *v : JsonValue();
}

void SettingsStore::set_json(const std::string& key, const JsonValue& value) {
	std::lock_guard<std::recursive_mutex> lock(mutex_);
	ensure_loaded();
	doc_[key] = value;
	save();
}

JsonValue SettingsStore::tool_settings() const {
	std::lock_guard<std::recursive_mutex> lock(mutex_);
	ensure_loaded();
	JsonValue defaults = JsonValue::object();
	// every built-in tool defaults to enabled + unrestricted + default limit
	for (const char* name : {"bash", "read", "write", "edit", "ls", "grep", "web_search", "fetch",
	                         "image_generate", "video_generate", "music_generate", "tts_speak",
	                         "agent"}) {
		defaults["enabled"][name] = true;
		defaults["permissions"][name] = "everywhere";
	}
	defaults["outputLimits"]["bash"] = 120000;
	defaults["outputLimits"]["read"] = 120000;
	defaults["outputLimits"]["ls"] = 60000;
	defaults["outputLimits"]["grep"] = 60000;
	defaults["subagent"] = JsonValue::object();
	defaults["subagent"]["tools"] = JsonValue::array();  // empty = all parent tools
	// Media model selection (image_generate / video_generate / music_generate /
	// tts_speak). The shape is: {
	//   "image_dit": <id>, "image_te": ..., "image_vae": ...,
	//   "video_dit": ..., "video_te": ..., "video_vae": ..., "video_avae": ...,
	//   "music_dit": ..., "music_te": ..., "music_lm": ..., "music_vae": ...,
	//   "tts_model": ..., "tts_codec": ..., "tts_tokenizer": ...,
	//   "image_steps": <1..100>, "video_steps": <1..100>, "music_steps": <1..100>,
	//   "image_sampler"/"image_scheduler": ComfyUI names (euler + simple),
	//   "video_sampler"/"video_scheduler": res_multistep + beta,
	//   "music_sampler"/"music_scheduler": euler + simple,
	//   "loras": [<id|path>, ...]   // applied in order, at most kMaxLoras
	// }
	// An empty section keeps the shipped checkpoints (media_models.cpp resolves the
	// defaults).
	defaults["media"] = JsonValue::object();
	defaults["media"]["loras"] = JsonValue::array();
	// 采样步数：每次请求的参数（不需要重建引擎），默认就是两条链各自的默认步数。
	// 缺这个键时工具会回落到 kImageSteps / kVideoSteps（tools_media.cpp）。
	defaults["media"]["image_steps"] = 25;   // qwen image 2.1
	defaults["media"]["video_steps"] = 20;   // minimax h3
	defaults["media"]["music_steps"] = 50;   // ace-step 1.5
	// 采样器 / 调度器：每条链的默认值就是它发布的 workflow 里的那一对。设置面板里
	// 的选择随时可改；缺这个键时工具回落到同一个默认（tools_media.cpp）。
	defaults["media"]["image_sampler"] = "euler";
	defaults["media"]["image_scheduler"] = "simple";
	defaults["media"]["video_sampler"] = "res_multistep";
	defaults["media"]["video_scheduler"] = "beta";
	defaults["media"]["music_sampler"] = "euler";
	defaults["media"]["music_scheduler"] = "simple";

	const JsonValue* stored = doc_.find("tools");
	if (!stored || !stored->is_object()) return defaults;
	// shallow-merge per section so unknown/missing keys keep their defaults
	auto merge_section = [&](const char* section) {
		const JsonValue* s = stored->find(section);
		if (s && s->is_object()) {
			for (const auto& [k, v] : s->entries()) defaults[section][k] = v;
		}
	};
	merge_section("enabled");
	merge_section("permissions");
	merge_section("outputLimits");
	merge_section("media");
	const JsonValue* sub = stored->find("subagent");
	if (sub && sub->is_object()) {
		for (const auto& [k, v] : sub->entries()) defaults["subagent"][k] = v;
	}
	return defaults;
}

void SettingsStore::set_tool_settings(const JsonValue& settings) {
	set_json("tools", settings);
}

// ── UI settings ────────────────────────────────────────────────────────────

// The three accessors below read doc_ directly, so they must load it first:
// the very first caller is the HTTP thread serving the UI shell, and without an
// explicit load it saw an empty document and fell back to the dark defaults —
// which is why a light theme used to be preceded by a dark first paint.
JsonValue SettingsStore::hotkeys() const {
	std::lock_guard<std::recursive_mutex> lock(mutex_);
	ensure_loaded();
	const JsonValue* v = doc_.find("hotkeys");
	if (!v || !v->is_object()) return JsonValue::object();
	return *v;
}
void SettingsStore::set_hotkeys(const JsonValue& h) {
	set_json("hotkeys", h);
}

JsonValue SettingsStore::appearance() const {
	std::lock_guard<std::recursive_mutex> lock(mutex_);
	ensure_loaded();
	const JsonValue* v = doc_.find("appearance");
	if (!v || !v->is_object()) return JsonValue::object();
	return *v;
}
void SettingsStore::set_appearance(const JsonValue& a) {
	set_json("appearance", a);
}

JsonValue SettingsStore::prefs() const {
	std::lock_guard<std::recursive_mutex> lock(mutex_);
	ensure_loaded();
	const JsonValue* v = doc_.find("prefs");
	if (!v || !v->is_object()) return JsonValue::object();
	return *v;
}
void SettingsStore::set_prefs(const JsonValue& p) {
	set_json("prefs", p);
}

// ── webui-configs.json ─────────────────────────────────────────────────────

JsonValue subagent_to_json(const SubagentConfig& s) {
	JsonValue j = JsonValue::object();
	j["apiKey"] = s.api_key;
	j["baseUrl"] = s.base_url;
	j["model"] = s.model;
	j["apiFormat"] = s.api_format;
	return j;
}

SubagentConfig subagent_from_json(const JsonValue& j) {
	SubagentConfig s;
	s.api_key = j["apiKey"].as_string("");
	s.base_url = j["baseUrl"].as_string("");
	s.model = j["model"].as_string("");
	s.api_format = j["apiFormat"].as_string("anthropic-messages");
	return s;
}

JsonValue profile_to_json(const StoredProfile& p) {
	JsonValue j = JsonValue::object();
	j["name"] = p.name;
	j["provider"] = p.provider;
	j["model"] = p.model;
	if (!p.api_key.empty()) j["apiKey"] = p.api_key;
	if (!p.base_url.empty()) j["baseUrl"] = p.base_url;
	if (!p.api_format.empty()) j["apiFormat"] = p.api_format;
	if (p.subagent) j["subagent"] = subagent_to_json(*p.subagent);
	if (!p.system_prompt_language.empty()) j["systemPromptLanguage"] = p.system_prompt_language;
	return j;
}

StoredProfile profile_from_json(const JsonValue& j) {
	StoredProfile p;
	p.name = j["name"].as_string("");
	p.provider = j["provider"].as_string("anthropic");
	p.model = j["model"].as_string("");
	p.api_key = j["apiKey"].as_string("");
	p.base_url = j["baseUrl"].as_string("");
	p.api_format = j["apiFormat"].as_string("anthropic-messages");
	if (const JsonValue* sub = j.find("subagent")) {
		if (!sub->is_null()) p.subagent = subagent_from_json(*sub);
	}
	p.system_prompt_language = j["systemPromptLanguage"].as_string("");
	return p;
}

ConfigStore::ConfigStore(const std::string& agent_dir, ModelRuntime& runtime, SettingsStore& settings)
	: agent_dir_(agent_dir), configs_path_(path_join(agent_dir, "webui-configs.json")), runtime_(&runtime),
	  settings_(&settings) {}

std::string ConfigStore::file_path() const { return configs_path_; }

JsonValue ConfigStore::load_configs() const {
	JsonValue configs = JsonValue::object();
	auto raw = read_file(configs_path_);
	if (raw) {
		auto parsed = json_parse(*raw);
		if (parsed && parsed->is_object()) configs = *parsed;
	}
	return configs;
}

void ConfigStore::save_configs(const JsonValue& configs) {
	mkdirs(path_dirname(configs_path_));
	// guard: if the on-disk file exists but is not a parseable JSON object,
	// the caller almost certainly built `configs` from an empty document
	// (load_configs() returns {} on parse failure). Keep a copy of the
	// broken-but-possibly-recoverable file before overwriting it.
	if (auto raw = read_file(configs_path_)) {
		auto parsed = json_parse(*raw);
		if (!parsed || !parsed->is_object()) {
			write_file_atomic(configs_path_ + ".bak", *raw);
		}
	}
	write_file_atomic(configs_path_, configs.dump(2) + "\n");
}

ConfigStore::Listing ConfigStore::list() const {
	JsonValue configs = load_configs();
	Listing out;
	out.active = configs["active"].as_string("");
	for (const auto& p : configs["profiles"].items()) {
		StoredProfile stored = profile_from_json(p);
		out.profiles.push_back(std::move(stored));
	}
	bool active_valid = false;
	for (const auto& p : out.profiles) {
		if (p.name == out.active) active_valid = true;
	}
	if (!active_valid) out.active = "";
	const StoredProfile* active_stored = nullptr;
	for (const auto& p : out.profiles) {
		if (p.name == out.active) active_stored = &p;
	}
	if (active_stored && active_stored->subagent) {
		out.subagent = active_stored->subagent;
	} else if (const JsonValue* sub = configs.find("subagent")) {
		if (!sub->is_null()) out.subagent = subagent_from_json(*sub);
	}
	out.system_prompt_language = settings_->system_prompt_language();
	return out;
}

std::string ConfigStore::gateway_provider_id(const std::string& profile_name) {
	std::string slug;
	for (char c : to_lower(profile_name)) {
		if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) slug.push_back(c);
		else slug.push_back('-');
	}
	while (!slug.empty() && slug.front() == '-') slug.erase(slug.begin());
	while (!slug.empty() && slug.back() == '-') slug.pop_back();
	if (slug.size() > 40) slug = slug.substr(0, 40);
	// non-ASCII names slug to empty (every char maps to '-') — they would ALL
	// collapse to "gw-gateway" and activating one would re-point another
	// profile's provider. Add a stable hash suffix when the slug is
	// non-ASCII-only or short, and always for uniqueness beyond 40 chars.
	bool ascii = true;
	for (unsigned char c : profile_name) {
		if (c >= 0x80) ascii = false;
	}
	if (slug.empty() || !ascii || slug.size() >= 40) {
		slug += (slug.empty() ? "" : "-") +
			std::to_string((unsigned long long)std::hash<std::string>{}(profile_name) % 1000000ull);
	}
	return "gw-" + (slug.empty() ? "gateway" : slug);
}

std::string ConfigStore::mask_key(const std::string& key) {
	if (key.empty()) return "";
	if (key.size() <= 10) return "••••";
	return key.substr(0, 7) + "…" + key.substr(key.size() - 4);
}

void ConfigStore::save(const ProfileDraft& draft) {
	std::string name = trim(draft.name);
	if (name.empty()) throw std::string("配置名称不能为空");
	std::string base_url = trim(draft.base_url);
	if (base_url.empty()) throw std::string("请求地址不能为空");
	std::string model = trim(draft.model);
	if (model.empty()) throw std::string("模型不能为空");

	// load the config file ONCE and derive everything from that single snapshot:
	// the old code listed() the file, then re-loaded it before writing, so an
	// external change landing between the two reads was silently overwritten
	// by a profile list rebuilt from the stale first load
	JsonValue configs = load_configs();
	Listing listing;
	listing.active = configs["active"].as_string("");
	for (const auto& p : configs["profiles"].items()) {
		listing.profiles.push_back(profile_from_json(p));
	}
	bool active_valid = false;
	for (const auto& p : listing.profiles) {
		if (p.name == listing.active) active_valid = true;
	}
	if (!active_valid) listing.active = "";

	StoredProfile updated;
	updated.name = name;
	updated.provider = draft.provider.empty() ? "anthropic" : draft.provider;
	updated.model = model;
	updated.api_key = trim(draft.api_key).empty() ? "" : trim(draft.api_key);
	updated.base_url = base_url;
	updated.api_format = trim(draft.api_format).empty() ? "anthropic-messages" : trim(draft.api_format);
	updated.subagent = draft.subagent;
	updated.system_prompt_language = draft.system_prompt_language;

	bool replaced = false;
	for (auto& existing : listing.profiles) {
		if (existing.name == name) {
			if (updated.api_key.empty()) updated.api_key = existing.api_key;
			// 表单预填的掩码串（如 sk-abc…wxyz）未改动时，保留真实密钥
			else if (updated.api_key == mask_key(existing.api_key)) updated.api_key = existing.api_key;
			if (!updated.subagent && existing.subagent) updated.subagent = existing.subagent;
			if (updated.subagent && existing.subagent && !updated.subagent->api_key.empty() &&
			    updated.subagent->api_key == mask_key(existing.subagent->api_key)) {
				updated.subagent->api_key = existing.subagent->api_key;
			}
			if (updated.system_prompt_language.empty()) {
				updated.system_prompt_language = existing.system_prompt_language;
			}
			existing = updated;
			replaced = true;
			break;
		}
	}
	if (!replaced) listing.profiles.push_back(updated);

	// mutate the SAME loaded object: rebuilding from scratch dropped the
	// top-level "subagent" entry written by save_subagent() (the gw-subagent
	// gateway silently vanished after any profile save/remove)
	configs["active"] = listing.active.empty() ? JsonValue(nullptr) : JsonValue(listing.active);
	JsonValue profiles = JsonValue::array();
	for (const auto& p : listing.profiles) profiles.push_back(profile_to_json(p));
	configs["profiles"] = profiles;
	save_configs(configs);

	// active profile: subagent gateway / language take effect immediately
	if (listing.active == name) {
		apply_subagent(updated);
		if (!updated.system_prompt_language.empty()) {
			settings_->set_system_prompt_language(updated.system_prompt_language);
		}
	}
}

void ConfigStore::remove(const std::string& name) {
	Listing listing = list();
	std::string active = listing.active == name ? "" : listing.active;
	// load + mutate: same subagent-entry preservation as save()
	JsonValue configs = load_configs();
	configs["active"] = active.empty() ? JsonValue(nullptr) : JsonValue(active);
	JsonValue profiles = JsonValue::array();
	for (const auto& p : listing.profiles) {
		if (p.name != name) profiles.push_back(profile_to_json(p));
	}
	configs["profiles"] = profiles;
	save_configs(configs);
}

void ConfigStore::activate_gateway(const StoredProfile& profile) {
	std::string provider_id = gateway_provider_id(profile.name);
	ProviderDef provider;
	provider.id = provider_id;
	provider.name = profile.name;
	provider.base_url = profile.base_url;
	provider.api = profile.api_format.empty() ? "anthropic-messages" : profile.api_format;
	provider.api_key = profile.api_key;  // store the profile's key for direct lookup

	ModelDef model;
	model.provider = provider_id;
	model.id = profile.model;
	model.name = profile.model;
	model.api = provider.api;
	model.base_url = profile.base_url;
	model.reasoning = true;
	model.input = {"text", "image"};
	model.cost = {};
	model.context_window = 1000000;
	model.max_tokens = 64000;
	if (provider.api == "anthropic-messages") {
		// gateway thinking map: off unsupported, xhigh/max clamp to high
		model.thinking_level_map["off"] = std::nullopt;
		model.thinking_level_map["minimal"] = std::string("low");
		model.thinking_level_map["low"] = std::string("low");
		model.thinking_level_map["medium"] = std::string("medium");
		model.thinking_level_map["high"] = std::string("high");
		model.thinking_level_map["xhigh"] = std::string("high");
		model.thinking_level_map["max"] = std::string("high");
	}
	provider.models.push_back(std::move(model));
	runtime_->set_user_provider(provider_id, provider);
	runtime_->save_models_json();

	// 保留用户已保存的思考强度默认值，不因切换配置而清空
	settings_->set_defaults(provider_id, profile.model, settings_->default_thinking_level().value_or(""));
}

void ConfigStore::activate_direct(const StoredProfile& profile) {
	// api_key is stored in config.json — resolved via key_provider_callback at runtime
	settings_->set_defaults(profile.provider, profile.model, settings_->default_thinking_level().value_or(""));
}

void ConfigStore::activate(const std::string& name) {
	Listing listing = list();
	const StoredProfile* profile = nullptr;
	for (const auto& p : listing.profiles) {
		if (p.name == name) profile = &p;
	}
	if (!profile) throw std::string("配置不存在: ") + name;
	if (profile->api_key.empty()) throw "配置「" + name + "」没有保存 API 密钥";

	if (!profile->base_url.empty()) {
		activate_gateway(*profile);
	} else {
		activate_direct(*profile);
	}
	apply_subagent(*profile);
	if (!profile->system_prompt_language.empty()) {
		settings_->set_system_prompt_language(profile->system_prompt_language);
	}

	// persist active
	JsonValue configs = load_configs();
	configs["active"] = name;
	JsonValue profiles = JsonValue::array();
	for (const auto& p : listing.profiles) profiles.push_back(profile_to_json(p));
	configs["profiles"] = profiles;
	save_configs(configs);
}

std::optional<StoredProfile> ConfigStore::apply_active() {
	JsonValue configs = load_configs();
	std::string active = configs["active"].as_string("");
	if (active.empty()) return std::nullopt;
	for (const auto& p : configs["profiles"].items()) {
		StoredProfile stored = profile_from_json(p);
		if (stored.name == active) {
			try {
				activate(active);
			} catch (...) {
				return std::nullopt;
			}
			return stored;
		}
	}
	return std::nullopt;
}

void ConfigStore::apply_subagent(const StoredProfile& profile) {
	if (!profile.subagent || profile.subagent->base_url.empty() || profile.subagent->model.empty()) return;
	const SubagentConfig& sub = *profile.subagent;

	// build the desired provider def
	ProviderDef provider;
	provider.id = "gw-subagent";
	provider.name = "SubAgent";
	provider.base_url = sub.base_url;
	provider.api = sub.api_format.empty() ? "anthropic-messages" : sub.api_format;
	provider.api_key = sub.api_key;

	ModelDef model;
	model.provider = "gw-subagent";
	model.id = sub.model;
	model.name = sub.model;
	model.api = provider.api;
	model.base_url = sub.base_url;
	model.reasoning = true;
	model.input = {"text", "image"};
	model.cost = {};
	model.context_window = 1000000;
	model.max_tokens = 64000;
	if (provider.api == "anthropic-messages") {
		model.thinking_level_map["off"] = std::nullopt;
		model.thinking_level_map["minimal"] = std::string("low");
		model.thinking_level_map["low"] = std::string("low");
		model.thinking_level_map["medium"] = std::string("medium");
		model.thinking_level_map["high"] = std::string("high");
		model.thinking_level_map["xhigh"] = std::string("high");
		model.thinking_level_map["max"] = std::string("high");
	}
	provider.models.push_back(std::move(model));

	// skip the registration + disk write when nothing changed: apply_subagent
	// runs on every profile save/activate and rewriting models.json each time
	// is pure churn
	const auto& current = runtime_->user_providers();
	auto it = current.find("gw-subagent");
	if (it != current.end() && it->second.id == provider.id && it->second.name == provider.name &&
		it->second.base_url == provider.base_url && it->second.api == provider.api &&
		it->second.api_key == provider.api_key && it->second.models.size() == 1 &&
		it->second.models[0].id == provider.models[0].id) {
		return;
	}
	runtime_->set_user_provider("gw-subagent", provider);
	runtime_->save_models_json();
}

void ConfigStore::save_subagent(const SubagentConfig& config) {
	JsonValue configs = load_configs();
	configs["subagent"] = subagent_to_json(config);
	save_configs(configs);
	apply_subagent(StoredProfile{.subagent = config});
}

std::optional<std::string> ConfigStore::saved_cwd() const {
	auto raw = read_file(path_join(agent_dir_, "webui-cwd.json"));
	if (!raw) return std::nullopt;
	auto parsed = json_parse(*raw);
	if (!parsed) return std::nullopt;
	std::string cwd = (*parsed)["cwd"].as_string("");
	if (cwd.empty()) return std::nullopt;
	return cwd;
}

void ConfigStore::save_cwd(const std::string& cwd) {
	JsonValue j = JsonValue::object();
	j["cwd"] = cwd;
	mkdirs(agent_dir_);
	write_file_atomic(path_join(agent_dir_, "webui-cwd.json"), j.dump(2) + "\n");
}

}  // namespace phi
