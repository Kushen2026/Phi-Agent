#include "llm/models.hpp"

#include <algorithm>

#include "util/base.hpp"

namespace phi {

// optional heap-guard scan hook (defined by tests/probe_guard.cpp when linked)
extern "C" void phi_guard_scan(const char*) __attribute__((weak));
#define PHI_GUARD_SCAN(tag) do { if (phi_guard_scan) phi_guard_scan(tag); } while (0)

const char* const THINKING_LEVELS[] = {"off", "minimal", "low", "medium", "high", "xhigh", "max"};
const int THINKING_LEVEL_COUNT = 7;

bool ModelDef::supports_thinking(const std::string& level) const {
	if (level == "off") return true;
	if (!reasoning) return false;
	auto it = thinking_level_map.find(level);
	if (it != thinking_level_map.end()) return it->second.has_value();
	return true;
}

std::string ModelDef::map_thinking(const std::string& level) const {
	if (level == "off" || !reasoning) return "off";
	auto it = thinking_level_map.find(level);
	if (it != thinking_level_map.end() && it->second.has_value()) return *it->second;
	return level;
}

std::vector<std::string> ModelDef::supported_thinking_levels() const {
	if (!reasoning) return {"off"};
	std::vector<std::string> out;
	for (int i = 1; i < THINKING_LEVEL_COUNT; i++) {
		if (supports_thinking(THINKING_LEVELS[i])) out.push_back(THINKING_LEVELS[i]);
	}
	if (out.empty()) out.push_back("medium");
	return out;
}

bool ModelDef::supports_image() const {
	for (const auto& modality : input) {
		if (modality == "image") return true;
	}
	return false;
}

ModelDef model_def_from_json(const std::string& provider, const JsonValue& j) {
	ModelDef m;
	m.provider = j["provider"].as_string(provider);
	m.id = j["id"].as_string("");
	m.name = j["name"].as_string(m.id);
	m.api = j["api"].as_string("");
	m.base_url = j["baseUrl"].as_string("");
	m.reasoning = j["reasoning"].as_bool(false);
	for (const auto& modality : j["input"].items()) {
		if (modality.is_string()) m.input.push_back(modality.as_string());
	}
	if (const JsonValue* map = j.find("thinkingLevelMap")) {
		for (const auto& [level, mapped] : map->entries()) {
			if (mapped.is_null()) {
				m.thinking_level_map[level] = std::nullopt;
			} else if (mapped.is_string()) {
				m.thinking_level_map[level] = mapped.as_string();
			}
		}
	}
	if (const JsonValue* cost = j.find("cost")) {
		m.cost.input = (*cost)["input"].as_number(0);
		m.cost.output = (*cost)["output"].as_number(0);
		m.cost.cache_read = (*cost)["cacheRead"].as_number(0);
		m.cost.cache_write = (*cost)["cacheWrite"].as_number(0);
	}
	m.context_window = j["contextWindow"].as_int(0);
	m.max_tokens = j["maxTokens"].as_int(0);
	return m;
}

static ProviderDef provider_from_json(const std::string& id, const JsonValue& j) {
	ProviderDef p;
	p.id = id;
	p.name = j["name"].as_string(id);
	p.base_url = j["baseUrl"].as_string("");
	p.api = j["api"].as_string("");
	p.api_key = j["apiKey"].as_string("");
	for (const auto& m : j["models"].items()) {
		ModelDef def = model_def_from_json(id, m);
		if (def.api.empty()) def.api = p.api;
		if (def.base_url.empty()) def.base_url = p.base_url;
		def.provider = id;
		// after provider-level fallbacks both must be resolvable: a model entry
		// that cannot produce a usable endpoint is skipped (and reported) rather
		// than silently listed and failing on every call
		if (def.id.empty() || def.api.empty() || def.base_url.empty()) {
			fprintf(stderr, "[models] skipping models.json model %s/%s: missing %s\n",
				id.c_str(), def.id.c_str(),
				def.id.empty() ? "id" : (def.api.empty() ? "api" : "baseUrl"));
			continue;
		}
		p.models.push_back(std::move(def));
	}
	return p;
}

static JsonValue provider_to_json(const ProviderDef& p) {
	JsonValue j = JsonValue::object();
	j["name"] = p.name;
	j["baseUrl"] = p.base_url;
	j["apiKey"] = p.api_key;
	j["api"] = p.api;
	JsonValue models = JsonValue::array();
	for (const auto& m : p.models) {
		JsonValue jm = JsonValue::object();
		jm["id"] = m.id;
		jm["name"] = m.name;
		jm["api"] = m.api;
		jm["provider"] = p.id;
		jm["baseUrl"] = m.base_url;
		jm["reasoning"] = m.reasoning;
		if (!m.thinking_level_map.empty()) {
			JsonValue map = JsonValue::object();
			for (const auto& [level, mapped] : m.thinking_level_map) {
				map[level] = mapped ? JsonValue(*mapped) : JsonValue(nullptr);
			}
			jm["thinkingLevelMap"] = map;
		}
		JsonValue input = JsonValue::array();
		for (const auto& i : m.input) input.push_back(i);
		jm["input"] = input;
		JsonValue cost = JsonValue::object();
		cost["input"] = m.cost.input;
		cost["output"] = m.cost.output;
		cost["cacheRead"] = m.cost.cache_read;
		cost["cacheWrite"] = m.cost.cache_write;
		jm["cost"] = cost;
		jm["contextWindow"] = (int64_t)m.context_window;
		jm["maxTokens"] = (int64_t)m.max_tokens;
		models.push_back(jm);
	}
	j["models"] = models;
	return j;
}

ModelRuntime::ModelRuntime(const std::string& data_dir) : data_dir_(data_dir) {
	load_catalog();
	load_models_json();
}

void ModelRuntime::load_catalog() {
	PHI_GUARD_SCAN("catalog-pre");
	std::string path = path_join(path_dirname(data_dir_), "assets/models-catalog.json");
	auto raw = read_file(path);
	PHI_GUARD_SCAN("catalog-read");
	if (!raw) return;
	auto parsed = json_parse(*raw);
	PHI_GUARD_SCAN("catalog-parse");
	if (!parsed || !parsed->is_object()) return;
	for (const auto& [provider, models] : parsed->entries()) {
		ProviderDef p;
		p.id = provider;
		p.name = provider;
		for (const auto& [model_id, def] : models.entries()) {
			ModelDef m = model_def_from_json(provider, def);
			if (m.id.empty()) m.id = model_id;
			if (m.name.empty()) m.name = m.id;
			// validate required fields: a catalog entry without api/base_url would
			// produce a model that always fails at request time with a confusing
			// transport error — skip it loudly instead of loading it silently
			if (m.api.empty() || m.base_url.empty()) {
				fprintf(stderr, "[models] skipping catalog model %s/%s: missing %s\n",
					provider.c_str(), m.id.c_str(), m.api.empty() ? "api" : "baseUrl");
				continue;
			}
			p.models.push_back(std::move(m));
		}
		if (!p.models.empty()) catalog_[provider] = std::move(p);
	}
}

void ModelRuntime::load_models_json() {
	auto raw = read_file(path_join(data_dir_, "models.json"));
	if (!raw) return;
	auto parsed = json_parse(*raw);
	if (!parsed) return;
	if (const JsonValue* providers = parsed->find("providers")) {
		for (const auto& [id, def] : providers->entries()) {
			user_providers_[id] = provider_from_json(id, def);
		}
	}
}



std::vector<ModelDef> ModelRuntime::get_models() const {
	std::vector<ModelDef> out;
	for (const auto& [id, p] : user_providers_) {
		for (const auto& m : p.models) out.push_back(m);
	}
	for (const auto& [id, p] : catalog_) {
		for (const auto& m : p.models) out.push_back(m);
	}
	std::sort(out.begin(), out.end(), [](const ModelDef& a, const ModelDef& b) {
		if (a.provider != b.provider) return a.provider < b.provider;
		return a.name < b.name;
	});
	return out;
}

std::optional<ModelDef> ModelRuntime::get_model(const std::string& provider, const std::string& id) const {
	auto find_in = [&](const std::map<std::string, ProviderDef>& set) -> std::optional<ModelDef> {
		auto it = set.find(provider);
		if (it == set.end()) return std::nullopt;
		// empty id = "whatever model this provider is configured with": the
		// subagent tool asks for a whole provider (gw-subagent) because the
		// profile stores one model, and no model id is ever the empty string,
		// so an exact match here could never succeed (this was the
		// "未配置子代理模型" false negative: the provider *was* configured)
		if (id.empty()) {
			if (!it->second.models.empty()) return it->second.models.front();
			return std::nullopt;
		}
		for (const auto& m : it->second.models) {
			if (m.id == id) return m;
		}
		return std::nullopt;
	};
	if (auto m = find_in(user_providers_)) return m;
	return find_in(catalog_);
}



static std::string env_key_for_provider(const std::string& provider) {
	std::string upper = to_lower(provider);
	for (auto& c : upper) c = (char)std::toupper((unsigned char)c);
	return upper + "_API_KEY";
}

bool ModelRuntime::has_configured_auth(const std::string& provider) const {
	return resolve_api_key(provider).has_value();
}

std::optional<std::string> ModelRuntime::resolve_api_key(const std::string& provider) const {
	// external key provider (e.g. ConfigStore profiles)
	if (key_provider_callback_) {
		if (auto key = key_provider_callback_(provider)) return key;
	}
	auto up = user_providers_.find(provider);
	if (up != user_providers_.end() && !up->second.api_key.empty()) return up->second.api_key;
	auto cat = catalog_.find(provider);
	if (cat != catalog_.end() && !cat->second.api_key.empty()) return cat->second.api_key;
	return get_env(env_key_for_provider(provider));
}

void ModelRuntime::set_user_provider(const std::string& id, const ProviderDef& def) {
	user_providers_[id] = def;
}

void ModelRuntime::save_models_json() const {
	JsonValue providers = JsonValue::object();
	for (const auto& [id, p] : user_providers_) {
		providers[id] = provider_to_json(p);
	}
	JsonValue root = JsonValue::object();
	root["providers"] = providers;
	mkdirs(data_dir_);
	std::string path = path_join(data_dir_, "models.json");
	// guard: an unparseable on-disk file would otherwise be silently replaced
	// by the (empty) in-memory provider set. Keep the original for recovery.
	if (auto raw = read_file(path)) {
		auto parsed = json_parse(*raw);
		if (!parsed || !parsed->is_object()) {
			write_file_atomic(path + ".bak", *raw);
		}
	}
	write_file_atomic(path, root.dump(2) + "\n");
}

}  // namespace phi
