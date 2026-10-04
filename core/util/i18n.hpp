// Language helpers: the session's system-prompt language (settings.json →
// systemPromptLanguage, surfaced as Agent::Config::language) drives every
// built-in prompt string, so tools and the subagent plumbing never hard-code a
// language. English strings are plain, understandable translations of the
// Chinese ones.
#pragma once

#include <string>

namespace phi {

// "en" is the only English locale; everything else falls back to Chinese
inline bool lang_is_zh(const std::string& language) { return language != "en"; }

// pick the Chinese string for zh sessions, the English one otherwise
inline std::string tr(const std::string& language, std::string zh, std::string en) {
	return lang_is_zh(language) ? std::move(zh) : std::move(en);
}

}  // namespace phi
