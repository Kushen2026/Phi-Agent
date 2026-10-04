// Web tools: web_search (Bing HTML) and fetch (URL → readable text).
// Bing is the search backend: it serves plain HTML without cookies or keys and
// is reachable from mainland China (duckduckgo is blocked there, which used to
// hang/fail web_search entirely). Page extraction is ported from pi-web-access,
// on top of the raw-socket http client.
#pragma once

#include <string>

#include "agent/agent.hpp"

namespace phi {

ToolDef web_search_tool();
ToolDef fetch_tool();

// html → plain text (script/style stripped, block tags → newlines, entities
// decoded); exposed for tests
std::string html_to_text(const std::string& html);
std::string html_extract_title(const std::string& html);
std::string url_percent_encode(const std::string& text);
// parse a bing results page into a numbered "1. title\n url\n snippet" list;
// returns "" when nothing matches (exposed for offline tests)
std::string web_parse_bing_results(const std::string& html, int max_results);

}  // namespace phi
