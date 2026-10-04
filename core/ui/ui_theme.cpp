// See ui_theme.hpp. Deliberately dependency-light: a tiny scan of the settings
// JSON plus a lookup in styles.css, both of which are small files read once per
// launch (before the window is created).
#include "ui/ui_theme.hpp"

#include "util/base.hpp"
#include "util/json.hpp"

namespace phi {
namespace {

struct CssBlock {
	size_t begin = 0;  // index of '{'
	size_t end = 0;    // index of '}'
	bool valid = false;
};

// finds the declaration block of `body[data-theme="<theme>"] { ... }`.
// Themes can be listed in shared selector groups (scrollbar colouring), so only
// blocks that actually declare --bg count.
CssBlock find_theme_block(const std::string& css, const std::string& theme) {
	CssBlock out;
	if (theme.empty()) return out;
	const std::string needle = "body[data-theme=\"" + theme + "\"]";
	size_t pos = 0;
	while ((pos = css.find(needle, pos)) != std::string::npos) {
		size_t brace = css.find('{', pos + needle.size());
		if (brace == std::string::npos) return out;
		size_t close = css.find('}', brace);
		if (close == std::string::npos) return out;
		// a declaration block (not a pseudo-element chain like ::-webkit-scrollbar)
		if (css.compare(pos + needle.size(), 2, "::") != 0 &&
			css.find("--bg:", brace) < close) {
			out.begin = brace;
			out.end = close;
			out.valid = true;
			return out;
		}
		pos += needle.size();
	}
	return out;
}

bool parse_hex_color(const std::string& text, int& r, int& g, int& b) {
	size_t hash = text.find('#');
	if (hash == std::string::npos) return false;
	unsigned value = 0;
	int digits = 0;
	for (size_t i = hash + 1; i < text.size() && digits < 6; ++i) {
		char c = text[i];
		int d;
		if (c >= '0' && c <= '9') d = c - '0';
		else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
		else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
		else break;
		value = (value << 4) | (unsigned)d;
		++digits;
	}
	if (digits != 6) return false;
	r = (int)((value >> 16) & 0xFF);
	g = (int)((value >> 8) & 0xFF);
	b = (int)(value & 0xFF);
	return true;
}

std::string read_theme_id(const std::string& base_dir) {
	auto settings = read_file(path_join(base_dir, "data/settings.json"));
	if (!settings) return "";
	auto parsed = json_parse(*settings);
	if (!parsed || !parsed->is_object()) return "";
	return trim((*parsed)["appearance"]["theme"].as_string(""));
}

}  // namespace

UiCanvasTheme resolve_ui_canvas_theme(const std::string& base_dir) {
	return css_canvas_theme(base_dir, read_theme_id(base_dir));
}

UiCanvasTheme css_canvas_theme(const std::string& base_dir, const std::string& theme_id) {
	UiCanvasTheme out;
	out.theme = theme_id;

	auto css = read_file(path_join(base_dir, "ui/styles.css"));
	if (!css) return out;

	CssBlock block = find_theme_block(*css, out.theme);
	if (!block.valid) return out;  // unknown/absent theme keeps the :root default

	const std::string body = css->substr(block.begin, block.end - block.begin);

	size_t bg = body.find("--bg:");
	if (bg != std::string::npos) {
		size_t semi = body.find(';', bg);
		std::string value = body.substr(bg + 5, semi == std::string::npos ? std::string::npos : semi - bg - 5);
		parse_hex_color(value, out.r, out.g, out.b);
	}
	size_t scheme = body.find("color-scheme:");
	if (scheme != std::string::npos) {
		size_t semi = body.find(';', scheme);
		std::string value = body.substr(scheme + 13, semi == std::string::npos ? std::string::npos : semi - scheme - 13);
		out.light = contains(value, "light");
	}
	return out;
}

std::string apply_first_paint_theme(std::string html, const std::string& theme_id,
	const UiCanvasTheme& canvas) {
	if (theme_id.empty()) return html;

	// 1) <html data-theme="..."> — makes the :root-level variables valid as soon as
	//    the stylesheet is parsed, before any script runs
	size_t tag = html.find("<html");
	if (tag != std::string::npos) {
		size_t close = html.find('>', tag);
		if (close != std::string::npos && html.find("data-theme", tag) > close) {
			html.insert(close, " data-theme=\"" + theme_id + "\"");
		}
	}

	// 2) color-scheme: decides the canvas colour the browser paints before and
	//    behind the document (a light theme must not sit on a dark canvas)
	size_t meta = html.find("<meta name=\"color-scheme\"");
	if (meta != std::string::npos) {
		size_t close = html.find('>', meta);
		if (close != std::string::npos) {
			html.replace(meta, close - meta + 1,
				std::string("<meta name=\"color-scheme\" content=\"") +
					(canvas.light ? "light" : "dark") + "\" />");
		}
	}

	// 3) keep the flag the inline script reads (localStorage fallback path)
	size_t head = html.find("</head>");
	if (head != std::string::npos) {
		std::string inject = "<script>window.__PHI_THEME__=\"";
		inject += theme_id;
		inject += "\";</script>";
		html.insert(head, inject);
	}
	return html;
}

}  // namespace phi
