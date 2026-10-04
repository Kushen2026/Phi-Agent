// Resolve the "first frame" appearance (theme id + canvas colour) from the
// persisted settings, so the native splash and the webview canvas can use the
// user's theme before any CSS/JS has run.
#pragma once

#include <string>

namespace phi {

struct UiCanvasTheme {
	std::string theme;            // theme id in effect ("" = unknown -> default)
	bool light = false;           // the theme's color-scheme
	int r = 21, g = 21, b = 30;   // canvas background (the theme's --bg)
};

// Looks a theme id up in <base_dir>/ui/styles.css (--bg + color-scheme). An
// unknown id keeps the stylesheet's :root default.
UiCanvasTheme css_canvas_theme(const std::string& base_dir, const std::string& theme_id);

// Reads <base_dir>/data/settings.json (appearance.theme) and resolves it via
// css_canvas_theme(). Used by the shell, which runs before the settings store
// exists; the protocol server passes its already-loaded value instead.
UiCanvasTheme resolve_ui_canvas_theme(const std::string& base_dir);

// Rewrites the served index.html so the *first* paint is already themed:
//  - <html data-theme="..."> so the :root-level theme variables match the user's
//    choice the moment styles.css is parsed
//  - <meta name="color-scheme"> matching the theme, so the browser's own canvas
//    (painted before/behind the document) is not dark for a light theme
//  - window.__PHI_THEME__ kept for the inline script's localStorage fallback
// Pure function: same input, same output (unit-testable without a GUI).
std::string apply_first_paint_theme(std::string html, const std::string& theme_id,
	const UiCanvasTheme& canvas);

}  // namespace phi
