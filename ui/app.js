"use strict";
(() => {
  // src/web/icons.ts
  var stroke = (inner, size = 24) => `<svg class="icon" viewBox="0 0 ${size} ${size}" aria-hidden="true" focusable="false">${inner}</svg>`;
  var S = 'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"';
  var icons = {
    send: stroke(`<path d="m3 11 18-8-8 18-2.5-7.5L3 11z" ${S}/>`),
    stop: stroke(`<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/>`),
    check: stroke(`<path d="m5 12.5 4.5 4.5L19 7.5" ${S}/>`),
    close: stroke(`<path d="M6 6l12 12M18 6 6 18" ${S}/>`),
    refresh: stroke(
      `<path d="M20 11a8 8 0 0 0-14.9-3M4 13a8 8 0 0 0 14.9 3" ${S}/><path d="M20 4v7h-7M4 20v-7h7" ${S}/>`
    ),
    copy: stroke(`<rect x="9" y="9" width="11" height="11" rx="2" ${S}/><path d="M5 15V5a2 2 0 0 1 2-2h10" ${S}/>`),
    search: stroke(`<circle cx="10.5" cy="10.5" r="6.5" ${S}/><path d="m21 21-4.3-4.3" ${S}/>`),
    chevronDown: stroke(`<path d="m6 9 6 6 6-6" ${S}/>`),
    chevronRight: stroke(`<path d="m9 6 6 6-6 6" ${S}/>`),
    command: stroke(`<path d="M9.5 6 5 12l4.5 6M14.5 6 19 12l-4.5 6" ${S}/>`),
    sparkle: stroke(`<path d="M13 2 4.5 13.5H11L9.5 22 19 10h-6.5L13 2z" ${S}/>`),
    panel: stroke(`<rect x="3" y="4" width="18" height="16" rx="2" ${S}/><path d="M15 4v16" ${S}/>`),
    plus: stroke(`<path d="M12 5v14M5 12h14" ${S}/>`),
    terminal: stroke(`<rect x="3" y="4" width="18" height="16" rx="2" ${S}/><path d="m7 9 3 3-3 3M13 15h4" ${S}/>`),
    zap: stroke(`<path d="M13 2 4.5 13.5H11L9.5 22 19 10h-6.5L13 2z" ${S}/>`),
    book: stroke(`<path d="M4 5a2 2 0 0 1 2-2h14v17H6a2 2 0 0 0-2 2V5z" ${S}/><path d="M4 20a2 2 0 0 1 2-2h14" ${S}/>`),
    folder: stroke(`<path d="M3 7a2 2 0 0 1 2-2h4l2 3h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z" ${S}/>`),
    star: stroke(`<path d="m12 3 2.7 5.6 6.1.8-4.5 4.3 1.1 6-5.4-2.9-5.4 2.9 1.1-6L3.2 9.4l6.1-.8L12 3z" ${S}/>`),
    alert: stroke(
      `<path d="M12 9v5M12 17.5v.5" ${S}/><path d="M10.3 3.7 2.7 17a2 2 0 0 0 1.7 3h15.2a2 2 0 0 0 1.7-3L13.7 3.7a2 2 0 0 0-3.4 0z" ${S}/>`
    ),
    info: stroke(`<circle cx="12" cy="12" r="9" ${S}/><path d="M12 11v5M12 8v.5" ${S}/>`),
    cpu: stroke(
      `<rect x="6" y="6" width="12" height="12" rx="2" ${S}/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" ${S}/>`
    ),
    loader: stroke(
      `<path d="M12 2v4M12 18v4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8M2 12h4M18 12h4M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8" ${S}/>`
    ),
    paperclip: stroke(
      `<path d="m21.4 11.1-9.3 9.3a5.5 5.5 0 0 1-7.8-7.8l9.3-9.3a3.8 3.8 0 1 1 5.4 5.4l-9.3 9.3a2 2 0 1 1-2.8-2.8l8.6-8.6" ${S}/>`
    ),
    user: stroke(`<circle cx="12" cy="8" r="4" ${S}/><path d="M4.5 21a7.5 7.5 0 0 1 15 0" ${S}/>`),
    trash: stroke(
      `<path d="M4 7h16M10 11v6M14 11v6" ${S}/><path d="M6 7l1 13a1.8 1.8 0 0 0 1.8 1.7h6.4A1.8 1.8 0 0 0 17 20l1-13" ${S}/><path d="M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2" ${S}/>`
    )
  };

  // src/web/ui.ts
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== void 0) node.textContent = text;
    return node;
  }
  function clear(node) {
    node.replaceChildren();
  }
  function escapeHtml(value) {
    return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
  }
  function fuzzyMatches(query, text) {
    const q = query.toLowerCase();
    const t2 = text.toLowerCase();
    let qi = 0;
    for (let ti = 0; ti < t2.length && qi < q.length; ti++) {
      if (t2[ti] === q[qi]) qi += 1;
    }
    return qi === q.length;
  }
  function formatRelative(ms, nowOverride) {
    // nowOverride: renderSessions caches Date.now() once per render pass so
    // all rows on screen share the same reference point
    const diff = (nowOverride ?? Date.now()) - ms;
    if (diff < 6e4) return "\u521A\u521A";
    if (diff < 36e5) return `${Math.floor(diff / 6e4)} \u5206\u949F\u524D`;
    if (diff < 864e5) return `${Math.floor(diff / 36e5)} \u5C0F\u65F6\u524D`;
    return `${Math.floor(diff / 864e5)} \u5929\u524D`;
  }
  function formatTokens(n) {
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
    return String(n);
  }

  // src/web/i18n.ts
  var t = /* @__PURE__ */ (() => {
    const dict = {
      // Header / top bar
      "idle": "\u7A7A\u95F2",
      "streaming": "\u6D41\u5F0F\u4F20\u8F93\u4E2D\u2026",
      "compacting": "\u538B\u7F29\u4E2D\u2026",
      "compaction_role": "\u4E0A\u4E0B\u6587\u538B\u7F29",
      "compaction_tokens": (n) => `\u538B\u7F29\u524D\u4E0A\u4E0B\u6587 \u2248 ${formatTokens(n)} tokens`,
      "compaction_tokens_after": (n) => `\u538B\u7F29\u540E\u7EA6 ${formatTokens(n)} tokens`,
      "subagent_role": "\u5B50\u4EE3\u7406",
      "subagent_finished": (name) => `\u5B50\u4EE3\u7406\u300C${name}\u300D\u5DF2\u5B8C\u6210`,
      "subagent_failed": (name) => `\u5B50\u4EE3\u7406\u300C${name}\u300D\u672A\u6B63\u5E38\u5B8C\u6210`,
      "busy": "\u5FD9\u788C\u4E2D",
      "model": "\u6A21\u578B",
      "model_placeholder": "\u6A21\u578B \u2014",
      "no_model_selected": "\u672A\u9009\u62E9\u6A21\u578B",
      "thinking": "Thinking",
      // Thinking levels (indices match ["off","low","medium","high","xhigh","max"])
      "thinking_levels": ["Off", "Low", "Medium", "High", "Extreme", "Max"],
      "not_supported": "\u5F53\u524D\u6A21\u578B\u4E0D\u652F\u6301\u6B64\u9009\u9879",
      "current_model": "\u5F53\u524D\u6A21\u578B",
      "add_favorite": "\u6DFB\u52A0\u5230\u6536\u85CF",
      "remove_favorite": "\u4ECE\u6536\u85CF\u79FB\u9664",
      "favorites": "\u6536\u85CF",
      "no_models_available": "\u6CA1\u6709\u53EF\u7528\u6A21\u578B\u3002\u8BF7\u5728\u5DE6\u4FA7\u300CAPI \u914D\u7F6E\u300D\u9762\u677F\u4E2D\u6DFB\u52A0 API \u5BC6\u94A5\u540E\u6FC0\u6D3B\u3002",
      // Side panel
      "skills": "\u6280\u80FD",
      "commands": "\u547D\u4EE4",
      "sessions": "\u4F1A\u8BDD",
      "loaded_skills": "\u5DF2\u52A0\u8F7D\u6280\u80FD",
      "slash_commands": "\u659C\u6760\u547D\u4EE4",
      "no_skills_loaded": "\u672A\u52A0\u8F7D\u4EFB\u4F55\u6280\u80FD\u3002\u6280\u80FD\u4F1A\u81EA\u52A8\u4ECE data/skills \u52A0\u8F7D\u3002",
      "no_commands_available": "\u6682\u65E0\u53EF\u7528\u547D\u4EE4\u3002",
      "no_sessions": "\u5F53\u524D\u76EE\u5F55\u8FD8\u6CA1\u6709\u4FDD\u5B58\u7684\u4F1A\u8BDD\u3002",
      "reload_all": "\u91CD\u65B0\u52A0\u8F7D\u6280\u80FD\u3001\u6269\u5C55\u548C\u63D0\u793A\u8BCD",
      "refresh": "\u5237\u65B0",
      // Commands
      "run": "\u8FD0\u884C",
      "copy_path": "\u590D\u5236\u8DEF\u5F84",
      "copied": "\u5DF2\u590D\u5236 \u2713",
      "delete_skill_title": "\u5220\u9664\u6B64\u6280\u80FD\uFF08\u6587\u4EF6\u5939\u79FB\u5165\u56DE\u6536\u7AD9\uFF09",
      "delete_skill_confirm": "\u5220\u9664\u6280\u80FD\u300C{0}\u300D\uFF1F\u8BE5\u6280\u80FD\u7684\u6587\u4EF6\u5939\u4F1A\u88AB\u79FB\u52A8\u5230\u56DE\u6536\u7AD9\u3002",
      "skill_deleted": "\u6280\u80FD\u5DF2\u5220\u9664\uFF0C\u5DF2\u79FB\u5165\u56DE\u6536\u7AD9\u3002",
      // Session name
      "phi_session": "Phi \u4F1A\u8BDD",
      "rename_session": "\u91CD\u547D\u540D\u4F1A\u8BDD",
      "unnamed_session": "\u672A\u547D\u540D\u4F1A\u8BDD",
      "delete_session_title": "\u5220\u9664\u6B64\u4F1A\u8BDD\uFF08\u6587\u4EF6\u79FB\u5165\u56DE\u6536\u7AD9\uFF09",
      "delete_session_confirm": "\u5220\u9664\u4F1A\u8BDD\u300C{0}\u300D\uFF1F\u8BE5\u4F1A\u8BDD\u6587\u4EF6\u4F1A\u88AB\u79FB\u52A8\u5230\u56DE\u6536\u7AD9\u3002",
      "session_deleted": "\u4F1A\u8BDD\u5DF2\u5220\u9664\uFF0C\u5DF2\u79FB\u5165\u56DE\u6536\u7AD9\u3002",
      // New session
      "new_session": "\u65B0\u4F1A\u8BDD",

      // Input
      "message_phi": "\u7ED9 phi \u53D1\u6D88\u606F\u2026  (/ \u67E5\u770B\u547D\u4EE4)",
      "send_message": "\u53D1\u9001\u6D88\u606F",
      "stop_generation": "\u505C\u6B62\u751F\u6210",
      // Status line
      "no_model": "\u65E0\u6A21\u578B",
      "thinking_label": "Thinking: ",
      // Empty state
      "pi_agent": "phi agent",
      "connecting": "\u6B63\u5728\u8FDE\u63A5\u2026",
      "session_closed": "\u4F1A\u8BDD\u5DF2\u5173\u95ED\u3002",
      // Command palette
      "command_palette": "\u547D\u4EE4\u9762\u677F",
      "type_command": "\u8F93\u5165\u547D\u4EE4\u2026",
      "no_match": "\u6CA1\u6709\u5339\u914D\u300C{query}\u300D\u7684\u547D\u4EE4\u3002",
      "run_key": "\u21B5 \u8FD0\u884C",
      "navigate_keys": "\u2191\u2193 \u6D4F\u89C8",
      "close_keys": "esc \u5173\u95ED",
      // Scroll hint
      "new_activity": "\u2193 \u65B0\u52A8\u6001",
      // Tool cards
      "thinking_chars": (n) => `${n} chars`,
      "thinking_redacted": "(redacted)",
      "tool_failed": "\u5DE5\u5177\u6267\u884C\u5931\u8D25",
      // Messages
      "you": "\u4F60",
      "phi": "phi",
      "tool": "\u5DE5\u5177",
      "input_tokens": (n) => `\u8F93\u5165 ${n}`,
      "output_tokens": (n) => `\u8F93\u51FA ${n}`,
      // Connection
      "reconnecting": "\u6B63\u5728\u91CD\u8FDE\u2026",
      "connection_lost": "\u8FDE\u63A5\u5DF2\u65AD\u5F00",
      "not_connected": "\u672A\u8FDE\u63A5",
      "request_timeout": "\u8BF7\u6C42\u8D85\u65F6\uFF0C\u8BF7\u91CD\u8BD5",
      "creating_session": "\u6B63\u5728\u521B\u5EFA\u4F1A\u8BDD\u2026",
      "reloaded": "\u5DF2\u91CD\u65B0\u52A0\u8F7D\u6280\u80FD\u3001\u6269\u5C55\u548C\u63D0\u793A\u8BCD\u3002",
      // Web UI limitations
      "web_ui_not_available": "/{name} \u5728 Web UI \u4E2D\u4E0D\u53EF\u7528\u3002",
      // Attachments
      "remove_attachment": "\u79FB\u9664\u9644\u4EF6",
      "upload_failed": "\u4E0A\u4F20\u9644\u4EF6\u300C{0}\u300D\u5931\u8D25\uFF1A{1}",
      "attachment_image": "\u9644\u4EF6\u56FE\u7247",
      "image_preview": "\u56FE\u7247\u9884\u89C8",
      "image_preview_hint": "\u70B9\u51FB\u9884\u89C8\uFF08Esc \u5173\u95ED\uFF09",
      "image_preview_close": "\u5173\u95ED\u9884\u89C8",
      // Working directory
      "cwd_title": "\u5DE5\u4F5C\u76EE\u5F55\uFF1A{0}\n\u70B9\u51FB\u53EF\u5207\u6362\u76EE\u5F55",
      "cwd_required": "\u8BF7\u8F93\u5165\u76EE\u5F55\u8DEF\u5F84",
      "cwd_changed": "\u5DF2\u5207\u6362\u5DE5\u4F5C\u76EE\u5F55\uFF1A{0}",
      "browse_directory": "\u6D4F\u89C8\u2026",
      "picking_directory": "\u9009\u62E9\u4E2D\u2026",
      "pick_directory_failed": "\u9009\u62E9\u76EE\u5F55\u5931\u8D25\uFF1A{0}",
      // Suggestions
      "no_suggestions": "\u65E0\u5EFA\u8BAE",
      // Config panel (API 配置)
      "config_panel": "API \u914D\u7F6E",

      "configs": "\u914D\u7F6E",
      "new_config": "\u65B0\u5EFA\u914D\u7F6E",
      "edit_config": "\u7F16\u8F91\u914D\u7F6E",
      "config_name": "\u914D\u7F6E\u540D\u79F0",
      "config_name_placeholder": "\u4F8B\u5982\uFF1ADeepSeek",
      "provider": "\u670D\u52A1\u5546",
      "provider_placeholder": "\u9009\u62E9\u670D\u52A1\u5546",
      "api_key": "API \u5BC6\u94A5",
      "api_key_placeholder": "\u8F93\u5165 API \u5BC6\u94A5",
      "api_key_keep": "\u7559\u7A7A\u5219\u4FDD\u6301\u539F\u5BC6\u94A5",
      "show_key": "\u663E\u793A\u5BC6\u94A5",
      "hide_key": "\u9690\u85CF\u5BC6\u94A5",
      "select_model": "\u9009\u62E9\u6A21\u578B",
      "thinking_select": "\u601D\u8003\u7EA7\u522B",
      "save": "\u4FDD\u5B58",
      "cancel": "\u53D6\u6D88",
      "activate": "\u6FC0\u6D3B",
      "active_badge": "\u4F7F\u7528\u4E2D",
      "delete": "\u5220\u9664",
      "edit": "\u7F16\u8F91",
      "no_configs": "\u8FD8\u6CA1\u6709\u4FDD\u5B58\u7684\u914D\u7F6E\u3002\u65B0\u5EFA\u4E00\u4E2A API \u914D\u7F6E\uFF0C\u4FDD\u5B58\u540E\u70B9\u51FB\u5373\u53EF\u5FEB\u6377\u5207\u6362\u3002",
      "no_config": "API",
      "config_saved": "\u914D\u7F6E\u5DF2\u4FDD\u5B58\u3002",
      "config_saved": "\u914D\u7F6E\u5DF2\u4FDD\u5B58\u3002",
      "config_deleted": "\u914D\u7F6E\u5DF2\u5220\u9664\u3002",
      "config_activated": "\u5DF2\u6FC0\u6D3B\u914D\u7F6E\u300C{0}\u300D\u3002",
      "config_delete_confirm": "\u5220\u9664\u914D\u7F6E\u300C{0}\u300D\uFF1F",
      "config_activate_confirm": "\u5207\u6362\u5230\u914D\u7F6E\u300C{0}\u300D\uFF1F\u5F53\u524D\u4F1A\u8BDD\u7684\u6A21\u578B\u548C\u601D\u8003\u7EA7\u522B\u4F1A\u540C\u6B65\u5207\u6362\u3002",
      "no_api_key": "\u672A\u8BBE\u7F6E\u5BC6\u94A5",
      "no_key_error": "\u8BE5\u914D\u7F6E\u6CA1\u6709\u4FDD\u5B58 API \u5BC6\u94A5\uFF0C\u65E0\u6CD5\u6FC0\u6D3B\u3002\u8BF7\u5148\u7F16\u8F91\u5E76\u4FDD\u5B58\u5BC6\u94A5\u3002",
      "oauth_only": "\u4EC5\u652F\u6301 OAuth \u767B\u5F55\uFF0CWeb \u754C\u9762\u4E0D\u53EF\u7528",
      "err_config_name_required": "\u8BF7\u8F93\u5165\u914D\u7F6E\u540D\u79F0",
      "err_api_key_required": "\u8BF7\u8F93\u5165 API \u5BC6\u94A5",
      "err_baseurl_required": "\u8BF7\u8F93\u5165\u8BF7\u6C42\u5730\u5740",
      "err_model_required": "\u8BF7\u8F93\u5165\u6A21\u578B\u540D\u79F0",
      "err_subagent_incomplete": "\u5B50\u4EE3\u7406\u8BF7\u6C42\u5730\u5740\u548C\u6A21\u578B\u9700\u540C\u65F6\u586B\u5199",
      "subagent": "\u5B50\u4EE3\u7406",
      "configs_path_title": "\u914D\u7F6E\u4FDD\u5B58\u5728",
      "activate_hint": "\u70B9\u51FB\u884C\u5373\u53EF\u5207\u6362\u5230\u6B64\u914D\u7F6E",

      "base_url_label": "\u7F51\u5173\u5730\u5740\uFF08\u53EF\u9009\uFF09",
      "base_url_placeholder": "https://api.deepseek.com/anthropic",
      "base_url_hint": "\u7559\u7A7A\u4E3A\u5B98\u65B9 API\uFF1B\u586B\u5199\u5219\u4F5C\u4E3A Anthropic \u517C\u5BB9\u7F51\u5173\uFF08CC Switch \u98CE\u683C\uFF09",
      "quick_commands": "\u5FEB\u6377\u547D\u4EE4 \u2014 \u70B9\u51FB\u8FD0\u884C",
      "lang_zh": "\u4E2D\u6587",
      "lang_en": "English",
      // Builtin command handling
      "no_assistant_message": "\u8FD8\u6CA1\u6709\u53EF\u590D\u5236\u7684\u52A9\u624B\u6D88\u606F\u3002",
      "session_info": (name, id, model, thinking) => `\u4F1A\u8BDD\u300C${name}\u300D(${id}) \xB7 \u6A21\u578B: ${model} \xB7 \u601D\u8003: ${thinking}`,
      "palette_usage": (hint) => `\u7528\u6CD5: ${hint}`,
      "command_prefilled": "\u8F93\u5165\u53C2\u6570\u540E\u6309 Enter \u6267\u884C\uFF0C\u6216\u76F4\u63A5\u56DE\u8F66\u8FD0\u884C",
      // Time / relative
      "just_now": "\u521A\u521A",
      "minutes_ago": (n) => `${n} \u5206\u949F\u524D`,
      "hours_ago": (n) => `${n} \u5C0F\u65F6\u524D`,
      "days_ago": (n) => `${n} \u5929\u524D`
    };
    return function t2(key, ...args) {
      const raw = dict[key];
      if (raw === void 0) return key;
      if (typeof raw === "function") return raw(...args);
      if (Array.isArray(raw)) return raw;
      let result = raw;
      args.forEach((arg, i) => {
        result = result.replace(new RegExp(`\\{${i}\\}`, "g"), String(arg));
      });
      return result;
    };
  })();

  // src/web/panels.ts
  var Panels = class {
    constructor(api, elements) {
      this.currentThinking = "medium";
      this.currentConfig = null;
      this.configs = [];
      this.models = [];
      this.skills = [];
      this.api = api;
      this.elements = elements;
      elements.thinkingBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        this.toggleMenu(elements.thinkingMenu);
      });
      elements.configBtnFooter.addEventListener("click", (event) => {
        event.stopPropagation();
        this.toggleMenu(elements.configMenu);
      });
      document.addEventListener("click", (event) => {
        const target = event.target;
        const thinkingDropdown = document.getElementById("thinking-dropdown");
        const configDropdown = document.getElementById("config-dropdown");
        const thinkingMenu = elements.thinkingMenu;
        const configMenu = elements.configMenu;
        const isThinkingDropdown = thinkingDropdown?.contains(target);
        const isConfigDropdown = configDropdown?.contains(target);
        if (!thinkingMenu.hidden && !isThinkingDropdown) {
          thinkingMenu.hidden = true;
        }
        if (!configMenu.hidden && !isConfigDropdown) {
          configMenu.hidden = true;
        }
      });
      const tabs = [
        ["tab-skills", elements.paneSkills, this.elements.skillsList],
        ["tab-sessions", elements.paneSessions, this.elements.sessionsList]
      ];
      for (const [tabId, pane] of tabs) {
        const tab = document.getElementById(tabId);
        tab?.addEventListener("click", () => this.activateTab(tabId, pane));
      }
    }
    setSession(thinking) {
      this.currentThinking = thinking;
      this.renderThinkingMenu();
      this.updateHeaderLabels();
    }
    setConfig(configName, profiles) {
      this.currentConfig = configName;
      this.configs = profiles;
      this.renderConfigMenu();
      this.updateConfigLabel();
    }
    setModels(models) {
      this.models = models;
      this.renderThinkingMenu();
    }
    setSkills(skills) {
      this.skills = skills;
      this.elements.skillsDot.hidden = skills.length === 0;
      this.renderSkills();
    }
    setSessions(sessions, activeId) {
      this.renderSessions(sessions, activeId);
    }
    // ------------------------------------------------------------------------
    // Dropdown helpers
    // ------------------------------------------------------------------------
    toggleMenu(menu) {
      const willOpen = menu.hidden;
      menu.hidden = !willOpen;
    }
    updateHeaderLabels() {
      const levelIndex = ["off", "low", "medium", "high", "xhigh", "max"].indexOf(this.currentThinking);
      this.elements.thinkingLabel.textContent = levelIndex >= 0 ? t("thinking_levels")[levelIndex] : this.currentThinking;
    }
    updateConfigLabel() {
      this.elements.configLabel.textContent = this.currentConfig || t("no_config");
    }
    // ------------------------------------------------------------------------
    // Config menu
    // ------------------------------------------------------------------------
    renderConfigMenu() {
      const menu = this.elements.configMenu;
      clear(menu);
      this.updateConfigLabel();
      for (const profile of this.configs) {
        const item = el("div", "menu-item");
        item.setAttribute("role", "option");
        item.setAttribute("aria-selected", String(profile.name === this.currentConfig));
        const name = el("div", "menu-name", profile.name);
        const check = el("span", "menu-check", profile.name === this.currentConfig ? "\u25CF" : "");
        item.appendChild(name);
        item.appendChild(check);
        item.addEventListener("click", () => {
          menuClose(menu);
          if (profile.name !== this.currentConfig) {
            this.api.activateConfig(profile.name);
          }
        });
        menu.appendChild(item);
      }
    }
    // ------------------------------------------------------------------------
    // Thinking menu
    // ------------------------------------------------------------------------
    renderThinkingMenu() {
      const menu = this.elements.thinkingMenu;
      clear(menu);
      this.updateHeaderLabels();
      // highest first: the menu reads top-down as Max → Off
      const allLevels = [...THINKING_LEVELS].reverse();
      for (const level of allLevels) {
        const levelLabel = t("thinking_levels")[THINKING_LEVELS.indexOf(level)];
        const item = el("div", "menu-item");
        item.setAttribute("role", "option");
        item.setAttribute("aria-selected", String(level === this.currentThinking));
        const name = el("div", "menu-name", levelLabel);
        const check = el("span", "menu-check", level === this.currentThinking ? "\u25CF" : "");
        item.appendChild(name);
        item.appendChild(check);
        item.addEventListener("click", () => {
          menuClose(menu);
          if (level !== this.currentThinking) {
            this.api.setThinking(level);
          }
        });
        menu.appendChild(item);
      }
    }
    // ------------------------------------------------------------------------
    // Side panel
    // ------------------------------------------------------------------------
    togglePanel() {
      this.elements.sidePanel.classList.toggle("open");
    }
    isPanelOpen() {
      return this.elements.sidePanel.classList.contains("open");
    }
    activateTab(tabId, pane) {
      for (const tab of document.querySelectorAll(".panel-tab")) {
        tab.classList.toggle("active", tab.id === tabId);
        tab.setAttribute("aria-selected", String(tab.id === tabId));
      }
      for (const paneElement of document.querySelectorAll(".panel-pane")) {
        paneElement.classList.toggle("active", paneElement === pane);
      }
    }
    /** Switch the right panel to a tab by id (e.g. "tab-sessions"). */
    openTab(tabId) {
      const tabs = [
        ["tab-skills", this.elements.paneSkills],
        ["tab-sessions", this.elements.paneSessions]
      ];
      const entry = tabs.find(([id]) => id === tabId);
      if (!entry) return;
      this.activateTab(entry[0], entry[1]);
    }
    renderSkills() {
      const list = this.elements.skillsList;
      clear(list);
      if (this.skills.length === 0) {
        const empty = el("div", "pane-empty");
        empty.textContent = t("no_skills_loaded");
        list.appendChild(empty);
        return;
      }
      for (const skill of this.skills) {
        list.appendChild(this.buildSkillCard(skill));
      }
    }
    buildSkillCard(skill) {
      const card = el("div", "skill-card");
      const head = el("div", "skill-head");
      head.appendChild(el("span", "skill-name", skill.name));
      const scopeTag = el("span", "skill-tag");
      const baseDirName = skill.baseDir.split(/[\\/]/).filter(Boolean).slice(-2).join("/");
      scopeTag.textContent = baseDirName || "\u6280\u80FD";
      head.appendChild(scopeTag);
      if (skill.disableModelInvocation) {
        const tag = el("span", "skill-tag", "\u624B\u52A8");
        tag.title = "\u6A21\u578B\u65E0\u6CD5\u81EA\u52A8\u8C03\u7528\u6B64\u6280\u80FD\uFF1B\u8BF7\u4F7F\u7528 /skill:name";
        head.appendChild(tag);
      }
      card.appendChild(head);
      card.appendChild(el("div", "skill-desc", skill.description || "\u2014"));
      const actions = el("div", "skill-actions");
      const copyBtn = el("button", "btn", t("copy_path"));
      copyBtn.type = "button";
      copyBtn.addEventListener("click", () => {
        void navigator.clipboard.writeText(skill.filePath);
        copyBtn.textContent = t("copied");
        setTimeout(() => {
          copyBtn.textContent = t("copy_path");
        }, 1200);
      });
      actions.appendChild(copyBtn);
      const deleteBtn = el("button", "btn btn-danger", t("delete"));
      deleteBtn.type = "button";
      deleteBtn.title = t("delete_skill_title");
      deleteBtn.addEventListener("click", () => {
        this.api.deleteSkill(skill.filePath);
      });
      actions.appendChild(deleteBtn);
      card.appendChild(actions);
      card.appendChild(el("div", "skill-path", skill.filePath));
      return card;
    }
    renderSessions(sessions, activeId) {
      const list = this.elements.sessionsList;
      clear(list);
      if (sessions.length === 0) {
        const empty = el("div", "pane-empty");
        empty.textContent = t("no_sessions");
        list.appendChild(empty);
        return;
      }
      const now = Date.now();  // one timestamp per render pass
      for (const session of sessions) {
        const item = el("div", "session-item");
        item.classList.toggle("active", session.id === activeId);
        const open = el("button", "session-open");
        open.type = "button";
        const main = el("div", "session-main");
        main.appendChild(el("div", "session-name-line", session.name ?? session.id.slice(0, 12)));
        main.appendChild(
          el(
            "div",
            "session-time",
            `${session.updatedAt ? formatRelative(session.updatedAt, now) : ""} \xB7 ${session.id.slice(0, 8)}`
          )
        );
        open.appendChild(main);
        open.addEventListener("click", () => {
          if (session.id !== activeId) {
            this.api.openSession(session.id);
          }
        });
        item.appendChild(open);
        const deleteBtn = el("button", "session-delete");
        deleteBtn.type = "button";
        deleteBtn.title = t("delete_session_title");
        deleteBtn.innerHTML = icons.trash;
        deleteBtn.addEventListener("click", (event) => {
          event.stopPropagation();
          this.api.deleteSession(session.id);
        });
        item.appendChild(deleteBtn);
        list.appendChild(item);
      }
    }
  };
  function menuClose(menu) {
    menu.hidden = true;
  }
  var CommandPalette = class {
    constructor(overlay, input, results) {
      this.commands = [];
      this.selectedIndex = 0;
      this.filtered = [];
      this.overlay = overlay;
      this.input = input;
      this.results = results;
      input.addEventListener("input", () => this.filter());
      input.addEventListener("keydown", (event) => this.onKeyDown(event));
      overlay.addEventListener("mousedown", (event) => {
        if (event.target === overlay) {
          this.close();
        }
      });
    }
    setCommands(commands) {
      this.commands = commands;
    }
    open(onRun) {
      this.onRun = onRun;
      this.overlay.hidden = false;
      this.input.value = "";
      this.filter();
      this.input.focus();
    }
    close() {
      this.overlay.hidden = true;
    }
    isOpen() {
      return !this.overlay.hidden;
    }
    filter() {
      const query = this.input.value.trim();
      if (!query) {
        this.filtered = [...this.commands];
      } else {
        const ranked = [];
        for (const command of this.commands) {
          if (fuzzyMatches(query, command.name)) {
            ranked.push({ command, rank: 0 });
          } else if (command.description.toLowerCase().includes(query.toLowerCase())) {
            ranked.push({ command, rank: 1 });
          }
        }
        ranked.sort((a, b) => a.rank - b.rank || a.command.name.localeCompare(b.command.name));
        this.filtered = ranked.map((entry) => entry.command);
      }
      this.selectedIndex = 0;
      this.render();
    }
    render() {
      clear(this.results);
      if (this.filtered.length === 0) {
        const empty = el("div", "palette-empty", t("no_match", this.input.value));
        this.results.appendChild(empty);
        return;
      }
      const visible = this.filtered.slice(0, 50);
      this.selectedIndex = Math.min(this.selectedIndex, visible.length - 1);
      visible.forEach((command, index) => {
        const item = el("div", "palette-item");
        item.setAttribute("role", "option");
        item.classList.toggle("selected", index === this.selectedIndex);
        item.appendChild(el("span", "palette-name", `/${command.name}`));
        if (command.argumentHint) {
          item.appendChild(el("span", "palette-usage", t("palette_usage", command.argumentHint)));
        }
        item.appendChild(el("span", "palette-desc", command.description));
        item.appendChild(el("span", "palette-src", command.source));
        item.addEventListener("mousedown", (event) => {
          event.preventDefault();
          this.run(command);
        });
        this.results.appendChild(item);
      });
      const selected = this.results.children[this.selectedIndex];
      selected?.scrollIntoView({ block: "nearest" });
    }
    onKeyDown(event) {
      if (event.key === "Escape") {
        event.preventDefault();
        this.close();
        return;
      }
      if (event.key === "ArrowDown") {
        event.preventDefault();
        this.selectedIndex = Math.min(this.selectedIndex + 1, this.filtered.length - 1);
        this.render();
        return;
      }
      if (event.key === "ArrowUp") {
        event.preventDefault();
        this.selectedIndex = Math.max(this.selectedIndex - 1, 0);
        this.render();
        return;
      }
      if (event.key === "Enter") {
        event.preventDefault();
        const command = this.filtered[this.selectedIndex];
        if (command) this.run(command);
      }
    }
    run(command) {
      this.close();
      this.onRun?.(command);
    }
  };
  var THINKING_LEVELS = ["off", "low", "medium", "high", "xhigh", "max"];
  // ── client-side preference managers (localStorage) ────────────────────────

  var THEMES = [
    // ── 深色主题 ──
    { id: "midnight", name: "午夜", hint: "默认深色 · 青灰", colors: { accent: "#8abeb7", bg: "#13131c", bgActive: "#2e2e44" } },
    { id: "abyss", name: "深海", hint: "深蓝 · 青绿高亮", colors: { accent: "#4fd6c5", bg: "#0a1220", bgActive: "#22365a" } },
    { id: "aurora", name: "极光", hint: "紫罗兰 · 粉紫高亮", colors: { accent: "#c39bff", bg: "#13111e", bgActive: "#352e56" } },
    { id: "ember", name: "暖烬", hint: "暖棕 · 琥珀高亮", colors: { accent: "#e8a15c", bg: "#171210", bgActive: "#3c3028" } },
    { id: "graphite", name: "石墨", hint: "中性深灰 · 银白高亮", colors: { accent: "#b8bcc6", bg: "#141416", bgActive: "#32323b" } },
    { id: "crimson", name: "血月", hint: "纯黑 · 猩红高亮", colors: { accent: "#ff5c6c", bg: "#0e0a0b", bgActive: "#362226" } },
    { id: "cyber", name: "赛博朋克", hint: "暗夜 · 霓虹粉青", colors: { accent: "#ff3ea5", bg: "#0d0716", bgActive: "#301d55" } },
    { id: "matrix", name: "代码雨", hint: "全黑 · 荧光绿", colors: { accent: "#3dff6e", bg: "#050a05", bgActive: "#1d321d" } },
    { id: "nord", name: "北境", hint: "冰蓝灰 · 霜蓝高亮", colors: { accent: "#88c0d0", bg: "#2e3440", bgActive: "#4c566a" } },
    { id: "dracula", name: "暗爵", hint: "深紫 · 亮紫粉绿", colors: { accent: "#bd93f9", bg: "#282a36", bgActive: "#474b5e" } },
    { id: "monokai", name: "经典", hint: "橄榄灰 · 黄绿粉", colors: { accent: "#a6e22e", bg: "#272822", bgActive: "#4a4d3d" } },
    { id: "tokyo", name: "东京夜", hint: "鲛蓝 · 霓虹青紫", colors: { accent: "#7aa2f7", bg: "#1a1b26", bgActive: "#3b4261" } },
    { id: "ocean", name: "碧海", hint: "海军蓝 · 天青高亮", colors: { accent: "#38bdf8", bg: "#0b1626", bgActive: "#22406c" } },
    { id: "forest", name: "夜林", hint: "墨绿 · 荧光黄绿", colors: { accent: "#a3e635", bg: "#0c120c", bgActive: "#284028" } },
    { id: "coffee", name: "深焙", hint: "深咖啡 · 焦糖高亮", colors: { accent: "#d9a066", bg: "#17110c", bgActive: "#3f2f21" } },
    { id: "sunset", name: "落日", hint: "暗紫 · 橙红高亮", colors: { accent: "#ff8a5c", bg: "#1a0f1d", bgActive: "#472851" } },
    { id: "steel", name: "钢青", hint: "蓝灰 · 亮钢青高亮", colors: { accent: "#6fc3df", bg: "#10161c", bgActive: "#2b3d4c" } },
    // ── 浅色主题 ──
    { id: "paper", name: "纸白", hint: "浅色 · 墨绿点缀", colors: { accent: "#0e7a6e", bg: "#f4f3ee", bgActive: "#e0ddd1" } },
    { id: "moss", name: "苔原", hint: "浅色 · 森绿点缀", colors: { accent: "#4a7c3a", bg: "#f1f4ec", bgActive: "#d8e0c7" } },
    { id: "snow", name: "初雪", hint: "浅色 · 靛蓝点缀", colors: { accent: "#0969da", bg: "#f6f7f9", bgActive: "#dfe3ea" } },
    { id: "latte", name: "拿铁", hint: "米白 · 陶土点缀", colors: { accent: "#dc8a78", bg: "#f7f3ec", bgActive: "#e0d6c4" } },
    { id: "mint", name: "薄荷", hint: "浅色 · 青绿点缀", colors: { accent: "#0d9488", bg: "#f0f7f2", bgActive: "#c8e3d0" } },
    { id: "lavender", name: "薰衣草", hint: "浅色 · 紫罗兰点缀", colors: { accent: "#7c5cd6", bg: "#f4f2f9", bgActive: "#dbd3ec" } },
    { id: "peach", name: "蜜桃", hint: "奶白粉 · 珊瑚点缀", colors: { accent: "#e2643f", bg: "#faf1ec", bgActive: "#ecd4c6" } },
    { id: "sky", name: "晴空", hint: "浅蓝 · 湛蓝点缀", colors: { accent: "#0284c7", bg: "#eef5fb", bgActive: "#cfe1f0" } },
    { id: "sand", name: "暖沙", hint: "米黄 · 赭石点缀", colors: { accent: "#a4740a", bg: "#f6f1e7", bgActive: "#ded1b9" } }
  ];

  var TOOL_META = [
    { name: "bash", label: "bash", desc: "执行 shell 命令（实时输出）", perm: true, limit: { key: "bash", def: 120000 } },
    { name: "read", label: "read", desc: "带行号读取文件", perm: true, limit: { key: "read", def: 120000 } },
    { name: "write", label: "write", desc: "创建/覆盖文件", perm: true, limit: null },
    { name: "edit", label: "edit", desc: "精确字符串替换", perm: true, limit: null },
    { name: "ls", label: "ls", desc: "列出目录内容", perm: true, limit: { key: "ls", def: 60000 } },
    { name: "grep", label: "grep", desc: "搜索文件内容", perm: true, limit: { key: "grep", def: 60000 } },
    { name: "web_search", label: "web_search", desc: "联网搜索", perm: false, limit: null },
    { name: "fetch", label: "fetch", desc: "抓取网页内容", perm: false, limit: null },
    // 内置媒体引擎：Qwen-Image-2.1 文生图/参考图生图 / H3 视频+音频。两者都通过 output 路径落盘，
    // 因此和文件工具一样受「工作目录内/无限制」约束。
    // 步数不写死在这里：它在工具设置里可改（图像链 qwen image 2.1 默认 25 步，
    // 视频链 minimax h3 默认 20 步），写死过一次就成为过期信息（曾经还写着
    // Z-Image 的 8 步 / H3 的 16 步）。
    { name: "image_generate", label: "image_generate", desc: "Qwen-Image 2.1 文生图/参考图生图（输出 PNG）", perm: true, limit: null },
    { name: "video_generate", label: "video_generate", desc: "MiniMax H3 视频+音频生成（输出 mp4）", perm: true, limit: null },
    { name: "music_generate", label: "music_generate", desc: "ACE-Step 1.5 文生音乐（输出 wav）", perm: true, limit: null },
    { name: "tts_speak", label: "tts_speak", desc: "Breeze-TTS-2 音色描述合成语音（输出 wav）", perm: true, limit: null },
    { name: "agent", label: "agent", desc: "派发子代理任务", perm: false, limit: null }
  ];

  // 媒体模型选择：每个模型（链）各自成组，每个角色一个下拉
  // （settings.json → tools.media）。选项就是 models/ 目录里的文件，由服务端
  // 每次打开面板时实时列出，没有内置清单，也没有「默认」项——默认留空，
  // 留空时按该角色内置的默认文件名去 models/ 里找。
  // 每个模型用到的 JSON / 文本配置也是它自己的角色，绝不共用，而且拆到「文件」
  // 粒度：三条链各有自己的 vocab.json、merges.txt、tokenizer_config.json 三个下拉
  // （分词器真的需要这三个文件，缺 vocab.json 或 merges.txt 会直接加载失败），
  // TTS 的模型 config.json 与编解码器 config.json 也是两个不同的文件。
  var MEDIA_ROLES = {
    image: [
      { key: "image_dit", label: "图像 DiT" },
      { key: "image_te", label: "图像文本编码器" },
      { key: "image_vae", label: "图像 VAE" },
      { key: "image_tokenizer_vocab", label: "分词器 vocab.json" },
      { key: "image_tokenizer_merges", label: "分词器 merges.txt" },
      { key: "image_tokenizer_config", label: "分词器 tokenizer_config.json" }
    ],
    video: [
      { key: "video_dit", label: "视频 DiT" },
      { key: "video_te", label: "视频文本编码器" },
      { key: "video_vae", label: "视频 VAE" },
      { key: "video_avae", label: "音频 VAE + 声码器" },
      { key: "video_tokenizer_vocab", label: "分词器 vocab.json" },
      { key: "video_tokenizer_merges", label: "分词器 merges.txt" },
      { key: "video_tokenizer_config", label: "分词器 tokenizer_config.json" }
    ],
    music: [
      { key: "music_dit", label: "音乐 DiT" },
      { key: "music_te", label: "文本/语义码 tower（0.6B）" },
      { key: "music_lm", label: "音频语义码 LM（4B）" },
      { key: "music_vae", label: "音频 VAE" },
      { key: "music_tokenizer_vocab", label: "分词器 vocab.json" },
      { key: "music_tokenizer_merges", label: "分词器 merges.txt" },
      { key: "music_tokenizer_config", label: "分词器 tokenizer_config.json" }
    ],
    tts: [
      { key: "tts_model", label: "TTS 主模型" },
      { key: "tts_codec", label: "音频编解码器" },
      { key: "tts_tokenizer", label: "分词器 tokenizer.json" },
      { key: "tts_config", label: "TTS config.json" },
      { key: "tts_codec_config", label: "编解码器 config.json" }
    ]
  };
  // 采样步数：每组一个数字输入，图像链（qwen image 2.1）默认 25 步，视频链
  // （minimax h3）默认 20 步。服务端每次生成都会重读这两个值，所以保存后下一次
  // 生成就生效（不需要重启，也不需要重新加载模型）。
  var MEDIA_STEPS = {
    image: {
      key: "image_steps",
      label: "采样步数（qwen image 2.1）",
      desc: "image_generate 使用采样的步数",
      def: 25,
      min: 1,
      max: 100
    },
    video: {
      key: "video_steps",
      label: "采样步数（minimax h3）",
      desc: "video_generate 使用采样的步数",
      def: 20,
      min: 1,
      max: 100
    },
    music: {
      key: "music_steps",
      label: "采样步数（ace-step 1.5）",
      desc: "music_generate 使用采样的步数",
      def: 50,
      min: 1,
      max: 100
    }
  };
  // 采样器 / 调度器：移植自 ComfyUI，每条链一个下拉（settings.json → tools.media.
  // <chain>_sampler / <chain>_scheduler）。选项就是引擎真正移植并验证过的名字，
  // 每条链的默认值就是它发布的 workflow 里的那一对；服务端每次生成都重读，所以
  // 保存后下一次生成就生效。视频链的 H3 是联合音视频采样器，只支持 euler /
  // res_multistep 两种积分器，所以它的采样器选项只有这两个。
  var MEDIA_SAMPLERS = {
    image: {
      key: "image_sampler",
      label: "采样器（qwen image 2.1）",
      options: ["euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2s_ancestral", "dpmpp_2m_sde", "dpmpp_sde", "res_multistep", "lcm"],
      def: "euler"
    },
    video: {
      key: "video_sampler",
      label: "采样器（minimax h3 联合音视频）",
      options: ["res_multistep", "euler"],
      def: "res_multistep"
    },
    music: {
      key: "music_sampler",
      label: "采样器（ace-step 1.5）",
      options: ["euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2s_ancestral", "dpmpp_2m_sde", "dpmpp_sde", "res_multistep", "lcm"],
      def: "euler"
    }
  };
  var MEDIA_SCHEDULERS = {
    image: {
      key: "image_scheduler",
      label: "调度器（qwen image 2.1）",
      options: ["simple", "normal", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"],
      def: "simple"
    },
    video: {
      key: "video_scheduler",
      label: "调度器（minimax h3）",
      options: ["beta", "simple", "normal", "sgm_uniform", "karras", "exponential", "ddim_uniform", "linear_quadratic", "kl_optimal"],
      def: "beta"
    },
    music: {
      key: "music_scheduler",
      label: "调度器（ace-step 1.5）",
      options: ["simple", "normal", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"],
      def: "simple"
    }
  };
  // 有 LoRA 槽位的链（图像 2 个、视频 2 个，互不共用）。音乐与语音合成链没有
  // LoRA 选择：ACE-Step / Breeze 的权重是整套加载的，引擎也还没有它们的
  // LoRA 合并路径，所以它们没有 LoRA 列表元素，渲染循环靠 if (loraList) 跳过。
  var MEDIA_LORA_CHAINS = ["image", "video"];
  var MEDIA_CHAINS = ["image", "video", "music", "tts"];

  var HOTKEY_META = [
    { id: "input.send", label: "发送消息", def: "Enter" },
    { id: "input.newline", label: "输入换行", def: "Shift+Enter" },
    { id: "session.new", label: "新会话", def: "Ctrl+Alt+N" },
    { id: "input.focus", label: "聚焦输入框", def: "Ctrl+I" }
  ];

  var Hotkeys = {
    key: "phi.hotkeys",
    defaults: HOTKEY_META.reduce((map, h) => { map[h.id] = h.def; return map; }, {}),
    data: null,
    load() {
      try {
        this.data = Object.assign({}, this.defaults, JSON.parse(localStorage.getItem(this.key) || "{}"));
      } catch {
        this.data = Object.assign({}, this.defaults);
      }
      return this.data;
    },
    save() {
      try { localStorage.setItem(this.key, JSON.stringify(this.data)); } catch {}
    },
    loadFromJson(obj) {
      if (!obj || typeof obj !== "object") return;
      this.data = Object.assign({}, this.defaults, obj);
    },
    reset() {
      this.data = Object.assign({}, this.defaults);
      this.save();
    },
    normalizeKey(key) {
      const map = { " ": "Space", ArrowUp: "↑", ArrowDown: "↓", ArrowLeft: "←", ArrowRight: "→" };
      if (map[key]) return map[key];
      if (key.length === 1) return key.toUpperCase();
      return key;
    },
    /** serialize a keyboard event into the canonical combo string */
    combo(event) {
      const parts = [];
      if (event.ctrlKey || event.metaKey) parts.push("Ctrl");
      if (event.altKey) parts.push("Alt");
      if (event.shiftKey) parts.push("Shift");
      parts.push(this.normalizeKey(event.key));
      return parts.join("+");
    },
    /** true when the event exactly matches the stored combo for an action */
    match(event, action) {
      const want = (this.data || this.load())[action];
      if (!want) return false;
      const parts = want.split("+");
      const wantKey = parts[parts.length - 1];
      if (this.normalizeKey(event.key) !== wantKey) return false;
      const ctrl = event.ctrlKey || event.metaKey;
      if (ctrl !== parts.includes("Ctrl")) return false;
      if (event.altKey !== parts.includes("Alt")) return false;
      if (event.shiftKey !== parts.includes("Shift")) return false;
      return true;
    }
  };

  var Appearance = {
    key: "phi.appearance",
    defaults: { theme: "midnight", radius: "md", density: "cozy", motion: false, collapseThinking: false, collapseTools: false },
    data: null,
    // 上一次已经套用到 DOM 上的折叠偏好：用来区分「用户改了设置」与「其他
    // 设置项一起保存」，后者不应该覆盖用户手动开合的卡片
    appliedCollapseThinking: null,
    appliedCollapseTools: null,
    load() {
      try {
        this.data = Object.assign({}, this.defaults, JSON.parse(localStorage.getItem(this.key) || "{}"));
      } catch {
        this.data = Object.assign({}, this.defaults);
      }
      // 启动时就当作「已套用」：首帧的空对话里没有卡片可同步，之后新渲染的
      // 卡片直接读下面的两个取值，所以不会漏掉一次同步
      this.appliedCollapseThinking = this.data.collapseThinking === true;
      this.appliedCollapseTools = this.data.collapseTools === true;
      return this.data;
    },
    /**
     * 折叠偏好：只决定卡片「新加载 / 重建」时的默认开合（false = 展开）。
     *
     * 会话内用户手动点开的开合状态不在这里，也不会持久化：它只存在于 DOM 上，
     * 由 TranscriptView 在复用节点时保留（见 patchAssistantBody），所以重启 / 重新
     * 打开会话后回到偏好值 —— 这正是需求里「重启后还是展开 / 还是折叠」的语义。
     */
    collapseThinking() {
      if (!this.data) this.load();
      return this.data.collapseThinking === true;
    },
    collapseTools() {
      if (!this.data) this.load();
      return this.data.collapseTools === true;
    },
    /**
     * 把偏好套用到「此刻已经渲染出来的」卡片上：只在偏好真的变了的时候调用
     * （见 apply），这样点「保存」可以立刻看到效果，而保存其他设置项不会把手动
     * 开合过的卡片重置回去。
     */
    syncCollapseState() {
      const thinkingOpen = !this.collapseThinking();
      for (const node of document.querySelectorAll("#transcript details.thinking")) node.open = thinkingOpen;
      const toolsOpen = !this.collapseTools();
      for (const node of document.querySelectorAll("#transcript details.tool-card")) node.open = toolsOpen;
    },
    /** 偏好变了才同步已有卡片（见 syncCollapseState）；没变就什么都不做 */
    syncCollapseIfChanged() {
      const thinking = this.data.collapseThinking === true;
      const tools = this.data.collapseTools === true;
      if (this.appliedCollapseThinking === thinking && this.appliedCollapseTools === tools) return;
      this.appliedCollapseThinking = thinking;
      this.appliedCollapseTools = tools;
      this.syncCollapseState();
    },
    save() {
      try { localStorage.setItem(this.key, JSON.stringify(this.data)); } catch {}
    },
    apply() {
      if (!this.data) this.load();
      const body = document.body;
      body.dataset.theme = this.data.theme;
      body.dataset.radius = this.data.radius;
      body.dataset.density = this.data.density;
      body.dataset.motion = this.data.motion ? "reduced" : "normal";
      // the theme variables also hang off :root (so the server can stamp
      // data-theme on <html> and have the first paint already themed) — keep the
      // two in sync when the user switches theme in the settings panel
      if (document.documentElement.dataset.theme !== this.data.theme) {
        document.documentElement.dataset.theme = this.data.theme;
      }
      // 折叠偏好变了才同步已有卡片：新渲染的卡片自己读偏好，无需在这里管
      this.syncCollapseIfChanged();
    },
    loadFromJson(obj) {
      if (!obj || typeof obj !== "object") return;
      this.data = Object.assign({}, this.defaults, obj);
      // 服务端的偏好可能与本地缓存不同（换机器 / 手动改过 settings.json），
      // 同样要立刻套到屏幕上已有的卡片上
      this.syncCollapseIfChanged();
    }
  };

  var Prefs = {
    key: "phi.prefs",
    defaults: { sendKey: "enter", statusTokens: true },
    data: null,
    load() {
      try {
        this.data = Object.assign({}, this.defaults, JSON.parse(localStorage.getItem(this.key) || "{}"));
      } catch {
        this.data = Object.assign({}, this.defaults);
      }
      return this.data;
    },
    save() {
      try { localStorage.setItem(this.key, JSON.stringify(this.data)); } catch {}
    },
    loadFromJson(obj) {
      if (!obj || typeof obj !== "object") return;
      this.data = Object.assign({}, this.defaults, obj);
    }
  };

  // ── SettingsPanel: full-window settings with a fixed page list ────────────

  var SettingsPanel = class {
    constructor(api, elements) {
      this.api = api;
      this.elements = elements;
      this.profiles = [];
      this.active = null;
      this.editing = null;
      this.currentPage = "api";
      this.modelLists = { main: [], subagent: [] };
      this.toolSettings = null;
      this.hotkeyCapture = null;
      // Draft values for non-API settings (saved only on "保存", discarded on "取消")
      this.draftToolSettings = null;
      this.draftHotkeys = null;
      this.draftAppearance = null;
      this.draftPrefs = null;

      // Load from localStorage first (synchronous, available immediately);
      // getUiSettings is called later after the ready event confirms the server is up.
      Hotkeys.load();
      Appearance.load();
      // theme is applied synchronously by the inline script in index.html before first paint
      // (reads localStorage directly), so no early Appearance.apply() is needed here.
      Prefs.load();
      // 一次性迁移：旧版「发送消息」下拉设置（Prefs.sendKey）→ 两个可绑定快捷键
      try {
        if (!localStorage.getItem("phi.sendkey.migrated")) {
          if (Prefs.data.sendKey === "ctrl-enter") {
            Hotkeys.data["input.send"] = "Ctrl+Enter";
            Hotkeys.data["input.newline"] = "Enter";
            Hotkeys.save();
          }
          localStorage.setItem("phi.sendkey.migrated", "1");
        }
      } catch {}

      // ── navigation (fixed, non-collapsible page list) ──
      for (const item of elements.menu.querySelectorAll(".settings-menu-item")) {
        item.addEventListener("click", () => this.showPage(item.dataset.page));
      }
      elements.closeBtn.addEventListener("click", () => this.commit());
      document.getElementById("settings-cancel").addEventListener("click", () => this.discardAndClose());
      elements.overlay.addEventListener("mousedown", (event) => {
        if (event.target === elements.overlay) this.close();
      });
      // hotkey capture runs in the capture phase so it wins over app handlers;
      // Esc cancels the capture instead of closing the window
      document.addEventListener("keydown", (event) => {
        if (!this.hotkeyCapture) return;
        event.preventDefault();
        event.stopPropagation();
        // 修饰键单独按下：不结束捕获，继续等待组合键
        if (event.key === "Control" || event.key === "Shift" || event.key === "Alt" || event.key === "Meta") return;
        const finish = this.hotkeyCapture;
        this.hotkeyCapture = null;
        if (event.key === "Escape") {
          finish("", true);
        } else {
          finish(Hotkeys.combo(event), false);
        }
      }, { capture: true });
      // 点击绑定框以外的任何位置都会取消捕获，避免捕获态卡死输入
      document.addEventListener("mousedown", (event) => {
        if (!this.hotkeyCapture) return;
        if (this.captureKbd && this.captureKbd.contains(event.target)) return;
        this.cancelHotkeyCapture();
      }, { capture: true });
      // Ensure capture is cancelled if panel is hidden externally
      // (kept as a reference so close() can disconnect it)
      this._obsPanel = new MutationObserver(() => {
        if (this.elements.overlay.hidden && this.hotkeyCapture) this.cancelHotkeyCapture();
      });
      this._obsPanel.observe(this.elements.overlay, { attributes: true, attributeFilter: ["hidden"] });

      // ── API config page ──
      this.keyToggleUpdate = this.bindKeyToggle(this.el("config-key"), this.el("config-key-toggle"));
      this.subagentKeyToggleUpdate = this.bindKeyToggle(this.el("subagent-key"), this.el("subagent-key-toggle"));
      this.el("config-new-btn").addEventListener("click", () => this.startCreate());
      this.el("config-cancel").addEventListener("click", () => this.hideForm());
      this.el("config-form").addEventListener("submit", (event) => {
        event.preventDefault();
        void this.submit();
      });
      this.el("config-fetch-models").addEventListener("click", () => void this.fetchModels("main"));
      this.el("subagent-fetch-models").addEventListener("click", () => void this.fetchModels("subagent"));
      this.el("config-model").addEventListener("change", () => {
        // 已移除手动输入入口
      });
      this.el("subagent-model").addEventListener("change", () => {
        // 已移除手动输入入口
      });
      // 已移除手动输入相关的 back 按钮

      // ── tools page ──
      // Tool settings are edited via draft (commit on 保存).

      // ── appearance page ──
      this.el("appearance-radius").addEventListener("change", () => {
        if (!this.draftAppearance) return;
        this.draftAppearance.radius = this.el("appearance-radius").value;
      });
      this.el("appearance-density").addEventListener("change", () => {
        if (!this.draftAppearance) return;
        this.draftAppearance.density = this.el("appearance-density").value;
      });
      this.el("appearance-motion").addEventListener("change", () => {
        if (!this.draftAppearance) return;
        this.draftAppearance.motion = this.el("appearance-motion").checked;
      });
      this.el("appearance-collapse-thinking").addEventListener("change", () => {
        if (!this.draftAppearance) return;
        this.draftAppearance.collapseThinking = this.el("appearance-collapse-thinking").checked;
      });
      this.el("appearance-collapse-tools").addEventListener("change", () => {
        if (!this.draftAppearance) return;
        this.draftAppearance.collapseTools = this.el("appearance-collapse-tools").checked;
      });

      // ── general page ──
      this.el("general-status-tokens").addEventListener("change", () => {
        if (!this.draftPrefs) return;
        this.draftPrefs.statusTokens = this.el("general-status-tokens").checked;
        this.el("general-status-tokens").dataset.dirty = "1";
      });
      this.el("general-reset-all").addEventListener("click", () => {
        if (!window.confirm("重置除 API 配置以外的所有设置？\n\n主题、快捷键、通用选项将恢复默认；工具设置（启用/权限/截断/子代理）也将重置。")) return;
        // Reset drafts instead of live data
        this.draftHotkeys = Object.assign({}, Hotkeys.defaults);
        this.draftAppearance = Object.assign({}, Appearance.defaults);
        this.draftPrefs = Object.assign({}, Prefs.defaults);
        // Reset tool settings to defaults
        const defaults = {
          enabled: {},
          permissions: {},
          outputLimits: { bash: 120000, read: 120000, ls: 60000, grep: 60000 },
          subagent: { tools: [] }
        };
        for (const tool of TOOL_META) {
          defaults.enabled[tool.name] = true;
          defaults.permissions[tool.name] = "everywhere";
        }
        this.draftToolSettings = defaults;
        this.renderHotkeys();
        this.renderAppearance();
        this.applyStatusTokens();
        if (this.toolSettings) this.renderToolSettings();
        this.api.showBanner("已重置全部设置（API 配置不受影响），请点击「保存」保存。", "info");
      });

      this.renderThemeCards();
      this.renderHotkeys();
      this.renderAppearance();
      this.renderGeneral();
    }

    el(id) {
      return document.getElementById(id);
    }

    // ------------------------------------------------------------------------
    // Page switching
    // ------------------------------------------------------------------------
    showPage(page) {
      this.currentPage = page;
      for (const item of this.elements.menu.querySelectorAll(".settings-menu-item")) {
        item.classList.toggle("active", item.dataset.page === page);
      }
      for (const section of this.elements.content.querySelectorAll(".settings-page")) {
        section.classList.toggle("active", section.dataset.page === page);
        section.hidden = section.dataset.page !== page;
      }
      if (page === "tools") {
        // 实时列出 models/ 下的文件：每次进入工具设置都重新问一次服务端。
        if (!this.toolSettings) void this.refreshToolSettings();
        else void this.refreshMediaModels();
      }
      if (page === "hotkeys") this.renderHotkeys();
      if (page === "appearance") this.renderAppearance();
      if (page === "general") this.renderGeneral();
    }

    isOpen() {
      return !this.elements.overlay.hidden;
    }
    toggle() {
      if (this.isOpen()) this.close();
      else this.open();
    }
    open(page) {
      // Capture current state as draft before allowing edits
      this.draftToolSettings = this.toolSettings ? JSON.parse(JSON.stringify(this.toolSettings)) : null;
      this.draftHotkeys = JSON.parse(JSON.stringify(Hotkeys.data));
      this.draftAppearance = JSON.parse(JSON.stringify(Appearance.data));
      this.draftPrefs = JSON.parse(JSON.stringify(Prefs.data));
      this.elements.overlay.hidden = false;
      // showPage() 同时负责「左侧菜单高亮」和「右侧分区显隐」，所以不带 page 参数
      // 时（齿轮按钮的 toggle() → open()）也必须调用：右侧 API 配置分区的 active
      // 是 index.html 里硬编码的，而菜单项的高亮只能由 showPage 给 —— 漏掉它，
      // 启动后第一次打开设置就会是「正文可见、左侧一项都没选中」。
      // 回落顺序：显式传入的页 → 上次停留的页（构造时初始化为 api）。
      this.showPage(page || this.currentPage || "api");
      void this.refresh();
      if (this.toolSettings === null) void this.refreshToolSettings();
    }
    close() {
      this.elements.overlay.hidden = true;
      this.hideForm();
      this.cancelHotkeyCapture();
      if (this._obsPanel) { this._obsPanel.disconnect(); this._obsPanel = null; }
    }
    // ── Commit all pending changes at once ─────────────────────────────────
    async commit() {
      // 1) Tool settings → server, then apply
      if (this.draftToolSettings) {
        this.toolSettings = this.draftToolSettings;
        this.draftToolSettings = null;
        await this.api.saveToolSettings(this.toolSettings);
        this.api.applyToolSettings?.();
      }
      // 2) Merge drafts into live data, save everything to server
      Hotkeys.data = this.draftHotkeys || Hotkeys.data;
      Appearance.data = this.draftAppearance || Appearance.data;
      Prefs.data = this.draftPrefs || Prefs.data;
      this.draftHotkeys = null;
      this.draftAppearance = null;
      this.draftPrefs = null;
      // Apply visual state immediately
      Hotkeys.save();
      Appearance.save(); Appearance.apply();
      Prefs.save();
      // Persist to server (non-blocking: UI already applied from local data)
      this.api.saveUiSettings({ hotkeys: Hotkeys.data, appearance: Appearance.data, prefs: Prefs.data });
      this.applyStatusTokens();
      this.close();
    }
    // ── Discard all pending changes and close ──────────────────────────────
    discardAndClose() {
      this.draftToolSettings = null;
      this.draftHotkeys = null;
      this.draftAppearance = null;
      this.draftPrefs = null;
      this.elements.overlay.hidden = true;
      this.hideForm();
      this.cancelHotkeyCapture();
      if (this._obsPanel) { this._obsPanel.disconnect(); this._obsPanel = null; }
    }

    // ------------------------------------------------------------------------
    // API configs
    // ------------------------------------------------------------------------
    setConfigs(result) {
      this.active = result.active;
      this.profiles = result.profiles;
      this.renderProfiles();
      this.onConfigsChange?.(result);
    }
    async refresh() {
      const result = await this.api.fetchConfigs();
      if (result) this.setConfigs(result);
    }
    /** 密钥输入框的显示/隐藏切换：默认 password 圆点，点击后切换为明文 */
    bindKeyToggle(input, toggleBtn) {
      const update = (visible) => {
        input.type = visible ? "text" : "password";
        toggleBtn.title = visible ? "隐藏密钥" : "显示密钥";
        toggleBtn.setAttribute("aria-label", toggleBtn.title);
      };
      toggleBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        update(input.type === "password");
      });
      update(false);
      return update;
    }
    /** Populate the language + subagent fields of the edit form from a profile. */
    fillProfileExtras(profile) {
      this.el("config-lang").value = profile?.systemPromptLanguage ?? this.defaultLanguage();
      const subagent = profile?.subagent;
      this.el("subagent-baseurl").value = subagent?.baseUrl ?? "";
      this.el("subagent-apiformat").value = subagent?.apiFormat ?? "anthropic-messages";
      this.fillModelSelect("subagent", this.modelLists.subagent, subagent?.model ?? "");
      this.el("subagent-key").value = subagent?.apiKey || subagent?.apiKeyMasked || "";
      this.el("subagent-key").placeholder = subagent?.apiKeyMasked ? "留空则保持原密钥" : "输入 API 密钥";
      this.el("subagent-key-hint").hidden = !subagent?.apiKeyMasked;
      this.subagentKeyToggleUpdate(false);
    }
    /** Default language for a new profile: inherit the active profile's choice. */
    defaultLanguage() {
      const active = this.profiles.find((profile) => profile.name === this.active);
      return active?.systemPromptLanguage ?? "zh";
    }

    // ------------------------------------------------------------------------
    // Model select + 获取模型列表
    // ------------------------------------------------------------------------
    /** (re)populate a model <select>; populate from remote list or current value */
    fillModelSelect(which, models, current) {
      const select = this.el(which === "main" ? "config-model" : "subagent-model");
      clear(select);
      const seen = new Set();
      const addOption = (value, label) => {
        if (seen.has(value)) return;
        seen.add(value);
        const option = el("option");
        option.value = value;
        option.textContent = label;
        if (value === current) option.selected = true;
        select.appendChild(option);
      };
      if (current) addOption(current, current);
      for (const m of models) addOption(m, m);
      if (models.length === 0 && !current) {
        const placeholder = el("option");
        placeholder.value = "";
        placeholder.textContent = "点击「获取模型列表」获取可选模型";
        select.appendChild(placeholder);
      }
    }

    /** current model value for a slot */
    modelValue(which) {
      return this.el(which === "main" ? "config-model" : "subagent-model").value;
    }

    looksMasked(key) {
      return !key || key.includes("…") || key.startsWith("••••");
    }

    /** POST /api/fetch-models: pull the model list from the request URL */
    async fetchModels(which) {
      const button = this.el(which === "main" ? "config-fetch-models" : "subagent-fetch-models");
      const baseUrl = this.el(which === "main" ? "config-baseurl" : "subagent-baseurl").value.trim();
      const apiFormat = this.el(which === "main" ? "config-apiformat" : "subagent-apiformat").value;
      if (!baseUrl) {
        this.api.showBanner("请先填写请求地址", "error");
        return;
      }
      let apiKey = this.el(which === "main" ? "config-key" : "subagent-key").value.trim();
      if (this.looksMasked(apiKey)) apiKey = "";
      button.disabled = true;
      button.classList.add("loading");
      try {
        const response = await fetch("/api/fetch-models", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ baseUrl, apiKey, apiFormat })
        });
        const result = await response.json().catch(() => null);
        if (!result) {
          throw new Error(`HTTP ${response.status}：响应不是有效的 JSON`);
        }
        const models = result.models || [];
        if (models.length === 0) {
          // 直接报错（含错误代码），不再切换到手动输入
          const detail = result.error || "服务器未返回模型列表";
          throw new Error(`${detail}｜错误代码 HTTP ${response.status}`);
        }
        this.modelLists[which] = models;
        if (which === "main") {
          this.fillModelSelect("main", models, this.modelValue("main"));
        } else {
          this.fillModelSelect("subagent", models, this.modelValue("subagent"));
        }
        this.api.showBanner(`已获取 ${models.length} 个模型。`, "info");
      } catch (error) {
        this.api.showBanner(`获取模型列表失败：${error instanceof Error ? error.message : String(error)}`, "error");
      } finally {
        button.disabled = false;
        button.classList.remove("loading");
      }
    }

    // ------------------------------------------------------------------------
    // Profiles list
    // ------------------------------------------------------------------------
    renderProfiles() {
      const list = this.el("config-profiles");
      clear(list);
      if (this.profiles.length === 0) {
        list.appendChild(el("div", "config-empty", t("no_configs")));
        return;
      }
      for (const profile of this.profiles) {
        list.appendChild(this.buildProfileCard(profile));
      }
    }
    buildProfileCard(profile) {
      const card = el("div", "config-profile");
      card.classList.toggle("active", profile.name === this.active);
      card.title = t("activate_hint");
      const head = el("div", "config-profile-head");
      const avatar = el("span", "config-avatar");
      avatar.textContent = (profile.name.trim()[0] ?? "?").toUpperCase();
      avatar.style.setProperty("--avatar-hue", this.hueFor(profile.name));
      head.appendChild(avatar);
      head.appendChild(el("span", "config-profile-name", profile.name));
      if (profile.name === this.active) {
        head.appendChild(el("span", "config-active-badge", t("active_badge")));
      }
      card.appendChild(head);
      const meta = el("div", "config-profile-meta");
      // API 格式才是关键信息（provider 字段恒为兼容层名称，显示会误导）
      meta.appendChild(el("span", "skill-tag", profile.apiFormat === "openai-completions" ? "OpenAI 兼容" : "Anthropic 兼容"));
      meta.appendChild(el("span", "skill-tag", profile.model));
      if (profile.systemPromptLanguage) {
        meta.appendChild(el("span", "skill-tag", profile.systemPromptLanguage === "en" ? "EN" : "中文"));
      }
      card.appendChild(meta);
      if (profile.baseUrl) {
        const urlLine = el("div", "config-profile-key");
        urlLine.textContent = profile.baseUrl;
        urlLine.title = profile.baseUrl;
        card.appendChild(urlLine);
      }
      const keyLine = el("div", "config-profile-key");
      keyLine.textContent = profile.apiKeyMasked || t("no_api_key");
      keyLine.style.color = profile.apiKeyMasked ? "var(--text-faint)" : "var(--yellow)";
      card.appendChild(keyLine);
      const actions = el("div", "config-profile-actions");
      const editBtn = el("button", "btn", t("edit"));
      editBtn.type = "button";
      editBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        this.startEdit(profile);
      });
      actions.appendChild(editBtn);
      const deleteBtn = el("button", "btn", t("delete"));
      deleteBtn.type = "button";
      deleteBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        void this.remove(profile.name);
      });
      actions.appendChild(deleteBtn);
      card.appendChild(actions);
      card.addEventListener("click", () => {
        if (profile.name !== this.active && profile.apiKeyMasked) {
          void this.activate(profile.name);
        }
      });
      return card;
    }
    /** Stable hue per config name for the avatar dot. */
    hueFor(name) {
      let hash = 0;
      for (const char of name) hash = (hash * 31 + char.charCodeAt(0)) % 360;
      return String(hash);
    }
    async activate(name) {
      const result = await this.api.activateConfig(name);
      if (result) {
        this.setConfigs(result);
        this.onActivated?.(name);
      }
    }
    async remove(name) {
      if (!window.confirm(t("config_delete_confirm", name))) return;
      const result = await this.api.deleteConfig(name);
      if (result) this.setConfigs(result);
    }

    // ------------------------------------------------------------------------
    // Profile form
    // ------------------------------------------------------------------------
    startCreate() {
      this.editing = null;
      this.el("config-form-title").textContent = t("new_config");
      this.el("config-name").value = "";
      this.el("config-name").placeholder = t("config_name_placeholder");
      this.el("config-baseurl").value = "";
      this.el("config-apiformat").value = "anthropic-messages";
      this.modelLists.main = [];
      this.fillModelSelect("main", [], "");
      this.el("config-key").value = "";
      this.el("config-key").placeholder = t("api_key_placeholder");
      this.el("config-key-hint").hidden = true;
      this.el("subagent-baseurl").value = "";
      this.el("subagent-apiformat").value = "anthropic-messages";
      this.el("subagent-key").value = "";
      this.el("subagent-key").placeholder = t("api_key_placeholder");
      this.el("subagent-key-hint").hidden = true;
      this.subagentKeyToggleUpdate(false);
      this.fillModelSelect("subagent", this.modelLists.subagent, "");
      this.el("config-lang").value = this.defaultLanguage();
      this.showForm();
    }
    startEdit(profile) {
      this.editing = profile.name;
      this.el("config-form-title").textContent = t("edit_config");
      this.el("config-name").value = profile.name;
      this.el("config-baseurl").value = profile.baseUrl ?? "";
      this.el("config-apiformat").value = profile.apiFormat ?? "anthropic-messages";
      this.fillModelSelect("main", this.modelLists.main, profile.model);
      // 预填真实密钥（默认圆点隐藏，点击「显示」可查看原密钥）；保存时后端识别掩码串并保留原密钥
      this.el("config-key").value = profile.apiKey || profile.apiKeyMasked || "";
      this.el("config-key").placeholder = profile.apiKeyMasked ? t("api_key_keep") : t("api_key_placeholder");
      this.el("config-key-hint").hidden = !profile.apiKeyMasked;
      this.fillProfileExtras(profile);
      this.showForm();
    }
    showForm() {
      this.el("config-form").hidden = false;
      this.el("config-name").focus();
      this.el("config-form").scrollIntoView({ block: "nearest" });
    }
    hideForm() {
      this.el("config-form").hidden = true;
      this.editing = null;
    }
    async submit() {
      const name = this.el("config-name").value.trim();
      if (!name) {
        this.showError(t("err_config_name_required"));
        return;
      }
      const baseUrl = this.el("config-baseurl").value.trim();
      if (!baseUrl) {
        this.showError(t("err_baseurl_required"));
        return;
      }
      const model = this.modelValue("main");
      if (!model) {
        this.showError(t("err_model_required"));
        return;
      }
      const isNew = this.editing === null;
      const key = this.el("config-key").value.trim();
      if (isNew && !key) {
        this.showError(t("err_api_key_required"));
        return;
      }
      const saBaseUrl = this.el("subagent-baseurl").value.trim();
      const saModel = this.modelValue("subagent");
      if ((saBaseUrl && !saModel) || (!saBaseUrl && saModel)) {
        this.showError(t("err_subagent_incomplete"));
        return;
      }
      const systemPromptLanguage = this.el("config-lang").value === "en" ? "en" : "zh";
      const apiFormat = this.el("config-apiformat").value;
      const draft = {
        name,
        // provider 跟随 API 格式，供直连（无网关地址）时登记密钥使用
        provider: apiFormat === "openai-completions" ? "openai" : "anthropic",
        model,
        apiKey: key,
        baseUrl,
        apiFormat,
        // Subagent + language are part of the profile; an empty subagent
        // clears any previously stored one.
        subagent: {
          baseUrl: saBaseUrl,
          model: saModel,
          apiFormat: this.el("subagent-apiformat").value,
          apiKey: this.el("subagent-key").value.trim()
        },
        systemPromptLanguage
      };
      const result = await this.api.saveConfig(draft);
      if (result) {
        this.setConfigs(result);
        this.hideForm();
        this.onSaved?.(name);
      }
    }
    showError(message) {
      this.api.showBanner(message, "error");
    }

    // ------------------------------------------------------------------------
    // Tool settings page
    // ------------------------------------------------------------------------
    async refreshToolSettings() {
      const result = await this.api.getToolSettings();
      if (result && result.kind === "tool_settings") {
        this.toolSettings = result.settings;
        // Update draft if panel is open
        if (!this.elements.overlay.hidden) {
          this.draftToolSettings = JSON.parse(JSON.stringify(result.settings));
        }
        this.renderToolSettings();
      }
      void this.refreshMediaModels();
    }
    async refreshMediaModels() {
      if (!this.api.listMediaModels) return;
      const result = await this.api.listMediaModels();
      if (result && result.kind === "media_models") {
        this.mediaModels = result;
        this.renderMediaSettings();
      }
    }
    async loadUiSettings() {
      const result = await this.api.getUiSettings();
      if (result && result.kind === "ui_settings") {
        Hotkeys.loadFromJson(result.settings.hotkeys);
        Appearance.loadFromJson(result.settings.appearance);
        Prefs.loadFromJson(result.settings.prefs);
        // Update draft values so in-session edits start from server state
        this.draftHotkeys = result.settings.hotkeys ? JSON.parse(JSON.stringify(result.settings.hotkeys)) : null;
        this.draftAppearance = result.settings.appearance ? JSON.parse(JSON.stringify(result.settings.appearance)) : null;
        this.draftPrefs = result.settings.prefs ? JSON.parse(JSON.stringify(result.settings.prefs)) : null;
        // Re-apply only if the theme currently active (on <html>, where the
        // server stamped it for the first paint) differs from the server's.
        // Note: comparing Appearance.data is useless here — loadFromJson() above
        // already made it equal to the server value, which silently skipped apply().
        const applied = document.documentElement.dataset.theme || document.body.dataset.theme;
        const serverTheme = result.settings.appearance?.theme;
        if (serverTheme && applied !== serverTheme) {
          Appearance.apply();
        }
        this.applyStatusTokens();
      }
    }
    applyStatusTokens() {
      const show = Prefs.data.statusTokens;
      const sep = this.elements.content?.querySelector("#status-tokens-sep") || document.getElementById("status-tokens-sep");
      const tokens = document.getElementById("status-tokens");
      if (sep) sep.hidden = !show;
      if (tokens) tokens.hidden = !show;
    }

    renderToolSettings() {
      const ts = this.draftToolSettings || this.toolSettings;
      if (!ts) return;
      // ── enable + permission rows ──
      const permList = this.el("tools-perm-list");
      clear(permList);
      for (const tool of TOOL_META) {
        const row = el("div", "tool-row");
        const main = el("div", "switch-row-main");
        main.appendChild(el("span", "switch-row-name", tool.label));
        main.appendChild(el("span", "switch-row-desc", tool.desc));
        row.appendChild(main);
        if (tool.perm) {
          const select = el("select", "inline-select");
          for (const [value, label] of [["everywhere", "无限制"], ["in-cwd", "工作目录内"]]) {
            const option = el("option");
            option.value = value;
            option.textContent = label;
            select.appendChild(option);
          }
          select.value = ts.permissions?.[tool.name] === "in-cwd" ? "in-cwd" : "everywhere";
          select.addEventListener("change", () => {
            ts.permissions[tool.name] = select.value;
          });
          row.appendChild(select);
        }
        const sw = el("label", "switch");
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = ts.enabled?.[tool.name] !== false;
        input.addEventListener("change", () => {
          ts.enabled[tool.name] = input.checked;
        });
        sw.appendChild(input);
        sw.appendChild(el("span", "switch-track"));
        row.appendChild(sw);
        permList.appendChild(row);
      }

      // ── truncation sliders ──
      const truncList = this.el("tools-trunc-list");
      clear(truncList);
      for (const tool of TOOL_META) {
        if (!tool.limit) continue;
        const value = Number(ts.outputLimits?.[tool.limit.key] ?? tool.limit.def);
        const row = el("div", "tool-row");
        const main = el("div", "slider-main");
        const head = el("div", "slider-head");
        head.appendChild(el("span", "switch-row-name", tool.label));
        const valueLabel = el("span", "slider-value", this.formatChars(value));
        head.appendChild(valueLabel);
        main.appendChild(head);
        const slider = el("input", "slider");
        slider.type = "range";
        slider.min = "2000";
        slider.max = "200000";
        slider.step = "2000";
        slider.value = String(value);
        slider.addEventListener("input", () => {
          valueLabel.textContent = this.formatChars(Number(slider.value));
          ts.outputLimits[tool.limit.key] = Number(slider.value);
        });
        slider.addEventListener("change", () => {
          ts.outputLimits[tool.limit.key] = Number(slider.value);
        });
        main.appendChild(slider);
        row.appendChild(main);
        truncList.appendChild(row);
      }

      // ── subagent tool switches（子代理的启停用上方工具列表里的 agent 开关；
      //    空列表 = 继承全部工具，“全部关闭”用哨兵条目表达）──
      const saList = this.el("subagent-tools-list");
      clear(saList);
      const parentTools = TOOL_META.filter((tool) => tool.name !== "agent");
      for (const tool of parentTools) {
        const row = el("div", "tool-row");
        const main = el("div", "switch-row-main");
        main.appendChild(el("span", "switch-row-name", tool.label));
        main.appendChild(el("span", "switch-row-desc", tool.desc));
        row.appendChild(main);
        const sw = el("label", "switch");
        const input = document.createElement("input");
        input.type = "checkbox";
        const configured = Array.isArray(ts.subagent?.tools) ? ts.subagent.tools : [];
        input.checked = configured.length === 0 || configured.includes(tool.name);
        input.addEventListener("change", () => {
          const current = Array.isArray(ts.subagent.tools) ? ts.subagent.tools : [];
          const set = new Set(
            current.length === 0 ? parentTools.map((t) => t.name) : current.filter((n) => n !== "(none)")
          );
          if (input.checked) set.add(tool.name);
          else set.delete(tool.name);
          // 服务端把空列表解释为"继承全部"，因此全部关闭时写入哨兵条目
          ts.subagent.tools = set.size === 0 ? ["(none)"] : Array.from(set);
        });
        sw.appendChild(input);
        sw.appendChild(el("span", "switch-track"));
        row.appendChild(sw);
        saList.appendChild(row);
      }
      this.renderMediaSettings();
    }

    // ── media model + 采样步数 + LoRA selection (settings.json → tools.media) ──
    //
    // 图像与视频分成两组，各自有自己的模型角色、采样步数与 LoRA 槽位。下拉的选项
    // 就是 models/ 目录里的文件（服务端实时枚举），没有内置清单、没有「默认」项，
    // 默认留空——留空时引擎按各自内置的默认文件名去 models/ 里找。
    renderMediaSettings() {
      const ts = this.draftToolSettings || this.toolSettings;
      if (!ts) return;
      const media = ts.media || (ts.media = {});
      // 迁移：旧版把所有 LoRA 存在一个共享的 `loras` 里。服务端仍会读它，但不删掉
      // 就会一直生效——即使用户在新下拉里清空也一样。所以一次性摊到两条链上。
      // 迁移：上一版只有一个共享的 `tokenizer_config`（本轮换成了每链
      // vocab/merges/config 三个角色）。留着它会让服务端的兼容分支继续生效，
      // 即使用户已经在新下拉里单独选过，所以一次性删掉——之后各链用自己的默认值，
      // 或用户在对应下拉里显式选择。
      if (typeof media.tokenizer_config === "string") delete media.tokenizer_config;
      if (Array.isArray(media.loras)) {
        const legacy = media.loras.filter((v) => typeof v === "string" && v);
        for (const chain of MEDIA_LORA_CHAINS) {
          const key = chain + "_loras";
          if (!Array.isArray(media[key]) || media[key].length === 0) media[key] = legacy.slice(0, 2);
        }
        delete media.loras;
      }
      const models = this.mediaModels;
      const files = (models && models.files) || [];
      for (const chain of MEDIA_CHAINS) {
        const list = this.el("media-" + chain + "-models-list");
        const loraList = this.el("media-" + chain + "-loras-list");
        const stepsList = this.el("media-" + chain + "-steps-list");
        const samplerList = this.el("media-" + chain + "-sampler-list");
        const schedulerList = this.el("media-" + chain + "-scheduler-list");
        if (list) {
          clear(list);
          if (!models) {
            list.appendChild(el("div", "settings-group-desc", "正在加载模型列表…"));
          } else {
            for (const role of MEDIA_ROLES[chain]) list.appendChild(this.mediaRoleRow(media, role, files));
          }
        }
        // 采样步数是模型之外的一项：不依赖 models/ 列表，也不依赖链是否已加载。
        if (stepsList) {
          clear(stepsList);
          stepsList.appendChild(this.mediaStepsRow(media, chain));
        }
        // 采样器 / 调度器：同样与模型文件无关，每条链一个下拉。
        if (samplerList && MEDIA_SAMPLERS[chain]) {
          clear(samplerList);
          samplerList.appendChild(this.mediaChoiceRow(media, MEDIA_SAMPLERS[chain]));
        }
        if (schedulerList && MEDIA_SCHEDULERS[chain]) {
          clear(schedulerList);
          schedulerList.appendChild(this.mediaChoiceRow(media, MEDIA_SCHEDULERS[chain]));
        }
        if (loraList) {
          clear(loraList);
          const key = chain + "_loras";
          if (!Array.isArray(media[key])) media[key] = [];
          for (let i = 0; i < 2; i++) {
            loraList.appendChild(this.mediaLoraRow(media, key, i, files));
          }
        }
      }
    }
    // 一个角色下拉：选项来自 models/ 的文件列表，默认空。
    mediaRoleRow(media, role, files) {
      const row = el("div", "tool-row");
      const main = el("div", "switch-row-main");
      main.appendChild(el("span", "switch-row-name", role.label));
      row.appendChild(main);
      const select = this.mediaFileSelect(media[role.key], files, "（留空）");
      select.addEventListener("change", () => {
        if (select.value) media[role.key] = select.value;
        else delete media[role.key];
      });
      row.appendChild(select);
      return row;
    }
    // 采样步数：一个数字输入。图像链（qwen image 2.1）默认 25 步，视频链
    // （minimax h3）默认 20 步；服务端每次生成都重读该值，保存后下一次生成生效。
    mediaStepsRow(media, chain) {
      const meta = MEDIA_STEPS[chain];
      const row = el("div", "tool-row");
      const main = el("div", "switch-row-main");
      main.appendChild(el("span", "switch-row-name", meta.label));
      main.appendChild(el("span", "switch-row-desc", meta.desc));
      row.appendChild(main);
      const input = el("input", "inline-number");
      input.type = "number";
      input.min = String(meta.min);
      input.max = String(meta.max);
      input.step = "1";
      const stored = Number(media[meta.key]);
      const value = Number.isFinite(stored) && stored >= meta.min && stored <= meta.max ? Math.round(stored) : meta.def;
      input.value = String(value);
      const commit = () => {
        let v = Math.round(Number(input.value));
        if (!Number.isFinite(v) || v < meta.min) v = meta.min;
        if (v > meta.max) v = meta.max;
        input.value = String(v);
        media[meta.key] = v;
      };
      input.addEventListener("change", commit);
      row.appendChild(input);
      return row;
    }
    // 采样器 / 调度器：一个下拉，选项是引擎移植并验证过的名字。
    mediaChoiceRow(media, meta) {
      const row = el("div", "tool-row");
      const main = el("div", "switch-row-main");
      main.appendChild(el("span", "switch-row-name", meta.label));
      row.appendChild(main);
      const select = el("select", "inline-select");
      for (const name of meta.options) {
        const o = el("option");
        o.value = name;
        o.textContent = name;
        select.appendChild(o);
      }
      const stored = typeof media[meta.key] === "string" && media[meta.key] ? media[meta.key] : meta.def;
      select.value = meta.options.includes(stored) ? stored : meta.def;
      select.addEventListener("change", () => {
        media[meta.key] = select.value;
      });
      row.appendChild(select);
      return row;
    }
    // 一个 LoRA 下拉：每链 2 个槽位（key = image_loras / video_loras）。
    mediaLoraRow(media, key, i, files) {
      const row = el("div", "tool-row");
      const main = el("div", "switch-row-main");
      main.appendChild(
        el("span", "switch-row-name", "LoRA " + (i + 1) + (i === 0 ? "（先应用）" : "（后应用）"))
      );
      row.appendChild(main);
      const select = this.mediaFileSelect(media[key][i], files, "（不使用）");
      select.addEventListener("change", () => {
        media[key][i] = select.value;
        media[key] = media[key].slice(0, 2);
      });
      row.appendChild(select);
      return row;
    }
    // 共享的下拉构造：空选项 + models/ 的每个文件（含后缀）。
    mediaFileSelect(value, files, emptyLabel) {
      const select = el("select", "inline-select");
      const none = el("option");
      none.value = "";
      none.textContent = emptyLabel;
      select.appendChild(none);
      const seen = new Set();
      for (const f of files) {
        if (seen.has(f.name)) continue;
        seen.add(f.name);
        const o = el("option");
        o.value = f.name;
        o.textContent = f.name;
        select.appendChild(o);
      }
      // 已保存但当前不在列表里的值（例如文件被移走）仍然保留可见，避免静默丢失。
      if (value && !seen.has(value)) {
        const o = el("option");
        o.value = value;
        o.textContent = value + "（当前不可用）";
        select.appendChild(o);
      }
      select.value = value || "";
      return select;
    }
    onMediaDownload(msg) {
      const status = this.el("media-download-status");
      const pct = Math.round((msg.progress || 0) * 100);
      if (status) {
        if (msg.status === "done") status.textContent = "下载完成，可刷新列表后选择该模型";
        else if (msg.status === "error") status.textContent = "下载失败：" + (msg.error || "");
        else status.textContent = "下载中… " + (pct > 0 ? pct + "%" : "");
      }
      if (msg.status === "done") void this.refreshMediaModels();
    }
    formatChars(n) {
      if (n >= 1000) return `${Math.round(n / 100) / 10}k 字符`;
      return `${n} 字符`;
    }

    // ------------------------------------------------------------------------
    // Hotkeys page
    // ------------------------------------------------------------------------
    renderHotkeys() {
      const list = this.el("hotkeys-list");
      clear(list);
      const src = this.draftHotkeys || Hotkeys.data;
      for (const meta of HOTKEY_META) {
        const row = el("div", "tool-row");
        const main = el("div", "switch-row-main");
        main.appendChild(el("span", "switch-row-name", meta.label));
        main.appendChild(el("span", "switch-row-desc", `默认 ${meta.def}`));
        row.appendChild(main);
        const kbd = el("button", "kbd-bind", src[meta.id] || "未绑定");
        kbd.type = "button";
        kbd.title = "点击修改快捷键";
        if (src[meta.id] !== meta.def) kbd.classList.add("custom");
        kbd.addEventListener("click", () => this.startHotkeyCapture(meta.id, kbd));
        row.appendChild(kbd);
        list.appendChild(row);
      }
    }
    /**
     * 进入快捷键捕获态。
     * - 再次点击同一个绑定框 = 取消捕获（避免双击后进入无提示的捕获态、吞掉所有按键）
     * - 点击其他位置时由构造器里的 mousedown 监听取消捕获
     * - 捕获结束时尽量原地恢复按钮而不是重建整个列表，保证后续点击事件不丢失
     */
    startHotkeyCapture(action, kbd) {
      if (this.hotkeyCapture && this.captureAction === action) {
        this.cancelHotkeyCapture();
        return;
      }
      this.cancelHotkeyCapture();
      this.captureAction = action;
      this.captureKbd = kbd;
      kbd.classList.add("capturing");
      kbd.textContent = "按下组合键…";
      this.hotkeyCapture = (combo, cancelled) => {
        const captureKbd = this.captureKbd;
        this.hotkeyCapture = null;
        this.captureAction = null;
        this.captureKbd = null;
        if (cancelled) {
          // 原地恢复；仅当按钮已不在文档中时才重建列表
          if (captureKbd && captureKbd.isConnected) {
            captureKbd.classList.remove("capturing");
            captureKbd.textContent = (this.draftHotkeys || Hotkeys.data)[action] || "未绑定";
          } else {
            this.renderHotkeys();
          }
          return;
        }
        for (const meta of HOTKEY_META) {
          if (meta.id !== action && (this.draftHotkeys || Hotkeys.data)[meta.id] === combo) {
            this.api.showBanner(`与「${meta.label}」冲突，请换一个组合键。`, "error");
            this.renderHotkeys();
            return;
          }
        }
        Hotkeys.data[action] = combo;
        // Save to draft; actual persistence happens on 保存
        if (this.draftHotkeys) this.draftHotkeys[action] = combo;
        this.renderHotkeys();
      };
    }
    cancelHotkeyCapture() {
      if (!this.hotkeyCapture) return;
      const finish = this.hotkeyCapture;
      this.hotkeyCapture = null;
      finish("", true);
    }

    // ------------------------------------------------------------------------
    // Appearance page
    // ------------------------------------------------------------------------
    renderThemeCards() {
      const grid = this.el("theme-list");
      clear(grid);
      let lastGroup = null;
      for (const theme of THEMES) {
        // 插入分组标题（深色 / 浅色）
        const group = theme.colors ? "dark" : "light";
        if (group !== lastGroup) {
          lastGroup = group;
          const label = group === "dark" ? "深色主题" : "浅色主题";
          const sec = el("div", "theme-section-title", label);
          grid.appendChild(sec);
        }
        const card = el("button", "theme-card");
        card.type = "button";
        card.dataset.theme = theme.id;
        // 将主题色注入卡片，让预览框使用对应颜色而非当前主题的颜色
        if (theme.colors) {
          card.style.setProperty("--accent", theme.colors.accent);
          card.style.setProperty("--bg", theme.colors.bg);
          card.style.setProperty("--bg-active", theme.colors.bgActive);
        }
        card.addEventListener("click", () => {
          const src = this.draftAppearance || Appearance.data;
          src.theme = theme.id;
          this.renderThemeCards();
          this.renderAppearance();
        });
        card.appendChild(el("span", "theme-swatch"));
        card.appendChild(el("span", "theme-name", theme.name));
        card.appendChild(el("span", "theme-hint", theme.hint));
        grid.appendChild(card);
      }
    }
    renderAppearance() {
      const src = this.draftAppearance || Appearance.data;
      for (const card of this.el("theme-list").querySelectorAll(".theme-card")) {
        card.classList.toggle("active", card.dataset.theme === src.theme);
      }
      this.el("appearance-radius").value = src.radius;
      this.el("appearance-density").value = src.density;
      this.el("appearance-motion").checked = !!src.motion;
      this.el("appearance-collapse-thinking").checked = !!src.collapseThinking;
      this.el("appearance-collapse-tools").checked = !!src.collapseTools;
    }

    // ------------------------------------------------------------------------
    // General page
    // ------------------------------------------------------------------------
    renderGeneral() {
      const src = this.draftPrefs || Prefs.data;
      this.el("general-status-tokens").checked = !!src.statusTokens;
    }

    // ------------------------------------------------------------------------
    // About page
    // ------------------------------------------------------------------------
    setAboutInfo(info) {
      this.aboutInfo = info || {};
      const rows = this.el("about-rows");
      clear(rows);
      const s = this.aboutInfo;
      const entries = [
        ["界面版本", `Phi Web UI · 协议 v${s.version ?? "1"}`],
        ["运行方式", "本地 HTTP + WebSocket，经 WebView2 渲染"],
        ["配置文件", "数据目录下 settings.json / webui-configs.json / models.json"],
        ["工具设置", "数据目录下 settings.json 的 tools 段"],
        ["界面偏好", "保存在本机（localStorage）：主题 / 快捷键 / 通用选项"]
      ];
      for (const [k, v] of entries) {
        const row = el("div", "about-row");
        row.appendChild(el("span", "about-key", k));
        row.appendChild(el("span", "about-value", v));
        rows.appendChild(row);
      }
    }
  };
  // ---- perf instrumentation (?perf=1 in the URL) -------------------------------
  // Records per-flush DOM cost and WS arrival timing so a stall can be
  // attributed to the network (no ws-arr marks during the gap) or to the
  // renderer (ws-arr marks keep coming but flush/patchBody ms explode).
  // Zero overhead when disabled: perfMark returns before touching the log.
  const PHI_PERF = typeof location !== "undefined" && new URLSearchParams(location.search).has("perf");
  const perfLog = [];
  function perfMark(tag, ms, extra) {
    if (!PHI_PERF) return;
    perfLog.push({ t: Math.round(performance.now()), e: Date.now(), tag, ms: Math.round(ms * 100) / 100, ...(extra || {}) });
  }
  if (PHI_PERF) {
    setInterval(() => {
      if (perfLog.length === 0) return;
      const latest = perfLog.slice(-14).map((m) => {
        let s = `${m.t}:${m.tag}`;
        if (m.ms != null) s += `(${m.ms}ms)`;
        if (m.len != null) s += ` len=${m.len}`;
        if (m.gap != null) s += ` gap=${m.gap}ms`;
        if (m.ids != null) s += ` ids=${m.ids}`;
        return s;
      }).join(" | ");
      console.info(`[perf] ${perfLog.length} marks since last dump :: ${latest}`);
      try {
        fetch("/api/perf", { method: "POST", body: JSON.stringify(perfLog) }).catch(() => {});
      } catch {}
      perfLog.length = 0;
    }, 2000);
  }

  // ---- patch render coalescer (DOM-free, unit-tested in tests/patch_coalescer_test.mjs) ----
  // Deltas arrive at the server's 30ms cadence; rebuilding the streaming
  // message's markdown + forced layout on every one pegs the renderer.
  // Coalesce DOM work to at most one render per window; state merging stays
  // per-patch so no data is dropped.
  // BEGIN PATCH_COALESCER
  const PATCH_RENDER_WINDOW_MS = 100;
  class PatchCoalescer {
    constructor(windowMs = 100) {
      this.windowMs = windowMs;
      this.dirty = false;
      this.lastFlush = -Infinity;
    }
    due(now) {
      return this.dirty && now - this.lastFlush >= this.windowMs;
    }
    offer(now) {
      this.dirty = true;
      return this.due(now);
    }
    flush(now) {
      const had = this.dirty;
      this.dirty = false;
      this.lastFlush = now;
      return had;
    }
  }
  // END PATCH_COALESCER

  // src/web/markdown.ts
  var INLINE_RE = /`([^`\n]+)`|\*\*([^*\n]+)\*\*|\*([^*\n]+)\*|~~([^~\n]+)~~|\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g;
  function renderInline(text) {
    const parts = [];
    let lastIndex = 0;
    for (const match of text.matchAll(INLINE_RE)) {
      const index = match.index ?? 0;
      if (index > lastIndex) {
        parts.push(escapeHtml(text.slice(lastIndex, index)));
      }
      const [, code, bold, italic, strike, linkText, linkUrl] = match;
      if (code !== void 0) {
        parts.push(`<code>${escapeHtml(code)}</code>`);
      } else if (bold !== void 0) {
        parts.push(`<strong>${escapeHtml(bold)}</strong>`);
      } else if (italic !== void 0) {
        parts.push(`<em>${escapeHtml(italic)}</em>`);
      } else if (strike !== void 0) {
        parts.push(`<s>${escapeHtml(strike)}</s>`);
      } else if (linkText !== void 0 && linkUrl !== void 0) {
        parts.push(
          `<a href="${escapeHtml(linkUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(linkText)}</a>`
        );
      }
      lastIndex = index + match[0].length;
    }
    if (lastIndex < text.length) {
      parts.push(escapeHtml(text.slice(lastIndex)));
    }
    return parts.join("");
  }
  function renderFence(lang, code) {
    const langLabel = lang.trim() || "code";
    return `<div class="pre-block"><div class="pre-header"><span>${escapeHtml(langLabel)}</span><button type="button" data-copy="1" aria-label="Copy code"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M5 15V5a2 2 0 0 1 2-2h10" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>copy</button></div><pre><code>${escapeHtml(code)}</code></pre></div>`;
  }
  function renderMarkdown(source) {
    const lines = source.replace(/\r\n/g, "\n").split("\n");
    const blocks = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      const fence = line.match(/^```([^\s`]*)/);
      if (fence) {
        const lang = fence[1];
        const code = [];
        i += 1;
        while (i < lines.length && !lines[i].trimStart().startsWith("```")) {
          code.push(lines[i]);
          i += 1;
        }
        i += 1;
        blocks.push(renderFence(lang, code.join("\n")));
        continue;
      }
      const heading = line.match(/^(#{1,4})\s+(.*)$/);
      if (heading) {
        const level = heading[1].length;
        blocks.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        i += 1;
        continue;
      }
      if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) {
        blocks.push("<hr/>");
        i += 1;
        continue;
      }
      if (line.startsWith(">")) {
        const quote = [];
        while (i < lines.length && lines[i].startsWith(">")) {
          quote.push(lines[i].slice(1).trimStart());
          i += 1;
        }
        blocks.push(`<blockquote>${renderInline(quote.join("\n"))}</blockquote>`);
        continue;
      }
      if (/^\s*[-*+]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
          items.push(`<li>${renderInline(lines[i].replace(/^\s*[-*+]\s+/, ""))}</li>`);
          i += 1;
        }
        blocks.push(`<ul>${items.join("")}</ul>`);
        continue;
      }
      if (/^\s*\d+[.)]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
          items.push(`<li>${renderInline(lines[i].replace(/^\s*\d+[.)]\s+/, ""))}</li>`);
          i += 1;
        }
        blocks.push(`<ol>${items.join("")}</ol>`);
        continue;
      }
      const paragraph = [];
      while (i < lines.length && lines[i].trim() !== "" && !/^\s*[-*+]\s+/.test(lines[i]) && !/^\s*\d+[.)]\s+/.test(lines[i]) && !lines[i].startsWith(">") && !lines[i].trimStart().startsWith("```") && !/^#{1,4}\s/.test(lines[i])) {
        paragraph.push(lines[i]);
        i += 1;
      }
      if (paragraph.length > 0) {
        blocks.push(`<p>${renderInline(paragraph.join("\n"))}</p>`);
        continue;
      }
      i += 1;
    }
    return blocks.join("");
  }

  // src/web/render.ts
  /**
   * Pick the element that hosts the blinking streaming caret: the LAST
   * top-level block, descending into the last <li> when the message ends with
   * a list. The previous implementation used
   * `querySelector("p:last-child, li:last-child, ...")` which returns the
   * FIRST match in document order — and `:last-child` matches the last <li> of
   * EVERY list — so any earlier list won and the caret landed mid-message.
   * An <hr> can't host an ::after caret, and an empty body has no target: both
   * fall back to appending an empty paragraph.
   */
  function pickStreamCursorTarget(blocks, makeFallback) {
    const kids = blocks.children;
    let last = kids.length > 0 ? kids[kids.length - 1] : null;
    if (last && (last.tagName === "UL" || last.tagName === "OL")) {
      const items = last.querySelectorAll("li");
      last = items.length > 0 ? items[items.length - 1] : last;
    }
    if (!last || last.tagName === "HR") {
      last = makeFallback();
      blocks.appendChild(last);
    }
    return last;
  }
  var TranscriptView = class {
    constructor(transcript) {
      this.messageNodes = /* @__PURE__ */ new Map();
      this.messageSignatures = /* @__PURE__ */ new Map();
      this.autoScroll = true;
      this.hasNewWhilePaused = false;
      this.scrollHint = null;
      this.lastSnapshot = null;
      this.transcript = transcript;
      this.inner = el("div", "transcript-inner");
      transcript.appendChild(this.inner);
      transcript.addEventListener("scroll", () => this.onScroll());
      transcript.addEventListener("click", (event) => this.onClick(event));
    }
    render(snapshot) {
      const messages = snapshot.messages;
      // index tool runs once per render: signatureOf/patchAssistantBody lookup
      // by toolCallId must not be O(runs) per tool call (quadratic over history)
      const runsById = new Map((snapshot.toolRuns ?? []).map((run) => [run.toolCallId, run]));
      const ids = new Set(messages.map((message) => message.id));
      for (const [id, node] of this.messageNodes) {
        if (!ids.has(id)) {
          node.remove();
          this.messageNodes.delete(id);
          this.messageSignatures.delete(id);
        }
      }
      let anchor = this.inner.firstChild;
      for (const message of messages) {
        let node = this.messageNodes.get(message.id);
        if (!node) {
          node = this.createMessageNode(message);
          this.messageNodes.set(message.id, node);
        } else if (node.dataset.role !== message.role) {
          // role changed (e.g. assistant message reused its ID for a compactionSummary
          // after compaction); update the CSS class so the correct box styles apply
          node.className = `msg msg-${message.role}`;
          node.dataset.role = message.role;
        }
        if (anchor === null || node !== anchor) {
          if (anchor !== null) {
            this.inner.insertBefore(node, anchor);
          } else {
            this.inner.appendChild(node);
          }
        }
        anchor = node.nextSibling;
        const signature = this.signatureOf(message, snapshot, runsById);
        if (this.messageSignatures.get(message.id) !== signature) {
          this.messageSignatures.set(message.id, signature);
          this.updateMessage(node, message, snapshot, runsById);
        }
      }
      while (anchor !== null) {
        const next = anchor.nextSibling;
        anchor.remove();
        anchor = next;
      }
      const isStreaming = snapshot.isStreaming || snapshot.phase === "streaming" || snapshot.phase === "compacting";
      const wasStreaming = this.lastSnapshot?.isStreaming === true;
      this.lastSnapshot = snapshot;
      if (isStreaming && !wasStreaming) {
        this.hasNewWhilePaused = false;
      }
      if (this.autoScroll) {
        this.scrollToBottom();
      } else if (isStreaming) {
        this.hasNewWhilePaused = true;
        this.updateScrollHint();
      }
    }
    /**
     * Update only the message nodes named in `ids` (stream_patch path): the
     * full render() walk is O(history) per tick, patches must be O(change).
     */
    updateMessages(ids, snapshot, runsById) {
      let changed = false;
      for (const id of ids) {
        const message = snapshot.messages.find((m) => m.id === id);
        const node = this.messageNodes.get(id);
        if (!message || !node) continue;
        // Sync role class in case it changed (e.g. compactionSummary reusing an ID)
        if (node.dataset.role !== message.role) {
          node.className = `msg msg-${message.role}`;
          node.dataset.role = message.role;
        }
        const signature = this.signatureOf(message, snapshot, runsById);
        if (this.messageSignatures.get(id) !== signature) {
          this.messageSignatures.set(id, signature);
          this.updateMessage(node, message, snapshot, runsById);
          changed = true;
        }
      }
      if (changed && this.autoScroll) {
        this.scrollToBottom();
      }
    }
    clearMessages() {
      clear(this.inner);
      this.messageNodes.clear();
      this.messageSignatures.clear();
      this.lastSnapshot = null;
      this.hasNewWhilePaused = false;
      this.updateScrollHint();
    }
    scrollToBottom() {
      const t0 = performance.now();
      this.transcript.scrollTop = this.transcript.scrollHeight;
      perfMark("scroll", performance.now() - t0, {});
    }
    onScroll() {
      const node = this.transcript;
      const nearBottom = node.scrollTop + node.clientHeight >= node.scrollHeight - 48;
      if (nearBottom !== this.autoScroll) {
        this.autoScroll = nearBottom;
        if (nearBottom) this.hasNewWhilePaused = false;
        this.updateScrollHint();
      }
    }
    updateScrollHint() {
      if (!this.hasNewWhilePaused && this.scrollHint) {
        this.scrollHint.remove();
        this.scrollHint = null;
        return;
      }
      if (this.hasNewWhilePaused && !this.scrollHint) {
        const hint = el("button", "scroll-hint", `\u2193 ${t("new_activity")}`);
        hint.type = "button";
        hint.addEventListener("click", () => {
          this.autoScroll = true;
          this.hasNewWhilePaused = false;
          this.scrollToBottom();
          this.updateScrollHint();
        });
        // inside the transcript container: appending to document.body left
        // a stale hint behind after panel switches
        this.transcript.appendChild(hint);
        this.scrollHint = hint;
      }
    }
    onClick(event) {
      const target = event.target;
      const copyButton = target.closest("[data-copy]");
      if (copyButton) {
        const block = copyButton.closest(".pre-block");
        const code = block?.querySelector("code");
        if (code?.textContent !== void 0) {
          void navigator.clipboard.writeText(code.textContent);
          const label = copyButton.textContent;
          copyButton.textContent = t("copied");
          setTimeout(() => {
            copyButton.textContent = label;
          }, 1200);
        }
        return;
      }
      const link = target.closest("a[href]");
      if (link && link.target === "_blank") {
        event.stopPropagation();
      }
    }
    // ------------------------------------------------------------------------
    // Message construction
    // ------------------------------------------------------------------------
    createMessageNode(message) {
      return el("div", `msg msg-${message.role}`);
    }
    /**
     * Compact signature of everything a message renders (meta + body). When it
     * is unchanged across snapshots the message node is left alone, so history
     * isn't re-parsed and re-flowed on every streaming tick.
     */
    signatureOf(message, snapshot, runsById) {
      const parts = [
        message.role,
        message.model?.name ?? "",
        message.model?.provider ?? "",
        message.model?.id ?? "",
        message.status ?? "",
        message.errorMessage ?? "",
        String(message.usage?.totalTokens ?? 0),
        String(message.usage?.input ?? 0),
        String(message.usage?.output ?? 0),
        String(message.usage?.cacheRead ?? 0),
        String(message.usage?.cacheWrite ?? 0),
        String(message.summary ?? ""),
        String(message.tokensBefore ?? 0),
        String(message.tokensAfter ?? 0),
        String(message.description ?? ""),
        message.isError ? "1" : "0"
      ];
      for (const content of message.content) {
        if (content.type === "text") {
          parts.push(`t${content.text.length}`);
        } else if (content.type === "thinking") {
          parts.push(`h${content.thinking.length}:${content.redacted ? "r" : ""}`);
        } else if (content.type === "toolCall") {
          const run = runsById
            ? runsById.get(content.id)
            : (snapshot.toolRuns ?? []).find((candidate) => candidate.toolCallId === content.id);
          // A partial output is identified by its LENGTH *and its TAIL*, never by
          // its length alone: media progress lines ("采样 3/30…", "sampling 4/30…")
          // all have the same length, so a length-only signature made the card
          // repaint only when the digit count changed — i.e. at step 0 and again
          // at step 10 — and silently dropped every step in between. Taking the
          // tail is O(1) (V8 slices) and every progress line changes at its end.
          const output = run?.output ?? "";
          parts.push(
            `c${content.id}:${content.name}:${run?.status ?? ""}:${run?.isError ? "1" : "0"}:${output.length}:${output.slice(-24)}`
          );
        } else if (content.type === "image") {
          parts.push(`i${content.mimeType}`);
        }
      }
      return parts.join("");
    }
    updateMessage(node, message, snapshot, runsById) {
      const meta = this.buildMeta(message);
      let body = node.querySelector(".msg-body");
      if (!body) {
        body = el("div", "msg-body");
        node.appendChild(meta);
        node.appendChild(body);
      } else {
        const existing = node.querySelector(".msg-meta");
        if (existing) existing.replaceWith(meta);
      }
      if (message.role === "user") {
        body.replaceChildren(this.buildUserBody(message));
      } else if (message.role === "compactionSummary") {
        body.replaceChildren(this.buildCompactionBody(message));
      } else if (message.role === "subagentSummary") {
        body.replaceChildren(this.buildSubagentBody(message));
      } else {
        const t0 = performance.now();
        this.patchAssistantBody(body, message, snapshot, runsById);
        if (PHI_PERF) perfMark("patchBody", performance.now() - t0, { len: JSON.stringify(message.content).length });
      }
    }
    buildUserBody(message) {
      const body = el("div", "msg-body");
      const text = message.content.filter((content) => content.type === "text").map((content) => content.text).join("\n");
      if (text) {
        body.appendChild(document.createTextNode(text));
      }
      for (const content of message.content) {
        if (content.type === "image" && content.data) {
          const wrap = el("div", "msg-image");
          const img = document.createElement("img");
          img.src = `data:${content.mimeType};base64,${content.data}`;
          img.alt = t("attachment_image");
          img.loading = "lazy";
          wrap.appendChild(img);
          body.appendChild(wrap);
        }
      }
      return body;
    }
    /** /compact 压缩后的上下文摘要（含压缩前后 token 数），对用户可见 */
    /** 子代理完成后的摘要：独立的一个框，与工具调用 / 上下文压缩的框都不同；
     * 颜色用窗口主题的工具调用框颜色（--tool-bg） */
    buildSubagentBody(message) {
      const body = el("div", "msg-body");
      const desc = message.description || "";
      const head = el("div", "subagent-head");
      head.textContent = message.isError ? t("subagent_failed", desc) : t("subagent_finished", desc);
      body.appendChild(head);
      const text = message.content
        .filter((content) => content.type === "text")
        .map((content) => content.text)
        .join("\n");
      if (text) {
        const prose = el("div", "prose");
        prose.innerHTML = renderMarkdown(text);
        body.appendChild(prose);
      }
      return body;
    }
    buildCompactionBody(message) {
      const body = el("div", "msg-body");
      const tokensLine = el("div", "compaction-tokens");
      tokensLine.appendChild(document.createTextNode(t("compaction_tokens", message.tokensBefore ?? 0)));
      if (message.tokensAfter > 0) {
        const sep = document.createTextNode("  |  ");
        tokensLine.appendChild(sep);
        tokensLine.appendChild(document.createTextNode(t("compaction_tokens_after", message.tokensAfter)));
      }
      body.appendChild(tokensLine);
      const prose = el("div", "prose");
      prose.innerHTML = renderMarkdown(message.summary ?? "");
      body.appendChild(prose);
      return body;
    }
    buildMeta(message) {
      const meta = el("div", "msg-meta");
      const roleLabel = message.role === "user" ? t("you") : message.role === "assistant" ? t("phi") : message.role === "compactionSummary" ? t("compaction_role") : message.role === "subagentSummary" ? t("subagent_role") : t("tool");
      meta.appendChild(el("span", "msg-role", roleLabel));
      if (message.role === "assistant" && message.model?.name) {
        const model = el("span", "msg-model", message.model.name);
        model.title = `${message.model.provider}/${message.model.id}`;
        meta.appendChild(model);
      }
      if (message.status === "error" || message.status === "aborted") {
        const statusText = message.status === "error" ? "\u9519\u8BEF" : "\u5DF2\u4E2D\u65AD";
        meta.appendChild(el("span", "msg-error-tag", statusText));
      }
      if (message.usage && message.usage.totalTokens > 0) {
        const usage = el("span", "msg-usage");
        usage.textContent = `\u2191${formatTokens(message.usage.input)} \u2193${formatTokens(message.usage.output)}`;
        usage.title = `${t("input_tokens", message.usage.input)} \xB7 ${t("output_tokens", message.usage.output)}`;
        meta.appendChild(usage);
      }
      return meta;
    }
    /**
     * Patch the assistant body in place instead of rebuilding it.
     *
     * Snapshots arrive every ~30ms while streaming; rebuilding the body from
     * scratch on each one would reset the thinking box's inner scroll position
     * (only the first lines of a growing thought stay visible) and flicker the
     * whole message. Reused nodes are updated in place; a node is only replaced
     * when the node kind at that position changed.
     */
    patchAssistantBody(body, message, snapshot, runsById) {
      const isStreamingTurn = snapshot.isStreaming || snapshot.phase === "streaming";
      const streaming = message.status === "streaming";
      const existing = Array.from(body.children);
      let cursor = 0;
      for (const content of message.content) {
        if (content.type !== "thinking") continue;
        const redacted = content.redacted === true;
        const current = existing[cursor];
        if (current instanceof HTMLDetailsElement && current.classList.contains("thinking") && current.dataset.redacted === (redacted ? "1" : "0")) {
          this.updateThinking(current, content.thinking, streaming);
        } else {
          const fresh = this.buildThinking(content.thinking, redacted);
          if (current) {
            body.replaceChild(fresh, current);
            existing[cursor] = fresh;
          } else {
            body.appendChild(fresh);
          }
        }
        cursor++;
      }
      const textParts = [];
      for (const content of message.content) {
        if (content.type === "text") {
          textParts.push(content.text);
        }
      }
      const totalText = textParts.join("\n\n");
      if (totalText) {
        let blocks;
        const current = existing[cursor];
        if (current instanceof HTMLElement && current.classList.contains("prose")) {
          blocks = current;
        } else {
          blocks = el("div", "prose");
          if (current) {
            body.replaceChild(blocks, current);
            existing[cursor] = blocks;
          } else {
            body.appendChild(blocks);
          }
        }
        blocks.innerHTML = renderMarkdown(totalText);
        if (streaming) {
          pickStreamCursorTarget(blocks, () => el("p", "stream-cursor", "")).classList.add("stream-cursor");
        }
        cursor++;
      }
      for (const content of message.content) {
        if (content.type !== "toolCall") continue;
        const run = runsById
          ? runsById.get(content.id)
          : (snapshot.toolRuns ?? []).find((candidate) => candidate.toolCallId === content.id);
        const isLastMessage = snapshot.messages[snapshot.messages.length - 1]?.id === message.id;
        const defaultStatus = isLastMessage && isStreamingTurn ? "running" : "complete";
        const current = existing[cursor];
        const fresh = this.buildToolCard(content, run, defaultStatus);
        // 同一个工具调用沿用用户当前的展开/折叠（含手动的）；位置上的节点是
        // 别的工具卡时不能继承，否则折叠偏好的默认值会被邻居覆盖
        if (current instanceof HTMLDetailsElement && current.classList.contains("tool-card") &&
            current.dataset.toolCallId === content.id) {
          fresh.open = current.open;
        }
        if (current) {
          body.replaceChild(fresh, current);
          existing[cursor] = fresh;
        } else {
          body.appendChild(fresh);
        }
        cursor++;
      }
      if (message.errorMessage && (message.status === "error" || message.status === "aborted")) {
        const error = el("div", "msg-error");
        error.textContent = message.errorMessage;
        error.style.color = "var(--red)";
        error.style.marginTop = "8px";
        error.style.fontSize = "12.5px";
        const current = existing[cursor];
        if (current) {
          body.replaceChild(error, current);
          existing[cursor] = error;
        } else {
          body.appendChild(error);
        }
        cursor++;
      }
      while (cursor < existing.length) {
        existing[cursor].remove();
        cursor++;
      }
    }
    buildThinking(thinking, redacted) {
      const details = el("details", "thinking");
      // 新加载（含重启/重新打开会话）时按 UI 设置里的「折叠思考」决定开合；
      // 节点被复用时保留用户手动开合的状态（updateThinking 不会重建节点）
      details.open = !Appearance.collapseThinking();
      details.dataset.redacted = redacted ? "1" : "0";
      const summary = el("summary");
      summary.innerHTML = this.thinkingSummaryHtml(redacted, thinking.length);
      const body = el("div", "thinking-body", thinking);
      body.addEventListener("scroll", () => {
        const nearBottom = body.scrollTop + body.clientHeight >= body.scrollHeight - 24;
        details.dataset.pinned = nearBottom ? "true" : "false";
      });
      details.appendChild(summary);
      details.appendChild(body);
      return details;
    }
    /**
     * Update an existing thinking node in place. Reusing the node keeps its
     * open state and inner scroll position across streaming rebuilds.
     */
    updateThinking(details, thinking, streaming) {
      const body = details.querySelector(".thinking-body");
      if (body && body.textContent !== thinking) {
        const t0 = performance.now();
        body.textContent = thinking;
        perfMark("think-set", performance.now() - t0, { len: thinking.length });
      }
      const summary = details.querySelector("summary");
      if (summary) {
        summary.innerHTML = this.thinkingSummaryHtml(details.dataset.redacted === "1", thinking.length);
      }
      if (streaming && details.dataset.pinned !== "false" && body) {
        body.scrollTop = body.scrollHeight;
      }
    }
    thinkingSummaryHtml(redacted, chars) {
      const redactedText = redacted ? t("thinking_redacted") : "";
      return `<span class="caret">${icons.chevronRight}</span><span>${t("thinking")}${redactedText} \xB7 ${t("thinking_chars", chars)}</span>`;
    }
    buildToolCard(toolCall, run, defaultStatus = "complete") {
      const details = el("details", "tool-card");
      // 新加载（含重启/重新打开会话）时按 UI 设置里的「折叠工具调用」决定开合，
      // 不再看 run 的状态：默认展开时运行中/已完成都是展开的。会话内用户手动
      // 开合的状态由 patchAssistantBody 在复用节点时保留（fresh.open = current.open）
      details.open = !Appearance.collapseTools();
      // 卡片自带 toolCallId：重建时只有同一个工具调用才允许继承它当前的开合状态，
      // 否则位置错位（节点被换到别的工具卡上）会让新卡凭空继承邻卡的状态
      details.dataset.toolCallId = toolCall.id;
      const status = run?.status ?? defaultStatus;
      details.dataset.status = status;
      const summary = el("summary", "tool-head");
      const statusIcon = el("span", "tool-status");
      if (status === "running") {
        statusIcon.innerHTML = icons.loader;
      } else if (status === "error") {
        statusIcon.innerHTML = icons.alert;
      } else {
        statusIcon.innerHTML = icons.check;
      }
      summary.appendChild(statusIcon);
      summary.appendChild(el("span", "tool-name", toolCall.name));
      const argsText = this.previewArgs(toolCall.arguments);
      if (argsText) {
        summary.appendChild(el("span", "tool-args", argsText));
      }
      summary.appendChild(el("span", "tool-callid", toolCall.id.slice(0, 8)));
      const caret = el("span", "tool-caret");
      caret.innerHTML = icons.chevronRight;
      summary.appendChild(caret);
      details.appendChild(summary);
      const body = el("div", "tool-body");
      if (run && run.args !== void 0) {
        const argsPre = el("div", "tool-args-pre");
        argsPre.textContent = JSON.stringify(run.args, null, 2);
        body.appendChild(argsPre);
      }
      if (run?.output) {
        const output = el("div", "tool-output", run.output);
        body.appendChild(output);
      }
      if (run?.isError) {
        const errorText = el("div", "tool-error-text", t("tool_failed"));
        body.appendChild(errorText);
      }
      details.appendChild(body);
      return details;
    }
    previewArgs(args) {
      if (typeof args === "string") return args;
      if (args === null || args === void 0) return "";
      try {
        const json = JSON.stringify(args);
        return json.length > 80 ? `${json.slice(0, 77)}\u2026` : json;
      } catch {
        return String(args);
      }
    }
  };

  // src/web/main.ts
  var WsClient = class {
    constructor(onMessage, onStatus) {
      this.nextId = 0;
      this.pending = /* @__PURE__ */ new Map();
      this.reconnectAttempt = 0;
      // Only the newest socket may drive connection state. A superseded
      // socket's late "close" used to schedule ANOTHER reconnect while a fresh
      // socket was still CONNECTING; that next attempt replaced this.ws and the
      // half-open socket was dropped WITHOUT being closed, so it stayed
      // registered on the server and kept receiving every broadcast. A handful
      // of those piling up is what turned a single stall into a connection-lost
      // spiral. Every handler is generation-guarded instead.
      this.generation = 0;
      this.reconnectTimer = 0;
      this.probeTimer = 0;
      this.onMessage = onMessage;
      this.onStatus = onStatus;
    }
    /** settle every in-flight request (socket replaced or closed) */
    failPending(reason) {
      for (const resolve of this.pending.values()) {
        resolve({ kind: "error", error: reason });
      }
      this.pending.clear();
    }
    connect() {
      const gen = ++this.generation;
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = 0;
      }
      if (this.probeTimer) {
        clearTimeout(this.probeTimer);
        this.probeTimer = 0;
      }
      // exactly one live socket: cancel pending retries and drop the previous
      // socket before opening the next one
      const previous = this.ws;
      if (previous) {
        this.ws = null;
        this.failPending(t("connection_lost"));
        try {
          previous.close();
        } catch {}
      }
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${protocol}//${location.host}/ws`);
      this.ws = ws;
      let lastMsgT = 0;
      ws.addEventListener("open", () => {
        if (gen !== this.generation) return;
        this.reconnectAttempt = 0;
        this.onStatus(true);
      });
      ws.addEventListener("message", (event) => {
        if (gen !== this.generation) return;
        const raw = String(event.data);
        const now = performance.now();
        if (PHI_PERF && lastMsgT > 0 && now - lastMsgT > 800) perfMark("ws-gap", 0, { gap: Math.round(now - lastMsgT) });
        lastMsgT = now;
        let message;
        try {
          const t0 = now;
          message = JSON.parse(raw);
          if (PHI_PERF) perfMark("ws-arr", performance.now() - t0, { len: raw.length });
        } catch {
          return;
        }
        if (message.type === "response") {
          const resolve = this.pending.get(message.id);
          if (!resolve) return;
          this.pending.delete(message.id);
          if (message.ok) {
            resolve(message.result);
          } else {
            resolve({ kind: "error", error: message.error.message });
          }
          return;
        }
        this.onMessage(message);
      });
      ws.addEventListener("close", () => {
        if (gen !== this.generation) return;  // superseded socket: ignore
        this.ws = null;
        this.onStatus(false);
        this.failPending(t("connection_lost"));
        this.scheduleReconnect();
      });
      ws.addEventListener("error", () => {
        if (gen !== this.generation) return;
        try {
          ws.close();
        } catch {}
      });
    }
    scheduleReconnect() {
      // never give up: the shell keeps the HTTP server running even while a
      // long agent run pegs the CPU, so a failed attempt is transient. A retry
      // cap turned a temporary stall into a permanent "dead" UI that only a
      // manual reload fixed. The capped backoff (15s) + immediate-open probe
      // below means a healthy server is picked up within one interval.
      const delay = Math.min(1e3 * 2 ** this.reconnectAttempt, 15e3);
      this.reconnectAttempt += 1;
      // at most one retry timer and one probe timer: every close used to queue a
      // fresh pair, so a flapping link could fire several connects at once and
      // leak the losers
      if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = 0;
        if (this.ws?.readyState === WebSocket.OPEN) return;
        this.connect();
      }, delay);
      // parallel probe: a WebSocket close event also fires when the server
      // drops a broken connection while a NEW one would connect fine — don't
      // wait out the whole backoff in that case.
      if (this.probeTimer) clearTimeout(this.probeTimer);
      this.probeTimer = setTimeout(() => {
        this.probeTimer = 0;
        if (this.ws?.readyState === WebSocket.OPEN) return;
        fetch(`${location.origin}/api/settings`, { cache: "no-store" }).then((res) => {
          if (res.ok && this.ws?.readyState !== WebSocket.OPEN) this.connect();
        }).catch(() => {});
      }, Math.min(delay, 1500));
    }
    send(request) {
      const id = `${++this.nextId}`;
      return new Promise((resolve) => {
        let timer = 0;
        const settle = (value) => {
          clearTimeout(timer);
          this.pending.delete(id);
          resolve(value);
        };
        this.pending.set(id, settle);
        // requests must not hang forever when the link drops between the
        // OPEN check and send(): pending entries are cleaned on "close", but
        // a stuck connection can stay half-open indefinitely.
        timer = setTimeout(() => {
          this.pending.delete(id);
          resolve({ kind: "error", error: t("request_timeout") });
        }, 12e4);
        const ws = this.ws;
        if (!ws || ws.readyState !== WebSocket.OPEN) {
          this.pending.delete(id);
          clearTimeout(timer);
          resolve({ kind: "error", error: t("not_connected") });
          return;
        }
        ws.send(JSON.stringify({ ...request, id }));
      });
    }
  };
  var UNSUPPORTED_IN_WEB = /* @__PURE__ */ new Set([
    "import",
    "share",
    "fork",
    "clone",
    "tree",
    "trust",
    "scoped-models",
    "changelog",
    "quit"
  ]);
  var App = class {
    constructor() {
      this.state = {
        connected: false,
        session: null,
        models: [],
        skills: [],
        commands: [],
        sessions: [],
        settings: null,
        configs: null
      };
      this.configPending = true;
      this.$ = {
        transcript: document.getElementById("transcript"),
        emptyState: document.getElementById("empty-state"),
        emptySub: document.getElementById("empty-sub"),
        sessionName: document.getElementById("session-name"),
        cwd: document.getElementById("cwd"),
        cwdText: document.getElementById("cwd-text"),
        phaseBadge: document.getElementById("phase-badge"),
        phaseBadgeText: document.getElementById("phase-badge-text"),
        thinkingBtn: document.getElementById("thinking-btn"),
        thinkingLabel: document.getElementById("thinking-label"),
        thinkingMenu: document.getElementById("thinking-menu"),
        configBtnFooter: document.getElementById("config-btn-footer"),
        configLabel: document.getElementById("config-label"),
        configMenu: document.getElementById("config-menu"),
        skillsDot: document.getElementById("skills-dot"),
        sidePanel: document.getElementById("side-panel"),
        paneSkills: document.getElementById("pane-skills"),
        paneSessions: document.getElementById("pane-sessions"),
        skillsList: document.getElementById("skills-list"),
        sessionsList: document.getElementById("sessions-list"),
        skillsBtn: document.getElementById("skills-btn"),
        newSessionBtn: document.getElementById("new-session-btn"),
        configBtn: document.getElementById("config-btn"),
        configDot: document.getElementById("config-dot"),
        settingsOverlay: document.getElementById("settings-overlay"),
        settingsMenu: document.getElementById("settings-menu"),
        settingsContent: document.getElementById("settings-content"),
        settingsClose: document.getElementById("settings-close"),
        settingsResetAppearance: null,
        slashBtn: document.getElementById("slash-btn"),
        commandMenu: document.getElementById("command-menu"),
        paletteOverlay: document.getElementById("palette-overlay"),
        paletteInput: document.getElementById("palette-input"),
        paletteResults: document.getElementById("palette-results"),
        input: document.getElementById("input"),
        sendBtn: document.getElementById("send-btn"),
        abortBtn: document.getElementById("abort-btn"),
        attachBtn: document.getElementById("attach-btn"),
        fileInput: document.getElementById("file-input"),
        attachments: document.getElementById("attachments"),
        cwdDialog: document.getElementById("cwd-dialog"),
        cwdCurrent: document.getElementById("cwd-current"),
        cwdInput: document.getElementById("cwd-input"),
        cwdConfirm: document.getElementById("cwd-confirm"),
        cwdCancel: document.getElementById("cwd-cancel"),
        cwdBrowse: document.getElementById("cwd-browse"),
        cwdError: document.getElementById("cwd-error"),
        imageLightbox: document.getElementById("image-lightbox"),
        lightboxImage: document.getElementById("lightbox-image"),
        lightboxCaption: document.getElementById("lightbox-caption"),
        lightboxClose: document.getElementById("lightbox-close"),
        composerSuggest: document.getElementById("composer-suggest"),
        composerRow: document.querySelector(".composer-row"),
        statusModel: document.getElementById("status-model"),
        statusThinking: document.getElementById("status-thinking"),
        statusPhase: document.getElementById("status-phase"),
        statusTokens: document.getElementById("status-tokens"),
        statusTokensSep: document.getElementById("status-tokens-sep"),
        banner: document.getElementById("banner")
      };
      this.suggestIndex = 0;
      this.suggestItems = [];
      this.attachments = [];
      this.pendingSnapshot = null;
      this.renderQueued = false;
      /** path → data URL, kept while an image attachment is pending (for previews). */
      this.attachmentPreviews = /* @__PURE__ */ new Map();
      // ------------------------------------------------------------------------
      // Commands (client-side dispatch table + server passthrough)
      // ------------------------------------------------------------------------
      /**
       * Builtin commands handled directly in the web UI. Each handler either
       * performs the action or returns false to fall through to the normal flow.
       */
      this.builtinCommands = {
        settings: () => {
          this.settingsPanel.open("api");
          return true;
        },
        model: () => {
          this.settingsPanel.open("api");
          return true;
        },
        hotkeys: () => {
          this.settingsPanel.open("hotkeys");
          return true;
        },
        thinking: () => {
          this.$.thinkingBtn.click();
          return true;
        },
        compact: () => {
          this.compact();
          return true;
        },
        new: () => {
          this.newSession();
          return true;
        },
        copy: () => {
          this.copyLastAssistantMessage();
          return true;
        },
        session: () => {
          this.showSessionInfo();
          return true;
        }
      };
      this.transcript = new TranscriptView(this.$.transcript);
      this.panels = new Panels(
        {
          setThinking: (level) => this.sendThinking(level),
          activateConfig: (name) => this.settingsPanel.activate(name),
          openSession: (sessionId) => this.openSession(sessionId),
          reloadResources: () => this.reloadResources(),
          deleteSkill: (filePath) => this.deleteSkill(filePath),
          deleteSession: (sessionId) => this.deleteSession(sessionId),
          listSessions: () => this.fetchSessions()
        },
        {
          thinkingBtn: this.$.thinkingBtn,
          thinkingLabel: this.$.thinkingLabel,
          thinkingMenu: this.$.thinkingMenu,
          configBtnFooter: this.$.configBtnFooter,
          configLabel: this.$.configLabel,
          configMenu: this.$.configMenu,
          skillsDot: this.$.skillsDot,
          sidePanel: this.$.sidePanel,
          paneSkills: this.$.paneSkills,
          paneSessions: this.$.paneSessions,
          skillsList: this.$.skillsList,
          sessionsList: this.$.sessionsList
        }
      );
      this.palette = new CommandPalette(this.$.paletteOverlay, this.$.paletteInput, this.$.paletteResults);
      this.settingsPanel = new SettingsPanel(
        {
          fetchConfigs: () => this.wsSend("list_configs"),
          saveConfig: (profile) => this.wsSend("save_config", { profile }),
          deleteConfig: (name) => this.wsSend("delete_config", { name }),
          activateConfig: (name) => this.wsSend("activate_config", { name }),
          getToolSettings: () => this.wsSend("get_tool_settings"),
          saveToolSettings: (settings) => this.wsSend("save_tool_settings", { settings }),
          listMediaModels: () => this.wsSend("list_media_models"),
          saveMediaSettings: (media) => this.wsSend("save_media_settings", { media }),
          downloadMediaModel: (id) => this.wsSend("download_media_model", { id }),
          applyToolSettings: () => this.applyToolSettings(),
          getUiSettings: () => this.wsSend("get_ui_settings"),
          saveUiSettings: (settings) => this.wsSend("save_ui_settings", { settings }),
          showBanner: (message, kind) => this.showBanner(message, kind)
        },
        {
          overlay: this.$.settingsOverlay,
          menu: this.$.settingsMenu,
          content: this.$.settingsContent,
          closeBtn: this.$.settingsClose
        }
      );
      this.settingsPanel.onConfigsChange = (result) => {
        this.$.configDot.hidden = result.active === null;
        this.state.configs = result;
        this.panels.setConfig(result.active, result.profiles);
      };
      this.settingsPanel.onActivated = (name) => {
        this.showBanner(t("config_activated", name), "info");
        void this.fetchModels();
      };
      this.settingsPanel.onSaved = () => {
        this.showBanner(t("config_saved"), "info");
      };
      this.ws = new WsClient(
        (message) => this.onServerMessage(message),
        (connected) => this.onConnectionStatus(connected)
      );
      this.wireEvents();
      this.ws.connect();
    }
    // ------------------------------------------------------------------------
    // Wiring
    // ------------------------------------------------------------------------
    wireEvents() {
      this.$.skillsBtn.addEventListener("click", () => {
        this.panels.togglePanel();
        if (this.panels.isPanelOpen()) this.fetchSessions();
      });
      this.$.configBtn.addEventListener("click", () => this.settingsPanel.toggle());
      this.$.newSessionBtn.addEventListener("click", () => this.newSession());
      this.$.slashBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        this.toggleCommandMenu();
      });
      document.addEventListener("click", () => this.hideCommandMenu());
      document.getElementById("panel-close")?.addEventListener("click", () => this.panels.togglePanel());
      document.getElementById("skills-reload")?.addEventListener("click", () => this.reloadResources());
      document.getElementById("sessions-refresh")?.addEventListener("click", () => this.fetchSessions());
      for (const id of ["panel-close", "skills-reload", "sessions-refresh"]) {
        const button = document.getElementById(id);
        if (!button) continue;
        button.replaceChildren();
        button.insertAdjacentHTML("beforeend", this.iconSvg(id === "panel-close" ? "close" : "refresh"));
      }
      this.$.sessionName.addEventListener("click", () => this.startNameEdit());
      this.$.input.addEventListener("input", () => {
        this.autoGrow();
        this.updateSuggestions();
      });
      this.$.input.addEventListener("keydown", (event) => this.onInputKeyDown(event));
      this.$.input.addEventListener("blur", () => this.hideSuggestions());
      this.$.sendBtn.addEventListener("click", () => this.sendInput());
      this.$.abortBtn.addEventListener("click", () => this.abort());
      this.$.attachBtn.addEventListener("click", () => this.$.fileInput.click());
      this.$.fileInput.addEventListener("change", () => this.onFilesSelected(this.$.fileInput.files));
      this.$.cwd.addEventListener("click", () => this.openCwdDialog());
      this.$.cwdConfirm.addEventListener("click", () => this.confirmCwd());
      this.$.cwdCancel.addEventListener("click", () => this.closeCwdDialog());
      this.$.cwdBrowse.addEventListener("click", () => this.browseDirectory());
      this.$.cwdInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          this.confirmCwd();
        } else if (event.key === "Escape") {
          this.closeCwdDialog();
        }
      });
      this.$.cwdDialog.addEventListener("mousedown", (event) => {
        if (event.target === this.$.cwdDialog) {
          this.closeCwdDialog();
        }
      });
      this.$.lightboxClose.title = t("image_preview_close");
      this.$.lightboxClose.addEventListener("click", () => this.closeImagePreview());
      this.$.imageLightbox.addEventListener("mousedown", (event) => {
        if (event.target === this.$.imageLightbox) this.closeImagePreview();
      });
      document.addEventListener("keydown", (event) => {
        // configurable hotkeys (设置 → 快捷键); the settings panel's capture-phase
        // listener intercepts keydowns while a capture is in progress
        if (Hotkeys.match(event, "session.new")) {
          event.preventDefault();
          this.newSession();
          return;
        }
        if (Hotkeys.match(event, "input.focus")) {
          event.preventDefault();
          this.$.input.focus();
          return;
        }
        if (event.key === "Escape") {
          if (this.isImagePreviewOpen()) {
            // the preview owns Esc while it is up, so the other overlays stay shut
            event.preventDefault();
            this.closeImagePreview();
            return;
          }
          if (this.settingsPanel.isOpen()) this.settingsPanel.close();
          this.palette.close();
          this.hideSuggestions();
          this.hideCommandMenu();
          this.closeCwdDialog();
        }
      });
    }
    // ------------------------------------------------------------------------
    // Connection & server messages
    // ------------------------------------------------------------------------
    onConnectionStatus(connected) {
      this.state.connected = connected;
      if (!connected) {
        this.$.emptySub.textContent = t("reconnecting");
      }
    }
    onServerMessage(message) {
      switch (message.type) {
        case "ready":
          this.state.settings = message.settings;
          this.$.cwdText.textContent = message.settings.cwd;
          this.$.cwd.title = t("cwd_title", message.settings.cwd);
          this.$.emptySub.textContent = "";
          this.settingsPanel.setAboutInfo({ version: message.settings.version });
          void this.fetchSessions();
          void this.fetchCommands();
          // Wait for configs to load before applying the initial snapshot
          this.configPending = true;
          void this.settingsPanel.refresh().then(() => {
            this.configPending = false;
            void this.requestFullSnapshot();
            // Load server UI settings after session is ready to avoid theme/content flicker
            void this.settingsPanel.loadUiSettings();
          });
          break;
        case "session_updated":
          this.onSessionUpdated(message.snapshot);
          break;
        case "stream_patch":
          if (PHI_PERF) perfMark("patch-in", 0, { len: message.streaming && message.streaming.content ? JSON.stringify(message.streaming.content).length : 0 });
          this.applyStreamPatch(message);
          break;
        case "session_closed":
          this.$.emptySub.textContent = t("session_closed");
          break;
        case "models":
          this.state.models = message.models;
          this.panels.setModels(message.models);
          break;
        case "skills":
          this.state.skills = message.skills;
          this.panels.setSkills(message.skills);
          break;
        case "media_download":
          this.settingsPanel?.onMediaDownload?.(message);
          break;
        case "log":
          this.showBanner(message.message, message.level === "error" ? "error" : "info");
          break;
      }
    }
    onSessionUpdated(snapshot) {
      // Defer until configs are loaded so the config selector populates correctly
      if (this.configPending) {
        this.pendingSnapshot = snapshot;
        return;
      }
      this.pendingSnapshot = snapshot;
      if (this.renderQueued) return;
      this.renderQueued = true;
      requestAnimationFrame(() => {
        this.renderQueued = false;
        const snap = this.pendingSnapshot;
        this.pendingSnapshot = null;
        if (snap) this.applySnapshot(snap);
      });
    }
    /**
     * Incremental stream tick (server sends these instead of full snapshots
     * while deltas flow): merge the streaming message + changed tool runs into
     * the local session state and re-render ONLY the affected nodes. Payload
     * is O(change), so per-tick work stays independent of history length.
     */
    applyStreamPatch(message) {
      const session = this.pendingSnapshot ?? this.state.session;
      if (!session || !Array.isArray(session.messages)) {
        void this.requestFullSnapshot();
        return;
      }
      const gen = Number(message.generation ?? 0);
      if (gen && this.lastFullGeneration && gen < this.lastFullGeneration) return;  // stale
      const runs = Array.isArray(message.toolRuns) ? message.toolRuns : [];
      if (runs.length) {
        const byId = new Map((session.toolRuns ?? []).map((run) => [run.toolCallId, run]));
        for (const run of runs) byId.set(run.toolCallId, run);
        session.toolRuns = Array.from(byId.values());
      }
      const streaming = message.streaming;
      if (streaming && streaming.id) {
        const index = session.messages.findIndex((m) => m.id === streaming.id);
        if (index < 0) {
          // unknown streaming id (missed the MessageStart full snapshot):
          // re-sync instead of guessing
          void this.requestFullSnapshot();
          return;
        }
        session.messages[index] = streaming;
        this.queuePatchRender([streaming.id]);
      } else if (runs.length) {
        // tool output updates while no message is streaming (tool running
        // between messages): refresh nodes whose tool cards show the output
        const changedIds = new Set(runs.map((run) => run.toolCallId));
        const affected = session.messages
          .filter((m) => Array.isArray(m.content) && m.content.some((c) => c.type === "toolCall" && changedIds.has(c.id)))
          .map((m) => m.id);
        this.queuePatchRender(affected);
      }
    }
    /**
     * Queue a DOM refresh for the affected message ids. State merging above
     * runs on every patch (cheap object writes); the expensive part (markdown
     * rebuild of the whole streaming message, forced layout from autoscroll)
     * is coalesced to at most one render per PATCH_RENDER_WINDOW_MS.
     */
    queuePatchRender(ids) {
      const session = this.pendingSnapshot ?? this.state.session;
      if (!ids.length || !session) return;
      if (!this.patchCoalescer) this.patchCoalescer = new PatchCoalescer(PATCH_RENDER_WINDOW_MS);
      this.patchSession = session;
      const set = this.patchIds ?? (this.patchIds = new Set());
      for (const id of ids) set.add(id);
      if (this.patchCoalescer.offer(performance.now())) {
        this.flushPatchRender();
      } else if (this.patchFlushTimer == null) {
        this.patchFlushTimer = setTimeout(() => {
          this.patchFlushTimer = null;
          if (this.patchCoalescer && this.patchCoalescer.due(performance.now())) {
            this.flushPatchRender();
          }
        }, PATCH_RENDER_WINDOW_MS);
      }
    }
    flushPatchRender() {
      if (this.patchFlushTimer != null) {
        clearTimeout(this.patchFlushTimer);
        this.patchFlushTimer = null;
      }
      if (this.patchCoalescer) this.patchCoalescer.flush(performance.now());
      const ids = this.patchIds;
      const session = this.patchSession;
      this.patchIds = null;
      this.patchSession = null;
      if (!ids || !session) return;
      const runsById = new Map((session.toolRuns ?? []).map((run) => [run.toolCallId, run]));
      const t0 = performance.now();
      this.transcript.updateMessages([...ids], session, runsById);
      perfMark("flush", performance.now() - t0, { ids: ids.length });
    }
    /** A full snapshot render subsumes any queued patch render. */
    discardPatchRender() {
      if (this.patchFlushTimer != null) {
        clearTimeout(this.patchFlushTimer);
        this.patchFlushTimer = null;
      }
      this.patchCoalescer = null;
      this.patchIds = null;
      this.patchSession = null;
    }
    async requestFullSnapshot() {
      if (!this.ws || this.ws.ws?.readyState !== WebSocket.OPEN) return;
      const result = await this.ws.send({ type: "get_snapshot" });
      if (result && result.kind === "session" && result.session) {
        this.onSessionUpdated(result.session);
      }
    }
    /**
     * 立即应用工具设置：空闲时重启会话加载新设置（历史保留）；
     * 会话忙碌时跳过（下次重启会话后生效）。连续修改自动合并。
     */
    applyToolSettings() {
      if (this.applyToolTimer) clearTimeout(this.applyToolTimer);
      this.applyToolTimer = setTimeout(() => {
        this.applyToolTimer = null;
        void this.applyToolSettingsNow();
      }, 800);
    }
    async applyToolSettingsNow() {
      const session = this.state.session;
      const phase = session?.phase ?? "idle";
      if (!session || phase !== "idle" || session.isStreaming) return;
      // 重新打开当前会话（历史保留），让新的工具设置立即生效
      const result = await this.wsSend("open_session", { sessionId: session.id });
      if (result && result.kind === "session" && result.session) {
        this.onSessionUpdated(result.session);
      }
    }
    applySnapshot(snapshot) {
      // a full render carries the latest state: drop any queued patch render
      this.discardPatchRender();
      const firstSnapshot = this.state.session === null || this.state.session.id !== snapshot.id;
      this.state.session = snapshot;
      // patches older than the newest full snapshot are dropped (concurrent
      // emit paths can reorder in flight)
      this.lastFullGeneration = Number(snapshot.generation ?? 0);
      this.transcript.render(snapshot);
      this.$.emptyState.hidden = snapshot.messages.length > 0 || snapshot.isStreaming;
      this.$.sessionName.textContent = snapshot.name ?? t("phi_session");
      this.$.sessionName.title = snapshot.name ?? t("unnamed_session");
      const phaseMap = {
        idle: t("idle"),
        streaming: t("streaming"),
        compacting: t("compacting"),
        busy: t("busy")
      };
      this.$.phaseBadge.dataset.phase = snapshot.phase;
      this.$.phaseBadgeText.textContent = phaseMap[snapshot.phase] ?? snapshot.phase;
      this.$.phaseBadge.title = snapshot.isStreaming ? "\u4EE3\u7406\u6B63\u5728\u5904\u7406\u2026" : "";
      this.$.composerRow.classList.toggle("running", snapshot.phase !== "idle" || snapshot.isStreaming);
      this.$.cwdText.textContent = snapshot.cwd;
      this.$.cwd.title = t("cwd_title", snapshot.cwd);
      let totalTokens = 0;
      for (const message of snapshot.messages) {
        if (message.usage) {
          totalTokens += message.usage.totalTokens;
        }
      }
      const statusModel = snapshot.modelName ?? (snapshot.model && snapshot.model.provider !== "unknown" && snapshot.model.id !== "unknown" ? `${snapshot.model.provider}/${snapshot.model.id}` : t("no_model"));
      this.$.statusModel.textContent = statusModel;
      const levelIndex = ["off", "low", "medium", "high", "xhigh", "max"].indexOf(snapshot.thinkingLevel);
      const thinkingLabel = levelIndex >= 0 ? t("thinking_levels")[levelIndex] : snapshot.thinkingLevel;
      this.$.statusThinking.textContent = `${t("thinking_label")}${thinkingLabel}`;
      const phaseText = phaseMap[snapshot.phase] ?? snapshot.phase;
      this.$.statusPhase.textContent = phaseText;
      this.$.statusPhase.dataset.phase = snapshot.phase;
      this.$.statusTokens.hidden = totalTokens === 0 || !Prefs.data.statusTokens;
      this.$.statusTokensSep.hidden = totalTokens === 0 || !Prefs.data.statusTokens;
      this.$.statusTokens.textContent = `${formatTokens(totalTokens)} token`;
      this.panels.setSession(snapshot.thinkingLevel);
      this.panels.setConfig(snapshot.activeConfig || this.state.configs?.active || "", this.state.configs?.profiles || []);
      this.panels.setSessions(this.state.sessions, snapshot.id);
      this.panels.setSkills(this.state.skills);
      this.$.sendBtn.hidden = snapshot.isStreaming;
      this.$.abortBtn.hidden = !snapshot.isStreaming;
      this.$.sendBtn.disabled = false;
      if (firstSnapshot && snapshot.messages.length > 0) {
        this.transcript.scrollToBottom();
      }
    }
    runCommand(name, rawArgs = "") {
      const handler = this.builtinCommands[name];
      if (handler && handler(rawArgs)) return;
      if (UNSUPPORTED_IN_WEB.has(name)) {
        this.showBanner(t("web_ui_not_available", name), "info");
        return;
      }
      if (!rawArgs) {
        const listed = this.state.commands.find((command) => command.name === name);
        if (listed?.argumentHint) {
          this.$.input.value = `/${name} `;
          this.$.input.focus();
          this.autoGrow();
          return;
        }
      }
      this.sendPrompt(`/${name}${rawArgs ? ` ${rawArgs}` : ""}`);
    }
    compact() {
      void this.ws.send({ type: "compact" }).then((result) => {
        if (result.kind === "error") this.showBanner(result.error, "error");
      });
    }
    copyLastAssistantMessage() {
      const session = this.state.session;
      if (!session) return;
      for (let i = session.messages.length - 1; i >= 0; i--) {
        const message = session.messages[i];
        if (message.role !== "assistant") continue;
        const text = message.content.filter((content) => content.type === "text").map((content) => content.text).join("\n\n");
        if (text) {
          void navigator.clipboard.writeText(text).then(() => this.showBanner(t("copied"), "info"));
          return;
        }
      }
      this.showBanner(t("no_assistant_message"), "info");
    }
    showSessionInfo() {
      const session = this.state.session;
      if (!session) {
        this.showBanner(t("session_closed"), "info");
        return;
      }
      const levelIndex = ["off", "low", "medium", "high", "xhigh", "max"].indexOf(session.thinkingLevel);
      const thinkingLabel = levelIndex >= 0 ? t("thinking_levels")[levelIndex] : session.thinkingLevel;
      const modelName = session.modelName ?? (session.model ? `${session.model.provider}/${session.model.id}` : t("no_model"));
      this.showBanner(t("session_info", session.name ?? t("phi_session"), session.id.slice(0, 8), modelName, thinkingLabel), "info");
    }
    sendPrompt(text) {
      const attachments = this.attachments.length > 0 ? this.attachments : void 0;
      void this.ws.send({ type: "prompt", text, mode: "prompt", attachments }).then((result) => {
        if (result.kind === "error") {
          this.showBanner(result.error, "error");
        } else {
          this.clearAttachments();
        }
      });
      this.$.input.value = "";
      this.autoGrow();
      this.hideSuggestions();
    }
    sendThinking(level) {
      void this.ws.send({ type: "set_thinking", thinkingLevel: level }).then((result) => {
        if (result.kind === "error") this.showBanner(result.error, "error");
      });
    }
    abort() {
      void this.ws.send({ type: "abort" });
    }
    newSession() {
      void this.ws.send({ type: "create_session" }).then((result) => {
        if (result.kind === "error") this.showBanner(result.error, "error");
      });
    }
    openSession(sessionId) {
      void this.ws.send({ type: "open_session", sessionId }).then((result) => {
        if (result.kind === "error") this.showBanner(result.error, "error");
      });
    }
    reloadResources() {
      void this.ws.send({ type: "reload_resources" }).then((result) => {
        if (result.kind === "error") {
          this.showBanner(result.error, "error");
        } else {
          this.showBanner(t("reloaded"), "info");
        }
      });
    }
    deleteSkill(filePath) {
      void this.ws.send({ type: "delete_skill", filePath }).then((result) => {
        if (result.kind === "error") {
          this.showBanner(result.error, "error");
        } else {
          this.showBanner(t("skill_deleted"), "info");
        }
      });
    }
    deleteSession(sessionId) {
      void this.ws.send({ type: "delete_session", sessionId }).then((result) => {
        if (result.kind === "error") {
          this.showBanner(result.error, "error");
        } else {
          this.showBanner(t("session_deleted"), "info");
          void this.fetchSessions();
        }
      });
    }
    // ------------------------------------------------------------------------
    // File attachments
    // ------------------------------------------------------------------------
    onFilesSelected(fileList) {
      if (!fileList || fileList.length === 0) return;
      for (const file of Array.from(fileList)) {
        void this.uploadFile(file);
      }
      this.$.fileInput.value = "";
    }
    async uploadFile(file) {
      const dataUrl = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result ?? ""));
        reader.onerror = () => resolve("");
        reader.readAsDataURL(file);
      });
      const mimeType = file.type || "application/octet-stream";
      const data = dataUrl.split(",")[1] ?? "";
      try {
        const response = await fetch("/api/upload", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: file.name, mimeType, data })
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const result = await response.json();
        this.attachments.push(result);
        if (mimeType.startsWith("image/") && data) {
          this.attachmentPreviews.set(result.path, `data:${mimeType};base64,${data}`);
        }
        this.renderAttachments();
      } catch (error) {
        this.showBanner(
          t("upload_failed", file.name, error instanceof Error ? error.message : String(error)),
          "error"
        );
      }
    }
    clearAttachments() {
      this.attachments = [];
      this.attachmentPreviews.clear();
      this.renderAttachments();
    }
    removeAttachment(index) {
      const removed = this.attachments.splice(index, 1)[0];
      if (removed) this.attachmentPreviews.delete(removed.path);
      this.renderAttachments();
    }
    renderAttachments() {
      const container = this.$.attachments;
      clear(container);
      if (this.attachments.length === 0) {
        container.hidden = true;
        return;
      }
      container.hidden = false;
      this.attachments.forEach((attachment, index) => {
        const isImage = attachment.mimeType.startsWith("image/");
        const preview = this.attachmentPreviews.get(attachment.path);
        const remove = el("button", "btn btn-xs attach-remove", "\u2715");
        remove.type = "button";
        remove.title = t("remove_attachment");
        remove.addEventListener("click", () => this.removeAttachment(index));
        if (isImage && preview) {
          const wrap = el("span", "attach-image");
          const img = document.createElement("img");
          img.src = preview;
          img.alt = attachment.name;
          img.title = t("image_preview_hint");
          img.loading = "lazy";
          img.addEventListener("click", () => this.openImagePreview(preview, attachment.name));
          wrap.appendChild(img);
          const nameRow = el("span", "attach-img-name");
          const name = el("span", "attach-name", attachment.name);
          name.title = attachment.path;
          nameRow.appendChild(name);
          nameRow.appendChild(remove);
          wrap.appendChild(nameRow);
          container.appendChild(wrap);
        } else {
          const chip = el("span", "attach-chip");
          const name = el("span", "attach-name", attachment.name);
          name.title = attachment.path;
          chip.appendChild(name);
          chip.appendChild(remove);
          container.appendChild(chip);
        }
      });
    }
    // ------------------------------------------------------------------------
    // Image lightbox
    // ------------------------------------------------------------------------
    /** Show the full-size image in a small window inside the app itself.
     *  `window.open` used to handle this, but a new WebView2 window has no
     *  document of its own and came up blank (about:blank). */
    openImagePreview(src, name) {
      if (!src) return;
      this.$.lightboxImage.src = src;
      this.$.lightboxImage.alt = name || t("image_preview");
      this.$.lightboxCaption.textContent = name || "";
      this.$.lightboxCaption.title = name || "";
      this.$.imageLightbox.hidden = false;
    }
    closeImagePreview() {
      if (this.$.imageLightbox.hidden) return;
      this.$.imageLightbox.hidden = true;
      // drop the data URL so a large image is not kept alive by the hidden <img>
      this.$.lightboxImage.removeAttribute("src");
    }
    isImagePreviewOpen() {
      return !this.$.imageLightbox.hidden;
    }
    // ------------------------------------------------------------------------
    // Working directory
    // ------------------------------------------------------------------------
    openCwdDialog() {
      const current = this.state.settings?.cwd ?? this.state.session?.cwd ?? "";
      this.$.cwdCurrent.textContent = current;
      this.$.cwdInput.value = current;
      this.$.cwdError.hidden = true;
      this.$.cwdDialog.hidden = false;
      this.$.cwdInput.focus();
      this.$.cwdInput.select();
    }
    closeCwdDialog() {
      this.$.cwdDialog.hidden = true;
    }
    showCwdError(message) {
      this.$.cwdError.textContent = message;
      this.$.cwdError.hidden = false;
    }
    confirmCwd() {
      const cwd = this.$.cwdInput.value.trim();
      if (!cwd) {
        this.showCwdError(t("cwd_required"));
        return;
      }
      this.$.cwdConfirm.disabled = true;
      void this.ws.send({ type: "set_cwd", cwd }).then((result) => {
        this.$.cwdConfirm.disabled = false;
        if (result.kind === "settings") {
          this.state.settings = result.settings;
          this.$.cwdText.textContent = result.settings.cwd;
          this.$.cwd.title = t("cwd_title", result.settings.cwd);
          this.closeCwdDialog();
          this.showBanner(t("cwd_changed", result.settings.cwd), "info");
          this.$.emptySub.textContent = "";
          void this.fetchSessions();
          void this.fetchCommands();
        } else if (result.kind === "error") {
          this.showCwdError(result.error);
        }
      });
    }
    /** Open the native folder picker; on selection, switch to the chosen folder. */
    async browseDirectory() {
      this.$.cwdBrowse.disabled = true;
      this.$.cwdBrowse.textContent = t("picking_directory");
      try {
        const response = await fetch("/api/pick-directory");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const result = await response.json();
        if (result.path) {
          this.$.cwdInput.value = result.path;
          this.showCwdError("");
          this.confirmCwd();
        }
      } catch (error) {
        this.showCwdError(t("pick_directory_failed", error instanceof Error ? error.message : String(error)));
      } finally {
        this.$.cwdBrowse.disabled = false;
        this.$.cwdBrowse.textContent = t("browse_directory");
      }
    }
    async fetchSessions() {
      const result = await this.ws.send({ type: "list_sessions" });
      if (result.kind === "sessions") {
        this.state.sessions = result.sessions;
        this.panels.setSessions(result.sessions, this.state.session?.id);
      }
    }
    async fetchCommands() {
      const result = await this.ws.send({ type: "list_commands" });
      if (result.kind === "commands") {
        this.state.commands = result.commands;
        this.palette.setCommands(result.commands);
      }
    }
    async fetchModels() {
      const result = await this.ws.send({ type: "list_models" });
      if (result.kind === "models") {
        this.state.models = result.models;
        this.panels.setModels(result.models);
      }
    }
    /** Send a config-panel request; maps the response to a ConfigsResult or null on error. */
    async wsSend(type, payload = {}) {
      const request = { type, ...payload };
      const result = await this.ws.send(request);
      if (result.kind === "configs") {
        return {
          active: result.active,
          profiles: result.profiles
        };
      }
      if (result.kind === "error") {
        this.showBanner(result.error, "error");
        return null;
      }
      // other kinds (tool_settings / session / ok) pass through untouched
      return result;
    }
    // ------------------------------------------------------------------------
    // Composer
    // ------------------------------------------------------------------------
    sendInput() {
      const text = this.$.input.value.trim();
      if (!text) return;
      const slashMatch = text.match(/^\/([\w:-]+)(?:\s+([\s\S]*))?$/);
      if (slashMatch) {
        const name = slashMatch[1];
        const args = slashMatch[2] ?? "";
        const known = name in this.builtinCommands || UNSUPPORTED_IN_WEB.has(name) || this.state.commands.some((command) => command.name === name);
        if (known) {
          this.runCommand(name, args);
          this.$.input.value = "";
          this.autoGrow();
          this.hideSuggestions();
          return;
        }
      }
      if (this.state.session === null) {
        this.showBanner(t("creating_session"), "info");
        void this.ws.send({ type: "create_session" }).then((result) => {
          if (result.kind !== "error") {
            this.sendPrompt(text);
          } else {
            this.showBanner(result.error, "error");
          }
        });
        return;
      }
      this.sendPrompt(text);
    }
    autoGrow() {
      const input = this.$.input;
      input.style.height = "auto";
      input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
    }
    onInputKeyDown(event) {
      const suggestionsActive = !this.$.composerSuggest.hidden;
      if (suggestionsActive) {
        if (event.key === "ArrowDown") {
          event.preventDefault();
          this.suggestIndex = (this.suggestIndex + 1) % this.suggestItems.length;
          this.renderSuggestions();
          return;
        }
        if (event.key === "ArrowUp") {
          event.preventDefault();
          this.suggestIndex = (this.suggestIndex - 1 + this.suggestItems.length) % this.suggestItems.length;
          this.renderSuggestions();
          return;
        }
        if (event.key === "Tab" || event.key === "Enter") {
          event.preventDefault();
          this.completeSuggestion(this.suggestItems[this.suggestIndex]);
          return;
        }
        if (event.key === "Escape") {
          this.hideSuggestions();
          return;
        }
      }
      // 可绑定的输入快捷键（设置 → 快捷键）：发送 / 换行；IME 组合中不拦截
      if (event.isComposing || event.keyCode === 229) return;
      if (Hotkeys.match(event, "input.send")) {
        event.preventDefault();
        this.sendInput();
        return;
      }
      if (Hotkeys.match(event, "input.newline")) {
        if (!event.ctrlKey && !event.altKey && !event.metaKey) return; // 文本域默认插入换行
        event.preventDefault();
        const input = this.$.input;
        const start = input.selectionStart ?? input.value.length;
        const end = input.selectionEnd ?? start;
        input.value = input.value.slice(0, start) + "\n" + input.value.slice(end);
        const pos = start + 1;
        input.setSelectionRange(pos, pos);
        this.autoGrow();
        this.updateSuggestions();
        return;
      }
    }
    updateSuggestions() {
      const text = this.$.input.value;
      if (!text.startsWith("/") || text.includes(" ")) {
        this.hideSuggestions();
        return;
      }
      const prefix = text.slice(1).toLowerCase();
      const matched = this.state.commands.filter((command) => command.name.startsWith(prefix) || command.name.includes(prefix)).slice(0, 8);
      if (matched.length === 0) {
        this.hideSuggestions();
        return;
      }
      this.suggestItems = matched;
      this.suggestIndex = 0;
      this.$.composerSuggest.hidden = false;
      this.renderSuggestions();
    }
    renderSuggestions() {
      const container = this.$.composerSuggest;
      clear(container);
      this.suggestItems.forEach((command, index) => {
        const item = el("button", "suggest-item");
        item.type = "button";
        item.classList.toggle("selected", index === this.suggestIndex);
        item.appendChild(el("span", "suggest-name", `/${command.name}`));
        if (command.argumentHint) {
          item.appendChild(el("span", "suggest-usage", t("palette_usage", command.argumentHint)));
        }
        item.appendChild(el("span", "suggest-desc", command.description));
        item.addEventListener("mousedown", (event) => {
          event.preventDefault();
          this.completeSuggestion(command);
        });
        container.appendChild(item);
      });
    }
    completeSuggestion(command) {
      this.$.input.value = `/${command.name} `;
      this.$.input.focus();
      this.hideSuggestions();
    }
    hideSuggestions() {
      this.$.composerSuggest.hidden = true;
    }
    // ------------------------------------------------------------------------
    // `/` quick-command menu
    // ------------------------------------------------------------------------
    toggleCommandMenu() {
      if (!this.$.commandMenu.hidden) {
        this.hideCommandMenu();
        return;
      }
      this.renderCommandMenu();
      this.$.commandMenu.hidden = false;
    }
    hideCommandMenu() {
      this.$.commandMenu.hidden = true;
    }
    renderCommandMenu() {
      const menu = this.$.commandMenu;
      clear(menu);
      const commands = this.state.commands;
      if (commands.length === 0) {
        menu.appendChild(el("div", "command-menu-empty", t("no_commands_available")));
        return;
      }
      menu.appendChild(el("div", "command-menu-title", t("quick_commands")));
      for (const command of commands) {
        const item = el("button", "command-menu-item");
        item.type = "button";
        item.appendChild(el("span", "command-menu-name", `/${command.name}`));
        if (command.argumentHint) {
          item.appendChild(el("span", "command-menu-usage", command.argumentHint));
        }
        item.appendChild(el("span", "command-menu-desc", command.description));
        item.addEventListener("click", (event) => {
          event.stopPropagation();
          this.hideCommandMenu();
          this.runCommand(command.name);
        });
        menu.appendChild(item);
      }
    }
    // ------------------------------------------------------------------------
    // Session name editing
    // ------------------------------------------------------------------------
    startNameEdit() {
      const button = this.$.sessionName;
      const current = this.state.session?.name ?? "";
      const input = document.createElement("input");
      input.type = "text";
      input.value = current;
      input.className = "session-name editing";
      input.maxLength = 80;
      button.replaceWith(input);
      input.focus();
      input.select();
      const commit = () => {
        const name = input.value.trim();
        input.replaceWith(button);
        if (name && name !== current) {
          void this.ws.send({ type: "set_session_name", name });
        }
      };
      const cancel = () => {
        input.replaceWith(button);
      };
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") commit();
        if (event.key === "Escape") cancel();
      });
      input.addEventListener("blur", commit);
    }
    // ------------------------------------------------------------------------
    // Banner
    // ------------------------------------------------------------------------
    showBanner(message, kind) {
      const banner = this.$.banner;
      banner.textContent = message;
      banner.className = `banner ${kind}`;
      banner.hidden = false;
      if (this.bannerTimer) clearTimeout(this.bannerTimer);
      this.bannerTimer = setTimeout(
        () => {
          banner.hidden = true;
        },
        kind === "error" ? 12e3 : 6e3
      );
    }
    iconSvg(name) {
      if (name === "close") {
        return '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
      }
      return '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11a8 8 0 0 0-14.9-3M4 13a8 8 0 0 0 14.9 3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M20 4v7h-7M4 20v-7h7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    }
  };
  window.addEventListener("DOMContentLoaded", () => {
    void new App();
  });
})();

