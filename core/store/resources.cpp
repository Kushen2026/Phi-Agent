#include "store/resources.hpp"

#include <algorithm>

#include "util/base.hpp"

namespace phi {

bool parse_frontmatter(const std::string& text, std::vector<std::pair<std::string, std::string>>& out,
	std::string& body) {
	if (!starts_with(text, "---")) return false;
	// normalize CRLF first: a windows-edited file ("\r\n") made every find("\n")
	// land one char late and left \r inside keys/values, so frontmatter parsing
	// failed on files saved by notepad/VS-CRLF
	std::string normalized = replace_all(text, "\r\n", "\n");
	const std::string& t = normalized;
	size_t line_end = t.find('\n');
	if (line_end == std::string::npos) return false;
	size_t close = t.find("\n---", line_end);
	if (close == std::string::npos) return false;
	std::string fm = t.substr(line_end + 1, close - line_end - 1);
	size_t body_start = close + 4;  // "\n---"
	if (body_start < t.size() && t[body_start] == '\n') body_start++;
	body = t.substr(body_start);

	out.clear();
	for (auto& line : split(fm, '\n')) {
		line = trim(line);
		if (line.empty()) continue;
		size_t colon = line.find(':');
		if (colon == std::string::npos) continue;
		out.emplace_back(trim(line.substr(0, colon)), trim(line.substr(colon + 1)));
	}
	return true;
}

static std::string strip_bom(std::string s) {
	if (s.size() >= 3 && (unsigned char)s[0] == 0xEF && (unsigned char)s[1] == 0xBB && (unsigned char)s[2] == 0xBF) {
		s = s.substr(3);
	}
	return s;
}

Resources::Resources(std::string agent_dir, std::string cwd)
	: agent_dir_(std::move(agent_dir)), cwd_(std::move(cwd)) {
	reload();
}

void Resources::reload() {
	skills_.clear();
	templates_.clear();
	diagnostics_.clear();
	scan_skills();
	scan_templates();
}

// ── skills ─────────────────────────────────────────────────────────────────

static bool add_skill_file(const std::string& path, const std::string& base_dir,
	std::vector<SkillInfo>& out, std::vector<std::string>& diagnostics) {
	auto content = read_file(path);
	if (!content) return false;
	std::vector<std::pair<std::string, std::string>> fm;
	std::string body;
	if (!parse_frontmatter(strip_bom(*content), fm, body)) {
		diagnostics.push_back("skill missing frontmatter: " + path);
		return false;
	}
	SkillInfo info;
	info.base_dir = base_dir;
	info.file_path = path;
	for (const auto& [k, v] : fm) {
		if (k == "name") info.name = v;
		else if (k == "description") info.description = v;
		else if (k == "disable-model-invocation") info.disable_model_invocation = (v == "true");
	}
	if (info.name.empty()) info.name = path_basename(base_dir.empty() ? path : base_dir);
	out.push_back(std::move(info));
	return true;
}

// Discovery only: the skills panel needs the list to render its cards and to
// validate a delete. Nothing loads a skill into a prompt any more.
void Resources::scan_skills() {
	// global: <agent_dir>/skills/<name>/SKILL.md and <agent_dir>/skills/*.md
	std::string global_dir = path_join(agent_dir_, "skills");
	if (path_is_dir(global_dir)) {
		for (const auto& entry : list_dir(global_dir)) {
			std::string full = path_join(global_dir, entry);
			if (path_is_dir(full)) {
				std::string skill_md = path_join(full, "SKILL.md");
				if (path_exists(skill_md)) {
					add_skill_file(skill_md, full, skills_, diagnostics_);
				}
			} else if (ends_with(entry, ".md")) {
				add_skill_file(full, "", skills_, diagnostics_);
			}
		}
	}
	// project: <cwd>/data/skills/<name>/SKILL.md and *.md
	std::string project_dir = path_join(cwd_, "data/skills");
	if (path_is_dir(project_dir)) {
		for (const auto& entry : list_dir(project_dir)) {
			std::string full = path_join(project_dir, entry);
			if (path_is_dir(full)) {
				std::string skill_md = path_join(full, "SKILL.md");
				if (path_exists(skill_md)) {
					add_skill_file(skill_md, full, skills_, diagnostics_);
				}
			} else if (ends_with(entry, ".md")) {
				add_skill_file(full, "", skills_, diagnostics_);
			}
		}
	}
}

// ── prompt templates ───────────────────────────────────────────────────────

void Resources::scan_templates() {
	std::vector<std::string> dirs = {path_join(agent_dir_, "prompts"), path_join(cwd_, ".phi/prompts")};
	for (const auto& dir : dirs) {
		if (!path_is_dir(dir)) continue;
		for (const auto& entry : list_dir(dir)) {
			if (!ends_with(entry, ".md")) continue;
			auto content = read_file(path_join(dir, entry));
			if (!content) continue;
			PromptTemplate t;
			t.name = entry.substr(0, entry.size() - 3);
			std::vector<std::pair<std::string, std::string>> fm;
			std::string body = strip_bom(*content);
			if (parse_frontmatter(body, fm, body)) {
				for (const auto& [k, v] : fm) {
					if (k == "description") t.description = v;
					else if (k == "argument-hint") t.argument_hint = v;
				}
			}
			t.content = body;
			templates_.push_back(std::move(t));
		}
	}
}

}  // namespace phi
