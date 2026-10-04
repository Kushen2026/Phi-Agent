// Scans skills, prompt templates and project context files.
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace phi {

struct SkillInfo {
	std::string name;
	std::string description;
	std::string base_dir;
	std::string file_path;
	bool disable_model_invocation = false;
};

struct PromptTemplate {
	std::string name;
	std::string description;
	std::string argument_hint;
	std::string content;  // template body with $1/$ARGUMENTS placeholders
};

class Resources {
public:
	Resources(std::string agent_dir, std::string cwd);

	void reload();

	// the skills panel lists these and deletes them by file_path
	const std::vector<SkillInfo>& skills() const { return skills_; }
	const std::vector<PromptTemplate>& templates() const { return templates_; }
	std::vector<std::string> diagnostics() const { return diagnostics_; }

private:
	std::string agent_dir_;
	std::string cwd_;
	std::vector<SkillInfo> skills_;
	std::vector<PromptTemplate> templates_;
	std::vector<std::string> diagnostics_;

	void scan_skills();
	void scan_templates();
};

// minimal frontmatter parser: "---\nkey: value\n---\n" -> map + body
bool parse_frontmatter(const std::string& text, std::vector<std::pair<std::string, std::string>>& out,
	std::string& body);

}  // namespace phi
