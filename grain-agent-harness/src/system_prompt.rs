//! System-prompt assembly fragments.
//!
//! Ports `packages/agent/src/harness/system-prompt.ts`. Skills loaded from
//! disk (or any other source) are rendered into a fixed XML block to be
//! appended onto the agent system prompt.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub file_path: String,
    #[serde(default)]
    pub disable_model_invocation: bool,
    /// Full body content after the frontmatter `---` fence.
    /// Used by TUI slash-palette skill injection.
    #[serde(default)]
    pub body: String,
}

/// Format the model-visible portion of a skill list for system-prompt
/// inclusion. Skills with `disable_model_invocation = true` are filtered out.
pub fn format_skills_for_system_prompt(skills: &[Skill]) -> String {
    let visible: Vec<&Skill> = skills
        .iter()
        .filter(|s| !s.disable_model_invocation)
        .collect();
    if visible.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    out.push_str("The following skills provide specialized instructions for specific tasks.\n");
    out.push_str("Read the full skill file when the task matches its description.\n");
    out.push_str(
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.\n\n",
    );
    out.push_str("<available_skills>\n");
    for skill in visible {
        out.push_str("  <skill>\n");
        out.push_str(&format!("    <name>{}</name>\n", escape_xml(&skill.name)));
        out.push_str(&format!(
            "    <description>{}</description>\n",
            escape_xml(&skill.description)
        ));
        out.push_str(&format!(
            "    <location>{}</location>\n",
            escape_xml(&skill.file_path)
        ));
        out.push_str("  </skill>\n");
    }
    out.push_str("</available_skills>");
    out
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Directory portion of a skill path, by upstream's rules.
///
/// Port of `dirnameEnvPath` (`packages/agent/src/harness/skills.ts:356-360`
/// @ 34239180): strip trailing slashes, cut at the last `/`, and treat a
/// leading-slash-only result as `/`. Deliberately *not* `std::path::Path` —
/// these are environment paths that are always `/`-separated, including when
/// the host happens to be Windows.
fn dirname_env_path(path: &str) -> &str {
    let normalized = path.trim_end_matches('/');
    match normalized.rfind('/') {
        Some(i) if i > 0 => &normalized[..i],
        _ => "/",
    }
}

/// Render a skill invocation prompt.
///
/// Port of upstream `formatSkillInvocation`
/// (`packages/agent/src/harness/skills.ts:38-41` @ 34239180). The
/// `location` attribute and the "References are relative to" line are what
/// let a skill body use paths relative to its own directory.
///
/// `additional_instructions`, when present, follows the block after a blank
/// line.
pub fn format_skill_invocation(skill: &Skill, additional_instructions: Option<&str>) -> String {
    let block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name,
        skill.file_path,
        dirname_env_path(&skill.file_path),
        skill.body
    );
    match additional_instructions {
        Some(extra) => format!("{block}\n\n{extra}"),
        None => block,
    }
}

#[cfg(test)]
mod invocation_tests {
    use super::*;

    fn skill() -> Skill {
        Skill {
            name: "bash".into(),
            description: "Runs shell commands".into(),
            file_path: "/skills/bash/SKILL.md".into(),
            disable_model_invocation: false,
            body: "Run things.".into(),
        }
    }

    #[test]
    fn renders_upstream_block_shape() {
        let out = format_skill_invocation(&skill(), None);
        assert_eq!(
            out,
            "<skill name=\"bash\" location=\"/skills/bash/SKILL.md\">\n\
             References are relative to /skills/bash.\n\n\
             Run things.\n</skill>"
        );
    }

    #[test]
    fn appends_additional_instructions_after_a_blank_line() {
        let out = format_skill_invocation(&skill(), Some("also be brief"));
        assert!(out.ends_with("</skill>\n\nalso be brief"));
    }

    #[test]
    fn dirname_matches_upstream_edge_cases() {
        assert_eq!(dirname_env_path("/skills/bash/SKILL.md"), "/skills/bash");
        // Trailing slashes are stripped first.
        assert_eq!(dirname_env_path("/skills/bash/"), "/skills");
        // A path with no directory part, and a root-level one, both yield "/".
        assert_eq!(dirname_env_path("SKILL.md"), "/");
        assert_eq!(dirname_env_path("/SKILL.md"), "/");
    }
}
