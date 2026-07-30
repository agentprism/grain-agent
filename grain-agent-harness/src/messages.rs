//! Custom message types and harness `convert_to_llm`.
//!
//! Ports `packages/agent/src/harness/messages.ts`. The TS code uses
//! declaration-merging to add typed message variants to `CustomAgentMessages`;
//! in Rust, custom messages live under [`grain_agent_core::AgentMessage::Custom`]
//! as JSON values with a `role` discriminator.

use grain_agent_core::{AgentMessage, Message, TextContent, UserContent, UserMessage};
use serde::{Deserialize, Serialize};

pub const COMPACTION_SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";

pub const BRANCH_SUMMARY_PREFIX: &str =
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";
pub const BRANCH_SUMMARY_SUFFIX: &str = "</summary>";

/// Record of a user-initiated shell execution surfaced in the transcript.
///
/// Port of the upstream `BashExecutionMessage`
/// (`packages/agent/src/harness/messages.ts:19-29` @ 34239180).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashExecutionMessage {
    /// Always `"bashExecution"`.
    pub role: String,
    pub command: String,
    pub output: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    #[serde(default)]
    pub cancelled: bool,
    #[serde(default)]
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
    pub timestamp: i64,
    /// When true, the execution stays visible in the transcript but is
    /// dropped from the LLM context (messages.ts:124-126).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_from_context: Option<bool>,
}

/// Build an [`AgentMessage::Custom`] carrying a bash execution record.
pub fn bash_execution_message(msg: BashExecutionMessage) -> AgentMessage {
    AgentMessage::Custom(serde_json::to_value(msg).expect("bash execution serialises"))
}

/// Render a bash execution as the text shown to the model.
///
/// Port of `bashExecutionToText` (messages.ts:63-79), matching the upstream
/// copy exactly.
pub fn bash_execution_to_text(msg: &BashExecutionMessage) -> String {
    let mut text = format!("Ran `{}`\n", msg.command);
    if !msg.output.is_empty() {
        text.push_str(&format!("```\n{}\n```", msg.output));
    } else {
        text.push_str("(no output)");
    }
    if msg.cancelled {
        text.push_str("\n\n(command cancelled)");
    } else if let Some(code) = msg.exit_code
        && code != 0
    {
        text.push_str(&format!("\n\nCommand exited with code {code}"));
    }
    if msg.truncated
        && let Some(path) = &msg.full_output_path
    {
        text.push_str(&format!("\n\n[Output truncated. Full output: {path}]"));
    }
    text
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryMessage {
    /// Always `"branchSummary"`.
    pub role: String,
    pub summary: String,
    pub from_id: String,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSummaryMessage {
    /// Always `"compactionSummary"`.
    pub role: String,
    pub summary: String,
    pub tokens_before: u64,
    pub timestamp: i64,
}

/// Arbitrary user-defined message variant (e.g. a UI artifact).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessage {
    /// Always `"custom"`.
    pub role: String,
    pub custom_type: String,
    /// `String` for plain text, or an array of `UserContent` for rich content.
    pub content: serde_json::Value,
    pub display: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    pub timestamp: i64,
}

/// Build an [`AgentMessage::Custom`] carrying a branch summary.
pub fn branch_summary_message(
    summary: impl Into<String>,
    from_id: impl Into<String>,
    timestamp: i64,
) -> AgentMessage {
    let msg = BranchSummaryMessage {
        role: "branchSummary".into(),
        summary: summary.into(),
        from_id: from_id.into(),
        timestamp,
    };
    AgentMessage::Custom(serde_json::to_value(msg).expect("branch summary serialises"))
}

/// Build an [`AgentMessage::Custom`] carrying a compaction summary.
pub fn compaction_summary_message(
    summary: impl Into<String>,
    tokens_before: u64,
    timestamp: i64,
) -> AgentMessage {
    let msg = CompactionSummaryMessage {
        role: "compactionSummary".into(),
        summary: summary.into(),
        tokens_before,
        timestamp,
    };
    AgentMessage::Custom(serde_json::to_value(msg).expect("compaction summary serialises"))
}

/// Build an [`AgentMessage::Custom`] carrying an arbitrary application payload.
///
/// `content` may be either a string or an array of `UserContent`.
pub fn custom_message(
    custom_type: impl Into<String>,
    content: serde_json::Value,
    display: bool,
    details: Option<serde_json::Value>,
    timestamp: i64,
) -> AgentMessage {
    let msg = CustomMessage {
        role: "custom".into(),
        custom_type: custom_type.into(),
        content,
        display,
        details,
        timestamp,
    };
    AgentMessage::Custom(serde_json::to_value(msg).expect("custom message serialises"))
}

fn parse_user_content(value: &serde_json::Value) -> Vec<UserContent> {
    if let Some(text) = value.as_str() {
        return vec![UserContent::Text(TextContent { text: text.into() })];
    }
    if value.is_array()
        && let Ok(parsed) = serde_json::from_value::<Vec<UserContent>>(value.clone())
    {
        return parsed;
    }
    Vec::new()
}

/// Harness-aware [`grain_agent_core::ConvertToLlmFn`] body.
///
/// Translates harness-specific custom-message variants into plain user messages
/// before they reach the LLM, then passes through standard messages.
pub fn convert_to_llm(messages: Vec<AgentMessage>) -> Vec<Message> {
    messages
        .into_iter()
        .filter_map(|m| match m {
            AgentMessage::Standard(m) => Some(m),
            AgentMessage::Custom(value) => convert_custom(value),
        })
        .collect()
}

fn convert_custom(value: serde_json::Value) -> Option<Message> {
    let role = value.get("role").and_then(|r| r.as_str())?;
    let timestamp = value.get("timestamp").and_then(|t| t.as_i64()).unwrap_or(0);

    match role {
        // messages.ts:124-132 — excluded executions are dropped from the LLM
        // context; the rest become plain user messages with the rendered
        // command transcript.
        "bashExecution" => {
            let msg: BashExecutionMessage = serde_json::from_value(value).ok()?;
            if msg.exclude_from_context == Some(true) {
                return None;
            }
            let text = bash_execution_to_text(&msg);
            Some(Message::User(UserMessage {
                content: vec![UserContent::Text(TextContent { text })],
                timestamp: msg.timestamp,
            }))
        }
        "branchSummary" => {
            let summary = value.get("summary").and_then(|s| s.as_str()).unwrap_or("");
            let text = format!(
                "{}{}{}",
                BRANCH_SUMMARY_PREFIX, summary, BRANCH_SUMMARY_SUFFIX
            );
            Some(Message::User(UserMessage {
                content: vec![UserContent::Text(TextContent { text })],
                timestamp,
            }))
        }
        "compactionSummary" => {
            let summary = value.get("summary").and_then(|s| s.as_str()).unwrap_or("");
            let text = format!(
                "{}{}{}",
                COMPACTION_SUMMARY_PREFIX, summary, COMPACTION_SUMMARY_SUFFIX
            );
            Some(Message::User(UserMessage {
                content: vec![UserContent::Text(TextContent { text })],
                timestamp,
            }))
        }
        "custom" => {
            let content = value
                .get("content")
                .map(parse_user_content)
                .unwrap_or_default();
            if content.is_empty() {
                return None;
            }
            Some(Message::User(UserMessage { content, timestamp }))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bash_msg() -> BashExecutionMessage {
        BashExecutionMessage {
            role: "bashExecution".into(),
            command: "ls".into(),
            output: "a.txt".into(),
            exit_code: Some(0),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 42,
            exclude_from_context: None,
        }
    }

    /// patch-10: bashExecution converts to a plain user message carrying
    /// the rendered transcript (upstream convertToLlm, messages.ts:124-132
    /// @ 34239180).
    #[test]
    fn bash_execution_converts_to_user_message() {
        let converted = convert_to_llm(vec![bash_execution_message(bash_msg())]);
        assert_eq!(converted.len(), 1);
        let Message::User(user) = &converted[0] else {
            panic!("expected a user message, got {:?}", converted[0]);
        };
        assert_eq!(user.timestamp, 42);
        let UserContent::Text(text) = &user.content[0] else {
            panic!("expected text content");
        };
        assert_eq!(text.text, "Ran `ls`\n```\na.txt\n```");
    }

    /// patch-10: excludeFromContext drops the execution from the LLM
    /// context entirely (messages.ts:124-126).
    #[test]
    fn bash_execution_exclude_from_context_is_dropped() {
        let mut msg = bash_msg();
        msg.exclude_from_context = Some(true);
        let converted = convert_to_llm(vec![bash_execution_message(msg)]);
        assert!(converted.is_empty());
    }

    /// Upstream bashExecutionToText copy: no output, non-zero exit code,
    /// cancellation, and truncation notes (messages.ts:63-79).
    #[test]
    fn bash_execution_to_text_matches_upstream_copy() {
        let mut msg = bash_msg();
        msg.output = String::new();
        msg.exit_code = Some(2);
        assert_eq!(
            bash_execution_to_text(&msg),
            "Ran `ls`\n(no output)\n\nCommand exited with code 2"
        );

        let mut msg = bash_msg();
        msg.cancelled = true;
        // Cancellation suppresses the exit-code note.
        msg.exit_code = Some(130);
        assert_eq!(
            bash_execution_to_text(&msg),
            "Ran `ls`\n```\na.txt\n```\n\n(command cancelled)"
        );

        let mut msg = bash_msg();
        msg.truncated = true;
        msg.full_output_path = Some("/tmp/full.log".into());
        assert_eq!(
            bash_execution_to_text(&msg),
            "Ran `ls`\n```\na.txt\n```\n\n[Output truncated. Full output: /tmp/full.log]"
        );

        // exitCode undefined → no exit-code note (msg.exitCode !== null &&
        // !== undefined guard upstream).
        let mut msg = bash_msg();
        msg.exit_code = None;
        assert_eq!(bash_execution_to_text(&msg), "Ran `ls`\n```\na.txt\n```");
    }
}
