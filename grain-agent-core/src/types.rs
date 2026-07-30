//! Core message, tool, event, and state types.
//!
//! Ports `packages/agent/src/types.ts` from the reference TypeScript implementation,
//! plus the message/model primitives that live in `@earendil-works/pi-ai`.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Content blocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TextContent {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    pub data: String,
    pub mime_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    pub thinking: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Provider-specific raw reasoning payload preserved so the next turn can
    /// replay it verbatim (e.g. OpenAI o-series `reasoning_content`,
    /// DeepSeek-R1 `reasoning_content`). Anthropic uses [`Self::signature`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// Content blocks legal in an assistant message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AssistantContent {
    Text(TextContent),
    Image(ImageContent),
    Thinking(ThinkingContent),
    ToolCall(ToolCall),
}

/// Content blocks legal in user or tool-result messages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum UserContent {
    Text(TextContent),
    Image(ImageContent),
}

impl UserContent {
    /// Convenience constructor: `UserContent::Text(TextContent { text })`.
    pub fn text(s: impl Into<String>) -> Self {
        UserContent::Text(TextContent { text: s.into() })
    }
}

// ---------------------------------------------------------------------------
// Usage / cost / model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Cost {
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
    #[serde(default, rename = "cacheRead")]
    pub cache_read: f64,
    #[serde(default, rename = "cacheWrite")]
    pub cache_write: f64,
    #[serde(default)]
    pub total: f64,
}

impl Cost {
    /// Compute the USD cost for a single `Usage` reading using this
    /// pricing table. Field semantics match `models.dev`: per-million-token
    /// prices, with `cache_read` being the discounted rate for cached
    /// prompt tokens. `usage.input` is the **total** prompt tokens (cached
    /// + uncached), so the uncached share is `input - cache_read`.
    ///
    /// Returns 0.0 when the descriptor has no pricing data (all zeros).
    pub fn cost_for(&self, usage: &Usage) -> f64 {
        // Defensive: providers occasionally report `cache_read` greater
        // than `input` due to retry double-counting. Clamp to keep the
        // uncached share non-negative.
        let cached = usage.cache_read.min(usage.input) as f64;
        let uncached = (usage.input.saturating_sub(usage.cache_read)) as f64;
        let cache_write = usage.cache_write as f64;
        let output = usage.output as f64;
        (uncached * self.input
            + cached * self.cache_read
            + cache_write * self.cache_write
            + output * self.output)
            / 1_000_000.0
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    #[serde(default)]
    pub input: u64,
    #[serde(default)]
    pub output: u64,
    #[serde(default)]
    pub cache_read: u64,
    #[serde(default)]
    pub cache_write: u64,
    /// Subset of `cache_write` written with 1h retention. Only Anthropic
    /// reports this split (pi-ai `types.ts:373-374` @ 34239180,
    /// wire name `cacheWrite1h`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u64>,
    /// Reasoning/thinking tokens, when the provider reports them. A subset
    /// of `output` (already included there); `Some(0)` is meaningful —
    /// providers without a reasoning breakdown leave it `None`
    /// (pi-ai `types.ts:375-380`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default)]
    pub cost: Cost,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: String,
    pub provider: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub context_window: u64,
    #[serde(default)]
    pub max_tokens: u64,
    #[serde(default)]
    pub cost: Cost,
}

impl Model {
    /// Fallback "unknown" model used when no model descriptor is available.
    /// All identifying fields are set to `"unknown"` and pricing is zeroed.
    pub fn unknown() -> Self {
        Model {
            id: "unknown".into(),
            name: "unknown".into(),
            api: "unknown".into(),
            provider: "unknown".into(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Stop reasons, thinking levels, queue / execution modes
// ---------------------------------------------------------------------------

/// Assistant-message stop reason.
///
/// Mirrors the pi-ai `StopReason` union (`types.ts:391` @ 34239180:
/// `"pending" | "stop" | "length" | "toolUse" | "error" | "aborted"`).
/// `Refused` is a grain-side extension with no upstream counterpart, kept
/// for downstream crates that classify provider refusals distinctly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    /// The message is still streaming (partial snapshot).
    Pending,
    Stop,
    ToolUse,
    Length,
    Error,
    Aborted,
    Refused,
}

/// Thinking/reasoning level.
///
/// Mirrors the agent-facing union (`packages/agent/src/types.ts:294`
/// @ 34239180: `"off" | "minimal" | "low" | "medium" | "high" | "xhigh" |
/// "max"`). `xhigh` and `max` are only supported by selected model
/// families.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    #[default]
    Off,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

/// Token budgets for each thinking level (token-based providers only).
///
/// Port of the pi-ai `ThinkingBudgets` (`types.ts:92-98` @ 34239180).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingBudgets {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimal: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub medium: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolExecutionMode {
    Sequential,
    #[default]
    Parallel,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QueueMode {
    All,
    #[default]
    OneAtATime,
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UserMessage {
    /// Message content blocks.
    ///
    /// The upstream wire format admits `content: string |
    /// (TextContent | ImageContent)[]` (pi-ai types.ts:393-397): a bare
    /// string is a first-class user-message shape and deserializes as a
    /// single text block; a missing/null content normalizes to `[]`
    /// (upstream's `content ?? []` treatment). Serialization always emits
    /// the structured array form, which upstream equally accepts.
    #[serde(default, deserialize_with = "deserialize_user_content_list")]
    pub content: Vec<UserContent>,
    pub timestamp: i64,
}

/// Deserializer for [`UserMessage::content`]: accepts a bare string, the
/// structured block array, or null (patch-7; upstream wire format per pi-ai
/// types.ts:395).
fn deserialize_user_content_list<'de, D>(deserializer: D) -> Result<Vec<UserContent>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::String(text) => Ok(vec![UserContent::Text(TextContent { text })]),
        serde_json::Value::Array(_) => serde_json::from_value(value).map_err(D::Error::custom),
        serde_json::Value::Null => Ok(Vec::new()),
        other => Err(D::Error::custom(format!(
            "invalid user message content: expected string, array, or null, got {other}"
        ))),
    }
}

/// Machine-readable classification of a loop-terminating failure — the
/// structured companion to [`AssistantMessage::error_message`]'s free text.
///
/// Grain-side extension with no upstream counterpart (same stance as
/// [`StopReason::Refused`]): upstream pi carries failures as free-text
/// `errorMessage` only, which forces embedders to classify by substring
/// matching. This channel lets whatever produced the failure (a provider
/// adapter, an engine-bridging [`crate::stream::LlmStream`], a hook) attach
/// a code the embedder can `match` on; the loop preserves it verbatim onto
/// the final assistant message.
///
/// Producers with a reason outside the named variants use
/// [`ErrorCode::Other`] with their own code string, carried verbatim.
///
/// Wire form is a plain string (`"budget_exhausted"`,
/// `"agent_limit_exceeded"`, or the `Other` string). An `Other` value that
/// spells a named variant canonicalizes to that variant on round-trip.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// A resource budget (tokens, cost) was exhausted.
    BudgetExhausted,
    /// An agent/turn count limit was exceeded.
    AgentLimitExceeded,
    /// Any other producer-defined code, carried verbatim.
    #[serde(untagged)]
    Other(String),
}

impl ErrorCode {
    /// The code as a string, however it was produced.
    pub fn as_str(&self) -> &str {
        match self {
            ErrorCode::BudgetExhausted => "budget_exhausted",
            ErrorCode::AgentLimitExceeded => "agent_limit_exceeded",
            ErrorCode::Other(code) => code.as_str(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub content: Vec<AssistantContent>,
    pub api: String,
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Structured code for a loop-terminating failure, when the producer
    /// supplied one (see [`ErrorCode`]). Grain-side extension: absent from
    /// upstream transcripts and omitted from the wire when `None`, so
    /// serialized messages round-trip against upstream shapes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<ErrorCode>,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<UserContent>,
    #[serde(default)]
    pub details: serde_json::Value,
    /// Usage from the tool execution itself, if available. Not part of main
    /// LLM context accounting (pi-ai `types.ts:420-422` @ 34239180).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Names from the tool set that became available after this result.
    /// Present only when non-empty, matching upstream's conditional spread
    /// into the toolResult message (agent-loop.ts:783).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub added_tool_names: Option<Vec<String>>,
    pub is_error: bool,
    pub timestamp: i64,
}

/// Standard LLM message accepted by `convertToLlm`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
}

impl Message {
    /// Returns the serde tag for this message variant:
    /// `"user"`, `"assistant"`, or `"toolResult"`.
    pub fn role(&self) -> &'static str {
        match self {
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::ToolResult(_) => "toolResult",
        }
    }
}

/// Agent transcript message: a standard LLM message or an opaque custom payload.
///
/// Custom variants mirror the TypeScript `CustomAgentMessages` extension point:
/// applications stash app-specific records here and filter / convert them in
/// `convertToLlm` before reaching the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum AgentMessage {
    Standard(Message),
    Custom(serde_json::Value),
}

impl AgentMessage {
    /// Wrap a [`UserMessage`] in the `Standard` variant.
    pub fn user(message: UserMessage) -> Self {
        AgentMessage::Standard(Message::User(message))
    }
    /// Wrap an [`AssistantMessage`] in the `Standard` variant.
    pub fn assistant(message: AssistantMessage) -> Self {
        AgentMessage::Standard(Message::Assistant(message))
    }
    /// Wrap a [`ToolResultMessage`] in the `Standard` variant.
    pub fn tool_result(message: ToolResultMessage) -> Self {
        AgentMessage::Standard(Message::ToolResult(message))
    }

    /// Role tag for this message. For `Custom` variants, reads the
    /// `role` JSON field, defaulting to `"custom"` when absent.
    pub fn role(&self) -> &str {
        match self {
            AgentMessage::Standard(m) => m.role(),
            AgentMessage::Custom(v) => v.get("role").and_then(|r| r.as_str()).unwrap_or("custom"),
        }
    }

    /// Return the inner [`AssistantMessage`] when this is a `Standard`
    /// assistant message; `None` for user / tool-result / custom entries.
    pub fn as_assistant(&self) -> Option<&AssistantMessage> {
        match self {
            AgentMessage::Standard(Message::Assistant(m)) => Some(m),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tool result + tools
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentToolResult {
    pub content: Vec<UserContent>,
    #[serde(default)]
    pub details: serde_json::Value,
    /// Usage from the tool execution itself, if available. Merged into the
    /// persisted toolResult message (pi-ai `types.ts:360-361`,
    /// agent-loop.ts:782 @ 34239180).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Names of tools introduced by this result and available from this
    /// transcript point onward (pi-ai `types.ts:362-363`).
    #[serde(
        default,
        rename = "addedToolNames",
        skip_serializing_if = "Option::is_none"
    )]
    pub added_tool_names: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminate: Option<bool>,
}

impl AgentToolResult {
    /// Success result containing a single text block.
    pub fn text(message: impl Into<String>) -> Self {
        AgentToolResult {
            content: vec![UserContent::text(message)],
            details: serde_json::Value::Object(Default::default()),
            ..Default::default()
        }
    }

    /// Error result (currently an alias for [`Self::text`]; future-proofed
    /// so callers can switch on error semantics without renames).
    pub fn error(message: impl Into<String>) -> Self {
        AgentToolResult::text(message)
    }
}

pub type ToolUpdateCallback = Arc<dyn Fn(AgentToolResult) + Send + Sync>;

/// Serializable description of a tool, suitable for forwarding to an LLM provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinition {
    pub name: String,
    pub label: String,
    pub description: String,
    /// JSON Schema (typebox-equivalent) describing the parameters.
    #[serde(default)]
    pub parameters: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_mode: Option<ToolExecutionMode>,
}

/// Trait implemented by agent tools.
#[async_trait::async_trait]
pub trait AgentTool: Send + Sync {
    fn definition(&self) -> &ToolDefinition;

    /// Pre-processes raw tool-call arguments before schema validation.
    /// Defaults to identity.
    fn prepare_arguments(
        &self,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, AgentToolError> {
        Ok(args)
    }

    /// Validates arguments against the tool's JSON schema and returns the
    /// (potentially coerced) arguments the loop must pass to hooks and
    /// [`Self::execute`].
    ///
    /// The default implementation ports upstream `validateToolArguments`
    /// (pi-ai `utils/validation.ts:278-310`): schema-driven coercion followed
    /// by full JSON-Schema validation. Malformed arguments become an error
    /// tool result without the tool executing (agent-loop.ts:616-663).
    fn validate_arguments(
        &self,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, AgentToolError> {
        crate::validation::validate_tool_arguments(
            &self.definition().name,
            &self.definition().parameters,
            args,
        )
        .map_err(AgentToolError::Message)
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        args: serde_json::Value,
        cancel: CancellationToken,
        on_update: ToolUpdateCallback,
    ) -> Result<AgentToolResult, AgentToolError>;
}

impl dyn AgentTool {
    /// Shortcut for `self.definition().name`.
    pub fn name(&self) -> &str {
        &self.definition().name
    }

    /// Shortcut for `self.definition().execution_mode`.
    pub fn execution_mode(&self) -> Option<ToolExecutionMode> {
        self.definition().execution_mode
    }
}

impl fmt::Debug for dyn AgentTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentTool")
            .field("definition", self.definition())
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AgentToolError {
    #[error("{0}")]
    Message(String),
    #[error("operation aborted")]
    Aborted,
    #[error("validation failed: {0}")]
    Validation(String),
}

impl AgentToolError {
    /// Convenience constructor for a general-purpose error message.
    pub fn msg(s: impl Into<String>) -> Self {
        AgentToolError::Message(s.into())
    }
}

// ---------------------------------------------------------------------------
// Streaming events
// ---------------------------------------------------------------------------

/// Streaming events emitted while an assistant message is being produced.
///
/// Mirrors the event stream returned by `streamSimple` in `@earendil-works/pi-ai`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantMessageEvent {
    Start {
        partial: AssistantMessage,
    },
    TextStart {
        partial: AssistantMessage,
        content_index: usize,
    },
    TextDelta {
        partial: AssistantMessage,
        content_index: usize,
        delta: String,
    },
    TextEnd {
        partial: AssistantMessage,
        content_index: usize,
    },
    ThinkingStart {
        partial: AssistantMessage,
        content_index: usize,
    },
    ThinkingDelta {
        partial: AssistantMessage,
        content_index: usize,
        delta: String,
    },
    ThinkingEnd {
        partial: AssistantMessage,
        content_index: usize,
    },
    ToolcallStart {
        partial: AssistantMessage,
        content_index: usize,
    },
    ToolcallDelta {
        partial: AssistantMessage,
        content_index: usize,
        delta: String,
    },
    ToolcallEnd {
        partial: AssistantMessage,
        content_index: usize,
    },
    Done {
        result: AssistantMessage,
    },
    Error {
        error: String,
        result: AssistantMessage,
    },
}

impl AssistantMessageEvent {
    /// Returns a reference to the `partial` [`AssistantMessage`] snapshot
    /// carried by this event, if any. `Done` and `Error` events use a
    /// different field shape and return `None`.
    pub fn partial(&self) -> Option<&AssistantMessage> {
        match self {
            AssistantMessageEvent::Start { partial }
            | AssistantMessageEvent::TextStart { partial, .. }
            | AssistantMessageEvent::TextDelta { partial, .. }
            | AssistantMessageEvent::TextEnd { partial, .. }
            | AssistantMessageEvent::ThinkingStart { partial, .. }
            | AssistantMessageEvent::ThinkingDelta { partial, .. }
            | AssistantMessageEvent::ThinkingEnd { partial, .. }
            | AssistantMessageEvent::ToolcallStart { partial, .. }
            | AssistantMessageEvent::ToolcallDelta { partial, .. }
            | AssistantMessageEvent::ToolcallEnd { partial, .. } => Some(partial),
            _ => None,
        }
    }

    /// Returns `true` for terminal events (`Done` / `Error`).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// Agent-level events
// ---------------------------------------------------------------------------

/// Top-level events emitted by the agent loop.
///
/// `MessageUpdate` carries the largest payload (a full [`AssistantMessageEvent`]),
/// but events are emitted at low rate and frequently cloned by listeners, so the
/// variant size is acceptable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum AgentEvent {
    AgentStart,
    AgentEnd {
        messages: Vec<AgentMessage>,
    },
    TurnStart,
    TurnEnd {
        message: AssistantMessage,
        tool_results: Vec<ToolResultMessage>,
    },
    MessageStart {
        message: AgentMessage,
    },
    MessageUpdate {
        message: AssistantMessage,
        assistant_message_event: AssistantMessageEvent,
    },
    MessageEnd {
        message: AgentMessage,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
        partial_result: AgentToolResult,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: AgentToolResult,
        is_error: bool,
    },
}

// ---------------------------------------------------------------------------
// Context snapshots + state
// ---------------------------------------------------------------------------

/// LLM-facing context: filtered transcript ready to send to the model.
#[derive(Debug, Default, Clone)]
pub struct LlmContext {
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
}

/// Agent-facing context snapshot passed into the low-level loop.
#[derive(Default, Clone)]
pub struct AgentContext {
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<Arc<dyn AgentTool>>,
}

impl fmt::Debug for AgentContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentContext")
            .field("system_prompt", &self.system_prompt)
            .field("messages", &self.messages)
            .field(
                "tools",
                &self
                    .tools
                    .iter()
                    .map(|t| t.definition().name.clone())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Public snapshot of agent state.
#[derive(Clone)]
pub struct AgentState {
    pub system_prompt: String,
    pub model: Model,
    pub thinking_level: ThinkingLevel,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub messages: Vec<AgentMessage>,
    pub is_streaming: bool,
    pub streaming_message: Option<AgentMessage>,
    pub pending_tool_calls: HashSet<String>,
    pub error_message: Option<String>,
}

impl fmt::Debug for AgentState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentState")
            .field("system_prompt", &self.system_prompt)
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .field("tools", &self.tools.len())
            .field("messages", &self.messages.len())
            .field("is_streaming", &self.is_streaming)
            .field("streaming_message", &self.streaming_message.is_some())
            .field("pending_tool_calls", &self.pending_tool_calls)
            .field("error_message", &self.error_message)
            .finish()
    }
}

#[cfg(test)]
mod cost_tests {
    use super::*;

    fn pricing() -> Cost {
        // DeepSeek V4-flash-style numbers post-discount (per 1M tokens).
        Cost {
            input: 0.14,
            output: 0.28,
            cache_read: 0.0028,
            cache_write: 0.14,
            total: 0.0,
        }
    }

    #[test]
    fn cost_for_all_uncached_input() {
        let cost = pricing().cost_for(&Usage {
            input: 1_000_000,
            output: 1_000_000,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 2_000_000,
            cost: Cost::default(),
            ..Usage::default()
        });
        // 1M @ 0.14 + 1M @ 0.28 = 0.42
        assert!((cost - 0.42).abs() < 1e-9);
    }

    #[test]
    fn cost_for_fully_cached_input_uses_cache_rate() {
        let cost = pricing().cost_for(&Usage {
            input: 1_000_000,
            output: 0,
            cache_read: 1_000_000,
            cache_write: 0,
            total_tokens: 1_000_000,
            cost: Cost::default(),
            ..Usage::default()
        });
        // 0 uncached + 1M @ 0.0028 = 0.0028
        assert!((cost - 0.0028).abs() < 1e-9);
    }

    #[test]
    fn cost_for_partial_cache_splits_correctly() {
        let cost = pricing().cost_for(&Usage {
            input: 1_000_000,
            output: 500_000,
            cache_read: 800_000,
            cache_write: 0,
            total_tokens: 1_500_000,
            cost: Cost::default(),
            ..Usage::default()
        });
        // 200k @ 0.14 + 800k @ 0.0028 + 500k @ 0.28 = 0.028 + 0.00224 + 0.14
        let expected = 0.2e6 * 0.14 / 1e6 + 0.8e6 * 0.0028 / 1e6 + 0.5e6 * 0.28 / 1e6;
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn cost_for_clamps_cache_read_above_input() {
        // Providers occasionally report cache_read > input due to retries.
        // Must not produce a negative uncached share.
        let cost = pricing().cost_for(&Usage {
            input: 1_000,
            output: 0,
            cache_read: 9_999,
            cache_write: 0,
            total_tokens: 1_000,
            cost: Cost::default(),
            ..Usage::default()
        });
        // Uncached clamps to 0; treat all 1_000 as cached.
        let expected = 1_000.0 * 0.0028 / 1_000_000.0;
        assert!((cost - expected).abs() < 1e-12);
    }

    #[test]
    fn cost_for_zero_pricing_yields_zero() {
        let cost = Cost::default().cost_for(&Usage {
            input: 999_999,
            output: 999_999,
            ..Usage::default()
        });
        assert_eq!(cost, 0.0);
    }
}
