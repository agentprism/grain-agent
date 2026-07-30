//! Anthropic Messages request construction for the native transport.
//!
//! **Deliberately narrow.** This builds only the request shape the grain
//! harness actually emits — it is not a reimplementation of genai's Anthropic
//! adapter. The supported surface is enumerated in [`SUPPORTED_SURFACE`] and
//! anything outside it is rejected by [`UnsupportedFeature`] *at request
//! time*, so an unsupported knob fails loudly instead of silently producing a
//! request the caller did not ask for.
//!
//! Audit basis: at the WP25 pin, no crate in the workspace sets a single
//! `genai::chat::ChatOptions` request field (`temperature`, `max_tokens`,
//! `top_p`, `stop_sequences`, `tool_choice`, `cache_control`, `extra_body`,
//! `seed`, `service_tier`, `extra_headers` — verified by grep across all
//! members). The harness drives the provider entirely through
//! `LlmContext` (system prompt, messages, tools) plus
//! `StreamOptions::reasoning`. That is exactly what this module covers.

use grain_agent_core::{
    AssistantContent, AssistantMessage, LlmContext, Message, Model, StreamOptions, ThinkingLevel,
    ToolDefinition, ToolResultMessage, UserContent, UserMessage,
};
use serde_json::{Map, Value, json};

/// The Anthropic API version header this transport pins.
///
/// Matches genai 0.6.5 (`adapter_shared.rs::ANTHROPIC_VERSION`) so switching
/// backends does not change how the provider interprets the request.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Human-readable inventory of what the native transport supports, used in
/// error messages and reproduced in the module docs / README.
pub const SUPPORTED_SURFACE: &str = "system prompt, user/assistant/tool-result messages \
     (text, images, thinking with signatures, tool calls), tool definitions, \
     streaming, max_tokens, and extended thinking via StreamOptions::reasoning";

/// A request feature the caller asked for that this transport cannot honor.
///
/// Returned instead of quietly dropping the field. Every variant names the
/// caller-visible knob, not an internal detail, so the message is actionable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedFeature {
    /// The caller-facing name of the unsupported knob.
    pub feature: &'static str,
    /// Why it cannot be honored, and what to do instead.
    pub detail: String,
}

impl UnsupportedFeature {
    fn new(feature: &'static str, detail: impl Into<String>) -> Self {
        UnsupportedFeature {
            feature,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for UnsupportedFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the native Anthropic transport does not support `{}`: {}. \
             Supported surface: {}. Unset the option, or select the genai \
             backend (the default), which is the broader-coverage path.",
            self.feature, self.detail, SUPPORTED_SURFACE
        )
    }
}

impl std::error::Error for UnsupportedFeature {}

/// Reject any `StreamOptions` knob this transport would otherwise ignore.
///
/// Only fields that are actually **set** are rejected — a `None` field asks
/// for nothing, so honoring it is vacuous. This is stricter than the genai
/// backend, which silently ignores the same knobs (tracked as debt item G6);
/// making the gap loud here is the point.
pub fn reject_unsupported_options(options: &StreamOptions) -> Result<(), UnsupportedFeature> {
    if options.api_key.is_some() {
        return Err(UnsupportedFeature::new(
            "StreamOptions::api_key",
            "per-call API keys are not plumbed; this transport resolves auth once at \
             construction time",
        ));
    }
    if options.session_id.is_some() {
        return Err(UnsupportedFeature::new(
            "StreamOptions::session_id",
            "the Anthropic Messages API has no session concept on this path",
        ));
    }
    if let Some(transport) = &options.transport {
        return Err(UnsupportedFeature::new(
            "StreamOptions::transport",
            format!("no alternate transport is implemented (requested {transport:?})"),
        ));
    }
    if options.max_retry_delay_ms.is_some() {
        return Err(UnsupportedFeature::new(
            "StreamOptions::max_retry_delay_ms",
            "this transport performs no automatic retries, so a retry delay would be \
             silently meaningless",
        ));
    }
    if options.on_payload.is_some() {
        return Err(UnsupportedFeature::new(
            "StreamOptions::on_payload",
            "the payload inspection/replacement hook is not invoked by this transport, \
             so a caller relying on it would silently send an unmodified request",
        ));
    }
    if options.on_response.is_some() {
        return Err(UnsupportedFeature::new(
            "StreamOptions::on_response",
            "the response hook is not invoked by this transport",
        ));
    }
    if !options.extra.is_null() {
        return Err(UnsupportedFeature::new(
            "StreamOptions::extra",
            "provider-specific extras are not merged into the request body",
        ));
    }
    Ok(())
}

/// Model-appropriate `max_tokens`, which Anthropic requires on every request.
///
/// Mirrors genai 0.6.5 `AnthropicAdapter::resolve_max_tokens` value for value
/// so that switching backends does not change the generation ceiling. Nothing
/// in the workspace sets an explicit `max_tokens` today, so this default is
/// what every request actually carries.
pub fn resolve_max_tokens(model_name: &str) -> u32 {
    const MAX_TOKENS_64K: u32 = 64000;
    const MAX_TOKENS_32K: u32 = 32000;
    const MAX_TOKENS_8K: u32 = 8192;
    const MAX_TOKENS_4K: u32 = 4096;

    if model_name.contains("claude-sonnet")
        || model_name.contains("claude-haiku")
        || model_name.contains("claude-3-7-sonnet")
        || model_name.contains("claude-opus-4-5")
    {
        MAX_TOKENS_64K
    } else if model_name.contains("claude-opus-4") {
        MAX_TOKENS_32K
    } else if model_name.contains("claude-3-5") {
        MAX_TOKENS_8K
    } else if model_name.contains("3-opus") || model_name.contains("3-haiku") {
        MAX_TOKENS_4K
    } else {
        MAX_TOKENS_64K
    }
}

/// Extended-thinking budget for a grain [`ThinkingLevel`].
///
/// Caller-supplied [`grain_agent_core::ThinkingBudgets`] win; otherwise the
/// defaults match genai's legacy budget constants (`REASONING_LOW` 1024,
/// `REASONING_MEDIUM` 8000, `REASONING_HIGH` 24000).
///
/// **Known divergence from the genai backend — and the largest unverified
/// assumption in this transport.** genai additionally emits the newer
/// `output_config.effort` and `thinking: {type:"adaptive"}` shapes for the
/// model families that support them, gated on hard-coded model-name lists.
/// This transport always emits the legacy
/// `thinking: {type:"enabled", budget_tokens}` shape.
///
/// The reasoning for the simpler shape is that replicating genai's
/// model-family tables duplicates exactly the kind of list that drifts
/// silently. The *assumption* is that every extended-thinking Anthropic model
/// still accepts the legacy shape, including the adaptive-thinking era ones.
/// **That assumption is untested.** It is stated here as an assumption rather
/// than a fact because only a live request against an adaptive-era model can
/// settle it, and no fixture at the pin exercises thinking at all. If it is
/// wrong, the failure is a request-time 400 on exactly those models — loud,
/// not silent, but a hard failure for anyone who enables thinking on them.
/// Tracked as deferred item W5; it is the item most in need of live access.
fn thinking_budget(level: ThinkingLevel, options: &StreamOptions) -> Option<u64> {
    let budgets = options.thinking_budgets;
    let explicit = budgets.and_then(|b| match level {
        ThinkingLevel::Minimal => b.minimal,
        ThinkingLevel::Low => b.low,
        ThinkingLevel::Medium => b.medium,
        ThinkingLevel::High | ThinkingLevel::XHigh | ThinkingLevel::Max => b.high,
        ThinkingLevel::Off => None,
    });
    if explicit.is_some() {
        return explicit;
    }
    match level {
        ThinkingLevel::Off => None,
        ThinkingLevel::Minimal | ThinkingLevel::Low => Some(1024),
        ThinkingLevel::Medium => Some(8000),
        ThinkingLevel::High | ThinkingLevel::XHigh | ThinkingLevel::Max => Some(24000),
    }
}

/// Build the JSON body for `POST /messages`.
pub fn build_request(
    model: &Model,
    ctx: &LlmContext,
    options: &StreamOptions,
) -> Result<Value, UnsupportedFeature> {
    reject_unsupported_options(options)?;

    let model_name = bare_model_name(&model.id);
    let mut payload = Map::new();
    payload.insert("model".into(), json!(model_name));
    payload.insert("stream".into(), json!(true));
    payload.insert(
        "max_tokens".into(),
        json!(resolve_max_tokens(model_name.as_str())),
    );

    if !ctx.system_prompt.is_empty() {
        payload.insert("system".into(), json!(ctx.system_prompt));
    }

    let corrupt = corrupt_tool_call_ids(&ctx.messages);
    let messages = build_messages(&ctx.messages, &corrupt);
    payload.insert("messages".into(), Value::Array(messages));

    if !ctx.tools.is_empty() {
        let tools: Vec<Value> = ctx.tools.iter().map(tool_to_json).collect();
        payload.insert("tools".into(), Value::Array(tools));
    }

    if let Some(level) = options.reasoning
        && level != ThinkingLevel::Off
        && let Some(budget) = thinking_budget(level, options)
    {
        payload.insert(
            "thinking".into(),
            json!({"type": "enabled", "budget_tokens": budget}),
        );
    }

    Ok(Value::Object(payload))
}

/// Strip a grain `provider/model` namespace down to the provider-native id.
fn bare_model_name(model_id: &str) -> String {
    match model_id.split_once('/') {
        Some((_, name)) => name.to_string(),
        None => model_id.to_string(),
    }
}

/// Tool calls whose arguments never resolved to an object. Mirrors the genai
/// path's guard (`mapping::outbound::collect_corrupt_tool_call_ids`): the
/// provider rejects a `tool_use` whose `input` is not an object, so the call
/// and its paired `tool_result` are dropped together rather than failing the
/// whole turn.
fn corrupt_tool_call_ids(messages: &[Message]) -> std::collections::HashSet<String> {
    let mut ids = std::collections::HashSet::new();
    for msg in messages {
        if let Message::Assistant(a) = msg {
            for c in &a.content {
                if let AssistantContent::ToolCall(tc) = c
                    && !tc.arguments.is_object()
                {
                    ids.insert(tc.id.clone());
                }
            }
        }
    }
    ids
}

fn build_messages(
    messages: &[Message],
    corrupt: &std::collections::HashSet<String>,
) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    for msg in messages {
        match msg {
            Message::User(u) => out.push(user_message(u)),
            Message::Assistant(a) => {
                if let Some(m) = assistant_message(a, corrupt) {
                    out.push(m);
                }
            }
            Message::ToolResult(t) => {
                if !corrupt.contains(&t.tool_call_id) {
                    out.push(tool_result_message(t));
                }
            }
        }
    }
    out
}

/// The opaque payload of a redacted-thinking block, if this is one.
///
/// Stored in `provider_metadata` by the stream state machine; see the
/// `redacted_thinking` arm in [`crate::anthropic::state`].
fn redacted_thinking_data(t: &grain_agent_core::ThinkingContent) -> Option<String> {
    let meta = t.provider_metadata.as_ref()?;
    if meta.get("type").and_then(Value::as_str)? != "redacted_thinking" {
        return None;
    }
    meta.get("data")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Project user-side content blocks, dropping empty text.
///
/// Anthropic rejects a `text` block whose `text` is empty
/// ("text content blocks must be non-empty"), which fails the **whole
/// request**, not just the block. The assistant path has always guarded this;
/// the user and tool-result paths did not — and empty tool output is entirely
/// routine in a coding harness (a silent `write`, a command with no stdout),
/// so this was the likeliest of the empty-block cases to fire in practice.
fn user_content_blocks(content: &[UserContent]) -> Vec<Value> {
    content
        .iter()
        .filter(|c| !matches!(c, UserContent::Text(t) if t.text.is_empty()))
        .map(user_content_block)
        .collect()
}

fn user_content_block(c: &UserContent) -> Value {
    match c {
        UserContent::Text(t) => json!({"type": "text", "text": t.text}),
        UserContent::Image(i) => json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": i.mime_type,
                "data": i.data,
            }
        }),
    }
}

fn user_message(u: &UserMessage) -> Value {
    json!({
        "role": "user",
        "content": user_content_blocks(&u.content),
    })
}

fn assistant_message(
    a: &AssistantMessage,
    corrupt: &std::collections::HashSet<String>,
) -> Option<Value> {
    let mut blocks: Vec<Value> = Vec::with_capacity(a.content.len());
    for c in &a.content {
        match c {
            AssistantContent::Text(t) if !t.text.is_empty() => {
                blocks.push(json!({"type": "text", "text": t.text}));
            }
            AssistantContent::Text(_) => {}
            AssistantContent::Image(i) => blocks.push(json!({
                "type": "image",
                "source": {"type": "base64", "media_type": i.mime_type, "data": i.data}
            })),
            // A redacted-thinking block replays verbatim as
            // `{type:"redacted_thinking", data}` -- it has no signature and
            // must not be judged by the signed-thinking rule below, or it
            // would be dropped and the reasoning chain would break.
            AssistantContent::Thinking(t) if redacted_thinking_data(t).is_some() => {
                let data = redacted_thinking_data(t).unwrap_or_default();
                blocks.push(json!({"type": "redacted_thinking", "data": data}));
            }
            // Anthropic only accepts an ordinary thinking block back when it
            // carries the provider's signature; an unsigned one is rejected,
            // so it is dropped from the replay exactly as the genai path does.
            AssistantContent::Thinking(t) => {
                if let Some(sig) = &t.signature {
                    blocks.push(json!({
                        "type": "thinking",
                        "thinking": t.thinking,
                        "signature": sig,
                    }));
                }
            }
            AssistantContent::ToolCall(tc) => {
                if corrupt.contains(&tc.id) {
                    continue;
                }
                blocks.push(json!({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }));
            }
        }
    }
    if blocks.is_empty() {
        return None;
    }
    Some(json!({"role": "assistant", "content": blocks}))
}

fn tool_result_message(t: &ToolResultMessage) -> Value {
    let content: Vec<Value> = user_content_blocks(&t.content);
    let mut block = Map::new();
    block.insert("type".into(), json!("tool_result"));
    block.insert("tool_use_id".into(), json!(t.tool_call_id));
    block.insert("content".into(), Value::Array(content));
    if t.is_error {
        block.insert("is_error".into(), json!(true));
    }
    json!({"role": "user", "content": [Value::Object(block)]})
}

fn tool_to_json(t: &ToolDefinition) -> Value {
    json!({
        "name": t.name,
        "description": t.description,
        "input_schema": t.parameters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use grain_agent_core::{TextContent, ToolCall};

    fn model() -> Model {
        Model {
            id: "anthropic/claude-haiku-4-5".into(),
            name: "Claude Haiku 4.5".into(),
            api: "anthropic-messages".into(),
            provider: "anthropic".into(),
            ..Default::default()
        }
    }

    fn ctx_with(messages: Vec<Message>) -> LlmContext {
        LlmContext {
            system_prompt: String::new(),
            messages,
            tools: Vec::new(),
        }
    }

    fn user(text: &str) -> Message {
        Message::User(UserMessage {
            content: vec![UserContent::text(text)],
            timestamp: 0,
        })
    }

    #[test]
    fn builds_the_minimal_request_shape() {
        let req = build_request(
            &model(),
            &ctx_with(vec![user("hello")]),
            &StreamOptions::default(),
        )
        .expect("supported");

        assert_eq!(req["model"], json!("claude-haiku-4-5"), "namespace stripped");
        assert_eq!(req["stream"], json!(true));
        assert_eq!(req["max_tokens"], json!(64000));
        assert!(req.get("system").is_none(), "empty system prompt omitted");
        assert!(req.get("tools").is_none(), "empty tool set omitted");
        assert_eq!(
            req["messages"],
            json!([{"role":"user","content":[{"type":"text","text":"hello"}]}])
        );
    }

    #[test]
    fn system_prompt_and_tools_are_attached_when_present() {
        let ctx = LlmContext {
            system_prompt: "be brief".into(),
            messages: vec![user("hi")],
            tools: vec![ToolDefinition {
                name: "read".into(),
                label: "read".into(),
                description: "read a file".into(),
                parameters: json!({"type":"object","properties":{}}),
                execution_mode: None,
            }],
        };
        let req = build_request(&model(), &ctx, &StreamOptions::default()).expect("supported");
        assert_eq!(req["system"], json!("be brief"));
        assert_eq!(req["tools"][0]["name"], json!("read"));
        assert_eq!(req["tools"][0]["input_schema"]["type"], json!("object"));
    }

    #[test]
    fn max_tokens_matches_the_genai_backend_per_model_family() {
        assert_eq!(resolve_max_tokens("claude-haiku-4-5"), 64000);
        assert_eq!(resolve_max_tokens("claude-sonnet-4-5"), 64000);
        assert_eq!(resolve_max_tokens("claude-opus-4-5"), 64000);
        assert_eq!(resolve_max_tokens("claude-opus-4-1"), 32000);
        assert_eq!(resolve_max_tokens("claude-3-5-sonnet"), 8192);
        assert_eq!(resolve_max_tokens("claude-3-opus"), 4096);
        assert_eq!(resolve_max_tokens("something-else"), 64000);
    }

    #[test]
    fn thinking_is_requested_only_when_reasoning_is_on() {
        let off = build_request(
            &model(),
            &ctx_with(vec![user("hi")]),
            &StreamOptions::default(),
        )
        .expect("supported");
        assert!(off.get("thinking").is_none());

        let on = build_request(
            &model(),
            &ctx_with(vec![user("hi")]),
            &StreamOptions {
                reasoning: Some(ThinkingLevel::Medium),
                ..StreamOptions::default()
            },
        )
        .expect("supported");
        assert_eq!(
            on["thinking"],
            json!({"type": "enabled", "budget_tokens": 8000})
        );
    }

    #[test]
    fn caller_supplied_thinking_budgets_win() {
        let req = build_request(
            &model(),
            &ctx_with(vec![user("hi")]),
            &StreamOptions {
                reasoning: Some(ThinkingLevel::High),
                thinking_budgets: Some(grain_agent_core::ThinkingBudgets {
                    high: Some(5555),
                    ..Default::default()
                }),
                ..StreamOptions::default()
            },
        )
        .expect("supported");
        assert_eq!(req["thinking"]["budget_tokens"], json!(5555));
    }

    #[test]
    fn signed_thinking_replays_and_unsigned_is_dropped() {
        use grain_agent_core::{StopReason, ThinkingContent, Usage};
        let assistant = |signature: Option<String>| {
            Message::Assistant(AssistantMessage {
                content: vec![
                    AssistantContent::Thinking(ThinkingContent {
                        thinking: "hmm".into(),
                        signature,
                        provider_metadata: None,
                    }),
                    AssistantContent::Text(TextContent { text: "hi".into() }),
                ],
                api: "anthropic-messages".into(),
                provider: "anthropic".into(),
                model: "anthropic/claude-haiku-4-5".into(),
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                error_code: None,
                timestamp: 0,
            })
        };

        let signed = build_request(
            &model(),
            &ctx_with(vec![assistant(Some("sig-abc".into()))]),
            &StreamOptions::default(),
        )
        .expect("supported");
        assert_eq!(signed["messages"][0]["content"][0]["type"], json!("thinking"));
        assert_eq!(
            signed["messages"][0]["content"][0]["signature"],
            json!("sig-abc")
        );

        let unsigned = build_request(
            &model(),
            &ctx_with(vec![assistant(None)]),
            &StreamOptions::default(),
        )
        .expect("supported");
        assert_eq!(
            unsigned["messages"][0]["content"][0]["type"],
            json!("text"),
            "an unsigned thinking block must not be replayed"
        );
    }

    #[test]
    fn corrupt_tool_call_and_its_result_are_dropped_together() {
        use grain_agent_core::{StopReason, Usage};
        let assistant = Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: "call_bad".into(),
                name: "edit".into(),
                // Never resolved to an object — a truncated stream.
                arguments: json!(r#"{"path":"/x","text":"abc"#),
            })],
            api: "anthropic-messages".into(),
            provider: "anthropic".into(),
            model: "anthropic/claude-haiku-4-5".into(),
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            error_code: None,
            timestamp: 0,
        });
        let result = Message::ToolResult(ToolResultMessage {
            tool_call_id: "call_bad".into(),
            tool_name: "edit".into(),
            content: vec![UserContent::text("ok")],
            details: Value::Null,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 0,
        });

        let req = build_request(
            &model(),
            &ctx_with(vec![user("go"), assistant, result]),
            &StreamOptions::default(),
        )
        .expect("supported");
        assert_eq!(
            req["messages"].as_array().unwrap().len(),
            1,
            "only the user message survives"
        );
    }

    /// W4: the redacted block must survive a full round trip — captured by
    /// the state machine, replayed verbatim on the next request.
    #[test]
    fn redacted_thinking_replays_verbatim() {
        use grain_agent_core::{StopReason, ThinkingContent, Usage};
        let assistant = Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::Thinking(ThinkingContent {
                thinking: String::new(),
                signature: None,
                provider_metadata: Some(json!({
                    "type": "redacted_thinking",
                    "data": "EroBCkYIB..."
                })),
            })],
            api: "anthropic-messages".into(),
            provider: "anthropic".into(),
            model: "anthropic/claude-haiku-4-5".into(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            error_code: None,
            timestamp: 0,
        });

        let req = build_request(
            &model(),
            &ctx_with(vec![assistant]),
            &StreamOptions::default(),
        )
        .expect("supported");

        assert_eq!(
            req["messages"][0]["content"][0],
            json!({"type": "redacted_thinking", "data": "EroBCkYIB..."}),
            "a redacted block has no signature and must not be dropped by the \
             signed-thinking rule"
        );
    }

    /// W6: Anthropic rejects empty `text` blocks, failing the whole request.
    /// Empty tool output is routine in a coding harness.
    #[test]
    fn empty_text_blocks_are_dropped_from_user_and_tool_result_messages() {
        let empty_result = Message::ToolResult(ToolResultMessage {
            tool_call_id: "call_1".into(),
            tool_name: "bash".into(),
            content: vec![UserContent::text("")],
            details: Value::Null,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 0,
        });
        let req = build_request(
            &model(),
            &ctx_with(vec![empty_result]),
            &StreamOptions::default(),
        )
        .expect("supported");
        assert_eq!(
            req["messages"][0]["content"][0]["content"],
            json!([]),
            "an empty tool result must not emit an empty text block"
        );

        let mixed = Message::User(UserMessage {
            content: vec![
                UserContent::text(""),
                UserContent::text("real"),
                UserContent::text(""),
            ],
            timestamp: 0,
        });
        let req = build_request(&model(), &ctx_with(vec![mixed]), &StreamOptions::default())
            .expect("supported");
        assert_eq!(
            req["messages"][0]["content"],
            json!([{"type": "text", "text": "real"}]),
            "only the non-empty block survives"
        );
    }

    #[test]
    fn non_empty_user_content_is_untouched() {
        let req = build_request(
            &model(),
            &ctx_with(vec![user("hello")]),
            &StreamOptions::default(),
        )
        .expect("supported");
        assert_eq!(
            req["messages"][0]["content"],
            json!([{"type": "text", "text": "hello"}])
        );
    }

    // -- loud failures -------------------------------------------------------

    #[test]
    fn every_unsupported_stream_option_fails_loudly() {
        let cases: Vec<(&str, StreamOptions)> = vec![
            (
                "StreamOptions::api_key",
                StreamOptions {
                    api_key: Some("k".into()),
                    ..Default::default()
                },
            ),
            (
                "StreamOptions::session_id",
                StreamOptions {
                    session_id: Some("s".into()),
                    ..Default::default()
                },
            ),
            (
                "StreamOptions::transport",
                StreamOptions {
                    transport: Some("t".into()),
                    ..Default::default()
                },
            ),
            (
                "StreamOptions::max_retry_delay_ms",
                StreamOptions {
                    max_retry_delay_ms: Some(1),
                    ..Default::default()
                },
            ),
            (
                "StreamOptions::extra",
                StreamOptions {
                    extra: json!({"a": 1}),
                    ..Default::default()
                },
            ),
        ];

        for (feature, options) in cases {
            let err = build_request(&model(), &ctx_with(vec![user("hi")]), &options)
                .expect_err(&format!("{feature} must be rejected"));
            assert_eq!(err.feature, feature);
            let rendered = err.to_string();
            assert!(
                rendered.contains(feature) && rendered.contains("genai backend"),
                "the error must name the knob and point at the fallback: {rendered}"
            );
        }
    }

    #[test]
    fn default_stream_options_are_fully_supported() {
        // The shape the agent loop actually sends when nothing exotic is
        // configured must never trip the guard.
        assert!(reject_unsupported_options(&StreamOptions::default()).is_ok());
        assert!(
            reject_unsupported_options(&StreamOptions {
                reasoning: Some(ThinkingLevel::High),
                thinking_budgets: None,
                ..Default::default()
            })
            .is_ok()
        );
    }
}
