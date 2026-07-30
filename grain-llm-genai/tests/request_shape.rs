//! Permanent request-shape goldens for the native Anthropic transport.
//!
//! # Why this exists
//!
//! The seam-vector suite asserts only what the transport *decodes*. It makes
//! **zero assertions about what the transport sends**, which is the other half
//! of "is this request the one the caller asked for?" — and the half that a
//! live API judges first. Two of the defects found in review
//! (`redacted_thinking` losing its payload on replay, and empty text blocks
//! being emitted where Anthropic rejects them) were request-side, offline
//! determinable, and invisible to every existing test.
//!
//! These goldens capture the exact JSON body for representative contexts. They
//! are not a substitute for live verification — a golden proves the body is
//! *stable and intended*, never that Anthropic *accepts* it — but they turn
//! any silent change in request shape into a failing diff, and they document
//! the wire contract in one readable place.
//!
//! # What a golden change means
//!
//! If one of these fails, the request the harness sends has changed. That is
//! sometimes correct — but it must be a decision, not a side effect. Update
//! the golden in the same commit as the behavior, and say why in the message.
//!
//! Two assertions here are load-bearing beyond formatting:
//!
//! - **no empty `text` blocks anywhere** — Anthropic rejects them and fails
//!   the whole request;
//! - **`max_tokens` present on every request** — Anthropic requires it.
//!
//! Both are asserted structurally over every golden, so they cannot regress
//! through a body this file does not happen to enumerate.

use grain_agent_core::{
    AssistantContent, AssistantMessage, LlmContext, Message, Model, StopReason, StreamOptions,
    TextContent, ThinkingContent, ThinkingLevel, ToolCall, ToolDefinition, ToolResultMessage,
    Usage, UserContent, UserMessage,
};
use grain_llm_genai::anthropic::build_request;
use serde_json::{Value, json};

fn model() -> Model {
    Model {
        id: "anthropic/claude-sonnet-4-5".into(),
        name: "Claude Sonnet 4.5".into(),
        api: "anthropic-messages".into(),
        provider: "anthropic".into(),
        ..Default::default()
    }
}

fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: vec![UserContent::Text(TextContent { text: text.into() })],
        timestamp: 0,
    })
}

fn assistant(content: Vec<AssistantContent>) -> Message {
    Message::Assistant(AssistantMessage {
        content,
        api: "anthropic-messages".into(),
        provider: "anthropic".into(),
        model: "anthropic/claude-sonnet-4-5".into(),
        usage: Usage::default(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        error_code: None,
        timestamp: 0,
    })
}

fn tool_result(id: &str, text: &str, is_error: bool) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: id.into(),
        tool_name: "bash".into(),
        content: vec![UserContent::Text(TextContent { text: text.into() })],
        details: Value::Null,
        usage: None,
        added_tool_names: None,
        is_error,
        timestamp: 0,
    })
}

fn build(ctx: &LlmContext, options: &StreamOptions) -> Value {
    build_request(&model(), ctx, options).expect("supported request")
}

// ---------------------------------------------------------------------------
// Structural invariants — applied to every golden below
// ---------------------------------------------------------------------------

/// Anthropic rejects a `text` block whose `text` is empty, failing the entire
/// request. Walk the whole body rather than trusting per-case assertions.
fn assert_no_empty_text_blocks(v: &Value, path: &str) {
    match v {
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("text")
                && let Some(text) = map.get("text").and_then(Value::as_str)
            {
                assert!(
                    !text.is_empty(),
                    "empty text block at {path} — Anthropic rejects the whole request"
                );
            }
            for (k, child) in map {
                assert_no_empty_text_blocks(child, &format!("{path}.{k}"));
            }
        }
        Value::Array(items) => {
            for (i, child) in items.iter().enumerate() {
                assert_no_empty_text_blocks(child, &format!("{path}[{i}]"));
            }
        }
        _ => {}
    }
}

fn assert_request_invariants(req: &Value) {
    assert_no_empty_text_blocks(req, "$");
    assert!(
        req.get("max_tokens").and_then(Value::as_u64).is_some(),
        "Anthropic requires max_tokens on every request: {req}"
    );
    assert_eq!(req["stream"], json!(true), "the transport only streams");
    assert!(
        req.get("model").and_then(Value::as_str).is_some_and(|m| !m.contains('/')),
        "the grain namespace must be stripped from the model id: {req}"
    );
}

// ---------------------------------------------------------------------------
// Goldens
// ---------------------------------------------------------------------------

#[test]
fn golden_minimal_turn() {
    let ctx = LlmContext {
        system_prompt: String::new(),
        messages: vec![user("hello")],
        tools: Vec::new(),
    };
    let req = build(&ctx, &StreamOptions::default());
    assert_request_invariants(&req);

    assert_eq!(
        req,
        json!({
            "model": "claude-sonnet-4-5",
            "stream": true,
            "max_tokens": 64000,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]}
            ]
        })
    );
}

#[test]
fn golden_system_prompt_and_tools() {
    let ctx = LlmContext {
        system_prompt: "Be concise.".into(),
        messages: vec![user("list files")],
        tools: vec![ToolDefinition {
            name: "bash".into(),
            label: "bash".into(),
            description: "Run a shell command".into(),
            parameters: json!({
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"]
            }),
            execution_mode: None,
        }],
    };
    let req = build(&ctx, &StreamOptions::default());
    assert_request_invariants(&req);

    assert_eq!(req["system"], json!("Be concise."));
    assert_eq!(
        req["tools"],
        json!([{
            "name": "bash",
            "description": "Run a shell command",
            "input_schema": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"]
            }
        }])
    );
}

/// A full tool round trip, including the empty tool output that is routine in
/// a coding harness (a command that writes nothing to stdout).
#[test]
fn golden_tool_round_trip_with_empty_output() {
    let ctx = LlmContext {
        system_prompt: String::new(),
        messages: vec![
            user("touch a file"),
            assistant(vec![AssistantContent::ToolCall(ToolCall {
                id: "toolu_1".into(),
                name: "bash".into(),
                arguments: json!({"cmd": "touch x"}),
            })]),
            tool_result("toolu_1", "", false),
            user("done?"),
        ],
        tools: Vec::new(),
    };
    let req = build(&ctx, &StreamOptions::default());
    assert_request_invariants(&req);

    assert_eq!(
        req["messages"],
        json!([
            {"role": "user", "content": [{"type": "text", "text": "touch a file"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "bash",
                 "input": {"cmd": "touch x"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": []}
            ]},
            {"role": "user", "content": [{"type": "text", "text": "done?"}]}
        ]),
        "empty tool output must yield an empty content array, never an empty \
         text block"
    );
}

#[test]
fn golden_tool_error_result_sets_is_error() {
    let ctx = LlmContext {
        system_prompt: String::new(),
        messages: vec![tool_result("toolu_9", "command not found", true)],
        tools: Vec::new(),
    };
    let req = build(&ctx, &StreamOptions::default());
    assert_request_invariants(&req);

    assert_eq!(
        req["messages"][0]["content"][0],
        json!({
            "type": "tool_result",
            "tool_use_id": "toolu_9",
            "content": [{"type": "text", "text": "command not found"}],
            "is_error": true
        })
    );
}

/// Signed thinking replays; unsigned thinking is dropped (Anthropic rejects
/// it); redacted thinking replays verbatim as its opaque payload.
#[test]
fn golden_thinking_replay_variants() {
    let ctx = LlmContext {
        system_prompt: String::new(),
        messages: vec![assistant(vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "signed reasoning".into(),
                signature: Some("sig-abc".into()),
                provider_metadata: None,
            }),
            AssistantContent::Thinking(ThinkingContent {
                thinking: "unsigned reasoning".into(),
                signature: None,
                provider_metadata: None,
            }),
            AssistantContent::Thinking(ThinkingContent {
                thinking: String::new(),
                signature: None,
                provider_metadata: Some(json!({
                    "type": "redacted_thinking",
                    "data": "EroBCkYIB..."
                })),
            }),
            AssistantContent::Text(TextContent {
                text: "answer".into(),
            }),
        ])],
        tools: Vec::new(),
    };
    let req = build(&ctx, &StreamOptions::default());
    assert_request_invariants(&req);

    assert_eq!(
        req["messages"][0]["content"],
        json!([
            {"type": "thinking", "thinking": "signed reasoning", "signature": "sig-abc"},
            {"type": "redacted_thinking", "data": "EroBCkYIB..."},
            {"type": "text", "text": "answer"}
        ]),
        "unsigned thinking is dropped; redacted thinking survives verbatim"
    );
}

#[test]
fn golden_extended_thinking_request() {
    let ctx = LlmContext {
        system_prompt: String::new(),
        messages: vec![user("think hard")],
        tools: Vec::new(),
    };
    let req = build(
        &ctx,
        &StreamOptions {
            reasoning: Some(ThinkingLevel::High),
            ..StreamOptions::default()
        },
    );
    assert_request_invariants(&req);

    assert_eq!(
        req["thinking"],
        json!({"type": "enabled", "budget_tokens": 24000})
    );
}

#[test]
fn golden_images_use_base64_source_blocks() {
    let ctx = LlmContext {
        system_prompt: String::new(),
        messages: vec![Message::User(UserMessage {
            content: vec![
                UserContent::Text(TextContent {
                    text: "what is this?".into(),
                }),
                UserContent::Image(grain_agent_core::ImageContent {
                    data: "aGVsbG8=".into(),
                    mime_type: "image/png".into(),
                }),
            ],
            timestamp: 0,
        })],
        tools: Vec::new(),
    };
    let req = build(&ctx, &StreamOptions::default());
    assert_request_invariants(&req);

    assert_eq!(
        req["messages"][0]["content"][1],
        json!({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}
        })
    );
}

/// `max_tokens` is per model family and must match what the genai backend
/// would have sent, so switching backends never moves the ceiling.
#[test]
fn golden_max_tokens_tracks_the_model_family() {
    for (id, expected) in [
        ("anthropic/claude-sonnet-4-5", 64000u64),
        ("anthropic/claude-haiku-4-5", 64000),
        ("anthropic/claude-opus-4-1", 32000),
        ("anthropic/claude-3-5-sonnet", 8192),
        ("anthropic/claude-3-opus", 4096),
    ] {
        let m = Model {
            id: id.into(),
            name: id.into(),
            api: "anthropic-messages".into(),
            provider: "anthropic".into(),
            ..Default::default()
        };
        let ctx = LlmContext {
            system_prompt: String::new(),
            messages: vec![user("hi")],
            tools: Vec::new(),
        };
        let req = build_request(&m, &ctx, &StreamOptions::default()).expect("supported");
        assert_eq!(req["max_tokens"], json!(expected), "model {id}");
    }
}

/// A corrupt tool call (arguments that never resolved to an object) and its
/// paired result are dropped together — the provider rejects a `tool_use`
/// whose `input` is not an object, and an orphaned `tool_result` is equally
/// invalid.
#[test]
fn golden_corrupt_tool_call_and_result_are_dropped_together() {
    let ctx = LlmContext {
        system_prompt: String::new(),
        messages: vec![
            user("go"),
            assistant(vec![AssistantContent::ToolCall(ToolCall {
                id: "toolu_bad".into(),
                name: "write".into(),
                arguments: json!(r#"{"path":"/x","text":"trunc"#),
            })]),
            tool_result("toolu_bad", "ok", false),
        ],
        tools: Vec::new(),
    };
    let req = build(&ctx, &StreamOptions::default());
    assert_request_invariants(&req);

    assert_eq!(
        req["messages"],
        json!([{"role": "user", "content": [{"type": "text", "text": "go"}]}]),
        "neither the corrupt call nor its result may reach the provider"
    );
}
