//! WP21 — end-to-end proof that a tool call whose streamed arguments are
//! malformed only in their *string literals* now survives the seam, instead
//! of being silently dropped from the conversation.
//!
//! Chain exercised: `genai::chat::ChatStreamEvent` (the seam) → inbound state
//! machine → terminal `AssistantMessage` → `to_chat_request` (the outbound
//! corrupt-args guard) → the `genai::chat::ChatRequest` that would go back to
//! the provider on the next turn.
//!
//! Before WP21 the inbound side parsed accumulated tool arguments with a bare
//! `serde_json::from_str` and stored the raw text as `Value::String` on
//! failure, which is exactly the shape
//! `mapping::outbound::collect_corrupt_tool_call_ids` treats as corrupt — so
//! one unescaped TAB or one stray `\H` anywhere in a tool's arguments cost the
//! user the entire tool call (and its paired `tool_result`), with no error
//! surfaced anywhere. See `src/mapping/json_repair.rs`.

use genai::chat::{ChatStreamEvent, StreamEnd, ToolCall as GenaiToolCall, ToolChunk};
use grain_agent_core::{
    AssistantContent, AssistantMessageEvent, LlmContext, Message, Model, TextContent,
    ToolResultMessage, UserContent, UserMessage,
};
use grain_llm_genai::{InboundState, to_chat_request};
use serde_json::json;

fn model() -> Model {
    Model {
        id: "anthropic/claude-haiku-4-5".into(),
        name: "Claude Haiku 4.5".into(),
        api: "anthropic".into(),
        provider: "anthropic".into(),
        ..Default::default()
    }
}

fn tool_result(tool_call_id: String) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id,
        tool_name: "edit".into(),
        content: vec![UserContent::Text(TextContent { text: "ok".into() })],
        details: serde_json::Value::Null,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 0,
    }
}

/// Feed one cumulative tool-call chunk plus the terminal, and return the
/// assistant message the loop would receive.
fn terminal_message_for(raw_arguments: &str) -> grain_agent_core::AssistantMessage {
    let mut state = InboundState::new(&model());
    state.on_event(ChatStreamEvent::Start);
    state.on_event(ChatStreamEvent::ToolCallChunk(ToolChunk {
        tool_call: GenaiToolCall {
            call_id: "toolu_test".into(),
            fn_name: "edit".into(),
            // genai delivers the accumulated buffer as a JSON string for both
            // the Anthropic and OpenAI streamers.
            fn_arguments: json!(raw_arguments),
            thought_signatures: None,
        },
    }));
    let events = state.on_event(ChatStreamEvent::End(StreamEnd::default()));
    match events.into_iter().last().expect("a terminal event") {
        AssistantMessageEvent::Done { result } => result,
        AssistantMessageEvent::Error { result, .. } => result,
        other => panic!("expected a terminal event, got {other:?}"),
    }
}

fn tool_call_of(msg: &grain_agent_core::AssistantMessage) -> &grain_agent_core::ToolCall {
    msg.content
        .iter()
        .find_map(|c| match c {
            AssistantContent::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .expect("the stream carried a tool call")
}

/// The exact malformation from upstream pi-ai's
/// `anthropic-sse-parsing.test.ts` "repairs malformed SSE JSON and malformed
/// streamed tool JSON" case (seam vector AV-1), inner layer: an invalid `\H`
/// escape and a raw TAB inside a string literal.
#[test]
fn upstream_malformed_tool_arguments_parse_into_a_usable_object() {
    let msg = terminal_message_for("{\"path\":\"A\\H\",\"text\":\"col1\tcol2\"}");
    let tc = tool_call_of(&msg);

    assert_eq!(
        tc.arguments,
        json!({"path": "A\\H", "text": "col1\tcol2"}),
        "arguments must be a real object with upstream's expected values, \
         not the raw text"
    );
    assert!(
        tc.arguments.is_object(),
        "a String-shaped value here is what the outbound guard drops"
    );
}

/// The user-visible consequence: the repaired call is replayed to the
/// provider on the next turn instead of vanishing from history.
#[test]
fn repaired_tool_call_survives_the_outbound_corrupt_args_guard() {
    let assistant = terminal_message_for("{\"path\":\"A\\H\",\"text\":\"col1\tcol2\"}");
    let call_id = tool_call_of(&assistant).id.clone();

    let ctx = LlmContext {
        system_prompt: String::new(),
        messages: vec![
            Message::User(UserMessage {
                content: vec![UserContent::Text(TextContent {
                    text: "edit the file".into(),
                })],
                timestamp: 0,
            }),
            Message::Assistant(assistant),
            Message::ToolResult(tool_result(call_id)),
        ],
        tools: Vec::new(),
    };

    let req = to_chat_request(&ctx);

    // Three messages in, three messages out: the assistant turn and its
    // paired tool_result both survive. Before the repair, the guard dropped
    // both and only the user message reached the provider.
    assert_eq!(
        req.messages.len(),
        3,
        "the assistant turn and its tool_result must both survive the \
         corrupt-args guard; got: {:?}",
        req.messages
    );
}

/// The conservative half of the contract: a genuinely **truncated** buffer is
/// still treated as corrupt. Upstream coerces these into a partial object via
/// `partial-json`; grain deliberately does not, because that would replay a
/// tool call with silently truncated arguments (e.g. a write whose content
/// lost its tail) rather than dropping it. Recorded as a deliberate
/// divergence in `tests/SEAM-VECTORS.md` §5.
#[test]
fn truncated_tool_arguments_are_still_treated_as_corrupt() {
    let msg = terminal_message_for(r#"{"path":"/x","text":"abc"#);
    let tc = tool_call_of(&msg);

    assert!(
        tc.arguments.is_string(),
        "a truncated buffer must stay String-shaped so the outbound guard \
         still drops it, got: {:?}",
        tc.arguments
    );

    let call_id = tc.id.clone();
    let ctx = LlmContext {
        system_prompt: String::new(),
        messages: vec![
            Message::Assistant(msg.clone()),
            Message::ToolResult(tool_result(call_id)),
        ],
        tools: Vec::new(),
    };
    assert_eq!(
        to_chat_request(&ctx).messages.len(),
        0,
        "the truncated call and its tool_result are both dropped"
    );
}

/// Well-formed arguments are untouched — the repair must not perturb the
/// overwhelmingly common path.
#[test]
fn well_formed_tool_arguments_are_unchanged() {
    let msg = terminal_message_for(r#"{"path":"src/main.rs","line":42}"#);
    assert_eq!(
        tool_call_of(&msg).arguments,
        json!({"path": "src/main.rs", "line": 42})
    );
}
