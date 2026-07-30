//! Unit tests for the inbound state machine.
//!
//! Constructs hand-crafted `ChatStreamEvent` sequences and asserts the grain
//! `AssistantMessageEvent` output. No I/O.

use genai::chat::{ChatStreamEvent, StreamChunk, StreamEnd, ToolCall as GenaiToolCall, ToolChunk};
use grain_agent_core::{AssistantContent, AssistantMessageEvent, Model, StopReason};
use grain_llm_genai::InboundState;

fn model() -> Model {
    Model {
        id: "anthropic/claude-sonnet-4-5".into(),
        name: "Claude Sonnet 4.5".into(),
        api: "anthropic".into(),
        provider: "anthropic".into(),
        ..Default::default()
    }
}

fn chunk(s: &str) -> ChatStreamEvent {
    ChatStreamEvent::Chunk(StreamChunk { content: s.into() })
}

fn reasoning(s: &str) -> ChatStreamEvent {
    ChatStreamEvent::ReasoningChunk(StreamChunk { content: s.into() })
}

fn thought_sig(s: &str) -> ChatStreamEvent {
    ChatStreamEvent::ThoughtSignatureChunk(StreamChunk { content: s.into() })
}

fn tool_call(id: &str, name: &str, args: serde_json::Value) -> ChatStreamEvent {
    ChatStreamEvent::ToolCallChunk(ToolChunk {
        tool_call: GenaiToolCall {
            call_id: id.into(),
            fn_name: name.into(),
            fn_arguments: args,
            thought_signatures: None,
        },
    })
}

/// A cumulative-arguments chunk as the OpenAI / Anthropic genai adapters
/// deliver it: `fn_arguments` is a `Value::String` holding the latest
/// accumulated (possibly partial) JSON.
fn tool_call_cumulative(id: &str, name: &str, accumulated: &str) -> ChatStreamEvent {
    tool_call(id, name, serde_json::json!(accumulated))
}

fn delta_of(e: &AssistantMessageEvent) -> Option<&str> {
    match e {
        AssistantMessageEvent::TextDelta { delta, .. }
        | AssistantMessageEvent::ThinkingDelta { delta, .. }
        | AssistantMessageEvent::ToolcallDelta { delta, .. } => Some(delta),
        _ => None,
    }
}

fn end_normal() -> ChatStreamEvent {
    ChatStreamEvent::End(StreamEnd::default())
}

fn run(
    events: impl IntoIterator<Item = ChatStreamEvent>,
) -> (Vec<AssistantMessageEvent>, InboundState) {
    let mut state = InboundState::new(&model());
    let mut all = Vec::new();
    for ev in events {
        all.extend(state.on_event(ev));
    }
    (all, state)
}

fn tag(e: &AssistantMessageEvent) -> &'static str {
    match e {
        AssistantMessageEvent::Start { .. } => "Start",
        AssistantMessageEvent::TextStart { .. } => "TextStart",
        AssistantMessageEvent::TextDelta { .. } => "TextDelta",
        AssistantMessageEvent::TextEnd { .. } => "TextEnd",
        AssistantMessageEvent::ThinkingStart { .. } => "ThinkingStart",
        AssistantMessageEvent::ThinkingDelta { .. } => "ThinkingDelta",
        AssistantMessageEvent::ThinkingEnd { .. } => "ThinkingEnd",
        AssistantMessageEvent::ToolcallStart { .. } => "ToolcallStart",
        AssistantMessageEvent::ToolcallDelta { .. } => "ToolcallDelta",
        AssistantMessageEvent::ToolcallEnd { .. } => "ToolcallEnd",
        AssistantMessageEvent::Done { .. } => "Done",
        AssistantMessageEvent::Error { .. } => "Error",
    }
}

#[test]
fn start_then_end_emits_start_and_done() {
    let (events, _) = run([ChatStreamEvent::Start, end_normal()]);
    let tags: Vec<_> = events.iter().map(tag).collect();
    assert_eq!(tags, vec!["Start", "Done"]);
}

#[test]
fn text_chunks_aggregate_into_one_block() {
    let (events, _) = run([
        ChatStreamEvent::Start,
        chunk("hello "),
        chunk("world"),
        end_normal(),
    ]);
    let tags: Vec<_> = events.iter().map(tag).collect();
    assert_eq!(
        tags,
        vec![
            "Start",
            "TextStart",
            "TextDelta",
            "TextDelta",
            "TextEnd",
            "Done"
        ]
    );
    let done = events.last().unwrap();
    if let AssistantMessageEvent::Done { result } = done {
        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            AssistantContent::Text(t) => assert_eq!(t.text, "hello world"),
            _ => panic!("expected text block"),
        }
        assert_eq!(result.stop_reason, StopReason::Stop);
    } else {
        panic!("expected Done");
    }
}

#[test]
fn reasoning_then_text_splits_into_two_blocks() {
    let (events, _) = run([
        ChatStreamEvent::Start,
        reasoning("hmm "),
        reasoning("let me think"),
        chunk("the answer is 42"),
        end_normal(),
    ]);
    let tags: Vec<_> = events.iter().map(tag).collect();
    assert_eq!(
        tags,
        vec![
            "Start",
            "ThinkingStart",
            "ThinkingDelta",
            "ThinkingDelta",
            "ThinkingEnd",
            "TextStart",
            "TextDelta",
            "TextEnd",
            "Done"
        ]
    );
    if let AssistantMessageEvent::Done { result } = events.last().unwrap() {
        assert_eq!(result.content.len(), 2);
        assert!(matches!(result.content[0], AssistantContent::Thinking(_)));
        assert!(matches!(result.content[1], AssistantContent::Text(_)));
    } else {
        panic!();
    }
}

#[test]
fn thought_signature_absorbs_silently_into_thinking_block() {
    let (events, _) = run([
        ChatStreamEvent::Start,
        reasoning("reasoning text"),
        thought_sig("sig-part-1"),
        thought_sig("sig-part-2"),
        end_normal(),
    ]);
    // No grain event for the signature itself.
    let tags: Vec<_> = events.iter().map(tag).collect();
    assert_eq!(
        tags,
        vec![
            "Start",
            "ThinkingStart",
            "ThinkingDelta",
            "ThinkingEnd",
            "Done"
        ]
    );
    if let AssistantMessageEvent::Done { result } = events.last().unwrap() {
        if let AssistantContent::Thinking(t) = &result.content[0] {
            assert_eq!(t.thinking, "reasoning text");
            assert_eq!(t.signature.as_deref(), Some("sig-part-1sig-part-2"));
        } else {
            panic!("expected thinking block")
        }
    } else {
        panic!();
    }
}

#[test]
fn tool_call_closes_open_text_and_emits_start_delta_end() {
    // Gemini-style complete call in one chunk: Start, the full argument
    // JSON as one delta, then End when the stream terminates (mirrors
    // upstream pi-ai google-generative-ai.ts: start → delta(full JSON) →
    // end).
    let (events, _) = run([
        ChatStreamEvent::Start,
        chunk("calling tool now"),
        tool_call("call-1", "echo", serde_json::json!({ "v": 1 })),
        end_normal(),
    ]);
    let tags: Vec<_> = events.iter().map(tag).collect();
    assert_eq!(
        tags,
        vec![
            "Start",
            "TextStart",
            "TextDelta",
            "TextEnd",
            "ToolcallStart",
            "ToolcallDelta",
            "ToolcallEnd",
            "Done"
        ]
    );
    let deltas: Vec<_> = events.iter().filter_map(delta_of).collect();
    assert_eq!(deltas, vec!["calling tool now", r#"{"v":1}"#]);
    if let AssistantMessageEvent::Done { result } = events.last().unwrap() {
        assert_eq!(result.content.len(), 2);
        match &result.content[1] {
            AssistantContent::ToolCall(tc) => {
                assert_eq!(tc.id, "call-1");
                assert_eq!(tc.arguments, serde_json::json!({ "v": 1 }));
            }
            other => panic!("expected tool call, got {other:?}"),
        }
        assert_eq!(
            result.stop_reason,
            StopReason::ToolUse,
            "inferred from tool call presence"
        );
    } else {
        panic!();
    }
}

#[test]
fn cumulative_tool_chunks_emit_suffix_deltas() {
    // OpenAI / Anthropic style: chunks for the same call_id repeat with
    // cumulatively growing argument strings. Expect Start on first
    // appearance, one delta per growth step carrying exactly the new
    // suffix, and End with the final assembled call.
    let (events, _) = run([
        ChatStreamEvent::Start,
        tool_call_cumulative("call-1", "get_weather", ""),
        tool_call_cumulative("call-1", "get_weather", r#"{"city":"#),
        tool_call_cumulative("call-1", "get_weather", r#"{"city":"Paris"}"#),
        end_normal(),
    ]);
    let tags: Vec<_> = events.iter().map(tag).collect();
    assert_eq!(
        tags,
        vec![
            "Start",
            "ToolcallStart",
            "ToolcallDelta",
            "ToolcallDelta",
            "ToolcallEnd",
            "Done"
        ]
    );
    let deltas: Vec<_> = events.iter().filter_map(delta_of).collect();
    assert_eq!(deltas, vec![r#"{"city":"#, r#""Paris"}"#]);
    if let AssistantMessageEvent::Done { result } = events.last().unwrap() {
        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            AssistantContent::ToolCall(tc) => {
                assert_eq!(tc.id, "call-1");
                assert_eq!(tc.name, "get_weather");
                assert_eq!(tc.arguments, serde_json::json!({ "city": "Paris" }));
            }
            other => panic!("expected tool call, got {other:?}"),
        }
        assert_eq!(result.stop_reason, StopReason::ToolUse);
    } else {
        panic!();
    }
}

#[test]
fn duplicate_identical_tool_chunks_are_deduped() {
    // The same complete chunk repeated (observed on providers that re-send
    // the assembled call on the finish chunk) must not produce duplicate
    // blocks or duplicate events.
    let args = serde_json::json!({ "v": 1 });
    let (events, _) = run([
        ChatStreamEvent::Start,
        tool_call("call-1", "echo", args.clone()),
        tool_call("call-1", "echo", args.clone()),
        end_normal(),
    ]);
    let tags: Vec<_> = events.iter().map(tag).collect();
    assert_eq!(
        tags,
        vec![
            "Start",
            "ToolcallStart",
            "ToolcallDelta",
            "ToolcallEnd",
            "Done"
        ]
    );
    if let AssistantMessageEvent::Done { result } = events.last().unwrap() {
        assert_eq!(result.content.len(), 1, "one block per unique call_id");
    } else {
        panic!();
    }
}

#[test]
fn parallel_tool_calls_close_previous_block() {
    // A second call_id appearing closes the first call's block (End) before
    // opening its own, so Start/End pairs never interleave.
    let (events, _) = run([
        ChatStreamEvent::Start,
        tool_call_cumulative("call-a", "alpha", r#"{"a":1}"#),
        tool_call_cumulative("call-b", "beta", r#"{"b":2}"#),
        end_normal(),
    ]);
    let tags: Vec<_> = events.iter().map(tag).collect();
    assert_eq!(
        tags,
        vec![
            "Start",
            "ToolcallStart",
            "ToolcallDelta",
            "ToolcallEnd",
            "ToolcallStart",
            "ToolcallDelta",
            "ToolcallEnd",
            "Done"
        ]
    );
    // Block indices: End(a) at 0, Start(b) at 1.
    let indices: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AssistantMessageEvent::ToolcallStart { content_index, .. }
            | AssistantMessageEvent::ToolcallEnd { content_index, .. } => Some(*content_index),
            _ => None,
        })
        .collect();
    assert_eq!(indices, vec![0, 0, 1, 1]);
    if let AssistantMessageEvent::Done { result } = events.last().unwrap() {
        assert_eq!(result.content.len(), 2);
    } else {
        panic!();
    }
}

#[test]
fn tool_chunk_after_block_closed_updates_silently() {
    // If arguments for an already-closed call keep growing (End was emitted
    // when another block opened), the block refreshes without further
    // events — subscribers saw well-formed Start/End; the loop executes
    // from the final message.
    let (events, _) = run([
        ChatStreamEvent::Start,
        tool_call_cumulative("call-1", "echo", r#"{"v":1"#),
        chunk("interleaved text"),
        tool_call_cumulative("call-1", "echo", r#"{"v":12}"#),
        end_normal(),
    ]);
    let tags: Vec<_> = events.iter().map(tag).collect();
    assert_eq!(
        tags,
        vec![
            "Start",
            "ToolcallStart",
            "ToolcallDelta",
            "ToolcallEnd",
            "TextStart",
            "TextDelta",
            "TextEnd",
            "Done"
        ],
        "no Toolcall events after the block closed"
    );
    if let AssistantMessageEvent::Done { result } = events.last().unwrap() {
        match &result.content[0] {
            AssistantContent::ToolCall(tc) => {
                assert_eq!(
                    tc.arguments,
                    serde_json::json!({ "v": 12 }),
                    "final message carries the fully-accumulated args"
                );
            }
            other => panic!("expected tool call, got {other:?}"),
        }
    } else {
        panic!();
    }
}

#[test]
fn captured_usage_populates_final_message() {
    let usage = genai::chat::Usage {
        prompt_tokens: Some(100),
        completion_tokens: Some(50),
        total_tokens: Some(150),
        ..Default::default()
    };

    let end = ChatStreamEvent::End(StreamEnd {
        captured_usage: Some(usage),
        captured_content: None,
        captured_reasoning_content: None,
        captured_response_id: None,
        captured_stop_reason: None,
    });

    let (events, _) = run([ChatStreamEvent::Start, chunk("hi"), end]);
    if let AssistantMessageEvent::Done { result } = events.last().unwrap() {
        assert_eq!(result.usage.input, 100);
        assert_eq!(result.usage.output, 50);
        assert_eq!(result.usage.total_tokens, 150);
    } else {
        panic!();
    }
}

#[test]
fn into_aborted_synthesizes_terminal_error() {
    let mut state = InboundState::new(&model());
    let _ = state.on_event(ChatStreamEvent::Start);
    let _ = state.on_event(chunk("partial"));
    let term = state.into_aborted();
    match term {
        AssistantMessageEvent::Error { error, result } => {
            assert_eq!(error, "aborted");
            assert_eq!(result.stop_reason, StopReason::Aborted);
            assert_eq!(result.error_message.as_deref(), Some("aborted"));
            // Partial text survives.
            assert_eq!(result.content.len(), 1);
        }
        other => panic!("expected Error, got {other:?}"),
    }
}

#[test]
fn into_error_msg_preserves_accumulated_content() {
    let mut state = InboundState::new(&model());
    state.on_event(ChatStreamEvent::Start);
    state.on_event(chunk("preamble"));
    state.on_event(reasoning("mid"));

    let term = state.into_error_msg("upstream 500");
    if let AssistantMessageEvent::Error { error, result } = term {
        assert_eq!(error, "upstream 500");
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.error_message.as_deref(), Some("upstream 500"));
        assert_eq!(result.content.len(), 2);
    } else {
        panic!();
    }
}

#[test]
fn duplicate_start_event_is_idempotent() {
    let (events, _) = run([ChatStreamEvent::Start, ChatStreamEvent::Start, end_normal()]);
    let starts = events
        .iter()
        .filter(|e| matches!(e, AssistantMessageEvent::Start { .. }))
        .count();
    assert_eq!(starts, 1, "subsequent Start events are absorbed");
}

#[test]
fn chunk_before_explicit_start_synthesizes_start() {
    // Defensive: if a provider emits content before an explicit Start, the
    // state machine still surfaces a single Start before the first Delta.
    let (events, _) = run([chunk("immediate"), end_normal()]);
    let tags: Vec<_> = events.iter().map(tag).collect();
    assert_eq!(
        tags,
        vec!["Start", "TextStart", "TextDelta", "TextEnd", "Done"]
    );
}
