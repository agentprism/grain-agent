//! S-2 / S-6 / ledger 13 — end-to-end proof that the response metadata
//! Anthropic puts on the wire actually reaches the grain `AssistantMessage`.
//!
//! These three fields were "structural gaps" for as long as
//! `grain_agent_core::AssistantMessage` had nowhere to put them. WP19 added
//! the slots; for `response_id` and `response_model` this transport is the
//! one place in the workspace where the values genuinely exist, because it
//! parses the Anthropic event stream itself instead of going through genai
//! (whose Anthropic streamer hard-codes `captured_response_id: None` and
//! never reads `chunk.model`). `raw_stop_reason` is populated here too —
//! and, since WP32, on the default genai path as well (the raw string
//! crosses genai inside every `StopReason` variant; see
//! `mapping::inbound` and `tests/inbound.rs`).
//!
//! The unit tests in `src/anthropic/state.rs` drive the state machine
//! directly. These drive the WHOLE transport over a real socket —
//! `AnthropicStream::stream` → HTTP → SSE decode → state machine → terminal
//! event — so what is asserted is what a caller would actually observe, not
//! what an internal helper returns.

use futures::StreamExt;
use grain_agent_core::{
    AssistantMessage, AssistantMessageEvent, LlmContext, LlmStream, Message, Model, StreamOptions,
    TextContent, UserContent, UserMessage,
};
use grain_llm_genai::{AnthropicStream, AnthropicTransportConfig};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

/// Serve exactly one request with `body`, and return the base URL.
async fn serve(body: String) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");

        // Drain the request head (and body, if any) before replying.
        let mut buf: Vec<u8> = Vec::new();
        let mut scratch = [0u8; 4096];
        let head_end = loop {
            let n = socket.read(&mut scratch).await.expect("read head");
            if n == 0 {
                return;
            }
            buf.extend_from_slice(&scratch[..n]);
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break pos + 4;
            }
        };
        let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
        let content_length: usize = head
            .lines()
            .find_map(|l| {
                let (k, v) = l.split_once(':')?;
                k.eq_ignore_ascii_case("content-length")
                    .then(|| v.trim().parse().ok())?
            })
            .unwrap_or(0);
        while buf.len() < head_end + content_length {
            let n = socket.read(&mut scratch).await.expect("read body");
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&scratch[..n]);
        }

        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");
        socket.flush().await.expect("flush");
        socket.shutdown().await.ok();
    });

    format!("http://{addr}/")
}

/// One SSE frame is `event: <name>\ndata: <json>\n`, and frames are separated
/// by a blank line — hence the `\n` join on top of each frame's trailing `\n`.
fn sse(events: &[(&str, String)]) -> String {
    events
        .iter()
        .map(|(e, d)| format!("event: {e}\ndata: {d}\n"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// `model.id` is namespaced; the transport strips it to `claude-haiku-4-5`
/// before putting it on the wire. That gap is exactly what the
/// `response_model` comparison has to get right.
fn model() -> Model {
    Model {
        id: "anthropic/claude-haiku-4-5".into(),
        name: "Claude Haiku 4.5".into(),
        api: "anthropic-messages".into(),
        provider: "anthropic".into(),
        ..Default::default()
    }
}

fn ctx() -> LlmContext {
    LlmContext {
        system_prompt: String::new(),
        messages: vec![Message::User(UserMessage {
            content: vec![UserContent::Text(TextContent { text: "hi".into() })],
            timestamp: 0,
        })],
        tools: Vec::new(),
    }
}

/// Drive the real transport against `body` and return the terminal message.
async fn terminal_message(body: String) -> AssistantMessage {
    let base_url = serve(body).await;
    let transport = AnthropicStream::new(
        AnthropicTransportConfig::with_api_key("test-key").with_base_url(base_url),
    );

    let mut stream = transport
        .stream(
            &model(),
            &ctx(),
            &StreamOptions::default(),
            CancellationToken::new(),
        )
        .await
        .expect("stream opens");

    let mut last = None;
    while let Some(ev) = stream.next().await {
        let terminal = ev.is_terminal();
        if terminal {
            last = Some(ev);
            break;
        }
    }
    match last.expect("the stream must reach a terminal event") {
        AssistantMessageEvent::Done { result } => result,
        AssistantMessageEvent::Error { result, .. } => result,
        other => panic!("unexpected terminal {other:?}"),
    }
}

/// A body whose `message_start` carries `id` and `model`, and whose
/// `message_delta` carries a raw `stop_reason`.
fn body_with(served_model: &str, stop_reason: &str) -> String {
    sse(&[
        (
            "message_start",
            json!({"type":"message_start","message":{
                "id":"msg_01ABCDEF","model":served_model,
                "usage":{"input_tokens":11,"output_tokens":0}}})
            .to_string(),
        ),
        (
            "content_block_start",
            json!({"type":"content_block_start","index":0,
                   "content_block":{"type":"text","text":""}})
            .to_string(),
        ),
        (
            "content_block_delta",
            json!({"type":"content_block_delta","index":0,
                   "delta":{"type":"text_delta","text":"ok"}})
            .to_string(),
        ),
        (
            "content_block_stop",
            json!({"type":"content_block_stop","index":0}).to_string(),
        ),
        (
            "message_delta",
            json!({"type":"message_delta","delta":{"stop_reason":stop_reason},
                   "usage":{"output_tokens":3}})
            .to_string(),
        ),
        (
            "message_stop",
            json!({"type":"message_stop"}).to_string(),
        ),
    ])
}

/// S-2 + S-6 + ledger 13 together: all three values the wire carried arrive
/// on the message a caller receives.
#[tokio::test]
async fn wire_metadata_reaches_the_caller() {
    // The provider resolved our alias to a dated snapshot — a real routing
    // event, and the case `response_model` exists to report.
    let msg = terminal_message(body_with("claude-haiku-4-5-20251001", "end_turn")).await;

    assert_eq!(
        msg.response_id.as_deref(),
        Some("msg_01ABCDEF"),
        "S-2: the provider's message id must reach the assistant message"
    );
    assert_eq!(
        msg.response_model.as_deref(),
        Some("claude-haiku-4-5-20251001"),
        "S-6: the concretely-served model must be reported when it differs"
    );
    assert_eq!(
        msg.raw_stop_reason.as_deref(),
        Some("end_turn"),
        "ledger 13: the provider's verbatim stop string must be preserved"
    );

    // The requested model is NOT overwritten by what was served.
    assert_eq!(msg.model, "anthropic/claude-haiku-4-5");
    // And the normalized stop reason still works.
    assert_eq!(msg.stop_reason, grain_agent_core::StopReason::Stop);
}

/// Upstream's rule is "only when it differs". This transport strips the
/// `anthropic/` namespace before sending, so the echo `claude-haiku-4-5` is a
/// MATCH against what went on the wire even though it differs from the
/// namespaced `model.id`. Getting this wrong would populate `response_model`
/// on every single response and turn a routing signal into constant noise.
#[tokio::test]
async fn served_model_equal_to_the_requested_one_is_not_reported() {
    let msg = terminal_message(body_with("claude-haiku-4-5", "end_turn")).await;

    assert_eq!(
        msg.response_model, None,
        "a served model matching the request is not a routing event"
    );
    // The other two still arrive — this is a not-set case for ONE field.
    assert_eq!(msg.response_id.as_deref(), Some("msg_01ABCDEF"));
    assert_eq!(msg.raw_stop_reason.as_deref(), Some("end_turn"));
}

/// The raw stop string earns its slot precisely where normalization is lossy.
/// Anthropic's `end_turn`, `stop_sequence` and `pause_turn` ALL collapse onto
/// `StopReason::Stop` with no error message, so after normalization they are
/// indistinguishable — yet they mean different things, and `pause_turn` most
/// of all: it signals a turn Anthropic expects to be CONTINUED, which a
/// caller reading only a plain `Stop` would wrongly treat as finished.
#[tokio::test]
async fn raw_stop_strings_that_normalize_identically_stay_distinguishable() {
    let end_turn = terminal_message(body_with("claude-haiku-4-5", "end_turn")).await;
    let stop_sequence = terminal_message(body_with("claude-haiku-4-5", "stop_sequence")).await;
    let pause_turn = terminal_message(body_with("claude-haiku-4-5", "pause_turn")).await;

    // Precondition: normalization really does erase the difference.
    assert_eq!(end_turn.stop_reason, stop_sequence.stop_reason);
    assert_eq!(end_turn.stop_reason, pause_turn.stop_reason);
    assert_eq!(end_turn.error_message, None);
    assert_eq!(pause_turn.error_message, None);

    // The raw channel is what still tells them apart.
    assert_eq!(end_turn.raw_stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(stop_sequence.raw_stop_reason.as_deref(), Some("stop_sequence"));
    assert_eq!(pause_turn.raw_stop_reason.as_deref(), Some("pause_turn"));
}

/// A stream that never sends `message_delta` has no provider stop string —
/// and must report `None` rather than inventing one.
#[tokio::test]
async fn absent_stop_reason_is_none_not_a_guess() {
    let body = sse(&[
        (
            "message_start",
            json!({"type":"message_start","message":{"id":"msg_1",
                   "usage":{"input_tokens":1}}})
            .to_string(),
        ),
        (
            "message_stop",
            json!({"type":"message_stop"}).to_string(),
        ),
    ]);
    let msg = terminal_message(body).await;

    assert_eq!(msg.raw_stop_reason, None);
    // A `message_start` with no `model` key likewise reports nothing.
    assert_eq!(msg.response_model, None);
    assert_eq!(msg.response_id.as_deref(), Some("msg_1"));
}
