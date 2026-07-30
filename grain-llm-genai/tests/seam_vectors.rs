//! WP5 — seam conformance vectors.
//!
//! Measures whether grain-llm-genai's event stream is faithful to upstream
//! pi-ai (`/home/vikash/pi/packages/ai`, pinned commit 34239180), using
//! upstream's own recorded stream-parsing fixtures as the standard.
//!
//! Chain under test (the full production path, no shortcuts):
//!
//! ```text
//! recorded provider SSE (upstream fixture, frame-faithful)
//!   → local mock HTTP endpoint
//!   → genai 0.6.5 real client + provider streamer
//!   → genai ChatStreamEvents
//!   → grain-llm-genai inbound adapter (InboundState via GenaiStream)
//!   → grain AssistantMessageEvents
//! ```
//!
//! Every vector is classified in `tests/SEAM-VECTORS.md`:
//! - **PASS** vectors run in the default suite and must stay green.
//! - **STRUCTURAL** vectors are `#[ignore]`d with the precise reason; they
//!   assert the *upstream-translated* expectation and are intentionally RED
//!   when run with `--ignored` — that red is the measurement. Do not "fix"
//!   them by weakening the expectation.
//!
//! Comparison scope (documented in SEAM-VECTORS.md §3): event kind order,
//! `content_index`, `delta` payloads, and the full terminal message
//! (content blocks, stop reason, error message, usage). Timestamps,
//! mid-stream `partial` usage (structural gap S-1: genai only surfaces
//! usage at `End`), and fields with no grain slot (`rawStopReason`,
//! `responseId`) are excluded and tracked as named gaps instead.

use futures::StreamExt;
use genai::ServiceTarget;
use genai::resolver::{AuthData, Endpoint};
use grain_agent_core::{
    AssistantContent, AssistantMessageEvent, LlmContext, LlmStream, Message, Model, StopReason,
    StreamOptions, TextContent, ToolCall, ToolDefinition, Usage, UserContent, UserMessage,
};
use grain_llm_genai::{
    AnthropicStream, AnthropicTransportConfig, GenaiStream, baseline_chat_options,
};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Mock SSE endpoint
// ---------------------------------------------------------------------------

/// Serve exactly one HTTP request with the given SSE body, then close.
///
/// Returns the base URL to point genai at and a handle resolving to the
/// request line (`"POST /messages HTTP/1.1"` …) so vectors can assert the
/// real adapter URL construction was exercised.
async fn serve_sse_once(body: String) -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock SSE listener");
    let addr = listener.local_addr().expect("mock listener addr");

    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept mock connection");

        // Read the full request head, then drain the content-length body.
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 4096];
        let head_end = loop {
            let n = socket.read(&mut chunk).await.expect("read request");
            if n == 0 {
                panic!("client closed before sending a full request head");
            }
            buf.extend_from_slice(&chunk[..n]);
            if let Some(pos) = find_subsequence(&buf, b"\r\n\r\n") {
                break pos + 4;
            }
        };
        let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
        let request_line = head.lines().next().unwrap_or_default().to_string();

        let content_length: usize = head
            .lines()
            .find_map(|l| {
                let (k, v) = l.split_once(':')?;
                k.eq_ignore_ascii_case("content-length")
                    .then(|| v.trim().parse().ok())?
            })
            .unwrap_or(0);
        while buf.len() < head_end + content_length {
            let n = socket.read(&mut chunk).await.expect("read request body");
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
        }

        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write mock response");
        socket.flush().await.expect("flush mock response");
        socket.shutdown().await.ok();

        request_line
    });

    (format!("http://{addr}/"), handle)
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

// ---------------------------------------------------------------------------
// Harness: drive the full chain
// ---------------------------------------------------------------------------

/// Run one vector: serve `sse_body`, point genai's real client at it (only
/// the endpoint/auth are overridden — adapter kind, URL construction,
/// request building, and stream parsing are all production code), stream
/// through `GenaiStream`, and collect the grain events plus the request
/// line the adapter actually issued.
async fn run_vector(
    model: &Model,
    ctx: &LlmContext,
    sse_body: String,
) -> (Vec<AssistantMessageEvent>, String) {
    let (base_url, request_handle) = serve_sse_once(sse_body).await;

    let target_resolver =
        move |mut target: ServiceTarget| -> genai::resolver::Result<ServiceTarget> {
            target.endpoint = Endpoint::from_owned(base_url.clone());
            target.auth = AuthData::from_single("seam-test-key");
            Ok(target)
        };
    let client = genai::Client::builder()
        .with_auth_resolver_fn(|_| Ok(Some(AuthData::from_single("seam-test-key"))))
        .with_service_target_resolver_fn(target_resolver)
        .build();
    let stream_impl = GenaiStream::with_client_and_options(client, baseline_chat_options());

    let mut stream = stream_impl
        .stream(
            model,
            ctx,
            &StreamOptions::default(),
            CancellationToken::new(),
        )
        .await
        .expect("GenaiStream::stream never errors for runtime failures");

    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        let terminal = ev.is_terminal();
        events.push(ev);
        if terminal {
            break;
        }
    }

    let request_line = request_handle.await.expect("mock server task");
    (events, request_line)
}

/// Run one Anthropic vector through the **native Anthropic transport**
/// (`grain_llm_genai::anthropic`) instead of genai.
///
/// Same recorded upstream fixture, same local socket, same assertions — only
/// the backend differs. The genai backend's behavior on these very fixtures is
/// pinned separately in `tests/genai_seam_limits.rs`, which asserts the
/// defects directly at genai's seam (the S-3 double count, the dropped
/// `stop_details`, the vanishing empty block), so switching these vectors to
/// the native path measures the better backend without losing the measurement
/// of the default one.
async fn run_anthropic_vector(
    model: &Model,
    ctx: &LlmContext,
    sse_body: String,
) -> (Vec<AssistantMessageEvent>, String) {
    let (base_url, request_handle) = serve_sse_once(sse_body).await;

    let stream_impl = AnthropicStream::new(
        AnthropicTransportConfig::with_api_key("seam-test-key").with_base_url(base_url),
    );

    let mut stream = stream_impl
        .stream(
            model,
            ctx,
            &StreamOptions::default(),
            CancellationToken::new(),
        )
        .await
        .expect("AnthropicStream::stream never errors for runtime failures");

    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        let terminal = ev.is_terminal();
        events.push(ev);
        if terminal {
            break;
        }
    }

    let request_line = request_handle.await.expect("mock server task");
    (events, request_line)
}

// ---------------------------------------------------------------------------
// Expected-event DSL + comparison
// ---------------------------------------------------------------------------

/// Terminal payload translated from upstream's expected `AssistantMessage`.
///
/// `model` / `api` / `provider` echoes and `timestamp` are excluded (grain
/// configuration echoes, nondeterministic clock). `usage.cost` is compared
/// implicitly via `Usage: PartialEq` — zero on both sides for these vectors
/// (upstream fixtures use zero-cost models).
#[derive(Debug)]
struct Terminal {
    content: Vec<AssistantContent>,
    stop: StopReason,
    error_message: Option<String>,
    usage: Usage,
}

#[derive(Debug)]
enum Expect {
    Start,
    TextStart(usize),
    TextDelta(usize, &'static str),
    TextEnd(usize),
    ToolcallStart(usize),
    ToolcallDelta(usize, String),
    ToolcallEnd(usize),
    Done(Terminal),
    /// Terminal error event; the event's `error` string must equal the
    /// terminal's `error_message` (upstream sets both from the same value).
    Error(Terminal),
}

fn event_tag(e: &AssistantMessageEvent) -> &'static str {
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

fn describe_actual(events: &[AssistantMessageEvent]) -> String {
    events
        .iter()
        .map(|e| match e {
            AssistantMessageEvent::TextDelta {
                content_index,
                delta,
                ..
            }
            | AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
                ..
            }
            | AssistantMessageEvent::ToolcallDelta {
                content_index,
                delta,
                ..
            } => format!("{}(#{content_index}, {delta:?})", event_tag(e)),
            AssistantMessageEvent::Done { result } => format!(
                "Done(stop={:?}, content={:?}, usage={:?})",
                result.stop_reason, result.content, result.usage
            ),
            AssistantMessageEvent::Error { error, result } => format!(
                "Error({error:?}, stop={:?}, content={:?}, usage={:?})",
                result.stop_reason, result.content, result.usage
            ),
            other => event_tag(other).to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n  ")
}

fn assert_terminal(
    vector: &str,
    position: usize,
    expected: &Terminal,
    actual_stop: StopReason,
    actual_content: &[AssistantContent],
    actual_error: Option<&str>,
    actual_usage: &Usage,
) {
    assert_eq!(
        actual_stop, expected.stop,
        "[{vector}] terminal event #{position}: stop_reason mismatch"
    );
    assert_eq!(
        actual_error,
        expected.error_message.as_deref(),
        "[{vector}] terminal event #{position}: error_message mismatch"
    );
    assert_eq!(
        actual_content, expected.content,
        "[{vector}] terminal event #{position}: content mismatch"
    );
    assert_eq!(
        actual_usage, &expected.usage,
        "[{vector}] terminal event #{position}: usage mismatch"
    );
}

fn assert_events(vector: &str, actual: &[AssistantMessageEvent], expected: &[Expect]) {
    let dump = describe_actual(actual);
    assert_eq!(
        actual.len(),
        expected.len(),
        "[{vector}] event count mismatch: expected {} events, got {}.\nActual:\n  {dump}",
        expected.len(),
        actual.len()
    );

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        match (a, e) {
            (AssistantMessageEvent::Start { .. }, Expect::Start) => {}
            (AssistantMessageEvent::TextStart { content_index, .. }, Expect::TextStart(idx))
            | (
                AssistantMessageEvent::ToolcallStart { content_index, .. },
                Expect::ToolcallStart(idx),
            )
            | (AssistantMessageEvent::TextEnd { content_index, .. }, Expect::TextEnd(idx))
            | (
                AssistantMessageEvent::ToolcallEnd { content_index, .. },
                Expect::ToolcallEnd(idx),
            ) => {
                assert_eq!(
                    content_index,
                    idx,
                    "[{vector}] event #{i} ({}): content_index mismatch.\nActual:\n  {dump}",
                    event_tag(a)
                );
            }
            (
                AssistantMessageEvent::TextDelta {
                    content_index,
                    delta,
                    ..
                },
                Expect::TextDelta(idx, expected_delta),
            ) => {
                assert_eq!(
                    content_index, idx,
                    "[{vector}] event #{i} (TextDelta): content_index mismatch"
                );
                assert_eq!(
                    delta, expected_delta,
                    "[{vector}] event #{i} (TextDelta): delta mismatch"
                );
            }
            (
                AssistantMessageEvent::ToolcallDelta {
                    content_index,
                    delta,
                    ..
                },
                Expect::ToolcallDelta(idx, expected_delta),
            ) => {
                assert_eq!(
                    content_index, idx,
                    "[{vector}] event #{i} (ToolcallDelta): content_index mismatch"
                );
                assert_eq!(
                    delta, expected_delta,
                    "[{vector}] event #{i} (ToolcallDelta): delta mismatch"
                );
            }
            (AssistantMessageEvent::Done { result }, Expect::Done(t)) => {
                assert_terminal(
                    vector,
                    i,
                    t,
                    result.stop_reason,
                    &result.content,
                    result.error_message.as_deref(),
                    &result.usage,
                );
            }
            (AssistantMessageEvent::Error { error, result }, Expect::Error(t)) => {
                assert_eq!(
                    Some(error.as_str()),
                    t.error_message.as_deref(),
                    "[{vector}] event #{i} (Error): error string mismatch"
                );
                assert_terminal(
                    vector,
                    i,
                    t,
                    result.stop_reason,
                    &result.content,
                    result.error_message.as_deref(),
                    &result.usage,
                );
            }
            _ => panic!(
                "[{vector}] event #{i}: expected {e:?}, got {} .\nActual:\n  {dump}",
                event_tag(a)
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Fixture + payload helpers
// ---------------------------------------------------------------------------

fn model_of(id: &str, api: &str, provider: &str) -> Model {
    Model {
        id: id.into(),
        name: id.into(),
        api: api.into(),
        provider: provider.into(),
        ..Default::default()
    }
}

fn user_ctx(text: &str) -> LlmContext {
    LlmContext {
        system_prompt: String::new(),
        messages: vec![Message::User(UserMessage {
            content: vec![UserContent::Text(TextContent { text: text.into() })],
            timestamp: 0,
        })],
        tools: Vec::new(),
    }
}

fn tool_def(name: &str, params: serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        label: name.into(),
        description: format!("{name} tool"),
        parameters: params,
        execution_mode: None,
    }
}

fn text_block(s: &str) -> AssistantContent {
    AssistantContent::Text(TextContent { text: s.into() })
}

fn tool_call_block(id: &str, name: &str, args: serde_json::Value) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: args,
        thought_signature: None,
    })
}

fn usage_of(input: u64, output: u64, total: u64) -> Usage {
    Usage {
        input,
        output,
        total_tokens: total,
        ..Usage::default()
    }
}

/// Frame-faithful port of upstream `createSseResponse`
/// (`test/anthropic-sse-parsing.test.ts`): `event: <e>\ndata: <d>\n` blocks
/// joined with `\n`, **no** trailing blank line — genai's SSE splitter must
/// flush the final `message_stop` on EOF, exactly like upstream's parser.
/// Frame payloads built with `serde_json::json!` carry alphabetized object
/// keys (serde_json without `preserve_order`), which is semantically
/// identical JSON; the one payload where byte order matters — AV-1's
/// malformed frame, whose brokenness IS the fixture — is a raw string and
/// byte-exact.
fn anthropic_sse(events: &[(&str, String)]) -> String {
    events
        .iter()
        .map(|(event, data)| format!("event: {event}\ndata: {data}\n"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn anthropic_message_start(id: &str, input_tokens: u64) -> (&'static str, String) {
    (
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": id,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            }
        })
        .to_string(),
    )
}

fn anthropic_message_stop() -> (&'static str, String) {
    ("message_stop", json!({"type": "message_stop"}).to_string())
}

/// Upstream `minimalAnthropicEvents` fixture, verbatim.
fn minimal_anthropic_events() -> Vec<(&'static str, String)> {
    vec![
        anthropic_message_start("msg_test", 12),
        (
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"}
            })
            .to_string(),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}).to_string(),
        ),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            })
            .to_string(),
        ),
        anthropic_message_stop(),
    ]
}

/// OpenAI chat-completions wire framing: upstream mocks the SDK with chunk
/// objects; on the wire each chunk is a `data: <json>` SSE frame and the
/// stream terminates with `data: [DONE]` (which the SDK consumes before the
/// mock layer). This reconstruction is the exact production wire format.
fn openai_sse(chunks: &[serde_json::Value]) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

/// Gemini `streamGenerateContent?alt=sse` framing: one `data: <json>` frame
/// per GenerateContentResponse chunk; no `[DONE]` sentinel (the finishReason
/// frame ends the stream).
fn gemini_sse(chunks: &[serde_json::Value]) -> String {
    chunks
        .iter()
        .map(|c| format!("data: {c}\n\n"))
        .collect::<String>()
}

// ---------------------------------------------------------------------------
// Anthropic vectors — upstream test/anthropic-sse-parsing.test.ts
// ---------------------------------------------------------------------------

/// AV-1 — upstream: "repairs malformed SSE JSON and malformed streamed tool
/// JSON".
///
/// Upstream repairs the invalid `\H` escape and the literal TAB inside the
/// streamed tool JSON (`parseStreamingJson` + lenient SSE parsing) and
/// finishes with a successful toolUse turn. genai's anthropic streamer
/// `serde_json::from_str`s the raw frame and aborts the stream with
/// `Error::StreamParse`, so the grain chain terminates with an `Error`
/// event instead of the repaired `Done`.
#[tokio::test]
async fn av1_anthropic_repairs_malformed_sse_and_tool_json() {
    let malformed_tool_json_delta = "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"A\\H\\\",\\\"text\\\":\\\"col1\tcol2\\\"}\"}}".to_string();

    let sse = anthropic_sse(&[
        anthropic_message_start("msg_test", 12),
        (
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_test", "name": "edit", "input": {}}
            })
            .to_string(),
        ),
        ("content_block_delta", malformed_tool_json_delta),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}).to_string(),
        ),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            })
            .to_string(),
        ),
        anthropic_message_stop(),
    ]);

    let mut ctx = user_ctx("Use the edit tool.");
    ctx.tools = vec![tool_def(
        "edit",
        json!({
            "type": "object",
            "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
            "required": ["path", "text"]
        }),
    )];
    let model = model_of(
        "anthropic/claude-haiku-4-5",
        "anthropic-messages",
        "anthropic",
    );
    let (events, request_line) = run_anthropic_vector(&model, &ctx, sse).await;
    assert!(
        request_line.contains("/messages"),
        "expected the real anthropic adapter URL, got {request_line:?}"
    );

    // Upstream expectation, mechanically translated: the malformed frame is
    // repaired, the tool call completes, and the turn ends in toolUse with
    // arguments {"path": "A\H", "text": "col1<TAB>col2"} and usage
    // replaced per-field by message_delta (input 12, output 5, total 17).
    assert_events(
        "AV-1",
        &events,
        &[
            Expect::Start,
            Expect::ToolcallStart(0),
            Expect::ToolcallDelta(0, "{\"path\":\"A\\H\",\"text\":\"col1\tcol2\"}".to_string()),
            Expect::ToolcallEnd(0),
            Expect::Done(Terminal {
                content: vec![tool_call_block(
                    "toolu_test",
                    "edit",
                    json!({"path": "A\\H", "text": "col1\tcol2"}),
                )],
                stop: StopReason::ToolUse,
                error_message: None,
                usage: usage_of(12, 5, 17),
            }),
        ],
    );
}

/// AV-2 — upstream: "preserves refusal stop details from message_delta".
const REFUSAL_EXPLANATION: &str = "This request triggered restrictions on violative cyber content and was blocked under Anthropic's Usage Policy. To learn more, provide feedback, or request an exemption based on how you use Claude, visit our help center: https://support.claude.com/en/articles/14604842-real-time-cyber-safeguards-on-claude.";

#[tokio::test]
async fn av2_anthropic_preserves_refusal_stop_details() {
    let sse = anthropic_sse(&[
        anthropic_message_start("msg_01XFUDYJgAACzvnptvVoYEL", 412),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": "refusal",
                    "stop_details": {
                        "type": "refusal",
                        "category": "cyber",
                        "explanation": REFUSAL_EXPLANATION,
                    }
                },
                "usage": {
                    "input_tokens": 412,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            })
            .to_string(),
        ),
        anthropic_message_stop(),
    ]);

    let model = model_of(
        "anthropic/claude-fable-5",
        "anthropic-messages",
        "anthropic",
    );
    let (events, _) = run_anthropic_vector(&model, &user_ctx("blocked request"), sse).await;

    // Upstream: stopReason "error", errorMessage = stop_details.explanation,
    // usage replaced per-field (input 412, output 0, total 412).
    assert_events(
        "AV-2",
        &events,
        &[
            Expect::Start,
            Expect::Error(Terminal {
                content: vec![],
                stop: StopReason::Error,
                error_message: Some(REFUSAL_EXPLANATION.to_string()),
                usage: usage_of(412, 0, 412),
            }),
        ],
    );
}

/// AV-3 — upstream: "preserves sensitive stop reasons with a descriptive
/// error message".
///
/// The stop-reason semantics themselves pass since adapter fix AB-1
/// (`Other("sensitive")` → Error + "Provider stopped with: sensitive");
/// the residual failure is genai's usage accumulation.
#[tokio::test]
async fn av3_anthropic_sensitive_stop_reason() {
    let sse = anthropic_sse(&[
        anthropic_message_start("msg_sensitive", 12),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": "sensitive"},
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            })
            .to_string(),
        ),
        anthropic_message_stop(),
    ]);

    let model = model_of(
        "anthropic/claude-haiku-4-5",
        "anthropic-messages",
        "anthropic",
    );
    let (events, _) = run_anthropic_vector(&model, &user_ctx("blocked request"), sse).await;

    assert_events(
        "AV-3",
        &events,
        &[
            Expect::Start,
            Expect::Error(Terminal {
                content: vec![],
                stop: StopReason::Error,
                error_message: Some("Provider stopped with: sensitive".to_string()),
                usage: usage_of(12, 0, 12),
            }),
        ],
    );
}

/// AV-4 — upstream: "treats message_delta without usage as a no-op for
/// usage accumulation". PASS.
#[tokio::test]
async fn av4_anthropic_message_delta_without_usage_is_noop() {
    let events_fixture: Vec<(&str, String)> = minimal_anthropic_events()
        .into_iter()
        .map(|(event, data)| {
            if event == "message_delta" {
                (
                    event,
                    json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}})
                        .to_string(),
                )
            } else {
                (event, data)
            }
        })
        .collect();
    let sse = anthropic_sse(&events_fixture);

    let model = model_of(
        "anthropic/claude-haiku-4-5",
        "anthropic-messages",
        "anthropic",
    );
    let (events, request_line) = run_anthropic_vector(&model, &user_ctx("Say hello."), sse).await;
    assert!(
        request_line.contains("/messages"),
        "expected the real anthropic adapter URL, got {request_line:?}"
    );

    // Upstream: stopReason "stop", content [text "Hello"], usage.input 12,
    // usage.totalTokens 12 (message_start counts preserved).
    assert_events(
        "AV-4",
        &events,
        &[
            Expect::Start,
            Expect::TextStart(0),
            Expect::TextDelta(0, "Hello"),
            Expect::TextEnd(0),
            Expect::Done(Terminal {
                content: vec![text_block("Hello")],
                stop: StopReason::Stop,
                error_message: None,
                usage: usage_of(12, 0, 12),
            }),
        ],
    );
}

/// AV-5 — upstream: "ignores unknown SSE events after message_stop".
///
/// The vector's own semantic passes (genai stops polling after
/// message_stop, so `event: done` / `event: proxy.stats` junk is never
/// parsed); the residual failure is the same S-3 usage accumulation as
/// AV-3, because this fixture's message_delta carries usage.
#[tokio::test]
async fn av5_anthropic_ignores_unknown_events_after_message_stop() {
    let mut events_fixture = minimal_anthropic_events();
    events_fixture.push(("done", "[DONE]".to_string()));
    events_fixture.push(("proxy.stats", "not json".to_string()));
    let sse = anthropic_sse(&events_fixture);

    let model = model_of(
        "anthropic/claude-haiku-4-5",
        "anthropic-messages",
        "anthropic",
    );
    let (events, _) = run_anthropic_vector(&model, &user_ctx("Say hello."), sse).await;

    // Upstream: stopReason "stop", content [text "Hello"], no error; usage
    // replaced per-field by message_delta → input 12, output 5, total 17.
    assert_events(
        "AV-5",
        &events,
        &[
            Expect::Start,
            Expect::TextStart(0),
            Expect::TextDelta(0, "Hello"),
            Expect::TextEnd(0),
            Expect::Done(Terminal {
                content: vec![text_block("Hello")],
                stop: StopReason::Stop,
                error_message: None,
                usage: usage_of(12, 5, 17),
            }),
        ],
    );
}

// ---------------------------------------------------------------------------
// OpenAI-completions vectors
// ---------------------------------------------------------------------------

/// OV-1 — upstream test/openai-completions-raw-stop-reason.test.ts:
/// "preserves raw finish reasons for successful stops". PASS.
///
/// Upstream also asserts `rawStopReason === "stop"` — the raw string
/// crosses genai (`StopReason::Completed("stop")`) but grain's
/// `AssistantMessage` has no slot for it: reported gap AB-R1.
#[tokio::test]
async fn ov1_openai_preserves_raw_finish_reason_stop() {
    let sse = openai_sse(&[json!({
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    })]);

    let model = model_of("openai/test-model", "openai-completions", "openai");
    let (events, request_line) = run_vector(&model, &user_ctx("hello"), sse).await;
    assert!(
        request_line.contains("/chat/completions"),
        "expected the real openai adapter URL, got {request_line:?}"
    );

    assert_events(
        "OV-1",
        &events,
        &[
            Expect::Start,
            Expect::Done(Terminal {
                content: vec![],
                stop: StopReason::Stop,
                error_message: None,
                usage: usage_of(0, 0, 0),
            }),
        ],
    );
}

/// OV-2 — upstream test/openai-completions-raw-stop-reason.test.ts:
/// "preserves raw finish reasons for provider error stops". PASS since
/// adapter fix AB-1 (previously reported a clean `Stop`).
#[tokio::test]
async fn ov2_openai_content_filter_finish_reason_is_error() {
    let sse = openai_sse(&[json!({
        "id": "chatcmpl-2",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]
    })]);

    let model = model_of("openai/test-model", "openai-completions", "openai");
    let (events, _) = run_vector(&model, &user_ctx("hello"), sse).await;

    assert_events(
        "OV-2",
        &events,
        &[
            Expect::Start,
            Expect::Error(Terminal {
                content: vec![],
                stop: StopReason::Error,
                error_message: Some("Provider finish_reason: content_filter".to_string()),
                usage: usage_of(0, 0, 0),
            }),
        ],
    );
}

/// OV-3 — upstream test/openai-completions-response-model.test.ts:
/// "surfaces routed chunk.model on responseModel without changing model".
#[tokio::test]
#[ignore = "structural, and now genai-ONLY: upstream expects responseModel=\"anthropic/claude-opus-4.8\" captured from chunk.model. The grain half of this gap is CLOSED — AssistantMessage.response_model exists (WP19) and the native Anthropic transport populates it from message_start, proven end-to-end over a socket in tests/response_metadata.rs. What still blocks THIS vector is entirely genai 0.6.5: its openai streamer never reads chunk.model and StreamEnd has no field for it, so the value never crosses the seam on the genai path this vector exercises. genai would need to capture chunk.model on StreamEnd (e.g. captured_response_model)"]
async fn ov3_openai_routed_response_model_surfaces() {
    let sse = openai_sse(&[
        json!({
            "id": "chatcmpl-1",
            "model": "anthropic/claude-opus-4.8",
            "choices": [{"index": 0, "delta": {"content": "hi"}}]
        }),
        json!({
            "id": "chatcmpl-1",
            "model": "anthropic/claude-opus-4.8",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0}
            }
        }),
    ]);

    let model = model_of("openai/auto", "openai-completions", "openrouter");
    let (events, _) = run_vector(&model, &user_ctx("hi"), sse).await;

    // The representable translation passes end-to-end…
    assert_events(
        "OV-3",
        &events,
        &[
            Expect::Start,
            Expect::TextStart(0),
            Expect::TextDelta(0, "hi"),
            Expect::TextEnd(0),
            Expect::Done(Terminal {
                content: vec![text_block("hi")],
                stop: StopReason::Stop,
                error_message: None,
                usage: usage_of(10, 5, 15),
            }),
        ],
    );

    // …but the vector's actual subject cannot be expressed at this seam.
    panic!(
        "structural gap (measured, not representable): upstream expects \
         message.responseModel == \"anthropic/claude-opus-4.8\" from chunk.model; \
         chunk.model never crosses genai 0.6.5's ChatStreamEvent API and grain's \
         AssistantMessage has no response_model field"
    );
}

/// OV-4 — upstream test/openai-completions-response-model.test.ts:
/// "leaves responseModel undefined when chunks echo the requested id".
/// PASS (the absence semantics are vacuously exact: nothing is surfaced).
/// Exercises adapter fix AB-2 (total computed when the wire omits
/// total_tokens: upstream totalTokens = 1 + 1 = 2).
#[tokio::test]
async fn ov4_openai_response_model_echo_stays_unset() {
    let sse = openai_sse(&[
        json!({
            "id": "chatcmpl-2",
            "model": "openrouter/auto",
            "choices": [{"index": 0, "delta": {"content": "hi"}}]
        }),
        json!({
            "id": "chatcmpl-2",
            "model": "openrouter/auto",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0}
            }
        }),
    ]);

    let model = model_of("openai/auto", "openai-completions", "openrouter");
    let (events, _) = run_vector(&model, &user_ctx("hi"), sse).await;

    assert_events(
        "OV-4",
        &events,
        &[
            Expect::Start,
            Expect::TextStart(0),
            Expect::TextDelta(0, "hi"),
            Expect::TextEnd(0),
            Expect::Done(Terminal {
                content: vec![text_block("hi")],
                stop: StopReason::Stop,
                error_message: None,
                usage: usage_of(1, 1, 2),
            }),
        ],
    );
}

/// OV-5 — upstream test/openai-completions-response-model.test.ts:
/// "ignores empty or missing chunk.model". PASS (AB-2: total 1+2 = 3).
#[tokio::test]
async fn ov5_openai_ignores_empty_or_missing_chunk_model() {
    let sse = openai_sse(&[
        json!({
            "id": "chatcmpl-3",
            "choices": [{"index": 0, "delta": {"content": "hi"}}]
        }),
        json!({
            "id": "chatcmpl-3",
            "model": "",
            "choices": [{"index": 0, "delta": {"content": "!"}}]
        }),
        json!({
            "id": "chatcmpl-3",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0}
            }
        }),
    ]);

    let model = model_of("openai/auto", "openai-completions", "openrouter");
    let (events, _) = run_vector(&model, &user_ctx("hi"), sse).await;

    assert_events(
        "OV-5",
        &events,
        &[
            Expect::Start,
            Expect::TextStart(0),
            Expect::TextDelta(0, "hi"),
            Expect::TextDelta(0, "!"),
            Expect::TextEnd(0),
            Expect::Done(Terminal {
                content: vec![text_block("hi!")],
                stop: StopReason::Stop,
                error_message: None,
                usage: usage_of(1, 2, 3),
            }),
        ],
    );
}

/// OV-6 — upstream test/openai-completions-reasoning-details.test.ts:
/// "preserves reasoning_details that arrive before their matching tool
/// call".
#[tokio::test]
#[ignore = "structural, and now genai-ONLY: delta.reasoning_details never crosses genai — the openai streamer reads only delta.content / delta.reasoning_content / delta.reasoning (adapter/adapters/openai/streamer.rs), so the encrypted reasoning detail upstream attaches as toolCall.thoughtSignature is dropped. The grain half is CLOSED: ToolCall.thought_signature exists (WP19) and the cross-model replay rule is implemented in grain_agent_core::strip_cross_model_thought_signatures. genai would need to surface reasoning_details on its tool-call chunks for the replay contract to survive this seam. Note this vector is OpenAI/OpenRouter-specific and is NOT blocked by the native Anthropic transport: the Anthropic Messages wire carries no signature on tool_use at all (its signature lives on thinking blocks, which the transport already maps to ThinkingContent.signature), so there is nothing for that transport to attach"]
async fn ov6_openai_reasoning_details_attach_to_tool_call() {
    let reasoning_detail = json!({
        "type": "reasoning.encrypted",
        "id": "call_1",
        "data": "encrypted-signature"
    });
    let sse = openai_sse(&[
        json!({
            "id": "chatcmpl-test",
            "model": "google/gemini-test",
            "choices": [{"index": 0, "delta": {"reasoning_details": [reasoning_detail]}, "finish_reason": null}]
        }),
        json!({
            "id": "chatcmpl-test",
            "model": "google/gemini-test",
            "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": "{\"path\":\"README.md\"}"}
            }]}, "finish_reason": null}]
        }),
        json!({
            "id": "chatcmpl-test",
            "model": "google/gemini-test",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
        }),
    ]);

    let mut ctx = user_ctx("read the readme");
    ctx.tools = vec![tool_def(
        "read",
        json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }),
    )];
    let model = model_of("openai/gemini-test", "openai-completions", "openrouter");
    let (events, _) = run_vector(&model, &ctx, sse).await;

    // The representable translation (tool call itself + toolUse stop)
    // passes end-to-end…
    assert_events(
        "OV-6",
        &events,
        &[
            Expect::Start,
            Expect::ToolcallStart(0),
            Expect::ToolcallDelta(0, "{\"path\":\"README.md\"}".to_string()),
            Expect::ToolcallEnd(0),
            Expect::Done(Terminal {
                content: vec![tool_call_block(
                    "call_1",
                    "read",
                    json!({"path": "README.md"}),
                )],
                stop: StopReason::ToolUse,
                error_message: None,
                usage: usage_of(0, 0, 0),
            }),
        ],
    );

    // …but the vector's subject cannot be expressed at this seam.
    panic!(
        "structural gap (measured, not representable): upstream expects the \
         toolCall to carry thoughtSignature == {} for context replay; \
         delta.reasoning_details never crosses genai 0.6.5's ChatStreamEvent \
         API and grain's ToolCall has no signature slot",
        json!({"type": "reasoning.encrypted", "id": "call_1", "data": "encrypted-signature"})
    );
}

// ---------------------------------------------------------------------------
// Google / Gemini vectors — upstream test/google-raw-stop-reason.test.ts
// ---------------------------------------------------------------------------

/// GV-1 — upstream: "preserves raw Gemini finish reasons for Google
/// Generative AI errors" (MALFORMED_FUNCTION_CALL). PASS since adapter fix
/// AB-1 (previously reported a clean `Stop`).
#[tokio::test]
async fn gv1_google_malformed_function_call_is_error() {
    let sse = gemini_sse(&[json!({
        "responseId": "google-response-id",
        "candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 0,
            "totalTokenCount": 1
        }
    })]);

    let model = model_of("google/gemini-2.5-flash", "google-generative-ai", "google");
    let (events, request_line) = run_vector(&model, &user_ctx("hello"), sse).await;
    assert!(
        request_line.contains("/models/gemini-2.5-flash:streamGenerateContent?alt=sse"),
        "expected the real gemini adapter URL, got {request_line:?}"
    );

    assert_events(
        "GV-1",
        &events,
        &[
            Expect::Start,
            Expect::Error(Terminal {
                content: vec![],
                stop: StopReason::Error,
                error_message: Some("Provider stopped with: MALFORMED_FUNCTION_CALL".to_string()),
                usage: usage_of(1, 0, 1),
            }),
        ],
    );
}

/// GV-2 — upstream: "preserves raw Gemini finish reasons for Google Vertex
/// errors" (SAFETY). PASS since adapter fix AB-1.
///
/// Upstream drives this through the google-vertex transport; the wire shape
/// (GenerateContentResponse SSE) is identical. genai 0.6.5 does have a
/// dedicated Vertex adapter (`AdapterKind::Vertex`), but grain routes all
/// google models through the gemini namespace (`ProviderRouter`:
/// `google → gemini`), so at this seam the vertex leg collapses onto the
/// gemini wire — documented in SEAM-VECTORS.md.
#[tokio::test]
async fn gv2_google_vertex_safety_is_error() {
    let sse = gemini_sse(&[json!({
        "responseId": "google-response-id",
        "candidates": [{"finishReason": "SAFETY"}],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 0,
            "totalTokenCount": 1
        }
    })]);

    let model = model_of(
        "google/gemini-3-flash-preview",
        "google-vertex",
        "google-vertex",
    );
    let (events, _) = run_vector(&model, &user_ctx("hello"), sse).await;

    assert_events(
        "GV-2",
        &events,
        &[
            Expect::Start,
            Expect::Error(Terminal {
                content: vec![],
                stop: StopReason::Error,
                error_message: Some("Provider stopped with: SAFETY".to_string()),
                usage: usage_of(1, 0, 1),
            }),
        ],
    );
}
