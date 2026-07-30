//! WP21 — characterization of what genai 0.6.5 **does and does not** deliver
//! across its public streaming API.
//!
//! These tests are the evidence behind the structural-gap verdicts in
//! `tests/SEAM-VECTORS.md` §6 and the upstream bug report in
//! `UPSTREAM-GENAI.md`. They deliberately assert genai's *current, defective*
//! behavior rather than the behavior we want: each one is a tripwire that
//! fails loudly if a genai bump changes the seam, at which point the
//! corresponding structural gap can be re-measured and the matching seam
//! vector un-`#[ignore]`d.
//!
//! Unlike `seam_vectors.rs`, which drives the full chain through
//! `GenaiStream` and asserts *grain* events, these tests consume
//! `genai::chat::ChatStreamEvent` directly — the seam itself. Nothing in
//! `grain-llm-genai` sits between the assertions and genai.
//!
//! The load-bearing test is [`s3_double_count_is_indistinguishable_at_the_seam`]:
//! it proves that the information needed to *correct* the S-3 Anthropic usage
//! double-count never crosses genai's **streaming event API**. That scopes the
//! claim precisely — it is a fact about `ChatStreamEvent`, not a claim that no
//! adapter-side mechanism exists at any price. One does: genai lets the caller
//! choose the endpoint it connects to, so a recording relay can tee the wire
//! and re-derive the truth. That route is rejected on cost, not feasibility —
//! see `SEAM-VECTORS.md` §6, "The endpoint-tee relay: possible, costed,
//! rejected".

use futures::StreamExt;
use genai::ServiceTarget;
use genai::chat::{ChatOptions, ChatRequest, ChatStreamEvent};
use genai::resolver::{AuthData, Endpoint};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ---------------------------------------------------------------------------
// Mock SSE endpoint (same shape as tests/seam_vectors.rs::serve_sse_once)
// ---------------------------------------------------------------------------

async fn serve_sse_once(body: String) -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock SSE listener");
    let addr = listener.local_addr().expect("mock listener addr");

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept mock connection");

        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 4096];
        let head_end = loop {
            let n = socket.read(&mut chunk).await.expect("read request");
            if n == 0 {
                panic!("client closed before sending a full request head");
            }
            buf.extend_from_slice(&chunk[..n]);
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
    });

    format!("http://{addr}/")
}

/// Drive genai's real Anthropic client against `sse_body` and collect the
/// raw `ChatStreamEvent`s that cross its public API.
async fn genai_events(sse_body: String) -> Vec<ChatStreamEvent> {
    let base_url = serve_sse_once(sse_body).await;
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

    // The same capture flags production uses (see `stream.rs`
    // `chat_options_with_runtime`), plus `capture_raw_body`, which the
    // streaming path ignores entirely — see
    // `capture_raw_body_is_inert_on_the_streaming_path`.
    let options = ChatOptions::default()
        .with_capture_usage(true)
        .with_capture_content(true)
        .with_capture_tool_calls(true)
        .with_capture_reasoning_content(true)
        .with_capture_raw_body(true);

    let resp = client
        .exec_chat_stream(
            "anthropic::claude-haiku-4-5",
            ChatRequest::new(vec![genai::chat::ChatMessage::user("probe")]),
            Some(&options),
        )
        .await
        .expect("exec_chat_stream against the mock endpoint");

    let mut stream = resp.stream;
    let mut out = Vec::new();
    while let Some(ev) = stream.next().await {
        match ev {
            Ok(ev) => {
                let terminal = matches!(ev, ChatStreamEvent::End(_));
                out.push(ev);
                if terminal {
                    break;
                }
            }
            Err(err) => panic!("genai stream error: {err}"),
        }
    }
    out
}

fn anthropic_sse(events: &[(&str, String)]) -> String {
    events
        .iter()
        .map(|(event, data)| format!("event: {event}\ndata: {data}\n"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn message_start(input_tokens: u64, output_tokens: u64) -> (&'static str, String) {
    (
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_probe",
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            }
        })
        .to_string(),
    )
}

fn message_stop() -> (&'static str, String) {
    ("message_stop", json!({"type": "message_stop"}).to_string())
}

fn usage_of(end: &ChatStreamEvent) -> (i32, i32, i32) {
    let ChatStreamEvent::End(end) = end else {
        panic!("expected an End event, got {end:?}");
    };
    let u = end
        .captured_usage
        .as_ref()
        .expect("capture_usage was enabled, so End must carry usage");
    (
        u.prompt_tokens.unwrap_or(0),
        u.completion_tokens.unwrap_or(0),
        u.total_tokens.unwrap_or(0),
    )
}

// ---------------------------------------------------------------------------
// S-3 — the Anthropic usage double-count, and why it is unrecoverable
// from the streaming event API
// ---------------------------------------------------------------------------

/// **The S-3 impossibility proof.**
///
/// Two *materially different* Anthropic wire streams:
///
/// - stream A ("the live shape"): `message_start` reports 12 input tokens and
///   `message_delta` repeats the same cumulative 12 — the real API always
///   repeats `input_tokens` in `message_delta`. The provider reported
///   **12** input tokens. Upstream pi-ai replaces per-field and reports 12
///   (`anthropic-messages.ts`, `message_delta` handler).
/// - stream B: `message_start` reports 24 input tokens and `message_delta`
///   carries no `usage` object at all. The provider reported **24** input
///   tokens. Upstream pi-ai reports 24 (this is exactly upstream's
///   "treats message_delta without usage as a no-op" case = seam vector AV-4).
///
/// genai 0.6.5's Anthropic streamer accumulates with `*val += input_tokens`
/// for both `message_start` and `message_delta`
/// (`adapter/adapters/anthropic/streamer.rs::capture_usage`), so both streams
/// collapse onto the **byte-identical** `StreamEnd`.
///
/// Consequently no function of what crosses genai's API can map stream A to
/// 12 and stream B to 24: any post-hoc correction applied at our boundary
/// must return the same answer for both, and exactly one of those answers is
/// wrong. Halving unconditionally would fix A and break B (B is a real
/// upstream-asserted shape, and AV-4 pins it green today).
///
/// This is what makes S-3 **unrecoverable from `ChatStreamEvent`**: the
/// corrective term — whether `message_delta` carried `input_tokens` at all —
/// is destroyed inside genai before anything the streaming API exposes.
///
/// Scope note, because the distinction matters: this is *not* a proof that
/// the adapter cannot fix S-3 by any means. genai lets the caller choose the
/// endpoint it connects to (`ServiceTargetResolver`, `WebConfig::with_proxy`,
/// `AuthData::RequestOverride`, `ModelSpec::Target`), so interposing a
/// recording relay and re-parsing the wire does recover the true 12 vs 24 —
/// this very test file uses that redirect mechanism to reach a local socket.
/// That route is rejected because re-parsing Anthropic SSE ourselves is most
/// of a provider backend, at which point keeping genai in the path buys
/// nothing. See `SEAM-VECTORS.md` §6.
#[tokio::test]
async fn s3_double_count_is_indistinguishable_at_the_seam() {
    // Stream A: 12 at message_start, repeated 12 at message_delta.
    let stream_a = anthropic_sse(&[
        message_start(12, 0),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            })
            .to_string(),
        ),
        message_stop(),
    ]);

    // Stream B: 24 at message_start, message_delta carries no usage at all.
    let stream_b = anthropic_sse(&[
        message_start(24, 0),
        (
            "message_delta",
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}).to_string(),
        ),
        message_stop(),
    ]);

    let events_a = genai_events(stream_a).await;
    let events_b = genai_events(stream_b).await;

    let end_a = events_a.last().expect("stream A produced events");
    let end_b = events_b.last().expect("stream B produced events");

    // genai inflates stream A to 24 (the S-3 defect) …
    assert_eq!(
        usage_of(end_a),
        (24, 0, 24),
        "S-3: genai must be observed adding message_delta.input_tokens onto \
         message_start (12 + 12 = 24). If this now reports 12, genai fixed \
         the defect — re-measure S-3 and un-ignore AV-3/AV-5."
    );

    // … and reports stream B faithfully as 24.
    assert_eq!(
        usage_of(end_b),
        (24, 0, 24),
        "stream B's provider-reported 24 must survive unchanged"
    );

    // The proof: the two are identical at the seam, so no adapter-side
    // correction can distinguish the inflated 24 from the honest 24.
    assert_eq!(
        usage_of(end_a),
        usage_of(end_b),
        "S-3 impossibility: a stream whose provider reported 12 and a stream \
         whose provider reported 24 are indistinguishable at genai's public \
         API. Any adapter-side correction returns one answer for both."
    );
}

/// S-1 — usage crosses genai's API **only** on `End`. No earlier event
/// carries a usage payload, so grain partials cannot mirror upstream's
/// running usage (upstream sets it at `message_start`, before any content
/// event is emitted).
#[tokio::test]
async fn s1_usage_crosses_only_at_end() {
    let sse = anthropic_sse(&[
        message_start(12, 0),
        (
            "content_block_start",
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})
                .to_string(),
        ),
        (
            "content_block_delta",
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}})
                .to_string(),
        ),
        (
            "content_block_stop",
            json!({"type":"content_block_stop","index":0}).to_string(),
        ),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 12, "output_tokens": 5}
            })
            .to_string(),
        ),
        message_stop(),
    ]);

    let events = genai_events(sse).await;

    // `ChatStreamEvent` has exactly one variant with a usage-bearing payload
    // (`End(StreamEnd)`); Start / Chunk / ReasoningChunk /
    // ThoughtSignatureChunk / ToolCallChunk carry no usage field at all.
    // Assert positionally: everything before the terminal is usage-free.
    let (last, leading) = events.split_last().expect("stream produced events");
    assert!(
        matches!(last, ChatStreamEvent::End(_)),
        "expected a terminal End event, got {last:?}"
    );
    assert!(
        leading.iter().all(|e| !matches!(e, ChatStreamEvent::End(_))),
        "S-1: only the terminal event may be End"
    );
    assert!(
        !leading.is_empty(),
        "fixture should stream content before End"
    );

    // And the terminal does carry it — i.e. the information exists, it is
    // simply not available until the stream is over.
    assert_eq!(usage_of(last), (24, 5, 29));
}

/// S-2 — `StreamEnd::captured_response_id` exists on the type but genai
/// 0.6.5's Anthropic streamer hard-codes it to `None`
/// (`streamer.rs`, the `message_stop` arm), so `message_start.message.id`
/// (which upstream captures as `responseId`) never crosses.
#[tokio::test]
async fn s2_response_id_never_crosses() {
    let sse = anthropic_sse(&[
        message_start(12, 0),
        (
            "message_delta",
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}).to_string(),
        ),
        message_stop(),
    ]);

    let events = genai_events(sse).await;
    let ChatStreamEvent::End(end) = events.last().expect("stream produced events") else {
        panic!("expected End");
    };
    assert_eq!(
        end.captured_response_id, None,
        "S-2: the wire carried message.id = \"msg_probe\"; if genai now \
         surfaces it, re-measure S-2"
    );
}

/// S-4 — Anthropic `delta.stop_details` is dropped. genai captures only
/// `delta.stop_reason`, so the refusal explanation upstream surfaces as
/// `errorMessage` never crosses. All that survives is the bare reason
/// string inside `StopReason::Other`.
#[tokio::test]
async fn s4_stop_details_never_crosses() {
    let explanation = "Blocked under Anthropic's Usage Policy.";
    let sse = anthropic_sse(&[
        message_start(412, 0),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": "refusal",
                    "stop_details": {
                        "type": "refusal",
                        "category": "cyber",
                        "explanation": explanation,
                    }
                },
                "usage": {"input_tokens": 412, "output_tokens": 0}
            })
            .to_string(),
        ),
        message_stop(),
    ]);

    let events = genai_events(sse).await;
    let ChatStreamEvent::End(end) = events.last().expect("stream produced events") else {
        panic!("expected End");
    };

    // The reason crosses …
    assert_eq!(
        end.captured_stop_reason,
        Some(genai::chat::StopReason::Other("refusal".to_string())),
        "the bare stop_reason string does cross genai"
    );
    // … but nothing on StreamEnd can carry the explanation. Serialize the
    // whole terminal payload and prove the string is absent from it.
    let serialized = serde_json::to_string(&end).expect("StreamEnd is Serialize");
    assert!(
        !serialized.contains(explanation),
        "S-4: stop_details.explanation must not appear anywhere on StreamEnd; \
         if it does, genai started parsing stop_details — re-measure S-4"
    );
    assert!(
        !serialized.contains("cyber"),
        "S-4: stop_details.category must not appear anywhere on StreamEnd"
    );
}

/// S-8 — Anthropic `content_block_start` / `content_block_stop` are consumed
/// inside genai and produce no seam event, so block boundaries must be
/// synthesized by the adapter. A **text** block opened and closed without any
/// delta therefore vanishes entirely: it emits nothing at the seam.
#[tokio::test]
async fn s8_empty_block_vanishes_at_the_seam() {
    let sse = anthropic_sse(&[
        message_start(12, 0),
        (
            "content_block_start",
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})
                .to_string(),
        ),
        // No content_block_delta at all — upstream still emits
        // text_start + text_end for this block (pi-ai maps
        // content_block_start/stop 1:1).
        (
            "content_block_stop",
            json!({"type":"content_block_stop","index":0}).to_string(),
        ),
        (
            "message_delta",
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}).to_string(),
        ),
        message_stop(),
    ]);

    let events = genai_events(sse).await;
    let kinds: Vec<&str> = events
        .iter()
        .map(|e| match e {
            ChatStreamEvent::Start => "Start",
            ChatStreamEvent::Chunk(_) => "Chunk",
            ChatStreamEvent::ReasoningChunk(_) => "ReasoningChunk",
            ChatStreamEvent::ThoughtSignatureChunk(_) => "ThoughtSignatureChunk",
            ChatStreamEvent::ToolCallChunk(_) => "ToolCallChunk",
            ChatStreamEvent::End(_) => "End",
        })
        .collect();
    assert_eq!(
        kinds,
        vec!["Start", "End"],
        "S-8: the empty text block produced no seam event whatsoever, so the \
         adapter cannot reproduce upstream's text_start/text_end pair"
    );
}

/// `capture_raw_body` is the only genai option that sounds like it could
/// expose the wire to the adapter. It is inert on every streaming path in
/// 0.6.5 — `ChatStreamResponse` has just `stream` + `model_iden`, and no
/// streamer consults the flag. Pinned here because it is the load-bearing
/// "no escape hatch" half of the S-1/S-3 impossibility argument.
#[tokio::test]
async fn capture_raw_body_is_inert_on_the_streaming_path() {
    let sse = anthropic_sse(&[
        message_start(12, 0),
        (
            "message_delta",
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}).to_string(),
        ),
        message_stop(),
    ]);

    // `genai_events` already enables `with_capture_raw_body(true)`.
    let events = genai_events(sse).await;
    let ChatStreamEvent::End(end) = events.last().expect("stream produced events") else {
        panic!("expected End");
    };
    let serialized = serde_json::to_string(&end).expect("StreamEnd is Serialize");
    assert!(
        !serialized.contains("message_start"),
        "capture_raw_body must not smuggle raw wire frames onto StreamEnd; \
         if it does, the S-1/S-3 impossibility argument needs re-measuring"
    );
}
