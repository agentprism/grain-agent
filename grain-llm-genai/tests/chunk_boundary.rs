//! Permanent chunk-boundary fuzzing for the native Anthropic transport.
//!
//! # Why this exists
//!
//! `tests/seam_vectors.rs` replays each recorded fixture as a **single socket
//! write**. That is one delivery pattern out of unboundedly many, and it is
//! the *least* representative one: a real `bytes_stream()` chunk is a
//! TCP-shaped slice that can split anything, including the middle of a
//! multi-byte UTF-8 scalar or the middle of a `\r\n` pair.
//!
//! Two defects lived in exactly that blind spot and were invisible to the
//! whole vector suite:
//!
//! - a per-chunk `String::from_utf8_lossy` turned any scalar straddling a
//!   chunk boundary into replacement characters, **silently**;
//! - a per-chunk `replace("\r\n", "\n")` could not see a CR ending one chunk
//!   and an LF starting the next, fabricating a frame separator that split one
//!   event into two unparseable halves.
//!
//! Neither needed the network to find — both are decode-side and reproducible
//! offline. This file is therefore **infrastructure, not a regression test**:
//! it re-fuzzes every delivery pattern on every run, so a future change to the
//! decoder or the stream loop is checked automatically rather than only when
//! someone remembers to think about boundaries.
//!
//! # What it asserts
//!
//! The invariant is *delivery-independence*: for a given response body, the
//! grain event stream must be identical no matter how the bytes are chunked.
//! The single-write result is the baseline (it is what the vector suite
//! pins), and every other split pattern must equal it.
//!
//! Coverage per body: every single split offset `0..=len`, byte-at-a-time
//! delivery, and a set of deterministic pseudo-random multi-split patterns.

use futures::StreamExt;
use grain_agent_core::{
    AssistantMessageEvent, LlmContext, LlmStream, Message, Model, StreamOptions, TextContent,
    UserContent, UserMessage,
};
use grain_llm_genai::{AnthropicStream, AnthropicTransportConfig};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// A mock endpoint that writes the body in caller-chosen slices
// ---------------------------------------------------------------------------

/// Serve one request, writing `body` as the given byte slices with a flush
/// between each — so the client observes genuine chunk boundaries.
async fn serve_sse_chunked(chunks: Vec<Vec<u8>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock listener");
    let addr = listener.local_addr().expect("mock addr");

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");

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

        let total: usize = chunks.iter().map(Vec::len).sum();
        let header = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {total}\r\nconnection: close\r\n\r\n"
        );
        socket
            .write_all(header.as_bytes())
            .await
            .expect("write head");
        socket.flush().await.expect("flush head");

        for chunk in chunks {
            if chunk.is_empty() {
                continue;
            }
            socket.write_all(&chunk).await.expect("write chunk");
            socket.flush().await.expect("flush chunk");
            // Yield so the client is genuinely woken between writes rather
            // than coalescing everything into one read.
            tokio::task::yield_now().await;
        }
        socket.shutdown().await.ok();
    });

    format!("http://{addr}/")
}

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

/// Drive the native transport against a body delivered as `chunks`, and
/// project the result onto a comparable shape (event kind + payload).
async fn run_chunked(chunks: Vec<Vec<u8>>) -> Vec<String> {
    let base_url = serve_sse_chunked(chunks).await;
    let stream_impl = AnthropicStream::new(
        AnthropicTransportConfig::with_api_key("test-key").with_base_url(base_url),
    );

    let mut stream = stream_impl
        .stream(
            &model(),
            &ctx(),
            &StreamOptions::default(),
            CancellationToken::new(),
        )
        .await
        .expect("stream");

    let mut out = Vec::new();
    while let Some(ev) = stream.next().await {
        let terminal = ev.is_terminal();
        out.push(describe(&ev));
        if terminal {
            break;
        }
    }
    out
}

/// A comparison projection that captures everything delivery could corrupt:
/// event kinds, deltas verbatim, terminal content, usage and stop reason.
fn describe(e: &AssistantMessageEvent) -> String {
    match e {
        AssistantMessageEvent::Start { .. } => "Start".into(),
        AssistantMessageEvent::TextStart { content_index, .. } => {
            format!("TextStart({content_index})")
        }
        AssistantMessageEvent::TextDelta {
            content_index,
            delta,
            ..
        } => format!("TextDelta({content_index},{delta:?})"),
        AssistantMessageEvent::TextEnd { content_index, .. } => format!("TextEnd({content_index})"),
        AssistantMessageEvent::ThinkingStart { content_index, .. } => {
            format!("ThinkingStart({content_index})")
        }
        AssistantMessageEvent::ThinkingDelta {
            content_index,
            delta,
            ..
        } => format!("ThinkingDelta({content_index},{delta:?})"),
        AssistantMessageEvent::ThinkingEnd { content_index, .. } => {
            format!("ThinkingEnd({content_index})")
        }
        AssistantMessageEvent::ToolcallStart { content_index, .. } => {
            format!("ToolcallStart({content_index})")
        }
        AssistantMessageEvent::ToolcallDelta {
            content_index,
            delta,
            ..
        } => format!("ToolcallDelta({content_index},{delta:?})"),
        AssistantMessageEvent::ToolcallEnd { content_index, .. } => {
            format!("ToolcallEnd({content_index})")
        }
        AssistantMessageEvent::Done { result } => format!(
            "Done(stop={:?},content={:?},usage={}/{}/{})",
            result.stop_reason,
            result.content,
            result.usage.input,
            result.usage.output,
            result.usage.total_tokens
        ),
        AssistantMessageEvent::Error { error, result } => format!(
            "Error({error:?},stop={:?},content={:?},usage={}/{}/{})",
            result.stop_reason,
            result.content,
            result.usage.input,
            result.usage.output,
            result.usage.total_tokens
        ),
    }
}

// ---------------------------------------------------------------------------
// Bodies under fuzz
// ---------------------------------------------------------------------------

fn sse(events: &[(&str, String)], sep: &str) -> String {
    events
        .iter()
        .map(|(e, d)| format!("event: {e}{sep}data: {d}{sep}"))
        .collect::<Vec<_>>()
        .join(sep)
}

/// Text containing the characters Claude actually emits constantly: em-dash,
/// smart quotes, CJK, and an astral-plane emoji.
fn multibyte_body() -> String {
    sse(
        &[
            (
                "message_start",
                json!({"type":"message_start","message":{"id":"msg_x","usage":{
                    "input_tokens":11,"output_tokens":0,
                    "cache_read_input_tokens":0,"cache_creation_input_tokens":0}}})
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
                       "delta":{"type":"text_delta","text":"Cost: 12 — “quoted” 日本語 🎉"}})
                .to_string(),
            ),
            (
                "content_block_stop",
                json!({"type":"content_block_stop","index":0}).to_string(),
            ),
            (
                "message_delta",
                json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},
                       "usage":{"input_tokens":11,"output_tokens":9}})
                .to_string(),
            ),
            (
                "message_stop",
                json!({"type":"message_stop"}).to_string(),
            ),
        ],
        "\n",
    )
}

/// The same stream framed with CRLF, so every separator is a two-byte pair
/// that a chunk boundary can bisect.
fn crlf_body() -> String {
    sse(
        &[
            (
                "message_start",
                json!({"type":"message_start","message":{"id":"msg_x","usage":{
                    "input_tokens":7,"output_tokens":0}}})
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
                json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},
                       "usage":{"input_tokens":7,"output_tokens":2}})
                .to_string(),
            ),
            (
                "message_stop",
                json!({"type":"message_stop"}).to_string(),
            ),
        ],
        "\r\n",
    )
}

/// A tool call whose arguments carry multi-byte content, so the accumulated
/// `partial_json` buffer is also boundary-sensitive.
fn toolcall_body() -> String {
    sse(
        &[
            (
                "message_start",
                json!({"type":"message_start","message":{"id":"msg_t","usage":{
                    "input_tokens":5,"output_tokens":0}}})
                .to_string(),
            ),
            (
                "content_block_start",
                json!({"type":"content_block_start","index":0,
                       "content_block":{"type":"tool_use","id":"toolu_1","name":"write"}})
                .to_string(),
            ),
            (
                "content_block_delta",
                json!({"type":"content_block_delta","index":0,
                       "delta":{"type":"input_json_delta","partial_json":"{\"text\":\"héllo — 日本"}})
                .to_string(),
            ),
            (
                "content_block_delta",
                json!({"type":"content_block_delta","index":0,
                       "delta":{"type":"input_json_delta","partial_json":"語 🎉\"}"}})
                .to_string(),
            ),
            (
                "content_block_stop",
                json!({"type":"content_block_stop","index":0}).to_string(),
            ),
            (
                "message_delta",
                json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},
                       "usage":{"input_tokens":5,"output_tokens":4}})
                .to_string(),
            ),
            (
                "message_stop",
                json!({"type":"message_stop"}).to_string(),
            ),
        ],
        "\n",
    )
}

fn bodies() -> Vec<(&'static str, String)> {
    vec![
        ("multibyte", multibyte_body()),
        ("crlf", crlf_body()),
        ("toolcall", toolcall_body()),
    ]
}

// ---------------------------------------------------------------------------
// The fuzz
// ---------------------------------------------------------------------------

/// Baseline sanity: the single-write delivery every other case is compared
/// against must itself be correct, or the whole suite is vacuous.
#[tokio::test]
async fn baselines_are_correct_and_non_trivial() {
    let events = run_chunked(vec![multibyte_body().into_bytes()]).await;
    let joined = events.join(" | ");
    assert!(
        joined.contains("Cost: 12 — “quoted” 日本語 🎉"),
        "baseline must carry the multi-byte text intact: {joined}"
    );
    assert!(
        joined.contains("usage=11/9/20"),
        "baseline must carry replaced (not accumulated) usage: {joined}"
    );
    assert!(
        events.len() >= 5,
        "baseline should be a full event stream, got {events:?}"
    );

    let crlf = run_chunked(vec![crlf_body().into_bytes()]).await;
    assert!(
        crlf.join(" | ").contains("usage=7/2/9"),
        "CRLF baseline must parse: {crlf:?}"
    );
}

/// **The invariant.** For each body, every single split offset must produce
/// exactly the single-write event stream.
#[tokio::test]
async fn every_single_split_offset_is_delivery_independent() {
    for (name, body) in bodies() {
        let bytes = body.as_bytes().to_vec();
        let baseline = run_chunked(vec![bytes.clone()]).await;

        for split in 0..=bytes.len() {
            let chunks = vec![bytes[..split].to_vec(), bytes[split..].to_vec()];
            let got = run_chunked(chunks).await;
            assert_eq!(
                got, baseline,
                "[{name}] split at byte {split} changed the event stream"
            );
        }
    }
}

/// The pathological delivery: one byte per chunk.
#[tokio::test]
async fn byte_at_a_time_delivery_is_delivery_independent() {
    for (name, body) in bodies() {
        let bytes = body.as_bytes().to_vec();
        let baseline = run_chunked(vec![bytes.clone()]).await;
        let chunks: Vec<Vec<u8>> = bytes.iter().map(|b| vec![*b]).collect();
        let got = run_chunked(chunks).await;
        assert_eq!(got, baseline, "[{name}] byte-at-a-time delivery diverged");
    }
}

/// Deterministic pseudo-random multi-split patterns, so the fuzz covers
/// interactions between several boundaries in one stream rather than only a
/// single cut.
#[tokio::test]
async fn randomized_multi_split_patterns_are_delivery_independent() {
    for (name, body) in bodies() {
        let bytes = body.as_bytes().to_vec();
        let baseline = run_chunked(vec![bytes.clone()]).await;

        // xorshift64*, seeded per body — reproducible, no dev-dependency.
        let mut state: u64 = 0x9E3779B97F4A7C15 ^ (bytes.len() as u64);
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state = state.wrapping_mul(0x2545F4914F6CDD1D);
            state
        };

        for round in 0..24 {
            let mut chunks: Vec<Vec<u8>> = Vec::new();
            let mut pos = 0usize;
            while pos < bytes.len() {
                // Chunk sizes in 1..=7 bytes: small enough to bisect scalars
                // and CRLF pairs constantly.
                let take = 1 + (next() % 7) as usize;
                let end = (pos + take).min(bytes.len());
                chunks.push(bytes[pos..end].to_vec());
                pos = end;
            }
            let got = run_chunked(chunks).await;
            assert_eq!(
                got, baseline,
                "[{name}] randomized split round {round} diverged"
            );
        }
    }
}
