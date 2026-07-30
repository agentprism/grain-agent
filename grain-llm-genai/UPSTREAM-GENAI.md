# Upstream bug report — `genai` (jeremychone/rust-genai)

Prepared by the grain-agent team (WP21). **Not yet filed.** This file is a
self-contained report that can be pasted into an upstream issue as-is; it
deliberately contains no grain-internal jargon. Nothing here has been
reported to, or filed against, any third-party repository.

Primary defect: **§1 — Anthropic streaming usage is double-counted.** The
remaining sections are secondary observations from the same audit, included
because they share a root cause (the streaming seam discards provider detail
before it becomes observable) and are cheap to address together.

- **Affected versions:** `0.6.5` (latest stable at the time of writing) and
  `0.7.0-beta.15` (latest published, verified unfixed — see §1.6).
- **Reference implementation used as the standard:** the TypeScript
  `pi-ai` package, which parses the same Anthropic wire format.
- **Reproduction:** self-contained, no provider credentials required (§1.4).

---

## 1. Anthropic streaming usage is double-counted (`input_tokens` and `output_tokens`)

### 1.1 Summary

`AnthropicStreamer::capture_usage` **adds** the token counters from every
usage-bearing event onto a running total. Anthropic's `message_delta.usage`
is not a delta — it is a **cumulative snapshot** of the whole message.
Adding it onto the `message_start` snapshot double-counts every counter the
provider repeats.

On any real Anthropic stream this inflates the reported prompt tokens by
roughly 2x, and inflates completion tokens by whatever `message_start`
reported (Anthropic sends a small non-zero `output_tokens` there).

### 1.2 Root cause

`src/adapter/adapters/anthropic/streamer.rs`

Both `message_start` and `message_delta` are routed into the same
accumulator (lines 58–71):

```rust
"message_start" => {
    self.capture_usage(message_type, &message.data)?;
    continue;
}
"message_delta" => {
    self.capture_usage(message_type, &message.data)?;
    // ... capture stop_reason ...
    continue;
}
```

`capture_usage` (line 262) selects the right JSON pointers per event type
(lines 267–271) and then **accumulates with `+=`** (lines 281–299):

```rust
// -- Capture/Add the eventual input_tokens
if let Ok(input_tokens) = data.x_get::<i32>(input_path) {
    let val = self
        .captured_data
        .usage
        .get_or_insert(Usage::default())
        .prompt_tokens
        .get_or_insert(0);
    *val += input_tokens;          // line 288
}

if let Ok(output_tokens) = data.x_get::<i32>(output_path) {
    let val = self
        .captured_data
        .usage
        .get_or_insert(Usage::default())
        .completion_tokens
        .get_or_insert(0);
    *val += output_tokens;         // line 298
}
```

The `+=` is correct for a *delta*-shaped wire protocol. Anthropic's is not.

Per Anthropic's Messages streaming documentation, `message_delta.usage`
carries the **cumulative** usage for the message so far; the final
`message_delta` carries the totals for the whole message. Current API
versions include `input_tokens`, `cache_creation_input_tokens` and
`cache_read_input_tokens` there in addition to `output_tokens`, and the
input counters simply repeat the values already sent in `message_start`.

The reference implementation treats each field as a **replacement**, guarded
so that a field absent from `message_delta` preserves the `message_start`
value:

```ts
// pi-ai, packages/ai/src/api/anthropic-messages.ts (message_delta handler)
if (event.usage) {
    if (event.usage.input_tokens != null) {
        output.usage.input = event.usage.input_tokens;      // replace, not +=
    }
    if (event.usage.output_tokens != null) {
        output.usage.output = event.usage.output_tokens;
    }
    if (event.usage.cache_read_input_tokens != null) {
        output.usage.cacheRead = event.usage.cache_read_input_tokens;
    }
    if (event.usage.cache_creation_input_tokens != null) {
        output.usage.cacheWrite = event.usage.cache_creation_input_tokens;
    }
}
```

### 1.3 Observed vs expected

Wire (abridged; this is the standard shape of a real Anthropic stream):

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_x","usage":{
        "input_tokens":12,"output_tokens":0,
        "cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{
        "input_tokens":12,"output_tokens":5,
        "cache_read_input_tokens":0,"cache_creation_input_tokens":0}}

event: message_stop
data: {"type":"message_stop"}
```

The provider reported **12 input / 5 output / 17 total**.

| counter | genai 0.6.5 | expected |
|---|---|---|
| `prompt_tokens` | **24** | 12 |
| `completion_tokens` | 5 | 5 |
| `total_tokens` | **29** | 17 |

`completion_tokens` happens to be correct here only because this stream
reports `output_tokens: 0` at `message_start`. A live stream reports a small
non-zero value there, so in production **both** counters are inflated.

Cache counters are handled separately (line 304 onward, `message_start`
only) and are not double-counted; the defect is confined to `input_tokens`
and `output_tokens`.

### 1.4 Minimal reproduction

Self-contained: serves the frames above from a local socket and points the
client at it via `ServiceTargetResolver`. No API key, no network egress.

```rust
// Cargo.toml: genai = "0.6.5", tokio = { version = "1", features = ["full"] },
//             futures = "0.3", serde_json = "1"

use futures::StreamExt;
use genai::ServiceTarget;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, ChatStreamEvent};
use genai::resolver::{AuthData, Endpoint};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const SSE: &str = concat!(
    "event: message_start\n",
    r#"data: {"type":"message_start","message":{"id":"msg_x","usage":{"input_tokens":12,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#, "\n",
    "\n",
    "event: message_delta\n",
    r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":12,"output_tokens":5,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}"#, "\n",
    "\n",
    "event: message_stop\n",
    r#"data: {"type":"message_stop"}"#, "\n",
);

#[tokio::main]
async fn main() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 8192];
        let _ = sock.read(&mut buf).await.unwrap();
        let res = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            SSE.len(), SSE);
        sock.write_all(res.as_bytes()).await.unwrap();
        sock.shutdown().await.ok();
    });

    let base = format!("http://{addr}/");
    let client = genai::Client::builder()
        .with_auth_resolver_fn(|_| Ok(Some(AuthData::from_single("x"))))
        .with_service_target_resolver_fn(move |mut t: ServiceTarget| {
            t.endpoint = Endpoint::from_owned(base.clone());
            t.auth = AuthData::from_single("x");
            Ok(t)
        })
        .build();

    let resp = client
        .exec_chat_stream(
            "anthropic::claude-haiku-4-5",
            ChatRequest::new(vec![ChatMessage::user("hi")]),
            Some(&ChatOptions::default().with_capture_usage(true)),
        )
        .await
        .unwrap();

    let mut stream = resp.stream;
    while let Some(Ok(ev)) = stream.next().await {
        if let ChatStreamEvent::End(end) = ev {
            println!("{:?}", end.captured_usage);
            // observed: prompt_tokens: Some(24), completion_tokens: Some(5),
            //           total_tokens: Some(29)
            // expected: prompt_tokens: Some(12), completion_tokens: Some(5),
            //           total_tokens: Some(17)
        }
    }
}
```

### 1.5 Impact

Any consumer that bills, budgets, or displays cost from
`StreamEnd::captured_usage` over-reports Anthropic prompt tokens by ~2x.

Critically, the error is **not correctable downstream**. `capture_usage`
collapses `message_start` and `message_delta` into a single accumulator
before anything is observable, and the streaming API exposes no other
carrier for the raw values (`ChatStreamResponse` has only `stream` and
`model_iden`; `ChatOptions::capture_raw_body` is not consulted by any
streaming path). Two materially different streams therefore become
byte-identical at the public API:

| stream | provider truth | genai `StreamEnd` |
|---|---|---|
| `message_start` 12, `message_delta` repeats 12 | 12 | 24 |
| `message_start` 24, `message_delta` carries no `usage` | 24 | 24 |

Both are legal Anthropic streams (the second is what proxies that omit
usage from `message_delta` produce — and the `+= `-based code is in fact
correct for that one). A downstream consumer cannot tell them apart, so it
cannot apply any correction that is right for both. The fix has to live in
the streamer.

### 1.6 Still present on `main`

Verified against the published `0.7.0-beta.15` crate: `capture_usage` still
accumulates with `*val += input_tokens` / `*val += output_tokens` (lines 307
and 317 of the same file). That release *did* revisit the surrounding cache
logic — it added a `message_delta` fallback for gateways that report cache
counters there, with an explicit comment about avoiding double counting for
those fields — but the `input_tokens` / `output_tokens` accumulation was not
changed.

### 1.7 Suggested fix

Give `message_delta` replacement semantics while leaving `message_start` as
the initializer, and skip fields the event omits so proxies that drop
`input_tokens` from `message_delta` keep the `message_start` value:

```rust
if let Ok(input_tokens) = data.x_get::<i32>(input_path) {
    let usage = self.captured_data.usage.get_or_insert(Usage::default());
    match message_type {
        // message_delta carries cumulative totals for the whole message.
        "message_delta" => usage.prompt_tokens = Some(input_tokens),
        _ => *usage.prompt_tokens.get_or_insert(0) += input_tokens,
    }
}
```

…and the same for `completion_tokens`. Note the cache block at line 304
adds `cache_creation + cache_read` into `prompt_tokens` for `message_start`;
if `message_delta` replaces `prompt_tokens`, that cache contribution must be
re-applied from the `message_delta` counters (or the replacement must be
applied to a cache-exclusive sub-total) so cache-inclusive accounting stays
consistent. A regression test built from the wire in §1.3 pins the whole
interaction.

---

## 2. Secondary observations

These are lower severity and reported together because they stem from the
same design point: provider detail is consumed inside the streamer and has
no representation on `ChatStreamEvent` / `StreamEnd`.

### 2.1 `captured_response_id` is hard-coded `None` for Anthropic

`src/adapter/adapters/anthropic/streamer.rs:234` sets
`captured_response_id: None` in the `InterStreamEnd` built at `message_stop`,
even though the id is available: `message_start.message.id` is parsed a few
lines earlier. The OpenAI and Gemini streamers do the same. The field exists
on the public `StreamEnd`, so it reads as an oversight rather than a design
choice. The reference implementation captures `message.id` (Anthropic) and
`chunk.id` (OpenAI chat-completions) as `responseId`.

### 2.2 Anthropic `delta.stop_details` is dropped

The `message_delta` arm (lines 62–71) captures `delta.stop_reason` but not
`delta.stop_details`. For `stop_reason: "refusal"`, Anthropic sends

```json
"stop_details": {"type":"refusal","category":"cyber","explanation":"<why>"}
```

and the `explanation` is the only human-meaningful description of the
refusal. It is currently unreachable by consumers — only the bare string
`"refusal"` survives, inside `StopReason::Other`. Surfacing it on
`StreamEnd` (e.g. `captured_stop_details`) would let consumers report the
provider's actual reason.

### 2.3 A malformed SSE frame aborts the whole stream

`content_block_delta` (and the other structural arms) call
`serde_json::from_str` on the raw frame and propagate `Error::StreamParse`,
which terminates the stream and discards everything already accumulated.

Providers do occasionally emit frames whose *string literals* are malformed
— an invalid escape such as `\H`, or a raw TAB that was not escaped. The
reference implementation repairs exactly those two cases (escape raw control
characters inside strings; double backslashes before invalid escapes) and
completes the turn normally. A lenient retry on parse failure, limited to
that narrow repair, would convert a class of hard stream aborts into
successful turns without loosening JSON validity anywhere else.

### 2.4 OpenAI chat-completions: `chunk.model` is never read

The OpenAI streamer does not read `chunk.model`, so a routed model id (an
OpenRouter `auto` request resolving to a concrete model, for example) cannot
be recovered by the caller. A `captured_response_model` on `StreamEnd`,
populated when `chunk.model` is non-empty and differs from the requested id,
would cover it.

### 2.5 OpenAI chat-completions: `delta.reasoning_details` is dropped

The streamer reads `delta.content`, `delta.reasoning_content` and
`delta.reasoning`, but not `delta.reasoning_details`. OpenRouter-style
encrypted reasoning details arrive there, keyed by tool-call id, and must be
replayed on the following request for providers that require signed
reasoning context. Without them the reasoning chain cannot be reconstructed
across turns.

### 2.6 Usage is only observable at `End`

`ChatStreamEvent` has no usage-bearing variant other than `End(StreamEnd)`.
Anthropic reports input tokens in `message_start`, before any content is
generated, so a consumer that wants to show or budget prompt cost while the
response streams (or that loses the connection mid-stream) cannot: the
information has arrived on the wire but is held until the stream completes.
An early usage event, or a usage field on the existing events, would expose
what the provider already sent.

### 2.7 Block boundaries are not observable

Anthropic's `content_block_start` / `content_block_stop` are consumed
internally and produce no event. Consumers must infer block transitions from
changes in event kind, which cannot represent a block that opens and closes
without any delta — such a block disappears entirely. Emitting explicit
block-boundary events (or annotating chunks with a block index) would make
the structure recoverable.

---

## 3. Environment

- `genai` `0.6.5` (also verified against `0.7.0-beta.15`)
- Reproduction is transport-local (`127.0.0.1`), no provider account needed
- Frames in §1.3 / §1.4 are the standard Anthropic Messages streaming shape
