# WP5 — Seam Conformance Vectors

Measurement of whether grain-llm-genai's event stream is faithful to
upstream pi-ai, using **upstream's own stream-parsing fixtures** as the
standard.

- Upstream reference: `pi/packages/ai` (TypeScript), pinned commit
  `34239180ac5c80366def592b529a3a1b882b4a16` (read-only).
- genai version under measurement: **0.6.5** (workspace pin).
- Harness: `tests/seam_vectors.rs`.

## 1. Chain under test

```
recorded provider SSE (upstream fixture, frame-faithful)
  → local mock HTTP endpoint (tests/seam_vectors.rs::serve_sse_once)
  → genai 0.6.5 real client + provider streamer   (production code)
  → genai ChatStreamEvents                        (the seam)
  → grain-llm-genai inbound adapter (GenaiStream) (production code)
  → grain AssistantMessageEvents                  (asserted)
```

Only the endpoint URL and auth are overridden (via genai's
`ServiceTargetResolver` / `AuthResolver`, the same hooks production
`GenaiStreamBuilder` uses). Adapter selection, URL construction, request
building, SSE parsing, and stream accumulation are all real genai code;
one representative vector per adapter family (4 of 13: AV-1 and AV-4 for
anthropic, OV-1 for openai, GV-1 for gemini) asserts the request line
(`POST /messages`, `POST /chat/completions`,
`POST /models/<m>:streamGenerateContent?alt=sse`) to prove it — the other
vectors share the same adapter paths.

Wire reconstruction: upstream's Anthropic fixtures are already raw SSE and
are served **frame-faithfully** — same event framing, same
`event:`/`data:` lines, same absence of a trailing blank line (genai's
splitter must flush the final `message_stop` on EOF, and does). Frame
payloads rebuilt with `serde_json::json!` carry alphabetized object keys
(serde_json without `preserve_order`), which is semantically identical
JSON; the one payload where byte order/content is load-bearing — AV-1's
malformed `content_block_delta`, whose brokenness IS the fixture — is a
raw string and **byte-exact**. Upstream's OpenAI and Google fixtures mock
the SDK layer with wire-shaped chunk objects; the SDKs parse SSE `data:`
frames 1:1 into those objects, so the reconstruction (`data: <chunk>`
frames, plus the `data: [DONE]` sentinel the OpenAI SDK consumes
internally) is the exact production wire format.

## 2. Event-mapping table (upstream → grain)

Upstream protocol: `packages/ai/src/types.ts` `AssistantMessageEvent`.
Grain protocol: `grain-agent-core::AssistantMessageEvent`.

| upstream event | grain event | payload mapping |
|---|---|---|
| `start {partial}` | `Start {partial}` | 1:1 |
| `text_start {contentIndex, partial}` | `TextStart {content_index, partial}` | 1:1 |
| `text_delta {contentIndex, delta, partial}` | `TextDelta {content_index, delta, partial}` | 1:1 |
| `text_end {contentIndex, content, partial}` | `TextEnd {content_index, partial}` | upstream's `content` payload ≙ `partial.content[i].text` |
| `thinking_start/_delta/_end` | `ThinkingStart/Delta/End` | as text; upstream `thinkingSignature` ≙ `ThinkingContent::signature` |
| `toolcall_start {contentIndex, partial}` | `ToolcallStart {content_index, partial}` | 1:1 |
| `toolcall_delta {contentIndex, delta, partial}` | `ToolcallDelta {content_index, delta, partial}` | 1:1 |
| `toolcall_end {contentIndex, toolCall, partial}` | `ToolcallEnd {content_index, partial}` | upstream's `toolCall` payload ≙ `partial.content[i]` |
| `done {reason, message}` | `Done {result}` | `reason` ∈ stop\|length\|toolUse folded into `result.stop_reason` |
| `error {reason, error}` | `Error {error, result}` | upstream's `error` is the message with `stopReason` error\|aborted and `errorMessage`; grain splits the string + result |

`AssistantMessage` field mapping:

| upstream field | grain field | status |
|---|---|---|
| `content[]` (text/thinking/toolCall) | `content: Vec<AssistantContent>` | mapped; `toolCall.thoughtSignature` has **no grain slot** (see S-7 / AB-R1 family) |
| `stopReason` stop/length/toolUse/error/aborted | `StopReason` Stop/Length/ToolUse/Error/Aborted | mapped; upstream's mid-stream `"pending"` placeholder has no grain variant — grain partials carry the default `Stop` (cosmetic, excluded from comparison) |
| `rawStopReason` | — none | **AB-R1** (info crosses genai; grain drops it) |
| `errorMessage` | `error_message` | mapped |
| `usage.input` (cache-**exclusive**) | `usage.input` (cache-**inclusive**, see `Cost::cost_for`) | convention differs; numerically equal whenever cache counters are 0 (all vectors here) |
| `usage.output/cacheRead/cacheWrite/totalTokens` | `output/cache_read/cache_write/total_tokens` | mapped (total via AB-2 fallback when the wire omits it; see AB-2 divergence notes in §5) |
| `usage.reasoning` | `usage.reasoning: Option<u64>` | mapped (**AB-R2 closed** after the WP4 rebase gave grain the field). Nuance: genai's `zero_as_none` deserialization turns a wire `reasoning_tokens: 0` into `None`, so upstream's `Some(0)` for openai-completions/google is not reproducible — wire zero and absent field are indistinguishable at this seam |
| `responseId` | — none | **S-2** |
| `responseModel` | — none | **S-6** |
| `model`/`api`/`provider` | same names | grain echoes its own namespaced config (`anthropic/claude-…`); excluded from comparison |
| `timestamp` | `timestamp` | excluded (wall clock) |

Stop-reason resolution at the seam (adapter fix AB-1,
`src/mapping/inbound.rs::resolve_stop_reason`): genai's
`StreamEnd::captured_stop_reason` preserves the raw provider string inside
each variant; grain maps `Completed/StopSequence → Stop`,
`MaxTokens → Length`, `ToolCall → ToolUse`, `ContentFilter/Other → Error`
with upstream-parity messages (`Provider finish_reason: <raw>` for
`openai*` APIs, `Provider stopped with: <raw>` otherwise;
`Other("pause_turn") → Stop` unconditionally and `Other("refusal")` →
anthropic's no-details refusal message, per upstream `mapStopReason`
tables). The tool-call `→ ToolUse` override on `Completed/StopSequence` is
scoped to the google API family exactly as upstream scopes it
(`google-generative-ai.ts:231-235`, `google-vertex.ts:231-235`); anthropic
and openai map `end_turn`/`stop`/`stop_sequence` to `Stop` verbatim even
with tool-call content streamed — we mirror upstream per provider rather
than generalizing over it. Pinned by unit tests in `tests/inbound.rs`
(`gemini_completed_with_tool_calls_overrides_to_tool_use`,
`anthropic_end_turn_with_tool_calls_stays_stop`,
`captured_pause_turn_is_resubmittable_stop`).

## 3. Comparison scope

Asserted per vector: exact event-kind sequence, `content_index` per event,
`delta` payloads verbatim, and the full terminal message (content blocks,
stop reason, error message, usage incl. cost struct).

Excluded and tracked as named gaps instead (so one cross-cutting gap does
not smear every vector into STRUCTURAL):

- mid-stream `partial.usage` — upstream partials carry running usage from
  `message_start` / per-chunk usage; genai surfaces usage only at `End`
  (**S-1**, the canonical genai-seam gap);
- mid-stream `partial.stopReason` `"pending"` placeholder (no grain
  variant; cosmetic);
- `timestamp`, `model`/`api`/`provider` echoes;
- `rawStopReason` (**AB-R1**), `usage.reasoning` (**AB-R2**),
  `responseId` (**S-2**) — no grain slot to assert against; each is
  inventoried below with the vectors upstream asserts it on.

STRUCTURAL vectors assert the *upstream-translated* expectation and are
intentionally **red** under `cargo test -- --ignored`; the red is the
measurement. Where the upstream expectation is not even representable in
grain types (OV-3 responseModel, OV-6 thoughtSignature), the vector first
proves the representable translation passes end-to-end, then fails with an
explicit `structural gap` panic naming the unrepresentable field.

## 4. Vector inventory

| Vector | Provider / API | Upstream test (`packages/ai/test/`) | Upstream case | Classification | Reason |
|---|---|---|---|---|---|
| AV-1 | anthropic | `anthropic-sse-parsing.test.ts` | repairs malformed SSE JSON and malformed streamed tool JSON | **STRUCTURAL** | S-5: genai serde-parses each frame; malformed JSON aborts the stream (`Error::StreamParse`) instead of being repaired. Observed: chain stops after `Start, ToolcallStart` with a terminal `Error` |
| AV-2 | anthropic | `anthropic-sse-parsing.test.ts` | preserves refusal stop details from message_delta | **STRUCTURAL** | S-4: `delta.stop_details.explanation` never crosses genai — grain can only produce the no-details refusal default. Also hits S-3 (usage 824 vs 412) |
| AV-3 | anthropic | `anthropic-sse-parsing.test.ts` | preserves sensitive stop reasons with a descriptive error message | **STRUCTURAL** | S-3: genai adds message_delta `input_tokens` onto message_start (24/24 vs upstream 12/12). Stop semantics (`error` + `Provider stopped with: sensitive`) pass since AB-1 |
| AV-4 | anthropic | `anthropic-sse-parsing.test.ts` | treats message_delta without usage as a no-op for usage accumulation | **PASS** | Full chain reproduces upstream: events, content, stop, usage 12/0/12 |
| AV-5 | anthropic | `anthropic-sse-parsing.test.ts` | ignores unknown SSE events after message_stop | **STRUCTURAL** | S-3 (24/5/29 vs upstream 12/5/17). The vector's own semantic — junk events after message_stop ignored — passes (genai stops polling after message_stop) |
| OV-1 | openai-completions | `openai-completions-raw-stop-reason.test.ts` | preserves raw finish reasons for successful stops | **PASS** | Events/stop/usage exact. Upstream's `rawStopReason === "stop"` leg is AB-R1 (no grain slot; raw string does cross genai) |
| OV-2 | openai-completions | `openai-completions-raw-stop-reason.test.ts` | preserves raw finish reasons for provider error stops | **PASS** | Fixed by AB-1 (was: silent `Done/Stop`). Error event + `Provider finish_reason: content_filter` exact. `rawStopReason` → AB-R1 |
| OV-3 | openai-completions | `openai-completions-response-model.test.ts` | surfaces routed chunk.model on responseModel without changing model | **STRUCTURAL** | S-6: `chunk.model` never crosses genai; grain also lacks `response_model`. Representable remainder (text events, stop, usage 10/5/15) passes |
| OV-4 | openai-completions | `openai-completions-response-model.test.ts` | leaves responseModel undefined when chunks echo the requested id | **PASS** | Absence semantics vacuously exact; usage total 2 via AB-2 |
| OV-5 | openai-completions | `openai-completions-response-model.test.ts` | ignores empty or missing chunk.model | **PASS** | Two text deltas aggregate; usage total 3 via AB-2 |
| OV-6 | openai-completions (openrouter-flavored) | `openai-completions-reasoning-details.test.ts` | preserves reasoning_details that arrive before their matching tool call | **STRUCTURAL** | S-7: `delta.reasoning_details` never crosses genai; grain `ToolCall` has no signature slot. Representable remainder (toolcall events, args, toolUse stop) passes |
| GV-1 | google-generative-ai | `google-raw-stop-reason.test.ts` | preserves raw Gemini finish reasons for Google Generative AI errors (`MALFORMED_FUNCTION_CALL`) | **PASS** | Fixed by AB-1 (was: silent `Done/Stop`). Error + `Provider stopped with: MALFORMED_FUNCTION_CALL`, usage 1/0/1 exact. `rawStopReason` → AB-R1 |
| GV-2 | google-vertex | `google-raw-stop-reason.test.ts` | preserves raw Gemini finish reasons for Google Vertex errors (`SAFETY`) | **PASS** | Fixed by AB-1. Upstream drives this through the vertex transport; the wire (GenerateContentResponse SSE) is identical. genai 0.6.5 does have a dedicated Vertex adapter (`AdapterKind::Vertex`) — the collapse onto the gemini wire is grain's routing (`ProviderRouter`: `google → gemini`; no grain models route to the Vertex adapter today) |

## 5. Summary

| Classification | Count | Vectors |
|---|---|---|
| PASS | 7 | AV-4, OV-1, OV-2, OV-4, OV-5, GV-1, GV-2 |
| STRUCTURAL | 6 | AV-1, AV-2, AV-3, AV-5, OV-3, OV-6 |
| ADAPTER-BUG (vector left failing) | 0 | — |

Adapter bugs found while building the vectors:

**Fixed in-branch:**

- **AB-1** — `InboundState::on_end` ignored genai's
  `StreamEnd::captured_stop_reason` entirely and inferred Stop/ToolUse from
  content, silently reporting a clean `Stop` for content-filtered, refused,
  malformed-function-call, and truncated turns. Fixed in
  `src/mapping/inbound.rs` (`resolve_stop_reason`), upstream-parity
  mapping + error-event terminal. Cited by OV-1, OV-2, GV-1, GV-2, AV-3.
- **AB-2** — `map_usage` trusted genai's `total_tokens` verbatim; genai's
  OpenAI streamer passes an absent wire total through as `None`, so grain
  reported `total_tokens = 0` where upstream always computes the total from
  components. Fixed in `src/mapping/usage.rs` (fallback
  `input + output + cache_write`, respecting grain's cache-inclusive
  `input` convention; a provider-supplied total still wins). Cited by
  OV-4, OV-5. Known divergences from upstream, accepted and documented:
  - the fallback adds `cache_write` on top of `input`; for a provider that
    already reports cache-write tokens *inside* `prompt_tokens`, the
    fallback overcounts by that share (upstream computes from its
    cache-exclusive components and cannot). No vector at the pin exercises
    a nonzero `cache_write`;
  - grain lets a nonzero provider-supplied total win, whereas upstream's
    openai-completions path *always* computes
    `totalTokens = input + output + cacheRead + cacheWrite` and ignores
    any wire total. A provider whose wire total disagrees with its
    components will differ across the seam.
- **AB-R2 (closed on rebase)** — `usage.reasoning` was dropped: the count
  crosses genai (`completion_tokens_details.reasoning_tokens`; the Gemini
  adapter even normalizes `thoughtsTokenCount` into it) but grain's
  `Usage` had no field at the time of measurement. The WP4 merge added
  `Usage::reasoning: Option<u64>`; now wired through in `map_usage`
  (unit-tested in `tests/inbound.rs`). Residual nuance: genai's
  `zero_as_none` turns a wire `reasoning_tokens: 0` into `None`, so
  upstream's `Some(0)` is not reproducible. Not asserted by any vector at
  the pin (fixtures carry 0).

**Reported, not fixed (requires a grain-agent-core type change, out of
adapter scope):**

- **AB-R1** — `rawStopReason` is dropped. The raw provider string *does*
  cross genai (every `genai::chat::StopReason` variant wraps it:
  `Completed("stop")`, `ContentFilter("SAFETY")`,
  `Other("MALFORMED_FUNCTION_CALL")`, …) but grain's `AssistantMessage`
  has no `raw_stop_reason` field to carry it. Adding one is a public
  core-type change (every struct literal in the workspace + the pi-compat
  serialization surface). Upstream asserts it on OV-1, OV-2, AV-2, AV-3,
  GV-1, GV-2.

## 6. Structural gap list

Exact information gaps at the genai seam. **This list defines the measured
scope for a potential pi-ai-shaped `StreamFn` backend** (product decision
out of WP5 scope; this is the measurement).

- **S-1 — per-event usage.** Upstream partials carry running usage from
  `message_start` / per-chunk `usage` onward; genai only surfaces usage on
  `ChatStreamEvent::End(StreamEnd::captured_usage)`. genai would need to
  attach usage to stream events (or emit an early usage event) for grain
  partials to match upstream partials. (This is why partial-usage is
  excluded from per-event comparison rather than failing every vector.)
- **S-2 — responseId.** Upstream captures the provider message/chunk id
  on every message. genai's `InterStreamEnd.captured_response_id` field
  exists but all three streamers (anthropic, openai, gemini) hard-code it
  to `None` in 0.6.5; grain's `AssistantMessage` also has no field. genai
  would need to populate `captured_response_id` (and grain grow a slot).
- **S-3 — Anthropic usage accumulation defect.** genai's anthropic
  streamer *adds* `message_delta.usage.input_tokens` onto the
  `message_start` count (`capture_usage`: `*val += input_tokens`); the
  real API (and upstream pi-ai) treat message_delta usage as a per-field
  *replacement*. Any anthropic stream whose message_delta repeats
  `input_tokens` — which the live API does — reaches grain with inflated
  input/total (measured: 24/24 vs 12/12 on AV-3; 24/29 vs 12/17 on AV-5).
  genai would need replace semantics. [AV-2, AV-3, AV-5]
- **S-4 — Anthropic `stop_details` dropped.** genai captures only
  `delta.stop_reason`; `delta.stop_details` (refusal category +
  explanation, surfaced by upstream as `errorMessage`) is never parsed.
  genai would need to parse and expose it on `StreamEnd`. [AV-2]
- **S-5 — no malformed-JSON repair.** genai serde-parses every SSE frame
  and every accumulated tool-argument buffer; a malformed frame aborts the
  stream with `Error::StreamParse`. Upstream repairs both malformed SSE
  JSON and malformed streamed tool JSON (`parseStreamingJson` et al.) and
  completes the turn. genai would need equivalent lenient parsing. [AV-1]
- **S-6 — `chunk.model` / responseModel not captured.** genai's openai
  streamer never reads `chunk.model`, so routed model ids (OpenRouter
  `auto` etc.) cannot be surfaced. genai would need e.g.
  `captured_response_model` on `StreamEnd` (and grain a field). [OV-3]
- **S-7 — `delta.reasoning_details` not captured.** genai's openai
  streamer reads only `delta.content` / `delta.reasoning_content` /
  `delta.reasoning`; OpenRouter-style encrypted reasoning details (which
  upstream attaches to the matching toolCall as `thoughtSignature` for
  replay) are dropped. genai would need to surface them on tool-call
  chunks — and grain's `ToolCall` needs a signature slot. [OV-6]
- **S-8 — block boundaries are synthesized.** genai has no block
  open/close events (Anthropic `content_block_start/stop` are consumed
  internally); grain infers block transitions from event-kind changes and
  closes the open block at `End`. Event *order* matched upstream on every
  vector here, but `*_end` timing shifts to the next block start / stream
  end, and a provider block opened and closed without any delta would
  vanish entirely. Not exercised as a failure by the fixtures at the pin;
  recorded for completeness.

## 7. Upstream fixtures at the pin that were not used

Stream-parsing scope was anthropic SSE, google/gemini, and
openai-completions (the WP minimum); within that scope every recorded
stream-parsing fixture at the pin is used. Not used, and why:

- `anthropic-eager-tool-input-compat.test.ts`,
  `anthropic-empty-thinking-signature-compat.test.ts`,
  `anthropic-cache-write-1h-cost.test.ts`, `anthropic-auth-token.test.ts`,
  `openai-completions-{prompt-cache,cache-control-format,tool-choice,empty-tools,tool-result-images}.test.ts`,
  `google-shared-*.test.ts`, `google-thinking-signature.test.ts` —
  request-side assertions (payload/header construction) or pure helper
  unit tests; no expected *event* sequence to conform to. Outbound-mapping
  fidelity is WP3 territory (`tests/outbound.rs`).
- `openai-completions-thinking-as-text.test.ts` (third case) — serves real
  SSE, but its assertion target is the request body (thinking replayed as
  text parts); the stream leg is a trivial `ok` + stop already covered by
  OV-4/OV-5.
- `openai-completions-retry.test.ts` — asserts pi's transport retry
  policy; grain delegates transport to genai's client, which has no
  equivalent per-request retry hook at this seam (already tracked as the
  `max_retry_delay_ms` gap in `stream.rs` docs).
- `openai-codex-stream.test.ts`, `openai-responses-*.test.ts`,
  `azure-openai-responses-*.test.ts` — OpenAI **Responses** API event
  protocol. grain-llm-genai routes no models to genai's `OpenAIResp`
  adapter today; out of the WP5 minimum scope.
- `mistral-raw-stop-reason.test.ts`, `cloudflare-stream.test.ts` —
  providers outside the WP5 minimum, and genai 0.6.5 has no Mistral or
  Cloudflare adapter at all.
- `bedrock-raw-stop-reason.test.ts` — provider outside the WP5 minimum.
  genai 0.6.5 *does* have Bedrock Converse adapters
  (`AdapterKind::BedrockApi` / `BedrockSigv4`), but grain-llm-genai routes
  no models to them today, and the upstream fixture mocks the AWS SDK's
  Converse-stream object layer (`messageStart`/`messageStop` events), not
  recorded wire data.
- `xai-responses.test.ts` — xAI via the OpenAI **Responses** API
  (`response.completed` events): same Responses-protocol exclusion as
  `openai-responses-*` above, and outside the WP5 minimum.
- `stream.test.ts`, `text.test.ts`, `interleaved-thinking.test.ts`,
  `unicode-surrogate.test.ts`, `tool-call-id-normalization.test.ts`,
  `overflow.test.ts`, `abort.test.ts`, cross-provider handoff/e2e suites —
  live tests against real provider credentials, no recorded wire data.
- `error-body.test.ts`, `provider-error-body-*.test.ts` — unit tests of
  pi's SDK-error normalizer (SDK-shaped error objects, not wire data).

## 8. Running

```bash
# PASS vectors (run in the default suite; must stay green):
cargo test -p grain-llm-genai --test seam_vectors

# STRUCTURAL vectors (intentionally red — the red IS the measurement):
cargo test -p grain-llm-genai --test seam_vectors -- --ignored
```
