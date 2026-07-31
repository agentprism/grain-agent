# WP5 — Seam Conformance Vectors

Measurement of whether grain-llm-genai's event stream is faithful to
upstream pi-ai, using **upstream's own stream-parsing fixtures** as the
standard.

- Upstream reference: `pi/packages/ai` (TypeScript), pinned commit
  `34239180ac5c80366def592b529a3a1b882b4a16` (read-only).
- genai version under measurement: **0.6.5** (workspace pin).
- Harness: `tests/seam_vectors.rs`.

**WP32 status (this revision).** **AB-R1 is closed.** The default genai path
now assigns the provider's verbatim stop string into
`AssistantMessage::raw_stop_reason` (`mapping::inbound::on_end`, from
`StopReason::raw()` — the raw string was always preserved inside every genai
variant; only the assignment was missing). Every vector's terminal comparison
now asserts the field, which turns upstream's previously-excluded
`rawStopReason` legs (OV-1, OV-2, GV-1, GV-2, AV-2, AV-3) into pinned
behavior. **No count moves: 11 PASS / 2 STRUCTURAL stands** — the six vectors
that gained the assertion were already PASS on their representable remainder;
they are now PASS with nothing excluded. The two STRUCTURAL vectors (OV-3,
OV-6) are blocked by genai for other fields and are unaffected.

**WP25 status (prior revision).** The Anthropic vectors now run through the
**native Anthropic transport** (`src/anthropic/`), a second `LlmStream`
behind the same seam, built because the WP21 measurement below showed the gaps
unreachable from `ChatStreamEvent` and the one alternative route (an
endpoint-tee relay, costed in §6) more expensive than owning the transport. **AV-1, AV-2, AV-3 and AV-5 are un-ignored and pass**;
counts moved from 7 PASS / 6 STRUCTURAL to **11 PASS / 2 STRUCTURAL**. No
vector was weakened or deleted — the upstream expectations are unchanged, a
better backend now meets them. genai remains the **default** backend and its
behavior on these same fixtures is still pinned, directly at its own seam, by
`tests/genai_seam_limits.rs`. The native transport is verified against
recorded fixtures only, **never against the live Anthropic API** — see
`src/anthropic/stream.rs`.

**WP21 status (prior revision).** Every structural gap in §6 was re-examined
against the question "is this closable in `grain-llm-genai`, from what genai's
streaming API delivers?" The answer is **no for all eight** — with one partial exception,
the tool-argument half of S-5, which was closed (adapter fix AB-3, §5). The
blocker is a single architectural fact, not eight separate ones: genai's
streaming seam is `ChatStreamEvent`, and everything the provider sent beyond
those variants is consumed inside the streamer with no public carrier and no
extension point below it. §6 documents this once and annotates each gap with
what specifically is destroyed and where; `tests/genai_seam_limits.rs` proves
it empirically against genai's real client. **No previously-`#[ignore]`d
vector became passable, so the counts in §5 are unchanged (7 PASS / 6
STRUCTURAL).** The S-3 defect is written up upstream-ready in
`../UPSTREAM-GENAI.md` (unfiled; still unfixed at genai `0.7.0-beta.15`).

## 1. Chain under test

```
recorded provider SSE (upstream fixture, frame-faithful)
  → local mock HTTP endpoint (tests/seam_vectors.rs::serve_sse_once)
  → genai 0.6.5 real client + provider streamer   (production code)
  → genai ChatStreamEvents                        (the seam)
  → grain-llm-genai inbound adapter (GenaiStream) (production code)
  → grain AssistantMessageEvents                  (asserted)
```

Since WP25 the **anthropic** vectors (AV-*) run the other backend instead —
same fixture, same socket, same assertions, different transport:

```
recorded provider SSE (upstream fixture, frame-faithful)
  → local mock HTTP endpoint (tests/seam_vectors.rs::serve_sse_once)
  → grain-llm-genai native Anthropic transport    (production code)
      src/anthropic/{wire,state,request}.rs
  → grain AssistantMessageEvents                  (asserted)
```

genai is still the default backend at runtime; its behavior on these same
Anthropic fixtures is pinned separately, and directly at its own seam, by
`tests/genai_seam_limits.rs`. So routing the vectors to the better backend
adds a measurement rather than replacing one.

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
| `content[]` (text/thinking/toolCall) | `content: Vec<AssistantContent>` | mapped; `toolCall.thoughtSignature` now HAS a grain slot (`ToolCall::thought_signature`, WP19) but no producer on either path — genai drops `reasoning_details` (S-7), and the Anthropic wire carries no signature on `tool_use` at all |
| `stopReason` stop/length/toolUse/error/aborted | `StopReason` Stop/Length/ToolUse/Error/Aborted | mapped; upstream's mid-stream `"pending"` placeholder has no grain variant — grain partials carry the default `Stop` (cosmetic, excluded from comparison) |
| `rawStopReason` | `raw_stop_reason` | mapped (**AB-R1 closed**, WP32): the verbatim provider string genai preserves inside every `StopReason` variant is assigned at `on_end`; asserted on every vector's terminal |
| `errorMessage` | `error_message` | mapped |
| `usage.input` (cache-**exclusive**) | `usage.input` (cache-**inclusive**, see `Cost::cost_for`) | convention differs; numerically equal whenever cache counters are 0 (all vectors here) |
| `usage.output/cacheRead/cacheWrite/totalTokens` | `output/cache_read/cache_write/total_tokens` | mapped (total via AB-2 fallback when the wire omits it; see AB-2 divergence notes in §5) |
| `usage.reasoning` | `usage.reasoning: Option<u64>` | mapped (**AB-R2 closed** after the WP4 rebase gave grain the field). Nuance: genai's `zero_as_none` deserialization turns a wire `reasoning_tokens: 0` into `None`, so upstream's `Some(0)` for openai-completions/google is not reproducible — wire zero and absent field are indistinguishable at this seam |
| `responseId` | `response_id` | **S-2 closed for the native transport** (WP27); still unreachable on the genai path, which never surfaces it |
| `responseModel` | `response_model` | **S-6 closed for the native transport** (WP27); still unreachable on the genai path (genai never reads `chunk.model`) — see OV-3 |
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
`captured_pause_turn_is_resubmittable_stop`). The verbatim raw string is
assigned separately from this mapping (WP32,
`InboundState::on_end` → `AssistantMessage::raw_stop_reason`) and pinned
by `captured_stop_reason_raw_string_is_preserved_verbatim`,
`error_class_stop_reasons_also_carry_the_raw_string`,
`missing_captured_stop_reason_leaves_raw_none`, and
`locally_synthesized_terminals_carry_no_raw_stop_reason` in the same file.

## 3. Comparison scope

Asserted per vector: exact event-kind sequence, `content_index` per event,
`delta` payloads verbatim, and the full terminal message (content blocks,
stop reason, raw provider stop string, error message, usage incl. cost
struct).

Excluded and tracked as named gaps instead (so one cross-cutting gap does
not smear every vector into STRUCTURAL):

- mid-stream `partial.usage` — upstream partials carry running usage from
  `message_start` / per-chunk usage; genai surfaces usage only at `End`
  (**S-1**, the canonical genai-seam gap);
- mid-stream `partial.stopReason` `"pending"` placeholder (no grain
  variant; cosmetic);
- `timestamp`, `model`/`api`/`provider` echoes;
- `usage.reasoning` (**AB-R2**) — inventoried below with the vectors
  upstream asserts it on. `rawStopReason` (**AB-R1**, closed WP32),
  `responseId` (**S-2**) no longer belong on this list at all for
  `raw_stop_reason` and partially for the rest: `raw_stop_reason` is
  populated on BOTH paths (the genai adapter assigns it from
  `StopReason::raw()`; the native transport from `message_delta`) and every
  vector asserts it; `responseId` has a grain slot (WP19) populated and
  asserted on the native transport (`tests/response_metadata.rs`) but
  remains unassertable on the **genai** path, which never populates
  `captured_response_id`.

STRUCTURAL vectors assert the *upstream-translated* expectation and are
intentionally **red** under `cargo test -- --ignored`; the red is the
measurement. Where the upstream expectation is not even representable in
grain types (OV-3 responseModel, OV-6 thoughtSignature), the vector first
proves the representable translation passes end-to-end, then fails with an
explicit `structural gap` panic naming the unrepresentable field.

## 4. Vector inventory

| Vector | Provider / API | Upstream test (`packages/ai/test/`) | Upstream case | Classification | Reason |
|---|---|---|---|---|---|
| AV-1 | anthropic | `anthropic-sse-parsing.test.ts` | repairs malformed SSE JSON and malformed streamed tool JSON | **PASS** (native transport) | Was S-5. The native transport parses each frame through `mapping::json_repair`, so the invalid `\H` escape and raw TAB are repaired at both the frame and tool-argument layers and the turn completes as upstream does |
| AV-2 | anthropic | `anthropic-sse-parsing.test.ts` | preserves refusal stop details from message_delta | **PASS** (native transport) | Was S-4 + S-3. `delta.stop_details.explanation` is captured and surfaced as `error_message`; usage replaces per field (412/0/412, not 824) |
| AV-3 | anthropic | `anthropic-sse-parsing.test.ts` | preserves sensitive stop reasons with a descriptive error message | **PASS** (native transport) | Was S-3. `message_delta.usage` now replaces rather than adds: 12/0/12, not 24/24 |
| AV-4 | anthropic | `anthropic-sse-parsing.test.ts` | treats message_delta without usage as a no-op for usage accumulation | **PASS** (native transport) | Passed on genai too; on the native path the same green comes from replacement semantics skipping absent fields — the case that makes an unconditional halving of genai's total unsound |
| AV-5 | anthropic | `anthropic-sse-parsing.test.ts` | ignores unknown SSE events after message_stop | **PASS** (native transport) | Was S-3. 12/5/17, not 24/5/29. Trailing junk frames are never read: the transport stops at the terminal |
| OV-1 | openai-completions | `openai-completions-raw-stop-reason.test.ts` | preserves raw finish reasons for successful stops | **PASS** | Events/stop/usage exact, and since WP32 the `rawStopReason === "stop"` leg is asserted too — AB-R1 closed. The slot arrived in WP19, the raw string always crossed genai (preserved inside every `StopReason` variant), and the assignment landed in `mapping::inbound::on_end`. Nothing about this vector is excluded anymore |
| OV-2 | openai-completions | `openai-completions-raw-stop-reason.test.ts` | preserves raw finish reasons for provider error stops | **PASS** | Fixed by AB-1 (was: silent `Done/Stop`). Error event + `Provider finish_reason: content_filter` exact; `rawStopReason === "content_filter"` asserted since WP32 (AB-R1 closed) |
| OV-3 | openai-completions | `openai-completions-response-model.test.ts` | surfaces routed chunk.model on responseModel without changing model | **STRUCTURAL** | S-6: `chunk.model` never crosses genai (grain's `response_model` slot now exists and the native transport fills it — this vector is genai-blocked only). Representable remainder (text events, stop, usage 10/5/15) passes |
| OV-4 | openai-completions | `openai-completions-response-model.test.ts` | leaves responseModel undefined when chunks echo the requested id | **PASS** | Absence semantics vacuously exact; usage total 2 via AB-2 |
| OV-5 | openai-completions | `openai-completions-response-model.test.ts` | ignores empty or missing chunk.model | **PASS** | Two text deltas aggregate; usage total 3 via AB-2 |
| OV-6 | openai-completions (openrouter-flavored) | `openai-completions-reasoning-details.test.ts` | preserves reasoning_details that arrive before their matching tool call | **STRUCTURAL** | S-7: `delta.reasoning_details` never crosses genai; grain `ToolCall` has no signature slot. Representable remainder (toolcall events, args, toolUse stop) passes |
| GV-1 | google-generative-ai | `google-raw-stop-reason.test.ts` | preserves raw Gemini finish reasons for Google Generative AI errors (`MALFORMED_FUNCTION_CALL`) | **PASS** | Fixed by AB-1 (was: silent `Done/Stop`). Error + `Provider stopped with: MALFORMED_FUNCTION_CALL`, usage 1/0/1 exact; `rawStopReason === "MALFORMED_FUNCTION_CALL"` asserted since WP32 (AB-R1 closed) |
| GV-2 | google-vertex | `google-raw-stop-reason.test.ts` | preserves raw Gemini finish reasons for Google Vertex errors (`SAFETY`) | **PASS** | Fixed by AB-1. Upstream drives this through the vertex transport; the wire (GenerateContentResponse SSE) is identical. genai 0.6.5 does have a dedicated Vertex adapter (`AdapterKind::Vertex`) — the collapse onto the gemini wire is grain's routing (`ProviderRouter`: `google → gemini`; no grain models route to the Vertex adapter today) |

## 5. Summary

| Classification | Count | Vectors |
|---|---|---|
| PASS | 11 | AV-1, AV-2, AV-3, AV-4, AV-5, OV-1, OV-2, OV-4, OV-5, GV-1, GV-2 |
| STRUCTURAL | 2 | OV-3, OV-6 |

Movement since WP21 (7 PASS / 6 STRUCTURAL): **AV-1, AV-2, AV-3, AV-5** moved
STRUCTURAL → PASS when the Anthropic vectors were routed through the native
transport (WP25). WP32 moved no vector — it widened what PASS means: the
terminal comparison now also asserts `raw_stop_reason`, so the six upstream
`rawStopReason` legs previously excluded as AB-R1 are pinned inside the
existing greens. The two remaining STRUCTURAL vectors are openai-completions
(OV-3 responseModel, OV-6 reasoning_details); their grain-side slots exist
(WP19), so both are blocked solely on genai surfacing the value at the seam —
not reachable from any backend or adapter work here.
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
- **AB-3 (WP21)** — malformed tool-argument JSON dropped the whole tool
  call. `parse_tool_args` used a bare `serde_json::from_str` and stored the
  raw text as `Value::String` on failure — which is exactly the shape the
  outbound corrupt-args guard (`collect_corrupt_tool_call_ids`) treats as
  corrupt, so one unescaped TAB or one stray `\H` anywhere in a tool's
  arguments silently cost the user the entire tool call *and* its paired
  `tool_result`, with no error surfaced. Fixed in
  `src/mapping/json_repair.rs`: a statement-for-statement port of upstream's
  `repairJson` / `parseJsonWithRepair`
  (`packages/ai/src/utils/json-parse.ts`), wired into `parse_tool_args`.
  This is the **adapter-reachable half of S-5** — the accumulated buffer
  does cross the seam (`ToolCallChunk.fn_arguments` as `Value::String`;
  genai's OpenAI streamer keeps the string verbatim when its own parse
  fails). It does **not** un-ignore AV-1, whose failure is at the SSE-frame
  layer inside genai (see §6 S-5). Proven by `tests/tool_arg_repair.rs`
  (round trip: repaired call survives into the next request) plus 15 unit
  tests including upstream's exact AV-1 argument payload.
  Deliberate divergence, accepted and documented: upstream's
  truncation-tolerant tier (`parseStreamingJson` → the `partial-json`
  package) is **not** ported. Coercing a cut-off buffer into a partial
  object would replay a tool call with silently truncated arguments — a
  file-write whose `content` lost its tail — where grain currently drops it.
  No vector at the pin forces that, so the safer behavior is kept; a
  truncated buffer stays `Value::String` and is still dropped.
- **AB-R2 (closed on rebase)** — `usage.reasoning` was dropped: the count
  crosses genai (`completion_tokens_details.reasoning_tokens`; the Gemini
  adapter even normalizes `thoughtsTokenCount` into it) but grain's
  `Usage` had no field at the time of measurement. The WP4 merge added
  `Usage::reasoning: Option<u64>`; now wired through in `map_usage`
  (unit-tested in `tests/inbound.rs`). Residual nuance: genai's
  `zero_as_none` turns a wire `reasoning_tokens: 0` into `None`, so
  upstream's `Some(0)` is not reproducible. Not asserted by any vector at
  the pin (fixtures carry 0).
- **AB-R1 (closed, WP32)** — `rawStopReason` was dropped. The raw provider
  string always crossed genai (every `genai::chat::StopReason` variant
  wraps it: `Completed("stop")`, `ContentFilter("SAFETY")`,
  `Other("MALFORMED_FUNCTION_CALL")`, …) but at measurement time grain's
  `AssistantMessage` had no field to carry it — a public core-type change,
  reported rather than made. WP19 added the slot
  (`AssistantMessage::raw_stop_reason`, rust-host ledger 13) and the
  native Anthropic transport populated it for its provider (WP27); WP32
  completed the picture by assigning `StopReason::raw()` in
  `mapping::inbound::on_end` on the DEFAULT genai path, verbatim and
  before normalization, exactly where upstream assigns it
  (`openai-completions.ts:459`, `google-generative-ai.ts:215`,
  `google-vertex.ts:232`, `anthropic-messages.ts:709`). Locally
  synthesized terminals (abort, transport error, missing captured reason)
  stay `None`, matching upstream's undefined. Asserted on every vector's
  terminal here, and in isolation by four unit tests in
  `tests/inbound.rs`. Upstream asserts it on OV-1, OV-2, AV-2, AV-3,
  GV-1, GV-2 — all six legs are now pinned.

No reported-not-fixed adapter bugs remain at this revision.

## 6. Structural gap list

Exact information gaps at the genai seam. **This list defines the measured
scope for a potential pi-ai-shaped `StreamFn` backend** (product decision
out of WP5 scope; this is the measurement).

### WP21 verdict: reachable-from-the-seam vs reachable-at-any-cost

WP21 re-examined every gap below against the question "can this be closed in
`grain-llm-genai`, from what genai's streaming API delivers?" The answer is
**no for all eight**, for one shared reason:

> `genai::chat::ChatStreamEvent` is the entire seam. Its only payload-bearing
> terminal is `End(StreamEnd)`, whose fields are `captured_usage`,
> `captured_stop_reason`, `captured_content`, `captured_reasoning_content`,
> `captured_response_id`. `ChatStreamResponse` carries only `stream` and
> `model_iden`. `ChatOptions::capture_raw_body` exists but **no streaming
> path consults it** — it is honored only on the non-streaming and embedding
> paths. Everything else the provider sent is consumed inside the streamer
> and has no representation at all.

There is likewise no extension point that lets us *decode* the response
differently in place: `genai::webc` exports only `Error`
(`EventSourceStream`, `WebStream`, `WebClient` are all `pub(crate)`),
`genai::adapter` exports only `AdapterKind` (the adapter modules are
`pub(super)`, so `AnthropicStreamer` is unreachable and no custom adapter can
be registered), and `ClientBuilder::with_reqwest` accepts a bare
`reqwest::Client`, which has no response-body middleware hook.

#### The endpoint-tee relay: possible, costed, rejected

What genai *does* expose is control over **where it connects**. The full set
of public hooks that can redirect a request is larger than a first pass
suggests:

- `ClientBuilder::with_service_target_resolver_fn` → `Endpoint::from_owned`
  (this crate already installs one in production, `builder.rs:286-289`);
- `ClientBuilder::with_web_config` → `WebConfig::with_proxy` / the public
  `proxy` field;
- `AuthData::RequestOverride { url, headers }` — a public variant with public
  fields that overrides the request URL (`client_impl.rs:188-195`);
- `ModelSpec::Target`.

So a consumer *can* recover the truth without forking genai or patching
`Cargo.toml`: interpose a recording relay between genai and the provider, tee
the SSE body, and re-derive usage from the raw `message_start` /
`message_delta` frames. This is demonstrated working, and it is not exotic —
genai's own test suite ships such a proxy
(`tests/support/yakbak/server.rs`), and this repository already uses the same
redirect mechanism in `tests/genai_seam_limits.rs` and in
`../UPSTREAM-GENAI.md` §1.4.

**We reject it on cost, not on possibility.** Re-deriving usage from the wire
means re-implementing Anthropic's SSE parsing on our side of the relay — which
is most of a provider backend already — and then maintaining it in lockstep
with genai's, while paying a proxy hop, a second socket, and a place for the
API key to transit on every request. Having paid for the SSE parser, keeping
genai in the path buys nothing: the honest version of that design is simply to
own the transport. So the conclusion the gap list needs is not "impossible"
but a cost judgement: **closing any of these means owning the provider
transport — the pi-ai-shaped-backend decision — because every cheaper route
ends up there anyway.**

Each gap below is annotated with what specifically is unavailable and where.
Where a gap says "not reachable from `ChatStreamEvent`", read it as a
statement about the streaming API's surface, **not** a claim that no
adapter-side mechanism exists at any cost.

Evidence: `tests/genai_seam_limits.rs` drives genai's real Anthropic client
against recorded wire and asserts genai's current behavior directly at the
seam (no grain code in between). Those tests are tripwires — each fails
loudly if a genai bump changes the seam, which is the signal to re-measure
the gap and un-`#[ignore]` its vector.

The one exception is the **tool-argument half of S-5**, which *is* reachable
because genai lets the accumulated buffer cross as a `Value::String`. It was
closed by adapter fix AB-3 (§5). It does not un-ignore AV-1.

### WP25 resolution: what the native Anthropic transport closed

The verdicts above stand — they are statements about what is reachable *above
genai*, and they remain true. WP25 acted on them by owning the transport for
one provider, which is the only lever they leave. For Anthropic models served
by `src/anthropic/` (opt-in; genai is still the default):

| gap | status on the native transport |
|---|---|
| S-1 per-event usage | **closed** — partials carry running usage from `message_start`, matching upstream's observable behavior. Not vector-asserted: partial usage is excluded from comparison scope (§3) |
| S-3 usage double-count | **closed** — `UsageAccumulator` replaces per field [AV-2, AV-3, AV-5] |
| S-4 `stop_details` | **closed** — explanation → `error_message` [AV-2] |
| S-5 malformed repair | **closed**, both halves — frames and tool arguments [AV-1] |
| S-8 block boundaries | **closed** — driven by `content_block_start`/`stop`; an empty block now emits its `*_start`/`*_end` pair |
| S-2 `responseId` | **closed** (WP27) — captured from `message_start.message.id` and carried on `AssistantMessage::response_id`, first-write-wins [`tests/response_metadata.rs`] |
| S-6 `responseModel` | **closed** (WP27) — captured from `message_start.message.model` and carried on `AssistantMessage::response_model`, reported ONLY when it differs from the name actually sent on the wire (upstream's rule, `openai-completions.ts:442-444`) [`tests/response_metadata.rs`] |
| S-7 `reasoning_details` | **out of reach** — openai-completions only, and needs a genai change. The core half is now closed (`ToolCall::thought_signature`, WP19); not applicable to this transport, since the Anthropic wire carries no signature on `tool_use` |

Only ONE row is now unclosed, and it is a different provider family
(openai-completions), not an Anthropic-wire gap. S-2 and S-6 closed in WP27:
WP19 supplied the core slots the earlier "blocked on core" verdict was waiting
for, and the native transport — which parses the Anthropic stream itself and
therefore actually holds the values — now carries them. Everything the
Anthropic wire actually carries reaches the loop.

- **S-1 — per-event usage.** Upstream partials carry running usage from
  `message_start` / per-chunk `usage` onward; genai only surfaces usage on
  `ChatStreamEvent::End(StreamEnd::captured_usage)`. genai would need to
  attach usage to stream events (or emit an early usage event) for grain
  partials to match upstream partials. (This is why partial-usage is
  excluded from per-event comparison rather than failing every vector.)
  **WP21 verdict — not reachable from `ChatStreamEvent`.** `End` is the
  only `ChatStreamEvent` variant with a usage payload; nothing earlier
  carries one, so grain partials cannot mirror upstream's running usage
  (upstream sets it while handling `message_start`, before any content event
  is emitted). Pinned by
  `tests/genai_seam_limits.rs::s1_usage_crosses_only_at_end`. Note the
  upstream shape is subtler than "per-event": pi-ai's `partial` is a *live
  reference* to one mutating object, mutated only at `message_start` and
  `message_delta`, so a live consumer observes message_start-era usage on
  every content event and final usage on `done`/`error`.
- **S-2 — responseId.** Upstream captures the provider message/chunk id
  on every message. genai's `InterStreamEnd.captured_response_id` field
  exists but all three streamers (anthropic, openai, gemini) hard-code it
  to `None` in 0.6.5; grain's `AssistantMessage` also has no field. genai
  would need to populate `captured_response_id`. **The grain half is
  closed** (`AssistantMessage::response_id`, WP19; populated by the native
  transport in WP27) — what remains here is genai-only, and applies to the
  genai path alone.
  **WP21 verdict — not reachable from `ChatStreamEvent`, and doubly
  blocked.** genai never populates the field (`streamer.rs:234` sets
  `captured_response_id: None` although `message_start.message.id` was
  parsed moments earlier), *and* grain-agent-core has no `response_id` slot
  to carry it — a core type change that is not WP21's to make and is not in
  WP19's scope either. Pinned by
  `tests/genai_seam_limits.rs::s2_response_id_never_crosses`.
- **S-3 — Anthropic usage accumulation defect.** genai's anthropic
  streamer *adds* `message_delta.usage.input_tokens` onto the
  `message_start` count (`capture_usage`: `*val += input_tokens`); the
  real API (and upstream pi-ai) treat message_delta usage as a per-field
  *replacement*. Any anthropic stream whose message_delta repeats
  `input_tokens` — which the live API does — reaches grain with inflated
  input/total (measured: 24/24 vs 12/12 on AV-3; 24/29 vs 12/17 on AV-5).
  genai would need replace semantics. [AV-2, AV-3, AV-5]

  **WP21 verdict — not reachable from `ChatStreamEvent`, and provably so.**
  Not merely awkward: the corrective term is *destroyed*, not just hidden.
  `capture_usage` folds `message_start` and `message_delta` into one
  accumulator before anything is observable, so these two streams are
  byte-identical at `StreamEnd`:

  | wire | provider truth | genai `StreamEnd` |
  |---|---|---|
  | `message_start` 12, `message_delta` repeats 12 | 12 | 24 |
  | `message_start` 24, `message_delta` has no `usage` | 24 | 24 |

  Any correction applied to `StreamEnd` is a function of what crosses, so it
  must return the *same* answer for both — and exactly one of those answers
  is then wrong. Halving unconditionally would fix the first and corrupt the
  second, and the second is not hypothetical: it is upstream's own
  "message_delta without usage" case, pinned green today as **AV-4**. There
  is no discriminator anywhere else on the seam either — `prompt_tokens_details`
  is populated from `message_start` only, `total_tokens` is recomputed from
  the already-inflated components at `message_stop`
  (`streamer.rs:217`), and `capture_raw_body` is inert while streaming.

  Pinned empirically by
  `tests/genai_seam_limits.rs::s3_double_count_is_indistinguishable_at_the_seam`,
  which runs both streams through genai's real client and asserts the
  outputs are equal — a statement about `StreamEnd`, not about the adapter as
  a whole: the endpoint-tee relay described above *does* recover the true 12
  vs 24, and is rejected on cost rather than feasibility. Reported
  upstream-ready in `UPSTREAM-GENAI.md` §1
  (still unfixed at genai `0.7.0-beta.15`). Closing it requires replace
  semantics inside the streamer, or owning the Anthropic transport.
- **S-4 — Anthropic `stop_details` dropped.** genai captures only
  `delta.stop_reason`; `delta.stop_details` (refusal category +
  explanation, surfaced by upstream as `errorMessage`) is never parsed.
  genai would need to parse and expose it on `StreamEnd`. [AV-2]
  **WP21 verdict — not reachable from `ChatStreamEvent`.** Only the bare
  reason string survives, inside `StopReason::Other("refusal")`; the
  explanation and category appear nowhere on `StreamEnd`. Pinned by
  `tests/genai_seam_limits.rs::s4_stop_details_never_crosses`, which
  serializes the whole terminal payload and asserts neither string occurs in
  it. grain *does* have the destination slot (`error_message`), so unlike
  S-7 this one is blocked solely by genai (as, since WP27, are S-2 and
  S-6 — their core halves are closed). AV-2 additionally needs
  S-3, so it stays ignored regardless.
- **S-5 — no malformed-JSON repair.** genai serde-parses every SSE frame
  and every accumulated tool-argument buffer; a malformed frame aborts the
  stream with `Error::StreamParse`. Upstream repairs both malformed SSE
  JSON and malformed streamed tool JSON (`parseStreamingJson` et al.) and
  completes the turn. genai would need equivalent lenient parsing. [AV-1]
  **WP21 verdict — split.** The *tool-argument* half **is** adapter-reachable
  and was closed (adapter fix AB-3, §5): the accumulated buffer crosses the
  seam as `ToolCallChunk.fn_arguments`, so upstream's repair can be applied
  above genai. The *SSE-frame* half is not reachable from the seam: genai
  `serde_json::from_str`s each frame inside its own `poll_next` and returns
  `Err(Error::StreamParse)`, so the malformed bytes never reach us and the
  stream is already dead when we learn of it. AV-1's fixture fails at the
  frame layer, so it stays ignored even though its inner payload now
  repairs correctly (proven in `tests/tool_arg_repair.rs`).
- **S-6 — `chunk.model` / responseModel not captured.** genai's openai
  streamer never reads `chunk.model`, so routed model ids (OpenRouter
  `auto` etc.) cannot be surfaced. genai would need e.g.
  `captured_response_model` on `StreamEnd`. [OV-3]
  **WP21 verdict — not reachable from `ChatStreamEvent`, and doubly
  blocked.** **WP27 update: no longer doubly blocked.** The grain half is
  closed (`AssistantMessage::response_model`, WP19; populated by the native
  transport, which reads `message_start.message.model` directly). The
  remaining blocker is genai-only, so OV-3 stays red on the genai path.
- **S-7 — `delta.reasoning_details` not captured.** genai's openai
  streamer reads only `delta.content` / `delta.reasoning_content` /
  `delta.reasoning`; OpenRouter-style encrypted reasoning details (which
  upstream attaches to the matching toolCall as `thoughtSignature` for
  replay) are dropped. genai would need to surface them on tool-call
  chunks — and grain's `ToolCall` needs a signature slot. [OV-6]
  **WP21 verdict — not reachable from `ChatStreamEvent`, and doubly
  blocked.** genai reads only `delta.content` / `delta.reasoning_content` /
  `delta.reasoning`, so `delta.reasoning_details` never crosses. WP19 is
  adding `ToolCall.thought_signature` to core, which closes the *grain* half
  — but even with that field landed the value to put in it does not exist at
  this seam, so OV-6 stays ignored until genai surfaces the details. See the
  WP21 report for the exact wiring that remains.
- **S-8 — block boundaries are synthesized.** genai has no block
  open/close events (Anthropic `content_block_start/stop` are consumed
  internally); grain infers block transitions from event-kind changes and
  closes the open block at `End`. Event *order* matched upstream on every
  vector here, but `*_end` timing shifts to the next block start / stream
  end, and a provider block opened and closed without any delta would
  vanish entirely. Not exercised as a failure by the fixtures at the pin;
  recorded for completeness.
  **WP21 verdict — not reachable from `ChatStreamEvent`.** Confirmed
  sharper than "recorded for completeness": a text block opened and closed
  with no delta produces *no seam event at all* — the whole stream reduces
  to `Start, End` — so the adapter cannot synthesize the `text_start` /
  `text_end` pair upstream emits. Pinned by
  `tests/genai_seam_limits.rs::s8_empty_block_vanishes_at_the_seam`.
- **S-9 — Anthropic `error` events and thought signatures dropped at the
  genai seam (WP25 audit; not exercised by any vector at the pin).** Two
  defects found while auditing genai for the transport work, both in
  `adapter/adapters/anthropic/streamer.rs`:
  - **Mid-stream provider errors vanish.** The event match ends with
    `other => tracing::warn!("UNKNOWN MESSAGE TYPE: {other}")` (line 242).
    Anthropic's documented event set includes `error`, emitted mid-stream for
    `overloaded_error` / `api_error`. It matches no arm, so it is logged and
    discarded; the streamer polls to end-of-body, never sees `message_stop`,
    and returns `Poll::Ready(None)`. Our adapter then reports
    `"stream ended without terminal event"` — so a user hitting an overloaded
    provider sees a generic truncation message instead of the provider's
    actual, actionable error, with no way to distinguish the two.
    **This one affects anyone on the genai default path today.**
  - **`captured_thought_signatures` is hard-coded `None`** (line 233), even
    though the same streamer emits `ThoughtSignatureChunk` for
    `signature_delta`. Signatures stream chunk by chunk but are absent from
    the terminal aggregate, unlike text / reasoning / tool calls. genai's own
    `into_assistant_message_for_tool_use()` therefore silently loses the
    signatures Anthropic requires on replay. The Gemini path populates the
    field, so this reads as an oversight rather than an asymmetry by design.
    grain is insulated: our inbound state machine attaches signatures from the
    chunk events (`mapping::inbound::attach_thought_signature`) rather than
    from the aggregate, and the native transport reads `signature_delta`
    directly.

  Both are reported in `../UPSTREAM-GENAI.md` §2.6 / §2.7. Neither is covered
  by an upstream fixture at the pin, so neither carries a seam vector — they
  are recorded here because the `error`-event half is a live user-facing
  defect on the default path, not a measurement artifact.

  **Closed on the native Anthropic transport (WP25):** `error` frames
  terminate the turn carrying the provider's own message
  (`state.rs`, pinned by `provider_error_event_terminates_with_its_message`),
  and `signature_delta` is read directly into the thinking block, so neither
  defect applies to opted-in callers. S-9 remains open for the genai default
  path, which is where most callers are.

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

# WP21: what genai does and does not deliver at the seam. These consume
# genai::chat::ChatStreamEvent directly (no grain code in between) and pin
# genai's current, defective behavior — they are the evidence behind the
# §6 verdicts and the tripwires that fire when a genai bump changes it:
cargo test -p grain-llm-genai --test genai_seam_limits

# WP21: the adapter-reachable half of S-5 (adapter fix AB-3), end to end:
cargo test -p grain-llm-genai --test tool_arg_repair

# WP25: the native Anthropic transport (request shape, SSE framing, event
# state machine, usage replacement, stop mapping):
cargo test -p grain-llm-genai --lib anthropic::
```

### A note on what "green" proves here

Every assertion in this suite replays a **recorded** fixture from a local
socket. For the genai backend that is the whole story: genai's request
building is exercised by the same production code paths that run live. For the
**native Anthropic transport it is not** — fixture parity proves the decode
path reproduces upstream pi-ai exactly, and proves nothing about whether the
live Anthropic API accepts the request that transport builds. That is why the
native transport is opt-in and genai remains the default. Live verification is
scheduled separately.

### When a genai bump changes the seam

`tests/genai_seam_limits.rs` fails rather than silently passing when genai
starts delivering something it previously dropped. That failure is the
signal to (a) re-measure the affected gap in §6, (b) un-`#[ignore]` its
vector in `seam_vectors.rs`, and (c) update the counts in §5. Do not
"repair" those tests by relaxing them — like the STRUCTURAL vectors, their
red is the measurement.
