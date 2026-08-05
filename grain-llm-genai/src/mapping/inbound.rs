//! Inbound: turn `genai::chat::ChatStreamEvent` into
//! `grain_agent_core::AssistantMessageEvent`.
//!
//! Modeled as a pure state machine: each genai event mutates the partial
//! [`AssistantMessage`] and returns zero or more grain events. No I/O.
//! Tested in isolation by feeding a hand-crafted sequence of events
//! (see `tests/inbound.rs`).

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use genai::chat::{ChatStreamEvent, StreamEnd, ToolCall as GenaiToolCall};
use grain_agent_core::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, Model, StopReason, TextContent,
    ThinkingContent, ToolCall as GrainToolCall, Usage,
};

use crate::mapping::usage::map_usage;

/// Streaming state for one assistant turn.
///
/// The state machine emits well-formed grain events (matching the contract in
/// `grain-agent-core::stream`): exactly one [`AssistantMessageEvent::Start`]
/// followed by Text/Thinking/Toolcall block events, terminated by exactly one
/// [`AssistantMessageEvent::Done`] or [`AssistantMessageEvent::Error`].
///
/// **Tool-call delta synthesis (WP3, realizing design note M-3)**: genai 0.6.5
/// has no argument-delta event — `ChatStreamEvent::ToolCallChunk` always
/// carries a complete `ToolCall`, and chunks for the same `call_id` repeat
/// with *cumulatively growing* `fn_arguments` (verified against genai 0.6.5:
/// `adapter/adapters/openai/streamer.rs` `capture_tool_call` merges fragments
/// and returns the accumulated call; `adapter/adapters/anthropic/streamer.rs`
/// pushes each `input_json_delta` onto the in-progress input and re-emits the
/// accumulated string; `adapter/adapters/gemini/streamer.rs` emits one
/// complete call per functionCall part). The state machine synthesizes the
/// upstream `pi-ai` event shape from that: [`AssistantMessageEvent::ToolcallStart`]
/// when a `call_id` first appears, [`AssistantMessageEvent::ToolcallDelta`]
/// carrying the new suffix as later chunks grow the arguments (mirroring
/// `packages/ai/src/api/anthropic-messages.ts` `input_json_delta` →
/// `toolcall_delta`), and [`AssistantMessageEvent::ToolcallEnd`] with the
/// final assembled call when the block closes (next block starts, or the
/// stream ends). Identical repeated chunks are deduplicated.
pub struct InboundState {
    base: AssistantMessage,
    blocks: Vec<AssistantContent>,
    open: Option<OpenBlock>,
    started: bool,
    /// Map from provider's `tool_call_id` to the accumulation progress of
    /// that call. genai 0.6.5 streams tool-call arguments as **cumulative**
    /// chunks — each subsequent chunk carries the latest accumulated JSON
    /// for the same call_id. Without this map, the old "push a new block
    /// every chunk" behavior produced N duplicate ToolCall blocks for one
    /// call, which the agent loop would then execute N times and which
    /// provider validation (DeepSeek's 400, etc.) rejects as duplicate
    /// tool_call_ids. The stored `raw` string is what lets us compute the
    /// suffix delta between consecutive chunks.
    tool_calls: HashMap<String, ToolCallProgress>,
    /// Thought signatures that arrived with nowhere to attach yet: before
    /// any thinking block existed (Gemini emits `ThoughtSignatureChunk`
    /// *before* the `ReasoningChunk` that would open the block — genai
    /// 0.6.5 `adapter/adapters/gemini/streamer.rs` queues Thought →
    /// Reasoning → Text → ToolCall), or after every existing thinking
    /// block was already signed. One entry per **distinct** signature —
    /// upstream forbids merging signatures across parts
    /// (`packages/ai/src/api/google-shared.ts:29-31`), and Google
    /// validates each signature as base64, so two concatenated padded
    /// signatures would be rejected. The most recent entry seeds the next
    /// thinking block that opens; the rest flush as one signature-only
    /// thinking block each at terminal (e.g. Gemini thinking mode with
    /// thought summaries disabled, where signatures ride on text /
    /// functionCall parts — upstream preserves them as-is for context
    /// replay rather than dropping them).
    pending_thought_signatures: Vec<String>,
}

/// Cumulative-argument progress for one streamed tool call.
struct ToolCallProgress {
    /// Index in `blocks` holding the in-flight ToolCall block.
    index: usize,
    /// Latest accumulated raw argument JSON string, used to compute suffix
    /// deltas and to dedup identical repeated chunks.
    raw: String,
}

/// Grain-owned classification of a [`ChatStreamEvent`] — the
/// forward-compatibility seam required by rust-host.md ("Forward-compatible
/// event handling": the inbound mapping tolerates stream-event variants it
/// does not know without panicking and without emitting loop events;
/// dedicated arms land with the releases that ship them).
///
/// genai 0.6.5's `ChatStreamEvent` is **not** `#[non_exhaustive]`, which
/// makes the doc's sentence impossible to satisfy with a plain `match`: a
/// bare `_` arm is an unreachable pattern today (rejected by the CI-pinned
/// `-D warnings`), and omitting it turns the first genai release that adds a
/// variant into a build failure — the opposite of "tolerates". So unknowns
/// are made representable in a type grain owns: every event is routed
/// through [`From<ChatStreamEvent>`], whose single, narrowly-scoped
/// `#[allow(unreachable_patterns)]` wildcard is the one place a future
/// variant lands. Until a dedicated arm is added for it, such a variant
/// classifies as [`InboundEvent::Unknown`] and
/// [`InboundState::on_classified`] skips it — state intact, no events, no
/// panic. The model is the native Anthropic transport's wire-level handling
/// ([`crate::anthropic::state::AnthropicState::on_frame`], `_ =>
/// Vec::new()`), where unknown event names are naturally representable as
/// strings; this seam buys the Rust enum the same property.
#[derive(Debug)]
pub enum InboundEvent {
    Start,
    /// Text content delta ([`ChatStreamEvent::Chunk`]).
    Text(String),
    /// Reasoning/thinking delta ([`ChatStreamEvent::ReasoningChunk`]).
    Reasoning(String),
    /// Thought-signature payload ([`ChatStreamEvent::ThoughtSignatureChunk`]).
    ThoughtSignature(String),
    /// Cumulative tool-call snapshot ([`ChatStreamEvent::ToolCallChunk`]).
    ToolCall(GenaiToolCall),
    /// Terminal event ([`ChatStreamEvent::End`]).
    End(StreamEnd),
    /// A stream-event variant this adapter has no dedicated arm for.
    /// Unreachable at genai 0.6.5 (the enum is exhaustive here); reachable
    /// the moment a genai bump ships a new variant (e.g. `Heartbeat`, which
    /// exists on genai's unreleased main branch). Skipped without emitting
    /// loop events.
    Unknown,
}

impl From<ChatStreamEvent> for InboundEvent {
    fn from(event: ChatStreamEvent) -> Self {
        match event {
            ChatStreamEvent::Start => InboundEvent::Start,
            ChatStreamEvent::Chunk(c) => InboundEvent::Text(c.content),
            ChatStreamEvent::ReasoningChunk(c) => InboundEvent::Reasoning(c.content),
            ChatStreamEvent::ThoughtSignatureChunk(c) => InboundEvent::ThoughtSignature(c.content),
            ChatStreamEvent::ToolCallChunk(t) => InboundEvent::ToolCall(t.tool_call),
            ChatStreamEvent::End(e) => InboundEvent::End(e),
            // Unreachable at genai 0.6.5 — see the type-level docs for why
            // the arm exists anyway. When a genai bump makes it reachable,
            // rustc's unreachable-pattern diagnostic goes quiet on its own;
            // the allow is scoped to this expression precisely so it cannot
            // hide a dead arm anywhere else.
            #[allow(unreachable_patterns)]
            _ => InboundEvent::Unknown,
        }
    }
}

#[derive(Debug)]
enum OpenBlock {
    Text { index: usize },
    Thinking { index: usize },
    ToolCall { index: usize },
}

impl InboundState {
    /// Initialize using `model` to populate `api` / `provider` / `model`
    /// fields on the partial [`AssistantMessage`].
    pub fn new(model: &Model) -> Self {
        InboundState {
            base: empty_assistant(model),
            blocks: Vec::new(),
            open: None,
            started: false,
            tool_calls: HashMap::new(),
            pending_thought_signatures: Vec::new(),
        }
    }

    fn partial(&self) -> AssistantMessage {
        let mut m = self.base.clone();
        m.content.clone_from(&self.blocks);
        m
    }

    /// Dispatch a single genai event. May produce 0, 1, or 2+ grain events
    /// in emission order (e.g. a text → tool-call transition closes the open
    /// text block then opens the tool-call block).
    ///
    /// Routed through the [`InboundEvent`] classification seam so a
    /// stream-event variant this adapter does not know degrades to a skipped
    /// [`InboundEvent::Unknown`] instead of failing the build or the stream
    /// (rust-host.md, "Forward-compatible event handling").
    pub fn on_event(&mut self, event: ChatStreamEvent) -> Vec<AssistantMessageEvent> {
        self.on_classified(InboundEvent::from(event))
    }

    /// Dispatch one already-classified event. Public so the
    /// forward-compatibility contract is testable: [`InboundEvent::Unknown`]
    /// cannot be constructed *from* genai 0.6.5 (no unknown variant exists
    /// yet), but its handling — skip, keep state, emit nothing — must hold
    /// before the genai bump that makes it reachable.
    pub fn on_classified(&mut self, event: InboundEvent) -> Vec<AssistantMessageEvent> {
        match event {
            InboundEvent::Start => self.on_start(),
            InboundEvent::Text(content) => self.on_text_chunk(content),
            InboundEvent::Reasoning(content) => self.on_reasoning_chunk(content),
            InboundEvent::ThoughtSignature(content) => self.on_thought_signature(content),
            InboundEvent::ToolCall(tool_call) => self.on_tool_call(tool_call),
            InboundEvent::End(end) => self.on_end(end),
            // Tolerated, not surfaced: no loop events, no state change, no
            // panic. Mirrors the native transport's wire-level
            // `_ => Vec::new()` for unknown SSE event names.
            InboundEvent::Unknown => Vec::new(),
        }
    }

    fn on_start(&mut self) -> Vec<AssistantMessageEvent> {
        if self.started {
            return Vec::new();
        }
        self.started = true;
        vec![AssistantMessageEvent::Start {
            partial: self.partial(),
        }]
    }

    fn ensure_started(&mut self, out: &mut Vec<AssistantMessageEvent>) {
        if !self.started {
            self.started = true;
            out.push(AssistantMessageEvent::Start {
                partial: self.partial(),
            });
        }
    }

    fn on_text_chunk(&mut self, content: String) -> Vec<AssistantMessageEvent> {
        let mut out = Vec::new();
        self.ensure_started(&mut out);
        // Close mismatched open block (thinking or tool-call).
        if !matches!(self.open, None | Some(OpenBlock::Text { .. })) {
            self.close_open(&mut out);
        }
        if self.open.is_none() {
            self.blocks
                .push(AssistantContent::Text(TextContent::default()));
            let idx = self.blocks.len() - 1;
            self.open = Some(OpenBlock::Text { index: idx });
            out.push(AssistantMessageEvent::TextStart {
                partial: self.partial(),
                content_index: idx,
            });
        }
        if let Some(OpenBlock::Text { index }) = self.open {
            if let AssistantContent::Text(t) = &mut self.blocks[index] {
                t.text.push_str(&content);
            }
            out.push(AssistantMessageEvent::TextDelta {
                partial: self.partial(),
                content_index: index,
                delta: content,
            });
        }
        out
    }

    fn on_reasoning_chunk(&mut self, content: String) -> Vec<AssistantMessageEvent> {
        let mut out = Vec::new();
        self.ensure_started(&mut out);
        // Close mismatched open block (text or tool-call).
        if !matches!(self.open, None | Some(OpenBlock::Thinking { .. })) {
            self.close_open(&mut out);
        }
        if self.open.is_none() {
            self.blocks
                .push(AssistantContent::Thinking(ThinkingContent {
                    thinking: String::new(),
                    // Adopt the most recently pending signature: genai's
                    // Gemini streamer queues the ThoughtSignatureChunk
                    // immediately before the first ReasoningChunk of the
                    // part it certifies, so the newest pending entry is
                    // this block's signature. Earlier pending entries
                    // belong to other parts and stay pending (flushed as
                    // their own blocks at terminal) — signatures are never
                    // moved between parts.
                    signature: self.pending_thought_signatures.pop(),
                    provider_metadata: None,
                }));
            let idx = self.blocks.len() - 1;
            self.open = Some(OpenBlock::Thinking { index: idx });
            out.push(AssistantMessageEvent::ThinkingStart {
                partial: self.partial(),
                content_index: idx,
            });
        }
        if let Some(OpenBlock::Thinking { index }) = self.open {
            if let AssistantContent::Thinking(t) = &mut self.blocks[index] {
                t.thinking.push_str(&content);
            }
            out.push(AssistantMessageEvent::ThinkingDelta {
                partial: self.partial(),
                content_index: index,
                delta: content,
            });
        }
        out
    }

    /// Anthropic-style signed thinking: silently update the thinking
    /// signature (grain's natural slot, `ThinkingContent::signature`). No
    /// separate grain event — subscribers see the updated signature on the
    /// next partial. Mirrors upstream `pi-ai`:
    /// - `packages/ai/src/api/anthropic-messages.ts` (`signature_delta`)
    ///   appends fragments of the ONE in-progress signature to the open
    ///   thinking block;
    /// - `packages/ai/src/api/google-shared.ts` documents that a Gemini
    ///   signature can arrive on any part — including *after* the thinking
    ///   content it certifies — must be preserved for context replay, and
    ///   must never be merged with another part's signature (lines 29-31;
    ///   Google validates each signature as base64, so concatenation would
    ///   corrupt both). An orphan signature therefore attaches to the most
    ///   recent *unsigned* thinking block, or is held pending as its own
    ///   distinct entry (see [`Self::pending_thought_signatures`]). No
    ///   ordering panics.
    fn on_thought_signature(&mut self, content: String) -> Vec<AssistantMessageEvent> {
        self.attach_thought_signature(content);
        Vec::new()
    }

    fn attach_thought_signature(&mut self, content: String) {
        // 1. Open thinking block: append — fragments of the same signature
        //    stream while the block is open (anthropic signature_delta
        //    shape). Distinct-signature chunks never interleave with an
        //    open block: Gemini closes the block (via the next non-thinking
        //    part's events) before another part's signature arrives.
        if let Some(OpenBlock::Thinking { index }) = self.open
            && let AssistantContent::Thinking(t) = &mut self.blocks[index]
        {
            append_signature(t, &content);
            return;
        }
        // 2. The most recent thinking block is closed and *unsigned*: the
        //    signature certifies that reasoning (Gemini "signature on the
        //    last part" case) — attach it whole.
        if let Some(AssistantContent::Thinking(t)) = self
            .blocks
            .iter_mut()
            .rev()
            .find(|b| matches!(b, AssistantContent::Thinking(_)))
            && t.signature.is_none()
        {
            t.signature = Some(content);
            return;
        }
        // 3. No thinking block yet, or every thinking block already signed:
        //    this is a DISTINCT signature — preserve its identity as its
        //    own pending entry (never concatenate two signatures). The
        //    newest entry seeds the next thinking block that opens; the
        //    rest flush as signature-only blocks at terminal (see
        //    `flush_pending_signatures`).
        self.pending_thought_signatures.push(content);
    }

    /// Preserve any thought signatures still pending at terminal: one
    /// signature-only thinking block per distinct signature, in arrival
    /// order, so outbound replay (`mapping::outbound`, which forwards
    /// `ThinkingContent::signature` values as `thought_signatures` on the
    /// first outgoing tool call) round-trips each signature as-is —
    /// dropping or merging them would break Gemini/Anthropic multi-turn
    /// signed-thinking flows.
    fn flush_pending_signatures(&mut self) {
        for sig in std::mem::take(&mut self.pending_thought_signatures) {
            self.blocks
                .push(AssistantContent::Thinking(ThinkingContent {
                    thinking: String::new(),
                    signature: Some(sig),
                    provider_metadata: None,
                }));
        }
    }

    fn on_tool_call(&mut self, tc: GenaiToolCall) -> Vec<AssistantMessageEvent> {
        let mut out = Vec::new();
        self.ensure_started(&mut out);

        // Rare inbound path: a chunk-level ToolCall carrying thought
        // signatures directly (genai's non-streaming Gemini shape). Route
        // them through the same attachment logic as ThoughtSignatureChunk.
        if let Some(sigs) = tc.thought_signatures.clone() {
            for sig in sigs {
                self.attach_thought_signature(sig);
            }
        }

        let raw = raw_tool_args(&tc.fn_arguments);

        // Later chunk for a call we've already seen: cumulative growth.
        if let Some(progress) = self.tool_calls.get(&tc.call_id) {
            if raw == progress.raw {
                // Identical repeated chunk — dedup, no event.
                return out;
            }
            let idx = progress.index;
            // Cumulative semantics: the new chunk normally extends the
            // previous accumulation, so the delta is the new suffix. A
            // non-prefix replacement (not observed in genai 0.6.5, which
            // only ever grows the accumulation) degrades to emitting the
            // full new serialization as the delta.
            let delta = match raw.strip_prefix(progress.raw.as_str()) {
                Some(suffix) => suffix.to_string(),
                None => raw.clone(),
            };
            if let AssistantContent::ToolCall(existing) = &mut self.blocks[idx] {
                existing.arguments = parse_tool_args(&raw);
            }
            if let Some(p) = self.tool_calls.get_mut(&tc.call_id) {
                p.raw = raw;
            }
            // Only emit a delta while the block is still open. If it was
            // already closed (another block started since), Start/End were
            // emitted for it — refresh the block silently; the agent loop
            // executes tool calls from the final message on `Done`, so it
            // always picks up the fully-accumulated args.
            if matches!(self.open, Some(OpenBlock::ToolCall { index }) if index == idx) {
                out.push(AssistantMessageEvent::ToolcallDelta {
                    partial: self.partial(),
                    content_index: idx,
                    delta,
                });
            }
            return out;
        }

        // First chunk for this call_id: open a tool-call block.
        if self.open.is_some() {
            self.close_open(&mut out);
        }
        let grain_tc = GrainToolCall {
            id: tc.call_id.clone(),
            name: tc.fn_name,
            arguments: parse_tool_args(&raw),
            // WP19 mechanical fill only — preserves today's behavior exactly.
            // Populating this from `pending_thought_signatures` (instead of
            // the empty-thinking-block workaround below) is adapter work
            // owned by the concurrent grain-llm-genai package.
            thought_signature: None,
        };
        self.blocks.push(AssistantContent::ToolCall(grain_tc));
        let idx = self.blocks.len() - 1;
        self.tool_calls.insert(
            tc.call_id,
            ToolCallProgress {
                index: idx,
                raw: raw.clone(),
            },
        );
        self.open = Some(OpenBlock::ToolCall { index: idx });
        out.push(AssistantMessageEvent::ToolcallStart {
            partial: self.partial(),
            content_index: idx,
        });
        // If the first chunk already carries arguments (Gemini delivers the
        // complete call at once; upstream `google-generative-ai.ts` emits
        // toolcall_start → toolcall_delta(full JSON) → toolcall_end), emit
        // the initial content as a delta. An empty first chunk (Anthropic
        // opens the block with empty args) emits Start only — deltas follow
        // as the accumulation grows.
        if !raw.is_empty() {
            out.push(AssistantMessageEvent::ToolcallDelta {
                partial: self.partial(),
                content_index: idx,
                delta: raw,
            });
        }
        out
    }

    fn on_end(&mut self, end: StreamEnd) -> Vec<AssistantMessageEvent> {
        let mut out = Vec::new();
        self.ensure_started(&mut out);
        if self.open.is_some() {
            self.close_open(&mut out);
        }
        self.flush_pending_signatures();

        let mut result = self.base.clone();
        result.content = std::mem::take(&mut self.blocks);
        if let Some(u) = end.captured_usage {
            result.usage = map_usage(u);
        }
        result.timestamp = now_ms();

        // WP32 (rust-host ledger item 13; closes WP5 AB-R1): carry the
        // provider's VERBATIM stop string alongside the normalized reason.
        // genai preserves it inside every `StopReason` variant
        // (`StopReason::raw()`), and upstream pi-ai assigns it at the moment
        // the raw reason arrives, before mapping and unconditionally —
        // `output.rawStopReason = choice.finish_reason`
        // (`openai-completions.ts:459`), `= candidate.finishReason`
        // (`google-generative-ai.ts:215`, `google-vertex.ts:232`),
        // `= event.delta.stop_reason` (`anthropic-messages.ts:709`), all at
        // pin 34239180. Locally synthesized terminals (abort, transport
        // error, a stream that ends without a captured reason) have no
        // provider string and stay `None`, matching upstream's undefined.
        result.raw_stop_reason = end
            .captured_stop_reason
            .as_ref()
            .map(|reason| reason.raw().to_string());

        // WP5 (adapter-bug AB-1, seam vectors OV-1/OV-2/GV-1/GV-2/AV-3):
        // genai 0.6.5 delivers the provider's finish reason on
        // `StreamEnd::captured_stop_reason` — with the raw provider string
        // preserved inside each variant — and this adapter used to ignore
        // it entirely, silently reporting `Stop` for content-filtered /
        // refused / truncated turns. Resolve it the way upstream pi-ai
        // does; error-class reasons terminate with an `Error` event (the
        // upstream streams throw and emit `{type:"error"}`), carrying all
        // accumulated content and usage.
        let resolution = resolve_stop_reason(
            end.captured_stop_reason.as_ref(),
            &result.content,
            &self.base.api,
        );
        result.stop_reason = resolution.stop_reason;
        match resolution.error_message {
            Some(msg) => {
                result.error_message = Some(msg.clone());
                out.push(AssistantMessageEvent::Error { error: msg, result });
            }
            None => out.push(AssistantMessageEvent::Done { result }),
        }
        out
    }

    /// Consume self and emit a terminal aborted error event.
    ///
    /// `StreamFn` contract (mirroring upstream `pi-ai`'s error event, which
    /// carries the partial `AssistantMessage` — see the catch handler in
    /// `packages/ai/src/api/anthropic-messages.ts`): the terminal event
    /// preserves all content accumulated so far.
    pub fn into_aborted(mut self) -> AssistantMessageEvent {
        self.flush_pending_signatures();
        let mut result = self.base.clone();
        result.content = std::mem::take(&mut self.blocks);
        result.stop_reason = StopReason::Aborted;
        result.error_message = Some("aborted".into());
        result.timestamp = now_ms();
        AssistantMessageEvent::Error {
            error: "aborted".into(),
            result,
        }
    }

    /// Consume self and emit a terminal error event with the given message.
    ///
    /// Like [`Self::into_aborted`], the terminal carries the partial
    /// assistant message with all accumulated content (upstream parity).
    pub fn into_error_msg(mut self, msg: impl Into<String>) -> AssistantMessageEvent {
        self.flush_pending_signatures();
        let msg = msg.into();
        let mut result = self.base.clone();
        result.content = std::mem::take(&mut self.blocks);
        result.stop_reason = StopReason::Error;
        result.error_message = Some(msg.clone());
        result.timestamp = now_ms();
        AssistantMessageEvent::Error { error: msg, result }
    }

    fn close_open(&mut self, out: &mut Vec<AssistantMessageEvent>) {
        let Some(open) = self.open.take() else { return };
        match open {
            OpenBlock::Text { index } => out.push(AssistantMessageEvent::TextEnd {
                partial: self.partial(),
                content_index: index,
            }),
            OpenBlock::Thinking { index } => out.push(AssistantMessageEvent::ThinkingEnd {
                partial: self.partial(),
                content_index: index,
            }),
            // The block's arguments already hold the latest accumulated
            // parse (updated on every chunk), so the partial carried here
            // contains the final assembled call.
            OpenBlock::ToolCall { index } => out.push(AssistantMessageEvent::ToolcallEnd {
                partial: self.partial(),
                content_index: index,
            }),
        }
    }
}

fn append_signature(t: &mut ThinkingContent, content: &str) {
    match &mut t.signature {
        Some(existing) => existing.push_str(content),
        None => t.signature = Some(content.to_string()),
    }
}

fn infer_stop_reason(content: &[AssistantContent]) -> StopReason {
    if content
        .iter()
        .any(|c| matches!(c, AssistantContent::ToolCall(_)))
    {
        StopReason::ToolUse
    } else {
        StopReason::Stop
    }
}

/// Outcome of terminal stop-reason resolution: the grain stop reason plus,
/// for error-class provider reasons, the error message that turns the
/// terminal event into [`AssistantMessageEvent::Error`].
struct StopResolution {
    stop_reason: StopReason,
    error_message: Option<String>,
}

impl StopResolution {
    fn ok(stop_reason: StopReason) -> Self {
        StopResolution {
            stop_reason,
            error_message: None,
        }
    }

    fn error(message: String) -> Self {
        StopResolution {
            stop_reason: StopReason::Error,
            error_message: Some(message),
        }
    }
}

/// Map genai's captured [`genai::chat::StopReason`] onto grain semantics,
/// mirroring upstream pi-ai's per-provider `mapStopReason` tables
/// (`packages/ai/src/api/anthropic-messages.ts`, `openai-completions.ts`,
/// `google-shared.ts` at pin 34239180):
///
/// | genai variant           | grain outcome                                        |
/// |-------------------------|------------------------------------------------------|
/// | `Completed(_)`          | `Stop`; google APIs only: → `ToolUse` when tool calls streamed |
/// | `StopSequence(_)`       | `Stop` (upstream: "should never happen", maps stop; same google-only override) |
/// | `MaxTokens(_)`          | `Length`                                             |
/// | `ToolCall(_)`           | `ToolUse`                                            |
/// | `ContentFilter(raw)`    | `Error` + provider-stop message                      |
/// | `Other("pause_turn")`   | `Stop`, unconditionally (anthropic: resubmittable pause) |
/// | `Other("refusal")`      | `Error` + anthropic's no-details refusal message     |
/// | `Other(raw)`            | `Error` + provider-stop message                      |
/// | `None` (not captured)   | infer from content (pre-WP5 behavior)                |
///
/// The `Stop` → `ToolUse` content override is scoped to the google API
/// family exactly as upstream scopes it: both google modules apply it
/// after mapping the finish reason (`google-generative-ai.ts:231-235`,
/// `google-vertex.ts:231-235` — Gemini reports `STOP` even when the chunk
/// carried functionCall parts), while `anthropic-messages.ts` and
/// `openai-completions.ts` trust the provider's reason verbatim —
/// `end_turn` / `stop` / `stop_sequence` map to `stop` even if tool-call
/// content streamed. We mirror upstream per provider rather than
/// generalizing the override, since the adapter knows the API family from
/// the model descriptor.
///
/// Error-message phrasing follows the upstream API family owning the
/// vector: `openai-*` APIs phrase as `Provider finish_reason: <raw>`
/// (`openai-completions.ts` `mapStopReason`), everything else as
/// `Provider stopped with: <raw>` (`anthropic-messages.ts`,
/// `google-generative-ai.ts`). The raw provider string is available here
/// because genai preserves it inside every `StopReason` variant, and since
/// WP32 [`InboundState::on_end`] assigns it verbatim into
/// `AssistantMessage::raw_stop_reason` (rust-host ledger item 13; the
/// residual gap formerly reported as WP5 AB-R1 is closed — see the
/// assignment site for the upstream references).
fn resolve_stop_reason(
    captured: Option<&genai::chat::StopReason>,
    content: &[AssistantContent],
    api: &str,
) -> StopResolution {
    use genai::chat::StopReason as GenaiStop;

    let Some(captured) = captured else {
        return StopResolution::ok(infer_stop_reason(content));
    };

    match captured {
        GenaiStop::Completed(_) | GenaiStop::StopSequence(_) => {
            // Gemini reports `STOP` alongside functionCall parts; upstream
            // forces toolUse whenever tool calls are present — but ONLY in
            // the two google modules. Anthropic/OpenAI map their completed
            // reasons to `stop` verbatim, tool-call content or not.
            if is_google_api(api) {
                StopResolution::ok(infer_stop_reason(content))
            } else {
                StopResolution::ok(StopReason::Stop)
            }
        }
        GenaiStop::MaxTokens(_) => StopResolution::ok(StopReason::Length),
        GenaiStop::ToolCall(_) => StopResolution::ok(StopReason::ToolUse),
        GenaiStop::ContentFilter(raw) => StopResolution::error(provider_stop_message(api, raw)),
        GenaiStop::Other(raw) => match raw.as_str() {
            // anthropic-messages.ts: pause_turn → "stop is good enough,
            // resubmit". Hard-mapped to Stop, no content inference —
            // upstream's table is unconditional.
            "pause_turn" => StopResolution::ok(StopReason::Stop),
            // anthropic-messages.ts: refusal → stopDetails.explanation ||
            // default message. genai never parses `stop_details`
            // (structural gap S-4), so only the default is reachable.
            "refusal" => {
                StopResolution::error("The model refused to complete the request".to_string())
            }
            _ => StopResolution::error(provider_stop_message(api, raw)),
        },
    }
}

/// The google/Gemini API family — the only modules where upstream applies
/// the tool-call `ToolUse` override after finish-reason mapping
/// (`google-generative-ai.ts`, `google-vertex.ts`). Matches both upstream's
/// api ids (`google-generative-ai`, `google-vertex`, `google`) and grain's
/// production wire name (`grain_llm_models::ApiKind::Gemini` →
/// `Model::api == "gemini"`).
fn is_google_api(api: &str) -> bool {
    api.starts_with("google") || api == "gemini"
}

/// Upstream-parity phrasing for error-class provider stop reasons. See
/// [`resolve_stop_reason`] for the per-API-family sources.
fn provider_stop_message(api: &str, raw: &str) -> String {
    if api.starts_with("openai") {
        format!("Provider finish_reason: {raw}")
    } else {
        format!("Provider stopped with: {raw}")
    }
}

fn empty_assistant(model: &Model) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_id: None,
        response_model: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        // Stays `None` on partials and on locally synthesized terminals;
        // `on_end` overwrites it with the verbatim provider stop string when
        // genai captured one (WP32, rust-host ledger item 13).
        raw_stop_reason: None,
        error_message: None,
        error_code: None,
        timestamp: now_ms(),
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Project a `tool_call.fn_arguments` value onto the canonical raw argument
/// JSON string used for delta computation.
///
/// genai 0.6.5 delivers `fn_arguments` either as a `Value::String` holding
/// the (possibly partial) accumulated JSON (OpenAI / Anthropic adapters) or
/// as an already-structured `Value` (Gemini delivers the complete object).
/// `Null` normalizes to the empty accumulation.
fn raw_tool_args(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Parse the accumulated raw argument string into the block's `arguments`.
///
/// - Empty accumulation → empty object (upstream starts toolCall blocks with
///   `arguments: {}`).
/// - Complete JSON → parsed value.
/// - JSON malformed only in its **string literals** (an invalid escape such
///   as `\H`, or a raw control character like a TAB that the provider failed
///   to escape) → repaired and parsed, mirroring upstream pi-ai's
///   `parseJsonWithRepair` (see [`crate::mapping::json_repair`]). Before
///   WP21 these fell through to the `Value::String` branch below, where the
///   outbound corrupt-args guard
///   ([`crate::mapping::outbound::to_chat_request`]) dropped the entire tool
///   call — one stray escape anywhere in the arguments silently cost the
///   user the whole call. This is the adapter-reachable half of structural
///   gap S-5.
/// - Partial / truncated JSON (mid-stream, or a stream that ended early) →
///   kept as `Value::String`; the outbound layer's corrupt-args guard
///   recognizes that shape if it ever escapes a terminal (e.g. aborted
///   stream), and the final chunk's complete JSON replaces it on the happy
///   path. Upstream additionally coerces truncated buffers into a partial
///   object; grain deliberately does not — see the scope note in
///   [`crate::mapping::json_repair`].
fn parse_tool_args(raw: &str) -> serde_json::Value {
    if raw.is_empty() {
        return serde_json::Value::Object(Default::default());
    }
    crate::mapping::json_repair::parse_json_with_repair(raw)
        .unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}
