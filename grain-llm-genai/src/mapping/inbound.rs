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
    pub fn on_event(&mut self, event: ChatStreamEvent) -> Vec<AssistantMessageEvent> {
        match event {
            ChatStreamEvent::Start => self.on_start(),
            ChatStreamEvent::Chunk(c) => self.on_text_chunk(c.content),
            ChatStreamEvent::ReasoningChunk(c) => self.on_reasoning_chunk(c.content),
            ChatStreamEvent::ThoughtSignatureChunk(c) => self.on_thought_signature(c.content),
            ChatStreamEvent::ToolCallChunk(t) => self.on_tool_call(t.tool_call),
            ChatStreamEvent::End(e) => self.on_end(e),
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
        result.stop_reason = infer_stop_reason(&result.content);
        result.timestamp = now_ms();

        out.push(AssistantMessageEvent::Done { result });
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

fn empty_assistant(model: &Model) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
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
/// - Partial JSON (mid-stream) → kept as `Value::String`; the outbound
///   layer's corrupt-args guard recognizes that shape if it ever escapes a
///   terminal (e.g. aborted stream), and the final chunk's complete JSON
///   replaces it on the happy path.
fn parse_tool_args(raw: &str) -> serde_json::Value {
    if raw.is_empty() {
        return serde_json::Value::Object(Default::default());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}
