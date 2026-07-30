//! Anthropic stream event → grain `AssistantMessageEvent` state machine.
//!
//! This is where the structural gaps the genai seam destroys are actually
//! closed, because here the provider's own events are still intact:
//!
//! - **S-3 (usage double-count).** `message_delta.usage` is a *cumulative
//!   snapshot*, not a delta. [`UsageAccumulator`] therefore **replaces**
//!   per field, and skips fields the event omits so a `message_delta` without
//!   usage is a no-op (upstream's own asserted case). The genai backend adds
//!   instead, inflating every counter the provider repeats.
//! - **S-4 (`stop_details`).** `delta.stop_details.explanation` is captured and
//!   surfaced as the terminal `error_message`.
//! - **S-8 (block boundaries).** `content_block_start` / `content_block_stop`
//!   drive the `*_start` / `*_end` events directly instead of being inferred
//!   from a change in event kind, so a block that opens and closes with no
//!   delta still produces its pair.
//! - **S-5 (frame repair).** Frame payloads are parsed with
//!   [`crate::mapping::json_repair`], so a frame malformed only in its string
//!   literals is repaired rather than aborting the turn.
//! - **S-2 / S-6 (`responseId` / `responseModel`).** Both *are* captured here
//!   ([`AnthropicState::response_id`], [`AnthropicState::response_model`])
//!   because the wire carries them — but `grain_agent_core::AssistantMessage`
//!   has no slot for either, so they stop at this boundary. See the WP25
//!   report; adding the core fields is unowned work.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use grain_agent_core::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, Model, StopReason, TextContent,
    ThinkingContent, ToolCall, Usage,
};
use serde_json::Value;

use crate::mapping::json_repair::parse_json_with_repair;

/// Per-field replacement accumulator for Anthropic usage.
///
/// Anthropic reports usage twice: once in `message_start` (input side known,
/// output still 0) and once in the final `message_delta` (cumulative totals
/// for the whole message). Each field present in a later event **replaces**
/// the earlier value; a field that is absent leaves the earlier value intact.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UsageAccumulator {
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
}

impl UsageAccumulator {
    /// Apply an Anthropic `usage` object with replacement semantics.
    pub fn apply(&mut self, usage: &Value) {
        if let Some(v) = u64_at(usage, "input_tokens") {
            self.input = v;
        }
        if let Some(v) = u64_at(usage, "output_tokens") {
            self.output = v;
        }
        if let Some(v) = u64_at(usage, "cache_read_input_tokens") {
            self.cache_read = v;
        }
        if let Some(v) = u64_at(usage, "cache_creation_input_tokens") {
            self.cache_write = v;
        }
    }

    /// Project onto grain's `Usage`.
    ///
    /// grain's `input` is **cache-inclusive** (documented on
    /// `grain_agent_core::Cost::cost_for`), whereas Anthropic's `input_tokens`
    /// excludes both cache counters — so they are folded in here, and the
    /// total is computed from the components exactly as upstream does.
    pub fn to_usage(self) -> Usage {
        let input = self.input + self.cache_read + self.cache_write;
        Usage {
            input,
            output: self.output,
            cache_read: self.cache_read,
            cache_write: self.cache_write,
            total_tokens: input + self.output,
            ..Usage::default()
        }
    }
}

fn u64_at(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| x.as_u64())
}

fn str_at<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(|x| x.as_str())
}

#[derive(Debug, Clone, Copy)]
enum OpenBlock {
    Text(usize),
    Thinking(usize),
    ToolCall(usize),
}

impl OpenBlock {
    fn index(self) -> usize {
        match self {
            OpenBlock::Text(i) | OpenBlock::Thinking(i) | OpenBlock::ToolCall(i) => i,
        }
    }
}

/// Streaming state for one Anthropic assistant turn.
pub struct AnthropicState {
    base: AssistantMessage,
    blocks: Vec<AssistantContent>,
    /// Anthropic block index → our `blocks` index.
    open: HashMap<u64, OpenBlock>,
    /// Accumulated raw `partial_json` per `blocks` index, for tool calls.
    tool_raw: HashMap<usize, String>,
    started: bool,
    usage: UsageAccumulator,
    stop_reason: Option<String>,
    stop_explanation: Option<String>,
    /// Captured from `message_start.message.id`. No grain slot today (S-2).
    pub response_id: Option<String>,
    /// Captured from `message_start.message.model`. No grain slot today (S-6).
    pub response_model: Option<String>,
    finished: bool,
}

impl AnthropicState {
    pub fn new(model: &Model) -> Self {
        AnthropicState {
            base: AssistantMessage {
                content: Vec::new(),
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                // WP19 mechanical fill only, matching the sibling genai path
                // in `mapping::inbound::empty_assistant`. This transport
                // *does* capture both values (see `Self::response_id` /
                // `Self::response_model`, populated from `message_start`),
                // and WP19 has since added the slots to carry them — but
                // landing them here is a live behavior change on a transport
                // that is opt-in and not yet live-verified, so it stays with
                // the transport owner (WP24) rather than riding in on a
                // build fix. Wiring these three closes G12/G13.
                response_id: None,
                response_model: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                // Likewise: `Self::stop_reason` holds the provider's verbatim
                // stop string and is exactly what this slot wants. Left None
                // for the same reason as above.
                raw_stop_reason: None,
                error_message: None,
                error_code: None,
                timestamp: now_ms(),
            },
            blocks: Vec::new(),
            open: HashMap::new(),
            tool_raw: HashMap::new(),
            started: false,
            usage: UsageAccumulator::default(),
            stop_reason: None,
            stop_explanation: None,
            response_id: None,
            response_model: None,
            finished: false,
        }
    }

    /// True once a terminal event has been emitted.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    fn partial(&self) -> AssistantMessage {
        let mut m = self.base.clone();
        m.content.clone_from(&self.blocks);
        m.usage = self.usage.to_usage();
        m
    }

    fn ensure_started(&mut self, out: &mut Vec<AssistantMessageEvent>) {
        if !self.started {
            self.started = true;
            out.push(AssistantMessageEvent::Start {
                partial: self.partial(),
            });
        }
    }

    /// Handle one decoded SSE frame.
    ///
    /// `event_name` is the SSE `event:` field; `data` is the raw payload,
    /// which is parsed leniently (S-5 frame repair). A frame that cannot be
    /// parsed even after repair is ignored rather than aborting the turn —
    /// upstream's behavior, and strictly better than genai's hard abort.
    pub fn on_frame(&mut self, event_name: &str, data: &str) -> Vec<AssistantMessageEvent> {
        if self.finished {
            return Vec::new();
        }
        // Events after `message_stop` (proxy junk like `done` / `proxy.stats`)
        // never reach here because the caller stops on the terminal.
        let Ok(value) = parse_json_with_repair(data) else {
            return Vec::new();
        };

        match event_name {
            "message_start" => self.on_message_start(&value),
            "content_block_start" => self.on_block_start(&value),
            "content_block_delta" => self.on_block_delta(&value),
            "content_block_stop" => self.on_block_stop(&value),
            "message_delta" => {
                self.on_message_delta(&value);
                Vec::new()
            }
            "message_stop" => self.finish(),
            "error" => {
                let msg = value
                    .get("error")
                    .and_then(|e| str_at(e, "message"))
                    .unwrap_or("provider stream error")
                    .to_string();
                self.into_error(msg)
            }
            // "ping" and anything unrecognized: ignore.
            _ => Vec::new(),
        }
    }

    fn on_message_start(&mut self, value: &Value) -> Vec<AssistantMessageEvent> {
        if let Some(message) = value.get("message") {
            self.response_id = str_at(message, "id").map(str::to_string);
            self.response_model = str_at(message, "model").map(str::to_string);
            if let Some(usage) = message.get("usage") {
                self.usage.apply(usage);
            }
        }
        let mut out = Vec::new();
        self.ensure_started(&mut out);
        out
    }

    fn on_block_start(&mut self, value: &Value) -> Vec<AssistantMessageEvent> {
        let mut out = Vec::new();
        self.ensure_started(&mut out);

        let Some(wire_index) = u64_at(value, "index") else {
            return out;
        };
        let Some(block) = value.get("content_block") else {
            return out;
        };

        match str_at(block, "type") {
            Some("text") => {
                self.blocks.push(AssistantContent::Text(TextContent {
                    text: str_at(block, "text").unwrap_or_default().to_string(),
                }));
                let idx = self.blocks.len() - 1;
                self.open.insert(wire_index, OpenBlock::Text(idx));
                out.push(AssistantMessageEvent::TextStart {
                    partial: self.partial(),
                    content_index: idx,
                });
            }
            // A `redacted_thinking` block carries NEITHER `thinking` nor
            // `signature` -- its whole payload is an opaque `data` string that
            // must be replayed verbatim. Reading the normal fields would yield
            // an empty, unsigned block, which the replay path then drops, so
            // the block would vanish from history entirely. Upstream pi-ai
            // stores `content_block.data` and marks the block redacted; grain
            // has no dedicated field, so it rides in `provider_metadata` --
            // whose documented purpose is exactly this (a provider-specific
            // raw reasoning payload preserved for verbatim replay).
            Some("redacted_thinking") => {
                self.blocks
                    .push(AssistantContent::Thinking(ThinkingContent {
                        thinking: String::new(),
                        signature: None,
                        provider_metadata: Some(serde_json::json!({
                            "type": "redacted_thinking",
                            "data": str_at(block, "data").unwrap_or_default(),
                        })),
                    }));
                let idx = self.blocks.len() - 1;
                self.open.insert(wire_index, OpenBlock::Thinking(idx));
                out.push(AssistantMessageEvent::ThinkingStart {
                    partial: self.partial(),
                    content_index: idx,
                });
            }
            Some("thinking") => {
                self.blocks
                    .push(AssistantContent::Thinking(ThinkingContent {
                        thinking: str_at(block, "thinking").unwrap_or_default().to_string(),
                        signature: str_at(block, "signature").map(str::to_string),
                        provider_metadata: None,
                    }));
                let idx = self.blocks.len() - 1;
                self.open.insert(wire_index, OpenBlock::Thinking(idx));
                out.push(AssistantMessageEvent::ThinkingStart {
                    partial: self.partial(),
                    content_index: idx,
                });
            }
            Some("tool_use") => {
                self.blocks.push(AssistantContent::ToolCall(ToolCall {
                    id: str_at(block, "id").unwrap_or_default().to_string(),
                    name: str_at(block, "name").unwrap_or_default().to_string(),
                    arguments: Value::Object(Default::default()),
                    // Anthropic does not emit a thought signature on the tool
                    // call itself (it carries one on `thinking` blocks, which
                    // this transport already maps to `ThinkingContent`), so
                    // there is genuinely nothing to supply here.
                    thought_signature: None,
                }));
                let idx = self.blocks.len() - 1;
                self.open.insert(wire_index, OpenBlock::ToolCall(idx));
                self.tool_raw.insert(idx, String::new());
                out.push(AssistantMessageEvent::ToolcallStart {
                    partial: self.partial(),
                    content_index: idx,
                });
            }
            _ => {}
        }
        out
    }

    fn on_block_delta(&mut self, value: &Value) -> Vec<AssistantMessageEvent> {
        let mut out = Vec::new();
        let Some(wire_index) = u64_at(value, "index") else {
            return out;
        };
        let Some(open) = self.open.get(&wire_index).copied() else {
            return out;
        };
        let Some(delta) = value.get("delta") else {
            return out;
        };
        let idx = open.index();

        match (open, str_at(delta, "type")) {
            (OpenBlock::Text(_), Some("text_delta")) => {
                let text = str_at(delta, "text").unwrap_or_default().to_string();
                if let AssistantContent::Text(t) = &mut self.blocks[idx] {
                    t.text.push_str(&text);
                }
                out.push(AssistantMessageEvent::TextDelta {
                    partial: self.partial(),
                    content_index: idx,
                    delta: text,
                });
            }
            (OpenBlock::Thinking(_), Some("thinking_delta")) => {
                let text = str_at(delta, "thinking").unwrap_or_default().to_string();
                if let AssistantContent::Thinking(t) = &mut self.blocks[idx] {
                    t.thinking.push_str(&text);
                }
                out.push(AssistantMessageEvent::ThinkingDelta {
                    partial: self.partial(),
                    content_index: idx,
                    delta: text,
                });
            }
            // Signature fragments update the block silently; grain has no
            // signature event (same convention as the genai path).
            (OpenBlock::Thinking(_), Some("signature_delta")) => {
                let sig = str_at(delta, "signature").unwrap_or_default().to_string();
                if let AssistantContent::Thinking(t) = &mut self.blocks[idx] {
                    match &mut t.signature {
                        Some(existing) => existing.push_str(&sig),
                        None => t.signature = Some(sig),
                    }
                }
            }
            (OpenBlock::ToolCall(_), Some("input_json_delta")) => {
                let fragment = str_at(delta, "partial_json").unwrap_or_default().to_string();
                let raw = self.tool_raw.entry(idx).or_default();
                raw.push_str(&fragment);
                let accumulated = raw.clone();
                if let AssistantContent::ToolCall(tc) = &mut self.blocks[idx] {
                    tc.arguments = parse_tool_args(&accumulated);
                }
                out.push(AssistantMessageEvent::ToolcallDelta {
                    partial: self.partial(),
                    content_index: idx,
                    delta: fragment,
                });
            }
            _ => {}
        }
        out
    }

    fn on_block_stop(&mut self, value: &Value) -> Vec<AssistantMessageEvent> {
        let mut out = Vec::new();
        let Some(wire_index) = u64_at(value, "index") else {
            return out;
        };
        let Some(open) = self.open.remove(&wire_index) else {
            return out;
        };
        let idx = open.index();
        let partial = self.partial();
        out.push(match open {
            OpenBlock::Text(_) => AssistantMessageEvent::TextEnd {
                partial,
                content_index: idx,
            },
            OpenBlock::Thinking(_) => AssistantMessageEvent::ThinkingEnd {
                partial,
                content_index: idx,
            },
            OpenBlock::ToolCall(_) => AssistantMessageEvent::ToolcallEnd {
                partial,
                content_index: idx,
            },
        });
        out
    }

    fn on_message_delta(&mut self, value: &Value) {
        if let Some(delta) = value.get("delta") {
            if let Some(reason) = str_at(delta, "stop_reason") {
                self.stop_reason = Some(reason.to_string());
            }
            // S-4: the refusal explanation the genai seam destroys.
            if let Some(details) = delta.get("stop_details")
                && let Some(explanation) = str_at(details, "explanation")
            {
                self.stop_explanation = Some(explanation.to_string());
            }
        }
        if let Some(usage) = value.get("usage") {
            self.usage.apply(usage);
        }
    }

    /// Close any still-open blocks and emit the terminal event.
    pub fn finish(&mut self) -> Vec<AssistantMessageEvent> {
        let mut out = Vec::new();
        self.ensure_started(&mut out);
        self.close_dangling(&mut out);

        let mut result = self.base.clone();
        result.content = std::mem::take(&mut self.blocks);
        result.usage = self.usage.to_usage();
        result.timestamp = now_ms();

        let resolution = resolve_stop(
            self.stop_reason.as_deref(),
            self.stop_explanation.as_deref(),
            &result.content,
        );
        result.stop_reason = resolution.0;
        self.finished = true;
        match resolution.1 {
            Some(msg) => {
                result.error_message = Some(msg.clone());
                out.push(AssistantMessageEvent::Error {
                    error: msg,
                    result,
                });
            }
            None => out.push(AssistantMessageEvent::Done { result }),
        }
        out
    }

    /// Terminal error carrying everything accumulated so far (upstream parity:
    /// pi-ai's error event carries the partial assistant message).
    pub fn into_error(&mut self, msg: impl Into<String>) -> Vec<AssistantMessageEvent> {
        let msg = msg.into();
        let mut out = Vec::new();
        self.ensure_started(&mut out);
        self.close_dangling(&mut out);

        let mut result = self.base.clone();
        result.content = std::mem::take(&mut self.blocks);
        result.usage = self.usage.to_usage();
        result.stop_reason = StopReason::Error;
        result.error_message = Some(msg.clone());
        result.timestamp = now_ms();
        self.finished = true;
        out.push(AssistantMessageEvent::Error {
            error: msg,
            result,
        });
        out
    }

    /// Terminal aborted event (cancellation).
    pub fn into_aborted(&mut self) -> Vec<AssistantMessageEvent> {
        let mut out = Vec::new();
        self.ensure_started(&mut out);
        self.close_dangling(&mut out);

        let mut result = self.base.clone();
        result.content = std::mem::take(&mut self.blocks);
        result.usage = self.usage.to_usage();
        result.stop_reason = StopReason::Aborted;
        result.error_message = Some("aborted".into());
        result.timestamp = now_ms();
        self.finished = true;
        out.push(AssistantMessageEvent::Error {
            error: "aborted".into(),
            result,
        });
        out
    }

    /// Emit `*_end` for blocks the provider never closed (truncated stream).
    fn close_dangling(&mut self, out: &mut Vec<AssistantMessageEvent>) {
        if self.open.is_empty() {
            return;
        }
        let mut still_open: Vec<(u64, OpenBlock)> =
            self.open.drain().collect();
        // Deterministic order by the provider's own block index.
        still_open.sort_by_key(|(wire_index, _)| *wire_index);
        for (_, open) in still_open {
            let idx = open.index();
            let partial = self.partial();
            out.push(match open {
                OpenBlock::Text(_) => AssistantMessageEvent::TextEnd {
                    partial,
                    content_index: idx,
                },
                OpenBlock::Thinking(_) => AssistantMessageEvent::ThinkingEnd {
                    partial,
                    content_index: idx,
                },
                OpenBlock::ToolCall(_) => AssistantMessageEvent::ToolcallEnd {
                    partial,
                    content_index: idx,
                },
            });
        }
    }
}

/// Parse accumulated tool arguments, applying the same repair ladder the genai
/// path uses (`crate::mapping::json_repair`). A buffer that is truncated
/// rather than merely malformed stays `Value::String`, which the request
/// builder recognizes as corrupt and drops — deliberately *not* upstream's
/// partial-object coercion. See `crate::mapping::json_repair` module docs.
fn parse_tool_args(raw: &str) -> Value {
    if raw.is_empty() {
        return Value::Object(Default::default());
    }
    parse_json_with_repair(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// Map Anthropic's raw `stop_reason` onto grain semantics.
///
/// Mirrors upstream pi-ai's `anthropic-messages.ts` table exactly, and matches
/// the genai path's `resolve_stop_reason` so the two backends agree:
/// `end_turn`/`stop_sequence` → Stop, `max_tokens` → Length, `tool_use` →
/// ToolUse, `pause_turn` → Stop (resubmittable), `refusal` → Error with
/// `stop_details.explanation` when present, anything else → Error with
/// `Provider stopped with: <raw>`.
fn resolve_stop(
    raw: Option<&str>,
    explanation: Option<&str>,
    content: &[AssistantContent],
) -> (StopReason, Option<String>) {
    let Some(raw) = raw else {
        // No stop_reason seen: infer from content, as the genai path does.
        let inferred = if content
            .iter()
            .any(|c| matches!(c, AssistantContent::ToolCall(_)))
        {
            StopReason::ToolUse
        } else {
            StopReason::Stop
        };
        return (inferred, None);
    };

    match raw {
        "end_turn" | "stop_sequence" | "pause_turn" => (StopReason::Stop, None),
        "max_tokens" => (StopReason::Length, None),
        "tool_use" => (StopReason::ToolUse, None),
        "refusal" => (
            StopReason::Error,
            Some(
                explanation
                    .unwrap_or("The model refused to complete the request")
                    .to_string(),
            ),
        ),
        other => (
            StopReason::Error,
            Some(format!("Provider stopped with: {other}")),
        ),
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- S-3: usage replacement, the whole point of this transport ----------

    #[test]
    fn message_delta_usage_replaces_rather_than_adds() {
        let mut u = UsageAccumulator::default();
        u.apply(&json!({
            "input_tokens": 12, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0
        }));
        // The live shape: message_delta repeats input_tokens.
        u.apply(&json!({
            "input_tokens": 12, "output_tokens": 5,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0
        }));
        let usage = u.to_usage();
        assert_eq!(
            (usage.input, usage.output, usage.total_tokens),
            (12, 5, 17),
            "genai's backend reports 24/5/29 here; replacement is the fix"
        );
    }

    #[test]
    fn message_delta_without_usage_preserves_message_start() {
        let mut u = UsageAccumulator::default();
        u.apply(&json!({"input_tokens": 12, "output_tokens": 0}));
        // No `usage` object at all -> apply is never called; nothing changes.
        let usage = u.to_usage();
        assert_eq!((usage.input, usage.output, usage.total_tokens), (12, 0, 12));
    }

    #[test]
    fn absent_fields_do_not_clobber_earlier_values() {
        let mut u = UsageAccumulator::default();
        u.apply(&json!({"input_tokens": 100, "cache_read_input_tokens": 20}));
        // A delta carrying only output must not zero the input side.
        u.apply(&json!({"output_tokens": 7}));
        let usage = u.to_usage();
        assert_eq!(usage.input, 120, "cache-inclusive input preserved");
        assert_eq!(usage.cache_read, 20);
        assert_eq!(usage.output, 7);
        assert_eq!(usage.total_tokens, 127);
    }

    #[test]
    fn cache_counters_fold_into_grain_cache_inclusive_input() {
        let mut u = UsageAccumulator::default();
        u.apply(&json!({
            "input_tokens": 10, "output_tokens": 3,
            "cache_read_input_tokens": 5, "cache_creation_input_tokens": 2
        }));
        let usage = u.to_usage();
        assert_eq!(usage.input, 17, "10 + 5 read + 2 write");
        assert_eq!(usage.cache_read, 5);
        assert_eq!(usage.cache_write, 2);
        assert_eq!(usage.total_tokens, 20);
    }

    // -- stop-reason mapping -------------------------------------------------

    #[test]
    fn stop_reason_table_matches_upstream() {
        assert_eq!(resolve_stop(Some("end_turn"), None, &[]).0, StopReason::Stop);
        assert_eq!(
            resolve_stop(Some("stop_sequence"), None, &[]).0,
            StopReason::Stop
        );
        assert_eq!(
            resolve_stop(Some("pause_turn"), None, &[]).0,
            StopReason::Stop
        );
        assert_eq!(
            resolve_stop(Some("max_tokens"), None, &[]).0,
            StopReason::Length
        );
        assert_eq!(
            resolve_stop(Some("tool_use"), None, &[]).0,
            StopReason::ToolUse
        );
        assert_eq!(
            resolve_stop(Some("sensitive"), None, &[]),
            (
                StopReason::Error,
                Some("Provider stopped with: sensitive".to_string())
            )
        );
    }

    /// S-4: the explanation the genai seam drops.
    #[test]
    fn refusal_uses_stop_details_explanation_when_present() {
        assert_eq!(
            resolve_stop(Some("refusal"), Some("because reasons"), &[]),
            (StopReason::Error, Some("because reasons".to_string()))
        );
        assert_eq!(
            resolve_stop(Some("refusal"), None, &[]),
            (
                StopReason::Error,
                Some("The model refused to complete the request".to_string())
            )
        );
    }

    // -- S-8: block boundaries ----------------------------------------------

    fn model() -> Model {
        Model {
            id: "anthropic/claude-haiku-4-5".into(),
            name: "h".into(),
            api: "anthropic-messages".into(),
            provider: "anthropic".into(),
            ..Default::default()
        }
    }

    fn tags(events: &[AssistantMessageEvent]) -> Vec<&'static str> {
        events
            .iter()
            .map(|e| match e {
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
            })
            .collect()
    }

    /// The case the genai seam cannot represent at all: a block that opens and
    /// closes with no delta still produces its start/end pair.
    #[test]
    fn empty_block_still_emits_start_and_end() {
        let mut s = AnthropicState::new(&model());
        let mut events = Vec::new();
        events.extend(s.on_frame(
            "message_start",
            &json!({"message":{"id":"m","usage":{"input_tokens":1}}}).to_string(),
        ));
        events.extend(s.on_frame(
            "content_block_start",
            &json!({"index":0,"content_block":{"type":"text","text":""}}).to_string(),
        ));
        events.extend(s.on_frame("content_block_stop", &json!({"index":0}).to_string()));
        events.extend(s.on_frame("message_stop", &json!({}).to_string()));

        assert_eq!(tags(&events), vec!["Start", "TextStart", "TextEnd", "Done"]);
    }

    #[test]
    fn dangling_block_is_closed_at_the_terminal() {
        let mut s = AnthropicState::new(&model());
        s.on_frame(
            "content_block_start",
            &json!({"index":0,"content_block":{"type":"text","text":""}}).to_string(),
        );
        // No content_block_stop — the stream just ends.
        let events = s.finish();
        assert!(
            tags(&events).contains(&"TextEnd"),
            "a truncated stream must still close its open block: {:?}",
            tags(&events)
        );
    }

    #[test]
    fn interleaved_block_indices_map_independently() {
        let mut s = AnthropicState::new(&model());
        let mut events = Vec::new();
        events.extend(s.on_frame(
            "content_block_start",
            &json!({"index":0,"content_block":{"type":"text","text":""}}).to_string(),
        ));
        events.extend(s.on_frame(
            "content_block_delta",
            &json!({"index":0,"delta":{"type":"text_delta","text":"a"}}).to_string(),
        ));
        events.extend(s.on_frame("content_block_stop", &json!({"index":0}).to_string()));
        events.extend(s.on_frame(
            "content_block_start",
            &json!({"index":1,"content_block":{"type":"tool_use","id":"t1","name":"read"}})
                .to_string(),
        ));
        events.extend(s.on_frame(
            "content_block_delta",
            &json!({"index":1,"delta":{"type":"input_json_delta","partial_json":"{\"p\":1}"}})
                .to_string(),
        ));
        events.extend(s.on_frame("content_block_stop", &json!({"index":1}).to_string()));
        events.extend(s.on_frame("message_stop", &json!({}).to_string()));

        assert_eq!(
            tags(&events),
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
        let AssistantMessageEvent::Done { result } = events.last().unwrap() else {
            panic!("expected Done");
        };
        assert_eq!(result.content.len(), 2);
        assert!(matches!(result.content[0], AssistantContent::Text(_)));
        match &result.content[1] {
            AssistantContent::ToolCall(tc) => assert_eq!(tc.arguments, json!({"p": 1})),
            other => panic!("expected a tool call, got {other:?}"),
        }
    }

    /// S-5 frame half: a frame malformed only in its string literals is
    /// repaired instead of aborting the turn.
    #[test]
    fn malformed_frame_is_repaired_not_fatal() {
        let mut s = AnthropicState::new(&model());
        s.on_frame(
            "content_block_start",
            &json!({"index":0,"content_block":{"type":"tool_use","id":"t","name":"edit"}})
                .to_string(),
        );
        // Raw TAB inside a string literal — invalid JSON as-is.
        let malformed =
            "{\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"t\\\":\\\"a\tb\\\"}\"}}";
        let events = s.on_frame("content_block_delta", malformed);
        assert_eq!(tags(&events), vec!["ToolcallDelta"]);
    }

    #[test]
    fn unparseable_frame_is_ignored_rather_than_fatal() {
        let mut s = AnthropicState::new(&model());
        assert!(s.on_frame("content_block_delta", "not json at all").is_empty());
        assert!(!s.is_finished(), "an unparseable frame must not end the turn");
    }

    // -- S-2 / S-6: captured here, but no core slot -------------------------

    #[test]
    fn response_id_and_model_are_captured_even_though_grain_cannot_carry_them() {
        let mut s = AnthropicState::new(&model());
        s.on_frame(
            "message_start",
            &json!({"message":{"id":"msg_abc","model":"claude-haiku-4-5-20991231",
                    "usage":{"input_tokens":1}}})
            .to_string(),
        );
        assert_eq!(s.response_id.as_deref(), Some("msg_abc"));
        assert_eq!(
            s.response_model.as_deref(),
            Some("claude-haiku-4-5-20991231")
        );
    }

    /// W4: a `redacted_thinking` block carries only an opaque `data` payload.
    /// Reading `thinking`/`signature` yields an empty, unsigned block, which
    /// the replay path then drops — losing the block entirely.
    #[test]
    fn redacted_thinking_preserves_its_opaque_payload() {
        let mut s = AnthropicState::new(&model());
        let events = s.on_frame(
            "content_block_start",
            &json!({
                "index": 0,
                "content_block": {"type": "redacted_thinking", "data": "EroBCkYIB..."}
            })
            .to_string(),
        );
        assert_eq!(tags(&events), vec!["Start", "ThinkingStart"]);
        s.on_frame("content_block_stop", &json!({"index":0}).to_string());
        let done = s.finish();

        let AssistantMessageEvent::Done { result } = done.last().unwrap() else {
            panic!("expected Done");
        };
        match &result.content[0] {
            AssistantContent::Thinking(t) => {
                let meta = t
                    .provider_metadata
                    .as_ref()
                    .expect("redacted payload must be preserved");
                assert_eq!(meta["type"], json!("redacted_thinking"));
                assert_eq!(
                    meta["data"],
                    json!("EroBCkYIB..."),
                    "the opaque data is the entire content of the block"
                );
            }
            other => panic!("expected a thinking block, got {other:?}"),
        }
    }

    #[test]
    fn provider_error_event_terminates_with_its_message() {
        let mut s = AnthropicState::new(&model());
        let events = s.on_frame(
            "error",
            &json!({"error":{"type":"overloaded_error","message":"Overloaded"}}).to_string(),
        );
        assert_eq!(tags(&events), vec!["Start", "Error"]);
        let AssistantMessageEvent::Error { error, result } = events.last().unwrap() else {
            panic!("expected Error");
        };
        assert_eq!(error, "Overloaded");
        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(s.is_finished());
    }
}
