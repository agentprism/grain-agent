//! Shared helpers for the ported pi-agent-core loop test-suite.
//!
//! Mirrors the fixtures at the top of upstream
//! `packages/agent/test/agent-loop.test.ts` / `agent.test.ts`
//! (pinned upstream commit 34239180): `createModel`, `createAssistantMessage`,
//! `createUserMessage`, `identityConverter`, and the `MockAssistantStream`
//! pattern (a stream that pushes a single terminal `done` event).
#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::BoxFuture;
use grain_agent_core::{
    AgentEvent, AgentMessage, AgentTool, AgentToolError, AgentToolResult, AssistantContent,
    AssistantMessage, AssistantMessageEvent, AssistantStream, ConvertToLlmFn, Cost, EventSink,
    LlmContext, LlmStream, Message, Model, StopReason, StreamError, StreamFn, StreamOptions,
    TextContent, ToolCall, ToolDefinition, ToolExecutionMode, ToolUpdateCallback, Usage,
    UserContent, UserMessage,
};
use tokio_util::sync::CancellationToken;

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Port of upstream `createModel()`.
pub fn create_model() -> Model {
    Model {
        id: "mock".into(),
        name: "mock".into(),
        api: "openai-responses".into(),
        provider: "openai".into(),
        base_url: "https://example.invalid".into(),
        reasoning: false,
        context_window: 8192,
        max_tokens: 2048,
        cost: Cost::default(),
    }
}

/// Port of upstream `createAssistantMessage(content, stopReason)`.
pub fn create_assistant_message(
    content: Vec<AssistantContent>,
    stop_reason: StopReason,
) -> AssistantMessage {
    AssistantMessage {
        content,
        api: "openai-responses".into(),
        provider: "openai".into(),
        model: "mock".into(),
        usage: Usage::default(),
        stop_reason,
        error_message: None,
        error_code: None,
        timestamp: now_ms(),
        raw_stop_reason: None,
    }
}

pub fn text(text: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(TextContent { text: text.into() })
}

pub fn tool_call(
    id: impl Into<String>,
    name: impl Into<String>,
    arguments: serde_json::Value,
) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: id.into(),
        name: name.into(),
        arguments,
        thought_signature: None,
    })
}

/// Port of upstream `createUserMessage(text)`.
///
/// Upstream uses a plain string as `UserMessage.content`; the Rust
/// `UserMessage` only models structured content, so this builds the
/// equivalent single-text-block form. The string-content wire shape itself
/// is covered by `user_message_string_content_wire_format` (patch-7).
pub fn create_user_message(text: &str) -> AgentMessage {
    AgentMessage::user(UserMessage {
        content: vec![UserContent::text(text)],
        timestamp: now_ms(),
    })
}

/// Port of upstream `identityConverter`: pass standard messages through,
/// drop anything that is not user / assistant / toolResult.
pub fn identity_converter() -> ConvertToLlmFn {
    Arc::new(|messages: Vec<AgentMessage>| {
        Box::pin(async move {
            messages
                .into_iter()
                .filter_map(|m| match m {
                    AgentMessage::Standard(m) => Some(m),
                    AgentMessage::Custom(_) => None,
                })
                .collect()
        })
    })
}

/// Event sink that records every emitted event in order.
pub fn recording_sink() -> (EventSink, Arc<StdMutex<Vec<AgentEvent>>>) {
    let events: Arc<StdMutex<Vec<AgentEvent>>> = Arc::new(StdMutex::new(Vec::new()));
    let sink_events = events.clone();
    let sink: EventSink = Arc::new(move |event: AgentEvent| {
        let events = sink_events.clone();
        Box::pin(async move {
            events.lock().unwrap().push(event);
        })
    });
    (sink, events)
}

/// Snake-case tag for an event, matching the TS `event.type` strings.
pub fn event_type(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::AgentStart => "agent_start",
        AgentEvent::AgentEnd { .. } => "agent_end",
        AgentEvent::TurnStart => "turn_start",
        AgentEvent::TurnEnd { .. } => "turn_end",
        AgentEvent::MessageStart { .. } => "message_start",
        AgentEvent::MessageUpdate { .. } => "message_update",
        AgentEvent::MessageEnd { .. } => "message_end",
        AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
        AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
        AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
    }
}

pub fn event_types(events: &[AgentEvent]) -> Vec<&'static str> {
    events.iter().map(event_type).collect()
}

/// First text block of a user message, if any.
pub fn user_text(message: &UserMessage) -> Option<String> {
    message.content.iter().find_map(|c| match c {
        UserContent::Text(t) => Some(t.text.clone()),
        UserContent::Image(_) => None,
    })
}

/// First text block of a tool result / `AgentToolResult` content list.
pub fn content_text(content: &[UserContent]) -> Option<String> {
    content.iter().find_map(|c| match c {
        UserContent::Text(t) => Some(t.text.clone()),
        UserContent::Image(_) => None,
    })
}

// ---------------------------------------------------------------------------
// Stream mock: per-call factory, mirroring the TS `streamFn` closures that
// return a `MockAssistantStream`.
// ---------------------------------------------------------------------------

type StreamFactory = dyn Fn(usize, &Model, &LlmContext, &StreamOptions, CancellationToken) -> AssistantStream
    + Send
    + Sync;

pub struct FnStream {
    calls: AtomicUsize,
    factory: Box<StreamFactory>,
}

impl FnStream {
    pub fn new(
        factory: impl Fn(
            usize,
            &Model,
            &LlmContext,
            &StreamOptions,
            CancellationToken,
        ) -> AssistantStream
        + Send
        + Sync
        + 'static,
    ) -> Arc<Self> {
        Arc::new(FnStream {
            calls: AtomicUsize::new(0),
            factory: Box::new(factory),
        })
    }

    /// Number of `stream()` invocations so far (TS tests' `callIndex` /
    /// `llmCalls` counters).
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmStream for FnStream {
    async fn stream(
        &self,
        model: &Model,
        context: &LlmContext,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> Result<AssistantStream, StreamError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        Ok((self.factory)(n, model, context, options, cancel))
    }
}

/// Terminal-only stream: a single `done` event, exactly like the TS
/// `MockAssistantStream` usage in the loop tests (no `start` event pushed).
pub fn done_stream(message: AssistantMessage) -> AssistantStream {
    futures::stream::iter(vec![AssistantMessageEvent::Done { result: message }]).boxed()
}

/// Port of upstream `unusedStreamFunction`: panics if the loop ever asks for
/// an LLM turn.
pub fn unused_stream_fn() -> StreamFn {
    FnStream::new(|_, _, _, _, _| panic!("Unexpected stream call"))
}

// ---------------------------------------------------------------------------
// Closure-driven test tool
// ---------------------------------------------------------------------------

pub type ToolExecuteFn = Arc<
    dyn Fn(
            String,
            serde_json::Value,
            CancellationToken,
            ToolUpdateCallback,
        ) -> BoxFuture<'static, Result<AgentToolResult, AgentToolError>>
        + Send
        + Sync,
>;

pub type ToolPrepareFn =
    Arc<dyn Fn(serde_json::Value) -> Result<serde_json::Value, AgentToolError> + Send + Sync>;

pub struct TestTool {
    def: ToolDefinition,
    prepare: Option<ToolPrepareFn>,
    exec: ToolExecuteFn,
}

impl TestTool {
    pub fn new(
        name: &str,
        label: &str,
        description: &str,
        parameters: serde_json::Value,
        exec: ToolExecuteFn,
    ) -> Self {
        TestTool {
            def: ToolDefinition {
                name: name.into(),
                label: label.into(),
                description: description.into(),
                parameters,
                execution_mode: None,
            },
            prepare: None,
            exec,
        }
    }

    pub fn with_execution_mode(mut self, mode: ToolExecutionMode) -> Self {
        self.def.execution_mode = Some(mode);
        self
    }

    pub fn with_prepare(mut self, prepare: ToolPrepareFn) -> Self {
        self.prepare = Some(prepare);
        self
    }

    pub fn arc(self) -> Arc<dyn AgentTool> {
        Arc::new(self)
    }
}

#[async_trait]
impl AgentTool for TestTool {
    fn definition(&self) -> &ToolDefinition {
        &self.def
    }

    fn prepare_arguments(
        &self,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, AgentToolError> {
        match &self.prepare {
            Some(prepare) => prepare(args),
            None => Ok(args),
        }
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        args: serde_json::Value,
        cancel: CancellationToken,
        on_update: ToolUpdateCallback,
    ) -> Result<AgentToolResult, AgentToolError> {
        (self.exec)(tool_call_id.to_string(), args, cancel, on_update).await
    }
}

/// JSON schema for the upstream `Type.Object({ value: Type.String() })` tool.
pub fn value_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": { "value": { "type": "string" } },
        "required": ["value"]
    })
}

/// Upstream `echo` tool: records executed values and echoes them back.
pub fn echo_tool(executed: Arc<StdMutex<Vec<String>>>) -> Arc<dyn AgentTool> {
    TestTool::new(
        "echo",
        "Echo",
        "Echo tool",
        value_schema(),
        Arc::new(move |_id, args, _cancel, _on_update| {
            let executed = executed.clone();
            Box::pin(async move {
                let value = args
                    .get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                executed.lock().unwrap().push(value.clone());
                Ok(AgentToolResult {
                    content: vec![UserContent::text(format!("echoed: {value}"))],
                    details: serde_json::json!({ "value": value }),
                    terminate: None,
                    ..Default::default()
                })
            })
        }),
    )
    .arc()
}

/// Roles of a message list, matching the TS `messages.map((m) => m.role)`.
pub fn roles(messages: &[AgentMessage]) -> Vec<String> {
    messages.iter().map(|m| m.role().to_string()).collect()
}

/// Extract the tool-call ids from `tool_execution_end` events, in emission order.
pub fn tool_execution_end_ids(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect()
}

/// Extract tool-result ids from `message_end` events, in emission order.
pub fn message_end_tool_result_ids(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::MessageEnd {
                message: AgentMessage::Standard(Message::ToolResult(tr)),
            } => Some(tr.tool_call_id.clone()),
            _ => None,
        })
        .collect()
}
