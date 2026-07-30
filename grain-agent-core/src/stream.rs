//! Stream abstraction for LLM provider integration.
//!
//! The agent loop is parameterized over [`LlmStream`] so the core crate has no
//! dependency on any concrete LLM SDK. Apps inject an implementation; a default
//! provider (e.g. Anthropic, OpenAI) lives in a separate crate.
//!
//! This corresponds to the `streamFn` injection point in the TypeScript
//! `@earendil-works/pi-agent-core` package.

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use tokio_util::sync::CancellationToken;

use crate::types::{AssistantMessageEvent, LlmContext, Model, ThinkingBudgets, ThinkingLevel};

/// Boxed stream of streaming-protocol events ending with `Done` or `Error`.
pub type AssistantStream = BoxStream<'static, AssistantMessageEvent>;

/// HTTP response metadata surfaced to [`StreamOptions::on_response`].
///
/// Port of the pi-ai `ProviderResponse` (`types.ts:111-114` @ 34239180).
#[derive(Debug, Clone, Default)]
pub struct ProviderResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
}

/// Callback for inspecting or replacing provider payloads before sending.
/// Return `None` to keep the payload unchanged (the upstream `onPayload`
/// "return undefined" contract, pi-ai `types.ts:143-147`).
pub type OnPayloadFn = Arc<
    dyn Fn(serde_json::Value, Model) -> BoxFuture<'static, Option<serde_json::Value>>
        + Send
        + Sync,
>;

/// Callback invoked after an HTTP response is received and before its body
/// stream is consumed (the upstream `onResponse`, pi-ai `types.ts:148-152`).
pub type OnResponseFn =
    Arc<dyn Fn(ProviderResponse, Model) -> BoxFuture<'static, ()> + Send + Sync>;

/// Options passed to an [`LlmStream`] implementation for a single request.
///
/// Mirrors the pi-ai `SimpleStreamOptions` surface the agent forwards
/// (`types.ts:116-198, 305-310`); provider adapters honor the fields they
/// understand and ignore the rest.
#[derive(Default, Clone)]
pub struct StreamOptions {
    pub api_key: Option<String>,
    pub reasoning: Option<ThinkingLevel>,
    /// Per-level thinking token budgets (token-based providers only).
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub session_id: Option<String>,
    pub transport: Option<String>,
    pub max_retry_delay_ms: Option<u64>,
    /// Payload inspection/replacement callback.
    pub on_payload: Option<OnPayloadFn>,
    /// Response-received callback.
    pub on_response: Option<OnResponseFn>,
    /// Provider-specific extras forwarded as opaque JSON.
    pub extra: serde_json::Value,
}

impl std::fmt::Debug for StreamOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamOptions")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("reasoning", &self.reasoning)
            .field("thinking_budgets", &self.thinking_budgets)
            .field("session_id", &self.session_id)
            .field("transport", &self.transport)
            .field("max_retry_delay_ms", &self.max_retry_delay_ms)
            .field("on_payload", &self.on_payload.as_ref().map(|_| "<fn>"))
            .field("on_response", &self.on_response.as_ref().map(|_| "<fn>"))
            .field("extra", &self.extra)
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("{0}")]
    Other(String),
    #[error("aborted")]
    Aborted,
}

impl StreamError {
    /// Convenience constructor wrapping a message into [`StreamError::Other`].
    pub fn msg(s: impl Into<String>) -> Self {
        StreamError::Other(s.into())
    }
}

/// Trait implemented by an LLM provider adapter.
///
/// Contract (mirroring the TS `StreamFn`):
/// - Must not panic or return an `Err` for request/model/runtime failures.
/// - Must surface failures in the returned stream via a terminal
///   [`AssistantMessageEvent::Error`] (or [`AssistantMessageEvent::Done`])
///   carrying a final [`crate::types::AssistantMessage`] with
///   [`crate::types::StopReason::Error`] or [`crate::types::StopReason::Aborted`]
///   and a populated `error_message`.
/// - The stream MUST end with exactly one terminal event.
#[async_trait::async_trait]
pub trait LlmStream: Send + Sync {
    async fn stream(
        &self,
        model: &Model,
        context: &LlmContext,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> Result<AssistantStream, StreamError>;
}

/// Shared, type-erased stream handle.
pub type StreamFn = Arc<dyn LlmStream>;
