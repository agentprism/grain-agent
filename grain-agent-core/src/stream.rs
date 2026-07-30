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
    dyn Fn(serde_json::Value, Model) -> BoxFuture<'static, Option<serde_json::Value>> + Send + Sync,
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
    /// Preferred transport for providers supporting more than one.
    /// Upstream's `Transport` union is `"sse" | "websocket" |
    /// "websocket-cached" | "auto"` (pi-ai `types.ts:103`); carried here as a
    /// free string so an adapter can accept transports grain does not know.
    pub transport: Option<String>,
    /// Sampling temperature (pi-ai `StreamOptions.temperature`,
    /// `types.ts:117`).
    pub temperature: Option<f32>,
    /// Cap on tokens generated for this request (pi-ai
    /// `StreamOptions.maxTokens`, `types.ts:118`).
    pub max_tokens: Option<u64>,
    /// Prompt-cache retention preference (pi-ai
    /// `StreamOptions.cacheRetention`, `types.ts:136`). Upstream default is
    /// [`CacheRetention::Short`]; `None` leaves the choice to the adapter.
    pub cache_retention: Option<CacheRetention>,
    /// Max client-side retry attempts (pi-ai `StreamOptions.maxRetries`,
    /// `types.ts:177`). Upstream notes the OpenAI/Anthropic SDKs default to 2.
    pub max_retries: Option<u32>,
    /// Cap on how long a server-requested retry delay may be before the
    /// request fails outright (pi-ai `StreamOptions.maxRetryDelayMs`,
    /// `types.ts:185`). Upstream default 60000; 0 disables the cap.
    pub max_retry_delay_ms: Option<u64>,
    /// HTTP request timeout (pi-ai `StreamOptions.timeoutMs`,
    /// `types.ts:166`).
    pub timeout_ms: Option<u64>,
    /// WebSocket connect/open-handshake timeout; stream idleness after
    /// connect uses [`Self::timeout_ms`] (pi-ai
    /// `StreamOptions.websocketConnectTimeoutMs`, `types.ts:172`).
    pub websocket_connect_timeout_ms: Option<u64>,
    /// Extra HTTP headers, merged over provider defaults (pi-ai
    /// `StreamOptions.headers` / `ProviderHeaders`, `types.ts:161` and
    /// `types.ts:107`). Upstream's type is `Record<string, string | null>`
    /// where a **`None` value suppresses** a provider default header of the
    /// same name — hence `Option<String>` rather than `String`.
    pub headers: Option<HashMap<String, Option<String>>>,
    /// Request metadata; adapters take the fields they understand and ignore
    /// the rest (pi-ai `StreamOptions.metadata`, `types.ts:191`). Upstream
    /// notes Anthropic reads `user_id` for abuse tracking and rate limiting.
    pub metadata: Option<serde_json::Value>,
    /// Provider-scoped environment values, taking precedence over the
    /// process environment for provider configuration such as regional
    /// settings, endpoint placeholders and proxy variables (pi-ai
    /// `StreamOptions.env` / `ProviderEnv`, `types.ts:197` and `types.ts:106`).
    pub env: Option<HashMap<String, String>>,
    /// Payload inspection/replacement callback.
    pub on_payload: Option<OnPayloadFn>,
    /// Response-received callback.
    pub on_response: Option<OnResponseFn>,
    /// Provider-specific extras forwarded as opaque JSON.
    ///
    /// Upstream widens per-call options structurally
    /// (`ProviderStreamOptions = StreamOptions & Record<string, unknown>`,
    /// pi-ai `types.ts:200`); Rust has no structural widening, so this is the
    /// escape hatch for provider-specific keys with no typed slot above.
    pub extra: serde_json::Value,
}

/// Prompt-cache retention preference.
///
/// Port of pi-ai `CacheRetention` (`packages/ai/src/types.ts:101` @ 34239180:
/// `"none" | "short" | "long"`). Providers map these onto their own supported
/// values; upstream documents the default as `"short"`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
    None,
    #[default]
    Short,
    Long,
}

impl std::fmt::Debug for StreamOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamOptions")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("reasoning", &self.reasoning)
            .field("thinking_budgets", &self.thinking_budgets)
            .field("session_id", &self.session_id)
            .field("transport", &self.transport)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("cache_retention", &self.cache_retention)
            .field("max_retries", &self.max_retries)
            .field("max_retry_delay_ms", &self.max_retry_delay_ms)
            .field("timeout_ms", &self.timeout_ms)
            .field(
                "websocket_connect_timeout_ms",
                &self.websocket_connect_timeout_ms,
            )
            // Header *names* are safe to print; values can carry credentials.
            .field(
                "headers",
                &self
                    .headers
                    .as_ref()
                    .map(|h| h.keys().cloned().collect::<Vec<_>>()),
            )
            .field("metadata", &self.metadata)
            // Values can carry credentials; names alone are enough to debug.
            .field(
                "env",
                &self
                    .env
                    .as_ref()
                    .map(|e| e.keys().cloned().collect::<Vec<_>>()),
            )
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
    /// A failure carrying a structured [`ErrorCode`] alongside its message.
    /// The loop's degrade-gracefully path preserves the code onto the
    /// synthesized error assistant message (`error_code`), so embedders can
    /// classify the failure without parsing `error_message` text.
    #[error("{message}")]
    Coded {
        code: crate::types::ErrorCode,
        message: String,
    },
}

impl StreamError {
    /// Convenience constructor wrapping a message into [`StreamError::Other`].
    pub fn msg(s: impl Into<String>) -> Self {
        StreamError::Other(s.into())
    }

    /// Convenience constructor for [`StreamError::Coded`].
    pub fn coded(code: crate::types::ErrorCode, message: impl Into<String>) -> Self {
        StreamError::Coded {
            code,
            message: message.into(),
        }
    }

    /// The structured code, when this error carries one.
    pub fn code(&self) -> Option<&crate::types::ErrorCode> {
        match self {
            StreamError::Coded { code, .. } => Some(code),
            _ => None,
        }
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
