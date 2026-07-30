//! [`grain_agent_core::LlmStream`] backed by a native Anthropic Messages
//! transport.
//!
//! # Status: opt-in, fixture-verified, **not live-verified**
//!
//! This backend exists because several structural gaps at the genai seam are
//! unfixable above it — see `tests/SEAM-VECTORS.md` §6 and
//! `../UPSTREAM-GENAI.md`. It is **not** the default: [`crate::GenaiStream`]
//! still routes Anthropic through genai unless the native transport is
//! explicitly selected via
//! [`crate::GenaiStreamBuilder::with_native_anthropic_transport`].
//!
//! **Everything asserted about this transport is asserted against recorded
//! fixtures** (upstream pi-ai's own stream-parsing fixtures, replayed from a
//! local socket). No assertion here has been confirmed against the live
//! Anthropic API. Fixture parity proves the *decode* path reproduces upstream;
//! it proves nothing about whether the live API accepts our *request* shape.
//! Live verification is scheduled separately.
//!
//! Promoting this to the default is deliberately a one-line change (flip the
//! default in [`crate::GenaiStreamBuilder`]) plus a test run — no migration.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use grain_agent_core::{
    AssistantStream, LlmContext, LlmStream, Model, StreamError, StreamOptions,
};
use tokio_util::sync::CancellationToken;

use crate::anthropic::request::{ANTHROPIC_VERSION, build_request};
use crate::anthropic::state::AnthropicState;
use crate::anthropic::wire::SseDecoder;

/// How the transport authenticates to Anthropic.
#[derive(Debug, Clone)]
pub enum AnthropicAuth {
    /// Standard API key — sent as `x-api-key`.
    ApiKey(String),
    /// OAuth access token (Claude Pro/Max) — sent as `authorization: Bearer`
    /// with the OAuth beta header.
    ///
    /// Wiring the OAuth *login* flow is owned elsewhere (debt item G7); this
    /// variant only carries an already-obtained token.
    OauthToken(String),
}

/// Configuration for [`AnthropicStream`].
#[derive(Debug, Clone)]
pub struct AnthropicTransportConfig {
    /// Base URL, including trailing slash (e.g. `https://api.anthropic.com/v1/`).
    pub base_url: String,
    pub auth: AnthropicAuth,
}

impl AnthropicTransportConfig {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.anthropic.com/v1/";

    pub fn with_api_key(key: impl Into<String>) -> Self {
        AnthropicTransportConfig {
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            auth: AnthropicAuth::ApiKey(key.into()),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let mut url = base_url.into();
        if !url.ends_with('/') {
            url.push('/');
        }
        self.base_url = url;
        self
    }
}

/// Native Anthropic Messages streaming backend.
pub struct AnthropicStream {
    http: reqwest13::Client,
    config: Arc<AnthropicTransportConfig>,
}

impl AnthropicStream {
    pub fn new(config: AnthropicTransportConfig) -> Self {
        AnthropicStream {
            http: reqwest13::Client::new(),
            config: Arc::new(config),
        }
    }

    /// Construct with a caller-supplied HTTP client (proxy bypass, timeouts).
    pub fn with_client(http: reqwest13::Client, config: AnthropicTransportConfig) -> Self {
        AnthropicStream {
            http,
            config: Arc::new(config),
        }
    }

    fn messages_url(&self) -> String {
        format!("{}messages", self.config.base_url)
    }
}

/// Build a one-shot stream carrying a single terminal error, preserving the
/// `LlmStream` contract that runtime failures are reported as events rather
/// than `Err`.
fn one_shot_error(model: &Model, msg: String) -> AssistantStream {
    let mut state = AnthropicState::new(model);
    let events = state.into_error(msg);
    Box::pin(futures::stream::iter(events))
}

#[async_trait]
impl LlmStream for AnthropicStream {
    async fn stream(
        &self,
        model: &Model,
        context: &LlmContext,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> Result<AssistantStream, StreamError> {
        // Unsupported request knobs fail loudly, before any network I/O.
        let body = match build_request(model, context, options) {
            Ok(body) => body,
            Err(unsupported) => {
                return Ok(one_shot_error(model, unsupported.to_string()));
            }
        };

        let mut req = self
            .http
            .post(self.messages_url())
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .header("anthropic-version", ANTHROPIC_VERSION);

        req = match &self.config.auth {
            AnthropicAuth::ApiKey(key) => req.header("x-api-key", key.as_str()),
            AnthropicAuth::OauthToken(token) => req
                .header("authorization", format!("Bearer {token}"))
                .header("anthropic-beta", "oauth-2025-04-20"),
        };

        let response = match req.json(&body).send().await {
            Ok(r) => r,
            Err(err) => {
                return Ok(one_shot_error(
                    model,
                    format!("anthropic transport: request failed: {err}"),
                ));
            }
        };

        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            let detail = detail.chars().take(2000).collect::<String>();
            return Ok(one_shot_error(
                model,
                format!("anthropic transport: HTTP {status}: {detail}"),
            ));
        }

        let model_for_state = model.clone();
        let out = async_stream::stream! {
            let mut state = AnthropicState::new(&model_for_state);
            let mut decoder = SseDecoder::new();
            let mut body_stream = response.bytes_stream();

            'outer: loop {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        for ev in state.into_aborted() {
                            yield ev;
                        }
                        break 'outer;
                    }
                    next = body_stream.next() => {
                        match next {
                            Some(Ok(bytes)) => {
                                let chunk = String::from_utf8_lossy(&bytes).to_string();
                                for frame in decoder.push(&chunk) {
                                    for ev in state.on_frame(&frame.event, &frame.data) {
                                        yield ev;
                                    }
                                    if state.is_finished() {
                                        break 'outer;
                                    }
                                }
                            }
                            Some(Err(err)) => {
                                for ev in state.into_error(
                                    format!("anthropic transport: stream error: {err}")
                                ) {
                                    yield ev;
                                }
                                break 'outer;
                            }
                            None => {
                                // Body ended. Flush the trailing frame -- the
                                // Anthropic stream's `message_stop` is not
                                // followed by a blank line.
                                if let Some(frame) = decoder.finish() {
                                    for ev in state.on_frame(&frame.event, &frame.data) {
                                        yield ev;
                                    }
                                }
                                if !state.is_finished() {
                                    // Truncated stream: close open blocks and
                                    // report, preserving accumulated content.
                                    for ev in state.into_error(
                                        "anthropic transport: stream ended without message_stop"
                                    ) {
                                        yield ev;
                                    }
                                }
                                break 'outer;
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_gains_a_trailing_slash_and_messages_is_appended() {
        let cfg = AnthropicTransportConfig::with_api_key("k").with_base_url("http://localhost:1234");
        let s = AnthropicStream::new(cfg);
        assert_eq!(s.messages_url(), "http://localhost:1234/messages");
    }

    #[test]
    fn default_base_url_targets_the_messages_endpoint() {
        let s = AnthropicStream::new(AnthropicTransportConfig::with_api_key("k"));
        assert_eq!(s.messages_url(), "https://api.anthropic.com/v1/messages");
    }
}
