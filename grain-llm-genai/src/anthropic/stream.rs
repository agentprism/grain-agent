//! [`grain_agent_core::LlmStream`] backed by a native Anthropic Messages
//! transport.
//!
//! # Status: opt-in, fixture-verified, **not live-verified**
//!
//! This backend exists because several structural gaps at the genai seam are
//! unreachable from `ChatStreamEvent`, and the one alternative that does
//! reach them (proxying the transport and re-parsing the wire) costs more
//! than owning it — see `tests/SEAM-VECTORS.md` §6 and
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
//!
//! # Known gaps, recorded rather than fixed
//!
//! - **W5 — extended-thinking request shape.** Always the legacy
//!   `{type:"enabled", budget_tokens}` form; see
//!   [`crate::anthropic::request`]. Whether adaptive-era models still accept
//!   it is an assumption only live access can settle, and it is the single
//!   largest unknown here.
//! - **W7 — reasoning-suffix stripping.** The genai path strips provider
//!   reasoning suffixes from model ids; this transport passes the bare model
//!   name through. Latent today because no grain model id carries such a
//!   suffix on the Anthropic route.
//! - **W8 — no retries, no per-request timeout.** A dropped connection
//!   surfaces as a terminal error rather than being retried. This is at
//!   parity with the genai backend, which has no per-request retry hook at
//!   this seam either, so it is a shared gap and not a regression.
//! - **W9 — no prompt caching.** No `cache_control` is emitted, so Anthropic
//!   prompt caching is unavailable on this path. Nothing in the workspace
//!   requests it today (no crate sets `ChatOptions::cache_control`), but on a
//!   long coding session this is a real cost difference, not only a feature
//!   gap. Relatedly, the terminal error for a provider `error` frame uses
//!   `error.message` and discards `error.type` — and this transport is the
//!   one place `overloaded_error` is actually available, so a caller that
//!   wanted to distinguish retryable overload from a hard failure cannot.
//!   Surfacing the type belongs with the retry work in W8.

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

/// The `anthropic-beta` value sent with an OAuth (subscription) token.
///
/// **Deliberately incomplete, and owned by the follow-on OAuth package.**
/// `docs/oauth-claude-subscription-spec.md` §6 specifies the full Claude Code
/// request shape a subscription token requires: this beta list extended with
/// `claude-code-20250219`, plus `user-agent: claude-cli/<v>`, `x-app: cli`,
/// `accept: application/json`,
/// `anthropic-dangerous-direct-browser-access: true`, and a prepended Claude
/// Code identity system block. None of that is implemented here — it cannot be
/// verified offline, and that package owns it. What this transport provides is
/// the seam it needs: Bearer auth with `x-api-key` suppressed, per-request
/// credential resolution, and a single place where request headers are built.
pub const OAUTH_BETA: &str = "oauth-2025-04-20";

/// How the transport authenticates to Anthropic.
#[derive(Clone)]
pub enum AnthropicAuth {
    /// Standard API key — sent as `x-api-key`.
    ApiKey(String),
    /// OAuth access token (Claude Pro/Max) — sent as `authorization: Bearer`,
    /// with `x-api-key` **not** sent at all (sending both is an auth error;
    /// `docs/oauth-claude-subscription-spec.md` §6).
    ///
    /// Wiring the OAuth *login* flow is owned elsewhere (debt item G7); this
    /// variant only carries an already-obtained token.
    OauthToken(String),
    /// Resolved immediately before **every** request.
    ///
    /// Required for OAuth: Anthropic subscription access tokens expire in
    /// roughly an hour, so a credential captured once at client-construction
    /// time starts returning 401 partway through any long session, with no
    /// recovery short of a restart. The genai backend avoids this by resolving
    /// inside its per-request auth resolver; this variant is the equivalent
    /// seam, and is what the builder installs.
    PerRequest(Arc<dyn Fn() -> Option<AnthropicAuth> + Send + Sync>),
}

impl std::fmt::Debug for AnthropicAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnthropicAuth::ApiKey(_) => f.write_str("ApiKey(<redacted>)"),
            AnthropicAuth::OauthToken(_) => f.write_str("OauthToken(<redacted>)"),
            AnthropicAuth::PerRequest(_) => f.write_str("PerRequest(<closure>)"),
        }
    }
}

impl AnthropicAuth {
    /// Produce the concrete credential to use for one request.
    ///
    /// Returns `None` when nothing could be resolved, which the caller turns
    /// into a terminal error naming the problem rather than issuing an
    /// unauthenticated request and surfacing an opaque provider 401.
    fn resolve_for_request(&self) -> Option<AnthropicAuth> {
        match self {
            AnthropicAuth::PerRequest(resolve) => match resolve() {
                // One level only: a resolver returning another resolver would
                // otherwise loop. Treated as "unresolved" and reported.
                Some(AnthropicAuth::PerRequest(_)) | None => None,
                resolved => resolved,
            },
            concrete => Some(concrete.clone()),
        }
    }
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

        // Resolve the credential per request, not once at construction: an
        // OAuth access token captured at build time expires mid-session.
        let Some(auth) = self.config.auth.resolve_for_request() else {
            return Ok(one_shot_error(
                model,
                "anthropic transport: no Anthropic credential could be resolved (set ANTHROPIC_API_KEY, or configure an OAuth provider profile)"
                    .to_string(),
            ));
        };

        let mut req = self
            .http
            .post(self.messages_url())
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .header("anthropic-version", ANTHROPIC_VERSION);

        req = match &auth {
            AnthropicAuth::ApiKey(key) => req.header("x-api-key", key.as_str()),
            // Bearer, and `x-api-key` deliberately absent.
            AnthropicAuth::OauthToken(token) => req
                .header("authorization", format!("Bearer {token}"))
                .header("anthropic-beta", OAUTH_BETA),
            // `resolve_for_request` never yields this.
            AnthropicAuth::PerRequest(_) => req,
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
                                // Raw bytes: the decoder owns UTF-8 and CRLF
                                // state across chunk boundaries. Decoding per
                                // chunk here would corrupt any scalar (or
                                // CRLF pair) split by the network -- see
                                // `wire::SseDecoder`.
                                for frame in decoder.push_bytes(&bytes) {
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
    fn per_request_auth_is_resolved_on_every_call() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let auth = AnthropicAuth::PerRequest(Arc::new(move || {
            let n = seen.fetch_add(1, Ordering::SeqCst);
            Some(AnthropicAuth::OauthToken(format!("token-{n}")))
        }));

        // Each resolution must re-run the closure and observe the fresh value
        // -- this is what keeps an expiring OAuth token from wedging a session.
        for expected in ["token-0", "token-1", "token-2"] {
            match auth.resolve_for_request() {
                Some(AnthropicAuth::OauthToken(t)) => assert_eq!(t, expected),
                other => panic!("expected a fresh OAuth token, got {other:?}"),
            }
        }
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn unresolvable_credentials_are_reported_not_sent_unauthenticated() {
        let auth = AnthropicAuth::PerRequest(Arc::new(|| None));
        assert!(auth.resolve_for_request().is_none());

        // A resolver that returns another resolver must not loop.
        let nested = AnthropicAuth::PerRequest(Arc::new(|| {
            Some(AnthropicAuth::PerRequest(Arc::new(|| None)))
        }));
        assert!(nested.resolve_for_request().is_none());
    }

    #[test]
    fn concrete_credentials_resolve_to_themselves() {
        match AnthropicAuth::ApiKey("k".into()).resolve_for_request() {
            Some(AnthropicAuth::ApiKey(k)) => assert_eq!(k, "k"),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn auth_debug_never_leaks_the_credential() {
        let rendered = format!(
            "{:?} {:?}",
            AnthropicAuth::ApiKey("sk-ant-super-secret".into()),
            AnthropicAuth::OauthToken("sk-ant-oat-secret".into())
        );
        assert!(!rendered.contains("secret"), "credential leaked: {rendered}");
    }

    #[test]
    fn default_base_url_targets_the_messages_endpoint() {
        let s = AnthropicStream::new(AnthropicTransportConfig::with_api_key("k"));
        assert_eq!(s.messages_url(), "https://api.anthropic.com/v1/messages");
    }

    /// The no-credential error reaches users verbatim (TUI, headless logs),
    /// so its text is part of the surface: normalized single spacing, no
    /// leftover interior runs from source-level line wrapping (G38).
    #[tokio::test]
    async fn no_credential_error_message_is_normalized() {
        use futures::StreamExt;

        let s = AnthropicStream::new(AnthropicTransportConfig {
            base_url: AnthropicTransportConfig::DEFAULT_BASE_URL.to_string(),
            auth: AnthropicAuth::PerRequest(Arc::new(|| None)),
        });
        let model = Model {
            id: "anthropic/claude-haiku-4-5".into(),
            name: "claude-haiku-4-5".into(),
            api: "anthropic".into(),
            provider: "anthropic".into(),
            ..Default::default()
        };
        let stream = s
            .stream(
                &model,
                &LlmContext::default(),
                &StreamOptions::default(),
                CancellationToken::new(),
            )
            .await
            .expect("no-credential path must yield a one-shot error stream");

        // `one_shot_error` preserves the event contract (Start … Error), so
        // the terminal is the LAST event, not necessarily the first.
        let events: Vec<_> = stream.collect().await;
        let Some(grain_agent_core::AssistantMessageEvent::Error { error, .. }) = events.last()
        else {
            panic!("expected a terminal Error event, got {events:?}");
        };
        let error = error.clone();
        assert_eq!(
            error,
            "anthropic transport: no Anthropic credential could be resolved \
             (set ANTHROPIC_API_KEY, or configure an OAuth provider profile)"
        );
        assert!(
            !error.contains("  "),
            "user-facing error must not carry interior space runs: {error:?}"
        );
    }
}
