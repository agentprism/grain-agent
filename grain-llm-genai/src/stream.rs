//! [`grain_agent_core::LlmStream`] implementation backed by `genai 0.6`.
//!
//! The streaming logic lives here; the message ↔ event translation lives in
//! [`crate::mapping`]; the client construction + provider routing lives in
//! [`crate::builder`].

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use genai::chat::{ChatOptions, ReasoningEffort};
use grain_agent_core::{
    AssistantMessageEvent, AssistantStream, LlmContext, LlmStream, Model, StreamError,
    StreamOptions, ThinkingLevel,
};
use grain_llm_models::Registry;
use tokio_util::sync::CancellationToken;

use crate::anthropic::AnthropicStream;
use crate::builder::GenaiStreamBuilder;
use crate::config::ProviderRouter;
use crate::mapping::inbound::InboundState;
use crate::mapping::outbound::{baseline_chat_options, to_chat_request};

/// [`LlmStream`] implementation backed by [`genai::Client`].
///
/// Build via [`GenaiStream::builder()`] for full configuration (env-var
/// key resolution, OpenAI-compat presets, model registry); [`GenaiStream::new`]
/// retains the zero-config behavior from PR 3b.
pub struct GenaiStream {
    client: genai::Client,
    chat_options: ChatOptions,
    provider_router: ProviderRouter,
    #[allow(dead_code)] // Reserved for harness hooks / future adapters.
    registry: Option<Arc<Registry>>,
    /// Opt-in native Anthropic transport ([`crate::anthropic`]). `None` -- the
    /// default -- routes Anthropic through genai like every other provider.
    /// Set via [`crate::GenaiStreamBuilder::with_native_anthropic_transport`].
    native_anthropic: Option<Arc<AnthropicStream>>,
}

impl Default for GenaiStream {
    fn default() -> Self {
        GenaiStream::new()
    }
}

impl GenaiStream {
    /// Zero-config: default genai client + [`baseline_chat_options`].
    pub fn new() -> Self {
        GenaiStream {
            client: genai::Client::default(),
            chat_options: baseline_chat_options(),
            provider_router: ProviderRouter::default(),
            registry: None,
            native_anthropic: None,
        }
    }

    /// Start a configuration chain. See [`GenaiStreamBuilder`].
    pub fn builder() -> GenaiStreamBuilder {
        GenaiStreamBuilder::new()
    }

    /// Construct from a fully-configured client. Used by tests that want to
    /// inject a mock client without going through the builder.
    pub fn with_client_and_options(client: genai::Client, chat_options: ChatOptions) -> Self {
        GenaiStream {
            client,
            chat_options,
            provider_router: ProviderRouter::default(),
            registry: None,
            native_anthropic: None,
        }
    }

    /// Construct from a builder-prepared client. Public for [`GenaiStreamBuilder::build`]
    /// to plumb its config in.
    pub fn with_client_options_and_router(
        client: genai::Client,
        chat_options: ChatOptions,
        provider_router: ProviderRouter,
        registry: Option<Arc<Registry>>,
    ) -> Self {
        GenaiStream {
            client,
            chat_options,
            provider_router,
            registry,
            native_anthropic: None,
        }
    }

    /// Attach (or clear) the opt-in native Anthropic transport. Used by
    /// [`GenaiStreamBuilder::build`]; see
    /// [`crate::GenaiStreamBuilder::with_native_anthropic_transport`].
    pub fn with_native_anthropic(mut self, native: Option<Arc<AnthropicStream>>) -> Self {
        self.native_anthropic = native;
        self
    }

    /// Whether `model` should be served by the native Anthropic transport.
    ///
    /// Gated on both the opt-in being set *and* the model actually routing to
    /// the anthropic namespace, so enabling it never affects other providers.
    fn native_anthropic_for(&self, model: &Model) -> Option<&Arc<AnthropicStream>> {
        let native = self.native_anthropic.as_ref()?;
        self.translate_model_id(&model.id)
            .starts_with("anthropic::")
            .then_some(native)
    }

    /// Translate a grain model id (`"anthropic/claude-sonnet-4-5"`) into the
    /// `"<namespace>::<model>"` form genai dispatches on. Provider names with
    /// no `/` pass through unchanged so callers can also feed genai-native
    /// identifiers directly.
    pub fn translate_model_id(&self, model_id: &str) -> String {
        match model_id.split_once('/') {
            Some((provider, name)) => {
                let ns = self.provider_router.namespace_for(provider);
                format!("{ns}::{name}")
            }
            None => model_id.to_string(),
        }
    }
}

/// Project a per-request `StreamOptions` onto a fresh `ChatOptions`.
///
/// Currently honored:
/// - `reasoning` → `ChatOptions::with_reasoning_effort` (ThinkingLevel maps
///   onto genai's `ReasoningEffort` variants 1:1 by name; see
///   [`thinking_level_to_effort`]).
///
/// **Always enforced**, regardless of what the builder-provided base
/// options say:
/// - `capture_usage`: the `LlmStream` terminal contract maps
///   `StreamEnd::captured_usage` into the final message's `Usage`; without
///   the flag genai never populates it.
/// - `capture_tool_calls`: genai 0.6.5's OpenAI streamer only merges
///   tool-call fragments into id-stable cumulative chunks when this is on
///   (`adapter/adapters/openai/streamer.rs::capture_tool_call`); with it
///   off, follow-up fragments arrive under a synthetic `call_{index}` id
///   and the inbound accumulator would treat them as new calls.
///
/// **Not yet honored** (verified against genai 0.6.5: `ChatOptions` has no
/// per-call slots for these — auth goes through the client-build-time
/// `AuthResolver`, and transport/retry config is fixed on the client — so
/// wiring them up requires a fuller refactor of the client builder; see the
/// M-2 code-review entry). `grain_agent_core::StreamOptions` carries all of
/// the following per call; this adapter consumes only `reasoning`:
/// - `api_key`: would need a dynamic auth resolver per-call.
/// - `session_id` / `transport`: provider-specific transport knobs.
/// - `max_retries` / `max_retry_delay_ms` / `timeout_ms` /
///   `websocket_connect_timeout_ms`: WebConfig is set at client build time.
/// - `temperature` / `max_tokens` / `cache_retention` / `headers` /
///   `metadata` / `env` / `extra`.
/// - `on_payload` / `on_response`: no genai interception point.
///
/// Note `headers` uses upstream's null-suppression semantics — a `None`
/// value removes a provider default header rather than being a no-op.
fn chat_options_with_runtime(base: ChatOptions, options: &StreamOptions) -> ChatOptions {
    let mut chat = base.with_capture_usage(true).with_capture_tool_calls(true);
    if let Some(level) = options.reasoning
        && let Some(effort) = thinking_level_to_effort(level)
    {
        chat = chat.with_reasoning_effort(effort);
    }
    chat
}

/// Map grain's [`ThinkingLevel`] onto genai 0.6's [`ReasoningEffort`], 1:1
/// by name:
///
/// | grain `ThinkingLevel` | genai `ReasoningEffort` |
/// |-----------------------|-------------------------|
/// | `Off`                 | `None`                  |
/// | `Minimal`             | `Minimal`               |
/// | `Low`                 | `Low`                   |
/// | `Medium`              | `Medium`                |
/// | `High`                | `High`                  |
/// | `XHigh`               | `XHigh`                 |
/// | `Max`                 | `Max`                   |
///
/// genai additionally offers `ReasoningEffort::Budget(u32)`, which has no
/// grain-side counterpart yet — it stays unmapped until a later WP adds a
/// budget-based `ThinkingLevel` variant. `Max` maps directly since WP4's
/// patch-9 added `ThinkingLevel::Max`. (The historical `XHigh` → `High`
/// collapse was stale adapter code: every genai release this adapter has
/// pinned — 0.6.0-beta.20 onward — already had `ReasoningEffort::XHigh`,
/// so the collapse silently downgraded the user's intent for no reason. It
/// was removed in the 0.6.5 migration; `XHigh` now passes through
/// unchanged.)
///
/// Forward-compat note: the genai 0.7 line renames `ReasoningEffort::None`
/// to `ReasoningEffort::Zero` (with `#[serde(alias = "None")]`), so on that
/// bump the `Off` arm becomes `ReasoningEffort::Zero` — a mechanical change.
fn thinking_level_to_effort(level: ThinkingLevel) -> Option<ReasoningEffort> {
    match level {
        ThinkingLevel::Off => Some(ReasoningEffort::None),
        ThinkingLevel::Minimal => Some(ReasoningEffort::Minimal),
        ThinkingLevel::Low => Some(ReasoningEffort::Low),
        ThinkingLevel::Medium => Some(ReasoningEffort::Medium),
        ThinkingLevel::High => Some(ReasoningEffort::High),
        ThinkingLevel::XHigh => Some(ReasoningEffort::XHigh),
        ThinkingLevel::Max => Some(ReasoningEffort::Max),
    }
}

#[async_trait]
impl LlmStream for GenaiStream {
    async fn stream(
        &self,
        model: &Model,
        context: &LlmContext,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> Result<AssistantStream, StreamError> {
        // Opt-in: serve Anthropic from the native transport, which reports
        // usage correctly (genai double-counts it -- see `crate::anthropic`).
        if let Some(native) = self.native_anthropic_for(model) {
            return native.stream(model, context, options, cancel).await;
        }

        let chat_req = to_chat_request(context);
        let chat_options = chat_options_with_runtime(self.chat_options.clone(), options);
        let model_for_genai = self.translate_model_id(&model.id);

        let stream_resp = match self
            .client
            .exec_chat_stream(&model_for_genai, chat_req, Some(&chat_options))
            .await
        {
            Ok(r) => r,
            Err(err) => {
                // `LlmStream` contract: don't return `Err` for runtime failures.
                // Synthesize a terminal Error event with the failure message.
                let state = InboundState::new(model);
                let event = state.into_error_msg(format!("genai exec_chat_stream: {err}"));
                let one_shot = futures::stream::iter(std::iter::once(event));
                return Ok(Box::pin(one_shot));
            }
        };

        let model_for_state = model.clone();
        let inner = stream_resp.stream;

        // Terminal contract audit (WP3): every exit from the loop below
        // yields exactly one terminal event carrying the assistant message —
        // `Done` from the state machine on `ChatStreamEvent::End`, or an
        // `Error` synthesized via `into_aborted` / `into_error_msg`, both of
        // which preserve all content accumulated so far (mirroring upstream
        // pi-ai, whose error event carries the partial AssistantMessage).
        // The transport-failure path above returns a one-shot Error stream
        // with an empty (nothing streamed yet) message for the same reason.
        let out = async_stream::stream! {
            let mut state = InboundState::new(&model_for_state);
            let mut inner = inner;
            let cancel = cancel.clone();

            loop {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        yield state.into_aborted();
                        break;
                    }
                    event = inner.next() => {
                        match event {
                            Some(Ok(ev)) => {
                                let mut terminal = false;
                                for grain_event in state.on_event(ev) {
                                    if matches!(
                                        grain_event,
                                        AssistantMessageEvent::Done { .. }
                                            | AssistantMessageEvent::Error { .. }
                                    ) {
                                        terminal = true;
                                    }
                                    yield grain_event;
                                }
                                if terminal {
                                    break;
                                }
                            }
                            Some(Err(err)) => {
                                yield state.into_error_msg(format!("genai stream error: {err}"));
                                break;
                            }
                            None => {
                                yield state.into_error_msg("stream ended without terminal event");
                                break;
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
    fn runtime_options_enforce_usage_and_tool_call_capture() {
        // Even when a caller supplies bare ChatOptions via the builder
        // (dropping the baseline capture flags), the per-request projection
        // must re-enable the captures the adapter's correctness depends on.
        let bare = ChatOptions::default();
        let projected = chat_options_with_runtime(bare, &StreamOptions::default());
        assert_eq!(projected.capture_usage, Some(true));
        assert_eq!(projected.capture_tool_calls, Some(true));
    }

    #[test]
    fn runtime_options_override_explicit_false_capture_flags() {
        // A builder-supplied ChatOptions that explicitly disables the
        // capture flags (Some(false), not merely unset) must still be
        // overridden — the terminal-usage contract and the cumulative
        // tool-chunk accumulator are not opt-outs.
        let base = ChatOptions::default()
            .with_capture_usage(false)
            .with_capture_tool_calls(false);
        let projected = chat_options_with_runtime(base, &StreamOptions::default());
        assert_eq!(projected.capture_usage, Some(true));
        assert_eq!(projected.capture_tool_calls, Some(true));
    }

    fn model_id(id: &str) -> Model {
        Model {
            id: id.into(),
            name: id.into(),
            api: "anthropic-messages".into(),
            provider: "anthropic".into(),
            ..Default::default()
        }
    }

    /// Constraint: genai stays the default backend. A builder that says
    /// nothing about transports must not route Anthropic anywhere new.
    #[test]
    fn genai_is_the_default_backend_for_anthropic() {
        let stream = GenaiStreamBuilder::new().build();
        assert!(
            stream.native_anthropic.is_none(),
            "the native Anthropic transport must be opt-in, never implicit"
        );
        assert!(
            stream
                .native_anthropic_for(&model_id("anthropic/claude-haiku-4-5"))
                .is_none()
        );
    }

    /// Opting in routes Anthropic to the native transport — and nothing else.
    #[test]
    fn opt_in_routes_only_anthropic_models_to_the_native_transport() {
        let stream = GenaiStreamBuilder::new()
            .with_native_anthropic_transport(true)
            .build();
        assert!(stream.native_anthropic.is_some());

        assert!(
            stream
                .native_anthropic_for(&model_id("anthropic/claude-haiku-4-5"))
                .is_some(),
            "anthropic models must use the native transport when opted in"
        );
        for other in ["openai/gpt-5", "google/gemini-2.5-pro", "deepseek/deepseek-chat"] {
            assert!(
                stream.native_anthropic_for(&model_id(other)).is_none(),
                "{other} must stay on genai"
            );
        }
    }

    #[test]
    fn runtime_options_map_thinking_level() {
        let projected = chat_options_with_runtime(
            ChatOptions::default(),
            &StreamOptions {
                reasoning: Some(ThinkingLevel::XHigh),
                ..StreamOptions::default()
            },
        );
        assert!(matches!(
            projected.reasoning_effort,
            Some(ReasoningEffort::XHigh)
        ));
    }
}
