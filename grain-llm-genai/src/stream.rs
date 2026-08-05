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

    /// Public introspection of the routing decision in
    /// [`Self::native_anthropic_for`]: would `model` be served by the native
    /// Anthropic transport rather than genai?
    ///
    /// Exists so consumers that opt in via
    /// [`GenaiStreamBuilder::with_native_anthropic_transport`] can PIN their
    /// construction choice in their own test suites (the AgentPrism host
    /// does) without reaching into private fields: `true` only when the
    /// opt-in was set *and* `model` routes to the anthropic namespace.
    pub fn uses_native_anthropic_for(&self, model: &Model) -> bool {
        self.native_anthropic_for(model).is_some()
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
/// - `thinking_budgets` → `ReasoningEffort::Budget` when the caller supplied
///   an explicit token budget for the requested level AND `api` is a
///   token-budget family; see [`effort_for_request`].
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
/// the following per call; this adapter consumes only `reasoning` and
/// `thinking_budgets`:
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
fn chat_options_with_runtime(base: ChatOptions, options: &StreamOptions, api: &str) -> ChatOptions {
    let mut chat = base.with_capture_usage(true).with_capture_tool_calls(true);
    if let Some(level) = options.reasoning
        && let Some(effort) = effort_for_request(level, options, api)
    {
        chat = chat.with_reasoning_effort(effort);
    }
    chat
}

/// Resolve the [`ReasoningEffort`] for one request: an explicit caller
/// budget for the requested level becomes [`ReasoningEffort::Budget`] on
/// token-budget API families; everything else takes the 1:1 named mapping
/// ([`thinking_level_to_effort`]).
///
/// This is what makes `Budget(u32)` reachable, completing the full-range
/// mapping rust-host.md's effort-mapping-rework item describes. The rule
/// mirrors both the native Anthropic transport (`anthropic/request.rs::
/// thinking_budget`: a caller-supplied [`ThinkingBudgets`] entry wins over
/// the level's default budget) and upstream pi-ai, where `thinkingBudgets`
/// are consumed only by the token-budget provider modules
/// (`adjustMaxTokensForThinking` in `simple-options.ts` for
/// anthropic/bedrock, `getGoogleBudget` in the google modules @ 34239180) —
/// the openai modules send the named effort and never read budgets.
///
/// The family gate matters for more than fidelity: genai's OpenAI adapter
/// deliberately ignores `Budget(_)` (`adapter/adapters/openai/
/// adapter_shared.rs:27`, `return Ok(())` — no `reasoning_effort` set at
/// all), so mapping budgets unconditionally would silently DROP the named
/// reasoning level on OpenAI-family models whenever a caller configures
/// budgets globally (the harness forwards `thinking_budgets` on every
/// request regardless of provider). `Off` never becomes a budget — off
/// means no thinking, and maps to `ReasoningEffort::None` as before.
///
/// Budgets are `u64` grain-side and `u32` in genai; values above
/// `u32::MAX` saturate.
fn effort_for_request(
    level: ThinkingLevel,
    options: &StreamOptions,
    api: &str,
) -> Option<ReasoningEffort> {
    if is_token_budget_api(api)
        && let Some(budget) = explicit_budget_for(level, options.thinking_budgets)
    {
        return Some(ReasoningEffort::Budget(
            u32::try_from(budget).unwrap_or(u32::MAX),
        ));
    }
    thinking_level_to_effort(level)
}

/// The caller-supplied budget for `level`, if any. Level → field mapping
/// mirrors the explicit half of the native transport's
/// `anthropic/request.rs::thinking_budget` exactly: `XHigh` / `Max` share
/// the `high` budget (upstream's `clampReasoning` collapses them to `high`
/// before the budget lookup, `simple-options.ts:49-50, 68-69`), and `Off`
/// carries no budget by construction.
fn explicit_budget_for(
    level: ThinkingLevel,
    budgets: Option<grain_agent_core::ThinkingBudgets>,
) -> Option<u64> {
    let budgets = budgets?;
    match level {
        ThinkingLevel::Off => None,
        ThinkingLevel::Minimal => budgets.minimal,
        ThinkingLevel::Low => budgets.low,
        ThinkingLevel::Medium => budgets.medium,
        ThinkingLevel::High | ThinkingLevel::XHigh | ThinkingLevel::Max => budgets.high,
    }
}

/// API families where genai maps `ReasoningEffort::Budget` onto a real
/// token-budget wire field — anthropic (`thinking.budget_tokens`,
/// `adapter/adapters/anthropic/adapter_shared.rs:719,751`), google/Gemini
/// (`generationConfig.thinkingConfig.thinkingBudget`,
/// `adapter/adapters/gemini/adapter_impl.rs:34-38`), and Bedrock Converse
/// (`adapter/adapters/bedrock/converse.rs:233`) — exactly the modules where
/// upstream pi-ai consumes `thinkingBudgets`. Matches grain's production
/// wire names (`ApiKind::wire_name`: `anthropic`, `gemini`) as well as
/// upstream's api ids (`anthropic-messages`, `google-generative-ai`,
/// `google-vertex`, `bedrock-converse-stream`), same breadth as
/// `mapping::inbound::is_google_api`.
fn is_token_budget_api(api: &str) -> bool {
    api.starts_with("anthropic")
        || api.starts_with("google")
        || api == "gemini"
        || api.starts_with("bedrock")
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
/// genai additionally offers `ReasoningEffort::Budget(u32)`, reached via
/// [`effort_for_request`] when the caller supplies an explicit
/// `StreamOptions::thinking_budgets` entry for the requested level on a
/// token-budget API family — this named 1:1 table is the default for every
/// other request. `Max` maps directly since WP4's
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
        let chat_options = chat_options_with_runtime(self.chat_options.clone(), options, &model.api);
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
        let projected = chat_options_with_runtime(bare, &StreamOptions::default(), "openai");
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
        let projected = chat_options_with_runtime(base, &StreamOptions::default(), "openai");
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
        assert!(!stream.uses_native_anthropic_for(&model_id("anthropic/claude-haiku-4-5")));
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
        assert!(
            stream.uses_native_anthropic_for(&model_id("anthropic/claude-haiku-4-5")),
            "the public routing probe must agree with the private routing decision"
        );
        for other in ["openai/gpt-5", "google/gemini-2.5-pro", "deepseek/deepseek-chat"] {
            assert!(
                stream.native_anthropic_for(&model_id(other)).is_none(),
                "{other} must stay on genai"
            );
            assert!(!stream.uses_native_anthropic_for(&model_id(other)));
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
            "openai",
        );
        assert!(matches!(
            projected.reasoning_effort,
            Some(ReasoningEffort::XHigh)
        ));
    }

    // -- G17: ReasoningEffort::Budget reachability -------------------------

    use grain_agent_core::ThinkingBudgets;

    fn budgeted(level: ThinkingLevel, budgets: ThinkingBudgets) -> StreamOptions {
        StreamOptions {
            reasoning: Some(level),
            thinking_budgets: Some(budgets),
            ..StreamOptions::default()
        }
    }

    /// An explicit caller budget for the requested level reaches genai as
    /// `ReasoningEffort::Budget` on token-budget API families — the path
    /// that makes rust-host.md's "the mapping covers the full range"
    /// (named levels PLUS `Budget(u32)`) true.
    #[test]
    fn explicit_budget_maps_to_reasoning_effort_budget_on_token_budget_apis() {
        for api in ["anthropic", "anthropic-messages", "gemini", "google-vertex"] {
            let projected = chat_options_with_runtime(
                ChatOptions::default(),
                &budgeted(
                    ThinkingLevel::Medium,
                    ThinkingBudgets {
                        medium: Some(8192),
                        ..ThinkingBudgets::default()
                    },
                ),
                api,
            );
            assert!(
                matches!(
                    projected.reasoning_effort,
                    Some(ReasoningEffort::Budget(8192))
                ),
                "{api}: expected Budget(8192), got {:?}",
                projected.reasoning_effort
            );
        }
    }

    /// OpenAI-family requests keep the NAMED effort even when budgets are
    /// configured: upstream's openai modules never read `thinkingBudgets`,
    /// and genai's OpenAI adapter silently drops `Budget(_)` (no
    /// `reasoning_effort` set at all) — mapping it would lose the user's
    /// level whenever budgets are configured globally.
    #[test]
    fn openai_family_keeps_named_effort_despite_budgets() {
        let projected = chat_options_with_runtime(
            ChatOptions::default(),
            &budgeted(
                ThinkingLevel::Medium,
                ThinkingBudgets {
                    medium: Some(8192),
                    ..ThinkingBudgets::default()
                },
            ),
            "openai",
        );
        assert!(matches!(
            projected.reasoning_effort,
            Some(ReasoningEffort::Medium)
        ));
    }

    /// A budgets struct that carries no entry for the requested level falls
    /// back to the named mapping.
    #[test]
    fn missing_budget_for_the_level_falls_back_to_named_effort() {
        let projected = chat_options_with_runtime(
            ChatOptions::default(),
            &budgeted(
                ThinkingLevel::High,
                ThinkingBudgets {
                    low: Some(2048),
                    ..ThinkingBudgets::default()
                },
            ),
            "anthropic",
        );
        assert!(matches!(
            projected.reasoning_effort,
            Some(ReasoningEffort::High)
        ));
    }

    /// `Off` never becomes a budget — off means no thinking, and maps to
    /// `ReasoningEffort::None` exactly as without budgets.
    #[test]
    fn off_is_never_budgeted() {
        let projected = chat_options_with_runtime(
            ChatOptions::default(),
            &budgeted(
                ThinkingLevel::Off,
                ThinkingBudgets {
                    high: Some(24000),
                    ..ThinkingBudgets::default()
                },
            ),
            "anthropic",
        );
        assert!(matches!(
            projected.reasoning_effort,
            Some(ReasoningEffort::None)
        ));
    }

    /// `XHigh` / `Max` share the `high` budget entry — the same collapse the
    /// native transport applies (`anthropic/request.rs::thinking_budget`)
    /// and upstream's `clampReasoning` performs before its budget lookup.
    #[test]
    fn xhigh_and_max_share_the_high_budget() {
        for level in [ThinkingLevel::XHigh, ThinkingLevel::Max] {
            let projected = chat_options_with_runtime(
                ChatOptions::default(),
                &budgeted(
                    level,
                    ThinkingBudgets {
                        high: Some(24000),
                        ..ThinkingBudgets::default()
                    },
                ),
                "gemini",
            );
            assert!(
                matches!(
                    projected.reasoning_effort,
                    Some(ReasoningEffort::Budget(24000))
                ),
                "{level:?}: got {:?}",
                projected.reasoning_effort
            );
        }
    }

    /// grain budgets are u64; genai's slot is u32. Oversized values saturate
    /// rather than truncate or panic.
    #[test]
    fn oversized_budget_saturates_at_u32_max() {
        let projected = chat_options_with_runtime(
            ChatOptions::default(),
            &budgeted(
                ThinkingLevel::Medium,
                ThinkingBudgets {
                    medium: Some(u64::from(u32::MAX) + 5),
                    ..ThinkingBudgets::default()
                },
            ),
            "anthropic",
        );
        assert!(matches!(
            projected.reasoning_effort,
            Some(ReasoningEffort::Budget(u32::MAX))
        ));
    }
}
