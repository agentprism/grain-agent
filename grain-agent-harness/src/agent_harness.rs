//! `AgentHarness` — top-level orchestrator that bundles `Agent` +
//! `Session` + tools + skills + queues into a single façade.
//!
//! Implemented here:
//!
//! - [`AgentHarnessOptions`] / [`AgentHarness::new`]
//! - [`AgentHarness::prompt_text`] / [`AgentHarness::prompt`]
//! - [`AgentHarness::subscribe`] — listener fires with
//!   [`AgentHarnessEvent`], a superset of `AgentEvent` plus
//!   harness-own lifecycle markers.
//! - [`AgentHarness::abort`] / [`AgentHarness::wait_for_idle`]
//! - [`AgentHarness::session`] — clone-cheap handle on the owned
//!   session.
//! - Queues: [`AgentHarness::steer`] / [`AgentHarness::follow_up`] /
//!   [`AgentHarness::next_turn`].
//! - State: [`AgentHarness::set_model`] /
//!   [`AgentHarness::set_thinking_level`] /
//!   [`AgentHarness::set_active_tools`], each persisted to the session
//!   (batched while a run is in flight, flushed at turn boundaries).
//! - Session control: [`AgentHarness::append_entry`] /
//!   [`AgentHarness::navigate_tree`] / [`AgentHarness::compact`].
//! - `Resources` (skills + prompt templates) with
//!   [`AgentHarness::prompt_from_template`] / [`AgentHarness::skill`].
//! - Session ownership: harness seeds the agent's transcript from
//!   the session's branch on construction, then mirrors every
//!   `MessageEnd` back into the session via an internal listener.
//!   [`crate::resume`] is the same seeding, for callers that want a bare
//!   `Agent` rather than the whole harness.
//!
//! Known gaps against upstream (`packages/agent/src/harness/agent-harness.ts`
//! @ 34239180) — see the repo's debt ledger for ownership:
//!
//! - `SystemPrompt::Dynamic` is resolved once at construction and never
//!   re-rendered; upstream re-renders it every turn in `createTurnState`.
//!   Today a `Dynamic` prompt collapses to the empty string.
//! - `next_turn` is aliased onto the follow-up queue rather than being a
//!   third bucket with its own prepend-at-next-prompt semantics.
//! - The provider-hook triad (`before_provider_request`,
//!   `before_provider_payload`, `after_provider_response`) is absent, and
//!   `AgentHarnessOptions` exposes no `on_payload` / `on_response` /
//!   `thinking_budgets` passthrough.
//! - [`AgentHarness::skill`] synthesizes a prompt string instead of
//!   validating the name against `Resources::skills` and rendering
//!   upstream's `<skill>` invocation block.
//! - No harness-level `fork` over the ported [`crate::session::SessionRepo`]
//!   fork. (Upstream has no `Agent.fork`; the fork lives at the session and
//!   runtime layers.)

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use futures::future::BoxFuture;
use grain_agent_core::{
    AfterToolCallFn, Agent, AgentError, AgentEvent, AgentLoopTurnUpdate, AgentMessage,
    AgentOptions, AgentTool, AssistantMessage, AssistantMessageEvent, BeforeToolCallFn,
    ConvertToLlmFn, GetApiKeyFn, Model, OnPayloadFn, OnResponseFn, PrepareNextTurnContext,
    PrepareNextTurnFn, QueueMode, StreamFn, ThinkingBudgets, ThinkingLevel, ToolExecutionMode,
    TransformContextFn,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::compaction::{
    CompactionError, DEFAULT_COMPACTION_PROMPT, compact_transcript, first_kept_context_entry_id,
};
use crate::messages::convert_to_llm as convert_to_llm_sync;
use crate::session::{Session, SessionError};
use crate::system_prompt::{Skill, format_skill_invocation};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Constructor input for [`AgentHarness`]. Mirrors pi's
/// `AgentHarnessOptions` minus the generic skill / template / tool
/// type parameters — we use concrete trait objects throughout.
pub struct AgentHarnessOptions {
    /// Initial session. The harness clones the handle and seeds the
    /// agent's transcript from the session's current branch.
    pub session: Session,
    pub model: Model,
    pub stream_fn: StreamFn,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub resources: Resources,
    pub system_prompt: SystemPrompt,
    pub thinking_level: ThinkingLevel,
    /// `None` means every tool is active. Names given here that are not in
    /// [`Self::tools`] are ignored at construction; the later
    /// [`AgentHarness::set_active_tools`] validates and rejects them.
    pub active_tool_names: Option<Vec<String>>,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
    pub get_api_key: Option<GetApiKeyFn>,
    pub transform_context: Option<TransformContextFn>,
    pub tool_execution: ToolExecutionMode,
    pub session_id: Option<String>,
    pub transport: Option<String>,
    pub max_retry_delay_ms: Option<u64>,
    /// Inspect or replace the provider payload before it is sent. Forwarded
    /// verbatim to [`AgentOptions::on_payload`].
    ///
    /// This is the narrow form of upstream's `before_provider_payload` hook
    /// (`agent-harness.ts:303-318` @ 34239180): one callback rather than a
    /// chain of subscribers. Whether the shipped provider adapter honors it
    /// is an adapter question — `grain-llm-genai` currently does not.
    pub on_payload: Option<OnPayloadFn>,
    /// Called once an HTTP response is received, before its body is consumed.
    /// Forwarded to [`AgentOptions::on_response`]. Narrow form of upstream's
    /// `after_provider_response` event (`agent-harness.ts:414-420`).
    pub on_response: Option<OnResponseFn>,
    /// Per-level thinking token budgets, forwarded to
    /// [`AgentOptions::thinking_budgets`]. Upstream exposes this on `Agent`
    /// but not on its harness; grain surfaces it here so a harness-only
    /// embedder is not forced down to the bare `Agent`.
    pub thinking_budgets: Option<ThinkingBudgets>,
    /// Optional Agent hook: gate a tool call before it executes
    /// (storm suppression, schema repair, …). Passed through verbatim
    /// to [`AgentOptions::before_tool_call`].
    pub before_tool_call: Option<BeforeToolCallFn>,
    /// Optional Agent hook: rewrite / inspect a tool result after
    /// execution (error-streak terminator, result-truncation, …).
    /// Passed through to [`AgentOptions::after_tool_call`].
    pub after_tool_call: Option<AfterToolCallFn>,
    /// Optional Agent hook: swap model / thinking level between
    /// turns (failure-signal escalation, etc.). Passed through to
    /// [`AgentOptions::prepare_next_turn`].
    pub prepare_next_turn: Option<PrepareNextTurnFn>,
    /// Optional Agent hook: override the default projection from
    /// `AgentMessage[]` → `Message[]`. **Replaces** the harness's
    /// custom-message routing — callers wanting both behaviors
    /// should call [`crate::convert_to_llm`] inside their wrapper.
    /// Most callers want `None` (let the harness do the right thing).
    pub convert_to_llm: Option<ConvertToLlmFn>,
}

impl AgentHarnessOptions {
    /// Minimal constructor — everything beyond model / stream / session
    /// gets a sane default. Mirrors `AgentOptions::new`.
    pub fn new(session: Session, model: Model, stream_fn: StreamFn) -> Self {
        AgentHarnessOptions {
            session,
            model,
            stream_fn,
            tools: Vec::new(),
            resources: Resources::default(),
            system_prompt: SystemPrompt::Static(String::new()),
            thinking_level: ThinkingLevel::Off,
            active_tool_names: None,
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::OneAtATime,
            get_api_key: None,
            transform_context: None,
            tool_execution: ToolExecutionMode::Parallel,
            session_id: None,
            transport: None,
            max_retry_delay_ms: None,
            on_payload: None,
            on_response: None,
            thinking_budgets: None,
            before_tool_call: None,
            after_tool_call: None,
            prepare_next_turn: None,
            convert_to_llm: None,
        }
    }
}

/// Closure shape for [`SystemPrompt::Dynamic`]. Phase 1 stashes
/// these but doesn't yet re-render — Phase 2 ties it to model /
/// thinking / resources changes.
pub type DynamicSystemPromptFn =
    Arc<dyn Fn(&SystemPromptCtx) -> BoxFuture<'static, String> + Send + Sync>;

/// System prompt resolution: either a fixed string baked into the
/// agent at construction, or a closure re-evaluated per
/// reconfiguration. Phase 1 hands the static string straight to
/// `AgentOptions::system_prompt` and stashes the dynamic variant for
/// future phases.
pub enum SystemPrompt {
    Static(String),
    Dynamic(DynamicSystemPromptFn),
}

impl std::fmt::Debug for SystemPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SystemPrompt::Static(s) => f.debug_tuple("Static").field(s).finish(),
            SystemPrompt::Dynamic(_) => f.debug_tuple("Dynamic").field(&"<closure>").finish(),
        }
    }
}

/// Inputs visible to a [`SystemPrompt::Dynamic`] closure. Phase 1
/// supplies model + thinking level + active tool count + resource
/// counts; later phases extend.
pub struct SystemPromptCtx<'a> {
    pub model: &'a Model,
    pub thinking_level: ThinkingLevel,
    pub active_tools: &'a [Arc<dyn AgentTool>],
    pub resources: &'a Resources,
}

/// Skills + prompt templates shipped with the harness. `prompt_templates`
/// becomes meaningful in Phase 4 (`prompt_from_template`); Phase 1
/// only consumes `skills` indirectly via the rendered system prompt
/// block (callers can do that themselves).
#[derive(Default)]
pub struct Resources {
    pub skills: Vec<Skill>,
    pub prompt_templates: Vec<PromptTemplate>,
}

impl std::fmt::Debug for Resources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resources")
            .field("skills", &self.skills.len())
            .field("prompt_templates", &self.prompt_templates.len())
            .finish()
    }
}

/// One pi-style prompt template. The `render` closure produces the
/// final prompt text from JSON args supplied by the caller. Phase 1
/// stores templates but doesn't wire `prompt_from_template` yet.
pub struct PromptTemplate {
    pub name: String,
    pub description: String,
    pub render: Arc<dyn Fn(&serde_json::Value) -> String + Send + Sync>,
}

impl std::fmt::Debug for PromptTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `render` is a closure — deliberately omitted. `finish_non_exhaustive`
        // marks the struct as not fully printed so readers know there's
        // more behind the curtain.
        f.debug_struct("PromptTemplate")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Superset of `grain_agent_core::AgentEvent`. Phase 1 emits the
/// pass-through variants + a small set of harness-own markers
/// (`Settled`, `Abort`). Phase 3/4 will add the rest of the pi event
/// surface (`BeforeAgentStart`, `Context`, `SessionBeforeCompact`,
/// `SessionCompact`, `SessionTree`, `ModelSelect`, etc.).
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentHarnessEvent {
    // --- pass-throughs from grain-agent-core::AgentEvent ---
    AgentStart,
    AgentEnd {
        messages: Vec<AgentMessage>,
    },
    TurnStart,
    TurnEnd {
        message: AssistantMessage,
        tool_results: Vec<grain_agent_core::ToolResultMessage>,
    },
    MessageStart {
        message: AgentMessage,
    },
    MessageUpdate {
        message: AssistantMessage,
        assistant_message_event: AssistantMessageEvent,
    },
    MessageEnd {
        message: AgentMessage,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
        partial_result: grain_agent_core::AgentToolResult,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: grain_agent_core::AgentToolResult,
        is_error: bool,
    },

    // --- harness-own ---
    /// The harness was told to abort the running turn. Fires
    /// regardless of whether anything was actually running.
    ///
    /// Carries the queued messages the abort discarded, mirroring upstream's
    /// `AbortEvent` (`harness/types.ts:934-938` @ 34239180) — a listener that
    /// wants to offer "resend?" needs the messages, not just the fact that
    /// some were dropped.
    Abort {
        cleared_steer: Vec<AgentMessage>,
        cleared_follow_up: Vec<AgentMessage>,
    },
    /// Convenience marker — fired immediately after `AgentEnd` so
    /// callers that only want to know "the turn is done, including
    /// any harness-side post-processing" have a single subscribe
    /// point.
    Settled,
    /// One of the harness queues changed (steer / follow_up / next_turn).
    ///
    /// Upstream reports each bucket's messages
    /// (`harness/types.ts:927-932` @ 34239180); grain reports depths, which
    /// is what a status line actually renders, without cloning transcripts
    /// on every queue mutation.
    QueueUpdate {
        /// True when any bucket is non-empty.
        has_queued: bool,
        steer_count: usize,
        follow_up_count: usize,
        next_turn_count: usize,
    },
    /// `set_model` ran successfully. Pi fires this after every
    /// runtime model swap so listeners (e.g. the TUI status bar)
    /// can re-render.
    ModelSelect {
        model: Model,
    },
    /// `set_thinking_level` ran successfully.
    ThinkingLevelSelect {
        level: ThinkingLevel,
    },
    /// Active-tool subset changed via `set_active_tools`. Carries
    /// the names that are now exposed to the LLM.
    ActiveToolsSelect {
        names: Vec<String>,
    },
    /// `append_entry` ran. Carries the session entry id + type tag —
    /// extensions / persistence layers can use this as a save point
    /// without inspecting the session storage directly.
    AppendEntry {
        entry_id: String,
        type_tag: String,
    },
    /// `compact()` is about to drive a summarization round trip.
    /// Carries the messages that will be summarized (the prefix
    /// before the keep-recent tail).
    SessionBeforeCompact {
        messages: Vec<AgentMessage>,
    },
    /// `compact()` finished successfully. `kept_from` is the session
    /// entry id the new transcript prefix starts from. Pi also
    /// carries the summary text; Phase 3 keeps it minimal —
    /// subscribers wanting the summary can read it from the
    /// preceding `MessageEnd` event.
    SessionCompact {
        kept_from: Option<String>,
    },
    /// `navigate_tree` is about to switch leaves. Both ids are
    /// `Option` because the root leaf is `None`.
    SessionBeforeTree {
        from: Option<String>,
        to: Option<String>,
    },
    /// `navigate_tree` finished — agent transcript now reflects
    /// `current_leaf`'s branch.
    SessionTree {
        current_leaf: Option<String>,
    },
    /// Fired by the harness immediately before kicking off the
    /// agent loop for a new turn (from `prompt_text` / `prompt` /
    /// `prompt_from_template` / `skill`). Mirrors pi's
    /// `before_agent_start` — gives extensions a final chance to
    /// inspect the turn shape. The payload includes the rendered
    /// system prompt, the seeded transcript, and the active tool
    /// names (full `Arc<dyn AgentTool>` references aren't
    /// serializable, hence the name list).
    BeforeAgentStart {
        system_prompt: String,
        messages: Vec<AgentMessage>,
        tool_names: Vec<String>,
    },
    /// `set_resources` ran. Carries counts so the UI can re-render
    /// any palettes that show skills or templates.
    ResourcesUpdate {
        skills: usize,
        templates: usize,
    },
}

impl AgentHarnessEvent {
    fn from_agent_event(e: AgentEvent) -> Self {
        match e {
            AgentEvent::AgentStart => AgentHarnessEvent::AgentStart,
            AgentEvent::AgentEnd { messages } => AgentHarnessEvent::AgentEnd { messages },
            AgentEvent::TurnStart => AgentHarnessEvent::TurnStart,
            AgentEvent::TurnEnd {
                message,
                tool_results,
            } => AgentHarnessEvent::TurnEnd {
                message,
                tool_results,
            },
            AgentEvent::MessageStart { message } => AgentHarnessEvent::MessageStart { message },
            AgentEvent::MessageUpdate {
                message,
                assistant_message_event,
            } => AgentHarnessEvent::MessageUpdate {
                message,
                assistant_message_event,
            },
            AgentEvent::MessageEnd { message } => AgentHarnessEvent::MessageEnd { message },
            AgentEvent::ToolExecutionStart {
                tool_call_id,
                tool_name,
                args,
            } => AgentHarnessEvent::ToolExecutionStart {
                tool_call_id,
                tool_name,
                args,
            },
            AgentEvent::ToolExecutionUpdate {
                tool_call_id,
                tool_name,
                args,
                partial_result,
            } => AgentHarnessEvent::ToolExecutionUpdate {
                tool_call_id,
                tool_name,
                args,
                partial_result,
            },
            AgentEvent::ToolExecutionEnd {
                tool_call_id,
                tool_name,
                result,
                is_error,
            } => AgentHarnessEvent::ToolExecutionEnd {
                tool_call_id,
                tool_name,
                result,
                is_error,
            },
        }
    }
}

/// Listener signature mirrors `grain_agent_core::EventListener` but
/// receives `AgentHarnessEvent`.
pub type HarnessEventListener =
    Arc<dyn Fn(AgentHarnessEvent, CancellationToken) -> BoxFuture<'static, ()> + Send + Sync>;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    #[error("agent: {0}")]
    Agent(#[from] AgentError),
    #[error("session: {0}")]
    Session(#[from] SessionError),
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("compaction: {0}")]
    Compaction(#[from] CompactionError),
    #[error("compact: transcript is empty — nothing to summarize")]
    EmptyTranscript,
    #[error("unknown prompt template: {0}")]
    UnknownTemplate(String),
    #[error("unknown skill: {0}")]
    UnknownSkill(String),
}

// ---------------------------------------------------------------------------
// AgentHarness
// ---------------------------------------------------------------------------

/// Returned by [`AgentHarness::subscribe`]. Dropping or `await`ing
/// `cancel()` removes the listener from the harness's registry.
pub struct HarnessUnsubscribe {
    inner: Arc<Mutex<HarnessInner>>,
    id: u64,
}

impl HarnessUnsubscribe {
    /// Remove this listener from the harness. Safe to call concurrently;
    /// removal is id-based so the wrong listener is never dropped.
    pub async fn cancel(self) {
        self.inner.lock().await.listeners.remove(&self.id);
    }
}

/// A session write deferred until the harness is next idle.
///
/// Port of upstream's `PendingSessionWrite`
/// (`packages/agent/src/harness/types.ts:555-559` @ 34239180: a
/// `SessionTreeEntry` minus `id` / `parentId` / `timestamp`, since those are
/// minted at flush time). Upstream batches exactly the state-change entries
/// below; loop-produced messages are **not** batched — they append
/// synchronously from the `message_end` handler
/// (`agent-harness.ts:539-543`), and grain's session-mirror listener does the
/// same.
#[derive(Debug, Clone)]
enum PendingSessionWrite {
    Model { provider: String, model_id: String },
    ThinkingLevel { thinking_level: String },
    ActiveTools { active_tool_names: Vec<String> },
}

struct HarnessInner {
    listeners: BTreeMap<u64, HarnessEventListener>,
    next_listener_id: u64,
    /// True between a prompt entrypoint and the agent's `AgentEnd`.
    ///
    /// Stands in for upstream's `phase` (`agent-harness.ts:559` et al). Only
    /// the idle/not-idle distinction is modeled, which is all the batching
    /// rule needs — upstream's other phases (`compaction`, `branch_summary`)
    /// exist to reject re-entrant calls, which grain gates differently.
    running: bool,
    /// Session writes deferred while [`Self::running`], drained in FIFO order
    /// once the run settles.
    pending_session_writes: Vec<PendingSessionWrite>,
    /// The third queue bucket. Unlike steer and follow-up (which live on the
    /// `Agent` and feed the *running* loop), this one is owned by the harness
    /// and drains at the start of the next prompt.
    next_turn_queue: Vec<AgentMessage>,
    /// Snapshot of every tool passed to `AgentHarnessOptions::tools`,
    /// kept unfiltered so `set_active_tools` can re-derive the
    /// active subset without losing the full catalog.
    all_tools: Vec<Arc<dyn AgentTool>>,
    /// `None` = all tools active. Otherwise the active subset by
    /// name (validated against `all_tools` at set time).
    active_tool_names: Option<HashSet<String>>,
    /// Shared `StreamFn` clone — `compact()` needs to re-invoke the
    /// summarizer using the same provider the active turn would use.
    stream_fn: StreamFn,
    /// Live skills + prompt templates. Replaced atomically via
    /// `set_resources`; `prompt_from_template` looks up here.
    resources: Resources,
    /// How the system prompt is produced. Kept (not just its rendered
    /// output) so a [`SystemPrompt::Dynamic`] closure can be re-run each
    /// turn, as upstream's `createTurnState` does.
    system_prompt: SystemPrompt,
    /// Mirror of the agent's current model / thinking level, maintained by
    /// the setters. The dynamic-prompt closure needs both, and reading them
    /// from here avoids a circular dependency on the `Agent` the closure is
    /// installed into.
    model: Model,
    thinking_level: ThinkingLevel,
}

/// What an [`AgentHarness::abort`] discarded.
///
/// Port of upstream's `AbortResult` (`agent-harness.ts:1051`): the queued
/// messages that were dropped, so a caller can requeue or report them.
#[derive(Debug, Clone, Default)]
pub struct AbortResult {
    pub cleared_steer: Vec<AgentMessage>,
    pub cleared_follow_up: Vec<AgentMessage>,
}

/// The orchestrator.
pub struct AgentHarness {
    agent: Arc<Agent>,
    session: Session,
    inner: Arc<Mutex<HarnessInner>>,
}

impl AgentHarness {
    /// Build a new harness. Seeds the agent's transcript from the
    /// session's current branch + installs an internal listener that
    /// mirrors every `MessageEnd` back into the session so the
    /// session and the agent stay in sync.
    pub async fn new(options: AgentHarnessOptions) -> Self {
        let AgentHarnessOptions {
            session,
            model,
            stream_fn,
            tools,
            resources,
            system_prompt,
            thinking_level,
            active_tool_names,
            steering_mode,
            follow_up_mode,
            get_api_key,
            transform_context,
            tool_execution,
            session_id,
            transport,
            max_retry_delay_ms,
            on_payload,
            on_response,
            thinking_budgets,
            before_tool_call,
            after_tool_call,
            prepare_next_turn,
            convert_to_llm,
        } = options;

        // Seed the agent with the session's current branch context.
        let seeded_messages = session.build_context().await.messages;

        // Normalize active-tool selection and derive the initial
        // filtered subset that goes into the Agent.
        let active_set: Option<HashSet<String>> =
            active_tool_names.map(|v| v.into_iter().collect());
        let filtered_tools = filter_tools(&tools, active_set.as_ref());

        let stream_fn_for_compact = stream_fn.clone();

        // `inner` is built before the Agent so the dynamic-prompt closure can
        // read live model / thinking / tools / resources without needing a
        // handle on the Agent it is installed into.
        let inner = Arc::new(Mutex::new(HarnessInner {
            listeners: BTreeMap::new(),
            next_listener_id: 0,
            running: false,
            pending_session_writes: Vec::new(),
            next_turn_queue: Vec::new(),
            all_tools: tools,
            active_tool_names: active_set,
            stream_fn: stream_fn_for_compact,
            resources,
            system_prompt,
            model: model.clone(),
            thinking_level,
        }));

        let resolved_system_prompt = render_system_prompt(&inner).await;

        let mut agent_opts = AgentOptions::new(model, stream_fn);
        agent_opts.system_prompt = resolved_system_prompt;
        agent_opts.thinking_level = thinking_level;
        agent_opts.tools = filtered_tools;
        agent_opts.messages = seeded_messages;
        // `convert_to_llm`: caller-supplied wins; otherwise install
        // the harness's custom-message-aware default (routes
        // branchSummary / compactionSummary / custom payloads).
        agent_opts.convert_to_llm = Some(convert_to_llm.unwrap_or_else(|| {
            Arc::new(|messages: Vec<AgentMessage>| {
                Box::pin(async move { convert_to_llm_sync(messages) })
            })
        }));
        agent_opts.transform_context = transform_context;
        agent_opts.get_api_key = get_api_key;
        agent_opts.steering_mode = steering_mode;
        agent_opts.follow_up_mode = follow_up_mode;
        agent_opts.session_id = session_id;
        agent_opts.transport = transport;
        agent_opts.max_retry_delay_ms = max_retry_delay_ms;
        agent_opts.on_payload = on_payload;
        agent_opts.on_response = on_response;
        agent_opts.thinking_budgets = thinking_budgets;
        agent_opts.tool_execution = tool_execution;
        // Provider-agnostic hooks (Phase 3.0): pass through verbatim
        // to the underlying Agent. Lets callers wire storm
        // suppression / error-streak terminator / failure
        // escalation / debug-log capture without dropping
        // AgentHarness's other features.
        agent_opts.before_tool_call = before_tool_call;
        agent_opts.after_tool_call = after_tool_call;
        // Re-render the system prompt between turns. Upstream rebuilds the
        // whole turn state in `prepareNextTurn` and hands the loop a fresh
        // context (`agent-harness.ts:485-494` @ 34239180); grain overlays the
        // freshly rendered prompt onto whatever context the turn is carrying,
        // after any caller-supplied hook has had its say.
        let inner_for_prompt = inner.clone();
        agent_opts.prepare_next_turn =
            Some(Arc::new(move |ctx: PrepareNextTurnContext, cancel| {
                let inner = inner_for_prompt.clone();
                let caller = prepare_next_turn.clone();
                Box::pin(async move {
                    let mut update = match &caller {
                        Some(hook) => hook(ctx.clone(), cancel).await.unwrap_or_default(),
                        None => AgentLoopTurnUpdate::default(),
                    };
                    // A caller hook that swapped model / thinking level must be
                    // visible to the prompt closure that runs next.
                    {
                        let mut g = inner.lock().await;
                        if let Some(m) = &update.model {
                            g.model = m.clone();
                        }
                        if let Some(t) = update.thinking_level {
                            g.thinking_level = t;
                        }
                    }
                    let rendered = render_system_prompt(&inner).await;
                    let mut context = update.context.unwrap_or_else(|| (*ctx.context).clone());
                    if context.system_prompt != rendered {
                        context.system_prompt = rendered;
                    }
                    update.context = Some(context);
                    Some(update)
                })
            }));

        let agent = Arc::new(Agent::new(agent_opts));

        // Install internal session-mirror listener: every finalized
        // message lands back in the session. Mirrors what
        // SessionWriter currently does in headless / TUI manually.
        let session_for_mirror = session.clone();
        agent
            .subscribe(Arc::new(move |event, _signal| {
                let session = session_for_mirror.clone();
                Box::pin(async move {
                    if let AgentEvent::MessageEnd { message } = event
                        && let Err(e) = session.append_message(message).await
                    {
                        // Session persistence failures are non-fatal —
                        // log to stderr like the existing SessionWriter
                        // path does.
                        eprintln!("[warn] harness session append: {e}");
                    }
                })
            }))
            .await;

        // Install fan-out listener: `AgentEvent` → `AgentHarnessEvent`
        // → every harness listener. Tacks on `Settled` after
        // `AgentEnd` so callers have a single end-of-turn signal.
        let inner_for_broadcast = inner.clone();
        let session_for_flush = session.clone();
        agent
            .subscribe(Arc::new(move |event, signal| {
                let inner = inner_for_broadcast.clone();
                let session = session_for_flush.clone();
                Box::pin(async move {
                    let is_end = matches!(event, AgentEvent::AgentEnd { .. });
                    // Upstream flushes deferred writes at `turn_end` and
                    // `agent_end` (agent-harness.ts:551-558), so a long
                    // multi-turn run persists state changes at turn
                    // boundaries rather than hoarding them until the end.
                    let is_turn_end = matches!(event, AgentEvent::TurnEnd { .. });
                    let mapped = AgentHarnessEvent::from_agent_event(event);
                    broadcast(&inner, mapped, signal.clone()).await;
                    if is_turn_end || is_end {
                        if is_end {
                            inner.lock().await.running = false;
                        }
                        flush_pending_session_writes(&inner, &session).await;
                    }
                    if is_end {
                        broadcast(&inner, AgentHarnessEvent::Settled, signal).await;
                    }
                })
            }))
            .await;

        AgentHarness {
            agent,
            session,
            inner,
        }
    }

    /// Convenience: start a new prompt from a string. Emits
    /// [`AgentHarnessEvent::BeforeAgentStart`] before dispatching
    /// to the agent loop.
    pub async fn prompt_text(&self, text: impl Into<String>) -> Result<(), HarnessError> {
        let message = AgentMessage::user(grain_agent_core::UserMessage {
            content: vec![grain_agent_core::UserContent::text(text.into())],
            timestamp: now_ms(),
        });
        self.prompt(vec![message]).await
    }

    /// Start a new prompt from a batch of messages. Emits
    /// [`AgentHarnessEvent::BeforeAgentStart`] before dispatching.
    pub async fn prompt(&self, messages: Vec<AgentMessage>) -> Result<(), HarnessError> {
        self.emit_before_start().await;
        // Queued next-turn messages lead this prompt (upstream
        // `messages = [...queuedMessages, messages[0]!]`, agent-harness.ts:596).
        let mut queued = self.take_next_turn_queue().await;
        let messages = if queued.is_empty() {
            messages
        } else {
            queued.extend(messages);
            queued
        };
        self.mark_running().await;
        self.agent.prompt(messages).await.map_err(Into::into)
    }

    /// Render the named prompt template with `args` and submit the
    /// result as a fresh prompt. Errors with
    /// [`HarnessError::UnknownTemplate`] when the name isn't
    /// registered in [`Resources::prompt_templates`].
    pub async fn prompt_from_template(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<(), HarnessError> {
        let rendered = {
            let g = self.inner.lock().await;
            let tpl = g
                .resources
                .prompt_templates
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| HarnessError::UnknownTemplate(name.to_string()))?;
            (tpl.render)(&args)
        };
        self.prompt_text(rendered).await
    }

    /// Invoke a skill by name.
    ///
    /// Port of upstream `AgentHarness.skill`
    /// (`agent-harness.ts:673-688` @ 34239180): look the name up in
    /// [`Resources::skills`], fail with [`HarnessError::UnknownSkill`] if it
    /// is not there, and submit [`format_skill_invocation`]'s block. The
    /// previous behavior — synthesizing "Use the `x` skill" for *any* name —
    /// meant a typo silently became a prompt asking the model to use a skill
    /// that does not exist.
    ///
    /// `additional_instructions` is appended after the skill block, as
    /// upstream does.
    pub async fn skill(
        &self,
        name: &str,
        additional_instructions: Option<&str>,
    ) -> Result<(), HarnessError> {
        let text = {
            let g = self.inner.lock().await;
            let skill = g
                .resources
                .skills
                .iter()
                .find(|candidate| candidate.name == name)
                .ok_or_else(|| HarnessError::UnknownSkill(name.to_string()))?;
            format_skill_invocation(skill, additional_instructions)
        };
        self.prompt_text(text).await
    }

    /// Swap the harness's resources atomically. Emits
    /// [`AgentHarnessEvent::ResourcesUpdate`] with the new counts.
    pub async fn set_resources(&self, resources: Resources) {
        let (skills, templates) = (resources.skills.len(), resources.prompt_templates.len());
        self.inner.lock().await.resources = resources;
        self.emit(AgentHarnessEvent::ResourcesUpdate { skills, templates })
            .await;
    }

    /// Continue from the current transcript. See
    /// `Agent::continue_` for the contract (the tail message must
    /// convert to a `user` or `toolResult` LLM message).
    pub async fn continue_(&self) -> Result<(), HarnessError> {
        self.agent.continue_().await.map_err(Into::into)
    }

    /// Subscribe to harness events. Returns a [`HarnessUnsubscribe`]
    /// whose `cancel().await` removes the listener. The harness's
    /// internal Agent-listener fans pass-through + harness-own
    /// events to every registered listener, so subscribers see one
    /// unified stream.
    pub async fn subscribe(&self, listener: HarnessEventListener) -> HarnessUnsubscribe {
        let mut g = self.inner.lock().await;
        let id = g.next_listener_id;
        g.next_listener_id += 1;
        g.listeners.insert(id, listener);
        HarnessUnsubscribe {
            inner: self.inner.clone(),
            id,
        }
    }

    /// Replace the active model. Forwards to `Agent::set_model` and
    /// emits [`AgentHarnessEvent::ModelSelect`].
    pub async fn set_model(&self, model: Model) {
        self.agent.set_model(model.clone()).await;
        self.inner.lock().await.model = model.clone();
        self.refresh_system_prompt().await;
        self.record_session_write(PendingSessionWrite::Model {
            provider: model.provider.clone(),
            model_id: model.id.clone(),
        })
        .await;
        self.emit(AgentHarnessEvent::ModelSelect { model }).await;
    }

    /// Replace the system prompt for subsequent turns.
    pub async fn set_system_prompt(&self, prompt: String) {
        self.agent.set_system_prompt(prompt).await;
    }

    /// Replace the thinking level. Forwards to `Agent::set_thinking_level`
    /// and emits [`AgentHarnessEvent::ThinkingLevelSelect`].
    pub async fn set_thinking_level(&self, level: ThinkingLevel) {
        self.agent.set_thinking_level(level).await;
        self.inner.lock().await.thinking_level = level;
        self.refresh_system_prompt().await;
        self.record_session_write(PendingSessionWrite::ThinkingLevel {
            thinking_level: thinking_level_tag(level).to_string(),
        })
        .await;
        self.emit(AgentHarnessEvent::ThinkingLevelSelect { level })
            .await;
    }

    /// Persist a session state change, or defer it when a run is in flight.
    ///
    /// Upstream's rule (`agent-harness.ts:724-734`, `:894`, `:913`, `:940`
    /// @ 34239180): when the harness is idle the write goes straight through;
    /// while a run is in flight it joins `pendingSessionWrites` and lands at
    /// the next turn boundary. Batching keeps mid-run state churn off the
    /// hot path and out of the middle of a turn's entry chain.
    async fn record_session_write(&self, write: PendingSessionWrite) {
        {
            let mut g = self.inner.lock().await;
            if g.running {
                g.pending_session_writes.push(write);
                return;
            }
        }
        flush_one_session_write(&self.inner, &self.session, write).await;
    }

    /// Re-render a [`SystemPrompt::Dynamic`] prompt and push it onto the
    /// agent immediately.
    ///
    /// Reconfiguration is the other half of upstream's re-render trigger: a
    /// `Dynamic` prompt that mentions the model or the active tool set must
    /// not keep describing the previous ones. A `Static` prompt is untouched,
    /// so this never clobbers a caller's `set_system_prompt`.
    async fn refresh_system_prompt(&self) {
        let is_dynamic = matches!(
            self.inner.lock().await.system_prompt,
            SystemPrompt::Dynamic(_)
        );
        if !is_dynamic {
            return;
        }
        let rendered = render_system_prompt(&self.inner).await;
        self.agent.set_system_prompt(rendered).await;
    }

    /// Mark a run as started so subsequent state changes batch rather than
    /// interleave with the turn's entries.
    async fn mark_running(&self) {
        self.inner.lock().await.running = true;
    }

    /// Restrict the agent's visible tool list to the named subset.
    /// Names not present in the original `options.tools` return
    /// [`HarnessError::UnknownTool`] and no change is applied.
    /// Empty slice = no tools active. Pass `None` to reactivate
    /// every tool (no separate method yet — call with all known
    /// names instead).
    pub async fn set_active_tools(&self, names: &[String]) -> Result<(), HarnessError> {
        let new_filtered;
        let new_names;
        {
            let mut g = self.inner.lock().await;
            let known: HashSet<&str> = g
                .all_tools
                .iter()
                .map(|t| t.definition().name.as_str())
                .collect();
            for n in names {
                if !known.contains(n.as_str()) {
                    return Err(HarnessError::UnknownTool(n.clone()));
                }
            }
            g.active_tool_names = Some(names.iter().cloned().collect());
            new_filtered = filter_tools(&g.all_tools, g.active_tool_names.as_ref());
            new_names = names.to_vec();
        }
        self.agent.set_tools(new_filtered).await;
        self.refresh_system_prompt().await;
        self.record_session_write(PendingSessionWrite::ActiveTools {
            active_tool_names: new_names.clone(),
        })
        .await;
        self.emit(AgentHarnessEvent::ActiveToolsSelect { names: new_names })
            .await;
        Ok(())
    }

    /// Queue a steer message (delivered before the next assistant
    /// turn begins). Emits a [`AgentHarnessEvent::QueueUpdate`].
    pub async fn steer(&self, message: AgentMessage) {
        self.agent.steer(message).await;
        self.emit_queue_update().await;
    }

    /// Queue a follow-up message (delivered after the current
    /// assistant turn finishes its tool calls).
    pub async fn follow_up(&self, message: AgentMessage) {
        self.agent.follow_up(message).await;
        self.emit_queue_update().await;
    }

    /// Queue a message to lead the *next* prompt.
    ///
    /// The third bucket, ported from upstream's `nextTurnQueue`
    /// (`agent-harness.ts:719-722`, drained at `:588-597` @ 34239180). It
    /// differs from steer and follow-up in three ways that matter:
    ///
    /// - **Callable while idle.** Steering and follow-up only mean something
    ///   relative to a run in flight; this is "say this next time you go".
    /// - **Drains at the next prompt**, prepended *ahead* of that prompt's
    ///   own message, rather than being consumed mid-run.
    /// - **Always drains fully**, ignoring [`QueueMode`].
    ///
    /// It also survives [`Self::abort`], which clears only the other two.
    pub async fn next_turn(&self, message: AgentMessage) {
        self.inner.lock().await.next_turn_queue.push(message);
        self.emit_queue_update().await;
    }

    /// Drain the next-turn bucket, returning its messages in queue order.
    async fn take_next_turn_queue(&self) -> Vec<AgentMessage> {
        let queued = std::mem::take(&mut self.inner.lock().await.next_turn_queue);
        if !queued.is_empty() {
            self.emit_queue_update().await;
        }
        queued
    }

    async fn emit_queue_update(&self) {
        let (steer_count, follow_up_count) = self.agent.queued_counts().await;
        let next_turn_count = self.inner.lock().await.next_turn_queue.len();
        self.emit(AgentHarnessEvent::QueueUpdate {
            has_queued: steer_count + follow_up_count + next_turn_count > 0,
            steer_count,
            follow_up_count,
            next_turn_count,
        })
        .await;
    }

    // ----- Phase 3 — session control -------------------------------------

    /// Append a free-form custom entry to the session. Use this for
    /// pi-style `appendEntry` extension state — these entries are
    /// session-resident only, NOT projected into the LLM context.
    /// `type_tag` is a free-form discriminator (e.g.
    /// `"extension/my-plugin/note"`).
    pub async fn append_entry(
        &self,
        type_tag: &str,
        data: serde_json::Value,
    ) -> Result<String, HarnessError> {
        let id = self.session.append_custom(type_tag, Some(data)).await?;
        self.emit(AgentHarnessEvent::AppendEntry {
            entry_id: id.clone(),
            type_tag: type_tag.into(),
        })
        .await;
        Ok(id)
    }

    /// Switch the session's active leaf to `target_leaf` (or root if
    /// `None`). Rewrites the agent's transcript to the new branch's
    /// `build_context` result so subsequent turns see the right
    /// history. Emits `SessionBeforeTree` then `SessionTree`.
    pub async fn navigate_tree(&self, target_leaf: Option<String>) -> Result<(), HarnessError> {
        let from = self.session.leaf_id().await;
        self.emit(AgentHarnessEvent::SessionBeforeTree {
            from: from.clone(),
            to: target_leaf.clone(),
        })
        .await;
        self.session
            .storage()
            .set_leaf_id(target_leaf.clone())
            .await?;
        let new_messages = self.session.build_context().await.messages;
        self.agent.set_messages(new_messages).await;
        self.emit(AgentHarnessEvent::SessionTree {
            current_leaf: target_leaf,
        })
        .await;
        Ok(())
    }

    /// Run a compaction pass on the current transcript: summarize
    /// the first `messages.len() - keep_recent` messages via the
    /// active stream/model and replace the transcript with a
    /// `compactionSummary` message + the kept tail. Persists a
    /// `Compaction` entry to the session.
    ///
    /// `keep_recent` is clamped to `[1, total)`. Empty transcripts
    /// return [`HarnessError::EmptyTranscript`].
    pub async fn compact(&self, keep_recent: usize) -> Result<String, HarnessError> {
        let state = self.agent.state().await;
        let messages = state.messages.clone();
        let total = messages.len();
        if total == 0 {
            return Err(HarnessError::EmptyTranscript);
        }
        let keep = keep_recent.max(1).min(total);
        let prefix_len = total - keep;
        // Snap forward past any tool-call / tool-result pairs
        // so the summarizer never sees an orphaned tool_call.
        let prefix_len = crate::compaction::snap_to_safe_boundary(&messages, prefix_len);
        if prefix_len < 2 {
            return Err(HarnessError::EmptyTranscript);
        }

        self.emit(AgentHarnessEvent::SessionBeforeCompact {
            messages: messages[..prefix_len].to_vec(),
        })
        .await;

        let stream_fn = self.inner.lock().await.stream_fn.clone();
        let signal = self
            .agent
            .signal()
            .await
            .unwrap_or_else(CancellationToken::new);
        let new_transcript = compact_transcript(
            &stream_fn,
            &state.model,
            &state.system_prompt,
            &messages,
            prefix_len,
            DEFAULT_COMPACTION_PROMPT,
            signal,
        )
        .await?;

        let first_kept = match first_kept_context_entry_id(&self.session, prefix_len).await {
            Some(id) => id,
            None => self.session.leaf_id().await.unwrap_or_default(),
        };
        let (summary, tokens_before) = match new_transcript.first() {
            Some(AgentMessage::Custom(value)) => (
                value
                    .get("summary")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                value
                    .get("tokensBefore")
                    .and_then(|v| v.as_u64())
                    .unwrap_or_default(),
            ),
            _ => (String::new(), 0),
        };
        let entry_id = self
            .session
            .append_compaction(
                summary,
                first_kept.clone(),
                tokens_before,
                None,
                Some(false),
            )
            .await?;

        self.agent.set_messages(new_transcript).await;
        self.emit(AgentHarnessEvent::SessionCompact {
            kept_from: if first_kept.is_empty() {
                None
            } else {
                Some(first_kept)
            },
        })
        .await;
        Ok(entry_id)
    }

    /// Build and broadcast a `BeforeAgentStart` snapshot. Called by
    /// every prompt entrypoint (`prompt_text`, `prompt`, `skill`,
    /// `prompt_from_template`).
    async fn emit_before_start(&self) {
        let state = self.agent.state().await;
        let tool_names: Vec<String> = state
            .tools
            .iter()
            .map(|t| t.definition().name.clone())
            .collect();
        self.emit(AgentHarnessEvent::BeforeAgentStart {
            system_prompt: state.system_prompt.clone(),
            messages: state.messages.clone(),
            tool_names,
        })
        .await;
    }

    /// Emit a harness-own event to every registered listener. Uses
    /// the active run's cancellation token if any, else a fresh
    /// token (so listeners can `.is_cancelled()` safely).
    async fn emit(&self, event: AgentHarnessEvent) {
        let signal = self
            .agent
            .signal()
            .await
            .unwrap_or_else(CancellationToken::new);
        broadcast(&self.inner, event, signal).await;
    }

    /// Abort the current turn (if any), discarding queued steer and
    /// follow-up messages and returning them.
    ///
    /// Port of upstream `AgentHarness.abort`
    /// (`agent-harness.ts:1024-1051` @ 34239180), in its order: snapshot and
    /// clear both queues, cancel the run, emit `QueueUpdate`, wait for the
    /// run to settle, then emit `Abort`. Waiting before the event means a
    /// listener that reacts to `Abort` observes a genuinely idle harness.
    /// Fires `Abort` even when nothing was running, so listeners have a
    /// stable signal.
    pub async fn abort(&self) -> AbortResult {
        let cleared_steer = self.agent.take_steering_queue().await;
        let cleared_follow_up = self.agent.take_follow_up_queue().await;
        self.agent.abort().await;
        self.emit_queue_update().await;
        self.agent.wait_for_idle().await;
        self.emit(AgentHarnessEvent::Abort {
            cleared_steer: cleared_steer.clone(),
            cleared_follow_up: cleared_follow_up.clone(),
        })
        .await;
        AbortResult {
            cleared_steer,
            cleared_follow_up,
        }
    }

    /// Wait until the agent is idle. Delegates to the core
    /// [`Agent::wait_for_idle`], which resolves only after the current run
    /// fully settles — including awaited async subscribers — with no start
    /// race or polling cadence (patch-5; mirrors TS `waitForIdle`,
    /// agent.ts:321-323).
    pub async fn wait_for_idle(&self) {
        self.agent.wait_for_idle().await;
    }

    /// A clone-cheap handle on the owned session. Internally an
    /// `Arc::clone` — both handles see the same storage.
    pub fn session(&self) -> Session {
        self.session.clone()
    }

    /// Direct access to the wrapped `Agent`. Phase 1 escape hatch —
    /// when callers need behavior that isn't exposed on the harness
    /// yet (e.g. `state()` for debugging). Will be narrowed in later
    /// phases as more of `Agent`'s surface gets first-class harness
    /// methods.
    pub fn agent(&self) -> &Arc<Agent> {
        &self.agent
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn filter_tools(
    all: &[Arc<dyn AgentTool>],
    active: Option<&HashSet<String>>,
) -> Vec<Arc<dyn AgentTool>> {
    match active {
        None => all.to_vec(),
        Some(names) => all
            .iter()
            .filter(|t| names.contains(&t.definition().name))
            .cloned()
            .collect(),
    }
}

async fn broadcast(
    inner: &Arc<Mutex<HarnessInner>>,
    event: AgentHarnessEvent,
    signal: CancellationToken,
) {
    // Snapshot the listener list under a short lock then iterate —
    // never call a listener with the lock held.
    let listeners: Vec<HarnessEventListener> =
        inner.lock().await.listeners.values().cloned().collect();
    for l in listeners {
        l(event.clone(), signal.clone()).await;
    }
}

/// Milliseconds since the Unix epoch, matching how core stamps messages.
fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Render the current system prompt.
///
/// [`SystemPrompt::Static`] returns its string; [`SystemPrompt::Dynamic`]
/// re-runs the closure against live model / thinking level / active tools /
/// resources — upstream's `createTurnState` behavior
/// (`agent-harness.ts:354-387` @ 34239180).
///
/// The closure borrows out of the guard, but the future it returns is
/// `'static`, so the lock is released before awaiting: a prompt closure that
/// itself calls back into the harness cannot deadlock.
async fn render_system_prompt(inner: &Arc<Mutex<HarnessInner>>) -> String {
    let fut = {
        let g = inner.lock().await;
        match &g.system_prompt {
            SystemPrompt::Static(s) => return s.clone(),
            SystemPrompt::Dynamic(render) => {
                let active = filter_tools(&g.all_tools, g.active_tool_names.as_ref());
                let ctx = SystemPromptCtx {
                    model: &g.model,
                    thinking_level: g.thinking_level,
                    active_tools: &active,
                    resources: &g.resources,
                };
                render(&ctx)
            }
        }
    };
    fut.await
}

/// Serialize a [`ThinkingLevel`] to the session's wire tag.
///
/// Matches the lowercase union upstream persists
/// (`"off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"`), which
/// is also what `build_session_context` reads back.
fn thinking_level_tag(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off => "off",
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::XHigh => "xhigh",
        ThinkingLevel::Max => "max",
    }
}

/// Apply a single session write immediately, re-queueing it on failure.
async fn flush_one_session_write(
    inner: &Arc<Mutex<HarnessInner>>,
    session: &Session,
    write: PendingSessionWrite,
) {
    inner.lock().await.pending_session_writes.push(write);
    flush_pending_session_writes(inner, session).await;
}

/// Drain deferred session writes in FIFO order.
///
/// Port of upstream `flushPendingSessionWrites`
/// (`agent-harness.ts:512-536` @ 34239180). Upstream removes an entry from
/// the queue only *after* its write resolves, so a failing write stays at the
/// head and is retried on the next flush; this keeps that ordering by popping
/// the front and pushing it back on failure. Writes happen without the lock
/// held, so a mutator racing the flush appends behind the drained batch
/// rather than deadlocking.
async fn flush_pending_session_writes(inner: &Arc<Mutex<HarnessInner>>, session: &Session) {
    loop {
        let next = {
            let mut g = inner.lock().await;
            if g.pending_session_writes.is_empty() {
                return;
            }
            g.pending_session_writes.remove(0)
        };

        let result = match &next {
            PendingSessionWrite::Model { provider, model_id } => session
                .append_model_change(provider.clone(), model_id.clone())
                .await
                .map(|_| ()),
            PendingSessionWrite::ThinkingLevel { thinking_level } => session
                .append_thinking_level_change(thinking_level.clone())
                .await
                .map(|_| ()),
            PendingSessionWrite::ActiveTools { active_tool_names } => session
                .append_active_tools_change(active_tool_names.clone())
                .await
                .map(|_| ()),
        };

        if let Err(e) = result {
            // Non-fatal, like the message-mirror path: put the entry back at
            // the head so the next flush retries it, and stop draining so a
            // persistent failure doesn't spin.
            eprintln!("[warn] harness pending session write: {e}");
            inner.lock().await.pending_session_writes.insert(0, next);
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{InMemorySessionStorage, SessionMetadata};
    use async_trait::async_trait;
    use grain_agent_core::{
        AssistantContent, AssistantStream, LlmContext, LlmStream, Message, Model, StopReason,
        StreamError, StreamOptions, TextContent, Usage, UserContent, UserMessage,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Tiny stub `LlmStream` impl that emits one `Done` event with a
    /// static "stub response" payload. Lets us drive the harness
    /// lifecycle end-to-end without genai or a real LLM.
    struct StubStream;

    #[async_trait]
    impl LlmStream for StubStream {
        async fn stream(
            &self,
            _model: &Model,
            _context: &LlmContext,
            _options: &StreamOptions,
            _cancel: CancellationToken,
        ) -> Result<AssistantStream, StreamError> {
            use futures::stream;
            let msg = AssistantMessage {
                content: vec![AssistantContent::Text(TextContent {
                    text: "stub response".into(),
                })],
                api: "stub".into(),
                provider: "stub".into(),
                model: "stub-model".into(),
                response_id: None,
                response_model: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                error_code: None,
                timestamp: 0,
                raw_stop_reason: None,
            };
            let evt = grain_agent_core::AssistantMessageEvent::Done { result: msg };
            Ok(Box::pin(stream::once(async move { evt })))
        }
    }

    fn stub_stream_fn() -> StreamFn {
        Arc::new(StubStream)
    }

    fn empty_session() -> Session {
        Session::new(Arc::new(
            InMemorySessionStorage::new(SessionMetadata::new()),
        ))
    }

    fn dummy_model() -> Model {
        Model {
            id: "stub-model".into(),
            name: "stub-model".into(),
            api: "stub".into(),
            provider: "stub".into(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn new_seeds_empty_transcript_from_empty_session() {
        let h = AgentHarness::new(AgentHarnessOptions::new(
            empty_session(),
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        // No previous turns → agent state has an empty transcript.
        let state = h.agent().state().await;
        assert!(state.messages.is_empty());
    }

    #[tokio::test]
    async fn new_seeds_transcript_from_existing_session_messages() {
        let session = empty_session();
        let user = AgentMessage::user(UserMessage {
            content: vec![UserContent::text("prior turn")],
            timestamp: 0,
        });
        session.append_message(user.clone()).await.unwrap();

        let h = AgentHarness::new(AgentHarnessOptions::new(
            session,
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        let state = h.agent().state().await;
        assert_eq!(state.messages.len(), 1);
        // Prior user prompt visible in the agent's transcript.
        match &state.messages[0] {
            AgentMessage::Standard(Message::User(u)) => {
                let UserContent::Text(t) = &u.content[0] else {
                    panic!("expected text content");
                };
                assert_eq!(t.text, "prior turn");
            }
            other => panic!("unexpected seeded message: {other:?}"),
        }
    }

    #[tokio::test]
    async fn subscribe_translates_agent_events_and_emits_settled() {
        let h = AgentHarness::new(AgentHarnessOptions::new(
            empty_session(),
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        let starts = Arc::new(AtomicUsize::new(0));
        let ends = Arc::new(AtomicUsize::new(0));
        let settles = Arc::new(AtomicUsize::new(0));
        let s_clone = starts.clone();
        let e_clone = ends.clone();
        let st_clone = settles.clone();
        h.subscribe(Arc::new(move |event, _signal| {
            let s = s_clone.clone();
            let e = e_clone.clone();
            let st = st_clone.clone();
            Box::pin(async move {
                match event {
                    AgentHarnessEvent::AgentStart => {
                        s.fetch_add(1, Ordering::SeqCst);
                    }
                    AgentHarnessEvent::AgentEnd { .. } => {
                        e.fetch_add(1, Ordering::SeqCst);
                    }
                    AgentHarnessEvent::Settled => {
                        st.fetch_add(1, Ordering::SeqCst);
                    }
                    _ => {}
                }
            })
        }))
        .await;

        h.prompt_text("hello").await.unwrap();
        h.wait_for_idle().await;

        assert_eq!(starts.load(Ordering::SeqCst), 1, "AgentStart fires once");
        assert_eq!(ends.load(Ordering::SeqCst), 1, "AgentEnd fires once");
        assert_eq!(
            settles.load(Ordering::SeqCst),
            1,
            "Settled fires once after AgentEnd"
        );
    }

    #[tokio::test]
    async fn session_handle_shares_storage_after_a_turn() {
        let h = AgentHarness::new(AgentHarnessOptions::new(
            empty_session(),
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        h.prompt_text("hi").await.unwrap();
        h.wait_for_idle().await;
        // The internal mirror listener wrote MessageEnd payloads
        // back to the session. We didn't hand callers a separate
        // copy — the session() handle sees them.
        let entries = h.session().entries().await;
        assert!(
            entries
                .iter()
                .any(|e| matches!(e.kind, crate::session::SessionTreeEntryKind::Message { .. })),
            "expected at least one Message entry, got {entries:?}"
        );
    }

    #[tokio::test]
    async fn abort_is_a_noop_when_nothing_is_running() {
        let h = AgentHarness::new(AgentHarnessOptions::new(
            empty_session(),
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        // Should not panic / hang. Just exercises the path.
        h.abort().await;
        h.wait_for_idle().await;
    }

    // ----- Phase 2 ----------------------------------------------------------

    use grain_agent_core::{AgentToolError, ToolDefinition, ToolUpdateCallback};

    struct StubTool {
        def: ToolDefinition,
    }

    #[async_trait]
    impl AgentTool for StubTool {
        fn definition(&self) -> &ToolDefinition {
            &self.def
        }
        async fn execute(
            &self,
            _tool_call_id: &str,
            _args: serde_json::Value,
            _cancel: CancellationToken,
            _on_update: ToolUpdateCallback,
        ) -> Result<grain_agent_core::AgentToolResult, AgentToolError> {
            Ok(grain_agent_core::AgentToolResult::text("ok"))
        }
    }

    fn stub_tool(name: &str) -> Arc<dyn AgentTool> {
        Arc::new(StubTool {
            def: ToolDefinition {
                name: name.into(),
                label: name.into(),
                description: format!("stub: {name}"),
                parameters: serde_json::Value::Object(Default::default()),
                execution_mode: None,
            },
        })
    }

    #[tokio::test]
    async fn set_model_fires_model_select() {
        let mut opts = AgentHarnessOptions::new(empty_session(), dummy_model(), stub_stream_fn());
        opts.tools = vec![];
        let h = AgentHarness::new(opts).await;
        let count = Arc::new(AtomicUsize::new(0));
        let c = count.clone();
        h.subscribe(Arc::new(move |event, _| {
            let c = c.clone();
            Box::pin(async move {
                if matches!(event, AgentHarnessEvent::ModelSelect { .. }) {
                    c.fetch_add(1, Ordering::SeqCst);
                }
            })
        }))
        .await;
        h.set_model(Model {
            id: "another".into(),
            name: "another".into(),
            api: "stub".into(),
            provider: "stub".into(),
            ..Default::default()
        })
        .await;
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn set_thinking_level_fires_event() {
        let h = AgentHarness::new(AgentHarnessOptions::new(
            empty_session(),
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        let saw = Arc::new(AtomicUsize::new(0));
        let s = saw.clone();
        h.subscribe(Arc::new(move |event, _| {
            let s = s.clone();
            Box::pin(async move {
                if matches!(event, AgentHarnessEvent::ThinkingLevelSelect { .. }) {
                    s.fetch_add(1, Ordering::SeqCst);
                }
            })
        }))
        .await;
        h.set_thinking_level(ThinkingLevel::High).await;
        assert_eq!(saw.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn set_active_tools_filters_agent_tool_list_and_emits_event() {
        let mut opts = AgentHarnessOptions::new(empty_session(), dummy_model(), stub_stream_fn());
        opts.tools = vec![stub_tool("alpha"), stub_tool("beta"), stub_tool("gamma")];
        let h = AgentHarness::new(opts).await;
        // Default: all three visible.
        assert_eq!(h.agent().state().await.tools.len(), 3);

        let saw = Arc::new(AtomicUsize::new(0));
        let s = saw.clone();
        h.subscribe(Arc::new(move |event, _| {
            let s = s.clone();
            Box::pin(async move {
                if matches!(event, AgentHarnessEvent::ActiveToolsSelect { .. }) {
                    s.fetch_add(1, Ordering::SeqCst);
                }
            })
        }))
        .await;

        h.set_active_tools(&["alpha".into(), "gamma".into()])
            .await
            .unwrap();
        assert_eq!(h.agent().state().await.tools.len(), 2);
        assert_eq!(saw.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn set_active_tools_rejects_unknown_names() {
        let mut opts = AgentHarnessOptions::new(empty_session(), dummy_model(), stub_stream_fn());
        opts.tools = vec![stub_tool("real")];
        let h = AgentHarness::new(opts).await;
        let err = h.set_active_tools(&["nope".into()]).await.unwrap_err();
        match err {
            HarnessError::UnknownTool(n) => assert_eq!(n, "nope"),
            other => panic!("unexpected error: {other:?}"),
        }
        // Tool list unchanged.
        assert_eq!(h.agent().state().await.tools.len(), 1);
    }

    #[tokio::test]
    async fn steer_emits_queue_update() {
        let h = AgentHarness::new(AgentHarnessOptions::new(
            empty_session(),
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        let count = Arc::new(AtomicUsize::new(0));
        let c = count.clone();
        h.subscribe(Arc::new(move |event, _| {
            let c = c.clone();
            Box::pin(async move {
                if matches!(
                    event,
                    AgentHarnessEvent::QueueUpdate {
                        has_queued: true,
                        ..
                    }
                ) {
                    c.fetch_add(1, Ordering::SeqCst);
                }
            })
        }))
        .await;
        h.steer(AgentMessage::user(UserMessage {
            content: vec![UserContent::text("nudge")],
            timestamp: 0,
        }))
        .await;
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    // ----- Phase 3 ----------------------------------------------------------

    #[tokio::test]
    async fn append_entry_writes_custom_session_entry_and_emits_event() {
        let h = AgentHarness::new(AgentHarnessOptions::new(
            empty_session(),
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        let saw = Arc::new(AtomicUsize::new(0));
        let s = saw.clone();
        h.subscribe(Arc::new(move |event, _| {
            let s = s.clone();
            Box::pin(async move {
                if matches!(event, AgentHarnessEvent::AppendEntry { .. }) {
                    s.fetch_add(1, Ordering::SeqCst);
                }
            })
        }))
        .await;
        let id = h
            .append_entry("ext/note", serde_json::json!({ "text": "hi" }))
            .await
            .unwrap();
        assert!(!id.is_empty());
        assert_eq!(saw.load(Ordering::SeqCst), 1);
        // Custom entry visible in the session.
        let custom = h
            .session()
            .entries()
            .await
            .into_iter()
            .find(|e| matches!(e.kind, crate::session::SessionTreeEntryKind::Custom { .. }));
        assert!(custom.is_some());
    }

    #[tokio::test]
    async fn navigate_tree_to_none_resets_branch_and_emits_events() {
        let session = empty_session();
        let _ = session
            .append_message(AgentMessage::user(UserMessage {
                content: vec![UserContent::text("first")],
                timestamp: 0,
            }))
            .await
            .unwrap();
        let h = AgentHarness::new(AgentHarnessOptions::new(
            session,
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        let before = Arc::new(AtomicUsize::new(0));
        let after = Arc::new(AtomicUsize::new(0));
        let b = before.clone();
        let a = after.clone();
        h.subscribe(Arc::new(move |event, _| {
            let b = b.clone();
            let a = a.clone();
            Box::pin(async move {
                match event {
                    AgentHarnessEvent::SessionBeforeTree { .. } => {
                        b.fetch_add(1, Ordering::SeqCst);
                    }
                    AgentHarnessEvent::SessionTree { .. } => {
                        a.fetch_add(1, Ordering::SeqCst);
                    }
                    _ => {}
                }
            })
        }))
        .await;
        h.navigate_tree(None).await.unwrap();
        assert_eq!(before.load(Ordering::SeqCst), 1);
        assert_eq!(after.load(Ordering::SeqCst), 1);
        // Agent transcript follows.
        assert!(h.agent().state().await.messages.is_empty());
    }

    #[tokio::test]
    async fn compact_on_empty_transcript_errors() {
        let h = AgentHarness::new(AgentHarnessOptions::new(
            empty_session(),
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        let err = h.compact(1).await.unwrap_err();
        assert!(matches!(err, HarnessError::EmptyTranscript), "{err:?}");
    }

    // ----- Phase 4 ----------------------------------------------------------

    #[tokio::test]
    async fn before_agent_start_fires_with_state_snapshot() {
        let h = AgentHarness::new(AgentHarnessOptions::new(
            empty_session(),
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        let saw = Arc::new(AtomicUsize::new(0));
        let s = saw.clone();
        h.subscribe(Arc::new(move |event, _| {
            let s = s.clone();
            Box::pin(async move {
                if matches!(event, AgentHarnessEvent::BeforeAgentStart { .. }) {
                    s.fetch_add(1, Ordering::SeqCst);
                }
            })
        }))
        .await;
        h.prompt_text("hello").await.unwrap();
        h.wait_for_idle().await;
        assert_eq!(saw.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn prompt_from_template_renders_and_submits() {
        let mut opts = AgentHarnessOptions::new(empty_session(), dummy_model(), stub_stream_fn());
        opts.resources = Resources {
            skills: vec![],
            prompt_templates: vec![PromptTemplate {
                name: "greet".into(),
                description: "Hi".into(),
                render: Arc::new(|args| {
                    let who = args.get("who").and_then(|v| v.as_str()).unwrap_or("world");
                    format!("Hello, {who}!")
                }),
            }],
        };
        let h = AgentHarness::new(opts).await;
        h.prompt_from_template("greet", serde_json::json!({ "who": "Yoda" }))
            .await
            .unwrap();
        h.wait_for_idle().await;
        // The rendered text is what landed as the first user message.
        let msgs = h.agent().state().await.messages;
        let first_user = msgs.iter().find_map(|m| match m {
            AgentMessage::Standard(Message::User(u)) => Some(u),
            _ => None,
        });
        let UserContent::Text(t) = &first_user.unwrap().content[0] else {
            panic!("expected text content");
        };
        assert_eq!(t.text, "Hello, Yoda!");
    }

    #[tokio::test]
    async fn prompt_from_unknown_template_errors() {
        let h = AgentHarness::new(AgentHarnessOptions::new(
            empty_session(),
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        let err = h
            .prompt_from_template("bogus", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(matches!(err, HarnessError::UnknownTemplate(ref n) if n == "bogus"));
    }

    #[tokio::test]
    async fn set_resources_emits_resources_update() {
        let h = AgentHarness::new(AgentHarnessOptions::new(
            empty_session(),
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        let saw = Arc::new(AtomicUsize::new(0));
        let s = saw.clone();
        h.subscribe(Arc::new(move |event, _| {
            let s = s.clone();
            Box::pin(async move {
                if matches!(
                    event,
                    AgentHarnessEvent::ResourcesUpdate {
                        skills: 0,
                        templates: 1,
                    }
                ) {
                    s.fetch_add(1, Ordering::SeqCst);
                }
            })
        }))
        .await;
        h.set_resources(Resources {
            skills: vec![],
            prompt_templates: vec![PromptTemplate {
                name: "x".into(),
                description: "".into(),
                render: Arc::new(|_| String::new()),
            }],
        })
        .await;
        assert_eq!(saw.load(Ordering::SeqCst), 1);
    }

    fn triage_skill() -> Skill {
        Skill {
            name: "triage".into(),
            description: "Triage an incident".into(),
            file_path: "/skills/triage/SKILL.md".into(),
            disable_model_invocation: false,
            body: "Sort by severity.".into(),
        }
    }

    #[tokio::test]
    async fn skill_submits_the_upstream_invocation_block() {
        let mut opts = AgentHarnessOptions::new(empty_session(), dummy_model(), stub_stream_fn());
        opts.resources.skills = vec![triage_skill()];
        let h = AgentHarness::new(opts).await;

        h.skill("triage", Some("focus on paging alerts"))
            .await
            .unwrap();
        h.wait_for_idle().await;

        let msgs = h.agent().state().await.messages;
        let user = msgs.iter().find_map(|m| match m {
            AgentMessage::Standard(Message::User(u)) => Some(u),
            _ => None,
        });
        let UserContent::Text(t) = &user.unwrap().content[0] else {
            panic!("expected a text block");
        };
        assert!(t.text.contains("<skill name=\"triage\""));
        assert!(t.text.contains("location=\"/skills/triage/SKILL.md\""));
        assert!(
            t.text
                .contains("References are relative to /skills/triage.")
        );
        assert!(t.text.contains("Sort by severity."));
        assert!(t.text.ends_with("focus on paging alerts"));
    }

    /// A name that is not in `Resources::skills` is an error, not a prompt
    /// politely asking the model to use a skill that does not exist.
    #[tokio::test]
    async fn unknown_skill_is_rejected() {
        let mut opts = AgentHarnessOptions::new(empty_session(), dummy_model(), stub_stream_fn());
        opts.resources.skills = vec![triage_skill()];
        let h = AgentHarness::new(opts).await;

        match h.skill("triaage", None).await {
            Err(HarnessError::UnknownSkill(name)) => assert_eq!(name, "triaage"),
            other => panic!("expected UnknownSkill, got {other:?}"),
        }
        assert!(
            h.agent().state().await.messages.is_empty(),
            "a rejected skill must not put anything in the transcript"
        );
    }

    #[tokio::test]
    async fn unsubscribe_stops_event_delivery() {
        let h = AgentHarness::new(AgentHarnessOptions::new(
            empty_session(),
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        let count = Arc::new(AtomicUsize::new(0));
        let c = count.clone();
        let unsub = h
            .subscribe(Arc::new(move |_event, _| {
                let c = c.clone();
                Box::pin(async move {
                    c.fetch_add(1, Ordering::SeqCst);
                })
            }))
            .await;
        h.set_thinking_level(ThinkingLevel::High).await;
        assert!(count.load(Ordering::SeqCst) >= 1, "listener fired");
        unsub.cancel().await;
        let before = count.load(Ordering::SeqCst);
        h.set_thinking_level(ThinkingLevel::High).await;
        assert_eq!(count.load(Ordering::SeqCst), before, "no more events");
    }

    #[tokio::test]
    async fn wait_for_idle_returns_immediately_when_idle() {
        let h = AgentHarness::new(AgentHarnessOptions::new(
            empty_session(),
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        // Fresh harness has no active run.
        h.wait_for_idle().await;
    }

    // ---- Phase 3.0: hook pass-through -----------------------------

    #[tokio::test]
    async fn prepare_next_turn_hook_is_plumbed_to_underlying_agent() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let calls = Arc::new(AtomicU32::new(0));
        let calls_for_hook = calls.clone();
        let hook: grain_agent_core::PrepareNextTurnFn = Arc::new(move |_ctx, _cancel| {
            let calls = calls_for_hook.clone();
            Box::pin(async move {
                calls.fetch_add(1, Ordering::Relaxed);
                None
            })
        });

        let mut opts = AgentHarnessOptions::new(empty_session(), dummy_model(), stub_stream_fn());
        opts.prepare_next_turn = Some(hook);
        let h = AgentHarness::new(opts).await;

        h.prompt_text("hello").await.unwrap();
        h.wait_for_idle().await;

        // Stub stream emits one turn (StopReason::Stop, no tool
        // calls); loop fires prepare_next_turn after that turn before
        // deciding to stop.
        assert!(
            calls.load(Ordering::Relaxed) >= 1,
            "prepare_next_turn hook should have been invoked at least once"
        );
    }

    #[tokio::test]
    async fn convert_to_llm_default_routes_custom_messages() {
        // No caller-supplied convert_to_llm → harness installs its
        // own custom-message-aware default. This is a smoke check
        // that nothing panics + the agent state seeds normally.
        let h = AgentHarness::new(AgentHarnessOptions::new(
            empty_session(),
            dummy_model(),
            stub_stream_fn(),
        ))
        .await;
        h.prompt_text("hi").await.unwrap();
        h.wait_for_idle().await;
    }

    #[tokio::test]
    async fn convert_to_llm_user_override_replaces_default() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let calls = Arc::new(AtomicU32::new(0));
        let calls_for_hook = calls.clone();
        let user_fn: grain_agent_core::ConvertToLlmFn =
            Arc::new(move |messages: Vec<AgentMessage>| {
                let calls = calls_for_hook.clone();
                Box::pin(async move {
                    calls.fetch_add(1, Ordering::Relaxed);
                    // Same projection as the default but with a
                    // side-effect to prove the override wins.
                    messages
                        .into_iter()
                        .filter_map(|m| match m {
                            AgentMessage::Standard(msg) => Some(msg),
                            AgentMessage::Custom(_) => None,
                        })
                        .collect()
                })
            });
        let mut opts = AgentHarnessOptions::new(empty_session(), dummy_model(), stub_stream_fn());
        opts.convert_to_llm = Some(user_fn);
        let h = AgentHarness::new(opts).await;
        h.prompt_text("hi").await.unwrap();
        h.wait_for_idle().await;
        assert!(
            calls.load(Ordering::Relaxed) >= 1,
            "user-supplied convert_to_llm must replace the default"
        );
    }
}
