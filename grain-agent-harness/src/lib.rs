//! `grain-agent-harness` — Rust port of the `@earendil-works/pi-agent-core` harness layer.
//!
//! Sits on top of [`grain-agent-core`](::grain_agent_core) and adds the engineering
//! plumbing needed to ship a real agent process:
//!
//! - [`session`] — typed session tree (entries, branches, leaf cursor) and an
//!   `InMemorySessionRepo`.
//! - [`session_jsonl`] — on-disk JSONL persistence behind the same traits, and
//!   the durability path a worker uses to survive a host restart.
//! - [`resume`] — rehydrate an `Agent` from a persisted session.
//! - [`agent_harness`] — the top-level `AgentHarness` orchestrator.
//! - [`compaction`], [`context_guard`], [`pruning`], [`retry_overflow`],
//!   [`escalation`], [`prefix_pin`], [`repair`] — context-window and
//!   failure-handling policy layers.
//! - [`messages`] — custom-message helpers (`branch_summary`, `compaction_summary`,
//!   `custom_message`) and the harness-aware `convert_to_llm`.
//! - [`system_prompt`] — assembles skill descriptors into the XML block injected
//!   into the system prompt.
//! - [`truncate`] — head/tail truncation utilities for tool output.
//!
//! What is NOT in this crate (see TS source for the reference behavior):
//! - skills **loading from disk** (`harness/skills.ts`) — the loader lives in
//!   `grain-ai-agent-headless`; this crate only renders already-loaded skills
//!   into the system prompt via [`system_prompt`].
//! - execution environment (`harness/env/*`) and shell-output capture.
//! - upstream's harness provider-hook *triad*
//!   (`before_provider_request`, `before_provider_payload`,
//!   `after_provider_response`) as a subscriber chain with patch-merge
//!   semantics. The narrow single-callback form is wired:
//!   `AgentHarnessOptions` forwards `on_payload` / `on_response` /
//!   `thinking_budgets` to `grain_agent_core::StreamOptions`.

// ---------------------------------------------------------------------------
// Pre-existing clippy debt (DEBT.md G33), scoped so CI can run `-D warnings`.
//
// CI lints the workspace with `cargo clippy --workspace --all-targets --
// -D warnings`. This crate carries 2 findings that predate that job. Rather
// than weaken the gate to report-only — which would never catch anything —
// each lint is allowed HERE and ONLY here, so:
//   - any OTHER lint in this crate still fails CI;
//   - these same lints still fail CI in every other crate;
//   - closing G33 means deleting the two attributes below, nothing else.
//
// These are crate-root attributes rather than Cargo.toml `[lints]` on
// purpose: manifest lint levels are emitted BEFORE the trailing `-D warnings`
// on clippy's command line, so rustc's last-flag-wins ordering lets
// `-D warnings` override them. Source attributes are scoped inside the crate
// and take precedence over command-line levels, so they actually hold.
//
// Owner: WP19 grain-product backlog (G33). Do not add entries here to
// silence NEW findings — fix those instead.
// ---------------------------------------------------------------------------
// src/session.rs:164 — large size difference between `SessionEntry` variants.
#![allow(clippy::large_enum_variant)]
// src/session.rs:637 — `if` collapsible into the outer `match`.
#![allow(clippy::collapsible_match)]

pub mod agent_harness;
pub mod compaction;
pub mod context_guard;
pub mod escalation;
pub mod messages;
pub mod prefix_pin;
pub mod pruning;
pub mod repair;
pub mod resume;
pub mod retry_overflow;
pub mod session;
pub mod session_jsonl;
pub mod system_prompt;
pub mod truncate;

pub use agent_harness::{
    AbortResult, AgentHarness, AgentHarnessEvent, AgentHarnessOptions, DynamicSystemPromptFn,
    HarnessError, HarnessEventListener, HarnessUnsubscribe, PromptTemplate, Resources,
    SystemPrompt, SystemPromptCtx,
};

pub use compaction::{
    CompactionError, CompactionNotice, CompactionNoticePhase, CompactionNotifyFn, CompactionPolicy,
    CompactionSettings, CompactionSettingsResolver, DEFAULT_COMPACTION_PROMPT,
    DEFAULT_COMPACTION_SETTINGS, DEFAULT_KEEP_RECENT, DEFAULT_MESSAGE_THRESHOLD,
    DEFAULT_TOOL_RESULT_TRUNCATION_CAP_CHARS, MessageCountPolicy, TokenBudgetPolicy,
    compact_transcript, compaction_prepare_next_turn, compaction_prepare_next_turn_with_notify,
    resolve_threshold_tokens, semantic_compress, should_compact, snap_to_safe_boundary,
    tool_result_truncation_hook, truncate_tool_results,
};
pub use context_guard::{
    ActiveModelHandle, ActiveModelInfo, ContextGuard, ContextGuardPolicy, TokenEstimator,
};
pub use escalation::{
    EscalationConfig, EscalationState, count_failures, decide_escalation, failure_escalation_hook,
};
pub use messages::{
    BRANCH_SUMMARY_PREFIX, BRANCH_SUMMARY_SUFFIX, BashExecutionMessage, BranchSummaryMessage,
    COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX, CompactionSummaryMessage, CustomMessage,
    bash_execution_message, bash_execution_to_text, branch_summary_message,
    compaction_summary_message, convert_to_llm, custom_message,
};
pub use prefix_pin::{PinnedSystemPrompt, append_only_guard};
pub use pruning::{PruneConfig, PruneOutcome, prune_tool_outputs};
pub use repair::{StormConfig, storm_hook};
pub use resume::{
    RestoredState, ResumedAgent, resume_agent_from_session, seed_options_from_context,
    thinking_level_from_tag,
};
pub use retry_overflow::{
    OverflowDetector, RetryNotify, RetryOnOverflowConfig, RetryOnOverflowStream,
};
pub use session::{
    ForkPosition, InMemorySessionRepo, InMemorySessionStorage, Session, SessionContext,
    SessionError, SessionMetadata, SessionRepo, SessionStorage, SessionTreeEntry,
    SessionTreeEntryKind, uuidv7,
};
pub use session_jsonl::{JsonlSessionRepo, JsonlSessionStorage};
pub use system_prompt::{Skill, format_skill_invocation, format_skills_for_system_prompt};
pub use truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, GREP_MAX_LINE_LENGTH, TruncationOptions,
    TruncationResult, format_size, truncate_head, truncate_line, truncate_tail,
};
