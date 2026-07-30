//! Rehydrating an [`Agent`] from a persisted session.
//!
//! This is the durability path: a worker agent whose host process died comes
//! back with its transcript intact by loading its session from disk and
//! seeding a fresh [`Agent`] from it.
//!
//! Upstream reaches the same state through `AgentHarness`'s constructor, which
//! calls `session.buildContext()` and feeds the result into the agent it
//! builds (`packages/agent/src/harness/agent-harness.ts:354-387` @ 34239180).
//! [`crate::AgentHarness`] already does this. What this module adds is the
//! plain-[`Agent`] route for embedders that do not want the whole harness, and
//! a single call that goes from a **session id** to a live agent —
//! [`JsonlSessionRepo::resume_agent`](crate::JsonlSessionRepo::resume_agent).
//!
//! ## What is and is not restored automatically
//!
//! Restored onto the agent:
//! - **messages** — the full branch, projected through
//!   [`build_session_context`](crate::session::build_session_context), so
//!   compaction boundaries and custom messages are honored exactly as a live
//!   run would see them.
//! - **thinking level** — reconstructible from its wire tag alone.
//! - **active tools** — the caller's tool list filtered to the recorded names.
//!   Recorded names with no matching tool are reported in
//!   [`RestoredState::unknown_tool_names`] rather than failing the resume: the
//!   tool catalog is a property of the *process*, and a rebuilt process may
//!   legitimately ship a different one.
//!
//! **Not** restored automatically: the **model**. A session records only
//! `(provider, model_id)`, while [`Model`] additionally carries context
//! window, pricing and limits — data that lives in a model registry, not the
//! transcript. Inventing those fields would produce an agent that silently
//! misprices and mis-budgets, so the recorded pair is handed back in
//! [`RestoredState::model`] for the embedder to resolve through its own
//! registry. Upstream has the same split; it just always has a `ModelRuntime`
//! on hand to do the lookup.

use std::sync::Arc;

use grain_agent_core::{Agent, AgentEvent, AgentOptions, ThinkingLevel};

use crate::session::{Session, SessionContext, SessionError};

/// What a resume recovered from the session, beyond the transcript itself.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RestoredState {
    /// Number of messages seeded onto the agent.
    pub message_count: usize,
    /// Thinking level recorded in the session, already applied to the agent.
    pub thinking_level: ThinkingLevel,
    /// `(provider, model_id)` last recorded in the session, if any.
    ///
    /// **Not applied** — resolve it through your model registry and call
    /// [`Agent::set_model`] if it differs from the one you supplied. See the
    /// module docs for why.
    pub model: Option<(String, String)>,
    /// Active tool names recorded in the session, or `None` when the session
    /// never restricted the tool set (meaning: all tools active).
    pub active_tool_names: Option<Vec<String>>,
    /// Recorded active-tool names with no matching tool in the supplied
    /// catalog. Empty in the common case; non-empty means this process ships
    /// a different tool set than the one that wrote the session.
    pub unknown_tool_names: Vec<String>,
}

/// An [`Agent`] rebuilt from a session, plus the session handle it came from.
pub struct ResumedAgent {
    /// The rehydrated agent, already carrying the session's history.
    pub agent: Agent,
    /// The session the agent was rebuilt from. Keep it to continue appending.
    pub session: Session,
    /// What was recovered, and what the caller still has to apply.
    pub restored: RestoredState,
}

/// Parse a thinking-level wire tag, defaulting to [`ThinkingLevel::Off`].
///
/// The tags are upstream's lowercase union
/// (`"off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"`). An
/// unrecognized tag degrades to `Off` rather than failing the resume — a
/// transcript written by a newer build must still load.
pub fn thinking_level_from_tag(tag: &str) -> ThinkingLevel {
    match tag {
        "minimal" => ThinkingLevel::Minimal,
        "low" => ThinkingLevel::Low,
        "medium" => ThinkingLevel::Medium,
        "high" => ThinkingLevel::High,
        "xhigh" => ThinkingLevel::XHigh,
        "max" => ThinkingLevel::Max,
        _ => ThinkingLevel::Off,
    }
}

/// Apply a session's derived context onto agent options, in place.
///
/// Seeds `messages`, sets `thinking_level`, and filters `tools` down to the
/// recorded active set. Returns what was recovered.
pub fn seed_options_from_context(
    options: &mut AgentOptions,
    context: SessionContext,
) -> RestoredState {
    let SessionContext {
        messages,
        thinking_level,
        model,
        active_tool_names,
    } = context;

    let message_count = messages.len();
    options.messages = messages;

    let level = thinking_level_from_tag(&thinking_level);
    options.thinking_level = level;

    let mut unknown_tool_names = Vec::new();
    if let Some(names) = &active_tool_names {
        let available: Vec<String> = options
            .tools
            .iter()
            .map(|t| t.definition().name.clone())
            .collect();
        for name in names {
            if !available.contains(name) {
                unknown_tool_names.push(name.clone());
            }
        }
        options
            .tools
            .retain(|t| names.contains(&t.definition().name));
    }

    RestoredState {
        message_count,
        thinking_level: level,
        model,
        active_tool_names,
        unknown_tool_names,
    }
}

/// Rebuild an [`Agent`] from an already-open [`Session`], wired so that
/// everything it goes on to produce is persisted back.
///
/// The supplied `options` provide everything the transcript cannot: the stream
/// function, tool catalog, system prompt, hooks. Session-derived state is
/// layered on top per [`seed_options_from_context`].
///
/// # The returned agent writes back
///
/// A resumed agent installs the same session-mirror listener
/// [`crate::AgentHarness`] does: every finalized message is appended to the
/// session as it completes. Without it a resumed agent would read history and
/// then write none of its own — recovering correctly from the first restart
/// and losing everything after it, silently, because the in-memory transcript
/// still looks right. Durability that survives exactly one restart is worse
/// than none, since nothing surfaces the loss.
///
/// Persistence failures are non-fatal and logged, matching the harness: a full
/// disk should not abort a turn that is otherwise working.
pub async fn resume_agent_from_session(
    session: &Session,
    mut options: AgentOptions,
) -> Result<ResumedAgent, SessionError> {
    let restored = seed_options_from_context(&mut options, session.build_context().await);
    let agent = Agent::new(options);

    let session_for_mirror = session.clone();
    agent
        .subscribe(Arc::new(move |event, _signal| {
            let session = session_for_mirror.clone();
            Box::pin(async move {
                if let AgentEvent::MessageEnd { message } = event
                    && let Err(e) = session.append_message(message).await
                {
                    eprintln!("[warn] resumed agent session append: {e}");
                }
            })
        }))
        .await;

    Ok(ResumedAgent {
        agent,
        session: session.clone(),
        restored,
    })
}
