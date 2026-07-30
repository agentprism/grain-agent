//! G10 — on-disk JSONL session durability and Agent rehydration.
//!
//! The property under test is the one worker durability rests on: a session
//! written to disk by one process is recoverable, in full, by the next
//! process — including after a crash mid-write.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use grain_agent_core::{
    AgentMessage, AgentOptions, AgentTool, AgentToolError, AgentToolResult, AssistantContent,
    AssistantMessage, AssistantMessageEvent, AssistantStream, Cost, LlmContext, LlmStream, Model,
    StopReason, StreamError, StreamFn, StreamOptions, TextContent, ThinkingLevel, ToolDefinition,
    ToolUpdateCallback, Usage, UserContent, UserMessage,
};
use grain_agent_harness::{
    AgentHarness, AgentHarnessOptions, JsonlSessionRepo, Session, SessionRepo,
};
use tokio_util::sync::CancellationToken;

fn model() -> Model {
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

fn user(text: &str) -> AgentMessage {
    AgentMessage::user(UserMessage {
        content: vec![UserContent::text(text)],
        timestamp: 0,
    })
}

/// Stream fn that always replies with one fixed line of text.
struct Replier(String);

#[async_trait]
impl LlmStream for Replier {
    async fn stream(
        &self,
        model: &Model,
        _context: &LlmContext,
        _options: &StreamOptions,
        _cancel: CancellationToken,
    ) -> Result<AssistantStream, StreamError> {
        let msg = AssistantMessage {
            content: vec![AssistantContent::Text(TextContent {
                text: self.0.clone(),
            })],
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_id: None,
            response_model: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            raw_stop_reason: None,
            error_message: None,
            error_code: None,
            timestamp: 0,
        };
        Ok(futures::stream::iter(vec![AssistantMessageEvent::Done { result: msg }]).boxed())
    }
}

fn replier(text: &str) -> StreamFn {
    Arc::new(Replier(text.to_string()))
}

struct NoopTool(ToolDefinition);

#[async_trait]
impl AgentTool for NoopTool {
    fn definition(&self) -> &ToolDefinition {
        &self.0
    }
    async fn execute(
        &self,
        _id: &str,
        _args: serde_json::Value,
        _cancel: CancellationToken,
        _on_update: ToolUpdateCallback,
    ) -> Result<AgentToolResult, AgentToolError> {
        Ok(AgentToolResult::default())
    }
}

fn tool(name: &str) -> Arc<dyn AgentTool> {
    Arc::new(NoopTool(ToolDefinition {
        name: name.into(),
        label: name.into(),
        description: String::new(),
        parameters: serde_json::json!({ "type": "object" }),
        execution_mode: None,
    }))
}

fn agent_options(stream: StreamFn) -> AgentOptions {
    AgentOptions::new(model(), stream)
}

fn text_of(message: &AgentMessage) -> String {
    match message {
        AgentMessage::Standard(grain_agent_core::Message::User(u)) => u
            .content
            .iter()
            .map(|c| match c {
                UserContent::Text(t) => t.text.clone(),
                UserContent::Image(_) => String::new(),
            })
            .collect(),
        AgentMessage::Standard(grain_agent_core::Message::Assistant(a)) => a
            .content
            .iter()
            .map(|c| match c {
                AssistantContent::Text(t) => t.text.clone(),
                _ => String::new(),
            })
            .collect(),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Round trip through a simulated process restart
// ---------------------------------------------------------------------------

/// Run a turn, drop every handle (the process dies), then reopen the session
/// by id in a fresh repo and get an agent that still knows the conversation.
#[tokio::test]
async fn session_round_trips_through_a_process_restart() {
    let dir = tempfile::tempdir().unwrap();

    // ---- process 1 -------------------------------------------------------
    {
        let repo = JsonlSessionRepo::new(dir.path()).unwrap();
        let session = repo.create(Some("worker-7".into())).await.unwrap();
        let harness = AgentHarness::new(AgentHarnessOptions::new(
            session,
            model(),
            replier("first answer"),
        ))
        .await;
        harness.prompt_text("first question").await.unwrap();
        harness.wait_for_idle().await;
        // Everything above goes out of scope here: the file lock is released
        // exactly as it would be when the process exits.
    }

    // ---- process 2 -------------------------------------------------------
    let repo = JsonlSessionRepo::new(dir.path()).unwrap();
    let resumed = repo
        .resume_agent("worker-7", agent_options(replier("second answer")))
        .await
        .expect("session must be resumable by id");

    let texts: Vec<String> = resumed
        .agent
        .state()
        .await
        .messages
        .iter()
        .map(text_of)
        .collect();
    assert!(
        texts.iter().any(|t| t.contains("first question")),
        "user turn must survive the restart, got {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("first answer")),
        "assistant turn must survive the restart, got {texts:?}"
    );
    assert_eq!(resumed.restored.message_count, texts.len());

    // The rehydrated agent keeps appending to the same history.
    resumed.agent.prompt_text("second question").await.unwrap();
    let after: Vec<String> = resumed
        .agent
        .state()
        .await
        .messages
        .iter()
        .map(text_of)
        .collect();
    assert!(after.iter().any(|t| t.contains("first answer")));
    assert!(after.iter().any(|t| t.contains("second answer")));
}

/// Resuming a session id that was never written is a clean `NotFound`, not a
/// silently-empty agent.
#[tokio::test]
async fn resuming_an_unknown_session_is_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let repo = JsonlSessionRepo::new(dir.path()).unwrap();
    match repo
        .resume_agent("ghost", agent_options(replier("x")))
        .await
    {
        Ok(_) => panic!("expected NotFound for a session that was never written"),
        Err(grain_agent_harness::SessionError::NotFound(_)) => {}
        Err(other) => panic!("expected NotFound, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Crash during write
// ---------------------------------------------------------------------------

/// Truncate the file mid-entry — the shape a crash during `write` leaves —
/// and confirm the reopen degrades to "all complete entries" rather than
/// losing the file or surfacing a half-parsed record.
#[tokio::test]
async fn truncated_final_line_degrades_to_all_complete_entries() {
    let dir = tempfile::tempdir().unwrap();
    let entries_path = dir.path().join("crashy").join("entries.jsonl");

    let (id_a, id_b) = {
        let repo = JsonlSessionRepo::new(dir.path()).unwrap();
        let session = repo.create(Some("crashy".into())).await.unwrap();
        let a = session.append_message(user("alpha")).await.unwrap();
        let b = session.append_message(user("bravo")).await.unwrap();
        session.append_message(user("charlie")).await.unwrap();
        (a, b)
    };

    // Chop the file partway through the third entry: a complete prefix, then
    // a fragment with no trailing newline.
    let raw = tokio::fs::read_to_string(&entries_path).await.unwrap();
    let lines: Vec<&str> = raw.lines().collect();
    assert_eq!(lines.len(), 3, "expected one line per entry");
    let keep = format!("{}\n{}\n", lines[0], lines[1]);
    let fragment = &lines[2][..lines[2].len() / 2];
    tokio::fs::write(&entries_path, format!("{keep}{fragment}"))
        .await
        .unwrap();

    let repo = JsonlSessionRepo::new(dir.path()).unwrap();
    let session = repo
        .open(&grain_agent_harness::SessionMetadata::with_id("crashy"))
        .await
        .expect("a truncated tail must not make the session unopenable");

    let entries = session.entries().await;
    assert_eq!(
        entries.len(),
        2,
        "the two complete entries survive; the fragment is dropped"
    );
    assert_eq!(entries[0].id, id_a);
    assert_eq!(entries[1].id, id_b);
    assert_eq!(
        session.leaf_id().await.as_deref(),
        Some(id_b.as_str()),
        "leaf must fall back to the last complete entry"
    );

    // The session is still writable, and the new entry chains onto the last
    // complete one rather than the lost fragment.
    let id_d = session.append_message(user("delta")).await.unwrap();
    let branch = session.branch(None).await;
    assert_eq!(branch.len(), 3);
    assert_eq!(branch[2].id, id_d);
    assert_eq!(branch[2].parent_id.as_deref(), Some(id_b.as_str()));

    // And it survives another restart, with no corruption carried forward.
    drop(session);
    let repo = JsonlSessionRepo::new(dir.path()).unwrap();
    let reopened = repo
        .open(&grain_agent_harness::SessionMetadata::with_id("crashy"))
        .await
        .unwrap();
    assert_eq!(reopened.entries().await.len(), 3);
}

/// An empty (zero-byte) entries file is a valid brand-new session, not a
/// corrupt one — the state a crash between create and first append leaves.
#[tokio::test]
async fn empty_entries_file_opens_as_a_fresh_session() {
    let dir = tempfile::tempdir().unwrap();
    {
        let repo = JsonlSessionRepo::new(dir.path()).unwrap();
        repo.create(Some("blank".into())).await.unwrap();
    }
    tokio::fs::write(dir.path().join("blank").join("entries.jsonl"), "")
        .await
        .unwrap();

    let repo = JsonlSessionRepo::new(dir.path()).unwrap();
    let session = repo
        .open(&grain_agent_harness::SessionMetadata::with_id("blank"))
        .await
        .unwrap();
    assert!(session.entries().await.is_empty());
    assert_eq!(session.leaf_id().await, None);
    session.append_message(user("first")).await.unwrap();
    assert_eq!(session.entries().await.len(), 1);
}

/// A crash *between* the entry append and the leaf-cursor update leaves
/// `state.json` pointing at the previous entry; the JSONL is authoritative.
#[tokio::test]
async fn stale_leaf_cursor_recovers_from_the_entry_log() {
    let dir = tempfile::tempdir().unwrap();
    let last = {
        let repo = JsonlSessionRepo::new(dir.path()).unwrap();
        let session = repo.create(Some("stale".into())).await.unwrap();
        session.append_message(user("one")).await.unwrap();
        session.append_message(user("two")).await.unwrap()
    };

    // Rewind the cursor to nothing, as if the state write never landed.
    tokio::fs::write(
        dir.path().join("stale").join("state.json"),
        r#"{"leafId":null}"#,
    )
    .await
    .unwrap();

    let repo = JsonlSessionRepo::new(dir.path()).unwrap();
    let session = repo
        .open(&grain_agent_harness::SessionMetadata::with_id("stale"))
        .await
        .unwrap();
    assert_eq!(session.leaf_id().await.as_deref(), Some(last.as_str()));
}

/// Interleaved appends land as one entry per line, so a reader that stops at
/// any newline boundary sees a consistent prefix.
#[tokio::test]
async fn every_entry_occupies_exactly_one_line() {
    let dir = tempfile::tempdir().unwrap();
    {
        let repo = JsonlSessionRepo::new(dir.path()).unwrap();
        let session = repo.create(Some("lines".into())).await.unwrap();
        for i in 0..5 {
            // Embedded newlines in payloads must not break the framing.
            session
                .append_message(user(&format!("line {i}\nwith an embedded newline")))
                .await
                .unwrap();
        }
    }
    let raw = tokio::fs::read_to_string(dir.path().join("lines").join("entries.jsonl"))
        .await
        .unwrap();
    assert_eq!(raw.lines().count(), 5);
    assert!(raw.ends_with('\n'), "a complete entry ends with a newline");
    for line in raw.lines() {
        serde_json::from_str::<serde_json::Value>(line).expect("each line parses on its own");
    }
}

// ---------------------------------------------------------------------------
// State-change persistence + batching
// ---------------------------------------------------------------------------

/// While idle, a state change writes straight through; the next process sees
/// it. Upstream: `agent-harness.ts:891-894`.
#[tokio::test]
async fn idle_state_changes_persist_immediately() {
    let dir = tempfile::tempdir().unwrap();
    {
        let repo = JsonlSessionRepo::new(dir.path()).unwrap();
        let session = repo.create(Some("s".into())).await.unwrap();
        let harness =
            AgentHarness::new(AgentHarnessOptions::new(session, model(), replier("hi"))).await;
        harness.set_thinking_level(ThinkingLevel::High).await;
    }

    let repo = JsonlSessionRepo::new(dir.path()).unwrap();
    let resumed = repo
        .resume_agent("s", agent_options(replier("hi")))
        .await
        .unwrap();
    assert_eq!(
        resumed.restored.thinking_level,
        ThinkingLevel::High,
        "thinking level must survive a restart"
    );
    assert_eq!(
        resumed.agent.state().await.thinking_level,
        ThinkingLevel::High
    );
}

/// A model change made while idle is recorded for the next process to resolve.
#[tokio::test]
async fn model_change_is_recorded_for_the_next_process() {
    let dir = tempfile::tempdir().unwrap();
    {
        let repo = JsonlSessionRepo::new(dir.path()).unwrap();
        let session = repo.create(Some("m".into())).await.unwrap();
        let harness =
            AgentHarness::new(AgentHarnessOptions::new(session, model(), replier("hi"))).await;
        harness
            .set_model(Model {
                id: "claude-sonnet-4".into(),
                provider: "anthropic".into(),
                ..model()
            })
            .await;
    }

    let repo = JsonlSessionRepo::new(dir.path()).unwrap();
    let resumed = repo
        .resume_agent("m", agent_options(replier("hi")))
        .await
        .unwrap();
    assert_eq!(
        resumed.restored.model,
        Some(("anthropic".into(), "claude-sonnet-4".into())),
        "the recorded provider/model pair is handed back for registry lookup"
    );
}

/// Active-tool selection survives, and the rehydrated agent's tool list is
/// filtered to it.
#[tokio::test]
async fn active_tool_selection_survives_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let repo = JsonlSessionRepo::new(dir.path()).unwrap();
        let session = repo.create(Some("t".into())).await.unwrap();
        let mut opts = AgentHarnessOptions::new(session, model(), replier("hi"));
        opts.tools = vec![tool("read"), tool("write"), tool("bash")];
        let harness = AgentHarness::new(opts).await;
        harness
            .set_active_tools(&["read".to_string(), "bash".to_string()])
            .await
            .unwrap();
    }

    let repo = JsonlSessionRepo::new(dir.path()).unwrap();
    let mut options = agent_options(replier("hi"));
    options.tools = vec![tool("read"), tool("write"), tool("bash")];
    let resumed = repo.resume_agent("t", options).await.unwrap();

    assert_eq!(
        resumed.restored.active_tool_names,
        Some(vec!["read".to_string(), "bash".to_string()])
    );
    let mut names: Vec<String> = resumed
        .agent
        .state()
        .await
        .tools
        .iter()
        .map(|t| t.definition().name.clone())
        .collect();
    names.sort();
    assert_eq!(names, vec!["bash".to_string(), "read".to_string()]);
    assert!(resumed.restored.unknown_tool_names.is_empty());
}

/// A rebuilt process shipping a different tool catalog still resumes; the
/// names it can no longer honor are reported rather than silently dropped.
#[tokio::test]
async fn unknown_recorded_tools_are_reported_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    {
        let repo = JsonlSessionRepo::new(dir.path()).unwrap();
        let session = repo.create(Some("u".into())).await.unwrap();
        let mut opts = AgentHarnessOptions::new(session, model(), replier("hi"));
        opts.tools = vec![tool("read"), tool("retired_tool")];
        let harness = AgentHarness::new(opts).await;
        harness
            .set_active_tools(&["read".to_string(), "retired_tool".to_string()])
            .await
            .unwrap();
    }

    let repo = JsonlSessionRepo::new(dir.path()).unwrap();
    let mut options = agent_options(replier("hi"));
    options.tools = vec![tool("read")]; // `retired_tool` no longer shipped
    let resumed = repo.resume_agent("u", options).await.unwrap();

    assert_eq!(resumed.restored.unknown_tool_names, vec!["retired_tool"]);
    let names: Vec<String> = resumed
        .agent
        .state()
        .await
        .tools
        .iter()
        .map(|t| t.definition().name.clone())
        .collect();
    assert_eq!(names, vec!["read".to_string()]);
}

/// While a run is in flight, state changes are deferred rather than spliced
/// into the middle of the turn's entry chain — and they still land by the
/// time the run settles. Upstream: `agent-harness.ts:724-734`, `:551-558`.
#[tokio::test]
async fn state_changes_during_a_run_are_batched_then_flushed() {
    let dir = tempfile::tempdir().unwrap();
    let repo = JsonlSessionRepo::new(dir.path()).unwrap();
    let session = repo.create(Some("batch".into())).await.unwrap();
    let observer: Session = repo.open_or_create_id("batch").await.unwrap_or(session);

    let harness = AgentHarness::new(AgentHarnessOptions::new(
        observer.clone(),
        model(),
        replier("done"),
    ))
    .await;

    harness.prompt_text("go").await.unwrap();
    harness.set_thinking_level(ThinkingLevel::High).await;
    harness.wait_for_idle().await;

    let kinds: Vec<&'static str> = observer
        .entries()
        .await
        .iter()
        .map(|e| e.kind.type_tag())
        .collect();
    assert!(
        kinds.contains(&"thinking_level_change"),
        "the deferred change must land once the run settles, got {kinds:?}"
    );
    let level_idx = kinds
        .iter()
        .position(|k| *k == "thinking_level_change")
        .unwrap();
    let last_message_idx = kinds.iter().rposition(|k| *k == "message").unwrap();
    assert!(
        level_idx > last_message_idx,
        "a mid-run change must be flushed after the turn's messages, not \
         interleaved into them: {kinds:?}"
    );
}

// ---------------------------------------------------------------------------
// Seeding helper
// ---------------------------------------------------------------------------

/// `resume_agent_from_session` works against any `Session`, not just the
/// JSONL one, so in-memory tests and disk-backed workers share a path.
#[tokio::test]
async fn resume_works_against_any_session_backend() {
    use grain_agent_harness::{InMemorySessionRepo, resume_agent_from_session};

    let repo = InMemorySessionRepo::new();
    let session = repo.create(Some("mem".into())).await.unwrap();
    session.append_message(user("remembered")).await.unwrap();

    let resumed = resume_agent_from_session(&session, agent_options(replier("x")))
        .await
        .unwrap();
    assert_eq!(resumed.restored.message_count, 1);
    let texts: Vec<String> = resumed
        .agent
        .state()
        .await
        .messages
        .iter()
        .map(text_of)
        .collect();
    assert_eq!(texts, vec!["remembered".to_string()]);
}

/// Unknown thinking-level tags degrade to `Off` instead of failing a resume,
/// so a transcript from a newer build still loads.
#[test]
fn unknown_thinking_level_tag_degrades_to_off() {
    use grain_agent_harness::thinking_level_from_tag;
    assert_eq!(thinking_level_from_tag("xhigh"), ThinkingLevel::XHigh);
    assert_eq!(thinking_level_from_tag("max"), ThinkingLevel::Max);
    assert_eq!(thinking_level_from_tag("off"), ThinkingLevel::Off);
    assert_eq!(
        thinking_level_from_tag("from-the-future"),
        ThinkingLevel::Off
    );
}

/// A fork records which session it came from, so a forked worker's lineage is
/// recoverable from its metadata alone.
#[tokio::test]
async fn fork_records_its_parent_session() {
    use grain_agent_harness::{ForkPosition, SessionMetadata};

    let dir = tempfile::tempdir().unwrap();
    let repo = JsonlSessionRepo::new(dir.path()).unwrap();
    let source = repo.create(Some("origin".into())).await.unwrap();
    source.append_message(user("a")).await.unwrap();
    let b = source.append_message(user("b")).await.unwrap();
    let meta = source.metadata().await;
    drop(source);

    let forked = repo
        .fork(&meta, Some(&b), ForkPosition::At, Some("child".into()))
        .await
        .unwrap();
    assert_eq!(forked.metadata().await.parent_session(), Some("origin"));
    drop(forked);

    // And it survives a restart, since it rides the persisted metadata.
    let repo = JsonlSessionRepo::new(dir.path()).unwrap();
    let reopened = repo.open(&SessionMetadata::with_id("child")).await.unwrap();
    assert_eq!(reopened.metadata().await.parent_session(), Some("origin"));

    // A session that was never forked has no parent.
    assert_eq!(
        repo.open(&SessionMetadata::with_id("origin"))
            .await
            .unwrap()
            .metadata()
            .await
            .parent_session(),
        None
    );
}
