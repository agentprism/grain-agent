//! Port of upstream `packages/agent/test/agent.test.ts`
//! (pi pinned commit 34239180).
//!
//! Each test mirrors one upstream `it(...)` block. Tests that fail against
//! the current implementation are kept exact and marked
//! `#[ignore = "patch-N: ..."]` per the WP1/WP4 debt ledger.
//! See `tests/PORTING.md` for the full mapping table.

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use common::*;
use futures::StreamExt;
use grain_agent_core::{
    Agent, AgentError, AgentEvent, AgentMessage, AgentOptions, AgentToolResult,
    AssistantMessageEvent, EventListener, Message, Model, StopReason, ThinkingLevel,
    ToolUpdateCallback, UserContent, UserMessage,
};
use serde_json::json;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

fn options_with(stream: grain_agent_core::StreamFn) -> AgentOptions {
    AgentOptions::new(create_model(), stream)
}

/// Subscribe with a plain synchronous recorder, mirroring
/// `agent.subscribe((event) => { events.push(event.type) })`.
async fn subscribe_recorder(agent: &Agent) -> Arc<StdMutex<Vec<AgentEvent>>> {
    let events: Arc<StdMutex<Vec<AgentEvent>>> = Arc::new(StdMutex::new(Vec::new()));
    let capture = events.clone();
    let listener: EventListener = Arc::new(move |event: AgentEvent, _signal| {
        let events = capture.clone();
        Box::pin(async move {
            events.lock().unwrap().push(event);
        })
    });
    agent.subscribe(listener).await;
    events
}

/// Abort-responsive stream: emits `start`, then waits for cancellation and
/// terminates with an aborted assistant message — the port of the TS
/// `checkAbort` mock used by the abort / while-streaming tests.
fn abort_responsive_stream() -> Arc<FnStream> {
    FnStream::new(|_n, _model, _ctx, _opts, cancel| {
        let partial = create_assistant_message(vec![text("")], StopReason::Stop);
        let aborted = grain_agent_core::AssistantMessage {
            content: vec![text("Aborted")],
            stop_reason: StopReason::Aborted,
            error_message: None,
            ..create_assistant_message(vec![], StopReason::Aborted)
        };
        async_stream::stream! {
            yield AssistantMessageEvent::Start { partial };
            cancel.cancelled().await;
            yield AssistantMessageEvent::Error { error: "Aborted".into(), result: aborted };
        }
        .boxed()
    })
}

// ---------------------------------------------------------------------------
// describe("Agent")
// ---------------------------------------------------------------------------
//
// TS: "uses the configured default when a legacy caller omits streamFn"
// SKIPPED: exercises `setDefaultStreamFn` plus `Reflect.construct(Agent, [{}])`
// (constructing without a streamFn). The Rust `AgentOptions` requires an
// explicit `StreamFn` and there is no global default-stream registry, so the
// mechanism under test does not exist. See tests/PORTING.md.

/// TS: "should create an agent instance with default state"
///
/// Translation note: TS applies `DEFAULT_MODEL` ("unknown") when
/// `initialState.model` is omitted; the Rust constructor requires a model, so
/// the test passes `Model::unknown()` — the exported equivalent of the TS
/// default — and asserts the same default state surface.
#[tokio::test]
async fn creates_agent_with_default_state() {
    let agent = Agent::new(AgentOptions::new(Model::unknown(), unused_stream_fn()));
    let state = agent.state().await;

    assert_eq!(state.system_prompt, "");
    assert_eq!(state.model.id, "unknown");
    assert_eq!(state.thinking_level, ThinkingLevel::Off);
    assert!(state.tools.is_empty());
    assert!(state.messages.is_empty());
    assert!(!state.is_streaming);
    assert!(state.streaming_message.is_none());
    assert!(state.pending_tool_calls.is_empty());
    assert!(state.error_message.is_none());
}

/// TS: "should create an agent instance with custom initial state"
///
/// Translation note: TS resolves the model via `getModel("openai",
/// "gpt-4o-mini")`; the Rust crate has no model registry, so an equivalent
/// custom descriptor is constructed inline.
#[tokio::test]
async fn creates_agent_with_custom_initial_state() {
    let custom_model = Model {
        id: "gpt-4o-mini".into(),
        name: "GPT-4o mini".into(),
        api: "openai-responses".into(),
        provider: "openai".into(),
        ..Model::unknown()
    };
    let mut options = AgentOptions::new(custom_model.clone(), unused_stream_fn());
    options.system_prompt = "You are a helpful assistant.".into();
    options.thinking_level = ThinkingLevel::Low;

    let agent = Agent::new(options);
    let state = agent.state().await;

    assert_eq!(state.system_prompt, "You are a helpful assistant.");
    assert_eq!(state.model, custom_model);
    assert_eq!(state.thinking_level, ThinkingLevel::Low);
}

/// TS: "should subscribe to events"
#[tokio::test]
async fn subscribe_to_events() {
    let agent = Agent::new(options_with(unused_stream_fn()));

    let event_count = Arc::new(AtomicUsize::new(0));
    let count_capture = event_count.clone();
    let listener: EventListener = Arc::new(move |_event, _signal| {
        let count = count_capture.clone();
        Box::pin(async move {
            count.fetch_add(1, Ordering::SeqCst);
        })
    });
    let unsubscribe = agent.subscribe(listener).await;

    // No initial event on subscribe
    assert_eq!(event_count.load(Ordering::SeqCst), 0);

    // State mutators don't emit events
    agent.set_system_prompt("Test prompt".into()).await;
    assert_eq!(event_count.load(Ordering::SeqCst), 0);
    assert_eq!(agent.state().await.system_prompt, "Test prompt");

    // Unsubscribe should work
    unsubscribe.cancel().await;
    agent.set_system_prompt("Another prompt".into()).await;
    assert_eq!(event_count.load(Ordering::SeqCst), 0); // Should not increase
}

/// TS: "emits full lifecycle events for thrown run failures"
///
/// Translation note: the TS mock throws synchronously from `streamFn`; the
/// Rust `LlmStream` contract expresses the same failure as an `Err` from
/// `stream()`, which the loop must degrade into the identical event sequence.
#[tokio::test]
async fn emits_full_lifecycle_events_for_thrown_run_failures() {
    struct FailingStream;
    #[async_trait::async_trait]
    impl grain_agent_core::LlmStream for FailingStream {
        async fn stream(
            &self,
            _model: &Model,
            _context: &grain_agent_core::LlmContext,
            _options: &grain_agent_core::StreamOptions,
            _cancel: CancellationToken,
        ) -> Result<grain_agent_core::AssistantStream, grain_agent_core::StreamError> {
            Err(grain_agent_core::StreamError::msg("provider exploded"))
        }
    }

    let agent = Agent::new(options_with(Arc::new(FailingStream)));
    let events = subscribe_recorder(&agent).await;

    agent.prompt_text("hello").await.expect("prompt failed");

    assert_eq!(
        event_types(&events.lock().unwrap()),
        vec![
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
    );
    let state = agent.state().await;
    let last = state.messages.last().expect("expected messages");
    let assistant = last.as_assistant().expect("expected assistant message");
    assert_eq!(assistant.stop_reason, StopReason::Error);
    assert_eq!(
        assistant.error_message.as_deref(),
        Some("provider exploded")
    );
    assert_eq!(state.error_message.as_deref(), Some("provider exploded"));
}

/// TS: "should await async subscribers before prompt resolves"
#[tokio::test]
async fn awaits_async_subscribers_before_prompt_resolves() {
    let barrier = Arc::new(Notify::new());
    let stream = FnStream::new(|_n, _model, _ctx, _opts, _cancel| {
        done_stream(create_assistant_message(vec![text("ok")], StopReason::Stop))
    });
    let agent = Arc::new(Agent::new(options_with(stream)));

    let listener_finished = Arc::new(AtomicBool::new(false));
    let barrier_capture = barrier.clone();
    let finished_capture = listener_finished.clone();
    let listener: EventListener = Arc::new(move |event, _signal| {
        let barrier = barrier_capture.clone();
        let finished = finished_capture.clone();
        Box::pin(async move {
            if matches!(event, AgentEvent::AgentEnd { .. }) {
                barrier.notified().await;
                finished.store(true, Ordering::SeqCst);
            }
        })
    });
    agent.subscribe(listener).await;

    let prompt_agent = agent.clone();
    let prompt_handle = tokio::spawn(async move { prompt_agent.prompt_text("hello").await });

    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(!prompt_handle.is_finished());
    assert!(!listener_finished.load(Ordering::SeqCst));
    assert!(agent.state().await.is_streaming);

    barrier.notify_one();
    prompt_handle
        .await
        .expect("join failed")
        .expect("prompt failed");

    assert!(listener_finished.load(Ordering::SeqCst));
    assert!(!agent.state().await.is_streaming);
}

/// TS: "waitForIdle should wait for async subscribers"
///
/// Translation note: in TS, `agent.prompt(...)` synchronously registers the
/// active run before `waitForIdle()` is called on the next line. The Rust
/// prompt runs on a spawned task, so the port first waits for the run to be
/// observably streaming (the listener blocks the run on the barrier well
/// before `agent_end`), then asserts the same contract: `wait_for_idle`
/// resolves only after the async `message_end` subscriber settles and the
/// run fully finishes.
#[tokio::test]
async fn wait_for_idle_waits_for_async_subscribers() {
    let barrier = Arc::new(Notify::new());
    let stream = FnStream::new(|_n, _model, _ctx, _opts, _cancel| {
        done_stream(create_assistant_message(vec![text("ok")], StopReason::Stop))
    });
    let agent = Arc::new(Agent::new(options_with(stream)));

    let barrier_capture = barrier.clone();
    let listener: EventListener = Arc::new(move |event, _signal| {
        let barrier = barrier_capture.clone();
        Box::pin(async move {
            if let AgentEvent::MessageEnd { message } = &event
                && message.role() == "assistant"
            {
                barrier.notified().await;
            }
        })
    });
    agent.subscribe(listener).await;

    let prompt_agent = agent.clone();
    let prompt_handle = tokio::spawn(async move { prompt_agent.prompt_text("hello").await });

    // Let the run start; it is now blocked inside the message_end listener.
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(agent.state().await.is_streaming);

    let idle_resolved = Arc::new(AtomicBool::new(false));
    let idle_capture = idle_resolved.clone();
    let idle_agent = agent.clone();
    let idle_handle = tokio::spawn(async move {
        idle_agent.wait_for_idle().await;
        idle_capture.store(true, Ordering::SeqCst);
    });

    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(!idle_resolved.load(Ordering::SeqCst));
    assert!(agent.state().await.is_streaming);

    barrier.notify_one();
    prompt_handle
        .await
        .expect("join failed")
        .expect("prompt failed");
    idle_handle.await.expect("join failed");

    assert!(idle_resolved.load(Ordering::SeqCst));
    assert!(!agent.state().await.is_streaming);
}

/// TS: "should pass the active abort signal to subscribers"
#[tokio::test]
async fn passes_active_abort_signal_to_subscribers() {
    let agent = Arc::new(Agent::new(options_with(abort_responsive_stream())));

    let received_signal: Arc<StdMutex<Option<CancellationToken>>> = Arc::new(StdMutex::new(None));
    let signal_capture = received_signal.clone();
    let listener: EventListener = Arc::new(move |event, signal| {
        let received = signal_capture.clone();
        Box::pin(async move {
            if matches!(event, AgentEvent::AgentStart) {
                *received.lock().unwrap() = Some(signal);
            }
        })
    });
    agent.subscribe(listener).await;

    let prompt_agent = agent.clone();
    let prompt_handle = tokio::spawn(async move { prompt_agent.prompt_text("hello").await });
    tokio::time::sleep(Duration::from_millis(10)).await;

    let token = received_signal.lock().unwrap().clone();
    let token = token.expect("expected signal to be captured");
    assert!(!token.is_cancelled());

    agent.abort().await;
    prompt_handle
        .await
        .expect("join failed")
        .expect("prompt failed");

    assert!(token.is_cancelled());
}

/// TS: "should ignore tool updates after the tool execution settles"
///
/// Translation note: the upstream `unhandledRejection` bookkeeping is a
/// Node-specific mechanism with no Rust equivalent and is not ported; the
/// behavioural assertions (exactly one update delivered, none after the tool
/// settles) are kept exact.
#[tokio::test]
async fn ignores_tool_updates_after_execution_settles() {
    let delayed_update: Arc<StdMutex<Option<ToolUpdateCallback>>> = Arc::new(StdMutex::new(None));
    let update_capture = delayed_update.clone();
    let tool = TestTool::new(
        "delayed_tool",
        "Delayed Tool",
        "Captures progress callbacks",
        json!({ "type": "object", "properties": {} }),
        Arc::new(move |_id, _args, _cancel, on_update| {
            let capture = update_capture.clone();
            Box::pin(async move {
                *capture.lock().unwrap() = Some(on_update.clone());
                on_update(AgentToolResult {
                    content: vec![UserContent::text("running")],
                    details: json!({ "status": "running" }),
                    terminate: None,
                    ..Default::default()
                });
                Ok(AgentToolResult {
                    content: vec![UserContent::text("ok")],
                    details: json!({ "status": "done" }),
                    terminate: Some(true),
                    ..Default::default()
                })
            })
        }),
    )
    .arc();

    let stream = FnStream::new(|_n, _model, _ctx, _opts, _cancel| {
        done_stream(create_assistant_message(
            vec![tool_call("call-1", "delayed_tool", json!({}))],
            StopReason::ToolUse,
        ))
    });
    let mut options = options_with(stream);
    options.tools = vec![tool];
    let agent = Agent::new(options);
    let events = subscribe_recorder(&agent).await;

    agent.prompt_text("run tool").await.expect("prompt failed");
    let event_count_after_prompt = events.lock().unwrap().len();

    // Late update after the tool execution settled.
    let late = delayed_update.lock().unwrap().clone();
    if let Some(update) = late {
        update(AgentToolResult {
            content: vec![UserContent::text("late")],
            details: json!({ "status": "late" }),
            terminate: None,
            ..Default::default()
        });
    }
    tokio::time::sleep(Duration::from_millis(0)).await;
    tokio::task::yield_now().await;

    let events = events.lock().unwrap();
    let update_count = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionUpdate { .. }))
        .count();
    assert_eq!(update_count, 1);
    assert_eq!(events.len(), event_count_after_prompt);
}

/// TS: "should ignore a settled parallel tool update while another tool is
/// still running"
#[tokio::test]
async fn ignores_settled_parallel_tool_update_while_other_tool_running() {
    let slow_started = Arc::new(Notify::new());
    let settled_tool_ended = Arc::new(Notify::new());
    let release_slow = Arc::new(Notify::new());
    let settled_tool_update: Arc<StdMutex<Option<ToolUpdateCallback>>> =
        Arc::new(StdMutex::new(None));

    let update_capture = settled_tool_update.clone();
    let settled_tool = TestTool::new(
        "settled_tool",
        "Settled Tool",
        "Captures progress callbacks",
        json!({ "type": "object", "properties": {} }),
        Arc::new(move |_id, _args, _cancel, on_update| {
            let capture = update_capture.clone();
            Box::pin(async move {
                *capture.lock().unwrap() = Some(on_update);
                Ok(AgentToolResult {
                    content: vec![UserContent::text("done")],
                    details: json!({ "status": "done" }),
                    terminate: Some(true),
                    ..Default::default()
                })
            })
        }),
    )
    .arc();

    let slow_started_tool = slow_started.clone();
    let release_slow_tool = release_slow.clone();
    let slow_tool = TestTool::new(
        "slow_tool",
        "Slow Tool",
        "Keeps the agent run active",
        json!({ "type": "object", "properties": {} }),
        Arc::new(move |_id, _args, _cancel, _on_update| {
            let started = slow_started_tool.clone();
            let release = release_slow_tool.clone();
            Box::pin(async move {
                started.notify_one();
                release.notified().await;
                Ok(AgentToolResult {
                    content: vec![UserContent::text("done")],
                    details: json!({ "status": "done" }),
                    terminate: Some(true),
                    ..Default::default()
                })
            })
        }),
    )
    .arc();

    let stream = FnStream::new(|_n, _model, _ctx, _opts, _cancel| {
        done_stream(create_assistant_message(
            vec![
                tool_call("call-1", "settled_tool", json!({})),
                tool_call("call-2", "slow_tool", json!({})),
            ],
            StopReason::ToolUse,
        ))
    });
    let mut options = options_with(stream);
    options.tools = vec![settled_tool, slow_tool];
    let agent = Arc::new(Agent::new(options));

    let events: Arc<StdMutex<Vec<AgentEvent>>> = Arc::new(StdMutex::new(Vec::new()));
    let events_capture = events.clone();
    let settled_ended_capture = settled_tool_ended.clone();
    let listener: EventListener = Arc::new(move |event: AgentEvent, _signal| {
        let events = events_capture.clone();
        let settled_ended = settled_ended_capture.clone();
        Box::pin(async move {
            if let AgentEvent::ToolExecutionEnd { tool_call_id, .. } = &event
                && tool_call_id == "call-1"
            {
                settled_ended.notify_one();
            }
            events.lock().unwrap().push(event);
        })
    });
    agent.subscribe(listener).await;

    let prompt_agent = agent.clone();
    let prompt_handle = tokio::spawn(async move { prompt_agent.prompt_text("run tools").await });
    slow_started.notified().await;
    settled_tool_ended.notified().await;
    let event_count_before_late_update = events.lock().unwrap().len();

    let late = settled_tool_update.lock().unwrap().clone();
    if let Some(update) = late {
        update(AgentToolResult {
            content: vec![UserContent::text("late")],
            details: json!({ "status": "late" }),
            terminate: None,
            ..Default::default()
        });
    }
    tokio::time::sleep(Duration::from_millis(0)).await;
    tokio::task::yield_now().await;
    assert_eq!(events.lock().unwrap().len(), event_count_before_late_update);

    release_slow.notify_one();
    prompt_handle
        .await
        .expect("join failed")
        .expect("prompt failed");
    let update_count = events
        .lock()
        .unwrap()
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionUpdate { .. }))
        .count();
    assert_eq!(update_count, 0);
}

/// TS: "should update state with mutators"
///
/// PARTIAL PORT: the TS test also asserts reference-identity behaviour
/// (`state.tools`/`state.messages` are defensive copies of the assigned
/// arrays) and that pushing onto the live `state.messages` array appends to
/// agent state. The Rust `Agent` snapshots state by value (`state()` returns
/// an owned `AgentState`), so identity checks and live-array mutation are
/// TS-only mechanisms; the setter round-trips are asserted exactly.
#[tokio::test]
async fn updates_state_with_mutators() {
    let agent = Agent::new(options_with(unused_stream_fn()));

    // Test setSystemPrompt
    agent.set_system_prompt("Custom prompt".into()).await;
    assert_eq!(agent.state().await.system_prompt, "Custom prompt");

    // Test setModel
    let new_model = Model {
        id: "gemini-2.5-flash".into(),
        name: "Gemini 2.5 Flash".into(),
        api: "google-generative-ai".into(),
        provider: "google".into(),
        ..Model::unknown()
    };
    agent.set_model(new_model.clone()).await;
    assert_eq!(agent.state().await.model, new_model);

    // Test setThinkingLevel
    agent.set_thinking_level(ThinkingLevel::High).await;
    assert_eq!(agent.state().await.thinking_level, ThinkingLevel::High);

    // Test setTools
    let tool = TestTool::new(
        "test",
        "Test",
        "test tool",
        json!({ "type": "object", "properties": {} }),
        Arc::new(|_id, _args, _cancel, _on_update| {
            Box::pin(async move { Ok(AgentToolResult::text("ok")) })
        }),
    )
    .arc();
    agent.set_tools(vec![tool]).await;
    let state = agent.state().await;
    assert_eq!(state.tools.len(), 1);
    assert_eq!(state.tools[0].definition().name, "test");

    // Test replaceMessages
    let messages = vec![create_user_message("Hello")];
    agent.set_messages(messages.clone()).await;
    assert_eq!(agent.state().await.messages, messages);

    // Test clearMessages
    agent.set_messages(vec![]).await;
    assert!(agent.state().await.messages.is_empty());
}

/// TS: "should support steering message queue"
#[tokio::test]
async fn supports_steering_message_queue() {
    let agent = Agent::new(options_with(unused_stream_fn()));

    let message = create_user_message("Steering message");
    agent.steer(message.clone()).await;

    // The message is queued but not yet in state.messages
    assert!(!agent.state().await.messages.contains(&message));
}

/// TS: "should support follow-up message queue"
#[tokio::test]
async fn supports_follow_up_message_queue() {
    let agent = Agent::new(options_with(unused_stream_fn()));

    let message = create_user_message("Follow-up message");
    agent.follow_up(message.clone()).await;

    // The message is queued but not yet in state.messages
    assert!(!agent.state().await.messages.contains(&message));
}

/// TS: "should handle abort controller"
#[tokio::test]
async fn handles_abort_with_no_active_run() {
    let agent = Agent::new(options_with(unused_stream_fn()));

    // Should not panic even if nothing is running
    agent.abort().await;
}

/// TS: "should throw when prompt() called while streaming"
///
/// Translation note: the TS error text ("Agent is already processing a
/// prompt. Use steer() or followUp() ...") is API-surface copy; the Rust
/// equivalent is the typed `AgentError::AlreadyRunning`.
#[tokio::test]
async fn errors_when_prompt_called_while_streaming() {
    let agent = Arc::new(Agent::new(options_with(abort_responsive_stream())));

    // Start first prompt (don't await, it will block until abort)
    let prompt_agent = agent.clone();
    let first_prompt = tokio::spawn(async move { prompt_agent.prompt_text("First message").await });

    // Wait a tick for isStreaming to be set
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(agent.state().await.is_streaming);

    // Second prompt should reject
    let err = agent
        .prompt_text("Second message")
        .await
        .expect_err("expected second prompt to fail");
    assert!(matches!(err, AgentError::AlreadyRunning));

    // Cleanup - abort to stop the stream
    agent.abort().await;
    let _ = first_prompt.await.expect("join failed");
}

/// TS: "should throw when continue() called while streaming"
#[tokio::test]
async fn errors_when_continue_called_while_streaming() {
    let agent = Arc::new(Agent::new(options_with(abort_responsive_stream())));

    // Start first prompt
    let prompt_agent = agent.clone();
    let first_prompt = tokio::spawn(async move { prompt_agent.prompt_text("First message").await });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(agent.state().await.is_streaming);

    // continue() should reject
    let err = agent
        .continue_()
        .await
        .expect_err("expected continue to fail");
    assert!(matches!(err, AgentError::AlreadyRunning));

    // Cleanup
    agent.abort().await;
    let _ = first_prompt.await.expect("join failed");
}

/// TS: "continue() should process queued follow-up messages after an
/// assistant turn"
#[tokio::test]
async fn continue_processes_queued_follow_up_after_assistant_turn() {
    let stream = FnStream::new(|_n, _model, _ctx, _opts, _cancel| {
        done_stream(create_assistant_message(
            vec![text("Processed")],
            StopReason::Stop,
        ))
    });
    let agent = Agent::new(options_with(stream));

    agent
        .set_messages(vec![
            AgentMessage::user(UserMessage {
                content: vec![UserContent::text("Initial")],
                timestamp: now_ms() - 10,
            }),
            AgentMessage::assistant(create_assistant_message(
                vec![text("Initial response")],
                StopReason::Stop,
            )),
        ])
        .await;

    agent
        .follow_up(AgentMessage::user(UserMessage {
            content: vec![UserContent::text("Queued follow-up")],
            timestamp: now_ms(),
        }))
        .await;

    agent.continue_().await.expect("continue failed");

    let state = agent.state().await;
    let has_queued_follow_up = state.messages.iter().any(|m| match m {
        AgentMessage::Standard(Message::User(u)) => {
            user_text(u).as_deref() == Some("Queued follow-up")
        }
        _ => false,
    });
    assert!(has_queued_follow_up);
    assert_eq!(
        state
            .messages
            .last()
            .map(|m| m.role().to_string())
            .as_deref(),
        Some("assistant")
    );
}

/// TS: "continue() should keep one-at-a-time steering semantics from
/// assistant tail"
#[tokio::test]
async fn continue_keeps_one_at_a_time_steering_from_assistant_tail() {
    let stream = FnStream::new(|n, _model, _ctx, _opts, _cancel| {
        done_stream(create_assistant_message(
            vec![text(format!("Processed {}", n + 1))],
            StopReason::Stop,
        ))
    });
    let agent = Agent::new(options_with(stream.clone()));

    agent
        .set_messages(vec![
            AgentMessage::user(UserMessage {
                content: vec![UserContent::text("Initial")],
                timestamp: now_ms() - 10,
            }),
            AgentMessage::assistant(create_assistant_message(
                vec![text("Initial response")],
                StopReason::Stop,
            )),
        ])
        .await;

    agent
        .steer(AgentMessage::user(UserMessage {
            content: vec![UserContent::text("Steering 1")],
            timestamp: now_ms(),
        }))
        .await;
    agent
        .steer(AgentMessage::user(UserMessage {
            content: vec![UserContent::text("Steering 2")],
            timestamp: now_ms() + 1,
        }))
        .await;

    agent.continue_().await.expect("continue failed");

    let state = agent.state().await;
    let recent: Vec<String> = state.messages[state.messages.len().saturating_sub(4)..]
        .iter()
        .map(|m| m.role().to_string())
        .collect();
    assert_eq!(recent, vec!["user", "assistant", "user", "assistant"]);
    assert_eq!(stream.calls(), 2);
}

/// TS: "keeps legacy prepareNextTurn signal callback behavior"
///
/// Translation note: TS distinguishes the legacy `prepareNextTurn(signal)`
/// callback from `prepareNextTurnWithContext(context, signal)`; the Rust API
/// has a single `prepare_next_turn(context, cancel)` hook, so this asserts
/// the shared semantic — the hook runs between turns and receives the active
/// run's cancellation token.
#[tokio::test]
async fn prepare_next_turn_receives_run_cancellation_token() {
    let noop_tool = TestTool::new(
        "noop",
        "Noop",
        "Noop tool",
        json!({ "type": "object", "properties": {} }),
        Arc::new(|_id, _args, _cancel, _on_update| {
            Box::pin(async move {
                Ok(AgentToolResult {
                    content: vec![UserContent::text("ok")],
                    details: json!({}),
                    terminate: None,
                    ..Default::default()
                })
            })
        }),
    )
    .arc();

    let stream = FnStream::new(|n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            done_stream(create_assistant_message(
                vec![tool_call("tool-1", "noop", json!({}))],
                StopReason::ToolUse,
            ))
        } else {
            done_stream(create_assistant_message(
                vec![text("done")],
                StopReason::Stop,
            ))
        }
    });

    let saw_signal = Arc::new(AtomicBool::new(false));
    let saw_capture = saw_signal.clone();
    let mut options = options_with(stream.clone());
    options.tools = vec![noop_tool];
    options.prepare_next_turn = Some(Arc::new(move |_ctx, cancel| {
        let saw = saw_capture.clone();
        Box::pin(async move {
            // The hook receives the run's live (non-cancelled) token.
            saw.store(!cancel.is_cancelled(), Ordering::SeqCst);
            None
        })
    }));

    let agent = Agent::new(options);
    agent.prompt_text("start").await.expect("prompt failed");

    assert_eq!(stream.calls(), 2);
    assert!(saw_signal.load(Ordering::SeqCst));
}

/// TS: "forwards sessionId to streamFunction options"
///
/// Full port including the mid-life setter half (agent.test.ts:725-730):
/// `agent.sessionId = "session-def"` re-targets subsequent prompts and the
/// next stream call receives the new id (patch-11 added the Rust setter).
#[tokio::test]
async fn forwards_session_id_to_stream_options() {
    let received_session_id: Arc<StdMutex<Option<String>>> = Arc::new(StdMutex::new(None));
    let capture = received_session_id.clone();
    let stream = FnStream::new(move |_n, _model, _ctx, opts, _cancel| {
        *capture.lock().unwrap() = opts.session_id.clone();
        done_stream(create_assistant_message(vec![text("ok")], StopReason::Stop))
    });

    let mut options = options_with(stream);
    options.session_id = Some("session-abc".into());
    let agent = Agent::new(options);

    agent.prompt_text("hello").await.expect("prompt failed");
    assert_eq!(
        received_session_id.lock().unwrap().as_deref(),
        Some("session-abc")
    );

    // Test setter (agent.test.ts:725-730).
    agent.set_session_id(Some("session-def".into())).await;
    assert_eq!(agent.session_id().await.as_deref(), Some("session-def"));

    agent
        .prompt_text("hello again")
        .await
        .expect("prompt failed");
    assert_eq!(
        received_session_id.lock().unwrap().as_deref(),
        Some("session-def")
    );
}
