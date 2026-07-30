//! G9 — harness wiring: dynamic system-prompt re-render, provider-hook
//! passthrough, and the abort contract.

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::StreamExt;
use grain_agent_core::{
    AgentMessage, AgentTool, AgentToolError, AgentToolResult, AssistantContent, AssistantMessage,
    AssistantMessageEvent, AssistantStream, Cost, LlmContext, LlmStream, Model, StopReason,
    StreamError, StreamFn, StreamOptions, TextContent, ThinkingLevel, ToolDefinition,
    ToolUpdateCallback, Usage, UserContent, UserMessage,
};
use grain_agent_harness::{
    AgentHarness, AgentHarnessEvent, AgentHarnessOptions, InMemorySessionRepo, SessionRepo,
    SystemPrompt,
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

/// Records the system prompt of every request it serves.
struct PromptSpy {
    seen: Arc<StdMutex<Vec<String>>>,
    turns: AtomicUsize,
    /// Number of turns to keep the loop going by emitting a tool call.
    tool_turns: usize,
}

#[async_trait]
impl LlmStream for PromptSpy {
    async fn stream(
        &self,
        model: &Model,
        context: &LlmContext,
        _options: &StreamOptions,
        _cancel: CancellationToken,
    ) -> Result<AssistantStream, StreamError> {
        self.seen
            .lock()
            .unwrap()
            .push(context.system_prompt.clone());
        let n = self.turns.fetch_add(1, Ordering::SeqCst);
        let (content, stop) = if n < self.tool_turns {
            (
                vec![AssistantContent::ToolCall(grain_agent_core::ToolCall {
                    id: format!("call_{n}"),
                    name: "noop".into(),
                    arguments: serde_json::json!({}),
                    thought_signature: None,
                })],
                StopReason::ToolUse,
            )
        } else {
            (
                vec![AssistantContent::Text(TextContent {
                    text: "done".into(),
                })],
                StopReason::Stop,
            )
        };
        let msg = AssistantMessage {
            content,
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            usage: Usage::default(),
            stop_reason: stop,
            raw_stop_reason: None,
            error_message: None,
            error_code: None,
            timestamp: 0,
        };
        Ok(futures::stream::iter(vec![AssistantMessageEvent::Done { result: msg }]).boxed())
    }
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

async fn session() -> grain_agent_harness::Session {
    InMemorySessionRepo::new().create(None).await.unwrap()
}

fn user(text: &str) -> AgentMessage {
    AgentMessage::user(UserMessage {
        content: vec![UserContent::text(text)],
        timestamp: 0,
    })
}

// ---------------------------------------------------------------------------
// Dynamic system prompt
// ---------------------------------------------------------------------------

/// A dynamic prompt used to collapse to the empty string, silently discarding
/// whatever the caller wrote. It must render at construction.
#[tokio::test]
async fn dynamic_system_prompt_renders_at_construction() {
    let seen = Arc::new(StdMutex::new(Vec::new()));
    let spy = Arc::new(PromptSpy {
        seen: seen.clone(),
        turns: AtomicUsize::new(0),
        tool_turns: 0,
    });

    let mut opts = AgentHarnessOptions::new(session().await, model(), spy as StreamFn);
    opts.system_prompt = SystemPrompt::Dynamic(Arc::new(|ctx| {
        let id = ctx.model.id.clone();
        Box::pin(async move { format!("prompt for {id}") })
    }));
    let harness = AgentHarness::new(opts).await;
    harness.prompt_text("hi").await.unwrap();
    harness.wait_for_idle().await;

    assert_eq!(seen.lock().unwrap().as_slice(), ["prompt for mock"]);
}

/// Reconfiguration re-renders: a prompt that names the model must stop naming
/// the old one after `set_model`.
#[tokio::test]
async fn dynamic_system_prompt_rerenders_on_model_change() {
    let seen = Arc::new(StdMutex::new(Vec::new()));
    let spy = Arc::new(PromptSpy {
        seen: seen.clone(),
        turns: AtomicUsize::new(0),
        tool_turns: 0,
    });

    let mut opts = AgentHarnessOptions::new(session().await, model(), spy as StreamFn);
    opts.system_prompt = SystemPrompt::Dynamic(Arc::new(|ctx| {
        let id = ctx.model.id.clone();
        let level = ctx.thinking_level;
        Box::pin(async move { format!("model={id} thinking={level:?}") })
    }));
    let harness = AgentHarness::new(opts).await;

    harness.prompt_text("one").await.unwrap();
    harness.wait_for_idle().await;

    harness
        .set_model(Model {
            id: "claude-sonnet-4".into(),
            provider: "anthropic".into(),
            ..model()
        })
        .await;
    harness.set_thinking_level(ThinkingLevel::High).await;

    harness.prompt_text("two").await.unwrap();
    harness.wait_for_idle().await;

    let prompts = seen.lock().unwrap().clone();
    assert_eq!(prompts.len(), 2);
    assert_eq!(prompts[0], "model=mock thinking=Off");
    assert_eq!(prompts[1], "model=claude-sonnet-4 thinking=High");
}

/// The active tool set is part of the render context, so restricting tools
/// re-renders the prompt.
#[tokio::test]
async fn dynamic_system_prompt_sees_active_tools() {
    let seen = Arc::new(StdMutex::new(Vec::new()));
    let spy = Arc::new(PromptSpy {
        seen: seen.clone(),
        turns: AtomicUsize::new(0),
        tool_turns: 0,
    });

    let mut opts = AgentHarnessOptions::new(session().await, model(), spy as StreamFn);
    opts.tools = vec![tool("noop"), tool("other")];
    opts.system_prompt = SystemPrompt::Dynamic(Arc::new(|ctx| {
        let mut names: Vec<String> = ctx
            .active_tools
            .iter()
            .map(|t| t.definition().name.clone())
            .collect();
        names.sort();
        Box::pin(async move { format!("tools: {}", names.join(",")) })
    }));
    let harness = AgentHarness::new(opts).await;

    harness.prompt_text("one").await.unwrap();
    harness.wait_for_idle().await;
    harness
        .set_active_tools(&["noop".to_string()])
        .await
        .unwrap();
    harness.prompt_text("two").await.unwrap();
    harness.wait_for_idle().await;

    let prompts = seen.lock().unwrap().clone();
    assert_eq!(prompts[0], "tools: noop,other");
    assert_eq!(prompts[1], "tools: noop");
}

/// Within a single multi-turn run the prompt is re-rendered between turns,
/// which is what upstream's per-turn `createTurnState` guarantees.
#[tokio::test]
async fn dynamic_system_prompt_rerenders_between_turns_of_one_run() {
    let seen = Arc::new(StdMutex::new(Vec::new()));
    let spy = Arc::new(PromptSpy {
        seen: seen.clone(),
        turns: AtomicUsize::new(0),
        // First turn calls a tool, so the loop runs a second turn.
        tool_turns: 1,
    });

    let renders = Arc::new(AtomicUsize::new(0));
    let render_count = renders.clone();
    let mut opts = AgentHarnessOptions::new(session().await, model(), spy as StreamFn);
    opts.tools = vec![tool("noop")];
    opts.system_prompt = SystemPrompt::Dynamic(Arc::new(move |_ctx| {
        let n = render_count.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { format!("render #{n}") })
    }));
    let harness = AgentHarness::new(opts).await;

    harness.prompt_text("go").await.unwrap();
    harness.wait_for_idle().await;

    let prompts = seen.lock().unwrap().clone();
    assert_eq!(prompts.len(), 2, "expected two turns, got {prompts:?}");
    assert_ne!(
        prompts[0], prompts[1],
        "the prompt must be re-rendered between turns, got {prompts:?}"
    );
}

/// A static prompt is never re-rendered and reconfiguration must not clobber
/// a caller's explicit `set_system_prompt`.
#[tokio::test]
async fn static_system_prompt_is_untouched_by_reconfiguration() {
    let seen = Arc::new(StdMutex::new(Vec::new()));
    let spy = Arc::new(PromptSpy {
        seen: seen.clone(),
        turns: AtomicUsize::new(0),
        tool_turns: 0,
    });

    let mut opts = AgentHarnessOptions::new(session().await, model(), spy as StreamFn);
    opts.system_prompt = SystemPrompt::Static("fixed".into());
    let harness = AgentHarness::new(opts).await;

    harness.set_system_prompt("explicitly set".into()).await;
    harness.set_thinking_level(ThinkingLevel::High).await;
    harness.prompt_text("go").await.unwrap();
    harness.wait_for_idle().await;

    assert_eq!(seen.lock().unwrap().as_slice(), ["explicitly set"]);
}

// ---------------------------------------------------------------------------
// Provider-hook passthrough
// ---------------------------------------------------------------------------

/// `on_payload` / `on_response` / `thinking_budgets` set on the harness reach
/// the per-request `StreamOptions`, so a harness-only embedder no longer has
/// to drop down to the bare `Agent`.
#[tokio::test]
async fn harness_forwards_provider_hooks_to_stream_options() {
    use grain_agent_core::{OnPayloadFn, OnResponseFn, ThinkingBudgets};

    struct OptionSpy(Arc<StdMutex<Option<StreamOptions>>>);

    #[async_trait]
    impl LlmStream for OptionSpy {
        async fn stream(
            &self,
            model: &Model,
            _context: &LlmContext,
            options: &StreamOptions,
            _cancel: CancellationToken,
        ) -> Result<AssistantStream, StreamError> {
            *self.0.lock().unwrap() = Some(options.clone());
            let msg = AssistantMessage {
                content: vec![AssistantContent::Text(TextContent { text: "ok".into() })],
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
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

    let captured = Arc::new(StdMutex::new(None));
    let spy = Arc::new(OptionSpy(captured.clone()));

    let on_payload: OnPayloadFn = Arc::new(|_payload, _model| Box::pin(async { None }));
    let on_response: OnResponseFn = Arc::new(|_response, _model| Box::pin(async {}));
    let budgets = ThinkingBudgets {
        minimal: Some(512),
        low: Some(1024),
        medium: Some(4096),
        high: Some(16384),
    };

    let mut opts = AgentHarnessOptions::new(session().await, model(), spy as StreamFn);
    opts.on_payload = Some(on_payload);
    opts.on_response = Some(on_response);
    opts.thinking_budgets = Some(budgets);
    let harness = AgentHarness::new(opts).await;
    harness.prompt_text("hi").await.unwrap();
    harness.wait_for_idle().await;

    let o = captured
        .lock()
        .unwrap()
        .clone()
        .expect("stream fn not called");
    assert!(o.on_payload.is_some(), "on_payload must reach the seam");
    assert!(o.on_response.is_some(), "on_response must reach the seam");
    assert_eq!(o.thinking_budgets, Some(budgets));
}

// ---------------------------------------------------------------------------
// Abort
// ---------------------------------------------------------------------------

/// `abort()` fires the event it always claimed to fire, and hands back the
/// queued messages it discarded instead of dropping them silently.
#[tokio::test]
async fn abort_emits_and_returns_cleared_queues() {
    let seen = Arc::new(StdMutex::new(Vec::new()));
    let spy = Arc::new(PromptSpy {
        seen: seen.clone(),
        turns: AtomicUsize::new(0),
        tool_turns: 0,
    });
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        session().await,
        model(),
        spy as StreamFn,
    ))
    .await;

    let events: Arc<StdMutex<Vec<&'static str>>> = Arc::new(StdMutex::new(Vec::new()));
    let sink = events.clone();
    harness
        .subscribe(Arc::new(move |event, _signal| {
            let sink = sink.clone();
            Box::pin(async move {
                if matches!(event, AgentHarnessEvent::Abort { .. }) {
                    sink.lock().unwrap().push("abort");
                }
            })
        }))
        .await;

    harness.steer(user("steer me")).await;
    harness.follow_up(user("later")).await;

    let result = harness.abort().await;

    assert_eq!(result.cleared_steer.len(), 1);
    assert_eq!(result.cleared_follow_up.len(), 1);
    assert_eq!(
        events.lock().unwrap().as_slice(),
        ["abort"],
        "abort must emit exactly one Abort event"
    );
    assert!(
        !harness.agent().has_queued_messages().await,
        "both queues must be empty after abort"
    );
}

/// Aborting an idle harness is still a stable signal, with empty payloads.
#[tokio::test]
async fn abort_while_idle_still_emits() {
    let seen = Arc::new(StdMutex::new(Vec::new()));
    let spy = Arc::new(PromptSpy {
        seen: seen.clone(),
        turns: AtomicUsize::new(0),
        tool_turns: 0,
    });
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        session().await,
        model(),
        spy as StreamFn,
    ))
    .await;

    let fired = Arc::new(AtomicUsize::new(0));
    let counter = fired.clone();
    harness
        .subscribe(Arc::new(move |event, _signal| {
            let counter = counter.clone();
            Box::pin(async move {
                if matches!(event, AgentHarnessEvent::Abort { .. }) {
                    counter.fetch_add(1, Ordering::SeqCst);
                }
            })
        }))
        .await;

    let result = harness.abort().await;
    assert!(result.cleared_steer.is_empty());
    assert!(result.cleared_follow_up.is_empty());
    assert_eq!(fired.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------
// The next_turn bucket
// ---------------------------------------------------------------------------

/// `next_turn` is callable while idle — that is the whole point of the third
/// bucket, and it is what `steer` / `follow_up` cannot do.
#[tokio::test]
async fn next_turn_is_callable_while_idle_and_leads_the_next_prompt() {
    let seen = Arc::new(StdMutex::new(Vec::new()));
    let spy = Arc::new(PromptSpy {
        seen: seen.clone(),
        turns: AtomicUsize::new(0),
        tool_turns: 0,
    });
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        session().await,
        model(),
        spy as StreamFn,
    ))
    .await;

    // Nothing is running.
    harness.next_turn(user("queued first")).await;
    harness.next_turn(user("queued second")).await;
    harness.prompt_text("the actual prompt").await.unwrap();
    harness.wait_for_idle().await;

    let texts: Vec<String> = harness
        .agent()
        .state()
        .await
        .messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Standard(grain_agent_core::Message::User(u)) => match &u.content[0] {
                UserContent::Text(t) => Some(t.text.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();

    assert_eq!(
        texts,
        vec![
            "queued first".to_string(),
            "queued second".to_string(),
            "the actual prompt".to_string(),
        ],
        "queued messages must lead the prompt, in queue order"
    );
}

/// The bucket drains fully and does not leak into a later prompt.
#[tokio::test]
async fn next_turn_queue_drains_once() {
    let seen = Arc::new(StdMutex::new(Vec::new()));
    let spy = Arc::new(PromptSpy {
        seen: seen.clone(),
        turns: AtomicUsize::new(0),
        tool_turns: 0,
    });
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        session().await,
        model(),
        spy as StreamFn,
    ))
    .await;

    harness.next_turn(user("only once")).await;
    harness.prompt_text("first").await.unwrap();
    harness.wait_for_idle().await;
    harness.prompt_text("second").await.unwrap();
    harness.wait_for_idle().await;

    let count = harness
        .agent()
        .state()
        .await
        .messages
        .iter()
        .filter(|m| match m {
            AgentMessage::Standard(grain_agent_core::Message::User(u)) => {
                matches!(&u.content[0], UserContent::Text(t) if t.text == "only once")
            }
            _ => false,
        })
        .count();
    assert_eq!(count, 1, "the queued message must appear exactly once");
}

/// `abort` clears steer and follow-up but leaves the next-turn bucket alone,
/// matching upstream's `AbortEvent` payload (which reports only the first two).
#[tokio::test]
async fn next_turn_survives_abort() {
    let seen = Arc::new(StdMutex::new(Vec::new()));
    let spy = Arc::new(PromptSpy {
        seen: seen.clone(),
        turns: AtomicUsize::new(0),
        tool_turns: 0,
    });
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        session().await,
        model(),
        spy as StreamFn,
    ))
    .await;

    harness.next_turn(user("survives")).await;
    let result = harness.abort().await;
    assert!(result.cleared_steer.is_empty());
    assert!(result.cleared_follow_up.is_empty());

    harness.prompt_text("after abort").await.unwrap();
    harness.wait_for_idle().await;

    let texts: Vec<String> = harness
        .agent()
        .state()
        .await
        .messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Standard(grain_agent_core::Message::User(u)) => match &u.content[0] {
                UserContent::Text(t) => Some(t.text.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(
        texts,
        vec!["survives".to_string(), "after abort".to_string()]
    );
}

/// `QueueUpdate` reports per-bucket depth, so a status line can distinguish
/// "two steers pending" from "one thing queued for next time".
#[tokio::test]
async fn queue_update_reports_per_bucket_depth() {
    let seen = Arc::new(StdMutex::new(Vec::new()));
    let spy = Arc::new(PromptSpy {
        seen: seen.clone(),
        turns: AtomicUsize::new(0),
        tool_turns: 0,
    });
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        session().await,
        model(),
        spy as StreamFn,
    ))
    .await;

    let last: Arc<StdMutex<Option<(usize, usize, usize)>>> = Arc::new(StdMutex::new(None));
    let sink = last.clone();
    harness
        .subscribe(Arc::new(move |event, _signal| {
            let sink = sink.clone();
            Box::pin(async move {
                if let AgentHarnessEvent::QueueUpdate {
                    steer_count,
                    follow_up_count,
                    next_turn_count,
                    ..
                } = event
                {
                    *sink.lock().unwrap() = Some((steer_count, follow_up_count, next_turn_count));
                }
            })
        }))
        .await;

    harness.next_turn(user("a")).await;
    assert_eq!(*last.lock().unwrap(), Some((0, 0, 1)));
    harness.next_turn(user("b")).await;
    assert_eq!(*last.lock().unwrap(), Some((0, 0, 2)));
    harness.steer(user("s")).await;
    assert_eq!(*last.lock().unwrap(), Some((1, 0, 2)));
    harness.follow_up(user("f")).await;
    assert_eq!(*last.lock().unwrap(), Some((1, 1, 2)));
}
