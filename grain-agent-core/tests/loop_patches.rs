//! Fresh WP4 loop-fidelity vectors.
//!
//! These tests cover WP4 ledger entries that have **no** covering vector in
//! the two ported upstream files (see `tests/PORTING.md`, "Patches with no
//! covering upstream test"): the asserted behavior is derived directly from
//! the upstream TypeScript sources at pi @ 34239180, cited per test.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use common::*;
use grain_agent_core::{
    Agent, AgentContext, AgentEvent, AgentLoopConfig, AgentOptions, AgentToolError,
    AgentToolResult, StopReason, ThinkingBudgets, UserContent, run_agent_loop,
};
use serde_json::json;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// patch-1: default tool-argument schema validation + coercion
// (upstream: prepareToolCall → validateToolArguments,
//  agent-loop.ts:616-663 + pi-ai utils/validation.ts:278-310)
// ---------------------------------------------------------------------------

/// Malformed arguments must become an `isError` tool result without the tool
/// executing, and the loop continues so the model can retry
/// (agent-loop.ts:657-663 catch → immediate error outcome).
#[tokio::test]
async fn malformed_tool_args_fail_validation_without_executing() {
    let executed: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let context = AgentContext {
        system_prompt: "".into(),
        messages: vec![],
        tools: vec![echo_tool(executed.clone())],
    };
    let config = AgentLoopConfig::new(create_model(), identity_converter());

    let stream = FnStream::new(|n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            // `value` must be a string; an object is not coercible.
            done_stream(create_assistant_message(
                vec![tool_call(
                    "tool-1",
                    "echo",
                    json!({ "value": { "nested": true } }),
                )],
                StopReason::ToolUse,
            ))
        } else {
            done_stream(create_assistant_message(
                vec![text("done")],
                StopReason::Stop,
            ))
        }
    });

    let (sink, events) = recording_sink();
    let messages = run_agent_loop(
        vec![create_user_message("echo something")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream.clone(),
    )
    .await
    .expect("loop failed");

    // The tool must never execute with invalid arguments.
    assert!(executed.lock().unwrap().is_empty());

    let events = events.lock().unwrap();
    let tool_end = events
        .iter()
        .find(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }))
        .expect("expected tool_execution_end");
    if let AgentEvent::ToolExecutionEnd {
        is_error, result, ..
    } = tool_end
    {
        assert!(*is_error);
        let text = content_text(&result.content).unwrap_or_default();
        assert!(
            text.starts_with("Validation failed for tool \"echo\":"),
            "expected upstream validation message shape, got: {text}"
        );
        assert!(
            text.contains("Received arguments:"),
            "expected received-arguments dump, got: {text}"
        );
    }

    // The error result is fed back and the loop continues.
    assert_eq!(stream.calls(), 2);
    assert_eq!(
        messages.last().map(|m| m.role().to_string()).as_deref(),
        Some("assistant")
    );
}

/// Coerced arguments (not the raw wire arguments) are what `beforeToolCall`
/// and `execute` observe (agent-loop.ts:619-655 `validatedArgs`; coercion per
/// pi-ai utils/validation.ts:58-230).
#[tokio::test]
async fn coerced_tool_args_are_passed_to_hooks_and_execute() {
    let executed_args: Arc<StdMutex<Vec<serde_json::Value>>> = Arc::new(StdMutex::new(Vec::new()));
    let executed_capture = executed_args.clone();
    let tool = TestTool::new(
        "convert",
        "Convert",
        "Coercion tool",
        json!({
            "type": "object",
            "properties": {
                "count": { "type": "number" },
                "flag": { "type": "boolean" },
                "note": { "type": "string" }
            },
            "required": ["count", "flag", "note"]
        }),
        Arc::new(move |_id, args, _cancel, _on_update| {
            let executed = executed_capture.clone();
            Box::pin(async move {
                executed.lock().unwrap().push(args);
                Ok(AgentToolResult::text("ok"))
            })
        }),
    )
    .arc();

    let context = AgentContext {
        system_prompt: "".into(),
        messages: vec![],
        tools: vec![tool],
    };

    let hook_args: Arc<StdMutex<Option<serde_json::Value>>> = Arc::new(StdMutex::new(None));
    let hook_capture = hook_args.clone();
    let mut config = AgentLoopConfig::new(create_model(), identity_converter());
    config.before_tool_call = Some(Arc::new(move |ctx, _cancel| {
        let hook_args = hook_capture.clone();
        Box::pin(async move {
            *hook_args.lock().unwrap() = Some(ctx.args.clone());
            Ok(None)
        })
    }));

    let stream = FnStream::new(|n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            done_stream(create_assistant_message(
                vec![tool_call(
                    "tool-1",
                    "convert",
                    json!({ "count": "42", "flag": "true", "note": 7 }),
                )],
                StopReason::ToolUse,
            ))
        } else {
            done_stream(create_assistant_message(
                vec![text("done")],
                StopReason::Stop,
            ))
        }
    });

    let (sink, _events) = recording_sink();
    run_agent_loop(
        vec![create_user_message("convert")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream,
    )
    .await
    .expect("loop failed");

    let expected = json!({ "count": 42, "flag": true, "note": "7" });
    assert_eq!(*executed_args.lock().unwrap(), vec![expected.clone()]);
    assert_eq!(hook_args.lock().unwrap().clone(), Some(expected));
}

// ---------------------------------------------------------------------------
// patch-2: fallible before/afterToolCall hooks
// (upstream: prepareToolCall try/catch agent-loop.ts:616-663;
//  finalizeExecutedToolCall try/catch agent-loop.ts:720-747)
// ---------------------------------------------------------------------------

/// A `beforeToolCall` error is contained as an `isError` tool result for
/// that call: the tool never executes and the loop continues.
#[tokio::test]
async fn before_tool_call_error_is_contained_as_error_result() {
    let executed: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let context = AgentContext {
        system_prompt: "".into(),
        messages: vec![],
        tools: vec![echo_tool(executed.clone())],
    };

    let mut config = AgentLoopConfig::new(create_model(), identity_converter());
    config.before_tool_call = Some(Arc::new(|_ctx, _cancel| {
        Box::pin(async move { Err(AgentToolError::msg("before hook exploded")) })
    }));

    let stream = FnStream::new(|n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            done_stream(create_assistant_message(
                vec![tool_call("tool-1", "echo", json!({ "value": "hello" }))],
                StopReason::ToolUse,
            ))
        } else {
            done_stream(create_assistant_message(
                vec![text("done")],
                StopReason::Stop,
            ))
        }
    });

    let (sink, events) = recording_sink();
    run_agent_loop(
        vec![create_user_message("echo something")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream.clone(),
    )
    .await
    .expect("loop must not abort on a hook error");

    // The tool never executed.
    assert!(executed.lock().unwrap().is_empty());

    let events = events.lock().unwrap();
    let tool_end = events
        .iter()
        .find(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }))
        .expect("expected tool_execution_end");
    if let AgentEvent::ToolExecutionEnd {
        is_error, result, ..
    } = tool_end
    {
        assert!(*is_error);
        assert_eq!(
            content_text(&result.content).as_deref(),
            Some("before hook exploded")
        );
    }

    // The loop continued past the contained failure.
    assert_eq!(stream.calls(), 2);
}

/// An `afterToolCall` error replaces the executed result with an error tool
/// result (isError=true) instead of panicking or aborting the run.
#[tokio::test]
async fn after_tool_call_error_replaces_result_with_error() {
    let executed: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let context = AgentContext {
        system_prompt: "".into(),
        messages: vec![],
        tools: vec![echo_tool(executed.clone())],
    };

    let mut config = AgentLoopConfig::new(create_model(), identity_converter());
    config.after_tool_call = Some(Arc::new(|_ctx, _cancel| {
        Box::pin(async move { Err(AgentToolError::msg("after hook exploded")) })
    }));

    let stream = FnStream::new(|n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            done_stream(create_assistant_message(
                vec![tool_call("tool-1", "echo", json!({ "value": "hello" }))],
                StopReason::ToolUse,
            ))
        } else {
            done_stream(create_assistant_message(
                vec![text("done")],
                StopReason::Stop,
            ))
        }
    });

    let (sink, events) = recording_sink();
    let messages = run_agent_loop(
        vec![create_user_message("echo something")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream.clone(),
    )
    .await
    .expect("loop must not abort on a hook error");

    // The tool DID execute; the hook error replaced its result.
    assert_eq!(*executed.lock().unwrap(), vec!["hello".to_string()]);

    let events = events.lock().unwrap();
    let tool_end = events
        .iter()
        .find(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }))
        .expect("expected tool_execution_end");
    if let AgentEvent::ToolExecutionEnd {
        is_error, result, ..
    } = tool_end
    {
        assert!(*is_error);
        // createErrorToolResult replaces content and details wholesale
        // (agent-loop.ts:744, 756-761).
        assert_eq!(
            content_text(&result.content).as_deref(),
            Some("after hook exploded")
        );
        assert_eq!(result.details, json!({}));
    }

    // The error result is persisted as the toolResult message.
    let tool_result = messages.iter().find_map(|m| match m {
        grain_agent_core::AgentMessage::Standard(grain_agent_core::Message::ToolResult(tr)) => {
            Some(tr.clone())
        }
        _ => None,
    });
    let tool_result = tool_result.expect("expected a toolResult message");
    assert!(tool_result.is_error);
    assert_eq!(
        content_text(&tool_result.content).as_deref(),
        Some("after hook exploded")
    );

    assert_eq!(stream.calls(), 2);
}

/// A tool-declared `prepare_arguments` runs before validation, and its output
/// is what gets validated/coerced (agent-loop.ts:586-598, 617).
#[tokio::test]
async fn prepared_arguments_are_validated_and_coerced() {
    let executed_args: Arc<StdMutex<Vec<serde_json::Value>>> = Arc::new(StdMutex::new(Vec::new()));
    let executed_capture = executed_args.clone();
    let tool = TestTool::new(
        "count",
        "Count",
        "Counting tool",
        json!({
            "type": "object",
            "properties": { "count": { "type": "integer" } },
            "required": ["count"]
        }),
        Arc::new(move |_id, args, _cancel, _on_update| {
            let executed = executed_capture.clone();
            Box::pin(async move {
                executed.lock().unwrap().push(args);
                Ok(AgentToolResult::text("ok"))
            })
        }),
    )
    .with_prepare(Arc::new(|args| {
        // Legacy alias: fold `n` into `count` (still a string here — the
        // subsequent validation pass coerces it).
        if let Some(n) = args.get("n").cloned() {
            Ok(json!({ "count": n }))
        } else {
            Ok(args)
        }
    }))
    .arc();

    let context = AgentContext {
        system_prompt: "".into(),
        messages: vec![],
        tools: vec![tool],
    };
    let config = AgentLoopConfig::new(create_model(), identity_converter());

    let stream = FnStream::new(|n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            done_stream(create_assistant_message(
                vec![tool_call("tool-1", "count", json!({ "n": "5" }))],
                StopReason::ToolUse,
            ))
        } else {
            done_stream(create_assistant_message(
                vec![text("done")],
                StopReason::Stop,
            ))
        }
    });

    let (sink, _events) = recording_sink();
    run_agent_loop(
        vec![create_user_message("count")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream,
    )
    .await
    .expect("loop failed");

    assert_eq!(*executed_args.lock().unwrap(), vec![json!({ "count": 5 })]);
}

// ---------------------------------------------------------------------------
// patch-8: AgentOptions → StreamOptions provider extras
// (upstream: AgentOptions.onPayload/.onResponse/.thinkingBudgets,
//  agent.ts:103-104,117 → createLoopConfig agent.ts:434-469 →
//  streamAssistantResponse spreads config into stream options,
//  agent-loop.ts:308-312; option shapes per pi-ai types.ts:92-98,143-152)
// ---------------------------------------------------------------------------

/// on_payload / on_response / thinking_budgets set on `AgentOptions` are
/// forwarded into the per-request `StreamOptions` the stream fn receives.
#[tokio::test]
async fn forwards_provider_extras_to_stream_options() {
    use grain_agent_core::{OnPayloadFn, OnResponseFn, ProviderResponse, StreamOptions};

    let captured: Arc<StdMutex<Option<StreamOptions>>> = Arc::new(StdMutex::new(None));
    let capture = captured.clone();
    let stream = FnStream::new(move |_n, _model, _ctx, opts, _cancel| {
        *capture.lock().unwrap() = Some(opts.clone());
        done_stream(create_assistant_message(vec![text("ok")], StopReason::Stop))
    });

    let payload_seen: Arc<StdMutex<Option<serde_json::Value>>> = Arc::new(StdMutex::new(None));
    let payload_capture = payload_seen.clone();
    let on_payload: OnPayloadFn = Arc::new(move |payload, _model| {
        let seen = payload_capture.clone();
        Box::pin(async move {
            *seen.lock().unwrap() = Some(payload);
            // Replace the payload (upstream: returning a value swaps it in).
            Some(json!({ "replaced": true }))
        })
    });

    let response_seen = Arc::new(AtomicBool::new(false));
    let response_capture = response_seen.clone();
    let on_response: OnResponseFn = Arc::new(move |response, _model| {
        let seen = response_capture.clone();
        Box::pin(async move {
            assert_eq!(response.status, 200);
            seen.store(true, Ordering::SeqCst);
        })
    });

    let budgets = ThinkingBudgets {
        minimal: Some(512),
        low: Some(1024),
        medium: Some(4096),
        high: Some(16384),
    };

    let mut options = AgentOptions::new(create_model(), stream);
    options.on_payload = Some(on_payload);
    options.on_response = Some(on_response);
    options.thinking_budgets = Some(budgets);

    let agent = Agent::new(options);
    agent.prompt_text("hello").await.expect("prompt failed");

    let opts = captured
        .lock()
        .unwrap()
        .clone()
        .expect("stream fn must have been called");

    // thinkingBudgets forwarded verbatim.
    assert_eq!(opts.thinking_budgets, Some(budgets));

    // The forwarded callbacks are the configured ones: invoking them runs
    // the caller-supplied logic (this is what a provider adapter does).
    let forwarded_on_payload = opts.on_payload.expect("on_payload must be forwarded");
    let replaced = forwarded_on_payload(json!({ "model": "mock" }), create_model()).await;
    assert_eq!(replaced, Some(json!({ "replaced": true })));
    assert_eq!(
        payload_seen.lock().unwrap().clone(),
        Some(json!({ "model": "mock" }))
    );

    let forwarded_on_response = opts.on_response.expect("on_response must be forwarded");
    forwarded_on_response(
        ProviderResponse {
            status: 200,
            headers: Default::default(),
        },
        create_model(),
    )
    .await;
    assert!(response_seen.load(Ordering::SeqCst));
}

// ---------------------------------------------------------------------------
// patch-9: type drift vs. upstream
// (upstream: StopReason "pending" pi-ai types.ts:391; Usage.cacheWrite1h /
//  Usage.reasoning types.ts:368-389; ThinkingLevel "max"
//  packages/agent/src/types.ts:294; AgentToolResult.usage/.addedToolNames
//  types.ts:354-369; toolResult spread-merge agent-loop.ts:773-787)
// ---------------------------------------------------------------------------

/// StopReason::Pending and ThinkingLevel::Max use the upstream wire names.
#[test]
fn stop_reason_pending_and_thinking_level_max_wire_names() {
    use grain_agent_core::ThinkingLevel;
    assert_eq!(
        serde_json::to_value(StopReason::Pending).unwrap(),
        json!("pending")
    );
    assert_eq!(
        serde_json::from_value::<StopReason>(json!("pending")).unwrap(),
        StopReason::Pending
    );
    assert_eq!(
        serde_json::to_value(ThinkingLevel::Max).unwrap(),
        json!("max")
    );
    assert_eq!(
        serde_json::from_value::<ThinkingLevel>(json!("max")).unwrap(),
        ThinkingLevel::Max
    );
}

/// Usage carries cacheWrite1h and reasoning with upstream wire names, and
/// omits them when absent (they are optional in pi-ai types.ts:373-380).
#[test]
fn usage_cache_write_1h_and_reasoning_wire_format() {
    use grain_agent_core::Usage;
    let usage = Usage {
        input: 10,
        output: 20,
        cache_read: 1,
        cache_write: 2,
        cache_write_1h: Some(2),
        reasoning: Some(0),
        total_tokens: 30,
        ..Default::default()
    };
    let wire = serde_json::to_value(&usage).unwrap();
    assert_eq!(wire["cacheWrite1h"], json!(2));
    // Some(0) is meaningful: providers with a reasoning breakdown report 0.
    assert_eq!(wire["reasoning"], json!(0));

    let round: Usage = serde_json::from_value(wire).unwrap();
    assert_eq!(round, usage);

    // Absent fields stay off the wire and deserialize to None.
    let bare = serde_json::to_value(Usage::default()).unwrap();
    assert!(bare.get("cacheWrite1h").is_none());
    assert!(bare.get("reasoning").is_none());
}

/// AgentToolResult.addedToolNames flows into the persisted toolResult
/// message only when non-empty (the upstream conditional spread,
/// agent-loop.ts:783), and tool-result usage flows through unconditionally.
#[tokio::test]
async fn added_tool_names_spread_into_tool_result_message_only_when_non_empty() {
    use grain_agent_core::{Message, Usage};

    // Tool 1 returns addedToolNames + usage; tool 2 returns an empty list.
    let tool = TestTool::new(
        "spawner",
        "Spawner",
        "Introduces tools",
        value_schema(),
        Arc::new(move |_id, args, _cancel, _on_update| {
            Box::pin(async move {
                let value = args
                    .get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let added = if value == "with" {
                    Some(vec!["new_tool".to_string()])
                } else {
                    Some(Vec::new())
                };
                Ok(AgentToolResult {
                    content: vec![UserContent::text("ok")],
                    details: json!({}),
                    usage: Some(Usage {
                        input: 7,
                        ..Default::default()
                    }),
                    added_tool_names: added,
                    ..Default::default()
                })
            })
        }),
    )
    .arc();

    let context = AgentContext {
        system_prompt: "".into(),
        messages: vec![],
        tools: vec![tool],
    };
    let config = AgentLoopConfig::new(create_model(), identity_converter());

    let stream = FnStream::new(|n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            done_stream(create_assistant_message(
                vec![
                    tool_call("tool-1", "spawner", json!({ "value": "with" })),
                    tool_call("tool-2", "spawner", json!({ "value": "without" })),
                ],
                StopReason::ToolUse,
            ))
        } else {
            done_stream(create_assistant_message(
                vec![text("done")],
                StopReason::Stop,
            ))
        }
    });

    let (sink, _events) = recording_sink();
    let messages = run_agent_loop(
        vec![create_user_message("go")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream,
    )
    .await
    .expect("loop failed");

    let tool_results: Vec<_> = messages
        .iter()
        .filter_map(|m| match m {
            grain_agent_core::AgentMessage::Standard(Message::ToolResult(tr)) => Some(tr.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_results.len(), 2);
    assert_eq!(
        tool_results[0].added_tool_names,
        Some(vec!["new_tool".to_string()])
    );
    // Empty list is dropped, matching `addedToolNames?.length ? ... : {}`.
    assert_eq!(tool_results[1].added_tool_names, None);
    for tr in &tool_results {
        assert_eq!(
            tr.usage,
            Some(Usage {
                input: 7,
                ..Default::default()
            })
        );
    }

    // Wire name check: the serialized toolResult message uses camelCase
    // addedToolNames and omits it when absent.
    let wire = serde_json::to_value(&tool_results[0]).unwrap();
    assert_eq!(wire["addedToolNames"], json!(["new_tool"]));
    let wire = serde_json::to_value(&tool_results[1]).unwrap();
    assert!(wire.get("addedToolNames").is_none());
}
