//! Port of upstream `packages/agent/test/agent-loop.test.ts`
//! (pi pinned commit 34239180).
//!
//! Each test mirrors one upstream `it(...)` block and asserts the same
//! semantics: event sequences, orderings, and edge cases. Tests that fail
//! against the current loop are kept exact and marked
//! `#[ignore = "patch-N: ..."]` per the WP1/WP4 debt ledger.
//! See `tests/PORTING.md` for the full mapping table.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use common::*;
use grain_agent_core::{
    AgentContext, AgentEvent, AgentLoopConfig, AgentLoopTurnUpdate, AgentMessage, AgentToolResult,
    Message, StopReason, ToolExecutionMode, UserContent, UserMessage, run_agent_loop,
    run_agent_loop_continue,
};
use serde_json::json;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

/// Spawn a task that releases `notify` after `ms` milliseconds — the port of
/// the upstream `setTimeout(() => releaseFirst?.(), 20)` pattern.
fn release_after(notify: Arc<Notify>, ms: u64) {
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(ms)).await;
        notify.notify_one();
    });
}

// ---------------------------------------------------------------------------
// describe("default stream function compatibility")
// ---------------------------------------------------------------------------
//
// TS: "uses the configured default when a legacy caller omits streamFn"
// SKIPPED: exercises `setDefaultStreamFn` + a legacy call signature that omits
// the streamFn argument. The Rust API has no global default-stream registry
// and `run_agent_loop` takes `StreamFn` as a required parameter, so the
// mechanism under test does not exist. See tests/PORTING.md.

// ---------------------------------------------------------------------------
// describe("agentLoop with AgentMessage")
// ---------------------------------------------------------------------------

/// TS: "should emit events with AgentMessage types"
#[tokio::test]
async fn emits_events_with_agent_message_types() {
    let context = AgentContext {
        system_prompt: "You are helpful.".into(),
        messages: vec![],
        tools: vec![],
    };
    let config = AgentLoopConfig::new(create_model(), identity_converter());
    let stream = FnStream::new(|_n, _model, _ctx, _opts, _cancel| {
        done_stream(create_assistant_message(
            vec![text("Hi there!")],
            StopReason::Stop,
        ))
    });

    let (sink, events) = recording_sink();
    let messages = run_agent_loop(
        vec![create_user_message("Hello")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream,
    )
    .await
    .expect("loop failed");

    // Should have user message and assistant message
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role(), "user");
    assert_eq!(messages[1].role(), "assistant");

    // Verify event sequence
    let types = event_types(&events.lock().unwrap());
    for expected in [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ] {
        assert!(types.contains(&expected), "missing event {expected}");
    }
}

/// TS: "should handle custom message types via convertToLlm"
#[tokio::test]
async fn handles_custom_message_types_via_convert_to_llm() {
    let notification = AgentMessage::Custom(json!({
        "role": "notification",
        "text": "This is a notification",
        "timestamp": now_ms(),
    }));
    let context = AgentContext {
        system_prompt: "You are helpful.".into(),
        messages: vec![notification],
        tools: vec![],
    };

    let converted: Arc<StdMutex<Vec<Message>>> = Arc::new(StdMutex::new(Vec::new()));
    let converted_capture = converted.clone();
    let mut config = AgentLoopConfig::new(create_model(), identity_converter());
    config.convert_to_llm = Arc::new(move |messages: Vec<AgentMessage>| {
        let converted = converted_capture.clone();
        Box::pin(async move {
            // Filter out notifications, convert rest
            let result: Vec<Message> = messages
                .into_iter()
                .filter(|m| m.role() != "notification")
                .filter_map(|m| match m {
                    AgentMessage::Standard(m) => Some(m),
                    AgentMessage::Custom(_) => None,
                })
                .collect();
            *converted.lock().unwrap() = result.clone();
            result
        })
    });

    let stream = FnStream::new(|_n, _model, _ctx, _opts, _cancel| {
        done_stream(create_assistant_message(
            vec![text("Response")],
            StopReason::Stop,
        ))
    });

    let (sink, _events) = recording_sink();
    run_agent_loop(
        vec![create_user_message("Hello")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream,
    )
    .await
    .expect("loop failed");

    // The notification should have been filtered out in convertToLlm
    let converted = converted.lock().unwrap();
    assert_eq!(converted.len(), 1); // Only user message
    assert_eq!(converted[0].role(), "user");
}

/// TS: "should apply transformContext before convertToLlm"
#[tokio::test]
async fn applies_transform_context_before_convert_to_llm() {
    let context = AgentContext {
        system_prompt: "You are helpful.".into(),
        messages: vec![
            create_user_message("old message 1"),
            AgentMessage::assistant(create_assistant_message(
                vec![text("old response 1")],
                StopReason::Stop,
            )),
            create_user_message("old message 2"),
            AgentMessage::assistant(create_assistant_message(
                vec![text("old response 2")],
                StopReason::Stop,
            )),
        ],
        tools: vec![],
    };

    let transformed: Arc<StdMutex<Vec<AgentMessage>>> = Arc::new(StdMutex::new(Vec::new()));
    let converted: Arc<StdMutex<Vec<Message>>> = Arc::new(StdMutex::new(Vec::new()));

    let mut config = AgentLoopConfig::new(create_model(), identity_converter());
    let transformed_capture = transformed.clone();
    config.transform_context = Some(Arc::new(move |messages: Vec<AgentMessage>, _cancel| {
        let transformed = transformed_capture.clone();
        Box::pin(async move {
            // Keep only last 2 messages (prune old ones)
            let kept: Vec<AgentMessage> = messages[messages.len().saturating_sub(2)..].to_vec();
            *transformed.lock().unwrap() = kept.clone();
            kept
        })
    }));
    let converted_capture = converted.clone();
    config.convert_to_llm = Arc::new(move |messages: Vec<AgentMessage>| {
        let converted = converted_capture.clone();
        Box::pin(async move {
            let result: Vec<Message> = messages
                .into_iter()
                .filter_map(|m| match m {
                    AgentMessage::Standard(m) => Some(m),
                    AgentMessage::Custom(_) => None,
                })
                .collect();
            *converted.lock().unwrap() = result.clone();
            result
        })
    });

    let stream = FnStream::new(|_n, _model, _ctx, _opts, _cancel| {
        done_stream(create_assistant_message(
            vec![text("Response")],
            StopReason::Stop,
        ))
    });

    let (sink, _events) = recording_sink();
    run_agent_loop(
        vec![create_user_message("new message")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream,
    )
    .await
    .expect("loop failed");

    // transformContext should have been called first, keeping only last 2
    assert_eq!(transformed.lock().unwrap().len(), 2);
    // Then convertToLlm receives the pruned messages
    assert_eq!(converted.lock().unwrap().len(), 2);
}

/// TS: "should handle tool calls and results"
///
/// Full port including the upstream usage assertions (agent-loop.test.ts:
/// 277-292, 320-323, 365-368): `tool.execute` returns a `usage` reading,
/// `afterToolCall` observes `result.usage` equal to the tool's reading and
/// replaces it via `{ usage: patched }`, and the persisted toolResult
/// message carries the patched usage (patch-9 restored the plumbing).
#[tokio::test]
async fn handles_tool_calls_and_results() {
    let tool_usage = grain_agent_core::Usage {
        input: 1,
        output: 2,
        cache_read: 3,
        cache_write: 4,
        total_tokens: 10,
        cost: grain_agent_core::Cost {
            input: 0.1,
            output: 0.2,
            cache_read: 0.3,
            cache_write: 0.4,
            total: 1.0,
        },
        ..Default::default()
    };
    let patched_tool_usage = grain_agent_core::Usage {
        input: 5,
        output: 6,
        cache_read: 7,
        cache_write: 8,
        total_tokens: 26,
        cost: grain_agent_core::Cost {
            input: 0.5,
            output: 0.6,
            cache_read: 0.7,
            cache_write: 0.8,
            total: 2.6,
        },
        ..Default::default()
    };

    let executed: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    // Upstream echo tool extended with the usage reading returned by
    // execute (agent-loop.test.ts:294-307).
    let executed_capture = executed.clone();
    let usage_for_tool = tool_usage.clone();
    let tool = TestTool::new(
        "echo",
        "Echo",
        "Echo tool",
        value_schema(),
        Arc::new(move |_id, args, _cancel, _on_update| {
            let executed = executed_capture.clone();
            let usage = usage_for_tool.clone();
            Box::pin(async move {
                let value = args
                    .get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                executed.lock().unwrap().push(value.clone());
                Ok(AgentToolResult {
                    content: vec![UserContent::text(format!("echoed: {value}"))],
                    details: json!({ "value": value }),
                    usage: Some(usage),
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

    let observed_result_text: Arc<StdMutex<Option<String>>> = Arc::new(StdMutex::new(None));
    let observed_tool_usage: Arc<StdMutex<Option<grain_agent_core::Usage>>> =
        Arc::new(StdMutex::new(None));
    let mut config = AgentLoopConfig::new(create_model(), identity_converter());
    let observed_capture = observed_result_text.clone();
    let observed_usage_capture = observed_tool_usage.clone();
    let patched_for_hook = patched_tool_usage.clone();
    config.after_tool_call = Some(Arc::new(move |ctx, _cancel| {
        let observed = observed_capture.clone();
        let observed_usage = observed_usage_capture.clone();
        let patched = patched_for_hook.clone();
        Box::pin(async move {
            *observed.lock().unwrap() = content_text(&ctx.result.content);
            // TS: observedToolUsage = result.usage; return { usage: patched }
            // (agent-loop.test.ts:320-323).
            *observed_usage.lock().unwrap() = ctx.result.usage.clone();
            Ok(Some(grain_agent_core::AfterToolCallResult {
                usage: Some(patched),
                ..Default::default()
            }))
        })
    }));

    let stream = FnStream::new(|n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            // First call: return tool call
            done_stream(create_assistant_message(
                vec![tool_call("tool-1", "echo", json!({ "value": "hello" }))],
                StopReason::ToolUse,
            ))
        } else {
            // Second call: return final response
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
        stream,
    )
    .await
    .expect("loop failed");

    // Tool should have been executed
    assert_eq!(*executed.lock().unwrap(), vec!["hello".to_string()]);

    // Should have tool execution events
    let events = events.lock().unwrap();
    let tool_start = events
        .iter()
        .find(|e| matches!(e, AgentEvent::ToolExecutionStart { .. }));
    let tool_end = events
        .iter()
        .find(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }));
    assert!(tool_start.is_some());
    assert!(tool_end.is_some());
    if let Some(AgentEvent::ToolExecutionEnd { is_error, .. }) = tool_end {
        assert!(!is_error);
    }
    // afterToolCall observed the result produced by execute.
    assert_eq!(
        observed_result_text.lock().unwrap().as_deref(),
        Some("echoed: hello")
    );
    // TS: expect(observedToolUsage).toEqual(toolUsage) — the hook saw the
    // execute-returned usage (agent-loop.test.ts:365).
    assert_eq!(observed_tool_usage.lock().unwrap().clone(), Some(tool_usage));
    // TS: the persisted toolResult message carries the patched usage
    // (agent-loop.test.ts:366-368).
    let tool_result = messages.iter().find_map(|m| match m {
        AgentMessage::Standard(Message::ToolResult(tr)) => Some(tr.clone()),
        _ => None,
    });
    let tool_result = tool_result.expect("expected a toolResult message");
    assert_eq!(tool_result.usage, Some(patched_tool_usage));
}

/// TS: "should not execute tool calls from a length-truncated assistant message"
#[tokio::test]
async fn does_not_execute_tool_calls_from_length_truncated_message() {
    let executed: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let context = AgentContext {
        system_prompt: "".into(),
        messages: vec![],
        tools: vec![echo_tool(executed.clone())],
    };
    let config = AgentLoopConfig::new(create_model(), identity_converter());

    let stream = FnStream::new(|n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            // Output hit the token limit mid tool call. The salvage parser can
            // produce arguments that validate but are silently truncated, so
            // nothing in this message may execute.
            done_stream(create_assistant_message(
                vec![tool_call("tool-1", "echo", json!({ "value": "hel" }))],
                StopReason::Length,
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

    // The tool must never execute with potentially truncated arguments.
    assert!(executed.lock().unwrap().is_empty());

    let events = events.lock().unwrap();
    let tool_end = events
        .iter()
        .find(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }));
    assert!(tool_end.is_some());
    if let Some(AgentEvent::ToolExecutionEnd {
        is_error, result, ..
    }) = tool_end
    {
        assert!(*is_error);
        let text = content_text(&result.content).unwrap_or_default();
        assert!(
            text.contains("output token limit"),
            "expected truncation notice, got: {text}"
        );
    }

    // The loop continues so the model can re-issue the tool call.
    assert_eq!(stream.calls(), 2);
    assert_eq!(
        messages.last().map(|m| m.role().to_string()).as_deref(),
        Some("assistant")
    );
}

/// TS: "should execute mutated beforeToolCall args without revalidation"
///
/// Upstream mutates the shared `args` object inside `beforeToolCall`, and the
/// tool then executes with the mutated value (no revalidation). The Rust hook
/// receives a cloned `args` value, so the rewrite goes through the explicit
/// `BeforeToolCallResult::args` override channel (patch-3). The tool schema
/// requires `value` to be a string; the hook rewrites it to the number `123`
/// and `execute` observes `123` — proof that no second validation pass runs.
#[tokio::test]
async fn executes_mutated_before_tool_call_args_without_revalidation() {
    let executed: Arc<StdMutex<Vec<serde_json::Value>>> = Arc::new(StdMutex::new(Vec::new()));
    let executed_capture = executed.clone();
    let tool = TestTool::new(
        "echo",
        "Echo",
        "Echo tool",
        value_schema(),
        Arc::new(move |_id, args, _cancel, _on_update| {
            let executed = executed_capture.clone();
            Box::pin(async move {
                let value = args
                    .get("value")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                executed.lock().unwrap().push(value.clone());
                Ok(AgentToolResult {
                    content: vec![UserContent::text(format!("echoed: {value}"))],
                    details: json!({ "value": value }),
                    terminate: None,
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

    let mut config = AgentLoopConfig::new(create_model(), identity_converter());
    config.before_tool_call = Some(Arc::new(|ctx, _cancel| {
        Box::pin(async move {
            // Upstream: `args.value = 123` on the shared object
            // (agent-loop.test.ts:471-475). The Rust equivalent returns the
            // rewritten args through the override channel.
            let mut args = ctx.args.clone();
            args["value"] = json!(123);
            Ok(Some(grain_agent_core::BeforeToolCallResult {
                args: Some(args),
                ..Default::default()
            }))
        })
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

    let (sink, _events) = recording_sink();
    run_agent_loop(
        vec![create_user_message("echo something")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream,
    )
    .await
    .expect("loop failed");

    assert_eq!(*executed.lock().unwrap(), vec![json!(123)]);
}

/// TS: "should prepare tool arguments for validation"
#[tokio::test]
async fn prepares_tool_arguments_for_validation() {
    let executed: Arc<StdMutex<Vec<serde_json::Value>>> = Arc::new(StdMutex::new(Vec::new()));
    let executed_capture = executed.clone();
    let tool = TestTool::new(
        "edit",
        "Edit",
        "Edit tool",
        json!({
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": { "type": "string" },
                            "newText": { "type": "string" }
                        },
                        "required": ["oldText", "newText"]
                    }
                }
            },
            "required": ["edits"]
        }),
        Arc::new(move |_id, args, _cancel, _on_update| {
            let executed = executed_capture.clone();
            Box::pin(async move {
                let edits = args.get("edits").cloned().unwrap_or(json!([]));
                let count = edits.as_array().map(|a| a.len()).unwrap_or(0);
                executed.lock().unwrap().push(edits);
                Ok(AgentToolResult {
                    content: vec![UserContent::text(format!("edited {count}"))],
                    details: json!({ "count": count }),
                    terminate: None,
                    ..Default::default()
                })
            })
        }),
    )
    .with_prepare(Arc::new(|args| {
        // Port of the upstream `prepareArguments`: fold a flat
        // { oldText, newText } shape into { edits: [...] }.
        if !args.is_object() {
            return Ok(args);
        }
        let old_text = args.get("oldText").and_then(|v| v.as_str());
        let new_text = args.get("newText").and_then(|v| v.as_str());
        let (Some(old_text), Some(new_text)) = (old_text, new_text) else {
            return Ok(args);
        };
        let mut edits = args
            .get("edits")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        edits.push(json!({ "oldText": old_text, "newText": new_text }));
        Ok(json!({ "edits": edits }))
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
                vec![tool_call(
                    "tool-1",
                    "edit",
                    json!({ "oldText": "before", "newText": "after" }),
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
        vec![create_user_message("edit something")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream,
    )
    .await
    .expect("loop failed");

    assert_eq!(
        *executed.lock().unwrap(),
        vec![json!([{ "oldText": "before", "newText": "after" }])]
    );
}

/// TS: "should emit tool_execution_end in completion order but persist tool
/// results in source order"
#[tokio::test]
async fn emits_tool_execution_end_in_completion_order_persists_results_in_source_order() {
    let first_resolved = Arc::new(AtomicBool::new(false));
    let parallel_observed = Arc::new(AtomicBool::new(false));
    let first_done = Arc::new(Notify::new());

    let first_resolved_tool = first_resolved.clone();
    let parallel_observed_tool = parallel_observed.clone();
    let first_done_tool = first_done.clone();
    let tool = TestTool::new(
        "echo",
        "Echo",
        "Echo tool",
        value_schema(),
        Arc::new(move |_id, args, _cancel, _on_update| {
            let first_resolved = first_resolved_tool.clone();
            let parallel_observed = parallel_observed_tool.clone();
            let first_done = first_done_tool.clone();
            Box::pin(async move {
                let value = args
                    .get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                if value == "first" {
                    first_done.notified().await;
                    first_resolved.store(true, Ordering::SeqCst);
                }
                if value == "second" && !first_resolved.load(Ordering::SeqCst) {
                    parallel_observed.store(true, Ordering::SeqCst);
                }
                Ok(AgentToolResult {
                    content: vec![UserContent::text(format!("echoed: {value}"))],
                    details: json!({ "value": value }),
                    terminate: None,
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
    let mut config = AgentLoopConfig::new(create_model(), identity_converter());
    config.tool_execution = ToolExecutionMode::Parallel;

    let release = first_done.clone();
    let stream = FnStream::new(move |n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            release_after(release.clone(), 20);
            done_stream(create_assistant_message(
                vec![
                    tool_call("tool-1", "echo", json!({ "value": "first" })),
                    tool_call("tool-2", "echo", json!({ "value": "second" })),
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

    let (sink, events) = recording_sink();
    run_agent_loop(
        vec![create_user_message("echo both")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream,
    )
    .await
    .expect("loop failed");

    let events = events.lock().unwrap();
    let turn_tool_result_ids: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::TurnEnd { tool_results, .. } => Some(
                tool_results
                    .iter()
                    .map(|tr| tr.tool_call_id.clone())
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .flatten()
        .collect();

    assert!(parallel_observed.load(Ordering::SeqCst));
    assert_eq!(tool_execution_end_ids(&events), vec!["tool-2", "tool-1"]);
    assert_eq!(
        message_end_tool_result_ids(&events),
        vec!["tool-1", "tool-2"]
    );
    assert_eq!(turn_tool_result_ids, vec!["tool-1", "tool-2"]);
}

/// TS: "should inject queued messages after all tool calls complete"
#[tokio::test]
async fn injects_queued_messages_after_all_tool_calls_complete() {
    let executed: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let context = AgentContext {
        system_prompt: "".into(),
        messages: vec![],
        tools: vec![echo_tool(executed.clone())],
    };

    let queued_delivered = Arc::new(AtomicBool::new(false));
    let saw_interrupt_in_context = Arc::new(AtomicBool::new(false));

    let mut config = AgentLoopConfig::new(create_model(), identity_converter());
    config.tool_execution = ToolExecutionMode::Sequential;
    let executed_steering = executed.clone();
    let queued_delivered_steering = queued_delivered.clone();
    config.get_steering_messages = Some(Arc::new(move || {
        let executed = executed_steering.clone();
        let queued_delivered = queued_delivered_steering.clone();
        Box::pin(async move {
            // Return steering message after tool execution has started.
            // Upstream: `executed.length >= 1 && !queuedDelivered`.
            if !executed.lock().unwrap().is_empty()
                && !queued_delivered.swap(true, Ordering::SeqCst)
            {
                vec![create_user_message("interrupt")]
            } else {
                Vec::new()
            }
        })
    }));

    let saw_interrupt_capture = saw_interrupt_in_context.clone();
    let stream = FnStream::new(move |n, _model, ctx, _opts, _cancel| {
        // Check if interrupt message is in context on second call
        if n == 1 {
            let saw = ctx.messages.iter().any(|m| match m {
                Message::User(u) => user_text(u).as_deref() == Some("interrupt"),
                _ => false,
            });
            saw_interrupt_capture.store(saw, Ordering::SeqCst);
        }
        if n == 0 {
            // First call: return two tool calls
            done_stream(create_assistant_message(
                vec![
                    tool_call("tool-1", "echo", json!({ "value": "first" })),
                    tool_call("tool-2", "echo", json!({ "value": "second" })),
                ],
                StopReason::ToolUse,
            ))
        } else {
            // Second call: return final response
            done_stream(create_assistant_message(
                vec![text("done")],
                StopReason::Stop,
            ))
        }
    });

    let (sink, events) = recording_sink();
    run_agent_loop(
        vec![create_user_message("start")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream,
    )
    .await
    .expect("loop failed");

    // Both tools should execute before steering is injected
    assert_eq!(
        *executed.lock().unwrap(),
        vec!["first".to_string(), "second".to_string()]
    );

    let events = events.lock().unwrap();
    let tool_ends: Vec<bool> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolExecutionEnd { is_error, .. } => Some(*is_error),
            _ => None,
        })
        .collect();
    assert_eq!(tool_ends.len(), 2);
    assert!(!tool_ends[0]);
    assert!(!tool_ends[1]);

    // Queued message should appear in events after both tool result messages
    let event_sequence: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::MessageStart { message } => match message {
                AgentMessage::Standard(Message::ToolResult(tr)) => {
                    Some(format!("tool:{}", tr.tool_call_id))
                }
                AgentMessage::Standard(Message::User(u)) => user_text(u),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let index_of = |needle: &str| {
        event_sequence
            .iter()
            .position(|s| s == needle)
            .unwrap_or_else(|| panic!("{needle} not found in {event_sequence:?}"))
    };
    assert!(event_sequence.contains(&"interrupt".to_string()));
    assert!(index_of("tool:tool-1") < index_of("interrupt"));
    assert!(index_of("tool:tool-2") < index_of("interrupt"));

    // Interrupt message should be in context when second LLM call is made
    assert!(saw_interrupt_in_context.load(Ordering::SeqCst));
}

/// TS: "should force sequential execution when a tool has
/// executionMode=sequential even with default parallel config"
#[tokio::test]
async fn forces_sequential_when_tool_has_sequential_execution_mode() {
    let first_resolved = Arc::new(AtomicBool::new(false));
    let parallel_observed = Arc::new(AtomicBool::new(false));
    let first_done = Arc::new(Notify::new());

    let first_resolved_tool = first_resolved.clone();
    let parallel_observed_tool = parallel_observed.clone();
    let first_done_tool = first_done.clone();
    let slow_tool = TestTool::new(
        "slow",
        "Slow",
        "Slow tool",
        value_schema(),
        Arc::new(move |_id, args, _cancel, _on_update| {
            let first_resolved = first_resolved_tool.clone();
            let parallel_observed = parallel_observed_tool.clone();
            let first_done = first_done_tool.clone();
            Box::pin(async move {
                let value = args
                    .get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                if value == "first" {
                    first_done.notified().await;
                    first_resolved.store(true, Ordering::SeqCst);
                }
                if value == "second" && !first_resolved.load(Ordering::SeqCst) {
                    parallel_observed.store(true, Ordering::SeqCst);
                }
                Ok(AgentToolResult {
                    content: vec![UserContent::text(format!("slow: {value}"))],
                    details: json!({ "value": value }),
                    terminate: None,
                    ..Default::default()
                })
            })
        }),
    )
    .with_execution_mode(ToolExecutionMode::Sequential)
    .arc();

    let context = AgentContext {
        system_prompt: "".into(),
        messages: vec![],
        tools: vec![slow_tool],
    };
    // config is parallel (default), but tool forces sequential
    let config = AgentLoopConfig::new(create_model(), identity_converter());

    let release = first_done.clone();
    let stream = FnStream::new(move |n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            release_after(release.clone(), 20);
            done_stream(create_assistant_message(
                vec![
                    tool_call("tool-1", "slow", json!({ "value": "first" })),
                    tool_call("tool-2", "slow", json!({ "value": "second" })),
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

    let (sink, events) = recording_sink();
    run_agent_loop(
        vec![create_user_message("run both")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream,
    )
    .await
    .expect("loop failed");

    // With sequential execution, second tool should NOT start before first finishes
    assert!(!parallel_observed.load(Ordering::SeqCst));

    let events = events.lock().unwrap();
    assert_eq!(
        message_end_tool_result_ids(&events),
        vec!["tool-1", "tool-2"]
    );
}

/// TS: "should force sequential execution when one of multiple tools has
/// executionMode=sequential"
#[tokio::test]
async fn forces_sequential_when_one_of_multiple_tools_is_sequential() {
    let execution_order: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let slow_done = Arc::new(Notify::new());

    let order_slow = execution_order.clone();
    let slow_done_tool = slow_done.clone();
    let slow_tool = TestTool::new(
        "slow",
        "Slow",
        "Slow tool",
        value_schema(),
        Arc::new(move |_id, args, _cancel, _on_update| {
            let order = order_slow.clone();
            let slow_done = slow_done_tool.clone();
            Box::pin(async move {
                let value = args
                    .get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                order.lock().unwrap().push(format!("slow:{value}"));
                if value == "a" {
                    slow_done.notified().await;
                }
                Ok(AgentToolResult {
                    content: vec![UserContent::text(format!("slow: {value}"))],
                    details: json!({ "value": value }),
                    terminate: None,
                    ..Default::default()
                })
            })
        }),
    )
    .with_execution_mode(ToolExecutionMode::Sequential)
    .arc();

    let order_fast = execution_order.clone();
    // no executionMode = defaults to parallel
    let fast_tool = TestTool::new(
        "fast",
        "Fast",
        "Fast tool",
        value_schema(),
        Arc::new(move |_id, args, _cancel, _on_update| {
            let order = order_fast.clone();
            Box::pin(async move {
                let value = args
                    .get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                order.lock().unwrap().push(format!("fast:{value}"));
                Ok(AgentToolResult {
                    content: vec![UserContent::text(format!("fast: {value}"))],
                    details: json!({ "value": value }),
                    terminate: None,
                    ..Default::default()
                })
            })
        }),
    )
    .arc();

    let context = AgentContext {
        system_prompt: "".into(),
        messages: vec![],
        tools: vec![slow_tool, fast_tool],
    };
    // parallel by default, but slowTool forces sequential
    let config = AgentLoopConfig::new(create_model(), identity_converter());

    let release = slow_done.clone();
    let stream = FnStream::new(move |n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            release_after(release.clone(), 20);
            done_stream(create_assistant_message(
                vec![
                    tool_call("tool-1", "slow", json!({ "value": "a" })),
                    tool_call("tool-2", "fast", json!({ "value": "b" })),
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
    run_agent_loop(
        vec![create_user_message("run both")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream,
    )
    .await
    .expect("loop failed");

    // Fast tool should NOT run before slow tool finishes
    let order = execution_order.lock().unwrap();
    assert_eq!(order.first().map(String::as_str), Some("slow:a"));
    assert!(order.contains(&"fast:b".to_string()));
}

/// TS: "should allow parallel execution when all tools have
/// executionMode=parallel"
#[tokio::test]
async fn allows_parallel_when_all_tools_parallel() {
    let first_resolved = Arc::new(AtomicBool::new(false));
    let parallel_observed = Arc::new(AtomicBool::new(false));
    let first_done = Arc::new(Notify::new());

    let first_resolved_tool = first_resolved.clone();
    let parallel_observed_tool = parallel_observed.clone();
    let first_done_tool = first_done.clone();
    let tool = TestTool::new(
        "echo",
        "Echo",
        "Echo tool",
        value_schema(),
        Arc::new(move |_id, args, _cancel, _on_update| {
            let first_resolved = first_resolved_tool.clone();
            let parallel_observed = parallel_observed_tool.clone();
            let first_done = first_done_tool.clone();
            Box::pin(async move {
                let value = args
                    .get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                if value == "first" {
                    first_done.notified().await;
                    first_resolved.store(true, Ordering::SeqCst);
                }
                if value == "second" && !first_resolved.load(Ordering::SeqCst) {
                    parallel_observed.store(true, Ordering::SeqCst);
                }
                Ok(AgentToolResult {
                    content: vec![UserContent::text(format!("echoed: {value}"))],
                    details: json!({ "value": value }),
                    terminate: None,
                    ..Default::default()
                })
            })
        }),
    )
    .with_execution_mode(ToolExecutionMode::Parallel)
    .arc();

    let context = AgentContext {
        system_prompt: "".into(),
        messages: vec![],
        tools: vec![tool],
    };
    let config = AgentLoopConfig::new(create_model(), identity_converter());

    let release = first_done.clone();
    let stream = FnStream::new(move |n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            release_after(release.clone(), 20);
            done_stream(create_assistant_message(
                vec![
                    tool_call("tool-1", "echo", json!({ "value": "first" })),
                    tool_call("tool-2", "echo", json!({ "value": "second" })),
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
    run_agent_loop(
        vec![create_user_message("echo both")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream,
    )
    .await
    .expect("loop failed");

    // With executionMode=parallel, second tool should start before first finishes
    assert!(parallel_observed.load(Ordering::SeqCst));
}

/// TS: "should use prepareNextTurn snapshot before continuing"
#[tokio::test]
async fn uses_prepare_next_turn_snapshot_before_continuing() {
    let executed: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let context = AgentContext {
        system_prompt: "first prompt".into(),
        messages: vec![],
        tools: vec![echo_tool(executed)],
    };

    let prepared = Arc::new(AtomicBool::new(false));
    let mut config = AgentLoopConfig::new(create_model(), identity_converter());
    let prepared_hook = prepared.clone();
    config.prepare_next_turn = Some(Arc::new(move |ctx, _cancel| {
        let prepared = prepared_hook.clone();
        Box::pin(async move {
            if prepared.swap(true, Ordering::SeqCst) {
                return None;
            }
            Some(AgentLoopTurnUpdate {
                context: Some(AgentContext {
                    system_prompt: "second prompt".into(),
                    messages: ctx.context.messages.clone(),
                    tools: ctx.context.tools.clone(),
                }),
                ..Default::default()
            })
        })
    }));

    let second_turn_system_prompt: Arc<StdMutex<String>> = Arc::new(StdMutex::new(String::new()));
    let prompt_capture = second_turn_system_prompt.clone();
    let stream = FnStream::new(move |n, _model, ctx, _opts, _cancel| {
        if n == 1 {
            *prompt_capture.lock().unwrap() = ctx.system_prompt.clone();
        }
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

    let (sink, _events) = recording_sink();
    run_agent_loop(
        vec![create_user_message("echo something")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream.clone(),
    )
    .await
    .expect("loop failed");

    assert_eq!(stream.calls(), 2);
    assert_eq!(*second_turn_system_prompt.lock().unwrap(), "second prompt");
}

/// TS: "should stop after the current turn when shouldStopAfterTurn returns true"
#[tokio::test]
async fn stops_after_turn_when_should_stop_after_turn_returns_true() {
    let executed: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let context = AgentContext {
        system_prompt: "".into(),
        messages: vec![],
        tools: vec![echo_tool(executed.clone())],
    };

    let steering_polls = Arc::new(StdMutex::new(0usize));
    let follow_up_polls = Arc::new(StdMutex::new(0usize));
    let callback_tool_result_ids: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let callback_context_roles: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));

    let mut config = AgentLoopConfig::new(create_model(), identity_converter());
    let steering_capture = steering_polls.clone();
    config.get_steering_messages = Some(Arc::new(move || {
        let polls = steering_capture.clone();
        Box::pin(async move {
            *polls.lock().unwrap() += 1;
            Vec::new()
        })
    }));
    let follow_up_capture = follow_up_polls.clone();
    config.get_follow_up_messages = Some(Arc::new(move || {
        let polls = follow_up_capture.clone();
        Box::pin(async move {
            *polls.lock().unwrap() += 1;
            vec![create_user_message("follow up should stay queued")]
        })
    }));
    let ids_capture = callback_tool_result_ids.clone();
    let roles_capture = callback_context_roles.clone();
    config.should_stop_after_turn = Some(Arc::new(move |ctx| {
        let ids = ids_capture.clone();
        let roles = roles_capture.clone();
        Box::pin(async move {
            // Upstream asserts `message.role === "assistant"`; in Rust the
            // hook's `message` field is typed `AssistantMessage`, so this
            // holds by construction.
            *ids.lock().unwrap() = ctx
                .tool_results
                .iter()
                .map(|tr| tr.tool_call_id.clone())
                .collect();
            *roles.lock().unwrap() = ctx
                .context
                .messages
                .iter()
                .map(|m| m.role().to_string())
                .collect();
            true
        })
    }));

    let stream = FnStream::new(|n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            done_stream(create_assistant_message(
                vec![tool_call("tool-1", "echo", json!({ "value": "hello" }))],
                StopReason::ToolUse,
            ))
        } else {
            done_stream(create_assistant_message(
                vec![text("should not run")],
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

    assert_eq!(stream.calls(), 1);
    assert_eq!(*executed.lock().unwrap(), vec!["hello".to_string()]);
    assert_eq!(*steering_polls.lock().unwrap(), 1);
    assert_eq!(*follow_up_polls.lock().unwrap(), 0);
    assert_eq!(*callback_tool_result_ids.lock().unwrap(), vec!["tool-1"]);
    assert_eq!(
        *callback_context_roles.lock().unwrap(),
        vec!["user", "assistant", "toolResult"]
    );
    assert_eq!(roles(&messages), vec!["user", "assistant", "toolResult"]);
    assert_eq!(
        event_types(&events.lock().unwrap()),
        vec![
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "tool_execution_start",
            "tool_execution_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
    );
}

/// TS: "should stop after a tool batch when every tool result sets terminate=true"
#[tokio::test]
async fn stops_after_tool_batch_when_all_results_terminate() {
    let tool = TestTool::new(
        "echo",
        "Echo",
        "Echo tool",
        value_schema(),
        Arc::new(|_id, args, _cancel, _on_update| {
            Box::pin(async move {
                let value = args
                    .get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                Ok(AgentToolResult {
                    content: vec![UserContent::text(format!("echoed: {value}"))],
                    details: json!({ "value": value }),
                    terminate: Some(true),
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

    let stream = FnStream::new(|_n, _model, _ctx, _opts, _cancel| {
        done_stream(create_assistant_message(
            vec![tool_call("tool-1", "echo", json!({ "value": "hello" }))],
            StopReason::ToolUse,
        ))
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

    assert_eq!(stream.calls(), 1);
    assert_eq!(roles(&messages), vec!["user", "assistant", "toolResult"]);
    let turn_ends = events
        .lock()
        .unwrap()
        .iter()
        .filter(|e| matches!(e, AgentEvent::TurnEnd { .. }))
        .count();
    assert_eq!(turn_ends, 1);
}

/// TS: "should continue after parallel tool calls when not all tool results terminate"
#[tokio::test]
async fn continues_after_parallel_tool_calls_when_not_all_terminate() {
    let tool = TestTool::new(
        "echo",
        "Echo",
        "Echo tool",
        value_schema(),
        Arc::new(|_id, args, _cancel, _on_update| {
            Box::pin(async move {
                let value = args
                    .get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let terminate = value == "first";
                Ok(AgentToolResult {
                    content: vec![UserContent::text(format!("echoed: {value}"))],
                    details: json!({ "value": value }),
                    terminate: Some(terminate),
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
    let mut config = AgentLoopConfig::new(create_model(), identity_converter());
    config.tool_execution = ToolExecutionMode::Parallel;

    let stream = FnStream::new(|n, _model, _ctx, _opts, _cancel| {
        if n == 0 {
            done_stream(create_assistant_message(
                vec![
                    tool_call("tool-1", "echo", json!({ "value": "first" })),
                    tool_call("tool-2", "echo", json!({ "value": "second" })),
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
        vec![create_user_message("echo both")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream.clone(),
    )
    .await
    .expect("loop failed");

    assert_eq!(stream.calls(), 2);
    assert_eq!(
        roles(&messages),
        vec!["user", "assistant", "toolResult", "toolResult", "assistant"]
    );
}

/// TS: "should allow afterToolCall to mark a tool batch as terminating"
#[tokio::test]
async fn after_tool_call_can_mark_batch_terminating() {
    let executed: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let context = AgentContext {
        system_prompt: "".into(),
        messages: vec![],
        tools: vec![echo_tool(executed)],
    };

    let mut config = AgentLoopConfig::new(create_model(), identity_converter());
    config.after_tool_call = Some(Arc::new(|_ctx, _cancel| {
        Box::pin(async move {
            Ok(Some(grain_agent_core::AfterToolCallResult {
                terminate: Some(true),
                ..Default::default()
            }))
        })
    }));

    let stream = FnStream::new(|_n, _model, _ctx, _opts, _cancel| {
        done_stream(create_assistant_message(
            vec![tool_call("tool-1", "echo", json!({ "value": "hello" }))],
            StopReason::ToolUse,
        ))
    });

    let (sink, _events) = recording_sink();
    run_agent_loop(
        vec![create_user_message("echo something")],
        context,
        config,
        sink,
        CancellationToken::new(),
        stream.clone(),
    )
    .await
    .expect("loop failed");

    assert_eq!(stream.calls(), 1);
}

// ---------------------------------------------------------------------------
// describe("agentLoopContinue with AgentMessage")
// ---------------------------------------------------------------------------

/// TS: "should throw when context has no messages"
#[tokio::test]
async fn continue_errors_when_context_has_no_messages() {
    let context = AgentContext {
        system_prompt: "You are helpful.".into(),
        messages: vec![],
        tools: vec![],
    };
    let config = AgentLoopConfig::new(create_model(), identity_converter());

    let (sink, _events) = recording_sink();
    let err = run_agent_loop_continue(
        context,
        config,
        sink,
        CancellationToken::new(),
        unused_stream_fn(),
    )
    .await
    .expect_err("expected continue to fail");

    assert_eq!(err.to_string(), "Cannot continue: no messages in context");
}

/// TS: "should continue from existing context without emitting user message events"
#[tokio::test]
async fn continue_from_existing_context_without_user_message_events() {
    let context = AgentContext {
        system_prompt: "You are helpful.".into(),
        messages: vec![create_user_message("Hello")],
        tools: vec![],
    };
    let config = AgentLoopConfig::new(create_model(), identity_converter());

    let stream = FnStream::new(|_n, _model, _ctx, _opts, _cancel| {
        done_stream(create_assistant_message(
            vec![text("Response")],
            StopReason::Stop,
        ))
    });

    let (sink, events) = recording_sink();
    let messages = run_agent_loop_continue(context, config, sink, CancellationToken::new(), stream)
        .await
        .expect("continue failed");

    // Should only return the new assistant message (not the existing user message)
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role(), "assistant");

    // Should NOT have user message events (that's the key difference from agentLoop)
    let events = events.lock().unwrap();
    let message_end_roles: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::MessageEnd { message } => Some(message.role().to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(message_end_roles, vec!["assistant"]);
}

/// TS: "should allow custom message types as last message (caller responsibility)"
#[tokio::test]
async fn continue_allows_custom_message_as_last_message() {
    let custom_message = AgentMessage::Custom(json!({
        "role": "custom",
        "text": "Hook content",
        "timestamp": now_ms(),
    }));
    let context = AgentContext {
        system_prompt: "You are helpful.".into(),
        messages: vec![custom_message],
        tools: vec![],
    };

    let mut config = AgentLoopConfig::new(create_model(), identity_converter());
    config.convert_to_llm = Arc::new(|messages: Vec<AgentMessage>| {
        Box::pin(async move {
            // Convert custom to user message
            messages
                .into_iter()
                .filter_map(|m| match m {
                    AgentMessage::Custom(v)
                        if v.get("role").and_then(|r| r.as_str()) == Some("custom") =>
                    {
                        Some(Message::User(UserMessage {
                            content: vec![UserContent::text(
                                v.get("text").and_then(|t| t.as_str()).unwrap_or_default(),
                            )],
                            timestamp: v.get("timestamp").and_then(|t| t.as_i64()).unwrap_or(0),
                        }))
                    }
                    AgentMessage::Standard(m) => Some(m),
                    AgentMessage::Custom(_) => None,
                })
                .collect()
        })
    });

    let stream = FnStream::new(|_n, _model, _ctx, _opts, _cancel| {
        done_stream(create_assistant_message(
            vec![text("Response to custom message")],
            StopReason::Stop,
        ))
    });

    // Should not fail - the custom message will be converted to user message
    let (sink, _events) = recording_sink();
    let messages = run_agent_loop_continue(context, config, sink, CancellationToken::new(), stream)
        .await
        .expect("continue failed");

    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role(), "assistant");
}

// ---------------------------------------------------------------------------
// Fixture-derived wire-format vector (not a named upstream `it` block)
// ---------------------------------------------------------------------------

/// Upstream's `createUserMessage` fixture produces `{ role: "user",
/// content: "<string>", timestamp }` — string content is a first-class
/// `UserMessage` shape in pi-ai (types.ts:393-397) and every loop test runs
/// on it. Patch-7 normalizes the string form into a single text block on
/// deserialization instead of letting it fall through `AgentMessage`'s
/// untagged enum into `Custom`.
#[test]
fn user_message_string_content_wire_format() {
    let wire = json!({
        "role": "user",
        "content": "Hello",
        "timestamp": 1700000000000i64,
    });
    let parsed: AgentMessage = serde_json::from_value(wire).expect("user message should parse");
    assert!(
        matches!(parsed, AgentMessage::Standard(Message::User(_))),
        "string-content user message must deserialize as a standard user message, got {parsed:?}"
    );
    // The string normalizes to the structured single-text-block form.
    let AgentMessage::Standard(Message::User(user)) = &parsed else {
        unreachable!()
    };
    assert_eq!(user.content, vec![UserContent::text("Hello")]);
    assert_eq!(user.timestamp, 1700000000000i64);
}
