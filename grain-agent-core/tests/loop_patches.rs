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
    AgentContext, AgentEvent, AgentLoopConfig, AgentToolError, AgentToolResult, StopReason,
    UserContent, run_agent_loop,
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
                vec![tool_call("tool-1", "echo", json!({ "value": { "nested": true } }))],
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
            None
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
