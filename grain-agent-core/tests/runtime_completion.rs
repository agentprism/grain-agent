//! WP19 runtime-completion coverage: rust-host ledger items 12 and 13, plus
//! the G6 per-call provider-configuration channel.
//!
//! Upstream references (pinned pi commit 34239180):
//! - `ToolCall.thoughtSignature` — `packages/ai/src/types.ts:365`
//! - thought-signature replay rule — `packages/ai/src/api/transform-messages.ts:127-145`
//! - `AssistantMessage.rawStopReason` — `packages/ai/src/types.ts:411`
//! - per-call option surface — `packages/ai/src/types.ts:116-198`
//!   (`AgentLoopConfig extends SimpleStreamOptions`, `packages/agent/src/types.ts:144`,
//!   spread into the stream call at `packages/agent/src/agent-loop.ts:308-312`)

mod common;

use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use common::{FnStream, create_assistant_message, create_model, done_stream, text};
use futures::StreamExt;
use grain_agent_core::{
    Agent, AgentMessage, AgentOptions, AssistantContent, AssistantMessage, CacheRetention, Message,
    Model, StopReason, StreamOptions, ToolCall, is_same_model,
    strip_cross_model_thought_signatures,
};
use serde_json::json;

/// A model distinct from `create_model()` in every one of the three fields
/// upstream's `isSameModel` compares.
fn other_model() -> Model {
    Model {
        id: "claude-sonnet-4".into(),
        name: "claude-sonnet-4".into(),
        api: "anthropic-messages".into(),
        provider: "anthropic".into(),
        ..create_model()
    }
}

fn signed_tool_call(signature: Option<&str>) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: "call_1".into(),
        name: "read".into(),
        arguments: json!({ "path": "/x" }),
        thought_signature: signature.map(str::to_string),
    })
}

/// The tool call's `thought_signature` as the stream fn observed it.
fn seam_signature(ctx_messages: &[Message]) -> Option<String> {
    ctx_messages.iter().find_map(|m| match m {
        Message::Assistant(a) => a.content.iter().find_map(|b| match b {
            AssistantContent::ToolCall(tc) => Some(tc.thought_signature.clone()),
            _ => None,
        }),
        _ => None,
    })?
}

// ---------------------------------------------------------------------------
// Ledger 12 — ToolCall.thought_signature + its replay path
// ---------------------------------------------------------------------------

/// Upstream's `isSameModel` is a three-part test — provider AND api AND the
/// message's `model` against the model's **`id`**. Any single mismatch makes
/// it a cross-model replay.
#[test]
fn is_same_model_compares_provider_api_and_model_id() {
    let model = create_model();
    let mut msg = create_assistant_message(vec![text("x")], StopReason::Stop);
    assert!(is_same_model(&msg, &model), "identical triple must match");

    for mutate in [
        (|m: &mut AssistantMessage| m.provider = "anthropic".into()) as fn(&mut AssistantMessage),
        |m: &mut AssistantMessage| m.api = "anthropic-messages".into(),
        |m: &mut AssistantMessage| m.model = "some-other-id".into(),
    ] {
        let mut divergent = msg.clone();
        mutate(&mut divergent);
        assert!(
            !is_same_model(&divergent, &model),
            "a single differing field must make it cross-model"
        );
    }

    // The comparison is against `model.id`, not `model.name`.
    msg.model = model.name.clone();
    msg.model.push_str("-not-the-id");
    assert!(!is_same_model(&msg, &model));
}

/// Same-model replay keeps the signature; cross-model replay drops it.
/// Text and thinking blocks are untouched either way.
#[test]
fn strip_cross_model_thought_signatures_matches_upstream_rule() {
    let build = || {
        vec![Message::Assistant(AssistantMessage {
            content: vec![text("hello"), signed_tool_call(Some("sig-abc"))],
            ..create_assistant_message(vec![], StopReason::ToolUse)
        })]
    };

    let mut same = build();
    strip_cross_model_thought_signatures(&mut same, &create_model());
    assert_eq!(
        seam_signature(&same),
        Some("sig-abc".into()),
        "same-model replay must preserve the signature verbatim"
    );

    let mut cross = build();
    strip_cross_model_thought_signatures(&mut cross, &other_model());
    assert_eq!(
        seam_signature(&cross),
        None,
        "cross-model replay must drop the signature"
    );

    // Non-tool-call blocks survive the strip untouched, and no message is
    // added or removed.
    let Message::Assistant(a) = &cross[0] else {
        panic!("expected assistant message")
    };
    assert_eq!(cross.len(), 1);
    assert_eq!(a.content.len(), 2);
    assert!(matches!(a.content[0], AssistantContent::Text(_)));
}

/// A signature-only turn needs no empty-text thinking block to carry the
/// signature: the slot on the tool call is the replay path.
#[test]
fn signature_rides_the_tool_call_without_a_thinking_block() {
    let mut messages = vec![Message::Assistant(AssistantMessage {
        content: vec![signed_tool_call(Some("sig-only"))],
        ..create_assistant_message(vec![], StopReason::ToolUse)
    })];
    strip_cross_model_thought_signatures(&mut messages, &create_model());

    let Message::Assistant(a) = &messages[0] else {
        panic!("expected assistant message")
    };
    assert!(
        !a.content
            .iter()
            .any(|b| matches!(b, AssistantContent::Thinking(_))),
        "no synthesized thinking block should be needed"
    );
    assert_eq!(seam_signature(&messages), Some("sig-only".into()));
}

/// The loop applies the replay rule before handing the context to the stream
/// fn — same model in, signature intact at the seam.
#[tokio::test]
async fn loop_preserves_signature_for_same_model_replay() {
    let seen: Arc<StdMutex<Option<Vec<Message>>>> = Arc::new(StdMutex::new(None));
    let capture = seen.clone();
    let stream = FnStream::new(move |_n, _model, ctx, _opts, _cancel| {
        *capture.lock().unwrap() = Some(ctx.messages.clone());
        done_stream(create_assistant_message(vec![text("ok")], StopReason::Stop))
    });

    let mut options = AgentOptions::new(create_model(), stream);
    options.messages = vec![AgentMessage::assistant(AssistantMessage {
        content: vec![signed_tool_call(Some("sig-same"))],
        ..create_assistant_message(vec![], StopReason::ToolUse)
    })];

    Agent::new(options)
        .prompt_text("go")
        .await
        .expect("prompt failed");

    let observed = seen.lock().unwrap().clone().expect("stream fn not called");
    assert_eq!(seam_signature(&observed), Some("sig-same".into()));
}

/// The transcript was produced by one model and is being replayed to another
/// (the shape `prepare_next_turn`'s model swap produces): the signature must
/// not reach the new provider.
#[tokio::test]
async fn loop_strips_signature_for_cross_model_replay() {
    let seen: Arc<StdMutex<Option<Vec<Message>>>> = Arc::new(StdMutex::new(None));
    let capture = seen.clone();
    let stream = FnStream::new(move |_n, _model, ctx, _opts, _cancel| {
        *capture.lock().unwrap() = Some(ctx.messages.clone());
        done_stream(create_assistant_message(vec![text("ok")], StopReason::Stop))
    });

    // Transcript entry is from `create_model()`; the agent targets `other_model()`.
    let mut options = AgentOptions::new(other_model(), stream);
    options.messages = vec![AgentMessage::assistant(AssistantMessage {
        content: vec![signed_tool_call(Some("sig-cross"))],
        ..create_assistant_message(vec![], StopReason::ToolUse)
    })];

    Agent::new(options)
        .prompt_text("go")
        .await
        .expect("prompt failed");

    let observed = seen.lock().unwrap().clone().expect("stream fn not called");
    assert_eq!(
        seam_signature(&observed),
        None,
        "signature must not cross the model boundary"
    );
}

/// Wire shape: camelCase key, omitted entirely when absent, and an upstream
/// payload without the key deserializes cleanly.
#[test]
fn thought_signature_wire_shape_matches_upstream() {
    let signed = ToolCall {
        id: "c1".into(),
        name: "read".into(),
        arguments: json!({}),
        thought_signature: Some("sig".into()),
    };
    let v = serde_json::to_value(&signed).unwrap();
    assert_eq!(v["thoughtSignature"], json!("sig"));

    let bare = ToolCall {
        thought_signature: None,
        ..signed.clone()
    };
    let v = serde_json::to_value(&bare).unwrap();
    assert!(
        v.get("thoughtSignature").is_none(),
        "absent signature must not appear on the wire"
    );

    // Upstream transcripts predating the field still load.
    let parsed: ToolCall =
        serde_json::from_value(json!({ "id": "c1", "name": "read", "arguments": {} })).unwrap();
    assert_eq!(parsed.thought_signature, None);

    let round: ToolCall = serde_json::from_value(serde_json::to_value(&signed).unwrap()).unwrap();
    assert_eq!(round.thought_signature, Some("sig".into()));
}

// ---------------------------------------------------------------------------
// Ledger 13 — AssistantMessage.raw_stop_reason
// ---------------------------------------------------------------------------

/// The provider's raw stop string rides the terminal event through the loop
/// and lands on the agent's transcript unchanged, independently of the
/// normalized `stop_reason`.
#[tokio::test]
async fn raw_stop_reason_crosses_the_provider_seam() {
    let stream = FnStream::new(|_n, _model, _ctx, _opts, _cancel| {
        // Anthropic maps "refusal" onto the normalized `Stop`; only the raw
        // slot preserves which of several raw reasons actually fired.
        done_stream(AssistantMessage {
            raw_stop_reason: Some("refusal".into()),
            ..create_assistant_message(vec![text("no")], StopReason::Stop)
        })
    });

    let agent = Agent::new(AgentOptions::new(create_model(), stream));
    agent.prompt_text("hi").await.expect("prompt failed");

    let state = agent.state().await;
    let last = state
        .messages
        .iter()
        .rev()
        .find_map(|m| match m {
            AgentMessage::Standard(Message::Assistant(a)) => Some(a.clone()),
            _ => None,
        })
        .expect("no assistant message in transcript");

    assert_eq!(last.raw_stop_reason.as_deref(), Some("refusal"));
    assert_eq!(
        last.stop_reason,
        StopReason::Stop,
        "the normalized reason stays independent of the raw one"
    );
}

/// Locally synthesized failures have no provider stop string, so the slot
/// stays empty rather than inventing one.
#[tokio::test]
async fn locally_synthesized_failures_carry_no_raw_stop_reason() {
    let stream = FnStream::new(|_n, _model, _ctx, _opts, _cancel| {
        // Terminate without ever emitting a terminal event.
        futures::stream::iter(Vec::new()).boxed()
    });

    let agent = Agent::new(AgentOptions::new(create_model(), stream));
    agent.prompt_text("hi").await.expect("prompt failed");

    let state = agent.state().await;
    let last = state
        .messages
        .iter()
        .rev()
        .find_map(|m| match m {
            AgentMessage::Standard(Message::Assistant(a)) => Some(a.clone()),
            _ => None,
        })
        .expect("no assistant message in transcript");

    assert_eq!(last.stop_reason, StopReason::Error);
    assert_eq!(last.raw_stop_reason, None);
}

/// Wire shape: camelCase key, omitted when absent, upstream payloads load.
#[test]
fn raw_stop_reason_wire_shape_matches_upstream() {
    let msg = AssistantMessage {
        raw_stop_reason: Some("MALFORMED_FUNCTION_CALL".into()),
        ..create_assistant_message(vec![text("x")], StopReason::Error)
    };
    let v = serde_json::to_value(&msg).unwrap();
    assert_eq!(v["rawStopReason"], json!("MALFORMED_FUNCTION_CALL"));

    let bare = create_assistant_message(vec![text("x")], StopReason::Stop);
    let v = serde_json::to_value(&bare).unwrap();
    assert!(
        v.get("rawStopReason").is_none(),
        "absent raw stop reason must not appear on the wire"
    );

    let round: AssistantMessage =
        serde_json::from_value(serde_json::to_value(&msg).unwrap()).unwrap();
    assert_eq!(
        round.raw_stop_reason.as_deref(),
        Some("MALFORMED_FUNCTION_CALL")
    );
}

// ---------------------------------------------------------------------------
// G6 — per-call provider configuration reaches the StreamFn seam
// ---------------------------------------------------------------------------

/// Every per-call knob set on the loop's `StreamOptions` arrives at the
/// stream fn for that request. Upstream gets this by spreading the whole
/// `AgentLoopConfig` (which *is* a `SimpleStreamOptions`) into the call;
/// grain carries the same surface in `AgentLoopConfig::stream_options`.
#[tokio::test]
async fn per_call_configuration_reaches_the_stream_fn() {
    use grain_agent_core::{AgentLoopConfig, EventSink, run_agent_loop};
    use std::collections::HashMap;

    let captured: Arc<StdMutex<Option<StreamOptions>>> = Arc::new(StdMutex::new(None));
    let capture = captured.clone();
    let stream = FnStream::new(move |_n, _model, _ctx, opts, _cancel| {
        *capture.lock().unwrap() = Some(opts.clone());
        done_stream(create_assistant_message(vec![text("ok")], StopReason::Stop))
    });

    let mut headers = HashMap::new();
    headers.insert("x-trace".to_string(), Some("abc123".to_string()));
    // A `None` value suppresses a provider default header of the same name.
    headers.insert("user-agent".to_string(), None);

    let mut env = HashMap::new();
    env.insert("AWS_REGION".to_string(), "us-west-2".to_string());

    let mut config = AgentLoopConfig::new(create_model(), common::identity_converter());
    config.stream_options = StreamOptions {
        api_key: Some("sk-static".into()),
        session_id: Some("sess-42".into()),
        transport: Some("websocket".into()),
        temperature: Some(0.25),
        max_tokens: Some(2048),
        cache_retention: Some(CacheRetention::Long),
        max_retries: Some(5),
        max_retry_delay_ms: Some(30_000),
        timeout_ms: Some(600_000),
        websocket_connect_timeout_ms: Some(15_000),
        headers: Some(headers),
        metadata: Some(json!({ "user_id": "u-7" })),
        env: Some(env),
        extra: json!({ "provider_specific": true }),
        ..StreamOptions::default()
    };

    let sink: EventSink = Arc::new(|_| Box::pin(async {}));
    let context = grain_agent_core::AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Vec::new(),
    };
    run_agent_loop(
        vec![common::create_user_message("hi")],
        context,
        config,
        sink,
        Default::default(),
        stream,
    )
    .await
    .expect("loop failed");

    let o = captured
        .lock()
        .unwrap()
        .clone()
        .expect("stream fn not called");
    assert_eq!(o.api_key.as_deref(), Some("sk-static"));
    assert_eq!(o.session_id.as_deref(), Some("sess-42"));
    assert_eq!(o.transport.as_deref(), Some("websocket"));
    assert_eq!(o.temperature, Some(0.25));
    assert_eq!(o.max_tokens, Some(2048));
    assert_eq!(o.cache_retention, Some(CacheRetention::Long));
    assert_eq!(o.max_retries, Some(5));
    assert_eq!(o.max_retry_delay_ms, Some(30_000));
    assert_eq!(o.timeout_ms, Some(600_000));
    assert_eq!(o.websocket_connect_timeout_ms, Some(15_000));
    assert_eq!(o.metadata, Some(json!({ "user_id": "u-7" })));
    assert_eq!(o.extra, json!({ "provider_specific": true }));

    let h = o.headers.expect("headers must survive the seam");
    assert_eq!(h.get("x-trace"), Some(&Some("abc123".to_string())));
    assert_eq!(
        h.get("user-agent"),
        Some(&None),
        "an explicit null must survive as a suppression, not vanish"
    );
    assert_eq!(
        o.env.expect("env must survive").get("AWS_REGION"),
        Some(&"us-west-2".to_string())
    );
}

/// Upstream resolves the key as `getApiKey(provider) || config.apiKey`
/// (`packages/agent/src/agent-loop.ts:306`): the dynamic provider wins when it
/// yields a key, and the statically configured key is the fallback.
#[tokio::test]
async fn dynamic_api_key_overrides_static_and_falls_back_to_it() {
    use grain_agent_core::{AgentLoopConfig, EventSink, GetApiKeyFn, run_agent_loop};

    async fn resolve(get_api_key: Option<GetApiKeyFn>) -> Option<String> {
        let captured: Arc<StdMutex<Option<StreamOptions>>> = Arc::new(StdMutex::new(None));
        let capture = captured.clone();
        let stream = FnStream::new(move |_n, _model, _ctx, opts, _cancel| {
            *capture.lock().unwrap() = Some(opts.clone());
            done_stream(create_assistant_message(vec![text("ok")], StopReason::Stop))
        });

        let mut config = AgentLoopConfig::new(create_model(), common::identity_converter());
        config.stream_options.api_key = Some("sk-static".into());
        config.get_api_key = get_api_key;

        let sink: EventSink = Arc::new(|_| Box::pin(async {}));
        let context = grain_agent_core::AgentContext {
            system_prompt: String::new(),
            messages: Vec::new(),
            tools: Vec::new(),
        };
        run_agent_loop(
            vec![common::create_user_message("hi")],
            context,
            config,
            sink,
            Default::default(),
            stream,
        )
        .await
        .expect("loop failed");
        let o = captured.lock().unwrap().clone().expect("not called");
        o.api_key
    }

    // No dynamic provider → the static key is used.
    assert_eq!(resolve(None).await.as_deref(), Some("sk-static"));

    // Dynamic provider yields a key → it wins (the expiring-token path).
    let dynamic: GetApiKeyFn = Arc::new(|_provider| Box::pin(async { Some("sk-fresh".into()) }));
    assert_eq!(resolve(Some(dynamic)).await.as_deref(), Some("sk-fresh"));

    // Dynamic provider yields nothing → fall back to the static key.
    let empty: GetApiKeyFn = Arc::new(|_provider| Box::pin(async { None }));
    assert_eq!(resolve(Some(empty)).await.as_deref(), Some("sk-static"));
}
