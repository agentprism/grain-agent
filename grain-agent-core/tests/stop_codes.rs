//! The structured error-code channel (grain-side extension; see
//! `ErrorCode` in `types.rs`).
//!
//! These are fresh grain vectors, not ports: upstream pi carries
//! loop-terminating failures as free-text `errorMessage` only, which forces
//! embedders into substring matching. The channel under test lets a
//! producer (provider adapter, engine bridge, scripted stream) attach a
//! typed code that survives the loop boundary verbatim — on the final
//! assistant message's `error_code` — so embedders `match` instead of
//! sniffing text.

mod common;

use std::sync::Arc;

use common::*;
use futures::StreamExt;
use grain_agent_core::{
    Agent, AgentMessage, AgentOptions, AssistantMessageEvent, ErrorCode, Message, StopReason,
    StreamError, StreamFn,
};

fn final_assistant(agent_messages: &[AgentMessage]) -> grain_agent_core::AssistantMessage {
    agent_messages
        .iter()
        .rev()
        .find_map(|m| match m {
            AgentMessage::Standard(Message::Assistant(a)) => Some(a.clone()),
            _ => None,
        })
        .expect("an assistant message must exist")
}

// ---------------------------------------------------------------------------
// Serde: the wire shape is a plain optional string, upstream-compatible.
// ---------------------------------------------------------------------------

#[test]
fn error_code_serde_round_trips_and_tolerates_upstream_shapes() {
    // An upstream-shaped message (no errorCode key) deserializes with None.
    let upstream = serde_json::json!({
        "content": [],
        "api": "openai-responses",
        "provider": "openai",
        "model": "mock",
        "stopReason": "error",
        "errorMessage": "boom",
        "timestamp": 1
    });
    let message: grain_agent_core::AssistantMessage =
        serde_json::from_value(upstream).expect("upstream shape must deserialize");
    assert_eq!(message.error_code, None);

    // None is omitted from the wire, keeping serialized messages
    // byte-compatible with upstream transcripts.
    let encoded = serde_json::to_value(&message).unwrap();
    assert!(encoded.get("errorCode").is_none(), "{encoded}");

    // Named variants round-trip through their snake_case strings.
    for (code, wire) in [
        (ErrorCode::BudgetExhausted, "budget_exhausted"),
        (ErrorCode::AgentLimitExceeded, "agent_limit_exceeded"),
    ] {
        let encoded = serde_json::to_value(&code).unwrap();
        assert_eq!(encoded, serde_json::json!(wire));
        let decoded: ErrorCode = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, code);
        assert_eq!(code.as_str(), wire);
    }

    // Unknown code strings survive verbatim through Other.
    let decoded: ErrorCode =
        serde_json::from_value(serde_json::json!("SESSION_NOT_FOUND")).unwrap();
    assert_eq!(decoded, ErrorCode::Other("SESSION_NOT_FOUND".into()));
    assert_eq!(
        serde_json::to_value(&decoded).unwrap(),
        serde_json::json!("SESSION_NOT_FOUND")
    );
    assert_eq!(decoded.as_str(), "SESSION_NOT_FOUND");

    // An Other spelling a named variant canonicalizes on round-trip
    // (documented behavior).
    let recoded: ErrorCode = serde_json::from_value(
        serde_json::to_value(ErrorCode::Other("budget_exhausted".into())).unwrap(),
    )
    .unwrap();
    assert_eq!(recoded, ErrorCode::BudgetExhausted);
}

// ---------------------------------------------------------------------------
// The loop preserves a coded terminal-Error event onto the final message.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn coded_terminal_error_event_survives_the_loop_boundary() {
    let stream: StreamFn = FnStream::new(|_, _, _, _, _| {
        let mut result = create_assistant_message(vec![], StopReason::Error);
        result.error_message = Some("token budget exhausted after 1.2M tokens".into());
        result.error_code = Some(ErrorCode::BudgetExhausted);
        futures::stream::iter(vec![AssistantMessageEvent::Error {
            error: "token budget exhausted after 1.2M tokens".into(),
            result,
        }])
        .boxed()
    });

    let agent = Agent::new(AgentOptions::new(create_model(), stream));
    agent.prompt_text("go").await.expect("prompt starts");
    agent.wait_for_idle().await;

    let last = final_assistant(&agent.state().await.messages);
    assert_eq!(last.stop_reason, StopReason::Error);
    assert_eq!(last.error_code, Some(ErrorCode::BudgetExhausted));
    assert_eq!(
        last.error_message.as_deref(),
        Some("token budget exhausted after 1.2M tokens")
    );
}

// ---------------------------------------------------------------------------
// The degrade-gracefully path (StreamFn returns Err) carries the code too.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn coded_stream_error_reaches_the_synthesized_error_message() {
    struct CodedErrStream;
    #[async_trait::async_trait]
    impl grain_agent_core::LlmStream for CodedErrStream {
        async fn stream(
            &self,
            _model: &grain_agent_core::Model,
            _context: &grain_agent_core::LlmContext,
            _options: &grain_agent_core::StreamOptions,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> Result<grain_agent_core::AssistantStream, StreamError> {
            Err(StreamError::coded(
                ErrorCode::AgentLimitExceeded,
                "agent limit exceeded: 32 workers",
            ))
        }
    }

    let agent = Agent::new(AgentOptions::new(create_model(), Arc::new(CodedErrStream)));
    agent.prompt_text("go").await.expect("prompt starts");
    agent.wait_for_idle().await;

    let last = final_assistant(&agent.state().await.messages);
    assert_eq!(last.stop_reason, StopReason::Error);
    assert_eq!(last.error_code, Some(ErrorCode::AgentLimitExceeded));
    assert_eq!(
        last.error_message.as_deref(),
        Some("agent limit exceeded: 32 workers")
    );

    // The accessor pair on StreamError itself.
    let err = StreamError::coded(ErrorCode::Other("X".into()), "msg");
    assert_eq!(err.code(), Some(&ErrorCode::Other("X".into())));
    assert_eq!(err.to_string(), "msg");
    assert_eq!(StreamError::msg("plain").code(), None);
    assert_eq!(StreamError::Aborted.code(), None);
}

// ---------------------------------------------------------------------------
// A failure whose MESSAGE TEXT merely mentions a code string carries no
// structured code — the channel is explicit, never inferred (the exact
// misclassification substring matching produced).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn uncoded_error_text_mentioning_a_code_stays_uncoded() {
    let stream: StreamFn = FnStream::new(|_, _, _, _, _| {
        let mut result = create_assistant_message(vec![], StopReason::Error);
        result.error_message = Some("worker report cites TOKEN_BUDGET_EXHAUSTED in prose".into());
        // No error_code set by the producer.
        futures::stream::iter(vec![AssistantMessageEvent::Error {
            error: "worker report cites TOKEN_BUDGET_EXHAUSTED in prose".into(),
            result,
        }])
        .boxed()
    });

    let agent = Agent::new(AgentOptions::new(create_model(), stream));
    agent.prompt_text("go").await.expect("prompt starts");
    agent.wait_for_idle().await;

    let last = final_assistant(&agent.state().await.messages);
    assert_eq!(last.stop_reason, StopReason::Error);
    assert_eq!(
        last.error_code, None,
        "codes are explicit, never inferred from text"
    );
}

// ---------------------------------------------------------------------------
// A successful run carries no code.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn successful_runs_carry_no_error_code() {
    let stream: StreamFn = FnStream::new(|_, _, _, _, _| {
        done_stream(create_assistant_message(
            vec![text("done")],
            StopReason::Stop,
        ))
    });
    let agent = Agent::new(AgentOptions::new(create_model(), stream));
    agent.prompt_text("go").await.expect("prompt starts");
    agent.wait_for_idle().await;

    let last = final_assistant(&agent.state().await.messages);
    assert_eq!(last.stop_reason, StopReason::Stop);
    assert_eq!(last.error_code, None);
}
