# pi-agent-core loop test port (WP1 → WP4)

This directory carries the executable spec of the agent-loop semantics: a
faithful port of the upstream TypeScript loop test-suite into Rust.

- **Upstream repo:** <https://github.com/earendil-works/pi>
- **Pinned spec commit:** `34239180` (`34239180ac5c80366def592b529a3a1b882b4a16`)
- **Ported files:**
  - `packages/agent/test/agent-loop.test.ts` → `tests/agent_loop.rs`
  - `packages/agent/test/agent.test.ts` → `tests/agent.rs`
  - shared fixtures (`createModel`, `createAssistantMessage`,
    `createUserMessage`, `identityConverter`, `MockAssistantStream`) →
    `tests/common/mod.rs`
- **Fresh WP4 vectors:** `tests/loop_patches.rs` — behaviors in the WP4
  divergence ledger with no covering vector in the two ported files
  (patch-1, patch-2, patch-8, patch-9 serde/wire vectors), derived directly
  from the upstream sources and cited per test.

## Rules of this suite

- A ported test asserts the **same semantics** as its TS source: same event
  sequences, same orderings, same edge cases. Assertions are never weakened to
  make a test pass.
- A faithfully-ported test that fails against the current loop stays exact and
  is marked `#[ignore = "patch-N: <reason>"]`, where `patch-N` refers to the
  WP4 divergence ledger. **As of WP4 the ledger is repaid: no ignored tests
  remain** (`cargo test -p grain-agent-core -- --ignored` runs nothing).
- A TS test that exercises a TS-only mechanism is **skipped** (not ported,
  not approximated) and documented below.

## Mapping table — `agent-loop.test.ts` → `tests/agent_loop.rs`

| TS test | Rust test fn | Status |
| --- | --- | --- |
| default stream function compatibility › uses the configured default when a legacy caller omits streamFn | — | skipped (TS-only: `setDefaultStreamFn` global registry + legacy omitted-`streamFn` call signature; Rust `run_agent_loop` requires an explicit `StreamFn`) |
| should emit events with AgentMessage types | `emits_events_with_agent_message_types` | passing |
| should handle custom message types via convertToLlm | `handles_custom_message_types_via_convert_to_llm` | passing |
| should apply transformContext before convertToLlm | `applies_transform_context_before_convert_to_llm` | passing |
| should handle tool calls and results | `handles_tool_calls_and_results` | passing — **full port** (patch-9 restored the tool-result usage plumbing; the previously dropped usage assertions are now included) |
| should not execute tool calls from a length-truncated assistant message | `does_not_execute_tool_calls_from_length_truncated_message` | passing (patch-6) |
| should execute mutated beforeToolCall args without revalidation | `executes_mutated_before_tool_call_args_without_revalidation` | passing (patch-3: hook rewrites args via the `BeforeToolCallResult::args` override channel; assertion unchanged — execute sees `123`) |
| should prepare tool arguments for validation | `prepares_tool_arguments_for_validation` | passing |
| should emit tool_execution_end in completion order but persist tool results in source order | `emits_tool_execution_end_in_completion_order_persists_results_in_source_order` | passing |
| should inject queued messages after all tool calls complete | `injects_queued_messages_after_all_tool_calls_complete` | passing |
| should force sequential execution when a tool has executionMode=sequential even with default parallel config | `forces_sequential_when_tool_has_sequential_execution_mode` | passing |
| should force sequential execution when one of multiple tools has executionMode=sequential | `forces_sequential_when_one_of_multiple_tools_is_sequential` | passing |
| should allow parallel execution when all tools have executionMode=parallel | `allows_parallel_when_all_tools_parallel` | passing |
| should use prepareNextTurn snapshot before continuing | `uses_prepare_next_turn_snapshot_before_continuing` | passing |
| should stop after the current turn when shouldStopAfterTurn returns true | `stops_after_turn_when_should_stop_after_turn_returns_true` | passing |
| should stop after a tool batch when every tool result sets terminate=true | `stops_after_tool_batch_when_all_results_terminate` | passing |
| should continue after parallel tool calls when not all tool results terminate | `continues_after_parallel_tool_calls_when_not_all_terminate` | passing |
| should allow afterToolCall to mark a tool batch as terminating | `after_tool_call_can_mark_batch_terminating` | passing |
| agentLoopContinue › should throw when context has no messages | `continue_errors_when_context_has_no_messages` | passing |
| agentLoopContinue › should continue from existing context without emitting user message events | `continue_from_existing_context_without_user_message_events` | passing |
| agentLoopContinue › should allow custom message types as last message (caller responsibility) | `continue_allows_custom_message_as_last_message` | passing |
| *(fixture-derived, no named TS test)* upstream `createUserMessage` wire shape (`content: "<string>"`) | `user_message_string_content_wire_format` | passing (patch-7) |

## Mapping table — `agent.test.ts` → `tests/agent.rs`

| TS test | Rust test fn | Status |
| --- | --- | --- |
| uses the configured default when a legacy caller omits streamFn | — | skipped (TS-only: `setDefaultStreamFn` + `Reflect.construct(Agent, [{}])`; Rust `AgentOptions` requires an explicit `StreamFn`) |
| should create an agent instance with default state | `creates_agent_with_default_state` | passing (model passed explicitly as `Model::unknown()`, the exported equivalent of TS `DEFAULT_MODEL`) |
| should create an agent instance with custom initial state | `creates_agent_with_custom_initial_state` | passing (no `getModel` registry in core; equivalent descriptor built inline) |
| should subscribe to events | `subscribe_to_events` | passing |
| emits full lifecycle events for thrown run failures | `emits_full_lifecycle_events_for_thrown_run_failures` | passing (TS sync throw expressed as `Err` from `LlmStream::stream`) |
| should await async subscribers before prompt resolves | `awaits_async_subscribers_before_prompt_resolves` | passing |
| waitForIdle should wait for async subscribers | `wait_for_idle_waits_for_async_subscribers` | passing (patch-5: `Agent::wait_for_idle` resolves only after the run and all awaited `agent_end` subscribers settle; see translation note) |
| should pass the active abort signal to subscribers | `passes_active_abort_signal_to_subscribers` | passing (`AbortSignal` ↔ `CancellationToken`) |
| should ignore tool updates after the tool execution settles | `ignores_tool_updates_after_execution_settles` | passing (patch-4) |
| should ignore a settled parallel tool update while another tool is still running | `ignores_settled_parallel_tool_update_while_other_tool_running` | passing (patch-4) |
| should update state with mutators | `updates_state_with_mutators` | passing — **partial** (reference-identity / live-array-push checks are TS-only, see notes) |
| should support steering message queue | `supports_steering_message_queue` | passing |
| should support follow-up message queue | `supports_follow_up_message_queue` | passing |
| should handle abort controller | `handles_abort_with_no_active_run` | passing |
| should throw when prompt() called while streaming | `errors_when_prompt_called_while_streaming` | passing (asserts `AgentError::AlreadyRunning` instead of the TS error copy) |
| should throw when continue() called while streaming | `errors_when_continue_called_while_streaming` | passing (same note) |
| continue() should process queued follow-up messages after an assistant turn | `continue_processes_queued_follow_up_after_assistant_turn` | passing |
| continue() should keep one-at-a-time steering semantics from assistant tail | `continue_keeps_one_at_a_time_steering_from_assistant_tail` | passing |
| keeps legacy prepareNextTurn signal callback behavior | `prepare_next_turn_receives_run_cancellation_token` | passing (legacy vs. with-context signature split is TS-only; Rust has one `prepare_next_turn(context, cancel)` hook) |
| forwards sessionId to streamFunction options | `forwards_session_id_to_stream_options` | passing — **full port** (patch-11 added `Agent::set_session_id`; the mid-life setter half is now included) |

## Counts

- TS tests found: 21 (`agent-loop.test.ts`) + 20 (`agent.test.ts`) = **41**
- Ported: **39** (1 of them partial: see notes) + 1 fixture-derived extra
- Passing: **40** (39 ports + the fixture-derived wire vector)
- Ignored: **0** — the WP1 patch-3/4/6/7 debt and the patch-5 skip are repaid
- Skipped: **2** — both TS-only `setDefaultStreamFn` mechanisms
- Fresh WP4 vectors: `tests/loop_patches.rs` (9 tests: patch-1 ×3, patch-2
  ×2, patch-8 ×1, patch-9 ×3) plus harness-side patch-10 vectors in
  `grain-agent-harness` (`session.rs` / `messages.rs` test modules) and
  validation unit vectors in `src/validation.rs`

## Translation notes

- **Fixtures.** `MockAssistantStream` + `queueMicrotask(push done)` is ported
  as a stream yielding a single terminal `Done` event (`done_stream`). The TS
  tests never push a `start` event except in the abort mocks, which are ported
  as an async stream yielding `Start`, then awaiting cancellation, then
  yielding `Error` with an `Aborted` message.
- **`setTimeout(() => release(), 20)`** → `release_after(notify, 20)`
  (spawned `tokio::time::sleep` + `Notify`). Deferred promises → `Notify`;
  shared flags → atomics.
- **`handles_tool_calls_and_results` (patch-9, now full).** Upstream asserts:
  `tool.execute` returns `usage`, `afterToolCall` observes `result.usage`
  (equal to the tool's reading) and replaces it via `{ usage: patched }`, and
  the persisted toolResult message carries the patched usage. WP4's patch-9
  added `AgentToolResult.usage`, `AfterToolCallResult.usage`, and
  `ToolResultMessage.usage` with the upstream spread-merge
  (agent-loop.ts:734-741, 773-787), and the previously dropped assertions are
  now part of the ported test.
- **`executes_mutated_before_tool_call_args_without_revalidation`
  (patch-3, now passing).** Upstream mutates the shared `args` object inside
  `beforeToolCall` by reference. Rust hooks receive a cloned
  `serde_json::Value`, so the rewrite goes through the explicit
  `BeforeToolCallResult::args` override channel; the tool executes with the
  returned args without revalidation. The observable assertion is unchanged:
  `execute` sees `123` (a number, against a string-typed schema — proof no
  second validation pass runs now that patch-1 made validation real).
- **`wait_for_idle_waits_for_async_subscribers` (patch-5, now ported).** In
  TS, `agent.prompt(...)` synchronously registers the active run before
  `waitForIdle()` is called on the next line. The Rust prompt runs on a
  spawned task, so the port first waits for the run to be observably
  streaming (the listener blocks the run on a barrier well before
  `agent_end`), then asserts the same contract.
- **`user_message_string_content_wire_format` (patch-7, now passing).** Not a
  named TS test: it pins the upstream `createUserMessage` fixture shape
  (`content: "<string>"`), which every upstream loop test runs on. Patch-7
  normalizes string content into the structured single-text-block form on
  deserialization (and `null`/missing content to `[]`).
- **`updates_state_with_mutators` (partial).** The TS assertions
  `expect(state.tools).not.toBe(tools)` (defensive copy identity) and
  `state.messages.push(...)` appending to live agent state depend on the TS
  mutable-`state`-object API. Rust `state()` returns an owned snapshot, so
  only the setter round-trips are portable.
- **Error-copy strings.** TS asserts exact human-readable error copy for
  prompt/continue-while-streaming; the Rust port asserts the typed
  `AgentError::AlreadyRunning`. The `agentLoopContinue` empty-context error
  string ("Cannot continue: no messages in context") is identical in both
  implementations and asserted exactly.
- **`process.on("unhandledRejection")`** bookkeeping in the settled-update
  test is Node-specific and not ported; the behavioural assertions are kept.
- **`shouldStopAfterTurn` message-role assertion.** TS asserts
  `message.role === "assistant"`; in Rust the hook's `message` is typed
  `AssistantMessage`, so the assertion holds by construction.
- **Validation error copy (patch-1).** The Rust default validator mirrors
  upstream `validateToolArguments` (pi-ai `utils/validation.ts:278-310`):
  same coercion pipeline, same failure shape (`Validation failed for tool
  "<name>":\n  - <path>: <message>\n\nReceived arguments:\n<json>`). The
  per-keyword `<message>` text comes from the `jsonschema` crate rather than
  typebox, so the human-readable copy of individual validation errors
  differs; vectors assert the structural shape and the exact upstream
  behavioural semantics (no execution, isError result, loop continues).

## Repaid divergences (WP4)

The WP1 ledger entries patch-1 … patch-11 have all landed; see the
`wp4/loop-patches` commit series for the per-patch citations. The previously
"unmapped divergence" (`Agent.sessionId` setter) landed as patch-11.

## Behaviors covered only by fresh vectors

Upstream has no test in the two ported files for: patch-1 (schema
validation/coercion), patch-2 (hook throw → isError containment), patch-8
(`on_payload` / `on_response` / `thinking_budgets` plumbing), patch-9's wire
names (`pending`, `max`, `cacheWrite1h`, `reasoning`, `addedToolNames`) and
patch-10 (session entry kinds — lives outside `packages/agent`'s loop tests).
Those are pinned by `tests/loop_patches.rs`, `src/validation.rs` unit
vectors, and the `grain-agent-harness` session/messages test modules, each
citing the upstream source lines they mirror.
