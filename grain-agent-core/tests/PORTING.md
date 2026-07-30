# pi-agent-core loop test port (WP1)

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

## Rules of this suite

- A ported test asserts the **same semantics** as its TS source: same event
  sequences, same orderings, same edge cases. Assertions are never weakened to
  make a test pass.
- A faithfully-ported test that fails against the current loop stays exact and
  is marked `#[ignore = "patch-N: <reason>"]`, where `patch-N` refers to the
  WP4 divergence ledger. `cargo test -p grain-agent-core` is green because
  ignored tests do not run by default; `cargo test -- --ignored` shows the
  open debt.
- A TS test that exercises a TS-only mechanism is **skipped** (not ported,
  not approximated) and documented below.
- When WP4 lands a patch, un-ignore the corresponding tests; they are the
  acceptance criteria for that patch.

## Mapping table — `agent-loop.test.ts` → `tests/agent_loop.rs`

| TS test | Rust test fn | Status |
| --- | --- | --- |
| default stream function compatibility › uses the configured default when a legacy caller omits streamFn | — | skipped (TS-only: `setDefaultStreamFn` global registry + legacy omitted-`streamFn` call signature; Rust `run_agent_loop` requires an explicit `StreamFn`) |
| should emit events with AgentMessage types | `emits_events_with_agent_message_types` | passing |
| should handle custom message types via convertToLlm | `handles_custom_message_types_via_convert_to_llm` | passing |
| should apply transformContext before convertToLlm | `applies_transform_context_before_convert_to_llm` | passing |
| should handle tool calls and results | `handles_tool_calls_and_results` | passing — **partial** (usage assertions untranslatable: patch-9, see notes) |
| should not execute tool calls from a length-truncated assistant message | `does_not_execute_tool_calls_from_length_truncated_message` | ignored (patch-6) |
| should execute mutated beforeToolCall args without revalidation | `executes_mutated_before_tool_call_args_without_revalidation` | ignored (patch-3) |
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
| *(fixture-derived, no named TS test)* upstream `createUserMessage` wire shape (`content: "<string>"`) | `user_message_string_content_wire_format` | ignored (patch-7) |

## Mapping table — `agent.test.ts` → `tests/agent.rs`

| TS test | Rust test fn | Status |
| --- | --- | --- |
| uses the configured default when a legacy caller omits streamFn | — | skipped (TS-only: `setDefaultStreamFn` + `Reflect.construct(Agent, [{}])`; Rust `AgentOptions` requires an explicit `StreamFn`) |
| should create an agent instance with default state | `creates_agent_with_default_state` | passing (model passed explicitly as `Model::unknown()`, the exported equivalent of TS `DEFAULT_MODEL`) |
| should create an agent instance with custom initial state | `creates_agent_with_custom_initial_state` | passing (no `getModel` registry in core; equivalent descriptor built inline) |
| should subscribe to events | `subscribe_to_events` | passing |
| emits full lifecycle events for thrown run failures | `emits_full_lifecycle_events_for_thrown_run_failures` | passing (TS sync throw expressed as `Err` from `LlmStream::stream`) |
| should await async subscribers before prompt resolves | `awaits_async_subscribers_before_prompt_resolves` | passing |
| waitForIdle should wait for async subscribers | — | skipped (patch-5: `Agent` exposes no `wait_for_idle`; only the harness crate has one, a 10ms poll with a start race — the contract cannot be expressed against grain-agent-core) |
| should pass the active abort signal to subscribers | `passes_active_abort_signal_to_subscribers` | passing (`AbortSignal` ↔ `CancellationToken`) |
| should ignore tool updates after the tool execution settles | `ignores_tool_updates_after_execution_settles` | ignored (patch-4) |
| should ignore a settled parallel tool update while another tool is still running | `ignores_settled_parallel_tool_update_while_other_tool_running` | ignored (patch-4) |
| should update state with mutators | `updates_state_with_mutators` | passing — **partial** (reference-identity / live-array-push checks are TS-only, see notes) |
| should support steering message queue | `supports_steering_message_queue` | passing |
| should support follow-up message queue | `supports_follow_up_message_queue` | passing |
| should handle abort controller | `handles_abort_with_no_active_run` | passing |
| should throw when prompt() called while streaming | `errors_when_prompt_called_while_streaming` | passing (asserts `AgentError::AlreadyRunning` instead of the TS error copy) |
| should throw when continue() called while streaming | `errors_when_continue_called_while_streaming` | passing (same note) |
| continue() should process queued follow-up messages after an assistant turn | `continue_processes_queued_follow_up_after_assistant_turn` | passing |
| continue() should keep one-at-a-time steering semantics from assistant tail | `continue_keeps_one_at_a_time_steering_from_assistant_tail` | passing |
| keeps legacy prepareNextTurn signal callback behavior | `prepare_next_turn_receives_run_cancellation_token` | passing (legacy vs. with-context signature split is TS-only; Rust has one `prepare_next_turn(context, cancel)` hook) |
| forwards sessionId to streamFunction options | `forwards_session_id_to_stream_options` | passing — **partial** (mid-life `agent.sessionId = …` setter half untranslatable: **unmapped divergence**, see below) |

## Counts

- TS tests found: 21 (`agent-loop.test.ts`) + 20 (`agent.test.ts`) = **41**
- Ported: **38** (3 of them partial: see notes) + 1 fixture-derived extra
- Passing: **34**
- Ignored: **5** — patch-3 ×1, patch-4 ×2, patch-6 ×1, patch-7 ×1
- Skipped: **3** — 2× TS-only `setDefaultStreamFn`, 1× patch-5 (`wait_for_idle` API absent)

All five ignored tests were run with `-- --ignored` and fail exactly for the
documented patch reason (none fail for an unrelated cause).

## Translation notes

- **Fixtures.** `MockAssistantStream` + `queueMicrotask(push done)` is ported
  as a stream yielding a single terminal `Done` event (`done_stream`). The TS
  tests never push a `start` event except in the abort mocks, which are ported
  as an async stream yielding `Start`, then awaiting cancellation, then
  yielding `Error` with an `Aborted` message.
- **`setTimeout(() => release(), 20)`** → `release_after(notify, 20)`
  (spawned `tokio::time::sleep` + `Notify`). Deferred promises → `Notify`;
  shared flags → atomics.
- **`handles_tool_calls_and_results` (patch-9 partial).** Upstream asserts:
  `tool.execute` returns `usage`, `afterToolCall` observes `result.usage`
  (equal to the tool's reading) and replaces it via `{ usage: patched }`, and
  the persisted toolResult message carries the patched usage. Rust
  `AgentToolResult`, `AfterToolCallResult`, and `ToolResultMessage` have no
  `usage` field (patch-9 type drift: `AgentToolResult.usage` merge), so those
  assertions cannot be written today. The port keeps the hook in the loop and
  asserts it observed the executed result; WP4's patch-9 must restore the
  usage plumbing and extend this test with the dropped assertions.
- **`executes_mutated_before_tool_call_args_without_revalidation`
  (patch-3).** Upstream mutates the shared `args` object inside
  `beforeToolCall`. Rust hooks receive a cloned `serde_json::Value` and
  `BeforeToolCallResult` has no args-override channel, so the rewrite is
  structurally impossible today. The ported test performs the closest
  expression (mutating its copy) and pins the observable semantic
  (`execute` sees `123`). WP4's patch-3 will need an explicit args-override
  channel; update the hook body to use it, keep the assertion.
- **`user_message_string_content_wire_format` (patch-7).** Not a named TS
  test: it pins the upstream `createUserMessage` fixture shape
  (`content: "<string>"`), which every upstream loop test runs on. The typed
  Rust ports necessarily use the structured single-text-block form, so this
  serde vector carries the string-content semantic explicitly.
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

## Unmapped divergences (not in the patch-1…10 ledger)

- **`Agent.sessionId` setter missing.** Upstream `agent.sessionId =
  "session-def"` re-targets subsequent prompts mid-life and the test asserts
  the next stream call receives the new id. The Rust `Agent` stores
  `session_id` privately at construction with no setter, so the second half of
  "forwards sessionId to streamFunction options" cannot be ported. This does
  not map to any listed patch (patch-8 covers `on_payload` / `on_response` /
  `thinking_budgets`, not the session-id setter) — flagging as a candidate
  addition to the WP4 ledger.

## Patches with no covering upstream test

For the record, these WP4 ledger entries have **no** vector in the two ported
files (upstream does not test them there): patch-1 (schema
validation/coercion no-op — only indirectly touched via
`prepares_tool_arguments_for_validation`, which passes either way), patch-2
(hook throw → isError containment), patch-8 (`on_payload` / `on_response` /
`thinking_budgets` plumbing), patch-10 (session entry kinds — lives in
pi-coding-agent, out of scope for `packages/agent`), and most of patch-9
(StopReason::Pending, `cache_write_1h` / `reasoning` usage fields,
ThinkingLevel::Max). WP4 should add fresh vectors for those alongside the
fixes.
