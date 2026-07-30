//! Shared scalar conversions used by both directions.

use grain_agent_core::Usage;

/// Project genai's nullable token counters into grain's flat `Usage` struct.
///
/// Mapped from genai 0.6.5's `chat::Usage`:
/// - `prompt_tokens` → `input` (total prompt tokens, cached + uncached —
///   grain's documented convention, see `Cost::cost_for`),
/// - `completion_tokens` → `output`,
/// - `total_tokens` → `total_tokens`,
/// - `prompt_tokens_details.cached_tokens` → `cache_read`,
/// - `prompt_tokens_details.cache_creation_tokens` → `cache_write`,
/// - `completion_tokens_details.reasoning_tokens` → `reasoning`
///   (WP5 AB-R2, closed after the WP4 rebase gave `Usage` the field:
///   `Option` maps through directly — grain documents `None` as "no
///   reasoning breakdown reported". Note genai's `zero_as_none`
///   deserialization turns a wire `reasoning_tokens: 0` into `None`
///   before it reaches this seam, so upstream's `Some(0)` for
///   openai-completions/google is not reproducible — a wire zero and an
///   absent field are indistinguishable here).
///
/// Left at defaults (no grain-side counterpart today):
/// - `Usage::cost` — computed downstream by the loop from the model's
///   pricing table (`Cost::cost_for`), never by the adapter;
/// - `Usage::cache_write_1h` — genai exposes the Anthropic 1h split via
///   `prompt_tokens_details.cache_creation_details.ephemeral_1h_tokens`;
///   wiring it is follow-up work, no seam vector exercises it;
/// - genai's audio-token details — grain's `Usage` has no matching fields;
/// - any scalar counter genai reports as `None` maps to 0.
///
/// **Total fallback (WP5, adapter-bug AB-2, seam vectors OV-4/OV-5)**: some
/// providers omit `total_tokens` from streaming usage frames, and genai's
/// OpenAI streamer passes the field through as `None` rather than computing
/// it (verified against genai 0.6.5 `adapter/adapters/openai/streamer.rs`,
/// which — unlike the Anthropic streamer's `message_stop` handler — never
/// derives a total). Upstream pi-ai always computes
/// `totalTokens = input + output + cacheRead + cacheWrite`
/// (`packages/ai/src/api/openai-completions.ts` `parseChunkUsage`,
/// `anthropic-messages.ts` usage accumulation). Mirror that here whenever
/// genai reports no usable total; a provider-supplied total still wins.
///
/// The fallback is `input + output + cache_write` — NOT upstream's literal
/// `input + output + cacheRead + cacheWrite` — because grain's `input` is
/// documented as the *total* prompt tokens (cached + uncached, see
/// [`grain_agent_core::Cost::cost_for`]), whereas upstream's `input`
/// excludes the cached share. The two formulas produce the same absolute
/// value under each side's own convention.
pub fn map_usage(g: genai::chat::Usage) -> Usage {
    let input = g.prompt_tokens.unwrap_or(0).max(0) as u64;
    let output = g.completion_tokens.unwrap_or(0).max(0) as u64;
    let reported_total = g.total_tokens.unwrap_or(0).max(0) as u64;

    let (cache_read, cache_write) = g
        .prompt_tokens_details
        .as_ref()
        .map(|d| {
            (
                d.cached_tokens.unwrap_or(0).max(0) as u64,
                d.cache_creation_tokens.unwrap_or(0).max(0) as u64,
            )
        })
        .unwrap_or((0, 0));

    let reasoning = g
        .completion_tokens_details
        .as_ref()
        .and_then(|d| d.reasoning_tokens)
        .map(|r| r.max(0) as u64);

    let total_tokens = if reported_total > 0 {
        reported_total
    } else {
        input + output + cache_write
    };

    Usage {
        input,
        output,
        cache_read,
        cache_write,
        reasoning,
        total_tokens,
        ..Usage::default()
    }
}
