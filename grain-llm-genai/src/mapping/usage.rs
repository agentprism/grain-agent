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
/// - `prompt_tokens_details.cache_creation_tokens` → `cache_write`.
///
/// Left at defaults (no grain-side counterpart today):
/// - `Usage::cost` — computed downstream by the loop from the model's
///   pricing table (`Cost::cost_for`), never by the adapter;
/// - genai's `completion_tokens_details` (reasoning/audio breakdown) and
///   `prompt_tokens_details.audio_tokens` — grain's `Usage` has no
///   matching fields;
/// - any counter genai reports as `None` maps to 0.
pub fn map_usage(g: genai::chat::Usage) -> Usage {
    let input = g.prompt_tokens.unwrap_or(0).max(0) as u64;
    let output = g.completion_tokens.unwrap_or(0).max(0) as u64;
    let total = g.total_tokens.unwrap_or(0).max(0) as u64;

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

    Usage {
        input,
        output,
        cache_read,
        cache_write,
        total_tokens: total,
        ..Usage::default()
    }
}
