//! `grain-ai-agent-tui` — ratatui-based terminal UI on top of
//! `grain-ai-agent-headless`. Same agent capabilities (file tools,
//! shell, web fetch, semantic search, session persistence, skills,
//! slash commands) wrapped in a multi-pane terminal interface.
//!
//! Architecture:
//!
//! - [`app::AppState`] is the pure UI state — what to display, what's
//!   focused, what's in the input line. No I/O, fully unit-testable.
//! - [`event::TuiEvent`] enumerates everything the main loop can react
//!   to: a key press, a terminal resize, an [`grain_agent_core::AgentEvent`]
//!   from the running Agent, a periodic tick.
//! - [`agent_worker`] owns the actual `Agent` on a dedicated tokio task
//!   and shuttles events to / commands from the UI via `mpsc` channels.
//! - [`ui`] renders [`AppState`] into a ratatui `Frame`.
//! - [`run::run_tui`] ties the terminal lifecycle, event polling, and
//!   render loop together. `src/bin/grain_tui.rs` is a tiny entry point
//!   that calls into it.

// Lint philosophy: be strict about correctness, warn on common mistakes,
// and let pedantic lints be opt-in so they don't turn into noise.
#![deny(clippy::correctness)]
#![warn(clippy::suspicious)]
#![warn(clippy::style)]
#![warn(clippy::complexity)]
#![warn(clippy::perf)]
#![warn(clippy::undocumented_unsafe_blocks)]
// missing_docs is noisy for a binary-first crate with many internal
// modules; enable selectively on the public API surface once that
// surface stabilises.
// #![warn(missing_docs)]
//
// ---------------------------------------------------------------------------
// Pre-existing clippy debt (DEBT.md G33), scoped so CI can run `-D warnings`.
//
// CI lints the workspace with `cargo clippy --workspace --all-targets --
// -D warnings`. This crate carries 7 findings that predate that job. Rather
// than weaken the gate to report-only — which would never catch anything —
// each lint is allowed HERE and ONLY here, so:
//   - any OTHER lint in this crate still fails CI;
//   - these same lints still fail CI in every other crate;
//   - closing G33 means deleting the three attributes below, nothing else.
//
// These are crate-root attributes rather than Cargo.toml `[lints]` on
// purpose: manifest lint levels are emitted BEFORE the trailing `-D warnings`
// on clippy's command line, so rustc's last-flag-wins ordering lets
// `-D warnings` override them. Source attributes are scoped inside the crate
// and take precedence over command-line levels, so they actually hold.
//
// Owner: WP19 grain-product backlog (G33). Do not add entries here to
// silence NEW findings — fix those instead.
// ---------------------------------------------------------------------------
// src/agent_worker.rs:870 — `.filter(..).next()` should be `.find(..)`.
#![allow(clippy::filter_next)]
// src/app.rs:2047, :2052, :3204 — `if` collapsible into the outer `match`.
#![allow(clippy::collapsible_match)]
// src/run.rs:323 (8/7), src/ui.rs:649 (9/7), src/ui.rs:3249 (8/7) — render
// and run entrypoints that thread widget/layout state positionally.
#![allow(clippy::too_many_arguments)]
// src/ui.rs:853 — `if let ... else { return None }` that reads as `?`.
// Same pre-existing code as the rest of this list, but only surfaced by
// clippy 1.97; 1.96 did not flag it. That version sensitivity is why the
// CI clippy job pins its toolchain — see .github/workflows/ci.yml.
#![allow(clippy::question_mark)]
pub mod agent_worker;
pub mod anim;
pub mod app;
pub mod cli;
pub mod config_apply;
pub mod event;
pub mod md_render;
pub mod persist;
pub mod run;
pub mod theme;
pub mod ui;

pub use app::{AppState, Focus, Overlay, TranscriptKind, TranscriptLine};
pub use cli::Args;
pub use event::TuiEvent;
// Provider profile types live in `grain-llm-genai` — the natural home,
// since that's where the genai service-target resolver they plug into
// also lives. Re-exported here for convenience to TUI callers.
pub use grain_llm_genai::{
    ProviderAuth, ProviderKind, ProviderProfile, load_profiles, resolve_providers_file,
};
pub use run::{TuiError, run_tui};
pub use theme::{Palette, Theme, ThemeSource, builtin_themes, load_user_themes};
