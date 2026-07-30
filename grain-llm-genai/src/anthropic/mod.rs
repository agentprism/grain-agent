//! Native Anthropic Messages transport — a second [`grain_agent_core::LlmStream`]
//! implementation behind the same seam.
//!
//! # Why this exists
//!
//! Six of the eight structural gaps measured in `tests/SEAM-VECTORS.md` are
//! unreachable from `ChatStreamEvent`: it is the entire streaming seam, and
//! everything else the provider sent is consumed inside genai's streamer with
//! no public carrier. The decisive case is **S-3**, the Anthropic usage
//! double-count: genai adds `message_delta.usage` onto `message_start.usage`
//! where the wire semantics are *replacement*, and the corrective term is
//! destroyed before anything the streaming API exposes — proven in
//! `tests/genai_seam_limits.rs`.
//!
//! Owning the transport is not the *only* conceivable fix — genai lets the
//! caller choose the endpoint, so a recording relay could tee the wire and
//! re-derive usage. It is the only *reasonable* one: teeing means
//! re-implementing Anthropic SSE parsing anyway, at which point keeping genai
//! in the path buys nothing. `SEAM-VECTORS.md` §6 costs that alternative and
//! rejects it explicitly. Because the harness bills
//! token budgets and reports cost from what crosses this seam, the inflation
//! is a user-visible wrong number, not a cosmetic gap.
//!
//! # What this closes
//!
//! | gap | how |
//! |---|---|
//! | S-3 usage double-count | [`state::UsageAccumulator`] replaces per field |
//! | S-4 `stop_details` | explanation captured → terminal `error_message` |
//! | S-8 block boundaries | `content_block_start`/`stop` drive `*_start`/`*_end` |
//! | S-5 frame repair | frames parsed via [`crate::mapping::json_repair`] |
//!
//! S-2 (`responseId`) and S-6 (`responseModel`) are *captured* here — the wire
//! carries both — but `grain_agent_core::AssistantMessage` has no slot for
//! either, so they stop at this boundary. Adding those core fields is unowned
//! work, flagged in the WP25 report.
//!
//! # Scope and safety
//!
//! - **Opt-in.** genai remains the default backend. Select this one explicitly
//!   via [`crate::GenaiStreamBuilder::with_native_anthropic_transport`].
//! - **Narrow by design.** [`request`] builds only the request shape the
//!   harness actually emits and rejects anything else *loudly* at request time
//!   ([`request::UnsupportedFeature`]) rather than silently sending a
//!   different request than the caller asked for.
//! - **Fixture-verified, not live-verified.** Every assertion is against
//!   upstream pi-ai's recorded fixtures replayed from a local socket. That
//!   proves decode parity; it does not prove the live API accepts our request
//!   shape. See [`stream`] for the full caveat.

pub mod request;
pub mod state;
pub mod stream;
pub mod wire;

pub use request::{SUPPORTED_SURFACE, UnsupportedFeature, build_request};
pub use state::AnthropicState;
pub use stream::{AnthropicAuth, AnthropicStream, AnthropicTransportConfig};
pub use wire::{SseDecoder, SseFrame};
