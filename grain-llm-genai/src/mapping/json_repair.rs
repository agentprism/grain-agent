//! Lenient JSON parsing for streamed tool-call arguments — the half of
//! structural gap **S-5** that is reachable from above genai.
//!
//! Upstream pi-ai never lets a malformed tool-argument buffer kill a turn:
//! it repairs the two malformations providers actually emit and parses the
//! result (`packages/ai/src/utils/json-parse.ts` at pin `34239180`). grain
//! could not, because [`crate::mapping::inbound::parse_tool_args`] used a
//! bare `serde_json::from_str` and fell back to `Value::String(raw)` — and
//! a `String`-shaped argument value is exactly what the outbound layer
//! treats as corrupt and **drops the whole tool call for**
//! (`mapping::outbound::collect_corrupt_tool_call_ids`). A single stray
//! escape anywhere in a tool's arguments therefore cost the user the entire
//! tool call, silently.
//!
//! genai *does* let the raw accumulated buffer cross the seam
//! (`ChatStreamEvent::ToolCallChunk` carries `fn_arguments` as
//! `Value::String` for both the OpenAI and Anthropic streamers, and genai's
//! OpenAI streamer keeps the string verbatim when its own parse fails —
//! `adapter/adapters/openai/streamer.rs`), so this repair is applicable at
//! our boundary without any genai change. That is what separates this from
//! the rest of S-5: a malformed *SSE frame* aborts the stream inside genai
//! and never reaches us (see `tests/SEAM-VECTORS.md` §6).
//!
//! ## Scope: what is repaired, and what deliberately is not
//!
//! [`repair_json`] is a faithful port of upstream's `repairJson`. It is a
//! narrow **string-literal** repair, not a general JSON fixer — it escapes
//! raw control characters inside strings and doubles backslashes before
//! invalid escapes. It does not balance brackets, insert missing quotes or
//! commas, strip trailing commas, or strip comments.
//!
//! Upstream layers a *truncation*-tolerant tier on top (`parseStreamingJson`
//! falls through to the `partial-json` package, yielding a partial object
//! for a cut-off buffer). **That tier is intentionally not ported here.**
//! Adopting it would change grain's terminal behavior from "drop a tool call
//! whose arguments never finished streaming" to "execute it with silently
//! truncated arguments" — e.g. a file-write whose `content` lost its tail.
//! No seam vector at the pin forces that, so the conservative behavior is
//! kept and the divergence is recorded in `SEAM-VECTORS.md` §5 and in the
//! WP21 deferred set rather than taken silently.

use serde_json::Value;

/// Escapes JSON treats as valid after a backslash.
const VALID_JSON_ESCAPES: [char; 9] = ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'];

fn is_control_character(c: char) -> bool {
    (c as u32) <= 0x1f
}

fn escape_control_character(c: char) -> String {
    match c {
        '\u{8}' => "\\b".to_string(),
        '\u{c}' => "\\f".to_string(),
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\t' => "\\t".to_string(),
        other => format!("\\u{:04x}", other as u32),
    }
}

/// Repair malformed JSON **string literals**, mirroring upstream pi-ai's
/// `repairJson` (`packages/ai/src/utils/json-parse.ts:39-83`) statement for
/// statement:
///
/// - raw control characters (`U+0000`–`U+001F`) inside a string are escaped
///   — `\b`/`\f`/`\n`/`\r`/`\t` by name, everything else as `\u00xx`;
/// - a backslash before an *invalid* escape character is doubled, so `\H`
///   becomes `\\H` and parses back to the two characters `\H`;
/// - a dangling backslash at end of input is doubled;
/// - `\uXXXX` with four hex digits is preserved verbatim.
///
/// Faithfully reproduced quirk: when `\u` is **not** followed by four hex
/// digits, upstream's hex branch declines and control falls through to the
/// valid-escape table — which contains `u` — so the sequence is emitted
/// unchanged and stays malformed. Upstream does not repair it and neither do
/// we; `parse_json_with_repair` simply reports the original parse error for
/// such input.
pub fn repair_json(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut repaired = String::with_capacity(input.len());
    let mut in_string = false;
    let mut index = 0usize;

    while index < chars.len() {
        let c = chars[index];

        if !in_string {
            repaired.push(c);
            if c == '"' {
                in_string = true;
            }
            index += 1;
            continue;
        }

        if c == '"' {
            repaired.push(c);
            in_string = false;
            index += 1;
            continue;
        }

        if c == '\\' {
            let Some(&next) = chars.get(index + 1) else {
                // Dangling backslash at end of input.
                repaired.push_str("\\\\");
                index += 1;
                continue;
            };

            if next == 'u' {
                let hex: String = chars
                    .get(index + 2..index + 6)
                    .map(|s| s.iter().collect())
                    .unwrap_or_default();
                if hex.chars().count() == 4 && hex.chars().all(|h| h.is_ascii_hexdigit()) {
                    repaired.push_str("\\u");
                    repaired.push_str(&hex);
                    index += 6;
                    continue;
                }
                // Fall through: `u` is itself a valid escape char, so the
                // next branch emits `\u` unchanged (upstream parity).
            }

            if VALID_JSON_ESCAPES.contains(&next) {
                repaired.push('\\');
                repaired.push(next);
                index += 2;
                continue;
            }

            // Invalid escape: double the backslash and re-process `next` as
            // an ordinary character on the following iteration.
            repaired.push_str("\\\\");
            index += 1;
            continue;
        }

        if is_control_character(c) {
            repaired.push_str(&escape_control_character(c));
        } else {
            repaired.push(c);
        }
        index += 1;
    }

    repaired
}

/// Strict parse, then exactly one repair attempt — upstream's
/// `parseJsonWithRepair` (`json-parse.ts:85-95`).
///
/// When the repair is a no-op the **original** parse error is returned, so
/// callers never see a misleading second-order error.
pub fn parse_json_with_repair(input: &str) -> Result<Value, serde_json::Error> {
    match serde_json::from_str::<Value>(input) {
        Ok(value) => Ok(value),
        Err(original) => {
            let repaired = repair_json(input);
            if repaired != input {
                serde_json::from_str::<Value>(&repaired)
            } else {
                Err(original)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- The upstream vector -------------------------------------------------

    /// The exact payload from upstream's
    /// `anthropic-sse-parsing.test.ts` "repairs malformed SSE JSON and
    /// malformed streamed tool JSON" case (seam vector AV-1), inner layer:
    /// an invalid `\H` escape plus a raw TAB inside a string literal.
    #[test]
    fn repairs_the_upstream_av1_tool_argument_payload() {
        let raw = "{\"path\":\"A\\H\",\"text\":\"col1\tcol2\"}";
        let parsed = parse_json_with_repair(raw).expect("repaired payload must parse");
        assert_eq!(
            parsed,
            json!({"path": "A\\H", "text": "col1\tcol2"}),
            "upstream expects path to be the three characters A\\H and text \
             to contain a real tab"
        );
    }

    // -- repair_json unit behavior ------------------------------------------

    #[test]
    fn doubles_backslash_before_an_invalid_escape() {
        assert_eq!(repair_json(r#"{"a":"A\H"}"#), r#"{"a":"A\\H"}"#);
    }

    #[test]
    fn preserves_valid_escapes_verbatim() {
        let valid = r#"{"a":"q\"q\\q\/q\bq\fq\nq\rq\tq"}"#;
        assert_eq!(repair_json(valid), valid);
        assert!(serde_json::from_str::<Value>(valid).is_ok());
    }

    #[test]
    fn preserves_well_formed_unicode_escapes() {
        let valid = r#"{"a":"é\uD83D"}"#;
        assert_eq!(repair_json(valid), valid);
    }

    #[test]
    fn escapes_raw_control_characters_inside_strings() {
        // Named escapes for the five JSON has spellings for.
        assert_eq!(repair_json("{\"a\":\"x\ty\"}"), r#"{"a":"x\ty"}"#);
        assert_eq!(repair_json("{\"a\":\"x\ny\"}"), r#"{"a":"x\ny"}"#);
        assert_eq!(repair_json("{\"a\":\"x\ry\"}"), r#"{"a":"x\ry"}"#);
        assert_eq!(repair_json("{\"a\":\"x\u{8}y\"}"), r#"{"a":"x\by"}"#);
        assert_eq!(repair_json("{\"a\":\"x\u{c}y\"}"), r#"{"a":"x\fy"}"#);
        // Everything else in the control range as \u00xx, lower-case hex.
        assert_eq!(repair_json("{\"a\":\"x\u{1}y\"}"), r#"{"a":"x\u0001y"}"#);
        assert_eq!(repair_json("{\"a\":\"x\u{1f}y\"}"), r#"{"a":"x\u001fy"}"#);
    }

    #[test]
    fn leaves_control_characters_outside_strings_alone() {
        // Whitespace between tokens is legal JSON and must not be rewritten.
        let with_newline = "{\n\"a\": 1\n}";
        assert_eq!(repair_json(with_newline), with_newline);
    }

    #[test]
    fn doubles_a_dangling_trailing_backslash() {
        assert_eq!(repair_json(r#"{"a":"x\"#), r#"{"a":"x\\"#);
    }

    #[test]
    fn tracks_string_boundaries_across_escaped_quotes() {
        // The escaped quote must not be mistaken for the string terminator,
        // so the tab that follows is still "inside a string" and gets escaped.
        assert_eq!(
            repair_json("{\"a\":\"say \\\"hi\\\"\tnow\"}"),
            r#"{"a":"say \"hi\"\tnow"}"#
        );
    }

    #[test]
    fn is_idempotent_on_already_valid_json() {
        for valid in [
            r#"{"path":"src/main.rs","line":42}"#,
            r#"{"nested":{"a":[1,2,{"b":null}]},"t":true}"#,
            r#"[]"#,
            r#"{}"#,
            r#""plain string""#,
        ] {
            assert_eq!(repair_json(valid), valid, "unchanged: {valid}");
            assert_eq!(repair_json(&repair_json(valid)), repair_json(valid));
        }
    }

    #[test]
    fn repairing_twice_is_stable_for_malformed_input() {
        let once = repair_json(r#"{"a":"A\H"}"#);
        assert_eq!(repair_json(&once), once, "repair must reach a fixed point");
    }

    // -- parse_json_with_repair ---------------------------------------------

    #[test]
    fn valid_json_parses_without_repair() {
        let parsed = parse_json_with_repair(r#"{"a":1}"#).expect("valid");
        assert_eq!(parsed, json!({"a": 1}));
    }

    #[test]
    fn reports_the_original_error_when_repair_is_a_no_op() {
        // Truncated input: nothing for the string-literal repair to change,
        // so the caller sees the real "unexpected end of input" error rather
        // than a confusing second-order one.
        let err = parse_json_with_repair(r#"{"a":"#).expect_err("must not parse");
        assert!(
            err.to_string().contains("EOF") || err.to_string().contains("end of input"),
            "expected the original EOF error, got: {err}"
        );
    }

    #[test]
    fn truncation_is_not_repaired() {
        // Upstream's `partial-json` tier is deliberately not ported; a
        // truncated buffer must stay an error so the caller keeps its
        // conservative fallback. See the module docs.
        assert!(parse_json_with_repair(r#"{"path":"src/ma"#).is_err());
        assert!(parse_json_with_repair(r#"{"a":[1,2"#).is_err());
    }

    #[test]
    fn unrepairable_unicode_escape_stays_an_error() {
        // `\uZZ` — the hex branch declines, `u` is a valid escape char, so
        // the sequence is emitted unchanged (upstream parity) and the input
        // remains malformed.
        assert!(parse_json_with_repair(r#"{"a":"x\uZZ"}"#).is_err());
    }

    #[test]
    fn empty_and_whitespace_input_stay_errors() {
        assert!(parse_json_with_repair("").is_err());
        assert!(parse_json_with_repair("   ").is_err());
    }
}
