//! SSE framing for the Anthropic Messages stream.
//!
//! Split out from the state machine so the framing rules are testable without
//! any HTTP: feed bytes, get frames.
//!
//! # The decoder is byte-oriented, and that is load-bearing
//!
//! A `bytes_stream()` chunk is a TCP-shaped slice of the response body, not a
//! logical unit. It can split *anything*, including the middle of a multi-byte
//! UTF-8 scalar and the middle of a `\r\n` pair. The decoder therefore owns
//! two pieces of cross-chunk state, and both are correctness-critical:
//!
//! - **`pending_bytes`** — the tail of a chunk that is a *prefix* of a valid
//!   UTF-8 sequence. Decoding each chunk independently (e.g.
//!   `String::from_utf8_lossy` per chunk) turns every scalar that straddles a
//!   chunk boundary into replacement characters, silently: an em-dash split
//!   after its first byte becomes `���`. Claude emits em-dashes, smart quotes,
//!   CJK and emoji constantly and TCP boundaries are arbitrary, so this is a
//!   matter of when, not if. It is also silent — and a corrupted `thinking`
//!   block replayed with the signature Anthropic computed over the *original*
//!   text is rejected on the following turn, so the damage surfaces a turn
//!   away from its cause.
//!
//!   Upstream pi-ai has the same requirement and meets it the same way: one
//!   long-lived `TextDecoder` with `{stream: true}` plus a final flush
//!   (`anthropic-messages.ts:407`).
//!
//! - **`pending_cr`** — whether the last character seen was a `\r` that may
//!   still be joined by a `\n`. Normalizing per chunk (`replace("\r\n", "\n")`
//!   on each chunk in isolation) cannot see a CR ending one chunk and an LF
//!   starting the next, so it leaves `\n\n` where the wire meant one newline —
//!   fabricating a frame separator that splits one event into two halves,
//!   neither of which parses.
//!
//! Both are exercised at **every possible split offset** by
//! `tests/chunk_boundary.rs`, which is permanent infrastructure rather than a
//! one-off regression test: any future decoder change is re-fuzzed
//! automatically, over the real fixtures, through the real socket path.
//!
//! # Framing rules
//!
//! - **CR / CRLF / LF all normalize to LF**, then frames split on a blank
//!   line. The SSE spec allows `\n\n`, `\r\n\r\n` and `\r\r` as the separator.
//! - **Flush the final frame at EOF.** Anthropic's stream does not end with a
//!   trailing blank line, so the terminal `message_stop` only appears if the
//!   decoder flushes what is left when the body ends. A decoder that requires
//!   a trailing separator silently loses the terminal event.

/// One decoded SSE frame: the `event:` name and the joined `data:` payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseFrame {
    pub event: String,
    pub data: String,
}

/// Incremental, chunk-boundary-safe SSE frame decoder.
#[derive(Debug, Default)]
pub struct SseDecoder {
    /// Bytes that form a *prefix* of a UTF-8 scalar and need more input.
    pending_bytes: Vec<u8>,
    /// Newline-normalized text not yet forming a complete frame.
    buf: String,
    /// The last character was `\r`; a `\n` immediately after it belongs to
    /// that CRLF pair and must not produce a second newline.
    pending_cr: bool,
}

impl SseDecoder {
    pub fn new() -> Self {
        SseDecoder::default()
    }

    /// Feed a chunk of the response body; returns every frame that completed.
    ///
    /// This is the entry point the transport uses. Bytes that cannot yet be
    /// decoded (an incomplete trailing UTF-8 sequence) are retained for the
    /// next call rather than being lossily replaced.
    pub fn push_bytes(&mut self, chunk: &[u8]) -> Vec<SseFrame> {
        self.pending_bytes.extend_from_slice(chunk);
        let decoded = self.decode_available();
        self.absorb(&decoded);
        self.take_frames()
    }

    /// Convenience for callers that already hold valid UTF-8 (tests).
    pub fn push(&mut self, chunk: &str) -> Vec<SseFrame> {
        self.push_bytes(chunk.as_bytes())
    }

    /// Flush whatever remains when the body ends.
    ///
    /// Required for Anthropic: the stream's last frame (`message_stop`) is not
    /// followed by a blank line. Any still-undecodable trailing bytes are a
    /// genuinely truncated scalar at end of stream and become one replacement
    /// character — the only point at which lossy decoding is the right answer.
    pub fn finish(&mut self) -> Option<SseFrame> {
        if !self.pending_bytes.is_empty() {
            self.pending_bytes.clear();
            self.absorb("\u{FFFD}");
        }
        let block = std::mem::take(&mut self.buf);
        self.pending_cr = false;
        parse_block(&block)
    }

    /// Decode every complete UTF-8 scalar currently buffered, retaining an
    /// incomplete trailing sequence for the next chunk.
    fn decode_available(&mut self) -> String {
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending_bytes) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending_bytes.clear();
                    break;
                }
                Err(err) => {
                    let valid = err.valid_up_to();
                    if valid > 0 {
                        match std::str::from_utf8(&self.pending_bytes[..valid]) {
                            Ok(s) => out.push_str(s),
                            // `valid_up_to` guarantees this prefix decodes.
                            Err(_) => break,
                        }
                    }
                    match err.error_len() {
                        // Incomplete trailing sequence: keep it, await more.
                        None => {
                            self.pending_bytes.drain(..valid);
                            break;
                        }
                        // Genuinely malformed: emit U+FFFD, skip, keep going.
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending_bytes.drain(..valid + bad);
                        }
                    }
                }
            }
        }
        out
    }

    /// Append text to the frame buffer, normalizing CR / CRLF / LF to LF
    /// across chunk boundaries.
    fn absorb(&mut self, text: &str) {
        for ch in text.chars() {
            if self.pending_cr {
                self.pending_cr = false;
                if ch == '\n' {
                    // CRLF: the CR already emitted the newline.
                    continue;
                }
            }
            if ch == '\r' {
                self.pending_cr = true;
                self.buf.push('\n');
                continue;
            }
            self.buf.push(ch);
        }
    }

    fn take_frames(&mut self) -> Vec<SseFrame> {
        let mut frames = Vec::new();
        while let Some(pos) = self.buf.find("\n\n") {
            let block: String = self.buf[..pos].to_string();
            self.buf.drain(..pos + 2);
            if let Some(frame) = parse_block(&block) {
                frames.push(frame);
            }
        }
        frames
    }
}

fn parse_block(block: &str) -> Option<SseFrame> {
    let mut event = String::from("message");
    let mut data = String::new();
    let mut saw_data = false;

    for line in block.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(e) = line.strip_prefix("event:") {
            event = e.trim().to_string();
        } else if let Some(d) = line.strip_prefix("data:") {
            if saw_data {
                data.push('\n');
            }
            data.push_str(d.trim());
            saw_data = true;
        }
    }

    if !saw_data {
        return None;
    }
    Some(SseFrame { event, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(body: &str) -> Vec<SseFrame> {
        let mut d = SseDecoder::new();
        let mut frames = d.push(body);
        frames.extend(d.finish());
        frames
    }

    /// Feed `body` one byte at a time — the worst case for cross-chunk state.
    fn byte_at_a_time(body: &str) -> Vec<SseFrame> {
        let mut d = SseDecoder::new();
        let mut frames = Vec::new();
        for b in body.as_bytes() {
            frames.extend(d.push_bytes(&[*b]));
        }
        frames.extend(d.finish());
        frames
    }

    #[test]
    fn decodes_event_and_data_pairs() {
        let frames = all("event: message_start\ndata: {\"a\":1}\n\nevent: ping\ndata: {}\n\n");
        assert_eq!(
            frames,
            vec![
                SseFrame {
                    event: "message_start".into(),
                    data: "{\"a\":1}".into()
                },
                SseFrame {
                    event: "ping".into(),
                    data: "{}".into()
                },
            ]
        );
    }

    /// The Anthropic shape: no trailing blank line after the last frame.
    #[test]
    fn flushes_the_final_frame_without_a_trailing_blank_line() {
        let frames = all("event: message_stop\ndata: {\"type\":\"message_stop\"}\n");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event, "message_stop");
    }

    #[test]
    fn handles_crlf_and_bare_cr_separators() {
        for body in [
            "event: a\r\ndata: 1\r\n\r\nevent: b\r\ndata: 2\r\n\r\n",
            "event: a\rdata: 1\r\revent: b\rdata: 2\r\r",
        ] {
            let frames = all(body);
            assert_eq!(frames.len(), 2, "body: {body:?}");
            assert_eq!(frames[0].data, "1");
            assert_eq!(frames[1].data, "2");
        }
    }

    #[test]
    fn joins_multiple_data_lines_with_newlines() {
        let frames = all("event: x\ndata: one\ndata: two\n\n");
        assert_eq!(frames[0].data, "one\ntwo");
    }

    #[test]
    fn skips_comments_and_frames_without_data() {
        let frames = all(": keep-alive comment\n\nevent: only_event\n\nevent: x\ndata: 1\n\n");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event, "x");
    }

    #[test]
    fn reassembles_frames_split_across_chunk_boundaries() {
        let mut d = SseDecoder::new();
        assert!(d.push("event: message_st").is_empty());
        assert!(d.push("art\ndata: {\"a\"").is_empty());
        let frames = d.push(":1}\n\n");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event, "message_start");
        assert_eq!(frames[0].data, "{\"a\":1}");
    }

    #[test]
    fn defaults_the_event_name_when_only_data_is_present() {
        let frames = all("data: 1\n\n");
        assert_eq!(frames[0].event, "message");
    }

    // -- W1: multi-byte scalars split across chunks --------------------------

    /// The exact defect: an em-dash (`E2 80 94`) split after its first byte.
    /// Decoding chunks independently yields replacement characters.
    #[test]
    fn multibyte_scalar_split_across_chunks_survives() {
        let body = "event: x\ndata: Cost: 12 — done\n\n";
        let bytes = body.as_bytes();
        let dash = body.find('—').expect("em-dash present");

        let mut d = SseDecoder::new();
        let mut frames = d.push_bytes(&bytes[..dash + 1]); // split mid-scalar
        frames.extend(d.push_bytes(&bytes[dash + 1..]));
        frames.extend(d.finish());

        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0].data, "Cost: 12 — done",
            "an em-dash split across chunks must not become replacement chars"
        );
    }

    #[test]
    fn every_split_offset_of_a_multibyte_body_is_lossless() {
        let body = "event: x\ndata: — “quoted” 日本語 🎉 done\n\n";
        let bytes = body.as_bytes();
        let expected = all(body);
        assert_eq!(expected.len(), 1);

        for split in 0..=bytes.len() {
            let mut d = SseDecoder::new();
            let mut frames = d.push_bytes(&bytes[..split]);
            frames.extend(d.push_bytes(&bytes[split..]));
            frames.extend(d.finish());
            assert_eq!(frames, expected, "split at byte {split} changed the frame");
        }
    }

    #[test]
    fn byte_at_a_time_delivery_is_lossless() {
        let body = "event: a\ndata: — 日本語 🎉\n\nevent: b\ndata: ok\n";
        assert_eq!(byte_at_a_time(body), all(body));
    }

    // -- W2: CRLF split across chunks ---------------------------------------

    /// A `\r` ending one chunk and `\n` starting the next must remain ONE
    /// newline. Normalizing per chunk fabricates a `\n\n` separator, which
    /// splits a frame in half and destroys it.
    #[test]
    fn crlf_split_across_chunks_does_not_fabricate_a_separator() {
        let body = "event: x\r\ndata: {\"input\":7}\r\n\r\n";
        let bytes = body.as_bytes();
        let cr = body.find("\r\ndata").expect("CRLF present");

        let mut d = SseDecoder::new();
        let mut frames = d.push_bytes(&bytes[..cr + 1]); // ends on the CR
        frames.extend(d.push_bytes(&bytes[cr + 1..])); // starts on the LF
        frames.extend(d.finish());

        assert_eq!(
            frames.len(),
            1,
            "a CRLF straddling a chunk boundary must not split the frame"
        );
        assert_eq!(frames[0].event, "x");
        assert_eq!(frames[0].data, "{\"input\":7}");
    }

    #[test]
    fn every_split_offset_of_a_crlf_body_is_stable() {
        let body = "event: a\r\ndata: 1\r\n\r\nevent: b\r\ndata: 2\r\n\r\n";
        let bytes = body.as_bytes();
        let expected = all(body);
        assert_eq!(expected.len(), 2);

        for split in 0..=bytes.len() {
            let mut d = SseDecoder::new();
            let mut frames = d.push_bytes(&bytes[..split]);
            frames.extend(d.push_bytes(&bytes[split..]));
            frames.extend(d.finish());
            assert_eq!(frames, expected, "split at byte {split} changed the frames");
        }
    }

    #[test]
    fn lone_cr_still_separates_frames_when_split() {
        let body = "event: a\rdata: 1\r\revent: b\rdata: 2\r\r";
        assert_eq!(byte_at_a_time(body), all(body));
        assert_eq!(all(body).len(), 2);
    }

    // -- lossy decoding is confined to genuinely invalid input ---------------

    #[test]
    fn genuinely_invalid_bytes_become_one_replacement_char() {
        let mut d = SseDecoder::new();
        let mut frames = d.push_bytes(b"event: x\ndata: a");
        frames.extend(d.push_bytes(&[0xFF])); // never valid UTF-8
        frames.extend(d.push_bytes(b"b\n\n"));
        frames.extend(d.finish());
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, "a\u{FFFD}b");
    }

    #[test]
    fn truncated_scalar_at_end_of_stream_flushes_lossily() {
        let mut d = SseDecoder::new();
        let mut frames = d.push_bytes(b"event: x\ndata: ok\n");
        // Body ends mid-scalar: the first two bytes of an em-dash.
        frames.extend(d.push_bytes(&[0xE2, 0x80]));
        frames.extend(d.finish());
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, "ok");
    }

    #[test]
    fn incomplete_trailing_scalar_is_retained_not_replaced() {
        let mut d = SseDecoder::new();
        // Just the lead byte of an em-dash — no frame, no corruption yet.
        assert!(d.push_bytes(b"event: x\ndata: ").is_empty());
        assert!(d.push_bytes(&[0xE2]).is_empty());
        assert!(d.push_bytes(&[0x80]).is_empty());
        let frames = d.push_bytes(&[0x94, b'\n', b'\n']);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, "—");
    }
}
