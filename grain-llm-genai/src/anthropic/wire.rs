//! SSE framing for the Anthropic Messages stream.
//!
//! Split out from the state machine so the framing rules are testable without
//! any HTTP: feed bytes, get frames.
//!
//! Two framing details are load-bearing and both mirror upstream pi-ai's
//! parser:
//!
//! - **CR/CRLF normalization then split on a blank line.** The SSE spec allows
//!   `\n\n`, `\r\n\r\n` and `\r\r` as the event separator.
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

/// Incremental SSE frame decoder.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buf: String,
}

impl SseDecoder {
    pub fn new() -> Self {
        SseDecoder::default()
    }

    /// Feed a chunk of the response body; returns every frame that completed.
    pub fn push(&mut self, chunk: &str) -> Vec<SseFrame> {
        self.buf.push_str(&chunk.replace("\r\n", "\n").replace('\r', "\n"));
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

    /// Flush whatever remains when the body ends.
    ///
    /// Required for Anthropic: the stream's last frame (`message_stop`) is not
    /// followed by a blank line.
    pub fn finish(&mut self) -> Option<SseFrame> {
        let block = std::mem::take(&mut self.buf);
        parse_block(&block)
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
}
