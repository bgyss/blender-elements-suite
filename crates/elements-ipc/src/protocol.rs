//! The control plane: newline-delimited JSON over a stream socket.
//!
//! JSON rather than a binary encoding because the Blender addon must parse it
//! with the Python standard library and no compiled dependency. The bytes that
//! matter travel on the data plane; these messages are small.

use std::io::{BufRead, Read, Write};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// The wire protocol version. Bumped whenever a message changes shape.
pub const ELEMENTS_PROTOCOL_VERSION: u32 = 2;

/// The largest single message accepted, to bound a desynced peer's damage.
pub const MAX_MESSAGE_BYTES: usize = 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("malformed message: {0}")]
    Json(#[from] serde_json::Error),
    /// The line (payload plus trailing newline) exceeded `MAX_MESSAGE_BYTES`.
    ///
    /// The reader stops as soon as the cap is crossed, so the unread remainder
    /// of the oversized line — including its eventual real `\n` — is still
    /// sitting in the stream. The next `read_message` call would resume
    /// mid-line, not at the next message boundary. Callers MUST treat this as
    /// fatal and close the connection rather than continue reading from the
    /// same stream.
    #[error("message of {bytes} bytes exceeds the {MAX_MESSAGE_BYTES} byte limit")]
    OversizedFrame { bytes: usize },
}

/// A classification the addon can branch on without parsing prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    ProtocolVersion,
    Document,
    Graph,
    Gpu,
    DeviceLost,
    /// The GPU ran out of memory; a retry at the same resolution will fail again.
    OutOfMemory,
    Io,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineError {
    pub kind: ErrorKind,
    pub message: String,
}

impl EngineError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

/// Client to daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Command {
    /// Must be the first message. The daemon rejects a version mismatch.
    Hello {
        protocol_version: u32,
    },
    /// Load a `.elements` document by absolute path.
    LoadGraph {
        path: String,
    },
    /// Evaluate the loaded graph and publish the result.
    Render {
        frame: u32,
    },
    Shutdown,
}

/// Daemon to client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    HelloAck {
        protocol_version: u32,
        engine_version: String,
        adapter: String,
    },
    Loaded {
        dims: [u32; 3],
        nodes: u32,
    },
    Frame {
        seq: u64,
        channel: String,
        dims: [u32; 3],
    },
    Bye,
    Error(EngineError),
}

/// Write one compact JSON object followed by a newline.
pub fn write_message<W: Write, T: Serialize>(w: &mut W, msg: &T) -> Result<(), ProtocolError> {
    let mut line = serde_json::to_vec(msg)?;
    debug_assert!(
        !line.contains(&b'\n'),
        "compact JSON must never contain a raw newline"
    );
    line.push(b'\n');
    w.write_all(&line)?;
    w.flush()?;
    Ok(())
}

/// Read one message. Returns `Ok(None)` at a clean end of stream.
pub fn read_message<R: BufRead, T: DeserializeOwned>(
    r: &mut R,
) -> Result<Option<T>, ProtocolError> {
    loop {
        let mut line = Vec::new();
        // Cap the read so a peer that never sends a newline cannot exhaust memory.
        let mut limited: std::io::Take<&mut R> = r.by_ref().take((MAX_MESSAGE_BYTES + 1) as u64);
        let read = limited.read_until(b'\n', &mut line)?;

        if read == 0 {
            return Ok(None);
        }
        if read > MAX_MESSAGE_BYTES {
            return Err(ProtocolError::OversizedFrame { bytes: read });
        }

        let mut trimmed: &[u8] = &line;
        if let Some(rest) = trimmed.strip_suffix(b"\n") {
            trimmed = rest;
        }
        if let Some(rest) = trimmed.strip_suffix(b"\r") {
            trimmed = rest;
        }

        if trimmed.is_empty() {
            continue; // tolerate keepalive blank lines
        }
        // Note: if a peer sends a complete, well-formed JSON object and then
        // closes the connection without a trailing `\n`, `read_until` returns
        // those bytes with no delimiter found. The strip_suffix calls above
        // are then no-ops, `trimmed` still holds the whole object, and it
        // parses successfully here — this function has no way to tell that
        // case apart from a properly newline-terminated message, and does
        // not try to. This is intentional: rejecting it would penalize a
        // peer that behaved correctly except for the final delimiter, and
        // genuinely partial/truncated JSON still fails with `ProtocolError::Json`.
        return Ok(Some(serde_json::from_slice(trimmed)?));
    }
}
