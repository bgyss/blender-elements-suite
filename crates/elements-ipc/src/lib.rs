#![deny(unsafe_op_in_unsafe_fn)]

//! Inter-process transport between the Elements engine and its clients.
//!
//! Two planes: a newline-delimited JSON control plane (`protocol`) and a
//! memory-mapped double-buffered data plane (`channel`, added in Task 15).

mod channel;
mod protocol;
mod transport;

pub use channel::{
    CHANNEL_HEADER_BYTES, CHANNEL_MAGIC, CHANNEL_VERSION, ChannelError, ChannelHeader, FrameReader,
    FrameWriter,
};
pub use protocol::{
    Command, ELEMENTS_PROTOCOL_VERSION, EngineError, ErrorKind, MAX_MESSAGE_BYTES, ProtocolError,
    Response, read_message, write_message,
};
pub use transport::{Listener, Stream, default_endpoint};
