#![forbid(unsafe_code)]

//! Core runtime for the Elements Suite: GPU context, fields, and the node graph.

/// The version of this crate, surfaced for protocol handshakes.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod gpu;
pub mod graph;
pub mod nodes;
