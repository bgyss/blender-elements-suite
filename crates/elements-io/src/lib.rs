#![forbid(unsafe_code)]

//! File output for the Elements Suite: golden arrays, previews, and OpenVDB.

mod npy;
mod preview;

pub use npy::{read_npy, write_npy};
pub use preview::write_slice_png;

/// Everything that can go wrong writing an Elements output file.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("npy error: {0}")]
    Npy(String),
    #[error("png error: {0}")]
    Png(String),
    #[error("slice z={z} is out of bounds for a field of depth {depth}")]
    BadSlice { z: u32, depth: u32 },
    #[error("expected {expected} values, got {got}")]
    LengthMismatch { expected: usize, got: usize },
}
