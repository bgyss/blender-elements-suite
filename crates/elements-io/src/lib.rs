#![forbid(unsafe_code)]

//! File output for the Elements Suite: golden arrays, previews, and OpenVDB.

mod npy;
mod preview;
pub mod vdb;

pub use npy::{read_npy, write_npy};
pub use preview::write_slice_png;
pub use vdb::write_float_grid;

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
    #[error("field {dims:?} exceeds the 4096^3 limit of a single root child")]
    FieldTooLarge { dims: [u32; 3] },
    #[error("OpenVDB UUIDs are 36 ASCII characters, got {len}")]
    BadUuid { len: usize },
}
