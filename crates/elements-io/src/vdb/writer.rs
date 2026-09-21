//! Low-level little-endian byte output for the OpenVDB container format.

use std::io::{Seek, SeekFrom, Write};

use crate::IoError;

/// OpenVDB's magic number, written as a little-endian u64.
pub const OPENVDB_MAGIC: u64 = 0x5644_4220;
/// The file format version we emit. `vdb-rs` supports 213 and up.
pub const OPENVDB_FILE_VERSION: u32 = 224;
pub const OPENVDB_LIBRARY_MAJOR: u32 = 12;
pub const OPENVDB_LIBRARY_MINOR: u32 = 0;

/// Only active values are stored in nodes whose metadata byte says so.
pub const COMPRESSION_ACTIVE_MASK: u32 = 0x2;

/// The grid type string for a standard single-precision float tree.
pub const FLOAT_GRID_TYPE: &str = "Tree_float_5_4_3";

/// Node metadata byte: store only the active values.
pub const NO_MASK_OR_INACTIVE_VALS: u8 = 0;
/// Node metadata byte: store every value in the node.
pub const NO_MASK_AND_ALL_VALS: u8 = 6;

/// A seekable little-endian writer with offset patching.
pub struct ByteWriter<W: Write + Seek> {
    inner: W,
}

impl<W: Write + Seek> ByteWriter<W> {
    pub fn new(inner: W) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> W {
        self.inner
    }

    pub fn pos(&mut self) -> Result<u64, IoError> {
        Ok(self.inner.stream_position()?)
    }

    pub fn raw(&mut self, bytes: &[u8]) -> Result<(), IoError> {
        self.inner.write_all(bytes)?;
        Ok(())
    }

    pub fn u8(&mut self, v: u8) -> Result<(), IoError> {
        self.raw(&[v])
    }

    pub fn u32(&mut self, v: u32) -> Result<(), IoError> {
        self.raw(&v.to_le_bytes())
    }

    pub fn i32(&mut self, v: i32) -> Result<(), IoError> {
        self.raw(&v.to_le_bytes())
    }

    pub fn u64(&mut self, v: u64) -> Result<(), IoError> {
        self.raw(&v.to_le_bytes())
    }

    pub fn i64(&mut self, v: i64) -> Result<(), IoError> {
        self.raw(&v.to_le_bytes())
    }

    pub fn f32(&mut self, v: f32) -> Result<(), IoError> {
        self.raw(&v.to_le_bytes())
    }

    pub fn f64(&mut self, v: f64) -> Result<(), IoError> {
        self.raw(&v.to_le_bytes())
    }

    /// A `u32` length followed by the raw bytes, OpenVDB's string encoding.
    pub fn string(&mut self, s: &str) -> Result<(), IoError> {
        self.u32(s.len() as u32)?;
        self.raw(s.as_bytes())
    }

    /// Three little-endian `f64`s.
    pub fn dvec3(&mut self, v: [f64; 3]) -> Result<(), IoError> {
        for c in v {
            self.f64(c)?;
        }
        Ok(())
    }

    /// Overwrite a previously reserved `u64` slot, then return to the end.
    pub fn patch_u64_at(&mut self, offset: u64, value: u64) -> Result<(), IoError> {
        let here = self.pos()?;
        self.inner.seek(SeekFrom::Start(offset))?;
        self.inner.write_all(&value.to_le_bytes())?;
        self.inner.seek(SeekFrom::Start(here))?;
        Ok(())
    }
}

/// A typed metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    String(String),
    Bool(bool),
    I32(i32),
    I64(i64),
    F32(f32),
    Vec3i([i32; 3]),
}

impl MetaValue {
    fn type_name(&self) -> &'static str {
        match self {
            Self::String(_) => "string",
            Self::Bool(_) => "bool",
            Self::I32(_) => "int32",
            Self::I64(_) => "int64",
            Self::F32(_) => "float",
            Self::Vec3i(_) => "vec3i",
        }
    }

    fn byte_len(&self) -> u32 {
        match self {
            Self::String(s) => s.len() as u32,
            Self::Bool(_) => 1,
            Self::I32(_) | Self::F32(_) => 4,
            Self::I64(_) => 8,
            Self::Vec3i(_) => 12,
        }
    }

    /// Write type name, payload length, and payload.
    pub fn write<W: Write + Seek>(&self, w: &mut ByteWriter<W>) -> Result<(), IoError> {
        w.string(self.type_name())?;
        w.u32(self.byte_len())?;
        match self {
            Self::String(s) => w.raw(s.as_bytes()),
            Self::Bool(b) => w.u8(u8::from(*b)),
            Self::I32(v) => w.i32(*v),
            Self::I64(v) => w.i64(*v),
            Self::F32(v) => w.f32(*v),
            Self::Vec3i(v) => {
                for c in v {
                    w.i32(*c)?;
                }
                Ok(())
            }
        }
    }
}

/// Write a metadata map: a count followed by name/type/length/payload records.
pub fn write_metadata<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    entries: &[(&str, MetaValue)],
) -> Result<(), IoError> {
    w.u32(entries.len() as u32)?;
    for (name, value) in entries {
        w.string(name)?;
        value.write(w)?;
    }
    Ok(())
}

/// Write the archive header, up to and including the grid count of 1.
///
/// `uuid` must be exactly 36 ASCII characters; the format stores it unprefixed.
pub fn write_archive_header<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    uuid: &str,
) -> Result<(), IoError> {
    if uuid.len() != 36 {
        return Err(IoError::BadUuid { len: uuid.len() });
    }

    w.u64(OPENVDB_MAGIC)?;
    w.u32(OPENVDB_FILE_VERSION)?;
    w.u32(OPENVDB_LIBRARY_MAJOR)?;
    w.u32(OPENVDB_LIBRARY_MINOR)?;
    w.u8(1)?; // has_grid_offsets
    w.raw(uuid.as_bytes())?;
    write_metadata(w, &[])?; // no file-level metadata
    w.u32(1)?; // grid_count
    Ok(())
}

/// Where the three grid offsets live, so they can be patched once known.
#[derive(Debug, Clone, Copy)]
pub struct GridOffsets {
    pub grid_pos_at: u64,
    pub block_pos_at: u64,
    pub end_pos_at: u64,
}

/// Write a grid descriptor with placeholder offsets.
pub fn write_grid_descriptor<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    name: &str,
) -> Result<GridOffsets, IoError> {
    w.string(name)?;
    w.string(FLOAT_GRID_TYPE)?;
    w.string("")?; // instance_parent: this grid owns its tree

    let grid_pos_at = w.pos()?;
    w.u64(0)?;
    let block_pos_at = w.pos()?;
    w.u64(0)?;
    let end_pos_at = w.pos()?;
    w.u64(0)?;

    Ok(GridOffsets {
        grid_pos_at,
        block_pos_at,
        end_pos_at,
    })
}
