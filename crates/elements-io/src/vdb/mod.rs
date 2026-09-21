//! A minimal OpenVDB writer for single-precision float grids.
//!
//! No Rust crate writes OpenVDB, so this module implements the container format
//! directly. It supports exactly what Core v1 needs: one uncompressed
//! `FloatGrid` with a uniform scale transform and a single root child.

mod tree;
mod writer;

pub use tree::{
    BitMask, internal_child_offset, internal_node_slot_offset, leaf_voxel_offset, write_float_grid,
};
pub use writer::{
    ByteWriter, COMPRESSION_ACTIVE_MASK, FLOAT_GRID_TYPE, GridOffsets, MetaValue,
    NO_MASK_AND_ALL_VALS, NO_MASK_OR_INACTIVE_VALS, OPENVDB_FILE_VERSION, OPENVDB_LIBRARY_MAJOR,
    OPENVDB_LIBRARY_MINOR, OPENVDB_MAGIC, write_archive_header, write_grid_descriptor,
    write_metadata,
};
