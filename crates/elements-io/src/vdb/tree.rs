//! Assembles a 5-4-3 OpenVDB tree from a dense field and serializes it.

use std::collections::BTreeMap;
use std::io::{Seek, Write};
use std::path::Path;

use crate::IoError;

use super::writer::{
    ByteWriter, COMPRESSION_ACTIVE_MASK, MetaValue, NO_MASK_AND_ALL_VALS, NO_MASK_OR_INACTIVE_VALS,
    write_archive_header, write_grid_descriptor, write_metadata,
};

/// Voxels per leaf edge.
#[allow(dead_code)]
const LEAF_DIM: u32 = 8;
/// Voxels covered by one internal node edge (16 leaves of 8).
#[allow(dead_code)]
const INTERNAL_DIM: u32 = 128;
/// Voxels covered by one root-child edge (32 internal nodes of 128).
const ROOT_CHILD_DIM: u32 = 4096;

const LEAF_VOXELS: usize = 512;
const INTERNAL_CHILDREN: usize = 4096;
const ROOT_CHILD_CHILDREN: usize = 32768;

/// Linear index of a voxel within its leaf.
pub fn leaf_voxel_offset(x: u32, y: u32, z: u32) -> usize {
    (((x & 7) << 6) | ((y & 7) << 3) | (z & 7)) as usize
}

/// Linear index of a leaf within its internal node.
pub fn internal_child_offset(x: u32, y: u32, z: u32) -> usize {
    ((((x & 127) >> 3) << 8) | (((y & 127) >> 3) << 4) | ((z & 127) >> 3)) as usize
}

/// Linear index of an internal node within the root child.
pub fn internal_node_slot_offset(x: u32, y: u32, z: u32) -> usize {
    ((((x & 4095) >> 7) << 10) | (((y & 4095) >> 7) << 5) | ((z & 4095) >> 7)) as usize
}

/// A bit array serialized as little-endian `u64` words, LSB-first.
#[derive(Debug, Clone)]
pub struct BitMask {
    words: Vec<u64>,
    bits: usize,
}

impl BitMask {
    pub fn new(bits: usize) -> Self {
        Self {
            words: vec![0; bits.div_ceil(64)],
            bits,
        }
    }

    pub fn set(&mut self, index: usize) {
        debug_assert!(index < self.bits);
        self.words[index / 64] |= 1u64 << (index % 64);
    }

    pub fn get(&self, index: usize) -> bool {
        index < self.bits && self.words[index / 64] & (1u64 << (index % 64)) != 0
    }

    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.bits).filter(|i| self.get(*i))
    }

    pub fn write<W: Write + Seek>(&self, w: &mut ByteWriter<W>) -> Result<(), IoError> {
        for word in &self.words {
            w.u64(*word)?;
        }
        Ok(())
    }
}

/// One leaf: its active-value mask and its 512 values.
struct Leaf {
    value_mask: BitMask,
    values: Vec<f32>,
}

/// One internal node: which leaves exist, and the leaves themselves.
struct Internal {
    child_mask: BitMask,
    leaves: BTreeMap<usize, Leaf>,
}

/// Bucket a dense field into the 5-4-3 tree.
///
/// Every voxel inside `dims` is marked active. Voxels that fall inside a leaf
/// but outside `dims` stay inactive and carry `background`.
fn build_tree(
    values: &[f32],
    dims: [u32; 3],
    background: f32,
) -> Result<(BitMask, BTreeMap<usize, Internal>), IoError> {
    if dims[0] > ROOT_CHILD_DIM || dims[1] > ROOT_CHILD_DIM || dims[2] > ROOT_CHILD_DIM {
        return Err(IoError::FieldTooLarge { dims });
    }
    let expected = dims[0] as usize * dims[1] as usize * dims[2] as usize;
    if values.len() != expected {
        return Err(IoError::LengthMismatch {
            expected,
            got: values.len(),
        });
    }

    let mut root_mask = BitMask::new(ROOT_CHILD_CHILDREN);
    let mut internals: BTreeMap<usize, Internal> = BTreeMap::new();

    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                // `values` is x-fastest, matching Field::read_back.
                let linear =
                    (z as usize * dims[1] as usize + y as usize) * dims[0] as usize + x as usize;
                let value = values[linear];

                let r = internal_node_slot_offset(x, y, z);
                let i = internal_child_offset(x, y, z);
                let v = leaf_voxel_offset(x, y, z);

                root_mask.set(r);
                let internal = internals.entry(r).or_insert_with(|| Internal {
                    child_mask: BitMask::new(INTERNAL_CHILDREN),
                    leaves: BTreeMap::new(),
                });
                internal.child_mask.set(i);
                let leaf = internal.leaves.entry(i).or_insert_with(|| Leaf {
                    value_mask: BitMask::new(LEAF_VOXELS),
                    values: vec![background; LEAF_VOXELS],
                });
                leaf.value_mask.set(v);
                leaf.values[v] = value;
            }
        }
    }

    Ok((root_mask, internals))
}

/// Write one uncompressed `FloatGrid` to `path`.
///
/// `values` is x-fastest with `dims[0] * dims[1] * dims[2]` entries.
/// `voxel_size` is the uniform world-space size of one voxel.
pub fn write_float_grid(
    path: &Path,
    name: &str,
    values: &[f32],
    dims: [u32; 3],
    voxel_size: f64,
    background: f32,
) -> Result<(), IoError> {
    let (root_mask, internals) = build_tree(values, dims, background)?;

    let file = std::fs::File::create(path)?;
    let mut w = ByteWriter::new(std::io::BufWriter::new(file));

    write_archive_header(&mut w, "00000000-0000-4000-8000-000000000000")?;
    let offsets = write_grid_descriptor(&mut w, name)?;

    let grid_pos = w.pos()?;
    w.u32(COMPRESSION_ACTIVE_MASK)?;
    write_metadata(
        &mut w,
        &[
            // Readers (Blender's bundled OpenVDB among them) take a grid's
            // display name from this "name" entry in its own metadata map,
            // not from the archive's grid descriptor -- the descriptor name
            // is only a unique key for instancing within the file. Without
            // this entry a grid opens with an empty name, which is silent in
            // `vdb-rs` (it reads the descriptor name instead) but breaks any
            // consumer, such as Blender's Volume object, that looks a grid
            // up by name after import.
            ("name", MetaValue::String(name.to_owned())),
            ("file_bbox_min", MetaValue::Vec3i([0, 0, 0])),
            (
                "file_bbox_max",
                MetaValue::Vec3i([
                    dims[0].saturating_sub(1) as i32,
                    dims[1].saturating_sub(1) as i32,
                    dims[2].saturating_sub(1) as i32,
                ]),
            ),
            ("class", MetaValue::String("unknown".to_owned())),
            (
                "file_voxel_count",
                MetaValue::I64(dims[0] as i64 * dims[1] as i64 * dims[2] as i64),
            ),
        ],
    )?;

    write_uniform_scale_transform(&mut w, voxel_size)?;
    write_tree_topology(&mut w, &root_mask, &internals, background)?;

    let block_pos = w.pos()?;
    write_tree_data(&mut w, &root_mask, &internals)?;
    let end_pos = w.pos()?;

    w.patch_u64_at(offsets.grid_pos_at, grid_pos)?;
    w.patch_u64_at(offsets.block_pos_at, block_pos)?;
    w.patch_u64_at(offsets.end_pos_at, end_pos)?;

    Ok(())
}

/// A `UniformScaleMap`: five `Vec3d`s derived from one scale.
fn write_uniform_scale_transform<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    voxel_size: f64,
) -> Result<(), IoError> {
    let s = voxel_size;
    let inv = 1.0 / s;
    w.string("UniformScaleMap")?;
    w.dvec3([s, s, s])?; // scale_values
    w.dvec3([s, s, s])?; // voxel_size
    w.dvec3([inv, inv, inv])?; // scale_values_inverse
    w.dvec3([inv * inv, inv * inv, inv * inv])?; // inv_scale_sqr
    w.dvec3([0.5 * inv, 0.5 * inv, 0.5 * inv])?; // inv_twice_scale
    Ok(())
}

/// The topology pass: node masks, no leaf values.
fn write_tree_topology<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    root_mask: &BitMask,
    internals: &BTreeMap<usize, Internal>,
    background: f32,
) -> Result<(), IoError> {
    w.u32(1)?; // buffer_count
    w.u32(background.to_bits())?; // root background value
    w.u32(0)?; // tile count
    w.u32(1)?; // root child count

    // The single root child, anchored at the origin.
    w.i32(0)?;
    w.i32(0)?;
    w.i32(0)?;

    root_mask.write(w)?;
    // Root child holds no tiles, so its value mask is empty.
    BitMask::new(ROOT_CHILD_CHILDREN).write(w)?;
    // With ACTIVE_MASK compression and this metadata byte, zero values follow.
    w.u8(NO_MASK_OR_INACTIVE_VALS)?;

    for index in root_mask.iter_ones() {
        let internal = &internals[&index];
        internal.child_mask.write(w)?;
        BitMask::new(INTERNAL_CHILDREN).write(w)?;
        w.u8(NO_MASK_OR_INACTIVE_VALS)?;

        for leaf_index in internal.child_mask.iter_ones() {
            internal.leaves[&leaf_index].value_mask.write(w)?;
        }
    }

    Ok(())
}

/// The data pass: each leaf's mask repeated, then all 512 values.
fn write_tree_data<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    root_mask: &BitMask,
    internals: &BTreeMap<usize, Internal>,
) -> Result<(), IoError> {
    for index in root_mask.iter_ones() {
        let internal = &internals[&index];
        for leaf_index in internal.child_mask.iter_ones() {
            let leaf = &internal.leaves[&leaf_index];
            leaf.value_mask.write(w)?;
            w.u8(NO_MASK_AND_ALL_VALS)?;
            for v in &leaf.values {
                w.f32(*v)?;
            }
        }
    }
    Ok(())
}
