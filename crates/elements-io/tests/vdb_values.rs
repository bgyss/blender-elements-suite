//! Task 12: prove the OpenVDB writer is *correct*, not just parseable.
//!
//! Tasks 10-11 established that `vdb-rs` can parse our files. This module
//! checks that voxels land at the right coordinates with the right values,
//! using dimensions that are deliberately asymmetric so that a transposition
//! or an ordering bug cannot hide behind a cube or a single leaf.

use std::collections::HashMap;

/// Read every *active* voxel back via `vdb_rs::Grid::iter()`.
///
/// `Grid::iter()` walks each leaf's `value_mask` (not its `child_mask`), so
/// inactive voxels are never yielded -- confirmed by reading
/// `vdb-rs-0.6.0/src/data_structure.rs`: the `VdbLevel::Voxel` arm of
/// `GridIter::next` drives itself from `node_3.value_mask.iter_ones()`. That
/// makes this the right (and simpler) API for every test here except the
/// background-fill test, which needs the raw buffer at an INACTIVE offset and
/// so has to walk the node structs by hand.
fn read_active_voxels(path: &std::path::Path, name: &str) -> HashMap<(i32, i32, i32), f32> {
    let file = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let mut reader = vdb_rs::VdbReader::new(file).unwrap();
    let grid = reader.read_grid::<f32>(name).unwrap();

    let mut out = HashMap::new();
    for (coord, value, _level) in grid.iter() {
        out.insert((coord.x as i32, coord.y as i32, coord.z as i32), value);
    }
    out
}

fn ramp(dims: [u32; 3]) -> Vec<f32> {
    let n = (dims[0] * dims[1] * dims[2]) as usize;
    (0..n).map(|i| i as f32 * 0.25).collect()
}

#[test]
fn values_round_trip_for_an_aligned_field() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("aligned.vdb");
    let dims = [8u32, 8, 8];
    let values = ramp(dims);

    elements_io::write_float_grid(&path, "density", &values, dims, 0.1, 0.0).unwrap();
    let voxels = read_active_voxels(&path, "density");

    assert_eq!(voxels.len(), 512);
    for z in 0..8i32 {
        for y in 0..8i32 {
            for x in 0..8i32 {
                let linear = (z as usize * 8 + y as usize) * 8 + x as usize;
                assert_eq!(
                    voxels[&(x, y, z)],
                    values[linear],
                    "mismatch at ({x}, {y}, {z})"
                );
            }
        }
    }
}

#[test]
fn values_round_trip_across_multiple_leaves() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.vdb");
    let dims = [16u32, 16, 16];
    let values = ramp(dims);

    elements_io::write_float_grid(&path, "density", &values, dims, 0.1, 0.0).unwrap();
    let voxels = read_active_voxels(&path, "density");

    assert_eq!(voxels.len(), 4096, "8 leaves of 512 voxels");
    // Hand-worked: x=11, y=3, z=9, dims=[16,16,16].
    // linear = (z * dims[1] + y) * dims[0] + x
    //        = (9 * 16 + 3) * 16 + 11
    //        = (144 + 3) * 16 + 11
    //        = 147 * 16 + 11
    //        = 2352 + 11
    //        = 2363
    let linear = (9usize * 16 + 3) * 16 + 11;
    assert_eq!(linear, 2363);
    assert_eq!(voxels[&(11, 3, 9)], values[linear]);
}

#[test]
fn partial_leaves_mark_only_real_voxels_active() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial.vdb");
    // 5^3 occupies one leaf but activates only 125 of its 512 voxels.
    let dims = [5u32, 5, 5];
    let values = ramp(dims);

    elements_io::write_float_grid(&path, "density", &values, dims, 0.1, 0.0).unwrap();
    let voxels = read_active_voxels(&path, "density");

    assert_eq!(voxels.len(), 125, "only in-bounds voxels are active");
    assert!(voxels.contains_key(&(4, 4, 4)));
    assert!(
        !voxels.contains_key(&(5, 0, 0)),
        "outside dims stays inactive"
    );
    assert_eq!(voxels[&(4, 4, 4)], values[(4 * 5 + 4) * 5 + 4]);
}

#[test]
fn a_field_spanning_two_internal_nodes_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wide.vdb");
    // 136 > 128 forces a second internal node along x.
    let dims = [136u32, 8, 8];
    let values = ramp(dims);

    elements_io::write_float_grid(&path, "density", &values, dims, 0.1, 0.0).unwrap();
    let voxels = read_active_voxels(&path, "density");

    assert_eq!(voxels.len(), 136 * 8 * 8);
    // Hand-worked: x=130, y=0, z=0, dims=[136,8,8].
    // linear = (z * dims[1] + y) * dims[0] + x = (0 * 8 + 0) * 136 + 130 = 130
    let linear = 130usize;
    assert_eq!(voxels[&(130, 0, 0)], values[linear]);
}

#[test]
fn background_fills_inactive_voxels_in_a_partial_leaf() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("background.vdb");
    let dims = [2u32, 2, 2];

    elements_io::write_float_grid(&path, "density", &[1.0; 8], dims, 0.1, -3.5).unwrap();

    let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let mut reader = vdb_rs::VdbReader::new(file).unwrap();
    let grid = reader.read_grid::<f32>("density").unwrap();

    // Inactive voxels are only visible by walking the node structs directly:
    // `Grid::iter()` deliberately skips them.
    let root = &grid.tree.root_nodes[0];
    let internal = root.nodes.values().next().unwrap();
    let leaf = internal.nodes.values().next().unwrap();

    assert_eq!(leaf.value_mask.count_ones(), 8);
    // Offset 511 is voxel (7, 7, 7): outside dims, so it carries the background.
    assert_eq!(leaf.buffer[511], -3.5);
}

#[test]
fn a_length_mismatch_is_rejected_before_any_file_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mismatch.vdb");
    let err = elements_io::write_float_grid(&path, "density", &[1.0, 2.0], [4, 4, 4], 0.1, 0.0)
        .unwrap_err();

    assert!(
        matches!(
            err,
            elements_io::IoError::LengthMismatch {
                expected: 64,
                got: 2
            }
        ),
        "got {err:?}"
    );
    assert!(!path.exists(), "no partial file should be left behind");
}
