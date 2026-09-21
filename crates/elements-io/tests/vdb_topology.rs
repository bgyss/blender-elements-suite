use elements_io::vdb::{
    BitMask, internal_child_offset, internal_node_slot_offset, leaf_voxel_offset,
};
use elements_io::write_float_grid;

#[test]
fn leaf_voxel_offsets_are_x_major() {
    assert_eq!(leaf_voxel_offset(0, 0, 0), 0);
    assert_eq!(leaf_voxel_offset(0, 0, 1), 1);
    assert_eq!(leaf_voxel_offset(0, 1, 0), 8);
    assert_eq!(leaf_voxel_offset(1, 0, 0), 64);
    assert_eq!(leaf_voxel_offset(7, 7, 7), 511);
    // Coordinates wrap within the node.
    assert_eq!(leaf_voxel_offset(8, 0, 0), 0);
}

#[test]
fn internal_child_offsets_index_16_cubed() {
    assert_eq!(internal_child_offset(0, 0, 0), 0);
    assert_eq!(internal_child_offset(0, 0, 8), 1);
    assert_eq!(internal_child_offset(0, 8, 0), 16);
    assert_eq!(internal_child_offset(8, 0, 0), 256);
    assert_eq!(internal_child_offset(127, 127, 127), 4095);
}

#[test]
fn internal_node_slot_offsets_index_32_cubed() {
    assert_eq!(internal_node_slot_offset(0, 0, 0), 0);
    assert_eq!(internal_node_slot_offset(0, 0, 128), 1);
    assert_eq!(internal_node_slot_offset(0, 128, 0), 32);
    assert_eq!(internal_node_slot_offset(128, 0, 0), 1024);
}

#[test]
fn bitmask_sets_and_counts() {
    let mut m = BitMask::new(512);
    assert_eq!(m.count_ones(), 0);
    m.set(0);
    m.set(63);
    m.set(64);
    m.set(511);
    assert!(m.get(0) && m.get(63) && m.get(64) && m.get(511));
    assert!(!m.get(1));
    assert_eq!(m.count_ones(), 4);
}

#[test]
fn bitmask_serializes_as_little_endian_u64_words() {
    let mut m = BitMask::new(128);
    m.set(0);
    m.set(65);

    let mut buf = Vec::new();
    {
        let mut w = elements_io::vdb::ByteWriter::new(std::io::Cursor::new(&mut buf));
        m.write(&mut w).unwrap();
    }
    assert_eq!(buf.len(), 16);
    assert_eq!(&buf[0..8], &1u64.to_le_bytes());
    assert_eq!(&buf[8..16], &2u64.to_le_bytes());
}

#[test]
fn vdb_rs_reads_a_full_tree_from_our_writer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zeros.vdb");
    let dims = [8u32, 8, 8];
    let values = vec![0.0f32; 512];

    write_float_grid(&path, "density", &values, dims, 0.1, 0.0).unwrap();

    let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let mut reader = vdb_rs::VdbReader::new(file).unwrap();
    let grid = reader.read_grid::<f32>("density").expect("tree must parse");

    assert_eq!(grid.tree.root_nodes.len(), 1);
    let root = &grid.tree.root_nodes[0];
    assert_eq!(root.child_mask.count_ones(), 1, "one 128^3 internal node");
}

#[test]
fn leaf_values_are_not_scrambled_across_leaves() {
    // Two leaves side by side along x (dims[0]=16 spans leaf origin 0 and
    // leaf origin 8). Each voxel's value is its x coordinate, so leaf 0
    // holds 0..8 and leaf 1 holds 8..16. If the topology pass and the data
    // pass ever disagree on leaf order, the data pass reads the wrong
    // leaf's bytes into the wrong leaf's slot, and this swaps whole ranges
    // of values between the two leaves.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two_leaves.vdb");
    let dims = [16u32, 8, 8];
    let mut values = vec![0.0f32; 16 * 8 * 8];
    for z in 0..8u32 {
        for y in 0..8u32 {
            for x in 0..16u32 {
                let linear = (z as usize * 8 + y as usize) * 16 + x as usize;
                values[linear] = x as f32;
            }
        }
    }

    write_float_grid(&path, "density", &values, dims, 0.1, -1.0).unwrap();

    let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let mut reader = vdb_rs::VdbReader::new(file).unwrap();
    let grid = reader.read_grid::<f32>("density").expect("tree must parse");

    let mut by_coord = std::collections::HashMap::new();
    for (coord, value, _level) in grid.iter() {
        by_coord.insert((coord.x as i32, coord.y as i32, coord.z as i32), value);
    }

    // One representative voxel from each leaf is enough to catch a swap.
    assert_eq!(by_coord[&(0, 0, 0)], 0.0, "leaf 0's voxel read back wrong");
    assert_eq!(by_coord[&(8, 0, 0)], 8.0, "leaf 1's voxel read back wrong");
}

#[test]
fn rejects_fields_larger_than_one_root_child() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("huge.vdb");
    let err = write_float_grid(&path, "density", &[], [5000, 1, 1], 1.0, 0.0).unwrap_err();
    assert!(
        matches!(err, elements_io::IoError::FieldTooLarge { .. }),
        "got {err:?}"
    );
}
