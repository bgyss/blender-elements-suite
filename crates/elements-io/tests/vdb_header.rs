use std::io::Cursor;

use elements_io::vdb::{
    ByteWriter, COMPRESSION_ACTIVE_MASK, MetaValue, OPENVDB_MAGIC, write_archive_header,
    write_float_grid, write_grid_descriptor, write_metadata,
};

/// Build a file containing a header and one grid whose body is metadata only.
/// `vdb-rs` reads descriptors without touching the transform or the tree, so
/// this is enough to prove the header and descriptor are well formed.
fn header_only_file(name: &str) -> Vec<u8> {
    let mut w = ByteWriter::new(Cursor::new(Vec::new()));

    write_archive_header(&mut w, "00000000-0000-4000-8000-000000000000").unwrap();
    let offsets = write_grid_descriptor(&mut w, name).unwrap();

    let grid_pos = w.pos().unwrap();
    w.u32(COMPRESSION_ACTIVE_MASK).unwrap();
    write_metadata(
        &mut w,
        &[
            ("file_bbox_min", MetaValue::Vec3i([0, 0, 0])),
            ("file_bbox_max", MetaValue::Vec3i([7, 7, 7])),
        ],
    )
    .unwrap();
    let end_pos = w.pos().unwrap();

    w.patch_u64_at(offsets.grid_pos_at, grid_pos).unwrap();
    w.patch_u64_at(offsets.block_pos_at, end_pos).unwrap();
    w.patch_u64_at(offsets.end_pos_at, end_pos).unwrap();

    w.into_inner().into_inner()
}

#[test]
fn vdb_rs_reads_our_archive_header() {
    let bytes = header_only_file("density");
    let reader = vdb_rs::VdbReader::new(Cursor::new(bytes)).expect("header must parse");

    assert_eq!(reader.header.file_version, 224);
    assert_eq!(reader.header.grid_count, 1);
    assert!(reader.header.has_grid_offsets);
    assert_eq!(reader.header.guid.len(), 36);
}

#[test]
fn vdb_rs_lists_our_grid_by_name() {
    let bytes = header_only_file("density");
    let reader = vdb_rs::VdbReader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.available_grids(), vec!["density".to_string()]);
}

#[test]
fn grid_metadata_round_trips_through_vdb_rs() {
    let bytes = header_only_file("density");
    let reader = vdb_rs::VdbReader::new(Cursor::new(bytes)).unwrap();
    let gd = &reader.grid_descriptors["density"];

    assert_eq!(gd.grid_type, "Tree_float_5_4_3");
    assert_eq!(gd.aabb_min().unwrap(), glam::IVec3::new(0, 0, 0));
    assert_eq!(gd.aabb_max().unwrap(), glam::IVec3::new(7, 7, 7));
    assert!(!gd.meta_data.is_half_float());
}

#[test]
fn byte_writer_patches_offsets_in_place() {
    let mut w = ByteWriter::new(Cursor::new(Vec::new()));
    w.u32(0xdead_beef).unwrap();
    let slot = w.pos().unwrap();
    w.u64(0).unwrap();
    w.u32(0x1234_5678).unwrap();
    w.patch_u64_at(slot, 0x0102_0304_0506_0708).unwrap();

    let bytes = w.into_inner().into_inner();
    assert_eq!(&bytes[0..4], &0xdead_beefu32.to_le_bytes());
    assert_eq!(&bytes[4..12], &0x0102_0304_0506_0708u64.to_le_bytes());
    assert_eq!(&bytes[12..16], &0x1234_5678u32.to_le_bytes());
}

#[test]
fn archive_header_starts_with_the_openvdb_magic() {
    let bytes = header_only_file("density");
    // vdb-rs's magic check is a blocklist that only rejects one byte-swapped
    // value, so a successful parse alone would not prove the magic is right.
    assert_eq!(&bytes[0..8], &OPENVDB_MAGIC.to_le_bytes());
}

#[test]
fn write_archive_header_rejects_a_malformed_uuid() {
    let mut w = ByteWriter::new(Cursor::new(Vec::new()));
    let err = write_archive_header(&mut w, "not-a-uuid").unwrap_err();
    assert!(
        matches!(err, elements_io::IoError::BadUuid { len: 10 }),
        "got {err:?}"
    );
}

/// `vdb_rs_lists_our_grid_by_name` reads the grid name from the archive's grid
/// DESCRIPTOR, so it would pass identically whether or not the grid's own
/// metadata map carries a "name" entry -- it is not an oracle for the fix in
/// `tree.rs`, which adds that entry because real OpenVDB (and Blender's
/// bundled build) reads a grid's *display* name from its metadata map, not
/// the descriptor. This test scans the written bytes directly, independent
/// of `vdb-rs`, for the metadata record that stores it: a length-prefixed key
/// "name", a length-prefixed type "string", the payload length, and the name
/// bytes -- exactly the encoding `MetaValue::write` produces, using the same
/// byte-scanning technique as `archive_header_starts_with_the_openvdb_magic`.
#[test]
fn grid_metadata_contains_the_grid_name() {
    let dir = std::env::temp_dir().join(format!(
        "elements-io-test-{}-{}",
        std::process::id(),
        "grid_metadata_contains_the_grid_name"
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("density.vdb");

    let name = "density";
    write_float_grid(&path, name, &[0.0f32; 8], [2, 2, 2], 1.0, 0.0).unwrap();
    let bytes = std::fs::read(&path).unwrap();

    let mut expected = Vec::new();
    let key = b"name";
    expected.extend_from_slice(&(key.len() as u32).to_le_bytes());
    expected.extend_from_slice(key);
    let type_name = b"string";
    expected.extend_from_slice(&(type_name.len() as u32).to_le_bytes());
    expected.extend_from_slice(type_name);
    expected.extend_from_slice(&(name.len() as u32).to_le_bytes());
    expected.extend_from_slice(name.as_bytes());

    let found = bytes
        .windows(expected.len())
        .any(|window| window == expected.as_slice());
    assert!(
        found,
        "did not find a metadata record naming the grid {name:?} in the written file bytes"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn strings_are_length_prefixed() {
    let mut w = ByteWriter::new(Cursor::new(Vec::new()));
    w.string("abc").unwrap();
    let bytes = w.into_inner().into_inner();
    assert_eq!(bytes, vec![3, 0, 0, 0, b'a', b'b', b'c']);
}
