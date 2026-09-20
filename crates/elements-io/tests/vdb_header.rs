use std::io::Cursor;

use elements_io::vdb::{
    ByteWriter, COMPRESSION_ACTIVE_MASK, MetaValue, write_archive_header, write_grid_descriptor,
    write_metadata,
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
fn strings_are_length_prefixed() {
    let mut w = ByteWriter::new(Cursor::new(Vec::new()));
    w.string("abc").unwrap();
    let bytes = w.into_inner().into_inner();
    assert_eq!(bytes, vec![3, 0, 0, 0, b'a', b'b', b'c']);
}
