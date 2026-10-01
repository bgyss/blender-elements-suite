//! FT5: several grids in one OpenVDB file, read back with `vdb-rs`.

use std::collections::HashMap;
use std::path::Path;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/single_grid_4x3x2.vdb"
);

fn ramp(n: usize, offset: f32) -> Vec<f32> {
    (0..n).map(|i| offset + i as f32 * 0.25).collect()
}

/// Writes the fixture from the single-grid writer as it was before FT5.
/// Run once on unchanged code:
/// `cargo test -p elements-io --test vdb_multi -- --ignored write_single_grid_fixture`.
#[test]
#[ignore = "writes tests/fixtures/single_grid_4x3x2.vdb"]
fn write_single_grid_fixture() {
    std::fs::create_dir_all(Path::new(FIXTURE).parent().unwrap()).unwrap();
    elements_io::write_float_grid(
        Path::new(FIXTURE),
        "density",
        &ramp(24, 0.0),
        [4, 3, 2],
        0.1,
        0.0,
    )
    .unwrap();
}

#[test]
fn a_single_grid_file_is_byte_identical_to_the_pre_ft5_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("one.vdb");
    elements_io::write_float_grid(&path, "density", &ramp(24, 0.0), [4, 3, 2], 0.1, 0.0).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(FIXTURE).expect("fixture committed"),
        "write_float_grid's bytes changed"
    );
}
