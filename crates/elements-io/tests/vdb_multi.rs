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

fn read_active(path: &Path, name: &str) -> HashMap<(i32, i32, i32), f32> {
    let file = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let mut reader = vdb_rs::VdbReader::new(file).unwrap();
    let grid = reader.read_grid::<f32>(name).unwrap();
    grid.iter()
        .map(|(c, v, _)| ((c.x as i32, c.y as i32, c.z as i32), v))
        .collect()
}

#[test]
fn every_grid_of_a_multi_grid_file_round_trips() {
    // Asymmetric dims cross leaf borders on every axis.
    let dims = [9u32, 5, 3];
    let n = 9 * 5 * 3;
    let (a, b, c) = (ramp(n, 0.0), ramp(n, 100.0), ramp(n, -50.0));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.vdb");
    elements_io::write_float_grids(
        &path,
        &[
            elements_io::GridSpec {
                name: "density",
                values: &a,
            },
            elements_io::GridSpec {
                name: "flame",
                values: &b,
            },
            elements_io::GridSpec {
                name: "temperature",
                values: &c,
            },
        ],
        dims,
        0.1,
        0.0,
    )
    .unwrap();

    let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let reader = vdb_rs::VdbReader::new(file).unwrap();
    let mut names = reader.available_grids();
    names.sort();
    assert_eq!(names, ["density", "flame", "temperature"]);

    for (name, values) in [("density", &a), ("flame", &b), ("temperature", &c)] {
        let voxels = read_active(&path, name);
        assert_eq!(voxels.len(), n, "{name}");
        for z in 0..3i32 {
            for y in 0..5i32 {
                for x in 0..9i32 {
                    let linear = (z as usize * 5 + y as usize) * 9 + x as usize;
                    assert_eq!(voxels[&(x, y, z)], values[linear], "{name} ({x}, {y}, {z})");
                }
            }
        }
    }
}

#[test]
fn an_empty_grid_list_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let err = elements_io::write_float_grids(&dir.path().join("e.vdb"), &[], [2, 2, 2], 0.1, 0.0)
        .unwrap_err();
    assert!(matches!(err, elements_io::IoError::NoGrids), "{err:?}");
}

#[test]
fn a_duplicate_grid_name_is_an_error_and_writes_no_file() {
    let v = vec![0.0f32; 8];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.vdb");
    let spec = elements_io::GridSpec {
        name: "fuel",
        values: &v,
    };
    let err =
        elements_io::write_float_grids(&path, &[spec, spec], [2, 2, 2], 0.1, 0.0).unwrap_err();
    match err {
        elements_io::IoError::DuplicateGrid { name } => assert_eq!(name, "fuel"),
        other => panic!("expected DuplicateGrid, got {other:?}"),
    }
    assert!(!path.exists());
}

#[test]
fn a_grid_of_the_wrong_length_is_an_error() {
    let ok = vec![0.0f32; 8];
    let short = vec![0.0f32; 7];
    let dir = tempfile::tempdir().unwrap();
    let err = elements_io::write_float_grids(
        &dir.path().join("l.vdb"),
        &[
            elements_io::GridSpec {
                name: "a",
                values: &ok,
            },
            elements_io::GridSpec {
                name: "b",
                values: &short,
            },
        ],
        [2, 2, 2],
        0.1,
        0.0,
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            elements_io::IoError::LengthMismatch {
                expected: 8,
                got: 7
            }
        ),
        "{err:?}"
    );
}

/// `vdb-rs` reads names from the descriptor, so it cannot tell whether each
/// grid's own metadata map carries its own "name" (which Blender reads).
/// Scan the bytes for each grid's `name` record.
#[test]
fn every_grid_carries_its_own_name_in_its_metadata() {
    let v = vec![0.0f32; 8];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("n.vdb");
    let names = ["density", "flame", "temperature"];
    let specs: Vec<_> = names
        .iter()
        .map(|&name| elements_io::GridSpec { name, values: &v })
        .collect();
    elements_io::write_float_grids(&path, &specs, [2, 2, 2], 0.1, 0.0).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    for name in names {
        let mut expected = Vec::new();
        for part in ["name", "string"] {
            expected.extend_from_slice(&(part.len() as u32).to_le_bytes());
            expected.extend_from_slice(part.as_bytes());
        }
        expected.extend_from_slice(&(name.len() as u32).to_le_bytes());
        expected.extend_from_slice(name.as_bytes());
        assert!(
            bytes
                .windows(expected.len())
                .any(|w| w == expected.as_slice()),
            "no metadata record names grid {name:?}"
        );
    }
}
