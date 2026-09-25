//! The Mantaflow cache reader, which lives with the benchmark example so
//! that `vdb-rs` stays a dev-dependency.

#[path = "../examples/common/mantaflow.rs"]
mod mantaflow;

use std::path::PathBuf;

use elements_core::gpu::FieldDims;
use mantaflow::{CacheError, read_float_grid, read_frame, read_frame_with};

const N: u32 = 16;
const DX: f64 = 0.125;

fn fixture() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mantaflow_16/fluid_data_0005.vdb"
    ))
}

fn at(i: u32, j: u32, k: u32, nx: u32, ny: u32) -> usize {
    (i + nx * (j + ny * k)) as usize
}

#[test]
fn a_cache_frame_reads_into_the_domain_layout() {
    let f = read_frame(&fixture(), FieldDims::new(N, N, N), DX).unwrap();
    assert_eq!(f.cells, FieldDims::new(N, N, N));
    assert_eq!(f.density.len(), (N * N * N) as usize);
    let stored = f.density.iter().filter(|&&d| d != 0.0).count();
    assert_eq!(stored, 240, "the README's density value count");
    // The README's index box for density is (4,4,1)–(11,11,7).
    for (n, &d) in f.density.iter().enumerate() {
        let (i, j, k) = (n as u32 % N, (n as u32 / N) % N, n as u32 / (N * N));
        if d != 0.0 {
            assert!(
                (4..=11).contains(&i) && (4..=11).contains(&j) && (1..=7).contains(&k),
                "({i}, {j}, {k})"
            );
        }
    }
    // The README's reference values: total density and one known cell.
    let sum: f64 = f.density.iter().map(|&d| d as f64).sum();
    assert!((sum - 49.2145).abs() < 1e-3, "{sum}");
    let d882 = f.density[at(8, 8, 2, N, N)];
    assert!((d882 - 0.999351).abs() < 1e-5, "{d882}");
    // Face grids are one longer along their axis, and face n is empty.
    assert_eq!(f.faces[0].len(), ((N + 1) * N * N) as usize);
    assert_eq!(f.faces[1].len(), (N * (N + 1) * N) as usize);
    assert_eq!(f.faces[2].len(), (N * N * (N + 1)) as usize);
    let top = &f.faces[2][(N * N * N) as usize..];
    assert!(top.iter().all(|&w| w == 0.0));
    // The plume rises, at a plausible speed in m/s (not grid units).
    let wmax = f.faces[2].iter().copied().fold(f32::MIN, f32::max);
    assert!(wmax > 0.0 && wmax < 5.0, "{wmax}");
    // The README's velocity reference, in grid units, times dx / 0.4:
    // z sum 104.634 and index (8,8,3) z 3.11418, on the −z face of (8,8,3).
    let scale = DX / 0.4;
    let wsum: f64 = f.faces[2].iter().map(|&w| w as f64).sum();
    assert!((wsum - 104.634 * scale).abs() < 1e-3, "{wsum}");
    let w883 = f.faces[2][at(8, 8, 3, N, N)] as f64;
    assert!((w883 - 3.11418 * scale).abs() < 1e-5, "{w883}");
    let usum: f64 = f.faces[0].iter().map(|&u| u as f64).sum();
    assert!((usum - 3.08923 * scale).abs() < 1e-4, "{usum}");
}

/// Units: a stored value s is s · dx / 0.4 m/s, so halving dx halves every
/// velocity and leaves density alone.
#[test]
fn velocity_scales_with_the_voxel_size() {
    let a = read_frame(&fixture(), FieldDims::new(N, N, N), DX).unwrap();
    let b = read_frame(&fixture(), FieldDims::new(N, N, N), DX / 2.0).unwrap();
    assert_eq!(a.density, b.density);
    for axis in 0..3 {
        for (x, y) in a.faces[axis].iter().zip(&b.faces[axis]) {
            assert!((x / 2.0 - y).abs() <= 1e-7 * x.abs().max(1.0), "{x} {y}");
        }
    }
}

#[test]
fn a_smaller_domain_than_the_cache_is_an_error() {
    let err = read_frame(&fixture(), FieldDims::new(8, 8, 8), DX).err();
    assert!(
        matches!(err, Some(CacheError::OutOfDomain { .. })),
        "{err:?}"
    );
}

#[test]
fn a_missing_file_is_an_error() {
    let err = read_frame(
        &fixture().with_file_name("nope.vdb"),
        FieldDims::new(N, N, N),
        DX,
    )
    .err();
    assert!(matches!(err, Some(CacheError::Open { .. })), "{err:?}");
}

#[test]
fn a_missing_grid_is_an_error() {
    let err = read_frame_with(&fixture(), FieldDims::new(N, N, N), DX, "smoke", "velocity").err();
    assert!(
        matches!(err, Some(CacheError::MissingGrid { .. })),
        "{err:?}"
    );
}

/// `vdb-rs` does not check value types; the reader must. Reading the vector
/// grid as density is a type error, not an empty or garbled grid.
#[test]
fn a_grid_of_the_wrong_type_is_an_error() {
    let err = read_frame_with(
        &fixture(),
        FieldDims::new(N, N, N),
        DX,
        "velocity",
        "velocity",
    )
    .err();
    assert!(matches!(err, Some(CacheError::WrongType { .. })), "{err:?}");
}

/// The fixture's density and velocity cover the same cells.
#[test]
fn the_fixture_misses_no_velocity() {
    let f = read_frame(&fixture(), FieldDims::new(N, N, N), DX).unwrap();
    assert_eq!(f.missing_velocity, Vec::<[u32; 3]>::new());
}

/// The fixture's `shadow` grid fills the whole domain, much of it as tiles,
/// while velocity covers only the smoke. Read as the density grid, every
/// cell outside the smoke is missing velocity, and no smoky cell is.
#[test]
fn a_cell_with_density_but_no_velocity_is_listed() {
    let f = read_frame_with(
        &fixture(),
        FieldDims::new(N, N, N),
        DX,
        "shadow",
        "velocity",
    )
    .unwrap();
    let smoke = read_frame(&fixture(), FieldDims::new(N, N, N), DX).unwrap();
    let smoky = |c: &[u32; 3]| smoke.density[at(c[0], c[1], c[2], N, N)] != 0.0;
    assert_eq!(f.missing_velocity.len(), (N * N * N) as usize - 240);
    assert!(f.missing_velocity.contains(&[0, 0, 0]));
    assert!(
        !f.missing_velocity.iter().any(smoky),
        "a smoky cell is listed"
    );
}

#[test]
fn any_float_grid_reads_into_the_domain_layout() {
    let cells = FieldDims::new(N, N, N);
    let t = read_float_grid(&fixture(), cells, "temperature")
        .unwrap()
        .expect("the fixture has a temperature grid");
    let sum: f64 = t.iter().map(|&v| f64::from(v)).sum();
    // The README's reference value.
    assert!((sum - 62.6631).abs() < 1e-3, "temperature sum {sum}");
    let flame = read_float_grid(&fixture(), cells, "flame")
        .unwrap()
        .expect("the fixture has an empty flame grid");
    assert!(flame.iter().all(|&v| v == 0.0), "flame is empty");
    assert!(
        read_float_grid(&fixture(), cells, "fuel")
            .unwrap()
            .is_none(),
        "a smoke-only cache has no fuel grid"
    );
}
