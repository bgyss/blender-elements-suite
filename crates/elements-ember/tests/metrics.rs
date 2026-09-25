use elements_core::gpu::FieldDims;
use elements_ember::boundaries::DEFAULT_OPEN_MASK;
use elements_ember::metrics::{SMOKE_THRESHOLD, Sample, cell_centred, drift, measure};

const N: u32 = 16;
const DX: f64 = 0.125; // a 2 m domain

fn cells() -> FieldDims {
    FieldDims::new(N, N, N)
}

fn idx(i: u32, j: u32, k: u32) -> usize {
    (i + N * (j + N * k)) as usize
}

/// Face velocities from a function of position (metres). The x-face at
/// index (i, j, k) sits at (i·dx, (j + ½)·dx, (k + ½)·dx), and likewise for
/// y and z.
fn faces(f: impl Fn([f64; 3]) -> [f64; 3]) -> [Vec<f32>; 3] {
    std::array::from_fn(|axis| {
        let mut dims = [N; 3];
        dims[axis] += 1;
        let mut out = vec![0.0f32; (dims[0] * dims[1] * dims[2]) as usize];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let mut p = [i, j, k].map(|c| (c as f64 + 0.5) * DX);
                    p[axis] -= 0.5 * DX;
                    out[(i + dims[0] * (j + dims[1] * k)) as usize] = f(p)[axis] as f32;
                }
            }
        }
        out
    })
}

/// Density 1 everywhere: every interior cell is smoke.
fn smoke() -> Vec<f32> {
    vec![1.0; (N * N * N) as usize]
}

fn sample<'a>(density: &'a [f32], faces: &'a [Vec<f32>; 3], solid: &'a [bool]) -> Sample<'a> {
    Sample {
        cells: cells(),
        dx: DX,
        density,
        faces,
        solid,
        open_mask: DEFAULT_OPEN_MASK,
    }
}

/// Interior cells when the whole domain is smoke: 1..=14 on each axis.
const INTERIOR: f64 = 14.0 * 14.0 * 14.0;

#[test]
fn a_uniform_flow_has_no_divergence_or_vorticity() {
    let v = faces(|_| [0.3, -0.2, 0.5]);
    let d = smoke();
    let m = measure(&sample(&d, &v, &[]));
    assert_eq!(m.measured_cells, INTERIOR as u64);
    assert!(m.divergence_max < 1e-6, "{m:?}");
    assert!(m.vorticity < 1e-6, "{m:?}");
    let expected = 0.5 * 0.38 * INTERIOR * DX.powi(3);
    assert!(
        (m.kinetic_energy - expected).abs() < 1e-6 * expected,
        "{m:?}"
    );
}

/// Solid-body rotation about z at rate ω has vorticity 2ω everywhere, and no
/// divergence.
#[test]
fn solid_body_rotation_has_vorticity_twice_its_rate() {
    let w = 1.5;
    let v = faces(|p| [-w * (p[1] - 1.0), w * (p[0] - 1.0), 0.0]);
    let d = smoke();
    let m = measure(&sample(&d, &v, &[]));
    let expected = 2.0 * w * INTERIOR * DX.powi(3);
    assert!((m.vorticity - expected).abs() < 1e-4 * expected, "{m:?}");
    assert!(m.divergence_max < 1e-5, "{m:?}");
}

/// u = (a x, 0, 0) has divergence a in every cell.
#[test]
fn a_linear_expansion_has_divergence_equal_to_its_slope() {
    let v = faces(|p| [0.8 * p[0], 0.0, 0.0]);
    let d = smoke();
    let m = measure(&sample(&d, &v, &[]));
    assert!((m.divergence_max - 0.8).abs() < 1e-5, "{m:?}");
    assert!((m.divergence_rms - 0.8).abs() < 1e-5, "{m:?}");
}

#[test]
fn cell_centring_averages_the_two_faces() {
    let c = FieldDims::new(2, 1, 1);
    let f = [vec![1.0, 3.0, 7.0], vec![0.0; 4], vec![0.0; 4]];
    let [x, _, _] = cell_centred(&f, c);
    assert_eq!(x, vec![2.0, 5.0]);
}

/// Only cells whose whole neighbourhood is smoke are measured: a 4³ block
/// of smoke leaves its inner 2³.
#[test]
fn velocity_is_measured_only_inside_the_smoke() {
    let v = faces(|_| [1.0, 0.0, 0.0]);
    let mut d = vec![0.0; (N * N * N) as usize];
    for k in 4..8 {
        for j in 4..8 {
            for i in 4..8 {
                d[idx(i, j, k)] = 1.0;
            }
        }
    }
    let m = measure(&sample(&d, &v, &[]));
    assert_eq!(m.measured_cells, 8, "{m:?}");
    assert!(
        (m.kinetic_energy - 0.5 * 8.0 * DX.powi(3)).abs() < 1e-12,
        "{m:?}"
    );
}

#[test]
fn density_at_the_threshold_is_not_smoke() {
    let v = faces(|_| [1.0, 0.0, 0.0]);
    let d = vec![SMOKE_THRESHOLD; (N * N * N) as usize];
    assert_eq!(measure(&sample(&d, &v, &[])).measured_cells, 0);
}

/// A solid cell removes itself and its 26 neighbours from the measured set.
#[test]
fn solids_and_their_neighbours_are_not_measured() {
    let v = faces(|p| [0.8 * p[0], 0.0, 0.0]);
    let d = smoke();
    let mut solid = vec![false; (N * N * N) as usize];
    solid[idx(8, 8, 8)] = true;
    let m = measure(&sample(&d, &v, &solid));
    assert_eq!(m.measured_cells, INTERIOR as u64 - 27, "{m:?}");
    assert!((m.divergence_rms - 0.8).abs() < 1e-5, "{m:?}");
}

/// A slab of density 2 in layers k = 4..8: mass, centroid and top follow.
#[test]
fn a_density_slab_has_known_mass_centroid_and_top() {
    let mut d = vec![0.0; (N * N * N) as usize];
    for k in 4..8 {
        for j in 0..N {
            for i in 0..N {
                d[idx(i, j, k)] = 2.0;
            }
        }
    }
    let v = faces(|_| [0.0; 3]);
    let m = measure(&sample(&d, &v, &[]));
    let cells_in_slab = (N * N * 4) as f64;
    assert!(
        (m.mass - 2.0 * cells_in_slab * DX.powi(3)).abs() < 1e-9,
        "{m:?}"
    );
    assert!((m.centroid_m.unwrap() - 6.0 * DX).abs() < 1e-9, "{m:?}");
    // 95% of four equal layers is reached in the fourth: k = 7, centre 7.5 dx.
    assert!((m.top_m.unwrap() - 7.5 * DX).abs() < 1e-9, "{m:?}");
}

/// A whole layer of density just below 1% of the peak holds 2.3 times the
/// peak cell's density, so the top would be that layer if it counted.
#[test]
fn cells_below_one_percent_of_the_peak_do_not_move_the_top() {
    let mut d = vec![0.0; (N * N * N) as usize];
    d[idx(3, 3, 2)] = 1.0;
    for j in 0..N {
        for i in 0..N {
            d[idx(i, j, 14)] = 0.009; // below 1% of the peak
        }
    }
    let v = faces(|_| [0.0; 3]);
    let m = measure(&sample(&d, &v, &[]));
    assert!((m.top_m.unwrap() - 2.5 * DX).abs() < 1e-9, "{m:?}");
}

/// Outflow is the net upwind flux through the z-faces at index N − 2:
/// upward flow carries the density below the plane out, downward flow
/// carries the density above it back in, and layer N − 1 is ignored.
#[test]
fn outflow_is_the_net_upwind_flux_two_cells_below_the_top() {
    let mut d = vec![0.0; (N * N * N) as usize];
    d[idx(2, 2, N - 3)] = 3.0; // below an upward face: leaves
    d[idx(2, 2, N - 2)] = 7.0; // above an upward face: not upwind
    d[idx(9, 9, N - 3)] = 5.0; // below a downward face: not upwind
    d[idx(9, 9, N - 2)] = 2.0; // above a downward face: comes back in
    d[idx(4, 4, N - 1)] = 50.0; // the top layer is outside the measurement
    let v = faces(|p| [0.0, 0.0, if p[0] < 1.0 { 0.4 } else { -0.4 }]);
    let m = measure(&sample(&d, &v, &[]));
    let expected = (3.0 * 0.4 - 2.0 * 0.4) * DX * DX;
    assert!((m.outflow_rate - expected).abs() < 1e-9, "{m:?}");
}

/// With only the top open, `mass_inside` is the mass of layers 0 … N − 3.
#[test]
fn mass_inside_leaves_out_the_top_two_layers() {
    let mut d = vec![0.0; (N * N * N) as usize];
    d[idx(1, 1, N - 3)] = 1.0;
    d[idx(1, 1, N - 2)] = 1.0;
    d[idx(1, 1, N - 1)] = 1.0;
    let v = faces(|_| [0.0; 3]);
    let m = measure(&sample(&d, &v, &[]));
    assert!((m.mass - 3.0 * DX.powi(3)).abs() < 1e-12, "{m:?}");
    assert!((m.mass_inside - DX.powi(3)).abs() < 1e-12, "{m:?}");
}

/// Open +x and top (bits 1 and 5).
const OPEN_POS_X_AND_TOP: u32 = 0b10_0010;

/// 2b-3c spec §6: outflow is counted through a plane two cells in from
/// every open face, and the control volume stops at those planes. Each plane
/// spans only the control volume's cross-section, so a corner cell counts
/// once in the mass and once through each face it leaves by.
///
/// Density 1 fills x layers 13…15 and z layers 13…15 (N = 16), and the air
/// moves at 0.4 m/s along +x and +z. The +x plane (x-face 14) sees layer 13
/// over y 0…15, z 0…13: 224 cells. The top plane (z-face 14) sees layer 13
/// over x 0…13, y 0…15: 224 cells. The control volume, x < 14 and z < 14,
/// holds the 224 + 224 − 16 cells of the two inner layers.
#[test]
fn outflow_counts_every_open_face() {
    let mut d = vec![0.0; (N * N * N) as usize];
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                if i >= N - 3 || k >= N - 3 {
                    d[idx(i, j, k)] = 1.0;
                }
            }
        }
    }
    let v = faces(|_| [0.4, 0.0, 0.4]);
    let m = measure(&Sample {
        open_mask: OPEN_POS_X_AND_TOP,
        ..sample(&d, &v, &[])
    });
    let expected = (224.0 + 224.0) * 0.4 * DX * DX;
    // 0.4 is an f32 in the faces: 0.4000000059.
    assert!((m.outflow_rate - expected).abs() < 1e-6, "{m:?}");
    let filled = 3.0 * 256.0 + 3.0 * 256.0 - 9.0 * 16.0;
    assert!((m.mass - filled * DX.powi(3)).abs() < 1e-12, "{m:?}");
    assert!(
        (m.mass_inside - (224.0 + 224.0 - 16.0) * DX.powi(3)).abs() < 1e-12,
        "{m:?}"
    );
}

/// Open −x and top (bits 0 and 5).
const OPEN_NEG_X_AND_TOP: u32 = 0b10_0001;

/// A low face's plane is x-face 2, and flow towards −x is outflow: upwind
/// there is cell 2, inside; cell 1, beyond the plane, is not counted.
#[test]
fn outflow_through_a_low_face_is_positive_outwards() {
    let mut d = vec![0.0; (N * N * N) as usize];
    for k in 0..N {
        for j in 0..N {
            d[idx(1, j, k)] = 5.0; // beyond the plane: not upwind, not inside
            d[idx(2, j, k)] = 1.0; // inside, next to the plane: leaves
        }
    }
    let v = faces(|_| [-0.4, 0.0, 0.0]);
    let m = measure(&Sample {
        open_mask: OPEN_NEG_X_AND_TOP,
        ..sample(&d, &v, &[])
    });
    // y 0…15, z 0…13 below the top plane.
    let expected = 16.0 * 14.0 * 0.4 * DX * DX;
    // 0.4 is an f32 in the faces: 0.4000000059.
    assert!((m.outflow_rate - expected).abs() < 1e-6, "{m:?}");
    assert!(
        (m.mass_inside - 16.0 * 14.0 * DX.powi(3)).abs() < 1e-12,
        "{m:?}"
    );
}

/// Mass 10, falling to 8 while 2 flows out: no drift.
#[test]
fn drift_counts_outflow_as_accounted_for() {
    let mass = [10.0, 9.0, 8.0];
    let outflow = [1.0, 1.0, 1.0]; // per second
    let d = drift(&mass, &outflow, 1.0, 0);
    assert_eq!(d.len(), 3);
    for x in d {
        assert!(x.abs() < 1e-12, "{x}");
    }
}

#[test]
fn drift_integrates_outflow_with_the_trapezoid_rule() {
    let mass = [10.0, 10.0];
    let outflow = [0.0, 2.0];
    // Trapezoid over one second: (0 + 2) / 2 = 1, so drift = 10 + 1 − 10.
    assert_eq!(drift(&mass, &outflow, 1.0, 0), vec![0.0, 1.0]);
}

#[test]
fn drift_is_zero_before_its_start() {
    let d = drift(&[5.0, 6.0, 7.0], &[0.0; 3], 1.0, 1);
    assert_eq!(d, vec![0.0, 0.0, 1.0]);
}

#[test]
fn fire_metrics_sum_fuel_and_count_flame_cells() {
    use elements_ember::metrics::{FLAME_THRESHOLD, fire_metrics};
    let fuel = [1.0, 2.0, 0.0, 0.5];
    let flame = [0.0, FLAME_THRESHOLD, 0.5, 0.02];
    let (mass, volume) = fire_metrics(&fuel, &flame, 0.5);
    assert_eq!(mass, 3.5 * 0.125);
    // Strictly above the threshold: two cells.
    assert_eq!(volume, 2.0 * 0.125);
}
