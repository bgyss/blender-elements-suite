mod common;

use common::*;
use elements_core::gpu::{ComputeBatch, FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_ember::kernels::{StepConstants, Uniforms, divergence, pressure, subtract_gradient};

const CELLS: FieldDims = FieldDims { x: 8, y: 6, z: 5 };

fn constants(dx: f32) -> StepConstants {
    StepConstants {
        cells: CELLS,
        h: 0.1,
        dx,
        alpha: 0.0,
        beta: 0.0,
    }
}

/// A velocity equal to position along one axis has divergence 1 everywhere.
/// Each axis is checked on its own, so a swapped offset cannot hide.
#[test]
fn a_linear_velocity_has_unit_divergence_along_each_axis() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    let dx = 0.125;
    for axis in 0..3 {
        let mut pool = FieldPool::new();
        let faces: [Vec<f32>; 3] = std::array::from_fn(|a| {
            let d = face_dims(CELLS, a);
            let mut face = vec![0.0; d.voxel_count()];
            if a == axis {
                for k in 0..d.z {
                    for j in 0..d.y {
                        for i in 0..d.x {
                            // Face position along its own axis, in metres.
                            face[index(d, i, j, k)] = [i, j, k][a] as f32 * dx;
                        }
                    }
                }
            }
            face
        });
        let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
        let div = pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap();
        let u = Uniforms::new(&gpu, &constants(dx)).unwrap();
        let mut batch = ComputeBatch::new();
        divergence(&gpu, &mut cache, &mut batch, &u, &velocity, &div).unwrap();
        batch.submit(&gpu).unwrap();
        let got = div.read_back(&gpu).unwrap();
        assert!(
            got.iter().all(|&v| v == 1.0),
            "axis {axis}: {:?}",
            &got[..4]
        );
    }
}

#[test]
fn red_black_sweeps_match_the_cpu_reference() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = constants(1.0);
    let div_values = pattern(CELLS, 8);
    let phi0 = pattern(CELLS, 9);
    let div = upload(&gpu, &mut pool, CELLS, &div_values);
    let phi = upload(&gpu, &mut pool, CELLS, &phi0);

    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    pressure(&gpu, &mut cache, &mut batch, &u, &phi, &div, 3).unwrap();
    batch.submit(&gpu).unwrap();

    let mut want = phi0.clone();
    cpu_red_black(&mut want, &div_values, CELLS, 1.0, 3);
    assert_close(&phi.read_back(&gpu).unwrap(), &want, 1e-4, "phi");
}

#[test]
fn subtracting_the_gradient_zeroes_solid_walls_and_matches_the_cpu() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let dx = 0.125;
    let faces = velocity_pattern(CELLS);
    let phi_values = pattern(CELLS, 10);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let phi = upload(&gpu, &mut pool, CELLS, &phi_values);

    let u = Uniforms::new(&gpu, &constants(dx)).unwrap();
    let mut batch = ComputeBatch::new();
    subtract_gradient(&gpu, &mut cache, &mut batch, &u, &velocity, &phi).unwrap();
    batch.submit(&gpu).unwrap();

    let got = read_staggered(&gpu, &velocity);
    for a in 0..3 {
        let d = face_dims(CELLS, a);
        let n = [CELLS.x, CELLS.y, CELLS.z][a];
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let ijk = [i, j, k];
                    let at = index(d, i, j, k);
                    if is_wall(CELLS, a, ijk[a]) {
                        assert_eq!(got[a][at], 0.0, "wall face {a} {ijk:?}");
                        continue;
                    }
                    let upper = if ijk[a] < n {
                        phi_values[index(CELLS, i, j, k)]
                    } else {
                        0.0
                    };
                    let mut low = ijk;
                    low[a] -= 1;
                    let lower = phi_values[index(CELLS, low[0], low[1], low[2])];
                    let want = faces[a][at] - (upper - lower) / dx;
                    assert!(
                        (got[a][at] - want).abs() <= 1e-4,
                        "face {a} {ijk:?}: {} vs {want}",
                        got[a][at]
                    );
                }
            }
        }
    }
}

/// Divergence, a converged solve, then the gradient: the result is
/// divergence-free. This is what ties the stencil and the gradient together:
/// each is tested against its own reference above, but only this shows they
/// agree with each other at the walls and the open top.
#[test]
fn a_converged_projection_removes_divergence() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let dx = 0.125;
    let faces = walled_velocity_pattern(CELLS);
    let before = cpu_max_divergence(&faces, CELLS, dx);
    assert!(
        before > 1.0,
        "the test field must start divergent, got {before}"
    );
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let div = pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap();
    let phi = pool.acquire_zeroed(&gpu, &mut cache, CELLS).unwrap();

    let u = Uniforms::new(&gpu, &constants(dx)).unwrap();
    let mut batch = ComputeBatch::new();
    divergence(&gpu, &mut cache, &mut batch, &u, &velocity, &div).unwrap();
    pressure(&gpu, &mut cache, &mut batch, &u, &phi, &div, 2000).unwrap();
    subtract_gradient(&gpu, &mut cache, &mut batch, &u, &velocity, &phi).unwrap();
    batch.submit(&gpu).unwrap();

    let after = cpu_max_divergence(&read_staggered(&gpu, &velocity), CELLS, dx);
    assert!(after < 1e-3 * before, "max |div| {before} -> {after}");
}
