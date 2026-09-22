mod common;

use common::*;
use elements_core::gpu::{ComputeBatch, FieldDims, FieldPool, PipelineCache};
use elements_ember::kernels::{StepConstants, Uniforms, buoyancy, emit};

const CELLS: FieldDims = FieldDims { x: 8, y: 6, z: 5 };

fn constants() -> StepConstants {
    StepConstants {
        cells: CELLS,
        h: 0.25,
        dx: 0.125,
        alpha: 0.5,
        beta: 2.0,
    }
}

#[test]
fn emit_adds_the_source_rate_times_the_substep() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let before = pattern(CELLS, 3);
    let rate = pattern(CELLS, 4);
    let dst = upload(&gpu, &mut pool, CELLS, &before);
    let src = upload(&gpu, &mut pool, CELLS, &rate);

    let u = Uniforms::new(&gpu, &constants()).unwrap();
    let mut batch = ComputeBatch::new();
    emit(&gpu, &mut cache, &mut batch, &u, &dst, &src).unwrap();
    batch.submit(&gpu).unwrap();

    let want: Vec<f32> = before
        .iter()
        .zip(&rate)
        .map(|(b, r)| b + r * 0.25)
        .collect();
    assert_close(&dst.read_back(&gpu).unwrap(), &want, 1e-6, "emitted");
}

#[test]
fn buoyancy_matches_the_cpu_reference_and_leaves_boundary_faces_alone() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = constants();
    let zd = face_dims(CELLS, 2);
    let w_before = pattern(zd, 5);
    let rho = pattern(CELLS, 6);
    let temp = pattern(CELLS, 7);
    let w = upload(&gpu, &mut pool, zd, &w_before);
    let density = upload(&gpu, &mut pool, CELLS, &rho);
    let temperature = upload(&gpu, &mut pool, CELLS, &temp);

    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    buoyancy(&gpu, &mut cache, &mut batch, &u, &w, &density, &temperature).unwrap();
    batch.submit(&gpu).unwrap();

    let mut want = w_before.clone();
    for k in 1..CELLS.z {
        for j in 0..CELLS.y {
            for i in 0..CELLS.x {
                let r = 0.5 * (rho[index(CELLS, i, j, k - 1)] + rho[index(CELLS, i, j, k)]);
                let t = 0.5 * (temp[index(CELLS, i, j, k - 1)] + temp[index(CELLS, i, j, k)]);
                want[index(zd, i, j, k)] += c.h * (c.beta * t - c.alpha * r);
            }
        }
    }
    let got = w.read_back(&gpu).unwrap();
    assert_close(&got, &want, 1e-5, "w after buoyancy");
    for j in 0..CELLS.y {
        for i in 0..CELLS.x {
            for k in [0, CELLS.z] {
                assert_eq!(
                    got[index(zd, i, j, k)].to_bits(),
                    w_before[index(zd, i, j, k)].to_bits(),
                    "boundary face ({i}, {j}, {k}) must be untouched"
                );
            }
        }
    }
}
