mod common;

use common::*;
use elements_core::gpu::{ComputeBatch, FieldDims, FieldPool, PipelineCache};
use elements_ember::kernels::{StepConstants, Uniforms, blend_velocity, buoyancy, emit, wind};

const CELLS: FieldDims = FieldDims { x: 8, y: 6, z: 5 };

fn constants() -> StepConstants {
    StepConstants {
        alpha: 0.5,
        beta: 2.0,
        ..StepConstants::new(CELLS, 0.25, 0.125)
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
    buoyancy(
        &gpu,
        &mut cache,
        &mut batch,
        &u,
        &w,
        &density,
        &temperature,
        None,
    )
    .unwrap();
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

/// Spec §3.3: where the weight is w, still fluid approaches the target as
/// 1 − exp(−w·t), independent of the substep. Wall faces are untouched.
#[test]
fn velocity_emission_approaches_the_target_exponentially() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = StepConstants::new(CELLS, 0.1, 0.125);
    let zero: [Vec<f32>; 3] = std::array::from_fn(|a| vec![0.0; face_dims(CELLS, a).voxel_count()]);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &zero);
    let weight = upload(&gpu, &mut pool, CELLS, &vec![2.0; CELLS.voxel_count()]);
    let target_faces: [Vec<f32>; 3] = std::array::from_fn(|a| {
        vec![if a == 0 { 1.0 } else { 0.0 }; face_dims(CELLS, a).voxel_count()]
    });
    let target = upload_staggered(&gpu, &mut pool, CELLS, &target_faces);
    let u = Uniforms::new(&gpu, &c).unwrap();
    for _ in 0..3 {
        let mut batch = ComputeBatch::new();
        blend_velocity(
            &gpu, &mut cache, &mut batch, &u, &velocity, &weight, &target, None,
        )
        .unwrap();
        batch.submit(&gpu).unwrap();
    }
    let got = read_staggered(&gpu, &velocity);
    let want = 1.0 - (-2.0f32 * 0.3).exp();
    let d = face_dims(CELLS, 0);
    for k in 0..d.z {
        for j in 0..d.y {
            for i in 0..d.x {
                let v = got[0][index(d, i, j, k)];
                if is_wall(CELLS, 0, i) {
                    assert_eq!(v, 0.0, "wall x face {i}");
                } else {
                    assert!(
                        (v - want).abs() <= 1e-5,
                        "x face {:?}: {v} vs {want}",
                        [i, j, k]
                    );
                }
            }
        }
    }
    assert!(got[1].iter().chain(&got[2]).all(|&v| v == 0.0));
}

/// Spec §3.4: wind is a uniform acceleration on every non-wall face.
#[test]
fn wind_accelerates_every_non_wall_face_uniformly() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = StepConstants {
        wind: [0.5, 0.0, -1.0],
        ..StepConstants::new(CELLS, 0.1, 0.125)
    };
    let zero: [Vec<f32>; 3] = std::array::from_fn(|a| vec![0.0; face_dims(CELLS, a).voxel_count()]);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &zero);
    let u = Uniforms::new(&gpu, &c).unwrap();
    for _ in 0..4 {
        let mut batch = ComputeBatch::new();
        wind(&gpu, &mut cache, &mut batch, &u, &velocity, None).unwrap();
        batch.submit(&gpu).unwrap();
    }
    let got = read_staggered(&gpu, &velocity);
    for (a, rate) in [(0usize, 0.5f32), (1, 0.0), (2, -1.0)] {
        let d = face_dims(CELLS, a);
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let want = if is_wall(CELLS, a, [i, j, k][a]) {
                        0.0
                    } else {
                        4.0 * 0.1 * rate
                    };
                    let v = got[a][index(d, i, j, k)];
                    assert!(
                        (v - want).abs() <= 1e-6,
                        "face {a} {:?}: {v} vs {want}",
                        [i, j, k]
                    );
                }
            }
        }
    }
}
