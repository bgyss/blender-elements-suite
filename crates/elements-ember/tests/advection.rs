mod common;

use common::*;
use elements_core::gpu::{ComputeBatch, FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_ember::boundaries::DEFAULT_OPEN_MASK;
use elements_ember::kernels::{StepConstants, Uniforms, advect_scalar, advect_velocity};

const CELLS: FieldDims = FieldDims { x: 8, y: 6, z: 5 };

fn constants(h: f32, dx: f32) -> StepConstants {
    StepConstants::new(CELLS, h, dx)
}

/// A uniform velocity of exactly two voxels per substep along x moves a
/// scalar by exactly two cells: every interpolation weight is 0 or 1.
#[test]
fn an_integer_uniform_velocity_shifts_a_scalar_exactly() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    // Powers of two, so u * h / dx is exactly 2.
    let c = constants(0.25, 0.125);
    let faces = [
        vec![1.0; face_dims(CELLS, 0).voxel_count()],
        vec![0.0; face_dims(CELLS, 1).voxel_count()],
        vec![0.0; face_dims(CELLS, 2).voxel_count()],
    ];
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let src_values = pattern(CELLS, 1);
    let src = upload(&gpu, &mut pool, CELLS, &src_values);
    let dst = pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap();

    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    advect_scalar(&gpu, &mut cache, &mut batch, &u, &velocity, &src, &dst).unwrap();
    batch.submit(&gpu).unwrap();

    let out = dst.read_back(&gpu).unwrap();
    for k in 0..CELLS.z {
        for j in 0..CELLS.y {
            for i in 0..CELLS.x {
                // Backtraces left of the domain clamp to its first column.
                let from = i.saturating_sub(2);
                assert_eq!(
                    out[index(CELLS, i, j, k)].to_bits(),
                    src_values[index(CELLS, from, j, k)].to_bits(),
                    "cell ({i}, {j}, {k})"
                );
            }
        }
    }
}

#[test]
fn a_fractional_velocity_advects_a_scalar_like_the_cpu_reference() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = constants(0.1, 0.125);
    let faces = velocity_pattern(CELLS);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let src_values = pattern(CELLS, 2);
    let src = upload(&gpu, &mut pool, CELLS, &src_values);
    let dst = pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap();

    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    advect_scalar(&gpu, &mut cache, &mut batch, &u, &velocity, &src, &dst).unwrap();
    batch.submit(&gpu).unwrap();

    let want = cpu_advect_scalar(&faces, CELLS, DEFAULT_OPEN_MASK, &src_values, c.h / c.dx);
    assert_close(
        &dst.read_back(&gpu).unwrap(),
        &want,
        1e-5,
        "advected scalar",
    );
}

#[test]
fn velocity_advection_matches_the_cpu_reference_and_zeroes_solid_walls() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = constants(0.1, 0.125);
    let faces = velocity_pattern(CELLS);
    let src = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let dst = pool.acquire_staggered_uninit(&gpu, CELLS).unwrap();

    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    advect_velocity(&gpu, &mut cache, &mut batch, &u, &src, &dst).unwrap();
    batch.submit(&gpu).unwrap();

    let got = read_staggered(&gpu, &dst);
    let want = cpu_advect_velocity(&faces, CELLS, DEFAULT_OPEN_MASK, c.h / c.dx);
    for a in 0..3 {
        assert_close(&got[a], &want[a], 1e-5, &format!("face {a}"));
    }
    // The open top is advected, not zeroed: the pattern makes it nonzero.
    let d = face_dims(CELLS, 2);
    let top: Vec<f32> = (0..CELLS.y)
        .flat_map(|j| (0..CELLS.x).map(move |i| (i, j)))
        .map(|(i, j)| got[2][index(d, i, j, CELLS.z)])
        .collect();
    assert!(
        top.iter().any(|&v| v != 0.0),
        "the open top must not be a wall"
    );
}

/// Spec §4.2, risk (d): a scalar drawn in through an open face is clean
/// air. Only `-x` is open, and a uniform +x velocity of half a cell per step
/// pulls air in through it. The first column mixes half ambient 0 with half
/// its own value; with a clamp it would stay 1.
#[test]
fn inflow_across_an_open_face_carries_clean_air() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let mask = 0b000001;
    let c = StepConstants {
        open_mask: mask,
        ..StepConstants::new(CELLS, 0.25, 0.125)
    };
    // 0.25 m/s × 0.25 s / 0.125 m = half a cell.
    let faces: [Vec<f32>; 3] = std::array::from_fn(|a| {
        vec![if a == 0 { 0.25 } else { 0.0 }; face_dims(CELLS, a).voxel_count()]
    });
    let ones = vec![1.0; CELLS.voxel_count()];
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let src = upload(&gpu, &mut pool, CELLS, &ones);
    let dst = pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap();
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    advect_scalar(&gpu, &mut cache, &mut batch, &u, &velocity, &src, &dst).unwrap();
    batch.submit(&gpu).unwrap();

    let got = dst.read_back(&gpu).unwrap();
    let want = cpu_advect_scalar(&faces, CELLS, mask, &ones, c.h / c.dx);
    assert_close(&got, &want, 1e-6, "advected");
    for k in 0..CELLS.z {
        for j in 0..CELLS.y {
            assert_eq!(got[index(CELLS, 0, j, k)], 0.5, "first column ({j}, {k})");
            assert_eq!(got[index(CELLS, 1, j, k)], 1.0, "second column ({j}, {k})");
        }
    }
}
