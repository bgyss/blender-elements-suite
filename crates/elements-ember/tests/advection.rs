mod common;

use common::*;
use elements_core::gpu::{
    ComputeBatch, Field, FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache,
    StaggeredField,
};
use elements_ember::boundaries::DEFAULT_OPEN_MASK;
use elements_ember::kernels::{
    Advection, Carried, Pass, Solids, StepConstants, Uniforms, advect, maccormack,
};

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
    advect(
        &gpu,
        &mut cache,
        &mut batch,
        &u,
        Carried::Density,
        Pass::SemiLagrangian,
        &velocity,
        &src,
        &dst,
        None,
    )
    .unwrap();
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
    advect(
        &gpu,
        &mut cache,
        &mut batch,
        &u,
        Carried::Density,
        Pass::SemiLagrangian,
        &velocity,
        &src,
        &dst,
        None,
    )
    .unwrap();
    batch.submit(&gpu).unwrap();

    let want = cpu_advect(
        &faces,
        CELLS,
        DEFAULT_OPEN_MASK,
        Grid::Cell,
        &src_values,
        c.h / c.dx,
        1.0,
    );
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
    for (a, axis) in AXES.iter().enumerate() {
        advect(
            &gpu,
            &mut cache,
            &mut batch,
            &u,
            Carried::Face(*axis),
            Pass::SemiLagrangian,
            &src,
            src.face(*axis),
            dst.face(*axis),
            None,
        )
        .unwrap();
        let _ = a;
    }
    batch.submit(&gpu).unwrap();

    let got = read_staggered(&gpu, &dst);
    for (a, _axis) in AXES.iter().enumerate() {
        let want = cpu_advect(
            &faces,
            CELLS,
            DEFAULT_OPEN_MASK,
            Grid::Face(a),
            &faces[a],
            c.h / c.dx,
            1.0,
        );
        assert_close(&got[a], &want, 1e-5, &format!("face {a}"));
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
    advect(
        &gpu,
        &mut cache,
        &mut batch,
        &u,
        Carried::Density,
        Pass::SemiLagrangian,
        &velocity,
        &src,
        &dst,
        None,
    )
    .unwrap();
    batch.submit(&gpu).unwrap();

    let got = dst.read_back(&gpu).unwrap();
    let want = cpu_advect(&faces, CELLS, mask, Grid::Cell, &ones, c.h / c.dx, 1.0);
    assert_close(&got, &want, 1e-6, "advected");
    for k in 0..CELLS.z {
        for j in 0..CELLS.y {
            assert_eq!(got[index(CELLS, 0, j, k)], 0.5, "first column ({j}, {k})");
            assert_eq!(got[index(CELLS, 1, j, k)], 1.0, "second column ({j}, {k})");
        }
    }
}

/// Spec §4.1: the backtrace is an RK2 midpoint step. A solid-body rotation
/// about the domain's vertical centre line curves every path, so an Euler
/// step lands measurably elsewhere.
#[test]
fn rk2_follows_a_rotation_like_the_cpu_reference() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(12, 10, 5);
    let c = StepConstants::new(cells, 0.1, 0.125);
    let omega = 2.0; // rad/s: 0.2 rad per step
    let (cx, cy) = (6.0 * c.dx, 5.0 * c.dx);
    let faces: [Vec<f32>; 3] = std::array::from_fn(|a| {
        let d = face_dims(cells, a);
        let off = face_offset(a);
        let mut face = vec![0.0; d.voxel_count()];
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let x = (i as f32 + off[0]) * c.dx;
                    let y = (j as f32 + off[1]) * c.dx;
                    face[index(d, i, j, k)] = match a {
                        0 => -omega * (y - cy),
                        1 => omega * (x - cx),
                        _ => 0.0,
                    };
                }
            }
        }
        face
    });
    let src_values = pattern(cells, 21);
    let velocity = upload_staggered(&gpu, &mut pool, cells, &faces);
    let src = upload(&gpu, &mut pool, cells, &src_values);
    let dst = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    advect(
        &gpu,
        &mut cache,
        &mut batch,
        &u,
        Carried::Density,
        Pass::SemiLagrangian,
        &velocity,
        &src,
        &dst,
        None,
    )
    .unwrap();
    batch.submit(&gpu).unwrap();
    let want = cpu_advect(
        &faces,
        cells,
        DEFAULT_OPEN_MASK,
        Grid::Cell,
        &src_values,
        c.h / c.dx,
        1.0,
    );
    assert_close(&dst.read_back(&gpu).unwrap(), &want, 1e-5, "rotated");
}

/// Record one MacCormack step of `src` (on `carried`'s grid) into `dst`,
/// with pooled scratch for the forward and backward passes.
#[allow(clippy::too_many_arguments)]
fn maccormack_step(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    pool: &mut FieldPool,
    u: &Uniforms,
    carried: Carried,
    velocity: &StaggeredField,
    src: &Field,
    dst: &Field,
) {
    let fwd = pool
        .acquire(gpu, src.dims(), FieldFormat::R32Float)
        .unwrap();
    let bwd = pool
        .acquire(gpu, src.dims(), FieldFormat::R32Float)
        .unwrap();
    let mut batch = ComputeBatch::new();
    advect(
        gpu,
        cache,
        &mut batch,
        u,
        carried,
        Pass::Forward,
        velocity,
        src,
        &fwd,
        None,
    )
    .unwrap();
    advect(
        gpu,
        cache,
        &mut batch,
        u,
        carried,
        Pass::Backward,
        velocity,
        &fwd,
        &bwd,
        None,
    )
    .unwrap();
    maccormack(
        gpu, cache, &mut batch, u, carried, velocity, src, &fwd, &bwd, dst, None,
    )
    .unwrap();
    batch.submit(gpu).unwrap();
    pool.release(fwd);
    pool.release(bwd);
}

#[test]
fn maccormack_matches_the_cpu_reference() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = constants(0.2, 0.125);
    let faces = walled_velocity_pattern(CELLS);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let u = Uniforms::new(&gpu, &c).unwrap();
    let k = c.h / c.dx;

    let src_values = pattern(CELLS, 22);
    let src = upload(&gpu, &mut pool, CELLS, &src_values);
    let dst = pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap();
    maccormack_step(
        &gpu,
        &mut cache,
        &mut pool,
        &u,
        Carried::Density,
        &velocity,
        &src,
        &dst,
    );
    let want = cpu_maccormack(
        &faces,
        CELLS,
        DEFAULT_OPEN_MASK,
        Grid::Cell,
        &src_values,
        k,
        1.0,
    );
    assert_close(&dst.read_back(&gpu).unwrap(), &want, 1e-4, "scalar");

    for (a, axis) in AXES.iter().enumerate() {
        let d = face_dims(CELLS, a);
        let dst = pool.acquire(&gpu, d, FieldFormat::R32Float).unwrap();
        maccormack_step(
            &gpu,
            &mut cache,
            &mut pool,
            &u,
            Carried::Face(*axis),
            &velocity,
            velocity.face(*axis),
            &dst,
        );
        let want = cpu_maccormack(
            &faces,
            CELLS,
            DEFAULT_OPEN_MASK,
            Grid::Face(a),
            &faces[a],
            k,
            1.0,
        );
        assert_close(
            &dst.read_back(&gpu).unwrap(),
            &want,
            1e-4,
            &format!("face {a}"),
        );
    }
}

/// Advect `values` along +x at 0.3 cells a step for `steps` steps, in a
/// 32×4×4 domain, with either scheme.
fn carry_along_x(values: &[f32], steps: u32, advection: Advection) -> Vec<f32> {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(32, 4, 4);
    let c = StepConstants::new(cells, 0.1, 0.125);
    // 0.375 m/s × 0.1 s / 0.125 m = 0.3 cells.
    let faces: [Vec<f32>; 3] = std::array::from_fn(|a| {
        vec![if a == 0 { 0.375 } else { 0.0 }; face_dims(cells, a).voxel_count()]
    });
    let velocity = upload_staggered(&gpu, &mut pool, cells, &faces);
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut q = upload(&gpu, &mut pool, cells, values);
    for _ in 0..steps {
        let next = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
        match advection {
            Advection::SemiLagrangian => {
                let mut batch = ComputeBatch::new();
                advect(
                    &gpu,
                    &mut cache,
                    &mut batch,
                    &u,
                    Carried::Density,
                    Pass::SemiLagrangian,
                    &velocity,
                    &q,
                    &next,
                    None,
                )
                .unwrap();
                batch.submit(&gpu).unwrap();
            }
            Advection::MacCormack => {
                maccormack_step(
                    &gpu,
                    &mut cache,
                    &mut pool,
                    &u,
                    Carried::Density,
                    &velocity,
                    &q,
                    &next,
                );
            }
        }
        pool.release(std::mem::replace(&mut q, next));
    }
    q.read_back(&gpu).unwrap()
}

/// Spec §4.1: MacCormack is why 2b-1 exists (Mantaflow uses it). A smooth
/// bump carried 6 cells keeps a clearly higher peak than semi-Lagrangian.
#[test]
fn maccormack_keeps_a_bump_sharper_than_semi_lagrangian() {
    let cells = FieldDims::new(32, 4, 4);
    let mut bump = vec![0.0; cells.voxel_count()];
    for k in 0..4 {
        for j in 0..4 {
            for i in 0..32 {
                let d = i as f32 - 10.0;
                bump[index(cells, i, j, k)] = (-d * d / 8.0).exp();
            }
        }
    }
    let peak = |v: &[f32]| v.iter().copied().fold(0.0f32, f32::max);
    let semi = peak(&carry_along_x(&bump, 20, Advection::SemiLagrangian));
    let mac = peak(&carry_along_x(&bump, 20, Advection::MacCormack));
    assert!(
        mac >= semi + 0.05,
        "MacCormack peak {mac}, semi-Lagrangian {semi}"
    );
}

/// Spec §4.1: the clamp keeps MacCormack inside the range of what it
/// interpolated, so a step never overshoots.
#[test]
fn maccormack_never_leaves_the_source_range() {
    let cells = FieldDims::new(32, 4, 4);
    let mut step = vec![0.0; cells.voxel_count()];
    for k in 0..4 {
        for j in 0..4 {
            for i in 0..16 {
                step[index(cells, i, j, k)] = 1.0;
            }
        }
    }
    for v in carry_along_x(&step, 5, Advection::MacCormack) {
        assert!((0.0..=1.0).contains(&v), "left [0, 1]: {v}");
    }
}

/// Spec §4.5: dissipation multiplies by exp(−rate·h) in the last pass, and
/// only for the scalar it is set on.
#[test]
fn dissipation_decays_by_exp_of_rate_times_h() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    for advection in [Advection::SemiLagrangian, Advection::MacCormack] {
        let mut pool = FieldPool::new();
        let c = StepConstants {
            density_dissipation: 2.0,
            ..constants(0.25, 0.125)
        };
        let zero: [Vec<f32>; 3] =
            std::array::from_fn(|a| vec![0.0; face_dims(CELLS, a).voxel_count()]);
        let velocity = upload_staggered(&gpu, &mut pool, CELLS, &zero);
        let u = Uniforms::new(&gpu, &c).unwrap();
        let values = pattern(CELLS, 23);
        let src = upload(&gpu, &mut pool, CELLS, &values);
        for (carried, rate) in [(Carried::Density, 2.0f32), (Carried::Temperature, 0.0)] {
            let dst = pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap();
            match advection {
                Advection::SemiLagrangian => {
                    let mut batch = ComputeBatch::new();
                    advect(
                        &gpu,
                        &mut cache,
                        &mut batch,
                        &u,
                        carried,
                        Pass::SemiLagrangian,
                        &velocity,
                        &src,
                        &dst,
                        None,
                    )
                    .unwrap();
                    batch.submit(&gpu).unwrap();
                }
                Advection::MacCormack => {
                    maccormack_step(
                        &gpu, &mut cache, &mut pool, &u, carried, &velocity, &src, &dst,
                    );
                }
            }
            let decay = (-rate * c.h).exp();
            let want: Vec<f32> = values.iter().map(|v| v * decay).collect();
            assert_close(
                &dst.read_back(&gpu).unwrap(),
                &want,
                1e-6,
                &format!("{advection:?} {carried:?}"),
            );
        }
    }
}

/// Spec §3.2 (as corrected): scalar sampling never reads a solid's contents.
/// Solid cells hold 100 and fluid cells 1. After a short advection step every
/// fluid cell still holds exactly 1, because solid corners are replaced by
/// the mean of the fluid corners.
#[test]
fn scalars_next_to_a_solid_never_sample_its_contents() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(12, 10, 8);
    // 0.6 m/s × 0.02 s / 0.125 m ≈ 0.1 cell, so every stencil keeps a fluid corner.
    let c = StepConstants {
        has_solids: true,
        open_mask: 0,
        ..StepConstants::new(cells, 0.02, 0.125)
    };
    let mask_values = block_mask(cells);
    let src_values: Vec<f32> = mask_values
        .iter()
        .map(|&m| if m > 0.5 { 100.0 } else { 1.0 })
        .collect();
    let mask = upload(&gpu, &mut pool, cells, &mask_values);
    let obstacle = pool
        .acquire_staggered_zeroed(&gpu, &mut cache, cells)
        .unwrap();
    let velocity = upload_staggered(&gpu, &mut pool, cells, &velocity_pattern(cells));
    let src = upload(&gpu, &mut pool, cells, &src_values);
    let dst = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    advect(
        &gpu,
        &mut cache,
        &mut batch,
        &u,
        Carried::Density,
        Pass::SemiLagrangian,
        &velocity,
        &src,
        &dst,
        Some(Solids {
            mask: &mask,
            velocity: &obstacle,
        }),
    )
    .unwrap();
    batch.submit(&gpu).unwrap();
    let got = dst.read_back(&gpu).unwrap();
    for (n, m) in mask_values.iter().enumerate() {
        if *m < 0.5 {
            assert!((got[n] - 1.0).abs() <= 1e-5, "fluid cell {n}: {}", got[n]);
        }
    }
}

/// One step of `src` along x at 2/3 of a cell, towards +x when `speed` is
/// 1 and −x when it is −1, in a 16×2×2 domain whose faces `open_mask`
/// names are open, by semi-Lagrangian and by MacCormack.
fn leave_along_x(src_values: &[f32], open_mask: u32, speed: f32) -> (Vec<f32>, Vec<f32>) {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(16, 2, 2);
    // 1 m/s × (1/24) s / (1/16) m = 2/3 of a cell.
    let c = StepConstants {
        open_mask,
        ..StepConstants::new(cells, 1.0 / 24.0, 1.0 / 16.0)
    };
    let faces: [Vec<f32>; 3] = std::array::from_fn(|a| {
        let n = face_dims(cells, a).voxel_count();
        let mut v = vec![if a == 0 { speed } else { 0.0 }; n];
        // A wall face carries no velocity, as projection would leave it.
        if a == 0 {
            for k in 0..cells.z {
                for j in 0..cells.y {
                    for i in [0, cells.x] {
                        if is_wall_in(cells, open_mask, 0, i) {
                            v[index(face_dims(cells, 0), i, j, k)] = 0.0;
                        }
                    }
                }
            }
        }
        v
    });
    let velocity = upload_staggered(&gpu, &mut pool, cells, &faces);
    let u = Uniforms::new(&gpu, &c).unwrap();
    let src = upload(&gpu, &mut pool, cells, src_values);
    let semi = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let mut batch = ComputeBatch::new();
    advect(
        &gpu,
        &mut cache,
        &mut batch,
        &u,
        Carried::Density,
        Pass::SemiLagrangian,
        &velocity,
        &src,
        &semi,
        None,
    )
    .unwrap();
    batch.submit(&gpu).unwrap();
    let mac = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    maccormack_step(
        &gpu,
        &mut cache,
        &mut pool,
        &u,
        Carried::Density,
        &velocity,
        &src,
        &mac,
    );
    (semi.read_back(&gpu).unwrap(), mac.read_back(&gpu).unwrap())
}

/// 2b-3c spec §5: MacCormack's backward pass reads ambient 0 beyond an
/// open face, and its correction then put back part of what flowed out, so
/// smoke piled up against open faces. Where a trace crosses an open face,
/// MacCormack now takes the first-order value there. Walls and the
/// interior keep the corrected value, and MacCormack is not conservative in
/// the interior either, so a blob leaving the domain loses about, not
/// exactly, what semi-Lagrangian loses: 1.494 against 1.617, where before
/// it lost 1.126.
#[test]
fn maccormack_falls_back_to_first_order_where_a_trace_crosses_an_open_face() {
    let cells = FieldDims::new(16, 2, 2);
    // A blob whose peak sits two cells inside the +x face.
    let blob: Vec<f32> = (0..cells.voxel_count())
        .map(|n| {
            let x = (n % 16) as f32 + 0.5;
            (-(x - 13.5).powi(2) / 8.0).exp()
        })
        .collect();
    let total = |v: &[f32]| v.iter().map(|&q| f64::from(q)).sum::<f64>();
    let column = |v: &[f32], i: u32| -> Vec<f32> {
        (0..4).map(|r| v[index(cells, i, r % 2, r / 2)]).collect()
    };

    let (semi, mac) = leave_along_x(&blob, 0b10, 1.0);
    let lost_semi = total(&blob) - total(&semi);
    let lost_mac = total(&blob) - total(&mac);
    eprintln!("+x open: lost by semi-Lagrangian {lost_semi}, by MacCormack {lost_mac}");
    // The last column's backward trace leaves through the open face.
    assert_eq!(
        column(&mac, 15),
        column(&semi, 15),
        "the open face's column"
    );
    // Well inside, MacCormack still corrects.
    assert_ne!(column(&mac, 12), column(&semi, 12), "an interior column");
    assert!(
        lost_mac >= 0.9 * lost_semi,
        "MacCormack lost {lost_mac}, semi-Lagrangian {lost_semi}"
    );

    // Against a wall nothing crosses an open face, and the corrected value
    // stays.
    let (semi, mac) = leave_along_x(&blob, 0, 1.0);
    assert_ne!(column(&mac, 14), column(&semi, 14), "beside a wall");

    // The same through the low face: the blob mirrored, moving −x.
    let mirrored: Vec<f32> = (0..cells.voxel_count())
        .map(|n| blob[n - n % 16 + 15 - n % 16])
        .collect();
    let (semi, mac) = leave_along_x(&mirrored, 0b01, -1.0);
    assert_eq!(column(&mac, 0), column(&semi, 0), "the −x face's column");
    assert_ne!(column(&mac, 3), column(&semi, 3), "an interior column, −x");

    // Inflow: clean air enters through the open −x face beside the
    // mirrored blob. The first column's forward trace reaches past that
    // face, where the correction would otherwise draw on the ambient ghost.
    let (semi, mac) = leave_along_x(&mirrored, 0b01, 1.0);
    assert_eq!(column(&mac, 0), column(&semi, 0), "the inflow column");
}
