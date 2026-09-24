mod common;

use common::*;
use elements_core::gpu::{
    ComputeBatch, Field, FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache,
    StaggeredField,
};
use elements_ember::bench::GATE_RATIO;
use elements_ember::boundaries::DEFAULT_OPEN_MASK;
use elements_ember::collider::{ColliderFields, ColliderParams, fill_collider};
use elements_ember::emitter::{Sphere, fill_sphere};
use elements_ember::kernels::{Solids, StepConstants, Uniforms, solidify};
use elements_ember::metrics::{centroid_z, divergence};
use elements_ember::solver::{Emission, PressureSolve, SolverState, Sources, Substep, substep};
use elements_ember::transform::{Key, Shape, Transform};

#[test]
fn divergence_statistics_are_max_and_root_mean_square() {
    // Two cells along x with divergences 3 and 4.
    let cells = FieldDims::new(2, 1, 1);
    let faces = [vec![0.0, 3.0, 7.0], vec![0.0; 4], vec![0.0; 4]];
    let stats = divergence(&faces, cells, 1.0);
    assert_eq!(stats.max_abs, 4.0);
    assert!(
        (stats.rms - 12.5_f64.sqrt()).abs() < 1e-12,
        "rms {}",
        stats.rms
    );
}

#[test]
fn the_centroid_is_in_cell_units_at_cell_centres() {
    let cells = FieldDims::new(2, 2, 4);
    let mut density = vec![0.0; cells.voxel_count()];
    density[index(cells, 1, 0, 3)] = 2.0;
    density[index(cells, 0, 1, 1)] = 2.0;
    assert_eq!(centroid_z(&density, cells), Some(2.5));
    assert_eq!(centroid_z(&vec![0.0; cells.voxel_count()], cells), None);
}

/// RMS divergence after projection over before it, after 20 frames of the
/// 16³ plume with the given open faces.
fn projection_ratio(open_mask: u32) -> f64 {
    const ITERATIONS: PressureSolve = PressureSolve::GaussSeidel(160);
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(16, 16, 16);
    let dx = 2.0 / 16.0;
    let density_source = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let temperature_source = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let sphere = Sphere {
        center: [1.0, 1.0, 0.4],
        radius: 0.3,
        density_rate: 1.0,
        temperature_rate: 2.0,
    };
    fill_sphere(
        &gpu,
        &mut cache,
        &density_source,
        &temperature_source,
        &sphere,
        dx,
    )
    .unwrap();
    let sources = Sources::new(&density_source, &temperature_source);
    let constants = StepConstants {
        beta: 1.0,
        open_mask,
        ..StepConstants::new(cells, 1.0 / 24.0, dx)
    };
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
    for _ in 0..20 {
        substep(
            &gpu, &mut cache, &mut pool, &mut state, sources, &constants, ITERATIONS,
        )
        .unwrap();
    }

    let mut step = Substep::new(&gpu, &constants).unwrap();
    step.pre_projection(&gpu, &mut cache, &mut pool, &mut state, sources)
        .unwrap();
    step.submit(&gpu, &mut pool).unwrap();
    let before = divergence(&state.read_velocity(&gpu).unwrap(), cells, dx);

    let mut step = Substep::new(&gpu, &constants).unwrap();
    step.project(&gpu, &mut cache, &mut pool, &mut state, ITERATIONS, None)
        .unwrap();
    step.submit(&gpu, &mut pool).unwrap();
    let after = divergence(&state.read_velocity(&gpu).unwrap(), cells, dx);

    assert!(before.rms > 0.0, "the plume must be moving");
    after.rms / before.rms
}

/// Umbrella §6, spec §4.2: projection cuts RMS divergence to at most 10% of
/// its value before projection, at the default iteration count the gate chose.
#[test]
fn projection_leaves_at_most_a_tenth_of_the_divergence() {
    let ratio = projection_ratio(DEFAULT_OPEN_MASK);
    assert!(ratio <= 0.1, "ratio {ratio}");
}

/// Spec §4.2: with every face a wall, the same rule holds.
#[test]
fn a_closed_box_meets_the_same_divergence_rule() {
    let ratio = projection_ratio(0);
    assert!(ratio <= 0.1, "ratio {ratio}");
}

/// Umbrella §6: a hot blob rises, and its density centroid climbs strictly
/// every frame.
#[test]
fn a_hot_blob_rises_every_frame() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(16, 16, 24);
    let dx = 2.0 / 24.0;
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
    // The blob itself: density and temperature 1 inside a sphere, no emission.
    let blob = Sphere {
        center: [0.667, 0.667, 0.4],
        radius: 0.2,
        density_rate: 1.0,
        temperature_rate: 1.0,
    };
    fill_sphere(
        &gpu,
        &mut cache,
        &state.density,
        &state.temperature,
        &blob,
        dx,
    )
    .unwrap();
    let zero = pool.acquire_zeroed(&gpu, &mut cache, cells).unwrap();
    let sources = Sources::new(&zero, &zero);
    let constants = StepConstants {
        beta: 1.0,
        ..StepConstants::new(cells, 1.0 / 24.0, dx)
    };

    let mut heights = vec![centroid_z(&state.density.read_back(&gpu).unwrap(), cells).unwrap()];
    for _ in 0..20 {
        substep(
            &gpu,
            &mut cache,
            &mut pool,
            &mut state,
            sources,
            &constants,
            PressureSolve::GaussSeidel(80),
        )
        .unwrap();
        heights.push(centroid_z(&state.density.read_back(&gpu).unwrap(), cells).unwrap());
    }
    for (frame, pair) in heights.windows(2).enumerate() {
        assert!(
            pair[1] > pair[0],
            "frame {}: centroid {} -> {}; all: {heights:?}",
            frame + 1,
            pair[0],
            pair[1]
        );
    }
}

/// Spec §4.3, risk (b): the slot holds p, so after the substep length drops
/// threefold the warm start is still close, and one 20-iteration projection
/// still meets 2a's divergence rule. A relative bound against a no-switch
/// run can't hold: unconverged projections leave residual divergence that
/// does not scale with h, so any change of h costs something (measured on M1
/// Max: switched 0.062, same-history no-switch 0.024, storing h·p instead
/// 0.20).
#[test]
fn the_warm_start_survives_a_change_of_substep_length() {
    const ITERATIONS: PressureSolve = PressureSolve::GaussSeidel(20);
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(16, 16, 16);
    let dx = 2.0 / 16.0;
    let density_source = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let temperature_source = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let sphere = Sphere {
        center: [1.0, 1.0, 0.4],
        radius: 0.3,
        density_rate: 1.0,
        temperature_rate: 2.0,
    };
    fill_sphere(
        &gpu,
        &mut cache,
        &density_source,
        &temperature_source,
        &sphere,
        dx,
    )
    .unwrap();
    let sources = Sources::new(&density_source, &temperature_source);
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
    let warm = StepConstants {
        beta: 1.0,
        ..StepConstants::new(cells, 1.0 / 24.0, dx)
    };
    for _ in 0..24 {
        substep(
            &gpu, &mut cache, &mut pool, &mut state, sources, &warm, ITERATIONS,
        )
        .unwrap();
    }
    let last = StepConstants {
        beta: 1.0,
        ..StepConstants::new(cells, 1.0 / 72.0, dx)
    };
    let mut step = Substep::new(&gpu, &last).unwrap();
    step.pre_projection(&gpu, &mut cache, &mut pool, &mut state, sources)
        .unwrap();
    step.submit(&gpu, &mut pool).unwrap();
    let before = divergence(&state.read_velocity(&gpu).unwrap(), cells, dx);
    let mut step = Substep::new(&gpu, &last).unwrap();
    step.project(&gpu, &mut cache, &mut pool, &mut state, ITERATIONS, None)
        .unwrap();
    step.submit(&gpu, &mut pool).unwrap();
    let after = divergence(&state.read_velocity(&gpu).unwrap(), cells, dx);
    let switched = after.rms / before.rms;
    assert!(switched <= GATE_RATIO, "switched {switched}");
}

/// Spec §4.4: confinement strengthens a plume's vorticity. Twenty frames of
/// the 16³ plume with ε = 4 end with clearly more total |ω| than with ε = 0.
#[test]
fn confinement_strengthens_a_plumes_vorticity() {
    fn total_vorticity(vorticity: f32) -> f64 {
        let gpu = gpu();
        let mut pool = FieldPool::new();
        let mut cache = PipelineCache::new();
        let cells = FieldDims::new(16, 16, 16);
        let dx = 2.0 / 16.0;
        let density_source = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
        let temperature_source = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
        let sphere = Sphere {
            center: [1.0, 1.0, 0.4],
            radius: 0.3,
            density_rate: 1.0,
            temperature_rate: 2.0,
        };
        fill_sphere(
            &gpu,
            &mut cache,
            &density_source,
            &temperature_source,
            &sphere,
            dx,
        )
        .unwrap();
        let sources = Sources::new(&density_source, &temperature_source);
        let constants = StepConstants {
            beta: 1.0,
            vorticity,
            ..StepConstants::new(cells, 1.0 / 24.0, dx)
        };
        let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
        for _ in 0..20 {
            substep(
                &gpu,
                &mut cache,
                &mut pool,
                &mut state,
                sources,
                &constants,
                PressureSolve::GaussSeidel(160),
            )
            .unwrap();
        }
        let omega = cpu_curl(&state.read_velocity(&gpu).unwrap(), cells, 1.0 / dx);
        omega[3].iter().map(|&v| f64::from(v)).sum()
    }
    let without = total_vorticity(0.0);
    let with = total_vorticity(4.0);
    assert!(with >= 1.05 * without, "with {with}, without {without}");
}

/// Task 5's blend and wind stages run inside a substep. Every x and z domain
/// face is open, so a uniform flow along either axis is divergence-free and
/// projection leaves it as the stage made it.
#[test]
fn the_solver_runs_velocity_emission_and_wind() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(12, 10, 8);
    let dx = 0.125;
    // Open: -x, +x, -z, +z (bits 0, 1, 4, 5).
    let open_mask = 0b11_0011;
    let zero = pool.acquire_zeroed(&gpu, &mut cache, cells).unwrap();
    let constants = StepConstants {
        open_mask,
        ..StepConstants::new(cells, 1.0 / 24.0, dx)
    };

    // Velocity emission alone: weight 1000/s pulls x faces to 1.
    let weight = upload(&gpu, &mut pool, cells, &vec![1000.0; cells.voxel_count()]);
    let target_faces: [Vec<f32>; 3] = std::array::from_fn(|a| {
        vec![if a == 0 { 1.0 } else { 0.0 }; face_dims(cells, a).voxel_count()]
    });
    let target = upload_staggered(&gpu, &mut pool, cells, &target_faces);
    let sources = Sources::new(&zero, &zero).with_emission(Emission {
        weight: &weight,
        velocity: &target,
    });
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
    substep(
        &gpu,
        &mut cache,
        &mut pool,
        &mut state,
        sources,
        &constants,
        PressureSolve::GaussSeidel(40),
    )
    .unwrap();
    let u = state.read_velocity(&gpu).unwrap();
    let d = face_dims(cells, 0);
    for k in 0..d.z {
        for j in 0..d.y {
            for i in 1..d.x - 1 {
                let v = u[0][index(d, i, j, k)];
                assert!((v - 1.0).abs() <= 0.05, "x face {:?}: {v}", [i, j, k]);
            }
        }
    }
    state.release_to(&mut pool);

    // Wind alone: 2 m/s² along +z.
    let windy = StepConstants {
        wind: [0.0, 0.0, 2.0],
        ..constants
    };
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
    substep(
        &gpu,
        &mut cache,
        &mut pool,
        &mut state,
        Sources::new(&zero, &zero),
        &windy,
        PressureSolve::GaussSeidel(40),
    )
    .unwrap();
    let u = state.read_velocity(&gpu).unwrap();
    let d = face_dims(cells, 2);
    let (mut sum, mut n) = (0.0f64, 0.0f64);
    for k in 1..d.z - 1 {
        for j in 0..d.y {
            for i in 0..d.x {
                sum += f64::from(u[2][index(d, i, j, k)]);
                n += 1.0;
            }
        }
    }
    assert!(sum / n > 0.0, "mean interior z face {}", sum / n);
}

/// Build the frame's solid mask from `collider` at `frame`: returns (sdf values, mask, collider velocity).
fn collider_at(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    pool: &mut FieldPool,
    cells: FieldDims,
    dx: f32,
    collider: &ColliderParams,
    frame: f64,
) -> (Vec<f32>, Field, StaggeredField) {
    let sdf = pool.acquire(gpu, cells, FieldFormat::R32Float).unwrap();
    let velocity = pool.acquire_staggered_uninit(gpu, cells).unwrap();
    let pose = collider.transform.pose(frame, 1.0 / 24.0);
    fill_collider(
        gpu,
        cache,
        collider,
        &pose,
        dx,
        ColliderFields {
            sdf: &sdf,
            velocity: &velocity,
        },
    )
    .unwrap();
    let mask = pool.acquire(gpu, cells, FieldFormat::R32Float).unwrap();
    let u = Uniforms::new(gpu, &StepConstants::new(cells, 1.0, dx)).unwrap();
    let mut batch = ComputeBatch::new();
    solidify(gpu, cache, &mut batch, &u, &sdf, &mask).unwrap();
    batch.submit(gpu).unwrap();
    let values = sdf.read_back(gpu).unwrap();
    pool.release(sdf);
    (values, mask, velocity)
}

/// The collider scene: a plume rising for 40 frames at 32³ from a sphere
/// source at z = 0.3, with or without a static collider sphere of radius
/// 0.25 at (1, 1, 0.8). Returns the mean density beside the collider's
/// position (cells with |z − 0.8| ≤ 0.1 and distance from the vertical axis
/// through (1, 1) in (0.25 + dx, 0.25 + 3·dx], all outside the sphere) and,
/// with the collider, the largest density inside it and the peak density.
fn collider_scene(with_collider: bool) -> (f64, Option<(f32, f32)>) {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(32, 32, 32);
    let dx = 2.0 / 32.0;
    let ds = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let ts = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let sphere = Sphere {
        center: [1.0, 1.0, 0.3],
        radius: 0.2,
        density_rate: 1.0,
        temperature_rate: 2.0,
    };
    fill_sphere(&gpu, &mut cache, &ds, &ts, &sphere, dx).unwrap();
    let collider = ColliderParams {
        shape: Shape::Sphere { radius: 0.25 },
        transform: Transform::at([1.0, 1.0, 0.8]),
    };
    let solid =
        with_collider.then(|| collider_at(&gpu, &mut cache, &mut pool, cells, dx, &collider, 0.0));
    let mut sources = Sources::new(&ds, &ts);
    if let Some((_, mask, obstacle)) = &solid {
        sources = sources.with_solids(Solids {
            mask,
            velocity: obstacle,
        });
    }
    let constants = StepConstants {
        beta: 1.0,
        has_solids: with_collider,
        ..StepConstants::new(cells, 1.0 / 24.0, dx)
    };
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
    for _ in 0..40 {
        substep(
            &gpu,
            &mut cache,
            &mut pool,
            &mut state,
            sources,
            &constants,
            PressureSolve::GaussSeidel(160),
        )
        .unwrap();
    }
    let density = state.density.read_back(&gpu).unwrap();

    let h = f64::from(dx);
    let (mut beside, mut n) = (0.0f64, 0.0f64);
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let [x, y, z] = [i, j, k].map(|c| (f64::from(c) + 0.5) * h);
                let radial = ((x - 1.0).powi(2) + (y - 1.0).powi(2)).sqrt();
                if (z - 0.8).abs() <= 0.1 && radial > 0.25 + h && radial <= 0.25 + 3.0 * h {
                    beside += f64::from(density[index(cells, i, j, k)]);
                    n += 1.0;
                }
            }
        }
    }
    assert!(n > 0.0, "the ring beside the collider must contain cells");
    let inside_and_peak = solid.map(|(sdf, _, _)| {
        let peak = density.iter().copied().fold(0.0f32, f32::max);
        let inside = density
            .iter()
            .zip(&sdf)
            .filter(|(_, s)| **s < 0.0)
            .map(|(d, _)| *d)
            .fold(0.0f32, f32::max);
        (inside, peak)
    });
    (beside / n, inside_and_peak)
}

/// Umbrella §6: no density enters a collider, and the plume is deflected
/// sideways around it. Density inside stays at most 1% of the peak, which
/// holds by construction now that scalar advection zeroes solid cells (spec
/// §3.2). Beside the collider the plume is more than twice as dense as the
/// same ring with no collider, because the sphere pushes it outward.
#[test]
fn a_plume_does_not_enter_a_collider() {
    let (with, inside_and_peak) = collider_scene(true);
    let (without, _) = collider_scene(false);
    let (inside, peak) = inside_and_peak.unwrap();
    assert!(peak > 0.0, "the plume must exist");
    assert!(
        with > 2.0 * without,
        "the collider must push the plume sideways: with {with}, without {without}"
    );
    assert!(inside <= 0.01 * peak, "inside {inside}, peak {peak}");
}

/// Spec §3.2: a moving collider pushes the fluid. A box sliding along +x at
/// 0.8 m/s through still air drives flow ahead of it in the same direction.
#[test]
fn a_moving_collider_pushes_the_fluid() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(24, 16, 16);
    let dx = 2.0 / 24.0;
    let zero = pool.acquire_zeroed(&gpu, &mut cache, cells).unwrap();
    let collider = ColliderParams {
        shape: Shape::Box {
            half_extents: [0.15; 3],
        },
        transform: Transform {
            keys: vec![
                Key {
                    frame: 0.0,
                    translate: [0.5, 0.67, 0.67],
                    rotate: None,
                },
                Key {
                    frame: 24.0,
                    translate: [1.3, 0.67, 0.67],
                    rotate: None,
                },
            ],
        },
    };
    let constants = StepConstants {
        has_solids: true,
        ..StepConstants::new(cells, 1.0 / 24.0, dx)
    };
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
    let frames = 12;
    for frame in 0..frames {
        let (_, mask, obstacle) = collider_at(
            &gpu,
            &mut cache,
            &mut pool,
            cells,
            dx,
            &collider,
            f64::from(frame),
        );
        let sources = Sources::new(&zero, &zero).with_solids(Solids {
            mask: &mask,
            velocity: &obstacle,
        });
        substep(
            &gpu,
            &mut cache,
            &mut pool,
            &mut state,
            sources,
            &constants,
            PressureSolve::GaussSeidel(80),
        )
        .unwrap();
        pool.release(mask);
        pool.release_staggered(obstacle);
    }
    // The box's leading face, at the last frame stepped.
    let front = 0.5 + 0.8 * f64::from(frames - 1) / 24.0 + 0.15;
    let u = state.read_velocity(&gpu).unwrap();
    let d = face_dims(cells, 0);
    let (mut sum, mut n) = (0.0, 0.0);
    for k in 0..d.z {
        for j in 0..d.y {
            for i in 0..d.x {
                let x = f64::from(i) * f64::from(dx);
                let y = (f64::from(j) + 0.5) * f64::from(dx);
                let z = (f64::from(k) + 0.5) * f64::from(dx);
                if x > front + 0.5 * f64::from(dx)
                    && x <= front + 2.5 * f64::from(dx)
                    && (y - 0.67).abs() <= 0.1
                    && (z - 0.67).abs() <= 0.1
                {
                    sum += f64::from(u[0][index(d, i, j, k)]);
                    n += 1.0;
                }
            }
        }
    }
    assert!(n > 0.0, "the slab ahead of the box must contain faces");
    assert!(sum / n > 0.1, "mean flow ahead of the box {}", sum / n);
}
