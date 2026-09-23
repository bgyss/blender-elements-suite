mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_ember::bench::GATE_RATIO;
use elements_ember::boundaries::DEFAULT_OPEN_MASK;
use elements_ember::emitter::{Sphere, fill_sphere};
use elements_ember::kernels::StepConstants;
use elements_ember::metrics::{centroid_z, divergence};
use elements_ember::solver::{SolverState, Sources, Substep, substep};

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
    const ITERATIONS: u32 = 160;
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
    let sources = Sources {
        density: &density_source,
        temperature: &temperature_source,
    };
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
    step.project(&gpu, &mut cache, &mut pool, &mut state, ITERATIONS)
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
    let sources = Sources {
        density: &zero,
        temperature: &zero,
    };
    let constants = StepConstants {
        beta: 1.0,
        ..StepConstants::new(cells, 1.0 / 24.0, dx)
    };

    let mut heights = vec![centroid_z(&state.density.read_back(&gpu).unwrap(), cells).unwrap()];
    for _ in 0..20 {
        substep(
            &gpu, &mut cache, &mut pool, &mut state, sources, &constants, 80,
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
    const ITERATIONS: u32 = 20;
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
    let sources = Sources {
        density: &density_source,
        temperature: &temperature_source,
    };
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
    step.project(&gpu, &mut cache, &mut pool, &mut state, ITERATIONS)
        .unwrap();
    step.submit(&gpu, &mut pool).unwrap();
    let after = divergence(&state.read_velocity(&gpu).unwrap(), cells, dx);
    let switched = after.rms / before.rms;
    assert!(switched <= GATE_RATIO, "switched {switched}");
}
