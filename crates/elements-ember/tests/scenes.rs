mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, PipelineCache};
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

/// Umbrella §6, spec §4.2: projection cuts RMS divergence to at most 10% of
/// its value before projection, at the default iteration count the gate chose.
#[test]
fn projection_leaves_at_most_a_tenth_of_the_divergence() {
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
        cells,
        h: 1.0 / 24.0,
        dx,
        alpha: 0.0,
        beta: 1.0,
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
    assert!(
        after.rms <= 0.1 * before.rms,
        "RMS divergence {} -> {} (ratio {})",
        before.rms,
        after.rms,
        after.rms / before.rms
    );
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
        cells,
        h: 1.0 / 24.0,
        dx,
        alpha: 0.0,
        beta: 1.0,
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
