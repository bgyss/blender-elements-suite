mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldPool, PipelineCache};
use elements_core::graph::{Document, NodeError, StateStore, Time};
use elements_ember::cfl::{SubstepPlan, measure_speed, plan_substeps};

#[test]
fn substeps_are_the_ceiling_of_cells_travelled_over_cfl() {
    // dt = 0.1 s, dx = 0.1 m: 2.5 m/s travels 2.5 cells in a frame.
    let plan = |speed, cfl, cap| plan_substeps(speed, 0.1, 0.1, cfl, cap);
    let ok = |count, clamped| Some(SubstepPlan { count, clamped });
    assert_eq!(plan(2.5, 1.0, 8), ok(3, false));
    assert_eq!(plan(2.5, 2.5, 8), ok(1, false));
    assert_eq!(plan(0.0, 1.0, 8), ok(1, false));
    assert_eq!(plan(2.5, 1.0, 2), ok(2, true));
    assert_eq!(plan(f32::NAN, 1.0, 8), None);
    assert_eq!(plan(f32::INFINITY, 1.0, 8), None);
}

#[test]
fn the_measured_speed_is_the_fastest_face_and_a_nan_is_not_hidden() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(8, 6, 5);
    let mut faces = velocity_pattern(cells); // about [-0.6, 0.6]
    let last = faces[2].len() - 1;
    faces[2][last] = -3.0;
    let velocity = upload_staggered(&gpu, &mut pool, cells, &faces);
    assert_eq!(measure_speed(&gpu, &mut cache, &velocity).unwrap(), 3.0);

    faces[1][7] = f32::NAN;
    let velocity = upload_staggered(&gpu, &mut pool, cells, &faces);
    let speed = measure_speed(&gpu, &mut cache, &velocity).unwrap();
    assert!(!speed.is_finite(), "a NaN face read as {speed}");
}

/// A 8³ plume document with the given solver parameters (a JSON object body).
fn plume(emitter_temperature_rate: &str, solver: &str) -> String {
    format!(
        r#"{{
  "version": 3, "dims": [8, 8, 8], "fps": 24.0, "domain_size": 2.0,
  "nodes": [
    {{ "id": 0, "kind": "ember.sphere_emitter",
       "params": {{ "center": [1.0, 1.0, 0.5], "radius": 0.4,
                   "density_rate": 1.0, "temperature_rate": {emitter_temperature_rate} }} }},
    {{ "id": 1, "kind": "ember.smoke_solver", "params": {solver} }},
    {{ "id": 2, "kind": "core.output", "params": {{}} }}
  ],
  "edges": [
    {{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }},
    {{ "from_node": 0, "from_index": 1, "to_node": 1, "to_index": 1 }},
    {{ "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 0 }}
  ],
  "output": 2
}}"#
    )
}

/// Spec §3: a frame whose CFL count exceeds the cap still runs, and the
/// evaluation counts it. Frame 1 starts still. After it the plume moves at
/// a few cm/s (w ≈ h·β·T ≈ 0.035 m/s), and with cfl 1e-4 and dx = 0.25 m
/// that wants dozens of substeps.
#[test]
fn a_frame_over_the_substep_cap_is_counted() {
    let doc = plume(
        "20.0",
        r#"{ "cfl": 0.0001, "max_substeps": 1, "pressure_iterations": 4 }"#,
    );
    let (graph, dims) = Document::from_json(&doc)
        .unwrap()
        .into_graph(&elements_ember::registry())
        .unwrap();
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let mut counts = Vec::new();
    for frame in 1..=3 {
        let evaluated = graph
            .eval_frame(
                &gpu,
                &mut pool,
                &mut pipelines,
                &mut state,
                Time::at(frame, 1, 24.0),
                dims,
            )
            .unwrap();
        counts.push(evaluated.stats.cfl_clamped);
        evaluated.value.release_to(&mut pool);
    }
    state.clear(&mut pool);
    assert_eq!(counts, vec![0, 1, 1]);
}

/// Spec §3: once the velocity is no longer finite, the solver stops with
/// `SolverDiverged` instead of stepping NaNs, and every texture goes back to
/// the pool. An absurd (but finite, so valid) emission rate makes it happen.
#[test]
fn a_diverging_simulation_says_so_and_returns_every_field() {
    let doc = plume("3e38", r#"{ "pressure_iterations": 4 }"#);
    let (graph, dims) = Document::from_json(&doc)
        .unwrap()
        .into_graph(&elements_ember::registry())
        .unwrap();
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let mut failure = None;
    for frame in 1..=20 {
        match graph.eval_frame(
            &gpu,
            &mut pool,
            &mut pipelines,
            &mut state,
            Time::at(frame, 1, 24.0),
            dims,
        ) {
            Ok(evaluated) => evaluated.value.release_to(&mut pool),
            Err(e) => {
                failure = Some((frame, e));
                break;
            }
        }
    }
    let (frame, err) = failure.expect("the simulation must diverge within 20 frames");
    assert!(
        matches!(err, NodeError::SolverDiverged { .. }),
        "frame {frame}: {err:?}"
    );
    state.clear(&mut pool);
    assert_eq!(
        pool.pooled_count() as u64,
        pool.allocation_count(),
        "every allocated texture must be back in the pool"
    );
}
