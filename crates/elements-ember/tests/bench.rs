mod common;

use elements_core::gpu::{FieldPool, PipelineCache};
use elements_core::graph::{Document, Timeline};
use elements_ember::bench::{GateRow, Scene, gate_verdict};

fn row(iterations: u32, step_ms_median: f64, ratio: f64) -> GateRow {
    GateRow {
        iterations,
        step_ms_median,
        ratio,
    }
}

/// Spec §4.3's pre-registered rule, encoded once and tested here.
#[test]
fn the_gate_passes_with_the_largest_n_meeting_both_limits() {
    let rows = [
        row(20, 30.0, 0.30),
        row(40, 50.0, 0.09),
        row(80, 90.0, 0.05),
        row(160, 170.0, 0.02),
    ];
    assert_eq!(gate_verdict(&rows), Some(80));
}

#[test]
fn the_gate_limits_are_inclusive() {
    assert_eq!(gate_verdict(&[row(40, 100.0, 0.10)]), Some(40));
}

#[test]
fn a_row_that_misses_the_ratio_limit_does_not_pass() {
    // step is within GATE_STEP_MS, but ratio 0.1004 > GATE_RATIO (0.10).
    assert!(!row(40, 50.0, 0.1004).passes());
}

#[test]
fn the_gate_fails_when_no_n_meets_both_limits() {
    // Fast enough but too divergent, or accurate enough but too slow.
    assert_eq!(
        gate_verdict(&[row(20, 40.0, 0.2), row(160, 120.0, 0.05)]),
        None
    );
}

#[test]
fn the_plume_scene_loads_and_steps() {
    let text = Scene::plume(16).document().to_json().unwrap();
    let doc = Document::from_json(&text).unwrap();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(&elements_ember::registry()).unwrap();
    let gpu = common::gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut timeline = Timeline::new(config);
    let frame = timeline
        .goto(&graph, &gpu, &mut pool, &mut pipelines, dims, 2)
        .unwrap();
    let density = frame.value.as_field().unwrap().read_back(&gpu).unwrap();
    assert!(density.iter().any(|&v| v > 0.0), "the plume must emit");
}
