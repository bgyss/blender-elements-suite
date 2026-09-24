mod common;

use elements_core::gpu::{FieldPool, PipelineCache};
use elements_core::graph::{Document, Timeline};
use elements_ember::bench::{
    GateRow, PresetRow, Scene, gate_verdict, preview_substeps_verdict, report,
};
use elements_ember::metrics::FrameMetrics;

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

fn preset(max_substeps: u32, frame_ms_median: f64) -> PresetRow {
    PresetRow {
        max_substeps,
        frame_ms_median,
    }
}

/// Spec §6: preview takes the largest cap whose median full frame fits.
#[test]
fn the_preview_cap_is_the_largest_that_fits() {
    let rows = [
        preset(1, 40.0),
        preset(2, 80.0),
        preset(3, 120.0),
        preset(4, 160.0),
    ];
    assert_eq!(preview_substeps_verdict(&rows), Some(2));
}

#[test]
fn the_preview_frame_limit_is_inclusive() {
    assert!(preset(3, 100.0).passes());
}

#[test]
fn no_cap_fits_when_even_one_substep_is_too_slow() {
    assert_eq!(preview_substeps_verdict(&[preset(1, 101.0)]), None);
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

/// The scene's document round-trips through JSON, builds, and emits by frame 2.
fn loads_and_steps(scene: &Scene) {
    let text = scene.document().to_json().unwrap();
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
    assert!(density.iter().any(|&v| v > 0.0), "{} must emit", scene.name);
}

#[test]
fn the_plume_scene_loads_and_steps() {
    loads_and_steps(&Scene::plume(16));
}

#[test]
fn the_plume_collider_scene_loads_and_steps() {
    let scene = Scene::plume_collider(16);
    assert!(
        scene
            .document()
            .nodes
            .iter()
            .any(|n| n.kind == elements_ember::collider::KIND)
    );
    loads_and_steps(&scene);
}

#[test]
fn the_plume_wind_scene_loads_and_steps() {
    let scene = Scene::plume_wind(16);
    assert_eq!(scene.solver.wind, [0.5, 0.0, 0.0]);
    // The document is what 2b-3's Mantaflow side and the daemon read, so the
    // wind must survive serialisation, not just sit on the struct.
    let text = scene.document().to_json().unwrap();
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    let solver = doc["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["id"] == 1)
        .expect("node 1 is the solver");
    assert_eq!(
        solver["params"]["wind"],
        serde_json::json!([0.5, 0.0, 0.0]),
        "the serialised solver params must carry the wind"
    );
    loads_and_steps(&scene);
}

use elements_ember::bench::EMISSION_FRAMES;

/// 2b-3 spec §3: every bench scene uses the general emitter, a static
/// sphere with no noise or velocity, emitting for frames 1–60.
#[test]
fn bench_scenes_use_the_general_emitter_with_a_window() {
    for scene in [
        Scene::plume(32),
        Scene::plume_collider(32),
        Scene::plume_wind(32),
    ] {
        let doc = scene.document();
        let emitter = &doc.nodes[0];
        assert_eq!(emitter.kind, "ember.emitter", "{}", scene.name);
        assert_eq!(
            emitter.params["active_frames"],
            serde_json::json!(EMISSION_FRAMES),
            "{}",
            scene.name
        );
        assert_eq!(scene.emitter.velocity_blend, 0.0);
        assert!(scene.emitter.noise.is_none());
        // The document still evaluates.
        let registry = elements_ember::registry();
        doc.into_graph(&registry).unwrap();
    }
}

#[test]
fn the_collider_mask_marks_cells_inside_the_sphere() {
    let scene = Scene::plume_collider(32);
    let mask = scene.solid_mask();
    let n = 32u32;
    assert_eq!(mask.len(), (n * n * n) as usize);
    let at = |i: u32, j: u32, k: u32| mask[(i + n * (j + n * k)) as usize];
    // The collider is centred at (1, 1, 0.8) m, radius 0.25; dx = 1/16 m.
    assert!(at(15, 15, 12), "the cell at the centre is solid");
    assert!(!at(15, 15, 20), "a cell 0.5 m above is not");
    assert!(Scene::plume(32).solid_mask().is_empty());
}

#[test]
fn a_summary_round_trips_through_its_file() {
    let dir = tempfile::tempdir().unwrap();
    let frame = |n: f64| FrameMetrics {
        divergence_max: 0.5 * n,
        divergence_rms: 0.25 * n,
        kinetic_energy: 1.5 * n,
        vorticity: 2.0 * n,
        measured_cells: 100 * n as u64,
        mass: 0.125 * n,
        mass_below: 0.0625 * n,
        centroid_m: if n > 1.0 { Some(0.3 * n) } else { None },
        top_m: Some(0.4 * n),
        outflow_rate: 0.01 * n,
    };
    let s = report::RunSummary {
        solver: "ember".into(),
        scene: "plume".into(),
        resolution: 64,
        runs: 3,
        frame_ms_median: 12.5,
        frame_ms_min: 11.0,
        frame_ms_max: 14.25,
        peak_bytes: 123_456_789,
        load_before: 1.5,
        load_after: 2.25,
        blender: None,
        frames: vec![frame(1.0), frame(2.0)],
        drift: vec![0.0, -0.001],
    };
    report::write_summary(dir.path(), &s).unwrap();
    let path = report::summary_path(dir.path(), "ember", "plume", 64);
    let back: report::RunSummary =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(back, s);
    assert!(dir.path().join("ember-plume-64.csv").exists());
}

/// The Mantaflow twin is built from the same scene (2b-3 spec §5): every
/// parameter the scene script needs comes from `mantaflow_json`.
#[test]
fn the_mantaflow_json_carries_every_matched_parameter() {
    let s = Scene::plume_collider(64);
    let j = s.mantaflow_json();
    assert_eq!(j["name"], "plume_collider");
    assert_eq!(j["resolution"], 64);
    assert_eq!(j["frames"], 120);
    assert_eq!(j["emitter"]["active_frames"], serde_json::json!([1, 60]));
    assert_eq!(j["emitter"]["radius"], 0.2);
    assert_eq!(j["emitter"]["center"], serde_json::json!([1.0, 1.0, 0.3]));
    assert_eq!(j["collider"]["radius"], 0.25);
    assert_eq!(j["collider"]["center"], serde_json::json!([1.0, 1.0, 0.8]));
    assert_eq!(j["boundaries"]["pos_z"], "open");
    assert_eq!(j["boundaries"]["neg_z"], "wall");
    assert_eq!(j["substeps"], s.solver.max_substeps);
    let w = Scene::plume_wind(64).mantaflow_json();
    assert_eq!(w["wind"], serde_json::json!([0.5, 0.0, 0.0]));
    assert!(Scene::plume(64).mantaflow_json()["collider"].is_null());
}

/// A run with 120 frames of made-up metrics; only the shape matters to the
/// results table.
fn summary(solver: &str, scene: &str, res: u32) -> report::RunSummary {
    let frames = (1..=120)
        .map(|n| {
            let n = f64::from(n);
            FrameMetrics {
                divergence_max: 1e-3 * n,
                divergence_rms: 1e-4 * n,
                kinetic_energy: 0.01 * n,
                vorticity: 0.02 * n,
                measured_cells: 50 * n as u64,
                mass: 0.001 * n,
                mass_below: 0.0009 * n,
                centroid_m: Some(0.01 * n),
                top_m: Some(0.015 * n),
                outflow_rate: 0.0,
            }
        })
        .collect();
    report::RunSummary {
        solver: solver.into(),
        scene: scene.into(),
        resolution: res,
        runs: if res >= 256 { 1 } else { 3 },
        frame_ms_median: 10.0,
        frame_ms_min: 9.0,
        frame_ms_max: 12.0,
        peak_bytes: 64 << 20,
        load_before: 1.0,
        load_after: 1.5,
        blender: (solver == "mantaflow").then(|| "5.2.2 LTS".to_owned()),
        frames,
        drift: vec![0.0; 61],
    }
}

fn all() -> Vec<report::RunSummary> {
    let mut v = Vec::new();
    for solver in report::SOLVERS {
        for scene in report::SCENES {
            for res in report::RESOLUTIONS {
                v.push(summary(solver, scene, res));
            }
        }
    }
    v
}

fn context() -> report::Context {
    report::Context {
        machine: "Apple M-test".into(),
        os: "macOS 99.1".into(),
        commit: "abc1234".into(),
        blender: "5.2.2 LTS".into(),
        date: "2026-09-23".into(),
    }
}

#[test]
fn the_results_table_needs_every_run() {
    let ctx = context();
    let mut s = all();
    s.retain(|r| !(r.solver == "mantaflow" && r.scene == "plume_wind" && r.resolution == 256));
    let missing = report::results_markdown(&s, &ctx).unwrap_err();
    assert_eq!(missing, vec!["mantaflow-plume_wind-256".to_owned()]);
}

#[test]
fn the_results_table_has_one_section_per_scene_and_names_run_counts() {
    let ctx = context();
    let md = report::results_markdown(&all(), &ctx).unwrap();
    for scene in report::SCENES {
        assert!(md.contains(&format!("## `{scene}`")), "{scene}");
    }
    assert!(md.contains("1 run"), "256³ says it is one run");
    assert!(md.contains("median of 3"), "64³ and 128³ say three");
    assert!(md.contains(&ctx.commit) && md.contains(&ctx.blender));
}
