mod common;

use elements_core::gpu::{Axis, FieldFormat, FieldPool, PipelineCache};
use elements_core::graph::{Document, StateStore, Time, Timeline};
use elements_ember::bench::{
    GateRow, PresetRow, SOLVER_NODE, Scene, gate_verdict, preview_substeps_verdict, report,
};
use elements_ember::boundaries::Face;
use elements_ember::metrics::{FrameMetrics, Sample, drift, measure};
use elements_ember::shape_emitter::{EmitterFields, fill_emitter};
use elements_ember::solver;

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

/// 2b-3c spec §6: `plume_wind`'s air relaxes towards 1 m/s along +x at
/// 1/s, and blows in at −x and out at +x, with the top still open.
#[test]
fn plume_wind_blows_through_open_sides() {
    let scene = Scene::plume_wind(16);
    assert_eq!(scene.solver.wind_velocity, [1.0, 0.0, 0.0]);
    assert_eq!(scene.solver.wind_rate, 1.0);
    let b = scene.solver.boundaries;
    assert_eq!((b.neg_x, b.pos_x), (Face::Open, Face::Open));
    assert_eq!((b.neg_y, b.pos_y), (Face::Wall, Face::Wall));
    assert_eq!((b.neg_z, b.pos_z), (Face::Wall, Face::Open));
    // The document is what the daemon and the benchmark read, so the wind
    // and the boundaries must survive serialisation, not just sit on the
    // struct.
    let text = scene.document().to_json().unwrap();
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    let solver = doc["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["id"] == 1)
        .expect("node 1 is the solver");
    let params = &solver["params"];
    assert_eq!(params["wind_velocity"], serde_json::json!([1.0, 0.0, 0.0]));
    assert_eq!(params["wind_rate"], 1.0);
    assert!(params.get("wind").is_none(), "{params}");
    assert_eq!(params["boundaries"]["-x"], "open");
    assert_eq!(params["boundaries"]["+x"], "open");
    loads_and_steps(&scene);
}

/// A regression guard for 2b-3's wind, an acceleration nothing balanced:
/// the air sped up without bound and blew the smoke out of its own
/// accounting. At 32³ over the scene's 120 frames the air must stay below
/// 3 m/s with no NaN; an unbalanced 1 m/s² passes 3 m/s only after about
/// 3 s, so the 60 emitting frames alone would not see it. At frame 60, the
/// last emitting frame, the mass inside plus the mass that left through the
/// open faces must be within 10% of what the emitter put in.
#[test]
fn plume_wind_stays_bounded_and_keeps_its_mass() {
    let scene = Scene::plume_wind(32);
    let gpu = common::gpu();
    let registry = elements_ember::registry();
    let doc = scene.document();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(&registry).unwrap();
    let dx = scene.domain_size / f64::from(*scene.cells.iter().max().unwrap());
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();

    // What the emitter adds per second: its density rate field summed.
    let emitted_per_second = {
        let f: [_; 3] =
            std::array::from_fn(|_| pool.acquire(&gpu, dims, FieldFormat::R32Float).unwrap());
        let v = pool.acquire_staggered_uninit(&gpu, dims).unwrap();
        let spf = 1.0 / scene.fps;
        fill_emitter(
            &gpu,
            &mut pipelines,
            &scene.emitter,
            &scene.emitter.transform.pose(1.0, spf),
            spf,
            dx as f32,
            EmitterFields {
                density: &f[0],
                temperature: &f[1],
                weight: &f[2],
                velocity: &v,
            },
        )
        .unwrap();
        let rate: f64 = f[0]
            .read_back(&gpu)
            .unwrap()
            .iter()
            .map(|&r| f64::from(r))
            .sum();
        for field in f {
            pool.release(field);
        }
        pool.release_staggered(v);
        rate * dx.powi(3)
    };

    let open_mask = scene.solver.boundaries.open_mask();
    let first = config.start_frame;
    let last_emitting = EMISSION_FRAMES[1];
    let mut frames = Vec::new();
    let mut max_speed = 0.0f32;
    for frame in first..first + scene.frames {
        let time = Time::at(frame, first, config.fps);
        let evaluated = graph
            .eval_frame(&gpu, &mut pool, &mut pipelines, &mut state, time, dims)
            .unwrap();
        evaluated.value.release_to(&mut pool);
        let density = state
            .get(SOLVER_NODE, solver::DENSITY)
            .unwrap()
            .as_field()
            .unwrap()
            .read_back(&gpu)
            .unwrap();
        let velocity = state
            .get(SOLVER_NODE, solver::VELOCITY)
            .unwrap()
            .as_vector_field()
            .unwrap();
        let faces = [Axis::X, Axis::Y, Axis::Z].map(|a| velocity.face(a).read_back(&gpu).unwrap());
        for v in faces.iter().flatten().chain(&density) {
            assert!(v.is_finite(), "frame {frame}: a non-finite value");
        }
        max_speed = faces
            .iter()
            .flatten()
            .fold(max_speed, |m, v| m.max(v.abs()));
        frames.push(measure(&Sample {
            cells: dims,
            dx,
            density: &density,
            faces: &faces,
            solid: &[],
            open_mask,
        }));
    }
    state.clear(&mut pool);

    // Frames 1 … 60, the emitting ones.
    let emitting = (last_emitting - first + 1) as usize;
    let mass: Vec<f64> = frames[..emitting].iter().map(|f| f.mass_inside).collect();
    let rate: Vec<f64> = frames[..emitting].iter().map(|f| f.outflow_rate).collect();
    // `drift` from frame 1 is M(60) + outflow − M(1); add M(1) back.
    let spf = 1.0 / scene.fps;
    let accounted = drift(&mass, &rate, spf, 0)[emitting - 1] + mass[0];
    let emitted = emitted_per_second * spf * emitting as f64;
    let ratio = accounted / emitted;
    eprintln!(
        "plume_wind 32³: max |u| {max_speed} m/s over {} frames; at frame {last_emitting}, \
         mass_inside {} + outflow {} = {accounted} of {emitted} emitted ({ratio})",
        scene.frames,
        mass[emitting - 1],
        accounted - mass[emitting - 1],
    );
    assert!(max_speed < 3.0, "max |u| {max_speed} m/s");
    assert!((ratio - 1.0).abs() <= 0.10, "accounted/emitted {ratio}");
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

/// 2b-3c spec §4: one layer of solid across the domain at z = 0.8 m, open
/// only in a slit two cells wide in x, along all of y, above the emitter.
#[test]
fn the_plate_mask_leaves_only_the_slit_open() {
    let n = 64u32;
    let mask = Scene::plume_plate(n).solid_mask();
    assert_eq!(mask.len(), (n * n * n) as usize);
    // dx = 1/32 m, so 0.8 m lies in layer 25 and x = 1 m between cells 31
    // and 32.
    let (layer, slit) = (25, [31, 32]);
    let mut open_in_layer = Vec::new();
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let solid = mask[(i + n * (j + n * k)) as usize];
                if k != layer {
                    assert!(!solid, "({i}, {j}, {k}) is outside the plate");
                } else if !solid {
                    open_in_layer.push((i, j));
                }
            }
        }
    }
    let expected: Vec<(u32, u32)> = (0..n).flat_map(|j| slit.map(|i| (i, j))).collect();
    assert_eq!(
        open_in_layer, expected,
        "only the slit is open in the plate"
    );
}

#[test]
fn the_plume_plate_scene_loads_and_steps() {
    loads_and_steps(&Scene::plume_plate(16));
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
        mass_inside: 0.0625 * n,
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
        commit: "abc1234-dirty".into(),
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

/// 2b-3's result files call `mass_inside` by its old name, `mass_below`;
/// the report still reads them until they are rerun.
#[test]
fn a_frame_written_as_mass_below_still_loads() {
    let json = serde_json::json!({
        "divergence_max": 0.0, "divergence_rms": 0.0, "kinetic_energy": 0.0,
        "vorticity": 0.0, "measured_cells": 0, "mass": 2.0, "mass_below": 1.5,
        "centroid_m": null, "top_m": null, "outflow_rate": 0.0
    });
    let f: FrameMetrics = serde_json::from_value(json).unwrap();
    assert_eq!(f.mass_inside, 1.5);
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
    assert_eq!(w["wind_velocity"], serde_json::json!([1.0, 0.0, 0.0]));
    assert_eq!(w["wind_rate"], 1.0);
    assert!(w.get("wind").is_none(), "{w}");
    assert_eq!(w["boundaries"]["neg_x"], "open");
    assert_eq!(w["boundaries"]["pos_x"], "open");
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
                measured_cells: 1000 + 50 * n as u64,
                mass: 0.001 * n,
                mass_inside: 0.0009 * n,
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
        commit: "abc1234".into(),
        frames,
        // drift[i] is frame 60 + i; each frame's value is distinct, so a
        // wrong index shows up as a wrong number.
        drift: (0..=60).map(|i| 1e-4 * f64::from(i)).collect(),
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

/// The row reads drift at frames 80 and 120 (`drift[20]`, `drift[60]`) as a
/// percentage of `mass_inside` at frame 60, and divides kinetic energy and
/// vorticity by the measured cells of the same frame.
#[test]
fn the_results_row_pins_drift_frames_and_per_cell_values() {
    let md = report::results_markdown(&all(), &context()).unwrap();
    // mass_inside at 60 is 0.0009 · 60 = 0.054.
    // Frame 120: 0.006, 11.1%. Frame 80: 0.002, 3.7%.
    assert!(
        md.contains("| 0.00200 (+3.7%) | 0.00600 (+11.1%) |"),
        "{md}"
    );
    // Kinetic energy 0.01 n over 1000 + 50 n cells: 0.6 / 4000 at 60,
    // 1.2 / 7000 at 120. Vorticity is twice that.
    assert!(
        md.contains("| 1.50e-4 / 1.71e-4 |"),
        "kinetic energy per cell"
    );
    assert!(md.contains("| 3.00e-4 / 3.43e-4 |"), "vorticity per cell");
}

#[test]
fn the_report_refuses_summaries_from_more_than_one_commit() {
    let mut s = all();
    assert_eq!(report::single_commit(&s), Ok("abc1234".to_owned()));
    s[4].commit = "def5678".into();
    let lines = report::single_commit(&s).unwrap_err();
    assert_eq!(lines.len(), s.len(), "every file is listed with its commit");
    assert!(
        lines.contains(&"ember-plume_collider-128: def5678".to_owned()),
        "{lines:?}"
    );
    assert!(
        lines.contains(&"ember-plume-64: abc1234".to_owned()),
        "{lines:?}"
    );
}

/// The drift-at-80 cell says when a run's outflow started, and how much of
/// the frame-60 mass the outflow estimate credits by frame 80, only for a
/// run whose outflow rate is non-zero by then.
#[test]
fn the_drift_cell_marks_outflow_before_frame_80() {
    let mut s = all();
    let early = s
        .iter_mut()
        .find(|r| r.solver == "ember" && r.scene == "plume" && r.resolution == 64)
        .unwrap();
    for f in &mut early.frames[69..] {
        f.outflow_rate = 0.01;
    }
    // mass_inside goes 0.054 → 0.072 from frame 60 to 80, so a drift of
    // 0.0234 credits 0.0054 to outflow: 10.0% of 0.054.
    early.drift[20] = 0.0234;
    let late = s
        .iter_mut()
        .find(|r| r.solver == "mantaflow" && r.scene == "plume" && r.resolution == 64)
        .unwrap();
    for f in &mut late.frames[80..] {
        f.outflow_rate = 0.01;
    }
    let md = report::results_markdown(&s, &context()).unwrap();
    assert!(
        md.contains("| 0.0234 (+43.3%); outflow from frame 70 credits 10.0% |"),
        "{md}"
    );
    assert_eq!(
        md.matches("outflow from frame").count(),
        1,
        "outflow that starts at frame 81 is not marked"
    );
}

/// A frame whose mass is below the floor shows "—" for its centroid, top
/// and per-cell values, rather than numbers from a few stray cells.
#[test]
fn a_near_empty_frame_shows_no_shape_or_per_cell_values() {
    let mut s = all();
    let empty = s
        .iter_mut()
        .find(|r| r.solver == "mantaflow" && r.scene == "plume_wind" && r.resolution == 128)
        .unwrap();
    empty.frames[119].mass = 1e-12;
    let md = report::results_markdown(&s, &context()).unwrap();
    let wind = md.split("## `plume_wind`").nth(1).unwrap();
    let row = wind
        .lines()
        .find(|l| l.starts_with("| mantaflow | 128³"))
        .unwrap();
    assert!(row.contains("| 1.50e-4 / — |"), "KE per cell: {row}");
    assert!(row.contains("| 3.00e-4 / — |"), "vorticity per cell: {row}");
    assert!(
        row.contains("| 0.600 / — | 0.900 / — |"),
        "centroid, top: {row}"
    );
    // Totals are still shown: they are sums, not shape.
    assert!(row.contains("| 0.600 / 1.20 |"), "kinetic energy: {row}");
}

/// The Notes compare the solvers' emitted mass from the runs themselves,
/// over frames 12–24 and at frame 60.
#[test]
fn the_notes_compute_the_emitted_mass_ratio() {
    let mut s = all();
    let e = s
        .iter_mut()
        .find(|r| r.solver == "ember" && r.scene == "plume_collider" && r.resolution == 128)
        .unwrap();
    e.frames[11].mass *= 1.1;
    e.frames[59].mass *= 0.5;
    let md = report::results_markdown(&s, &context()).unwrap();
    assert!(
        md.contains("over frames 12–24, Ember's mass is 1.00–1.10× Mantaflow's"),
        "{md}"
    );
    assert!(md.contains("By frame 60 the ratio is 0.50–1.00×"), "{md}");
}

/// Mantaflow's twin is built for a cube (one `dx` in `mapping.py`).
#[test]
#[should_panic(expected = "needs a cubic domain")]
fn the_mantaflow_twin_refuses_a_non_cubic_domain() {
    let mut s = Scene::plume(64);
    s.cells = [64, 64, 128];
    let _ = s.mantaflow_json();
}

/// The mask is computed from the collider's one key; a keyframed collider
/// would give a mask that matches neither solver.
#[test]
#[should_panic(expected = "bench colliders are static")]
fn the_bench_mask_refuses_a_keyframed_collider() {
    let mut s = Scene::plume_collider(16);
    let c = &mut s.colliders[0];
    let mut key = c.transform.keys[0];
    key.frame += 10.0;
    c.transform.keys.push(key);
    let _ = s.solid_mask();
}
