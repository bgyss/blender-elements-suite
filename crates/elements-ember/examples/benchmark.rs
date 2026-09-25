//! The Mantaflow benchmark (2b-3 spec). Run everything with `just bench`.
//!
//! Modes:
//!   benchmark ember SCENE RES      time and measure Ember
//!   benchmark scene-json SCENE RES print the scene's Mantaflow twin as JSON,
//!                                  for tests/bench/mantaflow_scene.py
//!   benchmark mantaflow SCENE RES  bake, time and measure Mantaflow in
//!                                  Blender ($BLENDER_BIN, or the macOS app)
//!   benchmark latency SCENE RES    time Ember from a parameter change to
//!                                  frame N, in a warm process (2b-3b §2)
//!   benchmark mantaflow-latency SCENE RES
//!                                  time Mantaflow's re-bake to frame N after
//!                                  a parameter change, in one open Blender
//!   benchmark report               build docs/bench/results.md
//!   benchmark latency-report       build docs/bench/latency.md
//!
//! Each run writes docs/bench/results/{solver}-{scene}-{res}.json and .csv;
//! the latency modes write docs/bench/results/latency-{solver}-{scene}-{res}.json.

mod common;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use elements_core::gpu::{Axis, FieldDims, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{NodeRegistry, StateStore, Time};
use elements_ember::bench::report::{
    Context, LatencyPoint, LatencySummary, RunSummary, latency_markdown, load_average,
    results_markdown, single_commit, single_latency_commit, write_latency, write_summary,
};
use elements_ember::bench::{EMISSION_FRAMES, SOLVER_NODE, Scene};
use elements_ember::metrics::{FrameMetrics, Sample, drift, measure};
use elements_ember::solver;

use common::{commit_label_excluding, median, shell};

type Res<T> = Result<T, Box<dyn Error>>;

/// The benchmark's own output, relative to the repository root. Writing it
/// must not make later runs' commit label dirty (see `bench_commit`).
const OUTPUTS: [&str; 3] = [
    "docs/bench/results",
    "docs/bench/results.md",
    "docs/bench/latency.md",
];

/// The commit label recorded in every summary: `commit_label`, ignoring the
/// benchmark's own output, so one `just bench` records one label throughout.
fn bench_commit() -> String {
    commit_label_excluding(&OUTPUTS)
}

fn results_dir() -> PathBuf {
    workspace().join("docs/bench/results")
}

fn workspace() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn scene(name: &str, res: u32) -> Res<Scene> {
    Ok(match name {
        "plume" => Scene::plume(res),
        "plume_collider" => Scene::plume_collider(res),
        "plume_wind" => Scene::plume_wind(res),
        _ => return Err(format!("unknown scene {name}").into()),
    })
}

/// 3 timed runs below 256³, one at 256³ (spec §1).
fn runs_for(res: u32) -> u32 {
    if res >= 256 { 1 } else { 3 }
}

/// One timed run: each frame's `eval_frame` plus a blocking wait, in ms,
/// frame 1 excluded. Returns the frame times and the pool's allocated bytes.
fn timed_run(gpu: &GpuContext, registry: &NodeRegistry, scene: &Scene) -> Res<(Vec<f64>, u64)> {
    let doc = scene.document();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(registry)?;
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let mut frames = Vec::new();
    let first = config.start_frame;
    for frame in first..first + scene.frames {
        let time = Time::at(frame, first, config.fps);
        let start = Instant::now();
        let evaluated = graph.eval_frame(gpu, &mut pool, &mut pipelines, &mut state, time, dims)?;
        gpu.wait()?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        evaluated.value.release_to(&mut pool);
        if frame > first {
            frames.push(ms);
        }
    }
    let bytes = pool.allocated_bytes();
    state.clear(&mut pool);
    Ok((frames, bytes))
}

/// The metrics run: after every frame, read the solver's density and
/// velocity back and measure them. Never timed: readback stalls the GPU and
/// copies whole grids, which would dominate the frame times (spec §6.1).
fn metrics_run(gpu: &GpuContext, registry: &NodeRegistry, scene: &Scene) -> Res<Vec<FrameMetrics>> {
    let doc = scene.document();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(registry)?;
    let dx = scene.domain_size / f64::from(*scene.cells.iter().max().unwrap());
    let solid = scene.solid_mask();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let mut out = Vec::new();
    let first = config.start_frame;
    for frame in first..first + scene.frames {
        let time = Time::at(frame, first, config.fps);
        let evaluated = graph.eval_frame(gpu, &mut pool, &mut pipelines, &mut state, time, dims)?;
        evaluated.value.release_to(&mut pool);
        let density = state
            .get(SOLVER_NODE, solver::DENSITY)
            .ok_or("no solver density in state")?
            .as_field()?
            .read_back(gpu)?;
        let velocity = state
            .get(SOLVER_NODE, solver::VELOCITY)
            .ok_or("no solver velocity in state")?
            .as_vector_field()?;
        let faces = [
            velocity.face(Axis::X).read_back(gpu)?,
            velocity.face(Axis::Y).read_back(gpu)?,
            velocity.face(Axis::Z).read_back(gpu)?,
        ];
        out.push(measure(&Sample {
            cells: dims,
            dx,
            density: &density,
            faces: &faces,
            solid: &solid,
            open_mask: scene.solver.boundaries.open_mask(),
        }));
    }
    state.clear(&mut pool);
    Ok(out)
}

fn run_ember(name: &str, res: u32) -> Res<()> {
    let gpu = GpuContext::new_headless()?;
    let registry = elements_ember::registry();
    let scene = scene(name, res)?;
    let load_before = load_average();
    let runs = runs_for(res);
    let mut medians = Vec::new();
    let mut all = Vec::new();
    let mut peak = 0;
    for r in 0..runs {
        eprintln!("ember {name} {res}³: timed run {} of {runs}", r + 1);
        let (frames, bytes) = timed_run(&gpu, &registry, &scene)?;
        medians.push(median(&frames));
        all.extend(frames);
        peak = peak.max(bytes);
    }
    let load_after = load_average();
    eprintln!("ember {name} {res}³: metrics run");
    let frames = metrics_run(&gpu, &registry, &scene)?;
    let summary = RunSummary {
        solver: "ember".into(),
        scene: name.into(),
        resolution: res,
        runs,
        frame_ms_median: median(&medians),
        frame_ms_min: all.iter().copied().fold(f64::INFINITY, f64::min),
        frame_ms_max: all.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        peak_bytes: peak,
        load_before,
        load_after,
        blender: None,
        commit: bench_commit(),
        drift: drift_from_cut_off(&frames, scene.fps),
        frames,
    };
    write_summary(&results_dir(), &summary)?;
    Ok(())
}

/// The frames each latency point evaluates up to (2b-3b spec §2).
const LATENCY_FRAMES: [u32; 4] = [1, 24, 60, 120];
/// Ember's timed runs per latency point.
const LATENCY_RUNS: u32 = 5;
/// Mantaflow's timed runs per latency point: 5 up to N = 24, 3 above, since
/// a 128³ re-bake to frame 120 takes about a minute (spec §2).
const MANTAFLOW_LATENCY_RUNS: [u32; 4] = [5, 5, 3, 3];
/// The warm-up evaluates the unchanged scene this far, once, untimed.
const WARM_UP_FRAMES: u32 = 24;

/// Evaluate `scene` from frame 1 to `frames` in a fresh graph, pool and
/// state, as a daemon does after a parameter change, ending with a blocking
/// wait. Returns the seconds from before graph construction to frame N
/// ready. The pipeline cache is the caller's and stays warm.
fn latency_run(
    gpu: &GpuContext,
    registry: &NodeRegistry,
    pipelines: &mut PipelineCache,
    scene: &Scene,
    frames: u32,
) -> Res<f64> {
    let doc = scene.document();
    let config = doc.timeline_config();
    let first = config.start_frame;
    let start = Instant::now();
    let (graph, dims) = doc.into_graph(registry)?;
    let mut pool = FieldPool::new();
    let mut state = StateStore::new();
    for frame in first..first + frames {
        let time = Time::at(frame, first, config.fps);
        let evaluated = graph.eval_frame(gpu, &mut pool, pipelines, &mut state, time, dims)?;
        evaluated.value.release_to(&mut pool);
    }
    gpu.wait()?;
    let seconds = start.elapsed().as_secs_f64();
    state.clear(&mut pool);
    Ok(seconds)
}

/// Spec §2: one process, device, pipeline cache and registry throughout;
/// each run changes the emitter's density rate, so no two runs within a
/// point share a document, then times graph construction plus frames 1..=N.
/// The N = 1 median is the time to the first frame. `load_before` is taken
/// after the warm-up, just before the first timed run.
fn run_latency(name: &str, res: u32) -> Res<()> {
    let gpu = GpuContext::new_headless()?;
    let registry = elements_ember::registry();
    let mut pipelines = PipelineCache::new();
    let scene = scene(name, res)?;
    eprintln!("latency {name} {res}³: warm-up to frame {WARM_UP_FRAMES}");
    latency_run(&gpu, &registry, &mut pipelines, &scene, WARM_UP_FRAMES)?;
    // Every pipeline the timed runs need should exist by now; a later
    // compile would land inside a timing.
    let warm_pipelines = pipelines.len();
    let load_before = load_average();
    let mut points = Vec::new();
    for n in LATENCY_FRAMES {
        let mut runs_s = Vec::new();
        for run in 0..LATENCY_RUNS {
            let rate = 1.0 + 0.01 * (run + 1) as f32;
            let changed = scene.clone().with_density_rate(rate);
            let s = latency_run(&gpu, &registry, &mut pipelines, &changed, n)?;
            eprintln!(
                "latency {name} {res}³: N = {n}, run {} of {LATENCY_RUNS}: {s:.3} s",
                run + 1
            );
            runs_s.push(s);
        }
        points.push(LatencyPoint {
            frame: n,
            median_s: median(&runs_s),
            runs_s,
        });
    }
    let load_after = load_average();
    let compiled = pipelines.len() - warm_pipelines;
    if compiled > 0 {
        eprintln!(
            "\n*** WARNING: latency {name} {res}³: {compiled} pipelines were compiled during \
             the timed runs, so some timings include a shader compile ***\n"
        );
    }
    let summary = LatencySummary {
        solver: "ember".into(),
        scene: name.into(),
        resolution: res,
        commit: bench_commit(),
        load_before,
        load_after,
        blender: None,
        pipelines_compiled: Some(compiled),
        points,
    };
    let path = write_latency(&results_dir(), &summary)?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

/// Drift from the last emitting frame on, so `drift[0]` is frame 60.
fn drift_from_cut_off(frames: &[FrameMetrics], fps: f64) -> Vec<f64> {
    let from = (EMISSION_FRAMES[1] - 1) as usize;
    let mass: Vec<f64> = frames.iter().map(|f| f.mass_inside).collect();
    let outflow: Vec<f64> = frames.iter().map(|f| f.outflow_rate).collect();
    drift(&mass, &outflow, 1.0 / fps, from)[from..].to_vec()
}

fn blender() -> String {
    std::env::var("BLENDER_BIN")
        .unwrap_or_else(|_| "/Applications/Blender.app/Contents/MacOS/Blender".to_owned())
}

/// Blender's own lines kept from its stderr for error messages.
const STDERR_TAIL: usize = 40;

/// The last `STDERR_TAIL` lines Blender wrote, without the rusage block that
/// `/usr/bin/time -l` appends (it starts with the line holding "real").
fn blender_tail(stderr: &str) -> String {
    let lines: Vec<&str> = stderr.lines().collect();
    let end = lines
        .iter()
        .rposition(|l| l.contains(" real ") && l.contains(" user "))
        .unwrap_or(lines.len());
    let start = end.saturating_sub(STDERR_TAIL);
    lines[start..end].join("\n")
}

/// Run the scene script under `/usr/bin/time -l` with `extra` arguments
/// after its two positional ones, returning peak resident bytes and the tail
/// of Blender's stderr, for errors found after it exits. `-l` prints
/// "maximum resident set size" in bytes on macOS.
fn run_blender(scene_json: &Path, out: &Path, extra: &[&str]) -> Res<(u64, String)> {
    let script = workspace().join("tests/bench/mantaflow_scene.py");
    let mut cmd = Command::new("/usr/bin/time");
    cmd.arg("-l")
        .arg(blender())
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
        ])
        .arg("--python")
        .arg(&script)
        .arg("--")
        .arg(scene_json)
        .arg(out)
        .args(extra);
    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let tail = blender_tail(&stderr);
    if !output.status.success() {
        return Err(format!(
            "Blender exited with {}; its stderr ends:\n{tail}",
            output.status
        )
        .into());
    }
    let rss = stderr
        .lines()
        .find(|l| l.contains("maximum resident set size"))
        .and_then(|l| l.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| {
            with_stderr(
                "no \"maximum resident set size\" in /usr/bin/time -l output",
                &tail,
            )
        })?;
    Ok((rss, tail))
}

/// `e` with Blender's stderr tail appended, for failures found after a bake
/// that Blender reported as successful.
fn with_stderr(e: impl std::fmt::Display, tail: &str) -> Box<dyn Error> {
    format!("{e}\nBlender exited 0; its stderr ends:\n{tail}").into()
}

/// `timings.json` from the scene script.
#[derive(serde::Deserialize)]
struct Timings {
    frames: Vec<FrameTime>,
    blender: String,
}

#[derive(serde::Deserialize)]
struct FrameTime {
    frame: u32,
    mtime_ns: u128,
}

/// Each frame's time in ms from consecutive cache-file modification times,
/// frames 2 onwards (notes: Timing).
fn frame_times(t: &Timings, frames: u32) -> Res<Vec<f64>> {
    let expected: Vec<u32> = (1..=frames).collect();
    let got: Vec<u32> = t.frames.iter().map(|f| f.frame).collect();
    if got != expected {
        return Err(format!("timings.json lists frames {got:?}, expected 1..={frames}").into());
    }
    Ok(t.frames
        .windows(2)
        .map(|w| w[1].mtime_ns.saturating_sub(w[0].mtime_ns) as f64 / 1e6)
        .collect())
}

fn clear_dir(dir: &Path) -> Res<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir)?;
    Ok(())
}

/// The scene's scratch directory under `target/bench/`, with its
/// Mantaflow twin written there as `scene.json`; returns both paths.
fn scene_scratch(name: &str, res: u32, scene: &Scene) -> Res<(PathBuf, PathBuf)> {
    let scratch = workspace().join(format!("target/bench/{name}-{res}"));
    std::fs::create_dir_all(&scratch)?;
    let scene_json = scratch.join("scene.json");
    std::fs::write(
        &scene_json,
        serde_json::to_string_pretty(&scene.mantaflow_json())? + "\n",
    )?;
    Ok((scratch, scene_json))
}

fn run_mantaflow(name: &str, res: u32) -> Res<()> {
    let scene = scene(name, res)?;
    let context =
        |e: Box<dyn Error>| -> Box<dyn Error> { format!("mantaflow {name} {res}³: {e}").into() };
    let (scratch, scene_json) = scene_scratch(name, res, &scene)?;

    eprintln!("mantaflow {name} {res}³: baseline (no bake)");
    let base_dir = scratch.join("baseline");
    clear_dir(&base_dir)?;
    let (baseline, _) = run_blender(&scene_json, &base_dir, &["--no-bake"]).map_err(context)?;

    // The scene script never clears an old cache, so a stale frame would
    // pass its missing-frame check: clear the output before every bake.
    let out = scratch.join("bake");
    let load_before = load_average();
    let runs = runs_for(res);
    let mut medians = Vec::new();
    let mut all = Vec::new();
    let mut peak = 0;
    let mut version = String::new();
    let mut tail = String::new();
    for r in 0..runs {
        eprintln!("mantaflow {name} {res}³: timed run {} of {runs}", r + 1);
        clear_dir(&out)?;
        let (rss, stderr) = run_blender(&scene_json, &out, &[]).map_err(context)?;
        tail = stderr;
        let path = out.join("timings.json");
        let timings: Timings = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))
            .and_then(|text| {
                serde_json::from_str(&text)
                    .map_err(|e| format!("cannot parse {}: {e}", path.display()))
            })
            .map_err(|e| context(with_stderr(e, &tail)))?;
        let frames =
            frame_times(&timings, scene.frames).map_err(|e| context(with_stderr(e, &tail)))?;
        medians.push(median(&frames));
        all.extend(frames);
        peak = peak.max(rss);
        version = timings.blender;
    }
    let load_after = load_average();

    eprintln!("mantaflow {name} {res}³: reading the last run's cache");
    let cells = FieldDims::new(scene.cells[0], scene.cells[1], scene.cells[2]);
    let dx = scene.domain_size / f64::from(*scene.cells.iter().max().unwrap());
    let solid = scene.solid_mask();
    let mut frames = Vec::new();
    let mut omitted = 0usize;
    for n in 1..=scene.frames {
        let path = out.join(format!("cache/data/fluid_data_{n:04}.vdb"));
        let f = common::mantaflow::read_frame(&path, cells, dx)
            .map_err(|e| context(with_stderr(e, &tail)))?;
        // A missing velocity is accepted only where every face it holds
        // is a collider wall, which no metric reads (notes: Clipping).
        omitted += common::mantaflow::check_coverage(&f, &solid, cells)
            .map_err(|e| context(with_stderr(format!("frame {n}: {e}"), &tail)))?;
        frames.push(measure(&Sample {
            cells,
            dx,
            density: &f.density,
            faces: &f.faces,
            solid: &solid,
            open_mask: scene.solver.boundaries.open_mask(),
        }));
    }
    if omitted > 0 {
        eprintln!(
            "mantaflow {name} {res}³: {omitted} cell-frames store density but no velocity, \
             all at collider wall faces, which no metric reads"
        );
    }
    let summary = RunSummary {
        solver: "mantaflow".into(),
        scene: name.into(),
        resolution: res,
        runs,
        frame_ms_median: median(&medians),
        frame_ms_min: all.iter().copied().fold(f64::INFINITY, f64::min),
        frame_ms_max: all.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        peak_bytes: peak.saturating_sub(baseline),
        load_before,
        load_after,
        blender: Some(version),
        commit: bench_commit(),
        drift: drift_from_cut_off(&frames, scene.fps),
        frames,
    };
    write_summary(&results_dir(), &summary)?;
    Ok(())
}

/// `latency.json` from the scene script's `--latency` mode.
#[derive(serde::Deserialize)]
struct MantaflowLatency {
    solver: String,
    resolution: u32,
    load_before: f64,
    load_after: f64,
    blender: String,
    points: Vec<LatencyPoint>,
}

/// Spec §2, Mantaflow's side: one Blender process builds the scene once
/// and times re-bakes to each N after a density change, its cache freed
/// between them (`mantaflow_scene.py --latency`). Blender's startup and the
/// scene's construction are outside every timing.
fn run_mantaflow_latency(name: &str, res: u32) -> Res<()> {
    let scene = scene(name, res)?;
    let context = |e: Box<dyn Error>| -> Box<dyn Error> {
        format!("mantaflow latency {name} {res}³: {e}").into()
    };
    let (scratch, scene_json) = scene_scratch(name, res, &scene)?;
    let out = scratch.join("latency");
    clear_dir(&out)?;
    let list = |v: &[u32]| v.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    let (frames, runs) = (list(&LATENCY_FRAMES), list(&MANTAFLOW_LATENCY_RUNS));
    eprintln!("mantaflow latency {name} {res}³: N = {frames}, runs {runs}");
    let (_, tail) = run_blender(
        &scene_json,
        &out,
        &["--latency", frames.as_str(), "--runs", runs.as_str()],
    )
    .map_err(context)?;
    let path = out.join("latency.json");
    let m: MantaflowLatency = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))
        .and_then(|text| {
            serde_json::from_str(&text).map_err(|e| format!("cannot parse {}: {e}", path.display()))
        })
        .map_err(|e| context(with_stderr(e, &tail)))?;
    let shape: Vec<(u32, u32)> = m
        .points
        .iter()
        .map(|p| (p.frame, p.runs_s.len() as u32))
        .collect();
    let expected: Vec<(u32, u32)> = LATENCY_FRAMES
        .into_iter()
        .zip(MANTAFLOW_LATENCY_RUNS)
        .collect();
    if m.solver != "mantaflow" || m.resolution != res || shape != expected {
        return Err(context(with_stderr(
            format!(
                "{} holds solver {:?}, resolution {}, (N, runs) {shape:?}; expected \
                 mantaflow, {res}, {expected:?}",
                path.display(),
                m.solver,
                m.resolution
            ),
            &tail,
        )));
    }
    for p in &m.points {
        eprintln!(
            "mantaflow latency {name} {res}³: N = {}: median {:.3} s",
            p.frame, p.median_s
        );
    }
    let summary = LatencySummary {
        solver: "mantaflow".into(),
        scene: name.into(),
        resolution: res,
        commit: bench_commit(),
        load_before: m.load_before,
        load_after: m.load_after,
        blender: Some(m.blender),
        pipelines_compiled: None,
        points: m.points,
    };
    let path = write_latency(&results_dir(), &summary)?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

/// Build `docs/bench/latency.md` from every latency file, or name the
/// missing ones and fail.
fn latency_report() -> Res<()> {
    let dir = results_dir();
    let mut summaries = Vec::new();
    if dir.exists() {
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            let latency = path
                .file_name()
                .is_some_and(|f| f.to_string_lossy().starts_with("latency-"));
            if latency && path.extension().is_some_and(|e| e == "json") {
                let text = std::fs::read_to_string(&path)?;
                let s: LatencySummary =
                    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
                summaries.push(s);
            }
        }
    }
    let commit = single_latency_commit(&summaries).map_err(|lines| {
        for l in &lines {
            eprintln!("{l}");
        }
        "latency files come from more than one commit; latency.md not written"
    })?;
    let ctx = Context {
        machine: shell("sysctl", &["-n", "machdep.cpu.brand_string"]),
        os: format!("macOS {}", shell("sw_vers", &["-productVersion"])),
        commit,
        blender: summaries
            .iter()
            .find_map(|s| s.blender.clone())
            .unwrap_or_else(|| "unknown".to_owned()),
        date: shell("date", &["-u", "+%Y-%m-%d"]),
    };
    match latency_markdown(&summaries, &ctx) {
        Ok(md) => {
            let path = workspace().join("docs/bench/latency.md");
            std::fs::write(&path, md)?;
            eprintln!("wrote {}", path.display());
            Ok(())
        }
        Err(missing) => {
            for m in &missing {
                eprintln!("missing: {}", dir.join(format!("{m}.json")).display());
            }
            Err(format!(
                "{} latency files missing; latency.md not written",
                missing.len()
            )
            .into())
        }
    }
}

/// Build `docs/bench/results.md` from every result file, or name the
/// missing ones and fail.
fn report() -> Res<()> {
    let dir = results_dir();
    let mut summaries = Vec::new();
    if dir.exists() {
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            // Latency files have their own shape and their own report.
            let latency = path
                .file_name()
                .is_some_and(|f| f.to_string_lossy().starts_with("latency-"));
            if path.extension().is_some_and(|e| e == "json") && !latency {
                let text = std::fs::read_to_string(&path)?;
                let s: RunSummary =
                    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
                summaries.push(s);
            }
        }
    }
    // The header names the commit the results came from, not the one that
    // happens to build the report.
    let commit = single_commit(&summaries).map_err(|lines| {
        for l in &lines {
            eprintln!("{l}");
        }
        "result files come from more than one commit; results.md not written"
    })?;
    let ctx = Context {
        machine: shell("sysctl", &["-n", "machdep.cpu.brand_string"]),
        os: format!("macOS {}", shell("sw_vers", &["-productVersion"])),
        commit,
        blender: summaries
            .iter()
            .find_map(|s| s.blender.clone())
            .unwrap_or_else(|| "unknown".to_owned()),
        date: shell("date", &["-u", "+%Y-%m-%d"]),
    };
    match results_markdown(&summaries, &ctx) {
        Ok(md) => {
            let path = workspace().join("docs/bench/results.md");
            std::fs::write(&path, md)?;
            eprintln!("wrote {}", path.display());
            Ok(())
        }
        Err(missing) => {
            for m in &missing {
                eprintln!("missing: {}", dir.join(format!("{m}.json")).display());
            }
            Err(format!(
                "{} result files missing; results.md not written",
                missing.len()
            )
            .into())
        }
    }
}

/// Errors print with Display, so a multi-line Blender stderr tail stays
/// readable rather than one escaped Debug string.
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["ember", name, res] => run_ember(name, res.parse()?),
        ["mantaflow", name, res] => run_mantaflow(name, res.parse()?),
        ["latency", name, res] => run_latency(name, res.parse()?),
        ["mantaflow-latency", name, res] => run_mantaflow_latency(name, res.parse()?),
        ["report"] => report(),
        ["latency-report"] => latency_report(),
        ["scene-json", name, res] => {
            let json = scene(name, res.parse()?)?.mantaflow_json();
            println!("{}", serde_json::to_string_pretty(&json)?);
            Ok(())
        }
        _ => Err(
            "usage: benchmark ember SCENE RES | scene-json SCENE RES | mantaflow SCENE RES \
             | latency SCENE RES | mantaflow-latency SCENE RES | report | latency-report"
                .into(),
        ),
    }
}
