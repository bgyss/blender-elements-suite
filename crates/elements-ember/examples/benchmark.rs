//! The Mantaflow benchmark (2b-3 spec). Run everything with `just bench`.
//!
//! Modes:
//!   benchmark ember SCENE RES      time and measure Ember
//!   benchmark scene-json SCENE RES print the scene's Mantaflow twin as JSON,
//!                                  for tests/bench/mantaflow_scene.py
//!   benchmark mantaflow SCENE RES  bake, time and measure Mantaflow in
//!                                  Blender ($BLENDER_BIN, or the macOS app)
//!   benchmark report               build docs/bench/results.md
//!
//! Each run writes docs/bench/results/{solver}-{scene}-{res}.json and .csv.

mod common;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use elements_core::gpu::{Axis, FieldDims, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{NodeRegistry, StateStore, Time};
use elements_ember::bench::report::{
    Context, RunSummary, load_average, results_markdown, single_commit, write_summary,
};
use elements_ember::bench::{EMISSION_FRAMES, SOLVER_NODE, Scene};
use elements_ember::metrics::{FrameMetrics, Sample, drift, measure};
use elements_ember::solver;

use common::{commit_label, median, shell};

type Res<T> = Result<T, Box<dyn Error>>;

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
        commit: commit_label(),
        drift: drift_from_cut_off(&frames, scene.fps),
        frames,
    };
    write_summary(&results_dir(), &summary)?;
    Ok(())
}

/// Drift from the last emitting frame on, so `drift[0]` is frame 60.
fn drift_from_cut_off(frames: &[FrameMetrics], fps: f64) -> Vec<f64> {
    let from = (EMISSION_FRAMES[1] - 1) as usize;
    let mass: Vec<f64> = frames.iter().map(|f| f.mass_below).collect();
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

/// Run the scene script under `/usr/bin/time -l`, returning peak resident
/// bytes and the tail of Blender's stderr, for errors found after it exits.
/// `-l` prints "maximum resident set size" in bytes on macOS.
fn run_blender(scene_json: &Path, out: &Path, no_bake: bool) -> Res<(u64, String)> {
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
        .arg(out);
    if no_bake {
        cmd.arg("--no-bake");
    }
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
        .ok_or("no \"maximum resident set size\" in /usr/bin/time -l output")?;
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

fn run_mantaflow(name: &str, res: u32) -> Res<()> {
    let scene = scene(name, res)?;
    let context =
        |e: Box<dyn Error>| -> Box<dyn Error> { format!("mantaflow {name} {res}³: {e}").into() };
    let scratch = workspace().join(format!("target/bench/{name}-{res}"));
    std::fs::create_dir_all(&scratch)?;
    let scene_json = scratch.join("scene.json");
    std::fs::write(
        &scene_json,
        serde_json::to_string_pretty(&scene.mantaflow_json())? + "\n",
    )?;

    eprintln!("mantaflow {name} {res}³: baseline (no bake)");
    let base_dir = scratch.join("baseline");
    clear_dir(&base_dir)?;
    let (baseline, _) = run_blender(&scene_json, &base_dir, true).map_err(context)?;

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
        let (rss, stderr) = run_blender(&scene_json, &out, false).map_err(context)?;
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
    for n in 1..=scene.frames {
        let path = out.join(format!("cache/data/fluid_data_{n:04}.vdb"));
        let f = common::mantaflow::read_frame(&path, cells, dx)
            .map_err(|e| context(with_stderr(e, &tail)))?;
        frames.push(measure(&Sample {
            cells,
            dx,
            density: &f.density,
            faces: &f.faces,
            solid: &solid,
        }));
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
        commit: commit_label(),
        drift: drift_from_cut_off(&frames, scene.fps),
        frames,
    };
    write_summary(&results_dir(), &summary)?;
    Ok(())
}

/// Build `docs/bench/results.md` from every result file, or name the
/// missing ones and fail.
fn report() -> Res<()> {
    let dir = results_dir();
    let mut summaries = Vec::new();
    if dir.exists() {
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "json") {
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
        ["report"] => report(),
        ["scene-json", name, res] => {
            let json = scene(name, res.parse()?)?.mantaflow_json();
            println!("{}", serde_json::to_string_pretty(&json)?);
            Ok(())
        }
        _ => Err(
            "usage: benchmark ember SCENE RES | scene-json SCENE RES | mantaflow SCENE RES | report"
                .into(),
        ),
    }
}
