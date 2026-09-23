//! The Mantaflow benchmark (2b-3 spec). Run everything with `just bench`.
//!
//! Modes:
//!   benchmark ember SCENE RES      time and measure Ember
//!   benchmark mantaflow SCENE RES  bake, time and measure Mantaflow (Task 8)
//!   benchmark report               build docs/bench/results.md (Task 8)
//!
//! Each run writes docs/bench/results/{solver}-{scene}-{res}.json and .csv.

// The report mode (Task 8) uses the rest of the shared helpers.
#[allow(dead_code)]
mod common;

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use elements_core::gpu::{Axis, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{NodeRegistry, StateStore, Time};
use elements_ember::bench::report::{RunSummary, load_average, write_summary};
use elements_ember::bench::{EMISSION_FRAMES, SOLVER_NODE, Scene};
use elements_ember::metrics::{FrameMetrics, Sample, drift, measure};
use elements_ember::solver;

use common::median;

type Res<T> = Result<T, Box<dyn Error>>;

fn results_dir() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/bench/results"
    ))
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
    // The 0-based index of the last emitting frame, so `drift[0]` is frame 60.
    let from = (EMISSION_FRAMES[1] - 1) as usize;
    let mass: Vec<f64> = frames.iter().map(|f| f.mass_below).collect();
    let outflow: Vec<f64> = frames.iter().map(|f| f.outflow_rate).collect();
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
        drift: drift(&mass, &outflow, 1.0 / scene.fps, from)[from..].to_vec(),
        frames,
    };
    write_summary(&results_dir(), &summary)?;
    Ok(())
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["ember", name, res] => run_ember(name, res.parse()?),
        _ => Err("usage: benchmark ember SCENE RES | mantaflow SCENE RES | report".into()),
    }
}
