//! The piece 2a speed gate (spec §4.3). Run with `just bench-gate`.
//!
//! Writes `docs/bench/speed-gate.md`. The decision line at its end is filled
//! in by hand after the user decides; this program only applies the rule.
//!
//! `just bench-sweep` instead sweeps the counts in `SPEED_GATE_ITERATIONS`
//! (comma-separated) into `docs/bench/iteration-sweep.md`. That is exploration
//! for choosing presets, not the gate, so it never touches the gate's record.
//!
//! Since piece 2b-1 the plume scene uses the preview preset (MacCormack, CFL
//! substeps), so a rerun no longer reproduces 2a's table. `speed-gate.md`
//! records the commit its numbers came from.

mod common;

use std::error::Error;
use std::time::Instant;

use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{NodeRegistry, StateStore, Time};
use elements_ember::bench::{GATE_RATIO, GATE_STEP_MS, GateRow, Scene, gate_verdict};
use elements_ember::emitter::fill_sphere;
use elements_ember::metrics::{DivergenceStats, divergence};
use elements_ember::solver::{SolverState, Sources, Substep, substep};

use common::{commit_label, median, shell};

const RESOLUTION: u32 = 128;
const ITERATIONS: [u32; 4] = [20, 40, 80, 160];
const RUNS: usize = 3;
const WARMUP: u32 = 24;
const TIMED: u32 = 24;

type Res<T> = Result<T, Box<dyn Error>>;

struct Timing {
    /// Each run's median step, ms.
    run_medians: Vec<f64>,
    /// Every timed step of every run, ms.
    all: Vec<f64>,
    /// Every snapshot, ms.
    snapshots: Vec<f64>,
}

fn wait(gpu: &GpuContext) -> Res<()> {
    Ok(gpu.wait()?)
}

/// Step the scene's graph through `eval_frame`, the path the timeline uses,
/// timing each timed frame as `eval` plus a blocking poll.
fn time_scene(
    gpu: &GpuContext,
    registry: &NodeRegistry,
    scene: &Scene,
    timing: &mut Timing,
) -> Res<()> {
    let doc = scene.document();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(registry)?;
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let mut steps = Vec::new();
    let first = config.start_frame;
    for frame in first..first + WARMUP + TIMED {
        let time = Time::at(frame, first, config.fps);
        let start = Instant::now();
        let evaluated = graph.eval_frame(gpu, &mut pool, &mut pipelines, &mut state, time, dims)?;
        wait(gpu)?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        evaluated.value.release_to(&mut pool);
        if frame >= first + WARMUP {
            steps.push(ms);
            let start = Instant::now();
            let snapshot = state.snapshot(gpu, &mut pool)?;
            wait(gpu)?;
            timing.snapshots.push(start.elapsed().as_secs_f64() * 1e3);
            snapshot.release_to(&mut pool);
        }
    }
    timing.run_medians.push(median(&steps));
    timing.all.extend(steps);
    state.clear(&mut pool);
    Ok(())
}

/// Reach the frame-48 state through the kernel API, then measure divergence
/// before and after one projection with `n` iterations.
fn divergence_at(
    gpu: &GpuContext,
    scene: &Scene,
    frames: u32,
) -> Res<(DivergenceStats, DivergenceStats)> {
    let [x, y, z] = scene.cells;
    let cells = FieldDims::new(x, y, z);
    let dx = (scene.domain_size / x.max(y).max(z) as f64) as f32;
    let n = scene.solver.pressure_iterations;
    let substeps = scene.solver.max_substeps;
    let constants =
        scene
            .solver
            .step_constants(cells, (1.0 / scene.fps / substeps as f64) as f32, dx);
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let density_source = pool.acquire(gpu, cells, FieldFormat::R32Float)?;
    let temperature_source = pool.acquire(gpu, cells, FieldFormat::R32Float)?;
    fill_sphere(
        gpu,
        &mut cache,
        &density_source,
        &temperature_source,
        &scene.emitter,
        dx,
    )?;
    let sources = Sources::new(&density_source, &temperature_source);
    let mut state = SolverState::zeroed(gpu, &mut cache, &mut pool, cells)?;
    for _ in 0..frames * substeps {
        substep(
            gpu, &mut cache, &mut pool, &mut state, sources, &constants, n,
        )?;
    }

    let mut step = Substep::new(gpu, &constants)?;
    step.pre_projection(gpu, &mut cache, &mut pool, &mut state, sources)?;
    step.submit(gpu, &mut pool)?;
    let before = divergence(&state.read_velocity(gpu)?, cells, dx);

    let mut step = Substep::new(gpu, &constants)?;
    step.project(gpu, &mut cache, &mut pool, &mut state, n)?;
    step.submit(gpu, &mut pool)?;
    let after = divergence(&state.read_velocity(gpu)?, cells, dx);
    Ok((before, after))
}

fn main() -> Res<()> {
    let gpu = GpuContext::new_headless()?;
    let registry = elements_ember::registry();
    let mut table = String::new();
    let mut rows = Vec::new();
    let scene_substeps = Scene::plume(RESOLUTION).solver.max_substeps;
    let sweep: Option<Vec<u32>> = match std::env::var("SPEED_GATE_ITERATIONS") {
        Ok(list) => Some(
            list.split(',')
                .map(|n| n.trim().parse::<u32>())
                .collect::<Result<_, _>>()?,
        ),
        Err(_) => None,
    };
    let iterations = sweep.clone().unwrap_or_else(|| ITERATIONS.to_vec());

    for n in iterations {
        let scene = Scene::plume(RESOLUTION).with_iterations(n);
        let mut timing = Timing {
            run_medians: Vec::new(),
            all: Vec::new(),
            snapshots: Vec::new(),
        };
        for run in 0..RUNS {
            eprintln!("N = {n}: run {} of {RUNS}", run + 1);
            time_scene(&gpu, &registry, &scene, &mut timing)?;
        }
        eprintln!("N = {n}: divergence");
        let (before, after) = divergence_at(&gpu, &scene, WARMUP + TIMED)?;
        let step = median(&timing.run_medians);
        let ratio = after.rms / before.rms;
        let min = timing.all.iter().copied().fold(f64::INFINITY, f64::min);
        let max = timing.all.iter().copied().fold(0.0, f64::max);
        let gate_row = GateRow {
            iterations: n,
            step_ms_median: step,
            ratio,
        };
        table.push_str(&format!(
            "| {n} | {step:.2} ({min:.2}–{max:.2}) | {:.2} | {:.3e} | {:.3e} | {ratio:.4} | {:.3e} | {} |\n",
            median(&timing.snapshots),
            before.rms,
            after.rms,
            after.max_abs,
            if gate_row.passes() { "yes" } else { "no" },
        ));
        rows.push(gate_row);
    }

    let verdict = match gate_verdict(&rows) {
        Some(n) => format!("**PASS**, provisional `pressure_iterations` = {n}"),
        None => "**FAIL**: no N meets both limits".to_owned(),
    };
    let commit = commit_label();
    let substeps = scene_substeps;
    let (title, file, decision) = if sweep.is_some() {
        (
            "Ember pressure-iteration sweep (exploration, not the gate)",
            "iteration-sweep.md",
            "The gate's own record is `speed-gate.md`; this sweep only informs 2b's presets.\n",
        )
    } else {
        (
            "Ember speed gate (piece 2a)",
            "speed-gate.md",
            "Decision (recorded by the user): _pending_\n",
        )
    };
    let report = format!(
        "# {title}\n\n\
         - Machine: {cpu} ({adapter})\n\
         - OS: macOS {os}\n\
         - Ember commit: {commit}\n\
         - Date: {date}\n\
         - Scene: `plume`, {RESOLUTION}³, max_substeps {substeps}. Frames {first}–{last} timed after \
         {WARMUP} warm-up frames, as `eval` plus a blocking poll; median of {RUNS} runs' medians. \
         The step ms min–max range is pooled over all timed frames of all {RUNS} runs, not a \
         single run. Divergence from the frame-{last} state.\n\n\
         | N | step ms (median, min–max) | snapshot ms | RMS div before | RMS div after | ratio | max div after | pass |\n\
         |---|---|---|---|---|---|---|---|\n\
         {table}\n\
         Pre-registered rule (spec §4.3): PASS if some N has a median step of at most \
         {GATE_STEP_MS} ms and a ratio of at most {GATE_RATIO}; the provisional default is the \
         largest passing N. The `pass` column applies this same per-row rule \
         (`GateRow::passes`).\n\n\
         Rule applied: {verdict}.\n\n\
         Note: the ratio measures residual divergence, which weights high frequencies. The smooth \
         pressure error Gauss–Seidel leaves behind shows in plume shape, not in this number.\n\n\
         {decision}",
        cpu = shell("sysctl", &["-n", "machdep.cpu.brand_string"]),
        adapter = gpu.adapter_name(),
        os = shell("sw_vers", &["-productVersion"]),
        date = shell("date", &["-u", "+%Y-%m-%d"]),
        first = 1 + WARMUP,
        last = WARMUP + TIMED,
    );
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/bench");
    std::fs::create_dir_all(dir)?;
    std::fs::write(format!("{dir}/{file}"), &report)?;
    println!("{report}");
    Ok(())
}
