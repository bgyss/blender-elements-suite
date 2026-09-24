//! The preview preset's substep cap (2b-1 spec §6). Run with
//! `just bench-presets`.
//!
//! Appends a section to `docs/bench/presets.md`, or to
//! `presets-semi-lagrangian.md` when run with
//! `PRESET_ADVECTION=semi_lagrangian` (the rule's fallback), so the decisions
//! recorded there by hand are kept. The decision line at the section's end is
//! filled in by hand after the user decides; this program only applies the
//! rule.
//!
//! The scene runs the preview preset's own pressure solve and mass
//! correction; only the substep cap, the advection and the CFL number are
//! set here.

mod common;

use std::error::Error;
use std::time::Instant;

use elements_core::gpu::{FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{NodeRegistry, StateStore, Time};
use elements_ember::bench::{PRESET_FRAME_MS, PresetRow, Scene, preview_substeps_verdict};
use elements_ember::kernels::Advection;
use elements_ember::solver::{PressureSolver, Quality};

use common::{commit_label, median, shell};

const RESOLUTION: u32 = 128;
const CAPS: [u32; 4] = [1, 2, 3, 4];
const RUNS: usize = 3;
const WARMUP: u32 = 24;
const TIMED: u32 = 24;

type Res<T> = Result<T, Box<dyn Error>>;

/// One run of `scene`: each timed frame's full time in ms (`eval_frame`,
/// which includes the CFL measurement, plus a blocking wait), and how many
/// timed frames hit the substep cap.
fn run(gpu: &GpuContext, registry: &NodeRegistry, scene: &Scene) -> Res<(Vec<f64>, u32)> {
    let doc = scene.document();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(registry)?;
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let mut frames = Vec::new();
    let mut clamped = 0;
    let first = config.start_frame;
    for frame in first..first + WARMUP + TIMED {
        let time = Time::at(frame, first, config.fps);
        let start = Instant::now();
        let evaluated = graph.eval_frame(gpu, &mut pool, &mut pipelines, &mut state, time, dims)?;
        gpu.wait()?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        if frame >= first + WARMUP {
            frames.push(ms);
            clamped += evaluated.stats.cfl_clamped;
        }
        evaluated.value.release_to(&mut pool);
    }
    state.clear(&mut pool);
    Ok((frames, clamped))
}

fn main() -> Res<()> {
    let gpu = GpuContext::new_headless()?;
    let registry = elements_ember::registry();
    let (advection, file) = match std::env::var("PRESET_ADVECTION").as_deref() {
        Ok("semi_lagrangian") => (Advection::SemiLagrangian, "presets-semi-lagrangian.md"),
        _ => (Advection::MacCormack, "presets.md"),
    };
    let load_before = shell("sysctl", &["-n", "vm.loadavg"]);
    let mut table = String::new();
    let mut rows = Vec::new();
    for cap in CAPS {
        let mut scene = Scene::plume(RESOLUTION).with_max_substeps(cap);
        let preview = Quality::Preview.params();
        scene.solver.pressure_solver = preview.pressure_solver;
        scene.solver.pressure_cycles = preview.pressure_cycles;
        scene.solver.pressure_iterations = preview.pressure_iterations;
        scene.solver.conserve_mass = preview.conserve_mass;
        scene.solver.cfl = 1.0;
        scene.solver.advection = advection;
        scene.solver.vorticity = 0.0;
        let mut medians = Vec::new();
        let mut all = Vec::new();
        let mut clamped = 0;
        for r in 0..RUNS {
            eprintln!("max_substeps = {cap}: run {} of {RUNS}", r + 1);
            let (frames, c) = run(&gpu, &registry, &scene)?;
            medians.push(median(&frames));
            all.extend(frames);
            clamped += c;
        }
        let row = PresetRow {
            max_substeps: cap,
            frame_ms_median: median(&medians),
        };
        let min = all.iter().copied().fold(f64::INFINITY, f64::min);
        let max = all.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        table.push_str(&format!(
            "| {cap} | {:.2} ({min:.2}–{max:.2}) | {clamped} of {} | {} |\n",
            row.frame_ms_median,
            TIMED as usize * RUNS,
            if row.passes() { "yes" } else { "no" },
        ));
        rows.push(row);
    }

    let verdict = match preview_substeps_verdict(&rows) {
        Some(n) => format!("**preview `max_substeps` = {n}**"),
        None => {
            "**no cap fits**: rerun with `PRESET_ADVECTION=semi_lagrangian` (spec §6)".to_owned()
        }
    };
    let preview = Quality::Preview.params();
    let pressure = match preview.pressure_solver {
        PressureSolver::GaussSeidel => format!("Gauss–Seidel ×{}", preview.pressure_iterations),
        PressureSolver::Multigrid => format!("multigrid ×{}", preview.pressure_cycles),
        PressureSolver::Mgpcg => format!("MGPCG ×{}", preview.pressure_cycles),
    };
    let body = format!(
        "- Machine: {cpu} ({adapter})\n\
         - OS: macOS {os}\n\
         - Ember commit: {commit}\n\
         - Date: {date}\n\
         - Scene: `plume`, {RESOLUTION}³, preview's pressure solve ({pressure}), mass \
         correction {correction}, cfl 1.0, advection {advection:?}, vorticity 0. Frames {first}–{last} timed after {WARMUP} warm-up frames, each as \
         `eval_frame` (the CFL measurement and every substep) plus a blocking wait; median of \
         {RUNS} runs' medians. The min–max range is pooled over all timed frames of all runs.\n\n\
         | max_substeps | frame ms (median, min–max) | frames CFL-clamped | pass |\n\
         |---|---|---|---|\n\
         {table}\n\
         Pre-registered rule (2b-1 spec §6): preview's `max_substeps` is the largest cap whose \
         median full frame is at most {PRESET_FRAME_MS} ms. If even 1 fails, preview falls back \
         to semi-Lagrangian advection and the sweep runs again.\n\n\
         Rule applied: {verdict}.\n\n\
         Load average (1, 5, 15 minutes): {load_before} before the run, {load_after} after it.\n\n\
         Decision (recorded by the user): _pending_\n",
        correction = if preview.conserve_mass { "on" } else { "off" },
        load_after = shell("sysctl", &["-n", "vm.loadavg"]),
        cpu = shell("sysctl", &["-n", "machdep.cpu.brand_string"]),
        adapter = gpu.adapter_name(),
        os = shell("sw_vers", &["-productVersion"]),
        commit = commit_label(),
        date = shell("date", &["-u", "+%Y-%m-%d"]),
        first = 1 + WARMUP,
        last = WARMUP + TIMED,
    );
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/bench");
    std::fs::create_dir_all(dir)?;
    let path = format!("{dir}/{file}");
    let report = match std::fs::read_to_string(&path) {
        Ok(existing) => format!(
            "{}\n\n## Run of {} at {}\n\n{body}",
            existing.trim_end(),
            shell("date", &["-u", "+%Y-%m-%d"]),
            commit_label(),
        ),
        Err(_) => format!("# Ember preview preset sweep (piece 2b-1)\n\n{body}"),
    };
    std::fs::write(&path, &report)?;
    println!("{body}");
    Ok(())
}
