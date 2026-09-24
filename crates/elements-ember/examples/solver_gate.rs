//! The 2b-3c solver gate (spec §4). Run with `just bench-solver`.
//!
//! Decides whether multigrid V-cycles or MGPCG replace the 160-iteration
//! Gauss–Seidel pressure solve in the presets. The rule was fixed before any
//! numbers existed; this program applies it and writes
//! `docs/bench/solver-gate.md`. The decision line at its end is filled in by
//! hand after the user decides.
//!
//! The sweep is staged to keep the run near an hour. At the low resolution
//! counts rise until the divergence rule holds in `plume`, `plume_collider`
//! and `plume_wind` and the thin-plate rule holds in `plume_plate`; that
//! count is checked at the high resolution, stepping up if needed; only the
//! final candidates are timed against Gauss–Seidel, interleaved in this one
//! process so drifting machine load hits every solver alike.
//!
//! Timing is per frame (`eval_frame` plus a blocking wait), not per solve:
//! runs of one scene differ only in the pressure solver, so comparing frame
//! times is the same comparison without instrumenting the solver.
//!
//! `SOLVER_GATE_SMOKE=1` runs a short plumbing pass at 32³ and 64³ with small
//! caps and writes to `$SOLVER_GATE_OUT` (default: the system temp dir)
//! instead of the gate's record.

mod common;

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use elements_core::gpu::{
    ComputeBatch, FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache,
};
use elements_core::graph::{NodeRegistry, StateStore, Time};
use elements_ember::bench::{SOLVER_NODE, Scene};
use elements_ember::cfl;
use elements_ember::collider::{ColliderFields, fill_collider};
use elements_ember::kernels::{self, Solids, StepConstants, Uniforms};
use elements_ember::metrics::{DivergenceStats, Sample, divergence, measure};
use elements_ember::shape_emitter::{EmitterFields, fill_emitter};
use elements_ember::solver::{
    self, PressureSolve, PressureSolver, SolverState, Sources, Substep, substep,
};
use elements_ember::unions::union_colliders;

use common::{commit_label, median, shell};

type Res<T> = Result<T, Box<dyn Error>>;

/// Today's preview solve, the baseline every candidate is timed against.
const GS_ITERATIONS: u32 = 160;
/// The frames whose divergence the rule reads.
const MEASURED: [u32; 2] = [60, 120];
/// The frames each timed run steps, and the ones its median covers.
const TIMED_FRAMES: u32 = 48;
const TIMED_FROM: u32 = 25;
/// `plume_plate`'s rule: RMS divergence after projection over before it.
const PLATE_RATIO: f64 = 1e-3;
/// Past-floor stability: this count's divergence in `plume_collider` must be
/// at most `STABILITY_FACTOR` times the chosen count's.
const STABILITY_COUNT: u32 = 40;
const STABILITY_FACTOR: f64 = 4.0;
/// A load average above this is flagged in the record.
const LOAD_FLAG: f64 = 2.0;

const SCENES: [&str; 3] = ["plume", "plume_collider", "plume_wind"];

/// Mantaflow's masked divergence RMS at frames 60 and 120, from
/// `docs/bench/results.md` (the 7fe5d9d run), at [128³, 256³]. `plume_wind`
/// uses `plume`'s until the 2b-3c rerun gives the new wind a reference
/// (spec §4).
fn mantaflow(scene: &str) -> [[f64; 2]; 2] {
    match scene {
        "plume" | "plume_wind" => [[1.49e-4, 1.07e-4], [7.41e-5, 4.51e-5]],
        "plume_collider" => [[2.84e-4, 1.09e-4], [9.12e-5, 5.25e-5]],
        _ => unreachable!("no Mantaflow reference for {scene}"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Method {
    Multigrid,
    Mgpcg,
}

impl Method {
    fn name(self) -> &'static str {
        match self {
            Self::Multigrid => "multigrid",
            Self::Mgpcg => "mgpcg",
        }
    }

    fn solve(self, count: u32) -> PressureSolve {
        match self {
            Self::Multigrid => PressureSolve::Multigrid(count),
            Self::Mgpcg => PressureSolve::Mgpcg(count),
        }
    }
}

fn label(solve: PressureSolve) -> String {
    match solve {
        PressureSolve::GaussSeidel(n) => format!("gauss_seidel ×{n}"),
        PressureSolve::Multigrid(n) => format!("multigrid ×{n}"),
        PressureSolve::Mgpcg(n) => format!("mgpcg ×{n}"),
    }
}

/// What this run covers: the gate, or the smoke pass.
struct Plan {
    /// [low, high] resolution.
    res: [u32; 2],
    /// Highest count swept, per method.
    multigrid_cap: u32,
    mgpcg_cap: u32,
    runs: usize,
    out: PathBuf,
    smoke: bool,
}

impl Plan {
    fn from_env() -> Self {
        if std::env::var_os("SOLVER_GATE_SMOKE").is_some() {
            let out = std::env::var_os("SOLVER_GATE_OUT")
                .map(PathBuf::from)
                .unwrap_or_else(|| std::env::temp_dir().join("solver-gate-smoke.md"));
            Self {
                res: [32, 64],
                multigrid_cap: 3,
                mgpcg_cap: 16,
                runs: 1,
                out,
                smoke: true,
            }
        } else {
            Self {
                res: [128, 256],
                // Spec §4: V-cycles 1–8; MGPCG 1–16, and up to 24 for the
                // thin plate (the user's amendment).
                multigrid_cap: 8,
                mgpcg_cap: 24,
                runs: 3,
                out: PathBuf::from(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../docs/bench/solver-gate.md"
                )),
                smoke: false,
            }
        }
    }

    fn cap(&self, method: Method) -> u32 {
        match method {
            Method::Multigrid => self.multigrid_cap,
            Method::Mgpcg => self.mgpcg_cap,
        }
    }
}

fn scene(name: &str, res: u32) -> Scene {
    match name {
        "plume" => Scene::plume(res),
        "plume_collider" => Scene::plume_collider(res),
        "plume_wind" => Scene::plume_wind(res),
        "plume_plate" => Scene::plume_plate(res),
        _ => unreachable!("unknown scene {name}"),
    }
}

/// `scene` with its pressure solve replaced by `solve`.
fn configure(mut scene: Scene, solve: PressureSolve) -> Scene {
    let s = &mut scene.solver;
    match solve {
        PressureSolve::GaussSeidel(n) => {
            s.pressure_solver = PressureSolver::GaussSeidel;
            s.pressure_iterations = n;
        }
        PressureSolve::Multigrid(n) => {
            s.pressure_solver = PressureSolver::Multigrid;
            s.pressure_cycles = n;
        }
        PressureSolve::Mgpcg(n) => {
            s.pressure_solver = PressureSolver::Mgpcg;
            s.pressure_cycles = n;
        }
    }
    scene
}

fn load() -> String {
    shell("sysctl", &["-n", "vm.loadavg"])
}

/// The 1-minute figure of a `vm.loadavg` string.
fn load_1m(text: &str) -> f64 {
    text.split_whitespace()
        .find_map(|t| t.parse::<f64>().ok())
        .unwrap_or(-1.0)
}

struct Clock(Instant);

impl Clock {
    fn log(&self, msg: impl AsRef<str>) {
        eprintln!(
            "[{:7.1} s] {}",
            self.0.elapsed().as_secs_f64(),
            msg.as_ref()
        );
    }
}

/// The metrics run through `eval_frame`, the path the timeline uses: masked
/// divergence RMS (2b-3 spec §4.1) at the `MEASURED` frames. Read back only
/// on those frames. Also returns how many cells each measurement covered: an
/// empty smoke mask gives an RMS of 0.
fn masked_divergence(
    gpu: &GpuContext,
    registry: &NodeRegistry,
    scene: &Scene,
) -> Res<([f64; 2], [u64; 2])> {
    let doc = scene.document();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(registry)?;
    let dx = scene.domain_size / f64::from(*scene.cells.iter().max().unwrap());
    let solid = scene.solid_mask();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let mut out = [f64::NAN; 2];
    let mut cells = [0u64; 2];
    let first = config.start_frame;
    let result = (|| -> Res<()> {
        for frame in first..=MEASURED[1] {
            let time = Time::at(frame, first, config.fps);
            let evaluated =
                graph.eval_frame(gpu, &mut pool, &mut pipelines, &mut state, time, dims)?;
            evaluated.value.release_to(&mut pool);
            let Some(slot) = MEASURED.iter().position(|&m| m == frame) else {
                continue;
            };
            let density = state
                .get(SOLVER_NODE, solver::DENSITY)
                .ok_or("no solver density in state")?
                .as_field()?
                .read_back(gpu)?;
            let velocity = state
                .get(SOLVER_NODE, solver::VELOCITY)
                .ok_or("no solver velocity in state")?
                .as_vector_field()?;
            let faces = velocity_faces(gpu, velocity)?;
            let metrics = measure(&Sample {
                cells: dims,
                dx,
                density: &density,
                faces: &faces,
                solid: &solid,
                open_mask: scene.solver.boundaries.open_mask(),
            });
            out[slot] = metrics.divergence_rms;
            cells[slot] = metrics.measured_cells;
        }
        Ok(())
    })();
    state.clear(&mut pool);
    result.map(|()| (out, cells))
}

fn velocity_faces(
    gpu: &GpuContext,
    velocity: &elements_core::gpu::StaggeredField,
) -> Res<[Vec<f32>; 3]> {
    use elements_core::gpu::Axis;
    Ok([
        velocity.face(Axis::X).read_back(gpu)?,
        velocity.face(Axis::Y).read_back(gpu)?,
        velocity.face(Axis::Z).read_back(gpu)?,
    ])
}

/// One timed run through `eval_frame`: frames 1–48, each `eval` plus a
/// blocking wait; the median of frames 25–48, ms.
fn timed_run(gpu: &GpuContext, registry: &NodeRegistry, scene: &Scene) -> Res<f64> {
    let doc = scene.document();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(registry)?;
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let mut times = Vec::new();
    let first = config.start_frame;
    let result = (|| -> Res<()> {
        for frame in first..first + TIMED_FRAMES {
            let time = Time::at(frame, first, config.fps);
            let start = Instant::now();
            let evaluated =
                graph.eval_frame(gpu, &mut pool, &mut pipelines, &mut state, time, dims)?;
            gpu.wait()?;
            let ms = start.elapsed().as_secs_f64() * 1e3;
            evaluated.value.release_to(&mut pool);
            if frame >= TIMED_FROM {
                times.push(ms);
            }
        }
        Ok(())
    })();
    state.clear(&mut pool);
    result.map(|()| median(&times))
}

/// Divergence before and after one frame's projection.
type Around = (DivergenceStats, DivergenceStats);

/// Step `scene` through the kernel API, doing what `ember.smoke_solver` does
/// for its document, up to and including frame `last`. On each frame in
/// `instrument`, the first substep is split, as the speed gate's
/// `divergence_at` does: stages 1–3, measure, project with `solve`,
/// measure, then advect the scalars. Returns those measurements and the
/// velocity after `last`.
///
/// The collider mask is built from the colliders' union by `solidify`, as
/// the solver builds it, and checked against `Scene::solid_mask`.
fn kernel_path(
    gpu: &GpuContext,
    scene: &Scene,
    solve: PressureSolve,
    last: u32,
    instrument: &[u32],
) -> Res<(Vec<Around>, [Vec<f32>; 3])> {
    let [x, y, z] = scene.cells;
    let cells = FieldDims::new(x, y, z);
    let dx = (scene.domain_size / f64::from(x.max(y).max(z))) as f32;
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();

    // The colliders, merged into one SDF and velocity, then the mask.
    let mut merged: Option<(
        elements_core::gpu::Field,
        elements_core::gpu::StaggeredField,
    )> = None;
    for collider in &scene.colliders {
        let sdf = pool.acquire(gpu, cells, FieldFormat::R32Float)?;
        let velocity = pool.acquire_staggered_uninit(gpu, cells)?;
        let pose = collider.transform.pose(1.0, 1.0 / scene.fps);
        fill_collider(
            gpu,
            &mut cache,
            collider,
            &pose,
            dx,
            ColliderFields {
                sdf: &sdf,
                velocity: &velocity,
            },
        )?;
        merged = Some(match merged {
            None => (sdf, velocity),
            Some((a_sdf, a_velocity)) => {
                let out_sdf = pool.acquire(gpu, cells, FieldFormat::R32Float)?;
                let out_velocity = pool.acquire_staggered_uninit(gpu, cells)?;
                union_colliders(
                    gpu,
                    &mut cache,
                    ColliderFields {
                        sdf: &a_sdf,
                        velocity: &a_velocity,
                    },
                    ColliderFields {
                        sdf: &sdf,
                        velocity: &velocity,
                    },
                    ColliderFields {
                        sdf: &out_sdf,
                        velocity: &out_velocity,
                    },
                )?;
                (out_sdf, out_velocity)
            }
        });
    }
    let mask = match &merged {
        None => None,
        Some((sdf, _)) => {
            let mask = pool.acquire(gpu, cells, FieldFormat::R32Float)?;
            let u = Uniforms::new(gpu, &StepConstants::new(cells, 1.0, dx))?;
            let mut batch = ComputeBatch::new();
            kernels::solidify(gpu, &mut cache, &mut batch, &u, sdf, &mask)?;
            batch.submit(gpu)?;
            let gpu_mask = mask.read_back(gpu)?;
            let cpu_mask = scene.solid_mask();
            let differ = gpu_mask
                .iter()
                .zip(&cpu_mask)
                .filter(|(g, c)| (**g > 0.5) != **c)
                .count();
            if differ != 0 {
                return Err(format!(
                    "{}: the GPU mask differs from solid_mask in {differ} cells",
                    scene.name
                )
                .into());
            }
            Some(mask)
        }
    };
    let solids = match (&mask, &merged) {
        (Some(mask), Some((_, velocity))) => Some(Solids { mask, velocity }),
        _ => None,
    };

    let density_source = pool.acquire(gpu, cells, FieldFormat::R32Float)?;
    let temperature_source = pool.acquire(gpu, cells, FieldFormat::R32Float)?;
    let weight = pool.acquire(gpu, cells, FieldFormat::R32Float)?;
    let target = pool.acquire_staggered_uninit(gpu, cells)?;
    let zero_density = pool.acquire_zeroed(gpu, &mut cache, cells)?;
    let zero_temperature = pool.acquire_zeroed(gpu, &mut cache, cells)?;
    let mut state = SolverState::zeroed(gpu, &mut cache, &mut pool, cells)?;
    let params = &scene.solver;
    let mut measured = Vec::new();
    for frame in 1..=last {
        let time = Time::at(frame, 1, scene.fps);
        let mut sources = if scene.emitter.is_active(frame) {
            let pose = scene.emitter.transform.pose(f64::from(frame), time.dt);
            fill_emitter(
                gpu,
                &mut cache,
                &scene.emitter,
                &pose,
                time.seconds,
                dx,
                EmitterFields {
                    density: &density_source,
                    temperature: &temperature_source,
                    weight: &weight,
                    velocity: &target,
                },
            )?;
            Sources::new(&density_source, &temperature_source)
        } else {
            Sources::new(&zero_density, &zero_temperature)
        };
        if let Some(s) = solids {
            sources = sources.with_solids(s);
        }
        let speed = cfl::measure_speed(gpu, &mut cache, &state.velocity)?;
        let plan = cfl::plan_substeps(speed, time.dt, dx, params.cfl, params.max_substeps)
            .ok_or_else(|| format!("{}: diverged entering frame {frame}", scene.name))?;
        let constants = StepConstants {
            has_solids: solids.is_some(),
            ..params.step_constants(cells, (time.dt / f64::from(plan.count)) as f32, dx)
        };
        for n in 0..plan.count {
            if n > 0 || !instrument.contains(&frame) {
                substep(
                    gpu, &mut cache, &mut pool, &mut state, sources, &constants, solve,
                )?;
                continue;
            }
            let mut step = Substep::new(gpu, &constants)?;
            step.pre_projection(gpu, &mut cache, &mut pool, &mut state, sources)?;
            step.submit(gpu, &mut pool)?;
            let before = divergence(&state.read_velocity(gpu)?, cells, dx);
            let mut step = Substep::new(gpu, &constants)?;
            step.project(gpu, &mut cache, &mut pool, &mut state, solve, solids)?;
            step.submit(gpu, &mut pool)?;
            let after = divergence(&state.read_velocity(gpu)?, cells, dx);
            let mut step = Substep::new(gpu, &constants)?;
            step.advect_scalars(gpu, &mut cache, &mut pool, &mut state, solids)?;
            step.submit(gpu, &mut pool)?;
            measured.push((before, after));
        }
    }
    let velocity = state.read_velocity(gpu)?;
    Ok((measured, velocity))
}

/// `plume_plate`'s ratios, after over before, at the `MEASURED` frames.
fn plate_ratios(gpu: &GpuContext, res: u32, solve: PressureSolve) -> Res<[f64; 2]> {
    let scene = configure(scene("plume_plate", res), solve);
    let (measured, _) = kernel_path(gpu, &scene, solve, MEASURED[1], &MEASURED)?;
    Ok([0, 1].map(|i| measured[i].1.rms / measured[i].0.rms))
}

/// The kernel path against `eval_frame` for `plume_plate`, through an
/// instrumented frame: the largest face-velocity difference after `last`.
/// 0 means `kernel_path` reproduces the solver's frames bit for bit.
fn kernel_path_check(
    gpu: &GpuContext,
    registry: &NodeRegistry,
    res: u32,
    solve: PressureSolve,
    last: u32,
) -> Res<(f32, usize)> {
    let scene = configure(scene("plume_plate", res), solve);
    let (_, kernel) = kernel_path(gpu, &scene, solve, last, &[MEASURED[0]])?;
    let doc = scene.document();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(registry)?;
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    for frame in 1..=last {
        let time = Time::at(frame, config.start_frame, config.fps);
        let evaluated = graph.eval_frame(gpu, &mut pool, &mut pipelines, &mut state, time, dims)?;
        evaluated.value.release_to(&mut pool);
    }
    let velocity = state
        .get(SOLVER_NODE, solver::VELOCITY)
        .ok_or("no solver velocity in state")?
        .as_vector_field()?;
    let graph_faces = velocity_faces(gpu, velocity)?;
    state.clear(&mut pool);
    let mut max = 0.0f32;
    let mut differ = 0;
    for (a, b) in kernel.iter().zip(&graph_faces) {
        for (x, y) in a.iter().zip(b) {
            if x.to_bits() != y.to_bits() {
                differ += 1;
                max = max.max((x - y).abs());
            }
        }
    }
    Ok((max, differ))
}

/// One count's divergence at one resolution.
struct Row {
    /// Per scene: masked RMS at frames 60 and 120, or the error that stopped
    /// the run. `None`: not run (an earlier scene already failed).
    scenes: Vec<Option<Result<[f64; 2], String>>>,
    /// `plume_plate`'s ratios; low resolution only.
    plate: Option<Result<[f64; 2], String>>,
}

impl Row {
    fn scene_passes(&self, i: usize, stage: usize) -> bool {
        match &self.scenes[i] {
            Some(Ok(d)) => {
                let reference = mantaflow(SCENES[i])[stage];
                (0..2).all(|f| d[f] <= reference[f])
            }
            _ => false,
        }
    }

    fn plate_passes(&self) -> bool {
        matches!(&self.plate, Some(Ok(r)) if r.iter().all(|&x| x <= PLATE_RATIO))
    }

    fn passes(&self, stage: usize) -> bool {
        (0..SCENES.len()).all(|i| self.scene_passes(i, stage))
            && (stage == 1 || self.plate_passes())
    }
}

struct Gate<'a> {
    gpu: &'a GpuContext,
    registry: &'a NodeRegistry,
    plan: &'a Plan,
    clock: Clock,
    /// (method, count, stage) → row.
    rows: BTreeMap<(Method, u32, usize), Row>,
    /// Divergence measurements whose smoke mask held no cells.
    empty: Vec<String>,
}

impl Gate<'_> {
    /// The divergence rows for `method` at `count` and resolution `stage`.
    /// At the high resolution a count stops at its first failing scene.
    fn evaluate(&mut self, method: Method, count: u32, stage: usize) -> bool {
        if let Some(row) = self.rows.get(&(method, count, stage)) {
            return row.passes(stage);
        }
        let res = self.plan.res[stage];
        let solve = method.solve(count);
        let mut row = Row {
            scenes: Vec::new(),
            plate: None,
        };
        for (i, name) in SCENES.iter().enumerate() {
            if stage == 1 && i > 0 && !row.scene_passes(i - 1, stage) {
                row.scenes.push(None);
                continue;
            }
            let s = configure(scene(name, res), solve);
            let result = masked_divergence(self.gpu, self.registry, &s);
            if let Ok((_, cells)) = &result {
                for (f, &n) in MEASURED.iter().zip(cells) {
                    if n == 0 {
                        self.empty
                            .push(format!("{res}³ `{}` {name} frame {f}", label(solve)));
                    }
                }
            }
            self.clock
                .log(format!("{res}³ {} {name}: {result:?}", label(solve)));
            let result = result.map(|(d, _)| d).map_err(|e| e.to_string());
            row.scenes.push(Some(result));
        }
        if stage == 0 {
            let result = plate_ratios(self.gpu, res, solve).map_err(|e| e.to_string());
            self.clock.log(format!(
                "{res}³ {} plume_plate ratios: {result:?}",
                label(solve)
            ));
            row.plate = Some(result);
        }
        let passes = row.passes(stage);
        self.clock.log(format!(
            "{res}³ {}: {}",
            label(solve),
            if passes { "PASSES" } else { "fails" }
        ));
        self.rows.insert((method, count, stage), row);
        passes
    }

    /// Spec §4 / brief step 2–3: the lowest count passing at the low
    /// resolution, then stepped up until it also passes at the high one.
    fn candidate(&mut self, method: Method) -> Option<u32> {
        let cap = self.plan.cap(method);
        let low = (1..=cap).find(|&c| self.evaluate(method, c, 0))?;
        (low..=cap).find(|&c| self.evaluate(method, c, 1) && self.evaluate(method, c, 0))
    }
}

fn fmt_div(d: &Option<Result<[f64; 2], String>>) -> String {
    match d {
        None => "—".to_owned(),
        Some(Ok([a, b])) => format!("{a:.2e} / {b:.2e}"),
        Some(Err(e)) => format!("error: {}", e.replace('|', "/")),
    }
}

fn main() -> Res<()> {
    let plan = Plan::from_env();
    let clock = Clock(Instant::now());
    let load_before = load();
    let commit = commit_label();
    let date = shell("date", &["-u", "+%Y-%m-%d %H:%M UTC"]);
    clock.log(format!("load before: {load_before}; commit {commit}"));
    let gpu = GpuContext::new_headless()?;
    let registry = elements_ember::registry();
    let [low, high] = plan.res;

    // Plumbing: the kernel path must reproduce the solver's frames.
    let check_solve = PressureSolve::Mgpcg(4);
    let check = kernel_path_check(&gpu, &registry, low, check_solve, MEASURED[0] + 5)?;
    clock.log(format!(
        "kernel path vs eval_frame, plume_plate {low}³ {}, frame {}: {} faces differ, max |Δ| {:e}",
        label(check_solve),
        MEASURED[0] + 5,
        check.1,
        check.0
    ));

    let mut gate = Gate {
        gpu: &gpu,
        registry: &registry,
        plan: &plan,
        clock,
        rows: BTreeMap::new(),
        empty: Vec::new(),
    };

    // Steps 2–3: sweep.
    let mut candidates: Vec<(Method, Option<u32>)> = Vec::new();
    for method in [Method::Multigrid, Method::Mgpcg] {
        let c = gate.candidate(method);
        gate.clock
            .log(format!("{} candidate: {c:?}", method.name()));
        candidates.push((method, c));
    }

    // Past-floor stability, for each candidate.
    let mut stability: BTreeMap<Method, Result<[f64; 2], String>> = BTreeMap::new();
    for &(method, count) in &candidates {
        let Some(count) = count else { continue };
        let at_count = match &gate.rows[&(method, count, 0)].scenes[1] {
            Some(Ok(d)) => *d,
            _ => unreachable!("a candidate passed plume_collider"),
        };
        let s = configure(scene("plume_collider", low), method.solve(STABILITY_COUNT));
        let result = masked_divergence(&gpu, &registry, &s)
            .map(|(d, _)| [0, 1].map(|f| d[f] / at_count[f]))
            .map_err(|e| e.to_string());
        gate.clock.log(format!(
            "stability {} ×{STABILITY_COUNT} / ×{count}: {result:?}",
            method.name()
        ));
        stability.insert(method, result);
    }

    // Step 4: timing, interleaved.
    let mut configs = vec![PressureSolve::GaussSeidel(GS_ITERATIONS)];
    configs.extend(
        candidates
            .iter()
            .filter_map(|&(m, c)| c.map(|c| m.solve(c))),
    );
    // (config index, stage, scene index) → run medians.
    let mut times: BTreeMap<(usize, usize, usize), Vec<f64>> = BTreeMap::new();
    for (stage, &res) in plan.res.iter().enumerate() {
        for (si, name) in SCENES.iter().enumerate() {
            for run in 0..plan.runs {
                for (ci, &solve) in configs.iter().enumerate() {
                    let s = configure(scene(name, res), solve);
                    let ms = timed_run(&gpu, &registry, &s)?;
                    gate.clock.log(format!(
                        "time {res}³ {name} run {} {}: {ms:.2} ms",
                        run + 1,
                        label(solve)
                    ));
                    times.entry((ci, stage, si)).or_default().push(ms);
                }
            }
        }
    }
    let frame_ms = |ci: usize, stage: usize, si: usize| median(&times[&(ci, stage, si)]);
    let faster = |ci: usize| {
        (0..2).all(|stage| {
            (0..SCENES.len()).all(|si| frame_ms(ci, stage, si) <= frame_ms(0, stage, si))
        })
    };

    let load_after = load();
    gate.clock.log(format!("load after: {load_after}"));

    // The rule.
    let mut verdicts = Vec::new();
    for (ci, &solve) in configs.iter().enumerate().skip(1) {
        let method = if matches!(solve, PressureSolve::Multigrid(_)) {
            Method::Multigrid
        } else {
            Method::Mgpcg
        };
        let stable = matches!(
            stability.get(&method),
            Some(Ok(r)) if r.iter().all(|&x| x <= STABILITY_FACTOR)
        );
        verdicts.push((method, solve, faster(ci), stable));
    }
    let winner = verdicts
        .iter()
        .filter(|(m, _, fast, stable)| *m == Method::Mgpcg && *fast && *stable)
        .map(|(_, s, _, _)| *s)
        .next();
    let multigrid_note = verdicts
        .iter()
        .find(|(m, _, fast, stable)| *m == Method::Multigrid && *fast && *stable)
        .map(|(_, s, _, _)| {
            format!(
                " {} also passes the divergence, thin-plate, timing and stability checks, but \
                 plain V-cycles cannot be the default (spec §4).",
                label(*s)
            )
        })
        .unwrap_or_default();
    let verdict = match winner {
        Some(s) => format!("**fastest passing: `{}`**.{multigrid_note}", label(s)),
        None => format!("**none passes**.{multigrid_note}"),
    };

    // The record.
    let mut sweep = String::new();
    for (&(method, count, stage), row) in &gate.rows {
        let _ = writeln!(
            sweep,
            "| {} | {count} | {}³ | {} | {} | {} | {} | {} |",
            method.name(),
            plan.res[stage],
            fmt_div(&row.scenes[0]),
            fmt_div(&row.scenes[1]),
            fmt_div(&row.scenes[2]),
            if stage == 0 {
                fmt_div(&row.plate)
            } else {
                "n/a".to_owned()
            },
            if row.passes(stage) { "yes" } else { "no" },
        );
    }
    let mut timing = String::new();
    for (stage, &res) in plan.res.iter().enumerate() {
        for (si, name) in SCENES.iter().enumerate() {
            let cells: Vec<String> = (0..configs.len())
                .map(|ci| {
                    let t = &times[&(ci, stage, si)];
                    let lo = t.iter().copied().fold(f64::INFINITY, f64::min);
                    let hi = t.iter().copied().fold(0.0, f64::max);
                    let mark = if ci == 0 {
                        String::new()
                    } else if frame_ms(ci, stage, si) <= frame_ms(0, stage, si) {
                        " ✓".to_owned()
                    } else {
                        " ✗".to_owned()
                    };
                    format!("{:.1} ({lo:.1}–{hi:.1}){mark}", frame_ms(ci, stage, si))
                })
                .collect();
            let _ = writeln!(timing, "| {res}³ | {name} | {} |", cells.join(" | "));
        }
    }
    let timing_head = configs
        .iter()
        .map(|&s| format!("`{}` ms", label(s)))
        .collect::<Vec<_>>()
        .join(" | ");
    let timing_rule = "|---".repeat(configs.len() + 2);
    let mut candidate_lines = String::new();
    for &(method, count) in &candidates {
        let _ = writeln!(
            candidate_lines,
            "- `{}`: {}",
            method.name(),
            match count {
                Some(c) => format!("count {c}"),
                None => format!(
                    "no count up to {} passes; the method fails",
                    plan.cap(method)
                ),
            }
        );
    }
    let mut stability_lines = String::new();
    for (method, result) in &stability {
        let _ = writeln!(
            stability_lines,
            "- `{}`: ×{STABILITY_COUNT} over the candidate count, frames 60 / 120: {}",
            method.name(),
            match result {
                Ok([a, b]) => format!(
                    "{a:.3} / {b:.3} — {}",
                    if *a <= STABILITY_FACTOR && *b <= STABILITY_FACTOR {
                        "stable"
                    } else {
                        "UNSTABLE"
                    }
                ),
                Err(e) => format!("error: {e}"),
            }
        );
    }
    if stability_lines.is_empty() {
        stability_lines.push_str("- no candidate to check\n");
    }
    let mut verdict_lines = String::new();
    for (method, solve, fast, stable) in &verdicts {
        let _ = writeln!(
            verdict_lines,
            "- `{}`: divergence and thin plate pass; frame time {} Gauss–Seidel's in every \
             scene at both resolutions; past-floor {}{}",
            label(*solve),
            if *fast { "at most" } else { "NOT at most" },
            if *stable { "stable" } else { "NOT stable" },
            if *method == Method::Multigrid {
                " (cannot be the default)"
            } else {
                ""
            },
        );
    }
    let flag = |text: &str| {
        if load_1m(text) > LOAD_FLAG {
            format!("{text} — **above {LOAD_FLAG}: the machine was loaded**")
        } else {
            text.to_owned()
        }
    };
    let title = if plan.smoke {
        "Ember solver gate — SMOKE PASS (plumbing only, not the gate)"
    } else {
        "Ember solver gate (2b-3c)"
    };
    let refs = if plan.smoke {
        format!(
            "Smoke pass: the {low}³ and {high}³ columns are compared against Mantaflow's 128³ \
             and 256³ references, which proves only the plumbing.\n\n"
        )
    } else {
        String::new()
    };
    let report = format!(
        "# {title}\n\n\
         - Machine: {cpu} ({adapter})\n\
         - OS: macOS {os}\n\
         - Ember commit: {commit}\n\
         - Date: {date}\n\
         - Load average (1, 5, 15 min) before: {lb}\n\
         - Load average after: {la}\n\
         - Kernel path check: `plume_plate` {low}³ with `{check_label}` through frame {check_frame} \
         (frame {m0} instrumented), against `eval_frame`: {check_n} faces differ, max |Δ| {check_max:e}.\n\n\
         ## The rule (spec §4, fixed before any numbers)\n\n\
         A configuration passes when, at {low}³ and {high}³ and in `plume`, `plume_collider` and \
         `plume_wind`, its median frame time over frames {TIMED_FROM}–{TIMED_FRAMES} is at most the \
         Gauss–Seidel ×{GS_ITERATIONS} median in the same process, and its masked divergence RMS at \
         frames 60 and 120 is at most Mantaflow's for that scene and resolution \
         (`docs/bench/results.md`, the 7fe5d9d run; `plume_wind` uses `plume`'s until the rerun). \
         It must also pass `plume_plate` at {low}³: RMS divergence after projection over before it, \
         at frames 60 and 120, through the kernel API, at most {PLATE_RATIO:e}. The chosen \
         configuration must be stable past the floor: in `plume_collider` at {low}³ the masked \
         divergence after {STABILITY_COUNT} iterations is at most {STABILITY_FACTOR}× that after \
         the chosen count. Plain V-cycles (`multigrid`) are measured but cannot be the default. \
         The fastest passing configuration becomes the default; if none passes, work stops.\n\n\
         Mantaflow references (masked RMS, f60 / f120): `plume` and `plume_wind` \
         1.49e-4 / 1.07e-4 at 128³, 7.41e-5 / 4.51e-5 at 256³; `plume_collider` 2.84e-4 / 1.09e-4 \
         at 128³, 9.12e-5 / 5.25e-5 at 256³.\n\n\
         {refs}\
         ## Procedure\n\n\
         Staged sweep: at {low}³, counts rise from 1 (`multigrid` to {mg_cap}, `mgpcg` to \
         {pcg_cap}) until the three scenes and `plume_plate` all pass; that count is checked at \
         {high}³ and stepped up until it passes there too (and still at {low}³). A {high}³ count \
         stops at its first failing scene (— below). Divergence runs read back only at frames 60 \
         and 120. Then each candidate and Gauss–Seidel ×{GS_ITERATIONS} are timed: {runs} runs of \
         frames 1–{TIMED_FRAMES}, each run's median over frames {TIMED_FROM}–{TIMED_FRAMES}, \
         interleaved (Gauss–Seidel, multigrid, mgpcg, Gauss–Seidel, …) so drifting load hits all \
         of them. Timing is per frame (`eval_frame` plus a blocking wait), not per solve: runs of \
         one scene differ only in the solver, so it is the same comparison without instrumenting \
         inside the solver.\n\n\
         ## Divergence sweep\n\n\
         Masked divergence RMS, 1/s, frames 60 / 120. `plume_plate`: after/before ratio, frames 60 / 120.\n\n\
         | method | count | res | plume | plume_collider | plume_wind | plume_plate ratio | pass |\n\
         |---|---|---|---|---|---|---|---|\n\
         {sweep}\n\
         {empty_note}\
         Candidates:\n\n{candidate_lines}\n\
         ## Timing\n\n\
         Median frame ms over the {runs} runs' medians (min–max of the run medians). ✓: at most \
         Gauss–Seidel's.\n\n\
         | res | scene | {timing_head} |\n\
         {timing_rule}|\n\
         {timing}\n\
         ## Past-floor stability (`plume_collider`, {low}³)\n\n{stability_lines}\n\
         ## Rule applied\n\n{verdict_lines}\n\
         Rule applied: {verdict}\n\n\
         Decision (recorded by the user): _pending_\n",
        cpu = shell("sysctl", &["-n", "machdep.cpu.brand_string"]),
        adapter = gpu.adapter_name(),
        os = shell("sw_vers", &["-productVersion"]),
        lb = flag(&load_before),
        la = flag(&load_after),
        check_label = label(check_solve),
        check_frame = MEASURED[0] + 5,
        m0 = MEASURED[0],
        check_n = check.1,
        check_max = check.0,
        mg_cap = plan.multigrid_cap,
        pcg_cap = plan.mgpcg_cap,
        runs = plan.runs,
        empty_note = if gate.empty.is_empty() {
            String::new()
        } else {
            format!(
                "Measurements with no measured cells (the smoke mask was empty, so the RMS is 0 \
                 and passes vacuously): {}.\n\n",
                gate.empty.join("; ")
            )
        },
    );
    if let Some(dir) = plan.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&plan.out, &report)?;
    gate.clock.log(format!("wrote {}", plan.out.display()));
    println!("{report}");
    Ok(())
}
