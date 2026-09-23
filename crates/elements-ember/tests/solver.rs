mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext, GpuError, PipelineCache};
use elements_core::graph::{
    DocError, Document, EvalCtx, Graph, Node, NodeError, NodeId, SocketId, SocketSpec, SocketType,
    StateStore, Time, Timeline, TimelineConfig, Value,
};
use elements_ember::cfl;
use elements_ember::kernels::Advection;
use elements_ember::kernels::StepConstants;
use elements_ember::solver::{KIND, SolverParams, SolverState, Sources, resolve_params, substep};
use std::sync::{Arc, Mutex};

fn rejected(params: serde_json::Value) -> bool {
    matches!(
        elements_ember::registry().build(KIND, &params),
        Err(DocError::BadParams { .. })
    )
}

#[test]
fn rejects_out_of_range_solver_parameters() {
    assert!(!rejected(serde_json::json!({})), "defaults must build");
    assert!(!rejected(
        serde_json::json!({ "substeps": 16, "pressure_iterations": 1000 })
    ));
    assert!(rejected(serde_json::json!({ "substeps": 0 })));
    assert!(rejected(serde_json::json!({ "substeps": 17 })));
    assert!(!rejected(serde_json::json!({ "quality": "final" })));
    assert!(rejected(serde_json::json!({ "quality": "ultra" })));
    assert!(!rejected(
        serde_json::json!({ "max_substeps": 16, "cfl": 10.0 })
    ));
    assert!(rejected(serde_json::json!({ "max_substeps": 17 })));
    assert!(rejected(
        serde_json::json!({ "substeps": 2, "max_substeps": 2 })
    ));
    assert!(rejected(serde_json::json!({ "cfl": 0.0 })));
    assert!(rejected(serde_json::json!({ "cfl": 10.5 })));
    assert!(rejected(serde_json::json!({ "pressure_iterations": 0 })));
    assert!(rejected(serde_json::json!({ "pressure_iterations": 1001 })));
    assert!(rejected(
        serde_json::json!({ "buoyancy_temperature": 1e39 })
    ));
    assert!(rejected(serde_json::json!({ "buoyancy_density": 1e39 })));
    assert!(!rejected(serde_json::json!({ "vorticity": 1.0 })));
    assert!(rejected(serde_json::json!({ "vorticity": -1.0 })));
    assert!(rejected(serde_json::json!({ "vorticity": 1e39 })));
    assert!(!rejected(
        serde_json::json!({ "boundaries": { "-x": "open", "+z": "wall" } })
    ));
    assert!(rejected(
        serde_json::json!({ "boundaries": { "+w": "open" } })
    ));
    assert!(rejected(
        serde_json::json!({ "boundaries": { "-x": "porous" } })
    ));
    assert!(!rejected(
        serde_json::json!({ "advection": "semi_lagrangian" })
    ));
    assert!(!rejected(
        serde_json::json!({ "advection": "maccormack", "density_dissipation": 0.5 })
    ));
    assert!(rejected(serde_json::json!({ "advection": "bfecc" })));
    assert!(rejected(serde_json::json!({ "density_dissipation": -1.0 })));
    assert!(rejected(
        serde_json::json!({ "temperature_dissipation": 1e39 })
    ));
}

/// Spec §6: a preset fills only the fields a document leaves unset, and
/// 2a's `substeps` still works as `max_substeps`.
#[test]
fn a_preset_fills_only_what_the_document_leaves_unset() {
    let p = resolve_params(&serde_json::json!({ "quality": "final", "pressure_iterations": 50 }))
        .unwrap();
    assert_eq!(p.pressure_iterations, 50, "an explicit field wins");
    assert_eq!(p.max_substeps, 8, "final's cap");
    let alias = resolve_params(&serde_json::json!({ "substeps": 3 })).unwrap();
    assert_eq!(alias.max_substeps, 3, "the 2a alias");
    assert_eq!(
        resolve_params(&serde_json::Value::Null).unwrap(),
        SolverParams::default()
    );
}

/// Every per-substep parameter a document sets reaches the kernels.
#[test]
fn step_constants_carry_every_solver_parameter() {
    let p = resolve_params(&serde_json::json!({
        "advection": "semi_lagrangian", "vorticity": 3.0,
        "density_dissipation": 0.5, "temperature_dissipation": 0.25,
        "buoyancy_density": 0.75, "buoyancy_temperature": 2.0,
        "boundaries": { "-x": "open" }
    }))
    .unwrap();
    let c = p.step_constants(FieldDims::new(8, 6, 5), 0.1, 0.125);
    assert_eq!(c.advection, Advection::SemiLagrangian);
    assert_eq!(c.vorticity, 3.0);
    assert_eq!(c.density_dissipation, 0.5);
    assert_eq!(c.temperature_dissipation, 0.25);
    assert_eq!(c.alpha, 0.75);
    assert_eq!(c.beta, 2.0);
    assert_eq!(c.open_mask, 0b100001);
}

/// Umbrella §6: no emitters and nothing to be buoyant, so nothing moves.
/// The buoyancy coefficients are nonzero on purpose: the force is zero only
/// because density and temperature are.
#[test]
fn a_still_domain_stays_exactly_still() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(8, 6, 5);
    let zero = pool.acquire_zeroed(&gpu, &mut cache, cells).unwrap();
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
    let constants = StepConstants {
        alpha: 0.5,
        beta: 2.0,
        ..StepConstants::new(cells, 1.0 / 24.0, 0.25)
    };
    let sources = Sources {
        density: &zero,
        temperature: &zero,
    };
    for _ in 0..10 {
        substep(
            &gpu, &mut cache, &mut pool, &mut state, sources, &constants, 20,
        )
        .unwrap();
    }
    for (a, face) in state.read_velocity(&gpu).unwrap().iter().enumerate() {
        assert!(face.iter().all(|&v| v == 0.0), "face {a} moved");
    }
}

/// A 16³ plume whose output is density. `solver` is the solver's params
/// object.
fn plume_16(solver: &str) -> String {
    format!(
        r#"{{
  "version": 3,
  "dims": [16, 16, 16],
  "fps": 24.0,
  "domain_size": 2.0,
  "nodes": [
    {{ "id": 0, "kind": "ember.sphere_emitter",
      "params": {{ "center": [1.0, 1.0, 0.4], "radius": 0.3,
                  "density_rate": 1.0, "temperature_rate": 2.0 }} }},
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

/// The preview defaults: one substep, no confinement, no dissipation, open top.
const PREVIEW: &str = r#"{ "pressure_iterations": 40, "buoyancy_temperature": 1.0 }"#;

struct Session {
    gpu: GpuContext,
    pool: FieldPool,
    pipelines: PipelineCache,
    graph: Graph,
    dims: FieldDims,
}

impl Session {
    fn new(doc: &str) -> Self {
        let (graph, dims) = Document::from_json(doc)
            .unwrap()
            .into_graph(&elements_ember::registry())
            .unwrap();
        Self {
            gpu: gpu(),
            pool: FieldPool::new(),
            pipelines: PipelineCache::new(),
            graph,
            dims,
        }
    }

    fn density_bits(&mut self, timeline: &mut Timeline, frame: u32) -> Vec<u32> {
        let evaluated = timeline
            .goto(
                &self.graph,
                &self.gpu,
                &mut self.pool,
                &mut self.pipelines,
                self.dims,
                frame,
            )
            .unwrap();
        let bits = evaluated
            .value
            .as_field()
            .unwrap()
            .read_back(&self.gpu)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        evaluated.value.release_to(&mut self.pool);
        bits
    }
}

fn timeline(budget_bytes: u64) -> Timeline {
    Timeline::new(TimelineConfig {
        fps: 24.0,
        start_frame: 1,
        cache_budget_bytes: budget_bytes,
    })
}

/// Umbrella §4 and §6: frame 40 is bit-identical in order, after scrubbing
/// back and forth, and after eviction forced a recompute.
fn assert_frame_40_is_bit_identical(solver: &str) {
    let mut s = Session::new(&plume_16(solver));

    let mut in_order = timeline(0);
    let mut reference = Vec::new();
    for frame in 1..=40 {
        reference = s.density_bits(&mut in_order, frame);
    }
    assert!(
        reference.iter().any(|&b| f32::from_bits(b) != 0.0),
        "the plume must exist"
    );

    let mut scrubbed = timeline(512 * 1024 * 1024);
    s.density_bits(&mut scrubbed, 40);
    s.density_bits(&mut scrubbed, 10);
    assert!(
        s.density_bits(&mut scrubbed, 40) == reference,
        "after scrubbing"
    );

    // About ten 16³ snapshots fit, so reaching 40 evicts most of them.
    let mut evicting = timeline(1024 * 1024);
    s.density_bits(&mut evicting, 40);
    s.density_bits(&mut evicting, 5);
    assert!(
        s.density_bits(&mut evicting, 40) == reference,
        "after eviction"
    );
}

#[test]
fn frame_40_is_bit_identical_however_it_is_reached() {
    assert_frame_40_is_bit_identical(PREVIEW);
}

/// Everything 2b-1 added, switched on: a CFL count that varies from frame to
/// frame, confinement, and dissipation. `cfl` is small enough that the count
/// changes within the first 40 frames (`substep_counts` shows it does).
const STRESSED: &str = r#"{ "pressure_iterations": 40, "buoyancy_temperature": 1.0,
    "max_substeps": 8, "cfl": 0.1, "vorticity": 2.0,
    "density_dissipation": 0.2, "temperature_dissipation": 0.5 }"#;

/// `STRESSED` in a box with every face a wall, so the closed-domain pressure
/// mean removal runs too.
const STRESSED_CLOSED: &str = r#"{ "pressure_iterations": 40, "buoyancy_temperature": 1.0,
    "max_substeps": 8, "cfl": 0.1, "vorticity": 2.0,
    "density_dissipation": 0.2, "temperature_dissipation": 0.5,
    "boundaries": { "-x": "wall", "+x": "wall", "-y": "wall",
                    "+y": "wall", "-z": "wall", "+z": "wall" } }"#;

/// Passes density through, and records the fastest face speed of the
/// velocity it is given: a copy of the state entering the next frame.
struct SpeedProbe(Arc<Mutex<Vec<f32>>>);

impl Node for SpeedProbe {
    fn kind(&self) -> &'static str {
        "test.speed_probe"
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field, SocketType::VectorField],
            outputs: vec![SocketType::Field],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let velocity = ctx.take_input(1)?;
        let speed = ctx.with_gpu(|gpu, cache| {
            cfl::measure_speed(gpu, cache, velocity.as_vector_field().unwrap())
        });
        ctx.release(velocity);
        self.0.lock().unwrap().push(speed?);
        Ok(vec![ctx.take_input(0)?])
    }
}

/// The substep count each of frames 1..=40 runs with, reconstructed from
/// outside: the solver plans from the state entering a frame (spec §3),
/// which is what `SpeedProbe` saw at the end of the frame before.
fn substep_counts(solver: &str) -> Vec<u32> {
    let doc: serde_json::Value = serde_json::from_str(&plume_16(solver)).unwrap();
    let params = resolve_params(&doc["nodes"][1]["params"]).unwrap();
    let registry = elements_ember::registry();
    let build = |i: usize| {
        let node = &doc["nodes"][i];
        registry
            .build(node["kind"].as_str().unwrap(), &node["params"])
            .unwrap()
    };
    let mut graph = Graph::new();
    let emitter = graph.add_node(build(0));
    let solver = graph.add_node(build(1));
    let speeds = Arc::new(Mutex::new(Vec::new()));
    let probe = graph.add_node(Box::new(SpeedProbe(Arc::clone(&speeds))));
    let output = graph.add_node(build(2));
    let socket = |node: NodeId, index: u32| SocketId { node, index };
    for (from, to) in [
        (socket(emitter, 0), socket(solver, 0)),
        (socket(emitter, 1), socket(solver, 1)),
        (socket(solver, 0), socket(probe, 0)),
        (socket(solver, 2), socket(probe, 1)),
        (socket(probe, 0), socket(output, 0)),
    ] {
        graph.connect(from, to).unwrap();
    }
    graph.set_output(output);
    graph.set_domain_size(2.0);

    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut in_order = timeline(0);
    for frame in 1..=39 {
        let evaluated = in_order
            .goto(
                &graph,
                &gpu,
                &mut pool,
                &mut pipelines,
                FieldDims::new(16, 16, 16),
                frame,
            )
            .unwrap();
        evaluated.value.release_to(&mut pool);
    }
    // Frame 1 starts still; frame N + 1 plans from the speed after frame N.
    let after = speeds.lock().unwrap().clone();
    std::iter::once(0.0)
        .chain(after)
        .map(|speed| {
            cfl::plan_substeps(
                speed,
                1.0 / 24.0,
                2.0 / 16.0,
                params.cfl,
                params.max_substeps,
            )
            .expect("a finite speed")
            .count
        })
        .collect()
}

fn assert_counts_vary(solver: &str) {
    let counts = substep_counts(solver);
    eprintln!("substep counts, frames 1..=40: {counts:?}");
    assert!(
        counts.iter().any(|&c| c != counts[0]),
        "the CFL count must change within 40 frames: {counts:?}"
    );
}

#[test]
fn frame_40_is_bit_identical_with_varying_substeps_confinement_and_dissipation() {
    assert_counts_vary(STRESSED);
    assert_frame_40_is_bit_identical(STRESSED);
}

#[test]
fn frame_40_is_bit_identical_in_a_closed_domain() {
    assert_counts_vary(STRESSED_CLOSED);
    assert_frame_40_is_bit_identical(STRESSED_CLOSED);
}

/// Outputs a zero field one cell larger than the domain: a mis-sized source.
struct WrongSize;

impl Node for WrongSize {
    fn kind(&self) -> &'static str {
        "test.wrong_size"
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Field],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let d = ctx.dims();
        let dims = FieldDims::new(d.x + 1, d.y, d.z);
        let field =
            ctx.with_gpu_pool(|gpu, _, pool| pool.acquire(gpu, dims, FieldFormat::R32Float))?;
        Ok(vec![Value::Field(field)])
    }
}

/// A step that fails part-way returns every texture to the pool: the state
/// it took out of the store and anything it acquired.
#[test]
fn a_failed_step_returns_every_field_to_the_pool() {
    let registry = elements_ember::registry();
    let mut graph = Graph::new();
    let emitter = graph.add_node(
        registry
            .build(
                "ember.sphere_emitter",
                &serde_json::json!({ "center": [1.0, 1.0, 0.4], "radius": 0.3 }),
            )
            .unwrap(),
    );
    let wrong = graph.add_node(Box::new(WrongSize));
    let solver = graph.add_node(registry.build(KIND, &serde_json::json!({})).unwrap());
    let output = graph.add_node(
        registry
            .build("core.output", &serde_json::json!({}))
            .unwrap(),
    );
    let socket = |node: NodeId, index: u32| SocketId { node, index };
    graph
        .connect(socket(emitter, 0), socket(solver, 0))
        .unwrap();
    graph.connect(socket(wrong, 0), socket(solver, 1)).unwrap();
    graph.connect(socket(solver, 0), socket(output, 0)).unwrap();
    graph.set_output(output);

    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut timeline = timeline(0);
    let err = timeline
        .goto(
            &graph,
            &gpu,
            &mut pool,
            &mut pipelines,
            FieldDims::new(8, 8, 8),
            1,
        )
        .unwrap_err();
    assert!(
        matches!(err, NodeError::Gpu(GpuError::Validation(_))),
        "got {err:?}"
    );
    assert!(pool.allocation_count() > 0);
    assert_eq!(
        pool.pooled_count() as u64,
        pool.allocation_count(),
        "every allocated texture must be back in the pool"
    );
}

/// A substep that fails after `pre_projection` has already retired fields
/// (the three old velocity faces) and `project` has acquired `div` must
/// still return every one of them to the pool. Unlike
/// `a_failed_step_returns_every_field_to_the_pool`, which fails at the very
/// first `emit` before `Substep::retired` holds anything, this exercises the
/// release loop in `Substep::abandon` itself.
#[test]
fn a_substep_failing_after_retiring_fields_returns_them_all() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(8, 6, 5);
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();

    // Replace `pressure` with a field one cell larger along x. `project`
    // acquires `div` at the domain's dims and only then discovers `p`
    // (state.pressure) is the wrong size, in `kernels::pressure`'s dims check.
    let wrong_dims = FieldDims::new(cells.x + 1, cells.y, cells.z);
    let wrong_pressure = pool.acquire_zeroed(&gpu, &mut cache, wrong_dims).unwrap();
    let old_pressure = std::mem::replace(&mut state.pressure, wrong_pressure);
    pool.release(old_pressure);

    let zero = pool.acquire_zeroed(&gpu, &mut cache, cells).unwrap();
    let constants = StepConstants {
        alpha: 0.5,
        beta: 2.0,
        ..StepConstants::new(cells, 1.0 / 24.0, 0.25)
    };
    let sources = Sources {
        density: &zero,
        temperature: &zero,
    };

    let err = substep(
        &gpu, &mut cache, &mut pool, &mut state, sources, &constants, 20,
    )
    .unwrap_err();
    assert!(matches!(err, GpuError::Validation(_)), "got {err:?}");

    state.release_to(&mut pool);
    pool.release(zero);
    assert_eq!(
        pool.pooled_count() as u64,
        pool.allocation_count(),
        "every allocated texture must be back in the pool"
    );
}

/// Which field one probe graph reads out.
#[derive(Clone, Copy)]
enum Probe {
    /// An emitter output, straight into `core.output`.
    Source(u32),
    /// A solver output after one frame, fed by the emitter.
    Solver(u32),
}

/// Total of the probed field over the domain, after frame 1.
fn total(probe: Probe) -> f64 {
    let registry = elements_ember::registry();
    let mut graph = Graph::new();
    let emitter = graph.add_node(
        registry
            .build(
                "ember.sphere_emitter",
                &serde_json::json!({ "center": [1.0, 1.0, 1.0], "radius": 0.5,
                                     "density_rate": 2.0, "temperature_rate": 5.0 }),
            )
            .unwrap(),
    );
    let output = graph.add_node(
        registry
            .build("core.output", &serde_json::json!({}))
            .unwrap(),
    );
    let socket = |node: NodeId, index: u32| SocketId { node, index };
    match probe {
        Probe::Source(index) => graph
            .connect(socket(emitter, index), socket(output, 0))
            .unwrap(),
        Probe::Solver(index) => {
            // No buoyancy, so nothing moves and advection returns its input exactly.
            let solver = graph.add_node(
                registry
                    .build(
                        KIND,
                        &serde_json::json!({ "buoyancy_density": 0.0,
                                                      "buoyancy_temperature": 0.0 }),
                    )
                    .unwrap(),
            );
            graph
                .connect(socket(emitter, 0), socket(solver, 0))
                .unwrap();
            graph
                .connect(socket(emitter, 1), socket(solver, 1))
                .unwrap();
            graph
                .connect(socket(solver, index), socket(output, 0))
                .unwrap();
        }
    }
    graph.set_output(output);

    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut timeline = timeline(0);
    let frame = timeline
        .goto(
            &graph,
            &gpu,
            &mut pool,
            &mut pipelines,
            FieldDims::new(16, 16, 16),
            1,
        )
        .unwrap();
    let sum = frame
        .value
        .as_field()
        .unwrap()
        .read_back(&gpu)
        .unwrap()
        .iter()
        .map(|&v| v as f64)
        .sum();
    frame.value.release_to(&mut pool);
    sum
}

/// One frame adds exactly rate × dt of each quantity, into the right output.
/// Density and temperature are emitted at different rates, so swapping them
/// at the solver's inputs or outputs changes both totals by a factor of 2.5.
#[test]
fn one_frame_adds_each_emitted_quantity_to_its_own_output() {
    let dt = 1.0 / 24.0;
    for (index, what) in [(0, "density"), (1, "temperature")] {
        let want = total(Probe::Source(index)) * dt;
        let got = total(Probe::Solver(index));
        assert!(want > 0.0, "{what}: the emitter must emit");
        assert!(
            ((got - want) / want).abs() <= 1e-5,
            "{what}: solver holds {got}, emitted {want}"
        );
    }
}

/// Reads all three solver outputs and passes density on.
struct Sink;

impl Node for Sink {
    fn kind(&self) -> &'static str {
        "test.sink"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![
                SocketType::Field,
                SocketType::Field,
                SocketType::VectorField,
            ],
            outputs: vec![SocketType::Field],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        Ok(vec![ctx.take_input(0)?])
    }
}

/// Pool acquisitions over one frame, with either every solver output read
/// or only density.
fn acquisitions_for_one_frame(read_all: bool) -> u64 {
    let registry = elements_ember::registry();
    let mut graph = Graph::new();
    let emitter = graph.add_node(
        registry
            .build(
                "ember.sphere_emitter",
                &serde_json::json!({ "center": [1.0, 1.0, 0.4], "radius": 0.3 }),
            )
            .unwrap(),
    );
    let solver = graph.add_node(
        registry
            .build(KIND, &serde_json::json!({ "pressure_iterations": 4 }))
            .unwrap(),
    );
    let output = graph.add_node(
        registry
            .build("core.output", &serde_json::json!({}))
            .unwrap(),
    );
    let socket = |node: NodeId, index: u32| SocketId { node, index };
    graph
        .connect(socket(emitter, 0), socket(solver, 0))
        .unwrap();
    graph
        .connect(socket(emitter, 1), socket(solver, 1))
        .unwrap();
    if read_all {
        let sink = graph.add_node(Box::new(Sink));
        for index in 0..3 {
            graph
                .connect(socket(solver, index), socket(sink, index))
                .unwrap();
        }
        graph.connect(socket(sink, 0), socket(output, 0)).unwrap();
    } else {
        graph.connect(socket(solver, 0), socket(output, 0)).unwrap();
    }
    graph.set_output(output);

    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let evaluated = graph
        .eval_frame(
            &gpu,
            &mut pool,
            &mut pipelines,
            &mut state,
            Time::at(1, 1, 24.0),
            FieldDims::new(8, 6, 5),
        )
        .unwrap();
    evaluated.value.release_to(&mut pool);
    state.clear(&mut pool);
    pool.acquisitions()
}

/// Risk (h): outputs nobody reads are never copied. Temperature is one
/// field and velocity three faces, so reading only density saves four.
#[test]
fn outputs_nobody_reads_are_never_copied() {
    let all = acquisitions_for_one_frame(true);
    let density_only = acquisitions_for_one_frame(false);
    assert_eq!(
        all - density_only,
        4,
        "all {all}, density only {density_only}"
    );
}
