mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext, GpuError, PipelineCache};
use elements_core::graph::{
    DocError, Document, EvalCtx, Graph, Node, NodeError, NodeId, SocketId, SocketSpec, SocketType,
    Timeline, TimelineConfig, Value,
};
use elements_ember::kernels::StepConstants;
use elements_ember::solver::{KIND, SolverState, Sources, substep};

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
    assert!(rejected(serde_json::json!({ "pressure_iterations": 0 })));
    assert!(rejected(serde_json::json!({ "pressure_iterations": 1001 })));
    assert!(rejected(
        serde_json::json!({ "buoyancy_temperature": 1e39 })
    ));
    assert!(rejected(serde_json::json!({ "buoyancy_density": 1e39 })));
    assert!(rejected(serde_json::json!({ "vorticity": 1.0 })));
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
        cells,
        h: 1.0 / 24.0,
        dx: 0.25,
        alpha: 0.5,
        beta: 2.0,
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

const PLUME_16: &str = r#"{
  "version": 3,
  "dims": [16, 16, 16],
  "fps": 24.0,
  "domain_size": 2.0,
  "nodes": [
    { "id": 0, "kind": "ember.sphere_emitter",
      "params": { "center": [1.0, 1.0, 0.4], "radius": 0.3,
                  "density_rate": 1.0, "temperature_rate": 2.0 } },
    { "id": 1, "kind": "ember.smoke_solver",
      "params": { "pressure_iterations": 40, "buoyancy_temperature": 1.0 } },
    { "id": 2, "kind": "core.output", "params": {} }
  ],
  "edges": [
    { "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 },
    { "from_node": 0, "from_index": 1, "to_node": 1, "to_index": 1 },
    { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 0 }
  ],
  "output": 2
}"#;

struct Session {
    gpu: GpuContext,
    pool: FieldPool,
    pipelines: PipelineCache,
    graph: Graph,
    dims: FieldDims,
}

impl Session {
    fn new() -> Self {
        let (graph, dims) = Document::from_json(PLUME_16)
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
#[test]
fn frame_40_is_bit_identical_however_it_is_reached() {
    let mut s = Session::new();

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
    // acquires `div` at the domain's dims and only then discovers `phi`
    // (state.pressure) is the wrong size, in `kernels::pressure`'s dims check.
    let wrong_dims = FieldDims::new(cells.x + 1, cells.y, cells.z);
    let wrong_pressure = pool.acquire_zeroed(&gpu, &mut cache, wrong_dims).unwrap();
    let old_pressure = std::mem::replace(&mut state.pressure, wrong_pressure);
    pool.release(old_pressure);

    let zero = pool.acquire_zeroed(&gpu, &mut cache, cells).unwrap();
    let constants = StepConstants {
        cells,
        h: 1.0 / 24.0,
        dx: 0.25,
        alpha: 0.5,
        beta: 2.0,
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
