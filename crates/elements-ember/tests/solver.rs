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
