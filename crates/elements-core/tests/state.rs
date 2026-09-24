use elements_core::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{
    Document, EvalCtx, Graph, Node, NodeError, NodeId, NodeRegistry, SocketSpec, SocketType,
    StateStore, Time, Value,
};

/// constant 0.5 -> accumulate -> output, on a 4³ domain.
const ACCUMULATE_DOC: &str = r#"{
  "version": 1,
  "dims": [4, 4, 4],
  "nodes": [
    { "id": 0, "kind": "core.constant_field", "params": { "value": 0.5 } },
    { "id": 1, "kind": "core.accumulate", "params": {} },
    { "id": 2, "kind": "core.output", "params": {} }
  ],
  "edges": [
    { "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 },
    { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 0 }
  ],
  "output": 2
}"#;

struct Harness {
    gpu: GpuContext,
    pool: FieldPool,
    pipelines: PipelineCache,
}

impl Harness {
    fn new() -> Self {
        Self {
            gpu: GpuContext::new_headless().expect("no GPU adapter available"),
            pool: FieldPool::new(),
            pipelines: PipelineCache::new(),
        }
    }

    /// Evaluate `frame` against `state` and read the output back. The output
    /// field is released to the pool, so repeated frames do not reallocate.
    fn frame(
        &mut self,
        graph: &Graph,
        state: &mut StateStore,
        frame: u32,
        dims: FieldDims,
    ) -> Vec<f32> {
        let evaluated = graph
            .eval_frame(
                &self.gpu,
                &mut self.pool,
                &mut self.pipelines,
                state,
                Time::at(frame, 1, 24.0),
                dims,
            )
            .unwrap();
        let values = evaluated
            .value
            .as_field()
            .unwrap()
            .read_back(&self.gpu)
            .unwrap();
        evaluated.value.release_to(&mut self.pool);
        values
    }
}

fn accumulate_graph() -> (Graph, FieldDims) {
    Document::from_json(ACCUMULATE_DOC)
        .unwrap()
        .into_graph(&NodeRegistry::with_builtins())
        .unwrap()
}

/// The closed form for ACCUMULATE_DOC at `frame`, with start frame 1 at 24 fps.
fn expected(frame: u32) -> f32 {
    0.5 * frame as f32 / 24.0
}

fn assert_all_near(values: &[f32], expected: f32) {
    for &v in values {
        assert!((v - expected).abs() < 1e-6, "expected {expected}, got {v}");
    }
}

#[test]
fn time_reports_seconds_since_the_start_frame() {
    let t = Time::at(25, 1, 24.0);
    assert_eq!(t.frame, 25);
    assert_eq!(t.seconds, 1.0);
    assert_eq!(t.dt, 1.0 / 24.0);
    assert_eq!(Time::at(0, 1, 24.0).seconds, 0.0);
}

#[test]
fn accumulate_integrates_its_input_over_frames() {
    let mut h = Harness::new();
    let (graph, dims) = accumulate_graph();
    let mut state = StateStore::new();
    for frame in 1..=5 {
        let values = h.frame(&graph, &mut state, frame, dims);
        assert_all_near(&values, expected(frame));
    }
    assert_eq!(state.len(), 1, "one slot: accumulate's sum");
}

#[test]
fn plain_eval_of_a_stateful_graph_is_always_its_first_frame() {
    let mut h = Harness::new();
    let (graph, dims) = accumulate_graph();
    assert!(graph.is_stateful());
    for _ in 0..3 {
        let value = graph
            .eval(&h.gpu, &mut h.pool, &mut h.pipelines, dims)
            .unwrap();
        assert_all_near(
            &value.as_field().unwrap().read_back(&h.gpu).unwrap(),
            expected(1),
        );
        value.release_to(&mut h.pool);
    }
}

/// A node that does not declare itself stateful, but reaches for state anyway.
struct Sneaky;

impl Node for Sneaky {
    fn kind(&self) -> &'static str {
        "test.sneaky"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Scalar],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        ctx.take_state("anything")?;
        Ok(vec![Value::Scalar(0.0)])
    }
}

#[test]
fn a_stateless_node_cannot_touch_state() {
    let mut h = Harness::new();
    let mut graph = Graph::new();
    let id = graph.add_node(Box::new(Sneaky));
    graph.set_output(id);
    let mut state = StateStore::new();
    let err = graph
        .eval_frame(
            &h.gpu,
            &mut h.pool,
            &mut h.pipelines,
            &mut state,
            Time::at(1, 1, 24.0),
            FieldDims::new(4, 4, 4),
        )
        .unwrap_err();
    assert!(matches!(err, NodeError::NotStateful { .. }), "got {err:?}");
}

#[test]
fn state_of_the_wrong_shape_is_reported() {
    let mut h = Harness::new();
    let (graph, dims) = accumulate_graph();
    let mut state = StateStore::new();
    h.frame(&graph, &mut state, 1, dims);

    let err = graph
        .eval_frame(
            &h.gpu,
            &mut h.pool,
            &mut h.pipelines,
            &mut state,
            Time::at(2, 1, 24.0),
            FieldDims::new(8, 8, 8),
        )
        .unwrap_err();
    assert!(
        matches!(err, NodeError::StateShape { slot: "sum", .. }),
        "got {err:?}"
    );
}

/// A stateless node that outputs the time it was evaluated at.
struct Clock;

impl Node for Clock {
    fn kind(&self) -> &'static str {
        "test.clock"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Scalar],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        Ok(vec![Value::Scalar(ctx.time().seconds as f32)])
    }
}

#[test]
fn time_reaches_nodes() {
    let mut h = Harness::new();
    let mut graph = Graph::new();
    let id = graph.add_node(Box::new(Clock));
    graph.set_output(id);
    let mut state = StateStore::new();
    let evaluated = graph
        .eval_frame(
            &h.gpu,
            &mut h.pool,
            &mut h.pipelines,
            &mut state,
            Time::at(49, 1, 24.0),
            FieldDims::new(1, 1, 1),
        )
        .unwrap();
    assert_eq!(evaluated.value.as_scalar().unwrap(), 2.0);
}

#[test]
fn a_restored_snapshot_resumes_exactly_where_it_was_taken() {
    let mut h = Harness::new();
    let (graph, dims) = accumulate_graph();
    let mut state = StateStore::new();
    for frame in 1..=3 {
        h.frame(&graph, &mut state, frame, dims);
    }

    let snapshot = state.snapshot(&h.gpu, &mut h.pool).unwrap();
    let frame_4: Vec<u32> = h
        .frame(&graph, &mut state, 4, dims)
        .iter()
        .map(|v| v.to_bits())
        .collect();
    h.frame(&graph, &mut state, 5, dims);
    h.frame(&graph, &mut state, 6, dims);

    // Restoring twice proves the snapshot survives being restored.
    for _ in 0..2 {
        state.restore(&snapshot, &h.gpu, &mut h.pool).unwrap();
        let again: Vec<u32> = h
            .frame(&graph, &mut state, 4, dims)
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(
            again, frame_4,
            "frame 4 after restore must be bit-identical"
        );
    }
    snapshot.release_to(&mut h.pool);
}

#[test]
fn a_snapshot_counts_every_stored_texel() {
    let mut h = Harness::new();
    let (graph, dims) = accumulate_graph();
    let mut state = StateStore::new();

    let empty = state.snapshot(&h.gpu, &mut h.pool).unwrap();
    assert_eq!(empty.bytes(), 0);

    h.frame(&graph, &mut state, 1, dims);
    let one = state.snapshot(&h.gpu, &mut h.pool).unwrap();
    assert_eq!(one.bytes(), 4 * 4 * 4 * 4, "one 4³ R32Float field");
}

/// State can be read between frames without taking it out of the store.
#[test]
fn get_reads_a_slot_without_removing_it() {
    let mut h = Harness::new();
    let (graph, dims) = accumulate_graph();
    let mut state = StateStore::new();
    h.frame(&graph, &mut state, 1, dims);
    let sum = state
        .get(NodeId(1), "sum")
        .expect("accumulate keeps its sum in the \"sum\" slot");
    assert_all_near(
        &sum.as_field().unwrap().read_back(&h.gpu).unwrap(),
        expected(1),
    );
    assert!(state.get(NodeId(1), "no such slot").is_none());
    assert_eq!(state.len(), 1, "get leaves the slot in place");
}
