use elements_core::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{
    Document, EvalCtx, Evaluated, Graph, Node, NodeError, NodeId, NodeRegistry, SocketId,
    SocketSpec, SocketType, StateStore, Time, Timeline, TimelineConfig, Value,
};
use elements_core::nodes::{ConstantField, Output};

struct Literal(f32);

impl Node for Literal {
    fn kind(&self) -> &'static str {
        "test.literal"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Scalar],
        }
    }
    fn eval(&self, _ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        Ok(vec![Value::Scalar(self.0)])
    }
}

fn socket(node: NodeId, index: u32) -> SocketId {
    SocketId { node, index }
}

/// Node 0 = constant 0.25 -> Output (node 2); node 1 = constant 0.75, not upstream of Output.
fn graph() -> Graph {
    let mut g = Graph::new();
    let a = g.add_node(Box::new(ConstantField { value: 0.25 }));
    let _b = g.add_node(Box::new(ConstantField { value: 0.75 }));
    let out = g.add_node(Box::new(Output));
    g.connect(socket(a, 0), socket(out, 0)).unwrap();
    g.set_output(out);
    g
}

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

    fn eval(&mut self, graph: &Graph) -> Evaluated {
        let mut state = StateStore::new();
        graph
            .eval_frame(
                &self.gpu,
                &mut self.pool,
                &mut self.pipelines,
                &mut state,
                Time::at(1, 1, 24.0),
                FieldDims::new(4, 4, 4),
            )
            .unwrap()
    }

    fn read(&self, value: &Value) -> Vec<f32> {
        value.as_field().unwrap().read_back(&self.gpu).unwrap()
    }
}

#[test]
fn extras_come_back_in_the_order_they_were_named() {
    let mut h = Harness::new();
    let mut g = graph();
    g.set_extra_outputs(vec![socket(NodeId(1), 0), socket(NodeId(0), 0)])
        .unwrap();
    let evaluated = h.eval(&g);
    assert_eq!(evaluated.extras.len(), 2);
    assert!(h.read(&evaluated.extras[0]).iter().all(|&v| v == 0.75));
    assert!(h.read(&evaluated.extras[1]).iter().all(|&v| v == 0.25));
    assert!(h.read(&evaluated.value).iter().all(|&v| v == 0.25));
    evaluated.release_to(&mut h.pool);
}

#[test]
fn a_socket_that_also_feeds_the_output_is_copied_not_moved() {
    let mut h = Harness::new();
    let mut g = graph();
    g.set_extra_outputs(vec![socket(NodeId(0), 0)]).unwrap();
    let evaluated = h.eval(&g);
    assert!(h.read(&evaluated.value).iter().all(|&v| v == 0.25));
    assert!(h.read(&evaluated.extras[0]).iter().all(|&v| v == 0.25));
    evaluated.release_to(&mut h.pool);
}

#[test]
fn a_graph_without_extras_returns_none() {
    let mut h = Harness::new();
    let evaluated = h.eval(&graph());
    assert!(evaluated.extras.is_empty());
    evaluated.release_to(&mut h.pool);
}

#[test]
fn set_extra_outputs_rejects_bad_sockets() {
    let mut g = graph();
    let scalar = g.add_node(Box::new(Literal(1.0)));

    let err = g
        .set_extra_outputs(vec![socket(NodeId(99), 0)])
        .unwrap_err();
    assert!(matches!(err, NodeError::UnknownNode(_)), "got {err:?}");

    for (what, sockets) in [
        ("no such output", vec![socket(NodeId(0), 9)]),
        (
            "duplicate",
            vec![socket(NodeId(0), 0), socket(NodeId(0), 0)],
        ),
        ("result socket", vec![socket(NodeId(2), 0)]),
        ("scalar socket", vec![socket(scalar, 0)]),
    ] {
        let err = g.set_extra_outputs(sockets).unwrap_err();
        assert!(
            matches!(err, NodeError::BadExtraOutput { .. }),
            "{what}: got {err:?}"
        );
    }
    assert!(
        g.extra_outputs().is_empty(),
        "a rejected call must leave the old list"
    );
}

#[test]
fn evaluated_release_to_returns_every_field_to_the_pool() {
    let mut h = Harness::new();
    let mut g = graph();
    g.set_extra_outputs(vec![socket(NodeId(1), 0), socket(NodeId(0), 0)])
        .unwrap();
    let evaluated = h.eval(&g);
    let before = h.pool.pooled_count();
    evaluated.release_to(&mut h.pool);
    assert_eq!(h.pool.pooled_count(), before + 3);
}

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

/// Go to frame 3 of the accumulate graph on a fresh timeline and release the result.
/// Returns (extras seen, fields allocated, fields back in the pool).
fn goto_three(with_extra: bool) -> (usize, u64, usize) {
    let (mut graph, dims) = Document::from_json(ACCUMULATE_DOC)
        .unwrap()
        .into_graph(&NodeRegistry::with_builtins())
        .unwrap();
    if with_extra {
        graph.set_extra_outputs(vec![socket(NodeId(1), 0)]).unwrap();
    }
    let gpu = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut timeline = Timeline::new(TimelineConfig {
        start_frame: 1,
        ..TimelineConfig::default()
    });
    let evaluated = timeline
        .goto(&graph, &gpu, &mut pool, &mut pipelines, dims, 3)
        .unwrap();
    let extras = evaluated.extras.len();
    if with_extra {
        let v = evaluated.extras[0]
            .as_field()
            .unwrap()
            .read_back(&gpu)
            .unwrap();
        assert!(
            v.iter().all(|&x| (x - 0.0625).abs() < 1e-6),
            "frame 3 is 0.5 * 3 / 24"
        );
    }
    evaluated.release_to(&mut pool);
    (extras, pool.allocation_count(), pool.pooled_count())
}

#[test]
fn a_timeline_goto_returns_extras_for_the_asked_frame_and_releases_skipped_ones() {
    let (plain_extras, plain_alloc, plain_pooled) = goto_three(false);
    let (extras, alloc, pooled) = goto_three(true);
    assert_eq!(plain_extras, 0);
    assert_eq!(extras, 1);
    // Whatever the extra run allocated beyond the plain one must be back in the pool.
    assert_eq!(
        pooled as i64 - plain_pooled as i64,
        alloc as i64 - plain_alloc as i64,
        "a skipped frame's extra leaked"
    );
}
