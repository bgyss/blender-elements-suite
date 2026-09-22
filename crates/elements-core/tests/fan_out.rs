use elements_core::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache, fill_constant};
use elements_core::graph::{
    EvalCtx, Evaluated, Graph, Node, NodeError, NodeId, SocketId, SocketSpec, SocketType,
    StateStore, Time, Value,
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

/// Reads both inputs by reference, so it never forces a copy.
struct Add;

impl Node for Add {
    fn kind(&self) -> &'static str {
        "test.add"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Scalar, SocketType::Scalar],
            outputs: vec![SocketType::Scalar],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let a = ctx.input(0)?.as_scalar()?;
        let b = ctx.input(1)?.as_scalar()?;
        Ok(vec![Value::Scalar(a + b)])
    }
}

/// Takes its field and overwrites it. If it were handed a texture another
/// branch also holds, that branch would see this node's value.
struct Overwrite(f32);

impl Node for Overwrite {
    fn kind(&self) -> &'static str {
        "test.overwrite"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field],
            outputs: vec![SocketType::Field],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let field = match ctx.take_input(0)? {
            Value::Field(field) => field,
            other => {
                ctx.release(other);
                return Err(NodeError::TypeMismatch {
                    node: ctx.node_id(),
                    index: 0,
                    expected: SocketType::Field,
                });
            }
        };
        let value = self.0;
        ctx.with_gpu(|gpu, cache| fill_constant(gpu, cache, &field, value))?;
        Ok(vec![Value::Field(field)])
    }
}

/// Outputs its second input. Its first input is only borrowed, never read,
/// so the evaluator must release it.
struct PickSecond;

impl Node for PickSecond {
    fn kind(&self) -> &'static str {
        "test.pick_second"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field, SocketType::Field],
            outputs: vec![SocketType::Field],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        Ok(vec![ctx.take_input(1)?])
    }
}

fn link(g: &mut Graph, from: NodeId, from_index: u32, to: NodeId, to_index: u32) {
    g.connect(
        SocketId {
            node: from,
            index: from_index,
        },
        SocketId {
            node: to,
            index: to_index,
        },
    )
    .unwrap();
}

/// constant(0.5) feeds BOTH Overwrite(9) and a pass-through. PickSecond outputs
/// the pass-through's value, which must still be 0.5.
fn branching_graph() -> Graph {
    let mut g = Graph::new();
    let source = g.add_node(Box::new(ConstantField { value: 0.5 }));
    let overwrite = g.add_node(Box::new(Overwrite(9.0)));
    let keep = g.add_node(Box::new(Output));
    let pick = g.add_node(Box::new(PickSecond));
    let out = g.add_node(Box::new(Output));
    link(&mut g, source, 0, overwrite, 0);
    link(&mut g, source, 0, keep, 0);
    link(&mut g, overwrite, 0, pick, 0);
    link(&mut g, keep, 0, pick, 1);
    link(&mut g, pick, 0, out, 0);
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
}

#[test]
fn a_field_feeding_two_consumers_is_copied_for_the_first() {
    let mut h = Harness::new();
    let evaluated = h.eval(&branching_graph());
    let values = evaluated
        .value
        .as_field()
        .unwrap()
        .read_back(&h.gpu)
        .unwrap();
    assert!(
        values.iter().all(|&v| v == 0.5),
        "the untouched branch saw {values:?}"
    );
    assert_eq!(evaluated.stats.copies, 1);
}

#[test]
fn a_chain_with_no_branching_copies_nothing() {
    let mut h = Harness::new();
    let mut g = Graph::new();
    let source = g.add_node(Box::new(ConstantField { value: 0.5 }));
    let out = g.add_node(Box::new(Output));
    link(&mut g, source, 0, out, 0);
    g.set_output(out);
    assert_eq!(h.eval(&g).stats.copies, 0);
}

#[test]
fn a_scalar_can_feed_both_inputs_of_one_node() {
    let mut h = Harness::new();
    let mut g = Graph::new();
    let two = g.add_node(Box::new(Literal(2.0)));
    let sum = g.add_node(Box::new(Add));
    link(&mut g, two, 0, sum, 0);
    link(&mut g, two, 0, sum, 1);
    g.set_output(sum);
    let evaluated = h.eval(&g);
    assert_eq!(evaluated.value.as_scalar().unwrap(), 4.0);
    assert_eq!(evaluated.stats.copies, 0);
}

#[test]
fn fields_return_to_the_pool_after_their_last_use() {
    let mut h = Harness::new();
    let graph = branching_graph();

    let first = h.eval(&graph);
    first.value.release_to(&mut h.pool);
    let after_first = h.pool.allocation_count();

    for _ in 0..3 {
        let again = h.eval(&graph);
        again.value.release_to(&mut h.pool);
    }
    assert_eq!(
        h.pool.allocation_count(),
        after_first,
        "every frame after the first must be served entirely from the pool"
    );
}

#[test]
fn an_input_accepts_only_one_edge() {
    let mut g = Graph::new();
    let a = g.add_node(Box::new(Literal(1.0)));
    let b = g.add_node(Box::new(Literal(2.0)));
    let sum = g.add_node(Box::new(Add));
    link(&mut g, a, 0, sum, 0);
    let err = g
        .connect(
            SocketId { node: b, index: 0 },
            SocketId {
                node: sum,
                index: 0,
            },
        )
        .unwrap_err();
    assert!(
        matches!(err, NodeError::InputAlreadyConnected { index: 0, .. }),
        "got {err:?}"
    );
}
