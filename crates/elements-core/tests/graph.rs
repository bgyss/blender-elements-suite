use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{
    EvalCtx, Graph, Node, NodeError, SocketId, SocketSpec, SocketType, Value,
};

/// A node that emits a fixed scalar, exercising the graph without a GPU.
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

/// A node that adds its two scalar inputs.
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

/// A node with two output sockets, both fed by a single scalar input, used to
/// build a genuine diamond: one producer reached by two distinct paths.
struct Fork;

impl Node for Fork {
    fn kind(&self) -> &'static str {
        "test.fork"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Scalar],
            outputs: vec![SocketType::Scalar, SocketType::Scalar],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let v = ctx.take_input(0)?.as_scalar()?;
        Ok(vec![Value::Scalar(v), Value::Scalar(v)])
    }
}

fn harness() -> (GpuContext, FieldPool, PipelineCache) {
    (
        GpuContext::new_headless().expect("no GPU adapter available"),
        FieldPool::new(),
        PipelineCache::new(),
    )
}

#[test]
fn evaluates_two_producers_into_one_consumer() {
    let mut g = Graph::new();
    let two = g.add_node(Box::new(Literal(2.0)));
    let three = g.add_node(Box::new(Literal(3.0)));
    let sum = g.add_node(Box::new(Add));

    g.connect(
        SocketId {
            node: two,
            index: 0,
        },
        SocketId {
            node: sum,
            index: 0,
        },
    )
    .unwrap();
    g.connect(
        SocketId {
            node: three,
            index: 0,
        },
        SocketId {
            node: sum,
            index: 1,
        },
    )
    .unwrap();
    g.set_output(sum);

    let (gpu, mut pool, mut pipelines) = harness();
    let value = g
        .eval(&gpu, &mut pool, &mut pipelines, FieldDims::new(4, 4, 4))
        .unwrap();
    assert_eq!(value.as_scalar().unwrap(), 5.0);
}

#[test]
fn detects_cycles() {
    let mut g = Graph::new();
    let a = g.add_node(Box::new(Add));
    let b = g.add_node(Box::new(Add));

    g.connect(
        SocketId { node: a, index: 0 },
        SocketId { node: b, index: 0 },
    )
    .unwrap();
    g.connect(
        SocketId { node: b, index: 0 },
        SocketId { node: a, index: 0 },
    )
    .unwrap();

    match g.topological_order() {
        Err(NodeError::Cycle(_)) => {}
        other => panic!("expected a cycle error, got {other:?}"),
    }
}

#[test]
fn a_diamond_is_not_reported_as_a_cycle() {
    let mut g = Graph::new();
    let lit = g.add_node(Box::new(Literal(3.0)));
    let fork = g.add_node(Box::new(Fork));
    let sum = g.add_node(Box::new(Add));

    g.connect(
        SocketId {
            node: lit,
            index: 0,
        },
        SocketId {
            node: fork,
            index: 0,
        },
    )
    .unwrap();
    g.connect(
        SocketId {
            node: fork,
            index: 0,
        },
        SocketId {
            node: sum,
            index: 0,
        },
    )
    .unwrap();
    g.connect(
        SocketId {
            node: fork,
            index: 1,
        },
        SocketId {
            node: sum,
            index: 1,
        },
    )
    .unwrap();
    g.set_output(sum);

    assert!(g.topological_order().is_ok());

    let (gpu, mut pool, mut pipelines) = harness();
    let value = g
        .eval(&gpu, &mut pool, &mut pipelines, FieldDims::new(4, 4, 4))
        .unwrap();
    assert_eq!(value.as_scalar().unwrap(), 6.0);
}

#[test]
fn rejects_a_nonexistent_socket() {
    let mut g = Graph::new();
    let lit = g.add_node(Box::new(Literal(1.0)));
    let add = g.add_node(Box::new(Add));

    let err = g
        .connect(
            SocketId {
                node: lit,
                index: 0,
            },
            SocketId {
                node: add,
                index: 9,
            },
        )
        .unwrap_err();
    assert!(matches!(err, NodeError::MissingInput { .. }), "got {err:?}");
}

#[test]
fn rejects_a_second_consumer_of_one_output() {
    let mut g = Graph::new();
    let lit = g.add_node(Box::new(Literal(1.0)));
    let add = g.add_node(Box::new(Add));

    g.connect(
        SocketId {
            node: lit,
            index: 0,
        },
        SocketId {
            node: add,
            index: 0,
        },
    )
    .unwrap();
    let err = g
        .connect(
            SocketId {
                node: lit,
                index: 0,
            },
            SocketId {
                node: add,
                index: 1,
            },
        )
        .unwrap_err();
    assert!(
        matches!(err, NodeError::AlreadyConsumed { .. }),
        "got {err:?}"
    );
}

#[test]
fn unconnected_input_is_an_error_at_eval_time() {
    let mut g = Graph::new();
    let add = g.add_node(Box::new(Add));
    g.set_output(add);

    let (gpu, mut pool, mut pipelines) = harness();
    let err = g
        .eval(&gpu, &mut pool, &mut pipelines, FieldDims::new(4, 4, 4))
        .unwrap_err();
    assert!(matches!(err, NodeError::MissingInput { .. }), "got {err:?}");
}

#[test]
fn evaluates_only_the_output_subgraph() {
    let mut g = Graph::new();
    let used = g.add_node(Box::new(Literal(1.0)));
    let _unused = g.add_node(Box::new(Add)); // would error if evaluated
    g.set_output(used);

    let (gpu, mut pool, mut pipelines) = harness();
    let value = g
        .eval(&gpu, &mut pool, &mut pipelines, FieldDims::new(4, 4, 4))
        .unwrap();
    assert_eq!(value.as_scalar().unwrap(), 1.0);
}

#[test]
fn values_report_type_mismatches() {
    let v = Value::Scalar(1.0);
    assert!(matches!(v.as_field(), Err(NodeError::TypeMismatch { .. })));
}

#[test]
fn field_values_carry_their_dims() {
    let (gpu, mut pool, _pipelines) = harness();
    let field = pool
        .acquire(&gpu, FieldDims::new(2, 3, 4), FieldFormat::R32Float)
        .unwrap();
    let v = Value::Field(field);
    assert_eq!(v.as_field().unwrap().dims(), FieldDims::new(2, 3, 4));
}

/// A buggy node: takes its input, then tries to read it again. Used to prove
/// this is reported distinctly from an input that was never connected.
struct TakeTwice;

impl Node for TakeTwice {
    fn kind(&self) -> &'static str {
        "test.take_twice"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Scalar],
            outputs: vec![SocketType::Scalar],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let _ = ctx.take_input(0)?;
        let _ = ctx.input(0)?;
        unreachable!("input(0) should have errored")
    }
}

#[test]
fn input_after_take_input_reports_the_input_was_taken() {
    let mut g = Graph::new();
    let lit = g.add_node(Box::new(Literal(1.0)));
    let bug = g.add_node(Box::new(TakeTwice));

    g.connect(
        SocketId {
            node: lit,
            index: 0,
        },
        SocketId {
            node: bug,
            index: 0,
        },
    )
    .unwrap();
    g.set_output(bug);

    let (gpu, mut pool, mut pipelines) = harness();
    let err = g
        .eval(&gpu, &mut pool, &mut pipelines, FieldDims::new(4, 4, 4))
        .unwrap_err();
    assert!(
        matches!(err, NodeError::InputAlreadyTaken { .. }),
        "got {err:?}"
    );
}

/// A buggy node: declares a scalar input but reads it as a field, which
/// `Value::as_field` cannot attribute to a node on its own (it stamps a
/// sentinel `NodeId(u32::MAX)`). Used to prove `Graph::eval` fills in the
/// real node id before the error leaves the graph.
struct WrongType;

impl Node for WrongType {
    fn kind(&self) -> &'static str {
        "test.wrong_type"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Scalar],
            outputs: vec![SocketType::Scalar],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let _ = ctx.input(0)?.as_field()?;
        unreachable!("as_field on a Scalar should have errored")
    }
}

#[test]
fn a_type_mismatch_inside_a_node_names_that_node() {
    let mut g = Graph::new();
    let lit = g.add_node(Box::new(Literal(1.0)));
    let bug = g.add_node(Box::new(WrongType));

    g.connect(
        SocketId {
            node: lit,
            index: 0,
        },
        SocketId {
            node: bug,
            index: 0,
        },
    )
    .unwrap();
    g.set_output(bug);

    let (gpu, mut pool, mut pipelines) = harness();
    let err = g
        .eval(&gpu, &mut pool, &mut pipelines, FieldDims::new(4, 4, 4))
        .unwrap_err();
    match err {
        NodeError::TypeMismatch { node, .. } => assert_eq!(
            node, bug,
            "the error should name the node that was actually evaluating, not a sentinel"
        ),
        other => panic!("expected TypeMismatch, got {other:?}"),
    }
}

#[test]
fn eval_without_an_output_is_an_error() {
    let g = Graph::new();
    let (gpu, mut pool, mut pipelines) = harness();
    let err = g
        .eval(&gpu, &mut pool, &mut pipelines, FieldDims::new(4, 4, 4))
        .unwrap_err();
    assert!(matches!(err, NodeError::NoOutput), "got {err:?}");
}
