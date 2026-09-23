use std::sync::{Arc, Mutex};

use elements_core::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{
    EvalCtx, Graph, Node, NodeError, SocketId, SocketSpec, SocketType, StateStore, Time, Value,
};
use elements_core::nodes::ConstantField;

/// Two field inputs, one output; records what `input_connected` said.
struct Probe(Arc<Mutex<Vec<bool>>>);

impl Node for Probe {
    fn kind(&self) -> &'static str {
        "test.probe"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field, SocketType::Field],
            outputs: vec![SocketType::Field],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        *self.0.lock().unwrap() = vec![ctx.input_connected(0), ctx.input_connected(1)];
        Ok(vec![ctx.take_input(0)?])
    }
}

#[test]
fn an_unconnected_input_is_reported_and_may_stay_unread() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut g = Graph::new();
    let source = g.add_node(Box::new(ConstantField { value: 1.0 }));
    let probe = g.add_node(Box::new(Probe(Arc::clone(&seen))));
    g.connect(
        SocketId {
            node: source,
            index: 0,
        },
        SocketId {
            node: probe,
            index: 0,
        },
    )
    .unwrap();
    g.set_output(probe);
    let gpu = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let evaluated = g
        .eval_frame(
            &gpu,
            &mut pool,
            &mut pipelines,
            &mut state,
            Time::at(1, 1, 24.0),
            FieldDims::new(4, 4, 4),
        )
        .unwrap();
    evaluated.value.release_to(&mut pool);
    assert_eq!(*seen.lock().unwrap(), vec![true, false]);
}
