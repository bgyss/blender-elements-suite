use std::sync::{Arc, Mutex};

use elements_core::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{
    EvalCtx, Graph, Node, NodeError, NodeId, SocketId, SocketSpec, SocketType, StateStore, Time,
    Value,
};
use elements_core::nodes::{ConstantField, Output};

/// Two field outputs; records what `output_wanted` said about each.
struct Probe(Arc<Mutex<Vec<bool>>>);

impl Node for Probe {
    fn kind(&self) -> &'static str {
        "test.probe"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field],
            outputs: vec![SocketType::Field, SocketType::Field],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        *self.0.lock().unwrap() = vec![ctx.output_wanted(0), ctx.output_wanted(1)];
        let field = ctx.take_input(0)?;
        Ok(vec![field, Value::Scalar(0.0)])
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

fn wanted(probe_is_result: bool) -> Vec<bool> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut g = Graph::new();
    let source = g.add_node(Box::new(ConstantField { value: 1.0 }));
    let probe = g.add_node(Box::new(Probe(Arc::clone(&seen))));
    link(&mut g, source, 0, probe, 0);
    if probe_is_result {
        g.set_output(probe);
    } else {
        let out = g.add_node(Box::new(Output));
        link(&mut g, probe, 0, out, 0);
        g.set_output(out);
    }
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
    seen.lock().unwrap().clone()
}

#[test]
fn an_output_is_wanted_when_a_node_reads_it_or_it_is_the_result() {
    assert_eq!(wanted(false), vec![true, false], "read by the output node");
    assert_eq!(
        wanted(true),
        vec![true, false],
        "socket 0 is the graph's result"
    );
}
