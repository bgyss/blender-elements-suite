//! The graph's terminal node. Passes its input through unchanged.

use crate::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};

pub const KIND: &str = "core.output";

#[derive(Debug, Clone, Default)]
pub struct Output;

impl Node for Output {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field],
            outputs: vec![SocketType::Field],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        // Move the field through rather than copying it on the GPU.
        let value = ctx.take_input(0)?;

        // `Value::as_field` cannot name its own position (it stamps
        // `NodeId(u32::MAX)` as a sentinel — see its doc comment), so a
        // type-mismatched input here would otherwise report a meaningless
        // node id. Check the type at this call site instead, where we know
        // both the real node (`ctx.node_id()`) and the real socket index
        // (0), and substitute them into the error before it escapes.
        if let Err(NodeError::TypeMismatch { expected, .. }) = value.as_field() {
            return Err(NodeError::TypeMismatch {
                node: ctx.node_id(),
                index: 0,
                expected,
            });
        }

        Ok(vec![value])
    }
}

pub(crate) fn build(_params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    Ok(Box::new(Output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};
    use crate::graph::NodeId;

    /// The ADDITIONAL REQUIREMENT: when a built-in node reads an input of
    /// the wrong type, the error must name the real node and the real
    /// socket index, not the `NodeId(u32::MAX)` sentinel that a bare
    /// `Value` reports on its own.
    ///
    /// This bypasses `Graph::connect` (which already rejects a
    /// type-mismatched wire at document-build time) to exercise the
    /// defense inside `Output::eval` itself: an `EvalCtx` built by hand
    /// with a `Scalar` sitting in the `Field` input slot, as if a future
    /// bug elsewhere ever let one through.
    #[test]
    fn type_mismatch_on_input_names_node_and_socket() {
        let gpu = GpuContext::new_headless().expect("no GPU adapter available");
        let mut pool = FieldPool::new();
        let mut pipelines = PipelineCache::new();

        let mut ctx = EvalCtx {
            gpu: &gpu,
            pool: &mut pool,
            pipelines: &mut pipelines,
            dims: FieldDims::new(2, 2, 2),
            node: NodeId(42),
            inputs: vec![Some(Value::Scalar(1.0))],
            taken: vec![false],
        };

        let err = Output.eval(&mut ctx).unwrap_err();
        match err {
            NodeError::TypeMismatch { node, index, .. } => {
                assert_eq!(
                    node,
                    NodeId(42),
                    "should name the real node, not a sentinel"
                );
                assert_eq!(index, 0, "should name the real socket index");
            }
            other => panic!("expected TypeMismatch, got {other:?}"),
        }
    }
}
