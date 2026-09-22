//! Integrates its input over time: `sum += input * dt` every frame.
//!
//! The simplest possible stateful node, and the proof that state, time and the
//! timeline work. Its output at frame N has a closed form a test can check:
//! for a constant input `c`, `c * (N - start_frame + 1) / fps`.

use crate::gpu::accumulate_into;
use crate::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};

pub const KIND: &str = "core.accumulate";

const SUM: &str = "sum";

#[derive(Debug, Clone, Default)]
pub struct Accumulate;

impl Node for Accumulate {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field],
            outputs: vec![SocketType::Field],
        }
    }

    fn stateful(&self) -> bool {
        true
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let sum = match ctx.take_state(SUM)? {
            Some(Value::Field(field)) => field,
            Some(other) => {
                ctx.release(other);
                return Err(NodeError::StateShape {
                    node: ctx.node_id(),
                    slot: SUM,
                });
            }
            // The first step starts from zero, so the store must hand back a
            // cleared field, not a recycled one.
            None => ctx.acquire_zeroed()?,
        };

        let input = ctx.take_input(0)?;
        let dt = ctx.time().dt as f32;
        let result = match input.as_field() {
            Ok(field) => ctx.with_gpu(|gpu, cache| accumulate_into(gpu, cache, &sum, field, dt)),
            Err(_) => Err(NodeError::TypeMismatch {
                node: ctx.node_id(),
                index: 0,
                expected: SocketType::Field,
            }),
        };
        ctx.release(input);
        if let Err(e) = result {
            ctx.release(Value::Field(sum));
            return Err(e);
        }

        // The output is a copy: the sum itself stays in the store for the next frame.
        let out = ctx.duplicate(&sum)?;
        ctx.put_state(SUM, Value::Field(sum))?;
        Ok(vec![Value::Field(out)])
    }
}

pub(crate) fn build(_params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    Ok(Box::new(Accumulate))
}
