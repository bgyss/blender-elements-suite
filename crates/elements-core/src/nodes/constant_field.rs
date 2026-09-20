//! A field filled with a single scalar.

use serde::Deserialize;

use crate::gpu::{FieldFormat, fill_constant};
use crate::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};

pub const KIND: &str = "core.constant_field";

#[derive(Debug, Clone, Deserialize)]
pub struct ConstantField {
    pub value: f32,
}

impl Node for ConstantField {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Field],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let field = ctx.acquire(FieldFormat::R32Float)?;
        let value = self.value;
        ctx.with_gpu(|gpu, cache| fill_constant(gpu, cache, &field, value))?;
        Ok(vec![Value::Field(field)])
    }
}

pub(crate) fn build(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    let node: ConstantField =
        serde_json::from_value(params.clone()).map_err(|e| DocError::BadParams {
            kind: KIND.to_owned(),
            reason: e.to_string(),
        })?;
    Ok(Box::new(node))
}
