//! A field of seeded value noise.

use serde::Deserialize;

use crate::gpu::{FieldFormat, fill_curl_noise};
use crate::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};

pub const KIND: &str = "core.noise_field";

fn default_frequency() -> f32 {
    4.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct NoiseField {
    pub seed: u64,
    #[serde(default = "default_frequency")]
    pub frequency: f32,
}

impl Node for NoiseField {
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
        let (seed, frequency) = (self.seed, self.frequency);
        ctx.with_gpu(|gpu, cache| fill_curl_noise(gpu, cache, &field, seed, frequency))?;
        Ok(vec![Value::Field(field)])
    }
}

pub(crate) fn build(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    let node: NoiseField =
        serde_json::from_value(params.clone()).map_err(|e| DocError::BadParams {
            kind: KIND.to_owned(),
            reason: e.to_string(),
        })?;
    Ok(Box::new(node))
}
