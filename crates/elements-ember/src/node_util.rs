//! Small helpers for nodes that take several inputs or acquire several fields.

use elements_core::gpu::{Field, FieldFormat};
use elements_core::graph::{EvalCtx, NodeError, Value};

/// Take inputs `0..n` in order. On failure, every input already taken is released.
pub(crate) fn take_inputs(ctx: &mut EvalCtx<'_>, n: u32) -> Result<Vec<Value>, NodeError> {
    let mut taken = Vec::with_capacity(n as usize);
    for i in 0..n {
        match ctx.take_input(i) {
            Ok(value) => taken.push(value),
            Err(e) => {
                for value in taken {
                    ctx.release(value);
                }
                return Err(e);
            }
        }
    }
    Ok(taken)
}

/// Acquire `n` uninitialised `R32Float` fields at the domain's dims. On
/// failure, every field already acquired is released.
pub(crate) fn acquire_cells(ctx: &mut EvalCtx<'_>, n: usize) -> Result<Vec<Field>, NodeError> {
    let mut fields = Vec::with_capacity(n);
    for _ in 0..n {
        match ctx.acquire_uninit(FieldFormat::R32Float) {
            Ok(field) => fields.push(field),
            Err(e) => {
                for field in fields {
                    ctx.release(Value::Field(field));
                }
                return Err(e);
            }
        }
    }
    Ok(fields)
}
