//! Small helpers for nodes that take several inputs or acquire several fields.

use elements_core::gpu::{Field, FieldFormat, GpuContext, GpuError, PipelineCache, StaggeredField};
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

/// Whether inputs `a` and `b`, which mean something only together, are
/// connected: both (true), neither (false), or one alone (an error naming both).
pub(crate) fn pair(ctx: &EvalCtx<'_>, a: u32, b: u32) -> Result<bool, NodeError> {
    let node = ctx.node_id();
    match (ctx.input_connected(a), ctx.input_connected(b)) {
        (true, true) => Ok(true),
        (false, false) => Ok(false),
        (true, false) => Err(NodeError::IncompletePair {
            node,
            connected: a,
            missing: b,
        }),
        (false, true) => Err(NodeError::IncompletePair {
            node,
            connected: b,
            missing: a,
        }),
    }
}

/// Take the listed inputs, in order, with their indices. On failure, every
/// input already taken is released.
pub(crate) fn take_listed(
    ctx: &mut EvalCtx<'_>,
    indices: &[u32],
) -> Result<Vec<(u32, Value)>, NodeError> {
    let mut taken = Vec::with_capacity(indices.len());
    for &i in indices {
        match ctx.take_input(i) {
            Ok(value) => taken.push((i, value)),
            Err(e) => {
                for (_, value) in taken {
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
fn acquire_cells(ctx: &mut EvalCtx<'_>, n: usize) -> Result<Vec<Field>, NodeError> {
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

/// Acquire `cells` uninitialised `R32Float` fields and one staggered field at
/// the domain's dims, run `fill` on them, and return them as values: the cell
/// fields in order, then the vector field. If any acquisition or `fill` fails,
/// everything acquired goes back to the pool before the error returns.
pub(crate) fn produce(
    ctx: &mut EvalCtx<'_>,
    cells: usize,
    fill: impl FnOnce(
        &GpuContext,
        &mut PipelineCache,
        &[Field],
        &StaggeredField,
    ) -> Result<(), GpuError>,
) -> Result<Vec<Value>, NodeError> {
    let fields = acquire_cells(ctx, cells)?;
    let vector = match ctx.acquire_vector_uninit() {
        Ok(v) => v,
        Err(e) => {
            for field in fields {
                ctx.release(Value::Field(field));
            }
            return Err(e);
        }
    };
    let filled = ctx.with_gpu(|gpu, cache| fill(gpu, cache, &fields, &vector));
    let mut values: Vec<Value> = fields.into_iter().map(Value::Field).collect();
    values.push(Value::VectorField(vector));
    if let Err(e) = filled {
        for value in values {
            ctx.release(value);
        }
        return Err(e);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use elements_core::gpu::{FieldDims, FieldPool};
    use elements_core::graph::{Graph, Node, SocketSpec, SocketType};

    /// A node whose fill always fails, after `produce` has acquired everything.
    #[derive(Debug)]
    struct FailingFill {
        cells: usize,
    }

    impl Node for FailingFill {
        fn kind(&self) -> &'static str {
            "test.failing_fill"
        }

        fn sockets(&self) -> SocketSpec {
            let mut outputs = vec![SocketType::Field; self.cells];
            outputs.push(SocketType::VectorField);
            SocketSpec {
                inputs: vec![],
                outputs,
            }
        }

        fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
            produce(ctx, self.cells, |_, _, _, _| {
                Err(GpuError::Validation("fill failed on purpose".to_owned()))
            })
        }
    }

    #[test]
    fn a_failing_fill_returns_every_field_to_the_pool() {
        let gpu = GpuContext::new_headless().expect("no GPU adapter available");
        for cells in [1, 3] {
            let mut graph = Graph::new();
            let node = graph.add_node(Box::new(FailingFill { cells }));
            graph.set_output(node);
            let mut pool = FieldPool::new();
            let mut pipelines = PipelineCache::new();
            let err = graph
                .eval(&gpu, &mut pool, &mut pipelines, FieldDims::new(4, 5, 6))
                .unwrap_err();
            assert!(
                matches!(err, NodeError::Gpu(GpuError::Validation(_))),
                "got {err:?}"
            );
            assert_eq!(
                pool.allocation_count(),
                cells as u64 + 3,
                "{cells} cell fields and three faces were acquired"
            );
            assert_eq!(
                pool.pooled_count() as u64,
                pool.allocation_count(),
                "every allocated texture must be back in the pool ({cells} cells)"
            );
        }
    }
}
