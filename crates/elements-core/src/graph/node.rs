//! The `Node` trait, the values that flow between nodes, and node errors.

use crate::gpu::{
    Field, FieldDims, FieldFormat, FieldPool, GpuContext, GpuError, PipelineCache, StaggeredField,
};

use super::socket::{NodeId, SocketSpec, SocketType};

/// Everything that can go wrong building or evaluating a graph.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("the graph contains a cycle through node {0:?}")]
    Cycle(NodeId),
    #[error("node {node:?} input {index} is not connected")]
    MissingInput { node: NodeId, index: u32 },
    #[error("node {node:?} input {index} was already taken by this node")]
    InputAlreadyTaken { node: NodeId, index: u32 },
    #[error("node {node:?} socket {index} expected {expected:?}")]
    TypeMismatch {
        node: NodeId,
        index: u32,
        expected: SocketType,
    },
    #[error("node {node:?} output {index} already feeds another input")]
    AlreadyConsumed { node: NodeId, index: u32 },
    #[error("no such node: {0:?}")]
    UnknownNode(NodeId),
    #[error("no output node is set on this graph")]
    NoOutput,
    #[error(transparent)]
    Gpu(#[from] GpuError),
}

/// A value travelling along a connection.
///
/// `Field` wraps a GPU texture and is deliberately not `Clone` — cloning it
/// would alias or double-free the underlying GPU resource. `Value` therefore
/// cannot be `Clone` either, and the evaluator moves values between producer
/// and consumer instead of sharing them.
///
/// `VectorField` is three `Field`s and so is much larger than `Scalar`, but
/// values only move at graph evaluation boundaries (a handful of times per
/// frame), so the copy is not a hot path. Boxing it would add an indirection
/// to every call site that matches on `Value` for no measurable benefit.
#[allow(clippy::large_enum_variant)]
pub enum Value {
    Field(Field),
    Scalar(f32),
    VectorField(StaggeredField),
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Field(field) => f.debug_tuple("Field").field(&field.dims()).finish(),
            Self::Scalar(s) => f.debug_tuple("Scalar").field(s).finish(),
            Self::VectorField(v) => f.debug_tuple("VectorField").field(&v.cells()).finish(),
        }
    }
}

impl NodeError {
    /// Fill in the node actually being evaluated when the error was raised
    /// from inside a bare `Value` (which does not know its own position).
    ///
    /// `Value::as_field`/`as_scalar` cannot name the node they belong to, so
    /// they stamp `NodeId(u32::MAX)` as a sentinel. `Graph::eval` calls this
    /// on every error a node's `eval` returns, replacing that sentinel with
    /// the id it was actually evaluating, so a type-mismatch bug inside a
    /// node reports which node failed instead of a meaningless
    /// `NodeId(4294967295)`. Errors that already name a real node pass
    /// through unchanged.
    pub(crate) fn with_evaluating_node(self, node: NodeId) -> Self {
        match self {
            Self::TypeMismatch {
                node: NodeId(u32::MAX),
                index,
                expected,
            } => Self::TypeMismatch {
                node,
                index,
                expected,
            },
            other => other,
        }
    }
}

impl Value {
    pub fn as_field(&self) -> Result<&Field, NodeError> {
        match self {
            Self::Field(f) => Ok(f),
            _ => Err(NodeError::TypeMismatch {
                node: NodeId(u32::MAX),
                index: 0,
                expected: SocketType::Field,
            }),
        }
    }

    pub fn as_scalar(&self) -> Result<f32, NodeError> {
        match self {
            Self::Scalar(s) => Ok(*s),
            _ => Err(NodeError::TypeMismatch {
                node: NodeId(u32::MAX),
                index: 0,
                expected: SocketType::Scalar,
            }),
        }
    }

    pub fn as_vector_field(&self) -> Result<&StaggeredField, NodeError> {
        match self {
            Self::VectorField(v) => Ok(v),
            _ => Err(NodeError::TypeMismatch {
                node: NodeId(u32::MAX),
                index: 0,
                expected: SocketType::VectorField,
            }),
        }
    }

    pub fn socket_type(&self) -> SocketType {
        match self {
            Self::Field(_) => SocketType::Field,
            Self::Scalar(_) => SocketType::Scalar,
            Self::VectorField(_) => SocketType::VectorField,
        }
    }

    /// Return every GPU texture this value owns to `pool`.
    ///
    /// Dropping a `Value` frees its textures instead, forcing the next frame to
    /// allocate again. Code that is finished with a value calls this.
    pub fn release_to(self, pool: &mut FieldPool) {
        match self {
            Self::Field(f) => pool.release(f),
            Self::VectorField(v) => pool.release_staggered(v),
            Self::Scalar(_) => {}
        }
    }
}

/// What a node is handed when it evaluates.
///
/// Inputs are owned: the evaluator moves each producer's value into the
/// consumer, which is why an output may feed only one input.
pub struct EvalCtx<'a> {
    pub(crate) gpu: &'a GpuContext,
    pub(crate) pool: &'a mut FieldPool,
    pub(crate) pipelines: &'a mut PipelineCache,
    pub(crate) dims: FieldDims,
    pub(crate) node: NodeId,
    pub(crate) inputs: Vec<Option<Value>>,
    /// Tracks which indices `take_input` has already removed, so `input` can
    /// report a distinct error for "taken" versus "never connected".
    pub(crate) taken: Vec<bool>,
}

impl EvalCtx<'_> {
    pub fn gpu(&self) -> &GpuContext {
        self.gpu
    }

    /// The resolution this evaluation is running at.
    pub fn dims(&self) -> FieldDims {
        self.dims
    }

    pub fn node_id(&self) -> NodeId {
        self.node
    }

    /// Borrow input `index`.
    pub fn input(&self, index: u32) -> Result<&Value, NodeError> {
        self.inputs
            .get(index as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                if self.taken.get(index as usize).copied().unwrap_or(false) {
                    NodeError::InputAlreadyTaken {
                        node: self.node,
                        index,
                    }
                } else {
                    NodeError::MissingInput {
                        node: self.node,
                        index,
                    }
                }
            })
    }

    /// Take ownership of input `index`, leaving it unavailable to later reads.
    /// This is how pass-through nodes forward a GPU field without copying it.
    ///
    /// Trap: calling `input(index)` after this will not report `MissingInput`
    /// (which would look like a wiring mistake in the graph) but
    /// `InputAlreadyTaken` (a bug in this node's own `eval`, since it read the
    /// same input twice).
    pub fn take_input(&mut self, index: u32) -> Result<Value, NodeError> {
        let value = self
            .inputs
            .get_mut(index as usize)
            .and_then(Option::take)
            .ok_or(NodeError::MissingInput {
                node: self.node,
                index,
            })?;
        if let Some(slot) = self.taken.get_mut(index as usize) {
            *slot = true;
        }
        Ok(value)
    }

    /// Acquire a pooled field at the current evaluation dims WITHOUT clearing it.
    ///
    /// The contents are whatever the texture's last user left. Use this only
    /// when the node writes every voxel before anything reads the field.
    pub fn acquire_uninit(&mut self, format: FieldFormat) -> Result<Field, NodeError> {
        let dims = self.dims;
        Ok(self.pool.acquire(self.gpu, dims, format)?)
    }

    /// Acquire a pooled `R32Float` field at the current evaluation dims, filled with zeros.
    pub fn acquire_zeroed(&mut self) -> Result<Field, NodeError> {
        let dims = self.dims;
        Ok(self.pool.acquire_zeroed(self.gpu, self.pipelines, dims)?)
    }

    /// Acquire a staggered vector field for the current domain, contents unspecified.
    pub fn acquire_vector_uninit(&mut self) -> Result<StaggeredField, NodeError> {
        let dims = self.dims;
        Ok(self.pool.acquire_staggered_uninit(self.gpu, dims)?)
    }

    /// Acquire a staggered vector field for the current domain with every face zeroed.
    pub fn acquire_vector_zeroed(&mut self) -> Result<StaggeredField, NodeError> {
        let dims = self.dims;
        Ok(self
            .pool
            .acquire_staggered_zeroed(self.gpu, self.pipelines, dims)?)
    }

    /// Run GPU work with the device and pipeline cache borrowed together.
    ///
    /// Nodes must use this rather than borrowing `gpu()` and the cache
    /// separately, which the borrow checker rejects.
    pub fn with_gpu<T>(
        &mut self,
        f: impl FnOnce(&GpuContext, &mut PipelineCache) -> Result<T, GpuError>,
    ) -> Result<T, NodeError> {
        Ok(f(self.gpu, self.pipelines)?)
    }
}

/// One unit of computation in a graph.
pub trait Node: Send + Sync {
    /// Stable identifier used in the `.elements` document, e.g. `"core.noise_field"`.
    fn kind(&self) -> &'static str;

    /// This node's input and output types.
    fn sockets(&self) -> SocketSpec;

    /// Compute this node's outputs. Must return exactly `sockets().outputs.len()` values.
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `with_evaluating_node` must only replace the `NodeId(u32::MAX)`
    /// sentinel. An error that already names a real node — such as
    /// `MissingInput` naming the node whose input is unconnected — must pass
    /// through unchanged, or the error would end up naming the wrong node
    /// and mislead whoever reads it.
    #[test]
    fn does_not_clobber_an_error_that_already_names_a_real_node() {
        let err = NodeError::MissingInput {
            node: NodeId(7),
            index: 0,
        };

        let result = err.with_evaluating_node(NodeId(3));

        match result {
            NodeError::MissingInput { node, .. } => {
                assert_eq!(
                    node,
                    NodeId(7),
                    "should not have clobbered the real node id"
                )
            }
            other => panic!("expected MissingInput, got {other:?}"),
        }
    }
}
