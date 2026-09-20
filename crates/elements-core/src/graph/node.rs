//! The `Node` trait, the values that flow between nodes, and node errors.

use crate::gpu::{Field, FieldDims, FieldFormat, FieldPool, GpuContext, GpuError, PipelineCache};

use super::socket::{NodeId, SocketSpec, SocketType};

/// Everything that can go wrong building or evaluating a graph.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("the graph contains a cycle through node {0:?}")]
    Cycle(NodeId),
    #[error("node {node:?} input {index} is not connected")]
    MissingInput { node: NodeId, index: u32 },
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
pub enum Value {
    Field(Field),
    Scalar(f32),
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Field(field) => f.debug_tuple("Field").field(&field.dims()).finish(),
            Self::Scalar(s) => f.debug_tuple("Scalar").field(s).finish(),
        }
    }
}

impl Value {
    pub fn as_field(&self) -> Result<&Field, NodeError> {
        match self {
            Self::Field(f) => Ok(f),
            Self::Scalar(_) => Err(NodeError::TypeMismatch {
                node: NodeId(u32::MAX),
                index: 0,
                expected: SocketType::Field,
            }),
        }
    }

    pub fn as_scalar(&self) -> Result<f32, NodeError> {
        match self {
            Self::Scalar(s) => Ok(*s),
            Self::Field(_) => Err(NodeError::TypeMismatch {
                node: NodeId(u32::MAX),
                index: 0,
                expected: SocketType::Scalar,
            }),
        }
    }

    pub fn socket_type(&self) -> SocketType {
        match self {
            Self::Field(_) => SocketType::Field,
            Self::Scalar(_) => SocketType::Scalar,
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
            .ok_or(NodeError::MissingInput {
                node: self.node,
                index,
            })
    }

    /// Take ownership of input `index`, leaving it unavailable to later reads.
    /// This is how pass-through nodes forward a GPU field without copying it.
    pub fn take_input(&mut self, index: u32) -> Result<Value, NodeError> {
        self.inputs
            .get_mut(index as usize)
            .and_then(Option::take)
            .ok_or(NodeError::MissingInput {
                node: self.node,
                index,
            })
    }

    /// Acquire a pooled field at the current evaluation dims.
    pub fn acquire(&mut self, format: FieldFormat) -> Result<Field, NodeError> {
        let dims = self.dims;
        Ok(self.pool.acquire(self.gpu, dims, format)?)
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
