//! The `Node` trait, the values that flow between nodes, and node errors.

use std::collections::HashMap;

use crate::gpu::{
    Axis, Field, FieldDims, FieldFormat, FieldPool, GpuContext, GpuError, PipelineCache,
    StaggeredField,
};

use super::socket::{NodeId, SocketId, SocketSpec, SocketType};
use super::state::StateStore;
use super::time::Time;

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
    #[error("node {node:?} input {index} is already connected")]
    InputAlreadyConnected { node: NodeId, index: u32 },
    #[error("no such node: {0:?}")]
    UnknownNode(NodeId),
    #[error("no output node is set on this graph")]
    NoOutput,
    #[error("node {node:?} is not stateful but tried to use persistent state")]
    NotStateful { node: NodeId },
    #[error("node {node:?} state slot {slot:?} does not match the current domain")]
    StateShape { node: NodeId, slot: &'static str },
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

/// Counters describing one evaluation, for tests and profiling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EvalStats {
    /// GPU copies made because a field fed more than one input, and a
    /// consumer that was not the last one took ownership of it.
    pub copies: u32,
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

    /// Bytes of GPU memory this value's textures occupy (4 for a scalar).
    pub fn gpu_bytes(&self) -> u64 {
        let field_bytes =
            |f: &Field| f.dims().voxel_count() as u64 * f.format().bytes_per_voxel() as u64;
        match self {
            Self::Field(f) => field_bytes(f),
            Self::Scalar(_) => std::mem::size_of::<f32>() as u64,
            Self::VectorField(v) => Axis::ALL.iter().map(|&a| field_bytes(v.face(a))).sum(),
        }
    }

    /// A copy of this value, with GPU contents copied into pooled textures.
    pub fn duplicate(&self, gpu: &GpuContext, pool: &mut FieldPool) -> Result<Value, GpuError> {
        Ok(match self {
            Self::Field(f) => Self::Field(pool.duplicate(gpu, f)?),
            Self::Scalar(s) => Self::Scalar(*s),
            Self::VectorField(v) => {
                let x = pool.duplicate(gpu, v.face(Axis::X))?;
                let y = pool.duplicate(gpu, v.face(Axis::Y))?;
                let z = pool.duplicate(gpu, v.face(Axis::Z))?;
                Self::VectorField(StaggeredField::from_faces(v.cells(), [x, y, z])?)
            }
        })
    }
}

/// What a node is handed when it evaluates.
///
/// Inputs are lent, not given: `input` borrows, and `take_input` moves only
/// when this node is the value's last outstanding use (otherwise it copies).
pub struct EvalCtx<'a> {
    pub(crate) gpu: &'a GpuContext,
    pub(crate) pool: &'a mut FieldPool,
    pub(crate) pipelines: &'a mut PipelineCache,
    pub(crate) dims: FieldDims,
    /// Metres along the domain's longest axis.
    pub(crate) domain_size: f64,
    pub(crate) node: NodeId,
    /// For each input index, the output socket feeding it, if any.
    pub(crate) sources: Vec<Option<SocketId>>,
    /// Tracks which indices `take_input` has already removed, so `input` can
    /// report a distinct error for "taken" versus "never connected".
    pub(crate) taken: Vec<bool>,
    /// Every value produced so far this evaluation, keyed by output socket.
    pub(crate) produced: &'a mut HashMap<SocketId, Value>,
    /// Input uses of each produced value that have not finished yet.
    pub(crate) remaining: &'a mut HashMap<SocketId, u32>,
    pub(crate) stats: &'a mut EvalStats,
    /// Persistent state. Only a node whose `stateful()` is true may touch it.
    pub(crate) state: &'a mut StateStore,
    pub(crate) stateful: bool,
    pub(crate) time: Time,
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
        let i = index as usize;
        if self.taken.get(i).copied().unwrap_or(false) {
            return Err(NodeError::InputAlreadyTaken {
                node: self.node,
                index,
            });
        }
        self.sources
            .get(i)
            .copied()
            .flatten()
            .and_then(|src| self.produced.get(&src))
            .ok_or(NodeError::MissingInput {
                node: self.node,
                index,
            })
    }

    /// Take ownership of input `index`.
    ///
    /// If this is the value's last outstanding use, it is moved, with no copy.
    /// If other inputs still need it, this returns a GPU copy and counts it in
    /// `EvalStats::copies`. So a node that only reads an input should call
    /// `input` instead.
    ///
    /// Trap: calling `input(index)` after this does not report `MissingInput`
    /// (which would look like a wiring mistake in the graph). It reports
    /// `InputAlreadyTaken`, a bug in this node's own `eval`, since it read the
    /// same input twice.
    pub fn take_input(&mut self, index: u32) -> Result<Value, NodeError> {
        let i = index as usize;
        let node = self.node;
        let missing = || NodeError::MissingInput { node, index };
        if self.taken.get(i).copied().unwrap_or(false) {
            return Err(NodeError::InputAlreadyTaken { node, index });
        }
        let src = self.sources.get(i).copied().flatten().ok_or_else(missing)?;

        let left = self.remaining.get(&src).copied().unwrap_or(0);
        let value = if left <= 1 {
            self.remaining.remove(&src);
            self.produced.remove(&src).ok_or_else(missing)?
        } else {
            let shared = self.produced.get(&src).ok_or_else(missing)?;
            let copy = shared.duplicate(self.gpu, self.pool)?;
            if !matches!(copy, Value::Scalar(_)) {
                self.stats.copies += 1;
            }
            self.remaining.insert(src, left - 1);
            copy
        };

        if let Some(slot) = self.taken.get_mut(i) {
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

    /// When this evaluation is happening.
    pub fn time(&self) -> Time {
        self.time
    }

    /// Metres per voxel. Voxels are cubic, and the document's `domain_size`
    /// spans the domain's longest axis.
    pub fn voxel_size(&self) -> f32 {
        let longest = self.dims.x.max(self.dims.y).max(self.dims.z).max(1);
        (self.domain_size / longest as f64) as f32
    }

    /// Run GPU work that also needs the field pool, such as a solver that
    /// acquires scratch fields between kernels.
    pub fn with_gpu_pool<T>(
        &mut self,
        f: impl FnOnce(&GpuContext, &mut PipelineCache, &mut FieldPool) -> Result<T, GpuError>,
    ) -> Result<T, NodeError> {
        Ok(f(self.gpu, self.pipelines, self.pool)?)
    }

    /// Take this node's state `slot` out of the store, or `None` on the first
    /// step or after a reset. Put it back with `put_state` before returning.
    ///
    /// A stored field whose dims no longer match the domain is released and
    /// reported as `StateShape`. The timeline answers that with a reset.
    pub fn take_state(&mut self, slot: &'static str) -> Result<Option<Value>, NodeError> {
        if !self.stateful {
            return Err(NodeError::NotStateful { node: self.node });
        }
        let Some(value) = self.state.take(self.node, slot) else {
            return Ok(None);
        };
        let fits = match &value {
            Value::Field(f) => f.dims() == self.dims,
            Value::VectorField(v) => v.cells() == self.dims,
            Value::Scalar(_) => true,
        };
        if !fits {
            value.release_to(self.pool);
            return Err(NodeError::StateShape {
                node: self.node,
                slot,
            });
        }
        Ok(Some(value))
    }

    /// Store `value` in this node's state `slot`, releasing anything it replaces.
    pub fn put_state(&mut self, slot: &'static str, value: Value) -> Result<(), NodeError> {
        if !self.stateful {
            return Err(NodeError::NotStateful { node: self.node });
        }
        self.state.put(self.node, slot, value, self.pool);
        Ok(())
    }

    /// Return a value this node is finished with to the pool.
    pub fn release(&mut self, value: Value) {
        value.release_to(self.pool);
    }

    /// A pooled GPU copy of `field`.
    pub fn duplicate(&mut self, field: &Field) -> Result<Field, NodeError> {
        Ok(self.pool.duplicate(self.gpu, field)?)
    }
}

/// One unit of computation in a graph.
pub trait Node: Send + Sync {
    /// Stable identifier used in the `.elements` document, e.g. `"core.noise_field"`.
    fn kind(&self) -> &'static str;

    /// This node's input and output types.
    fn sockets(&self) -> SocketSpec;

    /// Whether this node keeps state between frames. Only stateful nodes may
    /// call `EvalCtx::take_state` / `put_state`, and a graph containing one is
    /// stepped frame by frame by a `Timeline`.
    fn stateful(&self) -> bool {
        false
    }

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
