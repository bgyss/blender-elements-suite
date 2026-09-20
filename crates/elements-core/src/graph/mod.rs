//! The node graph: connections, topological evaluation, and cycle detection.

mod document;
mod node;
mod registry;
mod socket;

pub use document::{DocEdge, DocError, DocNode, Document, ELEMENTS_DOC_VERSION};
pub use node::{EvalCtx, Node, NodeError, Value};
pub use registry::{NodeCtor, NodeRegistry};
pub use socket::{NodeId, SocketId, SocketSpec, SocketType};

use std::collections::HashMap;

use crate::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};

/// A directed acyclic graph of nodes with one designated output.
///
/// Each output socket may feed at most one input socket (`connect` enforces
/// this). `Value::Field` wraps a GPU texture that is deliberately not
/// `Clone`, so `eval` cannot hand the same produced value to two consumers —
/// it moves each value out of the producer's slot into its single consumer.
/// Allowing a second connection would mean the second consumer silently gets
/// `None` and fails with a confusing `MissingInput` instead of a clear
/// rejection at graph-construction time.
#[derive(Default)]
pub struct Graph {
    nodes: Vec<Box<dyn Node>>,
    /// Maps a destination input socket to the source output socket feeding it.
    edges: HashMap<SocketId, SocketId>,
    output: Option<NodeId>,
}

// `Node` trait objects don't implement `Debug`, so this can't be derived.
// Kept intentionally shallow — it exists so `Result<(Graph, _), _>` can be
// unwrapped/panicked on in tests, not as a graph-inspection tool.
impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("node_count", &self.nodes.len())
            .field("edge_count", &self.edges.len())
            .field("output", &self.output)
            .finish()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, node: Box<dyn Node>) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    pub fn node(&self, id: NodeId) -> Result<&dyn Node, NodeError> {
        self.nodes
            .get(id.0 as usize)
            .map(|b| b.as_ref())
            .ok_or(NodeError::UnknownNode(id))
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        (0..self.nodes.len() as u32).map(NodeId)
    }

    /// Connect an output socket to an input socket, validating both ends.
    ///
    /// Rejects type mismatches, nonexistent sockets, and any attempt to feed a
    /// second input from an output that already has a consumer.
    pub fn connect(&mut self, from: SocketId, to: SocketId) -> Result<(), NodeError> {
        let from_spec = self.node(from.node)?.sockets();
        let to_spec = self.node(to.node)?.sockets();

        let out_ty =
            from_spec
                .outputs
                .get(from.index as usize)
                .copied()
                .ok_or(NodeError::MissingInput {
                    node: from.node,
                    index: from.index,
                })?;
        let in_ty =
            to_spec
                .inputs
                .get(to.index as usize)
                .copied()
                .ok_or(NodeError::MissingInput {
                    node: to.node,
                    index: to.index,
                })?;

        if out_ty != in_ty {
            return Err(NodeError::TypeMismatch {
                node: to.node,
                index: to.index,
                expected: in_ty,
            });
        }

        if self.edges.values().any(|src| *src == from) {
            return Err(NodeError::AlreadyConsumed {
                node: from.node,
                index: from.index,
            });
        }

        self.edges.insert(to, from);
        Ok(())
    }

    pub fn set_output(&mut self, node: NodeId) {
        self.output = Some(node);
    }

    pub fn output(&self) -> Option<NodeId> {
        self.output
    }

    /// All nodes in dependency order. Errors if the graph contains a cycle.
    pub fn topological_order(&self) -> Result<Vec<NodeId>, NodeError> {
        #[derive(Clone, Copy, PartialEq)]
        enum Mark {
            Unvisited,
            InProgress,
            Done,
        }

        let mut marks = vec![Mark::Unvisited; self.nodes.len()];
        let mut order = Vec::with_capacity(self.nodes.len());

        // Iterative depth-first search: deep graphs must not blow the stack.
        for start in self.node_ids() {
            if marks[start.0 as usize] != Mark::Unvisited {
                continue;
            }
            let mut stack = vec![(start, false)];
            while let Some((id, children_done)) = stack.pop() {
                if children_done {
                    marks[id.0 as usize] = Mark::Done;
                    order.push(id);
                    continue;
                }
                match marks[id.0 as usize] {
                    Mark::Done => continue,
                    Mark::InProgress => return Err(NodeError::Cycle(id)),
                    Mark::Unvisited => {}
                }
                marks[id.0 as usize] = Mark::InProgress;
                stack.push((id, true));
                for source in self.sources_of(id) {
                    match marks[source.0 as usize] {
                        Mark::InProgress => return Err(NodeError::Cycle(source)),
                        Mark::Unvisited => stack.push((source, false)),
                        Mark::Done => {}
                    }
                }
            }
        }

        Ok(order)
    }

    /// The nodes feeding `id`'s inputs.
    fn sources_of(&self, id: NodeId) -> Vec<NodeId> {
        let Ok(spec) = self.node(id).map(|n| n.sockets()) else {
            return Vec::new();
        };
        (0..spec.inputs.len() as u32)
            .filter_map(|index| self.edges.get(&SocketId { node: id, index }))
            .map(|src| src.node)
            .collect()
    }

    /// Nodes the output transitively depends on, in dependency order.
    fn evaluation_order(&self) -> Result<Vec<NodeId>, NodeError> {
        let output = self.output.ok_or(NodeError::NoOutput)?;
        self.node(output)?;
        let full = self.topological_order()?;

        let mut needed = vec![false; self.nodes.len()];
        let mut stack = vec![output];
        while let Some(id) = stack.pop() {
            if std::mem::replace(&mut needed[id.0 as usize], true) {
                continue;
            }
            stack.extend(self.sources_of(id));
        }

        Ok(full
            .into_iter()
            .filter(|id| needed[id.0 as usize])
            .collect())
    }

    /// Evaluate the output node and everything it depends on.
    pub fn eval(
        &self,
        gpu: &GpuContext,
        pool: &mut FieldPool,
        pipelines: &mut PipelineCache,
        dims: FieldDims,
    ) -> Result<Value, NodeError> {
        let output = self.output.ok_or(NodeError::NoOutput)?;
        let order = self.evaluation_order()?;

        // Outputs of each evaluated node. Values are moved out as they are
        // consumed, so each slot holds `None` once its consumer has run.
        let mut produced: HashMap<NodeId, Vec<Option<Value>>> = HashMap::new();

        for id in order {
            let node = self.node(id)?;
            let spec = node.sockets();

            let mut inputs: Vec<Option<Value>> = Vec::with_capacity(spec.inputs.len());
            for index in 0..spec.inputs.len() as u32 {
                let value = match self.edges.get(&SocketId { node: id, index }) {
                    Some(src) => produced
                        .get_mut(&src.node)
                        .and_then(|outs| outs.get_mut(src.index as usize))
                        .and_then(Option::take),
                    None => None,
                };
                inputs.push(value);
            }

            let taken = vec![false; inputs.len()];
            let mut ctx = EvalCtx {
                gpu,
                pool,
                pipelines,
                dims,
                node: id,
                inputs,
                taken,
            };
            let outputs = node
                .eval(&mut ctx)
                .map_err(|err| err.with_evaluating_node(id))?;
            produced.insert(id, outputs.into_iter().map(Some).collect());
        }

        produced
            .get_mut(&output)
            .and_then(|outs| outs.first_mut())
            .and_then(Option::take)
            .ok_or(NodeError::UnknownNode(output))
    }
}
