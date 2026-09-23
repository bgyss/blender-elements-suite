//! The node graph: connections, topological evaluation, and cycle detection.

mod document;
mod node;
mod registry;
mod socket;
mod state;
mod time;
mod timeline;

pub use document::{DocEdge, DocError, DocNode, Document, ELEMENTS_DOC_VERSION};
pub use node::{EvalCtx, EvalStats, Node, NodeError, Value};
pub use registry::{NodeCtor, NodeRegistry};
pub use socket::{NodeId, SocketId, SocketSpec, SocketType};
pub use state::{Snapshot, StateStore};
pub use time::{DEFAULT_FPS, DEFAULT_START_FRAME, Time};
pub use timeline::{DEFAULT_CACHE_BUDGET_MB, Timeline, TimelineConfig};

use std::collections::{HashMap, HashSet};

use crate::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};

/// Metres spanned by the grid's longest axis when a document does not say.
/// Matches Blender's default cube.
pub const DEFAULT_DOMAIN_SIZE: f64 = 2.0;

/// A directed acyclic graph of nodes with one designated output.
///
/// Each input socket accepts exactly one edge (`connect` enforces this). An
/// output socket may feed any number of inputs: the evaluator lends the value
/// to each consumer, moves it to the last, and copies it for any earlier
/// consumer that takes ownership. See `EvalCtx::take_input`.
pub struct Graph {
    nodes: Vec<Box<dyn Node>>,
    /// Maps a destination input socket to the source output socket feeding it.
    edges: HashMap<SocketId, SocketId>,
    output: Option<NodeId>,
    /// Metres along the domain's longest axis. See `EvalCtx::voxel_size`.
    domain_size: f64,
}

impl Default for Graph {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            edges: HashMap::new(),
            output: None,
            domain_size: DEFAULT_DOMAIN_SIZE,
        }
    }
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

/// The result of evaluating one frame.
#[derive(Debug)]
pub struct Evaluated {
    /// The output node's first output. The caller owns it and should return it
    /// to the pool with `Value::release_to` when finished.
    pub value: Value,
    pub stats: EvalStats,
}

/// Everything one `eval_frame` call threads through its nodes.
struct Run<'a> {
    gpu: &'a GpuContext,
    pool: &'a mut FieldPool,
    pipelines: &'a mut PipelineCache,
    state: &'a mut StateStore,
    time: Time,
    dims: FieldDims,
    domain_size: f64,
    produced: HashMap<SocketId, Value>,
    remaining: HashMap<SocketId, u32>,
    stats: EvalStats,
}

impl Run<'_> {
    /// One use of `src` finished without taking ownership. If it was the last
    /// use, the value goes back to the pool.
    fn consume(&mut self, src: SocketId) {
        let Some(left) = self.remaining.get_mut(&src) else {
            return;
        };
        *left = left.saturating_sub(1);
        if *left == 0 {
            self.remaining.remove(&src);
            if let Some(value) = self.produced.remove(&src) {
                value.release_to(self.pool);
            }
        }
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
    /// Rejects type mismatches, nonexistent sockets, and a second edge into an
    /// input that already has one.
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

        if self.edges.contains_key(&to) {
            return Err(NodeError::InputAlreadyConnected {
                node: to.node,
                index: to.index,
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

    /// Set how many metres the domain's longest axis spans.
    pub fn set_domain_size(&mut self, metres: f64) {
        self.domain_size = metres;
    }

    pub fn domain_size(&self) -> f64 {
        self.domain_size
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

    /// Whether any node in this graph keeps state between frames.
    pub fn is_stateful(&self) -> bool {
        self.nodes.iter().any(|node| node.stateful())
    }

    /// Evaluate one frame with no persistent state: the first frame, at the default rate.
    ///
    /// Stateful nodes see an empty store, which is discarded afterwards, so
    /// repeated calls never accumulate. Anything that needs frames to follow
    /// one another goes through `eval_frame`, normally via a `Timeline`.
    pub fn eval(
        &self,
        gpu: &GpuContext,
        pool: &mut FieldPool,
        pipelines: &mut PipelineCache,
        dims: FieldDims,
    ) -> Result<Value, NodeError> {
        let mut scratch = StateStore::new();
        let time = Time::at(DEFAULT_START_FRAME, DEFAULT_START_FRAME, DEFAULT_FPS);
        let result = self.eval_frame(gpu, pool, pipelines, &mut scratch, time, dims);
        scratch.clear(pool);
        result.map(|evaluated| evaluated.value)
    }

    /// Evaluate the output node and everything it depends on, as frame `time`,
    /// reading and writing persistent state in `state`.
    ///
    /// Every field produced along the way is back in `pool` when this returns,
    /// except the result, whether evaluation succeeds or fails.
    pub fn eval_frame(
        &self,
        gpu: &GpuContext,
        pool: &mut FieldPool,
        pipelines: &mut PipelineCache,
        state: &mut StateStore,
        time: Time,
        dims: FieldDims,
    ) -> Result<Evaluated, NodeError> {
        let output = self.output.ok_or(NodeError::NoOutput)?;
        let order = self.evaluation_order()?;

        let mut run = Run {
            gpu,
            pool,
            pipelines,
            state,
            time,
            dims,
            domain_size: self.domain_size,
            produced: HashMap::new(),
            remaining: HashMap::new(),
            stats: EvalStats::default(),
        };
        let result = self.run(&mut run, &order, output);
        for (_, value) in run.produced.drain() {
            value.release_to(run.pool);
        }
        Ok(Evaluated {
            value: result?,
            stats: run.stats,
        })
    }

    fn run(&self, run: &mut Run<'_>, order: &[NodeId], output: NodeId) -> Result<Value, NodeError> {
        // Count every input use of every output, among the nodes that will run.
        let needed: HashSet<NodeId> = order.iter().copied().collect();
        for (to, from) in &self.edges {
            if needed.contains(&to.node) {
                *run.remaining.entry(*from).or_insert(0) += 1;
            }
        }

        let result_socket = SocketId {
            node: output,
            index: 0,
        };

        for &id in order {
            let node = self.node(id)?;
            let spec = node.sockets();
            let sources: Vec<Option<SocketId>> = (0..spec.inputs.len() as u32)
                .map(|index| self.edges.get(&SocketId { node: id, index }).copied())
                .collect();

            let mut ctx = EvalCtx {
                gpu: run.gpu,
                pool: &mut *run.pool,
                pipelines: &mut *run.pipelines,
                dims: run.dims,
                domain_size: run.domain_size,
                node: id,
                sources: sources.clone(),
                taken: vec![false; sources.len()],
                produced: &mut run.produced,
                remaining: &mut run.remaining,
                result: result_socket,
                stats: &mut run.stats,
                state: &mut *run.state,
                stateful: node.stateful(),
                time: run.time,
            };
            let outputs = node
                .eval(&mut ctx)
                .map_err(|err| err.with_evaluating_node(id))?;
            let taken = std::mem::take(&mut ctx.taken);

            // Inputs this node only borrowed are finished with now.
            for (index, src) in sources.iter().enumerate() {
                if let Some(src) = src
                    && !taken[index]
                {
                    run.consume(*src);
                }
            }

            for (index, value) in outputs.into_iter().enumerate() {
                let socket = SocketId {
                    node: id,
                    index: index as u32,
                };
                let wanted = node::is_wanted(result_socket, &run.remaining, socket);
                if wanted {
                    run.produced.insert(socket, value);
                } else {
                    value.release_to(run.pool);
                }
            }
        }

        run.produced
            .remove(&result_socket)
            .ok_or(NodeError::UnknownNode(output))
    }
}
