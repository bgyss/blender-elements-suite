//! Serialization of graphs to and from the versioned `.elements` format.

use serde::{Deserialize, Serialize};

use crate::gpu::FieldDims;

use super::Graph;
use super::node::NodeError;
use super::registry::NodeRegistry;
use super::socket::{NodeId, SocketId};

/// The `.elements` schema version this build reads and writes.
///
/// Bumping this requires a migration test in `tests/document.rs`.
pub const ELEMENTS_DOC_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum DocError {
    #[error("malformed document: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported document version {0}")]
    UnsupportedVersion(u32),
    #[error("unknown node kind: {0}")]
    UnknownKind(String),
    #[error("bad parameters for node kind {kind}: {reason}")]
    BadParams { kind: String, reason: String },
    #[error(transparent)]
    Graph(#[from] NodeError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocNode {
    pub id: u32,
    pub kind: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocEdge {
    pub from_node: u32,
    pub from_index: u32,
    pub to_node: u32,
    pub to_index: u32,
}

/// A serialized node graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub version: u32,
    pub dims: [u32; 3],
    pub nodes: Vec<DocNode>,
    pub edges: Vec<DocEdge>,
    pub output: u32,
}

impl Document {
    pub fn from_json(text: &str) -> Result<Self, DocError> {
        let doc: Document = serde_json::from_str(text)?;
        // Deliberately `!=`, not `>`: this build reads exactly
        // `ELEMENTS_DOC_VERSION`. An older document may be missing fields a
        // newer reader assumes exist, so "old" is just as unsupported as
        // "new" until a migration is written.
        if doc.version != ELEMENTS_DOC_VERSION {
            return Err(DocError::UnsupportedVersion(doc.version));
        }
        Ok(doc)
    }

    pub fn to_json(&self) -> Result<String, DocError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Instantiate the graph this document describes.
    ///
    /// Node ids are positional: the node with `id` N must be the Nth entry in
    /// `nodes`. That keeps `NodeId` allocation inside `Graph::add_node` and the
    /// document in agreement without a translation table.
    pub fn into_graph(self, registry: &NodeRegistry) -> Result<(Graph, FieldDims), DocError> {
        let mut graph = Graph::new();

        for (position, doc_node) in self.nodes.iter().enumerate() {
            if doc_node.id as usize != position {
                return Err(DocError::BadParams {
                    kind: doc_node.kind.clone(),
                    reason: format!(
                        "node id {} is at position {position}; ids must be positional",
                        doc_node.id
                    ),
                });
            }
            let node = registry.build(&doc_node.kind, &doc_node.params)?;
            graph.add_node(node);
        }

        if self.output as usize >= self.nodes.len() {
            return Err(DocError::BadParams {
                kind: "document".to_string(),
                reason: format!(
                    "output node {} is out of range; the document has {} node(s)",
                    self.output,
                    self.nodes.len()
                ),
            });
        }

        for edge in &self.edges {
            graph.connect(
                SocketId {
                    node: NodeId(edge.from_node),
                    index: edge.from_index,
                },
                SocketId {
                    node: NodeId(edge.to_node),
                    index: edge.to_index,
                },
            )?;
        }

        graph.set_output(NodeId(self.output));

        Ok((
            graph,
            FieldDims::new(self.dims[0], self.dims[1], self.dims[2]),
        ))
    }
}
