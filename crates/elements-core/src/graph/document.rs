//! Serialization of graphs to and from the versioned `.elements` format.

use serde::{Deserialize, Serialize};

use crate::gpu::{FieldDims, validate_dims_fit_buffer_limit};

use super::Graph;
use super::node::NodeError;
use super::registry::NodeRegistry;
use super::socket::{NodeId, SocketId};
use super::time::{DEFAULT_FPS, DEFAULT_START_FRAME};
use super::timeline::{DEFAULT_CACHE_BUDGET_MB, TimelineConfig};

/// The `.elements` schema version this build writes.
///
/// It also reads versions 1 and 2, which predate time and physical units: see
/// `Document::from_json`.
/// Bumping this requires a migration test in `tests/document.rs`.
pub const ELEMENTS_DOC_VERSION: u32 = 3;

fn default_fps() -> f64 {
    DEFAULT_FPS
}

fn default_start_frame() -> u32 {
    DEFAULT_START_FRAME
}

fn default_cache_budget_mb() -> u32 {
    DEFAULT_CACHE_BUDGET_MB
}

/// The range of `domain_size`, in metres, that a document may ask for.
///
/// Positive and finite as an f64 is not enough: nodes see the voxel size as an
/// f32, and a size like 1e-300 or 1e300 makes it 0 or infinity, which turns
/// every field NaN with no error. A millimetre to 100 km keeps the voxel size a
/// normal f32 at any grid a device can hold.
const DOMAIN_SIZE_RANGE: std::ops::RangeInclusive<f64> = 1e-3..=1e5;

fn default_domain_size() -> f64 {
    super::DEFAULT_DOMAIN_SIZE
}

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
    /// Frames per second. Version-1 documents have none and get 24.
    #[serde(default = "default_fps")]
    pub fps: f64,
    /// The first frame of the simulation. Earlier frames produce this one.
    #[serde(default = "default_start_frame")]
    pub start_frame: u32,
    /// GPU memory the frame cache may hold, in MiB.
    #[serde(default = "default_cache_budget_mb")]
    pub cache_budget_mb: u32,
    /// Metres along the grid's longest axis. Versions 1 and 2 have none and get 2.0.
    #[serde(default = "default_domain_size")]
    pub domain_size: f64,
    pub nodes: Vec<DocNode>,
    pub edges: Vec<DocEdge>,
    pub output: u32,
}

impl Document {
    pub fn from_json(text: &str) -> Result<Self, DocError> {
        let mut doc: Document = serde_json::from_str(text)?;
        match doc.version {
            // Versions 1 and 2 predate time and physical units. Their only
            // migration is the defaults serde has already filled in above.
            1 | 2 => doc.version = ELEMENTS_DOC_VERSION,
            ELEMENTS_DOC_VERSION => {}
            // Deliberately exact, not a range: a newer document may rely on
            // fields this build would silently ignore.
            other => return Err(DocError::UnsupportedVersion(other)),
        }
        if !(doc.fps.is_finite() && doc.fps > 0.0) {
            return Err(DocError::BadParams {
                kind: "document".to_string(),
                reason: format!("fps must be a positive, finite number, got {}", doc.fps),
            });
        }
        // `contains` is false for NaN and infinities, so this is also the finite check.
        if !DOMAIN_SIZE_RANGE.contains(&doc.domain_size) {
            return Err(DocError::BadParams {
                kind: "document".to_string(),
                reason: format!(
                    "domain_size must be {} to {} metres, got {}",
                    DOMAIN_SIZE_RANGE.start(),
                    DOMAIN_SIZE_RANGE.end(),
                    doc.domain_size
                ),
            });
        }
        Ok(doc)
    }

    pub fn to_json(&self) -> Result<String, DocError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Reject a document whose implied fields would not fit this device's
    /// buffer limit.
    ///
    /// `max_texture_dimension_3d` alone is not enough: `Field::read_back`
    /// (`elements_core::gpu::field`) allocates a staging BUFFER, and
    /// `max_buffer_size` (now requested from the adapter itself, via
    /// `elements_core::gpu::required_limits`, rather than fixed at the
    /// `downlevel_defaults` 256 MiB) is still reachable well inside the
    /// texture-dimension limit — a 2047^3 domain is far under a 2048-wide
    /// texture cap but its padded R32Float readback is tens of gigabytes.
    /// Call this before any texture or the frame channel is
    /// allocated, so an oversized document is a typed, load-time
    /// `DocError::BadParams` rather than a `GpuError::Validation` from
    /// `Field::read_back` (which a caller may treat as transient) or an
    /// out-of-memory abort that `GpuContext::scoped` cannot catch at all
    /// (`scoped` only captures validation errors, not device-timeline OOM).
    ///
    /// Everything here is computed in `u64` so a document with absurd `dims`
    /// can never overflow the check meant to catch it.
    pub fn validate_for(&self, limits: &wgpu::Limits) -> Result<(), DocError> {
        let dims = FieldDims::new(self.dims[0], self.dims[1], self.dims[2]);
        validate_dims_fit_buffer_limit(dims, limits.max_buffer_size).map_err(|reason| {
            DocError::BadParams {
                kind: "document".to_string(),
                reason,
            }
        })
    }

    /// How a timeline for this document should run.
    pub fn timeline_config(&self) -> TimelineConfig {
        TimelineConfig {
            fps: self.fps,
            start_frame: self.start_frame,
            cache_budget_bytes: self.cache_budget_mb as u64 * 1024 * 1024,
        }
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

        graph.set_domain_size(self.domain_size);
        graph.set_output(NodeId(self.output));

        Ok((
            graph,
            FieldDims::new(self.dims[0], self.dims[1], self.dims[2]),
        ))
    }
}
