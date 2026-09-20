//! Maps `.elements` node-kind strings to node constructors.

use std::collections::HashMap;

use super::document::DocError;
use super::node::Node;

/// Builds a node from its serialized parameters.
pub type NodeCtor = fn(&serde_json::Value) -> Result<Box<dyn Node>, DocError>;

/// The set of node kinds a document may reference.
///
/// Core owns the graph; products register their own kinds on top of the
/// built-ins, which is the seam that keeps solvers out of `elements-core`.
#[derive(Default)]
pub struct NodeRegistry {
    ctors: HashMap<&'static str, NodeCtor>,
}

impl NodeRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// The node kinds every Elements build ships.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        crate::nodes::register_builtins(&mut registry);
        registry
    }

    pub fn register(&mut self, kind: &'static str, ctor: NodeCtor) {
        self.ctors.insert(kind, ctor);
    }

    pub fn build(&self, kind: &str, params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
        let ctor = self
            .ctors
            .get(kind)
            .ok_or_else(|| DocError::UnknownKind(kind.to_owned()))?;
        ctor(params)
    }

    pub fn kinds(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.ctors.keys().copied()
    }
}
