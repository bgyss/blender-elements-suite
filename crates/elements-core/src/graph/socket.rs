//! Socket identity and typing.

/// Identifies a node within one graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// Identifies one socket on one node. Whether `index` refers to an input or an
/// output is determined by how the `SocketId` is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SocketId {
    pub node: NodeId,
    pub index: u32,
}

/// The value kinds that can travel along a connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocketType {
    Field,
    Scalar,
}

/// A node's input and output signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SocketSpec {
    pub inputs: Vec<SocketType>,
    pub outputs: Vec<SocketType>,
}
