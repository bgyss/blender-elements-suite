//! Per-node state that survives from one frame to the next.

use std::collections::BTreeMap;

use crate::gpu::FieldPool;

use super::node::Value;
use super::socket::NodeId;

/// Fields and scalars that outlive a single `Graph::eval_frame`.
///
/// Keyed by `(NodeId, slot)`. A `BTreeMap` rather than a `HashMap` so iteration
/// order, and therefore snapshot order, is deterministic.
///
/// Nodes reach their own slots through `EvalCtx::take_state` / `put_state`.
/// A node takes a value out, works on it, and puts it back, which keeps the
/// borrow checker out of the way while it also acquires pool fields.
#[derive(Default)]
pub struct StateStore {
    pub(crate) slots: BTreeMap<(NodeId, &'static str), Value>,
}

impl StateStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of occupied slots.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Empty every slot, returning its textures to `pool`.
    pub fn clear(&mut self, pool: &mut FieldPool) {
        for (_, value) in std::mem::take(&mut self.slots) {
            value.release_to(pool);
        }
    }

    pub(crate) fn take(&mut self, node: NodeId, slot: &'static str) -> Option<Value> {
        self.slots.remove(&(node, slot))
    }

    pub(crate) fn put(
        &mut self,
        node: NodeId,
        slot: &'static str,
        value: Value,
        pool: &mut FieldPool,
    ) {
        if let Some(old) = self.slots.insert((node, slot), value) {
            old.release_to(pool);
        }
    }
}
