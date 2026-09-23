//! Per-node state that survives from one frame to the next.

use std::collections::BTreeMap;

use crate::gpu::{FieldPool, GpuContext, GpuError};

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

    /// The value in `node`'s `slot`, if any. Read-only: benchmarks and
    /// tests read state between frames through this.
    pub fn get(&self, node: NodeId, slot: &'static str) -> Option<&Value> {
        self.slots.get(&(node, slot))
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

    /// Copy every slot. The store itself is unchanged.
    pub fn snapshot(&self, gpu: &GpuContext, pool: &mut FieldPool) -> Result<Snapshot, GpuError> {
        let mut slots = Vec::with_capacity(self.slots.len());
        let mut bytes = 0;
        for (key, value) in &self.slots {
            match value.duplicate(gpu, pool) {
                Ok(copy) => {
                    bytes += copy.gpu_bytes();
                    slots.push((*key, copy));
                }
                Err(e) => {
                    for (_, partial) in slots {
                        partial.release_to(pool);
                    }
                    return Err(e);
                }
            }
        }
        Ok(Snapshot { slots, bytes })
    }

    /// Replace the store's contents with a copy of `snapshot`.
    ///
    /// Copies rather than moves: the timeline keeps the snapshot cached,
    /// because the same frame may be scrubbed to again. On error the store is
    /// left partially restored, and the caller must clear it.
    pub fn restore(
        &mut self,
        snapshot: &Snapshot,
        gpu: &GpuContext,
        pool: &mut FieldPool,
    ) -> Result<(), GpuError> {
        self.clear(pool);
        for (key, value) in &snapshot.slots {
            let copy = value.duplicate(gpu, pool)?;
            self.slots.insert(*key, copy);
        }
        Ok(())
    }
}

/// A GPU copy of every slot in a `StateStore` at one moment.
pub struct Snapshot {
    slots: Vec<((NodeId, &'static str), Value)>,
    bytes: u64,
}

impl Snapshot {
    /// GPU memory this snapshot holds.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Return this snapshot's textures to `pool`.
    pub fn release_to(self, pool: &mut FieldPool) {
        for (_, value) in self.slots {
            value.release_to(pool);
        }
    }
}
