#![forbid(unsafe_code)]

//! Ember: a dense-grid smoke solver for the Elements Suite.
//!
//! Registers node kinds on top of core's built-ins. Core knows nothing about
//! smoke; this crate is the seam `NodeRegistry` exists for.

pub mod bench;
pub mod boundaries;
pub mod cfl;
pub mod collider;
pub mod emitter;
pub mod kernels;
pub mod metrics;
mod node_util;
mod params;
pub mod shape_emitter;
pub mod solver;
pub mod transform;
pub mod unions;

use elements_core::graph::NodeRegistry;

/// Add Ember's node kinds to `registry`.
pub fn register(registry: &mut NodeRegistry) {
    registry.register(emitter::KIND, emitter::build);
    registry.register(solver::KIND, solver::build);
    registry.register(shape_emitter::KIND, shape_emitter::build);
    registry.register(unions::EMITTER_UNION_KIND, unions::build_emitter_union);
    registry.register(collider::KIND, collider::build);
    registry.register(unions::COLLIDER_UNION_KIND, unions::build_collider_union);
}

/// Core's built-in node kinds plus Ember's.
pub fn registry() -> NodeRegistry {
    let mut registry = NodeRegistry::with_builtins();
    register(&mut registry);
    registry
}
