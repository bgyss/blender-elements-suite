#![forbid(unsafe_code)]

//! Ember: a dense-grid smoke solver for the Elements Suite.
//!
//! Registers node kinds on top of core's built-ins. Core knows nothing about
//! smoke; this crate is the seam `NodeRegistry` exists for.

pub mod emitter;
pub mod kernels;
mod params;
pub mod solver;

use elements_core::graph::NodeRegistry;

/// Add Ember's node kinds to `registry`.
pub fn register(registry: &mut NodeRegistry) {
    registry.register(emitter::KIND, emitter::build);
    registry.register(solver::KIND, solver::build);
}

/// Core's built-in node kinds plus Ember's.
pub fn registry() -> NodeRegistry {
    let mut registry = NodeRegistry::with_builtins();
    register(&mut registry);
    registry
}
