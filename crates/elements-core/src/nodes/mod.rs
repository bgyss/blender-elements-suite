//! The node kinds every Elements build ships.

pub mod accumulate;
pub mod constant_field;
pub mod noise_field;
pub mod output;

pub use accumulate::Accumulate;
pub use constant_field::ConstantField;
pub use noise_field::NoiseField;
pub use output::Output;

use crate::graph::NodeRegistry;

/// Register every built-in kind on `registry`.
pub fn register_builtins(registry: &mut NodeRegistry) {
    registry.register(accumulate::KIND, accumulate::build);
    registry.register(constant_field::KIND, constant_field::build);
    registry.register(noise_field::KIND, noise_field::build);
    registry.register(output::KIND, output::build);
}
