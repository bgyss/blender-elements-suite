//! Parsing and validating node parameters from untrusted documents.

use elements_core::graph::DocError;
use serde::de::DeserializeOwned;

/// A `BadParams` error for node kind `kind`.
pub(crate) fn bad(kind: &str, reason: impl Into<String>) -> DocError {
    DocError::BadParams {
        kind: kind.to_owned(),
        reason: reason.into(),
    }
}

/// Deserialize a node's parameters, naming the node kind on failure.
pub(crate) fn parse<T: DeserializeOwned>(
    kind: &str,
    params: &serde_json::Value,
) -> Result<T, DocError> {
    T::deserialize(params).map_err(|e| bad(kind, e.to_string()))
}

/// Reject a parameter holding NaN or an infinity.
///
/// JSON has no NaN, but a number too large for `f32` (for example `1e39`)
/// deserializes to infinity.
pub(crate) fn finite(kind: &str, name: &str, values: &[f32]) -> Result<(), DocError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(bad(kind, format!("{name} must be finite, got {values:?}")))
    }
}
