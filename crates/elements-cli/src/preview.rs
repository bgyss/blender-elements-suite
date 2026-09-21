//! PNG slice previews from the command line.

use std::path::Path;

use anyhow::Context;

use crate::bake::evaluate_document;

/// Parse a `LO,HI` display range.
pub fn parse_range(spec: &str) -> anyhow::Result<(f32, f32)> {
    let (lo, hi) = spec
        .split_once(',')
        .context("range must be written as LO,HI")?;
    Ok((
        lo.trim().parse().context("range low")?,
        hi.trim().parse().context("range high")?,
    ))
}

pub fn render_preview(
    graph: &Path,
    out: &Path,
    slice_z: Option<u32>,
    range: (f32, f32),
) -> anyhow::Result<()> {
    let (values, dims) = evaluate_document(graph)?;
    let z = slice_z.unwrap_or(dims[2] / 2);
    elements_io::write_slice_png(out, &values, dims, z, range)
        .with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}

pub fn dump_npy(graph: &Path, out: &Path) -> anyhow::Result<()> {
    let (values, dims) = evaluate_document(graph)?;
    elements_io::write_npy(out, &values, dims)
        .with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_range;

    #[test]
    fn parse_range_accepts_a_negative_low_bound() {
        assert_eq!(parse_range("-1,1").unwrap(), (-1.0, 1.0));
    }

    #[test]
    fn parse_range_accepts_a_zero_low_bound() {
        assert_eq!(parse_range("0,1").unwrap(), (0.0, 1.0));
    }

    #[test]
    fn parse_range_rejects_a_missing_comma() {
        assert!(parse_range("1").is_err());
    }

    #[test]
    fn parse_range_rejects_a_missing_high_bound() {
        assert!(parse_range("1,").is_err());
    }

    #[test]
    fn parse_range_rejects_a_missing_low_bound() {
        assert!(parse_range(",1").is_err());
    }

    #[test]
    fn parse_range_rejects_an_empty_string() {
        assert!(parse_range("").is_err());
    }

    #[test]
    fn parse_range_rejects_non_numeric_bounds() {
        assert!(parse_range("a,b").is_err());
    }
}
