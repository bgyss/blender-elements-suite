//! Single-slice PNG previews: they turn a broken shader into an image diff.

use std::path::Path;

use crate::IoError;

/// Write one z-slice as 8-bit greyscale, mapping `range` linearly onto 0..=255.
///
/// Values outside `range` are clamped rather than wrapped, so a blown-up
/// simulation reads as saturated white instead of noise. Non-finite values
/// (`NaN`, `+inf`, and `-inf`) bypass the clamp entirely and saturate to
/// white (255): `NaN` and `+inf` clamp to white already, and `-inf` is
/// mapped to white too rather than black, because these previews exist so a
/// diverged simulation is visually obvious, and a single, consistent
/// "non-finite means saturated white" rule is easier to trust at a glance
/// than a rule that renders some kinds of blow-up as black.
pub fn write_slice_png(
    path: &Path,
    values: &[f32],
    dims: [u32; 3],
    z: u32,
    range: (f32, f32),
) -> Result<(), IoError> {
    if z >= dims[2] {
        return Err(IoError::BadSlice { z, depth: dims[2] });
    }
    let expected = dims[0] as usize * dims[1] as usize * dims[2] as usize;
    if values.len() != expected {
        return Err(IoError::LengthMismatch {
            expected,
            got: values.len(),
        });
    }

    let (lo, hi) = range;
    let span = if (hi - lo).abs() < f32::EPSILON {
        1.0
    } else {
        hi - lo
    };

    let slice_len = dims[0] as usize * dims[1] as usize;
    let start = z as usize * slice_len;
    let pixels: Vec<u8> = values[start..start + slice_len]
        .iter()
        .map(|v| {
            if !v.is_finite() {
                255
            } else {
                (((v - lo) / span).clamp(0.0, 1.0) * 255.0).round() as u8
            }
        })
        .collect();

    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), dims[0], dims[1]);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder
        .write_header()
        .map_err(|e| IoError::Png(e.to_string()))?;
    writer
        .write_image_data(&pixels)
        .map_err(|e| IoError::Png(e.to_string()))
}
