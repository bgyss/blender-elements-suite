//! Graph evaluation shared by every subcommand, plus VDB baking.

use std::path::Path;

use anyhow::Context;
use elements_core::gpu::{Axis, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{DocOutput, Document, Timeline, Value};

use crate::grids;

/// Load a `.elements` document, evaluate it, and read the result back.
///
/// Returns values in x-fastest order with the document's dimensions.
pub fn evaluate_document(path: &Path) -> anyhow::Result<(Vec<f32>, [u32; 3])> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let doc = Document::from_json(&text)?;

    let gpu = GpuContext::new_headless().context("acquiring a GPU device")?;

    // Reject a document whose implied fields would not fit this device's
    // `max_buffer_size` (requested from the adapter itself, via
    // `elements_core::gpu::required_limits`, not a fixed cap) before any GPU
    // texture is allocated. See `Document::validate_for`'s doc comment: this
    // is what turns an oversized document into a clean error instead of a
    // panic from `Field::read_back` or `Queue`'s staging-buffer allocation.
    doc.validate_for(&gpu.device().limits())
        .context("validating document dimensions against this device's limits")?;

    let registry = elements_ember::registry();
    let (graph, dims) = doc.into_graph(&registry)?;

    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();

    let value = graph.eval(&gpu, &mut pool, &mut pipelines, dims)?;
    let field = value.as_field()?;
    let values = field.read_back(&gpu)?;

    Ok((values, [dims.x, dims.y, dims.z]))
}

/// Parse an inclusive `A-B` frame range, or a single frame `N`.
pub fn parse_frames(spec: &str) -> anyhow::Result<(u32, u32)> {
    match spec.split_once('-') {
        Some((a, b)) => {
            let start: u32 = a.trim().parse().context("frame range start")?;
            let end: u32 = b.trim().parse().context("frame range end")?;
            anyhow::ensure!(start <= end, "frame range {spec} runs backwards");
            Ok((start, end))
        }
        None => {
            let n: u32 = spec.trim().parse().context("frame number")?;
            Ok((n, n))
        }
    }
}

/// Write one `.vdb` per frame into `out_dir`.
///
/// Each file is named `{name}.{frame:04}.vdb`. The frame number is
/// zero-padded to AT LEAST four digits: `{:04}` in Rust (like `%04d` in
/// ffmpeg, Houdini, and Nuke) is a *minimum* width, not a fixed one, so frame
/// `10000` renders as five digits (`10000`), not four. A consumer that lists
/// the output directory and sorts filenames lexicographically will therefore
/// place `density.10000.vdb` before `density.9999.vdb` (`'1' < '9'`).
/// Consumers MUST parse the numeric frame out of the filename and sort/compare
/// numerically rather than relying on string/lexicographic order.
///
/// Frames are produced by a timeline, so a stateful graph is simulated from
/// the document's start frame even when `frames` starts later. Only frames in
/// `frames` are written.
///
/// With an `outputs` list in the document, each frame's file holds those grids and
/// `name` is only the file stem.
pub fn bake(
    graph: &Path,
    out_dir: &Path,
    frames: (u32, u32),
    name: &str,
    voxel_size: f64,
) -> anyhow::Result<()> {
    let text =
        std::fs::read_to_string(graph).with_context(|| format!("reading {}", graph.display()))?;
    let doc = Document::from_json(&text)?;

    let gpu = GpuContext::new_headless().context("acquiring a GPU device")?;

    // See the matching check in `evaluate_document`.
    doc.validate_for(&gpu.device().limits())
        .context("validating document dimensions against this device's limits")?;

    let config = doc.timeline_config();
    let outputs = doc.outputs.clone();
    let registry = elements_ember::registry();
    let (graph, dims) = doc.into_graph(&registry)?;
    // Names are checked before any GPU evaluation and before the output directory exists.
    let names = if outputs.is_empty() {
        Vec::new()
    } else {
        grids::expanded_names(&graph, &outputs)?
    };

    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut timeline = Timeline::new(config);

    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    for frame in frames.0..=frames.1 {
        let evaluated = timeline.goto(&graph, &gpu, &mut pool, &mut pipelines, dims, frame)?;
        if let Some(warning) = timeline.take_warning() {
            eprintln!("warning: {warning}");
        }
        let path = out_dir.join(format!("{name}.{frame:04}.vdb"));
        if outputs.is_empty() {
            let values = evaluated.value.as_field()?.read_back(&gpu)?;
            evaluated.release_to(&mut pool);
            elements_io::write_float_grid(
                &path,
                name,
                &values,
                [dims.x, dims.y, dims.z],
                voxel_size,
                0.0,
            )
            .with_context(|| format!("writing {}", path.display()))?;
        } else {
            let cells = [dims.x, dims.y, dims.z];
            let (value, extras) = (evaluated.value, evaluated.extras);
            value.release_to(&mut pool);
            let grids = read_grids(&gpu, &mut pool, &outputs, extras, cells)
                .with_context(|| format!("reading frame {frame}"))?;
            debug_assert_eq!(grids.len(), names.len());
            let specs: Vec<elements_io::GridSpec<'_>> = names
                .iter()
                .zip(&grids)
                .map(|(n, v)| elements_io::GridSpec { name: n, values: v })
                .collect();
            if let Err(e) = elements_io::write_float_grids(&path, &specs, cells, voxel_size, 0.0) {
                let _ = std::fs::remove_file(&path);
                return Err(e).with_context(|| format!("writing {}", path.display()));
            }
        }
    }

    Ok(())
}

/// Read every extra back: a field as itself, a vector field as three cell-centred grids.
/// Values return to `pool` whether or not a read fails.
fn read_grids(
    gpu: &GpuContext,
    pool: &mut FieldPool,
    outputs: &[DocOutput],
    extras: Vec<Value>,
    cells: [u32; 3],
) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut grids = Vec::new();
    let mut failure = None;
    for (entry, value) in outputs.iter().zip(extras) {
        if failure.is_none() {
            match read_one(gpu, &value, cells) {
                Ok(mut read) => grids.append(&mut read),
                Err(e) => failure = Some(e.context(format!("output {:?}", entry.name))),
            }
        }
        value.release_to(pool);
    }
    match failure {
        Some(e) => Err(e),
        None => Ok(grids),
    }
}

fn read_one(gpu: &GpuContext, value: &Value, cells: [u32; 3]) -> anyhow::Result<Vec<Vec<f32>>> {
    match value {
        Value::Field(field) => Ok(vec![field.read_back(gpu)?]),
        Value::VectorField(velocity) => {
            let fx = velocity.face(Axis::X).read_back(gpu)?;
            let fy = velocity.face(Axis::Y).read_back(gpu)?;
            let fz = velocity.face(Axis::Z).read_back(gpu)?;
            Ok(grids::cell_centred_velocity(cells, [&fx, &fy, &fz])?.into())
        }
        other => anyhow::bail!("{other:?} cannot be written as a grid"),
    }
}

#[cfg(test)]
mod tests {
    use super::parse_frames;

    #[test]
    fn parse_frames_accepts_a_single_frame() {
        assert_eq!(parse_frames("5").unwrap(), (5, 5));
    }

    #[test]
    fn parse_frames_accepts_a_range() {
        assert_eq!(parse_frames("1-3").unwrap(), (1, 3));
    }

    #[test]
    fn parse_frames_accepts_a_single_point_range() {
        assert_eq!(parse_frames("5-5").unwrap(), (5, 5));
    }

    #[test]
    fn parse_frames_rejects_a_backwards_range() {
        assert!(parse_frames("3-1").is_err());
    }

    #[test]
    fn parse_frames_rejects_a_missing_range_end() {
        assert!(parse_frames("1-").is_err());
    }

    #[test]
    fn parse_frames_rejects_a_missing_range_start() {
        assert!(parse_frames("-1").is_err());
    }

    #[test]
    fn parse_frames_rejects_an_empty_string() {
        assert!(parse_frames("").is_err());
    }

    #[test]
    fn parse_frames_rejects_a_number_that_overflows_u32() {
        assert!(parse_frames("99999999999999999999").is_err());
    }
}
