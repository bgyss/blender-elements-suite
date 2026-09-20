//! Graph evaluation shared by every subcommand, plus VDB baking.

use std::path::Path;

use anyhow::Context;
use elements_core::gpu::{FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{Document, NodeRegistry};

/// Load a `.elements` document, evaluate it, and read the result back.
///
/// Returns values in x-fastest order with the document's dimensions.
pub fn evaluate_document(path: &Path) -> anyhow::Result<(Vec<f32>, [u32; 3])> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let doc = Document::from_json(&text)?;
    let registry = NodeRegistry::with_builtins();
    let (graph, dims) = doc.into_graph(&registry)?;

    let gpu = GpuContext::new_headless().context("acquiring a GPU device")?;
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
pub fn bake(
    graph: &Path,
    out_dir: &Path,
    frames: (u32, u32),
    name: &str,
    voxel_size: f64,
) -> anyhow::Result<()> {
    let (values, dims) = evaluate_document(graph)?;
    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    for frame in frames.0..=frames.1 {
        let path = out_dir.join(format!("{name}.{frame:04}.vdb"));
        elements_io::write_float_grid(&path, name, &values, dims, voxel_size, 0.0)
            .with_context(|| format!("writing {}", path.display()))?;
    }

    Ok(())
}
