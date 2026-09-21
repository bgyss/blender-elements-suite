use elements_core::gpu::{FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{Document, NodeRegistry};

fn eval_doc(json: &str) -> Vec<f32> {
    let doc = Document::from_json(json).unwrap();
    let registry = NodeRegistry::with_builtins();
    let (graph, dims) = doc.into_graph(&registry).unwrap();

    let gpu = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();

    let value = graph.eval(&gpu, &mut pool, &mut pipelines, dims).unwrap();
    value.as_field().unwrap().read_back(&gpu).unwrap()
}

const CONSTANT_DOC: &str = r#"{
  "version": 1,
  "dims": [4, 4, 4],
  "nodes": [
    { "id": 0, "kind": "core.constant_field", "params": { "value": 0.5 } },
    { "id": 1, "kind": "core.output", "params": {} }
  ],
  "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
  "output": 1
}"#;

const NOISE_DOC: &str = r#"{
  "version": 1,
  "dims": [8, 8, 8],
  "nodes": [
    { "id": 0, "kind": "core.noise_field", "params": { "seed": 7, "frequency": 4.0 } },
    { "id": 1, "kind": "core.output", "params": {} }
  ],
  "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
  "output": 1
}"#;

#[test]
fn constant_field_through_output() {
    let values = eval_doc(CONSTANT_DOC);
    assert_eq!(values.len(), 64);
    for v in &values {
        approx::assert_abs_diff_eq!(*v, 0.5, epsilon = 1e-3);
    }
}

#[test]
fn noise_field_through_output_is_seed_stable() {
    assert_eq!(eval_doc(NOISE_DOC), eval_doc(NOISE_DOC));
}

#[test]
fn noise_frequency_defaults_when_omitted() {
    let doc = NOISE_DOC.replace(", \"frequency\": 4.0", "");
    let values = eval_doc(&doc);
    assert_eq!(values.len(), 512);
    assert!(values.iter().all(|v| v.is_finite()));
    // A frequency of 0.0 (what a plain `#[serde(default)]` would yield)
    // samples the noise function at the same coordinate for every voxel,
    // producing a constant field. Proving the values actually vary catches
    // that regression; `is_finite` alone would not, since a frequency-0
    // field is still finite.
    let first = values[0];
    assert!(
        values.iter().any(|v| *v != first),
        "expected the default frequency to produce varying noise, got a constant field"
    );
}

#[test]
fn registry_exposes_all_three_builtins() {
    let registry = NodeRegistry::with_builtins();
    let mut kinds: Vec<_> = registry.kinds().collect();
    kinds.sort_unstable();
    assert_eq!(
        kinds,
        ["core.constant_field", "core.noise_field", "core.output"]
    );
}
