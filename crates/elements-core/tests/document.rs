use elements_core::gpu::{FieldFormat, FieldPool, GpuContext, PipelineCache, fill_constant};
use elements_core::graph::{
    DocEdge, DocError, DocNode, Document, ELEMENTS_DOC_VERSION, EvalCtx, Node, NodeError,
    NodeRegistry, SocketSpec, SocketType, TimelineConfig, Value,
};

const MINIMAL: &str = include_str!("fixtures/v1_minimal.elements");

#[test]
fn parses_the_v1_fixture() {
    let doc = Document::from_json(MINIMAL).unwrap();
    assert_eq!(doc.version, ELEMENTS_DOC_VERSION);
    assert_eq!(doc.dims, [8, 8, 8]);
    assert_eq!(doc.nodes.len(), 2);
    assert_eq!(doc.edges.len(), 1);
    assert_eq!(doc.output, 1);
}

#[test]
fn round_trips_without_loss() {
    let doc = Document::from_json(MINIMAL).unwrap();
    let json = doc.to_json().unwrap();
    let again = Document::from_json(&json).unwrap();

    assert_eq!(again.version, doc.version);
    assert_eq!(again.dims, doc.dims);
    assert_eq!(again.output, doc.output);
    assert_eq!(again.nodes.len(), doc.nodes.len());
    for (a, b) in again.nodes.iter().zip(doc.nodes.iter()) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.kind, b.kind);
        assert_eq!(a.params, b.params);
    }
    assert_eq!(again.edges, doc.edges);
}

#[test]
fn rejects_a_future_version() {
    let future = MINIMAL.replace("\"version\": 1", "\"version\": 999");
    match Document::from_json(&future) {
        Err(DocError::UnsupportedVersion(999)) => {}
        other => panic!("expected UnsupportedVersion(999), got {other:?}"),
    }
}

#[test]
fn rejects_an_old_version() {
    let old = MINIMAL.replace("\"version\": 1", "\"version\": 0");
    match Document::from_json(&old) {
        Err(DocError::UnsupportedVersion(0)) => {}
        other => panic!("expected UnsupportedVersion(0), got {other:?}"),
    }
}

#[test]
fn rejects_non_positional_node_ids() {
    let shuffled = MINIMAL.replace("\"id\": 0", "\"id\": 5");
    let doc = Document::from_json(&shuffled).unwrap();
    let registry = NodeRegistry::with_builtins();
    match doc.into_graph(&registry) {
        Err(DocError::BadParams { reason, .. }) => {
            assert!(reason.contains("positional"), "got {reason}")
        }
        other => panic!("expected BadParams, got {other:?}"),
    }
}

#[test]
fn rejects_an_unknown_node_kind() {
    let doc =
        Document::from_json(&MINIMAL.replace("core.noise_field", "core.does_not_exist")).unwrap();
    let registry = NodeRegistry::with_builtins();
    match doc.into_graph(&registry) {
        Err(DocError::UnknownKind(k)) => assert_eq!(k, "core.does_not_exist"),
        other => panic!("expected UnknownKind, got {other:?}"),
    }
}

#[test]
fn rejects_malformed_params() {
    let doc =
        Document::from_json(&MINIMAL.replace("\"seed\": 7", "\"seed\": \"not-a-number\"")).unwrap();
    let registry = NodeRegistry::with_builtins();
    match doc.into_graph(&registry) {
        Err(DocError::BadParams { kind, .. }) => assert_eq!(kind, "core.noise_field"),
        other => panic!("expected BadParams, got {other:?}"),
    }
}

#[test]
fn builds_a_graph_with_the_declared_output() {
    let doc = Document::from_json(MINIMAL).unwrap();
    let registry = NodeRegistry::with_builtins();
    let (graph, dims) = doc.into_graph(&registry).unwrap();

    assert_eq!(dims.x, 8);
    assert_eq!(graph.output().unwrap().0, 1);
    assert_eq!(graph.node_count(), 2);
}

/// The migration test `ELEMENTS_DOC_VERSION` requires: a version-1 document
/// loads, and gets the time settings version 1 implied.
#[test]
fn a_version_1_document_migrates_with_time_defaults() {
    let doc = Document::from_json(MINIMAL).unwrap();
    assert_eq!(doc.version, ELEMENTS_DOC_VERSION);
    assert_eq!(doc.fps, 24.0);
    assert_eq!(doc.start_frame, 1);
    assert_eq!(doc.cache_budget_mb, 2048);
}

#[test]
fn version_2_time_fields_reach_the_timeline_config() {
    let v2 = MINIMAL.replace(
        "\"version\": 1,",
        "\"version\": 2, \"fps\": 30.0, \"start_frame\": 1001, \"cache_budget_mb\": 64,",
    );
    let doc = Document::from_json(&v2).unwrap();
    assert_eq!(
        doc.timeline_config(),
        TimelineConfig {
            fps: 30.0,
            start_frame: 1001,
            cache_budget_bytes: 64 * 1024 * 1024,
        }
    );
}

#[test]
fn rejects_a_non_positive_fps() {
    for fps in ["0.0", "-24.0"] {
        let bad = MINIMAL.replace(
            "\"version\": 1,",
            &format!("\"version\": 2, \"fps\": {fps},"),
        );
        match Document::from_json(&bad) {
            Err(DocError::BadParams { reason, .. }) => assert!(reason.contains("fps"), "{reason}"),
            other => panic!("fps {fps}: expected BadParams, got {other:?}"),
        }
    }
}

/// A node with a single scalar output, used to build a document where its
/// single output feeds both inputs of one consumer.
struct Producer;

impl Node for Producer {
    fn kind(&self) -> &'static str {
        "test.producer"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Scalar],
        }
    }
    fn eval(&self, _ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        Ok(vec![Value::Scalar(1.0)])
    }
}

/// A node with two scalar inputs, used as a consumer whose two inputs are
/// both fed by one producer's single output.
struct Consumer;

impl Node for Consumer {
    fn kind(&self) -> &'static str {
        "test.consumer"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Scalar, SocketType::Scalar],
            outputs: vec![],
        }
    }
    fn eval(&self, _ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        Ok(vec![])
    }
}

fn test_registry() -> NodeRegistry {
    let mut registry = NodeRegistry::new();
    registry.register("test.producer", |_params| Ok(Box::new(Producer)));
    registry.register("test.consumer", |_params| Ok(Box::new(Consumer)));
    registry
}

#[test]
fn accepts_a_document_wiring_one_output_to_two_inputs() {
    let doc = Document {
        version: ELEMENTS_DOC_VERSION,
        dims: [8, 8, 8],
        fps: 24.0,
        start_frame: 1,
        cache_budget_mb: 2048,
        domain_size: elements_core::graph::DEFAULT_DOMAIN_SIZE,
        nodes: vec![
            DocNode {
                id: 0,
                kind: "test.producer".to_string(),
                params: serde_json::Value::Null,
            },
            DocNode {
                id: 1,
                kind: "test.consumer".to_string(),
                params: serde_json::Value::Null,
            },
        ],
        edges: vec![
            DocEdge {
                from_node: 0,
                from_index: 0,
                to_node: 1,
                to_index: 0,
            },
            DocEdge {
                from_node: 0,
                from_index: 0,
                to_node: 1,
                to_index: 1,
            },
        ],
        output: 1,
    };

    let registry = test_registry();
    assert!(
        doc.into_graph(&registry).is_ok(),
        "one output may now feed several inputs"
    );
}

#[test]
fn rejects_an_out_of_range_output() {
    let doc = Document {
        version: ELEMENTS_DOC_VERSION,
        dims: [8, 8, 8],
        fps: 24.0,
        start_frame: 1,
        cache_budget_mb: 2048,
        domain_size: elements_core::graph::DEFAULT_DOMAIN_SIZE,
        nodes: vec![DocNode {
            id: 0,
            kind: "test.producer".to_string(),
            params: serde_json::Value::Null,
        }],
        edges: vec![],
        output: 5,
    };

    let registry = test_registry();
    match doc.into_graph(&registry) {
        Err(DocError::BadParams { reason, .. }) => {
            assert!(reason.contains('5'), "got {reason}");
            assert!(reason.contains('1'), "got {reason}");
        }
        other => panic!("expected BadParams, got {other:?}"),
    }
}

/// Fills its output with `EvalCtx::voxel_size()`.
struct VoxelSizeProbe;

impl Node for VoxelSizeProbe {
    fn kind(&self) -> &'static str {
        "test.voxel_size_probe"
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Field],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let field = ctx.acquire_uninit(FieldFormat::R32Float)?;
        let size = ctx.voxel_size();
        ctx.with_gpu(|gpu, cache| fill_constant(gpu, cache, &field, size))?;
        Ok(vec![Value::Field(field)])
    }
}

fn probe_doc(version_and_size: &str) -> String {
    format!(
        r#"{{
          {version_and_size}
          "dims": [8, 4, 2],
          "nodes": [
            {{ "id": 0, "kind": "test.voxel_size_probe", "params": {{}} }},
            {{ "id": 1, "kind": "core.output", "params": {{}} }}
          ],
          "edges": [{{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }}],
          "output": 1
        }}"#
    )
}

#[test]
fn a_version_2_document_migrates_with_the_default_domain_size() {
    let doc = Document::from_json(&probe_doc(r#""version": 2,"#)).unwrap();
    assert_eq!(doc.version, ELEMENTS_DOC_VERSION);
    assert_eq!(doc.version, 3);
    assert_eq!(doc.domain_size, 2.0);
}

#[test]
fn domain_size_reaches_nodes_as_metres_per_voxel_along_the_longest_axis() {
    let mut registry = NodeRegistry::with_builtins();
    registry.register("test.voxel_size_probe", |_| {
        Ok(Box::new(VoxelSizeProbe) as Box<dyn Node>)
    });
    let (graph, dims) = Document::from_json(&probe_doc(r#""version": 3, "domain_size": 4.0,"#))
        .unwrap()
        .into_graph(&registry)
        .unwrap();
    assert_eq!(graph.domain_size(), 4.0);

    let gpu = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let value = graph.eval(&gpu, &mut pool, &mut pipelines, dims).unwrap();
    let voxels = value.as_field().unwrap().read_back(&gpu).unwrap();
    // 4 m along the longest axis, which has 8 cells.
    assert!(voxels.iter().all(|&v| v == 0.5), "got {:?}", &voxels[..4]);
}

#[test]
fn rejects_a_domain_size_that_is_not_positive() {
    for size in ["0.0", "-1.0"] {
        let text = probe_doc(&format!(r#""version": 3, "domain_size": {size},"#));
        match Document::from_json(&text) {
            Err(DocError::BadParams { .. }) => {}
            other => panic!("domain_size {size}: expected BadParams, got {other:?}"),
        }
    }
}

/// A size that is positive and finite as an f64 can still give an f32 voxel
/// size of 0 or infinity, which turns every field NaN without any error.
#[test]
fn rejects_a_domain_size_that_f32_voxels_cannot_hold() {
    for size in ["1e-300", "1e300"] {
        let text = probe_doc(&format!(r#""version": 3, "domain_size": {size},"#));
        match Document::from_json(&text) {
            Err(DocError::BadParams { .. }) => {}
            other => panic!("domain_size {size}: expected BadParams, got {other:?}"),
        }
    }
    for size in ["1e-3", "1e5"] {
        let text = probe_doc(&format!(r#""version": 3, "domain_size": {size},"#));
        assert!(
            Document::from_json(&text).is_ok(),
            "domain_size {size} is inside the range and must load"
        );
    }
}
