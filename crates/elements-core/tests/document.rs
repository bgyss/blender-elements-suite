use elements_core::graph::{DocError, Document, ELEMENTS_DOC_VERSION, NodeRegistry};

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

// unignore in Task 8
#[test]
#[ignore]
fn rejects_an_unknown_node_kind() {
    let doc =
        Document::from_json(&MINIMAL.replace("core.noise_field", "core.does_not_exist")).unwrap();
    let registry = NodeRegistry::with_builtins();
    match doc.into_graph(&registry) {
        Err(DocError::UnknownKind(k)) => assert_eq!(k, "core.does_not_exist"),
        other => panic!("expected UnknownKind, got {other:?}"),
    }
}

// unignore in Task 8
#[test]
#[ignore]
fn rejects_malformed_params() {
    let doc =
        Document::from_json(&MINIMAL.replace("\"seed\": 7", "\"seed\": \"not-a-number\"")).unwrap();
    let registry = NodeRegistry::with_builtins();
    match doc.into_graph(&registry) {
        Err(DocError::BadParams { kind, .. }) => assert_eq!(kind, "core.noise_field"),
        other => panic!("expected BadParams, got {other:?}"),
    }
}

// unignore in Task 8
#[test]
#[ignore]
fn builds_a_graph_with_the_declared_output() {
    let doc = Document::from_json(MINIMAL).unwrap();
    let registry = NodeRegistry::with_builtins();
    let (graph, dims) = doc.into_graph(&registry).unwrap();

    assert_eq!(dims.x, 8);
    assert_eq!(graph.output().unwrap().0, 1);
    assert_eq!(graph.node_count(), 2);
}
