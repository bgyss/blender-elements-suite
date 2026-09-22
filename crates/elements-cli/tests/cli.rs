use std::process::Command;

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_elements"))
}

const NOISE_GRAPH: &str = r#"{
  "version": 1,
  "dims": [8, 8, 8],
  "nodes": [
    { "id": 0, "kind": "core.noise_field", "params": { "seed": 7, "frequency": 4.0 } },
    { "id": 1, "kind": "core.output", "params": {} }
  ],
  "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
  "output": 1
}"#;

fn write_graph(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("noise.elements");
    std::fs::write(&path, NOISE_GRAPH).unwrap();
    path
}

#[test]
fn bake_writes_one_vdb_per_frame() {
    let dir = tempfile::tempdir().unwrap();
    let graph = write_graph(dir.path());
    let out = dir.path().join("vdb");

    let status = cli()
        .args(["bake", graph.to_str().unwrap()])
        .args(["--out", out.to_str().unwrap()])
        .args(["--frames", "1-3"])
        .status()
        .unwrap();
    assert!(status.success());

    for frame in 1..=3 {
        let path = out.join(format!("density.{frame:04}.vdb"));
        assert!(path.exists(), "missing {}", path.display());
        let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
        let reader = vdb_rs::VdbReader::new(file).expect("baked file must be valid VDB");
        assert_eq!(reader.available_grids(), vec!["density".to_string()]);
    }
}

#[test]
fn bake_pads_frame_numbers_to_at_least_four_digits() {
    let dir = tempfile::tempdir().unwrap();
    let graph = write_graph(dir.path());
    let out = dir.path().join("vdb");

    let status = cli()
        .args(["bake", graph.to_str().unwrap()])
        .args(["--out", out.to_str().unwrap()])
        .args(["--frames", "9999-10001"])
        .status()
        .unwrap();
    assert!(status.success());

    for name in ["density.9999.vdb", "density.10000.vdb", "density.10001.vdb"] {
        let path = out.join(name);
        assert!(path.exists(), "missing {}", path.display());
    }
}

#[test]
fn dump_npy_matches_the_golden_field() {
    let dir = tempfile::tempdir().unwrap();
    let graph = write_graph(dir.path());
    let out = dir.path().join("actual.npy");

    let status = cli()
        .args(["dump-npy", graph.to_str().unwrap()])
        .args(["--out", out.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(status.success());

    let (actual, actual_dims) = elements_io::read_npy(&out).unwrap();
    let (expected, expected_dims) =
        elements_io::read_npy(std::path::Path::new("tests/golden/noise_8.npy")).unwrap();

    assert_eq!(actual_dims, expected_dims);
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        approx::assert_abs_diff_eq!(a, e, epsilon = 1e-3);
        assert!(a.is_finite(), "non-finite value at index {i}");
    }
}

#[test]
fn render_preview_writes_a_png_of_the_right_size() {
    let dir = tempfile::tempdir().unwrap();
    let graph = write_graph(dir.path());
    let out = dir.path().join("preview.png");

    let status = cli()
        .args(["render-preview", graph.to_str().unwrap()])
        .args(["--slice-z", "4"])
        .args(["--range", "-1,1"])
        .args(["--out", out.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(status.success());

    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&out).unwrap()));
    let reader = decoder.read_info().unwrap();
    assert_eq!(reader.info().width, 8);
    assert_eq!(reader.info().height, 8);
}

#[test]
fn a_malformed_graph_fails_with_a_nonzero_exit_and_a_message() {
    let dir = tempfile::tempdir().unwrap();
    let graph = dir.path().join("broken.elements");
    std::fs::write(&graph, "{ not json").unwrap();

    let output = cli()
        .args(["dump-npy", graph.to_str().unwrap()])
        .args(["--out", dir.path().join("x.npy").to_str().unwrap()])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("malformed document"),
        "stderr was: {stderr}"
    );
}

fn accumulate_fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/elements-cli is two levels below the root")
        .join("tests/graphs/accumulate_4.elements")
}

fn bake(out: &std::path::Path, frames: &str) {
    let status = cli()
        .args(["bake", accumulate_fixture().to_str().unwrap()])
        .args(["--out", out.to_str().unwrap()])
        .args(["--frames", frames])
        .status()
        .unwrap();
    assert!(status.success(), "bake --frames {frames} failed");
}

#[test]
fn bake_steps_a_stateful_graph_frame_by_frame() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("range");
    bake(&out, "1-3");

    let files: Vec<Vec<u8>> = (1..=3)
        .map(|f| std::fs::read(out.join(format!("density.{f:04}.vdb"))).unwrap())
        .collect();
    assert_ne!(files[0], files[1]);
    assert_ne!(files[1], files[2]);
}

/// The VDB writer is deterministic (fixed UUID, no timestamps), so equal
/// bytes mean equal fields.
#[test]
fn baking_one_frame_gives_the_true_frame_not_the_first_step() {
    let dir = tempfile::tempdir().unwrap();
    let range = dir.path().join("range");
    let single = dir.path().join("single");
    bake(&range, "1-3");
    bake(&single, "3");

    assert_eq!(
        std::fs::read(single.join("density.0003.vdb")).unwrap(),
        std::fs::read(range.join("density.0003.vdb")).unwrap()
    );
}

#[test]
fn an_unsupported_version_names_the_version() {
    let dir = tempfile::tempdir().unwrap();
    let graph = dir.path().join("future.elements");
    std::fs::write(
        &graph,
        NOISE_GRAPH.replace("\"version\": 1", "\"version\": 42"),
    )
    .unwrap();

    let output = cli()
        .args(["dump-npy", graph.to_str().unwrap()])
        .args(["--out", dir.path().join("x.npy").to_str().unwrap()])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("42"));
}
