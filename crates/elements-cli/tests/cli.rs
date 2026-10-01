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

const SPHERE_GRAPH: &str = r#"{
  "version": 3,
  "dims": [8, 8, 8],
  "nodes": [
    { "id": 0, "kind": "ember.sphere_emitter",
      "params": { "center": [1.0, 1.0, 1.0], "radius": 0.5, "density_rate": 1.0 } },
    { "id": 1, "kind": "core.output", "params": {} }
  ],
  "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
  "output": 1
}"#;

#[test]
fn the_cli_knows_ember_node_kinds() {
    let dir = tempfile::tempdir().unwrap();
    let graph = dir.path().join("sphere.elements");
    std::fs::write(&graph, SPHERE_GRAPH).unwrap();
    let out = dir.path().join("sphere.npy");
    let output = cli()
        .args(["dump-npy", graph.to_str().unwrap()])
        .args(["--out", out.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(out.exists());
}

fn read_active(path: &std::path::Path, name: &str) -> Vec<((i32, i32, i32), f32)> {
    let file = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let mut reader = vdb_rs::VdbReader::new(file).unwrap();
    let grid = reader.read_grid::<f32>(name).unwrap();
    let mut v: Vec<_> = grid
        .iter()
        .map(|(c, v, _)| ((c.x as i32, c.y as i32, c.z as i32), v))
        .collect();
    v.sort_by_key(|(c, _)| *c);
    v
}

const TWO_NOISES: &str = r#"{
  "version": 4, "dims": [8, 8, 8],
  "nodes": [
    { "id": 0, "kind": "core.noise_field", "params": { "seed": 7, "frequency": 4.0 } },
    { "id": 1, "kind": "core.output", "params": {} },
    { "id": 2, "kind": "core.noise_field", "params": { "seed": 9, "frequency": 2.0 } }
  ],
  "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
  "output": 1,
  "outputs": [
    { "node": 0, "socket": 0, "name": "first" },
    { "node": 2, "socket": 0, "name": "second" }
  ]
}"#;

#[test]
fn bake_with_outputs_writes_every_named_grid_into_one_file() {
    let dir = tempfile::tempdir().unwrap();
    let graph = dir.path().join("two.elements");
    std::fs::write(&graph, TWO_NOISES).unwrap();
    let out = dir.path().join("vdb");
    assert!(
        cli()
            .args(["bake", graph.to_str().unwrap()])
            .args(["--out", out.to_str().unwrap(), "--frames", "1-2"])
            .args(["--name", "shot"])
            .status()
            .unwrap()
            .success()
    );
    // The same two noises baked alone, each through the plain single-grid path.
    let reference: Vec<_> = [(7, "4.0"), (9, "2.0")]
        .iter()
        .map(|(seed, freq)| {
            let doc = NOISE_GRAPH
                .replace("\"seed\": 7", &format!("\"seed\": {seed}"))
                .replace("4.0", freq);
            let g = dir.path().join(format!("ref{seed}.elements"));
            std::fs::write(&g, doc).unwrap();
            let o = dir.path().join(format!("ref{seed}"));
            assert!(
                cli()
                    .args(["bake", g.to_str().unwrap(), "--out", o.to_str().unwrap()])
                    .args(["--frames", "1"])
                    .status()
                    .unwrap()
                    .success()
            );
            read_active(&o.join("density.0001.vdb"), "density")
        })
        .collect();
    for frame in 1..=2 {
        let path = out.join(format!("shot.{frame:04}.vdb"));
        let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
        let mut names = vdb_rs::VdbReader::new(file).unwrap().available_grids();
        names.sort();
        assert_eq!(names, ["first", "second"]);
        // Each name carries its own socket's values: compare with a plain bake of each noise.
        assert!(
            read_active(&path, "first") == reference[0],
            "first, frame {frame}"
        );
        assert!(
            read_active(&path, "second") == reference[1],
            "second, frame {frame}"
        );
        assert!(reference[0] != reference[1]);
    }
}

#[test]
fn bake_rejects_two_outputs_that_expand_to_the_same_grid_before_baking() {
    // Two entries named "a": `Document::from_json` rejects this itself, so it does NOT cover
    // the ordering of `expanded_names`; the vector-expansion test below does.
    let doc = TWO_NOISES
        .replace("\"first\"", "\"a\"")
        .replace("\"second\"", "\"a\"");
    let dir = tempfile::tempdir().unwrap();
    let graph = dir.path().join("dup.elements");
    std::fs::write(&graph, doc).unwrap();
    let out = dir.path().join("vdb");
    let status = cli()
        .args(["bake", graph.to_str().unwrap()])
        .args(["--out", out.to_str().unwrap(), "--frames", "1"])
        .status()
        .unwrap();
    assert!(!status.success());
    assert!(!out.exists(), "the output directory must not be created");
}

#[test]
fn bake_rejects_a_vector_expansion_collision_before_creating_the_output_directory() {
    // `velocity` expands to velocity_x/_y/_z, colliding with the scalar `velocity_x`. Only
    // `expanded_names` sees this, so it proves names are checked before `create_dir_all`.
    const DOC: &str = r#"{
      "version": 4, "dims": [8, 8, 8],
      "nodes": [
        { "id": 0, "kind": "ember.emitter", "params": {
            "shape": { "sphere": { "radius": 0.1 } },
            "transform": { "keys": [{ "frame": 0 }] } } },
        { "id": 1, "kind": "ember.smoke_solver", "params": {} },
        { "id": 2, "kind": "core.output", "params": {} }
      ],
      "edges": [
        { "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 },
        { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 0 }
      ],
      "output": 2,
      "outputs": [
        { "node": 1, "socket": 2, "name": "velocity" },
        { "node": 1, "socket": 0, "name": "velocity_x" }
      ]
    }"#;
    let dir = tempfile::tempdir().unwrap();
    let graph = dir.path().join("collide.elements");
    std::fs::write(&graph, DOC).unwrap();
    let out = dir.path().join("vdb");
    let result = cli()
        .args(["bake", graph.to_str().unwrap()])
        .args(["--out", out.to_str().unwrap(), "--frames", "1"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("velocity_x"), "{stderr}");
    assert!(!out.exists(), "the output directory must not be created");
}
