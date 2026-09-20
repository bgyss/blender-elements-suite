use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

/// The repository root, derived from this crate's manifest directory.
fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/elementsd is two levels below the root")
        .to_path_buf()
}

fn python() -> &'static str {
    if cfg!(windows) { "python" } else { "python3" }
}

#[test]
fn the_python_client_speaks_the_real_protocol() {
    let dir = tempfile::tempdir().unwrap();
    let channel = dir.path().join("frame.bin");
    let endpoint = if cfg!(windows) {
        format!("\\\\.\\pipe\\elements-pycontract-{}", std::process::id())
    } else {
        dir.path()
            .join("control.sock")
            .to_string_lossy()
            .into_owned()
    };

    let graph = dir.path().join("noise.elements");
    std::fs::write(
        &graph,
        r#"{
          "version": 1,
          "dims": [8, 8, 8],
          "nodes": [
            { "id": 0, "kind": "core.noise_field", "params": { "seed": 7 } },
            { "id": 1, "kind": "core.output", "params": {} }
          ],
          "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
          "output": 1
        }"#,
    )
    .unwrap();

    let mut daemon = Command::new(env!("CARGO_BIN_EXE_elementsd"))
        .args(["--endpoint", &endpoint])
        .args(["--channel", channel.to_str().unwrap()])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    let mut line = String::new();
    BufReader::new(daemon.stdout.as_mut().unwrap())
        .read_line(&mut line)
        .unwrap();
    assert!(line.starts_with("ready "), "daemon said: {line}");

    let script = repo_root().join("tests/python/contract.py");
    let output = Command::new(python())
        .arg(&script)
        .arg(&endpoint)
        .arg(channel.to_str().unwrap())
        .arg(graph.to_str().unwrap())
        .output()
        .expect("python3 must be on PATH");

    let _ = daemon.kill();
    let _ = daemon.wait();

    assert!(
        output.status.success(),
        "python contract failed\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("python contract ok"));
}
