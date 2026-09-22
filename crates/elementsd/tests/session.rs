use std::io::{BufRead, BufReader};
use std::process::{Child, Command as ProcCommand, Stdio};

use elements_ipc::{
    Command, ELEMENTS_PROTOCOL_VERSION, ErrorKind, FrameReader, Response, read_message,
    write_message,
};

struct Daemon {
    child: Child,
    endpoint: String,
    channel: std::path::PathBuf,
    _dir: tempfile::TempDir,
}

impl Daemon {
    fn start() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let channel = dir.path().join("frame.bin");
        let endpoint = if cfg!(windows) {
            format!("\\\\.\\pipe\\elements-test-{}", std::process::id())
        } else {
            dir.path()
                .join("control.sock")
                .to_string_lossy()
                .into_owned()
        };

        let mut child = ProcCommand::new(env!("CARGO_BIN_EXE_elementsd"))
            .args(["--endpoint", &endpoint])
            .args(["--channel", channel.to_str().unwrap()])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();

        // Wait for the readiness line rather than sleeping.
        let mut line = String::new();
        BufReader::new(child.stdout.as_mut().unwrap())
            .read_line(&mut line)
            .unwrap();
        assert!(line.starts_with("ready "), "daemon said: {line}");

        Self {
            child,
            endpoint,
            channel,
            _dir: dir,
        }
    }

    fn connect(&self) -> elements_ipc::Stream {
        elements_ipc::Stream::connect(&self.endpoint).unwrap()
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn graph_file(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("noise.elements");
    std::fs::write(
        &path,
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
    path
}

#[test]
fn a_full_session_loads_renders_and_shuts_down() {
    let daemon = Daemon::start();
    let dir = tempfile::tempdir().unwrap();
    let graph = graph_file(dir.path());

    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());

    write_message(
        &mut stream,
        &Command::Hello {
            protocol_version: ELEMENTS_PROTOCOL_VERSION,
        },
    )
    .unwrap();
    let ack: Response = read_message(&mut reader).unwrap().unwrap();
    match ack {
        Response::HelloAck {
            protocol_version,
            adapter,
            ..
        } => {
            assert_eq!(protocol_version, ELEMENTS_PROTOCOL_VERSION);
            assert!(!adapter.is_empty());
        }
        other => panic!("expected HelloAck, got {other:?}"),
    }

    write_message(
        &mut stream,
        &Command::LoadGraph {
            path: graph.to_string_lossy().into_owned(),
        },
    )
    .unwrap();
    match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
        Response::Loaded { dims, nodes } => {
            assert_eq!(dims, [8, 8, 8]);
            assert_eq!(nodes, 2);
        }
        other => panic!("expected Loaded, got {other:?}"),
    }

    write_message(&mut stream, &Command::Render { frame: 1 }).unwrap();
    let seq = match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
        Response::Frame { seq, dims, .. } => {
            assert_eq!(dims, [8, 8, 8]);
            seq
        }
        other => panic!("expected Frame, got {other:?}"),
    };
    assert_eq!(seq, 1);

    let frame_reader = FrameReader::open(&daemon.channel).unwrap();
    let (read_seq, values) = frame_reader.read_latest().unwrap();
    assert_eq!(read_seq, 1);
    assert_eq!(values.len(), 512);
    assert!(
        values.iter().any(|v| *v != 0.0),
        "the frame must not be blank"
    );
    assert!(values.iter().all(|v| v.is_finite()));

    write_message(&mut stream, &Command::Shutdown).unwrap();
    assert!(matches!(
        read_message::<_, Response>(&mut reader).unwrap().unwrap(),
        Response::Bye
    ));
}

#[test]
fn a_protocol_version_mismatch_is_refused_with_a_typed_error() {
    let daemon = Daemon::start();
    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());

    write_message(
        &mut stream,
        &Command::Hello {
            protocol_version: 999,
        },
    )
    .unwrap();
    match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
        Response::Error(e) => {
            assert_eq!(e.kind, ErrorKind::ProtocolVersion);
            assert!(e.message.contains("999"), "message was: {}", e.message);
        }
        other => panic!("expected Error, got {other:?}"),
    }
}

#[test]
fn rendering_before_loading_a_graph_is_an_error_not_a_crash() {
    let daemon = Daemon::start();
    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());

    write_message(
        &mut stream,
        &Command::Hello {
            protocol_version: ELEMENTS_PROTOCOL_VERSION,
        },
    )
    .unwrap();
    let _: Response = read_message(&mut reader).unwrap().unwrap();

    write_message(&mut stream, &Command::Render { frame: 1 }).unwrap();
    match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
        Response::Error(e) => assert_eq!(e.kind, ErrorKind::Graph),
        other => panic!("expected Error, got {other:?}"),
    }

    // The daemon must still be usable afterwards.
    write_message(&mut stream, &Command::Shutdown).unwrap();
    assert!(matches!(
        read_message::<_, Response>(&mut reader).unwrap().unwrap(),
        Response::Bye
    ));
}

#[test]
fn a_malformed_graph_reports_a_document_error_and_keeps_the_session() {
    let daemon = Daemon::start();
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.elements");
    std::fs::write(&bad, "{ not json").unwrap();

    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    write_message(
        &mut stream,
        &Command::Hello {
            protocol_version: ELEMENTS_PROTOCOL_VERSION,
        },
    )
    .unwrap();
    let _: Response = read_message(&mut reader).unwrap().unwrap();

    write_message(
        &mut stream,
        &Command::LoadGraph {
            path: bad.to_string_lossy().into_owned(),
        },
    )
    .unwrap();
    match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
        Response::Error(e) => assert_eq!(e.kind, ErrorKind::Document),
        other => panic!("expected Error, got {other:?}"),
    }

    // A second, valid load on the same connection must succeed.
    let good = graph_file(dir.path());
    write_message(
        &mut stream,
        &Command::LoadGraph {
            path: good.to_string_lossy().into_owned(),
        },
    )
    .unwrap();
    assert!(matches!(
        read_message::<_, Response>(&mut reader).unwrap().unwrap(),
        Response::Loaded { .. }
    ));
}

#[test]
fn a_graph_with_absurd_dimensions_is_an_error_not_a_crash() {
    let daemon = Daemon::start();
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("absurd.elements");
    std::fs::write(
        &bad,
        r#"{
          "version": 1,
          "dims": [3000000, 3000000, 3000000],
          "nodes": [
            { "id": 0, "kind": "core.noise_field", "params": { "seed": 7 } },
            { "id": 1, "kind": "core.output", "params": {} }
          ],
          "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
          "output": 1
        }"#,
    )
    .unwrap();

    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    write_message(
        &mut stream,
        &Command::Hello {
            protocol_version: ELEMENTS_PROTOCOL_VERSION,
        },
    )
    .unwrap();
    let _: Response = read_message(&mut reader).unwrap().unwrap();

    write_message(
        &mut stream,
        &Command::LoadGraph {
            path: bad.to_string_lossy().into_owned(),
        },
    )
    .unwrap();
    match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
        Response::Error(e) => assert_eq!(e.kind, ErrorKind::Document),
        other => panic!("expected Error, got {other:?}"),
    }

    // The session must still be usable: a valid graph loads normally.
    let good = graph_file(dir.path());
    write_message(
        &mut stream,
        &Command::LoadGraph {
            path: good.to_string_lossy().into_owned(),
        },
    )
    .unwrap();
    assert!(matches!(
        read_message::<_, Response>(&mut reader).unwrap().unwrap(),
        Response::Loaded { .. }
    ));
}

/// `GpuContext` now requests the adapter's own resolution limits, so
/// `max_texture_dimension_3d` is the real hardware cap (2048 on Apple
/// Silicon) rather than the `downlevel_defaults` 256. `dims = [65536, 4, 4]`
/// (well under the `FieldTooLarge` overflow bound exercised by
/// `a_graph_with_absurd_dimensions_is_an_error_not_a_crash`, but over any
/// real adapter's 3D limit) must fail at `LoadGraph` with a `Document` error
/// naming the offending dimension, not answer `Loaded` and fail later on the
/// first `Render` as a generic `ErrorKind::Gpu` the add-on treats as
/// transient and worth retrying.
#[test]
fn dims_exceeding_the_devices_texture_limit_are_rejected_at_load_not_render() {
    let daemon = Daemon::start();
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("oversized.elements");
    std::fs::write(
        &bad,
        r#"{
          "version": 1,
          "dims": [65536, 4, 4],
          "nodes": [
            { "id": 0, "kind": "core.noise_field", "params": { "seed": 7 } },
            { "id": 1, "kind": "core.output", "params": {} }
          ],
          "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
          "output": 1
        }"#,
    )
    .unwrap();

    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    write_message(
        &mut stream,
        &Command::Hello {
            protocol_version: ELEMENTS_PROTOCOL_VERSION,
        },
    )
    .unwrap();
    let _: Response = read_message(&mut reader).unwrap().unwrap();

    write_message(
        &mut stream,
        &Command::LoadGraph {
            path: bad.to_string_lossy().into_owned(),
        },
    )
    .unwrap();
    match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
        Response::Error(e) => {
            assert_eq!(e.kind, ErrorKind::Document);
            assert!(
                e.message.contains("65536"),
                "message should name the offending dimension: {}",
                e.message
            );
        }
        other => panic!("expected Error, got {other:?}"),
    }

    // The session must still be usable: a valid graph loads normally.
    let good = graph_file(dir.path());
    write_message(
        &mut stream,
        &Command::LoadGraph {
            path: good.to_string_lossy().into_owned(),
        },
    )
    .unwrap();
    assert!(matches!(
        read_message::<_, Response>(&mut reader).unwrap().unwrap(),
        Response::Loaded { .. }
    ));
}

#[test]
fn rendering_before_hello_is_refused() {
    let daemon = Daemon::start();
    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());

    write_message(&mut stream, &Command::Render { frame: 1 }).unwrap();
    match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
        Response::Error(e) => assert_eq!(e.kind, ErrorKind::ProtocolVersion),
        other => panic!("expected Error, got {other:?}"),
    }
}

#[test]
fn repeated_renders_advance_the_sequence() {
    let daemon = Daemon::start();
    let dir = tempfile::tempdir().unwrap();
    let graph = graph_file(dir.path());

    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    write_message(
        &mut stream,
        &Command::Hello {
            protocol_version: ELEMENTS_PROTOCOL_VERSION,
        },
    )
    .unwrap();
    let _: Response = read_message(&mut reader).unwrap().unwrap();
    write_message(
        &mut stream,
        &Command::LoadGraph {
            path: graph.to_string_lossy().into_owned(),
        },
    )
    .unwrap();
    let _: Response = read_message(&mut reader).unwrap().unwrap();

    for expected in 1..=4u64 {
        write_message(
            &mut stream,
            &Command::Render {
                frame: expected as u32,
            },
        )
        .unwrap();
        match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
            Response::Frame { seq, .. } => assert_eq!(seq, expected),
            other => panic!("expected Frame, got {other:?}"),
        }
    }
}

/// `Daemon::start`'s own `read_line` has no timeout: if the daemon never
/// prints (or never flushes) its readiness line, that helper — and every
/// other test in this file, since they all go through it — hangs forever
/// instead of failing. A hang is indistinguishable from a slow machine in a
/// test runner and gives no error to act on, which is exactly the failure
/// mode the readiness-line contract exists to prevent.
///
/// This test exercises the same contract but bounds the wait itself, so a
/// broken contract shows up as a deterministic assertion failure within a
/// few seconds rather than as a CI job that never returns.
#[test]
fn the_readiness_line_arrives_promptly_after_bind() {
    let dir = tempfile::tempdir().unwrap();
    let channel = dir.path().join("frame.bin");
    let endpoint = if cfg!(windows) {
        format!(
            "\\\\.\\pipe\\elements-test-readiness-{}",
            std::process::id()
        )
    } else {
        dir.path()
            .join("control.sock")
            .to_string_lossy()
            .into_owned()
    };

    let mut child = ProcCommand::new(env!("CARGO_BIN_EXE_elementsd"))
        .args(["--endpoint", &endpoint])
        .args(["--channel", channel.to_str().unwrap()])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stdout = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    // The read itself has no timeout, so it runs on its own thread; the
    // bound comes from `recv_timeout` on the main thread instead. If the
    // child never writes a line, this thread blocks forever, but it is a
    // daemon-less detached thread and the process exits over it once the
    // test binary finishes, so it does not hang the test run.
    std::thread::spawn(move || {
        let mut line = String::new();
        let result = BufReader::new(&mut stdout).read_line(&mut line);
        let _ = tx.send(result.map(|_| line));
    });

    let outcome = rx.recv_timeout(std::time::Duration::from_secs(5));

    // Clean up the child regardless of how the assertion below turns out.
    let _ = child.kill();
    let _ = child.wait();

    match outcome {
        Ok(Ok(line)) => {
            assert!(line.starts_with("ready "), "daemon said: {line}");
        }
        Ok(Err(e)) => panic!("failed to read the daemon's stdout: {e}"),
        Err(_) => panic!(
            "no readiness line arrived on stdout within 5s of spawning the daemon; \
             either it was never printed or stdout was never flushed"
        ),
    }
}

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/elementsd is two levels below the root")
        .to_path_buf()
}

/// Render `frame` and read the published values back from the channel.
fn render_values(
    stream: &mut elements_ipc::Stream,
    reader: &mut BufReader<elements_ipc::Stream>,
    channel: &std::path::Path,
    frame: u32,
) -> Vec<f32> {
    write_message(stream, &Command::Render { frame }).unwrap();
    match read_message::<_, Response>(reader).unwrap().unwrap() {
        Response::Frame { .. } => {}
        other => panic!("expected Frame, got {other:?}"),
    }
    let frames = FrameReader::open(channel).unwrap();
    frames.read_latest().unwrap().1
}

#[test]
fn render_produces_the_requested_frame_of_a_stateful_graph() {
    let daemon = Daemon::start();
    let graph = repo_root().join("tests/graphs/accumulate_4.elements");

    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    write_message(
        &mut stream,
        &Command::Hello {
            protocol_version: ELEMENTS_PROTOCOL_VERSION,
        },
    )
    .unwrap();
    let _: Response = read_message(&mut reader).unwrap().unwrap();
    write_message(
        &mut stream,
        &Command::LoadGraph {
            path: graph.to_string_lossy().into_owned(),
        },
    )
    .unwrap();
    assert!(matches!(
        read_message::<_, Response>(&mut reader).unwrap().unwrap(),
        Response::Loaded { .. }
    ));

    let near = |values: &[f32], frame: u32| {
        let want = 0.5 * frame as f32 / 24.0;
        assert!(
            values.iter().all(|v| (v - want).abs() < 1e-6),
            "frame {frame}: expected {want}, got {:?}",
            &values[..4]
        );
    };

    let three = render_values(&mut stream, &mut reader, &daemon.channel, 3);
    near(&three, 3);
    let one = render_values(&mut stream, &mut reader, &daemon.channel, 1);
    near(&one, 1);
    let three_again = render_values(&mut stream, &mut reader, &daemon.channel, 3);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&three_again),
        bits(&three),
        "frame 3 must be bit-identical on revisit"
    );
}
