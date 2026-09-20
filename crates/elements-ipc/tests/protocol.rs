use std::io::{BufReader, Cursor};

use elements_ipc::{
    Command, ELEMENTS_PROTOCOL_VERSION, EngineError, ErrorKind, ProtocolError, Response,
    read_message, write_message,
};

#[test]
fn commands_round_trip_as_ndjson() {
    let mut buf = Vec::new();
    write_message(
        &mut buf,
        &Command::Hello {
            protocol_version: ELEMENTS_PROTOCOL_VERSION,
        },
    )
    .unwrap();
    write_message(&mut buf, &Command::Render { frame: 12 }).unwrap();
    write_message(&mut buf, &Command::Shutdown).unwrap();

    let mut reader = BufReader::new(Cursor::new(buf));
    let a: Command = read_message(&mut reader).unwrap().unwrap();
    let b: Command = read_message(&mut reader).unwrap().unwrap();
    let c: Command = read_message(&mut reader).unwrap().unwrap();
    let end: Option<Command> = read_message(&mut reader).unwrap();

    assert!(matches!(
        a,
        Command::Hello {
            protocol_version: 1
        }
    ));
    assert!(matches!(b, Command::Render { frame: 12 }));
    assert!(matches!(c, Command::Shutdown));
    assert!(end.is_none(), "clean end of stream is None, not an error");
}

#[test]
fn messages_are_one_line_each() {
    let mut buf = Vec::new();
    write_message(&mut buf, &Command::Render { frame: 1 }).unwrap();
    let text = String::from_utf8(buf).unwrap();

    assert!(text.ends_with('\n'));
    assert_eq!(text.matches('\n').count(), 1, "no embedded newlines");
}

#[test]
fn the_wire_format_is_tagged_and_snake_case() {
    let mut buf = Vec::new();
    write_message(
        &mut buf,
        &Command::LoadGraph {
            path: "/tmp/a".into(),
        },
    )
    .unwrap();
    let text = String::from_utf8(buf).unwrap();

    assert!(text.contains("\"type\":\"load_graph\""), "got {text}");
    assert!(text.contains("\"path\":\"/tmp/a\""), "got {text}");
}

#[test]
fn responses_round_trip() {
    let mut buf = Vec::new();
    write_message(
        &mut buf,
        &Response::Frame {
            seq: 7,
            channel: "/tmp/elements/frame.bin".into(),
            dims: [8, 8, 8],
        },
    )
    .unwrap();

    let mut reader = BufReader::new(Cursor::new(buf));
    let msg: Response = read_message(&mut reader).unwrap().unwrap();
    match msg {
        Response::Frame { seq, dims, .. } => {
            assert_eq!(seq, 7);
            assert_eq!(dims, [8, 8, 8]);
        }
        other => panic!("expected Frame, got {other:?}"),
    }
}

#[test]
fn engine_errors_carry_a_machine_readable_kind() {
    let mut buf = Vec::new();
    write_message(
        &mut buf,
        &Response::Error(EngineError {
            kind: ErrorKind::DeviceLost,
            message: "adapter reset".into(),
        }),
    )
    .unwrap();
    let text = String::from_utf8(buf.clone()).unwrap();
    assert!(text.contains("\"kind\":\"device_lost\""), "got {text}");

    let mut reader = BufReader::new(Cursor::new(buf));
    let msg: Response = read_message(&mut reader).unwrap().unwrap();
    assert!(matches!(
        msg,
        Response::Error(EngineError {
            kind: ErrorKind::DeviceLost,
            ..
        })
    ));
}

#[test]
fn malformed_json_is_an_error_not_a_panic() {
    let mut reader = BufReader::new(Cursor::new(b"{ not json }\n".to_vec()));
    let result: Result<Option<Command>, _> = read_message(&mut reader);
    assert!(matches!(result, Err(ProtocolError::Json(_))));
}

#[test]
fn oversized_lines_are_rejected_before_parsing() {
    let mut line = vec![b'x'; 2 * 1024 * 1024];
    line.push(b'\n');
    let mut reader = BufReader::new(Cursor::new(line));
    let result: Result<Option<Command>, _> = read_message(&mut reader);
    assert!(
        matches!(result, Err(ProtocolError::OversizedFrame { .. })),
        "a hostile or desynced peer must not be able to exhaust memory"
    );
}

#[test]
fn blank_lines_are_skipped() {
    let mut buf = b"\n\n".to_vec();
    write_message(&mut buf, &Command::Shutdown).unwrap();
    let mut reader = BufReader::new(Cursor::new(buf));
    let msg: Command = read_message(&mut reader).unwrap().unwrap();
    assert!(matches!(msg, Command::Shutdown));
}
