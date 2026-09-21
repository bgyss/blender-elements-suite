use elements_ipc::{
    CHANNEL_HEADER_BYTES, CHANNEL_MAGIC, CHANNEL_VERSION, ChannelError, FrameReader, FrameWriter,
};

#[test]
fn publish_then_read_returns_the_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");

    let mut writer = FrameWriter::create(&path, [4, 4, 4], 1).unwrap();
    let values: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let seq = writer.publish(&values).unwrap();
    assert_eq!(seq, 1, "the first published frame is sequence 1");

    let reader = FrameReader::open(&path).unwrap();
    let (read_seq, read_values) = reader.read_latest().unwrap();
    assert_eq!(read_seq, 1);
    assert_eq!(read_values, values);
}

#[test]
fn the_header_describes_the_field() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");

    let mut writer = FrameWriter::create(&path, [2, 3, 4], 1).unwrap();
    assert_eq!(writer.dims(), [2, 3, 4]);
    assert_eq!(writer.channels(), 1);
    writer.publish(&[0.0; 24]).unwrap();

    let reader = FrameReader::open(&path).unwrap();
    let header = reader.header();
    assert_eq!(header.magic, CHANNEL_MAGIC);
    assert_eq!(header.version, 1);
    assert_eq!(header.dims, [2, 3, 4]);
    assert_eq!(header.channels, 1);
    assert_eq!(header.buffer_bytes, 24 * 4);
}

#[test]
fn the_file_is_header_plus_two_buffers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");
    FrameWriter::create(&path, [4, 4, 4], 1).unwrap();

    let len = std::fs::metadata(&path).unwrap().len() as usize;
    assert_eq!(len, CHANNEL_HEADER_BYTES + 2 * 64 * 4);
}

#[test]
fn successive_publishes_alternate_buffers_and_advance_seq() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");

    let mut writer = FrameWriter::create(&path, [2, 2, 2], 1).unwrap();
    let reader = FrameReader::open(&path).unwrap();

    for n in 1..=5u64 {
        let values = vec![n as f32; 8];
        assert_eq!(writer.publish(&values).unwrap(), n);
        let (seq, read) = reader.read_latest().unwrap();
        assert_eq!(seq, n);
        assert_eq!(read, values);
    }
}

#[test]
fn a_wrong_length_payload_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");
    let mut writer = FrameWriter::create(&path, [4, 4, 4], 1).unwrap();

    match writer.publish(&[1.0, 2.0]) {
        Err(ChannelError::LengthMismatch {
            expected: 64,
            got: 2,
        }) => {}
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn reading_before_any_publish_reports_sequence_zero() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");
    FrameWriter::create(&path, [2, 2, 2], 1).unwrap();

    let reader = FrameReader::open(&path).unwrap();
    let (seq, values) = reader.read_latest().unwrap();
    assert_eq!(seq, 0, "no frame published yet");
    assert!(values.iter().all(|v| *v == 0.0));
}

#[test]
fn a_foreign_file_is_rejected_by_magic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("not-a-channel.bin");
    std::fs::write(&path, vec![0u8; 256]).unwrap();

    match FrameReader::open(&path) {
        Err(ChannelError::BadMagic { .. }) => {}
        other => panic!("expected BadMagic, got {other:?}"),
    }
}

#[test]
fn a_truncated_channel_file_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.bin");

    // A valid 64-byte header claiming a large `buffer_bytes`, with no buffer
    // data following it at all. This mimics a process crashing mid-write, or
    // a hand-built fixture with a wrong declared size.
    let mut header = vec![0u8; CHANNEL_HEADER_BYTES];
    header[0..4].copy_from_slice(&CHANNEL_MAGIC.to_le_bytes());
    header[4..8].copy_from_slice(&CHANNEL_VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&4u32.to_le_bytes());
    header[12..16].copy_from_slice(&4u32.to_le_bytes());
    header[16..20].copy_from_slice(&4u32.to_le_bytes());
    header[20..24].copy_from_slice(&1u32.to_le_bytes());
    header[24..32].copy_from_slice(&1_000_000u64.to_le_bytes());
    header[32..40].copy_from_slice(&0u64.to_le_bytes());
    std::fs::write(&path, &header).unwrap();

    match FrameReader::open(&path) {
        Err(ChannelError::LengthMismatch { .. }) => {}
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn absurd_dims_report_field_too_large_instead_of_overflowing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");

    match FrameWriter::create(&path, [3_000_000, 3_000_000, 3_000_000], 1) {
        Err(ChannelError::FieldTooLarge { .. }) => {}
        Err(other) => panic!("expected FieldTooLarge, got {other:?}"),
        Ok(_) => panic!("expected FieldTooLarge, got Ok"),
    }
}

/// Helper: a bare `CHANNEL_HEADER_BYTES`-long header with the given fields,
/// no buffer data following it. Mirrors `a_truncated_channel_file_is_rejected`.
fn raw_header(dims: [u32; 3], channels: u32, buffer_bytes: u64) -> Vec<u8> {
    let mut header = vec![0u8; CHANNEL_HEADER_BYTES];
    header[0..4].copy_from_slice(&CHANNEL_MAGIC.to_le_bytes());
    header[4..8].copy_from_slice(&CHANNEL_VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&dims[0].to_le_bytes());
    header[12..16].copy_from_slice(&dims[1].to_le_bytes());
    header[16..20].copy_from_slice(&dims[2].to_le_bytes());
    header[20..24].copy_from_slice(&channels.to_le_bytes());
    header[24..32].copy_from_slice(&buffer_bytes.to_le_bytes());
    header[32..40].copy_from_slice(&0u64.to_le_bytes());
    header
}

// The following three regression tests reproduce panics found by review in
// `FrameReader::open`/`read_latest` on a malformed channel file: an
// internally-inconsistent header used to reach `copy_from_slice` (length
// mismatch) in `read_latest`, and both `buffer_bytes = u64::MAX` and
// `dims = [u32::MAX; 3]` used to panic on overflowing arithmetic in `open`
// itself (in debug; silently wrapped in release). `FrameReader::open` now
// routes through `checked_value_count_and_bytes` and cross-checks
// `buffer_bytes` against what `dims`/`channels` imply, so all three now
// return a typed `ChannelError` instead.

#[test]
fn an_internally_inconsistent_header_is_rejected_not_panicked() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("inconsistent.bin");

    // dims=[2,2,2], channels=1 implies 8 values (32 bytes), but the header
    // claims buffer_bytes=64. The file is padded to the *declared* size
    // (header + 2*64) so the old code's map-length check alone would not
    // catch this -- only the removed cross-check in `open`, or the
    // `copy_from_slice` in `read_latest` (64 vs. 32 bytes), would.
    let header = raw_header([2, 2, 2], 1, 64);
    let mut bytes = header;
    bytes.resize(CHANNEL_HEADER_BYTES + 2 * 64, 0);
    std::fs::write(&path, &bytes).unwrap();

    let reader = match FrameReader::open(&path) {
        Err(ChannelError::LengthMismatch { .. }) => return,
        Err(other) => panic!("expected LengthMismatch or Ok, got {other:?}"),
        Ok(reader) => reader,
    };
    match reader.read_latest() {
        Err(ChannelError::LengthMismatch { .. }) => {}
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn a_buffer_bytes_of_u64_max_is_rejected_not_panicked() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("huge-buffer-bytes.bin");

    // `CHANNEL_HEADER_BYTES + 2 * buffer_bytes` previously overflowed
    // computing this (panic in debug, silent wraparound in release).
    let header = raw_header([1, 1, 1], 1, u64::MAX);
    std::fs::write(&path, &header).unwrap();

    match FrameReader::open(&path) {
        Err(ChannelError::LengthMismatch { .. }) => {}
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn dims_of_u32_max_are_rejected_not_panicked() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("huge-dims.bin");

    // `value_count` previously computed `dims[0] * dims[1] * dims[2] *
    // channels` in `usize` with a plain multiplication, overflowing.
    let header = raw_header([u32::MAX, u32::MAX, u32::MAX], 1, 0);
    std::fs::write(&path, &header).unwrap();

    match FrameReader::open(&path) {
        Err(ChannelError::FieldTooLarge { .. }) => {}
        other => panic!("expected FieldTooLarge, got {other:?}"),
    }
}

#[test]
fn a_reader_detects_a_reallocated_channel() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");

    let mut writer = FrameWriter::create(&path, [8, 8, 8], 1).unwrap();
    writer.publish(&[1.0; 512]).unwrap();

    let reader = FrameReader::open(&path).unwrap();
    let (seq, values) = reader.read_latest().unwrap();
    assert_eq!(seq, 1);
    assert_eq!(values, vec![1.0; 512]);

    // Reallocate the channel in place at the same path with different dims,
    // as `daemon::load` does on a resolution change.
    let _new_writer = FrameWriter::create(&path, [4, 4, 4], 1).unwrap();

    match reader.read_latest() {
        Err(ChannelError::Reallocated) => {}
        other => panic!("expected Reallocated, got {other:?}"),
    }
}

#[test]
fn a_reader_survives_a_writer_publishing_concurrently() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");
    let mut writer = FrameWriter::create(&path, [8, 8, 8], 1).unwrap();
    writer.publish(&[1.0; 512]).unwrap();

    let reader = FrameReader::open(&path).unwrap();
    let handle = std::thread::spawn(move || {
        for n in 2..200u64 {
            writer.publish(&[n as f32; 512]).unwrap();
        }
    });

    // Every successful read must be internally consistent: one uniform value.
    for _ in 0..500 {
        if let Ok((_seq, values)) = reader.read_latest() {
            let first = values[0];
            assert!(
                values.iter().all(|v| *v == first),
                "a torn frame leaked through: saw mixed values"
            );
        }
    }

    handle.join().unwrap();
}
