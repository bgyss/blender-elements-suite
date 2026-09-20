use elements_io::{IoError, read_npy, write_npy, write_slice_png};

fn ramp(dims: [u32; 3]) -> Vec<f32> {
    let n = (dims[0] * dims[1] * dims[2]) as usize;
    (0..n).map(|i| i as f32 / n as f32).collect()
}

#[test]
fn npy_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("field.npy");
    let dims = [4, 3, 2];
    let values = ramp(dims);

    write_npy(&path, &values, dims).unwrap();
    let (back, back_dims) = read_npy(&path).unwrap();

    assert_eq!(back_dims, dims);
    assert_eq!(back, values);
}

#[test]
fn npy_rejects_a_length_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.npy");
    match write_npy(&path, &[1.0, 2.0], [4, 4, 4]) {
        Err(IoError::LengthMismatch {
            expected: 64,
            got: 2,
        }) => {}
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn png_writes_the_requested_slice() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("slice.png");
    // Each z-slice is constant and equal to its index, so output is checkable.
    let mut values = vec![0.0f32; 8 * 8 * 4];
    for z in 0..4 {
        for i in 0..64 {
            values[z * 64 + i] = z as f32;
        }
    }

    write_slice_png(&path, &values, [8, 8, 4], 2, (0.0, 3.0)).unwrap();

    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();

    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
    // z = 2 mapped through range 0..3 is 2/3 of full scale, rounding to 170.
    assert!(buf[..info.buffer_size()].iter().all(|b| *b == 170));
}

#[test]
fn png_clamps_out_of_range_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clamped.png");
    let values = vec![-5.0f32, 5.0, -5.0, 5.0];

    write_slice_png(&path, &values, [2, 2, 1], 0, (0.0, 1.0)).unwrap();

    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    reader.next_frame(&mut buf).unwrap();

    assert_eq!(&buf[..4], &[0, 255, 0, 255]);
}

#[test]
fn png_rejects_an_out_of_bounds_slice() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("oob.png");
    match write_slice_png(&path, &ramp([2, 2, 2]), [2, 2, 2], 9, (0.0, 1.0)) {
        Err(IoError::BadSlice { z: 9, depth: 2 }) => {}
        other => panic!("expected BadSlice, got {other:?}"),
    }
}
