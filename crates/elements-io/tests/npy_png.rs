use elements_io::{IoError, read_npy, write_npy, write_slice_png};
use ndarray::Array3;
use ndarray_npy::ReadNpyExt;

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

    // This dims assertion only catches an axis-transposition bug because
    // dims[0] (4) != dims[2] (2) in this fixture: reversing [d0,d1,d2] to
    // [d2,d1,d0] happens to equal the original whenever d0 == d2. It is not
    // a reliable structural check. See `npy_axes_are_stored_as_z_y_x` below
    // for a test that verifies axis semantics regardless of the dimensions
    // chosen.
    assert_eq!(back_dims, dims);
    assert_eq!(back, values);
}

#[test]
fn npy_axes_are_stored_as_z_y_x() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("axes.npy");
    let dims = [4, 3, 2]; // (x, y, z)

    // Every value encodes its own (z, y, x) position so a transposition
    // bug produces a detectable mismatch regardless of dimension sizes.
    let mut values = vec![0.0f32; (dims[0] * dims[1] * dims[2]) as usize];
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let idx = (z * dims[1] * dims[0] + y * dims[0] + x) as usize;
                values[idx] = (z * 100 + y * 10 + x) as f32;
            }
        }
    }

    write_npy(&path, &values, dims).unwrap();

    let file = std::fs::File::open(&path).unwrap();
    let array = Array3::<f32>::read_npy(std::io::BufReader::new(file)).unwrap();

    assert_eq!(array.shape(), &[2, 3, 4]); // (z, y, x)
    assert_eq!(array[[1, 2, 3]], 123.0);
    assert_eq!(array[[0, 0, 3]], 3.0);
    assert_eq!(array[[1, 0, 0]], 100.0);
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
fn png_renders_non_finite_values_as_white() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nonfinite.png");
    let values = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5];

    write_slice_png(&path, &values, [2, 2, 1], 0, (0.0, 1.0)).unwrap();

    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    reader.next_frame(&mut buf).unwrap();

    // NAN and INFINITY clamp to white anyway, but NEG_INFINITY does not fall
    // out of the clamp naturally (it would clamp to 0.0, i.e. black). We
    // deliberately render ALL non-finite values as saturated white (255):
    // these previews exist to make a diverged simulation visually obvious,
    // and a single consistent rule ("non-finite => white") is more
    // trustworthy at a glance than one where -inf looks like a plausible
    // dark region while NaN and +inf look like blow-up.
    assert_eq!(buf[0], 255, "NaN should render as white");
    assert_eq!(buf[1], 255, "+inf should render as white");
    assert_eq!(buf[2], 255, "-inf should render as white");
    assert_eq!(buf[3], 128, "finite values are unaffected");
}

#[test]
fn png_handles_a_zero_width_range() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zero_width.png");
    // range = (0.5, 0.5): the zero-guard substitutes span = 1.0 instead of
    // dividing by zero, so the mapping becomes (v - 0.5) clamped to [0, 1]
    // then scaled to 0..=255. The chosen (and documented) behaviour: values
    // at or below `lo` saturate to black, values at or above `lo + 1.0`
    // saturate to white, with a linear ramp of exactly one unit of value
    // in between.
    //
    // -10.0 and 0.5 sit at or below `lo`, so both the guarded and an
    // unguarded (divide-by-zero) implementation agree they map to 0 --
    // they do not distinguish the two. 1.0 is the discriminating value:
    // under the guard it is the midpoint of the unit ramp (0.5 span
    // fraction -> 128), but without the guard, dividing by zero sends any
    // v != lo straight to +/-infinity, which clamps and casts to 255. So a
    // regression that drops the guard changes this pixel from 128 to 255,
    // which is what the mutation check below exercises. 2.5 is a control
    // value that already saturates to 255 either way.
    let values = vec![-10.0f32, 0.5, 1.0, 2.5];

    write_slice_png(&path, &values, [2, 2, 1], 0, (0.5, 0.5)).unwrap();

    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    reader.next_frame(&mut buf).unwrap();

    assert_eq!(&buf[..4], &[0, 0, 128, 255]);
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
