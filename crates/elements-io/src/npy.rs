//! `.npy` output, the storage format for golden tests.

use std::path::Path;

use ndarray::Array3;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};

use crate::IoError;

/// Write `values` as a C-order `(z, y, x)` float32 array.
///
/// `values` is x-fastest, matching `Field::read_back`.
pub fn write_npy(path: &Path, values: &[f32], dims: [u32; 3]) -> Result<(), IoError> {
    let expected = dims[0] as usize * dims[1] as usize * dims[2] as usize;
    if values.len() != expected {
        return Err(IoError::LengthMismatch {
            expected,
            got: values.len(),
        });
    }

    let array = Array3::from_shape_vec(
        (dims[2] as usize, dims[1] as usize, dims[0] as usize),
        values.to_vec(),
    )
    .map_err(|e| IoError::Npy(e.to_string()))?;

    let file = std::fs::File::create(path)?;
    array
        .write_npy(std::io::BufWriter::new(file))
        .map_err(|e| IoError::Npy(e.to_string()))
}

/// Read back an array written by [`write_npy`].
pub fn read_npy(path: &Path) -> Result<(Vec<f32>, [u32; 3]), IoError> {
    let file = std::fs::File::open(path)?;
    let array = Array3::<f32>::read_npy(std::io::BufReader::new(file))
        .map_err(|e| IoError::Npy(e.to_string()))?;

    let shape = array.shape();
    let dims = [shape[2] as u32, shape[1] as u32, shape[0] as u32];
    let values = array.into_raw_vec_and_offset().0;

    Ok((values, dims))
}
