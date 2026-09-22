//! Physical-quality metrics computed on the CPU from read-back fields.
//!
//! The same functions serve the validation scenes, the speed gate, and 2b's
//! Mantaflow benchmark (spec §5.4), so both solvers are measured by one piece
//! of code. Sums are in `f64`: a 128³ grid has two million cells.

use elements_core::gpu::{FieldDims, StaggeredField};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DivergenceStats {
    /// Largest |div u| over all cells, 1/s.
    pub max_abs: f64,
    /// Root mean square of div u over all cells, 1/s.
    pub rms: f64,
}

fn at(dims: FieldDims, i: u32, j: u32, k: u32) -> usize {
    (i + dims.x * (j + dims.y * k)) as usize
}

/// Divergence of a staggered velocity, from its X, Y and Z faces (x-fastest,
/// as `Field::read_back` returns them), with voxel edge `dx` metres.
pub fn divergence(faces: &[Vec<f32>; 3], cells: FieldDims, dx: f32) -> DivergenceStats {
    use elements_core::gpu::Axis;
    let xd = StaggeredField::face_dims(cells, Axis::X);
    let yd = StaggeredField::face_dims(cells, Axis::Y);
    let zd = StaggeredField::face_dims(cells, Axis::Z);
    let dx = dx as f64;
    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let d = (faces[0][at(xd, i + 1, j, k)] as f64 - faces[0][at(xd, i, j, k)] as f64
                    + faces[1][at(yd, i, j + 1, k)] as f64
                    - faces[1][at(yd, i, j, k)] as f64
                    + faces[2][at(zd, i, j, k + 1)] as f64
                    - faces[2][at(zd, i, j, k)] as f64)
                    / dx;
                max_abs = max_abs.max(d.abs());
                sum_sq += d * d;
            }
        }
    }
    DivergenceStats {
        max_abs,
        rms: (sum_sq / cells.voxel_count() as f64).sqrt(),
    }
}

/// The density-weighted mean height, in cell units (cell k's centre is at
/// k + 0.5). `None` when there is no density.
pub fn centroid_z(density: &[f32], cells: FieldDims) -> Option<f64> {
    let mut mass = 0.0f64;
    let mut moment = 0.0f64;
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let m = density[at(cells, i, j, k)] as f64;
                mass += m;
                moment += m * (k as f64 + 0.5);
            }
        }
    }
    (mass > 0.0).then(|| moment / mass)
}
