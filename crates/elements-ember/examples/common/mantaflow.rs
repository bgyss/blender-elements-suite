//! Reads one frame of Blender's Mantaflow cache (`fluid_data_NNNN.vdb`) into
//! the dense, x-fastest arrays that `elements_ember::metrics::Sample` takes.
//!
//! This lives beside the benchmark example, not in the library, because
//! `vdb-rs` is a dev-dependency: the daemon must never link it. The facts it
//! relies on are in `docs/bench/mantaflow-notes.md`:
//!
//! - Index (0, 0, 0) is the domain's first cell. The VDB transform is ignored.
//! - `velocity` is a staggered grid: index (i, j, k) holds the −x, −y and −z
//!   faces of cell (i, j, k), Ember's face convention. The cache stores it only
//!   where there is smoke, and never at face index n.
//! - Stored velocity is in cells per Mantaflow time unit (0.4 s), so
//!   u [m/s] = stored · dx / 0.4.
//! - `vdb-rs` does not check value types, so the reader checks each grid's
//!   type string before reading it.
//! - The writer leaves out a velocity equal to the background, (0, 0, 0),
//!   even where there is smoke, as at a collider. The reader lists the cells
//!   that have density but no velocity. `check_coverage` accepts one only
//!   when all three faces it holds are wall faces by the collider mask (the
//!   cell or its −axis neighbour is solid, on every axis). No metric reads a
//!   wall face, so the value is never used; any other missing cell is an
//!   error.

use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use elements_core::gpu::{Axis, FieldDims, StaggeredField};
use vdb_rs::{VdbLevel, VdbReader};

pub const DENSITY_GRID: &str = "density";
pub const VELOCITY_GRID: &str = "velocity";
pub const FLOAT_TREE: &str = "Tree_float_5_4_3";
pub const VEC3_TREE: &str = "Tree_vec3s_5_4_3";

/// A Mantaflow time unit in seconds, at any fps (notes: Velocity location and
/// units).
const TIME_UNIT_S: f64 = 0.4;

pub struct CacheFrame {
    /// The file the frame was read from, for error messages.
    pub path: PathBuf,
    pub cells: FieldDims,
    /// x-fastest, at the cell dims; 0 where the cache stores nothing.
    pub density: Vec<f32>,
    /// Face velocities in m/s, x-fastest, each axis one longer along itself
    /// (`StaggeredField::face_dims`), as `metrics::Sample::faces` expects.
    /// Faces the cache does not store, including every face at index n, are 0.
    pub faces: [Vec<f32>; 3],
    /// Cells that store density but no velocity, in x-fastest order. Their
    /// faces read as 0 in `faces`; `check_coverage` decides if that is safe.
    pub missing_velocity: Vec<[u32; 3]>,
}

#[derive(Debug)]
pub enum CacheError {
    Open {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: PathBuf,
        message: String,
    },
    MissingGrid {
        path: PathBuf,
        grid: String,
        found: Vec<String>,
    },
    WrongType {
        path: PathBuf,
        grid: String,
        expected: &'static str,
        found: String,
    },
    OutOfDomain {
        path: PathBuf,
        grid: String,
        index: [i32; 3],
        cells: [u32; 3],
    },
    /// A cell stores density but no velocity, and not every face that value
    /// holds is a wall face, so a metric could read the missing value as 0.
    /// `cell` is the first such cell; `count` is how many the frame has.
    MissingVelocity {
        path: PathBuf,
        cell: [u32; 3],
        density: f32,
        count: usize,
    },
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheError::Open { path, source } => {
                write!(f, "cannot open {}: {source}", path.display())
            }
            CacheError::Parse { path, message } => {
                write!(f, "cannot parse {}: {message}", path.display())
            }
            CacheError::MissingGrid { path, grid, found } => write!(
                f,
                "{} has no grid {grid:?}; it has {found:?}",
                path.display()
            ),
            CacheError::WrongType {
                path,
                grid,
                expected,
                found,
            } => write!(
                f,
                "grid {grid:?} in {} is {found}, expected {expected}",
                path.display()
            ),
            CacheError::OutOfDomain {
                path,
                grid,
                index,
                cells,
            } => write!(
                f,
                "grid {grid:?} in {} stores index {index:?}, outside a domain of {cells:?} cells",
                path.display()
            ),
            CacheError::MissingVelocity {
                path,
                cell: [i, j, k],
                density,
                count,
            } => write!(
                f,
                "{} stores density {density:e} but no velocity at cell ({i}, {j}, {k}), \
                 and not all of that cell's −x, −y and −z faces are collider walls \
                 ({count} such cells in this frame); the missing velocity would read as 0",
                path.display()
            ),
        }
    }
}

impl std::error::Error for CacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CacheError::Open { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Read one cache frame of a domain of `cells` with voxel edge `dx` metres.
pub fn read_frame(path: &Path, cells: FieldDims, dx: f64) -> Result<CacheFrame, CacheError> {
    read_frame_with(path, cells, dx, DENSITY_GRID, VELOCITY_GRID)
}

/// The same, with the grid names given, so a test can ask for a missing one.
pub fn read_frame_with(
    path: &Path,
    cells: FieldDims,
    dx: f64,
    density_grid: &str,
    velocity_grid: &str,
) -> Result<CacheFrame, CacheError> {
    let file = File::open(path).map_err(|source| CacheError::Open {
        path: path.to_owned(),
        source,
    })?;
    let mut reader = VdbReader::new(BufReader::new(file)).map_err(|e| parse(path, e))?;

    check_grid(&reader, path, density_grid, FLOAT_TREE)?;
    check_grid(&reader, path, velocity_grid, VEC3_TREE)?;

    let out_of_domain = |grid: &str, index: [i32; 3]| CacheError::OutOfDomain {
        path: path.to_owned(),
        grid: grid.to_owned(),
        index,
        cells: [cells.x, cells.y, cells.z],
    };

    // Density: one value per cell.
    let grid = reader
        .read_grid::<f32>(density_grid)
        .map_err(|e| parse(path, e))?;
    let mut density = vec![0.0f32; cells.voxel_count()];
    let mut has_density = vec![false; cells.voxel_count()];
    for (at, value, level) in grid.iter() {
        for index in expand([at.x, at.y, at.z], level) {
            let [i, j, k] =
                in_domain(index, cells).ok_or_else(|| out_of_domain(density_grid, index))?;
            let n = i + cells.x as usize * (j + cells.y as usize * k);
            density[n] = value;
            has_density[n] = true;
        }
    }

    // Velocity: index (i, j, k) is the −x, −y and −z face of cell (i, j, k),
    // which is face index (i, j, k) of each axis's face grid.
    let grid = reader
        .read_grid::<[f32; 3]>(velocity_grid)
        .map_err(|e| parse(path, e))?;
    let face_dims = [Axis::X, Axis::Y, Axis::Z].map(|a| StaggeredField::face_dims(cells, a));
    let mut faces = face_dims.map(|d| vec![0.0f32; d.voxel_count()]);
    let mut has_velocity = vec![false; cells.voxel_count()];
    for (at, value, level) in grid.iter() {
        for index in expand([at.x, at.y, at.z], level) {
            let [i, j, k] =
                in_domain(index, cells).ok_or_else(|| out_of_domain(velocity_grid, index))?;
            for axis in 0..3 {
                let d = face_dims[axis];
                faces[axis][i + d.x as usize * (j + d.y as usize * k)] =
                    (value[axis] as f64 * dx / TIME_UNIT_S) as f32;
            }
            has_velocity[i + cells.x as usize * (j + cells.y as usize * k)] = true;
        }
    }

    Ok(CacheFrame {
        path: path.to_owned(),
        cells,
        density,
        faces,
        missing_velocity: missing_cells(&has_density, &has_velocity, cells),
    })
}

fn parse(path: &Path, e: vdb_rs::ParseError) -> CacheError {
    CacheError::Parse {
        path: path.to_owned(),
        message: e.to_string(),
    }
}

fn check_grid<R: std::io::Read + std::io::Seek>(
    reader: &VdbReader<R>,
    path: &Path,
    grid: &str,
    expected: &'static str,
) -> Result<(), CacheError> {
    let Some(descriptor) = reader.grid_descriptors.get(grid) else {
        let mut found = reader.available_grids();
        found.sort();
        return Err(CacheError::MissingGrid {
            path: path.to_owned(),
            grid: grid.to_owned(),
            found,
        });
    };
    if descriptor.grid_type != expected {
        return Err(CacheError::WrongType {
            path: path.to_owned(),
            grid: grid.to_owned(),
            expected,
            found: descriptor.grid_type.clone(),
        });
    }
    Ok(())
}

/// Every voxel index a value covers: one for a voxel, `scale³` for a tile,
/// from the tile's minimum corner.
fn expand(at: [f32; 3], level: VdbLevel) -> impl Iterator<Item = [i32; 3]> {
    let s = level.scale() as i32;
    let [x, y, z] = at.map(|c| c as i32);
    (0..s).flat_map(move |dk| {
        (0..s).flat_map(move |dj| (0..s).map(move |di| [x + di, y + dj, z + dk]))
    })
}

/// The index as `usize`s when it lies inside `cells`.
fn in_domain(index: [i32; 3], cells: FieldDims) -> Option<[usize; 3]> {
    let dims = [cells.x, cells.y, cells.z];
    let mut out = [0usize; 3];
    for a in 0..3 {
        if index[a] < 0 || index[a] as u32 >= dims[a] {
            return None;
        }
        out[a] = index[a] as usize;
    }
    Some(out)
}

/// The cells with density but no velocity, x-fastest. A count comparison
/// is not enough: velocity stored outside the smoke would hide a hole
/// inside it.
fn missing_cells(has_density: &[bool], has_velocity: &[bool], cells: FieldDims) -> Vec<[u32; 3]> {
    let (nx, ny) = (cells.x as usize, cells.y as usize);
    has_density
        .iter()
        .zip(has_velocity)
        .enumerate()
        .filter(|(_, (d, v))| **d && !**v)
        .map(|(n, _)| {
            [
                (n % nx) as u32,
                ((n / nx) % ny) as u32,
                (n / (nx * ny)) as u32,
            ]
        })
        .collect()
}

/// Check that every missing-velocity cell of `frame` is safe to read as 0,
/// by the collider mask `solid` (x-fastest over `cells`; empty for no
/// collider). Returns how many cells it accepted.
///
/// A cell is accepted only when, on every axis, the cell or its −axis
/// neighbour is solid: all three faces its value holds (the −x, −y and −z
/// faces) are then wall faces. Every face a metric reads lies between two
/// fluid cells by the same mask, so an accepted value is never read. A
/// neighbour outside the domain is not solid. The writer omits a velocity
/// of (0, 0, 0), which is what walls give (notes: Clipping); a missing cell
/// anywhere else could be lost data (notes: tiles can drop values from the
/// other grids).
pub fn check_coverage(
    frame: &CacheFrame,
    solid: &[bool],
    cells: FieldDims,
) -> Result<usize, CacheError> {
    let dims = [cells.x, cells.y, cells.z].map(i64::from);
    let is_solid = |c: [i64; 3]| {
        !solid.is_empty()
            && (0..3).all(|a| (0..dims[a]).contains(&c[a]))
            && solid[(c[0] + dims[0] * (c[1] + dims[1] * c[2])) as usize]
    };
    let walls_only = |c: [u32; 3]| {
        let c = c.map(i64::from);
        (0..3).all(|a| {
            let mut below = c;
            below[a] -= 1;
            is_solid(c) || is_solid(below)
        })
    };
    let unexplained: Vec<[u32; 3]> = frame
        .missing_velocity
        .iter()
        .copied()
        .filter(|&c| !walls_only(c))
        .collect();
    match unexplained.first() {
        None => Ok(frame.missing_velocity.len()),
        Some(&cell) => {
            let [i, j, k] = cell;
            Err(CacheError::MissingVelocity {
                path: frame.path.clone(),
                cell,
                density: frame.density[(i + cells.x * (j + cells.y * k)) as usize],
                count: unexplained.len(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{CacheError, CacheFrame, check_coverage, missing_cells};
    use elements_core::gpu::FieldDims;

    const CELLS: FieldDims = FieldDims { x: 4, y: 4, z: 4 };

    fn mask(solid: &[[u32; 3]]) -> Vec<bool> {
        let mut m = vec![false; 64];
        for c in solid {
            m[(c[0] + 4 * (c[1] + 4 * c[2])) as usize] = true;
        }
        m
    }

    #[test]
    fn a_cell_with_density_but_no_velocity_is_missing() {
        let mut d = vec![false; 64];
        let mut v = vec![false; 64];
        d[1 + 4 * (2 + 4 * 3)] = true;
        d[5] = true;
        v[5] = true;
        // Velocity elsewhere does not make up for it, although the counts match.
        v[0] = true;
        assert_eq!(missing_cells(&d, &v, CELLS), vec![[1, 2, 3]]);
    }

    #[test]
    fn full_coverage_misses_nothing() {
        let d = vec![true; 64];
        assert_eq!(missing_cells(&d, &d, CELLS), Vec::<[u32; 3]>::new());
    }

    /// A 4³ frame with density 0.5 everywhere and these cells missing.
    fn frame(missing: &[[u32; 3]]) -> CacheFrame {
        CacheFrame {
            path: PathBuf::from("frame.vdb"),
            cells: CELLS,
            density: vec![0.5; 64],
            faces: [vec![0.0; 80], vec![0.0; 80], vec![0.0; 80]],
            missing_velocity: missing.to_vec(),
        }
    }

    fn rejected(result: Result<usize, CacheError>) -> ([u32; 3], usize) {
        match result {
            Err(CacheError::MissingVelocity { cell, count, .. }) => (cell, count),
            other => panic!("expected MissingVelocity, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_solid_cell_is_accepted() {
        let solid = mask(&[[1, 1, 1]]);
        let f = frame(&[[1, 1, 1]]);
        assert_eq!(check_coverage(&f, &solid, CELLS).unwrap(), 1);
    }

    #[test]
    fn a_fluid_cell_whose_faces_are_all_walls_is_accepted() {
        // (1, 1, 1) is fluid, but its −x, −y and −z neighbours are solid.
        let solid = mask(&[[0, 1, 1], [1, 0, 1], [1, 1, 0]]);
        let f = frame(&[[1, 1, 1]]);
        assert_eq!(check_coverage(&f, &solid, CELLS).unwrap(), 1);
    }

    #[test]
    fn one_fluid_face_rejects_a_cell_next_to_the_collider() {
        // Walls on two axes; the third face, between fluid cells, is read.
        let walls = [[0, 1, 1], [1, 0, 1], [1, 1, 0]];
        for fluid in 0..3 {
            let others: Vec<[u32; 3]> = (0..3).filter(|&a| a != fluid).map(|a| walls[a]).collect();
            let f = frame(&[[1, 1, 1]]);
            assert_eq!(
                rejected(check_coverage(&f, &mask(&others), CELLS)),
                ([1, 1, 1], 1),
                "axis {fluid} fluid"
            );
        }
    }

    #[test]
    fn a_diagonal_neighbour_of_the_collider_is_rejected() {
        let solid = mask(&[[0, 0, 0]]);
        let f = frame(&[[0, 0, 0], [1, 1, 1], [2, 0, 0]]);
        assert_eq!(rejected(check_coverage(&f, &solid, CELLS)), ([1, 1, 1], 2));
    }

    #[test]
    fn a_neighbour_past_the_domain_edge_is_not_solid() {
        // (0, 1, 1)'s −x neighbour (−1, 1, 1) would wrap to (3, 0, 1).
        let solid = mask(&[[3, 0, 1], [0, 0, 1], [0, 1, 0]]);
        let f = frame(&[[0, 1, 1]]);
        assert_eq!(rejected(check_coverage(&f, &solid, CELLS)), ([0, 1, 1], 1));
    }

    #[test]
    fn without_a_collider_a_missing_cell_is_rejected() {
        let f = frame(&[[1, 1, 1]]);
        assert_eq!(rejected(check_coverage(&f, &[], CELLS)), ([1, 1, 1], 1));
    }

    #[test]
    fn no_missing_cells_is_fine_with_or_without_a_collider() {
        assert_eq!(check_coverage(&frame(&[]), &[], CELLS).unwrap(), 0);
        assert_eq!(
            check_coverage(&frame(&[]), &mask(&[[1, 1, 1]]), CELLS).unwrap(),
            0
        );
    }

    #[test]
    fn the_error_names_the_file_the_cell_and_its_density() {
        let e = check_coverage(&frame(&[[1, 2, 3]]), &[], CELLS).unwrap_err();
        let text = e.to_string();
        for part in ["frame.vdb", "(1, 2, 3)", "5e-1", "1 such cells"] {
            assert!(text.contains(part), "{text:?} lacks {part:?}");
        }
    }
}
