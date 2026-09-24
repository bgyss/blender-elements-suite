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
//!   even where there is smoke. Against a collider that happens at cells
//!   whose −x, −y and −z faces are all walls. The reader lists the cells that
//!   have density but no velocity, and the caller decides whether each is
//!   such a cell (`unexplained_missing`).

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
    pub cells: FieldDims,
    /// x-fastest, at the cell dims; 0 where the cache stores nothing.
    pub density: Vec<f32>,
    /// Face velocities in m/s, x-fastest, each axis one longer along itself
    /// (`StaggeredField::face_dims`), as `metrics::Sample::faces` expects.
    /// Faces the cache does not store, including every face at index n, are 0.
    pub faces: [Vec<f32>; 3],
    /// Cells that store density but no velocity, in x-fastest order. Their
    /// faces read as 0 in `faces`.
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

/// The missing-velocity cells that are neither a collider cell nor one of a
/// collider cell's 26 neighbours, by `solid` (x-fastest over `cells`; empty
/// for no collider).
///
/// Next to a collider, a cell's −x, −y and −z faces can all be walls, so its
/// stored velocity is (0, 0, 0), which the writer leaves out because it
/// equals the background (notes: Clipping). Reading it as 0 is then its true
/// value. Anywhere else a missing value is lost data (notes: tiles can drop
/// values from the other grids).
pub fn unexplained_missing(
    missing: &[[u32; 3]],
    solid: &[bool],
    cells: FieldDims,
) -> Vec<[u32; 3]> {
    let dims = [cells.x, cells.y, cells.z].map(i64::from);
    let is_solid = |c: [i64; 3]| {
        (0..3).all(|a| (0..dims[a]).contains(&c[a]))
            && solid[(c[0] + dims[0] * (c[1] + dims[1] * c[2])) as usize]
    };
    let near_collider = |c: [u32; 3]| {
        let c = c.map(i64::from);
        (-1..=1).any(|dk| {
            (-1..=1).any(|dj| (-1..=1).any(|di| is_solid([c[0] + di, c[1] + dj, c[2] + dk])))
        })
    };
    if solid.is_empty() {
        return missing.to_vec();
    }
    missing
        .iter()
        .copied()
        .filter(|&c| !near_collider(c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{missing_cells, unexplained_missing};
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

    #[test]
    fn a_collider_cell_or_its_neighbour_is_explained() {
        let solid = mask(&[[1, 1, 1]]);
        // The cell itself, a face neighbour and a corner neighbour.
        let missing = [[1, 1, 1], [2, 1, 1], [2, 2, 2], [0, 0, 0]];
        assert_eq!(
            unexplained_missing(&missing, &solid, CELLS),
            Vec::<[u32; 3]>::new()
        );
    }

    #[test]
    fn a_cell_two_away_from_the_collider_is_unexplained() {
        let solid = mask(&[[0, 0, 0]]);
        let missing = [[1, 1, 1], [2, 0, 0], [0, 3, 0]];
        assert_eq!(
            unexplained_missing(&missing, &solid, CELLS),
            vec![[2, 0, 0], [0, 3, 0]]
        );
    }

    #[test]
    fn without_a_collider_every_missing_cell_is_unexplained() {
        let missing = [[1, 1, 1]];
        assert_eq!(unexplained_missing(&missing, &[], CELLS), vec![[1, 1, 1]]);
    }

    #[test]
    fn a_neighbour_past_the_domain_edge_is_not_solid() {
        // (3, 3, 3)'s neighbours past the edge must not wrap to (0, ...).
        let solid = mask(&[[0, 3, 3]]);
        assert_eq!(
            unexplained_missing(&[[3, 3, 3]], &solid, CELLS),
            vec![[3, 3, 3]]
        );
    }
}
