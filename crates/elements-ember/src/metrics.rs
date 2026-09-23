//! Physical-quality metrics computed on the CPU from read-back fields.
//!
//! The same functions serve the validation scenes, the speed gate, and 2b's
//! Mantaflow benchmark (spec §5.4), so both solvers are measured by one piece
//! of code. Sums are in `f64`: a 128³ grid has two million cells.
//!
//! [`measure`] is the benchmark's per-frame entry point (2b-3 spec §4). Both
//! solvers give face velocities with the same convention, but Mantaflow's
//! cache stores velocity only where there is smoke, so every velocity metric
//! is taken over the same smoke mask for both (see [`Sample`] and
//! [`FrameMetrics`]).

use elements_core::gpu::{Axis, FieldDims, StaggeredField};

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

/// The X, Y and Z face grids' dims for a cell grid.
fn face_grids(cells: FieldDims) -> [FieldDims; 3] {
    [Axis::X, Axis::Y, Axis::Z].map(|a| StaggeredField::face_dims(cells, a))
}

/// The face stencil at cell (i, j, k): the net outflow through its six faces
/// over its volume, 1/s. This is the quantity each projection minimises.
fn face_divergence(
    faces: &[Vec<f32>; 3],
    [xd, yd, zd]: [FieldDims; 3],
    i: u32,
    j: u32,
    k: u32,
    dx: f64,
) -> f64 {
    (faces[0][at(xd, i + 1, j, k)] as f64 - faces[0][at(xd, i, j, k)] as f64
        + faces[1][at(yd, i, j + 1, k)] as f64
        - faces[1][at(yd, i, j, k)] as f64
        + faces[2][at(zd, i, j, k + 1)] as f64
        - faces[2][at(zd, i, j, k)] as f64)
        / dx
}

/// Divergence of a staggered velocity, from its X, Y and Z faces (x-fastest,
/// as `Field::read_back` returns them), with voxel edge `dx` metres.
pub fn divergence(faces: &[Vec<f32>; 3], cells: FieldDims, dx: f32) -> DivergenceStats {
    let [xd, yd, zd] = face_grids(cells);
    let dx = dx as f64;
    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let d = face_divergence(faces, [xd, yd, zd], i, j, k, dx);
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

/// Density above which a cell counts as smoke: Mantaflow's default
/// `clipping`, below which its cache stores no velocity (spec §4.1).
pub const SMOKE_THRESHOLD: f32 = 1e-6;

/// One frame of either solver's fields, read back to the CPU.
pub struct Sample<'a> {
    pub cells: FieldDims,
    /// Voxel edge, metres.
    pub dx: f64,
    /// x-fastest, at the cell dims.
    pub density: &'a [f32],
    /// Face velocities in m/s, x-fastest, each axis's grid one longer along
    /// that axis (`StaggeredField::face_dims`), as `Field::read_back` returns.
    pub faces: &'a [Vec<f32>; 3],
    /// One entry per cell, true inside a collider; empty when there is none.
    pub solid: &'a [bool],
}

/// Everything the benchmark reports about one frame (spec §4).
///
/// The velocity metrics cover only *measured* cells: those whose whole
/// 27-cell neighbourhood lies inside the domain, outside any collider, and
/// in smoke. The density metrics cover every cell.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FrameMetrics {
    /// Over measured cells, 1/s.
    pub divergence_max: f64,
    pub divergence_rms: f64,
    /// ½ Σ|u|² dV over measured cells, m⁵/s².
    pub kinetic_energy: f64,
    /// Σ|∇×u| dV over measured cells, m³/s.
    pub vorticity: f64,
    /// How many cells the velocity metrics covered.
    pub measured_cells: u64,
    /// Σ ρ dV over every cell, density × m³.
    pub mass: f64,
    /// Σ ρ dV over cell layers 0 … nz − 3, below the outflow plane (spec §4.3).
    pub mass_below: f64,
    /// Density-weighted mean height, metres.
    pub centroid_m: Option<f64>,
    /// Height below which 95% of the above-threshold density lies, metres.
    pub top_m: Option<f64>,
    /// Net upwind flux of density through the z-faces at index nz − 2,
    /// density × m³ / s; positive leaves the control volume.
    pub outflow_rate: f64,
}

/// Face velocities resampled to cell centres: along each axis, the mean of
/// a cell's two faces on that axis (spec §4.1). The result is x-fastest at
/// the cell dims, one array per axis.
pub fn cell_centred(faces: &[Vec<f32>; 3], cells: FieldDims) -> [Vec<f32>; 3] {
    let grids = face_grids(cells);
    std::array::from_fn(|axis| {
        let fd = grids[axis];
        let mut out = Vec::with_capacity(cells.voxel_count());
        for k in 0..cells.z {
            for j in 0..cells.y {
                for i in 0..cells.x {
                    let mut hi = [i, j, k];
                    hi[axis] += 1;
                    let lo = faces[axis][at(fd, i, j, k)];
                    let hi = faces[axis][at(fd, hi[0], hi[1], hi[2])];
                    out.push(0.5 * (lo + hi));
                }
            }
        }
        out
    })
}

/// Whether cell (i, j, k) is measured: it and its 26 neighbours all lie
/// inside the domain, none is solid, and all hold density above
/// [`SMOKE_THRESHOLD`]. That keeps every face and central difference a
/// velocity metric touches inside the region Mantaflow's cache has data for.
fn measured(sample: &Sample<'_>, i: u32, j: u32, k: u32) -> bool {
    let c = sample.cells;
    let inside = |p: u32, n: u32| p >= 1 && p + 2 <= n;
    if !(inside(i, c.x) && inside(j, c.y) && inside(k, c.z)) {
        return false;
    }
    for z in k - 1..=k + 1 {
        for y in j - 1..=j + 1 {
            for x in i - 1..=i + 1 {
                let n = at(c, x, y, z);
                if !sample.solid.is_empty() && sample.solid[n] {
                    return false;
                }
                if sample.density[n] <= SMOKE_THRESHOLD {
                    return false;
                }
            }
        }
    }
    true
}

/// Every metric of spec §4 for one frame.
///
/// # Panics
///
/// If the arrays do not match `cells` (with `solid` allowed to be empty),
/// or the grid has fewer than three layers, which the outflow plane at
/// z-face index nz − 2 needs.
pub fn measure(sample: &Sample<'_>) -> FrameMetrics {
    let c = sample.cells;
    let dx = sample.dx;
    let dv = dx * dx * dx;
    let grids = face_grids(c);
    let n = c.voxel_count();
    assert_eq!(sample.density.len(), n, "density does not match the cells");
    assert!(
        sample.solid.is_empty() || sample.solid.len() == n,
        "solid does not match the cells"
    );
    for (axis, fd) in grids.iter().enumerate() {
        assert_eq!(
            sample.faces[axis].len(),
            fd.voxel_count(),
            "axis {axis} faces do not match the cells"
        );
    }
    assert!(c.z >= 3, "the outflow plane needs at least three layers");

    // Velocity, over measured cells.
    let centre = cell_centred(sample.faces, c);
    let u = |axis: usize, i: u32, j: u32, k: u32| centre[axis][at(c, i, j, k)] as f64;
    let mut measured_cells = 0u64;
    let mut div_max = 0.0f64;
    let mut div_sq = 0.0f64;
    let mut energy = 0.0f64;
    let mut vorticity = 0.0f64;
    for k in 0..c.z {
        for j in 0..c.y {
            for i in 0..c.x {
                if !measured(sample, i, j, k) {
                    continue;
                }
                measured_cells += 1;
                let d = face_divergence(sample.faces, grids, i, j, k, dx);
                div_max = div_max.max(d.abs());
                div_sq += d * d;
                let speed_sq = (0..3).map(|a| u(a, i, j, k).powi(2)).sum::<f64>();
                energy += 0.5 * speed_sq * dv;
                // Central differences: ∂/∂x of component a at (i, j, k).
                let ddx = |a: usize| (u(a, i + 1, j, k) - u(a, i - 1, j, k)) / (2.0 * dx);
                let ddy = |a: usize| (u(a, i, j + 1, k) - u(a, i, j - 1, k)) / (2.0 * dx);
                let ddz = |a: usize| (u(a, i, j, k + 1) - u(a, i, j, k - 1)) / (2.0 * dx);
                let curl = [ddy(2) - ddz(1), ddz(0) - ddx(2), ddx(1) - ddy(0)];
                let curl_mag = curl.iter().map(|w| w * w).sum::<f64>().sqrt();
                vorticity += curl_mag * dv;
            }
        }
    }
    let divergence_rms = if measured_cells == 0 {
        0.0
    } else {
        (div_sq / measured_cells as f64).sqrt()
    };

    // Density, over every cell. Per-layer sums serve the mass, the control
    // volume below the outflow plane, and the plume top.
    let rho = |i: u32, j: u32, k: u32| sample.density[at(c, i, j, k)] as f64;
    let peak = sample.density.iter().fold(0.0f64, |m, &r| m.max(r as f64));
    let cut = 0.01 * peak;
    let mut layer_mass = vec![0.0f64; c.z as usize];
    let mut layer_above_cut = vec![0.0f64; c.z as usize];
    for k in 0..c.z {
        for j in 0..c.y {
            for i in 0..c.x {
                let r = rho(i, j, k);
                layer_mass[k as usize] += r;
                if r >= cut {
                    layer_above_cut[k as usize] += r;
                }
            }
        }
    }
    let mass = layer_mass.iter().sum::<f64>() * dv;
    let mass_below = layer_mass[..(c.z - 2) as usize].iter().sum::<f64>() * dv;
    let top_m = (peak > 0.0).then(|| {
        let total: f64 = layer_above_cut.iter().sum();
        let mut cumulative = 0.0;
        let mut top = (c.z as f64 - 0.5) * dx;
        for (k, r) in layer_above_cut.iter().enumerate() {
            cumulative += r;
            if cumulative >= 0.95 * total {
                top = (k as f64 + 0.5) * dx;
                break;
            }
        }
        top
    });

    // Net upwind flux through the z-faces at index nz − 2.
    let plane = c.z - 2;
    let mut flux = 0.0f64;
    for j in 0..c.y {
        for i in 0..c.x {
            let w = sample.faces[2][at(grids[2], i, j, plane)] as f64;
            let upwind = if w > 0.0 {
                rho(i, j, plane - 1)
            } else {
                rho(i, j, plane)
            };
            flux += upwind * w;
        }
    }

    FrameMetrics {
        divergence_max: div_max,
        divergence_rms,
        kinetic_energy: energy,
        vorticity,
        measured_cells,
        mass,
        mass_below,
        centroid_m: centroid_z(sample.density, c).map(|z| z * dx),
        top_m,
        outflow_rate: flux * dx * dx,
    }
}

/// Mass drift at every frame index from `from` (0-based into the series),
/// spec §4.3: (M(n) + outflow integrated from `from` to n) − M(from), where
/// M is [`FrameMetrics::mass_below`] and the outflow rates are integrated
/// over `frame_seconds` per frame with the trapezoid rule. Entries before
/// `from` are 0.
///
/// # Panics
///
/// If `mass` and `outflow_rate` differ in length.
pub fn drift(mass: &[f64], outflow_rate: &[f64], frame_seconds: f64, from: usize) -> Vec<f64> {
    assert_eq!(
        mass.len(),
        outflow_rate.len(),
        "one outflow rate per frame's mass"
    );
    let mut out = vec![0.0; mass.len()];
    let mut outflow = 0.0;
    for n in from..mass.len() {
        if n > from {
            outflow += 0.5 * (outflow_rate[n - 1] + outflow_rate[n]) * frame_seconds;
        }
        out[n] = mass[n] + outflow - mass[from];
    }
    out
}
