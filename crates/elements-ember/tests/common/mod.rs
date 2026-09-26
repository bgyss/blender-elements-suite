#![allow(dead_code)]

use elements_core::gpu::{Field, FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{Document, Graph, Timeline, TimelineConfig};

pub fn gpu() -> GpuContext {
    GpuContext::new_headless().expect("no GPU adapter available")
}

/// A document loaded once, stepped through a `Timeline` as many times as a
/// test needs. Every integration test that must build a graph, step it
/// frame by frame and read the output field back shares this session rather
/// than duplicating `Document::from_json` + `into_graph` + a `FieldPool`.
pub struct Session {
    pub gpu: GpuContext,
    pub pool: FieldPool,
    pub pipelines: PipelineCache,
    pub graph: Graph,
    pub dims: FieldDims,
}

impl Session {
    pub fn new(doc: &str) -> Self {
        let (graph, dims) = Document::from_json(doc)
            .unwrap()
            .into_graph(&elements_ember::registry())
            .unwrap();
        Self {
            gpu: gpu(),
            pool: FieldPool::new(),
            pipelines: PipelineCache::new(),
            graph,
            dims,
        }
    }

    /// Evaluate `frame` on `timeline` and read the output field back as bits,
    /// so callers can hash or compare it bit-exactly.
    pub fn density_bits(&mut self, timeline: &mut Timeline, frame: u32) -> Vec<u32> {
        let evaluated = timeline
            .goto(
                &self.graph,
                &self.gpu,
                &mut self.pool,
                &mut self.pipelines,
                self.dims,
                frame,
            )
            .unwrap();
        let bits = evaluated
            .value
            .as_field()
            .unwrap()
            .read_back(&self.gpu)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        evaluated.value.release_to(&mut self.pool);
        bits
    }
}

pub fn timeline(budget_bytes: u64) -> Timeline {
    Timeline::new(TimelineConfig {
        fps: 24.0,
        start_frame: 1,
        cache_budget_bytes: budget_bytes,
    })
}

/// Umbrella §4 and §6: frame 40 is bit-identical in order, after scrubbing
/// back and forth, and after eviction forced a recompute.
pub fn assert_doc_frame_40_is_bit_identical(doc: &str) {
    let mut s = Session::new(doc);

    let mut in_order = timeline(0);
    let mut reference = Vec::new();
    for frame in 1..=40 {
        reference = s.density_bits(&mut in_order, frame);
    }
    assert!(
        reference.iter().any(|&b| f32::from_bits(b) != 0.0),
        "the plume must exist"
    );

    let mut scrubbed = timeline(512 * 1024 * 1024);
    s.density_bits(&mut scrubbed, 40);
    s.density_bits(&mut scrubbed, 10);
    assert!(
        s.density_bits(&mut scrubbed, 40) == reference,
        "after scrubbing"
    );

    // About ten 16³ snapshots fit, so reaching 40 evicts most of them.
    let mut evicting = timeline(1024 * 1024);
    s.density_bits(&mut evicting, 40);
    s.density_bits(&mut evicting, 5);
    assert!(
        s.density_bits(&mut evicting, 40) == reference,
        "after eviction"
    );
}

/// Index of voxel `(i, j, k)` in an x-fastest array of `dims`.
pub fn index(dims: FieldDims, i: u32, j: u32, k: u32) -> usize {
    (i + dims.x * (j + dims.y * k)) as usize
}

/// A pooled R32Float field holding `values`.
pub fn upload(gpu: &GpuContext, pool: &mut FieldPool, dims: FieldDims, values: &[f32]) -> Field {
    let field = pool.acquire(gpu, dims, FieldFormat::R32Float).unwrap();
    field.write(gpu, values).unwrap();
    field
}

/// A deterministic, irregular test pattern with values in about [-1, 1].
pub fn pattern(dims: FieldDims, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(dims.voxel_count());
    for k in 0..dims.z {
        for j in 0..dims.y {
            for i in 0..dims.x {
                let h = (i * 73 + j * 151 + k * 283 + seed * 997) % 211;
                out.push(h as f32 / 105.0 - 1.0);
            }
        }
    }
    out
}

pub fn assert_close(gpu: &[f32], cpu: &[f32], tol: f32, what: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{what}: length");
    for (n, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            (g - c).abs() <= tol,
            "{what}: element {n}: gpu {g}, cpu {c}"
        );
    }
}

use elements_core::gpu::{Axis, StaggeredField};
use elements_ember::boundaries::DEFAULT_OPEN_MASK;

pub const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

pub fn face_dims(cells: FieldDims, axis: usize) -> FieldDims {
    StaggeredField::face_dims(cells, AXES[axis])
}

/// Mirrors `face_offset` in `common.wgsl`. Kept as a name distinct from
/// `grid_offset` only where a face-grid-only offset is meant; `grid_offset`
/// below is the general mirror used by everything else.
pub fn face_offset(axis: usize) -> [f32; 3] {
    let mut o = [0.5; 3];
    o[axis] = 0.0;
    o
}

/// Mirrors `is_open` in `common.wgsl`.
pub fn is_open(mask: u32, axis: usize, side: usize) -> bool {
    (mask >> (2 * axis + side)) & 1 == 1
}

/// Mirrors `is_wall` in `common.wgsl`.
pub fn is_wall_in(cells: FieldDims, mask: u32, axis: usize, i: u32) -> bool {
    let n = [cells.x, cells.y, cells.z][axis];
    if i == 0 {
        return !is_open(mask, axis, 0);
    }
    if i == n {
        return !is_open(mask, axis, 1);
    }
    false
}

/// `is_wall_in` with 2a's boundaries: walls everywhere but `+z`.
pub fn is_wall(cells: FieldDims, axis: usize, i: u32) -> bool {
    is_wall_in(cells, DEFAULT_OPEN_MASK, axis, i)
}

/// Mirrors `texel` in `common.wgsl`. `open` is `Some(mask)` for a cell
/// grid, which reads 0 beyond an open face; face grids pass `None` and clamp.
pub fn texel(data: &[f32], dims: FieldDims, open: Option<u32>, c: [i32; 3]) -> f32 {
    let size = [dims.x as i32, dims.y as i32, dims.z as i32];
    let mut q = c;
    for a in 0..3 {
        if c[a] < 0 {
            if open.is_some_and(|m| is_open(m, a, 0)) {
                return 0.0;
            }
            q[a] = 0;
        } else if c[a] >= size[a] {
            if open.is_some_and(|m| is_open(m, a, 1)) {
                return 0.0;
            }
            q[a] = size[a] - 1;
        }
    }
    data[index(dims, q[0] as u32, q[1] as u32, q[2] as u32)]
}

/// Mirrors `corners` in `common.wgsl`: the 8 texels around `p` and the
/// fractional position between them.
pub fn corners(
    data: &[f32],
    dims: FieldDims,
    open: Option<u32>,
    p: [f32; 3],
) -> ([f32; 8], [f32; 3]) {
    let size = [dims.x as f32, dims.y as f32, dims.z as f32];
    let mut i0 = [0i32; 3];
    let mut t = [0f32; 3];
    for a in 0..3 {
        let q = p[a].clamp(-1.0, size[a]);
        let f = q.floor();
        i0[a] = f as i32;
        t[a] = q - f;
    }
    let c = std::array::from_fn(|n| {
        texel(
            data,
            dims,
            open,
            [
                i0[0] + (n & 1) as i32,
                i0[1] + ((n >> 1) & 1) as i32,
                i0[2] + ((n >> 2) & 1) as i32,
            ],
        )
    });
    (c, t)
}

/// Mirrors `sample_grid` in `common.wgsl`.
pub fn sample_grid(data: &[f32], dims: FieldDims, open: Option<u32>, p: [f32; 3]) -> f32 {
    let (c, t) = corners(data, dims, open, p);
    let mix = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;
    let c00 = mix(c[0], c[1], t[0]);
    let c10 = mix(c[2], c[3], t[0]);
    let c01 = mix(c[4], c[5], t[0]);
    let c11 = mix(c[6], c[7], t[0]);
    mix(mix(c00, c10, t[1]), mix(c01, c11, t[1]), t[2])
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Mirrors `velocity_at` in `velocity.wgsl`.
pub fn velocity_at(faces: &[Vec<f32>; 3], cells: FieldDims, x: [f32; 3]) -> [f32; 3] {
    let mut v = [0.0; 3];
    for (a, out) in v.iter_mut().enumerate() {
        *out = sample_grid(&faces[a], face_dims(cells, a), None, sub(x, face_offset(a)));
    }
    v
}

/// How a backtrace steps: `backtrace` in `velocity.wgsl` takes one Euler
/// step where the uniform's `euler_faces` is set (velocity faces while fire
/// burns, 2b-4 spec §3.2 step 5) and the RK2 midpoint otherwise.
#[derive(Clone, Copy, Debug)]
pub enum Trace {
    Rk2,
    Euler,
}

/// Mirrors `backtrace` in `velocity.wgsl`. `k` is direction · h / dx; a
/// negative `k` traces forward in time.
fn trace(faces: &[Vec<f32>; 3], cells: FieldDims, x: [f32; 3], k: f32, how: Trace) -> [f32; 3] {
    match how {
        Trace::Rk2 => backtrace(faces, cells, x, k),
        Trace::Euler => {
            let v = velocity_at(faces, cells, x);
            [x[0] - k * v[0], x[1] - k * v[1], x[2] - k * v[2]]
        }
    }
}

/// The RK2 midpoint step of `backtrace` in `velocity.wgsl`.
fn backtrace(faces: &[Vec<f32>; 3], cells: FieldDims, x: [f32; 3], k: f32) -> [f32; 3] {
    let v = velocity_at(faces, cells, x);
    let mid = [
        x[0] - 0.5 * k * v[0],
        x[1] - 0.5 * k * v[1],
        x[2] - 0.5 * k * v[2],
    ];
    let v = velocity_at(faces, cells, mid);
    [x[0] - k * v[0], x[1] - k * v[1], x[2] - k * v[2]]
}

/// The grid an advection pass carries.
#[derive(Clone, Copy, Debug)]
pub enum Grid {
    Face(usize),
    Cell,
}

fn grid_dims(cells: FieldDims, grid: Grid) -> FieldDims {
    match grid {
        Grid::Face(a) => face_dims(cells, a),
        Grid::Cell => cells,
    }
}

fn grid_offset(grid: Grid) -> [f32; 3] {
    match grid {
        Grid::Face(a) => face_offset(a),
        Grid::Cell => [0.5; 3],
    }
}

fn grid_open(grid: Grid, mask: u32) -> Option<u32> {
    match grid {
        Grid::Face(_) => None,
        Grid::Cell => Some(mask),
    }
}

fn is_wall_texel(cells: FieldDims, mask: u32, grid: Grid, ijk: [u32; 3]) -> bool {
    match grid {
        Grid::Face(a) => is_wall_in(cells, mask, a, ijk[a]),
        Grid::Cell => false,
    }
}

/// Mirrors `pass_over` in `advect.wgsl`, with the RK2 backtrace.
pub fn cpu_advect(
    faces: &[Vec<f32>; 3],
    cells: FieldDims,
    mask: u32,
    grid: Grid,
    src: &[f32],
    k: f32,
    decay: f32,
) -> Vec<f32> {
    cpu_advect_traced(faces, cells, mask, grid, src, k, decay, Trace::Rk2)
}

/// `cpu_advect` with the backtrace `how`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_advect_traced(
    faces: &[Vec<f32>; 3],
    cells: FieldDims,
    mask: u32,
    grid: Grid,
    src: &[f32],
    k: f32,
    decay: f32,
    how: Trace,
) -> Vec<f32> {
    let d = grid_dims(cells, grid);
    let off = grid_offset(grid);
    let mut out = vec![0.0; d.voxel_count()];
    for kk in 0..d.z {
        for j in 0..d.y {
            for i in 0..d.x {
                if is_wall_texel(cells, mask, grid, [i, j, kk]) {
                    continue;
                }
                let x = [i as f32 + off[0], j as f32 + off[1], kk as f32 + off[2]];
                let b = trace(faces, cells, x, k, how);
                out[index(d, i, j, kk)] =
                    sample_grid(src, d, grid_open(grid, mask), sub(b, off)) * decay;
            }
        }
    }
    out
}

/// Mirrors `beyond_open` in `maccormack.wgsl`: a cell-grid position past
/// the last cell centre towards an open face.
fn beyond_open(cells: FieldDims, mask: u32, p: [f32; 3]) -> bool {
    let n = [cells.x, cells.y, cells.z];
    (0..3).any(|a| {
        (p[a] < 0.0 && is_open(mask, a, 0)) || (p[a] > n[a] as f32 - 1.0 && is_open(mask, a, 1))
    })
}

/// Mirrors `advect.wgsl`'s forward and backward passes and `maccormack.wgsl`,
/// with the RK2 backtrace.
pub fn cpu_maccormack(
    faces: &[Vec<f32>; 3],
    cells: FieldDims,
    mask: u32,
    grid: Grid,
    src: &[f32],
    k: f32,
    decay: f32,
) -> Vec<f32> {
    cpu_maccormack_traced(faces, cells, mask, grid, src, k, decay, Trace::Rk2)
}

/// `cpu_maccormack` with the backtrace `how` in every pass.
#[allow(clippy::too_many_arguments)]
pub fn cpu_maccormack_traced(
    faces: &[Vec<f32>; 3],
    cells: FieldDims,
    mask: u32,
    grid: Grid,
    src: &[f32],
    k: f32,
    decay: f32,
    how: Trace,
) -> Vec<f32> {
    let fwd = cpu_advect_traced(faces, cells, mask, grid, src, k, 1.0, how);
    let bwd = cpu_advect_traced(faces, cells, mask, grid, &fwd, -k, 1.0, how);
    let d = grid_dims(cells, grid);
    let off = grid_offset(grid);
    let mut out = vec![0.0; d.voxel_count()];
    for kk in 0..d.z {
        for j in 0..d.y {
            for i in 0..d.x {
                if is_wall_texel(cells, mask, grid, [i, j, kk]) {
                    continue;
                }
                let x = [i as f32 + off[0], j as f32 + off[1], kk as f32 + off[2]];
                let b = trace(faces, cells, x, k, how);
                let at = index(d, i, j, kk);
                // A cell whose trace reaches past an open face takes q̂.
                let ahead = trace(faces, cells, x, -k, how);
                if matches!(grid, Grid::Cell)
                    && (beyond_open(cells, mask, sub(b, off))
                        || beyond_open(cells, mask, sub(ahead, off)))
                {
                    out[at] = fwd[at] * decay;
                    continue;
                }
                let (c, _) = corners(src, d, grid_open(grid, mask), sub(b, off));
                let lo = c.iter().copied().fold(c[0], f32::min);
                let hi = c.iter().copied().fold(c[0], f32::max);
                let corrected = fwd[at] + 0.5 * (src[at] - bwd[at]);
                out[at] = corrected.max(lo).min(hi) * decay;
            }
        }
    }
    out
}

/// A staggered field holding `faces` (X, Y, Z order).
pub fn upload_staggered(
    gpu: &GpuContext,
    pool: &mut FieldPool,
    cells: FieldDims,
    faces: &[Vec<f32>; 3],
) -> StaggeredField {
    let [x, y, z] = std::array::from_fn(|a| upload(gpu, pool, face_dims(cells, a), &faces[a]));
    StaggeredField::from_faces(cells, [x, y, z]).unwrap()
}

pub fn read_staggered(gpu: &GpuContext, v: &StaggeredField) -> [Vec<f32>; 3] {
    std::array::from_fn(|a| v.face(AXES[a]).read_back(gpu).unwrap())
}

/// Smooth-ish velocity faces with values in about [-0.6, 0.6] m/s.
pub fn velocity_pattern(cells: FieldDims) -> [Vec<f32>; 3] {
    std::array::from_fn(|a| {
        pattern(face_dims(cells, a), a as u32 + 11)
            .iter()
            .map(|v| 0.6 * v)
            .collect()
    })
}

/// Mirrors `relax` in `pressure.wgsl`: red (even i+j+k) then black, per
/// iteration, solving ∇²p = div / scale.
pub fn cpu_red_black(
    p: &mut [f32],
    div: &[f32],
    cells: FieldDims,
    mask: u32,
    dx2: f32,
    scale: f32,
    iterations: u32,
) {
    let n = [cells.x as i32, cells.y as i32, cells.z as i32];
    for _ in 0..iterations {
        for colour in [0, 1] {
            for k in 0..cells.z {
                for j in 0..cells.y {
                    for i in 0..cells.x {
                        if (i + j + k) % 2 != colour {
                            continue;
                        }
                        let c = [i as i32, j as i32, k as i32];
                        let at =
                            |q: [i32; 3]| p[index(cells, q[0] as u32, q[1] as u32, q[2] as u32)];
                        let mut sum = 0.0f32;
                        let mut count = 0.0f32;
                        for a in 0..3 {
                            let mut lo = c;
                            lo[a] -= 1;
                            let mut hi = c;
                            hi[a] += 1;
                            if c[a] > 0 {
                                sum += at(lo);
                                count += 1.0;
                            } else if is_open(mask, a, 0) {
                                count += 1.0;
                            }
                            if c[a] < n[a] - 1 {
                                sum += at(hi);
                                count += 1.0;
                            } else if is_open(mask, a, 1) {
                                count += 1.0;
                            }
                        }
                        if count == 0.0 {
                            continue;
                        }
                        let rhs = dx2 * div[index(cells, i, j, k)] / scale;
                        p[index(cells, i, j, k)] = (sum - rhs) / count;
                    }
                }
            }
        }
    }
}

/// Max |div| over every cell of `faces`, computed on the CPU.
pub fn cpu_max_divergence(faces: &[Vec<f32>; 3], cells: FieldDims, dx: f32) -> f32 {
    let mut worst = 0.0f32;
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let (xd, yd, zd) = (
                    face_dims(cells, 0),
                    face_dims(cells, 1),
                    face_dims(cells, 2),
                );
                let d = (faces[0][index(xd, i + 1, j, k)] - faces[0][index(xd, i, j, k)]
                    + faces[1][index(yd, i, j + 1, k)]
                    - faces[1][index(yd, i, j, k)]
                    + faces[2][index(zd, i, j, k + 1)]
                    - faces[2][index(zd, i, j, k)])
                    / dx;
                worst = worst.max(d.abs());
            }
        }
    }
    worst
}

/// Mirrors `centre_velocity` in `curl.wgsl`: face velocities averaged to
/// the centre of cell `c`, with `c` clamped into the domain.
pub fn centre_velocity(faces: &[Vec<f32>; 3], cells: FieldDims, c: [i32; 3]) -> [f32; 3] {
    let n = [cells.x as i32, cells.y as i32, cells.z as i32];
    let q: [u32; 3] = std::array::from_fn(|a| c[a].clamp(0, n[a] - 1) as u32);
    std::array::from_fn(|a| {
        let d = face_dims(cells, a);
        let mut hi = q;
        hi[a] += 1;
        0.5 * (faces[a][index(d, q[0], q[1], q[2])] + faces[a][index(d, hi[0], hi[1], hi[2])])
    })
}

/// Mirrors `curl.wgsl`: ωx, ωy, ωz and |ω| per cell.
pub fn cpu_curl(faces: &[Vec<f32>; 3], cells: FieldDims, inv_dx: f32) -> [Vec<f32>; 4] {
    let mut out: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0; cells.voxel_count()]);
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let c = [i as i32, j as i32, k as i32];
                let diff = |a: usize| {
                    let mut p = c;
                    p[a] += 1;
                    let mut m = c;
                    m[a] -= 1;
                    let (vp, vm) = (
                        centre_velocity(faces, cells, p),
                        centre_velocity(faces, cells, m),
                    );
                    [vp[0] - vm[0], vp[1] - vm[1], vp[2] - vm[2]]
                };
                let (ddx, ddy, ddz) = (diff(0), diff(1), diff(2));
                let s = 0.5 * inv_dx;
                let w = [
                    s * (ddy[2] - ddz[1]),
                    s * (ddz[0] - ddx[2]),
                    s * (ddx[1] - ddy[0]),
                ];
                let at = index(cells, i, j, k);
                out[0][at] = w[0];
                out[1][at] = w[1];
                out[2][at] = w[2];
                out[3][at] = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
            }
        }
    }
    out
}

/// `velocity_pattern` with every solid-wall face set to zero, as advection leaves it.
pub fn walled_velocity_pattern(cells: FieldDims) -> [Vec<f32>; 3] {
    let mut faces = velocity_pattern(cells);
    for (a, face) in faces.iter_mut().enumerate() {
        let d = face_dims(cells, a);
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    if is_wall(cells, a, [i, j, k][a]) {
                        face[index(d, i, j, k)] = 0.0;
                    }
                }
            }
        }
    }
    faces
}

/// Mirrors `face_solid` in solid.wgsl: face `p` of `axis`'s grid touches a
/// solid cell (mask > 0.5) on either side.
pub fn face_solid_cpu(mask: &[f32], cells: FieldDims, axis: usize, p: [u32; 3]) -> bool {
    let n = [cells.x, cells.y, cells.z];
    let solid = |q: [i64; 3]| {
        (0..3).all(|a| q[a] >= 0 && q[a] < i64::from(n[a]))
            && mask[index(cells, q[0] as u32, q[1] as u32, q[2] as u32)] > 0.5
    };
    let mut below = p.map(i64::from);
    below[axis] -= 1;
    solid(below) || solid(p.map(i64::from))
}

/// A mask with a solid block of cells i in 4..7, j in 3..6 and k in 2..5.
pub fn block_mask(cells: FieldDims) -> Vec<f32> {
    let mut mask = vec![0.0; cells.voxel_count()];
    for k in 2..5 {
        for j in 3..6 {
            for i in 4..7 {
                mask[index(cells, i, j, k)] = 1.0;
            }
        }
    }
    mask
}
