#![allow(dead_code)]

use elements_core::gpu::{Field, FieldDims, FieldFormat, FieldPool, GpuContext};

pub fn gpu() -> GpuContext {
    GpuContext::new_headless().expect("no GPU adapter available")
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

pub const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

pub fn face_dims(cells: FieldDims, axis: usize) -> FieldDims {
    StaggeredField::face_dims(cells, AXES[axis])
}

/// Mirrors `face_offset` in `common.wgsl`.
pub fn face_offset(axis: usize) -> [f32; 3] {
    let mut o = [0.5; 3];
    o[axis] = 0.0;
    o
}

/// Mirrors `is_wall` in `common.wgsl`.
pub fn is_wall(cells: FieldDims, axis: usize, i: u32) -> bool {
    let n = [cells.x, cells.y, cells.z][axis];
    if axis == 2 && i == n {
        return false;
    }
    i == 0 || i == n
}

/// Mirrors `trilinear` in `common.wgsl`, including its edge clamp.
pub fn trilinear(data: &[f32], dims: FieldDims, p: [f32; 3]) -> f32 {
    let size = [dims.x as i32, dims.y as i32, dims.z as i32];
    let mut i0 = [0i32; 3];
    let mut i1 = [0i32; 3];
    let mut t = [0f32; 3];
    for a in 0..3 {
        let last = size[a] - 1;
        let q = p[a].clamp(0.0, last as f32);
        i0[a] = q.floor() as i32;
        i1[a] = (i0[a] + 1).min(last);
        t[a] = q - i0[a] as f32;
    }
    let at = |x: i32, y: i32, z: i32| data[index(dims, x as u32, y as u32, z as u32)];
    let mix = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;
    let c00 = mix(at(i0[0], i0[1], i0[2]), at(i1[0], i0[1], i0[2]), t[0]);
    let c10 = mix(at(i0[0], i1[1], i0[2]), at(i1[0], i1[1], i0[2]), t[0]);
    let c01 = mix(at(i0[0], i0[1], i1[2]), at(i1[0], i0[1], i1[2]), t[0]);
    let c11 = mix(at(i0[0], i1[1], i1[2]), at(i1[0], i1[1], i1[2]), t[0]);
    mix(mix(c00, c10, t[1]), mix(c01, c11, t[1]), t[2])
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Mirrors `velocity_at` in `velocity.wgsl`.
pub fn velocity_at(faces: &[Vec<f32>; 3], cells: FieldDims, x: [f32; 3]) -> [f32; 3] {
    let mut v = [0.0; 3];
    for (a, out) in v.iter_mut().enumerate() {
        *out = trilinear(&faces[a], face_dims(cells, a), sub(x, face_offset(a)));
    }
    v
}

fn backtrace(faces: &[Vec<f32>; 3], cells: FieldDims, x: [f32; 3], h_inv_dx: f32) -> [f32; 3] {
    let v = velocity_at(faces, cells, x);
    [
        x[0] - v[0] * h_inv_dx,
        x[1] - v[1] * h_inv_dx,
        x[2] - v[2] * h_inv_dx,
    ]
}

/// Mirrors `advect_scalar.wgsl`.
pub fn cpu_advect_scalar(
    faces: &[Vec<f32>; 3],
    cells: FieldDims,
    src: &[f32],
    h_inv_dx: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; cells.voxel_count()];
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let x = [i as f32 + 0.5, j as f32 + 0.5, k as f32 + 0.5];
                let b = backtrace(faces, cells, x, h_inv_dx);
                out[index(cells, i, j, k)] = trilinear(src, cells, sub(b, [0.5; 3]));
            }
        }
    }
    out
}

/// Mirrors `advect_velocity.wgsl` for all three faces.
pub fn cpu_advect_velocity(
    faces: &[Vec<f32>; 3],
    cells: FieldDims,
    h_inv_dx: f32,
) -> [Vec<f32>; 3] {
    std::array::from_fn(|a| {
        let d = face_dims(cells, a);
        let mut out = vec![0.0; d.voxel_count()];
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    if is_wall(cells, a, [i, j, k][a]) {
                        continue;
                    }
                    let x = [i as f32, j as f32, k as f32];
                    let off = face_offset(a);
                    let x = [x[0] + off[0], x[1] + off[1], x[2] + off[2]];
                    let b = backtrace(faces, cells, x, h_inv_dx);
                    out[index(d, i, j, k)] = trilinear(&faces[a], d, sub(b, off));
                }
            }
        }
        out
    })
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

/// Mirrors `relax` in `pressure.wgsl`: red (even i+j+k) then black, per iteration.
pub fn cpu_red_black(phi: &mut [f32], div: &[f32], cells: FieldDims, dx2: f32, iterations: u32) {
    let (nx, ny, nz) = (cells.x, cells.y, cells.z);
    for _ in 0..iterations {
        for colour in [0, 1] {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        if (i + j + k) % 2 != colour {
                            continue;
                        }
                        let mut sum = 0.0f32;
                        let mut count = 0.0f32;
                        if i > 0 {
                            sum += phi[index(cells, i - 1, j, k)];
                            count += 1.0;
                        }
                        if i < nx - 1 {
                            sum += phi[index(cells, i + 1, j, k)];
                            count += 1.0;
                        }
                        if j > 0 {
                            sum += phi[index(cells, i, j - 1, k)];
                            count += 1.0;
                        }
                        if j < ny - 1 {
                            sum += phi[index(cells, i, j + 1, k)];
                            count += 1.0;
                        }
                        if k > 0 {
                            sum += phi[index(cells, i, j, k - 1)];
                            count += 1.0;
                        }
                        if k < nz - 1 {
                            sum += phi[index(cells, i, j, k + 1)];
                        }
                        count += 1.0;
                        phi[index(cells, i, j, k)] =
                            (sum - dx2 * div[index(cells, i, j, k)]) / count;
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
