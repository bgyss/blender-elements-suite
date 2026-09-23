mod common;

use common::*;
use elements_core::gpu::{ComputeBatch, FieldDims, FieldPool, PipelineCache};
use elements_ember::kernels::{StepConstants, Uniforms, confine, curl};

const CELLS: FieldDims = FieldDims { x: 8, y: 6, z: 5 };

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Mirrors `force` in `confine.wgsl`, at cell `c` (inside the domain).
fn force(omega: &[Vec<f32>; 4], c: [i32; 3], inv_dx: f32, confinement: f32) -> [f32; 3] {
    let n = [CELLS.x as i32, CELLS.y as i32, CELLS.z as i32];
    let mag = |p: [i32; 3]| {
        let q: [u32; 3] = std::array::from_fn(|a| p[a].clamp(0, n[a] - 1) as u32);
        omega[3][index(CELLS, q[0], q[1], q[2])]
    };
    let s = 0.5 * inv_dx;
    let g: [f32; 3] = std::array::from_fn(|a| {
        let mut p = c;
        p[a] += 1;
        let mut m = c;
        m[a] -= 1;
        s * (mag(p) - mag(m))
    });
    let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt() + 1e-6;
    let normal = [g[0] / len, g[1] / len, g[2] / len];
    let at = index(CELLS, c[0] as u32, c[1] as u32, c[2] as u32);
    let w = [omega[0][at], omega[1][at], omega[2][at]];
    cross(normal, w).map(|v| confinement * v)
}

#[test]
fn curl_matches_the_cpu_reference() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = StepConstants::new(CELLS, 0.1, 0.125);
    let faces = walled_velocity_pattern(CELLS);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let omega: [_; 4] =
        std::array::from_fn(|_| pool.acquire_zeroed(&gpu, &mut cache, CELLS).unwrap());
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    curl(
        &gpu,
        &mut cache,
        &mut batch,
        &u,
        &velocity,
        [&omega[0], &omega[1], &omega[2], &omega[3]],
        None,
    )
    .unwrap();
    batch.submit(&gpu).unwrap();
    let want = cpu_curl(&faces, CELLS, 1.0 / c.dx);
    for (n, field) in omega.iter().enumerate() {
        assert_close(
            &field.read_back(&gpu).unwrap(),
            &want[n],
            1e-4,
            &format!("omega {n}"),
        );
    }
}

/// Spec §4.4: u += h·ε·dx·(N × ω), averaged from the two cells each face
/// separates. Wall faces are left alone.
#[test]
fn confinement_matches_the_cpu_reference() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = StepConstants {
        vorticity: 3.0,
        ..StepConstants::new(CELLS, 0.1, 0.125)
    };
    let faces = walled_velocity_pattern(CELLS);
    let omega_values = cpu_curl(&faces, CELLS, 1.0 / c.dx);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let omega: [_; 4] = std::array::from_fn(|n| upload(&gpu, &mut pool, CELLS, &omega_values[n]));
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    confine(
        &gpu,
        &mut cache,
        &mut batch,
        &u,
        &velocity,
        [&omega[0], &omega[1], &omega[2], &omega[3]],
        None,
    )
    .unwrap();
    batch.submit(&gpu).unwrap();

    let got = read_staggered(&gpu, &velocity);
    let confinement = c.vorticity * c.dx;
    for a in 0..3 {
        let d = face_dims(CELLS, a);
        let n = [CELLS.x, CELLS.y, CELLS.z][a];
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let ijk = [i, j, k];
                    let at = index(d, i, j, k);
                    if is_wall(CELLS, a, ijk[a]) {
                        assert_eq!(got[a][at], faces[a][at], "wall face {a} {ijk:?}");
                        continue;
                    }
                    let p = [i as i32, j as i32, k as i32];
                    let mut f = 0.0;
                    let mut count = 0.0;
                    if ijk[a] > 0 {
                        let mut below = p;
                        below[a] -= 1;
                        f += force(&omega_values, below, 1.0 / c.dx, confinement)[a];
                        count += 1.0;
                    }
                    if ijk[a] < n {
                        f += force(&omega_values, p, 1.0 / c.dx, confinement)[a];
                        count += 1.0;
                    }
                    let want = faces[a][at] + c.h * f / count;
                    assert!(
                        (got[a][at] - want).abs() <= 1e-4,
                        "face {a} {ijk:?}: {} vs {want}",
                        got[a][at]
                    );
                }
            }
        }
    }
}
