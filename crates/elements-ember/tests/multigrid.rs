mod common;

use common::*;
use elements_core::gpu::{
    ComputeBatch, Field, FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache,
};
use elements_ember::boundaries::DEFAULT_OPEN_MASK;
use elements_ember::kernels::multigrid::{Hierarchy, residual, restrict_mask, v_cycles};
use elements_ember::kernels::{StepConstants, Uniforms, pressure};

const H: f32 = 0.1;

/// The GPU residual r = div − h·L(p), read back.
fn residual_values(
    gpu: &GpuContext,
    c: &StepConstants,
    p: &Field,
    div: &Field,
    mask: Option<&Field>,
) -> Vec<f32> {
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let out = pool.acquire(gpu, c.cells, FieldFormat::R32Float).unwrap();
    let u = Uniforms::new(gpu, c).unwrap();
    let mut batch = ComputeBatch::new();
    residual(gpu, &mut cache, &mut batch, &u, p, div, &out, mask).unwrap();
    batch.submit(gpu).unwrap();
    out.read_back(gpu).unwrap()
}

/// RMS of the residual r = div − h·L(p) over fluid cells, from the GPU
/// residual kernel.
fn residual_rms(
    gpu: &GpuContext,
    c: &StepConstants,
    p: &Field,
    div: &Field,
    mask: Option<&Field>,
) -> f64 {
    let r = residual_values(gpu, c, p, div, mask);
    let solid = mask.map(|m| m.read_back(gpu).unwrap());
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for (i, v) in r.iter().enumerate() {
        if solid.as_ref().is_some_and(|s| s[i] > 0.5) {
            continue;
        }
        sum += f64::from(*v) * f64::from(*v);
        n += 1;
    }
    (sum / n as f64).sqrt()
}

/// A right-hand side with zero mean, so closed-domain problems are solvable.
fn rhs(cells: FieldDims) -> Vec<f32> {
    let mut v = pattern(cells, 3);
    let m = v.iter().sum::<f32>() / v.len() as f32;
    v.iter_mut().for_each(|x| *x -= m);
    v
}

/// 1.0 inside a sphere of `radius` cells at the grid's centre.
fn sphere_mask(cells: FieldDims, radius: f32) -> Vec<f32> {
    let centre = [
        cells.x as f32 / 2.0,
        cells.y as f32 / 2.0,
        cells.z as f32 / 2.0,
    ];
    let mut out = vec![0.0; cells.voxel_count()];
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let d = [
                    i as f32 + 0.5 - centre[0],
                    j as f32 + 0.5 - centre[1],
                    k as f32 + 0.5 - centre[2],
                ];
                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < radius * radius {
                    out[index(cells, i, j, k)] = 1.0;
                }
            }
        }
    }
    out
}

/// h·L(p) on the CPU with the smoother's neighbour rules: a fluid neighbour
/// counts with its value, beyond an open face 0 counts, beyond a wall or a
/// solid neighbour nothing does. Solid cells get 0.
fn cpu_h_laplacian(
    p: &[f32],
    solid: &[f32],
    cells: FieldDims,
    mask: u32,
    dx: f32,
    h: f32,
) -> Vec<f32> {
    let n = [cells.x as i32, cells.y as i32, cells.z as i32];
    let at = |q: [i32; 3]| index(cells, q[0] as u32, q[1] as u32, q[2] as u32);
    let mut out = vec![0.0; p.len()];
    for k in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                let c = [i, j, k];
                if solid[at(c)] > 0.5 {
                    continue;
                }
                let mut sum = 0.0f32;
                let mut count = 0.0f32;
                for a in 0..3 {
                    for side in 0..2 {
                        let mut q = c;
                        q[a] += if side == 0 { -1 } else { 1 };
                        if q[a] < 0 || q[a] >= n[a] {
                            if is_open(mask, a, side) {
                                count += 1.0;
                            }
                        } else if solid[at(q)] <= 0.5 {
                            sum += p[at(q)];
                            count += 1.0;
                        }
                    }
                }
                out[at(c)] = h * ((sum - count * p[at(c)]) / (dx * dx));
            }
        }
    }
    out
}

#[test]
fn the_residual_kernel_is_zero_for_an_exact_solution() {
    let gpu = gpu();
    let cells = FieldDims::new(16, 16, 16);
    // p = i² (cell units): every stencil sum is an exact integer, so div
    // built on the CPU with the same arithmetic leaves no rounding behind.
    let mut p_values = vec![0.0; cells.voxel_count()];
    for k in 0..16 {
        for j in 0..16 {
            for i in 0..16 {
                p_values[index(cells, i, j, k)] = (i * i) as f32 + (j % 3) as f32;
            }
        }
    }
    // A 4³ solid block for the last case, off the centre.
    let mut block = vec![0.0; cells.voxel_count()];
    for k in 4..8 {
        for j in 6..10 {
            for i in 3..7 {
                block[index(cells, i, j, k)] = 1.0;
            }
        }
    }
    let none = vec![0.0; cells.voxel_count()];
    for (name, mask, solid) in [
        ("closed", 0u32, false),
        ("open top", DEFAULT_OPEN_MASK, false),
        ("+x and -y open, solid block", 0b000110, true),
    ] {
        let mut pool = FieldPool::new();
        let c = StepConstants {
            open_mask: mask,
            has_solids: solid,
            ..StepConstants::new(cells, H, 0.5)
        };
        let solid_values = if solid { &block } else { &none };
        let mut p_case = p_values.clone();
        for (v, s) in p_case.iter_mut().zip(solid_values) {
            if *s > 0.5 {
                *v = 0.0;
            }
        }
        let div_values = cpu_h_laplacian(&p_case, solid_values, cells, mask, c.dx, c.h);
        let p = upload(&gpu, &mut pool, cells, &p_case);
        let div = upload(&gpu, &mut pool, cells, &div_values);
        let mask_field = solid.then(|| upload(&gpu, &mut pool, cells, &block));
        let r = residual_values(&gpu, &c, &p, &div, mask_field.as_ref());
        let worst = r.iter().fold(0.0f32, |w, v| w.max(v.abs()));
        assert!(worst < 1e-5, "{name}: max |r| = {worst}");
    }
}

#[test]
fn a_v_cycle_cuts_the_residual_tenfold_in_every_boundary_setting() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    for (name, mask, sphere) in [
        ("closed", 0u32, false),
        ("open top", DEFAULT_OPEN_MASK, false),
        ("open top, sphere", DEFAULT_OPEN_MASK, true),
    ] {
        let cells = FieldDims::new(64, 64, 64);
        let mut pool = FieldPool::new();
        let c = StepConstants {
            open_mask: mask,
            has_solids: sphere,
            ..StepConstants::new(cells, H, 1.0 / 32.0)
        };
        let mut div_values = rhs(cells);
        let solid_field = sphere.then(|| {
            let solid = sphere_mask(cells, 10.0);
            for (d, s) in div_values.iter_mut().zip(&solid) {
                if *s > 0.5 {
                    *d = 0.0;
                }
            }
            upload(&gpu, &mut pool, cells, &solid)
        });
        let div = upload(&gpu, &mut pool, cells, &div_values);
        let p = upload(&gpu, &mut pool, cells, &vec![0.0; cells.voxel_count()]);
        let r0 = residual_rms(&gpu, &c, &p, &div, solid_field.as_ref());

        let mut batch = ComputeBatch::new();
        let h = Hierarchy::new(
            &gpu,
            &mut cache,
            &mut batch,
            &mut pool,
            &c,
            solid_field.as_ref(),
        )
        .unwrap();
        v_cycles(&gpu, &mut cache, &mut batch, &h, &p, &div, 1).unwrap();
        batch.submit(&gpu).unwrap();
        let r1 = residual_rms(&gpu, &c, &p, &div, solid_field.as_ref());
        let mut batch = ComputeBatch::new();
        v_cycles(&gpu, &mut cache, &mut batch, &h, &p, &div, 1).unwrap();
        batch.submit(&gpu).unwrap();
        let r2 = residual_rms(&gpu, &c, &p, &div, solid_field.as_ref());
        h.release(&mut pool);
        println!(
            "{name}: r0 {r0:.6e} -> r1 {r1:.6e} -> r2 {r2:.6e}, factors {:.2}, {:.2}",
            r0 / r1,
            r1 / r2
        );
        assert!(r1 < r0 / 10.0, "{name}: {r0} -> {r1}");
        // On noise the first cycle is mostly the smoother's work, so it would
        // pass even with a weak coarse correction. Two cycles must still
        // average 10×, which needs the coarse levels.
        assert!(r2 < r0 / 100.0, "{name}: {r0} -> {r1} -> {r2}");
    }
}

#[test]
fn v_cycles_beat_the_same_cost_of_gauss_seidel() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    let mut pool = FieldPool::new();
    let cells = FieldDims::new(128, 128, 128);
    let c = StepConstants::new(cells, H, 1.0 / 64.0);
    // Smooth, like a plume's divergence: Gauss–Seidel's weak case. On
    // white noise it is nearly as good as a V-cycle (75× apart, not 100×).
    let div_values = smooth_rhs(cells);
    let zeros = vec![0.0; cells.voxel_count()];
    let div = upload(&gpu, &mut pool, cells, &div_values);

    let p_gs = upload(&gpu, &mut pool, cells, &zeros);
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    pressure(&gpu, &mut cache, &mut batch, &u, &p_gs, &div, 40, 40, None).unwrap();
    batch.submit(&gpu).unwrap();
    let r_gs = residual_rms(&gpu, &c, &p_gs, &div, None);

    let p_mg = upload(&gpu, &mut pool, cells, &zeros);
    let mut batch = ComputeBatch::new();
    let h = Hierarchy::new(&gpu, &mut cache, &mut batch, &mut pool, &c, None).unwrap();
    v_cycles(&gpu, &mut cache, &mut batch, &h, &p_mg, &div, 4).unwrap();
    batch.submit(&gpu).unwrap();
    let r_mg = residual_rms(&gpu, &c, &p_mg, &div, None);
    h.release(&mut pool);

    println!(
        "40 red-black: {r_gs:.6e}; 4 V-cycles: {r_mg:.6e}; ratio {:.1}",
        r_gs / r_mg
    );
    assert!(r_mg * 100.0 < r_gs, "V-cycles {r_mg}, Gauss–Seidel {r_gs}");
}

#[test]
fn a_hierarchy_stops_at_two_cells_and_returns_every_field() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    for (cells, depth, solids) in [
        (FieldDims::new(128, 128, 128), 7, false),
        (FieldDims::new(256, 256, 256), 8, false),
        // 40×24×16 → 20×12×8 → 10×6×4 → 5×3×2.
        (FieldDims::new(40, 24, 16), 4, false),
        (FieldDims::new(64, 64, 64), 6, true),
    ] {
        // The fine mask comes from its own pool, so `pool` holds only what
        // the hierarchy acquired.
        let mut mask_pool = FieldPool::new();
        let mask = solids.then(|| upload(&gpu, &mut mask_pool, cells, &sphere_mask(cells, 10.0)));
        let mut pool = FieldPool::new();
        let c = StepConstants {
            has_solids: solids,
            ..StepConstants::new(cells, H, 1.0)
        };
        let mut batch = ComputeBatch::new();
        let h = Hierarchy::new(&gpu, &mut cache, &mut batch, &mut pool, &c, mask.as_ref()).unwrap();
        batch.submit(&gpu).unwrap();
        assert_eq!(h.depth(), depth, "{cells:?}");
        let acquired = pool.acquisitions();
        assert!(acquired > 0, "{cells:?}: nothing acquired");
        assert_eq!(
            pool.pooled_count(),
            0,
            "{cells:?}: fields free before release"
        );
        h.release(&mut pool);
        assert_eq!(
            pool.pooled_count() as u64,
            acquired,
            "{cells:?}: fields returned"
        );
    }
}

#[test]
fn a_coarse_cell_is_solid_only_when_all_its_children_are() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    let mut pool = FieldPool::new();
    let fine = FieldDims::new(4, 4, 4);
    let coarse = FieldDims::new(2, 2, 2);
    let mut values = vec![0.0; fine.voxel_count()];
    // Coarse cell (0, 0, 0): all eight children solid.
    for k in 0..2 {
        for j in 0..2 {
            for i in 0..2 {
                values[index(fine, i, j, k)] = 1.0;
            }
        }
    }
    // Coarse cell (1, 1, 1): its lower four children solid.
    for j in 2..4 {
        for i in 2..4 {
            values[index(fine, i, j, 2)] = 1.0;
        }
    }
    let mask = upload(&gpu, &mut pool, fine, &values);
    let out = pool.acquire(&gpu, coarse, FieldFormat::R32Float).unwrap();
    let c = StepConstants {
        has_solids: true,
        ..StepConstants::new(coarse, H, 2.0)
    };
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    restrict_mask(&gpu, &mut cache, &mut batch, &u, &mask, &out).unwrap();
    batch.submit(&gpu).unwrap();
    let got = out.read_back(&gpu).unwrap();
    let mut want = vec![0.0; coarse.voxel_count()];
    want[index(coarse, 0, 0, 0)] = 1.0;
    assert_eq!(got, want);
}

/// A smooth, plume-like right-hand side: a Gaussian blob low in the domain
/// minus its mean, so it is solvable in a closed domain too.
fn smooth_rhs(cells: FieldDims) -> Vec<f32> {
    let n = [cells.x as f32, cells.y as f32, cells.z as f32];
    let centre = [n[0] / 2.0, n[1] / 2.0, n[2] / 4.0];
    let sigma = n[0] / 8.0;
    let mut v = vec![0.0; cells.voxel_count()];
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let d = [
                    i as f32 + 0.5 - centre[0],
                    j as f32 + 0.5 - centre[1],
                    k as f32 + 0.5 - centre[2],
                ];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                v[index(cells, i, j, k)] = (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }
    let m = v.iter().sum::<f32>() / v.len() as f32;
    v.iter_mut().for_each(|x| *x -= m);
    v
}

/// A smooth right-hand side is what the pressure solve sees, and where the
/// coarse levels do the work: open faces must sit in the same place on every
/// level, and the coarsest solve must converge, or the cycle stalls.
#[test]
fn v_cycles_converge_steadily_on_a_smooth_right_hand_side() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    for (name, n, mask, sphere, floor) in [
        ("closed", 64u32, 0u32, false, 4.0),
        ("open top", 64, DEFAULT_OPEN_MASK, false, 4.0),
        ("open top, sphere", 64, DEFAULT_OPEN_MASK, true, 3.0),
        ("open top, 128³", 128, DEFAULT_OPEN_MASK, false, 4.0),
    ] {
        let cells = FieldDims::new(n, n, n);
        let mut pool = FieldPool::new();
        let c = StepConstants {
            open_mask: mask,
            has_solids: sphere,
            ..StepConstants::new(cells, H, 2.0 / n as f32)
        };
        let mut div_values = smooth_rhs(cells);
        let solid_field = sphere.then(|| {
            let solid = sphere_mask(cells, 10.0);
            for (d, s) in div_values.iter_mut().zip(&solid) {
                if *s > 0.5 {
                    *d = 0.0;
                }
            }
            upload(&gpu, &mut pool, cells, &solid)
        });
        let div = upload(&gpu, &mut pool, cells, &div_values);
        let p = upload(&gpu, &mut pool, cells, &vec![0.0; cells.voxel_count()]);
        let mut batch = ComputeBatch::new();
        let h = Hierarchy::new(
            &gpu,
            &mut cache,
            &mut batch,
            &mut pool,
            &c,
            solid_field.as_ref(),
        )
        .unwrap();
        batch.submit(&gpu).unwrap();
        let r0 = residual_rms(&gpu, &c, &p, &div, solid_field.as_ref());
        let mut r = r0;
        let mut factors = Vec::new();
        for _ in 0..4 {
            let mut batch = ComputeBatch::new();
            v_cycles(&gpu, &mut cache, &mut batch, &h, &p, &div, 1).unwrap();
            batch.submit(&gpu).unwrap();
            let next = residual_rms(&gpu, &c, &p, &div, solid_field.as_ref());
            factors.push(r / next);
            r = next;
        }
        h.release(&mut pool);
        // The mean over four cycles: the first cycle on a smooth rhs has
        // little high-frequency error for the smoother to remove, so it is
        // the slowest (2.8× with the sphere).
        let mean = (r0 / r).powf(0.25);
        println!("{name}: r0 {r0:.4e} -> {r:.4e}, per cycle {factors:.2?}, mean {mean:.2}");
        assert!(
            mean >= floor,
            "{name}: per cycle {factors:.2?}, mean {mean:.2}"
        );
    }
}
