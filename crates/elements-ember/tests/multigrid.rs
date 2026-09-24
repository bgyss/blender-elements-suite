mod common;

use common::*;
use elements_core::gpu::{
    ComputeBatch, Field, FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache,
};
use elements_ember::boundaries::DEFAULT_OPEN_MASK;
use elements_ember::kernels::multigrid::{
    Hierarchy, residual, residual_at, restrict_mask, v_cycle_from_zero, v_cycles,
};
use elements_ember::kernels::{StepConstants, Uniforms, mgpcg, pressure};

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
        // pass even with a weak coarse correction; the second needs the coarse
        // levels. Two cycles measure 92× (16.3 then 5.7); a halved coarse
        // correction reached 36× in the original design.
        assert!(r2 < r0 / 60.0, "{name}: {r0} -> {r1} -> {r2}");
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
fn a_hierarchy_halves_every_axis_down_to_one_cell_and_returns_every_field() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    for (cells, depth, solids) in [
        (FieldDims::new(128, 128, 128), 8, false),
        (FieldDims::new(256, 256, 256), 9, false),
        // 40 → 20 → 10 → 5 → 3 → 2 → 1 along x; y and z reach 1 sooner and
        // stop there while x goes on.
        (FieldDims::new(40, 24, 16), 7, false),
        // 256 → 1 is 9 levels; x and y reach 1 after 6.
        (FieldDims::new(32, 32, 256), 9, false),
        (FieldDims::new(64, 64, 64), 7, true),
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

/// With a mask where every cell is fluid, the solid path's fluid fractions
/// φ and prolongation weights are exactly 1 on every level, so it applies
/// the plain path's stencil and transfers. Odd sizes give partial coarse
/// cells at the domain's edge.
#[test]
fn an_all_fluid_mask_gives_unit_fractions_and_weights() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    for (cells, open) in [
        (FieldDims::new(32, 32, 32), DEFAULT_OPEN_MASK),
        (FieldDims::new(16, 16, 16), 0),
        (FieldDims::new(40, 24, 20), DEFAULT_OPEN_MASK),
        (FieldDims::new(21, 13, 11), 0b111111),
    ] {
        let mut pool = FieldPool::new();
        let mask = upload(&gpu, &mut pool, cells, &vec![0.0; cells.voxel_count()]);
        let c = StepConstants {
            has_solids: true,
            open_mask: open,
            ..StepConstants::new(cells, H, 1.0)
        };
        let mut batch = ComputeBatch::new();
        let h = Hierarchy::new(&gpu, &mut cache, &mut batch, &mut pool, &c, Some(&mask)).unwrap();
        batch.submit(&gpu).unwrap();
        for l in 0..h.depth() {
            let (phi, norm) = h.solid_weights(l);
            for (name, field) in [("φ", phi), ("prolongation weight", norm)] {
                let Some(field) = field else { continue };
                let values = field.read_back(&gpu).unwrap();
                let off: Vec<(usize, f32)> = values
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|&(_, v)| v != 1.0)
                    .collect();
                assert!(
                    off.is_empty(),
                    "{cells:?} open {open:#b} level {l}: {} {name}s are not 1, first {:?}",
                    off.len(),
                    &off[..off.len().min(4)]
                );
            }
        }
        h.release(&mut pool);
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

/// Fluid cells' values, the solid cells zeroed, with the fluid mean removed
/// when `closed`, so a closed problem is solvable.
fn fluid_rhs(mut v: Vec<f32>, solid: Option<&[f32]>, closed: bool) -> Vec<f32> {
    let fluid = |i: usize| solid.is_none_or(|s| s[i] <= 0.5);
    if closed {
        let (mut sum, mut n) = (0.0f64, 0usize);
        for (i, x) in v.iter().enumerate() {
            if fluid(i) {
                sum += f64::from(*x);
                n += 1;
            }
        }
        let mean = (sum / n as f64) as f32;
        v.iter_mut().for_each(|x| *x -= mean);
    }
    for (i, x) in v.iter_mut().enumerate() {
        if !fluid(i) {
            *x = 0.0;
        }
    }
    v
}

/// Residual RMS after each of `cycles` V-cycles from p = 0, with r0 first.
fn residual_history(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    cells: FieldDims,
    mask: u32,
    sphere: bool,
    cycles: usize,
) -> Vec<f64> {
    let mut pool = FieldPool::new();
    let c = StepConstants {
        open_mask: mask,
        has_solids: sphere,
        ..StepConstants::new(cells, H, 2.0 / cells.x as f32)
    };
    let solid = sphere.then(|| sphere_mask(cells, cells.x.min(cells.y).min(cells.z) as f32 / 6.0));
    let div_values = fluid_rhs(smooth_rhs(cells), solid.as_deref(), mask == 0);
    let solid_field = solid.map(|s| upload(gpu, &mut pool, cells, &s));
    let div = upload(gpu, &mut pool, cells, &div_values);
    let p = upload(gpu, &mut pool, cells, &vec![0.0; cells.voxel_count()]);
    let mut batch = ComputeBatch::new();
    let h = Hierarchy::new(gpu, cache, &mut batch, &mut pool, &c, solid_field.as_ref()).unwrap();
    batch.submit(gpu).unwrap();
    let mut history = vec![residual_rms(gpu, &c, &p, &div, solid_field.as_ref())];
    for _ in 0..cycles {
        let mut batch = ComputeBatch::new();
        v_cycles(gpu, cache, &mut batch, &h, &p, &div, 1).unwrap();
        batch.submit(gpu).unwrap();
        history.push(residual_rms(gpu, &c, &p, &div, solid_field.as_ref()));
    }
    h.release(&mut pool);
    history
}

/// A smooth right-hand side is what the pressure solve sees, and where the
/// coarse levels do the work. Every shape converges at the same steady rate:
/// sizes that halve unevenly (partial coarse cells), tall domains (axes that
/// stop halving before others), open faces and walls, with and without a
/// collider.
///
/// The floors are on the steady rate, the geometric mean of cycles 2–4. The
/// first cycle from p = 0 is always about 0.72× the steady one (3.0× against
/// 4.2×, 2.8× against 3.8× with a collider), so it gets its own floor. A
/// symmetric cycle (restriction = κ·Pᵀ, which MGPCG needs) runs at about
/// 4.2× where the unsymmetric averaging restriction ran at 6×; scaling the
/// coarse correction either way only made it worse (task 1 report).
#[test]
fn v_cycles_converge_steadily_on_every_shape() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    let top = DEFAULT_OPEN_MASK;
    for (name, cells, mask, sphere) in [
        ("64³ closed", FieldDims::new(64, 64, 64), 0u32, false),
        ("64³ open top", FieldDims::new(64, 64, 64), top, false),
        ("128³ open top", FieldDims::new(128, 128, 128), top, false),
        (
            "64×64×100 open top",
            FieldDims::new(64, 64, 100),
            top,
            false,
        ),
        ("100³ open top", FieldDims::new(100, 100, 100), top, false),
        ("80³ open top", FieldDims::new(80, 80, 80), top, false),
        ("100³ closed", FieldDims::new(100, 100, 100), 0, false),
        (
            "64×64×256 open top",
            FieldDims::new(64, 64, 256),
            top,
            false,
        ),
        (
            "32×32×256 open top",
            FieldDims::new(32, 32, 256),
            top,
            false,
        ),
        (
            "64³ open top, sphere",
            FieldDims::new(64, 64, 64),
            top,
            true,
        ),
        ("64³ closed, sphere", FieldDims::new(64, 64, 64), 0, true),
    ] {
        let r = residual_history(&gpu, &mut cache, cells, mask, sphere, 4);
        let factors: Vec<f64> = r.windows(2).map(|w| w[0] / w[1]).collect();
        let steady = (r[1] / r[4]).powf(1.0 / 3.0);
        let (first_floor, steady_floor) = if sphere { (2.5, 3.5) } else { (2.8, 4.0) };
        println!("{name}: per cycle {factors:.2?}, steady {steady:.2}");
        assert!(
            factors[0] >= first_floor && steady >= steady_floor,
            "{name}: per cycle {factors:.2?}, steady {steady:.2}"
        );
    }
}

/// MGPCG needs the preconditioner to be symmetric: ⟨Ma, b⟩ = ⟨a, Mb⟩ for
/// M = one V-cycle from zero. The real cycle measures 1.5e-10 to 1.1e-8;
/// post-smoothing in the wrong colour order measured 1.57e-5.
#[test]
fn a_v_cycle_is_a_symmetric_operator() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    for (name, cells, mask, sphere) in [
        (
            "open top",
            FieldDims::new(32, 32, 32),
            DEFAULT_OPEN_MASK,
            false,
        ),
        ("closed, uneven", FieldDims::new(30, 27, 21), 0u32, false),
        (
            "open top, sphere, uneven",
            FieldDims::new(30, 27, 33),
            DEFAULT_OPEN_MASK,
            true,
        ),
        ("closed, sphere", FieldDims::new(32, 32, 32), 0, true),
    ] {
        let mut pool = FieldPool::new();
        let c = StepConstants {
            open_mask: mask,
            has_solids: sphere,
            ..StepConstants::new(cells, H, 1.0 / 16.0)
        };
        let solid = sphere.then(|| sphere_mask(cells, 6.0));
        let solid_field = solid.as_ref().map(|s| upload(&gpu, &mut pool, cells, s));
        let closed = mask == 0;
        let a = fluid_rhs(pattern(cells, 11), solid.as_deref(), closed);
        let b = fluid_rhs(pattern(cells, 29), solid.as_deref(), closed);
        let mut apply = |v: &[f32]| -> Vec<f32> {
            let div = upload(&gpu, &mut pool, cells, v);
            // Whatever e holds beforehand, the cycle starts from zero.
            let p = upload(&gpu, &mut pool, cells, &pattern(cells, 47));
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
            v_cycle_from_zero(&gpu, &mut cache, &mut batch, &h, &p, &div).unwrap();
            batch.submit(&gpu).unwrap();
            let out = p.read_back(&gpu).unwrap();
            h.release(&mut pool);
            out
        };
        let (ma, mb) = (apply(&a), apply(&b));
        let dot = |x: &[f32], y: &[f32]| {
            x.iter()
                .zip(y)
                .map(|(u, v)| f64::from(*u) * f64::from(*v))
                .sum::<f64>()
        };
        let (x, y) = (dot(&ma, &b), dot(&a, &mb));
        let rel = (x - y).abs() / x.abs().max(y.abs());
        println!("{name}: <Ma,b> {x:.9e}, <a,Mb> {y:.9e}, relative difference {rel:.3e}");
        assert!(rel < 1e-6, "{name}: <Ma,b> {x}, <a,Mb> {y}, relative {rel}");
    }
}

/// The residual on coarse levels against a CPU finite-volume stencil built
/// from the geometry alone: each level cell spans [i·s, min((i+1)·s, N₀))
/// fine cells along each axis, a face weighs its area over the distance
/// between the centroids it joins (or to p = 0, 0.5 beyond an open face),
/// and the equation is divided by V/s_ref² (V = Πs, s_ref = min s). This
/// covers the partial last cells, the per-side open weights and the
/// per-axis weights of a level whose axes stopped halving at different
/// times.
#[test]
fn coarse_level_residuals_match_a_finite_volume_stencil() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    // −x, +y, +z open; 10×12×9 halves unevenly in x and z. 3×3×20 has
    // x and y reach 1 while z goes on (open top and bottom).
    for (fine, mask) in [
        (FieldDims::new(10, 12, 9), 0b101001u32),
        (FieldDims::new(3, 3, 20), 0b110000),
    ] {
        let mut pool = FieldPool::new();
        let dx0 = 0.25f32;
        let c = StepConstants {
            open_mask: mask,
            ..StepConstants::new(fine, H, dx0)
        };
        let mut batch = ComputeBatch::new();
        let h = Hierarchy::new(&gpu, &mut cache, &mut batch, &mut pool, &c, None).unwrap();
        batch.submit(&gpu).unwrap();
        let n0 = [fine.x, fine.y, fine.z];
        let mut scale = [1u32; 3];
        for l in 1..h.depth() {
            let prev = h.dims(l - 1);
            let dims = h.dims(l);
            let (pa, da) = ([prev.x, prev.y, prev.z], [dims.x, dims.y, dims.z]);
            for a in 0..3 {
                if da[a] != pa[a] {
                    scale[a] *= 2;
                }
            }
            let p_values = pattern(dims, 5);
            let div_values = pattern(dims, 6);
            let p = upload(&gpu, &mut pool, dims, &p_values);
            let div = upload(&gpu, &mut pool, dims, &div_values);
            let out = pool.acquire(&gpu, dims, FieldFormat::R32Float).unwrap();
            let mut batch = ComputeBatch::new();
            residual_at(&gpu, &mut cache, &mut batch, &h, l, &p, &div, &out).unwrap();
            batch.submit(&gpu).unwrap();
            let got = out.read_back(&gpu).unwrap();

            // The geometry, in fine cells.
            let extent = |a: usize, i: u32| {
                let lo = f64::from(i * scale[a]);
                let hi = f64::from(((i + 1) * scale[a]).min(n0[a]));
                (lo, hi)
            };
            let centroid = |a: usize, i: u32| {
                let (lo, hi) = extent(a, i);
                (lo + hi) / 2.0
            };
            let s_ref = f64::from(*scale.iter().min().unwrap());
            let k = f64::from(scale[0] * scale[1] * scale[2]) / (s_ref * s_ref);
            let dx2 = f64::from(dx0) * s_ref * f64::from(dx0) * s_ref;
            let mut want = vec![0.0f32; dims.voxel_count()];
            for kz in 0..dims.z {
                for jy in 0..dims.y {
                    for ix in 0..dims.x {
                        let cell = [ix, jy, kz];
                        let at = |q: [u32; 3]| p_values[index(dims, q[0], q[1], q[2])] as f64;
                        let mut sum = 0.0f64;
                        for a in 0..3 {
                            let area: f64 = (0..3)
                                .filter(|&b| b != a)
                                .map(|b| {
                                    let (lo, hi) = extent(b, cell[b]);
                                    hi - lo
                                })
                                .product();
                            for side in 0..2 {
                                let i = cell[a];
                                let inside = if side == 0 { i > 0 } else { i + 1 < da[a] };
                                if inside {
                                    let mut q = cell;
                                    q[a] = if side == 0 { i - 1 } else { i + 1 };
                                    let d = (centroid(a, q[a]) - centroid(a, i)).abs();
                                    sum += area / d / k * (at(q) - at(cell));
                                } else if is_open(mask, a, side) {
                                    let d = if side == 0 {
                                        centroid(a, i) + 0.5
                                    } else {
                                        f64::from(n0[a]) + 0.5 - centroid(a, i)
                                    };
                                    sum += area / d / k * (0.0 - at(cell));
                                }
                            }
                        }
                        let at_div = f64::from(div_values[index(dims, ix, jy, kz)]);
                        want[index(dims, ix, jy, kz)] = (at_div - f64::from(H) * sum / dx2) as f32;
                    }
                }
            }
            let worst = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs() / w.abs().max(1.0))
                .fold(0.0f32, f32::max);
            println!(
                "{fine:?} level {l} {dims:?} scale {scale:?}: worst relative error {worst:.2e}"
            );
            assert!(
                worst < 1e-4,
                "{fine:?} level {l}: worst relative error {worst}"
            );
        }
        h.release(&mut pool);
    }
}

/// A closed domain defines p only up to a constant; the V-cycle removes p's
/// mean over fluid cells, where the solid cells (held at 0) do not count.
#[test]
fn a_closed_domain_keeps_the_fluid_mean_of_p_at_zero() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    let mut pool = FieldPool::new();
    let cells = FieldDims::new(32, 32, 32);
    let c = StepConstants {
        open_mask: 0,
        has_solids: true,
        ..StepConstants::new(cells, H, 1.0 / 16.0)
    };
    let solid = sphere_mask(cells, 8.0);
    let div_values = fluid_rhs(smooth_rhs(cells), Some(&solid), true);
    // A warm start offset by 1 in the fluid, as a previous frame might leave.
    let p0: Vec<f32> = solid
        .iter()
        .map(|&s| if s > 0.5 { 0.0 } else { 1.0 })
        .collect();
    let solid_field = upload(&gpu, &mut pool, cells, &solid);
    let div = upload(&gpu, &mut pool, cells, &div_values);
    let p = upload(&gpu, &mut pool, cells, &p0);
    let mut batch = ComputeBatch::new();
    let h = Hierarchy::new(
        &gpu,
        &mut cache,
        &mut batch,
        &mut pool,
        &c,
        Some(&solid_field),
    )
    .unwrap();
    v_cycles(&gpu, &mut cache, &mut batch, &h, &p, &div, 2).unwrap();
    batch.submit(&gpu).unwrap();
    h.release(&mut pool);
    let got = p.read_back(&gpu).unwrap();
    let (mut sum, mut n, mut peak) = (0.0f64, 0usize, 0.0f32);
    for (v, s) in got.iter().zip(&solid) {
        if *s <= 0.5 {
            sum += f64::from(*v);
            n += 1;
            peak = peak.max(v.abs());
        }
    }
    let mean = sum / n as f64;
    println!("fluid mean {mean:.3e}, max |p| {peak:.3e}");
    assert!(
        mean.abs() < 1e-6 * f64::from(peak),
        "fluid mean {mean}, max |p| {peak}"
    );
}

/// MGPCG and plain V-cycles from p = 0 on the same problem: the reduction
/// r0/r after `steps` of each.
#[allow(clippy::too_many_arguments)]
fn reduction_after(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    cells: FieldDims,
    mask: u32,
    solid: Option<&[f32]>,
    div_values: &[f32],
    solver: Solver,
    steps: u32,
) -> f64 {
    let r = run(
        gpu, cache, cells, mask, solid, div_values, None, solver, steps,
    );
    r.r0 / r.r
}

/// What one solve left behind: the initial and final residual RMS over
/// fluid cells, and p.
struct Solved {
    r0: f64,
    r: f64,
    p: Vec<f32>,
}

/// `steps` V-cycles or MGPCG iterations on 64³-style test problems, from
/// `warm` (or p = 0), with dx = 2/nx as in `residual_history`.
#[allow(clippy::too_many_arguments)]
fn run(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    cells: FieldDims,
    mask: u32,
    solid: Option<&[f32]>,
    div_values: &[f32],
    warm: Option<&[f32]>,
    solver: Solver,
    steps: u32,
) -> Solved {
    let mut pool = FieldPool::new();
    let c = StepConstants {
        open_mask: mask,
        has_solids: solid.is_some(),
        ..StepConstants::new(cells, H, 2.0 / cells.x as f32)
    };
    let solid_field = solid.map(|s| upload(gpu, &mut pool, cells, s));
    let div = upload(gpu, &mut pool, cells, div_values);
    let zeros = vec![0.0; cells.voxel_count()];
    let p = upload(gpu, &mut pool, cells, warm.unwrap_or(&zeros));
    let r0 = residual_rms(gpu, &c, &p, &div, solid_field.as_ref());
    let mut batch = ComputeBatch::new();
    let h = Hierarchy::new(gpu, cache, &mut batch, &mut pool, &c, solid_field.as_ref()).unwrap();
    match solver {
        Solver::VCycles => v_cycles(gpu, cache, &mut batch, &h, &p, &div, steps).unwrap(),
        Solver::Mgpcg => mgpcg(gpu, cache, &mut batch, &mut pool, &h, &p, &div, steps).unwrap(),
    }
    batch.submit(gpu).unwrap();
    h.release(&mut pool);
    let r = residual_rms(gpu, &c, &p, &div, solid_field.as_ref());
    Solved {
        r0,
        r,
        p: p.read_back(gpu).unwrap(),
    }
}

/// On a smooth right-hand side around a sphere, 4 MGPCG iterations beat 4
/// V-cycles by 34× (5.37e3 against 158). The brief's 6 iterations would
/// measure the f32 floor instead: MGPCG reaches it, about 1e4, after 5.
#[test]
fn mgpcg_converges_faster_than_v_cycles_around_a_sphere() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(64, 64, 64);
    let solid = sphere_mask(cells, 64.0 / 6.0);
    let div = fluid_rhs(smooth_rhs(cells), Some(&solid), false);
    let top = DEFAULT_OPEN_MASK;
    let mut reduction =
        |solver| reduction_after(&gpu, &mut cache, cells, top, Some(&solid), &div, solver, 4);
    let (v, m) = (reduction(Solver::VCycles), reduction(Solver::Mgpcg));
    println!(
        "after 4: V-cycles {v:.3e}, MGPCG {m:.3e}, ratio {:.1}",
        m / v
    );
    assert!(m >= 10.0 * v, "V-cycles {v}, MGPCG {m}");
}

/// A closed box, where the fluid means of r, z and p are removed. The noise
/// right-hand side's f32 floor is about 4e6 below r0 (a smooth one's is
/// about 1.5e4, too close to 1e5 to test).
#[test]
fn mgpcg_reaches_a_tight_residual_in_a_closed_box() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(64, 64, 64);
    let div = fluid_rhs(rhs(cells), None, true);
    let s = run(
        &gpu,
        &mut cache,
        cells,
        0,
        None,
        &div,
        None,
        Solver::Mgpcg,
        10,
    );
    println!(
        "r0 {:.3e} -> r {:.3e}, reduction {:.3e}",
        s.r0,
        s.r,
        s.r0 / s.r
    );
    assert!(s.r < 1e-5 * s.r0, "r0 {} -> r {}", s.r0, s.r);
}

/// A closed box with a collider, as the solver meets it: div has a small
/// fluid mean, which no p can match, and the warm start is offset. MGPCG
/// removes the fluid mean of r each time r is formed, so it converges on the
/// solvable part, and it removes p's at the end.
#[test]
fn mgpcg_removes_fluid_means_in_a_closed_box() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(64, 64, 64);
    let solid = sphere_mask(cells, 64.0 / 6.0);
    let solvable = fluid_rhs(rhs(cells), Some(&solid), true);
    // A fluid mean of 0.05, about a tenth of the RMS.
    let div: Vec<f32> = solvable
        .iter()
        .zip(&solid)
        .map(|(d, s)| if *s > 0.5 { 0.0 } else { d + 0.05 })
        .collect();
    // Offset by about max |p|: a larger offset raises the f32 floor, since
    // the residual then cancels larger values.
    let warm: Vec<f32> = solid
        .iter()
        .map(|&s| if s > 0.5 { 0.0 } else { 0.02 })
        .collect();
    let s = run(
        &gpu,
        &mut cache,
        cells,
        0,
        Some(&solid),
        &div,
        Some(&warm),
        Solver::Mgpcg,
        10,
    );
    // The residual against the solvable part: the same p, the mean-free div.
    let mut pool = FieldPool::new();
    let c = StepConstants {
        open_mask: 0,
        has_solids: true,
        ..StepConstants::new(cells, H, 2.0 / cells.x as f32)
    };
    let solid_field = upload(&gpu, &mut pool, cells, &solid);
    let p = upload(&gpu, &mut pool, cells, &s.p);
    let zeros = upload(&gpu, &mut pool, cells, &vec![0.0; cells.voxel_count()]);
    let solvable_field = upload(&gpu, &mut pool, cells, &solvable);
    let r0 = residual_rms(&gpu, &c, &zeros, &solvable_field, Some(&solid_field));
    let r = residual_rms(&gpu, &c, &p, &solvable_field, Some(&solid_field));
    let (mut sum, mut n, mut peak) = (0.0f64, 0usize, 0.0f32);
    for (v, m) in s.p.iter().zip(&solid) {
        if *m <= 0.5 {
            sum += f64::from(*v);
            n += 1;
            peak = peak.max(v.abs());
        }
    }
    let mean = sum / n as f64;
    println!(
        "solvable part: r0 {r0:.3e} -> r {r:.3e}; fluid mean of p {mean:.3e}, max |p| {peak:.3e}"
    );
    assert!(r < 1e-5 * r0, "solvable part: r0 {r0} -> r {r}");
    assert!(
        mean.abs() < 1e-6 * f64::from(peak),
        "fluid mean {mean}, max |p| {peak}"
    );
}

/// MGPCG solves for the correction to the p it is given: from a converged
/// p, one more iteration stays converged. Starting from zero instead, one
/// iteration leaves about 1/12 of r0.
#[test]
fn mgpcg_keeps_a_warm_start() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(64, 64, 64);
    let top = DEFAULT_OPEN_MASK;
    let div = rhs(cells);
    let solved = run(
        &gpu,
        &mut cache,
        cells,
        top,
        None,
        &div,
        None,
        Solver::Mgpcg,
        8,
    );
    let again = run(
        &gpu,
        &mut cache,
        cells,
        top,
        None,
        &div,
        Some(&solved.p),
        Solver::Mgpcg,
        1,
    );
    println!(
        "cold r0 {:.3e}; converged {:.3e}; one more iteration {:.3e}",
        solved.r0, solved.r, again.r
    );
    assert!(
        again.r < 2.0 * solved.r,
        "converged {}, one more {}",
        solved.r,
        again.r
    );
}

/// Two solves from the same inputs give the same bits. They share one pool,
/// so the second gets back the first's fields, dirty: a solve that read a
/// field before writing it would differ.
#[test]
fn mgpcg_is_bit_identical_across_runs() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    let mut pool = FieldPool::new();
    let cells = FieldDims::new(32, 32, 32);
    let solid = sphere_mask(cells, 6.0);
    // Closed with a collider: every reduction, fluid mean included, runs.
    let div_values = fluid_rhs(smooth_rhs(cells), Some(&solid), true);
    let c = StepConstants {
        open_mask: 0,
        has_solids: true,
        ..StepConstants::new(cells, H, 1.0 / 16.0)
    };
    let mut once = || {
        let solid_field = upload(&gpu, &mut pool, cells, &solid);
        let div = upload(&gpu, &mut pool, cells, &div_values);
        let p = upload(&gpu, &mut pool, cells, &vec![0.0; cells.voxel_count()]);
        let mut batch = ComputeBatch::new();
        let h = Hierarchy::new(
            &gpu,
            &mut cache,
            &mut batch,
            &mut pool,
            &c,
            Some(&solid_field),
        )
        .unwrap();
        mgpcg(&gpu, &mut cache, &mut batch, &mut pool, &h, &p, &div, 5).unwrap();
        batch.submit(&gpu).unwrap();
        let out = p.read_back(&gpu).unwrap();
        h.release(&mut pool);
        for field in [solid_field, div, p] {
            pool.release(field);
        }
        out
    };
    let (a, b) = (once(), once());
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let differ = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    println!("{differ} of {} cells differ", a.len());
    assert!(bits(&a) == bits(&b), "{differ} cells differ");
    assert!(a.iter().any(|x| *x != 0.0), "the solve did nothing");
}

/// A one-cell plate across the domain at z = 30, with a 2-cell slit
/// (x in 31..33) along its whole length in y.
fn plate_with_slit(cells: FieldDims) -> Vec<f32> {
    let mut out = vec![0.0; cells.voxel_count()];
    for j in 0..cells.y {
        for i in 0..cells.x {
            if !(31..33).contains(&i) {
                out[index(cells, i, j, 30)] = 1.0;
            }
        }
    }
    out
}

/// A one-cell-thick tube about the z axis, radius 20 to 21 cells, from the
/// floor up to z = 48, so its inside reaches the rest of the domain only
/// over its top rim.
fn thin_tube(cells: FieldDims) -> Vec<f32> {
    let mut out = vec![0.0; cells.voxel_count()];
    let (cx, cy) = (cells.x as f32 / 2.0, cells.y as f32 / 2.0);
    for k in 0..48.min(cells.z) {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let (x, y) = (i as f32 + 0.5 - cx, j as f32 + 0.5 - cy);
                if (20.0..21.0).contains(&(x * x + y * y).sqrt()) {
                    out[index(cells, i, j, k)] = 1.0;
                }
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Solver {
    VCycles,
    Mgpcg,
}

/// A one-cell wall vanishes on coarse levels (a coarse cell is solid only
/// when all its children are), so the V-cycle misses the modes that jump
/// across it: plain V-cycles lose about 2% of the error per cycle. They reach
/// only 1.3e3 (plate) and 5.7e3 (tube) after 40 cycles on this right-hand
/// side, and on a smooth one the residual first grows. The V-cycle is still
/// symmetric positive definite, so CG converges: 1e4 after 14 (plate, open
/// top or closed) and 15 (tube) iterations. 20 leaves a margin; the f32
/// floor is about 1e6.
#[test]
fn mgpcg_converges_around_thin_colliders() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(64, 64, 64);
    let top = DEFAULT_OPEN_MASK;
    let plate = plate_with_slit(cells);
    let tube = thin_tube(cells);
    let mut failed = Vec::new();
    for (name, mask, solid) in [
        ("plate with a slit, open top", top, &plate),
        ("plate with a slit, closed", 0u32, &plate),
        ("thin tube, open top", top, &tube),
    ] {
        let div = fluid_rhs(rhs(cells), Some(solid), mask == 0);
        let mut reduction =
            |solver| reduction_after(&gpu, &mut cache, cells, mask, Some(solid), &div, solver, 20);
        let (v, m) = (reduction(Solver::VCycles), reduction(Solver::Mgpcg));
        println!("{name}: after 20, V-cycles {v:.3e}, MGPCG {m:.3e}");
        if m < 1e4 {
            failed.push(format!("{name}: {m:.3e}×"));
        }
    }
    assert!(
        failed.is_empty(),
        "MGPCG reduced the residual too little: {failed:?}"
    );
}

/// Wall-clock time per V-cycle, per MGPCG iteration and per Gauss–Seidel
/// sweep, at 128³ and 256³ with an open top and no collider, for the solver
/// gate (task 5). Each is the difference between a long and a short run,
/// each recorded, submitted and waited for, divided by the difference in
/// counts, so setup cancels. Not a pass/fail test: `cargo test -p
/// elements-ember --test multigrid time_the_solvers -- --ignored
/// --nocapture`.
#[test]
#[ignore]
fn time_the_solvers() {
    use std::time::Instant;
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    for n in [128u32, 256] {
        let cells = FieldDims::new(n, n, n);
        let mut pool = FieldPool::new();
        let c = StepConstants::new(cells, H, 1.0 / n as f32);
        let div = upload(&gpu, &mut pool, cells, &smooth_rhs(cells));
        let p = upload(&gpu, &mut pool, cells, &vec![0.0; cells.voxel_count()]);
        let u = Uniforms::new(&gpu, &c).unwrap();
        let mut batch = ComputeBatch::new();
        let h = Hierarchy::new(&gpu, &mut cache, &mut batch, &mut pool, &c, None).unwrap();
        batch.submit(&gpu).unwrap();
        gpu.wait().unwrap();
        let time = |cache: &mut PipelineCache, pool: &mut FieldPool, what: &str, count: u32| {
            let start = Instant::now();
            let mut batch = ComputeBatch::new();
            match what {
                "v" => v_cycles(&gpu, cache, &mut batch, &h, &p, &div, count).unwrap(),
                "pcg" => mgpcg(&gpu, cache, &mut batch, pool, &h, &p, &div, count).unwrap(),
                _ => pressure(&gpu, cache, &mut batch, &u, &p, &div, count, count, None).unwrap(),
            }
            batch.submit(&gpu).unwrap();
            gpu.wait().unwrap();
            start.elapsed().as_secs_f64() * 1e3
        };
        for (what, name, short, long) in [
            ("v", "V-cycle", 2u32, 12u32),
            ("pcg", "MGPCG iteration", 2, 12),
            ("gs", "red-black sweep", 20, 220),
        ] {
            time(&mut cache, &mut pool, what, short); // warm up pipelines and pool
            let mut per = Vec::new();
            for _ in 0..3 {
                let a = time(&mut cache, &mut pool, what, short);
                let b = time(&mut cache, &mut pool, what, long);
                per.push((b - a) / f64::from(long - short));
            }
            per.sort_by(f64::total_cmp);
            println!(
                "{n}³ {name}: {:.3} ms (median of 3; runs {per:.3?})",
                per[1]
            );
        }
        h.release(&mut pool);
    }
}
