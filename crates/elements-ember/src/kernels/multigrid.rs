//! Geometric multigrid for the pressure solve (2b-3c spec §3.2–3.3).
//!
//! A correction scheme over a hierarchy of cell grids. Each level halves
//! every axis still longer than 1 cell, rounding up, down to 1×1×1. An axis
//! that reaches 1 stops while the others go on (semi-coarsening), so axes
//! that still have neighbours always share one spacing and no level is
//! anisotropic where it couples. A level's last cell along an axis may be
//! partial, covering only the fine cells left before the domain's edge.
//!
//! Every level is a finite-volume discretisation of ∇²p = div/h over the
//! true geometry: a face's weight is its area over the distance between the
//! points it joins, with an open face's p = 0 where the fine grid puts it,
//! 0.5 fine cells beyond the face (`LevelParams`). With solids, a coarse
//! face also scales by its cells' fluid fractions (`stencil.wgsl`). The fine
//! grid's weights are all exactly 1, and it runs the unchanged
//! `pressure.wgsl`.
//!
//! The residual stays in the units of `div`, so each level solves
//! L e = r / h. Prolongation P is geometric (`transfer.wgsl`); restriction is
//! κ·Pᵀ; smoothing is red-black before the coarse correction and black-red
//! after it; the coarsest solve is a palindrome; and closed domains remove
//! the mean over fluid cells only. So one V-cycle from zero is a symmetric
//! operator, as MGPCG needs.

use std::sync::Arc;

use elements_core::gpu::{
    ComputeBatch, Field, FieldDims, FieldFormat, FieldPool, GpuContext, GpuError, PipelineCache,
    ReduceOp, ReduceTarget, reduce,
};

use super::project::{Smoother, mask_view};
use super::{Bind, StepConstants, Uniforms, bind_group, expect_dims, uniform_buffer};

const RESIDUAL: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/level_params.wgsl"),
    include_str!("shaders/stencil.wgsl"),
    include_str!("shaders/residual.wgsl"),
);

const RESTRICT: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/level_params.wgsl"),
    include_str!("shaders/transfer.wgsl"),
    include_str!("shaders/restrict.wgsl"),
);

const RESTRICT_MASK: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/levels.wgsl"),
    include_str!("shaders/restrict_mask.wgsl"),
);

const PROLONG_ADD: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/level_params.wgsl"),
    include_str!("shaders/transfer.wgsl"),
    include_str!("shaders/prolong_add.wgsl"),
);

const PROLONG_NORM: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/level_params.wgsl"),
    include_str!("shaders/transfer.wgsl"),
    include_str!("shaders/prolong_norm.wgsl"),
);

const RESTRICT_FRACTION: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/level_params.wgsl"),
    include_str!("shaders/restrict_fraction.wgsl"),
);

const SUBTRACT_FLUID_MEAN: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/level_params.wgsl"),
    include_str!("shaders/subtract_fluid_mean.wgsl"),
);

/// An axis is halved while it is longer than this (spec §3.2). At 2, a tall
/// domain's coarse levels were 2×2×n with n's cells twice as long, and
/// point Gauss–Seidel could not smooth that: 32×32×256 stalled at ×1.4.
const COARSEST: u32 = 1;
/// Red-black sweeps before and after the coarse correction on every level
/// but the coarsest.
const SMOOTHING_SWEEPS: u32 = 2;
/// Sweeps on the coarsest level, run red-black then again black-red, so the
/// coarsest solve is a palindrome and the V-cycle stays symmetric. The
/// coarsest level is always 1×1×1, which one red pass solves exactly (or,
/// closed, leaves at 0); more sweeps would only add dispatches.
const COARSEST_SWEEPS: u32 = 1;

/// Matches `LevelParams` in `shaders/level_params.wgsl`, 128 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct LevelParams {
    dims: [u32; 3],
    slot: u32,
    g: [f32; 3],
    use_phi: f32,
    open_lo: [f32; 3],
    _p1: f32,
    open_hi: [f32; 3],
    _p2: f32,
    last_face: [f32; 3],
    _p3: f32,
    frac: [f32; 3],
    _p4: f32,
    scale: [f32; 3],
    _p5: f32,
    n0: [f32; 3],
    _p6: f32,
}

const _: () = assert!(std::mem::size_of::<LevelParams>() == 128);

/// Matches `OtherDims` in `shaders/levels.wgsl`, 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OtherDims {
    dims: [u32; 3],
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<OtherDims>() == 16);

fn axes(d: FieldDims) -> [u32; 3] {
    [d.x, d.y, d.z]
}

/// The next level: every axis longer than `COARSEST` halved, rounding up.
fn halve(dims: FieldDims) -> FieldDims {
    let h = |n: u32| if n > COARSEST { n.div_ceil(2) } else { n };
    FieldDims::new(h(dims.x), h(dims.y), h(dims.z))
}

/// A level's geometry and face weights: `dims` cells of `scale` fine cells
/// each over a fine domain of `n0` cells. Lengths are in fine cells.
///
/// Divide a level's finite-volume equation by V/s_ref², where V = Πs is a
/// full cell's volume and s_ref = min s. A full face along axis a then weighs
/// g_a = (s_ref/s_a)², the smoother's dx² is (s_ref·dx₀)², and a right-hand
/// side in `div` units restricts with κ = 1/2 per halved axis. A face's
/// weight is g_a·s_a / (the distance it spans), times the fraction of its
/// area inside the domain. Every weight is exactly 1 when `scale` is all 1.
fn level_params(dims: FieldDims, scale: [u32; 3], n0: FieldDims, slot: u32) -> LevelParams {
    let n = axes(dims);
    let n0 = axes(n0);
    let s_ref = f64::from(*scale.iter().min().expect("three axes"));
    let mut p = LevelParams {
        dims: n,
        slot,
        g: [0.0; 3],
        use_phi: 0.0,
        open_lo: [0.0; 3],
        _p1: 0.0,
        open_hi: [0.0; 3],
        _p2: 0.0,
        last_face: [0.0; 3],
        _p3: 0.0,
        frac: [0.0; 3],
        _p4: 0.0,
        scale: [0.0; 3],
        _p5: 0.0,
        n0: [0.0; 3],
        _p6: 0.0,
    };
    for a in 0..3 {
        let s = f64::from(scale[a]);
        let (count, full) = (f64::from(n[a]), f64::from(n0[a]));
        // Centroid of cell i; the last cell covers ((n−1)·s, n0).
        let centroid = |i: f64| {
            if i == count - 1.0 {
                ((count - 1.0) * s + full) / 2.0
            } else {
                (i + 0.5) * s
            }
        };
        let g = (s_ref / s).powi(2);
        p.g[a] = g as f32;
        // p = 0 sits 0.5 fine cells beyond each open face, as on the fine grid.
        p.open_lo[a] = (g * s / (centroid(0.0) + 0.5)) as f32;
        p.open_hi[a] = (g * s / (full + 0.5 - centroid(count - 1.0))) as f32;
        p.last_face[a] = if n[a] >= 2 {
            (g * s / (centroid(count - 1.0) - centroid(count - 2.0))) as f32
        } else {
            g as f32
        };
        p.frac[a] = ((full - (count - 1.0) * s) / s) as f32;
        p.scale[a] = s as f32;
        p.n0[a] = full as f32;
    }
    p
}

fn params_buffer(gpu: &GpuContext, p: &LevelParams) -> Result<wgpu::Buffer, GpuError> {
    uniform_buffer(gpu, "ember-level", bytemuck::bytes_of(p))
}

/// One multigrid level: its uniforms (dims, dx = s_ref·dx0, same h and
/// open_mask), its face weights, its right-hand side and correction fields,
/// and its solid mask (None when the domain has no collider).
///
/// Level 0 solves on the caller's `p` and `div`, so it holds only `tmp`; the
/// coarsest level computes no residual, so it has no `tmp`. With solids,
/// every level but the coarsest holds `norm`, its prolongation weights.
pub struct Level {
    u: Uniforms,
    params: wgpu::Buffer,
    rhs: Option<Field>,
    e: Option<Field>,
    tmp: Option<Field>,
    mask: Option<Field>,
    norm: Option<Field>,
    /// With solids, a coarse level's fluid fractions (stencil.wgsl).
    phi: Option<Field>,
}

/// The level hierarchy for a domain, built once per substep from the fine
/// uniforms' constants and the fine solid mask. Every field comes from the
/// pool; `release` returns them.
pub struct Hierarchy {
    levels: Vec<Level>,
    /// Level 0's mask, the caller's field. The view keeps its texture alive,
    /// but the caller must not release the field to a pool while the
    /// hierarchy is in use.
    fine_mask: Option<wgpu::TextureView>,
    /// For removing means in a closed domain.
    sum: ReduceTarget,
    /// Each level's solid cell count, in its `slot`; 0 without solids.
    solids: ReduceTarget,
}

impl Hierarchy {
    /// Build the levels and record, into `batch`, each coarse level's solid
    /// mask, the prolongation weights and the solid counts.
    pub fn new(
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        batch: &mut ComputeBatch,
        pool: &mut FieldPool,
        fine: &StepConstants,
        fine_mask: Option<&Field>,
    ) -> Result<Self, GpuError> {
        if fine_mask.is_some() != fine.has_solids {
            return Err(GpuError::Validation(
                "multigrid: a mask must be given exactly when StepConstants::has_solids is set"
                    .to_owned(),
            ));
        }
        if let Some(m) = fine_mask {
            expect_dims("multigrid fine mask", m, fine.cells)?;
        }
        let mut shapes = vec![(fine.cells, [1u32; 3])];
        loop {
            let (d, s) = *shapes.last().expect("the fine level");
            let next = halve(d);
            if next == d {
                break;
            }
            let (a, b) = (axes(d), axes(next));
            let scale = std::array::from_fn(|i| if a[i] == b[i] { s[i] } else { 2 * s[i] });
            shapes.push((next, scale));
        }
        let depth = shapes.len();
        let solids = ReduceTarget::new(gpu, depth as u32)?;

        let mut levels: Vec<Level> = Vec::with_capacity(depth);
        let fraction_pipe = if fine.has_solids {
            Some(cache.get_or_create(
                gpu,
                "ember.mg.restrict_fraction",
                RESTRICT_FRACTION,
                "main",
            )?)
        } else {
            None
        };
        for (l, &(cells, scale)) in shapes.iter().enumerate() {
            let mut lp = level_params(cells, scale, fine.cells, l as u32);
            if l > 0 && fine.has_solids {
                lp.use_phi = 1.0;
            }
            let s_ref = *scale.iter().min().expect("three axes");
            let c = StepConstants {
                cells,
                dx: fine.dx * s_ref as f32,
                ..*fine
            };
            let u = Uniforms::new(gpu, &c)?;
            let params = params_buffer(gpu, &lp)?;
            let acquire = |pool: &mut FieldPool| pool.acquire(gpu, cells, FieldFormat::R32Float);
            let (rhs, e) = if l == 0 {
                (None, None)
            } else {
                (Some(acquire(pool)?), Some(acquire(pool)?))
            };
            let tmp = if l + 1 < depth {
                Some(acquire(pool)?)
            } else {
                None
            };
            let mask = if l > 0 && fine.has_solids {
                let out = acquire(pool)?;
                let prev = levels.last().expect("level l − 1");
                let prev_mask = prev.mask.as_ref().or(fine_mask).expect("checked above");
                restrict_mask(gpu, cache, batch, &u, prev_mask, &out)?;
                Some(out)
            } else {
                None
            };
            let norm = if fine.has_solids && l + 1 < depth {
                Some(acquire(pool)?)
            } else {
                None
            };
            let phi = match (&fraction_pipe, levels.last()) {
                (Some(pipeline), Some(prev)) if l > 0 => {
                    let out = acquire(pool)?;
                    // Level 1 reads the fine mask as 1 − φ; later levels φ.
                    let (fine_phi, from_mask) = match &prev.phi {
                        Some(f) => (f, 0u32),
                        None => (fine_mask.expect("checked above"), 1),
                    };
                    let flag = uniform_buffer(
                        gpu,
                        "ember-from-mask",
                        bytemuck::bytes_of(&[from_mask, 0, 0, 0]),
                    )?;
                    let group = bind_group(
                        gpu,
                        pipeline,
                        &[
                            Bind::Tex(fine_phi),
                            Bind::Tex(&out),
                            Bind::Buf(u.any()),
                            Bind::Buf(&prev.params),
                            Bind::Buf(&params),
                            Bind::Buf(&flag),
                        ],
                    )?;
                    batch.dispatch(pipeline, &group, cells);
                    Some(out)
                }
                _ => None,
            };
            let counted = if l == 0 { fine_mask } else { mask.as_ref() };
            if let Some(m) = counted {
                reduce(gpu, cache, batch, m, ReduceOp::Sum, &solids, l as u32)?;
            }
            levels.push(Level {
                u,
                params,
                rhs,
                e,
                tmp,
                mask,
                norm,
                phi,
            });
        }
        let h = Self {
            levels,
            fine_mask: fine_mask.map(|m| m.view().clone()),
            sum: ReduceTarget::new(gpu, 1)?,
            solids,
        };
        if fine.has_solids {
            let pipeline =
                cache.get_or_create(gpu, "ember.mg.prolong_norm", PROLONG_NORM, "main")?;
            for l in 0..depth.saturating_sub(1) {
                let level = &h.levels[l];
                let norm = level
                    .norm
                    .as_ref()
                    .expect("solids: norm on every level but the last");
                let group = bind_group(
                    gpu,
                    &pipeline,
                    &[
                        Bind::Tex(norm),
                        Bind::Buf(level.u.any()),
                        Bind::View(h.mask(l)),
                        Bind::View(h.mask(l + 1)),
                        Bind::Buf(&level.params),
                        Bind::Buf(&h.levels[l + 1].params),
                    ],
                )?;
                batch.dispatch(&pipeline, &group, level.u.cells());
            }
        }
        Ok(h)
    }

    /// Number of levels, the fine level included.
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Level `l`'s cells.
    pub fn dims(&self, l: usize) -> FieldDims {
        self.levels[l].u.cells()
    }

    /// Whether every domain face is a wall, so p is defined only up to a
    /// constant and means are removed.
    pub fn closed(&self) -> bool {
        self.levels[0].u.open_mask() == 0
    }

    /// With solids, level `l`'s fluid fractions φ (none on level 0) and its
    /// prolongation weights (none on the coarsest level), for tests: with an
    /// all-fluid mask the solid path must reduce to the plain one.
    pub fn solid_weights(&self, l: usize) -> (Option<&Field>, Option<&Field>) {
        let level = &self.levels[l];
        (level.phi.as_ref(), level.norm.as_ref())
    }

    /// The fine level's uniforms.
    pub(crate) fn fine(&self) -> &Uniforms {
        &self.levels[0].u
    }

    /// Return every field to `pool`. The batch that used them must have been
    /// submitted.
    pub fn release(self, pool: &mut FieldPool) {
        for field in self.into_fields() {
            pool.release(field);
        }
    }

    /// Every pooled field the hierarchy holds, for a caller that returns
    /// them to the pool itself once the batch that used them has run.
    pub fn into_fields(self) -> Vec<Field> {
        self.levels
            .into_iter()
            .flat_map(|level| {
                [
                    level.rhs, level.e, level.tmp, level.mask, level.norm, level.phi,
                ]
            })
            .flatten()
            .collect()
    }

    /// Level `l`'s `solid` view: its mask, or the placeholder without solids.
    fn mask(&self, l: usize) -> &wgpu::TextureView {
        let level = &self.levels[l];
        let mask = if l == 0 {
            self.fine_mask.as_ref()
        } else {
            level.mask.as_ref().map(Field::view)
        };
        mask.unwrap_or(level.u.placeholder())
    }

    /// Level `l`'s fluid fractions, or the placeholder where unused.
    fn phi(&self, l: usize) -> &wgpu::TextureView {
        let level = &self.levels[l];
        level
            .phi
            .as_ref()
            .map(Field::view)
            .unwrap_or(level.u.placeholder())
    }

    /// Level `l`'s prolongation weights, or the placeholder without solids.
    fn norm(&self, l: usize) -> &wgpu::TextureView {
        let level = &self.levels[l];
        level
            .norm
            .as_ref()
            .map(Field::view)
            .unwrap_or(level.u.placeholder())
    }
}

/// The bind groups one level's part of a V-cycle records.
struct Recorded {
    smoother: Smoother,
    /// Residual, restriction to the next level, prolongation from it; absent
    /// on the coarsest level.
    transfer: Option<[wgpu::BindGroup; 3]>,
    /// Removing the fluid mean of this coarse level's right-hand side in a
    /// closed domain.
    mean: Option<FluidMean>,
}

/// `cycles` V-cycles on `p` for ∇²p = div/h, starting from what `p` holds.
/// In a closed domain each coarse right-hand side has its fluid mean
/// removed, and so does `p` after the last cycle.
pub fn v_cycles(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    h: &Hierarchy,
    p: &Field,
    div: &Field,
    cycles: u32,
) -> Result<(), GpuError> {
    VCycle::new(gpu, cache, h, p, div)?.record(gpu, cache, batch, h, p, cycles)
}

/// One V-cycle from zero: `e` is zeroed, then one cycle solves for it with
/// `rhs` as the right-hand side. This is the operator MGPCG preconditions
/// with, and it is symmetric (`a_v_cycle_is_a_symmetric_operator`).
pub fn v_cycle_from_zero(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    h: &Hierarchy,
    e: &Field,
    rhs: &Field,
) -> Result<(), GpuError> {
    let cycle = VCycle::new(gpu, cache, h, e, rhs)?;
    let zero = super::mgpcg::Zero::new(gpu, cache, e)?;
    zero.record(batch);
    cycle.record(gpu, cache, batch, h, e, 1)
}

/// V-cycles on one `p` and `div`, with every bind group built once, so
/// MGPCG can record one per iteration without rebuilding them.
pub(crate) struct VCycle {
    residual: Arc<wgpu::ComputePipeline>,
    restrict: Arc<wgpu::ComputePipeline>,
    prolong: Arc<wgpu::ComputePipeline>,
    recorded: Vec<Recorded>,
    /// Removing p's fluid mean after the last cycle, in a closed domain.
    p_mean: Option<FluidMean>,
}

impl VCycle {
    pub(crate) fn new(
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        h: &Hierarchy,
        p: &Field,
        div: &Field,
    ) -> Result<Self, GpuError> {
        let fine = &h.levels[0].u;
        expect_dims("v_cycles p", p, fine.cells())?;
        expect_dims("v_cycles div", div, fine.cells())?;
        let closed = h.closed();
        let depth = h.depth();

        let residual_pipe = cache.get_or_create(gpu, "ember.mg.residual", RESIDUAL, "main")?;
        let restrict_pipe = cache.get_or_create(gpu, "ember.mg.restrict", RESTRICT, "main")?;
        let prolong_pipe = cache.get_or_create(gpu, "ember.mg.prolong_add", PROLONG_ADD, "main")?;

        let x = |l: usize| -> &Field {
            if l == 0 {
                p
            } else {
                h.levels[l].e.as_ref().expect("coarse levels hold e")
            }
        };
        let b = |l: usize| -> &Field {
            if l == 0 {
                div
            } else {
                h.levels[l].rhs.as_ref().expect("coarse levels hold rhs")
            }
        };

        let mut recorded = Vec::with_capacity(depth);
        for (l, level) in h.levels.iter().enumerate() {
            let smoother = if l == 0 {
                Smoother::with_view(gpu, cache, &level.u, p, div, h.mask(0))?
            } else {
                Smoother::weighted(
                    gpu,
                    cache,
                    &level.u,
                    x(l),
                    b(l),
                    h.mask(l),
                    &level.params,
                    h.phi(l),
                )?
            };
            let transfer = if l + 1 < depth {
                let next = &h.levels[l + 1];
                let tmp = level.tmp.as_ref().expect("non-coarsest levels hold tmp");
                let residual = bind_group(
                    gpu,
                    &residual_pipe,
                    &[
                        Bind::Tex(x(l)),
                        Bind::Tex(b(l)),
                        Bind::Tex(tmp),
                        Bind::Buf(level.u.any()),
                        Bind::View(h.mask(l)),
                        Bind::Buf(&level.params),
                        Bind::View(h.phi(l)),
                    ],
                )?;
                let restrict = bind_group(
                    gpu,
                    &restrict_pipe,
                    &[
                        Bind::Tex(tmp),
                        Bind::View(h.mask(l)),
                        Bind::View(h.norm(l)),
                        Bind::Tex(b(l + 1)),
                        Bind::Tex(x(l + 1)),
                        Bind::Buf(next.u.any()),
                        Bind::View(h.mask(l + 1)),
                        Bind::Buf(&level.params),
                        Bind::Buf(&next.params),
                    ],
                )?;
                let prolong = bind_group(
                    gpu,
                    &prolong_pipe,
                    &[
                        Bind::Tex(x(l)),
                        Bind::Tex(x(l + 1)),
                        Bind::Buf(level.u.any()),
                        Bind::View(h.mask(l)),
                        Bind::View(h.mask(l + 1)),
                        Bind::View(h.norm(l)),
                        Bind::Buf(&level.params),
                        Bind::Buf(&next.params),
                    ],
                )?;
                Some([residual, restrict, prolong])
            } else {
                None
            };
            // Each coarse level's rhs, in a closed domain.
            let mean = if closed && l > 0 {
                Some(FluidMean::new(gpu, cache, h, l, b(l))?)
            } else {
                None
            };
            recorded.push(Recorded {
                smoother,
                transfer,
                mean,
            });
        }
        let p_mean = if closed {
            Some(FluidMean::new(gpu, cache, h, 0, p)?)
        } else {
            None
        };
        Ok(Self {
            residual: residual_pipe,
            restrict: restrict_pipe,
            prolong: prolong_pipe,
            recorded,
            p_mean,
        })
    }

    /// Record `cycles` V-cycles on the `p` this was built for.
    pub(crate) fn record(
        &self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        batch: &mut ComputeBatch,
        h: &Hierarchy,
        p: &Field,
        cycles: u32,
    ) -> Result<(), GpuError> {
        let depth = h.depth();
        for cycle in 0..cycles {
            if cycle > 0 {
                // One V-cycle per submission keeps each well inside GPU
                // watchdog limits at any resolution.
                batch.flush(gpu)?;
            }
            // Down: pre-smooth, then hand the residual to the next level.
            for (l, level) in h.levels.iter().enumerate() {
                let rec = &self.recorded[l];
                let Some([residual, restrict, _]) = &rec.transfer else {
                    rec.smoother.record(batch, COARSEST_SWEEPS, false);
                    rec.smoother.record(batch, COARSEST_SWEEPS, true);
                    break;
                };
                rec.smoother.record(batch, SMOOTHING_SWEEPS, false);
                batch.dispatch(&self.residual, residual, level.u.cells());
                batch.dispatch(&self.restrict, restrict, h.levels[l + 1].u.cells());
                if let Some(mean) = &self.recorded[l + 1].mean {
                    // The Neumann problem is solvable only for a right-hand
                    // side with zero fluid sum; restriction does not keep it
                    // exact.
                    let rhs = h.levels[l + 1]
                        .rhs
                        .as_ref()
                        .expect("coarse levels hold rhs");
                    mean.record(gpu, cache, batch, h, rhs)?;
                }
            }
            // Up: add the coarse correction, then post-smooth in the reverse
            // colour order so the cycle is symmetric.
            for l in (0..depth.saturating_sub(1)).rev() {
                let rec = &self.recorded[l];
                let [_, _, prolong] = rec.transfer.as_ref().expect("not the coarsest level");
                batch.dispatch(&self.prolong, prolong, h.levels[l].u.cells());
                rec.smoother.record(batch, SMOOTHING_SWEEPS, true);
            }
        }
        if let Some(mean) = &self.p_mean {
            mean.record(gpu, cache, batch, h, p)?;
        }
        Ok(())
    }
}

/// The fine residual r = div − h·L(p) into `out`, bound once for reuse.
pub(crate) struct FineResidual {
    pipe: Arc<wgpu::ComputePipeline>,
    group: wgpu::BindGroup,
    cells: FieldDims,
}

impl FineResidual {
    pub(crate) fn new(
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        h: &Hierarchy,
        p: &Field,
        div: &Field,
        out: &Field,
    ) -> Result<Self, GpuError> {
        let level = &h.levels[0];
        let cells = level.u.cells();
        expect_dims("residual p", p, cells)?;
        expect_dims("residual div", div, cells)?;
        expect_dims("residual output", out, cells)?;
        let pipe = cache.get_or_create(gpu, "ember.mg.residual", RESIDUAL, "main")?;
        let group = bind_group(
            gpu,
            &pipe,
            &[
                Bind::Tex(p),
                Bind::Tex(div),
                Bind::Tex(out),
                Bind::Buf(level.u.any()),
                Bind::View(h.mask(0)),
                Bind::Buf(&level.params),
                Bind::View(h.phi(0)),
            ],
        )?;
        Ok(Self { pipe, group, cells })
    }

    pub(crate) fn record(&self, batch: &mut ComputeBatch) {
        batch.dispatch(&self.pipe, &self.group, self.cells);
    }
}

/// Removing one field's mean over fluid cells on one level: a `Sum`
/// reduction into the hierarchy's `sum`, then `subtract_fluid_mean`.
pub(crate) struct FluidMean {
    pipe: Arc<wgpu::ComputePipeline>,
    group: wgpu::BindGroup,
    cells: FieldDims,
}

impl FluidMean {
    /// For `field` on level `l` of `h`.
    pub(crate) fn new(
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        h: &Hierarchy,
        l: usize,
        field: &Field,
    ) -> Result<Self, GpuError> {
        let level = &h.levels[l];
        expect_dims("fluid mean field", field, level.u.cells())?;
        let pipe = cache.get_or_create(
            gpu,
            "ember.mg.subtract_fluid_mean",
            SUBTRACT_FLUID_MEAN,
            "main",
        )?;
        let group = bind_group(
            gpu,
            &pipe,
            &[
                Bind::Tex(field),
                Bind::Buf(h.sum.buffer()),
                Bind::Buf(h.solids.buffer()),
                Bind::Buf(level.u.any()),
                Bind::View(h.mask(l)),
                Bind::Buf(&level.params),
            ],
        )?;
        Ok(Self {
            pipe,
            group,
            cells: level.u.cells(),
        })
    }

    /// Record the removal; `field` must be the one this was built for.
    pub(crate) fn record(
        &self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        batch: &mut ComputeBatch,
        h: &Hierarchy,
        field: &Field,
    ) -> Result<(), GpuError> {
        reduce(gpu, cache, batch, field, ReduceOp::Sum, &h.sum, 0)?;
        batch.dispatch(&self.pipe, &self.group, self.cells);
        Ok(())
    }
}

/// r = div − h·L(p) on the fine grid, in the same units as `div`, so the
/// smoother solves a level's correction with the level's own `div`-shaped
/// right-hand side. Solid cells get 0. Exposed for tests and for MGPCG.
#[allow(clippy::too_many_arguments)] // every arg is load-bearing; see the doc above.
pub fn residual(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    p: &Field,
    div: &Field,
    out: &Field,
    mask: Option<&Field>,
) -> Result<(), GpuError> {
    let solid = mask_view("residual mask", u, mask)?;
    let unit = params_buffer(gpu, &level_params(u.cells(), [1; 3], u.cells(), 0))?;
    record_residual(
        gpu,
        cache,
        batch,
        u,
        p,
        div,
        out,
        solid,
        &unit,
        u.placeholder(),
    )
}

/// `residual` on level `level` of `h`, with that level's weights and mask:
/// `p`, `div` and `out` must have the level's dims. For tests.
#[allow(clippy::too_many_arguments)] // every arg is load-bearing; see the doc above.
pub fn residual_at(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    h: &Hierarchy,
    level: usize,
    p: &Field,
    div: &Field,
    out: &Field,
) -> Result<(), GpuError> {
    let l = &h.levels[level];
    record_residual(
        gpu,
        cache,
        batch,
        &l.u,
        p,
        div,
        out,
        h.mask(level),
        &l.params,
        h.phi(level),
    )
}

#[allow(clippy::too_many_arguments)]
fn record_residual(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    p: &Field,
    div: &Field,
    out: &Field,
    solid: &wgpu::TextureView,
    params: &wgpu::Buffer,
    phi: &wgpu::TextureView,
) -> Result<(), GpuError> {
    expect_dims("residual p", p, u.cells())?;
    expect_dims("residual div", div, u.cells())?;
    expect_dims("residual output", out, u.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.mg.residual", RESIDUAL, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(p),
            Bind::Tex(div),
            Bind::Tex(out),
            Bind::Buf(u.any()),
            Bind::View(solid),
            Bind::Buf(params),
            Bind::View(phi),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// Write the coarse solid mask for `coarse`'s grid into `out`: a coarse cell
/// is solid only when every child of it inside `fine_mask` is solid.
pub fn restrict_mask(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    coarse: &Uniforms,
    fine_mask: &Field,
    out: &Field,
) -> Result<(), GpuError> {
    if halve(fine_mask.dims()) != coarse.cells() {
        return Err(GpuError::Validation(format!(
            "restrict_mask: fine {:?} does not halve to coarse {:?}",
            fine_mask.dims(),
            coarse.cells()
        )));
    }
    expect_dims("restrict_mask output", out, coarse.cells())?;
    let d = OtherDims {
        dims: axes(fine_mask.dims()),
        _pad: 0,
    };
    let fine_dims = uniform_buffer(gpu, "ember-level-dims", bytemuck::bytes_of(&d))?;
    let pipeline = cache.get_or_create(gpu, "ember.mg.restrict_mask", RESTRICT_MASK, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(fine_mask),
            Bind::Tex(out),
            Bind::Buf(coarse.any()),
            Bind::Buf(&fine_dims),
        ],
    )?;
    batch.dispatch(&pipeline, &group, coarse.cells());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fine_level_weighs_every_face_one() {
        let d = FieldDims::new(10, 7, 9);
        let p = level_params(d, [1; 3], d, 0);
        for v in [p.g, p.open_lo, p.open_hi, p.last_face, p.frac] {
            assert_eq!(v, [1.0; 3]);
        }
    }
}
