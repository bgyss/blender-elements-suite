//! Geometric multigrid for the pressure solve (2b-3c spec §3).
//!
//! A correction scheme over a hierarchy of cell grids, each half the last
//! (rounding up), re-discretised at spacing 2ˡ·dx and run by the Gauss–Seidel
//! smoother. The residual is kept in the units of `div`: the smoother solves
//! (Σ − count·p) = dx²·div / h, so a coarse level solves L e = r / h by
//! running the smoother with `div` := r and its own dx².
//!
//! Two things keep the cycle converging with open faces (task 1 report):
//! every level puts p = 0 where the fine grid does, 0.5·dx₀ beyond an open
//! face (`open_weight`), and the hierarchy goes down to 2 cells, where 32
//! sweeps solve the coarsest problem. Stopping at 8 cells left the slowest
//! open-face mode at ×0.66 per cycle; the fine stencil unchanged on every
//! level moved the boundary out by 0.5·dx_l and diverged at 128³.

use elements_core::gpu::{
    ComputeBatch, Field, FieldDims, FieldFormat, FieldPool, GpuContext, GpuError, PipelineCache,
    ReduceTarget,
};

use super::project::{Smoother, mask_view};
use super::{Bind, StepConstants, Uniforms, bind_group, expect_dims, remove_mean, uniform_buffer};

const RESIDUAL: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/residual.wgsl"),
);

const RESTRICT: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/levels.wgsl"),
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
    include_str!("shaders/levels.wgsl"),
    include_str!("shaders/prolong_add.wgsl"),
);

/// A level no more than this many cells along its smallest axis ends the
/// hierarchy (spec §3.2).
const COARSEST: u32 = 2;
/// Red-black sweeps before and after the coarse correction on every level
/// but the coarsest.
const SMOOTHING_SWEEPS: u32 = 2;
/// Red-black sweeps on the coarsest level.
const COARSEST_SWEEPS: u32 = 32;

/// Matches `OtherDims` in `shaders/levels.wgsl`, 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OtherDims {
    dims: [u32; 3],
    ghost: f32,
}

const _: () = assert!(std::mem::size_of::<OtherDims>() == 16);

fn dims_buffer(gpu: &GpuContext, dims: FieldDims, ghost: f32) -> Result<wgpu::Buffer, GpuError> {
    let d = OtherDims {
        dims: [dims.x, dims.y, dims.z],
        ghost,
    };
    uniform_buffer(gpu, "ember-level-dims", bytemuck::bytes_of(&d))
}

/// What an open neighbour adds to `count` on level `l`: dx_l / d, where
/// d = 0.5·dx_l + 0.5·dx₀ is the distance from the last cell centre to p = 0,
/// which sits 0.5·dx₀ beyond the face on every level, as on the fine grid.
/// Exactly 1 on level 0, so the fine grid is today's stencil bit for bit.
fn open_weight(level: usize) -> f32 {
    let scale = (1u32 << level) as f32;
    2.0 * scale / (scale + 1.0)
}

fn halve(dims: FieldDims) -> FieldDims {
    FieldDims::new(dims.x.div_ceil(2), dims.y.div_ceil(2), dims.z.div_ceil(2))
}

/// One multigrid level: its uniforms (dims, dx = 2^l·dx0, same h and
/// open_mask), its right-hand side and correction fields, and its solid mask
/// (None on levels with no solid cell, or when the domain has no collider).
///
/// Level 0 solves on the caller's `p` and `div`, so it holds only `tmp`; the
/// coarsest level computes no residual, so it has no `tmp`.
pub struct Level {
    u: Uniforms,
    /// This level's dims, bound as the other grid by its neighbours' kernels.
    dims: wgpu::Buffer,
    rhs: Option<Field>,
    e: Option<Field>,
    tmp: Option<Field>,
    mask: Option<Field>,
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
}

impl Hierarchy {
    /// Build the levels and record, into `batch`, the restriction of the
    /// fine solid mask to every coarse level.
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
        let mut dims = vec![fine.cells];
        loop {
            let d = *dims.last().expect("the fine level");
            if d.x.min(d.y).min(d.z) <= COARSEST {
                break;
            }
            let next = halve(d);
            if next == d {
                break;
            }
            dims.push(next);
        }
        let depth = dims.len();

        let mut levels: Vec<Level> = Vec::with_capacity(depth);
        for (l, &cells) in dims.iter().enumerate() {
            let c = StepConstants {
                cells,
                dx: fine.dx * (1u32 << l) as f32,
                ..*fine
            };
            let weight = open_weight(l);
            let u = Uniforms::with_open_weight(gpu, &c, weight)?;
            // Beyond an open face the correction falls linearly to 0 at the
            // boundary, so the ghost is (1 − weight) times its neighbour.
            let dims_buf = dims_buffer(gpu, cells, 1.0 - weight)?;
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
                let prev = &levels[l - 1];
                let fine_view = match &prev.mask {
                    Some(m) => m.view(),
                    None => fine_mask.expect("checked above").view(),
                };
                restrict_mask_view(gpu, cache, batch, &u, fine_view, &prev.dims, &out)?;
                Some(out)
            } else {
                None
            };
            levels.push(Level {
                u,
                dims: dims_buf,
                rhs,
                e,
                tmp,
                mask,
            });
        }
        Ok(Self {
            levels,
            fine_mask: fine_mask.map(|m| m.view().clone()),
            sum: ReduceTarget::new(gpu, 1)?,
        })
    }

    /// Number of levels, the fine level included.
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Return every field to `pool`. The batch that used them must have been
    /// submitted.
    pub fn release(self, pool: &mut FieldPool) {
        for level in self.levels {
            for field in [level.rhs, level.e, level.tmp, level.mask]
                .into_iter()
                .flatten()
            {
                pool.release(field);
            }
        }
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
}

/// The bind groups one level's part of a V-cycle records.
struct Recorded {
    smoother: Smoother,
    /// Residual, restriction to the next level, prolongation from it; absent
    /// on the coarsest level.
    transfer: Option<[wgpu::BindGroup; 3]>,
}

/// `cycles` V-cycles on `p` for ∇²p = div/h, starting from what `p` holds.
/// In a closed domain each coarse right-hand side has its mean removed, and
/// so does `p` after the last cycle, as `solve_pressure` does.
pub fn v_cycles(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    h: &Hierarchy,
    p: &Field,
    div: &Field,
    cycles: u32,
) -> Result<(), GpuError> {
    let fine = &h.levels[0].u;
    expect_dims("v_cycles p", p, fine.cells())?;
    expect_dims("v_cycles div", div, fine.cells())?;
    let closed = fine.open_mask() == 0;
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
    for l in 0..depth {
        let level = &h.levels[l];
        let smoother = Smoother::with_view(gpu, cache, &level.u, x(l), b(l), h.mask(l))?;
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
                ],
            )?;
            let restrict = bind_group(
                gpu,
                &restrict_pipe,
                &[
                    Bind::Tex(tmp),
                    Bind::View(h.mask(l)),
                    Bind::Tex(b(l + 1)),
                    Bind::Tex(x(l + 1)),
                    Bind::Buf(next.u.any()),
                    Bind::Buf(&level.dims),
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
                    Bind::Buf(&next.dims),
                ],
            )?;
            Some([residual, restrict, prolong])
        } else {
            None
        };
        recorded.push(Recorded { smoother, transfer });
    }

    for cycle in 0..cycles {
        if cycle > 0 {
            // One V-cycle per submission keeps each well inside GPU
            // watchdog limits at any resolution.
            batch.flush(gpu)?;
        }
        // Down: pre-smooth, then hand the residual to the next level.
        for (l, (level, rec)) in h.levels.iter().zip(&recorded).enumerate() {
            let Some([residual, restrict, _]) = &rec.transfer else {
                rec.smoother.record(batch, COARSEST_SWEEPS, false);
                break;
            };
            rec.smoother.record(batch, SMOOTHING_SWEEPS, false);
            batch.dispatch(&residual_pipe, residual, level.u.cells());
            let next = &h.levels[l + 1];
            batch.dispatch(&restrict_pipe, restrict, next.u.cells());
            if closed {
                // The Neumann problem is solvable only for a zero-mean
                // right-hand side; restriction does not keep the mean exact.
                remove_mean(gpu, cache, batch, &next.u, b(l + 1), &h.sum)?;
            }
        }
        // Up: add the coarse correction, then post-smooth in the reverse
        // colour order so the cycle is symmetric.
        for l in (0..depth.saturating_sub(1)).rev() {
            let rec = &recorded[l];
            let [_, _, prolong] = rec.transfer.as_ref().expect("not the coarsest level");
            batch.dispatch(&prolong_pipe, prolong, h.levels[l].u.cells());
            rec.smoother.record(batch, SMOOTHING_SWEEPS, true);
        }
    }
    if closed {
        remove_mean(gpu, cache, batch, fine, p, &h.sum)?;
    }
    Ok(())
}

/// r = div − h·L(p) on the level's grid, in the same units as `div`, so the
/// existing smoother solves a level's correction with the level's own
/// `div`-shaped right-hand side. Solid cells get 0. Exposed for tests and
/// for MGPCG.
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
    expect_dims("residual p", p, u.cells())?;
    expect_dims("residual div", div, u.cells())?;
    expect_dims("residual output", out, u.cells())?;
    let solid = mask_view("residual mask", u, mask)?;
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
    let fine_dims = dims_buffer(gpu, fine_mask.dims(), 0.0)?;
    restrict_mask_view(gpu, cache, batch, coarse, fine_mask.view(), &fine_dims, out)
}

fn restrict_mask_view(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    coarse: &Uniforms,
    fine_mask: &wgpu::TextureView,
    fine_dims: &wgpu::Buffer,
    out: &Field,
) -> Result<(), GpuError> {
    expect_dims("restrict_mask output", out, coarse.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.mg.restrict_mask", RESTRICT_MASK, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::View(fine_mask),
            Bind::Tex(out),
            Bind::Buf(coarse.any()),
            Bind::Buf(fine_dims),
        ],
    )?;
    batch.dispatch(&pipeline, &group, coarse.cells());
    Ok(())
}
