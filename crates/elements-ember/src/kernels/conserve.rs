//! The mass-conserving correction for scalar advection (2b-3c spec §5).
//!
//! Semi-Lagrangian and MacCormack advection are not conservative: in a
//! divergence-free flow they gain mass (a 128³ plume gained 2–5% over 20
//! frames). Around each scalar's advection the solver measures, on the GPU,
//! what the total should become and rescales the advected field to match:
//!
//! 1. `measure_before`: M₀ = Σq, and the mass this substep removes, OUT:
//!    the first-order outflow over open faces plus what sits in solid cells
//!    (advection zeroes those, as Mantaflow's `resetInObstacle` does).
//! 2. The advection, as it has always run, dissipation included.
//! 3. `correct_after`: M₁ = Σq′, then q′ ×= s with
//!    s = decay·(M₀ − OUT) / M₁, clamped to [`MIN_SCALE`, `MAX_SCALE`].
//!
//! Two choices go beyond the spec. The target includes the scalar's
//! dissipation, decay = e^{−rate·h}: the advection has already applied it,
//! and without it the correction would undo it. And the contents of solid
//! cells count as removed: advection zeroes them, and spreading them over
//! the fluid would make a collider a source.
//!
//! A proportional rescale assumes q ≥ 0. Emitters accept negative rates, so
//! temperature can hold hot and cold at once; then Σq is far below Σ|q|, a
//! small error in it is a large relative one, and the scale would sit on its
//! clamp, compounding ×0.9 or ×1.1 every substep. So a field with any
//! negative cell before advection is left uncorrected: `measure_before`
//! records max|min(q, 0)| in slot [`NEG`], and the scaling pass does nothing
//! when it is above 0.
//!
//! Sums are of raw cell values; the cell volume cancels in the ratio. No
//! value comes back to the CPU: the scaling kernel reads the three slots and
//! computes s itself, the same arithmetic in every thread.
//!
//! The correction is global. It removes advection's net error, spread over
//! the field in proportion to each cell's value, and cannot move mass back
//! to where it belongs. The outflow is a first-order estimate per substep,
//! which the correction then enforces.

use elements_core::gpu::{
    Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, ReduceOp, ReduceTarget,
    StaggeredField, reduce,
};

use super::project::mask_view;
use super::{Bind, Carried, Uniforms, bind_group, expect_dims};

const BOUNDARY_FLUX: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/boundary_flux.wgsl"),
);

const NEGATIVE_PART: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/negative_part.wgsl"),
);

const MASS_SCALE: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/mass_scale.wgsl"),
);

/// Slot of M₀, the total before advection.
pub const M0: u32 = 0;
/// Slot of OUT, what this substep removes: outflow plus solid contents.
pub const OUT: u32 = 1;
/// Slot of M₁, the total after advection.
pub const M1: u32 = 2;
/// Slot of max|min(q, 0)| before advection: above 0 when the field has a
/// negative cell, and is then left uncorrected.
pub const NEG: u32 = 3;
/// Slots a correction's `ReduceTarget` needs.
pub const SLOTS: u32 = 4;

/// The smallest and largest scale one correction applies: a safeguard
/// (spec §5), not something a working solver should reach. Mirrors
/// `mass_scale.wgsl`.
pub const MIN_SCALE: f32 = 0.9;
pub const MAX_SCALE: f32 = 1.1;

fn expect_target(what: &str, target: &ReduceTarget) -> Result<(), GpuError> {
    if target.slots() >= SLOTS {
        Ok(())
    } else {
        Err(GpuError::Validation(format!(
            "{what}: a {}-slot target, expected at least {SLOTS}",
            target.slots()
        )))
    }
}

/// Before advecting `field`: reduce its sum into slot [`M0`] of `target`,
/// what this substep removes into slot [`OUT`], and its most negative
/// value's magnitude into slot [`NEG`]. That is, per cell of
/// `field` beside an open domain face, q · max(u_n, 0) · h / dx for each
/// such face, with `velocity` the projected velocity the advection uses and
/// u_n positive outward; plus q itself in every cell of `mask`. Wall faces
/// carry nothing. `scratch` receives the per-cell values, then min(q, 0).
#[allow(clippy::too_many_arguments)] // every arg is a distinct binding.
pub fn measure_before(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    field: &Field,
    velocity: &StaggeredField,
    scratch: &Field,
    target: &ReduceTarget,
    mask: Option<&Field>,
) -> Result<(), GpuError> {
    expect_dims("measure_before field", field, u.cells())?;
    expect_dims("measure_before scratch", scratch, u.cells())?;
    expect_target("measure_before", target)?;
    if velocity.cells() != u.cells() {
        return Err(GpuError::Validation(format!(
            "measure_before: velocity {:?}, domain {:?}",
            velocity.cells(),
            u.cells()
        )));
    }
    let solid = mask_view("measure_before mask", u, mask)?;
    reduce(gpu, cache, batch, field, ReduceOp::Sum, target, M0)?;
    let pipeline = cache.get_or_create(gpu, "ember.boundary_flux", BOUNDARY_FLUX, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(velocity.face(Axis::X)),
            Bind::Tex(velocity.face(Axis::Y)),
            Bind::Tex(velocity.face(Axis::Z)),
            Bind::Tex(field),
            Bind::Tex(scratch),
            Bind::Buf(u.any()),
            Bind::View(solid),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    reduce(gpu, cache, batch, scratch, ReduceOp::Sum, target, OUT)?;
    // Dispatches run in order, so `scratch` can be reused once OUT is summed.
    let pipeline = cache.get_or_create(gpu, "ember.negative_part", NEGATIVE_PART, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[Bind::Tex(field), Bind::Tex(scratch), Bind::Buf(u.any())],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    reduce(gpu, cache, batch, scratch, ReduceOp::MaxAbs, target, NEG)
}

/// After advecting into `advected`: reduce its sum into slot [`M1`], then
/// scale it by s = decay · (M₀ − OUT) / M₁, clamped to [`MIN_SCALE`,
/// `MAX_SCALE`], leaving it unchanged when M₁ ≤ 1e-12, M₀ − OUT ≤ 0, or
/// the field had a negative cell before advection ([`NEG`] above 0).
/// decay is `carried`'s dissipation over the substep, which the advection
/// has already applied and the target must include, or the correction
/// would undo it.
pub fn correct_after(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    carried: Carried,
    advected: &Field,
    target: &ReduceTarget,
) -> Result<(), GpuError> {
    expect_dims("correct_after field", advected, u.cells())?;
    expect_target("correct_after", target)?;
    if !matches!(carried, Carried::Density | Carried::Temperature) {
        return Err(GpuError::Validation(
            "correct_after: only cell-centred scalars are corrected".to_owned(),
        ));
    }
    reduce(gpu, cache, batch, advected, ReduceOp::Sum, target, M1)?;
    let pipeline = cache.get_or_create(gpu, "ember.mass_scale", MASS_SCALE, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(advected),
            Bind::Buf(target.buffer()),
            Bind::Buf(u.carried(carried)),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}
