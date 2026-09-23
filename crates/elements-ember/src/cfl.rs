//! CFL substepping (spec §3): how many substeps a frame needs, from the
//! fastest face velocity of the state entering it.

use elements_core::gpu::{
    Axis, ComputeBatch, GpuContext, GpuError, PipelineCache, ReduceOp, ReduceTarget,
    StaggeredField, reduce,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubstepPlan {
    /// Substeps to run this frame, 1..=max_substeps.
    pub count: u32,
    /// Whether CFL wanted more than `max_substeps`.
    pub clamped: bool,
}

/// n = clamp(ceil(max|u|·dt / (cfl·dx)), 1, max_substeps). `None` when
/// `max_speed` is not finite: the simulation has diverged.
pub fn plan_substeps(
    max_speed: f32,
    dt: f64,
    dx: f32,
    cfl: f32,
    max_substeps: u32,
) -> Option<SubstepPlan> {
    if !max_speed.is_finite() {
        return None;
    }
    let cells = f64::from(max_speed) * dt / (f64::from(cfl) * f64::from(dx));
    // `as` saturates, so an enormous but finite speed asks for u32::MAX and
    // the cap binds.
    let wanted = (cells.ceil() as u32).max(1);
    Some(SubstepPlan {
        count: wanted.min(max_substeps),
        clamped: wanted > max_substeps,
    })
}

/// The largest |u| over the three faces of `velocity`, read back to the
/// CPU; not finite if any face value is not finite.
///
/// WGSL's `max` may return either operand when one is NaN, so a max
/// reduction alone could hide a NaN. Each face's sum is reduced too, because
/// a sum does carry NaN and infinity through. A sum that overflows `f32`
/// also counts: velocities that large mean the simulation has diverged.
pub fn measure_speed(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    velocity: &StaggeredField,
) -> Result<f32, GpuError> {
    let target = ReduceTarget::new(gpu, 6)?;
    let mut batch = ComputeBatch::new();
    for (slot, axis) in (0u32..).zip(Axis::ALL) {
        let face = velocity.face(axis);
        reduce(
            gpu,
            cache,
            &mut batch,
            face,
            ReduceOp::MaxAbs,
            &target,
            slot,
        )?;
        reduce(
            gpu,
            cache,
            &mut batch,
            face,
            ReduceOp::Sum,
            &target,
            slot + 3,
        )?;
    }
    batch.submit(gpu)?;
    let values = target.read(gpu)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Ok(f32::NAN);
    }
    Ok(values[..3].iter().copied().fold(0.0, f32::max))
}
