//! Semi-Lagrangian advection (stages 3 and 5).

use elements_core::gpu::{
    Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField,
};

use super::{Bind, Uniforms, bind_group, expect_dims};

const ADVECT_VELOCITY: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/velocity.wgsl"),
    include_str!("shaders/advect_velocity.wgsl"),
);

const ADVECT_SCALAR: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/velocity.wgsl"),
    include_str!("shaders/advect_scalar.wgsl"),
);

/// Advect `src` through itself into `dst`. Solid-wall faces of `dst` are zero.
pub fn advect_velocity(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    src: &StaggeredField,
    dst: &StaggeredField,
) -> Result<(), GpuError> {
    if src.cells() != u.cells() || dst.cells() != u.cells() {
        return Err(GpuError::Validation(format!(
            "advect_velocity: src {:?}, dst {:?}, domain {:?}",
            src.cells(),
            dst.cells(),
            u.cells()
        )));
    }
    let pipeline = cache.get_or_create(gpu, "ember.advect_velocity", ADVECT_VELOCITY, "main")?;
    for axis in Axis::ALL {
        let group = bind_group(
            gpu,
            &pipeline,
            &[
                Bind::Tex(src.face(Axis::X)),
                Bind::Tex(src.face(Axis::Y)),
                Bind::Tex(src.face(Axis::Z)),
                Bind::Tex(dst.face(axis)),
                Bind::Buf(u.axis(axis)),
            ],
        )?;
        batch.dispatch(&pipeline, &group, dst.face(axis).dims());
    }
    Ok(())
}

/// Advect the cell-centred `src` through `velocity` into `dst`.
pub fn advect_scalar(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    src: &Field,
    dst: &Field,
) -> Result<(), GpuError> {
    let cells = u.cells();
    if velocity.cells() != cells {
        return Err(GpuError::Validation(format!(
            "advect_scalar: velocity {:?}, domain {cells:?}",
            velocity.cells()
        )));
    }
    expect_dims("advect_scalar src", src, cells)?;
    expect_dims("advect_scalar dst", dst, cells)?;
    let pipeline = cache.get_or_create(gpu, "ember.advect_scalar", ADVECT_SCALAR, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(velocity.face(Axis::X)),
            Bind::Tex(velocity.face(Axis::Y)),
            Bind::Tex(velocity.face(Axis::Z)),
            Bind::Tex(src),
            Bind::Tex(dst),
            Bind::Buf(u.any()),
        ],
    )?;
    batch.dispatch(&pipeline, &group, cells);
    Ok(())
}
