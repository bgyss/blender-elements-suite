//! Emission and buoyancy (stages 1 and 2).

use elements_core::gpu::{
    Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField,
};

use super::{Bind, Uniforms, bind_group, expect_dims};

const ADD_SCALED: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/add_scaled.wgsl"),
);

const BUOYANCY: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/buoyancy.wgsl"),
);

/// `dst += src * h`.
pub fn emit(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    dst: &Field,
    src: &Field,
) -> Result<(), GpuError> {
    expect_dims("emit dst", dst, u.cells())?;
    expect_dims("emit source", src, u.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.add_scaled", ADD_SCALED, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[Bind::Tex(dst), Bind::Tex(src), Bind::Buf(u.any())],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// Add buoyancy to the z faces of the velocity.
pub fn buoyancy(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity_z: &Field,
    density: &Field,
    temperature: &Field,
) -> Result<(), GpuError> {
    let cells = u.cells();
    expect_dims(
        "buoyancy z face",
        velocity_z,
        StaggeredField::face_dims(cells, Axis::Z),
    )?;
    expect_dims("buoyancy density", density, cells)?;
    expect_dims("buoyancy temperature", temperature, cells)?;
    let pipeline = cache.get_or_create(gpu, "ember.buoyancy", BUOYANCY, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(velocity_z),
            Bind::Tex(density),
            Bind::Tex(temperature),
            Bind::Buf(u.any()),
        ],
    )?;
    batch.dispatch(&pipeline, &group, velocity_z.dims());
    Ok(())
}
