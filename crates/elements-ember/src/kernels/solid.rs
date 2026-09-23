//! The frame's solid mask (2b-2 spec §3.2).

use elements_core::gpu::{ComputeBatch, Field, GpuContext, GpuError, PipelineCache};

use super::{Bind, Uniforms, bind_group, expect_dims};

const SOLIDIFY: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solidify.wgsl")
);

/// Record `mask` = 1 where `sdf` < 0, else 0.
pub fn solidify(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    sdf: &Field,
    mask: &Field,
) -> Result<(), GpuError> {
    expect_dims("solidify sdf", sdf, u.cells())?;
    expect_dims("solidify mask", mask, u.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.solidify", SOLIDIFY, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[Bind::Tex(sdf), Bind::Tex(mask), Bind::Buf(u.any())],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}
