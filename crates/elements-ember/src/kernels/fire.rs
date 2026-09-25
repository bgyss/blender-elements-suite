//! Fire (2b-4 spec §3.2): fuel emission, and in Task 4 the burn and the
//! flame output.

use elements_core::gpu::{ComputeBatch, Field, GpuContext, GpuError, PipelineCache};

use super::{Bind, Uniforms, bind_group, expect_dims};

const EMIT_FUEL: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/emit_fuel.wgsl"),
);

/// `fuel += src·h`, and react blends towards 1 by the fresh fuel's share
/// (spec §3.2 step 1).
pub fn emit_fuel(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    fuel: &Field,
    react: &Field,
    src: &Field,
) -> Result<(), GpuError> {
    expect_dims("emit_fuel fuel", fuel, u.cells())?;
    expect_dims("emit_fuel react", react, u.cells())?;
    expect_dims("emit_fuel source", src, u.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.emit_fuel", EMIT_FUEL, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(fuel),
            Bind::Tex(react),
            Bind::Tex(src),
            Bind::Buf(u.any()),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}
