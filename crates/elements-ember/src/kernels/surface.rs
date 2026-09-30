//! Surface ignition (FT4 spec §3.3): the burn on band cells, the gather into
//! the fluid, and the char output.

use elements_core::gpu::{ComputeBatch, Field, GpuContext, GpuError, PipelineCache};

use super::{Bind, Solids, Uniforms, bind_group, expect_dims, solid_views};

const BURN: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/surface.wgsl"),
    include_str!("shaders/surface_burn.wgsl"),
);
const GATHER: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/surface.wgsl"),
    include_str!("shaders/surface_gather.wgsl"),
);
const CHAR: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/surface_char.wgsl"),
);

/// Ignite and burn the band cells of `load` (spec §3.3 kernel 1). `burned`
/// grows in place; `emitted` gets this substep's emission per cell.
#[allow(clippy::too_many_arguments)]
pub fn surface_burn(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    load: &Field,
    temperature: &Field,
    solids: Solids<'_>,
    burned: &Field,
    emitted: &Field,
) -> Result<(), GpuError> {
    for (what, field) in [
        ("surface_burn load", load),
        ("surface_burn temperature", temperature),
        ("surface_burn burned", burned),
        ("surface_burn emitted", emitted),
    ] {
        expect_dims(what, field, u.cells())?;
    }
    let (solid, _) = solid_views(u, Some(solids), None)?;
    let pipeline = cache.get_or_create(gpu, "ember.surface_burn", BURN, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(load),
            Bind::Tex(temperature),
            Bind::Tex(burned),
            Bind::Tex(emitted),
            Bind::Buf(u.any()),
            Bind::View(solid),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// Gather `emitted` into the fluid as a fuel rate per second (spec §3.3 kernel 2).
pub fn surface_gather(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    solids: Solids<'_>,
    emitted: &Field,
    rate: &Field,
) -> Result<(), GpuError> {
    expect_dims("surface_gather emitted", emitted, u.cells())?;
    expect_dims("surface_gather rate", rate, u.cells())?;
    let (solid, _) = solid_views(u, Some(solids), None)?;
    let pipeline = cache.get_or_create(gpu, "ember.surface_gather", GATHER, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(emitted),
            Bind::Tex(rate),
            Bind::Buf(u.any()),
            Bind::View(solid),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// `dst` = the burned fraction of `load`, 0 where there is none (spec §3.3).
pub fn surface_char(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    burned: &Field,
    load: &Field,
    dst: &Field,
) -> Result<(), GpuError> {
    expect_dims("surface_char burned", burned, u.cells())?;
    expect_dims("surface_char load", load, u.cells())?;
    expect_dims("surface_char dst", dst, u.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.surface_char", CHAR, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(burned),
            Bind::Tex(load),
            Bind::Tex(dst),
            Bind::Buf(u.any()),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}
