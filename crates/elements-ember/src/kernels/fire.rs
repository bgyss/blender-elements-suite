//! Fire (2b-4 spec §3.2): fuel emission, the burn, and the flame output.

use elements_core::gpu::{ComputeBatch, Field, GpuContext, GpuError, PipelineCache};

use super::{Bind, Uniforms, bind_group, expect_dims};

const EMIT_FUEL: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/emit_fuel.wgsl"),
);

/// `fuel = clamp(fuel + src·h, 0, 10)`, Mantaflow's inflow clamp, and react
/// blends towards 1 by the fresh fuel's share of the clamped total
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

const BURN: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/burn.wgsl"),
);

const FLAME: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/flame.wgsl"),
);

/// Burn one substep's fuel in place: fuel, react, and the smoke and heat
/// the burning adds to density and temperature. React updates as
/// `r1 = r0 * (f1 / f0)` (divided first) but is 0 where fuel is at or below
/// 1e-6 (spec §3.2), with an exactness shortcut when `f1 == f0` (nothing
/// burned this cell); see `burn.wgsl` for why that shortcut exists and why
/// it is keyed on `f1 == f0` rather than `burn == 0`. The temperature
/// profile reads react clamped to [0, 1] (the global mass correction can
/// push react slightly above 1, which would otherwise put the temperature
/// above `max_temperature`); react itself is stored unclamped. With fire
/// off the uniform's `burn` is 0 and this would still rewrite temperature
/// where react > 0, so the solver records it only with fire on.
#[allow(clippy::too_many_arguments)]
pub fn burn(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    fuel: &Field,
    react: &Field,
    density: &Field,
    temperature: &Field,
) -> Result<(), GpuError> {
    for (what, field) in [
        ("burn fuel", fuel),
        ("burn react", react),
        ("burn density", density),
        ("burn temperature", temperature),
    ] {
        expect_dims(what, field, u.cells())?;
    }
    let pipeline = cache.get_or_create(gpu, "ember.burn", BURN, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(fuel),
            Bind::Tex(react),
            Bind::Tex(density),
            Bind::Tex(temperature),
            Bind::Buf(u.any()),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// `dst` = sqrt(clamp(react, 0, 1)): the flame (spec §3.1). React is clamped
/// to [0, 1] because the global mass correction can push it slightly above
/// 1; react itself is stored unclamped.
pub fn flame(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    react: &Field,
    dst: &Field,
) -> Result<(), GpuError> {
    expect_dims("flame react", react, u.cells())?;
    expect_dims("flame dst", dst, u.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.flame", FLAME, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[Bind::Tex(react), Bind::Tex(dst), Bind::Buf(u.any())],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}
