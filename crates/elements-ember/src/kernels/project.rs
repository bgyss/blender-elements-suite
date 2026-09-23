//! Pressure projection (stage 4).

use elements_core::gpu::{
    Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField,
};

use super::{Bind, Uniforms, bind_group, expect_dims};

const DIVERGENCE: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/divergence.wgsl"),
);

const PRESSURE: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/pressure.wgsl"),
);

const GRADIENT: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/gradient.wgsl"),
);

fn expect_velocity(what: &str, velocity: &StaggeredField, u: &Uniforms) -> Result<(), GpuError> {
    if velocity.cells() == u.cells() {
        Ok(())
    } else {
        Err(GpuError::Validation(format!(
            "{what}: velocity {:?}, domain {:?}",
            velocity.cells(),
            u.cells()
        )))
    }
}

/// Write the divergence of `velocity` into `div`.
pub fn divergence(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    div: &Field,
) -> Result<(), GpuError> {
    expect_velocity("divergence", velocity, u)?;
    expect_dims("divergence output", div, u.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.divergence", DIVERGENCE, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(velocity.face(Axis::X)),
            Bind::Tex(velocity.face(Axis::Y)),
            Bind::Tex(velocity.face(Axis::Z)),
            Bind::Tex(div),
            Bind::Buf(u.any()),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// `iterations` red-black Gauss–Seidel sweeps on `p`, solving ∇²p = div/h,
/// in place, starting from whatever `p` holds (the warm start).
pub fn pressure(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    p: &Field,
    div: &Field,
    iterations: u32,
) -> Result<(), GpuError> {
    expect_dims("pressure p", p, u.cells())?;
    expect_dims("pressure divergence", div, u.cells())?;
    let red = cache.get_or_create(gpu, "ember.pressure.red", PRESSURE, "red")?;
    let black = cache.get_or_create(gpu, "ember.pressure.black", PRESSURE, "black")?;
    // Auto layouts are never shared between pipelines, so each colour needs
    // its own bind group. Both are built once and reused every iteration.
    let entries = [Bind::Tex(p), Bind::Tex(div), Bind::Buf(u.any())];
    let red_group = bind_group(gpu, &red, &entries)?;
    let black_group = bind_group(gpu, &black, &entries)?;
    for _ in 0..iterations {
        batch.dispatch(&red, &red_group, u.cells());
        batch.dispatch(&black, &black_group, u.cells());
    }
    Ok(())
}

/// `velocity -= h·∇p`, with solid-wall faces forced to zero.
pub fn subtract_gradient(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    p: &Field,
) -> Result<(), GpuError> {
    expect_velocity("subtract_gradient", velocity, u)?;
    expect_dims("subtract_gradient p", p, u.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.gradient", GRADIENT, "main")?;
    for axis in Axis::ALL {
        let face = velocity.face(axis);
        let group = bind_group(
            gpu,
            &pipeline,
            &[Bind::Tex(face), Bind::Tex(p), Bind::Buf(u.axis(axis))],
        )?;
        batch.dispatch(&pipeline, &group, face.dims());
    }
    Ok(())
}
