//! Vorticity confinement (spec §4.4), between buoyancy and velocity advection.

use elements_core::gpu::{
    Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField,
};

use super::{Bind, Uniforms, bind_group, expect_dims};

const CURL: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/curl.wgsl"),
);

const CONFINE: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/confine.wgsl"),
);

fn check(
    what: &str,
    u: &Uniforms,
    velocity: &StaggeredField,
    omega: [&Field; 4],
) -> Result<(), GpuError> {
    if velocity.cells() != u.cells() {
        return Err(GpuError::Validation(format!(
            "{what}: velocity {:?}, domain {:?}",
            velocity.cells(),
            u.cells()
        )));
    }
    for field in omega {
        expect_dims(what, field, u.cells())?;
    }
    Ok(())
}

/// Record ω = ∇ × `velocity` into `omega`: its x, y and z components, then
/// |ω|, all at cell centres.
pub fn curl(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    omega: [&Field; 4],
) -> Result<(), GpuError> {
    check("curl", u, velocity, omega)?;
    let pipeline = cache.get_or_create(gpu, "ember.curl", CURL, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(velocity.face(Axis::X)),
            Bind::Tex(velocity.face(Axis::Y)),
            Bind::Tex(velocity.face(Axis::Z)),
            Bind::Tex(omega[0]),
            Bind::Tex(omega[1]),
            Bind::Tex(omega[2]),
            Bind::Tex(omega[3]),
            Bind::Buf(u.any()),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// Record u += h·ε·dx·(N × ω) on every non-wall face of `velocity`, from the
/// `omega` that `curl` wrote.
pub fn confine(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    omega: [&Field; 4],
) -> Result<(), GpuError> {
    check("confine", u, velocity, omega)?;
    let pipeline = cache.get_or_create(gpu, "ember.confine", CONFINE, "main")?;
    for axis in Axis::ALL {
        let face = velocity.face(axis);
        let group = bind_group(
            gpu,
            &pipeline,
            &[
                Bind::Tex(face),
                Bind::Tex(omega[0]),
                Bind::Tex(omega[1]),
                Bind::Tex(omega[2]),
                Bind::Tex(omega[3]),
                Bind::Buf(u.axis(axis)),
            ],
        )?;
        batch.dispatch(&pipeline, &group, face.dims());
    }
    Ok(())
}
