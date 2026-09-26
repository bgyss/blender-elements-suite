//! Vorticity confinement (spec §4.4), between buoyancy and velocity advection.

use elements_core::gpu::{
    Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField,
};

use super::{Bind, Solids, Uniforms, bind_group, expect_dims, solid_views};

const CURL: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/curl.wgsl"),
);

const CONFINE: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
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
/// |ω|, all at cell centres. Differences treat cells of `solids` like the
/// domain edge.
pub fn curl(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    omega: [&Field; 4],
    solids: Option<Solids<'_>>,
) -> Result<(), GpuError> {
    check("curl", u, velocity, omega)?;
    let (solid, _) = solid_views(u, solids, None)?;
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
            Bind::View(solid),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// Record u += h·strength(c)·(N × ω) on every face of `velocity` that is
/// neither a wall nor touching `solids`, from the `omega` that `curl` wrote.
/// `strength` is ε·dx, plus flame_vorticity·dx per unit `fuel` with fire on
/// (2b-4 spec §3.2); `fuel` must be `Some` exactly when
/// `StepConstants::fire` is set.
#[allow(clippy::too_many_arguments)] // every arg is load-bearing; see the doc above.
pub fn confine(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    omega: [&Field; 4],
    fuel: Option<&Field>,
    solids: Option<Solids<'_>>,
) -> Result<(), GpuError> {
    check("confine", u, velocity, omega)?;
    if fuel.is_some() != u.fire() {
        return Err(GpuError::Validation(
            "confine: fuel must be given exactly when StepConstants::fire is set".to_owned(),
        ));
    }
    if let Some(f) = fuel {
        expect_dims("confine fuel", f, u.cells())?;
    }
    let (solid, _) = solid_views(u, solids, None)?;
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
                Bind::View(solid),
                match fuel {
                    Some(f) => Bind::Tex(f),
                    None => Bind::View(u.placeholder()),
                },
            ],
        )?;
        batch.dispatch(&pipeline, &group, face.dims());
    }
    Ok(())
}
