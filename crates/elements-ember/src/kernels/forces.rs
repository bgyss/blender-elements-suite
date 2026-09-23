//! Emission, velocity emission, buoyancy and wind (stages 1 and 2).

use elements_core::gpu::{
    Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField,
};

use super::{Bind, Solids, Uniforms, bind_group, expect_dims, solid_views};

const ADD_SCALED: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/add_scaled.wgsl"),
);

const BUOYANCY: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/buoyancy.wgsl"),
);

const BLEND: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/weights.wgsl"),
    include_str!("shaders/blend.wgsl"),
);

const WIND: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/wind.wgsl"),
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

/// Add buoyancy to the z faces of the velocity, except faces touching `solids`.
#[allow(clippy::too_many_arguments)]
pub fn buoyancy(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity_z: &Field,
    density: &Field,
    temperature: &Field,
    solids: Option<Solids<'_>>,
) -> Result<(), GpuError> {
    let cells = u.cells();
    expect_dims(
        "buoyancy z face",
        velocity_z,
        StaggeredField::face_dims(cells, Axis::Z),
    )?;
    expect_dims("buoyancy density", density, cells)?;
    expect_dims("buoyancy temperature", temperature, cells)?;
    let (solid, _) = solid_views(u, solids, None)?;
    let pipeline = cache.get_or_create(gpu, "ember.buoyancy", BUOYANCY, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(velocity_z),
            Bind::Tex(density),
            Bind::Tex(temperature),
            Bind::Buf(u.any()),
            Bind::View(solid),
        ],
    )?;
    batch.dispatch(&pipeline, &group, velocity_z.dims());
    Ok(())
}

/// Pull every face that is neither a wall nor touching `solids` toward
/// `target` by 1 − exp(−w·h), where w is the face's velocity weight (spec §3.3).
#[allow(clippy::too_many_arguments)]
pub fn blend_velocity(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    weight: &Field,
    target: &StaggeredField,
    solids: Option<Solids<'_>>,
) -> Result<(), GpuError> {
    let cells = u.cells();
    expect_dims("blend weight", weight, cells)?;
    if velocity.cells() != cells || target.cells() != cells {
        return Err(GpuError::Validation(format!(
            "blend_velocity: velocity {:?}, target {:?}, domain {cells:?}",
            velocity.cells(),
            target.cells()
        )));
    }
    let (solid, _) = solid_views(u, solids, None)?;
    let pipeline = cache.get_or_create(gpu, "ember.blend", BLEND, "main")?;
    for axis in Axis::ALL {
        let face = velocity.face(axis);
        let group = bind_group(
            gpu,
            &pipeline,
            &[
                Bind::Tex(face),
                Bind::Tex(weight),
                Bind::Tex(target.face(axis)),
                Bind::Buf(u.axis(axis)),
                Bind::View(solid),
            ],
        )?;
        batch.dispatch(&pipeline, &group, face.dims());
    }
    Ok(())
}

/// Add h·wind at every face that is neither a wall nor touching `solids`, on
/// each axis whose wind is nonzero (spec §3.4).
pub fn wind(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    solids: Option<Solids<'_>>,
) -> Result<(), GpuError> {
    if velocity.cells() != u.cells() {
        return Err(GpuError::Validation(format!(
            "wind: velocity {:?}, domain {:?}",
            velocity.cells(),
            u.cells()
        )));
    }
    let (solid, _) = solid_views(u, solids, None)?;
    let pipeline = cache.get_or_create(gpu, "ember.wind", WIND, "main")?;
    for (axis, a) in Axis::ALL.into_iter().zip(u.wind()) {
        if a == 0.0 {
            continue;
        }
        let face = velocity.face(axis);
        let group = bind_group(
            gpu,
            &pipeline,
            &[Bind::Tex(face), Bind::Buf(u.axis(axis)), Bind::View(solid)],
        )?;
        batch.dispatch(&pipeline, &group, face.dims());
    }
    Ok(())
}
