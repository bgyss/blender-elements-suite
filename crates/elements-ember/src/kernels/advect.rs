//! Advection (stages 3 and 5): RK2 semi-Lagrangian passes, and MacCormack's
//! correction (spec §4.1).

use elements_core::gpu::{
    Axis, ComputeBatch, Field, FieldDims, GpuContext, GpuError, PipelineCache, StaggeredField,
};
use serde::{Deserialize, Serialize};

use super::{Bind, Solids, Uniforms, bind_group, expect_dims, solid_views};

const ADVECT: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/velocity.wgsl"),
    include_str!("shaders/advect.wgsl"),
);

const MACCORMACK: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/velocity.wgsl"),
    include_str!("shaders/maccormack.wgsl"),
);

/// How the solver advects velocity and scalars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Advection {
    /// One RK2 semi-Lagrangian pass: cheap, and diffusive.
    #[serde(rename = "semi_lagrangian")]
    SemiLagrangian,
    /// Forward, backward and a clamped correction: sharper, three passes.
    /// What Blender's Mantaflow gas uses (`order=2`).
    #[serde(rename = "maccormack")]
    MacCormack,
}

/// The grid an advection pass carries. It picks the pass's uniform: its
/// offset, its dims and, for a scalar, its dissipation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Carried {
    Face(Axis),
    Density,
    Temperature,
}

impl Carried {
    /// The face axis this grid lies on, or `None` for a cell-centred scalar.
    fn axis(self) -> Option<Axis> {
        match self {
            Self::Face(axis) => Some(axis),
            Self::Density | Self::Temperature => None,
        }
    }

    /// The texel dims of this grid in a domain of `cells`.
    pub fn dims(self, cells: FieldDims) -> FieldDims {
        match self {
            Self::Face(axis) => StaggeredField::face_dims(cells, axis),
            Self::Density | Self::Temperature => cells,
        }
    }
}

/// One semi-Lagrangian pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    /// The whole of plain semi-Lagrangian advection, dissipation included.
    SemiLagrangian,
    /// MacCormack's forward pass, q̂ = A(q).
    Forward,
    /// MacCormack's backward pass, q̃ = A⁻¹(q̂).
    Backward,
}

fn check(
    what: &str,
    u: &Uniforms,
    carried: Carried,
    velocity: &StaggeredField,
    fields: &[&Field],
) -> Result<(), GpuError> {
    if velocity.cells() != u.cells() {
        return Err(GpuError::Validation(format!(
            "{what}: velocity {:?}, domain {:?}",
            velocity.cells(),
            u.cells()
        )));
    }
    let dims = carried.dims(u.cells());
    for field in fields {
        expect_dims(what, field, dims)?;
    }
    Ok(())
}

/// Record one pass carrying `src` through `velocity` into `dst`. Solid-wall
/// faces of a velocity face grid come out zero, and faces touching `solids`
/// carry the collider's velocity; a scalar never samples a solid cell's
/// contents (spec §3.2).
#[allow(clippy::too_many_arguments)]
pub fn advect(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    carried: Carried,
    pass: Pass,
    velocity: &StaggeredField,
    src: &Field,
    dst: &Field,
    solids: Option<Solids<'_>>,
) -> Result<(), GpuError> {
    check("advect", u, carried, velocity, &[src, dst])?;
    let (solid, obstacle) = solid_views(u, solids, carried.axis())?;
    let (key, entry) = match pass {
        Pass::SemiLagrangian => ("ember.advect.semi_lagrangian", "semi_lagrangian"),
        Pass::Forward => ("ember.advect.forward", "forward"),
        Pass::Backward => ("ember.advect.backward", "backward"),
    };
    let pipeline = cache.get_or_create(gpu, key, ADVECT, entry)?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(velocity.face(Axis::X)),
            Bind::Tex(velocity.face(Axis::Y)),
            Bind::Tex(velocity.face(Axis::Z)),
            Bind::Tex(src),
            Bind::Tex(dst),
            Bind::Buf(u.carried(carried)),
            Bind::View(solid),
            Bind::View(obstacle),
        ],
    )?;
    batch.dispatch(&pipeline, &group, dst.dims());
    Ok(())
}

/// Record MacCormack's correction: `dst` = clamp(`fwd` + ½(`orig` − `bwd`)),
/// times the carried scalar's dissipation.
#[allow(clippy::too_many_arguments)]
pub fn maccormack(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    carried: Carried,
    velocity: &StaggeredField,
    orig: &Field,
    fwd: &Field,
    bwd: &Field,
    dst: &Field,
    solids: Option<Solids<'_>>,
) -> Result<(), GpuError> {
    check("maccormack", u, carried, velocity, &[orig, fwd, bwd, dst])?;
    let (solid, obstacle) = solid_views(u, solids, carried.axis())?;
    let pipeline = cache.get_or_create(gpu, "ember.maccormack", MACCORMACK, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(velocity.face(Axis::X)),
            Bind::Tex(velocity.face(Axis::Y)),
            Bind::Tex(velocity.face(Axis::Z)),
            Bind::Tex(orig),
            Bind::Tex(fwd),
            Bind::Tex(bwd),
            Bind::Tex(dst),
            Bind::Buf(u.carried(carried)),
            Bind::View(solid),
            Bind::View(obstacle),
        ],
    )?;
    batch.dispatch(&pipeline, &group, dst.dims());
    Ok(())
}
