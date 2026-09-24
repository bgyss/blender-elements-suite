//! Pressure projection (stage 4).

use elements_core::gpu::{
    Axis, ComputeBatch, Field, FieldDims, GpuContext, GpuError, PipelineCache, ReduceOp,
    ReduceTarget, StaggeredField, reduce,
};

use super::{Bind, Solids, Uniforms, bind_group, expect_dims, solid_views};

const DIVERGENCE: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/divergence.wgsl"),
);

const PRESSURE: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/pressure.wgsl"),
);

const GRADIENT: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/gradient.wgsl"),
);

const SUBTRACT_MEAN: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/subtract_mean.wgsl"),
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

/// About 100 ms of pressure sweeps per submission at the measured 0.19 ms
/// per iteration at 128³, well inside GPU watchdog limits (spec §4.3).
pub const CELL_SWEEPS_PER_SUBMIT: u64 = 1 << 30;

/// Pressure iterations per submission for a domain of `cells`: at least 1.
pub fn iterations_per_submit(cells: FieldDims) -> u32 {
    let per = CELL_SWEEPS_PER_SUBMIT / (cells.voxel_count() as u64).max(1);
    u32::try_from(per).unwrap_or(u32::MAX).max(1)
}

/// `iterations` red-black Gauss–Seidel sweeps on `p`, solving ∇²p = div/h,
/// in place, starting from whatever `p` holds (the warm start). The loop is
/// split across submissions of at most `per_submit` iterations each. Cells of
/// `solids` are not solved and hold 0; fluid cells treat them as walls.
#[allow(clippy::too_many_arguments)] // every arg is load-bearing; see the doc above.
pub fn pressure(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    p: &Field,
    div: &Field,
    iterations: u32,
    per_submit: u32,
    solids: Option<Solids<'_>>,
) -> Result<(), GpuError> {
    expect_dims("pressure p", p, u.cells())?;
    expect_dims("pressure divergence", div, u.cells())?;
    // Checks the collider's velocity dims too, though only the mask is read.
    solid_views(u, solids, None)?;
    let mask = solids.map(|s| s.mask);
    // A long loop is split across submissions, so one submission never runs
    // long enough to trip a GPU watchdog (risk f). Order is unchanged.
    let per_submit = per_submit.max(1);
    let mut done = 0;
    while done < iterations {
        if done > 0 {
            batch.flush(gpu)?;
        }
        let sweeps = per_submit.min(iterations - done);
        relax(gpu, cache, batch, u, p, div, sweeps, mask, false)?;
        done += sweeps;
    }
    Ok(())
}

/// The `solid` view a mask-only kernel binds: the mask, or the placeholder
/// when there are no solids. As with `solid_views`, the uniform's
/// `has_solids` must agree with `mask`.
pub(crate) fn mask_view<'a>(
    what: &str,
    u: &'a Uniforms,
    mask: Option<&'a Field>,
) -> Result<&'a wgpu::TextureView, GpuError> {
    if mask.is_some() != u.has_solids() {
        return Err(GpuError::Validation(format!(
            "{what}: a mask must be given exactly when StepConstants::has_solids is set"
        )));
    }
    match mask {
        Some(m) => {
            expect_dims(what, m, u.cells())?;
            Ok(m.view())
        }
        None => Ok(u.placeholder()),
    }
}

/// The red-black smoother bound to one grid: its pipelines and bind groups,
/// built once and recorded as many times as needed.
pub(crate) struct Smoother {
    red: std::sync::Arc<wgpu::ComputePipeline>,
    black: std::sync::Arc<wgpu::ComputePipeline>,
    red_group: wgpu::BindGroup,
    black_group: wgpu::BindGroup,
    cells: FieldDims,
}

impl Smoother {
    /// A smoother for `p` with `div` as the right-hand side, reading solids
    /// from `mask` alone.
    pub(crate) fn new(
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        u: &Uniforms,
        p: &Field,
        div: &Field,
        mask: Option<&Field>,
    ) -> Result<Self, GpuError> {
        let solid = mask_view("pressure mask", u, mask)?;
        Self::with_view(gpu, cache, u, p, div, solid)
    }

    /// As `new`, with the mask already resolved to a view (the placeholder
    /// when there are no solids).
    pub(crate) fn with_view(
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        u: &Uniforms,
        p: &Field,
        div: &Field,
        solid: &wgpu::TextureView,
    ) -> Result<Self, GpuError> {
        expect_dims("pressure p", p, u.cells())?;
        expect_dims("pressure divergence", div, u.cells())?;
        let red = cache.get_or_create(gpu, "ember.pressure.red", PRESSURE, "red")?;
        let black = cache.get_or_create(gpu, "ember.pressure.black", PRESSURE, "black")?;
        // Auto layouts are never shared between pipelines, so each colour
        // needs its own bind group.
        let entries = [
            Bind::Tex(p),
            Bind::Tex(div),
            Bind::Buf(u.any()),
            Bind::View(solid),
        ];
        let red_group = bind_group(gpu, &red, &entries)?;
        let black_group = bind_group(gpu, &black, &entries)?;
        Ok(Self {
            red,
            black,
            red_group,
            black_group,
            cells: u.cells(),
        })
    }

    /// Record `sweeps` sweeps. `reverse` runs black before red in each.
    pub(crate) fn record(&self, batch: &mut ComputeBatch, sweeps: u32, reverse: bool) {
        let (first, second) = if reverse {
            (
                (&self.black, &self.black_group),
                (&self.red, &self.red_group),
            )
        } else {
            (
                (&self.red, &self.red_group),
                (&self.black, &self.black_group),
            )
        };
        for _ in 0..sweeps {
            batch.dispatch(first.0, first.1, self.cells);
            batch.dispatch(second.0, second.1, self.cells);
        }
    }
}

/// `sweeps` red-black sweeps on `p` with `div` as the right-hand side,
/// reading solids from `mask` alone (the multigrid levels have no collider
/// velocity). `reverse` runs black before red in each sweep.
#[allow(clippy::too_many_arguments)] // mirrors `pressure`; every arg is load-bearing.
pub(crate) fn relax(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    p: &Field,
    div: &Field,
    sweeps: u32,
    mask: Option<&Field>,
    reverse: bool,
) -> Result<(), GpuError> {
    Smoother::new(gpu, cache, u, p, div, mask)?.record(batch, sweeps, reverse);
    Ok(())
}

/// Remove `field`'s mean on the GPU, with no readback: a sum reduction into
/// `sum` (one slot), then a pass subtracting sum / cells. The bind groups
/// keep `sum` alive until the batch has run.
pub fn remove_mean(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    field: &Field,
    sum: &ReduceTarget,
) -> Result<(), GpuError> {
    expect_dims("remove_mean field", field, u.cells())?;
    reduce(gpu, cache, batch, field, ReduceOp::Sum, sum, 0)?;
    let pipeline = cache.get_or_create(gpu, "ember.subtract_mean", SUBTRACT_MEAN, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(field),
            Bind::Buf(sum.buffer()),
            Bind::Buf(u.any()),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// The whole pressure solve: `iterations` red-black sweeps on `p`, starting
/// from whatever `p` holds (the warm start). In a closed domain (every face
/// a wall) the Neumann system defines p only up to a constant, so p's mean
/// is removed afterwards and the warm start cannot drift (spec §4.2).
#[allow(clippy::too_many_arguments)]
pub fn solve_pressure(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    p: &Field,
    div: &Field,
    iterations: u32,
    solids: Option<Solids<'_>>,
) -> Result<(), GpuError> {
    pressure(
        gpu,
        cache,
        batch,
        u,
        p,
        div,
        iterations,
        iterations_per_submit(u.cells()),
        solids,
    )?;
    if u.open_mask() == 0 {
        let sum = ReduceTarget::new(gpu, 1)?;
        remove_mean(gpu, cache, batch, u, p, &sum)?;
    }
    Ok(())
}

/// `velocity -= h·∇p`, with solid-wall faces forced to zero and faces
/// touching `solids` set to the collider's velocity (spec §3.2).
pub fn subtract_gradient(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    p: &Field,
    solids: Option<Solids<'_>>,
) -> Result<(), GpuError> {
    expect_velocity("subtract_gradient", velocity, u)?;
    expect_dims("subtract_gradient p", p, u.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.gradient", GRADIENT, "main")?;
    for axis in Axis::ALL {
        let face = velocity.face(axis);
        let (solid, obstacle) = solid_views(u, solids, Some(axis))?;
        let group = bind_group(
            gpu,
            &pipeline,
            &[
                Bind::Tex(face),
                Bind::Tex(p),
                Bind::Buf(u.axis(axis)),
                Bind::View(solid),
                Bind::View(obstacle),
            ],
        )?;
        batch.dispatch(&pipeline, &group, face.dims());
    }
    Ok(())
}
