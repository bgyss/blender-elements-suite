//! Solver kernels. Each function records dispatches into a `ComputeBatch`;
//! nothing runs until the batch is submitted.
//!
//! Every function has the shape `(gpu, cache, batch, uniforms, fields…)`.
//! Conventions (indexing, face positions, boundaries) are in the plan and in
//! `shaders/common.wgsl`.

mod advect;
pub mod conserve;
mod forces;
pub mod mgpcg;
pub mod multigrid;
mod project;
mod solid;
mod vorticity;

pub use advect::{Advection, Carried, Pass, advect, maccormack};
pub use forces::{blend_velocity, buoyancy, emit, wind};
pub use mgpcg::mgpcg;
pub use multigrid::{Hierarchy, v_cycle_from_zero, v_cycles};
pub use project::{
    CELL_SWEEPS_PER_SUBMIT, divergence, iterations_per_submit, pressure, remove_mean,
    solve_pressure, subtract_gradient,
};
pub use solid::solidify;
pub use vorticity::{confine, curl};

use elements_core::gpu::{Axis, Field, FieldDims, GpuContext, GpuError, StaggeredField};
use wgpu::util::DeviceExt;

use crate::boundaries::DEFAULT_OPEN_MASK;

/// Values every kernel in one substep shares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepConstants {
    /// The domain in cells.
    pub cells: FieldDims,
    /// Substep length, seconds.
    pub h: f32,
    /// Voxel edge, metres.
    pub dx: f32,
    /// Buoyancy per unit density (sinks), m/s².
    pub alpha: f32,
    /// Buoyancy per unit temperature (rises), m/s².
    pub beta: f32,
    /// Open domain faces; see `Boundaries::open_mask`.
    pub open_mask: u32,
    /// How velocity and scalars are advected.
    pub advection: Advection,
    /// Exponential decay rate of density, 1/s, applied in its last advection pass.
    pub density_dissipation: f32,
    /// Exponential decay rate of temperature, 1/s.
    pub temperature_dissipation: f32,
    /// Vorticity confinement strength ε, 1/s; 0 skips the stage (spec §4.4).
    pub vorticity: f32,
    /// The ambient airflow the air relaxes towards, m/s (2b-3c spec §6).
    pub wind_velocity: [f32; 3],
    /// How fast the air relaxes towards `wind_velocity`, 1/s; 0 skips the
    /// stage.
    pub wind_rate: f32,
    /// Whether this substep's kernels read a solid mask (spec §3.2).
    pub has_solids: bool,
    /// Whether each scalar advection is followed by the global mass
    /// correction (2b-3c spec §5).
    pub conserve_mass: bool,
}

impl StepConstants {
    /// No buoyancy, no dissipation, no vorticity confinement, no mass
    /// correction, MacCormack advection and 2a's boundaries. Build variations with
    /// `StepConstants { beta: 1.0, ..StepConstants::new(cells, h, dx) }`, so
    /// fields added later get their defaults here instead of breaking callers.
    pub fn new(cells: FieldDims, h: f32, dx: f32) -> Self {
        Self {
            cells,
            h,
            dx,
            alpha: 0.0,
            beta: 0.0,
            open_mask: DEFAULT_OPEN_MASK,
            advection: Advection::MacCormack,
            density_dissipation: 0.0,
            temperature_dissipation: 0.0,
            vorticity: 0.0,
            wind_velocity: [0.0; 3],
            wind_rate: 0.0,
            has_solids: false,
            conserve_mass: false,
        }
    }
}

/// Matches `Params` in `common.wgsl`, 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KernelParams {
    dims: [u32; 3],
    axis: u32,
    h: f32,
    inv_dx: f32,
    dx2: f32,
    pressure_scale: f32,
    alpha: f32,
    beta: f32,
    open_mask: u32,
    decay: f32,
    confinement: f32,
    face_wind: f32,
    has_solids: u32,
    wind_blend: f32,
}

// `Params` in `shaders/common.wgsl` is 64 bytes; a field added here without
// its WGSL twin (or padding) fails the build instead of silently shifting
// every uniform the kernels read.
const _: () = assert!(std::mem::size_of::<KernelParams>() == 64);

pub(crate) fn axis_index(axis: Axis) -> u32 {
    match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

/// One uniform buffer per grid an advection pass can carry, built once per
/// substep and shared by every kernel in it.
pub struct Uniforms {
    /// One per face axis: `axis` 0, 1, 2 and `decay` 1.
    faces: [wgpu::Buffer; 3],
    /// Cell grids (`axis` = CELL), each with its scalar's `decay`.
    density: wgpu::Buffer,
    temperature: wgpu::Buffer,
    cells: FieldDims,
    open_mask: u32,
    wind: bool,
    has_solids: bool,
    /// Bound as `solid` and `obstacle` when there are no solids. Kept so it
    /// outlives its view.
    _placeholder: wgpu::Texture,
    placeholder_view: wgpu::TextureView,
}

impl Uniforms {
    pub fn new(gpu: &GpuContext, c: &StepConstants) -> Result<Self, GpuError> {
        let make = |axis: u32, decay: f32| {
            let params = KernelParams {
                dims: [c.cells.x, c.cells.y, c.cells.z],
                axis,
                h: c.h,
                inv_dx: 1.0 / c.dx,
                dx2: c.dx * c.dx,
                pressure_scale: c.h,
                alpha: c.alpha,
                beta: c.beta,
                open_mask: c.open_mask,
                decay,
                confinement: c.vorticity * c.dx,
                face_wind: if axis < 3 {
                    c.wind_velocity[axis as usize]
                } else {
                    0.0
                },
                has_solids: u32::from(c.has_solids),
                // The fraction of the way to the wind one substep closes:
                // exact for any h, so the stage cannot overshoot.
                wind_blend: 1.0 - (-c.wind_rate * c.h).exp(),
            };
            gpu.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ember-params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        };
        const CELL: u32 = 3; // `CELL` in common.wgsl
        let (faces, density, temperature) = gpu.scoped(|| {
            (
                [make(0, 1.0), make(1, 1.0), make(2, 1.0)],
                make(CELL, (-c.density_dissipation * c.h).exp()),
                make(CELL, (-c.temperature_dissipation * c.h).exp()),
            )
        })?;
        let placeholder = gpu.scoped(|| {
            gpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("ember-no-solids"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        })?;
        let placeholder_view = placeholder.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(Self {
            faces,
            density,
            temperature,
            cells: c.cells,
            open_mask: c.open_mask,
            wind: c.wind_rate > 0.0,
            has_solids: c.has_solids,
            _placeholder: placeholder,
            placeholder_view,
        })
    }

    /// The domain these constants describe.
    pub fn cells(&self) -> FieldDims {
        self.cells
    }

    /// Open domain faces, as in `StepConstants::open_mask`.
    pub fn open_mask(&self) -> u32 {
        self.open_mask
    }

    /// Whether the wind stage runs: `StepConstants::wind_rate` is above 0.
    pub(crate) fn wind(&self) -> bool {
        self.wind
    }

    /// Whether kernels read a solid mask, as in `StepConstants::has_solids`.
    pub(crate) fn has_solids(&self) -> bool {
        self.has_solids
    }

    /// A 1×1×1 texture bound where a kernel has no mask or no obstacle to read.
    pub(crate) fn placeholder(&self) -> &wgpu::TextureView {
        &self.placeholder_view
    }

    pub(crate) fn axis(&self, axis: Axis) -> &wgpu::Buffer {
        &self.faces[axis_index(axis) as usize]
    }

    /// For kernels without an axis.
    pub(crate) fn any(&self) -> &wgpu::Buffer {
        &self.faces[0]
    }

    pub(crate) fn carried(&self, carried: Carried) -> &wgpu::Buffer {
        match carried {
            Carried::Face(axis) => self.axis(axis),
            Carried::Density => &self.density,
            Carried::Temperature => &self.temperature,
        }
    }
}

/// The frame's collider: the solid mask (1 inside) and the collider's
/// velocity at every face (spec §3.2).
#[derive(Clone, Copy)]
pub struct Solids<'a> {
    pub mask: &'a Field,
    pub velocity: &'a StaggeredField,
}

/// The `solid` and `obstacle` views a kernel binds: the mask and the
/// collider's velocity on `axis`'s faces, or the placeholder where a kernel
/// has no axis or there are no solids. The uniform's `has_solids` must agree
/// with `solids`, or kernels would read the placeholder as a mask.
pub(crate) fn solid_views<'a>(
    u: &'a Uniforms,
    solids: Option<Solids<'a>>,
    axis: Option<Axis>,
) -> Result<(&'a wgpu::TextureView, &'a wgpu::TextureView), GpuError> {
    if solids.is_some() != u.has_solids() {
        return Err(GpuError::Validation(
            "solids must be given exactly when StepConstants::has_solids is set".to_owned(),
        ));
    }
    let Some(s) = solids else {
        return Ok((u.placeholder(), u.placeholder()));
    };
    expect_dims("solid mask", s.mask, u.cells())?;
    if s.velocity.cells() != u.cells() {
        return Err(GpuError::Validation(format!(
            "collider velocity {:?}, domain {:?}",
            s.velocity.cells(),
            u.cells()
        )));
    }
    let obstacle = match axis {
        Some(a) => s.velocity.face(a).view(),
        None => u.placeholder(),
    };
    Ok((s.mask.view(), obstacle))
}

/// One bind group entry. Entries are bound at 0, 1, 2… in order.
pub(crate) enum Bind<'a> {
    Tex(&'a Field),
    Buf(&'a wgpu::Buffer),
    View(&'a wgpu::TextureView),
}

pub(crate) fn bind_group(
    gpu: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    entries: &[Bind<'_>],
) -> Result<wgpu::BindGroup, GpuError> {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: match entry {
                Bind::Tex(field) => wgpu::BindingResource::TextureView(field.view()),
                Bind::Buf(buffer) => buffer.as_entire_binding(),
                Bind::View(view) => wgpu::BindingResource::TextureView(view),
            },
        })
        .collect();
    gpu.scoped(|| {
        gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ember"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    })
}

/// A mis-sized field would make a kernel read the wrong texels with no error.
pub(crate) fn expect_dims(what: &str, field: &Field, dims: FieldDims) -> Result<(), GpuError> {
    if field.dims() == dims {
        Ok(())
    } else {
        Err(GpuError::Validation(format!(
            "{what} is {:?}, expected {dims:?}",
            field.dims()
        )))
    }
}

/// A uniform buffer holding `bytes`, created inside an error scope.
pub(crate) fn uniform_buffer(
    gpu: &GpuContext,
    label: &str,
    bytes: &[u8],
) -> Result<wgpu::Buffer, GpuError> {
    gpu.scoped(|| {
        gpu.device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    })
}
