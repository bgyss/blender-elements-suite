//! Solver kernels. Each function records dispatches into a `ComputeBatch`;
//! nothing runs until the batch is submitted.
//!
//! Every function has the shape `(gpu, cache, batch, uniforms, fields…)`.
//! Conventions (indexing, face positions, boundaries) are in the plan and in
//! `shaders/common.wgsl`.

mod advect;
mod forces;
mod project;
mod vorticity;

pub use advect::{Advection, Carried, Pass, advect, maccormack};
pub use forces::{blend_velocity, buoyancy, emit, wind};
pub use project::{
    CELL_SWEEPS_PER_SUBMIT, divergence, iterations_per_submit, pressure, remove_mean,
    solve_pressure, subtract_gradient,
};
pub use vorticity::{confine, curl};

use elements_core::gpu::{Axis, Field, FieldDims, GpuContext, GpuError};
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
    /// Wind, a uniform acceleration, m/s² (spec §3.4).
    pub wind: [f32; 3],
}

impl StepConstants {
    /// No buoyancy, no dissipation, no vorticity confinement, MacCormack
    /// advection and 2a's boundaries. Build variations with
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
            wind: [0.0; 3],
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
    face_accel: f32,
    _pad: [u32; 2],
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
    wind: [f32; 3],
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
                face_accel: if axis < 3 { c.wind[axis as usize] } else { 0.0 },
                _pad: [0; 2],
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
        Ok(Self {
            faces,
            density,
            temperature,
            cells: c.cells,
            open_mask: c.open_mask,
            wind: c.wind,
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

    /// Wind, as in `StepConstants::wind`.
    pub(crate) fn wind(&self) -> [f32; 3] {
        self.wind
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

/// One bind group entry. Entries are bound at 0, 1, 2… in order.
pub(crate) enum Bind<'a> {
    Tex(&'a Field),
    Buf(&'a wgpu::Buffer),
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
