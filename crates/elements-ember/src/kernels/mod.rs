//! Solver kernels. Each function records dispatches into a `ComputeBatch`;
//! nothing runs until the batch is submitted.
//!
//! Every function has the shape `(gpu, cache, batch, uniforms, fields…)`.
//! Conventions (indexing, face positions, boundaries) are in the plan and in
//! `shaders/common.wgsl`.

mod advect;
mod forces;
mod project;

pub use advect::{advect_scalar, advect_velocity};
pub use forces::{buoyancy, emit};
pub use project::{
    CELL_SWEEPS_PER_SUBMIT, divergence, iterations_per_submit, pressure, remove_mean,
    solve_pressure, subtract_gradient,
};

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
}

impl StepConstants {
    /// No buoyancy, and 2a's boundaries. Build variations with
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
        }
    }
}

/// Matches `Params` in `common.wgsl`, 48 bytes.
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
    _pad: u32,
}

pub(crate) fn axis_index(axis: Axis) -> u32 {
    match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

/// One uniform buffer per axis, identical except for `axis`. Built once per
/// substep and shared by every kernel in it.
pub struct Uniforms {
    per_axis: [wgpu::Buffer; 3],
    cells: FieldDims,
    open_mask: u32,
}

impl Uniforms {
    pub fn new(gpu: &GpuContext, c: &StepConstants) -> Result<Self, GpuError> {
        let make = |axis: u32| {
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
                _pad: 0,
            };
            gpu.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ember-params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        };
        let per_axis = gpu.scoped(|| [make(0), make(1), make(2)])?;
        Ok(Self {
            per_axis,
            cells: c.cells,
            open_mask: c.open_mask,
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

    pub(crate) fn axis(&self, axis: Axis) -> &wgpu::Buffer {
        &self.per_axis[axis_index(axis) as usize]
    }

    /// For kernels without an axis.
    pub(crate) fn any(&self) -> &wgpu::Buffer {
        &self.per_axis[0]
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
