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
pub use project::{divergence, pressure, subtract_gradient};

use elements_core::gpu::{Axis, Field, FieldDims, GpuContext, GpuError};
use wgpu::util::DeviceExt;

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
    alpha: f32,
    beta: f32,
    _pad: [u32; 3],
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
                alpha: c.alpha,
                beta: c.beta,
                _pad: [0; 3],
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
        })
    }

    /// The domain these constants describe.
    pub fn cells(&self) -> FieldDims {
        self.cells
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
