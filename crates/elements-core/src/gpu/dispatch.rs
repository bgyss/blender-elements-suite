//! Compute pipeline caching and workgroup dispatch.

use std::collections::HashMap;
use std::sync::Arc;

use wgpu::util::DeviceExt;

use super::{Field, FieldDims, GpuContext, GpuError};

/// Every Elements compute shader uses this workgroup size.
pub const WORKGROUP: u32 = 4;

/// Caches compiled compute pipelines by a static key.
#[derive(Default)]
pub struct PipelineCache {
    pipelines: HashMap<&'static str, Arc<wgpu::ComputePipeline>>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct pipelines compiled so far.
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    pub fn get_or_create(
        &mut self,
        ctx: &GpuContext,
        key: &'static str,
        source: &str,
        entry_point: &str,
    ) -> Result<Arc<wgpu::ComputePipeline>, GpuError> {
        if let Some(p) = self.pipelines.get(key) {
            return Ok(Arc::clone(p));
        }

        let pipeline = ctx.scoped(|| {
            let module = ctx
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(key),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            Arc::new(
                ctx.device()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(key),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry_point),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    }),
            )
        })?;

        self.pipelines.insert(key, Arc::clone(&pipeline));
        Ok(pipeline)
    }
}

/// Dispatch enough workgroups to cover `dims`, one invocation per voxel.
pub fn dispatch_over_field(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    dims: FieldDims,
) -> Result<(), GpuError> {
    ctx.scoped(|| {
        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("elements-dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("elements-dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(
                dims.x.div_ceil(WORKGROUP),
                dims.y.div_ceil(WORKGROUP),
                dims.z.div_ceil(WORKGROUP),
            );
        }
        ctx.queue().submit(Some(encoder.finish()));
    })
}

/// Parameters shared by the constant shader. `dims` is padded to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConstantParams {
    dims: [u32; 3],
    value: f32,
}

/// Fill every voxel of `field` with `value`.
pub fn fill_constant(
    ctx: &GpuContext,
    cache: &mut PipelineCache,
    field: &Field,
    value: f32,
) -> Result<(), GpuError> {
    let pipeline = cache.get_or_create(
        ctx,
        "constant",
        include_str!("shaders/constant.wgsl"),
        "main",
    )?;

    let dims = field.dims();
    let params = ConstantParams {
        dims: [dims.x, dims.y, dims.z],
        value,
    };

    let bind_group = ctx.scoped(|| {
        let uniform = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("constant-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("constant-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(field.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
            ],
        })
    })?;

    dispatch_over_field(ctx, &pipeline, &bind_group, dims)
}
