//! Compute pipeline caching and workgroup dispatch.

use std::collections::HashMap;
use std::sync::Arc;

use wgpu::util::DeviceExt;

use super::{Field, FieldDims, GpuContext, GpuError};

/// Every Elements compute shader uses this workgroup size.
pub const WORKGROUP: u32 = 4;

/// A cached pipeline plus the source it was compiled from.
///
/// Storing `source` lets `get_or_create` detect a key collision: two
/// different shaders accidentally registered under the same `&'static str`
/// key. Without this, a cache hit would silently hand back the wrong
/// compiled pipeline and run the wrong shader, with no error and nothing
/// pointing at the cache.
struct CachedPipeline {
    pipeline: Arc<wgpu::ComputePipeline>,
    source: String,
}

/// Caches compiled compute pipelines by a static key.
#[derive(Default)]
pub struct PipelineCache {
    pipelines: HashMap<&'static str, CachedPipeline>,
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
        if let Some(cached) = self.pipelines.get(key) {
            if cached.source != source {
                // `elementsd` ships release builds, where `debug_assert_eq!`
                // compiles out entirely -- a promise ("a cache hit cannot
                // silently return the wrong pipeline") that a debug-only
                // check cannot keep. This must be a real, always-on check:
                // a stale or reused key returning the wrong compiled
                // pipeline would run the wrong shader with no error at all.
                return Err(GpuError::Validation(format!(
                    "PipelineCache key collision: \"{key}\" was registered with different source"
                )));
            }
            return Ok(Arc::clone(&cached.pipeline));
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

        self.pipelines.insert(
            key,
            CachedPipeline {
                pipeline: Arc::clone(&pipeline),
                source: source.to_string(),
            },
        );
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

/// Parameters for the accumulate shader: `vec3<u32>` then `f32` pack into 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AccumulateParams {
    dims: [u32; 3],
    dt: f32,
}

/// `sum += input * dt`, in place on the GPU.
pub fn accumulate_into(
    ctx: &GpuContext,
    cache: &mut PipelineCache,
    sum: &Field,
    input: &Field,
    dt: f32,
) -> Result<(), GpuError> {
    if sum.dims() != input.dims() {
        return Err(GpuError::Validation(format!(
            "accumulate_into: sum is {:?}, input is {:?}",
            sum.dims(),
            input.dims()
        )));
    }
    let pipeline = cache.get_or_create(
        ctx,
        "accumulate",
        include_str!("shaders/accumulate.wgsl"),
        "main",
    )?;

    let dims = sum.dims();
    let params = AccumulateParams {
        dims: [dims.x, dims.y, dims.z],
        dt,
    };

    let bind_group = ctx.scoped(|| {
        let uniform = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("accumulate-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("accumulate-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(sum.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(input.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        })
    })?;

    dispatch_over_field(ctx, &pipeline, &bind_group, dims)
}

/// Parameters for the curl-noise shader. Laid out to match the WGSL struct.
///
/// WGSL alignment/offsets for `Params` (see `shaders/curl_noise.wgsl`):
///   dims: vec3<u32>   -> align 16, offset  0..12 (16 bytes reserved: vec3 aligns to 16)
///   frequency: f32    -> align 4,  offset 12..16
///   seed_lo: u32      -> align 4,  offset 16..20
///   seed_hi: u32      -> align 4,  offset 20..24
///   _pad0: u32        -> align 4,  offset 24..28
///   _pad1: u32        -> align 4,  offset 28..32
///   total size: 32 bytes
///
/// Rust `#[repr(C)]` layout for `NoiseParams` (no implicit vec3 padding,
/// fields laid out sequentially with each field's own natural alignment,
/// all 4-byte aligned here so no inter-field padding is inserted):
///   dims: [u32; 3]    -> offset  0..12
///   frequency: f32    -> offset 12..16
///   seed_lo: u32      -> offset 16..20
///   seed_hi: u32      -> offset 20..24
///   _pad: [u32; 2]    -> offset 24..32
///   total size: 32 bytes
///
/// Both sides agree member-for-member at every offset, so the padding is
/// mandatory, not decorative: without `_pad`, Rust's struct would be 24
/// bytes while WGSL's uniform still expects 32, and the shader would read
/// garbage past the end of the buffer for anything laid out after this one.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NoiseParams {
    dims: [u32; 3],
    frequency: f32,
    seed_lo: u32,
    seed_hi: u32,
    _pad: [u32; 2],
}

/// Fill `field` with seeded three-octave value noise in `[-1, 1]`.
pub fn fill_curl_noise(
    ctx: &GpuContext,
    cache: &mut PipelineCache,
    field: &Field,
    seed: u64,
    frequency: f32,
) -> Result<(), GpuError> {
    let pipeline = cache.get_or_create(
        ctx,
        "curl_noise",
        include_str!("shaders/curl_noise.wgsl"),
        "main",
    )?;

    let dims = field.dims();
    let params = NoiseParams {
        dims: [dims.x, dims.y, dims.z],
        frequency,
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
        _pad: [0, 0],
    };

    let bind_group = ctx.scoped(|| {
        let uniform = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("noise-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("noise-bind-group"),
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
