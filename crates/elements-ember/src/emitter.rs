//! `ember.sphere_emitter`: a sphere that emits density and temperature.
//!
//! Stateless. Its outputs are rates per second; the solver scales them by
//! the substep length when it adds them.

use elements_core::gpu::{ComputeBatch, Field, FieldFormat, GpuContext, GpuError, PipelineCache};
use elements_core::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;

use crate::params;

pub const KIND: &str = "ember.sphere_emitter";

/// A sphere in metres, relative to the domain's minimum corner.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
    /// Density added per second where the sphere is fully occupied.
    #[serde(default)]
    pub density_rate: f32,
    /// Temperature added per second where the sphere is fully occupied.
    #[serde(default)]
    pub temperature_rate: f32,
}

/// Matches `SphereParams` in `sphere.wgsl`: `vec3` members align to 16, so
/// `center` starts at byte 16 and the struct pads to 48.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SphereGpu {
    dims: [u32; 3],
    dx: f32,
    center: [f32; 3],
    radius: f32,
    density_rate: f32,
    temperature_rate: f32,
    _pad: [u32; 2],
}

/// Write `sphere`'s density and temperature rates into two fields of the same dims.
pub fn fill_sphere(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    density: &Field,
    temperature: &Field,
    sphere: &Sphere,
    dx: f32,
) -> Result<(), GpuError> {
    let dims = density.dims();
    if temperature.dims() != dims {
        return Err(GpuError::Validation(format!(
            "fill_sphere: density is {dims:?}, temperature is {:?}",
            temperature.dims()
        )));
    }
    let pipeline = cache.get_or_create(
        gpu,
        "ember.sphere",
        include_str!("kernels/shaders/sphere.wgsl"),
        "main",
    )?;
    let params = SphereGpu {
        dims: [dims.x, dims.y, dims.z],
        dx,
        center: sphere.center,
        radius: sphere.radius,
        density_rate: sphere.density_rate,
        temperature_rate: sphere.temperature_rate,
        _pad: [0; 2],
    };
    let bind_group = gpu.scoped(|| {
        let uniform = gpu
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ember-sphere-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ember-sphere"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(density.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(temperature.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        })
    })?;
    let mut batch = ComputeBatch::new();
    batch.dispatch(&pipeline, &bind_group, dims);
    batch.submit(gpu)
}

#[derive(Debug, Clone)]
pub struct SphereEmitter {
    sphere: Sphere,
}

impl Node for SphereEmitter {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Field, SocketType::Field],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let density = ctx.acquire_uninit(FieldFormat::R32Float)?;
        let temperature = match ctx.acquire_uninit(FieldFormat::R32Float) {
            Ok(field) => field,
            Err(e) => {
                ctx.release(Value::Field(density));
                return Err(e);
            }
        };
        let dx = ctx.voxel_size();
        let sphere = self.sphere;
        if let Err(e) =
            ctx.with_gpu(|gpu, cache| fill_sphere(gpu, cache, &density, &temperature, &sphere, dx))
        {
            ctx.release(Value::Field(density));
            ctx.release(Value::Field(temperature));
            return Err(e);
        }
        Ok(vec![Value::Field(density), Value::Field(temperature)])
    }
}

pub(crate) fn build(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    let sphere: Sphere = params::parse(KIND, params)?;
    params::finite(KIND, "center", &sphere.center)?;
    params::finite(
        KIND,
        "radius and rates",
        &[sphere.radius, sphere.density_rate, sphere.temperature_rate],
    )?;
    if sphere.radius <= 0.0 {
        return Err(params::bad(
            KIND,
            format!("radius must be positive, got {}", sphere.radius),
        ));
    }
    Ok(Box::new(SphereEmitter { sphere }))
}
