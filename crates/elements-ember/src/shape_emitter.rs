//! `ember.emitter`: a keyframed sphere or box that emits density,
//! temperature and velocity (2b-2 spec §2.2).

use elements_core::gpu::{
    Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField,
};
use elements_core::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};
use serde::Deserialize;

use crate::kernels::{Bind, axis_index, bind_group, expect_dims, uniform_buffer};
use crate::node_util::acquire_cells;
use crate::params;
use crate::transform::{Pose, Shape, ShapeGpu, Transform};

pub const KIND: &str = "ember.emitter";

const CELLS_WGSL: &str = concat!(
    include_str!("kernels/shaders/emitter_common.wgsl"),
    include_str!("kernels/shaders/shape.wgsl"),
    include_str!("kernels/shaders/emitter_cells.wgsl"),
);

const FACES_WGSL: &str = concat!(
    include_str!("kernels/shaders/emitter_common.wgsl"),
    include_str!("kernels/shaders/shape.wgsl"),
    include_str!("kernels/shaders/emitter_faces.wgsl"),
);

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmitterParams {
    pub shape: Shape,
    pub transform: Transform,
    /// Density added per second where fully occupied.
    #[serde(default)]
    pub density_rate: f32,
    /// Temperature added per second where fully occupied.
    #[serde(default)]
    pub temperature_rate: f32,
    /// Target velocity, m/s, world space. The emitter's own motion is added.
    #[serde(default)]
    pub velocity: [f32; 3],
    /// How fast the fluid is pulled to the target velocity, 1/s; 0 turns
    /// velocity emission off (spec §3.3).
    #[serde(default)]
    pub velocity_blend: f32,
}

impl EmitterParams {
    /// No emission at all: rates, velocity and blend are zero.
    pub fn new(shape: Shape, transform: Transform) -> Self {
        Self {
            shape,
            transform,
            density_rate: 0.0,
            temperature_rate: 0.0,
            velocity: [0.0; 3],
            velocity_blend: 0.0,
        }
    }
}

/// An emitter's four outputs.
pub struct EmitterFields<'a> {
    pub density: &'a Field,
    pub temperature: &'a Field,
    /// Occupancy × velocity_blend, 1/s.
    pub weight: &'a Field,
    pub velocity: &'a StaggeredField,
}

impl EmitterFields<'_> {
    /// Every field must match the domain `density` is sized for.
    pub(crate) fn check(&self, what: &str) -> Result<(), GpuError> {
        let cells = self.density.dims();
        expect_dims(what, self.temperature, cells)?;
        expect_dims(what, self.weight, cells)?;
        if self.velocity.cells() != cells {
            return Err(GpuError::Validation(format!(
                "{what}: velocity {:?}, domain {cells:?}",
                self.velocity.cells()
            )));
        }
        Ok(())
    }
}

/// Matches `Emitter` in `emitter_common.wgsl`, 80 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EmitterGpu {
    dims: [u32; 3],
    dx: f32,
    velocity: [f32; 3],
    axis: u32,
    density_rate: f32,
    temperature_rate: f32,
    velocity_blend: f32,
    noise_on: u32,
    noise_scale: f32,
    noise_amplitude: f32,
    noise_w: f32,
    seed_lo: u32,
    seed_hi: u32,
    _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<EmitterGpu>() == 80);

impl EmitterGpu {
    fn new(p: &EmitterParams, cells: [u32; 3], dx: f32, _seconds: f64) -> Self {
        Self {
            dims: cells,
            dx,
            velocity: p.velocity,
            axis: 0,
            density_rate: p.density_rate,
            temperature_rate: p.temperature_rate,
            velocity_blend: p.velocity_blend,
            noise_on: 0,
            noise_scale: 1.0,
            noise_amplitude: 0.0,
            noise_w: 0.0,
            seed_lo: 0,
            seed_hi: 0,
            _pad: [0; 3],
        }
    }
}

/// Write `params`' outputs at `pose` and `seconds` into `out`, in a domain
/// with voxel edge `dx`. Submits its own batch.
pub fn fill_emitter(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    params: &EmitterParams,
    pose: &Pose,
    seconds: f64,
    dx: f32,
    out: EmitterFields<'_>,
) -> Result<(), GpuError> {
    out.check("fill_emitter")?;
    let cells = out.density.dims();
    let cell_pipe = cache.get_or_create(gpu, "ember.emitter.cells", CELLS_WGSL, "main")?;
    let face_pipe = cache.get_or_create(gpu, "ember.emitter.faces", FACES_WGSL, "main")?;
    let shape = uniform_buffer(
        gpu,
        "ember-shape",
        bytemuck::bytes_of(&ShapeGpu::new(&params.shape, pose)),
    )?;
    let base = EmitterGpu::new(params, [cells.x, cells.y, cells.z], dx, seconds);
    let cell_params = uniform_buffer(gpu, "ember-emitter", bytemuck::bytes_of(&base))?;
    let mut batch = ComputeBatch::new();
    let group = bind_group(
        gpu,
        &cell_pipe,
        &[
            Bind::Tex(out.density),
            Bind::Tex(out.temperature),
            Bind::Tex(out.weight),
            Bind::Buf(&cell_params),
            Bind::Buf(&shape),
        ],
    )?;
    batch.dispatch(&cell_pipe, &group, cells);
    for axis in Axis::ALL {
        let face_params = uniform_buffer(
            gpu,
            "ember-emitter",
            bytemuck::bytes_of(&EmitterGpu {
                axis: axis_index(axis),
                ..base
            }),
        )?;
        let face = out.velocity.face(axis);
        let group = bind_group(
            gpu,
            &face_pipe,
            &[Bind::Tex(face), Bind::Buf(&face_params), Bind::Buf(&shape)],
        )?;
        batch.dispatch(&face_pipe, &group, face.dims());
    }
    batch.submit(gpu)
}

#[derive(Debug, Clone)]
pub struct Emitter {
    params: EmitterParams,
}

impl Node for Emitter {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![
                SocketType::Field,
                SocketType::Field,
                SocketType::Field,
                SocketType::VectorField,
            ],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let time = ctx.time();
        let pose = self.params.transform.pose(f64::from(time.frame), time.dt);
        let dx = ctx.voxel_size();
        let cells = acquire_cells(ctx, 3)?;
        let velocity = match ctx.acquire_vector_uninit() {
            Ok(v) => v,
            Err(e) => {
                for field in cells {
                    ctx.release(Value::Field(field));
                }
                return Err(e);
            }
        };
        let params = &self.params;
        let filled = ctx.with_gpu(|gpu, cache| {
            fill_emitter(
                gpu,
                cache,
                params,
                &pose,
                time.seconds,
                dx,
                EmitterFields {
                    density: &cells[0],
                    temperature: &cells[1],
                    weight: &cells[2],
                    velocity: &velocity,
                },
            )
        });
        let mut values: Vec<Value> = cells.into_iter().map(Value::Field).collect();
        values.push(Value::VectorField(velocity));
        if let Err(e) = filled {
            for value in values {
                ctx.release(value);
            }
            return Err(e);
        }
        Ok(values)
    }
}

pub(crate) fn build(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    let p: EmitterParams = params::parse(KIND, params)?;
    p.shape.validate(KIND)?;
    p.transform.validate(KIND)?;
    params::finite(
        KIND,
        "rates and velocity",
        &[
            p.density_rate,
            p.temperature_rate,
            p.velocity_blend,
            p.velocity[0],
            p.velocity[1],
            p.velocity[2],
        ],
    )?;
    if p.velocity_blend < 0.0 {
        return Err(params::bad(KIND, "velocity_blend must be at least 0"));
    }
    Ok(Box::new(Emitter { params: p }))
}
