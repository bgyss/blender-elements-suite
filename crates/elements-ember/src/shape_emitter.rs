//! `ember.emitter`: a keyframed sphere or box that emits density,
//! temperature and velocity (2b-2 spec §2.2).

use elements_core::gpu::{
    Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField, fill_constant,
};
use elements_core::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};
use serde::Deserialize;

use crate::kernels::{Bind, axis_index, bind_group, expect_dims, uniform_buffer};
use crate::node_util::produce;
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

/// The smallest noise cell a document may ask for, metres (spec §2.2).
pub const MIN_NOISE_SCALE_M: f32 = 1e-4;
/// The largest noise evolution rate, in either direction, per second
/// (spec §2.2).
pub const MAX_NOISE_EVOLUTION: f32 = 1e4;

/// Noise that modulates an emitter's rates (spec §2.2).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Noise {
    pub seed: u64,
    /// Size of one noise cell, metres; at least [`MIN_NOISE_SCALE_M`].
    pub scale_m: f32,
    /// 0 turns the noise off; 1 lets it cut emission to zero.
    pub amplitude: f32,
    /// How fast the pattern changes, per second; 0 is a still pattern. At
    /// most [`MAX_NOISE_EVOLUTION`] in magnitude.
    #[serde(default)]
    pub evolution: f32,
}

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
    /// Optional noise that multiplies the rates.
    #[serde(default)]
    pub noise: Option<Noise>,
    /// Frames on which the emitter emits, inclusive. `None` is always on.
    /// Outside the range every output is zero (2b-3 spec §3).
    #[serde(default)]
    pub active_frames: Option<[u32; 2]>,
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
            noise: None,
            active_frames: None,
        }
    }

    /// Whether the emitter emits on `frame`.
    pub fn is_active(&self, frame: u32) -> bool {
        self.active_frames
            .is_none_or(|[first, last]| (first..=last).contains(&frame))
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
    fn new(p: &EmitterParams, cells: [u32; 3], dx: f32, seconds: f64) -> Self {
        let noise = p.noise.unwrap_or(Noise {
            seed: 0,
            scale_m: 1.0,
            amplitude: 0.0,
            evolution: 0.0,
        });
        Self {
            dims: cells,
            dx,
            velocity: p.velocity,
            axis: 0,
            density_rate: p.density_rate,
            temperature_rate: p.temperature_rate,
            velocity_blend: p.velocity_blend,
            noise_on: u32::from(p.noise.is_some()),
            noise_scale: noise.scale_m,
            noise_amplitude: noise.amplitude,
            // Time enters only through this coordinate, so a frame's pattern
            // depends on the document's time alone.
            noise_w: (f64::from(noise.evolution) * seconds) as f32,
            seed_lo: noise.seed as u32,
            seed_hi: (noise.seed >> 32) as u32,
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
        if !self.params.is_active(time.frame) {
            return produce(ctx, 3, |gpu, cache, cells, velocity| {
                for field in cells {
                    fill_constant(gpu, cache, field, 0.0)?;
                }
                for axis in Axis::ALL {
                    fill_constant(gpu, cache, velocity.face(axis), 0.0)?;
                }
                Ok(())
            });
        }
        let pose = self.params.transform.pose(f64::from(time.frame), time.dt);
        let dx = ctx.voxel_size();
        let params = &self.params;
        produce(ctx, 3, |gpu, cache, cells, velocity| {
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
                    velocity,
                },
            )
        })
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
    if let Some([first, last]) = p.active_frames
        && first > last
    {
        return Err(params::bad(
            KIND,
            "active_frames must not end before it starts",
        ));
    }
    if let Some(n) = p.noise {
        params::finite(KIND, "noise", &[n.scale_m, n.amplitude, n.evolution])?;
        // A tiny scale puts many noise cells in one voxel, and the
        // position divided by it overflows f32; a huge evolution loses the
        // pattern's time coordinate to f32 rounding within seconds.
        if n.scale_m < MIN_NOISE_SCALE_M {
            return Err(params::bad(
                KIND,
                "noise scale_m must be at least 1e-4 metres",
            ));
        }
        if n.evolution.abs() > MAX_NOISE_EVOLUTION {
            return Err(params::bad(
                KIND,
                "noise evolution must be within [-1e4, 1e4] per second",
            ));
        }
        if !(0.0..=1.0).contains(&n.amplitude) {
            return Err(params::bad(KIND, "noise amplitude must be in [0, 1]"));
        }
    }
    Ok(Box::new(Emitter { params: p }))
}
