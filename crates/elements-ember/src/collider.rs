//! `ember.collider`: a keyframed sphere or box that the fluid cannot enter
//! (2b-2 spec §2.3). It outputs a signed-distance field and the collider's
//! velocity at every face, the pair piece 4's mesh voxelizer will also produce.

use elements_core::gpu::{
    Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField,
};
use elements_core::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};
use serde::{Deserialize, Serialize};

use crate::kernels::{Bind, axis_index, bind_group, uniform_buffer};
use crate::node_util::produce;
use crate::params;
use crate::transform::{Pose, Shape, ShapeGpu, Transform};

pub const KIND: &str = "ember.collider";

const CELLS_WGSL: &str = concat!(
    include_str!("kernels/shaders/collider_common.wgsl"),
    include_str!("kernels/shaders/shape.wgsl"),
    include_str!("kernels/shaders/collider_cells.wgsl"),
);

const FACES_WGSL: &str = concat!(
    include_str!("kernels/shaders/collider_common.wgsl"),
    include_str!("kernels/shaders/shape.wgsl"),
    include_str!("kernels/shaders/collider_faces.wgsl"),
);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColliderParams {
    pub shape: Shape,
    pub transform: Transform,
}

/// A collider's two outputs.
#[derive(Clone, Copy)]
pub struct ColliderFields<'a> {
    /// Signed distance, metres, negative inside.
    pub sdf: &'a Field,
    /// The collider's velocity at every face, m/s.
    pub velocity: &'a StaggeredField,
}

impl ColliderFields<'_> {
    pub(crate) fn check(&self, what: &str) -> Result<(), GpuError> {
        if self.velocity.cells() != self.sdf.dims() {
            return Err(GpuError::Validation(format!(
                "{what}: velocity {:?}, sdf {:?}",
                self.velocity.cells(),
                self.sdf.dims()
            )));
        }
        Ok(())
    }
}

/// Matches `Collider` in collider_common.wgsl, 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ColliderGpu {
    dims: [u32; 3],
    dx: f32,
    axis: u32,
    _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<ColliderGpu>() == 32);

/// Write `params`' SDF and velocity at `pose` into `out`. Submits its own batch.
pub fn fill_collider(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    params: &ColliderParams,
    pose: &Pose,
    dx: f32,
    out: ColliderFields<'_>,
) -> Result<(), GpuError> {
    out.check("fill_collider")?;
    let cells = out.sdf.dims();
    let dims = [cells.x, cells.y, cells.z];
    let cell_pipe = cache.get_or_create(gpu, "ember.collider.cells", CELLS_WGSL, "main")?;
    let face_pipe = cache.get_or_create(gpu, "ember.collider.faces", FACES_WGSL, "main")?;
    let shape = uniform_buffer(
        gpu,
        "ember-shape",
        bytemuck::bytes_of(&ShapeGpu::new(&params.shape, pose)),
    )?;
    let make = |axis: u32| ColliderGpu {
        dims,
        dx,
        axis,
        _pad: [0; 3],
    };
    let mut batch = ComputeBatch::new();
    let cell_params = uniform_buffer(gpu, "ember-collider", bytemuck::bytes_of(&make(0)))?;
    let group = bind_group(
        gpu,
        &cell_pipe,
        &[
            Bind::Tex(out.sdf),
            Bind::Buf(&cell_params),
            Bind::Buf(&shape),
        ],
    )?;
    batch.dispatch(&cell_pipe, &group, cells);
    for axis in Axis::ALL {
        let face_params = uniform_buffer(
            gpu,
            "ember-collider",
            bytemuck::bytes_of(&make(axis_index(axis))),
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
pub struct Collider {
    params: ColliderParams,
}

impl Node for Collider {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Field, SocketType::VectorField],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let time = ctx.time();
        let pose = self.params.transform.pose(f64::from(time.frame), time.dt);
        let dx = ctx.voxel_size();
        let params = &self.params;
        produce(ctx, 1, |gpu, cache, cells, velocity| {
            fill_collider(
                gpu,
                cache,
                params,
                &pose,
                dx,
                ColliderFields {
                    sdf: &cells[0],
                    velocity,
                },
            )
        })
    }
}

pub(crate) fn build(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    let p: ColliderParams = params::parse(KIND, params)?;
    p.shape.validate(KIND)?;
    p.transform.validate(KIND)?;
    Ok(Box::new(Collider { params: p }))
}
