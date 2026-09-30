//! `ember.mesh_collider`: a triangle mesh the fluid cannot enter, outputting
//! the same signed-distance and face-velocity pair as `ember.collider`
//! (flamethrower roadmap spec §4, FT2).

use elements_core::gpu::{Axis, ComputeBatch, GpuContext, GpuError, PipelineCache};
use elements_core::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};
use serde::{Deserialize, Serialize};

use crate::collider::{ColliderFields, SurfaceFuel, fill_surface_load};
use crate::kernels::{Bind, axis_index, bind_group, storage_buffer, uniform_buffer};
use crate::mesh::Mesh;
use crate::node_util::produce;
use crate::params;
use crate::transform::{Pose, Shape, ShapeGpu, Transform};

pub const KIND: &str = "ember.mesh_collider";

/// The most cell-triangle tests (cells x triangles) one dispatch may run, so a
/// legal mesh cannot hold the GPU long enough for the OS watchdog to kill it.
/// Measured on an M1 Max under load with the winding-number kernel, including
/// read-back: about 2e10 tests/s on the shot's 372-triangle procedural shack
/// (`docs/bench/mesh-collider.md`: 372 triangles, gap 0.01, broken_fraction
/// 0.2, seed 7, at 128³ and 256x128x128), so this budget is roughly 0.5 s a
/// dispatch.
pub const MAX_TRIANGLE_TESTS: u64 = 10_000_000_000;

const CELLS_WGSL: &str = concat!(
    include_str!("kernels/shaders/shape.wgsl"),
    include_str!("kernels/shaders/mesh_sdf.wgsl"),
);

// The face pass is the collider's: only the pose velocity is read from
// `shape`, so the shape kind is irrelevant.
const FACES_WGSL: &str = concat!(
    include_str!("kernels/shaders/collider_common.wgsl"),
    include_str!("kernels/shaders/shape.wgsl"),
    include_str!("kernels/shaders/collider_faces.wgsl"),
);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshColliderParams {
    /// Should be a closed surface: an open mesh has no inside (its winding
    /// number stays at or below one half everywhere), so it is all outside.
    /// It must also be wound outward: a mesh wound inward (negative signed
    /// volume, winding number -1) has no inside either, so the collider
    /// silently vanishes.
    pub mesh: Mesh,
    pub transform: Transform,
    /// Metres subtracted from the signed distance: inflates the solid so
    /// planks thinner than a voxel still block flow. Zero or more.
    #[serde(default)]
    pub offset: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface_fuel: Option<SurfaceFuel>,
}

/// Matches `MeshParams` in mesh_sdf.wgsl, 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshGpu {
    dims: [u32; 3],
    dx: f32,
    tri_count: u32,
    offset: f32,
    _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<MeshGpu>() == 32);

/// Matches `ColliderGpu` in collider.rs (the shared face pass), 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FaceGpu {
    dims: [u32; 3],
    dx: f32,
    axis: u32,
    _pad: [u32; 3],
}

/// Write `params`' SDF and velocity at `pose` into `out`. Submits its own batch.
pub fn fill_mesh_collider(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    params: &MeshColliderParams,
    pose: &Pose,
    dx: f32,
    out: ColliderFields<'_>,
) -> Result<(), GpuError> {
    out.check("fill_mesh_collider")?;
    let cells = out.sdf.dims();
    let dims = [cells.x, cells.y, cells.z];
    let tests = u64::from(cells.x)
        * u64::from(cells.y)
        * u64::from(cells.z)
        * params.mesh.triangle_count() as u64;
    if tests > MAX_TRIANGLE_TESTS {
        return Err(GpuError::Validation(format!(
            "fill_mesh_collider: {} cells x {} triangles = {tests} tests, over the budget of {MAX_TRIANGLE_TESTS}",
            u64::from(cells.x) * u64::from(cells.y) * u64::from(cells.z),
            params.mesh.triangle_count()
        )));
    }
    let cell_pipe = cache.get_or_create(gpu, "ember.mesh_collider.cells", CELLS_WGSL, "main")?;
    let face_pipe = cache.get_or_create(gpu, "ember.mesh_collider.faces", FACES_WGSL, "main")?;
    // `Shape` supplies the pose; the kind and extents are unused.
    let shape = uniform_buffer(
        gpu,
        "ember-shape",
        bytemuck::bytes_of(&ShapeGpu::new(&Shape::Sphere { radius: 1.0 }, pose)),
    )?;
    let vertices: Vec<[f32; 4]> = params
        .mesh
        .positions
        .iter()
        .map(|p| [p[0], p[1], p[2], 0.0])
        .collect();
    let vertex_buf = storage_buffer(gpu, "ember-mesh-vertices", bytemuck::cast_slice(&vertices))?;
    let index_buf = storage_buffer(
        gpu,
        "ember-mesh-indices",
        bytemuck::cast_slice(&params.mesh.indices),
    )?;
    let mesh_params = uniform_buffer(
        gpu,
        "ember-mesh",
        bytemuck::bytes_of(&MeshGpu {
            dims,
            dx,
            tri_count: params.mesh.triangle_count() as u32,
            offset: params.offset,
            _pad: [0; 2],
        }),
    )?;
    let mut batch = ComputeBatch::new();
    let group = bind_group(
        gpu,
        &cell_pipe,
        &[
            Bind::Tex(out.sdf),
            Bind::Buf(&mesh_params),
            Bind::Buf(&shape),
            Bind::Buf(&vertex_buf),
            Bind::Buf(&index_buf),
        ],
    )?;
    batch.dispatch(&cell_pipe, &group, cells);
    for axis in Axis::ALL {
        let face_params = uniform_buffer(
            gpu,
            "ember-mesh-faces",
            bytemuck::bytes_of(&FaceGpu {
                dims,
                dx,
                axis: axis_index(axis),
                _pad: [0; 3],
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
pub struct MeshCollider {
    params: MeshColliderParams,
}

impl Node for MeshCollider {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![
                SocketType::Field,
                SocketType::VectorField,
                SocketType::Field,
            ],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let time = ctx.time();
        let pose = self.params.transform.pose(f64::from(time.frame), time.dt);
        let dx = ctx.voxel_size();
        let params = &self.params;
        let load = params.surface_fuel.map_or(0.0, |s| s.load);
        let mut values = produce(ctx, 2, |gpu, cache, cells, velocity| {
            fill_mesh_collider(
                gpu,
                cache,
                params,
                &pose,
                dx,
                ColliderFields {
                    sdf: &cells[0],
                    velocity,
                },
            )?;
            fill_surface_load(gpu, cache, &cells[0], load, &cells[1])
        })?;
        // `produce` returns the cell fields then the vector: [sdf, load, velocity].
        values.swap(1, 2);
        Ok(values)
    }
}

pub(crate) fn build(value: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    let p: MeshColliderParams = params::parse(KIND, value)?;
    p.mesh.validate(KIND)?;
    p.transform.validate(KIND)?;
    params::finite(KIND, "offset", &[p.offset])?;
    if p.offset < 0.0 {
        return Err(params::bad(
            KIND,
            format!("offset must not be negative, got {}", p.offset),
        ));
    }
    if let Some(s) = &p.surface_fuel {
        s.validate(KIND, &p.transform)?;
    }
    Ok(Box::new(MeshCollider { params: p }))
}
