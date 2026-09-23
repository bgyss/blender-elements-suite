//! `ember.emitter_union` (and, from Task 4, `ember.collider_union`): merge two
//! emitters or two colliders into one (2b-2 spec §2.4). Chain them for more.

use elements_core::gpu::{Axis, ComputeBatch, GpuContext, GpuError, PipelineCache};
use elements_core::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};

use crate::kernels::{Bind, axis_index, bind_group, uniform_buffer};
use crate::node_util::{produce, take_inputs};
use crate::shape_emitter::EmitterFields;

pub const EMITTER_UNION_KIND: &str = "ember.emitter_union";

const EMITTER_CELLS_WGSL: &str = include_str!("kernels/shaders/emitter_union_cells.wgsl");
const EMITTER_FACES_WGSL: &str = concat!(
    include_str!("kernels/shaders/weights.wgsl"),
    include_str!("kernels/shaders/emitter_union_faces.wgsl"),
);

/// Matches `Grid` in the union shaders, 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GridGpu {
    dims: [u32; 3],
    axis: u32,
}

const _: () = assert!(std::mem::size_of::<GridGpu>() == 16);

/// `out` = the union of `a` and `b`. Submits its own batch.
pub fn union_emitters(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    a: EmitterFields<'_>,
    b: EmitterFields<'_>,
    out: EmitterFields<'_>,
) -> Result<(), GpuError> {
    a.check("union_emitters a")?;
    b.check("union_emitters b")?;
    out.check("union_emitters out")?;
    let cells = out.density.dims();
    if a.density.dims() != cells || b.density.dims() != cells {
        return Err(GpuError::Validation(
            "union_emitters: inputs differ in size".to_owned(),
        ));
    }
    let dims = [cells.x, cells.y, cells.z];
    let cell_pipe =
        cache.get_or_create(gpu, "ember.emitter_union.cells", EMITTER_CELLS_WGSL, "main")?;
    let face_pipe =
        cache.get_or_create(gpu, "ember.emitter_union.faces", EMITTER_FACES_WGSL, "main")?;
    let grid = uniform_buffer(
        gpu,
        "ember-union",
        bytemuck::bytes_of(&GridGpu { dims, axis: 0 }),
    )?;
    let mut batch = ComputeBatch::new();
    let group = bind_group(
        gpu,
        &cell_pipe,
        &[
            Bind::Tex(a.density),
            Bind::Tex(a.temperature),
            Bind::Tex(a.weight),
            Bind::Tex(b.density),
            Bind::Tex(b.temperature),
            Bind::Tex(b.weight),
            Bind::Tex(out.density),
            Bind::Tex(out.temperature),
            Bind::Tex(out.weight),
            Bind::Buf(&grid),
        ],
    )?;
    batch.dispatch(&cell_pipe, &group, cells);
    for axis in Axis::ALL {
        let grid = uniform_buffer(
            gpu,
            "ember-union",
            bytemuck::bytes_of(&GridGpu {
                dims,
                axis: axis_index(axis),
            }),
        )?;
        let dst = out.velocity.face(axis);
        let group = bind_group(
            gpu,
            &face_pipe,
            &[
                Bind::Tex(a.weight),
                Bind::Tex(b.weight),
                Bind::Tex(a.velocity.face(axis)),
                Bind::Tex(b.velocity.face(axis)),
                Bind::Tex(dst),
                Bind::Buf(&grid),
            ],
        )?;
        batch.dispatch(&face_pipe, &group, dst.dims());
    }
    batch.submit(gpu)
}

/// The four emitter outputs of `values[at..at + 4]`, type-checked.
fn emitter_fields(values: &[Value], at: usize) -> Result<EmitterFields<'_>, NodeError> {
    Ok(EmitterFields {
        density: values[at].as_field()?,
        temperature: values[at + 1].as_field()?,
        weight: values[at + 2].as_field()?,
        velocity: values[at + 3].as_vector_field()?,
    })
}

#[derive(Debug, Clone)]
pub struct EmitterUnion;

impl Node for EmitterUnion {
    fn kind(&self) -> &'static str {
        EMITTER_UNION_KIND
    }

    fn sockets(&self) -> SocketSpec {
        let group = [
            SocketType::Field,
            SocketType::Field,
            SocketType::Field,
            SocketType::VectorField,
        ];
        SocketSpec {
            inputs: [group, group].concat(),
            outputs: group.to_vec(),
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let inputs = take_inputs(ctx, 8)?;
        let result = union_node(ctx, &inputs);
        for value in inputs {
            ctx.release(value);
        }
        result
    }
}

fn union_node(ctx: &mut EvalCtx<'_>, inputs: &[Value]) -> Result<Vec<Value>, NodeError> {
    let (a, b) = (emitter_fields(inputs, 0)?, emitter_fields(inputs, 4)?);
    produce(ctx, 3, |gpu, cache, cells, velocity| {
        union_emitters(
            gpu,
            cache,
            a,
            b,
            EmitterFields {
                density: &cells[0],
                temperature: &cells[1],
                weight: &cells[2],
                velocity,
            },
        )
    })
}

pub(crate) fn build_emitter_union(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    if !(params.is_null() || params.as_object().is_some_and(|o| o.is_empty())) {
        return Err(crate::params::bad(
            EMITTER_UNION_KIND,
            "takes no parameters",
        ));
    }
    Ok(Box::new(EmitterUnion))
}
