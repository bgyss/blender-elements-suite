// ember.emitter_union, face pass: the target velocity, weighted by each
// emitter's face weight.

struct Grid {
    dims: vec3<u32>,
    axis: u32,
};

@group(0) @binding(0) var w1: texture_3d<f32>;
@group(0) @binding(1) var w2: texture_3d<f32>;
@group(0) @binding(2) var u1: texture_3d<f32>;
@group(0) @binding(3) var u2: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var<uniform> grid: Grid;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = grid.axis;
    var dims = grid.dims;
    dims[axis] = dims[axis] + 1u;
    if (any(gid >= dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let a = face_weight(w1, axis, p, grid.dims);
    let b = face_weight(w2, axis, p, grid.dims);
    let u = (a * textureLoad(u1, p, 0).x + b * textureLoad(u2, p, 0).x) / max(a + b, 1e-12);
    textureStore(dst, p, vec4<f32>(u, 0.0, 0.0, 0.0));
}
