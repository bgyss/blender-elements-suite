// ember.collider_union, face pass: the velocity of whichever collider is
// nearer to the face.

struct Grid {
    dims: vec3<u32>,
    axis: u32,
};

@group(0) @binding(0) var s1: texture_3d<f32>;
@group(0) @binding(1) var s2: texture_3d<f32>;
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
    var u = textureLoad(u2, p, 0).x;
    if (face_min(s1, axis, p, grid.dims) <= face_min(s2, axis, p, grid.dims)) {
        u = textureLoad(u1, p, 0).x;
    }
    textureStore(dst, p, vec4<f32>(u, 0.0, 0.0, 0.0));
}
