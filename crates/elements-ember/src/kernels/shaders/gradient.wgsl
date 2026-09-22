// Stage 4c: u -= ∇φ on one axis's faces. Dispatched once per axis over that
// axis's face dims. Face i lies between cells i - 1 and i.

@group(0) @binding(0) var face: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var phi: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = params.axis;
    if (any(gid >= face_dims(axis))) {
        return;
    }
    let p = vec3<i32>(gid);
    let i = gid[axis];
    if (is_wall(axis, i)) {
        textureStore(face, p, vec4<f32>(0.0));
        return;
    }
    var e = vec3<i32>(0);
    e[axis] = 1;
    // Above the open top, φ = 0.
    var upper = 0.0;
    if (i < params.dims[axis]) {
        upper = textureLoad(phi, p, 0).x;
    }
    let lower = textureLoad(phi, p - e, 0).x;
    let u = textureLoad(face, p).x - (upper - lower) * params.inv_dx;
    textureStore(face, p, vec4<f32>(u, 0.0, 0.0, 0.0));
}
