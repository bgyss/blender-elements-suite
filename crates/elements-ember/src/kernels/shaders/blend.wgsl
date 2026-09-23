// Velocity emission (2b-2 spec §3.3): u += (u_e − u)·(1 − exp(−w·h)) on one
// axis's faces, where w is the face's velocity weight in 1/s. The exponential
// makes the result independent of the substep length. Wall faces are left alone.

@group(0) @binding(0) var face: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var weight: texture_3d<f32>;
@group(0) @binding(2) var goal: texture_3d<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var solid: texture_3d<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = params.axis;
    if (any(gid >= grid_dims(axis))) {
        return;
    }
    let p = vec3<i32>(gid);
    // Walls, and faces touching a collider (spec §3.2), are left alone.
    if (is_wall(axis, gid[axis]) || face_solid(axis, p)) {
        return;
    }
    let w = face_weight(weight, axis, p, params.dims);
    if (w <= 0.0) {
        return;
    }
    let u = textureLoad(face, p).x;
    let next = u + (textureLoad(goal, p, 0).x - u) * (1.0 - exp(-w * params.h));
    textureStore(face, p, vec4<f32>(next, 0.0, 0.0, 0.0));
}
