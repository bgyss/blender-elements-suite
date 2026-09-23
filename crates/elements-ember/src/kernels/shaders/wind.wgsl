// Wind (2b-2 spec §3.4): u += h·a on one axis's faces, where a is the wind
// along that axis. Wall faces stay as they are.

@group(0) @binding(0) var face: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var solid: texture_3d<f32>;

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
    let u = textureLoad(face, p).x + params.h * params.face_accel;
    textureStore(face, p, vec4<f32>(u, 0.0, 0.0, 0.0));
}
