// Wind (2b-3c spec §6): the air relaxes towards an ambient airflow,
// u += (w − u)·blend on one axis's faces, where w is the wind velocity along
// that axis and blend = 1 − exp(−rate·h). The emitter's velocity blend has
// the same form: exact for any h, so it never carries u past w. Wall faces
// stay as they are.

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
    let u0 = textureLoad(face, p).x;
    let u = u0 + (params.face_wind - u0) * params.wind_blend;
    textureStore(face, p, vec4<f32>(u, 0.0, 0.0, 0.0));
}
