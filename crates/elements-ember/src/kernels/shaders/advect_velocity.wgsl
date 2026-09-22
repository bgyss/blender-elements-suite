// Stage 3: semi-Lagrangian advection of one velocity component. Dispatched
// once per axis, over that axis's face dims, writing a fresh face.

@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var<uniform> params: Params;

fn component(axis: u32, p: vec3<f32>) -> f32 {
    switch axis {
        case 0u: { return trilinear(vel_x, p); }
        case 1u: { return trilinear(vel_y, p); }
        default: { return trilinear(vel_z, p); }
    }
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = params.axis;
    if (any(gid >= face_dims(axis))) {
        return;
    }
    let p = vec3<i32>(gid);
    // Solid walls carry no normal velocity, before projection as well as
    // after, so the divergence the solve sees matches what projection enforces.
    if (is_wall(axis, gid[axis])) {
        textureStore(dst, p, vec4<f32>(0.0));
        return;
    }
    let x = vec3<f32>(gid) + face_offset(axis);
    let back = x - velocity_at(x) * (params.h * params.inv_dx);
    textureStore(dst, p, vec4<f32>(component(axis, back - face_offset(axis)), 0.0, 0.0, 0.0));
}
