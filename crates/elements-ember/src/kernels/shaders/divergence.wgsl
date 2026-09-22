// Stage 4a: divergence of the staggered velocity, per cell, in 1/s.

@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var div: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    let du = textureLoad(vel_x, c + vec3<i32>(1, 0, 0), 0).x - textureLoad(vel_x, c, 0).x;
    let dv = textureLoad(vel_y, c + vec3<i32>(0, 1, 0), 0).x - textureLoad(vel_y, c, 0).x;
    let dw = textureLoad(vel_z, c + vec3<i32>(0, 0, 1), 0).x - textureLoad(vel_z, c, 0).x;
    textureStore(div, c, vec4<f32>((du + dv + dw) * params.inv_dx, 0.0, 0.0, 0.0));
}
