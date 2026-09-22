// Stage 5: semi-Lagrangian advection of a cell-centred scalar.

@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var src: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let x = vec3<f32>(gid) + vec3<f32>(0.5);
    let back = x - velocity_at(x) * (params.h * params.inv_dx);
    let value = trilinear(src, back - vec3<f32>(0.5));
    textureStore(dst, vec3<i32>(gid), vec4<f32>(value, 0.0, 0.0, 0.0));
}
