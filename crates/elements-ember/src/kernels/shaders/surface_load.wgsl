// Surface fuel load (FT4 spec §3.1): `load` where the collider's signed
// distance at the cell centre is below zero, 0 elsewhere.

struct Load {
    dims: vec3<u32>,
    load: f32,
};

@group(0) @binding(0) var sdf: texture_3d<f32>;
@group(0) @binding(1) var out_load: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Load;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let l = select(0.0, params.load, textureLoad(sdf, p, 0).x < 0.0);
    textureStore(out_load, p, vec4<f32>(l, 0.0, 0.0, 0.0));
}
