// The frame's solid mask (2b-2 spec §3.2): 1 where the collider's signed
// distance at the cell centre is below zero, 0 elsewhere.

@group(0) @binding(0) var sdf: texture_3d<f32>;
@group(0) @binding(1) var mask: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let solid = select(0.0, 1.0, textureLoad(sdf, p, 0).x < 0.0);
    textureStore(mask, p, vec4<f32>(solid, 0.0, 0.0, 0.0));
}
