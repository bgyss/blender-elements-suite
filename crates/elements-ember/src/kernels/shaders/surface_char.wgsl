// The char output (FT4 spec §3.3): the burned fraction of a cell's load,
// 0 where there is no load.

@group(0) @binding(0) var burned: texture_3d<f32>;
@group(0) @binding(1) var load: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let l = textureLoad(load, p, 0).x;
    let c = select(0.0, clamp(textureLoad(burned, p, 0).x / max(l, 1e-30), 0.0, 1.0), l > 0.0);
    textureStore(dst, p, vec4<f32>(c, 0.0, 0.0, 0.0));
}
