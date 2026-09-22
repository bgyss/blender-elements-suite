// Stage 1: dst += src * h. Emission: `src` is a rate per second.

@group(0) @binding(0) var dst: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let next = textureLoad(dst, p).x + textureLoad(src, p, 0).x * params.h;
    textureStore(dst, p, vec4<f32>(next, 0.0, 0.0, 0.0));
}
