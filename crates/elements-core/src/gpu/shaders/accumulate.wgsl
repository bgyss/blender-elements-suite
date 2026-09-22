// sum += input * dt, in place.
//
// `sum` is read_write storage, which the WebGPU baseline allows for r32float.
// `input` is read through textureLoad: R32Float is not filterable without an
// optional feature, and no sampler is needed for a same-texel read.

struct Params {
    dims: vec3<u32>,
    dt: f32,
};

@group(0) @binding(0) var sum: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var input: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.dims.x || gid.y >= params.dims.y || gid.z >= params.dims.z) {
        return;
    }
    let p = vec3<i32>(gid);
    let next = textureLoad(sum, p).x + textureLoad(input, p, 0).x * params.dt;
    textureStore(sum, p, vec4<f32>(next, 0.0, 0.0, 0.0));
}
