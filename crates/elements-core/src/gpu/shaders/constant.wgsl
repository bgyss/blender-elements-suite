// Fill every voxel of a storage texture with a constant.

struct Params {
    dims: vec3<u32>,
    value: f32,
};

@group(0) @binding(0) var field: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.dims.x || gid.y >= params.dims.y || gid.z >= params.dims.z) {
        return;
    }
    textureStore(field, vec3<i32>(gid), vec4<f32>(params.value, 0.0, 0.0, 0.0));
}
