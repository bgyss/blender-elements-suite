// Subtract a field's mean, given its sum from a reduction recorded earlier
// in the same batch. Used on p when every domain face is a wall (spec §4.2).

@group(0) @binding(0) var field: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var<storage, read> sum: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let cells = f32(params.dims.x * params.dims.y * params.dims.z);
    let p = vec3<i32>(gid);
    let value = textureLoad(field, p).x - sum[0] / cells;
    textureStore(field, p, vec4<f32>(value, 0.0, 0.0, 0.0));
}
