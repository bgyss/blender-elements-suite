// min(q, 0) per cell, so that a `MaxAbs` reduction afterwards is above 0
// exactly when some cell of q is negative. The mass correction uses it to
// skip fields holding both signs (2b-3c spec §5).

@group(0) @binding(0) var field: texture_3d<f32>;
@group(0) @binding(1) var negative: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    textureStore(negative, c, vec4<f32>(min(textureLoad(field, c, 0).x, 0.0), 0.0, 0.0, 0.0));
}
