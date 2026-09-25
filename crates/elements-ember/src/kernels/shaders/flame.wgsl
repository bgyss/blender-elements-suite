// The flame output: sqrt(clamp(react, 0, 1)), 0 where react ≤ 0 (Mantaflow's
// updateFlame). React is clamped to [0, 1] before the sqrt because the
// global mass correction can push react slightly above 1; react itself is
// stored unclamped.

@group(0) @binding(0) var react: texture_3d<f32>;
@group(0) @binding(1) var flame: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let r = textureLoad(react, p, 0).x;
    textureStore(flame, p, vec4<f32>(sqrt(clamp(r, 0.0, 1.0)), 0.0, 0.0, 0.0));
}
