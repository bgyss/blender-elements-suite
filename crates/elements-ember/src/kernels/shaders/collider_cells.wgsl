// ember.collider, cell pass: the signed distance at every cell centre, metres.

@group(0) @binding(0) var sdf: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var<uniform> collider: Collider;
@group(0) @binding(2) var<uniform> shape: Shape;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= collider.dims)) {
        return;
    }
    let x = (vec3<f32>(gid) + vec3<f32>(0.5)) * collider.dx;
    textureStore(sdf, vec3<i32>(gid), vec4<f32>(shape_sdf(x), 0.0, 0.0, 0.0));
}
