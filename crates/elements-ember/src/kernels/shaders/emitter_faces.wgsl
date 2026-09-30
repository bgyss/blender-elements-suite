// ember.emitter, face pass: the target velocity on one axis's faces, the
// emitter's `velocity` plus its own motion there (2b-2 spec §2.2).

@group(0) @binding(0) var face: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var<uniform> emitter: Emitter;
@group(0) @binding(2) var<uniform> shape: Shape;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = emitter.axis;
    var dims = emitter.dims;
    dims[axis] = dims[axis] + 1u;
    if (any(gid >= dims)) {
        return;
    }
    var offset = vec3<f32>(0.5);
    offset[axis] = 0.0;
    let x = (vec3<f32>(gid) + offset) * emitter.dx;
    // A local-frame velocity turns with the emitter. `world_to_local` is the
    // transpose of local-to-world, so `v * M` is `transpose(M) * v`.
    var target_velocity = emitter.velocity;
    if (emitter.velocity_local != 0u) {
        target_velocity = emitter.velocity * shape.world_to_local;
    }
    let v = target_velocity + shape_velocity(x);
    textureStore(face, vec3<i32>(gid), vec4<f32>(v[axis], 0.0, 0.0, 0.0));
}
