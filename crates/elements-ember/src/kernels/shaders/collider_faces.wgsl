// ember.collider, face pass: the collider's material velocity at every face
// centre of one axis, m/s. Defined everywhere; the solver uses it only on
// faces that touch a solid cell.

@group(0) @binding(0) var face: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var<uniform> collider: Collider;
@group(0) @binding(2) var<uniform> shape: Shape;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = collider.axis;
    var dims = collider.dims;
    dims[axis] = dims[axis] + 1u;
    if (any(gid >= dims)) {
        return;
    }
    var offset = vec3<f32>(0.5);
    offset[axis] = 0.0;
    let x = (vec3<f32>(gid) + offset) * collider.dx;
    textureStore(face, vec3<i32>(gid), vec4<f32>(shape_velocity(x)[axis], 0.0, 0.0, 0.0));
}
