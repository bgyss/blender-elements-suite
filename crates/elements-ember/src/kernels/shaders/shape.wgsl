// Shapes for emitters and colliders (2b-2 spec §2.1). Concatenated in front
// of a kernel that declares `var<uniform> shape: Shape`. The CPU evaluates the
// keyframes; this only reads the resulting pose.

struct Shape {
    world_to_local: mat3x3<f32>,
    origin: vec3<f32>,   // the object's origin in world space, metres
    kind: u32,           // 0 sphere, 1 box
    extents: vec3<f32>,  // sphere: radius in .x; box: half extents
    _pad0: u32,
    linear: vec3<f32>,   // dT/dt, m/s
    _pad1: u32,
    angular: vec3<f32>,  // ω, rad/s
    _pad2: u32,
};

// Signed distance from world point `x` (metres) to the surface; negative inside.
fn shape_sdf(x: vec3<f32>) -> f32 {
    let p = shape.world_to_local * (x - shape.origin);
    if (shape.kind == 0u) {
        return length(p) - shape.extents.x;
    }
    // Exact box distance: outside, the distance to the nearest point; inside,
    // minus the distance to the nearest face.
    let q = abs(p) - shape.extents;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Velocity of the shape's material at world point `x`, m/s.
fn shape_velocity(x: vec3<f32>) -> vec3<f32> {
    return shape.linear + cross(shape.angular, x - shape.origin);
}
