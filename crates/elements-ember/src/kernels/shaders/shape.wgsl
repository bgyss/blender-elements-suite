// Shapes for emitters and colliders (2b-2 spec §2.1). Concatenated in front
// of a kernel that declares `var<uniform> shape: Shape`. The CPU evaluates the
// keyframes; this only reads the resulting pose.

struct Shape {
    world_to_local: mat3x3<f32>,
    origin: vec3<f32>,   // the object's origin in world space, metres
    kind: u32,           // 0 sphere, 1 box, 2 cone
    extents: vec3<f32>,  // sphere: radius in .x; box: half extents;
                         // cone: half length, radius at -x, radius at +x
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
    if (shape.kind == 2u) {
        // Capped cone (Inigo Quilez's exact formula) with its axis along local x.
        let h = shape.extents.x;
        let r1 = shape.extents.y;   // radius at -x
        let r2 = shape.extents.z;   // radius at +x
        let q = vec2<f32>(length(p.yz), p.x);
        let k1 = vec2<f32>(r2, h);
        let k2 = vec2<f32>(r2 - r1, 2.0 * h);
        let ca = vec2<f32>(q.x - min(q.x, select(r2, r1, q.y < 0.0)), abs(q.y) - h);
        let t = clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
        let cb = q - k1 + k2 * t;
        let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
        return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
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
