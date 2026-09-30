// ember.mesh_collider, cell pass: the signed distance from every cell centre
// to a triangle mesh, metres, negative inside. Brute force: every cell tests
// every triangle, so cost is cells × triangles. Concatenated after
// `shape.wgsl`, which supplies `Shape` (used here only for the pose).

struct MeshParams {
    dims: vec3<u32>,
    dx: f32,
    tri_count: u32,
    offset: f32,     // subtracted from the distance: inflates the solid
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var sdf: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var<uniform> mesh: MeshParams;
@group(0) @binding(2) var<uniform> shape: Shape;
@group(0) @binding(3) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> tris: array<u32>;

// Ericson, Real-Time Collision Detection §5.1.5: the closest point on
// triangle abc to p.
fn closest_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return a; }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return b; }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { return a + ab * (d1 / (d1 - d3)); }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return c; }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { return a + ac * (d2 / (d2 - d6)); }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= mesh.dims)) {
        return;
    }
    let world = (vec3<f32>(gid) + vec3<f32>(0.5)) * mesh.dx;
    let p = shape.world_to_local * (world - shape.origin);

    var best = 1.0e30;
    var winding = 0.0;
    for (var t = 0u; t < mesh.tri_count; t = t + 1u) {
        let a = vertices[tris[3u * t]].xyz;
        let b = vertices[tris[3u * t + 1u]].xyz;
        let c = vertices[tris[3u * t + 2u]].xyz;
        let ab = b - a;
        let ac = c - a;
        let n = cross(ab, ac);
        // A zero-area or sliver triangle has no surface; its normal is rounding
        // noise, so the threshold is relative to the edge lengths.
        if (dot(n, n) <= 1.0e-12 * dot(ab, ab) * dot(ac, ac)) {
            continue;
        }
        best = min(best, length(p - closest_on_triangle(p, a, b, c)));
        // Van Oosterom-Strakhov solid angle of the triangle seen from p. The
        // sum over a closed surface is 4 pi inside and 0 outside, however
        // the parts of the mesh touch or overlap.
        let ra = a - p;
        let rb = b - p;
        let rc = c - p;
        let la = length(ra);
        let lb = length(rb);
        let lc = length(rc);
        let den = la * lb * lc + dot(ra, rb) * lc + dot(rb, rc) * la + dot(rc, ra) * lb;
        // p on a vertex gives a zero length: the distance is 0 there, so the
        // sign is immaterial, but the angle must not be NaN.
        if (la > 0.0 && lb > 0.0 && lc > 0.0) {
            winding = winding + 2.0 * atan2(dot(ra, cross(rb, rc)), den);
        }
    }
    let sign = select(1.0, -1.0, winding / (4.0 * 3.14159265358979) > 0.5);
    textureStore(sdf, vec3<i32>(gid), vec4<f32>(sign * best - mesh.offset, 0.0, 0.0, 0.0));
}
