// Stage 4b: red-black Gauss–Seidel for ∇²φ = div, where φ = h·p.
// Cells of one colour never read each other, so each sweep is free of races
// and bit-deterministic.

@group(0) @binding(0) var phi: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var div: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn relax(gid: vec3<u32>, colour: u32) {
    if (any(gid >= params.dims) || ((gid.x + gid.y + gid.z) & 1u) != colour) {
        return;
    }
    let c = vec3<i32>(gid);
    let n = vec3<i32>(params.dims);
    var sum = 0.0;
    var count = 0.0;
    // Solid walls are Neumann: a neighbour outside the domain is left out.
    if (c.x > 0) { sum += textureLoad(phi, c - vec3<i32>(1, 0, 0)).x; count += 1.0; }
    if (c.x < n.x - 1) { sum += textureLoad(phi, c + vec3<i32>(1, 0, 0)).x; count += 1.0; }
    if (c.y > 0) { sum += textureLoad(phi, c - vec3<i32>(0, 1, 0)).x; count += 1.0; }
    if (c.y < n.y - 1) { sum += textureLoad(phi, c + vec3<i32>(0, 1, 0)).x; count += 1.0; }
    if (c.z > 0) { sum += textureLoad(phi, c - vec3<i32>(0, 0, 1)).x; count += 1.0; }
    // Above: an interior neighbour, or the open top, where φ = 0. Either way it counts.
    if (c.z < n.z - 1) { sum += textureLoad(phi, c + vec3<i32>(0, 0, 1)).x; }
    count += 1.0;
    let value = (sum - params.dx2 * textureLoad(div, c, 0).x) / count;
    textureStore(phi, c, vec4<f32>(value, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn red(@builtin(global_invocation_id) gid: vec3<u32>) {
    relax(gid, 0u);
}

@compute @workgroup_size(4, 4, 4)
fn black(@builtin(global_invocation_id) gid: vec3<u32>) {
    relax(gid, 1u);
}
