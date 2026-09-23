// Stage 4b: red-black Gauss–Seidel for ∇²p = div / h. The state slot holds
// p, not h·p, so the warm start stays valid when h changes (spec §4.3).
// Cells of one colour never read each other, so each sweep is race-free and
// bit-deterministic.

@group(0) @binding(0) var pressure: texture_storage_3d<r32float, read_write>;
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
    for (var a = 0u; a < 3u; a = a + 1u) {
        var e = vec3<i32>(0);
        e[a] = 1;
        // A neighbour inside counts with its value. Beyond an open face
        // p = 0 (Dirichlet), so it counts with nothing added. Beyond a wall
        // (Neumann) it is left out.
        if (c[a] > 0) {
            sum += textureLoad(pressure, c - e).x;
            count += 1.0;
        } else if (is_open(a, 0u)) {
            count += 1.0;
        }
        if (c[a] < n[a] - 1) {
            sum += textureLoad(pressure, c + e).x;
            count += 1.0;
        } else if (is_open(a, 1u)) {
            count += 1.0;
        }
    }
    // Only a closed 1×1×1 domain has no neighbours, and nothing to solve.
    if (count == 0.0) {
        return;
    }
    let rhs = params.dx2 * textureLoad(div, c, 0).x / params.pressure_scale;
    textureStore(pressure, c, vec4<f32>((sum - rhs) / count, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn red(@builtin(global_invocation_id) gid: vec3<u32>) {
    relax(gid, 0u);
}

@compute @workgroup_size(4, 4, 4)
fn black(@builtin(global_invocation_id) gid: vec3<u32>) {
    relax(gid, 1u);
}
