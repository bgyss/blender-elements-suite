// Stage 4b: red-black Gauss–Seidel for ∇²p = div / h. The state slot holds
// p, not h·p, so the warm start stays valid when h changes (spec §4.3).
// Cells of one colour never read each other, so each sweep is race-free and
// bit-deterministic.

@group(0) @binding(0) var pressure: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var div: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var solid: texture_3d<f32>;

fn relax(gid: vec3<u32>, colour: u32) {
    if (any(gid >= params.dims) || ((gid.x + gid.y + gid.z) & 1u) != colour) {
        return;
    }
    let c = vec3<i32>(gid);
    if (cell_solid(c)) {
        // A solid cell is not solved; fluid cells never read it (spec §3.2).
        textureStore(pressure, c, vec4<f32>(0.0));
        return;
    }
    let n = vec3<i32>(params.dims);
    var sum = 0.0;
    var count = 0.0;
    // A neighbour inside counts with its value. Beyond an open face p = 0
    // (Dirichlet), so it counts with nothing added. Beyond a wall (Neumann)
    // it is left out, and so is a solid neighbour (spec §3.2). Written out per side on purpose: as a loop over axes
    // with dynamic vector indexing, this kernel ran about 40% slower through
    // naga's Metal backend (0.28 against 0.20 ms per iteration at 128³).
    if (c.x > 0) {
        let q = c - vec3<i32>(1, 0, 0);
        if (!cell_solid(q)) {
            sum += textureLoad(pressure, q).x;
            count += 1.0;
        }
    } else if (is_open(0u, 0u)) {
        count += 1.0;
    }
    if (c.x < n.x - 1) {
        let q = c + vec3<i32>(1, 0, 0);
        if (!cell_solid(q)) {
            sum += textureLoad(pressure, q).x;
            count += 1.0;
        }
    } else if (is_open(0u, 1u)) {
        count += 1.0;
    }
    if (c.y > 0) {
        let q = c - vec3<i32>(0, 1, 0);
        if (!cell_solid(q)) {
            sum += textureLoad(pressure, q).x;
            count += 1.0;
        }
    } else if (is_open(1u, 0u)) {
        count += 1.0;
    }
    if (c.y < n.y - 1) {
        let q = c + vec3<i32>(0, 1, 0);
        if (!cell_solid(q)) {
            sum += textureLoad(pressure, q).x;
            count += 1.0;
        }
    } else if (is_open(1u, 1u)) {
        count += 1.0;
    }
    if (c.z > 0) {
        let q = c - vec3<i32>(0, 0, 1);
        if (!cell_solid(q)) {
            sum += textureLoad(pressure, q).x;
            count += 1.0;
        }
    } else if (is_open(2u, 0u)) {
        count += 1.0;
    }
    if (c.z < n.z - 1) {
        let q = c + vec3<i32>(0, 0, 1);
        if (!cell_solid(q)) {
            sum += textureLoad(pressure, q).x;
            count += 1.0;
        }
    } else if (is_open(2u, 1u)) {
        count += 1.0;
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
