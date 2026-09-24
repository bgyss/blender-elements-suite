// Multigrid residual r = div − h·L(p), in the units of `div`, with exactly
// the smoother's neighbour rules (pressure.wgsl): a fluid neighbour counts
// with its value, beyond an open face p = 0 counts open_weight with nothing
// added, and
// beyond a wall or a solid neighbour nothing counts. A coarse level solves
// L e = r / h with the unchanged smoother, `div` := r and its own dx².

@group(0) @binding(0) var p: texture_3d<f32>;
@group(0) @binding(1) var div: texture_3d<f32>;
@group(0) @binding(2) var out: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var solid: texture_3d<f32>;

fn nb(q: vec3<i32>, sum: ptr<function, f32>, count: ptr<function, f32>) {
    if (!cell_solid(q)) {
        *sum += textureLoad(p, q, 0).x;
        *count += 1.0;
    }
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    if (cell_solid(c)) {
        textureStore(out, c, vec4<f32>(0.0));
        return;
    }
    let n = vec3<i32>(params.dims);
    var sum = 0.0;
    var count = 0.0;
    // Same neighbour rules as pressure.wgsl, written out per side for the
    // same naga-Metal reason (a loop over axes ran ~40% slower there).
    if (c.x > 0) { nb(c - vec3<i32>(1, 0, 0), &sum, &count); } else if (is_open(0u, 0u)) { count += params.open_weight; }
    if (c.x < n.x - 1) { nb(c + vec3<i32>(1, 0, 0), &sum, &count); } else if (is_open(0u, 1u)) { count += params.open_weight; }
    if (c.y > 0) { nb(c - vec3<i32>(0, 1, 0), &sum, &count); } else if (is_open(1u, 0u)) { count += params.open_weight; }
    if (c.y < n.y - 1) { nb(c + vec3<i32>(0, 1, 0), &sum, &count); } else if (is_open(1u, 1u)) { count += params.open_weight; }
    if (c.z > 0) { nb(c - vec3<i32>(0, 0, 1), &sum, &count); } else if (is_open(2u, 0u)) { count += params.open_weight; }
    if (c.z < n.z - 1) { nb(c + vec3<i32>(0, 0, 1), &sum, &count); } else if (is_open(2u, 1u)) { count += params.open_weight; }
    let lap = (sum - count * textureLoad(p, c, 0).x) / params.dx2;
    let r = textureLoad(div, c, 0).x - params.pressure_scale * lap;
    textureStore(out, c, vec4<f32>(r, 0.0, 0.0, 0.0));
}
