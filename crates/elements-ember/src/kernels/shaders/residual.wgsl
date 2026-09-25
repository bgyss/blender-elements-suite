// Multigrid residual r = div − h·L(p), in the units of `div`, with the
// smoother's neighbour rules: a fluid neighbour counts with its value,
// beyond an open face p = 0 counts with nothing added, and beyond a wall or
// a solid neighbour nothing counts. Faces are weighted by the level's
// stencil (stencil.wgsl); on the fine grid every weight is exactly 1, so
// this is pressure.wgsl's stencil. A coarse level solves L e = r / h with
// its smoother, `div` := r.

@group(0) @binding(0) var p: texture_3d<f32>;
@group(0) @binding(1) var div: texture_3d<f32>;
@group(0) @binding(2) var out: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var solid: texture_3d<f32>;
@group(0) @binding(5) var<uniform> level: LevelParams;
@group(0) @binding(6) var phi: texture_3d<f32>;

fn nb(c: vec3<i32>, q: vec3<i32>, axis: u32, side: u32, sum: ptr<function, f32>, count: ptr<function, f32>) {
    if (!cell_solid(q)) {
        let w = inner_weight(c, q, axis, side);
        *sum += w * textureLoad(p, q, 0).x;
        *count += w;
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
    // Written out per side for the same naga-Metal reason as pressure.wgsl.
    if (c.x > 0) { nb(c, c - vec3<i32>(1, 0, 0), 0u, 0u, &sum, &count); } else if (is_open(0u, 0u)) { count += open_weight(c, 0u, 0u); }
    if (c.x < n.x - 1) { nb(c, c + vec3<i32>(1, 0, 0), 0u, 1u, &sum, &count); } else if (is_open(0u, 1u)) { count += open_weight(c, 0u, 1u); }
    if (c.y > 0) { nb(c, c - vec3<i32>(0, 1, 0), 1u, 0u, &sum, &count); } else if (is_open(1u, 0u)) { count += open_weight(c, 1u, 0u); }
    if (c.y < n.y - 1) { nb(c, c + vec3<i32>(0, 1, 0), 1u, 1u, &sum, &count); } else if (is_open(1u, 1u)) { count += open_weight(c, 1u, 1u); }
    if (c.z > 0) { nb(c, c - vec3<i32>(0, 0, 1), 2u, 0u, &sum, &count); } else if (is_open(2u, 0u)) { count += open_weight(c, 2u, 0u); }
    if (c.z < n.z - 1) { nb(c, c + vec3<i32>(0, 0, 1), 2u, 1u, &sum, &count); } else if (is_open(2u, 1u)) { count += open_weight(c, 2u, 1u); }
    let lap = (sum - count * textureLoad(p, c, 0).x) / params.dx2;
    let r = textureLoad(div, c, 0).x - params.pressure_scale * lap;
    textureStore(out, c, vec4<f32>(r, 0.0, 0.0, 0.0));
}
