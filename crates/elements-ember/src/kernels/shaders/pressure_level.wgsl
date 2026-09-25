// Red-black Gauss–Seidel on a coarse multigrid level: pressure.wgsl's sweep,
// with each face weighted by stencil.wgsl. The fine grid runs pressure.wgsl.

@group(0) @binding(0) var pressure: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var div: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var solid: texture_3d<f32>;
@group(0) @binding(4) var<uniform> level: LevelParams;
@group(0) @binding(5) var phi: texture_3d<f32>;

fn nb(c: vec3<i32>, q: vec3<i32>, axis: u32, side: u32, sum: ptr<function, f32>, count: ptr<function, f32>) {
    if (!cell_solid(q)) {
        let w = inner_weight(c, q, axis, side);
        *sum += w * textureLoad(pressure, q).x;
        *count += w;
    }
}

fn relax(gid: vec3<u32>, colour: u32) {
    if (any(gid >= params.dims) || ((gid.x + gid.y + gid.z) & 1u) != colour) {
        return;
    }
    let c = vec3<i32>(gid);
    if (cell_solid(c)) {
        textureStore(pressure, c, vec4<f32>(0.0));
        return;
    }
    let n = vec3<i32>(params.dims);
    var sum = 0.0;
    var count = 0.0;
    if (c.x > 0) { nb(c, c - vec3<i32>(1, 0, 0), 0u, 0u, &sum, &count); } else if (is_open(0u, 0u)) { count += open_weight(c, 0u, 0u); }
    if (c.x < n.x - 1) { nb(c, c + vec3<i32>(1, 0, 0), 0u, 1u, &sum, &count); } else if (is_open(0u, 1u)) { count += open_weight(c, 0u, 1u); }
    if (c.y > 0) { nb(c, c - vec3<i32>(0, 1, 0), 1u, 0u, &sum, &count); } else if (is_open(1u, 0u)) { count += open_weight(c, 1u, 0u); }
    if (c.y < n.y - 1) { nb(c, c + vec3<i32>(0, 1, 0), 1u, 1u, &sum, &count); } else if (is_open(1u, 1u)) { count += open_weight(c, 1u, 1u); }
    if (c.z > 0) { nb(c, c - vec3<i32>(0, 0, 1), 2u, 0u, &sum, &count); } else if (is_open(2u, 0u)) { count += open_weight(c, 2u, 0u); }
    if (c.z < n.z - 1) { nb(c, c + vec3<i32>(0, 0, 1), 2u, 1u, &sum, &count); } else if (is_open(2u, 1u)) { count += open_weight(c, 2u, 1u); }
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
