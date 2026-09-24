// Multigrid restriction, run on the coarse grid: each coarse cell C takes the
// mean of its fluid children among 2C + {0,1}³ that lie inside the fine grid,
// or 0 when it has none. It also zeroes the coarse correction `e`, the
// starting guess for the coarse solve, saving a dispatch.

@group(0) @binding(0) var fine: texture_3d<f32>;
@group(0) @binding(1) var fine_solid: texture_3d<f32>;
@group(0) @binding(2) var rhs: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var e: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<uniform> fine_dims: OtherDims;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    let n = vec3<i32>(fine_dims.dims);
    var sum = 0.0;
    var count = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let q = 2 * c + vec3<i32>(i32(i & 1u), i32((i >> 1u) & 1u), i32((i >> 2u) & 1u));
        if (any(q >= n)) {
            continue;
        }
        if (params.has_solids == 1u && textureLoad(fine_solid, q, 0).x > 0.5) {
            continue;
        }
        sum += textureLoad(fine, q, 0).x;
        count += 1.0;
    }
    let value = select(0.0, sum / max(count, 1.0), count > 0.0);
    textureStore(rhs, c, vec4<f32>(value, 0.0, 0.0, 0.0));
    textureStore(e, c, vec4<f32>(0.0));
}
