// Multigrid restriction, run on the coarse level: rhs = κ·Pᵀ r, the exact
// transpose of prolong_add's P gathered at each coarse cell, with
// κ = 1/2 per halved axis so a constant restricts to itself in the
// interior. A fine cell reaches coarse cell C along a halved axis only from
// indices 2C − 2 ..= 2C + 3. Solid coarse cells get 0. It also zeroes the
// coarse correction `e`, the coarse solve's starting guess.

@group(0) @binding(0) var fine: texture_3d<f32>;
@group(0) @binding(1) var fine_solid: texture_3d<f32>;
@group(0) @binding(2) var norm: texture_3d<f32>;
@group(0) @binding(3) var rhs: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var e: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var solid: texture_3d<f32>;
@group(0) @binding(7) var<uniform> fine_level: LevelParams;
@group(0) @binding(8) var<uniform> coarse_level: LevelParams;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    textureStore(e, c, vec4<f32>(0.0));
    if (cell_solid(c)) {
        textureStore(rhs, c, vec4<f32>(0.0));
        return;
    }
    // Along each axis, the fine indices whose P row reaches c, and the weight.
    var js: array<array<i32, 6>, 3>;
    var ws: array<array<f32, 6>, 3>;
    var counts = vec3<u32>(0u);
    var kappa = 1.0;
    for (var a = 0u; a < 3u; a = a + 1u) {
        let nf = i32(fine_level.dims[a]);
        var lo = c[a];
        var hi = c[a];
        if (fine_level.dims[a] != coarse_level.dims[a]) {
            lo = max(2 * c[a] - 2, 0);
            hi = min(2 * c[a] + 3, nf - 1);
            kappa = kappa * 0.5;
        }
        var k = 0u;
        for (var j = lo; j <= hi; j = j + 1) {
            let w1 = prolong_axis(a, j);
            var w = 0.0;
            if (w1.i0 == c[a]) {
                w += w1.w0;
            }
            if (w1.i1 == c[a]) {
                w += w1.w1;
            }
            if (w != 0.0) {
                js[a][k] = j;
                ws[a][k] = w;
                k = k + 1u;
            }
        }
        counts[a] = k;
    }
    var sum = 0.0;
    for (var iz = 0u; iz < counts.z; iz = iz + 1u) {
        for (var iy = 0u; iy < counts.y; iy = iy + 1u) {
            for (var ix = 0u; ix < counts.x; ix = ix + 1u) {
                let f = vec3<i32>(js[0][ix], js[1][iy], js[2][iz]);
                var w = ws[0][ix] * ws[1][iy] * ws[2][iz];
                if (params.has_solids == 1u) {
                    if (textureLoad(fine_solid, f, 0).x > 0.5) {
                        continue;
                    }
                    w = w * textureLoad(norm, f, 0).x;
                }
                sum += w * textureLoad(fine, f, 0).x;
            }
        }
    }
    textureStore(rhs, c, vec4<f32>(kappa * sum, 0.0, 0.0, 0.0));
}
