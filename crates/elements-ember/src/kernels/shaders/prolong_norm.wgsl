// With solids, P renormalises each fluid fine cell's weights over the coarse
// corners it keeps: solid corners drop out, open-face points (holding 0)
// stay. This writes 1 / that weight, or 0 for a solid fine cell or one with
// no corner kept, once per hierarchy, so prolong_add and restrict apply the
// identical factor and restriction stays exactly Pᵀ.

@group(0) @binding(0) var out: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var solid: texture_3d<f32>;
@group(0) @binding(3) var coarse_solid: texture_3d<f32>;
@group(0) @binding(4) var<uniform> fine_level: LevelParams;
@group(0) @binding(5) var<uniform> coarse_level: LevelParams;

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
    let ax = prolong_axis(0u, c.x);
    let ay = prolong_axis(1u, c.y);
    let az = prolong_axis(2u, c.z);
    var kept = 0.0;
    for (var b = 0u; b < 8u; b = b + 1u) {
        let px = pick(ax, b & 1u);
        let py = pick(ay, (b >> 1u) & 1u);
        let pz = pick(az, (b >> 2u) & 1u);
        let w = px.y * py.y * pz.y;
        let q = vec3<i32>(i32(px.x), i32(py.x), i32(pz.x));
        if (w == 0.0) {
            continue;
        }
        if (!beyond(q) && textureLoad(coarse_solid, q, 0).x > 0.5) {
            continue;
        }
        kept += w;
    }
    textureStore(out, c, vec4<f32>(select(0.0, 1.0 / kept, kept > 0.0), 0.0, 0.0, 0.0));
}
