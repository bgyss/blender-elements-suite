// Multigrid prolongation, run on the fine grid: each fluid fine cell c adds
// the coarse correction sampled trilinearly at coarse position
// (c + 0.5) / 2 − 0.5. A sample beyond an open face is the linear ghost
// `coarse_dims.ghost` times its clamped neighbour, so the correction falls to
// 0 where p = 0 sits; beyond a wall indices just clamp. Solid coarse
// samples are skipped by renormalising the weights over the fluid ones, and
// nothing is added when all eight are solid. Solid fine cells are untouched
// (the smoother holds them at 0).

@group(0) @binding(0) var p: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var coarse: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var solid: texture_3d<f32>;
@group(0) @binding(4) var coarse_solid: texture_3d<f32>;
@group(0) @binding(5) var<uniform> coarse_dims: OtherDims;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    if (cell_solid(c)) {
        return;
    }
    let pos = (vec3<f32>(c) + 0.5) * 0.5 - 0.5;
    let f = floor(pos);
    let t = pos - f;
    let i0 = vec3<i32>(f);
    let last = vec3<i32>(coarse_dims.dims) - vec3<i32>(1);
    var sum = 0.0;
    var weight = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let o = vec3<u32>(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
        let q = clamp(i0 + vec3<i32>(o), vec3<i32>(0), last);
        if (params.has_solids == 1u && textureLoad(coarse_solid, q, 0).x > 0.5) {
            continue;
        }
        let w3 = select(vec3<f32>(1.0) - t, t, o == vec3<u32>(1u));
        let w = w3.x * w3.y * w3.z;
        let raw = i0 + vec3<i32>(o);
        var v = textureLoad(coarse, q, 0).x;
        for (var a = 0u; a < 3u; a = a + 1u) {
            if ((raw[a] < 0 && is_open(a, 0u)) || (raw[a] > last[a] && is_open(a, 1u))) {
                v *= coarse_dims.ghost;
            }
        }
        sum += w * v;
        weight += w;
    }
    if (weight > 0.0) {
        let value = textureLoad(p, c).x + sum / weight;
        textureStore(p, c, vec4<f32>(value, 0.0, 0.0, 0.0));
    }
}
