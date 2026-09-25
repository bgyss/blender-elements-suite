// Multigrid prolongation, run on the fine level: each fluid fine cell adds
// P·e (transfer.wgsl). Solid coarse cells are skipped, and with solids the
// sum is renormalised by `norm`, 1 / (the weight of the corners kept, the
// open-face points included), computed once per hierarchy by
// prolong_norm.wgsl. Solid fine cells are untouched (the smoother holds
// them at 0).

@group(0) @binding(0) var p: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var coarse: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var solid: texture_3d<f32>;
@group(0) @binding(4) var coarse_solid: texture_3d<f32>;
@group(0) @binding(5) var norm: texture_3d<f32>;
@group(0) @binding(6) var<uniform> fine_level: LevelParams;
@group(0) @binding(7) var<uniform> coarse_level: LevelParams;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    if (cell_solid(c)) {
        return;
    }
    let ax = prolong_axis(0u, c.x);
    let ay = prolong_axis(1u, c.y);
    let az = prolong_axis(2u, c.z);
    var sum = 0.0;
    for (var b = 0u; b < 8u; b = b + 1u) {
        let px = pick(ax, b & 1u);
        let py = pick(ay, (b >> 1u) & 1u);
        let pz = pick(az, (b >> 2u) & 1u);
        let w = px.y * py.y * pz.y;
        let q = vec3<i32>(i32(px.x), i32(py.x), i32(pz.x));
        if (w == 0.0 || beyond(q)) {
            continue;
        }
        if (params.has_solids == 1u && textureLoad(coarse_solid, q, 0).x > 0.5) {
            continue;
        }
        sum += w * textureLoad(coarse, q, 0).x;
    }
    if (params.has_solids == 1u) {
        sum = sum * textureLoad(norm, c, 0).x;
    }
    let value = textureLoad(p, c).x + sum;
    textureStore(p, c, vec4<f32>(value, 0.0, 0.0, 0.0));
}
