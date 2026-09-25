// A coarse cell's fluid fraction φ: the volume-weighted mean of its
// children's, each child weighted by the fine cells it covers inside the
// domain. On level 0 a child's φ is 1 − its solid mask. Run on the coarse
// level once per hierarchy, only with solids.

@group(0) @binding(0) var fine_phi: texture_3d<f32>;
@group(0) @binding(1) var out: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<uniform> fine_level: LevelParams;
@group(0) @binding(4) var<uniform> coarse_level: LevelParams;
@group(0) @binding(5) var<uniform> from_mask: vec4<u32>; // x: 1 when fine_phi is a solid mask

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    let n = vec3<i32>(fine_level.dims);
    let halved = coarse_level.dims < fine_level.dims;
    var sum = 0.0;
    var volume = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let o = vec3<i32>(i32(i & 1u), i32((i >> 1u) & 1u), i32((i >> 2u) & 1u));
        if (any(!halved & (o == vec3<i32>(1)))) {
            continue;
        }
        let q = select(c, 2 * c + o, halved);
        if (any(q >= n)) {
            continue;
        }
        // The child's extent in fine cells: its last cell along an axis is partial.
        var v = 1.0;
        for (var a = 0u; a < 3u; a = a + 1u) {
            if (q[a] == n[a] - 1) {
                v = v * fine_level.frac[a];
            }
        }
        var f = textureLoad(fine_phi, q, 0).x;
        if (from_mask.x == 1u) {
            f = 1.0 - f;
        }
        sum += v * f;
        volume += v;
    }
    textureStore(out, c, vec4<f32>(sum / volume, 0.0, 0.0, 0.0));
}
