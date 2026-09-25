// A coarse cell is solid (1) only when every one of its children inside the
// fine grid is solid, so a thin fluid gap never closes on a coarse level
// (spec §3.2). Run on the coarse grid. An axis the level did not halve has
// one child per cell along it.

@group(0) @binding(0) var fine_solid: texture_3d<f32>;
@group(0) @binding(1) var out: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<uniform> fine_dims: OtherDims;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    let n = vec3<i32>(fine_dims.dims);
    let halved = params.dims < fine_dims.dims;
    var all_solid = true;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let o = vec3<i32>(i32(i & 1u), i32((i >> 1u) & 1u), i32((i >> 2u) & 1u));
        if (any(!halved & (o == vec3<i32>(1)))) {
            continue;
        }
        let q = select(c, 2 * c + o, halved);
        if (any(q >= n)) {
            continue;
        }
        if (textureLoad(fine_solid, q, 0).x <= 0.5) {
            all_solid = false;
        }
    }
    textureStore(out, c, vec4<f32>(select(0.0, 1.0, all_solid), 0.0, 0.0, 0.0));
}
