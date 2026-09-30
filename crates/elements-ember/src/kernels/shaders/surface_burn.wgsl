// Surface burn (FT4 spec §3.3, kernel 1), per band cell: ignite from hot
// neighbouring gas, burn `params.surface_burn` of the load, record what was
// emitted. `emitted` is written in every cell, 0 where nothing burned.

@group(0) @binding(0) var load: texture_3d<f32>;
@group(0) @binding(1) var temperature: texture_3d<f32>;
@group(0) @binding(2) var burned: texture_storage_3d<r32float, read_write>;
@group(0) @binding(3) var emitted: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var solid: texture_3d<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    var e = 0.0;
    let wood = textureLoad(load, p, 0).x;
    if (cell_solid(p) && wood > 0.0 && fluid_count(p) > 0u) {
        let b = textureLoad(burned, p).x;
        var hottest = -3.0e38;
        for (var n = 0u; n < 6u; n = n + 1u) {
            let q = neighbour(p, n);
            if (is_fluid(q)) {
                hottest = max(hottest, textureLoad(temperature, q, 0).x);
            }
        }
        if ((b > 0.0 || hottest > params.ignition_temperature) && b < wood) {
            e = min(params.surface_burn, wood - b);
            textureStore(burned, p, vec4<f32>(b + e, 0.0, 0.0, 0.0));
        }
    }
    textureStore(emitted, p, vec4<f32>(e, 0.0, 0.0, 0.0));
}
