// Surface gather (FT4 spec §3.3, kernel 2), per fluid cell: each solid
// neighbour gives its emission, split over its own fluid neighbours. The
// sum over h is a fuel rate per second, which `emit_fuel` consumes. Gathering
// on the fluid side needs no atomics and is deterministic.

@group(0) @binding(0) var emitted: texture_3d<f32>;
@group(0) @binding(1) var rate: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var solid: texture_3d<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    var sum = 0.0;
    if (is_fluid(p)) {
        for (var n = 0u; n < 6u; n = n + 1u) {
            let q = neighbour(p, n);
            if (in_domain(q) && cell_solid(q)) {
                let e = textureLoad(emitted, q, 0).x;
                if (e > 0.0) {
                    sum = sum + e / f32(fluid_count(q));
                }
            }
        }
    }
    textureStore(rate, p, vec4<f32>(sum / params.h, 0.0, 0.0, 0.0));
}
