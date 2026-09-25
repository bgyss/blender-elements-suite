// Fuel emission (2b-4 spec §3.2 step 1): fuel += rate·h, and react blends
// towards 1 by the fresh fuel's share of the new total, as Mantaflow's
// inflow does (Ember blends towards 1 rather than towards Mantaflow's
// occupancy-dependent value; spec §3.3).

@group(0) @binding(0) var fuel: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var react: texture_storage_3d<r32float, read_write>;
@group(0) @binding(2) var src: texture_3d<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let d = textureLoad(src, p, 0).x * params.h;
    let f1 = textureLoad(fuel, p).x + d;
    textureStore(fuel, p, vec4<f32>(f1, 0.0, 0.0, 0.0));
    if (f1 > 1e-6) {
        let r0 = textureLoad(react, p).x;
        let r1 = clamp(r0 + d / f1 * (1.0 - r0), 0.0, 1.0);
        textureStore(react, p, vec4<f32>(r1, 0.0, 0.0, 0.0));
    }
}
