// Fuel emission (2b-4 spec §3.2 step 1): fuel += rate·h, and react blends
// towards 1 by the fresh fuel's share of the new total, as Mantaflow's
// inflow does (Ember blends towards 1 rather than towards Mantaflow's
// occupancy-dependent value; spec §3.3). Fuel is clamped to [0, MAX_FUEL]
// first, and react blends by Δ over the clamped fuel.

// Mantaflow's inflow clamps a cell's fuel to [0, 10] (docs/bench/mantaflow-notes.md,
// "Fire"). Without the upper clamp fuel in an emitter's core keeps growing,
// and flame vorticity with it (2b-4 spec §3.2 step 1). Without the lower
// clamp a negative `fuel_rate` can drive fuel negative, which makes the burn
// (spec §3.2 step 2) remove density instead of adding it and reverses
// confinement.
const MAX_FUEL: f32 = 10.0;

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
    let f1 = clamp(textureLoad(fuel, p).x + d, 0.0, MAX_FUEL);
    textureStore(fuel, p, vec4<f32>(f1, 0.0, 0.0, 0.0));
    if (f1 > 1e-6) {
        let r0 = textureLoad(react, p).x;
        let r1 = clamp(r0 + d / f1 * (1.0 - r0), 0.0, 1.0);
        textureStore(react, p, vec4<f32>(r1, 0.0, 0.0, 0.0));
    }
}
