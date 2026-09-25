// The burn (2b-4 spec §3.2, the fuel-and-reaction model Blender's Mantaflow
// fire uses), per cell: fuel falls by `burn`, react falls in proportion
// (divided first: `r1 = r0 * (f1 / f0)`) but is 0 where fuel is at or below
// 1e-6 (spec §3.2: react' = 0 where fuel ≤ 1e-6), burning makes smoke, and
// where there is flame the temperature follows its profile from ignition
// (edge) to max (core). See the `select` below for why `f1 == f0` (not
// `params.burn == 0.0`) is the exactness shortcut. The flame here reads
// react clamped to [0, 1]: the global mass correction can push react
// slightly above 1, which would otherwise push the temperature above
// `max_temperature`; react itself is stored unclamped. Reimplemented from
// the published model; four read_write storage textures, the per-stage
// limit.

@group(0) @binding(0) var fuel: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var react: texture_storage_3d<r32float, read_write>;
@group(0) @binding(2) var density: texture_storage_3d<r32float, read_write>;
@group(0) @binding(3) var temperature: texture_storage_3d<r32float, read_write>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let f0 = textureLoad(fuel, p).x;
    let r0 = textureLoad(react, p).x;
    let f1 = max(f0 - params.burn, 0.0);
    // Below 1e-6, a cell holds no fuel and react ends (spec §3.2): react' =
    // 0 there. Above it, react divides by the fuel's own share `f1 / f0`,
    // except when `f1 == f0` exactly (nothing burned this cell, whether
    // because `burn` is 0 or the cell just wasn't touched): then react
    // passes through as `r0` untouched rather than going through the
    // division. `f1 / f0` is mathematically 1 there, but division is not
    // guaranteed to round to exactly 1.0 on every GPU, and that would
    // perturb react by a ULP even though nothing burned.
    var r1 = 0.0;
    if (f0 > 1e-6) {
        r1 = select(r0 * (f1 / f0), r0, f1 == f0);
    }
    let smoke = (0.5 + 0.5 * max(1.0 - f0, 0.0)) * (f0 - f1) * 0.1 * params.flame_smoke;
    textureStore(fuel, p, vec4<f32>(f1, 0.0, 0.0, 0.0));
    textureStore(react, p, vec4<f32>(r1, 0.0, 0.0, 0.0));
    let d = textureLoad(density, p).x;
    textureStore(density, p, vec4<f32>(d + smoke, 0.0, 0.0, 0.0));
    let flame = sqrt(clamp(r1, 0.0, 1.0));
    if (flame > 0.0) {
        let t = (1.0 - flame) * params.ignition_temperature + flame * params.max_temperature;
        textureStore(temperature, p, vec4<f32>(t, 0.0, 0.0, 0.0));
    }
}
