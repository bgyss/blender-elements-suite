// The burn (2b-4 spec §3.2, the fuel-and-reaction model Blender's Mantaflow
// fire uses), per cell: fuel falls by `burn`, react falls in proportion
// (divided first: `r1 = r0 * (f1 / f0)`), burning makes smoke, and where
// there is flame the temperature follows its profile from ignition (edge)
// to max (core). With burn 0, react instead passes through untouched
// (see below) so it is unchanged bit-for-bit. The flame here reads react
// clamped to [0, 1]: the global mass correction can push react slightly
// above 1, which would otherwise push the temperature above
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
    // With burn 0 (fire on, not burning), f1 == f0 exactly, so react passes
    // through untouched rather than through the division: `f1 / f0` is not
    // guaranteed to round to exactly 1.0 on every GPU (division here is not
    // required to be correctly rounded), which would perturb react by up to
    // a ULP even though nothing burned. `params.burn` is uniform across the
    // dispatch, so this branch is uniform too.
    var r1 = r0;
    if (params.burn != 0.0) {
        // A hard zero check, not an epsilon: any nonzero f0, however small,
        // divides exactly by itself once burn actually moves it.
        r1 = 0.0;
        if (f0 != 0.0) {
            r1 = r0 * (f1 / f0);
        }
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
