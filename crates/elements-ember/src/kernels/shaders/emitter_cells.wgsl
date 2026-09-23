// ember.emitter, cell pass: density and temperature rates, and the velocity
// weight, occupancy × velocity_blend in 1/s (2b-2 spec §2.2).

@group(0) @binding(0) var density: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var temperature: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var weight: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<uniform> emitter: Emitter;
@group(0) @binding(4) var<uniform> shape: Shape;

// Integer hash of a 4D lattice point and the 64-bit seed, to [0, 1]. The
// same construction as core's noise: integer operations only, so it is
// deterministic across backends.
fn hash4(p: vec4<i32>) -> f32 {
    var h: u32 = emitter.seed_lo ^ (emitter.seed_hi * 0x27d4eb2du);
    h = h ^ (u32(p.x) * 0x9e3779b9u);
    h = h ^ (u32(p.y) * 0x85ebca6bu);
    h = h ^ (u32(p.z) * 0xc2b2ae35u);
    h = h ^ (u32(p.w) * 0x165667b1u);
    h = h ^ (h >> 15u);
    h = h * 0x2c1b3c6du;
    h = h ^ (h >> 12u);
    h = h * 0x297a2d39u;
    h = h ^ (h >> 15u);
    return f32(h & 0xffffffu) * (1.0 / 16777215.0);
}

// Value noise in 4D: the 16 lattice corners around `q`, blended with a
// smoothstep fade. Averages 0.5. Once per cell per frame, so the loop is fine.
fn value_noise(q: vec4<f32>) -> f32 {
    let base = floor(q);
    let i = vec4<i32>(base);
    let f = q - base;
    let t = f * f * (vec4<f32>(3.0) - 2.0 * f);
    var acc = 0.0;
    for (var n = 0u; n < 16u; n = n + 1u) {
        let o = vec4<u32>(n & 1u, (n >> 1u) & 1u, (n >> 2u) & 1u, (n >> 3u) & 1u);
        let w = select(vec4<f32>(1.0) - t, t, o == vec4<u32>(1u));
        acc += w.x * w.y * w.z * w.w * hash4(i + vec4<i32>(o));
    }
    return acc;
}

// The emission factor at world point `x`: 1 − amplitude · (1 − n), where `n` is
// noise in metre coordinates plus time as a fourth coordinate (spec §2.2).
fn noise_factor(x: vec3<f32>) -> f32 {
    if (emitter.noise_on == 0u) {
        return 1.0;
    }
    let n = value_noise(vec4<f32>(x / emitter.noise_scale, emitter.noise_w));
    return 1.0 - emitter.noise_amplitude * (1.0 - n);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= emitter.dims)) {
        return;
    }
    let x = (vec3<f32>(gid) + vec3<f32>(0.5)) * emitter.dx;
    // A one-voxel linear ramp centred on the surface, as ember.sphere_emitter
    // uses, so the emitted total matches the volume at any resolution.
    let occupancy = clamp(0.5 - shape_sdf(x) / emitter.dx, 0.0, 1.0);
    let rate = occupancy * noise_factor(x);
    let p = vec3<i32>(gid);
    textureStore(density, p, vec4<f32>(rate * emitter.density_rate, 0.0, 0.0, 0.0));
    textureStore(temperature, p, vec4<f32>(rate * emitter.temperature_rate, 0.0, 0.0, 0.0));
    textureStore(weight, p, vec4<f32>(occupancy * emitter.velocity_blend, 0.0, 0.0, 0.0));
}
