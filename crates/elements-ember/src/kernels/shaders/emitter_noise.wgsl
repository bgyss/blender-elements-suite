// Noise shared by the emitter's cell, face and fuel passes (2b-2 spec §2.2).
// Each including file declares its own `emitter: Emitter` uniform.

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
