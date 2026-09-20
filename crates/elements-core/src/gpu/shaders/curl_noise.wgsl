// Seeded value noise, summed over three octaves.
//
// Deterministic across backends: integer hashing only, no trigonometric or
// transcendental functions, whose precision varies between drivers.

struct Params {
    dims: vec3<u32>,
    frequency: f32,
    seed_lo: u32,
    seed_hi: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var field: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var<uniform> params: Params;

fn hash3(p: vec3<i32>, seed: u32) -> f32 {
    var h: u32 = seed;
    h = h ^ (u32(p.x) * 0x9e3779b9u);
    h = h ^ (u32(p.y) * 0x85ebca6bu);
    h = h ^ (u32(p.z) * 0xc2b2ae35u);
    h = h ^ (h >> 15u);
    h = h * 0x2c1b3c6du;
    h = h ^ (h >> 12u);
    h = h * 0x297a2d39u;
    h = h ^ (h >> 15u);
    // Map to [-1, 1] using only exact binary fractions.
    return f32(h & 0xffffffu) * (2.0 / 16777215.0) - 1.0;
}

fn smoothstep3(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

fn value_noise(p: vec3<f32>, seed: u32) -> f32 {
    let cell = vec3<i32>(floor(p));
    let f = p - floor(p);
    let w = vec3<f32>(smoothstep3(f.x), smoothstep3(f.y), smoothstep3(f.z));

    var acc: f32 = 0.0;
    for (var dz: i32 = 0; dz <= 1; dz = dz + 1) {
        for (var dy: i32 = 0; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = 0; dx <= 1; dx = dx + 1) {
                let corner = cell + vec3<i32>(dx, dy, dz);
                let wx = select(1.0 - w.x, w.x, dx == 1);
                let wy = select(1.0 - w.y, w.y, dy == 1);
                let wz = select(1.0 - w.z, w.z, dz == 1);
                acc = acc + hash3(corner, seed) * wx * wy * wz;
            }
        }
    }
    return acc;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.dims.x || gid.y >= params.dims.y || gid.z >= params.dims.z) {
        return;
    }

    let uvw = (vec3<f32>(gid) + vec3<f32>(0.5)) / vec3<f32>(params.dims);
    // Mix with a large odd multiplier rather than plain XOR: XOR of the two
    // halves collapses distinct u64 seeds that share the same halves in
    // swapped positions (e.g. 0x00000001_00000000 and 0x00000000_00000001
    // both XOR to 1), which silently defeats determinism-by-seed.
    let seed = params.seed_lo ^ (params.seed_hi * 0x9e3779b9u);

    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var freq: f32 = params.frequency;
    for (var octave: u32 = 0u; octave < 3u; octave = octave + 1u) {
        value = value + value_noise(uvw * freq, seed + octave * 0x9e3779b9u) * amplitude;
        amplitude = amplitude * 0.5;
        freq = freq * 2.0;
    }

    // Three octaves at 0.5/0.25/0.125 sum to at most 0.875, so this clamp is
    // unreachable by construction with the current amplitudes: hash3 is
    // bounded to [-1, 1], the trilinear weights partition unity, and the
    // per-octave amplitudes bound the sum to [-0.875, 0.875]. It stays as a
    // defensive guard in case a future change to the octave count or
    // amplitudes makes it reachable; it costs nothing at runtime.
    textureStore(field, vec3<i32>(gid), vec4<f32>(clamp(value, -1.0, 1.0), 0.0, 0.0, 0.0));
}
