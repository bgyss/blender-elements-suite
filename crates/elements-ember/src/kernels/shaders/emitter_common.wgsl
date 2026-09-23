// Shared by the emitter's cell and face passes (2b-2 spec §2.2). 80 bytes;
// mirrors `EmitterGpu` in shape_emitter.rs.

struct Emitter {
    dims: vec3<u32>,
    dx: f32,
    velocity: vec3<f32>,   // target velocity, m/s, world space
    axis: u32,             // the face pass's axis (0, 1, 2)
    density_rate: f32,
    temperature_rate: f32,
    velocity_blend: f32,   // 1/s
    noise_on: u32,
    noise_scale: f32,      // metres
    noise_amplitude: f32,
    noise_w: f32,          // evolution × seconds: noise's fourth coordinate
    seed_lo: u32,
    seed_hi: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
