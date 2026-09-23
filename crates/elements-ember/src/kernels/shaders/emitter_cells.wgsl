// ember.emitter, cell pass: density and temperature rates, and the velocity
// weight, occupancy × velocity_blend in 1/s (2b-2 spec §2.2).

@group(0) @binding(0) var density: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var temperature: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var weight: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<uniform> emitter: Emitter;
@group(0) @binding(4) var<uniform> shape: Shape;

// The emission factor from noise at world point `x`. Task 3 fills this in.
fn noise_factor(x: vec3<f32>) -> f32 {
    return 1.0;
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
