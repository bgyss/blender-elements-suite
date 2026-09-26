// ember.emitter, fuel pass (2b-4 spec §4.1): occupancy × noise × fuel_rate,
// the same pattern as the density rate. Run only when the fuel output is read.

@group(0) @binding(0) var fuel: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var<uniform> emitter: Emitter;
@group(0) @binding(2) var<uniform> shape: Shape;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= emitter.dims)) {
        return;
    }
    let x = (vec3<f32>(gid) + vec3<f32>(0.5)) * emitter.dx;
    let occupancy = clamp(0.5 - shape_sdf(x) / emitter.dx, 0.0, 1.0);
    let rate = occupancy * noise_factor(x);
    textureStore(fuel, vec3<i32>(gid), vec4<f32>(rate * emitter.fuel_rate, 0.0, 0.0, 0.0));
}
