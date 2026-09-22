// A sphere emitter: rate times occupancy, into a density and a temperature source.

struct SphereParams {
    dims: vec3<u32>,
    dx: f32,
    center: vec3<f32>,
    radius: f32,
    density_rate: f32,
    temperature_rate: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var density: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var temperature: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: SphereParams;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    // Cell centre in metres, from the domain's minimum corner.
    let x = (vec3<f32>(gid) + vec3<f32>(0.5)) * params.dx;
    let sd = length(x - params.center) - params.radius;
    // A one-voxel linear ramp centred on the surface, so the occupied volume
    // matches the sphere's to first order at any resolution.
    let occupancy = clamp(0.5 - sd / params.dx, 0.0, 1.0);
    let p = vec3<i32>(gid);
    textureStore(density, p, vec4<f32>(occupancy * params.density_rate, 0.0, 0.0, 0.0));
    textureStore(temperature, p, vec4<f32>(occupancy * params.temperature_rate, 0.0, 0.0, 0.0));
}
