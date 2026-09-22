// Stage 2: Boussinesq buoyancy on the z faces, w += h * (beta * T - alpha * rho),
// with density and temperature averaged from the two cells each face separates.
// Dispatched over the z-face dims.

@group(0) @binding(0) var vel_z: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var density: texture_3d<f32>;
@group(0) @binding(2) var temperature: texture_3d<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= face_dims(2u))) {
        return;
    }
    // The floor is a solid wall, and the open top's value comes from projection.
    if (gid.z == 0u || gid.z == params.dims.z) {
        return;
    }
    let above = vec3<i32>(gid);
    let below = above - vec3<i32>(0, 0, 1);
    let rho = 0.5 * (textureLoad(density, below, 0).x + textureLoad(density, above, 0).x);
    let temp = 0.5 * (textureLoad(temperature, below, 0).x + textureLoad(temperature, above, 0).x);
    let w = textureLoad(vel_z, above).x + params.h * (params.beta * temp - params.alpha * rho);
    textureStore(vel_z, above, vec4<f32>(w, 0.0, 0.0, 0.0));
}
