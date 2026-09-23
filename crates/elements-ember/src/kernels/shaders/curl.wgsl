// Vorticity confinement, part 1 (spec §4.4): ω = ∇ × u at cell centres, and
// |ω|. Face velocities are averaged to cell centres on the fly. Differences
// are central, with neighbours clamped into the domain at its edge.

@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var omega_x: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var omega_y: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var omega_z: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var omega_mag: texture_storage_3d<r32float, write>;
@group(0) @binding(7) var<uniform> params: Params;
@group(0) @binding(8) var solid: texture_3d<f32>;

// Velocity at the centre of cell `c`, clamped into the domain.
fn centre_velocity(c: vec3<i32>) -> vec3<f32> {
    let q = clamp(c, vec3<i32>(0), vec3<i32>(params.dims) - vec3<i32>(1));
    return 0.5 * vec3<f32>(
        textureLoad(vel_x, q, 0).x + textureLoad(vel_x, q + vec3<i32>(1, 0, 0), 0).x,
        textureLoad(vel_y, q, 0).x + textureLoad(vel_y, q + vec3<i32>(0, 1, 0), 0).x,
        textureLoad(vel_z, q, 0).x + textureLoad(vel_z, q + vec3<i32>(0, 0, 1), 0).x,
    );
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    // A solid neighbour is replaced by `c` itself, like the domain edge (spec §3.2).
    let ddx = centre_velocity(fluid_neighbour(c, vec3<i32>(1, 0, 0)))
        - centre_velocity(fluid_neighbour(c, vec3<i32>(-1, 0, 0)));
    let ddy = centre_velocity(fluid_neighbour(c, vec3<i32>(0, 1, 0)))
        - centre_velocity(fluid_neighbour(c, vec3<i32>(0, -1, 0)));
    let ddz = centre_velocity(fluid_neighbour(c, vec3<i32>(0, 0, 1)))
        - centre_velocity(fluid_neighbour(c, vec3<i32>(0, 0, -1)));
    let w = 0.5 * params.inv_dx * vec3<f32>(ddy.z - ddz.y, ddz.x - ddx.z, ddx.y - ddy.x);
    textureStore(omega_x, c, vec4<f32>(w.x, 0.0, 0.0, 0.0));
    textureStore(omega_y, c, vec4<f32>(w.y, 0.0, 0.0, 0.0));
    textureStore(omega_z, c, vec4<f32>(w.z, 0.0, 0.0, 0.0));
    textureStore(omega_mag, c, vec4<f32>(length(w), 0.0, 0.0, 0.0));
}
