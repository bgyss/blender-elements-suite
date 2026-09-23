// Velocity at a cell-unit position. For kernels that declare `vel_x`,
// `vel_y` and `vel_z` as `texture_3d<f32>`.
fn velocity_at(x: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        sample_grid(vel_x, 0u, x - grid_offset(0u)),
        sample_grid(vel_y, 1u, x - grid_offset(1u)),
        sample_grid(vel_z, 2u, x - grid_offset(2u)),
    );
}
