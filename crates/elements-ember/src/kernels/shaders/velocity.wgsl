// Velocity at a cell-unit position. For kernels that declare `vel_x`,
// `vel_y` and `vel_z` as `texture_3d<f32>`.
fn velocity_at(x: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        trilinear(vel_x, x - face_offset(0u)),
        trilinear(vel_y, x - face_offset(1u)),
        trilinear(vel_z, x - face_offset(2u)),
    );
}
