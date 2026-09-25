// Velocity at a cell-unit position. For kernels that declare `vel_x`,
// `vel_y` and `vel_z` as `texture_3d<f32>`.
fn velocity_at(x: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        sample_grid(vel_x, 0u, x - grid_offset(0u)),
        sample_grid(vel_y, 1u, x - grid_offset(1u)),
        sample_grid(vel_z, 2u, x - grid_offset(2u)),
    );
}

// Backtrace from `x`, in cell units, over one substep. `direction` 1 traces
// back in time; -1 traces forward, for MacCormack's backward pass. RK2
// (midpoint) by default; one Euler step where the uniform's `euler_faces`
// is set, which is velocity faces while fire burns (2b-4 spec §3.2 step 5):
// at preview's CFL of 5–30 in a fire's core the midpoint lands beside a
// velocity spike and advects it onto itself, and flame vorticity grows it.
fn backtrace(x: vec3<f32>, direction: f32) -> vec3<f32> {
    let k = direction * params.h * params.inv_dx;
    if (params.euler_faces == 1u) {
        return x - k * velocity_at(x);
    }
    let mid = x - 0.5 * k * velocity_at(x);
    return x - k * velocity_at(mid);
}
