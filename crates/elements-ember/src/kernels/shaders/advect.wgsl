// Stages 3 and 5: one RK2 semi-Lagrangian pass over one grid, either a
// velocity face (params.axis 0–2) or a cell-centred scalar (params.axis =
// CELL). MacCormack runs `forward`, then `backward`, then the correction in
// `maccormack.wgsl`. Plain semi-Lagrangian runs `semi_lagrangian` alone,
// which applies the scalar's dissipation too.

@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var src: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var solid: texture_3d<f32>;
@group(0) @binding(7) var obstacle: texture_3d<f32>;

fn pass_over(gid: vec3<u32>, direction: f32, decay: f32) {
    let axis = params.axis;
    if (any(gid >= grid_dims(axis))) {
        return;
    }
    let p = vec3<i32>(gid);
    if (axis != CELL) {
        // Walls carry no normal velocity, and faces touching a collider carry
        // its velocity (spec §3.2), before projection as well as after, so the
        // divergence the solve sees matches what projection enforces.
        if (is_wall(axis, gid[axis])) {
            textureStore(dst, p, vec4<f32>(0.0));
            return;
        }
        if (face_solid(axis, p)) {
            textureStore(dst, p, vec4<f32>(textureLoad(obstacle, p, 0).x, 0.0, 0.0, 0.0));
            return;
        }
    }
    // Scalars inside solids are zeroed every substep (spec §3.2), as Mantaflow's resetInObstacle does.
    if (axis == CELL && cell_solid(p)) {
        textureStore(dst, p, vec4<f32>(0.0));
        return;
    }
    let x = vec3<f32>(gid) + grid_offset(axis);
    let value = sample_fluid(src, axis, backtrace(x, direction) - grid_offset(axis));
    textureStore(dst, p, vec4<f32>(value * decay, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn semi_lagrangian(@builtin(global_invocation_id) gid: vec3<u32>) {
    pass_over(gid, 1.0, params.decay);
}

@compute @workgroup_size(4, 4, 4)
fn forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    pass_over(gid, 1.0, 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    pass_over(gid, -1.0, 1.0);
}
