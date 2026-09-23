// MacCormack's correction (Selle et al. 2008): q̂ + ½(q − q̃), clamped to
// the range of the 8 texels of q that the forward pass interpolated, then
// the scalar's dissipation. `fwd` is q̂ = A(q) and `bwd` is q̃ = A⁻¹(q̂).

@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var orig: texture_3d<f32>;
@group(0) @binding(4) var fwd: texture_3d<f32>;
@group(0) @binding(5) var bwd: texture_3d<f32>;
@group(0) @binding(6) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(7) var<uniform> params: Params;
@group(0) @binding(8) var solid: texture_3d<f32>;
@group(0) @binding(9) var obstacle: texture_3d<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = params.axis;
    if (any(gid >= grid_dims(axis))) {
        return;
    }
    let p = vec3<i32>(gid);
    if (axis != CELL) {
        // As in advect.wgsl: walls carry 0, faces touching a collider its velocity.
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
    let k = fluid_corners(orig, axis, backtrace(x, 1.0) - grid_offset(axis));
    var lo = k.c[0];
    var hi = k.c[0];
    for (var n = 1u; n < 8u; n = n + 1u) {
        lo = min(lo, k.c[n]);
        hi = max(hi, k.c[n]);
    }
    let corrected = textureLoad(fwd, p, 0).x
        + 0.5 * (textureLoad(orig, p, 0).x - textureLoad(bwd, p, 0).x);
    textureStore(dst, p, vec4<f32>(clamp(corrected, lo, hi) * params.decay, 0.0, 0.0, 0.0));
}
