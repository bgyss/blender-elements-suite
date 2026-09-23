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

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = params.axis;
    if (any(gid >= grid_dims(axis))) {
        return;
    }
    let p = vec3<i32>(gid);
    if (axis != CELL && is_wall(axis, gid[axis])) {
        textureStore(dst, p, vec4<f32>(0.0));
        return;
    }
    let x = vec3<f32>(gid) + grid_offset(axis);
    let k = corners(orig, axis, backtrace(x, 1.0) - grid_offset(axis));
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
