// Stage 4c: u -= h·∇p on one axis's faces. Dispatched once per axis over
// that axis's face dims. Face i lies between cells i - 1 and i.

@group(0) @binding(0) var face: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var pressure: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var solid: texture_3d<f32>;
@group(0) @binding(4) var obstacle: texture_3d<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = params.axis;
    if (any(gid >= grid_dims(axis))) {
        return;
    }
    let p = vec3<i32>(gid);
    let i = gid[axis];
    if (is_wall(axis, i)) {
        textureStore(face, p, vec4<f32>(0.0));
        return;
    }
    // A face touching a collider carries its velocity (spec §3.2).
    if (face_solid(axis, p)) {
        textureStore(face, p, vec4<f32>(textureLoad(obstacle, p, 0).x, 0.0, 0.0, 0.0));
        return;
    }
    var e = vec3<i32>(0);
    e[axis] = 1;
    // Beyond an open face, p = 0.
    var upper = 0.0;
    if (i < params.dims[axis]) {
        upper = textureLoad(pressure, p, 0).x;
    }
    var lower = 0.0;
    if (i > 0u) {
        lower = textureLoad(pressure, p - e, 0).x;
    }
    let u = textureLoad(face, p).x - params.pressure_scale * (upper - lower) * params.inv_dx;
    textureStore(face, p, vec4<f32>(u, 0.0, 0.0, 0.0));
}
