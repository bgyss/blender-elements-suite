// Vorticity confinement, part 2 (spec §4.4, Fedkiw et al. 2001): on one
// axis's faces, u += h·f, where f = ε·dx·(N × ω) and N = ∇|ω| / |∇|ω||,
// averaged from the cells each face separates. Wall faces are left alone.

@group(0) @binding(0) var face: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var omega_x: texture_3d<f32>;
@group(0) @binding(2) var omega_y: texture_3d<f32>;
@group(0) @binding(3) var omega_z: texture_3d<f32>;
@group(0) @binding(4) var omega_mag: texture_3d<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn magnitude(c: vec3<i32>) -> f32 {
    let q = clamp(c, vec3<i32>(0), vec3<i32>(params.dims) - vec3<i32>(1));
    return textureLoad(omega_mag, q, 0).x;
}

// The confinement force at the centre of cell `c`, which is inside the domain.
fn force(c: vec3<i32>) -> vec3<f32> {
    let g = 0.5 * params.inv_dx * vec3<f32>(
        magnitude(c + vec3<i32>(1, 0, 0)) - magnitude(c - vec3<i32>(1, 0, 0)),
        magnitude(c + vec3<i32>(0, 1, 0)) - magnitude(c - vec3<i32>(0, 1, 0)),
        magnitude(c + vec3<i32>(0, 0, 1)) - magnitude(c - vec3<i32>(0, 0, 1)),
    );
    let n = g / (length(g) + 1e-6);
    let w = vec3<f32>(
        textureLoad(omega_x, c, 0).x,
        textureLoad(omega_y, c, 0).x,
        textureLoad(omega_z, c, 0).x,
    );
    return params.confinement * cross(n, w);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = params.axis;
    if (any(gid >= grid_dims(axis))) {
        return;
    }
    let i = gid[axis];
    if (is_wall(axis, i)) {
        return;
    }
    let p = vec3<i32>(gid);
    var e = vec3<i32>(0);
    e[axis] = 1;
    // Face i lies between cells i − 1 and i. An open boundary face has only one.
    var f = 0.0;
    var count = 0.0;
    if (i > 0u) {
        f += force(p - e)[axis];
        count += 1.0;
    }
    if (i < params.dims[axis]) {
        f += force(p)[axis];
        count += 1.0;
    }
    let u = textureLoad(face, p).x + params.h * f / count;
    textureStore(face, p, vec4<f32>(u, 0.0, 0.0, 0.0));
}
