// Hand-written trilinear sampling for R32Float fields and staggered velocity.
//
// A library, not an entry point: kernels prepend it to their own source.
// R32Float is not filterable in the WebGPU baseline (FLOAT32_FILTERABLE is an
// optional feature), so there is no sampler. Every read is a textureLoad.
//
// Positions are in cell units: cell (i, j, k) spans [i, i + 1) on each axis,
// and its centre is (i + 0.5, j + 0.5, k + 0.5). Texel (i, j, k) of the X face
// sits at (i, j + 0.5, k + 0.5), of the Y face at (i + 0.5, j, k + 0.5), and of
// the Z face at (i + 0.5, j + 0.5, k).

const X_FACE_OFFSET: vec3<f32> = vec3<f32>(0.0, 0.5, 0.5);
const Y_FACE_OFFSET: vec3<f32> = vec3<f32>(0.5, 0.0, 0.5);
const Z_FACE_OFFSET: vec3<f32> = vec3<f32>(0.5, 0.5, 0.0);

// Trilinearly interpolate `tex` at texel coordinates `p`, where texel (i, j, k)
// is sampled exactly at p = (i, j, k). Coordinates outside the texture clamp to
// its edge.
fn sample_trilinear(tex: texture_3d<f32>, p: vec3<f32>) -> f32 {
    let last = vec3<i32>(textureDimensions(tex)) - vec3<i32>(1);
    let q = clamp(p, vec3<f32>(0.0), vec3<f32>(last));
    let i0 = vec3<i32>(floor(q));
    let i1 = min(i0 + vec3<i32>(1), last);
    let f = q - vec3<f32>(i0);

    let c000 = textureLoad(tex, vec3<i32>(i0.x, i0.y, i0.z), 0).x;
    let c100 = textureLoad(tex, vec3<i32>(i1.x, i0.y, i0.z), 0).x;
    let c010 = textureLoad(tex, vec3<i32>(i0.x, i1.y, i0.z), 0).x;
    let c110 = textureLoad(tex, vec3<i32>(i1.x, i1.y, i0.z), 0).x;
    let c001 = textureLoad(tex, vec3<i32>(i0.x, i0.y, i1.z), 0).x;
    let c101 = textureLoad(tex, vec3<i32>(i1.x, i0.y, i1.z), 0).x;
    let c011 = textureLoad(tex, vec3<i32>(i0.x, i1.y, i1.z), 0).x;
    let c111 = textureLoad(tex, vec3<i32>(i1.x, i1.y, i1.z), 0).x;

    let c00 = mix(c000, c100, f.x);
    let c10 = mix(c010, c110, f.x);
    let c01 = mix(c001, c101, f.x);
    let c11 = mix(c011, c111, f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

// Velocity at position `x` (cell units) from the X, Y and Z face textures.
fn sample_velocity(u: texture_3d<f32>, v: texture_3d<f32>, w: texture_3d<f32>, x: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        sample_trilinear(u, x - X_FACE_OFFSET),
        sample_trilinear(v, x - Y_FACE_OFFSET),
        sample_trilinear(w, x - Z_FACE_OFFSET)
    );
}

// Velocity at the centre of cell `c`: the mean of the two faces bounding it on
// each axis. Equal to sample_velocity at c + 0.5, but with six loads instead of 24.
fn velocity_at_cell(u: texture_3d<f32>, v: texture_3d<f32>, w: texture_3d<f32>, c: vec3<i32>) -> vec3<f32> {
    return 0.5 * vec3<f32>(
        textureLoad(u, c, 0).x + textureLoad(u, c + vec3<i32>(1, 0, 0), 0).x,
        textureLoad(v, c, 0).x + textureLoad(v, c + vec3<i32>(0, 1, 0), 0).x,
        textureLoad(w, c, 0).x + textureLoad(w, c + vec3<i32>(0, 0, 1), 0).x
    );
}
