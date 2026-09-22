// Shared by every Ember kernel except the sphere emitter. Concatenated in
// front of each kernel's own source. Each kernel declares its own
// `params: Params` binding; module-scope order does not matter in WGSL.

struct Params {
    dims: vec3<u32>, // the domain in cells
    axis: u32,       // 0, 1, 2 for x, y, z; ignored by kernels without an axis
    h: f32,          // substep length, seconds
    inv_dx: f32,     // 1 / voxel edge, 1/m
    dx2: f32,        // voxel edge squared, m²
    alpha: f32,      // buoyancy per unit density (sinks), m/s²
    beta: f32,       // buoyancy per unit temperature (rises), m/s²
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// Trilinear interpolation of `tex` at `p`, in the texture's own texel index
// space: texel (i, j, k) sits at p = (i, j, k). Positions outside clamp to the
// edge. Done by hand because R32Float is not filterable without an optional
// feature (umbrella E2).
fn trilinear(tex: texture_3d<f32>, p: vec3<f32>) -> f32 {
    let last = vec3<i32>(textureDimensions(tex, 0)) - vec3<i32>(1);
    let q = clamp(p, vec3<f32>(0.0), vec3<f32>(last));
    let i0 = vec3<i32>(floor(q));
    let i1 = min(i0 + vec3<i32>(1), last);
    let t = q - vec3<f32>(i0);
    let c000 = textureLoad(tex, vec3<i32>(i0.x, i0.y, i0.z), 0).x;
    let c100 = textureLoad(tex, vec3<i32>(i1.x, i0.y, i0.z), 0).x;
    let c010 = textureLoad(tex, vec3<i32>(i0.x, i1.y, i0.z), 0).x;
    let c110 = textureLoad(tex, vec3<i32>(i1.x, i1.y, i0.z), 0).x;
    let c001 = textureLoad(tex, vec3<i32>(i0.x, i0.y, i1.z), 0).x;
    let c101 = textureLoad(tex, vec3<i32>(i1.x, i0.y, i1.z), 0).x;
    let c011 = textureLoad(tex, vec3<i32>(i0.x, i1.y, i1.z), 0).x;
    let c111 = textureLoad(tex, vec3<i32>(i1.x, i1.y, i1.z), 0).x;
    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}

// From a position in cell units to a face texture's texel index space.
fn face_offset(axis: u32) -> vec3<f32> {
    var o = vec3<f32>(0.5, 0.5, 0.5);
    o[axis] = 0.0;
    return o;
}

// The texel dims of `axis`'s face texture: one more than the domain along it.
fn face_dims(axis: u32) -> vec3<u32> {
    var d = params.dims;
    d[axis] = d[axis] + 1u;
    return d;
}

// Whether face `i` along `axis` is a solid wall. The top face along z is open.
fn is_wall(axis: u32, i: u32) -> bool {
    let n = params.dims[axis];
    if (axis == 2u && i == n) {
        return false;
    }
    return i == 0u || i == n;
}
