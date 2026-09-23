// Shared by every Ember kernel except the sphere emitter. Concatenated in
// front of each kernel's own source. Each kernel declares its own
// `params: Params` binding; module-scope order does not matter in WGSL.

struct Params {
    dims: vec3<u32>,     // the domain in cells
    axis: u32,           // 0, 1, 2: a face grid along x, y, z; CELL: cell centres
    h: f32,              // substep length, seconds
    inv_dx: f32,         // 1 / voxel edge, 1/m
    dx2: f32,            // voxel edge squared, m²
    pressure_scale: f32, // h: the solve is ∇²p = div / h, and the subtract is u -= h·∇p
    alpha: f32,          // buoyancy per unit density (sinks), m/s²
    beta: f32,           // buoyancy per unit temperature (rises), m/s²
    open_mask: u32,      // bit 2·axis + side is set when that domain face is open
    decay: f32,          // exp(−rate·h) for the scalar a pass carries; 1 otherwise
    confinement: f32,    // ε·dx, the vorticity confinement strength scaled to this grid
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// `axis` for a cell-centred grid.
const CELL: u32 = 3u;

// Whether the domain face on `side` (0 low, 1 high) of `axis` is open.
fn is_open(axis: u32, side: u32) -> bool {
    return ((params.open_mask >> (2u * axis + side)) & 1u) == 1u;
}

// Whether face `i` along `axis` is a solid wall: a boundary face that is
// not open. Everything wall-related goes through this one function.
fn is_wall(axis: u32, i: u32) -> bool {
    if (i == 0u) {
        return !is_open(axis, 0u);
    }
    if (i == params.dims[axis]) {
        return !is_open(axis, 1u);
    }
    return false;
}

// From a position in cell units to a grid's own texel index space.
fn grid_offset(axis: u32) -> vec3<f32> {
    var o = vec3<f32>(0.5, 0.5, 0.5);
    if (axis < 3u) {
        o[axis] = 0.0;
    }
    return o;
}

// A grid's texel dims: the domain, plus one along a face grid's own axis.
fn grid_dims(axis: u32) -> vec3<u32> {
    var d = params.dims;
    if (axis < 3u) {
        d[axis] = d[axis] + 1u;
    }
    return d;
}

// One texel of a grid at any integer index. A face grid clamps to its edge.
// A cell grid reads the ambient value 0 beyond an open face (spec §4.2)
// and clamps beyond a wall.
fn texel(tex: texture_3d<f32>, axis: u32, c: vec3<i32>) -> f32 {
    if (axis != CELL) {
        let last = vec3<i32>(textureDimensions(tex, 0)) - vec3<i32>(1);
        return textureLoad(tex, clamp(c, vec3<i32>(0), last), 0).x;
    }
    let n = vec3<i32>(params.dims);
    var q = c;
    for (var a = 0u; a < 3u; a = a + 1u) {
        if (c[a] < 0) {
            if (is_open(a, 0u)) {
                return 0.0;
            }
            q[a] = 0;
        } else if (c[a] >= n[a]) {
            if (is_open(a, 1u)) {
                return 0.0;
            }
            q[a] = n[a] - 1;
        }
    }
    return textureLoad(tex, q, 0).x;
}

// The 8 texels around `p` (in the grid's texel index space) and p's
// fractional position between them. `p` is clamped to one ghost layer.
struct Corners {
    c: array<f32, 8>, // bit 0 of the index steps x, bit 1 y, bit 2 z
    t: vec3<f32>,
};

fn corners(tex: texture_3d<f32>, axis: u32, p: vec3<f32>) -> Corners {
    let q = clamp(p, vec3<f32>(-1.0), vec3<f32>(textureDimensions(tex, 0)));
    let f = floor(q);
    let i0 = vec3<i32>(f);
    var out: Corners;
    out.t = q - f;
    for (var n = 0u; n < 8u; n = n + 1u) {
        let o = vec3<i32>(i32(n & 1u), i32((n >> 1u) & 1u), i32((n >> 2u) & 1u));
        out.c[n] = texel(tex, axis, i0 + o);
    }
    return out;
}

// Trilinear interpolation, done by hand because R32Float is not filterable
// without an optional feature (umbrella E2).
fn sample_grid(tex: texture_3d<f32>, axis: u32, p: vec3<f32>) -> f32 {
    let k = corners(tex, axis, p);
    let c00 = mix(k.c[0], k.c[1], k.t.x);
    let c10 = mix(k.c[2], k.c[3], k.t.x);
    let c01 = mix(k.c[4], k.c[5], k.t.x);
    let c11 = mix(k.c[6], k.c[7], k.t.x);
    return mix(mix(c00, c10, k.t.y), mix(c01, c11, k.t.y), k.t.z);
}
