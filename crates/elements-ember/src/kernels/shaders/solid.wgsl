// Collider solids (2b-2 spec §3.2), for kernels that declare
// `var solid: texture_3d<f32>`: the frame's cell mask, 1 inside a collider, or
// a 1×1×1 placeholder that is never read because params.has_solids is 0.

fn cell_solid(c: vec3<i32>) -> bool {
    if (params.has_solids == 0u) {
        return false;
    }
    if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(params.dims))) {
        return false;
    }
    return textureLoad(solid, c, 0).x > 0.5;
}

// Whether face `p` of `axis`'s grid touches a solid cell on either side.
fn face_solid(axis: u32, p: vec3<i32>) -> bool {
    if (params.has_solids == 0u) {
        return false;
    }
    var e = vec3<i32>(0);
    e[axis] = 1;
    return cell_solid(p - e) || cell_solid(p);
}

// Neighbour `c + d` for a difference stencil, or `c` itself when that
// neighbour is outside the domain or solid, so solids act like the domain edge.
fn fluid_neighbour(c: vec3<i32>, d: vec3<i32>) -> vec3<i32> {
    let n = c + d;
    if (any(n < vec3<i32>(0)) || any(n >= vec3<i32>(params.dims)) || cell_solid(n)) {
        return c;
    }
    return n;
}

// `corners()` for a cell-centred scalar near solids: each solid corner takes
// the mean of the fluid corners (0 when all eight are solid), so a solid's
// contents never reach the fluid through interpolation. Face grids and frames
// without solids are unchanged.
fn fluid_corners(tex: texture_3d<f32>, axis: u32, p: vec3<f32>) -> Corners {
    var k = corners(tex, axis, p);
    if (params.has_solids == 0u || axis != CELL) {
        return k;
    }
    let i0 = vec3<i32>(floor(clamp(p, vec3<f32>(-1.0), vec3<f32>(textureDimensions(tex, 0)))));
    var sum = 0.0;
    var count = 0.0;
    var solid_bits = 0u;
    for (var n = 0u; n < 8u; n = n + 1u) {
        let o = vec3<i32>(i32(n & 1u), i32((n >> 1u) & 1u), i32((n >> 2u) & 1u));
        if (cell_solid(i0 + o)) {
            solid_bits = solid_bits | (1u << n);
        } else {
            sum += k.c[n];
            count += 1.0;
        }
    }
    if (solid_bits == 0u) {
        return k;
    }
    let fill = select(0.0, sum / max(count, 1.0), count > 0.0);
    for (var n = 0u; n < 8u; n = n + 1u) {
        if (((solid_bits >> n) & 1u) == 1u) {
            k.c[n] = fill;
        }
    }
    return k;
}

// `sample_grid()` over `fluid_corners()`.
fn sample_fluid(tex: texture_3d<f32>, axis: u32, p: vec3<f32>) -> f32 {
    let k = fluid_corners(tex, axis, p);
    let c00 = mix(k.c[0], k.c[1], k.t.x);
    let c10 = mix(k.c[2], k.c[3], k.t.x);
    let c01 = mix(k.c[4], k.c[5], k.t.x);
    let c11 = mix(k.c[6], k.c[7], k.t.x);
    return mix(mix(c00, c10, k.t.y), mix(c01, c11, k.t.y), k.t.z);
}
