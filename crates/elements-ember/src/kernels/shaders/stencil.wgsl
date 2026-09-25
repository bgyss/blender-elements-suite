// Face weights of the pressure stencil on a multigrid level, for kernels
// that declare `var<uniform> level: LevelParams` and `var phi: texture_3d<f32>`
// (a placeholder, never read, when level.use_phi is 0). A face's weight is its
// area over the distance between the two points it connects, scaled so a
// full interior face along the finest-spaced axis weighs 1. The stencil is
// symmetric: both cells of a face compute the same weight.

// The area factor of a face along `axis` at cell c: cells in a partial last
// layer along another axis have smaller faces.
fn face_area(c: vec3<i32>, axis: u32) -> f32 {
    let n = vec3<i32>(params.dims);
    var f = 1.0;
    if (axis != 0u && c.x == n.x - 1) {
        f = f * level.frac.x;
    }
    if (axis != 1u && c.y == n.y - 1) {
        f = f * level.frac.y;
    }
    if (axis != 2u && c.z == n.z - 1) {
        f = f * level.frac.z;
    }
    return f;
}

// The weight of the face between c and its in-grid neighbour q on `side`.
// With solids, a coarse face scales by √(φ_c·φ_q), the cells' fluid
// fractions: a symmetric stand-in for the Galerkin operator, whose row for
// a partly solid cell is about φ times the full stencil, as the restricted
// right-hand side is.
fn inner_weight(c: vec3<i32>, q: vec3<i32>, axis: u32, side: u32) -> f32 {
    let n = i32(params.dims[axis]);
    let last = (side == 1u && c[axis] == n - 2) || (side == 0u && c[axis] == n - 1);
    var w = face_area(c, axis) * select(level.g[axis], level.last_face[axis], last);
    if (level.use_phi != 0.0) {
        w = w * sqrt(textureLoad(phi, c, 0).x * textureLoad(phi, q, 0).x);
    }
    return w;
}

// The weight of the open domain face on `side` of `axis`, beyond which p = 0.
fn open_weight(c: vec3<i32>, axis: u32, side: u32) -> f32 {
    var w = face_area(c, axis) * select(level.open_lo[axis], level.open_hi[axis], side == 1u);
    if (level.use_phi != 0.0) {
        w = w * textureLoad(phi, c, 0).x;
    }
    return w;
}
