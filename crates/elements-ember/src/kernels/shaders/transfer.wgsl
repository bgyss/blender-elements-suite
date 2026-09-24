// Multigrid transfers between a fine level and the next coarser one, for
// kernels that declare `fine_level` and `coarse_level` (LevelParams) and
// `params`. Prolongation P is geometric: along each axis a fine cell's
// value is interpolated linearly between the two coarse points around its
// centroid. The points are coarse centroids (a partial last cell's centroid
// sits inside its fine cells), and beyond an open face the point where
// p = 0, 0.5 beyond the face. Beyond a wall the last coarse value holds.
// Restriction is κ·Pᵀ, gathered (restrict.wgsl), so the V-cycle is
// symmetric.

// A cell's centroid along `axis`, in fine cells, on a level of `n` cells of
// size `s` over a fine domain of `n0` cells.
fn centroid(n: u32, s: f32, n0: f32, i: i32) -> f32 {
    if (i == i32(n) - 1) {
        return (f32(i) * s + n0) * 0.5;
    }
    return (f32(i) + 0.5) * s;
}

// One axis of P for fine index j: coarse indices i0, i1 with weights w0, w1.
// An index of −1 or the coarse size is the open-face point, holding 0.
struct Axis1 {
    i0: i32,
    i1: i32,
    w0: f32,
    w1: f32,
};

fn prolong_axis(axis: u32, j: i32) -> Axis1 {
    let nf = fine_level.dims[axis];
    let nc = coarse_level.dims[axis];
    if (nf == nc) {
        // Not halved on this level: the coarse cell is the fine cell.
        return Axis1(j, j, 1.0, 0.0);
    }
    let n0 = fine_level.n0[axis];
    let sc = coarse_level.scale[axis];
    let x = centroid(nf, fine_level.scale[axis], n0, j);
    let last = i32(nc) - 1;
    var i = clamp(i32(floor(x / sc - 0.5)), -1, last);
    if (i < last && x >= centroid(nc, sc, n0, i + 1)) {
        i = i + 1;
    }
    if (i < 0) {
        if (!is_open(axis, 0u)) {
            return Axis1(0, 0, 1.0, 0.0);
        }
        let hi = centroid(nc, sc, n0, 0);
        let t = (x + 0.5) / (hi + 0.5);
        return Axis1(-1, 0, 1.0 - t, t);
    }
    if (i == last) {
        if (!is_open(axis, 1u)) {
            return Axis1(i, i, 1.0, 0.0);
        }
        let lo = centroid(nc, sc, n0, i);
        let t = (x - lo) / (n0 + 0.5 - lo);
        return Axis1(i, i + 1, 1.0 - t, t);
    }
    let lo = centroid(nc, sc, n0, i);
    let t = (x - lo) / (centroid(nc, sc, n0, i + 1) - lo);
    return Axis1(i, i + 1, 1.0 - t, t);
}

// Whether coarse index q is the open-face point rather than a cell.
fn beyond(q: vec3<i32>) -> bool {
    return any(q < vec3<i32>(0)) || any(q >= vec3<i32>(coarse_level.dims));
}

fn pick(a: Axis1, bit: u32) -> vec2<f32> {
    // (index, weight) of corner `bit` along one axis.
    return select(vec2<f32>(f32(a.i0), a.w0), vec2<f32>(f32(a.i1), a.w1), bit == 1u);
}
