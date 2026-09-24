// One multigrid level's geometry and face weights (`LevelParams` in
// multigrid.rs, 128 bytes). Lengths are in fine cells. A level's last cell
// along an axis may be partial: it covers only the fine cells left before
// the domain's edge, so its centroid, its face areas and its distance to an
// open face's p = 0 all differ from a full cell's. The weights below carry
// that, and are exactly 1 on the fine grid.

struct LevelParams {
    dims: vec3<u32>,      // the level's cells
    slot: u32,            // this level's slot in the fluid-count reduction
    g: vec3<f32>,         // weight of a face along each axis: (s_ref / s)²
    use_phi: f32,         // 1 when faces scale by the cells' fluid fractions `phi` (solids, coarse levels)
    open_lo: vec3<f32>,   // weight of the low open face, to p = 0 at −0.5
    _p1: f32,
    open_hi: vec3<f32>,   // weight of the high open face, to p = 0 at N₀ + 0.5
    _p2: f32,
    last_face: vec3<f32>, // weight of the face between the last two cells
    _p3: f32,
    frac: vec3<f32>,      // the last cell's extent along each axis, as a fraction of s
    _p4: f32,
    scale: vec3<f32>,     // s: fine cells per level cell along each axis
    _p5: f32,
    n0: vec3<f32>,        // the fine domain's size, cells
    _p6: f32,
};
