// The fine grid's dims for restrict_mask.wgsl, which runs on the coarse grid.

struct OtherDims {
    dims: vec3<u32>,
    _pad: u32,
};
