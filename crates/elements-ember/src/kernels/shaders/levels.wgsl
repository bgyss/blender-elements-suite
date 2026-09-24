// The other grid's dims for the multigrid transfer kernels, which run on
// one level and read the next: the fine dims for a restriction, the coarse
// dims for a prolongation.

struct OtherDims {
    dims: vec3<u32>,
    ghost: f32, // the coarse ghost value beyond an open face, as a fraction of its neighbour
};
