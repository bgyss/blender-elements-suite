// Shared by the collider's cell and face passes (2b-2 spec §2.3). 32 bytes;
// mirrors `ColliderGpu` in collider.rs.

struct Collider {
    dims: vec3<u32>,
    dx: f32,
    axis: u32,   // the face pass's axis (0, 1, 2)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
