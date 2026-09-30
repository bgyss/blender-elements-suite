//! A procedural plank shack for the flamethrower shot (roadmap spec §4, FT0):
//! four walls of horizontal planks with gaps, some planks removed, and a roof
//! slab. Our own geometry; it does not copy any reference asset.

use crate::mesh::Mesh;

/// Shape of the procedural shack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShackParams {
    /// Footprint x, footprint y, wall height, metres.
    pub size: [f32; 3],
    pub plank_height: f32,
    pub thickness: f32,
    /// Vertical gap between plank rows, metres.
    pub gap: f32,
    /// Fraction of wall planks removed, 0 to 1.
    pub broken_fraction: f32,
    pub seed: u64,
}

/// SplitMix64: a small explicit-seed generator, so breakage is reproducible.
fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build the shack standing on z = 0, footprint centred on the origin in x/y.
pub fn shack(p: &ShackParams) -> Mesh {
    let [w, d, h] = p.size;
    let (hw, hd, t) = (w / 2.0, d / 2.0, p.thickness);
    let rows = (h / (p.plank_height + p.gap)).floor() as usize;
    let mut rng = p.seed;
    let mut mesh = Mesh {
        positions: vec![],
        indices: vec![],
    };
    // The roof always remains, so the mesh is never empty.
    mesh.merge(&Mesh::box_mesh(
        [-hw - t, -hd - t, h],
        [hw + t, hd + t, h + t],
    ));
    for row in 0..rows {
        let z0 = row as f32 * (p.plank_height + p.gap);
        let z1 = z0 + p.plank_height;
        // South, north, west, east walls; the front/back walls span the full
        // width and the side walls sit between them so planks do not overlap.
        let walls = [
            ([-hw, -hd, z0], [hw, -hd + t, z1]),
            ([-hw, hd - t, z0], [hw, hd, z1]),
            ([-hw, -hd + t, z0], [-hw + t, hd - t, z1]),
            ([hw - t, -hd + t, z0], [hw, hd - t, z1]),
        ];
        for (lo, hi) in walls {
            let roll = (next(&mut rng) >> 11) as f64 / (1u64 << 53) as f64;
            if roll >= f64::from(p.broken_fraction) {
                mesh.merge(&Mesh::box_mesh(lo, hi));
            }
        }
    }
    mesh
}
