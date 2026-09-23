//! Which faces of the domain are solid walls and which are open (spec §4.2).

use serde::{Deserialize, Serialize};

/// 2a's boundaries: walls everywhere except the top (`+z`), so a plume
/// leaves the domain instead of piling up at the ceiling.
pub const DEFAULT_OPEN_MASK: u32 = 1 << 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Face {
    /// Zero normal velocity; Neumann pressure.
    Wall,
    /// Pressure 0 beyond it; scalars flowing in are ambient, 0.
    Open,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Boundaries {
    #[serde(rename = "-x")]
    pub neg_x: Face,
    #[serde(rename = "+x")]
    pub pos_x: Face,
    #[serde(rename = "-y")]
    pub neg_y: Face,
    #[serde(rename = "+y")]
    pub pos_y: Face,
    #[serde(rename = "-z")]
    pub neg_z: Face,
    #[serde(rename = "+z")]
    pub pos_z: Face,
}

impl Default for Boundaries {
    fn default() -> Self {
        Self {
            pos_z: Face::Open,
            ..Self::closed()
        }
    }
}

impl Boundaries {
    /// Every face a wall.
    pub fn closed() -> Self {
        Self {
            neg_x: Face::Wall,
            pos_x: Face::Wall,
            neg_y: Face::Wall,
            pos_y: Face::Wall,
            neg_z: Face::Wall,
            pos_z: Face::Wall,
        }
    }

    /// Bit `2·axis + side` is set when that face is open; side 0 is the low
    /// face. This is `Params::open_mask` in `common.wgsl`.
    pub fn open_mask(&self) -> u32 {
        [
            self.neg_x, self.pos_x, self.neg_y, self.pos_y, self.neg_z, self.pos_z,
        ]
        .iter()
        .enumerate()
        .filter(|(_, face)| **face == Face::Open)
        .fold(0, |mask, (bit, _)| mask | 1 << bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mask_has_one_bit_per_open_face_in_axis_then_side_order() {
        assert_eq!(Boundaries::default().open_mask(), DEFAULT_OPEN_MASK);
        assert_eq!(Boundaries::closed().open_mask(), 0);
        let neg_y = Boundaries {
            neg_y: Face::Open,
            ..Boundaries::closed()
        };
        assert_eq!(neg_y.open_mask(), 0b000100);
        let pos_x = Boundaries {
            pos_x: Face::Open,
            ..Boundaries::closed()
        };
        assert_eq!(pos_x.open_mask(), 0b000010);
    }
}
