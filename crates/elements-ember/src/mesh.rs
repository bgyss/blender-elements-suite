//! Triangle meshes for `ember.mesh_collider` and the procedural shack
//! (flamethrower roadmap spec §4, FT2). Positions are in the object's local
//! space, metres; triangles wind counter-clockwise seen from outside.

use elements_core::graph::DocError;
use serde::{Deserialize, Serialize};

use crate::params;

/// The most triangles a mesh may hold. This bounds memory and buffer size, not
/// frame cost; `mesh_collider::MAX_TRIANGLE_TESTS` bounds that.
pub const MAX_TRIANGLES: usize = 1_000_000;

/// The most positions a mesh may hold: at 16 bytes each (padded to a vec4 on
/// the GPU) this stays under the 128 MiB downlevel storage-binding limit.
pub const MAX_VERTICES: usize = 8_388_608;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    /// Three vertex indices per triangle.
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn validate(&self, kind: &str) -> Result<(), DocError> {
        if self.indices.is_empty() {
            return Err(params::bad(kind, "mesh has no triangles"));
        }
        if !self.indices.len().is_multiple_of(3) {
            return Err(params::bad(
                kind,
                format!(
                    "mesh index count {} is not a multiple of 3",
                    self.indices.len()
                ),
            ));
        }
        if self.triangle_count() > MAX_TRIANGLES {
            return Err(params::bad(
                kind,
                format!(
                    "mesh has {} triangles, at most {MAX_TRIANGLES}",
                    self.triangle_count()
                ),
            ));
        }
        if self.positions.len() > MAX_VERTICES {
            return Err(params::bad(
                kind,
                format!(
                    "mesh has {} positions, at most {MAX_VERTICES}",
                    self.positions.len()
                ),
            ));
        }
        for p in &self.positions {
            params::finite(kind, "mesh position", p)?;
        }
        let n = self.positions.len();
        if let Some(&i) = self.indices.iter().find(|&&i| i as usize >= n) {
            return Err(params::bad(
                kind,
                format!("mesh index {i} is out of range for {n} positions"),
            ));
        }
        Ok(())
    }

    /// Axis-aligned bounds, `(min, max)`. Needs at least one position.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in &self.positions {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        (lo, hi)
    }

    /// A closed, outward-wound box: vertex `i` is `min` or `max` per axis by
    /// the bits of `i` (x = 1, y = 2, z = 4).
    pub fn box_mesh(min: [f32; 3], max: [f32; 3]) -> Mesh {
        let positions = (0..8u32)
            .map(|i| {
                [
                    if i & 1 == 0 { min[0] } else { max[0] },
                    if i & 2 == 0 { min[1] } else { max[1] },
                    if i & 4 == 0 { min[2] } else { max[2] },
                ]
            })
            .collect();
        let indices = vec![
            0, 4, 6, 0, 6, 2, // -x
            1, 3, 7, 1, 7, 5, // +x
            0, 1, 5, 0, 5, 4, // -y
            2, 6, 7, 2, 7, 3, // +y
            0, 2, 3, 0, 3, 1, // -z
            4, 5, 7, 4, 7, 6, // +z
        ];
        Mesh { positions, indices }
    }

    /// Append `other`, offsetting its indices.
    pub fn merge(&mut self, other: &Mesh) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&other.positions);
        self.indices.extend(other.indices.iter().map(|&i| i + base));
    }

    /// Parse Wavefront OBJ text: `v` and `f` lines only. Faces may use
    /// `v/vt/vn` indices and any vertex count (fanned into triangles).
    /// Negative (relative) indices are not supported.
    pub fn from_obj(text: &str) -> Result<Mesh, String> {
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let mut words = line.split_whitespace();
            match words.next() {
                Some("v") => {
                    let v: Vec<f32> = words
                        .take(3)
                        .map(str::parse)
                        .collect::<Result<_, _>>()
                        .map_err(|e| format!("line {}: {e}", n + 1))?;
                    if v.len() != 3 {
                        return Err(format!("line {}: vertex needs 3 coordinates", n + 1));
                    }
                    positions.push([v[0], v[1], v[2]]);
                }
                Some("f") => {
                    let face: Vec<u32> = words
                        .map(|w| {
                            let i: i64 = w
                                .split('/')
                                .next()
                                .unwrap_or("")
                                .parse()
                                .map_err(|e| format!("line {}: {e}", n + 1))?;
                            if i < 1 || i as usize > positions.len() {
                                return Err(format!(
                                    "line {}: index {i} needs an earlier positive vertex",
                                    n + 1
                                ));
                            }
                            Ok((i - 1) as u32)
                        })
                        .collect::<Result<_, String>>()?;
                    if face.len() < 3 {
                        return Err(format!("line {}: face needs 3 vertices", n + 1));
                    }
                    for k in 1..face.len() - 1 {
                        indices.extend_from_slice(&[face[0], face[k], face[k + 1]]);
                    }
                }
                _ => {}
            }
        }
        Ok(Mesh { positions, indices })
    }
}
