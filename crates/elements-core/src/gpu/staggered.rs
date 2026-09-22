//! Staggered (MAC) vector fields: one `R32Float` texture per velocity component.

use super::{Field, FieldDims, FieldFormat, GpuError};

/// A grid axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

/// A vector field stored on cell faces.
///
/// Component `X` lives on the faces between cells along x, so its texture has
/// `nx + 1` texels along x and matches the domain on y and z. The same holds for
/// `Y` and `Z`. Texel `(i, j, k)` of the X face sits at position
/// `(i, j + 0.5, k + 0.5)` in cell units, where cell `(0, 0, 0)` spans `[0, 1)³`.
///
/// Three `R32Float` textures rather than one `Rgba32Float`: the WebGPU baseline
/// allows `read_write` storage access only on `R32Float`, and pressure
/// projection on a staggered grid is exact where a cell-centred layout
/// checkerboards.
pub struct StaggeredField {
    faces: [Field; 3],
    cells: FieldDims,
}

impl std::fmt::Debug for StaggeredField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaggeredField")
            .field("cells", &self.cells)
            .finish()
    }
}

impl StaggeredField {
    /// The texture dims of `axis`'s face for a domain of `cells`.
    pub fn face_dims(cells: FieldDims, axis: Axis) -> FieldDims {
        match axis {
            Axis::X => FieldDims::new(cells.x + 1, cells.y, cells.z),
            Axis::Y => FieldDims::new(cells.x, cells.y + 1, cells.z),
            Axis::Z => FieldDims::new(cells.x, cells.y, cells.z + 1),
        }
    }

    /// Assemble a field from its X, Y and Z faces, in that order.
    ///
    /// Rejects any face whose dims or format are wrong. A mis-shaped face
    /// would make every sampling kernel read the wrong texels with no error.
    pub fn from_faces(cells: FieldDims, faces: [Field; 3]) -> Result<Self, GpuError> {
        for axis in Axis::ALL {
            let face = &faces[axis.index()];
            let expected = Self::face_dims(cells, axis);
            if face.dims() != expected || face.format() != FieldFormat::R32Float {
                return Err(GpuError::Validation(format!(
                    "staggered {axis:?} face is {:?} {:?}, expected {expected:?} R32Float",
                    face.dims(),
                    face.format()
                )));
            }
        }
        Ok(Self { faces, cells })
    }

    /// The domain's cell dims.
    pub fn cells(&self) -> FieldDims {
        self.cells
    }

    pub fn face(&self, axis: Axis) -> &Field {
        &self.faces[axis.index()]
    }

    /// The X, Y and Z faces, in that order.
    pub fn into_faces(self) -> [Field; 3] {
        self.faces
    }
}
