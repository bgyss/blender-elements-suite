//! 3D scalar and vector fields backed by GPU textures.

use super::{GpuContext, GpuError};

/// The extent of a field in voxels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldDims {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl FieldDims {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub fn voxel_count(&self) -> usize {
        self.x as usize * self.y as usize * self.z as usize
    }

    pub(crate) fn extent(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.x,
            height: self.y,
            depth_or_array_layers: self.z,
        }
    }
}

/// Check that a field of `dims` would fit inside `max_buffer_size`, both as a
/// texture (`x * y * z * 4` bytes, `R32Float`) and as `Field::read_back`'s
/// padded staging buffer (rows aligned to `COPY_BYTES_PER_ROW_ALIGNMENT`).
///
/// `dims` comes from an untrusted document, and a document with 3-million-cube
/// dims is exactly the kind of input this exists to reject: `x * y * z * 4`
/// for such a document overflows even `u64` (3,000,000^3 * 4 ~= 1.08e20 versus
/// `u64::MAX` ~= 1.8e19). So every multiplication here is done in `u128`,
/// which cannot overflow for any `u32` dims, and the final comparison against
/// `max_buffer_size` (a `u64`) is done by widening the limit, never by
/// narrowing the computed byte count.
///
/// Returns `Err` with a human-readable reason naming the offending size, or
/// `Ok(())` if both fit.
pub fn validate_dims_fit_buffer_limit(dims: FieldDims, max_buffer_size: u64) -> Result<(), String> {
    let x = dims.x as u128;
    let y = dims.y as u128;
    let z = dims.z as u128;
    let max_buffer_size = max_buffer_size as u128;

    let texture_bytes = x * y * z * 4;
    if texture_bytes > max_buffer_size {
        return Err(format!(
            "dims {:?}: an R32Float texture would need {texture_bytes} bytes, more than this \
             device's max_buffer_size of {max_buffer_size}",
            [dims.x, dims.y, dims.z]
        ));
    }

    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u128;
    let unpadded_row = x * 4;
    let padded_row = unpadded_row.div_ceil(align) * align;
    let readback_bytes = padded_row * y * z;
    if readback_bytes > max_buffer_size {
        return Err(format!(
            "dims {:?}: the padded R32Float readback buffer would need {readback_bytes} bytes, \
             more than this device's max_buffer_size of {max_buffer_size}",
            [dims.x, dims.y, dims.z]
        ));
    }

    Ok(())
}

/// The storage formats Core v1 supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldFormat {
    R32Float,
    Rgba16Float,
}

impl FieldFormat {
    pub fn channels(&self) -> u32 {
        match self {
            Self::R32Float => 1,
            Self::Rgba16Float => 4,
        }
    }

    pub fn bytes_per_voxel(&self) -> u32 {
        match self {
            Self::R32Float => 4,
            Self::Rgba16Float => 8,
        }
    }

    pub(crate) fn wgpu_format(&self) -> wgpu::TextureFormat {
        match self {
            Self::R32Float => wgpu::TextureFormat::R32Float,
            Self::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        }
    }
}

/// A 3D texture plus the metadata needed to interpret it.
pub struct Field {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) dims: FieldDims,
    pub(crate) format: FieldFormat,
    /// Identifies which fresh GPU allocation backs this field.
    ///
    /// `wgpu` 30 does not expose a `global_id()` on `Texture` (unlike some
    /// earlier versions), so `FieldPool` stamps this counter on every fresh
    /// allocation and never changes it on reuse. Tests use it in place of
    /// `texture().global_id()` to assert that a field handed back by
    /// `FieldPool::acquire` is the *same* GPU texture as one released earlier,
    /// not a fresh allocation that merely looks equivalent.
    pub(crate) generation: u64,
}

impl Field {
    pub fn dims(&self) -> FieldDims {
        self.dims
    }

    pub fn format(&self) -> FieldFormat {
        self.format
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The pool allocation generation backing this field's GPU texture.
    ///
    /// Two fields with the same generation share the same underlying texture.
    /// See the field's doc comment for why this exists instead of
    /// `texture().global_id()`.
    pub fn pool_generation(&self) -> u64 {
        self.generation
    }

    /// Copy the field to the CPU as `f32`, x-fastest, `channels()` values per voxel.
    ///
    /// GPU buffer rows must be aligned to `COPY_BYTES_PER_ROW_ALIGNMENT`, so this
    /// copies into a padded staging buffer and strips the padding on the way out.
    pub fn read_back(&self, ctx: &GpuContext) -> Result<Vec<f32>, GpuError> {
        let unpadded_row = self.dims.x * self.format.bytes_per_voxel();
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = unpadded_row.div_ceil(align) * align;
        let buffer_size = padded_row as u64 * self.dims.y as u64 * self.dims.z as u64;

        // The staging buffer's `size` is untrusted-input-derived (it scales with
        // the document's `dims`), so it must be created INSIDE the error scope.
        // `create_buffer` validates `size` against `device.limits().max_buffer_size`
        // and, outside a scope, an over-limit request goes to wgpu's
        // uncaptured-error handler, which panics rather than returning an error.
        let staging = ctx.scoped(|| {
            let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("field-readback"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut encoder =
                ctx.device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("field-readback"),
                    });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(self.dims.y),
                    },
                },
                self.dims.extent(),
            );
            ctx.queue().submit(Some(encoder.finish()));
            staging
        })?;

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::DeviceLost(e.to_string()))?;
        rx.recv()
            .map_err(|e| GpuError::Validation(e.to_string()))?
            .map_err(|e| GpuError::Validation(e.to_string()))?;

        let channels = self.format.channels() as usize;
        let mut out = Vec::with_capacity(self.dims.voxel_count() * channels);
        {
            let mapped = slice
                .get_mapped_range()
                .map_err(|e| GpuError::Validation(e.to_string()))?;
            let row_bytes = self.dims.x as usize * self.format.bytes_per_voxel() as usize;
            for z in 0..self.dims.z as usize {
                for y in 0..self.dims.y as usize {
                    let start = (z * self.dims.y as usize + y) * padded_row as usize;
                    let end = start + row_bytes;
                    let row = &mapped[start..end];
                    match self.format {
                        FieldFormat::R32Float => {
                            let floats: &[f32] = bytemuck::cast_slice(row);
                            out.extend_from_slice(floats);
                        }
                        FieldFormat::Rgba16Float => {
                            let halves: &[half::f16] = bytemuck::cast_slice(row);
                            out.extend(halves.iter().map(|h| h.to_f32()));
                        }
                    }
                }
            }
        }
        staging.unmap();

        Ok(out)
    }

    /// Upload `values` (x-fastest, one `f32` per voxel) into an `R32Float` field.
    ///
    /// `Queue::write_texture` has no row-alignment requirement, unlike
    /// buffer-to-texture copies, so no padding is needed.
    pub fn write(&self, ctx: &GpuContext, values: &[f32]) -> Result<(), GpuError> {
        if self.format != FieldFormat::R32Float || values.len() != self.dims.voxel_count() {
            return Err(GpuError::Validation(format!(
                "write: {} values into a {:?} {:?} field",
                values.len(),
                self.dims,
                self.format
            )));
        }
        ctx.scoped(|| {
            ctx.queue().write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(values),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.dims.x * 4),
                    rows_per_image: Some(self.dims.y),
                },
                self.dims.extent(),
            );
        })
    }

    /// Copy this field's contents into `dst` on the GPU.
    ///
    /// Both fields must have identical dims and format.
    /// `copy_texture_to_texture` is core WebGPU, not an optional feature.
    pub fn copy_to(&self, ctx: &GpuContext, dst: &Field) -> Result<(), GpuError> {
        if self.dims != dst.dims || self.format != dst.format {
            return Err(GpuError::Validation(format!(
                "copy_to: source is {:?} {:?}, destination is {:?} {:?}",
                self.dims, self.format, dst.dims, dst.format
            )));
        }
        ctx.scoped(|| {
            let mut encoder =
                ctx.device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("field-copy"),
                    });
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &dst.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                self.dims.extent(),
            );
            ctx.queue().submit(Some(encoder.finish()));
        })
    }
}
