//! Recycles 3D textures so per-frame evaluation does not reallocate.

use std::collections::HashMap;

use super::{
    Axis, Field, FieldDims, FieldFormat, GpuContext, GpuError, PipelineCache, StaggeredField,
    fill_constant,
};

/// Pools fields keyed by `(dims, format)`.
#[derive(Default)]
pub struct FieldPool {
    free: HashMap<(FieldDims, FieldFormat), Vec<Field>>,
    /// Counts fresh GPU allocations. Stamped onto `Field::generation` so tests
    /// can prove a reused field is the same texture, not a lookalike fresh one.
    next_generation: u64,
}

impl FieldPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many fresh GPU textures this pool has ever allocated. It never
    /// decreases, so a stable count across frames proves reuse.
    pub fn allocation_count(&self) -> u64 {
        self.next_generation
    }

    /// Take a field of this shape from the pool, allocating only if none is free.
    ///
    /// The returned field's contents are unspecified unless it was freshly
    /// allocated, in which case it is zeroed. Callers must fully write a field
    /// before reading it.
    pub fn acquire(
        &mut self,
        ctx: &GpuContext,
        dims: FieldDims,
        format: FieldFormat,
    ) -> Result<Field, GpuError> {
        if let Some(bucket) = self.free.get_mut(&(dims, format))
            && let Some(field) = bucket.pop()
        {
            return Ok(field);
        }

        let generation = self.next_generation;
        self.next_generation += 1;

        ctx.scoped(|| {
            let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("elements-field"),
                size: dims.extent(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: format.wgpu_format(),
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            Field {
                texture,
                view,
                dims,
                format,
                generation,
            }
        })
    }

    /// Take an `R32Float` field of these dims and fill it with zeros.
    ///
    /// Use this whenever a caller may write only part of a field, such as an
    /// emitter, or a state slot on its first step. The zeroing is a compute pass
    /// (`constant.wgsl` at 0), because `CommandEncoder::clear_texture` needs
    /// the optional `Features::CLEAR_TEXTURE`, and the engine enables no optional features.
    pub fn acquire_zeroed(
        &mut self,
        ctx: &GpuContext,
        cache: &mut PipelineCache,
        dims: FieldDims,
    ) -> Result<Field, GpuError> {
        let field = self.acquire(ctx, dims, FieldFormat::R32Float)?;
        fill_constant(ctx, cache, &field, 0.0)?;
        Ok(field)
    }

    /// Take a staggered field for a domain of `cells`, contents unspecified.
    pub fn acquire_staggered_uninit(
        &mut self,
        ctx: &GpuContext,
        cells: FieldDims,
    ) -> Result<StaggeredField, GpuError> {
        let x = self.acquire(
            ctx,
            StaggeredField::face_dims(cells, Axis::X),
            FieldFormat::R32Float,
        )?;
        let y = self.acquire(
            ctx,
            StaggeredField::face_dims(cells, Axis::Y),
            FieldFormat::R32Float,
        )?;
        let z = self.acquire(
            ctx,
            StaggeredField::face_dims(cells, Axis::Z),
            FieldFormat::R32Float,
        )?;
        StaggeredField::from_faces(cells, [x, y, z])
    }

    /// Take a staggered field for a domain of `cells` with every face zeroed.
    pub fn acquire_staggered_zeroed(
        &mut self,
        ctx: &GpuContext,
        cache: &mut PipelineCache,
        cells: FieldDims,
    ) -> Result<StaggeredField, GpuError> {
        let x = self.acquire_zeroed(ctx, cache, StaggeredField::face_dims(cells, Axis::X))?;
        let y = self.acquire_zeroed(ctx, cache, StaggeredField::face_dims(cells, Axis::Y))?;
        let z = self.acquire_zeroed(ctx, cache, StaggeredField::face_dims(cells, Axis::Z))?;
        StaggeredField::from_faces(cells, [x, y, z])
    }

    /// Return all three faces for reuse.
    pub fn release_staggered(&mut self, field: StaggeredField) {
        for face in field.into_faces() {
            self.release(face);
        }
    }

    /// Return a field for reuse.
    pub fn release(&mut self, field: Field) {
        self.free
            .entry((field.dims, field.format))
            .or_default()
            .push(field);
    }

    /// Drop every pooled texture, giving its GPU memory back.
    ///
    /// The daemon calls this after running out of GPU memory: a pool that
    /// kept its free textures would hold on to the memory a retry needs.
    pub fn clear(&mut self) {
        self.free.clear();
    }

    /// How many fields are currently held for reuse.
    pub fn pooled_count(&self) -> usize {
        self.free.values().map(Vec::len).sum()
    }

    /// A pooled field with the same shape as `src`, holding a GPU copy of its contents.
    pub fn duplicate(&mut self, ctx: &GpuContext, src: &Field) -> Result<Field, GpuError> {
        let dst = self.acquire(ctx, src.dims(), src.format())?;
        src.copy_to(ctx, &dst)?;
        Ok(dst)
    }
}
