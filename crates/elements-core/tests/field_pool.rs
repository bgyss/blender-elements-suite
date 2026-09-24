use elements_core::gpu::{
    FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache, fill_constant,
};

#[test]
fn dims_report_voxel_count() {
    let dims = FieldDims::new(4, 5, 6);
    assert_eq!(dims.voxel_count(), 120);
}

#[test]
fn formats_report_their_size() {
    assert_eq!(FieldFormat::R32Float.channels(), 1);
    assert_eq!(FieldFormat::R32Float.bytes_per_voxel(), 4);
    assert_eq!(FieldFormat::Rgba16Float.channels(), 4);
    assert_eq!(FieldFormat::Rgba16Float.bytes_per_voxel(), 8);
}

#[test]
fn pool_recycles_identical_fields() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let dims = FieldDims::new(8, 8, 8);

    let first = pool.acquire(&ctx, dims, FieldFormat::R32Float).unwrap();
    let first_id = first.pool_generation();
    pool.release(first);
    assert_eq!(pool.pooled_count(), 1);

    let second = pool.acquire(&ctx, dims, FieldFormat::R32Float).unwrap();
    assert_eq!(
        second.pool_generation(),
        first_id,
        "should reuse the texture"
    );
    assert_eq!(pool.pooled_count(), 0);
}

#[test]
fn pool_does_not_recycle_across_shapes() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();

    let a = pool
        .acquire(&ctx, FieldDims::new(8, 8, 8), FieldFormat::R32Float)
        .unwrap();
    let a_id = a.pool_generation();
    pool.release(a);

    let b = pool
        .acquire(&ctx, FieldDims::new(16, 8, 8), FieldFormat::R32Float)
        .unwrap();
    assert_ne!(b.pool_generation(), a_id);
    assert_eq!(pool.pooled_count(), 1, "the 8^3 field stays pooled");
}

#[test]
fn fresh_field_reads_back_as_zeros() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let dims = FieldDims::new(4, 4, 4);

    let field = pool.acquire(&ctx, dims, FieldFormat::R32Float).unwrap();
    let values = field.read_back(&ctx).unwrap();

    assert_eq!(values.len(), dims.voxel_count());
    assert!(values.iter().all(|v| *v == 0.0), "a new texture is zeroed");
}

/// `acquire_zeroed` must clear a recycled texture, not just a fresh one.
/// Fresh allocations are already zero, so this test dirties a texture, returns
/// it to the pool, and asserts that the zeroed acquire got the SAME texture back.
/// Otherwise it would prove nothing.
#[test]
fn acquire_zeroed_clears_a_dirty_recycled_field() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let dims = FieldDims::new(4, 4, 4);

    let dirty = pool.acquire(&ctx, dims, FieldFormat::R32Float).unwrap();
    fill_constant(&ctx, &mut cache, &dirty, 5.0).unwrap();
    let generation = dirty.pool_generation();
    pool.release(dirty);

    let field = pool.acquire_zeroed(&ctx, &mut cache, dims).unwrap();
    assert_eq!(
        field.pool_generation(),
        generation,
        "must reuse the dirty texture, or this test proves nothing"
    );
    let values = field.read_back(&ctx).unwrap();
    assert!(values.iter().all(|&v| v == 0.0), "got {values:?}");
}

#[test]
fn clearing_the_pool_drops_every_free_texture() {
    let gpu = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let dims = FieldDims::new(4, 4, 4);
    let a = pool.acquire(&gpu, dims, FieldFormat::R32Float).unwrap();
    let b = pool.acquire(&gpu, dims, FieldFormat::R32Float).unwrap();
    pool.release(a);
    pool.release(b);
    assert_eq!(pool.pooled_count(), 2);
    pool.clear();
    assert_eq!(pool.pooled_count(), 0);
    // The next acquire is a fresh allocation, not a reused texture.
    let before = pool.allocation_count();
    let _c = pool.acquire(&gpu, dims, FieldFormat::R32Float).unwrap();
    assert_eq!(pool.allocation_count(), before + 1);
}

/// Reuse allocates nothing; a new shape adds its bytes.
#[test]
fn allocated_bytes_counts_only_fresh_textures() {
    let gpu = GpuContext::new_headless().unwrap();
    let mut pool = FieldPool::new();
    assert_eq!(pool.allocated_bytes(), 0);
    let a = pool
        .acquire(&gpu, FieldDims::new(4, 4, 4), FieldFormat::R32Float)
        .unwrap();
    assert_eq!(pool.allocated_bytes(), 4 * 4 * 4 * 4);
    pool.release(a);
    let b = pool
        .acquire(&gpu, FieldDims::new(4, 4, 4), FieldFormat::R32Float)
        .unwrap();
    assert_eq!(pool.allocated_bytes(), 4 * 4 * 4 * 4, "reuse is free");
    let _c = pool
        .acquire(&gpu, FieldDims::new(8, 4, 4), FieldFormat::R32Float)
        .unwrap();
    assert_eq!(pool.allocated_bytes(), 4 * 4 * 4 * 4 + 8 * 4 * 4 * 4);
    pool.release(b);
}
