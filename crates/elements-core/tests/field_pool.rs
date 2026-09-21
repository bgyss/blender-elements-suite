use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext};

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
