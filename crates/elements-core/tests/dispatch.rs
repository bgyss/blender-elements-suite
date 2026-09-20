use elements_core::gpu::{
    FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache, fill_constant,
};

#[test]
fn constant_fill_writes_every_voxel() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let dims = FieldDims::new(8, 8, 8);

    let field = pool.acquire(&ctx, dims, FieldFormat::R32Float).unwrap();
    fill_constant(&ctx, &mut cache, &field, 0.75).unwrap();

    let values = field.read_back(&ctx).unwrap();
    assert_eq!(values.len(), dims.voxel_count());
    for v in &values {
        approx::assert_abs_diff_eq!(*v, 0.75, epsilon = 1e-3);
    }
}

#[test]
fn constant_fill_handles_non_multiple_of_workgroup() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    // 5 is not a multiple of the workgroup size 4: the shader must bounds-check.
    let dims = FieldDims::new(5, 5, 5);

    let field = pool.acquire(&ctx, dims, FieldFormat::R32Float).unwrap();
    fill_constant(&ctx, &mut cache, &field, 1.0).unwrap();

    let values = field.read_back(&ctx).unwrap();
    assert_eq!(values.len(), 125);
    for v in &values {
        approx::assert_abs_diff_eq!(*v, 1.0, epsilon = 1e-3);
    }
}

#[test]
fn pipeline_cache_reuses_compiled_pipelines() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut cache = PipelineCache::new();
    let mut pool = FieldPool::new();
    let field = pool
        .acquire(&ctx, FieldDims::new(4, 4, 4), FieldFormat::R32Float)
        .unwrap();

    fill_constant(&ctx, &mut cache, &field, 0.1).unwrap();
    fill_constant(&ctx, &mut cache, &field, 0.2).unwrap();

    assert_eq!(cache.len(), 1, "the same shader must compile only once");
}
