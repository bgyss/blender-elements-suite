use elements_core::gpu::{
    FieldDims, FieldFormat, FieldPool, GpuContext, GpuError, PipelineCache, fill_constant,
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

/// This test detects under-dispatch, not the shader's bounds guard.
///
/// With `dims = 5` and a workgroup size of 4, using plain integer division
/// (`5 / 4 == 1`) instead of `div_ceil` (`5.div_ceil(4) == 2`) would launch a
/// single workgroup per axis, covering only voxels 0..=3 and leaving voxel 4
/// of each axis unwritten. `read_back` would then observe the padding value
/// there instead of `1.0`, which this test's per-voxel assertion catches.
///
/// It does NOT prove the shader's bounds guard is necessary: WGSL defines an
/// out-of-bounds `textureStore` as a discarded no-op, so on a storage
/// *texture* removing the guard leaves this test passing. The guard stays in
/// the shader anyway, because the moment any shader here writes to a storage
/// *buffer* instead, out-of-bounds writes are undefined behaviour rather than
/// a harmless no-op, and the guard also avoids launching pointless
/// invocations for the excess lanes in each dispatched workgroup.
#[test]
fn constant_fill_handles_non_multiple_of_workgroup() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    // 5 is not a multiple of the workgroup size 4: dispatch must round up.
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
fn pipeline_cache_returns_the_same_compiled_pipeline() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut cache = PipelineCache::new();
    let source = include_str!("../src/gpu/shaders/constant.wgsl");

    let first = cache
        .get_or_create(&ctx, "constant", source, "main")
        .unwrap();
    let second = cache
        .get_or_create(&ctx, "constant", source, "main")
        .unwrap();

    // `HashMap::insert` on a repeated key leaves `len()` at 1 whether or not
    // the value was recomputed, so a length check alone cannot tell a cache
    // hit from a silent recompile on every call. Comparing `Arc::ptr_eq`
    // instead asserts the second call returned the SAME compiled pipeline
    // object rather than a fresh one that merely overwrote the map entry.
    // Do not "simplify" this back to a length-only check.
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "get_or_create must return the same cached pipeline, not recompile it"
    );
    assert_eq!(cache.len(), 1);
}

#[test]
fn pipeline_cache_detects_a_key_collision() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut cache = PipelineCache::new();
    let constant_source = include_str!("../src/gpu/shaders/constant.wgsl");
    let noise_source = include_str!("../src/gpu/shaders/curl_noise.wgsl");

    cache
        .get_or_create(&ctx, "shared-key", constant_source, "main")
        .unwrap();
    // Same key, different source: this must be caught, not silently accepted.
    // A real, always-on `Err` (not a `debug_assert_eq!`, which compiles out
    // of the release builds `elementsd` ships) so a cache hit can never
    // silently hand back the wrong compiled pipeline.
    match cache.get_or_create(&ctx, "shared-key", noise_source, "main") {
        Err(GpuError::Validation(message)) => {
            assert!(
                message.contains("shared-key"),
                "error should name the colliding key, got: {message}"
            );
        }
        other => panic!("expected GpuError::Validation, got {other:?}"),
    }
}

#[test]
fn broken_shader_surfaces_as_a_validation_error_not_a_panic() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut cache = PipelineCache::new();

    let result = cache.get_or_create(
        &ctx,
        "broken-shader-for-testing",
        "@compute fn main() { this is not wgsl }",
        "main",
    );

    match result {
        Err(GpuError::Validation(message)) => {
            assert!(
                !message.is_empty(),
                "validation error should carry a non-empty message"
            );
        }
        other => panic!("expected Err(GpuError::Validation(_)), got {other:?}"),
    }
}
