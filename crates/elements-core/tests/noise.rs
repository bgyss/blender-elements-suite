use elements_core::gpu::{
    FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache, fill_curl_noise,
};

fn noise_values(seed: u64, frequency: f32) -> Vec<f32> {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let field = pool
        .acquire(&ctx, FieldDims::new(16, 16, 16), FieldFormat::R32Float)
        .unwrap();
    fill_curl_noise(&ctx, &mut cache, &field, seed, frequency).unwrap();
    field.read_back(&ctx).unwrap()
}

#[test]
fn noise_is_deterministic_for_a_seed() {
    let a = noise_values(7, 4.0);
    let b = noise_values(7, 4.0);
    assert_eq!(a, b, "the same seed must produce bit-identical output");
}

#[test]
fn different_seeds_produce_different_fields() {
    let a = noise_values(7, 4.0);
    let b = noise_values(8, 4.0);
    assert_ne!(a, b);
}

#[test]
fn noise_stays_in_range_and_is_finite() {
    let values = noise_values(7, 4.0);
    assert_eq!(values.len(), 16 * 16 * 16);
    for v in &values {
        assert!(v.is_finite(), "noise produced a non-finite value: {v}");
        assert!((-1.001..=1.001).contains(v), "noise escaped [-1, 1]: {v}");
    }
}

#[test]
fn noise_is_not_constant() {
    let values = noise_values(7, 4.0);
    let first = values[0];
    assert!(
        values.iter().any(|v| (v - first).abs() > 1e-2),
        "noise must actually vary across the field"
    );
}
