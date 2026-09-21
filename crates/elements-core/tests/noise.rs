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
        // Three octaves at amplitudes 0.5/0.25/0.125 bound the sum to
        // [-0.875, 0.875] given hash3's [-1, 1] range and the trilinear
        // weights' partition of unity. 0.876 leaves a small epsilon for
        // float error while still catching a change to the amplitudes or
        // the hash range.
        assert!(
            (-0.876..=0.876).contains(v),
            "noise escaped the analytic bound [-0.875, 0.875]: {v}"
        );
    }
}

#[test]
fn seeds_that_collide_under_xor_produce_different_fields() {
    let a = noise_values(0x0000_0001_0000_0000u64, 4.0);
    let b = noise_values(0x0000_0000_0000_0001u64, 4.0);
    assert_ne!(
        a, b,
        "seeds that collide under plain XOR of their halves must still produce different fields"
    );
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
