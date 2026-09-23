use elements_core::gpu::{
    ComputeBatch, FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache, ReduceOp,
    ReduceTarget, reduce,
};

fn gpu() -> GpuContext {
    GpuContext::new_headless().expect("no GPU adapter available")
}

/// Irregular values in about [-1, 1], with the largest magnitude placed in
/// the very last voxel so a reduction that drops the last partial workgroup
/// gets the max wrong, not just the sum.
fn values(dims: FieldDims) -> Vec<f32> {
    let n = dims.voxel_count();
    let mut out: Vec<f32> = (0..n)
        .map(|i| ((i * 73 + 29) % 211) as f32 / 105.0 - 1.0)
        .collect();
    out[n - 1] = -5.0;
    out
}

fn run(gpu: &GpuContext, dims: FieldDims, data: &[f32]) -> [f32; 2] {
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let field = pool.acquire(gpu, dims, FieldFormat::R32Float).unwrap();
    field.write(gpu, data).unwrap();
    let target = ReduceTarget::new(gpu, 3).unwrap();
    let mut batch = ComputeBatch::new();
    reduce(
        gpu,
        &mut cache,
        &mut batch,
        &field,
        ReduceOp::MaxAbs,
        &target,
        0,
    )
    .unwrap();
    reduce(
        gpu,
        &mut cache,
        &mut batch,
        &field,
        ReduceOp::Sum,
        &target,
        2,
    )
    .unwrap();
    batch.submit(gpu).unwrap();
    let got = target.read(gpu).unwrap();
    assert_eq!(got[1], 0.0, "slot 1 was never written and must stay zero");
    [got[0], got[2]]
}

#[test]
fn max_abs_and_sum_match_the_cpu() {
    let gpu = gpu();
    // 105 voxels: one workgroup. 9240 voxels: three, the last one partial.
    for dims in [FieldDims::new(7, 5, 3), FieldDims::new(40, 33, 7)] {
        let data = values(dims);
        let [max, sum] = run(&gpu, dims, &data);
        let want_max = data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let want_sum: f64 = data.iter().map(|&v| f64::from(v)).sum();
        assert_eq!(max, want_max, "{dims:?} max");
        assert!(
            (f64::from(sum) - want_sum).abs() <= 1e-3 * want_sum.abs().max(1.0),
            "{dims:?} sum {sum} vs {want_sum}"
        );
    }
}

#[test]
fn a_reduction_is_bit_identical_run_to_run() {
    let gpu = gpu();
    let dims = FieldDims::new(40, 33, 7);
    let data = values(dims);
    let first = run(&gpu, dims, &data);
    for _ in 0..3 {
        let again = run(&gpu, dims, &data);
        assert_eq!(first.map(f32::to_bits), again.map(f32::to_bits));
    }
}
