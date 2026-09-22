mod common;

use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_core::graph::DocError;
use elements_ember::emitter::{KIND, Sphere, fill_sphere};

/// Total emitted per second: the sum of rate × occupancy × voxel volume.
fn total_emitted(resolution: u32) -> (f64, f64) {
    let gpu = common::gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let dims = FieldDims::new(resolution, resolution, resolution);
    let dx = 2.0 / resolution as f32;
    let density = pool.acquire(&gpu, dims, FieldFormat::R32Float).unwrap();
    let temperature = pool.acquire(&gpu, dims, FieldFormat::R32Float).unwrap();
    let sphere = Sphere {
        center: [1.0, 1.0, 0.7],
        radius: 0.3,
        density_rate: 2.0,
        temperature_rate: 5.0,
    };
    fill_sphere(&gpu, &mut cache, &density, &temperature, &sphere, dx).unwrap();
    let volume = (dx as f64).powi(3);
    let sum = |f: &elements_core::gpu::Field| {
        f.read_back(&gpu)
            .unwrap()
            .iter()
            .map(|&v| v as f64)
            .sum::<f64>()
            * volume
    };
    (sum(&density), sum(&temperature))
}

/// The emitted amount must not depend on resolution, or a 128³ preview and a
/// 512³ bake of one scene would differ (spec §2.2, §4.1).
#[test]
fn the_emitted_total_matches_the_sphere_at_any_resolution() {
    let analytic = 4.0 / 3.0 * std::f64::consts::PI * 0.3_f64.powi(3);
    let (d32, t32) = total_emitted(32);
    let (d64, t64) = total_emitted(64);
    for (got, rate, what) in [
        (d32, 2.0, "density 32³"),
        (d64, 2.0, "density 64³"),
        (t32, 5.0, "temperature 32³"),
        (t64, 5.0, "temperature 64³"),
    ] {
        let want = analytic * rate;
        assert!(
            ((got - want) / want).abs() <= 0.05,
            "{what}: emitted {got}, sphere holds {want}"
        );
    }
    assert!(((d32 - d64) / d64).abs() <= 0.05, "32³ {d32} vs 64³ {d64}");
}

fn rejected(params: serde_json::Value) -> bool {
    matches!(
        elements_ember::registry().build(KIND, &params),
        Err(DocError::BadParams { .. })
    )
}

#[test]
fn rejects_bad_sphere_parameters() {
    let ok = serde_json::json!({ "center": [1.0, 1.0, 0.3], "radius": 0.2 });
    assert!(!rejected(ok), "a valid sphere must build");
    assert!(rejected(
        serde_json::json!({ "center": [1.0, 1.0, 0.3], "radius": 0.0 })
    ));
    assert!(rejected(
        serde_json::json!({ "center": [1.0, 1.0, 0.3], "radius": -1.0 })
    ));
    assert!(rejected(
        serde_json::json!({ "center": [1e39, 1.0, 0.3], "radius": 0.2 })
    ));
    assert!(rejected(
        serde_json::json!({ "center": [1.0, 1.0, 0.3], "radius": 0.2, "density_rate": 1e39 })
    ));
    assert!(rejected(
        serde_json::json!({ "center": [1.0, 1.0, 0.3], "radius": 0.2, "colour": 1 })
    ));
    assert!(rejected(serde_json::json!({ "radius": 0.2 })));
}
