mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache};
use elements_ember::shape_emitter::{EmitterFields, EmitterParams, Noise, fill_emitter};
use elements_ember::transform::{Shape, Transform};

const SPF: f64 = 1.0 / 24.0;

/// A box filling the whole 2 m domain, density rate 1, with `noise`; the
/// density output is then exactly the noise factor in every cell.
fn factors(gpu: &GpuContext, n: u32, noise: Noise, seconds: f64) -> Vec<f32> {
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(n, n, n);
    let dx = 2.0 / n as f32;
    let p = EmitterParams {
        density_rate: 1.0,
        noise: Some(noise),
        ..EmitterParams::new(
            Shape::Box {
                half_extents: [2.0; 3],
            },
            Transform::at([1.0; 3]),
        )
    };
    let f: [_; 3] =
        std::array::from_fn(|_| pool.acquire(gpu, cells, FieldFormat::R32Float).unwrap());
    let v = pool.acquire_staggered_uninit(gpu, cells).unwrap();
    let pose = p.transform.pose(seconds / SPF, SPF);
    fill_emitter(
        gpu,
        &mut cache,
        &p,
        &pose,
        seconds,
        dx,
        EmitterFields {
            density: &f[0],
            temperature: &f[1],
            weight: &f[2],
            velocity: &v,
        },
    )
    .unwrap();
    f[0].read_back(gpu).unwrap()
}

fn noise(seed: u64, evolution: f32) -> Noise {
    Noise {
        seed,
        scale_m: 0.1,
        amplitude: 0.8,
        evolution,
    }
}

/// Spec §2.2: the same seed gives the same pattern, bit for bit, and
/// another seed does not.
#[test]
fn noise_is_deterministic_for_a_seed() {
    let gpu = gpu();
    let a = factors(&gpu, 32, noise(7, 0.0), 0.0);
    let b = factors(&gpu, 32, noise(7, 0.0), 0.0);
    let c = factors(&gpu, 32, noise(8, 0.0), 0.0);
    assert!(
        a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
        "same seed"
    );
    assert!(a != c, "another seed");
}

/// Value noise averages 0.5, so the factor 1 − amplitude·(1 − n) averages
/// 1 − amplitude/2, and it stays within [1 − amplitude, 1].
#[test]
fn the_mean_factor_is_one_minus_half_the_amplitude() {
    let gpu = gpu();
    let f = factors(&gpu, 32, noise(3, 0.0), 0.0);
    let mean = f.iter().map(|&v| f64::from(v)).sum::<f64>() / f.len() as f64;
    assert!((mean - 0.6).abs() <= 0.03, "mean {mean}");
    assert!(
        f.iter().all(|&v| (0.2 - 1e-6..=1.0 + 1e-6).contains(&v)),
        "range"
    );
}

/// With `evolution` > 0 the pattern changes over time; with 0 it does not.
#[test]
fn noise_evolves_only_when_asked() {
    let gpu = gpu();
    let still = (
        factors(&gpu, 32, noise(5, 0.0), 0.0),
        factors(&gpu, 32, noise(5, 0.0), 1.0),
    );
    assert!(still.0 == still.1, "evolution 0 must not change");
    let moving = (
        factors(&gpu, 32, noise(5, 2.0), 0.0),
        factors(&gpu, 32, noise(5, 2.0), 1.0),
    );
    let changed = moving
        .0
        .iter()
        .zip(&moving.1)
        .filter(|(a, b)| (**a - **b).abs() > 1e-3)
        .count();
    assert!(changed * 2 > moving.0.len(), "only {changed} cells changed");
}

/// Noise is in metres, so the pattern is the same at any resolution: each
/// 32³ cell matches the mean of the eight 64³ cells it covers, within the
/// smoothing that averaging adds.
#[test]
fn the_pattern_does_not_depend_on_resolution() {
    let gpu = gpu();
    let n = Noise {
        seed: 11,
        scale_m: 0.5,
        amplitude: 1.0,
        evolution: 0.0,
    };
    let coarse = factors(&gpu, 32, n, 0.0);
    let fine = factors(&gpu, 64, n, 0.0);
    let (c, f) = (FieldDims::new(32, 32, 32), FieldDims::new(64, 64, 64));
    let mut worst = 0.0f32;
    for k in 0..32 {
        for j in 0..32 {
            for i in 0..32 {
                let mut sum = 0.0;
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            sum += fine[index(f, 2 * i + dx, 2 * j + dy, 2 * k + dz)];
                        }
                    }
                }
                worst = worst.max((sum / 8.0 - coarse[index(c, i, j, k)]).abs());
            }
        }
    }
    assert!(worst <= 0.05, "worst difference {worst}");
}

#[test]
fn bad_noise_is_rejected() {
    let base = serde_json::json!({ "shape": { "sphere": { "radius": 0.2 } }, "transform": { "keys": [{ "frame": 0 }] } });
    let with = |noise: serde_json::Value| {
        let mut o = base.clone();
        o["noise"] = noise;
        elements_ember::registry()
            .build(elements_ember::shape_emitter::KIND, &o)
            .is_err()
    };
    assert!(!with(
        serde_json::json!({ "seed": 1, "scale_m": 0.1, "amplitude": 0.5 })
    ));
    assert!(with(
        serde_json::json!({ "seed": 1, "scale_m": 0.0, "amplitude": 0.5 })
    ));
    assert!(with(
        serde_json::json!({ "seed": 1, "scale_m": 1e-40, "amplitude": 0.5 })
    ));
    assert!(with(
        serde_json::json!({ "seed": 1, "scale_m": 0.1, "amplitude": 0.5, "evolution": 1e38 })
    ));
    assert!(with(
        serde_json::json!({ "seed": 1, "scale_m": 0.1, "amplitude": 1.5 })
    ));
    assert!(
        with(serde_json::json!({ "scale_m": 0.1, "amplitude": 0.5 })),
        "seed is required"
    );
}
