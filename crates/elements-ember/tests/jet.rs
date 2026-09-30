mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_ember::shape_emitter::{EmitterFields, EmitterParams, Noise, fill_emitter, fill_fuel};
use elements_ember::transform::{Key, Rotate, Shape, Transform};

const SPF: f64 = 1.0 / 24.0;

/// Face velocities [u, v, w] of one nozzle emitter at `frame`, and its blend weight.
fn run(params: &EmitterParams, frame: f64) -> ([Vec<f32>; 3], Vec<f32>) {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(32, 32, 32);
    let f: [_; 3] =
        std::array::from_fn(|_| pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap());
    let v = pool.acquire_staggered_uninit(&gpu, cells).unwrap();
    let pose = params.transform.pose(frame, SPF);
    fill_emitter(
        &gpu,
        &mut cache,
        params,
        &pose,
        frame * SPF,
        2.0 / 32.0,
        EmitterFields {
            density: &f[0],
            temperature: &f[1],
            weight: &f[2],
            velocity: &v,
        },
    )
    .unwrap();
    (read_staggered(&gpu, &v), f[2].read_back(&gpu).unwrap())
}

fn nozzle(rotate: Option<Rotate>, local: bool) -> EmitterParams {
    EmitterParams {
        density_rate: 1.0,
        velocity: [14.0, 0.0, 0.0],
        velocity_blend: 20.0,
        velocity_local: local,
        ..EmitterParams::new(
            Shape::Cone {
                length: 0.6,
                radius_start: 0.1,
                radius_end: 0.2,
            },
            Transform {
                keys: vec![Key {
                    frame: 0.0,
                    translate: [1.0, 1.0, 1.0],
                    rotate,
                }],
            },
        )
    }
}

fn turn_z_90() -> Option<Rotate> {
    Some(Rotate {
        axis: [0.0, 0.0, 1.0],
        degrees: 90.0,
    })
}

// Face (16,16,16) of each array. A u-face (i,j,k) sits at (i*dx, (j+.5)dx,
// (k+.5)dx), so this one is at (1.0, 1.031, 1.031); the v-face is at
// (1.031, 1.0, 1.031). Both are within 0.03 m of the nozzle's axis (the
// turned cone runs along y from 0.7 to 1.3), so inside it.
const UI: usize = 16 + 33 * (16 + 32 * 16);
const VI: usize = 16 + 32 * (16 + 33 * 16);
// Cell (16,16,16) is inside the cone; cell (2,2,2), at 0.16 m, is far outside.
const CELL_IN: usize = 16 + 32 * (16 + 32 * 16);
const CELL_OUT: usize = 2 + 32 * (2 + 32 * 2);

/// A local-frame velocity follows the nozzle: turned 90° about z, the +x jet
/// points along +y (not -y), and no momentum goes along x.
#[test]
fn a_rotated_nozzle_pushes_along_its_own_axis() {
    let ([u, v, w], weight) = run(&nozzle(turn_z_90(), true), 0.0);
    assert!(
        weight[CELL_IN] > 1.0,
        "inside the cone, weight {}",
        weight[CELL_IN]
    );
    assert_eq!(weight[CELL_OUT], 0.0, "outside the cone");
    assert!((v[VI] - 14.0).abs() < 0.1, "v {}", v[VI]);
    assert!(u[UI].abs() < 1e-4, "u {}", u[UI]);
    assert!(w.iter().all(|x| x.abs() < 1e-4), "no z momentum");
}

/// Without `velocity_local`, the velocity stays in world space even when the
/// nozzle turns, which is the behaviour every existing document relies on.
#[test]
fn a_world_frame_velocity_ignores_the_rotation() {
    let ([u, v, _], weight) = run(&nozzle(turn_z_90(), false), 0.0);
    assert!(
        weight[CELL_IN] > 1.0,
        "inside the cone, weight {}",
        weight[CELL_IN]
    );
    assert_eq!(weight[CELL_OUT], 0.0, "outside the cone");
    assert!((u[UI] - 14.0).abs() < 0.1, "u {}", u[UI]);
    assert!(v[VI].abs() < 1e-4, "v {}", v[VI]);
}

fn cone() -> Shape {
    Shape::Cone {
        length: 0.6,
        radius_start: 0.1,
        radius_end: 0.2,
    }
}

fn cone_at(rotate: Option<Rotate>) -> Transform {
    Transform {
        keys: vec![Key {
            frame: 0.0,
            translate: [1.0, 1.0, 1.0],
            rotate,
        }],
    }
}

/// Volume of the truncated cone above, m³: pi h (r1² + r1 r2 + r2²) / 3.
fn cone_volume() -> f64 {
    let (h, r1, r2) = (0.6f64, 0.1f64, 0.2f64);
    std::f64::consts::PI * h * (r1 * r1 + r1 * r2 + r2 * r2) / 3.0
}

/// Density of one cone emitter at `n`³ cells in the 2 m domain.
fn cone_density(gpu: &elements_core::gpu::GpuContext, n: u32, params: &EmitterParams) -> Vec<f32> {
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(n, n, n);
    let f: [_; 3] =
        std::array::from_fn(|_| pool.acquire(gpu, cells, FieldFormat::R32Float).unwrap());
    let v = pool.acquire_staggered_uninit(gpu, cells).unwrap();
    let pose = params.transform.pose(0.0, SPF);
    fill_emitter(
        gpu,
        &mut cache,
        params,
        &pose,
        0.0,
        2.0 / n as f32,
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

/// The nozzle cone emits its analytic truncated-cone volume at any resolution
/// and orientation (roadmap spec §4, FT1).
#[test]
fn a_cone_emits_its_volume_at_any_resolution_and_orientation() {
    let gpu = gpu();
    let volume = cone_volume();
    for n in [32u32, 64] {
        let dx = 2.0 / f64::from(n);
        for rotate in [
            None,
            Some(Rotate {
                axis: [1.0, 1.0, 0.0],
                degrees: 30.0,
            }),
        ] {
            let p = EmitterParams {
                density_rate: 1.0,
                ..EmitterParams::new(cone(), cone_at(rotate))
            };
            let d = cone_density(&gpu, n, &p);
            let total = d.iter().map(|&v| f64::from(v)).sum::<f64>() * dx.powi(3);
            assert!(
                (total - volume).abs() <= 0.05 * volume,
                "{n}³ {rotate:?}: {total} vs {volume}"
            );
        }
    }
}

/// `fuel_rate` fills the same cone: the fuel field's total is rate × volume.
#[test]
fn a_cone_emits_fuel_at_its_rate_times_its_volume() {
    let gpu = gpu();
    let n = 32u32;
    let dx = 2.0 / f64::from(n);
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let fuel = pool
        .acquire(&gpu, FieldDims::new(n, n, n), FieldFormat::R32Float)
        .unwrap();
    let p = EmitterParams {
        fuel_rate: 3.0,
        ..EmitterParams::new(cone(), cone_at(None))
    };
    let pose = p.transform.pose(0.0, SPF);
    fill_fuel(&gpu, &mut cache, &p, &pose, 0.0, dx as f32, &fuel).unwrap();
    let total = fuel
        .read_back(&gpu)
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum::<f64>()
        * dx.powi(3);
    let want = 3.0 * cone_volume();
    assert!(
        (total - want).abs() <= 0.05 * want,
        "fuel {total} vs {want}"
    );
}

/// Two identical fills of a noisy cone emitter agree bit for bit.
#[test]
fn a_noisy_cone_fills_deterministically() {
    let gpu = gpu();
    let p = EmitterParams {
        density_rate: 1.0,
        noise: Some(Noise {
            seed: 7,
            scale_m: 0.1,
            amplitude: 0.8,
            evolution: 0.0,
        }),
        ..EmitterParams::new(cone(), cone_at(turn_z_90()))
    };
    let a = cone_density(&gpu, 32, &p);
    let b = cone_density(&gpu, 32, &p);
    assert!(a.iter().any(|&v| v > 0.0), "the fill is not empty");
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    let quiet = EmitterParams { noise: None, ..p };
    let c = cone_density(&gpu, 32, &quiet);
    assert!(
        a.iter().zip(&c).any(|(x, y)| x != y),
        "the noise changes the fill"
    );
}
