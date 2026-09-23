mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache};
use elements_ember::collider::{ColliderFields, ColliderParams, fill_collider};
use elements_ember::transform::{Key, Pose, Rotate, Shape, Transform};
use elements_ember::unions::union_colliders;

const SPF: f64 = 1.0 / 24.0;
const CELLS: FieldDims = FieldDims { x: 12, y: 10, z: 8 };
const DX: f32 = 2.0 / 12.0;

fn run(gpu: &GpuContext, params: &ColliderParams, frame: f64) -> (Vec<f32>, [Vec<f32>; 3], Pose) {
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let sdf = pool.acquire(gpu, CELLS, FieldFormat::R32Float).unwrap();
    let v = pool.acquire_staggered_uninit(gpu, CELLS).unwrap();
    let pose = params.transform.pose(frame, SPF);
    fill_collider(
        gpu,
        &mut cache,
        params,
        &pose,
        DX,
        ColliderFields {
            sdf: &sdf,
            velocity: &v,
        },
    )
    .unwrap();
    (sdf.read_back(gpu).unwrap(), read_staggered(gpu, &v), pose)
}

/// CPU box distance, mirroring `shape_sdf` in shape.wgsl.
fn box_sdf(pose: &Pose, half: [f64; 3], x: [f64; 3]) -> f64 {
    let m = pose.world_to_local();
    let d: [f64; 3] = std::array::from_fn(|a| x[a] - pose.translate[a]);
    let p: [f64; 3] = std::array::from_fn(|r| m[r][0] * d[0] + m[r][1] * d[1] + m[r][2] * d[2]);
    let q: [f64; 3] = std::array::from_fn(|a| p[a].abs() - half[a]);
    let outside = q.iter().map(|v| v.max(0.0).powi(2)).sum::<f64>().sqrt();
    outside + q[0].max(q[1]).max(q[2]).min(0.0)
}

fn cell_centre(i: u32, j: u32, k: u32) -> [f64; 3] {
    [i, j, k].map(|c| (f64::from(c) + 0.5) * f64::from(DX))
}

/// Spec §2.3: the SDF output is the exact signed distance, negative inside,
/// for spheres and for rotated boxes.
#[test]
fn the_sdf_is_the_signed_distance_to_the_shape() {
    let gpu = gpu();
    let sphere = ColliderParams {
        shape: Shape::Sphere { radius: 0.3 },
        transform: Transform::at([1.0, 0.9, 0.7]),
    };
    let (s, _, _) = run(&gpu, &sphere, 0.0);
    let half = [0.3, 0.2, 0.25];
    let boxed = ColliderParams {
        shape: Shape::Box {
            half_extents: half.map(|v| v as f32),
        },
        transform: Transform {
            keys: vec![Key {
                frame: 0.0,
                translate: [1.0, 0.8, 0.6],
                rotate: Some(Rotate {
                    axis: [0.0, 1.0, 1.0],
                    degrees: 30.0,
                }),
            }],
        },
    };
    let (b, _, pose) = run(&gpu, &boxed, 0.0);
    let mut inside = 0;
    for k in 0..CELLS.z {
        for j in 0..CELLS.y {
            for i in 0..CELLS.x {
                let x = cell_centre(i, j, k);
                let at = index(CELLS, i, j, k);
                let r = ((x[0] - 1.0).powi(2) + (x[1] - 0.9).powi(2) + (x[2] - 0.7).powi(2)).sqrt();
                assert!(
                    (f64::from(s[at]) - (r - 0.3)).abs() <= 1e-5,
                    "sphere {:?}",
                    [i, j, k]
                );
                let want = box_sdf(&pose, half, x);
                assert!(
                    (f64::from(b[at]) - want).abs() <= 1e-5,
                    "box {:?}: {} vs {want}",
                    [i, j, k],
                    b[at]
                );
                inside += usize::from(want < 0.0);
            }
        }
    }
    assert!(inside > 0, "some cell must be inside the box");
}

/// Spec §2.3: the velocity output is v + ω × r at every face centre.
#[test]
fn the_velocity_is_the_colliders_material_velocity_at_each_face() {
    let gpu = gpu();
    let spinning = ColliderParams {
        shape: Shape::Box {
            half_extents: [0.2, 0.2, 0.2],
        },
        transform: Transform {
            keys: vec![
                Key {
                    frame: 0.0,
                    translate: [0.8, 0.8, 0.6],
                    rotate: None,
                },
                Key {
                    frame: 24.0,
                    translate: [1.2, 0.9, 0.6],
                    rotate: Some(Rotate {
                        axis: [0.0, 0.0, 1.0],
                        degrees: 90.0,
                    }),
                },
            ],
        },
    };
    let (_, faces, pose) = run(&gpu, &spinning, 6.0);
    for (a, face) in faces.iter().enumerate() {
        let d = face_dims(CELLS, a);
        let off = face_offset(a);
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let x = [i as f32 + off[0], j as f32 + off[1], k as f32 + off[2]]
                        .map(|c| f64::from(c) * f64::from(DX));
                    let want = pose.velocity_at(x)[a];
                    let got = f64::from(face[index(d, i, j, k)]);
                    assert!(
                        (got - want).abs() <= 1e-5,
                        "face {a} {:?}: {got} vs {want}",
                        [i, j, k]
                    );
                }
            }
        }
    }
}

/// Mirrors `face_min` in weights.wgsl.
fn face_min(s: &[f32], axis: usize, p: [u32; 3]) -> f32 {
    let n = [CELLS.x as i32, CELLS.y as i32, CELLS.z as i32];
    let at = |q: [i32; 3]| {
        let c: [u32; 3] = std::array::from_fn(|a| q[a].clamp(0, n[a] - 1) as u32);
        s[index(CELLS, c[0], c[1], c[2])]
    };
    let mut below = p.map(|v| v as i32);
    below[axis] -= 1;
    at(below).min(at(p.map(|v| v as i32)))
}

/// Spec §2.4: the union's SDF is the minimum, and each face takes the
/// velocity of the collider nearer to it.
#[test]
fn a_collider_union_takes_the_nearer_colliders_distance_and_velocity() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let moving = |from: [f32; 3], to: [f32; 3]| Transform {
        keys: vec![
            Key {
                frame: 0.0,
                translate: from,
                rotate: None,
            },
            Key {
                frame: 24.0,
                translate: to,
                rotate: None,
            },
        ],
    };
    let a = ColliderParams {
        shape: Shape::Sphere { radius: 0.3 },
        transform: moving([0.6, 0.8, 0.6], [0.6, 0.8, 1.0]),
    };
    let b = ColliderParams {
        shape: Shape::Box {
            half_extents: [0.25; 3],
        },
        transform: moving([1.4, 0.8, 0.6], [1.0, 0.8, 0.6]),
    };
    let make = |pool: &mut FieldPool| {
        (
            pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap(),
            pool.acquire_staggered_uninit(&gpu, CELLS).unwrap(),
        )
    };
    let (sa, va) = make(&mut pool);
    let (sb, vb) = make(&mut pool);
    let (so, vo) = make(&mut pool);
    for (p, s, v) in [(&a, &sa, &va), (&b, &sb, &vb)] {
        let pose = p.transform.pose(6.0, SPF);
        fill_collider(
            &gpu,
            &mut cache,
            p,
            &pose,
            DX,
            ColliderFields {
                sdf: s,
                velocity: v,
            },
        )
        .unwrap();
    }
    union_colliders(
        &gpu,
        &mut cache,
        ColliderFields {
            sdf: &sa,
            velocity: &va,
        },
        ColliderFields {
            sdf: &sb,
            velocity: &vb,
        },
        ColliderFields {
            sdf: &so,
            velocity: &vo,
        },
    )
    .unwrap();
    let (ca, cb, co) = (
        sa.read_back(&gpu).unwrap(),
        sb.read_back(&gpu).unwrap(),
        so.read_back(&gpu).unwrap(),
    );
    for n in 0..CELLS.voxel_count() {
        assert_eq!(co[n], ca[n].min(cb[n]), "sdf {n}");
    }
    let (ua, ub, uo) = (
        read_staggered(&gpu, &va),
        read_staggered(&gpu, &vb),
        read_staggered(&gpu, &vo),
    );
    for ax in 0..3 {
        let d = face_dims(CELLS, ax);
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let at = index(d, i, j, k);
                    let nearer_a = face_min(&ca, ax, [i, j, k]) <= face_min(&cb, ax, [i, j, k]);
                    let want = if nearer_a { ua[ax][at] } else { ub[ax][at] };
                    assert_eq!(uo[ax][at], want, "face {ax} {:?}", [i, j, k]);
                }
            }
        }
    }
}

#[test]
fn bad_collider_parameters_are_rejected() {
    let build = |p: serde_json::Value| {
        elements_ember::registry()
            .build(elements_ember::collider::KIND, &p)
            .is_err()
    };
    assert!(!build(
        serde_json::json!({ "shape": { "sphere": { "radius": 0.2 } }, "transform": { "keys": [{ "frame": 0 }] } })
    ));
    assert!(build(
        serde_json::json!({ "shape": { "sphere": { "radius": -0.2 } }, "transform": { "keys": [{ "frame": 0 }] } })
    ));
    assert!(
        build(serde_json::json!({ "shape": { "sphere": { "radius": 0.2 } } })),
        "transform is required"
    );
    assert!(build(
        serde_json::json!({ "shape": { "sphere": { "radius": 0.2 } }, "transform": { "keys": [{ "frame": 0 }] }, "density_rate": 1.0 })
    ));
}
