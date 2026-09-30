mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_ember::collider::ColliderFields;
use elements_ember::mesh::Mesh;
use elements_ember::mesh_collider::{MeshColliderParams, fill_mesh_collider};
use elements_ember::transform::{Key, Rotate, Transform};

const SPF: f64 = 1.0 / 24.0;
const N: u32 = 32;
const DX: f32 = 2.0 / 32.0;

fn fill(params: &MeshColliderParams, frame: f64) -> (Vec<f32>, [Vec<f32>; 3]) {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(N, N, N);
    let sdf = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let v = pool.acquire_staggered_uninit(&gpu, cells).unwrap();
    let pose = params.transform.pose(frame, SPF);
    fill_mesh_collider(
        &gpu,
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
    (sdf.read_back(&gpu).unwrap(), read_staggered(&gpu, &v))
}

fn at(translate: [f32; 3], rotate: Option<Rotate>) -> Transform {
    Transform {
        keys: vec![Key {
            frame: 0.0,
            translate,
            rotate,
        }],
    }
}

fn cell(i: u32) -> f64 {
    (f64::from(i) + 0.5) * f64::from(DX)
}

/// Exact distance to the axis-aligned box [lo, hi], negative inside.
fn box_sdf(lo: [f64; 3], hi: [f64; 3], p: [f64; 3]) -> f64 {
    let c: [f64; 3] = std::array::from_fn(|a| (lo[a] + hi[a]) / 2.0);
    let h: [f64; 3] = std::array::from_fn(|a| (hi[a] - lo[a]) / 2.0);
    let q: [f64; 3] = std::array::from_fn(|a| (p[a] - c[a]).abs() - h[a]);
    let out = q.iter().map(|v| v.max(0.0).powi(2)).sum::<f64>().sqrt();
    out + q[0].max(q[1]).max(q[2]).min(0.0)
}

/// A box mesh gives the exact box distance, inside and out, at every cell.
#[test]
fn a_box_mesh_matches_the_analytic_box_distance() {
    let params = MeshColliderParams {
        mesh: Mesh::box_mesh([-0.3, -0.2, -0.25], [0.3, 0.2, 0.25]),
        transform: at([1.0, 1.0, 1.0], None),
        offset: 0.0,
    };
    let (sdf, _) = fill(&params, 0.0);
    let (lo, hi) = ([0.7, 0.8, 0.75], [1.3, 1.2, 1.25]);
    let cells = FieldDims::new(N, N, N);
    let mut worst = 0.0f64;
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                let want = box_sdf(lo, hi, [cell(i), cell(j), cell(k)]);
                let got = f64::from(sdf[index(cells, i, j, k)]);
                worst = worst.max((got - want).abs());
            }
        }
    }
    assert!(worst < 1e-4, "worst error {worst} m");
}

/// A rotated mesh is the box turned: the axis-aligned check on a box turned
/// 90° about z swaps its x and y half-extents.
#[test]
fn a_rotated_mesh_is_the_rotated_shape() {
    let params = MeshColliderParams {
        mesh: Mesh::box_mesh([-0.5, -0.1, -0.1], [0.5, 0.1, 0.1]),
        transform: at(
            [1.0, 1.0, 1.0],
            Some(Rotate {
                axis: [0.0, 0.0, 1.0],
                degrees: 90.0,
            }),
        ),
        offset: 0.0,
    };
    let (sdf, _) = fill(&params, 0.0);
    let cells = FieldDims::new(N, N, N);
    assert!(sdf[index(cells, 16, 22, 16)] < 0.0, "along +y is inside");
    assert!(sdf[index(cells, 22, 16, 16)] > 0.0, "along +x is outside");
}

/// `offset` inflates the solid by exactly that distance.
#[test]
fn offset_shrinks_the_distance() {
    let base = MeshColliderParams {
        mesh: Mesh::box_mesh([-0.3; 3], [0.3; 3]),
        transform: at([1.0; 3], None),
        offset: 0.0,
    };
    let inflated = MeshColliderParams {
        offset: 0.05,
        ..base.clone()
    };
    let (a, _) = fill(&base, 0.0);
    let (b, _) = fill(&inflated, 0.0);
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y - 0.05).abs() < 1e-5, "{x} {y}");
    }
}

/// A zero-area triangle inside a closed mesh is skipped, not turned into NaN.
#[test]
fn a_degenerate_triangle_does_not_poison_the_field() {
    let mut mesh = Mesh::box_mesh([-0.3; 3], [0.3; 3]);
    mesh.positions.push([0.1, 0.1, 0.1]);
    let v = (mesh.positions.len() - 1) as u32;
    mesh.indices.extend_from_slice(&[v, v, v]);
    let params = MeshColliderParams {
        mesh,
        transform: at([1.0; 3], None),
        offset: 0.0,
    };
    let (sdf, _) = fill(&params, 0.0);
    assert!(sdf.iter().all(|d| d.is_finite()));
    let cells = FieldDims::new(N, N, N);
    assert!(sdf[index(cells, 16, 16, 16)] < 0.0);
}

/// A moving mesh carries its material velocity to the faces, as
/// `ember.collider` does.
#[test]
fn a_moving_mesh_has_its_linear_velocity_on_the_faces() {
    let mut t = at([1.0, 1.0, 1.0], None);
    t.keys.push(Key {
        frame: 24.0,
        translate: [2.0, 1.0, 1.0],
        rotate: None,
    });
    let params = MeshColliderParams {
        mesh: Mesh::box_mesh([-0.3; 3], [0.3; 3]),
        transform: t,
        offset: 0.0,
    };
    let (_, [u, v, _]) = fill(&params, 12.0);
    assert!(
        (u[0] - 1.0 / (24.0 * SPF) as f32).abs() < 1e-4,
        "u {}",
        u[0]
    );
    assert!(v[0].abs() < 1e-5);
}

#[test]
fn the_node_is_registered_and_validates_its_mesh() {
    use elements_core::graph::Document;
    let ok = r#"{ "version": 3, "dims": [8,8,8], "fps": 24.0, "start_frame": 1, "domain_size": 2.0,
      "nodes": [ { "id": 0, "kind": "ember.mesh_collider",
        "params": { "mesh": { "positions": [[0,0,0],[1,0,0],[0,1,0]], "indices": [0,1,2] },
                    "transform": { "keys": [ { "frame": 0.0, "translate": [1,1,1] } ] } } },
        { "id": 1, "kind": "core.output", "params": {} } ],
      "edges": [ { "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 } ], "output": 1 }"#;
    Document::from_json(ok)
        .unwrap()
        .into_graph(&elements_ember::registry())
        .unwrap();
    let bad = ok.replace("\"indices\": [0,1,2]", "\"indices\": [0,1,9]");
    let err = Document::from_json(&bad)
        .unwrap()
        .into_graph(&elements_ember::registry());
    assert!(
        err.is_err(),
        "an out-of-range index must be rejected at build"
    );
}
