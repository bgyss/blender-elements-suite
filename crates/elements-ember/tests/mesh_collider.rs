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
fn a_zero_area_triangle_is_ignored() {
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
    let d = sdf[index(cells, 16, 16, 16)];
    assert!(d < 0.0);
    // The stray point is not a surface: the distance is to the box's wall
    // (0.3 - 0.03125 away), not the 0.119 m to the stray vertex.
    assert!((d + 0.26875).abs() < 1e-4, "{d}");
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

fn cell_centres() -> impl Iterator<Item = (u32, u32, u32)> {
    (0..N).flat_map(|k| (0..N).flat_map(move |j| (0..N).map(move |i| (i, j, k))))
}

/// Two closed boxes sharing a coincident face, as the shack merges its planks:
/// the sign must come from the whole surface, not the nearest triangle's side.
#[test]
fn touching_boxes_have_the_union_sign_everywhere() {
    let (alo, ahi) = ([0.4, 0.4, 0.4], [1.6, 0.8, 1.6]);
    let (blo, bhi) = ([0.4, 0.8, 0.4], [0.8, 1.6, 1.6]);
    let mut mesh = Mesh::box_mesh([0.4f32, 0.4, 0.4], [1.6, 0.8, 1.6]);
    mesh.merge(&Mesh::box_mesh([0.4, 0.8, 0.4], [0.8, 1.6, 1.6]));
    let params = MeshColliderParams {
        mesh,
        transform: at([0.0; 3], None),
        offset: 0.0,
    };
    let (sdf, _) = fill(&params, 0.0);
    let cells = FieldDims::new(N, N, N);
    let (mut wrong_sign, mut inside, mut worst_out) = (0, 0, 0.0f64);
    for (i, j, k) in cell_centres() {
        let p = [cell(i), cell(j), cell(k)];
        let (da, db) = (box_sdf(alo, ahi, p), box_sdf(blo, bhi, p));
        let want = da.min(db);
        let got = f64::from(sdf[index(cells, i, j, k)]);
        if want.abs() > 1e-3 {
            inside += usize::from(want < 0.0);
            wrong_sign += usize::from((got < 0.0) != (want < 0.0));
        }
        // Outside, the distance is exact.
        if want > 0.0 {
            worst_out = worst_out.max((got - want).abs());
        }
    }
    assert!(inside > 1000, "{inside} inside cells");
    assert_eq!(wrong_sign, 0, "wrong-sign cells");
    assert!(worst_out < 1e-4, "worst outside error {worst_out}");
}

/// Ericson's closest point, f64, for the CPU reference.
fn closest(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let sub = |x: [f64; 3], y: [f64; 3]| [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
    let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let at = |o: [f64; 3], d: [f64; 3], s: f64| [o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s];
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return at(a, ab, d1 / (d1 - d3));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return at(a, ac, d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return at(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    let q = at(a, ab, vb * denom);
    at(q, ac, vc * denom)
}

/// Signed distance by min distance and the generalized winding number, f64.
fn reference_sdf(mesh: &Mesh, p: [f64; 3]) -> f64 {
    let v = |i: u32| mesh.positions[i as usize].map(f64::from);
    let (mut best, mut winding) = (f64::MAX, 0.0);
    for t in mesh.indices.chunks(3) {
        let (a, b, c) = (v(t[0]), v(t[1]), v(t[2]));
        let q = closest(p, a, b, c);
        best = best
            .min(((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt());
        let r = |x: [f64; 3]| [x[0] - p[0], x[1] - p[1], x[2] - p[2]];
        let (ra, rb, rc) = (r(a), r(b), r(c));
        let len = |x: [f64; 3]| (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
        let cross = |x: [f64; 3], y: [f64; 3]| {
            [
                x[1] * y[2] - x[2] * y[1],
                x[2] * y[0] - x[0] * y[2],
                x[0] * y[1] - x[1] * y[0],
            ]
        };
        let (la, lb, lc) = (len(ra), len(rb), len(rc));
        if la == 0.0 || lb == 0.0 || lc == 0.0 {
            continue;
        }
        let den = la * lb * lc + dot(ra, rb) * lc + dot(rb, rc) * la + dot(rc, ra) * lb;
        winding += 2.0 * dot(ra, cross(rb, rc)).atan2(den);
    }
    if winding / (4.0 * std::f64::consts::PI) > 0.5 {
        -best
    } else {
        best
    }
}

/// A sharp, thin tetrahedron (acute dihedral angles) against a brute-force
/// f64 reference at every cell, including outside near the acute vertex.
#[test]
fn a_sharp_wedge_matches_the_reference_at_every_cell() {
    let positions = vec![
        [0.5f32, 0.6, 0.6],
        [1.6, 0.6, 0.6],
        [0.7, 0.75, 0.6],
        [0.8, 0.65, 1.4],
    ];
    let mut indices = vec![0u32, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2];
    // Orient outward: a negative signed volume means the winding is inverted.
    let vol: f64 = indices
        .chunks(3)
        .map(|t| {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| positions[i as usize].map(f64::from));
            a[0] * (b[1] * c[2] - b[2] * c[1])
                + a[1] * (b[2] * c[0] - b[0] * c[2])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
        })
        .sum();
    if vol < 0.0 {
        for t in indices.chunks_mut(3) {
            t.swap(1, 2);
        }
    }
    let mesh = Mesh { positions, indices };
    let params = MeshColliderParams {
        mesh: mesh.clone(),
        transform: at([0.0; 3], None),
        offset: 0.0,
    };
    let (sdf, _) = fill(&params, 0.0);
    let cells = FieldDims::new(N, N, N);
    let (mut inside, mut worst) = (0, 0.0f64);
    for (i, j, k) in cell_centres() {
        let want = reference_sdf(&mesh, [cell(i), cell(j), cell(k)]);
        let got = f64::from(sdf[index(cells, i, j, k)]);
        inside += usize::from(want < 0.0);
        worst = worst.max((got - want).abs());
    }
    assert!(inside > 10, "{inside} inside cells");
    assert!(worst < 1e-4, "worst error {worst} m");
}

/// A sliver (three points collinear to 1e-8 m) floating outside a closed box,
/// exactly through a row of cell centres, is no surface: every cell keeps the
/// analytic box distance and sign. Without the sliver skip it would be the
/// strict nearest triangle (distance 0) to that row.
#[test]
fn a_sliver_beside_a_box_is_not_a_surface() {
    let mut mesh = Mesh::box_mesh([-0.3; 3], [0.3; 3]);
    let n = mesh.positions.len() as u32;
    // Local x = 0.40625 and z = 0.03125 are world 1.40625 and 1.03125: the
    // centres of cells i = 22 and k = 16, so cells (22, 13..=18, 16) lie on it.
    mesh.positions.extend_from_slice(&[
        [0.40625, -0.1875, 0.03125],
        [0.40625, 0.0, 0.03125 + 1e-8],
        [0.40625, 0.1875, 0.03125],
    ]);
    mesh.indices.extend_from_slice(&[n, n + 1, n + 2]);
    let params = MeshColliderParams {
        mesh,
        transform: at([1.0; 3], None),
        offset: 0.0,
    };
    let (sdf, _) = fill(&params, 0.0);
    let cells = FieldDims::new(N, N, N);
    let (lo, hi) = ([0.7; 3], [1.3; 3]);
    let want_on_sliver = box_sdf(lo, hi, [cell(22), cell(16), cell(16)]);
    assert!(want_on_sliver > 0.1, "the row is outside the box");
    let mut worst = 0.0f64;
    for (i, j, k) in cell_centres() {
        let want = box_sdf(lo, hi, [cell(i), cell(j), cell(k)]);
        let got = f64::from(sdf[index(cells, i, j, k)]);
        assert!(got.is_finite());
        if want.abs() > 1e-3 {
            assert_eq!(
                got < 0.0,
                want < 0.0,
                "sign at ({i},{j},{k}): {got} vs {want}"
            );
        }
        worst = worst.max((got - want).abs());
    }
    assert!(worst < 1e-4, "worst error {worst} m");
}

/// A valid mesh over the dispatch budget is refused up front, quickly.
#[test]
fn a_dispatch_over_the_triangle_test_budget_is_refused() {
    use elements_ember::mesh_collider::MAX_TRIANGLE_TESTS;
    let tris = (MAX_TRIANGLE_TESTS / u64::from(N * N * N)) as usize + 1000;
    let mesh = Mesh {
        positions: vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        indices: (0..tris).flat_map(|_| [0, 1, 2]).collect(),
    };
    mesh.validate("test").unwrap();
    let params = MeshColliderParams {
        mesh,
        transform: at([1.0; 3], None),
        offset: 0.0,
    };
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(N, N, N);
    let sdf = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let v = pool.acquire_staggered_uninit(&gpu, cells).unwrap();
    let pose = params.transform.pose(0.0, SPF);
    let start = std::time::Instant::now();
    let err = fill_mesh_collider(
        &gpu,
        &mut cache,
        &params,
        &pose,
        DX,
        ColliderFields {
            sdf: &sdf,
            velocity: &v,
        },
    )
    .unwrap_err()
    .to_string();
    assert!(start.elapsed().as_secs_f64() < 1.0);
    assert!(err.contains("budget"), "{err}");
    // The same mesh on a tiny domain is under budget and runs.
    let tiny = FieldDims::new(2, 2, 2);
    let sdf = pool.acquire(&gpu, tiny, FieldFormat::R32Float).unwrap();
    let v = pool.acquire_staggered_uninit(&gpu, tiny).unwrap();
    fill_mesh_collider(
        &gpu,
        &mut cache,
        &params,
        &pose,
        DX,
        ColliderFields {
            sdf: &sdf,
            velocity: &v,
        },
    )
    .unwrap();
}

/// The shack is concave and open: a cell in the middle of the room is outside
/// (positive), a cell inside a wall plank is inside (negative), and a cell
/// above the roof is outside again.
#[test]
fn a_concave_shack_keeps_its_interior_outside_the_solid() {
    use elements_ember::shack::{ShackParams, shack};
    let p = ShackParams {
        size: [1.2, 1.0, 1.0],
        plank_height: 0.1,
        thickness: 0.05,
        gap: 0.0,
        broken_fraction: 0.0,
        seed: 1,
    };
    let params = MeshColliderParams {
        mesh: shack(&p),
        transform: at([1.0, 1.0, 0.0], None),
        offset: 0.0,
    };
    let (sdf, _) = fill(&params, 0.0);
    let cells = FieldDims::new(N, N, N);
    // Cell (16, 16, 8) is at (1.03, 1.03, 0.53): the room's middle, well clear
    // of every wall (nearest is the south wall's inner face, y = 0.55).
    let inside_room = sdf[index(cells, 16, 16, 8)];
    assert!(inside_room > 0.2, "the room is open air, sdf {inside_room}");
    // The south wall spans y in [0.5, 0.55]; cell row 8 is y = 0.53125, inside
    // it (x = 1.03 is within the wall's 0.4..1.6 span, z = 0.53 is mid-plank).
    let in_wall = sdf[index(cells, 16, 8, 8)];
    assert!(in_wall < 0.0, "the wall plank is solid, sdf {in_wall}");
    // The roof's top is z = 1.05; cell layer 20 is z = 1.28.
    let above = sdf[index(cells, 16, 16, 20)];
    assert!(above > 0.0, "above the roof, sdf {above}");
    // Under the roof, inside the room near its ceiling (z = 0.97, layer 15)
    // is still outside the solid: the winding number sees the room as empty.
    let under_roof = sdf[index(cells, 16, 16, 15)];
    assert!(
        under_roof > 0.0,
        "under the roof is open air, sdf {under_roof}"
    );
}

/// Cost of one SDF fill of the shot's shack. Run by hand:
/// `cargo nextest run -p elements-ember --run-ignored ignored-only -E 'test(shack_fill_cost)'`
/// (see docs/bench/mesh-collider.md).
#[test]
#[ignore = "timing measurement; prints, asserts nothing about speed"]
fn shack_fill_cost() {
    use elements_ember::shack::{ShackParams, shack};
    let mesh = shack(&ShackParams {
        size: [1.2, 1.0, 1.0],
        plank_height: 0.1,
        thickness: 0.02,
        gap: 0.01,
        broken_fraction: 0.2,
        seed: 7,
    });
    let params = MeshColliderParams {
        mesh,
        transform: at([1.0, 1.0, 0.0], None),
        offset: 0.0,
    };
    let gpu = gpu();
    println!("adapter: {}", gpu.adapter_name());
    println!("triangles: {}", params.mesh.triangle_count());
    let pose = params.transform.pose(0.0, SPF);
    for dims in [FieldDims::new(128, 128, 128), FieldDims::new(256, 128, 128)] {
        let mut pool = FieldPool::new();
        let mut cache = PipelineCache::new();
        let sdf = pool.acquire(&gpu, dims, FieldFormat::R32Float).unwrap();
        let v = pool.acquire_staggered_uninit(&gpu, dims).unwrap();
        let mut run = || {
            let start = std::time::Instant::now();
            fill_mesh_collider(
                &gpu,
                &mut cache,
                &params,
                &pose,
                DX,
                ColliderFields {
                    sdf: &sdf,
                    velocity: &v,
                },
            )
            .unwrap();
            gpu.wait().unwrap();
            start.elapsed().as_secs_f64() * 1e3
        };
        run(); // warm-up: pipeline compile
        let mut ms: Vec<f64> = (0..5).map(|_| run()).collect();
        ms.sort_by(f64::total_cmp);
        println!(
            "{}x{}x{}: median {:.1} ms, range {:.1}..{:.1} ms",
            dims.x, dims.y, dims.z, ms[2], ms[0], ms[4]
        );
    }
}
