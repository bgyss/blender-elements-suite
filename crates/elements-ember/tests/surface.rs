//! FT4 surface ignition (spec 2026-09-30-flamethrower-ft4-surface-ignition-design.md).

mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldPool, PipelineCache};
use elements_ember::collider::{ColliderFields, ColliderParams, SurfaceFuel, fill_collider};
use elements_ember::transform::{Key, Shape, Transform};

const DX: f32 = 0.125;

fn still_box(surface_fuel: Option<SurfaceFuel>) -> ColliderParams {
    ColliderParams {
        shape: Shape::Box {
            half_extents: [0.25, 0.25, 0.125],
        },
        transform: Transform {
            keys: vec![Key {
                frame: 0.0,
                translate: [0.5, 0.5, 0.5],
                rotate: None,
            }],
        },
        surface_fuel,
    }
}

#[test]
fn a_collider_outputs_its_load_where_the_sdf_is_negative() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(8, 8, 8);
    let sdf = pool
        .acquire(&gpu, cells, elements_core::gpu::FieldFormat::R32Float)
        .unwrap();
    let load = pool
        .acquire(&gpu, cells, elements_core::gpu::FieldFormat::R32Float)
        .unwrap();
    let velocity = pool.acquire_staggered_uninit(&gpu, cells).unwrap();
    let params = still_box(Some(SurfaceFuel { load: 7.5 }));
    let pose = params.transform.pose(0.0, 1.0 / 24.0);
    fill_collider(
        &gpu,
        &mut cache,
        &params,
        &pose,
        DX,
        ColliderFields {
            sdf: &sdf,
            velocity: &velocity,
        },
    )
    .unwrap();
    elements_ember::collider::fill_surface_load(&gpu, &mut cache, &sdf, 7.5, &load).unwrap();
    let sdf_data = sdf.read_back(&gpu).unwrap();
    let load_data = load.read_back(&gpu).unwrap();
    let inside = sdf_data.iter().filter(|&&d| d < 0.0).count();
    assert!(inside > 0, "the box must cover some cells");
    for (d, l) in sdf_data.iter().zip(&load_data) {
        let want = if *d < 0.0 { 7.5 } else { 0.0 };
        assert_eq!(*l, want, "sdf {d}");
    }
}

#[test]
fn surface_fuel_rejects_a_moving_collider_and_a_negative_load() {
    let moving = serde_json::json!({
        "shape": { "sphere": { "radius": 0.1 } },
        "transform": { "keys": [
            { "frame": 0, "translate": [0.0, 0.0, 0.0] },
            { "frame": 10, "translate": [1.0, 0.0, 0.0] } ] },
        "surface_fuel": { "load": 3.0 }
    });
    assert!(
        elements_ember::registry()
            .build("ember.collider", &moving)
            .is_err()
    );
    let negative = serde_json::json!({
        "shape": { "sphere": { "radius": 0.1 } },
        "transform": { "keys": [{ "frame": 0, "translate": [0.0, 0.0, 0.0] }] },
        "surface_fuel": { "load": -1.0 }
    });
    assert!(
        elements_ember::registry()
            .build("ember.collider", &negative)
            .is_err()
    );
    let still = serde_json::json!({
        "shape": { "sphere": { "radius": 0.1 } },
        "transform": { "keys": [{ "frame": 0, "translate": [0.0, 0.0, 0.0] }] },
        "surface_fuel": { "load": 3.0 }
    });
    assert!(
        elements_ember::registry()
            .build("ember.collider", &still)
            .is_ok()
    );
}

#[test]
fn a_union_keeps_the_larger_load_and_a_missing_side_counts_as_zero() {
    use elements_ember::unions::union_surface_load;
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(4, 4, 4);
    let n = cells.voxel_count();
    let a: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i * 3) % 4) as f32).collect();
    let fa = upload(&gpu, &mut pool, cells, &a);
    let fb = upload(&gpu, &mut pool, cells, &b);
    let out = upload(&gpu, &mut pool, cells, &vec![-1.0; n]);
    union_surface_load(&gpu, &mut cache, &fa, &fb, &out).unwrap();
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x.max(*y)).collect();
    assert_eq!(out.read_back(&gpu).unwrap(), want);
}

#[test]
fn a_union_with_one_unconnected_load_outputs_the_other() {
    let doc = r#"{ "version": 3, "dims": [8, 8, 8], "fps": 24.0, "domain_size": 1.0,
      "nodes": [
        { "id": 0, "kind": "ember.collider", "params": {
            "shape": { "box": { "half_extents": [0.2, 0.2, 0.1] } },
            "transform": { "keys": [{ "frame": 0, "translate": [0.5, 0.5, 0.5] }] },
            "surface_fuel": { "load": 4.0 } } },
        { "id": 1, "kind": "ember.collider", "params": {
            "shape": { "sphere": { "radius": 0.05 } },
            "transform": { "keys": [{ "frame": 0, "translate": [0.1, 0.1, 0.1] }] } } },
        { "id": 2, "kind": "ember.collider_union", "params": {} },
        { "id": 3, "kind": "core.output", "params": {} } ],
      "edges": [
        { "from_node": 0, "from_index": 0, "to_node": 2, "to_index": 0 },
        { "from_node": 0, "from_index": 1, "to_node": 2, "to_index": 1 },
        { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 2 },
        { "from_node": 1, "from_index": 1, "to_node": 2, "to_index": 3 },
        { "from_node": 0, "from_index": 2, "to_node": 2, "to_index": 4 },
        { "from_node": 2, "from_index": 2, "to_node": 3, "to_index": 0 } ],
      "output": 3 }"#;
    let mut s = Session::new(doc);
    let mut t = timeline(0);
    let bits = s.density_bits(&mut t, 1);
    let loads: Vec<f32> = bits.iter().map(|b| f32::from_bits(*b)).collect();
    assert!(loads.contains(&4.0), "the first collider's load survives");
    assert!(loads.iter().all(|&l| l == 0.0 || l == 4.0), "{loads:?}");
}
