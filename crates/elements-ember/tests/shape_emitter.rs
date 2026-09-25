mod common;

use common::*;
use elements_core::gpu::{
    Field, FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache, StaggeredField,
};
use elements_core::graph::{DocError, Document, StateStore, Time};
use elements_ember::shape_emitter::{EmitterFields, EmitterParams, fill_emitter};
use elements_ember::transform::{Key, Rotate, Shape, Transform};
use elements_ember::unions::union_emitters;

const SPF: f64 = 1.0 / 24.0;

/// Run one emitter at `frame` in a 2 m domain of `cells`: the three cell
/// outputs (density, temperature, weight) and the three velocity faces.
fn run(
    gpu: &GpuContext,
    cells: FieldDims,
    params: &EmitterParams,
    frame: f64,
) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let dx = 2.0 / cells.x.max(cells.y).max(cells.z) as f32;
    let f: [_; 3] =
        std::array::from_fn(|_| pool.acquire(gpu, cells, FieldFormat::R32Float).unwrap());
    let v = pool.acquire_staggered_uninit(gpu, cells).unwrap();
    let pose = params.transform.pose(frame, SPF);
    fill_emitter(
        gpu,
        &mut cache,
        params,
        &pose,
        frame * SPF,
        dx,
        EmitterFields {
            density: &f[0],
            temperature: &f[1],
            weight: &f[2],
            velocity: &v,
        },
    )
    .unwrap();
    (
        f.map(|x| x.read_back(gpu).unwrap()),
        read_staggered(gpu, &v),
    )
}

fn keyed(translate: [f32; 3], rotate: Option<Rotate>) -> Transform {
    Transform {
        keys: vec![Key {
            frame: 0.0,
            translate,
            rotate,
        }],
    }
}

/// Spec §2.2: occupancy has the one-voxel smoothed edge, so the emitted total
/// matches the shape's volume at any resolution, and rotation does not change
/// it.
#[test]
fn a_box_emits_its_volume_at_any_resolution_and_orientation() {
    let gpu = gpu();
    let volume = 0.6 * 0.4 * 0.5;
    for n in [32u32, 64] {
        let cells = FieldDims::new(n, n, n);
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
                ..EmitterParams::new(
                    Shape::Box {
                        half_extents: [0.3, 0.2, 0.25],
                    },
                    keyed([1.0, 1.0, 1.0], rotate),
                )
            };
            let ([d, _, _], _) = run(&gpu, cells, &p, 0.0);
            let total = d.iter().map(|&v| f64::from(v)).sum::<f64>() * dx.powi(3);
            assert!(
                (total - volume).abs() <= 0.05 * volume,
                "{n}³ {rotate:?}: {total} vs {volume}"
            );
        }
    }
}

/// A long thin box turned 90° about z points along y: the point 0.4 m along
/// +y from its centre is inside, and the point 0.4 m along +x is not.
#[test]
fn rotation_turns_the_box() {
    let gpu = gpu();
    let cells = FieldDims::new(32, 32, 32);
    let p = EmitterParams {
        density_rate: 1.0,
        ..EmitterParams::new(
            Shape::Box {
                half_extents: [0.5, 0.1, 0.1],
            },
            keyed(
                [1.0, 1.0, 1.0],
                Some(Rotate {
                    axis: [0.0, 0.0, 1.0],
                    degrees: 90.0,
                }),
            ),
        )
    };
    let ([d, _, _], _) = run(&gpu, cells, &p, 0.0);
    // dx = 1/16 m: cell 22 is centred at 1.40625 m, cell 16 at 1.03125 m.
    assert_eq!(d[index(cells, 16, 22, 16)], 1.0, "along +y");
    assert_eq!(d[index(cells, 22, 16, 16)], 0.0, "along +x");
}

/// Spec §2.2: the target velocity on every face is the emitter's `velocity`
/// plus its own motion there, and the weight is occupancy × blend.
#[test]
fn target_velocity_carries_the_emitters_motion_and_weight_is_occupancy_times_blend() {
    let gpu = gpu();
    let cells = FieldDims::new(12, 10, 8);
    let dx: f32 = 2.0 / 12.0;
    let transform = Transform {
        keys: vec![
            Key {
                frame: 0.0,
                translate: [0.8, 0.8, 0.6],
                rotate: None,
            },
            Key {
                frame: 24.0,
                translate: [1.2, 0.8, 0.7],
                rotate: Some(Rotate {
                    axis: [0.0, 0.0, 1.0],
                    degrees: 60.0,
                }),
            },
        ],
    };
    let p = EmitterParams {
        velocity: [0.5, -1.0, 0.25],
        velocity_blend: 2.0,
        ..EmitterParams::new(Shape::Sphere { radius: 0.3 }, transform.clone())
    };
    let frame = 6.0;
    let ([_, _, w], faces) = run(&gpu, cells, &p, frame);
    let pose = transform.pose(frame, SPF);
    for (a, face) in faces.iter().enumerate() {
        let d = face_dims(cells, a);
        let off = face_offset(a);
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let x = [i as f32 + off[0], j as f32 + off[1], k as f32 + off[2]]
                        .map(|c| f64::from(c) * f64::from(dx));
                    let want = f64::from(p.velocity[a]) + pose.velocity_at(x)[a];
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
    // The sphere's centre cell is fully occupied: weight = 1 × blend.
    let c = pose.translate.map(|v| (v / f64::from(dx)) as u32);
    assert_eq!(w[index(cells, c[0], c[1], c[2])], 2.0);
    assert_eq!(w[index(cells, 0, 0, 0)], 0.0);
}

/// Mirrors `face_weight` in `weights.wgsl`.
fn face_weight(w: &[f32], cells: FieldDims, axis: usize, p: [u32; 3]) -> f32 {
    let n = [cells.x as i32, cells.y as i32, cells.z as i32];
    let at = |q: [i32; 3]| {
        let c: [u32; 3] = std::array::from_fn(|a| q[a].clamp(0, n[a] - 1) as u32);
        w[index(cells, c[0], c[1], c[2])]
    };
    let mut below = p.map(|v| v as i32);
    below[axis] -= 1;
    at(below).max(at(p.map(|v| v as i32)))
}

/// The four emitter outputs held in `f` and `v`.
fn fields<'a>(f: &'a [Field; 3], v: &'a StaggeredField) -> EmitterFields<'a> {
    EmitterFields {
        density: &f[0],
        temperature: &f[1],
        weight: &f[2],
        velocity: v,
    }
}

/// Spec §2.4: rates add, the weight is the maximum, and the target velocity
/// is weighted by each emitter's face weight.
#[test]
fn a_union_adds_rates_takes_the_max_weight_and_weights_velocity() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(12, 10, 8);
    let dx = 2.0 / 12.0;
    let a = EmitterParams {
        density_rate: 1.0,
        temperature_rate: 1.0,
        velocity: [1.0, 0.0, 0.0],
        velocity_blend: 1.0,
        ..EmitterParams::new(
            Shape::Box {
                half_extents: [0.3, 0.3, 0.3],
            },
            keyed([0.8, 0.8, 0.6], None),
        )
    };
    let b = EmitterParams {
        density_rate: 2.0,
        temperature_rate: 0.5,
        velocity: [0.0, 0.0, 2.0],
        velocity_blend: 3.0,
        ..EmitterParams::new(Shape::Sphere { radius: 0.35 }, keyed([1.1, 0.8, 0.7], None))
    };
    let make = |pool: &mut FieldPool| {
        let f: [_; 3] =
            std::array::from_fn(|_| pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap());
        (f, pool.acquire_staggered_uninit(&gpu, cells).unwrap())
    };
    let (fa, va) = make(&mut pool);
    let (fb, vb) = make(&mut pool);
    let (fo, vo) = make(&mut pool);
    for (p, f, v) in [(&a, &fa, &va), (&b, &fb, &vb)] {
        let pose = p.transform.pose(0.0, SPF);
        fill_emitter(&gpu, &mut cache, p, &pose, 0.0, dx, fields(f, v)).unwrap();
    }
    union_emitters(
        &gpu,
        &mut cache,
        fields(&fa, &va),
        fields(&fb, &vb),
        fields(&fo, &vo),
    )
    .unwrap();

    let read = |f: &[Field; 3]| f.each_ref().map(|x| x.read_back(&gpu).unwrap());
    let (ca, cb, co) = (read(&fa), read(&fb), read(&fo));
    for n in 0..cells.voxel_count() {
        assert_eq!(co[0][n], ca[0][n] + cb[0][n], "density {n}");
        assert_eq!(co[1][n], ca[1][n] + cb[1][n], "temperature {n}");
        assert_eq!(co[2][n], ca[2][n].max(cb[2][n]), "weight {n}");
    }
    let (ua, ub, uo) = (
        read_staggered(&gpu, &va),
        read_staggered(&gpu, &vb),
        read_staggered(&gpu, &vo),
    );
    for ax in 0..3 {
        let d = face_dims(cells, ax);
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let at = index(d, i, j, k);
                    let (wa, wb) = (
                        face_weight(&ca[2], cells, ax, [i, j, k]),
                        face_weight(&cb[2], cells, ax, [i, j, k]),
                    );
                    let want = (wa * ua[ax][at] + wb * ub[ax][at]) / (wa + wb).max(1e-12);
                    assert!(
                        (uo[ax][at] - want).abs() <= 1e-6,
                        "face {ax} {:?}: {} vs {want}",
                        [i, j, k],
                        uo[ax][at]
                    );
                }
            }
        }
    }
}

fn rejected(params: serde_json::Value) -> bool {
    matches!(
        elements_ember::registry().build(elements_ember::shape_emitter::KIND, &params),
        Err(DocError::BadParams { .. })
    )
}

#[test]
fn bad_emitter_parameters_are_rejected() {
    let ok = serde_json::json!({ "shape": { "sphere": { "radius": 0.2 } }, "transform": { "keys": [{ "frame": 0 }] } });
    assert!(!rejected(ok.clone()));
    let with = |k: &str, v: serde_json::Value| {
        let mut o = ok.clone();
        o[k] = v;
        o
    };
    assert!(rejected(with("velocity_blend", serde_json::json!(-1.0))));
    assert!(rejected(with("density_rate", serde_json::json!(1e39))));
    assert!(rejected(with(
        "transform",
        serde_json::json!({ "keys": [] })
    )));
    assert!(rejected(with(
        "shape",
        serde_json::json!({ "box": { "half_extents": [0.1, 0.0, 0.1] } })
    )));
    assert!(rejected(with("colour", serde_json::json!(1.0))));
}

/// A 16³ document: one sphere emitter wired straight to the output, with
/// `active` as its activity window.
fn window_doc(active: &str) -> Document {
    serde_json::from_str(&format!(
        r#"{{
      "version": 3, "dims": [16, 16, 16], "fps": 24.0, "domain_size": 2.0,
      "nodes": [
        {{ "id": 0, "kind": "ember.emitter", "params": {{
            "shape": {{ "sphere": {{ "radius": 0.4 }} }},
            "transform": {{ "keys": [ {{ "frame": 1, "translate": [1.0, 1.0, 1.0] }} ] }},
            "density_rate": 1.0, "active_frames": {active} }} }},
        {{ "id": 1, "kind": "core.output", "params": {{}} }}
      ],
      "edges": [ {{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }} ],
      "output": 1
    }}"#
    ))
    .unwrap()
}

/// The emitter's density output at `frame`.
fn density_at(doc: Document, frame: u32) -> Vec<f32> {
    let gpu = gpu();
    let registry = elements_ember::registry();
    let (graph, dims) = doc.into_graph(&registry).unwrap();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let mut state = StateStore::new();
    let time = Time::at(frame, 1, 24.0);
    let out = graph
        .eval_frame(&gpu, &mut pool, &mut cache, &mut state, time, dims)
        .unwrap();
    out.value.as_field().unwrap().read_back(&gpu).unwrap()
}

/// Spec §3: the emitter emits on its first and last active frames, and
/// nothing on the frame after.
#[test]
fn an_emitter_is_silent_outside_its_active_frames() {
    let on_last = density_at(window_doc("[1, 60]"), 60);
    assert!(on_last.iter().any(|&d| d > 0.0), "frame 60 emits");
    let on_first = density_at(window_doc("[5, 60]"), 5);
    assert!(on_first.iter().any(|&d| d > 0.0), "frame 5 emits");
    let after = density_at(window_doc("[1, 60]"), 61);
    assert!(after.iter().all(|&d| d == 0.0), "frame 61 is silent");
    let before = density_at(window_doc("[5, 60]"), 4);
    assert!(before.iter().all(|&d| d == 0.0), "frame 4 is silent");
    let always = density_at(window_doc("null"), 500);
    assert!(always.iter().any(|&d| d > 0.0), "no window is always on");
}

#[test]
fn an_active_range_must_not_run_backwards() {
    let registry = elements_ember::registry();
    assert!(
        window_doc("[60, 60]").into_graph(&registry).is_ok(),
        "a one-frame window is valid"
    );
    let err = window_doc("[61, 60]").into_graph(&registry).err();
    assert!(matches!(err, Some(DocError::BadParams { .. })), "{err:?}");
}

/// Total of an emitter output after frame 1, from a one-node document.
fn emitter_total(params: &str, output: u32) -> Vec<f32> {
    let doc = format!(
        r#"{{ "version": 3, "dims": [16, 16, 16], "fps": 24.0, "domain_size": 2.0,
  "nodes": [
    {{ "id": 0, "kind": "ember.emitter", "params": {params} }},
    {{ "id": 1, "kind": "core.output", "params": {{}} }} ],
  "edges": [ {{ "from_node": 0, "from_index": {output}, "to_node": 1, "to_index": 0 }} ],
  "output": 1 }}"#
    );
    let (graph, dims) = Document::from_json(&doc)
        .unwrap()
        .into_graph(&elements_ember::registry())
        .unwrap();
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let time = Time::at(1, 1, 24.0);
    let out = graph
        .eval_frame(&gpu, &mut pool, &mut pipelines, &mut state, time, dims)
        .unwrap();
    let v = out.value.as_field().unwrap().read_back(&gpu).unwrap();
    out.value.release_to(&mut pool);
    v
}

const SPHERE: &str = r#""shape": { "sphere": { "radius": 0.4 } },
    "transform": { "keys": [ { "frame": 1, "translate": [1.0, 1.0, 1.0] } ] }"#;

#[test]
fn fuel_is_emitted_where_density_is_at_its_own_rate() {
    let params = format!(
        r#"{{ {SPHERE}, "density_rate": 1.0, "fuel_rate": 3.0,
             "noise": {{ "seed": 7, "scale_m": 0.3, "amplitude": 0.5 }} }}"#
    );
    let density = emitter_total(&params, 0);
    let fuel = emitter_total(&params, 4);
    let want: Vec<f32> = density.iter().map(|d| 3.0 * d).collect();
    assert!(want.iter().any(|&v| v > 0.0));
    assert_close(&fuel, &want, 1e-6, "fuel");
}

#[test]
fn an_inactive_emitter_emits_no_fuel() {
    let params = format!(r#"{{ {SPHERE}, "fuel_rate": 3.0, "active_frames": [5, 9] }}"#);
    assert!(emitter_total(&params, 4).iter().all(|&v| v == 0.0));
}

/// A one-emitter document, with an optional fuel edge to a union that feeds
/// `core.output`. `centers` places two emitters; only those whose fuel rate
/// is nonzero and whose fuel output is wired to the union contribute.
fn union_fuel_doc(a_fuel_edge: bool, b_fuel_edge: bool) -> String {
    let a_edge = if a_fuel_edge {
        r#",{ "from_node": 0, "from_index": 4, "to_node": 2, "to_index": 8 }"#
    } else {
        ""
    };
    let b_edge = if b_fuel_edge {
        r#",{ "from_node": 1, "from_index": 4, "to_node": 2, "to_index": 9 }"#
    } else {
        ""
    };
    format!(
        r#"{{ "version": 3, "dims": [16, 16, 16], "fps": 24.0, "domain_size": 2.0,
  "nodes": [
    {{ "id": 0, "kind": "ember.emitter", "params": {{
        "shape": {{ "sphere": {{ "radius": 0.4 }} }},
        "transform": {{ "keys": [ {{ "frame": 1, "translate": [0.7, 1.0, 1.0] }} ] }},
        "fuel_rate": 1.0 }} }},
    {{ "id": 1, "kind": "ember.emitter", "params": {{
        "shape": {{ "sphere": {{ "radius": 0.4 }} }},
        "transform": {{ "keys": [ {{ "frame": 1, "translate": [1.3, 1.0, 1.0] }} ] }},
        "fuel_rate": 2.0 }} }},
    {{ "id": 2, "kind": "ember.emitter_union", "params": {{}} }},
    {{ "id": 3, "kind": "core.output", "params": {{}} }} ],
  "edges": [
    {{ "from_node": 0, "from_index": 0, "to_node": 2, "to_index": 0 }},
    {{ "from_node": 0, "from_index": 1, "to_node": 2, "to_index": 1 }},
    {{ "from_node": 0, "from_index": 2, "to_node": 2, "to_index": 2 }},
    {{ "from_node": 0, "from_index": 3, "to_node": 2, "to_index": 3 }},
    {{ "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 4 }},
    {{ "from_node": 1, "from_index": 1, "to_node": 2, "to_index": 5 }},
    {{ "from_node": 1, "from_index": 2, "to_node": 2, "to_index": 6 }},
    {{ "from_node": 1, "from_index": 3, "to_node": 2, "to_index": 7 }}{a_edge}{b_edge},
    {{ "from_node": 2, "from_index": 4, "to_node": 3, "to_index": 0 }} ],
  "output": 3 }}"#
    )
}

/// The union's output-4 total after frame 1, from `union_fuel_doc`.
fn union_fuel_total(a_fuel_edge: bool, b_fuel_edge: bool) -> Vec<f32> {
    let doc = union_fuel_doc(a_fuel_edge, b_fuel_edge);
    let (graph, dims) = Document::from_json(&doc)
        .unwrap()
        .into_graph(&elements_ember::registry())
        .unwrap();
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let time = Time::at(1, 1, 24.0);
    let out = graph
        .eval_frame(&gpu, &mut pool, &mut pipelines, &mut state, time, dims)
        .unwrap();
    let v = out.value.as_field().unwrap().read_back(&gpu).unwrap();
    out.value.release_to(&mut pool);
    v
}

#[test]
fn the_union_fuel_output_is_the_sum_of_the_connected_fuel_inputs() {
    let a = emitter_total(
        r#"{ "shape": { "sphere": { "radius": 0.4 } },
            "transform": { "keys": [ { "frame": 1, "translate": [0.7, 1.0, 1.0] } ] },
            "fuel_rate": 1.0 }"#,
        4,
    );
    let b = emitter_total(
        r#"{ "shape": { "sphere": { "radius": 0.4 } },
            "transform": { "keys": [ { "frame": 1, "translate": [1.3, 1.0, 1.0] } ] },
            "fuel_rate": 2.0 }"#,
        4,
    );
    assert!(a.iter().any(|&v| v > 0.0), "a emits");
    assert!(b.iter().any(|&v| v > 0.0), "b emits");

    let both = union_fuel_total(true, true);
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_close(&both, &want, 1e-6, "a + b");

    let a_only = union_fuel_total(true, false);
    assert_close(&a_only, &a, 1e-6, "a only");

    let neither = union_fuel_total(false, false);
    assert!(neither.iter().all(|&v| v == 0.0), "neither connected");
}

/// Existing union documents (inputs 0–7 only, no fuel edges) must still
/// validate: the new sockets are optional and appended.
#[test]
fn a_union_document_without_fuel_edges_still_validates() {
    let doc = union_fuel_doc(false, false);
    let registry = elements_ember::registry();
    Document::from_json(&doc)
        .unwrap()
        .into_graph(&registry)
        .expect("fuel-less union document must still validate");
}
