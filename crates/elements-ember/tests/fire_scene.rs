//! Fire through documents and the solver node (2b-4 spec §4.3, §5).

mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldPool, PipelineCache};
use elements_core::graph::{Document, Graph, NodeError, NodeId, StateStore, Time};
use elements_ember::bench::{FIRE_FUEL_RATE, Scene};
use elements_ember::collider::ColliderParams;
use elements_ember::metrics::centroid_z;
use elements_ember::solver::{FUEL, REACT};
use elements_ember::transform::{Shape, Transform};

const SOLVER: NodeId = NodeId(1);

/// A 16³ plume, optionally fed fuel at `fuel_rate` from a second sphere
/// emitter's density output, with the solver's output `socket` as the
/// document's output.
fn doc(fuel_rate: Option<f32>, socket: u32) -> String {
    let (fuel_node, fuel_edge) = match fuel_rate {
        None => (String::new(), String::new()),
        Some(r) => (
            format!(
                r#",{{ "id": 3, "kind": "ember.sphere_emitter", "params": {{
                "center": [1.0, 1.0, 0.4], "radius": 0.3,
                "density_rate": {r:?}, "temperature_rate": 0.0 }} }}"#
            ),
            r#",{ "from_node": 3, "from_index": 0, "to_node": 1, "to_index": 6 }"#.to_owned(),
        ),
    };
    format!(
        r#"{{ "version": 3, "dims": [16, 16, 16], "fps": 24.0, "domain_size": 2.0,
  "nodes": [
    {{ "id": 0, "kind": "ember.sphere_emitter", "params": {{ "center": [1.0, 1.0, 0.4],
       "radius": 0.3, "density_rate": 1.0, "temperature_rate": 2.0 }} }},
    {{ "id": 1, "kind": "ember.smoke_solver", "params": {{ "buoyancy_temperature": 1.0 }} }},
    {{ "id": 2, "kind": "core.output", "params": {{}} }}{fuel_node}
  ],
  "edges": [
    {{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }},
    {{ "from_node": 0, "from_index": 1, "to_node": 1, "to_index": 1 }},
    {{ "from_node": 1, "from_index": {socket}, "to_node": 2, "to_index": 0 }}{fuel_edge}
  ],
  "output": 2 }}"#
    )
}

struct Run {
    gpu: elements_core::gpu::GpuContext,
    pool: FieldPool,
    pipelines: PipelineCache,
    state: StateStore,
    graph: Graph,
    dims: FieldDims,
}

impl Run {
    fn new(doc: &str) -> Self {
        let (graph, dims) = Document::from_json(doc)
            .unwrap()
            .into_graph(&elements_ember::registry())
            .unwrap();
        Self {
            gpu: gpu(),
            pool: FieldPool::new(),
            pipelines: PipelineCache::new(),
            state: StateStore::new(),
            graph,
            dims,
        }
    }

    /// Evaluate `frame` and read the output back.
    fn frame(&mut self, frame: u32) -> Result<Vec<f32>, NodeError> {
        let time = Time::at(frame, 1, 24.0);
        let out = self.graph.eval_frame(
            &self.gpu,
            &mut self.pool,
            &mut self.pipelines,
            &mut self.state,
            time,
            self.dims,
        )?;
        let values = out.value.as_field().unwrap().read_back(&self.gpu).unwrap();
        out.value.release_to(&mut self.pool);
        Ok(values)
    }

    fn slot(&self, slot: &'static str) -> Vec<f32> {
        self.state
            .get(SOLVER, slot)
            .unwrap()
            .as_field()
            .unwrap()
            .read_back(&self.gpu)
            .unwrap()
    }
}

#[test]
fn the_flame_output_is_the_square_root_of_react() {
    let mut r = Run::new(&doc(Some(24.0), 3));
    let mut flame = Vec::new();
    for f in 1..=10 {
        flame = r.frame(f).unwrap();
    }
    // `kernels::flame` clamps react to [0, 1] before the square root (the
    // global mass correction can push react slightly above 1), so this must
    // match that clamp rather than a bare `max(0.0)`.
    let want: Vec<f32> = r
        .slot(REACT)
        .iter()
        .map(|v| v.clamp(0.0, 1.0).sqrt())
        .collect();
    assert!(want.iter().any(|&v| v > 0.0), "something burns");
    assert_close(&flame, &want, 1e-6, "flame");
}

#[test]
fn fuel_at_zero_rate_changes_nothing() {
    let mut off = Run::new(&doc(None, 0));
    let mut on = Run::new(&doc(Some(0.0), 0));
    for f in 1..=20 {
        let (a, b) = (off.frame(f).unwrap(), on.frame(f).unwrap());
        // IEEE ==, so −0 equals +0.
        assert!(
            a.iter().zip(&b).all(|(x, y)| x == y),
            "density differs at frame {f}"
        );
    }
    assert!(on.slot(FUEL).iter().all(|&v| v == 0.0));
    let mut flame = Run::new(&doc(Some(0.0), 3));
    assert!(
        flame.frame(1).unwrap().iter().all(|&v| v == 0.0),
        "no fuel, no flame"
    );
}

#[test]
fn without_fuel_the_flame_output_is_zero() {
    let mut r = Run::new(&doc(None, 3));
    for f in 1..=3 {
        assert!(r.frame(f).unwrap().iter().all(|&v| v == 0.0));
    }
}

#[test]
fn connecting_fuel_changes_the_state_shape() {
    for (first, then, slot) in [(None, Some(24.0), FUEL), (Some(24.0), None, FUEL)] {
        let mut a = Run::new(&doc(first, 0));
        a.frame(1).unwrap();
        let (graph, _) = Document::from_json(&doc(then, 0))
            .unwrap()
            .into_graph(&elements_ember::registry())
            .unwrap();
        a.graph = graph;
        match a.frame(2) {
            Err(NodeError::StateShape { slot: s, .. }) => assert_eq!(s, slot),
            other => panic!("{first:?} → {then:?}: {other:?}"),
        }
        // Frame 1 left its state fields (density, temperature, pressure,
        // velocity's three faces) out of the pool, held in the state store.
        // `take_state`'s mismatch path takes them all out of the state store
        // before finding the mismatch, so it must release every one of them
        // before returning the error, following
        // `a_failed_step_with_mgpcg_returns_every_field_to_the_pool` in
        // tests/solver.rs.
        assert_eq!(
            a.pool.pooled_count() as u64,
            a.pool.allocation_count(),
            "{first:?} → {then:?}: the failed frame must return every taken field to the pool"
        );
    }
}

fn scene_run(scene: &Scene, socket: u32) -> Run {
    Run::new(&scene.document_with_output(socket).to_json().unwrap())
}

#[test]
fn the_fire_scene_wires_fuel_to_the_solver() {
    let doc = Scene::fire(16).document();
    assert!(
        doc.edges
            .iter()
            .any(|e| (e.from_node, e.from_index, e.to_node, e.to_index) == (0, 4, 1, 6))
    );
    assert!(
        !Scene::plume(16)
            .document()
            .edges
            .iter()
            .any(|e| e.to_index == 6)
    );
    let j = Scene::fire(64).mantaflow_json();
    assert_eq!(
        j["emitter"]["fuel_rate"],
        serde_json::json!(FIRE_FUEL_RATE as f64)
    );
    assert_eq!(j["fire"]["burning_rate"], serde_json::json!(1.875));
    assert!(Scene::plume(64).mantaflow_json()["fire"].is_null());
}

#[test]
fn the_flame_rises() {
    let scene = Scene::fire(32);
    let mut r = scene_run(&scene, 3);
    let cells = FieldDims::new(32, 32, 32);
    let mut heights = Vec::new();
    for f in 1..=24 {
        let flame = r.frame(f).unwrap();
        if [4, 12, 24].contains(&f) {
            heights.push(centroid_z(&flame, cells).expect("there is flame"));
        }
    }
    assert!(
        heights[0] < heights[1] && heights[1] < heights[2],
        "flame centroid {heights:?}"
    );
}

#[test]
fn fire_frame_40_is_bit_identical_however_it_is_reached() {
    // Density carries the burn's smoke; frame 40 in order, after a scrub,
    // and after eviction (the same checks as the smoke scenes).
    let doc = Scene::fire(16).document().to_json().unwrap();
    assert_doc_frame_40_is_bit_identical(&doc);
}

#[test]
fn no_fuel_or_flame_enters_a_collider() {
    let mut scene = Scene::fire(32);
    scene.colliders = vec![ColliderParams {
        shape: Shape::Sphere { radius: 0.25 },
        transform: Transform::at([1.0, 1.0, 0.8]),
    }];
    let solid = scene.solid_mask();
    let mut r = scene_run(&scene, 3);
    for f in 1..=40 {
        let flame = r.frame(f).unwrap();
        if f >= 10 {
            let fuel = r.slot(FUEL);
            for (n, &s) in solid.iter().enumerate() {
                if s {
                    assert!(
                        fuel[n].abs() <= 1e-6 && flame[n] <= 1e-3,
                        "cell {n} frame {f}"
                    );
                }
            }
        }
    }
}
