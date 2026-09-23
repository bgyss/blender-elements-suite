//! Benchmark scenes and the speed gate's pre-registered rule (spec §4.3, §5.3).

use elements_core::graph::{DocEdge, DocNode, Document, ELEMENTS_DOC_VERSION, NodeId};

use crate::collider::{self, ColliderParams};
use crate::shape_emitter::{self, EmitterParams};
use crate::solver::{self, SolverParams};
use crate::transform::{Shape, Transform};

pub mod report;

/// A step must take at most this long at 128³ (≥ 10 fps).
pub const GATE_STEP_MS: f64 = 100.0;
/// Projection must leave at most this fraction of the RMS divergence.
pub const GATE_RATIO: f64 = 0.10;

/// Frames on which every bench scene emits (2b-3 spec §3). Frames after the
/// last measure mass drift with no sources.
pub const EMISSION_FRAMES: [u32; 2] = [1, 60];

/// The solver's node id in every `Scene::document`.
pub const SOLVER_NODE: NodeId = NodeId(1);

/// A benchmark scene, defined once (spec §5.3). It generates the `.elements`
/// document; the matching Mantaflow scene comes from the same values
/// (`docs/superpowers/specs/2026-09-23-ember-mantaflow-benchmark-2b3-design.md`).
#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    pub name: &'static str,
    pub cells: [u32; 3],
    /// Metres along the longest axis.
    pub domain_size: f64,
    pub fps: f64,
    pub frames: u32,
    pub emitter: EmitterParams,
    pub solver: SolverParams,
    /// A static collider wired to the solver's inputs 4 and 5, if any.
    pub collider: Option<ColliderParams>,
}

impl Scene {
    /// A hot, dense sphere at the domain floor, buoyancy only.
    pub fn plume(resolution: u32) -> Self {
        Self {
            name: "plume",
            cells: [resolution; 3],
            domain_size: 2.0,
            fps: 24.0,
            frames: 120,
            emitter: EmitterParams {
                density_rate: 1.0,
                temperature_rate: 1.0,
                active_frames: Some(EMISSION_FRAMES),
                ..EmitterParams::new(
                    Shape::Sphere { radius: 0.2 },
                    Transform::at([1.0, 1.0, 0.3]),
                )
            },
            solver: SolverParams {
                pressure_iterations: 160,
                buoyancy_density: 0.0,
                buoyancy_temperature: 1.0,
                ..SolverParams::default()
            },
            collider: None,
        }
    }

    /// `plume` with a static sphere collider of radius 0.25 m, 0.5 m above the
    /// emitter. Piece 2 spec §5.3 names the scene; the numbers are from the
    /// 2b-2 spec §6.
    pub fn plume_collider(resolution: u32) -> Self {
        Self {
            name: "plume_collider",
            collider: Some(ColliderParams {
                shape: Shape::Sphere { radius: 0.25 },
                transform: Transform::at([1.0, 1.0, 0.8]),
            }),
            ..Self::plume(resolution)
        }
    }

    /// `plume` with wind of 0.5 m/s² along +x. Piece 2 spec §5.3 names the
    /// scene; the numbers are from the 2b-2 spec §6.
    pub fn plume_wind(resolution: u32) -> Self {
        let mut scene = Self::plume(resolution);
        scene.name = "plume_wind";
        scene.solver.wind = [0.5, 0.0, 0.0];
        scene
    }

    pub fn with_iterations(mut self, n: u32) -> Self {
        self.solver.pressure_iterations = n;
        self
    }

    pub fn with_max_substeps(mut self, n: u32) -> Self {
        self.solver.max_substeps = n;
        self
    }

    /// True for each cell (x-fastest) whose centre lies inside the collider.
    /// The metrics use it for both solvers, so it is computed on the CPU from
    /// the scene rather than read from either solver. Empty when there is no
    /// collider.
    pub fn solid_mask(&self) -> Vec<bool> {
        let Some(collider) = &self.collider else {
            return Vec::new();
        };
        let keys = &collider.transform.keys;
        debug_assert_eq!(keys.len(), 1, "bench colliders are static");
        debug_assert!(keys[0].rotate.is_none(), "bench colliders are unrotated");
        let centre = keys[0].translate.map(f64::from);
        let [nx, ny, nz] = self.cells;
        let dx = self.domain_size / f64::from(nx.max(ny).max(nz));
        let mut mask = Vec::with_capacity((nx * ny * nz) as usize);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let d = [i, j, k]
                        .map(|n| (f64::from(n) + 0.5) * dx)
                        .iter()
                        .zip(centre)
                        .map(|(p, c)| p - c)
                        .collect::<Vec<_>>();
                    mask.push(match collider.shape {
                        Shape::Sphere { radius } => {
                            d.iter().map(|v| v * v).sum::<f64>() <= f64::from(radius).powi(2)
                        }
                        Shape::Box { half_extents } => d
                            .iter()
                            .zip(half_extents)
                            .all(|(v, h)| v.abs() <= f64::from(h)),
                    });
                }
            }
        }
        mask
    }

    /// The values `tests/bench/mantaflow_scene.py` builds the Mantaflow twin
    /// from (2b-3 spec §5). Ember's units throughout; `tests/bench/mapping.py`
    /// converts them. Bench emitters and colliders are static spheres.
    pub fn mantaflow_json(&self) -> serde_json::Value {
        let sphere = |shape: &Shape, transform: &Transform, what: &str| {
            let Shape::Sphere { radius } = *shape else {
                panic!("the Mantaflow twin supports only a sphere {what}");
            };
            debug_assert_eq!(transform.keys.len(), 1, "bench {what}s are static");
            (transform.keys[0].translate.map(decimal), decimal(radius))
        };
        let (center, radius) = sphere(&self.emitter.shape, &self.emitter.transform, "emitter");
        let collider = self.collider.as_ref().map(|c| {
            let (center, radius) = sphere(&c.shape, &c.transform, "collider");
            serde_json::json!({ "center": center, "radius": radius })
        });
        // Each face as `Face` serialises it, keyed by the Rust field name.
        let face = |f| serde_json::to_value(f).expect("a Face always serializes");
        let b = &self.solver.boundaries;
        let s = &self.solver;
        serde_json::json!({
            "name": self.name,
            "domain_size": self.domain_size,
            "resolution": self.cells.into_iter().max(),
            "fps": self.fps,
            "frames": self.frames,
            "substeps": s.max_substeps,
            "emitter": {
                "center": center,
                "radius": radius,
                "density_rate": decimal(self.emitter.density_rate),
                "temperature_rate": decimal(self.emitter.temperature_rate),
                "active_frames": self.emitter.active_frames,
            },
            "buoyancy_density": decimal(s.buoyancy_density),
            "buoyancy_temperature": decimal(s.buoyancy_temperature),
            "vorticity": decimal(s.vorticity),
            "density_dissipation": decimal(s.density_dissipation),
            "temperature_dissipation": decimal(s.temperature_dissipation),
            "wind": s.wind.map(decimal),
            "collider": collider,
            "boundaries": {
                "neg_x": face(b.neg_x),
                "pos_x": face(b.pos_x),
                "neg_y": face(b.neg_y),
                "pos_y": face(b.pos_y),
                "neg_z": face(b.neg_z),
                "pos_z": face(b.pos_z),
            },
        })
    }

    /// The scene as an `.elements` document: emitter → solver → output, plus
    /// the collider when there is one.
    pub fn document(&self) -> Document {
        let to_value = |v: serde_json::Result<serde_json::Value>| {
            v.expect("scene parameters are plain numbers and always serialize")
        };
        let edge = |from_node, from_index, to_node, to_index| DocEdge {
            from_node,
            from_index,
            to_node,
            to_index,
        };
        let mut nodes = vec![
            DocNode {
                id: 0,
                kind: shape_emitter::KIND.to_owned(),
                params: to_value(serde_json::to_value(&self.emitter)),
            },
            DocNode {
                id: 1,
                kind: solver::KIND.to_owned(),
                params: to_value(serde_json::to_value(self.solver)),
            },
            DocNode {
                id: 2,
                kind: "core.output".to_owned(),
                params: serde_json::json!({}),
            },
        ];
        let mut edges = vec![edge(0, 0, 1, 0), edge(0, 1, 1, 1), edge(1, 0, 2, 0)];
        if let Some(collider) = &self.collider {
            nodes.push(DocNode {
                id: 3,
                kind: collider::KIND.to_owned(),
                params: to_value(serde_json::to_value(collider)),
            });
            edges.extend([edge(3, 0, 1, 4), edge(3, 1, 1, 5)]);
        }
        Document {
            version: ELEMENTS_DOC_VERSION,
            dims: self.cells,
            fps: self.fps,
            start_frame: 1,
            cache_budget_mb: 0,
            domain_size: self.domain_size,
            nodes,
            edges,
            output: 2,
        }
    }
}

/// An `f32` parameter as the decimal it was written as: 0.2, not
/// 0.20000000298. Rust prints the shortest string that reads back as the same
/// `f32`, so the JSON carries the scene's own numbers.
fn decimal(x: f32) -> f64 {
    x.to_string()
        .parse()
        .expect("a finite f32 prints as a valid f64")
}

/// One row of the gate's table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateRow {
    pub iterations: u32,
    /// Median over three runs of each run's median step time.
    pub step_ms_median: f64,
    /// RMS divergence after projection over RMS divergence before it.
    pub ratio: f64,
}

impl GateRow {
    /// Spec §4.3's per-row pass rule: the step and ratio both within limits.
    /// `gate_verdict` uses this too, so the rule exists in exactly one place.
    pub fn passes(&self) -> bool {
        self.step_ms_median <= GATE_STEP_MS && self.ratio <= GATE_RATIO
    }
}

/// Spec §4.3: PASS with the largest N whose step takes at most
/// `GATE_STEP_MS` and whose ratio is at most `GATE_RATIO`; `None` is FAIL.
pub fn gate_verdict(rows: &[GateRow]) -> Option<u32> {
    rows.iter()
        .filter(|r| r.passes())
        .map(|r| r.iterations)
        .max()
}

/// A preview frame at 128³, with the CFL measurement and every substep, must
/// take at most this long (spec §6).
pub const PRESET_FRAME_MS: f64 = 100.0;

/// One row of the preview-preset sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PresetRow {
    pub max_substeps: u32,
    /// Median over three runs of each run's median frame time.
    pub frame_ms_median: f64,
}

impl PresetRow {
    /// Spec §6's per-row rule. `preview_substeps_verdict` uses it too.
    pub fn passes(&self) -> bool {
        self.frame_ms_median <= PRESET_FRAME_MS
    }
}

/// Spec §6: the largest `max_substeps` whose median full frame fits. `None`
/// means even one substep does not, and preview falls back to
/// semi-Lagrangian advection before the sweep runs again.
pub fn preview_substeps_verdict(rows: &[PresetRow]) -> Option<u32> {
    rows.iter()
        .filter(|r| r.passes())
        .map(|r| r.max_substeps)
        .max()
}
