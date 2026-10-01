//! The flamethrower shot's jet into the shack at a test size (flamethrower
//! plan Task 10, FT3b): generates `examples/jet_shack.elements` and measures
//! fire divergence with the shack in the way (`docs/bench/fire-divergence.md`).
//!
//! The scene is the FT0 shot spec (`docs/superpowers/specs/2026-09-30-
//! flamethrower-shot-ft0.md`) at half size: dims [128, 64, 64] over 2.0 m
//! instead of [256, 128, 128] over 4.0 m, so dx stays 15.625 mm and every
//! position and length is halved. The jet's speed, blend and the fire
//! settings are unchanged.

mod common;

use common::*;
use elements_core::gpu::{Axis, FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_core::graph::{DocEdge, DocNode, Document, ELEMENTS_DOC_VERSION, StateStore, Time};
use elements_ember::bench::{FIRE_FUEL_RATE, SOLVER_NODE, Scene, report::load_average};
use elements_ember::collider::{self, ColliderFields, ColliderParams};
use elements_ember::mesh_collider::{self, MeshColliderParams, fill_mesh_collider};
use elements_ember::metrics::{
    FLAME_THRESHOLD, SMOKE_THRESHOLD, Sample, divergence, fire_metrics, measure,
};
use elements_ember::shack::{ShackParams, shack};
use elements_ember::solver::{self, DENSITY, FUEL, REACT, VELOCITY};
use elements_ember::transform::{Key, Shape, Transform};

const DIMS: [u32; 3] = [128, 64, 64];
const DOMAIN: f64 = 2.0;
const DX: f64 = DOMAIN / 128.0;
const FRAMES: u32 = 60;
/// The shot's pulse, frames 10–150 of 240, compressed into a 60-frame test.
const ACTIVE: [u32; 2] = [5, 40];
/// The shack's origin (footprint centre, on the floor) and size: the shot's
/// (2.9, 1.0, 0.0) and [1.2, 1.0, 1.0], halved.
const SHACK_AT: [f32; 3] = [1.45, 0.5, 0.0];
const SHACK_SIZE: [f32; 3] = [0.6, 0.5, 0.5];
const PLANK_THICKNESS: f32 = 0.01;
/// The shack's outer face towards the nozzle, metres along x.
const SHACK_FRONT_X: f64 = 1.45 - 0.3;
const SHACK_BACK_X: f64 = 1.45 + 0.3;
const EXAMPLE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/jet_shack.elements"
);

fn shack_collider() -> MeshColliderParams {
    MeshColliderParams {
        mesh: shack(&ShackParams {
            size: SHACK_SIZE,
            plank_height: 0.05,
            thickness: PLANK_THICKNESS,
            gap: 0.005,
            broken_fraction: 0.2,
            seed: 7,
        }),
        transform: Transform {
            keys: vec![Key {
                frame: 0.0,
                translate: SHACK_AT,
                rotate: None,
            }],
        },
        // Planks are thinner than a voxel (10 mm against 15.6 mm), so the
        // spec inflates them by half a voxel.
        offset: (0.5 * DX) as f32,
        surface_fuel: None,
    }
}

/// What stands in the jet's way.
#[derive(Clone, Copy, PartialEq)]
enum Obstacle {
    /// The plank shack, through `ember.mesh_collider`.
    Shack,
    /// A solid `ember.collider` box with the shack's outer extent: the same
    /// obstacle without thin walls.
    SolidBox,
    /// Nothing: the control.
    None,
}

/// One baked variant: preview except for the named overrides.
#[derive(Clone, Copy)]
struct Variant {
    label: &'static str,
    pressure_cycles: Option<u32>,
    max_substeps: Option<u32>,
    obstacle: Obstacle,
}

const BASELINE: Variant = Variant {
    label: "shack, preview (x4)",
    pressure_cycles: None,
    max_substeps: None,
    obstacle: Obstacle::Shack,
};

/// The solid box's centre and half extents, matching the shack's outside.
const BOX_AT: [f32; 3] = [1.45, 0.5, 0.25];
const BOX_HALF: [f32; 3] = [0.3, 0.25, 0.25];

/// The jet-into-shack document for `v`.
fn document(v: Variant) -> Document {
    let edge = |from_node, from_index, to_node, to_index| DocEdge {
        from_node,
        from_index,
        to_node,
        to_index,
    };
    // The nozzle: a cone along +x whose local-frame velocity is the jet.
    let emitter = serde_json::json!({
        "shape": { "cone": { "length": 0.15, "radius_start": 0.015, "radius_end": 0.03 } },
        "transform": { "keys": [{ "frame": 0.0, "translate": [0.2, 0.5, 0.3] }] },
        "fuel_rate": FIRE_FUEL_RATE as f64,
        "temperature_rate": 1.0,
        "velocity": [14.0, 0.0, 0.0],
        "velocity_blend": 20.0,
        "velocity_local": true,
        "noise": { "seed": 7, "scale_m": 0.04, "amplitude": 0.6 },
        "active_frames": ACTIVE,
    });
    // The fire benchmark's solver settings: preview, buoyancy from heat only.
    let mut solver_params =
        serde_json::to_value(Scene::fire(64).solver).expect("solver params serialize");
    if let Some(n) = v.pressure_cycles {
        solver_params["pressure_cycles"] = n.into();
    }
    if let Some(n) = v.max_substeps {
        solver_params["max_substeps"] = n.into();
    }
    let mut nodes = vec![
        DocNode {
            id: 0,
            kind: "ember.emitter".to_owned(),
            params: emitter,
        },
        DocNode {
            id: 1,
            kind: solver::KIND.to_owned(),
            params: solver_params,
        },
        DocNode {
            id: 2,
            kind: "core.output".to_owned(),
            params: serde_json::json!({}),
        },
    ];
    // Density, temperature, weight, velocity, fuel into solver 0–3 and 6.
    let mut edges = vec![
        edge(0, 0, 1, 0),
        edge(0, 1, 1, 1),
        edge(0, 2, 1, 2),
        edge(0, 3, 1, 3),
        edge(0, 4, 1, 6),
        edge(1, 0, 2, 0),
    ];
    let collider = match v.obstacle {
        Obstacle::Shack => Some(DocNode {
            id: 3,
            kind: mesh_collider::KIND.to_owned(),
            params: serde_json::to_value(shack_collider()).expect("mesh params serialize"),
        }),
        Obstacle::SolidBox => Some(DocNode {
            id: 3,
            kind: collider::KIND.to_owned(),
            params: serde_json::to_value(ColliderParams {
                shape: Shape::Box {
                    half_extents: BOX_HALF,
                },
                transform: Transform::at(BOX_AT),
                surface_fuel: None,
            })
            .expect("collider params serialize"),
        }),
        Obstacle::None => None,
    };
    if let Some(node) = collider {
        nodes.push(node);
        edges.extend([edge(3, 0, 1, 4), edge(3, 1, 1, 5)]);
    }
    Document {
        version: ELEMENTS_DOC_VERSION,
        dims: DIMS,
        fps: 24.0,
        start_frame: 1,
        cache_budget_mb: 0,
        domain_size: DOMAIN,
        nodes,
        edges,
        output: 2,
        outputs: Vec::new(),
    }
}

/// The committed example is exactly the generated baseline scene, and it
/// loads into a graph. Regenerate it with `write_jet_shack_example`.
#[test]
fn the_jet_shack_example_is_the_generated_scene() {
    let text = std::fs::read_to_string(EXAMPLE).expect("examples/jet_shack.elements exists");
    let generated = document(BASELINE).to_json().unwrap() + "\n";
    assert!(
        text == generated,
        "examples/jet_shack.elements is stale; run write_jet_shack_example"
    );
    Document::from_json(&text)
        .unwrap()
        .into_graph(&elements_ember::registry())
        .unwrap();
}

/// Writes `examples/jet_shack.elements`. Run by hand:
/// `cargo nextest run -p elements-ember --run-ignored ignored-only -E 'test(write_jet_shack_example)'`.
#[test]
#[ignore = "writes examples/jet_shack.elements"]
fn write_jet_shack_example() {
    let text = document(BASELINE).to_json().unwrap();
    std::fs::write(EXAMPLE, text + "\n").unwrap();
}

/// The shack's solid mask, as the solver computes it: the mesh collider's
/// SDF below zero at each cell centre (`solidify.wgsl`).
fn shack_mask(cells: FieldDims) -> Vec<bool> {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let params = shack_collider();
    let sdf = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let v = pool.acquire_staggered_uninit(&gpu, cells).unwrap();
    let pose = params.transform.pose(0.0, 1.0 / 24.0);
    fill_mesh_collider(
        &gpu,
        &mut cache,
        &params,
        &pose,
        DX as f32,
        ColliderFields {
            sdf: &sdf,
            velocity: &v,
        },
    )
    .unwrap();
    sdf.read_back(&gpu)
        .unwrap()
        .iter()
        .map(|&d| d < 0.0)
        .collect()
}

fn centre(n: u32) -> f64 {
    (f64::from(n) + 0.5) * DX
}

#[derive(Default)]
struct Reach {
    /// Σ density dV in cells at or past the shack's front plane.
    smoke_past_front: f64,
    /// Σ density dV in cells past the shack's back plane.
    smoke_past_back: f64,
    /// Flame volume (m³) at or past the front plane.
    flame_past_front: f64,
    /// Fluid cells beside a shack cell holding smoke, fuel or flame.
    contact_cells: u64,
}

fn reach(cells: FieldDims, mask: &[bool], density: &[f32], fuel: &[f32], flame: &[f32]) -> Reach {
    let dv = DX * DX * DX;
    let mut r = Reach::default();
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let n = index(cells, i, j, k);
                let x = centre(i);
                if x >= SHACK_FRONT_X {
                    r.smoke_past_front += f64::from(density[n]) * dv;
                    if flame[n] > FLAME_THRESHOLD {
                        r.flame_past_front += dv;
                    }
                }
                if x >= SHACK_BACK_X {
                    r.smoke_past_back += f64::from(density[n]) * dv;
                }
                if mask[n] {
                    continue;
                }
                let lit = density[n] > SMOKE_THRESHOLD
                    || fuel[n] > SMOKE_THRESHOLD
                    || flame[n] > FLAME_THRESHOLD;
                if !lit {
                    continue;
                }
                let touches = [
                    (1i32, 0i32, 0i32),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ]
                .iter()
                .any(|&(a, b, c)| {
                    let (x, y, z) = (i as i32 + a, j as i32 + b, k as i32 + c);
                    x >= 0
                        && y >= 0
                        && z >= 0
                        && (x as u32) < cells.x
                        && (y as u32) < cells.y
                        && (z as u32) < cells.z
                        && mask[index(cells, x as u32, y as u32, z as u32)]
                });
                if touches {
                    r.contact_cells += 1;
                }
            }
        }
    }
    r
}

/// Largest |u| on faces with a solid cell on both sides: zero when the
/// solver holds the static shack's interior still.
fn inside_solid_speed(cells: FieldDims, mask: &[bool], faces: &[Vec<f32>; 3]) -> f32 {
    let mut worst = 0.0f32;
    for (a, axis) in [Axis::X, Axis::Y, Axis::Z].into_iter().enumerate() {
        let fd = elements_core::gpu::StaggeredField::face_dims(cells, axis);
        for k in 0..cells.z {
            for j in 0..cells.y {
                for i in 0..cells.x {
                    let mut hi = [i, j, k];
                    hi[a] += 1;
                    if hi[a] >= [cells.x, cells.y, cells.z][a] {
                        continue;
                    }
                    if mask[index(cells, i, j, k)] && mask[index(cells, hi[0], hi[1], hi[2])] {
                        let u = faces[a][(hi[0] + fd.x * (hi[1] + fd.y * hi[2])) as usize];
                        worst = worst.max(u.abs());
                    }
                }
            }
        }
    }
    worst
}

/// `mask` grown by `r` cells in every direction (a cube neighbourhood).
fn dilate(mask: &[bool], cells: FieldDims, r: u32) -> Vec<bool> {
    let n = [cells.x, cells.y, cells.z];
    let mut out = mask.to_vec();
    for axis in 0..3 {
        let src = out.clone();
        for k in 0..cells.z {
            for j in 0..cells.y {
                for i in 0..cells.x {
                    let p = [i, j, k];
                    let lo = p[axis].saturating_sub(r);
                    let hi = (p[axis] + r).min(n[axis] - 1);
                    out[index(cells, i, j, k)] = (lo..=hi).any(|q| {
                        let mut s = p;
                        s[axis] = q;
                        src[index(cells, s[0], s[1], s[2])]
                    });
                }
            }
        }
    }
    out
}

/// Cells more than this many cells from the shack are its far field.
const FAR: u32 = 4;
/// Frames with a printed snapshot: mid-pulse, the pulse's last frame, and
/// frame 60, the frame Task 9 compares.
const SNAPSHOTS: [u32; 3] = [30, 40, 60];

/// One frame's fields, read back.
struct Frame {
    density: Vec<f32>,
    fuel: Vec<f32>,
    flame: Vec<f32>,
    faces: [Vec<f32>; 3],
}

fn snapshot(frame: u32, cells: FieldDims, solid: &[bool], far: &[bool], mask: &[bool], f: &Frame) {
    let open_mask = Scene::fire(64).solver.boundaries.open_mask();
    let sample = |solid| Sample {
        cells,
        dx: DX,
        density: &f.density,
        faces: &f.faces,
        solid,
        open_mask,
    };
    let m = measure(&sample(solid));
    let away = measure(&sample(far));
    let full = divergence(&f.faces, cells, DX as f32);
    let (fuel_mass, flame_volume) = fire_metrics(&f.fuel, &f.flame, DX);
    let r = reach(cells, mask, &f.density, &f.fuel, &f.flame);
    println!(
        "f{frame}: div RMS {:.3e} over {} measured cells (max {:.3e}); far field (> {FAR} cells \
         from the shack) {:.3e} over {} cells; all cells {:.3e}",
        m.divergence_rms,
        m.measured_cells,
        m.divergence_max,
        away.divergence_rms,
        away.measured_cells,
        full.rms
    );
    println!(
        "f{frame}: fuel {fuel_mass:.5}, flame {flame_volume:.5} m3, smoke {:.5}; smoke past \
         front {:.5}, past back {:.5}, flame past front {:.5} m3, contact cells {}",
        m.mass, r.smoke_past_front, r.smoke_past_back, r.flame_past_front, r.contact_cells
    );
    if frame == 60 {
        println!(
            "f60: max |u| on faces inside the shack {:.3e} m/s",
            inside_solid_speed(cells, mask, &f.faces)
        );
    }
}

/// The solid box's cells: centres inside it (no centre lies on its faces).
fn box_mask(cells: FieldDims) -> Vec<bool> {
    let mut out = Vec::with_capacity(cells.voxel_count());
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let p = [centre(i), centre(j), centre(k)];
                out.push(
                    (0..3).all(|a| (p[a] - f64::from(BOX_AT[a])).abs() < f64::from(BOX_HALF[a])),
                );
            }
        }
    }
    out
}

/// Bakes `v` and prints its measurements. Reach, contact and the far field
/// are taken against the obstacle's cells, or the shack's for the control.
fn run(v: Variant, shack_mask: &[bool]) {
    let gpu = gpu();
    let doc = document(v);
    let config = doc.timeline_config();
    let (graph, cells) = doc.into_graph(&elements_ember::registry()).unwrap();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let boxed = box_mask(cells);
    let mask: &[bool] = if v.obstacle == Obstacle::SolidBox {
        &boxed
    } else {
        shack_mask
    };
    let empty: Vec<bool> = Vec::new();
    let solid: &[bool] = if v.obstacle == Obstacle::None {
        &empty
    } else {
        mask
    };
    let far = dilate(mask, cells, FAR);
    println!("== {} ==", v.label);
    println!(
        "obstacle cells {}; far field excludes {}",
        mask.iter().filter(|&&s| s).count(),
        far.iter().filter(|&&s| s).count()
    );
    let load_before = load_average();
    let mut peak = (0.0f32, 0u32);
    let mut first_contact = None;
    let mut first_past_front = None;
    let mut flame_front = (0.0f64, 0u32);
    let mut clamped = 0;
    let mut frame_ms = Vec::new();
    for frame in config.start_frame..config.start_frame + FRAMES {
        let time = Time::at(frame, config.start_frame, config.fps);
        let start = std::time::Instant::now();
        let evaluated =
            match graph.eval_frame(&gpu, &mut pool, &mut pipelines, &mut state, time, cells) {
                Ok(e) => e,
                Err(e) => {
                    println!("ERROR at frame {frame}: {e}");
                    break;
                }
            };
        gpu.wait().unwrap();
        frame_ms.push(start.elapsed().as_secs_f64() * 1e3);
        clamped += evaluated.stats.cfl_clamped;
        evaluated.value.release_to(&mut pool);
        let read = |slot: &'static str| {
            state
                .get(SOLVER_NODE, slot)
                .unwrap()
                .as_field()
                .unwrap()
                .read_back(&gpu)
                .unwrap()
        };
        let v = state.get(SOLVER_NODE, VELOCITY).unwrap();
        let f = Frame {
            density: read(DENSITY),
            fuel: read(FUEL),
            flame: read(REACT)
                .iter()
                .map(|r| r.clamp(0.0, 1.0).sqrt())
                .collect(),
            faces: read_staggered(&gpu, v.as_vector_field().unwrap()),
        };
        if f.faces.iter().flatten().any(|u| !u.is_finite())
            || f.density.iter().any(|d| !d.is_finite())
        {
            println!("ERROR at frame {frame}: non-finite velocity or density");
            break;
        }
        let max_u = f
            .faces
            .iter()
            .flatten()
            .fold(0.0f32, |m, &u| m.max(u.abs()));
        if max_u > peak.0 {
            peak = (max_u, frame);
        }
        let r = reach(cells, mask, &f.density, &f.fuel, &f.flame);
        if r.contact_cells > 0 && first_contact.is_none() {
            first_contact = Some(frame);
        }
        if r.smoke_past_front > 0.0 && first_past_front.is_none() {
            first_past_front = Some(frame);
        }
        if r.flame_past_front > flame_front.0 {
            flame_front = (r.flame_past_front, frame);
        }
        if SNAPSHOTS.contains(&frame) {
            snapshot(frame, cells, solid, &far, mask, &f);
        }
    }
    let load_after = load_average();
    let mut sorted = frame_ms.clone();
    sorted.sort_by(f64::total_cmp);
    println!("frames run: {}", frame_ms.len());
    println!(
        "peak |u| frames 1-60: {:.2} m/s at frame {}",
        peak.0, peak.1
    );
    println!("CFL-clamped frames: {clamped}");
    println!(
        "first frame with smoke past the front plane: {first_past_front:?}; first contact: \
         {first_contact:?}; most flame past the front plane: {:.5} m3 at frame {}",
        flame_front.0, flame_front.1
    );
    println!(
        "frame ms (eval + wait, loaded upper bound): median {:.1}, min {:.1}, max {:.1}",
        sorted[sorted.len() / 2],
        sorted[0],
        sorted[sorted.len() - 1]
    );
    println!("load (1-min) before {load_before:.2} after {load_after:.2}");
    state.clear(&mut pool);
}

/// Bakes the jet into the shack for 60 frames under preview (MGPCG ×4) and
/// ×6 cycles, plus the same two without the shack as a control, and prints
/// divergence, blow-up and reach. ×10 with the shack is a diagnostic: does
/// the pressure solve converge around the planks with more cycles? Run by
/// hand (`JET_SHACK_ONLY=<label part>` runs a subset):
/// `cargo nextest run -p elements-ember --release --run-ignored ignored-only -E 'test(jet_shack_divergence)' --no-capture`
#[test]
#[ignore = "a GPU measurement, a minute or more; prints, asserts nothing"]
fn jet_shack_divergence() {
    let cells = FieldDims::new(DIMS[0], DIMS[1], DIMS[2]);
    let mask = shack_mask(cells);
    let solid_cells = mask.iter().filter(|&&s| s).count();
    println!("adapter: {}", gpu().adapter_name());
    println!(
        "shack: {} triangles, {solid_cells} solid cells (sdf < 0 with offset {:.5} m)",
        shack_collider().mesh.triangle_count(),
        0.5 * DX
    );
    let only = std::env::var("JET_SHACK_ONLY").ok();
    let v = |label, pressure_cycles, max_substeps, obstacle| Variant {
        label,
        pressure_cycles,
        max_substeps,
        obstacle,
    };
    for variant in [
        BASELINE,
        v("shack, pressure x6", Some(6), None, Obstacle::Shack),
        v(
            "control (no shack), preview (x4)",
            None,
            None,
            Obstacle::None,
        ),
        v(
            "control (no shack), pressure x6",
            Some(6),
            None,
            Obstacle::None,
        ),
        v(
            "diagnostic: shack, pressure x10",
            Some(10),
            None,
            Obstacle::Shack,
        ),
        v(
            "diagnostic: solid box, preview (x4)",
            None,
            None,
            Obstacle::SolidBox,
        ),
        v(
            "diagnostic: solid box, pressure x6",
            Some(6),
            None,
            Obstacle::SolidBox,
        ),
        v(
            "diagnostic: shack, 8 substeps (x4)",
            None,
            Some(8),
            Obstacle::Shack,
        ),
        v(
            "diagnostic: control, 8 substeps (x4)",
            None,
            Some(8),
            Obstacle::None,
        ),
        // `final`'s substeps and cycles, the rest preview.
        v(
            "diagnostic: shack, 8 substeps x10",
            Some(10),
            Some(8),
            Obstacle::Shack,
        ),
        v(
            "diagnostic: control, 8 substeps x10",
            Some(10),
            Some(8),
            Obstacle::None,
        ),
    ] {
        if only.as_deref().is_some_and(|o| !variant.label.contains(o)) {
            continue;
        }
        run(variant, &mask);
    }
}
