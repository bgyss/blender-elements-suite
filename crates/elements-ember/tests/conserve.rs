//! The mass-conserving correction for scalar advection (2b-3c spec §5).

mod common;

use common::*;
use elements_core::gpu::{
    Axis, ComputeBatch, FieldDims, FieldPool, GpuContext, PipelineCache, ReduceTarget,
};
use elements_core::graph::{DocEdge, DocNode, Document, StateStore, Time};
use elements_ember::bench::{SOLVER_NODE, Scene};
use elements_ember::boundaries::Boundaries;
use elements_ember::kernels::conserve::{self, M0, M1, OUT, SLOTS};
use elements_ember::kernels::{Carried, StepConstants, Uniforms};
use elements_ember::solver;
use elements_ember::transform::Transform;
use elements_ember::unions::EMITTER_UNION_KIND;

const H: f32 = 1.0 / 24.0;
const DX: f32 = 0.125;

/// A positive field, so its sum is well away from zero.
fn positive(dims: FieldDims, seed: u32) -> Vec<f32> {
    pattern(dims, seed).iter().map(|v| v.abs() + 0.1).collect()
}

fn sum(values: &[f32]) -> f64 {
    values.iter().map(|&v| f64::from(v)).sum()
}

/// What `boundary_flux.wgsl` should remove, on the CPU: per cell beside an
/// open face, q · max(u_n, 0) · h / dx with u_n positive outward, and q in
/// a solid cell.
fn cpu_removed(
    q: &[f32],
    faces: &[Vec<f32>; 3],
    cells: FieldDims,
    open_mask: u32,
    solid: Option<&[f32]>,
    h: f32,
    dx: f32,
) -> f64 {
    let n = [cells.x, cells.y, cells.z];
    let mut total = 0.0f64;
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let c = [i, j, k];
                let at = index(cells, i, j, k);
                if solid.is_some_and(|s| s[at] > 0.5) {
                    total += f64::from(q[at]);
                    continue;
                }
                let mut speed = 0.0f64;
                for a in 0..3 {
                    let fd = face_dims(cells, a);
                    if c[a] == 0 && is_open(open_mask, a, 0) {
                        let u = faces[a][index(fd, c[0], c[1], c[2])];
                        speed += f64::from((-u).max(0.0));
                    }
                    if c[a] == n[a] - 1 && is_open(open_mask, a, 1) {
                        let mut f = c;
                        f[a] += 1;
                        let u = faces[a][index(fd, f[0], f[1], f[2])];
                        speed += f64::from(u.max(0.0));
                    }
                }
                total += f64::from(q[at]) * speed * f64::from(h) / f64::from(dx);
            }
        }
    }
    total
}

/// `measure_before`'s two slots against the CPU, with the open faces on
/// both sides of two axes and, optionally, a block of solid cells.
fn assert_measured(solids: bool) {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    // `block_mask`'s block reaches the top layer, so some solid cells lie
    // beside the open top and must count their contents, not a flux.
    let cells = FieldDims::new(8, 7, 5);
    // −x, +x and +z open; y and −z walls.
    let open_mask = 0b10_0011;
    let constants = StepConstants {
        open_mask,
        has_solids: solids,
        ..StepConstants::new(cells, H, DX)
    };
    let u = Uniforms::new(&gpu, &constants).unwrap();
    let q = positive(cells, 3);
    let faces = velocity_pattern(cells);
    // Both signs must occur on the open faces, or inflow would go untested.
    assert!(faces[0].iter().any(|&v| v > 0.0) && faces[0].iter().any(|&v| v < 0.0));
    let mask = solids.then(|| block_mask(cells));
    let field = upload(&gpu, &mut pool, cells, &q);
    let velocity = upload_staggered(&gpu, &mut pool, cells, &faces);
    let scratch = pool
        .acquire(&gpu, cells, elements_core::gpu::FieldFormat::R32Float)
        .unwrap();
    let mask_field = mask.as_ref().map(|m| upload(&gpu, &mut pool, cells, m));
    let target = ReduceTarget::new(&gpu, SLOTS).unwrap();
    let mut batch = ComputeBatch::new();
    conserve::measure_before(
        &gpu,
        &mut cache,
        &mut batch,
        &u,
        &field,
        &velocity,
        &scratch,
        &target,
        mask_field.as_ref(),
    )
    .unwrap();
    batch.submit(&gpu).unwrap();
    let slots = target.read(&gpu).unwrap();
    let expected = cpu_removed(&q, &faces, cells, open_mask, mask.as_deref(), H, DX);
    eprintln!(
        "solids {solids}: M0 {} (cpu {}), OUT {} (cpu {expected})",
        slots[M0 as usize],
        sum(&q),
        slots[OUT as usize]
    );
    assert!(expected > 0.1, "the case must remove something: {expected}");
    let rel = |gpu: f32, cpu: f64| (f64::from(gpu) - cpu).abs() / cpu;
    assert!(rel(slots[M0 as usize], sum(&q)) < 1e-5, "M0");
    assert!(rel(slots[OUT as usize], expected) < 1e-5, "OUT");
}

/// Spec §5 step 2: the outflow is the upwind value times the outward
/// normal velocity over each open face, times h / dx; inflow and walls
/// count nothing.
#[test]
fn the_boundary_flux_counts_upwind_outflow_over_open_faces() {
    assert_measured(false);
}

/// Advection zeroes scalars in solid cells (Mantaflow's `resetInObstacle`),
/// so their contents count as removed rather than being put back.
#[test]
fn solid_cells_count_as_removed() {
    assert_measured(true);
}

/// `correct_after` on `advected` after `measure_before` on `before`, in a
/// closed domain at rest (so OUT is 0), with the given density dissipation.
fn corrected(before: &[f32], advected: &[f32], dissipation: f32) -> Vec<f32> {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(4, 4, 4);
    let constants = StepConstants {
        open_mask: 0,
        density_dissipation: dissipation,
        ..StepConstants::new(cells, H, DX)
    };
    let u = Uniforms::new(&gpu, &constants).unwrap();
    let still = [0, 1, 2].map(|a| vec![0.0; face_dims(cells, a).voxel_count()]);
    let velocity = upload_staggered(&gpu, &mut pool, cells, &still);
    let src = upload(&gpu, &mut pool, cells, before);
    let dst = upload(&gpu, &mut pool, cells, advected);
    let scratch = pool
        .acquire(&gpu, cells, elements_core::gpu::FieldFormat::R32Float)
        .unwrap();
    let target = ReduceTarget::new(&gpu, SLOTS).unwrap();
    let mut batch = ComputeBatch::new();
    conserve::measure_before(
        &gpu, &mut cache, &mut batch, &u, &src, &velocity, &scratch, &target, None,
    )
    .unwrap();
    conserve::correct_after(
        &gpu,
        &mut cache,
        &mut batch,
        &u,
        Carried::Density,
        &dst,
        &target,
    )
    .unwrap();
    batch.submit(&gpu).unwrap();
    let slots = target.read(&gpu).unwrap();
    eprintln!(
        "M0 {} OUT {} M1 {}",
        slots[M0 as usize], slots[OUT as usize], slots[M1 as usize]
    );
    dst.read_back(&gpu).unwrap()
}

fn scaled(values: &[f32], s: f32) -> Vec<f32> {
    values.iter().map(|v| v * s).collect()
}

/// Spec §5's safeguard: the scale is clamped to [0.9, 1.1], and an empty
/// field is left alone.
#[test]
fn the_correction_never_scales_beyond_ten_percent() {
    let cells = FieldDims::new(4, 4, 4);
    let q = positive(cells, 5);
    // Doubled: s would be 0.5, and is 0.9.
    assert_close(
        &corrected(&q, &scaled(&q, 2.0), 0.0),
        &scaled(&q, 1.8),
        1e-5,
        "doubled",
    );
    // Halved: s would be 2, and is 1.1.
    assert_close(
        &corrected(&q, &scaled(&q, 0.5), 0.0),
        &scaled(&q, 0.55),
        1e-5,
        "halved",
    );
    // Inside the clamp the total is restored exactly.
    assert_close(
        &corrected(&q, &scaled(&q, 1.05), 0.0),
        &q,
        1e-5,
        "a 5% gain",
    );
    // Nothing before: nothing to restore, so the field is left as it is.
    let zero = vec![0.0; q.len()];
    assert_eq!(corrected(&zero, &q, 0.0), q, "an empty field before");
    // Next to nothing after (a sum of 6.4e-14, under 1e-12): left alone,
    // where the clamp would otherwise scale it by 1.1.
    let tiny = vec![1e-15; q.len()];
    assert_eq!(corrected(&q, &tiny, 0.0), tiny, "an empty field after");
}

/// A field with a negative cell before advection is left uncorrected: a
/// proportional rescale assumes q ≥ 0.
#[test]
fn a_field_with_a_negative_cell_is_not_rescaled() {
    let cells = FieldDims::new(4, 4, 4);
    let mut q = positive(cells, 9);
    q[17] = -0.05;
    let gained = scaled(&q, 1.05);
    assert_eq!(corrected(&q, &gained, 0.0), gained, "one negative cell");
}

/// The advection applies dissipation before the correction runs, so the
/// target must include it; otherwise the correction would undo it.
#[test]
fn the_correction_keeps_what_dissipation_removed() {
    let cells = FieldDims::new(4, 4, 4);
    let q = positive(cells, 7);
    let rate = 2.4;
    let decay = (-rate * H).exp();
    assert!(decay < 0.95, "the decay must be visible: {decay}");
    // A 3% advection gain on top of the decay.
    let got = corrected(&q, &scaled(&q, decay * 1.03), rate);
    assert_close(&got, &scaled(&q, decay), 1e-5, "decayed");
}

/// A 32³ plume emitting on frames 1–10 only, with `boundaries` and
/// `conserve_mass` set.
fn plume(boundaries: Boundaries, conserve_mass: bool) -> Scene {
    let mut scene = Scene::plume(32);
    scene.emitter.active_frames = Some([1, 10]);
    scene.solver.boundaries = boundaries;
    scene.solver.conserve_mass = conserve_mass;
    scene
}

/// `scene` run to frame `last`. `visit(frame, density, faces)` sees the
/// solver's state after every frame.
fn run(scene: &Scene, last: u32, mut visit: impl FnMut(u32, &[f32], &[Vec<f32>; 3])) {
    run_doc(scene.document(), last, |frame, density, _, faces| {
        visit(frame, density, faces)
    });
}

/// `doc` run to frame `last`; `visit(frame, density, temperature, faces)`
/// sees the solver's state after every frame.
fn run_doc(doc: Document, last: u32, mut visit: impl FnMut(u32, &[f32], &[f32], &[Vec<f32>; 3])) {
    let gpu: GpuContext = gpu();
    let registry = elements_ember::registry();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(&registry).unwrap();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    for frame in config.start_frame..=last {
        let time = Time::at(frame, config.start_frame, config.fps);
        let evaluated = graph
            .eval_frame(&gpu, &mut pool, &mut pipelines, &mut state, time, dims)
            .unwrap();
        evaluated.value.release_to(&mut pool);
        let read = |slot| {
            state
                .get(SOLVER_NODE, slot)
                .unwrap()
                .as_field()
                .unwrap()
                .read_back(&gpu)
                .unwrap()
        };
        let density = read(solver::DENSITY);
        let temperature = read(solver::TEMPERATURE);
        let velocity = state
            .get(SOLVER_NODE, solver::VELOCITY)
            .unwrap()
            .as_vector_field()
            .unwrap();
        let faces = [Axis::X, Axis::Y, Axis::Z].map(|a| velocity.face(a).read_back(&gpu).unwrap());
        visit(frame, &density, &temperature, &faces);
    }
    state.clear(&mut pool);
}

/// Total density at frames 11 and 60 of the closed-box plume.
fn closed_box_totals(conserve_mass: bool) -> (f64, f64) {
    let (mut at_11, mut at_60) = (0.0, 0.0);
    let scene = plume(Boundaries::closed(), conserve_mass);
    run(&scene, 60, |frame, density, _| match frame {
        11 => at_11 = sum(density),
        60 => at_60 = sum(density),
        _ => {}
    });
    (at_11, at_60)
}

/// Spec §7: in a closed box with no sources, total mass is constant to
/// rounding with `conserve_mass`, and drifts measurably without it.
#[test]
fn a_closed_box_keeps_its_mass_to_rounding() {
    let (on_11, on_60) = closed_box_totals(true);
    let (off_11, off_60) = closed_box_totals(false);
    let on = (on_60 - on_11) / on_11;
    let off = (off_60 - off_11) / off_11;
    eprintln!(
        "closed 32³, frames 11→60: with the correction {on_11} → {on_60} ({on:e}); \
         without {off_11} → {off_60} ({off:e})"
    );
    assert!(on_11 > 1.0, "the plume must exist: {on_11}");
    assert!(on.abs() <= 1e-5, "with the correction: {on:e}");
    assert!(off.abs() > 1e-3, "without it: {off:e}");
}

/// Σ density and the cumulative first-order outflow, frame by frame from
/// frame 11, for an open-top plume that starts high and rises fast, so that
/// much of it leaves by frame 60. With one substep a frame, frame n's
/// advection starts from frame n − 1's density (the emitter is off) and
/// uses frame n's projected velocity, which is what the state holds after
/// frame n.
fn open_top_accounting(conserve_mass: bool, last: u32) -> Vec<(u32, f64, f64)> {
    let mut rows = Vec::new();
    let mut previous: Option<Vec<f32>> = None;
    let mut outflow = 0.0;
    let boundaries = Boundaries::default();
    let open_mask = boundaries.open_mask();
    let mut scene = plume(boundaries, conserve_mass);
    scene.emitter.transform = Transform::at([1.0, 1.0, 1.3]);
    scene.solver.buoyancy_temperature = 4.0;
    assert_eq!(scene.solver.max_substeps, 1, "one substep a frame");
    assert_eq!(scene.fps, 24.0, "h is 1/24 s");
    let cells = FieldDims::new(32, 32, 32);
    let dx = (2.0 / 32.0) as f32;
    run(&scene, last, |frame, density, faces| {
        if frame > 11 {
            let before = previous.as_deref().unwrap();
            outflow += cpu_removed(before, faces, cells, open_mask, None, H, dx);
        }
        if frame >= 11 {
            rows.push((frame, sum(density), outflow));
            previous = Some(density.to_vec());
        }
    });
    rows
}

/// Spec §7: with an open top, mass plus the cumulative outflow through it
/// stays what it was when emission stopped.
///
/// This depends on `open_top_accounting`'s hand-tuned fast plume (emitter
/// at z = 1.3 m, `buoyancy_temperature` 4): the default plume emitting for
/// ten frames never reaches the top by frame 120 at 32³, so the outflow
/// term would go untested. The final assertion guards that.
#[test]
fn open_top_mass_plus_outflow_is_constant() {
    let rows = open_top_accounting(true, 60);
    let m11 = rows[0].1;
    let mut worst = 0.0f64;
    for &(frame, mass, outflow) in &rows {
        let rel = (mass + outflow - m11) / m11;
        if frame % 10 == 0 {
            eprintln!("frame {frame}: mass {mass} + outflow {outflow} vs {m11}: {rel:e}");
        }
        worst = worst.max(rel.abs());
    }
    let (_, last_mass, last_outflow) = rows[rows.len() - 1];
    eprintln!(
        "worst {worst:e}; outflow at 60 is {} of m11",
        last_outflow / m11
    );
    // The accounting must be exercised: a real share of the smoke leaves.
    assert!(
        last_outflow > 0.01 * m11,
        "outflow {last_outflow}, mass {last_mass}"
    );
    assert!(worst <= 1e-3, "mass + outflow drifted {worst:e}");
}

/// The closed-box plume with a second, cold emitter above the hot one
/// (negative `temperature_rate`), both on frames 1–10 and merged by an
/// `ember.emitter_union`, so the temperature field holds both signs.
fn hot_and_cold(conserve_mass: bool) -> Document {
    let scene = plume(Boundaries::closed(), conserve_mass);
    let mut doc = scene.document();
    let mut cold = scene.emitter.clone();
    cold.transform = Transform::at([1.0, 1.0, 1.4]);
    // Half as strong, so ΣT stays positive and the scale is not skipped
    // for want of a positive target.
    cold.temperature_rate = -0.5;
    let (cold_id, union_id) = (3, 4);
    doc.nodes.push(DocNode {
        id: cold_id,
        kind: doc.nodes[0].kind.clone(),
        params: serde_json::to_value(&cold).unwrap(),
    });
    doc.nodes.push(DocNode {
        id: union_id,
        kind: EMITTER_UNION_KIND.to_owned(),
        params: serde_json::json!({}),
    });
    let edge = |from_node, from_index, to_node, to_index| DocEdge {
        from_node,
        from_index,
        to_node,
        to_index,
    };
    // The hot emitter's edges into the solver now go through the union.
    doc.edges.retain(|e| e.from_node != 0);
    for i in 0..4 {
        doc.edges.push(edge(0, i, union_id, i));
        doc.edges.push(edge(cold_id, i, union_id, 4 + i));
    }
    doc.edges.push(edge(union_id, 0, 1, 0));
    doc.edges.push(edge(union_id, 1, 1, 1));
    doc
}

/// (Σ density at frame 11, at frame 60, the largest |T| at frame 60, the
/// smallest and largest T at frame 10) for `hot_and_cold`.
fn hot_and_cold_run(conserve_mass: bool) -> (f64, f64, f32, f32, f32) {
    let (mut at_11, mut at_60, mut max_t, mut lo, mut hi) = (0.0, 0.0, 0.0f32, 0.0f32, 0.0f32);
    run_doc(
        hot_and_cold(conserve_mass),
        60,
        |frame, density, temperature, _| match frame {
            60 => {
                at_60 = sum(density);
                max_t = temperature.iter().fold(0.0, |m: f32, t| m.max(t.abs()));
            }
            10 => {
                lo = temperature.iter().copied().fold(f32::MAX, f32::min);
                hi = temperature.iter().copied().fold(f32::MIN, f32::max);
            }
            11 => at_11 = sum(density),
            _ => {}
        },
    );
    (at_11, at_60, max_t, lo, hi)
}

/// Emitters accept negative rates, so temperature can mix hot and cold. A
/// proportional rescale assumes q ≥ 0: with both signs, Σq is far below
/// Σ|q| and the scale would sit on its clamp, compounding ×0.9 or ×1.1
/// every substep. Such a field is left uncorrected. Density, all
/// nonnegative, is still conserved.
#[test]
fn a_field_with_both_signs_is_left_uncorrected() {
    let (on_11, on_60, on_max, lo, hi) = hot_and_cold_run(true);
    let (_, _, off_max, _, _) = hot_and_cold_run(false);
    let drift = (on_60 - on_11) / on_11;
    eprintln!(
        "hot and cold, closed 32³: T at frame 10 in [{lo}, {hi}]; max |T| at frame 60 \
         {on_max} with the correction, {off_max} without; density drift {drift:e}"
    );
    assert!(
        lo < -0.01 && hi > 0.01,
        "both signs must occur: [{lo}, {hi}]"
    );
    assert!(
        (on_max / off_max - 1.0).abs() <= 0.01,
        "max |T| {on_max} with the correction, {off_max} without"
    );
    assert!(drift.abs() <= 1e-5, "density drift {drift:e}");
}
