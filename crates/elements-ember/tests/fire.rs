//! Fire at the kernel and substep level (2b-4 spec §3, §5).

mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldPool, PipelineCache};
use elements_ember::kernels::StepConstants;
use elements_ember::solver::{PressureSolve, SolverState, Sources, Substep, substep};

const CELLS: FieldDims = FieldDims { x: 8, y: 6, z: 5 };
const SOLVE: PressureSolve = PressureSolve::GaussSeidel(40);

/// Fire on, burning off: fuel only moves.
fn transport_only() -> StepConstants {
    StepConstants {
        fire: true,
        ..StepConstants::new(CELLS, 0.25, 0.125)
    }
}

fn abs_pattern(seed: u32) -> Vec<f32> {
    pattern(CELLS, seed).iter().map(|v| v.abs()).collect()
}

#[test]
fn fuel_is_emitted_into_fuel_and_react_at_the_rate() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let zero = vec![0.0; CELLS.voxel_count()];
    let rate = abs_pattern(3);
    let density_src = upload(&gpu, &mut pool, CELLS, &zero);
    let temperature_src = upload(&gpu, &mut pool, CELLS, &zero);
    let fuel_src = upload(&gpu, &mut pool, CELLS, &rate);
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, CELLS).unwrap();
    state.add_fire(&gpu, &mut cache, &mut pool).unwrap();
    let sources = Sources::new(&density_src, &temperature_src).with_fuel(&fuel_src);
    // Nothing moves: no buoyancy, still air, so advection returns its input.
    substep(
        &gpu,
        &mut cache,
        &mut pool,
        &mut state,
        sources,
        &transport_only(),
        SOLVE,
    )
    .unwrap();
    let want: Vec<f32> = rate.iter().map(|r| r * 0.25).collect();
    // From empty, all fuel is fresh: react blends fully to 1 wherever fuel
    // arrived (spec §3.2 step 1), and stays 0 elsewhere.
    let want_react: Vec<f32> = want
        .iter()
        .map(|&f| if f > 1e-6 { 1.0 } else { 0.0 })
        .collect();
    let fire = state.fire.as_ref().unwrap();
    assert_close(&fire.fuel.read_back(&gpu).unwrap(), &want, 1e-6, "fuel");
    assert_close(
        &fire.react.read_back(&gpu).unwrap(),
        &want_react,
        1e-6,
        "react",
    );
}

#[test]
fn fresh_fuel_blends_react_towards_one_by_its_share() {
    use elements_core::gpu::ComputeBatch;
    use elements_ember::kernels::{Uniforms, emit_fuel};
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let f0: Vec<f32> = pattern(CELLS, 21)
        .iter()
        .map(|v| v.max(0.0) * 3.0)
        .collect();
    let r0 = abs_pattern(22);
    let rate = abs_pattern(23);
    let fuel = upload(&gpu, &mut pool, CELLS, &f0);
    let react = upload(&gpu, &mut pool, CELLS, &r0);
    let src = upload(&gpu, &mut pool, CELLS, &rate);
    let u = Uniforms::new(&gpu, &transport_only()).unwrap();
    let mut batch = ComputeBatch::new();
    emit_fuel(&gpu, &mut cache, &mut batch, &u, &fuel, &react, &src).unwrap();
    batch.submit(&gpu).unwrap();
    let mut want_fuel = Vec::new();
    let mut want_react = Vec::new();
    for n in 0..f0.len() {
        let d = rate[n] * 0.25;
        let f1 = f0[n] + d;
        want_fuel.push(f1);
        want_react.push(if f1 > 1e-6 {
            (r0[n] + d / f1 * (1.0 - r0[n])).clamp(0.0, 1.0)
        } else {
            r0[n]
        });
    }
    assert_close(&fuel.read_back(&gpu).unwrap(), &want_fuel, 1e-6, "fuel");
    assert_close(&react.read_back(&gpu).unwrap(), &want_react, 1e-6, "react");
}

#[test]
fn fuel_and_react_are_advected_as_density_is() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let zero = vec![0.0; CELLS.voxel_count()];
    let rate = abs_pattern(5);
    let density_src = upload(&gpu, &mut pool, CELLS, &rate);
    let temperature_src = upload(&gpu, &mut pool, CELLS, &zero);
    let fuel_src = upload(&gpu, &mut pool, CELLS, &rate);
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, CELLS).unwrap();
    state.add_fire(&gpu, &mut cache, &mut pool).unwrap();
    let moving = upload_staggered(&gpu, &mut pool, CELLS, &walled_velocity_pattern(CELLS));
    let old = std::mem::replace(&mut state.velocity, moving);
    for face in old.into_faces() {
        pool.release(face);
    }
    let sources = Sources::new(&density_src, &temperature_src).with_fuel(&fuel_src);
    let c = StepConstants {
        conserve_mass: true,
        ..transport_only()
    };
    for _ in 0..3 {
        substep(&gpu, &mut cache, &mut pool, &mut state, sources, &c, SOLVE).unwrap();
    }
    let density = state.density.read_back(&gpu).unwrap();
    let fire = state.fire.as_ref().unwrap();
    let fuel = fire.fuel.read_back(&gpu).unwrap();
    assert!(density.iter().any(|&d| d > 0.0), "something was emitted");
    assert!(
        fuel.iter()
            .zip(&density)
            .all(|(f, d)| f.to_bits() == d.to_bits()),
        "fuel took the same path as density"
    );
    // React is a fraction: advection and the correction keep it in [0, 1]
    // up to MacCormack's clamp, and it has spread with the fuel.
    let react = fire.react.read_back(&gpu).unwrap();
    assert!(
        react.iter().all(|&r| (-1e-6..=1.0 + 1e-6).contains(&r)),
        "react out of [0, 1]"
    );
    // `rate` (like any hash-based pattern) lands on an exact 0 at at least
    // one cell: that cell's fuel and react start, and stay, at 0 unless
    // advection carries content in from a neighbour. It is the one place
    // in this grid where "react present" can only mean "react travelled",
    // rather than "this cell's own emission set it near 1 already" (every
    // other cell's rate is nonzero, so its own local emission alone would
    // keep react high with or without react's own advection). 0.1 (not the
    // 0.5 a saturated cell would reach) is enough to tell "travelled" from
    // "never touched"; a mutation that skips react's advection leaves such
    // a cell's react at exactly 0 forever.
    for (n, (&f, &r)) in fuel.iter().zip(&react).enumerate() {
        if f > 1e-3 && rate[n] <= 1e-6 {
            assert!(
                r > 0.1,
                "cell {n}: fuel {f} arrived without its react ({r})"
            );
        }
    }
}

#[test]
fn a_fire_substep_without_fuel_state_is_an_error() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let zero = vec![0.0; CELLS.voxel_count()];
    let src = upload(&gpu, &mut pool, CELLS, &zero);
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, CELLS).unwrap();
    let mut step = Substep::new(&gpu, &transport_only()).unwrap();
    let err = step
        .pre_projection(
            &gpu,
            &mut cache,
            &mut pool,
            &mut state,
            Sources::new(&src, &src).with_fuel(&src),
        )
        .unwrap_err();
    assert!(format!("{err:?}").contains("fire"), "{err:?}");
    step.abandon(&gpu, &mut pool);
}

/// 32³ conservation setup: a sphere source, low in the domain, that feeds
/// both temperature (to drive buoyancy under `beta`) and fuel; the same
/// field stands for both, so zeroing "the fuel source" for the second phase
/// also stops the temperature that has been keeping the flow moving.
const FIRE_CELLS: FieldDims = FieldDims {
    x: 32,
    y: 32,
    z: 32,
};
const FIRE_H: f32 = 1.0 / 24.0;
const FIRE_DX: f32 = 0.1;
const FIRE_SOLVE: PressureSolve = PressureSolve::GaussSeidel(40);

/// `rate` inside a sphere of `radius` cells at `centre`, 0 elsewhere.
fn sphere_source(cells: FieldDims, centre: [f32; 3], radius: f32, rate: f32) -> Vec<f32> {
    let mut out = vec![0.0; cells.voxel_count()];
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let d = [
                    i as f32 + 0.5 - centre[0],
                    j as f32 + 0.5 - centre[1],
                    k as f32 + 0.5 - centre[2],
                ];
                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < radius * radius {
                    out[index(cells, i, j, k)] = rate;
                }
            }
        }
    }
    out
}

fn sum_f64(values: &[f32]) -> f64 {
    values.iter().map(|&v| f64::from(v)).sum()
}

/// Spec §3.2: while it does not burn, fuel is carried like density — the
/// mass correction keeps its total constant once emission stops, in a
/// closed domain where nothing can leave.
#[test]
fn fuel_is_conserved_while_it_does_not_burn() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let zero = vec![0.0; FIRE_CELLS.voxel_count()];
    let source = sphere_source(FIRE_CELLS, [16.0, 16.0, 6.0], 4.0, 3.0);
    let density_src = upload(&gpu, &mut pool, FIRE_CELLS, &zero);
    let on = upload(&gpu, &mut pool, FIRE_CELLS, &source);
    let off = upload(&gpu, &mut pool, FIRE_CELLS, &zero);
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, FIRE_CELLS).unwrap();
    state.add_fire(&gpu, &mut cache, &mut pool).unwrap();
    let c = StepConstants {
        beta: 1.0,
        open_mask: 0,
        conserve_mass: true,
        fire: true,
        ..StepConstants::new(FIRE_CELLS, FIRE_H, FIRE_DX)
    };
    for _ in 0..10 {
        let sources = Sources::new(&density_src, &on).with_fuel(&on);
        substep(
            &gpu, &mut cache, &mut pool, &mut state, sources, &c, FIRE_SOLVE,
        )
        .unwrap();
    }
    let cutoff = sum_f64(&state.fire.as_ref().unwrap().fuel.read_back(&gpu).unwrap());
    for _ in 0..50 {
        let sources = Sources::new(&density_src, &off).with_fuel(&off);
        substep(
            &gpu, &mut cache, &mut pool, &mut state, sources, &c, FIRE_SOLVE,
        )
        .unwrap();
    }
    let end = sum_f64(&state.fire.as_ref().unwrap().fuel.read_back(&gpu).unwrap());
    let drift = (end - cutoff) / cutoff;
    eprintln!("fuel conservation: cutoff {cutoff}, end {end}, drift {drift:e}");
    assert!(cutoff > 0.1, "fuel must have been emitted: {cutoff}");
    assert!(drift.abs() <= 1e-5, "fuel drift {drift:e}");
}
