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

/// Cells i in 1..4, j in 1..3, k in 1..3 of `CELLS` (12 cells, well inside
/// the domain on every side) at `rate`, 0 elsewhere.
fn block_source(cells: FieldDims, rate: f32) -> Vec<f32> {
    let mut out = vec![0.0; cells.voxel_count()];
    for k in 1..3 {
        for j in 1..3 {
            for i in 1..4 {
                out[index(cells, i, j, k)] = rate;
            }
        }
    }
    out
}

/// Fuel and react track density bit-for-bit through advection and the mass
/// correction: they start identical to density (one substep at rate 1/h,
/// still air, puts density = fuel = react = 1.0 in the block and 0
/// elsewhere exactly, spec §3.2 step 1), then all three see the same
/// velocity, the same zero sources, and the same `conserve_mass` for three
/// more substeps. Nothing but their own field distinguishes them from
/// density in this setup, so any divergence is a bug in how fuel or react
/// is carried, not a property of the test data.
#[test]
fn fuel_and_react_are_advected_and_mass_corrected_as_density_is() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let zero = vec![0.0; CELLS.voxel_count()];
    let rate = block_source(CELLS, 1.0 / 0.25);
    let on = upload(&gpu, &mut pool, CELLS, &rate);
    let off = upload(&gpu, &mut pool, CELLS, &zero);
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, CELLS).unwrap();
    state.add_fire(&gpu, &mut cache, &mut pool).unwrap();
    // One substep, still air: density = fuel = react = 1.0 in the block, 0
    // elsewhere, exactly.
    let seed = Sources::new(&on, &off).with_fuel(&on);
    substep(
        &gpu,
        &mut cache,
        &mut pool,
        &mut state,
        seed,
        &transport_only(),
        SOLVE,
    )
    .unwrap();
    // Now move it: sources off, velocity on, with the mass correction.
    let moving = upload_staggered(&gpu, &mut pool, CELLS, &walled_velocity_pattern(CELLS));
    let old = std::mem::replace(&mut state.velocity, moving);
    for face in old.into_faces() {
        pool.release(face);
    }
    let sources = Sources::new(&off, &off).with_fuel(&off);
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
    let react = fire.react.read_back(&gpu).unwrap();
    let spread = fuel
        .iter()
        .zip(&rate)
        .filter(|&(&f, &r)| r == 0.0 && f > 1e-3)
        .count();
    assert!(
        spread > 0,
        "fuel must have spread beyond the block by transport"
    );
    let fuel_diffs = fuel
        .iter()
        .zip(&density)
        .filter(|(f, d)| f.to_bits() != d.to_bits())
        .count();
    assert_eq!(
        fuel_diffs, 0,
        "fuel differs from density at {fuel_diffs} cells"
    );
    let react_diffs = react
        .iter()
        .zip(&density)
        .filter(|(r, d)| r.to_bits() != d.to_bits())
        .count();
    assert_eq!(
        react_diffs, 0,
        "react differs from density at {react_diffs} cells"
    );
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
    assert!(cutoff > 0.1, "fuel must have been emitted: {cutoff}");
    assert!(
        drift.abs() <= 1e-5,
        "fuel drift {drift:e} (cutoff {cutoff}, end {end})"
    );
}

use elements_core::gpu::ComputeBatch;
use elements_ember::kernels::{Uniforms, burn, flame};

fn burning() -> StepConstants {
    StepConstants {
        fire: true,
        burning_rate: 1.2, // × h 0.25 = 0.3 a substep
        flame_smoke: 1.5,
        ignition_temperature: 1.5,
        max_temperature: 3.0,
        ..StepConstants::new(CELLS, 0.25, 0.125)
    }
}

/// The burn of 2b-4 spec §3.2 on the CPU: (fuel, react, density, temperature).
/// The flame reads react clamped to [0, 1] (the global mass correction can
/// push react slightly above 1); react itself is left unclamped.
fn cpu_burn(c: &StepConstants, f0: f32, r0: f32, d: f32, t: f32) -> (f32, f32, f32, f32) {
    let burn = c.burning_rate * c.h;
    let f1 = (f0 - burn).max(0.0);
    // See burn.wgsl: react passes through untouched when burn is 0 (rather
    // than dividing by f0, which need not round to exactly 1), and
    // otherwise divides by a hard zero check, not an epsilon.
    let r1 = if burn == 0.0 {
        r0
    } else if f0 != 0.0 {
        r0 * (f1 / f0)
    } else {
        0.0
    };
    let smoke = (0.5 + 0.5 * (1.0 - f0).max(0.0)) * (f0 - f1) * 0.1 * c.flame_smoke;
    let f = r1.clamp(0.0, 1.0).sqrt();
    let t1 = if f > 0.0 {
        (1.0 - f) * c.ignition_temperature + f * c.max_temperature
    } else {
        t
    };
    (f1, r1, d + smoke, t1)
}

#[test]
fn the_burn_matches_the_cpu_reference() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    // Fuel in [0, 2] with zeros; react in [0, 1].
    let f0: Vec<f32> = pattern(CELLS, 11)
        .iter()
        .map(|v| v.max(0.0) * 2.0)
        .collect();
    let r0 = abs_pattern(12);
    let d0 = pattern(CELLS, 13);
    let t0 = pattern(CELLS, 14);
    let fields = [&f0, &r0, &d0, &t0].map(|v| upload(&gpu, &mut pool, CELLS, v));
    let c = burning();
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    burn(
        &gpu, &mut cache, &mut batch, &u, &fields[0], &fields[1], &fields[2], &fields[3],
    )
    .unwrap();
    batch.submit(&gpu).unwrap();
    let want: Vec<_> = (0..f0.len())
        .map(|n| cpu_burn(&c, f0[n], r0[n], d0[n], t0[n]))
        .collect();
    let got = fields.each_ref().map(|f| f.read_back(&gpu).unwrap());
    assert_close(
        &got[0],
        &want.iter().map(|w| w.0).collect::<Vec<_>>(),
        1e-6,
        "fuel",
    );
    assert_close(
        &got[1],
        &want.iter().map(|w| w.1).collect::<Vec<_>>(),
        1e-6,
        "react",
    );
    assert_close(
        &got[2],
        &want.iter().map(|w| w.2).collect::<Vec<_>>(),
        1e-6,
        "density",
    );
    assert_close(
        &got[3],
        &want.iter().map(|w| w.3).collect::<Vec<_>>(),
        1e-5,
        "temperature",
    );
    // Where there was no fuel, temperature is untouched, bit for bit.
    for n in 0..f0.len() {
        if f0[n] == 0.0 {
            assert_eq!(got[3][n].to_bits(), t0[n].to_bits(), "cell {n} had no fuel");
        }
    }
}

/// A cell whose react is pushed above 1 by the mass correction still clamps
/// to max_temperature rather than exceeding it.
#[test]
fn temperature_clamps_react_above_one_to_max_temperature() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let n = CELLS.voxel_count();
    // burn = 0 here so f1 == f0 and react passes through unchanged, letting
    // us isolate the clamp in the temperature profile.
    let c = StepConstants {
        fire: true,
        flame_smoke: 1.5,
        ignition_temperature: 1.5,
        max_temperature: 3.0,
        ..StepConstants::new(CELLS, 0.25, 0.125)
    };
    let fuel = upload(&gpu, &mut pool, CELLS, &vec![1.0; n]);
    let mut react_v = vec![0.5; n];
    react_v[0] = 1.4; // above 1, as the mass correction can produce
    let react = upload(&gpu, &mut pool, CELLS, &react_v);
    let density = upload(&gpu, &mut pool, CELLS, &vec![0.0; n]);
    let temperature = upload(&gpu, &mut pool, CELLS, &vec![0.0; n]);
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    burn(
        &gpu,
        &mut cache,
        &mut batch,
        &u,
        &fuel,
        &react,
        &density,
        &temperature,
    )
    .unwrap();
    batch.submit(&gpu).unwrap();
    let got_temperature = temperature.read_back(&gpu).unwrap();
    let got_react = react.read_back(&gpu).unwrap();
    // React itself stays unclamped...
    assert_close(&got_react[0..1], &[1.4], 1e-6, "react cell 0");
    // ...but the temperature it drives is clamped to max_temperature, not
    // pushed above it.
    assert_close(
        &got_temperature[0..1],
        &[c.max_temperature],
        1e-5,
        "temperature cell 0",
    );
    // flame() reads react through the same [0, 1] clamp, so its output at
    // this cell is sqrt(1.0) = 1.0, not sqrt(1.4).
    let out = upload(&gpu, &mut pool, CELLS, &vec![0.0; n]);
    let mut flame_batch = ComputeBatch::new();
    flame(&gpu, &mut cache, &mut flame_batch, &u, &react, &out).unwrap();
    flame_batch.submit(&gpu).unwrap();
    let got_flame = out.read_back(&gpu).unwrap();
    assert_close(&got_flame[0..1], &[1.0], 1e-6, "flame cell 0");
}

#[test]
fn fuel_burns_out_on_the_closed_form() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let n = CELLS.voxel_count();
    let fuel = upload(&gpu, &mut pool, CELLS, &vec![1.0; n]);
    let react = upload(&gpu, &mut pool, CELLS, &vec![1.0; n]);
    let density = upload(&gpu, &mut pool, CELLS, &vec![0.0; n]);
    let temperature = upload(&gpu, &mut pool, CELLS, &vec![0.0; n]);
    let u = Uniforms::new(&gpu, &burning()).unwrap();
    let out = upload(&gpu, &mut pool, CELLS, &vec![0.0; n]);
    for step in 1..=4 {
        let mut batch = ComputeBatch::new();
        burn(
            &gpu,
            &mut cache,
            &mut batch,
            &u,
            &fuel,
            &react,
            &density,
            &temperature,
        )
        .unwrap();
        flame(&gpu, &mut cache, &mut batch, &u, &react, &out).unwrap();
        batch.submit(&gpu).unwrap();
        let want_fuel = (1.0 - 0.3 * step as f32).max(0.0);
        assert_close(
            &fuel.read_back(&gpu).unwrap(),
            &vec![want_fuel; n],
            1e-6,
            "fuel",
        );
        // react = fuel / F₀ with F₀ = 1.
        assert_close(
            &react.read_back(&gpu).unwrap(),
            &vec![want_fuel; n],
            1e-6,
            "react",
        );
        assert_close(
            &out.read_back(&gpu).unwrap(),
            &vec![want_fuel.sqrt(); n],
            1e-6,
            "flame",
        );
    }
}

#[test]
fn a_substep_burns_after_it_emits() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let n = CELLS.voxel_count();
    let zero = vec![0.0; n];
    let src = upload(&gpu, &mut pool, CELLS, &zero);
    // 4 fuel/s × h 0.25 = 1 unit emitted; burning then takes 0.3.
    let fuel_src = upload(&gpu, &mut pool, CELLS, &vec![4.0; n]);
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, CELLS).unwrap();
    state.add_fire(&gpu, &mut cache, &mut pool).unwrap();
    let sources = Sources::new(&src, &src).with_fuel(&fuel_src);
    substep(
        &gpu,
        &mut cache,
        &mut pool,
        &mut state,
        sources,
        &burning(),
        SOLVE,
    )
    .unwrap();
    let fire = state.fire.as_ref().unwrap();
    assert_close(
        &fire.fuel.read_back(&gpu).unwrap(),
        &vec![0.7; n],
        1e-6,
        "fuel",
    );
}
