# Fire divergence: which variant closes the gap (FT3a, 2026-09-30)

In `fire`, Ember's RMS divergence at frame 60 is 4.0×, 26× and 28×
Mantaflow's at 64³, 128³ and 256³ ([results.md](results.md), Fire). In the
smoke scenes Ember usually leaves less than Mantaflow does. This experiment
changes one thing at a time to find what the gap tracks. No solver change is
committed. It repeats and extends the 64³ probes in
[fire-divergence-investigation.md](fire-divergence-investigation.md) (2026-09-26),
adding 128³ and 256³, peak |u| and fuel.

**Result.** The gap tracks the pressure solve's convergence. The Euler backtrace
has little effect on it. Six MGPCG cycles instead of preview's four close the
gap at 64³ (0.08× Mantaflow) and reach about parity at 128³ (0.91×). They do
not close it at 256³, where 1.58× remains. Fuel and flame volume stay within
2% of the baseline; measured cells fall about 10% at 64³. Flame shape was
not inspected. Euler for every grid leaves divergence where it was.
`flame_vorticity = 0` lowers divergence by weakening the flame, which does not
fix anything. The `final` preset goes well below Mantaflow and changes the fire's
size. It runs eight substeps, and in loaded timings its frames took about 10×
as long as preview's.

## Method

- **Machine:** Apple M1 Max (MacBookPro18,2), macOS 27.2, Metal.
- **Base commit:** `0400839` on `worktree-flamethrower-ft0-ft3`.
- **Scene:** `Scene::fire(res)` from `crates/elements-ember/src/bench/mod.rs`,
  run to frame 60 under preview (1 substep, MGPCG ×4, mass correction on,
  fuel clamped to [0, 10], flame vorticity 12/s). Velocity faces use the Euler
  backtrace while fire burns, and scalars use RK2.
- **Harness:** a scratch `fire-variant RES` mode added to the `benchmark`
  example. It reads variants from environment variables, and it writes no
  result files, so `docs/bench/results/` is untouched. For frame 60 it uses
  the benchmark's own `metrics_run`, so divergence, measured cells, fuel mass
  and flame volume are computed exactly as in `results.md`.
- **Peak |u|:** the largest max |u| over all velocity faces in frames 1–60,
  as the 2026-09-25 fire diagnosis measured it. That diagnosis is
  `.superpowers/sdd/fire-diagnosis.md`, a git-ignored local file in the main
  checkout, not in the repository, so its numbers cannot be checked from this
  branch. The two numbers used here are copied from it: Mantaflow's fire peaks
  at 15.3 m/s at 32³ and 11.7 m/s at 64³.
- **Frame ms:** the median of frames 2–60 from one timed run of `eval_frame`
  plus a blocking wait. `results.md` uses 120 frames and three runs.
- **Scratch branch:** `scratch/fire-divergence-experiments`, left in place.

| # | variant | code | how to run |
|---|---|---|---|
| 1 | baseline (preview) | `2f21cb4` | `benchmark fire-variant RES` |
| 2 | Euler backtrace for every grid while fire burns | `e1b2fae` (`euler_faces: u32::from(c.fire)` in `kernels/mod.rs`) | `benchmark fire-variant RES` |
| 3 | `flame_vorticity = 0` | `2f21cb4` | `FV=0 benchmark fire-variant RES` |
| 4 | `final` preset (8 substeps, MGPCG ×10) | `2f21cb4` | `QUALITY=final benchmark fire-variant RES` |
| 5 | pressure cycles 6 (preview otherwise) | `2f21cb4` | `CYCLES=6 benchmark fire-variant RES` |

Build with `cargo build --release -p elements-ember --example benchmark` at the
listed commit. The binary is `target/release/examples/benchmark`.

On variant 4: the brief described `final` as "8 substeps, RK2". As shipped,
`final` still traces velocity faces with Euler while fire burns, because
`euler_faces` depends only on fire. It also raises pressure cycles from 4 to
10. It is therefore two changes from preview, and it was run as shipped. The
2026-09-26 probe separated the two changes at 64³ only: 8 substeps with ×4
gave 2.44e-5.

Variant 5 is the one extra variant. The 2026-09-26 probe had named pressure
cycles as the likely cause but had recorded no fuel or peak |u| for it. This
run checks whether the fix changes the fire itself.

## Results

Mantaflow's RMS divergence at frame 60 comes from [results.md](results.md):
1.25e-4 at 64³, 4.60e-5 at 128³ and 9.00e-5 at 256³. Mantaflow's peak |u| is
11.7 m/s at 64³ (the local fire diagnosis, see Method). Mantaflow's fuel at frame 60 is 0.696,
0.517 and 0.443, and its flame volume 0.695, 0.327 and 0.205 m³.

| variant | res | div. RMS f60 (1/s) | × Mantaflow | measured cells | fuel f60 (× baseline) | flame m³ f60 | peak \|u\| m/s (frame) | frame ms | 1-min load before → after |
|---|---|---|---|---|---|---|---|---|---|
| 1 baseline | 64³ | 4.98e-4 | 3.98× | 21,617 | 0.610 | 0.373 | 10.8 (16) | 20.9 | 28.2 → 25.7 |
| 1 baseline | 128³ | 1.19e-3 | 25.8× | 118,033 | 0.599 | 0.343 | 12.9 (27) | 106.4 | 25.7 → 21.4 |
| 1 baseline | 256³ | 2.50e-3 | 27.7× | 782,469 | 0.631 | 0.367 | 14.2 (14) | 867.6 | 18.9 → 9.8 |
| 2 Euler every grid | 64³ | 5.16e-4 | 4.13× | 21,343 | 0.513 (0.84×) | 0.490 | 9.6 (19) | 16.3 | 9.0 → 9.0 |
| 2 Euler every grid | 128³ | 1.16e-3 | 25.1× | 154,333 | 0.659 (1.10×) | 0.578 | 12.0 (31) | 74.0 | 9.0 → 7.7 |
| 2 Euler every grid | 256³ | 2.36e-3 | 26.2× | 845,025 | 0.790 (1.25×) | 0.519 | 12.1 (25) | 547.9 | 7.7 → 6.0 |
| 3 flame_vorticity 0 | 64³ | 5.32e-5 | 0.43× | 4,279 | 0.646 (1.06×) | 0.148 | 3.3 (57) | 20.0 | 9.8 → 9.4 |
| 3 flame_vorticity 0 | 128³ | 3.49e-4 | 7.6× | 42,015 | 0.689 (1.15×) | 0.163 | 3.5 (58) | 103.8 | 9.4 → 9.5 |
| 4 `final` | 64³ | 2.72e-6 | 0.022× | 41,749 | 0.543 (0.89×) | 0.485 | 9.4 (41) | 214.7 | 9.5 → 9.0 |
| 4 `final` | 128³ | 8.06e-6 | 0.18× | 258,014 | 0.764 (1.28×) | 0.449 | 10.3 (48) | 1127.0 | 9.0 → 10.3 |
| 5 pressure ×6 | 64³ | 1.00e-5 | 0.080× | 19,428 | 0.599 (0.98×) | 0.373 | 10.2 (24) | 23.2 | 6.4 → 6.6 |
| 5 pressure ×6 | 128³ | 4.18e-5 | 0.91× | 115,436 | 0.590 (0.99×) | 0.339 | 13.0 (27) | 117.0 | 6.6 → 6.7 |
| 5 pressure ×6 | 256³ | 1.42e-4 | 1.58× | 752,560 | 0.645 (1.02×) | 0.363 | 17.5 (14) | 891.1 | 6.7 → 12.0 |

I skipped `flame_vorticity = 0` and `final` at 256³. They show no
resolution trend that a 256³ run would decide, and `final` at 256³ would take
about 20 minutes.

**Baseline reproduces `results.md`.** At all three resolutions, the RMS
divergence, measured cells and frame-60 fuel match `results.md`'s Ember row to
the printed digits: 4.98e-4 / 1.19e-3 / 2.50e-3, and fuel 0.610 / 0.599 /
0.631. Reruns of variants 1 and 2 at 128³ gave bit-identical metrics, as
expected from one backend on one machine.

**Load.** Every run was under load: the 1-minute load average was between 6.0
and 28.2, and other processes were running. Every frame time is an upper bound
and cannot be compared across rows. Divergence, fuel and peak |u| do not
depend on load.

## What the gap tracks

- **Pressure convergence.** This is the main cause. Changing only the cycle
  count from 4 to 6 lowers divergence 50× at 64³, 28× at 128³ and 18× at
  256³. The fire is unchanged: fuel is 0.98–1.02× the baseline, flame volume
  is within 1.2%, and measured cells are within 10% (−10.1% at 64³).
  - At 256³ it still leaves 1.58× Mantaflow's divergence. The 2026-09-26
    probe found 1.50e-4 there with ×10, so extra cycles at one substep stop
    helping at that point. The rest of the 256³ gap is something else.
  - Peak |u| at 256³ rises from 14.2 to 17.5 m/s, at frame 14. That is the
    early core spike, not a runaway, but it is higher than Mantaflow's
    15.3 m/s at 32³ and 11.7 m/s at 64³ (the local fire diagnosis; no 256³
    Mantaflow peak is recorded).
- **Not the Euler backtrace.** Tracing every grid with Euler changes
  divergence by −6% to +4%, which is noise-sized next to a 26× gap.
  - It lowers peak |u| (9.6 / 12.0 / 12.1 m/s, matching the diagnosis).
  - It changes the fire. The flame grows 1.3–1.7×, which moves it away from
    Mantaflow at 128³ and 256³. Fuel goes 0.84× the baseline at 64³, but
    1.10× and 1.25× at 128³ and 256³.
  - The handoff's suspect, the Euler velocity trace, therefore does not
    explain the divergence. Moving the scalars onto the same trace does not
    help either.
- **`flame_vorticity = 0` does not fix the gap; it removes most of the
  fire.** Divergence falls 9.4× at 64³ and 3.4× at 128³, and still leaves
  7.6× Mantaflow's at 128³. The measured cells shrink to 20–36% of the
  baseline, flame volume to about 0.4–0.5×, and peak |u| to 3.3–3.5 m/s.
  Flame vorticity drives the fast, rotational core that 4 cycles cannot
  project, but turning it off is not a match for Mantaflow.
- **`final` goes well below Mantaflow**: 0.02× at 64³ and 0.18× at 128³.
  - It runs eight substeps. In loaded timings, which are upper bounds taken
    at different loads, it took 1,127 ms a frame at 128³ against the
    baseline's 106 ms.
  - It changes the fire: 2.2× the measured cells at 128³, fuel 1.28×, and
    flame 1.3×.
  - It is an offline bake setting, not the fix for preview.

## Side finding: the Euler backtrace's cost

Euler for every grid ran faster than the baseline in this harness.

This is data from a loaded machine, not a timing result. Every figure here
is an upper bound.

- **Paired run at 128³.** I alternated the two binaries twice at a 1-minute
  load of 8–10. Euler for every grid took 74.0 and 73.7 ms a frame, and the
  baseline took 106.4 and 106.6 ms.
- **Unpaired runs.** The 64³ and 256³ rows in the table ran at very different
  loads, so they do not corroborate the paired run.
- **Preview cost.** The cost of Euler for every grid, and of ×6 cycles, has
  not been measured on an idle machine. Any claim about either's preview cost
  needs an idle-machine paired run first.

## Recommendation (the user decides)

1. **Scope a follow-up task: raise fire's pressure solve to 6 MGPCG cycles.**
   This could be a fire-only preview setting or a new preview default.
   - **Why this variant.** It is the one variant that closes the gap at 64³
     and reaches about parity at 128³ while keeping fuel and flame volume
     within 2%. It leaves 1.58× at 256³.
   - **Flame shape was not inspected.** Nothing was rendered, and only flame
     volume and measured-cell counts were compared.
   - **It needs no solver code**, only a preset value.
   - **Before deciding,** measure its preview cost at 128³ with an
     idle-machine paired run. The cost is unmeasured on an idle machine, and
     fire preview was already over 100 ms under load in `results.md`.
   - Afterwards, rerun `just bench fire` (Ember and Mantaflow, an hour or more)
     and update `results.md`, `presets-fire.md` and the renders.
2. **Do not adopt Euler for every grid, or `flame_vorticity = 0`, as a
   divergence fix.** Neither closes the gap on equal terms. The first leaves it
   as it was, and the second removes most of the flame.
3. **Optional, same follow-up.** Time Euler for every grid, with and without
   ×6 cycles, in an idle-machine paired run. Its cost is unmeasured on an idle
   machine; the loaded runs above are no basis for a claim. That pair's
   divergence was not tested here. Adopting it would also change fire's flame
   size (1.3–1.7×). It would contradict the 2b-1 spec's RK2 wording, which the
   2b-4 exception already relaxes for velocity faces.
4. **The 256³ residual** (1.58× with ×6, and no better with ×10) is not
   explained. It is small next to the original 28×.

## What this does not show

- **Frame 60 only.** There is no burn-out or drift at frames 90 and 120.
  `results.md` shows Ember burning out more slowly than Mantaflow there.
- **Flame shape was not inspected or rendered.** Only flame volume and
  measured-cell counts were compared.
- **One run per variant**, so there is no spread. Divergence is deterministic
  on this machine: reruns at 128³ were bit-identical.
- **Mantaflow was not re-run.** Its numbers come from `results.md`, and its
  peak |u| from the local fire diagnosis (see Method).
- **Variants 3 and 4 were not run at 256³.**
- **The 256³ baseline ran under load** (18.9 → 9.8), not on the idle machine
  the brief preferred. Divergence does not depend on load; timings do.
- **Variant 5 raises peak |u| at 256³**, from 14.2 to 17.5 m/s.
- **Every timing is a loaded upper bound.** None of them supports a
  conclusion about preview cost.

# FT3b: jet into the shack (2026-09-30)

The fire-scene result above does not carry over to the shot. With the shack
in the jet's path, divergence is 58× the same jet without the shack under
preview, and 509× with 6 cycles. Six cycles lower it only 1.75× at frame 60.
These are smoke-only numbers: no flame reaches the shack at one substep.
Merely having a collider does not explain the excess: a solid box of the same
outer size leaves divergence at the no-shack level. The excess is not local
to the shack either, because cells more than 4 cells from it have the same
RMS. That is consistent with the pressure solve converging slowly around
one-cell walls, which is known risk (m) (piece 2 spec §6). Two other causes
are untested (see "What the gap tracks here"). No residual or convergence
quantity was measured. Nothing blew up. Smoke reaches the shack. The flame
does not reach it under preview, and barely does with 8 substeps.

## Method

- **Machine and commit:** Apple M1 Max, Metal. Base commit `038e9cc` on
  `worktree-flamethrower-ft0-ft3`; the harness is committed with this record.
- **Scene:** `examples/jet_shack.elements`, generated by
  `write_jet_shack_example` in `crates/elements-ember/tests/jet_shack.rs`.
  It is the FT0 shot spec at half size: dims [128, 64, 64] over 2.0 m, so
  dx stays 15.625 mm, and every position and length is halved.
  - Shack: size [0.6, 0.5, 0.5], plank height 0.05, thickness 0.01, gap
    0.005, broken fraction 0.2, seed 7, at (1.45, 0.5, 0). It is an
    `ember.mesh_collider` with `offset` 0.5 dx and 372 triangles, and it
    gives 6,755 solid cells.
  - Nozzle: cone length 0.15, radii 0.015 and 0.03, at (0.2, 0.5, 0.3).
    Velocity [14, 0, 0] with `velocity_local`, `velocity_blend` 20. Noise
    amplitude 0.6, scale 0.04 m (the spec's 0.08 m halved), seed 7.
    `fuel_rate` 24 and `temperature_rate` 1, as in `Scene::fire`, with no
    density. The pulse runs on frames 5–40 of 60 (the shot has 10–150 of 240).
  - Solver: `Scene::fire`'s solver parameters (preview: 1 substep, MGPCG ×4,
    buoyancy from heat). Emitter outputs 0–4 go into solver inputs 0–3 and 6,
    and the collider's into 4 and 5.
- **Variants** are document overrides of `pressure_cycles` and
  `max_substeps`. There is no solver change. The control has no collider.
  The solid-box diagnostic replaces the shack with one `ember.collider` box
  of the shack's outer extent, [1.15, 1.75] × [0.25, 0.75] × [0, 0.5].
- **Harness:** `jet_shack_divergence` (`#[ignore]`) in the same file:
  `cargo nextest run -p elements-ember --release --test jet_shack
  --run-ignored ignored-only -E 'test(jet_shack_divergence)' --no-capture`.
  It steps `eval_frame` as the benchmark's `metrics_run` does and measures
  with `metrics::measure`, so "div RMS" is over measured cells, as in
  `results.md` and the table above. Solid cells come from the collider's
  own SDF, below zero, as `solidify.wgsl` decides.
- **Far field:** the same `measure`, with the shack's mask grown by 4 cells
  passed as the solid mask, so that only cells well away from the shack are
  measured. The control uses the same region.
- **Reach:** smoke (density) mass at or past the shack's front plane
  (x ≥ 1.15 m), flame volume past that plane (flame > 0.01), and contact
  (fluid cells beside a shack cell that hold smoke, fuel or flame). In the
  control, contact counts the cells where the shack would be.
- **Blow-up:** any non-finite velocity or density in any frame, or peak |u|
  over all faces in frames 1–60 above 30 m/s.
- **Solid check:** the largest |u| on faces with a shack cell on both sides
  at frame 60. It is 0 in every run with the shack, and 0.84–1.05 m/s on the
  same faces in the controls. The solver therefore holds the shack solid.
- **CLI check:** `elements bake examples/jet_shack.elements --frames 1-12`
  loads and bakes the committed document.
- **The harness asserts nothing, by design.** It is an experiment, not a
  check. On a non-finite value it prints an `ERROR at frame N` line and
  stops that variant, but the test still passes. Read its output. Its
  `f30`, `f40` and `f60` lines print divergence, fuel, flame, smoke and
  reach for each variant.

## Results

Frame 60, 128×64×64. "× control" divides by the run with the same substeps
and cycles and no collider. The ×10 row at one substep has no such control,
so it divides by the ×6 control.

| variant | div RMS f40 | div RMS f60 | measured cells f60 | far field f60 | × control f60 | peak \|u\| m/s (frame) | smoke past front f60 (share) | first smoke past front / contact | max flame past front m³ (frame) | frame ms | 1-min load before → after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| shack, preview ×4 | 6.63e-2 | 8.84e-3 | 29,866 | 8.72e-3 | 58× | 11.6 (38) | 2.8e-4 (41%) | f30 / f30 | 0 | 38.4 | 7.68 → 7.23 |
| shack, ×6 | 4.18e-3 | 5.04e-3 | 26,866 | 5.02e-3 | 509× | 11.6 (22) | 2.4e-4 (38%) | f30 / f29 | 0 | 43.7 | 7.23 → 7.23 |
| control, ×4 | 5.04e-4 | 1.52e-4 | 33,594 | 1.45e-4 | — | 10.9 (20) | 3.4e-4 (50%) | f30 / f29 | 0 | 23.9 | 7.23 → 7.13 |
| control, ×6 | 2.70e-5 | 9.91e-6 | 33,409 | 9.23e-6 | — | 11.7 (20) | 3.3e-4 (48%) | f29 / f29 | 0 | 27.7 | 7.13 → 7.13 |
| diag.: shack, ×10 | 3.15e-3 | 2.78e-4 | 32,318 | 2.71e-4 | 28× (×6 control) | 12.0 (17) | 2.6e-4 (41%) | f30 / f30 | 0 | 53.8 | 7.13 → 6.72 |
| diag.: solid box, ×4 | 8.44e-4 | 1.87e-4 | 29,310 | 1.88e-4 | 1.2× | 11.8 (17) | 2.4e-4 (38%) | f31 / f31 | 0 | 28.3 | 6.72 → 6.72 |
| diag.: solid box, ×6 | 2.93e-5 | 6.11e-6 | 28,052 | 6.19e-6 | 0.62× | 12.1 (18) | 2.2e-4 (34%) | f30 / f30 | 0 | 33.3 | 6.72 → 6.34 |
| diag.: shack, 8 substeps ×4 | 1.41e-3 | 5.07e-4 | 44,044 | 4.55e-4 | 39× | 9.1 (38) | 2.2e-4 (37%) | f13 / f14 | 8e-5 (33) | 214.9 | 6.34 → 7.34 |
| diag.: control, 8 substeps ×4 | 6.05e-5 | 1.30e-5 | 44,511 | 1.24e-5 | — | 8.4 (8) | 3.9e-4 (65%) | f13 / f14 | 3e-5 (16) | 165.7 | 7.34 → 7.07 |
| diag.: shack, 8 substeps ×10 | 2.15e-4 | 2.11e-5 | 47,344 | 1.83e-5 | 12× | 8.4 (8) | 2.1e-4 (35%) | f13 / f14 | 7e-5 (33) | 331.0 | 5.12 → 4.25 |
| diag.: control, 8 substeps ×10 | 3.31e-6 | 1.72e-6 | 44,981 | 1.83e-6 | — | 8.5 (40) | 3.8e-4 (63%) | f13 / f14 | 8e-5 (36) | 246.7 | 4.25 → 5.18 |

For comparison, the `fire` scene at 128³, frame 60 (the table above): Ember
preview 1.19e-3, ×6 4.18e-5, and Mantaflow 4.60e-5.

The fire while the pulse runs, from the harness's `f30` and `f40` lines.
Fuel is Σ fuel · dx³, and flame is the volume with flame above 0.01. Both
are 0 at frame 60 in every run.

| variant | fuel f30 | flame f30 m³ | fuel f40 | flame f40 m³ |
|---|---|---|---|---|
| shack, preview ×4 | 0.00154 | 0.00639 | 0.00140 | 0.00459 |
| shack, ×6 | 0.00139 | 0.00404 | 0.00140 | 0.00466 |
| control, ×4 | 0.00144 | 0.00539 | 0.00127 | 0.00467 |
| control, ×6 | 0.00139 | 0.00547 | 0.00146 | 0.00400 |
| diag.: shack, ×10 | 0.00132 | 0.00412 | 0.00135 | 0.00484 |
| diag.: solid box, ×4 | 0.00127 | 0.00476 | 0.00129 | 0.00409 |
| diag.: solid box, ×6 | 0.00127 | 0.00468 | 0.00140 | 0.00483 |
| diag.: shack, 8 substeps ×4 | 0.00032 | 0.00315 | 0.00032 | 0.00330 |
| diag.: control, 8 substeps ×4 | 0.00031 | 0.00322 | 0.00032 | 0.00317 |
| diag.: shack, 8 substeps ×10 | 0.00032 | 0.00324 | 0.00030 | 0.00304 |
| diag.: control, 8 substeps ×10 | 0.00031 | 0.00318 | 0.00033 | 0.00324 |

- **No blow-up.** No run had a non-finite value. Peak |u| stayed at or
  below 12.1 m/s, which is the nozzle's 14 m/s target blended in, far under
  30 m/s.
- **The fire has burnt out by frame 60.** The pulse ends at frame 40, and
  fuel and flame are 0 at frame 60 in every run. Frame 60 therefore
  measures the flow the fire left behind. Frame 40, the last pulse frame, is
  the burning comparison: there, flame volume is 0.0040–0.0048 m³ at one
  substep and 0.0030–0.0033 m³ at 8 substeps (the fire table).
  - The shack's ×6 divergence **rises** after the pulse ends, from 4.18e-3
    at frame 40 to 5.04e-3 at frame 60. Every other run falls over the same
    frames. This is not explained. So frame 60 is not simply the burning
    flow's error decaying.
- **Consistent with thin walls (risk (m)); not simply having a collider.** The solid box
  stays within 0.6–1.2× the control at frame 60. The shack is 58× (×4) and
  509× (×6). Far-field RMS matches measured-cell RMS in every shack run
  (8.72e-3 against 8.84e-3 under preview), so the error is spread across
  the domain and not confined to the planks' neighbourhood. At frame 30,
  before any smoke has passed the front plane, the shack run already leaves
  24× the control's divergence (2.04e-2 against 8.61e-4). A pressure solve
  that has not converged around one-cell walls fits that pattern. Risk (m)
  records the same for `plume_plate`: preview's ×4 misses the thin plate's
  target, and `final`'s ×10 passes.
  - The box differs from the shack in more than wall thickness, so the
    evidence does not isolate the walls.
  - No residual or convergence quantity was measured.
- **What the gap tracks here: two causes are untested.** Neither was run for
  this record.
  1. **The enclosed cavity.** The shack is hollow and nearly sealed: a
     pocket of fluid cells sits behind planks with gaps of about 5 mm,
     under one voxel. A pressure solve with a poorly connected fluid region
     is another plausible cause. The solid box has no cavity.
  2. **The mesh-collider SDF path.** The shack goes through
     `ember.mesh_collider`, and the box through the analytic
     `ember.collider`.
  - **Controls that would separate them:**
    - A hollow analytic box with the shack's wall thickness, which would
      have thin walls and a cavity but no mesh path.
    - The same shape built as a mesh.
    - The shack with one wall opened, which would keep the thin walls but
      remove the enclosure.
- **More cycles help, but not in proportion.** ×6 gives 1.75× at frame 60
  (16× at frame 40), and ×10 gives 32× at frame 60. With the shack, ×10
  lands at 1.8× the no-shack preview. With 8 substeps and ×10 (`final`'s
  substeps and cycles), the shack scene reaches 2.1e-5. That is still 12×
  its own control.
- **Reach.** Smoke reaches the shack in every run: 35–41% of the smoke is
  past the front plane at frame 60, and contact starts at frame 29–31 at one
  substep. In the controls, 48–65% of the smoke is past that plane, so the
  shack holds some back.
  - At one substep the jet is slow. Smoke takes about 25 frames to travel
    about 1 m, though the nozzle target is 14 m/s. CFL asks for about 31
    substeps at 14 m/s and dx 15.6 mm, and the solver clamps 55 of 60 frames
    to 1.
  - **No flame ever passes the front plane at one substep.** With 8
    substeps smoke arrives by frame 13, and at most 3e-5 to 8e-5 m³ of flame
    (about 8–21 cells) crosses the plane, with or without the shack.
  - The flame is short in this half-size scene whatever the solver settings.
    The shot's acceptance point 1, flame that reaches and wraps the shack, is
    not met by this scene at any setting tried.

### What this does and does not show

- **One run per variant, frames 30, 40 and 60 only.** Reruns of the first
  five variants gave identical metrics, since one backend on one machine is
  deterministic. There are no later frames.
- **A small domain.** It is the shot at half size with the shot's dx. The
  nozzle's volume is 1/8 of the shot's, so this scene has much less fuel
  than `fire` (fuel 0.0015 at frame 30 against about 0.6 in `fire` at
  frame 60). Its divergence is not directly comparable to `fire`'s or
  Mantaflow's.
- **Nothing was rendered.** Reach is measured by mass and cell counts, not
  looked at.
- **No Mantaflow twin** exists for this scene.
- **Every timing is a loaded upper bound.** The 1-minute load was 4.3–7.7
  for the runs in the table. Two earlier runs of the first four and five
  variants, at load 14.8 → 15.2 and 15.1 → 13.3, gave the same metrics.
  They are not in the table.
  Frame ms includes a blocking wait but not the read-backs. No conclusion
  about cost follows from them.
- **The noise scale was halved** with the other lengths, to 0.04 m. The
  spec gives 0.08 m for the full-size shot.
- **No bug found in Tasks 4–8, but one is not ruled out.**
  - The cone emits at the nozzle, and `velocity_local` gives the +x jet.
  - The mesh collider gives a solid (6,755 cells, zero face velocity inside).
  - The excess divergence is consistent with thin walls (risk (m)). The
    analytic solid box does not show it, and the analytic `plume_plate`,
    another thin wall, has the same known weakness.
  - The mesh-collider SDF path and the enclosed cavity remain untested as
    causes (see above).

## Recommendation for FT3 (the user decides)

**(b) Scope a solver task for pressure convergence around thin colliders.
The variant to scope is `pressure_cycles` 10 when a thin collider is
present.** This is `final`'s cycle count, as a preset rule or a document
default for the shot. The task should also weigh the adaptive residual stop
that risk (m) left out of scope.

**This recommendation is provisional.**
- **One run per variant, frame 60 only, after the pulse ended.** Frame 40 is
  also recorded.
- **The flow is smoke only.**
  - The 58× and 509× figures are smoke-only flow. At one substep no flame
    ever crosses the shack's front plane, so no run measures flame meeting
    the shack.
  - The shot needs about 8 substeps or more to get flame there. At 8
    substeps, even ×10 leaves 12× its control (2.1e-5).
  - The "×10 cuts divergence 32×" support therefore comes from the
    one-substep regime, which the shot cannot use.
- **Before choosing a cycle count,** the scoped solver task should
  re-measure with flame reaching the shack, at 8 substeps or the `final`
  preset. It should also run the cavity and mesh-path controls above.

- ×6, Task 9's fire-scene variant, does not carry over. It closes the
  fire-scene gap, but with the shack it leaves 5.0e-3, 509× its control.
- ×10 cuts the shack's frame-60 divergence 32× under preview, to 1.8× the
  no-shack preview level. It does not close the gap to its own control.
  With 8 substeps it reaches 2.1e-5, still 12× its control.
- Its cost is unmeasured on an idle machine, as for ×6.
- A separate finding for FT6, not part of this choice: at preview's single
  substep, the 14 m/s jet does not carry flame to the shack. FT6 needs
  more substeps, a longer or larger flame (fuel rate, burning rate, or the
  full-size nozzle), or both, before the shot's acceptance point 1 can be
  met.

The alternatives: (a) is not supported, since divergence is 58× the
no-shack level and flame does not reach the shack. (c), accepting the gap,
is reasonable if FT6 bakes the shot in `final` anyway, since `final`'s
settings already land at 2.1e-5 here. No solver change was made.
