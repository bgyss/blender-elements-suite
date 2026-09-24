# Ember solver gate (2b-3c)

- Machine: Apple M1 Max (Apple M1 Max)
- OS: macOS 27.2
- Ember commit: 6d43473
- Date: 2026-09-24 15:02 UTC
- Load average (1, 5, 15 min) before: { 18.03 13.46 12.04 } — **above 2: the machine was loaded**
- Load average after: { 1.79 3.02 4.76 }
- Kernel path check: `plume_plate` 128³ with `mgpcg ×4` through frame 65 (frame 60 instrumented), against `eval_frame`: 0 faces differ, max |Δ| 0e0.

## The rule (spec §4, fixed before any numbers)

A configuration passes when, at 128³ and 256³ and in `plume`, `plume_collider` and `plume_wind`, its median frame time over frames 25–48 is at most the Gauss–Seidel ×160 median in the same process, and its masked divergence RMS at frames 60 and 120 is at most Mantaflow's for that scene and resolution (`docs/bench/results.md`, the 7fe5d9d run; `plume_wind` uses `plume`'s until the rerun). It must also pass `plume_plate` at 128³: RMS divergence after projection over before it, at frames 60 and 120, through the kernel API, at most 1e-3. The chosen configuration must be stable past the floor: in `plume_collider` at 128³ the masked divergence after 40 iterations is at most 4× that after the chosen count. Plain V-cycles (`multigrid`) are measured but cannot be the default. The fastest passing configuration becomes the default; if none passes, work stops.

Mantaflow references (masked RMS, f60 / f120): `plume` and `plume_wind` 1.49e-4 / 1.07e-4 at 128³, 7.41e-5 / 4.51e-5 at 256³; `plume_collider` 2.84e-4 / 1.09e-4 at 128³, 9.12e-5 / 5.25e-5 at 256³.

## Procedure

Staged sweep: at 128³, counts rise from 1 (`multigrid` to 8, `mgpcg` to 24) until the three scenes and `plume_plate` all pass; that count is checked at 256³ and stepped up until it passes there too (and still at 128³). A 256³ count stops at its first failing scene (— below). Divergence runs read back only at frames 60 and 120. Then each candidate and Gauss–Seidel ×160 are timed: 3 runs of frames 1–48, each run's median over frames 25–48, interleaved (Gauss–Seidel, multigrid, mgpcg, Gauss–Seidel, …) so drifting load hits all of them. Timing is per frame (`eval_frame` plus a blocking wait), not per solve: runs of one scene differ only in the solver, so it is the same comparison without instrumenting inside the solver.

## Divergence sweep

Masked divergence RMS, 1/s, frames 60 / 120. `plume_plate`: after/before ratio, frames 60 / 120.

| method | count | res | plume | plume_collider | plume_wind | plume_plate ratio | pass |
|---|---|---|---|---|---|---|---|
| multigrid | 1 | 128³ | 2.16e-1 / 5.67e-2 | 1.82e-1 / 1.12e-1 | 1.00e-3 / 0.00e0 | 1.11e-1 / 7.14e-2 | no |
| multigrid | 2 | 128³ | 3.90e-2 / 8.57e-3 | 2.91e-2 / 1.60e-2 | 1.77e-4 / 0.00e0 | 2.55e-2 / 1.11e-2 | no |
| multigrid | 3 | 128³ | 7.73e-3 / 1.58e-3 | 5.78e-3 / 3.11e-3 | 3.82e-5 / 0.00e0 | 1.16e-2 / 4.38e-3 | no |
| multigrid | 4 | 128³ | 1.65e-3 / 3.25e-4 | 1.22e-3 / 6.22e-4 | 8.78e-6 / 0.00e0 | 7.28e-3 / 4.22e-3 | no |
| multigrid | 5 | 128³ | 3.54e-4 / 6.91e-5 | 2.64e-4 / 1.30e-4 | 2.66e-6 / 0.00e0 | 4.91e-3 / 3.39e-3 | no |
| multigrid | 6 | 128³ | 7.79e-5 / 1.62e-5 | 5.85e-5 / 2.82e-5 | 1.88e-6 / 0.00e0 | 3.24e-3 / 1.86e-3 | no |
| multigrid | 7 | 128³ | 1.93e-5 / 5.96e-6 | 1.39e-5 / 6.86e-6 | 1.85e-6 / 0.00e0 | 2.03e-3 / 8.51e-4 | no |
| multigrid | 8 | 128³ | 9.02e-6 / 4.75e-6 | 5.39e-6 / 3.33e-6 | 1.85e-6 / 0.00e0 | 1.18e-3 / 4.11e-4 | no |
| mgpcg | 1 | 128³ | 1.79e-1 / 8.86e-2 | 1.86e-1 / 1.17e-1 | 1.00e-3 / 0.00e0 | 1.12e-1 / 7.02e-2 | no |
| mgpcg | 2 | 128³ | 1.17e-2 / 4.78e-3 | 9.92e-3 / 5.69e-3 | 5.01e-5 / 0.00e0 | 2.38e-2 / 6.73e-3 | no |
| mgpcg | 3 | 128³ | 6.25e-4 / 3.18e-4 | 6.78e-4 / 3.59e-4 | 4.10e-6 / 0.00e0 | 7.33e-3 / 6.66e-3 | no |
| mgpcg | 4 | 128³ | 3.63e-5 / 2.17e-5 | 4.39e-5 / 2.52e-5 | 1.92e-6 / 0.00e0 | 4.72e-3 / 1.23e-3 | no |
| mgpcg | 5 | 128³ | 9.85e-6 / 5.69e-6 | 6.15e-6 / 3.98e-6 | 1.92e-6 / 0.00e0 | 6.59e-3 / 1.36e-3 | no |
| mgpcg | 6 | 128³ | 9.97e-6 / 5.72e-6 | 5.71e-6 / 3.83e-6 | 1.91e-6 / 0.00e0 | 3.15e-3 / 8.73e-4 | no |
| mgpcg | 7 | 128³ | 9.94e-6 / 5.61e-6 | 5.74e-6 / 3.86e-6 | 1.92e-6 / 0.00e0 | 2.24e-3 / 5.78e-4 | no |
| mgpcg | 8 | 128³ | 9.99e-6 / 5.75e-6 | 5.70e-6 / 3.83e-6 | 1.91e-6 / 0.00e0 | 1.88e-3 / 4.39e-4 | no |
| mgpcg | 9 | 128³ | 9.96e-6 / 5.69e-6 | 5.76e-6 / 3.82e-6 | 1.92e-6 / 0.00e0 | 1.06e-3 / 3.08e-4 | no |
| mgpcg | 10 | 128³ | 9.97e-6 / 5.68e-6 | 5.69e-6 / 3.83e-6 | 1.91e-6 / 0.00e0 | 7.41e-4 / 2.55e-4 | yes |
| mgpcg | 10 | 256³ | 3.67e-5 / 1.73e-5 | 2.35e-5 / 1.72e-5 | 5.46e-6 / 0.00e0 | n/a | yes |

Measurements with no measured cells (the smoke mask was empty, so the RMS is 0 and passes vacuously): 128³ `multigrid ×1` plume_wind frame 120; 128³ `multigrid ×2` plume_wind frame 120; 128³ `multigrid ×3` plume_wind frame 120; 128³ `multigrid ×4` plume_wind frame 120; 128³ `multigrid ×5` plume_wind frame 120; 128³ `multigrid ×6` plume_wind frame 120; 128³ `multigrid ×7` plume_wind frame 120; 128³ `multigrid ×8` plume_wind frame 120; 128³ `mgpcg ×1` plume_wind frame 120; 128³ `mgpcg ×2` plume_wind frame 120; 128³ `mgpcg ×3` plume_wind frame 120; 128³ `mgpcg ×4` plume_wind frame 120; 128³ `mgpcg ×5` plume_wind frame 120; 128³ `mgpcg ×6` plume_wind frame 120; 128³ `mgpcg ×7` plume_wind frame 120; 128³ `mgpcg ×8` plume_wind frame 120; 128³ `mgpcg ×9` plume_wind frame 120; 128³ `mgpcg ×10` plume_wind frame 120; 256³ `mgpcg ×10` plume_wind frame 120.

Candidates:

- `multigrid`: no count up to 8 passes; the method fails
- `mgpcg`: count 10

## Timing

Median frame ms over the 3 runs' medians (min–max of the run medians). ✓: at most Gauss–Seidel's.

| res | scene | `gauss_seidel ×160` ms | `mgpcg ×10` ms |
|---|---|---|---|
| 128³ | plume | 93.0 (92.5–93.4) | 92.9 (80.7–93.0) ✓ |
| 128³ | plume_collider | 140.4 (138.1–141.2) | 139.4 (105.8–142.3) ✓ |
| 128³ | plume_wind | 93.3 (92.7–94.1) | 73.4 (73.1–93.8) ✓ |
| 256³ | plume | 720.7 (720.4–746.3) | 694.9 (679.0–736.2) ✓ |
| 256³ | plume_collider | 1149.3 (1147.7–1155.1) | 1115.0 (889.9–1120.1) ✓ |
| 256³ | plume_wind | 759.7 (757.1–761.6) | 712.7 (711.9–719.1) ✓ |

## Past-floor stability (`plume_collider`, 128³)

- `mgpcg`: ×40 over the candidate count, frames 60 / 120: 1.010 / 1.000 — stable

## Rule applied

- `mgpcg ×10`: divergence and thin plate pass; frame time at most Gauss–Seidel's in every scene at both resolutions; past-floor stable

Rule applied: **fastest passing: `mgpcg ×10`**.

Decision (recorded by the user): _pending_
