# Ember against Mantaflow

- Machine: Apple M1 Max
- OS: macOS 27.2
- Ember commit: 31cfeb2
- Blender: 5.2.2 LTS
- Date: 2026-09-25

Timings from a run whose 1-minute load average was above 2 before or after its timed runs should be redone on an idle machine. Such runs: `ember-plume-64` (12.5 → 11.8), `ember-plume-128` (27.9 → 19.8), `ember-plume-256` (18.7 → 16.3), `ember-plume_collider-64` (18.9 → 17.7), `ember-plume_collider-128` (32.1 → 25.2), `ember-plume_collider-256` (43.2 → 15.5), `ember-plume_wind-64` (28.6 → 27.0), `ember-plume_wind-128` (25.0 → 17.9), `ember-plume_wind-256` (17.2 → 9.2), `mantaflow-plume-64` (11.8 → 18.9), `mantaflow-plume-128` (17.9 → 32.0), `mantaflow-plume-256` (7.5 → 58.7), `mantaflow-plume_collider-64` (17.7 → 28.6), `mantaflow-plume_collider-128` (20.7 → 26.4), `mantaflow-plume_collider-256` (10.0 → 22.4), `mantaflow-plume_wind-64` (25.4 → 27.9), `mantaflow-plume_wind-128` (15.4 → 19.5), `mantaflow-plume_wind-256` (7.2 → 89.7).

## Summary

Latency from a parameter change and a side-by-side Cycles render are in
`docs/bench/latency.md` and `docs/bench/render/README.md` (2b-3b).

This is 2b-3c's rerun. Ember now uses the preview preset: MGPCG ×4 and the
global mass correction. The 2b-3 run, on Gauss–Seidel ×160, is kept with its
Summary and tables in `docs/bench/results-2b3/`. Mantaflow's `plume` and
`plume_collider` bakes match 2b-3's to every digit. Its `plume_wind` is new,
since it now uses a wind field with `flow`. Every run was under load (1-minute
load average 7–90, above), so treat the timings as upper bounds. The quality
metrics do not depend on load.

**Speed.** An Ember frame is 5.8–8.4× faster than Mantaflow's in every scene
and at every resolution. At 256³ it went from 811 to 516 ms for `plume` and
from 1,126 to 695 ms for `plume_collider`. The benchmark times whole frames:
per solve, MGPCG ×4 took 42–59% of Gauss–Seidel's time at 128³ and 34–41% at
256³ in the solver gate (`solver-gate.md`), measured before the MacCormack
fallback and the mass correction. `plume_collider` at 128³ takes 97.3 ms, close to
preview's 100 ms budget.

**Memory.** The multigrid levels add 14–17% (at 256³, `plume` went from 1,539
to 1,758 MiB). Ember uses 0.36–0.39× Mantaflow's memory at 64³, 0.55× at 128³
and 0.58–0.62× at 256³.

**Divergence.** At frame 60, `plume` has 113× less than in 2b-3 at 64³, 822×
less at 128³ and 3,750× less at 256³. Ember is below Mantaflow in every scene
and resolution except at 256³, where Mantaflow is lower in `plume_collider`
at frames 60 and 120 and in `plume` at frame 120 (below). In `plume_wind` it
is 30–74× below at frame 60; by 120 Ember's smoke has all left.

**Conservation.** Drift at 80 is 0.0% in `plume` and `plume_collider` at
every resolution: at most 7.8e-8 in absolute terms (`plume` at 64³), which
is 9.2e-7 of that run's frame-60 mass, so under 1e-6 of it everywhere. In
2b-3 it was +1.2% to +10.6%. Mantaflow's is −9.5% to +8.7%. At 120,
`plume_collider`, whose smoke barely reaches the top, is still 0.0%: under
1.5e-6 of its frame-60 mass at every frame through 120 (at most 1.45e-6, at
128³ in frame 119). For `plume`, drift at 120 is +0.7%, −1.1% and −2.0%.
That is most likely the error of the frame-resolution outflow estimate once
smoke has left, not mass the solver lost: the correction enforces a
first-order outflow each substep at the domain faces themselves, while the
metric estimates outflow once a frame through planes two cells in, so the
two count different flux. Mantaflow's is +24.5%, +10.6% and +2.0%.

**Wind no longer fails.** Mass follows the emitter exactly until smoke
reaches an open face. From then on, mass inside plus the estimated outflow
stays within 3% of what was emitted (by the CSVs' `mass_inside` and
`outflow_rate`), and within 0.4% from frame 90.
Ember's smoke leaves through +x from frames 23–34, against Mantaflow's
45–59, and is gone by frame 90.

**Where Mantaflow still wins.**
- **Divergence at 256³:** `plume_collider` at frames 60 and 120 (1.16× and
  1.49× lower than Ember), and `plume` at 120 (1.28×). Preview's 4 iterations
  are the limit here. Final's 10 passed the solver gate (`solver-gate.md`),
  but that was measured before the MacCormack fallback and the mass
  correction, and final has not been re-measured on the shipped solver.
- **Energy and height:** at frame 60, Mantaflow's `plume` has 1.3–1.9× the
  kinetic energy per cell, its centroid is 0.14–0.18 m higher and its top
  0.19–0.27 m higher. Its `plume_collider` has 1.08× the kinetic energy per
  cell at 64³, 1.43× at 128³ and 2.12× at 256³, and its centroid is
  0.03–0.11 m higher. At 128³ and 256³ it also has more vorticity per cell
  at 60: up to 15% more in `plume` and up to 42% more in `plume_collider`.
  Part of this comes from the heat mapping (see Heat).

**Known limits.**
- Preview's 4 iterations miss the thin-plate target of 1e-3 (4.7e-3).
- The mass correction is global. It skips the whole of any field that has a
  negative value, and it can hide a local error.
- Mantaflow's wind pushes only smoke, while Ember's moves all the air, so the
  `plume_wind` shapes differ by construction.
- Load, as above.

## `plume`

| solver | cells | frame ms (median, min–max) | peak MiB | div. RMS 1/s (60 / 120) | measured cells (60 / 120) | kinetic energy m⁵/s² (60 / 120) | KE per cell m⁵/s² (60 / 120) | vorticity m³/s (60 / 120) | vorticity per cell m³/s (60 / 120) | centroid m (60 / 120) | top m (60 / 120) | drift at 80 (% of mass at 60) | drift at 120 (% of mass at 60) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ember | 64³ | 15.6 (12.4–34.4), median of 3 | 27.6 | 2.28e-5 / 5.87e-6 | 4920 / 3668 | 0.0218 / 0.0132 | 4.43e-6 / 3.61e-6 | 0.779 / 0.511 | 1.58e-4 / 1.39e-4 | 0.780 / 1.63 | 1.17 / 1.98 | 7.78e-8 (+0.0%); outflow from frame 74 credits 0.0% | 6.18e-4 (+0.7%) |
| mantaflow | 64³ | 94.0 (74.2–216), median of 3 | 72.4 | 9.54e-5 / 3.50e-5 | 10931 / 12889 | 0.0642 / 0.0118 | 5.87e-6 / 9.14e-7 | 1.75 / 1.03 | 1.60e-4 / 8.00e-5 | 0.925 / 1.75 | 1.36 / 1.95 | 0.00767 (+8.7%); outflow from frame 78 credits 0.2% | 0.0217 (+24.5%) |
| ember | 128³ | 71.1 (68.0–83.8), median of 3 | 220.2 | 3.65e-5 / 1.85e-5 | 34380 / 47492 | 0.0180 / 0.0128 | 5.25e-7 / 2.70e-7 | 0.821 / 1.06 | 2.39e-5 / 2.24e-5 | 0.777 / 1.64 | 1.15 / 1.93 | 6.06e-9 (+0.0%) | -9.18e-4 (-1.1%) |
| mantaflow | 128³ | 508 (421–732), median of 3 | 401.2 | 1.49e-4 / 1.07e-4 | 71047 / 62486 | 0.0652 / 0.0106 | 9.17e-7 / 1.69e-7 | 1.85 / 1.09 | 2.61e-5 / 1.75e-5 | 0.956 / 1.72 | 1.41 / 1.98 | 0.00247 (+2.9%); outflow from frame 76 credits 17.1% | 0.00893 (+10.6%) |
| ember | 256³ | 516 (507–560), 1 run | 1758.4 | 6.69e-5 / 5.76e-5 | 251172 / 384715 | 0.0180 / 0.0110 | 7.16e-8 / 2.85e-8 | 0.926 / 1.41 | 3.69e-6 / 3.66e-6 | 0.789 / 1.56 | 1.16 / 1.81 | 3.90e-8 (+0.0%) | -0.00171 (-2.0%) |
| mantaflow | 256³ | 3583 (3322–3917), 1 run | 2857.2 | 7.41e-5 / 4.51e-5 | 460188 / 322052 | 0.0617 / 0.0123 | 1.34e-7 / 3.83e-8 | 1.95 / 1.31 | 4.24e-6 / 4.06e-6 | 0.964 / 1.78 | 1.40 / 1.99 | -0.00105 (-1.3%); outflow from frame 78 credits 7.2% | 0.00160 (+2.0%) |

## `plume_collider`

| solver | cells | frame ms (median, min–max) | peak MiB | div. RMS 1/s (60 / 120) | measured cells (60 / 120) | kinetic energy m⁵/s² (60 / 120) | KE per cell m⁵/s² (60 / 120) | vorticity m³/s (60 / 120) | vorticity per cell m³/s (60 / 120) | centroid m (60 / 120) | top m (60 / 120) | drift at 80 (% of mass at 60) | drift at 120 (% of mass at 60) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ember | 64³ | 19.6 (14.9–47.0), median of 3 | 34.1 | 9.40e-6 / 3.87e-6 | 4384 / 14172 | 0.00709 / 0.0122 | 1.62e-6 / 8.62e-7 | 0.653 / 2.05 | 1.49e-4 / 1.44e-4 | 0.612 / 1.13 | 0.828 / 1.39 | 2.02e-8 (+0.0%) | 1.13e-7 (+0.0%) |
| mantaflow | 64³ | 113 (92.3–275), median of 3 | 95.2 | 9.22e-5 / 4.19e-5 | 10150 / 36908 | 0.0178 / 0.0281 | 1.75e-6 / 7.60e-7 | 1.47 / 3.87 | 1.45e-4 / 1.05e-4 | 0.641 / 1.09 | 0.859 / 1.36 | -0.00685 (-9.5%) | -0.0139 (-19.3%) |
| ember | 128³ | 97.3 (69.6–116), median of 3 | 271.8 | 4.37e-5 / 2.15e-5 | 35668 / 121156 | 0.00895 / 0.0156 | 2.51e-7 / 1.29e-7 | 1.01 / 3.03 | 2.84e-5 / 2.50e-5 | 0.653 / 1.23 | 0.914 / 1.55 | 6.59e-8 (+0.0%) | 1.14e-7 (+0.0%) |
| mantaflow | 128³ | 687 (582–1018), median of 3 | 495.2 | 2.84e-4 / 1.09e-4 | 78261 / 316262 | 0.0280 / 0.0289 | 3.58e-7 / 9.15e-8 | 2.65 / 6.75 | 3.39e-5 / 2.13e-5 | 0.730 / 1.20 | 0.977 / 1.46 | -0.00314 (-3.9%) | -0.00371 (-4.7%) |
| ember | 256³ | 695 (692–726), 1 run | 2170.6 | 1.06e-4 / 7.80e-5 | 252013 / 976935 | 0.00744 / 0.0150 | 2.95e-8 / 1.54e-8 | 1.05 / 4.57 | 4.17e-6 / 4.68e-6 | 0.662 / 1.30 | 0.918 / 1.54 | 5.34e-9 (+0.0%) | 7.12e-8 (+0.0%) |
| mantaflow | 256³ | 5827 (5391–6633), 1 run | 3759.0 | 9.12e-5 / 5.25e-5 | 500061 / 1927015 | 0.0313 / 0.0496 | 6.26e-8 / 2.57e-8 | 2.97 / 9.07 | 5.93e-6 / 4.70e-6 | 0.775 / 1.51 | 1.05 / 1.79 | -0.00206 (-2.6%) | 0.00434 (+5.5%) |

## `plume_wind`

| solver | cells | frame ms (median, min–max) | peak MiB | div. RMS 1/s (60 / 120) | measured cells (60 / 120) | kinetic energy m⁵/s² (60 / 120) | KE per cell m⁵/s² (60 / 120) | vorticity m³/s (60 / 120) | vorticity per cell m³/s (60 / 120) | centroid m (60 / 120) | top m (60 / 120) | drift at 80 (% of mass at 60) | drift at 120 (% of mass at 60) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ember | 64³ | 15.1 (12.2–17.0), median of 3 | 27.6 | 8.33e-7 / 0 | 3892 / 0 | 0.0511 / 0 | 1.31e-5 / — | 0.0831 / 0 | 2.13e-5 / — | 0.327 / — | 0.484 / — | -9.99e-5 (-0.3%); outflow from frame 23 credits 85.8% | -6.14e-4 (-1.7%) |
| mantaflow | 64³ | 98.0 (72.5–186), median of 3 | 70.9 | 6.17e-5 / 4.29e-4 | 13811 / 2781 | 0.0794 / 0.00370 | 5.75e-6 / 1.33e-6 | 1.35 / 0.276 | 9.80e-5 / 9.93e-5 | 0.689 / 1.56 | 1.05 / 1.83 | 0.00124 (+1.6%); outflow from frame 45 credits 68.5% | -9.80e-4 (-1.3%) |
| ember | 128³ | 68.4 (67.6–77.5), median of 3 | 220.2 | 1.91e-6 / 0 | 33720 / 0 | 0.0555 / 0 | 1.64e-6 / — | 0.102 / 0 | 3.02e-6 / — | 0.329 / — | 0.477 / — | -1.63e-4 (-0.4%); outflow from frame 30 credits 83.7% | -6.67e-4 (-1.8%) |
| mantaflow | 128³ | 501 (420–638), median of 3 | 397.2 | 9.50e-5 / 8.63e-5 | 93244 / 20116 | 0.0674 / 0.00336 | 7.23e-7 / 1.67e-7 | 1.72 / 0.309 | 1.84e-5 / 1.54e-5 | 0.755 / 1.62 | 1.10 / 1.84 | 0.00520 (+6.3%); outflow from frame 55 credits 60.0% | 0.00597 (+7.2%) |
| ember | 256³ | 526 (503–581), 1 run | 1758.4 | 5.39e-6 / 0 | 280281 / 0 | 0.0577 / 0 | 2.06e-7 / — | 0.118 / 0 | 4.20e-7 / — | 0.330 / — | 0.480 / — | -1.92e-4 (-0.5%); outflow from frame 34 credits 82.7% | -7.07e-4 (-1.9%) |
| mantaflow | 256³ | 3534 (3092–6387), 1 run | 2854.8 | 1.61e-4 / 1.91e-4 | 547502 / 188959 | 0.0557 / 0.00825 | 1.02e-7 / 4.37e-8 | 2.03 / 0.669 | 3.70e-6 / 3.54e-6 | 0.774 / 1.62 | 1.12 / 1.84 | 0.00364 (+4.4%); outflow from frame 59 credits 52.4% | 0.00491 (+5.9%) |

## Notes

- **Velocity metrics** (divergence, kinetic energy, vorticity) cover only measured cells: those whose whole 3×3×3 neighbourhood is inside the domain, outside any collider and holds smoke (density > 1e-6). The rule is the same for both solvers, because Mantaflow's cache stores velocity only where there is smoke (spec §4.1). The measured-cell count says how much of each field that is.
- **Per-cell values.** The solvers' smoky regions differ in size, so their measured cells differ too, and the kinetic energy and vorticity totals partly measure that size. The totals divided by the measured-cell count are the fairer comparison.
- **Near-empty domains.** Where a frame's mass is below 1e-9, its centroid, top and per-cell values are shown as —, since they would describe a few stray cells.
- **Wind.** Mantaflow's wind field acts only on cells that hold smoke; Ember's air relaxes towards the ambient airflow in every cell. `plume_wind` compares the plume's shape, not a matched force field.
- **Heat.** Mantaflow's emitter heat is held at a set value (Ember's emitter-centre heat at frame 24), not added at Ember's rate, so Mantaflow's emitter is hotter before frame 24 and cooler after it. Plume centroid and top carry that difference. Emitted mass matches while both solvers are still emitting and the plumes have not yet diverged: over frames 12–24, Ember's mass is 0.99–1.04× Mantaflow's across every run. By frame 60 the ratio is 0.45–1.17×, since it then also carries what each solver's advection gains or loses (most of all in `plume_wind`).
- **Pressure.** Mantaflow solves with multigrid-preconditioned conjugate gradients to a tolerance; Ember runs the same method for a fixed count, the preview preset's MGPCG ×4 per substep, and then its global mass correction on density and temperature.
- **Drift** is (mass inside the outflow planes + outflow since frame 60) − that mass at frame 60, with outflow estimated at frame resolution as the net upwind flux out through a plane two cells in from each open face (the top, and in `plume_wind` both x sides), and mass summed over the cells inside those planes (2b-3 spec §4.3, 2b-3c spec §6). Emission stops after frame 60, so a perfect solver drifts 0. Where no outflow has started by frame 80, drift at 80 is mass gained or lost inside the domain. Where it has, the frame-80 cell names the first frame whose outflow rate is non-zero and the share of the frame-60 mass that the outflow estimate credits by frame 80; that share rests on the frame-resolution estimate, and so does that part of the drift. Drift at 120 covers the whole outflow period.
- **Frame times** exclude frame 1. Ember's frame is `eval_frame` plus a blocking GPU wait; Mantaflow's is the difference between consecutive cache files' modification times, so it includes writing the cache. 256³ is one run of each solver, the other resolutions the median of three runs' medians; the min–max range pools every timed frame.
- **Peak memory** is in MiB (2²⁰ bytes), and the two solvers' figures count different things. Ember's is the field pool's allocated bytes, with the frame cache off: textures only, not buffers, pipelines or the driver. Mantaflow's is Blender's peak resident memory while baking, minus the same scene's peak without a bake.
