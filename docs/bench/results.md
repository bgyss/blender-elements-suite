# Ember against Mantaflow (2b-3)

- Machine: Apple M1 Max
- OS: macOS 27.2
- Ember commit: 7fe5d9d
- Blender: 5.2.2 LTS
- Date: 2026-09-24

Timings from a run whose 1-minute load average was above 2 before or after its timed runs should be redone on an idle machine. Such runs: `ember-plume-64` (9.4 → 9.0), `ember-plume-128` (12.3 → 8.3), `ember-plume-256` (69.0 → 21.4), `ember-plume_collider-64` (12.6 → 11.9), `ember-plume_collider-128` (23.9 → 13.9), `ember-plume_collider-256` (30.7 → 10.5), `ember-plume_wind-64` (15.1 → 13.5), `ember-plume_wind-128` (13.7 → 9.7), `ember-plume_wind-256` (22.7 → 10.2), `mantaflow-plume-64` (8.8 → 12.6), `mantaflow-plume-128` (7.2 → 25.4), `mantaflow-plume-256` (7.9 → 40.9), `mantaflow-plume_collider-64` (11.4 → 15.1), `mantaflow-plume_collider-128` (11.4 → 14.3), `mantaflow-plume_collider-256` (9.2 → 35.5), `mantaflow-plume_wind-64` (13.5 → 12.3), `mantaflow-plume_wind-128` (8.8 → 69.7), `mantaflow-plume_wind-256` (8.3 → 44.2).

## Summary

Every run here was measured under load (1-minute load average 7–70; see
above), so read the timings as indicative. The quality metrics do not depend
on load.

**Speed and memory: Ember wins.** An Ember frame is 3–8× faster than a
Mantaflow frame at every resolution and in every scene. For example, `plume`
runs at 12 against 95 ms at 64³ and 811 against 3,777 ms at 256³, and
`plume_collider` at 1,126 against 5,843 ms at 256³. Mantaflow's frames include
writing its cache. Ember's peak memory is about a third of Mantaflow's at 64³
and about half at 256³ (1.5 against 2.9 GB for `plume`).

**Incompressibility and conservation: Mantaflow wins.** Mantaflow's conjugate
gradient solve leaves an RMS divergence of about 1e-4 at every resolution.
In `plume` at frame 60, Ember's fixed 160 Gauss–Seidel iterations leave 27×
more at 64³ and about 3,400× more at 256³ (0.0026 and 0.25). The gap widens
with resolution, because a fixed iteration count converges less on a larger
grid.
After emission stops, Mantaflow approaches mass conservation as resolution
rises: its `plume` drift at frame 80 is +8.7%, +2.9% and −1.3%. Ember gains mass
through advection at every resolution: +3.8%, +6.0% and +10.6%.

**Detail: mixed, and confounded.** Per measured cell, Mantaflow's `plume`
carries 1.3–1.9× Ember's kinetic energy at frame 60, and 1.0–1.2× its
vorticity. Its plume also rises higher, with a centroid 0.13–0.17 m above
Ember's at frame 60. But Mantaflow's emitter is hotter than Ember's before
frame 24 (see Heat), so part of that extra energy and height comes from the
heat mapping, not from the solver. By frame 120, Ember keeps as much vorticity
per cell as Mantaflow or more.

**`plume_wind` fails in Ember, badly at 128³ and 256³.** Its divergence at
frame 60 is 0.14 at 64³, 55× `plume`'s, and it reaches 1.4 at 128³ and 2.4–3.3
at 256³. At 128³ every bit of smoke has left through the top by frame 84. A follow-up run at 128³ separated two causes:
- More pressure iterations cut the divergence at frame 60 from 1.40 (160
  iterations) to 0.17 (1,000), and 4 substeps cut it to 0.12. So much of the
  divergence is an unconverged solve.
- The smoke still leaves by frame 80–90 in every configuration. Ember's wind
  pushes all of the air against closed side walls, while the open top holds
  pressure at zero, and that drives a circulation out through the top.
  Mantaflow's wind pushes only the smoke, so it never sets that up.

`plume_wind`'s Ember rows therefore measure this failure, not the plume,
most of all at 128³ and 256³. The drift estimate also misses such a fast exit.

## `plume`

| solver | cells | frame ms (median, min–max) | peak MiB | div. RMS 1/s (60 / 120) | measured cells (60 / 120) | kinetic energy m⁵/s² (60 / 120) | KE per cell m⁵/s² (60 / 120) | vorticity m³/s (60 / 120) | vorticity per cell m³/s (60 / 120) | centroid m (60 / 120) | top m (60 / 120) | drift at 80 (% of mass at 60) | drift at 120 (% of mass at 60) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ember | 64³ | 12.3 (11.5–15.6), median of 3 | 24.2 | 0.00258 / 0.00121 | 4940 / 3440 | 0.0219 / 0.0131 | 4.44e-6 / 3.82e-6 | 0.780 / 0.525 | 1.58e-4 / 1.53e-4 | 0.781 / 1.64 | 1.17 / 1.98 | 0.00322 (+3.8%) | 0.0213 (+24.9%) |
| mantaflow | 64³ | 94.8 (78.2–127), median of 3 | 84.2 | 9.54e-5 / 3.50e-5 | 10931 / 12889 | 0.0642 / 0.0118 | 5.87e-6 / 9.14e-7 | 1.75 / 1.03 | 1.60e-4 / 8.00e-5 | 0.925 / 1.75 | 1.36 / 1.95 | 0.00767 (+8.7%) | 0.0217 (+24.5%) |
| ember | 128³ | 174 (95.4–1186), median of 3 | 192.8 | 0.0300 / 0.00756 | 35160 / 48520 | 0.0185 / 0.0136 | 5.25e-7 / 2.81e-7 | 0.836 / 1.17 | 2.38e-5 / 2.41e-5 | 0.783 / 1.65 | 1.15 / 1.95 | 0.00508 (+6.0%) | 0.0194 (+22.9%) |
| mantaflow | 128³ | 542 (456–681), median of 3 | 405.0 | 1.49e-4 / 1.07e-4 | 71047 / 62486 | 0.0652 / 0.0106 | 9.17e-7 / 1.69e-7 | 1.85 / 1.09 | 2.61e-5 / 1.75e-5 | 0.956 / 1.72 | 1.41 / 1.98 | 0.00247 (+2.9%) | 0.00893 (+10.6%) |
| ember | 256³ | 811 (758–1015), 1 run | 1539.0 | 0.251 / 0.0282 | 282814 / 415261 | 0.0199 / 0.0110 | 7.03e-8 / 2.66e-8 | 0.991 / 1.74 | 3.51e-6 / 4.18e-6 | 0.829 / 1.61 | 1.18 / 1.86 | 0.0102 (+10.6%) | 0.0225 (+23.5%) |
| mantaflow | 256³ | 3777 (3683–4161), 1 run | 2858.2 | 7.41e-5 / 4.51e-5 | 460188 / 322052 | 0.0617 / 0.0123 | 1.34e-7 / 3.83e-8 | 1.95 / 1.31 | 4.24e-6 / 4.06e-6 | 0.964 / 1.78 | 1.40 / 1.99 | -0.00105 (-1.3%) | 0.00160 (+2.0%) |

## `plume_collider`

| solver | cells | frame ms (median, min–max) | peak MiB | div. RMS 1/s (60 / 120) | measured cells (60 / 120) | kinetic energy m⁵/s² (60 / 120) | KE per cell m⁵/s² (60 / 120) | vorticity m³/s (60 / 120) | vorticity per cell m³/s (60 / 120) | centroid m (60 / 120) | top m (60 / 120) | drift at 80 (% of mass at 60) | drift at 120 (% of mass at 60) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ember | 64³ | 20.8 (16.6–23.0), median of 3 | 29.2 | 2.54e-4 / 6.05e-4 | 4452 / 13512 | 0.00730 / 0.0118 | 1.64e-6 / 8.73e-7 | 0.646 / 1.95 | 1.45e-4 / 1.44e-4 | 0.606 / 1.12 | 0.828 / 1.39 | 9.70e-4 (+1.2%) | 0.0123 (+15.0%) |
| mantaflow | 64³ | 113 (101–215), median of 3 | 86.8 | 9.22e-5 / 4.19e-5 | 10150 / 36908 | 0.0178 / 0.0281 | 1.75e-6 / 7.60e-7 | 1.47 / 3.87 | 1.45e-4 / 1.05e-4 | 0.641 / 1.09 | 0.859 / 1.36 | -0.00685 (-9.5%) | -0.0139 (-19.3%) |
| ember | 128³ | 172 (142–540), median of 3 | 232.9 | 0.00784 / 0.00363 | 36018 / 126183 | 0.00911 / 0.0169 | 2.53e-7 / 1.34e-7 | 1.02 / 3.26 | 2.84e-5 / 2.59e-5 | 0.657 / 1.25 | 0.914 / 1.57 | 0.00210 (+2.4%) | 0.0116 (+13.3%) |
| mantaflow | 128³ | 699 (618–892), median of 3 | 492.5 | 2.84e-4 / 1.09e-4 | 78261 / 316262 | 0.0280 / 0.0289 | 3.58e-7 / 9.15e-8 | 2.65 / 6.75 | 3.39e-5 / 2.13e-5 | 0.730 / 1.20 | 0.977 / 1.46 | -0.00314 (-3.9%) | -0.00371 (-4.7%) |
| ember | 256³ | 1126 (1120–1298), 1 run | 1859.8 | 0.0712 / 0.0178 | 267493 / 1045631 | 0.00802 / 0.0162 | 3.00e-8 / 1.55e-8 | 1.12 / 4.82 | 4.17e-6 / 4.61e-6 | 0.682 / 1.32 | 0.934 / 1.60 | 0.00225 (+2.5%) | 0.00969 (+10.7%) |
| mantaflow | 256³ | 5843 (5579–6320), 1 run | 3755.8 | 9.12e-5 / 5.25e-5 | 500061 / 1927015 | 0.0313 / 0.0496 | 6.26e-8 / 2.57e-8 | 2.97 / 9.07 | 5.93e-6 / 4.70e-6 | 0.775 / 1.51 | 1.05 / 1.79 | -0.00206 (-2.6%) | 0.00434 (+5.5%) |

## `plume_wind`

| solver | cells | frame ms (median, min–max) | peak MiB | div. RMS 1/s (60 / 120) | measured cells (60 / 120) | kinetic energy m⁵/s² (60 / 120) | KE per cell m⁵/s² (60 / 120) | vorticity m³/s (60 / 120) | vorticity per cell m³/s (60 / 120) | centroid m (60 / 120) | top m (60 / 120) | drift at 80 (% of mass at 60) | drift at 120 (% of mass at 60) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ember | 64³ | 18.9 (14.0–84.8), median of 3 | 24.2 | 0.142 / 0 | 6624 / 0 | 0.154 / 0 | 2.33e-5 / — | 0.691 / 0 | 1.04e-4 / — | 0.915 / 1.71 | 1.42 / 1.98 | 0.0248 (+30.2%) | 0.0403 (+49.2%) |
| mantaflow | 64³ | 96.9 (85.0–171), median of 3 | 73.4 | 1.22e-4 / 8.89e-5 | 17061 / 35182 | 0.110 / 0.109 | 6.44e-6 / 3.09e-6 | 2.84 / 5.57 | 1.66e-4 / 1.58e-4 | 0.875 / 1.53 | 1.30 / 1.95 | 0.00744 (+8.3%) | 0.0532 (+59.7%) |
| ember | 128³ | 98.1 (95.0–105), median of 3 | 192.8 | 1.40 / 0 | 49083 / 0 | 0.298 / 0 | 6.06e-6 / — | 0.792 / 0 | 1.61e-5 / — | 0.904 / — | 1.37 / — | -0.00329 (-6.1%) | -0.00328 (-6.1%) |
| mantaflow | 128³ | 559 (472–838), median of 3 | 405.6 | 3.90e-5 / 3.46e-5 | 89595 / 208081 | 0.0990 / 0.0916 | 1.11e-6 / 4.40e-7 | 3.00 / 6.87 | 3.35e-5 / 3.30e-5 | 0.914 / 1.60 | 1.32 / 1.96 | 0.00478 (+5.6%) | 0.0340 (+40.0%) |
| ember | 256³ | 761 (749–816), 1 run | 1539.0 | 2.41 / 3.32 | 264789 / 1986 | 0.127 / 0.0176 | 4.81e-7 / 8.88e-6 | 1.46 / 0.272 | 5.50e-6 / 1.37e-4 | 0.724 / 1.25 | 1.10 / 1.89 | -0.0298 (-69.7%) | -0.0333 (-77.9%) |
| mantaflow | 256³ | 3841 (3528–5449), 1 run | 2858.3 | 8.41e-5 / 7.46e-5 | 515032 / 1084920 | 0.0847 / 0.0596 | 1.65e-7 / 5.50e-8 | 2.96 / 6.87 | 5.74e-6 / 6.33e-6 | 0.930 / 1.55 | 1.36 / 1.96 | -1.56e-4 (-0.2%) | 0.0170 (+21.5%) |

## Notes

- **Velocity metrics** (divergence, kinetic energy, vorticity) cover only measured cells: those whose whole 3×3×3 neighbourhood is inside the domain, outside any collider and holds smoke (density > 1e-6). The rule is the same for both solvers, because Mantaflow's cache stores velocity only where there is smoke (spec §4.1). The measured-cell count says how much of each field that is.
- **Per-cell values.** The solvers' smoky regions differ in size, so their measured cells differ too, and the kinetic energy and vorticity totals partly measure that size. The totals divided by the measured-cell count are the fairer comparison.
- **Wind.** Mantaflow's wind field acts only on cells that hold smoke; Ember's wind accelerates every cell. `plume_wind` compares the plume's shape, not a matched force field.
- **Heat.** Mantaflow's emitter heat is held at a set value (Ember's emitter-centre heat at frame 24), not added at Ember's rate, so Mantaflow's emitter is hotter before frame 24 and cooler after it. Plume centroid and top carry that difference. Emitted mass matches within 3%.
- **Pressure.** Mantaflow solves with multigrid-preconditioned conjugate gradients to a tolerance; Ember runs a fixed count of red-black Gauss–Seidel iterations.
- **Drift** is (mass below the outflow plane + outflow since frame 60) − that mass at frame 60, with outflow estimated at frame resolution as the net upwind flux through the z-faces two cells below the top, and mass summed over the layers below them (spec §4.3). Emission stops after frame 60, so a perfect solver drifts 0. Drift at 80 is almost entirely mass gained by advection, since little or no smoke has reached the outflow plane by then (in the 64³ smoke test, neither solver had outflow before frame 74; each run's CSV `outflow_rate` column shows when it starts). Drift at 120 also covers the outflow period, where the frame-resolution outflow estimate adds uncertainty.
- **Frame times** exclude frame 1. Ember's frame is `eval_frame` plus a blocking GPU wait; Mantaflow's is the difference between consecutive cache files' modification times, so it includes writing the cache. 256³ is one run of each solver, the other resolutions the median of three runs' medians; the min–max range pools every timed frame.
- **Peak memory.** Ember's is the field pool's allocated bytes, with the frame cache off. Mantaflow's is Blender's peak resident memory while baking, minus the same scene's peak without a bake.
