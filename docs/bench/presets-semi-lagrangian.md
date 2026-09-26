# Ember preview preset sweep (piece 2b-1)

- Machine: Apple M1 Max (Apple M1 Max)
- OS: macOS 27.2
- Ember commit: 1f6f0bd-dirty
- Date: 2026-09-26
- Scene: `plume_collider`, 128³, preview's pressure solve (MGPCG ×4), mass correction on, cfl 1.0, advection SemiLagrangian, vorticity 0. Frames 25–48 timed after 24 warm-up frames, each as `eval_frame` (the CFL measurement and every substep) plus a blocking wait; median of 3 runs' medians. The min–max range is pooled over all timed frames of all runs.

| max_substeps | frame ms (median, min–max) | frames CFL-clamped | pass |
|---|---|---|---|
| 1 | 67.75 (48.18–75.15) | 72 of 72 | yes |
| 2 | 129.47 (90.58–151.92) | 0 of 72 | no |
| 3 | 130.53 (93.20–144.12) | 0 of 72 | no |
| 4 | 134.46 (93.65–155.00) | 0 of 72 | no |

Pre-registered rule (2b-1 spec §6): preview's `max_substeps` is the largest cap whose median full frame is at most 100 ms. If even 1 fails, preview falls back to semi-Lagrangian advection and the sweep runs again.

Rule applied: **preview `max_substeps` = 1**.

Load average (1, 5, 15 minutes): { 6.39 49.04 51.90 } before the run, { 6.59 42.46 49.30 } after it.

Decision (recorded by the user): _pending_
