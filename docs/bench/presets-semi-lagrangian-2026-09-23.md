# Ember preview preset sweep, semi-Lagrangian, 2026-09-23 (exploration for risk (j))

- Machine: Apple M1 Max (Apple M1 Max)
- OS: macOS 27.0
- Ember commit: 19cbbe0-dirty
- Date: 2026-09-23
- Scene: `plume`, 128³, N = 160, cfl 1.0, advection SemiLagrangian, vorticity 0. Frames 25–48 timed after 24 warm-up frames, each as `eval_frame` (the CFL measurement and every substep) plus a blocking wait; median of 3 runs' medians. The min–max range is pooled over all timed frames of all runs.

| max_substeps | frame ms (median, min–max) | frames CFL-clamped | pass |
|---|---|---|---|
| 1 | 54.61 (53.77–55.81) | 72 of 72 | yes |
| 2 | 107.31 (105.92–124.19) | 45 of 72 | no |
| 3 | 176.87 (116.92–181.39) | 15 of 72 | no |
| 4 | 178.78 (117.62–242.70) | 0 of 72 | no |

Pre-registered rule (2b-1 spec §6): preview's `max_substeps` is the largest cap whose median full frame is at most 100 ms. If even 1 fails, preview falls back to semi-Lagrangian advection and the sweep runs again.

Rule applied: **preview `max_substeps` = 1**.

No decision: this is exploration, not the rule's fallback. Preview stays at
MacCormack with `max_substeps` = 1 (`presets.md`). The run isolates MacCormack's
cost: a cap-1 frame takes 54.61 ms here against 91.77 ms with MacCormack.

Conditions (added by hand after the run): the 1-minute load average was 10.13 at the
start and 9.37 at the end, so the timings are upper bounds.
