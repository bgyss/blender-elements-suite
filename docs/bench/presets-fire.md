# Ember fire preview frame (piece 2b-4)

- Machine: Apple M1 Max (Apple M1 Max)
- OS: macOS 27.2
- Ember commit: 8ecb78e-dirty
- Date: 2026-09-26
- Scene: `fire`, 128³, preview's pressure solve (MGPCG ×4), mass correction on, cfl 1.0, advection MacCormack, vorticity 0. Frames 25–48 timed after 24 warm-up frames, each as `eval_frame` (the CFL measurement and every substep) plus a blocking wait; median of 3 runs' medians. The min–max range is pooled over all timed frames of all runs.

| max_substeps | frame ms (median, min–max) | frames CFL-clamped | pass |
|---|---|---|---|
| 1 | 103.46 (80.54–109.04) | 72 of 72 | no |
| 2 | 203.89 (202.75–206.51) | 72 of 72 | no |
| 3 | 305.25 (302.44–313.18) | 72 of 72 | no |
| 4 | 404.72 (402.60–407.62) | 72 of 72 | no |

Recorded, not gated (2b-4 spec §6.4): if cap 1's median is above 100 ms, it becomes an open risk in piece 2's spec §6, alongside (k). The graph's output is the solver's density, so the flame output is not read; reading it adds one small submit per frame. The pass column applies 2b-1's rule for comparison only.

Load average (1, 5, 15 minutes): { 10.65 9.25 8.72 } before the run, { 6.78 8.10 8.33 } after it.

Notes (by hand): the commit is dirty only by `examples/presets.rs`, this
program's `PRESET_SCENE` switch; no scene or solver code differs. "Advection
MacCormack" is the scene's setting: while fire burns, the solver traces
velocity with Euler regardless (2b-4 spec, results.md Fire). Cap 1's median,
103.5 ms, is above preview's 100 ms, under load, and is recorded as risk (n)
in `docs/superpowers/specs/2026-09-21-ember-solver-design.md` §6. It agrees
with the benchmark's 104 ms fire frame at 128³ (`docs/bench/results.md`).
