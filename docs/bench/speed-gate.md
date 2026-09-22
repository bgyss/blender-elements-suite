# Ember speed gate (piece 2a)

- Machine: Apple M1 Max (Apple M1 Max)
- OS: macOS 27.0
- Ember commit: a2f07f0
- Date: 2026-09-22
- Scene: `plume`, 128³, substeps 1. Frames 25–48 timed after 24 warm-up frames, as `eval` plus a blocking poll; median of 3 runs' medians. The step ms min–max range is pooled over all timed frames of all 3 runs, not a single run. Divergence from the frame-48 state.

| N | step ms (median, min–max) | snapshot ms | RMS div before | RMS div after | ratio | max div after | pass |
|---|---|---|---|---|---|---|---|
| 20 | 7.74 (7.27–9.21) | 0.78 | 1.627e-1 | 1.095e-1 | 0.6727 | 1.817e0 | no |
| 40 | 11.88 (11.21–17.69) | 0.79 | 1.443e-1 | 4.354e-2 | 0.3018 | 7.899e-1 | no |
| 80 | 19.92 (18.75–21.42) | 0.76 | 1.415e-1 | 1.668e-2 | 0.1179 | 2.955e-1 | no |
| 160 | 35.66 (33.90–52.86) | 0.78 | 1.413e-1 | 6.055e-3 | 0.0429 | 9.046e-2 | yes |

Pre-registered rule (spec §4.3): PASS if some N has a median step of at most 100 ms and a ratio of at most 0.1; the provisional default is the largest passing N. The `pass` column applies this same per-row rule (`GateRow::passes`).

Rule applied: **PASS**, provisional `pressure_iterations` = 160.

Note: the ratio measures residual divergence, which weights high frequencies. The smooth pressure error Gauss–Seidel leaves behind shows in plume shape, not in this number.

Conditions (added by hand after the run): the machine was not idle. The 1-minute
load average was 57 at the start and 116 at the end, mostly Backblaze uploading,
WindowServer and other desktop apps. Load can only slow the timings, and the
divergence columns do not depend on it, because the solver is deterministic. So the
verdict holds; the timings are an upper bound, not a clean measurement. Step time
grows by about 0.2 ms per iteration above roughly 3.6 ms of fixed cost.

Decision (recorded by the user, 2026-09-22): **PASS accepted; the default
`pressure_iterations` is 160.** A follow-up sweep (`iteration-sweep.md`) found the
100 ms ceiling at 480, but 160 was kept as the interactive default for its roughly
threefold headroom. Higher counts are for piece 2b's offline presets.
