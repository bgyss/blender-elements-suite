# Ember pressure-iteration sweep (exploration, not the gate)

- Machine: Apple M1 Max (Apple M1 Max)
- OS: macOS 27.0
- Ember commit: 5374cca
- Date: 2026-09-22
- Scene: `plume`, 128³, substeps 1. Frames 25–48 timed after 24 warm-up frames, as `eval` plus a blocking poll; median of 3 runs' medians. The step ms min–max range is pooled over all timed frames of all 3 runs, not a single run. Divergence from the frame-48 state.

| N | step ms (median, min–max) | snapshot ms | RMS div before | RMS div after | ratio | max div after | pass |
|---|---|---|---|---|---|---|---|
| 160 | 34.07 (33.65–36.06) | 0.77 | 1.413e-1 | 6.055e-3 | 0.0429 | 9.046e-2 | yes |
| 240 | 49.29 (48.73–74.87) | 0.77 | 1.414e-1 | 3.214e-3 | 0.0227 | 4.073e-2 | yes |
| 320 | 63.99 (63.67–69.71) | 0.76 | 1.414e-1 | 2.032e-3 | 0.0144 | 2.223e-2 | yes |
| 400 | 79.34 (78.74–126.53) | 0.76 | 1.415e-1 | 1.427e-3 | 0.0101 | 1.366e-2 | yes |
| 480 | 94.40 (93.81–101.14) | 0.76 | 1.415e-1 | 1.077e-3 | 0.0076 | 9.113e-3 | yes |
| 560 | 109.71 (108.66–117.65) | 0.76 | 1.415e-1 | 8.527e-4 | 0.0060 | 6.464e-3 | no |
| 640 | 125.24 (123.96–135.23) | 0.78 | 1.415e-1 | 6.956e-4 | 0.0049 | 4.777e-3 | no |

Pre-registered rule (spec §4.3): PASS if some N has a median step of at most 100 ms and a ratio of at most 0.1; the provisional default is the largest passing N. The `pass` column applies this same per-row rule (`GateRow::passes`).

Rule applied: **PASS**, provisional `pressure_iterations` = 480.

Note: the ratio measures residual divergence, which weights high frequencies. The smooth pressure error Gauss–Seidel leaves behind shows in plume shape, not in this number.

The gate's own record is `speed-gate.md`; this sweep only informs 2b's presets.

Conditions (added by hand after the run): the 1-minute load average was 20 at the
start and 7 at the end, much quieter than the gate's run (57 to 116). The N = 160 row
reproduces the gate's divergence columns bit for bit, and its step time within 5%
(34.1 ms vs 35.7 ms under load). Step time is about 3.6 ms + 0.19 ms per iteration.
The residual ratio falls roughly as N^-1.5, so each doubling of N buys about a
2.8-fold reduction.
