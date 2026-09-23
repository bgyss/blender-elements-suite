# Ember pressure-iteration sweep, 2026-09-23 (exploration for risk (j), not the gate)

- Machine: Apple M1 Max (Apple M1 Max)
- OS: macOS 27.0
- Ember commit: 19cbbe0
- Date: 2026-09-23
- Scene: `plume`, 128³, max_substeps 1. Frames 25–48 timed after 24 warm-up frames, as `eval` plus a blocking poll; median of 3 runs' medians. The step ms min–max range is pooled over all timed frames of all 3 runs, not a single run. Divergence from the frame-48 state.

| N | step ms (median, min–max) | snapshot ms | RMS div before | RMS div after | ratio | max div after | pass |
|---|---|---|---|---|---|---|---|
| 20 | 46.26 (45.64–49.21) | 0.74 | 2.075e-1 | 1.257e-1 | 0.6055 | 2.162e0 | no |
| 160 | 105.06 (91.71–112.58) | 0.77 | 1.823e-1 | 6.446e-3 | 0.0354 | 9.311e-2 | no |

Pre-registered rule (spec §4.3): PASS if some N has a median step of at most 100 ms and a ratio of at most 0.1; the provisional default is the largest passing N. The `pass` column applies this same per-row rule (`GateRow::passes`).

Rule applied: **FAIL**: no N meets both limits.

Note: the ratio measures residual divergence, which weights high frequencies. The smooth pressure error Gauss–Seidel leaves behind shows in plume shape, not in this number.

The gate's own record is `speed-gate.md`, and 2a's sweep is `iteration-sweep.md`.
This run measures the code after 2b-1 (MacCormack, RK2, CFL readback), for risk (j)
in the piece 2 spec. The FAIL line applies the gate's rule and means nothing here.

Conditions (added by hand after the run): the 1-minute load average was 9.40 at the
start and 7.88 at the end, so the timings are upper bounds. A two-point fit gives
0.42 ms per pressure iteration and 37.9 ms of fixed cost from the medians (0.33 ms
and 39.1 ms from the fastest frames), against 2a's 0.19 ms and 3.6 ms.
