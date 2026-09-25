# Ember solver gate (2b-3c)

- Machine: Apple M1 Max (Apple M1 Max)
- OS: macOS 27.2
- Ember commit: 6d43473
- Date: 2026-09-24 15:02 UTC
- Load average (1, 5, 15 min) before: { 18.03 13.46 12.04 } — **above 2: the machine was loaded**
- Load average after: { 1.79 3.02 4.76 }
- Kernel path check: `plume_plate` 128³ with `mgpcg ×10` through frame 65 (frame 60 instrumented), against `eval_frame`: 0 faces differ, max |Δ| 0e0.
- Timing phase rerun per solve (spec §4): commit 4615095, 2026-09-24 17:01 UTC

## The rule (spec §4, fixed before any numbers)

A configuration passes when, at 128³ and 256³ and in `plume`, `plume_collider` and `plume_wind`, its pressure-solve time per substep (the median over frames 25–48 of the time from before the solve to a blocking wait after it; the median of the runs' medians) is at most Gauss–Seidel ×160's in the same process, and its masked divergence RMS at frames 60 and 120 is at most Mantaflow's for that scene and resolution (`docs/bench/results-2b3/results.md`, the 7fe5d9d run; `plume_wind` uses `plume`'s until the rerun). It must also pass `plume_plate` at 128³: RMS divergence after projection over before it, at frames 60 and 120, through the kernel API, at most 1e-3. The chosen configuration must be stable past the floor: in `plume_collider` at 128³ the masked divergence after 40 iterations is at most 4× that after the chosen count. Plain V-cycles (`multigrid`) are measured but cannot be the default. The fastest passing configuration becomes the default; if none passes, work stops.

Mantaflow references (masked RMS, f60 / f120): `plume` and `plume_wind` 1.49e-4 / 1.07e-4 at 128³, 7.41e-5 / 4.51e-5 at 256³; `plume_collider` 2.84e-4 / 1.09e-4 at 128³, 9.12e-5 / 5.25e-5 at 256³.

## Procedure

Staged sweep: at 128³, counts rise from 1 (`multigrid` to 8, `mgpcg` to 24) until the three scenes and `plume_plate` all pass; that count is checked at 256³ and stepped up until it passes there too (and still at 128³). A 256³ count stops at its first failing scene (— below). Divergence runs read back only at frames 60 and 120. Then each candidate and Gauss–Seidel ×160 are timed per solve, with MGPCG ×4 and ×20 for information only: 5 runs of frames 1–48 each, through the kernel API path the plate uses (checked bit-identical to `eval_frame`), each run's median over frames 25–48. The runs rotate the order of the configurations (run r starts with the r-th), so every configuration takes every position. The timed interval starts from an idle GPU, with stages 1–3 submitted and waited for and the substep's uniforms built, and ends at a blocking wait after the projection. It covers recording and running the divergence, the multigrid hierarchy build (multigrid and MGPCG), the solve and the gradient subtraction, for the frame's first (here only) substep. Frame time, from before the emitter fill to a blocking wait after the last substep, is kept as a secondary column. It includes the split's extra waits, so it is a little above `eval_frame`'s.

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

Timing phase load average: { 20.72 14.37 10.38 } — **above 2: the machine was loaded** at the start; { 8.40 9.96 10.52 } — **above 2: the machine was loaded** when timing began (after waiting 600 s); { 40.01 39.36 26.89 } — **above 2: the machine was loaded** after.

Per-solve time, ms: the median of the 5 runs' medians (min–max of the run medians). ✓ / ✗: at most / above Gauss–Seidel's, the rule's timing clause. "(info)" columns are not part of the rule.

| res | scene | `gauss_seidel ×160` solve ms | `mgpcg ×10` solve ms | `mgpcg ×4` (info) solve ms | `mgpcg ×20` (info) solve ms |
|---|---|---|---|---|---|
| 128³ | plume | 38.25 (37.99–39.71) | 42.83 (42.55–53.94) ✗ | 18.53 (18.00–19.41) | 83.39 (82.43–86.25) |
| 128³ | plume_collider | 73.84 (72.03–96.69) | 104.52 (103.05–106.06) ✗ | 43.50 (43.28–47.71) | 205.19 (201.87–213.03) |
| 128³ | plume_wind | 60.33 (58.68–61.46) | 60.94 (46.92–65.25) ✗ | 25.13 (21.84–26.28) | 108.53 (89.51–127.27) |
| 256³ | plume | 434.82 (425.13–456.25) | 359.56 (343.98–375.78) ✓ | 145.71 (141.78–152.56) | 701.52 (695.62–810.31) |
| 256³ | plume_collider | 843.81 (819.45–885.89) | 839.18 (809.37–856.82) ✓ | 343.38 (332.64–359.38) | 1662.61 (1648.17–1701.41) |
| 256³ | plume_wind | 510.24 (495.08–547.64) | 488.11 (441.99–510.64) ✓ | 190.71 (180.14–201.93) | 905.11 (818.04–931.98) |

Every run's per-solve median, in run order:

| res | scene | configuration | solve ms, each run's median (run 1 … run n) |
|---|---|---|---|
| 128³ | plume | `gauss_seidel ×160` | 38.16, 38.25, 37.99, 39.45, 39.71 |
| 128³ | plume | `mgpcg ×10` | 42.55, 42.83, 42.58, 44.23, 53.94 |
| 128³ | plume | `mgpcg ×4` | 18.53, 18.20, 18.00, 18.77, 19.41 |
| 128³ | plume | `mgpcg ×20` | 86.25, 82.43, 83.10, 83.43, 83.39 |
| 128³ | plume_collider | `gauss_seidel ×160` | 73.84, 75.12, 73.35, 72.03, 96.69 |
| 128³ | plume_collider | `mgpcg ×10` | 105.75, 103.06, 103.05, 106.06, 104.52 |
| 128³ | plume_collider | `mgpcg ×4` | 43.28, 43.31, 43.50, 47.71, 45.18 |
| 128³ | plume_collider | `mgpcg ×20` | 205.19, 201.87, 201.93, 213.03, 205.90 |
| 128³ | plume_wind | `gauss_seidel ×160` | 58.74, 61.46, 60.33, 60.41, 58.68 |
| 128³ | plume_wind | `mgpcg ×10` | 46.92, 65.25, 60.94, 59.03, 61.99 |
| 128³ | plume_wind | `mgpcg ×4` | 24.88, 26.08, 26.28, 25.13, 21.84 |
| 128³ | plume_wind | `mgpcg ×20` | 108.53, 113.09, 127.27, 107.86, 89.51 |
| 256³ | plume | `gauss_seidel ×160` | 456.25, 433.65, 434.82, 425.13, 439.40 |
| 256³ | plume | `mgpcg ×10` | 375.78, 359.56, 355.03, 343.98, 363.14 |
| 256³ | plume | `mgpcg ×4` | 145.81, 145.07, 145.71, 141.78, 152.56 |
| 256³ | plume | `mgpcg ×20` | 713.40, 695.62, 701.52, 698.48, 810.31 |
| 256³ | plume_collider | `gauss_seidel ×160` | 848.36, 843.81, 885.89, 819.45, 827.05 |
| 256³ | plume_collider | `mgpcg ×10` | 831.12, 839.18, 856.82, 809.37, 843.72 |
| 256³ | plume_collider | `mgpcg ×4` | 344.85, 343.38, 341.88, 332.64, 359.38 |
| 256³ | plume_collider | `mgpcg ×20` | 1660.27, 1648.17, 1682.28, 1662.61, 1701.41 |
| 256³ | plume_wind | `gauss_seidel ×160` | 495.08, 509.46, 538.15, 547.64, 510.24 |
| 256³ | plume_wind | `mgpcg ×10` | 448.58, 441.99, 490.23, 488.11, 510.64 |
| 256³ | plume_wind | `mgpcg ×4` | 180.14, 183.41, 190.71, 201.93, 191.51 |
| 256³ | plume_wind | `mgpcg ×20` | 818.04, 859.30, 905.11, 931.98, 909.57 |

Frame time, ms (secondary, not the rule), the same statistic:

| res | scene | `gauss_seidel ×160` frame ms | `mgpcg ×10` frame ms | `mgpcg ×4` (info) frame ms | `mgpcg ×20` (info) frame ms |
|---|---|---|---|---|---|
| 128³ | plume | 72.55 (71.86–75.48) | 77.10 (76.43–96.77) | 52.48 (51.40–56.07) | 117.72 (116.69–121.99) |
| 128³ | plume_collider | 111.17 (108.41–142.54) | 150.29 (148.50–153.16) | 89.26 (88.57–99.77) | 252.06 (247.23–259.59) |
| 128³ | plume_wind | 109.94 (107.29–112.80) | 109.52 (85.17–116.33) | 71.62 (61.74–76.82) | 151.41 (125.16–179.54) |
| 256³ | plume | 749.74 (733.47–792.71) | 672.38 (642.65–701.44) | 453.79 (442.29–472.45) | 1009.44 (1004.88–1181.58) |
| 256³ | plume_collider | 1205.90 (1171.30–1266.95) | 1254.98 (1198.13–1291.13) | 749.80 (718.24–783.20) | 2096.72 (2069.05–2150.72) |
| 256³ | plume_wind | 925.43 (892.50–985.58) | 935.38 (841.69–977.08) | 616.70 (581.96–658.10) | 1333.41 (1183.08–1384.19) |

## Past-floor stability (`plume_collider`, 128³)

- `mgpcg`: ×40 over the candidate count, frames 60 / 120: 1.010 / 1.000 — stable

## Rule applied

- `mgpcg ×10`: divergence and thin plate pass; per-solve time ABOVE Gauss–Seidel's in `plume` at 128³, `plume_collider` at 128³, `plume_wind` at 128³; past-floor stable

Rule applied: **none passes**: no candidate meets the per-solve timing clause (and the other clauses) in every scene at both resolutions. `plume_wind` constrained only frame 60: at frame 120 its smoke mask had no measured cells, so its RMS of 0 passed vacuously.

## After the rule: Gauss–Seidel ×160 against MGPCG ×4

Measured 2026-09-24 at commit 2b9acab-dirty (`SOLVER_GATE_COMPARE=1`), by the gate's own accuracy measures,
to decide the presets after the registered rule failed. It is not part of the rule.

| measure | res | frames 60 / 120: `gauss_seidel ×160` | `mgpcg ×4` |
|---|---|---|---|
| `plume` masked RMS | 128³ | 3.00e-2 / 7.56e-3 | 3.63e-5 / 2.17e-5 |
| `plume_collider` masked RMS | 128³ | 7.84e-3 / 3.63e-3 | 4.39e-5 / 2.52e-5 |
| `plume_wind` masked RMS | 128³ | 3.25e-4 / 0.00e0 | 1.92e-6 / 0.00e0 |
| `plume_plate` ratio | 128³ | 8.70e-3 / 6.45e-3 | 4.72e-3 / 1.23e-3 |
| `plume` masked RMS | 256³ | 2.51e-1 / 2.82e-2 | 6.66e-5 / 6.19e-5 |
| `plume_collider` masked RMS | 256³ | 7.12e-2 / 1.78e-2 | 1.09e-4 / 7.68e-5 |
| `plume_wind` masked RMS | 256³ | 2.22e-3 / 0.00e0 | 5.42e-6 / 0.00e0 |

MGPCG ×4 is at least as accurate as Gauss–Seidel ×160 on every row: on the thin plate by 1.8× (frame
60) and 5.2× (frame 120), and in the three scenes by two to four orders of magnitude. Gauss–Seidel ×160,
today's preview, misses Mantaflow's reference in every scene with smoke to measure. MGPCG ×4 meets it at
128³. At 256³ it misses by up to 1.5×: `plume` at frame 120 (6.19e-5 against 4.51e-5), and
`plume_collider` at frames 60 and 120 (1.09e-4 against 9.12e-5, 7.68e-5 against 5.25e-5). `plume_wind`'s
frame 120 has no measured cells for either solver.

*Provenance.* Every number above was measured before 63a7c7e, which makes
the multigrid prolongation normaliser exactly 1 where no corner is dropped
(it was 1.0000001 in f32 next to open faces). That changes results with a
collider at rounding level only, so the figures are not bit-reproducible
from later commits; the conclusions stand.

Decision (recorded by the user, 2026-09-24): the registered rule failed. MGPCG ×10's solve is 1–42% slower
than Gauss–Seidel's at 128³ (12% in `plume`, 42% in `plume_collider`, 1% in `plume_wind`). Preview uses
MGPCG ×4. Its solve takes 42–59% of Gauss–Seidel ×160's at 128³ and 34–41% at 256³. It meets Mantaflow's
divergence in all three scenes at 128³ and misses it by at most 1.5× at 256³, where Gauss–Seidel misses by
up to 3400×. It is more accurate than Gauss–Seidel on the thin plate (4.72e-3 / 1.23e-3 against
8.70e-3 / 6.45e-3), but misses the plate's 1e-3 target. Final uses MGPCG ×10, which passes every accuracy
check, the thin plate included.
