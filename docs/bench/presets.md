# Ember preview preset sweep (piece 2b-1)

- Machine: Apple M1 Max (Apple M1 Max)
- OS: macOS 27.0
- Ember commit: fe51fa6
- Date: 2026-09-23 (UTC; 2026-09-22 local)
- Scene: `plume`, 128³, N = 160, cfl 1.0, advection MacCormack, vorticity 0. Frames 25–48 timed after 24 warm-up frames, each as `eval_frame` (the CFL measurement and every substep) plus a blocking wait; median of 3 runs' medians. The min–max range is pooled over all timed frames of all runs.

| max_substeps | frame ms (median, min–max) | frames CFL-clamped | pass |
|---|---|---|---|
| 1 | 91.77 (91.28–93.90) | 72 of 72 | yes |
| 2 | 182.77 (181.59–184.88) | 51 of 72 | no |
| 3 | 272.31 (180.45–275.97) | 27 of 72 | no |
| 4 | 272.81 (181.28–366.02) | 0 of 72 | no |

Conditions: the machine was not idle. The 1-minute load average was 7.73 before the run (5- and 15-minute: 14.57, 13.47) and 6.19 after it (12.39, 12.74), with Backblaze's `bztransmit` holding about 99% CPU throughout, so these timings are upper bounds.

Pre-registered rule (2b-1 spec §6): preview's `max_substeps` is the largest cap whose median full frame is at most 100 ms. If even 1 fails, preview falls back to semi-Lagrangian advection and the sweep runs again.

Rule applied: **preview `max_substeps` = 1**.

Decision (recorded by the user, 2026-09-22): **preview `max_substeps` = 1 accepted.** Every timed frame wanted more substeps (72 of 72), so preview runs CFL-clamped and reports it. A full preview frame now costs about 92 ms against 2a's 34 ms at the same N; the ~58 ms increase is unexplained (MacCormack's three passes, RK2's extra samples, or the per-frame CFL readback) and is to be profiled before 2b-3's benchmark.

## Run of 2026-09-25 at ce881a5

- Machine: Apple M1 Max (Apple M1 Max)
- OS: macOS 27.2
- Ember commit: ce881a5
- Date: 2026-09-25
- Scene: `plume`, 128³, preview's pressure solve (MGPCG ×4), mass correction on, cfl 1.0, advection MacCormack, vorticity 0. Frames 25–48 timed after 24 warm-up frames, each as `eval_frame` (the CFL measurement and every substep) plus a blocking wait; median of 3 runs' medians. The min–max range is pooled over all timed frames of all runs.

| max_substeps | frame ms (median, min–max) | frames CFL-clamped | pass |
|---|---|---|---|
| 1 | 71.47 (67.89–72.55) | 72 of 72 | yes |
| 2 | 140.50 (133.04–161.25) | 48 of 72 | no |
| 3 | 206.06 (136.86–229.86) | 27 of 72 | no |
| 4 | 211.59 (135.25–306.24) | 0 of 72 | no |

Pre-registered rule (2b-1 spec §6): preview's `max_substeps` is the largest cap whose median full frame is at most 100 ms. If even 1 fails, preview falls back to semi-Lagrangian advection and the sweep runs again.

Rule applied: **preview `max_substeps` = 1**.

Load average (1, 5, 15 minutes): { 9.59 10.36 8.91 } before the run, { 9.51 10.10 8.91 } after it.

Result (2026-09-25, 2b-3c Task 8; not a new decision): **cap 1 still fits, so the 2026-09-22 decision stands.** Preview now solves with MGPCG ×4 and runs the mass correction, and one CFL-clamped substep costs 71.5 ms against 91.8 ms with Gauss–Seidel ×160, leaving about 28 ms of the 100 ms budget. Cap 2 (140.5 ms) still does not fit. The machine was loaded throughout (1-minute load average 9.6 before and 9.5 after), so these are upper bounds; the idle rerun of risk (k) is still owed.
