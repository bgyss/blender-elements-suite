# Latency: a parameter change to frame N

- Machine: Apple M1 Max
- OS: macOS 27.2
- Ember commit: 60f13a3
- Blender: 5.2.2 LTS
- Date: 2026-09-25

| run | load before | load after | above 2 |
|---|---|---|---|
| `latency-ember-plume-128` | 5.85 | 6.01 | **yes** |
| `latency-mantaflow-plume-128` | 7.98 | 14.96 | **yes** |
| `latency-ember-plume_collider-128` | 14.96 | 9.40 | **yes** |
| `latency-mantaflow-plume_collider-128` | 10.71 | 14.17 | **yes** |
| `latency-ember-plume_wind-128` | 14.17 | 9.23 | **yes** |
| `latency-mantaflow-plume_wind-128` | 10.11 | 13.66 | **yes** |

Timings from a run whose 1-minute load average was above 2 before or after its timed runs should be redone on an idle machine. Such runs: `latency-ember-plume-128`, `latency-mantaflow-plume-128`, `latency-ember-plume_collider-128`, `latency-mantaflow-plume_collider-128`, `latency-ember-plume_wind-128`, `latency-mantaflow-plume_wind-128`.

## `plume` (128³)

Ember's time to its first frame, graph construction included, is the N = 1 median: 0.0657 s.

| N | Ember s (median) | Mantaflow s (median) | Mantaflow / Ember |
|---|---|---|---|
| 1 | 0.0657 | 0.691 | 10.5× |
| 24 | 1.63 | 11.8 | 7.22× |
| 60 | 4.09 | 29.0 | 7.10× |
| 120 | 8.19 | 56.0 | 6.84× |

## `plume_collider` (128³)

Ember's time to its first frame, graph construction included, is the N = 1 median: 0.101 s.

| N | Ember s (median) | Mantaflow s (median) | Mantaflow / Ember |
|---|---|---|---|
| 1 | 0.101 | 0.958 | 9.50× |
| 24 | 2.20 | 16.3 | 7.41× |
| 60 | 5.49 | 39.6 | 7.21× |
| 120 | 11.0 | 78.8 | 7.14× |

## `plume_wind` (128³)

Ember's time to its first frame, graph construction included, is the N = 1 median: 0.0750 s.

| N | Ember s (median) | Mantaflow s (median) | Mantaflow / Ember |
|---|---|---|---|
| 1 | 0.0750 | 0.712 | 9.49× |
| 24 | 1.65 | 11.8 | 7.15× |
| 60 | 4.11 | 29.3 | 7.14× |
| 120 | 8.22 | 58.2 | 7.07× |

## Notes

- **The change.** Every run changes the emitter's density (Ember's `density_rate`, Mantaflow's flow density, each ×(1 + 0.01·(run + 1))), so no run can reuse an earlier result. Each is the median of its runs (5 at N = 1, 5 at N = 24, 5 at N = 60, 5 at N = 120 for Ember; 5 at N = 1, 5 at N = 24, 3 at N = 60, 3 at N = 120 for Mantaflow).
- **Ember** runs in one warm process, as a live daemon does: one GPU device, registry and pipeline cache throughout, warmed by one untimed evaluation of the unchanged scene to frame 24. Each run times graph construction plus `eval_frame` for frames 1..N in a fresh field pool and state, ending with a blocking GPU wait. Nothing is read back or written to disk during the timed interval.
- **Mantaflow's time is a re-bake in an open Blender with its cache freed**, excluding Blender's startup and the scene's construction. After one untimed bake of the unchanged scene to frame 24, each run frees the cache (`fluid.free_all`, checked to leave no data file), sets the cache's last frame to N and times `fluid.bake_all` from just before the call to frame N's data file existing: the later of the call returning and the file's modification time. The time therefore includes Mantaflow writing N uncompressed 128³ OpenVDB files, one per frame with every grid, as frame times in `results.md` include writing the cache. These runs predate recording how long `bake_all` returns after frame N's file is written (`return_after_write_s`), so how much the call's return adds is not known here. A run whose frame N file matches another run's, apart from the VDB header's random UUID, is refused.
- **Loads are not idle readings, and every figure here needs an idle rerun.** Each is the 1-minute load average. Mantaflow's `load_before` is taken right after its own warm-up bake, and its `load_after` after up to 80 s of multi-threaded baking, so both include Mantaflow's own work. Each Ember run after the first starts right after the previous scene's Blender run, so its `load_before` inherits that load (Ember `plume_collider`'s 14.96 is exactly Mantaflow `plume`'s `load_after`; Ember `plume_wind`'s 14.17 is exactly Mantaflow `plume_collider`'s `load_after`). There was outside load too: before the recipe's load wait, `vm.loadavg` (1, 5 and 15 minutes) was `{ 12.62 13.15 13.46 }`. Mantaflow is CPU-bound and Ember mostly GPU-bound, so CPU contention probably slows Mantaflow more than Ember and may inflate the ratio in Ember's favour: neither the seconds nor the ratios should be read as idle figures.
