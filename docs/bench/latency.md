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
- **Ember** runs in one warm process, as a live daemon does: one GPU device, registry and pipeline cache throughout, warmed by one untimed evaluation of the unchanged scene to frame 24. Each run times graph construction plus `eval_frame` for frames 1..N in a fresh field pool and state, ending with a blocking GPU wait.
- **Mantaflow's time is a re-bake in an open Blender with its cache freed**, excluding Blender's startup and the scene's construction. After one untimed bake of the unchanged scene to frame 24, each run frees the cache (`fluid.free_all`, checked to leave no data file), sets the cache's last frame to N and times `fluid.bake_all` from just before the call to frame N's data file existing: the later of the call returning and the file's modification time. Frame time therefore includes Mantaflow writing its cache, as in `results.md`. A run whose frame N file matches another run's, apart from the VDB header's random UUID, is refused.
- **Loads** are the 1-minute load average, taken after the warm-up and after the last run.
