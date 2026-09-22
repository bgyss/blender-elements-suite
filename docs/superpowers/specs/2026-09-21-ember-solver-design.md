# Ember Piece 2 — Solver Design

**Date:** 2026-09-21
**Status:** DRAFT. Only §5 (benchmarking against Mantaflow) is written. The rest of
this spec is written in piece 2's brainstorming session, and §5 constrains it.
**Parent:** `2026-09-21-ember-design.md` (piece 2 of 4)

## 1. Goal

To be written in brainstorming. The umbrella fixes the targets:

- the solver step sequence (umbrella §3)
- the validation scenes (umbrella §6)
- 128³ at 10 fps or better interactively, and 256³–512³ offline

## 2–4. Components, data flow, testing

To be written in brainstorming.

---

## 5. Benchmarking against Blender's stock solver

### 5.1 Why

Blender's smoke and fire solver is **Mantaflow**, which runs on the CPU. The suite
spec lists replacing it for existing workflows as a non-goal: Ember's claim is
interactive authoring, not a better offline solver. Every claim must still be a
number, measured the same way on both solvers. "It feels faster" and "it looks
better" are not results.

Ember expects to win on **speed and interactivity**. v1's semi-Lagrangian
advection and fixed-iteration Gauss–Seidel pressure solve are not expected to
beat Mantaflow on **detail**. The benchmark must be able to show that honestly,
and the results table must say so when it is true.

### 5.2 What exists to build on

Verified in the local Blender 5.2.2 LTS (`--background --factory-startup`, via
the Python API):

- The fluid modifier has `domain_type` `GAS`, `resolution_max`,
  `use_adaptive_domain`, `alpha` / `beta` (buoyancy), `vorticity`,
  `use_dissolve_smoke`, `use_noise`, `cfl_condition`, `timesteps_min`,
  `timesteps_max`, `use_adaptive_timesteps` and `time_scale`.
- `cache_data_format` offers `OPENVDB`, and `openvdb_cache_compress_type` offers
  `NONE`. The latter matters: `vdb-rs`, the read-back oracle, is not known to
  read ZIP- or Blosc-compressed grids.
- `bpy.ops.fluid.bake_all` and `bpy.ops.fluid.bake_data` exist.

**Not yet verified, and the first benchmark task must prove each one:**

- that `bake_all` runs headlessly (it may need a context override)
- the grid names Mantaflow writes (for example whether velocity is `velocity` or `vel`)
- where velocity is stored: cell centres or faces
- that `vdb-rs` can read an uncompressed Mantaflow cache

If `vdb-rs` cannot read it, the fallback is a small Blender-side Python exporter
that writes the needed statistics directly. That is GPL-side code, and it lives
under `tests/bench/`, never in a crate.

### 5.3 Matched scenes

Each scene is defined once, as data: domain, emitter, forces, resolution and frame
count. From that one definition come an `.elements` document and a Blender script
that builds the equivalent Mantaflow domain. Neither is written by hand.

| Scene | Content | Stresses |
|---|---|---|
| `plume` | a hot, dense sphere emitter at the domain floor, buoyancy only | advection, projection, buoyancy |
| `plume_collider` | `plume` with a sphere collider above the emitter | solid boundaries |
| `plume_wind` | `plume` with constant horizontal wind | long-distance advection, dissipation |

Each scene runs at 64³, 128³ and 256³, for 120 frames at 24 fps.

**Fairness rules**, applied to Mantaflow and recorded with every result:

- `use_noise = False`: no wavelet upres, which v1 does not have
- `use_adaptive_domain = False`
- `use_adaptive_timesteps = False`, with `timesteps_min = timesteps_max` equal to Ember's substep count
- the same emitter position, radius, density and temperature
- the same buoyancy coefficients: Mantaflow's `alpha` / `beta`, and Ember's own names, mapped explicitly and documented
- the same vorticity strength, with dissolve off unless the scene says otherwise
- the cache format is `OPENVDB` with `NONE` compression, so cache I/O does not dominate the timing

Parameter meanings do not map exactly between the solvers. Every mapping is
written down next to the scene definition, and a scene is compared only on the
metrics in §5.4, never voxel for voxel.

### 5.4 Metrics

**Speed:**
- Wall-clock seconds per frame. Take the median of 3 runs, excluding the first
  frame, which pays for shader compilation and cache setup.
- Peak memory: resident memory for Mantaflow, allocated textures plus cache for Ember.

**Interactivity** (Ember's actual claim):
- Latency from a parameter change to the first updated frame visible, at 128³.
- For Mantaflow, this is the time to re-bake up to the current frame.

**Physical quality.** These are computed by the same code on both solvers'
output, after resampling velocity to cell centres:
- Max and RMS **divergence** after projection (central differences)
- **Total density** over time: mass conservation, with emission accounted for
- **Kinetic energy** and **total vorticity magnitude** over time: numerical dissipation
- **Plume height**: the density-weighted centroid z, and the 95th-percentile top of density above a threshold, over time

**Visual.** Both VDB sequences are rendered with one Cycles setup (same camera, lights
and volume shader) and shown side by side. This is judged by eye and recorded
as a note, not a number.

### 5.5 The speed gate

This comes first, and decides whether piece 2 continues as designed.

As soon as advection and projection kernels exist, before emitters, colliders
or add-on work, measure Ember's step time at 128³ with the default quality preset
on this machine (Apple Silicon, Metal).

- **Pass:** a full step, including every substep, takes ≤ 100 ms, which gives
  ≥ 10 fps. Continue.
- **Fail:** stop and bring the numbers to the user before writing more solver code.
  The options then are:
  - fewer pressure iterations, and what that costs in divergence
  - the multigrid stretch goal
  - a lower interactive resolution

This is the umbrella's highest risk ("wgpu compute on Metal is too slow"), checked while it is still cheap to change course.

### 5.6 Tooling

- `just bench` runs every scene at every resolution on both solvers and writes
  results. It is **not** part of `just check`: it takes minutes, and it needs Blender.
- `just bench-render` renders the side-by-side comparison.
- Results are committed to `docs/bench/results.md` as one table per scene. Each
  table records the machine, OS, Ember commit, Blender version and date, so
  numbers from different runs are never mixed without their context.
- Timing numbers are for comparison on one machine. They never feed pass/fail
  tests, except for the speed gate in §5.5, which is a recorded human decision,
  not a CI check.

### 5.7 Out of scope

- Liquids (Mantaflow's `LIQUID` domain), which belong to Tide.
- Comparisons against JangaFX EmberGen. It is closed-source, and its output
  cannot be driven headlessly under the same fairness rules.
- Automatic statistical significance testing beyond medians of 3 runs.
