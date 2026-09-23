# Ember Piece 2b-3 — Mantaflow Benchmark Design

**Date:** 2026-09-23
**Status:** In progress. §4, §5, §6.2 and §8 revised on 2026-09-23 after
Task 3's findings (`docs/bench/mantaflow-notes.md`) and the user's decisions.
**Parent:** `2026-09-21-ember-solver-design.md` (piece 2), whose §5 is the
benchmark design this cycle builds. Follows 2b-2
(`2026-09-23-ember-scene-content-2b2-design.md`), which added the
`plume_collider` and `plume_wind` scenes.
**Branch base:** `main` at 58a71f5 (PR #6 merged).

## 1. Goal and scope

2b-3 turns piece 2 §5 into numbers: Ember and Blender's Mantaflow run the same
three scenes, and one piece of Rust computes every quality metric from both
solvers' fields. User decisions, 2026-09-23:

- **Numbers now, the rest later.** This cycle measures speed, peak memory and
  the physical-quality metrics of piece 2 §5.4 for `plume`, `plume_collider`
  and `plume_wind` at 64³, 128³ and 256³. The side-by-side Cycles render and
  the parameter-change latency measurement become 2b-3b.
- **Ember in-process, Mantaflow through its VDB cache.** Ember's fields are
  read back from the GPU into arrays. Mantaflow's uncompressed OpenVDB cache
  is read into the same layout. The same `metrics` functions run on both.
- **One timed run at 256³.** 64³ and 128³ report the median of 3 runs, as
  §5.4 says. 256³ reports one run, and the table says so next to the figure.
- **Emission stops at frame 60.** All three scenes emit for frames 1–60, and
  frames 61–120 measure mass drift with no sources (§4.3).

**Out of scope:**
- The Cycles side-by-side render and the latency measurement (2b-3b).
- Flame (2b-4).
- Any change to the solver itself. The only Ember changes are the emitter's
  `active_frames` (§3) and, if the field pool lacks one, a high-water mark
  (§4.4).
- Liquids, and Mantaflow's noise upres (disabled by the fairness rules).

## 2. Order of work

1. **Idle-machine preset rerun (risk (k)).** First move the bench scenes onto
   `ember.emitter` (§3), then run `just bench-presets` on an idle machine and
   record the result in `docs/bench/presets.md`. If a 128³ preview frame no
   longer fits 100 ms, stop and bring the numbers to the user before any
   benchmark work: the benchmark would otherwise measure a preset that is about
   to change. *The machine was never idle when this ran (load 6–19), so by the
   user's decision on 2026-09-23 the rerun moves to just before the full
   benchmark run, where the same stop rule applies.*
2. **Verify the Mantaflow unknowns** (piece 2 §5.2). Bake a small gas domain
   headlessly in the local Blender (5.2.2 LTS) and record in
   `docs/bench/mantaflow-notes.md`:
   - whether `bpy.ops.fluid.bake_all` runs under `--background` and what
     context it needs;
   - the grid names in the cache, and their value types;
   - whether velocity is stored at cell centres or on faces;
   - whether `vdb-rs` 0.6 reads the float grids and the vector velocity grid;
   - how grid index space maps onto the domain (origin, voxel size, any
     padding cells);
   - how to time each frame (expected: the cache files' modification times);
   - the parameter mappings of §5, found from Blender's source and checked by
     experiment;
   - Mantaflow's rate per frame at 256³, to confirm the run-time estimate.

   If `vdb-rs` cannot read the cache, the fallback is a Blender-side script
   that converts it to `.npy` with Blender's bundled OpenVDB Python module;
   `elements-io` already has `read_npy`. The same task commits a tiny 16³
   cache (a few KB) as the reader's test fixture. **If any finding changes
   this design, stop and bring it to the user.**
3. **Matched scenes** (§5).
4. **Metrics** (§4).
5. **Ember runner** (§6.1).
6. **Mantaflow reader** (§6.2).
7. **`just bench` and the results** (§7).

## 3. Emitter activity window

`ember.emitter` gains one parameter:

```rust
/// Frames on which the emitter emits, inclusive. `None` (the default) is
/// always on.
#[serde(default)]
pub active_frames: Option<[u32; 2]>,
```

Outside the range, all four outputs are zero: density rate, temperature rate,
velocity weight and target velocity. The node reads the frame from
`ctx.time().frame`, which the evaluation context already carries. A range
whose first frame is after its last is a `DocError` at validation.

The bench scenes move from 2a's `ember.sphere_emitter` to `ember.emitter` with
a sphere shape, a static transform, no noise and no velocity emission, and
`active_frames = [1, 60]`. `ember.sphere_emitter` does not gain the parameter.
The speed gate and preset examples build their documents from the same
`Scene`, so they move too. This is why task 1 moves the scenes first and then
reruns the presets.

**Test.** Frames 60 and 61 each reproduce bit for bit when played in order
and when reached by scrubbing. Frame 61's total density emitted is zero: one
frame with the emitter wired straight to an output reads all zeros at frame
61 and non-zero at frame 60. Proven able to fail by making the range check
exclusive at the top.

## 4. Metrics

Every metric takes arrays in x-fastest layout, and reports physical units:
metres, m/s, and density × m³ for mass. They live in
`elements-ember::metrics` beside `divergence` and `centroid_z`. Sums are in
`f64`.

### 4.1 Velocity

*Revised after Task 3.* Mantaflow's cache stores velocity on faces, with the
same convention as Ember (x of cell (i, j, k) on the face between cells i − 1
and i). But it stores velocity **only where there is smoke** (density above
the domain's `clipping`, 1e-6), and setting `clipping` to 0 does not change
that. User decision, 2026-09-23: every velocity metric is computed for both
solvers over the same smoke mask.

- **Measured cells.** A cell is measured when the cell and all 26 of its
  neighbours lie inside the domain, none is a collider cell, and all have
  density above 1e-6. The neighbourhood condition keeps every face and every
  central difference inside the region where Mantaflow's cache has data.
  The same rule applies to both solvers.
- **Divergence:** the face stencil (`metrics::divergence`'s), which is the
  quantity each projection minimises, over measured cells.
- **Resampling to cell centres:** the mean of a cell's two faces along each
  axis. Kinetic energy and vorticity use cell-centred velocity.
- **Kinetic energy:** ½ Σ|u|² dV over measured cells.
- **Total vorticity magnitude:** Σ|∇×u| dV over measured cells, with central
  differences of the cell-centred velocity.
- The table says that velocity metrics cover only the smoke.

### 4.2 Density and height

- **Total mass:** Σ ρ dV.
- **Centroid height:** `centroid_z`, converted from cells to metres.
- **Plume top:** among cells whose density is at least 1% of the frame's
  peak, the height below which 95% of their density lies.

### 4.3 Mass drift after emission stops

The domain top is open in both solvers (Ember's default boundaries: closed
sides and floor, open top), so density leaves the domain during frames
61–120.

- **Outflow per frame:** Σ over the z-faces one cell below the top, index
  nz − 1 (between cells nz − 2 and nz − 1), of ρ · max(w, 0) · dA, where ρ is
  the density of cell nz − 2 and w the face velocity. The top face itself is
  not used, because Mantaflow's cache does not store it; the cell layer above
  the measuring plane is one voxel thick.
- **Cumulative outflow:** those per-frame fluxes integrated over frame time
  with the trapezoid rule. This is an estimate at frame resolution, not
  substep resolution, and the table says so.
- **Drift at frame n:** (M(n) + outflow(60→n)) − M(60), reported absolutely
  and relative to M(60).

Both solvers are measured by this same estimator.

### 4.4 Speed and memory

- **Frame time:** wall clock per frame; median, min and max; frame 1
  excluded.
- **Mantaflow peak memory:** the peak resident memory of the Blender process
  from `/usr/bin/time -l`, minus the peak of a run that loads the same scene
  and exits without baking.
- **Ember peak memory:** the field pool's high-water mark in bytes plus the
  frame cache's size. If the pool does not track a high-water mark, task 5
  adds one.

### 4.5 Tests

Each metric is tested on an analytic field with a known answer (a uniform
translation for zero divergence and vorticity, solid-body rotation for known
vorticity, a density slab for mass, centroid and top), and each test is
proven able to fail by a single-change mutation, recorded in the ledger.

## 5. Matched scenes

`bench::Scene` gains `mantaflow_json(&self) -> serde_json::Value`, which
writes every field the Blender script needs: domain size and resolution,
fps and frame count, the emitter sphere, its rates and active frames, the
buoyancy, vorticity and dissipation settings, the collider, the wind, the
boundaries, and the substep count. Rust never generates Python.

`tests/bench/mantaflow_scene.py` (GPL-3.0-or-later, never in a crate) takes
the JSON path, an output directory and a resolution. Run with
`blender --background --factory-startup --python`, it:

- builds a `GAS` domain under the fairness rules of piece 2 §5.3: no noise,
  no adaptive domain, fixed timesteps equal to Ember's substep count, and an
  `OPENVDB` cache with `NONE` compression and 32-bit values;
- sets the domain walls to match Ember's boundaries: closed sides and floor,
  open top (Blender's default is all six open);
- adds a sphere mesh as the flow object, emitting through its volume, with
  its emission keyframed off after the last active frame;
- adds an effector for the collider, and a wind force field for
  `plume_wind`;
- bakes with `bake_all` and writes `timings.json`, one entry per frame.

Blender runs with `--python-exit-code 1`; without it a failing script still
exits 0.

Every parameter mapping is in `tests/bench/mapping.py` and in the table in
`docs/bench/mantaflow-notes.md`, with its source and the experiment that
confirmed it. The ones that are not exact, found in Task 3:

- **Inflow** is additive (`use_absolute = False`, `surface_distance = 0`,
  density = rate / fps). Mass matches Ember's within 3% over frames 12–60.
- **Temperature** cannot be additive: Mantaflow raises the emitter's heat to
  a set value, never above it, while Ember's grows at its rate. The mapping
  holds it at Ember's frame-24 emitter value, so plume heights differ (at
  64³, a centroid of 0.925 m against 0.781 m at frame 60). Task 6 may try a
  flow temperature keyframed to rise as Ember's does, and keeps it only if it
  brings the heights closer without changing mass.
- **Wind** acts only on cells holding smoke in Mantaflow
  (`update_effectors_task_cb` in Blender's `fluid.cc`), but on every cell in
  Ember. User decision, 2026-09-23: `plume_wind` is kept, and the results say
  so.
- **The pressure solve** differs: Mantaflow uses multigrid-preconditioned CG
  to a tolerance, Ember a fixed number of Gauss–Seidel iterations.
- The inflow, wind and vorticity mappings hold for one solver step per frame,
  which every bench scene uses.

Scenes are compared only on §4's metrics, never voxel by voxel.

The script is checked by `ruff` like the add-on. It cannot run in CI, which
has no Blender. A `plume` bake at 32³ in Task 6 and a `plume` run at 64³ in
Task 8 check it.

## 6. Runners

### 6.1 Ember

A new example, `crates/elements-ember/examples/benchmark.rs`, per scene and
resolution:

- **Timed runs.** The speed gate's method: each frame is `eval` plus a
  blocking wait, with timeline caching off. 3 runs at 64³ and 128³, one at
  256³.
- **Metrics run.** A separate run that reads back density and the three face
  grids after every frame and computes §4. Readback never happens inside a
  timed run.
- Writes one results file per scene and resolution (§7).

### 6.2 Mantaflow

A second example, or a mode of the same one, runs the Blender script for each
scene and resolution, then reads the cache into the same layout as §6.1 and
computes the same metrics.

- **Reader.** `vdb-rs` 0.6.0 parses a `Vec3s` grid's root values at 4 bytes
  and returns Mantaflow's velocity as an empty tree without an error. User
  decision, 2026-09-23: the workspace vendors a patched copy in
  `vendor/vdb-rs/` through `[patch.crates-io]` (`vendor/vdb-rs/PATCHED.md`).
- `vdb-rs` does not check value types, so the reader checks each grid's type
  (`Tree_float_5_4_3`, `Tree_vec3s_5_4_3`) before reading it, and expands
  tiles to their full extent.
- Index (0, 0, 0) is the domain's first cell; the VDB transform is ignored.
  Velocity converts to m/s as stored · dx / 0.4.
- A missing grid, a wrong grid type, or a velocity grid holding fewer values
  than density is an error, never a zero.
- The reader's test runs on the committed 16³ fixture from task 2, so reading
  the cache is checked in CI without Blender.

## 7. Results and failures

`just bench` runs both solvers on every scene and resolution. It is not part
of `just check`.

- Each scene and resolution first writes its own result file under
  `docs/bench/results/` (per-frame series as CSV, summary as JSON). The final
  `docs/bench/results.md` is built only from complete files, so a failed
  256³ run does not lose the others.
- **`results.md`:**
  - one table per scene: resolution against frame time, peak memory,
    divergence, kinetic energy and vorticity at frames 60 and 120, plume
    height at 60 and 120, and drift at 120;
  - the header records the machine, OS, Ember commit, Blender version, date,
    and run count per resolution;
  - a summary written in plain terms that says where Mantaflow wins, as piece
    2 §5.1 requires.
- Missing Blender, a failed bake, or an unreadable cache stops `just bench`
  with the scene, resolution and Blender's stderr. It never writes a partial
  `results.md`.
- Timing and benchmark numbers are never pass/fail tests.

## 8. Risks

- **(a) Resolved.** `vdb-rs` could read the float grids but not the vector
  grid. The workspace vendors a patched copy (§6.2).
- **(b) Parameter mappings are approximate.** Mantaflow's inflow and
  buoyancy do not correspond one to one with Ember's. Mitigation: the metrics
  are integral and shape-level, the mappings are written down, and the
  summary says where they limit the comparison.
- **(c) Machine load.** Risk (j) showed timings vary with GPU state under
  load. `just bench` records the load average at the start and end of each
  run, and the results header says to rerun on an idle machine if it was
  above 2.
- **(d) Run time at 256³.** Task 3 measured 4.1 s a frame at 256³ under load,
  about 36 minutes for the whole benchmark.
