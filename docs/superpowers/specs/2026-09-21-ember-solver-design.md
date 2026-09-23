# Ember Piece 2 — Solver Design

**Date:** 2026-09-21 (§1–4 written 2026-09-22)
**Status:** Piece 2a (§1–4) is complete. The 128³ gate passed, and the default is
160 pressure iterations (`docs/bench/speed-gate.md`). §5 is the benchmark design,
whose Mantaflow half is built in 2b. §6 lists the risks 2b inherits.
Piece 2b-1 (solver correctness) is complete; see `2026-09-22-ember-solver-2b1-design.md`.
Of §6's risks, (b), (c), (d), (f) and (h) are resolved; (e), the open part of (g), and (j) remain.
**Parent:** `2026-09-21-ember-design.md` (piece 2 of 4)

## 1. Goal and split

Piece 2 is built in **two spec/plan cycles**, because the speed gate (§4.3) can
change the design: fewer pressure iterations, multigrid, or a lower interactive
resolution. Nothing past the gate is planned in detail until the gate has run.

**2a — to the gate (this spec, §2–4):** the `elements-ember` crate; the
`ember.smoke_solver` stateful node with emit, buoyancy, velocity advection,
pressure projection and scalar advection; `ember.sphere_emitter`; the
still-domain, divergence-free, buoyant-blob and determinism validation scenes;
the 128³ speed gate and its recorded decision.

**2b — after the gate:** split into three cycles (user decision, 2026-09-22):
2b-1 solver correctness (`2026-09-22-ember-solver-2b1-design.md`), 2b-2 scene
content, 2b-3 the Mantaflow benchmark. The original list: vorticity
confinement, dissipation, flame, box and noise-modulated emitters, sphere and
box colliders, animatable transforms, CFL substepping, quality presets, the
collider validation scene, per-face boundary settings, and the Mantaflow
benchmark of §5.

The umbrella's targets are unchanged: 128³ at ≥ 10 fps interactively, 256³–512³
offline, the step order of umbrella §3, and the scenes of umbrella §6.

**Precondition, as the first task of 2a's plan:** CI runs on lavapipe. Umbrella
§7 requires it before piece 2, and the solver's tolerances would otherwise be
tuned on Metal only. It needs a git remote, which the user creates or approves.

## 2. Components

### 2.1 Crate

`crates/elements-ember`, `Apache-2.0 OR MIT`, `#![forbid(unsafe_code)]`,
depending on `elements-core`. It exposes `register(&mut NodeRegistry)`, which
the CLI and daemon call after `NodeRegistry::with_builtins()`. Core gains no
solver code; this is the seam the registry exists for.

### 2.2 Physical units

The document gains an optional `domain_size`: metres along the grid's longest
axis, default `2.0` (Blender's default cube). Voxels are cubic, so
`voxel_size = domain_size / max(nx, ny, nz)`. `EvalCtx` exposes it. The field is
optional with a default, so existing documents load unchanged.

Every solver and emitter parameter is in metres and seconds, so a scene
previewed at 128³ looks the same baked at 512³. Parameters in voxel units would
break exactly that workflow.

### 2.3 `ember.sphere_emitter` (stateless)

- Params: `center` `[f32; 3]` and `radius` in metres, relative to the domain's
  minimum corner; `density_rate` and `temperature_rate` per second.
- Outputs: `[density_source: Field, temperature_source: Field]`, each the rate
  times an occupancy with a one-voxel smoothed edge, so the emitted total does
  not depend on resolution.
- The transform is fixed in 2a. Animation is 2b.

### 2.4 `ember.smoke_solver` (stateful)

| | |
|---|---|
| Inputs | `density_source: Field`, `temperature_source: Field` |
| Outputs | `density: Field`, `temperature: Field`, `velocity: VectorField` |
| Params | `substeps` (fixed, default 1), `pressure_iterations`, `buoyancy_density` (α, m/s² per unit density, sinks), `buoyancy_temperature` (β, m/s² per unit temperature, rises) |
| State slots | `velocity` (staggered), `density`, `temperature`, `pressure` |

Buoyancy is Boussinesq and acts along +z: `w += h·(β·T − α·ρ)`, with ambient
temperature zero. Gravity is the direction of this term, not a separate force.

`pressure` persists between steps as a **warm start** for red-black
Gauss–Seidel, which converges far faster from last frame's pressure than from
zero. It lives in the state store, so snapshots include it and scrubbing stays
bit-deterministic.

**Boundaries in 2a:** solid walls (zero normal velocity, Neumann pressure) on
five faces; the top face (+z) is open (pressure 0), so a plume leaves the domain
instead of piling up at the ceiling. Per-face settings are 2b.

The density output feeds the existing `core.output`, so the CLI, daemon and
viewport need no change beyond registration.

### 2.5 `ComputeBatch` (in `elements-core::gpu`)

Records many dispatches into one command encoder and submits once, inside
`GpuContext::scoped`. It lives in core because Tide needs it too.
`dispatch_over_field` submits per dispatch; at 128³ with 160 iterations that is
over 320 submits per substep, and submit overhead alone could fail the gate for
reasons unrelated to the solver's maths.

## 3. Data flow

Per substep, with `h = dt / substeps` and `dt` from the document's `fps`:

| # | Stage | Kernel | Storage bindings |
|---|---|---|---|
| 1 | Emit | `density += src·h`; same for temperature | 2 (read_write) |
| 2 | Forces | buoyancy onto z-faces, cell values averaged to faces | 1 |
| 3 | Advect velocity | semi-Lagrangian backtrace from each face centre, hand trilinear (E2); old faces read as `texture_3d`, new pooled faces written | 3 |
| 4a | Divergence | per cell from the faces | 1 |
| 4b | Pressure | N × {red, black} in-place sweeps on `pressure` (read_write) | 1 |
| 4c | Gradient subtract | faces −= h·∇p; normal velocity on solid walls forced to 0 | 3 |
| 5 | Advect scalars | density and temperature through the projected velocity | 2 |

- Every kernel stays within the 4-storage-texture limit. Neighbour reads go
  through `texture_3d<f32>` + `textureLoad`, except the red-black sweep, which
  reads and writes `pressure` through a single `read_write` binding.
- Red-black is bit-deterministic: cells of one colour never read each other.
- One substep is one `ComputeBatch` submit. Red and black are two entry points,
  so two cached pipelines. Bind groups and one uniform buffer (dims, voxel size,
  `h`) are built once per substep and reused across all N iterations. Nothing
  is allocated inside the pressure loop.
- Scratch fields (new faces, divergence, new scalars) come from the pool and
  return to it each step. A steady-state step allocates no textures.
- Vorticity (between 2 and 3) and dissipation (after 5) slot in during 2b.

### 3.1 Errors

- Params are validated at document load: `substeps` in 1..=16,
  `pressure_iterations` in 1..=1000, every float finite, `radius > 0`,
  `domain_size > 0`. Violations are `DocError`s. Documents are untrusted, and
  the daemon must not panic.
- On any error mid-step, every taken state value and every scratch field goes
  back to the pool before the error returns.
- State is written back only after a whole step succeeds. A failed step's
  half-updated state is released, never stored; the timeline already clears
  its store and cursor on any error and recomputes from its cache next time.

## 4. Testing and the speed gate

Every test records a single-change mutation that makes it fail (CLAUDE.md).
Solver test grids are ≤ 32³ so `just check` stays fast (the emitter's
resolution check fills one 64³ field, which is cheap); only the gate runs at 128³.
Kernel tests use non-cubic grids (e.g. 8×6×5) so an axis swap cannot pass by
symmetry.

### 4.1 Kernel tests (GPU against a CPU reference, tolerance ~1e-5)

- **Advection:** a uniform velocity of an integer number of voxels shifts a
  field exactly; a fractional one matches the CPU trilinear reference.
- **Divergence:** `u = (x, 0, 0)` gives `div = 1` in the interior.
- **Red-black sweep:** matches a CPU red-black reference; a Neumann wall and the
  Dirichlet top are each covered.
- **Gradient subtract:** normal velocity on solid walls is exactly 0 afterwards.
- **Emit and buoyancy:** face averaging matches the CPU. The sphere emitter's
  total emitted amount agrees within 5% between 32³ and 64³.
- **`ComputeBatch`:** a dispatch reading the previous dispatch's output in the
  same batch sees the updated data.
- **Doc validation:** `substeps` 0 and 17, a NaN rate, and
  `pressure_iterations` 0 are each rejected with a `DocError`.
- **Error path:** a forced mid-step failure returns every field to the pool;
  pool counts balance before and after.

### 4.2 Validation scenes (umbrella §6, those 2a can reach)

| Scene | Assertion | Example mutation |
|---|---|---|
| Still domain | velocity exactly zero after 10 frames | forces kernel adds a constant |
| Divergence-free | RMS divergence after projection ≤ 10% of before, at N = 160 (the default the user chose from the gate, 2026-09-22) | delete the black sweep |
| Buoyant blob | density centroid z rises strictly every frame for 20 frames | flip the sign of β |
| Determinism | frame 40 bit-identical in order, after scrubbing, and after eviction | keep `pressure` outside the state store |

The collider scene is 2b.

### 4.3 The speed gate

This refines §5.5, which predates the 2a/2b split and names a "default quality
preset" that 2a does not have.

**Scene:** §5.3's `plume` at 128³, `substeps = 1`. It is defined once as a Rust
`Scene` struct in `elements-ember::bench` that serialises to an `.elements`
document. In 2b the same struct also generates the Mantaflow script.

**Runner:** `just bench-gate` runs
`cargo run --release -p elements-ember --example speed_gate`. It is not part of
`just check`.

For each N in {20, 40, 80, 160}:

1. **Warm up:** step frames 1–24 untimed (shader compilation, plume
   development, pressure warm start).
2. **Time:** frames 25–48, each measured as `eval` plus a blocking poll, with
   timeline caching off. Wall clock, because GPU timestamp queries need the
   optional `TIMESTAMP_QUERY` feature. Median, min and max per run; the reported
   figure is the median of 3 runs.
3. **Snapshot cost:** timed separately and reported as its own column. It is
   real interactive cost, but not the solver's.
4. **Divergence:** from the frame-48 state, run stages 1–3 through the kernel
   API, read back the faces and compute divergence; run the projection with N
   iterations and compute it again. `metrics::divergence` computes max and RMS
   over all cells on the CPU; it is the same function §5.4 uses in 2b.

**Output:** `docs/bench/speed-gate.md` records the machine, OS, Ember commit,
date, and:

| N | step ms (median, min–max) | snapshot ms | RMS div before | RMS div after | ratio | max div after |
|---|---|---|---|---|---|---|

**Pre-registered rule.** It is fixed here, before any number exists, and is not
revised to fit the results.

- **PASS** if some N has a median step of ≤ 100 ms **and** a ratio of ≤ 0.10.
  The provisional default `pressure_iterations` is the **largest** passing N:
  the best quality that fits the budget. The user confirms it, and 2b's presets
  start from there.
- **FAIL** otherwise. Work stops, and the table goes to the user with §5.5's
  three options.

The decision is recorded as a line in `docs/bench/speed-gate.md`. It is a human
decision, never a CI check.

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
or add-on work, measure Ember's step time at 128³ on this machine (Apple Silicon, Metal).
The procedure and pass rule are in §4.3.

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

---

## 6. Risks carried into piece 2b

Found by piece 2a's whole-branch review. (a) and (i) were resolved in 2a, and
(g) mostly. 2b-1 resolved (b), (c), (d), (f) and (h). Still open: (e), the open
part of (g), and (j).

- **(a) Resolved (eac0cc3).** `GpuContext` now also requests the adapter's
  `max_buffer_size`, so 512³ domains are no longer capped by the downlevel
  256 MiB limit.
- **(b) Resolved (45a8e1f).** The pressure slot now stores p and the solve
  is ∇²p = div/h, so the warm start survives a change of `h`. As found: the
  slot stored φ = h·p, so CFL substepping, which changes `h`, would have had to
  rescale the stored φ by h_new/h_old before each solve.
- **(c) Resolved (3ef9ccd and d86071e).** Substeps are chosen each frame by
  CFL, capped at `max_substeps` (3ef9ccd), and advection is MacCormack over an
  RK2 backtrace with density and temperature dissipation (d86071e). As found:
  there was no CFL limit and no dissipation. Sampling is bounded, so nothing
  blew up, but temperature and buoyancy accumulated near a constant emitter and
  the backtrace was a single Euler step. At 128³ every 1 m/s is about 2.7 cells
  per step.
- **(d) Resolved (45a8e1f and 93d6c47).** Scalars read the ambient value 0
  beyond an open face (45a8e1f), and a fully closed domain removes p's mean
  after each solve (93d6c47). As found: a backtrace that left through the open
  top clamped to the top layer instead of taking the ambient value, so smoke
  was drawn back in, and per-face boundaries would make a fully closed domain's
  Neumann system singular.
- **(e) Pressure stays unconverged at low frequencies.** The residual falls
  about as N^-1.5 (`docs/bench/iteration-sweep.md`), so expect plume shape to
  differ from Mantaflow's PCG. Multigrid is the stretch goal, and the warm start
  is load-bearing.
- **(f) Resolved (d0cc411).** The pressure loop is split across submissions
  every `iterations_per_submit(cells)` iterations, bit-identical to one
  submission. As found: one substep was one submission, up to 2000 dispatches,
  which at 256³–512³ with offline iteration counts could run for seconds and
  trip GPU watchdogs, surfacing as `DeviceLost`.
- **(g) Resolved, partly open (37c01a9 and 02ee266).**
  `GpuContext::scoped` now pushes OutOfMemory, Internal and Validation scopes,
  so an out-of-memory error from texture creation is reported as
  `GpuError::OutOfMemory`, not panicked through wgpu's default handler. An
  uncaptured-error handler holds the most severe error raised outside any
  scope for the next `scoped` call, and a stray out-of-memory error outranks
  in-scope internal and validation errors. Blocking waits go through
  `GpuContext::wait`, which catches the panic from an unrecognised
  `Device::poll` failure (wgpu's `handle_error_fatal`, the ordinary Vulkan
  device-lost path) and reports it as `DeviceLost`; this resolves the earlier
  `Device::poll` caveat. Still open: on unified memory the OS may swap or end
  the process before wgpu ever reports the condition; the daemon's
  reset-and-clear path for this error is untested; and estimating the working
  set when a document is validated was dropped from the hardening slice by the
  user's choice to capture the error rather than predict it, so it remains
  open.
- **(h) Resolved (23e8d6e).** The solver now copies only the outputs
  `EvalCtx::output_wanted` reports as read, and leaves a placeholder in the
  rest. As found: all three solver outputs were duplicated every frame even
  when only density was consumed: free at 128³, real bandwidth at 512³.
- **(i) Resolved (8d8bc64).** `crates/elements-ember/tests/solver.rs`
  adds `one_frame_adds_each_emitted_quantity_to_its_own_output`, which drives
  the graph through both an emitter-straight-to-output probe and a
  through-the-solver probe and asserts each of density and temperature
  matches rate × dt to 1e-5. Proved to fail by swapping the solver's source
  fields in `SmokeSolver::step`.
- **(j) Preview frame cost.** A 128³ preview frame costs about 92 ms with
  MacCormack, RK2 and the CFL readback, against 2a's 34 ms per step at the same
  N (`docs/bench/presets.md`). That leaves about 8 ms of headroom at one
  substep, with CFL clamped on every frame. The increase is unexplained and
  should be profiled before 2b-3.

  2b-1's final review narrowed it down. The preset sweep is linear in the
  substep count, at about 90.5 ms per substep plus about 1 ms per frame
  (cap 1 = 91.77 ms, cap 2 = 182.77 ms, cap 3 = 272.31 ms). The per-frame
  work, the CFL readback and the output copies, is therefore not the cause:
  the regression is inside the substep. The likely causes, most likely first:

  1. `pressure.wgsl`'s runtime axis loop, with dynamic vector indexing,
     `is_open` reads of the uniform, and a division per cell. naga's Metal
     backend adds loop bounding to every loop (`force_loop_bounding`), which
     tends to block unrolling. The kernel runs in 320 dispatches per substep
     at N = 160.
  2. The sampling path's runtime loops and the local `array<f32, 8>` in
     `texel()` and `corners()`.
  3. MacCormack plus RK2, which raise texel reads per grid cell by about 5×,
     including the forward backtrace the correction pass recomputes.

  The cheap way to tell them apart: `just bench-sweep 20,160` (the
  recipe sets `SPEED_GATE_ITERATIONS` itself, so an outer value is
  overridden; it rewrites `docs/bench/iteration-sweep.md`) gives the
  per-iteration slope, which isolates (1), and one frame with
  `advection: semi_lagrangian` isolates (3). A fix may let preview afford 2
  substeps, which would reopen the preset decision in `docs/bench/presets.md`.

  **Measured 2026-09-23, on main after 2b-1 (19cbbe0).** Both runs were under a
  load average of about 8–10, so the numbers are upper bounds; the split between
  causes held on both the medians and the fastest frames.

  | 128³, N = 160, one substep | Frame |
  |---|---|
  | 2a, semi-Lagrangian (gate) | ~34 ms |
  | now, semi-Lagrangian | 54.6 ms |
  | now, MacCormack (preview) | 91.8 ms |

  - **The pressure solve is about 2× slower per iteration.** `just bench-sweep
    20,160` fits 0.33–0.42 ms per iteration against 2a's 0.19
    (`docs/bench/iteration-sweep-2026-09-23.md`), about +20 ms at N = 160. The
    semi-Lagrangian frame's 54.6 ms against 2a's 34 matches this, so cause (1)
    is real.
  - **MacCormack costs about 37 ms per substep.** It accounts for essentially all
    of the sweep's ~38 ms fixed cost (2a: 3.6 ms)
    (`docs/bench/presets-semi-lagrangian-2026-09-23.md`). With semi-Lagrangian
    advection the non-pressure work is back to a few ms, so cause (2), the
    sampling loops on their own, is not significant. The cost is cause (3):
    the backward pass and the correction pass that recomputes the RK2 backtrace,
    over five grids.
  - **What a fix buys.** Restoring the pressure kernel to 2a's per-iteration cost
    brings a semi-Lagrangian substep to about 34 ms (2 substeps ≈ 70 ms) and a
    MacCormack substep to about 70 ms (still one substep). Two MacCormack
    substeps also need MacCormack to get cheaper, for example by storing the
    backtrace point instead of recomputing it and unrolling the corner loops.
    Semi-Lagrangian at 2 substeps takes 107 ms today, over the budget.

  **Root causes, 2026-09-23 (branch `pressure-kernel`).** Two separate causes
  slow the pressure solve. The first is fixed; the second is not.

  1. **Fixed in 5536e87: the axis loop in `pressure.wgsl`.** The per-iteration
     cost was bisected commit by commit (exported trees, each rebuilt). It is
     0.19 ms through 23e8d6e and 0.27 ms from 45a8e1f, the commit that turned
     the six neighbour checks into a loop over axes. Timed alone at 128³, the
     looped kernel runs at 0.279 ms per iteration, the same stencil written
     out per side at 0.198, and 2a's kernel at 0.208, whatever the data. The
     per-cell division and the open-face checks cost nothing measurable.
  2. **Open: recycled textures in the MacCormack scene.** Inside a MacCormack
     substep the same solve runs at about 0.30 ms per iteration, against 0.195
     with semi-Lagrangian. That is roughly 17 ms of a preview frame at N = 160.
     - **What it is not:** the values, since the scene's exact values copied
       into fresh textures run at 0.195; subnormal or non-finite values, of
       which there are none; or the shared submission, since flushing before
       the solve changes nothing.
     - **What it looks like:** it follows which pool-recycled textures hold p
       and div. Every one of 56 pairings of freshly allocated textures is fast,
       and rewriting a slow texture from the CPU sometimes makes it fast.
     - **Likely source:** Metal or driver state of recycled textures, such as
       compression or residency, rather than the kernel.
     - **Next steps:** try dedicated, never-recycled textures for p and div;
       take a Metal GPU capture with Instruments counters.
