# Ember Piece 2b-4 — Fire Design

**Date:** 2026-09-25
**Status:** Design approved in conversation; awaiting review of this document.
**Parent:** `2026-09-21-ember-solver-design.md` (piece 2), whose list
included flame. 2b-2 (`2026-09-23-ember-scene-content-2b2-design.md` §1)
deferred it to this cycle, after the benchmark.
**Branch:** `ember-fire-2b4`, from `ember-bench-2b3b` at ac62baf (PR #10
open), so that the ledger and bench records stay continuous.

## 1. Goal and scope

Fire for `ember.smoke_solver`. User decisions, 2026-09-25:

- **Model: fuel and reaction**, as Blender's Mantaflow fire does. Emitters
  add fuel; fuel burns at a rate, releasing heat and smoke; flame is derived
  from a reaction coordinate. It is a published model, reimplemented here;
  no Mantaflow or Blender source is copied. There is no combustion expansion
  term, so velocity stays divergence-free.
- **Validation: a fire scene in the Mantaflow benchmark** (timings,
  divergence, masses, and the side-by-side render), on top of in-engine
  tests.
- **Interface: an extension of `ember.smoke_solver`**, not a new node kind.
  Connecting a fuel input turns fire on; unconnected, the solver is exactly
  today's.

**Out of scope:** flame in the Blender add-on (it shows what the daemon
streams; VDB export and viewport shading of flame belong to pieces 3 and 4);
smoke colour grids (`flame_smoke_color`); a separate fire solver kind; a
divergence source for expansion.

## 2. Mantaflow's step, as verified

From the smoke script embedded in the Blender binary
(`/Applications/Blender.app`, the version used by 2b-3), one fire step is:

1. `applyEmission` into density, heat, **fuel and react** (from `fuelIn`
   and `reactIn`).
2. `processBurn(fuel, density, react, heat, burningRate, flameSmoke,
   ignitionTemp, maxTemp)`.
3. `smoke_step`: advect density, heat, **fuel, react** (order 2); advect
   velocity; vorticity confinement with
   `strengthCell = fuel · flameVorticity · timestep / frameLength`;
   buoyancy; walls; pressure.
4. `updateFlame(react, flame)`: `flame = sqrt(react)` where `react > 0`,
   else 0.

## 3. The model in Ember

### 3.1 State

With fire on, the solver keeps two more slots, `fuel` and `react`, both
cell-centred `R32Float`. `flame` is not stored: it is
`sqrt(clamp(react, 0, 1))`, computed when the output is copied, which
matches step 4 above. The upper clamp is Ember's (decided 2026-09-25 in
Task 4): the mass correction can lift react slightly above 1, which would
otherwise put the flame's heat above `max_temperature`. React itself is
stored unclamped, and the burn's heat profile uses the same clamped flame. A snapshot for frame N remains the state entering N.

With fire off there are no fire slots. A stored state whose slots do not
match the document (fuel connected or disconnected since) is a
`StateShape` error, which the timeline already answers with a reset
(`crates/elements-core/src/graph/timeline.rs`).

### 3.2 Substep order with fire on

New work is in bold.

1. Emit density, temperature and **fuel, clamped to [0, 10] per cell as
   Mantaflow's inflow does, and blend `react` towards 1 by
   the fresh fuel's share**, marking fresh fuel as unburnt. With
   `Δ = fuel_rate · h` and `fuel' = min(fuel + Δ, 10)`: where `fuel' > 1e-6`,
   `react' = react + (Δ / fuel') · (1 − react)`, clamped to [0, 1]. The
   §6.1 probe found that Mantaflow blends rather than adds (revised
   2026-09-25; an additive react would exceed 1 and push the flame's heat
   above `max_temperature`). Mantaflow blends towards
   `1 − (1 − occupancy)²`, and Ember towards 1, because the solver receives
   rates and not the emitter's occupancy. The two agree in fully occupied
   cells and differ at an emitter's one-voxel surface ramp (§3.3).
2. **Burn**, one cell-local kernel with substep length `h` (seconds):
   - `fuel' = max(fuel − burning_rate · h, 0)`
   - `react' = react · fuel' / fuel` where `fuel > 1e-6`, else `react' = 0`
   - `f = sqrt(max(react', 0))`
   - `density += (0.5 + 0.5 · max(1 − fuel, 0)) · (fuel − fuel') · 0.1 · flame_smoke`
   - where `f > 0`: `temperature = (1 − f) · ignition_temperature + f · max_temperature`
     (an overwrite, as in Mantaflow)
3. Buoyancy and wind, unchanged. Flame rises through the existing
   `β · temperature` term.
4. **Vorticity confinement with a per-cell strength**
   `ε + flame_vorticity · fuel`. Confinement runs when `vorticity > 0` or
   (fire on and `flame_vorticity > 0`).
5. Advect velocity and project. **With fire on, velocity faces are traced
   back with one Euler step** instead of the RK2 midpoint (2b-1 spec §4.1),
   as Mantaflow's `advectSemiLagrange(order=2)` does by default
   (`orderTrace=1`). User decision, 2026-09-26, after Task 9 found the
   `fire` scene diverging at preview's single substep: at CFL 5–30 in the
   fire's core the RK2 midpoint misses a spike at the emitter's corner,
   which flame vorticity then amplifies (×2.5 a frame, NaN by frame 40 at
   32³). Euler velocity plus the fuel clamp ran 120 frames at 32³, 64³ and
   128³ with peak |u| 9.5, 10.8 and 12.9 m/s (Mantaflow: 15.3 and 11.7 at
   32³ and 64³). Scalars keep RK2, and fire-off scenes keep RK2 everywhere,
   so smoke output is unchanged (`.superpowers/sdd/fire-diagnosis.md`).
6. Advect density, temperature, **fuel and react**, with the same scheme
   (MacCormack by default) and collider handling. With `conserve_mass`, fuel
   and react each get the 2b-3c global correction; both are non-negative,
   and the mixed-sign rule applies unchanged.

Fuel has no dissipation parameter; Mantaflow has none.

### 3.3 Deliberate differences from Mantaflow

Recorded in `docs/bench/mantaflow-notes.md` and the results:

- **No density clamp.** `processBurn` clamps density to [0, 1] in every
  cell. Ember keeps mass, as it already does for smoke.
- **React blends towards 1**, not towards Mantaflow's
  `1 − (1 − occupancy)²` (§3.2 step 1): only cells on an emitter's surface
  ramp differ.
- **Confinement reads fuel before advection.** Mantaflow burns before
  advection and confines on the advected fuel; Ember confines before
  advecting velocity, on the fuel after the burn.
- **No colour grids.**
- **Velocity backtrace with fire on is Euler, scalars stay RK2.**
  Mantaflow traces every grid with one Euler step; Ember matches it only
  for velocity, and only while fire is on (§3.2 step 5).
- **Mantaflow's advection gains fuel; Ember's conserves it.** Measured in
  2b-4 Task 9's diagnosis: emission (139.7 vs 140 a frame at 32³) and the
  burn law match, but Mantaflow's fuel advection adds +26% of the emitted
  fuel by frame 30 and +103% by frame 60 at 32³. Ember's fuel totals are
  expected to sit below Mantaflow's for that reason, not a mapping error.
- **Units.** `h` is Ember's substep in seconds. `burning_rate` and
  `flame_vorticity` are per second in Ember; `mapping.fire` (§6.1) converts
  Mantaflow's frame-scaled values.

### 3.4 Fire off

Fuel input unconnected: no fire slots, no burn dispatch, and confinement
keeps its uniform ε. Output is bit-identical to the solver before 2b-4
(§5, test "fire off is unchanged").

## 4. Node interface

Every change appends sockets or adds optional parameters, so existing
documents keep their indices and still validate.

### 4.1 `ember.emitter`

- `fuel_rate`: fuel added per second where fully occupied; default 0.
  Noise and `active_frames` modulate it as they do the other rates.
- Output 4, `fuel`, a Field. A placeholder when no consumer reads it
  (`EvalCtx::output_wanted`).

### 4.2 `ember.emitter_union`

- Inputs 0–7 unchanged. New optional inputs 8 (fuel A) and 9 (fuel B).
- Output 4: the sum of the connected fuel inputs, or zero when neither is
  connected.

### 4.3 `ember.smoke_solver`

- Input 6: fuel rate, optional. Connected turns fire on; nothing else does.
- Output 3: `flame`, copied only when read. With fire off, a read gets a
  zeroed field.
- Parameters, independent of the preset:

  | parameter | meaning | validation |
  |---|---|---|
  | `burning_rate` | fuel burnt per second | finite, ≥ 0 |
  | `flame_smoke` | smoke per unit fuel burnt (Mantaflow's factor) | finite, ≥ 0 |
  | `flame_vorticity` | extra confinement per unit fuel, 1/s | finite, ≥ 0 |
  | `ignition_temperature` | heat at the flame's edge, Ember temperature units | finite |
  | `max_temperature` | heat at the flame's core | finite, ≥ `ignition_temperature` |

  Defaults are Blender's defaults (burning rate 0.75, flame smoke 1.0, flame
  vorticity 0.5, ignition 1.5, maximum 3.0) converted through
  `mapping.fire` at 24 fps. The probe task fixes and records the converted
  values; until then the plan uses Blender's numbers.
- Fire parameters set with fuel unconnected are accepted and ignored, as
  wind parameters are when `wind_rate = 0`.

### 4.4 Errors

The fire kernels follow the existing patterns: recorded into the substep's
batch, replaced fields retired until `submit` or `abandon`, and state never
written back after a failed step. No new error variants.

## 5. In-engine tests

Each is proven able to fail by a single mutation (CLAUDE.md), with the real
failing output recorded.

- **Burn closed form.** Uniform fuel F₀ and react 1 in a closed, still
  domain with buoyancy off: after n substeps, fuel = max(F₀ − n · rate · h,
  0) and react = fuel / F₀, against a CPU reference; flame = sqrt(react).
- **Heat profile.** Where flame > 0, temperature = (1 − f) · T_ign + f ·
  T_max; where fuel is 0, temperature is untouched by the burn.
- **Smoke from burning.** Density added equals the CPU formula summed over
  cells.
- **No fuel, no flame.** Fire on with `fuel_rate = 0`: fuel and flame
  exactly 0. (Velocity, density and temperature are no longer bit-identical
  to the fire-off run, since fire on traces velocity faces with Euler,
  §3.2 step 5.)
- **Fire off is unchanged.** `plume` frame 40 is bit-identical to a hash
  recorded from the pre-2b-4 build.
- **Flame vorticity.** With `vorticity = 0`, a fuel-laden swirl gains
  kinetic energy with `flame_vorticity > 0` that it does not gain with 0.
- **Fire rises.** A fuel sphere at the floor: the flame centroid rises
  monotonically over N frames.
- **Colliders.** No fuel or flame inside a collider beyond tolerance.
- **Conservation.** With `conserve_mass` and `burning_rate = 0`, fuel drift
  stays at the ~1e-6 level of density.
- **Determinism.** Fire scene frame 40, reached in order and after a scrub,
  bit-identical.
- **Parameters.** Each validation error, including ignition above maximum.
- **Sockets.** Emitter and union fuel outputs; union fuel inputs optional;
  existing socket indices unchanged.

## 6. Benchmark

### 6.1 Mapping probe (first task, before any kernel)

A 32³ Mantaflow fire bake with known fuel, measured from its VDB cache,
settles:

- how `reactIn` relates to `fuelIn`;
- fuel burnt per frame against `burning_rate`, giving Ember's rate in 1/s;
- the timestep scale on `flame_vorticity`.

The result is `mapping.fire(...)` in the bench mapping module, tested in
`tests/bench/test_mapping.py`, with a single-mutation proof per conversion
as wind's mapping had.

### 6.2 The `fire` scene

A sphere fuel emitter at the floor of an open-top domain, Blender's default
fire settings mapped to Ember. It joins `just bench` at 64³, 128³ and 256³
with the existing timing, divergence and mass tables, plus fuel mass and
flame volume (cells with flame > 0.01). With `cache_resumable` on,
Mantaflow's cache writes `flame` and `fuel` (the probe found `fuel` and
`react` only in a resumable cache, and that resumable does not change the
simulation), so both are compared directly. `docs/bench/results.md` is
regenerated with the new rows.

### 6.3 Render

`just bench-render` gains `fire`: flame drawn as emission (blackbody from
temperature) over smoke density, the same shader for both solvers. The
bench's Ember VDB writer adds a `flame` grid. The verdict is recorded as
2b-3b's was, scoped to what the images show.

### 6.4 Cost

`just bench-presets` measures the fire scene's 128³ preview frame. It is
recorded, not gated. Above 100 ms it becomes an open risk in piece 2's spec
§6, alongside (k).

## 7. Risks

- **Preview cost.** Fire adds two MacCormack advections with mass
  correction, a burn pass and per-cell confinement to every substep. Smoke
  scenes pay nothing (§3.4); fire scenes may exceed preview's budget (§6.4).
- **Heat overwrite and conservation.** The burn overwrites temperature,
  which is not conservative. The correction measures temperature after the
  burn (it runs at advection), so it corrects advection only, as intended.
- **Mapping uncertainty.** Burn and vorticity scales come from a probe
  bake, as wind's did; a wrong mapping shows as a mismatch in §6.2 rather
  than a test failure.
