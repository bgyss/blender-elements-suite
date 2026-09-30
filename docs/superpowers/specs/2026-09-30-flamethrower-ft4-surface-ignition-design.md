# Flamethrower FT4: surface ignition

**Date:** 2026-09-30. **Roadmap:** `2026-09-30-flamethrower-napalm-roadmap-design.md` §4 (FT4),
§6 (the open item on where the reservoir lives, decided below).
**Shot:** `2026-09-30-flamethrower-shot-ft0.md`.

## 1. Purpose

A burning surface for the shack. A collider can carry a wood budget. It ignites when the gas next
to it is hot enough, feeds fuel into the fluid while it burns, runs out, and leaves a `char` mask
for shading. Spread along the surface is emergent: a burning cell heats the gas, and the gas
ignites its neighbours. There is no spread rule.

Out of scope: moving surfaces (§3.4), the multi-grid export and render (FT5), the shot (FT6), and
any change to fire's pressure quality (FT3's open recommendation).

## 2. Decisions recorded

- **User, 2026-09-30:** the reservoir extends `ember.collider`. No new node kind.
- **Only a per-cell load travels with the collider.** Sockets carry fields, so a collider's
  `surface_fuel.load` becomes a `Field`. The burn rate and ignition temperature are solver
  parameters, not per-collider. (Earlier chat text listed rates on the collider; that was wrong.)
- **The solver stores `burned`, not `remaining`.** `burned` starts at zero, so nothing has to be
  seeded on the first frame. Remaining = `load − burned`. `char` = `burned / load`.
- **No separate heat term.** Emitted fuel goes through the existing `emit_fuel` and `burn`
  kernels, and the flame temperature profile heats the gas. That is what ignites the neighbours.

## 3. Design

### 3.1 Sockets

| Node | Change |
|---|---|
| `ember.collider`, `ember.mesh_collider` | optional param `surface_fuel: { load }` (load ≥ 0, in fuel-grid units per cell). New output 2, a `Field`: `load` where the SDF is negative, 0 elsewhere; all-zero when the param is absent. |
| collider union | adds the load pair: output 2 is the per-cell max of the two loads. Existing outputs unchanged. |
| `ember.smoke_solver` | new input 7 (`Field`, the load) and new output 4 (`Field`, `char`). |

Existing sockets keep their indices. A document that does not connect input 7 behaves as before.

### 3.2 State and parameters

`SolverState` gains `surface: Option<Field>`, the `burned` grid, present once input 7 is connected.
It is a new state slot `burned`, expected exactly when input 7 is connected, like `fuel` and `react`
with fire (`StateShape` on mismatch, answered by a timeline reset).

Input 7 needs the fuel input (fire) too. Connecting it without fuel is a `NodeError`, not a silent
no-op.

New `SolverParams`: `surface_burn_rate` (fuel per second per cell, ≥ 0, default 2.0). The ignition
threshold is the existing `ignition_temperature`.

### 3.3 Kernels

Two kernels, recorded each substep in `pre_projection` after the existing `emit_fuel` and before
`burn`, only when input 7 is connected. The wood cells sit inside the solid mask, so the fluid
solve never sees them.

1. **`surface_burn`**, on band cells. A band cell has `load > 0`, is solid (mask > 0.5), and has at
   least one fluid neighbour (6-neighbourhood). Let `T` be the maximum temperature of its fluid
   neighbours and `b` its `burned`.
   - **Burning:** `b > 0` or `T > ignition_temperature`, and `b < load`.
   - **Emission:** `e = min(surface_burn_rate · h, load − b)` when burning, else 0.
   - **Update:** `b += e`; write `e` to a scratch field.
2. **`surface_gather`**, on fluid cells. Each solid neighbour `n` that emitted `e_n` gives
   `e_n / fluid_neighbours(n)` to this cell. The sum divided by `h` is a fuel rate, and
   `emit_fuel` consumes it like any other fuel source (fuel clamp, react blend).

The gather form avoids atomics and is deterministic. The budget closes exactly: Σ`burned` =
Σ fuel given to the fluid, before the `[0, 10]` clamp at emission (tests keep cells under it).

`char = burned / load` where `load > 0`, else 0, is computed in `copy_outputs` when output 4 is
wanted.

### 3.4 Limits

- **Static colliders only.** `burned` is indexed by grid cell, so a collider with `surface_fuel`
  must not move. `build` rejects a keyframed or animated transform on it.
- The band test uses the solid mask, so a plank thinner than a voxel works only with the mesh
  collider's `offset`, as FT2 established.
- A band cell whose fluid neighbours are all cold and that has never burned stays inert. There is
  no heat conduction through the wood.

### 3.5 Off-switch

With input 7 unconnected nothing is allocated, dispatched or written, and `StepConstants` gains no
new behaviour. With fire on and the surface off, the output stays bit-identical to today. A guard
hash records it, as the fire-off guard does.

## 4. Testing

Each test is proven to fail under one mutation, per AGENTS.md. Small synthetic scenes (a plank and
a heat source) on the GPU; none depends on the FT3 pressure decision.

| # | Test | Mutation that must fail it |
|---|---|---|
| 1 | No heat, no ignition: cold gas, `burned` stays 0 for N frames and `char` is 0 everywhere | use `>=` with threshold 0, or ignore the temperature condition |
| 2 | Heat ignites: a hot cell next to one plank cell sets `burned > 0` only at or above the threshold | move the threshold comparison to the wrong operand |
| 3 | Budget closes: Σ`burned` equals Σ fuel gained by the fluid (fire's own `burning_rate` set to 0 so gas fuel is not consumed, closed domain) | drop the `/ fluid_neighbours` division |
| 4 | Depletion: `burned` never exceeds `load`, `char` reaches 1 and emission stops | remove the `min` against `load − b` |
| 5 | Front spreads: a plank under a steady heat source at one end ignites outward, with ignition frames non-decreasing with distance and at least k cells lit by frame N | zero the gather, so no fuel reaches the gas |
| 6 | Union: the load of a union of two colliders is the per-cell max | swap max for min |
| 7 | Determinism: two runs are bit-identical; a timeline restore matches a straight run | scatter with atomics in place of the gather (order-dependent) |
| 8 | Surface off: bit-identical to the pre-FT4 fire output (guard hash) | connect an empty load field |
| 9 | Sockets: `surface_fuel` without the fuel input is an error; a moving collider with `surface_fuel` is rejected at build | drop either check |

Test 5 is emergent and therefore the soft one. It is tied to a recorded seed, resolution and
threshold, and if the front does not spread for physical reasons the result is recorded, not
tuned away (the roadmap's rule on honest verdicts).

## 5. Done when

The roadmap's FT4 acceptance holds: no ignition without heat, a front spreads across a plank
under a steady heat source, the reservoir and emitted fuel balance, output is deterministic, and
each mutation above is recorded with its real failure output. `just check` passes, CI is green on
llvmpipe, and `AGENTS.md` and `.superpowers/sdd/progress.md` are updated.

## 6. Open after FT4

- A `char` shader and the export are FT5. The shack's real behaviour under flame is FT6, where
  preview's single substep may not bring flame to the shack at all (FT3b).
- Heat conduction and moving surfaces are NP2's concern if napalm needs them.
