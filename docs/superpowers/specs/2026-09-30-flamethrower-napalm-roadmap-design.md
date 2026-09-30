# Flamethrower-vs-shack and napalm: roadmap and design

**Date:** 2026-09-30
**Status:** reviewed and approved by the user (2026-09-30). FT0-FT3 were implemented on
branch `worktree-flamethrower-ft0-ft3` (plan: `docs/superpowers/plans/2026-09-30-flamethrower-ft0-ft3.md`).

**Plan FT0-FT3 outcome.** FT0-FT2 are done. FT3's results are recorded in
`docs/bench/fire-divergence.md` with a PROVISIONAL recommendation. Two findings change the
next plan: flame does not reach the shack at preview's one substep (the shot needs more
substeps and/or a bigger flame), and filling the mesh SDF every frame has a real cost
(risk (o) in the piece 2 spec), which needs a static-pose cache. `shack()` also needs input
validation before it becomes a node parameter.
**Builds on:** Ember fire (2b-4), the research brief
(`docs/research/2026-09-26-particles-fluids-terrain-brief.md`), the core design's Tide row.

## 1. Purpose and success criteria

Two efforts, in order:

- **Track A: flamethrower vs. shack.** Reproduce the idea of Jason Key's "flamethower vs. shack"
  (EmberGen 1.2.11, 30 s, posted 2026-09-01: a fast flame jet hits a dark, broken wooden shack,
  with rolling smoke and flame wrapping the structure) inside this project, on the Ember fire/smoke
  solver. Deliverable: **a Blender-rendered shot** (Cycles, clip plus stills) produced by a scripted
  scene baked through `elements-cli`/`elementsd`. The shack **ignites and burns** (core scope, decided
  2026-09-30).
- **Track B: napalm.** A sticky, viscous, burning liquid lobbed onto a surface. It starts only after
  Tide (the FLIP liquid solver) passes a maturity gate.

Success is an honest side-by-side with the reference, with a recorded verdict, not parity with
EmberGen's speed or detail. The reference clip is used for comparison only. **Do not copy the author's
shack or other assets**; model our own procedural shack. Reference frames stay out of the repository.

Non-goals: live in-Blender preview (needs the node editor, multi-grid handoff and the preview budget;
a later milestone chain), heat-wave distortion simulated optically (done in comp from a temperature
pass at most), ML detail.

## 2. Starting point (verified 2026-09-30)

- Ember has fuel, burn, `flame` and temperature outputs, box and sphere emitters and colliders, unions
  and wind. Fire divergence is 4–28× Mantaflow's at frame 60 (64³/128³/256³); a 128³ fire preview
  frame is 103.5 ms against a 100 ms budget, measured under load (risk n).
- `collider.rs` names a "piece 4 mesh voxelizer" as the source of the general SDF + face-velocity
  pair. No mesh collider exists.
- Emitters are box or sphere shapes, with noise and velocity emission. No nozzle or cone jet.
- `elements-cli bake` writes one output grid per run; the fire render needed two simulations.
- Tide does not exist. Core design row: "Tide v1: FLIP liquids, classical, follows Ember v1".

## 3. Approach and rationale

Demo-first with a quality gate. Build only what the shot needs, in dependency order; put the fire
divergence experiments inside the plan (FT3) as a time-boxed gate, not a prerequisite. Quality-first
would delay any visible result; Tide-first would delay a gas-only demo by a whole solver. Napalm
stays a roadmap until the Tide gate is met.

## 4. Track A milestones

Each milestone is its own spec-and-plan cycle where non-trivial. All follow the repo rules:
mutation-proven tests, recorded benchmarks, idle-machine reruns for timing claims, CI on llvmpipe
checked separately from local Metal.

| # | Milestone | Done when |
|---|---|---|
| FT0 | **Shot spec.** Storyboard in our own terms: procedural plank shack, jet path, camera, domain, resolution, frame count. Verify Ember supports non-cubic domains (else decide). Fix the memory and time budget on the M1 Max (risk g). | Acceptance list committed; budget fits the hardware. |
| FT1 | **Nozzle jet emitter:** cone shape, exit velocity, noise, fuel rate, pulse keyframes. | Emitted momentum and fuel rate match analytic values; determinism test; each test shown to fail under a single mutation. |
| FT2 | **Mesh collider (piece 4 voxelizer):** mesh to signed distance and face velocity, concave and thin-walled. | Matches analytic SDFs on box and sphere meshes; a thin-plank leak test passes; risk (m) measured. |
| FT3 | **Fire quality gate.** Run the handoff experiments: Euler for all grids, `flame_vorticity = 0`, `final` preset. Measure divergence in a jet-into-collider scene, plus the fuel-ratio budget at 128³/256³. | Numbers recorded; the user decides if quality suffices or a solver change is scoped. Time-boxed. |
| FT4 | **Surface ignition (core).** The shack's surface carries a fuel reservoir (wood budget) on the collider band. It ignites when adjacent gas temperature exceeds ignition temperature, emits fuel and heat into adjacent fluid cells while it burns, and depletes. Spread is emergent from gas heat; output a `char` mask for shading. | No ignition without heat; a front spreads across a plank under a steady heat source; reservoir plus emitted fuel balances; deterministic; single-mutation failures recorded. |
| FT5 | **Multi-grid VDB export and Blender render template:** density, flame, temperature, fuel, velocity and char in one VDB per frame; Cycles blackbody flame, smoke, a fuel pass; a scene script following the `render_compare` pattern. | One bake yields all grids; Blender renders a frame from them; `just` recipe exists. |
| FT6 | **Shot assembly:** scripted scene, bake at the FT0 resolution, render clip and stills, side-by-side with the reference, verdict recorded including what we lack. | Clip and stills in `docs/`; verdict and benchmark numbers recorded; load noted. |

Optional after FT6: a temperature pass for heat-wave distortion in comp; sparks and embers.

Dependencies: FT4 needs FT1 (ignition source) and FT2 (surface); FT5 is independent of FT4's internals
but carries its `char` output; FT3 can run beside FT1–FT2.

Known risks: (g) GPU out-of-memory on unified memory at higher resolution; (m) thin colliders
under-converging under preview's ×4 MGPCG; (n) fire cost; thin planks may need a finer grid than the
jet; a surface reservoir adds state and a new kernel pair that must stay bit-identical with fire off.

## 5. Track B milestones (gated)

| # | Milestone | Done when |
|---|---|---|
| G-T | **Tide maturity gate** (research brief T1): FLIP with reseeding and pressure projection on dam break, splash, moving obstacle, thin sheet. | Volume loss, pressure residual and penetration within agreed bounds; bake cost benchmarked against Mantaflow on matched scenes. |
| NP0 | Napalm spec: behaviours (viscous, clings, burns long, black smoke, detached burning blobs) and acceptance; method survey. | Written and reviewed. |
| NP1 | Viscosity and surface adhesion in Tide. | A blob adheres to a wall at a measurable contact angle; viscosity scaling test. |
| NP2 | **Liquid-to-gas fuel coupling** as a typed graph edge (per Safvati and Ghoniem, "FLIP Fluids as a Bi-directional Fuel Source in a Volumetric Fluid Simulation"): liquid surface emits fuel and heat; burning consumes liquid. | Fuel and mass budget closes, using the fuel-budget method. |
| NP3 | Ignition and spread: per-particle fuel mass, temperature-driven ignition, burn duration. | Fire spreads along the liquid surface and burns out on a measured timescale. |
| NP4 | Burning droplets and embers (secondary particles). | Tests defined in NP0. |
| NP5 | Napalm shot: lob onto the shack or a hillside; liquid meshing with a glowing shader plus volumetric flame; Blender render. | Rendered shot; verdict recorded. |

Dependencies: G-T first; NP2 reuses FT4's fuel-emission path and FT5's export; NP5 reuses FT2 and FT6
scene tooling. No claim of novelty: nothing found in the published literature covers sticky, viscous,
burning fluid end to end, so NP1–NP3 carry research risk and each needs an ablation against its
simpler baseline.

## 6. Open items for later cycles

- Whether the `char` mask and surface reservoir belong in `ember.collider` or a new node kind.
- Fire divergence: whether FT3 leads to a solver change (user decision).
- Tide licensing and any paper source: review before incorporating source or weights; no GPL code in
  `elements-*` crates.
- Node editor (`ember-node-editor`, paused) must cover the new sockets before an interactive
  workflow exists; out of scope here.

## 7. Next step

Plan FT0–FT2 first (writing-plans), run FT3 alongside, then FT4–FT6. Track B stays a roadmap until
G-T. Register the new spec in AGENTS.md's document list when the plan is executed.
