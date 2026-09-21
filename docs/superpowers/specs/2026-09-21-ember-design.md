# Ember v1 — Umbrella Design

**Date:** 2026-09-21
**Status:** Approved for planning
**Scope of this document:** the normative decomposition of Ember v1 into four
pieces, and the solver behaviour those pieces must add up to. Each piece gets
its own spec, plan and implementation cycle. Only piece 1 is specified in full
so far: `2026-09-21-ember-core-sim-foundations-design.md`.

Parent: `2026-09-19-elements-suite-core-design.md` (roadmap row "Ember v1:
grid gas solver, VDB export, no ML").

---

## 1. Goal

A dense-grid smoke and fire solver that:

- previews **128³ at 10 fps or better** in the Blender viewport on Apple
  Silicon / Metal while parameters are being tweaked, and
- bakes **256³ to 512³** offline to a multi-grid OpenVDB sequence.

It uses no ML. `required_features` stays `wgpu::Features::empty()`.

### Non-goals for v1

- Sparse or bricked grids. The grid is dense and fixed to the domain.
- Liquids (Tide), terrain (Strata), procedural assets (Weave).
- Any ML. The ML work listed in the suite spec (upres, video to smoke) is Ember ML, a later phase.
- Scenes with more than one domain.
- GPU multigrid, unless it lands as piece 2's stretch goal.

---

## 2. Decisions

**E1 — Staggered (MAC) velocity.** Velocity is three `R32Float` textures, one
per axis, each one cell larger along its own axis: `(nx+1, ny, nz)`,
`(nx, ny+1, nz)`, `(nx, ny, nz+1)`. Chosen over a cell-centred layout because
pressure projection on a staggered grid is exact and has no checkerboard mode.
Chosen over `Rgba32Float` because the WebGPU baseline allows `read_write`
storage access only for `R32Float` (verified in `wgpu-types` 30.0.1,
`guaranteed_format_features`: `R32Float` is `s_all`, `Rgba32Float` is
`s_ro_wo`).

**E2 — Trilinear interpolation is done by hand.** Neither `R32Float` nor
`Rgba32Float` is filterable without the optional `FLOAT32_FILTERABLE` feature,
so every sampling step in advection loads eight texels and blends them in WGSL.

**E3 — Simulation state lives in core, behind stateful nodes.** Core owns
persistent fields, time, and a frame cache. A solver is a *stateful node* in an
acyclic graph, not a feedback loop wired from graph nodes, and not a stepper
living outside the graph. This keeps D2 of the suite spec (the graph is the
shared abstraction) and lets Tide reuse the same time and cache machinery.

**E4 — Emitters and colliders are fields.** The solver consumes fields, never
meshes. v1 starts with analytic emitters and colliders expressed as graph
nodes. Emitting from Blender objects arrives in piece 4, with the add-on
voxelizing a mesh into a signed-distance field (SDF) and sending it over the data plane.

---

## 3. Solver behaviour (piece 2's target)

It is fixed here so that piece 1 builds the right primitives.

Per substep:

1. **Emit.** Emitter fields add density, temperature, flame and velocity.
2. **Forces.** Buoyancy from density and temperature, plus wind and gravity.
3. **Vorticity confinement.**
4. **Advect velocity.** Semi-Lagrangian, with trilinear interpolation done by hand (E2).
5. **Pressure projection.** Red-black Gauss–Seidel with a fixed iteration
   count per quality preset. Collider SDFs give solid-wall boundaries.
6. **Advect scalars.** Density, temperature and flame.
7. **Dissipate.**

`dt` comes from the document's `fps`. Substeps are chosen by the CFL
condition, with a per-quality maximum.

**Stretch goal for piece 2:** a multigrid pressure solve. v1 does not need it.

---

## 4. Determinism

The same document and seed on the same machine give **bit-identical frames**.
That holds however a frame is reached: in order, by scrubbing, or after a cache
eviction. Comparisons across backends (Metal against lavapipe) use a tolerance.

---

## 5. Pieces

| # | Piece | Delivers | Proof it works |
|---|---|---|---|
| 1 | **Core sim foundations** | persistent state, timeline and frame cache, time reaching nodes, staggered vector fields, several consumers per output, explicit acquire contract, fields released after their last use | `core.accumulate` steps, resets and scrubs deterministically against a CPU closed form |
| 2 | **Solver** (new crate `elements-ember`) | `ember.smoke_solver`; analytic emitters (sphere, box, noise-modulated) and colliders (sphere, box) with animatable transforms; forces | the validation scenes in §6 |
| 3 | **Export and handoff** | multi-grid VDB (density, temperature, flame, and velocity as vec3 resampled to cell centres); bake sequences; in-memory volume handoff to Blender; f16 on the wire; disk-backed cache if RAM is not enough | a VDB sequence opens in Blender by hand; Live preview with no disk round-trip |
| 4 | **Blender UX** | emitters and colliders from Blender objects (the add-on voxelizes to an SDF); playback of the cache; Live preview done properly | manual verification in an interactive Blender session |

The pieces are built in this order. Each depends only on the pieces before it.

---

## 6. Validation scenes (piece 2)

Each must be proven able to fail by a single mutation, per CLAUDE.md.

- **Still domain.** No emitters and no forces: velocity stays exactly zero.
- **Divergence-free.** After projection, max |div u| falls below a tolerance
  scaled by the iteration count.
- **Buoyant blob.** A hot blob rises, and its density centre of mass moves
  monotonically upward over N frames.
- **Collider.** No density enters a collider's interior (SDF < 0) beyond a tolerance.
- **Determinism.** Frame 40 is bit-identical whether reached in order or after scrubbing.

---

## 7. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| wgpu compute on Metal is too slow for 128³ at 10 fps with Gauss–Seidel | High | Measure in piece 2 first; quality presets trade iterations; multigrid is the stretch goal |
| Snapshot memory at 256³+ exceeds RAM | Medium | Cache budget with eviction (piece 1); disk-backed cache (piece 3) |
| Metal and lavapipe numerics diverge | Medium | Bit-exactness only within one backend; tolerances across backends; CI must run before piece 2 |
| `downlevel_defaults` allows only 4 storage textures per shader stage | Medium | Solver kernels read through `texture_3d<f32>` + `textureLoad` and write through storage; six velocity faces are never all bound as storage |
| `max_buffer_size` is 256 MiB; a 512³ `R32Float` readback is 512 MiB | Medium | Piece 3 exports read back in z-slabs |
| "Ember" name collision (Ember.js) | Low | Trademark search before public release (suite spec §7) |

**Precondition.** The Core v1 close-out items in `.superpowers/sdd/progress.md`
(CI has never run; the Live toggle is unverified) should be closed before
piece 1 implementation starts. Otherwise Ember is built on a base only ever
tested on Metal.
