# Ember Piece 2b-2 — Scene Content Design

**Date:** 2026-09-23
**Status:** Approved for planning
**Parent:** `2026-09-21-ember-solver-design.md` (piece 2). Follows 2b-1
(`2026-09-22-ember-solver-2b1-design.md`), whose §4.2 lists the places a
collider's solid mask must reach.
**Branch base:** `main` at 9a225db (PR #5 merged).

## 1. Goal and scope

2b-2 gives the solver things to put in a scene. User decision, 2026-09-23:
everything on piece 2's list except flame.

- Box and sphere emitters with keyframed transforms, noise modulation and
  velocity emission.
- Box and sphere colliders, static or animated, with obstacle velocity.
- Wind.
- The collider validation scene (umbrella §6).
- The `plume_collider` and `plume_wind` benchmark scenes, which 2b-3 needs.

**Out of scope:**
- Flame and combustion (fuel, reaction, heat release, flame output). They
  become their own cycle, 2b-4, after the benchmark.
- Spatial force fields. Wind is uniform (§3.4).
- Cut-cell or fractional solids. Solids are voxelized (§3.2).
- Emitters and colliders from Blender objects. Those belong to piece 4, whose
  mesh voxelizer will produce the same SDF representation (§2.4).

Open follow-ups from 2b-1 (the idle-machine preset rerun, and risk (j)) are
recorded in the progress ledger. They do not block this work.

## 2. Nodes

Every emitter and collider is a graph node that outputs fields (umbrella E4).
All lengths are metres and all rates are per second, measured from the
domain's minimum corner, so a scene looks the same at any resolution.

### 2.1 Shapes and transforms

Emitters and colliders share both.

- **`shape`** is `{"sphere": {"radius": r}}` or
  `{"box": {"half_extents": [x, y, z]}}`, in the object's local frame.
- **`transform`** is `{"keys": [{"frame": f, "translate": [x, y, z],
  "rotate": {"axis": [x, y, z], "degrees": d}}, …]}`.
  - Keys are in strictly increasing frame order.
  - Translation interpolates linearly between keys, and rotation by slerp.
  - Before the first key and after the last, the transform holds.
  - One key makes a static object.
  - `translate` defaults to zero and `rotate` to none.
- **Evaluation happens at the frame's time.** The graph evaluates stateless
  nodes once per frame. Velocity is the slope of the curve between the two
  keys that bracket the frame: `v(x) = Δtranslate/Δt + ω × (x − c(t))`, where
  `c(t)` is the object's centre and ω is the angular velocity of that key
  segment's rotation. Outside the keyed range, velocity is zero.
- **One WGSL module, `shape.wgsl`,** holds the signed distance for each shape
  in local space, and the world-to-local transform, as a rotation matrix plus
  a translation carried in the uniform. The CPU computes the matrices; the
  GPU never interpolates.

### 2.2 `ember.emitter`

| Param | Default | Meaning |
|---|---|---|
| `shape`, `transform` | required | §2.1 |
| `density_rate`, `temperature_rate` | 0 | amount added per second where fully occupied |
| `velocity` | [0, 0, 0] | target velocity in m/s, world space; the emitter's own motion is added |
| `velocity_blend` | 0 | 1/s; 0 disables velocity emission (§3.3) |
| `noise` | none | `{seed, scale_m, amplitude, evolution}` |

Outputs, in this order:

1. density rate (Field)
2. temperature rate (Field)
3. occupancy (Field, in [0, 1])
4. emitted velocity (staggered VectorField)

Occupancy is `clamp(0.5 − sdf/dx, 0, 1)`, the one-voxel smoothed edge
`ember.sphere_emitter` already uses. The rates are the occupancy times the
rate times the noise factor. The emitted velocity at each face is `velocity`
plus the emitter's own motion `v(x)` there.

**Noise.** Rates are multiplied by `1 − amplitude · (1 − n)`, where `n` is
in [0, 1]:
- `n` is value noise at position `x / scale_m`, with a fourth coordinate
  `evolution · t` that makes the pattern change over time.
- It uses core's integer hash, with no transcendental functions, so it is
  deterministic across backends.
- `seed` is a `u64`, as every stochastic node's is.
- `amplitude` is in [0, 1], and `scale_m` must be above zero.

`ember.sphere_emitter` stays unchanged, so existing documents and the `plume`
scene keep loading.

### 2.3 `ember.collider`

Params: `shape` and `transform`. Outputs, in this order:

1. the SDF (Field, in metres, negative inside)
2. the obstacle velocity (staggered VectorField), `v(x)` at each face centre
   in m/s

### 2.4 Unions

Each union has two inputs. Chain them to combine more.

- **`ember.emitter_union`:** the rates add, occupancy is the maximum, and the
  emitted velocity is `(o₁·u₁ + o₂·u₂) / max(o₁ + o₂, ε)`, taken per face
  from the face occupancies.
- **`ember.collider_union`:** the SDF is the minimum, and each face takes the
  velocity of the collider whose SDF is smaller there.

Piece 4's mesh voxelizer will produce the same pair, an SDF and a velocity,
so the solver never needs a second collider representation.

## 3. Solver integration

### 3.1 Sockets

| Index | Input | Type | Rule |
|---|---|---|---|
| 0 | density rate | Field | required (as in 2a/2b-1) |
| 1 | temperature rate | Field | required |
| 2 | occupancy | Field | optional; requires 3 |
| 3 | emitted velocity | VectorField | optional; requires 2 |
| 4 | collider SDF | Field | optional; requires 5 |
| 5 | collider velocity | VectorField | optional; requires 4 |

Core gains `EvalCtx::input_connected(index) -> bool`. The graph already
allows an input to be left unconnected; reading one is what fails. Connecting
one input of a pair without the other is a `NodeError` that names both
sockets. Documents that wire only inputs 0 and 1 load and step exactly as
before.

### 3.2 The solid mask

**When it is built.** Once per frame, before the CFL measurement, and only
when a collider is connected. A `solidify` kernel writes a scratch cell field
that is 1 where the SDF at the cell centre is below zero and 0 elsewhere. It
is never stored in state; it is rebuilt from the document's time every frame,
so scrubbing stays deterministic. It stays fixed across the frame's substeps,
which is first-order but acceptable, because CFL limits a collider to about
a cell per substep.

**Where it applies.** Every place 2b-1 spec §4.2 lists:

- **Face wall test.** `is_wall(axis, i)` becomes the domain-boundary wall OR
  either adjacent cell solid.
- **Solid faces carry the obstacle's velocity.**
  - Advection, the MacCormack correction and the gradient stage write the
    collider's velocity component along the face's axis at solid faces,
    instead of zero.
  - Divergence then sees the flux a moving body pushes, and projection does
    not change it.
  - Confinement leaves solid faces alone.
- **Pressure stencil.**
  - A solid neighbour is Neumann: it is left out, like a wall.
  - A solid cell is not solved. Its p is written as 0, and fluid cells never
    read it.
  - In a closed domain, 2b-1's mean removal still runs over every cell.
    Solid cells' p then holds a constant that nothing reads, so this is
    harmless. A fluid pocket sealed off by solids is left to Gauss–Seidel:
    its constant mode drifts, but only its gradient reaches the velocity, and
    a constant has no gradient.
- **Scalar sampling (`texel()`).** A solid texel returns the clamp-to-nearest
  fluid value along the axis being clamped. The simpler "return 0" would
  darken smoke along walls.
- **Buoyancy and wind** skip solid faces.
- **Curl and confinement** treat solid neighbours like the domain edge.

Scalars inside solids are not zeroed. With no flow through solid faces,
advection carries nothing in, and §5's collider scene checks exactly that.

### 3.3 Velocity emission

Each substep, after emit and before buoyancy:

`u ← u + (u_e − u) · occ_face · (1 − exp(−velocity_blend · h))`

- `occ_face` is the larger occupancy of the face's two cells.
- The exponential form makes the blend independent of `h`.
- A large `velocity_blend` approximates setting the velocity outright.
- Solid faces are skipped.

### 3.4 Wind

A solver parameter, `wind: [f32; 3]` in m/s², default zero, finite. It is a
uniform acceleration added at every non-wall face alongside buoyancy, like
Mantaflow's wind effector without falloff. A spatial force-field node can
come later if a scene needs one.

### 3.5 Errors

- **Validated when the document loads (`DocError`):**
  - radius and half-extents above zero;
  - keys in strictly increasing frame order, and at least one key;
  - a nonzero rotation axis;
  - every number finite;
  - noise `amplitude` in [0, 1] and `scale_m` above zero;
  - `velocity_blend` at least 0.
- **At runtime:** a mismatched socket pair is a `NodeError`, and the pool
  rules of 2a and 2b-1 hold. The solid mask is a scratch field retired like
  any other.

## 4. Data flow per frame

1. Stateless nodes evaluate at the frame's time: the emitters, the colliders
   and the unions.
2. The solver takes its inputs. If a collider is connected, `solidify` builds
   the mask.
3. The CFL measurement runs, as in 2b-1.
4. Each substep runs: emit, velocity blend, buoyancy and wind, vorticity,
   advect velocity, project, advect scalars. Every stage honours the solid
   mask.
5. The wanted outputs are copied, as in 2b-1.

## 5. Testing

Every test records the single mutation that makes it fail (CLAUDE.md).
Grids are at most 32³ and non-cubic, except where a test says otherwise.

| Test | Asserts | Mutation |
|---|---|---|
| Box emitter | emitted total within 5% of the box volume × rate at 32³ and 64³; within 5% when the box is rotated 30° | drop the rotation from the world-to-local transform |
| Keyframes | the midpoint between two keys gives the linear translation and the slerp rotation; velocity equals Δtranslate/Δt; the transform holds before the first key and after the last | lerp the rotation instead of slerp |
| Obstacle velocity | a box spinning at a known ω has face velocities `ω × r` against a CPU reference | leave out ω × r |
| Noise | bit-identical for a seed; mean factor within 5% of `1 − amplitude/2`; the pattern changes over time when `evolution > 0`; the same pattern within tolerance at 32³ and 64³ | noise in grid units rather than metres |
| Emitter union | the rates add, occupancy is the maximum, and velocity is occupancy-weighted | max instead of sum for the rates |
| Collider union | the SDF is the minimum, and each face's velocity comes from the nearer collider | take the velocity from the farther collider |
| Velocity blend | still fluid in a fully occupied emitter reaches `u_e·(1 − exp(−rate·t))` | linear `rate·h` |
| Solid faces | after projection, every solid face carries exactly the obstacle's velocity component along the face's axis | skip the solid test in `is_wall` |
| Collider scene (umbrella §6) | plume past a static sphere at 32³: density inside the sphere stays at most 1% of the plume's peak over 40 frames; 2a's divergence rule holds | let `texel()` read solid texels |
| Moving collider | a box pushed through still air drives mean flow ahead of it in its direction of motion; the divergence rule holds | obstacle velocity zero |
| Wind | still air under uniform wind accelerates at the wind's rate, to 1e-5 | apply wind at wall faces too |
| Sockets | occupancy without velocity, or SDF without velocity, is an error; a 2-input document steps bit-identically to before | require all six inputs |
| Determinism | frame 40 is bit-identical in order, after scrubbing and after eviction, with an animated collider and an animated, noisy emitter | cache the solid mask inside the node across frames |

## 6. Benchmark scenes

`bench::Scene` gains two variants, from piece 2 spec §5.3, so 2b-3 can build
the Mantaflow side of each from the same definition:

- **`plume_collider`:** `plume` with a static sphere collider of radius
  0.25 m, centred 0.5 m above the emitter.
- **`plume_wind`:** `plume` with `wind = [0.5, 0, 0]` m/s².

Each serialises to an `.elements` document. Both are covered by a load-and-step
test like `the_plume_scene_loads_and_steps`.

## 7. Task order

1. Core `EvalCtx::input_connected`. Shapes and keyframed transforms: CPU
   evaluation, velocities, and the WGSL SDF module.
2. `ember.emitter` (sphere and box, rates, occupancy, emitted velocity) and
   `ember.emitter_union`.
3. Noise modulation.
4. `ember.collider`, obstacle velocity, `ember.collider_union`.
5. The solver's optional sockets, `solidify`, and the solid mask in every
   boundary place, with the collider scene.
6. Velocity emission and wind.
7. Determinism under animation, the two benchmark scenes, and docs (CLAUDE.md
   and the piece 2 spec's status).
