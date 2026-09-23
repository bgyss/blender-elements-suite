# Ember Piece 2b-1 — Solver Correctness Design

**Date:** 2026-09-22
**Status:** Approved for planning
**Parent:** `2026-09-21-ember-solver-design.md` (piece 2). Its §6 lists the risks
this spec resolves.
**Branch base:** `main` after the hardening merge (PR #2, 0182146). This work
needs `GpuContext::wait` and the scoped error handling from it.

## 1. Goal and split

Piece 2b is built in three spec/plan cycles, in order (user decision, 2026-09-22):

| Cycle | Content |
|---|---|
| **2b-1 (this spec)** | CFL substepping, RK2 + MacCormack advection, vorticity confinement, dissipation, per-face boundaries, storing `p`, submission splitting, skipping unwanted outputs, quality presets |
| 2b-2 | box and noise-modulated emitters, sphere and box colliders, wind, flame, animatable transforms, the collider validation scene |
| 2b-3 | the Mantaflow benchmark (piece 2 spec §5) |

2b-1 resolves risks (b), (c), (d), (f) and (h) of piece 2 spec §6. Risk (e),
unconverged pressure, stays open: multigrid remains the stretch goal.

The boundary code is written so 2b-2's collider solid mask plugs into the same
wall test (§4.2). No collider work is done here.

## 2. Parameters

`ember.smoke_solver` gains these fields. Each has a default, so existing
documents still load. They do not step identically: the `preview` defaults
switch them to RK2, MacCormack and CFL substeps. A 2a document that needs 2a's
behaviour sets `advection = "semi_lagrangian"` and `max_substeps = 1`, which
leaves the RK2 backtrace as the only difference.

| Param | Default | Meaning |
|---|---|---|
| `quality` | `"preview"` | preset that fills every field the document leaves unset (§6) |
| `cfl` | from preset | target maximum cells travelled per substep; (0, 10] |
| `max_substeps` | from preset | cap on CFL substeps; 1..=16 |
| `pressure_iterations` | from preset | as in 2a; 1..=1000 |
| `advection` | from preset | `"maccormack"` or `"semi_lagrangian"` |
| `vorticity` | `0.0` | confinement strength ε, 1/s; finite, ≥ 0 |
| `density_dissipation` | `0.0` | exponential decay rate, 1/s; finite, ≥ 0 |
| `temperature_dissipation` | `0.0` | exponential decay rate, 1/s; finite, ≥ 0 |
| `boundaries` | 5 walls, `+z` open | per face (`-x`, `+x`, `-y`, `+y`, `-z`, `+z`): `"wall"` or `"open"` |

`buoyancy_density` and `buoyancy_temperature` are unchanged.

**`substeps` stays as an alias for `max_substeps`**, so 2a documents and the
gate scene keep loading. A document that sets both is rejected. With CFL on,
a fixed count only makes sense as a cap.

Every field is validated when the document loads, as a `DocError`. Documents
are untrusted, and the daemon must not panic. An unknown preset name, face name
or boundary kind is also a `DocError`.

## 3. One frame

1. **CFL.** A max-reduction (§4.6) over `|u|` on all three face textures, then a
   4-byte readback through `GpuContext::wait`:
   `n = clamp(ceil(max|u| · dt / (cfl · dx)), 1, max_substeps)`, and `h = dt / n`
   for the whole frame (user decision: one measurement per frame, before stepping).
   `n` depends only on the entering state, so frames stay bit-deterministic
   however they are reached. The daemon already blocks every frame to publish,
   so the sync costs a reduction and one round-trip.
2. **Each of the `n` substeps:** emit → buoyancy → vorticity → advect velocity →
   project → advect scalars and dissipate. This is umbrella §3's order.
3. **Outputs:** only the wanted ones are copied (§4.7).

When the cap binds (the uncapped `n` would exceed `max_substeps`), the frame
still runs at `max_substeps`, and `EvalStats` gains a `cfl_clamped` count, so
the add-on can say "raise quality or max substeps" instead of failing silently.

A non-finite maximum speed is a new `NodeError::SolverDiverged { node }`. It is
never cast to an integer (`ceil(NaN) as u32` is 0).

Mantaflow orders its step differently: dissolve, advect, vorticity, buoyancy,
forces, walls, pressure (§8). Ember keeps umbrella §3's order. The difference
goes in 2b-3's mapping notes and is not copied.

## 4. Kernels

### 4.1 Advection

**RK2 midpoint backtrace**, for velocity faces and scalar cells alike:
`x_mid = x − ½h·u(x)`, `x_back = x − h·u(x_mid)`. This replaces 2a's single
Euler step and doubles the velocity sampling per backtrace.

**MacCormack** (Selle et al. 2008) when `advection = "maccormack"`, as three
dispatches per advected quantity (per face axis for velocity):

1. forward: `q̂ = A(q)`, the RK2 semi-Lagrangian step
2. backward: `q̃ = A⁻¹(q̂)`, the same step with the velocity negated
3. correct: `q_new = q̂ + ½(q − q̃)`, clamped to the min and max of the 8 texels of
   `q` around the forward backtrace point

The correction kernel recomputes the forward backtrace point instead of storing
it. `q̂` and `q̃` are pooled scratch fields. Each dispatch writes one storage
texture and reads everything else through `texture_3d`, so every kernel stays
within the 4-storage-texture limit. The advecting velocity is the velocity that
enters the stage, for all three passes.

With `advection = "semi_lagrangian"`, only the forward pass runs.

### 4.2 Boundaries

The uniform gains `open_mask: u32`, one bit per face. `is_wall(axis, i)` reads
it. **Everything wall-related goes through that one function**, and in 2b-2 it
becomes `face_is_wall || solid(cell)`.

- **Wall face:** zero normal velocity, before projection and after. Pressure is
  Neumann: the out-of-domain neighbour is left out of the sum.
- **Open face:** pressure is Dirichlet, `p = 0`, so the neighbour counts with
  value 0. Velocity extrapolates with zero gradient, which is 2a's clamp.
  **Scalars sample the ambient value 0 beyond an open face.** A backtrace that
  leaves through an open face picks up clean air instead of the edge layer. This
  resolves risk (d). Inflow through an open face stays allowed.
- **Buoyancy** skips boundary faces through `is_wall` and the open mask. It no
  longer hard-codes `z = 0` and `z = n`.
- **Fully closed domain** (`open_mask == 0`): the Neumann system is singular,
  and p is defined only up to a constant. After the solve, p's mean is
  removed with a GPU sum reduction and a subtract pass, with no readback, so
  the warm start cannot drift. This matches what Mantaflow's
  `zeroPressureFixing` achieves, but without pinning a cell, which
  Gauss–Seidel would converge poorly around. *Revised while writing the plan:*
  removing the divergence's mean before the solve was dropped. Gauss–Seidel
  converges in every non-constant component even on an inconsistent system,
  and the only effect of inconsistency is a drift in the constant, which
  removing p's mean takes out. A closed box's divergence also sums to zero up
  to rounding, since it telescopes to wall faces. No test could observe the
  step.

### 4.3 Pressure

- The `pressure` slot **stores `p`**, not `φ = h·p`. The sweep solves
  `∇²p = div / h`, and the gradient stage does `u −= h·∇p`. The warm start no
  longer depends on `h`, which resolves risk (b) as CFL varies `h` between
  frames. The uniform carries `dx² / h` precomputed.
- **Submission splitting (risk f).** The pressure loop flushes the
  `ComputeBatch` every `K = max(1, ⌊2³⁰ / cells⌋)` iterations. At the measured
  0.19 ms per iteration at 128³, that is about 100 ms of GPU work per submission.
  At 128³, K = 512, so every preset stays a single submission; at 512³, K = 8.
  `ComputeBatch` gains a flush that submits and starts a new encoder. Fields a
  stage retires still wait until the substep's last submission. The constant is
  a `const` with an override the tests can set.

### 4.4 Vorticity confinement

Between buoyancy and velocity advection, when `vorticity > 0`. Setting ε = 0
skips both dispatches, so the step is bit-identical to one without them.

- `curl`: writes ωx, ωy, ωz and |ω| at cell centres (exactly 4 storage
  textures), from face velocities averaged to cell centres on the fly, with
  central differences clamped at the domain edge.
- `confine`, per face axis: `N = ∇|ω| / (|∇|ω|| + 1e-6)`,
  `f = ε · dx · (N × ω)` in each of the two cells the face separates, averaged
  to the face, then `u += h · f`. Wall faces are left at zero.

ε is in 1/s, the form in Fedkiw et al. 2001. Confinement is resolution-dependent
by nature: finer grids resolve more vorticity. The spec says so rather than
promising that a 128³ preview matches a 512³ bake here.

### 4.5 Dissipation

Fused into the final scalar advection pass (the MacCormack correction, or the
semi-Lagrangian pass), which writes `q_new · exp(−rate · h)`. It adds no
full-grid pass, and dissipation still comes after advection. A rate of 0
multiplies by exactly 1.

### 4.6 `reduce` (in `elements-core::gpu`)

A generic reduction of a field to one `f32` (max of |x|, or sum) into a small
storage buffer, as a fixed tree: per-workgroup partials, then one final
workgroup. The order is fixed, so the result is bit-deterministic within one
backend. A readback helper returns the value through `GpuContext::wait`. It
lives in core because Tide needs it too. Max over the staggered velocity
reduces each face and takes the max of three.

### 4.7 Unwanted outputs (risk h)

Core gains `EvalCtx::output_wanted(index) -> bool`: the result socket, or a
remaining consumer count above 0. The solver copies only wanted outputs and puts
`Value::Scalar(0.0)` in the other slots. `Graph::run` already releases unwanted
outputs unseen, so no consumer can observe the placeholder. The doc comment on
`output_wanted` states this contract.

## 5. Errors

2a's rules carry over unchanged: every taken state value and scratch field goes
back to the pool before an error returns, and state is written back only after
a whole frame succeeds. The new failure points are the CFL readback (a
`GpuError` through `GpuContext::wait`), a divergent maximum speed
(`SolverDiverged`), and failures between split submissions, where fields
retired by the substep are released only after the in-flight submission's
`wait`.

## 6. Quality presets

A preset fills only the fields a document leaves unset. An explicit field
always wins.

| Field | `preview` | `final` |
|---|---|---|
| `pressure_iterations` | 160 | 480 |
| `advection` | `maccormack` | `maccormack` |
| `cfl` | 1.0 | 1.0 |
| `max_substeps` | measured (below) | 8 |

`final`'s 480 comes from `docs/bench/iteration-sweep.md` (ratio 0.0076). It has
no time budget; its 256³ frame time is recorded for information.

**`preview`'s `max_substeps` is decided by a pre-registered rule.** It is fixed
here, before any number exists, and is not revised to fit the results.

- Scene: `plume` at 128³, `cfl` 1.0, N = 160, `maccormack`, `vorticity` 0.
- Sweep `max_substeps` ∈ {1, 2, 3, 4}. Measure as the gate does (piece 2 spec
  §4.3): 24 warm-up frames, then frames 25–48, and the median of 3 runs. The
  measurement here is the **full frame**: the CFL reduction and readback plus
  every substep. Record `cfl_clamped` per row.
- `preview.max_substeps` is the **largest** value whose median full frame is
  ≤ 100 ms.
- If even `max_substeps = 1` fails, `preview.advection` becomes
  `semi_lagrangian`, and the sweep runs again.

The table goes to `docs/bench/presets.md`. The runner is `just bench-presets`,
a sibling of `bench-gate`, and it is not part of `just check`. The user confirms
the result, as with the gate. It is a human decision, never a CI check.

**Expected risk:** at 128³, a 1–2 m/s plume moves about 3–5 cells a frame, so
honest CFL substeps probably do not fit in 100 ms. `preview` will likely cap at
2 and report `cfl_clamped` on every frame. That is what the stat exists to say.

## 7. Testing

Every test records the single mutation that makes it fail (CLAUDE.md). Grids are
≤ 32³ and non-cubic, so an axis swap cannot pass by symmetry. Kernel tests
compare GPU output with a CPU reference, to about 1e-5.

| Test | Asserts | Mutation |
|---|---|---|
| `reduce` | max and sum match the CPU on 7×5×3; bit-identical across repeated runs | skip the last partial workgroup |
| CFL | a known uniform speed gives the expected `n`; a speed past the cap runs at `max_substeps` and increments `cfl_clamped` | `floor` for `ceil` |
| CFL, divergent | a NaN in the velocity returns `SolverDiverged` and balances the pool | clamp NaN to 1 substep |
| RK2 | solid-body rotation matches a CPU RK2 reference | Euler backtrace |
| MacCormack | matches the CPU reference; a smooth bump keeps a higher peak than semi-Lagrangian after k steps; a step input never leaves the source min/max | remove the clamp |
| Dissipation | fluid at rest: `q · exp(−rate·h)` to 1e-6 | `1 − rate·h` |
| Open face | inflow across an open face samples 0, not the edge value | revert to the clamp |
| Closed domain | from a warm start offset by 3, p's mean is ≈ 0 after the solve; a closed 16³ plume meets 2a's divergence-ratio rule | skip the mean removal |
| Stored `p` | when `n` changes from 1 to 3 mid-run, the divergence ratio stays within 10% of a run at a constant 3 | store `h·p` again |
| Split submissions | forcing K = 1 is bit-identical to one submission | drop the dispatch recorded just before each flush |
| Vorticity | `curl` and `confine` match the CPU; confinement strengthens a 16³ plume's total |ω| by at least 5% | flip the cross product |
| `output_wanted` | with only density consumed, pool acquisitions fall by the unwanted copies | always return true |
| Params | `cfl` 0, `max_substeps` 17, `substeps` together with `max_substeps`, a negative rate and an unknown preset are each a `DocError` | drop the `cfl` bound |

All of 2a's validation scenes (still domain, divergence-free, buoyant blob,
determinism at frame 40 in order, after scrubbing and after eviction) run again
under the new defaults: `maccormack`, CFL. The `plume` bench scene gains the new
fields, so 2b-3 inherits them.

## 8. Mantaflow facts verified for 2b-3

These were read from the Mantaflow script embedded in the local Blender 5.2.2
binary (`strings` on `Blender.app/Contents/MacOS/Blender`, `smoke_step_$ID$`),
and are recorded here as inputs for 2b-3:

- Every gas quantity is advected with `advectSemiLagrange(..., order=2)`:
  MacCormack. This is why 2b-1 adds it.
- Pressure is `solvePressure` with preconditioner `PcMGStatic`, or `PcMGDynamic`
  when a moving obstacle is present: multigrid-preconditioned conjugate
  gradient, converged, where Ember uses a fixed number of Gauss–Seidel
  iterations. Expect plume shape to differ (risk e); the results must say so.
- Closed domains use `zeroPressureFixing=domainClosed`.
- Vorticity and dissolve strengths are scaled by `timestep / frameLengthUnscaled`,
  making them per-frame quantities. The ε mapping must convert.
- Step order: dissolve, advect (density, heat, fuel, velocity), reset outflow,
  vorticity, buoyancy from heat and from density, force fields, reset in
  obstacles, wall boundary conditions, pressure. Emission runs before the step.

## 9. Task order

1. Core: `reduce` and `EvalCtx::output_wanted`.
2. Stored `p`, `open_mask` boundaries, ambient scalars at open faces, closed domain.
3. Submission splitting.
4. RK2 backtrace, MacCormack, fused dissipation.
5. Vorticity confinement.
6. CFL substepping, the new params, the `substeps` alias, `cfl_clamped`, `SolverDiverged`.
7. The preset sweep and its recorded decision; update CLAUDE.md, the piece 2
   spec's status, and the README.
