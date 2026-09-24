# Ember Piece 2b-3c — Solver Quality Design

**Date:** 2026-09-24
**Status:** Design approved; plan not yet written.
**Parent:** `2026-09-21-ember-solver-design.md` (piece 2). Follows 2b-3
(`2026-09-23-ember-mantaflow-benchmark-2b3-design.md`), whose results
(`docs/bench/results.md`) this cycle answers. It comes before 2b-3b (render
and latency).
**Branch base:** `ember-bench-2b3` at 9ae4cf0 (PR #8).

## 1. Goal and evidence

The user's goal: Ember should beat Mantaflow on every axis, not only speed.
2b-3 showed Ember 3–8× faster but behind on divergence and conservation, and
failing on wind. A whole-domain probe at 128³ (2026-09-24, figures in the
progress ledger; probe kept in the session scratchpad, not the repo) found
three root causes:

- **Divergence** is the fixed 160-iteration red-black Gauss–Seidel solve
  alone. With 1,000 iterations it is 0. Gauss–Seidel removes large-scale
  error at a rate that falls with n², so a fixed count converges less as
  resolution grows: masked RMS 0.0026 at 64³, 0.25 at 256³.
- **Mass gain** is mostly advection. A divergence-free plume in a closed box
  gains 5.0% from frame 60 to 80 with MacCormack and 2.2% with
  semi-Lagrangian; the unconverged solve adds about 0.4 points. The same run
  with an open top gains the same, so it is not a boundary bug.
- **Wind** as a uniform acceleration against closed side walls, with an open
  top at p = 0, has no pressure that balances it: the air near the top
  accelerates without bound (14 m/s, kinetic energy 0.06 → 165 by frame 90),
  one CFL-clamped substep makes advection unstable, and smoke is destroyed
  inside the domain (∫ρ∇·u down to −0.044/s). In a closed box the same wind
  is balanced exactly.

## 2. Scope and order

User decisions, 2026-09-24: one cycle in gated parts; multigrid as V-cycles
first and then MGPCG, keeping the faster that passes; a fixed cycle or
iteration count per preset; a global mass correction; wind as ambient
airflow.

1. Multigrid V-cycle (§3).
2. MGPCG around it (§3.5).
3. **The gate** (§4). A stop point.
4. Mass-conserving correction (§5).
5. Wind as ambient airflow, and the scenes (§6).
6. Benchmark rerun and records (§7).

**Out of scope:**
- **Mass-conserving semi-Lagrangian advection** (Lentine et al., 2011). It
  conserves locally at any CFL but is substantially more complex: each
  cell's mass is redistributed to where its backtrace lands. The user asked
  to look into it at a later date; it is recorded as future work in the
  piece 2 spec.
- Flux-form advection, and tolerance-based stopping.

## 3. Multigrid pressure

### 3.1 The problem

Unchanged: ∇²p = div / h, Neumann at walls and colliders, p = 0 beyond open
faces, the state slot holding p so the warm start from the previous frame
stays valid.

### 3.2 Levels

- Each level halves every dimension, rounding up, until the smallest is at
  most 8: five levels at 128³, six at 256³.
- A coarse cell is solid only if all 8 of its children are solid, so thin
  fluid gaps never close.
- The open-face mask is the same on every level.
- Each level is re-discretised at spacing 2ˡ·dx, so its stencil is exactly
  today's kernel.

### 3.3 The V-cycle

A correction scheme:

1. Fine residual r = b − Ap.
2. Restrict r by averaging the fluid children.
3. On each coarser level: 2 red-black sweeps, recurse, 2 sweeps in the
   reverse colour order (black, then red), so the cycle is symmetric.
4. Coarsest level: 32 sweeps.
5. Prolong the correction trilinearly onto fluid cells and add it.

Closed domains are singular, so each level's residual has its mean removed,
as the finest level's pressure already does.

### 3.4 Kernels and memory

- The existing smoother (`pressure.wgsl`) gains per-level uniforms: dims,
  dx², right-hand-side texture, solid texture.
- New kernels: `residual`, `restrict`, `prolong_add`.
- Each level holds a pressure, a right-hand side and a solid mask, about 3/7
  of one field in total, all from the pool.
- Red-black ordering and fixed-order reductions keep every frame
  bit-identical.

### 3.5 MGPCG

Conjugate gradients with one V-cycle (zero initial guess) as the
preconditioner, as Mantaflow does (McAdams et al., 2010).

- Four extra fields: r, z, d, q.
- Per iteration: q = Ad through the residual kernel's operator; two dot
  products, each a multiply pass and a `Sum` reduction into `ReduceTarget`
  slots; a one-thread kernel computes α and β from the slots, so nothing is
  read back to the CPU; update p and r; z = V-cycle(r); d = z + βd.
- It warm-starts from the previous frame's pressure by solving for the
  correction.

### 3.6 Selecting the solver

`SolverParams` gains `pressure_solver`: `gauss_seidel`, `multigrid` or
`mgpcg`, with a count for each (`pressure_iterations` stays the Gauss–Seidel
count; `pressure_cycles` counts V-cycles or PCG iterations). After the gate,
both presets move to the winner with the counts the sweep chose. Gauss–Seidel
stays available.

## 4. The gate

Pre-registered, not revised to fit the results.

- **Runner:** `just bench-solver`, writing `docs/bench/solver-gate.md`. At
  128³ and 256³, in `plume`, `plume_collider` and `plume_wind` (the new wind,
  §6), three solvers run back to back in one process, so machine load hits
  all of them alike:
  - Gauss–Seidel, 160 iterations (today's preview);
  - V-cycles, swept over 1–8 cycles;
  - MGPCG, swept over 1–16 iterations.
- **Measured:** the pressure-solve time per substep (the median over frames
  25–48 of the time from before the solve to a blocking wait after it), and
  the masked divergence RMS at frames 60 and 120 (2b-3 spec §4.1).
- **Rule:** a solver configuration **passes** when, at both resolutions and
  in all three scenes, its solve time is at most Gauss–Seidel's and its
  divergence at frames 60 and 120 is at most Mantaflow's for that scene and
  resolution in `docs/bench/results.md`. For `plume_wind` the Mantaflow
  reference is the rerun's (§7), since the scene changes; until then the
  `plume` value at that resolution stands in.
- **Choice:** the fastest passing configuration becomes the default.
- **Fail:** if none passes, work stops and the table goes to the user.
- The decision is a line the user fills in, never a CI check.

## 5. Mass-conserving correction

Every substep, right after density and temperature are advected (emission
happened earlier in the substep and is not corrected), for each of the two:

1. Before advection, reduce M₀ = Σq·dV into a `ReduceTarget` slot.
2. A new `boundary_flux` kernel sums, over every open domain face, the
   upwind value × the outward normal velocity × dA × h, with the velocity
   the advection used. Inflow brings ambient 0 and counts nothing.
3. After advection, reduce M₁.
4. A one-thread kernel computes s = (M₀ − outflow) / M₁, and a scaling pass
   multiplies the advected field by s.

- **Guards:** no change when M₁ ≤ 1e-12 or the target is ≤ 0; s is limited
  to [0.9, 1.1] as a documented safeguard.
- Solid cells are already zeroed and walls and collider faces carry no flux.
- `SolverParams` gains `conserve_mass: bool`, on in both presets. Off
  reproduces today's behaviour exactly.
- **Limits, stated in the docs:** the correction is global, spreading the
  removed error over the field in proportion to each cell's value; and the
  outflow is a first-order estimate per substep, which the correction then
  enforces.

## 6. Wind as ambient airflow

- **Parameters:** `wind` (m/s²) is replaced by `wind_velocity` ([f32; 3],
  m/s) and `wind_rate` (f32, 1/s, ≥ 0). A document still using `wind` fails
  validation with a message naming the replacements. Non-finite values and
  a negative rate are rejected.
- **Kernel:** `wind.wgsl` becomes u += (w_axis − u)(1 − e^{−rate·h}) on each
  axis's non-wall, non-collider faces, the emitter velocity blend's form:
  stable and bounded for any h, never overshooting w.
- **`plume_wind`:** `wind_velocity` [1, 0, 0] m/s, `wind_rate` 1/s, with −x,
  +x and the top open, the other sides and the floor closed.
- **Drift accounting:** `metrics` outflow, today measured through the top
  plane only (2b-3 spec §4.3), is extended to every open face: for a side
  face, the plane two cells in from it, as for the top, and a control volume
  excluding the layers beyond each plane. The results notes say so.
- **Mantaflow's twin** opens the same sides (`boundaries` already reaches the
  script). Blender's wind field `flow` setting pulls smoke towards the
  wind's velocity, the nearest match to relaxation. The task finds the
  `flow` and strength mapping and confirms it by bake (smoke at rest reaches
  about 1 m/s with a time constant of about 1 s), recorded in
  `docs/bench/mantaflow-notes.md`. Mantaflow still applies it only to smoky
  cells, and the results say so.

## 7. Testing, rerun and records

- **V-cycle:** analytic Poisson problems with known solutions, in an
  all-wall box, with one open face, and around a solid sphere. The residual
  falls by at least 10× per cycle, and the measured rate is recorded.
  Gauss–Seidel is the reference.
- **MGPCG:** converges on the same problems in fewer iterations than plain
  V-cycles.
- **Mass:** in a closed box, total mass is constant to rounding over 60
  frames with `conserve_mass` on, and drifts measurably with it off. With an
  open top, mass plus cumulative boundary flux is constant.
- **Wind:** the relaxation reaches w without overshoot for any h; an old
  `wind` document is rejected; `plume_wind` at 64³ keeps its smoke until it
  leaves sideways (no loss inside the domain before outflow) and its largest
  speed stays bounded (at most the wind speed plus the buoyant plume's).
- **Determinism:** the frame-40 bit-identity tests run with multigrid and
  the mass correction on.
- Every test is proven able to fail by a single-change mutation.
- **Rerun:** the idle-machine preset rerun (risk (k)) then the full
  `just bench`; if the machine is not idle, runs are flagged as before.
  `results.md`'s Summary compares against the 2b-3 run.
- **Records:** close risk (l) and update risk (j) in the piece 2 spec; add
  Lentine et al. there as future work; update CLAUDE.md.

## 8. Risks

- **(a) Coarse levels with colliders.** "Solid only if all children are
  solid" can leave coarse cells that barely touch fluid, slowing V-cycle
  convergence near colliders. MGPCG is the mitigation; the gate measures it.
- **(b) Solve cost on naga Metal.** Many small coarse-level dispatches may
  cost more in launch overhead than they save. The gate measures real time,
  and level count is the lever.
- **(c) The global mass correction** can hide a local error. The per-scene
  metrics and render in 2b-3b are where that would show.
- **(d) Load.** As in 2b-3, timings are only as clean as the machine; the
  gate compares solvers in one process to limit this.
