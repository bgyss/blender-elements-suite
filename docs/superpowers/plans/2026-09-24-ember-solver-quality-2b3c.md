# Ember Piece 2b-3c — Solver Quality Implementation Plan

> **Superseded in parts.** Where this plan and the spec, `docs/bench/solver-gate.md`
> or the task reports (`.superpowers/sdd/2b3c/task-*-report.md`) disagree, those
> win; the plan is kept as written. Two parts in particular were replaced:
> the gate's per-frame timing (median frame times), reversed in 4615095 to
> time the pressure solve itself as spec §4 asks; and Task 1's original
> multigrid design (coarsening until a level is ≤ 8 cells, 32 coarsest-level
> sweeps, averaging restriction), replaced by a symmetric restriction
> (κ·Pᵀ, which MGPCG needs) coarsened to a single cell with one red-black and
> one black-red sweep there.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Ember beat Mantaflow on divergence and mass conservation
while keeping its speed lead, and make wind a bounded ambient airflow; then
rerun the benchmark.

**Architecture:** A geometric multigrid V-cycle reuses the existing
red-black smoother at every level, and MGPCG wraps it; a pre-registered gate
picks the fastest configuration that matches Mantaflow's divergence. A
global mass correction runs after scalar advection, entirely on the GPU. Wind
becomes an exponential relaxation towards `wind_velocity`, and `plume_wind`
gets open sides.

**Tech Stack:** Rust, wgpu 30 compute (WGSL) on Metal, `elements-core`
reductions (`ReduceTarget`, `ReduceOp::Sum`), Blender 5.2.2 for the
Mantaflow twin.

**Spec:** `docs/superpowers/specs/2026-09-24-ember-solver-quality-2b3c-design.md`

## Global Constraints

- `just check` passes before every commit.
- Scalar fields are `R32Float`; `required_features` stays empty; at most 4
  storage textures per shader stage (read neighbours through
  `texture_3d<f32>` + `textureLoad`).
- `#![forbid(unsafe_code)]` in every crate except `elements-ipc`. Crate
  manifests use `dep.workspace = true`.
- Kernels with solids include `solid.wgsl` and declare
  `var solid: texture_3d<f32>`; with no collider a 1×1×1 placeholder is
  bound and `has_solids` = 0 keeps it unread.
- Frames stay bit-identical within one backend on one machine: in order,
  scrubbed, and after eviction. No CPU readback inside a solve or a
  correction.
- Uniform mirrors keep their const size asserts (`KernelParams` is 64
  bytes).
- **Every new test is proven able to fail.** Apply a single-change mutation
  to the code it covers, watch it fail, restore, and record the real output
  in the ledger (`.superpowers/sdd/progress.md`). Replace an equivalent
  mutant and record both.
- Timings are never pass/fail tests. The gate is a recorded human decision.
- Never set `WGPU_BACKEND` locally. Commit with plain git (jj-colocated; do
  not run `jj`). Commits: imperative subject, a body explaining why, ending
  with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TSRpZQEY9fnKQqHLh5WHpU
  ```
- `crates/*` is Apache-2.0 OR MIT; Blender-side Python under `tests/bench/`
  is GPL-3.0-or-later. Published algorithms are reimplemented; no GPL or
  Mantaflow source is copied.

## Deviations from the spec

- **Gate timing is per frame, not per solve.** The spec §4 measures the
  pressure-solve time. Runs of one scene differ only in the solver, so
  comparing median frame times (`eval_frame` plus a blocking wait, frames
  25–48) is the same comparison without instrumenting inside the solver.
  `solver-gate.md` says so.
- **The sweep is staged to keep the run near an hour.** Counts rise at 128³
  until the divergence rule holds in all three scenes; that count is then
  checked at 256³, stepping up if needed; only the final candidates are
  timed against Gauss–Seidel at both resolutions.
- **No one-thread kernels.** The spec §3.5 and §5 compute α, β and the mass
  scale in a one-thread kernel. Each consuming kernel instead reads the
  reduction slots and computes the scalar itself: identical arithmetic in
  every thread, one fewer dispatch.
- **`mass_below` becomes `mass_inside`** in `metrics::FrameMetrics`, since
  with open sides the control volume is no longer "below" a plane (Task 4).
- **Order.** Wind (Task 4) comes before the gate (Task 5), not after it as in
  spec §2, so the gate measures `plume_wind` with the new, bounded wind.

## File structure

| File | Responsibility | Task |
|---|---|---|
| `crates/elements-ember/src/kernels/multigrid.rs` | levels, restriction, prolongation, residual, V-cycle | 1 |
| `crates/elements-ember/src/kernels/shaders/residual.wgsl`, `restrict.wgsl`, `restrict_mask.wgsl`, `prolong_add.wgsl` | multigrid kernels | 1 |
| `crates/elements-ember/tests/multigrid.rs` | Poisson tests | 1, 2 |
| `crates/elements-ember/src/kernels/mgpcg.rs`, `shaders/pcg.wgsl` | MGPCG | 2 |
| `crates/elements-ember/src/solver.rs` | `pressure_solver`, `pressure_cycles`, `conserve_mass`, `wind_velocity`, `wind_rate`, presets | 3, 4, 5, 6 |
| `crates/elements-ember/src/kernels/project.rs` | solver dispatch | 3 |
| `crates/elements-ember/examples/solver_gate.rs`, `justfile` | the gate | 5 |
| `docs/bench/solver-gate.md` | gate record | 5 |
| `crates/elements-ember/src/kernels/conserve.rs`, `shaders/boundary_flux.wgsl`, `shaders/mass_scale.wgsl` | mass correction | 6 |
| `crates/elements-ember/src/kernels/shaders/wind.wgsl`, `kernels/mod.rs` | wind relaxation | 4 |
| `crates/elements-ember/src/bench/mod.rs`, `src/metrics.rs` | `plume_wind` scene, open-face outflow | 4 |
| `tests/bench/mantaflow_scene.py`, `tests/bench/mapping.py`, `docs/bench/mantaflow-notes.md` | Mantaflow wind twin | 7 |
| `docs/bench/results.md`, specs, `CLAUDE.md` | rerun and records | 8 |

---

### Task 1: Multigrid V-cycle

**Files:**
- Create: `crates/elements-ember/src/kernels/multigrid.rs` (and `pub mod multigrid;` plus re-exports in `kernels/mod.rs`)
- Create: `crates/elements-ember/src/kernels/shaders/residual.wgsl`, `restrict.wgsl`, `restrict_mask.wgsl`, `prolong_add.wgsl`
- Modify: `crates/elements-ember/src/kernels/project.rs` (a mask-only relax entry point)
- Test: `crates/elements-ember/tests/multigrid.rs`

**Interfaces:**
- Consumes: `Uniforms::new(&GpuContext, &StepConstants)`, `StepConstants`
  (`cells`, `h`, `dx`, `open_mask`, `has_solids`), the red-black kernel in
  `pressure.wgsl`, `ReduceTarget`/`reduce` and `remove_mean` for closed
  domains.
- Produces:

```rust
/// One multigrid level: its uniforms (dims, dx = 2^l·dx0, same h and
/// open_mask), its right-hand side and correction fields, and its solid mask
/// (None on levels with no solid cell, or when the domain has no collider).
pub struct Level { /* private */ }

/// The level hierarchy for a domain, built once per substep from the fine
/// uniforms' constants and the fine solid mask. Every field comes from the
/// pool; `release` returns them.
pub struct Hierarchy { /* private */ }

impl Hierarchy {
    pub fn new(
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        batch: &mut ComputeBatch,
        pool: &mut FieldPool,
        fine: &StepConstants,
        fine_mask: Option<&Field>,
    ) -> Result<Self, GpuError>;
    /// Number of levels, the fine level included.
    pub fn depth(&self) -> usize;
    pub fn release(self, pool: &mut FieldPool);
}

/// `cycles` V-cycles on `p` for ∇²p = div/h, starting from what `p` holds.
pub fn v_cycles(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    h: &Hierarchy,
    p: &Field,
    div: &Field,
    cycles: u32,
) -> Result<(), GpuError>;

/// r = div − h·L(p) on the level's grid, in the same units as `div`, so the
/// existing smoother solves a level's correction with the level's own
/// `div`-shaped right-hand side. Exposed for tests and for MGPCG.
pub fn residual(/* gpu, cache, batch, u: &Uniforms, p, div, out, mask: Option<&Field> */) -> Result<(), GpuError>;
```

**Units.** The smoother solves `(Σ neighbours − count·p) = dx²·div / h`
(`pressure.wgsl`). The residual kernel returns, in `div` units,
`r = div − (h/dx²)·(Σ − count·p)`, with the same neighbour rules as the
smoother (inside and fluid: its value; beyond an open face: 0 with count +1;
beyond a wall or a solid neighbour: left out). A coarse level with spacing
`2dx` then solves `L_c e = r_c / h` by running the unchanged smoother with
`div := r_c` and `dx2 := (2dx)²`. That is why each level is just a
`Uniforms` built from `StepConstants { cells: level dims, dx: 2^l·dx, .. }`.

**Kernels (WGSL, each prefixed by `common.wgsl` and, where it reads
solids, `solid.wgsl`).**

`residual.wgsl`:

```wgsl
@group(0) @binding(0) var p: texture_3d<f32>;
@group(0) @binding(1) var div: texture_3d<f32>;
@group(0) @binding(2) var out: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var solid: texture_3d<f32>;

fn nb(q: vec3<i32>, sum: ptr<function, f32>, count: ptr<function, f32>) {
    if (!cell_solid(q)) {
        *sum += textureLoad(p, q, 0).x;
        *count += 1.0;
    }
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) { return; }
    let c = vec3<i32>(gid);
    if (cell_solid(c)) {
        textureStore(out, c, vec4<f32>(0.0));
        return;
    }
    let n = vec3<i32>(params.dims);
    var sum = 0.0;
    var count = 0.0;
    // Same neighbour rules as pressure.wgsl, written out per side for the
    // same naga-Metal reason (a loop over axes ran ~40% slower there).
    if (c.x > 0) { nb(c - vec3<i32>(1, 0, 0), &sum, &count); } else if (is_open(0u, 0u)) { count += 1.0; }
    if (c.x < n.x - 1) { nb(c + vec3<i32>(1, 0, 0), &sum, &count); } else if (is_open(0u, 1u)) { count += 1.0; }
    if (c.y > 0) { nb(c - vec3<i32>(0, 1, 0), &sum, &count); } else if (is_open(1u, 0u)) { count += 1.0; }
    if (c.y < n.y - 1) { nb(c + vec3<i32>(0, 1, 0), &sum, &count); } else if (is_open(1u, 1u)) { count += 1.0; }
    if (c.z > 0) { nb(c - vec3<i32>(0, 0, 1), &sum, &count); } else if (is_open(2u, 0u)) { count += 1.0; }
    if (c.z < n.z - 1) { nb(c + vec3<i32>(0, 0, 1), &sum, &count); } else if (is_open(2u, 1u)) { count += 1.0; }
    let lap = (sum - count * textureLoad(p, c, 0).x) / params.dx2;
    let r = textureLoad(div, c, 0).x - params.pressure_scale * lap;
    textureStore(out, c, vec4<f32>(r, 0.0, 0.0, 0.0));
}
```

`restrict.wgsl` (fine → coarse, run on the coarse grid; binds the fine
field and fine mask as `texture_3d`, the coarse output as storage, and the
**coarse** uniforms plus a small `FineDims` uniform with the fine dims):
each coarse cell `C` averages the fluid children among
`2C + {0,1}³` that lie inside the fine grid and are not solid in the fine
mask. With no fluid child it writes 0.

`restrict_mask.wgsl`: coarse cell solid (1.0) only when every child inside
the fine grid is solid; otherwise 0.0.

`prolong_add.wgsl` (run on the fine grid, binding the coarse correction as
`texture_3d` and the fine `p` as `read_write` storage): for each fluid fine
cell `c`, sample the coarse correction trilinearly at coarse position
`(c + 0.5)/2 − 0.5`, clamping sample indices to the coarse grid and skipping
solid coarse samples by renormalising the weights over the fluid ones (all
solid → add nothing), then `p[c] += e`. Solid fine cells are left at 0.

**Level construction (`Hierarchy::new`).** Level 0 is the fine grid with
`fine_mask`. Level l+1 dims are `ceil(dims_l / 2)` per axis; stop when the
smallest dim is ≤ 8 or a halving would not shrink any axis. Each coarse
level's mask comes from `restrict_mask`; a level whose mask has no solid cell
may still keep it (simplest), with `has_solids = true` whenever the fine
level has solids. Each level gets two pool fields: `rhs` and `e`.

**V-cycle (recursive, levels 0..depth):**

```
vcycle(l, x, b):            // x: level-l solution/correction, b: rhs in div units
  if l == depth−1: relax(l, x, b, 32 sweeps); return
  relax(l, x, b, 2 sweeps, red-then-black)
  residual(l, x, b → tmp_l)
  restrict(tmp_l → rhs_{l+1});  zero e_{l+1}
  if closed domain: remove_mean(rhs_{l+1})
  vcycle(l+1, e_{l+1}, rhs_{l+1})
  prolong_add(e_{l+1} → x)
  relax(l, x, b, 2 sweeps, black-then-red)
```

Level 0's `x` is `p` and `b` is `div`. `tmp_l` is one more pool field per
level (the residual), so each level holds `rhs`, `e`, `tmp` and its mask.
`relax` is the existing red-black pass, given a mask-only entry point in
`project.rs`:

```rust
/// `sweeps` red-black sweeps on `p` with `div` as the right-hand side,
/// reading solids from `mask` alone (the multigrid levels have no collider
/// velocity). `reverse` runs black before red in each sweep.
pub(crate) fn relax(gpu, cache, batch, u: &Uniforms, p: &Field, div: &Field,
    sweeps: u32, mask: Option<&Field>, reverse: bool) -> Result<(), GpuError>;
```

`pressure(...)` keeps its signature and calls `relax` with
`solids.map(|s| s.mask)` and `reverse = false`, so its tests and callers are
unchanged. In a closed domain, `v_cycles` removes `p`'s mean after the last
cycle, as `solve_pressure` does.

- [ ] **Step 1: Write the failing tests** in `crates/elements-ember/tests/multigrid.rs`:

```rust
mod common;

use common::*;
use elements_core::gpu::{ComputeBatch, FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_ember::boundaries::DEFAULT_OPEN_MASK;
use elements_ember::kernels::multigrid::{Hierarchy, residual, v_cycles};
use elements_ember::kernels::{StepConstants, Uniforms, pressure};

const H: f32 = 0.1;

/// RMS of the residual r = div − h·L(p), from the GPU residual kernel.
fn residual_rms(gpu: &elements_core::gpu::GpuContext, c: &StepConstants,
                p: &elements_core::gpu::Field, div: &elements_core::gpu::Field,
                mask: Option<&elements_core::gpu::Field>) -> f64 { /* residual() into a
    pool field, read back, RMS over fluid cells */ }

/// A right-hand side with zero mean, so closed-domain problems are solvable.
fn rhs(cells: FieldDims) -> Vec<f32> { let mut v = pattern(cells, 3); let m = v.iter().sum::<f32>() / v.len() as f32; v.iter_mut().for_each(|x| *x -= m); v }

#[test]
fn the_residual_kernel_is_zero_for_an_exact_solution() {
    // p = x² on a 16³ grid with walls everywhere has a known discrete
    // Laplacian away from the walls; build div = h·L(p) on the CPU with the
    // same stencil (see cpu_red_black in tests/common for the rules) and
    // check the GPU residual is < 1e-5 everywhere.
}

#[test]
fn a_v_cycle_cuts_the_residual_tenfold_in_every_boundary_setting() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    for (name, mask, sphere) in [
        ("closed", 0u32, false),
        ("open top", DEFAULT_OPEN_MASK, false),
        ("open top, sphere", DEFAULT_OPEN_MASK, true),
    ] {
        let cells = FieldDims::new(64, 64, 64);
        let mut pool = FieldPool::new();
        let c = StepConstants { open_mask: mask, has_solids: sphere, ..StepConstants::new(cells, H, 1.0 / 32.0) };
        // sphere: a solid mask of radius 10 cells at the centre, uploaded.
        // div = rhs(cells) (zeroed in solid cells); p = 0.
        // r0 = residual_rms(p = 0); run 1 V-cycle; r1 = residual_rms.
        // assert!(r1 < r0 / 10.0, "{name}: {r0} -> {r1}");
        // Record the measured factor in the ledger.
    }
}

#[test]
fn v_cycles_beat_the_same_cost_of_gauss_seidel() {
    // 128³ open top: 4 V-cycles against 40 red-black iterations from p = 0
    // (a generous equal-cost budget). The V-cycles' residual must be at least
    // 100× smaller.
}

#[test]
fn a_hierarchy_stops_at_eight_cells_and_returns_every_field() {
    // 128³ → 5 levels, 256³ → 6, 40×24×16 → levels until the smallest dim ≤ 8.
    // After release, pool.pooled_count() equals the number acquired.
}

#[test]
fn a_coarse_cell_is_solid_only_when_all_its_children_are() {
    // A 4×4×4 mask with one fully solid 2×2×2 block and one half-solid
    // block: the coarse 2×2×2 mask has exactly one solid cell.
}
```

Write the elided bodies in full, using the existing helpers in
`tests/common/mod.rs` (`gpu`, `upload`, `pattern`, `cpu_red_black`,
`index`). Keep grids at 64³ except where stated.

- [ ] **Step 2: Run them; they fail to compile.**
Run: `cargo nextest run -p elements-ember --test multigrid`

- [ ] **Step 3: Implement** the kernels, `relax`, `residual`, `Hierarchy` and
`v_cycles` as above. Build every bind group through `kernels::bind_group`
with `Bind::Tex`/`Bind::Buf`/`Bind::View`; bind the placeholder
(`u.placeholder()`) where a level has no mask.

- [ ] **Step 4: Run the tests; they pass.** Also run
`cargo nextest run -p elements-ember` — `pressure`'s callers are unchanged.

- [ ] **Step 5: Prove each test can fail**, one mutation each, output recorded:
1. residual: drop `count * p` from the Laplacian.
2. tenfold: prolongate with weight 0.5 instead of adding the full correction.
3. equal-cost: skip the post-smoothing sweeps.
4. hierarchy: stop at 16 instead of 8.
5. coarse mask: solid when *any* child is solid.

- [ ] **Step 6: `just check`, commit.** Subject: "Add a multigrid V-cycle
for the pressure solve". Body: why (Gauss–Seidel's fixed count converges
less with resolution; spec §1).

---

### Task 2: MGPCG

**Files:**
- Create: `crates/elements-ember/src/kernels/mgpcg.rs`, `shaders/pcg.wgsl`
- Test: `crates/elements-ember/tests/multigrid.rs`

**Interfaces:**
- Consumes: `Hierarchy`, `residual`, `v_cycles` (with a zero-start variant)
  from Task 1; `reduce`, `ReduceTarget`.
- Produces:

```rust
/// `iterations` of conjugate gradients on L(p) = div/h, preconditioned by one
/// V-cycle from zero, warm-started from `p` (it solves for the correction to
/// `p`). Scalars stay on the GPU.
pub fn mgpcg(gpu, cache, batch, pool: &mut FieldPool, h: &Hierarchy,
             p: &Field, div: &Field, iterations: u32) -> Result<(), GpuError>;
```

**Algorithm, in `div` units** (the operator `A x = h·L(x)` restricted to
fluid cells, so `r = div − A p` is the Task 1 residual):

```
r = residual(p, div);  z = V(r);  d = z;  rz = <r,z>
repeat iterations:
  q = A d                      // residual kernel with div := 0, then negate: q = −residual(d, 0)
  α = rz / <d,q>
  p += α d;  r −= α q
  z = V(r);  rz' = <r,z>;  β = rz'/rz;  rz = rz'
  d = z + β d
```

`<a,b>` is a `pcg.wgsl` `multiply` pass into a scratch field followed by
`reduce(Sum)` into a slot of one `ReduceTarget` with 4 slots
(`rz`, `dq`, `rz_new`, spare). The update kernels (`axpy_p_r`,
`update_d`) read the slots and compute α and β in every thread. Guard
divisions: if `dq` or `rz` is ≤ 1e-30 the update adds nothing (converged).
`V(r)` zeroes `e_0` and runs one V-cycle with `r` as the right-hand side —
add `pub fn v_cycle_from_zero(.., h, e: &Field, rhs: &Field)` to Task 1's
module. In a closed domain, remove the mean of `r` and of `z` each time
they are formed, and of `p` at the end.

- [ ] **Step 1: Failing tests** (append to `tests/multigrid.rs`):
  - `mgpcg_converges_faster_than_v_cycles_around_a_sphere`: 64³, open top, a
    solid sphere, `p = 0`; residual after 6 MGPCG iterations is at least
    10× smaller than after 6 V-cycles.
  - `mgpcg_reaches_a_tight_residual_in_a_closed_box`: 64³ closed; after 10
    iterations the residual RMS is below 1e-5 × the initial.
  - `mgpcg_is_bit_identical_across_runs`: two runs from the same inputs give
    identical bits.
- [ ] **Step 2: Run; fail to compile.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run; pass.**
- [ ] **Step 5: Mutations:** (1) `β = 0` (becomes steepest descent);
  (2) drop the preconditioner (`z = r`); (3) sum the dot products into the
  wrong slot for `rz_new`. Record outputs.
- [ ] **Step 6: `just check`, commit.** "Wrap the V-cycle in preconditioned conjugate gradients".

---

### Task 3: Choosing the pressure solver

**Files:**
- Modify: `crates/elements-ember/src/solver.rs` (params, validation, docs, `project`)
- Modify: `crates/elements-ember/src/kernels/project.rs` (`solve_pressure` dispatch)
- Test: `crates/elements-ember/tests/solver.rs`

**Interfaces:**
- Produces:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PressureSolver { #[default] GaussSeidel, Multigrid, Mgpcg }

// SolverParams gains:
pub pressure_solver: PressureSolver,
/// V-cycles (Multigrid) or PCG iterations (Mgpcg); 1..=64.
pub pressure_cycles: u32,
```

Presets keep `GaussSeidel` until Task 4's decision (so nothing changes
yet); `pressure_cycles` defaults to 4. `DocParams` accepts both as optional.
`project` builds the `Hierarchy` once per substep when the solver is not
Gauss–Seidel, passes `solids.map(|s| s.mask)`, and releases it through
`self.retired` (the batch may still reference its fields).

- [ ] Tests (fail first, then pass, then mutate):
  - `rejects_out_of_range_solver_parameters` gains `pressure_cycles: 0` and
    `65`.
  - `frame_40_is_bit_identical_with_multigrid` and `…_with_mgpcg`: the
    `ANIMATED` document with `"pressure_solver": "multigrid"` (and
    `"mgpcg"`), through `assert_doc_frame_40_is_bit_identical`.
  - `multigrid_leaves_less_divergence_than_gauss_seidel`: `plume_16`-style
    document at 32³, frame 20, masked divergence RMS (metrics::measure) with
    `multigrid` × 4 is below `gauss_seidel` × 40.
  - `a_failed_step_with_multigrid_returns_every_field_to_the_pool`: as the
    existing pool test, with multigrid on.
- Mutations: ignore `pressure_solver` (always Gauss–Seidel); release the
  hierarchy directly instead of through `retired` (the pool test or a
  validation error should catch it — record which).
- `just check`, commit: "Let documents choose the pressure solver".

---

### Task 4: Wind as ambient airflow, and the wind scene

**Files:**
- Modify: `crates/elements-ember/src/solver.rs` (`wind_velocity`, `wind_rate`, validation, the old-`wind` error)
- Modify: `crates/elements-ember/src/kernels/mod.rs` (`KernelParams`: `face_accel` → `face_wind`, `_pad` → `wind_blend`), `shaders/common.wgsl`, `shaders/wind.wgsl`
- Modify: `crates/elements-ember/src/bench/mod.rs` (`plume_wind`, `mantaflow_json`)
- Modify: `crates/elements-ember/src/metrics.rs` (open-face outflow, `mass_inside`), `crates/elements-ember/examples/benchmark.rs`, `src/bench/report.rs` (field rename)
- Test: `crates/elements-ember/tests/forces.rs`, `tests/solver.rs`, `tests/metrics.rs`, `tests/bench.rs`

**Parameters.** `SolverParams.wind: [f32; 3]` is removed; add
`wind_velocity: [f32; 3]` (m/s, default 0) and `wind_rate: f32` (1/s,
default 0, ≥ 0, finite). `DocParams` must reject the old key with a message
naming the new ones: keep a `wind: Option<serde_json::Value>` field in
`DocParams` and, when present, return `params::bad(KIND, "\"wind\" was
replaced by \"wind_velocity\" (m/s) and \"wind_rate\" (1/s)")`. The stage
runs when `wind_rate > 0`.

**Kernel.** `Params.face_accel` becomes `face_wind` (this axis's component
of `wind_velocity`) and the spare `_pad2` becomes `wind_blend` =
`1 − exp(−wind_rate · h)`, computed on the CPU. `wind.wgsl`:
`u += (params.face_wind − u) * params.wind_blend` on non-wall, non-solid
faces. Keep `KernelParams` at 64 bytes.

**Scene.** `plume_wind`: `wind_velocity = [1.0, 0.0, 0.0]`, `wind_rate =
1.0`, `boundaries` with `neg_x` and `pos_x` `Open` (top open, rest walls).
`mantaflow_json` writes `wind_velocity`, `wind_rate` and the boundaries
(already generic) instead of `wind`.

**Metrics: outflow through every open face.** `Sample` gains
`open_mask: u32` (the `Boundaries::open_mask` bits). For each open face, the
measuring plane is the faces two cells in from it (for the top: z index
nz − 2, as now; for −x: x-face index 2; for +x: x-face index nx − 2, and
likewise for y and −z), with net upwind flux counted positive outward. The
control volume excludes the two outer layers at each open face;
`FrameMetrics.mass_below` is renamed `mass_inside` and sums that volume.
Callers pass `scene.solver.boundaries.open_mask()` (Ember and Mantaflow
modes alike). Default open mask (top only) reproduces today's values
exactly — the existing metrics tests must pass unchanged apart from the
rename.

- [ ] Tests (fail first; pass; mutate):
  - `forces.rs`: `wind_relaxes_towards_its_velocity_without_overshoot` —
    faces start at 0 and at 3 m/s with w = 1, rate 1, for h in
    {0.01, 0.5, 5.0}: after one pass each face equals
    `u + (w − u)(1 − e^{−h})` to 1e-6 and lies between u and w.
  - `solver.rs`: `the_old_wind_parameter_is_rejected_with_its_replacements`.
  - `solver.rs`: `rejects_out_of_range_solver_parameters` gains a negative
    and a NaN `wind_rate`.
  - `metrics.rs`: `outflow_counts_every_open_face` — a density layer next to
    +x with u = +0.4 there and a layer next to the top with w = +0.4:
    `outflow_rate` is the sum of both; `mass_inside` excludes the outer two
    layers at both faces.
  - `bench.rs`: `plume_wind_blows_through_open_sides` (the scene's
    boundaries and wind), and a GPU run at 32³ for 60 frames asserting no
    NaN, max |u| below 3 m/s, and mass_inside + cumulative outflow at frame
    60 within 10% of total emitted (a regression guard for the old failure).
- Mutations: omit the `(w − u)` (add w·blend); count only the top face in
  outflow; drop the old-key check. Record.
- `just check`, commit: "Make wind an ambient airflow the air relaxes towards".

---

### Task 5: The solver gate

**Files:**
- Create: `crates/elements-ember/examples/solver_gate.rs`
- Modify: `justfile` (`bench-solver`)
- Create: `docs/bench/solver-gate.md` (written by the run)

**The rule** (spec §4, copied into the report verbatim): a configuration
passes when, at 128³ and 256³ and in `plume`, `plume_collider` and
`plume_wind`, its median frame time over frames 25–48 is at most the
Gauss–Seidel ×160 median in the same process, and its masked divergence RMS
at frames 60 and 120 is at most Mantaflow's for that scene and resolution.
Mantaflow's references, from `docs/bench/results.md` (7fe5d9d run), as
constants in the example:

| scene | 128³ (f60 / f120) | 256³ (f60 / f120) |
|---|---|---|
| plume | 1.49e-4 / 1.07e-4 | 7.41e-5 / 4.51e-5 |
| plume_collider | 2.84e-4 / 1.09e-4 | 9.12e-5 / 5.25e-5 |
| plume_wind (stand-in: plume) | 1.49e-4 / 1.07e-4 | 7.41e-5 / 4.51e-5 |

`plume_wind` runs with the new ambient wind from Task 4.

**Thin-collider amendment (user, 2026-09-24; spec §4).**
- Add `Scene::plume_plate(res)` in `bench/mod.rs`: `plume` plus a collider
  union of two static boxes, one cell thick (`half_extents` z = dx/2), that
  together span the domain at z = 0.8 m, leaving a slit 2 cells wide in x
  centred above the emitter. Test that its `solid_mask` has exactly the
  slit's cells open in that layer. It is a gate scene only, not a benchmark
  scene.
- For `plume_plate` at 128³, measure the ratio of RMS divergence after
  projection to before it, at frames 60 and 120, through the kernel API as
  `speed_gate.rs`'s `divergence_at` does (reach the frame's state, run
  stages 1–3, measure, project with the candidate solver, measure). Pass
  when ≤ 1e-3.
- A configuration passes only if it passes the three scenes' rule **and**
  `plume_plate`. Sweep counts until both hold; `mgpcg` up to 24.
- Past-floor stability: in `plume_collider` at 128³, the masked divergence
  after 40 iterations of the chosen method must be at most 4× that after the
  chosen count. Report it; a failure stops the gate like any other.
- Plain V-cycles (`multigrid`) are measured and reported but cannot be the
  default.

**Procedure:**
1. Load check: record `sysctl -n vm.loadavg` before and after; flag above 2.
2. Sweep at 128³: for `multigrid` with cycles 1..=8 and `mgpcg` with
   iterations 1..=16, in increasing order, run each scene's metrics to frame
   120 (readback only at 60 and 120) until the divergence rule holds in all
   three scenes; that count is the method's candidate. A method with no
   passing count is recorded as failing.
3. At 256³, check each candidate's divergence, stepping the count up until
   it holds (or the range ends).
4. Time each candidate and Gauss–Seidel ×160: 3 runs of frames 1–48 each,
   median of frames 25–48, per scene and resolution, interleaved (GS,
   multigrid, mgpcg, GS, …) so drifting load hits all three.
5. Write `solver-gate.md`: machine, OS, commit, date, load, the rule, the
   sweep table, the timing table, the rule applied ("fastest passing:
   …" or "none passes"), and `Decision (recorded by the user): _pending_`.

- [ ] Implement the example (reuse `common::{median, shell, commit_label}`),
  add `bench-solver:` to the justfile ("Takes about an hour, real GPU, not in
  `check`"), `cargo build --release --examples`, `just check`, commit
  "Add the solver gate".
- [ ] Run `just bench-solver` (background; it takes about an hour). Commit
  `docs/bench/solver-gate.md` as "Record the solver gate".
- [ ] **STOP. Bring the table to the user.** Record their decision in the
  file's decision line.
- [ ] After the decision: set both presets' `pressure_solver` and
  `pressure_cycles` to the chosen configuration (final preset: the same
  method, with the count the sweep found at 256³ doubled, unless the user
  says otherwise), update the preset docs in `solver.rs` to cite
  `solver-gate.md`, rerun `just check`, commit "Move the presets to the
  gate's pressure solver". If none passed, do not change presets; stop.

---

### Task 6: Mass-conserving correction

**Files:**
- Create: `crates/elements-ember/src/kernels/conserve.rs`, `shaders/boundary_flux.wgsl`, `shaders/mass_scale.wgsl`
- Modify: `crates/elements-ember/src/solver.rs` (`conserve_mass`, `advect_scalars`)
- Test: `crates/elements-ember/tests/conserve.rs`, `tests/solver.rs`

**Interfaces:**
- Produces: `SolverParams::conserve_mass: bool` (both presets `true`;
  documents may set it), and

```rust
/// Before advecting `field`: reduce its sum into slot `M0` of `target`, and
/// the outflow over open faces during this substep (upwind value × outward
/// normal velocity × h / dx, i.e. the cell-units flux; see below) into
/// slot `OUT`.
pub fn measure_before(gpu, cache, batch, u: &Uniforms, field: &Field,
                      velocity: &StaggeredField, scratch: &Field,
                      target: &ReduceTarget) -> Result<(), GpuError>;
/// After advecting into `advected`: reduce its sum into slot `M1`, then
/// scale it by s = (M0 − OUT) / M1, clamped to [0.9, 1.1], leaving it
/// unchanged when M1 ≤ 1e-12 or M0 − OUT ≤ 0.
pub fn correct_after(gpu, cache, batch, u: &Uniforms, advected: &Field,
                     target: &ReduceTarget) -> Result<(), GpuError>;
```

**Units.** All sums are of raw cell values (Σq); dV cancels in the ratio.
The outflow in the same units is Σ over open boundary faces of
`q_inside · max(u_n, 0) · h / dx` (one face of area dx² over a cell volume
dx³). `boundary_flux.wgsl` runs over cells; a cell adjacent to an open face
adds `q · max(u_n, 0) · h · inv_dx` for each such face (the boundary face's
velocity from the face grid, `u_n` positive outward), writing the per-cell
total into `scratch`, which is then summed. Wall and collider faces carry
nothing.

**Wiring.** In `Substep::advect_scalars`, for density and then
temperature: `measure_before` with the projected velocity the advection
uses, advect as today (including dissipation), then `correct_after` on the
result. Use one `ReduceTarget` with 3 slots per scalar, created per substep
(it is small) and kept alive in `self.retired`-style storage until the batch
runs. When `conserve_mass` is false, skip both calls, so today's output is
bit-identical.

- [ ] Tests (fail first; pass; mutate):
  - `a_closed_box_keeps_its_mass_to_rounding`: 32³ closed, an emitter active
    frames 1–10, then frames 11–60; total density at 60 equals total at 11
    within 1e-5 relative with `conserve_mass`, and differs by more than
    1e-3 relative without it (record the measured drift).
  - `open_top_mass_plus_outflow_is_constant`: 32³ default boundaries, same
    emission; from frame 11, Σmass + cumulative outflow (computed on the CPU
    from read-back fields with the same first-order rule) stays within 1e-3
    relative of frame 11's mass.
  - `the_correction_never_scales_beyond_ten_percent`: a unit test of the
    WGSL's clamp through a tiny field whose advected sum is doubled.
  - `conserve_mass_off_adds_no_work`: with `"conserve_mass": false`, one
    substep's `advect_scalars` makes exactly as many pool acquisitions
    (`FieldPool::acquisitions`) as before this task, and the frame-40 bits of
    a document with the flag off match a run of the same document with the
    correction code path compiled out by the flag (the two share one test
    via a helper that runs both and compares).
  - Frame-40 determinism with `conserve_mass` on (the presets now enable it,
    so the existing determinism tests cover it; confirm they run it).
- Mutations: drop the outflow term; clamp to [0.5, 2]; apply the scale to
  the pre-advection field. Record.
- `just check`, commit: "Conserve scalar mass across each advection".

---

### Task 7: Mantaflow's wind twin

**Files:**
- Modify: `tests/bench/mantaflow_scene.py`, `tests/bench/mapping.py`, `tests/bench/test_mapping.py`, `docs/bench/mantaflow-notes.md`

- [ ] **Find the mapping by bake.** Blender's wind field has `flow`: a drag
  towards the field's own velocity. Read `effect.cc` (`do_physical_effector`,
  wind with `flow`) to learn the form, then bake: a 32³ domain filled with
  smoke at rest, all sides open, one wind field along +x, and measure the
  domain-centre velocity per frame (converted with the notes' units rule).
  Find strength and `flow` so the smoke reaches about 1 m/s with a time
  constant of about 1 s. Record the formula, source and experiment in the
  notes.
- [ ] **`mapping.wind`** becomes `wind(wind_velocity, wind_rate, fps) ->
  (strength, flow, direction)`; `test_mapping.py` covers it (one known pair
  from the bake) and is proven to fail by one mutation.
- [ ] **Scene script:** read `wind_velocity` / `wind_rate`; set strength,
  `flow`, falloff and rotation from the mapping; the boundaries already come
  from the JSON (±x open now).
- [ ] **Check:** bake `plume_wind` at 64³ and compare the smoke's +x drift at
  frames 30 and 60 with Ember's (`benchmark ember plume_wind 64`). Record both
  in the notes; the notes and `results.md` keep saying Mantaflow's wind acts
  only on smoky cells.
- [ ] `ruff format --check tests/bench && ruff check tests`, `just check`,
  commit: "Match Mantaflow's wind to Ember's ambient airflow".

---

### Task 8: Rerun and records

- [ ] **Idle check and preset rerun (risk (k)).** Load under 2 for the
  rerun, as in 2b-3 Task 8; if the machine never idles, ask the user whether
  to run under load (they said yes last time) and flag it.
  `just bench-presets`; if a preview frame no longer fits 100 ms, **stop and
  report**.
- [ ] **Full `just bench`.** Move 2b-3's `docs/bench/results/` files aside
  first (they are from another commit and `report` refuses mixed commits);
  keep 2b-3's `results.md` Summary text for comparison. Run in the
  background.
- [ ] **Write the Summary** against the tables, 200–350 words, comparing
  with 2b-3's run: speed, memory, divergence, conservation, wind. Say where
  Mantaflow still wins, if anywhere. Check every figure against the run
  files.
- [ ] **Records:** close risk (l) and update (j) in
  `docs/superpowers/specs/2026-09-21-ember-solver-design.md`; add Lentine et
  al. (2011) there as future work; set the 2b-3c spec's status to Complete;
  update CLAUDE.md's status paragraph (2b-3c complete, 2b-3b next) and add
  the 2b-3c spec, plan and `solver-gate.md` to its list.
- [ ] `just check`, commit: "Rerun the Mantaflow benchmark on the improved solver".
