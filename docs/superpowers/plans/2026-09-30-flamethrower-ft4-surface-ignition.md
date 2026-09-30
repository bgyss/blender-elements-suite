# Flamethrower FT4: Surface Ignition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A collider can carry a wood budget that ignites from hot gas, feeds fuel into the fluid while it burns, depletes, and outputs a `char` mask.

**Architecture:** Colliders gain an optional `surface_fuel { load }` param and a third `Field` output (load where the SDF is negative). `smoke_solver` gains input 7 (that load) and output 4 (`char`), keeps a `burned` grid in `SolverState`, and each substep runs two kernels (`surface_burn` on band cells, `surface_gather` on fluid cells) whose result feeds the existing `emit_fuel` and `burn`. With input 7 unconnected nothing is allocated or dispatched.

**Tech Stack:** Rust, wgpu 30 (WGSL compute), `elements-ember` and `elements-core`, cargo-nextest, `just check`.

**Spec:** `docs/superpowers/specs/2026-09-30-flamethrower-ft4-surface-ignition-design.md` (read it first). Roadmap: `docs/superpowers/specs/2026-09-30-flamethrower-napalm-roadmap-design.md` §4.

## Global Constraints

- `#![forbid(unsafe_code)]` in `elements-ember`; `required_features` stays `wgpu::Features::empty()`.
- Scalar fields are `R32Float`. At most 4 **storage** textures per shader stage; read other fields through `texture_3d<f32>` + `textureLoad`.
- Kernels with solids include `solid.wgsl` and declare `var solid: texture_3d<f32>`.
- `crates/*` are `Apache-2.0 OR MIT`: no GPL code.
- Every stochastic node takes an explicit seed (none added here).
- Surface off (input 7 unconnected) is bit-identical to today, fire on or off.
- Verify wgpu APIs against the vendored crate source, not docs.rs (AGENTS.md "Verifying wgpu APIs").
- **Never set `WGPU_BACKEND` locally** (Metal on this Mac).
- **Mutation rule (AGENTS.md):** each test must be shown to fail. Mutate the covered code, watch the test fail, restore, record the real output in the commit body. A mutation changes exactly one thing.
- Commits: plain imperative subject, body explains why; end with the attribution trailers the harness specifies. `just check` must pass before each commit. Implementation commits use plain `git`, not `jj`. Do not push.
- Hashes and bit-exact assertions are valid only within one backend on one machine (this Mac's Apple M1 Max); gate on the adapter name like `tests/solver.rs` does.

## Review Focus

- A collider with `surface_fuel` but a `load` of 0 or a thin plank that is not solid anywhere must simply never burn (no NaN from `burned / load`): Task 4 tests `char` with load 0.
- `surface_fuel` given to a collider whose transform has more than one key (moving) must be rejected at build, not silently misburn: Task 1.
- Input 7 connected without fuel (input 6) or without a collider (inputs 4 and 5) must be an error, not a silent no-op: Task 5.
- A collider union where only one side has a load must give that side's load, not zero: Task 2.
- A timeline restore or scrub must give the same `char` as a straight run, since `burned` is new state: Task 5.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/collider.rs` | `SurfaceFuel` param, `surface_fuel` on `ColliderParams`, `fill_surface_load`, third output |
| `src/mesh_collider.rs` | same param and third output, reusing `fill_surface_load` |
| `src/unions.rs` | collider union: optional load inputs 4 and 5, output 2 = per-cell max |
| `src/kernels/shaders/surface_load.wgsl`, `collider_union_load.wgsl` | load and union-max kernels |
| `src/kernels/surface.rs` (new) | `surface_burn`, `surface_gather`, `surface_char` wrappers |
| `src/kernels/shaders/surface.wgsl`, `surface_burn.wgsl`, `surface_gather.wgsl`, `surface_char.wgsl` (new) | WGSL |
| `src/kernels/mod.rs`, `shaders/common.wgsl` | `StepConstants.surface_burn_rate`, uniform field `surface_burn` |
| `src/solver.rs` | `surface_burn_rate` param, `SolverState.surface`, `Sources.surface`, substep wiring, sockets, `char` output |
| `tests/surface.rs` (new) | all FT4 tests |
| `tests/collider.rs`, `tests/mesh_collider.rs`, `src/bench/mod.rs`, other literals | add `surface_fuel: None` / new param |

All paths are under `crates/elements-ember/` unless they begin with `docs/` or `AGENTS.md`.

---

### Task 1: Collider load output

**Files:**
- Modify: `src/collider.rs`, `src/mesh_collider.rs`
- Create: `src/kernels/shaders/surface_load.wgsl`
- Modify (mechanical): every `ColliderParams { .. }` / `MeshColliderParams { .. }` literal (find with `grep -rn "ColliderParams {" crates --include='*.rs'`: `tests/collider.rs`, `tests/mesh_collider.rs`, `tests/jet_shack.rs`, `tests/fire_scene.rs`, and any in `src/`)
- Test: `tests/surface.rs` (create)

**Interfaces:**
- Produces: `pub struct SurfaceFuel { pub load: f32 }` (serde, `deny_unknown_fields`); `ColliderParams.surface_fuel: Option<SurfaceFuel>` and `MeshColliderParams.surface_fuel: Option<SurfaceFuel>`, both `#[serde(default)]`; `pub fn fill_surface_load(gpu: &GpuContext, cache: &mut PipelineCache, sdf: &Field, load: f32, out: &Field) -> Result<(), GpuError>` in `collider.rs`; collider and mesh collider outputs become `[Field (sdf), VectorField (velocity), Field (load)]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/surface.rs`:

```rust
//! FT4 surface ignition (spec 2026-09-30-flamethrower-ft4-surface-ignition-design.md).

mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldPool, PipelineCache};
use elements_ember::collider::{ColliderFields, ColliderParams, SurfaceFuel, fill_collider};
use elements_ember::transform::{Key, Shape, Transform};

const DX: f32 = 0.125;

fn still_box(surface_fuel: Option<SurfaceFuel>) -> ColliderParams {
    ColliderParams {
        shape: Shape::Box {
            half_extents: [0.25, 0.25, 0.125],
        },
        transform: Transform {
            keys: vec![Key {
                frame: 0.0,
                translate: [0.5, 0.5, 0.5],
                rotate: None,
            }],
        },
        surface_fuel,
    }
}

#[test]
fn a_collider_outputs_its_load_where_the_sdf_is_negative() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims { x: 8, y: 8, z: 8 };
    let sdf = pool
        .acquire(&gpu, cells, elements_core::gpu::FieldFormat::R32Float)
        .unwrap();
    let load = pool
        .acquire(&gpu, cells, elements_core::gpu::FieldFormat::R32Float)
        .unwrap();
    let velocity = pool.acquire_staggered_zeroed(&gpu, &mut cache, cells).unwrap();
    let params = still_box(Some(SurfaceFuel { load: 7.5 }));
    let pose = params.transform.pose(0.0, 1.0 / 24.0);
    fill_collider(
        &gpu,
        &mut cache,
        &params,
        &pose,
        DX,
        ColliderFields {
            sdf: &sdf,
            velocity: &velocity,
        },
    )
    .unwrap();
    elements_ember::collider::fill_surface_load(&gpu, &mut cache, &sdf, 7.5, &load).unwrap();
    let sdf_data = sdf.read_back(&gpu).unwrap();
    let load_data = load.read_back(&gpu).unwrap();
    let inside = sdf_data.iter().filter(|&&d| d < 0.0).count();
    assert!(inside > 0, "the box must cover some cells");
    for (d, l) in sdf_data.iter().zip(&load_data) {
        let want = if *d < 0.0 { 7.5 } else { 0.0 };
        assert_eq!(*l, want, "sdf {d}");
    }
}

#[test]
fn surface_fuel_rejects_a_moving_collider_and_a_negative_load() {
    let moving = serde_json::json!({
        "shape": { "sphere": { "radius": 0.1 } },
        "transform": { "keys": [
            { "frame": 0, "translate": [0.0, 0.0, 0.0] },
            { "frame": 10, "translate": [1.0, 0.0, 0.0] } ] },
        "surface_fuel": { "load": 3.0 }
    });
    assert!(elements_ember::registry().build("ember.collider", &moving).is_err());
    let negative = serde_json::json!({
        "shape": { "sphere": { "radius": 0.1 } },
        "transform": { "keys": [{ "frame": 0, "translate": [0.0, 0.0, 0.0] }] },
        "surface_fuel": { "load": -1.0 }
    });
    assert!(elements_ember::registry().build("ember.collider", &negative).is_err());
    let still = serde_json::json!({
        "shape": { "sphere": { "radius": 0.1 } },
        "transform": { "keys": [{ "frame": 0, "translate": [0.0, 0.0, 0.0] }] },
        "surface_fuel": { "load": 3.0 }
    });
    assert!(elements_ember::registry().build("ember.collider", &still).is_ok());
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p elements-ember --test surface`
Expected: compile error, `SurfaceFuel` and `fill_surface_load` not found.

- [ ] **Step 3: Implement**

`src/kernels/shaders/surface_load.wgsl`:

```wgsl
// Surface fuel load (FT4 spec §3.1): `load` where the collider's signed
// distance at the cell centre is below zero, 0 elsewhere.

struct Load {
    dims: vec3<u32>,
    load: f32,
};

@group(0) @binding(0) var sdf: texture_3d<f32>;
@group(0) @binding(1) var out_load: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Load;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let l = select(0.0, params.load, textureLoad(sdf, p, 0).x < 0.0);
    textureStore(out_load, p, vec4<f32>(l, 0.0, 0.0, 0.0));
}
```

In `src/collider.rs` add (beside `ColliderGpu`):

```rust
const LOAD_WGSL: &str = include_str!("kernels/shaders/surface_load.wgsl");

/// A collider's wood budget (FT4 spec §3.1): fuel-grid units per solid cell.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceFuel {
    pub load: f32,
}

impl SurfaceFuel {
    pub(crate) fn validate(&self, kind: &str, transform: &Transform) -> Result<(), DocError> {
        params::finite(kind, "surface_fuel.load", &[self.load])?;
        if self.load < 0.0 {
            return Err(params::bad(
                kind,
                format!("surface_fuel.load must not be negative, got {}", self.load),
            ));
        }
        if transform.keys.len() > 1 {
            return Err(params::bad(
                kind,
                "surface_fuel needs a static collider: its reservoir is tied to grid cells, \
                 so the transform may have at most one key",
            ));
        }
        Ok(())
    }
}

/// Matches `Load` in surface_load.wgsl, 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LoadGpu {
    dims: [u32; 3],
    load: f32,
}

const _: () = assert!(std::mem::size_of::<LoadGpu>() == 16);

/// `out` = `load` where `sdf` < 0, else 0. Submits its own batch.
pub fn fill_surface_load(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    sdf: &Field,
    load: f32,
    out: &Field,
) -> Result<(), GpuError> {
    let cells = sdf.dims();
    if out.dims() != cells {
        return Err(GpuError::Validation(format!(
            "fill_surface_load: out {:?}, sdf {cells:?}",
            out.dims()
        )));
    }
    let pipe = cache.get_or_create(gpu, "ember.collider.load", LOAD_WGSL, "main")?;
    let params = uniform_buffer(
        gpu,
        "ember-load",
        bytemuck::bytes_of(&LoadGpu {
            dims: [cells.x, cells.y, cells.z],
            load,
        }),
    )?;
    let group = bind_group(gpu, &pipe, &[Bind::Tex(sdf), Bind::Tex(out), Bind::Buf(&params)])?;
    let mut batch = ComputeBatch::new();
    batch.dispatch(&pipe, &group, cells);
    batch.submit(gpu)
}
```

Change `ColliderParams` to add `#[serde(default)] pub surface_fuel: Option<SurfaceFuel>,`. In `build`, after the transform check add `if let Some(s) = &p.surface_fuel { s.validate(KIND, &p.transform)?; }`. Change `Collider::sockets` outputs to `vec![SocketType::Field, SocketType::VectorField, SocketType::Field]` and `eval` to:

```rust
let load = params.surface_fuel.map_or(0.0, |s| s.load);
let mut values = produce(ctx, 2, |gpu, cache, cells, velocity| {
    fill_collider(gpu, cache, params, &pose, dx, ColliderFields { sdf: &cells[0], velocity })?;
    fill_surface_load(gpu, cache, &cells[0], load, &cells[1])
})?;
// `produce` returns the cell fields then the vector: [sdf, load, velocity].
values.swap(1, 2);
Ok(values)
```

Do the same in `src/mesh_collider.rs` (`use crate::collider::{SurfaceFuel, fill_surface_load}`, add the field, validation with `p.surface_fuel`, sockets, eval). Then add `surface_fuel: None,` to every struct literal found by the grep above.

- [ ] **Step 4: Run tests to verify they pass, and the old suite still does**

Run: `cargo nextest run -p elements-ember`
Expected: all pass, including `surface`'s two tests.

- [ ] **Step 5: Mutation proofs**

1. In `surface_load.wgsl` change `< 0.0` to `> 0.0`. Expected: `a_collider_outputs_its_load_where_the_sdf_is_negative` fails (assert_eq). Restore.
2. In `SurfaceFuel::validate` change `transform.keys.len() > 1` to `> 2`. Expected: `surface_fuel_rejects_a_moving_collider_and_a_negative_load` fails at the first assert. Restore.

Record both failure messages.

- [ ] **Step 6: Commit**

```bash
just check
git add -A
git commit -m "Give colliders a surface fuel load output

The solver's wood reservoir needs a per-cell load, and only fields travel
on sockets. A static collider with surface_fuel outputs its load where the
SDF is negative; moving ones are rejected because the reservoir is tied to
grid cells.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016jxQnAfsTRPo8xRwrHJx4y"
```

---

### Task 2: Collider union load

**Files:**
- Modify: `src/unions.rs`
- Create: `src/kernels/shaders/collider_union_load.wgsl`
- Test: `tests/surface.rs`

**Interfaces:**
- Consumes: Task 1's third collider output.
- Produces: `ember.collider_union` inputs become `[Field, VectorField, Field, VectorField, Field (load a), Field (load b)]`, inputs 4 and 5 optional (an unconnected one counts as all zero); outputs `[Field, VectorField, Field (max of the loads)]`. `pub fn union_surface_load(gpu, cache, a: &Field, b: &Field, out: &Field) -> Result<(), GpuError>`.

- [ ] **Step 1: Write the failing test** (append to `tests/surface.rs`)

```rust
#[test]
fn a_union_keeps_the_larger_load_and_a_missing_side_counts_as_zero() {
    use elements_ember::unions::union_surface_load;
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims { x: 4, y: 4, z: 4 };
    let n = cells.voxel_count();
    let a: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i * 3) % 4) as f32).collect();
    let fa = upload(&gpu, &mut pool, cells, &a);
    let fb = upload(&gpu, &mut pool, cells, &b);
    let out = upload(&gpu, &mut pool, cells, &vec![-1.0; n]);
    union_surface_load(&gpu, &mut cache, &fa, &fb, &out).unwrap();
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x.max(*y)).collect();
    assert_eq!(out.read_back(&gpu).unwrap(), want);
}

#[test]
fn a_union_with_one_unconnected_load_outputs_the_other() {
    let doc = r#"{ "version": 3, "dims": [8, 8, 8], "fps": 24.0, "domain_size": 1.0,
      "nodes": [
        { "id": 0, "kind": "ember.collider", "params": {
            "shape": { "box": { "half_extents": [0.2, 0.2, 0.1] } },
            "transform": { "keys": [{ "frame": 0, "translate": [0.5, 0.5, 0.5] }] },
            "surface_fuel": { "load": 4.0 } } },
        { "id": 1, "kind": "ember.collider", "params": {
            "shape": { "sphere": { "radius": 0.05 } },
            "transform": { "keys": [{ "frame": 0, "translate": [0.1, 0.1, 0.1] }] } } },
        { "id": 2, "kind": "ember.collider_union", "params": {} },
        { "id": 3, "kind": "core.output", "params": {} } ],
      "edges": [
        { "from_node": 0, "from_index": 0, "to_node": 2, "to_index": 0 },
        { "from_node": 0, "from_index": 1, "to_node": 2, "to_index": 1 },
        { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 2 },
        { "from_node": 1, "from_index": 1, "to_node": 2, "to_index": 3 },
        { "from_node": 0, "from_index": 2, "to_node": 2, "to_index": 4 },
        { "from_node": 2, "from_index": 2, "to_node": 3, "to_index": 0 } ],
      "output": 3 }"#;
    let mut s = Session::new(doc);
    let mut t = timeline(0);
    let bits = s.density_bits(&mut t, 1);
    let loads: Vec<f32> = bits.iter().map(|b| f32::from_bits(*b)).collect();
    assert!(loads.iter().any(|&l| l == 4.0), "the first collider's load survives");
    assert!(loads.iter().all(|&l| l == 0.0 || l == 4.0), "{loads:?}");
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p elements-ember --test surface -- union`
Expected: compile error, `union_surface_load` not found.

- [ ] **Step 3: Implement**

`src/kernels/shaders/collider_union_load.wgsl`:

```wgsl
// ember.collider_union, load pass: the larger wood load (FT4 spec §3.1).

struct Grid {
    dims: vec3<u32>,
    axis: u32,
};

@group(0) @binding(0) var l1: texture_3d<f32>;
@group(0) @binding(1) var l2: texture_3d<f32>;
@group(0) @binding(2) var out_load: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<uniform> grid: Grid;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= grid.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    textureStore(out_load, p, vec4<f32>(max(textureLoad(l1, p, 0).x, textureLoad(l2, p, 0).x), 0.0, 0.0, 0.0));
}
```

In `src/unions.rs` add `const UNION_LOAD_WGSL: &str = include_str!("kernels/shaders/collider_union_load.wgsl");` and

```rust
/// `out` = the per-cell max of the wood loads `a` and `b`. Submits its own batch.
pub fn union_surface_load(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    a: &Field,
    b: &Field,
    out: &Field,
) -> Result<(), GpuError> {
    let cells = out.dims();
    if a.dims() != cells || b.dims() != cells {
        return Err(GpuError::Validation(
            "union_surface_load: inputs differ in size".to_owned(),
        ));
    }
    let pipe = cache.get_or_create(gpu, "ember.collider_union.load", UNION_LOAD_WGSL, "main")?;
    let grid = uniform_buffer(
        gpu,
        "ember-union",
        bytemuck::bytes_of(&GridGpu {
            dims: [cells.x, cells.y, cells.z],
            axis: 0,
        }),
    )?;
    let group = bind_group(
        gpu,
        &pipe,
        &[Bind::Tex(a), Bind::Tex(b), Bind::Tex(out), Bind::Buf(&grid)],
    )?;
    let mut batch = ComputeBatch::new();
    batch.dispatch(&pipe, &group, cells);
    batch.submit(gpu)
}
```

`ColliderUnion::sockets` becomes inputs `[Field, VectorField, Field, VectorField, Field, Field]`, outputs `[Field, VectorField, Field]`. Rewrite `eval` and `collider_union_node`:

```rust
fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
    let mut wanted = vec![0, 1, 2, 3];
    wanted.extend([4u32, 5].into_iter().filter(|i| ctx.input_connected(*i)));
    let inputs = take_listed(ctx, &wanted)?;
    let result = collider_union_node(ctx, &inputs);
    for (_, value) in inputs {
        ctx.release(value);
    }
    result
}

fn collider_union_node(
    ctx: &mut EvalCtx<'_>,
    inputs: &[(u32, Value)],
) -> Result<Vec<Value>, NodeError> {
    let at = |i: u32| inputs.iter().find(|(n, _)| *n == i).map(|(_, v)| v);
    let a = ColliderFields {
        sdf: at(0).expect("taken").as_field()?,
        velocity: at(1).expect("taken").as_vector_field()?,
    };
    let b = ColliderFields {
        sdf: at(2).expect("taken").as_field()?,
        velocity: at(3).expect("taken").as_vector_field()?,
    };
    let load_a = at(4).map(Value::as_field).transpose()?;
    let load_b = at(5).map(Value::as_field).transpose()?;
    let mut values = produce(ctx, 2, |gpu, cache, cells, velocity| {
        union_colliders(
            gpu,
            cache,
            a,
            b,
            ColliderFields {
                sdf: &cells[0],
                velocity,
            },
        )?;
        // A missing side is all zero: max(x, 0) for a non-negative load is x.
        let zero;
        let (la, lb) = match (load_a, load_b) {
            (Some(la), Some(lb)) => (la, lb),
            (Some(l), None) | (None, Some(l)) => {
                zero = zeroed_like(gpu, cache, l)?;
                (l, &zero)
            }
            (None, None) => {
                return fill_zero(gpu, cache, &cells[1]);
            }
        };
        union_surface_load(gpu, cache, la, lb, &cells[1])
    })?;
    values.swap(1, 2);
    Ok(values)
}
```

with two small private helpers in `unions.rs`: `zeroed_like(gpu, cache, like: &Field) -> Result<Field, GpuError>` (`Field::new`/pool-free zeroed field; check `elements_core::gpu` for a constructor such as `Field::zeroed`; if none exists use a temporary `FieldPool` and `acquire_zeroed`) and `fill_zero(gpu, cache, out: &Field)` (`out.write(gpu, &vec![0.0; n])`). `Value::as_field` returns `Result<&Field, NodeError>`; adapt `transpose` if its error type differs. Keep `take_inputs` imported only if still used elsewhere in the file.

- [ ] **Step 4: Run tests**

Run: `cargo nextest run -p elements-ember`
Expected: all pass (existing union tests use four connected inputs and stay valid).

- [ ] **Step 5: Mutation proof**

In `collider_union_load.wgsl` change `max(` to `min(`. Expected: `a_union_keeps_the_larger_load...` fails (assert_eq on vectors). Restore. Record the message.

- [ ] **Step 6: Commit**

```bash
just check
git add -A
git commit -m "Union colliders' surface loads

A union of a planked shack and a plain collider must keep the shack's
wood, so load takes the per-cell maximum and an unconnected side counts as
zero.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016jxQnAfsTRPo8xRwrHJx4y"
```

---

### Task 3: Characterization hash, parameter and uniform plumbing

**Files:**
- Modify: `src/solver.rs`, `src/kernels/mod.rs`, `src/kernels/shaders/common.wgsl`, `src/bench/mod.rs`
- Test: `tests/surface.rs`, `tests/solver.rs` (only if a serialized-params expectation changes)

**Interfaces:**
- Produces: `SolverParams.surface_burn_rate: f32` (doc key `surface_burn_rate`, default `DEFAULT_SURFACE_BURN_RATE = 2.0` in every preset, validated finite and ≥ 0); `StepConstants.surface_burn_rate: f32` (0 in `new`); uniform field `surface_burn` (= `surface_burn_rate * h`) in `Params` (WGSL) and `KernelParams`, taking the place of `_pad1` / half of `_pad`, size stays 96.

- [ ] **Step 1: Record the fire-on guard hash BEFORE changing anything solver-side**

Append to `tests/surface.rs`:

```rust
/// FT4 spec §3.5: with the surface unconnected, a fire document is bit-identical
/// to the one before FT4. Recorded on the commit before any FT4 solver change,
/// on this adapter; other adapters print and skip.
#[test]
fn without_a_surface_a_fire_document_matches_the_solver_before_ft4() {
    const RECORDED_ON: &str = "Apple M1 Max";
    const WANT: u64 = 0; // replaced in Step 2 from the first run
    let ctx = gpu();
    if ctx.adapter_name() != RECORDED_ON {
        eprintln!("skipped: recorded on {RECORDED_ON}, this is {}", ctx.adapter_name());
        return;
    }
    let mut s = Session::new(&fire_doc());
    let mut t = timeline(0);
    let mut bits = Vec::new();
    for frame in 1..=40 {
        bits = s.density_bits(&mut t, frame);
    }
    assert_eq!(fnv1a(&bits), WANT, "frame 40 density changed");
}

fn fnv1a(bits: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bits {
        for byte in b.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// A 16³ plume fed fuel by a second emitter, density (socket 0) as the output.
fn fire_doc() -> String {
    r#"{ "version": 3, "dims": [16, 16, 16], "fps": 24.0, "domain_size": 2.0,
      "nodes": [
        { "id": 0, "kind": "ember.sphere_emitter", "params": { "center": [1.0, 1.0, 0.4],
          "radius": 0.3, "density_rate": 1.0, "temperature_rate": 2.0 } },
        { "id": 1, "kind": "ember.smoke_solver", "params": { "buoyancy_temperature": 1.0 } },
        { "id": 2, "kind": "core.output", "params": {} },
        { "id": 3, "kind": "ember.sphere_emitter", "params": { "center": [1.0, 1.0, 0.4],
          "radius": 0.3, "density_rate": 1.0, "temperature_rate": 0.0 } } ],
      "edges": [
        { "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 },
        { "from_node": 0, "from_index": 1, "to_node": 1, "to_index": 1 },
        { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 0 },
        { "from_node": 3, "from_index": 0, "to_node": 1, "to_index": 6 } ],
      "output": 2 }"#
        .to_owned()
}
```

- [ ] **Step 2: Run, record the hash**

Run: `cargo nextest run -p elements-ember --test surface -- fire_document`
Expected: FAIL, `left: <hash> right: 0`. Put the printed left value (as `0x....`) in `WANT`, rerun, expect PASS. This hash was measured on the code with no FT4 solver changes (Tasks 1 and 2 only touch colliders).

- [ ] **Step 3: Implement the plumbing**

`common.wgsl`: replace `_pad1: u32,` with `surface_burn: f32,         // surface_burn_rate·h: wood burnt per substep (FT4)`. In `kernels/mod.rs` add `pub surface_burn_rate: f32,` to `StepConstants` (doc: "Wood burnt per second per cell where a surface burns (FT4 spec §3.2); 0 otherwise."), `surface_burn_rate: 0.0,` in `new`, and in `KernelParams` replace `_pad: [u32; 2]` by `surface_burn: f32, _pad: u32` with `surface_burn: c.surface_burn_rate * c.h, _pad: 0` where the params are built (the size assert stays 96).

In `solver.rs`: `pub const DEFAULT_SURFACE_BURN_RATE: f32 = 2.0;` beside the fire defaults; add `pub surface_burn_rate: f32` to `SolverParams` (doc: "Wood burnt per second per surface cell while it burns (FT4 spec §3.2); acts only while the surface input is connected."), `surface_burn_rate: DEFAULT_SURFACE_BURN_RATE` in `Quality::params`' literal, `surface_burn_rate: Option<f32>` in `DocParams`, `surface_burn_rate: doc.surface_burn_rate.unwrap_or(preset.surface_burn_rate)` in `resolve_params`, a finite and `>= 0` check in `validate` (copy the `burning_rate` check's style), and `surface_burn_rate: self.surface_burn_rate` in `step_constants`. Add the field to the `SolverParams` literal in `src/bench/mod.rs:66`.

- [ ] **Step 4: Add a failing-then-passing validation test** (append to `tests/surface.rs`)

```rust
#[test]
fn surface_burn_rate_must_be_finite_and_non_negative() {
    use elements_ember::solver::resolve_params;
    assert_eq!(
        resolve_params(&serde_json::json!({})).unwrap().surface_burn_rate,
        2.0
    );
    assert!(resolve_params(&serde_json::json!({ "surface_burn_rate": 0.5 })).is_ok());
    assert!(resolve_params(&serde_json::json!({ "surface_burn_rate": -1.0 })).is_err());
    assert!(resolve_params(&serde_json::json!({ "surface_burn_rate": f32::NAN })).is_err());
}
```

Run `cargo nextest run -p elements-ember`. Expected: all pass, including the guard hash (the uniform bytes beyond the old padding changed, but no kernel reads them yet). If a bench or solver test compares serialized `SolverParams`, update its expectation and say so in the commit body.

- [ ] **Step 5: Mutation proof**

In `validate`, delete the finiteness half of the new `surface_burn_rate` check (keep `>= 0`). Expected: the `f32::NAN` assert in `surface_burn_rate_must_be_finite_and_non_negative` fails. Restore. Record the failure message.

- [ ] **Step 6: Commit**

```bash
just check
git add -A
git commit -m "Add the surface burn rate and pin fire's output

Records frame 40 of a fire document before the surface kernels exist, so
every later FT4 change is checked against bit-identical output with the
surface unconnected. The rate reaches kernels as an uniform field in the
old padding.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016jxQnAfsTRPo8xRwrHJx4y"
```

---

### Task 4: Surface kernels

**Files:**
- Create: `src/kernels/surface.rs`, `src/kernels/shaders/surface.wgsl`, `surface_burn.wgsl`, `surface_gather.wgsl`, `surface_char.wgsl`
- Modify: `src/kernels/mod.rs` (`mod surface; pub use surface::{surface_burn, surface_char, surface_gather};`)
- Test: `tests/surface.rs`

**Interfaces:**
- Consumes: `Solids<'_>`, `Uniforms`, `StepConstants { has_solids: true, surface_burn_rate, ignition_temperature, .. }` (Task 3).
- Produces:
  - `surface_burn(gpu, cache, batch, u: &Uniforms, load: &Field, temperature: &Field, solids: Solids<'_>, burned: &Field, emitted: &Field) -> Result<(), GpuError>`
  - `surface_gather(gpu, cache, batch, u, solids: Solids<'_>, emitted: &Field, rate: &Field) -> Result<(), GpuError>` (`rate` is the fuel rate per second: `Σ e_n / fluid_neighbours(n) / h` over solid neighbours `n`, 0 on solid cells)
  - `surface_char(gpu, cache, batch, u, burned: &Field, load: &Field, dst: &Field) -> Result<(), GpuError>` (`dst = clamp(burned/load, 0, 1)` where `load > 0`, else 0)

- [ ] **Step 1: Write the failing tests** (append to `tests/surface.rs`)

```rust
use elements_core::gpu::{ComputeBatch, Field, FieldFormat, GpuContext};
use elements_ember::kernels::{
    StepConstants, Uniforms, Solids, surface_burn, surface_char, surface_gather,
};

const K: FieldDims = FieldDims { x: 8, y: 8, z: 8 };
const H: f32 = 1.0 / 48.0;
const IGN: f32 = 1.5;
const BURN_RATE: f32 = 2.0;

/// A vertical plank: solid cells i = 4, j in 2..6, k in 1..7, each with `load`.
fn plank(load: f32) -> (Vec<f32>, Vec<f32>) {
    let mut mask = vec![0.0; K.voxel_count()];
    let mut loads = vec![0.0; K.voxel_count()];
    for k in 1..7 {
        for j in 2..6 {
            mask[index(K, 4, j, k)] = 1.0;
            loads[index(K, 4, j, k)] = load;
        }
    }
    (mask, loads)
}

fn constants() -> StepConstants {
    StepConstants {
        has_solids: true,
        surface_burn_rate: BURN_RATE,
        ignition_temperature: IGN,
        ..StepConstants::new(K, H, 0.125)
    }
}

fn in_domain(c: [i32; 3]) -> bool {
    c.iter().all(|&v| (0..8).contains(&v))
}

fn neighbours(c: [i32; 3]) -> [[i32; 3]; 6] {
    let mut out = [c; 6];
    for n in 0..6 {
        out[n][n / 2] += if n % 2 == 1 { 1 } else { -1 };
    }
    out
}

fn fluid(mask: &[f32], c: [i32; 3]) -> bool {
    in_domain(c) && mask[index(K, c[0] as u32, c[1] as u32, c[2] as u32)] <= 0.5
}

fn at(c: [i32; 3]) -> usize {
    index(K, c[0] as u32, c[1] as u32, c[2] as u32)
}

/// FT4 spec §3.3, directly: returns (burned', emitted, rate).
fn cpu_surface(
    mask: &[f32],
    load: &[f32],
    burned: &[f32],
    temperature: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = K.voxel_count();
    let (mut b2, mut emitted, mut rate) = (burned.to_vec(), vec![0.0; n], vec![0.0; n]);
    let burn = BURN_RATE * H;
    for k in 0..8 {
        for j in 0..8 {
            for i in 0..8 {
                let c = [i, j, k];
                let fluid_n: Vec<[i32; 3]> = neighbours(c)
                    .into_iter()
                    .filter(|&q| fluid(mask, q))
                    .collect();
                if mask[at(c)] > 0.5 && load[at(c)] > 0.0 && !fluid_n.is_empty() {
                    let hottest = fluid_n
                        .iter()
                        .map(|&q| temperature[at(q)])
                        .fold(f32::MIN, f32::max);
                    let b = burned[at(c)];
                    if (b > 0.0 || hottest > IGN) && b < load[at(c)] {
                        let e = burn.min(load[at(c)] - b);
                        b2[at(c)] = b + e;
                        emitted[at(c)] = e;
                    }
                }
            }
        }
    }
    for k in 0..8 {
        for j in 0..8 {
            for i in 0..8 {
                let c = [i, j, k];
                if fluid(mask, c) {
                    let mut sum = 0.0;
                    for q in neighbours(c) {
                        if in_domain(q) && mask[at(q)] > 0.5 && emitted[at(q)] > 0.0 {
                            let nf = neighbours(q).into_iter().filter(|&r| fluid(mask, r)).count();
                            sum += emitted[at(q)] / nf as f32;
                        }
                    }
                    rate[at(c)] = sum / H;
                }
            }
        }
    }
    (b2, emitted, rate)
}

struct Kernels {
    gpu: GpuContext,
    pool: FieldPool,
    cache: PipelineCache,
    mask: Field,
    velocity: elements_core::gpu::StaggeredField,
    load: Field,
    burned: Field,
    emitted: Field,
    rate: Field,
}

impl Kernels {
    fn new(mask: &[f32], load: &[f32], burned: &[f32]) -> Self {
        let gpu = gpu();
        let mut pool = FieldPool::new();
        let mut cache = PipelineCache::new();
        let velocity = pool.acquire_staggered_zeroed(&gpu, &mut cache, K).unwrap();
        let mask = upload(&gpu, &mut pool, K, mask);
        let load = upload(&gpu, &mut pool, K, load);
        let burned = upload(&gpu, &mut pool, K, burned);
        let emitted = upload(&gpu, &mut pool, K, &vec![9.0; K.voxel_count()]);
        let rate = upload(&gpu, &mut pool, K, &vec![9.0; K.voxel_count()]);
        Self { gpu, pool, cache, mask, velocity, load, burned, emitted, rate }
    }

    /// One substep of the two kernels; returns (burned', emitted, rate).
    fn run(&mut self, temperature: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let temp = upload(&self.gpu, &mut self.pool, K, temperature);
        let u = Uniforms::new(&self.gpu, &constants()).unwrap();
        let solids = Solids { mask: &self.mask, velocity: &self.velocity };
        let mut batch = ComputeBatch::new();
        surface_burn(&self.gpu, &mut self.cache, &mut batch, &u, &self.load, &temp, solids,
            &self.burned, &self.emitted).unwrap();
        surface_gather(&self.gpu, &mut self.cache, &mut batch, &u, solids, &self.emitted,
            &self.rate).unwrap();
        batch.submit(&self.gpu).unwrap();
        (
            self.burned.read_back(&self.gpu).unwrap(),
            self.emitted.read_back(&self.gpu).unwrap(),
            self.rate.read_back(&self.gpu).unwrap(),
        )
    }
}

fn gas(temperature_at: &[([i32; 3], f32)]) -> Vec<f32> {
    let mut t = vec![0.0; K.voxel_count()];
    for (c, v) in temperature_at {
        t[at(*c)] = *v;
    }
    t
}

#[test]
fn cold_gas_ignites_nothing() {
    let (mask, load) = plank(5.0);
    let mut k = Kernels::new(&mask, &load, &vec![0.0; K.voxel_count()]);
    // Even gas exactly at the threshold does not ignite: strictly above.
    let t = vec![IGN; K.voxel_count()];
    for _ in 0..10 {
        let (burned, emitted, rate) = k.run(&t);
        assert!(burned.iter().all(|&b| b == 0.0));
        assert!(emitted.iter().all(|&e| e == 0.0));
        assert!(rate.iter().all(|&r| r == 0.0));
    }
}

#[test]
fn gas_above_the_threshold_ignites_only_its_neighbour() {
    let (mask, load) = plank(5.0);
    let mut k = Kernels::new(&mask, &load, &vec![0.0; K.voxel_count()]);
    // One hot fluid cell beside plank cell (4, 3, 3).
    let t = gas(&[([3, 3, 3], IGN + 0.01)]);
    let (burned, emitted, rate) = k.run(&t);
    let burn = BURN_RATE * H;
    assert_eq!(burned[at([4, 3, 3])], burn);
    assert_eq!(burned.iter().filter(|&&b| b > 0.0).count(), 1);
    assert_eq!(emitted[at([4, 3, 3])], burn);
    // Its fluid neighbours: (3,3,3), (5,3,3), (4,2,3)? no: j = 2 is plank.
    // The kernel and the CPU reference must agree on the split.
    let (cb, ce, cr) = cpu_surface(&mask, &load, &vec![0.0; K.voxel_count()], &t);
    assert_close(&burned, &cb, 1e-7, "burned");
    assert_close(&emitted, &ce, 1e-7, "emitted");
    assert_close(&rate, &cr, 1e-4, "rate");
}

#[test]
fn the_kernels_match_the_cpu_reference_over_many_substeps() {
    let (mask, load) = plank(0.15); // depletes within a few substeps
    let mut k = Kernels::new(&mask, &load, &vec![0.0; K.voxel_count()]);
    let mut burned = vec![0.0; K.voxel_count()];
    let hot = gas(&[([3, 3, 3], 2.0), ([5, 4, 5], 2.0), ([3, 5, 1], 1.6)]);
    for step in 0..12 {
        let (want_b, want_e, want_r) = cpu_surface(&mask, &load, &burned, &hot);
        let (got_b, got_e, got_r) = k.run(&hot);
        assert_close(&got_b, &want_b, 1e-7, &format!("burned step {step}"));
        assert_close(&got_e, &want_e, 1e-7, &format!("emitted step {step}"));
        assert_close(&got_r, &want_r, 1e-4, &format!("rate step {step}"));
        burned = want_b;
    }
}

#[test]
fn the_reservoir_depletes_and_emission_stops() {
    let (mask, load) = plank(0.1);
    let mut k = Kernels::new(&mask, &load, &vec![0.0; K.voxel_count()]);
    let hot = gas(&[([3, 3, 3], 2.0)]);
    let mut last = 0.0f32;
    for _ in 0..30 {
        let (burned, emitted, _) = k.run(&hot);
        assert!(burned[at([4, 3, 3])] <= 0.1 + 1e-7, "never past the load");
        last = emitted[at([4, 3, 3])];
    }
    let (burned, _, _) = k.run(&hot);
    assert_eq!(burned[at([4, 3, 3])], 0.1, "fully burned");
    assert_eq!(last, 0.0, "a spent cell emits nothing");
}

#[test]
fn what_the_wood_loses_the_gas_gains() {
    let (mask, load) = plank(3.0);
    let mut k = Kernels::new(&mask, &load, &vec![0.0; K.voxel_count()]);
    let hot = gas(&[([3, 3, 3], 2.0), ([5, 4, 5], 2.0)]);
    let (burned, _, rate) = k.run(&hot);
    let lost: f64 = burned.iter().map(|&b| f64::from(b)).sum();
    let gained: f64 = rate.iter().map(|&r| f64::from(r) * f64::from(H)).sum();
    assert!(lost > 0.0);
    assert!((lost - gained).abs() <= 1e-6 * lost, "lost {lost}, gained {gained}");
}

#[test]
fn char_is_the_burned_fraction_and_zero_without_a_load() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let n = K.voxel_count();
    let mut burned = vec![0.0; n];
    let mut load = vec![0.0; n];
    burned[0] = 1.0;
    load[0] = 4.0; // quarter burned
    burned[1] = 4.0;
    load[1] = 4.0; // spent
    burned[2] = 0.5; // burned but no load: must be 0, not NaN or inf
    let b = upload(&gpu, &mut pool, K, &burned);
    let l = upload(&gpu, &mut pool, K, &load);
    let dst = pool.acquire(&gpu, K, FieldFormat::R32Float).unwrap();
    let u = Uniforms::new(&gpu, &constants()).unwrap();
    let mut batch = ComputeBatch::new();
    surface_char(&gpu, &mut cache, &mut batch, &u, &b, &l, &dst).unwrap();
    batch.submit(&gpu).unwrap();
    let out = dst.read_back(&gpu).unwrap();
    assert_eq!(&out[..3], &[0.25, 1.0, 0.0]);
    assert!(out.iter().all(|v| v.is_finite()));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p elements-ember --test surface`
Expected: compile error, `surface_burn` etc. not found.

- [ ] **Step 3: Implement the WGSL**

`src/kernels/shaders/surface.wgsl` (helpers, concatenated after `common.wgsl` and `solid.wgsl`):

```wgsl
// Surface ignition helpers (FT4 spec §3.3). The kernel declares `params`
// and `solid`. A "fluid" cell is inside the domain and not solid.

fn neighbour(p: vec3<i32>, n: u32) -> vec3<i32> {
    var d = vec3<i32>(0);
    d[n / 2u] = select(-1, 1, (n & 1u) == 1u);
    return p + d;
}

fn in_domain(c: vec3<i32>) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(params.dims));
}

fn is_fluid(c: vec3<i32>) -> bool {
    return in_domain(c) && !cell_solid(c);
}

fn fluid_count(p: vec3<i32>) -> u32 {
    var count = 0u;
    for (var n = 0u; n < 6u; n = n + 1u) {
        if (is_fluid(neighbour(p, n))) {
            count = count + 1u;
        }
    }
    return count;
}
```

`surface_burn.wgsl`:

```wgsl
// Surface burn (FT4 spec §3.3, kernel 1), per band cell: ignite from hot
// neighbouring gas, burn `params.surface_burn` of the load, record what was
// emitted. `emitted` is written in every cell, 0 where nothing burned.

@group(0) @binding(0) var load: texture_3d<f32>;
@group(0) @binding(1) var temperature: texture_3d<f32>;
@group(0) @binding(2) var burned: texture_storage_3d<r32float, read_write>;
@group(0) @binding(3) var emitted: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var solid: texture_3d<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    var e = 0.0;
    let wood = textureLoad(load, p, 0).x;
    if (cell_solid(p) && wood > 0.0 && fluid_count(p) > 0u) {
        let b = textureLoad(burned, p).x;
        var hottest = -3.0e38;
        for (var n = 0u; n < 6u; n = n + 1u) {
            let q = neighbour(p, n);
            if (is_fluid(q)) {
                hottest = max(hottest, textureLoad(temperature, q, 0).x);
            }
        }
        if ((b > 0.0 || hottest > params.ignition_temperature) && b < wood) {
            e = min(params.surface_burn, wood - b);
            textureStore(burned, p, vec4<f32>(b + e, 0.0, 0.0, 0.0));
        }
    }
    textureStore(emitted, p, vec4<f32>(e, 0.0, 0.0, 0.0));
}
```

`surface_gather.wgsl`:

```wgsl
// Surface gather (FT4 spec §3.3, kernel 2), per fluid cell: each solid
// neighbour gives its emission, split over its own fluid neighbours. The
// sum over h is a fuel rate per second, which `emit_fuel` consumes. Gathering
// on the fluid side needs no atomics and is deterministic.

@group(0) @binding(0) var emitted: texture_3d<f32>;
@group(0) @binding(1) var rate: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var solid: texture_3d<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    var sum = 0.0;
    if (is_fluid(p)) {
        for (var n = 0u; n < 6u; n = n + 1u) {
            let q = neighbour(p, n);
            if (in_domain(q) && cell_solid(q)) {
                let e = textureLoad(emitted, q, 0).x;
                if (e > 0.0) {
                    sum = sum + e / f32(fluid_count(q));
                }
            }
        }
    }
    textureStore(rate, p, vec4<f32>(sum / params.h, 0.0, 0.0, 0.0));
}
```

`surface_char.wgsl`:

```wgsl
// The char output (FT4 spec §3.3): the burned fraction of a cell's load,
// 0 where there is no load.

@group(0) @binding(0) var burned: texture_3d<f32>;
@group(0) @binding(1) var load: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let l = textureLoad(load, p, 0).x;
    let c = select(0.0, clamp(textureLoad(burned, p, 0).x / max(l, 1e-30), 0.0, 1.0), l > 0.0);
    textureStore(dst, p, vec4<f32>(c, 0.0, 0.0, 0.0));
}
```

`src/kernels/surface.rs`:

```rust
//! Surface ignition (FT4 spec §3.3): the burn on band cells, the gather into
//! the fluid, and the char output.

use elements_core::gpu::{ComputeBatch, Field, GpuContext, GpuError, PipelineCache};

use super::{Bind, Solids, Uniforms, bind_group, expect_dims, solid_views};

const BURN: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/surface.wgsl"),
    include_str!("shaders/surface_burn.wgsl"),
);
const GATHER: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/solid.wgsl"),
    include_str!("shaders/surface.wgsl"),
    include_str!("shaders/surface_gather.wgsl"),
);
const CHAR: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/surface_char.wgsl"),
);

/// Ignite and burn the band cells of `load` (spec §3.3 kernel 1). `burned`
/// grows in place; `emitted` gets this substep's emission per cell.
#[allow(clippy::too_many_arguments)]
pub fn surface_burn(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    load: &Field,
    temperature: &Field,
    solids: Solids<'_>,
    burned: &Field,
    emitted: &Field,
) -> Result<(), GpuError> {
    for (what, field) in [
        ("surface_burn load", load),
        ("surface_burn temperature", temperature),
        ("surface_burn burned", burned),
        ("surface_burn emitted", emitted),
    ] {
        expect_dims(what, field, u.cells())?;
    }
    let (solid, _) = solid_views(u, Some(solids), None)?;
    let pipeline = cache.get_or_create(gpu, "ember.surface_burn", BURN, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(load),
            Bind::Tex(temperature),
            Bind::Tex(burned),
            Bind::Tex(emitted),
            Bind::Buf(u.any()),
            Bind::View(solid),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// Gather `emitted` into the fluid as a fuel rate per second (spec §3.3 kernel 2).
pub fn surface_gather(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    solids: Solids<'_>,
    emitted: &Field,
    rate: &Field,
) -> Result<(), GpuError> {
    expect_dims("surface_gather emitted", emitted, u.cells())?;
    expect_dims("surface_gather rate", rate, u.cells())?;
    let (solid, _) = solid_views(u, Some(solids), None)?;
    let pipeline = cache.get_or_create(gpu, "ember.surface_gather", GATHER, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(emitted),
            Bind::Tex(rate),
            Bind::Buf(u.any()),
            Bind::View(solid),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// `dst` = the burned fraction of `load`, 0 where there is none (spec §3.3).
pub fn surface_char(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    burned: &Field,
    load: &Field,
    dst: &Field,
) -> Result<(), GpuError> {
    expect_dims("surface_char burned", burned, u.cells())?;
    expect_dims("surface_char load", load, u.cells())?;
    expect_dims("surface_char dst", dst, u.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.surface_char", CHAR, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(burned),
            Bind::Tex(load),
            Bind::Tex(dst),
            Bind::Buf(u.any()),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}
```

Note: check `Uniforms::any()` is `pub(crate)` and reachable from this module (it is used by `fire.rs` the same way). `solid_views` requires `u.has_solids()`, so `StepConstants.has_solids` must be true (the tests do that).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo nextest run -p elements-ember`
Expected: all pass. If the reference and kernel disagree, fix the kernel against spec §3.3, not the test.

- [ ] **Step 5: Mutation proofs (one each, restore between)**

1. `surface_burn.wgsl`: change `hottest > params.ignition_temperature` to `hottest >= params.ignition_temperature`. Expected: `cold_gas_ignites_nothing` fails (`burned` non-zero).
2. `surface_burn.wgsl`: change `e = min(params.surface_burn, wood - b)` to `e = params.surface_burn`. Expected: `the_reservoir_depletes_and_emission_stops` fails ("never past the load").
3. `surface_gather.wgsl`: drop `/ f32(fluid_count(q))`. Expected: `what_the_wood_loses_the_gas_gains` fails (gained too large).
4. `surface_gather.wgsl`: replace the sum with `0.0` output. Expected: the same test and the CPU-reference tests fail.
5. `surface_burn.wgsl`: change `b > 0.0 ||` to `false ||` (a cell stops burning when the gas cools). The plan's scene must catch this: add to `the_kernels_match_the_cpu_reference_over_many_substeps` a second phase of 4 substeps where `hot` is replaced by all-zero gas, so the CPU reference keeps burning lit cells. Expected after adding it: the test fails in that phase. Restore.
6. `surface_char.wgsl`: replace `l > 0.0` with `true`. Expected: `char_is_the_burned_fraction...` fails at `out[2]`.

Record each real failure message.

- [ ] **Step 6: Commit**

```bash
just check
git add -A
git commit -m "Add the surface ignition kernels

Burning runs on the wood cells, and the fluid cells gather what they
emitted, so no atomics are needed and results are deterministic. A CPU
reference written from the spec pins both kernels, and the gather's
division by the emitter's fluid neighbours is what closes the budget.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016jxQnAfsTRPo8xRwrHJx4y"
```

---

### Task 5: Solver integration

**Files:**
- Modify: `src/solver.rs`
- Test: `tests/surface.rs`

**Interfaces:**
- Consumes: Task 4's kernels and `kernels::emit_fuel`, Task 3's `surface_burn_rate`.
- Produces: `SolverState.surface: Option<Field>` (the `burned` grid) and `SolverState::add_surface(&mut self, gpu, cache, pool) -> Result<(), GpuError>`; `Sources.surface: Option<&Field>` with `Sources::with_surface(self, load: &'a Field) -> Self`; state slot `pub const SURFACE: &str = "burned"`; solver sockets: input 7 `Field` (load), output 4 `Field` (`char`).

- [ ] **Step 1: Write the failing tests** (append to `tests/surface.rs`)

Node-level documents: a box collider with `surface_fuel`, a hot gas source beside it, fuel input fed by a zero-rate emitter, output wired to socket 4 (`char`).

```rust
/// A 16³ domain with a static wooden slab (surface_fuel) at x ≈ 1.0, a
/// temperature emitter touching it, and the fuel input connected at rate 0.
/// `socket` of the solver goes to the output. Options cut one connection.
fn slab_doc(socket: u32, heat: f32, connect_fuel: bool, connect_load: bool) -> String {
    let fuel_edge = if connect_fuel {
        r#",{ "from_node": 4, "from_index": 0, "to_node": 1, "to_index": 6 }"#
    } else {
        ""
    };
    let load_edge = if connect_load {
        r#",{ "from_node": 5, "from_index": 2, "to_node": 1, "to_index": 7 }"#
    } else {
        ""
    };
    format!(
        r#"{{ "version": 3, "dims": [16, 16, 16], "fps": 24.0, "domain_size": 2.0,
      "nodes": [
        {{ "id": 0, "kind": "ember.sphere_emitter", "params": {{ "center": [0.75, 1.0, 0.6],
           "radius": 0.2, "density_rate": 0.0, "temperature_rate": {heat:?} }} }},
        {{ "id": 1, "kind": "ember.smoke_solver", "params": {{ "buoyancy_temperature": 1.0 }} }},
        {{ "id": 2, "kind": "core.output", "params": {{}} }},
        {{ "id": 4, "kind": "ember.sphere_emitter", "params": {{ "center": [1.0, 1.0, 1.4],
           "radius": 0.1, "density_rate": 0.0, "temperature_rate": 0.0 }} }},
        {{ "id": 5, "kind": "ember.collider", "params": {{
           "shape": {{ "box": {{ "half_extents": [0.07, 0.4, 0.4] }} }},
           "transform": {{ "keys": [{{ "frame": 0, "translate": [1.0, 1.0, 0.6] }}] }},
           "surface_fuel": {{ "load": 4.0 }} }} }} ],
      "edges": [
        {{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }},
        {{ "from_node": 0, "from_index": 1, "to_node": 1, "to_index": 1 }},
        {{ "from_node": 5, "from_index": 0, "to_node": 1, "to_index": 4 }},
        {{ "from_node": 5, "from_index": 1, "to_node": 1, "to_index": 5 }},
        {{ "from_node": 1, "from_index": {socket}, "to_node": 2, "to_index": 0 }}{fuel_edge}{load_edge} ],
      "output": 2 }}"#
    )
}

fn char_at_frame(doc: &str, frame: u32) -> Vec<f32> {
    let mut s = Session::new(doc);
    let mut t = timeline(0);
    s.density_bits(&mut t, frame)
        .iter()
        .map(|b| f32::from_bits(*b))
        .collect()
}

#[test]
fn a_slab_with_no_heat_never_chars() {
    let char = char_at_frame(&slab_doc(4, 0.0, true, true), 20);
    assert!(char.iter().all(|&c| c == 0.0), "no heat, no ignition");
}

#[test]
fn a_slab_beside_a_heat_source_chars_and_stays_in_range() {
    let char = char_at_frame(&slab_doc(4, 30.0, true, true), 20);
    let charred = char.iter().filter(|&&c| c > 0.0).count();
    assert!(charred > 0, "a hot source must ignite the slab");
    assert!(char.iter().all(|&c| (0.0..=1.0).contains(&c)), "char in [0, 1]");
}

#[test]
fn char_is_bit_identical_however_frame_40_is_reached() {
    assert_doc_frame_40_is_bit_identical(&slab_doc(4, 30.0, true, true));
}

#[test]
fn the_load_input_needs_fuel_and_a_collider() {
    use elements_core::graph::NodeError;
    fn eval_error(doc: &str) -> NodeError {
        let mut s = Session::new(doc);
        let mut t = timeline(0);
        t.goto(&s.graph, &s.gpu, &mut s.pool, &mut s.pipelines, s.dims, 1)
            .err()
            .expect("must fail")
    }
    // Load without fuel.
    assert!(matches!(
        eval_error(&slab_doc(4, 1.0, false, true)),
        NodeError::IncompletePair { .. }
    ));
    // Load without a collider: drop the collider edges.
    let doc = slab_doc(4, 1.0, true, true)
        .replace(r#"{ "from_node": 5, "from_index": 0, "to_node": 1, "to_index": 4 },"#, "")
        .replace(r#"{ "from_node": 5, "from_index": 1, "to_node": 1, "to_index": 5 },"#, "");
    assert!(matches!(eval_error(&doc), NodeError::IncompletePair { .. }));
}

#[test]
fn connecting_the_load_changes_the_state_shape() {
    use elements_core::graph::{NodeId, StateStore, Time};
    use elements_ember::solver::SURFACE;
    // Run one frame with the surface connected, then evaluate a graph without
    // it against the same state: the timeline's StateShape reset path.
    let on = Session::new(&slab_doc(4, 30.0, true, true));
    let mut state = StateStore::new();
    let _ = (&on, &mut state, NodeId(1), Time { frame: 1, dt: 1.0 / 24.0 }, SURFACE);
    // The precise assertion is written against the StateStore API used by
    // tests/fire_scene.rs `connecting_fuel_changes_the_state_shape`; copy that
    // test and change the cut connection from fuel (input 6) to load (input 7).
    unimplemented!("port connecting_fuel_changes_the_state_shape");
}
```

The last test is a skeleton: open `tests/fire_scene.rs:155` (`connecting_fuel_changes_the_state_shape`), copy its body, and replace the fuel document by `slab_doc(4, 30.0, true, true)` versus `slab_doc(4, 30.0, true, false)`, asserting the same `StateShape { slot: SURFACE }` outcome. Replace the skeleton body completely; the `unimplemented!` must not remain.

Also add the substep-level budget test, through `substep()` (closed domain, fire on, `burning_rate` 0 so gas fuel is not consumed):

```rust
#[test]
fn a_full_substep_moves_the_wood_into_the_gas_fuel() {
    use elements_ember::solver::{PressureSolve, SolverState, Sources, substep};
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let (mask, load) = plank(3.0);
    let zero = vec![0.0; K.voxel_count()];
    let density_src = upload(&gpu, &mut pool, K, &zero);
    let heat = upload(&gpu, &mut pool, K, &gas(&[([3, 3, 3], 2.0)]));
    let no_fuel = upload(&gpu, &mut pool, K, &zero);
    let load_f = upload(&gpu, &mut pool, K, &load);
    let mask_f = upload(&gpu, &mut pool, K, &mask);
    let velocity = pool.acquire_staggered_zeroed(&gpu, &mut cache, K).unwrap();
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, K).unwrap();
    state.add_fire(&gpu, &mut cache, &mut pool).unwrap();
    state.add_surface(&gpu, &mut cache, &mut pool).unwrap();
    // The hot cell must exist in the gas before the burn reads it: substep
    // emits temperature first, so emit it at a rate that lands above IGN.
    let c = StepConstants {
        open_mask: 0,
        fire: true,
        has_solids: true,
        surface_burn_rate: BURN_RATE,
        ignition_temperature: IGN,
        max_temperature: 3.0,
        burning_rate: 0.0,
        ..StepConstants::new(K, H, 0.125)
    };
    let sources = Sources::new(&density_src, &heat)
        .with_fuel(&no_fuel)
        .with_solids(Solids { mask: &mask_f, velocity: &velocity })
        .with_surface(&load_f);
    substep(&gpu, &mut cache, &mut pool, &mut state, sources, &c, PressureSolve::GaussSeidel(40))
        .unwrap();
    let lost: f64 = state.surface.as_ref().unwrap().read_back(&gpu).unwrap()
        .iter().map(|&b| f64::from(b)).sum();
    let gained: f64 = state.fire.as_ref().unwrap().fuel.read_back(&gpu).unwrap()
        .iter().map(|&f| f64::from(f)).sum();
    assert!(lost > 0.0, "the hot cell must ignite the plank");
    assert!((lost - gained).abs() <= 1e-3 * lost, "lost {lost}, gained {gained}");
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p elements-ember --test surface`
Expected: compile errors (`add_surface`, `with_surface`, `SURFACE` missing).

- [ ] **Step 3: Implement**

1. **State.** Add `pub const SURFACE: &str = "burned";`, `const SURFACE_INPUT: u32 = 7;`. Add `pub surface: Option<Field>` to `SolverState` (set `surface: None` in `zeroed` and in `take_state`'s literal), release it in `release_to`, and

```rust
/// A zeroed `burned` grid, for a state that has a surface (FT4 spec §3.2).
pub fn add_surface(&mut self, gpu: &GpuContext, cache: &mut PipelineCache, pool: &mut FieldPool) -> Result<(), GpuError> {
    let burned = pool.acquire_zeroed(gpu, cache, self.density.dims())?;
    if let Some(old) = self.surface.replace(burned) {
        pool.release(old);
    }
    Ok(())
}
```

2. **Sources.** `pub surface: Option<&'a Field>` (doc: "The wood load per cell when a surface burns (FT4 spec §3.1)."), `surface: None` in `new`, `with_surface`.

3. **Substep.** In `pre_projection`, after the first `emit_fuel` and before `kernels::burn`, inside the `if self.fire` block, add:

```rust
if let Some(load) = sources.surface {
    let (Some(burned), Some(solids)) = (state.surface.as_ref(), sources.solids) else {
        return Err(GpuError::Validation(
            "a surface substep needs the burned state and a collider".to_owned(),
        ));
    };
    let cells = self.uniforms.cells();
    let emitted = pool.acquire(gpu, cells, FieldFormat::R32Float)?;
    let rate = match pool.acquire(gpu, cells, FieldFormat::R32Float) {
        Ok(rate) => rate,
        Err(e) => {
            pool.release(emitted);
            return Err(e);
        }
    };
    // Both scratch fields are written in every cell. They wait in `retired`
    // until the batch has run.
    self.retired.extend([emitted, rate]);
    let emitted = &self.retired[self.retired.len() - 2];
    let rate = &self.retired[self.retired.len() - 1];
    kernels::surface_burn(gpu, cache, &mut self.batch, u, load, &state.temperature, solids, burned, emitted)?;
    kernels::surface_gather(gpu, cache, &mut self.batch, u, solids, emitted, rate)?;
    kernels::emit_fuel(gpu, cache, &mut self.batch, u, &fire.fuel, &fire.react, rate)?;
}
```

The borrow of `self.retired` while `self.batch` is borrowed mutably and `u = &self.uniforms` is shared: if the borrow checker objects, bind `let batch = &mut self.batch;` and `let retired = &mut self.retired;` as disjoint field borrows before the block. Check `Field` release accepts a field pushed to `retired` (it is `Vec<Field>`).

4. **Node.** `take_state(ctx, fire, surface)`: add `SURFACE` to the `expected` list when `surface`; the zeroed branch calls `state.add_surface` after `add_fire` when `surface` (release `state` on error as the fire branch does); after taking `react`, `let surface_field = surface.then(|| field(SURFACE));` and put it in the `SolverState` literal (`surface: surface_field`); in the slot-collection loop chain `SURFACE_SLOTS: [&str; 1] = [SURFACE]`; the `wrong_shape` match treats `SURFACE` as `Value::Field` (it already does for any slot other than velocity). `put_state` puts `SURFACE`. In `eval`: `let surface = ctx.input_connected(SURFACE_INPUT);` and before `take_state`:

```rust
if surface && !fire {
    return Err(NodeError::IncompletePair { node: ctx.node_id(), connected: SURFACE_INPUT, missing: FUEL_INPUT });
}
if surface && !(ctx.input_connected(4) && ctx.input_connected(5)) {
    return Err(NodeError::IncompletePair { node: ctx.node_id(), connected: SURFACE_INPUT, missing: 4 });
}
```

`run` adds `if ctx.input_connected(SURFACE_INPUT) { wanted.push(SURFACE_INPUT); }`; `step` adds `if find(SURFACE_INPUT).is_some() { sources = sources.with_surface(field(SURFACE_INPUT)?); }` before `step_frame`; `step_frame`'s `StepConstants` literal adds nothing (the rate already comes from `step_constants`). `sockets` inputs gain a trailing `SocketType::Field`, outputs a trailing `SocketType::Field`. `run`'s `wanted` array for outputs becomes `[bool; 5]`, `copy_outputs` gets `wanted: [bool; 5]`, the index match gets:

```rust
4 => char_output(gpu, cache, pool, state, loads, dx).map(Value::Field),
```

where `char_output` mirrors `flame_output`: with `state.surface` `None` it returns `pool.acquire_zeroed`; otherwise it runs `kernels::surface_char(gpu, cache, &mut batch, &u, burned, load, &dst)` with `load` the input-7 field. `copy_outputs` therefore needs the load: change `run` to keep the taken input 7 until after `copy_outputs` (it currently releases all inputs before copying), pass `Option<&Field>` into `copy_outputs`, and release inputs after. Update the `unreachable!` message ("only five outputs exist").

5. **Docs in code.** Update the `SmokeSolver` doc/`sockets` comments to list input 7 and output 4.

- [ ] **Step 4: Run tests**

Run: `cargo nextest run -p elements-ember`
Expected: all pass, including the Task 3 guard hash, the existing fire and solver suites and every new test.

- [ ] **Step 5: Mutation proofs (one each, restore between)**

1. `solver.rs`: delete the `kernels::emit_fuel(... rate)` call. Expected: `a_full_substep_moves_the_wood_into_the_gas_fuel` fails (gained 0).
2. `solver.rs`: call `surface_burn` before the first `emit_fuel`/temperature emit (move it above `kernels::emit`). Expected: the substep test fails (the hot cell does not exist yet) — if not, record it as not load-bearing and drop the claim.
3. `solver.rs`: make `put_state` skip `SURFACE`. Expected: `char_is_bit_identical_however_frame_40_is_reached` and the state-shape test fail.
4. `solver.rs`: remove the `surface && !fire` check. Expected: `the_load_input_needs_fuel_and_a_collider` fails.
5. `char_output`: return zeros always. Expected: `a_slab_beside_a_heat_source_chars_and_stays_in_range` fails.

Record each failure message.

- [ ] **Step 6: Commit**

```bash
just check
git add -A
git commit -m "Wire the surface into the solver

Input 7 turns the surface on and adds the burned state, so a timeline
restore replays it exactly; the wood's emission goes through emit_fuel
ahead of the burn, which is how the gas gets both fuel and, through
the flame profile, the heat that ignites neighbouring wood.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016jxQnAfsTRPo8xRwrHJx4y"
```

---

### Task 6: The front spreads, and the record

**Files:**
- Modify: `tests/surface.rs`, `AGENTS.md`, `.superpowers/sdd/progress.md` (git-ignored or not: update it either way), `docs/superpowers/specs/2026-09-30-flamethrower-napalm-roadmap-design.md` (resolve §6's first open item)
- Create: `docs/bench/surface-ignition.md`

**Interfaces:**
- Consumes: Task 5's node and `slab_doc`-style documents.
- Produces: a recorded front-spread measurement and updated project docs.

- [ ] **Step 1: Write the spread test**

Append a scene where the heat source is at the bottom of a **vertical** wall, so flame (which rises) can ignite higher wall cells. Use the `substep()` API with open top and a closed-ish domain as in `tests/fire.rs`, or a document like `slab_doc` with a `temperature_rate` emitter at the wall's foot and a tall box (`half_extents [0.0625, 0.4, 0.8]`) in a 16×16×32 domain. Sample `char` at each frame; per row `k` of the wall record the first frame where any cell in that row has `char > 0` (`ignition_frame[k]`). Assert:

```rust
let rows: Vec<(usize, u32)> = /* rows that ignited, (k, first frame), ordered by k */;
assert!(rows.len() >= 3, "the front must reach at least 3 wall rows, got {rows:?}");
assert!(
    rows.windows(2).all(|w| w[1].1 >= w[0].1),
    "ignition frames must not decrease with height: {rows:?}"
);
assert!(rows[0].0 < rows.last().unwrap().0, "the front must have moved");
```

Choose the scene's constants once (source rate, wall height, frames ≤ 60, preview-style solver params) and state them in the test's doc comment. Allow at most three scene adjustments if the front does not spread; record every attempt (constants and outcome) in `docs/bench/surface-ignition.md`. If it still does not spread, do **not** loosen the assertions: commit the test marked `#[ignore = "front does not spread: see docs/bench/surface-ignition.md"]`, record the finding, and report DONE_WITH_CONCERNS. The emergent result is a finding, not a failure to hide.

- [ ] **Step 2: Run, mutation-prove**

Run: `cargo nextest run -p elements-ember --test surface -- spreads`
Mutation: in `surface_gather.wgsl` output `0.0` instead of the sum. Expected: the test fails (no spread beyond the heated cells), the same mutation as the spec's test 5. Record the message. Restore.

- [ ] **Step 3: Write `docs/bench/surface-ignition.md`**

Record: machine (Apple M1 Max, load average from `uptime`), base commit, scene constants, per-row ignition frames, the mutation output, and any scene adjustments. State what was not measured: no flame-meets-shack case (FT3b found preview's single substep never brings flame to the shack), no timing, one seed and resolution.

- [ ] **Step 4: Update the project records**

- `AGENTS.md`: in the flamethrower paragraph, add that FT4 (surface ignition) is complete, with a link to the spec and this plan; add both to the document list.
- Roadmap spec §6: replace the open item on where the reservoir lives with the decision (extend `ember.collider`; user decision 2026-09-30).
- `.superpowers/sdd/progress.md`: one ledger line per task with its commit.

- [ ] **Step 5: Whole-suite check and CI**

Run: `just check`. Expected: lint and tests pass (the guard hash passes on this adapter). Push is the user's call: do not push; report the branch for them to open a PR. Note in the summary that the llvmpipe CI run has not happened, since AGENTS.md requires distinguishing local Metal from CI results.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Test that the surface front spreads and record FT4

The front-spread test is the one emergent check in FT4, so its scene
constants and every adjustment are written down with the result.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016jxQnAfsTRPo8xRwrHJx4y"
```

---

## Self-Review

**Spec coverage.** §3.1 sockets: Task 1 (collider/mesh output), Task 2 (union), Task 5 (solver input 7, output 4). §3.2 state and params: Tasks 3 and 5. §3.3 kernels and `char`: Task 4 (kernels) and Task 5 (`char` output, ordering). §3.4 limits: the static-collider rejection is Task 1; the band/inert statements need no code. §3.5 off-switch: Task 3's hash, kept passing in Tasks 4 to 6. §4 tests: 1 (`cold_gas_ignites_nothing`, `a_slab_with_no_heat_never_chars`), 2 (`gas_above_the_threshold...`), 3 (`what_the_wood_loses...`, substep test), 4 (`the_reservoir_depletes...`), 5 (Task 6), 6 (Task 2), 7 (`char_is_bit_identical...`), 8 (Task 3), 9 (`the_load_input_needs_fuel...`, Task 1 rejection). §5 done-when: Task 6. The spec's test 7 mutation (atomics in place of the gather) is not practical to apply mechanically, so determinism is instead proven by the CPU-reference test plus the timeline test, and the `put_state` mutation in Task 5.

**Placeholder scan.** Task 5's state-shape test is a skeleton that tells the implementer to port `connecting_fuel_changes_the_state_shape` and must not leave `unimplemented!`; the zeroed-field helper in Task 2 names a constructor to look up. Both are concrete instructions, flagged in place. Hash `WANT` is measured, not guessed.

**Type consistency.** `SurfaceFuel`, `fill_surface_load`, `union_surface_load`, `surface_burn`/`surface_gather`/`surface_char`, `SolverState::add_surface`, `Sources::with_surface`, `SURFACE`, `SURFACE_INPUT` are used with the same names and signatures across tasks. `StepConstants.surface_burn_rate` (Task 3) is what Task 4's tests set.
