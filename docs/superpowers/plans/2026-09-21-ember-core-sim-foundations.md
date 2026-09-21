# Ember Piece 1 — Core Sim Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `elements-core` persistent simulation state, a deterministic timeline with a frame cache, time reaching nodes, staggered vector fields, and outputs with more than one consumer. All of it is proven by a toy stateful node, `core.accumulate`, with no solver written.

**Architecture:** A `StateStore` of fields keyed by `(NodeId, slot)` lives beside the `FieldPool` and survives across evaluations. Nodes opt in with `Node::stateful()`. A `Timeline` owns the state and a budgeted cache of snapshots of the state *entering* each frame. So producing any frame means restoring the nearest earlier snapshot and stepping forward, and a cache hit costs exactly one `eval`. The graph evaluator counts consumers per output, lends inputs by reference, moves a value to its last consumer and copies it for earlier ones. It returns every field to the pool after its last use.

**Tech Stack:** Rust 2024, `wgpu` 30.0.1 (no optional features), WGSL compute, `serde`/`serde_json`, `thiserror`, `anyhow` (CLI), Python 3.11 stdlib (add-on). Dev: `tempfile`, `vdb-rs`.

**Spec:** `docs/superpowers/specs/2026-09-21-ember-core-sim-foundations-design.md`
**Umbrella:** `docs/superpowers/specs/2026-09-21-ember-design.md`

---

## Global Constraints

Every task's requirements implicitly include this section.

- **`required_features` stays `wgpu::Features::empty()`.** Raising a *limit*
  is allowed. Enabling a *feature* is not: if something seems to need one, stop
  and report it as a design problem.
- **Scalar and face fields are `R32Float`.** Never `R16Float`.
- **`#![forbid(unsafe_code)]`** stays on every crate except `elements-ipc`.
- **Crate manifests use `dep.workspace = true`.** No new dependencies are needed by this plan. If you think you need one, stop and ask.
- **Every stochastic node takes an explicit `seed: u64`.** (No stochastic node is added here.)
- **Determinism:** the same document on the same machine gives bit-identical
  frames, however a frame is reached. Tests compare `f32::to_bits`.
- **The engine never panics on untrusted input.** Documents and IPC commands are untrusted.
- **Never set `WGPU_BACKEND` locally.** The backend is Metal. `just ci-test` alone sets it.
- **`just check` must pass before every commit.** It runs `cargo fmt --check`, `clippy -D warnings`, `ruff check` and the full test suite.
- **Prove each new test can fail.** Every task lists a mutation per test. Apply
  it, run the test, confirm it fails, restore, and paste the real failing output
  into the task report. **A mutation changes exactly one thing.**
- **Commit style (CLAUDE.md):** a plain imperative subject line, then a body explaining *why*. End every commit message with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01XCzEwqREbRMwabH11TvnMb
  ```
  The repo is jj-colocated. Commit with plain `git`, and do not run `jj` commands.
- **Python targets 3.11** (the compatibility floor) and is linted with `ruff`.
  Add-on code uses relative imports only.

---

## Corrections to the spec made while planning

Each was found by checking the code or the vendored `wgpu` source. The spec has been updated to match.

1. **The 3D texture limit must be raised.** `GpuContext` requests
   `Limits::downlevel_defaults()`, whose `max_texture_dimension_3d` is **256**
   (`wgpu-types-30.0.1/src/limits.rs`). A staggered face of a 256³ domain is 257
   wide and cannot be allocated, and the 512³ bakes the umbrella promises are
   impossible. Task 1 requests the adapter's own resolution limits with
   `Limits::using_resolution`, which is a limit, not a feature. The daemon then
   requires each domain axis to be strictly below the limit, so faces fit.
2. **The document version bumps to 2.** The spec said it could stay 1 because the
   new fields are optional. But `ELEMENTS_DOC_VERSION`'s own doc comment requires
   a bump plus a migration test for any schema change, and for good reason: serde
   ignores unknown fields, so an older engine would silently drop `fps`. Version 2
   makes an old engine reject a new document loudly. Version-1 documents still
   load, with defaults filled in (Task 9).
3. **Restore copies out of a snapshot.** The cache must keep a snapshot after it
   is restored, because the same frame may be scrubbed to again. So
   `StateStore::restore` duplicates rather than moves. The spec said "swaps them back in".
4. **There is no separate `StateValue` type.** State slots hold a `Value`, which already has exactly the needed variants (`Field`, `VectorField`, `Scalar`).
5. **`acquire_zeroed` is `R32Float` only.** It reuses `constant.wgsl`, which writes `r32float`. Nothing needs a zeroed `Rgba16Float`.
6. **`Graph::connect` rejects a second edge into one input.** Today it silently
   overwrites the first. The spec says "each input still accepts exactly one
   edge", so this is enforced with `NodeError::InputAlreadyConnected`, which replaces the deleted `AlreadyConsumed`.
7. **The timeline warning goes through `Timeline::take_warning()`**, not a log
   call. Core has no logging dependency. The daemon and CLI print it to stderr.
8. **Negative Blender frames are clamped to 0 in the add-on.** Blender allows
   negative frames. `Command::Render { frame: u32 }` and the CLI's `--frames` do not.

Also found, and recorded in the umbrella's risks for pieces 2 and 3 but **not in
scope here**: `downlevel_defaults` allows only **4 storage textures per shader
stage**, so solver kernels must read through `texture_3d<f32>` + `textureLoad`
and write through storage. And `max_buffer_size` is 256 MiB, so reading back a
512³ `R32Float` field (512 MiB) must be done in chunks.

---

## File Structure

```
crates/elements-core/src/
  gpu/mod.rs              MODIFY  adapter resolution limits; new exports
  gpu/field.rs            MODIFY  Field::copy_to, Field::write
  gpu/pool.rs             MODIFY  acquire_zeroed, duplicate, staggered acquire/release, allocation_count
  gpu/staggered.rs        CREATE  Axis, StaggeredField
  gpu/dispatch.rs         MODIFY  accumulate_into
  gpu/shaders/accumulate.wgsl     CREATE  sum += input * dt
  gpu/shaders/vector_sample.wgsl  CREATE  hand-written trilinear + staggered velocity sampling (a library)
  graph/mod.rs            MODIFY  eval_frame, fan-out evaluator, Evaluated, EvalStats, is_stateful
  graph/node.rs           MODIFY  Value::VectorField, release_to, duplicate, gpu_bytes;
                                  EvalCtx state/time/inputs; new NodeError variants
  graph/socket.rs         MODIFY  SocketType::VectorField
  graph/time.rs           CREATE  Time, DEFAULT_FPS, DEFAULT_START_FRAME
  graph/state.rs          CREATE  StateStore, Snapshot
  graph/timeline.rs       CREATE  Timeline, TimelineConfig, DEFAULT_CACHE_BUDGET_MB
  graph/document.rs       MODIFY  version 2, fps, start_frame, cache_budget_mb, timeline_config()
  nodes/accumulate.rs     CREATE  core.accumulate
  nodes/mod.rs            MODIFY  register core.accumulate
  nodes/constant_field.rs, nodes/noise_field.rs, nodes/output.rs  MODIFY (API renames)
crates/elements-core/tests/
  gpu_context.rs, field_pool.rs, graph.rs, document.rs, nodes.rs   MODIFY
  staggered.rs, state.rs, fan_out.rs, vector_sample.rs, timeline.rs CREATE
crates/elementsd/src/daemon.rs       MODIFY  timeline per loaded graph; limit check
crates/elementsd/tests/session.rs    MODIFY
crates/elementsd/tests/python_contract.rs MODIFY
crates/elements-cli/src/bake.rs      MODIFY  bake through the timeline
crates/elements-cli/tests/cli.rs     MODIFY
tests/graphs/accumulate_4.elements   CREATE  shared stateful fixture
tests/python/contract.py             MODIFY
addon/blender_elements/bakecmd.py    CREATE  bpy-free bake command builder
addon/blender_elements/handlers.py   MODIFY  bake the scene's frame
addon/blender_elements/ops.py        MODIFY
addon/blender_elements/client.py     MODIFY  clamp negative frames
```

Task order is dependency order. Each task leaves `just check` green.

---

## Task 1: Request the adapter's texture-size limits

**Why:** a staggered face is one cell larger than its domain along one axis.
Under `downlevel_defaults` (3D max 256), a 256³ domain's faces cannot be allocated at all.

**Files:**
- Modify: `crates/elements-core/src/gpu/mod.rs` (inside `new_headless_async`)
- Modify: `crates/elementsd/src/daemon.rs` (the `max_dim` loop in `load`)
- Test: `crates/elements-core/tests/gpu_context.rs`, `crates/elementsd/tests/session.rs`

**Interfaces:**
- Consumes: nothing new.
- Produces: devices whose `limits().max_texture_dimension_3d` equals the
  adapter's (2048 on Apple Silicon and on lavapipe). The daemon rejects any
  document axis `>= max_texture_dimension_3d`.

- [ ] **Step 1: Write the failing core test**

Append to `crates/elements-core/tests/gpu_context.rs`, and extend its `use` line to
`use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext, GpuError};`:

```rust
/// A staggered velocity face of a 256³ domain is 257 cells along its own axis.
/// `Limits::downlevel_defaults()` caps 3D textures at 256, so this allocation
/// is the capability Ember needs, tested directly.
#[test]
fn a_staggered_face_of_a_256_domain_can_be_allocated() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let face = pool.acquire(&ctx, FieldDims::new(257, 4, 4), FieldFormat::R32Float);
    assert!(face.is_ok(), "257-wide 3D texture was refused: {:?}", face.err());
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cargo nextest run -p elements-core --test gpu_context`
Expected: FAIL. The assertion message contains `Validation` and mentions the 3D dimension limit.

- [ ] **Step 3: Request the adapter's resolution limits**

In `crates/elements-core/src/gpu/mod.rs`, replace
`required_limits: wgpu::Limits::downlevel_defaults(),` with `required_limits,`
and add this just before the `request_device` call:

```rust
        // Resolution limits come from the adapter. `downlevel_defaults` caps
        // 3D textures at 256, which cannot hold a staggered face (n + 1 cells)
        // of a 256³ domain, let alone a 512³ bake. These are limits, not
        // features, so `required_features` stays empty and the portability
        // guarantee is unchanged.
        let required_limits =
            wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
```

- [ ] **Step 4: Run the core test and watch it pass**

Run: `cargo nextest run -p elements-core --test gpu_context`
Expected: PASS.

- [ ] **Step 5: Tighten the daemon's load check**

In `crates/elementsd/src/daemon.rs`, replace the comment block and loop that
check `max_dim` in `load` with:

```rust
    // `GpuContext` requests the adapter's own resolution limits, so this is
    // the real hardware cap (2048 on Apple Silicon). Without this check, a
    // document whose dims exceed it answers `Loaded` here (nothing above
    // touches the GPU) and only fails later, on every render, as a generic
    // `ErrorKind::Gpu`, which the add-on treats as transient and retries.
    //
    // The comparison is `>=`, not `>`: staggered vector fields store one more
    // face than cells along each axis, so a domain axis must stay strictly
    // below the limit for its faces to fit.
    let max_dim = session.gpu.device().limits().max_texture_dimension_3d;
    for (axis, value) in [("x", dims.x), ("y", dims.y), ("z", dims.z)] {
        if value >= max_dim {
            return Err(EngineError::new(
                ErrorKind::Document,
                format!(
                    "dims {:?}: {axis} = {value} must be below this device's \
                     max_texture_dimension_3d of {max_dim} (staggered faces need one extra cell)",
                    [dims.x, dims.y, dims.z]
                ),
            ));
        }
    }
```

- [ ] **Step 6: Update the daemon test that relied on the old 256 cap**

In `crates/elementsd/tests/session.rs`,
`dims_exceeding_the_devices_texture_limit_are_rejected_at_load_not_render` uses
`[300, 300, 300]`, which is now legal. Change the document's `dims` to
`[65536, 4, 4]` (above every real adapter's 3D limit, and still a small channel
file), and change the message assertion to:

```rust
            assert!(
                e.message.contains("65536"),
                "message should name the offending dimension: {}",
                e.message
            );
```

- [ ] **Step 7: Run the daemon tests**

Run: `cargo nextest run -p elementsd --test session`
Expected: PASS, all tests.

- [ ] **Step 8: Prove the tests can fail**

- Mutation A: in `gpu/mod.rs`, change `required_limits,` back to
  `required_limits: wgpu::Limits::downlevel_defaults(),`. Run the core test → FAIL. Restore.
- Mutation B: in `daemon.rs`, change `value >= max_dim` to `value > max_dim + 70000`.
  Run `cargo nextest run -p elementsd --test session dims_exceeding` → FAIL. Restore.

Record both failing outputs in the task report.

- [ ] **Step 9: `just check`, then commit**

```bash
just check
git add crates/elements-core/src/gpu/mod.rs crates/elements-core/tests/gpu_context.rs \
        crates/elementsd/src/daemon.rs crates/elementsd/tests/session.rs
git commit -F - <<'EOF'
Request the adapter's texture-size limits instead of downlevel defaults

downlevel_defaults caps 3D textures at 256. A staggered velocity face is
one cell larger than its domain, so a 256³ domain could not store its
velocity at all, and the 512³ bakes Ember promises were impossible.
Resolution limits are limits, not features, so required_features stays
empty. The daemon now keeps each axis strictly below the limit so that
faces always fit.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XCzEwqREbRMwabH11TvnMb
EOF
```

---

## Task 2: Make the acquire contract explicit

**Why:** a recycled field holds whatever its last user wrote. Every Core v1 node
writes every voxel, so that was safe. Piece 2's emitters write only part of a
field and would silently read the previous frame's data. Zeroing becomes a
named, deliberate choice.

**Files:**
- Modify: `crates/elements-core/src/gpu/pool.rs`
- Modify: `crates/elements-core/src/graph/node.rs` (`EvalCtx::acquire`)
- Modify: `crates/elements-core/src/nodes/constant_field.rs`, `crates/elements-core/src/nodes/noise_field.rs`
- Test: `crates/elements-core/tests/field_pool.rs`

**Interfaces:**
- Consumes: `fill_constant(ctx, cache, &Field, f32)` from `gpu/dispatch.rs`.
- Produces:
  - `FieldPool::acquire_zeroed(&mut self, ctx: &GpuContext, cache: &mut PipelineCache, dims: FieldDims) -> Result<Field, GpuError>` (always `R32Float`)
  - `EvalCtx::acquire_uninit(&mut self, format: FieldFormat) -> Result<Field, NodeError>` (renamed from `acquire`)
  - `EvalCtx::acquire_zeroed(&mut self) -> Result<Field, NodeError>` (`R32Float` at `ctx.dims()`)

- [ ] **Step 1: Write the failing test**

Append to `crates/elements-core/tests/field_pool.rs`, and change its `use` line to
`use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache, fill_constant};`:

```rust
/// `acquire_zeroed` must clear a recycled texture, not just a fresh one.
/// Fresh allocations are already zero, so this test dirties a texture, returns
/// it to the pool, and asserts that the zeroed acquire got the SAME texture back.
/// Otherwise it would prove nothing.
#[test]
fn acquire_zeroed_clears_a_dirty_recycled_field() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let dims = FieldDims::new(4, 4, 4);

    let dirty = pool.acquire(&ctx, dims, FieldFormat::R32Float).unwrap();
    fill_constant(&ctx, &mut cache, &dirty, 5.0).unwrap();
    let generation = dirty.pool_generation();
    pool.release(dirty);

    let field = pool.acquire_zeroed(&ctx, &mut cache, dims).unwrap();
    assert_eq!(
        field.pool_generation(),
        generation,
        "must reuse the dirty texture, or this test proves nothing"
    );
    let values = field.read_back(&ctx).unwrap();
    assert!(values.iter().all(|&v| v == 0.0), "got {values:?}");
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cargo nextest run -p elements-core --test field_pool`
Expected: FAIL to compile: `no method named acquire_zeroed`.

- [ ] **Step 3: Add `FieldPool::acquire_zeroed`**

In `crates/elements-core/src/gpu/pool.rs`, change the `use` line to
`use super::{Field, FieldDims, FieldFormat, GpuContext, GpuError, PipelineCache, fill_constant};`
and add inside `impl FieldPool`, after `acquire`:

```rust
    /// Take an `R32Float` field of these dims and fill it with zeros.
    ///
    /// Use this whenever a caller may write only part of a field, such as an
    /// emitter, or a state slot on its first step. The zeroing is a compute pass
    /// (`constant.wgsl` at 0), because `CommandEncoder::clear_texture` needs
    /// the optional `Features::CLEAR_TEXTURE`, and the engine enables no optional features.
    pub fn acquire_zeroed(
        &mut self,
        ctx: &GpuContext,
        cache: &mut PipelineCache,
        dims: FieldDims,
    ) -> Result<Field, GpuError> {
        let field = self.acquire(ctx, dims, FieldFormat::R32Float)?;
        fill_constant(ctx, cache, &field, 0.0)?;
        Ok(field)
    }
```

- [ ] **Step 4: Rename `EvalCtx::acquire` and add `acquire_zeroed`**

In `crates/elements-core/src/graph/node.rs`, replace the `acquire` method with:

```rust
    /// Acquire a pooled field at the current evaluation dims WITHOUT clearing it.
    ///
    /// The contents are whatever the texture's last user left. Use this only
    /// when the node writes every voxel before anything reads the field.
    pub fn acquire_uninit(&mut self, format: FieldFormat) -> Result<Field, NodeError> {
        let dims = self.dims;
        Ok(self.pool.acquire(self.gpu, dims, format)?)
    }

    /// Acquire a pooled `R32Float` field at the current evaluation dims, filled with zeros.
    pub fn acquire_zeroed(&mut self) -> Result<Field, NodeError> {
        let dims = self.dims;
        Ok(self.pool.acquire_zeroed(self.gpu, self.pipelines, dims)?)
    }
```

In `nodes/constant_field.rs` and `nodes/noise_field.rs`, change
`ctx.acquire(FieldFormat::R32Float)?` to `ctx.acquire_uninit(FieldFormat::R32Float)?`.
Both nodes write every voxel.

- [ ] **Step 5: Run the tests and watch them pass**

Run: `cargo nextest run -p elements-core`
Expected: PASS, all tests.

- [ ] **Step 6: Prove the test can fail**

Mutation: in `acquire_zeroed`, delete the line `fill_constant(ctx, cache, &field, 0.0)?;`.
Run `cargo nextest run -p elements-core --test field_pool acquire_zeroed` → FAIL,
printing values of `5.0`. Restore.

- [ ] **Step 7: `just check`, then commit**

```bash
just check
git add crates/elements-core/src/gpu/pool.rs crates/elements-core/src/graph/node.rs \
        crates/elements-core/src/nodes/constant_field.rs crates/elements-core/src/nodes/noise_field.rs \
        crates/elements-core/tests/field_pool.rs
git commit -F - <<'EOF'
Split field acquisition into explicit uninit and zeroed variants

A recycled field holds whatever its last user wrote, and the only guard
was a doc comment. Every Core v1 node writes every voxel, so that was
safe. The first masked write, an Ember emitter, would silently read the
previous frame's data. Naming the choice at every call site makes the
hazard visible where it happens.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XCzEwqREbRMwabH11TvnMb
EOF
```

---

## Task 3: Staggered vector fields

**Why:** Ember stores velocity on a MAC grid: three `R32Float` face textures,
each one cell larger along its own axis. That is the only layout that keeps
`read_write` storage access within the WebGPU baseline (umbrella E1).

**Files:**
- Create: `crates/elements-core/src/gpu/staggered.rs`
- Modify: `crates/elements-core/src/gpu/mod.rs` (module and exports)
- Modify: `crates/elements-core/src/gpu/pool.rs`
- Modify: `crates/elements-core/src/graph/socket.rs`
- Modify: `crates/elements-core/src/graph/node.rs`
- Test: `crates/elements-core/tests/staggered.rs` (create)

**Interfaces:**
- Consumes: `FieldPool::acquire`, `FieldPool::acquire_zeroed` (Task 2).
- Produces:
  - `pub enum Axis { X, Y, Z }` with `Axis::ALL: [Axis; 3]`
  - `pub struct StaggeredField` with
    - `StaggeredField::face_dims(cells: FieldDims, axis: Axis) -> FieldDims`
    - `StaggeredField::from_faces(cells: FieldDims, faces: [Field; 3]) -> Result<StaggeredField, GpuError>`
    - `.cells() -> FieldDims`, `.face(axis) -> &Field`, `.into_faces(self) -> [Field; 3]`
  - `FieldPool::acquire_staggered_uninit(&mut self, ctx, cells) -> Result<StaggeredField, GpuError>`
  - `FieldPool::acquire_staggered_zeroed(&mut self, ctx, cache, cells) -> Result<StaggeredField, GpuError>`
  - `FieldPool::release_staggered(&mut self, field: StaggeredField)`
  - `SocketType::VectorField`, `Value::VectorField(StaggeredField)`, `Value::as_vector_field(&self) -> Result<&StaggeredField, NodeError>`
  - `Value::release_to(self, pool: &mut FieldPool)`: returns every texture a value owns to the pool
  - `EvalCtx::acquire_vector_uninit(&mut self)` and `EvalCtx::acquire_vector_zeroed(&mut self)`, both `-> Result<StaggeredField, NodeError>` at `ctx.dims()`

- [ ] **Step 1: Write the failing tests**

Create `crates/elements-core/tests/staggered.rs`:

```rust
use elements_core::gpu::{
    Axis, FieldDims, FieldFormat, FieldPool, GpuContext, GpuError, PipelineCache, StaggeredField,
};
use elements_core::graph::{NodeError, SocketType, Value};

fn gpu() -> GpuContext {
    GpuContext::new_headless().expect("no GPU adapter available")
}

#[test]
fn face_dims_add_one_cell_along_their_own_axis() {
    let cells = FieldDims::new(4, 5, 6);
    assert_eq!(StaggeredField::face_dims(cells, Axis::X), FieldDims::new(5, 5, 6));
    assert_eq!(StaggeredField::face_dims(cells, Axis::Y), FieldDims::new(4, 6, 6));
    assert_eq!(StaggeredField::face_dims(cells, Axis::Z), FieldDims::new(4, 5, 7));
}

#[test]
fn acquire_staggered_zeroed_gives_three_zeroed_faces_of_the_right_shape() {
    let ctx = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(4, 5, 6);

    let field = pool.acquire_staggered_zeroed(&ctx, &mut cache, cells).unwrap();
    assert_eq!(field.cells(), cells);
    for axis in Axis::ALL {
        let face = field.face(axis);
        assert_eq!(face.dims(), StaggeredField::face_dims(cells, axis));
        assert_eq!(face.format(), FieldFormat::R32Float);
        assert!(face.read_back(&ctx).unwrap().iter().all(|&v| v == 0.0));
    }
}

#[test]
fn from_faces_rejects_a_face_of_the_wrong_shape() {
    let ctx = gpu();
    let mut pool = FieldPool::new();
    let cells = FieldDims::new(4, 5, 6);

    // The x face is given cell dims instead of (nx + 1, ny, nz).
    let x = pool.acquire(&ctx, cells, FieldFormat::R32Float).unwrap();
    let y = pool
        .acquire(&ctx, StaggeredField::face_dims(cells, Axis::Y), FieldFormat::R32Float)
        .unwrap();
    let z = pool
        .acquire(&ctx, StaggeredField::face_dims(cells, Axis::Z), FieldFormat::R32Float)
        .unwrap();

    match StaggeredField::from_faces(cells, [x, y, z]) {
        Err(GpuError::Validation(message)) => assert!(message.contains("X"), "{message}"),
        other => panic!("expected a Validation error, got {other:?}"),
    }
}

#[test]
fn releasing_a_staggered_field_returns_all_three_faces() {
    let ctx = gpu();
    let mut pool = FieldPool::new();
    let field = pool
        .acquire_staggered_uninit(&ctx, FieldDims::new(4, 4, 4))
        .unwrap();
    assert_eq!(pool.pooled_count(), 0);
    Value::VectorField(field).release_to(&mut pool);
    assert_eq!(pool.pooled_count(), 3);
}

#[test]
fn vector_values_are_typed() {
    let ctx = gpu();
    let mut pool = FieldPool::new();
    let field = pool
        .acquire_staggered_uninit(&ctx, FieldDims::new(2, 2, 2))
        .unwrap();
    let value = Value::VectorField(field);

    assert_eq!(value.socket_type(), SocketType::VectorField);
    assert!(value.as_vector_field().is_ok());
    assert!(matches!(
        value.as_field(),
        Err(NodeError::TypeMismatch { expected: SocketType::Field, .. })
    ));
    assert!(matches!(
        Value::Scalar(1.0).as_vector_field(),
        Err(NodeError::TypeMismatch { expected: SocketType::VectorField, .. })
    ));
}
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cargo nextest run -p elements-core --test staggered`
Expected: FAIL to compile: `unresolved imports elements_core::gpu::Axis, StaggeredField`.

- [ ] **Step 3: Create `gpu/staggered.rs`**

```rust
//! Staggered (MAC) vector fields: one `R32Float` texture per velocity component.

use super::{Field, FieldDims, FieldFormat, GpuError};

/// A grid axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

/// A vector field stored on cell faces.
///
/// Component `X` lives on the faces between cells along x, so its texture has
/// `nx + 1` texels along x and matches the domain on y and z. The same holds for
/// `Y` and `Z`. Texel `(i, j, k)` of the X face sits at position
/// `(i, j + 0.5, k + 0.5)` in cell units, where cell `(0, 0, 0)` spans `[0, 1)³`.
///
/// Three `R32Float` textures rather than one `Rgba32Float`: the WebGPU baseline
/// allows `read_write` storage access only on `R32Float`, and pressure
/// projection on a staggered grid is exact where a cell-centred layout
/// checkerboards.
pub struct StaggeredField {
    faces: [Field; 3],
    cells: FieldDims,
}

impl std::fmt::Debug for StaggeredField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaggeredField")
            .field("cells", &self.cells)
            .finish()
    }
}

impl StaggeredField {
    /// The texture dims of `axis`'s face for a domain of `cells`.
    pub fn face_dims(cells: FieldDims, axis: Axis) -> FieldDims {
        match axis {
            Axis::X => FieldDims::new(cells.x + 1, cells.y, cells.z),
            Axis::Y => FieldDims::new(cells.x, cells.y + 1, cells.z),
            Axis::Z => FieldDims::new(cells.x, cells.y, cells.z + 1),
        }
    }

    /// Assemble a field from its X, Y and Z faces, in that order.
    ///
    /// Rejects any face whose dims or format are wrong. A mis-shaped face
    /// would make every sampling kernel read the wrong texels with no error.
    pub fn from_faces(cells: FieldDims, faces: [Field; 3]) -> Result<Self, GpuError> {
        for axis in Axis::ALL {
            let face = &faces[axis.index()];
            let expected = Self::face_dims(cells, axis);
            if face.dims() != expected || face.format() != FieldFormat::R32Float {
                return Err(GpuError::Validation(format!(
                    "staggered {axis:?} face is {:?} {:?}, expected {expected:?} R32Float",
                    face.dims(),
                    face.format()
                )));
            }
        }
        Ok(Self { faces, cells })
    }

    /// The domain's cell dims.
    pub fn cells(&self) -> FieldDims {
        self.cells
    }

    pub fn face(&self, axis: Axis) -> &Field {
        &self.faces[axis.index()]
    }

    /// The X, Y and Z faces, in that order.
    pub fn into_faces(self) -> [Field; 3] {
        self.faces
    }
}
```

- [ ] **Step 4: Export it**

In `crates/elements-core/src/gpu/mod.rs`, add `mod staggered;` after `mod pool;`
and `pub use staggered::{Axis, StaggeredField};` after `pub use pool::FieldPool;`.

- [ ] **Step 5: Add the pool methods**

In `crates/elements-core/src/gpu/pool.rs`, extend the `use` line with
`Axis, StaggeredField` and add inside `impl FieldPool`:

```rust
    /// Take a staggered field for a domain of `cells`, contents unspecified.
    pub fn acquire_staggered_uninit(
        &mut self,
        ctx: &GpuContext,
        cells: FieldDims,
    ) -> Result<StaggeredField, GpuError> {
        let x = self.acquire(ctx, StaggeredField::face_dims(cells, Axis::X), FieldFormat::R32Float)?;
        let y = self.acquire(ctx, StaggeredField::face_dims(cells, Axis::Y), FieldFormat::R32Float)?;
        let z = self.acquire(ctx, StaggeredField::face_dims(cells, Axis::Z), FieldFormat::R32Float)?;
        StaggeredField::from_faces(cells, [x, y, z])
    }

    /// Take a staggered field for a domain of `cells` with every face zeroed.
    pub fn acquire_staggered_zeroed(
        &mut self,
        ctx: &GpuContext,
        cache: &mut PipelineCache,
        cells: FieldDims,
    ) -> Result<StaggeredField, GpuError> {
        let x = self.acquire_zeroed(ctx, cache, StaggeredField::face_dims(cells, Axis::X))?;
        let y = self.acquire_zeroed(ctx, cache, StaggeredField::face_dims(cells, Axis::Y))?;
        let z = self.acquire_zeroed(ctx, cache, StaggeredField::face_dims(cells, Axis::Z))?;
        StaggeredField::from_faces(cells, [x, y, z])
    }

    /// Return all three faces for reuse.
    pub fn release_staggered(&mut self, field: StaggeredField) {
        for face in field.into_faces() {
            self.release(face);
        }
    }
```

- [ ] **Step 6: Add the socket type**

In `crates/elements-core/src/graph/socket.rs`, add a variant to `SocketType`:

```rust
pub enum SocketType {
    Field,
    Scalar,
    /// A staggered (face-centred) vector field. See `gpu::StaggeredField`.
    VectorField,
}
```

- [ ] **Step 7: Add the value variant and helpers**

In `crates/elements-core/src/graph/node.rs`:

Change the `use crate::gpu::...` line to
`use crate::gpu::{Field, FieldDims, FieldFormat, FieldPool, GpuContext, GpuError, PipelineCache, StaggeredField};`

Replace the `Value` enum, its `Debug` impl and its `impl Value` block (keeping
the `NodeError::with_evaluating_node` impl in between untouched) with:

```rust
pub enum Value {
    Field(Field),
    Scalar(f32),
    VectorField(StaggeredField),
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Field(field) => f.debug_tuple("Field").field(&field.dims()).finish(),
            Self::Scalar(s) => f.debug_tuple("Scalar").field(s).finish(),
            Self::VectorField(v) => f.debug_tuple("VectorField").field(&v.cells()).finish(),
        }
    }
}
```

and

```rust
impl Value {
    pub fn as_field(&self) -> Result<&Field, NodeError> {
        match self {
            Self::Field(f) => Ok(f),
            _ => Err(NodeError::TypeMismatch {
                node: NodeId(u32::MAX),
                index: 0,
                expected: SocketType::Field,
            }),
        }
    }

    pub fn as_scalar(&self) -> Result<f32, NodeError> {
        match self {
            Self::Scalar(s) => Ok(*s),
            _ => Err(NodeError::TypeMismatch {
                node: NodeId(u32::MAX),
                index: 0,
                expected: SocketType::Scalar,
            }),
        }
    }

    pub fn as_vector_field(&self) -> Result<&StaggeredField, NodeError> {
        match self {
            Self::VectorField(v) => Ok(v),
            _ => Err(NodeError::TypeMismatch {
                node: NodeId(u32::MAX),
                index: 0,
                expected: SocketType::VectorField,
            }),
        }
    }

    pub fn socket_type(&self) -> SocketType {
        match self {
            Self::Field(_) => SocketType::Field,
            Self::Scalar(_) => SocketType::Scalar,
            Self::VectorField(_) => SocketType::VectorField,
        }
    }

    /// Return every GPU texture this value owns to `pool`.
    ///
    /// Dropping a `Value` frees its textures instead, forcing the next frame to
    /// allocate again. Code that is finished with a value calls this.
    pub fn release_to(self, pool: &mut FieldPool) {
        match self {
            Self::Field(f) => pool.release(f),
            Self::VectorField(v) => pool.release_staggered(v),
            Self::Scalar(_) => {}
        }
    }
}
```

Add to `impl EvalCtx<'_>`, after `acquire_zeroed`:

```rust
    /// Acquire a staggered vector field for the current domain, contents unspecified.
    pub fn acquire_vector_uninit(&mut self) -> Result<StaggeredField, NodeError> {
        let dims = self.dims;
        Ok(self.pool.acquire_staggered_uninit(self.gpu, dims)?)
    }

    /// Acquire a staggered vector field for the current domain with every face zeroed.
    pub fn acquire_vector_zeroed(&mut self) -> Result<StaggeredField, NodeError> {
        let dims = self.dims;
        Ok(self.pool.acquire_staggered_zeroed(self.gpu, self.pipelines, dims)?)
    }
```

- [ ] **Step 8: Run the tests and watch them pass**

Run: `cargo nextest run -p elements-core`
Expected: PASS, all tests. If any other file matches `Value` exhaustively, the
compiler names it. Add a `Value::VectorField(_)` arm that returns the same type-mismatch error its other non-matching arms return.

- [ ] **Step 9: Prove the tests can fail**

- Mutation A: in `face_dims`, change `Axis::Y => FieldDims::new(cells.x, cells.y + 1, cells.z)` to
  `Axis::Y => FieldDims::new(cells.x, cells.y, cells.z + 1)`. Run
  `cargo nextest run -p elements-core --test staggered face_dims` → FAIL. Restore.
- Mutation B: in `from_faces`, replace the `if ... { return Err(...) }` condition with `if false`.
  Run `... --test staggered from_faces` → FAIL (`expected a Validation error, got Ok`). Restore.
- Mutation C: in `release_to`, change the `VectorField` arm to `Self::VectorField(_) => {}`.
  Run `... --test staggered releasing` → FAIL (`left: 0, right: 3`). Restore.

- [ ] **Step 10: `just check`, then commit**

```bash
just check
git add crates/elements-core/src/gpu/staggered.rs crates/elements-core/src/gpu/mod.rs \
        crates/elements-core/src/gpu/pool.rs crates/elements-core/src/graph/socket.rs \
        crates/elements-core/src/graph/node.rs crates/elements-core/tests/staggered.rs
git commit -F - <<'EOF'
Add staggered vector fields as a graph value type

Ember stores velocity on a MAC grid: pressure projection is exact there,
where a cell-centred layout checkerboards. Three R32Float face textures,
rather than one Rgba32Float, is what keeps read_write storage access
within the WebGPU baseline. The face shape is validated on construction
because a mis-shaped face would make every sampling kernel read the
wrong texels without any error.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XCzEwqREbRMwabH11TvnMb
EOF
```

---
## Task 4: Time, persistent state, stateful nodes, and `core.accumulate`

**Why:** a solver's fields must outlive one evaluation, and nodes must know which frame they are computing. `core.accumulate` (`sum += input * dt`) is the proof: its output at any frame has a closed form, which is cheap to check on the CPU.

**Files:**
- Create: `crates/elements-core/src/graph/time.rs`
- Create: `crates/elements-core/src/graph/state.rs`
- Create: `crates/elements-core/src/gpu/shaders/accumulate.wgsl`
- Create: `crates/elements-core/src/nodes/accumulate.rs`
- Modify: `crates/elements-core/src/gpu/field.rs` (`Field::copy_to`)
- Modify: `crates/elements-core/src/gpu/pool.rs` (`FieldPool::duplicate`)
- Modify: `crates/elements-core/src/gpu/dispatch.rs` and `gpu/mod.rs` (`accumulate_into`)
- Modify: `crates/elements-core/src/graph/node.rs`, `crates/elements-core/src/graph/mod.rs`
- Modify: `crates/elements-core/src/nodes/mod.rs`, `crates/elements-core/src/nodes/output.rs` (its unit test)
- Test: `crates/elements-core/tests/state.rs` (create), `crates/elements-core/tests/nodes.rs`

**Interfaces:**
- Consumes: `EvalCtx::acquire_zeroed` (Task 2), `Value::release_to` (Task 3).
- Produces:
  - `graph::Time { pub frame: u32, pub seconds: f64, pub dt: f64 }` and `Time::at(frame, start_frame, fps) -> Time`
  - `graph::DEFAULT_FPS: f64 = 24.0` and `graph::DEFAULT_START_FRAME: u32 = 1`
  - `graph::StateStore` with `new()`, `len()`, `is_empty()`, `clear(&mut self, pool: &mut FieldPool)`, and crate-private `take`/`put`
  - `Node::stateful(&self) -> bool` (default `false`)
  - `EvalCtx::time() -> Time`
  - `EvalCtx::take_state(&mut self, slot: &'static str) -> Result<Option<Value>, NodeError>`
  - `EvalCtx::put_state(&mut self, slot: &'static str, value: Value) -> Result<(), NodeError>`
  - `EvalCtx::release(&mut self, value: Value)` and `EvalCtx::duplicate(&mut self, field: &Field) -> Result<Field, NodeError>`
  - `NodeError::NotStateful { node }` and `NodeError::StateShape { node, slot: &'static str }`
  - `graph::Evaluated { pub value: Value }` (Task 5 adds `stats`)
  - `Graph::eval_frame(&self, gpu, pool, pipelines, state: &mut StateStore, time: Time, dims) -> Result<Evaluated, NodeError>`
  - `Graph::eval(...)` keeps its signature. It evaluates the first frame at the default rate against a scratch store that it throws away afterwards.
  - `Graph::is_stateful(&self) -> bool`
  - `Field::copy_to(&self, ctx, dst: &Field) -> Result<(), GpuError>` and `FieldPool::duplicate(&mut self, ctx, src: &Field) -> Result<Field, GpuError>`
  - `gpu::accumulate_into(ctx, cache, sum: &Field, input: &Field, dt: f32) -> Result<(), GpuError>`
  - The node kind `"core.accumulate"`: one `Field` in, one `Field` out, no params, one state slot `"sum"`.

- [ ] **Step 1: Write the failing tests**

Create `crates/elements-core/tests/state.rs`:

```rust
use elements_core::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{
    Document, EvalCtx, Graph, Node, NodeError, NodeRegistry, SocketSpec, SocketType, StateStore,
    Time, Value,
};

/// constant 0.5 -> accumulate -> output, on a 4³ domain.
const ACCUMULATE_DOC: &str = r#"{
  "version": 1,
  "dims": [4, 4, 4],
  "nodes": [
    { "id": 0, "kind": "core.constant_field", "params": { "value": 0.5 } },
    { "id": 1, "kind": "core.accumulate", "params": {} },
    { "id": 2, "kind": "core.output", "params": {} }
  ],
  "edges": [
    { "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 },
    { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 0 }
  ],
  "output": 2
}"#;

struct Harness {
    gpu: GpuContext,
    pool: FieldPool,
    pipelines: PipelineCache,
}

impl Harness {
    fn new() -> Self {
        Self {
            gpu: GpuContext::new_headless().expect("no GPU adapter available"),
            pool: FieldPool::new(),
            pipelines: PipelineCache::new(),
        }
    }

    /// Evaluate `frame` against `state` and read the output back. The output
    /// field is released to the pool, so repeated frames do not reallocate.
    fn frame(
        &mut self,
        graph: &Graph,
        state: &mut StateStore,
        frame: u32,
        dims: FieldDims,
    ) -> Vec<f32> {
        let evaluated = graph
            .eval_frame(
                &self.gpu,
                &mut self.pool,
                &mut self.pipelines,
                state,
                Time::at(frame, 1, 24.0),
                dims,
            )
            .unwrap();
        let values = evaluated.value.as_field().unwrap().read_back(&self.gpu).unwrap();
        evaluated.value.release_to(&mut self.pool);
        values
    }
}

fn accumulate_graph() -> (Graph, FieldDims) {
    Document::from_json(ACCUMULATE_DOC)
        .unwrap()
        .into_graph(&NodeRegistry::with_builtins())
        .unwrap()
}

/// The closed form for ACCUMULATE_DOC at `frame`, with start frame 1 at 24 fps.
fn expected(frame: u32) -> f32 {
    0.5 * frame as f32 / 24.0
}

fn assert_all_near(values: &[f32], expected: f32) {
    for &v in values {
        assert!((v - expected).abs() < 1e-6, "expected {expected}, got {v}");
    }
}

#[test]
fn time_reports_seconds_since_the_start_frame() {
    let t = Time::at(25, 1, 24.0);
    assert_eq!(t.frame, 25);
    assert_eq!(t.seconds, 1.0);
    assert_eq!(t.dt, 1.0 / 24.0);
    assert_eq!(Time::at(0, 1, 24.0).seconds, 0.0);
}

#[test]
fn accumulate_integrates_its_input_over_frames() {
    let mut h = Harness::new();
    let (graph, dims) = accumulate_graph();
    let mut state = StateStore::new();
    for frame in 1..=5 {
        let values = h.frame(&graph, &mut state, frame, dims);
        assert_all_near(&values, expected(frame));
    }
    assert_eq!(state.len(), 1, "one slot: accumulate's sum");
}

#[test]
fn plain_eval_of_a_stateful_graph_is_always_its_first_frame() {
    let mut h = Harness::new();
    let (graph, dims) = accumulate_graph();
    assert!(graph.is_stateful());
    for _ in 0..3 {
        let value = graph
            .eval(&h.gpu, &mut h.pool, &mut h.pipelines, dims)
            .unwrap();
        assert_all_near(
            &value.as_field().unwrap().read_back(&h.gpu).unwrap(),
            expected(1),
        );
        value.release_to(&mut h.pool);
    }
}

/// A node that does not declare itself stateful, but reaches for state anyway.
struct Sneaky;

impl Node for Sneaky {
    fn kind(&self) -> &'static str {
        "test.sneaky"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Scalar],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        ctx.take_state("anything")?;
        Ok(vec![Value::Scalar(0.0)])
    }
}

#[test]
fn a_stateless_node_cannot_touch_state() {
    let mut h = Harness::new();
    let mut graph = Graph::new();
    let id = graph.add_node(Box::new(Sneaky));
    graph.set_output(id);
    let mut state = StateStore::new();
    let err = graph
        .eval_frame(
            &h.gpu,
            &mut h.pool,
            &mut h.pipelines,
            &mut state,
            Time::at(1, 1, 24.0),
            FieldDims::new(4, 4, 4),
        )
        .unwrap_err();
    assert!(matches!(err, NodeError::NotStateful { .. }), "got {err:?}");
}

#[test]
fn state_of_the_wrong_shape_is_reported() {
    let mut h = Harness::new();
    let (graph, dims) = accumulate_graph();
    let mut state = StateStore::new();
    h.frame(&graph, &mut state, 1, dims);

    let err = graph
        .eval_frame(
            &h.gpu,
            &mut h.pool,
            &mut h.pipelines,
            &mut state,
            Time::at(2, 1, 24.0),
            FieldDims::new(8, 8, 8),
        )
        .unwrap_err();
    assert!(
        matches!(err, NodeError::StateShape { slot: "sum", .. }),
        "got {err:?}"
    );
}

/// A stateless node that outputs the time it was evaluated at.
struct Clock;

impl Node for Clock {
    fn kind(&self) -> &'static str {
        "test.clock"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Scalar],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        Ok(vec![Value::Scalar(ctx.time().seconds as f32)])
    }
}

#[test]
fn time_reaches_nodes() {
    let mut h = Harness::new();
    let mut graph = Graph::new();
    let id = graph.add_node(Box::new(Clock));
    graph.set_output(id);
    let mut state = StateStore::new();
    let evaluated = graph
        .eval_frame(
            &h.gpu,
            &mut h.pool,
            &mut h.pipelines,
            &mut state,
            Time::at(49, 1, 24.0),
            FieldDims::new(1, 1, 1),
        )
        .unwrap();
    assert_eq!(evaluated.value.as_scalar().unwrap(), 2.0);
}
```

In `crates/elements-core/tests/nodes.rs`, rename `registry_exposes_all_three_builtins` to
`registry_exposes_every_builtin` and change its expected list to
`["core.accumulate", "core.constant_field", "core.noise_field", "core.output"]`.

- [ ] **Step 2: Run them and watch them fail**

Run: `cargo nextest run -p elements-core --test state --test nodes`
Expected: FAIL to compile: `unresolved imports ... StateStore, Time`.

- [ ] **Step 3: Create `graph/time.rs`**

```rust
//! Simulation time as nodes see it.

/// The frame rate a document gets when it does not specify one.
pub const DEFAULT_FPS: f64 = 24.0;

/// The first frame of a timeline when a document does not specify one.
/// Matches Blender's default scene start.
pub const DEFAULT_START_FRAME: u32 = 1;

/// When a node is being evaluated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Time {
    /// The frame being produced.
    pub frame: u32,
    /// Seconds elapsed since the timeline's start frame.
    pub seconds: f64,
    /// Seconds per frame. Substepping is a solver's own business.
    pub dt: f64,
}

impl Time {
    /// Time at `frame` on a timeline that starts at `start_frame` and runs at `fps`.
    ///
    /// Frames before `start_frame` report zero seconds. The timeline clamps
    /// them to the start frame anyway.
    pub fn at(frame: u32, start_frame: u32, fps: f64) -> Self {
        Self {
            frame,
            seconds: frame.saturating_sub(start_frame) as f64 / fps,
            dt: 1.0 / fps,
        }
    }
}
```

- [ ] **Step 4: Create `graph/state.rs`**

```rust
//! Per-node state that survives from one frame to the next.

use std::collections::BTreeMap;

use crate::gpu::FieldPool;

use super::node::Value;
use super::socket::NodeId;

/// Fields and scalars that outlive a single `Graph::eval_frame`.
///
/// Keyed by `(NodeId, slot)`. A `BTreeMap` rather than a `HashMap` so iteration
/// order, and therefore snapshot order, is deterministic.
///
/// Nodes reach their own slots through `EvalCtx::take_state` / `put_state`.
/// A node takes a value out, works on it, and puts it back, which keeps the
/// borrow checker out of the way while it also acquires pool fields.
#[derive(Default)]
pub struct StateStore {
    pub(crate) slots: BTreeMap<(NodeId, &'static str), Value>,
}

impl StateStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of occupied slots.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Empty every slot, returning its textures to `pool`.
    pub fn clear(&mut self, pool: &mut FieldPool) {
        for (_, value) in std::mem::take(&mut self.slots) {
            value.release_to(pool);
        }
    }

    pub(crate) fn take(&mut self, node: NodeId, slot: &'static str) -> Option<Value> {
        self.slots.remove(&(node, slot))
    }

    pub(crate) fn put(
        &mut self,
        node: NodeId,
        slot: &'static str,
        value: Value,
        pool: &mut FieldPool,
    ) {
        if let Some(old) = self.slots.insert((node, slot), value) {
            old.release_to(pool);
        }
    }
}
```

- [ ] **Step 5: Add GPU copying**

In `crates/elements-core/src/gpu/field.rs`, add inside `impl Field`:

```rust
    /// Copy this field's contents into `dst` on the GPU.
    ///
    /// Both fields must have identical dims and format.
    /// `copy_texture_to_texture` is core WebGPU, not an optional feature.
    pub fn copy_to(&self, ctx: &GpuContext, dst: &Field) -> Result<(), GpuError> {
        if self.dims != dst.dims || self.format != dst.format {
            return Err(GpuError::Validation(format!(
                "copy_to: source is {:?} {:?}, destination is {:?} {:?}",
                self.dims, self.format, dst.dims, dst.format
            )));
        }
        ctx.scoped(|| {
            let mut encoder =
                ctx.device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("field-copy"),
                    });
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &dst.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                self.dims.extent(),
            );
            ctx.queue().submit(Some(encoder.finish()));
        })
    }
```

In `crates/elements-core/src/gpu/pool.rs`, add inside `impl FieldPool`:

```rust
    /// A pooled field with the same shape as `src`, holding a GPU copy of its contents.
    pub fn duplicate(&mut self, ctx: &GpuContext, src: &Field) -> Result<Field, GpuError> {
        let dst = self.acquire(ctx, src.dims(), src.format())?;
        src.copy_to(ctx, &dst)?;
        Ok(dst)
    }
```

- [ ] **Step 6: Add the accumulate kernel**

Create `crates/elements-core/src/gpu/shaders/accumulate.wgsl`:

```wgsl
// sum += input * dt, in place.
//
// `sum` is read_write storage, which the WebGPU baseline allows for r32float.
// `input` is read through textureLoad: R32Float is not filterable without an
// optional feature, and no sampler is needed for a same-texel read.

struct Params {
    dims: vec3<u32>,
    dt: f32,
};

@group(0) @binding(0) var sum: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var input: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.dims.x || gid.y >= params.dims.y || gid.z >= params.dims.z) {
        return;
    }
    let p = vec3<i32>(gid);
    let next = textureLoad(sum, p).x + textureLoad(input, p, 0).x * params.dt;
    textureStore(sum, p, vec4<f32>(next, 0.0, 0.0, 0.0));
}
```

Append to `crates/elements-core/src/gpu/dispatch.rs`:

```rust
/// Parameters for the accumulate shader: `vec3<u32>` then `f32` pack into 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AccumulateParams {
    dims: [u32; 3],
    dt: f32,
}

/// `sum += input * dt`, in place on the GPU.
pub fn accumulate_into(
    ctx: &GpuContext,
    cache: &mut PipelineCache,
    sum: &Field,
    input: &Field,
    dt: f32,
) -> Result<(), GpuError> {
    if sum.dims() != input.dims() {
        return Err(GpuError::Validation(format!(
            "accumulate_into: sum is {:?}, input is {:?}",
            sum.dims(),
            input.dims()
        )));
    }
    let pipeline = cache.get_or_create(
        ctx,
        "accumulate",
        include_str!("shaders/accumulate.wgsl"),
        "main",
    )?;

    let dims = sum.dims();
    let params = AccumulateParams {
        dims: [dims.x, dims.y, dims.z],
        dt,
    };

    let bind_group = ctx.scoped(|| {
        let uniform = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("accumulate-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("accumulate-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(sum.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(input.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        })
    })?;

    dispatch_over_field(ctx, &pipeline, &bind_group, dims)
}
```

In `crates/elements-core/src/gpu/mod.rs`, change the dispatch re-export to:
`pub use dispatch::{PipelineCache, WORKGROUP, accumulate_into, dispatch_over_field, fill_constant, fill_curl_noise};`

- [ ] **Step 7: Give nodes state and time**

In `crates/elements-core/src/graph/node.rs`:

Add imports: `use super::state::StateStore;` and `use super::time::Time;`.

Add two variants to `NodeError`, just before `Gpu`:

```rust
    #[error("node {node:?} is not stateful but tried to use persistent state")]
    NotStateful { node: NodeId },
    #[error("node {node:?} state slot {slot:?} does not match the current domain")]
    StateShape { node: NodeId, slot: &'static str },
```

Add three fields to the end of `EvalCtx`:

```rust
    /// Persistent state. Only a node whose `stateful()` is true may touch it.
    pub(crate) state: &'a mut StateStore,
    pub(crate) stateful: bool,
    pub(crate) time: Time,
```

Add to `impl EvalCtx<'_>`:

```rust
    /// When this evaluation is happening.
    pub fn time(&self) -> Time {
        self.time
    }

    /// Take this node's state `slot` out of the store, or `None` on the first
    /// step or after a reset. Put it back with `put_state` before returning.
    ///
    /// A stored field whose dims no longer match the domain is released and
    /// reported as `StateShape`. The timeline answers that with a reset.
    pub fn take_state(&mut self, slot: &'static str) -> Result<Option<Value>, NodeError> {
        if !self.stateful {
            return Err(NodeError::NotStateful { node: self.node });
        }
        let Some(value) = self.state.take(self.node, slot) else {
            return Ok(None);
        };
        let fits = match &value {
            Value::Field(f) => f.dims() == self.dims,
            Value::VectorField(v) => v.cells() == self.dims,
            Value::Scalar(_) => true,
        };
        if !fits {
            value.release_to(self.pool);
            return Err(NodeError::StateShape {
                node: self.node,
                slot,
            });
        }
        Ok(Some(value))
    }

    /// Store `value` in this node's state `slot`, releasing anything it replaces.
    pub fn put_state(&mut self, slot: &'static str, value: Value) -> Result<(), NodeError> {
        if !self.stateful {
            return Err(NodeError::NotStateful { node: self.node });
        }
        self.state.put(self.node, slot, value, self.pool);
        Ok(())
    }

    /// Return a value this node is finished with to the pool.
    pub fn release(&mut self, value: Value) {
        value.release_to(self.pool);
    }

    /// A pooled GPU copy of `field`.
    pub fn duplicate(&mut self, field: &Field) -> Result<Field, NodeError> {
        Ok(self.pool.duplicate(self.gpu, field)?)
    }
```

Add a provided method to `trait Node`, after `sockets`:

```rust
    /// Whether this node keeps state between frames. Only stateful nodes may
    /// call `EvalCtx::take_state` / `put_state`, and a graph containing one is
    /// stepped frame by frame by a `Timeline`.
    fn stateful(&self) -> bool {
        false
    }
```

- [ ] **Step 8: Evaluate frames against a store**

In `crates/elements-core/src/graph/mod.rs`:

Add modules and exports:

```rust
mod state;
mod time;

pub use state::StateStore;
pub use time::{DEFAULT_FPS, DEFAULT_START_FRAME, Time};
```

Add, above `impl Graph`:

```rust
/// The result of evaluating one frame.
#[derive(Debug)]
pub struct Evaluated {
    /// The output node's first output. The caller owns it and should return it
    /// to the pool with `Value::release_to` when finished.
    pub value: Value,
}
```

Add inside `impl Graph`:

```rust
    /// Whether any node in this graph keeps state between frames.
    pub fn is_stateful(&self) -> bool {
        self.nodes.iter().any(|node| node.stateful())
    }
```

Replace the whole `eval` method with these two:

```rust
    /// Evaluate one frame with no persistent state: the first frame, at the default rate.
    ///
    /// Stateful nodes see an empty store, which is discarded afterwards, so
    /// repeated calls never accumulate. Anything that needs frames to follow
    /// one another goes through `eval_frame`, normally via a `Timeline`.
    pub fn eval(
        &self,
        gpu: &GpuContext,
        pool: &mut FieldPool,
        pipelines: &mut PipelineCache,
        dims: FieldDims,
    ) -> Result<Value, NodeError> {
        let mut scratch = StateStore::new();
        let time = Time::at(DEFAULT_START_FRAME, DEFAULT_START_FRAME, DEFAULT_FPS);
        let result = self.eval_frame(gpu, pool, pipelines, &mut scratch, time, dims);
        scratch.clear(pool);
        result.map(|evaluated| evaluated.value)
    }

    /// Evaluate the output node and everything it depends on, as frame `time`,
    /// reading and writing persistent state in `state`.
    pub fn eval_frame(
        &self,
        gpu: &GpuContext,
        pool: &mut FieldPool,
        pipelines: &mut PipelineCache,
        state: &mut StateStore,
        time: Time,
        dims: FieldDims,
    ) -> Result<Evaluated, NodeError> {
        let output = self.output.ok_or(NodeError::NoOutput)?;
        let order = self.evaluation_order()?;

        // Outputs of each evaluated node. Values are moved out as they are
        // consumed, so each slot holds `None` once its consumer has run.
        let mut produced: HashMap<NodeId, Vec<Option<Value>>> = HashMap::new();

        for id in order {
            let node = self.node(id)?;
            let spec = node.sockets();

            let mut inputs: Vec<Option<Value>> = Vec::with_capacity(spec.inputs.len());
            for index in 0..spec.inputs.len() as u32 {
                let value = match self.edges.get(&SocketId { node: id, index }) {
                    Some(src) => produced
                        .get_mut(&src.node)
                        .and_then(|outs| outs.get_mut(src.index as usize))
                        .and_then(Option::take),
                    None => None,
                };
                inputs.push(value);
            }

            let taken = vec![false; inputs.len()];
            let mut ctx = EvalCtx {
                gpu,
                pool,
                pipelines,
                dims,
                node: id,
                inputs,
                taken,
                state,
                stateful: node.stateful(),
                time,
            };
            let outputs = node
                .eval(&mut ctx)
                .map_err(|err| err.with_evaluating_node(id))?;
            produced.insert(id, outputs.into_iter().map(Some).collect());
        }

        produced
            .get_mut(&output)
            .and_then(|outs| outs.first_mut())
            .and_then(Option::take)
            .map(|value| Evaluated { value })
            .ok_or(NodeError::UnknownNode(output))
    }
```

(Task 5 replaces this loop. It is kept close to Core v1's here so this task's diff is about state and time only.)

- [ ] **Step 9: Create the node**

Create `crates/elements-core/src/nodes/accumulate.rs`:

```rust
//! Integrates its input over time: `sum += input * dt` every frame.
//!
//! The simplest possible stateful node, and the proof that state, time and the
//! timeline work. Its output at frame N has a closed form a test can check:
//! for a constant input `c`, `c * (N - start_frame + 1) / fps`.

use crate::gpu::accumulate_into;
use crate::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};

pub const KIND: &str = "core.accumulate";

const SUM: &str = "sum";

#[derive(Debug, Clone, Default)]
pub struct Accumulate;

impl Node for Accumulate {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field],
            outputs: vec![SocketType::Field],
        }
    }

    fn stateful(&self) -> bool {
        true
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let sum = match ctx.take_state(SUM)? {
            Some(Value::Field(field)) => field,
            Some(other) => {
                ctx.release(other);
                return Err(NodeError::StateShape {
                    node: ctx.node_id(),
                    slot: SUM,
                });
            }
            // The first step starts from zero, so the store must hand back a
            // cleared field, not a recycled one.
            None => ctx.acquire_zeroed()?,
        };

        let input = ctx.take_input(0)?;
        let dt = ctx.time().dt as f32;
        let result = match input.as_field() {
            Ok(field) => ctx.with_gpu(|gpu, cache| accumulate_into(gpu, cache, &sum, field, dt)),
            Err(_) => Err(NodeError::TypeMismatch {
                node: ctx.node_id(),
                index: 0,
                expected: SocketType::Field,
            }),
        };
        ctx.release(input);
        if let Err(e) = result {
            ctx.release(Value::Field(sum));
            return Err(e);
        }

        // The output is a copy: the sum itself stays in the store for the next frame.
        let out = ctx.duplicate(&sum)?;
        ctx.put_state(SUM, Value::Field(sum))?;
        Ok(vec![Value::Field(out)])
    }
}

pub(crate) fn build(_params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    Ok(Box::new(Accumulate))
}
```

In `crates/elements-core/src/nodes/mod.rs`, add `pub mod accumulate;` and
`pub use accumulate::Accumulate;`, and register it first in `register_builtins`:
`registry.register(accumulate::KIND, accumulate::build);`.

- [ ] **Step 10: Fix the `Output` unit test's hand-built context**

In `crates/elements-core/src/nodes/output.rs`'s test module, add
`use crate::graph::{StateStore, Time};`, create `let mut state = StateStore::new();`
next to `pipelines`, and add three fields to the `EvalCtx { ... }` literal:

```rust
            state: &mut state,
            stateful: false,
            time: Time::at(1, 1, 24.0),
```

- [ ] **Step 11: Run the tests and watch them pass**

Run: `cargo nextest run -p elements-core`
Expected: PASS, all tests.

- [ ] **Step 12: Prove the tests can fail**

Apply each mutation, run `cargo nextest run -p elements-core --test state`, confirm the named test fails, and restore.

- A: in `Accumulate::eval`, replace `ctx.put_state(SUM, Value::Field(sum))?;` with
  `ctx.release(Value::Field(sum));` → `accumulate_integrates_its_input_over_frames` fails at frame 2.
- B: in `take_state`, delete the `if !self.stateful { ... }` block → `a_stateless_node_cannot_touch_state` fails.
- C: in `take_state`, change the `let fits = match ...;` statement to `let fits = true;` →
  `state_of_the_wrong_shape_is_reported` fails with a `Gpu(Validation(... accumulate_into ...))`
  error instead of `StateShape`.
- D: in `Graph::eval_frame`, change `time,` in the `EvalCtx` literal to
  `time: Time::at(1, 1, 24.0),` → `time_reaches_nodes` fails (`0.0` vs `2.0`).

Separately, note in the report that `scratch.clear(pool)` in `Graph::eval` has no
test that fails without it: it only affects pool reuse, and dropping the store frees the
textures either way. Do not invent a test for it.

- [ ] **Step 13: `just check`, then commit**

```bash
just check
git add crates/elements-core/src crates/elements-core/tests/state.rs crates/elements-core/tests/nodes.rs
git commit -F - <<'EOF'
Give nodes persistent state and time, proven by core.accumulate

A solver's fields must outlive one evaluation, and emitters must know
which frame they are drawing. State lives in a store beside the pool,
keyed by node and slot, and only nodes that declare themselves stateful
may touch it, so an accidental dependency on the previous frame is an
error rather than a subtle bug. core.accumulate has a closed-form output,
which lets later tasks check the timeline exactly.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XCzEwqREbRMwabH11TvnMb
EOF
```

---

## Task 5: Several consumers per output, and release at last use

**Why:** Ember's graphs branch. A velocity field feeds both advection and
vorticity confinement. Core v1 allowed one consumer per output so it could move
values. That constraint also hid a leak: values nobody consumed were *dropped*,
which frees their textures, rather than being returned to the pool.

**Files:**
- Modify: `crates/elements-core/src/graph/mod.rs` (evaluator rewrite, `connect`)
- Modify: `crates/elements-core/src/graph/node.rs` (`EvalCtx` inputs, `EvalStats`, `Value::duplicate`, `NodeError`)
- Modify: `crates/elements-core/src/gpu/pool.rs` (`allocation_count`)
- Modify: `crates/elements-core/src/nodes/output.rs` (its unit test)
- Modify: `crates/elements-core/tests/graph.rs`, `crates/elements-core/tests/document.rs`
- Test: `crates/elements-core/tests/fan_out.rs` (create)

**Interfaces:**
- Consumes: `FieldPool::duplicate` (Task 4), `Value::release_to` (Task 3), `StaggeredField::from_faces` (Task 3).
- Produces:
  - `graph::EvalStats { pub copies: u32 }` (`Default`, `PartialEq`)
  - `Evaluated { pub value: Value, pub stats: EvalStats }`
  - `Value::duplicate(&self, gpu: &GpuContext, pool: &mut FieldPool) -> Result<Value, GpuError>`
  - `FieldPool::allocation_count(&self) -> u64`
  - `NodeError::InputAlreadyConnected { node, index }` replaces the deleted `NodeError::AlreadyConsumed`
  - Semantics: `EvalCtx::input` borrows. `EvalCtx::take_input` moves if this is
    the value's last outstanding use, and otherwise returns a GPU copy counted in
    `stats.copies`. Scalars are copied but not counted. Every field is returned
    to the pool after its last use, whether evaluation succeeds or fails.

- [ ] **Step 1: Write the failing tests**

Create `crates/elements-core/tests/fan_out.rs`:

```rust
use elements_core::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache, fill_constant};
use elements_core::graph::{
    EvalCtx, Evaluated, Graph, Node, NodeError, NodeId, SocketId, SocketSpec, SocketType,
    StateStore, Time, Value,
};
use elements_core::nodes::{ConstantField, Output};

struct Literal(f32);

impl Node for Literal {
    fn kind(&self) -> &'static str {
        "test.literal"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Scalar],
        }
    }
    fn eval(&self, _ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        Ok(vec![Value::Scalar(self.0)])
    }
}

/// Reads both inputs by reference, so it never forces a copy.
struct Add;

impl Node for Add {
    fn kind(&self) -> &'static str {
        "test.add"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Scalar, SocketType::Scalar],
            outputs: vec![SocketType::Scalar],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let a = ctx.input(0)?.as_scalar()?;
        let b = ctx.input(1)?.as_scalar()?;
        Ok(vec![Value::Scalar(a + b)])
    }
}

/// Takes its field and overwrites it. If it were handed a texture another
/// branch also holds, that branch would see this node's value.
struct Overwrite(f32);

impl Node for Overwrite {
    fn kind(&self) -> &'static str {
        "test.overwrite"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field],
            outputs: vec![SocketType::Field],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let field = match ctx.take_input(0)? {
            Value::Field(field) => field,
            other => {
                ctx.release(other);
                return Err(NodeError::TypeMismatch {
                    node: ctx.node_id(),
                    index: 0,
                    expected: SocketType::Field,
                });
            }
        };
        let value = self.0;
        ctx.with_gpu(|gpu, cache| fill_constant(gpu, cache, &field, value))?;
        Ok(vec![Value::Field(field)])
    }
}

/// Outputs its second input. Its first input is only borrowed, never read,
/// so the evaluator must release it.
struct PickSecond;

impl Node for PickSecond {
    fn kind(&self) -> &'static str {
        "test.pick_second"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field, SocketType::Field],
            outputs: vec![SocketType::Field],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        Ok(vec![ctx.take_input(1)?])
    }
}

fn link(g: &mut Graph, from: NodeId, from_index: u32, to: NodeId, to_index: u32) {
    g.connect(
        SocketId {
            node: from,
            index: from_index,
        },
        SocketId {
            node: to,
            index: to_index,
        },
    )
    .unwrap();
}

/// constant(0.5) feeds BOTH Overwrite(9) and a pass-through. PickSecond outputs
/// the pass-through's value, which must still be 0.5.
fn branching_graph() -> Graph {
    let mut g = Graph::new();
    let source = g.add_node(Box::new(ConstantField { value: 0.5 }));
    let overwrite = g.add_node(Box::new(Overwrite(9.0)));
    let keep = g.add_node(Box::new(Output));
    let pick = g.add_node(Box::new(PickSecond));
    let out = g.add_node(Box::new(Output));
    link(&mut g, source, 0, overwrite, 0);
    link(&mut g, source, 0, keep, 0);
    link(&mut g, overwrite, 0, pick, 0);
    link(&mut g, keep, 0, pick, 1);
    link(&mut g, pick, 0, out, 0);
    g.set_output(out);
    g
}

struct Harness {
    gpu: GpuContext,
    pool: FieldPool,
    pipelines: PipelineCache,
}

impl Harness {
    fn new() -> Self {
        Self {
            gpu: GpuContext::new_headless().expect("no GPU adapter available"),
            pool: FieldPool::new(),
            pipelines: PipelineCache::new(),
        }
    }

    fn eval(&mut self, graph: &Graph) -> Evaluated {
        let mut state = StateStore::new();
        graph
            .eval_frame(
                &self.gpu,
                &mut self.pool,
                &mut self.pipelines,
                &mut state,
                Time::at(1, 1, 24.0),
                FieldDims::new(4, 4, 4),
            )
            .unwrap()
    }
}

#[test]
fn a_field_feeding_two_consumers_is_copied_for_the_first() {
    let mut h = Harness::new();
    let evaluated = h.eval(&branching_graph());
    let values = evaluated.value.as_field().unwrap().read_back(&h.gpu).unwrap();
    assert!(
        values.iter().all(|&v| v == 0.5),
        "the untouched branch saw {values:?}"
    );
    assert_eq!(evaluated.stats.copies, 1);
}

#[test]
fn a_chain_with_no_branching_copies_nothing() {
    let mut h = Harness::new();
    let mut g = Graph::new();
    let source = g.add_node(Box::new(ConstantField { value: 0.5 }));
    let out = g.add_node(Box::new(Output));
    link(&mut g, source, 0, out, 0);
    g.set_output(out);
    assert_eq!(h.eval(&g).stats.copies, 0);
}

#[test]
fn a_scalar_can_feed_both_inputs_of_one_node() {
    let mut h = Harness::new();
    let mut g = Graph::new();
    let two = g.add_node(Box::new(Literal(2.0)));
    let sum = g.add_node(Box::new(Add));
    link(&mut g, two, 0, sum, 0);
    link(&mut g, two, 0, sum, 1);
    g.set_output(sum);
    let evaluated = h.eval(&g);
    assert_eq!(evaluated.value.as_scalar().unwrap(), 4.0);
    assert_eq!(evaluated.stats.copies, 0);
}

#[test]
fn fields_return_to_the_pool_after_their_last_use() {
    let mut h = Harness::new();
    let graph = branching_graph();

    let first = h.eval(&graph);
    first.value.release_to(&mut h.pool);
    let after_first = h.pool.allocation_count();

    for _ in 0..3 {
        let again = h.eval(&graph);
        again.value.release_to(&mut h.pool);
    }
    assert_eq!(
        h.pool.allocation_count(),
        after_first,
        "every frame after the first must be served entirely from the pool"
    );
}

#[test]
fn an_input_accepts_only_one_edge() {
    let mut g = Graph::new();
    let a = g.add_node(Box::new(Literal(1.0)));
    let b = g.add_node(Box::new(Literal(2.0)));
    let sum = g.add_node(Box::new(Add));
    link(&mut g, a, 0, sum, 0);
    let err = g
        .connect(
            SocketId { node: b, index: 0 },
            SocketId {
                node: sum,
                index: 0,
            },
        )
        .unwrap_err();
    assert!(
        matches!(err, NodeError::InputAlreadyConnected { index: 0, .. }),
        "got {err:?}"
    );
}
```

In `crates/elements-core/tests/graph.rs`, **delete** the test
`rejects_a_second_consumer_of_one_output`. Its rule no longer exists, and `fan_out.rs` covers the new behaviour.

In `crates/elements-core/tests/document.rs`, rename
`rejects_a_document_wiring_one_output_to_two_inputs` to
`accepts_a_document_wiring_one_output_to_two_inputs`, and replace its final `match` with:

```rust
    let registry = test_registry();
    assert!(
        doc.into_graph(&registry).is_ok(),
        "one output may now feed several inputs"
    );
```

Also update the doc comments on `Producer` and `Consumer` in that file, which
describe them as "competing consumers". They are now "a producer whose single
output feeds both inputs of one consumer".

- [ ] **Step 2: Run them and watch them fail**

Run: `cargo nextest run -p elements-core --test fan_out`
Expected: FAIL to compile (`no field stats`, `no method allocation_count`, `no variant InputAlreadyConnected`).

- [ ] **Step 3: `allocation_count` and `Value::duplicate`**

In `crates/elements-core/src/gpu/pool.rs`, add inside `impl FieldPool`:

```rust
    /// How many fresh GPU textures this pool has ever allocated. It never
    /// decreases, so a stable count across frames proves reuse.
    pub fn allocation_count(&self) -> u64 {
        self.next_generation
    }
```

In `crates/elements-core/src/graph/node.rs`, add `Axis` to the `crate::gpu` import, and add inside `impl Value`:

```rust
    /// A copy of this value, with GPU contents copied into pooled textures.
    pub fn duplicate(&self, gpu: &GpuContext, pool: &mut FieldPool) -> Result<Value, GpuError> {
        Ok(match self {
            Self::Field(f) => Self::Field(pool.duplicate(gpu, f)?),
            Self::Scalar(s) => Self::Scalar(*s),
            Self::VectorField(v) => {
                let x = pool.duplicate(gpu, v.face(Axis::X))?;
                let y = pool.duplicate(gpu, v.face(Axis::Y))?;
                let z = pool.duplicate(gpu, v.face(Axis::Z))?;
                Self::VectorField(StaggeredField::from_faces(v.cells(), [x, y, z])?)
            }
        })
    }
```

- [ ] **Step 4: Replace `AlreadyConsumed` and add `EvalStats`**

In `NodeError`, replace the `AlreadyConsumed` variant with:

```rust
    #[error("node {node:?} input {index} is already connected")]
    InputAlreadyConnected { node: NodeId, index: u32 },
```

Add to `node.rs`, after `NodeError`'s `impl` block:

```rust
/// Counters describing one evaluation, for tests and profiling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EvalStats {
    /// GPU copies made because a field fed more than one input, and a
    /// consumer that was not the last one took ownership of it.
    pub copies: u32,
}
```

- [ ] **Step 5: Rewrite `EvalCtx`'s input handling**

In `node.rs`, add `use std::collections::HashMap;` and change the socket import to
`use super::socket::{NodeId, SocketId, SocketSpec, SocketType};`.

Replace the `EvalCtx` struct (doc comment and all fields) with:

```rust
/// What a node is handed when it evaluates.
///
/// Inputs are lent, not given: `input` borrows, and `take_input` moves only
/// when this node is the value's last outstanding use (otherwise it copies).
pub struct EvalCtx<'a> {
    pub(crate) gpu: &'a GpuContext,
    pub(crate) pool: &'a mut FieldPool,
    pub(crate) pipelines: &'a mut PipelineCache,
    pub(crate) dims: FieldDims,
    pub(crate) node: NodeId,
    /// For each input index, the output socket feeding it, if any.
    pub(crate) sources: Vec<Option<SocketId>>,
    /// Tracks which indices `take_input` has already removed, so `input` can
    /// report a distinct error for "taken" versus "never connected".
    pub(crate) taken: Vec<bool>,
    /// Every value produced so far this evaluation, keyed by output socket.
    pub(crate) produced: &'a mut HashMap<SocketId, Value>,
    /// Input uses of each produced value that have not finished yet.
    pub(crate) remaining: &'a mut HashMap<SocketId, u32>,
    pub(crate) stats: &'a mut EvalStats,
    /// Persistent state. Only a node whose `stateful()` is true may touch it.
    pub(crate) state: &'a mut StateStore,
    pub(crate) stateful: bool,
    pub(crate) time: Time,
}
```

Replace `input` and `take_input` with:

```rust
    /// Borrow input `index`.
    pub fn input(&self, index: u32) -> Result<&Value, NodeError> {
        let i = index as usize;
        if self.taken.get(i).copied().unwrap_or(false) {
            return Err(NodeError::InputAlreadyTaken {
                node: self.node,
                index,
            });
        }
        self.sources
            .get(i)
            .copied()
            .flatten()
            .and_then(|src| self.produced.get(&src))
            .ok_or(NodeError::MissingInput {
                node: self.node,
                index,
            })
    }

    /// Take ownership of input `index`.
    ///
    /// If this is the value's last outstanding use, it is moved, with no copy.
    /// If other inputs still need it, this returns a GPU copy and counts it in
    /// `EvalStats::copies`. So a node that only reads an input should call
    /// `input` instead.
    ///
    /// Trap: calling `input(index)` after this does not report `MissingInput`
    /// (which would look like a wiring mistake in the graph). It reports
    /// `InputAlreadyTaken`, a bug in this node's own `eval`, since it read the
    /// same input twice.
    pub fn take_input(&mut self, index: u32) -> Result<Value, NodeError> {
        let i = index as usize;
        let node = self.node;
        let missing = || NodeError::MissingInput { node, index };
        if self.taken.get(i).copied().unwrap_or(false) {
            return Err(NodeError::InputAlreadyTaken { node, index });
        }
        let src = self.sources.get(i).copied().flatten().ok_or_else(missing)?;

        let left = self.remaining.get(&src).copied().unwrap_or(0);
        let value = if left <= 1 {
            self.remaining.remove(&src);
            self.produced.remove(&src).ok_or_else(missing)?
        } else {
            let shared = self.produced.get(&src).ok_or_else(missing)?;
            let copy = shared.duplicate(self.gpu, self.pool)?;
            if !matches!(copy, Value::Scalar(_)) {
                self.stats.copies += 1;
            }
            self.remaining.insert(src, left - 1);
            copy
        };

        if let Some(slot) = self.taken.get_mut(i) {
            *slot = true;
        }
        Ok(value)
    }
```

- [ ] **Step 6: Rewrite the evaluator**

In `crates/elements-core/src/graph/mod.rs`:

Change the `std::collections` import to `use std::collections::{HashMap, HashSet};`
and add `EvalStats` to the `pub use node::{...}` line.

Replace the `Graph` doc comment with:

```rust
/// A directed acyclic graph of nodes with one designated output.
///
/// Each input socket accepts exactly one edge (`connect` enforces this). An
/// output socket may feed any number of inputs: the evaluator lends the value
/// to each consumer, moves it to the last, and copies it for any earlier
/// consumer that takes ownership. See `EvalCtx::take_input`.
```

In `connect`, replace the `AlreadyConsumed` check with:

```rust
        if self.edges.contains_key(&to) {
            return Err(NodeError::InputAlreadyConnected {
                node: to.node,
                index: to.index,
            });
        }
```

and change the last sentence of its doc comment to: "Rejects type mismatches,
nonexistent sockets, and a second edge into an input that already has one."

Add `stats` to `Evaluated`:

```rust
#[derive(Debug)]
pub struct Evaluated {
    /// The output node's first output. The caller owns it and should return it
    /// to the pool with `Value::release_to` when finished.
    pub value: Value,
    pub stats: EvalStats,
}
```

Add, below `Evaluated`:

```rust
/// Everything one `eval_frame` call threads through its nodes.
struct Run<'a> {
    gpu: &'a GpuContext,
    pool: &'a mut FieldPool,
    pipelines: &'a mut PipelineCache,
    state: &'a mut StateStore,
    time: Time,
    dims: FieldDims,
    produced: HashMap<SocketId, Value>,
    remaining: HashMap<SocketId, u32>,
    stats: EvalStats,
}

impl Run<'_> {
    /// One use of `src` finished without taking ownership. If it was the last
    /// use, the value goes back to the pool.
    fn consume(&mut self, src: SocketId) {
        let Some(left) = self.remaining.get_mut(&src) else {
            return;
        };
        *left = left.saturating_sub(1);
        if *left == 0 {
            self.remaining.remove(&src);
            if let Some(value) = self.produced.remove(&src) {
                value.release_to(self.pool);
            }
        }
    }
}
```

Replace `eval_frame` (leave `eval` alone) with:

```rust
    /// Evaluate the output node and everything it depends on, as frame `time`,
    /// reading and writing persistent state in `state`.
    ///
    /// Every field produced along the way is back in `pool` when this returns,
    /// except the result, whether evaluation succeeds or fails.
    pub fn eval_frame(
        &self,
        gpu: &GpuContext,
        pool: &mut FieldPool,
        pipelines: &mut PipelineCache,
        state: &mut StateStore,
        time: Time,
        dims: FieldDims,
    ) -> Result<Evaluated, NodeError> {
        let output = self.output.ok_or(NodeError::NoOutput)?;
        let order = self.evaluation_order()?;

        let mut run = Run {
            gpu,
            pool,
            pipelines,
            state,
            time,
            dims,
            produced: HashMap::new(),
            remaining: HashMap::new(),
            stats: EvalStats::default(),
        };
        let result = self.run(&mut run, &order, output);
        for (_, value) in run.produced.drain() {
            value.release_to(run.pool);
        }
        Ok(Evaluated {
            value: result?,
            stats: run.stats,
        })
    }

    fn run(&self, run: &mut Run<'_>, order: &[NodeId], output: NodeId) -> Result<Value, NodeError> {
        // Count every input use of every output, among the nodes that will run.
        let needed: HashSet<NodeId> = order.iter().copied().collect();
        for (to, from) in &self.edges {
            if needed.contains(&to.node) {
                *run.remaining.entry(*from).or_insert(0) += 1;
            }
        }

        let result_socket = SocketId {
            node: output,
            index: 0,
        };

        for &id in order {
            let node = self.node(id)?;
            let spec = node.sockets();
            let sources: Vec<Option<SocketId>> = (0..spec.inputs.len() as u32)
                .map(|index| self.edges.get(&SocketId { node: id, index }).copied())
                .collect();

            let mut ctx = EvalCtx {
                gpu: run.gpu,
                pool: &mut *run.pool,
                pipelines: &mut *run.pipelines,
                dims: run.dims,
                node: id,
                sources: sources.clone(),
                taken: vec![false; sources.len()],
                produced: &mut run.produced,
                remaining: &mut run.remaining,
                stats: &mut run.stats,
                state: &mut *run.state,
                stateful: node.stateful(),
                time: run.time,
            };
            let outputs = node
                .eval(&mut ctx)
                .map_err(|err| err.with_evaluating_node(id))?;
            let taken = std::mem::take(&mut ctx.taken);

            // Inputs this node only borrowed are finished with now.
            for (index, src) in sources.iter().enumerate() {
                if let Some(src) = src
                    && !taken[index]
                {
                    run.consume(*src);
                }
            }

            for (index, value) in outputs.into_iter().enumerate() {
                let socket = SocketId {
                    node: id,
                    index: index as u32,
                };
                let wanted = socket == result_socket
                    || run.remaining.get(&socket).copied().unwrap_or(0) > 0;
                if wanted {
                    run.produced.insert(socket, value);
                } else {
                    value.release_to(run.pool);
                }
            }
        }

        run.produced
            .remove(&result_socket)
            .ok_or(NodeError::UnknownNode(output))
    }
```

- [ ] **Step 7: Fix the `Output` unit test's hand-built context again**

In `crates/elements-core/src/nodes/output.rs`'s test module, add
`use std::collections::HashMap;` and extend the graph import to
`use crate::graph::{EvalStats, NodeId, SocketId, StateStore, Time};`. Then build the
context from a produced map instead of an `inputs` vector:

```rust
        let src = SocketId {
            node: NodeId(0),
            index: 0,
        };
        let mut produced = HashMap::from([(src, Value::Scalar(1.0))]);
        let mut remaining = HashMap::from([(src, 1u32)]);
        let mut stats = EvalStats::default();
        let mut state = StateStore::new();

        let mut ctx = EvalCtx {
            gpu: &gpu,
            pool: &mut pool,
            pipelines: &mut pipelines,
            dims: FieldDims::new(2, 2, 2),
            node: NodeId(42),
            sources: vec![Some(src)],
            taken: vec![false],
            produced: &mut produced,
            remaining: &mut remaining,
            stats: &mut stats,
            state: &mut state,
            stateful: false,
            time: Time::at(1, 1, 24.0),
        };
```

Update the test's doc comment: the context is "built by hand with a `Scalar`
produced into the source feeding the `Field` input".

- [ ] **Step 8: Run all tests and watch them pass**

Run: `cargo nextest run -p elements-core`
Expected: PASS, all tests, including the existing `tests/graph.rs` cases (the
diamond, the unconnected input, `input_after_take_input_reports_the_input_was_taken`).

- [ ] **Step 9: Prove the tests can fail**

Run `cargo nextest run -p elements-core --test fan_out` after each mutation, and restore after each.

- A ("move instead of copy"): in `take_input`, change `if left <= 1 {` to `if true {`
  → `a_field_feeding_two_consumers_is_copied_for_the_first` fails (the second consumer gets `MissingInput`).
- B ("always copy"): change `if left <= 1 {` to `if left < 1 {`
  → `a_chain_with_no_branching_copies_nothing` fails (`copies` is 1).
- C ("drop instead of release"): in `Run::consume`, replace `value.release_to(self.pool);` with `drop(value);`
  → `fields_return_to_the_pool_after_their_last_use` fails.
- D: in `connect`, delete the `contains_key` check → `an_input_accepts_only_one_edge` fails.

- [ ] **Step 10: `just check`, then commit**

```bash
just check
git add crates/elements-core/src crates/elements-core/tests
git commit -F - <<'EOF'
Let one output feed several inputs and pool every field after last use

Ember's graphs branch: velocity feeds both advection and vorticity
confinement. Values are now lent to consumers, moved to the last one,
and copied only when an earlier consumer takes ownership. Copies are
counted, so tests can prove that a graph with no branching never copies.
The same bookkeeping closes a Core v1 leak where values nobody consumed
were dropped rather than pooled, so every frame reallocated them.

A second edge into one input used to silently replace the first. It is
now an error.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XCzEwqREbRMwabH11TvnMb
EOF
```

---

## Task 6: Staggered sampling in WGSL

**Why:** every piece-2 advection kernel and every piece-3 export resamples
velocity at arbitrary points. Neither `R32Float` nor `Rgba32Float` can be filtered in the
WebGPU baseline (umbrella E2), so interpolation is written by hand. It is written
once, here, and tested against a field where trilinear interpolation must be exact.

**Files:**
- Create: `crates/elements-core/src/gpu/shaders/vector_sample.wgsl`
- Modify: `crates/elements-core/src/gpu/field.rs` (`Field::write`)
- Test: `crates/elements-core/tests/vector_sample.rs` (create)

**Interfaces:**
- Consumes: `StaggeredField`, `Axis` (Task 3), `FieldPool::acquire_staggered_uninit` (Task 3).
- Produces:
  - WGSL library functions. A kernel prepends the library to its own source
    (`format!("{LIBRARY}\n{KERNEL}")`, or `concat!(include_str!(..), include_str!(..))` for static sources):
    - `fn sample_trilinear(tex: texture_3d<f32>, p: vec3<f32>) -> f32`: `p` is in texel coordinates, and it clamps to the edge
    - `fn sample_velocity(u: texture_3d<f32>, v: texture_3d<f32>, w: texture_3d<f32>, x: vec3<f32>) -> vec3<f32>`: `x` is in cell units
    - `fn velocity_at_cell(u: texture_3d<f32>, v: texture_3d<f32>, w: texture_3d<f32>, c: vec3<i32>) -> vec3<f32>`
  - `Field::write(&self, ctx: &GpuContext, values: &[f32]) -> Result<(), GpuError>`: uploads an `R32Float` field, x-fastest

- [ ] **Step 1: Write the failing tests**

Create `crates/elements-core/tests/vector_sample.rs`:

```rust
//! The hand-written staggered sampling library, checked against linear velocity
//! fields. Trilinear interpolation reproduces a linear function exactly, so any
//! error beyond float rounding is a bug in the sampling code.

use elements_core::gpu::{Axis, FieldDims, FieldPool, GpuContext, PipelineCache, StaggeredField};
use wgpu::util::DeviceExt;

const LIBRARY: &str = include_str!("../src/gpu/shaders/vector_sample.wgsl");

const PROBE: &str = r#"
@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> results: array<vec4<f32>>;

// points[i].w == 0: sample_velocity at xyz (cell units).
// points[i].w == 1: velocity_at_cell at the integer cell xyz.
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&points)) {
        return;
    }
    let p = points[i];
    if (p.w == 0.0) {
        results[i] = vec4<f32>(sample_velocity(vel_x, vel_y, vel_z, p.xyz), 0.0);
    } else {
        results[i] = vec4<f32>(velocity_at_cell(vel_x, vel_y, vel_z, vec3<i32>(p.xyz)), 0.0);
    }
}
"#;

fn u_at(p: [f32; 3]) -> f32 {
    1.0 + 2.0 * p[0] + 3.0 * p[1] + 5.0 * p[2]
}
fn v_at(p: [f32; 3]) -> f32 {
    7.0 - p[0] + 0.5 * p[1] + 2.0 * p[2]
}
fn w_at(p: [f32; 3]) -> f32 {
    -3.0 + 4.0 * p[0] - p[1] + 0.25 * p[2]
}

/// Where texel (0, 0, 0) of each face sits, in cell units.
fn face_offset(axis: Axis) -> [f32; 3] {
    match axis {
        Axis::X => [0.0, 0.5, 0.5],
        Axis::Y => [0.5, 0.0, 0.5],
        Axis::Z => [0.5, 0.5, 0.0],
    }
}

fn fill_face(ctx: &GpuContext, field: &StaggeredField, axis: Axis, f: fn([f32; 3]) -> f32) {
    let face = field.face(axis);
    let d = face.dims();
    let o = face_offset(axis);
    let mut values = Vec::with_capacity(d.voxel_count());
    for z in 0..d.z {
        for y in 0..d.y {
            for x in 0..d.x {
                values.push(f([x as f32 + o[0], y as f32 + o[1], z as f32 + o[2]]));
            }
        }
    }
    face.write(ctx, &values).unwrap();
}

fn linear_velocity(ctx: &GpuContext, pool: &mut FieldPool, cells: FieldDims) -> StaggeredField {
    let field = pool.acquire_staggered_uninit(ctx, cells).unwrap();
    fill_face(ctx, &field, Axis::X, u_at);
    fill_face(ctx, &field, Axis::Y, v_at);
    fill_face(ctx, &field, Axis::Z, w_at);
    field
}

/// Run the probe kernel over `points` and return one vec4 per point.
fn probe(
    ctx: &GpuContext,
    cache: &mut PipelineCache,
    field: &StaggeredField,
    points: &[[f32; 4]],
) -> Vec<[f32; 4]> {
    let source = format!("{LIBRARY}\n{PROBE}");
    let pipeline = cache
        .get_or_create(ctx, "test-vector-sample-probe", &source, "probe")
        .unwrap();
    let device = ctx.device();
    let size = std::mem::size_of_val(points) as u64;

    let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("probe-points"),
        contents: bytemuck::cast_slice(points),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let results = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-results"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe-bind-group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(field.face(Axis::X).view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(field.face(Axis::Y).view()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(field.face(Axis::Z).view()),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: point_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: results.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((points.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&results, 0, &staging, 0, size);
    ctx.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let out: Vec<[f32; 4]> = {
        let mapped = slice.get_mapped_range().unwrap();
        bytemuck::cast_slice(&mapped).to_vec()
    };
    staging.unmap();
    out
}

fn assert_near(got: [f32; 4], want: [f32; 3], at: [f32; 4]) {
    for axis in 0..3 {
        assert!(
            (got[axis] - want[axis]).abs() < 1e-3,
            "at {at:?}: component {axis} is {}, expected {}",
            got[axis],
            want[axis]
        );
    }
}

#[test]
fn sample_velocity_reproduces_a_linear_field_exactly() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let field = linear_velocity(&ctx, &mut pool, FieldDims::new(4, 5, 6));

    // Every point lies inside all three faces' sample hulls, [0.5, n - 0.5] on
    // each axis, so edge clamping never engages.
    let mut points = Vec::new();
    for x in [0.5, 1.25, 2.0, 3.5] {
        for y in [0.5, 2.75, 4.5] {
            for z in [0.5, 3.1, 5.5] {
                points.push([x, y, z, 0.0]);
            }
        }
    }

    let got = probe(&ctx, &mut cache, &field, &points);
    for (point, value) in points.iter().zip(got) {
        let p = [point[0], point[1], point[2]];
        assert_near(value, [u_at(p), v_at(p), w_at(p)], *point);
    }
}

#[test]
fn velocity_at_cell_is_the_velocity_at_the_cell_centre() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(4, 5, 6);
    let field = linear_velocity(&ctx, &mut pool, cells);

    let mut points = Vec::new();
    for x in 0..cells.x {
        for y in 0..cells.y {
            for z in 0..cells.z {
                points.push([x as f32, y as f32, z as f32, 1.0]);
            }
        }
    }

    let got = probe(&ctx, &mut cache, &field, &points);
    for (point, value) in points.iter().zip(got) {
        let centre = [point[0] + 0.5, point[1] + 0.5, point[2] + 0.5];
        assert_near(value, [u_at(centre), v_at(centre), w_at(centre)], *point);
    }
}
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cargo nextest run -p elements-core --test vector_sample`
Expected: FAIL to compile: `couldn't read .../vector_sample.wgsl` and `no method named write`.

- [ ] **Step 3: Add `Field::write`**

In `crates/elements-core/src/gpu/field.rs`, add inside `impl Field`:

```rust
    /// Upload `values` (x-fastest, one `f32` per voxel) into an `R32Float` field.
    ///
    /// `Queue::write_texture` has no row-alignment requirement, unlike
    /// buffer-to-texture copies, so no padding is needed.
    pub fn write(&self, ctx: &GpuContext, values: &[f32]) -> Result<(), GpuError> {
        if self.format != FieldFormat::R32Float || values.len() != self.dims.voxel_count() {
            return Err(GpuError::Validation(format!(
                "write: {} values into a {:?} {:?} field",
                values.len(),
                self.dims,
                self.format
            )));
        }
        ctx.scoped(|| {
            ctx.queue().write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(values),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.dims.x * 4),
                    rows_per_image: Some(self.dims.y),
                },
                self.dims.extent(),
            );
        })
    }
```

- [ ] **Step 4: Create the library**

Create `crates/elements-core/src/gpu/shaders/vector_sample.wgsl`:

```wgsl
// Hand-written trilinear sampling for R32Float fields and staggered velocity.
//
// A library, not an entry point: kernels prepend it to their own source.
// R32Float is not filterable in the WebGPU baseline (FLOAT32_FILTERABLE is an
// optional feature), so there is no sampler. Every read is a textureLoad.
//
// Positions are in cell units: cell (i, j, k) spans [i, i + 1) on each axis,
// and its centre is (i + 0.5, j + 0.5, k + 0.5). Texel (i, j, k) of the X face
// sits at (i, j + 0.5, k + 0.5), of the Y face at (i + 0.5, j, k + 0.5), and of
// the Z face at (i + 0.5, j + 0.5, k).

const X_FACE_OFFSET: vec3<f32> = vec3<f32>(0.0, 0.5, 0.5);
const Y_FACE_OFFSET: vec3<f32> = vec3<f32>(0.5, 0.0, 0.5);
const Z_FACE_OFFSET: vec3<f32> = vec3<f32>(0.5, 0.5, 0.0);

// Trilinearly interpolate `tex` at texel coordinates `p`, where texel (i, j, k)
// is sampled exactly at p = (i, j, k). Coordinates outside the texture clamp to
// its edge.
fn sample_trilinear(tex: texture_3d<f32>, p: vec3<f32>) -> f32 {
    let last = vec3<i32>(textureDimensions(tex)) - vec3<i32>(1);
    let q = clamp(p, vec3<f32>(0.0), vec3<f32>(last));
    let i0 = vec3<i32>(floor(q));
    let i1 = min(i0 + vec3<i32>(1), last);
    let f = q - vec3<f32>(i0);

    let c000 = textureLoad(tex, vec3<i32>(i0.x, i0.y, i0.z), 0).x;
    let c100 = textureLoad(tex, vec3<i32>(i1.x, i0.y, i0.z), 0).x;
    let c010 = textureLoad(tex, vec3<i32>(i0.x, i1.y, i0.z), 0).x;
    let c110 = textureLoad(tex, vec3<i32>(i1.x, i1.y, i0.z), 0).x;
    let c001 = textureLoad(tex, vec3<i32>(i0.x, i0.y, i1.z), 0).x;
    let c101 = textureLoad(tex, vec3<i32>(i1.x, i0.y, i1.z), 0).x;
    let c011 = textureLoad(tex, vec3<i32>(i0.x, i1.y, i1.z), 0).x;
    let c111 = textureLoad(tex, vec3<i32>(i1.x, i1.y, i1.z), 0).x;

    let c00 = mix(c000, c100, f.x);
    let c10 = mix(c010, c110, f.x);
    let c01 = mix(c001, c101, f.x);
    let c11 = mix(c011, c111, f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

// Velocity at position `x` (cell units) from the X, Y and Z face textures.
fn sample_velocity(u: texture_3d<f32>, v: texture_3d<f32>, w: texture_3d<f32>, x: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        sample_trilinear(u, x - X_FACE_OFFSET),
        sample_trilinear(v, x - Y_FACE_OFFSET),
        sample_trilinear(w, x - Z_FACE_OFFSET)
    );
}

// Velocity at the centre of cell `c`: the mean of the two faces bounding it on
// each axis. Equal to sample_velocity at c + 0.5, but with six loads instead of 24.
fn velocity_at_cell(u: texture_3d<f32>, v: texture_3d<f32>, w: texture_3d<f32>, c: vec3<i32>) -> vec3<f32> {
    return 0.5 * vec3<f32>(
        textureLoad(u, c, 0).x + textureLoad(u, c + vec3<i32>(1, 0, 0), 0).x,
        textureLoad(v, c, 0).x + textureLoad(v, c + vec3<i32>(0, 1, 0), 0).x,
        textureLoad(w, c, 0).x + textureLoad(w, c + vec3<i32>(0, 0, 1), 0).x
    );
}
```

- [ ] **Step 5: Run the tests and watch them pass**

Run: `cargo nextest run -p elements-core --test vector_sample`
Expected: PASS, both tests.

If a `wgpu` call in the test harness does not compile, check the signature in the vendored source (CLAUDE.md "Verifying wgpu APIs") rather than on docs.rs.

- [ ] **Step 6: Prove the tests can fail**

- A: in the WGSL, change `X_FACE_OFFSET` to `vec3<f32>(0.5, 0.0, 0.5)` → `sample_velocity...` fails
  (component 0 is off by 0.5). Restore.
- B: in `velocity_at_cell`, change `c + vec3<i32>(1, 0, 0)` to `c` → `velocity_at_cell...` fails
  (component 0 is off by 1.0). Restore.

- [ ] **Step 7: `just check`, then commit**

```bash
just check
git add crates/elements-core/src/gpu/shaders/vector_sample.wgsl crates/elements-core/src/gpu/field.rs \
        crates/elements-core/tests/vector_sample.rs
git commit -F - <<'EOF'
Add a WGSL library for sampling staggered velocity by hand

No 32-bit float format is filterable in the WebGPU baseline, so every
advection and export kernel has to interpolate by hand. Writing that
once, and checking it against linear fields where trilinear
interpolation must be exact, keeps a face-offset mistake from hiding
inside each solver kernel as plausible-looking but wrong smoke.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XCzEwqREbRMwabH11TvnMb
EOF
```

---

## Task 7: State snapshots

**Why:** the timeline caches the state *entering* each frame, so scrubbing back
restores it instead of re-simulating from the start.

**Files:**
- Modify: `crates/elements-core/src/graph/state.rs`
- Modify: `crates/elements-core/src/graph/node.rs` (`Value::gpu_bytes`)
- Modify: `crates/elements-core/src/graph/mod.rs` (export `Snapshot`)
- Test: `crates/elements-core/tests/state.rs`

**Interfaces:**
- Consumes: `Value::duplicate` (Task 5), `StateStore` (Task 4).
- Produces:
  - `Value::gpu_bytes(&self) -> u64` (a field's texel bytes; 4 for a scalar)
  - `graph::Snapshot` with `bytes(&self) -> u64` and `release_to(self, pool: &mut FieldPool)`
  - `StateStore::snapshot(&self, gpu: &GpuContext, pool: &mut FieldPool) -> Result<Snapshot, GpuError>`
  - `StateStore::restore(&mut self, snapshot: &Snapshot, gpu: &GpuContext, pool: &mut FieldPool) -> Result<(), GpuError>`.
    It **copies** out of the snapshot, which stays valid and can be restored again.

- [ ] **Step 1: Write the failing tests**

Append to `crates/elements-core/tests/state.rs`:

```rust
#[test]
fn a_restored_snapshot_resumes_exactly_where_it_was_taken() {
    let mut h = Harness::new();
    let (graph, dims) = accumulate_graph();
    let mut state = StateStore::new();
    for frame in 1..=3 {
        h.frame(&graph, &mut state, frame, dims);
    }

    let snapshot = state.snapshot(&h.gpu, &mut h.pool).unwrap();
    let frame_4: Vec<u32> = h
        .frame(&graph, &mut state, 4, dims)
        .iter()
        .map(|v| v.to_bits())
        .collect();
    h.frame(&graph, &mut state, 5, dims);
    h.frame(&graph, &mut state, 6, dims);

    // Restoring twice proves the snapshot survives being restored.
    for _ in 0..2 {
        state.restore(&snapshot, &h.gpu, &mut h.pool).unwrap();
        let again: Vec<u32> = h
            .frame(&graph, &mut state, 4, dims)
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(again, frame_4, "frame 4 after restore must be bit-identical");
    }
    snapshot.release_to(&mut h.pool);
}

#[test]
fn a_snapshot_counts_every_stored_texel() {
    let mut h = Harness::new();
    let (graph, dims) = accumulate_graph();
    let mut state = StateStore::new();

    let empty = state.snapshot(&h.gpu, &mut h.pool).unwrap();
    assert_eq!(empty.bytes(), 0);

    h.frame(&graph, &mut state, 1, dims);
    let one = state.snapshot(&h.gpu, &mut h.pool).unwrap();
    assert_eq!(one.bytes(), 4 * 4 * 4 * 4, "one 4³ R32Float field");
}
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cargo nextest run -p elements-core --test state`
Expected: FAIL to compile: `no method named snapshot`.

- [ ] **Step 3: `Value::gpu_bytes`**

In `node.rs`, add inside `impl Value`:

```rust
    /// Bytes of GPU memory this value's textures occupy (4 for a scalar).
    pub fn gpu_bytes(&self) -> u64 {
        let field_bytes =
            |f: &Field| f.dims().voxel_count() as u64 * f.format().bytes_per_voxel() as u64;
        match self {
            Self::Field(f) => field_bytes(f),
            Self::Scalar(_) => std::mem::size_of::<f32>() as u64,
            Self::VectorField(v) => Axis::ALL.iter().map(|&a| field_bytes(v.face(a))).sum(),
        }
    }
```

- [ ] **Step 4: Snapshots**

In `crates/elements-core/src/graph/state.rs`, change the imports to:

```rust
use std::collections::BTreeMap;

use crate::gpu::{FieldPool, GpuContext, GpuError};

use super::node::Value;
use super::socket::NodeId;
```

and add:

```rust
/// A GPU copy of every slot in a `StateStore` at one moment.
pub struct Snapshot {
    slots: Vec<((NodeId, &'static str), Value)>,
    bytes: u64,
}

impl Snapshot {
    /// GPU memory this snapshot holds.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Return this snapshot's textures to `pool`.
    pub fn release_to(self, pool: &mut FieldPool) {
        for (_, value) in self.slots {
            value.release_to(pool);
        }
    }
}
```

and inside `impl StateStore`:

```rust
    /// Copy every slot. The store itself is unchanged.
    pub fn snapshot(&self, gpu: &GpuContext, pool: &mut FieldPool) -> Result<Snapshot, GpuError> {
        let mut slots = Vec::with_capacity(self.slots.len());
        let mut bytes = 0;
        for (key, value) in &self.slots {
            match value.duplicate(gpu, pool) {
                Ok(copy) => {
                    bytes += copy.gpu_bytes();
                    slots.push((*key, copy));
                }
                Err(e) => {
                    for (_, partial) in slots {
                        partial.release_to(pool);
                    }
                    return Err(e);
                }
            }
        }
        Ok(Snapshot { slots, bytes })
    }

    /// Replace the store's contents with a copy of `snapshot`.
    ///
    /// Copies rather than moves: the timeline keeps the snapshot cached,
    /// because the same frame may be scrubbed to again. On error the store is
    /// left partially restored, and the caller must clear it.
    pub fn restore(
        &mut self,
        snapshot: &Snapshot,
        gpu: &GpuContext,
        pool: &mut FieldPool,
    ) -> Result<(), GpuError> {
        self.clear(pool);
        for (key, value) in &snapshot.slots {
            let copy = value.duplicate(gpu, pool)?;
            self.slots.insert(*key, copy);
        }
        Ok(())
    }
```

In `graph/mod.rs`, change `pub use state::StateStore;` to `pub use state::{Snapshot, StateStore};`.

- [ ] **Step 5: Run the tests and watch them pass**

Run: `cargo nextest run -p elements-core --test state`
Expected: PASS.

- [ ] **Step 6: Prove the tests can fail**

- A: in `restore`, delete the line `self.slots.insert(*key, copy);` → the restore test fails
  (frame 4 comes back as the first-frame value). Restore the line.
- B: in `gpu_bytes`, change the `Field` arm to `Self::Field(f) => f.format().bytes_per_voxel() as u64,`
  → `a_snapshot_counts_every_stored_texel` fails (`4` vs `256`). Restore.

- [ ] **Step 7: `just check`, then commit**

```bash
just check
git add crates/elements-core/src/graph crates/elements-core/tests/state.rs
git commit -F - <<'EOF'
Snapshot and restore node state on the GPU

Scrubbing back to a frame should restore the state entering it rather
than re-simulate from the start. A restore copies out of the snapshot
instead of moving it, because the cache must keep that snapshot for the
next time the same frame is scrubbed to.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XCzEwqREbRMwabH11TvnMb
EOF
```

---

## Task 8: The timeline and frame cache

**Why:** this is the component that makes a simulation behave like a timeline.
Blender asks for frames in any order, and each answer must be bit-identical to
what forward playback would have produced.

**Files:**
- Create: `crates/elements-core/src/graph/timeline.rs`
- Modify: `crates/elements-core/src/graph/mod.rs`
- Test: `crates/elements-core/tests/timeline.rs` (create)

**Interfaces:**
- Consumes: `Graph::eval_frame`, `Graph::is_stateful`, `StateStore::{snapshot, restore, clear}`, `Snapshot`, `Time::at`.
- Produces:
  - `graph::DEFAULT_CACHE_BUDGET_MB: u32 = 2048`
  - `graph::TimelineConfig { pub fps: f64, pub start_frame: u32, pub cache_budget_bytes: u64 }`, with a `Default` that uses the three defaults
  - `graph::Timeline` with:
    - `Timeline::new(config: TimelineConfig) -> Timeline`
    - `goto(&mut self, graph: &Graph, gpu: &GpuContext, pool: &mut FieldPool, pipelines: &mut PipelineCache, dims: FieldDims, frame: u32) -> Result<Evaluated, NodeError>`.
      The caller owns `Evaluated::value` and releases it.
    - `reset(&mut self, pool: &mut FieldPool)`: empties the state and cache, returning textures to the pool
    - `discard(&mut self)`: empties them without pooling, for use after the device is lost
    - `steps_run(&self) -> u64`, `cached_frames(&self) -> Vec<u32>`, `cached_bytes(&self) -> u64`, `caching_enabled(&self) -> bool`
    - `take_warning(&mut self) -> Option<String>`
    - `config(&self) -> TimelineConfig`

**Algorithm.** Tests hold the implementation to this:
- A snapshot for frame N is the state *entering* N, taken before N's `eval`.
- `goto(frame)`: clamp `frame` up to `start_frame`. For a graph with no stateful
  nodes, run one `eval_frame` and stop. Otherwise, pick a starting point: the held
  state (if its cursor is at or before `frame`) or the nearest cached snapshot at
  or before `frame`, whichever is later. If neither exists, clear the state and
  start at `start_frame`. Step forward to `frame`, snapshotting each frame
  entered that is not already cached, then run one `eval` for `frame`.
- On `StateShape`, reset once and retry. On any other error, clear the state
  (it may be half-consumed) but keep the cache.
- Eviction is least-recently-used, never evicting the snapshot just restored. A
  snapshot larger than the whole budget disables caching and sets a warning.

- [ ] **Step 1: Write the failing tests**

Create `crates/elements-core/tests/timeline.rs`:

```rust
use elements_core::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{Document, Graph, NodeRegistry, Timeline, TimelineConfig};

const ACCUMULATE_DOC: &str = r#"{
  "version": 1,
  "dims": [4, 4, 4],
  "nodes": [
    { "id": 0, "kind": "core.constant_field", "params": { "value": 0.5 } },
    { "id": 1, "kind": "core.accumulate", "params": {} },
    { "id": 2, "kind": "core.output", "params": {} }
  ],
  "edges": [
    { "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 },
    { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 0 }
  ],
  "output": 2
}"#;

const CONSTANT_DOC: &str = r#"{
  "version": 1,
  "dims": [4, 4, 4],
  "nodes": [
    { "id": 0, "kind": "core.constant_field", "params": { "value": 0.5 } },
    { "id": 1, "kind": "core.output", "params": {} }
  ],
  "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
  "output": 1
}"#;

/// One 4³ R32Float field.
const SNAPSHOT_BYTES: u64 = 4 * 4 * 4 * 4;

struct Harness {
    gpu: GpuContext,
    pool: FieldPool,
    pipelines: PipelineCache,
    graph: Graph,
    dims: FieldDims,
}

impl Harness {
    fn new(doc: &str) -> Self {
        let (graph, dims) = Document::from_json(doc)
            .unwrap()
            .into_graph(&NodeRegistry::with_builtins())
            .unwrap();
        Self {
            gpu: GpuContext::new_headless().expect("no GPU adapter available"),
            pool: FieldPool::new(),
            pipelines: PipelineCache::new(),
            graph,
            dims,
        }
    }

    fn goto_at(&mut self, timeline: &mut Timeline, frame: u32, dims: FieldDims) -> Vec<f32> {
        let evaluated = timeline
            .goto(&self.graph, &self.gpu, &mut self.pool, &mut self.pipelines, dims, frame)
            .unwrap();
        let values = evaluated.value.as_field().unwrap().read_back(&self.gpu).unwrap();
        evaluated.value.release_to(&mut self.pool);
        values
    }

    fn goto(&mut self, timeline: &mut Timeline, frame: u32) -> Vec<f32> {
        let dims = self.dims;
        self.goto_at(timeline, frame, dims)
    }
}

fn timeline(cache_budget_bytes: u64) -> Timeline {
    Timeline::new(TimelineConfig {
        fps: 24.0,
        start_frame: 1,
        cache_budget_bytes,
    })
}

fn expected(frame: u32) -> f32 {
    0.5 * frame as f32 / 24.0
}

fn assert_frame(values: &[f32], frame: u32) {
    let want = expected(frame);
    for &v in values {
        assert!((v - want).abs() < 1e-6, "frame {frame}: expected {want}, got {v}");
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn playing_forward_matches_the_closed_form() {
    let mut h = Harness::new(ACCUMULATE_DOC);
    let mut tl = timeline(1 << 30);
    for frame in 1..=10 {
        let values = h.goto(&mut tl, frame);
        assert_frame(&values, frame);
    }
    assert_eq!(tl.steps_run(), 10, "forward playback costs one step per frame");
}

#[test]
fn scrubbing_back_to_a_cached_frame_costs_one_step() {
    let mut h = Harness::new(ACCUMULATE_DOC);
    let mut tl = timeline(1 << 30);
    h.goto(&mut tl, 10);
    let before = tl.steps_run();
    let values = h.goto(&mut tl, 3);
    assert_frame(&values, 3);
    assert_eq!(tl.steps_run(), before + 1);
}

#[test]
fn resuming_from_an_earlier_cached_frame_steps_forward_correctly() {
    let mut h = Harness::new(ACCUMULATE_DOC);
    let mut tl = timeline(1 << 30);
    h.goto(&mut tl, 5);
    h.goto(&mut tl, 3);
    // The held state is entering 4. Frame 5's entering state is cached and is
    // later, so the timeline restores it and steps through 5 and 6 to 7.
    let values = h.goto(&mut tl, 7);
    assert_frame(&values, 7);
    assert_eq!(tl.steps_run(), 5 + 1 + 3);
}

#[test]
fn the_cache_stays_within_budget_and_evicted_frames_re_simulate() {
    let mut h = Harness::new(ACCUMULATE_DOC);
    // Room for three 4³ snapshots (the start frame's empty snapshot is free).
    let budget = 3 * SNAPSHOT_BYTES;
    let mut tl = timeline(budget);
    h.goto(&mut tl, 10);
    assert!(
        tl.cached_bytes() <= budget,
        "{} bytes cached against a {budget}-byte budget",
        tl.cached_bytes()
    );
    assert!(tl.cached_frames().contains(&10), "the newest frame survives eviction");
    assert!(
        !tl.cached_frames().contains(&5),
        "frame 5 must have been evicted for this test to mean anything: {:?}",
        tl.cached_frames()
    );
    let values = h.goto(&mut tl, 5);
    assert_frame(&values, 5);
}

#[test]
fn a_frame_is_bit_identical_however_it_is_reached() {
    let mut h = Harness::new(ACCUMULATE_DOC);

    let mut direct = timeline(1 << 30);
    let want = bits(&h.goto(&mut direct, 10));

    let mut wandering = timeline(1 << 30);
    h.goto(&mut wandering, 15);
    h.goto(&mut wandering, 3);
    let got = bits(&h.goto(&mut wandering, 10));

    assert_eq!(got, want);
}

#[test]
fn frames_before_the_start_are_the_start_frame() {
    let mut h = Harness::new(ACCUMULATE_DOC);
    let mut tl = timeline(1 << 30);
    assert_frame(&h.goto(&mut tl, 0), 1);
    assert_frame(&h.goto(&mut tl, 1), 1);
}

#[test]
fn a_budget_smaller_than_one_snapshot_disables_caching_but_stays_correct() {
    let mut h = Harness::new(ACCUMULATE_DOC);
    let mut tl = timeline(SNAPSHOT_BYTES - 1);
    assert_frame(&h.goto(&mut tl, 5), 5);
    assert!(!tl.caching_enabled());
    assert!(tl.take_warning().is_some(), "the user must be told scrubbing is slow");
    assert!(tl.cached_frames().is_empty());
    assert_frame(&h.goto(&mut tl, 2), 2);
}

#[test]
fn a_domain_change_resets_the_simulation() {
    let mut h = Harness::new(ACCUMULATE_DOC);
    let mut tl = timeline(1 << 30);
    h.goto(&mut tl, 3);
    // Same graph, bigger domain: every cached snapshot is now the wrong shape.
    let values = h.goto_at(&mut tl, 3, FieldDims::new(8, 8, 8));
    assert_eq!(values.len(), 8 * 8 * 8);
    assert_frame(&values, 3);
}

#[test]
fn a_stateless_graph_is_evaluated_directly() {
    let mut h = Harness::new(CONSTANT_DOC);
    let mut tl = timeline(1 << 30);
    for frame in [7, 2, 9] {
        let values = h.goto(&mut tl, frame);
        assert!(values.iter().all(|&v| v == 0.5));
    }
    assert!(tl.cached_frames().is_empty());
    assert_eq!(tl.steps_run(), 3);
}
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cargo nextest run -p elements-core --test timeline`
Expected: FAIL to compile: `unresolved imports ... Timeline, TimelineConfig`.

- [ ] **Step 3: Create `graph/timeline.rs`**

```rust
//! Stepping a stateful graph through time, with a cache of frame snapshots.

use std::collections::BTreeMap;

use crate::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};

use super::node::NodeError;
use super::state::{Snapshot, StateStore};
use super::time::{DEFAULT_FPS, DEFAULT_START_FRAME, Time};
use super::{Evaluated, Graph};

/// The frame-cache budget a document gets when it does not specify one.
pub const DEFAULT_CACHE_BUDGET_MB: u32 = 2048;

/// How one timeline runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelineConfig {
    pub fps: f64,
    pub start_frame: u32,
    /// GPU memory the snapshot cache may hold.
    pub cache_budget_bytes: u64,
}

impl Default for TimelineConfig {
    fn default() -> Self {
        Self {
            fps: DEFAULT_FPS,
            start_frame: DEFAULT_START_FRAME,
            cache_budget_bytes: DEFAULT_CACHE_BUDGET_MB as u64 * 1024 * 1024,
        }
    }
}

struct CacheEntry {
    snapshot: Snapshot,
    last_used: u64,
}

/// The resources one `goto` call borrows.
struct Env<'a> {
    graph: &'a Graph,
    gpu: &'a GpuContext,
    pool: &'a mut FieldPool,
    pipelines: &'a mut PipelineCache,
    dims: FieldDims,
}

/// Owns a graph's persistent state and produces any frame on request,
/// bit-identical to what forward playback would give.
///
/// A cached snapshot for frame N is the state *entering* N, taken before N's
/// `eval`. So producing N always means restoring or reaching the state entering
/// N, then running exactly one `eval`. A cache hit therefore costs one step,
/// and outputs never need caching.
pub struct Timeline {
    config: TimelineConfig,
    state: StateStore,
    /// The frame the held state is entering, or `None` if it is not usable.
    cursor: Option<u32>,
    cache: BTreeMap<u32, CacheEntry>,
    cache_bytes: u64,
    /// Never evicted: the snapshot most recently restored.
    protected: Option<u32>,
    clock: u64,
    caching: bool,
    steps: u64,
    warning: Option<String>,
}

impl Timeline {
    pub fn new(config: TimelineConfig) -> Self {
        Self {
            config,
            state: StateStore::new(),
            cursor: None,
            cache: BTreeMap::new(),
            cache_bytes: 0,
            protected: None,
            clock: 0,
            caching: true,
            steps: 0,
            warning: None,
        }
    }

    pub fn config(&self) -> TimelineConfig {
        self.config
    }

    /// Total `eval_frame` calls made so far.
    pub fn steps_run(&self) -> u64 {
        self.steps
    }

    /// Frames whose entering state is cached, in ascending order.
    pub fn cached_frames(&self) -> Vec<u32> {
        self.cache.keys().copied().collect()
    }

    pub fn caching_enabled(&self) -> bool {
        self.caching
    }

    /// GPU memory the snapshot cache currently holds.
    pub fn cached_bytes(&self) -> u64 {
        self.cache_bytes
    }

    /// A message the user should see, at most once.
    pub fn take_warning(&mut self) -> Option<String> {
        self.warning.take()
    }

    /// Forget all state and cached frames, returning their textures to `pool`.
    pub fn reset(&mut self, pool: &mut FieldPool) {
        self.state.clear(pool);
        for (_, entry) in std::mem::take(&mut self.cache) {
            entry.snapshot.release_to(pool);
        }
        self.cache_bytes = 0;
        self.cursor = None;
        self.protected = None;
        self.caching = true;
    }

    /// Forget all state and cached frames without pooling their textures.
    /// For use after the GPU device is lost, when they are worthless.
    pub fn discard(&mut self) {
        self.state = StateStore::new();
        self.cache.clear();
        self.cache_bytes = 0;
        self.cursor = None;
        self.protected = None;
    }

    /// Produce `frame`. Frames before `start_frame` produce `start_frame`.
    ///
    /// The caller owns the returned value and should release it to `pool`.
    pub fn goto(
        &mut self,
        graph: &Graph,
        gpu: &GpuContext,
        pool: &mut FieldPool,
        pipelines: &mut PipelineCache,
        dims: FieldDims,
        frame: u32,
    ) -> Result<Evaluated, NodeError> {
        let frame = frame.max(self.config.start_frame);

        if !graph.is_stateful() {
            let time = self.time(frame);
            self.steps += 1;
            return graph.eval_frame(gpu, pool, pipelines, &mut self.state, time, dims);
        }

        let mut env = Env {
            graph,
            gpu,
            pool,
            pipelines,
            dims,
        };
        match self.advance(&mut env, frame) {
            // Stored state no longer fits the domain: start over, once.
            Err(NodeError::StateShape { .. }) => {
                self.reset(env.pool);
                self.advance(&mut env, frame)
            }
            other => other,
        }
    }

    fn time(&self, frame: u32) -> Time {
        Time::at(frame, self.config.start_frame, self.config.fps)
    }

    fn advance(&mut self, env: &mut Env<'_>, frame: u32) -> Result<Evaluated, NodeError> {
        let result = self.advance_inner(env, frame);
        if result.is_err() {
            // A failed eval may have taken state out without putting it back.
            self.state.clear(env.pool);
            self.cursor = None;
        }
        result
    }

    fn advance_inner(&mut self, env: &mut Env<'_>, frame: u32) -> Result<Evaluated, NodeError> {
        if self.cursor != Some(frame) {
            let held = self.cursor.filter(|&c| c <= frame);
            let cached = if self.caching {
                self.cache.range(..=frame).next_back().map(|(&f, _)| f)
            } else {
                None
            };
            match (held, cached) {
                (Some(h), Some(c)) if h >= c => {}
                (_, Some(c)) => self.restore(env, c)?,
                (Some(_), None) => {}
                (None, None) => {
                    self.state.clear(env.pool);
                    self.cursor = Some(self.config.start_frame);
                }
            }
            while let Some(c) = self.cursor.filter(|&c| c < frame) {
                let stepped = self.step(env, c)?;
                stepped.value.release_to(env.pool);
            }
        }
        self.step(env, frame)
    }

    fn restore(&mut self, env: &mut Env<'_>, frame: u32) -> Result<(), NodeError> {
        self.clock += 1;
        let Some(entry) = self.cache.get_mut(&frame) else {
            // Unreachable: callers pass a key they just found. Fall back to a
            // full re-simulation rather than trusting the held state.
            self.state.clear(env.pool);
            self.cursor = Some(self.config.start_frame);
            return Ok(());
        };
        entry.last_used = self.clock;
        self.state.restore(&entry.snapshot, env.gpu, env.pool)?;
        self.cursor = Some(frame);
        self.protected = Some(frame);
        Ok(())
    }

    /// Evaluate `frame` from the state entering it, caching that state first.
    fn step(&mut self, env: &mut Env<'_>, frame: u32) -> Result<Evaluated, NodeError> {
        if self.caching && !self.cache.contains_key(&frame) {
            let snapshot = self.state.snapshot(env.gpu, env.pool)?;
            self.insert(frame, snapshot, env.pool);
        }
        let time = self.time(frame);
        let out = env.graph.eval_frame(
            env.gpu,
            env.pool,
            env.pipelines,
            &mut self.state,
            time,
            env.dims,
        )?;
        self.steps += 1;
        // `None` past u32::MAX: the held state then matches no frame.
        self.cursor = frame.checked_add(1);
        Ok(out)
    }

    fn insert(&mut self, frame: u32, snapshot: Snapshot, pool: &mut FieldPool) {
        let budget = self.config.cache_budget_bytes;
        let bytes = snapshot.bytes();

        if bytes > budget {
            self.caching = false;
            self.warning = Some(format!(
                "one frame of simulation state is {bytes} bytes, more than the whole \
                 {budget}-byte cache budget; frame caching is off, so scrubbing \
                 backwards re-simulates from the start frame"
            ));
            snapshot.release_to(pool);
            for (_, entry) in std::mem::take(&mut self.cache) {
                entry.snapshot.release_to(pool);
            }
            self.cache_bytes = 0;
            return;
        }

        while self.cache_bytes + bytes > budget {
            let victim = self
                .cache
                .iter()
                .filter(|(f, _)| Some(**f) != self.protected)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(f, _)| *f);
            let Some(victim) = victim else {
                // Only the protected snapshot is left, and the new one does not fit beside it.
                snapshot.release_to(pool);
                return;
            };
            if let Some(entry) = self.cache.remove(&victim) {
                self.cache_bytes -= entry.snapshot.bytes();
                entry.snapshot.release_to(pool);
            }
        }

        self.clock += 1;
        self.cache_bytes += bytes;
        self.cache.insert(
            frame,
            CacheEntry {
                snapshot,
                last_used: self.clock,
            },
        );
    }
}
```

- [ ] **Step 4: Export it**

In `crates/elements-core/src/graph/mod.rs`, add `mod timeline;` and
`pub use timeline::{DEFAULT_CACHE_BUDGET_MB, Timeline, TimelineConfig};`.

- [ ] **Step 5: Run the tests and watch them pass**

Run: `cargo nextest run -p elements-core --test timeline`
Expected: PASS, all nine.

- [ ] **Step 6: Prove the tests can fail**

Run `cargo nextest run -p elements-core --test timeline` after each mutation, and restore after each.

- A ("step twice per frame"): in `advance_inner`, change the final `self.step(env, frame)` to
  `{ let extra = self.step(env, frame)?; extra.value.release_to(env.pool); self.step(env, frame) }`
  → `playing_forward_matches_the_closed_form` fails.
- B ("skip the restore"): in `restore`, delete the line
  `self.state.restore(&entry.snapshot, env.gpu, env.pool)?;` → `scrubbing_back...` fails.
- C ("off by one in the frame to step from"): in `restore`, change `self.cursor = Some(frame);` to
  `self.cursor = Some(frame + 1);` → `resuming_from_an_earlier_cached_frame...` fails (frame 5 is skipped).
- C2 ("no eviction"): in `insert`, change `while self.cache_bytes + bytes > budget {` to `while false {`
  → `the_cache_stays_within_budget...` fails.
- D ("snapshot after eval, not before"): in `step`, move the whole
  `if self.caching && ... { ... }` block to just after `self.steps += 1;` →
  `a_frame_is_bit_identical_however_it_is_reached` fails.
- E ("no clamp"): in `goto`, delete `let frame = frame.max(self.config.start_frame);` →
  `frames_before_the_start_are_the_start_frame` fails (frame 1 comes back as two steps).
- F ("oversize fits"): in `insert`, change `if bytes > budget {` to `if false {` →
  `a_budget_smaller_than_one_snapshot...` fails.
- G ("no retry"): in `goto`, replace the `match self.advance(...) { ... }` with
  `self.advance(&mut env, frame)` → `a_domain_change_resets_the_simulation` fails with `StateShape`.
- H ("no stateless shortcut"): in `goto`, change `if !graph.is_stateful() {` to `if false {`
  → `a_stateless_graph_is_evaluated_directly` fails (empty snapshots are cached).

- [ ] **Step 7: `just check`, then commit**

```bash
just check
git add crates/elements-core/src/graph crates/elements-core/tests/timeline.rs
git commit -F - <<'EOF'
Add a timeline that produces any frame deterministically

Blender asks for frames in any order, and a simulation only knows how
to go forwards. The timeline caches the state entering each frame, so
reaching frame N means restoring the nearest earlier state and
stepping. A cache hit costs one step, and the answer is bit-identical
to forward playback. The cache has a memory budget with LRU eviction.
A domain change resets the simulation rather than failing, and a
budget too small for even one frame turns caching off with a warning,
without refusing to run.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XCzEwqREbRMwabH11TvnMb
EOF
```

---

## Task 9: Document version 2: `fps`, `start_frame`, `cache_budget_mb`

**Why:** the timeline needs a frame rate, a start frame and a memory budget, and
they belong to the scene, so they live in the document. The version bumps
because serde ignores unknown fields: without a bump, an older engine would
accept a new document and silently simulate at the wrong rate.

**Files:**
- Modify: `crates/elements-core/src/graph/document.rs`
- Create: `tests/graphs/accumulate_4.elements` (shared stateful fixture for Tasks 10–12)
- Test: `crates/elements-core/tests/document.rs`

**Interfaces:**
- Consumes: `TimelineConfig`, `DEFAULT_CACHE_BUDGET_MB` (Task 8), `DEFAULT_FPS`, `DEFAULT_START_FRAME` (Task 4).
- Produces:
  - `ELEMENTS_DOC_VERSION = 2`. Version-1 documents load, and come back with `version == 2` and the defaults filled in.
  - `Document { pub fps: f64, pub start_frame: u32, pub cache_budget_mb: u32, .. }`
  - `Document::timeline_config(&self) -> TimelineConfig`
  - `fps` that is not finite or not `> 0` is `DocError::BadParams { kind: "document", .. }`
  - `tests/graphs/accumulate_4.elements`: constant 0.5 → accumulate → output, 4³, 24 fps, start frame 1

- [ ] **Step 1: Write the failing tests**

In `crates/elements-core/tests/document.rs`, add `TimelineConfig` to the `use elements_core::graph::{...}` list, and append:

```rust
/// The migration test `ELEMENTS_DOC_VERSION` requires: a version-1 document
/// loads, and gets the time settings version 1 implied.
#[test]
fn a_version_1_document_migrates_with_time_defaults() {
    let doc = Document::from_json(MINIMAL).unwrap();
    assert_eq!(doc.version, 2);
    assert_eq!(doc.fps, 24.0);
    assert_eq!(doc.start_frame, 1);
    assert_eq!(doc.cache_budget_mb, 2048);
}

#[test]
fn version_2_time_fields_reach_the_timeline_config() {
    let v2 = MINIMAL.replace(
        "\"version\": 1,",
        "\"version\": 2, \"fps\": 30.0, \"start_frame\": 1001, \"cache_budget_mb\": 64,",
    );
    let doc = Document::from_json(&v2).unwrap();
    assert_eq!(
        doc.timeline_config(),
        TimelineConfig {
            fps: 30.0,
            start_frame: 1001,
            cache_budget_bytes: 64 * 1024 * 1024,
        }
    );
}

#[test]
fn rejects_a_non_positive_fps() {
    for fps in ["0.0", "-24.0"] {
        let bad = MINIMAL.replace(
            "\"version\": 1,",
            &format!("\"version\": 2, \"fps\": {fps},"),
        );
        match Document::from_json(&bad) {
            Err(DocError::BadParams { reason, .. }) => assert!(reason.contains("fps"), "{reason}"),
            other => panic!("fps {fps}: expected BadParams, got {other:?}"),
        }
    }
}
```

In the same file, both `Document { ... }` literals (in
`accepts_a_document_wiring_one_output_to_two_inputs` and the out-of-range-output
test) need the three new fields added after `dims`:

```rust
        fps: 24.0,
        start_frame: 1,
        cache_budget_mb: 2048,
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cargo nextest run -p elements-core --test document`
Expected: FAIL to compile: `no field fps on type Document`.

- [ ] **Step 3: Implement version 2**

In `crates/elements-core/src/graph/document.rs`:

Add imports:

```rust
use super::time::{DEFAULT_FPS, DEFAULT_START_FRAME};
use super::timeline::{DEFAULT_CACHE_BUDGET_MB, TimelineConfig};
```

Replace the version constant and its comment:

```rust
/// The `.elements` schema version this build writes.
///
/// It also reads version 1, which predates time: see `Document::from_json`.
/// Bumping this requires a migration test in `tests/document.rs`.
pub const ELEMENTS_DOC_VERSION: u32 = 2;

fn default_fps() -> f64 {
    DEFAULT_FPS
}

fn default_start_frame() -> u32 {
    DEFAULT_START_FRAME
}

fn default_cache_budget_mb() -> u32 {
    DEFAULT_CACHE_BUDGET_MB
}
```

Add three fields to `Document`, after `dims`:

```rust
    /// Frames per second. Version-1 documents have none and get 24.
    #[serde(default = "default_fps")]
    pub fps: f64,
    /// The first frame of the simulation. Earlier frames produce this one.
    #[serde(default = "default_start_frame")]
    pub start_frame: u32,
    /// GPU memory the frame cache may hold, in MiB.
    #[serde(default = "default_cache_budget_mb")]
    pub cache_budget_mb: u32,
```

Replace `from_json` with:

```rust
    pub fn from_json(text: &str) -> Result<Self, DocError> {
        let mut doc: Document = serde_json::from_str(text)?;
        match doc.version {
            // Version 1 predates time. Its only migration is the defaults serde
            // has already filled in above.
            1 => doc.version = ELEMENTS_DOC_VERSION,
            ELEMENTS_DOC_VERSION => {}
            // Deliberately exact, not a range: a newer document may rely on
            // fields this build would silently ignore.
            other => return Err(DocError::UnsupportedVersion(other)),
        }
        if !(doc.fps.is_finite() && doc.fps > 0.0) {
            return Err(DocError::BadParams {
                kind: "document".to_string(),
                reason: format!("fps must be a positive, finite number, got {}", doc.fps),
            });
        }
        Ok(doc)
    }

    /// How a timeline for this document should run.
    pub fn timeline_config(&self) -> TimelineConfig {
        TimelineConfig {
            fps: self.fps,
            start_frame: self.start_frame,
            cache_budget_bytes: self.cache_budget_mb as u64 * 1024 * 1024,
        }
    }
```

- [ ] **Step 4: Create the shared fixture**

Create `tests/graphs/accumulate_4.elements`:

```json
{
  "version": 2,
  "dims": [4, 4, 4],
  "fps": 24.0,
  "start_frame": 1,
  "nodes": [
    { "id": 0, "kind": "core.constant_field", "params": { "value": 0.5 } },
    { "id": 1, "kind": "core.accumulate", "params": {} },
    { "id": 2, "kind": "core.output", "params": {} }
  ],
  "edges": [
    { "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 },
    { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 0 }
  ],
  "output": 2
}
```

Its output at frame N is `0.5 * N / 24` for every voxel.

- [ ] **Step 5: Run the tests and watch them pass**

Run: `cargo nextest run --workspace`
Expected: PASS. Every existing `"version": 1` document in the tests still loads.

- [ ] **Step 6: Prove the tests can fail**

- A: in `from_json`, delete the `1 => ...` arm → `a_version_1_document_migrates...` fails (`UnsupportedVersion(1)`). Restore.
- B: in `timeline_config`, change `fps: self.fps` to `fps: DEFAULT_FPS` → `version_2_time_fields...` fails. Restore.
- C: delete the `if !(doc.fps.is_finite() ...)` block → `rejects_a_non_positive_fps` fails. Restore.

- [ ] **Step 7: `just check`, then commit**

```bash
just check
git add crates/elements-core/src/graph/document.rs crates/elements-core/tests/document.rs \
        tests/graphs/accumulate_4.elements
git commit -F - <<'EOF'
Add frame rate, start frame and cache budget to the document format

A timeline needs them, and they belong to the scene. The version goes to
2 even though the fields are optional, because serde ignores unknown
fields: an older engine would otherwise accept a new document and
simulate at the wrong rate without saying so. Version-1 documents still
load, with the defaults they always implied.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XCzEwqREbRMwabH11TvnMb
EOF
```

---

## Task 10: The daemon renders through the timeline

**Why:** `Command::Render { frame }` has carried a frame number since Core v1,
and the daemon has ignored it. This task makes it produce that frame.

**Files:**
- Modify: `crates/elementsd/src/daemon.rs`
- Test: `crates/elementsd/tests/session.rs`

**Interfaces:**
- Consumes: `Timeline`, `Document::timeline_config` (Tasks 8–9), `Value::release_to`.
- Produces: `Render { frame }` returns frame `frame` of the loaded graph.
  Loading a document discards the previous timeline's state and cache. A lost
  device discards them without pooling. Timeline warnings go to stderr, never
  stdout: stdout carries the readiness line.

- [ ] **Step 1: Write the failing test**

Append to `crates/elementsd/tests/session.rs`:

```rust
fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/elementsd is two levels below the root")
        .to_path_buf()
}

/// Render `frame` and read the published values back from the channel.
fn render_values(
    stream: &mut elements_ipc::Stream,
    reader: &mut BufReader<elements_ipc::Stream>,
    channel: &std::path::Path,
    frame: u32,
) -> Vec<f32> {
    write_message(stream, &Command::Render { frame }).unwrap();
    match read_message::<_, Response>(reader).unwrap().unwrap() {
        Response::Frame { .. } => {}
        other => panic!("expected Frame, got {other:?}"),
    }
    let mut frames = FrameReader::open(channel).unwrap();
    frames.read_latest().unwrap().1
}

#[test]
fn render_produces_the_requested_frame_of_a_stateful_graph() {
    let daemon = Daemon::start();
    let graph = repo_root().join("tests/graphs/accumulate_4.elements");

    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    write_message(
        &mut stream,
        &Command::Hello {
            protocol_version: ELEMENTS_PROTOCOL_VERSION,
        },
    )
    .unwrap();
    let _: Response = read_message(&mut reader).unwrap().unwrap();
    write_message(
        &mut stream,
        &Command::LoadGraph {
            path: graph.to_string_lossy().into_owned(),
        },
    )
    .unwrap();
    assert!(matches!(
        read_message::<_, Response>(&mut reader).unwrap().unwrap(),
        Response::Loaded { .. }
    ));

    let near = |values: &[f32], frame: u32| {
        let want = 0.5 * frame as f32 / 24.0;
        assert!(
            values.iter().all(|v| (v - want).abs() < 1e-6),
            "frame {frame}: expected {want}, got {:?}",
            &values[..4]
        );
    };

    let three = render_values(&mut stream, &mut reader, &daemon.channel, 3);
    near(&three, 3);
    let one = render_values(&mut stream, &mut reader, &daemon.channel, 1);
    near(&one, 1);
    let three_again = render_values(&mut stream, &mut reader, &daemon.channel, 3);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&three_again), bits(&three), "frame 3 must be bit-identical on revisit");
}
```

The `elements_ipc::Stream` type is what `Daemon::connect` returns. If `BufReader`'s
type parameter does not match, copy the exact type `read_message` is called with
elsewhere in this file.

- [ ] **Step 2: Run it and watch it fail**

Run: `cargo nextest run -p elementsd --test session render_produces`
Expected: FAIL. `frame 3: expected 0.0625, got [0.020833334, ...]`: the daemon ignores the frame number.

- [ ] **Step 3: Give the session a timeline**

In `crates/elementsd/src/daemon.rs`:

Change the graph import to `use elements_core::graph::{Document, Graph, NodeRegistry, Timeline};`.

Add a field to `Session` after `graph`:

```rust
    /// Time and simulation state for the loaded graph.
    timeline: Option<Timeline>,
```

and initialise it with `timeline: None,` in `Session::new`.

In `handle`, change the render arm to pass the frame through:

```rust
        Command::Render { frame } => {
            if let Err(e) = require_greeted(session) {
                return Response::Error(e);
            }
            match render(session, frame) {
                Ok(response) => response,
                Err(e) => Response::Error(e),
            }
        }
```

In `load`, capture the config before the document is consumed. Replace

```rust
    let (graph, dims) = doc
        .into_graph(&session.registry)
```

with

```rust
    let config = doc.timeline_config();
    let (graph, dims) = doc
        .into_graph(&session.registry)
```

and replace the tail, from `session.graph = Some((graph, dims));` down to the
`Ok(Response::Loaded {`, with:

```rust
    // A new document invalidates every frame the old one simulated. Its
    // textures go back to the pool, since the device is still good.
    if let Some(mut old) = session.timeline.take() {
        old.reset(&mut session.pool);
    }
    session.graph = Some((graph, dims));
    session.timeline = Some(Timeline::new(config));
```

Replace `render` with:

```rust
fn render(session: &mut Session, frame: u32) -> Result<Response, EngineError> {
    let no_graph = || EngineError::new(ErrorKind::Graph, "no graph is loaded");
    let (graph, dims) = session.graph.as_ref().ok_or_else(no_graph)?;
    let dims = *dims;
    let timeline = session.timeline.as_mut().ok_or_else(no_graph)?;

    let evaluated = match timeline.goto(
        graph,
        &session.gpu,
        &mut session.pool,
        &mut session.pipelines,
        dims,
        frame,
    ) {
        Ok(evaluated) => evaluated,
        Err(e) => {
            let err = map_node_error(e);
            if err.kind == ErrorKind::DeviceLost {
                timeline.discard();
            }
            return Err(err);
        }
    };
    if let Some(warning) = timeline.take_warning() {
        // stderr, never stdout: stdout carries the readiness line.
        eprintln!("elementsd: warning: {warning}");
    }

    let values = match evaluated.value.as_field() {
        Ok(field) => field.read_back(&session.gpu).map_err(|e| {
            let kind = if session.gpu.device_lost().is_some() {
                ErrorKind::DeviceLost
            } else {
                ErrorKind::Gpu
            };
            EngineError::new(kind, e.to_string())
        }),
        Err(e) => Err(map_node_error(e)),
    };
    evaluated.value.release_to(&mut session.pool);
    let values = match values {
        Ok(values) => values,
        Err(e) => {
            if e.kind == ErrorKind::DeviceLost {
                timeline.discard();
            }
            return Err(e);
        }
    };

    let writer = session
        .writer
        .as_mut()
        .ok_or_else(|| EngineError::new(ErrorKind::Io, "no frame channel"))?;
    let seq = writer
        .publish(&values)
        .map_err(|e| EngineError::new(ErrorKind::Io, e.to_string()))?;

    Ok(Response::Frame {
        seq,
        channel: session.channel_path.to_string_lossy().into_owned(),
        dims: [dims.x, dims.y, dims.z],
    })
}
```

- [ ] **Step 4: Run the daemon tests**

Run: `cargo nextest run -p elementsd`
Expected: PASS, all tests, including `repeated_renders_advance_the_sequence` (a stateless graph).

- [ ] **Step 5: Prove the test can fail**

Mutation ("ignore the frame", the Core v1 behaviour): in `handle`, change
`match render(session, frame)` to `match render(session, 1)`. Run the test
→ FAIL at frame 3. Restore.

- [ ] **Step 6: `just check`, then commit**

```bash
just check
git add crates/elementsd/src/daemon.rs crates/elementsd/tests/session.rs
git commit -F - <<'EOF'
Render the requested frame through a timeline in the daemon

Render has carried a frame number since Core v1 and the daemon ignored
it, which was harmless only while nothing had state. Each loaded
document now gets its own timeline, and loading another returns the old
one's textures to the pool. A lost device discards the timeline without
pooling, since its textures are worthless.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XCzEwqREbRMwabH11TvnMb
EOF
```

---

## Task 11: The CLI bakes through the timeline

**Why:** `elements bake --frames 1-100` wrote the same field a hundred times.
Stateful graphs need each file to be the true frame, including when a bake starts
partway through (`--frames 50-100` must match frames 50–100 of a full bake).

**Files:**
- Modify: `crates/elements-cli/src/bake.rs`
- Test: `crates/elements-cli/tests/cli.rs`

**Interfaces:**
- Consumes: `Timeline`, `Document::timeline_config`.
- Produces: `bake(graph, out_dir, (a, b), name, voxel_size)` drives the timeline
  from the document's `start_frame` and writes frames `a..=b` only.
  `evaluate_document` (used by `render-preview` and `dump-npy`) is unchanged: it
  always evaluates the first frame.

- [ ] **Step 1: Write the failing tests**

Append to `crates/elements-cli/tests/cli.rs`:

```rust
fn accumulate_fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/elements-cli is two levels below the root")
        .join("tests/graphs/accumulate_4.elements")
}

fn bake(out: &std::path::Path, frames: &str) {
    let status = cli()
        .args(["bake", accumulate_fixture().to_str().unwrap()])
        .args(["--out", out.to_str().unwrap()])
        .args(["--frames", frames])
        .status()
        .unwrap();
    assert!(status.success(), "bake --frames {frames} failed");
}

#[test]
fn bake_steps_a_stateful_graph_frame_by_frame() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("range");
    bake(&out, "1-3");

    let files: Vec<Vec<u8>> = (1..=3)
        .map(|f| std::fs::read(out.join(format!("density.{f:04}.vdb"))).unwrap())
        .collect();
    assert_ne!(files[0], files[1]);
    assert_ne!(files[1], files[2]);
}

/// The VDB writer is deterministic (fixed UUID, no timestamps), so equal
/// bytes mean equal fields.
#[test]
fn baking_one_frame_gives_the_true_frame_not_the_first_step() {
    let dir = tempfile::tempdir().unwrap();
    let range = dir.path().join("range");
    let single = dir.path().join("single");
    bake(&range, "1-3");
    bake(&single, "3");

    assert_eq!(
        std::fs::read(single.join("density.0003.vdb")).unwrap(),
        std::fs::read(range.join("density.0003.vdb")).unwrap()
    );
}
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cargo nextest run -p elements-cli --test cli`
Expected: `bake_steps_a_stateful_graph_frame_by_frame` FAILS (the three files are identical).

- [ ] **Step 3: Bake through the timeline**

In `crates/elements-cli/src/bake.rs`, change the graph import to
`use elements_core::graph::{Document, NodeRegistry, Timeline};` and replace `bake`
(keep its doc comment, and append the paragraph below to it):

```rust
///
/// Frames are produced by a timeline, so a stateful graph is simulated from
/// the document's start frame even when `frames` starts later. Only frames in
/// `frames` are written.
pub fn bake(
    graph: &Path,
    out_dir: &Path,
    frames: (u32, u32),
    name: &str,
    voxel_size: f64,
) -> anyhow::Result<()> {
    let text =
        std::fs::read_to_string(graph).with_context(|| format!("reading {}", graph.display()))?;
    let doc = Document::from_json(&text)?;
    let config = doc.timeline_config();
    let registry = NodeRegistry::with_builtins();
    let (graph, dims) = doc.into_graph(&registry)?;

    let gpu = GpuContext::new_headless().context("acquiring a GPU device")?;
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut timeline = Timeline::new(config);

    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    for frame in frames.0..=frames.1 {
        let evaluated = timeline.goto(&graph, &gpu, &mut pool, &mut pipelines, dims, frame)?;
        if let Some(warning) = timeline.take_warning() {
            eprintln!("warning: {warning}");
        }
        let values = evaluated.value.as_field()?.read_back(&gpu)?;
        evaluated.value.release_to(&mut pool);

        let path = out_dir.join(format!("{name}.{frame:04}.vdb"));
        elements_io::write_float_grid(&path, name, &values, [dims.x, dims.y, dims.z], voxel_size, 0.0)
            .with_context(|| format!("writing {}", path.display()))?;
    }

    Ok(())
}
```

- [ ] **Step 4: Run the CLI tests**

Run: `cargo nextest run -p elements-cli`
Expected: PASS, including the pre-existing tests that bake a stateless noise graph (`1-3` and `9999-10001`).

- [ ] **Step 5: Prove the tests can fail**

- A: in `bake`, change `for frame in frames.0..=frames.1 {` so the goto uses a fixed
  frame: `timeline.goto(&graph, &gpu, &mut pool, &mut pipelines, dims, frames.0)?`
  → `bake_steps_a_stateful_graph...` fails. Restore.
- B ("start the sim at A"): change `let mut timeline = Timeline::new(config);` to
  `let mut timeline = Timeline::new(elements_core::graph::TimelineConfig { start_frame: frames.0, ..config });`
  → `baking_one_frame_gives_the_true_frame...` fails. Restore.

- [ ] **Step 6: `just check`, then commit**

```bash
just check
git add crates/elements-cli/src/bake.rs crates/elements-cli/tests/cli.rs
git commit -F - <<'EOF'
Bake each frame through the timeline in the CLI

The bake loop wrote one evaluation to every frame's file, a placeholder
that only worked while nothing had state. It now drives a timeline from
the document's start frame, so a bake that begins partway through a
shot produces the same frames as a full one.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XCzEwqREbRMwabH11TvnMb
EOF
```

---

## Task 12: The add-on bakes the scene's frame

**Why:** the viewport's geometry comes from a CLI bake, not from the daemon
(`handlers.py`, KNOWN GAP), and that bake hard-codes `--frames 1`. With a
stateful graph, Blender would show frame 1 while the daemon simulated frame N.
Each viewport update now re-simulates from the start frame in a fresh process.
That is acceptable for piece 1's toy graphs, and piece 3's in-memory handoff removes it.

**Files:**
- Create: `addon/blender_elements/bakecmd.py`
- Modify: `addon/blender_elements/handlers.py`, `addon/blender_elements/ops.py`, `addon/blender_elements/client.py`
- Modify: `tests/python/contract.py`, `crates/elementsd/tests/python_contract.rs`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `tests/graphs/accumulate_4.elements` (Task 9), the daemon (Task 10), the CLI (Task 11).
- Produces:
  - `bakecmd.bake_command(cli: str, graph_path: str, out_dir: str, frame: int) -> tuple[list[str], str]`:
    the argv and the path of the file it writes. Negative frames clamp to 0.
  - `handlers.push_frame_to_volume(context, values, dims, frame)`: now takes the frame.
  - `ControlClient.render(frame)` clamps negative frames to 0 before sending.

- [ ] **Step 1: Write the failing contract checks**

In `tests/python/contract.py`:

Add `import os` to the imports, and after the existing `from blender_elements.client import (...)` add:

```python
from blender_elements.bakecmd import bake_command  # noqa: E402
```

Add this function next to `check_truncated_channel_file_rejected`:

```python
def check_bake_command() -> None:
    """The viewport bake must ask for the scene's frame, and read back the file
    the CLI actually names. Blender allows negative frames; the engine does not."""
    args, path = bake_command("elements", "g.elements", "out", 7)
    assert args[args.index("--frames") + 1] == "7", args
    assert path == os.path.join("out", "density.0007.vdb"), path

    args, path = bake_command("elements", "g.elements", "out", -3)
    assert args[args.index("--frames") + 1] == "0", args
    assert path == os.path.join("out", "density.0000.vdb"), path
```

Change `main` to take a fourth argument, `stateful_graph: str`. Call
`check_bake_command()` right after `check_truncated_channel_file_rejected()`, and
append this inside the `with ControlClient(endpoint) as client:` block, after
`assert client.load_graph(graph)["nodes"] == 2`:

```python
        # A stateful graph: the frame number must reach the timeline, and a
        # negative Blender frame must clamp rather than fail.
        loaded_stateful = client.load_graph(stateful_graph)
        assert loaded_stateful["dims"] == [4, 4, 4], loaded_stateful
        stateful_reader = FrameReader(channel)
        try:
            for frame in (3, 1, -5):
                client.render(frame)
                _, values = stateful_reader.read_latest()
                want = 0.5 * max(frame, 1) / 24.0
                assert all(abs(v - want) < 1e-6 for v in values), (frame, values[:4])
        finally:
            stateful_reader.close()
```

Change the entry point to `main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])`.

In `crates/elementsd/tests/python_contract.rs`, pass the fixture as the fourth
argument, after `.arg(graph.to_str().unwrap())`:

```rust
        .arg(repo_root().join("tests/graphs/accumulate_4.elements"))
```

- [ ] **Step 2: Run the contract test and watch it fail**

Run: `cargo nextest run -p elementsd --test python_contract`
Expected: FAIL. `ModuleNotFoundError: No module named 'blender_elements.bakecmd'`.

- [ ] **Step 3: Create `addon/blender_elements/bakecmd.py`**

```python
"""The engine CLI's bake command line, built without `bpy`.

Kept apart from `handlers.py`, which imports `bpy`, so the contract test can
exercise it outside Blender.
"""

import os


def bake_command(cli: str, graph_path: str, out_dir: str, frame: int) -> tuple[list[str], str]:
    """Return the argv that bakes `frame` of `graph_path`, and the file it writes.

    Blender frames may be negative and the engine's may not, so negative frames
    clamp to 0. The engine clamps again, up to the document's start frame.
    The CLI zero-pads frame numbers to at least four digits, which `:04d`
    matches exactly, including past frame 9999.
    """
    frame = max(0, int(frame))
    args = [
        cli,
        "bake",
        graph_path,
        "--out",
        out_dir,
        "--frames",
        str(frame),
        "--name",
        "density",
    ]
    return args, os.path.join(out_dir, f"density.{frame:04d}.vdb")
```

- [ ] **Step 4: Clamp negative frames in the client**

In `addon/blender_elements/client.py`, replace the body of `ControlClient.render` with:

```python
        # `Command::Render { frame: u32 }`: Blender allows negative frames, and a
        # negative number would be rejected as a malformed command. The engine
        # clamps anything below the document's start frame to the start frame,
        # so 0 behaves the same.
        return self._round_trip({"type": "render", "frame": max(0, int(frame))})
```

- [ ] **Step 5: Bake the scene's frame**

In `addon/blender_elements/handlers.py`:

Add `from .bakecmd import bake_command` below `from .client import ElementsError`.

Change `push_frame_to_volume`'s signature to
`def push_frame_to_volume(context, values, dims, frame: int) -> bpy.types.Object:`
and its bake call to `_bake_current_graph_to(path, frame)`. Add a sentence to its
docstring: "`frame` is the scene frame the volume must show. The CLI bake
re-simulates up to it."

Change `_bake_current_graph_to(path)` to `_bake_current_graph_to(path, frame: int)`.
Inside it, replace the `subprocess.run([...], ...)` argument list and the final
`os.replace(...)` so they use the builder:

```python
    args, baked = bake_command(cli, bpy.path.abspath(settings.graph_path), out_dir, frame)
```

```python
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=60,
        )
```

```python
    os.replace(baked, path)
```

Update the 60-second comment: a bake of a stateful graph now re-simulates from
the start frame, so the timeout bounds the whole run up to `frame`, not one evaluation.

In `_on_draw`, change the call to
`push_frame_to_volume(bpy.context, values, frame["dims"], scene.frame_current)`.

In `addon/blender_elements/ops.py`, in `ELEMENTS_OT_render_frame.execute`, change the call to
`push_frame_to_volume(context, values, frame["dims"], context.scene.frame_current)`.

- [ ] **Step 6: Run the contract test, the add-on build and the Blender test**

```bash
cargo nextest run -p elementsd --test python_contract
just addon
python -c "import zipfile,glob; ns=zipfile.ZipFile(sorted(glob.glob('dist/*.zip'))[-1]).namelist(); assert '__init__.py' in ns and 'blender_manifest.toml' in ns and 'bakecmd.py' in ns, ns"
just blender-test
```

Expected: all pass. The ZIP has `bakecmd.py` at its root beside `__init__.py`. The
Blender round trip still renders frame 1, the scene default.

- [ ] **Step 7: Prove the checks can fail**

- A: in `bake_command`, change `str(frame)` to `"1"` → the contract test fails in `check_bake_command`. Restore.
- B: in `client.py`, change `max(0, int(frame))` to `int(frame)` → the contract test fails at frame `-5`
  (an `ElementsError` from the daemon rejecting the command). Restore.

- [ ] **Step 8: Point CLAUDE.md at Ember**

In `CLAUDE.md`, replace the paragraph beginning "The suite is being built product by product." and its three bullet links with:

```markdown
The suite is being built product by product. **Core v1** is merged. **Ember**
(grid gas) is in progress in four pieces; piece 1, core sim foundations, adds
persistent state, a timeline with a frame cache, staggered vector fields, and
multi-consumer graph outputs.

- Core design spec: `docs/superpowers/specs/2026-09-19-elements-suite-core-design.md`
- Core v1 plan: `docs/superpowers/plans/2026-09-19-elements-core-v1.md`
- Ember umbrella spec: `docs/superpowers/specs/2026-09-21-ember-design.md`
- Ember piece 1 spec: `docs/superpowers/specs/2026-09-21-ember-core-sim-foundations-design.md`
- Ember piece 1 plan: `docs/superpowers/plans/2026-09-21-ember-core-sim-foundations.md`
- Live progress and open risks: `.superpowers/sdd/progress.md`
```

Also, under "Constraints that bite", add:

```markdown
- **`GpuContext` requests the adapter's resolution limits** on top of
  `downlevel_defaults`, which otherwise caps 3D textures at 256. Everything else
  in `downlevel_defaults` still applies. Notably, **4 storage textures per shader
  stage**: read fields through `texture_3d<f32>` + `textureLoad`, and write through storage.
- **A snapshot for frame N is the state entering N.** Producing N is always
  "restore or reach the entering state, then one `eval`". Never cache outputs.
```

- [ ] **Step 9: `just check`, then commit**

```bash
just check
git add addon/blender_elements tests/python/contract.py crates/elementsd/tests/python_contract.rs CLAUDE.md
git commit -F - <<'EOF'
Bake the scene's current frame for the viewport volume

The viewport's geometry comes from a CLI bake rather than the daemon,
and that bake always asked for frame 1. With a stateful graph, Blender
would have shown frame 1 while the daemon simulated the frame the user
was on. The bake command is built in a bpy-free module so the contract
test can check it outside Blender. Negative Blender frames are clamped,
because the protocol's frame is unsigned.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XCzEwqREbRMwabH11TvnMb
EOF
```

---

## Definition of done for piece 1

- [ ] `just check` passes on `main` after the last task.
- [ ] Every test added by this plan has a recorded single mutation that makes it fail, with the real output, in its task report.
- [ ] `just addon` builds a ZIP with `__init__.py`, `blender_manifest.toml` and `bakecmd.py` at the root and nothing nested.
- [ ] `just blender-test` passes.
- [ ] **By hand, in an interactive Blender session:** load
      `tests/graphs/accumulate_4.elements`, render at frames 1, 12 and 24, and
      confirm the volume's density rises. Then scrub back to 1 and confirm it
      falls again. This is the only check of the add-on's viewport path with
      a stateful graph.
- [ ] **Do not claim cross-backend determinism.** Bit-exactness is verified on Metal only. CI has still never run (Core v1 close-out item 1).

## Notes for piece 2 (the solver)

- Solver kernels read fields through `texture_3d<f32>` + `textureLoad` and write
  through `texture_storage_3d<r32float, write>`. `downlevel_defaults` allows 4
  storage textures per stage, which is not enough to bind six velocity faces as storage.
- Use `acquire_zeroed` / `acquire_vector_zeroed` for anything an emitter writes only partly.
- Prepend `vector_sample.wgsl` to every advection kernel. Do not re-derive the face offsets.
- The solver node is stateful, with slots for density, temperature, flame (fields) and velocity (a vector field).
- A branching graph is now legal. Prefer `input()` over `take_input()` in solver nodes, so no hidden copies are made.
- Reading back a 512³ `R32Float` field (512 MiB) exceeds `max_buffer_size` (256 MiB): piece 3's export must read back in z-slabs.
