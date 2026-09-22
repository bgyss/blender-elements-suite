# Core Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let documents use the adapter's full buffer limit, stop out-of-memory and stray GPU errors from panicking the daemon, and pin the solver's density/temperature wiring with a mass test.

**Architecture:** Two pure functions in `elements-core::gpu` carry the decisions: `required_limits`, and the error-precedence resolver `scoped` uses. Each can be unit-tested without a GPU. An uncaptured-error handler is the safety net for errors outside any scope. The daemon maps the new out-of-memory error to its own wire kind and frees GPU memory. The add-on's reaction lives in a bpy-free module the Python contract test can call.

**Tech Stack:** Rust 2024, wgpu 30.0.1, Python 3.11 stdlib (add-on).

**Spec:** `docs/superpowers/specs/2026-09-22-hardening-design.md`

---

## Global Constraints

- **Branch:** `hardening`, from `main`. Push after every task; CI runs on every push. Never push to `main`.
- **`required_features` stays `wgpu::Features::empty()`.** Raising a limit is allowed.
- **`#![forbid(unsafe_code)]`** on every crate except `elements-ipc`.
- **The engine never panics on untrusted input or on a GPU error.**
- **Never set `WGPU_BACKEND` locally.**
- **`just check` must pass before every commit.**
- **Prove each new test can fail** with one single-change mutation. Paste the real failing output into the report. Restore by editing the line back in place (`git checkout --` is blocked by a hook), confirm with `git diff`, and make sure every pasted FAIL and the final GREEN come from a fresh build. If a mutation does not make its test fail, report it; do not rewrite the assertion to force it.
- **Python targets 3.11**, is linted with `ruff`, and add-on code uses relative imports only.
- **Commit style (CLAUDE.md):** an imperative subject, then a body explaining *why*. End with:
  ```
  Co-Authored-By: Claude <model> <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TSRpZQEY9fnKQqHLh5WHpU
  ```
  where `<model>` is the model actually implementing. Commit with plain `git`; do not run `jj`.

---

### Task 1: Request the adapter's buffer limit

**Files:**
- Modify: `crates/elements-core/src/gpu/mod.rs`
- Modify: comments that state the 256 MiB cap as fixed: `crates/elements-core/src/graph/document.rs` (`validate_for` doc), `crates/elements-cli/src/bake.rs:~20`, `crates/elementsd/src/daemon.rs:~125`
- Modify: `crates/elementsd/tests/session.rs` (`a_512_cubed_document_is_rejected_at_load_with_the_size_named`)
- Test: `crates/elements-core/tests/gpu_context.rs`

**Interfaces:**
- Produces: `elements_core::gpu::required_limits(adapter: &wgpu::Limits) -> wgpu::Limits`.

- [ ] **Step 1: Write the failing test.** Append to `crates/elements-core/tests/gpu_context.rs`:

```rust
/// Only the resolution limits and `max_buffer_size` may come from the adapter.
/// Everything else stays at `downlevel_defaults`, so kernels keep running on
/// every backend, in particular within 4 storage textures per stage.
#[test]
fn required_limits_take_only_resolution_and_buffer_size_from_the_adapter() {
    let mut adapter = wgpu::Limits::downlevel_defaults();
    adapter.max_buffer_size = 4 << 30;
    adapter.max_texture_dimension_3d = 2048;
    adapter.max_storage_textures_per_shader_stage = 16;

    let got = elements_core::gpu::required_limits(&adapter);
    let base = wgpu::Limits::downlevel_defaults();
    assert_eq!(got.max_buffer_size, 4 << 30);
    assert_eq!(got.max_texture_dimension_3d, 2048);
    assert_eq!(
        got.max_storage_textures_per_shader_stage,
        base.max_storage_textures_per_shader_stage
    );
}
```

- [ ] **Step 2: Run it and confirm it fails** (it won't compile: `required_limits` doesn't exist): `cargo nextest run -p elements-core --test gpu_context`.

- [ ] **Step 3: Implement.** In `crates/elements-core/src/gpu/mod.rs`, add:

```rust
/// The limits Elements asks a device for, given what its adapter supports.
///
/// Everything is `downlevel_defaults` except two things taken from the adapter.
/// The resolution limits, because `downlevel_defaults` caps 3D textures at 256,
/// which cannot hold a staggered face of a 256³ domain. And `max_buffer_size`,
/// because at 256 MiB it rejects any domain above about 406³, and read-back of
/// a 512³ field needs 512 MiB. Both are limits, not features, so
/// `required_features` stays empty and the portability guarantee is unchanged.
pub fn required_limits(adapter: &wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_buffer_size: adapter.max_buffer_size,
        ..wgpu::Limits::downlevel_defaults().using_resolution(adapter.clone())
    }
}
```

In `new_headless_async`, replace the `required_limits` line and its comment with `let required_limits = required_limits(&adapter.limits());`.

- [ ] **Step 4: Update the daemon test.** In `a_512_cubed_document_is_rejected_at_load_with_the_size_named`:
  - rename it to `a_document_over_the_buffer_limit_is_rejected_at_load_with_the_size_named`;
  - change the dims to `[2047, 2047, 2047]` (about 34 GB as `R32Float`: over any adapter seen, and still under the 2048 3D-texture limit, so only the byte cap can reject it);
  - change the assertion to `e.message.contains("2047")`;
  - rewrite its doc comment to say this;
  - keep "Do not render such a graph in any test".

Also update the three comments listed under **Files** so none states the 256 MiB cap as fixed: say the limit is the adapter's `max_buffer_size`.

- [ ] **Step 5: Run the tests and `just check`.** `cargo nextest run -p elements-core --test gpu_context -p elementsd`, then `just check`.

- [ ] **Step 6: Prove the test can fail.** Mutation: delete the `max_buffer_size: adapter.max_buffer_size,` line in `required_limits`.

- [ ] **Step 7: Commit and push.** Subject: `Request the adapter's buffer limit`. After pushing, check the CI run with `gh run list --branch hardening --limit 1` and `gh run view <id> --log | grep -E 'deviceName|Summary|FAIL'`. If the 2047³ test fails on llvmpipe, llvmpipe reports a limit of at least 34 GB. Report that; don't weaken the test.

---

### Task 2: GPU errors never panic the daemon

**Files:**
- Modify: `crates/elements-core/src/gpu/mod.rs`, `crates/elements-core/src/gpu/pool.rs`
- Modify: `crates/elements-ipc/src/protocol.rs` (the `ErrorKind` variant; `ELEMENTS_PROTOCOL_VERSION` 1 → 2)
- Modify: `crates/elementsd/src/daemon.rs`
- Create: `addon/blender_elements/errors.py`
- Modify: `addon/blender_elements/client.py` (`PROTOCOL_VERSION = 2`), `addon/blender_elements/ops.py`, `addon/blender_elements/handlers.py`
- Modify: `tests/python/contract.py`
- Test: `crates/elements-core/tests/gpu_context.rs`, `crates/elements-core/tests/field_pool.rs`, the ipc crate's protocol tests, and the daemon's unit tests in `daemon.rs`

**Interfaces:**
- Produces: `GpuError::OutOfMemory(String)`, `GpuError::Internal(String)`; `elements_core::gpu::resolve_errors(lost: Option<String>, out_of_memory: Option<wgpu::Error>, internal: Option<wgpu::Error>, validation: Option<wgpu::Error>, stray: Option<GpuError>) -> Option<GpuError>`; `FieldPool::clear(&mut self)`; `elements_ipc::ErrorKind::OutOfMemory` (wire name `out_of_memory`); Python `errors.describe(kind: str, message: str) -> tuple[str, bool]`.

- [ ] **Step 1: Write the failing core tests.** Append to `crates/elements-core/tests/gpu_context.rs`:

```rust
fn source() -> wgpu::ErrorSource {
    Box::new(std::io::Error::other("test"))
}

fn oom() -> wgpu::Error {
    wgpu::Error::OutOfMemory { source: source() }
}

fn validation() -> wgpu::Error {
    wgpu::Error::Validation { source: source(), description: "bad".into() }
}

/// An out-of-memory error must reach the caller as `OutOfMemory`, not be
/// folded into `Validation`, and must win over a validation error captured
/// alongside it: it is the one that says why.
#[test]
fn out_of_memory_is_reported_as_itself_and_outranks_validation() {
    use elements_core::gpu::resolve_errors;
    assert!(matches!(
        resolve_errors(None, Some(oom()), None, None, None),
        Some(GpuError::OutOfMemory(_))
    ));
    assert!(matches!(
        resolve_errors(None, Some(oom()), None, Some(validation()), None),
        Some(GpuError::OutOfMemory(_))
    ));
    assert!(matches!(
        resolve_errors(Some("gone".into()), Some(oom()), None, None, None),
        Some(GpuError::DeviceLost(_))
    ));
    assert!(matches!(
        resolve_errors(None, None, None, None, Some(GpuError::Validation("earlier".into()))),
        Some(GpuError::Validation(_))
    ));
    assert!(resolve_errors(None, None, None, None, None).is_none());
}

/// A validation error raised outside any error scope used to reach wgpu's
/// default handler, which panics. It must instead be reported by the next
/// `scoped` call, and only once.
#[test]
fn an_error_outside_any_scope_is_reported_later_not_panicked() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let _ = ctx.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("deliberately-invalid-outside-a-scope"),
        size: wgpu::Extent3d { width: 0, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    match ctx.scoped(|| ()) {
        Err(GpuError::Validation(msg)) => assert!(!msg.is_empty()),
        other => panic!("expected the stray validation error, got {other:?}"),
    }
    assert!(ctx.scoped(|| ()).is_ok(), "a stray error is reported once, not forever");
}
```

Append to `crates/elements-core/tests/field_pool.rs` (reuse its existing imports and GPU setup):

```rust
#[test]
fn clearing_the_pool_drops_every_free_texture() {
    let gpu = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let dims = FieldDims::new(4, 4, 4);
    let a = pool.acquire(&gpu, dims, FieldFormat::R32Float).unwrap();
    let b = pool.acquire(&gpu, dims, FieldFormat::R32Float).unwrap();
    pool.release(a);
    pool.release(b);
    assert_eq!(pool.pooled_count(), 2);
    pool.clear();
    assert_eq!(pool.pooled_count(), 0);
    // The next acquire is a fresh allocation, not a reused texture.
    let before = pool.allocation_count();
    let _c = pool.acquire(&gpu, dims, FieldFormat::R32Float).unwrap();
    assert_eq!(pool.allocation_count(), before + 1);
}
```

- [ ] **Step 2: Run them and confirm they fail** (compile errors). Note that `an_error_outside_any_scope_…` would *panic* in wgpu on today's code. That panic is the bug.

- [ ] **Step 3: Implement the core.** In `crates/elements-core/src/gpu/mod.rs`:
  - Add the variants to `GpuError`: `#[error("the GPU ran out of memory: {0}")] OutOfMemory(String)` and `#[error("internal GPU error: {0}")] Internal(String)`.
  - Add a field to `GpuContext`: `uncaptured: Arc<Mutex<Option<GpuError>>>`.
  - In `new_headless_async`, after `set_device_lost_callback`:

```rust
        // Without this, an error raised outside any error scope goes to wgpu's
        // default handler, which panics, and a daemon panic is exactly what
        // running the engine out of process exists to prevent. Keep the first
        // one; `scoped` reports it.
        let uncaptured = Arc::new(Mutex::new(None));
        let uncaptured_sink = Arc::clone(&uncaptured);
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            let mut slot = uncaptured_sink.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                *slot = Some(from_wgpu(error));
            }
        }));
```

  - Add the helpers:

```rust
fn from_wgpu(error: wgpu::Error) -> GpuError {
    match error {
        wgpu::Error::OutOfMemory { .. } => GpuError::OutOfMemory(error.to_string()),
        wgpu::Error::Internal { .. } => GpuError::Internal(error.to_string()),
        wgpu::Error::Validation { .. } => GpuError::Validation(error.to_string()),
    }
}

/// What one `scoped` call reports when several things went wrong.
///
/// A lost device outranks everything, since nothing else can be retried, but
/// its message keeps the detail of whatever else was captured. Then out of
/// memory, which says why a step failed. Then internal errors, then
/// validation errors, then an error raised earlier outside any scope.
pub fn resolve_errors(
    lost: Option<String>,
    out_of_memory: Option<wgpu::Error>,
    internal: Option<wgpu::Error>,
    validation: Option<wgpu::Error>,
    stray: Option<GpuError>,
) -> Option<GpuError> {
    let captured = out_of_memory
        .map(from_wgpu)
        .or_else(|| internal.map(from_wgpu))
        .or_else(|| validation.map(from_wgpu))
        .or(stray);
    match (lost, captured) {
        (Some(message), Some(e)) => Some(GpuError::DeviceLost(format!(
            "{message} (another error was also captured: {e})"
        ))),
        (Some(message), None) => Some(GpuError::DeviceLost(message)),
        (None, captured) => captured,
    }
}
```

  - Replace the body of `scoped`, and update its doc comment's **Limitations** section to say that an error raised outside any scope is reported by the next `scoped` call, possibly one unrelated to its cause:

```rust
    pub fn scoped<T>(&self, f: impl FnOnce() -> T) -> Result<T, GpuError> {
        let out_of_memory = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let value = f();
        // Pop in reverse push order.
        let validation = pollster::block_on(validation.pop());
        let internal = pollster::block_on(internal.pop());
        let out_of_memory = pollster::block_on(out_of_memory.pop());
        let stray = self.uncaptured.lock().unwrap_or_else(|e| e.into_inner()).take();
        match resolve_errors(self.device_lost(), out_of_memory, internal, validation, stray) {
            Some(e) => Err(e),
            None => Ok(value),
        }
    }
```

  - In `crates/elements-core/src/gpu/pool.rs`, add:

```rust
    /// Drop every pooled texture, giving its GPU memory back.
    ///
    /// The daemon calls this after running out of GPU memory: a pool that
    /// kept its free textures would hold on to the memory a retry needs.
    pub fn clear(&mut self) {
        self.free.clear();
    }
```

  - Update `pub use` in `gpu/mod.rs` to export `resolve_errors`. Search the workspace for exhaustive `match`es on `GpuError` (`grep -rn "GpuError::" crates`) and extend any that no longer compile.

- [ ] **Step 4: Protocol and daemon.**
  - In `crates/elements-ipc/src/protocol.rs`, add `OutOfMemory` to `ErrorKind` with a doc comment ("the GPU ran out of memory; a retry at the same resolution will fail again"), and set `ELEMENTS_PROTOCOL_VERSION` to 2. Add a test next to the crate's existing protocol tests:

```rust
#[test]
fn out_of_memory_has_its_own_wire_name() {
    assert_eq!(
        serde_json::to_string(&ErrorKind::OutOfMemory).unwrap(),
        "\"out_of_memory\""
    );
}
```

  - In `crates/elementsd/src/daemon.rs`:
    - Add `fn gpu_kind(e: &GpuError) -> ErrorKind` mapping `DeviceLost` to `ErrorKind::DeviceLost`, `OutOfMemory` to `ErrorKind::OutOfMemory`, and everything else to `ErrorKind::Gpu`. Use it in `map_node_error` for `NodeError::Gpu(g)`, and in `render`'s read-back error mapping, keeping the `device_lost().is_some()` check first there.
    - In `render`, wherever it already calls `timeline.discard()` for `DeviceLost`, add an `OutOfMemory` branch: `timeline.reset(&mut session.pool); session.pool.clear();`, with a comment saying why: to give the memory back so a smaller document can load.
    - Add a unit test next to `device_lost_maps_to_device_lost_not_a_generic_gpu_error`:

```rust
    #[test]
    fn out_of_memory_maps_to_its_own_kind_not_a_generic_gpu_error() {
        let e = NodeError::Gpu(GpuError::OutOfMemory("test".into()));
        assert_eq!(map_node_error(e).kind, ErrorKind::OutOfMemory);
    }
```

- [ ] **Step 5: Add-on.**
  - Create `addon/blender_elements/errors.py`:

```python
"""What the add-on tells the user about an engine error.

No bpy import, so the contract test can check it outside Blender.
"""


def describe(kind: str, message: str) -> tuple[str, bool]:
    """Return the status line for an engine error, and whether Live mode must stop.

    Live mode re-renders on every frame change, so an error that will recur on
    the next frame must turn it off rather than repeat.
    """
    if kind == "device_lost":
        return f"device_lost: {message} — restart the engine", True
    if kind == "out_of_memory":
        return "Out of GPU memory: lower the resolution, then try again", True
    return f"{kind}: {message}", False
```

  - In `client.py`, set `PROTOCOL_VERSION = 2`.
  - In `ops.py`'s render operator, build the status with `describe(e.kind, e.message)`. Keep the existing `shutdown_engine()` for `device_lost`. If `describe` says stop, set `settings.live = False` when the settings have that property.
  - In `handlers.py`'s Live handler, catch `ElementsError`. Keep `shutdown_engine()` for `device_lost`. When `describe` says stop, set `settings.live = False` and `settings.status` to the described line before re-raising or returning, following how the handler already reports errors.
  - Use relative imports (`from .errors import describe`).
  - In `tests/python/contract.py`, add the following and call it from `main` beside `check_bake_command()`, importing `describe` from `blender_elements.errors` the way `bake_command` is imported:

```python
def check_error_descriptions() -> None:
    """Errors that will recur on the next frame must stop Live mode."""
    status, stop = describe("out_of_memory", "Out of Memory")
    assert stop, status
    assert "resolution" in status, status
    status, stop = describe("graph", "no graph is loaded")
    assert not stop, status
    assert status == "graph: no graph is loaded", status
```

- [ ] **Step 6: Run the tests and `just check`.** `cargo nextest run -p elements-core --test gpu_context --test field_pool -p elements-ipc -p elementsd`, then `just check`. The `python_contract` test in `elementsd` runs `contract.py` under Python 3.11.

- [ ] **Step 7: Prove each test can fail.**

| Test | Mutation |
|---|---|
| `out_of_memory_is_reported_as_itself_and_outranks_validation` | in `resolve_errors`, put validation first: `let captured = validation.map(from_wgpu).or_else(\|\| out_of_memory.map(from_wgpu)).or_else(\|\| internal.map(from_wgpu)).or(stray);` (the second assertion then gets `Validation`) |
| `an_error_outside_any_scope_is_reported_later_not_panicked` | delete the `device.on_uncaptured_error(...)` statement (the test then panics inside wgpu) |
| `clearing_the_pool_drops_every_free_texture` | make `clear` a no-op (empty body) |
| `out_of_memory_has_its_own_wire_name` | add `#[serde(rename = "gpu_oom")]` to the variant |
| `out_of_memory_maps_to_its_own_kind_…` | in `gpu_kind`, map `OutOfMemory` to `ErrorKind::Gpu` |
| `check_error_descriptions` (via `python_contract`) | in `describe`, return `False` for `out_of_memory` |

- [ ] **Step 8: Commit and push.** Subject: `Report GPU out-of-memory as an error, never a panic`. The body must say the daemon's reset-and-clear on `out_of_memory` is not exercised by any test, because a real out-of-memory error cannot be forced reliably.

---

### Task 3: A mass test that notices swapped wiring

**Files:**
- Test: `crates/elements-ember/tests/solver.rs`

**Interfaces:**
- Consumes: `elements_ember::registry()`, `Graph`, `Timeline`, `TimelineConfig`, the existing `gpu()` helper in `tests/common`.

- [ ] **Step 1: Write the test.** Append to `crates/elements-ember/tests/solver.rs` (reuse its imports; add `SocketId` or `NodeId` if missing):

```rust
/// Which field one probe graph reads out.
#[derive(Clone, Copy)]
enum Probe {
    /// An emitter output, straight into `core.output`.
    Source(u32),
    /// A solver output after one frame, fed by the emitter.
    Solver(u32),
}

/// Total of the probed field over the domain, after frame 1.
fn total(probe: Probe) -> f64 {
    let registry = elements_ember::registry();
    let mut graph = Graph::new();
    let emitter = graph.add_node(
        registry
            .build(
                "ember.sphere_emitter",
                &serde_json::json!({ "center": [1.0, 1.0, 1.0], "radius": 0.5,
                                     "density_rate": 2.0, "temperature_rate": 5.0 }),
            )
            .unwrap(),
    );
    let output = graph.add_node(registry.build("core.output", &serde_json::json!({})).unwrap());
    let socket = |node: NodeId, index: u32| SocketId { node, index };
    match probe {
        Probe::Source(index) => graph.connect(socket(emitter, index), socket(output, 0)).unwrap(),
        Probe::Solver(index) => {
            // No buoyancy, so nothing moves and advection returns its input exactly.
            let solver = graph.add_node(
                registry
                    .build(KIND, &serde_json::json!({ "buoyancy_density": 0.0,
                                                      "buoyancy_temperature": 0.0 }))
                    .unwrap(),
            );
            graph.connect(socket(emitter, 0), socket(solver, 0)).unwrap();
            graph.connect(socket(emitter, 1), socket(solver, 1)).unwrap();
            graph.connect(socket(solver, index), socket(output, 0)).unwrap();
        }
    }
    graph.set_output(output);

    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut timeline = timeline(0);
    let frame = timeline
        .goto(&graph, &gpu, &mut pool, &mut pipelines, FieldDims::new(16, 16, 16), 1)
        .unwrap();
    let sum = frame.value.as_field().unwrap().read_back(&gpu).unwrap()
        .iter().map(|&v| v as f64).sum();
    frame.value.release_to(&mut pool);
    sum
}

/// One frame adds exactly rate × dt of each quantity, into the right output.
/// Density and temperature are emitted at different rates, so swapping them
/// at the solver's inputs or outputs changes both totals by a factor of 2.5.
#[test]
fn one_frame_adds_each_emitted_quantity_to_its_own_output() {
    let dt = 1.0 / 24.0;
    for (index, what) in [(0, "density"), (1, "temperature")] {
        let want = total(Probe::Source(index)) * dt;
        let got = total(Probe::Solver(index));
        assert!(want > 0.0, "{what}: the emitter must emit");
        assert!(((got - want) / want).abs() <= 1e-5, "{what}: solver holds {got}, emitted {want}");
    }
}
```

`timeline(budget)` is the existing helper in `solver.rs` (fps 24, start frame 1). The solver's `sockets()` are density source then temperature source in, and density, temperature, velocity out. If a name differs, adapt it and say so in the report.

- [ ] **Step 2: Run it.** `cargo nextest run -p elements-ember --test solver`. It should PASS on the current code: this pins existing behaviour. If it fails, the solver's wiring or its zero-velocity advection is wrong. Stop and report the numbers.

- [ ] **Step 3: Prove it can fail.**

| Test | Mutation |
|---|---|
| `one_frame_adds_each_emitted_quantity_to_its_own_output` | in `crates/elements-ember/src/solver.rs` `SmokeSolver::step`, swap the sources: `density: field(temperature_source, 1)?` and `temperature: field(density_source, 0)?` |

- [ ] **Step 4: Commit and push.** Subject: `Pin the solver's density and temperature wiring with a mass test`.

- [ ] **Step 5: Update the spec.** In `docs/superpowers/specs/2026-09-21-ember-solver-design.md` §6, mark risks (a), (g) and (i) as resolved by this slice, with one line each naming the commit and anything left open (for (g): the OS may still end the process first on unified memory, and the daemon's reset-and-clear is untested). Commit with Task 3's changes or as a small follow-up commit.
