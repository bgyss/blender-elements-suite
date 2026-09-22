# Ember Piece 2a — Solver to the Speed Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A GPU smoke solver (`ember.smoke_solver`) with a sphere emitter, proven by four validation scenes, and a measured 128³ speed gate whose recorded result decides how piece 2b is designed.

**Architecture:** A new crate `elements-ember` registers two node kinds on top of core's built-ins. The solver is one stateful node whose state (staggered velocity, density, temperature, pressure) lives in core's `StateStore`. Each solver stage is a small WGSL kernel with a Rust wrapper that *records* into a core `ComputeBatch`, so a whole substep is one queue submission. Every kernel is tested alone against a CPU reference on tiny non-cubic grids.

**Tech Stack:** Rust 2024, `wgpu` 30.0.1 (no optional features), WGSL compute, `serde`/`serde_json`, `bytemuck`. Dev: `tempfile`.

**Spec:** `docs/superpowers/specs/2026-09-21-ember-solver-design.md` (§1–4; §5 is 2b's)
**Umbrella:** `docs/superpowers/specs/2026-09-21-ember-design.md`

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Branch:** all work is on `ember-solver-2a`, branched from `main`. Push the branch after every task (`git push`); CI runs on every push. Never push to `main`.
- **`required_features` stays `wgpu::Features::empty()`.** If something seems to need a feature (for example GPU timestamps), stop and report it as a design problem.
- **Scalar and face fields are `R32Float`.** Never `R16Float`.
- **At most 4 storage textures per shader stage.** Read neighbours through `texture_3d<f32>` + `textureLoad`, write through storage.
- **Every binding a WGSL entry point declares must be used by it.** Pipelines use `layout: None` (auto layout), and an auto layout drops unused bindings, which makes the bind group fail validation.
- **`#![forbid(unsafe_code)]`** on every crate except `elements-ipc`.
- **Crate manifests use `dep.workspace = true`** for external dependencies; workspace crates use `{ path = "../<crate>" }`. No new external dependencies. If you think you need one, stop and ask.
- **Determinism:** the same document on the same machine gives bit-identical frames, however a frame is reached. Tests compare `f32::to_bits`.
- **The engine never panics on untrusted input.** Documents are untrusted; parameter errors are `DocError::BadParams`.
- **Never set `WGPU_BACKEND` locally.** The backend is Metal. `just ci-test` alone sets it.
- **`just check` must pass before every commit.**
- **Prove each new test can fail.** Every task lists a mutation per test. Apply it, run the test, confirm it fails, restore, and paste the real failing output into the task report. **A mutation changes exactly one thing.** If a listed mutation does *not* make its test fail, do not weaken or rewrite the assertion to force it: report it. That is a finding about the test.
- **Never tune a threshold to make a test pass.** If a correct implementation fails a stated threshold (for example the 10% divergence ratio), stop and report the measured numbers.
- **Test grids:** solver tests use grids of at most 32³. Kernel tests use non-cubic grids (8×6×5) so a swapped axis cannot pass by symmetry.
- **Commit style (CLAUDE.md):** a plain imperative subject line, then a body explaining *why*. End every commit message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TSRpZQEY9fnKQqHLh5WHpU
  ```
  The repo is jj-colocated. Commit with plain `git`, and do not run `jj` commands.

---

## Corrections to the spec made while planning

1. **CI was already green when planning started.** The first run
   (35764608274, on `main`) passed 172 of 172 tests with no skips. Task 1
   therefore only makes the log name the device, rather than fixing failures.
2. **The document version bumps to 3** for `domain_size`, for the reason piece
   1 bumped it to 2: serde ignores unknown fields, so an older engine would
   silently drop the size. Versions 1 and 2 still load.
3. **The voxel size lives on the `Graph`**, set by `Document::into_graph`,
   rather than being threaded through `eval_frame` and `Timeline::goto`. That
   keeps every existing call site unchanged.
4. **A failed step does not "leave the store at the previous frame".** The node
   takes its state out of the store, and on failure releases it; the timeline
   already clears the store on any error. The spec's §3.1 is corrected to say so.
5. **The pressure slot holds `φ = h·p`**, not `p`, so projection is `u -= ∇φ`
   with no `h` in the kernels. Warm starts stay valid because `h` is fixed per document.

---

## Conventions every kernel shares

Read this before any kernel task. These are the definitions the CPU references in the tests reproduce.

- **Indexing.** Fields are x-fastest: voxel `(i, j, k)` of a field with dims `(nx, ny, nz)` is element `i + nx * (j + ny * k)` of `Field::read_back`'s output.
- **Positions in cell units.** Cell `(0, 0, 0)` spans `[0, 1)³`. Cell `(i, j, k)`'s centre is `(i + 0.5, j + 0.5, k + 0.5)`. Metres are cell units times `dx`, measured from the domain's minimum corner.
- **Staggered faces** (see `StaggeredField`'s doc): X-face texel `(i, j, k)` sits at `(i, j + 0.5, k + 0.5)`. So to sample a texture at cell-unit position `x`, subtract its offset: `(0.5, 0.5, 0.5)` for a cell field, and `(0.5, 0.5, 0.5)` with the component's own axis set to `0` for a face. X face `i` lies between cells `i - 1` and `i`.
- **Boundaries.** Faces `0` and `n` along x and y, and face `0` along z, are solid walls: their normal velocity is always exactly `0`. Face `nz` along z is the open top. Pressure is Neumann at solid walls and Dirichlet (`φ = 0`) above the top.
- **Pressure is stored as `φ = h · p`**, so projection is `u -= ∇φ` and the solve is `∇²φ = div u`. The discrete Gauss–Seidel update of cell `c` is
  `φ_c = (Σ φ_n − dx² · div_c) / count`,
  where the sum and count run over the cell's neighbours: an interior neighbour adds `φ_n` and counts; a solid wall is left out entirely; the open top counts but adds `0`. The state slot is still named `pressure`; it holds `φ`.
- **Uniform layout.** Every kernel except the sphere emitter uses the `Params` struct in `common.wgsl`, 48 bytes, matched by `KernelParams` in Rust.

---

## File structure

```
crates/elements-core/src/gpu/batch.rs            ComputeBatch (new)
crates/elements-core/src/graph/{mod,node,document}.rs   domain_size, voxel_size, with_gpu_pool
crates/elements-ember/
  Cargo.toml
  src/lib.rs            register(), registry()
  src/params.rs         param parsing and validation helpers
  src/emitter.rs        ember.sphere_emitter + fill_sphere
  src/kernels/mod.rs    StepConstants, Uniforms, bind-group helper, dims checks
  src/kernels/advect.rs      advect_velocity, advect_scalar
  src/kernels/forces.rs      emit, buoyancy
  src/kernels/project.rs     divergence, pressure, subtract_gradient
  src/kernels/shaders/*.wgsl
  src/solver.rs         SolverState, Substep, substep(), ember.smoke_solver
  src/metrics.rs        CPU divergence and centroid
  src/bench.rs          Scene, GateRow, gate_verdict
  examples/speed_gate.rs
  tests/common/mod.rs   GPU helpers and CPU references
  tests/{emitter,advection,forces,projection,solver,scenes,bench}.rs
```

---

### Task 1: Branch, and make CI prove it ran on lavapipe

CI ran for the first time on 2026-09-22 (run 35764608274) and passed: 172 tests, 0 skipped, with `WGPU_BACKEND=vulkan` on a runner with no GPU. But the log never names the adapter: the workflow pipes `vulkaninfo --summary` through `head -40`, which cuts it off before the device list. "Tests ran on lavapipe" is an inference until the log shows it.

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.superpowers/sdd/progress.md` (gitignored; controller's ledger)

- [ ] **Step 1: Create the branch**

```bash
git checkout -b ember-solver-2a
```

- [ ] **Step 2: Print the device list in CI**

In `.github/workflows/ci.yml`, in the "Install software Vulkan" step, replace

```yaml
          vulkaninfo --summary | head -40
```

with

```yaml
          vulkaninfo --summary | grep -iE 'deviceName|driverName|apiVersion'
```

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows/ci.yml
git commit -F - <<'MSG'
Show which Vulkan device CI's tests run on

The first CI run passed, but its log never named the adapter: vulkaninfo's
summary was cut to 40 lines, before the device list. Whether the GPU tests
ran on lavapipe was an inference from there being no GPU on the runner.
Printing the device and driver names makes the log itself the evidence.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TSRpZQEY9fnKQqHLh5WHpU
MSG
git push -u origin ember-solver-2a
```

- [ ] **Step 4: Confirm the device**

```bash
gh run watch "$(gh run list --branch ember-solver-2a --limit 1 --json databaseId -q '.[0].databaseId')" --exit-status
gh run view --log "$(gh run list --branch ember-solver-2a --limit 1 --json databaseId -q '.[0].databaseId')" | grep -iE 'deviceName|Summary'
```

Expected: a `deviceName = llvmpipe (...)` line and `172 tests run: 172 passed, 0 skipped`. If the device is anything other than llvmpipe, stop and report it.

- [ ] **Step 5: Record in the ledger**

Append to `.superpowers/sdd/progress.md` under a new heading `# Ember piece 2a — solver to the speed gate — progress ledger`: the remote URL `https://github.com/bgyss/blender-elements-suite`, the first run's id and result, the device line from Step 4, and that the Core v1 close-out item "CI has never run" is now closed.

---

### Task 2: Physical units in the document, and pool access for nodes

The solver works in metres and seconds, so a scene previewed at 128³ matches a 512³ bake (spec §2.2). Nodes need the voxel size, and the solver needs the field pool while it runs GPU work.

**Files:**
- Modify: `crates/elements-core/src/graph/mod.rs`
- Modify: `crates/elements-core/src/graph/node.rs`
- Modify: `crates/elements-core/src/graph/document.rs`
- Modify: `crates/elements-core/src/nodes/output.rs` (its test constructs an `EvalCtx`)
- Test: `crates/elements-core/tests/document.rs`

**Interfaces:**
- Produces: `elements_core::graph::DEFAULT_DOMAIN_SIZE: f64 = 2.0`; `Document::domain_size: f64`; `ELEMENTS_DOC_VERSION == 3`; `Graph::set_domain_size(&mut self, metres: f64)`; `Graph::domain_size(&self) -> f64`; `EvalCtx::voxel_size(&self) -> f32`; `EvalCtx::with_gpu_pool<T>(&mut self, f: impl FnOnce(&GpuContext, &mut PipelineCache, &mut FieldPool) -> Result<T, GpuError>) -> Result<T, NodeError>`.

- [ ] **Step 1: Write the failing tests**

Append to `crates/elements-core/tests/document.rs` (add any imports that are missing: `FieldFormat`, `fill_constant`, `FieldPool`, `GpuContext`, `PipelineCache` from `elements_core::gpu`, and `SocketSpec`, `SocketType`, `Value` from `elements_core::graph`):

```rust
/// Fills its output with `EvalCtx::voxel_size()`.
struct VoxelSizeProbe;

impl Node for VoxelSizeProbe {
    fn kind(&self) -> &'static str {
        "test.voxel_size_probe"
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Field],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let field = ctx.acquire_uninit(FieldFormat::R32Float)?;
        let size = ctx.voxel_size();
        ctx.with_gpu(|gpu, cache| fill_constant(gpu, cache, &field, size))?;
        Ok(vec![Value::Field(field)])
    }
}

fn probe_doc(version_and_size: &str) -> String {
    format!(
        r#"{{
          {version_and_size}
          "dims": [8, 4, 2],
          "nodes": [
            {{ "id": 0, "kind": "test.voxel_size_probe", "params": {{}} }},
            {{ "id": 1, "kind": "core.output", "params": {{}} }}
          ],
          "edges": [{{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }}],
          "output": 1
        }}"#
    )
}

#[test]
fn a_version_2_document_migrates_with_the_default_domain_size() {
    let doc = Document::from_json(&probe_doc(r#""version": 2,"#)).unwrap();
    assert_eq!(doc.version, ELEMENTS_DOC_VERSION);
    assert_eq!(doc.version, 3);
    assert_eq!(doc.domain_size, 2.0);
}

#[test]
fn domain_size_reaches_nodes_as_metres_per_voxel_along_the_longest_axis() {
    let mut registry = NodeRegistry::with_builtins();
    registry.register("test.voxel_size_probe", |_| {
        Ok(Box::new(VoxelSizeProbe) as Box<dyn Node>)
    });
    let (graph, dims) = Document::from_json(&probe_doc(r#""version": 3, "domain_size": 4.0,"#))
        .unwrap()
        .into_graph(&registry)
        .unwrap();
    assert_eq!(graph.domain_size(), 4.0);

    let gpu = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let value = graph.eval(&gpu, &mut pool, &mut pipelines, dims).unwrap();
    let voxels = value.as_field().unwrap().read_back(&gpu).unwrap();
    // 4 m along the longest axis, which has 8 cells.
    assert!(voxels.iter().all(|&v| v == 0.5), "got {:?}", &voxels[..4]);
}

#[test]
fn rejects_a_domain_size_that_is_not_positive() {
    for size in ["0.0", "-1.0"] {
        let text = probe_doc(&format!(r#""version": 3, "domain_size": {size},"#));
        match Document::from_json(&text) {
            Err(DocError::BadParams { .. }) => {}
            other => panic!("domain_size {size}: expected BadParams, got {other:?}"),
        }
    }
}
```

Also update the existing test `a_version_1_document_migrates_with_time_defaults`: change its `assert_eq!(doc.version, 2);` to `assert_eq!(doc.version, ELEMENTS_DOC_VERSION);`.

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo nextest run -p elements-core --test document`
Expected: compile errors (`domain_size`, `voxel_size` do not exist).

- [ ] **Step 3: Implement**

In `crates/elements-core/src/graph/mod.rs`, add below the `pub use` lines:

```rust
/// Metres spanned by the grid's longest axis when a document does not say.
/// Matches Blender's default cube.
pub const DEFAULT_DOMAIN_SIZE: f64 = 2.0;
```

Replace `#[derive(Default)]` on `Graph` with a manual impl (a derived default would make the domain zero metres), and add the field:

```rust
pub struct Graph {
    nodes: Vec<Box<dyn Node>>,
    /// Maps a destination input socket to the source output socket feeding it.
    edges: HashMap<SocketId, SocketId>,
    output: Option<NodeId>,
    /// Metres along the domain's longest axis. See `EvalCtx::voxel_size`.
    domain_size: f64,
}

impl Default for Graph {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            edges: HashMap::new(),
            output: None,
            domain_size: DEFAULT_DOMAIN_SIZE,
        }
    }
}
```

Add to `impl Graph`:

```rust
    /// Set how many metres the domain's longest axis spans.
    pub fn set_domain_size(&mut self, metres: f64) {
        self.domain_size = metres;
    }

    pub fn domain_size(&self) -> f64 {
        self.domain_size
    }
```

Add `domain_size: f64` to `struct Run`, set it to `self.domain_size` where `eval_frame` builds the `Run`, and pass `domain_size: run.domain_size` where `run` builds each `EvalCtx`.

In `crates/elements-core/src/graph/node.rs`, add to `EvalCtx`:

```rust
    /// Metres along the domain's longest axis.
    pub(crate) domain_size: f64,
```

and to `impl EvalCtx<'_>`:

```rust
    /// Metres per voxel. Voxels are cubic, and the document's `domain_size`
    /// spans the domain's longest axis.
    pub fn voxel_size(&self) -> f32 {
        let longest = self.dims.x.max(self.dims.y).max(self.dims.z).max(1);
        (self.domain_size / longest as f64) as f32
    }

    /// Run GPU work that also needs the field pool, such as a solver that
    /// acquires scratch fields between kernels.
    pub fn with_gpu_pool<T>(
        &mut self,
        f: impl FnOnce(&GpuContext, &mut PipelineCache, &mut FieldPool) -> Result<T, GpuError>,
    ) -> Result<T, NodeError> {
        Ok(f(self.gpu, self.pipelines, self.pool)?)
    }
```

In `crates/elements-core/src/nodes/output.rs`, the test at line 80 builds an `EvalCtx` literal; add `domain_size: crate::graph::DEFAULT_DOMAIN_SIZE,` to it.

In `crates/elements-core/src/graph/document.rs`:
- change `ELEMENTS_DOC_VERSION` to `3`, and its doc comment to "It also reads versions 1 and 2, which predate time and physical units: see `Document::from_json`."
- add `fn default_domain_size() -> f64 { super::DEFAULT_DOMAIN_SIZE }`
- add to `Document`, after `cache_budget_mb`:

```rust
    /// Metres along the grid's longest axis. Versions 1 and 2 have none and get 2.0.
    #[serde(default = "default_domain_size")]
    pub domain_size: f64,
```

- in `from_json`, change the `1 =>` arm to `1 | 2 => doc.version = ELEMENTS_DOC_VERSION,` with the comment "Versions 1 and 2 predate time and physical units. Their only migration is the defaults serde has already filled in above.", and after the `fps` check add:

```rust
        if !(doc.domain_size.is_finite() && doc.domain_size > 0.0) {
            return Err(DocError::BadParams {
                kind: "document".to_string(),
                reason: format!(
                    "domain_size must be a positive, finite number of metres, got {}",
                    doc.domain_size
                ),
            });
        }
```

- in `into_graph`, before `graph.set_output(...)`, add `graph.set_domain_size(self.domain_size);`.

Any other code constructing a `Document` literal (`grep -rn 'Document {' crates`) needs `domain_size: elements_core::graph::DEFAULT_DOMAIN_SIZE` (or `super::DEFAULT_DOMAIN_SIZE` inside core).

- [ ] **Step 4: Run the tests and `just check`**

Run: `cargo nextest run -p elements-core --test document` then `just check`
Expected: PASS.

- [ ] **Step 5: Prove each test can fail**

| Test | Mutation (one change) |
|---|---|
| `a_version_2_document_migrates_…` | in `from_json`, `1 \| 2 =>` back to `1 =>` |
| `domain_size_reaches_nodes_…` | in `voxel_size`, `.max(self.dims.y).max(self.dims.z)` → `.min(self.dims.y).min(self.dims.z)` |
| `rejects_a_domain_size_…` | delete the `domain_size` check in `from_json` |

- [ ] **Step 6: Commit and push**

Subject: `Give documents a physical domain size in metres`. The body says why: parameters in voxel units would make a 128³ preview and a 512³ bake of the same scene behave differently; the version bump is there because serde ignores unknown fields, so an older engine would otherwise silently drop the size.

---

### Task 3: `ComputeBatch`

The pressure solve alone is `2N` dispatches a substep. `dispatch_over_field` submits once per call, and at 128³ with 160 iterations that submit overhead could fail the gate for reasons that have nothing to do with the solver's maths (spec §2.5).

**Files:**
- Create: `crates/elements-core/src/gpu/batch.rs`
- Modify: `crates/elements-core/src/gpu/mod.rs`
- Test: `crates/elements-core/tests/batch.rs`

**Interfaces:**
- Produces: `elements_core::gpu::ComputeBatch` with `new() -> Self`, `dispatch(&mut self, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, dims: FieldDims)`, `len(&self) -> usize`, `is_empty(&self) -> bool`, `submit(self, ctx: &GpuContext) -> Result<(), GpuError>`.

- [ ] **Step 1: Write the failing test**

Create `crates/elements-core/tests/batch.rs`:

```rust
use elements_core::gpu::{ComputeBatch, FieldDims, FieldPool, GpuContext, PipelineCache};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    dims: [u32; 3],
    value: f32,
}

fn uniform(gpu: &GpuContext, dims: FieldDims, value: f32) -> wgpu::Buffer {
    gpu.device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test-params"),
            contents: bytemuck::bytes_of(&Params {
                dims: [dims.x, dims.y, dims.z],
                value,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

/// Every dispatch in one batch sees the writes of the dispatches recorded
/// before it. Here: fill B with 2, then twice do A += B * 0.5. In order that
/// gives A = 2. Any reordering or lost write gives something else.
#[test]
fn dispatches_in_one_batch_run_in_order_and_see_earlier_writes() {
    let gpu = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let dims = FieldDims::new(8, 6, 5);
    let a = pool.acquire_zeroed(&gpu, &mut cache, dims).unwrap();
    let b = pool.acquire_zeroed(&gpu, &mut cache, dims).unwrap();

    let constant = cache
        .get_or_create(
            &gpu,
            "constant",
            include_str!("../src/gpu/shaders/constant.wgsl"),
            "main",
        )
        .unwrap();
    let accumulate = cache
        .get_or_create(
            &gpu,
            "accumulate",
            include_str!("../src/gpu/shaders/accumulate.wgsl"),
            "main",
        )
        .unwrap();

    let fill_params = uniform(&gpu, dims, 2.0);
    let add_params = uniform(&gpu, dims, 0.5);
    let fill_b = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &constant.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(b.view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: fill_params.as_entire_binding(),
            },
        ],
    });
    let add_b_to_a = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &accumulate.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(a.view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(b.view()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: add_params.as_entire_binding(),
            },
        ],
    });

    let mut batch = ComputeBatch::new();
    batch.dispatch(&constant, &fill_b, dims);
    batch.dispatch(&accumulate, &add_b_to_a, dims);
    batch.dispatch(&accumulate, &add_b_to_a, dims);
    assert_eq!(batch.len(), 3);
    batch.submit(&gpu).unwrap();

    let values = a.read_back(&gpu).unwrap();
    assert!(values.iter().all(|&v| v == 2.0), "got {:?}", &values[..4]);
}

#[test]
fn an_empty_batch_submits_nothing_and_succeeds() {
    let gpu = GpuContext::new_headless().expect("no GPU adapter available");
    let batch = ComputeBatch::new();
    assert!(batch.is_empty());
    batch.submit(&gpu).unwrap();
}
```

Add `bytemuck.workspace = true` and `wgpu.workspace = true` to `[dev-dependencies]` of `crates/elements-core/Cargo.toml` only if the test fails to compile without them (integration tests can normally use the package's own dependencies).

- [ ] **Step 2: Run to verify it fails**

Run: `cargo nextest run -p elements-core --test batch`
Expected: compile error, `ComputeBatch` not found.

- [ ] **Step 3: Implement**

Create `crates/elements-core/src/gpu/batch.rs`:

```rust
//! Many compute dispatches, one queue submission.

use super::{FieldDims, GpuContext, GpuError, WORKGROUP};

/// Records compute dispatches and submits them together.
///
/// `dispatch_over_field` submits once per call, which suits a node that runs
/// one kernel a frame. A solver runs hundreds (every pressure iteration is
/// two), and per-submit overhead would then dominate its step time.
///
/// Dispatches run in the order they were recorded, and each one sees every
/// earlier dispatch's writes: WebGPU makes each dispatch its own usage scope
/// and orders storage writes between them.
///
/// Nothing touches the GPU until `submit`, so dropping a batch unsubmitted is
/// harmless. Fields a batch reads or writes must stay alive, and must not be
/// released to a pool for reuse, until `submit` returns.
#[derive(Default)]
pub struct ComputeBatch {
    dispatches: Vec<Dispatch>,
}

struct Dispatch {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    workgroups: [u32; 3],
}

impl ComputeBatch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one invocation per voxel of `dims`.
    pub fn dispatch(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        dims: FieldDims,
    ) {
        self.dispatches.push(Dispatch {
            pipeline: pipeline.clone(),
            bind_group: bind_group.clone(),
            workgroups: [
                dims.x.div_ceil(WORKGROUP),
                dims.y.div_ceil(WORKGROUP),
                dims.z.div_ceil(WORKGROUP),
            ],
        });
    }

    /// Number of dispatches recorded so far.
    pub fn len(&self) -> usize {
        self.dispatches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dispatches.is_empty()
    }

    /// Encode every recorded dispatch into one compute pass and submit it once.
    pub fn submit(self, ctx: &GpuContext) -> Result<(), GpuError> {
        if self.dispatches.is_empty() {
            return Ok(());
        }
        ctx.scoped(|| {
            let mut encoder = ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("elements-batch"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("elements-batch"),
                    timestamp_writes: None,
                });
                for d in &self.dispatches {
                    pass.set_pipeline(&d.pipeline);
                    pass.set_bind_group(0, &d.bind_group, &[]);
                    let [x, y, z] = d.workgroups;
                    pass.dispatch_workgroups(x, y, z);
                }
            }
            ctx.queue().submit(Some(encoder.finish()));
        })
    }
}
```

In `crates/elements-core/src/gpu/mod.rs`, add `mod batch;` and `pub use batch::ComputeBatch;`.

- [ ] **Step 4: Run the tests and `just check`**

Expected: PASS.

- [ ] **Step 5: Prove the test can fail**

| Test | Mutation |
|---|---|
| `dispatches_in_one_batch_run_in_order_…` | in `submit`, `for d in &self.dispatches` → `for d in self.dispatches.iter().rev()` |

- [ ] **Step 6: Commit and push**

Subject: `Batch compute dispatches into one submission`.

---

### Task 4: The `elements-ember` crate and the sphere emitter

**Files:**
- Modify: `Cargo.toml` (nothing: `members = ["crates/*"]` already picks the crate up)
- Create: `crates/elements-ember/Cargo.toml`, `src/lib.rs`, `src/params.rs`, `src/emitter.rs`, `src/kernels/shaders/sphere.wgsl`
- Modify: `crates/elements-cli/Cargo.toml`, `crates/elements-cli/src/bake.rs` (two call sites)
- Modify: `crates/elementsd/Cargo.toml`, `crates/elementsd/src/daemon.rs:48`
- Test: `crates/elements-ember/tests/common/mod.rs`, `crates/elements-ember/tests/emitter.rs`, `crates/elements-cli/tests/cli.rs`, `crates/elementsd/tests/session.rs`

**Interfaces:**
- Consumes: `EvalCtx::voxel_size`, `ComputeBatch`.
- Produces: `elements_ember::register(&mut NodeRegistry)`; `elements_ember::registry() -> NodeRegistry` (built-ins plus Ember); `elements_ember::emitter::{KIND, Sphere, fill_sphere}` where `Sphere { center: [f32; 3], radius: f32, density_rate: f32, temperature_rate: f32 }` derives `Debug, Clone, Copy, PartialEq, Serialize, Deserialize`, and `fill_sphere(gpu: &GpuContext, cache: &mut PipelineCache, density: &Field, temperature: &Field, sphere: &Sphere, dx: f32) -> Result<(), GpuError>`; `params::{parse, bad, finite}`; test helpers in `tests/common/mod.rs` (below).

- [ ] **Step 1: Create the crate skeleton**

`crates/elements-ember/Cargo.toml`:

```toml
[package]
name = "elements-ember"
version = "0.1.0"
edition.workspace = true
license.workspace = true
repository.workspace = true

[dependencies]
bytemuck.workspace = true
elements-core = { path = "../elements-core" }
serde.workspace = true
serde_json.workspace = true
wgpu.workspace = true
```

`crates/elements-ember/src/lib.rs`:

```rust
#![forbid(unsafe_code)]

//! Ember: a dense-grid smoke solver for the Elements Suite.
//!
//! Registers node kinds on top of core's built-ins. Core knows nothing about
//! smoke; this crate is the seam `NodeRegistry` exists for.

pub mod emitter;
mod params;

use elements_core::graph::NodeRegistry;

/// Add Ember's node kinds to `registry`.
pub fn register(registry: &mut NodeRegistry) {
    registry.register(emitter::KIND, emitter::build);
}

/// Core's built-in node kinds plus Ember's.
pub fn registry() -> NodeRegistry {
    let mut registry = NodeRegistry::with_builtins();
    register(&mut registry);
    registry
}
```

`crates/elements-ember/src/params.rs`:

```rust
//! Parsing and validating node parameters from untrusted documents.

use elements_core::graph::DocError;
use serde::de::DeserializeOwned;

/// A `BadParams` error for node kind `kind`.
pub(crate) fn bad(kind: &str, reason: impl Into<String>) -> DocError {
    DocError::BadParams {
        kind: kind.to_owned(),
        reason: reason.into(),
    }
}

/// Deserialize a node's parameters, naming the node kind on failure.
pub(crate) fn parse<T: DeserializeOwned>(
    kind: &str,
    params: &serde_json::Value,
) -> Result<T, DocError> {
    T::deserialize(params).map_err(|e| bad(kind, e.to_string()))
}

/// Reject a parameter holding NaN or an infinity.
///
/// JSON has no NaN, but a number too large for `f32` (for example `1e39`)
/// deserializes to infinity.
pub(crate) fn finite(kind: &str, name: &str, values: &[f32]) -> Result<(), DocError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(bad(kind, format!("{name} must be finite, got {values:?}")))
    }
}
```

- [ ] **Step 2: Write the failing tests**

`crates/elements-ember/tests/common/mod.rs` (shared by every Ember test file; later tasks add to it):

```rust
#![allow(dead_code)]

use elements_core::gpu::{Field, FieldDims, FieldFormat, FieldPool, GpuContext};

pub fn gpu() -> GpuContext {
    GpuContext::new_headless().expect("no GPU adapter available")
}

/// Index of voxel `(i, j, k)` in an x-fastest array of `dims`.
pub fn index(dims: FieldDims, i: u32, j: u32, k: u32) -> usize {
    (i + dims.x * (j + dims.y * k)) as usize
}

/// A pooled R32Float field holding `values`.
pub fn upload(gpu: &GpuContext, pool: &mut FieldPool, dims: FieldDims, values: &[f32]) -> Field {
    let field = pool.acquire(gpu, dims, FieldFormat::R32Float).unwrap();
    field.write(gpu, values).unwrap();
    field
}

/// A deterministic, irregular test pattern with values in about [-1, 1].
pub fn pattern(dims: FieldDims, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(dims.voxel_count());
    for k in 0..dims.z {
        for j in 0..dims.y {
            for i in 0..dims.x {
                let h = (i * 73 + j * 151 + k * 283 + seed * 997) % 211;
                out.push(h as f32 / 105.0 - 1.0);
            }
        }
    }
    out
}

pub fn assert_close(gpu: &[f32], cpu: &[f32], tol: f32, what: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{what}: length");
    for (n, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            (g - c).abs() <= tol,
            "{what}: element {n}: gpu {g}, cpu {c}"
        );
    }
}
```

`crates/elements-ember/tests/emitter.rs`:

```rust
mod common;

use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_core::graph::DocError;
use elements_ember::emitter::{KIND, Sphere, fill_sphere};

/// Total emitted per second: the sum of rate × occupancy × voxel volume.
fn total_emitted(resolution: u32) -> (f64, f64) {
    let gpu = common::gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let dims = FieldDims::new(resolution, resolution, resolution);
    let dx = 2.0 / resolution as f32;
    let density = pool.acquire(&gpu, dims, FieldFormat::R32Float).unwrap();
    let temperature = pool.acquire(&gpu, dims, FieldFormat::R32Float).unwrap();
    let sphere = Sphere {
        center: [1.0, 1.0, 0.7],
        radius: 0.3,
        density_rate: 2.0,
        temperature_rate: 5.0,
    };
    fill_sphere(&gpu, &mut cache, &density, &temperature, &sphere, dx).unwrap();
    let volume = (dx as f64).powi(3);
    let sum = |f: &elements_core::gpu::Field| {
        f.read_back(&gpu).unwrap().iter().map(|&v| v as f64).sum::<f64>() * volume
    };
    (sum(&density), sum(&temperature))
}

/// The emitted amount must not depend on resolution, or a 128³ preview and a
/// 512³ bake of one scene would differ (spec §2.2, §4.1).
#[test]
fn the_emitted_total_matches_the_sphere_at_any_resolution() {
    let analytic = 4.0 / 3.0 * std::f64::consts::PI * 0.3_f64.powi(3);
    let (d32, t32) = total_emitted(32);
    let (d64, t64) = total_emitted(64);
    for (got, rate, what) in [
        (d32, 2.0, "density 32³"),
        (d64, 2.0, "density 64³"),
        (t32, 5.0, "temperature 32³"),
        (t64, 5.0, "temperature 64³"),
    ] {
        let want = analytic * rate;
        assert!(
            ((got - want) / want).abs() <= 0.05,
            "{what}: emitted {got}, sphere holds {want}"
        );
    }
    assert!(((d32 - d64) / d64).abs() <= 0.05, "32³ {d32} vs 64³ {d64}");
}

fn rejected(params: serde_json::Value) -> bool {
    matches!(
        elements_ember::registry().build(KIND, &params),
        Err(DocError::BadParams { .. })
    )
}

#[test]
fn rejects_bad_sphere_parameters() {
    let ok = serde_json::json!({ "center": [1.0, 1.0, 0.3], "radius": 0.2 });
    assert!(!rejected(ok), "a valid sphere must build");
    assert!(rejected(serde_json::json!({ "center": [1.0, 1.0, 0.3], "radius": 0.0 })));
    assert!(rejected(serde_json::json!({ "center": [1.0, 1.0, 0.3], "radius": -1.0 })));
    assert!(rejected(serde_json::json!({ "center": [1e39, 1.0, 0.3], "radius": 0.2 })));
    assert!(rejected(serde_json::json!({ "center": [1.0, 1.0, 0.3], "radius": 0.2, "density_rate": 1e39 })));
    assert!(rejected(serde_json::json!({ "center": [1.0, 1.0, 0.3], "radius": 0.2, "colour": 1 })));
    assert!(rejected(serde_json::json!({ "radius": 0.2 })));
}
```

Append to `crates/elements-cli/tests/cli.rs`:

```rust
const SPHERE_GRAPH: &str = r#"{
  "version": 3,
  "dims": [8, 8, 8],
  "nodes": [
    { "id": 0, "kind": "ember.sphere_emitter",
      "params": { "center": [1.0, 1.0, 1.0], "radius": 0.5, "density_rate": 1.0 } },
    { "id": 1, "kind": "core.output", "params": {} }
  ],
  "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
  "output": 1
}"#;

#[test]
fn the_cli_knows_ember_node_kinds() {
    let dir = tempfile::tempdir().unwrap();
    let graph = dir.path().join("sphere.elements");
    std::fs::write(&graph, SPHERE_GRAPH).unwrap();
    let out = dir.path().join("sphere.npy");
    let output = cli()
        .args(["dump-npy", graph.to_str().unwrap()])
        .args(["--out", out.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(out.exists());
}
```

Append to `crates/elementsd/tests/session.rs` (it already imports `Command`, `Response`, `write_message`, `read_message`, `BufReader`, `ELEMENTS_PROTOCOL_VERSION`):

```rust
#[test]
fn the_daemon_knows_ember_node_kinds() {
    let daemon = Daemon::start();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sphere.elements");
    std::fs::write(
        &path,
        r#"{
          "version": 3,
          "dims": [8, 8, 8],
          "nodes": [
            { "id": 0, "kind": "ember.sphere_emitter",
              "params": { "center": [1.0, 1.0, 1.0], "radius": 0.5 } },
            { "id": 1, "kind": "core.output", "params": {} }
          ],
          "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
          "output": 1
        }"#,
    )
    .unwrap();

    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    write_message(
        &mut stream,
        &Command::Hello {
            protocol_version: ELEMENTS_PROTOCOL_VERSION,
        },
    )
    .unwrap();
    let _ack: Response = read_message(&mut reader).unwrap().unwrap();
    write_message(
        &mut stream,
        &Command::LoadGraph {
            path: path.to_string_lossy().into_owned(),
        },
    )
    .unwrap();
    match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
        Response::Loaded { nodes, .. } => assert_eq!(nodes, 2),
        other => panic!("expected Loaded, got {other:?}"),
    }
    write_message(&mut stream, &Command::Shutdown).unwrap();
}
```

- [ ] **Step 3: Run to verify they fail**

Run: `cargo nextest run -p elements-ember -p elements-cli -p elementsd`
Expected: compile errors (the `emitter` module does not exist).

- [ ] **Step 4: Implement the emitter**

`crates/elements-ember/src/kernels/shaders/sphere.wgsl`:

```wgsl
// A sphere emitter: rate times occupancy, into a density and a temperature source.

struct SphereParams {
    dims: vec3<u32>,
    dx: f32,
    center: vec3<f32>,
    radius: f32,
    density_rate: f32,
    temperature_rate: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var density: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var temperature: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: SphereParams;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    // Cell centre in metres, from the domain's minimum corner.
    let x = (vec3<f32>(gid) + vec3<f32>(0.5)) * params.dx;
    let sd = length(x - params.center) - params.radius;
    // A one-voxel linear ramp centred on the surface, so the occupied volume
    // matches the sphere's to first order at any resolution.
    let occupancy = clamp(0.5 - sd / params.dx, 0.0, 1.0);
    let p = vec3<i32>(gid);
    textureStore(density, p, vec4<f32>(occupancy * params.density_rate, 0.0, 0.0, 0.0));
    textureStore(temperature, p, vec4<f32>(occupancy * params.temperature_rate, 0.0, 0.0, 0.0));
}
```

`crates/elements-ember/src/emitter.rs`:

```rust
//! `ember.sphere_emitter`: a sphere that emits density and temperature.
//!
//! Stateless. Its outputs are rates per second; the solver scales them by
//! the substep length when it adds them.

use elements_core::gpu::{ComputeBatch, Field, FieldFormat, GpuContext, GpuError, PipelineCache};
use elements_core::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;

use crate::params;

pub const KIND: &str = "ember.sphere_emitter";

/// A sphere in metres, relative to the domain's minimum corner.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
    /// Density added per second where the sphere is fully occupied.
    #[serde(default)]
    pub density_rate: f32,
    /// Temperature added per second where the sphere is fully occupied.
    #[serde(default)]
    pub temperature_rate: f32,
}

/// Matches `SphereParams` in `sphere.wgsl`: `vec3` members align to 16, so
/// `center` starts at byte 16 and the struct pads to 48.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SphereGpu {
    dims: [u32; 3],
    dx: f32,
    center: [f32; 3],
    radius: f32,
    density_rate: f32,
    temperature_rate: f32,
    _pad: [u32; 2],
}

/// Write `sphere`'s density and temperature rates into two fields of the same dims.
pub fn fill_sphere(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    density: &Field,
    temperature: &Field,
    sphere: &Sphere,
    dx: f32,
) -> Result<(), GpuError> {
    let dims = density.dims();
    if temperature.dims() != dims {
        return Err(GpuError::Validation(format!(
            "fill_sphere: density is {dims:?}, temperature is {:?}",
            temperature.dims()
        )));
    }
    let pipeline = cache.get_or_create(
        gpu,
        "ember.sphere",
        include_str!("kernels/shaders/sphere.wgsl"),
        "main",
    )?;
    let params = SphereGpu {
        dims: [dims.x, dims.y, dims.z],
        dx,
        center: sphere.center,
        radius: sphere.radius,
        density_rate: sphere.density_rate,
        temperature_rate: sphere.temperature_rate,
        _pad: [0; 2],
    };
    let bind_group = gpu.scoped(|| {
        let uniform = gpu
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ember-sphere-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ember-sphere"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(density.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(temperature.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        })
    })?;
    let mut batch = ComputeBatch::new();
    batch.dispatch(&pipeline, &bind_group, dims);
    batch.submit(gpu)
}

#[derive(Debug, Clone)]
pub struct SphereEmitter {
    sphere: Sphere,
}

impl Node for SphereEmitter {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Field, SocketType::Field],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let density = ctx.acquire_uninit(FieldFormat::R32Float)?;
        let temperature = match ctx.acquire_uninit(FieldFormat::R32Float) {
            Ok(field) => field,
            Err(e) => {
                ctx.release(Value::Field(density));
                return Err(e);
            }
        };
        let dx = ctx.voxel_size();
        let sphere = self.sphere;
        if let Err(e) = ctx.with_gpu(|gpu, cache| {
            fill_sphere(gpu, cache, &density, &temperature, &sphere, dx)
        }) {
            ctx.release(Value::Field(density));
            ctx.release(Value::Field(temperature));
            return Err(e);
        }
        Ok(vec![Value::Field(density), Value::Field(temperature)])
    }
}

pub(crate) fn build(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    let sphere: Sphere = params::parse(KIND, params)?;
    params::finite(KIND, "center", &sphere.center)?;
    params::finite(
        KIND,
        "radius and rates",
        &[sphere.radius, sphere.density_rate, sphere.temperature_rate],
    )?;
    if sphere.radius <= 0.0 {
        return Err(params::bad(
            KIND,
            format!("radius must be positive, got {}", sphere.radius),
        ));
    }
    Ok(Box::new(SphereEmitter { sphere }))
}
```

Register Ember in the CLI and daemon:
- `crates/elements-cli/Cargo.toml` and `crates/elementsd/Cargo.toml`: add `elements-ember = { path = "../elements-ember" }` under `[dependencies]`.
- `crates/elements-cli/src/bake.rs`: both `let registry = NodeRegistry::with_builtins();` become `let registry = elements_ember::registry();`. Remove `NodeRegistry` from the `use` line if it is now unused.
- `crates/elementsd/src/daemon.rs:48`: `registry: NodeRegistry::with_builtins(),` becomes `registry: elements_ember::registry(),`.

- [ ] **Step 5: Run the tests and `just check`**

Run: `cargo nextest run -p elements-ember -p elements-cli -p elementsd` then `just check`
Expected: PASS.

- [ ] **Step 6: Prove each test can fail**

| Test | Mutation |
|---|---|
| `the_emitted_total_matches_…` | in `sphere.wgsl`, `let x = (vec3<f32>(gid) + vec3<f32>(0.5)) * params.dx;` → drop `* params.dx` |
| `rejects_bad_sphere_parameters` | in `build`, delete the `radius <= 0.0` check |
| `the_cli_knows_ember_node_kinds` | in `bake.rs`, the `dump-npy` path's registry back to `NodeRegistry::with_builtins()` (whichever of the two call sites `dump-npy` uses) |
| `the_daemon_knows_ember_node_kinds` | in `daemon.rs`, back to `NodeRegistry::with_builtins()` |

- [ ] **Step 7: Commit and push**

Subject: `Add the Ember crate with a sphere emitter`.

---

### Task 5: Advection kernels

**Files:**
- Create: `crates/elements-ember/src/kernels/mod.rs`, `src/kernels/advect.rs`, `src/kernels/shaders/common.wgsl`, `src/kernels/shaders/velocity.wgsl`, `src/kernels/shaders/advect_velocity.wgsl`, `src/kernels/shaders/advect_scalar.wgsl`
- Modify: `crates/elements-ember/src/lib.rs` (add `pub mod kernels;`)
- Modify: `crates/elements-ember/tests/common/mod.rs` (CPU references)
- Test: `crates/elements-ember/tests/advection.rs`

**Interfaces:**
- Consumes: `ComputeBatch`, `StaggeredField`, `Axis`.
- Produces:
  - `kernels::StepConstants { cells: FieldDims, h: f32, dx: f32, alpha: f32, beta: f32 }` (`Debug, Clone, Copy, PartialEq`)
  - `kernels::Uniforms::new(gpu: &GpuContext, c: &StepConstants) -> Result<Uniforms, GpuError>`, `.cells() -> FieldDims`
  - `kernels::advect_velocity(gpu, cache, batch: &mut ComputeBatch, u: &Uniforms, src: &StaggeredField, dst: &StaggeredField) -> Result<(), GpuError>`
  - `kernels::advect_scalar(gpu, cache, batch, u, velocity: &StaggeredField, src: &Field, dst: &Field) -> Result<(), GpuError>`
  - crate-internal: `Bind`, `bind_group`, `expect_dims`, `COMMON`, `VELOCITY`

  Every kernel function has the shape `(gpu: &GpuContext, cache: &mut PipelineCache, batch: &mut ComputeBatch, u: &Uniforms, …)` and *records*; nothing runs until the batch is submitted.

- [ ] **Step 1: Write the shared WGSL**

`src/kernels/shaders/common.wgsl`:

```wgsl
// Shared by every Ember kernel except the sphere emitter. Concatenated in
// front of each kernel's own source. Each kernel declares its own
// `params: Params` binding; module-scope order does not matter in WGSL.

struct Params {
    dims: vec3<u32>, // the domain in cells
    axis: u32,       // 0, 1, 2 for x, y, z; ignored by kernels without an axis
    h: f32,          // substep length, seconds
    inv_dx: f32,     // 1 / voxel edge, 1/m
    dx2: f32,        // voxel edge squared, m²
    alpha: f32,      // buoyancy per unit density (sinks), m/s²
    beta: f32,       // buoyancy per unit temperature (rises), m/s²
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// Trilinear interpolation of `tex` at `p`, in the texture's own texel index
// space: texel (i, j, k) sits at p = (i, j, k). Positions outside clamp to the
// edge. Done by hand because R32Float is not filterable without an optional
// feature (umbrella E2).
fn trilinear(tex: texture_3d<f32>, p: vec3<f32>) -> f32 {
    let last = vec3<i32>(textureDimensions(tex, 0)) - vec3<i32>(1);
    let q = clamp(p, vec3<f32>(0.0), vec3<f32>(last));
    let i0 = vec3<i32>(floor(q));
    let i1 = min(i0 + vec3<i32>(1), last);
    let t = q - vec3<f32>(i0);
    let c000 = textureLoad(tex, vec3<i32>(i0.x, i0.y, i0.z), 0).x;
    let c100 = textureLoad(tex, vec3<i32>(i1.x, i0.y, i0.z), 0).x;
    let c010 = textureLoad(tex, vec3<i32>(i0.x, i1.y, i0.z), 0).x;
    let c110 = textureLoad(tex, vec3<i32>(i1.x, i1.y, i0.z), 0).x;
    let c001 = textureLoad(tex, vec3<i32>(i0.x, i0.y, i1.z), 0).x;
    let c101 = textureLoad(tex, vec3<i32>(i1.x, i0.y, i1.z), 0).x;
    let c011 = textureLoad(tex, vec3<i32>(i0.x, i1.y, i1.z), 0).x;
    let c111 = textureLoad(tex, vec3<i32>(i1.x, i1.y, i1.z), 0).x;
    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}

// From a position in cell units to a face texture's texel index space.
fn face_offset(axis: u32) -> vec3<f32> {
    var o = vec3<f32>(0.5, 0.5, 0.5);
    o[axis] = 0.0;
    return o;
}

// The texel dims of `axis`'s face texture: one more than the domain along it.
fn face_dims(axis: u32) -> vec3<u32> {
    var d = params.dims;
    d[axis] = d[axis] + 1u;
    return d;
}

// Whether face `i` along `axis` is a solid wall. The top face along z is open.
fn is_wall(axis: u32, i: u32) -> bool {
    let n = params.dims[axis];
    if (axis == 2u && i == n) {
        return false;
    }
    return i == 0u || i == n;
}
```

`src/kernels/shaders/velocity.wgsl`:

```wgsl
// Velocity at a cell-unit position. For kernels that declare `vel_x`,
// `vel_y` and `vel_z` as `texture_3d<f32>`.
fn velocity_at(x: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        trilinear(vel_x, x - face_offset(0u)),
        trilinear(vel_y, x - face_offset(1u)),
        trilinear(vel_z, x - face_offset(2u)),
    );
}
```

`src/kernels/shaders/advect_velocity.wgsl`:

```wgsl
// Stage 3: semi-Lagrangian advection of one velocity component. Dispatched
// once per axis, over that axis's face dims, writing a fresh face.

@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var<uniform> params: Params;

fn component(axis: u32, p: vec3<f32>) -> f32 {
    switch axis {
        case 0u: { return trilinear(vel_x, p); }
        case 1u: { return trilinear(vel_y, p); }
        default: { return trilinear(vel_z, p); }
    }
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = params.axis;
    if (any(gid >= face_dims(axis))) {
        return;
    }
    let p = vec3<i32>(gid);
    // Solid walls carry no normal velocity, before projection as well as
    // after, so the divergence the solve sees matches what projection enforces.
    if (is_wall(axis, gid[axis])) {
        textureStore(dst, p, vec4<f32>(0.0));
        return;
    }
    let x = vec3<f32>(gid) + face_offset(axis);
    let back = x - velocity_at(x) * (params.h * params.inv_dx);
    textureStore(dst, p, vec4<f32>(component(axis, back - face_offset(axis)), 0.0, 0.0, 0.0));
}
```

`src/kernels/shaders/advect_scalar.wgsl`:

```wgsl
// Stage 5: semi-Lagrangian advection of a cell-centred scalar.

@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var src: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let x = vec3<f32>(gid) + vec3<f32>(0.5);
    let back = x - velocity_at(x) * (params.h * params.inv_dx);
    let value = trilinear(src, back - vec3<f32>(0.5));
    textureStore(dst, vec3<i32>(gid), vec4<f32>(value, 0.0, 0.0, 0.0));
}
```

- [ ] **Step 2: Write the Rust side**

`src/kernels/mod.rs`:

```rust
//! Solver kernels. Each function records dispatches into a `ComputeBatch`;
//! nothing runs until the batch is submitted.
//!
//! Every function has the shape `(gpu, cache, batch, uniforms, fields…)`.
//! Conventions (indexing, face positions, boundaries) are in the plan and in
//! `shaders/common.wgsl`.

mod advect;

pub use advect::{advect_scalar, advect_velocity};

use elements_core::gpu::{Axis, Field, FieldDims, GpuContext, GpuError};
use wgpu::util::DeviceExt;

pub(crate) const COMMON: &str = include_str!("shaders/common.wgsl");

/// Values every kernel in one substep shares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepConstants {
    /// The domain in cells.
    pub cells: FieldDims,
    /// Substep length, seconds.
    pub h: f32,
    /// Voxel edge, metres.
    pub dx: f32,
    /// Buoyancy per unit density (sinks), m/s².
    pub alpha: f32,
    /// Buoyancy per unit temperature (rises), m/s².
    pub beta: f32,
}

/// Matches `Params` in `common.wgsl`, 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KernelParams {
    dims: [u32; 3],
    axis: u32,
    h: f32,
    inv_dx: f32,
    dx2: f32,
    alpha: f32,
    beta: f32,
    _pad: [u32; 3],
}

pub(crate) fn axis_index(axis: Axis) -> u32 {
    match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

/// One uniform buffer per axis, identical except for `axis`. Built once per
/// substep and shared by every kernel in it.
pub struct Uniforms {
    per_axis: [wgpu::Buffer; 3],
    cells: FieldDims,
}

impl Uniforms {
    pub fn new(gpu: &GpuContext, c: &StepConstants) -> Result<Self, GpuError> {
        let make = |axis: u32| {
            let params = KernelParams {
                dims: [c.cells.x, c.cells.y, c.cells.z],
                axis,
                h: c.h,
                inv_dx: 1.0 / c.dx,
                dx2: c.dx * c.dx,
                alpha: c.alpha,
                beta: c.beta,
                _pad: [0; 3],
            };
            gpu.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ember-params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        };
        let per_axis = gpu.scoped(|| [make(0), make(1), make(2)])?;
        Ok(Self {
            per_axis,
            cells: c.cells,
        })
    }

    /// The domain these constants describe.
    pub fn cells(&self) -> FieldDims {
        self.cells
    }

    pub(crate) fn axis(&self, axis: Axis) -> &wgpu::Buffer {
        &self.per_axis[axis_index(axis) as usize]
    }

    /// For kernels without an axis.
    pub(crate) fn any(&self) -> &wgpu::Buffer {
        &self.per_axis[0]
    }
}

/// One bind group entry. Entries are bound at 0, 1, 2… in order.
pub(crate) enum Bind<'a> {
    Tex(&'a Field),
    Buf(&'a wgpu::Buffer),
}

pub(crate) fn bind_group(
    gpu: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    entries: &[Bind<'_>],
) -> Result<wgpu::BindGroup, GpuError> {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: match entry {
                Bind::Tex(field) => wgpu::BindingResource::TextureView(field.view()),
                Bind::Buf(buffer) => buffer.as_entire_binding(),
            },
        })
        .collect();
    gpu.scoped(|| {
        gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ember"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    })
}

/// A mis-sized field would make a kernel read the wrong texels with no error.
pub(crate) fn expect_dims(what: &str, field: &Field, dims: FieldDims) -> Result<(), GpuError> {
    if field.dims() == dims {
        Ok(())
    } else {
        Err(GpuError::Validation(format!(
            "{what} is {:?}, expected {dims:?}",
            field.dims()
        )))
    }
}
```

`src/kernels/advect.rs`:

```rust
//! Semi-Lagrangian advection (stages 3 and 5).

use elements_core::gpu::{Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField};

use super::{Bind, Uniforms, bind_group, expect_dims};

const ADVECT_VELOCITY: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/velocity.wgsl"),
    include_str!("shaders/advect_velocity.wgsl"),
);

const ADVECT_SCALAR: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/velocity.wgsl"),
    include_str!("shaders/advect_scalar.wgsl"),
);

/// Advect `src` through itself into `dst`. Solid-wall faces of `dst` are zero.
pub fn advect_velocity(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    src: &StaggeredField,
    dst: &StaggeredField,
) -> Result<(), GpuError> {
    if src.cells() != u.cells() || dst.cells() != u.cells() {
        return Err(GpuError::Validation(format!(
            "advect_velocity: src {:?}, dst {:?}, domain {:?}",
            src.cells(),
            dst.cells(),
            u.cells()
        )));
    }
    let pipeline = cache.get_or_create(gpu, "ember.advect_velocity", ADVECT_VELOCITY, "main")?;
    for axis in Axis::ALL {
        let group = bind_group(
            gpu,
            &pipeline,
            &[
                Bind::Tex(src.face(Axis::X)),
                Bind::Tex(src.face(Axis::Y)),
                Bind::Tex(src.face(Axis::Z)),
                Bind::Tex(dst.face(axis)),
                Bind::Buf(u.axis(axis)),
            ],
        )?;
        batch.dispatch(&pipeline, &group, dst.face(axis).dims());
    }
    Ok(())
}

/// Advect the cell-centred `src` through `velocity` into `dst`.
pub fn advect_scalar(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    src: &Field,
    dst: &Field,
) -> Result<(), GpuError> {
    let cells = u.cells();
    if velocity.cells() != cells {
        return Err(GpuError::Validation(format!(
            "advect_scalar: velocity {:?}, domain {cells:?}",
            velocity.cells()
        )));
    }
    expect_dims("advect_scalar src", src, cells)?;
    expect_dims("advect_scalar dst", dst, cells)?;
    let pipeline = cache.get_or_create(gpu, "ember.advect_scalar", ADVECT_SCALAR, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(velocity.face(Axis::X)),
            Bind::Tex(velocity.face(Axis::Y)),
            Bind::Tex(velocity.face(Axis::Z)),
            Bind::Tex(src),
            Bind::Tex(dst),
            Bind::Buf(u.any()),
        ],
    )?;
    batch.dispatch(&pipeline, &group, cells);
    Ok(())
}
```

`COMMON` in `mod.rs` is used by later tasks; if clippy flags it as dead in this task, add `#[allow(dead_code)]` on it and remove that attribute in Task 6.

- [ ] **Step 3: Write the CPU references and failing tests**

Append to `tests/common/mod.rs`:

```rust
use elements_core::gpu::{Axis, StaggeredField};

pub const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

pub fn face_dims(cells: FieldDims, axis: usize) -> FieldDims {
    StaggeredField::face_dims(cells, AXES[axis])
}

/// Mirrors `face_offset` in `common.wgsl`.
pub fn face_offset(axis: usize) -> [f32; 3] {
    let mut o = [0.5; 3];
    o[axis] = 0.0;
    o
}

/// Mirrors `is_wall` in `common.wgsl`.
pub fn is_wall(cells: FieldDims, axis: usize, i: u32) -> bool {
    let n = [cells.x, cells.y, cells.z][axis];
    if axis == 2 && i == n {
        return false;
    }
    i == 0 || i == n
}

/// Mirrors `trilinear` in `common.wgsl`, including its edge clamp.
pub fn trilinear(data: &[f32], dims: FieldDims, p: [f32; 3]) -> f32 {
    let size = [dims.x as i32, dims.y as i32, dims.z as i32];
    let mut i0 = [0i32; 3];
    let mut i1 = [0i32; 3];
    let mut t = [0f32; 3];
    for a in 0..3 {
        let last = size[a] - 1;
        let q = p[a].clamp(0.0, last as f32);
        i0[a] = q.floor() as i32;
        i1[a] = (i0[a] + 1).min(last);
        t[a] = q - i0[a] as f32;
    }
    let at = |x: i32, y: i32, z: i32| data[index(dims, x as u32, y as u32, z as u32)];
    let mix = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;
    let c00 = mix(at(i0[0], i0[1], i0[2]), at(i1[0], i0[1], i0[2]), t[0]);
    let c10 = mix(at(i0[0], i1[1], i0[2]), at(i1[0], i1[1], i0[2]), t[0]);
    let c01 = mix(at(i0[0], i0[1], i1[2]), at(i1[0], i0[1], i1[2]), t[0]);
    let c11 = mix(at(i0[0], i1[1], i1[2]), at(i1[0], i1[1], i1[2]), t[0]);
    mix(mix(c00, c10, t[1]), mix(c01, c11, t[1]), t[2])
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Mirrors `velocity_at` in `velocity.wgsl`.
pub fn velocity_at(faces: &[Vec<f32>; 3], cells: FieldDims, x: [f32; 3]) -> [f32; 3] {
    let mut v = [0.0; 3];
    for (a, out) in v.iter_mut().enumerate() {
        *out = trilinear(&faces[a], face_dims(cells, a), sub(x, face_offset(a)));
    }
    v
}

fn backtrace(faces: &[Vec<f32>; 3], cells: FieldDims, x: [f32; 3], h_inv_dx: f32) -> [f32; 3] {
    let v = velocity_at(faces, cells, x);
    [x[0] - v[0] * h_inv_dx, x[1] - v[1] * h_inv_dx, x[2] - v[2] * h_inv_dx]
}

/// Mirrors `advect_scalar.wgsl`.
pub fn cpu_advect_scalar(faces: &[Vec<f32>; 3], cells: FieldDims, src: &[f32], h_inv_dx: f32) -> Vec<f32> {
    let mut out = vec![0.0; cells.voxel_count()];
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let x = [i as f32 + 0.5, j as f32 + 0.5, k as f32 + 0.5];
                let b = backtrace(faces, cells, x, h_inv_dx);
                out[index(cells, i, j, k)] = trilinear(src, cells, sub(b, [0.5; 3]));
            }
        }
    }
    out
}

/// Mirrors `advect_velocity.wgsl` for all three faces.
pub fn cpu_advect_velocity(faces: &[Vec<f32>; 3], cells: FieldDims, h_inv_dx: f32) -> [Vec<f32>; 3] {
    std::array::from_fn(|a| {
        let d = face_dims(cells, a);
        let mut out = vec![0.0; d.voxel_count()];
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    if is_wall(cells, a, [i, j, k][a]) {
                        continue;
                    }
                    let x = [i as f32, j as f32, k as f32];
                    let off = face_offset(a);
                    let x = [x[0] + off[0], x[1] + off[1], x[2] + off[2]];
                    let b = backtrace(faces, cells, x, h_inv_dx);
                    out[index(d, i, j, k)] = trilinear(&faces[a], d, sub(b, off));
                }
            }
        }
        out
    })
}

/// A staggered field holding `faces` (X, Y, Z order).
pub fn upload_staggered(
    gpu: &GpuContext,
    pool: &mut FieldPool,
    cells: FieldDims,
    faces: &[Vec<f32>; 3],
) -> StaggeredField {
    let [x, y, z] = std::array::from_fn(|a| upload(gpu, pool, face_dims(cells, a), &faces[a]));
    StaggeredField::from_faces(cells, [x, y, z]).unwrap()
}

pub fn read_staggered(gpu: &GpuContext, v: &StaggeredField) -> [Vec<f32>; 3] {
    std::array::from_fn(|a| v.face(AXES[a]).read_back(gpu).unwrap())
}

/// Smooth-ish velocity faces with values in about [-0.6, 0.6] m/s.
pub fn velocity_pattern(cells: FieldDims) -> [Vec<f32>; 3] {
    std::array::from_fn(|a| {
        pattern(face_dims(cells, a), a as u32 + 11)
            .iter()
            .map(|v| 0.6 * v)
            .collect()
    })
}
```

Create `tests/advection.rs`:

```rust
mod common;

use common::*;
use elements_core::gpu::{ComputeBatch, FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_ember::kernels::{StepConstants, Uniforms, advect_scalar, advect_velocity};

const CELLS: FieldDims = FieldDims { x: 8, y: 6, z: 5 };

fn constants(h: f32, dx: f32) -> StepConstants {
    StepConstants {
        cells: CELLS,
        h,
        dx,
        alpha: 0.0,
        beta: 0.0,
    }
}

/// A uniform velocity of exactly two voxels per substep along x moves a
/// scalar by exactly two cells: every interpolation weight is 0 or 1.
#[test]
fn an_integer_uniform_velocity_shifts_a_scalar_exactly() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    // Powers of two, so u * h / dx is exactly 2.
    let c = constants(0.25, 0.125);
    let faces = [
        vec![1.0; face_dims(CELLS, 0).voxel_count()],
        vec![0.0; face_dims(CELLS, 1).voxel_count()],
        vec![0.0; face_dims(CELLS, 2).voxel_count()],
    ];
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let src_values = pattern(CELLS, 1);
    let src = upload(&gpu, &mut pool, CELLS, &src_values);
    let dst = pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap();

    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    advect_scalar(&gpu, &mut cache, &mut batch, &u, &velocity, &src, &dst).unwrap();
    batch.submit(&gpu).unwrap();

    let out = dst.read_back(&gpu).unwrap();
    for k in 0..CELLS.z {
        for j in 0..CELLS.y {
            for i in 0..CELLS.x {
                // Backtraces left of the domain clamp to its first column.
                let from = i.saturating_sub(2);
                assert_eq!(
                    out[index(CELLS, i, j, k)].to_bits(),
                    src_values[index(CELLS, from, j, k)].to_bits(),
                    "cell ({i}, {j}, {k})"
                );
            }
        }
    }
}

#[test]
fn a_fractional_velocity_advects_a_scalar_like_the_cpu_reference() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = constants(0.1, 0.125);
    let faces = velocity_pattern(CELLS);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let src_values = pattern(CELLS, 2);
    let src = upload(&gpu, &mut pool, CELLS, &src_values);
    let dst = pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap();

    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    advect_scalar(&gpu, &mut cache, &mut batch, &u, &velocity, &src, &dst).unwrap();
    batch.submit(&gpu).unwrap();

    let want = cpu_advect_scalar(&faces, CELLS, &src_values, c.h / c.dx);
    assert_close(&dst.read_back(&gpu).unwrap(), &want, 1e-5, "advected scalar");
}

#[test]
fn velocity_advection_matches_the_cpu_reference_and_zeroes_solid_walls() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = constants(0.1, 0.125);
    let faces = velocity_pattern(CELLS);
    let src = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let dst = pool.acquire_staggered_uninit(&gpu, CELLS).unwrap();

    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    advect_velocity(&gpu, &mut cache, &mut batch, &u, &src, &dst).unwrap();
    batch.submit(&gpu).unwrap();

    let got = read_staggered(&gpu, &dst);
    let want = cpu_advect_velocity(&faces, CELLS, c.h / c.dx);
    for a in 0..3 {
        assert_close(&got[a], &want[a], 1e-5, &format!("face {a}"));
    }
    // The open top is advected, not zeroed: the pattern makes it nonzero.
    let d = face_dims(CELLS, 2);
    let top: Vec<f32> = (0..CELLS.y)
        .flat_map(|j| (0..CELLS.x).map(move |i| (i, j)))
        .map(|(i, j)| got[2][index(d, i, j, CELLS.z)])
        .collect();
    assert!(top.iter().any(|&v| v != 0.0), "the open top must not be a wall");
}
```

Also add, at the end of `cpu_advect_velocity`'s doc, nothing: wall faces are left at `0.0` in `out`, which is what the kernel writes, so `assert_close` checks walls exactly.

- [ ] **Step 4: Run the tests, then `just check`**

Run: `cargo nextest run -p elements-ember --test advection`
Expected: PASS.

- [ ] **Step 5: Prove each test can fail**

| Test | Mutation |
|---|---|
| `an_integer_uniform_velocity_shifts_…` | in `advect_scalar.wgsl`, `let back = x - velocity_at(x) …` → `x + velocity_at(x) …` |
| `a_fractional_velocity_advects_…` | in `velocity.wgsl`, `trilinear(vel_y, x - face_offset(1u))` → `face_offset(2u)` |
| `velocity_advection_matches_…` | in `common.wgsl`'s `is_wall`, delete the `if (axis == 2u && i == n) { return false; }` block |

- [ ] **Step 6: Commit and push**

Subject: `Add semi-Lagrangian advection kernels`.

---

### Task 6: Emit and buoyancy kernels

**Files:**
- Create: `src/kernels/forces.rs`, `src/kernels/shaders/add_scaled.wgsl`, `src/kernels/shaders/buoyancy.wgsl`
- Modify: `src/kernels/mod.rs` (`mod forces; pub use forces::{emit, buoyancy};`)
- Test: `tests/forces.rs`

**Interfaces:**
- Produces: `kernels::emit(gpu, cache, batch, u, dst: &Field, src: &Field)` (stage 1, `dst += src · h`); `kernels::buoyancy(gpu, cache, batch, u, velocity_z: &Field, density: &Field, temperature: &Field)` (stage 2).

- [ ] **Step 1: Write the shaders and Rust**

`src/kernels/shaders/add_scaled.wgsl`:

```wgsl
// Stage 1: dst += src * h. Emission: `src` is a rate per second.

@group(0) @binding(0) var dst: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let next = textureLoad(dst, p).x + textureLoad(src, p, 0).x * params.h;
    textureStore(dst, p, vec4<f32>(next, 0.0, 0.0, 0.0));
}
```

`src/kernels/shaders/buoyancy.wgsl`:

```wgsl
// Stage 2: Boussinesq buoyancy on the z faces, w += h * (beta * T - alpha * rho),
// with density and temperature averaged from the two cells each face separates.
// Dispatched over the z-face dims.

@group(0) @binding(0) var vel_z: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var density: texture_3d<f32>;
@group(0) @binding(2) var temperature: texture_3d<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= face_dims(2u))) {
        return;
    }
    // The floor is a solid wall, and the open top's value comes from projection.
    if (gid.z == 0u || gid.z == params.dims.z) {
        return;
    }
    let above = vec3<i32>(gid);
    let below = above - vec3<i32>(0, 0, 1);
    let rho = 0.5 * (textureLoad(density, below, 0).x + textureLoad(density, above, 0).x);
    let temp = 0.5 * (textureLoad(temperature, below, 0).x + textureLoad(temperature, above, 0).x);
    let w = textureLoad(vel_z, above).x + params.h * (params.beta * temp - params.alpha * rho);
    textureStore(vel_z, above, vec4<f32>(w, 0.0, 0.0, 0.0));
}
```

`src/kernels/forces.rs`:

```rust
//! Emission and buoyancy (stages 1 and 2).

use elements_core::gpu::{Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField};

use super::{Bind, Uniforms, bind_group, expect_dims};

const ADD_SCALED: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/add_scaled.wgsl"),
);

const BUOYANCY: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/buoyancy.wgsl"),
);

/// `dst += src * h`.
pub fn emit(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    dst: &Field,
    src: &Field,
) -> Result<(), GpuError> {
    expect_dims("emit dst", dst, u.cells())?;
    expect_dims("emit source", src, u.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.add_scaled", ADD_SCALED, "main")?;
    let group = bind_group(gpu, &pipeline, &[Bind::Tex(dst), Bind::Tex(src), Bind::Buf(u.any())])?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// Add buoyancy to the z faces of the velocity.
pub fn buoyancy(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity_z: &Field,
    density: &Field,
    temperature: &Field,
) -> Result<(), GpuError> {
    let cells = u.cells();
    expect_dims("buoyancy z face", velocity_z, StaggeredField::face_dims(cells, Axis::Z))?;
    expect_dims("buoyancy density", density, cells)?;
    expect_dims("buoyancy temperature", temperature, cells)?;
    let pipeline = cache.get_or_create(gpu, "ember.buoyancy", BUOYANCY, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(velocity_z),
            Bind::Tex(density),
            Bind::Tex(temperature),
            Bind::Buf(u.any()),
        ],
    )?;
    batch.dispatch(&pipeline, &group, velocity_z.dims());
    Ok(())
}
```

If `COMMON` in `mod.rs` is still unused, delete it (the kernels use `include_str!` directly).

- [ ] **Step 2: Write the failing tests**

`tests/forces.rs`:

```rust
mod common;

use common::*;
use elements_core::gpu::{ComputeBatch, FieldDims, FieldPool, PipelineCache};
use elements_ember::kernels::{StepConstants, Uniforms, buoyancy, emit};

const CELLS: FieldDims = FieldDims { x: 8, y: 6, z: 5 };

fn constants() -> StepConstants {
    StepConstants {
        cells: CELLS,
        h: 0.25,
        dx: 0.125,
        alpha: 0.5,
        beta: 2.0,
    }
}

#[test]
fn emit_adds_the_source_rate_times_the_substep() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let before = pattern(CELLS, 3);
    let rate = pattern(CELLS, 4);
    let dst = upload(&gpu, &mut pool, CELLS, &before);
    let src = upload(&gpu, &mut pool, CELLS, &rate);

    let u = Uniforms::new(&gpu, &constants()).unwrap();
    let mut batch = ComputeBatch::new();
    emit(&gpu, &mut cache, &mut batch, &u, &dst, &src).unwrap();
    batch.submit(&gpu).unwrap();

    let want: Vec<f32> = before.iter().zip(&rate).map(|(b, r)| b + r * 0.25).collect();
    assert_close(&dst.read_back(&gpu).unwrap(), &want, 1e-6, "emitted");
}

#[test]
fn buoyancy_matches_the_cpu_reference_and_leaves_boundary_faces_alone() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = constants();
    let zd = face_dims(CELLS, 2);
    let w_before = pattern(zd, 5);
    let rho = pattern(CELLS, 6);
    let temp = pattern(CELLS, 7);
    let w = upload(&gpu, &mut pool, zd, &w_before);
    let density = upload(&gpu, &mut pool, CELLS, &rho);
    let temperature = upload(&gpu, &mut pool, CELLS, &temp);

    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    buoyancy(&gpu, &mut cache, &mut batch, &u, &w, &density, &temperature).unwrap();
    batch.submit(&gpu).unwrap();

    let mut want = w_before.clone();
    for k in 1..CELLS.z {
        for j in 0..CELLS.y {
            for i in 0..CELLS.x {
                let r = 0.5 * (rho[index(CELLS, i, j, k - 1)] + rho[index(CELLS, i, j, k)]);
                let t = 0.5 * (temp[index(CELLS, i, j, k - 1)] + temp[index(CELLS, i, j, k)]);
                want[index(zd, i, j, k)] += c.h * (c.beta * t - c.alpha * r);
            }
        }
    }
    let got = w.read_back(&gpu).unwrap();
    assert_close(&got, &want, 1e-5, "w after buoyancy");
    for j in 0..CELLS.y {
        for i in 0..CELLS.x {
            for k in [0, CELLS.z] {
                assert_eq!(
                    got[index(zd, i, j, k)].to_bits(),
                    w_before[index(zd, i, j, k)].to_bits(),
                    "boundary face ({i}, {j}, {k}) must be untouched"
                );
            }
        }
    }
}
```

- [ ] **Step 3: Run the tests, then `just check`**

Run: `cargo nextest run -p elements-ember --test forces`
Expected: PASS.

- [ ] **Step 4: Prove each test can fail**

| Test | Mutation |
|---|---|
| `emit_adds_the_source_rate_…` | in `add_scaled.wgsl`, `* params.h` → `* params.inv_dx` |
| `buoyancy_matches_…` | in `buoyancy.wgsl`'s `rho` line, `textureLoad(density, below, 0)` → `textureLoad(density, above, 0)` |

- [ ] **Step 5: Commit and push**

Subject: `Add emission and buoyancy kernels`.

---

### Task 7: Projection kernels

**Files:**
- Create: `src/kernels/project.rs`, `src/kernels/shaders/divergence.wgsl`, `src/kernels/shaders/pressure.wgsl`, `src/kernels/shaders/gradient.wgsl`
- Modify: `src/kernels/mod.rs` (`mod project; pub use project::{divergence, pressure, subtract_gradient};`)
- Modify: `tests/common/mod.rs` (CPU red-black reference)
- Test: `tests/projection.rs`

**Interfaces:**
- Produces: `kernels::divergence(gpu, cache, batch, u, velocity: &StaggeredField, div: &Field)`; `kernels::pressure(gpu, cache, batch, u, phi: &Field, div: &Field, iterations: u32)`; `kernels::subtract_gradient(gpu, cache, batch, u, velocity: &StaggeredField, phi: &Field)`.

- [ ] **Step 1: Write the shaders**

`src/kernels/shaders/divergence.wgsl`:

```wgsl
// Stage 4a: divergence of the staggered velocity, per cell, in 1/s.

@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var div: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    let du = textureLoad(vel_x, c + vec3<i32>(1, 0, 0), 0).x - textureLoad(vel_x, c, 0).x;
    let dv = textureLoad(vel_y, c + vec3<i32>(0, 1, 0), 0).x - textureLoad(vel_y, c, 0).x;
    let dw = textureLoad(vel_z, c + vec3<i32>(0, 0, 1), 0).x - textureLoad(vel_z, c, 0).x;
    textureStore(div, c, vec4<f32>((du + dv + dw) * params.inv_dx, 0.0, 0.0, 0.0));
}
```

`src/kernels/shaders/pressure.wgsl`:

```wgsl
// Stage 4b: red-black Gauss–Seidel for ∇²φ = div, where φ = h·p.
// Cells of one colour never read each other, so each sweep is free of races
// and bit-deterministic.

@group(0) @binding(0) var phi: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var div: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn relax(gid: vec3<u32>, colour: u32) {
    if (any(gid >= params.dims) || ((gid.x + gid.y + gid.z) & 1u) != colour) {
        return;
    }
    let c = vec3<i32>(gid);
    let n = vec3<i32>(params.dims);
    var sum = 0.0;
    var count = 0.0;
    // Solid walls are Neumann: a neighbour outside the domain is left out.
    if (c.x > 0) { sum += textureLoad(phi, c - vec3<i32>(1, 0, 0)).x; count += 1.0; }
    if (c.x < n.x - 1) { sum += textureLoad(phi, c + vec3<i32>(1, 0, 0)).x; count += 1.0; }
    if (c.y > 0) { sum += textureLoad(phi, c - vec3<i32>(0, 1, 0)).x; count += 1.0; }
    if (c.y < n.y - 1) { sum += textureLoad(phi, c + vec3<i32>(0, 1, 0)).x; count += 1.0; }
    if (c.z > 0) { sum += textureLoad(phi, c - vec3<i32>(0, 0, 1)).x; count += 1.0; }
    // Above: an interior neighbour, or the open top, where φ = 0. Either way it counts.
    if (c.z < n.z - 1) { sum += textureLoad(phi, c + vec3<i32>(0, 0, 1)).x; }
    count += 1.0;
    let value = (sum - params.dx2 * textureLoad(div, c, 0).x) / count;
    textureStore(phi, c, vec4<f32>(value, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn red(@builtin(global_invocation_id) gid: vec3<u32>) {
    relax(gid, 0u);
}

@compute @workgroup_size(4, 4, 4)
fn black(@builtin(global_invocation_id) gid: vec3<u32>) {
    relax(gid, 1u);
}
```

`src/kernels/shaders/gradient.wgsl`:

```wgsl
// Stage 4c: u -= ∇φ on one axis's faces. Dispatched once per axis over that
// axis's face dims. Face i lies between cells i - 1 and i.

@group(0) @binding(0) var face: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var phi: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = params.axis;
    if (any(gid >= face_dims(axis))) {
        return;
    }
    let p = vec3<i32>(gid);
    let i = gid[axis];
    if (is_wall(axis, i)) {
        textureStore(face, p, vec4<f32>(0.0));
        return;
    }
    var e = vec3<i32>(0);
    e[axis] = 1;
    // Above the open top, φ = 0.
    var upper = 0.0;
    if (i < params.dims[axis]) {
        upper = textureLoad(phi, p, 0).x;
    }
    let lower = textureLoad(phi, p - e, 0).x;
    let u = textureLoad(face, p).x - (upper - lower) * params.inv_dx;
    textureStore(face, p, vec4<f32>(u, 0.0, 0.0, 0.0));
}
```

- [ ] **Step 2: Write the Rust side**

`src/kernels/project.rs`:

```rust
//! Pressure projection (stage 4).

use elements_core::gpu::{Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField};

use super::{Bind, Uniforms, bind_group, expect_dims};

const DIVERGENCE: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/divergence.wgsl"),
);

const PRESSURE: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/pressure.wgsl"),
);

const GRADIENT: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/gradient.wgsl"),
);

fn expect_velocity(what: &str, velocity: &StaggeredField, u: &Uniforms) -> Result<(), GpuError> {
    if velocity.cells() == u.cells() {
        Ok(())
    } else {
        Err(GpuError::Validation(format!(
            "{what}: velocity {:?}, domain {:?}",
            velocity.cells(),
            u.cells()
        )))
    }
}

/// Write the divergence of `velocity` into `div`.
pub fn divergence(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    div: &Field,
) -> Result<(), GpuError> {
    expect_velocity("divergence", velocity, u)?;
    expect_dims("divergence output", div, u.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.divergence", DIVERGENCE, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(velocity.face(Axis::X)),
            Bind::Tex(velocity.face(Axis::Y)),
            Bind::Tex(velocity.face(Axis::Z)),
            Bind::Tex(div),
            Bind::Buf(u.any()),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// `iterations` red-black Gauss–Seidel sweeps on `phi`, in place, starting
/// from whatever `phi` holds (the warm start).
pub fn pressure(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    phi: &Field,
    div: &Field,
    iterations: u32,
) -> Result<(), GpuError> {
    expect_dims("pressure phi", phi, u.cells())?;
    expect_dims("pressure divergence", div, u.cells())?;
    let red = cache.get_or_create(gpu, "ember.pressure.red", PRESSURE, "red")?;
    let black = cache.get_or_create(gpu, "ember.pressure.black", PRESSURE, "black")?;
    // Auto layouts are never shared between pipelines, so each colour needs
    // its own bind group. Both are built once and reused every iteration.
    let entries = [Bind::Tex(phi), Bind::Tex(div), Bind::Buf(u.any())];
    let red_group = bind_group(gpu, &red, &entries)?;
    let black_group = bind_group(gpu, &black, &entries)?;
    for _ in 0..iterations {
        batch.dispatch(&red, &red_group, u.cells());
        batch.dispatch(&black, &black_group, u.cells());
    }
    Ok(())
}

/// `velocity -= ∇phi`, with solid-wall faces forced to zero.
pub fn subtract_gradient(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    phi: &Field,
) -> Result<(), GpuError> {
    expect_velocity("subtract_gradient", velocity, u)?;
    expect_dims("subtract_gradient phi", phi, u.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.gradient", GRADIENT, "main")?;
    for axis in Axis::ALL {
        let face = velocity.face(axis);
        let group = bind_group(gpu, &pipeline, &[Bind::Tex(face), Bind::Tex(phi), Bind::Buf(u.axis(axis))])?;
        batch.dispatch(&pipeline, &group, face.dims());
    }
    Ok(())
}
```

- [ ] **Step 3: Write the CPU reference and failing tests**

Append to `tests/common/mod.rs`:

```rust
/// Mirrors `relax` in `pressure.wgsl`: red (even i+j+k) then black, per iteration.
pub fn cpu_red_black(phi: &mut [f32], div: &[f32], cells: FieldDims, dx2: f32, iterations: u32) {
    let (nx, ny, nz) = (cells.x, cells.y, cells.z);
    for _ in 0..iterations {
        for colour in [0, 1] {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        if (i + j + k) % 2 != colour {
                            continue;
                        }
                        let mut sum = 0.0f32;
                        let mut count = 0.0f32;
                        if i > 0 { sum += phi[index(cells, i - 1, j, k)]; count += 1.0; }
                        if i < nx - 1 { sum += phi[index(cells, i + 1, j, k)]; count += 1.0; }
                        if j > 0 { sum += phi[index(cells, i, j - 1, k)]; count += 1.0; }
                        if j < ny - 1 { sum += phi[index(cells, i, j + 1, k)]; count += 1.0; }
                        if k > 0 { sum += phi[index(cells, i, j, k - 1)]; count += 1.0; }
                        if k < nz - 1 { sum += phi[index(cells, i, j, k + 1)]; }
                        count += 1.0;
                        phi[index(cells, i, j, k)] = (sum - dx2 * div[index(cells, i, j, k)]) / count;
                    }
                }
            }
        }
    }
}

/// Max |div| over every cell of `faces`, computed on the CPU.
pub fn cpu_max_divergence(faces: &[Vec<f32>; 3], cells: FieldDims, dx: f32) -> f32 {
    let mut worst = 0.0f32;
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let (xd, yd, zd) = (face_dims(cells, 0), face_dims(cells, 1), face_dims(cells, 2));
                let d = (faces[0][index(xd, i + 1, j, k)] - faces[0][index(xd, i, j, k)]
                    + faces[1][index(yd, i, j + 1, k)] - faces[1][index(yd, i, j, k)]
                    + faces[2][index(zd, i, j, k + 1)] - faces[2][index(zd, i, j, k)])
                    / dx;
                worst = worst.max(d.abs());
            }
        }
    }
    worst
}

/// `velocity_pattern` with every solid-wall face set to zero, as advection leaves it.
pub fn walled_velocity_pattern(cells: FieldDims) -> [Vec<f32>; 3] {
    let mut faces = velocity_pattern(cells);
    for (a, face) in faces.iter_mut().enumerate() {
        let d = face_dims(cells, a);
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    if is_wall(cells, a, [i, j, k][a]) {
                        face[index(d, i, j, k)] = 0.0;
                    }
                }
            }
        }
    }
    faces
}
```

`tests/projection.rs`:

```rust
mod common;

use common::*;
use elements_core::gpu::{ComputeBatch, FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_ember::kernels::{StepConstants, Uniforms, divergence, pressure, subtract_gradient};

const CELLS: FieldDims = FieldDims { x: 8, y: 6, z: 5 };

fn constants(dx: f32) -> StepConstants {
    StepConstants {
        cells: CELLS,
        h: 0.1,
        dx,
        alpha: 0.0,
        beta: 0.0,
    }
}

/// A velocity equal to position along one axis has divergence 1 everywhere.
/// Each axis is checked on its own, so a swapped offset cannot hide.
#[test]
fn a_linear_velocity_has_unit_divergence_along_each_axis() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    let dx = 0.125;
    for axis in 0..3 {
        let mut pool = FieldPool::new();
        let faces: [Vec<f32>; 3] = std::array::from_fn(|a| {
            let d = face_dims(CELLS, a);
            let mut face = vec![0.0; d.voxel_count()];
            if a == axis {
                for k in 0..d.z {
                    for j in 0..d.y {
                        for i in 0..d.x {
                            // Face position along its own axis, in metres.
                            face[index(d, i, j, k)] = [i, j, k][a] as f32 * dx;
                        }
                    }
                }
            }
            face
        });
        let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
        let div = pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap();
        let u = Uniforms::new(&gpu, &constants(dx)).unwrap();
        let mut batch = ComputeBatch::new();
        divergence(&gpu, &mut cache, &mut batch, &u, &velocity, &div).unwrap();
        batch.submit(&gpu).unwrap();
        let got = div.read_back(&gpu).unwrap();
        assert!(got.iter().all(|&v| v == 1.0), "axis {axis}: {:?}", &got[..4]);
    }
}

#[test]
fn red_black_sweeps_match_the_cpu_reference() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = constants(1.0);
    let div_values = pattern(CELLS, 8);
    let phi0 = pattern(CELLS, 9);
    let div = upload(&gpu, &mut pool, CELLS, &div_values);
    let phi = upload(&gpu, &mut pool, CELLS, &phi0);

    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    pressure(&gpu, &mut cache, &mut batch, &u, &phi, &div, 3).unwrap();
    batch.submit(&gpu).unwrap();

    let mut want = phi0.clone();
    cpu_red_black(&mut want, &div_values, CELLS, 1.0, 3);
    assert_close(&phi.read_back(&gpu).unwrap(), &want, 1e-4, "phi");
}

#[test]
fn subtracting_the_gradient_zeroes_solid_walls_and_matches_the_cpu() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let dx = 0.125;
    let faces = velocity_pattern(CELLS);
    let phi_values = pattern(CELLS, 10);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let phi = upload(&gpu, &mut pool, CELLS, &phi_values);

    let u = Uniforms::new(&gpu, &constants(dx)).unwrap();
    let mut batch = ComputeBatch::new();
    subtract_gradient(&gpu, &mut cache, &mut batch, &u, &velocity, &phi).unwrap();
    batch.submit(&gpu).unwrap();

    let got = read_staggered(&gpu, &velocity);
    for a in 0..3 {
        let d = face_dims(CELLS, a);
        let n = [CELLS.x, CELLS.y, CELLS.z][a];
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let ijk = [i, j, k];
                    let at = index(d, i, j, k);
                    if is_wall(CELLS, a, ijk[a]) {
                        assert_eq!(got[a][at], 0.0, "wall face {a} {ijk:?}");
                        continue;
                    }
                    let upper = if ijk[a] < n { phi_values[index(CELLS, i, j, k)] } else { 0.0 };
                    let mut low = ijk;
                    low[a] -= 1;
                    let lower = phi_values[index(CELLS, low[0], low[1], low[2])];
                    let want = faces[a][at] - (upper - lower) / dx;
                    assert!((got[a][at] - want).abs() <= 1e-4, "face {a} {ijk:?}: {} vs {want}", got[a][at]);
                }
            }
        }
    }
}

/// Divergence, a converged solve, then the gradient: the result is
/// divergence-free. This is what ties the stencil and the gradient together:
/// each is tested against its own reference above, but only this shows they
/// agree with each other at the walls and the open top.
#[test]
fn a_converged_projection_removes_divergence() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let dx = 0.125;
    let faces = walled_velocity_pattern(CELLS);
    let before = cpu_max_divergence(&faces, CELLS, dx);
    assert!(before > 1.0, "the test field must start divergent, got {before}");
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let div = pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap();
    let phi = pool.acquire_zeroed(&gpu, &mut cache, CELLS).unwrap();

    let u = Uniforms::new(&gpu, &constants(dx)).unwrap();
    let mut batch = ComputeBatch::new();
    divergence(&gpu, &mut cache, &mut batch, &u, &velocity, &div).unwrap();
    pressure(&gpu, &mut cache, &mut batch, &u, &phi, &div, 2000).unwrap();
    subtract_gradient(&gpu, &mut cache, &mut batch, &u, &velocity, &phi).unwrap();
    batch.submit(&gpu).unwrap();

    let after = cpu_max_divergence(&read_staggered(&gpu, &velocity), CELLS, dx);
    assert!(after < 1e-3 * before, "max |div| {before} -> {after}");
}
```

- [ ] **Step 4: Run the tests, then `just check`**

Run: `cargo nextest run -p elements-ember --test projection`
Expected: PASS. If `a_converged_projection_removes_divergence` fails with a correct-looking implementation, do not raise the iteration count or loosen `1e-3` to fit: report the measured `before` and `after`.

- [ ] **Step 5: Prove each test can fail**

| Test | Mutation |
|---|---|
| `a_linear_velocity_has_unit_divergence_…` | in `divergence.wgsl`'s `dv` line, `vec3<i32>(0, 1, 0)` → `vec3<i32>(1, 0, 0)` |
| `red_black_sweeps_match_…` (Dirichlet top) | in `pressure.wgsl`, delete the unconditional `count += 1.0;` after the `+z` line |
| `red_black_sweeps_match_…` (Neumann wall) | in `pressure.wgsl`, change `if (c.x > 0) { sum += …; count += 1.0; }` so `count += 1.0;` sits outside the `if` |
| `subtracting_the_gradient_…` | in `gradient.wgsl`, delete the `if (is_wall(axis, i)) { … return; }` block |
| `a_converged_projection_…` | in `gradient.wgsl`, `var upper = 0.0;` → `var upper = textureLoad(phi, p - e, 0).x;` (the top becomes Neumann in the gradient but not in the solve) |

- [ ] **Step 6: Commit and push**

Subject: `Add pressure projection kernels`.

---

### Task 8: The solver node

**Files:**
- Create: `crates/elements-ember/src/solver.rs`
- Modify: `crates/elements-ember/src/lib.rs` (`pub mod solver;` and register `solver::KIND`)
- Test: `crates/elements-ember/tests/solver.rs`

**Interfaces:**
- Consumes: every kernel from Tasks 5–7, `EvalCtx::with_gpu_pool`, `EvalCtx::voxel_size`.
- Produces:
  - `solver::KIND = "ember.smoke_solver"`
  - `solver::SolverParams { substeps: u32, pressure_iterations: u32, buoyancy_density: f32, buoyancy_temperature: f32 }` (`Debug, Clone, Copy, PartialEq, Serialize, Deserialize`); defaults 1, 80, 0.0, 1.0
  - `solver::SolverState { velocity: StaggeredField, density: Field, temperature: Field, pressure: Field }` with `zeroed(gpu, cache, pool, cells) -> Result<Self, GpuError>`, `release_to(self, pool: &mut FieldPool)`, `read_velocity(&self, gpu) -> Result<[Vec<f32>; 3], GpuError>`
  - `solver::Sources<'a> { density: &'a Field, temperature: &'a Field }` (`Clone, Copy`)
  - `solver::Substep` with `new(gpu, constants: &StepConstants) -> Result<Self, GpuError>`, `pre_projection(&mut self, gpu, cache, pool, state: &mut SolverState, sources: Sources<'_>) -> Result<(), GpuError>`, `project(&mut self, gpu, cache, pool, state: &mut SolverState, iterations: u32) -> Result<(), GpuError>`, `advect_scalars(&mut self, gpu, cache, pool, state: &mut SolverState) -> Result<(), GpuError>`, `submit(self, gpu, pool: &mut FieldPool) -> Result<(), GpuError>`, `abandon(self, pool: &mut FieldPool)`
  - `solver::substep(gpu, cache, pool, state, sources, constants: &StepConstants, iterations: u32) -> Result<(), GpuError>`

  Node sockets: inputs `[density_source: Field, temperature_source: Field]`; outputs `[density: Field, temperature: Field, velocity: VectorField]`.

- [ ] **Step 1: Write the failing tests**

`tests/solver.rs`:

```rust
mod common;

use common::*;
use elements_core::gpu::{
    FieldDims, FieldFormat, FieldPool, GpuContext, GpuError, PipelineCache,
};
use elements_core::graph::{
    DocError, Document, EvalCtx, Graph, Node, NodeError, NodeId, SocketId, SocketSpec,
    SocketType, Timeline, TimelineConfig, Value,
};
use elements_ember::kernels::StepConstants;
use elements_ember::solver::{KIND, SolverState, Sources, substep};

fn rejected(params: serde_json::Value) -> bool {
    matches!(
        elements_ember::registry().build(KIND, &params),
        Err(DocError::BadParams { .. })
    )
}

#[test]
fn rejects_out_of_range_solver_parameters() {
    assert!(!rejected(serde_json::json!({})), "defaults must build");
    assert!(!rejected(serde_json::json!({ "substeps": 16, "pressure_iterations": 1000 })));
    assert!(rejected(serde_json::json!({ "substeps": 0 })));
    assert!(rejected(serde_json::json!({ "substeps": 17 })));
    assert!(rejected(serde_json::json!({ "pressure_iterations": 0 })));
    assert!(rejected(serde_json::json!({ "pressure_iterations": 1001 })));
    assert!(rejected(serde_json::json!({ "buoyancy_temperature": 1e39 })));
    assert!(rejected(serde_json::json!({ "buoyancy_density": 1e39 })));
    assert!(rejected(serde_json::json!({ "vorticity": 1.0 })));
}

/// Umbrella §6: no emitters and nothing to be buoyant, so nothing moves.
/// The buoyancy coefficients are nonzero on purpose: the force is zero only
/// because density and temperature are.
#[test]
fn a_still_domain_stays_exactly_still() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(8, 6, 5);
    let zero = pool.acquire_zeroed(&gpu, &mut cache, cells).unwrap();
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
    let constants = StepConstants { cells, h: 1.0 / 24.0, dx: 0.25, alpha: 0.5, beta: 2.0 };
    let sources = Sources { density: &zero, temperature: &zero };
    for _ in 0..10 {
        substep(&gpu, &mut cache, &mut pool, &mut state, sources, &constants, 20).unwrap();
    }
    for (a, face) in state.read_velocity(&gpu).unwrap().iter().enumerate() {
        assert!(face.iter().all(|&v| v == 0.0), "face {a} moved");
    }
}

const PLUME_16: &str = r#"{
  "version": 3,
  "dims": [16, 16, 16],
  "fps": 24.0,
  "domain_size": 2.0,
  "nodes": [
    { "id": 0, "kind": "ember.sphere_emitter",
      "params": { "center": [1.0, 1.0, 0.4], "radius": 0.3,
                  "density_rate": 1.0, "temperature_rate": 2.0 } },
    { "id": 1, "kind": "ember.smoke_solver",
      "params": { "pressure_iterations": 40, "buoyancy_temperature": 1.0 } },
    { "id": 2, "kind": "core.output", "params": {} }
  ],
  "edges": [
    { "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 },
    { "from_node": 0, "from_index": 1, "to_node": 1, "to_index": 1 },
    { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 0 }
  ],
  "output": 2
}"#;

struct Session {
    gpu: GpuContext,
    pool: FieldPool,
    pipelines: PipelineCache,
    graph: Graph,
    dims: FieldDims,
}

impl Session {
    fn new() -> Self {
        let (graph, dims) = Document::from_json(PLUME_16)
            .unwrap()
            .into_graph(&elements_ember::registry())
            .unwrap();
        Self { gpu: gpu(), pool: FieldPool::new(), pipelines: PipelineCache::new(), graph, dims }
    }

    fn density_bits(&mut self, timeline: &mut Timeline, frame: u32) -> Vec<u32> {
        let evaluated = timeline
            .goto(&self.graph, &self.gpu, &mut self.pool, &mut self.pipelines, self.dims, frame)
            .unwrap();
        let bits = evaluated.value.as_field().unwrap().read_back(&self.gpu).unwrap()
            .iter().map(|v| v.to_bits()).collect();
        evaluated.value.release_to(&mut self.pool);
        bits
    }
}

fn timeline(budget_bytes: u64) -> Timeline {
    Timeline::new(TimelineConfig { fps: 24.0, start_frame: 1, cache_budget_bytes: budget_bytes })
}

/// Umbrella §4 and §6: frame 40 is bit-identical in order, after scrubbing
/// back and forth, and after eviction forced a recompute.
#[test]
fn frame_40_is_bit_identical_however_it_is_reached() {
    let mut s = Session::new();

    let mut in_order = timeline(0);
    let mut reference = Vec::new();
    for frame in 1..=40 {
        reference = s.density_bits(&mut in_order, frame);
    }
    assert!(reference.iter().any(|&b| f32::from_bits(b) != 0.0), "the plume must exist");

    let mut scrubbed = timeline(512 * 1024 * 1024);
    s.density_bits(&mut scrubbed, 40);
    s.density_bits(&mut scrubbed, 10);
    assert!(s.density_bits(&mut scrubbed, 40) == reference, "after scrubbing");

    // About ten 16³ snapshots fit, so reaching 40 evicts most of them.
    let mut evicting = timeline(1024 * 1024);
    s.density_bits(&mut evicting, 40);
    s.density_bits(&mut evicting, 5);
    assert!(s.density_bits(&mut evicting, 40) == reference, "after eviction");
}

/// Outputs a zero field one cell larger than the domain: a mis-sized source.
struct WrongSize;

impl Node for WrongSize {
    fn kind(&self) -> &'static str {
        "test.wrong_size"
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec { inputs: vec![], outputs: vec![SocketType::Field] }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let d = ctx.dims();
        let dims = FieldDims::new(d.x + 1, d.y, d.z);
        let field = ctx.with_gpu_pool(|gpu, _, pool| pool.acquire(gpu, dims, FieldFormat::R32Float))?;
        Ok(vec![Value::Field(field)])
    }
}

/// A step that fails part-way returns every texture to the pool: the state
/// it took out of the store and anything it acquired.
#[test]
fn a_failed_step_returns_every_field_to_the_pool() {
    let registry = elements_ember::registry();
    let mut graph = Graph::new();
    let emitter = graph.add_node(
        registry
            .build("ember.sphere_emitter", &serde_json::json!({ "center": [1.0, 1.0, 0.4], "radius": 0.3 }))
            .unwrap(),
    );
    let wrong = graph.add_node(Box::new(WrongSize));
    let solver = graph.add_node(registry.build(KIND, &serde_json::json!({})).unwrap());
    let output = graph.add_node(registry.build("core.output", &serde_json::json!({})).unwrap());
    let socket = |node: NodeId, index: u32| SocketId { node, index };
    graph.connect(socket(emitter, 0), socket(solver, 0)).unwrap();
    graph.connect(socket(wrong, 0), socket(solver, 1)).unwrap();
    graph.connect(socket(solver, 0), socket(output, 0)).unwrap();
    graph.set_output(output);

    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut timeline = timeline(0);
    let err = timeline
        .goto(&graph, &gpu, &mut pool, &mut pipelines, FieldDims::new(8, 8, 8), 1)
        .unwrap_err();
    assert!(matches!(err, NodeError::Gpu(GpuError::Validation(_))), "got {err:?}");
    assert!(pool.allocation_count() > 0);
    assert_eq!(
        pool.pooled_count() as u64,
        pool.allocation_count(),
        "every allocated texture must be back in the pool"
    );
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo nextest run -p elements-ember --test solver`
Expected: compile error, `solver` not found.

- [ ] **Step 3: Implement**

`crates/elements-ember/src/solver.rs`:

```rust
//! `ember.smoke_solver`: a dense-grid smoke solver (spec §2.4, §3).
//!
//! Per substep: emit, buoyancy, advect velocity, project, advect scalars.
//! Vorticity confinement and dissipation arrive in piece 2b.

use elements_core::gpu::{
    Axis, ComputeBatch, Field, FieldDims, FieldFormat, FieldPool, GpuContext, GpuError,
    PipelineCache, StaggeredField,
};
use elements_core::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};
use serde::{Deserialize, Serialize};

use crate::kernels::{self, StepConstants, Uniforms};
use crate::params;

pub const KIND: &str = "ember.smoke_solver";

const VELOCITY: &str = "velocity";
const DENSITY: &str = "density";
const TEMPERATURE: &str = "temperature";
/// Holds φ = h·p, the pressure scaled by the substep, not p itself.
const PRESSURE: &str = "pressure";
const SLOTS: [&str; 4] = [VELOCITY, DENSITY, TEMPERATURE, PRESSURE];

const MAX_SUBSTEPS: u32 = 16;
const MAX_PRESSURE_ITERATIONS: u32 = 1000;

fn default_substeps() -> u32 {
    1
}

/// Provisional until the speed gate records a default (spec §4.3).
fn default_pressure_iterations() -> u32 {
    80
}

/// Provisional until 2b maps parameters to Mantaflow's.
fn default_buoyancy_temperature() -> f32 {
    1.0
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverParams {
    /// Fixed substeps per frame. CFL-driven substepping is 2b.
    #[serde(default = "default_substeps")]
    pub substeps: u32,
    /// Red-black Gauss–Seidel iterations per substep.
    #[serde(default = "default_pressure_iterations")]
    pub pressure_iterations: u32,
    /// α: downward acceleration per unit density, m/s².
    #[serde(default)]
    pub buoyancy_density: f32,
    /// β: upward acceleration per unit temperature, m/s².
    #[serde(default = "default_buoyancy_temperature")]
    pub buoyancy_temperature: f32,
}

/// Everything the solver carries from one step to the next.
pub struct SolverState {
    pub velocity: StaggeredField,
    pub density: Field,
    pub temperature: Field,
    /// φ = h·p, kept as the next solve's warm start.
    pub pressure: Field,
}

impl SolverState {
    /// A still, empty domain.
    pub fn zeroed(
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        cells: FieldDims,
    ) -> Result<Self, GpuError> {
        let velocity = pool.acquire_staggered_zeroed(gpu, cache, cells)?;
        let mut fields = Vec::with_capacity(3);
        for _ in 0..3 {
            match pool.acquire_zeroed(gpu, cache, cells) {
                Ok(field) => fields.push(field),
                Err(e) => {
                    pool.release_staggered(velocity);
                    for field in fields {
                        pool.release(field);
                    }
                    return Err(e);
                }
            }
        }
        let [density, temperature, pressure]: [Field; 3] = match fields.try_into() {
            Ok(array) => array,
            Err(_) => unreachable!("exactly three fields were acquired"),
        };
        Ok(Self { velocity, density, temperature, pressure })
    }

    pub fn release_to(self, pool: &mut FieldPool) {
        pool.release_staggered(self.velocity);
        pool.release(self.density);
        pool.release(self.temperature);
        pool.release(self.pressure);
    }

    /// The X, Y and Z faces, x-fastest, for tests and the speed gate.
    pub fn read_velocity(&self, gpu: &GpuContext) -> Result<[Vec<f32>; 3], GpuError> {
        Ok([
            self.velocity.face(Axis::X).read_back(gpu)?,
            self.velocity.face(Axis::Y).read_back(gpu)?,
            self.velocity.face(Axis::Z).read_back(gpu)?,
        ])
    }
}

/// Emission rates per second, at the domain's dims.
#[derive(Clone, Copy)]
pub struct Sources<'a> {
    pub density: &'a Field,
    pub temperature: &'a Field,
}

/// One substep being recorded. Every stage records into one batch, so a
/// substep is one queue submission (spec §3).
///
/// A field a stage replaces cannot go back to the pool until the batch has
/// run, or a later stage could be handed the same texture while an earlier
/// dispatch still reads it. Replaced fields wait in `retired` until `submit`
/// or `abandon`.
pub struct Substep {
    uniforms: Uniforms,
    batch: ComputeBatch,
    retired: Vec<Field>,
}

impl Substep {
    pub fn new(gpu: &GpuContext, constants: &StepConstants) -> Result<Self, GpuError> {
        Ok(Self {
            uniforms: Uniforms::new(gpu, constants)?,
            batch: ComputeBatch::new(),
            retired: Vec::new(),
        })
    }

    /// Stages 1–3: emit, buoyancy, advect velocity.
    pub fn pre_projection(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        state: &mut SolverState,
        sources: Sources<'_>,
    ) -> Result<(), GpuError> {
        let u = &self.uniforms;
        kernels::emit(gpu, cache, &mut self.batch, u, &state.density, sources.density)?;
        kernels::emit(gpu, cache, &mut self.batch, u, &state.temperature, sources.temperature)?;
        kernels::buoyancy(
            gpu,
            cache,
            &mut self.batch,
            u,
            state.velocity.face(Axis::Z),
            &state.density,
            &state.temperature,
        )?;
        let advected = pool.acquire_staggered_uninit(gpu, u.cells())?;
        if let Err(e) = kernels::advect_velocity(gpu, cache, &mut self.batch, u, &state.velocity, &advected) {
            self.retired.extend(advected.into_faces());
            return Err(e);
        }
        let old = std::mem::replace(&mut state.velocity, advected);
        self.retired.extend(old.into_faces());
        Ok(())
    }

    /// Stage 4: make the velocity divergence-free, warm-starting from `state.pressure`.
    pub fn project(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        state: &mut SolverState,
        iterations: u32,
    ) -> Result<(), GpuError> {
        let u = &self.uniforms;
        let div = pool.acquire(gpu, u.cells(), FieldFormat::R32Float)?;
        let recorded = kernels::divergence(gpu, cache, &mut self.batch, u, &state.velocity, &div)
            .and_then(|()| {
                kernels::pressure(gpu, cache, &mut self.batch, u, &state.pressure, &div, iterations)
            })
            .and_then(|()| {
                kernels::subtract_gradient(gpu, cache, &mut self.batch, u, &state.velocity, &state.pressure)
            });
        // The batch may reference `div` whether or not recording finished.
        self.retired.push(div);
        recorded
    }

    /// Stage 5: carry density and temperature through the projected velocity.
    pub fn advect_scalars(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        state: &mut SolverState,
    ) -> Result<(), GpuError> {
        self.advect_one(gpu, cache, pool, &state.velocity, &mut state.density)?;
        self.advect_one(gpu, cache, pool, &state.velocity, &mut state.temperature)
    }

    fn advect_one(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        velocity: &StaggeredField,
        field: &mut Field,
    ) -> Result<(), GpuError> {
        let dst = pool.acquire(gpu, field.dims(), FieldFormat::R32Float)?;
        if let Err(e) = kernels::advect_scalar(gpu, cache, &mut self.batch, &self.uniforms, velocity, field, &dst) {
            self.retired.push(dst);
            return Err(e);
        }
        self.retired.push(std::mem::replace(field, dst));
        Ok(())
    }

    /// Run everything recorded, then return replaced fields to the pool.
    pub fn submit(self, gpu: &GpuContext, pool: &mut FieldPool) -> Result<(), GpuError> {
        let result = self.batch.submit(gpu);
        for field in self.retired {
            pool.release(field);
        }
        result
    }

    /// Discard everything recorded without running it.
    ///
    /// The state passed to the stages may now hold fields that were never
    /// written, so the caller must discard the state too.
    pub fn abandon(self, pool: &mut FieldPool) {
        for field in self.retired {
            pool.release(field);
        }
    }
}

/// One whole substep, stages 1–5, as one submission.
pub fn substep(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    pool: &mut FieldPool,
    state: &mut SolverState,
    sources: Sources<'_>,
    constants: &StepConstants,
    iterations: u32,
) -> Result<(), GpuError> {
    let mut step = Substep::new(gpu, constants)?;
    let recorded = step
        .pre_projection(gpu, cache, pool, state, sources)
        .and_then(|()| step.project(gpu, cache, pool, state, iterations))
        .and_then(|()| step.advect_scalars(gpu, cache, pool, state));
    match recorded {
        Ok(()) => step.submit(gpu, pool),
        Err(e) => {
            step.abandon(pool);
            Err(e)
        }
    }
}

#[derive(Debug, Clone)]
pub struct SmokeSolver {
    params: SolverParams,
}

impl SmokeSolver {
    /// Take the four state slots, or a zeroed state on the first step.
    fn take_state(ctx: &mut EvalCtx<'_>) -> Result<SolverState, NodeError> {
        let mut taken: Vec<(&'static str, Value)> = Vec::new();
        for slot in SLOTS {
            match ctx.take_state(slot) {
                Ok(Some(value)) => taken.push((slot, value)),
                Ok(None) => {}
                Err(e) => {
                    for (_, value) in taken {
                        ctx.release(value);
                    }
                    return Err(e);
                }
            }
        }
        if taken.is_empty() {
            let cells = ctx.dims();
            return ctx.with_gpu_pool(|gpu, cache, pool| SolverState::zeroed(gpu, cache, pool, cells));
        }

        let node = ctx.node_id();
        let missing = SLOTS
            .into_iter()
            .find(|slot| !taken.iter().any(|(name, _)| name == slot));
        let mut values = taken.into_iter().map(|(_, value)| value);
        match (values.next(), values.next(), values.next(), values.next()) {
            (
                Some(Value::VectorField(velocity)),
                Some(Value::Field(density)),
                Some(Value::Field(temperature)),
                Some(Value::Field(pressure)),
            ) if missing.is_none() => Ok(SolverState { velocity, density, temperature, pressure }),
            (a, b, c, d) => {
                for value in [a, b, c, d].into_iter().flatten() {
                    ctx.release(value);
                }
                Err(NodeError::StateShape { node, slot: missing.unwrap_or(VELOCITY) })
            }
        }
    }

    fn put_state(ctx: &mut EvalCtx<'_>, state: SolverState) -> Result<(), NodeError> {
        ctx.put_state(VELOCITY, Value::VectorField(state.velocity))?;
        ctx.put_state(DENSITY, Value::Field(state.density))?;
        ctx.put_state(TEMPERATURE, Value::Field(state.temperature))?;
        ctx.put_state(PRESSURE, Value::Field(state.pressure))
    }

    fn run(&self, ctx: &mut EvalCtx<'_>, state: &mut SolverState) -> Result<Vec<Value>, NodeError> {
        let density_source = ctx.take_input(0)?;
        let temperature_source = match ctx.take_input(1) {
            Ok(value) => value,
            Err(e) => {
                ctx.release(density_source);
                return Err(e);
            }
        };
        let stepped = self.step(ctx, state, &density_source, &temperature_source);
        ctx.release(density_source);
        ctx.release(temperature_source);
        stepped?;

        // The outputs are copies: the state stays in the store for the next frame.
        ctx.with_gpu_pool(|gpu, _, pool| {
            let density = pool.duplicate(gpu, &state.density)?;
            let temperature = match pool.duplicate(gpu, &state.temperature) {
                Ok(field) => field,
                Err(e) => {
                    pool.release(density);
                    return Err(e);
                }
            };
            let velocity = match duplicate_velocity(gpu, pool, &state.velocity) {
                Ok(velocity) => velocity,
                Err(e) => {
                    pool.release(density);
                    pool.release(temperature);
                    return Err(e);
                }
            };
            Ok(vec![Value::Field(density), Value::Field(temperature), Value::VectorField(velocity)])
        })
    }

    fn step(
        &self,
        ctx: &mut EvalCtx<'_>,
        state: &mut SolverState,
        density_source: &Value,
        temperature_source: &Value,
    ) -> Result<(), NodeError> {
        let node = ctx.node_id();
        let field = |value: &Value, index: u32| {
            value.as_field().map_err(|_| NodeError::TypeMismatch {
                node,
                index,
                expected: SocketType::Field,
            })
        };
        let sources = Sources {
            density: field(density_source, 0)?,
            temperature: field(temperature_source, 1)?,
        };
        let SolverParams { substeps, pressure_iterations, buoyancy_density, buoyancy_temperature } =
            self.params;
        let constants = StepConstants {
            cells: ctx.dims(),
            h: (ctx.time().dt / substeps as f64) as f32,
            dx: ctx.voxel_size(),
            alpha: buoyancy_density,
            beta: buoyancy_temperature,
        };
        ctx.with_gpu_pool(|gpu, cache, pool| {
            for _ in 0..substeps {
                substep(gpu, cache, pool, state, sources, &constants, pressure_iterations)?;
            }
            Ok(())
        })
    }
}

impl Node for SmokeSolver {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field, SocketType::Field],
            outputs: vec![SocketType::Field, SocketType::Field, SocketType::VectorField],
        }
    }

    fn stateful(&self) -> bool {
        true
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let mut state = Self::take_state(ctx)?;
        match self.run(ctx, &mut state) {
            Ok(outputs) => {
                Self::put_state(ctx, state)?;
                Ok(outputs)
            }
            Err(e) => {
                // A half-run step may hold fields that were never written.
                // Never write it back; the timeline starts over from its cache.
                ctx.with_gpu_pool(|_, _, pool| {
                    state.release_to(pool);
                    Ok(())
                })?;
                Err(e)
            }
        }
    }
}

pub(crate) fn build(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    // `deny_unknown_fields` rejects misspelt keys; an absent `params` is `null`.
    let params: SolverParams = if params.is_null() {
        params::parse(KIND, &serde_json::json!({}))?
    } else {
        params::parse(KIND, params)?
    };
    if !(1..=MAX_SUBSTEPS).contains(&params.substeps) {
        return Err(params::bad(KIND, format!("substeps must be 1..={MAX_SUBSTEPS}, got {}", params.substeps)));
    }
    if !(1..=MAX_PRESSURE_ITERATIONS).contains(&params.pressure_iterations) {
        return Err(params::bad(
            KIND,
            format!(
                "pressure_iterations must be 1..={MAX_PRESSURE_ITERATIONS}, got {}",
                params.pressure_iterations
            ),
        ));
    }
    params::finite(KIND, "buoyancy", &[params.buoyancy_density, params.buoyancy_temperature])?;
    Ok(Box::new(SmokeSolver { params }))
}
```

Also in `solver.rs`, the helper `run` uses for the velocity output:

```rust
/// A pooled copy of all three faces. On failure, faces already copied go back to the pool.
fn duplicate_velocity(
    gpu: &GpuContext,
    pool: &mut FieldPool,
    velocity: &StaggeredField,
) -> Result<StaggeredField, GpuError> {
    let mut faces = Vec::with_capacity(3);
    for axis in Axis::ALL {
        match pool.duplicate(gpu, velocity.face(axis)) {
            Ok(face) => faces.push(face),
            Err(e) => {
                for face in faces {
                    pool.release(face);
                }
                return Err(e);
            }
        }
    }
    let [x, y, z]: [Field; 3] = match faces.try_into() {
        Ok(array) => array,
        Err(_) => unreachable!("exactly three faces were copied"),
    };
    StaggeredField::from_faces(velocity.cells(), [x, y, z])
}
```

In `lib.rs`, add `pub mod kernels;` (if not already), `pub mod solver;`, and `registry.register(solver::KIND, solver::build);` in `register`.

- [ ] **Step 4: Run the tests, then `just check`**

Run: `cargo nextest run -p elements-ember --test solver`
Expected: PASS.

- [ ] **Step 5: Prove each test can fail**

| Test | Mutation |
|---|---|
| `rejects_out_of_range_solver_parameters` | `MAX_SUBSTEPS` 16 → 17 |
| `a_still_domain_stays_exactly_still` | in `buoyancy.wgsl`, `(params.beta * temp - params.alpha * rho)` → `(params.beta * temp - params.alpha * rho + 1.0)` |
| `frame_40_is_bit_identical_…` | in core's `Value::duplicate` (`crates/elements-core/src/graph/node.rs`), the Y-face line `pool.duplicate(gpu, v.face(Axis::Y))?` → `pool.acquire(gpu, v.face(Axis::Y).dims(), v.face(Axis::Y).format())?` (a snapshot that skips copying one face) |
| `a_failed_step_returns_every_field_to_the_pool` | in `SmokeSolver::eval`'s `Err` arm, replace the `with_gpu_pool(… state.release_to(pool) …)?;` statement with `drop(state);` |

The determinism test also closes piece 1's open finding that snapshotting a `VectorField` was untested.

- [ ] **Step 6: Commit and push**

Subject: `Add the smoke solver node`.

---

### Task 9: Metrics, and the divergence-free and buoyant-blob scenes

**Files:**
- Create: `crates/elements-ember/src/metrics.rs`
- Modify: `crates/elements-ember/src/lib.rs` (`pub mod metrics;`)
- Test: `crates/elements-ember/tests/scenes.rs`

**Interfaces:**
- Produces: `metrics::DivergenceStats { max_abs: f64, rms: f64 }` (`Debug, Clone, Copy, PartialEq`); `metrics::divergence(faces: &[Vec<f32>; 3], cells: FieldDims, dx: f32) -> DivergenceStats`; `metrics::centroid_z(density: &[f32], cells: FieldDims) -> Option<f64>` (cell units; `None` when the total density is 0).

- [ ] **Step 1: Write the failing tests**

`tests/scenes.rs`:

```rust
mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_ember::emitter::{Sphere, fill_sphere};
use elements_ember::kernels::StepConstants;
use elements_ember::metrics::{centroid_z, divergence};
use elements_ember::solver::{SolverState, Sources, Substep, substep};

#[test]
fn divergence_statistics_are_max_and_root_mean_square() {
    // Two cells along x with divergences 3 and 4.
    let cells = FieldDims::new(2, 1, 1);
    let faces = [vec![0.0, 3.0, 7.0], vec![0.0; 4], vec![0.0; 4]];
    let stats = divergence(&faces, cells, 1.0);
    assert_eq!(stats.max_abs, 4.0);
    assert!((stats.rms - 12.5_f64.sqrt()).abs() < 1e-12, "rms {}", stats.rms);
}

#[test]
fn the_centroid_is_in_cell_units_at_cell_centres() {
    let cells = FieldDims::new(2, 2, 4);
    let mut density = vec![0.0; cells.voxel_count()];
    density[index(cells, 1, 0, 3)] = 2.0;
    density[index(cells, 0, 1, 1)] = 2.0;
    assert_eq!(centroid_z(&density, cells), Some(2.5));
    assert_eq!(centroid_z(&vec![0.0; cells.voxel_count()], cells), None);
}

/// Umbrella §6, spec §4.2: projection cuts RMS divergence to at most 10% of
/// its value before projection, at the provisional iteration count.
#[test]
fn projection_leaves_at_most_a_tenth_of_the_divergence() {
    const ITERATIONS: u32 = 80;
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(16, 16, 16);
    let dx = 2.0 / 16.0;
    let density_source = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let temperature_source = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let sphere = Sphere { center: [1.0, 1.0, 0.4], radius: 0.3, density_rate: 1.0, temperature_rate: 2.0 };
    fill_sphere(&gpu, &mut cache, &density_source, &temperature_source, &sphere, dx).unwrap();
    let sources = Sources { density: &density_source, temperature: &temperature_source };
    let constants = StepConstants { cells, h: 1.0 / 24.0, dx, alpha: 0.0, beta: 1.0 };
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
    for _ in 0..20 {
        substep(&gpu, &mut cache, &mut pool, &mut state, sources, &constants, ITERATIONS).unwrap();
    }

    let mut step = Substep::new(&gpu, &constants).unwrap();
    step.pre_projection(&gpu, &mut cache, &mut pool, &mut state, sources).unwrap();
    step.submit(&gpu, &mut pool).unwrap();
    let before = divergence(&state.read_velocity(&gpu).unwrap(), cells, dx);

    let mut step = Substep::new(&gpu, &constants).unwrap();
    step.project(&gpu, &mut cache, &mut pool, &mut state, ITERATIONS).unwrap();
    step.submit(&gpu, &mut pool).unwrap();
    let after = divergence(&state.read_velocity(&gpu).unwrap(), cells, dx);

    assert!(before.rms > 0.0, "the plume must be moving");
    assert!(
        after.rms <= 0.1 * before.rms,
        "RMS divergence {} -> {} (ratio {})",
        before.rms,
        after.rms,
        after.rms / before.rms
    );
}

/// Umbrella §6: a hot blob rises, and its density centroid climbs strictly
/// every frame.
#[test]
fn a_hot_blob_rises_every_frame() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(16, 16, 24);
    let dx = 2.0 / 24.0;
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
    // The blob itself: density and temperature 1 inside a sphere, no emission.
    let blob = Sphere { center: [0.667, 0.667, 0.4], radius: 0.2, density_rate: 1.0, temperature_rate: 1.0 };
    fill_sphere(&gpu, &mut cache, &state.density, &state.temperature, &blob, dx).unwrap();
    let zero = pool.acquire_zeroed(&gpu, &mut cache, cells).unwrap();
    let sources = Sources { density: &zero, temperature: &zero };
    let constants = StepConstants { cells, h: 1.0 / 24.0, dx, alpha: 0.0, beta: 1.0 };

    let mut heights = vec![centroid_z(&state.density.read_back(&gpu).unwrap(), cells).unwrap()];
    for _ in 0..20 {
        substep(&gpu, &mut cache, &mut pool, &mut state, sources, &constants, 80).unwrap();
        heights.push(centroid_z(&state.density.read_back(&gpu).unwrap(), cells).unwrap());
    }
    for (frame, pair) in heights.windows(2).enumerate() {
        assert!(pair[1] > pair[0], "frame {}: centroid {} -> {}; all: {heights:?}", frame + 1, pair[0], pair[1]);
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo nextest run -p elements-ember --test scenes`
Expected: compile error, `metrics` not found.

- [ ] **Step 3: Implement**

`crates/elements-ember/src/metrics.rs`:

```rust
//! Physical-quality metrics computed on the CPU from read-back fields.
//!
//! The same functions serve the validation scenes, the speed gate, and 2b's
//! Mantaflow benchmark (spec §5.4), so both solvers are measured by one piece
//! of code. Sums are in `f64`: a 128³ grid has two million cells.

use elements_core::gpu::{FieldDims, StaggeredField};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DivergenceStats {
    /// Largest |div u| over all cells, 1/s.
    pub max_abs: f64,
    /// Root mean square of div u over all cells, 1/s.
    pub rms: f64,
}

fn at(dims: FieldDims, i: u32, j: u32, k: u32) -> usize {
    (i + dims.x * (j + dims.y * k)) as usize
}

/// Divergence of a staggered velocity, from its X, Y and Z faces (x-fastest,
/// as `Field::read_back` returns them), with voxel edge `dx` metres.
pub fn divergence(faces: &[Vec<f32>; 3], cells: FieldDims, dx: f32) -> DivergenceStats {
    use elements_core::gpu::Axis;
    let xd = StaggeredField::face_dims(cells, Axis::X);
    let yd = StaggeredField::face_dims(cells, Axis::Y);
    let zd = StaggeredField::face_dims(cells, Axis::Z);
    let dx = dx as f64;
    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let d = (faces[0][at(xd, i + 1, j, k)] as f64 - faces[0][at(xd, i, j, k)] as f64
                    + faces[1][at(yd, i, j + 1, k)] as f64
                    - faces[1][at(yd, i, j, k)] as f64
                    + faces[2][at(zd, i, j, k + 1)] as f64
                    - faces[2][at(zd, i, j, k)] as f64)
                    / dx;
                max_abs = max_abs.max(d.abs());
                sum_sq += d * d;
            }
        }
    }
    DivergenceStats {
        max_abs,
        rms: (sum_sq / cells.voxel_count() as f64).sqrt(),
    }
}

/// The density-weighted mean height, in cell units (cell k's centre is at
/// k + 0.5). `None` when there is no density.
pub fn centroid_z(density: &[f32], cells: FieldDims) -> Option<f64> {
    let mut mass = 0.0f64;
    let mut moment = 0.0f64;
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let m = density[at(cells, i, j, k)] as f64;
                mass += m;
                moment += m * (k as f64 + 0.5);
            }
        }
    }
    (mass > 0.0).then(|| moment / mass)
}
```

- [ ] **Step 4: Run the tests, then `just check`**

Run: `cargo nextest run -p elements-ember --test scenes`
Expected: PASS. If `projection_leaves_at_most_a_tenth_…` fails on a correct implementation, **stop and report the measured ratio**; this is pre-registered and must not be tuned. Likewise if the blob's centroid fails to rise on some frame: report the `heights` vector.

- [ ] **Step 5: Prove each test can fail**

| Test | Mutation |
|---|---|
| `divergence_statistics_…` | in `metrics::divergence`, `.sqrt()` removed from `rms` |
| `the_centroid_is_in_cell_units_…` | `(k as f64 + 0.5)` → `k as f64` |
| `projection_leaves_at_most_a_tenth_…` | in `kernels::pressure`, delete `batch.dispatch(&black, &black_group, u.cells());` |
| `a_hot_blob_rises_every_frame` | in `buoyancy.wgsl`, `params.beta * temp - params.alpha * rho` → `params.alpha * rho - params.beta * temp` |

- [ ] **Step 6: Commit and push**

Subject: `Add solver metrics and the divergence and buoyancy scenes`.

---

### Task 10: The speed-gate runner

**Files:**
- Create: `crates/elements-ember/src/bench.rs`
- Create: `crates/elements-ember/examples/speed_gate.rs`
- Modify: `crates/elements-ember/src/lib.rs` (`pub mod bench;`)
- Modify: `justfile`
- Test: `crates/elements-ember/tests/bench.rs`

**Interfaces:**
- Consumes: `Sphere`, `SolverParams`, `metrics::divergence`, `SolverState`, `Substep`, `substep`, `fill_sphere`.
- Produces: `bench::Scene { name: &'static str, cells: [u32; 3], domain_size: f64, fps: f64, frames: u32, emitter: Sphere, solver: SolverParams }` with `plume(resolution: u32) -> Scene`, `with_iterations(self, n: u32) -> Scene`, `document(&self) -> Document`; `bench::GateRow { iterations: u32, step_ms_median: f64, ratio: f64 }`; `bench::gate_verdict(rows: &[GateRow]) -> Option<u32>`; constants `GATE_STEP_MS = 100.0`, `GATE_RATIO = 0.10`.

- [ ] **Step 1: Write the failing tests**

`tests/bench.rs`:

```rust
mod common;

use elements_core::gpu::{FieldPool, PipelineCache};
use elements_core::graph::{Document, Timeline};
use elements_ember::bench::{GateRow, Scene, gate_verdict};

fn row(iterations: u32, step_ms_median: f64, ratio: f64) -> GateRow {
    GateRow { iterations, step_ms_median, ratio }
}

/// Spec §4.3's pre-registered rule, encoded once and tested here.
#[test]
fn the_gate_passes_with_the_largest_n_meeting_both_limits() {
    let rows = [row(20, 30.0, 0.30), row(40, 50.0, 0.09), row(80, 90.0, 0.05), row(160, 170.0, 0.02)];
    assert_eq!(gate_verdict(&rows), Some(80));
}

#[test]
fn the_gate_limits_are_inclusive() {
    assert_eq!(gate_verdict(&[row(40, 100.0, 0.10)]), Some(40));
}

#[test]
fn the_gate_fails_when_no_n_meets_both_limits() {
    // Fast enough but too divergent, or accurate enough but too slow.
    assert_eq!(gate_verdict(&[row(20, 40.0, 0.2), row(160, 120.0, 0.05)]), None);
}

#[test]
fn the_plume_scene_loads_and_steps() {
    let text = Scene::plume(16).document().to_json().unwrap();
    let doc = Document::from_json(&text).unwrap();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(&elements_ember::registry()).unwrap();
    let gpu = common::gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut timeline = Timeline::new(config);
    let frame = timeline.goto(&graph, &gpu, &mut pool, &mut pipelines, dims, 2).unwrap();
    let density = frame.value.as_field().unwrap().read_back(&gpu).unwrap();
    assert!(density.iter().any(|&v| v > 0.0), "the plume must emit");
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo nextest run -p elements-ember --test bench`
Expected: compile error, `bench` not found.

- [ ] **Step 3: Implement `bench.rs`**

```rust
//! Benchmark scenes and the speed gate's pre-registered rule (spec §4.3, §5.3).

use elements_core::graph::{DocEdge, DocNode, Document, ELEMENTS_DOC_VERSION};

use crate::emitter::{self, Sphere};
use crate::solver::{self, SolverParams};

/// A step must take at most this long at 128³ (≥ 10 fps).
pub const GATE_STEP_MS: f64 = 100.0;
/// Projection must leave at most this fraction of the RMS divergence.
pub const GATE_RATIO: f64 = 0.10;

/// A benchmark scene, defined once (spec §5.3). Piece 2a generates only the
/// `.elements` document from it; 2b adds the matching Mantaflow script.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scene {
    pub name: &'static str,
    pub cells: [u32; 3],
    /// Metres along the longest axis.
    pub domain_size: f64,
    pub fps: f64,
    pub frames: u32,
    pub emitter: Sphere,
    pub solver: SolverParams,
}

impl Scene {
    /// A hot, dense sphere at the domain floor, buoyancy only.
    pub fn plume(resolution: u32) -> Self {
        Self {
            name: "plume",
            cells: [resolution; 3],
            domain_size: 2.0,
            fps: 24.0,
            frames: 120,
            emitter: Sphere {
                center: [1.0, 1.0, 0.3],
                radius: 0.2,
                density_rate: 1.0,
                temperature_rate: 1.0,
            },
            solver: SolverParams {
                substeps: 1,
                pressure_iterations: 80,
                buoyancy_density: 0.0,
                buoyancy_temperature: 1.0,
            },
        }
    }

    pub fn with_iterations(mut self, n: u32) -> Self {
        self.solver.pressure_iterations = n;
        self
    }

    /// The scene as an `.elements` document: emitter → solver → output.
    pub fn document(&self) -> Document {
        let to_value = |v: serde_json::Result<serde_json::Value>| {
            v.expect("scene parameters are plain numbers and always serialize")
        };
        let edge = |from_node, from_index, to_node, to_index| DocEdge {
            from_node,
            from_index,
            to_node,
            to_index,
        };
        Document {
            version: ELEMENTS_DOC_VERSION,
            dims: self.cells,
            fps: self.fps,
            start_frame: 1,
            cache_budget_mb: 0,
            domain_size: self.domain_size,
            nodes: vec![
                DocNode {
                    id: 0,
                    kind: emitter::KIND.to_owned(),
                    params: to_value(serde_json::to_value(self.emitter)),
                },
                DocNode {
                    id: 1,
                    kind: solver::KIND.to_owned(),
                    params: to_value(serde_json::to_value(self.solver)),
                },
                DocNode {
                    id: 2,
                    kind: "core.output".to_owned(),
                    params: serde_json::json!({}),
                },
            ],
            edges: vec![edge(0, 0, 1, 0), edge(0, 1, 1, 1), edge(1, 0, 2, 0)],
            output: 2,
        }
    }
}

/// One row of the gate's table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateRow {
    pub iterations: u32,
    /// Median over three runs of each run's median step time.
    pub step_ms_median: f64,
    /// RMS divergence after projection over RMS divergence before it.
    pub ratio: f64,
}

/// Spec §4.3: PASS with the largest N whose step takes at most
/// `GATE_STEP_MS` and whose ratio is at most `GATE_RATIO`; `None` is FAIL.
pub fn gate_verdict(rows: &[GateRow]) -> Option<u32> {
    rows.iter()
        .filter(|r| r.step_ms_median <= GATE_STEP_MS && r.ratio <= GATE_RATIO)
        .map(|r| r.iterations)
        .max()
}
```

`cache_budget_mb: 0` turns the timeline cache off, which the gate's timing requires (spec §4.3), and prints a warning through `Timeline::take_warning` in the CLI and daemon; that is expected for a benchmark document.

- [ ] **Step 4: Write the example**

`crates/elements-ember/examples/speed_gate.rs`:

```rust
//! The piece 2a speed gate (spec §4.3). Run with `just bench-gate`.
//!
//! Writes `docs/bench/speed-gate.md`. The decision line at its end is filled
//! in by hand after the user decides; this program only applies the rule.

use std::error::Error;
use std::process::Command;
use std::time::Instant;

use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{NodeRegistry, StateStore, Time};
use elements_ember::bench::{GATE_RATIO, GATE_STEP_MS, GateRow, Scene, gate_verdict};
use elements_ember::emitter::fill_sphere;
use elements_ember::kernels::StepConstants;
use elements_ember::metrics::{DivergenceStats, divergence};
use elements_ember::solver::{SolverState, Sources, Substep, substep};

const RESOLUTION: u32 = 128;
const ITERATIONS: [u32; 4] = [20, 40, 80, 160];
const RUNS: usize = 3;
const WARMUP: u32 = 24;
const TIMED: u32 = 24;

type Res<T> = Result<T, Box<dyn Error>>;

struct Timing {
    /// Each run's median step, ms.
    run_medians: Vec<f64>,
    /// Every timed step of every run, ms.
    all: Vec<f64>,
    /// Every snapshot, ms.
    snapshots: Vec<f64>,
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n % 2 == 1 { sorted[n / 2] } else { 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]) }
}

fn wait(gpu: &GpuContext) -> Res<()> {
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Step the scene's graph through `eval_frame`, the path the timeline uses,
/// timing each timed frame as `eval` plus a blocking poll.
fn time_scene(gpu: &GpuContext, registry: &NodeRegistry, scene: &Scene, timing: &mut Timing) -> Res<()> {
    let doc = scene.document();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(registry)?;
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let mut steps = Vec::new();
    let first = config.start_frame;
    for frame in first..first + WARMUP + TIMED {
        let time = Time::at(frame, first, config.fps);
        let start = Instant::now();
        let evaluated = graph.eval_frame(gpu, &mut pool, &mut pipelines, &mut state, time, dims)?;
        wait(gpu)?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        evaluated.value.release_to(&mut pool);
        if frame >= first + WARMUP {
            steps.push(ms);
            let start = Instant::now();
            let snapshot = state.snapshot(gpu, &mut pool)?;
            wait(gpu)?;
            timing.snapshots.push(start.elapsed().as_secs_f64() * 1e3);
            snapshot.release_to(&mut pool);
        }
    }
    timing.run_medians.push(median(&steps));
    timing.all.extend(steps);
    state.clear(&mut pool);
    Ok(())
}

/// Reach the frame-48 state through the kernel API, then measure divergence
/// before and after one projection with `n` iterations.
fn divergence_at(gpu: &GpuContext, scene: &Scene, frames: u32) -> Res<(DivergenceStats, DivergenceStats)> {
    let [x, y, z] = scene.cells;
    let cells = FieldDims::new(x, y, z);
    let dx = (scene.domain_size / x.max(y).max(z) as f64) as f32;
    let n = scene.solver.pressure_iterations;
    let substeps = scene.solver.substeps;
    let constants = StepConstants {
        cells,
        h: (1.0 / scene.fps / substeps as f64) as f32,
        dx,
        alpha: scene.solver.buoyancy_density,
        beta: scene.solver.buoyancy_temperature,
    };
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let density_source = pool.acquire(gpu, cells, FieldFormat::R32Float)?;
    let temperature_source = pool.acquire(gpu, cells, FieldFormat::R32Float)?;
    fill_sphere(gpu, &mut cache, &density_source, &temperature_source, &scene.emitter, dx)?;
    let sources = Sources { density: &density_source, temperature: &temperature_source };
    let mut state = SolverState::zeroed(gpu, &mut cache, &mut pool, cells)?;
    for _ in 0..frames * substeps {
        substep(gpu, &mut cache, &mut pool, &mut state, sources, &constants, n)?;
    }

    let mut step = Substep::new(gpu, &constants)?;
    step.pre_projection(gpu, &mut cache, &mut pool, &mut state, sources)?;
    step.submit(gpu, &mut pool)?;
    let before = divergence(&state.read_velocity(gpu)?, cells, dx);

    let mut step = Substep::new(gpu, &constants)?;
    step.project(gpu, &mut cache, &mut pool, &mut state, n)?;
    step.submit(gpu, &mut pool)?;
    let after = divergence(&state.read_velocity(gpu)?, cells, dx);
    Ok((before, after))
}

fn shell(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn main() -> Res<()> {
    let gpu = GpuContext::new_headless()?;
    let registry = elements_ember::registry();
    let mut table = String::new();
    let mut rows = Vec::new();

    for n in ITERATIONS {
        let scene = Scene::plume(RESOLUTION).with_iterations(n);
        let mut timing = Timing { run_medians: Vec::new(), all: Vec::new(), snapshots: Vec::new() };
        for run in 0..RUNS {
            eprintln!("N = {n}: run {} of {RUNS}", run + 1);
            time_scene(&gpu, &registry, &scene, &mut timing)?;
        }
        eprintln!("N = {n}: divergence");
        let (before, after) = divergence_at(&gpu, &scene, WARMUP + TIMED)?;
        let step = median(&timing.run_medians);
        let ratio = after.rms / before.rms;
        let min = timing.all.iter().copied().fold(f64::INFINITY, f64::min);
        let max = timing.all.iter().copied().fold(0.0, f64::max);
        table.push_str(&format!(
            "| {n} | {step:.1} ({min:.1}–{max:.1}) | {:.2} | {:.3e} | {:.3e} | {ratio:.3} | {:.3e} |\n",
            median(&timing.snapshots),
            before.rms,
            after.rms,
            after.max_abs,
        ));
        rows.push(GateRow { iterations: n, step_ms_median: step, ratio });
    }

    let verdict = match gate_verdict(&rows) {
        Some(n) => format!("**PASS**, provisional `pressure_iterations` = {n}"),
        None => "**FAIL**: no N meets both limits".to_owned(),
    };
    let mut commit = shell("git", &["rev-parse", "--short", "HEAD"]);
    if !shell("git", &["status", "--porcelain"]).is_empty() {
        commit.push_str("-dirty");
    }
    let report = format!(
        "# Ember speed gate (piece 2a)\n\n\
         - Machine: {cpu} ({adapter})\n\
         - OS: macOS {os}\n\
         - Ember commit: {commit}\n\
         - Date: {date}\n\
         - Scene: `plume`, {RESOLUTION}³, substeps 1. Frames {first}–{last} timed after {WARMUP} \
         warm-up frames, as `eval` plus a blocking poll; median of {RUNS} runs' medians. \
         Divergence from the frame-{last} state.\n\n\
         | N | step ms (median, min–max) | snapshot ms | RMS div before | RMS div after | ratio | max div after |\n\
         |---|---|---|---|---|---|---|\n\
         {table}\n\
         Pre-registered rule (spec §4.3): PASS if some N has a median step of at most \
         {GATE_STEP_MS} ms and a ratio of at most {GATE_RATIO}; the provisional default is the \
         largest passing N.\n\n\
         Rule applied: {verdict}.\n\n\
         Note: the ratio measures residual divergence, which weights high frequencies. The smooth \
         pressure error Gauss–Seidel leaves behind shows in plume shape, not in this number.\n\n\
         Decision (recorded by the user): _pending_\n",
        cpu = shell("sysctl", &["-n", "machdep.cpu.brand_string"]),
        adapter = gpu.adapter_name(),
        os = shell("sw_vers", &["-productVersion"]),
        date = shell("date", &["-u", "+%Y-%m-%d"]),
        first = 1 + WARMUP,
        last = WARMUP + TIMED,
    );
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/bench/speed-gate.md");
    std::fs::create_dir_all(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/bench"))?;
    std::fs::write(path, &report)?;
    println!("{report}");
    Ok(())
}
```

Add to the `justfile`, after `blender-test`:

```make
# The piece 2a speed gate: step time and divergence at 128³ (spec §4.3).
# Takes minutes, needs the real GPU, and is not part of `check`.
bench-gate:
    cargo run --release -p elements-ember --example speed_gate
```

- [ ] **Step 5: Run the tests, then `just check`**

Run: `cargo nextest run -p elements-ember --test bench` then `just check` (clippy builds the example through `--all-targets`).
Expected: PASS. Do **not** run `just bench-gate` in this task.

- [ ] **Step 6: Prove each test can fail**

| Test | Mutation |
|---|---|
| `the_gate_passes_with_the_largest_n_…` | in `gate_verdict`, `.max()` → `.min()` |
| `the_gate_limits_are_inclusive` | `r.step_ms_median <= GATE_STEP_MS` → `<` |
| `the_gate_fails_when_no_n_meets_both_limits` | `&& r.ratio <= GATE_RATIO` → `\|\| r.ratio <= GATE_RATIO` |
| `the_plume_scene_loads_and_steps` | in `Scene::document`, `edge(1, 0, 2, 0)` → `edge(1, 2, 2, 0)` (the output then receives the velocity, a `VectorField`, and `into_graph` rejects the type mismatch) |

- [ ] **Step 7: Commit and push**

Subject: `Add the speed-gate runner`.

---

### Task 11: Run the gate and stop

This task produces a number and a decision, not code. **It must be run by the controller, not a subagent, on this machine with nothing else heavy running,** because timing is the whole point.

**Files:**
- Create: `docs/bench/speed-gate.md` (written by the runner)
- Modify: `.superpowers/sdd/progress.md`

- [ ] **Step 1: Run the gate**

```bash
just bench-gate
```

Expected: several minutes of progress on stderr, then the report on stdout. Read the whole table, not only the verdict line.

- [ ] **Step 2: Sanity-check the numbers before trusting them**

- Step times should grow roughly linearly with N. If they do not (for example, N = 160 is no slower than N = 20), the pressure loop is not doing what the table claims: stop and investigate.
- The "RMS div before" column should be similar across N, since each is the same scene; wildly different values need an explanation before the ratio means anything.
- The ratio should fall as N rises.

- [ ] **Step 3: Commit the report**

```bash
git add docs/bench/speed-gate.md
git commit -F - <<'MSG'
Record the piece 2a speed gate

This is the measurement piece 2a exists to take: step time and residual
divergence at 128³ for 20 to 160 pressure iterations, with the rule fixed
in the spec before any number existed. The decision line is left pending
for the user.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TSRpZQEY9fnKQqHLh5WHpU
MSG
git push
```

- [ ] **Step 4: Stop and bring the table to the user**

Update the ledger, then present the table and the mechanical verdict.

- **PASS:** propose the provisional `pressure_iterations`, and wait for the user to confirm. Only after confirmation: set `default_pressure_iterations()` in `solver.rs` and `ITERATIONS` in `projection_leaves_at_most_a_tenth_…` to the confirmed N (spec §4.2), fill in the decision line, commit, and move on to designing 2b.
- **FAIL:** write no more solver code. Present spec §5.5's options (fewer iterations and what that costs in divergence, the multigrid stretch goal, a lower interactive resolution) with the table's evidence for each.
