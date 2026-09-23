# Ember Piece 2b-1 — Solver Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ember.smoke_solver` correct enough to benchmark against Mantaflow: CFL substeps, RK2 + MacCormack advection, vorticity confinement, dissipation, per-face boundaries, a warm start that survives a change of `h`, split submissions, no copies of unread outputs, and quality presets.

**Architecture:** Every change lives in `elements-ember` except three generic additions to `elements-core`: a deterministic GPU `reduce`, `ComputeBatch::flush`, and `EvalCtx::output_wanted` (plus a counter and an error variant). Kernels keep 2a's shape: one Rust function per stage records dispatches into a `ComputeBatch`, and one WGSL file per kernel is concatenated after `common.wgsl`. The solver's per-frame flow becomes: measure max speed → choose `n` → `n` substeps → copy only the wanted outputs.

**Tech Stack:** Rust 2024, wgpu 30 (WGSL compute), serde, cargo-nextest; `just` recipes.

**Spec:** `docs/superpowers/specs/2026-09-22-ember-solver-2b1-design.md`. Read it first. Section numbers below (§n) refer to it.

## Global Constraints

- Scalar fields are `R32Float`, never `R16Float`.
- `required_features` stays `wgpu::Features::empty()`. No optional wgpu features.
- At most **4 storage textures per shader stage**. Read neighbours through `texture_3d<f32>` + `textureLoad`; write through storage.
- `#![forbid(unsafe_code)]` in `elements-core` and `elements-ember`.
- Crate manifests use `dep.workspace = true`; versions live only in the workspace `Cargo.toml`.
- The same document on the same machine gives bit-identical frames, however a frame is reached.
- A snapshot for frame N is the state entering N. Never cache outputs.
- Documents are untrusted: every parameter is validated at load as a `DocError`, and the daemon must not panic.
- On any error mid-step, every taken state value and every scratch field goes back to the pool before the error returns. State is written back only after a whole frame succeeds.
- **Prove each test can fail:** apply the task's listed mutation (exactly one change), run the test, see it fail, restore, and record the real failure output in the commit body.
- `just check` must pass before every commit. Never set `WGPU_BACKEND` locally.
- Check wgpu APIs against the vendored source, not docs.rs:
  `R=$(find ~/.cargo/registry/src -maxdepth 2 -type d -name 'wgpu-30*' | head -1)`.
- Commits: plain imperative subject, a body explaining **why**, ending with
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TSRpZQEY9fnKQqHLh5WHpU
  ```
  Commit with plain `git` (the repo is jj-colocated; do not run `jj`).
- Branch: `ember-solver-2b1` (already exists, based on main `0182146`).

## Conventions every task relies on

- **Face and cell grids.** A velocity face grid along axis `a` has dims `cells` plus one along `a`; its texel `i` sits at cell-unit position `i + grid_offset(a)`, where `grid_offset(a)` is 0.5 on every axis except 0 on `a`. A cell grid (`CELL = 3`) has dims `cells` and offset 0.5 on every axis.
- **Open mask.** Bit `2·axis + side` is set when that domain face is open; side 0 is the low face. So `-x`=bit0, `+x`=bit1, `-y`=bit2, `+y`=bit3, `-z`=bit4, `+z`=bit5. 2a's boundaries (walls everywhere but `+z`) are `1 << 5 = 32`.
- **Kernel uniform.** `Params` in `common.wgsl` mirrors `KernelParams` in `kernels/mod.rs` field for field. Any change to one changes the other in the same commit.
- **CPU references** live in `crates/elements-ember/tests/common/mod.rs` and mirror the WGSL exactly (same operation order), so GPU-vs-CPU tolerances stay at about 1e-5.

## File map

| File | Change | Responsibility |
|---|---|---|
| `crates/elements-core/src/gpu/reduce.rs` | create (T1) | `ReduceOp`, `ReduceTarget`, `reduce` |
| `crates/elements-core/src/gpu/shaders/reduce_{common,partial,final}.wgsl` | create (T1) | the two reduction passes |
| `crates/elements-core/src/gpu/batch.rs` | modify (T1, T4) | `dispatch_workgroups`, `flush`, `submitted_any` |
| `crates/elements-core/src/gpu/pool.rs` | modify (T1) | `acquisitions()` counter |
| `crates/elements-core/src/graph/node.rs`, `graph/mod.rs` | modify (T1, T7) | `output_wanted`, `cfl_clamped`, `SolverDiverged` |
| `crates/elements-ember/src/boundaries.rs` | create (T2) | `Face`, `Boundaries`, `DEFAULT_OPEN_MASK` |
| `crates/elements-ember/src/kernels/shaders/common.wgsl` | modify (T2, T5, T6) | `Params`, open mask, grid sampling |
| `crates/elements-ember/src/kernels/shaders/advect.wgsl`, `maccormack.wgsl` | create (T5) | RK2 advection passes, MacCormack correction |
| `crates/elements-ember/src/kernels/shaders/subtract_mean.wgsl` | create (T3) | closed-domain mean removal |
| `crates/elements-ember/src/kernels/shaders/curl.wgsl`, `confine.wgsl` | create (T6) | vorticity confinement |
| `crates/elements-ember/src/kernels/vorticity.rs` | create (T6) | records curl + confine |
| `crates/elements-ember/src/cfl.rs` | create (T7) | `plan_substeps`, `max_speed` |
| `crates/elements-ember/src/solver.rs` | modify (all) | params, presets, substep flow |
| `crates/elements-ember/src/bench.rs`, `examples/presets.rs`, `justfile` | modify/create (T8) | preset sweep and rule |

---

## Task 1: Core `reduce`, pool acquisition count, and `output_wanted`

Spec §4.6 and §4.7. Resolves risk (h).

**Files:**
- Create: `crates/elements-core/src/gpu/reduce.rs`
- Create: `crates/elements-core/src/gpu/shaders/reduce_common.wgsl`, `reduce_partial.wgsl`, `reduce_final.wgsl`
- Modify: `crates/elements-core/src/gpu/mod.rs` (module + re-exports)
- Modify: `crates/elements-core/src/gpu/batch.rs` (add `dispatch_workgroups`)
- Modify: `crates/elements-core/src/gpu/pool.rs` (add `acquisitions`)
- Modify: `crates/elements-core/src/graph/node.rs` (`EvalCtx::result`, `output_wanted`)
- Modify: `crates/elements-core/src/graph/mod.rs` (pass the result socket; share the wanted rule)
- Modify: `crates/elements-ember/src/solver.rs` (copy only wanted outputs)
- Test: `crates/elements-core/tests/reduce.rs` (create), `crates/elements-core/tests/output_wanted.rs` (create), `crates/elements-ember/tests/solver.rs`

**Interfaces:**
- Produces (core, re-exported from `elements_core::gpu`):
  - `pub enum ReduceOp { MaxAbs, Sum }`
  - `pub struct ReduceTarget` with `pub fn new(ctx: &GpuContext, slots: u32) -> Result<Self, GpuError>`, `pub fn buffer(&self) -> &wgpu::Buffer`, `pub fn slots(&self) -> u32`, `pub fn read(&self, ctx: &GpuContext) -> Result<Vec<f32>, GpuError>`
  - `pub fn reduce(ctx: &GpuContext, cache: &mut PipelineCache, batch: &mut ComputeBatch, src: &Field, op: ReduceOp, target: &ReduceTarget, slot: u32) -> Result<(), GpuError>`
  - `ComputeBatch::dispatch_workgroups(&mut self, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, workgroups: [u32; 3])`
  - `FieldPool::acquisitions(&self) -> u64`
  - `EvalCtx::output_wanted(&self, index: u32) -> bool`

- [ ] **Step 1: Write the failing reduction tests**

Create `crates/elements-core/tests/reduce.rs`:

```rust
use elements_core::gpu::{
    ComputeBatch, FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache, ReduceOp,
    ReduceTarget, reduce,
};

fn gpu() -> GpuContext {
    GpuContext::new_headless().expect("no GPU adapter available")
}

/// Irregular values in about [-1, 1], with the largest magnitude placed in
/// the very last voxel so a reduction that drops the last partial workgroup
/// gets the max wrong, not just the sum.
fn values(dims: FieldDims) -> Vec<f32> {
    let n = dims.voxel_count();
    let mut out: Vec<f32> = (0..n)
        .map(|i| ((i * 73 + 29) % 211) as f32 / 105.0 - 1.0)
        .collect();
    out[n - 1] = -5.0;
    out
}

fn run(gpu: &GpuContext, dims: FieldDims, data: &[f32]) -> [f32; 2] {
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let field = pool.acquire(gpu, dims, FieldFormat::R32Float).unwrap();
    field.write(gpu, data).unwrap();
    let target = ReduceTarget::new(gpu, 3).unwrap();
    let mut batch = ComputeBatch::new();
    reduce(gpu, &mut cache, &mut batch, &field, ReduceOp::MaxAbs, &target, 0).unwrap();
    reduce(gpu, &mut cache, &mut batch, &field, ReduceOp::Sum, &target, 2).unwrap();
    batch.submit(gpu).unwrap();
    let got = target.read(gpu).unwrap();
    assert_eq!(got[1], 0.0, "slot 1 was never written and must stay zero");
    [got[0], got[2]]
}

#[test]
fn max_abs_and_sum_match_the_cpu() {
    let gpu = gpu();
    // 105 voxels: one workgroup. 9240 voxels: three, the last one partial.
    for dims in [FieldDims::new(7, 5, 3), FieldDims::new(40, 33, 7)] {
        let data = values(dims);
        let [max, sum] = run(&gpu, dims, &data);
        let want_max = data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let want_sum: f64 = data.iter().map(|&v| f64::from(v)).sum();
        assert_eq!(max, want_max, "{dims:?} max");
        assert!(
            (f64::from(sum) - want_sum).abs() <= 1e-3 * want_sum.abs().max(1.0),
            "{dims:?} sum {sum} vs {want_sum}"
        );
    }
}

#[test]
fn a_reduction_is_bit_identical_run_to_run() {
    let gpu = gpu();
    let dims = FieldDims::new(40, 33, 7);
    let data = values(dims);
    let first = run(&gpu, dims, &data);
    for _ in 0..3 {
        let again = run(&gpu, dims, &data);
        assert_eq!(first.map(f32::to_bits), again.map(f32::to_bits));
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo nextest run -p elements-core --test reduce`
Expected: compile error, `ReduceOp`, `ReduceTarget` and `reduce` not found in `elements_core::gpu`.

- [ ] **Step 3: Add `dispatch_workgroups` to `ComputeBatch`**

In `crates/elements-core/src/gpu/batch.rs`, replace the body of `dispatch` with a call to a new method, and add the method:

```rust
    /// Record one invocation per voxel of `dims`.
    pub fn dispatch(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        dims: FieldDims,
    ) {
        self.dispatch_workgroups(
            pipeline,
            bind_group,
            [
                dims.x.div_ceil(WORKGROUP),
                dims.y.div_ceil(WORKGROUP),
                dims.z.div_ceil(WORKGROUP),
            ],
        );
    }

    /// Record a dispatch of exactly `workgroups`, for kernels that do not
    /// run one invocation per voxel, such as a reduction.
    pub fn dispatch_workgroups(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: [u32; 3],
    ) {
        self.dispatches.push(Dispatch {
            pipeline: pipeline.clone(),
            bind_group: bind_group.clone(),
            workgroups,
        });
    }
```

- [ ] **Step 4: Write the reduction shaders**

Create `crates/elements-core/src/gpu/shaders/reduce_common.wgsl`:

```wgsl
// Shared by both reduction passes, concatenated in front of each.
// `op` 0 is max |x|, 1 is sum. Both folds start from 0, which is the
// identity for each (|x| is never negative).

struct ReduceParams {
    dims: vec3<u32>,
    op: u32,
    count: u32, // partial pass: workgroups covering the field; final pass: partials to fold
    slot: u32,  // final pass: the target slot to write
    _pad0: u32,
    _pad1: u32,
};

const THREADS: u32 = 64u;    // must match THREADS in reduce.rs
const PER_THREAD: u32 = 64u; // must match PER_THREAD in reduce.rs

var<workgroup> scratch: array<f32, 64>;

fn combine(a: f32, b: f32) -> f32 {
    if (params.op == 0u) {
        return max(a, b);
    }
    return a + b;
}

// Fold `scratch` pairwise in a fixed order; the result ends in scratch[0].
// The order never depends on timing, so the result is bit-deterministic.
fn tree(lid: u32) {
    for (var stride = THREADS / 2u; stride > 0u; stride = stride / 2u) {
        if (lid < stride) {
            scratch[lid] = combine(scratch[lid], scratch[lid + stride]);
        }
        workgroupBarrier();
    }
}
```

Create `crates/elements-core/src/gpu/shaders/reduce_partial.wgsl`:

```wgsl
// Pass 1: each workgroup folds THREADS * PER_THREAD voxels into one partial.
// Workgroups are laid out in 2D, because one dimension allows at most 65535.

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<uniform> params: ReduceParams;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let group = wid.x + wid.y * groups.x;
    let total = params.dims.x * params.dims.y * params.dims.z;
    let base = group * THREADS * PER_THREAD;
    var acc = 0.0;
    for (var n = 0u; n < PER_THREAD; n = n + 1u) {
        let i = base + n * THREADS + lid;
        if (i < total) {
            let x = i % params.dims.x;
            let y = (i / params.dims.x) % params.dims.y;
            let z = i / (params.dims.x * params.dims.y);
            var v = textureLoad(src, vec3<i32>(vec3<u32>(x, y, z)), 0).x;
            if (params.op == 0u) {
                v = abs(v);
            }
            acc = combine(acc, v);
        }
    }
    scratch[lid] = acc;
    workgroupBarrier();
    tree(lid);
    if (lid == 0u && group < params.count) {
        partials[group] = scratch[0];
    }
}
```

Create `crates/elements-core/src/gpu/shaders/reduce_final.wgsl`:

```wgsl
// Pass 2: one workgroup folds every partial into the target slot.

@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> slots: array<f32>;
@group(0) @binding(2) var<uniform> params: ReduceParams;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32) {
    var acc = 0.0;
    for (var i = lid; i < params.count; i = i + THREADS) {
        acc = combine(acc, partials[i]);
    }
    scratch[lid] = acc;
    workgroupBarrier();
    tree(lid);
    if (lid == 0u) {
        slots[params.slot] = scratch[0];
    }
}
```

- [ ] **Step 5: Write `reduce.rs`**

Create `crates/elements-core/src/gpu/reduce.rs`:

```rust
//! Reducing a field to one number on the GPU: the max of |x|, or the sum.

use wgpu::util::DeviceExt;

use super::{ComputeBatch, Field, FieldFormat, GpuContext, GpuError, PipelineCache};

const PARTIAL: &str = concat!(
    include_str!("shaders/reduce_common.wgsl"),
    include_str!("shaders/reduce_partial.wgsl"),
);
const FINAL: &str = concat!(
    include_str!("shaders/reduce_common.wgsl"),
    include_str!("shaders/reduce_final.wgsl"),
);

/// Invocations per workgroup; `THREADS` in `reduce_common.wgsl`.
const THREADS: u32 = 64;
/// Voxels each invocation folds before the workgroup's tree; `PER_THREAD`.
const PER_THREAD: u32 = 64;
/// WebGPU's per-dimension limit on workgroup counts.
const MAX_GROUPS_PER_DIM: u32 = 65535;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// The largest |x|.
    MaxAbs,
    /// The sum of x.
    Sum,
}

/// Matches `ReduceParams` in `reduce_common.wgsl`, 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    dims: [u32; 3],
    op: u32,
    count: u32,
    slot: u32,
    _pad: [u32; 2],
}

/// Where reductions land: `slots` values in one GPU buffer, zero at creation.
///
/// A later kernel in the same batch can bind `buffer()` as
/// `var<storage, read>` and use a result without a round trip to the CPU.
pub struct ReduceTarget {
    buffer: wgpu::Buffer,
    slots: u32,
}

impl ReduceTarget {
    pub fn new(ctx: &GpuContext, slots: u32) -> Result<Self, GpuError> {
        let buffer = ctx.scoped(|| {
            ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("reduce-target"),
                size: u64::from(slots.max(1)) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })?;
        Ok(Self { buffer, slots })
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// Every slot, read back to the CPU. Call it after submitting the batch
    /// that wrote them; it blocks until the GPU has finished.
    pub fn read(&self, ctx: &GpuContext) -> Result<Vec<f32>, GpuError> {
        let size = self.buffer.size();
        let staging = ctx.scoped(|| {
            let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("reduce-readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder =
                ctx.device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("reduce-readback"),
                    });
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, size);
            ctx.queue().submit(Some(encoder.finish()));
            staging
        })?;

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        ctx.scoped(|| {
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            })
        })?;
        ctx.wait()?;
        rx.recv()
            .map_err(|e| GpuError::Validation(e.to_string()))?
            .map_err(|e| GpuError::Validation(e.to_string()))?;
        let values = {
            let mapped = slice
                .get_mapped_range()
                .map_err(|e| GpuError::Validation(e.to_string()))?;
            let floats: &[f32] = bytemuck::cast_slice(&mapped);
            floats[..self.slots as usize].to_vec()
        };
        ctx.scoped(|| staging.unmap())?;
        Ok(values)
    }
}

/// Record a reduction of `src` into `target`'s `slot`. Nothing runs until
/// `batch` is submitted.
///
/// Each workgroup folds a fixed run of voxels into a partial, and one
/// workgroup then folds the partials, all in a fixed order. So the result is
/// bit-identical from run to run on one backend. A sum's rounding differs
/// from a sequential CPU sum.
pub fn reduce(
    ctx: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    src: &Field,
    op: ReduceOp,
    target: &ReduceTarget,
    slot: u32,
) -> Result<(), GpuError> {
    if src.format() != FieldFormat::R32Float {
        return Err(GpuError::Validation(format!(
            "reduce: a {:?} field, expected R32Float",
            src.format()
        )));
    }
    if slot >= target.slots {
        return Err(GpuError::Validation(format!(
            "reduce: slot {slot} of a {}-slot target",
            target.slots
        )));
    }
    let dims = src.dims();
    let per_group = u64::from(THREADS * PER_THREAD);
    let groups = u32::try_from((dims.voxel_count() as u64).div_ceil(per_group))
        .map_err(|_| GpuError::Validation(format!("reduce: {dims:?} is too large")))?;
    let partial = cache.get_or_create(ctx, "core.reduce.partial", PARTIAL, "main")?;
    let fold = cache.get_or_create(ctx, "core.reduce.final", FINAL, "main")?;
    let params = ReduceParams {
        dims: [dims.x, dims.y, dims.z],
        op: match op {
            ReduceOp::MaxAbs => 0,
            ReduceOp::Sum => 1,
        },
        count: groups,
        slot,
        _pad: [0; 2],
    };
    // The bind groups hold the partials and uniform buffers alive until the
    // batch has run, so the handles can go out of scope here.
    let (partial_group, final_group) = ctx.scoped(|| {
        let partials = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduce-partials"),
            size: u64::from(groups) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let uniform = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reduce-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let partial_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduce-partial"),
            layout: &partial.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let final_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduce-final"),
            layout: &fold.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        (partial_group, final_group)
    })?;
    batch.dispatch_workgroups(
        &partial,
        &partial_group,
        [
            groups.min(MAX_GROUPS_PER_DIM),
            groups.div_ceil(MAX_GROUPS_PER_DIM),
            1,
        ],
    );
    batch.dispatch_workgroups(&fold, &final_group, [1, 1, 1]);
    Ok(())
}
```

Check `copy_buffer_to_buffer`'s last parameter in the vendored `wgpu` source (`grep -n 'fn copy_buffer_to_buffer' -A8 $R/src/api/command_encoder.rs`). If it takes `impl Into<Option<BufferAddress>>`, passing `size` works; if it takes `BufferAddress`, it also works. Only change it if it does not compile.

In `crates/elements-core/src/gpu/mod.rs`, add `mod reduce;` next to the other modules, and `pub use reduce::{ReduceOp, ReduceTarget, reduce};`.

- [ ] **Step 6: Run the reduction tests**

Run: `cargo nextest run -p elements-core --test reduce`
Expected: 2 passed.

- [ ] **Step 7: Prove the reduction test can fail**

Mutation (one change): in `reduce`, replace `.div_ceil(per_group)` with `/ per_group`, which drops the last, partial workgroup.
Run: `cargo nextest run -p elements-core --test reduce max_abs_and_sum_match_the_cpu`
Expected: FAIL on `FieldDims { x: 40, y: 33, z: 7 } max` (the -5.0 in the last voxel is lost). Record the output, then restore.

- [ ] **Step 8: Write the failing `output_wanted` test**

Create `crates/elements-core/tests/output_wanted.rs`:

```rust
use std::sync::{Arc, Mutex};

use elements_core::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{
    EvalCtx, Graph, Node, NodeError, NodeId, SocketId, SocketSpec, SocketType, StateStore, Time,
    Value,
};
use elements_core::nodes::{ConstantField, Output};

/// Two field outputs; records what `output_wanted` said about each.
struct Probe(Arc<Mutex<Vec<bool>>>);

impl Node for Probe {
    fn kind(&self) -> &'static str {
        "test.probe"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field],
            outputs: vec![SocketType::Field, SocketType::Field],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        *self.0.lock().unwrap() = vec![ctx.output_wanted(0), ctx.output_wanted(1)];
        let field = ctx.take_input(0)?;
        Ok(vec![field, Value::Scalar(0.0)])
    }
}

fn link(g: &mut Graph, from: NodeId, from_index: u32, to: NodeId, to_index: u32) {
    g.connect(
        SocketId { node: from, index: from_index },
        SocketId { node: to, index: to_index },
    )
    .unwrap();
}

fn wanted(probe_is_result: bool) -> Vec<bool> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut g = Graph::new();
    let source = g.add_node(Box::new(ConstantField { value: 1.0 }));
    let probe = g.add_node(Box::new(Probe(Arc::clone(&seen))));
    link(&mut g, source, 0, probe, 0);
    if probe_is_result {
        g.set_output(probe);
    } else {
        let out = g.add_node(Box::new(Output));
        link(&mut g, probe, 0, out, 0);
        g.set_output(out);
    }
    let gpu = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let evaluated = g
        .eval_frame(
            &gpu,
            &mut pool,
            &mut pipelines,
            &mut state,
            Time::at(1, 1, 24.0),
            FieldDims::new(4, 4, 4),
        )
        .unwrap();
    evaluated.value.release_to(&mut pool);
    let result = seen.lock().unwrap().clone();
    result
}

#[test]
fn an_output_is_wanted_when_a_node_reads_it_or_it_is_the_result() {
    assert_eq!(wanted(false), vec![true, false], "read by the output node");
    assert_eq!(wanted(true), vec![true, false], "socket 0 is the graph's result");
}
```

Run: `cargo nextest run -p elements-core --test output_wanted`
Expected: compile error, no method `output_wanted` on `EvalCtx`.

- [ ] **Step 9: Implement `output_wanted`**

In `crates/elements-core/src/graph/node.rs`, add a field to `EvalCtx` after `remaining`:

```rust
    /// The graph's result socket, which is wanted even though no node reads it.
    pub(crate) result: SocketId,
```

and a method in `impl EvalCtx<'_>`, after `node_id`:

```rust
    /// Whether output `index` of this node will be used: it is the graph's
    /// result, or a node still to run reads it.
    ///
    /// A node may skip computing an unwanted output and put any cheap value,
    /// such as `Value::Scalar(0.0)`, in its slot. The evaluator releases
    /// unwanted outputs without passing them to any node, so none can see it.
    pub fn output_wanted(&self, index: u32) -> bool {
        is_wanted(
            self.result,
            self.remaining,
            SocketId {
                node: self.node,
                index,
            },
        )
    }
```

and a free function at module level in `node.rs`:

```rust
/// The one rule for whether an output socket is used this evaluation.
pub(crate) fn is_wanted(
    result: SocketId,
    remaining: &HashMap<SocketId, u32>,
    socket: SocketId,
) -> bool {
    socket == result || remaining.get(&socket).copied().unwrap_or(0) > 0
}
```

In `crates/elements-core/src/graph/mod.rs`, inside `Graph::run`, add `result: result_socket,` to the `EvalCtx { … }` literal, and replace

```rust
                let wanted =
                    socket == result_socket || run.remaining.get(&socket).copied().unwrap_or(0) > 0;
```

with

```rust
                let wanted = node::is_wanted(result_socket, &run.remaining, socket);
```

Run `grep -rn "EvalCtx {" crates/elements-core/src` and add `result` to any other literal it finds. If `mod.rs` refers to the `node` module under another path, use that path.

- [ ] **Step 10: Run the `output_wanted` test**

Run: `cargo nextest run -p elements-core --test output_wanted`
Expected: 1 passed.

Mutation: make `output_wanted` return `true` unconditionally. Expected: FAIL with `left: [true, true]`. Record, restore.

- [ ] **Step 11: Count pool acquisitions**

In `crates/elements-core/src/gpu/pool.rs`, add a field `acquisitions: u64` to `FieldPool` (initialised to 0 in `new`), increment it at the end of every successful `acquire` (just before returning `Ok`), and add:

```rust
    /// Successful `acquire` calls, whether they reused a texture or created
    /// one. Every other acquiring method goes through `acquire`.
    pub fn acquisitions(&self) -> u64 {
        self.acquisitions
    }
```

- [ ] **Step 12: Write the failing solver test for unread outputs**

In `crates/elements-ember/tests/solver.rs`, add (reusing the file's imports; add `StateStore` and `Time` to the `elements_core::graph` import):

```rust
/// Reads all three solver outputs and passes density on.
struct Sink;

impl Node for Sink {
    fn kind(&self) -> &'static str {
        "test.sink"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field, SocketType::Field, SocketType::VectorField],
            outputs: vec![SocketType::Field],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        Ok(vec![ctx.take_input(0)?])
    }
}

/// Pool acquisitions over one frame, with either every solver output read
/// or only density.
fn acquisitions_for_one_frame(read_all: bool) -> u64 {
    let registry = elements_ember::registry();
    let mut graph = Graph::new();
    let emitter = graph.add_node(
        registry
            .build(
                "ember.sphere_emitter",
                &serde_json::json!({ "center": [1.0, 1.0, 0.4], "radius": 0.3 }),
            )
            .unwrap(),
    );
    let solver = graph.add_node(
        registry
            .build(KIND, &serde_json::json!({ "pressure_iterations": 4 }))
            .unwrap(),
    );
    let output = graph.add_node(registry.build("core.output", &serde_json::json!({})).unwrap());
    let socket = |node: NodeId, index: u32| SocketId { node, index };
    graph.connect(socket(emitter, 0), socket(solver, 0)).unwrap();
    graph.connect(socket(emitter, 1), socket(solver, 1)).unwrap();
    if read_all {
        let sink = graph.add_node(Box::new(Sink));
        for index in 0..3 {
            graph.connect(socket(solver, index), socket(sink, index)).unwrap();
        }
        graph.connect(socket(sink, 0), socket(output, 0)).unwrap();
    } else {
        graph.connect(socket(solver, 0), socket(output, 0)).unwrap();
    }
    graph.set_output(output);

    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let evaluated = graph
        .eval_frame(
            &gpu,
            &mut pool,
            &mut pipelines,
            &mut state,
            Time::at(1, 1, 24.0),
            FieldDims::new(8, 6, 5),
        )
        .unwrap();
    evaluated.value.release_to(&mut pool);
    state.clear(&mut pool);
    pool.acquisitions()
}

/// Risk (h): outputs nobody reads are never copied. Temperature is one
/// field and velocity three faces, so reading only density saves four.
#[test]
fn outputs_nobody_reads_are_never_copied() {
    let all = acquisitions_for_one_frame(true);
    let density_only = acquisitions_for_one_frame(false);
    assert_eq!(all - density_only, 4, "all {all}, density only {density_only}");
}
```

Run: `cargo nextest run -p elements-ember --test solver outputs_nobody_reads_are_never_copied`
Expected: FAIL, `all … density only …` differ by 0.

- [ ] **Step 13: Copy only wanted outputs in the solver**

In `crates/elements-ember/src/solver.rs`, replace the output-copying block at the end of `SmokeSolver::run` (from `// The outputs are copies` to the end of that `with_gpu_pool` call) with:

```rust
        // The outputs are copies: the state stays in the store for the next
        // frame. Outputs nobody reads are not copied at all.
        let wanted: [bool; 3] = std::array::from_fn(|i| ctx.output_wanted(i as u32));
        ctx.with_gpu_pool(|gpu, _, pool| copy_outputs(gpu, pool, state, wanted))
```

and add, next to `duplicate_velocity`:

```rust
/// Pooled copies of the wanted outputs, in socket order, with a
/// `Value::Scalar(0.0)` placeholder in each unwanted slot (see
/// `EvalCtx::output_wanted`). On failure, copies already made go back.
fn copy_outputs(
    gpu: &GpuContext,
    pool: &mut FieldPool,
    state: &SolverState,
    wanted: [bool; 3],
) -> Result<Vec<Value>, GpuError> {
    let mut outputs: Vec<Value> = Vec::with_capacity(3);
    for (index, wanted) in wanted.into_iter().enumerate() {
        let copied = if !wanted {
            Ok(Value::Scalar(0.0))
        } else {
            match index {
                0 => pool.duplicate(gpu, &state.density).map(Value::Field),
                1 => pool.duplicate(gpu, &state.temperature).map(Value::Field),
                _ => duplicate_velocity(gpu, pool, &state.velocity).map(Value::VectorField),
            }
        };
        match copied {
            Ok(value) => outputs.push(value),
            Err(e) => {
                for value in outputs {
                    value.release_to(pool);
                }
                return Err(e);
            }
        }
    }
    Ok(outputs)
}
```

- [ ] **Step 14: Run and prove it**

Run: `cargo nextest run -p elements-ember --test solver`
Expected: all pass, including `outputs_nobody_reads_are_never_copied`.

Mutation: in `SmokeSolver::run`, replace `ctx.output_wanted(i as u32)` with `true`. Expected: FAIL with a difference of 0. Record, restore.

- [ ] **Step 15: Run the gate and commit**

Run: `just check`
Expected: PASS.

```bash
git add crates/elements-core crates/elements-ember/src/solver.rs crates/elements-ember/tests/solver.rs
git commit   # subject: "Add a GPU reduction, and copy only solver outputs someone reads"
```

The body explains why: CFL and the closed-domain fix both need a deterministic GPU reduction, and core owns it because Tide will need it too; the solver was copying temperature and velocity every frame even when only density was read (risk h). Include both mutation outputs.

---
## Task 2: Per-face boundaries, stored `p`, and ambient scalars at open faces

Spec §2 (`boundaries`), §4.2 and §4.3 (first bullet). Resolves risks (b) and (d).

**Files:**
- Create: `crates/elements-ember/src/boundaries.rs`
- Modify: `crates/elements-ember/src/lib.rs` (`pub mod boundaries;`)
- Modify: `crates/elements-ember/src/kernels/mod.rs` (`StepConstants`, `KernelParams`, `Uniforms`)
- Modify: `crates/elements-ember/src/kernels/project.rs` (rename `phi` → `p`)
- Modify: `crates/elements-ember/src/kernels/shaders/common.wgsl`, `velocity.wgsl`, `advect_velocity.wgsl`, `advect_scalar.wgsl`, `buoyancy.wgsl`, `pressure.wgsl`, `gradient.wgsl`
- Modify: `crates/elements-ember/src/solver.rs` (`SolverParams`: container default, `boundaries`, `step_constants`)
- Modify: `crates/elements-ember/src/bench.rs`, `crates/elements-ember/examples/speed_gate.rs`
- Test: `crates/elements-ember/tests/common/mod.rs`, `tests/advection.rs`, `tests/projection.rs`, `tests/forces.rs`, `tests/scenes.rs`, `tests/solver.rs`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `elements_ember::boundaries::{Face, Boundaries, DEFAULT_OPEN_MASK}`; `Boundaries::open_mask(&self) -> u32`; `Boundaries::closed() -> Self` (every face a wall)
  - `StepConstants { cells, h, dx, alpha, beta, open_mask }` and `StepConstants::new(cells: FieldDims, h: f32, dx: f32) -> StepConstants` (no buoyancy, `DEFAULT_OPEN_MASK`). Later tasks add fields and give them defaults in `new`, so **every literal must use `..StepConstants::new(…)`**.
  - `Uniforms::open_mask(&self) -> u32`
  - `SolverParams` gets `impl Default` and `pub boundaries: Boundaries`, plus `SolverParams::step_constants(&self, cells: FieldDims, h: f32, dx: f32) -> StepConstants`
  - WGSL in `common.wgsl`: `CELL`, `is_open(axis, side)`, `is_wall(axis, i)`, `grid_offset(axis)`, `grid_dims(axis)`, `texel(tex, axis, c)`, `Corners`, `corners(tex, axis, p)`, `sample_grid(tex, axis, p)`. `trilinear`, `face_offset` and `face_dims` are removed.
  - CPU mirrors in `tests/common/mod.rs`: `is_open(mask, axis, side)`, `is_wall_in(cells, mask, axis, i)`, `is_wall(cells, axis, i)` (default mask), `texel`, `corners`, `sample_grid(data, dims, open: Option<u32>, p)`, `cpu_advect_scalar(faces, cells, mask, src, h_inv_dx)`, `cpu_advect_velocity(faces, cells, mask, h_inv_dx)`, `cpu_red_black(p, div, cells, mask, dx2, scale, iterations)`

- [ ] **Step 1: Write `boundaries.rs` with its unit tests**

Create `crates/elements-ember/src/boundaries.rs`:

```rust
//! Which faces of the domain are solid walls and which are open (spec §4.2).

use serde::{Deserialize, Serialize};

/// 2a's boundaries: walls everywhere except the top (`+z`), so a plume
/// leaves the domain instead of piling up at the ceiling.
pub const DEFAULT_OPEN_MASK: u32 = 1 << 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Face {
    /// Zero normal velocity; Neumann pressure.
    Wall,
    /// Pressure 0 beyond it; scalars flowing in are ambient, 0.
    Open,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Boundaries {
    #[serde(rename = "-x")]
    pub neg_x: Face,
    #[serde(rename = "+x")]
    pub pos_x: Face,
    #[serde(rename = "-y")]
    pub neg_y: Face,
    #[serde(rename = "+y")]
    pub pos_y: Face,
    #[serde(rename = "-z")]
    pub neg_z: Face,
    #[serde(rename = "+z")]
    pub pos_z: Face,
}

impl Default for Boundaries {
    fn default() -> Self {
        Self {
            pos_z: Face::Open,
            ..Self::closed()
        }
    }
}

impl Boundaries {
    /// Every face a wall.
    pub fn closed() -> Self {
        Self {
            neg_x: Face::Wall,
            pos_x: Face::Wall,
            neg_y: Face::Wall,
            pos_y: Face::Wall,
            neg_z: Face::Wall,
            pos_z: Face::Wall,
        }
    }

    /// Bit `2·axis + side` is set when that face is open; side 0 is the low
    /// face. This is `Params::open_mask` in `common.wgsl`.
    pub fn open_mask(&self) -> u32 {
        [
            self.neg_x, self.pos_x, self.neg_y, self.pos_y, self.neg_z, self.pos_z,
        ]
        .iter()
        .enumerate()
        .filter(|(_, face)| **face == Face::Open)
        .fold(0, |mask, (bit, _)| mask | 1 << bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mask_has_one_bit_per_open_face_in_axis_then_side_order() {
        assert_eq!(Boundaries::default().open_mask(), DEFAULT_OPEN_MASK);
        assert_eq!(Boundaries::closed().open_mask(), 0);
        let neg_y = Boundaries {
            neg_y: Face::Open,
            ..Boundaries::closed()
        };
        assert_eq!(neg_y.open_mask(), 0b000100);
        let pos_x = Boundaries {
            pos_x: Face::Open,
            ..Boundaries::closed()
        };
        assert_eq!(pos_x.open_mask(), 0b000010);
    }
}
```

Add `pub mod boundaries;` to `crates/elements-ember/src/lib.rs`.

Run: `cargo nextest run -p elements-ember --lib boundaries`
Expected: 1 passed. Mutation: swap `neg_y` and `pos_y` in the array. Expected: FAIL `left: 8, right: 4`. Record, restore.

- [ ] **Step 2: Update the CPU mirrors (the tests' side of the contract)**

In `crates/elements-ember/tests/common/mod.rs`:

1. Add `use elements_ember::boundaries::DEFAULT_OPEN_MASK;` near the other imports.
2. Replace `is_wall` and `trilinear` with:

```rust
/// Mirrors `is_open` in `common.wgsl`.
pub fn is_open(mask: u32, axis: usize, side: usize) -> bool {
    (mask >> (2 * axis + side)) & 1 == 1
}

/// Mirrors `is_wall` in `common.wgsl`.
pub fn is_wall_in(cells: FieldDims, mask: u32, axis: usize, i: u32) -> bool {
    let n = [cells.x, cells.y, cells.z][axis];
    if i == 0 {
        return !is_open(mask, axis, 0);
    }
    if i == n {
        return !is_open(mask, axis, 1);
    }
    false
}

/// `is_wall_in` with 2a's boundaries: walls everywhere but `+z`.
pub fn is_wall(cells: FieldDims, axis: usize, i: u32) -> bool {
    is_wall_in(cells, DEFAULT_OPEN_MASK, axis, i)
}

/// Mirrors `texel` in `common.wgsl`. `open` is `Some(mask)` for a cell
/// grid, which reads 0 beyond an open face; face grids pass `None` and clamp.
pub fn texel(data: &[f32], dims: FieldDims, open: Option<u32>, c: [i32; 3]) -> f32 {
    let size = [dims.x as i32, dims.y as i32, dims.z as i32];
    let mut q = c;
    for a in 0..3 {
        if c[a] < 0 {
            if open.is_some_and(|m| is_open(m, a, 0)) {
                return 0.0;
            }
            q[a] = 0;
        } else if c[a] >= size[a] {
            if open.is_some_and(|m| is_open(m, a, 1)) {
                return 0.0;
            }
            q[a] = size[a] - 1;
        }
    }
    data[index(dims, q[0] as u32, q[1] as u32, q[2] as u32)]
}

/// Mirrors `corners` in `common.wgsl`: the 8 texels around `p` and the
/// fractional position between them.
pub fn corners(data: &[f32], dims: FieldDims, open: Option<u32>, p: [f32; 3]) -> ([f32; 8], [f32; 3]) {
    let size = [dims.x as f32, dims.y as f32, dims.z as f32];
    let mut i0 = [0i32; 3];
    let mut t = [0f32; 3];
    for a in 0..3 {
        let q = p[a].clamp(-1.0, size[a]);
        let f = q.floor();
        i0[a] = f as i32;
        t[a] = q - f;
    }
    let c = std::array::from_fn(|n| {
        texel(
            data,
            dims,
            open,
            [
                i0[0] + (n & 1) as i32,
                i0[1] + ((n >> 1) & 1) as i32,
                i0[2] + ((n >> 2) & 1) as i32,
            ],
        )
    });
    (c, t)
}

/// Mirrors `sample_grid` in `common.wgsl`.
pub fn sample_grid(data: &[f32], dims: FieldDims, open: Option<u32>, p: [f32; 3]) -> f32 {
    let (c, t) = corners(data, dims, open, p);
    let mix = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;
    let c00 = mix(c[0], c[1], t[0]);
    let c10 = mix(c[2], c[3], t[0]);
    let c01 = mix(c[4], c[5], t[0]);
    let c11 = mix(c[6], c[7], t[0]);
    mix(mix(c00, c10, t[1]), mix(c01, c11, t[1]), t[2])
}
```

3. In `velocity_at`, replace the `trilinear(&faces[a], face_dims(cells, a), …)` call with `sample_grid(&faces[a], face_dims(cells, a), None, …)`.
4. Give `cpu_advect_scalar` a `mask: u32` parameter after `cells`, and sample with `sample_grid(src, cells, Some(mask), sub(b, [0.5; 3]))`.
5. Give `cpu_advect_velocity` a `mask: u32` parameter after `cells`; use `is_wall_in(cells, mask, a, [i, j, k][a])` and `sample_grid(&faces[a], d, None, sub(b, off))`.
6. Replace `cpu_red_black` with:

```rust
/// Mirrors `relax` in `pressure.wgsl`: red (even i+j+k) then black, per
/// iteration, solving ∇²p = div / scale.
pub fn cpu_red_black(
    p: &mut [f32],
    div: &[f32],
    cells: FieldDims,
    mask: u32,
    dx2: f32,
    scale: f32,
    iterations: u32,
) {
    let n = [cells.x as i32, cells.y as i32, cells.z as i32];
    for _ in 0..iterations {
        for colour in [0, 1] {
            for k in 0..cells.z {
                for j in 0..cells.y {
                    for i in 0..cells.x {
                        if (i + j + k) % 2 != colour {
                            continue;
                        }
                        let c = [i as i32, j as i32, k as i32];
                        let at = |q: [i32; 3]| p[index(cells, q[0] as u32, q[1] as u32, q[2] as u32)];
                        let mut sum = 0.0f32;
                        let mut count = 0.0f32;
                        for a in 0..3 {
                            let mut lo = c;
                            lo[a] -= 1;
                            let mut hi = c;
                            hi[a] += 1;
                            if c[a] > 0 {
                                sum += at(lo);
                                count += 1.0;
                            } else if is_open(mask, a, 0) {
                                count += 1.0;
                            }
                            if c[a] < n[a] - 1 {
                                sum += at(hi);
                                count += 1.0;
                            } else if is_open(mask, a, 1) {
                                count += 1.0;
                            }
                        }
                        if count == 0.0 {
                            continue;
                        }
                        let rhs = dx2 * div[index(cells, i, j, k)] / scale;
                        p[index(cells, i, j, k)] = (sum - rhs) / count;
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 3: Migrate every `StepConstants` literal and CPU call site**

This step leaves the build red until Step 5; that is expected.

In `crates/elements-ember/src/kernels/mod.rs`, add `open_mask` to `StepConstants` and a constructor:

```rust
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
    /// Open domain faces; see `Boundaries::open_mask`.
    pub open_mask: u32,
}

impl StepConstants {
    /// No buoyancy, and 2a's boundaries. Build variations with
    /// `StepConstants { beta: 1.0, ..StepConstants::new(cells, h, dx) }`, so
    /// fields added later get their defaults here instead of breaking callers.
    pub fn new(cells: FieldDims, h: f32, dx: f32) -> Self {
        Self {
            cells,
            h,
            dx,
            alpha: 0.0,
            beta: 0.0,
            open_mask: DEFAULT_OPEN_MASK,
        }
    }
}
```

(import `crate::boundaries::DEFAULT_OPEN_MASK`). Then rewrite each literal as `StepConstants { alpha: …, beta: …, ..StepConstants::new(cells, h, dx) }`, keeping its values, at:
`tests/forces.rs:10`, `tests/projection.rs:10`, `tests/advection.rs:10`, `tests/scenes.rs:65` and `:133`, `tests/solver.rs:47` and `:282`. The ones in `src/solver.rs` and `examples/speed_gate.rs` become `step_constants` calls in Step 6.

Update the CPU call sites: `tests/advection.rs` passes `DEFAULT_OPEN_MASK` as the new `mask` argument to `cpu_advect_scalar` and `cpu_advect_velocity`. `tests/projection.rs`'s `cpu_red_black(&mut want, &div_values, CELLS, 1.0, 3)` becomes `cpu_red_black(&mut want, &div_values, CELLS, DEFAULT_OPEN_MASK, 1.0, c.h, 3)`.

- [ ] **Step 4: Write the failing boundary and warm-start tests**

In `tests/advection.rs` add:

```rust
/// Spec §4.2, risk (d): a scalar drawn in through an open face is clean
/// air. Only `-x` is open, and a uniform +x velocity of half a cell per step
/// pulls air in through it. The first column mixes half ambient 0 with half
/// its own value; with a clamp it would stay 1.
#[test]
fn inflow_across_an_open_face_carries_clean_air() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let mask = 0b000001;
    let c = StepConstants {
        open_mask: mask,
        ..StepConstants::new(CELLS, 0.25, 0.125)
    };
    // 0.25 m/s × 0.25 s / 0.125 m = half a cell.
    let faces: [Vec<f32>; 3] = std::array::from_fn(|a| {
        vec![if a == 0 { 0.25 } else { 0.0 }; face_dims(CELLS, a).voxel_count()]
    });
    let ones = vec![1.0; CELLS.voxel_count()];
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let src = upload(&gpu, &mut pool, CELLS, &ones);
    let dst = pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap();
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    advect_scalar(&gpu, &mut cache, &mut batch, &u, &velocity, &src, &dst).unwrap();
    batch.submit(&gpu).unwrap();

    let got = dst.read_back(&gpu).unwrap();
    let want = cpu_advect_scalar(&faces, CELLS, mask, &ones, c.h / c.dx);
    assert_close(&got, &want, 1e-6, "advected");
    for k in 0..CELLS.z {
        for j in 0..CELLS.y {
            assert_eq!(got[index(CELLS, 0, j, k)], 0.5, "first column ({j}, {k})");
            assert_eq!(got[index(CELLS, 1, j, k)], 1.0, "second column ({j}, {k})");
        }
    }
}
```

(Add `FieldFormat` to the file's `elements_core::gpu` import if it is missing.)

In `tests/projection.rs`, change `red_black_sweeps_match_the_cpu_reference` to loop over two masks, and add a gradient test for an open low face:

```rust
#[test]
fn red_black_sweeps_match_the_cpu_reference() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    // 2a's boundaries, then +x and -y open: each open side is Dirichlet.
    for mask in [DEFAULT_OPEN_MASK, 0b000110] {
        let mut pool = FieldPool::new();
        let c = StepConstants {
            open_mask: mask,
            ..constants(1.0)
        };
        let div_values = pattern(CELLS, 8);
        let p0 = pattern(CELLS, 9);
        let div = upload(&gpu, &mut pool, CELLS, &div_values);
        let p = upload(&gpu, &mut pool, CELLS, &p0);

        let u = Uniforms::new(&gpu, &c).unwrap();
        let mut batch = ComputeBatch::new();
        pressure(&gpu, &mut cache, &mut batch, &u, &p, &div, 3).unwrap();
        batch.submit(&gpu).unwrap();

        let mut want = p0.clone();
        cpu_red_black(&mut want, &div_values, CELLS, mask, 1.0, c.h, 3);
        assert_close(&p.read_back(&gpu).unwrap(), &want, 1e-4, &format!("p, mask {mask:#b}"));
    }
}

/// Spec §4.2: an open face is not a wall, and beyond it p = 0, so its
/// velocity is corrected by h·(p_inside − 0)/dx.
#[test]
fn an_open_face_sees_zero_pressure_beyond_it() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = StepConstants {
        open_mask: 0b100001, // -x and +z open
        ..constants(0.125)
    };
    let faces = velocity_pattern(CELLS);
    let p_values = pattern(CELLS, 10);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let p = upload(&gpu, &mut pool, CELLS, &p_values);
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    subtract_gradient(&gpu, &mut cache, &mut batch, &u, &velocity, &p).unwrap();
    batch.submit(&gpu).unwrap();

    let got = read_staggered(&gpu, &velocity);
    let d = face_dims(CELLS, 0);
    for k in 0..d.z {
        for j in 0..d.y {
            let at = index(d, 0, j, k);
            let want = faces[0][at] - c.h * (p_values[index(CELLS, 0, j, k)] - 0.0) / c.dx;
            assert!(
                (got[0][at] - want).abs() <= 1e-4,
                "x face (0, {j}, {k}): {} vs {want}",
                got[0][at]
            );
        }
    }
}
```

Also in `subtracting_the_gradient_zeroes_solid_walls_and_matches_the_cpu`, the expected value now carries `h`: change `let want = faces[a][at] - (upper - lower) / dx;` to `let want = faces[a][at] - c.h * (upper - lower) / dx;`, with `let c = constants(dx);` bound before `Uniforms::new(&gpu, &c)`.

(Import `elements_ember::boundaries::DEFAULT_OPEN_MASK` in `projection.rs`.)

In `tests/scenes.rs`, add (reusing its imports: `fill_sphere`, `Sphere`, `divergence`, `SolverState`, `Sources`, `Substep`, `substep`):

```rust
/// Spec §4.3, risk (b): the pressure slot holds p, not h·p, so the warm
/// start stays right when the substep length changes. Run A reaches frame 24
/// at h = 1/24 and then takes one substep at h = 1/72; run B takes every
/// substep at 1/72. With only 20 iterations the warm start dominates, so a
/// warm start stale by a factor of 3 would leave A clearly more divergent.
#[test]
fn the_warm_start_survives_a_change_of_substep_length() {
    const ITERATIONS: u32 = 20;
    fn ratio_after(first_h: f32, first_steps: u32) -> f64 {
        let gpu = gpu();
        let mut pool = FieldPool::new();
        let mut cache = PipelineCache::new();
        let cells = FieldDims::new(16, 16, 16);
        let dx = 2.0 / 16.0;
        let density_source = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
        let temperature_source = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
        let sphere = Sphere {
            center: [1.0, 1.0, 0.4],
            radius: 0.3,
            density_rate: 1.0,
            temperature_rate: 2.0,
        };
        fill_sphere(&gpu, &mut cache, &density_source, &temperature_source, &sphere, dx).unwrap();
        let sources = Sources {
            density: &density_source,
            temperature: &temperature_source,
        };
        let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
        let warm = StepConstants {
            beta: 1.0,
            ..StepConstants::new(cells, first_h, dx)
        };
        for _ in 0..first_steps {
            substep(&gpu, &mut cache, &mut pool, &mut state, sources, &warm, ITERATIONS).unwrap();
        }
        let last = StepConstants {
            beta: 1.0,
            ..StepConstants::new(cells, 1.0 / 72.0, dx)
        };
        let mut step = Substep::new(&gpu, &last).unwrap();
        step.pre_projection(&gpu, &mut cache, &mut pool, &mut state, sources)
            .unwrap();
        step.submit(&gpu, &mut pool).unwrap();
        let before = divergence(&state.read_velocity(&gpu).unwrap(), cells, dx);
        let mut step = Substep::new(&gpu, &last).unwrap();
        step.project(&gpu, &mut cache, &mut pool, &mut state, ITERATIONS)
            .unwrap();
        step.submit(&gpu, &mut pool).unwrap();
        let after = divergence(&state.read_velocity(&gpu).unwrap(), cells, dx);
        f64::from(after.rms) / f64::from(before.rms)
    }
    let switched = ratio_after(1.0 / 24.0, 24);
    let constant = ratio_after(1.0 / 72.0, 72);
    assert!(
        switched <= 1.1 * constant,
        "switched {switched}, constant {constant}"
    );
}
```

In `tests/solver.rs`'s `rejects_out_of_range_solver_parameters`, add:

```rust
    assert!(!rejected(serde_json::json!({ "boundaries": { "-x": "open", "+z": "wall" } })));
    assert!(rejected(serde_json::json!({ "boundaries": { "+w": "open" } })));
    assert!(rejected(serde_json::json!({ "boundaries": { "-x": "porous" } })));
```

- [ ] **Step 5: Rewrite the shared WGSL and the kernels**

Replace `crates/elements-ember/src/kernels/shaders/common.wgsl` entirely with:

```wgsl
// Shared by every Ember kernel except the sphere emitter. Concatenated in
// front of each kernel's own source. Each kernel declares its own
// `params: Params` binding; module-scope order does not matter in WGSL.

struct Params {
    dims: vec3<u32>,     // the domain in cells
    axis: u32,           // 0, 1, 2: a face grid along x, y, z; CELL: cell centres
    h: f32,              // substep length, seconds
    inv_dx: f32,         // 1 / voxel edge, 1/m
    dx2: f32,            // voxel edge squared, m²
    pressure_scale: f32, // h: the solve is ∇²p = div / h, and the subtract is u -= h·∇p
    alpha: f32,          // buoyancy per unit density (sinks), m/s²
    beta: f32,           // buoyancy per unit temperature (rises), m/s²
    open_mask: u32,      // bit 2·axis + side is set when that domain face is open
    _pad0: u32,
};

// `axis` for a cell-centred grid.
const CELL: u32 = 3u;

// Whether the domain face on `side` (0 low, 1 high) of `axis` is open.
fn is_open(axis: u32, side: u32) -> bool {
    return ((params.open_mask >> (2u * axis + side)) & 1u) == 1u;
}

// Whether face `i` along `axis` is a solid wall: a boundary face that is
// not open. Everything wall-related goes through this one function.
fn is_wall(axis: u32, i: u32) -> bool {
    if (i == 0u) {
        return !is_open(axis, 0u);
    }
    if (i == params.dims[axis]) {
        return !is_open(axis, 1u);
    }
    return false;
}

// From a position in cell units to a grid's own texel index space.
fn grid_offset(axis: u32) -> vec3<f32> {
    var o = vec3<f32>(0.5, 0.5, 0.5);
    if (axis < 3u) {
        o[axis] = 0.0;
    }
    return o;
}

// A grid's texel dims: the domain, plus one along a face grid's own axis.
fn grid_dims(axis: u32) -> vec3<u32> {
    var d = params.dims;
    if (axis < 3u) {
        d[axis] = d[axis] + 1u;
    }
    return d;
}

// One texel of a grid at any integer index. A face grid clamps to its edge.
// A cell grid reads the ambient value 0 beyond an open face (spec §4.2)
// and clamps beyond a wall.
fn texel(tex: texture_3d<f32>, axis: u32, c: vec3<i32>) -> f32 {
    if (axis != CELL) {
        let last = vec3<i32>(textureDimensions(tex, 0)) - vec3<i32>(1);
        return textureLoad(tex, clamp(c, vec3<i32>(0), last), 0).x;
    }
    let n = vec3<i32>(params.dims);
    var q = c;
    for (var a = 0u; a < 3u; a = a + 1u) {
        if (c[a] < 0) {
            if (is_open(a, 0u)) {
                return 0.0;
            }
            q[a] = 0;
        } else if (c[a] >= n[a]) {
            if (is_open(a, 1u)) {
                return 0.0;
            }
            q[a] = n[a] - 1;
        }
    }
    return textureLoad(tex, q, 0).x;
}

// The 8 texels around `p` (in the grid's texel index space) and p's
// fractional position between them. `p` is clamped to one ghost layer.
struct Corners {
    c: array<f32, 8>, // bit 0 of the index steps x, bit 1 y, bit 2 z
    t: vec3<f32>,
};

fn corners(tex: texture_3d<f32>, axis: u32, p: vec3<f32>) -> Corners {
    let q = clamp(p, vec3<f32>(-1.0), vec3<f32>(textureDimensions(tex, 0)));
    let f = floor(q);
    let i0 = vec3<i32>(f);
    var out: Corners;
    out.t = q - f;
    for (var n = 0u; n < 8u; n = n + 1u) {
        let o = vec3<i32>(i32(n & 1u), i32((n >> 1u) & 1u), i32((n >> 2u) & 1u));
        out.c[n] = texel(tex, axis, i0 + o);
    }
    return out;
}

// Trilinear interpolation, done by hand because R32Float is not filterable
// without an optional feature (umbrella E2).
fn sample_grid(tex: texture_3d<f32>, axis: u32, p: vec3<f32>) -> f32 {
    let k = corners(tex, axis, p);
    let c00 = mix(k.c[0], k.c[1], k.t.x);
    let c10 = mix(k.c[2], k.c[3], k.t.x);
    let c01 = mix(k.c[4], k.c[5], k.t.x);
    let c11 = mix(k.c[6], k.c[7], k.t.x);
    return mix(mix(c00, c10, k.t.y), mix(c01, c11, k.t.y), k.t.z);
}
```

`velocity.wgsl` becomes:

```wgsl
// Velocity at a cell-unit position. For kernels that declare `vel_x`,
// `vel_y` and `vel_z` as `texture_3d<f32>`.
fn velocity_at(x: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        sample_grid(vel_x, 0u, x - grid_offset(0u)),
        sample_grid(vel_y, 1u, x - grid_offset(1u)),
        sample_grid(vel_z, 2u, x - grid_offset(2u)),
    );
}
```

In `advect_velocity.wgsl`: `component` samples with `sample_grid(vel_x, 0u, p)`, `sample_grid(vel_y, 1u, p)` and `sample_grid(vel_z, 2u, p)`; replace `face_dims(` with `grid_dims(` and `face_offset(` with `grid_offset(`.

In `advect_scalar.wgsl`, the sampling line becomes `let value = sample_grid(src, CELL, back - grid_offset(CELL));`.

In `buoyancy.wgsl`, replace `face_dims(2u)` with `grid_dims(2u)` and the boundary comment with `// Boundary faces are left alone: a wall stays zero, and an open face's value comes from projection.`

Replace `pressure.wgsl` with:

```wgsl
// Stage 4b: red-black Gauss–Seidel for ∇²p = div / h. The state slot holds
// p, not h·p, so the warm start stays valid when h changes (spec §4.3).
// Cells of one colour never read each other, so each sweep is race-free and
// bit-deterministic.

@group(0) @binding(0) var pressure: texture_storage_3d<r32float, read_write>;
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
    for (var a = 0u; a < 3u; a = a + 1u) {
        var e = vec3<i32>(0);
        e[a] = 1;
        // A neighbour inside counts with its value. Beyond an open face
        // p = 0 (Dirichlet), so it counts with nothing added. Beyond a wall
        // (Neumann) it is left out.
        if (c[a] > 0) {
            sum += textureLoad(pressure, c - e).x;
            count += 1.0;
        } else if (is_open(a, 0u)) {
            count += 1.0;
        }
        if (c[a] < n[a] - 1) {
            sum += textureLoad(pressure, c + e).x;
            count += 1.0;
        } else if (is_open(a, 1u)) {
            count += 1.0;
        }
    }
    // Only a closed 1×1×1 domain has no neighbours, and nothing to solve.
    if (count == 0.0) {
        return;
    }
    let rhs = params.dx2 * textureLoad(div, c, 0).x / params.pressure_scale;
    textureStore(pressure, c, vec4<f32>((sum - rhs) / count, 0.0, 0.0, 0.0));
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

Replace `gradient.wgsl` with:

```wgsl
// Stage 4c: u -= h·∇p on one axis's faces. Dispatched once per axis over
// that axis's face dims. Face i lies between cells i - 1 and i.

@group(0) @binding(0) var face: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var pressure: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = params.axis;
    if (any(gid >= grid_dims(axis))) {
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
    // Beyond an open face, p = 0.
    var upper = 0.0;
    if (i < params.dims[axis]) {
        upper = textureLoad(pressure, p, 0).x;
    }
    var lower = 0.0;
    if (i > 0u) {
        lower = textureLoad(pressure, p - e, 0).x;
    }
    let u = textureLoad(face, p).x - params.pressure_scale * (upper - lower) * params.inv_dx;
    textureStore(face, p, vec4<f32>(u, 0.0, 0.0, 0.0));
}
```

In `kernels/mod.rs`, `KernelParams` becomes (still 48 bytes):

```rust
/// Matches `Params` in `common.wgsl`, 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KernelParams {
    dims: [u32; 3],
    axis: u32,
    h: f32,
    inv_dx: f32,
    dx2: f32,
    pressure_scale: f32,
    alpha: f32,
    beta: f32,
    open_mask: u32,
    _pad: u32,
}
```

`Uniforms::new` fills `pressure_scale: c.h`, `open_mask: c.open_mask`, `_pad: 0`. `Uniforms` gains a field `open_mask: u32` and:

```rust
    /// Open domain faces, as in `StepConstants::open_mask`.
    pub fn open_mask(&self) -> u32 {
        self.open_mask
    }
```

In `kernels/project.rs`, rename the `phi` parameters of `pressure` and `subtract_gradient` to `p`, and update their doc comments: `pressure` does "sweeps on `p`, solving ∇²p = div/h", and `subtract_gradient` does "`velocity -= h·∇p`". Also in `src/solver.rs`: the `PRESSURE` slot comment becomes `/// Holds p. The warm start stays valid when h changes (spec §4.3).` and `SolverState::pressure`'s comment becomes `/// p, kept as the next solve's warm start.`

- [ ] **Step 6: Default `SolverParams`, add `boundaries` and `step_constants`**

In `src/solver.rs`, replace the `default_*` functions and the per-field `#[serde(default = …)]` attributes with a container default:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SolverParams {
    /// Fixed substeps per frame. CFL-driven substepping is Task 7.
    pub substeps: u32,
    /// Red-black Gauss–Seidel iterations per substep.
    pub pressure_iterations: u32,
    /// α: downward acceleration per unit density, m/s².
    pub buoyancy_density: f32,
    /// β: upward acceleration per unit temperature, m/s².
    pub buoyancy_temperature: f32,
    /// Which domain faces are open; the rest are walls (spec §4.2).
    pub boundaries: Boundaries,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            substeps: 1,
            // Chosen by the user from the speed gate
            // (`docs/bench/speed-gate.md`): 34 ms a step at 128³.
            pressure_iterations: 160,
            buoyancy_density: 0.0,
            // Provisional until 2b-3 maps parameters to Mantaflow's.
            buoyancy_temperature: 1.0,
            boundaries: Boundaries::default(),
        }
    }
}

impl SolverParams {
    /// Kernel constants for one substep of length `h`, in a domain of
    /// `cells` with voxel edge `dx`.
    pub fn step_constants(&self, cells: FieldDims, h: f32, dx: f32) -> StepConstants {
        StepConstants {
            alpha: self.buoyancy_density,
            beta: self.buoyancy_temperature,
            open_mask: self.boundaries.open_mask(),
            ..StepConstants::new(cells, h, dx)
        }
    }
}
```

In `SmokeSolver::step`, destructure only `substeps` and `pressure_iterations`, and build the constants with:

```rust
        let constants = self.params.step_constants(
            ctx.dims(),
            (ctx.time().dt / substeps as f64) as f32,
            ctx.voxel_size(),
        );
```

In `src/bench.rs`, `Scene::plume`'s solver becomes `SolverParams { substeps: 1, pressure_iterations: 160, buoyancy_density: 0.0, buoyancy_temperature: 1.0, ..SolverParams::default() }`. In `examples/speed_gate.rs`, `divergence_at` builds `let constants = scene.solver.step_constants(cells, (1.0 / scene.fps / substeps as f64) as f32, dx);`.

- [ ] **Step 7: Run everything**

Run: `cargo nextest run -p elements-ember`
Expected: all pass, including the four new tests.

- [ ] **Step 8: Prove the new tests can fail (one mutation each, restore after each)**

| Test | Mutation | Expected |
|---|---|---|
| `inflow_across_an_open_face_carries_clean_air` | in `texel` (`common.wgsl`), delete the `if (is_open(a, 0u)) { return 0.0; }` line | FAIL: first column is 1, not 0.5 |
| `red_black_sweeps_match_the_cpu_reference` | in `relax`, delete the low side's `else if (is_open(a, 0u)) { count += 1.0; }` branch | FAIL on `mask 0b110` |
| `an_open_face_sees_zero_pressure_beyond_it` | make `is_wall` return `i == 0u \|\| i == params.dims[axis]` | FAIL: x face 0 is 0 |
| `the_warm_start_survives_a_change_of_substep_length` | in `Uniforms::new`, `pressure_scale: 1.0` instead of `c.h` (the solver then stores h·p) | FAIL: switched ratio above 1.1 × constant |

Record each failure output. If the warm-start mutation does **not** fail, stop and report both ratios with and without the mutation. Do not change the 1.1 factor on your own.

- [ ] **Step 9: Run the gate and commit**

Run: `just check`
Expected: PASS.

```bash
git add crates/elements-ember
git commit   # subject: "Make domain faces walls or open, and store p for the warm start"
```

The body explains why: storing h·p made the warm start wrong whenever `h` changes, which CFL will do every frame (risk b); a backtrace leaving through the open top clamped to the edge layer and drew smoke back in (risk d); and one `is_wall` is the seam 2b-2's colliders plug into. Include the four mutation outputs.

---
## Task 3: Closed domains

Spec §4.2 (fully closed domain).

**A correction to the spec, found while writing this plan.** The spec removes the mean divergence before the solve *and* the mean of `p` after it. The first is redundant, because no test could observe it. On an inconsistent singular system, Gauss–Seidel still converges in every non-constant component, and the inconsistency only makes the constant component drift. That drift is exactly what removing p's mean takes out. For velocities from a closed box, the sum of the divergence is zero up to rounding anyway, because it telescopes to wall faces that are zero. So this task implements only the removal of p's mean, and Step 6 updates the spec's §4.2 and §7 to say so.

**Files:**
- Create: `crates/elements-ember/src/kernels/shaders/subtract_mean.wgsl`
- Modify: `crates/elements-ember/src/kernels/project.rs` (`remove_mean`, `solve_pressure`), `kernels/mod.rs` (re-exports)
- Modify: `crates/elements-ember/src/solver.rs` (`Substep::project` calls `solve_pressure`)
- Modify: `docs/superpowers/specs/2026-09-22-ember-solver-2b1-design.md` (§4.2, §7)
- Test: `crates/elements-ember/tests/projection.rs`, `tests/scenes.rs`

**Interfaces:**
- Consumes: `ReduceTarget`, `ReduceOp`, `reduce` (Task 1); `Uniforms::open_mask` (Task 2).
- Produces: `kernels::remove_mean(gpu, cache, batch, u: &Uniforms, field: &Field, sum: &ReduceTarget) -> Result<(), GpuError>`; `kernels::solve_pressure(gpu, cache, batch, u: &Uniforms, p: &Field, div: &Field, iterations: u32) -> Result<(), GpuError>`. Task 4 changes `pressure`'s signature; `solve_pressure`'s stays the same.

- [ ] **Step 1: Write the failing tests**

In `tests/projection.rs` (add `solve_pressure` to the `elements_ember::kernels` import):

```rust
/// Spec §4.2: in a closed domain p is defined only up to a constant, so the
/// solve removes p's mean, and a warm start cannot drift. Here the warm
/// start is offset by 3; afterwards the mean is gone.
#[test]
fn a_closed_domain_solve_leaves_p_with_zero_mean() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = StepConstants {
        open_mask: 0,
        ..constants(1.0)
    };
    let div = upload(&gpu, &mut pool, CELLS, &pattern(CELLS, 8));
    let p0: Vec<f32> = pattern(CELLS, 9).iter().map(|v| v + 3.0).collect();
    let p = upload(&gpu, &mut pool, CELLS, &p0);
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    solve_pressure(&gpu, &mut cache, &mut batch, &u, &p, &div, 40).unwrap();
    batch.submit(&gpu).unwrap();

    let got = p.read_back(&gpu).unwrap();
    let mean = got.iter().map(|&v| f64::from(v)).sum::<f64>() / got.len() as f64;
    let max = got.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(mean.abs() <= 1e-3 * f64::from(max), "mean {mean}, max {max}");
}
```

In `tests/scenes.rs`, turn the body of `projection_leaves_at_most_a_tenth_of_the_divergence` into a helper parameterised by the open mask, and call it twice:

```rust
/// RMS divergence after projection over before it, after 20 frames of the
/// 16³ plume with the given open faces.
fn projection_ratio(open_mask: u32) -> f64 {
    const ITERATIONS: u32 = 160;
    // … the existing body, with the constants built as
    // StepConstants { beta: 1.0, open_mask, ..StepConstants::new(cells, 1.0 / 24.0, dx) }
    // and ending in:
    assert!(before.rms > 0.0, "the plume must be moving");
    f64::from(after.rms) / f64::from(before.rms)
}

#[test]
fn projection_leaves_at_most_a_tenth_of_the_divergence() {
    let ratio = projection_ratio(DEFAULT_OPEN_MASK);
    assert!(ratio <= 0.1, "ratio {ratio}");
}

/// Spec §4.2: with every face a wall, the same rule holds.
#[test]
fn a_closed_box_meets_the_same_divergence_rule() {
    let ratio = projection_ratio(0);
    assert!(ratio <= 0.1, "ratio {ratio}");
}
```

(Import `elements_ember::boundaries::DEFAULT_OPEN_MASK`.)

Run: `cargo nextest run -p elements-ember --test projection a_closed_domain_solve_leaves_p_with_zero_mean`
Expected: compile error, `solve_pressure` not found.

- [ ] **Step 2: Write the mean-removal kernel**

Create `crates/elements-ember/src/kernels/shaders/subtract_mean.wgsl`:

```wgsl
// Subtract a field's mean, given its sum from a reduction recorded earlier
// in the same batch. Used on p when every domain face is a wall (spec §4.2).

@group(0) @binding(0) var field: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var<storage, read> sum: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let cells = f32(params.dims.x * params.dims.y * params.dims.z);
    let p = vec3<i32>(gid);
    let value = textureLoad(field, p).x - sum[0] / cells;
    textureStore(field, p, vec4<f32>(value, 0.0, 0.0, 0.0));
}
```

In `kernels/project.rs`, add:

```rust
const SUBTRACT_MEAN: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/subtract_mean.wgsl"),
);

/// Remove `field`'s mean on the GPU, with no readback: a sum reduction into
/// `sum` (one slot), then a pass subtracting sum / cells. The bind groups
/// keep `sum` alive until the batch has run.
pub fn remove_mean(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    field: &Field,
    sum: &ReduceTarget,
) -> Result<(), GpuError> {
    expect_dims("remove_mean field", field, u.cells())?;
    reduce(gpu, cache, batch, field, ReduceOp::Sum, sum, 0)?;
    let pipeline = cache.get_or_create(gpu, "ember.subtract_mean", SUBTRACT_MEAN, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[Bind::Tex(field), Bind::Buf(sum.buffer()), Bind::Buf(u.any())],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// The whole pressure solve: `iterations` red-black sweeps on `p`, starting
/// from whatever `p` holds (the warm start). In a closed domain (every face
/// a wall) the Neumann system defines p only up to a constant, so p's mean
/// is removed afterwards and the warm start cannot drift (spec §4.2).
pub fn solve_pressure(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    p: &Field,
    div: &Field,
    iterations: u32,
) -> Result<(), GpuError> {
    pressure(gpu, cache, batch, u, p, div, iterations)?;
    if u.open_mask() == 0 {
        let sum = ReduceTarget::new(gpu, 1)?;
        remove_mean(gpu, cache, batch, u, p, &sum)?;
    }
    Ok(())
}
```

(Import `elements_core::gpu::{ReduceOp, ReduceTarget, reduce}` in `project.rs`, and add `remove_mean` and `solve_pressure` to the `pub use project::{…}` line in `kernels/mod.rs`.)

- [ ] **Step 3: Use it in the solver**

In `Substep::project` (`src/solver.rs`), replace the `kernels::pressure(…)` call inside the `and_then` chain with `kernels::solve_pressure(gpu, cache, &mut self.batch, u, &state.pressure, &div, iterations)`.

- [ ] **Step 4: Run the tests**

Run: `cargo nextest run -p elements-ember`
Expected: all pass.

- [ ] **Step 5: Prove them**

| Test | Mutation | Expected |
|---|---|---|
| `a_closed_domain_solve_leaves_p_with_zero_mean` | in `solve_pressure`, change `u.open_mask() == 0` to `false` | FAIL: mean ≈ 3 |
| `a_closed_box_meets_the_same_divergence_rule` | in `projection_ratio`, `ITERATIONS` 160 → 10 | FAIL: ratio above 0.1 |

The second mutation only shows the scene test can fail. The kernel test is what guards the closed-domain code. Record both, restore.

- [ ] **Step 6: Correct the spec**

In `docs/superpowers/specs/2026-09-22-ember-solver-2b1-design.md`:

§4.2, replace the "Fully closed domain" bullet with:

```markdown
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
```

§7, the "Closed domain" row becomes:
`| Closed domain | from a warm start offset by 3, p's mean is ≈ 0 after the solve; a closed 16³ plume meets 2a's divergence-ratio rule | skip the mean removal |`

- [ ] **Step 7: Run the gate and commit**

Run: `just check`
Expected: PASS.

```bash
git add crates/elements-ember docs/superpowers/specs/2026-09-22-ember-solver-2b1-design.md
git commit   # subject: "Remove p's mean in a closed domain"
```

The body says why the spec's second removal was dropped, in the words of Step 6, and gives the mutation outputs.

---

## Task 4: Split long pressure loops across submissions

Spec §4.3 (second bullet). Resolves risk (f).

**Files:**
- Modify: `crates/elements-core/src/gpu/batch.rs` (`flush`, `submitted_any`)
- Modify: `crates/elements-ember/src/kernels/project.rs` (`pressure` takes `per_submit`; `iterations_per_submit`)
- Modify: `crates/elements-ember/src/solver.rs` (`Substep::abandon` waits if anything was submitted)
- Test: `crates/elements-ember/tests/projection.rs`

**Interfaces:**
- Consumes: `solve_pressure` (Task 3).
- Produces: `ComputeBatch::flush(&mut self, ctx: &GpuContext) -> Result<(), GpuError>`; `ComputeBatch::submitted_any(&self) -> bool`; `kernels::pressure(gpu, cache, batch, u, p, div, iterations: u32, per_submit: u32)`; `kernels::iterations_per_submit(cells: FieldDims) -> u32`; `kernels::CELL_SWEEPS_PER_SUBMIT: u64`; `Substep::abandon(self, gpu: &GpuContext, pool: &mut FieldPool)`.

- [ ] **Step 1: Write the failing tests**

In `tests/projection.rs` (add `iterations_per_submit` to the kernels import):

```rust
/// Spec §4.3, risk (f): splitting the pressure loop into several
/// submissions changes nothing, bit for bit.
#[test]
fn splitting_the_pressure_loop_changes_nothing() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    let c = constants(1.0);
    let mut run = |per_submit: u32| -> Vec<u32> {
        let mut pool = FieldPool::new();
        let div = upload(&gpu, &mut pool, CELLS, &pattern(CELLS, 8));
        let p = upload(&gpu, &mut pool, CELLS, &pattern(CELLS, 9));
        let u = Uniforms::new(&gpu, &c).unwrap();
        let mut batch = ComputeBatch::new();
        pressure(&gpu, &mut cache, &mut batch, &u, &p, &div, 7, per_submit).unwrap();
        batch.submit(&gpu).unwrap();
        p.read_back(&gpu).unwrap().iter().map(|v| v.to_bits()).collect()
    };
    let whole = run(7);
    assert!(run(1) == whole, "one iteration per submission");
    assert!(run(3) == whole, "three iterations per submission");
}

/// About 2³⁰ cell sweeps per submission: 100 ms of work at 128³.
#[test]
fn the_submission_budget_is_two_to_the_thirty_cell_sweeps() {
    assert_eq!(iterations_per_submit(FieldDims::new(128, 128, 128)), 512);
    assert_eq!(iterations_per_submit(FieldDims::new(512, 512, 512)), 8);
    assert_eq!(iterations_per_submit(FieldDims::new(2048, 2048, 2048)), 1);
}
```

Every existing `pressure(…, 3)` call in the tests gains a final argument `3` (one submission).

Run: `cargo nextest run -p elements-ember --test projection`
Expected: compile errors, wrong argument count and `iterations_per_submit` not found.

- [ ] **Step 2: Add `flush` to `ComputeBatch`**

In `crates/elements-core/src/gpu/batch.rs`, add `submissions: u32` to the struct (it derives `Default`, so it starts at 0). Move the encoding out of `submit` into `flush`, and make `submit` call it:

```rust
    /// Submit everything recorded so far as one submission, then keep
    /// recording. Dispatches recorded after a flush still see every write
    /// before it: the queue runs submissions in order.
    ///
    /// Fields the flushed work uses may still be in use on the GPU, so they
    /// must not go back to a pool until the batch's last submission.
    pub fn flush(&mut self, ctx: &GpuContext) -> Result<(), GpuError> {
        if self.dispatches.is_empty() {
            return Ok(());
        }
        let dispatches = std::mem::take(&mut self.dispatches);
        ctx.scoped(|| {
            let mut encoder =
                ctx.device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("elements-batch"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("elements-batch"),
                    timestamp_writes: None,
                });
                for d in &dispatches {
                    pass.set_pipeline(&d.pipeline);
                    pass.set_bind_group(0, &d.bind_group, &[]);
                    let [x, y, z] = d.workgroups;
                    pass.dispatch_workgroups(x, y, z);
                }
            }
            ctx.queue().submit(Some(encoder.finish()));
        })?;
        self.submissions += 1;
        Ok(())
    }

    /// Whether any recorded work has reached the queue.
    pub fn submitted_any(&self) -> bool {
        self.submissions > 0
    }

    /// Submit everything still recorded.
    pub fn submit(mut self, ctx: &GpuContext) -> Result<(), GpuError> {
        self.flush(ctx)
    }
```

Update the struct's doc comment: "Nothing touches the GPU until `flush` or `submit`, …".

- [ ] **Step 3: Split the loop in `pressure`**

In `kernels/project.rs`:

```rust
/// About 100 ms of pressure sweeps per submission at the measured 0.19 ms
/// per iteration at 128³, well inside GPU watchdog limits (spec §4.3).
pub const CELL_SWEEPS_PER_SUBMIT: u64 = 1 << 30;

/// Pressure iterations per submission for a domain of `cells`: at least 1.
pub fn iterations_per_submit(cells: FieldDims) -> u32 {
    let per = CELL_SWEEPS_PER_SUBMIT / (cells.voxel_count() as u64).max(1);
    u32::try_from(per).unwrap_or(u32::MAX).max(1)
}
```

Give `pressure` a final parameter `per_submit: u32` and replace its loop with:

```rust
    // A long loop is split across submissions, so one submission never runs
    // long enough to trip a GPU watchdog (risk f). Order is unchanged.
    let per_submit = per_submit.max(1);
    for i in 0..iterations {
        if i > 0 && i % per_submit == 0 {
            batch.flush(gpu)?;
        }
        batch.dispatch(&red, &red_group, u.cells());
        batch.dispatch(&black, &black_group, u.cells());
    }
```

`solve_pressure` passes `iterations_per_submit(u.cells())`. Re-export `iterations_per_submit` and `CELL_SWEEPS_PER_SUBMIT` from `kernels`.

- [ ] **Step 4: Wait before releasing retired fields after a partial submit**

In `src/solver.rs`, `Substep::abandon` becomes:

```rust
    /// Discard everything still recorded without running it.
    ///
    /// Work already flushed may still be reading retired fields, so this
    /// waits for the GPU before releasing them (spec §5). The state passed to
    /// the stages may now hold fields that were never written, so the caller
    /// must discard the state too.
    pub fn abandon(self, gpu: &GpuContext, pool: &mut FieldPool) {
        if self.batch.submitted_any() {
            // The step has already failed; a second error adds nothing.
            let _ = gpu.wait();
        }
        for field in self.retired {
            pool.release(field);
        }
    }
```

and `substep` calls `step.abandon(gpu, pool)`.

- [ ] **Step 5: Run and prove**

Run: `cargo nextest run -p elements-ember && cargo nextest run -p elements-core`
Expected: all pass.

| Test | Mutation | Expected |
|---|---|---|
| `splitting_the_pressure_loop_changes_nothing` | in `flush`, `let mut dispatches = std::mem::take(&mut self.dispatches); dispatches.pop();` (drop the dispatch recorded just before each flush) | FAIL: "one iteration per submission" |
| `the_submission_budget_is_two_to_the_thirty_cell_sweeps` | `CELL_SWEEPS_PER_SUBMIT` = `1 << 29` | FAIL: 256 ≠ 512 |

Record, restore.

- [ ] **Step 6: Run the gate and commit**

Run: `just check`
Expected: PASS.

```bash
git add crates/elements-core crates/elements-ember
git commit   # subject: "Split long pressure loops across submissions"
```

The body explains why: at 256³–512³ with offline iteration counts, a single substep could hold thousands of dispatches for seconds and trip a GPU watchdog, which surfaces as `DeviceLost` (risk f). 128³ presets are unaffected: K = 512 there.

---
## Task 5: RK2 backtrace, MacCormack advection, fused dissipation

Spec §4.1 and §4.5. Resolves risk (c), apart from CFL.

**Files:**
- Create: `crates/elements-ember/src/kernels/shaders/advect.wgsl`, `maccormack.wgsl`
- Delete: `crates/elements-ember/src/kernels/shaders/advect_velocity.wgsl`, `advect_scalar.wgsl`
- Modify: `crates/elements-ember/src/kernels/shaders/common.wgsl` (`decay`), `velocity.wgsl` (`backtrace`)
- Modify: `crates/elements-ember/src/kernels/advect.rs` (rewrite), `kernels/mod.rs`
- Modify: `crates/elements-ember/src/solver.rs` (`Substep` advection, `SolverParams` fields)
- Test: `tests/common/mod.rs`, `tests/advection.rs`, `tests/solver.rs`

**Interfaces:**
- Consumes: `sample_grid`, `corners`, `grid_offset`, `grid_dims`, `CELL` (Task 2).
- Produces:
  - `kernels::Advection { SemiLagrangian, MacCormack }` (serde names `"semi_lagrangian"`, `"maccormack"`)
  - `kernels::Carried { Face(Axis), Density, Temperature }`, with `Carried::dims(self, cells: FieldDims) -> FieldDims`
  - `kernels::Pass { SemiLagrangian, Forward, Backward }`
  - `kernels::advect(gpu, cache, batch, u, carried: Carried, pass: Pass, velocity: &StaggeredField, src: &Field, dst: &Field) -> Result<(), GpuError>`
  - `kernels::maccormack(gpu, cache, batch, u, carried: Carried, velocity: &StaggeredField, orig: &Field, fwd: &Field, bwd: &Field, dst: &Field) -> Result<(), GpuError>`
  - `StepConstants` gains `advection: Advection` (default `MacCormack` in `new`), `density_dissipation: f32`, `temperature_dissipation: f32` (default 0)
  - `SolverParams` gains `advection`, `density_dissipation`, `temperature_dissipation`
  - CPU mirrors: `Grid { Face(usize), Cell }`, `cpu_advect(faces, cells, mask, grid, src, k, decay)`, `cpu_maccormack(faces, cells, mask, grid, src, k, decay)`, where `k = h / dx` (the kernels multiply it by a direction). `cpu_advect_scalar` and `cpu_advect_velocity` are removed.
  - The Rust functions `advect_velocity` and `advect_scalar` are removed.

- [ ] **Step 1: Update the CPU mirrors**

In `tests/common/mod.rs`, replace `backtrace`, `cpu_advect_scalar` and `cpu_advect_velocity` with:

```rust
/// Mirrors `backtrace` in `velocity.wgsl`: an RK2 midpoint step. `k` is
/// direction · h / dx; a negative `k` traces forward in time.
fn backtrace(faces: &[Vec<f32>; 3], cells: FieldDims, x: [f32; 3], k: f32) -> [f32; 3] {
    let v = velocity_at(faces, cells, x);
    let mid = [
        x[0] - 0.5 * k * v[0],
        x[1] - 0.5 * k * v[1],
        x[2] - 0.5 * k * v[2],
    ];
    let v = velocity_at(faces, cells, mid);
    [x[0] - k * v[0], x[1] - k * v[1], x[2] - k * v[2]]
}

/// The grid an advection pass carries.
#[derive(Clone, Copy, Debug)]
pub enum Grid {
    Face(usize),
    Cell,
}

fn grid_dims(cells: FieldDims, grid: Grid) -> FieldDims {
    match grid {
        Grid::Face(a) => face_dims(cells, a),
        Grid::Cell => cells,
    }
}

fn grid_offset(grid: Grid) -> [f32; 3] {
    match grid {
        Grid::Face(a) => face_offset(a),
        Grid::Cell => [0.5; 3],
    }
}

fn grid_open(grid: Grid, mask: u32) -> Option<u32> {
    match grid {
        Grid::Face(_) => None,
        Grid::Cell => Some(mask),
    }
}

fn is_wall_texel(cells: FieldDims, mask: u32, grid: Grid, ijk: [u32; 3]) -> bool {
    match grid {
        Grid::Face(a) => is_wall_in(cells, mask, a, ijk[a]),
        Grid::Cell => false,
    }
}

/// Mirrors `pass_over` in `advect.wgsl`.
pub fn cpu_advect(
    faces: &[Vec<f32>; 3],
    cells: FieldDims,
    mask: u32,
    grid: Grid,
    src: &[f32],
    k: f32,
    decay: f32,
) -> Vec<f32> {
    let d = grid_dims(cells, grid);
    let off = grid_offset(grid);
    let mut out = vec![0.0; d.voxel_count()];
    for kk in 0..d.z {
        for j in 0..d.y {
            for i in 0..d.x {
                if is_wall_texel(cells, mask, grid, [i, j, kk]) {
                    continue;
                }
                let x = [i as f32 + off[0], j as f32 + off[1], kk as f32 + off[2]];
                let b = backtrace(faces, cells, x, k);
                out[index(d, i, j, kk)] =
                    sample_grid(src, d, grid_open(grid, mask), sub(b, off)) * decay;
            }
        }
    }
    out
}

/// Mirrors `advect.wgsl`'s forward and backward passes and `maccormack.wgsl`.
pub fn cpu_maccormack(
    faces: &[Vec<f32>; 3],
    cells: FieldDims,
    mask: u32,
    grid: Grid,
    src: &[f32],
    k: f32,
    decay: f32,
) -> Vec<f32> {
    let fwd = cpu_advect(faces, cells, mask, grid, src, k, 1.0);
    let bwd = cpu_advect(faces, cells, mask, grid, &fwd, -k, 1.0);
    let d = grid_dims(cells, grid);
    let off = grid_offset(grid);
    let mut out = vec![0.0; d.voxel_count()];
    for kk in 0..d.z {
        for j in 0..d.y {
            for i in 0..d.x {
                if is_wall_texel(cells, mask, grid, [i, j, kk]) {
                    continue;
                }
                let x = [i as f32 + off[0], j as f32 + off[1], kk as f32 + off[2]];
                let b = backtrace(faces, cells, x, k);
                let (c, _) = corners(src, d, grid_open(grid, mask), sub(b, off));
                let lo = c.iter().copied().fold(c[0], f32::min);
                let hi = c.iter().copied().fold(c[0], f32::max);
                let at = index(d, i, j, kk);
                let corrected = fwd[at] + 0.5 * (src[at] - bwd[at]);
                out[at] = corrected.max(lo).min(hi) * decay;
            }
        }
    }
    out
}
```

- [ ] **Step 2: Write the failing tests**

Rewrite the calls in `tests/advection.rs` (add `Advection`, `Carried`, `Pass`, `advect`, `maccormack` to the `elements_ember::kernels` import):

- `constants(h, dx)` returns `StepConstants::new(CELLS, h, dx)`.
- In `an_integer_uniform_velocity_shifts_a_scalar_exactly`, `a_fractional_velocity_advects_a_scalar_like_the_cpu_reference` and `inflow_across_an_open_face_carries_clean_air`, replace `advect_scalar(&gpu, &mut cache, &mut batch, &u, &velocity, &src, &dst)` with `advect(&gpu, &mut cache, &mut batch, &u, Carried::Density, Pass::SemiLagrangian, &velocity, &src, &dst)`, and each CPU call with `cpu_advect(&faces, CELLS, <its mask>, Grid::Cell, <its src values>, c.h / c.dx, 1.0)`.
- In `velocity_advection_matches_the_cpu_reference_and_zeroes_solid_walls`, loop `for (a, axis) in AXES.iter().enumerate()`, calling `advect(…, Carried::Face(*axis), Pass::SemiLagrangian, &velocity, velocity.face(*axis), dst.face(*axis))` into one batch. Compare each face with `cpu_advect(&faces, CELLS, DEFAULT_OPEN_MASK, Grid::Face(a), &faces[a], c.h / c.dx, 1.0)`.

Then add these tests to `tests/advection.rs`:

```rust
/// Spec §4.1: the backtrace is an RK2 midpoint step. A solid-body rotation
/// about the domain's vertical centre line curves every path, so an Euler
/// step lands measurably elsewhere.
#[test]
fn rk2_follows_a_rotation_like_the_cpu_reference() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(12, 10, 5);
    let c = StepConstants::new(cells, 0.1, 0.125);
    let omega = 2.0; // rad/s: 0.2 rad per step
    let (cx, cy) = (6.0 * c.dx, 5.0 * c.dx);
    let faces: [Vec<f32>; 3] = std::array::from_fn(|a| {
        let d = face_dims(cells, a);
        let off = face_offset(a);
        let mut face = vec![0.0; d.voxel_count()];
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let x = (i as f32 + off[0]) * c.dx;
                    let y = (j as f32 + off[1]) * c.dx;
                    face[index(d, i, j, k)] = match a {
                        0 => -omega * (y - cy),
                        1 => omega * (x - cx),
                        _ => 0.0,
                    };
                }
            }
        }
        face
    });
    let src_values = pattern(cells, 21);
    let velocity = upload_staggered(&gpu, &mut pool, cells, &faces);
    let src = upload(&gpu, &mut pool, cells, &src_values);
    let dst = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    advect(&gpu, &mut cache, &mut batch, &u, Carried::Density, Pass::SemiLagrangian, &velocity, &src, &dst).unwrap();
    batch.submit(&gpu).unwrap();
    let want = cpu_advect(&faces, cells, DEFAULT_OPEN_MASK, Grid::Cell, &src_values, c.h / c.dx, 1.0);
    assert_close(&dst.read_back(&gpu).unwrap(), &want, 1e-5, "rotated");
}

/// Record one MacCormack step of `src` (on `carried`'s grid) into `dst`,
/// with pooled scratch for the forward and backward passes.
#[allow(clippy::too_many_arguments)]
fn maccormack_step(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    pool: &mut FieldPool,
    u: &Uniforms,
    carried: Carried,
    velocity: &StaggeredField,
    src: &Field,
    dst: &Field,
) {
    let fwd = pool.acquire(gpu, src.dims(), FieldFormat::R32Float).unwrap();
    let bwd = pool.acquire(gpu, src.dims(), FieldFormat::R32Float).unwrap();
    let mut batch = ComputeBatch::new();
    advect(gpu, cache, &mut batch, u, carried, Pass::Forward, velocity, src, &fwd).unwrap();
    advect(gpu, cache, &mut batch, u, carried, Pass::Backward, velocity, &fwd, &bwd).unwrap();
    maccormack(gpu, cache, &mut batch, u, carried, velocity, src, &fwd, &bwd, dst).unwrap();
    batch.submit(gpu).unwrap();
    pool.release(fwd);
    pool.release(bwd);
}

#[test]
fn maccormack_matches_the_cpu_reference() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = constants(0.2, 0.125);
    let faces = walled_velocity_pattern(CELLS);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let u = Uniforms::new(&gpu, &c).unwrap();
    let k = c.h / c.dx;

    let src_values = pattern(CELLS, 22);
    let src = upload(&gpu, &mut pool, CELLS, &src_values);
    let dst = pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap();
    maccormack_step(&gpu, &mut cache, &mut pool, &u, Carried::Density, &velocity, &src, &dst);
    let want = cpu_maccormack(&faces, CELLS, DEFAULT_OPEN_MASK, Grid::Cell, &src_values, k, 1.0);
    assert_close(&dst.read_back(&gpu).unwrap(), &want, 1e-4, "scalar");

    for (a, axis) in AXES.iter().enumerate() {
        let d = face_dims(CELLS, a);
        let dst = pool.acquire(&gpu, d, FieldFormat::R32Float).unwrap();
        maccormack_step(&gpu, &mut cache, &mut pool, &u, Carried::Face(*axis), &velocity, velocity.face(*axis), &dst);
        let want = cpu_maccormack(&faces, CELLS, DEFAULT_OPEN_MASK, Grid::Face(a), &faces[a], k, 1.0);
        assert_close(&dst.read_back(&gpu).unwrap(), &want, 1e-4, &format!("face {a}"));
    }
}

/// Advect `values` along +x at 0.3 cells a step for `steps` steps, in a
/// 32×4×4 domain, with either scheme.
fn carry_along_x(values: &[f32], steps: u32, advection: Advection) -> Vec<f32> {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(32, 4, 4);
    let c = StepConstants::new(cells, 0.1, 0.125);
    // 0.375 m/s × 0.1 s / 0.125 m = 0.3 cells.
    let faces: [Vec<f32>; 3] = std::array::from_fn(|a| {
        vec![if a == 0 { 0.375 } else { 0.0 }; face_dims(cells, a).voxel_count()]
    });
    let velocity = upload_staggered(&gpu, &mut pool, cells, &faces);
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut q = upload(&gpu, &mut pool, cells, values);
    for _ in 0..steps {
        let next = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
        match advection {
            Advection::SemiLagrangian => {
                let mut batch = ComputeBatch::new();
                advect(&gpu, &mut cache, &mut batch, &u, Carried::Density, Pass::SemiLagrangian, &velocity, &q, &next).unwrap();
                batch.submit(&gpu).unwrap();
            }
            Advection::MacCormack => {
                maccormack_step(&gpu, &mut cache, &mut pool, &u, Carried::Density, &velocity, &q, &next);
            }
        }
        pool.release(std::mem::replace(&mut q, next));
    }
    q.read_back(&gpu).unwrap()
}

/// Spec §4.1: MacCormack is why 2b-1 exists (Mantaflow uses it). A smooth
/// bump carried 6 cells keeps a clearly higher peak than semi-Lagrangian.
#[test]
fn maccormack_keeps_a_bump_sharper_than_semi_lagrangian() {
    let cells = FieldDims::new(32, 4, 4);
    let mut bump = vec![0.0; cells.voxel_count()];
    for k in 0..4 {
        for j in 0..4 {
            for i in 0..32 {
                let d = i as f32 - 10.0;
                bump[index(cells, i, j, k)] = (-d * d / 8.0).exp();
            }
        }
    }
    let peak = |v: &[f32]| v.iter().copied().fold(0.0f32, f32::max);
    let semi = peak(&carry_along_x(&bump, 20, Advection::SemiLagrangian));
    let mac = peak(&carry_along_x(&bump, 20, Advection::MacCormack));
    assert!(mac >= semi + 0.05, "MacCormack peak {mac}, semi-Lagrangian {semi}");
}

/// Spec §4.1: the clamp keeps MacCormack inside the range of what it
/// interpolated, so a step never overshoots.
#[test]
fn maccormack_never_leaves_the_source_range() {
    let cells = FieldDims::new(32, 4, 4);
    let mut step = vec![0.0; cells.voxel_count()];
    for k in 0..4 {
        for j in 0..4 {
            for i in 0..16 {
                step[index(cells, i, j, k)] = 1.0;
            }
        }
    }
    for v in carry_along_x(&step, 5, Advection::MacCormack) {
        assert!((0.0..=1.0).contains(&v), "left [0, 1]: {v}");
    }
}

/// Spec §4.5: dissipation multiplies by exp(−rate·h) in the last pass, and
/// only for the scalar it is set on.
#[test]
fn dissipation_decays_by_exp_of_rate_times_h() {
    let gpu = gpu();
    let mut cache = PipelineCache::new();
    for advection in [Advection::SemiLagrangian, Advection::MacCormack] {
        let mut pool = FieldPool::new();
        let c = StepConstants {
            density_dissipation: 2.0,
            ..constants(0.25, 0.125)
        };
        let zero: [Vec<f32>; 3] =
            std::array::from_fn(|a| vec![0.0; face_dims(CELLS, a).voxel_count()]);
        let velocity = upload_staggered(&gpu, &mut pool, CELLS, &zero);
        let u = Uniforms::new(&gpu, &c).unwrap();
        let values = pattern(CELLS, 23);
        let src = upload(&gpu, &mut pool, CELLS, &values);
        for (carried, rate) in [(Carried::Density, 2.0f32), (Carried::Temperature, 0.0)] {
            let dst = pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap();
            match advection {
                Advection::SemiLagrangian => {
                    let mut batch = ComputeBatch::new();
                    advect(&gpu, &mut cache, &mut batch, &u, carried, Pass::SemiLagrangian, &velocity, &src, &dst).unwrap();
                    batch.submit(&gpu).unwrap();
                }
                Advection::MacCormack => {
                    maccormack_step(&gpu, &mut cache, &mut pool, &u, carried, &velocity, &src, &dst);
                }
            }
            let decay = (-rate * c.h).exp();
            let want: Vec<f32> = values.iter().map(|v| v * decay).collect();
            assert_close(
                &dst.read_back(&gpu).unwrap(),
                &want,
                1e-6,
                &format!("{advection:?} {carried:?}"),
            );
        }
    }
}
```

(Add `Field`, `GpuContext`, `StaggeredField` and `FieldFormat` to the file's `elements_core::gpu` import as needed, and `elements_ember::boundaries::DEFAULT_OPEN_MASK`.)

In `tests/solver.rs`'s parameter test, add:

```rust
    assert!(!rejected(serde_json::json!({ "advection": "semi_lagrangian" })));
    assert!(!rejected(
        serde_json::json!({ "advection": "maccormack", "density_dissipation": 0.5 })
    ));
    assert!(rejected(serde_json::json!({ "advection": "bfecc" })));
    assert!(rejected(serde_json::json!({ "density_dissipation": -1.0 })));
    assert!(rejected(serde_json::json!({ "temperature_dissipation": 1e39 })));
```

Run: `cargo nextest run -p elements-ember --test advection`
Expected: compile errors, `advect`, `Carried`, `Pass` and `Advection` not found.

- [ ] **Step 3: Write the shaders**

In `common.wgsl`, rename `_pad0` in `Params` to `decay: f32, // exp(−rate·h) for the scalar a pass carries; 1 otherwise`.

Append to `velocity.wgsl`:

```wgsl

// RK2 (midpoint) backtrace from `x`, in cell units, over one substep.
// `direction` 1 traces back in time; -1 traces forward, for MacCormack's
// backward pass.
fn backtrace(x: vec3<f32>, direction: f32) -> vec3<f32> {
    let k = direction * params.h * params.inv_dx;
    let mid = x - 0.5 * k * velocity_at(x);
    return x - k * velocity_at(mid);
}
```

Create `crates/elements-ember/src/kernels/shaders/advect.wgsl`:

```wgsl
// Stages 3 and 5: one RK2 semi-Lagrangian pass over one grid, either a
// velocity face (params.axis 0–2) or a cell-centred scalar (params.axis =
// CELL). MacCormack runs `forward`, then `backward`, then the correction in
// `maccormack.wgsl`. Plain semi-Lagrangian runs `semi_lagrangian` alone,
// which applies the scalar's dissipation too.

@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var src: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var<uniform> params: Params;

fn pass_over(gid: vec3<u32>, direction: f32, decay: f32) {
    let axis = params.axis;
    if (any(gid >= grid_dims(axis))) {
        return;
    }
    let p = vec3<i32>(gid);
    // Solid walls carry no normal velocity, before projection as well as
    // after, so the divergence the solve sees matches what projection enforces.
    if (axis != CELL && is_wall(axis, gid[axis])) {
        textureStore(dst, p, vec4<f32>(0.0));
        return;
    }
    let x = vec3<f32>(gid) + grid_offset(axis);
    let value = sample_grid(src, axis, backtrace(x, direction) - grid_offset(axis));
    textureStore(dst, p, vec4<f32>(value * decay, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn semi_lagrangian(@builtin(global_invocation_id) gid: vec3<u32>) {
    pass_over(gid, 1.0, params.decay);
}

@compute @workgroup_size(4, 4, 4)
fn forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    pass_over(gid, 1.0, 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    pass_over(gid, -1.0, 1.0);
}
```

Create `crates/elements-ember/src/kernels/shaders/maccormack.wgsl`:

```wgsl
// MacCormack's correction (Selle et al. 2008): q̂ + ½(q − q̃), clamped to
// the range of the 8 texels of q that the forward pass interpolated, then
// the scalar's dissipation. `fwd` is q̂ = A(q) and `bwd` is q̃ = A⁻¹(q̂).

@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var orig: texture_3d<f32>;
@group(0) @binding(4) var fwd: texture_3d<f32>;
@group(0) @binding(5) var bwd: texture_3d<f32>;
@group(0) @binding(6) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = params.axis;
    if (any(gid >= grid_dims(axis))) {
        return;
    }
    let p = vec3<i32>(gid);
    if (axis != CELL && is_wall(axis, gid[axis])) {
        textureStore(dst, p, vec4<f32>(0.0));
        return;
    }
    let x = vec3<f32>(gid) + grid_offset(axis);
    let k = corners(orig, axis, backtrace(x, 1.0) - grid_offset(axis));
    var lo = k.c[0];
    var hi = k.c[0];
    for (var n = 1u; n < 8u; n = n + 1u) {
        lo = min(lo, k.c[n]);
        hi = max(hi, k.c[n]);
    }
    let corrected = textureLoad(fwd, p, 0).x
        + 0.5 * (textureLoad(orig, p, 0).x - textureLoad(bwd, p, 0).x);
    textureStore(dst, p, vec4<f32>(clamp(corrected, lo, hi) * params.decay, 0.0, 0.0, 0.0));
}
```

Delete `advect_velocity.wgsl` and `advect_scalar.wgsl`.

- [ ] **Step 4: Rewrite `advect.rs` and extend the uniforms**

Replace `crates/elements-ember/src/kernels/advect.rs` with:

```rust
//! Advection (stages 3 and 5): RK2 semi-Lagrangian passes, and MacCormack's
//! correction (spec §4.1).

use elements_core::gpu::{
    Axis, ComputeBatch, Field, FieldDims, GpuContext, GpuError, PipelineCache, StaggeredField,
};
use serde::{Deserialize, Serialize};

use super::{Bind, Uniforms, bind_group, expect_dims};

const ADVECT: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/velocity.wgsl"),
    include_str!("shaders/advect.wgsl"),
);

const MACCORMACK: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/velocity.wgsl"),
    include_str!("shaders/maccormack.wgsl"),
);

/// How the solver advects velocity and scalars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Advection {
    /// One RK2 semi-Lagrangian pass: cheap, and diffusive.
    #[serde(rename = "semi_lagrangian")]
    SemiLagrangian,
    /// Forward, backward and a clamped correction: sharper, three passes.
    /// What Blender's Mantaflow gas uses (`order=2`).
    #[serde(rename = "maccormack")]
    MacCormack,
}

/// The grid an advection pass carries. It picks the pass's uniform: its
/// offset, its dims and, for a scalar, its dissipation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Carried {
    Face(Axis),
    Density,
    Temperature,
}

impl Carried {
    /// The texel dims of this grid in a domain of `cells`.
    pub fn dims(self, cells: FieldDims) -> FieldDims {
        match self {
            Self::Face(axis) => StaggeredField::face_dims(cells, axis),
            Self::Density | Self::Temperature => cells,
        }
    }
}

/// One semi-Lagrangian pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    /// The whole of plain semi-Lagrangian advection, dissipation included.
    SemiLagrangian,
    /// MacCormack's forward pass, q̂ = A(q).
    Forward,
    /// MacCormack's backward pass, q̃ = A⁻¹(q̂).
    Backward,
}

fn check(
    what: &str,
    u: &Uniforms,
    carried: Carried,
    velocity: &StaggeredField,
    fields: &[&Field],
) -> Result<(), GpuError> {
    if velocity.cells() != u.cells() {
        return Err(GpuError::Validation(format!(
            "{what}: velocity {:?}, domain {:?}",
            velocity.cells(),
            u.cells()
        )));
    }
    let dims = carried.dims(u.cells());
    for field in fields {
        expect_dims(what, field, dims)?;
    }
    Ok(())
}

/// Record one pass carrying `src` through `velocity` into `dst`. Solid-wall
/// faces of a velocity face grid come out zero.
#[allow(clippy::too_many_arguments)]
pub fn advect(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    carried: Carried,
    pass: Pass,
    velocity: &StaggeredField,
    src: &Field,
    dst: &Field,
) -> Result<(), GpuError> {
    check("advect", u, carried, velocity, &[src, dst])?;
    let (key, entry) = match pass {
        Pass::SemiLagrangian => ("ember.advect.semi_lagrangian", "semi_lagrangian"),
        Pass::Forward => ("ember.advect.forward", "forward"),
        Pass::Backward => ("ember.advect.backward", "backward"),
    };
    let pipeline = cache.get_or_create(gpu, key, ADVECT, entry)?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(velocity.face(Axis::X)),
            Bind::Tex(velocity.face(Axis::Y)),
            Bind::Tex(velocity.face(Axis::Z)),
            Bind::Tex(src),
            Bind::Tex(dst),
            Bind::Buf(u.carried(carried)),
        ],
    )?;
    batch.dispatch(&pipeline, &group, dst.dims());
    Ok(())
}

/// Record MacCormack's correction: `dst` = clamp(`fwd` + ½(`orig` − `bwd`)),
/// times the carried scalar's dissipation.
#[allow(clippy::too_many_arguments)]
pub fn maccormack(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    carried: Carried,
    velocity: &StaggeredField,
    orig: &Field,
    fwd: &Field,
    bwd: &Field,
    dst: &Field,
) -> Result<(), GpuError> {
    check("maccormack", u, carried, velocity, &[orig, fwd, bwd, dst])?;
    let pipeline = cache.get_or_create(gpu, "ember.maccormack", MACCORMACK, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(velocity.face(Axis::X)),
            Bind::Tex(velocity.face(Axis::Y)),
            Bind::Tex(velocity.face(Axis::Z)),
            Bind::Tex(orig),
            Bind::Tex(fwd),
            Bind::Tex(bwd),
            Bind::Tex(dst),
            Bind::Buf(u.carried(carried)),
        ],
    )?;
    batch.dispatch(&pipeline, &group, dst.dims());
    Ok(())
}
```

In `kernels/mod.rs`: `pub use advect::{Advection, Carried, Pass, advect, maccormack};`. `StepConstants` gains

```rust
    /// How velocity and scalars are advected.
    pub advection: Advection,
    /// Exponential decay rate of density, 1/s, applied in its last advection pass.
    pub density_dissipation: f32,
    /// Exponential decay rate of temperature, 1/s.
    pub temperature_dissipation: f32,
```

with `advection: Advection::MacCormack, density_dissipation: 0.0, temperature_dissipation: 0.0` in `new`. Update its doc to "No buoyancy, no dissipation, MacCormack advection and 2a's boundaries". In `KernelParams`, rename `_pad: u32` to `decay: f32`.

`Uniforms` now holds five buffers:

```rust
pub struct Uniforms {
    /// One per face axis: `axis` 0, 1, 2 and `decay` 1.
    faces: [wgpu::Buffer; 3],
    /// Cell grids (`axis` = CELL), each with its scalar's `decay`.
    density: wgpu::Buffer,
    temperature: wgpu::Buffer,
    cells: FieldDims,
    open_mask: u32,
}
```

`Uniforms::new`'s `make` closure takes `(axis: u32, decay: f32)` and builds the `KernelParams` literal as before, plus `decay`:

```rust
        const CELL: u32 = 3; // `CELL` in common.wgsl
        let (faces, density, temperature) = gpu.scoped(|| {
            (
                [make(0, 1.0), make(1, 1.0), make(2, 1.0)],
                make(CELL, (-c.density_dissipation * c.h).exp()),
                make(CELL, (-c.temperature_dissipation * c.h).exp()),
            )
        })?;
```

`axis(axis)` returns `&self.faces[axis_index(axis) as usize]`, `any()` returns `&self.faces[0]`, and there is a new accessor:

```rust
    pub(crate) fn carried(&self, carried: Carried) -> &wgpu::Buffer {
        match carried {
            Carried::Face(axis) => self.axis(axis),
            Carried::Density => &self.density,
            Carried::Temperature => &self.temperature,
        }
    }
```

- [ ] **Step 5: Advect through the new kernels in the solver**

In `src/solver.rs`, `Substep` gains `advection: Advection`, set from `constants.advection` in `Substep::new`. Replace the velocity-advection block at the end of `pre_projection` (from `let advected = pool.acquire_staggered_uninit` to `Ok(())`) with:

```rust
        let cells = self.uniforms.cells();
        let mut faces = Vec::with_capacity(3);
        for axis in Axis::ALL {
            match self.advect_grid(
                gpu,
                cache,
                pool,
                Carried::Face(axis),
                &state.velocity,
                state.velocity.face(axis),
            ) {
                Ok(face) => faces.push(face),
                Err(e) => {
                    self.retired.extend(faces);
                    return Err(e);
                }
            }
        }
        let [x, y, z]: [Field; 3] = match faces.try_into() {
            Ok(array) => array,
            Err(_) => unreachable!("exactly three faces were advected"),
        };
        let advected = StaggeredField::from_faces(cells, [x, y, z])?;
        let old = std::mem::replace(&mut state.velocity, advected);
        self.retired.extend(old.into_faces());
        Ok(())
```

Replace `advect_scalars` and `advect_one` with:

```rust
    /// Stage 5: carry density and temperature through the projected velocity.
    pub fn advect_scalars(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        state: &mut SolverState,
    ) -> Result<(), GpuError> {
        let density =
            self.advect_grid(gpu, cache, pool, Carried::Density, &state.velocity, &state.density)?;
        self.retired
            .push(std::mem::replace(&mut state.density, density));
        let temperature = self.advect_grid(
            gpu,
            cache,
            pool,
            Carried::Temperature,
            &state.velocity,
            &state.temperature,
        )?;
        self.retired
            .push(std::mem::replace(&mut state.temperature, temperature));
        Ok(())
    }

    /// `src`, carried through `velocity` into a fresh pooled field. Scratch
    /// fields, and the new field if recording fails, are retired.
    fn advect_grid(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        carried: Carried,
        velocity: &StaggeredField,
        src: &Field,
    ) -> Result<Field, GpuError> {
        let dst = pool.acquire(gpu, src.dims(), FieldFormat::R32Float)?;
        let recorded = match self.advection {
            Advection::SemiLagrangian => kernels::advect(
                gpu,
                cache,
                &mut self.batch,
                &self.uniforms,
                carried,
                Pass::SemiLagrangian,
                velocity,
                src,
                &dst,
            ),
            Advection::MacCormack => {
                self.maccormack(gpu, cache, pool, carried, velocity, src, &dst)
            }
        };
        match recorded {
            Ok(()) => Ok(dst),
            Err(e) => {
                self.retired.push(dst);
                Err(e)
            }
        }
    }

    /// Record MacCormack's three passes from `src` into `dst`.
    #[allow(clippy::too_many_arguments)]
    fn maccormack(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        carried: Carried,
        velocity: &StaggeredField,
        src: &Field,
        dst: &Field,
    ) -> Result<(), GpuError> {
        let fwd = pool.acquire(gpu, src.dims(), FieldFormat::R32Float)?;
        let bwd = match pool.acquire(gpu, src.dims(), FieldFormat::R32Float) {
            Ok(field) => field,
            Err(e) => {
                self.retired.push(fwd);
                return Err(e);
            }
        };
        let u = &self.uniforms;
        let recorded = kernels::advect(
            gpu, cache, &mut self.batch, u, carried, Pass::Forward, velocity, src, &fwd,
        )
        .and_then(|()| {
            kernels::advect(
                gpu, cache, &mut self.batch, u, carried, Pass::Backward, velocity, &fwd, &bwd,
            )
        })
        .and_then(|()| {
            kernels::maccormack(
                gpu, cache, &mut self.batch, u, carried, velocity, src, &fwd, &bwd, dst,
            )
        });
        // The batch may reference both whether or not recording finished.
        self.retired.push(fwd);
        self.retired.push(bwd);
        recorded
    }
```

`SolverParams` gains three fields, with defaults `Advection::MacCormack`, `0.0` and `0.0` in `Default`, and `step_constants` passes them on:

```rust
    /// How velocity and scalars are advected (spec §4.1).
    pub advection: Advection,
    /// Exponential decay of density, 1/s (spec §4.5).
    pub density_dissipation: f32,
    /// Exponential decay of temperature, 1/s.
    pub temperature_dissipation: f32,
```

`build` validates them after the buoyancy check:

```rust
    params::finite(
        KIND,
        "dissipation",
        &[params.density_dissipation, params.temperature_dissipation],
    )?;
    if params.density_dissipation < 0.0 || params.temperature_dissipation < 0.0 {
        return Err(params::bad(KIND, "dissipation rates must be at least 0"));
    }
```

Update the module doc of `solver.rs`: "Per substep: emit, buoyancy, advect velocity, project, and advect scalars with dissipation. Vorticity confinement arrives in the next task."

- [ ] **Step 6: Run everything**

Run: `cargo nextest run -p elements-ember`
Expected: all pass. The scene tests (`a_hot_blob_rises_every_frame`, `projection_leaves_at_most_a_tenth_of_the_divergence`, `a_closed_box_meets_the_same_divergence_rule`, `the_warm_start_survives_a_change_of_substep_length`) and `frame_40_is_bit_identical_however_it_is_reached` now run under MacCormack, because `StepConstants::new` and `SolverParams::default()` both default to it. If any of them fails, stop and report the output. Do not retune a threshold.

- [ ] **Step 7: Prove the new tests**

| Test | Mutation | Expected |
|---|---|---|
| `rk2_follows_a_rotation_like_the_cpu_reference` | in `backtrace` (`velocity.wgsl`), return `x - k * velocity_at(x)` | FAIL: "rotated" |
| `maccormack_matches_the_cpu_reference` | in `maccormack.wgsl`, `0.5 *` → `0.25 *` | FAIL: "scalar" |
| `maccormack_keeps_a_bump_sharper_than_semi_lagrangian` | in `maccormack.wgsl`, store `textureLoad(fwd, p, 0).x * params.decay` (no correction) | FAIL: the peaks are equal |
| `maccormack_never_leaves_the_source_range` | replace `clamp(corrected, lo, hi)` with `corrected` | FAIL: a value outside [0, 1] |
| `dissipation_decays_by_exp_of_rate_times_h` | in `Uniforms::new`, `1.0 - c.density_dissipation * c.h` instead of the `exp` | FAIL: "SemiLagrangian Density" |

Record each output, restore.

- [ ] **Step 8: Run the gate and commit**

Run: `just check`
Expected: PASS.

```bash
git add -A crates/elements-ember
git commit   # subject: "Advect with RK2 and MacCormack, and dissipate scalars"
```

The body explains why. Blender's Mantaflow gas advects everything with MacCormack (`order=2`, verified in the Blender 5.2.2 binary), so without it the benchmark would measure a detail gap that is cheap to close. RK2 replaces the single Euler step that risk (c) named. Dissipation is fused into the last pass, so it costs no extra full-grid pass. Face and cell grids share one shader, keyed by `params.axis`.

---
## Task 6: Vorticity confinement

Spec §4.4.

**Deviation from spec §7:** the spec's "ε = 0 is bit-identical to skipping the stage" is not a test here. With ε = 0 the solver never records the stage, so the claim holds by construction and no mutation of the kernels could make such a test fail. What replaces it is a physics check that can fail: confinement must strengthen a plume's vorticity.

**Files:**
- Create: `crates/elements-ember/src/kernels/shaders/curl.wgsl`, `confine.wgsl`
- Create: `crates/elements-ember/src/kernels/vorticity.rs`
- Modify: `crates/elements-ember/src/kernels/shaders/common.wgsl`, `kernels/mod.rs` (`confinement`, `vorticity`)
- Modify: `crates/elements-ember/src/solver.rs` (stage, param)
- Test: `crates/elements-ember/tests/vorticity.rs` (create), `tests/common/mod.rs`, `tests/scenes.rs`, `tests/solver.rs`

**Interfaces:**
- Consumes: `is_wall`, `grid_dims` (Task 2); `Uniforms::any`, `Uniforms::axis`.
- Produces:
  - `kernels::curl(gpu, cache, batch, u, velocity: &StaggeredField, omega: [&Field; 4]) -> Result<(), GpuError>`: writes ωx, ωy, ωz, |ω|
  - `kernels::confine(gpu, cache, batch, u, velocity: &StaggeredField, omega: [&Field; 4]) -> Result<(), GpuError>`
  - `StepConstants::vorticity: f32` (ε, 1/s; 0 in `new`); `SolverParams::vorticity: f32` (default 0)
  - CPU mirror in `tests/common/mod.rs`: `cpu_curl(faces: &[Vec<f32>; 3], cells: FieldDims, inv_dx: f32) -> [Vec<f32>; 4]`

- [ ] **Step 1: Write the CPU mirror and the failing tests**

Add to `tests/common/mod.rs`:

```rust
/// Mirrors `centre_velocity` in `curl.wgsl`: face velocities averaged to
/// the centre of cell `c`, with `c` clamped into the domain.
pub fn centre_velocity(faces: &[Vec<f32>; 3], cells: FieldDims, c: [i32; 3]) -> [f32; 3] {
    let n = [cells.x as i32, cells.y as i32, cells.z as i32];
    let q: [u32; 3] = std::array::from_fn(|a| c[a].clamp(0, n[a] - 1) as u32);
    std::array::from_fn(|a| {
        let d = face_dims(cells, a);
        let mut hi = q;
        hi[a] += 1;
        0.5 * (faces[a][index(d, q[0], q[1], q[2])] + faces[a][index(d, hi[0], hi[1], hi[2])])
    })
}

/// Mirrors `curl.wgsl`: ωx, ωy, ωz and |ω| per cell.
pub fn cpu_curl(faces: &[Vec<f32>; 3], cells: FieldDims, inv_dx: f32) -> [Vec<f32>; 4] {
    let mut out: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0; cells.voxel_count()]);
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                let c = [i as i32, j as i32, k as i32];
                let diff = |a: usize| {
                    let mut p = c;
                    p[a] += 1;
                    let mut m = c;
                    m[a] -= 1;
                    let (vp, vm) = (centre_velocity(faces, cells, p), centre_velocity(faces, cells, m));
                    [vp[0] - vm[0], vp[1] - vm[1], vp[2] - vm[2]]
                };
                let (ddx, ddy, ddz) = (diff(0), diff(1), diff(2));
                let s = 0.5 * inv_dx;
                let w = [
                    s * (ddy[2] - ddz[1]),
                    s * (ddz[0] - ddx[2]),
                    s * (ddx[1] - ddy[0]),
                ];
                let at = index(cells, i, j, k);
                out[0][at] = w[0];
                out[1][at] = w[1];
                out[2][at] = w[2];
                out[3][at] = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
            }
        }
    }
    out
}
```

Create `crates/elements-ember/tests/vorticity.rs`:

```rust
mod common;

use common::*;
use elements_core::gpu::{ComputeBatch, FieldDims, FieldPool, PipelineCache};
use elements_ember::kernels::{StepConstants, Uniforms, confine, curl};

const CELLS: FieldDims = FieldDims { x: 8, y: 6, z: 5 };

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Mirrors `force` in `confine.wgsl`, at cell `c` (inside the domain).
fn force(omega: &[Vec<f32>; 4], c: [i32; 3], inv_dx: f32, confinement: f32) -> [f32; 3] {
    let n = [CELLS.x as i32, CELLS.y as i32, CELLS.z as i32];
    let mag = |p: [i32; 3]| {
        let q: [u32; 3] = std::array::from_fn(|a| p[a].clamp(0, n[a] - 1) as u32);
        omega[3][index(CELLS, q[0], q[1], q[2])]
    };
    let s = 0.5 * inv_dx;
    let g: [f32; 3] = std::array::from_fn(|a| {
        let mut p = c;
        p[a] += 1;
        let mut m = c;
        m[a] -= 1;
        s * (mag(p) - mag(m))
    });
    let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt() + 1e-6;
    let normal = [g[0] / len, g[1] / len, g[2] / len];
    let at = index(CELLS, c[0] as u32, c[1] as u32, c[2] as u32);
    let w = [omega[0][at], omega[1][at], omega[2][at]];
    cross(normal, w).map(|v| confinement * v)
}

#[test]
fn curl_matches_the_cpu_reference() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = StepConstants::new(CELLS, 0.1, 0.125);
    let faces = walled_velocity_pattern(CELLS);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let omega: [_; 4] = std::array::from_fn(|_| pool.acquire_zeroed(&gpu, &mut cache, CELLS).unwrap());
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    curl(&gpu, &mut cache, &mut batch, &u, &velocity, [&omega[0], &omega[1], &omega[2], &omega[3]]).unwrap();
    batch.submit(&gpu).unwrap();
    let want = cpu_curl(&faces, CELLS, 1.0 / c.dx);
    for (n, field) in omega.iter().enumerate() {
        assert_close(&field.read_back(&gpu).unwrap(), &want[n], 1e-4, &format!("omega {n}"));
    }
}

/// Spec §4.4: u += h·ε·dx·(N × ω), averaged from the two cells each face
/// separates. Wall faces are left alone.
#[test]
fn confinement_matches_the_cpu_reference() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = StepConstants {
        vorticity: 3.0,
        ..StepConstants::new(CELLS, 0.1, 0.125)
    };
    let faces = walled_velocity_pattern(CELLS);
    let omega_values = cpu_curl(&faces, CELLS, 1.0 / c.dx);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &faces);
    let omega: [_; 4] = std::array::from_fn(|n| upload(&gpu, &mut pool, CELLS, &omega_values[n]));
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    confine(&gpu, &mut cache, &mut batch, &u, &velocity, [&omega[0], &omega[1], &omega[2], &omega[3]]).unwrap();
    batch.submit(&gpu).unwrap();

    let got = read_staggered(&gpu, &velocity);
    let confinement = c.vorticity * c.dx;
    for a in 0..3 {
        let d = face_dims(CELLS, a);
        let n = [CELLS.x, CELLS.y, CELLS.z][a];
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let ijk = [i, j, k];
                    let at = index(d, i, j, k);
                    if is_wall(CELLS, a, ijk[a]) {
                        assert_eq!(got[a][at], faces[a][at], "wall face {a} {ijk:?}");
                        continue;
                    }
                    let p = [i as i32, j as i32, k as i32];
                    let mut f = 0.0;
                    let mut count = 0.0;
                    if ijk[a] > 0 {
                        let mut below = p;
                        below[a] -= 1;
                        f += force(&omega_values, below, 1.0 / c.dx, confinement)[a];
                        count += 1.0;
                    }
                    if ijk[a] < n {
                        f += force(&omega_values, p, 1.0 / c.dx, confinement)[a];
                        count += 1.0;
                    }
                    let want = faces[a][at] + c.h * f / count;
                    assert!(
                        (got[a][at] - want).abs() <= 1e-4,
                        "face {a} {ijk:?}: {} vs {want}",
                        got[a][at]
                    );
                }
            }
        }
    }
}
```

In `tests/scenes.rs`, add:

```rust
/// Spec §4.4: confinement strengthens a plume's vorticity. Twenty frames of
/// the 16³ plume with ε = 4 end with clearly more total |ω| than with ε = 0.
#[test]
fn confinement_strengthens_a_plumes_vorticity() {
    fn total_vorticity(vorticity: f32) -> f64 {
        let gpu = gpu();
        let mut pool = FieldPool::new();
        let mut cache = PipelineCache::new();
        let cells = FieldDims::new(16, 16, 16);
        let dx = 2.0 / 16.0;
        let density_source = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
        let temperature_source = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
        let sphere = Sphere {
            center: [1.0, 1.0, 0.4],
            radius: 0.3,
            density_rate: 1.0,
            temperature_rate: 2.0,
        };
        fill_sphere(&gpu, &mut cache, &density_source, &temperature_source, &sphere, dx).unwrap();
        let sources = Sources {
            density: &density_source,
            temperature: &temperature_source,
        };
        let constants = StepConstants {
            beta: 1.0,
            vorticity,
            ..StepConstants::new(cells, 1.0 / 24.0, dx)
        };
        let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
        for _ in 0..20 {
            substep(&gpu, &mut cache, &mut pool, &mut state, sources, &constants, 160).unwrap();
        }
        let omega = cpu_curl(&state.read_velocity(&gpu).unwrap(), cells, 1.0 / dx);
        omega[3].iter().map(|&v| f64::from(v)).sum()
    }
    let without = total_vorticity(0.0);
    let with = total_vorticity(4.0);
    assert!(with >= 1.05 * without, "with {with}, without {without}");
}
```

In `tests/solver.rs`'s parameter test, replace `assert!(rejected(serde_json::json!({ "vorticity": 1.0 })));` with:

```rust
    assert!(!rejected(serde_json::json!({ "vorticity": 1.0 })));
    assert!(rejected(serde_json::json!({ "vorticity": -1.0 })));
    assert!(rejected(serde_json::json!({ "vorticity": 1e39 })));
```

Run: `cargo nextest run -p elements-ember --test vorticity`
Expected: compile error, `curl`, `confine` and `StepConstants::vorticity` not found.

- [ ] **Step 2: Extend the uniform**

`Params` in `common.wgsl` gains, after `decay`:

```wgsl
    confinement: f32,    // ε·dx, the vorticity confinement strength scaled to this grid
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
```

`KernelParams` gains `confinement: f32, _pad: [u32; 3]` after `decay` (64 bytes now; update its doc comment). `Uniforms::new`'s `make` fills `confinement: c.vorticity * c.dx, _pad: [0; 3]`. `StepConstants` gains

```rust
    /// Vorticity confinement strength ε, 1/s; 0 skips the stage (spec §4.4).
    pub vorticity: f32,
```

with `vorticity: 0.0` in `new`.

- [ ] **Step 3: Write the kernels**

Create `crates/elements-ember/src/kernels/shaders/curl.wgsl`:

```wgsl
// Vorticity confinement, part 1 (spec §4.4): ω = ∇ × u at cell centres, and
// |ω|. Face velocities are averaged to cell centres on the fly. Differences
// are central, with neighbours clamped into the domain at its edge.

@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var omega_x: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var omega_y: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var omega_z: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var omega_mag: texture_storage_3d<r32float, write>;
@group(0) @binding(7) var<uniform> params: Params;

// Velocity at the centre of cell `c`, clamped into the domain.
fn centre_velocity(c: vec3<i32>) -> vec3<f32> {
    let q = clamp(c, vec3<i32>(0), vec3<i32>(params.dims) - vec3<i32>(1));
    return 0.5 * vec3<f32>(
        textureLoad(vel_x, q, 0).x + textureLoad(vel_x, q + vec3<i32>(1, 0, 0), 0).x,
        textureLoad(vel_y, q, 0).x + textureLoad(vel_y, q + vec3<i32>(0, 1, 0), 0).x,
        textureLoad(vel_z, q, 0).x + textureLoad(vel_z, q + vec3<i32>(0, 0, 1), 0).x,
    );
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    let ddx = centre_velocity(c + vec3<i32>(1, 0, 0)) - centre_velocity(c - vec3<i32>(1, 0, 0));
    let ddy = centre_velocity(c + vec3<i32>(0, 1, 0)) - centre_velocity(c - vec3<i32>(0, 1, 0));
    let ddz = centre_velocity(c + vec3<i32>(0, 0, 1)) - centre_velocity(c - vec3<i32>(0, 0, 1));
    let w = 0.5 * params.inv_dx * vec3<f32>(ddy.z - ddz.y, ddz.x - ddx.z, ddx.y - ddy.x);
    textureStore(omega_x, c, vec4<f32>(w.x, 0.0, 0.0, 0.0));
    textureStore(omega_y, c, vec4<f32>(w.y, 0.0, 0.0, 0.0));
    textureStore(omega_z, c, vec4<f32>(w.z, 0.0, 0.0, 0.0));
    textureStore(omega_mag, c, vec4<f32>(length(w), 0.0, 0.0, 0.0));
}
```

Check that the CPU mirror's `sqrt(x² + y² + z²)` agrees with WGSL's `length` within the test's 1e-4 tolerance. It does in practice. If it does not, report it rather than loosening the tolerance.

Create `crates/elements-ember/src/kernels/shaders/confine.wgsl`:

```wgsl
// Vorticity confinement, part 2 (spec §4.4, Fedkiw et al. 2001): on one
// axis's faces, u += h·f, where f = ε·dx·(N × ω) and N = ∇|ω| / |∇|ω||,
// averaged from the cells each face separates. Wall faces are left alone.

@group(0) @binding(0) var face: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var omega_x: texture_3d<f32>;
@group(0) @binding(2) var omega_y: texture_3d<f32>;
@group(0) @binding(3) var omega_z: texture_3d<f32>;
@group(0) @binding(4) var omega_mag: texture_3d<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn magnitude(c: vec3<i32>) -> f32 {
    let q = clamp(c, vec3<i32>(0), vec3<i32>(params.dims) - vec3<i32>(1));
    return textureLoad(omega_mag, q, 0).x;
}

// The confinement force at the centre of cell `c`, which is inside the domain.
fn force(c: vec3<i32>) -> vec3<f32> {
    let g = 0.5 * params.inv_dx * vec3<f32>(
        magnitude(c + vec3<i32>(1, 0, 0)) - magnitude(c - vec3<i32>(1, 0, 0)),
        magnitude(c + vec3<i32>(0, 1, 0)) - magnitude(c - vec3<i32>(0, 1, 0)),
        magnitude(c + vec3<i32>(0, 0, 1)) - magnitude(c - vec3<i32>(0, 0, 1)),
    );
    let n = g / (length(g) + 1e-6);
    let w = vec3<f32>(
        textureLoad(omega_x, c, 0).x,
        textureLoad(omega_y, c, 0).x,
        textureLoad(omega_z, c, 0).x,
    );
    return params.confinement * cross(n, w);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = params.axis;
    if (any(gid >= grid_dims(axis))) {
        return;
    }
    let i = gid[axis];
    if (is_wall(axis, i)) {
        return;
    }
    let p = vec3<i32>(gid);
    var e = vec3<i32>(0);
    e[axis] = 1;
    // Face i lies between cells i − 1 and i. An open boundary face has only one.
    var f = 0.0;
    var count = 0.0;
    if (i > 0u) {
        f += force(p - e)[axis];
        count += 1.0;
    }
    if (i < params.dims[axis]) {
        f += force(p)[axis];
        count += 1.0;
    }
    let u = textureLoad(face, p).x + params.h * f / count;
    textureStore(face, p, vec4<f32>(u, 0.0, 0.0, 0.0));
}
```

Create `crates/elements-ember/src/kernels/vorticity.rs`:

```rust
//! Vorticity confinement (spec §4.4), between buoyancy and velocity advection.

use elements_core::gpu::{Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField};

use super::{Bind, Uniforms, bind_group, expect_dims};

const CURL: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/curl.wgsl"),
);

const CONFINE: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/confine.wgsl"),
);

fn check(what: &str, u: &Uniforms, velocity: &StaggeredField, omega: [&Field; 4]) -> Result<(), GpuError> {
    if velocity.cells() != u.cells() {
        return Err(GpuError::Validation(format!(
            "{what}: velocity {:?}, domain {:?}",
            velocity.cells(),
            u.cells()
        )));
    }
    for field in omega {
        expect_dims(what, field, u.cells())?;
    }
    Ok(())
}

/// Record ω = ∇ × `velocity` into `omega`: its x, y and z components, then
/// |ω|, all at cell centres.
pub fn curl(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    omega: [&Field; 4],
) -> Result<(), GpuError> {
    check("curl", u, velocity, omega)?;
    let pipeline = cache.get_or_create(gpu, "ember.curl", CURL, "main")?;
    let group = bind_group(
        gpu,
        &pipeline,
        &[
            Bind::Tex(velocity.face(Axis::X)),
            Bind::Tex(velocity.face(Axis::Y)),
            Bind::Tex(velocity.face(Axis::Z)),
            Bind::Tex(omega[0]),
            Bind::Tex(omega[1]),
            Bind::Tex(omega[2]),
            Bind::Tex(omega[3]),
            Bind::Buf(u.any()),
        ],
    )?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}

/// Record u += h·ε·dx·(N × ω) on every non-wall face of `velocity`, from the
/// `omega` that `curl` wrote.
pub fn confine(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    omega: [&Field; 4],
) -> Result<(), GpuError> {
    check("confine", u, velocity, omega)?;
    let pipeline = cache.get_or_create(gpu, "ember.confine", CONFINE, "main")?;
    for axis in Axis::ALL {
        let face = velocity.face(axis);
        let group = bind_group(
            gpu,
            &pipeline,
            &[
                Bind::Tex(face),
                Bind::Tex(omega[0]),
                Bind::Tex(omega[1]),
                Bind::Tex(omega[2]),
                Bind::Tex(omega[3]),
                Bind::Buf(u.axis(axis)),
            ],
        )?;
        batch.dispatch(&pipeline, &group, face.dims());
    }
    Ok(())
}
```

In `kernels/mod.rs`: `mod vorticity;` and `pub use vorticity::{confine, curl};`.

- [ ] **Step 4: Run the stage in the solver**

`Substep` gains `vorticity: bool`, set to `constants.vorticity > 0.0` in `new`. In `pre_projection`, between `kernels::buoyancy(…)?;` and the velocity advection, add:

```rust
        if self.vorticity {
            self.confine_vorticity(gpu, cache, pool, state)?;
        }
```

and the method:

```rust
    /// Vorticity confinement onto the velocity, through four pooled scratch
    /// fields for ω and |ω|.
    fn confine_vorticity(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        state: &SolverState,
    ) -> Result<(), GpuError> {
        let cells = self.uniforms.cells();
        let mut omega = Vec::with_capacity(4);
        for _ in 0..4 {
            match pool.acquire(gpu, cells, FieldFormat::R32Float) {
                Ok(field) => omega.push(field),
                Err(e) => {
                    self.retired.extend(omega);
                    return Err(e);
                }
            }
        }
        let refs = [&omega[0], &omega[1], &omega[2], &omega[3]];
        let u = &self.uniforms;
        let recorded = kernels::curl(gpu, cache, &mut self.batch, u, &state.velocity, refs)
            .and_then(|()| kernels::confine(gpu, cache, &mut self.batch, u, &state.velocity, refs));
        // The batch may reference them whether or not recording finished.
        self.retired.extend(omega);
        recorded
    }
```

`SolverParams` gains `pub vorticity: f32` (doc: `/// Vorticity confinement ε, 1/s; 0 turns it off (spec §4.4).`, default 0.0), and `step_constants` passes it on. In `build`, extend the dissipation check to cover it: `params::finite(KIND, "vorticity and dissipation", &[params.vorticity, params.density_dissipation, params.temperature_dissipation])?`, and reject a negative value with `"vorticity and dissipation rates must be at least 0"`. The module doc of `solver.rs` becomes: "Per substep: emit, buoyancy, vorticity confinement, advect velocity, project, and advect scalars with dissipation."

- [ ] **Step 5: Run and prove**

Run: `cargo nextest run -p elements-ember`
Expected: all pass.

| Test | Mutation | Expected |
|---|---|---|
| `curl_matches_the_cpu_reference` | in `curl.wgsl`, `ddz.x - ddx.z` → `ddx.z - ddz.x` | FAIL: "omega 1" |
| `confinement_matches_the_cpu_reference` | in `confine.wgsl`, `cross(n, w)` → `cross(w, n)` | FAIL |
| `confinement_strengthens_a_plumes_vorticity` | the same `cross(w, n)` flip | FAIL: `with` below 1.05 × `without` |

If the scene test's margin is not met without any mutation, stop and report both totals. Do not change ε or the 1.05 factor on your own. Record, restore.

- [ ] **Step 6: Run the gate and commit**

Run: `just check`
Expected: PASS.

```bash
git add crates/elements-ember docs/superpowers/specs/2026-09-22-ember-solver-2b1-design.md
git commit   # subject: "Add vorticity confinement"
```

Also, in `docs/superpowers/specs/2026-09-22-ember-solver-2b1-design.md` §7, change the "Vorticity" row's assertion to "`curl` and `confine` match the CPU; confinement strengthens a 16³ plume's total |ω| by at least 5%", and include that file in the commit.

The body explains why: semi-Lagrangian and MacCormack both damp small swirls, and confinement puts back the detail users expect from a smoke solver. Mantaflow has it too, so the benchmark can match ε. Also note the §7 deviation (the ε = 0 check is replaced by the physics test).

---

## Task 7: CFL substepping, the new parameters, and quality presets

Spec §2, §3 and §6 (the preset table and the provisional preview cap).

**Files:**
- Create: `crates/elements-ember/src/cfl.rs`
- Modify: `crates/elements-ember/src/lib.rs` (`pub mod cfl;`)
- Modify: `crates/elements-core/src/graph/node.rs` (`EvalStats::cfl_clamped`, `EvalCtx::count_cfl_clamped`, `NodeError::SolverDiverged`)
- Modify: `crates/elements-ember/src/solver.rs` (`Quality`, `DocParams`, `resolve_params`, `SolverParams`, `step`)
- Modify: `crates/elements-ember/src/bench.rs`, `crates/elements-ember/examples/speed_gate.rs`
- Modify: every exhaustive `match` on `NodeError` outside core (find them with `grep -rn "NodeError::" crates/elementsd crates/elements-cli`)
- Test: `crates/elements-ember/tests/cfl.rs` (create), `tests/solver.rs`

**Interfaces:**
- Consumes: `ReduceTarget`, `reduce` (Task 1); all `SolverParams` fields from Tasks 2, 5 and 6.
- Produces:
  - `cfl::SubstepPlan { count: u32, clamped: bool }`, `cfl::plan_substeps(max_speed: f32, dt: f64, dx: f32, cfl: f32, max_substeps: u32) -> Option<SubstepPlan>`, `cfl::measure_speed(gpu, cache, velocity: &StaggeredField) -> Result<f32, GpuError>`
  - `solver::Quality { Preview, Final }` with `Quality::params(self) -> SolverParams`
  - `solver::resolve_params(params: &serde_json::Value) -> Result<SolverParams, DocError>`
  - `SolverParams` fields: `max_substeps, cfl, pressure_iterations, advection, vorticity, density_dissipation, temperature_dissipation, buoyancy_density, buoyancy_temperature, boundaries`. It is `Serialize` only, and `substeps` is gone from it.
  - Core: `EvalStats::cfl_clamped: u32`, `EvalCtx::count_cfl_clamped(&mut self)`, `NodeError::SolverDiverged { node: NodeId }`
  - `bench::Scene::with_max_substeps(self, n: u32) -> Self`

- [ ] **Step 1: Write the failing tests**

Create `crates/elements-ember/tests/cfl.rs`:

```rust
mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldPool, PipelineCache};
use elements_core::graph::{Document, NodeError, StateStore, Time};
use elements_ember::cfl::{SubstepPlan, measure_speed, plan_substeps};

#[test]
fn substeps_are_the_ceiling_of_cells_travelled_over_cfl() {
    // dt = 0.1 s, dx = 0.1 m: 2.5 m/s travels 2.5 cells in a frame.
    let plan = |speed, cfl, cap| plan_substeps(speed, 0.1, 0.1, cfl, cap);
    let ok = |count, clamped| Some(SubstepPlan { count, clamped });
    assert_eq!(plan(2.5, 1.0, 8), ok(3, false));
    assert_eq!(plan(2.5, 2.5, 8), ok(1, false));
    assert_eq!(plan(0.0, 1.0, 8), ok(1, false));
    assert_eq!(plan(2.5, 1.0, 2), ok(2, true));
    assert_eq!(plan(f32::NAN, 1.0, 8), None);
    assert_eq!(plan(f32::INFINITY, 1.0, 8), None);
}

#[test]
fn the_measured_speed_is_the_fastest_face_and_a_nan_is_not_hidden() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(8, 6, 5);
    let mut faces = velocity_pattern(cells); // about [-0.6, 0.6]
    let last = faces[2].len() - 1;
    faces[2][last] = -3.0;
    let velocity = upload_staggered(&gpu, &mut pool, cells, &faces);
    assert_eq!(measure_speed(&gpu, &mut cache, &velocity).unwrap(), 3.0);

    faces[1][7] = f32::NAN;
    let velocity = upload_staggered(&gpu, &mut pool, cells, &faces);
    let speed = measure_speed(&gpu, &mut cache, &velocity).unwrap();
    assert!(!speed.is_finite(), "a NaN face read as {speed}");
}

/// A 8³ plume document with the given solver parameters (a JSON object body).
fn plume(emitter_temperature_rate: &str, solver: &str) -> String {
    format!(
        r#"{{
  "version": 3, "dims": [8, 8, 8], "fps": 24.0, "domain_size": 2.0,
  "nodes": [
    {{ "id": 0, "kind": "ember.sphere_emitter",
       "params": {{ "center": [1.0, 1.0, 0.5], "radius": 0.4,
                   "density_rate": 1.0, "temperature_rate": {emitter_temperature_rate} }} }},
    {{ "id": 1, "kind": "ember.smoke_solver", "params": {solver} }},
    {{ "id": 2, "kind": "core.output", "params": {{}} }}
  ],
  "edges": [
    {{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }},
    {{ "from_node": 0, "from_index": 1, "to_node": 1, "to_index": 1 }},
    {{ "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 0 }}
  ],
  "output": 2
}}"#
    )
}

/// Spec §3: a frame whose CFL count exceeds the cap still runs, and the
/// evaluation counts it. Frame 1 starts still. After it the plume moves at
/// a few cm/s (w ≈ h·β·T ≈ 0.035 m/s), and with cfl 1e-4 and dx = 0.25 m
/// that wants dozens of substeps.
#[test]
fn a_frame_over_the_substep_cap_is_counted() {
    let doc = plume("20.0", r#"{ "cfl": 0.0001, "max_substeps": 1, "pressure_iterations": 4 }"#);
    let (graph, dims) = Document::from_json(&doc)
        .unwrap()
        .into_graph(&elements_ember::registry())
        .unwrap();
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let mut counts = Vec::new();
    for frame in 1..=3 {
        let evaluated = graph
            .eval_frame(&gpu, &mut pool, &mut pipelines, &mut state, Time::at(frame, 1, 24.0), dims)
            .unwrap();
        counts.push(evaluated.stats.cfl_clamped);
        evaluated.value.release_to(&mut pool);
    }
    state.clear(&mut pool);
    assert_eq!(counts, vec![0, 1, 1]);
}

/// Spec §3: once the velocity is no longer finite, the solver stops with
/// `SolverDiverged` instead of stepping NaNs, and every texture goes back to
/// the pool. An absurd (but finite, so valid) emission rate makes it happen.
#[test]
fn a_diverging_simulation_says_so_and_returns_every_field() {
    let doc = plume("3e38", r#"{ "pressure_iterations": 4 }"#);
    let (graph, dims) = Document::from_json(&doc)
        .unwrap()
        .into_graph(&elements_ember::registry())
        .unwrap();
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let mut failure = None;
    for frame in 1..=20 {
        match graph.eval_frame(&gpu, &mut pool, &mut pipelines, &mut state, Time::at(frame, 1, 24.0), dims) {
            Ok(evaluated) => evaluated.value.release_to(&mut pool),
            Err(e) => {
                failure = Some((frame, e));
                break;
            }
        }
    }
    let (frame, err) = failure.expect("the simulation must diverge within 20 frames");
    assert!(matches!(err, NodeError::SolverDiverged { .. }), "frame {frame}: {err:?}");
    state.clear(&mut pool);
    assert_eq!(
        pool.pooled_count() as u64,
        pool.allocation_count(),
        "every allocated texture must be back in the pool"
    );
}
```

In `tests/solver.rs`, extend the parameter test and add a preset test (import `resolve_params` and `SolverParams` from `elements_ember::solver`):

```rust
    assert!(!rejected(serde_json::json!({ "quality": "final" })));
    assert!(rejected(serde_json::json!({ "quality": "ultra" })));
    assert!(!rejected(serde_json::json!({ "max_substeps": 16, "cfl": 10.0 })));
    assert!(rejected(serde_json::json!({ "max_substeps": 17 })));
    assert!(rejected(serde_json::json!({ "substeps": 2, "max_substeps": 2 })));
    assert!(rejected(serde_json::json!({ "cfl": 0.0 })));
    assert!(rejected(serde_json::json!({ "cfl": 10.5 })));
```

```rust
/// Spec §6: a preset fills only the fields a document leaves unset, and
/// 2a's `substeps` still works as `max_substeps`.
#[test]
fn a_preset_fills_only_what_the_document_leaves_unset() {
    let p = resolve_params(&serde_json::json!({ "quality": "final", "pressure_iterations": 50 }))
        .unwrap();
    assert_eq!(p.pressure_iterations, 50, "an explicit field wins");
    assert_eq!(p.max_substeps, 8, "final's cap");
    let alias = resolve_params(&serde_json::json!({ "substeps": 3 })).unwrap();
    assert_eq!(alias.max_substeps, 3, "the 2a alias");
    assert_eq!(resolve_params(&serde_json::Value::Null).unwrap(), SolverParams::default());
}
```

Run: `cargo nextest run -p elements-ember --test cfl`
Expected: compile error, no module `cfl`.

- [ ] **Step 2: Core additions**

In `crates/elements-core/src/graph/node.rs`:

- `EvalStats` gains:

```rust
    /// Frames in which a solver wanted more CFL substeps than its cap
    /// allowed. It still ran them, at the cap.
    pub cfl_clamped: u32,
```

- `EvalCtx` gains:

```rust
    /// Count this evaluation as one where a solver's CFL substeps hit its cap.
    pub fn count_cfl_clamped(&mut self) {
        self.stats.cfl_clamped += 1;
    }
```

- `NodeError` gains:

```rust
    #[error("node {node:?} diverged: its velocity is no longer finite")]
    SolverDiverged { node: NodeId },
```

Then run `grep -rn "NodeError::" crates/elementsd crates/elements-cli`. For every exhaustive `match` on `NodeError`, add an arm for `SolverDiverged` that treats it like the other document and node errors: the daemon reports it and keeps running, since it is not a GPU fault. Build the workspace to catch any you missed: `cargo build --workspace --all-targets`.

- [ ] **Step 3: Write `cfl.rs`**

Create `crates/elements-ember/src/cfl.rs`:

```rust
//! CFL substepping (spec §3): how many substeps a frame needs, from the
//! fastest face velocity of the state entering it.

use elements_core::gpu::{
    Axis, ComputeBatch, GpuContext, GpuError, PipelineCache, ReduceOp, ReduceTarget,
    StaggeredField, reduce,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubstepPlan {
    /// Substeps to run this frame, 1..=max_substeps.
    pub count: u32,
    /// Whether CFL wanted more than `max_substeps`.
    pub clamped: bool,
}

/// n = clamp(ceil(max|u|·dt / (cfl·dx)), 1, max_substeps). `None` when
/// `max_speed` is not finite: the simulation has diverged.
pub fn plan_substeps(
    max_speed: f32,
    dt: f64,
    dx: f32,
    cfl: f32,
    max_substeps: u32,
) -> Option<SubstepPlan> {
    if !max_speed.is_finite() {
        return None;
    }
    let cells = f64::from(max_speed) * dt / (f64::from(cfl) * f64::from(dx));
    // `as` saturates, so an enormous but finite speed asks for u32::MAX and
    // the cap binds.
    let wanted = (cells.ceil() as u32).max(1);
    Some(SubstepPlan {
        count: wanted.min(max_substeps),
        clamped: wanted > max_substeps,
    })
}

/// The largest |u| over the three faces of `velocity`, read back to the
/// CPU; not finite if any face value is not finite.
///
/// WGSL's `max` may return either operand when one is NaN, so a max
/// reduction alone could hide a NaN. Each face's sum is reduced too, because
/// a sum does carry NaN and infinity through. A sum that overflows `f32`
/// also counts: velocities that large mean the simulation has diverged.
pub fn measure_speed(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    velocity: &StaggeredField,
) -> Result<f32, GpuError> {
    let target = ReduceTarget::new(gpu, 6)?;
    let mut batch = ComputeBatch::new();
    for (slot, axis) in (0u32..).zip(Axis::ALL) {
        let face = velocity.face(axis);
        reduce(gpu, cache, &mut batch, face, ReduceOp::MaxAbs, &target, slot)?;
        reduce(gpu, cache, &mut batch, face, ReduceOp::Sum, &target, slot + 3)?;
    }
    batch.submit(gpu)?;
    let values = target.read(gpu)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Ok(f32::NAN);
    }
    Ok(values[..3].iter().copied().fold(0.0, f32::max))
}
```

Add `pub mod cfl;` to `lib.rs`.

- [ ] **Step 4: Parameters and presets**

In `src/solver.rs`, replace `SolverParams`, its `Default` impl and `build` with the following. Keep `step_constants` as it is: it already reads every field it needs.

```rust
const MAX_SUBSTEPS: u32 = 16;
const MAX_PRESSURE_ITERATIONS: u32 = 1000;
const MAX_CFL: f32 = 10.0;

/// A quality preset (spec §6). It fills every field a document leaves unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    /// Interactive: 128³ within 100 ms a frame.
    #[default]
    Preview,
    /// Offline bakes: no time budget.
    Final,
}

impl Quality {
    /// The preset's full parameter set.
    pub fn params(self) -> SolverParams {
        let (pressure_iterations, max_substeps) = match self {
            // `max_substeps` is provisional until the preset sweep decides it
            // (spec §6, `docs/bench/presets.md`).
            Self::Preview => (160, 2),
            // 480 iterations: ratio 0.0076 in `docs/bench/iteration-sweep.md`.
            Self::Final => (480, 8),
        };
        SolverParams {
            max_substeps,
            cfl: 1.0,
            pressure_iterations,
            advection: Advection::MacCormack,
            vorticity: 0.0,
            density_dissipation: 0.0,
            temperature_dissipation: 0.0,
            buoyancy_density: 0.0,
            // Provisional until 2b-3 maps parameters to Mantaflow's.
            buoyancy_temperature: 1.0,
            boundaries: Boundaries::default(),
        }
    }
}

/// `ember.smoke_solver`'s parameters, resolved: every field has a value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SolverParams {
    /// Cap on CFL substeps per frame (spec §3).
    pub max_substeps: u32,
    /// Target maximum cells travelled per substep.
    pub cfl: f32,
    /// Red-black Gauss–Seidel iterations per substep.
    pub pressure_iterations: u32,
    /// How velocity and scalars are advected (spec §4.1).
    pub advection: Advection,
    /// Vorticity confinement ε, 1/s; 0 turns it off (spec §4.4).
    pub vorticity: f32,
    /// Exponential decay of density, 1/s (spec §4.5).
    pub density_dissipation: f32,
    /// Exponential decay of temperature, 1/s.
    pub temperature_dissipation: f32,
    /// α: downward acceleration per unit density, m/s².
    pub buoyancy_density: f32,
    /// β: upward acceleration per unit temperature, m/s².
    pub buoyancy_temperature: f32,
    /// Which domain faces are open; the rest are walls (spec §4.2).
    pub boundaries: Boundaries,
}

impl Default for SolverParams {
    fn default() -> Self {
        Quality::default().params()
    }
}

/// A document's parameters as written. Every field is optional, and
/// `resolve_params` fills the gaps from the preset.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DocParams {
    quality: Option<Quality>,
    /// 2a's name for `max_substeps`, still accepted.
    substeps: Option<u32>,
    max_substeps: Option<u32>,
    cfl: Option<f32>,
    pressure_iterations: Option<u32>,
    advection: Option<Advection>,
    vorticity: Option<f32>,
    density_dissipation: Option<f32>,
    temperature_dissipation: Option<f32>,
    buoyancy_density: Option<f32>,
    buoyancy_temperature: Option<f32>,
    boundaries: Option<Boundaries>,
}

/// Parse `ember.smoke_solver`'s parameters from an untrusted document, fill
/// unset fields from the preset, and validate the result (spec §2).
pub fn resolve_params(params: &serde_json::Value) -> Result<SolverParams, DocError> {
    // `deny_unknown_fields` rejects misspelt keys; an absent `params` is `null`.
    let doc: DocParams = if params.is_null() {
        params::parse(KIND, &serde_json::json!({}))?
    } else {
        params::parse(KIND, params)?
    };
    if doc.substeps.is_some() && doc.max_substeps.is_some() {
        return Err(params::bad(
            KIND,
            "set max_substeps or its alias substeps, not both",
        ));
    }
    let preset = doc.quality.unwrap_or_default().params();
    let p = SolverParams {
        max_substeps: doc
            .max_substeps
            .or(doc.substeps)
            .unwrap_or(preset.max_substeps),
        cfl: doc.cfl.unwrap_or(preset.cfl),
        pressure_iterations: doc.pressure_iterations.unwrap_or(preset.pressure_iterations),
        advection: doc.advection.unwrap_or(preset.advection),
        vorticity: doc.vorticity.unwrap_or(preset.vorticity),
        density_dissipation: doc.density_dissipation.unwrap_or(preset.density_dissipation),
        temperature_dissipation: doc
            .temperature_dissipation
            .unwrap_or(preset.temperature_dissipation),
        buoyancy_density: doc.buoyancy_density.unwrap_or(preset.buoyancy_density),
        buoyancy_temperature: doc.buoyancy_temperature.unwrap_or(preset.buoyancy_temperature),
        boundaries: doc.boundaries.unwrap_or(preset.boundaries),
    };
    validate(&p)?;
    Ok(p)
}

fn validate(p: &SolverParams) -> Result<(), DocError> {
    if !(1..=MAX_SUBSTEPS).contains(&p.max_substeps) {
        return Err(params::bad(
            KIND,
            format!("max_substeps must be 1..={MAX_SUBSTEPS}, got {}", p.max_substeps),
        ));
    }
    if !(1..=MAX_PRESSURE_ITERATIONS).contains(&p.pressure_iterations) {
        return Err(params::bad(
            KIND,
            format!(
                "pressure_iterations must be 1..={MAX_PRESSURE_ITERATIONS}, got {}",
                p.pressure_iterations
            ),
        ));
    }
    params::finite(KIND, "cfl", &[p.cfl])?;
    if !(p.cfl > 0.0 && p.cfl <= MAX_CFL) {
        return Err(params::bad(
            KIND,
            format!("cfl must be in (0, {MAX_CFL}], got {}", p.cfl),
        ));
    }
    params::finite(KIND, "buoyancy", &[p.buoyancy_density, p.buoyancy_temperature])?;
    let rates = [p.vorticity, p.density_dissipation, p.temperature_dissipation];
    params::finite(KIND, "vorticity and dissipation", &rates)?;
    if rates.iter().any(|&r| r < 0.0) {
        return Err(params::bad(
            KIND,
            "vorticity and dissipation rates must be at least 0",
        ));
    }
    Ok(())
}

pub(crate) fn build(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    Ok(Box::new(SmokeSolver {
        params: resolve_params(params)?,
    }))
}
```

- [ ] **Step 5: Choose substeps each frame in `SmokeSolver::step`**

Replace everything in `step` after `let sources = Sources { … };` with:

```rust
        // Spec §3: one measurement per frame, from the entering state, so
        // the count is deterministic however the frame is reached.
        let dt = ctx.time().dt;
        let dx = ctx.voxel_size();
        let speed = ctx.with_gpu(|gpu, cache| cfl::measure_speed(gpu, cache, &state.velocity))?;
        let plan = cfl::plan_substeps(
            speed,
            dt,
            dx,
            self.params.cfl,
            self.params.max_substeps,
        )
        .ok_or(NodeError::SolverDiverged { node })?;
        if plan.clamped {
            ctx.count_cfl_clamped();
        }
        let constants =
            self.params
                .step_constants(ctx.dims(), (dt / f64::from(plan.count)) as f32, dx);
        let iterations = self.params.pressure_iterations;
        ctx.with_gpu_pool(|gpu, cache, pool| {
            for _ in 0..plan.count {
                substep(gpu, cache, pool, state, sources, &constants, iterations)?;
            }
            Ok(())
        })
```

(Import `crate::cfl`. The module doc gains: "Each frame first measures the fastest face and picks its substep count by CFL.")

- [ ] **Step 6: Update the bench scene and the gate example**

In `src/bench.rs`, `Scene::plume`'s solver becomes `SolverParams { pressure_iterations: 160, buoyancy_density: 0.0, buoyancy_temperature: 1.0, ..SolverParams::default() }`. Add:

```rust
    pub fn with_max_substeps(mut self, n: u32) -> Self {
        self.solver.max_substeps = n;
        self
    }
```

In `examples/speed_gate.rs`, rename `.solver.substeps` to `.solver.max_substeps` (two places), write `max_substeps {substeps}` in the report's scene line, and add to the module doc: "Since piece 2b-1 the plume scene uses the preview preset (MacCormack, CFL substeps), so a rerun no longer reproduces 2a's table. `speed-gate.md` records the commit its numbers came from."

- [ ] **Step 7: Run and prove**

Run: `cargo nextest run --workspace`
Expected: all pass, including `frame_40_is_bit_identical_however_it_is_reached` (the CFL measurement is deterministic).

| Test | Mutation | Expected |
|---|---|---|
| `substeps_are_the_ceiling_of_cells_travelled_over_cfl` | `cells.ceil()` → `cells.floor()` | FAIL: 2 ≠ 3 |
| `the_measured_speed_is_the_fastest_face_and_a_nan_is_not_hidden` | delete the `if values.iter().any(…) { return Ok(f32::NAN); }` block | FAIL: "a NaN face read as 3" or similar |
| `a_frame_over_the_substep_cap_is_counted` | delete `ctx.count_cfl_clamped();` | FAIL: `[0, 0, 0]` |
| `a_diverging_simulation_says_so_and_returns_every_field` | in `plan_substeps`, delete the `is_finite` early return | FAIL: "must diverge within 20 frames" |
| `a_preset_fills_only_what_the_document_leaves_unset` | `pressure_iterations: preset.pressure_iterations` (ignore the document) | FAIL: 480 ≠ 50 |

Record, restore.

- [ ] **Step 8: Run the gate and commit**

Run: `just check`
Expected: PASS.

```bash
git add crates
git commit   # subject: "Choose substeps by CFL each frame, and add quality presets"
```

The body explains why. At 128³, 1 m/s is about 2.7 cells a frame, so a fixed substep count either wastes time or loses accuracy (risk c). The count is measured once per frame from the entering state, which keeps frames deterministic (user decision). Divergence becomes an error instead of NaNs spreading, and presets give Blender two sensible starting points. Say that preview's cap of 2 is provisional until Task 8.

---

## Task 8: Measure the preview preset, and bring the docs up to date

Spec §6 (the pre-registered rule). This task contains a **human decision**: the sweep's result goes to the user, and the user confirms the preview cap before it is committed.

**Files:**
- Create: `crates/elements-ember/examples/common/mod.rs`, `crates/elements-ember/examples/presets.rs`
- Modify: `crates/elements-ember/examples/speed_gate.rs` (use `common`)
- Modify: `crates/elements-ember/src/bench.rs` (`PresetRow`, the rule)
- Modify: `justfile` (`bench-presets`)
- Create: `docs/bench/presets.md` (written by the example)
- Modify: `crates/elements-ember/src/solver.rs` (preview's `max_substeps`, after the decision)
- Modify: `CLAUDE.md`, `docs/superpowers/specs/2026-09-21-ember-solver-design.md` (status), `docs/superpowers/specs/2026-09-22-ember-solver-2b1-design.md` (status), `README.md` (if its status names 2b as next)
- Test: `crates/elements-ember/tests/bench.rs`

**Interfaces:**
- Consumes: `Scene::with_max_substeps`, `Scene::with_iterations`, `EvalStats::cfl_clamped`, `Quality::params`.
- Produces: `bench::PRESET_FRAME_MS: f64`, `bench::PresetRow { max_substeps: u32, frame_ms_median: f64 }` with `passes()`, `bench::preview_substeps_verdict(rows: &[PresetRow]) -> Option<u32>`.

- [ ] **Step 1: Write the failing rule tests**

In `tests/bench.rs` (import `PresetRow`, `preview_substeps_verdict`):

```rust
fn preset(max_substeps: u32, frame_ms_median: f64) -> PresetRow {
    PresetRow {
        max_substeps,
        frame_ms_median,
    }
}

/// Spec §6: preview takes the largest cap whose median full frame fits.
#[test]
fn the_preview_cap_is_the_largest_that_fits() {
    let rows = [preset(1, 40.0), preset(2, 80.0), preset(3, 120.0), preset(4, 160.0)];
    assert_eq!(preview_substeps_verdict(&rows), Some(2));
}

#[test]
fn the_preview_frame_limit_is_inclusive() {
    assert!(preset(3, 100.0).passes());
}

#[test]
fn no_cap_fits_when_even_one_substep_is_too_slow() {
    assert_eq!(preview_substeps_verdict(&[preset(1, 101.0)]), None);
}
```

Run: `cargo nextest run -p elements-ember --test bench`
Expected: compile error, `PresetRow` not found.

- [ ] **Step 2: Write the rule**

In `src/bench.rs`:

```rust
/// A preview frame at 128³, with the CFL measurement and every substep, must
/// take at most this long (spec §6).
pub const PRESET_FRAME_MS: f64 = 100.0;

/// One row of the preview-preset sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PresetRow {
    pub max_substeps: u32,
    /// Median over three runs of each run's median frame time.
    pub frame_ms_median: f64,
}

impl PresetRow {
    /// Spec §6's per-row rule. `preview_substeps_verdict` uses it too.
    pub fn passes(&self) -> bool {
        self.frame_ms_median <= PRESET_FRAME_MS
    }
}

/// Spec §6: the largest `max_substeps` whose median full frame fits. `None`
/// means even one substep does not, and preview falls back to
/// semi-Lagrangian advection before the sweep runs again.
pub fn preview_substeps_verdict(rows: &[PresetRow]) -> Option<u32> {
    rows.iter()
        .filter(|r| r.passes())
        .map(|r| r.max_substeps)
        .max()
}
```

Run: `cargo nextest run -p elements-ember --test bench`
Expected: pass. Mutations: `<` for `<=` (FAIL: inclusive), and `.min()` for `.max()` (FAIL: `Some(1)`). Record, restore.

- [ ] **Step 3: Share the example helpers**

Create `crates/elements-ember/examples/common/mod.rs` and move `median`, `shell` and `shell_checked` into it unchanged (make them `pub`), plus the commit label from `speed_gate.rs`'s `main`:

```rust
/// The short commit hash, with `-dirty` when the tree has changes, for a
/// results table's header.
pub fn commit_label() -> String {
    match shell_checked("git", &["rev-parse", "--short", "HEAD"]) {
        Some(hash) => match shell_checked("git", &["status", "--porcelain"]) {
            Some(status) if !status.is_empty() => format!("{hash}-dirty"),
            Some(_) => hash,
            // `git status` failed to run: don't claim a clean tree we didn't verify.
            None => format!("{hash} (dirty status unknown)"),
        },
        None => "unknown".to_owned(),
    }
}
```

In `speed_gate.rs`, add `mod common;` and `use common::{commit_label, median, shell};`, delete the moved functions, and replace the inline commit logic with `let commit = commit_label();`. Cargo only treats `examples/*.rs` and `examples/*/main.rs` as examples, so `examples/common/mod.rs` is a module, not an example.

- [ ] **Step 4: Write the sweep**

Create `crates/elements-ember/examples/presets.rs`:

```rust
//! The preview preset's substep cap (2b-1 spec §6). Run with
//! `just bench-presets`.
//!
//! Writes `docs/bench/presets.md`, or `presets-semi-lagrangian.md` when run
//! with `PRESET_ADVECTION=semi_lagrangian` (the rule's fallback). The
//! decision line at its end is filled in by hand after the user decides;
//! this program only applies the rule.

mod common;

use std::error::Error;
use std::time::Instant;

use elements_core::gpu::{FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{NodeRegistry, StateStore, Time};
use elements_ember::bench::{PRESET_FRAME_MS, PresetRow, Scene, preview_substeps_verdict};
use elements_ember::kernels::Advection;

use common::{commit_label, median, shell};

const RESOLUTION: u32 = 128;
const CAPS: [u32; 4] = [1, 2, 3, 4];
const RUNS: usize = 3;
const WARMUP: u32 = 24;
const TIMED: u32 = 24;

type Res<T> = Result<T, Box<dyn Error>>;

/// One run of `scene`: each timed frame's full time in ms (`eval_frame`,
/// which includes the CFL measurement, plus a blocking wait), and how many
/// timed frames hit the substep cap.
fn run(gpu: &GpuContext, registry: &NodeRegistry, scene: &Scene) -> Res<(Vec<f64>, u32)> {
    let doc = scene.document();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(registry)?;
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let mut frames = Vec::new();
    let mut clamped = 0;
    let first = config.start_frame;
    for frame in first..first + WARMUP + TIMED {
        let time = Time::at(frame, first, config.fps);
        let start = Instant::now();
        let evaluated = graph.eval_frame(gpu, &mut pool, &mut pipelines, &mut state, time, dims)?;
        gpu.wait()?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        if frame >= first + WARMUP {
            frames.push(ms);
            clamped += evaluated.stats.cfl_clamped;
        }
        evaluated.value.release_to(&mut pool);
    }
    state.clear(&mut pool);
    Ok((frames, clamped))
}

fn main() -> Res<()> {
    let gpu = GpuContext::new_headless()?;
    let registry = elements_ember::registry();
    let (advection, file) = match std::env::var("PRESET_ADVECTION").as_deref() {
        Ok("semi_lagrangian") => (Advection::SemiLagrangian, "presets-semi-lagrangian.md"),
        _ => (Advection::MacCormack, "presets.md"),
    };
    let mut table = String::new();
    let mut rows = Vec::new();
    for cap in CAPS {
        let mut scene = Scene::plume(RESOLUTION)
            .with_iterations(160)
            .with_max_substeps(cap);
        scene.solver.cfl = 1.0;
        scene.solver.advection = advection;
        scene.solver.vorticity = 0.0;
        let mut medians = Vec::new();
        let mut all = Vec::new();
        let mut clamped = 0;
        for r in 0..RUNS {
            eprintln!("max_substeps = {cap}: run {} of {RUNS}", r + 1);
            let (frames, c) = run(&gpu, &registry, &scene)?;
            medians.push(median(&frames));
            all.extend(frames);
            clamped += c;
        }
        let row = PresetRow {
            max_substeps: cap,
            frame_ms_median: median(&medians),
        };
        let min = all.iter().copied().fold(f64::INFINITY, f64::min);
        let max = all.iter().copied().fold(0.0, f64::max);
        table.push_str(&format!(
            "| {cap} | {:.2} ({min:.2}–{max:.2}) | {clamped} of {} | {} |\n",
            row.frame_ms_median,
            TIMED as usize * RUNS,
            if row.passes() { "yes" } else { "no" },
        ));
        rows.push(row);
    }

    let verdict = match preview_substeps_verdict(&rows) {
        Some(n) => format!("**preview `max_substeps` = {n}**"),
        None => "**no cap fits**: rerun with `PRESET_ADVECTION=semi_lagrangian` (spec §6)".to_owned(),
    };
    let report = format!(
        "# Ember preview preset sweep (piece 2b-1)\n\n\
         - Machine: {cpu} ({adapter})\n\
         - OS: macOS {os}\n\
         - Ember commit: {commit}\n\
         - Date: {date}\n\
         - Scene: `plume`, {RESOLUTION}³, N = 160, cfl 1.0, advection {advection:?}, \
         vorticity 0. Frames {first}–{last} timed after {WARMUP} warm-up frames, each as \
         `eval_frame` (the CFL measurement and every substep) plus a blocking wait; median of \
         {RUNS} runs' medians. The min–max range is pooled over all timed frames of all runs.\n\n\
         | max_substeps | frame ms (median, min–max) | frames at the cap | pass |\n\
         |---|---|---|---|\n\
         {table}\n\
         Pre-registered rule (2b-1 spec §6): preview's `max_substeps` is the largest cap whose \
         median full frame is at most {PRESET_FRAME_MS} ms. If even 1 fails, preview falls back \
         to semi-Lagrangian advection and the sweep runs again.\n\n\
         Rule applied: {verdict}.\n\n\
         Decision (recorded by the user): _pending_\n",
        cpu = shell("sysctl", &["-n", "machdep.cpu.brand_string"]),
        adapter = gpu.adapter_name(),
        os = shell("sw_vers", &["-productVersion"]),
        commit = commit_label(),
        date = shell("date", &["-u", "+%Y-%m-%d"]),
        first = 1 + WARMUP,
        last = WARMUP + TIMED,
    );
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/bench");
    std::fs::create_dir_all(dir)?;
    std::fs::write(format!("{dir}/{file}"), &report)?;
    println!("{report}");
    Ok(())
}
```

In `justfile`, after `bench-sweep`:

```
# The preview preset's substep cap at 128³ (2b-1 spec §6).
# Takes minutes, needs the real GPU, and is not part of `check`.
bench-presets:
    cargo run --release -p elements-ember --example presets
```

Run: `just check`
Expected: PASS (the examples compile under clippy's `--all-targets`).

Commit the rule and the tooling before running it, so the results table names a clean commit:

```bash
git add crates/elements-ember justfile
git commit   # subject: "Add the preview preset sweep and its pre-registered rule"
```

- [ ] **Step 5: Run the sweep**

Close what you can. Record the 1-minute load average before and after (`uptime`), because the gate's run showed how much load can move the timings.

Run: `just bench-presets`
Expected: it takes a few minutes and writes `docs/bench/presets.md`.

If the verdict is "no cap fits", run `PRESET_ADVECTION=semi_lagrangian just bench-presets` too, as the rule says.

- [ ] **Step 6: STOP — the user decides**

Do not edit the preset yet. Report the table, the rule's verdict, the frames-at-the-cap column and the load averages to the controller, who takes them to the user. The user confirms or overrides the preview cap, exactly as with the speed gate. Continue only with the confirmed value.

- [ ] **Step 7: Record the decision and set the preset**

1. In `docs/bench/presets.md`, add a "Conditions" paragraph with the load averages and replace `_pending_` with the user's decision and today's date.
2. In `Quality::params`, set preview's `max_substeps` to the confirmed value, and change its comment to cite `docs/bench/presets.md` and the decision date. If the fallback applied, also set preview's `advection` to `Advection::SemiLagrangian`, with the same citation.
3. Run `just check`. If the confirmed value is not 2, a test pinning the old default may fail. Update it only if it pins the preset value itself, and say so in the commit.

- [ ] **Step 8: Bring the docs up to date**

- `CLAUDE.md`, "What this is": replace the sentence beginning "**2b** is next:" with:
  "**2b** is split into three cycles. **2b-1**, solver correctness, is complete: CFL substeps, RK2 + MacCormack advection, vorticity confinement, dissipation, per-face boundaries, and the `preview` and `final` presets (`docs/bench/presets.md`). **2b-2**, scene content (emitters, colliders, wind, flame), is next. **2b-3**, the Mantaflow benchmark, follows it."
  Add `- Ember piece 2b-1 spec: docs/superpowers/specs/2026-09-22-ember-solver-2b1-design.md` and `- Ember piece 2b-1 plan: docs/superpowers/plans/2026-09-22-ember-solver-2b1.md` to the list of documents, and `docs/bench/presets.md` to the bench line. Under Commands, add `just bench-presets # the preview preset's substep cap; minutes, real GPU, not in check`.
- `docs/superpowers/specs/2026-09-21-ember-solver-design.md`, status line: add "Piece 2b-1 (solver correctness) is complete; see `2026-09-22-ember-solver-2b1-design.md`. Of §6's risks, (b), (c), (d), (f) and (h) are resolved; (e) and the open part of (g) remain."
- `docs/superpowers/specs/2026-09-22-ember-solver-2b1-design.md`: `**Status:** Complete. Preview's substep cap was decided on <date> (docs/bench/presets.md).`
- `README.md`: if its status section names 2b as next, update it the same way. Otherwise leave it alone.

- [ ] **Step 9: Run the gate and commit**

Run: `just check`
Expected: PASS.

```bash
git add docs CLAUDE.md README.md crates/elements-ember/src/solver.rs
git commit   # subject: "Record the preview preset decision, and mark 2b-1 complete"
```

The body gives the sweep's verdict, the user's decision and the conditions of the run.

Then push the branch and check CI on llvmpipe, because passing on Metal is not evidence of passing on lavapipe:

```bash
git push -u origin ember-solver-2b1
gh run list --branch ember-solver-2b1 --limit 1
```

Expected: the run completes green. If it fails, treat it like any failing test: find the cause before touching tolerances.
