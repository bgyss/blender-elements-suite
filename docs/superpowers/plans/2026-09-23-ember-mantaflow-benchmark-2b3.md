# Ember Piece 2b-3 — Mantaflow Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure Ember and Blender's Mantaflow on the same three scenes at
64³, 128³ and 256³, with one set of Rust metric functions reading both
solvers' fields, and commit the results table.

**Architecture:** Ember runs in-process in a new `benchmark` example that
reads the solver's state back from the GPU. Mantaflow runs in headless
Blender, driven by a GPL-side Python script built from the scene's JSON.
Its uncompressed OpenVDB cache is read with `vdb-rs` into the same array
layout. `elements_ember::metrics` computes every quality number for both
solvers, and `elements_ember::bench::report` builds `docs/bench/results.md`
only from a complete set of per-run result files.

**Tech Stack:** Rust (wgpu 30 on Metal), `vdb-rs` 0.6, `serde_json`,
Blender 5.2.2 LTS (`bpy`), Python linted by `ruff`, `just`.

**Spec:** `docs/superpowers/specs/2026-09-23-ember-mantaflow-benchmark-2b3-design.md`

## Global Constraints

- `just check` passes before every commit.
- `crates/*` is `Apache-2.0 OR MIT`. Blender-side Python lives under
  `tests/bench/` and is GPL-3.0-or-later. No GPL code, and no Mantaflow or
  Blender source, is copied into any crate.
- Scalar fields stay `R32Float`. `required_features` stays empty.
- `#![forbid(unsafe_code)]` in every crate except `elements-ipc`.
- Crate manifests use `dep.workspace = true`. Versions live only in the
  workspace `Cargo.toml`.
- Never set `WGPU_BACKEND` locally.
- **Every new test is proven able to fail.** Apply a single-change mutation
  to the code it covers, watch it fail, restore the code, and record the real
  failure output in the ledger (`.superpowers/sdd/progress.md`). If the
  mutation turns out to be an equivalent mutant, replace it with another
  and record both.
- Timing and benchmark numbers are never pass/fail tests.
- Commits: a plain imperative subject, then a body explaining **why**, ending
  with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TSRpZQEY9fnKQqHLh5WHpU
  ```
  Commit with plain `git`. The repo is jj-colocated; do not run `jj`.
- Blender is `/Applications/Blender.app/Contents/MacOS/Blender`, or
  `$BLENDER_BIN` when set.

## Deviations from the spec

- **`StateStore::get` in `elements-core`.** Spec §1 says the only Ember
  changes are `active_frames` and a pool high-water mark. The benchmark also
  has to read the solver's velocity, which the graph does not output (the
  output node passes one field). Task 5 adds a read-only
  `StateStore::get(node, slot)` to core and makes the solver's slot names
  public. Neither changes behaviour.
- **Determinism test frames.** Spec §3 names frames 60 and 61. The
  determinism helper in `tests/solver.rs` checks frame 40 of a 16³ scene, so
  Task 1 sets `active_frames = [1, 39]` and checks frame 40 (the first frame
  after the cut-off). The property is the same.
- **Pool high-water mark.** The pool never frees a texture except in `clear`,
  which the benchmark never calls, so the bytes it has ever allocated are its
  peak. Task 5 adds `FieldPool::allocated_bytes`, documented as exactly that.
  The frame cache is off in every run (`cache_budget_mb: 0`), so its size is
  0 and the table says so.

## File structure

| File | Responsibility | Task |
|---|---|---|
| `crates/elements-ember/src/shape_emitter.rs` | `active_frames` parameter, validation, and zero output outside it | 1 |
| `crates/elements-ember/tests/shape_emitter.rs`, `tests/solver.rs` | activity window and determinism tests | 1 |
| `crates/elements-ember/src/bench.rs` | scenes on `ember.emitter`; `mantaflow_json`; `solid_mask` | 2, 6 |
| `crates/elements-ember/examples/speed_gate.rs` | uses `fill_emitter` instead of `fill_sphere` | 2 |
| `docs/bench/presets.md` | idle-machine rerun | 2 |
| `tests/bench/probe_mantaflow.py` | the Task 3 bake that answers the unknowns | 3 |
| `tests/bench/mapping.py` | Ember → Mantaflow parameter conversions found in Task 3 | 3 |
| `docs/bench/mantaflow-notes.md` | findings and the mapping table | 3 |
| `crates/elements-ember/tests/fixtures/mantaflow_16/` | tiny committed cache for the reader test | 3 |
| `crates/elements-ember/src/metrics.rs` | new metrics | 4 |
| `crates/elements-ember/tests/metrics.rs` | analytic-field tests | 4 |
| `crates/elements-core/src/graph/state.rs`, `src/gpu/pool.rs` | `StateStore::get`, `FieldPool::allocated_bytes` | 5 |
| `crates/elements-ember/src/solver.rs` | public slot names | 5 |
| `crates/elements-ember/src/bench/report.rs` (with `bench.rs` → `bench/mod.rs`) | result file types and `results.md` assembly | 5, 8 |
| `crates/elements-ember/examples/benchmark.rs` | Ember runner, Mantaflow runner, report | 5, 8 |
| `tests/bench/mantaflow_scene.py` | builds and bakes the Mantaflow scene from JSON | 6 |
| `crates/elements-ember/src/mantaflow.rs` | reads a Mantaflow cache frame into arrays | 7 |
| `crates/elements-ember/tests/mantaflow.rs` | reader test on the fixture | 7 |
| `justfile` | `bench` recipe | 8 |
| `docs/bench/results.md`, `docs/bench/results/*` | the results | 8 |

---

### Task 1: Emitter activity window

**Files:**
- Modify: `crates/elements-ember/src/shape_emitter.rs` (`EmitterParams`, `EmitterParams::new`, `Emitter::eval`, `build`)
- Test: `crates/elements-ember/tests/shape_emitter.rs`, `crates/elements-ember/tests/solver.rs`

**Interfaces:**
- Produces: `EmitterParams::active_frames: Option<[u32; 2]>` (inclusive; `None` = always on); `EmitterParams::is_active(&self, frame: u32) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `crates/elements-ember/tests/shape_emitter.rs`:

```rust
use elements_core::graph::{Document, StateStore, Time};

/// A 16³ document: one sphere emitter wired straight to the output, with
/// `active` as its activity window.
fn window_doc(active: &str) -> Document {
    serde_json::from_str(&format!(
        r#"{{
      "version": 3, "dims": [16, 16, 16], "fps": 24.0, "domain_size": 2.0,
      "nodes": [
        {{ "id": 0, "kind": "ember.emitter", "params": {{
            "shape": {{ "sphere": {{ "radius": 0.4 }} }},
            "transform": {{ "keys": [ {{ "frame": 1, "translate": [1.0, 1.0, 1.0] }} ] }},
            "density_rate": 1.0, "active_frames": {active} }} }},
        {{ "id": 1, "kind": "core.output", "params": {{}} }}
      ],
      "edges": [ {{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }} ],
      "output": 1
    }}"#
    ))
    .unwrap()
}

/// The emitter's density output at `frame`.
fn density_at(doc: Document, frame: u32) -> Vec<f32> {
    let gpu = gpu();
    let registry = elements_ember::registry();
    let (graph, dims) = doc.into_graph(&registry).unwrap();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let mut state = StateStore::new();
    let time = Time::at(frame, 1, 24.0);
    let out = graph
        .eval_frame(&gpu, &mut pool, &mut cache, &mut state, time, dims)
        .unwrap();
    out.value.as_field().unwrap().read_back(&gpu).unwrap()
}

/// Spec §3: the emitter emits on its first and last active frames, and
/// nothing on the frame after.
#[test]
fn an_emitter_is_silent_outside_its_active_frames() {
    let on_last = density_at(window_doc("[1, 60]"), 60);
    assert!(on_last.iter().any(|&d| d > 0.0), "frame 60 emits");
    let on_first = density_at(window_doc("[5, 60]"), 5);
    assert!(on_first.iter().any(|&d| d > 0.0), "frame 5 emits");
    let after = density_at(window_doc("[1, 60]"), 61);
    assert!(after.iter().all(|&d| d == 0.0), "frame 61 is silent");
    let before = density_at(window_doc("[5, 60]"), 4);
    assert!(before.iter().all(|&d| d == 0.0), "frame 4 is silent");
    let always = density_at(window_doc("null"), 500);
    assert!(always.iter().any(|&d| d > 0.0), "no window is always on");
}

#[test]
fn an_active_range_must_not_run_backwards() {
    let registry = elements_ember::registry();
    let err = window_doc("[61, 60]").into_graph(&registry).err();
    assert!(
        matches!(err, Some(DocError::InvalidParams { .. })),
        "{err:?}"
    );
}
```

Check the exact `DocError` variant that `params::bad` returns
(`grep -n "pub fn bad" crates/elements-ember/src/params.rs`) and use it in
the `matches!`.

Append to `crates/elements-ember/tests/solver.rs`, after
`frame_40_is_bit_identical_with_animated_emitter_and_collider`:

```rust
/// Spec §3: the frame after the emitter switches off reproduces bit for bit
/// in order, after scrubbing across the cut-off, and after eviction.
#[test]
fn frame_40_is_bit_identical_after_the_emitter_switches_off() {
    let doc = ANIMATED.replace(
        r#""density_rate": 1.0,"#,
        r#""active_frames": [1, 39], "density_rate": 1.0,"#,
    );
    assert_ne!(doc, ANIMATED, "the window must be inserted");
    assert_doc_frame_40_is_bit_identical(&doc);
}
```

- [ ] **Step 2: Run them and check they fail**

Run: `cargo nextest run -p elements-ember --test shape_emitter --test solver`
Expected: the three new tests fail. Parsing the document fails with an
unknown-field error, because `EmitterParams` has `deny_unknown_fields`.

- [ ] **Step 3: Implement**

In `crates/elements-ember/src/shape_emitter.rs`, add the field to
`EmitterParams` after `noise`:

```rust
    /// Frames on which the emitter emits, inclusive. `None` is always on.
    /// Outside the range every output is zero (2b-3 spec §3).
    #[serde(default)]
    pub active_frames: Option<[u32; 2]>,
```

Set `active_frames: None` in `EmitterParams::new`, and add:

```rust
impl EmitterParams {
    /// Whether the emitter emits on `frame`.
    pub fn is_active(&self, frame: u32) -> bool {
        self.active_frames
            .is_none_or(|[first, last]| (first..=last).contains(&frame))
    }
}
```

In `build`, after the `velocity_blend` check:

```rust
    if let Some([first, last]) = p.active_frames
        && first > last
    {
        return Err(params::bad(KIND, "active_frames must not end before it starts"));
    }
```

In `Emitter::eval`, zero every output outside the window. `fill_constant`
is exported from `elements_core::gpu`:

```rust
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let time = ctx.time();
        if !self.params.is_active(time.frame) {
            return produce(ctx, 3, |gpu, cache, cells, velocity| {
                for field in cells {
                    fill_constant(gpu, cache, field, 0.0)?;
                }
                for axis in [Axis::X, Axis::Y, Axis::Z] {
                    fill_constant(gpu, cache, velocity.face(axis), 0.0)?;
                }
                Ok(())
            });
        }
        // ... the existing body, unchanged ...
    }
```

Check `fill_constant`'s signature
(`sed -n 130,160p crates/elements-core/src/gpu/dispatch.rs`) and adjust the
call to match it.

- [ ] **Step 4: Run the tests**

Run: `cargo nextest run -p elements-ember --test shape_emitter --test solver`
Expected: all pass.

- [ ] **Step 5: Prove each new test can fail**

Apply each mutation alone, run the named test, record the real output in
the ledger, then restore the code:
1. In `is_active`, change `first..=last` to `first..last`: the window test
   fails on "frame 60 emits".
2. In `eval`, change `!self.params.is_active(time.frame)` to `false`: the
   window test fails on "frame 61 is silent".
3. In `build`, change `first > last` to `first > last + 1000`: the
   backwards-range test fails.
4. For the determinism test, replace the `fill_constant(.., 0.0)` in the
   inactive branch with nothing (leave the fields uninitialised). The
   frame-40 test should then fail on "after scrubbing" or "after eviction",
   because pooled textures carry old contents. If it passes, that mutation
   is equivalent. Record it, and instead change `0.0` to `1.0` for density
   only, which must fail.

- [ ] **Step 6: `just check`, then commit**

```bash
git add crates/elements-ember/src/shape_emitter.rs crates/elements-ember/tests/shape_emitter.rs crates/elements-ember/tests/solver.rs
git commit   # subject: "Let an emitter emit only on a range of frames"
```

The body says why: the benchmark stops emission at frame 60 so that mass
drift can be measured for both solvers (spec §3, §4.3).

---

### Task 2: Bench scenes on `ember.emitter`, and the idle-machine preset rerun

**Files:**
- Modify: `crates/elements-ember/src/bench.rs`
- Modify: `crates/elements-ember/examples/speed_gate.rs` (`divergence_at`)
- Modify: `crates/elements-ember/tests/bench.rs`
- Modify: `docs/bench/presets.md` (regenerated by the run)

**Interfaces:**
- Consumes: `EmitterParams`, `EmitterParams::new`, `active_frames` (Task 1); `Transform::at`, `Shape::Sphere`.
- Produces: `Scene::emitter: EmitterParams`; `pub const EMISSION_FRAMES: [u32; 2] = [1, 60]`.

- [ ] **Step 1: Write the failing test**

Append to `crates/elements-ember/tests/bench.rs`:

```rust
use elements_ember::bench::EMISSION_FRAMES;

/// 2b-3 spec §3: every bench scene uses the general emitter, a static
/// sphere with no noise or velocity, emitting for frames 1–60.
#[test]
fn bench_scenes_use_the_general_emitter_with_a_window() {
    for scene in [
        Scene::plume(32),
        Scene::plume_collider(32),
        Scene::plume_wind(32),
    ] {
        let doc = scene.document();
        let emitter = &doc.nodes[0];
        assert_eq!(emitter.kind, "ember.emitter", "{}", scene.name);
        assert_eq!(
            emitter.params["active_frames"],
            serde_json::json!(EMISSION_FRAMES),
            "{}",
            scene.name
        );
        assert_eq!(scene.emitter.velocity_blend, 0.0);
        assert!(scene.emitter.noise.is_none());
        // The document still evaluates.
        let registry = elements_ember::registry();
        doc.into_graph(&registry).unwrap();
    }
}
```

- [ ] **Step 2: Run it and check it fails**

Run: `cargo nextest run -p elements-ember --test bench`
Expected: compile error, because `EMISSION_FRAMES` does not exist.

- [ ] **Step 3: Implement**

In `bench.rs`:
- Replace `use crate::emitter::{self, Sphere};` with
  `use crate::shape_emitter::{self, EmitterParams};`.
- Change the field to `pub emitter: EmitterParams,`.
- Add, next to the gate constants:

```rust
/// Frames on which every bench scene emits (2b-3 spec §3). Frames after the
/// last measure mass drift with no sources.
pub const EMISSION_FRAMES: [u32; 2] = [1, 60];
```

- In `plume`, replace the `Sphere { .. }` literal with:

```rust
            emitter: EmitterParams {
                density_rate: 1.0,
                temperature_rate: 1.0,
                active_frames: Some(EMISSION_FRAMES),
                ..EmitterParams::new(Shape::Sphere { radius: 0.2 }, Transform::at([1.0, 1.0, 0.3]))
            },
```

- In `document`, change the emitter node's kind to `shape_emitter::KIND`, and
  its params to `serde_json::to_value(&self.emitter)`. Its outputs 0 and 1
  (density and temperature rates) stay wired to solver inputs 0 and 1.
  Outputs 2 and 3 (velocity weight and target) stay unwired, because the
  scenes emit no velocity.
- Update the `Scene` doc comment, which says 2b adds the Mantaflow script,
  to point at the 2b-3 spec.

In `examples/speed_gate.rs`, `divergence_at` calls `fill_sphere` with
`&scene.emitter`. Replace it with `fill_emitter`, which writes all four
outputs:

```rust
    let weight = pool.acquire(gpu, cells, FieldFormat::R32Float)?;
    let target = pool.acquire_staggered_uninit(gpu, cells)?;
    let pose = scene.emitter.transform.pose(1.0, 1.0 / scene.fps);
    fill_emitter(
        gpu,
        &mut cache,
        &scene.emitter,
        &pose,
        0.0,
        dx,
        EmitterFields {
            density: &density_source,
            temperature: &temperature_source,
            weight: &weight,
            velocity: &target,
        },
    )?;
```

Keep `Sources::new(&density_source, &temperature_source)`: the weight and
target are zero, so the emission pair is not needed. Update the imports to
`use elements_ember::shape_emitter::{EmitterFields, fill_emitter};`.
`divergence_at` runs to frame 48, inside the emission window, so it does not
check `is_active`.

- [ ] **Step 4: Run the tests**

Run: `cargo nextest run -p elements-ember` then
`cargo build --release -p elements-ember --examples`.
Expected: all pass, and both examples build.

- [ ] **Step 5: Prove the test can fail**

Mutation: in `plume`, set `active_frames: None`. The test fails on the
`active_frames` assertion. Record the output and restore.

- [ ] **Step 6: `just check`, then commit**

Subject: "Run the bench scenes on the general emitter, with emission ending
at frame 60". The body says why: the benchmark needs the activity window,
which only `ember.emitter` has, and the speed gate and presets must measure
the same scene the benchmark uses.

- [ ] **Step 7: The idle-machine preset rerun (risk (k))**

This step needs an idle machine. Check the load first:

Run: `sysctl -n vm.loadavg`
If the 1-minute load is above 2, report it and wait. Do not run under load.
If it stays above 2 for 10 minutes, report to the user and stop.

Run: `just bench-presets` (several minutes).
It rewrites `docs/bench/presets.md`. Keep the file's hand-written decision
line: the program writes `_pending_`, so restore the previous decision text
from `git diff` and add a line under it giving the date, load average, and
the new cap-1 median.

- **If cap 1's median frame is at most 100 ms:** commit `docs/bench/presets.md`
  with subject "Rerun the preview preset sweep on an idle machine", and a body
  giving the load and the change from about 92 ms. Also add a dated paragraph
  under risk (k) in `docs/superpowers/specs/2026-09-21-ember-solver-design.md`
  §6 recording the result, in the same commit.
- **If it is above 100 ms, or cap 2 now fits:** commit the file as above, then
  **stop and report the table to the user.** The preview preset is theirs to
  reopen, and the benchmark would measure a preset that is about to change.

---

### Task 3: Answer the Mantaflow unknowns

This task is exploration. Its deliverables are a notes file, a mapping
module, a small fixture, and a decision on whether the design stands.
**If any finding contradicts the spec** (for example `vdb-rs` cannot read the
velocity grid, velocity is stored in a way §4.1 does not cover, or 256³
takes more than 90 minutes in total), **stop and report to the user before
Task 4.**

**Files:**
- Create: `tests/bench/probe_mantaflow.py`
- Create: `tests/bench/mapping.py`
- Create: `docs/bench/mantaflow-notes.md`
- Create: `crates/elements-ember/tests/fixtures/mantaflow_16/` (a cache frame and a `README.md`)
- Create: `crates/elements-ember/examples/vdb_probe.rs` (temporary; deleted before commit)

**Interfaces:**
- Produces, in `docs/bench/mantaflow-notes.md`: the cache file name
  pattern; each grid's name and value type; whether velocity is on faces or
  at cell centres; the index-space origin and extent for a domain of
  resolution n; the per-frame timing method; the mapping table.
- Produces `tests/bench/mapping.py` with exactly these functions, used by
  Task 6:

```python
def buoyancy(ember_density: float, ember_temperature: float) -> tuple[float, float]:
    """Return Mantaflow's (alpha, beta) for Ember's buoyancy coefficients."""

def inflow(density_rate: float, temperature_rate: float, fps: float) -> tuple[float, float]:
    """Return the flow object's (density, temperature) for Ember's rates."""

def vorticity(ember_confinement: float, dx: float) -> float:
    """Return the domain's vorticity setting for Ember's confinement."""

def wind(ember_accel: tuple[float, float, float]) -> tuple[float, tuple[float, float, float]]:
    """Return a wind field's (strength, direction) for Ember's acceleration."""

def rotation_to(direction: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Return the (w, x, y, z) quaternion that turns a wind field's blowing
    axis onto `direction`."""
```

- [ ] **Step 1: Write the probe script**

`tests/bench/probe_mantaflow.py`:

```python
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bake a tiny gas domain headlessly and print what the cache contains.

Run: Blender --background --factory-startup --python tests/bench/probe_mantaflow.py -- OUT_DIR RES FRAMES
"""

import json
import os
import sys
import time

import bpy


def main() -> None:
    out_dir, res, frames = sys.argv[sys.argv.index("--") + 1 :]
    res, frames = int(res), int(frames)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, frames
    scene.render.fps = 24

    # A 2 m domain from (0, 0, 0) to (2, 2, 2), matching Ember's convention.
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(1.0, 1.0, 1.0))
    domain = bpy.context.active_object
    fluid = domain.modifiers.new("Fluid", "FLUID")
    fluid.fluid_type = "DOMAIN"
    s = fluid.domain_settings
    s.domain_type = "GAS"
    s.resolution_max = res
    s.use_adaptive_domain = False
    s.use_noise = False
    s.use_adaptive_timesteps = False
    s.timesteps_min = s.timesteps_max = 1
    s.cache_directory = out_dir
    s.cache_type = "ALL"
    s.cache_data_format = "OPENVDB"
    s.openvdb_cache_compress_type = "NONE"
    s.openvdb_data_depth = "32"
    s.cache_frame_start, s.cache_frame_end = 1, frames

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(1.0, 1.0, 0.3))
    flow_obj = bpy.context.active_object
    flow = flow_obj.modifiers.new("Fluid", "FLUID")
    flow.fluid_type = "FLOW"
    flow.flow_settings.flow_type = "SMOKE"
    flow.flow_settings.flow_behavior = "INFLOW"

    start = time.perf_counter()
    with bpy.context.temp_override(object=domain, active_object=domain):
        result = bpy.ops.fluid.bake_all()
    elapsed = time.perf_counter() - start
    files = []
    for root, _, names in os.walk(out_dir):
        for name in sorted(names):
            path = os.path.join(root, name)
            files.append({"path": path, "mtime_ns": os.stat(path).st_mtime_ns})
    print(json.dumps({"result": list(result), "seconds": elapsed, "files": files}, indent=1))


main()
```

If Blender 5.2.2 rejects an attribute name (for example `openvdb_data_depth`),
find the right one with
`Blender --background --factory-startup --python-expr "import bpy; print([p.identifier for p in bpy.types.FluidDomainSettings.bl_rna.properties])"`
and record the correction in the notes.

- [ ] **Step 2: Run it at 16³ and at 32³**

```bash
B=${BLENDER_BIN:-/Applications/Blender.app/Contents/MacOS/Blender}
S=/private/tmp/claude-501/-Users-briangyss-src-blender-elements-suite/41bd0494-b460-4638-9323-394a223d8601/scratchpad
rm -rf "$S/probe16" && mkdir -p "$S/probe16"
"$B" --background --factory-startup --python tests/bench/probe_mantaflow.py -- "$S/probe16" 16 5
```

Record: whether `bake_all` returned `FINISHED`, the file layout, and the
file name pattern for frame n.

- [ ] **Step 3: Inspect a cache frame with `vdb-rs`**

Create a temporary example, `crates/elements-ember/examples/vdb_probe.rs`:

```rust
//! Temporary: print every grid in a VDB file (2b-3 Task 3). Deleted before commit.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: vdb_probe FILE");
    let mut reader = vdb_rs::VdbReader::new(std::io::BufReader::new(std::fs::File::open(&path)?))?;
    for name in reader.available_grids() {
        println!("grid {name}");
        match reader.read_grid::<f32>(&name) {
            Ok(g) => {
                let (mut n, mut lo, mut hi) = (0usize, [i32::MAX; 3], [i32::MIN; 3]);
                for (c, _v, level) in g.iter() {
                    n += 1;
                    for a in 0..3 {
                        lo[a] = lo[a].min(c[a] as i32);
                        hi[a] = hi[a].max(c[a] as i32);
                    }
                    if n == 1 { println!("  first level {level:?}"); }
                }
                println!("  as f32: {n} active, index bbox {lo:?}..={hi:?}");
            }
            Err(e) => println!("  as f32: {e}"),
        }
        match reader.read_grid::<[f32; 3]>(&name) {
            Ok(g) => println!("  as [f32; 3]: {} active", g.iter().count()),
            Err(e) => println!("  as [f32; 3]: {e}"),
        }
    }
    Ok(())
}
```

If `vdb-rs` is not already a dependency of `elements-ember`, add
`vdb-rs.workspace = true` under `[dependencies]` in
`crates/elements-ember/Cargo.toml`; Task 7 needs it anyway. Then run
`cargo run -p elements-ember --example vdb_probe -- <frame file>`.

Record, for every grid: its name, whether it reads as `f32` or `[f32; 3]`,
the active index bounding box, and whether any values come from tiles
(`VdbLevel::Node3` or `Node4`) rather than voxels. Check the grid's transform
metadata (voxel size and origin) through `grid.descriptor`, and record how
index (0, 0, 0) maps onto the domain's minimum corner.

**Velocity location.** Mantaflow simulates on a MAC grid. Find out whether
the cached `velocity` holds face values (x at the cell's −x face, and so on)
or cell-centre values. Check the grid's `vec_type` or class metadata if
`vdb-rs` exposes it, then confirm by experiment: bake a domain with a flow
object whose initial velocity is a uniform +x (set
`flow_settings.use_initial_velocity = True` and `velocity_coord = (1, 0, 0)`)
and see whether the non-zero x values extend half a cell beyond the emitter
on the −x side (faces) or not (centres).

If `vdb-rs` cannot read the float grids or the velocity grid, **stop and
report**. The spec's fallback (a Blender-side `.npy` exporter) changes Task 7
and needs the user's go-ahead.

- [ ] **Step 4: Find the parameter mappings**

For each quantity, find the conversion and write it in
`tests/bench/mapping.py`, with a one-line comment giving its source. The
source is Blender's Python API docs, the Mantaflow plugin source in Blender's
repository (read, never copied), or an experiment run here.
- **Buoyancy:** Ember's force is `buoyancy_temperature · T − buoyancy_density · ρ`
  along +z (`crates/elements-ember/src/kernels/shaders/buoyancy.wgsl`).
  Mantaflow's `alpha` (density) and `beta` (heat) scale its buoyancy term.
  Find the exact form, including gravity and any division by resolution.
- **Inflow:** Ember adds `density_rate · occupancy · h` per substep.
  Mantaflow's `INFLOW` sets density to the flow's `density` value inside the
  emitter each step. Choose the flow density and temperature so that inside
  the emitter both solvers hold the same steady value. Ember's value inside
  the emitter grows with time, so match Ember's density at the emitter
  centre at frame 24, measured with `density_at`-style code. Record the
  choice and why.
- **Vorticity:** the bench scenes use the preset's confinement. Find
  Mantaflow's scaling.
- **Wind:** a `WIND` force field's strength and falloff versus Ember's uniform
  acceleration of 0.5 m/s². Set falloff to none. Check by experiment that a
  closed 32³ domain with no emitter reaches the expected velocity after one
  second.
- **Substeps:** check that `timesteps_min = timesteps_max = 1` gives one
  solver step per frame.

- [ ] **Step 5: Time 256³**

Run the probe at 256³ for 10 frames and record seconds per frame from the
file modification times. Extrapolate the full run: 3 scenes × 120 frames ×
1 run at 256³, plus 3 runs at 64³ and 128³. If it is over 90 minutes,
**stop and report** (spec §8 risk (d)).

- [ ] **Step 6: Commit the fixture**

Bake the probe at 16³ for 5 frames. Copy the frame-5 data file into
`crates/elements-ember/tests/fixtures/mantaflow_16/`, and check it is under
200 KB. Write a `README.md` beside it giving the command that produced it,
the Blender version, and the grid inventory from Step 3.

- [ ] **Step 7: Write the notes, delete the probe example, commit**

`docs/bench/mantaflow-notes.md` has the sections: Headless baking, Cache
layout, Grids, Velocity location, Index space, Timing, Parameter mapping (a
table: quantity, Ember, Mantaflow, conversion, source), 256³ run time, and
Corrections to the spec (if any).

```bash
rm crates/elements-ember/examples/vdb_probe.rs
ruff format tests && ruff check tests
git add tests/bench docs/bench/mantaflow-notes.md crates/elements-ember/tests/fixtures/mantaflow_16 crates/elements-ember/Cargo.toml
git commit   # subject: "Record how Blender's Mantaflow cache is laid out, and map Ember's parameters onto it"
```

---

### Task 4: Metrics

**Files:**
- Modify: `crates/elements-ember/src/metrics.rs`
- Create: `crates/elements-ember/tests/metrics.rs`

**Interfaces:**
- Consumes: `metrics::divergence`, `metrics::centroid_z` (existing).
- Produces:

```rust
pub enum Velocity {
    /// Face values, x-fastest, with each axis's face grid one longer along it.
    Faces([Vec<f32>; 3]),
    /// Cell-centred values, x-fastest, at the cell dims.
    Centred([Vec<f32>; 3]),
}

pub struct Sample<'a> {
    pub cells: FieldDims,
    /// Voxel edge, metres.
    pub dx: f64,
    pub density: &'a [f32],
    pub velocity: &'a Velocity,
    /// One entry per cell, true inside a collider; empty when there is none.
    pub solid: &'a [bool],
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DivergenceRule { Faces, CentralDifferences }

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FrameMetrics {
    pub divergence_rule: DivergenceRule,
    pub divergence_max: f64,
    pub divergence_rms: f64,
    /// ½ Σ|u|² dV over interior cells, m⁵/s².
    pub kinetic_energy: f64,
    /// Σ|∇×u| dV over interior cells, m³/s.
    pub vorticity: f64,
    /// Σ ρ dV, density × m³.
    pub mass: f64,
    /// Density-weighted mean height, metres.
    pub centroid_m: Option<f64>,
    /// Height below which 95% of the above-threshold density lies, metres.
    pub top_m: Option<f64>,
    /// Upward flux of density through the domain top, density × m³ / s.
    pub outflow_rate: f64,
}

pub fn cell_centred(faces: &[Vec<f32>; 3], cells: FieldDims) -> [Vec<f32>; 3];
pub fn measure(sample: &Sample<'_>) -> FrameMetrics;
/// Drift from frame index `from` (0-based into the series) at every later
/// index: (M(n) + outflow integrated from `from` to n) − M(from).
pub fn drift(mass: &[f64], outflow_rate: &[f64], frame_seconds: f64, from: usize) -> Vec<f64>;
```

Definitions (spec §4):
- **Divergence.** `Faces` uses the existing `divergence` stencil. `Centred`
  uses central differences, (u[i+1] − u[i−1]) / 2dx per axis, on cells with
  both neighbours inside the domain. Both skip solid cells. Rename the
  existing function's loop into a helper that takes a skip predicate, so
  `divergence` keeps its signature and behaviour.
- **Interior** (kinetic energy and vorticity): cells with 1 ≤ i ≤ n − 2 on
  every axis and no solid cell among their 26 neighbours or themselves.
- **Vorticity:** |∇×u| using central differences of the cell-centred
  velocity.
- **Plume top:** among cells with ρ ≥ 0.01 · max ρ, sort by height
  (k + 0.5) · dx, and return the smallest height at which the cumulative
  density reaches 95% of their total. `None` when max ρ ≤ 0.
- **Outflow rate:** Σ over top cells (k = nz − 1) of ρ · max(w, 0) · dx²,
  where w is the top face's value (`Faces`: z-face index k = nz) or the
  top cell's centred value (`Centred`).

- [ ] **Step 1: Write the failing tests**

`crates/elements-ember/tests/metrics.rs`:

```rust
use elements_core::gpu::FieldDims;
use elements_ember::metrics::{
    DivergenceRule, Sample, Velocity, cell_centred, drift, measure,
};

const N: u32 = 16;
const DX: f64 = 0.125; // a 2 m domain

fn cells() -> FieldDims {
    FieldDims::new(N, N, N)
}

fn idx(i: u32, j: u32, k: u32) -> usize {
    (i + N * (j + N * k)) as usize
}

/// Centred velocity from a function of the cell centre (metres).
fn centred(f: impl Fn([f64; 3]) -> [f64; 3]) -> Velocity {
    let mut out: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; (N * N * N) as usize]);
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                let p = [i, j, k].map(|c| (c as f64 + 0.5) * DX);
                let u = f(p);
                for a in 0..3 {
                    out[a][idx(i, j, k)] = u[a] as f32;
                }
            }
        }
    }
    Velocity::Centred(out)
}

fn sample<'a>(density: &'a [f32], velocity: &'a Velocity, solid: &'a [bool]) -> Sample<'a> {
    Sample { cells: cells(), dx: DX, density, velocity, solid }
}

fn zeros() -> Vec<f32> {
    vec![0.0; (N * N * N) as usize]
}

#[test]
fn a_uniform_flow_has_no_divergence_or_vorticity() {
    let v = centred(|_| [0.3, -0.2, 0.5]);
    let d = zeros();
    let m = measure(&sample(&d, &v, &[]));
    assert_eq!(m.divergence_rule, DivergenceRule::CentralDifferences);
    assert!(m.divergence_max < 1e-6, "{m:?}");
    assert!(m.vorticity < 1e-6, "{m:?}");
    // Interior cells: 14³, each with |u|² = 0.38.
    let expected = 0.5 * 0.38 * 14f64.powi(3) * DX.powi(3);
    assert!((m.kinetic_energy - expected).abs() < 1e-6 * expected, "{m:?}");
}

/// Solid-body rotation about z at rate ω has vorticity 2ω everywhere.
#[test]
fn solid_body_rotation_has_vorticity_twice_its_rate() {
    let w = 1.5;
    let v = centred(|p| [-w * (p[1] - 1.0), w * (p[0] - 1.0), 0.0]);
    let d = zeros();
    let m = measure(&sample(&d, &v, &[]));
    let expected = 2.0 * w * 14f64.powi(3) * DX.powi(3);
    assert!((m.vorticity - expected).abs() < 1e-4 * expected, "{m:?}");
    assert!(m.divergence_max < 1e-5, "{m:?}");
}

/// A linear expansion u = (a x, 0, 0) has divergence a in every cell whose
/// neighbours are both in the domain.
#[test]
fn central_differences_measure_a_linear_expansion() {
    let v = centred(|p| [0.8 * p[0], 0.0, 0.0]);
    let d = zeros();
    let m = measure(&sample(&d, &v, &[]));
    assert!((m.divergence_max - 0.8).abs() < 1e-5, "{m:?}");
    assert!((m.divergence_rms - 0.8).abs() < 1e-5, "{m:?}");
}

/// A face velocity whose faces grow linearly along x has divergence equal to
/// the slope, through the face stencil.
#[test]
fn face_velocities_use_the_face_stencil() {
    let c = cells();
    let xd = (N + 1) * N * N;
    let mut x = vec![0.0f32; xd as usize];
    for k in 0..N {
        for j in 0..N {
            for i in 0..=N {
                x[(i + (N + 1) * (j + N * k)) as usize] = 0.8 * (i as f32 * DX as f32);
            }
        }
    }
    let faces = [x, vec![0.0; ((N + 1) * N * N) as usize], vec![0.0; ((N + 1) * N * N) as usize]];
    let v = Velocity::Faces(faces);
    let d = zeros();
    let m = measure(&Sample { cells: c, dx: DX, density: &d, velocity: &v, solid: &[] });
    assert_eq!(m.divergence_rule, DivergenceRule::Faces);
    assert!((m.divergence_max - 0.8).abs() < 1e-5, "{m:?}");
}

#[test]
fn cell_centring_averages_the_two_faces() {
    let c = FieldDims::new(2, 1, 1);
    let faces = [vec![1.0, 3.0, 7.0], vec![0.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 0.0]];
    let [x, _, _] = cell_centred(&faces, c);
    assert_eq!(x, vec![2.0, 5.0]);
}

/// A slab of density 2 in layers k = 4..8: mass, centroid and top follow.
#[test]
fn a_density_slab_has_known_mass_centroid_and_top() {
    let mut d = zeros();
    for k in 4..8 {
        for j in 0..N {
            for i in 0..N {
                d[idx(i, j, k)] = 2.0;
            }
        }
    }
    let v = centred(|_| [0.0; 3]);
    let m = measure(&sample(&d, &v, &[]));
    let cells_in_slab = (N * N * 4) as f64;
    assert!((m.mass - 2.0 * cells_in_slab * DX.powi(3)).abs() < 1e-9, "{m:?}");
    assert!((m.centroid_m.unwrap() - 6.0 * DX).abs() < 1e-9, "{m:?}");
    // 95% of four equal layers is reached in the fourth: k = 7, centre 7.5 dx.
    assert!((m.top_m.unwrap() - 7.5 * DX).abs() < 1e-9, "{m:?}");
}

#[test]
fn cells_below_one_percent_of_the_peak_do_not_move_the_top() {
    let mut d = zeros();
    d[idx(3, 3, 2)] = 1.0;
    d[idx(3, 3, 14)] = 0.009; // below 1% of the peak
    let v = centred(|_| [0.0; 3]);
    let m = measure(&sample(&d, &v, &[]));
    assert!((m.top_m.unwrap() - 2.5 * DX).abs() < 1e-9, "{m:?}");
}

#[test]
fn only_upward_flow_through_the_top_counts_as_outflow() {
    let mut d = zeros();
    d[idx(2, 2, N - 1)] = 3.0;
    d[idx(5, 5, N - 1)] = 3.0;
    let v = centred(|p| [0.0, 0.0, if p[0] < 0.5 { 0.4 } else { -0.4 }]);
    let m = measure(&sample(&d, &v, &[]));
    // Only (2, 2) has upward flow at the top: 3 × 0.4 × dx².
    assert!((m.outflow_rate - 3.0 * 0.4 * DX * DX).abs() < 1e-9, "{m:?}");
}

/// A solid cell's neighbourhood is left out of kinetic energy, and the solid
/// cell itself out of divergence.
#[test]
fn solids_are_left_out() {
    let v = centred(|p| [0.8 * p[0], 0.0, 0.0]);
    let d = zeros();
    let mut solid = vec![false; (N * N * N) as usize];
    solid[idx(8, 8, 8)] = true;
    let open = measure(&sample(&d, &v, &[]));
    let with = measure(&sample(&d, &v, &solid));
    assert!(with.kinetic_energy < open.kinetic_energy, "{with:?} vs {open:?}");
    // 27 cells excluded from the interior sum.
    let lost = open.kinetic_energy - with.kinetic_energy;
    assert!(lost > 0.0);
    assert!((with.divergence_rms - 0.8).abs() < 1e-5, "{with:?}");
}

/// Mass 10, falling to 8 while 2 flows out: no drift.
#[test]
fn drift_counts_outflow_as_accounted_for() {
    let mass = [10.0, 9.0, 8.0];
    let outflow = [1.0, 1.0, 1.0]; // per second
    let d = drift(&mass, &outflow, 1.0, 0);
    assert_eq!(d.len(), 3);
    for x in d {
        assert!(x.abs() < 1e-12, "{x}");
    }
}

#[test]
fn drift_integrates_outflow_with_the_trapezoid_rule() {
    let mass = [10.0, 10.0];
    let outflow = [0.0, 2.0];
    // Trapezoid over one second: (0 + 2) / 2 = 1, so drift = 10 + 1 − 10.
    assert_eq!(drift(&mass, &outflow, 1.0, 0), vec![0.0, 1.0]);
}
```

- [ ] **Step 2: Run them and check they fail**

Run: `cargo nextest run -p elements-ember --test metrics`
Expected: compile errors for the missing items.

- [ ] **Step 3: Implement** in `crates/elements-ember/src/metrics.rs`,
following the definitions above, with a doc comment per item. Keep sums in
`f64`. Put the interior predicate in one private function,
`fn interior(cells, solid, i, j, k) -> bool`, that kinetic energy and
vorticity share. Add `serde` to `elements-ember`'s dependencies if it is not
already there (it is, for params).

- [ ] **Step 4: Run the tests.** Expected: all pass. Existing callers of
`divergence` and `centroid_z` still compile unchanged.

- [ ] **Step 5: Prove each test can fail.** One mutation per test, each
applied alone, with the output recorded:
1. Uniform flow: drop the ½ in kinetic energy.
2. Rotation: use a one-sided difference for the curl's ∂v/∂x term.
3. Linear expansion: divide central differences by `dx` instead of `2dx`.
4. Face stencil: make `measure` resample faces to centres and always use
   central differences.
5. Cell centring: take the lower face instead of the mean.
6. Slab: use `k` instead of `k + 0.5` for the plume top's height.
7. Threshold: drop the 1% threshold.
8. Outflow: count |w| instead of max(w, 0).
9. Solids: ignore `solid` in the interior predicate.
10. Drift, first test: leave outflow out.
11. Drift, trapezoid: use the rectangle rule at the right end.

- [ ] **Step 6: `just check`, then commit.** Subject: "Measure energy,
vorticity, mass, plume height and outflow from either solver's fields". The
body says why: one set of functions reads both solvers (spec §4).

---

### Task 5: Read solver state, count pool bytes, and the Ember runner

**Files:**
- Modify: `crates/elements-core/src/graph/state.rs` (`StateStore::get`)
- Modify: `crates/elements-core/src/gpu/pool.rs` (`allocated_bytes`)
- Test: `crates/elements-core/tests/state.rs`, `crates/elements-core/tests/field_pool.rs`
- Modify: `crates/elements-ember/src/solver.rs` (public slot names)
- Move: `crates/elements-ember/src/bench.rs` → `crates/elements-ember/src/bench/mod.rs`
- Create: `crates/elements-ember/src/bench/report.rs`
- Modify: `crates/elements-ember/src/bench/mod.rs` (`Scene::solid_mask`, `pub mod report;`)
- Create: `crates/elements-ember/examples/benchmark.rs`

**Interfaces:**
- Consumes: `metrics::{Sample, Velocity, FrameMetrics, measure}` (Task 4); `EMISSION_FRAMES` (Task 2); `common::{median, shell, commit_label}`.
- Produces:

```rust
// elements-core
impl StateStore { pub fn get(&self, node: NodeId, slot: &'static str) -> Option<&Value>; }
impl FieldPool { pub fn allocated_bytes(&self) -> u64; }
// elements-ember::solver
pub const VELOCITY: &str; pub const DENSITY: &str;
// elements-ember::bench
impl Scene { pub fn solid_mask(&self) -> Vec<bool>; }
/// The solver's node id in every `Scene::document`.
pub const SOLVER_NODE: NodeId = NodeId(1);
// elements-ember::bench::report
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RunSummary {
    pub solver: String,          // "ember" or "mantaflow"
    pub scene: String,
    pub resolution: u32,
    pub runs: u32,
    pub frame_ms_median: f64,
    pub frame_ms_min: f64,
    pub frame_ms_max: f64,
    pub peak_bytes: u64,
    pub load_before: f64,
    pub load_after: f64,
    /// Blender's version string for Mantaflow runs; `None` for Ember.
    pub blender: Option<String>,
    /// One entry per frame, frame 1 first.
    pub frames: Vec<FrameMetrics>,
    /// `drift` from frame 60, one entry per frame from 60 on.
    pub drift: Vec<f64>,
}
pub fn summary_path(dir: &Path, solver: &str, scene: &str, resolution: u32) -> PathBuf; // {dir}/{solver}-{scene}-{resolution}.json
pub fn write_summary(dir: &Path, s: &RunSummary) -> std::io::Result<()>;  // JSON + a CSV of `frames` beside it
pub fn load_average() -> f64; // 1-minute load from `sysctl -n vm.loadavg`
```

- [ ] **Step 1: Write the failing core tests**

Append to `crates/elements-core/tests/field_pool.rs`:

```rust
/// Reuse allocates nothing; a new shape adds its bytes.
#[test]
fn allocated_bytes_counts_only_fresh_textures() {
    let gpu = GpuContext::new_headless().unwrap();
    let mut pool = FieldPool::new();
    assert_eq!(pool.allocated_bytes(), 0);
    let a = pool.acquire(&gpu, FieldDims::new(4, 4, 4), FieldFormat::R32Float).unwrap();
    assert_eq!(pool.allocated_bytes(), 4 * 4 * 4 * 4);
    pool.release(a);
    let b = pool.acquire(&gpu, FieldDims::new(4, 4, 4), FieldFormat::R32Float).unwrap();
    assert_eq!(pool.allocated_bytes(), 4 * 4 * 4 * 4, "reuse is free");
    let _c = pool.acquire(&gpu, FieldDims::new(8, 4, 4), FieldFormat::R32Float).unwrap();
    assert_eq!(pool.allocated_bytes(), 4 * 4 * 4 * 4 + 8 * 4 * 4 * 4);
    pool.release(b);
}
```

Append to `crates/elements-core/tests/state.rs` a test that evaluates a
graph containing a stateful node already used in that file, then asserts
`state.get(that_node, its_slot)` is `Some` and
`state.get(that_node, "no such slot")` is `None`. Read the file first and
reuse its existing stateful test node and slot name.

- [ ] **Step 2: Run them; they fail to compile.**

Run: `cargo nextest run -p elements-core --test field_pool --test state`

- [ ] **Step 3: Implement**

`state.rs`:

```rust
    /// The value in `node`'s `slot`, if any. Read-only: benchmarks and
    /// tests read state between frames through this.
    pub fn get(&self, node: NodeId, slot: &'static str) -> Option<&Value> {
        self.slots.get(&(node, slot))
    }
```

`pool.rs`: add an `allocated_bytes: u64` field to `FieldPool`. In `acquire`,
after a fresh texture is created successfully, add
`dims.voxel_count() as u64 * format.bytes_per_voxel() as u64`. Document the
method: "Bytes of every texture this pool has ever created. The pool frees
textures only in `clear`, so between clears this is its peak."

`solver.rs`: make `VELOCITY` and `DENSITY` `pub const`, with a doc comment
saying they name the solver's state slots.

- [ ] **Step 4: Run the core tests; they pass. Prove each can fail:**
1. `allocated_bytes`: count on every `acquire`, including reuse. The
   "reuse is free" assertion fails.
2. `get`: return `None` always. The `Some` assertion fails.

- [ ] **Step 5: Commit the core part** (`just check` first). Subject:
"Let callers read a node's state and a pool's allocated bytes". The body
says the benchmark needs the solver's velocity, which the graph does not
output, and Ember's peak memory.

- [ ] **Step 6: `solid_mask`, `SOLVER_NODE` and the report module, test first**

`git mv crates/elements-ember/src/bench.rs crates/elements-ember/src/bench/mod.rs`,
then add `pub mod report;` and:

```rust
/// The solver's node id in every `Scene::document`.
pub const SOLVER_NODE: NodeId = NodeId(1);

impl Scene {
    /// True for each cell (x-fastest) whose centre lies inside the collider.
    /// The metrics use it for both solvers, so it is computed on the CPU from
    /// the scene rather than read from either solver.
    pub fn solid_mask(&self) -> Vec<bool> { /* ... */ }
}
```

For the static `Shape::Sphere` and `Shape::Box`, use the collider's
`transform.keys[0].translate` as the centre (bench colliders are static;
`debug_assert_eq!(keys.len(), 1)`), and `dx = domain_size / max(cells)`.
Return an empty `Vec` when there is no collider.

Tests in `crates/elements-ember/tests/bench.rs`:

```rust
#[test]
fn the_collider_mask_marks_cells_inside_the_sphere() {
    let scene = Scene::plume_collider(32);
    let mask = scene.solid_mask();
    let n = 32u32;
    let at = |i: u32, j: u32, k: u32| mask[(i + n * (j + n * k)) as usize];
    // The collider is centred at (1, 1, 0.8) m, radius 0.25; dx = 1/16 m.
    assert!(at(15, 15, 12), "the cell at the centre is solid");
    assert!(!at(15, 15, 20), "a cell 0.5 m above is not");
    assert!(Scene::plume(32).solid_mask().is_empty());
}

#[test]
fn a_summary_round_trips_through_its_file() {
    let dir = tempfile::tempdir().unwrap();
    let s = report::RunSummary { /* fill with small literal values, two frames */ };
    report::write_summary(dir.path(), &s).unwrap();
    let path = report::summary_path(dir.path(), "ember", "plume", 64);
    let back: report::RunSummary = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(back, s);
    assert!(dir.path().join("ember-plume-64.csv").exists());
}
```

Write the `RunSummary` literal out in full in the test. Check whether
`tempfile` is already a workspace dev-dependency; if not, add it to the
workspace `Cargo.toml` and to `elements-ember`'s `[dev-dependencies]` with
`tempfile.workspace = true`.

The CSV has a header row, then one row per frame: `frame`, and each
`FrameMetrics` field in declaration order, with `divergence_rule` written as
its variant name and `None` as an empty cell.

Implement, run, and prove each test can fail, recording both outputs:
1. Mask: use the cell's minimum corner, `i * dx`, instead of its centre,
   `(i + 0.5) * dx`, and check that some assertion fails. If none does, the
   mutation is equivalent at this resolution. Record that, then swap `j` and
   `k` in the index, which must fail on "a cell 0.5 m above is not".
2. Summary: drop the CSV write. The `exists` assertion fails.

- [ ] **Step 7: The Ember runner**

`crates/elements-ember/examples/benchmark.rs`, mode `ember`:

```rust
//! The Mantaflow benchmark (2b-3 spec). Run everything with `just bench`.
//!
//! Modes:
//!   benchmark ember SCENE RES      time and measure Ember
//!   benchmark mantaflow SCENE RES  bake, time and measure Mantaflow (Task 8)
//!   benchmark report               build docs/bench/results.md (Task 8)
//!
//! Each run writes docs/bench/results/{solver}-{scene}-{res}.json and .csv.

mod common;

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use elements_core::gpu::{Axis, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{NodeRegistry, StateStore, Time};
use elements_ember::bench::report::{RunSummary, load_average, write_summary};
use elements_ember::bench::{EMISSION_FRAMES, SOLVER_NODE, Scene};
use elements_ember::metrics::{FrameMetrics, Sample, Velocity, drift, measure};
use elements_ember::solver;

use common::median;

type Res<T> = Result<T, Box<dyn Error>>;

fn results_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/bench/results"))
}

fn scene(name: &str, res: u32) -> Res<Scene> {
    Ok(match name {
        "plume" => Scene::plume(res),
        "plume_collider" => Scene::plume_collider(res),
        "plume_wind" => Scene::plume_wind(res),
        _ => return Err(format!("unknown scene {name}").into()),
    })
}

/// 3 timed runs below 256³, one at 256³ (spec §1).
fn runs_for(res: u32) -> u32 {
    if res >= 256 { 1 } else { 3 }
}

/// One timed run: each frame's `eval_frame` plus a blocking wait, in ms,
/// frame 1 excluded. Returns the frame times and the pool's allocated bytes.
fn timed_run(gpu: &GpuContext, registry: &NodeRegistry, scene: &Scene) -> Res<(Vec<f64>, u64)> {
    let doc = scene.document();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(registry)?;
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let mut frames = Vec::new();
    let first = config.start_frame;
    for frame in first..first + scene.frames {
        let time = Time::at(frame, first, config.fps);
        let start = Instant::now();
        let evaluated = graph.eval_frame(gpu, &mut pool, &mut pipelines, &mut state, time, dims)?;
        gpu.wait()?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        evaluated.value.release_to(&mut pool);
        if frame > first {
            frames.push(ms);
        }
    }
    let bytes = pool.allocated_bytes();
    state.clear(&mut pool);
    Ok((frames, bytes))
}

/// The metrics run: after every frame, read the solver's density and
/// velocity back and measure them. Never timed.
fn metrics_run(gpu: &GpuContext, registry: &NodeRegistry, scene: &Scene) -> Res<Vec<FrameMetrics>> {
    let doc = scene.document();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(registry)?;
    let dx = scene.domain_size / f64::from(*scene.cells.iter().max().unwrap());
    let solid = scene.solid_mask();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let mut out = Vec::new();
    let first = config.start_frame;
    for frame in first..first + scene.frames {
        let time = Time::at(frame, first, config.fps);
        let evaluated = graph.eval_frame(gpu, &mut pool, &mut pipelines, &mut state, time, dims)?;
        evaluated.value.release_to(&mut pool);
        let density = state
            .get(SOLVER_NODE, solver::DENSITY)
            .ok_or("no solver density in state")?
            .as_field()?
            .read_back(gpu)?;
        let velocity = state
            .get(SOLVER_NODE, solver::VELOCITY)
            .ok_or("no solver velocity in state")?
            .as_vector_field()?;
        let velocity = Velocity::Faces([
            velocity.face(Axis::X).read_back(gpu)?,
            velocity.face(Axis::Y).read_back(gpu)?,
            velocity.face(Axis::Z).read_back(gpu)?,
        ]);
        out.push(measure(&Sample { cells: dims, dx, density: &density, velocity: &velocity, solid: &solid }));
    }
    state.clear(&mut pool);
    Ok(out)
}
```

Import `Axis` with the other `elements_core::gpu` items. `run_ember` sets
`blender: None` in its summary.

```rust
fn run_ember(name: &str, res: u32) -> Res<()> {
    let gpu = GpuContext::new_headless()?;
    let registry = elements_ember::registry();
    let scene = scene(name, res)?;
    let load_before = load_average();
    let runs = runs_for(res);
    let mut medians = Vec::new();
    let mut all = Vec::new();
    let mut peak = 0;
    for r in 0..runs {
        eprintln!("ember {name} {res}³: timed run {} of {runs}", r + 1);
        let (frames, bytes) = timed_run(&gpu, &registry, &scene)?;
        medians.push(median(&frames));
        all.extend(frames);
        peak = peak.max(bytes);
    }
    eprintln!("ember {name} {res}³: metrics run");
    let frames = metrics_run(&gpu, &registry, &scene)?;
    let from = (EMISSION_FRAMES[1] - 1) as usize;
    let mass: Vec<f64> = frames.iter().map(|f| f.mass).collect();
    let outflow: Vec<f64> = frames.iter().map(|f| f.outflow_rate).collect();
    let summary = RunSummary {
        solver: "ember".into(),
        scene: name.into(),
        resolution: res,
        runs,
        frame_ms_median: median(&medians),
        frame_ms_min: all.iter().copied().fold(f64::INFINITY, f64::min),
        frame_ms_max: all.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        peak_bytes: peak,
        load_before,
        load_after: load_average(),
        drift: drift(&mass, &outflow, 1.0 / scene.fps, from)[from..].to_vec(),
        frames,
    };
    write_summary(&results_dir(), &summary)?;
    Ok(())
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["ember", name, res] => run_ember(name, res.parse()?),
        _ => Err("usage: benchmark ember SCENE RES | mantaflow SCENE RES | report".into()),
    }
}
```

Check `drift`'s return shape against Task 4: it returns one value per frame
index; slice from `from` so `drift[0]` is frame 60.

- [ ] **Step 8: Smoke-run it at 64³ for `plume`**

Run: `cargo run --release -p elements-ember --example benchmark -- ember plume 64`
Expected: it writes `docs/bench/results/ember-plume-64.json` and `.csv`.
Check by eye: frame 1's mass is above 0, mass grows until frame 60, the
centroid rises, and drift at frame 60 is 0. Delete the two files afterwards;
Task 8 writes the real ones.

- [ ] **Step 9: `just check`, then commit.** Subject: "Add the Ember half of
the Mantaflow benchmark". The body says why readback happens in a separate
untimed run (spec §6.1).

---

### Task 6: Build the Mantaflow scene from the Ember scene

**Files:**
- Modify: `crates/elements-ember/src/bench/mod.rs` (`Scene::mantaflow_json`)
- Test: `crates/elements-ember/tests/bench.rs`
- Create: `tests/bench/mantaflow_scene.py`

**Interfaces:**
- Consumes: `tests/bench/mapping.py` (Task 3); `EMISSION_FRAMES`.
- Produces: `Scene::mantaflow_json(&self) -> serde_json::Value`, with exactly
  these keys: `name`, `domain_size`, `resolution` (the longest axis's cell
  count), `fps`, `frames`, `substeps`, `emitter` (`center`, `radius`,
  `density_rate`, `temperature_rate`, `active_frames`), `buoyancy_density`,
  `buoyancy_temperature`, `vorticity`, `density_dissipation`,
  `temperature_dissipation`, `wind` ([x, y, z] m/s²), `collider`
  (`null` or `{ "center": [..], "radius": .. }`), `boundaries` (six strings,
  `"wall"` or `"open"`, keyed `neg_x` … `pos_z`).
- The script's command line: `Blender --background --factory-startup --python tests/bench/mantaflow_scene.py -- SCENE_JSON OUT_DIR [--no-bake]`. It writes `OUT_DIR/cache/` (the Mantaflow cache) and `OUT_DIR/timings.json`:
  `{"frames": [{"frame": 1, "mtime_ns": ...}, ...], "blender": "5.2.2 LTS"}`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn the_mantaflow_json_carries_every_matched_parameter() {
    let s = Scene::plume_collider(64);
    let j = s.mantaflow_json();
    assert_eq!(j["name"], "plume_collider");
    assert_eq!(j["resolution"], 64);
    assert_eq!(j["frames"], 120);
    assert_eq!(j["emitter"]["active_frames"], serde_json::json!([1, 60]));
    assert_eq!(j["emitter"]["radius"], 0.2);
    assert_eq!(j["collider"]["radius"], 0.25);
    assert_eq!(j["boundaries"]["pos_z"], "open");
    assert_eq!(j["boundaries"]["neg_z"], "wall");
    assert_eq!(j["substeps"], s.solver.max_substeps);
    let w = Scene::plume_wind(64).mantaflow_json();
    assert_eq!(w["wind"], serde_json::json!([0.5, 0.0, 0.0]));
    assert!(Scene::plume(64).mantaflow_json()["collider"].is_null());
}
```

Check how `Boundaries` and `Face` serialise, and write the six strings from
them rather than duplicating the defaults. Check `SolverParams`'s field
names for vorticity and dissipation (`grep -n "pub " crates/elements-ember/src/solver.rs | head -40`).

- [ ] **Step 2: Run it; it fails to compile.**
- [ ] **Step 3: Implement** `mantaflow_json` with `serde_json::json!`.
- [ ] **Step 4: Run it; it passes. Prove it can fail:** write
  `"resolution": self.cells[0] / 2`. Record and restore.

- [ ] **Step 5: Write `tests/bench/mantaflow_scene.py`**

Base it on `probe_mantaflow.py`, with every value from the scene JSON and
every conversion through `mapping.py`:

```python
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build and bake the Mantaflow equivalent of an Ember bench scene.

Run: Blender --background --factory-startup --python tests/bench/mantaflow_scene.py -- SCENE_JSON OUT_DIR [--no-bake]

Fairness rules (piece 2 spec §5.3): no noise upres, no adaptive domain, fixed
timesteps equal to Ember's substep count, and an uncompressed 32-bit OpenVDB
cache. Parameter conversions live in mapping.py, with their sources.
"""

import json
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapping  # noqa: E402

SIDES = {
    "neg_x": "use_collision_border_left",
    "pos_x": "use_collision_border_right",
    "neg_y": "use_collision_border_front",
    "pos_y": "use_collision_border_back",
    "neg_z": "use_collision_border_bottom",
    "pos_z": "use_collision_border_top",
}


def build(sc: dict, cache_dir: str) -> bpy.types.Object:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, sc["frames"]
    scene.render.fps = int(sc["fps"])
    size = sc["domain_size"]
    dx = size / sc["resolution"]

    bpy.ops.mesh.primitive_cube_add(size=size, location=(size / 2,) * 3)
    domain = bpy.context.active_object
    s = domain.modifiers.new("Fluid", "FLUID")
    s.fluid_type = "DOMAIN"
    d = s.domain_settings
    d.domain_type = "GAS"
    d.resolution_max = sc["resolution"]
    d.use_adaptive_domain = False
    d.use_noise = False
    d.use_adaptive_timesteps = False
    d.timesteps_min = d.timesteps_max = sc["substeps"]
    d.cache_directory = cache_dir
    d.cache_type = "ALL"
    d.cache_data_format = "OPENVDB"
    d.openvdb_cache_compress_type = "NONE"
    d.openvdb_data_depth = "32"
    d.cache_frame_start, d.cache_frame_end = 1, sc["frames"]
    d.alpha, d.beta = mapping.buoyancy(sc["buoyancy_density"], sc["buoyancy_temperature"])
    d.vorticity = mapping.vorticity(sc["vorticity"], dx)
    d.use_dissolve_smoke = False
    for side, prop in SIDES.items():
        setattr(d, prop, sc["boundaries"][side] == "wall")

    e = sc["emitter"]
    bpy.ops.mesh.primitive_uv_sphere_add(radius=e["radius"], location=e["center"])
    flow_obj = bpy.context.active_object
    f = flow_obj.modifiers.new("Fluid", "FLUID")
    f.fluid_type = "FLOW"
    fs = f.flow_settings
    fs.flow_type = "SMOKE"
    fs.flow_behavior = "INFLOW"
    fs.density, fs.temperature = mapping.inflow(e["density_rate"], e["temperature_rate"], sc["fps"])
    first, last = e["active_frames"]
    for frame, on in ((first, True), (last, True), (last + 1, False)):
        fs.use_inflow = on
        fs.keyframe_insert("use_inflow", frame=frame)

    c = sc["collider"]
    if c is not None:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=c["radius"], location=c["center"])
        eff = bpy.context.active_object.modifiers.new("Fluid", "FLUID")
        eff.fluid_type = "EFFECTOR"
        eff.effector_settings.effector_type = "COLLISION"

    if any(sc["wind"]):
        strength, direction = mapping.wind(tuple(sc["wind"]))
        bpy.ops.object.effector_add(type="WIND", location=(size / 2,) * 3)
        wind = bpy.context.active_object
        wind.field.strength = strength
        wind.field.falloff_type = "NONE"
        wind.rotation_mode = "QUATERNION"
        wind.rotation_quaternion = mapping.rotation_to(direction)
    return domain
```

Use the attribute names Task 3 verified; the ones above are the expected
names and must be checked against the notes (for example `use_inflow`, the
collision-border names, and which local axis a wind field blows along,
which `mapping.rotation_to` encodes). Then:

```python
def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :]
    scene_json, out_dir = args[0], args[1]
    no_bake = "--no-bake" in args
    with open(scene_json) as fh:
        sc = json.load(fh)
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    domain = build(sc, cache_dir)
    if no_bake:
        return
    with bpy.context.temp_override(object=domain, active_object=domain):
        result = bpy.ops.fluid.bake_all()
    if "FINISHED" not in result:
        sys.exit(f"bake_all returned {result}")
    frames = []
    for n in range(1, sc["frames"] + 1):
        path = os.path.join(cache_dir, "data", f"fluid_data_{n:04d}.vdb")
        if not os.path.exists(path):
            sys.exit(f"missing cache frame {n}: {path}")
        frames.append({"frame": n, "mtime_ns": os.stat(path).st_mtime_ns})
    with open(os.path.join(out_dir, "timings.json"), "w") as fh:
        json.dump({"frames": frames, "blender": bpy.app.version_string}, fh, indent=1)


main()
```

Replace the cache path pattern with the one recorded in the notes.

- [ ] **Step 6: Check the script on `plume` at 32³**

```bash
S=<scratchpad>
cargo run -p elements-ember --example benchmark -- scene-json plume 32 > "$S/plume32.json"
```

For this, add a small mode to `benchmark.rs`,
`benchmark scene-json SCENE RES`, that prints `mantaflow_json()`. Task 8's
runner uses it too. Then:

```bash
"$B" --background --factory-startup --python tests/bench/mantaflow_scene.py -- "$S/plume32.json" "$S/manta32"
```

Expected: exit 0, 120 cache frames, and `timings.json`. Check that density
stops being added after frame 60: read frames 60, 61 and 70 with a temporary
`vdb_probe` (from Task 3's code) and confirm total density does not rise
after 61 by more than advection noise. Record the check in the ledger.

- [ ] **Step 7: `ruff format tests && ruff check tests`, `just check`, commit.**
Subject: "Build each bench scene's Mantaflow twin from the same definition".

---

### Task 7: Read a Mantaflow cache frame

**Files:**
- Create: `crates/elements-ember/src/mantaflow.rs` (and `pub mod mantaflow;` in `lib.rs`)
- Test: `crates/elements-ember/tests/mantaflow.rs`

**Interfaces:**
- Consumes: the grid names, value types, velocity location and index-space
  mapping recorded in `docs/bench/mantaflow-notes.md` (Task 3); `metrics::Velocity`.
- Produces:

```rust
pub struct CacheFrame {
    pub cells: FieldDims,
    pub density: Vec<f32>,
    pub velocity: Velocity,
}

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("{path}: {source}")]
    Open { path: PathBuf, source: std::io::Error },
    #[error("{path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("{path}: no grid named {grid}; found {found:?}")]
    MissingGrid { path: PathBuf, grid: String, found: Vec<String> },
    #[error("{path}: grid {grid} has a voxel at {index:?}, outside a {cells:?} domain")]
    OutOfDomain { path: PathBuf, grid: String, index: [i32; 3], cells: [u32; 3] },
}

/// Read one cache frame into x-fastest arrays at `cells`. Inactive voxels are
/// zero; tiles fill their whole extent (`VdbLevel::scale`).
pub fn read_frame(path: &Path, cells: FieldDims) -> Result<CacheFrame, CacheError>;
```

Check whether `thiserror` is already a workspace dependency; if not, write
`Display` and `Error` by hand instead of adding one.

- [ ] **Step 1: Write the failing tests**, using the fixture from Task 3.
Its dims and expected values come from the fixture's `README.md`:

```rust
use std::path::PathBuf;

use elements_core::gpu::FieldDims;
use elements_ember::mantaflow::{CacheError, read_frame};
use elements_ember::metrics::Velocity;

fn fixture() -> PathBuf {
    // Replace FILE with the fixture's file name from its README.
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/mantaflow_16/FILE"))
}

#[test]
fn a_cache_frame_reads_into_the_domain_layout() {
    let f = read_frame(&fixture(), FieldDims::new(16, 16, 16)).unwrap();
    assert_eq!(f.density.len(), 16 * 16 * 16);
    let total: f64 = f.density.iter().map(|&d| f64::from(d)).sum();
    assert!(total > 0.0, "the probe's emitter left density");
    // The emitter sits at (1, 1, 0.3) m in a 2 m domain: the densest cell is
    // near the domain's x/y centre and low in z.
    let (imax, _) = f.density.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    let (i, j, k) = (imax % 16, (imax / 16) % 16, imax / 256);
    assert!((6..=9).contains(&i) && (6..=9).contains(&j), "({i}, {j}, {k})");
    assert!(k <= 5, "({i}, {j}, {k})");
    match &f.velocity {
        Velocity::Faces(v) | Velocity::Centred(v) => {
            assert!(v[2].iter().any(|&w| w > 0.0), "the plume rises");
        }
    }
}

#[test]
fn a_smaller_domain_than_the_cache_is_an_error() {
    let err = read_frame(&fixture(), FieldDims::new(8, 8, 8)).err();
    assert!(matches!(err, Some(CacheError::OutOfDomain { .. })), "{err:?}");
}

#[test]
fn a_missing_file_is_an_error() {
    let err = read_frame(&fixture().with_file_name("nope.vdb"), FieldDims::new(16, 16, 16)).err();
    assert!(matches!(err, Some(CacheError::Open { .. })), "{err:?}");
}
```

Also add a test that a missing grid is `MissingGrid`. `mantaflow.rs` has
`pub const DENSITY_GRID` and `pub const VELOCITY_GRID` (names from the
notes) and `pub fn read_frame_with(path, cells, density_grid: &str,
velocity_grid: &str)`, which `read_frame` calls with the two constants. The
test calls `read_frame_with` with a density grid name that does not exist.

- [ ] **Step 2: Run them; they fail to compile.**

- [ ] **Step 3: Implement.** Open with `vdb_rs::VdbReader::new(BufReader)`.
Read density as `f32` and velocity as `[f32; 3]` (or as the type the notes
record). Map each iterated index to the domain with the index offset from the
notes. For each item, fill `scale³` cells starting at the index, where
`scale = level.scale() as i32`. Any cell outside `cells` is `OutOfDomain`.
Choose `Velocity::Faces` or `Velocity::Centred` from the notes. For faces,
the arrays have one more entry along their axis, and index n along that axis
is the top face. If Mantaflow stores it at index n, include it.

- [ ] **Step 4: Run the tests; they pass. Prove each can fail:**
1. Read into the transpose (`k + n * (j + n * i)`): the location assertion fails.
2. Skip tiles instead of filling them. If the fixture has no tiles, this
   mutation is equivalent. Record that, then use: drop the index offset,
   which must fail on location or on `OutOfDomain`.
3. Remove the bounds check: the smaller-domain test fails (it panics
   instead of returning the error).
4. Missing file: return `Parse` for an open error.
5. Missing grid: substitute a zero grid when the name is absent.

- [ ] **Step 5: `just check`, then commit.** Subject: "Read a Mantaflow cache
frame into the metrics' array layout".

---

### Task 8: Run everything, and write the results

**Files:**
- Modify: `crates/elements-ember/examples/benchmark.rs` (`mantaflow` and `report` modes)
- Modify: `crates/elements-ember/src/bench/report.rs` (`results_markdown`)
- Test: `crates/elements-ember/tests/bench.rs`
- Modify: `justfile` (`bench`)
- Create: `docs/bench/results.md`, `docs/bench/results/*.json`, `*.csv`
- Modify: `CLAUDE.md`, the piece 2 spec §6, the 2b-3 spec status

**Interfaces:**
- Consumes: everything above.
- Produces:

```rust
pub const SCENES: [&str; 3] = ["plume", "plume_collider", "plume_wind"];
pub const RESOLUTIONS: [u32; 3] = [64, 128, 256];
pub const SOLVERS: [&str; 2] = ["ember", "mantaflow"];

pub struct Context { pub machine: String, pub os: String, pub commit: String, pub blender: String, pub date: String }

/// The whole table, or the list of result files that are missing. Never a
/// partial table (spec §7).
pub fn results_markdown(summaries: &[RunSummary], ctx: &Context) -> Result<String, Vec<String>>;
```

- [ ] **Step 1: Write the failing tests** for `results_markdown`:

```rust
fn summary(solver: &str, scene: &str, res: u32) -> report::RunSummary { /* small literal, 120 frames of FrameMetrics built in a loop */ }

fn all() -> Vec<report::RunSummary> {
    let mut v = Vec::new();
    for solver in report::SOLVERS {
        for scene in report::SCENES {
            for res in report::RESOLUTIONS {
                v.push(summary(solver, scene, res));
            }
        }
    }
    v
}

#[test]
fn the_results_table_needs_every_run() {
    let ctx = report::Context { /* literal strings */ };
    let mut s = all();
    s.retain(|r| !(r.solver == "mantaflow" && r.scene == "plume_wind" && r.resolution == 256));
    let missing = report::results_markdown(&s, &ctx).unwrap_err();
    assert_eq!(missing, vec!["mantaflow-plume_wind-256".to_owned()]);
}

#[test]
fn the_results_table_has_one_section_per_scene_and_names_run_counts() {
    let ctx = report::Context { /* literal strings */ };
    let md = report::results_markdown(&all(), &ctx).unwrap();
    for scene in report::SCENES {
        assert!(md.contains(&format!("## `{scene}`")), "{scene}");
    }
    assert!(md.contains("1 run"), "256³ says it is one run");
    assert!(md.contains("median of 3"), "64³ and 128³ say three");
    assert!(md.contains(&ctx.commit) && md.contains(&ctx.blender));
}
```

Write the `summary` and `Context` literals in full.

- [ ] **Step 2: Run them; they fail. Step 3: Implement** `results_markdown`:
- **Header:** machine, OS, Ember commit, Blender version, date. Add a note
  that timings from a run whose load average was above 2 should be redone
  idle, and list any such runs by name.
- **Per scene, `## \`{scene}\``:** a table with one row per solver and
  resolution. Columns:
  - frame ms (median, min–max, and "median of 3" or "1 run");
  - peak MiB;
  - divergence RMS at frames 60 and 120, with the rule;
  - kinetic energy at 60 and 120;
  - vorticity at 60 and 120;
  - centroid and top at 60 and 120 (m);
  - drift at 120, absolute and as a percentage of mass at 60.
- **Footnotes:**
  - the outflow estimate is at frame resolution;
  - Ember's peak memory is the pool's allocated bytes, with the frame cache
    off;
  - Mantaflow's peak memory is resident memory minus the no-bake baseline.
- **Summary:** leave a `## Summary` heading followed by `_to be written by
  hand from the tables_`. Step 7 replaces the placeholder text with prose.

Prove each test can fail: (1) skip the completeness check; (2) always write
"median of 3". Record them.

- [ ] **Step 4: The Mantaflow mode** in `benchmark.rs`:

```rust
fn blender() -> String {
    std::env::var("BLENDER_BIN")
        .unwrap_or_else(|_| "/Applications/Blender.app/Contents/MacOS/Blender".to_owned())
}

/// Run the scene script under `/usr/bin/time -l`, returning peak resident
/// bytes. `-l` prints "maximum resident set size" in bytes on macOS.
fn run_blender(scene_json: &Path, out: &Path, no_bake: bool) -> Res<u64> { /* ... */ }
```

- Write `mantaflow_json()` to `<scratch>/{scene}-{res}/scene.json`, with
  the scratch directory under `target/bench/`.
- Run the baseline once with `--no-bake`.
- For each of `runs_for(res)` runs, clear the output directory and bake.
- If Blender exits non-zero, return an error carrying the scene, resolution
  and Blender's stderr.
- **Frame times:** differences of consecutive `mtime_ns` in `timings.json`,
  for frames 2 onwards. The median per run, then the median of the run
  medians.
- **Peak memory:** the maximum over the bake runs, minus the baseline.
- **Metrics:** after the last run, read every frame with
  `mantaflow::read_frame` and `measure` it with `scene.solid_mask()` and the
  same `dx` as Ember. Compute drift as in `run_ember`, then write the
  summary.

Add the `report` mode: read every `*.json` in `docs/bench/results/`, build
the `Context` (use `common::shell` for machine, OS and date,
`common::commit_label` for the commit, and the Blender version from any
Mantaflow summary's `blender` field; the Mantaflow mode fills that field from
`timings.json`), and write `docs/bench/results.md` only on `Ok`. On `Err`,
print the missing list and exit non-zero.

- [ ] **Step 5: The `just bench` recipe**

```make
# The Mantaflow benchmark (2b-3 spec). Takes about an hour or more, needs the
# real GPU and Blender, and is not part of `check`. Writes
# docs/bench/results/ per run, then docs/bench/results.md from a complete set.
bench scenes="plume plume_collider plume_wind" resolutions="64 128 256":
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --release -p elements-ember --example benchmark
    B=target/release/examples/benchmark
    for res in {{resolutions}}; do
      for scene in {{scenes}}; do
        "$B" ember "$scene" "$res"
        "$B" mantaflow "$scene" "$res"
      done
    done
    "$B" report
```

- [ ] **Step 6: Run it**

First check the load (`sysctl -n vm.loadavg`); wait for a 1-minute load under
2, as in Task 2. Then run `just bench scenes=plume resolutions=64` as a smoke
test. `report` fails with the missing list, which is expected. Check the two
`plume-64` summaries by eye:
- both solvers' mass rises to frame 60 and then flattens or falls;
- centroids rise;
- divergence after projection is small against the kinetic energy scale;
- neither solver has NaNs.

If anything looks wrong, fix it before the full run.

Then run `just bench` in the background, and wait for it without polling on
a short interval. It takes 30–90 minutes by Task 3's estimate. On failure,
rerun only the missing scene and resolution with the example directly; the
others' files are kept.

- [ ] **Step 7: Write the summary and commit the results**

Replace the `## Summary` placeholder in `docs/bench/results.md` with plain
prose, 150–300 words:
- where Ember is faster, and by how much;
- where Mantaflow keeps more detail (vorticity and kinetic energy retained,
  plume top), stated plainly when it is true (piece 2 §5.1);
- what drift says about each solver;
- which mappings limit the comparison (from the notes).

Do not claim more than the numbers show.

Update:
- `CLAUDE.md`'s status paragraph: 2b-3 is complete; 2b-3b (render and
  latency) is next. Add the 2b-3 spec, plan and `docs/bench/results.md` to
  the list.
- The 2b-3 spec's status line: "Complete."
- Piece 2 spec §6: a dated line under risk (k) if Task 2 did not already
  close it, and a pointer to the results.

`just check`, then commit the results, `justfile`, example, docs and report
module. Subject: "Benchmark Ember against Mantaflow on the three plume
scenes". The body gives the headline numbers and says why 256³ is one run.
