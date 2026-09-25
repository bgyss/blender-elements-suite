# Ember Piece 2b-3b — Render Comparison and Latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the latency from a parameter change to frame N for Ember and
Mantaflow, and render both solvers' volumes side by side with one Cycles
setup.

**Architecture:** A `latency` mode in the `benchmark` example times Ember in a
warm process; a `--latency` mode in the Mantaflow scene script times re-bakes
in a warm Blender. `render_compare.py` loads both solvers' VDBs into one
headless Blender scene per frame and renders them side by side, with the
placement arithmetic in a pure, tested module.

**Tech Stack:** Rust (the `benchmark` example, `elements-cli bake`), Blender
5.2.2 (`bpy`, Cycles), stdlib Python for the pure module and its test.

**Spec:** `docs/superpowers/specs/2026-09-25-ember-render-latency-2b3b-design.md`

## Global Constraints

- `just check` passes before every commit (lint, Rust tests, `py-test`).
- `crates/*` is Apache-2.0 OR MIT; `tests/bench/` is GPL-3.0-or-later with
  SPDX headers. No Blender or Mantaflow source is copied.
- Blender is `/Applications/Blender.app/Contents/MacOS/Blender`, or
  `$BLENDER_BIN`, always with `--python-exit-code 1`.
- **Every new test is proven able to fail** by a single-change mutation, with
  the real output recorded in the ledger (`.superpowers/sdd/progress.md`).
- Measurements are never pass/fail tests. Record the 1-minute load average
  before and after each run, and flag any run above 2.
- Never set `WGPU_BACKEND` locally. Plain git (no `jj`). Commits: imperative
  subject, why-body, ending with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TSRpZQEY9fnKQqHLh5WHpU
  ```
- Scratch output goes under `target/bench/` (ignored), never in `docs/` except
  the committed results named below.

## File structure

| File | Responsibility | Task |
|---|---|---|
| `crates/elements-ember/src/bench/mod.rs` | `Scene::with_density_rate` for the changed document | 1 |
| `crates/elements-ember/examples/benchmark.rs` | `latency SCENE RES` and `document SCENE RES` modes | 1, 3 |
| `tests/bench/mantaflow_scene.py` | `--latency N,...` mode | 2 |
| `crates/elements-ember/examples/latency_report.rs` or a `latency-report` mode | writes `docs/bench/latency.md` | 2 |
| `justfile` | `bench-latency`, `bench-render` | 2, 3 |
| `tests/bench/placement.py`, `tests/bench/test_placement.py` | pure placement arithmetic | 3 |
| `tests/bench/render_compare.py` | the headless render | 3 |
| `docs/bench/latency.md`, `docs/bench/render/*` | results | 2, 3 |

---

### Task 1: Ember latency

**Files:**
- Modify: `crates/elements-ember/src/bench/mod.rs`, `crates/elements-ember/examples/benchmark.rs`
- Test: `crates/elements-ember/tests/bench.rs`

**Interfaces:**
- Produces: `Scene::with_density_rate(self, rate: f32) -> Scene` (sets
  `emitter.density_rate`); `benchmark latency SCENE RES` writes
  `docs/bench/results/latency-ember-{scene}-{res}.json`:

```json
{ "solver": "ember", "scene": "plume", "resolution": 128, "commit": "…",
  "load_before": 3.1, "load_after": 3.4,
  "points": [ { "frame": 1, "runs_s": [..5..], "median_s": 0.071, "first_frame_s": 0.071 }, … ] }
```

- [ ] **Step 1: Failing test** in `tests/bench.rs`:

```rust
/// The latency run's document must differ from the scene's only in the
/// emitter's density rate, so nothing the engine caches can be reused and
/// nothing else about the scene changes.
#[test]
fn the_latency_document_changes_only_the_density_rate() {
    for scene in [Scene::plume(32), Scene::plume_collider(32), Scene::plume_wind(32)] {
        let base = serde_json::to_value(scene.document()).unwrap();
        let changed = serde_json::to_value(scene.clone().with_density_rate(1.1).document()).unwrap();
        assert_ne!(base, changed);
        let mut patched = base.clone();
        patched["nodes"][0]["params"]["density_rate"] = serde_json::json!(1.1);
        assert_eq!(patched, changed, "{}", scene.name);
    }
}
```

Check that node 0 is the emitter in `Scene::document` before relying on the
index; if not, find it by kind.

- [ ] **Step 2: Run; it fails to compile.**
- [ ] **Step 3: Implement** `with_density_rate`, and the `latency` mode:
  one `GpuContext`, one `PipelineCache` and one registry for the whole run;
  a warm-up (build and evaluate the unchanged scene to frame 24 once, then
  discard). For each N in 1, 24, 60, 120 and each of 5 runs: rate =
  1.0 + 0.01·(run + 1) (a different document each run), start the clock,
  `doc.into_graph`, a fresh `FieldPool` and `StateStore`, `eval_frame` for
  frames 1..=N, `gpu.wait()`, stop; record `first_frame_s` on N = 1. Medians
  per N. Record load before and after. Frame 1's timing includes graph
  construction, which is the point.
- [ ] **Step 4: Run the test; pass. Mutation:** change a second parameter in
  `with_density_rate` (e.g. temperature_rate); record the failure.
- [ ] **Step 5: Smoke-run** `cargo run --release -p elements-ember --example
  benchmark -- latency plume 64` and check the JSON by eye (medians grow
  roughly linearly with N). Move the file out of `docs/` afterwards.
- [ ] **Step 6: `just check`, commit** "Time Ember from a parameter change to frame N".

---

### Task 2: Mantaflow latency and the latency table

**Files:**
- Modify: `tests/bench/mantaflow_scene.py` (a `--latency N,...` mode)
- Modify: `crates/elements-ember/examples/benchmark.rs` (`mantaflow-latency SCENE RES` and `latency-report` modes)
- Modify: `justfile` (`bench-latency`)
- Create: `docs/bench/latency.md` (written by the run)

- [ ] **Mantaflow's mode.** `mantaflow_scene.py -- SCENE_JSON OUT_DIR
  --latency 1,24,60,120 --runs 5,5,3,3` builds the scene once (the existing
  `build`), then for each N and run: set the flow's density to base ×
  (1 + 0.01·(run + 1)) through the mapping's density (so each bake is new),
  free the bake (`bpy.ops.fluid.free_all` with the domain as context), set
  `cache_frame_end = N`, then time from immediately before `bake_all` to the
  moment frame N's data file exists (the bake call returns after it; take the
  file's mtime minus the start as well, and report the later of the two).
  Write `OUT_DIR/latency.json` in the same shape as Ember's, with
  `"solver": "mantaflow"` and `"blender"`. Restore `cache_frame_end`
  afterwards. Refuse `substeps != 1` as the script already does.
- [ ] **`benchmark mantaflow-latency SCENE RES`** runs Blender with that mode
  (as `run_mantaflow` runs a bake: `--python-exit-code 1`, stderr tail on
  failure) and copies `latency.json` to
  `docs/bench/results/latency-mantaflow-{scene}-{res}.json`.
- [ ] **`benchmark latency-report`** reads every `latency-*.json` and writes
  `docs/bench/latency.md`: machine, OS, commits (refuse if they differ, as
  `report` does), Blender version, date, loads with the >2 flag, then one table
  per scene:

  | N | Ember s (median) | Mantaflow s (median) | Mantaflow / Ember |
  |---|---|---|---|

  plus Ember's time to its first frame, and a note that Mantaflow's time is a
  re-bake in an open Blender with its cache freed, excluding startup, and that
  frame time includes Mantaflow writing its cache (as in `results.md`).
  Pure formatting goes in `bench/report.rs` with a unit test (fail first,
  mutate: swap the ratio's order).
- [ ] **`just bench-latency`** builds the example and runs, for each of the
  three scenes at 128³, `latency` then `mantaflow-latency`, then
  `latency-report`.
- [ ] **Run it** (load check first; the user has said to run under load when
  the machine will not idle, flagged). Commit `docs/bench/latency.md` and the
  six JSON files with the code: "Measure latency from a parameter change to frame N".

---

### Task 3: The side-by-side render

**Files:**
- Create: `tests/bench/placement.py`, `tests/bench/test_placement.py` (and add it to `py-test`)
- Create: `tests/bench/render_compare.py`
- Modify: `crates/elements-ember/examples/benchmark.rs` (`document SCENE RES` prints the scene's `.elements` document)
- Modify: `justfile` (`bench-render`)
- Create: `docs/bench/render/*.png`, `docs/bench/render/README.md`

**Placement (pure).** Both solvers' grids cover the domain `[0, L]³` with
`dx = L / n`, cells indexed from 0. Where each VDB's index (i, j, k) sits in
world space depends on its transform:

- Ember's `elements-cli bake --voxel-size dx` writes a transform whose voxel
  centres are at `i·dx` (check `crates/elements-io`'s writer and record what
  it does); Mantaflow's puts voxel centres at `i·dx` too
  (`docs/bench/mantaflow-notes.md`, "Index space"). The true cell centre is
  `(i + 0.5)·dx`, so each Volume object is translated by `+0.5·dx` on every
  axis (confirm for Ember from the writer; if Ember's writer already includes
  the half cell, its offset is 0).
- The two domains sit side by side: Ember at x offset 0, Mantaflow at
  `L + gap` with `gap = 0.25·L`.

```python
def cell_offset(dx: float, writer_centres_at_index: bool) -> float
def object_location(slot: int, domain_size: float, dx: float, half_cell: bool, gap_frac: float = 0.25) -> tuple[float, float, float]
def camera_for(domain_size: float, gap_frac: float = 0.25, aspect: float = 16 / 9) -> dict   # location, rotation, ortho_scale or lens
```

- [ ] **Test** `tests/bench/test_placement.py`: the half-cell offset is
  `0.5·dx` when centres are at the index and 0 otherwise; slot 1 sits at
  `L + gap` on x; the camera frames both domains (both x extents inside the
  view). Prove each can fail (e.g. use `dx` instead of `0.5·dx`). Run by
  `py-test`.
- [ ] **Bakes.** For each scene at 128³, and `plume` at 256³: Ember with
  `benchmark document SCENE RES > target/bench/render/SCENE-RES.elements` and
  `elements-cli bake … --frames 30-90 --voxel-size <dx> --out
  target/bench/render/ember/SCENE-RES/` (only frames 30, 60 and 90 are used);
  Mantaflow with the existing `mantaflow_scene.py` bake into
  `target/bench/render/mantaflow/SCENE-RES/` (frames 1–90).
- [ ] **`render_compare.py -- EMBER_DIR MANTAFLOW_DIR SCENE RES OUT_DIR`**
  (headless, GPL): for frames 30, 60, 90, build a scene from factory
  settings: two Volume objects (Ember's `density.00NN.vdb`, Mantaflow's
  `data/fluid_data_00NN.vdb`, grid `density`) at `placement.object_location`;
  the same Principled Volume material on both (density 1.0, colour white,
  absorption off, no emission); a sun key light and a soft fill; a mid-grey
  world; the camera from `placement.camera_for`; Cycles, 64 samples, seed 0,
  denoiser off, 960×540; text objects "Ember" and "Mantaflow" under each
  domain, and a caption with scene, resolution and frame. Render to
  `OUT_DIR/{scene}-{res}-f{frame:03}.png`.
- [ ] **`just bench-render`** runs the bakes and renders (idempotent: skips
  bakes whose files exist), then writes `docs/bench/render/README.md`: the
  machine, Ember commit, Blender version, date, the render settings, a note
  that one density scale is fair because emitted masses match within 3.5%
  over frames 12–24 (`results.md`), one section per scene with its three
  images, and a `Verdict (recorded by the user): _pending_` line.
- [ ] **Run it**, check the images exist and are non-empty, commit the PNGs,
  README and code: "Render Ember and Mantaflow side by side".

---

### Task 4: Judgement and records

- [ ] The controller views each PNG and writes a factual description per
  scene into the README's notes (shape, detail, height, where the smoke has
  gone), without a verdict.
- [ ] Bring the images and description to the user; record their verdict in
  the README.
- [ ] Records: close piece 2 §5.4's visual and interactivity items in
  `docs/superpowers/specs/2026-09-21-ember-solver-design.md` with pointers;
  add links from `docs/bench/results.md` to `latency.md` and `render/`; set the
  2b-3b spec's status to Complete; update CLAUDE.md's status paragraph and doc
  list, and the roadmap (workstream 1: 2b-3b done).
- [ ] `just check`, commit "Record the render verdict and close piece 2's benchmark".
