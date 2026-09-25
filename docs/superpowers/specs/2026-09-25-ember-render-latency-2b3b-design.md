# Ember Piece 2b-3b — Render Comparison and Latency Design

**Date:** 2026-09-25
**Status:** Design approved; plan not yet written.
**Parent:** `2026-09-21-ember-solver-design.md` (piece 2), §5.4, whose
visual and interactivity items 2b-3 deferred. Follows 2b-3c
(`2026-09-24-ember-solver-quality-2b3c-design.md`).
**Branch base:** `main` at 8c05778 (PR #9 merged).

## 1. Goal and scope

Close piece 2 §5.4's last two items. User decisions, 2026-09-25:

- **Latency at the engine**, measured the same way for both solvers, not
  through Blender's UI (whose Live path today re-bakes a `.vdb` per frame and
  would measure that workaround).
- **Still frames** for the render comparison: every bench scene at 128³ at
  frames 30, 60 and 90, and `plume` at 256³.

**Out of scope:** latency through the Blender UI (after the fast viewport);
animated comparisons; any solver change.

## 2. Latency

Piece 2 §5.4: "latency from a parameter change to the first updated frame
visible, at 128³; for Mantaflow, the time to re-bake up to the current
frame."

- **Measured:** the time from "a parameter changed" to "frame N ready", at
  N = 1, 24, 60 and 120, in `plume`, `plume_collider` and `plume_wind` at
  128³, with the 1-minute load average before and after each scene.
- **Ember:** a `latency` mode in `crates/elements-ember/examples/benchmark.rs`
  keeps one process, GPU device and pipeline cache warm, as a live daemon
  does. Each run builds the scene's document with the emitter's
  `density_rate` changed (so nothing can be reused), then times graph
  construction plus `eval_frame` for frames 1..N, ending with a blocking wait.
  The median of 5 runs per N is reported, with the time to the first frame.
- **Mantaflow:** one Blender process stays open with the scene built
  (`tests/bench/mantaflow_scene.py`'s build, reused). Each run changes the
  flow's density, frees the cache, sets `cache_frame_end = N` and calls
  `bake_all`; the time runs from the bake call to frame N's data file being
  written. Blender's startup is not counted. The median of 5 runs per N for
  N ≤ 24, and of 3 above.
- **Output:** `docs/bench/latency.md`: per scene, a table of N against Ember
  and Mantaflow seconds and their ratio, the machine, commit, Blender version,
  date and load. Reported, never a pass/fail test.

## 3. Render comparison

- **Inputs:** Ember's density VDBs from `elements-cli bake`, and Mantaflow's
  caches from the bench script, both baked at the current commit, for each
  scene at 128³ and `plume` at 256³.
- **`tests/bench/render_compare.py`** (GPL-3.0-or-later, headless Blender):
  for each scene and frame (30, 60, 90), load both VDBs as Volume objects
  placed in their domain's space, side by side (Ember left, Mantaflow right).
  Mantaflow's VDB puts voxel centres at i·dx, half a cell off the domain
  (`docs/bench/mantaflow-notes.md`), so its object is shifted by half a cell.
  One camera, a key and a fill light, a world colour, the same Principled
  Volume shader and density scale on both, Cycles at a fixed sample count and
  seed, about 960×540, with "Ember" and "Mantaflow" labels in the image.
- **`just bench-render`** writes the PNGs into `docs/bench/render/` and a
  `docs/bench/render/README.md` that lays them out as a contact sheet, with the
  machine, commits, Blender version and date.
- **Judgement:** the controller describes what the images show; the user
  records the verdict in the README's note line.

## 4. Records

Close piece 2 §5.4's visual and interactivity items in the piece 2 spec; add
a pointer from `docs/bench/results.md` to `latency.md` and `render/`; update
CLAUDE.md's status and the roadmap (2b-3b done).

## 5. Testing

- The latency mode's document change is tested: the changed document differs
  from the scene's only in the emitter's `density_rate` (proven able to fail
  by a single-change mutation).
- `render_compare.py`'s placement arithmetic (domain placement, the half-cell
  shift) lives in pure functions in `tests/bench/`, tested by `py-test` and
  proven able to fail.
- The runs themselves are measurements, not tests.

## 6. Risks

- **(a) Load.** As before; loads are recorded and flagged.
- **(b) Density scale.** The same shader density on both solvers only makes
  sense because their emitted masses match (within 3.5% over frames 12–24,
  `results.md`); the README says so.
- **(c) Render time.** 12 images of two volumes at 256³ at most; if Cycles is
  slow, lower the sample count, not the image count.
