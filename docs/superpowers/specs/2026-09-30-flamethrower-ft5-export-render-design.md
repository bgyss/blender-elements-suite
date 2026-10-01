# FT5 — multi-grid VDB export and Blender render template

Date: 2026-09-30. Branch `flamethrower-ft5`, from main at `eeb5d8e` (FT4 merged).
Roadmap row: `docs/superpowers/specs/2026-09-30-flamethrower-napalm-roadmap-design.md` §4, FT5.

## 1. Goal and done-when

One bake of a document yields one `.vdb` per frame holding `density`, `flame`, `temperature`,
`fuel`, `char` and cell-centred `velocity_x/y/z`. A Blender script renders a frame from that file,
and a `just` recipe runs both on a small shack scene. The full shot, clip and reference comparison
are FT6.

Done when: one bake yields all grids; Blender renders a frame from them; the `just` recipe exists;
every test has been shown to fail under a single mutation; Metal-local and llvmpipe results are
recorded separately.

User decisions (2026-09-30): the recipe renders a small flamethrower-into-shack scene built from
FT0–FT4, not the `fire` benchmark scene; "one bake yields all grids" is a document `outputs` list
(approach A); Ember-only render.

## 2. Current state

- `elements_io::write_float_grid` writes exactly one `FloatGrid` per file. No multi-grid file, no
  `Vec3s` writer.
- A document has one `output` node. `elements-cli bake` writes one grid per run; the fire render
  bakes the solver twice (`density`, `flame`).
- The smoke solver has five output sockets: 0 density, 1 temperature, 2 velocity (a staggered
  `VectorField`), 3 flame, 4 char. **There is no fuel output.** Fuel is a state slot only.
- `examples/jet_shack.elements` (version 3) is the existing jet-and-shack document.
- `tests/bench/render_compare.py` builds Principled Volume materials, a flame material with a fixed
  1500 K blackbody, and placement via `placement.py`.

## 3. Design

### 3.1 Solver: fuel output

Add output socket 5, `fuel`, following `flame_output` and `char_output`: a copy of the fuel state
slot with fire on, zero without. The `wanted` array grows from five to six. With fire off, the
existing outputs and the fire-off guard hashes (`0x7fd45178bbd35c70`, `0xfaf6ca0296d5d74a`,
`0xfbcd_e02f_2e02_d31f` on Apple M1 Max) must not change. The node's output socket count and any
document that names socket indices are checked for breakage.

### 3.2 Document: `outputs`

The document gains an optional `outputs: [{node, socket, name}]` beside `output`. Version 3 becomes
version 4; versions 1–3 still load (a migration test in `tests/document.rs`, as the constant's doc
comment requires). The single-grid writer's file is byte-identical (pinned by a pre-FT5 fixture) and a document without `outputs` takes the unchanged single-grid branch; no end-to-end pre/post bake comparison exists.

`Document::from_json` rejects duplicate and empty names; `Document::into_graph` rejects a node or
socket out of range and a socket that is not a field (via `Graph::set_extra_outputs`). Any vector-field entry named `n` is written as `n_x`, `n_y`, `n_z` (the velocity field is `velocity`). Two entries
that would write the same grid name, counting a vector entry `velocity` as `velocity_x/_y/_z`, are
rejected by the bake before any GPU work (Task 5). Read-backs are sequential, so the existing
per-field `validate_for` check bounds each buffer.

### 3.3 Bake

`bake` evaluates the graph once per frame and reads back every `outputs` socket. Each frame of a
stateful graph is simulated once, not once per grid. For velocity, each cell's two faces per axis
are averaged on the CPU after read-back, giving three cell-centred scalar grids; no new kernels.

With `outputs`, the grids inside each file are named by the entries, and `--name` is only the file
stem: `{name}.{frame:04}.vdb`. Without `outputs`, `--name` is also the single grid's name, as today. A read-back
or write failure names the frame and the grid and deletes the partial file.

### 3.4 Writer

`elements_io::write_float_grids(path, &[GridSpec { name, values }], dims, voxel_size, background)`
writes several uncompressed `FloatGrid`s into one archive, all sharing one transform and dims. A
mismatched length returns `LengthMismatch`; an empty list or a duplicate name is an error.
`write_float_grid` becomes a one-grid call, and its existing files stay byte-identical (tested).
Read-back uses `vdb-rs` as the oracle, as in the existing tests.

Size: eight grids at 128³ are about 67 MB uncompressed per frame. Acceptable for the small scene;
FT6 records the cost for a clip.

### 3.5 Blender template

`tests/bench/render_shack.py`, GPL, run as
`Blender --background --factory-startup --python-exit-code 1 --python tests/bench/render_shack.py -- BAKE_DIR NAME FRAME SCENE_JSON OUT_DIR`
(SCENE_JSON is the `.elements` scene the bake used).
It reuses `placement.py` and the `render_compare` helpers; Cycles 64 spp, seed 0, no denoise, Metal
GPU when present.

1. **Contract check.** Loads the frame's file as one Volume object and exits nonzero if
   `volume.grids` lacks any expected name.
2. **Beauty material.** One Principled Volume: density from `density`; emission strength
   `flame` × strength; emission colour from blackbody driven by the real `temperature` grid through
   a Map Range from solver temperature to Kelvin. The Kelvin endpoints and strength are constants
   chosen by eye and recorded.
3. **Velocity.** The volume's `velocity_grid` is set to the `velocity` prefix. Checked on Blender 5.2.2 LTS: all 8
   grids load from the multi-grid file and `velocity_grid = "velocity"` is accepted (resolving
   `velocity_x/_y/_z`); a render with motion blur was NOT tested. If it fails, a `Vec3s` writer becomes part of FT5.
4. **Fuel and char passes.** Cycles surface shaders cannot sample a volume grid, so char cannot
   shade the shack mesh here. `fuel` and `char` each render as a separate emission-only still from
   the same camera by swapping the volume material: `beauty`, `fuel`, `char` per frame. Mesh-lookup
   char shading is FT6.
5. **Shack proxy.** The real planks mesh from the scene JSON, as a grey mesh (not grey boxes).
6. **Recipe.** `just render-shack` bakes `examples/jet_shack_render.elements`
   (128x64x64 over 2.0 m, not cut down; an `outputs` list names all grids, one `bake` call),
   renders one frame (default 45, not 60: the jet is active for frames 5-40 and the fire fades by 50, so frame 60 is nearly empty), and writes PNGs to `docs/bench/render-shack/`. Real GPU and Blender; not in
   `just check`.

## 4. Error handling

- Invalid `outputs` are rejected on load or in `into_graph` (§3.2). Writer and bake errors as in §3.3–3.4.
- The render script exits nonzero on a missing grid, a missing frame file, or a failed image check.
  PNG pixel values cannot be non-finite, so the script instead checks lit and warm pixel counts
  against floors: a warm-pixel floor on beauty; lit pixels on fuel and char; char within the shack's
  projected x extent plus or minus 4 voxels; fuel at least 2x as wide as char.
- `elements-io` and `elements-cli` keep `forbid(unsafe_code)` and `Apache-2.0 OR MIT`; the Blender
  script is GPL and lives in `tests/`; no code is copied between them. No new wgpu features.

## 5. Testing

Each test is proven to fail under one mutation, restored, with the real output recorded.

- **Rust, `elements-io`:** every grid of a multi-grid file round-trips through `vdb-rs`; a
  one-grid `write_float_grids` equals `write_float_grid` byte for byte; length mismatch, empty list
  and duplicate names error.
- **Rust, core:** `outputs` validation cases; version-3 document migration; a document without
  `outputs` takes the unchanged single-grid branch (the single-grid writer's file is byte-identical,
  pinned by a pre-FT5 fixture; no end-to-end pre/post bake comparison exists).
- **Rust, `elements-ember`:** the fuel output equals the fuel slot with fire on and is zero with it
  off; the existing fire-off hashes are unchanged.
- **Rust, `elements-cli`:** one bake evaluation per frame yields every named grid; velocity
  averaging matches a hand-computed staggered field.
- **Python, no Blender, in `py-test`:** the expected-grid contract, the Kelvin map endpoints, output
  file names.
- **Blender, inside the recipe:** the beauty still has warm pixels (R > B) near the flame; the
  `fuel` and `char` stills are nonzero only where their grids are; each shown to fail under a
  single mutation (constant temperature, swapped grid name). As implemented, the checks are the ones
  in §4. The temperature to Kelvin map is the one mutation they do not trip (reading `density` for
  temperature goes uncaught); it is covered by `kelvin()`'s unit test and by eye.

llvmpipe CI covers the Rust and pure-Python parts. The Blender render is Metal-local and is
recorded as such.

## 6. Out of scope

Mesh-lookup char shading; the full shot, clip and reference comparison (FT6); a `Vec3s` writer
unless §3.5(3) fails; heat-wave distortion, sparks and embers; any Mantaflow side of the render.

## 7. Risks

1. Velocity naming in Blender (§3.5(3)): checked on Blender 5.2.2 LTS, all 8 grids load and `velocity_grid = "velocity"` is accepted (resolving `velocity_x/_y/_z`); a render with motion blur was NOT tested.
2. The blackbody mapping is chosen by eye, so its colour is a judgement and is recorded as one.
3. File size grows with grid count; FT6 records the clip cost.
4. Adding solver output 5 touches a node interface the benchmark and examples use; every consumer
   of output sockets is checked in the plan.
