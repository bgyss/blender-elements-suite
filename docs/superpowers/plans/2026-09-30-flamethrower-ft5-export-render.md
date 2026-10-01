# FT5 — Multi-grid VDB Export and Blender Render Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One bake of a document writes one multi-grid `.vdb` per frame (`density`, `flame`, `temperature`, `fuel`, `char`, `velocity_x/y/z`), and a Blender script plus `just render-shack` render a half-size flamethrower-into-shack frame from it.

**Architecture:** The graph gains "extra outputs" (sockets kept and returned beside the primary result). The document gains an `outputs` list naming them (version 4). The smoke solver gains output socket 5, `fuel`. `elements-io` gains a multi-grid writer. `bake` evaluates once per frame, expands the staggered velocity into cell-centred scalar grids on the CPU, and writes one file. `tests/bench/render_shack.py` loads the file into one Cycles Volume and renders `beauty`, `fuel` and `char` stills.

**Tech Stack:** Rust (wgpu, serde, vdb-rs as test oracle), Python 3.11 + Blender 5.x `bpy`, `just`, cargo-nextest.

**Spec:** `docs/superpowers/specs/2026-09-30-flamethrower-ft5-export-render-design.md`

## Global Constraints

- `elements-io`, `elements-cli`, `elements-core`, `elements-ember`: `Apache-2.0 OR MIT`, `#![forbid(unsafe_code)]`. The Blender script is `GPL-3.0-or-later` and lives in `tests/bench/`. No code is copied between them.
- `required_features` stays `wgpu::Features::empty()`. No new kernels (velocity averaging is CPU-side).
- Scalar fields are `R32Float`. Crate manifests use `dep.workspace = true`.
- A document without `outputs` bakes byte-identically to today's single-grid file.
- With fire off, the solver's existing guard hashes must not change: `0x7fd45178bbd35c70` (mgpcg), `0xfaf6ca0296d5d74a` (gauss_seidel), `0xfbcd_e02f_2e02_d31f` (FT4 surface), on "Apple M1 Max".
- **Never set `WGPU_BACKEND` locally.** Metal is the local backend.
- Every test is proven to fail under ONE mutation (restore, record the real failure text in the commit body). A compound mutation proves nothing.
- Python: Ruff, line length 100, target 3.11. `just check` must pass before each commit.
- Commits: plain imperative subject, body explains why, end with
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_016jxQnAfsTRPo8xRwrHJx4y`. Plain `git`, never `jj`. Do not push.
- Blender: `/Applications/Blender.app/Contents/MacOS/Blender` (override with `BLENDER_BIN`).

## Review Focus

1. **Two outputs naming one grid, or a scalar named `velocity_x` beside a vector named `velocity`:** must fail with the colliding name, before any GPU work (Task 5 test).
2. **An `outputs` entry pointing at the primary result socket, at a missing node/socket, or at a `Scalar` socket:** a typed error, never a panic or a blank grid (Tasks 2–3 tests).
3. **Fire off (no fuel input connected) with `fuel` and `char` exported:** grids are written, all zero, not missing (Task 4 test).
4. **A bake starting at frame N > start:** the extras come from frame N, not from a skipped frame (Task 2 timeline test).
5. **Old (v1–v3) documents:** still load and bake exactly as before (Task 3 test).

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/elements-io/src/vdb/tree.rs` | `GridSpec`, `write_float_grids`; `write_float_grid` becomes a wrapper |
| `crates/elements-io/src/vdb/writer.rs` | `write_archive_header_with_count` |
| `crates/elements-io/src/lib.rs` | `IoError::{NoGrids, DuplicateGrid}`, re-exports |
| `crates/elements-io/tests/vdb_multi.rs` | multi-grid round trip, pinned single-grid bytes, errors |
| `crates/elements-core/src/graph/mod.rs` | `Graph::set_extra_outputs`, `Evaluated.extras`, `Evaluated::release_to` |
| `crates/elements-core/src/graph/timeline.rs` | release skipped frames' extras |
| `crates/elements-core/src/graph/document.rs` | `DocOutput`, `Document.outputs`, version 4, name validation |
| `crates/elements-core/tests/extra_outputs.rs` | graph extras |
| `crates/elements-core/tests/document.rs` | `outputs` and migration tests |
| `crates/elements-ember/src/solver.rs` | output socket 5 `fuel` |
| `crates/elements-ember/tests/fire.rs` | fuel output tests |
| `crates/elements-ember/tests/jet_shack.rs` | `render_document`, `examples/jet_shack_render.elements` |
| `crates/elements-cli/src/grids.rs` | velocity averaging, grid-name expansion |
| `crates/elements-cli/src/bake.rs` | multi-grid bake |
| `crates/elements-cli/tests/cli.rs` | multi-grid bake test |
| `tests/bench/shack_layout.py`, `test_shack_layout.py` | pure layout/contract/pixel-stat helpers, no Blender |
| `tests/bench/render_shack.py` | Blender scene script |
| `justfile`, `docs/bench/render-shack/`, `AGENTS.md` | recipe, record, status |

---

### Task 1: Multi-grid VDB writer

**Files:**
- Modify: `crates/elements-io/src/vdb/writer.rs` (`write_archive_header`, ~line 169)
- Modify: `crates/elements-io/src/vdb/tree.rs` (`write_float_grid`, ~line 152)
- Modify: `crates/elements-io/src/vdb/mod.rs`, `crates/elements-io/src/lib.rs`
- Create: `crates/elements-io/tests/vdb_multi.rs`, `crates/elements-io/tests/fixtures/single_grid_4x3x2.vdb`

**Interfaces:**
- Produces: `pub struct GridSpec<'a> { pub name: &'a str, pub values: &'a [f32] }`;
  `pub fn write_float_grids(path: &Path, grids: &[GridSpec<'_>], dims: [u32; 3], voxel_size: f64, background: f32) -> Result<(), IoError>`;
  `IoError::NoGrids`, `IoError::DuplicateGrid { name: String }`; `write_archive_header_with_count(w, uuid, grid_count: u32)`. `write_float_grid`'s signature and output bytes are unchanged.

- [ ] **Step 1: Pin today's single-grid bytes before touching the writer**

Create `crates/elements-io/tests/vdb_multi.rs` with only the fixture pieces:

```rust
//! FT5: several grids in one OpenVDB file, read back with `vdb-rs`.

use std::collections::HashMap;
use std::path::Path;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/single_grid_4x3x2.vdb"
);

fn ramp(n: usize, offset: f32) -> Vec<f32> {
    (0..n).map(|i| offset + i as f32 * 0.25).collect()
}

/// Writes the fixture from the single-grid writer as it was before FT5.
/// Run once on unchanged code:
/// `cargo test -p elements-io --test vdb_multi -- --ignored write_single_grid_fixture`.
#[test]
#[ignore = "writes tests/fixtures/single_grid_4x3x2.vdb"]
fn write_single_grid_fixture() {
    std::fs::create_dir_all(Path::new(FIXTURE).parent().unwrap()).unwrap();
    elements_io::write_float_grid(
        Path::new(FIXTURE),
        "density",
        &ramp(24, 0.0),
        [4, 3, 2],
        0.1,
        0.0,
    )
    .unwrap();
}

#[test]
fn a_single_grid_file_is_byte_identical_to_the_pre_ft5_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("one.vdb");
    elements_io::write_float_grid(&path, "density", &ramp(24, 0.0), [4, 3, 2], 0.1, 0.0).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(FIXTURE).expect("fixture committed"),
        "write_float_grid's bytes changed"
    );
}
```

- [ ] **Step 2: Generate the fixture on unchanged code and commit it with the test**

Run: `cargo test -p elements-io --test vdb_multi -- --ignored write_single_grid_fixture`
Then: `cargo nextest run -p elements-io --test vdb_multi` — Expected: PASS (1 test).

```bash
git add crates/elements-io/tests && git commit -m "Pin the single-grid VDB bytes before the multi-grid writer" \
  -m "FT5 refactors write_float_grid into a one-grid call of a multi-grid writer. The fixture was written by the unchanged writer, so any byte change in a single-grid file now fails a test." \
  -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016jxQnAfsTRPo8xRwrHJx4y"
```

- [ ] **Step 3: Write the failing multi-grid tests**

Append to `vdb_multi.rs`:

```rust
fn read_active(path: &Path, name: &str) -> HashMap<(i32, i32, i32), f32> {
    let file = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let mut reader = vdb_rs::VdbReader::new(file).unwrap();
    let grid = reader.read_grid::<f32>(name).unwrap();
    grid.iter()
        .map(|(c, v, _)| ((c.x as i32, c.y as i32, c.z as i32), v))
        .collect()
}

#[test]
fn every_grid_of_a_multi_grid_file_round_trips() {
    // Asymmetric dims cross leaf borders on every axis.
    let dims = [9u32, 5, 3];
    let n = 9 * 5 * 3;
    let (a, b, c) = (ramp(n, 0.0), ramp(n, 100.0), ramp(n, -50.0));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.vdb");
    elements_io::write_float_grids(
        &path,
        &[
            elements_io::GridSpec { name: "density", values: &a },
            elements_io::GridSpec { name: "flame", values: &b },
            elements_io::GridSpec { name: "temperature", values: &c },
        ],
        dims,
        0.1,
        0.0,
    )
    .unwrap();

    let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let reader = vdb_rs::VdbReader::new(file).unwrap();
    let mut names = reader.available_grids();
    names.sort();
    assert_eq!(names, ["density", "flame", "temperature"]);

    for (name, values) in [("density", &a), ("flame", &b), ("temperature", &c)] {
        let voxels = read_active(&path, name);
        assert_eq!(voxels.len(), n, "{name}");
        for z in 0..3i32 {
            for y in 0..5i32 {
                for x in 0..9i32 {
                    let linear = (z as usize * 5 + y as usize) * 9 + x as usize;
                    assert_eq!(voxels[&(x, y, z)], values[linear], "{name} ({x}, {y}, {z})");
                }
            }
        }
    }
}

#[test]
fn an_empty_grid_list_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let err = elements_io::write_float_grids(&dir.path().join("e.vdb"), &[], [2, 2, 2], 0.1, 0.0)
        .unwrap_err();
    assert!(matches!(err, elements_io::IoError::NoGrids), "{err:?}");
}

#[test]
fn a_duplicate_grid_name_is_an_error_and_writes_no_file() {
    let v = vec![0.0f32; 8];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.vdb");
    let spec = elements_io::GridSpec { name: "fuel", values: &v };
    let err = elements_io::write_float_grids(&path, &[spec, spec], [2, 2, 2], 0.1, 0.0).unwrap_err();
    match err {
        elements_io::IoError::DuplicateGrid { name } => assert_eq!(name, "fuel"),
        other => panic!("expected DuplicateGrid, got {other:?}"),
    }
    assert!(!path.exists());
}

#[test]
fn a_grid_of_the_wrong_length_is_an_error() {
    let ok = vec![0.0f32; 8];
    let short = vec![0.0f32; 7];
    let dir = tempfile::tempdir().unwrap();
    let err = elements_io::write_float_grids(
        &dir.path().join("l.vdb"),
        &[
            elements_io::GridSpec { name: "a", values: &ok },
            elements_io::GridSpec { name: "b", values: &short },
        ],
        [2, 2, 2],
        0.1,
        0.0,
    )
    .unwrap_err();
    assert!(
        matches!(err, elements_io::IoError::LengthMismatch { expected: 8, got: 7 }),
        "{err:?}"
    );
}
```

- [ ] **Step 4: Run to verify they fail**

Run: `cargo nextest run -p elements-io --test vdb_multi`
Expected: compile error `no function write_float_grids` / `no GridSpec` in `elements_io`.

- [ ] **Step 5: Implement**

`crates/elements-io/src/lib.rs` — add error variants and re-exports:

```rust
pub use vdb::{GridSpec, write_float_grid, write_float_grids};
```
(replacing the existing `pub use vdb::write_float_grid;`), and inside `IoError`:

```rust
    #[error("a VDB file needs at least one grid")]
    NoGrids,
    #[error("grid name {name:?} appears twice in one VDB file")]
    DuplicateGrid { name: String },
```

`crates/elements-io/src/vdb/writer.rs` — split the header:

```rust
pub fn write_archive_header<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    uuid: &str,
) -> Result<(), IoError> {
    write_archive_header_with_count(w, uuid, 1)
}

/// `write_archive_header` for a file holding `grid_count` grids.
pub fn write_archive_header_with_count<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    uuid: &str,
    grid_count: u32,
) -> Result<(), IoError> {
    if uuid.len() != 36 {
        return Err(IoError::BadUuid { len: uuid.len() });
    }
    w.u64(OPENVDB_MAGIC)?;
    w.u32(OPENVDB_FILE_VERSION)?;
    w.u32(OPENVDB_LIBRARY_MAJOR)?;
    w.u32(OPENVDB_LIBRARY_MINOR)?;
    w.u8(1)?; // has_grid_offsets
    w.raw(uuid.as_bytes())?;
    write_metadata(w, &[])?; // no file-level metadata
    w.u32(grid_count)?;
    Ok(())
}
```

`crates/elements-io/src/vdb/tree.rs` — replace `write_float_grid` (keep its doc comment) with:

```rust
/// One named grid of a multi-grid file; `values` is x-fastest.
#[derive(Debug, Clone, Copy)]
pub struct GridSpec<'a> {
    pub name: &'a str,
    pub values: &'a [f32],
}

/// Write one uncompressed `FloatGrid` to `path`.
///
/// `values` is x-fastest with `dims[0] * dims[1] * dims[2]` entries.
/// `voxel_size` is the uniform world-space size of one voxel.
pub fn write_float_grid(
    path: &Path,
    name: &str,
    values: &[f32],
    dims: [u32; 3],
    voxel_size: f64,
    background: f32,
) -> Result<(), IoError> {
    write_float_grids(path, &[GridSpec { name, values }], dims, voxel_size, background)
}

/// Write several uncompressed `FloatGrid`s, sharing `dims` and one transform,
/// into one archive. Each grid is a descriptor followed by its data, which is
/// how OpenVDB streams write a multi-grid file. Every tree is built before the
/// file is created, so a bad grid leaves no file behind.
pub fn write_float_grids(
    path: &Path,
    grids: &[GridSpec<'_>],
    dims: [u32; 3],
    voxel_size: f64,
    background: f32,
) -> Result<(), IoError> {
    if grids.is_empty() {
        return Err(IoError::NoGrids);
    }
    for (i, grid) in grids.iter().enumerate() {
        if grids[..i].iter().any(|earlier| earlier.name == grid.name) {
            return Err(IoError::DuplicateGrid {
                name: grid.name.to_owned(),
            });
        }
    }
    let trees = grids
        .iter()
        .map(|g| build_tree(g.values, dims, background))
        .collect::<Result<Vec<_>, _>>()?;

    let file = std::fs::File::create(path)?;
    let mut w = ByteWriter::new(std::io::BufWriter::new(file));
    write_archive_header_with_count(
        &mut w,
        "00000000-0000-4000-8000-000000000000",
        grids.len() as u32,
    )?;
    for (grid, (root_mask, internals)) in grids.iter().zip(&trees) {
        write_one_grid(&mut w, grid.name, dims, voxel_size, background, root_mask, internals)?;
    }
    Ok(())
}

/// One grid's descriptor and data; the body `write_float_grid` had before FT5.
fn write_one_grid<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    name: &str,
    dims: [u32; 3],
    voxel_size: f64,
    background: f32,
    root_mask: &BitMask,
    internals: &BTreeMap<usize, Internal>,
) -> Result<(), IoError> {
    let offsets = write_grid_descriptor(w, name)?;

    let grid_pos = w.pos()?;
    w.u32(COMPRESSION_ACTIVE_MASK)?;
    write_metadata(
        w,
        &[
            // (keep the comment on the "name" entry from the old body verbatim)
            ("name", MetaValue::String(name.to_owned())),
            ("file_bbox_min", MetaValue::Vec3i([0, 0, 0])),
            (
                "file_bbox_max",
                MetaValue::Vec3i([
                    dims[0].saturating_sub(1) as i32,
                    dims[1].saturating_sub(1) as i32,
                    dims[2].saturating_sub(1) as i32,
                ]),
            ),
            ("class", MetaValue::String("unknown".to_owned())),
            (
                "file_voxel_count",
                MetaValue::I64(dims[0] as i64 * dims[1] as i64 * dims[2] as i64),
            ),
        ],
    )?;

    write_uniform_scale_transform(w, voxel_size)?;
    write_tree_topology(w, root_mask, internals, background)?;

    let block_pos = w.pos()?;
    write_tree_data(w, root_mask, internals)?;
    let end_pos = w.pos()?;

    w.patch_u64_at(offsets.grid_pos_at, grid_pos)?;
    w.patch_u64_at(offsets.block_pos_at, block_pos)?;
    w.patch_u64_at(offsets.end_pos_at, end_pos)?;
    Ok(())
}
```

Notes: import `write_archive_header_with_count` instead of `write_archive_header` in `tree.rs`'s `use super::writer::{…}`; the existing `write_tree_topology` / `write_tree_data` already take `&root_mask, &internals`, so pass the references unchanged. Move the old explanatory comment on the `"name"` metadata entry into the new function unchanged. Export `GridSpec` and `write_float_grids` from `vdb/mod.rs` (`pub use tree::{…, GridSpec, write_float_grids}`), and `write_archive_header_with_count` from `writer`'s re-export list.

- [ ] **Step 6: Run the whole io suite**

Run: `cargo nextest run -p elements-io`
Expected: PASS, including `a_single_grid_file_is_byte_identical_to_the_pre_ft5_fixture` and every existing `vdb_*` test. If `vdb-rs` cannot read a later grid (`read_grid` seeks by `grid_pos`), stop and report: the descriptor layout is wrong, not the test.

- [ ] **Step 7: Prove the tests can fail (one mutation each; restore after each)**

1. In `write_float_grids`, pass `grids[0].values` to every `build_tree` call → `every_grid_of_a_multi_grid_file_round_trips` must fail on `flame`.
2. In `write_one_grid`, write `"density"` instead of `name` in the `"name"` metadata entry only → the round trip must fail (the grid cannot be found by name).
3. In `write_archive_header_with_count`, write `1` instead of `grid_count` → `every_grid…` must fail (only one grid listed).
4. Delete the duplicate check → `a_duplicate_grid_name…` fails.

Record each failure message.

- [ ] **Step 8: Commit**

```bash
just fmt && just check
git add crates/elements-io
git commit -m "Write several grids into one VDB file" \
  -m "A render needs density, flame, temperature, fuel, char and velocity from one frame; one file per grid meant six bakes. Mutations recorded: <paste four failure messages>." \
  -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016jxQnAfsTRPo8xRwrHJx4y"
```

---

### Task 2: Graph extra outputs

**Files:**
- Modify: `crates/elements-core/src/graph/mod.rs` (`Graph`, `Evaluated`, `evaluation_order`, `eval_frame`, `run`)
- Modify: `crates/elements-core/src/graph/timeline.rs` (`advance_inner`, the `stepped.value.release_to` call)
- Create: `crates/elements-core/tests/extra_outputs.rs`

**Interfaces:**
- Consumes: existing `SocketId { node: NodeId, index: u32 }`, `SocketType::{Field, VectorField}`, `Value`.
- Produces:
  `Graph::set_extra_outputs(&mut self, sockets: Vec<SocketId>) -> Result<(), NodeError>`;
  `Graph::extra_outputs(&self) -> &[SocketId]`;
  `Evaluated { value: Value, extras: Vec<Value>, stats: EvalStats }` where `extras[i]` is the value of `extra_outputs()[i]`;
  `Evaluated::release_to(self, pool: &mut FieldPool)` releasing `value` and every extra.

Rule: a socket is kept by giving it one extra entry in `run.remaining` (the existing "still wanted" count), then moving it out of `produced` after the run. No `EvalCtx` change. `set_extra_outputs` rejects: a node out of range (`NodeError::UnknownNode`), and every other problem with the new variant `NodeError::BadExtraOutput { node: NodeId, index: u32, reason: &'static str }` (`#[error("node {node:?} output {index} cannot be an extra output: {reason}")]`): no such output index, a socket type other than `Field` or `VectorField`, a duplicate socket, and the primary result socket `{output, 0}`.

- [ ] **Step 1: Write the failing tests**

`crates/elements-core/tests/extra_outputs.rs` (reuse the harness idiom of `tests/fan_out.rs`: `GpuContext::new_headless`, `FieldPool`, `PipelineCache`, `StateStore`, `Time::at(1, 1, 24.0)`, `FieldDims::new(4, 4, 4)`; nodes from `elements_core::nodes::{ConstantField, Output}` — read `fan_out.rs` lines 1–120 for the exact `ConstantField::new(..)` constructor and `link` helper and copy them):

```rust
// graph: const(0.25) -> out ; const(0.75) -> out2? 
```
Build: node 0 `ConstantField(0.25)`, node 1 `ConstantField(0.75)`, node 2 `Output` fed by node 0. Extra outputs: `[ {1,0}, {0,0} ]`.

Tests:
1. `extras_come_back_in_the_order_they_were_named`: eval; `evaluated.extras.len()==2`; read back `extras[0]` all 0.75, `extras[1]` all 0.25; `evaluated.value` all 0.25.
2. `a_socket_that_also_feeds_the_output_is_copied_not_moved`: extra `{0,0}` while node 0 feeds `Output`; both the result and the extra read 0.25 (proves the count keeps the value alive for both).
3. `a_graph_without_extras_returns_none`: `extras.is_empty()`.
4. `set_extra_outputs_rejects_bad_sockets`: unknown node → `UnknownNode`; index 9 on node 0 → `BadExtraOutput`; duplicate socket → `BadExtraOutput`; the result socket `{2,0}` → `BadExtraOutput`; a socket of `SocketType::Scalar` (use a node whose output is Scalar — search `nodes/` for one, or define a tiny test `Node` with `outputs: vec![SocketType::Scalar]` as `fan_out.rs` does) → `BadExtraOutput`.
5. `evaluated_release_to_returns_every_field_to_the_pool`: after `evaluated.release_to(&mut pool)`, `pool.free_count()` (use whatever public pool counter `field_pool.rs` tests use) increased by 3.
6. Timeline: `a_timeline_goto_returns_extras_for_the_asked_frame_and_releases_skipped_ones` — use the stateful accumulate node used in `tests/timeline.rs` (read its helper), set an extra on its output, `goto(frame 3)` from a fresh timeline; assert `extras.len()==1`, and after releasing, the pool's live-texture count equals that of a run without extras (no leak from the two skipped frames).

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p elements-core --test extra_outputs`
Expected: compile error (`no method set_extra_outputs`).

- [ ] **Step 3: Implement**

In `graph/mod.rs`:

```rust
pub struct Graph { /* existing */ extra_outputs: Vec<SocketId> }   // default: Vec::new() in Default impl
```
Add `extras: Vec<Value>` to `Evaluated` and:

```rust
impl Evaluated {
    /// Return the result and every extra to `pool`.
    pub fn release_to(self, pool: &mut FieldPool) {
        self.value.release_to(pool);
        for value in self.extras {
            value.release_to(pool);
        }
    }
}

impl Graph {
    pub fn set_extra_outputs(&mut self, sockets: Vec<SocketId>) -> Result<(), NodeError> {
        for (i, socket) in sockets.iter().enumerate() {
            let spec = self.node(socket.node)?.sockets();
            let ty = spec.outputs.get(socket.index as usize).copied().ok_or(
                NodeError::BadExtraOutput {
                    node: socket.node,
                    index: socket.index,
                    reason: "the node has no such output",
                },
            )?;
            let bad = |reason| NodeError::BadExtraOutput {
                node: socket.node,
                index: socket.index,
                reason,
            };
            if !matches!(ty, SocketType::Field | SocketType::VectorField) {
                return Err(bad("only a field or a vector field can be exported"));
            }
            if sockets[..i].contains(socket) {
                return Err(bad("it is named twice"));
            }
            if self.output == Some(socket.node) && socket.index == 0 {
                return Err(bad("it is the graph's result socket"));
            }
        }
        self.extra_outputs = sockets;
        Ok(())
    }

    pub fn extra_outputs(&self) -> &[SocketId] {
        &self.extra_outputs
    }
}
```
(`SocketId` must derive `PartialEq`; it is a `HashMap` key so it does.)

`evaluation_order`: seed `let mut stack = vec![output]; stack.extend(self.extra_outputs.iter().map(|s| s.node));`.

`run`: after the loop that counts edges, add
```rust
for socket in &self.extra_outputs {
    *run.remaining.entry(*socket).or_insert(0) += 1;
}
```
`eval_frame`: replace the result handling with
```rust
let result = self.run(&mut run, &order, output);
let extras = match &result {
    Ok(_) => self
        .extra_outputs
        .iter()
        .map(|s| run.produced.remove(s).ok_or(NodeError::UnknownNode(s.node)))
        .collect::<Result<Vec<_>, _>>(),
    Err(_) => Ok(Vec::new()),
};
for (_, value) in run.produced.drain() {
    value.release_to(run.pool);
}
Ok(Evaluated { value: result?, extras: extras?, stats: run.stats })
```
(If `extras` is an `Err` while `result` is `Ok`, the result value must be released first: write `let value = result?; let extras = match extras { Ok(e) => e, Err(e) => { value.release_to(run.pool); return Err(e); } };`.)

`node.rs`: add the `BadExtraOutput` variant. `timeline.rs`: the skipped-frame line becomes `stepped.release_to(env.pool);`.
Everything else that builds `Evaluated` (`grep -rn "Evaluated {" crates`) gets `extras: Vec::new()` — only `mod.rs:328` today.

- [ ] **Step 4: Run**

Run: `cargo nextest run -p elements-core` — Expected: PASS (all existing tests included, incl. `fan_out.rs` copy counts unchanged).

- [ ] **Step 5: Mutations (one each, restore after each)**

1. Remove the `run.remaining` increment → `extras_come_back…` fails (extra is released, `UnknownNode`).
2. In `evaluation_order`, drop the extra seeds → an extra whose node is not upstream of the output fails test 1.
3. Return extras in reverse order → test 1 fails.
4. In the timeline, release only `stepped.value` → test 6's pool count check fails.

- [ ] **Step 6: Commit** (`just fmt && just check` first; subject "Let a graph return extra output sockets beside its result"; body: a solver exposes five outputs but `eval` returned one, so a render needed a bake per grid; list the four mutation failures).

---

### Task 3: Document `outputs` and version 4

**Files:**
- Modify: `crates/elements-core/src/graph/document.rs`, `crates/elements-core/src/graph/mod.rs` (re-export `DocOutput`)
- Modify (compile fixes, add `outputs: Vec::new()`): every `Document { … }` literal — `crates/elements-ember/src/bench/mod.rs`, `crates/elements-ember/tests/{conserve,shape_emitter,jet_shack}.rs`, `crates/elements-core/tests/document.rs`
- Modify: `docs/superpowers/specs/2026-09-30-flamethrower-ft5-export-render-design.md` (see deviation below)
- Test: `crates/elements-core/tests/document.rs`; create fixture `crates/elements-core/tests/fixtures/v3_minimal.elements`

**Interfaces:**
- Consumes: `Graph::set_extra_outputs` (Task 2).
- Produces: `pub struct DocOutput { pub node: u32, pub socket: u32, pub name: String }` (serde, `Debug, Clone, PartialEq`); `Document.outputs: Vec<DocOutput>` (`#[serde(default, skip_serializing_if = "Vec::is_empty")]`); `ELEMENTS_DOC_VERSION = 4`; versions 1–3 load and become 4; `into_graph` calls `set_extra_outputs`.

**Deviation from spec §3.2 (record it in the spec in this task):** the spec asks `validate_for` to reject a read-back set that does not fit `max_buffer_size` together. Read-backs are sequential, each through its own staging buffer, so the existing per-field check already bounds every buffer; a combined check would reject documents that bake fine. Replace that sentence with: "Read-backs are sequential, so the existing per-field `validate_for` check bounds each buffer." Likewise the reserved-name rule becomes: "Two entries that would write the same grid name, counting a vector entry `velocity` as `velocity_x/_y/_z`, are rejected by the bake before any GPU work (Task 5); `from_json` rejects duplicate and empty names."

- [ ] **Step 1: Write the failing tests** (append to `tests/document.rs`; add the `v3_minimal` fixture as a copy of `v1_minimal.elements` with `"version": 3`):

```rust
const V3: &str = include_str!("fixtures/v3_minimal.elements");

#[test]
fn a_version_3_document_loads_as_version_4_and_bakes_no_outputs() {
    let doc = Document::from_json(V3).unwrap();
    assert_eq!(doc.version, 4);
    assert!(doc.outputs.is_empty());
    assert!(!doc.to_json().unwrap().contains("outputs"), "an empty list is not written");
}

fn with_outputs(outputs: &str) -> String {
    V3.replacen("\"output\": 1", &format!("\"output\": 1, \"outputs\": {outputs}"), 1)
}

#[test]
fn outputs_round_trip() {
    let json = with_outputs(r#"[{"node":0,"socket":0,"name":"density"}]"#);
    let doc = Document::from_json(&json).unwrap();
    assert_eq!(doc.outputs, vec![DocOutput { node: 0, socket: 0, name: "density".into() }]);
    assert_eq!(Document::from_json(&doc.to_json().unwrap()).unwrap().outputs, doc.outputs);
}

#[test]
fn duplicate_and_empty_output_names_are_rejected() {
    for bad in [
        r#"[{"node":0,"socket":0,"name":"a"},{"node":0,"socket":0,"name":"a"}]"#,
        r#"[{"node":0,"socket":0,"name":""}]"#,
    ] {
        assert!(matches!(
            Document::from_json(&with_outputs(bad)),
            Err(DocError::BadParams { .. })
        ), "{bad}");
    }
}

#[test]
fn into_graph_rejects_an_output_that_is_not_a_field() {
    // node 1 is core.output (socket 0 is its result), node 0 is the source.
    let registry = NodeRegistry::with_builtins();
    for bad in [
        r#"[{"node":9,"socket":0,"name":"a"}]"#, // no such node
        r#"[{"node":0,"socket":9,"name":"a"}]"#, // no such socket
        r#"[{"node":1,"socket":0,"name":"a"}]"#, // the result socket
    ] {
        let doc = Document::from_json(&with_outputs(bad)).unwrap();
        assert!(doc.into_graph(&registry).is_err(), "{bad}");
    }
}

#[test]
fn into_graph_installs_the_outputs_on_the_graph() {
    let registry = NodeRegistry::with_builtins();
    let doc = Document::from_json(&with_outputs(r#"[{"node":0,"socket":0,"name":"density"}]"#)).unwrap();
    let (graph, _) = doc.into_graph(&registry).unwrap();
    assert_eq!(graph.extra_outputs().len(), 1);
}
```
Adjust the node indices to the fixture (`v1_minimal`: node 0 is the noise/constant source, node 1 `core.output`, `"output": 1`); confirm by reading it.

- [ ] **Step 2: Run** `cargo nextest run -p elements-core --test document` — Expected: FAIL (`DocOutput` missing).

- [ ] **Step 3: Implement**

```rust
/// One grid a bake writes: socket `socket` of node `node`, named `name` in the file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocOutput {
    pub node: u32,
    pub socket: u32,
    pub name: String,
}
```
Add to `Document`, after `output`:
```rust
    /// Extra grids a bake writes into each frame's file (version 4).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<DocOutput>,
```
`ELEMENTS_DOC_VERSION = 4`; in `from_json`: `1 | 2 | 3 => doc.version = ELEMENTS_DOC_VERSION,`; update the doc comment ("reads versions 1–3"). After the domain-size check, validate names:

```rust
let mut seen = std::collections::HashSet::new();
for entry in &doc.outputs {
    if entry.name.is_empty() || !seen.insert(entry.name.as_str()) {
        return Err(DocError::BadParams {
            kind: "document".to_string(),
            reason: format!("output name {:?} is empty or used twice", entry.name),
        });
    }
}
```
In `into_graph`, before `graph.set_output`:
```rust
graph.set_extra_outputs(
    self.outputs
        .iter()
        .map(|o| SocketId { node: NodeId(o.node), index: o.socket })
        .collect(),
)?;
```
(`DocError: From<NodeError>` is already relied on by `graph.connect(..)?`.) Re-export `DocOutput` in `graph/mod.rs` line 11. Fix every `Document { … }` literal with `outputs: Vec::new(),`.

- [ ] **Step 4: Regenerate the committed example** — the `the_jet_shack_example_is_the_generated_scene` test now fails (version 4):
`cargo nextest run -p elements-ember --run-ignored ignored-only -E 'test(write_jet_shack_example)'` then `git diff examples/jet_shack.elements` must show only `"version": 3` → `4`. Grep the repo for other version consumers: `grep -rn "version\"\? *[:=] *3\|ELEMENTS_DOC_VERSION" addon crates tests --include=*.py --include=*.rs` and fix any (the add-on's Python must not hard-code 3; report if it does).

- [ ] **Step 5: Run** `just check` — Expected: PASS.

- [ ] **Step 6: Mutations**

1. Drop `1 | 2 | 3` to `1 | 2` → the v3 test fails (`UnsupportedVersion(3)`).
2. Remove `skip_serializing_if` → `…bakes_no_outputs` fails (`"outputs"` appears).
3. Skip the duplicate-name check → `duplicate_and_empty…` fails.
4. Remove the `set_extra_outputs` call in `into_graph` → `into_graph_installs…` fails.

- [ ] **Step 7: Commit** ("Add an outputs list to the document format"; body: why version 4 and why the spec's combined-buffer check and reserved-name rule were replaced; mutation failures).

---

### Task 4: Solver output 5, `fuel`

**Files:**
- Modify: `crates/elements-ember/src/solver.rs` (`sockets`, `eval` ~1421, `copy_outputs` ~1636–1651, doc comment at top)
- Test: `crates/elements-ember/tests/fire.rs` (read its helpers first; it already builds fire solvers and reads `FUEL`)

**Interfaces:**
- Produces: smoke-solver output socket 5 `fuel` (`Field`): a pooled copy of `state.fire.fuel` with fire on, zeros without. `copy_outputs` takes `wanted: [bool; 6]`.

- [ ] **Step 1: Write the failing tests** in `tests/fire.rs` (copy the setup of the existing `flame_output…` test: same emitter with `fuel_rate`, same solver build and `Harness`):
1. `fuel_output_equals_the_fuel_slot_with_fire_on` — run 3 steps with fuel emitted; read output socket 5 and the `FUEL` state slot through the same accessor the existing tests use (e.g. `cache_sums` or `read_slot`); assert bit-equal and `max > 0`.
2. `fuel_output_is_zero_with_fire_off` — solver with no fuel input; output 5 read back is all `0.0` and has the domain's dims (not a `Scalar`).
3. `fuel_output_is_not_computed_when_unwanted` — if an existing `output_wanted` style test exists for outputs 3/4, add the socket-5 twin; otherwise skip and say so.

- [ ] **Step 2: Run** `cargo nextest run -p elements-ember --test fire fuel_output` — Expected: FAIL (index out of range / 5 outputs).

- [ ] **Step 3: Implement**

`sockets()` outputs gain, after char:
```rust
                // 5: fuel, a copy of the fuel slot; zero without fire (FT5 spec §3.1).
                SocketType::Field,
```
`eval`: `let wanted: [bool; 6] = …`. `copy_outputs`: `wanted: [bool; 6]`, `Vec::with_capacity(6)`, `_ => unreachable!("only six outputs exist")`, and the arm
```rust
                5 => fuel_output(gpu, cache, pool, state).map(Value::Field),
```
```rust
/// The fuel output: a copy of the fuel state with fire on, zero without (FT5 spec §3.1).
fn fuel_output(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    pool: &mut FieldPool,
    state: &SolverState,
) -> Result<Field, GpuError> {
    match &state.fire {
        Some(fire) => pool.duplicate(gpu, &fire.fuel),
        None => pool.acquire_zeroed(gpu, cache, state.density.dims()),
    }
}
```
Update the module doc comment: "Output socket 5 is the fuel, zero without fire (FT5 spec §3.1)."

- [ ] **Step 4: Run** `cargo nextest run -p elements-ember` — Expected: PASS including the fire-off guard tests (hashes unchanged: `0x7fd45178bbd35c70`, `0xfaf6ca0296d5d74a`, `0xfbcd_e02f_2e02_d31f`). If a hash changes, stop: output 5 must not alter any existing output.
Also grep for other code that counts solver outputs: `grep -rn "outputs.len()\|\[bool; 5\]\|wanted:" crates/elements-ember crates/elements-cli addon` and fix any (e.g. the add-on's node definitions, `bench`).

- [ ] **Step 5: Mutations**

1. `Some(fire) => pool.duplicate(gpu, &fire.react)` → test 1 fails (react ≠ fuel).
2. `None => pool.duplicate(gpu, &state.density)` → test 2 fails (nonzero).
3. `_ => unreachable!` arm left but `5 =>` mapped to `flame_output` → test 1 fails.

- [ ] **Step 6: Commit** ("Give the smoke solver a fuel output"; body: fuel was state only, so a render could not read it; hashes unchanged, mutation failures).

---

### Task 5: Multi-grid `bake` with cell-centred velocity

**Files:**
- Create: `crates/elements-cli/src/grids.rs`
- Modify: `crates/elements-cli/src/bake.rs`, `crates/elements-cli/src/main.rs` (`mod grids;`)
- Test: unit tests in `grids.rs`; `crates/elements-cli/tests/cli.rs`

**Interfaces:**
- Consumes: `Document.outputs: Vec<DocOutput>`, `Evaluated.extras`, `Evaluated::release_to`, `elements_io::{GridSpec, write_float_grids}`, `StaggeredField::face`, `Axis::ALL`, `Field::read_back`.
- Produces:
  `grids::cell_centred_velocity(cells: [u32; 3], faces: [&[f32]; 3]) -> anyhow::Result<[Vec<f32>; 3]>` — component `a` of cell `(x,y,z)` is the mean of its two faces along axis `a`; face arrays are x-fastest with `StaggeredField::face_dims`.
  `grids::expanded_names(graph: &Graph, outputs: &[DocOutput]) -> anyhow::Result<Vec<String>>` — final grid names in file order: a `VectorField` entry `n` becomes `n_x`, `n_y`, `n_z`; errors on any name used twice after expansion, naming it.

- [ ] **Step 1: Failing unit tests in `grids.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::cell_centred_velocity;

    #[test]
    fn velocity_is_the_mean_of_each_cells_two_faces() {
        // 2x1x1 cells: x faces are 3x1x1, y faces 2x2x1, z faces 2x1x2. Face index is
        // (z * face_ny + y) * face_nx + x, so the y faces are [y=0: 10, 20 | y=1: 30, 50].
        let fx = [1.0, 3.0, 7.0];
        let fy = [10.0, 20.0, 30.0, 50.0];
        let fz = [100.0, 200.0, 300.0, 500.0];
        let [vx, vy, vz] = cell_centred_velocity([2, 1, 1], [&fx, &fy, &fz]).unwrap();
        assert_eq!(vx, [2.0, 5.0]);
        assert_eq!(vy, [20.0, 35.0]);
        assert_eq!(vz, [200.0, 350.0]);
    }

    #[test]
    fn a_face_array_of_the_wrong_length_is_an_error() {
        let fx = [0.0; 3];
        let short = [0.0; 3];
        assert!(cell_centred_velocity([2, 1, 1], [&fx, &short, &short]).is_err());
    }
}
```

- [ ] **Step 2: Failing CLI test** in `tests/cli.rs`:

```rust
#[test]
fn bake_with_outputs_writes_every_named_grid_into_one_file() {
    // const(0.25) -> output, plus a second constant exported as "second".
    const DOC: &str = r#"{
      "version": 4, "dims": [8, 8, 8],
      "nodes": [
        { "id": 0, "kind": "core.noise_field", "params": { "seed": 7, "frequency": 4.0 } },
        { "id": 1, "kind": "core.output", "params": {} },
        { "id": 2, "kind": "core.noise_field", "params": { "seed": 9, "frequency": 2.0 } }
      ],
      "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
      "output": 1,
      "outputs": [
        { "node": 0, "socket": 0, "name": "first" },
        { "node": 2, "socket": 0, "name": "second" }
      ]
    }"#;
    let dir = tempfile::tempdir().unwrap();
    let graph = dir.path().join("two.elements");
    std::fs::write(&graph, DOC).unwrap();
    let out = dir.path().join("vdb");
    assert!(cli().args(["bake", graph.to_str().unwrap()])
        .args(["--out", out.to_str().unwrap(), "--frames", "1-2", "--name", "shot"])
        .status().unwrap().success());
    for frame in 1..=2 {
        let path = out.join(format!("shot.{frame:04}.vdb"));
        let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
        let mut names = vdb_rs::VdbReader::new(file).unwrap().available_grids();
        names.sort();
        assert_eq!(names, ["first", "second"]);
    }
}

#[test]
fn bake_rejects_two_outputs_that_expand_to_the_same_grid_before_baking() {
    // Two entries named "a": from_json already rejects; the CLI reports it and writes nothing.
    // (Vector-expansion collisions are covered by expanded_names' unit test.)
    …same DOC with both names "a"…
    assert!(!status.success()); assert!(!out.exists() || out.read_dir().unwrap().next().is_none());
}
```
Does `core.noise_field` produce different values for seeds 7 and 9? Yes by its `seed` param. In the first test also assert the two grids' voxel values differ (read with the `read_active` helper idiom from `elements-io/tests/vdb_multi.rs`; copy it into `cli.rs`) — this proves the right socket went to the right name.

- [ ] **Step 3: Run** `cargo nextest run -p elements-cli` — Expected: FAIL.

- [ ] **Step 4: Implement `grids.rs`**

```rust
//! What `bake` writes per frame when a document names its outputs.

use anyhow::{Context, bail, ensure};
use elements_core::gpu::{Axis, FieldDims, StaggeredField};
use elements_core::graph::{DocOutput, Graph, NodeId, SocketType};

/// Average each cell's two faces along each axis (spec §3.3).
pub fn cell_centred_velocity(
    cells: [u32; 3],
    faces: [&[f32]; 3],
) -> anyhow::Result<[Vec<f32>; 3]> {
    let [nx, ny, nz] = cells.map(|c| c as usize);
    let cell_dims = FieldDims::new(cells[0], cells[1], cells[2]);
    let mut out = [Vec::new(), Vec::new(), Vec::new()];
    for axis in Axis::ALL {
        let a = axis.index();
        let fd = StaggeredField::face_dims(cell_dims, axis);
        let (fx, fy) = (fd.x as usize, fd.y as usize);
        ensure!(
            faces[a].len() == fx * fy * fd.z as usize,
            "{axis:?} face has {} values, expected {}",
            faces[a].len(),
            fx * fy * fd.z as usize
        );
        let at = |x: usize, y: usize, z: usize| faces[a][(z * fy + y) * fx + x];
        let mut v = Vec::with_capacity(nx * ny * nz);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let (lo, hi) = match axis {
                        Axis::X => (at(x, y, z), at(x + 1, y, z)),
                        Axis::Y => (at(x, y, z), at(x, y + 1, z)),
                        Axis::Z => (at(x, y, z), at(x, y, z + 1)),
                    };
                    v.push(0.5 * (lo + hi));
                }
            }
        }
        out[a] = v;
    }
    Ok(out)
}

/// The final grid names, in file order; a vector entry `n` becomes `n_x`, `n_y`, `n_z`.
pub fn expanded_names(graph: &Graph, outputs: &[DocOutput]) -> anyhow::Result<Vec<String>> {
    let mut names: Vec<String> = Vec::new();
    for entry in outputs {
        let spec = graph.node(NodeId(entry.node))?.sockets();
        let ty = *spec
            .outputs
            .get(entry.socket as usize)
            .with_context(|| format!("output {:?} has no socket {}", entry.name, entry.socket))?;
        let new: Vec<String> = match ty {
            SocketType::Field => vec![entry.name.clone()],
            SocketType::VectorField => ["x", "y", "z"]
                .iter()
                .map(|s| format!("{}_{s}", entry.name))
                .collect(),
            other => bail!("output {:?} is a {other:?}, not a field", entry.name),
        };
        for name in new {
            ensure!(!names.contains(&name), "two outputs would both write a grid named {name:?}");
            names.push(name);
        }
    }
    Ok(names)
}
```
Add a unit test for `expanded_names`'s collision case using a graph with a `VectorField` output: if building one needs the ember registry, put that test in `elements-ember`'s tests instead? `elements-cli` depends on `elements-ember`, so use `elements_ember::registry()` in a `cli` unit test with a document containing `ember.smoke_solver` (node kinds: see `examples/jet_shack.elements`), outputs `velocity`=(1,2) and a scalar `velocity_x`=(1,0): expect an error mentioning `"velocity_x"`. (Name check: `DocError` → the error from `graph.node` is `NodeError`, `?` converts into `anyhow`.)

`bake.rs`: before `doc.into_graph`, `let outputs = doc.outputs.clone();`. After `into_graph`, when `!outputs.is_empty()`: `let names = grids::expanded_names(&graph, &outputs)?;` (errors before any GPU eval, before `create_dir_all`). Replace the per-frame body with:

```rust
        let evaluated = timeline.goto(&graph, &gpu, &mut pool, &mut pipelines, dims, frame)?;
        if let Some(warning) = timeline.take_warning() {
            eprintln!("warning: {warning}");
        }
        let path = out_dir.join(format!("{name}.{frame:04}.vdb"));
        if outputs.is_empty() {
            let values = evaluated.value.as_field()?.read_back(&gpu)?;
            evaluated.release_to(&mut pool);
            elements_io::write_float_grid(&path, name, &values, [dims.x, dims.y, dims.z], voxel_size, 0.0)
                .with_context(|| format!("writing {}", path.display()))?;
        } else {
            let cells = [dims.x, dims.y, dims.z];
            let (value, extras) = (evaluated.value, evaluated.extras);
            value.release_to(&mut pool);
            let grids = read_grids(&gpu, &mut pool, &outputs, extras, cells)
                .with_context(|| format!("reading frame {frame}"))?;
            debug_assert_eq!(grids.len(), names.len());
            let specs: Vec<elements_io::GridSpec<'_>> = names
                .iter()
                .zip(&grids)
                .map(|(n, v)| elements_io::GridSpec { name: n, values: v })
                .collect();
            if let Err(e) = elements_io::write_float_grids(&path, &specs, cells, voxel_size, 0.0) {
                let _ = std::fs::remove_file(&path);
                return Err(e).with_context(|| format!("writing {}", path.display()));
            }
        }
```
with

```rust
/// Read every extra back: a field as itself, a vector field as three cell-centred grids.
/// Values return to `pool` whether or not a read fails.
fn read_grids(
    gpu: &GpuContext,
    pool: &mut FieldPool,
    outputs: &[DocOutput],
    extras: Vec<Value>,
    cells: [u32; 3],
) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut grids = Vec::new();
    let mut failure = None;
    for (entry, value) in outputs.iter().zip(extras) {
        if failure.is_none() {
            match read_one(gpu, &value, cells) {
                Ok(mut read) => grids.append(&mut read),
                Err(e) => failure = Some(e.context(format!("output {:?}", entry.name))),
            }
        }
        value.release_to(pool);
    }
    match failure {
        Some(e) => Err(e),
        None => Ok(grids),
    }
}

fn read_one(gpu: &GpuContext, value: &Value, cells: [u32; 3]) -> anyhow::Result<Vec<Vec<f32>>> {
    match value {
        Value::Field(field) => Ok(vec![field.read_back(gpu)?]),
        Value::VectorField(velocity) => {
            let fx = velocity.face(Axis::X).read_back(gpu)?;
            let fy = velocity.face(Axis::Y).read_back(gpu)?;
            let fz = velocity.face(Axis::Z).read_back(gpu)?;
            Ok(grids::cell_centred_velocity(cells, [&fx, &fy, &fz])?.into())
        }
        other => anyhow::bail!("{other:?} cannot be written as a grid"),
    }
}
```
Update the `bake` doc comment: "With an `outputs` list in the document, each frame's file holds those grids and `name` is only the file stem." Import `Axis`, `Value`, `DocOutput`.

- [ ] **Step 5: Run** `cargo nextest run -p elements-cli` — Expected: PASS (existing `bake_writes_one_vdb_per_frame` etc. unchanged).

- [ ] **Step 6: Mutations (one each)**

1. `cell_centred_velocity`: use `(x+1)` in both taps for `Axis::Y` → unit test fails (vy).
2. `0.5 *` → `1.0 *` → unit test fails.
3. In `read_grids`, zip `outputs` with `extras.into_iter().rev()` → `bake_with_outputs…` fails (values differ from names).
4. Delete the `ensure!(!names.contains(..))` → the collision unit test fails.
5. Write `evaluated.value` instead of extras when `outputs` non-empty → the file lists one grid → fails.

- [ ] **Step 7: Commit** ("Bake every named grid of a frame into one file"; body: why one evaluation per frame and why velocity is averaged on the CPU; mutation failures).

---

### Task 6: The shack render scene

**Files:**
- Modify: `crates/elements-ember/tests/jet_shack.rs`
- Create: `examples/jet_shack_render.elements` (generated)

**Interfaces:**
- Consumes: Tasks 3–4 (`outputs`, solver socket 5), FT4's `SurfaceFuel { load }`, mesh collider output 2 (load) and solver input 7.
- Produces: `examples/jet_shack_render.elements` — the baseline jet-shack scene (128×64×64 over 2.0 m) with `surface_fuel` on the shack, the load wired to solver input 7, and `outputs`: `density (1,0)`, `temperature (1,1)`, `velocity (1,2)`, `flame (1,3)`, `char (1,4)`, `fuel (1,5)`.

- [ ] **Step 1: Write the failing tests** (in `jet_shack.rs`):

```rust
const RENDER_EXAMPLE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/jet_shack_render.elements"
);

#[test]
fn the_render_example_is_the_generated_render_scene() {
    let text = std::fs::read_to_string(RENDER_EXAMPLE).expect("examples/jet_shack_render.elements exists");
    assert!(text == render_document().to_json().unwrap() + "\n",
        "examples/jet_shack_render.elements is stale; run write_jet_shack_render_example");
    Document::from_json(&text).unwrap().into_graph(&elements_ember::registry()).unwrap();
}

#[test]
fn the_render_scene_exports_six_entries_with_the_shack_load_wired_in() {
    let doc = render_document();
    let names: Vec<&str> = doc.outputs.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(names, ["density", "temperature", "velocity", "flame", "char", "fuel"]);
    assert!(doc.edges.iter().any(|e| (e.from_node, e.from_index, e.to_node, e.to_index) == (3, 2, 1, 7)));
}
```

- [ ] **Step 2: Implement** `render_document()` and the ignored writer:

```rust
/// The baseline scene plus what a render reads: the shack's wood load, wired into
/// the solver, and the grids to export (FT5 spec §3.5).
fn render_document() -> Document {
    let mut doc = document(BASELINE);
    let mut shack = shack_collider();
    shack.surface_fuel = Some(SurfaceFuel { load: RENDER_WOOD_LOAD });
    doc.nodes[3].params = serde_json::to_value(shack).expect("mesh params serialize");
    doc.edges.push(DocEdge { from_node: 3, from_index: 2, to_node: 1, to_index: 7 });
    let out = |socket, name: &str| DocOutput { node: 1, socket, name: name.to_owned() };
    doc.outputs = vec![
        out(0, "density"), out(1, "temperature"), out(2, "velocity"),
        out(3, "flame"), out(4, "char"), out(5, "fuel"),
    ];
    doc
}

#[test]
#[ignore = "writes examples/jet_shack_render.elements"]
fn write_jet_shack_render_example() {
    std::fs::write(RENDER_EXAMPLE, render_document().to_json().unwrap() + "\n").unwrap();
}
```
with `const RENDER_WOOD_LOAD: f32 = 4.0;` and imports `SurfaceFuel` (from `elements_ember::collider`) and `DocOutput`. Generate: `cargo nextest run -p elements-ember --run-ignored ignored-only -E 'test(write_jet_shack_render_example)'`.

- [ ] **Step 3: Measure that the shack chars in this scene** (the render's `char` still is empty otherwise). Bake frames 60 with the new example:
`cargo build --release -p elements-cli && target/release/elements bake examples/jet_shack_render.elements --out target/ft5/probe --frames 60 --name shack --voxel-size 0.015625`
Then read the `char` grid's max with a short Python/Rust check (vdb-rs via a throwaway `cargo test` or `python3` with `openvdb` if present; simplest: a temporary `#[test]` in `elements-cli/tests` that prints max, deleted afterwards). **Decision rule:** if `max(char) > 0` at frame 60, keep the defaults and record the number. If it is 0, set `temperature_dissipation` 8 and `surface_burn_rate` 8 on the solver params in `render_document()` (FT4's recorded settings, `docs/bench/surface-ignition.md`), regenerate, re-measure, and record both measurements in the commit body. If still 0, stop and report; do not tune further.

- [ ] **Step 4: Run** `cargo nextest run -p elements-ember --test jet_shack` — Expected: PASS.

- [ ] **Step 5: Mutations**

1. Drop the `(3,2,1,7)` edge from `render_document()` → the second test fails (and the measured char is 0).
2. Swap `"flame"` and `"char"` sockets → the first `names`/socket assertion... add an assertion on `(socket, name)` pairs in test 2 if the names-only check does not catch it.

- [ ] **Step 6: Commit** ("Add the shack render scene"; body: the wood load and exports, the char measurement and any setting change, mutation failures).

---

### Task 7: Blender layout helpers and the velocity check

**Files:**
- Create: `tests/bench/shack_layout.py`, `tests/bench/test_shack_layout.py`
- Modify: `justfile` (`py-test`)

**Interfaces:**
- Consumes: `placement.cell_offset(dx, True)`.
- Produces (all pure, no `bpy`):
  `GRIDS: tuple[str, ...]` = `("density","flame","temperature","fuel","char","velocity_x","velocity_y","velocity_z")`;
  `TEMPERATURE_RANGE = (0.0, 3.0)`, `KELVIN_RANGE = (1000.0, 2400.0)`, `FLAME_STRENGTH = 8.0`, `DENSITY = 20.0`;
  `kelvin(temperature: float) -> float` (linear map, clamped);
  `bake_file(bake_dir, name, frame) -> str` (`"{dir}/{name}.{frame:04d}.vdb"`);
  `still_name(kind, frame) -> str` (`"shack-{kind}-f{frame:03d}.png"`);
  `missing_grids(present: Iterable[str]) -> list[str]`;
  `camera(extent: tuple[float,float,float], aspect: float, margin: float = 0.1) -> dict` (`location`, `rotation`, `ortho_scale`; front view along +y like `placement.camera_for`);
  `warm_pixels(rgba: Sequence[float], threshold: float = 0.05) -> int` (count of pixels with `R - B > threshold`);
  `lit_bbox(rgba, width, height, threshold=0.02) -> tuple[int,int,int,int] | None` (min_x, min_y, max_x, max_y of pixels whose max(R,G,B) > threshold).

- [ ] **Step 1: Write failing tests** in `test_shack_layout.py` (same style as `test_placement.py`: plain functions, `if __name__ == "__main__"` runner — copy that file's footer):
  - `kelvin(0.0)==1000`, `kelvin(3.0)==2400`, `kelvin(1.5)==1700`, `kelvin(-1)==1000`, `kelvin(99)==2400`.
  - `bake_file("d","shack",7)=="d/shack.0007.vdb"`; `bake_file("d","shack",12345)=="d/shack.12345.vdb"`.
  - `missing_grids(GRIDS)==[]`; `missing_grids(["density"])` lists the other seven, in `GRIDS` order.
  - `camera((2.0,1.0,1.0), 16/9)`: visible x range (use the `view` helper from `test_placement.py`: copy it) contains `[0, 2.0]` and z range contains `[0, 1.0]`; `rotation == (pi/2, 0, 0)`.
  - `warm_pixels` on a hand-built 2×2 RGBA list (one orange pixel `(1,0.4,0.1,1)`, three grey) → 1; `lit_bbox` on a 4×3 image with lit pixels at (1,1) and (2,2) (note Blender's pixel rows run bottom-up; document that the helper treats row 0 as the bottom) → `(1,1,2,2)`; an all-black image → `None`.
  - `test_shack_constants_match_the_scene`: load `examples/jet_shack_render.elements` (path relative to the test file), find the node with `kind == "ember.mesh_collider"`, assert `params["transform"]["keys"][0]["translate"]` ≈ `[1.45, 0.5, 0.0]` and `SHACK_AT`/`SHACK_SIZE` defined in `shack_layout` (`(1.45,0.5,0.0)`, `(0.6,0.5,0.5)`) — these are the shack's footprint centre on the floor and size, mirrored from `crates/elements-ember/tests/jet_shack.rs`; the test guards the one that lives in the scene file.
- [ ] **Step 2: Run** `python3 tests/bench/test_shack_layout.py` — Expected: FAIL (`ModuleNotFoundError: shack_layout`).
- [ ] **Step 3: Implement** `shack_layout.py` (GPL header like `placement.py`; module docstring stating constants are chosen by eye and recorded). `camera`: `x = extent[0]/2`, `z = extent[2]/2`, `location = (x, -4 * max(extent), z)`, `rotation = (math.pi/2, 0.0, 0.0)`, `ortho_scale = max(extent[0]*(1+2*margin), extent[2]*(1+2*margin)*aspect)` (Blender's `ortho_scale` spans the frame's larger side; for aspect ≥ 1 that is the width).
- [ ] **Step 4: Add to `py-test`:** `python tests/bench/test_shack_layout.py`. Run `just py-test` — Expected: PASS. `ruff check tests/bench`.
- [ ] **Step 5: Mutations (one each):** `kelvin` without the clamp → fails; `rotation` sign flipped → fails; `warm_pixels` using `B - R` → fails; `lit_bbox` swapping x/y in the return → fails. Record the messages.
- [ ] **Step 6: The velocity-naming check (spec §3.5(3), risk 1).** Bake the real file: `cargo build --release -p elements-cli && target/release/elements bake examples/jet_shack_render.elements --out target/ft5/velocity --frames 30 --name shack --voxel-size 0.015625`. Then in Blender (scratch script under the scratchpad, not committed):
```python
import bpy, sys
bpy.ops.wm.read_factory_settings(use_empty=True)
vol = bpy.data.volumes.new("v"); vol.filepath = sys.argv[-1]; vol.is_sequence = False
assert vol.grids.load(), vol.grids.error_message
print("GRIDS", [g.name for g in vol.grids])
vol.velocity_grid = "velocity"
print("VELOCITY_X", vol.velocity_x_grid, "VELOCITY_Y", vol.velocity_y_grid, "VELOCITY_Z", vol.velocity_z_grid)
```
Run: `/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python-exit-code 1 --python <scratch.py> -- target/ft5/velocity/shack.0030.vdb`.
Expected: `GRIDS` lists all eight names (this also proves Blender reads a multi-grid file we wrote), and the `velocity_*_grid` properties name `velocity_x/_y/_z` (Blender derives them from the prefix). **If they do not, or Blender cannot read the file:** stop. Write the observed output into `docs/bench/render-shack/velocity-check.md`, and report: the plan then needs a `Vec3s` writer task before Task 8.
- [ ] **Step 7: Commit** ("Add the shack render's layout helpers"; body: the velocity check's observed Blender output, verbatim, and the mutation failures).

---

### Task 8: The Blender scene script

**Files:**
- Create: `tests/bench/render_shack.py`

**Interfaces:**
- Consumes: `shack_layout.*`, `render_compare` (`settings`, `flat_material`, `node_material`, `add_sun`, `add_world`, `link`, `unlit`, `KEY_STRENGTH`, `FILL_STRENGTH`, `WIDTH`, `HEIGHT`, `use_gpu` via `settings`), the scene JSON.
- Produces: `Blender … --python tests/bench/render_shack.py -- BAKE_DIR NAME FRAME SCENE_JSON OUT_DIR` writes `shack-beauty-f{FRAME:03d}.png`, `shack-fuel-f…png`, `shack-char-f…png` into `OUT_DIR`, writes `OUT_DIR/render-device`, and exits nonzero on: a missing file, a missing grid (names printed), a render that writes nothing, or a failed image check.

- [ ] **Step 1: Write the script.** `import render_compare as rc` (it only imports `bpy`, which is present inside Blender). Structure:

```python
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render one baked shack frame: beauty, fuel and char stills (FT5 spec §3.5).

Run: Blender --background --factory-startup --python-exit-code 1 \
    --python tests/bench/render_shack.py -- BAKE_DIR NAME FRAME SCENE_JSON OUT_DIR
"""
import json, math, os, sys
import bpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import placement, render_compare as rc, shack_layout as sl  # noqa: E402

PASSES = ("beauty", "fuel", "char")
EMISSION_PASS_STRENGTH = 4.0   # fuel/char are in [0, 1]-ish; chosen by eye


def beauty_material():
    """Smoke from `density`; emission = flame x FLAME_STRENGTH, coloured by
    blackbody at the Kelvin the real `temperature` grid maps to."""
    mat, nt = rc.node_material("beauty")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    pv = nt.nodes.new("ShaderNodeVolumePrincipled")
    pv.inputs["Color"].default_value = (1, 1, 1, 1)
    pv.inputs["Density"].default_value = sl.DENSITY
    pv.inputs["Density Attribute"].default_value = "density"
    pv.inputs["Absorption Color"].default_value = (0, 0, 0, 1)
    pv.inputs["Blackbody Intensity"].default_value = 0.0
    flame = nt.nodes.new("ShaderNodeAttribute"); flame.attribute_name = "flame"
    strength = nt.nodes.new("ShaderNodeMath"); strength.operation = "MULTIPLY"
    strength.inputs[1].default_value = sl.FLAME_STRENGTH
    temp = nt.nodes.new("ShaderNodeAttribute"); temp.attribute_name = "temperature"
    kelvin = nt.nodes.new("ShaderNodeMapRange")
    kelvin.clamp = True
    kelvin.inputs["From Min"].default_value, kelvin.inputs["From Max"].default_value = sl.TEMPERATURE_RANGE
    kelvin.inputs["To Min"].default_value, kelvin.inputs["To Max"].default_value = sl.KELVIN_RANGE
    body = nt.nodes.new("ShaderNodeBlackbody")
    nt.links.new(flame.outputs["Fac"], strength.inputs[0])
    nt.links.new(strength.outputs["Value"], pv.inputs["Emission Strength"])
    nt.links.new(temp.outputs["Fac"], kelvin.inputs["Value"])
    nt.links.new(kelvin.outputs["Result"], body.inputs["Temperature"])
    nt.links.new(body.outputs["Color"], pv.inputs["Emission Color"])
    nt.links.new(pv.outputs["Volume"], out.inputs["Volume"])
    return mat


def pass_material(grid):
    """Emission only: `grid` x EMISSION_PASS_STRENGTH, white, no density."""
    mat, nt = rc.node_material(f"pass-{grid}")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    pv = nt.nodes.new("ShaderNodeVolumePrincipled")
    pv.inputs["Density"].default_value = 0.0
    pv.inputs["Blackbody Intensity"].default_value = 0.0
    pv.inputs["Emission Color"].default_value = (1, 1, 1, 1)
    attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_name = grid
    scale = nt.nodes.new("ShaderNodeMath"); scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = EMISSION_PASS_STRENGTH
    nt.links.new(attr.outputs["Fac"], scale.inputs[0])
    nt.links.new(scale.outputs["Value"], pv.inputs["Emission Strength"])
    nt.links.new(pv.outputs["Volume"], out.inputs["Volume"])
    return mat
```

Then: `shack_mesh(scene_json)` — find the `ember.mesh_collider` node, read `mesh["positions"]` and `mesh["indices"]` (accept flat lists or lists of triples), translate by `transform.keys[0].translate`, build with `bpy.data.meshes.from_pydata(verts, [], faces)` where faces are index triples, material `rc.flat_material("shack", 0.25)`, `rc.unlit(obj)` **not** applied (the proxy should shade; give it a light grey surface so flame light falls on it: use a Principled BSDF diffuse via `rc.node_material`). `build(pass_name, …)`: `bpy.ops.wm.read_factory_settings(use_empty=True)`, `device = rc.settings(scene)`, volume via `rc.add_volume("shack", path, location, material, grid)` — **before adding the volume, check** `missing = sl.missing_grids(g.name for g in vol.grids)`; `rc.add_volume` already exits when the named grid is missing but only for one grid, so load once first: create the `bpy.data.volumes` datablock, `vol.grids.load()`, `sys.exit(f"{path}: missing grids {missing}")` if any, and set `vol.velocity_grid = "velocity"` (Task 7's check confirmed the naming). For the `fuel` and `char` passes: world colour black (`world.node_tree.nodes["Background"].inputs["Color"].default_value = (0,0,0,1)`), no suns, shack mesh omitted — the still then shows only the grid. For `beauty`: `rc.add_sun` key/fill as `render_compare.build` does, `rc.add_world()`, shack proxy added. Camera: `sl.camera(extent, rc.WIDTH / rc.HEIGHT)` with `extent = (domain_x, domain_y, domain_z)` from the scene JSON (`dims[i] * domain_size / max(dims)`), object location `(offset,)*3` with `offset = placement.cell_offset(dx, True)`.

`render_pass(pass_name, …)`: `bpy.context.scene.render.filepath = out`, `bpy.ops.render.render(write_still=True)`, exit if the file is absent/empty, then load the PNG (`img = bpy.data.images.load(out); px = list(img.pixels); w, h = img.size`) and run the checks:
- `beauty`: `sl.warm_pixels(px) > 0` else `sys.exit("beauty still has no warm pixels: the flame did not render")`.
- `fuel`: `bbox = sl.lit_bbox(px, w, h)`; `sys.exit` if `None`.
- `char`: `bbox` not `None`, and the char bbox lies inside the shack's projected x-range: compute the shack's x-range in pixels from `sl.SHACK_AT[0] ± sl.SHACK_SIZE[0]/2` using the camera's `ortho_scale` and image width (helper `project_x(x_m, cam, width)` — add it to `shack_layout` with a test in Task 7's file if not already there; do it here and add the test in this task), tolerance ±2 voxels; `sys.exit` with both boxes if outside.
- `fuel` **and** `char`: the fuel bbox must extend beyond the char bbox (fuel is also in the jet); `sys.exit` if `fuel_bbox ⊆ char_bbox` — this is the check that fails if the two grid names are swapped.
Checks that need both passes run in `main` after all three renders.

`main`: parse `sys.argv[sys.argv.index("--") + 1:]` into five args; `frame = int(...)`; print `render_shack: {out} on {device} in {secs:.1f} s`; write `OUT_DIR/render-device`.

- [ ] **Step 2: Smoke run by hand** on the Task 7 probe bake (the frame-30 file):
`/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python-exit-code 1 --python tests/bench/render_shack.py -- target/ft5/velocity shack 30 examples/jet_shack_render.elements target/ft5/still`
Expected: three PNGs in `target/ft5/still/` and exit code 0. A check failure here means either a real pipeline problem or a bad threshold; diagnose (look at the numbers printed, not the pictures) before changing a threshold, and record any change.
- [ ] **Step 3: Mutations (one each, on the script; re-run the smoke):**
  1. `attr.attribute_name = "char"` in `pass_material` for the fuel pass only → the fuel/char bbox-containment check exits nonzero.
  2. `temp.attribute_name = "density"` (constant-ish temperature) → the warm-pixel check does **not** necessarily fail; record whether it does. If it passes, say so in the commit — this check proves flame emission, not the temperature map, and the Kelvin map is covered by the pure test and by eye.
  3. `strength.inputs[1].default_value = 0.0` → `beauty` has no warm pixels → exit nonzero.
  4. Rename the `flame` attribute to `flam` → beauty check fails.
- [ ] **Step 4: Ruff + py-test:** `ruff check tests/bench && ruff format --check tests/bench && just py-test` — PASS.
- [ ] **Step 5: Commit** ("Render a baked shack frame in Cycles"; body: what the three stills are, which checks fail under which mutation and the one that does not, the thresholds' provenance).

---

### Task 9: `just render-shack`, the record, and status

**Files:**
- Modify: `justfile`, `AGENTS.md`
- Create: `docs/bench/render-shack/README.md` and the three PNGs
- Modify: `.superpowers/sdd/progress.md` (append FT5 lines)

**Interfaces:**
- Produces: `just render-shack frame="60"` — builds the CLI, bakes `examples/jet_shack_render.elements` for frames 1..`frame` once (`--frames {{frame}}` bakes from start and writes that frame only), renders, writes PNGs and `README.md` to `docs/bench/render-shack/`.

- [ ] **Step 1: Add the recipe** beside `bench-render`:

```make
# FT5: bake the half-size shack scene (every grid, one bake) and render one frame of it in
# Cycles. Real GPU and Blender; not in `check`. Stills go in docs/bench/render-shack/.
render-shack frame="60":
    #!/usr/bin/env bash
    set -euo pipefail
    BLENDER_BIN="${BLENDER_BIN:-/Applications/Blender.app/Contents/MacOS/Blender}"
    cargo build --release -p elements-cli
    CLI=target/release/elements
    SCENE=examples/jet_shack_render.elements
    R=target/ft5/render
    OUT=docs/bench/render-shack
    mkdir -p "$R" "$OUT"
    commit="$(git rev-parse --short HEAD)"
    [ -z "$(git status --porcelain -- crates Cargo.toml Cargo.lock tests/bench examples)" ] || commit="$commit-dirty"
    dx="$(python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(d['domain_size'] / max(d['dims']))" "$SCENE")"
    echo "bake shack frame {{frame}}; load $(sysctl -n vm.loadavg)"
    "$CLI" bake "$SCENE" --out "$R" --frames {{frame}} --name shack --voxel-size "$dx"
    echo "render; load $(sysctl -n vm.loadavg)"
    "$BLENDER_BIN" --background --factory-startup --python-exit-code 1 \
        --python tests/bench/render_shack.py -- "$R" shack {{frame}} "$SCENE" "$OUT"
    echo "$commit" > "$OUT/commit"
    echo "wrote $OUT (commit $commit, load $(sysctl -n vm.loadavg))"
```

- [ ] **Step 2: Run it** — `just render-shack 60` — Expected: exit 0, three PNGs in `docs/bench/render-shack/`, a `commit` file. Then `ls -l docs/bench/render-shack` and check each PNG is non-empty.
- [ ] **Step 3: Look at what the recipe cannot judge.** Agents cannot see images. Ask the user to open the three PNGs and say whether the beauty still reads as a flame against a shack; record their words verbatim in the README under "Verdict (recorded …)". Do not describe the pictures yourself.
- [ ] **Step 4: Write `docs/bench/render-shack/README.md`:** what the recipe runs; the commit (`cat commit`); load averages printed; device (`render-device`); the grids in the file (`python3 -c` listing via `vdb_rs`? use Blender's `GRIDS` print from Task 7 at frame 60 — rerun that scratch script and paste it); file size per frame (`ls -l target/ft5/render/shack.0060.vdb`); the bake wall time; each image check and the thresholds; which mutation the temperature-map check does not catch (Task 8); **explicitly** that char is a volume still not a shader on the planks, that Mantaflow is not involved, that timings were under whatever load was printed, and that llvmpipe CI did not run the render. Reference the spec and this plan.
- [ ] **Step 5: Status updates.** In `AGENTS.md` "What this is": after the FT4 paragraph add one qualified sentence: FT5 is complete locally on Metal (Apple M1 Max); one bake of `examples/jet_shack_render.elements` writes eight grids per frame, `just render-shack` renders them, llvmpipe CI has not run the render; link the spec, plan and `docs/bench/render-shack/README.md`; and add the Commands entry `just render-shack # FT5: bake + Cycles still of the shack scene; real GPU and Blender, not in check`. Append to `.superpowers/sdd/progress.md` one line per task with commit hashes and the mutation results. Do not claim CI status you have not checked (`gh run list`).
- [ ] **Step 6: `just check`** — PASS. Commit ("Record FT5: multi-grid export and the shack render"; body: what was measured, what was not (CI, the picture), decisions made — the spec deviations from Task 3).

---

## Self-Review

**Spec coverage:** §3.1 fuel output → Task 4. §3.2 `outputs`, version 4, validation (with the recorded deviation) → Task 3; graph support → Task 2. §3.3 bake, one evaluation per frame, velocity averaging, partial-file deletion, `--name` semantics → Task 5. §3.4 writer → Task 1. §3.5 Blender template items 1–6 → Tasks 7–9 (contract check and velocity in Tasks 7–8, materials Task 8, recipe Task 9; shack proxy uses the scene's actual planks mesh instead of grey boxes: same purpose, less to invent). §4 error handling → Tasks 1, 3, 5, 8. §5 testing list → every bullet has a test in Tasks 1–8; the Blender checks are in Task 8. §6 out of scope → nothing implemented. §7 risks → Task 7 step 6 (velocity), Task 8 (blackbody by eye), Task 9 (file size recorded), Task 4 step 4 (interface consumers).

**Known gaps (deliberate):** `Timeline.goto` computes every extra on every simulated frame, including frames a bake skips; the cost is six GPU copies against a ~70 ms solver step, so it is left and measured in Task 9's bake wall time (record the per-frame number; if material, a follow-up can gate extras on the target frame). The llvmpipe CI does not exercise Blender.

**Placeholder scan:** Test steps name the exact assertions; the two places that depend on reading a helper first (Task 2's pool counter and accumulate node, Task 4's fuel accessor) say which file to copy from. Expected vdb values in Task 5 step 1 are to be recomputed from the stated index formula before commit — the plan states the correct formula and one worked value.

**Type consistency:** `DocOutput { node, socket, name }`, `Graph::set_extra_outputs(Vec<SocketId>)`, `Evaluated { value, extras, stats }` with `release_to`, `GridSpec { name, values }`, `write_float_grids`, `cell_centred_velocity`, `expanded_names`, `shack_layout` names — used identically across tasks.
