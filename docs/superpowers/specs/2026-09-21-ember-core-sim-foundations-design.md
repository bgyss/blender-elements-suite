# Ember Piece 1 — Core Sim Foundations Design

**Date:** 2026-09-21
**Status:** Approved for planning
**Parent:** `2026-09-21-ember-design.md` (piece 1 of 4)

## 1. Goal

Give `elements-core` everything a time-stepping solver needs, and prove it with
a toy stateful node rather than a real solver. When this piece is done:

- a graph can hold state that persists across frames;
- the frame number reaches the nodes;
- scrubbing to any frame gives bit-identical results however it is reached;
- velocity can be stored as a staggered vector field.

No gas simulation is written here.

This closes these items from "Notes for the next plan (Ember)" in the Core v1
plan: several consumers per output, partial writes reading stale data, state
across frames, and time. The remaining two (in-memory volume handoff, f16 on
the wire) belong to piece 3.

## 2. Components

### 2.1 Staggered vector fields

- `SocketType::VectorField` and `Value::VectorField(StaggeredField)`.
- `StaggeredField` holds three `R32Float` `Field`s, the x, y and z faces, plus
  the domain's cell dims. Face dims are `(nx+1, ny, nz)`, `(nx, ny+1, nz)` and
  `(nx, ny, nz+1)`, and the constructor rejects any other shape.
- `EvalCtx::acquire_vector_zeroed()` / `acquire_vector_uninit()` build one from
  the pool. `FieldPool` is already keyed by `(dims, format)`, so face textures
  pool without change.
- `Value::as_vector_field()` behaves like `as_field()` and reports `TypeMismatch` with the same sentinel.
- `shaders/vector_sample.wgsl` is a WGSL include with trilinear sampling done by
  hand, for a single face texture and for the vector at a cell centre. Every
  32-bit float format in the WebGPU baseline is unfilterable (Ember umbrella
  E2). Piece 2 advection and piece 3 export both build on this.

### 2.2 Several consumers per output

- Remove the single-consumer rule: `Graph::connect` no longer returns
  `AlreadyConsumed`, and the variant is deleted. Each *input* still accepts
  exactly one edge.
- Before evaluating, the evaluator counts the consumers of each output socket.
- `EvalCtx::input(i)` borrows as today.
- `EvalCtx::take_input(i)` moves the value if the calling node is the last
  remaining consumer. Otherwise it returns a GPU copy, made with
  `CommandEncoder::copy_texture_to_texture` (a core feature, not an optional
  one) into a pooled field. Scalars are just copied.
- Every GPU copy increments `EvalStats::copies`, which `Graph::eval` returns
  next to the value. It is how tests prove that a graph with no branching never copies.

### 2.3 Release at last use

After an output's last consumer has run, any fields still held for that output
are **released to the pool**. Outputs with no consumers are released right after
their node runs, except the graph's output.

This fixes a Core v1 gap. Today `Graph::eval` *drops* the values that no node
consumes, which frees the texture instead of pooling it, so the next frame
allocates again.

### 2.4 Explicit acquire contract

- `EvalCtx::acquire` is renamed to `acquire_uninit` so the contract is in the
  name: "contents are whatever the last user left."
- New `acquire_zeroed` runs the existing `constant.wgsl` pass at 0.
  `CommandEncoder::clear_texture` is **not** usable, because it needs the optional
  `Features::CLEAR_TEXTURE` (verified in `wgpu-types` 30.0.1, `features.rs`).
- Existing nodes that write every voxel move to `acquire_uninit` with no change in behaviour.

### 2.5 Persistent state

- `StateStore` sits beside `FieldPool` in the session. It maps
  `(NodeId, &'static str slot)` to a `Value` (`Field`, `VectorField` or
  `Scalar`), and its contents survive across `Graph::eval` calls.
- `Node::stateful(&self) -> bool` defaults to `false`. Only stateful nodes can
  call `ctx.state()`, which returns a handle scoped to that node's id. Calling it
  from a stateless node is `NodeError::NotStateful`.
- On the first step, and after a reset, every slot is empty. A node initialises
  its slots itself, usually with `acquire_zeroed`.
- If a stored field's dims do not match the current domain,
  `NodeError::StateShape` is raised. The timeline handles it (§2.6).
- `StateStore::snapshot(&GpuContext, &mut FieldPool) -> Snapshot` GPU-copies
  every slot into pooled fields. `restore(&snapshot)` releases the store's
  current fields and **copies** the snapshot back in. It copies rather than
  moves because the cache keeps the snapshot for the next scrub to that frame.
  A snapshot reports its size in bytes.

### 2.6 Timeline

`Timeline { fps, start_frame, cache }` owns time for one loaded graph.

**Snapshot meaning.** A cached snapshot for frame N is the state *entering*
frame N, taken before that frame's `eval`. So producing frame N always means
restoring the state entering N and running exactly one `eval`. A cache hit
costs one step, and the output never needs caching separately.

**`goto(frame)`:**

1. A frame below `start_frame` is clamped to `start_frame`.
2. If the graph has no stateful nodes, run one `eval` and stop. Behaviour and
   cost are the same as Core v1.
3. If the state currently held is the state entering `frame`, which is the common
   case during forward playback, run one `eval`.
4. Otherwise, find the nearest cached snapshot at or before `frame`. If there is
   one, restore it. If not, reset (empty the state) and start at `start_frame`.
   Step forward, taking a snapshot entering each frame, until the state is
   the state entering `frame`, then run one `eval`.

**Time.** `EvalCtx::time()` returns `Time { frame, seconds, dt }`, where
`seconds = (frame - start_frame) as f64 / fps`, and `dt = 1 / fps`.
Substepping is the solver's own business (piece 2). The timeline steps whole frames.

**Cache budget.** The budget is `cache_budget_mb`. When adding a snapshot would
exceed it, the snapshot evicted is the **least recently used** one, never the
one just restored. A snapshot larger than the whole budget disables caching for
that session, logs a warning, and simulation carries on (scrubbing is just slower).

**Invalidation.** Loading a document (the only way parameters change today)
discards the cache and resets. `NodeError::StateShape` triggers a reset and one
retry. If the error recurs straight after that reset, it is a node bug and is surfaced.

### 2.7 Document

Three new optional fields. **`ELEMENTS_DOC_VERSION` becomes 2.** Serde ignores
unknown fields, so without a bump an older engine would accept a new document
and silently drop `fps`. Version-1 documents still load, with the defaults below:

- `fps: f64`: defaults to 24.0. It must be finite and > 0, otherwise
  `DocError`.
- `start_frame: u32`: defaults to 1, Blender's default.
- `cache_budget_mb: u32`: defaults to 2048.

### 2.8 Daemon and CLI

- `Command::Render { frame }` goes through `Timeline::goto(frame)`. The protocol's
  shape is unchanged, so there is no version bump.
- A device lost mid-sim discards the cache and state, and `ErrorKind::DeviceLost`
  is surfaced as today, so the client reconnects rather than retrying.
- `elements bake --frames A-B` (the existing range syntax) drives the timeline
  from `start_frame` and writes only frames A to B. For a stateful graph, each
  file differs, and `--frames 5` on its own is the true frame 5, not a first step
  that happens to be labelled 5.
- **Add-on (the one Python change in this piece).** The viewport geometry is
  produced by a CLI bake, not by the daemon (see `handlers.py`, KNOWN GAP), and
  that bake hard-codes `--frames 1`. It must pass `scene.frame_current` and read
  back the matching `density.NNNN.vdb`. Otherwise the viewport shows frame 1
  while the daemon has simulated frame N. Each viewport update of frame N
  re-simulates from `start_frame` in a fresh process. That is acceptable for
  piece 1's toy graphs, and piece 3's in-memory handoff removes it.

### 2.9 Device limits

`GpuContext` requests `Limits::downlevel_defaults().using_resolution(adapter.limits())`.
Without it, `max_texture_dimension_3d` is 256, and a staggered face of a 256³
domain (257 wide) cannot be allocated. This is a limit, not a feature, so
`required_features` stays empty. The daemon requires each domain axis to be
strictly below the limit, so faces always fit.

### 2.10 Proof node: `core.accumulate`

Inputs: `Field`. Outputs: `Field`. Stateful, with one slot `"sum"`.
Each step does `sum += input * dt` and outputs a copy of `sum`. With a
constant input `c` and frames starting at `start_frame`, the output at frame N
is `c * (N - start_frame + 1) / fps`. That closed form is cheap to check on the CPU.

## 3. Errors

Every change stays total, and untrusted documents never cause a panic.

| Condition | Behaviour |
|---|---|
| Stored field has the wrong dims | `StateShape` → reset and retry once, then surface |
| `ctx.state()` from a stateless node | `NodeError::NotStateful` (node bug) |
| `fps` is not finite or ≤ 0 | `DocError` at load |
| Snapshot is larger than the budget | caching disabled, warning logged, sim continues |
| Device lost | cache and state discarded, `ErrorKind::DeviceLost` |

## 4. Testing

Each test is proven able to fail by a **single** mutation, and the real failing
output is recorded in the task report (CLAUDE.md "Testing").

| Test | Mutation that must fail it |
|---|---|
| `accumulate` matches the closed form at frame N, reached in order | step twice per frame |
| same, reached by scrubbing backwards (a cache hit) | skip the restore |
| same, after its snapshot was evicted | off-by-one in the frame to step from |
| `goto(10)` bit-identical via 1→10 and via 1→15→3→10 | snapshot taken *after* `eval` instead of before |
| frames below `start_frame` equal `start_frame` | remove the clamp |
| branching graph: both branches correct; `copies ≥ 1` | move instead of copy for a non-final consumer |
| graph with no branching: `copies == 0` | always copy |
| `pooled_count` is stable over N frames | remove release-at-last-use |
| `acquire_zeroed` over a dirty recycled texture reads all zeros | skip the zero pass |
| staggered sampling reproduces a linear field exactly at faces and centres | swap the face offsets for one axis |
| `StaggeredField` rejects wrong face dims | remove the check |
| budget smaller than one snapshot: sim still correct, cache empty | treat an oversize snapshot as fitting |
| daemon: `Render{3}` then `Render{1}` stable and differs from frame 3 | ignore `frame` (the Core v1 behaviour) |
| Python client contract test under 3.11 with a stateful document | — (runs the real daemon) |
| `bake --frames` on a stateful graph gives distinct files | bake the same frame each time |
| `bake --frames 5` equals frame 5 of `bake --frames 1-5` | start the sim at A instead of `start_frame` |

## 5. Out of scope

- f16 on the wire, in-memory volume handoff, a disk-backed cache (piece 3).
- Substepping and any solver kernel (piece 2).
- Changing parameters without reloading the document.
- Blender UI for the timeline beyond passing the scene's current frame (§2.8).
- Making the viewport path fast. It stays a CLI bake per update until piece 3.
