# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An open-source Rust suite of real-time VFX authoring tools for Blender, covering
the problem space of the JangaFX tools (volumetric gas, liquids, terrain,
procedural assets) and using ML where it is proven rather than as a replacement
for solvers. JangaFX product names are trademarks and appear in the docs only to
identify prior art — never name a component after one.

The suite is being built product by product. **Core v1** is merged. **Ember**
(grid gas) is built in four pieces. Piece 1, core sim foundations, is merged:
persistent state, a timeline with a frame cache, staggered vector fields, and
multi-consumer graph outputs. Piece 2 is split at a speed gate. **2a** is
complete: the `elements-ember` crate, a GPU smoke solver that passed the 128³
gate at 34 ms a step with 160 pressure iterations. **2b** is split into cycles:
three for the solver and its benchmark, then flame. **2b-1**, solver
correctness, is complete: CFL substeps, RK2 + MacCormack advection, vorticity
confinement, dissipation, per-face boundaries, and the `preview` and `final`
presets (`docs/bench/presets.md`). Preview runs one CFL-clamped substep, now
about 71 ms a 128³ frame. Open risks in §6 of the piece 2 spec: the open
part of (g), GPU out-of-memory on unified memory; (j), preview's frame cost;
(k), preview's budget after 2b-2's solids; (m), preview's ×4
under-converging around thin colliders; and (n), fire's preview cost.
**2b-2**, scene content, is complete: keyframed box and
sphere emitters (with noise and velocity emission) and colliders, unions of
each, and wind
(`docs/superpowers/specs/2026-09-23-ember-scene-content-2b2-design.md`).
**2b-3**, the Mantaflow benchmark, is complete
(`docs/bench/results-2b3/results.md`). It found Ember faster but behind on
divergence and conservation, and its wind broken. **2b-3c**, solver
quality, is complete. It added MGPCG pressure (preview ×4, final ×10), a
global mass correction, and wind as an ambient airflow. In the rerun
(`docs/bench/results.md`), Ember is 5.8–8.4× faster than Mantaflow. In
`plume` and `plume_collider` its mass drift is under 1e-6 of the frame-60
mass at frame 80, and in `plume_collider`, where almost no smoke leaves, it
stays under 1.5e-6 through frame 120. It leaves less divergence everywhere
except `plume_collider` and `plume`'s frame 120 at 256³. All timings were
measured under load. **2b-3b**, the side-by-side render and the latency
measurement, is complete (`docs/bench/latency.md`, `docs/bench/render/`):
after a parameter change at 128³ Ember reaches frame N about 7–10× sooner than
Mantaflow re-bakes (under load, which may favour Ember), and the renders show
Ember matching Mantaflow's large-scale shape in `plume` and `plume_collider` with
visibly less fine detail (`plume_wind` differs by construction).
**2b-4**, fire, is complete
(`docs/superpowers/specs/2026-09-25-ember-fire-2b4-design.md`): a fuel input
that burns into heat and smoke, a `flame` output (√react), and a `fire`
benchmark scene. While fire burns, velocity is traced with Euler rather than
MacCormack, which blew up flame vorticity at one substep, and fuel is clamped
to [0, 10] at emission as in Mantaflow. In the fire benchmark
(`docs/bench/results.md`, Fire) Ember is 4.9–6.0× faster, holds 0.88–1.42×
Mantaflow's fuel at frame 60, and leaves 4–28× Mantaflow's divergence. A 128³
fire preview frame takes 103.5 ms under load, over budget: risk (n)
(`docs/bench/presets-fire.md`). The side-by-side render is in
`docs/bench/render/`, its fire verdict pending. The node editor (`docs/superpowers/specs/2026-09-25-ember-node-editor-design.md`,
branch `ember-node-editor`) is designed and planned but paused.

- Core design spec: `docs/superpowers/specs/2026-09-19-elements-suite-core-design.md`
- Core v1 plan: `docs/superpowers/plans/2026-09-19-elements-core-v1.md`
- Ember umbrella spec: `docs/superpowers/specs/2026-09-21-ember-design.md`
- Ember piece 1 spec: `docs/superpowers/specs/2026-09-21-ember-core-sim-foundations-design.md`
- Ember piece 1 plan: `docs/superpowers/plans/2026-09-21-ember-core-sim-foundations.md`
- Ember piece 2 spec: `docs/superpowers/specs/2026-09-21-ember-solver-design.md`
- Ember piece 2a plan: `docs/superpowers/plans/2026-09-22-ember-solver-2a.md`
- Ember piece 2b-1 spec: `docs/superpowers/specs/2026-09-22-ember-solver-2b1-design.md`
- Ember piece 2b-1 plan: `docs/superpowers/plans/2026-09-22-ember-solver-2b1.md`
- Ember piece 2b-2 spec: `docs/superpowers/specs/2026-09-23-ember-scene-content-2b2-design.md`
- Ember piece 2b-2 plan: `docs/superpowers/plans/2026-09-23-ember-scene-content-2b2.md`
- Ember piece 2b-3 spec: `docs/superpowers/specs/2026-09-23-ember-mantaflow-benchmark-2b3-design.md`
- Ember piece 2b-3 plan: `docs/superpowers/plans/2026-09-23-ember-mantaflow-benchmark-2b3.md`
- Ember piece 2b-3b spec and plan: `docs/superpowers/specs/2026-09-25-ember-render-latency-2b3b-design.md`, `docs/superpowers/plans/2026-09-25-ember-render-latency-2b3b.md`
- Latency and render comparison: `docs/bench/latency.md`, `docs/bench/render/README.md`
- Ember piece 2b-3c spec: `docs/superpowers/specs/2026-09-24-ember-solver-quality-2b3c-design.md`
- Ember piece 2b-3c plan: `docs/superpowers/plans/2026-09-24-ember-solver-quality-2b3c.md`
- Ember piece 2b-4 spec and plan: `docs/superpowers/specs/2026-09-25-ember-fire-2b4-design.md`, `docs/superpowers/plans/2026-09-25-ember-fire-2b4.md`
- Fire preview cost: `docs/bench/presets-fire.md`
- Mantaflow benchmark results and cache notes: `docs/bench/results.md` (2b-3c's rerun; 2b-3's run is in `docs/bench/results-2b3/`), `docs/bench/mantaflow-notes.md`
- Solver gate (Gauss–Seidel, multigrid and MGPCG): `docs/bench/solver-gate.md`
- Speed gate, iteration sweep and presets: `docs/bench/speed-gate.md`, `docs/bench/iteration-sweep.md`, `docs/bench/presets.md`
- Live progress and open risks: `.superpowers/sdd/progress.md`

**Read the progress ledger before starting work.** It records which tasks of
each plan are complete, the commits that prove it, and decisions already made.
Tasks listed complete there are done — do not redo them.

## Commands

`just check` is the commit gate. Everything else is in the `justfile`
(`just --list`).

```bash
just check        # lint + test; must pass before any commit
just test         # cargo nextest on the native GPU backend, plus tests/bench/test_mapping.py
just lint         # cargo fmt --check, clippy -D warnings, ruff check
just addon        # build + verify the Blender extension ZIP
just blender-test # Blender integration test; finds the macOS app bundle
just golden       # regenerate golden files (review the PNG by eye first)
just bench-gate   # the 128³ speed gate; minutes long, real GPU, not in `check`
just bench-sweep  # pressure-iteration sweep past the gate, for choosing presets
just bench-presets # the preview preset's substep cap (`PRESET_SCENE=fire` for fire's cost); minutes, real GPU
just bench-solver # the 2b-3c solver gate; about an hour, real GPU, not in check
just bench-latency # 2b-3b latency, Ember and Mantaflow at 128³; about 30 min, real GPU and Blender
just bench-render # 2b-3b side-by-side Cycles stills into docs/bench/render/; needs Blender
just bench        # the Mantaflow benchmark; an hour or more, real GPU and Blender, not in check
                  # (positional: `just bench plume 64` for one scene and resolution)
```

Single test: `cargo nextest run -p elements-core --test noise` (a test *file*),
or `cargo test -p elements-core --test noise -- noise_is_deterministic_for_a_seed`
for one case.

**Never set `WGPU_BACKEND` locally.** This is macOS on Apple Silicon and the
backend is Metal; `WGPU_BACKEND=vulkan` belongs only to `just ci-test`, which
targets software Vulkan on Linux.

## Toolchain ownership

Two managers, deliberately non-overlapping — do not let them both own a tool:

| Tool | Owner |
|---|---|
| Rust, cargo, clippy, rustfmt | `rustup` via `rust-toolchain.toml` |
| python, ruff, uv, just, cargo-nextest | `mise` via `mise.toml` |
| the same environment, declaratively | `flake.nix` (alternative entry point) |

`.cargo/config.toml` pins the macOS linker to `/usr/bin/cc`. This is load-bearing:
if Nix or another toolchain puts a `cc` earlier on `$PATH`, every link fails with
"symbol(s) not found for architecture arm64". See README.md.

## Architecture

One long-lived engine process; Blender is a thin client. The engine never links
`bpy`, which is what lets it stay permissively licensed and run headless in CI.

```
elements-core/  wgpu device, fields, field pool, compute dispatch, node graph
elements-ember/ smoke solver; registers node kinds, and the CLI and daemon
                build their registry with `elements_ember::registry()`
elements-io/    OpenVDB write, npy golden output, png previews
elements-ipc/   NDJSON control plane + mmap'd double-buffered data plane
elementsd/      the daemon binary
elements-cli/   headless bake and preview
addon/          pure-Python Blender extension (GPL-3.0-or-later)
```

Four decisions that are not obvious from the code:

1. **Out-of-process daemon.** Blender's renderer is territorial about the GPU
   device, so the engine owns its own. A solver panic shows a disconnect banner
   rather than losing a `.blend`.
2. **The node graph lives in core.** All four future products are node graphs
   over different data types; products register node *kinds*, they do not define
   their own graph.
3. **ML is a seam, never a dependency.** Every product must produce correct
   output with all models unloaded. Model weights ship as separate downloads,
   never inside the extension ZIP.
4. **Split licensing.** `crates/*` is `Apache-2.0 OR MIT`; `addon/` is
   `GPL-3.0-or-later`. **No GPL code may be copied into any `elements-*` crate.**
   Published algorithms may be reimplemented; source may not be vendored.

### Deviations from the spec worth knowing

The plan's "Deviations from the spec" section explains these in full. In short:
the control plane is newline-delimited JSON rather than CBOR and the data plane
is a memory-mapped file rather than POSIX shm, both so the Blender add-on can be
**pure Python with no wheel** — Python has no stdlib CBOR, and `SharedMemory`
gained its `track` parameter only in 3.13 while Blender 4.2/5.0 ship 3.11.

Separately: **no Rust crate can write OpenVDB.** `vdb-rs` is read-only and
unmaintained; the `openvdb` crate is an empty placeholder. `elements-io` contains
a hand-written minimal `FloatGrid` writer built from the byte layout in `vdb-rs`'s
parser, with `vdb-rs` used as the read-back oracle in tests.
`vdb-rs` 0.6.0 also misreads vector grids: it parses a `Vec3s` grid's root
values at 4 bytes and returns an empty tree. The workspace therefore vendors a
patched copy in `vendor/vdb-rs/` through `[patch.crates-io]` (see
`PATCHED.md`). It is a dev-dependency only, so the daemon never links it.

## Constraints that bite

- **Scalar fields are `R32Float`, never `R16Float`.** Verified in the wgpu source:
  the WebGPU baseline excludes `STORAGE_BINDING` from `R16Float`'s allowed usages
  entirely. Using it requires an optional adapter feature and forfeits
  portability. Half precision belongs on the wire and in exports, not in storage
  textures.
- **`required_features` must stay `wgpu::Features::empty()`.** The engine depends
  on no optional wgpu features. If something seems to need one, that is a design
  problem, not a flag to enable.
- **`#![forbid(unsafe_code)]` in every crate except `elements-ipc`**, which needs
  it for the mmap and documents a `# Safety` section per block.
- **Crate manifests use `dep.workspace = true`.** Versions live only in the
  workspace `Cargo.toml`.
- **Every stochastic node takes an explicit `seed: u64`.** No implicit entropy.
- **`gpu::required_limits` takes the resolution limits and `max_buffer_size`
  from the adapter**, since `downlevel_defaults` caps 3D textures at 256 and
  buffers at 256 MiB. Everything else stays at `downlevel_defaults`. Notably,
  **4 storage textures per shader stage**: read fields through
  `texture_3d<f32>` + `textureLoad`, and write through storage.
- **Kernels with solids** include `solid.wgsl` and must declare
  `var solid: texture_3d<f32>`. When there is no collider, a 1×1×1 placeholder
  is bound and `has_solids` = 0 keeps it unread.
- **A snapshot for frame N is the state entering N.** Producing N is always
  "restore or reach the entering state, then one `eval`". Never cache outputs.
- Blender extension ZIPs put `__init__.py` and `blender_manifest.toml` at the
  **archive root**, with nothing nested under a package directory, and the addon
  package uses relative imports only.

## Verifying wgpu APIs

**Check the vendored crate source, not docs.rs.** Six API errors in the plan came
from documentation that omits required struct fields:

```bash
R=$(find ~/.cargo/registry/src -maxdepth 2 -type d -name 'wgpu-types-30.0.1' | head -1)
grep -rn 'pub struct DeviceDescriptor' $R/src/
```

Known corrections already applied: `InstanceDescriptor::new_without_display_handle_from_env()`
passed by value; `RequestAdapterOptions` needs `apply_limit_buckets`;
`DeviceDescriptor` needs `experimental_features`; `PollType::wait_indefinitely()`
is a function, not a unit variant; `Texture::global_id()` does not exist.

## Testing

Agents cannot see rendered output, so a test that cannot fail is
indistinguishable from working code. Two rules, both learned by finding
decorative tests already merged:

1. **Prove each test can fail.** Mutate the code it covers, watch it fail,
   restore, and record the real output.
2. **A mutation changes exactly one thing.** A compound mutation proves nothing
   about either change.

Other things that matter here:

- `GpuContext::scoped` captures validation, out-of-memory and internal errors
  **at encoding time**. An error raised outside any scope is reported by the
  next `scoped` call, possibly an unrelated one. `queue.submit` returns before
  the GPU runs anything, so it still cannot catch device-timeline faults.
  Blocking waits go through `GpuContext::wait`, which turns a panicking `poll`
  into `DeviceLost`. Do not claim more than this in error handling.
- Bit-exact assertions are legitimate **within one backend on one machine**
  (determinism). Cross-backend comparisons use tolerance — Metal and software
  Vulkan genuinely disagree about what counts as a validation error.
- **CI runs on every push**, on llvmpipe (software Vulkan, named in the log), at
  https://github.com/bgyss/blender-elements-suite. Passing on Metal locally is
  not evidence of passing on lavapipe, so check the run: `gh run list`.

## Commits

Plain imperative subject line, then a body explaining **why** — not a bulleted
list of what changed. End with:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: <session url>
```

The repo is **jj-colocated**. Implementation commits are made with plain `git`;
jj imports them on your next jj command. Do not mix `jj` commands into the
implementation flow.
