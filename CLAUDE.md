# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An open-source Rust suite of real-time VFX authoring tools for Blender, covering
the problem space of the JangaFX tools (volumetric gas, liquids, terrain,
procedural assets) and using ML where it is proven rather than as a replacement
for solvers. JangaFX product names are trademarks and appear in the docs only to
identify prior art — never name a component after one.

The suite is being built product by product. Only **Core v1** exists so far: a
thin vertical slice proving every architectural seam before any solver is
written.

- Design spec: `docs/superpowers/specs/2026-09-19-elements-suite-core-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-19-elements-core-v1.md`
- Live progress and open risks: `.superpowers/sdd/progress.md`

**Read the progress ledger before starting work.** It records which of the 18
tasks are complete, the commits that prove it, and decisions already made. Tasks
listed complete there are done — do not redo them.

## Commands

`just check` is the commit gate. Everything else is in the `justfile`
(`just --list`).

```bash
just check        # lint + test; must pass before any commit
just test         # cargo nextest run --workspace, native GPU backend
just lint         # cargo fmt --check, clippy -D warnings, ruff check
just addon        # build + verify the Blender extension ZIP
just blender-test # Blender integration test; finds the macOS app bundle
just golden       # regenerate golden files (review the PNG by eye first)
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

- `GpuContext::scoped` captures **encoding-time** validation only. `queue.submit`
  returns before the GPU runs anything, so it cannot catch device-timeline
  faults; `Field::read_back`'s `poll` is what actually waits. Do not claim more
  than this in error handling.
- Bit-exact assertions are legitimate **within one backend on one machine**
  (determinism). Cross-backend comparisons use tolerance — Metal and software
  Vulkan genuinely disagree about what counts as a validation error.
- **CI has never run.** There is no git remote. Everything is verified on Metal
  only; passing locally is not evidence of passing on lavapipe. Adding a remote
  and fixing the resulting failures is a mandatory close-out step.

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
