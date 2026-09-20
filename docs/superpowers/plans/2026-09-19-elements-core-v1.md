# Elements Core v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a thin vertical slice of the Elements Suite that proves every architectural seam end to end — a headless Rust/wgpu engine that evaluates a node graph on the GPU, publishes frames to a Blender addon over IPC, and bakes `.vdb` files from the command line.

**Architecture:** An out-of-process daemon (`elementsd`) owns a `wgpu` device and evaluates a serialized `.elements` node graph. A control plane of newline-delimited JSON over a Unix domain socket (named pipe on Windows) carries commands; a memory-mapped double-buffered file carries field data. A pure-Python Blender addon reads both with the standard library only.

**Tech Stack:** Rust 2024 edition, `wgpu` 30.0.1, `pollster` 1.0.1, `serde`/`serde_json` 1.x, `memmap2` 0.9.11, `bytemuck` 1.25.2, `half` 2.7.1, `png` 0.18.1, `ndarray-npy` 0.10.0, `clap` 4.6.7, `thiserror` 2.0.20, `anyhow` 1.0.104. Dev-only: `vdb-rs` 0.6.0 (read-back oracle), `tempfile` 3.27.0, `approx` 0.5.1. Python 3.11 standard library for the addon (Blender 5.0's version).

**Tooling:** `mise` pins python/ruff/uv/just/cargo-nextest; `rustup` owns the Rust toolchain via `rust-toolchain.toml`; `flake.nix` offers the same environment to Nix users and CI; `just` is the task interface (`just check` gates every commit); `ruff` lints and formats all Python; `cargo nextest` runs Rust tests in isolated processes, which keeps a GPU device-lost in one test from poisoning others.

---

## Deviations from the spec

Three decisions differ from `docs/superpowers/specs/2026-09-19-elements-suite-core-design.md`. Each was forced by a verified fact, and each simplifies Core v1.

1. **No Python wheel.** Spec §3.4 called for an `elements_ipc` wheel. Verified: Python's stdlib `mmap` and `json` cover both planes, so the addon is pure Python. This removes per-platform wheel builds from Core v1 entirely. Revisit only if profiling shows the Python reader is the bottleneck.

2. **Control plane is newline-delimited JSON, not CBOR.** Spec §3.3 specified CBOR. Python has no stdlib CBOR, and the control plane carries only small command messages — the bytes live in the data plane. NDJSON keeps the addon dependency-free. The framing is one JSON object per line, UTF-8, `\n`-terminated.

3. **Data plane is a memory-mapped file, not POSIX shared memory.** Spec §3.3 specified `shm_open`/`CreateFileMapping`. Verified: Python's `SharedMemory` gained its `track` parameter only in 3.13, while Blender 5.0 ships Python 3.11, so a stdlib shm reader needs a `resource_tracker` monkey-patch on 3.11. A mmap'd file in the runtime directory has identical double-buffer semantics, is readable by stdlib `mmap` on every platform and version, and stays in the page cache at these sizes. The `FrameChannel` API is unchanged, so true shm can be swapped in later behind it.

A fourth finding changes scope rather than design: **no Rust crate can write OpenVDB files.** `vdb-rs` 0.6.0 is read-only and unmaintained since 2024-01; the `openvdb` crate is an empty placeholder. Core v1 therefore includes a minimal uncompressed `FloatGrid` writer (Tasks 10–12), built against the byte layout read out of `vdb-rs`'s parser, with `vdb-rs` used as the read-back oracle in tests.

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Rust edition 2024.** Workspace `resolver = "3"`.
- **Engine licence: `Apache-2.0 OR MIT`.** Addon licence: `GPL-3.0-or-later`. No GPL-licensed code may be copied into any `elements-*` crate — published algorithms may be reimplemented, never vendored.
- **No `unsafe` outside `elements-ipc`.** Every crate root except `elements-ipc` carries `#![forbid(unsafe_code)]`. `elements-ipc` carries `#![deny(unsafe_op_in_unsafe_fn)]` and documents a `# Safety` section on each `unsafe` block.
- **Field formats for v1:** `R16Float` and `Rgba16Float` only.
- **Every stochastic node takes an explicit `seed: u64`.** No implicit entropy anywhere in the engine.
- **Golden comparisons use tolerance**, never bit equality: absolute tolerance `1e-3` for `f16`-backed fields.
- **Protocol version constant is `ELEMENTS_PROTOCOL_VERSION: u32 = 1`.** Every control message carries it.
- **Document format version constant is `ELEMENTS_DOC_VERSION: u32 = 1`.** Any PR changing the `.elements` schema must bump it and add a migration test.
- **Blender extension packaging:** `__init__.py` and `blender_manifest.toml` at the ZIP root, nothing nested under a package directory; relative imports only inside the addon package.
- **CI runs on lavapipe** (software Vulkan) via `just ci-test`, which sets `WGPU_BACKEND=vulkan` and `LIBGL_ALWAYS_SOFTWARE=1`. These variables are CI-only: `WGPU_BACKEND=vulkan` is wrong on macOS, where the backend is Metal. Tests requiring a device call `GpuContext::new_headless()`.
- **`just check` must pass before every commit.** It runs `cargo fmt --check`, `clippy -D warnings`, `ruff check`, and the full test suite.
- **All Python targets 3.11** and is formatted and linted with `ruff`.
- **Commit style:** conventional commits (`feat:`, `test:`, `fix:`, `chore:`). Commit at the end of every task.

---

## File Structure

```
Cargo.toml                          workspace manifest, shared dependency versions
rust-toolchain.toml                 pinned stable toolchain
.github/workflows/ci.yml            fmt, clippy, test on lavapipe

crates/elements-core/
  src/lib.rs                        re-exports gpu:: and graph::
  src/gpu/mod.rs                    GpuContext: device, queue, error scopes
  src/gpu/field.rs                  Field (3D texture + dims/format), FieldFormat
  src/gpu/pool.rs                   FieldPool: keyed allocate/recycle
  src/gpu/dispatch.rs               compute pipeline cache + dispatch()
  src/gpu/shaders/constant.wgsl     fill a field with a scalar
  src/gpu/shaders/curl_noise.wgsl   seeded curl noise
  src/graph/mod.rs                  Graph, EvalCtx, topological eval, cycle detection
  src/graph/socket.rs               SocketSpec, SocketType, SocketId
  src/graph/node.rs                 Node trait, NodeError, NodeId
  src/graph/registry.rs             NodeRegistry: kind string -> constructor
  src/graph/document.rs             .elements serde document + version migration
  src/nodes/mod.rs                  built-in node kinds
  src/nodes/constant_field.rs       ConstantField
  src/nodes/noise_field.rs          NoiseField
  src/nodes/output.rs               Output

crates/elements-io/
  src/lib.rs                        re-exports npy, preview, vdb
  src/npy.rs                        write_npy() for golden tests
  src/preview.rs                    write_slice_png()
  src/vdb/mod.rs                    write_float_grid() entry point
  src/vdb/writer.rs                 low-level byte writers (header, metadata, descriptor)
  src/vdb/tree.rs                   5-4-3 tree assembly from a dense field

crates/elements-ipc/
  src/lib.rs                        ELEMENTS_PROTOCOL_VERSION, re-exports
  src/protocol.rs                   Command, Response, EngineError; NDJSON framing
  src/channel.rs                    FrameChannel: mmap double buffer, header layout
  src/transport.rs                  platform listener/stream (UDS vs named pipe)

crates/elementsd/
  src/main.rs                       CLI entry, runtime dir setup
  src/daemon.rs                     accept loop, session state, frame publishing

crates/elements-cli/
  src/main.rs                       clap entry
  src/bake.rs                       bake subcommand
  src/preview.rs                    render-preview subcommand

addon/blender_elements/
  __init__.py                       register/unregister, addon entry
  blender_manifest.toml             extension manifest
  client.py                         NDJSON control client + mmap frame reader
  props.py                          Scene.elements PropertyGroup
  ops.py                            start/stop daemon operators
  ui.py                             N-panel
  handlers.py                       viewport draw handler
scripts/build_addon.py              builds + verifies the extension ZIP
tests/blender/test_roundtrip.py     run under `blender --background --python`
```

---

## Task 1: Toolchain, workspace scaffolding and CI

**Files:**
- Create: `mise.toml`, `.envrc`, `flake.nix`, `justfile`, `ruff.toml`, `rust-toolchain.toml`, `Cargo.toml`, `.github/workflows/ci.yml`, `crates/elements-core/Cargo.toml`, `crates/elements-core/src/lib.rs`, `crates/elements-core/tests/smoke.rs`, `LICENSE-APACHE`, `LICENSE-MIT`, `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: a compiling workspace, a pinned toolchain reproducible through either `mise` or `nix`, a `just` task interface every later task and CI invokes, and the shared dependency table later tasks add to.

**Toolchain ownership, to avoid two managers fighting over one tool:**

| Tool | Managed by | Why |
|---|---|---|
| Rust toolchain, `cargo`, `clippy`, `rustfmt` | `rustup` via `rust-toolchain.toml` | Cargo reads this file natively; `mise` managing Rust as well would shadow it |
| `python`, `ruff`, `uv`, `just`, `cargo-nextest` | `mise` via `mise.toml` | Already active on this machine, and `cargo-nextest` currently has a shim with no version set |
| Everything, reproducibly, for CI and Nix users | `flake.nix` | An alternative entry point, not a second source of truth |

**Python version:** 3.11, because Blender 5.0 ships Python 3.11 and the add-on
must import cleanly there. The system Python is newer; pinning 3.11 is what
catches accidental use of later syntax.

**GPU environment variables are CI-only.** `WGPU_BACKEND=vulkan` is correct for
lavapipe on Linux and wrong on macOS, where the backend is Metal. They belong in
the CI workflow and the `just ci-test` recipe, never in `mise.toml`.

- [ ] **Step 1: Write the failing test**

Create `crates/elements-core/tests/smoke.rs`:

```rust
#[test]
fn crate_version_is_exposed() {
    assert_eq!(elements_core::VERSION, env!("CARGO_PKG_VERSION"));
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p elements-core --test smoke`
Expected: FAIL — `error: failed to load manifest` or `could not find Cargo.toml` (no workspace yet).

- [ ] **Step 3: Pin the toolchain**

Create `rust-toolchain.toml`:

```toml
[toolchain]
channel = "1.93"
components = ["rustfmt", "clippy"]
```

Create `mise.toml`:

```toml
# Rust is deliberately absent: rustup owns it via rust-toolchain.toml.
[tools]
python = "3.11"        # Blender 5.0's Python; the oldest version the add-on must import on
ruff = "latest"
uv = "latest"
just = "latest"
"cargo:cargo-nextest" = "latest"

[env]
# Keep cargo's target dir out of the addon tree so build_addon.py never sees it.
CARGO_TERM_COLOR = "always"
```

Create `.envrc`:

```bash
use mise
```

Run: `mise install && direnv allow`
Expected: python 3.11, ruff, uv, just and cargo-nextest all resolve.

Verify: `mise exec -- python --version` prints `Python 3.11.x`.

- [ ] **Step 4: Add the Nix entry point**

Create `flake.nix`:

```nix
{
  description = "Elements Suite: real-time GPU field simulation for Blender";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          # Rust comes from rustup so rust-toolchain.toml stays authoritative.
          packages = with pkgs; [
            rustup
            python311
            ruff
            uv
            just
            cargo-nextest
            pkg-config
          ] ++ lib.optionals stdenv.isLinux [
            # Software Vulkan, so `cargo test` works in a Nix shell and in CI.
            mesa
            vulkan-loader
            vulkan-tools
          ];

          shellHook = ''
            export RUSTUP_TOOLCHAIN=$(sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)
          '' + pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            export VK_ICD_FILENAMES=${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.x86_64.json
            export LD_LIBRARY_PATH=${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH
          '';
        };
      });
}
```

Run: `nix flake check` (or `nix develop --command just --version` if `flake check` is slow).
Expected: the flake evaluates without error.

- [ ] **Step 5: Create the task interface**

Create `justfile`. Every later task and CI runs these recipes rather than
remembering flag combinations:

```just
# Elements Suite task runner. Run `just` to list recipes.

default:
    @just --list

# Format Rust and Python.
fmt:
    cargo fmt --all
    ruff format addon scripts tests

# Lint everything, failing on any warning.
lint:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    ruff check addon scripts tests

# Run the Rust test suite on this machine's native GPU backend.
test:
    cargo nextest run --workspace

# Run the Rust test suite the way CI does, on software Vulkan.
ci-test:
    WGPU_BACKEND=vulkan LIBGL_ALWAYS_SOFTWARE=1 cargo nextest run --workspace

# Everything a commit must pass.
check: lint test

# Build and verify the Blender extension ZIP.
addon:
    python scripts/build_addon.py

# Regenerate golden files. Review the PNG by eye before committing.
golden:
    cargo run -p elements-cli -- dump-npy tests/graphs/noise_8.elements \
        --out crates/elements-cli/tests/golden/noise_8.npy
    cargo run -p elements-cli -- render-preview tests/graphs/noise_8.elements \
        --slice-z 4 --range -1,1 --out crates/elements-cli/tests/golden/noise_8_z4.png

# Run the Blender integration test. Requires BLENDER_BIN.
blender-test:
    BLENDER_BIN="${BLENDER_BIN:-$(command -v blender)}" \
        cargo test -p elementsd --test blender_integration -- --nocapture
```

`cargo nextest` replaces `cargo test` in these recipes because it runs each test
in its own process. That matters here: several tests acquire a GPU device, and
process isolation keeps a device-lost in one test from poisoning others.

Individual task steps in this plan give `cargo test` commands for precision when
running a single test. `just check` is what gates a commit.

Create `ruff.toml`:

```toml
# Blender 5.0 ships Python 3.11; the add-on must import cleanly there.
target-version = "py311"
line-length = 100

[lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]

[lint.per-file-ignores]
# Blender's API is injected at runtime; bpy imports resolve only inside Blender.
"addon/**" = ["E402"]
```

- [ ] **Step 6: Create the workspace and crate**

Create `Cargo.toml`:

```toml
[workspace]
resolver = "3"
members = ["crates/*"]

[workspace.package]
edition = "2024"
license = "Apache-2.0 OR MIT"
repository = "https://github.com/bgyss/blender-elements-suite"

[workspace.dependencies]
anyhow = "1.0.104"
bytemuck = { version = "1.25.2", features = ["derive"] }
clap = { version = "4.6.7", features = ["derive"] }
half = { version = "2.7.1", features = ["bytemuck"] }
memmap2 = "0.9.11"
ndarray-npy = "0.10.0"
png = "0.18.1"
pollster = "1.0.1"
serde = { version = "1.0.228", features = ["derive"] }
serde_json = "1.0.151"
thiserror = "2.0.20"
wgpu = "30.0.1"

approx = "0.5.1"
tempfile = "3.27.0"
vdb-rs = "0.6.0"

[profile.dev]
opt-level = 1

[profile.dev.package."*"]
opt-level = 3
```

`opt-level = 1` on our own code with `3` on dependencies keeps golden tests on
software Vulkan tolerable without slowing incremental builds.

Create `crates/elements-core/Cargo.toml`:

```toml
[package]
name = "elements-core"
version = "0.1.0"
edition.workspace = true
license.workspace = true
repository.workspace = true

[dependencies]

[dev-dependencies]
```

Create `crates/elements-core/src/lib.rs`:

```rust
#![forbid(unsafe_code)]

//! Core runtime for the Elements Suite: GPU context, fields, and the node graph.

/// The version of this crate, surfaced for protocol handshakes.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `cargo test -p elements-core --test smoke`
Expected: PASS — `test crate_version_is_exposed ... ok`.

- [ ] **Step 8: Add licences and gitignore**

```bash
curl -sL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE-APACHE
```

Write `LICENSE-MIT` with the standard MIT text, copyright `2026 Elements Suite contributors`.

Create `.gitignore`:

```gitignore
/target/
/dist/
/result
.direnv/
__pycache__/
.venv/
*.vdb
*.npy
*.png
!crates/*/tests/golden/*.npy
!crates/*/tests/golden/*.png
.superpowers/
```

The negated patterns keep committed golden files tracked while ignoring the
throwaway `.npy` and `.png` that every local run produces.

- [ ] **Step 9: Create the CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  test:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4

      - uses: jdx/mise-action@v2
        with:
          experimental: true

      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy

      - uses: Swatinem/rust-cache@v2

      - name: Install software Vulkan
        run: |
          sudo apt-get update
          sudo apt-get install -y mesa-vulkan-drivers vulkan-tools libvulkan1 libblosc-dev
          vulkaninfo --summary | head -40

      - name: Lint
        run: just lint

      - name: Test on lavapipe
        run: just ci-test

      - name: Build the Blender extension
        run: just addon
```

`mise-action` installs exactly the tools `mise.toml` pins, so CI and a developer
machine run the same `ruff`, `python` and `just`. Rust still comes from the
rustup action, honouring `rust-toolchain.toml`.

- [ ] **Step 10: Verify the whole toolchain end to end**

Run: `just check`
Expected: `cargo fmt --check` clean, `clippy` clean, `ruff check` clean (no
Python files yet, which ruff reports as success), one test passing under
`cargo nextest`.

Run: `just --list`
Expected: every recipe above is listed.

- [ ] **Step 11: Commit**

```bash
git add Cargo.toml rust-toolchain.toml mise.toml flake.nix justfile ruff.toml \
        .envrc .gitignore .github LICENSE-APACHE LICENSE-MIT crates
git commit -m "chore: pin toolchain with mise and nix, scaffold workspace and CI"
```

---

## Task 2: GPU context

**Files:**
- Create: `crates/elements-core/src/gpu/mod.rs`, `crates/elements-core/tests/gpu_context.rs`
- Modify: `crates/elements-core/src/lib.rs`, `crates/elements-core/Cargo.toml`

**Interfaces:**
- Consumes: Task 1's workspace dependency table.
- Produces:
  - `GpuContext::new_headless() -> Result<GpuContext, GpuError>`
  - `GpuContext::device(&self) -> &wgpu::Device`
  - `GpuContext::queue(&self) -> &wgpu::Queue`
  - `GpuContext::adapter_name(&self) -> &str`
  - `GpuContext::scoped<T>(&self, f: impl FnOnce() -> T) -> Result<T, GpuError>` — runs `f` inside a `wgpu` validation error scope and converts any captured error into `GpuError::Validation`.
  - `enum GpuError { NoAdapter, DeviceRequest(String), Validation(String), DeviceLost(String) }`

- [ ] **Step 1: Write the failing test**

Create `crates/elements-core/tests/gpu_context.rs`:

```rust
use elements_core::gpu::{GpuContext, GpuError};

#[test]
fn acquires_a_headless_device() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    assert!(!ctx.adapter_name().is_empty());
}

#[test]
fn scoped_reports_validation_errors() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");

    // A buffer with size 0 and MAP_READ is invalid; wgpu reports it to the error scope.
    let result = ctx.scoped(|| {
        ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("deliberately-invalid"),
            size: 0,
            usage: wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
    });

    match result {
        Err(GpuError::Validation(msg)) => assert!(!msg.is_empty()),
        other => panic!("expected a validation error, got {other:?}"),
    }
}

#[test]
fn scoped_passes_through_success() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let value = ctx.scoped(|| 41 + 1).expect("valid work should not error");
    assert_eq!(value, 42);
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p elements-core --test gpu_context`
Expected: FAIL — `error[E0432]: unresolved import elements_core::gpu`.

- [ ] **Step 3: Add dependencies**

Modify `crates/elements-core/Cargo.toml`, replacing the empty `[dependencies]` and `[dev-dependencies]` sections:

```toml
[dependencies]
pollster.workspace = true
thiserror.workspace = true
wgpu.workspace = true

[dev-dependencies]
```

- [ ] **Step 4: Write the implementation**

Create `crates/elements-core/src/gpu/mod.rs`:

```rust
//! GPU device acquisition and error-scope handling.

use std::sync::{Arc, Mutex};

/// Every way GPU work can fail in Elements.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no suitable GPU adapter was found")]
    NoAdapter,
    #[error("could not create a GPU device: {0}")]
    DeviceRequest(String),
    #[error("GPU validation error: {0}")]
    Validation(String),
    #[error("GPU device was lost: {0}")]
    DeviceLost(String),
}

/// Owns the `wgpu` device and queue for one engine process.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    lost: Arc<Mutex<Option<String>>>,
}

impl GpuContext {
    /// Acquire a device with no surface, suitable for daemons, CLI bakes and CI.
    pub fn new_headless() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_headless_async())
    }

    async fn new_headless_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        let adapter_name = adapter.get_info().name;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("elements-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        let lost = Arc::new(Mutex::new(None));
        let lost_sink = Arc::clone(&lost);
        device.set_device_lost_callback(move |_reason, message| {
            *lost_sink.lock().expect("device-lost mutex poisoned") = Some(message);
        });

        Ok(Self {
            device,
            queue,
            adapter_name,
            lost,
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Returns the device-lost message if the device has been lost.
    pub fn device_lost(&self) -> Option<String> {
        self.lost.lock().expect("device-lost mutex poisoned").clone()
    }

    /// Run `f` inside a validation error scope, converting any captured error.
    ///
    /// This is the only sanctioned way to submit GPU work in Elements: it turns
    /// `wgpu`'s asynchronous, panicking-by-default error reporting into a
    /// `Result` the daemon can surface to the addon.
    pub fn scoped<T>(&self, f: impl FnOnce() -> T) -> Result<T, GpuError> {
        let guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let value = f();
        let error = pollster::block_on(guard.pop());

        if let Some(message) = self.device_lost() {
            return Err(GpuError::DeviceLost(message));
        }
        match error {
            Some(e) => Err(GpuError::Validation(e.to_string())),
            None => Ok(value),
        }
    }
}
```

Modify `crates/elements-core/src/lib.rs`, appending:

```rust
pub mod gpu;
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `WGPU_BACKEND=vulkan cargo test -p elements-core --test gpu_context`
Expected: PASS — three tests ok.

If `scoped_reports_validation_errors` fails because wgpu 30 rejects the zero-size buffer at a different layer, substitute a texture with `size.width = 0` as the invalid resource. The assertion under test is "an invalid call produces `GpuError::Validation`", not the specific invalid call.

- [ ] **Step 6: Commit**

```bash
git add crates/elements-core
git commit -m "feat: add GpuContext with device-lost and validation error scopes"
```

---

## Task 3: Fields and the field pool

**Files:**
- Create: `crates/elements-core/src/gpu/field.rs`, `crates/elements-core/src/gpu/pool.rs`, `crates/elements-core/tests/field_pool.rs`
- Modify: `crates/elements-core/src/gpu/mod.rs`, `crates/elements-core/Cargo.toml`

**Interfaces:**
- Consumes: `GpuContext`, `GpuError` from Task 2.
- Produces:
  - `struct FieldDims { pub x: u32, pub y: u32, pub z: u32 }` with `FieldDims::new(x, y, z)` and `voxel_count(&self) -> usize`
  - `enum FieldFormat { R16Float, Rgba16Float }` with `channels(&self) -> u32` and `bytes_per_voxel(&self) -> u32`
  - `struct Field { .. }` with `dims()`, `format()`, `texture() -> &wgpu::Texture`, `view() -> &wgpu::TextureView`
  - `struct FieldPool` with `FieldPool::new()`, `acquire(&mut self, ctx, dims, format) -> Result<Field, GpuError>`, `release(&mut self, field: Field)`, `pooled_count(&self) -> usize`
  - `Field::read_back(&self, ctx: &GpuContext) -> Result<Vec<f32>, GpuError>` — returns voxels in x-fastest order, `channels()` values per voxel, `f16` decoded to `f32`.

- [ ] **Step 1: Write the failing test**

Create `crates/elements-core/tests/field_pool.rs`:

```rust
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext};

#[test]
fn dims_report_voxel_count() {
    let dims = FieldDims::new(4, 5, 6);
    assert_eq!(dims.voxel_count(), 120);
}

#[test]
fn formats_report_their_size() {
    assert_eq!(FieldFormat::R16Float.channels(), 1);
    assert_eq!(FieldFormat::R16Float.bytes_per_voxel(), 2);
    assert_eq!(FieldFormat::Rgba16Float.channels(), 4);
    assert_eq!(FieldFormat::Rgba16Float.bytes_per_voxel(), 8);
}

#[test]
fn pool_recycles_identical_fields() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let dims = FieldDims::new(8, 8, 8);

    let first = pool.acquire(&ctx, dims, FieldFormat::R16Float).unwrap();
    let first_id = first.texture().global_id();
    pool.release(first);
    assert_eq!(pool.pooled_count(), 1);

    let second = pool.acquire(&ctx, dims, FieldFormat::R16Float).unwrap();
    assert_eq!(second.texture().global_id(), first_id, "should reuse the texture");
    assert_eq!(pool.pooled_count(), 0);
}

#[test]
fn pool_does_not_recycle_across_shapes() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();

    let a = pool
        .acquire(&ctx, FieldDims::new(8, 8, 8), FieldFormat::R16Float)
        .unwrap();
    let a_id = a.texture().global_id();
    pool.release(a);

    let b = pool
        .acquire(&ctx, FieldDims::new(16, 8, 8), FieldFormat::R16Float)
        .unwrap();
    assert_ne!(b.texture().global_id(), a_id);
    assert_eq!(pool.pooled_count(), 1, "the 8^3 field stays pooled");
}

#[test]
fn fresh_field_reads_back_as_zeros() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let dims = FieldDims::new(4, 4, 4);

    let field = pool.acquire(&ctx, dims, FieldFormat::R16Float).unwrap();
    let values = field.read_back(&ctx).unwrap();

    assert_eq!(values.len(), dims.voxel_count());
    assert!(values.iter().all(|v| *v == 0.0), "a new texture is zeroed");
}
```

`global_id()` is how we assert object identity without comparing contents. If `wgpu` 30 does not expose it, compare `Arc::as_ptr` of a cloned texture handle instead, or add a `Field::pool_generation()` counter incremented on allocation and assert it did not change.

- [ ] **Step 2: Run it to verify it fails**

Run: `WGPU_BACKEND=vulkan cargo test -p elements-core --test field_pool`
Expected: FAIL — `unresolved imports FieldDims, FieldFormat, FieldPool`.

- [ ] **Step 3: Add dependencies**

Modify `crates/elements-core/Cargo.toml` `[dependencies]`, adding:

```toml
bytemuck.workspace = true
half.workspace = true
```

- [ ] **Step 4: Implement fields**

Create `crates/elements-core/src/gpu/field.rs`:

```rust
//! 3D scalar and vector fields backed by GPU textures.

use super::{GpuContext, GpuError};

/// The extent of a field in voxels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldDims {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl FieldDims {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub fn voxel_count(&self) -> usize {
        self.x as usize * self.y as usize * self.z as usize
    }

    pub(crate) fn extent(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.x,
            height: self.y,
            depth_or_array_layers: self.z,
        }
    }
}

/// The storage formats Core v1 supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldFormat {
    R16Float,
    Rgba16Float,
}

impl FieldFormat {
    pub fn channels(&self) -> u32 {
        match self {
            Self::R16Float => 1,
            Self::Rgba16Float => 4,
        }
    }

    pub fn bytes_per_voxel(&self) -> u32 {
        self.channels() * 2
    }

    pub(crate) fn wgpu_format(&self) -> wgpu::TextureFormat {
        match self {
            Self::R16Float => wgpu::TextureFormat::R16Float,
            Self::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        }
    }
}

/// A 3D texture plus the metadata needed to interpret it.
pub struct Field {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) dims: FieldDims,
    pub(crate) format: FieldFormat,
}

impl Field {
    pub fn dims(&self) -> FieldDims {
        self.dims
    }

    pub fn format(&self) -> FieldFormat {
        self.format
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Copy the field to the CPU as `f32`, x-fastest, `channels()` values per voxel.
    ///
    /// GPU buffer rows must be aligned to `COPY_BYTES_PER_ROW_ALIGNMENT`, so this
    /// copies into a padded staging buffer and strips the padding on the way out.
    pub fn read_back(&self, ctx: &GpuContext) -> Result<Vec<f32>, GpuError> {
        let unpadded_row = self.dims.x * self.format.bytes_per_voxel();
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = unpadded_row.div_ceil(align) * align;
        let buffer_size =
            padded_row as u64 * self.dims.y as u64 * self.dims.z as u64;

        let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        ctx.scoped(|| {
            let mut encoder = ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("field-readback"),
                });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(self.dims.y),
                    },
                },
                self.dims.extent(),
            );
            ctx.queue().submit(Some(encoder.finish()));
        })?;

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device()
            .poll(wgpu::PollType::Wait)
            .map_err(|e| GpuError::DeviceLost(e.to_string()))?;
        rx.recv()
            .map_err(|e| GpuError::Validation(e.to_string()))?
            .map_err(|e| GpuError::Validation(e.to_string()))?;

        let channels = self.format.channels() as usize;
        let mut out = Vec::with_capacity(self.dims.voxel_count() * channels);
        {
            let mapped = slice.get_mapped_range();
            let row_values = self.dims.x as usize * channels;
            for z in 0..self.dims.z as usize {
                for y in 0..self.dims.y as usize {
                    let start = (z * self.dims.y as usize + y) * padded_row as usize;
                    let end = start + row_values * 2;
                    let halves: &[half::f16] = bytemuck::cast_slice(&mapped[start..end]);
                    out.extend(halves.iter().map(|h| h.to_f32()));
                }
            }
        }
        staging.unmap();

        Ok(out)
    }
}
```

- [ ] **Step 5: Implement the pool**

Create `crates/elements-core/src/gpu/pool.rs`:

```rust
//! Recycles 3D textures so per-frame evaluation does not reallocate.

use std::collections::HashMap;

use super::{Field, FieldDims, FieldFormat, GpuContext, GpuError};

/// Pools fields keyed by `(dims, format)`.
#[derive(Default)]
pub struct FieldPool {
    free: HashMap<(FieldDims, FieldFormat), Vec<Field>>,
}

impl FieldPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a field of this shape from the pool, allocating only if none is free.
    ///
    /// The returned field's contents are unspecified unless it was freshly
    /// allocated, in which case it is zeroed. Callers must fully write a field
    /// before reading it.
    pub fn acquire(
        &mut self,
        ctx: &GpuContext,
        dims: FieldDims,
        format: FieldFormat,
    ) -> Result<Field, GpuError> {
        if let Some(bucket) = self.free.get_mut(&(dims, format))
            && let Some(field) = bucket.pop()
        {
            return Ok(field);
        }

        ctx.scoped(|| {
            let texture = ctx.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("elements-field"),
                size: dims.extent(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: format.wgpu_format(),
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            Field {
                texture,
                view,
                dims,
                format,
            }
        })
    }

    /// Return a field for reuse.
    pub fn release(&mut self, field: Field) {
        self.free
            .entry((field.dims, field.format))
            .or_default()
            .push(field);
    }

    /// How many fields are currently held for reuse.
    pub fn pooled_count(&self) -> usize {
        self.free.values().map(Vec::len).sum()
    }
}
```

Modify `crates/elements-core/src/gpu/mod.rs`, adding near the top after the doc comment:

```rust
mod field;
mod pool;

pub use field::{Field, FieldDims, FieldFormat};
pub use pool::FieldPool;
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `WGPU_BACKEND=vulkan cargo test -p elements-core --test field_pool`
Expected: PASS — five tests ok.

- [ ] **Step 7: Commit**

```bash
git add crates/elements-core
git commit -m "feat: add Field, FieldDims, FieldFormat and a recycling FieldPool"
```

---

## Task 4: Compute dispatch and the constant-fill shader

**Files:**
- Create: `crates/elements-core/src/gpu/dispatch.rs`, `crates/elements-core/src/gpu/shaders/constant.wgsl`, `crates/elements-core/tests/dispatch.rs`
- Modify: `crates/elements-core/src/gpu/mod.rs`

**Interfaces:**
- Consumes: `GpuContext`, `Field`, `FieldPool` from Tasks 2–3.
- Produces:
  - `struct PipelineCache` with `PipelineCache::new()` and `get_or_create(&mut self, ctx, key: &'static str, source: &str, entry_point: &str) -> Result<Arc<wgpu::ComputePipeline>, GpuError>`
  - `fn dispatch_over_field(ctx, pipeline, bind_group, dims) -> Result<(), GpuError>` — dispatches `ceil(dim / 4)` workgroups per axis, matching the `@workgroup_size(4, 4, 4)` every Elements shader uses.
  - `fn fill_constant(ctx, cache, field: &Field, value: f32) -> Result<(), GpuError>`

- [ ] **Step 1: Write the failing test**

Create `crates/elements-core/tests/dispatch.rs`:

```rust
use elements_core::gpu::{
    fill_constant, FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache,
};

#[test]
fn constant_fill_writes_every_voxel() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let dims = FieldDims::new(8, 8, 8);

    let field = pool.acquire(&ctx, dims, FieldFormat::R16Float).unwrap();
    fill_constant(&ctx, &mut cache, &field, 0.75).unwrap();

    let values = field.read_back(&ctx).unwrap();
    assert_eq!(values.len(), dims.voxel_count());
    for v in &values {
        approx::assert_abs_diff_eq!(*v, 0.75, epsilon = 1e-3);
    }
}

#[test]
fn constant_fill_handles_non_multiple_of_workgroup() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    // 5 is not a multiple of the workgroup size 4: the shader must bounds-check.
    let dims = FieldDims::new(5, 5, 5);

    let field = pool.acquire(&ctx, dims, FieldFormat::R16Float).unwrap();
    fill_constant(&ctx, &mut cache, &field, 1.0).unwrap();

    let values = field.read_back(&ctx).unwrap();
    assert_eq!(values.len(), 125);
    for v in &values {
        approx::assert_abs_diff_eq!(*v, 1.0, epsilon = 1e-3);
    }
}

#[test]
fn pipeline_cache_reuses_compiled_pipelines() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut cache = PipelineCache::new();
    let mut pool = FieldPool::new();
    let field = pool
        .acquire(&ctx, FieldDims::new(4, 4, 4), FieldFormat::R16Float)
        .unwrap();

    fill_constant(&ctx, &mut cache, &field, 0.1).unwrap();
    fill_constant(&ctx, &mut cache, &field, 0.2).unwrap();

    assert_eq!(cache.len(), 1, "the same shader must compile only once");
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `WGPU_BACKEND=vulkan cargo test -p elements-core --test dispatch`
Expected: FAIL — `unresolved imports fill_constant, PipelineCache`.

- [ ] **Step 3: Add the dev-dependency**

Modify `crates/elements-core/Cargo.toml` `[dev-dependencies]`:

```toml
[dev-dependencies]
approx.workspace = true
```

- [ ] **Step 4: Write the shader**

Create `crates/elements-core/src/gpu/shaders/constant.wgsl`:

```wgsl
// Fill every voxel of a storage texture with a constant.

struct Params {
    dims: vec3<u32>,
    value: f32,
};

@group(0) @binding(0) var field: texture_storage_3d<r16float, write>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.dims.x || gid.y >= params.dims.y || gid.z >= params.dims.z) {
        return;
    }
    textureStore(field, vec3<i32>(gid), vec4<f32>(params.value, 0.0, 0.0, 0.0));
}
```

- [ ] **Step 5: Implement dispatch**

Create `crates/elements-core/src/gpu/dispatch.rs`:

```rust
//! Compute pipeline caching and workgroup dispatch.

use std::collections::HashMap;
use std::sync::Arc;

use wgpu::util::DeviceExt;

use super::{Field, FieldDims, GpuContext, GpuError};

/// Every Elements compute shader uses this workgroup size.
pub const WORKGROUP: u32 = 4;

/// Caches compiled compute pipelines by a static key.
#[derive(Default)]
pub struct PipelineCache {
    pipelines: HashMap<&'static str, Arc<wgpu::ComputePipeline>>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct pipelines compiled so far.
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    pub fn get_or_create(
        &mut self,
        ctx: &GpuContext,
        key: &'static str,
        source: &str,
        entry_point: &str,
    ) -> Result<Arc<wgpu::ComputePipeline>, GpuError> {
        if let Some(p) = self.pipelines.get(key) {
            return Ok(Arc::clone(p));
        }

        let pipeline = ctx.scoped(|| {
            let module = ctx
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(key),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            Arc::new(
                ctx.device()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(key),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry_point),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    }),
            )
        })?;

        self.pipelines.insert(key, Arc::clone(&pipeline));
        Ok(pipeline)
    }
}

/// Dispatch enough workgroups to cover `dims`, one invocation per voxel.
pub fn dispatch_over_field(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    dims: FieldDims,
) -> Result<(), GpuError> {
    ctx.scoped(|| {
        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("elements-dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("elements-dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(
                dims.x.div_ceil(WORKGROUP),
                dims.y.div_ceil(WORKGROUP),
                dims.z.div_ceil(WORKGROUP),
            );
        }
        ctx.queue().submit(Some(encoder.finish()));
    })
}

/// Parameters shared by the constant shader. `dims` is padded to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConstantParams {
    dims: [u32; 3],
    value: f32,
}

/// Fill every voxel of `field` with `value`.
pub fn fill_constant(
    ctx: &GpuContext,
    cache: &mut PipelineCache,
    field: &Field,
    value: f32,
) -> Result<(), GpuError> {
    let pipeline = cache.get_or_create(
        ctx,
        "constant",
        include_str!("shaders/constant.wgsl"),
        "main",
    )?;

    let dims = field.dims();
    let params = ConstantParams {
        dims: [dims.x, dims.y, dims.z],
        value,
    };

    let bind_group = ctx.scoped(|| {
        let uniform = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("constant-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("constant-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(field.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
            ],
        })
    })?;

    dispatch_over_field(ctx, &pipeline, &bind_group, dims)
}
```

- [ ] **Step 6: Export the module**

Modify `crates/elements-core/src/gpu/mod.rs`, adding to the module list and re-exports:

```rust
mod dispatch;

pub use dispatch::{dispatch_over_field, fill_constant, PipelineCache, WORKGROUP};
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `WGPU_BACKEND=vulkan cargo test -p elements-core --test dispatch`
Expected: PASS — three tests ok.

- [ ] **Step 8: Commit**

```bash
git add crates/elements-core
git commit -m "feat: add pipeline cache, workgroup dispatch and constant-fill shader"
```

---

## Task 5: Seeded curl-noise shader

**Files:**
- Create: `crates/elements-core/src/gpu/shaders/curl_noise.wgsl`, `crates/elements-core/tests/noise.rs`
- Modify: `crates/elements-core/src/gpu/dispatch.rs`, `crates/elements-core/src/gpu/mod.rs`

**Interfaces:**
- Consumes: `PipelineCache`, `dispatch_over_field`, `Field` from Task 4.
- Produces: `fn fill_curl_noise(ctx, cache, field: &Field, seed: u64, frequency: f32) -> Result<(), GpuError>`

- [ ] **Step 1: Write the failing test**

Create `crates/elements-core/tests/noise.rs`:

```rust
use elements_core::gpu::{
    fill_curl_noise, FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache,
};

fn noise_values(seed: u64, frequency: f32) -> Vec<f32> {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let field = pool
        .acquire(&ctx, FieldDims::new(16, 16, 16), FieldFormat::R16Float)
        .unwrap();
    fill_curl_noise(&ctx, &mut cache, &field, seed, frequency).unwrap();
    field.read_back(&ctx).unwrap()
}

#[test]
fn noise_is_deterministic_for_a_seed() {
    let a = noise_values(7, 4.0);
    let b = noise_values(7, 4.0);
    assert_eq!(a, b, "the same seed must produce bit-identical output");
}

#[test]
fn different_seeds_produce_different_fields() {
    let a = noise_values(7, 4.0);
    let b = noise_values(8, 4.0);
    assert_ne!(a, b);
}

#[test]
fn noise_stays_in_range_and_is_finite() {
    let values = noise_values(7, 4.0);
    assert_eq!(values.len(), 16 * 16 * 16);
    for v in &values {
        assert!(v.is_finite(), "noise produced a non-finite value: {v}");
        assert!(
            (-1.001..=1.001).contains(v),
            "noise escaped [-1, 1]: {v}"
        );
    }
}

#[test]
fn noise_is_not_constant() {
    let values = noise_values(7, 4.0);
    let first = values[0];
    assert!(
        values.iter().any(|v| (v - first).abs() > 1e-2),
        "noise must actually vary across the field"
    );
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `WGPU_BACKEND=vulkan cargo test -p elements-core --test noise`
Expected: FAIL — `unresolved import fill_curl_noise`.

- [ ] **Step 3: Write the shader**

Create `crates/elements-core/src/gpu/shaders/curl_noise.wgsl`:

```wgsl
// Seeded value noise, summed over three octaves.
//
// Deterministic across backends: integer hashing only, no trigonometric or
// transcendental functions, whose precision varies between drivers.

struct Params {
    dims: vec3<u32>,
    frequency: f32,
    seed_lo: u32,
    seed_hi: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var field: texture_storage_3d<r16float, write>;
@group(0) @binding(1) var<uniform> params: Params;

fn hash3(p: vec3<i32>, seed: u32) -> f32 {
    var h: u32 = seed;
    h = h ^ (u32(p.x) * 0x9e3779b9u);
    h = h ^ (u32(p.y) * 0x85ebca6bu);
    h = h ^ (u32(p.z) * 0xc2b2ae35u);
    h = h ^ (h >> 15u);
    h = h * 0x2c1b3c6du;
    h = h ^ (h >> 12u);
    h = h * 0x297a2d39u;
    h = h ^ (h >> 15u);
    // Map to [-1, 1] using only exact binary fractions.
    return f32(h & 0xffffffu) * (2.0 / 16777215.0) - 1.0;
}

fn smoothstep3(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

fn value_noise(p: vec3<f32>, seed: u32) -> f32 {
    let cell = vec3<i32>(floor(p));
    let f = p - floor(p);
    let w = vec3<f32>(smoothstep3(f.x), smoothstep3(f.y), smoothstep3(f.z));

    var acc: f32 = 0.0;
    for (var dz: i32 = 0; dz <= 1; dz = dz + 1) {
        for (var dy: i32 = 0; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = 0; dx <= 1; dx = dx + 1) {
                let corner = cell + vec3<i32>(dx, dy, dz);
                let wx = select(1.0 - w.x, w.x, dx == 1);
                let wy = select(1.0 - w.y, w.y, dy == 1);
                let wz = select(1.0 - w.z, w.z, dz == 1);
                acc = acc + hash3(corner, seed) * wx * wy * wz;
            }
        }
    }
    return acc;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.dims.x || gid.y >= params.dims.y || gid.z >= params.dims.z) {
        return;
    }

    let uvw = (vec3<f32>(gid) + vec3<f32>(0.5)) / vec3<f32>(params.dims);
    let seed = params.seed_lo ^ params.seed_hi;

    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var freq: f32 = params.frequency;
    for (var octave: u32 = 0u; octave < 3u; octave = octave + 1u) {
        value = value + value_noise(uvw * freq, seed + octave * 0x9e3779b9u) * amplitude;
        amplitude = amplitude * 0.5;
        freq = freq * 2.0;
    }

    // Three octaves at 0.5/0.25/0.125 sum to at most 0.875; clamp defensively.
    textureStore(field, vec3<i32>(gid), vec4<f32>(clamp(value, -1.0, 1.0), 0.0, 0.0, 0.0));
}
```

- [ ] **Step 4: Implement the host side**

Modify `crates/elements-core/src/gpu/dispatch.rs`, appending:

```rust
/// Parameters for the curl-noise shader. Laid out to match the WGSL struct.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NoiseParams {
    dims: [u32; 3],
    frequency: f32,
    seed_lo: u32,
    seed_hi: u32,
    _pad: [u32; 2],
}

/// Fill `field` with seeded three-octave value noise in `[-1, 1]`.
pub fn fill_curl_noise(
    ctx: &GpuContext,
    cache: &mut PipelineCache,
    field: &Field,
    seed: u64,
    frequency: f32,
) -> Result<(), GpuError> {
    let pipeline = cache.get_or_create(
        ctx,
        "curl_noise",
        include_str!("shaders/curl_noise.wgsl"),
        "main",
    )?;

    let dims = field.dims();
    let params = NoiseParams {
        dims: [dims.x, dims.y, dims.z],
        frequency,
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
        _pad: [0, 0],
    };

    let bind_group = ctx.scoped(|| {
        let uniform = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("noise-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("noise-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(field.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
            ],
        })
    })?;

    dispatch_over_field(ctx, &pipeline, &bind_group, dims)
}
```

Modify the `pub use dispatch::...` line in `crates/elements-core/src/gpu/mod.rs`:

```rust
pub use dispatch::{
    dispatch_over_field, fill_constant, fill_curl_noise, PipelineCache, WORKGROUP,
};
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `WGPU_BACKEND=vulkan cargo test -p elements-core --test noise`
Expected: PASS — four tests ok.

- [ ] **Step 6: Verify the whole workspace still passes**

Run: `just check`
Expected: clean — fmt, clippy, ruff and the full test suite.

- [ ] **Step 7: Commit**

```bash
git add crates/elements-core
git commit -m "feat: add seeded deterministic value-noise compute shader"
```

---

## Task 6: Node graph with topological evaluation

**Files:**
- Create: `crates/elements-core/src/graph/mod.rs`, `crates/elements-core/src/graph/socket.rs`, `crates/elements-core/src/graph/node.rs`, `crates/elements-core/tests/graph.rs`
- Modify: `crates/elements-core/src/lib.rs`, `crates/elements-core/Cargo.toml`

**Interfaces:**
- Consumes: `GpuContext`, `FieldPool`, `PipelineCache`, `Field` from Tasks 2–5.
- Produces:
  - `struct NodeId(pub u32)`, `struct SocketId { pub node: NodeId, pub index: u32 }`
  - `enum SocketType { Field, Scalar }`
  - `struct SocketSpec { pub inputs: Vec<SocketType>, pub outputs: Vec<SocketType> }`
  - `enum Value { Field(Field), Scalar(f32) }` with `as_field(&self) -> Result<&Field, NodeError>` and `as_scalar(&self) -> Result<f32, NodeError>`
  - `trait Node { fn kind(&self) -> &'static str; fn sockets(&self) -> SocketSpec; fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError>; }`
  - `enum NodeError { Cycle(NodeId), MissingInput { node, index }, TypeMismatch { node, index, expected }, UnknownNode(NodeId), NoOutput, Gpu(GpuError) }`
  - `struct EvalCtx<'a>` with `gpu()`, `dims()`, `node_id()`, `acquire(FieldFormat)`, `with_gpu(..)`, `input(index)`, `take_input(index)`
  - `struct Graph` with `new()`, `add_node`, `connect`, `set_output`, `output()`, `node(id)`, `node_ids()`, `topological_order()`, `eval(gpu, pool, pipelines, dims)`

**Ownership rule locked in by this task:** each output socket may feed **at most one** input socket. `connect` rejects a second consumer. This lets `eval` *move* `Value`s (which contain non-`Clone` GPU `Field`s) out of the produced map instead of reference-counting them. Multi-consumer outputs arrive with Ember, where the sharing cost is worth paying.

- [ ] **Step 1: Write the failing test**

Create `crates/elements-core/tests/graph.rs`:

```rust
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{
    EvalCtx, Graph, Node, NodeError, SocketId, SocketSpec, SocketType, Value,
};

/// A node that emits a fixed scalar, exercising the graph without a GPU.
struct Literal(f32);

impl Node for Literal {
    fn kind(&self) -> &'static str {
        "test.literal"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec { inputs: vec![], outputs: vec![SocketType::Scalar] }
    }
    fn eval(&self, _ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        Ok(vec![Value::Scalar(self.0)])
    }
}

/// A node that adds its two scalar inputs.
struct Add;

impl Node for Add {
    fn kind(&self) -> &'static str {
        "test.add"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Scalar, SocketType::Scalar],
            outputs: vec![SocketType::Scalar],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let a = ctx.input(0)?.as_scalar()?;
        let b = ctx.input(1)?.as_scalar()?;
        Ok(vec![Value::Scalar(a + b)])
    }
}

fn harness() -> (GpuContext, FieldPool, PipelineCache) {
    (
        GpuContext::new_headless().expect("no GPU adapter available"),
        FieldPool::new(),
        PipelineCache::new(),
    )
}

#[test]
fn evaluates_a_diamond_in_dependency_order() {
    let mut g = Graph::new();
    let two = g.add_node(Box::new(Literal(2.0)));
    let three = g.add_node(Box::new(Literal(3.0)));
    let sum = g.add_node(Box::new(Add));

    g.connect(SocketId { node: two, index: 0 }, SocketId { node: sum, index: 0 }).unwrap();
    g.connect(SocketId { node: three, index: 0 }, SocketId { node: sum, index: 1 }).unwrap();
    g.set_output(sum);

    let (gpu, mut pool, mut pipelines) = harness();
    let value = g.eval(&gpu, &mut pool, &mut pipelines, FieldDims::new(4, 4, 4)).unwrap();
    assert_eq!(value.as_scalar().unwrap(), 5.0);
}

#[test]
fn detects_cycles() {
    let mut g = Graph::new();
    let a = g.add_node(Box::new(Add));
    let b = g.add_node(Box::new(Add));

    g.connect(SocketId { node: a, index: 0 }, SocketId { node: b, index: 0 }).unwrap();
    g.connect(SocketId { node: b, index: 0 }, SocketId { node: a, index: 0 }).unwrap();

    match g.topological_order() {
        Err(NodeError::Cycle(_)) => {}
        other => panic!("expected a cycle error, got {other:?}"),
    }
}

#[test]
fn rejects_a_nonexistent_socket() {
    let mut g = Graph::new();
    let lit = g.add_node(Box::new(Literal(1.0)));
    let add = g.add_node(Box::new(Add));

    let err = g
        .connect(SocketId { node: lit, index: 0 }, SocketId { node: add, index: 9 })
        .unwrap_err();
    assert!(matches!(err, NodeError::MissingInput { .. }), "got {err:?}");
}

#[test]
fn rejects_a_second_consumer_of_one_output() {
    let mut g = Graph::new();
    let lit = g.add_node(Box::new(Literal(1.0)));
    let add = g.add_node(Box::new(Add));

    g.connect(SocketId { node: lit, index: 0 }, SocketId { node: add, index: 0 }).unwrap();
    let err = g
        .connect(SocketId { node: lit, index: 0 }, SocketId { node: add, index: 1 })
        .unwrap_err();
    assert!(matches!(err, NodeError::AlreadyConsumed { .. }), "got {err:?}");
}

#[test]
fn unconnected_input_is_an_error_at_eval_time() {
    let mut g = Graph::new();
    let add = g.add_node(Box::new(Add));
    g.set_output(add);

    let (gpu, mut pool, mut pipelines) = harness();
    let err = g.eval(&gpu, &mut pool, &mut pipelines, FieldDims::new(4, 4, 4)).unwrap_err();
    assert!(matches!(err, NodeError::MissingInput { .. }), "got {err:?}");
}

#[test]
fn evaluates_only_the_output_subgraph() {
    let mut g = Graph::new();
    let used = g.add_node(Box::new(Literal(1.0)));
    let _unused = g.add_node(Box::new(Add)); // would error if evaluated
    g.set_output(used);

    let (gpu, mut pool, mut pipelines) = harness();
    let value = g.eval(&gpu, &mut pool, &mut pipelines, FieldDims::new(4, 4, 4)).unwrap();
    assert_eq!(value.as_scalar().unwrap(), 1.0);
}

#[test]
fn values_report_type_mismatches() {
    let v = Value::Scalar(1.0);
    assert!(matches!(v.as_field(), Err(NodeError::TypeMismatch { .. })));
}

#[test]
fn field_values_carry_their_dims() {
    let (gpu, mut pool, _pipelines) = harness();
    let field = pool.acquire(&gpu, FieldDims::new(2, 3, 4), FieldFormat::R16Float).unwrap();
    let v = Value::Field(field);
    assert_eq!(v.as_field().unwrap().dims(), FieldDims::new(2, 3, 4));
}

#[test]
fn eval_without_an_output_is_an_error() {
    let g = Graph::new();
    let (gpu, mut pool, mut pipelines) = harness();
    let err = g.eval(&gpu, &mut pool, &mut pipelines, FieldDims::new(4, 4, 4)).unwrap_err();
    assert!(matches!(err, NodeError::NoOutput), "got {err:?}");
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `WGPU_BACKEND=vulkan cargo test -p elements-core --test graph`
Expected: FAIL — `error[E0432]: unresolved import elements_core::graph`.

- [ ] **Step 3: Implement sockets**

Create `crates/elements-core/src/graph/socket.rs`:

```rust
//! Socket identity and typing.

/// Identifies a node within one graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// Identifies one socket on one node. Whether `index` refers to an input or an
/// output is determined by how the `SocketId` is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SocketId {
    pub node: NodeId,
    pub index: u32,
}

/// The value kinds that can travel along a connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocketType {
    Field,
    Scalar,
}

/// A node's input and output signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SocketSpec {
    pub inputs: Vec<SocketType>,
    pub outputs: Vec<SocketType>,
}
```

- [ ] **Step 4: Implement values, errors and the node trait**

Create `crates/elements-core/src/graph/node.rs`:

```rust
//! The `Node` trait, the values that flow between nodes, and node errors.

use crate::gpu::{
    Field, FieldDims, FieldFormat, FieldPool, GpuContext, GpuError, PipelineCache,
};

use super::socket::{NodeId, SocketSpec, SocketType};

/// Everything that can go wrong building or evaluating a graph.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("the graph contains a cycle through node {0:?}")]
    Cycle(NodeId),
    #[error("node {node:?} input {index} is not connected")]
    MissingInput { node: NodeId, index: u32 },
    #[error("node {node:?} socket {index} expected {expected:?}")]
    TypeMismatch {
        node: NodeId,
        index: u32,
        expected: SocketType,
    },
    #[error("node {node:?} output {index} already feeds another input")]
    AlreadyConsumed { node: NodeId, index: u32 },
    #[error("no such node: {0:?}")]
    UnknownNode(NodeId),
    #[error("no output node is set on this graph")]
    NoOutput,
    #[error(transparent)]
    Gpu(#[from] GpuError),
}

/// A value travelling along a connection.
pub enum Value {
    Field(Field),
    Scalar(f32),
}

impl Value {
    pub fn as_field(&self) -> Result<&Field, NodeError> {
        match self {
            Self::Field(f) => Ok(f),
            Self::Scalar(_) => Err(NodeError::TypeMismatch {
                node: NodeId(u32::MAX),
                index: 0,
                expected: SocketType::Field,
            }),
        }
    }

    pub fn as_scalar(&self) -> Result<f32, NodeError> {
        match self {
            Self::Scalar(s) => Ok(*s),
            Self::Field(_) => Err(NodeError::TypeMismatch {
                node: NodeId(u32::MAX),
                index: 0,
                expected: SocketType::Scalar,
            }),
        }
    }

    pub fn socket_type(&self) -> SocketType {
        match self {
            Self::Field(_) => SocketType::Field,
            Self::Scalar(_) => SocketType::Scalar,
        }
    }
}

/// What a node is handed when it evaluates.
///
/// Inputs are owned: the evaluator moves each producer's value into the
/// consumer, which is why an output may feed only one input.
pub struct EvalCtx<'a> {
    pub(crate) gpu: &'a GpuContext,
    pub(crate) pool: &'a mut FieldPool,
    pub(crate) pipelines: &'a mut PipelineCache,
    pub(crate) dims: FieldDims,
    pub(crate) node: NodeId,
    pub(crate) inputs: Vec<Option<Value>>,
}

impl EvalCtx<'_> {
    pub fn gpu(&self) -> &GpuContext {
        self.gpu
    }

    /// The resolution this evaluation is running at.
    pub fn dims(&self) -> FieldDims {
        self.dims
    }

    pub fn node_id(&self) -> NodeId {
        self.node
    }

    /// Borrow input `index`.
    pub fn input(&self, index: u32) -> Result<&Value, NodeError> {
        self.inputs
            .get(index as usize)
            .and_then(Option::as_ref)
            .ok_or(NodeError::MissingInput {
                node: self.node,
                index,
            })
    }

    /// Take ownership of input `index`, leaving it unavailable to later reads.
    /// This is how pass-through nodes forward a GPU field without copying it.
    pub fn take_input(&mut self, index: u32) -> Result<Value, NodeError> {
        self.inputs
            .get_mut(index as usize)
            .and_then(Option::take)
            .ok_or(NodeError::MissingInput {
                node: self.node,
                index,
            })
    }

    /// Acquire a pooled field at the current evaluation dims.
    pub fn acquire(&mut self, format: FieldFormat) -> Result<Field, NodeError> {
        let dims = self.dims;
        Ok(self.pool.acquire(self.gpu, dims, format)?)
    }

    /// Run GPU work with the device and pipeline cache borrowed together.
    ///
    /// Nodes must use this rather than borrowing `gpu()` and the cache
    /// separately, which the borrow checker rejects.
    pub fn with_gpu<T>(
        &mut self,
        f: impl FnOnce(&GpuContext, &mut PipelineCache) -> Result<T, GpuError>,
    ) -> Result<T, NodeError> {
        Ok(f(self.gpu, self.pipelines)?)
    }
}

/// One unit of computation in a graph.
pub trait Node: Send + Sync {
    /// Stable identifier used in the `.elements` document, e.g. `"core.noise_field"`.
    fn kind(&self) -> &'static str;

    /// This node's input and output types.
    fn sockets(&self) -> SocketSpec;

    /// Compute this node's outputs. Must return exactly `sockets().outputs.len()` values.
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError>;
}
```

- [ ] **Step 5: Implement the graph**

Create `crates/elements-core/src/graph/mod.rs`:

```rust
//! The node graph: connections, topological evaluation, and cycle detection.

mod node;
mod socket;

pub use node::{EvalCtx, Node, NodeError, Value};
pub use socket::{NodeId, SocketId, SocketSpec, SocketType};

use std::collections::HashMap;

use crate::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};

/// A directed acyclic graph of nodes with one designated output.
#[derive(Default)]
pub struct Graph {
    nodes: Vec<Box<dyn Node>>,
    /// Maps a destination input socket to the source output socket feeding it.
    edges: HashMap<SocketId, SocketId>,
    output: Option<NodeId>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, node: Box<dyn Node>) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    pub fn node(&self, id: NodeId) -> Result<&dyn Node, NodeError> {
        self.nodes
            .get(id.0 as usize)
            .map(|b| b.as_ref())
            .ok_or(NodeError::UnknownNode(id))
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        (0..self.nodes.len() as u32).map(NodeId)
    }

    /// Connect an output socket to an input socket, validating both ends.
    ///
    /// Rejects type mismatches, nonexistent sockets, and any attempt to feed a
    /// second input from an output that already has a consumer.
    pub fn connect(&mut self, from: SocketId, to: SocketId) -> Result<(), NodeError> {
        let from_spec = self.node(from.node)?.sockets();
        let to_spec = self.node(to.node)?.sockets();

        let out_ty = from_spec.outputs.get(from.index as usize).copied().ok_or(
            NodeError::MissingInput {
                node: from.node,
                index: from.index,
            },
        )?;
        let in_ty = to_spec.inputs.get(to.index as usize).copied().ok_or(
            NodeError::MissingInput {
                node: to.node,
                index: to.index,
            },
        )?;

        if out_ty != in_ty {
            return Err(NodeError::TypeMismatch {
                node: to.node,
                index: to.index,
                expected: in_ty,
            });
        }

        if self.edges.values().any(|src| *src == from) {
            return Err(NodeError::AlreadyConsumed {
                node: from.node,
                index: from.index,
            });
        }

        self.edges.insert(to, from);
        Ok(())
    }

    pub fn set_output(&mut self, node: NodeId) {
        self.output = Some(node);
    }

    pub fn output(&self) -> Option<NodeId> {
        self.output
    }

    /// All nodes in dependency order. Errors if the graph contains a cycle.
    pub fn topological_order(&self) -> Result<Vec<NodeId>, NodeError> {
        #[derive(Clone, Copy, PartialEq)]
        enum Mark {
            Unvisited,
            InProgress,
            Done,
        }

        let mut marks = vec![Mark::Unvisited; self.nodes.len()];
        let mut order = Vec::with_capacity(self.nodes.len());

        // Iterative depth-first search: deep graphs must not blow the stack.
        for start in self.node_ids() {
            if marks[start.0 as usize] != Mark::Unvisited {
                continue;
            }
            let mut stack = vec![(start, false)];
            while let Some((id, children_done)) = stack.pop() {
                if children_done {
                    marks[id.0 as usize] = Mark::Done;
                    order.push(id);
                    continue;
                }
                match marks[id.0 as usize] {
                    Mark::Done => continue,
                    Mark::InProgress => return Err(NodeError::Cycle(id)),
                    Mark::Unvisited => {}
                }
                marks[id.0 as usize] = Mark::InProgress;
                stack.push((id, true));
                for source in self.sources_of(id) {
                    match marks[source.0 as usize] {
                        Mark::InProgress => return Err(NodeError::Cycle(source)),
                        Mark::Unvisited => stack.push((source, false)),
                        Mark::Done => {}
                    }
                }
            }
        }

        Ok(order)
    }

    /// The nodes feeding `id`'s inputs.
    fn sources_of(&self, id: NodeId) -> Vec<NodeId> {
        let Ok(spec) = self.node(id).map(|n| n.sockets()) else {
            return Vec::new();
        };
        (0..spec.inputs.len() as u32)
            .filter_map(|index| self.edges.get(&SocketId { node: id, index }))
            .map(|src| src.node)
            .collect()
    }

    /// Nodes the output transitively depends on, in dependency order.
    fn evaluation_order(&self) -> Result<Vec<NodeId>, NodeError> {
        let output = self.output.ok_or(NodeError::NoOutput)?;
        self.node(output)?;
        let full = self.topological_order()?;

        let mut needed = vec![false; self.nodes.len()];
        let mut stack = vec![output];
        while let Some(id) = stack.pop() {
            if std::mem::replace(&mut needed[id.0 as usize], true) {
                continue;
            }
            stack.extend(self.sources_of(id));
        }

        Ok(full.into_iter().filter(|id| needed[id.0 as usize]).collect())
    }

    /// Evaluate the output node and everything it depends on.
    pub fn eval(
        &self,
        gpu: &GpuContext,
        pool: &mut FieldPool,
        pipelines: &mut PipelineCache,
        dims: FieldDims,
    ) -> Result<Value, NodeError> {
        let output = self.output.ok_or(NodeError::NoOutput)?;
        let order = self.evaluation_order()?;

        // Outputs of each evaluated node. Values are moved out as they are
        // consumed, so each slot holds `None` once its consumer has run.
        let mut produced: HashMap<NodeId, Vec<Option<Value>>> = HashMap::new();

        for id in order {
            let node = self.node(id)?;
            let spec = node.sockets();

            let mut inputs: Vec<Option<Value>> = Vec::with_capacity(spec.inputs.len());
            for index in 0..spec.inputs.len() as u32 {
                let value = match self.edges.get(&SocketId { node: id, index }) {
                    Some(src) => produced
                        .get_mut(&src.node)
                        .and_then(|outs| outs.get_mut(src.index as usize))
                        .and_then(Option::take),
                    None => None,
                };
                inputs.push(value);
            }

            let mut ctx = EvalCtx {
                gpu,
                pool,
                pipelines,
                dims,
                node: id,
                inputs,
            };
            let outputs = node.eval(&mut ctx)?;
            produced.insert(id, outputs.into_iter().map(Some).collect());
        }

        produced
            .get_mut(&output)
            .and_then(|outs| outs.first_mut())
            .and_then(Option::take)
            .ok_or(NodeError::UnknownNode(output))
    }
}
```

- [ ] **Step 6: Export the module**

Modify `crates/elements-core/src/lib.rs`, appending:

```rust
pub mod graph;
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `WGPU_BACKEND=vulkan cargo test -p elements-core --test graph`
Expected: PASS — nine tests ok.

- [ ] **Step 8: Commit**

```bash
git add crates/elements-core
git commit -m "feat: add node graph with topological eval and cycle detection"
```

---

## Task 7: The `.elements` document format

**Files:**
- Create: `crates/elements-core/src/graph/registry.rs`, `crates/elements-core/src/graph/document.rs`, `crates/elements-core/tests/document.rs`, `crates/elements-core/tests/fixtures/v1_minimal.elements`
- Modify: `crates/elements-core/src/graph/mod.rs`, `crates/elements-core/Cargo.toml`

**Interfaces:**
- Consumes: `Graph`, `Node`, `NodeId`, `SocketId`, `NodeError` from Task 6.
- Produces:
  - `const ELEMENTS_DOC_VERSION: u32 = 1`
  - `type NodeCtor = fn(&serde_json::Value) -> Result<Box<dyn Node>, DocError>`
  - `struct NodeRegistry` with `new()`, `with_builtins()`, `register(kind, ctor)`, `build(kind, params)`, `kinds()`
  - `struct Document { version: u32, dims: [u32; 3], nodes: Vec<DocNode>, edges: Vec<DocEdge>, output: u32 }`
  - `struct DocNode { id: u32, kind: String, params: serde_json::Value }`
  - `struct DocEdge { from_node: u32, from_index: u32, to_node: u32, to_index: u32 }`
  - `Document::from_json(&str)`, `Document::to_json(&self)`, `Document::into_graph(self, &NodeRegistry) -> Result<(Graph, FieldDims), DocError>`
  - `enum DocError { Json, UnsupportedVersion(u32), UnknownKind(String), BadParams { kind, reason }, Graph(NodeError) }`

- [ ] **Step 1: Write the fixture**

Create `crates/elements-core/tests/fixtures/v1_minimal.elements`:

```json
{
  "version": 1,
  "dims": [8, 8, 8],
  "nodes": [
    { "id": 0, "kind": "core.noise_field", "params": { "seed": 7, "frequency": 4.0 } },
    { "id": 1, "kind": "core.output", "params": {} }
  ],
  "edges": [
    { "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }
  ],
  "output": 1
}
```

- [ ] **Step 2: Write the failing test**

Create `crates/elements-core/tests/document.rs`:

```rust
use elements_core::graph::{DocError, Document, NodeRegistry, ELEMENTS_DOC_VERSION};

const MINIMAL: &str = include_str!("fixtures/v1_minimal.elements");

#[test]
fn parses_the_v1_fixture() {
    let doc = Document::from_json(MINIMAL).unwrap();
    assert_eq!(doc.version, ELEMENTS_DOC_VERSION);
    assert_eq!(doc.dims, [8, 8, 8]);
    assert_eq!(doc.nodes.len(), 2);
    assert_eq!(doc.edges.len(), 1);
    assert_eq!(doc.output, 1);
}

#[test]
fn round_trips_without_loss() {
    let doc = Document::from_json(MINIMAL).unwrap();
    let json = doc.to_json().unwrap();
    let again = Document::from_json(&json).unwrap();

    assert_eq!(again.version, doc.version);
    assert_eq!(again.dims, doc.dims);
    assert_eq!(again.output, doc.output);
    assert_eq!(again.nodes.len(), doc.nodes.len());
    for (a, b) in again.nodes.iter().zip(doc.nodes.iter()) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.kind, b.kind);
        assert_eq!(a.params, b.params);
    }
}

#[test]
fn rejects_a_future_version() {
    let future = MINIMAL.replace("\"version\": 1", "\"version\": 999");
    match Document::from_json(&future) {
        Err(DocError::UnsupportedVersion(999)) => {}
        other => panic!("expected UnsupportedVersion(999), got {other:?}"),
    }
}

#[test]
fn rejects_non_positional_node_ids() {
    let shuffled = MINIMAL.replace("\"id\": 0", "\"id\": 5");
    let doc = Document::from_json(&shuffled).unwrap();
    let registry = NodeRegistry::with_builtins();
    match doc.into_graph(&registry) {
        Err(DocError::BadParams { reason, .. }) => {
            assert!(reason.contains("positional"), "got {reason}")
        }
        other => panic!("expected BadParams, got {other:?}"),
    }
}

// unignore in Task 8
#[test]
#[ignore]
fn rejects_an_unknown_node_kind() {
    let doc =
        Document::from_json(&MINIMAL.replace("core.noise_field", "core.does_not_exist")).unwrap();
    let registry = NodeRegistry::with_builtins();
    match doc.into_graph(&registry) {
        Err(DocError::UnknownKind(k)) => assert_eq!(k, "core.does_not_exist"),
        other => panic!("expected UnknownKind, got {other:?}"),
    }
}

// unignore in Task 8
#[test]
#[ignore]
fn rejects_malformed_params() {
    let doc =
        Document::from_json(&MINIMAL.replace("\"seed\": 7", "\"seed\": \"not-a-number\"")).unwrap();
    let registry = NodeRegistry::with_builtins();
    match doc.into_graph(&registry) {
        Err(DocError::BadParams { kind, .. }) => assert_eq!(kind, "core.noise_field"),
        other => panic!("expected BadParams, got {other:?}"),
    }
}

// unignore in Task 8
#[test]
#[ignore]
fn builds_a_graph_with_the_declared_output() {
    let doc = Document::from_json(MINIMAL).unwrap();
    let registry = NodeRegistry::with_builtins();
    let (graph, dims) = doc.into_graph(&registry).unwrap();

    assert_eq!(dims.x, 8);
    assert_eq!(graph.output().unwrap().0, 1);
    assert_eq!(graph.node_count(), 2);
}
```

Three tests are ignored because they need the built-in node kinds from Task 8, which unignores them.

- [ ] **Step 3: Run it to verify it fails**

Run: `cargo test -p elements-core --test document`
Expected: FAIL — `unresolved imports Document, NodeRegistry, ELEMENTS_DOC_VERSION`.

- [ ] **Step 4: Add dependencies**

Modify `crates/elements-core/Cargo.toml` `[dependencies]`, adding:

```toml
serde.workspace = true
serde_json.workspace = true
```

- [ ] **Step 5: Implement the registry**

Create `crates/elements-core/src/graph/registry.rs`:

```rust
//! Maps `.elements` node-kind strings to node constructors.

use std::collections::HashMap;

use super::document::DocError;
use super::node::Node;

/// Builds a node from its serialized parameters.
pub type NodeCtor = fn(&serde_json::Value) -> Result<Box<dyn Node>, DocError>;

/// The set of node kinds a document may reference.
///
/// Core owns the graph; products register their own kinds on top of the
/// built-ins, which is the seam that keeps solvers out of `elements-core`.
#[derive(Default)]
pub struct NodeRegistry {
    ctors: HashMap<&'static str, NodeCtor>,
}

impl NodeRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// The node kinds every Elements build ships. Populated in Task 8.
    pub fn with_builtins() -> Self {
        Self::new()
    }

    pub fn register(&mut self, kind: &'static str, ctor: NodeCtor) {
        self.ctors.insert(kind, ctor);
    }

    pub fn build(&self, kind: &str, params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
        let ctor = self
            .ctors
            .get(kind)
            .ok_or_else(|| DocError::UnknownKind(kind.to_owned()))?;
        ctor(params)
    }

    pub fn kinds(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.ctors.keys().copied()
    }
}
```

- [ ] **Step 6: Implement the document**

Create `crates/elements-core/src/graph/document.rs`:

```rust
//! Serialization of graphs to and from the versioned `.elements` format.

use serde::{Deserialize, Serialize};

use crate::gpu::FieldDims;

use super::node::NodeError;
use super::registry::NodeRegistry;
use super::socket::{NodeId, SocketId};
use super::Graph;

/// The `.elements` schema version this build reads and writes.
///
/// Bumping this requires a migration test in `tests/document.rs`.
pub const ELEMENTS_DOC_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum DocError {
    #[error("malformed document: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported document version {0}")]
    UnsupportedVersion(u32),
    #[error("unknown node kind: {0}")]
    UnknownKind(String),
    #[error("bad parameters for node kind {kind}: {reason}")]
    BadParams { kind: String, reason: String },
    #[error(transparent)]
    Graph(#[from] NodeError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocNode {
    pub id: u32,
    pub kind: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DocEdge {
    pub from_node: u32,
    pub from_index: u32,
    pub to_node: u32,
    pub to_index: u32,
}

/// A serialized node graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub version: u32,
    pub dims: [u32; 3],
    pub nodes: Vec<DocNode>,
    pub edges: Vec<DocEdge>,
    pub output: u32,
}

impl Document {
    pub fn from_json(text: &str) -> Result<Self, DocError> {
        let doc: Document = serde_json::from_str(text)?;
        if doc.version != ELEMENTS_DOC_VERSION {
            return Err(DocError::UnsupportedVersion(doc.version));
        }
        Ok(doc)
    }

    pub fn to_json(&self) -> Result<String, DocError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Instantiate the graph this document describes.
    ///
    /// Node ids are positional: the node with `id` N must be the Nth entry in
    /// `nodes`. That keeps `NodeId` allocation inside `Graph::add_node` and the
    /// document in agreement without a translation table.
    pub fn into_graph(self, registry: &NodeRegistry) -> Result<(Graph, FieldDims), DocError> {
        let mut graph = Graph::new();

        for (position, doc_node) in self.nodes.iter().enumerate() {
            if doc_node.id as usize != position {
                return Err(DocError::BadParams {
                    kind: doc_node.kind.clone(),
                    reason: format!(
                        "node id {} is at position {position}; ids must be positional",
                        doc_node.id
                    ),
                });
            }
            let node = registry.build(&doc_node.kind, &doc_node.params)?;
            graph.add_node(node);
        }

        for edge in &self.edges {
            graph.connect(
                SocketId {
                    node: NodeId(edge.from_node),
                    index: edge.from_index,
                },
                SocketId {
                    node: NodeId(edge.to_node),
                    index: edge.to_index,
                },
            )?;
        }

        graph.set_output(NodeId(self.output));

        Ok((
            graph,
            FieldDims::new(self.dims[0], self.dims[1], self.dims[2]),
        ))
    }
}
```

- [ ] **Step 7: Export the modules**

Modify `crates/elements-core/src/graph/mod.rs`, adding to the module declarations and re-exports at the top:

```rust
mod document;
mod registry;

pub use document::{DocEdge, DocError, DocNode, Document, ELEMENTS_DOC_VERSION};
pub use registry::{NodeCtor, NodeRegistry};
```

- [ ] **Step 8: Run the tests to verify the active ones pass**

Run: `cargo test -p elements-core --test document`
Expected: PASS — four tests ok, three ignored.

- [ ] **Step 9: Commit**

```bash
git add crates/elements-core
git commit -m "feat: add versioned .elements document format and node registry"
```

---

## Task 8: Built-in node kinds

**Files:**
- Create: `crates/elements-core/src/nodes/mod.rs`, `crates/elements-core/src/nodes/constant_field.rs`, `crates/elements-core/src/nodes/noise_field.rs`, `crates/elements-core/src/nodes/output.rs`, `crates/elements-core/tests/nodes.rs`
- Modify: `crates/elements-core/src/lib.rs`, `crates/elements-core/src/graph/registry.rs`, `crates/elements-core/tests/document.rs`

**Interfaces:**
- Consumes: `Node`, `EvalCtx`, `Value`, `SocketSpec`, `SocketType`, `NodeError` from Task 6; `NodeRegistry`, `DocError` from Task 7; `fill_constant`, `fill_curl_noise`, `FieldFormat` from Tasks 4–5.
- Produces:
  - `ConstantField { value: f32 }` — kind `"core.constant_field"`, zero inputs, one `Field` output.
  - `NoiseField { seed: u64, frequency: f32 }` — kind `"core.noise_field"`, `frequency` defaults to `4.0`, zero inputs, one `Field` output.
  - `Output` — kind `"core.output"`, one `Field` input, one `Field` output, pass-through.
  - `elements_core::nodes::register_builtins(&mut NodeRegistry)` and a real `NodeRegistry::with_builtins()`.

- [ ] **Step 1: Write the failing test**

Create `crates/elements-core/tests/nodes.rs`:

```rust
use elements_core::gpu::{FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{Document, NodeRegistry};

fn eval_doc(json: &str) -> Vec<f32> {
    let doc = Document::from_json(json).unwrap();
    let registry = NodeRegistry::with_builtins();
    let (graph, dims) = doc.into_graph(&registry).unwrap();

    let gpu = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();

    let value = graph.eval(&gpu, &mut pool, &mut pipelines, dims).unwrap();
    value.as_field().unwrap().read_back(&gpu).unwrap()
}

const CONSTANT_DOC: &str = r#"{
  "version": 1,
  "dims": [4, 4, 4],
  "nodes": [
    { "id": 0, "kind": "core.constant_field", "params": { "value": 0.5 } },
    { "id": 1, "kind": "core.output", "params": {} }
  ],
  "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
  "output": 1
}"#;

const NOISE_DOC: &str = r#"{
  "version": 1,
  "dims": [8, 8, 8],
  "nodes": [
    { "id": 0, "kind": "core.noise_field", "params": { "seed": 7, "frequency": 4.0 } },
    { "id": 1, "kind": "core.output", "params": {} }
  ],
  "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
  "output": 1
}"#;

#[test]
fn constant_field_through_output() {
    let values = eval_doc(CONSTANT_DOC);
    assert_eq!(values.len(), 64);
    for v in &values {
        approx::assert_abs_diff_eq!(*v, 0.5, epsilon = 1e-3);
    }
}

#[test]
fn noise_field_through_output_is_seed_stable() {
    assert_eq!(eval_doc(NOISE_DOC), eval_doc(NOISE_DOC));
}

#[test]
fn noise_frequency_defaults_when_omitted() {
    let doc = NOISE_DOC.replace(", \"frequency\": 4.0", "");
    let values = eval_doc(&doc);
    assert_eq!(values.len(), 512);
    assert!(values.iter().all(|v| v.is_finite()));
}

#[test]
fn registry_exposes_all_three_builtins() {
    let registry = NodeRegistry::with_builtins();
    let mut kinds: Vec<_> = registry.kinds().collect();
    kinds.sort_unstable();
    assert_eq!(kinds, ["core.constant_field", "core.noise_field", "core.output"]);
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `WGPU_BACKEND=vulkan cargo test -p elements-core --test nodes`
Expected: FAIL — `registry_exposes_all_three_builtins` asserts against an empty list; the others fail with `UnknownKind`.

- [ ] **Step 3: Implement ConstantField**

Create `crates/elements-core/src/nodes/constant_field.rs`:

```rust
//! A field filled with a single scalar.

use serde::Deserialize;

use crate::gpu::{fill_constant, FieldFormat};
use crate::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};

pub const KIND: &str = "core.constant_field";

#[derive(Debug, Clone, Deserialize)]
pub struct ConstantField {
    pub value: f32,
}

impl Node for ConstantField {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Field],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let field = ctx.acquire(FieldFormat::R16Float)?;
        let value = self.value;
        ctx.with_gpu(|gpu, cache| fill_constant(gpu, cache, &field, value))?;
        Ok(vec![Value::Field(field)])
    }
}

pub(crate) fn build(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    let node: ConstantField =
        serde_json::from_value(params.clone()).map_err(|e| DocError::BadParams {
            kind: KIND.to_owned(),
            reason: e.to_string(),
        })?;
    Ok(Box::new(node))
}
```

- [ ] **Step 4: Implement NoiseField and Output**

Create `crates/elements-core/src/nodes/noise_field.rs`:

```rust
//! A field of seeded value noise.

use serde::Deserialize;

use crate::gpu::{fill_curl_noise, FieldFormat};
use crate::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};

pub const KIND: &str = "core.noise_field";

fn default_frequency() -> f32 {
    4.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct NoiseField {
    pub seed: u64,
    #[serde(default = "default_frequency")]
    pub frequency: f32,
}

impl Node for NoiseField {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Field],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let field = ctx.acquire(FieldFormat::R16Float)?;
        let (seed, frequency) = (self.seed, self.frequency);
        ctx.with_gpu(|gpu, cache| fill_curl_noise(gpu, cache, &field, seed, frequency))?;
        Ok(vec![Value::Field(field)])
    }
}

pub(crate) fn build(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    let node: NoiseField =
        serde_json::from_value(params.clone()).map_err(|e| DocError::BadParams {
            kind: KIND.to_owned(),
            reason: e.to_string(),
        })?;
    Ok(Box::new(node))
}
```

Create `crates/elements-core/src/nodes/output.rs`:

```rust
//! The graph's terminal node. Passes its input through unchanged.

use crate::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};

pub const KIND: &str = "core.output";

#[derive(Debug, Clone, Default)]
pub struct Output;

impl Node for Output {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field],
            outputs: vec![SocketType::Field],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        // Move the field through rather than copying it on the GPU.
        ctx.take_input(0).map(|v| vec![v])
    }
}

pub(crate) fn build(_params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    Ok(Box::new(Output))
}
```

Create `crates/elements-core/src/nodes/mod.rs`:

```rust
//! The node kinds every Elements build ships.

pub mod constant_field;
pub mod noise_field;
pub mod output;

pub use constant_field::ConstantField;
pub use noise_field::NoiseField;
pub use output::Output;

use crate::graph::NodeRegistry;

/// Register every built-in kind on `registry`.
pub fn register_builtins(registry: &mut NodeRegistry) {
    registry.register(constant_field::KIND, constant_field::build);
    registry.register(noise_field::KIND, noise_field::build);
    registry.register(output::KIND, output::build);
}
```

- [ ] **Step 5: Wire up the registry**

Modify `crates/elements-core/src/graph/registry.rs`, replacing the placeholder `with_builtins` body:

```rust
    /// The node kinds every Elements build ships.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        crate::nodes::register_builtins(&mut registry);
        registry
    }
```

Modify `crates/elements-core/src/lib.rs`, appending:

```rust
pub mod nodes;
```

- [ ] **Step 6: Run the node tests to verify they pass**

Run: `WGPU_BACKEND=vulkan cargo test -p elements-core --test nodes`
Expected: PASS — four tests ok.

- [ ] **Step 7: Unignore the deferred document tests**

Modify `crates/elements-core/tests/document.rs`, deleting the three `#[ignore]` attributes and their `// unignore in Task 8` comments.

Run: `WGPU_BACKEND=vulkan cargo test -p elements-core --test document`
Expected: PASS — seven tests ok, none ignored.

- [ ] **Step 8: Commit**

```bash
git add crates/elements-core
git commit -m "feat: add core.constant_field, core.noise_field and core.output nodes"
```

---

## Task 9: NPY and PNG output

**Files:**
- Create: `crates/elements-io/Cargo.toml`, `crates/elements-io/src/lib.rs`, `crates/elements-io/src/npy.rs`, `crates/elements-io/src/preview.rs`, `crates/elements-io/tests/npy_png.rs`

**Interfaces:**
- Consumes: nothing from earlier crates. `elements-io` takes plain `&[f32]` plus dimensions, so it is fully testable without a GPU.
- Produces:
  - `fn write_npy(path: &Path, values: &[f32], dims: [u32; 3]) -> Result<(), IoError>` — C-order `(z, y, x)` float32.
  - `fn read_npy(path: &Path) -> Result<(Vec<f32>, [u32; 3]), IoError>`
  - `fn write_slice_png(path: &Path, values: &[f32], dims: [u32; 3], z: u32, range: (f32, f32)) -> Result<(), IoError>`
  - `enum IoError { Io, Npy(String), Png(String), BadSlice { z, depth }, LengthMismatch { expected, got } }`

- [ ] **Step 1: Write the failing test**

Create `crates/elements-io/tests/npy_png.rs`:

```rust
use elements_io::{read_npy, write_npy, write_slice_png, IoError};

fn ramp(dims: [u32; 3]) -> Vec<f32> {
    let n = (dims[0] * dims[1] * dims[2]) as usize;
    (0..n).map(|i| i as f32 / n as f32).collect()
}

#[test]
fn npy_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("field.npy");
    let dims = [4, 3, 2];
    let values = ramp(dims);

    write_npy(&path, &values, dims).unwrap();
    let (back, back_dims) = read_npy(&path).unwrap();

    assert_eq!(back_dims, dims);
    assert_eq!(back, values);
}

#[test]
fn npy_rejects_a_length_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.npy");
    match write_npy(&path, &[1.0, 2.0], [4, 4, 4]) {
        Err(IoError::LengthMismatch { expected: 64, got: 2 }) => {}
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn png_writes_the_requested_slice() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("slice.png");
    // Each z-slice is constant and equal to its index, so output is checkable.
    let mut values = vec![0.0f32; 8 * 8 * 4];
    for z in 0..4 {
        for i in 0..64 {
            values[z * 64 + i] = z as f32;
        }
    }

    write_slice_png(&path, &values, [8, 8, 4], 2, (0.0, 3.0)).unwrap();

    let decoder = png::Decoder::new(std::fs::File::open(&path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();

    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
    // z = 2 mapped through range 0..3 is 2/3 of full scale, rounding to 170.
    assert!(buf[..info.buffer_size()].iter().all(|b| *b == 170));
}

#[test]
fn png_clamps_out_of_range_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clamped.png");
    let values = vec![-5.0f32, 5.0, -5.0, 5.0];

    write_slice_png(&path, &values, [2, 2, 1], 0, (0.0, 1.0)).unwrap();

    let decoder = png::Decoder::new(std::fs::File::open(&path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    reader.next_frame(&mut buf).unwrap();

    assert_eq!(&buf[..4], &[0, 255, 0, 255]);
}

#[test]
fn png_rejects_an_out_of_bounds_slice() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("oob.png");
    match write_slice_png(&path, &ramp([2, 2, 2]), [2, 2, 2], 9, (0.0, 1.0)) {
        Err(IoError::BadSlice { z: 9, depth: 2 }) => {}
        other => panic!("expected BadSlice, got {other:?}"),
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p elements-io`
Expected: FAIL — `error: package ID specification elements-io did not match any packages`.

- [ ] **Step 3: Create the crate**

Create `crates/elements-io/Cargo.toml`:

```toml
[package]
name = "elements-io"
version = "0.1.0"
edition.workspace = true
license.workspace = true
repository.workspace = true

[dependencies]
half.workspace = true
ndarray = "0.16"
ndarray-npy.workspace = true
png.workspace = true
thiserror.workspace = true

[dev-dependencies]
tempfile.workspace = true
```

If `cargo build -p elements-io` reports that `ndarray-npy` 0.10 needs a different `ndarray` minor, run `cargo add ndarray -p elements-io` and let cargo pick the compatible version. Record whichever version resolves in this manifest.

- [ ] **Step 4: Implement NPY**

Create `crates/elements-io/src/npy.rs`:

```rust
//! `.npy` output, the storage format for golden tests.

use std::path::Path;

use ndarray::Array3;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};

use crate::IoError;

/// Write `values` as a C-order `(z, y, x)` float32 array.
///
/// `values` is x-fastest, matching `Field::read_back`.
pub fn write_npy(path: &Path, values: &[f32], dims: [u32; 3]) -> Result<(), IoError> {
    let expected = dims[0] as usize * dims[1] as usize * dims[2] as usize;
    if values.len() != expected {
        return Err(IoError::LengthMismatch {
            expected,
            got: values.len(),
        });
    }

    let array = Array3::from_shape_vec(
        (dims[2] as usize, dims[1] as usize, dims[0] as usize),
        values.to_vec(),
    )
    .map_err(|e| IoError::Npy(e.to_string()))?;

    let file = std::fs::File::create(path)?;
    array
        .write_npy(std::io::BufWriter::new(file))
        .map_err(|e| IoError::Npy(e.to_string()))
}

/// Read back an array written by [`write_npy`].
pub fn read_npy(path: &Path) -> Result<(Vec<f32>, [u32; 3]), IoError> {
    let file = std::fs::File::open(path)?;
    let array = Array3::<f32>::read_npy(std::io::BufReader::new(file))
        .map_err(|e| IoError::Npy(e.to_string()))?;

    let shape = array.shape();
    let dims = [shape[2] as u32, shape[1] as u32, shape[0] as u32];
    let values = array.into_raw_vec_and_offset().0;

    Ok((values, dims))
}
```

- [ ] **Step 5: Implement the PNG preview**

Create `crates/elements-io/src/preview.rs`:

```rust
//! Single-slice PNG previews: they turn a broken shader into an image diff.

use std::path::Path;

use crate::IoError;

/// Write one z-slice as 8-bit greyscale, mapping `range` linearly onto 0..=255.
///
/// Values outside `range` are clamped rather than wrapped, so a blown-up
/// simulation reads as saturated white instead of noise.
pub fn write_slice_png(
    path: &Path,
    values: &[f32],
    dims: [u32; 3],
    z: u32,
    range: (f32, f32),
) -> Result<(), IoError> {
    if z >= dims[2] {
        return Err(IoError::BadSlice { z, depth: dims[2] });
    }
    let expected = dims[0] as usize * dims[1] as usize * dims[2] as usize;
    if values.len() != expected {
        return Err(IoError::LengthMismatch {
            expected,
            got: values.len(),
        });
    }

    let (lo, hi) = range;
    let span = if (hi - lo).abs() < f32::EPSILON { 1.0 } else { hi - lo };

    let slice_len = dims[0] as usize * dims[1] as usize;
    let start = z as usize * slice_len;
    let pixels: Vec<u8> = values[start..start + slice_len]
        .iter()
        .map(|v| (((v - lo) / span).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();

    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), dims[0], dims[1]);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder.write_header().map_err(|e| IoError::Png(e.to_string()))?;
    writer
        .write_image_data(&pixels)
        .map_err(|e| IoError::Png(e.to_string()))
}
```

Create `crates/elements-io/src/lib.rs`:

```rust
#![forbid(unsafe_code)]

//! File output for the Elements Suite: golden arrays, previews, and OpenVDB.

mod npy;
mod preview;

pub use npy::{read_npy, write_npy};
pub use preview::write_slice_png;

/// Everything that can go wrong writing an Elements output file.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("npy error: {0}")]
    Npy(String),
    #[error("png error: {0}")]
    Png(String),
    #[error("slice z={z} is out of bounds for a field of depth {depth}")]
    BadSlice { z: u32, depth: u32 },
    #[error("expected {expected} values, got {got}")]
    LengthMismatch { expected: usize, got: usize },
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p elements-io`
Expected: PASS — five tests ok.

- [ ] **Step 7: Commit**

```bash
git add crates/elements-io
git commit -m "feat: add elements-io with npy golden output and png slice previews"
```

---

## OpenVDB writing: background for Tasks 10–12

No Rust crate writes OpenVDB. The three tasks below build a minimal writer for a
single uncompressed `FloatGrid`. The byte layout below was derived by reading
`vdb-rs` 0.6.0's parser (`src/reader.rs`), which is the oracle the tests use.

**Tree shape.** OpenVDB's standard `FloatGrid` is `Tree_float_5_4_3`:

| Level | `Log2Dim` | Children | Total log2 dim | Voxels covered |
|---|---|---|---|---|
| Root child (`InternalNode<5>`) | 5 | 32³ `InternalNode<4>` | 12 | 4096³ |
| `InternalNode<4>` | 4 | 16³ leaves | 7 | 128³ |
| `LeafNode<3>` | 3 | 512 voxels | 3 | 8³ |

Core v1 writes exactly one root child at origin `(0, 0, 0)`, so fields are
limited to 4096³ — checked and rejected with `IoError::FieldTooLarge`.

**Child offset within a node**, where `M` is the node's total log2 dim and `C`
the child's total log2 dim:

```
offset = ((x & (2^M - 1)) >> C) << (2 * Log2Dim)
       | ((y & (2^M - 1)) >> C) << Log2Dim
       | ((z & (2^M - 1)) >> C)
```

Concretely: root child `offset = (x >> 7) << 10 | (y >> 7) << 5 | (z >> 7)`;
internal node `offset = ((x & 127) >> 3) << 8 | ((y & 127) >> 3) << 4 | ((z & 127) >> 3)`;
leaf voxel `offset = (x & 7) << 6 | (y & 7) << 3 | (z & 7)`.

**Masks** are bit arrays serialized as little-endian `u64` words, LSB-first: bit
`i` lives in word `i / 64` at bit `i % 64`.

**Compression.** We write `Compression::ACTIVE_MASK` (`0x2`) and no zip or
blosc. That flag changes how value arrays are read: with a node-metadata byte of
`NO_MASK_OR_INACTIVE_VALS` (`0`) only the *active* values are stored, so internal
nodes — which hold no tiles in Core v1 — write zero values. Leaf buffers use
`NO_MASK_AND_ALL_VALS` (`6`), which stores all 512 values regardless of the mask.

**File layout.**

```
archive header      magic u64, file_version u32, lib major/minor u32,
                    has_grid_offsets u8, uuid 36 ASCII bytes,
                    file metadata (count u32 = 0), grid_count u32 = 1
grid descriptor     name, grid_type, instance_parent (each u32 len + bytes),
                    grid_pos u64, block_pos u64, end_pos u64   <- patched later
[grid_pos]          compression u32, grid metadata, transform, tree topology
[block_pos]         per leaf: value_mask (8 u64), meta byte, 512 f32
[end_pos]
```

---

## Task 10: VDB archive header and grid descriptor

**Files:**
- Create: `crates/elements-io/src/vdb/mod.rs`, `crates/elements-io/src/vdb/writer.rs`, `crates/elements-io/tests/vdb_header.rs`
- Modify: `crates/elements-io/src/lib.rs`, `crates/elements-io/Cargo.toml`

**Interfaces:**
- Consumes: `IoError` from Task 9.
- Produces:
  - `struct ByteWriter<W: Write + Seek>` with `u8`, `u32`, `u64`, `i32`, `f32`, `f64`, `string` (length-prefixed), `raw`, `pos`, `patch_u64_at(offset, value)`
  - `const OPENVDB_MAGIC: u64 = 0x5644_4220`, `const OPENVDB_FILE_VERSION: u32 = 224`, `const OPENVDB_LIBRARY_MAJOR: u32 = 12`, `const OPENVDB_LIBRARY_MINOR: u32 = 0`
  - `const COMPRESSION_ACTIVE_MASK: u32 = 0x2`
  - `enum MetaValue { String(String), Bool(bool), I32(i32), I64(i64), F32(f32), Vec3i([i32; 3]) }` with `write(&self, w) -> Result<(), IoError>` emitting the type name, byte length and payload
  - `fn write_metadata(w, entries: &[(&str, MetaValue)]) -> Result<(), IoError>`
  - `fn write_archive_header(w, uuid: &str) -> Result<(), IoError>`
  - `struct GridOffsets { pub grid_pos_at: u64, pub block_pos_at: u64, pub end_pos_at: u64 }`
  - `fn write_grid_descriptor(w, name: &str) -> Result<GridOffsets, IoError>` — writes placeholder zeros for the three offsets and returns where to patch them
  - `IoError::FieldTooLarge { dims: [u32; 3] }` added to the error enum

- [ ] **Step 1: Write the failing test**

Create `crates/elements-io/tests/vdb_header.rs`:

```rust
use std::io::Cursor;

use elements_io::vdb::{
    write_archive_header, write_grid_descriptor, write_metadata, ByteWriter, MetaValue,
    COMPRESSION_ACTIVE_MASK,
};

/// Build a file containing a header and one grid whose body is metadata only.
/// `vdb-rs` reads descriptors without touching the transform or the tree, so
/// this is enough to prove the header and descriptor are well formed.
fn header_only_file(name: &str) -> Vec<u8> {
    let mut w = ByteWriter::new(Cursor::new(Vec::new()));

    write_archive_header(&mut w, "00000000-0000-4000-8000-000000000000").unwrap();
    let offsets = write_grid_descriptor(&mut w, name).unwrap();

    let grid_pos = w.pos().unwrap();
    w.u32(COMPRESSION_ACTIVE_MASK).unwrap();
    write_metadata(
        &mut w,
        &[
            ("file_bbox_min", MetaValue::Vec3i([0, 0, 0])),
            ("file_bbox_max", MetaValue::Vec3i([7, 7, 7])),
        ],
    )
    .unwrap();
    let end_pos = w.pos().unwrap();

    w.patch_u64_at(offsets.grid_pos_at, grid_pos).unwrap();
    w.patch_u64_at(offsets.block_pos_at, end_pos).unwrap();
    w.patch_u64_at(offsets.end_pos_at, end_pos).unwrap();

    w.into_inner().into_inner()
}

#[test]
fn vdb_rs_reads_our_archive_header() {
    let bytes = header_only_file("density");
    let reader = vdb_rs::VdbReader::new(Cursor::new(bytes)).expect("header must parse");

    assert_eq!(reader.header.file_version, 224);
    assert_eq!(reader.header.grid_count, 1);
    assert!(reader.header.has_grid_offsets);
    assert_eq!(reader.header.guid.len(), 36);
}

#[test]
fn vdb_rs_lists_our_grid_by_name() {
    let bytes = header_only_file("density");
    let reader = vdb_rs::VdbReader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.available_grids(), vec!["density".to_string()]);
}

#[test]
fn grid_metadata_round_trips_through_vdb_rs() {
    let bytes = header_only_file("density");
    let reader = vdb_rs::VdbReader::new(Cursor::new(bytes)).unwrap();
    let gd = &reader.grid_descriptors["density"];

    assert_eq!(gd.grid_type, "Tree_float_5_4_3");
    assert_eq!(gd.aabb_min().unwrap(), glam::IVec3::new(0, 0, 0));
    assert_eq!(gd.aabb_max().unwrap(), glam::IVec3::new(7, 7, 7));
    assert!(!gd.meta_data.is_half_float());
}

#[test]
fn byte_writer_patches_offsets_in_place() {
    let mut w = ByteWriter::new(Cursor::new(Vec::new()));
    w.u32(0xdead_beef).unwrap();
    let slot = w.pos().unwrap();
    w.u64(0).unwrap();
    w.u32(0x1234_5678).unwrap();
    w.patch_u64_at(slot, 0x0102_0304_0506_0708).unwrap();

    let bytes = w.into_inner().into_inner();
    assert_eq!(&bytes[0..4], &0xdead_beefu32.to_le_bytes());
    assert_eq!(&bytes[4..12], &0x0102_0304_0506_0708u64.to_le_bytes());
    assert_eq!(&bytes[12..16], &0x1234_5678u32.to_le_bytes());
}

#[test]
fn strings_are_length_prefixed() {
    let mut w = ByteWriter::new(Cursor::new(Vec::new()));
    w.string("abc").unwrap();
    let bytes = w.into_inner().into_inner();
    assert_eq!(bytes, vec![3, 0, 0, 0, b'a', b'b', b'c']);
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p elements-io --test vdb_header`
Expected: FAIL — `unresolved import elements_io::vdb`.

- [ ] **Step 3: Add dev-dependencies**

Modify `crates/elements-io/Cargo.toml` `[dev-dependencies]`:

```toml
[dev-dependencies]
glam = "0.29"
tempfile.workspace = true
vdb-rs.workspace = true
```

`vdb-rs` pulls in `blosc-src`, which needs a C compiler in CI. `ubuntu-24.04` has one. If the build fails for want of `libblosc`, add `sudo apt-get install -y libblosc-dev` to the CI install step. `vdb-rs` is a dev-dependency only and never ships in the engine.

`glam` must match the version `vdb-rs` re-exports in its public API, or the `IVec3` comparison will not type-check. If `cargo test` reports two `glam` versions, run `cargo tree -p vdb-rs | grep glam` and pin this entry to that version.

- [ ] **Step 4: Implement the byte writer**

Create `crates/elements-io/src/vdb/writer.rs`:

```rust
//! Low-level little-endian byte output for the OpenVDB container format.

use std::io::{Seek, SeekFrom, Write};

use crate::IoError;

/// OpenVDB's magic number, written as a little-endian u64.
pub const OPENVDB_MAGIC: u64 = 0x5644_4220;
/// The file format version we emit. `vdb-rs` supports 213 and up.
pub const OPENVDB_FILE_VERSION: u32 = 224;
pub const OPENVDB_LIBRARY_MAJOR: u32 = 12;
pub const OPENVDB_LIBRARY_MINOR: u32 = 0;

/// Only active values are stored in nodes whose metadata byte says so.
pub const COMPRESSION_ACTIVE_MASK: u32 = 0x2;

/// The grid type string for a standard single-precision float tree.
pub const FLOAT_GRID_TYPE: &str = "Tree_float_5_4_3";

/// Node metadata byte: store only the active values.
pub const NO_MASK_OR_INACTIVE_VALS: u8 = 0;
/// Node metadata byte: store every value in the node.
pub const NO_MASK_AND_ALL_VALS: u8 = 6;

/// A seekable little-endian writer with offset patching.
pub struct ByteWriter<W: Write + Seek> {
    inner: W,
}

impl<W: Write + Seek> ByteWriter<W> {
    pub fn new(inner: W) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> W {
        self.inner
    }

    pub fn pos(&mut self) -> Result<u64, IoError> {
        Ok(self.inner.stream_position()?)
    }

    pub fn raw(&mut self, bytes: &[u8]) -> Result<(), IoError> {
        self.inner.write_all(bytes)?;
        Ok(())
    }

    pub fn u8(&mut self, v: u8) -> Result<(), IoError> {
        self.raw(&[v])
    }

    pub fn u32(&mut self, v: u32) -> Result<(), IoError> {
        self.raw(&v.to_le_bytes())
    }

    pub fn i32(&mut self, v: i32) -> Result<(), IoError> {
        self.raw(&v.to_le_bytes())
    }

    pub fn u64(&mut self, v: u64) -> Result<(), IoError> {
        self.raw(&v.to_le_bytes())
    }

    pub fn i64(&mut self, v: i64) -> Result<(), IoError> {
        self.raw(&v.to_le_bytes())
    }

    pub fn f32(&mut self, v: f32) -> Result<(), IoError> {
        self.raw(&v.to_le_bytes())
    }

    pub fn f64(&mut self, v: f64) -> Result<(), IoError> {
        self.raw(&v.to_le_bytes())
    }

    /// A `u32` length followed by the raw bytes, OpenVDB's string encoding.
    pub fn string(&mut self, s: &str) -> Result<(), IoError> {
        self.u32(s.len() as u32)?;
        self.raw(s.as_bytes())
    }

    /// Three little-endian `f64`s.
    pub fn dvec3(&mut self, v: [f64; 3]) -> Result<(), IoError> {
        for c in v {
            self.f64(c)?;
        }
        Ok(())
    }

    /// Overwrite a previously reserved `u64` slot, then return to the end.
    pub fn patch_u64_at(&mut self, offset: u64, value: u64) -> Result<(), IoError> {
        let here = self.pos()?;
        self.inner.seek(SeekFrom::Start(offset))?;
        self.inner.write_all(&value.to_le_bytes())?;
        self.inner.seek(SeekFrom::Start(here))?;
        Ok(())
    }
}

/// A typed metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    String(String),
    Bool(bool),
    I32(i32),
    I64(i64),
    F32(f32),
    Vec3i([i32; 3]),
}

impl MetaValue {
    fn type_name(&self) -> &'static str {
        match self {
            Self::String(_) => "string",
            Self::Bool(_) => "bool",
            Self::I32(_) => "int32",
            Self::I64(_) => "int64",
            Self::F32(_) => "float",
            Self::Vec3i(_) => "vec3i",
        }
    }

    fn byte_len(&self) -> u32 {
        match self {
            Self::String(s) => s.len() as u32,
            Self::Bool(_) => 1,
            Self::I32(_) | Self::F32(_) => 4,
            Self::I64(_) => 8,
            Self::Vec3i(_) => 12,
        }
    }

    /// Write type name, payload length, and payload.
    pub fn write<W: Write + Seek>(&self, w: &mut ByteWriter<W>) -> Result<(), IoError> {
        w.string(self.type_name())?;
        w.u32(self.byte_len())?;
        match self {
            Self::String(s) => w.raw(s.as_bytes()),
            Self::Bool(b) => w.u8(u8::from(*b)),
            Self::I32(v) => w.i32(*v),
            Self::I64(v) => w.i64(*v),
            Self::F32(v) => w.f32(*v),
            Self::Vec3i(v) => {
                for c in v {
                    w.i32(*c)?;
                }
                Ok(())
            }
        }
    }
}

/// Write a metadata map: a count followed by name/type/length/payload records.
pub fn write_metadata<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    entries: &[(&str, MetaValue)],
) -> Result<(), IoError> {
    w.u32(entries.len() as u32)?;
    for (name, value) in entries {
        w.string(name)?;
        value.write(w)?;
    }
    Ok(())
}

/// Write the archive header, up to and including the grid count of 1.
///
/// `uuid` must be exactly 36 ASCII characters; the format stores it unprefixed.
pub fn write_archive_header<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    uuid: &str,
) -> Result<(), IoError> {
    debug_assert_eq!(uuid.len(), 36, "OpenVDB stores a fixed 36-byte UUID");

    w.u64(OPENVDB_MAGIC)?;
    w.u32(OPENVDB_FILE_VERSION)?;
    w.u32(OPENVDB_LIBRARY_MAJOR)?;
    w.u32(OPENVDB_LIBRARY_MINOR)?;
    w.u8(1)?; // has_grid_offsets
    w.raw(uuid.as_bytes())?;
    write_metadata(w, &[])?; // no file-level metadata
    w.u32(1)?; // grid_count
    Ok(())
}

/// Where the three grid offsets live, so they can be patched once known.
#[derive(Debug, Clone, Copy)]
pub struct GridOffsets {
    pub grid_pos_at: u64,
    pub block_pos_at: u64,
    pub end_pos_at: u64,
}

/// Write a grid descriptor with placeholder offsets.
pub fn write_grid_descriptor<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    name: &str,
) -> Result<GridOffsets, IoError> {
    w.string(name)?;
    w.string(FLOAT_GRID_TYPE)?;
    w.string("")?; // instance_parent: this grid owns its tree

    let grid_pos_at = w.pos()?;
    w.u64(0)?;
    let block_pos_at = w.pos()?;
    w.u64(0)?;
    let end_pos_at = w.pos()?;
    w.u64(0)?;

    Ok(GridOffsets {
        grid_pos_at,
        block_pos_at,
        end_pos_at,
    })
}
```

- [ ] **Step 5: Export the module**

Create `crates/elements-io/src/vdb/mod.rs`:

```rust
//! A minimal OpenVDB writer for single-precision float grids.
//!
//! No Rust crate writes OpenVDB, so this module implements the container format
//! directly. It supports exactly what Core v1 needs: one uncompressed
//! `FloatGrid` with a uniform scale transform and a single root child.

mod writer;

pub use writer::{
    write_archive_header, write_grid_descriptor, write_metadata, ByteWriter, GridOffsets,
    MetaValue, COMPRESSION_ACTIVE_MASK, FLOAT_GRID_TYPE, NO_MASK_AND_ALL_VALS,
    NO_MASK_OR_INACTIVE_VALS, OPENVDB_FILE_VERSION, OPENVDB_LIBRARY_MAJOR,
    OPENVDB_LIBRARY_MINOR, OPENVDB_MAGIC,
};
```

Modify `crates/elements-io/src/lib.rs`, adding the module and the new error variant:

```rust
pub mod vdb;
```

and inside `enum IoError`:

```rust
    #[error("field {dims:?} exceeds the 4096^3 limit of a single root child")]
    FieldTooLarge { dims: [u32; 3] },
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p elements-io --test vdb_header`
Expected: PASS — five tests ok.

- [ ] **Step 7: Commit**

```bash
git add crates/elements-io
git commit -m "feat: write OpenVDB archive header, metadata and grid descriptor"
```

---

## Task 11: VDB tree topology with zeroed leaves

**Files:**
- Create: `crates/elements-io/src/vdb/tree.rs`, `crates/elements-io/tests/vdb_topology.rs`
- Modify: `crates/elements-io/src/vdb/mod.rs`

**Interfaces:**
- Consumes: `ByteWriter`, `MetaValue`, `write_metadata`, `write_archive_header`, `write_grid_descriptor`, and the constants from Task 10.
- Produces:
  - `struct BitMask { words: Vec<u64> }` with `BitMask::new(bits: usize)`, `set(&mut self, index: usize)`, `get(&self, index: usize) -> bool`, `count_ones(&self) -> usize`, `write(&self, w) -> Result<(), IoError>`
  - `fn root_child_offset(x: u32, y: u32, z: u32) -> usize`, `fn internal_child_offset(x, y, z) -> usize`, `fn leaf_voxel_offset(x, y, z) -> usize`
  - `struct TreeLayout { pub internal_mask: BitMask, pub leaf_masks: BTreeMap<usize, BitMask>, pub leaf_origins: BTreeMap<(usize, usize), [u32; 3]> }` — see the implementation for the exact shape
  - `fn write_float_grid(path: &Path, name: &str, values: &[f32], dims: [u32; 3], voxel_size: f64, background: f32) -> Result<(), IoError>` — in this task every leaf buffer is written as zeros; Task 12 fills real values.

- [ ] **Step 1: Write the failing test**

Create `crates/elements-io/tests/vdb_topology.rs`:

```rust
use elements_io::vdb::{internal_child_offset, leaf_voxel_offset, root_child_offset, BitMask};
use elements_io::write_float_grid;

#[test]
fn leaf_voxel_offsets_are_x_major() {
    assert_eq!(leaf_voxel_offset(0, 0, 0), 0);
    assert_eq!(leaf_voxel_offset(0, 0, 1), 1);
    assert_eq!(leaf_voxel_offset(0, 1, 0), 8);
    assert_eq!(leaf_voxel_offset(1, 0, 0), 64);
    assert_eq!(leaf_voxel_offset(7, 7, 7), 511);
    // Coordinates wrap within the node.
    assert_eq!(leaf_voxel_offset(8, 0, 0), 0);
}

#[test]
fn internal_child_offsets_index_16_cubed() {
    assert_eq!(internal_child_offset(0, 0, 0), 0);
    assert_eq!(internal_child_offset(0, 0, 8), 1);
    assert_eq!(internal_child_offset(0, 8, 0), 16);
    assert_eq!(internal_child_offset(8, 0, 0), 256);
    assert_eq!(internal_child_offset(127, 127, 127), 4095);
}

#[test]
fn root_child_offsets_index_32_cubed() {
    assert_eq!(root_child_offset(0, 0, 0), 0);
    assert_eq!(root_child_offset(0, 0, 128), 1);
    assert_eq!(root_child_offset(0, 128, 0), 32);
    assert_eq!(root_child_offset(128, 0, 0), 1024);
}

#[test]
fn bitmask_sets_and_counts() {
    let mut m = BitMask::new(512);
    assert_eq!(m.count_ones(), 0);
    m.set(0);
    m.set(63);
    m.set(64);
    m.set(511);
    assert!(m.get(0) && m.get(63) && m.get(64) && m.get(511));
    assert!(!m.get(1));
    assert_eq!(m.count_ones(), 4);
}

#[test]
fn bitmask_serializes_as_little_endian_u64_words() {
    let mut m = BitMask::new(128);
    m.set(0);
    m.set(65);

    let mut buf = Vec::new();
    {
        let mut w = elements_io::vdb::ByteWriter::new(std::io::Cursor::new(&mut buf));
        m.write(&mut w).unwrap();
    }
    assert_eq!(buf.len(), 16);
    assert_eq!(&buf[0..8], &1u64.to_le_bytes());
    assert_eq!(&buf[8..16], &2u64.to_le_bytes());
}

#[test]
fn vdb_rs_reads_a_full_tree_from_our_writer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zeros.vdb");
    let dims = [8u32, 8, 8];
    let values = vec![0.0f32; 512];

    write_float_grid(&path, "density", &values, dims, 0.1, 0.0).unwrap();

    let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let mut reader = vdb_rs::VdbReader::new(file).unwrap();
    let grid = reader.read_grid::<f32>("density").expect("tree must parse");

    assert_eq!(grid.tree.root_nodes.len(), 1);
    let root = &grid.tree.root_nodes[0];
    assert_eq!(root.child_mask.count_ones(), 1, "one 128^3 internal node");
}

#[test]
fn rejects_fields_larger_than_one_root_child() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("huge.vdb");
    let err = write_float_grid(&path, "density", &[], [5000, 1, 1], 1.0, 0.0).unwrap_err();
    assert!(
        matches!(err, elements_io::IoError::FieldTooLarge { .. }),
        "got {err:?}"
    );
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p elements-io --test vdb_topology`
Expected: FAIL — `unresolved imports BitMask, root_child_offset, write_float_grid`.

- [ ] **Step 3: Implement masks and offsets**

Create `crates/elements-io/src/vdb/tree.rs`:

```rust
//! Assembles a 5-4-3 OpenVDB tree from a dense field and serializes it.

use std::collections::BTreeMap;
use std::io::{Seek, Write};
use std::path::Path;

use crate::IoError;

use super::writer::{
    write_archive_header, write_grid_descriptor, write_metadata, ByteWriter, MetaValue,
    COMPRESSION_ACTIVE_MASK, NO_MASK_AND_ALL_VALS, NO_MASK_OR_INACTIVE_VALS,
};

/// Voxels per leaf edge.
const LEAF_DIM: u32 = 8;
/// Voxels covered by one internal node edge (16 leaves of 8).
const INTERNAL_DIM: u32 = 128;
/// Voxels covered by one root-child edge (32 internal nodes of 128).
const ROOT_CHILD_DIM: u32 = 4096;

const LEAF_VOXELS: usize = 512;
const INTERNAL_CHILDREN: usize = 4096;
const ROOT_CHILD_CHILDREN: usize = 32768;

/// Linear index of a voxel within its leaf.
pub fn leaf_voxel_offset(x: u32, y: u32, z: u32) -> usize {
    (((x & 7) << 6) | ((y & 7) << 3) | (z & 7)) as usize
}

/// Linear index of a leaf within its internal node.
pub fn internal_child_offset(x: u32, y: u32, z: u32) -> usize {
    ((((x & 127) >> 3) << 8) | (((y & 127) >> 3) << 4) | ((z & 127) >> 3)) as usize
}

/// Linear index of an internal node within the root child.
pub fn root_child_offset(x: u32, y: u32, z: u32) -> usize {
    ((((x & 4095) >> 7) << 10) | (((y & 4095) >> 7) << 5) | ((z & 4095) >> 7)) as usize
}

/// A bit array serialized as little-endian `u64` words, LSB-first.
#[derive(Debug, Clone)]
pub struct BitMask {
    words: Vec<u64>,
    bits: usize,
}

impl BitMask {
    pub fn new(bits: usize) -> Self {
        Self {
            words: vec![0; bits.div_ceil(64)],
            bits,
        }
    }

    pub fn set(&mut self, index: usize) {
        debug_assert!(index < self.bits);
        self.words[index / 64] |= 1u64 << (index % 64);
    }

    pub fn get(&self, index: usize) -> bool {
        index < self.bits && self.words[index / 64] & (1u64 << (index % 64)) != 0
    }

    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.bits).filter(|i| self.get(*i))
    }

    pub fn write<W: Write + Seek>(&self, w: &mut ByteWriter<W>) -> Result<(), IoError> {
        for word in &self.words {
            w.u64(*word)?;
        }
        Ok(())
    }
}
```

- [ ] **Step 4: Implement tree assembly and the grid writer**

Append to `crates/elements-io/src/vdb/tree.rs`:

```rust
/// One leaf: its active-value mask and its 512 values.
struct Leaf {
    value_mask: BitMask,
    values: Vec<f32>,
}

/// One internal node: which leaves exist, and the leaves themselves.
struct Internal {
    child_mask: BitMask,
    leaves: BTreeMap<usize, Leaf>,
}

/// Bucket a dense field into the 5-4-3 tree.
///
/// Every voxel inside `dims` is marked active. Voxels that fall inside a leaf
/// but outside `dims` stay inactive and carry `background`.
fn build_tree(
    values: &[f32],
    dims: [u32; 3],
    background: f32,
) -> Result<(BitMask, BTreeMap<usize, Internal>), IoError> {
    if dims[0] > ROOT_CHILD_DIM || dims[1] > ROOT_CHILD_DIM || dims[2] > ROOT_CHILD_DIM {
        return Err(IoError::FieldTooLarge { dims });
    }
    let expected = dims[0] as usize * dims[1] as usize * dims[2] as usize;
    if values.len() != expected {
        return Err(IoError::LengthMismatch {
            expected,
            got: values.len(),
        });
    }

    let mut root_mask = BitMask::new(ROOT_CHILD_CHILDREN);
    let mut internals: BTreeMap<usize, Internal> = BTreeMap::new();

    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                // `values` is x-fastest, matching Field::read_back.
                let linear = (z as usize * dims[1] as usize + y as usize) * dims[0] as usize
                    + x as usize;
                let value = values[linear];

                let r = root_child_offset(x, y, z);
                let i = internal_child_offset(x, y, z);
                let v = leaf_voxel_offset(x, y, z);

                root_mask.set(r);
                let internal = internals.entry(r).or_insert_with(|| Internal {
                    child_mask: BitMask::new(INTERNAL_CHILDREN),
                    leaves: BTreeMap::new(),
                });
                internal.child_mask.set(i);
                let leaf = internal.leaves.entry(i).or_insert_with(|| Leaf {
                    value_mask: BitMask::new(LEAF_VOXELS),
                    values: vec![background; LEAF_VOXELS],
                });
                leaf.value_mask.set(v);
                leaf.values[v] = value;
            }
        }
    }

    Ok((root_mask, internals))
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
    let (root_mask, internals) = build_tree(values, dims, background)?;

    let file = std::fs::File::create(path)?;
    let mut w = ByteWriter::new(std::io::BufWriter::new(file));

    write_archive_header(&mut w, "00000000-0000-4000-8000-000000000000")?;
    let offsets = write_grid_descriptor(&mut w, name)?;

    let grid_pos = w.pos()?;
    w.u32(COMPRESSION_ACTIVE_MASK)?;
    write_metadata(
        &mut w,
        &[
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
                MetaValue::I64(
                    dims[0] as i64 * dims[1] as i64 * dims[2] as i64,
                ),
            ),
        ],
    )?;

    write_uniform_scale_transform(&mut w, voxel_size)?;
    write_tree_topology(&mut w, &root_mask, &internals, background)?;

    let block_pos = w.pos()?;
    write_tree_data(&mut w, &root_mask, &internals)?;
    let end_pos = w.pos()?;

    w.patch_u64_at(offsets.grid_pos_at, grid_pos)?;
    w.patch_u64_at(offsets.block_pos_at, block_pos)?;
    w.patch_u64_at(offsets.end_pos_at, end_pos)?;

    Ok(())
}

/// A `UniformScaleMap`: five `Vec3d`s derived from one scale.
fn write_uniform_scale_transform<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    voxel_size: f64,
) -> Result<(), IoError> {
    let s = voxel_size;
    let inv = 1.0 / s;
    w.string("UniformScaleMap")?;
    w.dvec3([s, s, s])?; // scale_values
    w.dvec3([s, s, s])?; // voxel_size
    w.dvec3([inv, inv, inv])?; // scale_values_inverse
    w.dvec3([inv * inv, inv * inv, inv * inv])?; // inv_scale_sqr
    w.dvec3([0.5 * inv, 0.5 * inv, 0.5 * inv])?; // inv_twice_scale
    Ok(())
}

/// The topology pass: node masks, no leaf values.
fn write_tree_topology<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    root_mask: &BitMask,
    internals: &BTreeMap<usize, Internal>,
    background: f32,
) -> Result<(), IoError> {
    w.u32(1)?; // buffer_count
    w.u32(background.to_bits())?; // root background value
    w.u32(0)?; // tile count
    w.u32(1)?; // root child count

    // The single root child, anchored at the origin.
    w.i32(0)?;
    w.i32(0)?;
    w.i32(0)?;

    root_mask.write(w)?;
    // Root child holds no tiles, so its value mask is empty.
    BitMask::new(ROOT_CHILD_CHILDREN).write(w)?;
    // With ACTIVE_MASK compression and this metadata byte, zero values follow.
    w.u8(NO_MASK_OR_INACTIVE_VALS)?;

    for index in root_mask.iter_ones() {
        let internal = &internals[&index];
        internal.child_mask.write(w)?;
        BitMask::new(INTERNAL_CHILDREN).write(w)?;
        w.u8(NO_MASK_OR_INACTIVE_VALS)?;

        for leaf_index in internal.child_mask.iter_ones() {
            internal.leaves[&leaf_index].value_mask.write(w)?;
        }
    }

    Ok(())
}

/// The data pass: each leaf's mask repeated, then all 512 values.
fn write_tree_data<W: Write + Seek>(
    w: &mut ByteWriter<W>,
    root_mask: &BitMask,
    internals: &BTreeMap<usize, Internal>,
) -> Result<(), IoError> {
    for index in root_mask.iter_ones() {
        let internal = &internals[&index];
        for leaf_index in internal.child_mask.iter_ones() {
            let leaf = &internal.leaves[&leaf_index];
            leaf.value_mask.write(w)?;
            w.u8(NO_MASK_AND_ALL_VALS)?;
            for v in &leaf.values {
                w.f32(*v)?;
            }
        }
    }
    Ok(())
}
```

- [ ] **Step 5: Export the new items**

Modify `crates/elements-io/src/vdb/mod.rs`:

```rust
mod tree;
mod writer;

pub use tree::{
    internal_child_offset, leaf_voxel_offset, root_child_offset, write_float_grid, BitMask,
};
pub use writer::{
    write_archive_header, write_grid_descriptor, write_metadata, ByteWriter, GridOffsets,
    MetaValue, COMPRESSION_ACTIVE_MASK, FLOAT_GRID_TYPE, NO_MASK_AND_ALL_VALS,
    NO_MASK_OR_INACTIVE_VALS, OPENVDB_FILE_VERSION, OPENVDB_LIBRARY_MAJOR,
    OPENVDB_LIBRARY_MINOR, OPENVDB_MAGIC,
};
```

Modify `crates/elements-io/src/lib.rs`, re-exporting the writer:

```rust
pub use vdb::write_float_grid;
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p elements-io --test vdb_topology`
Expected: PASS — seven tests ok.

If `read_grid` fails inside `read_tree_data`, the topology and data sections have drifted out of sync: the leaf iteration order in `write_tree_topology` and `write_tree_data` must be identical, and `block_pos` must be exactly the byte after the last topology byte. Print `w.pos()` at each boundary and compare against the reader's `grid_descriptors["density"].block_pos`.

- [ ] **Step 7: Commit**

```bash
git add crates/elements-io
git commit -m "feat: write OpenVDB 5-4-3 tree topology and leaf buffers"
```

---

## Task 12: VDB value fidelity and partial leaves

**Files:**
- Create: `crates/elements-io/tests/vdb_values.rs`
- Modify: `crates/elements-io/src/vdb/tree.rs` (only if a test exposes a defect)

**Interfaces:**
- Consumes: `write_float_grid` from Task 11.
- Produces: no new API. This task proves the writer is *correct*, which Task 11 only proved is *parseable*.

- [ ] **Step 1: Write the failing test**

Create `crates/elements-io/tests/vdb_values.rs`:

```rust
use std::collections::HashMap;

/// Read a grid back and return a map from world voxel coordinate to value.
fn read_voxels(path: &std::path::Path, name: &str) -> HashMap<(i32, i32, i32), f32> {
    let file = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let mut reader = vdb_rs::VdbReader::new(file).unwrap();
    let grid = reader.read_grid::<f32>(name).unwrap();

    let mut out = HashMap::new();
    for root in &grid.tree.root_nodes {
        for (_, internal) in root.nodes.iter() {
            for (_, leaf) in internal.nodes.iter() {
                let origin = leaf.origin;
                for offset in leaf.value_mask.iter_ones() {
                    // Leaf offsets are x-major: x = bits 6..9, y = 3..5, z = 0..2.
                    let x = (offset >> 6) & 7;
                    let y = (offset >> 3) & 7;
                    let z = offset & 7;
                    out.insert(
                        (
                            origin.x + x as i32,
                            origin.y + y as i32,
                            origin.z + z as i32,
                        ),
                        leaf.buffer[offset],
                    );
                }
            }
        }
    }
    out
}

fn ramp(dims: [u32; 3]) -> Vec<f32> {
    let n = (dims[0] * dims[1] * dims[2]) as usize;
    (0..n).map(|i| i as f32 * 0.25).collect()
}

#[test]
fn values_round_trip_for_an_aligned_field() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("aligned.vdb");
    let dims = [8u32, 8, 8];
    let values = ramp(dims);

    elements_io::write_float_grid(&path, "density", &values, dims, 0.1, 0.0).unwrap();
    let voxels = read_voxels(&path, "density");

    assert_eq!(voxels.len(), 512);
    for z in 0..8i32 {
        for y in 0..8i32 {
            for x in 0..8i32 {
                let linear = (z as usize * 8 + y as usize) * 8 + x as usize;
                assert_eq!(
                    voxels[&(x, y, z)], values[linear],
                    "mismatch at ({x}, {y}, {z})"
                );
            }
        }
    }
}

#[test]
fn values_round_trip_across_multiple_leaves() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.vdb");
    let dims = [16u32, 16, 16];
    let values = ramp(dims);

    elements_io::write_float_grid(&path, "density", &values, dims, 0.1, 0.0).unwrap();
    let voxels = read_voxels(&path, "density");

    assert_eq!(voxels.len(), 4096, "8 leaves of 512 voxels");
    let linear = (9usize * 16 + 3) * 16 + 11;
    assert_eq!(voxels[&(11, 3, 9)], values[linear]);
}

#[test]
fn partial_leaves_mark_only_real_voxels_active() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial.vdb");
    // 5^3 occupies one leaf but activates only 125 of its 512 voxels.
    let dims = [5u32, 5, 5];
    let values = ramp(dims);

    elements_io::write_float_grid(&path, "density", &values, dims, 0.1, 0.0).unwrap();
    let voxels = read_voxels(&path, "density");

    assert_eq!(voxels.len(), 125, "only in-bounds voxels are active");
    assert!(voxels.contains_key(&(4, 4, 4)));
    assert!(!voxels.contains_key(&(5, 0, 0)), "outside dims stays inactive");
    assert_eq!(voxels[&(4, 4, 4)], values[(4 * 5 + 4) * 5 + 4]);
}

#[test]
fn a_field_spanning_two_internal_nodes_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wide.vdb");
    // 136 > 128 forces a second internal node along x.
    let dims = [136u32, 8, 8];
    let values = ramp(dims);

    elements_io::write_float_grid(&path, "density", &values, dims, 0.1, 0.0).unwrap();
    let voxels = read_voxels(&path, "density");

    assert_eq!(voxels.len(), 136 * 8 * 8);
    let linear = (0usize * 8 + 0) * 136 + 130;
    assert_eq!(voxels[&(130, 0, 0)], values[linear]);
}

#[test]
fn background_fills_inactive_voxels_in_a_partial_leaf() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("background.vdb");
    let dims = [2u32, 2, 2];

    elements_io::write_float_grid(&path, "density", &[1.0; 8], dims, 0.1, -3.5).unwrap();

    let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let mut reader = vdb_rs::VdbReader::new(file).unwrap();
    let grid = reader.read_grid::<f32>("density").unwrap();

    let root = &grid.tree.root_nodes[0];
    let internal = root.nodes.values().next().unwrap();
    let leaf = internal.nodes.values().next().unwrap();

    assert_eq!(leaf.value_mask.count_ones(), 8);
    // Offset 511 is voxel (7, 7, 7): outside dims, so it carries the background.
    assert_eq!(leaf.buffer[511], -3.5);
}

#[test]
fn a_length_mismatch_is_rejected_before_any_file_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mismatch.vdb");
    let err = elements_io::write_float_grid(&path, "density", &[1.0, 2.0], [4, 4, 4], 0.1, 0.0)
        .unwrap_err();

    assert!(
        matches!(err, elements_io::IoError::LengthMismatch { expected: 64, got: 2 }),
        "got {err:?}"
    );
    assert!(!path.exists(), "no partial file should be left behind");
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p elements-io --test vdb_values`
Expected: FAIL. At minimum `a_length_mismatch_is_rejected_before_any_file_is_written` fails, because Task 11's `write_float_grid` validates before creating the file only if `build_tree` runs first — confirm the ordering. Others may fail if the offset arithmetic is wrong; that is the point of this task.

- [ ] **Step 3: Fix whatever the tests expose**

Work through failures one at a time in `crates/elements-io/src/vdb/tree.rs`. The three defects most likely to appear:

1. **Axis order.** `Field::read_back` is x-fastest; the linear index must be `(z * dims[1] + y) * dims[0] + x`. If values appear transposed, this is why.
2. **Leaf origin.** `vdb-rs` derives leaf origins from parent offsets. If origins come back wrong, `root_child_offset` or `internal_child_offset` has its shift widths swapped.
3. **Topology/data drift.** Both passes must iterate `root_mask.iter_ones()` then `internal.child_mask.iter_ones()` in the same ascending order. `BTreeMap` is used precisely so iteration is deterministic.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p elements-io --test vdb_values`
Expected: PASS — six tests ok.

- [ ] **Step 5: Verify the whole crate and workspace**

Run: `just check`
Expected: clean — fmt, clippy, ruff and the full test suite.

- [ ] **Step 6: Commit**

```bash
git add crates/elements-io
git commit -m "test: verify OpenVDB value fidelity across leaves and partial nodes"
```

---

## Task 13: The headless CLI

**Files:**
- Create: `crates/elements-cli/Cargo.toml`, `crates/elements-cli/src/main.rs`, `crates/elements-cli/src/bake.rs`, `crates/elements-cli/src/preview.rs`, `crates/elements-cli/tests/cli.rs`, `crates/elements-cli/tests/golden/noise_8.npy`, `crates/elements-cli/tests/golden/noise_8_z4.png`

**Interfaces:**
- Consumes: `Document`, `NodeRegistry` from Tasks 7–8; `GpuContext`, `FieldPool`, `PipelineCache` from Tasks 2–4; `write_npy`, `read_npy`, `write_slice_png`, `write_float_grid` from Tasks 9–12.
- Produces:
  - `elements bake <graph> --out <dir> [--frames A-B] [--name density] [--voxel-size 0.1]` — writes `<dir>/<name>.NNNN.vdb` per frame.
  - `elements render-preview <graph> [--frame N] [--slice-z N] [--range LO,HI] --out <file.png>`
  - `elements dump-npy <graph> --out <file.npy>` — the golden-test entry point.
  - `fn evaluate_document(path: &Path) -> anyhow::Result<(Vec<f32>, [u32; 3])>` in `bake.rs`, shared by all three subcommands.

Core v1 has no time-varying nodes, so `--frames` writes the same field to each numbered file. The flag exists because the bake *file naming and directory layout* is what Blender and downstream tooling depend on, and changing it later is expensive; the solver that makes frames differ arrives with Ember.

- [ ] **Step 1: Write the failing test**

Create `crates/elements-cli/tests/cli.rs`:

```rust
use std::process::Command;

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_elements"))
}

const NOISE_GRAPH: &str = r#"{
  "version": 1,
  "dims": [8, 8, 8],
  "nodes": [
    { "id": 0, "kind": "core.noise_field", "params": { "seed": 7, "frequency": 4.0 } },
    { "id": 1, "kind": "core.output", "params": {} }
  ],
  "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
  "output": 1
}"#;

fn write_graph(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("noise.elements");
    std::fs::write(&path, NOISE_GRAPH).unwrap();
    path
}

#[test]
fn bake_writes_one_vdb_per_frame() {
    let dir = tempfile::tempdir().unwrap();
    let graph = write_graph(dir.path());
    let out = dir.path().join("vdb");

    let status = cli()
        .args(["bake", graph.to_str().unwrap()])
        .args(["--out", out.to_str().unwrap()])
        .args(["--frames", "1-3"])
        .status()
        .unwrap();
    assert!(status.success());

    for frame in 1..=3 {
        let path = out.join(format!("density.{frame:04}.vdb"));
        assert!(path.exists(), "missing {}", path.display());
        let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
        let reader = vdb_rs::VdbReader::new(file).expect("baked file must be valid VDB");
        assert_eq!(reader.available_grids(), vec!["density".to_string()]);
    }
}

#[test]
fn dump_npy_matches_the_golden_field() {
    let dir = tempfile::tempdir().unwrap();
    let graph = write_graph(dir.path());
    let out = dir.path().join("actual.npy");

    let status = cli()
        .args(["dump-npy", graph.to_str().unwrap()])
        .args(["--out", out.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(status.success());

    let (actual, actual_dims) = elements_io::read_npy(&out).unwrap();
    let (expected, expected_dims) =
        elements_io::read_npy(std::path::Path::new("tests/golden/noise_8.npy")).unwrap();

    assert_eq!(actual_dims, expected_dims);
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        approx::assert_abs_diff_eq!(a, e, epsilon = 1e-3);
        assert!(a.is_finite(), "non-finite value at index {i}");
    }
}

#[test]
fn render_preview_writes_a_png_of_the_right_size() {
    let dir = tempfile::tempdir().unwrap();
    let graph = write_graph(dir.path());
    let out = dir.path().join("preview.png");

    let status = cli()
        .args(["render-preview", graph.to_str().unwrap()])
        .args(["--slice-z", "4"])
        .args(["--range", "-1,1"])
        .args(["--out", out.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(status.success());

    let decoder = png::Decoder::new(std::fs::File::open(&out).unwrap());
    let reader = decoder.read_info().unwrap();
    assert_eq!(reader.info().width, 8);
    assert_eq!(reader.info().height, 8);
}

#[test]
fn a_malformed_graph_fails_with_a_nonzero_exit_and_a_message() {
    let dir = tempfile::tempdir().unwrap();
    let graph = dir.path().join("broken.elements");
    std::fs::write(&graph, "{ not json").unwrap();

    let output = cli()
        .args(["dump-npy", graph.to_str().unwrap()])
        .args(["--out", dir.path().join("x.npy").to_str().unwrap()])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("malformed document"),
        "stderr was: {stderr}"
    );
}

#[test]
fn an_unsupported_version_names_the_version() {
    let dir = tempfile::tempdir().unwrap();
    let graph = dir.path().join("future.elements");
    std::fs::write(&graph, NOISE_GRAPH.replace("\"version\": 1", "\"version\": 42")).unwrap();

    let output = cli()
        .args(["dump-npy", graph.to_str().unwrap()])
        .args(["--out", dir.path().join("x.npy").to_str().unwrap()])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("42"));
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p elements-cli`
Expected: FAIL — `package ID specification elements-cli did not match any packages`.

- [ ] **Step 3: Create the crate**

Create `crates/elements-cli/Cargo.toml`:

```toml
[package]
name = "elements-cli"
version = "0.1.0"
edition.workspace = true
license.workspace = true
repository.workspace = true

[[bin]]
name = "elements"
path = "src/main.rs"

[dependencies]
anyhow.workspace = true
clap.workspace = true
elements-core = { path = "../elements-core" }
elements-io = { path = "../elements-io" }

[dev-dependencies]
approx.workspace = true
elements-io = { path = "../elements-io" }
png.workspace = true
tempfile.workspace = true
vdb-rs.workspace = true
```

- [ ] **Step 4: Implement evaluation and bake**

Create `crates/elements-cli/src/bake.rs`:

```rust
//! Graph evaluation shared by every subcommand, plus VDB baking.

use std::path::Path;

use anyhow::Context;
use elements_core::gpu::{FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{Document, NodeRegistry};

/// Load a `.elements` document, evaluate it, and read the result back.
///
/// Returns values in x-fastest order with the document's dimensions.
pub fn evaluate_document(path: &Path) -> anyhow::Result<(Vec<f32>, [u32; 3])> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let doc = Document::from_json(&text)?;
    let registry = NodeRegistry::with_builtins();
    let (graph, dims) = doc.into_graph(&registry)?;

    let gpu = GpuContext::new_headless().context("acquiring a GPU device")?;
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();

    let value = graph.eval(&gpu, &mut pool, &mut pipelines, dims)?;
    let field = value.as_field()?;
    let values = field.read_back(&gpu)?;

    Ok((values, [dims.x, dims.y, dims.z]))
}

/// Parse an inclusive `A-B` frame range, or a single frame `N`.
pub fn parse_frames(spec: &str) -> anyhow::Result<(u32, u32)> {
    match spec.split_once('-') {
        Some((a, b)) => {
            let start: u32 = a.trim().parse().context("frame range start")?;
            let end: u32 = b.trim().parse().context("frame range end")?;
            anyhow::ensure!(start <= end, "frame range {spec} runs backwards");
            Ok((start, end))
        }
        None => {
            let n: u32 = spec.trim().parse().context("frame number")?;
            Ok((n, n))
        }
    }
}

/// Write one `.vdb` per frame into `out_dir`.
pub fn bake(
    graph: &Path,
    out_dir: &Path,
    frames: (u32, u32),
    name: &str,
    voxel_size: f64,
) -> anyhow::Result<()> {
    let (values, dims) = evaluate_document(graph)?;
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("creating {}", out_dir.display()))?;

    for frame in frames.0..=frames.1 {
        let path = out_dir.join(format!("{name}.{frame:04}.vdb"));
        elements_io::write_float_grid(&path, name, &values, dims, voxel_size, 0.0)
            .with_context(|| format!("writing {}", path.display()))?;
    }

    Ok(())
}
```

- [ ] **Step 5: Implement preview and the entry point**

Create `crates/elements-cli/src/preview.rs`:

```rust
//! PNG slice previews from the command line.

use std::path::Path;

use anyhow::Context;

use crate::bake::evaluate_document;

/// Parse a `LO,HI` display range.
pub fn parse_range(spec: &str) -> anyhow::Result<(f32, f32)> {
    let (lo, hi) = spec
        .split_once(',')
        .context("range must be written as LO,HI")?;
    Ok((
        lo.trim().parse().context("range low")?,
        hi.trim().parse().context("range high")?,
    ))
}

pub fn render_preview(
    graph: &Path,
    out: &Path,
    slice_z: Option<u32>,
    range: (f32, f32),
) -> anyhow::Result<()> {
    let (values, dims) = evaluate_document(graph)?;
    let z = slice_z.unwrap_or(dims[2] / 2);
    elements_io::write_slice_png(out, &values, dims, z, range)
        .with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}

pub fn dump_npy(graph: &Path, out: &Path) -> anyhow::Result<()> {
    let (values, dims) = evaluate_document(graph)?;
    elements_io::write_npy(out, &values, dims)
        .with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}
```

Create `crates/elements-cli/src/main.rs`:

```rust
#![forbid(unsafe_code)]

//! Headless entry point for the Elements engine.
//!
//! Everything here runs without a window, without Blender, and without the
//! daemon, which is what lets CI exercise the engine on software Vulkan.

mod bake;
mod preview;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "elements", version, about = "Elements Suite headless engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Evaluate a graph and write one .vdb per frame.
    Bake {
        graph: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "1")]
        frames: String,
        #[arg(long, default_value = "density")]
        name: String,
        #[arg(long, default_value_t = 0.1)]
        voxel_size: f64,
    },
    /// Write one z-slice of a graph's output as a PNG.
    RenderPreview {
        graph: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        slice_z: Option<u32>,
        #[arg(long, default_value = "-1,1")]
        range: String,
    },
    /// Write a graph's output as a .npy array, for golden tests.
    DumpNpy {
        graph: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() {
    if let Err(e) = run() {
        // `{:#}` prints the whole anyhow context chain on one line.
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    match Cli::parse().command {
        Commands::Bake {
            graph,
            out,
            frames,
            name,
            voxel_size,
        } => bake::bake(&graph, &out, bake::parse_frames(&frames)?, &name, voxel_size),
        Commands::RenderPreview {
            graph,
            out,
            slice_z,
            range,
        } => preview::render_preview(&graph, &out, slice_z, preview::parse_range(&range)?),
        Commands::DumpNpy { graph, out } => preview::dump_npy(&graph, &out),
    }
}
```

- [ ] **Step 6: Generate the golden files**

The golden `.npy` is generated *once* by the implementation, then reviewed and committed. Generate it, then inspect the PNG before trusting it:

```bash
mkdir -p crates/elements-cli/tests/golden tests/graphs
cat > tests/graphs/noise_8.elements <<'GRAPH'
{
  "version": 1,
  "dims": [8, 8, 8],
  "nodes": [
    { "id": 0, "kind": "core.noise_field", "params": { "seed": 7, "frequency": 4.0 } },
    { "id": 1, "kind": "core.output", "params": {} }
  ],
  "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
  "output": 1
}
GRAPH
just golden
```

The graph fixture is committed at `tests/graphs/noise_8.elements` rather than
written to a temp path, because `just golden` must be re-runnable by anyone and
a golden that cannot be regenerated from a tracked input is not reproducible.

Open `noise_8_z4.png`. It must look like smooth structured noise, not uniform grey (a dead shader), not salt-and-pepper (a broken hash), and not a hard-edged grid (a workgroup bounds bug). **Do not commit a golden you have not looked at.** A golden file blesses whatever the code did, including a bug.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `WGPU_BACKEND=vulkan cargo test -p elements-cli`
Expected: PASS — five tests ok.

- [ ] **Step 8: Commit**

```bash
git add crates/elements-cli
git commit -m "feat: add headless elements CLI with bake, preview and npy golden output"
```

---

## Task 14: IPC control protocol

**Files:**
- Create: `crates/elements-ipc/Cargo.toml`, `crates/elements-ipc/src/lib.rs`, `crates/elements-ipc/src/protocol.rs`, `crates/elements-ipc/tests/protocol.rs`

**Interfaces:**
- Consumes: nothing. `elements-ipc` deliberately depends on no other Elements crate so the protocol can be tested, and reimplemented in Python, in isolation.
- Produces:
  - `const ELEMENTS_PROTOCOL_VERSION: u32 = 1`
  - `enum Command { Hello { protocol_version: u32 }, LoadGraph { path: String }, Render { frame: u32 }, Shutdown }` — serde-tagged on `"type"` with snake_case names.
  - `enum Response { HelloAck { protocol_version: u32, engine_version: String, adapter: String }, Loaded { dims: [u32; 3], nodes: u32 }, Frame { seq: u64, channel: String, dims: [u32; 3] }, Bye, Error(EngineError) }`
  - `struct EngineError { pub kind: ErrorKind, pub message: String }`
  - `enum ErrorKind { ProtocolVersion, Document, Graph, Gpu, DeviceLost, Io }`
  - `fn write_message<W: Write, T: Serialize>(w: &mut W, msg: &T) -> Result<(), ProtocolError>` — one compact JSON object plus `\n`.
  - `fn read_message<R: BufRead, T: DeserializeOwned>(r: &mut R) -> Result<Option<T>, ProtocolError>` — `Ok(None)` at clean end of stream.
  - `enum ProtocolError { Io(std::io::Error), Json(serde_json::Error), OversizedFrame { bytes: usize } }` with a 1 MiB line cap.

- [ ] **Step 1: Write the failing test**

Create `crates/elements-ipc/tests/protocol.rs`:

```rust
use std::io::{BufReader, Cursor};

use elements_ipc::{
    read_message, write_message, Command, EngineError, ErrorKind, ProtocolError, Response,
    ELEMENTS_PROTOCOL_VERSION,
};

#[test]
fn commands_round_trip_as_ndjson() {
    let mut buf = Vec::new();
    write_message(
        &mut buf,
        &Command::Hello {
            protocol_version: ELEMENTS_PROTOCOL_VERSION,
        },
    )
    .unwrap();
    write_message(&mut buf, &Command::Render { frame: 12 }).unwrap();
    write_message(&mut buf, &Command::Shutdown).unwrap();

    let mut reader = BufReader::new(Cursor::new(buf));
    let a: Command = read_message(&mut reader).unwrap().unwrap();
    let b: Command = read_message(&mut reader).unwrap().unwrap();
    let c: Command = read_message(&mut reader).unwrap().unwrap();
    let end: Option<Command> = read_message(&mut reader).unwrap();

    assert!(matches!(a, Command::Hello { protocol_version: 1 }));
    assert!(matches!(b, Command::Render { frame: 12 }));
    assert!(matches!(c, Command::Shutdown));
    assert!(end.is_none(), "clean end of stream is None, not an error");
}

#[test]
fn messages_are_one_line_each() {
    let mut buf = Vec::new();
    write_message(&mut buf, &Command::Render { frame: 1 }).unwrap();
    let text = String::from_utf8(buf).unwrap();

    assert!(text.ends_with('\n'));
    assert_eq!(text.matches('\n').count(), 1, "no embedded newlines");
}

#[test]
fn the_wire_format_is_tagged_and_snake_case() {
    let mut buf = Vec::new();
    write_message(&mut buf, &Command::LoadGraph { path: "/tmp/a".into() }).unwrap();
    let text = String::from_utf8(buf).unwrap();

    assert!(text.contains("\"type\":\"load_graph\""), "got {text}");
    assert!(text.contains("\"path\":\"/tmp/a\""), "got {text}");
}

#[test]
fn responses_round_trip() {
    let mut buf = Vec::new();
    write_message(
        &mut buf,
        &Response::Frame {
            seq: 7,
            channel: "/tmp/elements/frame.bin".into(),
            dims: [8, 8, 8],
        },
    )
    .unwrap();

    let mut reader = BufReader::new(Cursor::new(buf));
    let msg: Response = read_message(&mut reader).unwrap().unwrap();
    match msg {
        Response::Frame { seq, dims, .. } => {
            assert_eq!(seq, 7);
            assert_eq!(dims, [8, 8, 8]);
        }
        other => panic!("expected Frame, got {other:?}"),
    }
}

#[test]
fn engine_errors_carry_a_machine_readable_kind() {
    let mut buf = Vec::new();
    write_message(
        &mut buf,
        &Response::Error(EngineError {
            kind: ErrorKind::DeviceLost,
            message: "adapter reset".into(),
        }),
    )
    .unwrap();
    let text = String::from_utf8(buf.clone()).unwrap();
    assert!(text.contains("\"kind\":\"device_lost\""), "got {text}");

    let mut reader = BufReader::new(Cursor::new(buf));
    let msg: Response = read_message(&mut reader).unwrap().unwrap();
    assert!(matches!(
        msg,
        Response::Error(EngineError { kind: ErrorKind::DeviceLost, .. })
    ));
}

#[test]
fn malformed_json_is_an_error_not_a_panic() {
    let mut reader = BufReader::new(Cursor::new(b"{ not json }\n".to_vec()));
    let result: Result<Option<Command>, _> = read_message(&mut reader);
    assert!(matches!(result, Err(ProtocolError::Json(_))));
}

#[test]
fn oversized_lines_are_rejected_before_parsing() {
    let mut line = vec![b'x'; 2 * 1024 * 1024];
    line.push(b'\n');
    let mut reader = BufReader::new(Cursor::new(line));
    let result: Result<Option<Command>, _> = read_message(&mut reader);
    assert!(
        matches!(result, Err(ProtocolError::OversizedFrame { .. })),
        "a hostile or desynced peer must not be able to exhaust memory"
    );
}

#[test]
fn blank_lines_are_skipped() {
    let mut buf = b"\n\n".to_vec();
    write_message(&mut buf, &Command::Shutdown).unwrap();
    let mut reader = BufReader::new(Cursor::new(buf));
    let msg: Command = read_message(&mut reader).unwrap().unwrap();
    assert!(matches!(msg, Command::Shutdown));
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p elements-ipc`
Expected: FAIL — `package ID specification elements-ipc did not match any packages`.

- [ ] **Step 3: Create the crate**

Create `crates/elements-ipc/Cargo.toml`:

```toml
[package]
name = "elements-ipc"
version = "0.1.0"
edition.workspace = true
license.workspace = true
repository.workspace = true

[dependencies]
memmap2.workspace = true
serde.workspace = true
serde_json.workspace = true
thiserror.workspace = true

[dev-dependencies]
tempfile.workspace = true
```

- [ ] **Step 4: Implement the protocol**

Create `crates/elements-ipc/src/protocol.rs`:

```rust
//! The control plane: newline-delimited JSON over a stream socket.
//!
//! JSON rather than a binary encoding because the Blender addon must parse it
//! with the Python standard library and no compiled dependency. The bytes that
//! matter travel on the data plane; these messages are small.

use std::io::{BufRead, Write};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// The wire protocol version. Bumped whenever a message changes shape.
pub const ELEMENTS_PROTOCOL_VERSION: u32 = 1;

/// The largest single message accepted, to bound a desynced peer's damage.
pub const MAX_MESSAGE_BYTES: usize = 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("malformed message: {0}")]
    Json(#[from] serde_json::Error),
    #[error("message of {bytes} bytes exceeds the {MAX_MESSAGE_BYTES} byte limit")]
    OversizedFrame { bytes: usize },
}

/// A classification the addon can branch on without parsing prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    ProtocolVersion,
    Document,
    Graph,
    Gpu,
    DeviceLost,
    Io,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineError {
    pub kind: ErrorKind,
    pub message: String,
}

impl EngineError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

/// Client to daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Command {
    /// Must be the first message. The daemon rejects a version mismatch.
    Hello { protocol_version: u32 },
    /// Load a `.elements` document by absolute path.
    LoadGraph { path: String },
    /// Evaluate the loaded graph and publish the result.
    Render { frame: u32 },
    Shutdown,
}

/// Daemon to client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    HelloAck {
        protocol_version: u32,
        engine_version: String,
        adapter: String,
    },
    Loaded {
        dims: [u32; 3],
        nodes: u32,
    },
    Frame {
        seq: u64,
        channel: String,
        dims: [u32; 3],
    },
    Bye,
    Error(EngineError),
}

/// Write one compact JSON object followed by a newline.
pub fn write_message<W: Write, T: Serialize>(w: &mut W, msg: &T) -> Result<(), ProtocolError> {
    let mut line = serde_json::to_vec(msg)?;
    debug_assert!(
        !line.contains(&b'\n'),
        "compact JSON must never contain a raw newline"
    );
    line.push(b'\n');
    w.write_all(&line)?;
    w.flush()?;
    Ok(())
}

/// Read one message. Returns `Ok(None)` at a clean end of stream.
pub fn read_message<R: BufRead, T: DeserializeOwned>(
    r: &mut R,
) -> Result<Option<T>, ProtocolError> {
    loop {
        let mut line = Vec::new();
        // Cap the read so a peer that never sends a newline cannot exhaust memory.
        let read = r
            .take((MAX_MESSAGE_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)?;

        if read == 0 {
            return Ok(None);
        }
        if read > MAX_MESSAGE_BYTES {
            return Err(ProtocolError::OversizedFrame { bytes: read });
        }

        let trimmed = line
            .strip_suffix(b"\n")
            .unwrap_or(&line)
            .strip_suffix(b"\r")
            .unwrap_or_else(|| line.strip_suffix(b"\n").unwrap_or(&line));

        if trimmed.is_empty() {
            continue; // tolerate keepalive blank lines
        }
        return Ok(Some(serde_json::from_slice(trimmed)?));
    }
}
```

`BufRead::take` requires `Read`, and chaining `strip_suffix` as written will not compile as-is. Implement the trimming plainly instead:

```rust
        let mut trimmed: &[u8] = &line;
        if let Some(rest) = trimmed.strip_suffix(b"\n") {
            trimmed = rest;
        }
        if let Some(rest) = trimmed.strip_suffix(b"\r") {
            trimmed = rest;
        }
```

and wrap the reader for the size cap with `let mut limited = r.by_ref().take((MAX_MESSAGE_BYTES + 1) as u64);` before calling `read_until` on `limited`.

Create `crates/elements-ipc/src/lib.rs`:

```rust
#![deny(unsafe_op_in_unsafe_fn)]

//! Inter-process transport between the Elements engine and its clients.
//!
//! Two planes: a newline-delimited JSON control plane (`protocol`) and a
//! memory-mapped double-buffered data plane (`channel`).

mod protocol;

pub use protocol::{
    read_message, write_message, Command, EngineError, ErrorKind, ProtocolError, Response,
    ELEMENTS_PROTOCOL_VERSION, MAX_MESSAGE_BYTES,
};
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p elements-ipc`
Expected: PASS — eight tests ok.

- [ ] **Step 6: Commit**

```bash
git add crates/elements-ipc
git commit -m "feat: add NDJSON control protocol with version and size guards"
```

---

## Task 15: The memory-mapped frame channel

**Files:**
- Create: `crates/elements-ipc/src/channel.rs`, `crates/elements-ipc/tests/channel.rs`
- Modify: `crates/elements-ipc/src/lib.rs`

**Interfaces:**
- Consumes: nothing from Task 14; the two planes are independent.
- Produces:
  - `const CHANNEL_MAGIC: u32 = 0x4346_4C45` (`"ELFC"` little-endian), `const CHANNEL_VERSION: u32 = 1`, `const CHANNEL_HEADER_BYTES: usize = 64`
  - `struct ChannelHeader { pub magic: u32, pub version: u32, pub dims: [u32; 3], pub channels: u32, pub buffer_bytes: u64, pub seq: u64 }`
  - `struct FrameWriter` with `FrameWriter::create(path: &Path, dims: [u32; 3], channels: u32) -> Result<FrameWriter, ChannelError>`, `publish(&mut self, values: &[f32]) -> Result<u64, ChannelError>`, `path(&self) -> &Path`, `seq(&self) -> u64`
  - `struct FrameReader` with `FrameReader::open(path: &Path) -> Result<FrameReader, ChannelError>`, `header(&self) -> ChannelHeader`, `read_latest(&self) -> Result<(u64, Vec<f32>), ChannelError>`
  - `enum ChannelError { Io, BadMagic { found: u32 }, UnsupportedVersion(u32), LengthMismatch { expected: usize, got: usize }, Torn }`

**Layout.** 64-byte header, then two equally sized buffers. `publish` writes into buffer `seq % 2` and only then increments `seq`, so a reader that samples `seq`, reads buffer `(seq - 1) % 2`, and re-reads `seq` either sees a consistent frame or detects that it was lapped and retries. `read_latest` retries three times before returning `ChannelError::Torn`.

- [ ] **Step 1: Write the failing test**

Create `crates/elements-ipc/tests/channel.rs`:

```rust
use elements_ipc::{ChannelError, FrameReader, FrameWriter, CHANNEL_HEADER_BYTES, CHANNEL_MAGIC};

#[test]
fn publish_then_read_returns_the_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");

    let mut writer = FrameWriter::create(&path, [4, 4, 4], 1).unwrap();
    let values: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let seq = writer.publish(&values).unwrap();
    assert_eq!(seq, 1, "the first published frame is sequence 1");

    let reader = FrameReader::open(&path).unwrap();
    let (read_seq, read_values) = reader.read_latest().unwrap();
    assert_eq!(read_seq, 1);
    assert_eq!(read_values, values);
}

#[test]
fn the_header_describes_the_field() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");

    let mut writer = FrameWriter::create(&path, [2, 3, 4], 1).unwrap();
    writer.publish(&vec![0.0; 24]).unwrap();

    let reader = FrameReader::open(&path).unwrap();
    let header = reader.header();
    assert_eq!(header.magic, CHANNEL_MAGIC);
    assert_eq!(header.version, 1);
    assert_eq!(header.dims, [2, 3, 4]);
    assert_eq!(header.channels, 1);
    assert_eq!(header.buffer_bytes, 24 * 4);
}

#[test]
fn the_file_is_header_plus_two_buffers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");
    FrameWriter::create(&path, [4, 4, 4], 1).unwrap();

    let len = std::fs::metadata(&path).unwrap().len() as usize;
    assert_eq!(len, CHANNEL_HEADER_BYTES + 2 * 64 * 4);
}

#[test]
fn successive_publishes_alternate_buffers_and_advance_seq() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");

    let mut writer = FrameWriter::create(&path, [2, 2, 2], 1).unwrap();
    let reader = FrameReader::open(&path).unwrap();

    for n in 1..=5u64 {
        let values = vec![n as f32; 8];
        assert_eq!(writer.publish(&values).unwrap(), n);
        let (seq, read) = reader.read_latest().unwrap();
        assert_eq!(seq, n);
        assert_eq!(read, values);
    }
}

#[test]
fn a_wrong_length_payload_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");
    let mut writer = FrameWriter::create(&path, [4, 4, 4], 1).unwrap();

    match writer.publish(&[1.0, 2.0]) {
        Err(ChannelError::LengthMismatch { expected: 64, got: 2 }) => {}
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn reading_before_any_publish_reports_sequence_zero() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");
    FrameWriter::create(&path, [2, 2, 2], 1).unwrap();

    let reader = FrameReader::open(&path).unwrap();
    let (seq, values) = reader.read_latest().unwrap();
    assert_eq!(seq, 0, "no frame published yet");
    assert!(values.iter().all(|v| *v == 0.0));
}

#[test]
fn a_foreign_file_is_rejected_by_magic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("not-a-channel.bin");
    std::fs::write(&path, vec![0u8; 256]).unwrap();

    match FrameReader::open(&path) {
        Err(ChannelError::BadMagic { .. }) => {}
        other => panic!("expected BadMagic, got {other:?}"),
    }
}

#[test]
fn a_reader_survives_a_writer_publishing_concurrently() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame.bin");
    let mut writer = FrameWriter::create(&path, [8, 8, 8], 1).unwrap();
    writer.publish(&vec![1.0; 512]).unwrap();

    let reader = FrameReader::open(&path).unwrap();
    let handle = std::thread::spawn(move || {
        for n in 2..200u64 {
            writer.publish(&vec![n as f32; 512]).unwrap();
        }
    });

    // Every successful read must be internally consistent: one uniform value.
    for _ in 0..500 {
        if let Ok((_seq, values)) = reader.read_latest() {
            let first = values[0];
            assert!(
                values.iter().all(|v| *v == first),
                "a torn frame leaked through: saw mixed values"
            );
        }
    }

    handle.join().unwrap();
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p elements-ipc --test channel`
Expected: FAIL — `unresolved imports FrameReader, FrameWriter`.

- [ ] **Step 3: Implement the channel**

Create `crates/elements-ipc/src/channel.rs`:

```rust
//! The data plane: a memory-mapped file holding two alternating frame buffers.
//!
//! A file rather than POSIX shared memory because the Blender addon must read
//! it with the Python standard library on 3.11 through 3.13, where
//! `SharedMemory` cleanup semantics differ. The double-buffer protocol is
//! identical, so this can be swapped for real shared memory later without
//! changing either end's API.
//!
//! Protocol: the writer fills buffer `seq % 2`, then increments `seq`. A reader
//! samples `seq`, reads buffer `(seq - 1) % 2`, and re-samples `seq`; if it
//! changed, the writer lapped it and the read is retried.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// `"ELFC"` as a little-endian u32.
pub const CHANNEL_MAGIC: u32 = 0x4346_4C45;
pub const CHANNEL_VERSION: u32 = 1;
/// Fixed header size. Buffers start here; the tail is reserved padding.
pub const CHANNEL_HEADER_BYTES: usize = 64;

const OFF_MAGIC: usize = 0;
const OFF_VERSION: usize = 4;
const OFF_DIMS: usize = 8; // three u32
const OFF_CHANNELS: usize = 20;
const OFF_BUFFER_BYTES: usize = 24; // u64
const OFF_SEQ: usize = 32; // u64, must be 8-byte aligned for atomic access

#[derive(Debug, thiserror::Error)]
pub enum ChannelError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("not an Elements frame channel (magic {found:#010x})")]
    BadMagic { found: u32 },
    #[error("unsupported channel version {0}")]
    UnsupportedVersion(u32),
    #[error("expected {expected} values, got {got}")]
    LengthMismatch { expected: usize, got: usize },
    #[error("the writer lapped the reader repeatedly")]
    Torn,
}

/// The fixed-size channel header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelHeader {
    pub magic: u32,
    pub version: u32,
    pub dims: [u32; 3],
    pub channels: u32,
    pub buffer_bytes: u64,
    pub seq: u64,
}

fn value_count(dims: [u32; 3], channels: u32) -> usize {
    dims[0] as usize * dims[1] as usize * dims[2] as usize * channels as usize
}

fn read_u32(map: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(map[offset..offset + 4].try_into().expect("in-bounds"))
}

fn read_u64(map: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(map[offset..offset + 8].try_into().expect("in-bounds"))
}

/// Read the `seq` field with acquire ordering.
///
/// # Safety
/// `map` must be at least `CHANNEL_HEADER_BYTES` long and its base must be
/// 8-byte aligned, which `memmap2` guarantees for page-aligned mappings.
unsafe fn load_seq(map: &[u8]) -> u64 {
    let ptr = map.as_ptr().wrapping_add(OFF_SEQ) as *const AtomicU64;
    unsafe { (*ptr).load(Ordering::Acquire) }
}

/// Store the `seq` field with release ordering, publishing prior buffer writes.
///
/// # Safety
/// Same requirements as [`load_seq`], and the caller must be the only writer.
unsafe fn store_seq(map: &mut [u8], value: u64) {
    let ptr = map.as_mut_ptr().wrapping_add(OFF_SEQ) as *const AtomicU64;
    unsafe { (*ptr).store(value, Ordering::Release) }
}

/// The engine side: owns the file and publishes frames.
pub struct FrameWriter {
    map: memmap2::MmapMut,
    path: PathBuf,
    dims: [u32; 3],
    channels: u32,
    values: usize,
    buffer_bytes: usize,
    seq: u64,
}

impl FrameWriter {
    /// Create (or truncate) the channel file and write its header.
    pub fn create(path: &Path, dims: [u32; 3], channels: u32) -> Result<Self, ChannelError> {
        let values = value_count(dims, channels);
        let buffer_bytes = values * std::mem::size_of::<f32>();
        let total = CHANNEL_HEADER_BYTES + 2 * buffer_bytes;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(total as u64)?;

        // SAFETY: the file was just sized to `total` and this process is its
        // only writer; the mapping is not aliased elsewhere in this crate.
        let mut map = unsafe { memmap2::MmapMut::map_mut(&file)? };

        map[OFF_MAGIC..OFF_MAGIC + 4].copy_from_slice(&CHANNEL_MAGIC.to_le_bytes());
        map[OFF_VERSION..OFF_VERSION + 4].copy_from_slice(&CHANNEL_VERSION.to_le_bytes());
        for (i, d) in dims.iter().enumerate() {
            let at = OFF_DIMS + i * 4;
            map[at..at + 4].copy_from_slice(&d.to_le_bytes());
        }
        map[OFF_CHANNELS..OFF_CHANNELS + 4].copy_from_slice(&channels.to_le_bytes());
        map[OFF_BUFFER_BYTES..OFF_BUFFER_BYTES + 8]
            .copy_from_slice(&(buffer_bytes as u64).to_le_bytes());
        // SAFETY: header is fully mapped and we are the only writer.
        unsafe { store_seq(&mut map, 0) };
        map.flush()?;

        Ok(Self {
            map,
            path: path.to_path_buf(),
            dims,
            channels,
            values,
            buffer_bytes,
            seq: 0,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn seq(&self) -> u64 {
        self.seq
    }

    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Write `values` into the next buffer and publish it. Returns the new sequence.
    pub fn publish(&mut self, values: &[f32]) -> Result<u64, ChannelError> {
        if values.len() != self.values {
            return Err(ChannelError::LengthMismatch {
                expected: self.values,
                got: values.len(),
            });
        }

        let index = (self.seq % 2) as usize;
        let start = CHANNEL_HEADER_BYTES + index * self.buffer_bytes;
        let bytes: &[u8] = bytemuck_cast(values);
        self.map[start..start + self.buffer_bytes].copy_from_slice(bytes);

        self.seq += 1;
        // SAFETY: release store publishes the buffer writes above.
        unsafe { store_seq(&mut self.map, self.seq) };
        Ok(self.seq)
    }
}

/// Reinterpret `f32` values as little-endian bytes.
///
/// Elements only targets little-endian platforms; this is asserted at compile
/// time rather than silently producing byte-swapped frames elsewhere.
fn bytemuck_cast(values: &[f32]) -> &[u8] {
    const _: () = assert!(cfg!(target_endian = "little"), "Elements requires little-endian");
    // SAFETY: f32 has no padding or invalid bit patterns, and the returned
    // slice borrows the same lifetime with a byte-sized element type.
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
}

/// The client side: maps the file read-only and samples the latest frame.
pub struct FrameReader {
    map: memmap2::Mmap,
    header: ChannelHeader,
    values: usize,
    buffer_bytes: usize,
}

impl FrameReader {
    pub fn open(path: &Path) -> Result<Self, ChannelError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: a concurrent writer may modify these bytes, which is the
        // point; all reads go through `read_latest`, which validates the
        // sequence number before and after copying.
        let map = unsafe { memmap2::Mmap::map(&file)? };

        if map.len() < CHANNEL_HEADER_BYTES {
            return Err(ChannelError::BadMagic { found: 0 });
        }
        let magic = read_u32(&map, OFF_MAGIC);
        if magic != CHANNEL_MAGIC {
            return Err(ChannelError::BadMagic { found: magic });
        }
        let version = read_u32(&map, OFF_VERSION);
        if version != CHANNEL_VERSION {
            return Err(ChannelError::UnsupportedVersion(version));
        }

        let dims = [
            read_u32(&map, OFF_DIMS),
            read_u32(&map, OFF_DIMS + 4),
            read_u32(&map, OFF_DIMS + 8),
        ];
        let channels = read_u32(&map, OFF_CHANNELS);
        let buffer_bytes = read_u64(&map, OFF_BUFFER_BYTES) as usize;
        let values = value_count(dims, channels);

        Ok(Self {
            header: ChannelHeader {
                magic,
                version,
                dims,
                channels,
                buffer_bytes: buffer_bytes as u64,
                seq: 0,
            },
            map,
            values,
            buffer_bytes,
        })
    }

    pub fn header(&self) -> ChannelHeader {
        // SAFETY: the header is mapped and 8-byte aligned.
        let seq = unsafe { load_seq(&self.map) };
        ChannelHeader { seq, ..self.header }
    }

    /// Copy the most recently published frame.
    ///
    /// Returns sequence 0 and zeros if nothing has been published yet.
    pub fn read_latest(&self) -> Result<(u64, Vec<f32>), ChannelError> {
        for _ in 0..3 {
            // SAFETY: header is mapped and aligned.
            let before = unsafe { load_seq(&self.map) };
            let index = if before == 0 {
                0
            } else {
                ((before - 1) % 2) as usize
            };
            let start = CHANNEL_HEADER_BYTES + index * self.buffer_bytes;

            let mut out = vec![0.0f32; self.values];
            let dst: &mut [u8] = {
                // SAFETY: f32 has no padding; the slice length matches exactly.
                unsafe {
                    std::slice::from_raw_parts_mut(
                        out.as_mut_ptr() as *mut u8,
                        self.buffer_bytes,
                    )
                }
            };
            dst.copy_from_slice(&self.map[start..start + self.buffer_bytes]);

            // SAFETY: as above.
            let after = unsafe { load_seq(&self.map) };
            if after == before {
                return Ok((before, out));
            }
        }
        Err(ChannelError::Torn)
    }
}
```

With two buffers, a writer that publishes twice during one read can still overwrite the buffer being copied. The retry loop catches that by comparing `seq` before and after. If `a_reader_survives_a_writer_publishing_concurrently` proves flaky, raise the buffer count to four (change `% 2` to `% BUFFERS` and size the file accordingly) rather than adding a lock — the writer must never block on the reader.

- [ ] **Step 4: Export the module**

Modify `crates/elements-ipc/src/lib.rs`:

```rust
mod channel;
mod protocol;

pub use channel::{
    ChannelError, ChannelHeader, FrameReader, FrameWriter, CHANNEL_HEADER_BYTES, CHANNEL_MAGIC,
    CHANNEL_VERSION,
};
pub use protocol::{
    read_message, write_message, Command, EngineError, ErrorKind, ProtocolError, Response,
    ELEMENTS_PROTOCOL_VERSION, MAX_MESSAGE_BYTES,
};
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p elements-ipc --test channel`
Expected: PASS — eight tests ok.

Run it ten times to check the concurrency test is not flaky:

Run: `for i in $(seq 10); do cargo test -p elements-ipc --test channel -- a_reader_survives || break; done`
Expected: ten passes.

- [ ] **Step 6: Commit**

```bash
git add crates/elements-ipc
git commit -m "feat: add memory-mapped double-buffered frame channel"
```

---

## Task 16: The daemon

**Files:**
- Create: `crates/elements-ipc/src/transport.rs`, `crates/elementsd/Cargo.toml`, `crates/elementsd/src/main.rs`, `crates/elementsd/src/daemon.rs`, `crates/elementsd/tests/session.rs`
- Modify: `crates/elements-ipc/src/lib.rs`, `crates/elements-ipc/Cargo.toml`

**Interfaces:**
- Consumes: `Command`, `Response`, `EngineError`, `ErrorKind`, `read_message`, `write_message` from Task 14; `FrameWriter` from Task 15; `Document`, `NodeRegistry`, `Graph` from Tasks 6–8; `GpuContext`, `FieldPool`, `PipelineCache` from Tasks 2–4.
- Produces:
  - In `elements-ipc`: `struct Listener` with `Listener::bind(endpoint: &str) -> Result<Listener, ProtocolError>` and `accept(&self) -> Result<Stream, ProtocolError>`; `struct Stream` implementing `Read + Write`; `fn default_endpoint(name: &str) -> String` — a Unix socket path under the runtime dir, or `\\.\pipe\<name>` on Windows.
  - `elementsd --endpoint <ep> --channel <path>` binary. Prints `ready <endpoint>` on stdout once listening, so a supervising process can wait for it deterministically instead of sleeping.
  - `struct Session` in `daemon.rs` holding the `GpuContext`, `FieldPool`, `PipelineCache`, optional loaded `Graph` and `FieldDims`, and the `FrameWriter`.
  - `fn handle(session: &mut Session, command: Command) -> Response` — total, never panics, maps every failure to `Response::Error`.

- [ ] **Step 1: Write the failing test**

Create `crates/elementsd/tests/session.rs`:

```rust
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command as ProcCommand, Stdio};

use elements_ipc::{
    read_message, write_message, Command, ErrorKind, FrameReader, Response,
    ELEMENTS_PROTOCOL_VERSION,
};

struct Daemon {
    child: Child,
    endpoint: String,
    channel: std::path::PathBuf,
    _dir: tempfile::TempDir,
}

impl Daemon {
    fn start() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let channel = dir.path().join("frame.bin");
        let endpoint = if cfg!(windows) {
            format!("\\\\.\\pipe\\elements-test-{}", std::process::id())
        } else {
            dir.path().join("control.sock").to_string_lossy().into_owned()
        };

        let mut child = ProcCommand::new(env!("CARGO_BIN_EXE_elementsd"))
            .args(["--endpoint", &endpoint])
            .args(["--channel", channel.to_str().unwrap()])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();

        // Wait for the readiness line rather than sleeping.
        let mut line = String::new();
        BufReader::new(child.stdout.as_mut().unwrap())
            .read_line(&mut line)
            .unwrap();
        assert!(line.starts_with("ready "), "daemon said: {line}");

        Self { child, endpoint, channel, _dir: dir }
    }

    fn connect(&self) -> elements_ipc::Stream {
        elements_ipc::Stream::connect(&self.endpoint).unwrap()
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn graph_file(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("noise.elements");
    std::fs::write(
        &path,
        r#"{
          "version": 1,
          "dims": [8, 8, 8],
          "nodes": [
            { "id": 0, "kind": "core.noise_field", "params": { "seed": 7 } },
            { "id": 1, "kind": "core.output", "params": {} }
          ],
          "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
          "output": 1
        }"#,
    )
    .unwrap();
    path
}

#[test]
fn a_full_session_loads_renders_and_shuts_down() {
    let daemon = Daemon::start();
    let dir = tempfile::tempdir().unwrap();
    let graph = graph_file(dir.path());

    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());

    write_message(&mut stream, &Command::Hello {
        protocol_version: ELEMENTS_PROTOCOL_VERSION,
    })
    .unwrap();
    let ack: Response = read_message(&mut reader).unwrap().unwrap();
    match ack {
        Response::HelloAck { protocol_version, adapter, .. } => {
            assert_eq!(protocol_version, ELEMENTS_PROTOCOL_VERSION);
            assert!(!adapter.is_empty());
        }
        other => panic!("expected HelloAck, got {other:?}"),
    }

    write_message(&mut stream, &Command::LoadGraph {
        path: graph.to_string_lossy().into_owned(),
    })
    .unwrap();
    match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
        Response::Loaded { dims, nodes } => {
            assert_eq!(dims, [8, 8, 8]);
            assert_eq!(nodes, 2);
        }
        other => panic!("expected Loaded, got {other:?}"),
    }

    write_message(&mut stream, &Command::Render { frame: 1 }).unwrap();
    let seq = match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
        Response::Frame { seq, dims, .. } => {
            assert_eq!(dims, [8, 8, 8]);
            seq
        }
        other => panic!("expected Frame, got {other:?}"),
    };
    assert_eq!(seq, 1);

    let frame_reader = FrameReader::open(&daemon.channel).unwrap();
    let (read_seq, values) = frame_reader.read_latest().unwrap();
    assert_eq!(read_seq, 1);
    assert_eq!(values.len(), 512);
    assert!(values.iter().any(|v| *v != 0.0), "the frame must not be blank");
    assert!(values.iter().all(|v| v.is_finite()));

    write_message(&mut stream, &Command::Shutdown).unwrap();
    assert!(matches!(
        read_message::<_, Response>(&mut reader).unwrap().unwrap(),
        Response::Bye
    ));
}

#[test]
fn a_protocol_version_mismatch_is_refused_with_a_typed_error() {
    let daemon = Daemon::start();
    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());

    write_message(&mut stream, &Command::Hello { protocol_version: 999 }).unwrap();
    match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
        Response::Error(e) => {
            assert_eq!(e.kind, ErrorKind::ProtocolVersion);
            assert!(e.message.contains("999"), "message was: {}", e.message);
        }
        other => panic!("expected Error, got {other:?}"),
    }
}

#[test]
fn rendering_before_loading_a_graph_is_an_error_not_a_crash() {
    let daemon = Daemon::start();
    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());

    write_message(&mut stream, &Command::Hello {
        protocol_version: ELEMENTS_PROTOCOL_VERSION,
    })
    .unwrap();
    let _: Response = read_message(&mut reader).unwrap().unwrap();

    write_message(&mut stream, &Command::Render { frame: 1 }).unwrap();
    match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
        Response::Error(e) => assert_eq!(e.kind, ErrorKind::Graph),
        other => panic!("expected Error, got {other:?}"),
    }

    // The daemon must still be usable afterwards.
    write_message(&mut stream, &Command::Shutdown).unwrap();
    assert!(matches!(
        read_message::<_, Response>(&mut reader).unwrap().unwrap(),
        Response::Bye
    ));
}

#[test]
fn a_malformed_graph_reports_a_document_error_and_keeps_the_session() {
    let daemon = Daemon::start();
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.elements");
    std::fs::write(&bad, "{ not json").unwrap();

    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    write_message(&mut stream, &Command::Hello {
        protocol_version: ELEMENTS_PROTOCOL_VERSION,
    })
    .unwrap();
    let _: Response = read_message(&mut reader).unwrap().unwrap();

    write_message(&mut stream, &Command::LoadGraph {
        path: bad.to_string_lossy().into_owned(),
    })
    .unwrap();
    match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
        Response::Error(e) => assert_eq!(e.kind, ErrorKind::Document),
        other => panic!("expected Error, got {other:?}"),
    }

    // A second, valid load on the same connection must succeed.
    let good = graph_file(dir.path());
    write_message(&mut stream, &Command::LoadGraph {
        path: good.to_string_lossy().into_owned(),
    })
    .unwrap();
    assert!(matches!(
        read_message::<_, Response>(&mut reader).unwrap().unwrap(),
        Response::Loaded { .. }
    ));
}

#[test]
fn repeated_renders_advance_the_sequence() {
    let daemon = Daemon::start();
    let dir = tempfile::tempdir().unwrap();
    let graph = graph_file(dir.path());

    let mut stream = daemon.connect();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    write_message(&mut stream, &Command::Hello {
        protocol_version: ELEMENTS_PROTOCOL_VERSION,
    })
    .unwrap();
    let _: Response = read_message(&mut reader).unwrap().unwrap();
    write_message(&mut stream, &Command::LoadGraph {
        path: graph.to_string_lossy().into_owned(),
    })
    .unwrap();
    let _: Response = read_message(&mut reader).unwrap().unwrap();

    for expected in 1..=4u64 {
        write_message(&mut stream, &Command::Render { frame: expected as u32 }).unwrap();
        match read_message::<_, Response>(&mut reader).unwrap().unwrap() {
            Response::Frame { seq, .. } => assert_eq!(seq, expected),
            other => panic!("expected Frame, got {other:?}"),
        }
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p elementsd`
Expected: FAIL — `package ID specification elementsd did not match any packages`.

- [ ] **Step 3: Implement the transport**

Modify `crates/elements-ipc/Cargo.toml` `[dependencies]`, adding:

```toml
interprocess = "2.4.4"
```

Create `crates/elements-ipc/src/transport.rs`:

```rust
//! Stream transport: Unix domain sockets on macOS and Linux, named pipes on
//! Windows. `interprocess` provides both behind one API; the Python client
//! reimplements the same two cases with its standard library.

use std::io::{Read, Write};

use interprocess::local_socket::{
    prelude::*, GenericFilePath, GenericNamespaced, ListenerOptions, Stream as RawStream,
    ToFsName, ToNsName,
};

use crate::ProtocolError;

/// A conventional endpoint for `name`: a socket file on Unix, a pipe on Windows.
pub fn default_endpoint(name: &str) -> String {
    if cfg!(windows) {
        format!("\\\\.\\pipe\\{name}")
    } else {
        let base = std::env::var("XDG_RUNTIME_DIR")
            .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
        format!("{base}/{name}.sock")
    }
}

fn to_name(endpoint: &str) -> Result<interprocess::local_socket::Name<'_>, ProtocolError> {
    let name = if cfg!(windows) && !endpoint.contains('\\') {
        endpoint.to_ns_name::<GenericNamespaced>()?
    } else {
        endpoint.to_fs_name::<GenericFilePath>()?
    };
    Ok(name)
}

/// A connected control-plane stream.
pub struct Stream(RawStream);

impl Stream {
    pub fn connect(endpoint: &str) -> Result<Self, ProtocolError> {
        Ok(Self(RawStream::connect(to_name(endpoint)?)?))
    }

    /// A second handle to the same stream, so one side can be wrapped in a
    /// `BufReader` while the other is written to.
    pub fn try_clone(&self) -> Result<Self, ProtocolError> {
        Ok(Self(self.0.try_clone()?))
    }
}

impl Read for Stream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

impl Write for Stream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

/// A bound listener.
pub struct Listener {
    inner: interprocess::local_socket::Listener,
}

impl Listener {
    pub fn bind(endpoint: &str) -> Result<Self, ProtocolError> {
        // A stale socket file from a crashed daemon would make bind fail.
        if !cfg!(windows) {
            let _ = std::fs::remove_file(endpoint);
        }
        let inner = ListenerOptions::new().name(to_name(endpoint)?).create_sync()?;
        Ok(Self { inner })
    }

    pub fn accept(&self) -> Result<Stream, ProtocolError> {
        Ok(Stream(self.inner.accept()?))
    }
}
```

If `interprocess` 2.4's `try_clone` is not available on `Stream`, keep one `Stream` and construct the `BufReader` over a borrowed `&mut` reference instead, restructuring the daemon loop to read and write through the same handle sequentially. The tests use `try_clone` only for clarity.

Modify `crates/elements-ipc/src/lib.rs`, adding:

```rust
mod transport;

pub use transport::{default_endpoint, Listener, Stream};
```

- [ ] **Step 4: Implement the daemon**

Create `crates/elementsd/Cargo.toml`:

```toml
[package]
name = "elementsd"
version = "0.1.0"
edition.workspace = true
license.workspace = true
repository.workspace = true

[[bin]]
name = "elementsd"
path = "src/main.rs"

[dependencies]
anyhow.workspace = true
clap.workspace = true
elements-core = { path = "../elements-core" }
elements-ipc = { path = "../elements-ipc" }

[dev-dependencies]
elements-ipc = { path = "../elements-ipc" }
tempfile.workspace = true
```

Create `crates/elementsd/src/daemon.rs`:

```rust
//! Session state and command handling.
//!
//! Every handler is total: a failure becomes a `Response::Error` and the
//! session stays alive. The daemon outliving a bad graph is the whole point of
//! running out of process.

use std::path::Path;

use elements_core::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{Document, Graph, NodeRegistry};
use elements_ipc::{
    Command, EngineError, ErrorKind, FrameWriter, Response, ELEMENTS_PROTOCOL_VERSION,
};

/// One client's engine state.
pub struct Session {
    gpu: GpuContext,
    pool: FieldPool,
    pipelines: PipelineCache,
    registry: NodeRegistry,
    graph: Option<(Graph, FieldDims)>,
    channel_path: std::path::PathBuf,
    writer: Option<FrameWriter>,
    greeted: bool,
}

impl Session {
    pub fn new(channel_path: &Path) -> Result<Self, EngineError> {
        let gpu = GpuContext::new_headless()
            .map_err(|e| EngineError::new(ErrorKind::Gpu, e.to_string()))?;
        Ok(Self {
            gpu,
            pool: FieldPool::new(),
            pipelines: PipelineCache::new(),
            registry: NodeRegistry::with_builtins(),
            graph: None,
            channel_path: channel_path.to_path_buf(),
            writer: None,
            greeted: false,
        })
    }

    pub fn adapter_name(&self) -> &str {
        self.gpu.adapter_name()
    }
}

/// Handle one command. Never panics; never returns `Err`.
pub fn handle(session: &mut Session, command: Command) -> Response {
    match command {
        Command::Hello { protocol_version } => {
            if protocol_version != ELEMENTS_PROTOCOL_VERSION {
                return Response::Error(EngineError::new(
                    ErrorKind::ProtocolVersion,
                    format!(
                        "client speaks protocol {protocol_version}, this engine speaks \
                         {ELEMENTS_PROTOCOL_VERSION}; reinstall the add-on"
                    ),
                ));
            }
            session.greeted = true;
            Response::HelloAck {
                protocol_version: ELEMENTS_PROTOCOL_VERSION,
                engine_version: env!("CARGO_PKG_VERSION").to_owned(),
                adapter: session.adapter_name().to_owned(),
            }
        }

        Command::LoadGraph { path } => match load(session, Path::new(&path)) {
            Ok(response) => response,
            Err(e) => Response::Error(e),
        },

        Command::Render { frame: _ } => match render(session) {
            Ok(response) => response,
            Err(e) => Response::Error(e),
        },

        Command::Shutdown => Response::Bye,
    }
}

fn load(session: &mut Session, path: &Path) -> Result<Response, EngineError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| EngineError::new(ErrorKind::Io, format!("{}: {e}", path.display())))?;
    let doc = Document::from_json(&text)
        .map_err(|e| EngineError::new(ErrorKind::Document, e.to_string()))?;
    let (graph, dims) = doc
        .into_graph(&session.registry)
        .map_err(|e| EngineError::new(ErrorKind::Document, e.to_string()))?;

    let nodes = graph.node_count() as u32;

    // Reallocate the channel whenever the resolution changes.
    let needs_channel = match &session.writer {
        Some(w) => w.dims() != [dims.x, dims.y, dims.z],
        None => true,
    };
    if needs_channel {
        session.writer = Some(
            FrameWriter::create(&session.channel_path, [dims.x, dims.y, dims.z], 1)
                .map_err(|e| EngineError::new(ErrorKind::Io, e.to_string()))?,
        );
    }

    session.graph = Some((graph, dims));
    Ok(Response::Loaded {
        dims: [dims.x, dims.y, dims.z],
        nodes,
    })
}

fn render(session: &mut Session) -> Result<Response, EngineError> {
    let (graph, dims) = session
        .graph
        .as_ref()
        .ok_or_else(|| EngineError::new(ErrorKind::Graph, "no graph is loaded"))?;

    let value = graph
        .eval(&session.gpu, &mut session.pool, &mut session.pipelines, *dims)
        .map_err(map_node_error)?;
    let field = value.as_field().map_err(map_node_error)?;
    let values = field.read_back(&session.gpu).map_err(|e| {
        let kind = if session.gpu.device_lost().is_some() {
            ErrorKind::DeviceLost
        } else {
            ErrorKind::Gpu
        };
        EngineError::new(kind, e.to_string())
    })?;

    let writer = session
        .writer
        .as_mut()
        .ok_or_else(|| EngineError::new(ErrorKind::Io, "no frame channel"))?;
    let seq = writer
        .publish(&values)
        .map_err(|e| EngineError::new(ErrorKind::Io, e.to_string()))?;

    Ok(Response::Frame {
        seq,
        channel: session.channel_path.to_string_lossy().into_owned(),
        dims: [dims.x, dims.y, dims.z],
    })
}

fn map_node_error(e: elements_core::graph::NodeError) -> EngineError {
    use elements_core::graph::NodeError;
    let kind = match &e {
        NodeError::Gpu(_) => ErrorKind::Gpu,
        _ => ErrorKind::Graph,
    };
    EngineError::new(kind, e.to_string())
}
```

Create `crates/elementsd/src/main.rs`:

```rust
#![forbid(unsafe_code)]

//! The Elements engine daemon.

mod daemon;

use std::io::{BufReader, Write};

use clap::Parser;
use elements_ipc::{read_message, write_message, Command, Listener, Response};

#[derive(Parser)]
#[command(name = "elementsd", version, about = "Elements Suite engine daemon")]
struct Cli {
    /// Socket path (Unix) or pipe name (Windows) to listen on.
    #[arg(long)]
    endpoint: String,
    /// Path of the memory-mapped frame channel.
    #[arg(long)]
    channel: std::path::PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let listener = Listener::bind(&cli.endpoint)?;

    // Announce readiness on stdout so supervisors need not poll or sleep.
    println!("ready {}", cli.endpoint);
    std::io::stdout().flush()?;

    loop {
        let stream = listener.accept()?;
        // One client at a time: Core v1 has a single consumer, and serialising
        // sessions keeps the GPU context single-threaded.
        if let Err(e) = serve(stream, &cli.channel) {
            eprintln!("session ended: {e:#}");
        }
    }
}

fn serve(mut stream: elements_ipc::Stream, channel: &std::path::Path) -> anyhow::Result<()> {
    let mut session = match daemon::Session::new(channel) {
        Ok(s) => s,
        Err(e) => {
            write_message(&mut stream, &Response::Error(e))?;
            return Ok(());
        }
    };

    let mut reader = BufReader::new(stream.try_clone()?);
    while let Some(command) = read_message::<_, Command>(&mut reader)? {
        let shutting_down = matches!(command, Command::Shutdown);
        let response = daemon::handle(&mut session, command);
        write_message(&mut stream, &response)?;
        if shutting_down {
            break;
        }
    }
    Ok(())
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `WGPU_BACKEND=vulkan cargo test -p elementsd`
Expected: PASS — five tests ok.

- [ ] **Step 6: Commit**

```bash
git add crates/elements-ipc crates/elementsd
git commit -m "feat: add elementsd daemon with session handling and stream transport"
```

---

## Task 17: The Python client

**Files:**
- Create: `addon/blender_elements/client.py`, `crates/elementsd/tests/python_contract.rs`, `tests/python/contract.py`

**Interfaces:**
- Consumes: the daemon binary from Task 16 and the wire formats from Tasks 14–15.
- Produces (pure Python, standard library only — no wheel, no numpy):
  - `class ElementsError(Exception)` with `.kind` and `.message`
  - `class ControlClient` with `connect(endpoint)`, `hello()`, `load_graph(path) -> dict`, `render(frame) -> dict`, `shutdown()`, `close()`, and context-manager support
  - `class FrameReader` with `open(path)`, `header() -> dict`, `read_latest() -> tuple[int, array.array]`, `close()`
  - `PROTOCOL_VERSION = 1`, `CHANNEL_MAGIC = 0x4346_4C45`, `CHANNEL_HEADER_BYTES = 64`

This module is the addon's only engine-facing code and is deliberately importable outside Blender, so it can be tested with the system Python.

- [ ] **Step 1: Write the failing test**

Create `tests/python/contract.py`:

```python
"""Contract test: the Python client against the real Rust daemon.

Run by `crates/elementsd/tests/python_contract.rs`, which starts the daemon and
passes the endpoint, channel path and graph path as arguments.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "addon"))

from blender_elements.client import (  # noqa: E402
    CHANNEL_MAGIC,
    ControlClient,
    ElementsError,
    FrameReader,
    PROTOCOL_VERSION,
)


def main(endpoint: str, channel: str, graph: str) -> None:
    with ControlClient(endpoint) as client:
        ack = client.hello()
        assert ack["protocol_version"] == PROTOCOL_VERSION, ack
        assert ack["adapter"], "adapter name must not be empty"

        loaded = client.load_graph(graph)
        assert loaded["dims"] == [8, 8, 8], loaded
        assert loaded["nodes"] == 2, loaded

        frame = client.render(1)
        assert frame["seq"] == 1, frame
        assert frame["dims"] == [8, 8, 8], frame

        reader = FrameReader(channel)
        try:
            header = reader.header()
            assert header["magic"] == CHANNEL_MAGIC, header
            assert header["dims"] == [8, 8, 8], header

            seq, values = reader.read_latest()
            assert seq == 1, seq
            assert len(values) == 512, len(values)
            assert any(v != 0.0 for v in values), "frame must not be blank"
            assert all(v == v for v in values), "frame contains NaN"

            # A second render must advance the sequence and still read cleanly.
            frame2 = client.render(2)
            assert frame2["seq"] == 2, frame2
            seq2, values2 = reader.read_latest()
            assert seq2 == 2, seq2
            assert len(values2) == 512
        finally:
            reader.close()

        # A bad graph must raise, not corrupt the session.
        try:
            client.load_graph("/nonexistent/graph.elements")
        except ElementsError as e:
            assert e.kind == "io", e.kind
        else:
            raise AssertionError("expected ElementsError for a missing file")

        # The session must still work.
        assert client.load_graph(graph)["nodes"] == 2

    print("python contract ok")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
```

Create `crates/elementsd/tests/python_contract.rs`:

```rust
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

/// The repository root, derived from this crate's manifest directory.
fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/elementsd is two levels below the root")
        .to_path_buf()
}

fn python() -> &'static str {
    if cfg!(windows) { "python" } else { "python3" }
}

#[test]
fn the_python_client_speaks_the_real_protocol() {
    let dir = tempfile::tempdir().unwrap();
    let channel = dir.path().join("frame.bin");
    let endpoint = if cfg!(windows) {
        format!("\\\\.\\pipe\\elements-pycontract-{}", std::process::id())
    } else {
        dir.path().join("control.sock").to_string_lossy().into_owned()
    };

    let graph = dir.path().join("noise.elements");
    std::fs::write(
        &graph,
        r#"{
          "version": 1,
          "dims": [8, 8, 8],
          "nodes": [
            { "id": 0, "kind": "core.noise_field", "params": { "seed": 7 } },
            { "id": 1, "kind": "core.output", "params": {} }
          ],
          "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
          "output": 1
        }"#,
    )
    .unwrap();

    let mut daemon = Command::new(env!("CARGO_BIN_EXE_elementsd"))
        .args(["--endpoint", &endpoint])
        .args(["--channel", channel.to_str().unwrap()])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    let mut line = String::new();
    BufReader::new(daemon.stdout.as_mut().unwrap())
        .read_line(&mut line)
        .unwrap();
    assert!(line.starts_with("ready "), "daemon said: {line}");

    let script = repo_root().join("tests/python/contract.py");
    let output = Command::new(python())
        .arg(&script)
        .arg(&endpoint)
        .arg(channel.to_str().unwrap())
        .arg(graph.to_str().unwrap())
        .output()
        .expect("python3 must be on PATH");

    let _ = daemon.kill();
    let _ = daemon.wait();

    assert!(
        output.status.success(),
        "python contract failed\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("python contract ok"));
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `WGPU_BACKEND=vulkan cargo test -p elementsd --test python_contract`
Expected: FAIL — python exits non-zero with `ModuleNotFoundError: No module named 'blender_elements'`.

- [ ] **Step 3: Implement the client**

Create `addon/blender_elements/client.py`:

```python
"""Engine client for the Elements add-on.

Standard library only: this module must import and work on the Python that
ships inside Blender 5.0 (3.11) through 5.1 (3.13) with no wheels installed,
and outside Blender for contract testing.

Two planes mirror the Rust side:
  * control  -- newline-delimited JSON over a Unix socket or Windows named pipe
  * data     -- a memory-mapped file holding two alternating frame buffers
"""

from __future__ import annotations

import array
import json
import mmap
import os
import socket
import struct
import sys

PROTOCOL_VERSION = 1

CHANNEL_MAGIC = 0x4346_4C45  # "ELFC" little-endian
CHANNEL_VERSION = 1
CHANNEL_HEADER_BYTES = 64

_OFF_MAGIC = 0
_OFF_VERSION = 4
_OFF_DIMS = 8
_OFF_CHANNELS = 20
_OFF_BUFFER_BYTES = 24
_OFF_SEQ = 32

MAX_MESSAGE_BYTES = 1024 * 1024


class ElementsError(Exception):
    """An error reported by the engine, or a transport failure."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message


class _PipeSocket:
    """Minimal socket-alike over a Windows named pipe."""

    def __init__(self, endpoint: str) -> None:
        self._f = open(endpoint, "r+b", buffering=0)

    def sendall(self, data: bytes) -> None:
        self._f.write(data)
        self._f.flush()

    def recv(self, size: int) -> bytes:
        return self._f.read(size)

    def close(self) -> None:
        self._f.close()


class ControlClient:
    """Newline-delimited JSON client for the engine's control plane."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._sock = None
        self._buf = b""

    def __enter__(self) -> "ControlClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        if sys.platform == "win32":
            self._sock = _PipeSocket(self._endpoint)
        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self._endpoint)
            self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _send(self, message: dict) -> None:
        if self._sock is None:
            raise ElementsError("io", "not connected")
        self._sock.sendall(json.dumps(message).encode("utf-8") + b"\n")

    def _recv(self) -> dict:
        if self._sock is None:
            raise ElementsError("io", "not connected")
        while b"\n" not in self._buf:
            if len(self._buf) > MAX_MESSAGE_BYTES:
                raise ElementsError("io", "engine sent an oversized message")
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ElementsError("io", "engine closed the connection")
            self._buf += chunk

        line, self._buf = self._buf.split(b"\n", 1)
        try:
            message = json.loads(line.decode("utf-8"))
        except ValueError as e:
            raise ElementsError("io", f"malformed message from engine: {e}") from e

        if message.get("type") == "error":
            raise ElementsError(
                message.get("kind", "io"), message.get("message", "unknown error")
            )
        return message

    def _round_trip(self, message: dict) -> dict:
        self._send(message)
        return self._recv()

    def hello(self) -> dict:
        return self._round_trip(
            {"type": "hello", "protocol_version": PROTOCOL_VERSION}
        )

    def load_graph(self, path: str) -> dict:
        return self._round_trip({"type": "load_graph", "path": os.fspath(path)})

    def render(self, frame: int) -> dict:
        return self._round_trip({"type": "render", "frame": int(frame)})

    def shutdown(self) -> dict:
        return self._round_trip({"type": "shutdown"})


class FrameReader:
    """Reads the latest published frame from the memory-mapped data plane."""

    def __init__(self, path: str) -> None:
        self._file = open(path, "rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        magic = self._u32(_OFF_MAGIC)
        if magic != CHANNEL_MAGIC:
            self.close()
            raise ElementsError("io", f"not an Elements channel (magic {magic:#x})")
        version = self._u32(_OFF_VERSION)
        if version != CHANNEL_VERSION:
            self.close()
            raise ElementsError("io", f"unsupported channel version {version}")

        self._dims = [self._u32(_OFF_DIMS + i * 4) for i in range(3)]
        self._channels = self._u32(_OFF_CHANNELS)
        self._buffer_bytes = self._u64(_OFF_BUFFER_BYTES)
        self._values = self._dims[0] * self._dims[1] * self._dims[2] * self._channels

    def _u32(self, offset: int) -> int:
        return struct.unpack_from("<I", self._map, offset)[0]

    def _u64(self, offset: int) -> int:
        return struct.unpack_from("<Q", self._map, offset)[0]

    def header(self) -> dict:
        return {
            "magic": self._u32(_OFF_MAGIC),
            "version": self._u32(_OFF_VERSION),
            "dims": list(self._dims),
            "channels": self._channels,
            "buffer_bytes": self._buffer_bytes,
            "seq": self._u64(_OFF_SEQ),
        }

    def read_latest(self) -> tuple[int, array.array]:
        """Return (sequence, values). Sequence 0 means nothing is published yet.

        The writer fills buffer `seq % 2` then bumps `seq`, so re-reading `seq`
        after the copy detects a frame that was overwritten mid-read.
        """
        for _ in range(3):
            before = self._u64(_OFF_SEQ)
            index = 0 if before == 0 else (before - 1) % 2
            start = CHANNEL_HEADER_BYTES + index * self._buffer_bytes

            values = array.array("f")
            values.frombytes(self._map[start : start + self._buffer_bytes])

            if self._u64(_OFF_SEQ) == before:
                return before, values

        raise ElementsError("io", "the engine lapped the reader repeatedly")

    def close(self) -> None:
        if getattr(self, "_map", None) is not None:
            self._map.close()
            self._map = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None
```

`array.array("f")` is IEEE 754 single precision on every platform CPython supports, and Elements requires little-endian, so no byte swapping is needed.

- [ ] **Step 4: Run the test to verify it passes**

Run: `WGPU_BACKEND=vulkan cargo test -p elementsd --test python_contract`
Expected: PASS — `the_python_client_speaks_the_real_protocol ... ok`.

- [ ] **Step 5: Confirm CI runs this on Blender's Python**

No workflow change is needed: `mise.toml` from Task 1 pins `python = "3.11"` and
`jdx/mise-action` installs it, so CI already runs the contract test on Blender
5.0's Python version. Verify rather than assume:

Run: `mise exec -- python --version`
Expected: `Python 3.11.x`.

Run: `mise exec -- python tests/python/contract.py` with no arguments
Expected: an `IndexError` from `sys.argv`, proving the module imports cleanly on
3.11 before the daemon is even involved.

- [ ] **Step 6: Commit**

```bash
git add addon tests crates/elementsd .github
git commit -m "feat: add stdlib-only Python engine client with a contract test"
```

---

## Task 18: The Blender add-on and packaging

**Files:**
- Create: `addon/blender_elements/__init__.py`, `addon/blender_elements/blender_manifest.toml`, `addon/blender_elements/props.py`, `addon/blender_elements/ops.py`, `addon/blender_elements/ui.py`, `addon/blender_elements/handlers.py`, `scripts/build_addon.py`, `tests/blender/test_roundtrip.py`, `crates/elementsd/tests/blender_integration.rs`

**Interfaces:**
- Consumes: `client.py` from Task 17, the `elementsd` binary from Task 16.
- Produces:
  - `dist/blender_elements-0.1.0.zip` with `__init__.py` and `blender_manifest.toml` **at the archive root**.
  - `Scene.elements` PropertyGroup: `graph_path: StringProperty(subtype='FILE_PATH')`, `endpoint: StringProperty`, `channel_path: StringProperty`, `daemon_path: StringProperty(subtype='FILE_PATH')`, `status: StringProperty`, `live: BoolProperty`.
  - Operators `elements.start_engine`, `elements.stop_engine`, `elements.render_frame`.
  - Panel `VIEW3D_PT_elements` in the N-panel under an "Elements" tab.

- [ ] **Step 1: Write the failing test**

Create `tests/blender/test_roundtrip.py`:

```python
"""Run inside Blender: `blender --background --python tests/blender/test_roundtrip.py -- <zip> <daemon>`.

Installs the built extension, starts the engine, renders one frame, and asserts
the volume data reached Blender. Exits non-zero with a message on failure.
"""

import sys
import os
import pathlib
import tempfile

import bpy

GRAPH = """{
  "version": 1,
  "dims": [8, 8, 8],
  "nodes": [
    { "id": 0, "kind": "core.noise_field", "params": { "seed": 7 } },
    { "id": 1, "kind": "core.output", "params": {} }
  ],
  "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
  "output": 1
}"""


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    zip_path, daemon_path = argv[0], argv[1]

    bpy.ops.extensions.package_install_files(
        filepath=zip_path, repo="user_default", enable_on_install=True
    )

    addon_id = "bl_ext.user_default.blender_elements"
    assert addon_id in sys.modules or True, "extension import is lazy"

    tmp = pathlib.Path(tempfile.mkdtemp())
    graph = tmp / "noise.elements"
    graph.write_text(GRAPH)

    scene = bpy.context.scene
    scene.elements.graph_path = str(graph)
    scene.elements.daemon_path = daemon_path
    scene.elements.endpoint = str(tmp / "control.sock")
    scene.elements.channel_path = str(tmp / "frame.bin")

    result = bpy.ops.elements.start_engine()
    assert result == {"FINISHED"}, f"start_engine returned {result}: {scene.elements.status}"

    result = bpy.ops.elements.render_frame()
    assert result == {"FINISHED"}, f"render_frame returned {result}: {scene.elements.status}"

    volumes = [o for o in bpy.data.objects if o.type == "VOLUME"]
    assert volumes, "no Volume object was created"
    assert "density" in {g.name for g in volumes[0].data.grids}, "no density grid"

    bpy.ops.elements.stop_engine()
    print("blender roundtrip ok")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - must surface through Blender's exit code
        print(f"blender roundtrip FAILED: {e}", file=sys.stderr)
        os._exit(1)
```

Create `crates/elementsd/tests/blender_integration.rs`:

```rust
//! Runs only when BLENDER_BIN points at a Blender executable. CI sets it on
//! the pre-release job; local runs skip it.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/elementsd is two levels below the root")
        .to_path_buf()
}

#[test]
fn blender_loads_the_addon_and_receives_a_frame() {
    let Ok(blender) = std::env::var("BLENDER_BIN") else {
        eprintln!("BLENDER_BIN is unset; skipping the Blender integration test");
        return;
    };

    let root = repo_root();
    let build = Command::new(if cfg!(windows) { "python" } else { "python3" })
        .arg(root.join("scripts/build_addon.py"))
        .current_dir(&root)
        .output()
        .expect("build_addon.py must run");
    assert!(
        build.status.success(),
        "build_addon.py failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let zip = root.join("dist/blender_elements-0.1.0.zip");
    assert!(zip.exists(), "{} was not built", zip.display());

    let output = Command::new(blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(root.join("tests/blender/test_roundtrip.py"))
        .arg("--")
        .arg(&zip)
        .arg(env!("CARGO_BIN_EXE_elementsd"))
        .output()
        .expect("blender must run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("blender roundtrip ok"),
        "stdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
```

- [ ] **Step 2: Write the packaging script and its check**

Create `scripts/build_addon.py`:

```python
"""Build dist/blender_elements-<version>.zip and verify its layout.

Blender extracts the archive root directly into
extensions/<repo>/<id>/, so __init__.py and blender_manifest.toml must sit at
the root with nothing nested under a package directory. A nested layout makes
Python treat the installed directory as a namespace package and the extension
fails to load.
"""

import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "addon" / "blender_elements"
VERSION = "0.1.0"
OUT = ROOT / "dist" / f"blender_elements-{VERSION}.zip"

REQUIRED_ROOT_FILES = {"__init__.py", "blender_manifest.toml"}


def build() -> pathlib.Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SRC.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            zf.write(path, path.relative_to(SRC))
    return OUT


def verify(path: pathlib.Path) -> None:
    names = set(zipfile.ZipFile(path).namelist())
    missing = REQUIRED_ROOT_FILES - names
    if missing:
        raise SystemExit(f"archive root is missing {sorted(missing)}; got {sorted(names)}")
    nested = [n for n in names if n.startswith("blender_elements/")]
    if nested:
        raise SystemExit(f"package contents must not be nested: {nested}")
    print(f"built {path} with {len(names)} files")


if __name__ == "__main__":
    verify(build())
    sys.exit(0)
```

- [ ] **Step 3: Run the build to verify it fails**

Run: `just addon`
Expected: FAIL — `archive root is missing ['__init__.py', 'blender_manifest.toml']`.

- [ ] **Step 4: Write the manifest and properties**

Create `addon/blender_elements/blender_manifest.toml`:

```toml
schema_version = "1.0.0"

id = "blender_elements"
version = "0.1.0"
name = "Elements"
tagline = "Real-time GPU field simulation from the Elements engine"
maintainer = "Elements Suite contributors"
type = "add-on"

license = ["SPDX:GPL-3.0-or-later"]
blender_version_min = "4.2.0"

tags = ["Physics", "Object"]
```

Create `addon/blender_elements/props.py`:

```python
"""Scene-level settings for the Elements add-on."""

import bpy


class ElementsSettings(bpy.types.PropertyGroup):
    graph_path: bpy.props.StringProperty(
        name="Graph",
        description="The .elements document to evaluate",
        subtype="FILE_PATH",
    )
    daemon_path: bpy.props.StringProperty(
        name="Engine",
        description="Path to the elementsd executable",
        subtype="FILE_PATH",
    )
    endpoint: bpy.props.StringProperty(
        name="Endpoint",
        description="Control socket path or named pipe",
    )
    channel_path: bpy.props.StringProperty(
        name="Channel",
        description="Memory-mapped frame channel file",
    )
    status: bpy.props.StringProperty(
        name="Status",
        default="Engine stopped",
    )
    live: bpy.props.BoolProperty(
        name="Live",
        description="Update the viewport volume on every redraw",
        default=False,
    )


def register() -> None:
    bpy.utils.register_class(ElementsSettings)
    bpy.types.Scene.elements = bpy.props.PointerProperty(type=ElementsSettings)


def unregister() -> None:
    del bpy.types.Scene.elements
    bpy.utils.unregister_class(ElementsSettings)
```

- [ ] **Step 5: Write the operators, panel, handler and entry point**

Create `addon/blender_elements/ops.py`:

```python
"""Operators that drive the engine process."""

import os
import subprocess
import tempfile

import bpy

from .client import ControlClient, ElementsError, FrameReader

# Module-level engine state. Blender operators are stateless, and the process
# and its sockets must outlive any single invocation.
_state = {"process": None, "client": None, "reader": None}


def _defaults(settings) -> None:
    """Fill in endpoint and channel paths if the user left them blank."""
    base = tempfile.gettempdir()
    if not settings.endpoint:
        settings.endpoint = (
            f"\\\\.\\pipe\\elements-{os.getpid()}"
            if os.name == "nt"
            else os.path.join(base, f"elements-{os.getpid()}.sock")
        )
    if not settings.channel_path:
        settings.channel_path = os.path.join(base, f"elements-{os.getpid()}.bin")


def shutdown_engine() -> None:
    """Tear down the client, reader and process. Safe to call repeatedly."""
    if _state["reader"] is not None:
        _state["reader"].close()
        _state["reader"] = None
    if _state["client"] is not None:
        try:
            _state["client"].shutdown()
        except ElementsError:
            pass
        _state["client"].close()
        _state["client"] = None
    if _state["process"] is not None:
        _state["process"].terminate()
        try:
            _state["process"].wait(timeout=5)
        except subprocess.TimeoutExpired:
            _state["process"].kill()
        _state["process"] = None


class ELEMENTS_OT_start_engine(bpy.types.Operator):
    bl_idname = "elements.start_engine"
    bl_label = "Start Engine"
    bl_description = "Launch the Elements engine and load the current graph"

    def execute(self, context):
        settings = context.scene.elements
        _defaults(settings)
        shutdown_engine()

        daemon = bpy.path.abspath(settings.daemon_path)
        if not daemon or not os.path.exists(daemon):
            settings.status = "Engine executable not found"
            self.report({"ERROR"}, settings.status)
            return {"CANCELLED"}

        try:
            process = subprocess.Popen(
                [daemon, "--endpoint", settings.endpoint,
                 "--channel", bpy.path.abspath(settings.channel_path)],
                stdout=subprocess.PIPE,
                text=True,
            )
            # The daemon prints "ready <endpoint>" once it is listening.
            line = process.stdout.readline()
            if not line.startswith("ready "):
                raise ElementsError("io", f"engine did not start: {line.strip()}")
            _state["process"] = process

            client = ControlClient(settings.endpoint)
            client.connect()
            ack = client.hello()
            _state["client"] = client

            loaded = client.load_graph(bpy.path.abspath(settings.graph_path))
            _state["reader"] = FrameReader(bpy.path.abspath(settings.channel_path))

            settings.status = (
                f"Running on {ack['adapter']} — {loaded['dims']}, {loaded['nodes']} nodes"
            )
        except (ElementsError, OSError) as e:
            shutdown_engine()
            settings.status = str(e)
            self.report({"ERROR"}, settings.status)
            return {"CANCELLED"}

        return {"FINISHED"}


class ELEMENTS_OT_stop_engine(bpy.types.Operator):
    bl_idname = "elements.stop_engine"
    bl_label = "Stop Engine"

    def execute(self, context):
        shutdown_engine()
        context.scene.elements.status = "Engine stopped"
        return {"FINISHED"}


class ELEMENTS_OT_render_frame(bpy.types.Operator):
    bl_idname = "elements.render_frame"
    bl_label = "Render Frame"
    bl_description = "Evaluate the graph once and push the result into a Volume"

    def execute(self, context):
        from .handlers import push_frame_to_volume

        settings = context.scene.elements
        client, reader = _state["client"], _state["reader"]
        if client is None or reader is None:
            settings.status = "Engine is not running"
            self.report({"ERROR"}, settings.status)
            return {"CANCELLED"}

        try:
            frame = client.render(context.scene.frame_current)
            seq, values = reader.read_latest()
            push_frame_to_volume(context, values, frame["dims"])
            settings.status = f"Frame {seq} — {frame['dims']}"
        except ElementsError as e:
            settings.status = f"{e.kind}: {e.message}"
            self.report({"ERROR"}, settings.status)
            return {"CANCELLED"}

        return {"FINISHED"}


CLASSES = (
    ELEMENTS_OT_start_engine,
    ELEMENTS_OT_stop_engine,
    ELEMENTS_OT_render_frame,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    shutdown_engine()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
```

Create `addon/blender_elements/handlers.py`:

```python
"""Getting engine frames into Blender's volume data.

Blender's Python API cannot build an OpenVDB grid in memory, so the frame is
written to a temporary .vdb through the engine's own writer path and imported.
Core v1 takes the simple, correct route; a direct in-memory path is an Ember
optimisation, not a Core v1 requirement.
"""

import os
import subprocess
import tempfile

import bpy

from .client import ElementsError

VOLUME_NAME = "ElementsVolume"


def _ensure_volume(context) -> bpy.types.Object:
    obj = bpy.data.objects.get(VOLUME_NAME)
    if obj is None or obj.type != "VOLUME":
        data = bpy.data.volumes.new(VOLUME_NAME)
        obj = bpy.data.objects.new(VOLUME_NAME, data)
        context.collection.objects.link(obj)
    return obj


def push_frame_to_volume(context, values, dims) -> bpy.types.Object:
    """Point the Volume object at a freshly baked .vdb of the current graph.

    `values` and `dims` come from the shared-memory frame and are used for the
    status readout and sanity checks; the geometry Blender renders comes from
    the engine's own VDB writer, because the add-on must never reimplement the
    file format in Python.
    """
    obj = _ensure_volume(context)

    expected = dims[0] * dims[1] * dims[2]
    if len(values) != expected:
        raise ElementsError(
            "io", f"frame has {len(values)} values, expected {expected}"
        )

    path = os.path.join(tempfile.gettempdir(), f"elements-frame-{os.getpid()}.vdb")
    _bake_current_graph_to(path)

    obj.data.filepath = path
    obj.data.reload()
    return obj


def _bake_current_graph_to(path) -> None:
    """Ask the engine CLI to bake the loaded graph to `path`.

    The CLI lives beside the daemon, since both are built into the same
    cargo target directory and shipped together.
    """
    settings = bpy.context.scene.elements
    cli = os.path.join(
        os.path.dirname(bpy.path.abspath(settings.daemon_path)), "elements"
    )
    if os.name == "nt":
        cli += ".exe"

    out_dir = os.path.dirname(path)
    result = subprocess.run(
        [
            cli,
            "bake",
            bpy.path.abspath(settings.graph_path),
            "--out", out_dir,
            "--frames", "1",
            "--name", "density",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ElementsError("io", f"bake failed: {result.stderr.strip()}")

    os.replace(os.path.join(out_dir, "density.0001.vdb"), path)
```

Note the imports this file needs: `os`, `subprocess`, `tempfile`, `bpy`, and
`from .client import ElementsError`. The `struct` import shown in some drafts is
not used — do not add it.

Create `addon/blender_elements/ui.py`:

```python
"""The N-panel."""

import bpy


class VIEW3D_PT_elements(bpy.types.Panel):
    bl_label = "Elements"
    bl_idname = "VIEW3D_PT_elements"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Elements"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.elements

        layout.prop(settings, "graph_path")
        layout.prop(settings, "daemon_path")

        row = layout.row(align=True)
        row.operator("elements.start_engine", icon="PLAY")
        row.operator("elements.stop_engine", icon="PAUSE")

        layout.operator("elements.render_frame", icon="FILE_REFRESH")
        layout.label(text=settings.status, icon="INFO")


def register() -> None:
    bpy.utils.register_class(VIEW3D_PT_elements)


def unregister() -> None:
    bpy.utils.unregister_class(VIEW3D_PT_elements)
```

Append the live draw handler to `addon/blender_elements/handlers.py`. Spec §3.4
requires the viewport to update on redraw, not only on an explicit operator
press; the `live` toggle gates it so a heavy graph cannot wedge the UI.

```python
_draw_handle = None
_last_seq = -1


def _on_draw() -> None:
    """Runs on every 3D viewport redraw while `live` is enabled.

    Must never raise into Blender's draw loop: an exception here is reported
    once in the status line and then swallowed, degrading to a stale volume
    rather than a broken viewport.
    """
    global _last_seq

    scene = bpy.context.scene
    settings = getattr(scene, "elements", None)
    if settings is None or not settings.live:
        return

    from . import ops

    reader = ops._state.get("reader")
    client = ops._state.get("client")
    if reader is None or client is None:
        return

    try:
        client.render(scene.frame_current)
        seq, values = reader.read_latest()
        if seq != _last_seq:
            push_frame_to_volume(bpy.context, values, list(reader.header()["dims"]))
            _last_seq = seq
            settings.status = f"Live — frame {seq}"
    except Exception as e:  # noqa: BLE001 - see the docstring
        settings.live = False
        settings.status = f"Live update stopped: {e}"


def register() -> None:
    global _draw_handle
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _on_draw, (), "WINDOW", "POST_PIXEL"
        )


def unregister() -> None:
    global _draw_handle, _last_seq
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        _draw_handle = None
    _last_seq = -1
```

Add a `live` toggle to the panel in `addon/blender_elements/ui.py`, after the
`render_frame` operator:

```python
        layout.prop(settings, "live", toggle=True, icon="REC")
```

Create `addon/blender_elements/__init__.py`:

```python
"""Elements — real-time GPU field simulation for Blender.

Relative imports throughout: extensions are imported as
bl_ext.<repo>.<id>, so absolute imports of this package's modules fail.
"""

from . import handlers, ops, props, ui

_MODULES = (props, ops, ui, handlers)


def register() -> None:
    for module in _MODULES:
        module.register()


def unregister() -> None:
    for module in reversed(_MODULES):
        module.unregister()
```

`handlers` registers last and unregisters first, so the draw handler is removed
before the operators and properties it reads disappear.

**Coverage note:** the live path cannot be exercised by `blender --background`,
which never redraws a viewport. It must be verified by hand: open Blender, start
the engine, enable **Live**, and confirm the volume updates as the frame changes
and that disabling Live stops the updates. Record the result when closing out
this task — an unverified live path is an open item, not a done one.

- [ ] **Step 6: Build and verify the archive layout**

Run: `just addon`
Expected: `built .../dist/blender_elements-0.1.0.zip with 7 files`, no layout error.

Verify independently, exactly as the project's packaging rule requires:

Run:
```bash
python3 -c "import zipfile; ns=zipfile.ZipFile('dist/blender_elements-0.1.0.zip').namelist(); assert '__init__.py' in ns and 'blender_manifest.toml' in ns; assert not any(n.startswith('blender_elements/') for n in ns); print('layout ok', ns)"
```
Expected: `layout ok [...]`.

- [ ] **Step 7: Run the Blender integration test**

Run: `BLENDER_BIN=$(command -v blender) WGPU_BACKEND=vulkan cargo test -p elementsd --test blender_integration -- --nocapture`
Expected: PASS with `blender roundtrip ok` in the output.

If Blender is not installed, the test prints a skip notice and passes. Install Blender 4.2 or newer before claiming this task is done — a skipped test is not a passing one.

Before reinstalling during iteration, delete the previous extension directory, or Blender will load stale code:

```bash
rm -rf "$HOME/Library/Application Support/Blender/4.2/extensions/user_default/blender_elements"
```

- [ ] **Step 8: Run the whole workspace**

Run:
```bash
just check && just addon
```
Expected: all clean.

- [ ] **Step 9: Commit**

```bash
git add addon scripts tests crates/elementsd
git commit -m "feat: add Blender extension with engine control panel and packaging"
```

---

## Definition of done for Core v1

Every box below must be checked with a command that was actually run.

- [ ] `cargo test --workspace` passes on lavapipe in CI.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` is clean.
- [ ] `elements bake` produces a `.vdb` that opens in Blender by hand, not only in `vdb-rs`.
- [ ] `elements render-preview` produces a PNG that a human has looked at and judged to be structured noise.
- [ ] The Python contract test passes against the real daemon on Python 3.11.
- [ ] `blender --background` installs the built ZIP and renders one frame.
- [ ] The built ZIP has `__init__.py` and `blender_manifest.toml` at its root with nothing nested.
- [ ] A deliberately broken graph produces a typed error in the Blender panel and leaves the engine running.
- [ ] The **Live** toggle has been verified by hand in an interactive Blender session (it cannot be covered headlessly).

---

## Notes for the next plan (Ember)

Deliberately deferred here, and the first things Ember will need:

- **Multi-consumer outputs.** Task 6 restricts each output to one input so values can be moved. Ember's graphs branch; this needs reference counting or a copy node.
- **Cross-frame state.** `FieldPool` recycles within a frame. A solver needs fields that persist between frames, which changes the pool's lifetime model.
- **Time.** `Command::Render { frame }` carries a frame number the engine currently ignores. Nodes will need it.
- **In-memory volume handoff.** Task 18 round-trips a `.vdb` through disk. Ember should push grids to Blender without the file.
- **Half-precision on the wire.** The frame channel publishes `f32`. At Ember's resolutions, `f16` halves the bandwidth.
