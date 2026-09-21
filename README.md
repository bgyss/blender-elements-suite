# Elements Suite

An open-source, Rust-based suite of real-time VFX authoring tools for Blender. The core engine (`crates/`) is dual-licensed Apache-2.0 OR MIT; the Blender add-on (`addon/`) is GPL-3.0-or-later.

## Development setup

### Prerequisites

- **`mise`** — pins Python 3.11, ruff, uv, just, and cargo-nextest. Install from https://mise.jdx.dev/.
- **`rustup`** — owns the Rust toolchain via `rust-toolchain.toml`. Install from https://rustup.rs/.
- **On macOS: Xcode Command Line Tools** — run `xcode-select --install`. This is required because `.cargo/config.toml` pins the linker to `/usr/bin/cc`; if Nix or another toolchain provides a `cc` earlier on `$PATH`, linking fails with "symbol(s) not found for architecture arm64".

### Quick start

```bash
mise install
direnv allow  # optional
just check
```

`just check` runs linting and tests. For other tasks, run `just` to list all recipes.

### Alternative: Nix

If you prefer a fully declarative environment, run `nix develop` to enter a Nix shell with all dependencies.

### Common recipes

- **`just fmt`** — Format Rust and Python code.
- **`just lint`** — Check formatting, run Clippy, and lint Python. Fails on any warning.
- **`just test`** — Run the Rust test suite on your native GPU backend.
- **`just ci-test`** — Run tests on software Vulkan (same as CI).
- **`just check`** — Run lint and test (required before committing).
- **`just addon`** — Build and verify the Blender extension ZIP.
- **`just golden`** — Regenerate golden test files (review outputs before committing).
- **`just blender-test`** — Run the Blender integration test (requires Blender installed).
