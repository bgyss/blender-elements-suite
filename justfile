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

# Run the Blender integration test.
# Finds Blender on PATH or in the standard macOS application bundle.
blender-test:
    #!/usr/bin/env bash
    set -euo pipefail
    BLENDER_BIN="${BLENDER_BIN:-$(command -v blender || true)}"
    if [ -z "$BLENDER_BIN" ] && [ -x /Applications/Blender.app/Contents/MacOS/Blender ]; then
        BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender
    fi
    if [ -z "$BLENDER_BIN" ]; then
        echo "Blender not found. Set BLENDER_BIN." >&2
        exit 1
    fi
    echo "using $BLENDER_BIN ($("$BLENDER_BIN" --version | head -1))"
    BLENDER_BIN="$BLENDER_BIN" cargo test -p elementsd --test blender_integration -- --nocapture

# The piece 2a speed gate: step time and divergence at 128³ (spec §4.3).
# Takes minutes, needs the real GPU, and is not part of `check`.
bench-gate:
    cargo run --release -p elements-ember --example speed_gate
