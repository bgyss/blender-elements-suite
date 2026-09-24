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

# Run the Rust test suite on this machine's native GPU backend, and the
# Python checks that need no Blender.
test: py-test
    cargo nextest run --workspace

# Run the Rust test suite the way CI does, on software Vulkan.
ci-test: py-test
    WGPU_BACKEND=vulkan LIBGL_ALWAYS_SOFTWARE=1 cargo nextest run --workspace

# The benchmark's Ember -> Mantaflow parameter mappings; no Blender needed.
py-test:
    python tests/bench/test_mapping.py

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

# Sweep pressure iterations past the gate's range, for choosing 2b's presets.
# Writes docs/bench/iteration-sweep.md and leaves the gate's record alone.
bench-sweep iterations="160,240,320,400,480,560,640":
    SPEED_GATE_ITERATIONS={{iterations}} cargo run --release -p elements-ember --example speed_gate

# The preview preset's substep cap at 128³ (2b-1 spec §6).
# Takes minutes, needs the real GPU, and is not part of `check`.
bench-presets:
    cargo run --release -p elements-ember --example presets

# The 2b-3c solver gate: Gauss–Seidel against multigrid and MGPCG (spec §4).
# Writes docs/bench/solver-gate.md. Takes about an hour, real GPU, not in `check`.
bench-solver:
    cargo run --release -p elements-ember --example solver_gate

# Rerun only the solver gate's per-solve timing with MGPCG at `count`, into
# the existing docs/bench/solver-gate.md. About an hour, real GPU.
bench-solver-timing count="10":
    SOLVER_GATE_TIMING_ONLY={{count}} cargo run --release -p elements-ember --example solver_gate

# The Mantaflow benchmark (2b-3 spec). Takes about an hour or more, needs the
# real GPU and Blender, and is not part of `check`. Writes
# docs/bench/results/ per run, then docs/bench/results.md from a complete set.
bench scenes="plume plume_collider plume_wind" resolutions="64 128 256":
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --release -p elements-ember --example benchmark
    B=target/release/examples/benchmark
    for res in {{resolutions}}; do
      for scene in {{scenes}}; do
        "$B" ember "$scene" "$res"
        "$B" mantaflow "$scene" "$res"
      done
    done
    "$B" report
