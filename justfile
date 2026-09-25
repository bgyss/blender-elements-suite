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

# Python checks that need no Blender: the benchmark's mappings and render layout.
py-test:
    python tests/bench/test_mapping.py
    python tests/bench/test_placement.py

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

# Ember in a warm process against Mantaflow re-baking in one open Blender.
# Waits up to 5 minutes for the 1-minute load to fall below 2, then runs
# anyway (the report flags loaded runs). Takes 20-30 minutes, real GPU and
# Blender, not in `check`. Writes docs/bench/results/latency-*.json, then
# docs/bench/latency.md.
# Latency from a parameter change to frame N at 128³ (2b-3b spec §2).
bench-latency scenes="plume plume_collider plume_wind" res="128":
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --release -p elements-ember --example benchmark
    B=target/release/examples/benchmark
    # The load the benchmark did not cause, for the report.
    RECIPE_LOAD="$(sysctl -n vm.loadavg)"
    load() { sysctl -n vm.loadavg | awk '{print $2}'; }
    # The load wait runs once, before the first scene only: later solvers
    # start right after the previous one and inherit its load, which the
    # per-solver log lines below record.
    for _ in $(seq 30); do
      awk -v l="$(load)" 'BEGIN { exit !(l < 2) }' && break
      echo "load $(load): waiting for it to fall below 2"
      sleep 10
    done
    for scene in {{scenes}}; do
      echo "load before Ember $scene: $(sysctl -n vm.loadavg)"
      "$B" latency "$scene" {{res}}
      echo "load before Blender $scene: $(sysctl -n vm.loadavg)"
      "$B" mantaflow-latency "$scene" {{res}}
    done
    echo "load after: $(sysctl -n vm.loadavg)"
    "$B" latency-report "$RECIPE_LOAD"

# Each scene at 128³ and plume at 256³, frames 30, 60 and 90 (2b-3b spec §3).
# Bakes go under target/bench/render/ and are skipped when their files exist;
# the PNGs and README.md go in docs/bench/render/. Real GPU and Blender, about
# 15 minutes from nothing (6 of them plume's 256³ Mantaflow bake); not in
# `check`.
# Render Ember and Mantaflow side by side with one Cycles setup.
bench-render cases="plume:128 plume_collider:128 plume_wind:128 plume:256":
    #!/usr/bin/env bash
    set -euo pipefail
    BLENDER_BIN="${BLENDER_BIN:-/Applications/Blender.app/Contents/MacOS/Blender}"
    # Two builds: `--example` would limit a joint build to the example alone.
    cargo build --release -p elements-ember --example benchmark
    cargo build --release -p elements-cli
    B=target/release/examples/benchmark
    CLI=target/release/elements  # the elements-cli binary
    R=target/bench/render
    OUT=docs/bench/render
    mkdir -p "$R" "$OUT"
    # The Ember commit, marked dirty if the engine differs from it.
    commit="$(git rev-parse --short HEAD)"
    git diff --quiet HEAD -- crates Cargo.toml Cargo.lock || commit="$commit-dirty"
    for case in {{cases}}; do
      scene="${case%%:*}"; res="${case##*:}"
      "$B" scene-json "$scene" "$res" > "$R/$scene-$res.json"
      dx="$(python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(d['domain_size'] / d['resolution'])" "$R/$scene-$res.json")"
      ember="$R/ember/$scene-$res"
      if [ -f "$ember/commit" ]; then
        echo "ember $scene ${res}³: baked already at $(cat "$ember/commit")"
      else
        mkdir -p "$ember"
        "$B" document "$scene" "$res" > "$R/$scene-$res.elements"
        # Each bake simulates from frame 1 and writes only the one frame.
        for f in 30 60 90; do
          echo "ember $scene ${res}³: frame $f; load $(sysctl -n vm.loadavg)"
          "$CLI" bake "$R/$scene-$res.elements" --out "$ember" --frames "$f" \
              --name density --voxel-size "$dx"
        done
        echo "$commit" > "$ember/commit"
      fi
      manta="$R/mantaflow/$scene-$res"
      if [ -f "$manta/timings.json" ]; then
        echo "mantaflow $scene ${res}³: baked already"
      elif [ -d "$manta/cache/data" ]; then
        # mantaflow_scene.py never clears a cache, so a stale frame could
        # pass its missing-frame check.
        echo "mantaflow $scene ${res}³: $manta holds an unfinished bake; delete it and rerun" >&2
        exit 1
      else
        echo "mantaflow $scene ${res}³: baking frames 1-90; load $(sysctl -n vm.loadavg)"
        mkdir -p "$manta"
        # The benchmark's scene, cut to the 90 frames the render uses.
        python3 -c "import json, sys; d = json.load(open(sys.argv[1])); d['frames'] = 90; json.dump(d, open(sys.argv[2], 'w'), indent=1)" \
            "$R/$scene-$res.json" "$manta/scene.json"
        "$BLENDER_BIN" --background --factory-startup --python-exit-code 1 \
            --python tests/bench/mantaflow_scene.py -- "$manta/scene.json" "$manta" \
            > "$manta/blender.log" 2>&1 || { tail -30 "$manta/blender.log"; exit 1; }
      fi
      echo "render $scene ${res}³; load $(sysctl -n vm.loadavg)"
      "$BLENDER_BIN" --background --factory-startup --python-exit-code 1 \
          --python tests/bench/render_compare.py -- "$ember" "$manta/cache" "$scene" "$res" "$OUT" \
          > "$R/render-$scene-$res.log" 2>&1 || { tail -30 "$R/render-$scene-$res.log"; exit 1; }
    done
    echo "load after: $(sysctl -n vm.loadavg)"
    "$BLENDER_BIN" --background --factory-startup --python-exit-code 1 \
        --python tests/bench/render_compare.py -- --readme "$OUT" "$R" {{cases}} \
        > "$R/readme.log" 2>&1 || { tail -30 "$R/readme.log"; exit 1; }
    echo "wrote $OUT/README.md"
