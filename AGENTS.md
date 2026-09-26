# Repository Guidelines

## Project Structure & Module Organization

`crates/` contains the Rust workspace: graph and GPU core, Ember solver, IPC, I/O, CLI, and daemon. The Blender extension lives in `addon/blender_elements/`; `scripts/build_addon.py` packages it. Rust integration tests sit beside each crate in `crates/*/tests/`. Blender and Python checks are in `tests/blender/` and `tests/bench/`; graph fixtures are in `tests/graphs/`. Keep design plans and benchmark evidence in `docs/`, sample graphs in `examples/`, and the patched VDB dependency in `vendor/vdb-rs/`.

## Build, Test, and Development Commands

Use `mise` for Python 3.11, Ruff, Just, and cargo-nextest; `rust-toolchain.toml` selects Rust. Run `just` to list recipes. `just fmt` formats Rust and Python. `just check` runs lint plus native-backend tests and is the pre-commit gate. `just addon` builds and verifies the Blender extension ZIP. `just blender-test` runs the Blender round-trip test when Blender is installed. Use `just ci-test` for the software-Vulkan test path on Linux; GPU benchmarks such as `just bench-gate` are separate, longer-running checks.

## Coding Style & Naming Conventions

Use `cargo fmt` for Rust and Ruff for Python. Ruff targets Python 3.11, enforces a 100-character line length, and checks imports and common errors. Follow existing `snake_case` module, function, and test names; Rust types use `PascalCase`. Keep Blender API imports and runtime behavior within the add-on. Do not rename the patched vendor crate without checking `Cargo.toml` and `vendor/vdb-rs/PATCHED.md`.

## Testing Guidelines

Add focused Rust tests in the affected crate's `tests/` directory; name files for the behavior, such as `noise.rs`. Run one file with `cargo nextest run -p elements-core --test noise`. Python checks run through `just test`; Blender integration requires `just blender-test`. For GPU changes, test on the native backend and use tolerance for cross-backend comparisons. Review regenerated golden PNGs visually before committing. Tests should demonstrate that the intended behavior can fail when changed.

## Commits & Pull Requests

Recent commits use short, imperative subjects (for example, “Clamp emitted fuel to [0, 10]”) and a body explaining why. Keep each commit focused and run `just check` first. In pull requests, explain the behavior, link the relevant issue or design document, list validation performed, and include screenshots or benchmark results for visible or performance changes. Distinguish local Metal results from Linux software-Vulkan CI results.

## Working Context

Read `.superpowers/sdd/progress.md` before starting planned work so completed milestones and recorded decisions are respected. `CLAUDE.md` documents current architecture, command details, and known GPU constraints.
