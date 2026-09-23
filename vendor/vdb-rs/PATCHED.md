# Patched vdb-rs

This is vdb-rs **0.6.0** from crates.io (upstream
https://github.com/Traverse-Research/vdb-rs, commit
`f146fc083a2df19da555b1c976ed8cd6b5498748`, from `.cargo_vcs_info.json`),
MIT licensed (`LICENSE`). The workspace uses it through
`[patch.crates-io]` in the root `Cargo.toml`.

Patched 2026-09-23 for Ember piece 2b-3 (the Mantaflow benchmark).

## The change

In `src/reader.rs`, `read_tree_topology`, two reads:

- the root node's background value, and
- each root tile's value

were read with `reader.read_u32::<LittleEndian>()`, which is always 4 bytes.
They now read `size_of::<ValueTy>()` bytes into a `ValueTy::zeroed()`, using
`read_exact` and `bytes_of_mut`. Both values are still discarded, as
upstream does. Each changed line is marked `PATCHED`. Nothing else was
changed. The crate's own `Cargo.lock` and the `.cargo-ok` marker were not
copied.

## Why

OpenVDB writes these values at the grid's value type's size. For a float grid
that is 4 bytes, so upstream works. For a `Vec3s` grid it is 12 bytes, so
upstream reads the background's y and z as the tile and root-node counts.
Both are 0.0 for Blender's Mantaflow velocity cache, so the tree parses as
empty, with no error. After the patch, the 16³ probe cache's `velocity` grid
reads all 292 of its voxels, which matches the grid's `file_voxel_count`
metadata. See `docs/bench/mantaflow-notes.md`.

The patch does not add a value-type check: vdb-rs still reads a grid as
whatever type the caller asks for. The caller must check
`descriptor.grid_type`.
