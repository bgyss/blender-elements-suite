# Mantaflow 16³ cache fixture

`fluid_data_0005.vdb` is frame 5 of a Blender Mantaflow gas bake. It is the
test fixture for the Mantaflow cache reader (2b-3 Task 7), so that reading the
cache is checked in CI, which has no Blender. Findings about the cache are in
`docs/bench/mantaflow-notes.md`.

- **Size:** 967,548 bytes. The plan asked for a fixture under 200 KB. That is
  not possible uncompressed: the size hardly depends on resolution (an 8³ frame
  is 935,658 bytes), because uncompressed OpenVDB internal nodes store their
  full tile tables for every grid.
- **Blender:** 5.2.2 LTS (hash d13f752e3b9c, built 2026-09-15), macOS arm64,
  baked 2026-09-23.
- **SHA-256:** `95602a2eef42b3660b520934fd3edd60dfbc6828ceec9673513e5a993037c6cf`
- **Command**, from the repository root:

  ```bash
  Blender --background --factory-startup --python-exit-code 1 \
      --python tests/bench/probe_mantaflow.py -- OUT_DIR 16 5 bench=1 volume=1
  ```

  That bakes a 2 m `GAS` domain at 16³ (dx = 0.125 m) for 5 frames at 24 fps,
  with a smoke `INFLOW` sphere of radius 0.2 m at (1, 1, 0.3) m, volume
  emission 1, Blender's default flow density and temperature (1, 1), and the
  default alpha and beta (1, 1). Sides and floor are closed and the top is
  open. It uses one time step per frame, no noise, no adaptive domain, and
  32-bit uncompressed OpenVDB. The file is `OUT_DIR/data/fluid_data_0005.vdb`.

## Grid inventory

Read with the vendored, patched `vdb-rs` (`vendor/vdb-rs/PATCHED.md`):

| Grid | VDB type | Class | Values | Stored as | Index bbox |
|---|---|---|---|---|---|
| `density` | `Tree_float_5_4_3` | fog volume | 240 | voxels | (4,4,1)–(11,11,7) |
| `temperature` | `Tree_float_5_4_3` | fog volume | 240 | voxels | (4,4,1)–(11,11,7) |
| `velocity` | `Tree_vec3s_5_4_3` | staggered | 240 | voxels | (4,4,1)–(11,11,7) |
| `flame` | `Tree_float_5_4_3` | fog volume | 0 | — | empty |
| `shadow` | `Tree_float_5_4_3` | fog volume | 8 | `Node3` tiles, value −1 | (0,0,0)–(15,15,15) |

Every grid has `file_base_resolution` (16,16,16), `file_voxel_size` 0.125
and `is_saved_as_half_float = false`. The VDB transform is a uniform scale of
0.125 with no translation.

Reference values, from a dense read with absent voxels as 0 (for asserting a
reader):

- density: sum 49.2145, max 0.99935; at index (8,8,2), 0.999351
- temperature: sum 62.6631
- velocity (Mantaflow's grid units, face values): component sums
  (3.08923, 3.08923, 104.634); max |z| 3.89063; at index (8,8,3),
  (9.419e-05, 9.418e-05, 3.11418)
