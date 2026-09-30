# Mesh collider SDF fill cost (flamethrower FT2, Task 8)

Question: is a brute-force per-frame SDF fill of the shot's shack acceptable
for a 240-frame bake? Rule from the plan: acceptable if under 10% of a solver
step.

## Setup

- Test: `shack_fill_cost` in `crates/elements-ember/tests/mesh_collider.rs`
  (`cargo nextest run -p elements-ember --run-ignored ignored-only -E 'test(shack_fill_cost)' --no-capture`).
- Mesh: `shack` with size [1.2, 1.0, 1.0], plank_height 0.1, thickness 0.02,
  gap 0.01, broken_fraction 0.2, seed 7: **372 triangles**.
- Time: `fill_mesh_collider` end to end (cell pass plus three face passes,
  buffer uploads) followed by `gpu.wait()`; one warm-up fill (pipeline
  compile) then 5 timed fills.
- Adapter: Apple M1 Max (Metal). Load average (`uptime`): 21.45 before run 1,
  35.23 after run 2. **The machine was heavily loaded, so these are upper
  bounds.**

## Results (two separate runs of 5 fills)

| dims | run | median | range |
|---|---|---|---|
| 128³ | 1 | 37.6 ms | 37.0-38.3 ms |
| 128³ | 2 | 38.9 ms | 37.7-39.1 ms |
| 256×128×128 | 1 | 77.6 ms | 74.5-78.4 ms |
| 256×128×128 | 2 | 75.9 ms | 75.1-76.2 ms |

That is about 2e10 cell-triangle tests a second (128³ x 372 = 7.8e8 tests in
about 38 ms).

## Verdict

**Not acceptable as a per-frame cost.** A 128³ preview solver step is about
71.5 ms (`docs/bench/presets.md`, under load), so one fill is about 53% of a
step; at 256×128×128 a step is larger, but the fill is still well over 10%
(77 ms against a step that would have to exceed 770 ms). Even with a
generous allowance for the load, an idle machine would not bring it near
10%. (The loads differ: the fill was timed at load 21-35, the step at about
9.6, so the ratio overstates the fill's share by an unmeasured amount.)

Nothing was optimised here. Recorded as risk (o) in
`docs/superpowers/specs/2026-09-21-ember-solver-design.md` §6. Options for the
user: cache the SDF in node state while the pose is unchanged (the shack does
not move in the shot, so the fill would run once), or a coarse AABB early-out.
