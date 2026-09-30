# Flamethrower vs. shack: shot spec (FT0)

**Date:** 2026-09-30. **Roadmap:** `2026-09-30-flamethrower-napalm-roadmap-design.md` §4.

## Reference
Jason Key (@key_vfx), "flamethower vs. shack", 2026-09-01, EmberGen 1.2.11, 30 s. Used for
comparison only; no frames or assets are stored in this repository.

## Scene (Ember units, metres, z up)
- Domain `dims [256, 128, 128]`, `domain_size 4.0` (x longest): x 4.0 m, y 2.0 m, z 2.0 m,
  dx = 15.625 mm. Non-cubic support: the existing solver runs dims [32,32,64] with no change
  (Task 1, commit a769abf, `tests/noncubic.rs`). The shot's own [256,128,128] ran in the
  budget measurement below.
- Shack: `ShackParams { size [1.2, 1.0, 1.0], plank_height 0.1, thickness 0.02, gap 0.01,
  broken_fraction 0.2, seed 7 }`, collider origin at (2.9, 1.0, 0.0). Planks are thinner than
  one voxel, so the mesh collider uses `offset = 0.5 * dx`.
- Nozzle: cone along +x, `length 0.3`, `radius_start 0.03`, `radius_end 0.06`, origin at
  (0.4, 1.0, 0.6), emitting `fuel_rate`, `temperature_rate` and a local-frame velocity of
  `[14, 0, 0]` m/s with `velocity_blend 20`; noise amplitude 0.6, scale 0.08 m; pulse on
  frames 10–150.
- Frames 1–240 at 24 fps. Camera: side view, orthographic-like long lens framing the jet and
  the shack; set in FT6.

## Acceptance for the shot (FT6)
1. Jet visibly reaches the shack and wraps it; the shack ignites (FT4) and keeps burning.
2. Smoke and flame separate (flame output, density output, fuel output all available).
3. No NaN, no domain-edge blow-up; divergence recorded per FT3.
4. Verdict versus the reference, including detail and speed gaps, recorded honestly.

## Budget
Measured with the existing smoke path (`examples/plume.elements` with `dims [256, 128, 128]`,
`domain_size 4.0`, sphere emitter moved to `[0.5, 1.0, 0.6]`; preview-style defaults: 1
substep, 160 pressure iterations, no fire, no collider), baked with
`elements bake ... --frames 1-10` (release build, Apple Silicon, Metal). The domain fits:
no GPU out-of-memory, so the fallback to `[192, 96, 96]` was not needed and the Scene
section keeps `[256, 128, 128]`.

| Measure | Value |
|---|---|
| 10 frames, wall time (`/usr/bin/time -l`) | 6.77 s real, 0.90 s user, 0.31 s sys |
| Per frame | about 0.68 s, including the ~18 MB `.vdb` write per frame and startup (a 1-frame bake took 0.35 s) |
| Peak memory footprint | 1.59 GB at 10 frames (0.59 GB for 1 frame); max resident set 83 MB (GPU buffers are unified memory and show in the footprint, not the RSS) |
| Machine load (`uptime`) | load average 28.7 / 37.9 / 29.3 before the run, 25.1 / 36.8 / 29.0 after, on 10 cores |

**These timings were taken under heavy load (load average 25–38 on 10 cores) and are not
clean.** Treat 0.68 s a frame as an upper bound for the smoke path, not a benchmark. They
cover plain smoke only: fire (risk n), the mesh collider's per-cell triangle loop and the
cone emitter are not included and will add cost. Re-measure on an idle machine, and with the
shot's real graph, in FT6. The 1.59 GB footprint grows with frames because the timeline's
frame cache holds snapshots; it is well under the machine's unified memory.
