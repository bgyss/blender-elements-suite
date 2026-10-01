# FT5: the shack render

One bake of `examples/jet_shack_render.elements` writes every grid of a frame into one VDB, and
Blender renders that frame to three stills. Measured locally on Metal (Apple M1 Max). The llvmpipe
CI did not run the render. `git ls-remote --heads origin flamethrower-ft5` printed nothing when run
for the final review and again after it, so the branch was not pushed and CI has not run on it
(`gh run list` was not consulted).

Spec: `docs/superpowers/specs/2026-09-30-flamethrower-ft5-export-render-design.md`.
Plan: `docs/superpowers/plans/2026-09-30-flamethrower-ft5-export-render.md`.

## What `just render-shack` runs

`just render-shack` (default frame 45) builds `elements-cli`, bakes the scene with
`--frames 45 --name shack --voxel-size 0.015625` (the domain size over the largest dimension, read
from the scene JSON), then runs `tests/bench/render_shack.py` in Blender 5.2.2 LTS to write
`shack-beauty-f045.png`, `shack-fuel-f045.png` and `shack-char-f045.png`, `render-device` and
`commit` here. The bake simulates from frame 1 and writes only frame 45. It overwrites the stills,
so it can be rerun. The CLI prints "frame caching is off", which is expected.

- Commit the bake and render came from: see `commit` (65c4fbd; the recipe marks `-dirty` if
  `crates`, `Cargo.*`, `tests/bench` or `examples` differ; this run was not dirty). That hash is the commit the stills were rendered from, which is the commit
  before the record commit (d1957d3); the `render-shack` recipe itself exists only from d1957d3, and
  its dirty check excludes `justfile` and `docs`.
- Device: `GPU (Apple M1 Max (GPU - 32 cores))` (`render-device`). Cycles 64 spp, no denoise.
- Load average (1, 5, 15 min) printed by the recipe: before the bake `{ 12.40 13.61 16.57 }`, at the
  end `{ 11.96 13.50 16.51 }`. The machine was loaded, so timings are not clean.
- Bake wall time: 2 s for frames 1 to 45 (whole seconds, from the recipe's `$SECONDS`). The file is
  `target/ft5/render/shack.0045.vdb`, 17,911,147 bytes (17.9 MB) for one frame of eight grids.
  `Timeline.goto` computes the extra outputs on every simulated frame; at 2 s for 45 frames that is
  not material here.
- Render time: 2.2 s beauty, 1.1 s fuel, 0.9 s char.
- PNG sizes: beauty 317,311 bytes, fuel 118,685, char 104,001.

## Grids in the file

Blender 5.2.2 LTS, loading `shack.0045.vdb` with `volume.grids.load()`:

```
GRIDS ['char', 'density', 'flame', 'fuel', 'temperature', 'velocity_x', 'velocity_y', 'velocity_z']
VEL velocity FRAME      # volume.velocity_grid = "velocity" is accepted
```

Eight grids, as the scene's `outputs` list names them (Blender lists them alphabetically). Setting
`velocity_grid = "velocity"` was accepted; whether Blender then reads `velocity_x/y/z` for motion
blur was not tested by a render here.

## Image checks and thresholds

Measured on this run (script output):

| still | measured | check |
|---|---|---|
| beauty | 8066 warm pixels (red over blue); peak 0.9804 | at least 1000 warm pixels |
| fuel | 8337 pixels over 0.02; lit box (121, 161, 543, 279); 423 columns wide | lights some pixels; at least 2x as wide as char and not inside it |
| char | 242 pixels over 0.02; lit box (526, 267, 552, 279); 27 columns wide | lights some pixels; within the shack's projected x range (columns 515-805) +/- 4 voxels |

PNG pixel values cannot be non-finite, so the script checks lit and warm pixel counts against
floors instead. The 1000-pixel floor, the 2x width ratio and the 4-voxel tolerance are fitted to the
frame-45 measurement, not derived. The mutation notes in the script's constants record 0 warm pixels
with flame strength 0 or a misnamed flame attribute, and a fuel box of 23 columns when the fuel pass
reads char.

What the checks do not prove:

- The temperature to Kelvin map is not pixel-tested. A mutation that reads `density` for
  temperature is not caught by any check. It is covered by `kelvin()`'s unit test and by eye.
- The char check can only fail on its left bound in practice (see the script's docstring).
- `char` is a volume still, not a shader on the planks: Cycles surface shaders cannot sample a
  volume grid. Mesh-lookup char shading is FT6.
- The removal of a partial `.vdb` after a failed write is untested.
- Mantaflow is not involved.

## Char is almost absent at default settings

At default solver settings the shack barely chars in this scene. Char is 1 to 2 voxels at voxel
x 72 to 74 (the shack's front face) at frame 45 and absent before frame 40, so the `char` still
shows a handful of pixels. FT4 found the same at default settings (`docs/bench/surface-ignition.md`).
Retuning is FT6's decision.

## Frame survey

Measured on the unchanged scene at default settings: a 31-frame bake (frames 20-50, release CLI),
5.7 s wall. Temperature minimum is 0.0 in every frame, so ambient is 0; temperature counts use
> 0.1, other counts use > 0. The shack spans voxel x 74..112. The recipe's default frame is 45
because the jet is active for frames 5-40 and the fire fades by 50; frame 60 (from the Task 6 measurement) is nearly empty.

| frame | density max / nz | flame max / nz | temp max / nz(>0.1) | fuel max / nz | char max / nz | char x-range |
|---|---|---|---|---|---|---|
| 20 | 0.0771 / 6539 | 0.892 / 1448 | 2.857 / 4516 | 4.865 / 1434 | 0 / 0 | - |
| 25 | 0.0910 / 10879 | 0.876 / 1245 | 2.797 / 6676 | 5.243 / 1209 | 0 / 0 | - |
| 30 | 0.0955 / 17666 | 0.888 / 1729 | 2.865 / 9235 | 4.613 / 1717 | 0 / 0 | - |
| 35 | 0.1045 / 26955 | 0.888 / 1305 | 2.852 / 12620 | 5.341 / 1292 | 0 / 0 | - |
| 40 | 0.1157 / 39351 | 0.900 / 1257 | 2.860 / 16588 | 4.805 / 1237 | 0.0833 / 1 | 72..72 |
| 45 | 0.1003 / 57426 | 0.607 / 791 | 2.507 / 20959 | 1.774 / 782 | 0.1875 / 2 | 72..74 |
| 50 | 0.1050 / 77638 | 0.392 / 312 | 2.074 / 24507 | 0.610 / 310 | 0.2917 / 2 | 72..74 |
| 60 | not measured | not measured | not measured | 0.0075 / 6 | 0.50000006 / 4 | not measured |

## Verdict

Not yet recorded. The stills have not been looked at by a person; the checks in this README measure pixels, they do not judge the picture.

## Other notes

- `tests/bench/render_compare.py`'s `main()` got an `if __name__ == "__main__"` guard so
  `render_shack.py` can import its helpers. `just bench-render` is unaffected (not rerun here).
- Timings were taken under the load shown above.
