# Ember and Mantaflow side by side

- Machine: Apple M1 Max
- OS: macOS 27.2
- Ember bakes' commit: eb6b1c1
- Mantaflow bakes' commit: 2e1f9a9-dirty
- Blender: 5.2.2 LTS
- Date: 2026-09-25

Written by `just bench-render` (`tests/bench/render_compare.py`), 2b-3b spec §3.
Each image shows one frame of one scene: Ember's density on the left and
Mantaflow's on the right, each in its 2 m domain drawn as a thin box.
Both are baked from the same scene definition
(`Scene` in `crates/elements-ember/src/bench`, through `benchmark document`
and `benchmark scene-json`), Ember by `elements-cli bake` and Mantaflow by
`tests/bench/mantaflow_scene.py`. Mantaflow bakes frames 1–90 once. Ember is
baked once for each of frames 30, 60 and 90, and each bake simulates from
frame 1 and writes only that frame; the simulation is deterministic, and a
rebake gives byte-identical files. The images are for judging by eye;
nothing here measures them.

## Render settings

- Cycles, 64 samples a pixel (adaptive sampling off), seed 0, denoiser
  off, 960×540, Standard view transform. Device:
  GPU (Apple M1 Max (GPU - 32 cores)).
- One orthographic camera looking along +y, so both domains are seen through
  the same projection: no perspective, and depth along y is not visible.
- Both Volume objects read their file's `density` grid at full precision and
  share one Principled Volume material: colour white, **density 20**
  (a multiplier on the grid value), absorption colour black, no emission, no
  blackbody.
- A sun key light (strength 10, 3° wide) from the camera's upper
  left, a soft sun fill (strength 2, 40° wide) from its lower
  right, and a uniform world of grey 0.18 at strength 1.
- Placement: both files put voxel i's centre at i·dx, half a cell below the
  cell's centre, so each object is moved by +0.5·dx on every axis
  (`tests/bench/placement.py`). Mantaflow's domain is 1.25 domains to the right
  of Ember's.

**One density scale for both is fair**: while both solvers are still emitting
and the plumes have not yet diverged, the emitted masses match: over frames
12–24, Ember's mass is 0.99–1.04× Mantaflow's across every benchmark run
(`docs/bench/results.md`, Notes, Heat). The scale is not normalised per
solver, so a difference in brightness or opacity is a difference in density.
At density 20, though, a plume's core is optically thick (τ ≈ 6: the
multiplier times grid values near 1 over a core about 0.3 m across), so the
images compare shape and extent rather than peak density.
Known differences carry through (`results.md`, `mantaflow-notes.md`):
Mantaflow's emitter heat is held at Ember's frame-24 value rather than added
at Ember's rate, so its plume rises faster; Mantaflow's open top boundary layer
is a density sink; and Mantaflow clamps density to [0, 1] at the emitter while
Ember does not, so Ember's densest cells can exceed 1 (behind an optically
thick core, that changes little in the image).

Verdict (recorded by the user, 2026-09-25): at matched emitted mass, Ember
reproduces Mantaflow's large-scale behaviour in every scene: the rise, the
mushroom cap, and the flow over and around the collider, where the two match
most closely. Ember is visibly smoother. Mantaflow keeps finer turbulent detail
in the stem (streaks and ripples by frame 60, wisps and bulges by 90) and a
larger cap about 0.1 of the domain higher, which agrees with its 1.3–2.1× kinetic
energy per measured cell in `docs/bench/results.md`; part of that extra height
is the heat mapping (its emitter is hotter early), not the solver. The wind
scene differs by design: Ember's ambient airflow bends the plume into a low band
and carries it out through +x, while Mantaflow's wind pushes only the smoke,
which rises diagonally and piles against +x. The remaining gap is Ember's
small-scale detail. The levers are less numerical dissipation in advection,
vorticity confinement (0 in both solvers' bench settings today), and the ML
upresolution seam; a heat mapping that matches Ember's rate would make the
heights comparable.

## `plume` (128³)

![plume 128³ frame 30](plume-128-f030.png)

![plume 128³ frame 60](plume-128-f060.png)

![plume 128³ frame 90](plume-128-f090.png)

## `plume_collider` (128³)

The collider sphere is not drawn; it shows only as the faint outline the smoke leaves around it.

![plume_collider 128³ frame 30](plume_collider-128-f030.png)

![plume_collider 128³ frame 60](plume_collider-128-f060.png)

![plume_collider 128³ frame 90](plume_collider-128-f090.png)

## `plume_wind` (128³)

Both panels show solver behaviour, not the render. Ember's smoke leaves through the open +x side from about frames 23–34 and is gone by frame 90 (`docs/bench/results.md`), which is why its frame-90 panel is empty. Mantaflow's smoke stays in the domain against +x: at frame 60 its mass is 0.0842 against Ember's 0.0382 (sum of density × dx³ in these bakes). Mantaflow's wind acts only on cells that hold smoke, while Ember's moves all the air (`docs/bench/mantaflow-notes.md`, Wind as ambient airflow).

![plume_wind 128³ frame 30](plume_wind-128-f030.png)

![plume_wind 128³ frame 60](plume_wind-128-f060.png)

![plume_wind 128³ frame 90](plume_wind-128-f090.png)

## `plume` (256³)

![plume 256³ frame 30](plume-256-f030.png)

![plume 256³ frame 60](plume-256-f060.png)

![plume 256³ frame 90](plume-256-f090.png)
