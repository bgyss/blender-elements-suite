# Mantaflow cache and parameter notes (2b-3 Task 3)

This file records how Blender's Mantaflow cache is laid out, how to read it,
and how Ember's scene parameters map onto Mantaflow's. Later 2b-3 tasks build
on these facts: the scene script (Task 6), the cache reader (Task 7) and the
results (Task 8).

Machine: macOS on Apple Silicon, Blender 5.2.2 LTS (hash d13f752e3b9c, built
2026-09-15). Date: 2026-09-23. The machine was under load from other
processes throughout (load average 8 to 28).

Sources were read, never copied. Paths are in github.com/blender/blender at
`main`:

- `intern/mantaflow/intern/MANTA_main.cpp`: which RNA values reach the script
- `intern/mantaflow/intern/strings/smoke_script.h`, `fluid_script.h`: the generated solver script
- `extern/mantaflow/preprocessed/plugin/extforces.cpp`: `addBuoyancy`, `vorticityConfinement`, `addForceField`, `setInitialVelocity`
- `extern/mantaflow/preprocessed/plugin/initplugins.cpp`: `applyEmission`
- `extern/mantaflow/preprocessed/fileio/iovdb.cpp`: the VDB writer
- `source/blender/blenkernel/intern/fluid.cc`: emission and force-field sampling
- `source/blender/blenkernel/intern/effect.cc`: the wind field's force and falloff

The probe is `tests/bench/probe_mantaflow.py`. Its options are listed in its
docstring. The experiments below give the options they used. Setting the
domain's `export_manta_script = True` (probe option `script=1`) also writes the
generated solver script to `<cache>/script/smoke_script.py`. That is the
fastest way to see what a setting turns into.

## Headless baking

`bpy.ops.fluid.bake_all()` runs under
`Blender --background --factory-startup --python` and returns `{'FINISHED'}`.
It needs only the domain as the context object:

```python
with bpy.context.temp_override(object=domain, active_object=domain):
    bpy.ops.fluid.bake_all()
```

**Pass `--python-exit-code 1` before `--python`.** Without it, Blender exits 0
even when the script raises, with the traceback only on stderr.

Every property name the probe sets exists in 5.2.2, including
`openvdb_data_depth` (`"32"`), `openvdb_cache_compress_type` (`"NONE"`),
`timesteps_min` / `timesteps_max`, `use_collision_border_*` and `clipping`.
One does not: a field's `falloff_type` has no `"NONE"` (its values are `CONE`,
`SPHERE` and `TUBE`). See the wind row of the mapping.

After a background bake, `domain_settings.domain_resolution` and `cell_size`
read back as zeros. Do not use them; the resolution is `resolution_max` along
the longest side and the cell size is domain size / resolution.

## Cache layout

For cache directory `C` and frame `n` (4 digits, zero padded):

```
C/data/fluid_data_NNNN.vdb     every grid for frame n, one file
C/config/config_NNNN.uni       small solver state
C/script/smoke_script.py       only with export_manta_script = True
```

A 16³ frame is 967,548 bytes, and an 8³ frame 935,658: uncompressed OpenVDB
internal nodes store full tile tables for every grid, so the size barely
depends on resolution. A 256³ frame was 3.5 to 5 MB in the first ten frames,
and it grows with the smoke.

## Grids

Grids in `fluid_data_NNNN.vdb` for a `GAS` domain with one smoke flow, no
noise, 32-bit, uncompressed. Read at 16³, frame 5 (probe defaults):

| Name | VDB type | Class | Reads as | Values | Index bbox | Stored as |
|---|---|---|---|---|---|---|
| `density` | `Tree_float_5_4_3` | fog volume | `f32` | 292 | (3,3,1)–(12,12,7) | voxels |
| `temperature` | `Tree_float_5_4_3` | fog volume | `f32` | 284 | (3,3,1)–(12,12,7) | voxels |
| `velocity` | `Tree_vec3s_5_4_3` | **staggered** | `[f32; 3]` | 292 | (3,3,1)–(12,12,7) | voxels |
| `flame` | `Tree_float_5_4_3` | fog volume | `f32` | 0 | empty | — |
| `shadow` | `Tree_float_5_4_3` | fog volume | `f32` | 8 | (0,0,0)–(15,15,15) | `Node3` tiles, all −1 |

Every grid carries `file_base_resolution` (16,16,16) and `file_voxel_size`
(0.125) metadata, and `is_saved_as_half_float = false`.

### Reading with vdb-rs

**The workspace uses a patched `vdb-rs` 0.6.0** in `vendor/vdb-rs/`, through
`[patch.crates-io]` in the root `Cargo.toml`. `vendor/vdb-rs/PATCHED.md` has
the details. Upstream 0.6.0 reads the root node's background value and each
root tile's value as 4 bytes, whatever the grid's value type. For a `Vec3s`
grid those values are 12 bytes, so upstream parses Mantaflow's velocity as an
empty tree (0 values, no error). **With the patch, velocity reads fully: all
292 values of the 16³ frame above**, matching its `file_voxel_count`
metadata. The committed fixture also reads fully: 240 of 240
(`crates/elements-ember/tests/fixtures/mantaflow_16/README.md`). The float
grids read the same with and without the patch. The existing `elements-io`
VDB tests (vdb-rs as the oracle) pass on the patched crate.

**What the patch leaves alone.** Root-level tile values are still read and
discarded; that is harmless, because a root tile covers 4096³ and no domain of
256³ or less has one. Inactive voxels read as 0, not as the grid's background
value: fine for density and velocity, whose background is 0, but `shadow`'s is
−1. Half-float decoding exists only for `f32` grids, so a half-float velocity
grid would not decode; the scene script must keep `openvdb_data_depth = "32"`.

**`vdb-rs` does not check value types.** Reading `density` as `[f32; 3]`
returns 0 values without an error, and reading `temperature` as `[f32; 3]`
gives `IoError`. The Task 7 reader must check `descriptor.grid_type`
(`Tree_float_5_4_3` for scalars, `Tree_vec3s_5_4_3` for velocity) before
reading, and must expand tiles (`VdbLevel::Node3` is 8³ voxels, `Node4` is
128³).

### Clipping: what the cache leaves out

The data file is written with `clip = domain.clipping` (default 1e-6) and
`clipGrid = density` (`smoke_script.h`, `smoke_save_data`; `iovdb.cpp`,
`exportVDB`). Density and temperature store only values above the clip value,
and the sparse grids, `velocity` included, are then clipped to density's
active voxels. So **the cache holds velocity only where there is smoke**.
Every other cell reads as zero.

**Clipping test.** Two bakes of the same 32³ scene (probe defaults plus
`volume=1`, 20 frames) differed only in `clipping`: 1e-6 and 0.

- Density is the same simulation. In every frame, each voxel the 1e-6 bake
  stores has the identical value in the 0 bake (maximum difference 0.0).
  Velocity is identical on those voxels too. Total density differs by at most
  2.7e-7 relative, which is the mass in the extra voxels below 1e-6. The
  clip affects only the output.
- Velocity does **not** become dense with `clipping = 0`. It is still clipped
  to voxels with non-zero density: at frame 20, 22,278 of 32,768 cells
  (against 4,353 at 1e-6).

**Decision: keep the default `clipping` (1e-6), and compute the velocity
metrics for both solvers only over *measured* cells (spec §4.1): cells whose
whole 3×3×3 neighbourhood lies inside the domain, holds no collider cell and
has density above 1e-6.** The same mask is applied to Ember's fields.

**Why the mask is the whole neighbourhood, not the cell.** Velocity is
staggered (see Velocity location), and the cache stores velocity index
(i,j,k), which holds the −x, −y and −z faces of cell (i,j,k), only when cell
(i,j,k) has density. A smoky cell's +x face is index (i+1,j,k), which belongs
to the neighbour: at the plume's edge, where that neighbour has no smoke, the
face is missing and reads as 0. Face index n on each axis is never stored at
all, because the file has only indices 0 … n−1. If every cell in the 3×3×3
neighbourhood has smoke, every face and every central difference the metrics
touch (the divergence face stencil, the kinetic energy and vorticity from
cell-centred averages of two faces) is stored in Mantaflow's cache and present
in Ember's. Outflow is measured for the same reason on the z-faces at index
nz − 2, between layers nz − 3 and nz − 2, as the net upwind flux, and drift
is taken over the control volume of layers 0 … nz − 3 (spec §4.3). The reader
fills the missing face n with 0; no metric reads it.

**The open top layer is a density sink.** Whenever the domain is not closed,
the generated script calls `resetOutflow(flags, real=density)` every step
(`smoke_script.h`), which sets the open boundary layer nz − 1 to zero density.
So layer nz − 1 never holds smoke in the cache, face index nz − 1 is never
stored, and no cell in layer nz − 2 is ever a *measured* cell under §4.1's
neighbourhood rule. This was seen on two open-sided probe bakes: layer nz − 1
had 0 smoky cells, and velocity was stored at z index nz − 2 but never at
nz − 1. **Confirmed by Task 6** on `plume` at 32³, all 120 frames of
`tests/bench/mantaflow_scene.py`'s bake: layer 31 (nz − 1) held no cell above
1e-6 in any frame, and no velocity was stored at z index 31. Smoke reaches
layer 30 (nz − 2) at frame 78 (36 cells) and fills 419 cells of it by frame
120; in every one of those 43 frames each smoky cell of layer 30 has velocity
stored at its z-face index 30, and no other cell of that layer does.

**Tiles can drop values from the other grids.** In a domain-filling test
(32³, `fill=1 volume=1 alpha=0 beta=0.5`, frame 3), density was constant over whole 8³ blocks and was
stored as `Node3` tiles. Temperature then lost those blocks (22,904 of 27,000
values stored, and the rest read as 0), but velocity kept all 27,000.
OpenVDB's `clip` against a tiled mask evidently keeps only voxel-level
values, so a grid that is itself tiled there loses them. The bench scenes do
not make whole 8³ blocks of exactly equal density outside the emitter, but at
256³ the emitter's interior might. The reader must not silently use zeros
where a value is missing. The first design compared each grid's value count
with density's; the next paragraph shows why that was replaced by a list of
the missing cells and a wall-face rule.

**Zero velocity at a collider is left out, not lost.** The first full
`just bench` stopped on `plume_collider` 64³: frame 39 stored 9,021 velocity
values against 9,025 density values. The writer's `copyFromDense` stores no
value equal to the background, and velocity's background is (0, 0, 0). A
probe over all 120 frames of that bake found cells with density but no
velocity in 82 frames (39 to 120, 4 to 14 a frame, 659 cell-frames, 14
distinct cells). Every one is a collider cell of `Scene::solid_mask` (6
cells) or one of their 26 neighbours (8 cells), all on the sphere's upper
+x, +y side at z index 26 to 31, where a cell's −x, −y and −z neighbours
are all inside the sphere. For every one of the 659, each of the three
faces the value holds is a wall face by the mask: the cell or its −axis
neighbour is solid. Density was never stored as tiles in any frame of either
bake, so the tile mechanism above cannot apply, and no velocity value of
exactly (0, 0, 0) was stored anywhere. The `plume` 64³ bake had no such
cell in any frame. So the reader lists the cells, not the counts, and the
benchmark accepts a missing cell only by that wall-face rule: on every axis,
the cell or its −axis neighbour is solid by the mask (a neighbour outside the
domain is not solid). Every face a metric reads lies between two fluid cells
by the same mask, so an accepted value is never read, whatever the reason it
was left out. A cell merely next to the collider is not enough, since its
faces between fluid cells are read. Any other missing cell is still an error
that names the frame and the cell. Mantaflow's obstacle is not
exactly the mask: in 3,940 cell-frames of the same bake, velocity was
stored, and non-zero, where the mask makes all three faces walls, mostly on
the sphere's underside (z index 19).

## Velocity location and units

**Faces.** Three pieces of evidence:

- The grid's class metadata is `staggered` (`GRID_STAGGERED`). Mantaflow's
  writer sets that class for every `MACGrid` (`iovdb.cpp`, `writeObjectsVDB`).
- The writer copies the MAC grid's dense array index for index into the VDB
  grid (`exportVDB`, through `tools::Dense` and `copyFromDense`); it does not
  resample to centres.
- In Mantaflow's MAC convention, component x of `vel(i,j,k)` is on the face
  between cells i−1 and i (`KnAddBuoyancy` in `extforces.cpp` averages
  `factor(i,j,k)` and `factor(i,j,k−1)` for the z component). That is
  Ember's face convention.

So index (i,j,k) holds the −x, −y and −z faces of cell (i,j,k), and the cache
stores it only when that cell has smoke. A smoky cell's + faces can therefore
be missing at the plume's edge, and face n on each axis is never stored. That
is why the metrics measure only cells whose whole 3×3×3 neighbourhood is
smoke, and outflow uses face nz − 2 (see Clipping).

The brief's extent experiment cannot separate faces from centres. Velocity is
clipped to density's voxels, so the stored extent is always density's (32³,
`init_vel=1,0,0`, frame 1: both span (12,12,1)–(19,19,8)).

**Units: u[m/s] = u_stored · dx / 0.4**, that is 2.5 · dx · u_stored. By the
generated script's scale factors, a Mantaflow time unit is 0.4 s at any fps,
and velocity is in cells per time unit. `fluid.cc` scales a flow's initial
velocity by `n / L · 1 / (25 · 0.1)` into those units.

Confirmed by experiment (64³, `behavior=GEOMETRY volume=1 alpha=0 beta=0.5
heat_fill=1`): a smoke blob inside a domain of uniform heat rises under a
nearly uniform velocity. Its density centroid's speed from frame f to f+1 is
compared with the density-weighted cell-centred vertical velocity at frame f,
converted by the rule above. Density is advected with the velocity that the
previous frame saved. The ratio rises from 0.83 while the blob still sits
near the floor to **0.988, 1.001 and 0.991 over frames 7→8, 8→9 and
9→10**, once the velocity over the blob is uniform. Mass is conserved to 0.3% over the run.

(The earlier puzzling attempt, a domain-filling flow with an initial velocity,
is explained by `setInitialVelocity` in `extforces.cpp`. It only raises face
velocities towards the target, and the pressure solve then removes most of
it. That makes it a poor units test.)

## Index space

- Index (0,0,0) is the cell whose minimum corner is the domain's minimum
  corner. Cell i spans [i·dx, (i+1)·dx] in the domain's local frame. Evidence:
  the sphere flow of radius 0.2 at x = 1.0 in a 2 m domain at 32³ fills
  x indices 12–19, symmetric about index 16 = 1.0 m / 0.0625 m.
- Indices run 0 … n−1 on each axis for a cubic domain of resolution n. There
  are no padding cells in the file; Mantaflow's one-cell boundary layer is
  part of the n cells (a domain-filling flow fills indices 1 … n−2 only).
- The VDB transform is a `UniformScaleMap` with voxel size dx and **no
  translation**, so the file's world coordinates put voxel centres at i·dx,
  half a cell off the domain. Readers should use index space, not the VDB
  transform.

## Timing

Each frame's data file is written when that frame finishes, so the per-frame
time is the difference between consecutive `fluid_data_NNNN.vdb` modification
times. Frame 1 has no predecessor and is excluded, as the spec's frame-time
rule already says. With `timesteps_min = timesteps_max = 1` and adaptive
time steps off, the generated script sets `timestepMin = timestepMax =
frameLength`, so there is one solver step per frame.

## Parameter mapping

**Final signatures** of `tests/bench/mapping.py`, for Task 6:

```python
def buoyancy(ember_density: float, ember_temperature: float, domain_size: float, gravity: float) -> tuple[float, float]
    # (alpha, beta); domain_size = the longest side in m, gravity = |g| in m/s² (Blender default 9.81)
def inflow(density_rate: float, temperature_rate: float, fps: float) -> tuple[float, float]
    # the flow's (density, temperature) = (density_rate / fps, temperature_rate * 24 / fps);
    # needs use_absolute = False, surface_distance = 0, volume_density = 1;
    # raises ValueError if density leaves [0, 1]
def vorticity(ember_confinement: float, fps: float) -> float
def wind(wind_velocity: tuple[float, float, float], wind_rate: float, fps: float, domain_size: float) -> tuple[float, float, tuple[float, float, float]]
    # (strength, flow, unit direction); (0.0, 0.0, (0, 0, 1)) for wind_rate 0 (no field)
def rotation_to(direction: tuple[float, float, float]) -> tuple[float, float, float, float]
    # (w, x, y, z) turning the field's local +z onto direction
```

The brief's signatures lacked the domain size and gravity (`buoyancy`) and
the fps (`inflow` has it; `vorticity` had dx instead, which cancels; `wind`
had none). `vorticity` and `wind` assume one solver step per frame, as every
bench scene uses. 2b-3c Task 7 replaced `wind(ember_accel, fps)` with the
ambient-airflow form above; it takes the domain size too, since the field's
drag sees the velocity as a fraction of the domain per time unit.

Unit conversions used below, from the generated script: with domain size
L (the longest side) and resolution n, a Mantaflow time unit is 0.4 s at any
fps (`frameLengthRaw = 0.1 · 25`), a frame is 2.5 / fps units long, and
gravity is converted to cells per unit² by `scaleAcceleration = (n / L) · 0.4²`.

| Quantity | Ember | Mantaflow | Conversion | Source | Confirmed |
|---|---|---|---|---|---|
| Buoyancy, heat | `buoyancy_temperature` β: upward accel. β·T | domain `beta` | `beta = β · L / |g|` | `smoke_script.h` divides `beta` by L; `addBuoyancy` adds −g·dt·coef | yes, below |
| Buoyancy, density | `buoyancy_density` α: downward accel. α·ρ | domain `alpha`; positive lifts density | `alpha = −α · L / |g|` | as above | yes: sign and size, below |
| Gravity | none (buoyancy is its own term) | scene gravity × `effector_weights.global_gravity` | keep the defaults, (0, 0, −9.81) and 1, and pass 9.81 | `fluid.cc` `update_final_gravity` | — |
| Vorticity | `vorticity` ε, 1/s: Δu = h·ε·dx·(N×ω) | domain `vorticity`: Δu = v·(dt / frame)·(N×ω), grid units | `vorticity = ε / fps` (dx cancels) | `extforces.cpp` `KnConfForce`; `smoke_script.h` | no: bench ε = 0, so 0 |
| Wind | ambient airflow: u += (w − u)(1 − e^{−rate·h}) in every cell (`wind_velocity` w, `wind_rate`) | `WIND` field, strength S and `flow` F: one frame adds 0.2·S − 0.08·fps·F·u / L (m/s) | with k = 1 − e^{−rate / fps}: `F = 12.5 · k · L / fps`, `S = 5 · k · |w|`; `plume_wind` (1 m/s, 1/s, 24 fps, L = 2 m) gives S = 0.204053, F = 0.042511 | `effect.cc` `do_physical_effector` (S·nor ÷ fps, minus F·vel); `fluid.cc` `update_effectors_task_cb` (vel = grid velocity × 1/n, result × 0.2, clamp ±1); then `scaleSpeedFrames` and `addForceField` | yes, below (Wind as ambient airflow) |
| Wind falloff | none | `falloff_type = "SPHERE"`, `falloff_power = 0`, `use_min_distance = use_max_distance = False`, `z_direction = "BOTH"` | power 0 makes the falloff 1 everywhere, for the force and for the `flow` drag (`flow_falloff`) | `effect.cc` `effector_falloff`, `falloff_func` | yes (used in the checks) |
| Wind direction | the acceleration vector | the field object's local +z | object rotation `rotation_to(direction)` (quaternion); set `rotation_mode = "QUATERNION"` first, or `rotation_quaternion` is ignored | `effect.cc`, `PFIELD_WIND` uses `efd->nor` | yes for +x |
| Inflow, density | adds `rate · occupancy · h` each substep | with `use_absolute = False`, adds `density · emission` once per frame and clamps to [0, 1]; with `use_absolute = True` (Blender's default) it holds the value instead | `use_absolute = False`, `density = density_rate / fps` | `fluid.cc` `apply_inflow_fields` (and the reset of the inflow grids to the current grids before emission); `initplugins.cpp` `applyEmission` | yes: mass 0.97–1.03 of Ember's, below |
| Inflow, temperature | adds `rate · occupancy · h` each substep | raises heat to `temperature` in the emitter (`ADD_IF_LOWER`), never above it, in both modes | `temperature = temperature_rate · 24 / fps`: Ember's emitter-centre value at frame 24 | `fluid.cc` `ADD_IF_LOWER`, `apply_inflow_fields` | yes: held at 1.000, below |
| Emitter volume | a filled sphere with a one-cell soft edge | a mesh flow emits a **shell** by default (`volume_density = 0`), plus a falloff out to `surface_distance` cells outside the mesh (default 1.0) | `volume_density = 1`, `surface_distance = 0` | `fluid.cc` `sample_mesh`; a 32³ bake showed a hollow emitter; the default `surface_distance` added 33% mass | yes, below |
| Emitter activity | `active_frames` [1, 60] | no frame-range setting | key `flow_settings.use_inflow` ("Use Flow") True at frame 60 and False at frame 61; cache frame 61 is then the first without emission, as in Ember | RNA listing; bake | yes: see Emission switch-off, below |
| Substeps | `max_substeps` (preview: 1) | `timesteps_min = timesteps_max` | equal | generated script | yes (script) |
| Boundaries | closed sides and floor, open top | all six sides **open** by default (`xXyYzZ`) | `use_collision_border_{front,back,left,right,bottom} = True`, `top = False` (probe `bench=1`); per side, left/right are −x/+x, front/back −y/+y, bottom/top −z/+z | generated script: `boundConditions` | yes (script; Task 6 opened −x, +y and −z alone and got `'xYz'`) |
| Advection | MacCormack | `advectSemiLagrange(order=2)`, Mantaflow's MacCormack | none needed | generated script | yes (script) |
| Pressure | 160 red-black Gauss–Seidel iterations | multigrid-preconditioned CG to a tolerance | none possible; the table must say so | generated script | — |

### Experiments behind the table

All baked with the probe. Velocities are converted with the units rule above.

- **Heat buoyancy.** A domain filled with heat 1 (`fill=1 volume=1 alpha=0
  beta=0.5`), all sides open. The source predicts Δw per frame of
  |g| · (beta / L) / fps = 0.1022 m/s, which is 1.308 in grid units at 64³.
  Measured at the domain centre: 1.278, 1.277 and 1.272 per frame at 64³ (97.7%),
  and 0.624 per frame at 32³ against 0.654 (95.4%). The shortfall halves when
  the resolution doubles, so it comes from the boundary cell layer, not the
  formula.
- **Density buoyancy.** The same with density 1, no heat (`fill=1 volume=1
  temperature=0 alpha=0.5 beta=0`) at 32³: w = +0.6237, then +1.2466, which is
  **upward** and identical to the heat case. So `alpha` lifts density, and
  Ember's `buoyancy_density` α maps to `alpha = −α · L / |g|`.
- **Inflow.** Ember's bench `plume` at 64³ (`Scene::plume`, read back): the
  density at the emitter centre (mean of the 2×2 cells about x = y = 1 m in the
  layer at z = 0.3 m) is 0.25, 0.50 and **0.99** at frames 6, 12 and **24**,
  so rate · t while the plume is still. Three Mantaflow modes, all 64³ with
  `bench=1 volume=1 alpha=0 beta=0.2039 temperature=1`, against Ember. Mass is
  total density · dV and cz is the density centroid's height:

  | Frame | Ember mass / cz | absolute, density 1 | additive, density 1/24, `surface_distance` 1.0 | **additive, density 1/24, `surface_distance` 0** |
  |---|---|---|---|---|
  | 12 | 0.0169 / 0.310 | 0.0527 (×3.12) / 0.333 | 0.0225 (×1.33) / 0.342 | **0.0163 (×0.97) / 0.333** |
  | 24 | 0.0338 / 0.371 | 0.0808 (×2.39) / 0.422 | 0.0452 (×1.34) / 0.476 | **0.0338 (×1.00) / 0.457** |
  | 36 | 0.0510 / 0.512 | 0.1172 (×2.30) / 0.558 | 0.0668 (×1.31) / 0.639 | **0.0503 (×0.99) / 0.604** |
  | 48 | 0.0672 / 0.658 | 0.1426 (×2.12) / 0.714 | 0.0891 (×1.33) / 0.781 | **0.0678 (×1.01) / 0.742** |
  | 60 | 0.0855 / 0.781 | 0.1740 (×2.04) / 0.872 | 0.1161 (×1.36) / 0.955 | **0.0884 (×1.03) / 0.925** |

  (A `surface_distance` of 0.5 gave ×1.12 to ×1.17.) **The mapping is the
  additive mode with `surface_distance = 0`.** In this 64³ calibration, mass
  matches Ember's within 3% from frame 12 to 60. Across the benchmark's runs
  (`docs/bench/results.md`, Heat), it matches within 4% over frames 12–24
  only; by frame 60 the solvers' advection has moved it apart. The emitter-centre density rises as Ember's does:
  0.500 at frame 12 in both.

  Temperature cannot be matched the same way. Mantaflow raises the emitter's
  heat to the flow's temperature and never above it (1.000 in every checked
  frame), while Ember's grows as rate · t. The heat is set to Ember's frame-24
  value, so Mantaflow's emitter is hotter than Ember's before frame 24 and
  cooler after it. That is why Mantaflow's plume still rises faster: its
  centroid is 0.925 m against 0.781 m at frame 60 (0.872 m with the old
  absolute mapping). The heat drives the buoyancy, so the plume-height metrics
  carry this difference, and the summary must say so.

  **Temperature ramp (Task 6): rejected.** Keying the flow's `temperature`
  on every frame to Ember's growing heat, `temperature_rate · f / fps` at
  frame f through frame 60, brings the plume height closer to Ember's, but
  loses mass. Same 64³ `plume`, Ember regenerated with
  `benchmark ember plume 64`:

  | Frame | Ember mass / cz | fixed heat 1.0 (kept) | ramp to f / 24 |
  |---|---|---|---|
  | 24 | 0.0338 / 0.371 | 0.0338 / 0.457 | 0.0326 (−3.5% of fixed) / 0.356 |
  | 60 | 0.0855 / 0.781 | 0.0884 / 0.925 | 0.0764 (−13.6% of fixed) / 0.889 |

  The ramp is closer at both frames (off by 0.015 m and 0.108 m, against
  0.086 m and 0.144 m), but it moves mass by more than the 3% the plan allows,
  so the scene keeps the fixed heat. The probable cause is the density clamp:
  with a cooler emitter early on, smoke leaves the emitter more slowly, the
  emitter centre reaches 0.998 by frame 24 (0.526 with fixed heat), and
  additive emission into cells already near 1 is clamped away. Ember does not
  clamp.

  Mantaflow computes the inflow grids once per frame and copies them into the
  domain on every solver step (`applyEmission`, absolute copy). So with more
  than one step per frame the density would not be added once per step, and
  this mapping holds only for one step per frame.
- **Emission switch-off (Task 6).** `tests/bench/mantaflow_scene.py` keys
  `use_inflow` on at frame 60 and off at 61. `plume` at 32³, 120 frames,
  against two controls from the same script (emission always on; emission
  on frames 1–30 only). Mass is total density · dV; the emitter value is the
  mean density of the 2×2 cells about x = y = 1 m in the layer at z = 0.3 m:

  | Frame | [1, 60]: mass / emitter | always on: mass / emitter |
  |---|---|---|
  | 59 | 0.09302 / 0.590 | 0.09302 / 0.590 |
  | 60 | 0.09526 / 0.587 | 0.09526 / 0.587 |
  | 61 | 0.09629 / **0.543** | 0.09776 / 0.584 |
  | 62 | 0.09675 / 0.499 | 0.09983 / 0.582 |
  | 65 | 0.09839 / 0.373 | 0.10630 / 0.579 |
  | 70 | 0.10016 / 0.206 | 0.11674 / 0.582 |
  | 80 | 0.10367 / 0.065 | 0.13861 / 0.597 |

  The two runs are identical to frame 60 and part at 61: the emitter's
  density starts to drain at 61 and keeps falling, so frame 61 is the first
  frame without emission, as in Ember. Mass still rises after 61, by
  0.00046 a frame against 0.0021 a frame while emitting, until the smoke
  reaches the top at about frame 80 (0.1037 at 80, then 0.0328 at 120). That
  rise is the advection, not emission: Ember's `plume` at 32³ rises the same
  way (0.08585 at 60, 0.08636 at 61, 0.08700 at 63, 0.09242 at 80), and the
  [1, 30] run *loses* mass after its cut-off (0.04439 at 30, 0.04424 at 31,
  0.03801 at 50), with its emitter draining from 0.540 to 0.500 at frame 31.
- **Wind strength (2b-3, superseded by 2b-3c's ambient airflow below).** A domain filled with smoke (`fill=1 volume=1 alpha=0
  beta=0`), all sides open, at 32³, with `wind = S` along +x. With
  S = 0.1042 (0.5 m/s² at 24 fps), the centre's u rose 0.0201, 0.0402 and
  0.0599 m/s by frames 2, 3 and 4, against 0.0208 per frame (96.5%, the
  same boundary shortfall as buoyancy). The first frame gets no force, because
  the effectors are sampled before that frame's emission, when the domain
  holds no smoke yet. My first reading of the source, S = a / (0.2 · fps²),
  missed the ÷ fps in `do_physical_effector` and was 24× too weak.
- **Wind check with an emitter** (2b-3's uniform-acceleration wind,
  superseded below; replaces the brief's no-emitter check).
  The bench `plume_wind` at 64³ in both solvers. Mantaflow ran with the
  additive inflow mapping (`bench=1 volume=1 surface=0 absolute=0
  density=0.04167 temperature=1 alpha=0 beta=0.2039 wind=0.1042`), and a
  first run with the old absolute mapping is kept for comparison. The
  horizontal drift is the density centroid's x minus the no-wind plume's:

  | Frame | Ember drift (m) | Mantaflow, additive (m) | Mantaflow, absolute (m) |
  |---|---|---|---|
  | 12 | +0.005 | +0.017 | +0.015 |
  | 24 | +0.022 | +0.088 | +0.068 |
  | 36 | +0.061 | +0.189 | +0.155 |
  | 48 | +0.165 | +0.293 | +0.259 |
  | 60 | +0.368 | +0.413 | +0.377 |

  The sign is right (+x) and the size is close by frame 60 (+0.413 m against
  +0.368 m). Mantaflow drifts earlier, probably because its hotter early
  emitter lifts smoke sooner into the region where the wind has acted longest
  (see Inflow).

### Wind as ambient airflow (2b-3c Task 7)

Ember's wind became an ambient airflow the air relaxes towards: each substep,
u += (w − u)(1 − e^{−rate·h}) on every non-wall face, with `wind_velocity` w
and `wind_rate`. Blender's `WIND` field has a matching term, its `flow`
setting, a drag towards the field's own velocity.

**Formula, from the source.** `effect.cc` `do_physical_effector`: a `WIND`
field's force is strength · falloff · its object's +z (`efd->nor`), divided
by `vel_to_sec` (the fps, from `pd_point_from_loc`), and with `flow` ≠ 0 it
also adds −flow · falloff · the point's velocity. `fluid.cc`
`update_effectors_task_cb` passes the cell's velocity as the grid velocity
times `fds->dx`, which is 1 / n (`fds->dx = 1.0f / res`), so for u in m/s
(the units rule, u = u_grid · (L / n) / 0.4) the field sees u · 0.4 / L. The
result is scaled by 0.2 and clamped to ±1. The script multiplies the force by
`scaleSpeedFrames` = (n / L) · fps / 2.5 and `addForceField` adds it once a
step, which is F · fps m/s a frame for a force F (as the 2b-3 strength
experiment found). One frame therefore adds

  Δu = 0.2 · S − 0.08 · fps · F · u / L   (m/s),

a relaxation towards 2.5 · S · L / (fps · F) with blend 0.08 · fps · F / L a
frame. Matching Ember's per-step blend k = 1 − e^{−rate / fps} and target w:

  **F = 12.5 · k · L / fps,   S = 5 · k · |w|**,

turned onto w by `rotation_to`. The drag acts on all three components of the
smoke's velocity, as Ember's relaxation does. The clamp never binds:
0.2 · S / fps = k · |w| / fps is 0.0017 for `plume_wind`. `mapping.wind`
returns no field (strength and flow 0) when `wind_rate` is 0, and the scene
script then adds none. For `plume_wind` (w = 1 m/s along +x, rate 1/s,
24 fps, L = 2 m): k = 0.040811, **S = 0.204053, F = 0.042511**.

**Bakes.** Probe, domain filled with smoke at rest (`fill=1 volume=1 alpha=0
beta=0`), `wind=0.204054 wind_flow=0.042511` along +x, domain-centre u
(mean of the cell-centred u over the 2×2×2 cells about the centre),
converted by the units rule. Frame 1 gets no force, as before, so the
prediction is u(f) = 1 − (1 − k)^{f−1}.

- **All six sides open (the brief's setup), 32³, 96 frames.** u = 0.039,
  0.147, 0.361, 0.554, 0.704, 0.736 at frames 2, 5, 13, 25, 49, 96, against
  0.041, 0.154, 0.394, 0.632, 0.865, 0.981. It follows the prediction at
  first (96%) and then levels off near 0.74 m/s. The profile along the
  centre line in x was not uniform (at frame 96: 0.63 near −x, 0.745 in the
  middle, 0.68 near +x), probably because the unforced, smoke-free boundary
  layer on the open ±y and ±z sides lets the flow spread and shear. Closing
  those sides removes it (next bake).
- **Only ±x open, the other four sides walls** (`closed=1 open=left,right`),
  as in `plume_wind`. The interior velocity is then uniform to three places
  along x, y and z in every frame checked, so each frame is one number. At
  32³ it reaches 0.683 m/s by frame 96, at 64³ 0.700 by frame 72.
- **Why it levels off below 1 m/s: the open inlet, not the field.** The same
  domain with `flow` 0 (`wind=0.204054` alone, a constant force) should
  gain 0.2 · S = 0.0408 m/s a frame for ever (0.0388 is measured in the
  first frame, the resolution shortfall below). Its gain instead falls as
  u²: the
  shortfall is 0.0213 · u² a frame at 32³ and 0.0227 · u² at 64³, for u
  above 0.5, against h / L = 0.0208. That is still air entering through the
  open −x face: the smoke-free inlet layer gets no force, and
  incompressibility spreads its momentum deficit, u · (u h / L), over the
  whole channel. It does not depend on the resolution, and it is the same
  with or without `flow`.
- **The flow drag, net of the inlet.** Subtracting the force-only run's gain
  at the same u from the wind run's leaves the drag alone: −k_eff · u with
  **k_eff = 0.03895 a frame at 32³ (95.4% of k) and 0.04064 at 64³ (99.6%)**.
  The force term's first-frame gain is 0.03880 (95.1% of 0.2 · S) and
  0.03981 (97.5%), the same resolution shortfall as the buoyancy
  experiments. So the target is a / k_eff = **0.996 m/s at 32³ and 0.979 m/s
  at 64³**, with a **time constant of 1.05 s and 1.00 s** (Ember's: 1 s,
  1 m/s). The drag scales with `flow`: with S and F both doubled
  (`wind=0.408108 wind_flow=0.085022`, 32³), k_eff = 0.0777 (95.2% of 2k).

So Blender's `flow` reproduces Ember's relaxation: the right target speed
and time constant within 5% at 32³ and 2% at 64³. What the fill bakes reach,
0.68 to 0.74 m/s, is the inlet's drag on a domain whose every cell moves; it
is a property of Mantaflow's open face, not of the mapping.

**Check with an emitter: `plume_wind` at 64³.** Mantaflow through
`tests/bench/mantaflow_scene.py` (scene JSON from `benchmark scene-json`),
Ember through the same `Scene::plume_wind(64)` that `benchmark ember
plume_wind 64` runs (at 4345f9a; its mass and centroid height matched the
benchmark's CSV to every printed digit). Drift is the density centroid's x
minus the no-wind `plume`'s, which is 1.0000 m in both solvers:

| Frame | Ember drift (m) | Ember mass | Mantaflow drift (m) | Mantaflow mass |
|---|---|---|---|---|
| 12 | +0.073 | 0.0169 | +0.031 | 0.0164 |
| 24 | +0.246 | 0.0337 | +0.145 | 0.0341 |
| **30** | **+0.355** | 0.0421 | **+0.219** | 0.0432 |
| 36 | +0.474 | 0.0506 | +0.296 | 0.0521 |
| 48 | +0.564 | 0.0475 | +0.448 | 0.0696 |
| **60** | **+0.517** | 0.0385 | **+0.564** | 0.0799 |

Both drift +x. The drifts differ early (Ember's is 1.6× Mantaflow's at
frame 30) and meet by frames 48–60. Ember's smoke moves sooner, because its
whole domain of air is set moving, while Mantaflow drags only the smoke and
the smoke must push the still air around it. From frame 36 Ember's smoke
leaves through the +x face (its mass falls while it is still emitting), so
its centroid then stops describing the whole plume, and its frame-60 drift
is below its frame-48 one. Mantaflow keeps most of its smoke to frame 60 and
loses it later (0.0319 at frame 80). The earlier, uniform-acceleration wind
drifted +0.368 m (Ember) and +0.413 m (Mantaflow) by frame 60.

### Force fields act only on smoke

Mantaflow applies a force field only in cells that hold smoke:
`update_effectors_task_cb` in `source/blender/blenkernel/intern/fluid.cc`
skips every cell whose density (or fuel, when fire is active) is below
`FLT_EPSILON`, and cells inside obstacles. This holds for the `flow` drag
too. Ember's air relaxes towards the ambient airflow in every cell.
**`plume_wind` is kept as it is.** Its results must say that Mantaflow's
wind drags only the smoke towards the airflow, while Ember's moves all the
air, so the drift compares shape, not a matched force field. The wind force
is also computed once a frame and added once per solver step with no dt, so
the mapping holds only for one step per frame, which every bench scene uses.

### Other solver notes

- **Buoyancy in Mantaflow is a true acceleration.** `addBuoyancy` multiplies by
  the solver's dt, and with the generated `scale=False` it does not divide by
  dx. The face value uses the mean of the two cells either side, as Ember's
  does.
- **The domain's fire fields are on.** The generated script had
  `using_fire = True` for a smoke-only flow. The fuel grid is zero, so burning
  does nothing, and the flame-vorticity term that `vorticityConfinement` adds
  (`strengthCell = fuel · flame_vorticity`) is zero.

## Fire (2b-4 Task 1)

Probes for Ember's fire (spec
`docs/superpowers/specs/2026-09-25-ember-fire-2b4-design.md` §6.1). Blender
5.2.2, 2026-09-25, the same machine under load. Sources read, never copied:
`extern/mantaflow/preprocessed/plugin/fire.cpp` (`KnProcessBurn`,
`KnUpdateFlame`) and `plugin/extforces.cpp` (`KnConfForce`) in
github.com/blender/blender at `main`, which match
`source/plugin/fire.cpp` and `extforces.cpp` in github.com/tum-pbs/mantaflow
(Apache-2.0); `source/blender/blenkernel/intern/fluid.cc`
(`apply_inflow_fields`); and the generated script (`script=1`).

### Commands

With `B=/Applications/Blender.app/Contents/MacOS/Blender` and
`P="$B --background --factory-startup --python-exit-code 1 --python tests/bench/probe_mantaflow.py --"`,
per-frame sums and maxima come from
`cargo run -p elements-ember --example cache_sums -- DIR RES FRAMES GRID...`
(`DIR` is the one that holds `data/`):

| Bake | Command |
|---|---|
| inventory, smoke-style cache | `$P $S/inv 32 10 fire=1 bench=1 volume=1 surface=0 absolute=0 script=1` |
| inventory, resumable cache | the same plus `resumable=1` (`$S/inv_r`) |
| emission | `$P $S/emit 32 10 fire=1 bench=1 volume=1 surface=0 absolute=0 resumable=1 burning=0`, and again with `clipping=0` |
| burn, smoke and heat | `$P $S/burn 32 14 fill=1 behavior=GEOMETRY fire=1 fuel=1 closed=1 alpha=0 beta=0 burning=0.75 flame_smoke=1 volume=1 resumable=1` |

Which cells each grid stores was read with the `openvdb` module in Blender's
own Python (`import openvdb`, `copyToArray`, `citerOnValues`), a scratch
check that is not part of the repository.

### Blender's defaults and ranges

Read back from the inventory bake and from RNA (`bl_rna.properties`):

| Setting | Default | Hard range |
|---|---|---|
| domain `burning_rate` | 0.75 | [0.01, 4] |
| domain `flame_smoke` | 1.0 | [0, 8] |
| domain `flame_vorticity` | 0.5 | [0, 2] |
| domain `flame_ignition` | 1.5 | [0.5, 5] |
| domain `flame_max_temp` | 3.0 | [1, 10] |
| flow `fuel_amount` | 1.0 | [0, 10] |

**Blender clamps a value outside the range without an error**: the emission
probe asked for `burning=0` and read back 0.01. `mapping.fire` therefore
raises for a value outside these ranges.

### Grid inventory: fuel and react need a resumable cache

The generated script puts `flame` in the final data dictionary but `fuel`,
`react`, `fuel_inflow` and `react_inflow` in the *resume* dictionary, which
is written only when the cache is resumable. At frame 5 of the 32³
inventory bakes:

| Grid | default cache | `cache_resumable = True` |
|---|---|---|
| `density`, `temperature`, `velocity` | yes (656 voxels) | yes (656) |
| `flame` | yes (444) | yes (444) |
| `fuel`, `react` | **absent** | yes (444 each) |
| `fuel_inflow`, `react_inflow` | absent | yes (348 each) |
| `heat` | absent (heat is written as `temperature`) | absent |
| other resume grids | absent | `density_inflow`, `temperature_inflow`, `emission`, `flags`, `phi_*`, `velocity_previous` |

The resumable bake is the same simulation: its density, flame and
temperature sums equal the default bake's to every printed digit in all ten
frames. **The benchmark's fire scene must set `cache_resumable = True`** to
compare fuel. The Rust reader asks for grids by name, so the extra grids do
no harm.

### Emission: fuel as density's, react as a blend

Fuel is emitted like density's additive inflow. `fluid.cc` adds
`fuel_amount · emission` to the cell once per frame, clamped to **[0, 10]**
(not [0, 1] as density is), and a `FIRE` flow emits no density. Its
temperature is still raised to the flow's (`ADD_IF_LOWER`). Measured in the
emission bake (`burning` at its minimum, 0.01): `fuel_inflow` at frame 1 is
140.000000 over the sphere's 140 cells, max 1.000000; and
`fuel_inflow(f) − fuel(f − 1)` is 140.0000 in every frame 2–10 of the
`clipping=0` bake. Cells keep accumulating: `fuel_inflow` reached 7.8 per cell by
frame 8 at `fuel_amount` 1.

**React is not fuel emission added to react (spec §3.2 step 1 is
contradicted).** `fluid.cc` sets, where `fuel_in > FLT_EPSILON` and
`value > react`:

    value  = 1 − (1 − emission)²
    f      = fuel_flow / fuel_in        (fuel_in: the cell's fuel after this emission)
    react' = clamp(value · f + (1 − f) · react, 0, value)

so react is a fraction in [0, 1] that blends towards `value` by the share of
the cell's fuel that is fresh. Measured: at frame 1 `react_inflow` = 1 on
all 140 cells (sum 140.000000); at frame 2, in the emission bake, the
emitter-centre cell has fuel 0.998958 and react 0.998958 after frame 1, fuel
1.998958 after emission, and `react_inflow` max 0.999479, which is
`0.50026 · 1 + 0.49974 · 0.998958` (the additive rule would give 1.998958).
In the default-rate inventory bake, `react_inflow` max is 0.962525 at frame
2 against the formula's 0.96253. Over ten frames at fuel_amount 1 the react
maximum stays below 1 while `fuel_inflow` reaches 7.4. Under the spec's additive rule
react would equal fuel, `flame = √react` would exceed 1, and the burn would
write a temperature above `max_temperature`. **Ember must follow
Mantaflow's blend** (the spec allows this), in Ember's terms per substep:
with Δ = the fuel emitted into the cell, fuel' = fuel + Δ and `value` from
the cell's occupancy, `react += (Δ / fuel') · (value − react)` where
fuel' > 1e-6 and value > react. With a whole-cell occupancy of 1, value = 1.

### Burn rate

The fill bake (a domain-filling `GEOMETRY` fire flow, fuel 1, all walls
closed, no buoyancy, so nothing moves). Fuel per cell, frame 1 to 13:
1 − 0.078125 · f exactly (0.921875, 0.843750, 0.765624, …, 0.062497 at frame
12, then 0). So **d = 0.078125 a frame at `burning_rate` 0.75 and 24 fps;
d / 0.75 = 0.104167 = 2.5 / 24**, the frame length in time units, as
`KnProcessBurn`'s `burningRate · dt` says. Ember's rate per second is
d · fps = 0.75 / 0.4 = **1.875**. React equals fuel in every frame of this
bake (react starts at 1 and is scaled by fuel' / fuel), and flame is its
square root: 0.960143 at frame 1, 0.249994 at frame 12.

### Smoke and heat

In the same bake, per interior cell (max = every interior cell; nothing
moves). The spec's `Δdensity = (0.5 + 0.5 · max(1 − fuel, 0)) · (fuel − fuel') · 0.1 · flame_smoke`
predicts, with the recorded fuels:

| Frame | fuel before | predicted Δ | cumulative | measured density |
|---|---|---|---|---|
| 1 | 1.000000 | 0.00390625 | 0.00390625 | 0.003906 |
| 2 | 0.921875 | 0.00421143 | 0.00811768 | 0.008118 |
| 3 | 0.843750 | 0.00451660 | 0.01263428 | 0.012634 |
| 13 | 0.062497 | 0.00605440 | 0.07307 | 0.073071 |

(frame 13 burns only the 0.0625 left.) The density clamp to [0, 1] in
`KnProcessBurn` never binds here.

Heat: `(1 − flame) · 1.5 + flame · 3.0` gives 2.940215 at frame 1
(flame 0.960143) and 1.874991 at frame 12 (flame 0.249994); the measured
temperature maxima are 2.940215 and 1.874991. At frame 13, with no fuel,
flame is 0 and temperature stays at 1.874991: the overwrite happens only
where flame > 0, as the spec says.

### Flame vorticity: `KnConfForce` adds

`KnConfForce` computes `if (strGrid) str += (*strGrid)(i, j, k);` and then
`force = str · (N × curl)`, with no dx. The generated script fills the
strength grid with `fuel · flameVorticity · timestep / frameLengthUnscaled`
(it borrows the `flame` grid for this and recomputes flame after the step)
and passes `vorticity · timestep / frameLengthUnscaled` as the uniform
strength. So the per-cell strength is **a sum**, `vorticity + flameVorticity
· fuel` under the same scaling, which is spec §3.2 step 4's
`ε + flame_vorticity · fuel`. By `mapping.vorticity`'s derivation Ember's
equivalent of Mantaflow's `V` is **V · fps** (1/s per unit fuel): 12 for
Blender's 0.5 at 24 fps. Confirmed from the source, not by a bake.

Mantaflow runs confinement after advection, so the strength uses the
advected fuel; the burn runs before advection, straight after emission.

### The mapping

`mapping.fire(fuel_rate, burning_rate, flame_vorticity, fps)`, for one solver
step per frame and an additive flow:

| Quantity | Ember | Mantaflow | Conversion | Confirmed |
|---|---|---|---|---|
| Fuel emission | `fuel_rate` per second at occupancy 1 | flow `fuel_amount`, added once a frame | `fuel_amount = fuel_rate / fps`, in [0, 10] | yes: 140.000 a frame into 140 cells |
| Burn rate | `burning_rate`, fuel per second | domain `burning_rate`, per time unit (0.4 s) | `burning_rate_M = burning_rate · 0.4`, in [0.01, 4] | yes: 0.078125 a frame at 0.75, 24 fps |
| Flame vorticity | `flame_vorticity`, 1/s per unit fuel | domain `flame_vorticity`, grid units per frame | `flame_vorticity / fps`, in [0, 2] | source only (`KnConfForce` adds) |
| Flame smoke | `flame_smoke` | domain `flame_smoke` | equal | yes: the smoke table above |
| Ignition, max temperature | `ignition_temperature`, `max_temperature` | domain `flame_ignition`, `flame_max_temp` | equal | yes: the heat values above |

`mapping.EMBER_FIRE_DEFAULTS` holds Blender's defaults in Ember's units at
24 fps: `burning_rate` 1.875, `flame_smoke` 1.0, `flame_vorticity` 12.0,
`ignition_temperature` 1.5, `max_temperature` 3.0.

### Differences from Mantaflow (spec §3.3)

- **Density clamp.** `KnProcessBurn` clamps density to [0, 1] in every cell
  on every step (fuel or not). Ember does not clamp.
- **Fuel clamp.** Additive emission clamps a cell's fuel to [0, 10]. Ember
  does not clamp.
- **No colour grids** (`flame_smoke_color`).
- **Units.** Ember's `burning_rate` and `flame_vorticity` are per second;
  Mantaflow's are per time unit and per frame, converted as above.
- **React emission** follows Mantaflow's blend (above), not the spec's
  additive rule.

### What the cache stores where there is fuel

The data file is clipped to density's voxels (`clipGrid = density`, see
Clipping), and that holds for the fire grids too. In every frame of the
inventory and emission bakes, `fuel`, `react` and `flame` were never stored
in a cell without stored density, and velocity was stored on exactly
density's voxels (no cell with one and not the other). So **the cache stores
no velocity, and no fuel, where there is fuel but no density**. Burning
makes density wherever fuel burns, so this loses little: at the default
rate, `fuel_inflow(f) − fuel(f − 1)` was 140.000 in frames 2–9, so no fuel
was dropped. At the minimum rate, where density stays tiny, the saved fuel
at frame 2 was 277.5485 at `clipping` 1e-6 against 277.8176 at 0 (0.1%
lost in cells whose density was below 1e-6). For `check_coverage` (Task 9)
this means the existing rule, velocity against density, still covers every
cell the cache holds; a fuel cell with no density is simply absent, not a
missing velocity.

## 256³ run time

Probe scene (sphere inflow, default buoyancy, open borders), 10 frames,
per-frame time from data-file modification times, frames 2–10:

| Resolution | Median s/frame | Range | Load average at start → end |
|---|---|---|---|
| 64³ | 0.106 | 0.094–0.139 | 23.4 → 23.8 |
| 128³ | 0.542 | 0.520–0.649 | 23.8 → 23.8 |
| 256³ | 4.11 | 3.69–4.43 | 11.2 → 28.3 |

Extrapolated full benchmark (3 scenes × 120 frames; 1 run at 256³, 3 runs at
64³ and 128³): 256³ 1,480 s, 128³ 585 s, 64³ 114 s, about **36 minutes**,
plus about a second of Blender start-up per bake. The machine was loaded (load
average 11–28), so on an idle machine these early frames would run faster.
Frames past 10 are only estimated, because the smoke, and with it the sparse
files, keeps growing. The estimate is under the 90-minute stop line.

## Corrections to the spec

1. **`vdb-rs` 0.6.0 cannot read Mantaflow's velocity grid** (spec §8 risk
   (a)). Resolved: the workspace vendors a patched copy (`vendor/vdb-rs/`).
   The Task 7 reader must add the grid-type check that `vdb-rs` lacks.
2. **The cache holds velocity only where there is smoke** (density above
   `clipping`), and `clipping = 0` does not change that. Since velocity is
   stored per cell but holds that cell's − faces, a smoky cell's + faces can
   be missing, and face n is never stored. Resolved: the §4.1 metrics use
   only measured cells, whose whole 3×3×3 neighbourhood is inside the domain,
   collider-free and above density 1e-6, for both solvers; outflow uses the
   net upwind flux through face nz − 2, over the control volume of layers
   0 … nz − 3, since Mantaflow's open top layer nz − 1 is a density sink
   (`resetOutflow`). `results.md` must say so.
3. **Mantaflow's wind acts only on smoky cells.** `plume_wind` is kept and
   documented as above. The brief's no-emitter wind check cannot work, and an
   emitter check replaces it.
4. **The fixture cannot be under 200 KB.** An uncompressed frame is about
   1 MB at any small resolution. The committed 16³ frame is 967,548 bytes.
5. **`mapping.py` signatures** take the domain size, gravity and fps. See the
   final signatures above.
6. **Blender's defaults differ from the bench's.** All borders are open, and a
   mesh flow emits a shell. The scene script must set both, and must also run
   Blender with `--python-exit-code 1`.
7. **Inflow matches in mass, not in heat.** With `use_absolute = False`,
   `surface_distance = 0` and density = rate / fps, Mantaflow's mass is within
   3% of Ember's in the 64³ calibration (within 4% over frames 12–24 across
   the benchmark's runs). Mantaflow's emitter heat is held at a fixed value, while
   Ember's grows, so the plume heights differ (0.925 m against 0.781 m at
   frame 60). See Inflow.
8. **Tiled density can drop other grids' values** (see Clipping). The reader
   must report it rather than use zeros. As shipped (a2de579, 7fe5d9d), it
   lists every cell with density but no velocity, and `check_coverage`
   accepts one only when all three of its faces are walls by the mask (on
   every axis, the cell or its −axis neighbour is solid), since the writer
   omits those zero velocities at a collider. Any other missing cell is a
   `MissingVelocity` error. Comparing value counts, the first design, failed
   on `plume_collider`.
9. **The brief's `vdb_probe.rs`** formats `VdbLevel` with `{:?}`, which does
   not compile: `VdbLevel` has no `Debug`.
