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
nz − 1, with the density of layer nz − 2, not on face nz. The reader fills
the missing face n with 0; no metric reads it.

**Tiles can drop values from the other grids.** In a domain-filling test
(32³, `fill=1 volume=1 alpha=0 beta=0.5`, frame 3), density was constant over whole 8³ blocks and was
stored as `Node3` tiles. Temperature then lost those blocks (22,904 of 27,000
values stored, and the rest read as 0), but velocity kept all 27,000.
OpenVDB's `clip` against a tiled mask evidently keeps only voxel-level
values, so a grid that is itself tiled there loses them. The bench scenes do
not make whole 8³ blocks of exactly equal density outside the emitter, but at
256³ the emitter's interior might. The Task 7 reader should compare each
grid's value count with density's and report a shortfall, not silently use
zeros.

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
smoke, and outflow uses face nz − 1 (see Clipping).

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
def wind(ember_accel: tuple[float, float, float], fps: float) -> tuple[float, tuple[float, float, float]]
    # (strength, unit direction); (0.0, (0, 0, 1)) for no wind
def rotation_to(direction: tuple[float, float, float]) -> tuple[float, float, float, float]
    # (w, x, y, z) turning the field's local +z onto direction
```

The brief's signatures lacked the domain size and gravity (`buoyancy`) and
the fps (`inflow` has it; `vorticity` had dx instead, which cancels; `wind`
had none). `vorticity` and `wind` assume one solver step per frame, as every
bench scene uses.

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
| Wind | uniform acceleration a, m/s², every cell | `WIND` field, strength S | `S = |a| / (0.2 · fps)`: 0.5 m/s² at 24 fps gives 0.1042 | `fluid.cc` `update_effectors_task_cb` (× 0.2, clamp ±1); `effect.cc` `do_physical_effector` (÷ fps, `vel_to_sec`); then `scaleSpeedFrames` and `addForceField` | yes, below |
| Wind falloff | none | `falloff_type = "SPHERE"`, `falloff_power = 0`, `use_min_distance = use_max_distance = False`, `z_direction = "BOTH"`, `flow = 0` | power 0 makes the falloff 1 everywhere; `flow` (default 1 for WIND) would add drag towards the smoke's own velocity | `effect.cc` `effector_falloff`, `falloff_func` | yes (used in the check) |
| Wind direction | the acceleration vector | the field object's local +z | object rotation `rotation_to(direction)` (quaternion); set `rotation_mode = "QUATERNION"` first, or `rotation_quaternion` is ignored | `effect.cc`, `PFIELD_WIND` uses `efd->nor` | yes for +x |
| Inflow, density | adds `rate · occupancy · h` each substep | with `use_absolute = False`, adds `density · emission` once per frame and clamps to [0, 1]; with `use_absolute = True` (Blender's default) it holds the value instead | `use_absolute = False`, `density = density_rate / fps` | `fluid.cc` `apply_inflow_fields` (and the reset of the inflow grids to the current grids before emission); `initplugins.cpp` `applyEmission` | yes: mass 0.97–1.03 of Ember's, below |
| Inflow, temperature | adds `rate · occupancy · h` each substep | raises heat to `temperature` in the emitter (`ADD_IF_LOWER`), never above it, in both modes | `temperature = temperature_rate · 24 / fps`: Ember's emitter-centre value at frame 24 | `fluid.cc` `ADD_IF_LOWER`, `apply_inflow_fields` | yes: held at 1.000, below |
| Emitter volume | a filled sphere with a one-cell soft edge | a mesh flow emits a **shell** by default (`volume_density = 0`), plus a falloff out to `surface_distance` cells outside the mesh (default 1.0) | `volume_density = 1`, `surface_distance = 0` | `fluid.cc` `sample_mesh`; a 32³ bake showed a hollow emitter; the default `surface_distance` added 33% mass | yes, below |
| Emitter activity | `active_frames` [1, 60] | no frame-range setting | keyframe the flow's emission off after frame 60 (spec §5); Task 6 must verify the switch by a bake. The property is `flow_settings.use_inflow` ("Use Flow", "Control when to apply fluid flow", animatable, default True), found by listing `FluidFlowSettings` RNA in 5.2.2 | RNA listing | no: exists, not yet verified by a bake |
| Substeps | `max_substeps` (preview: 1) | `timesteps_min = timesteps_max` | equal | generated script | yes (script) |
| Boundaries | closed sides and floor, open top | all six sides **open** by default (`xXyYzZ`) | `use_collision_border_{front,back,left,right,bottom} = True`, `top = False` (probe `bench=1`) | generated script: `boundConditions` | yes (script) |
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
  additive mode with `surface_distance = 0`.** Mass matches Ember's within 3%
  from frame 12 to 60. The emitter-centre density rises as Ember's does:
  0.500 at frame 12 in both.

  Temperature cannot be matched the same way. Mantaflow raises the emitter's
  heat to the flow's temperature and never above it (1.000 in every checked
  frame), while Ember's grows as rate · t. The heat is set to Ember's frame-24
  value, so Mantaflow's emitter is hotter than Ember's before frame 24 and
  cooler after it. That is why Mantaflow's plume still rises faster: its
  centroid is 0.925 m against 0.781 m at frame 60 (0.872 m with the old
  absolute mapping). The heat drives the buoyancy, so the plume-height metrics
  carry this difference, and the summary must say so.

  Mantaflow computes the inflow grids once per frame and copies them into the
  domain on every solver step (`applyEmission`, absolute copy). So with more
  than one step per frame the density would not be added once per step, and
  this mapping holds only for one step per frame.
- **Wind strength.** A domain filled with smoke (`fill=1 volume=1 alpha=0
  beta=0`), all sides open, at 32³, with `wind = S` along +x. With
  S = 0.1042 (0.5 m/s² at 24 fps), the centre's u rose 0.0201, 0.0402 and
  0.0599 m/s by frames 2, 3 and 4, against 0.0208 per frame (96.5%, the
  same boundary shortfall as buoyancy). The first frame gets no force, because
  the effectors are sampled before that frame's emission, when the domain
  holds no smoke yet. My first reading of the source, S = a / (0.2 · fps²),
  missed the ÷ fps in `do_physical_effector` and was 24× too weak.
- **Wind check with an emitter** (replaces the brief's no-emitter check).
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

### Force fields act only on smoke

Mantaflow applies a force field only in cells that hold smoke:
`update_effectors_task_cb` in `source/blender/blenkernel/intern/fluid.cc`
skips every cell whose density (or fuel, when fire is active) is below
`FLT_EPSILON`, and cells inside obstacles. Ember's wind accelerates every
cell. **`plume_wind` is kept as it is.** Its results must say that
Mantaflow's wind pushes only the smoke, while Ember's pushes all the air, so
the drift compares shape, not a matched force field. The wind force is also
added once per solver step with no dt, so the mapping holds only for one step
per frame, which every bench scene uses.

### Other solver notes

- **Buoyancy in Mantaflow is a true acceleration.** `addBuoyancy` multiplies by
  the solver's dt, and with the generated `scale=False` it does not divide by
  dx. The face value uses the mean of the two cells either side, as Ember's
  does.
- **The domain's fire fields are on.** The generated script had
  `using_fire = True` for a smoke-only flow. The fuel grid is zero, so burning
  does nothing, and the flame-vorticity term that `vorticityConfinement` adds
  (`strengthCell = fuel · flame_vorticity`) is zero.

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
   collider-free and above density 1e-6, for both solvers; outflow uses face
   nz − 1 with layer nz − 2's density. `results.md` must say so.
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
   3% of Ember's. Mantaflow's emitter heat is held at a fixed value, while
   Ember's grows, so the plume heights differ (0.925 m against 0.781 m at
   frame 60). See Inflow.
8. **Tiled density can drop other grids' values** (see Clipping). The reader
   must report it rather than use zeros.
9. **The brief's `vdb_probe.rs`** formats `VdbLevel` with `{:?}`, which does
   not compile: `VdbLevel` has no `Debug`.
