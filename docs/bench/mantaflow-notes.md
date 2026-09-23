# Mantaflow cache and parameter notes (2b-3 Task 3)

**Status: incomplete, blocked.** `vdb-rs` 0.6 cannot read the velocity
grid, which is one of the stop conditions in the 2b-3 plan. The finding and
the options are in "Corrections to the spec" below. The sections below record
everything found before stopping. Items marked *not yet confirmed by
experiment* come from reading Blender's source and still need the experiment
the plan asks for.

Machine: macOS on Apple Silicon, Blender 5.2.2 LTS (hash d13f752e3b9c, built
2026-09-15). Date: 2026-09-23. The machine was under load from other
processes throughout (load average 8 to 28).

Sources were read, never copied. Paths are in github.com/blender/blender at
`main`:

- `intern/mantaflow/intern/MANTA_main.cpp`: which RNA values reach the script
- `intern/mantaflow/intern/strings/smoke_script.h`, `fluid_script.h`: the generated solver script
- `extern/mantaflow/preprocessed/plugin/extforces.cpp`: `addBuoyancy`, `vorticityConfinement`, `addForceField`
- `extern/mantaflow/preprocessed/plugin/initplugins.cpp`: `applyEmission`
- `extern/mantaflow/preprocessed/fileio/iovdb.cpp`: the VDB writer
- `source/blender/blenkernel/intern/fluid.cc`: emission and force-field sampling
- `source/blender/blenkernel/intern/effect.cc`: the wind field's force

The probe is `tests/bench/probe_mantaflow.py`. Setting the domain's
`export_manta_script = True` also writes the generated solver script to
`<cache>/script/smoke_script.py`. That is the fastest way to see what a setting
turns into.

## Headless baking

`bpy.ops.fluid.bake_all()` runs under
`Blender --background --factory-startup --python` and returns `{'FINISHED'}`.
It needs only the domain as the context object:

```python
with bpy.context.temp_override(object=domain, active_object=domain):
    bpy.ops.fluid.bake_all()
```

Every property name the probe sets exists in 5.2.2, including
`openvdb_data_depth` (`"32"`), `openvdb_cache_compress_type` (`"NONE"`),
`timesteps_min` / `timesteps_max`, `use_collision_border_*` and `clipping`.

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

A 16³ frame is 967,548 bytes, a 256³ frame 3.5 to 5 MB in the first ten
frames (it grows with the smoke).

## Grids

Grids in `fluid_data_NNNN.vdb` for a `GAS` domain with one smoke flow, no
noise, 32-bit, uncompressed. Read with `vdb-rs` 0.6 at 16³, frame 5:

| Name | VDB type | Class | Reads as | Active voxels | Index bbox | Tiles |
|---|---|---|---|---|---|---|
| `density` | `Tree_float_5_4_3` | fog volume | `f32`: yes | 292 | (3,3,1)–(12,12,7) | none |
| `temperature` | `Tree_float_5_4_3` | fog volume | `f32`: yes | 284 | (3,3,1)–(12,12,7) | none |
| `flame` | `Tree_float_5_4_3` | fog volume | `f32`: yes | 0 | empty | none |
| `shadow` | `Tree_float_5_4_3` | fog volume | `f32`: yes | 8 tiles | (0,0,0)–(8,8,8) | 8 `Node3` tiles, all −1 |
| `velocity` | `Tree_vec3s_5_4_3` | **staggered** | `[f32; 3]`: **no**, 0 of 292 voxels | 292 (metadata) | (3,3,1)–(12,12,7) (metadata) | — |

Every grid carries `file_base_resolution` (16,16,16) and `file_voxel_size`
(0.125) metadata, and `is_saved_as_half_float = false`.

**Grids are clipped to the smoke.** The data file is written with
`clip = domain.clipping` (default 1e-6) and `clipGrid = density`
(`smoke_script.h`, `smoke_save_data`; `iovdb.cpp`, `exportVDB`).
Density and temperature store only voxels above the clip value, and
`velocity` (a sparse grid) is then clipped to density's active voxels. So the
cache holds **velocity only where there is smoke**; every other cell reads as
the background, zero. That is why `velocity` has exactly density's 292 voxels.

**`vdb-rs` 0.6 cannot read the velocity grid.** It returns an empty tree, no
error. Its `read_tree_topology` reads the root node's background value as 4
bytes (`read_u32`) and each root tile's value as 4 bytes, whatever the grid's
value type. For a `Vec3s` grid those are 12 bytes, so the root-node counts
that follow are read from the background vector's y and z (both 0.0), and the
tree has no nodes. A copy of `vdb-rs` in the scratchpad with those two reads
changed to read `size_of::<ValueTy>()` bytes reads all 292 velocity voxels of
the same file. `vdb-rs` also does no type check: reading `density` as
`[f32; 3]` returns 0 voxels without an error, and reading `temperature` as
`[f32; 3]` gives `IoError`. A reader must check `descriptor.grid_type`
itself.

## Velocity location

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

The initial-velocity experiment in the plan could not separate the two cases:
velocity is clipped to density's voxels, so the stored extent is density's
extent in both cases (32³, `init_vel=1,0,0`, frame 1: velocity and density
both span index (12,12,1)–(19,19,8)).

Units of the stored velocity: Mantaflow's own, cells per Mantaflow time unit,
from the scale factors in the generated script. At 25 fps one frame is 0.1
Mantaflow time units, so 1 s = 2.5 units and u[m/s] = u_stored · dx / 0.4.
*Not yet confirmed by experiment*: the one attempt (a domain-filling
`GEOMETRY` flow with initial velocity 1 m/s along x) gave a non-uniform
x velocity of 0.02–0.08, and the stop came before it was understood.

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
- Grids are sparse: anything absent reads as zero, and `shadow` shows constant
  regions are stored as `Node3` tiles (8³ voxels), which a reader must expand.

## Timing

Each frame's data file is written when that frame finishes, so the per-frame
time is the difference between consecutive `fluid_data_NNNN.vdb` modification
times. Frame 1 has no predecessor and is excluded, as the spec's frame-time
rule already says. With `timesteps_min = timesteps_max = 1` and adaptive
time steps off, the generated script sets `timestepMin = timestepMax =
frameLength`, so there is one solver step per frame.

## Parameter mapping

Unit conversions used below, all from the generated script: with domain size
L (the longest side) and resolution n, one Mantaflow time unit is 0.4 s
whatever the fps (`frameLengthRaw = 0.1 · 25`), a frame is 2.5 / fps units
long, and gravity is converted to cells per unit² by
`scaleAcceleration = (n / L) · 0.4²`.

| Quantity | Ember | Mantaflow | Conversion | Source | Confirmed |
|---|---|---|---|---|---|
| Buoyancy, heat | `buoyancy_temperature` β, upward accel. β·T | domain `beta` | `beta = β · L / abs(g)` | `smoke_script.h` divides `beta` by L; `addBuoyancy` adds −g·dt·coef, g in scene units | no |
| Buoyancy, density | `buoyancy_density` α, downward accel. α·ρ | domain `alpha`, **upward** for positive values | `alpha = −α · L / abs(g)` | as above | no |
| Gravity | none (buoyancy is its own term) | scene gravity × effector weights' `global_gravity` | keep defaults: (0, 0, −9.81), weight 1 | `fluid.cc` `update_final_gravity` | — |
| Vorticity | `vorticity` ε, 1/s; Δu = h·ε·dx·(N×ω) | domain `vorticity`; Δu = v·(dt / frame)·(N×ω) in cells | `vorticity = ε / fps` (dx cancels) | `extforces.cpp` `KnConfForce`; `smoke_script.h` | no; bench ε = 0, so 0 |
| Wind | uniform acceleration a, m/s², every cell | `WIND` field strength S, falloff `NONE`, `flow = 0` | with one step per frame, `S = a / (0.2 · fps²)`; 0.5 m/s² at 24 fps gives S ≈ 0.00434 | `fluid.cc` `update_effectors_task_cb` (×0.2, clamp ±1), then `scaleSpeedFrames`; `addForceField` adds it each step with no dt | no |
| Wind direction | the acceleration vector | the field object's local +z | rotate +z onto the direction | `effect.cc`, `PFIELD_WIND` uses `efd->nor` | no |
| Inflow | adds `density_rate · occupancy · h` each substep | `INFLOW` sets the cell to `density · emission` each step (absolute mode) or adds and clamps to [0, 1] (the default); temperature is raised to the flow's `temperature` | not chosen yet; the plan asks to match Ember's density at the emitter centre at frame 24 | `fluid.cc` `apply_inflow_fields`; `initplugins.cpp` `applyEmission` | no |
| Emitter volume | a filled sphere (occupancy with a one-cell edge) | a mesh flow emits a **shell** by default: `volume_density = 0`, `surface_distance = 1.5` | set `volume_density = 1` | `fluid.cc`; the 32³ bake showed a hollow emitter | observed |
| Substeps | `max_substeps` (preview: 1) | `timesteps_min = timesteps_max` | equal | generated script | yes (script) |
| Boundaries | closed sides and floor, open top | all six sides **open** by default (`xXyYzZ`) | `use_collision_border_{front,back,left,right,bottom} = True`, top `False` | generated script: `boundConditions = 'xXyYzZ'` | yes (script) |
| Advection | MacCormack | `advectSemiLagrange(order=2)`, Mantaflow's MacCormack | none needed | generated script | yes (script) |
| Pressure | 160 red-black Gauss–Seidel iterations | multigrid-preconditioned CG to a tolerance | none possible; the table must say so | generated script | — |

Notes on the table:

- **Buoyancy in Mantaflow is a true acceleration.** `addBuoyancy` multiplies by
  the solver's dt, and with the generated `scale=False` it does not divide by
  dx. Its SI acceleration per unit density or temperature is abs(g) · coef,
  where the script's coef is `alpha / L` or `beta / L`. The face value uses the
  mean of the two cells either side, as Ember's does.
- **The domain's fire fields are on.** The generated script had
  `using_fire = True` for a smoke-only flow. The fuel grid is zero, so burning
  does nothing, and the flame vorticity term `vorticityConfinement` adds
  (`strengthCell = fuel · flame_vorticity`) is zero.
- **Force fields act only where there is smoke.** `update_effectors_task_cb`
  skips every cell whose density (or fuel) is below `FLT_EPSILON`, so
  Mantaflow's wind never touches clear air, while Ember's acts on every cell.
- **Wind has no dt.** The force is added once per solver step. With more than
  one step per frame it would be applied that many times. The bench uses one.

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
plus about a second of Blender start-up per bake. This is an **upper bound
for an idle machine** (the load average was 11–28) but only an estimate for
frames past 10: the smoke, and with it the sparse files, keeps growing. It is
under the 90-minute stop line.

## Corrections to the spec

1. **`vdb-rs` 0.6 cannot read Mantaflow's velocity grid** (spec §8 risk (a),
   the plan's stop condition). The float grids read correctly. The cause is
   two 4-byte reads in `vdb-rs`'s root-node parser that should be value-sized;
   fixing them makes the grid readable. Options for the user: patch `vdb-rs`
   (upstream or a `[patch]` fork; it is MIT/Apache), write the reader's own
   root-node parse in `elements-ember` or `elements-io`, or take the spec's
   `.npy` fallback through Blender's OpenVDB module.
2. **The cache holds velocity only where there is smoke.** Every sparse grid
   is clipped to density's active voxels at `clipping` (1e-6). Kinetic energy,
   vorticity and divergence from the cache would cover only smoky cells, and
   the top-boundary outflow only faces with smoke below them (for outflow that
   is harmless, since ρ = 0 elsewhere). Setting `clipping = 0` still drops
   cells whose density is exactly 0.
3. **Mantaflow's wind acts only on smoky cells**, so the `plume_wind` scenes
   differ in more than parameters, and the plan's wind check (a closed 32³
   domain with no emitter) cannot work: with no smoke no force is applied,
   and in a closed box a uniform force is removed by projection anyway. A
   check needs smoke everywhere and open sides.
4. **The fixture cannot be under 200 KB.** An uncompressed 16³ frame is
   967,548 bytes: uncompressed VDB internal nodes store their full tile tables
   (32³ values at the top level) for every grid.
5. **`mapping.py` signatures need more inputs.** `buoyancy` needs the domain
   size and gravity; `vorticity` needs the fps, not dx; `wind` needs the fps.
6. **Blender's default borders are all open** and a mesh flow emits a shell by
   default. Both are settings the scene script must change, not fairness
   rules the spec already lists.
7. **The brief's `vdb_probe.rs`** formats `VdbLevel` with `{:?}`, which does
   not compile: `VdbLevel` has no `Debug`.

Not done because of the stop: the buoyancy, inflow and wind experiments,
`tests/bench/mapping.py`, and the fixture.
