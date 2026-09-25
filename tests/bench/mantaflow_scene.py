# SPDX-License-Identifier: GPL-3.0-or-later
"""Build and bake the Mantaflow equivalent of an Ember bench scene.

Run:
  Blender --background --factory-startup --python-exit-code 1 \
      --python tests/bench/mantaflow_scene.py -- SCENE_JSON OUT_DIR [--no-bake]
      [--latency 1,24,60,120 --runs 5,5,3,3]

SCENE_JSON is `Scene::mantaflow_json()`, printed by
`cargo run -p elements-ember --example benchmark -- scene-json SCENE RES`.
The script writes the Mantaflow cache to OUT_DIR/cache/ and each frame's data
file modification time to OUT_DIR/timings.json.

With --latency, it instead times re-bakes in this one open Blender (2b-3b
spec §2): for each N and run, change the flow's density, free the cache, and
bake frames 1..N. OUT_DIR/latency.json holds the seconds from the bake call
to frame N's data file existing, in the shape of Ember's latency file.

Fairness rules (piece 2 spec §5.3): no noise upres, no adaptive domain, fixed
timesteps equal to Ember's substep count, and an uncompressed 32-bit OpenVDB
cache. Every value comes from the scene JSON, and every conversion goes through
mapping.py, which names its sources. Blender settings that are not
conversions are from docs/bench/mantaflow-notes.md, "Parameter mapping".
"""

import hashlib
import json
import os
import statistics
import sys
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapping  # noqa: E402

GRAVITY = 9.81  # |g|, Blender's default scene gravity, m/s²; left at its default

# Ember's face names onto the domain's border switches. Mantaflow's
# boundConditions letters: left/right are x, front/back are y, bottom/top z.
SIDES = {
    "neg_x": "use_collision_border_left",
    "pos_x": "use_collision_border_right",
    "neg_y": "use_collision_border_front",
    "pos_y": "use_collision_border_back",
    "neg_z": "use_collision_border_bottom",
    "pos_z": "use_collision_border_top",
}


def fire_settings(sc: dict) -> dict | None:
    """Mantaflow's fire settings for a scene with fire, through mapping.fire;
    None for a smoke scene."""
    if not sc.get("fire"):
        return None
    return mapping.fire(
        sc["emitter"]["fuel_rate"],
        sc["fire"]["burning_rate"],
        sc["fire"]["flame_vorticity"],
        sc["fps"],
    )


def build_domain(sc: dict, cache_dir: str, m: dict | None) -> bpy.types.Object:
    size = sc["domain_size"]
    # Ember's domain runs from the origin to (size, size, size).
    bpy.ops.mesh.primitive_cube_add(size=size, location=(size / 2,) * 3)
    domain = bpy.context.active_object
    s = domain.modifiers.new("Fluid", "FLUID")
    s.fluid_type = "DOMAIN"
    d = s.domain_settings
    d.domain_type = "GAS"
    d.resolution_max = sc["resolution"]
    d.use_adaptive_domain = False
    d.use_noise = False
    d.use_adaptive_timesteps = False
    # mapping.py's per-frame conversions (inflow density = rate / fps, wind,
    # vorticity) hold only for one Mantaflow step per frame.
    if sc["substeps"] != 1:
        sys.exit(
            f"substeps = {sc['substeps']}: the Mantaflow mappings hold only for one step per frame"
        )
    d.timesteps_min = d.timesteps_max = 1
    d.cache_directory = cache_dir
    d.cache_type = "ALL"
    d.cache_data_format = "OPENVDB"
    d.openvdb_cache_compress_type = "NONE"
    d.openvdb_data_depth = "32"  # the patched vdb-rs decodes Vec3s only at full precision
    d.cache_frame_start, d.cache_frame_end = 1, sc["frames"]
    d.alpha, d.beta = mapping.buoyancy(
        sc["buoyancy_density"], sc["buoyancy_temperature"], size, GRAVITY
    )
    d.vorticity = mapping.vorticity(sc["vorticity"], sc["fps"])
    # Mantaflow's dissolve is not Ember's exponential decay, and no bench scene
    # dissipates; refuse rather than bake a scene that silently differs.
    if sc["density_dissipation"] or sc["temperature_dissipation"]:
        sys.exit("dissipation has no Mantaflow mapping; the bench scenes use 0")
    d.use_dissolve_smoke = False
    # Blender opens all six borders by default; Ember's walls must be closed.
    for side, prop in SIDES.items():
        setattr(d, prop, sc["boundaries"][side] == "wall")
    if m is not None:
        # Only a resumable cache writes the fuel and react grids, and it bakes
        # the same simulation (mantaflow-notes.md, Fire: grid inventory).
        d.cache_resumable = True
        # mapping.fire converts the per-second rates (notes, Fire: the mapping).
        d.burning_rate = m["burning_rate"]
        d.flame_vorticity = m["flame_vorticity"]
        # No time unit: equal in both solvers (notes, Fire: smoke and heat).
        d.flame_smoke = sc["fire"]["flame_smoke"]
        d.flame_ignition = sc["fire"]["ignition_temperature"]
        d.flame_max_temp = sc["fire"]["max_temperature"]
    return domain


def build_emitter(sc: dict, m: dict | None) -> None:
    e = sc["emitter"]
    bpy.ops.mesh.primitive_uv_sphere_add(radius=e["radius"], location=e["center"])
    f = bpy.context.active_object.modifiers.new("Fluid", "FLUID")
    f.fluid_type = "FLOW"
    fs = f.flow_settings
    fs.flow_type = "SMOKE"
    fs.flow_behavior = "INFLOW"
    fs.flow_source = "MESH"
    fs.use_absolute = False  # additive: adds density each step, as Ember's rate does
    fs.surface_distance = 0.0  # no emission band outside the mesh
    fs.volume_density = 1.0  # emit through the volume, not just the shell
    fs.density, fs.temperature = mapping.inflow(e["density_rate"], e["temperature_rate"], sc["fps"])
    if m is not None:
        # A FIRE flow emits fuel and no density; BOTH emits the two (fluid.cc
        # apply_inflow_fields; notes, Fire: emission).
        fs.flow_type = "FIRE" if e["density_rate"] == 0 else "BOTH"
        # Added once a frame like density, clamped to [0, 10] (notes, Fire).
        fs.fuel_amount = m["fuel_amount"]
    # Mantaflow has no frame range: key "Use Flow" on through the last active
    # frame and off after it. Confirmed by a bake (mantaflow-notes.md, Inflow).
    first, last = e["active_frames"]
    keys = [(last, True), (last + 1, False)]
    if first > 1:
        keys = [(first - 1, False), (first, True), *keys]
    for frame, on in keys:
        fs.use_inflow = on
        fs.keyframe_insert("use_inflow", frame=frame)
    # The heat stays fixed. A keyed ramp to Ember's growing heat brought the
    # plume heights closer but lost 3.5% to 13.6% of the mass (notes, Inflow).


def build_collider(sc: dict) -> None:
    c = sc["collider"]
    if c is None:
        return
    bpy.ops.mesh.primitive_uv_sphere_add(radius=c["radius"], location=c["center"])
    eff = bpy.context.active_object.modifiers.new("Fluid", "FLUID")
    eff.fluid_type = "EFFECTOR"
    eff.effector_settings.effector_type = "COLLISION"


def build_wind(sc: dict) -> None:
    # A scene without wind has no wind_rate, or a rate of 0: no field at all.
    rate = sc.get("wind_rate", 0.0)
    if not rate:
        return
    size = sc["domain_size"]
    strength, flow, direction = mapping.wind(tuple(sc["wind_velocity"]), rate, sc["fps"], size)
    bpy.ops.object.effector_add(type="WIND", location=(size / 2,) * 3)
    wind = bpy.context.active_object
    f = wind.field
    f.shape = "PLANE"
    f.strength = strength
    f.flow = flow  # the drag towards the wind's velocity that makes it a relaxation
    # Blender has no "none" falloff: a sphere falloff of power 0 is 1 everywhere,
    # and BOTH keeps both sides of the field's plane.
    f.falloff_type = "SPHERE"
    f.falloff_power = 0.0
    f.use_min_distance = False
    f.use_max_distance = False
    f.z_direction = "BOTH"
    wind.rotation_mode = "QUATERNION"  # else rotation_quaternion is ignored
    wind.rotation_quaternion = mapping.rotation_to(direction)


def build(sc: dict, cache_dir: str) -> bpy.types.Object:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, sc["frames"]
    if float(sc["fps"]) != int(sc["fps"]):
        sys.exit(f"Blender's fps is an integer, got {sc['fps']}")
    scene.render.fps = int(sc["fps"])
    scene.render.fps_base = 1.0
    # Mantaflow's fire settings, or None for a smoke scene.
    fire = fire_settings(sc)
    domain = build_domain(sc, cache_dir, fire)
    build_emitter(sc, fire)
    build_collider(sc)
    build_wind(sc)
    return domain


def bake(domain: bpy.types.Object, cache_dir: str, frames: int) -> list[dict]:
    with bpy.context.temp_override(object=domain, active_object=domain):
        result = bpy.ops.fluid.bake_all()
    if "FINISHED" not in result:
        sys.exit(f"bake_all returned {result}")
    # Each data file is written when its frame finishes, so consecutive
    # modification times give the per-frame time (mantaflow-notes.md, Timing).
    out = []
    for n in range(1, frames + 1):
        path = os.path.join(cache_dir, "data", f"fluid_data_{n:04d}.vdb")
        if not os.path.exists(path):
            sys.exit(f"missing cache frame {n}: {path}")
        out.append({"frame": n, "mtime_ns": os.stat(path).st_mtime_ns})
    return out


# The untimed warm-up bakes the unchanged scene this far, as Ember's does,
# so the timed runs see a Blender that has already baked once.
WARM_UP_FRAMES = 24


def data_file(cache_dir: str, n: int) -> str:
    return os.path.join(cache_dir, "data", f"fluid_data_{n:04d}.vdb")


def data_files(cache_dir: str) -> list[str]:
    data = os.path.join(cache_dir, "data")
    return sorted(os.listdir(data)) if os.path.isdir(data) else []


def free(domain: bpy.types.Object, cache_dir: str) -> None:
    with bpy.context.temp_override(object=domain, active_object=domain):
        result = bpy.ops.fluid.free_all()
    if "FINISHED" not in result:
        sys.exit(f"free_all returned {result}")
    # A stale frame N would end the next run's clock early.
    left = data_files(cache_dir)
    if left:
        sys.exit(f"free_all left {len(left)} data files, e.g. {left[0]}")


def timed_bake(domain: bpy.types.Object, cache_dir: str, n: int) -> tuple[float, float]:
    """Seconds from just before `bake_all` to frame n's data file existing:
    the later of the call returning and the file's modification time; and
    the call's return minus that modification time."""
    domain.modifiers["Fluid"].domain_settings.cache_frame_end = n
    wall_ns = time.time_ns()
    start = time.perf_counter()
    with bpy.context.temp_override(object=domain, active_object=domain):
        result = bpy.ops.fluid.bake_all()
    returned = time.perf_counter() - start
    if "FINISHED" not in result:
        sys.exit(f"bake_all returned {result}")
    path = data_file(cache_dir, n)
    if not os.path.exists(path):
        sys.exit(f"bake to frame {n} wrote no {path}")
    written = (os.stat(path).st_mtime_ns - wall_ns) / 1e9
    return max(returned, written), returned - written


# An OpenVDB file header holds a random 36-character UUID at bytes 21..57,
# the only bytes two bakes of the same scene differ in (checked at 32³).
UUID = slice(21, 57)


def data_hash(path: str) -> str:
    """The data file's hash without its header UUID, so that equal bakes
    give equal hashes."""
    with open(path, "rb") as fh:
        data = fh.read()
    uuid = data[UUID]
    if len(uuid) != 36 or uuid[8:9] != b"-" or uuid[23:24] != b"-":
        sys.exit(f"{path}: no UUID at bytes 21..57, so the hash cannot skip it")
    return hashlib.sha256(data[: UUID.start] + data[UUID.stop :]).hexdigest()


def latency(
    domain: bpy.types.Object, flow, cache_dir: str, frames: list[int], runs: list[int]
) -> dict:
    d = domain.modifiers["Fluid"].domain_settings
    end, base = d.cache_frame_end, flow.density
    free(domain, cache_dir)
    print(f"latency: warm-up to frame {WARM_UP_FRAMES}", file=sys.stderr, flush=True)
    timed_bake(domain, cache_dir, WARM_UP_FRAMES)
    load_before = os.getloadavg()[0]
    points = []
    for n, count in zip(frames, runs, strict=True):
        runs_s, hashes = [], set()
        for run in range(count):
            # A new density each run, so no bake can repeat an earlier one.
            flow.density = base * (1 + 0.01 * (run + 1))
            free(domain, cache_dir)
            s, gap = timed_bake(domain, cache_dir, n)
            runs_s.append(s)
            hashes.add(data_hash(data_file(cache_dir, n)))
            print(
                f"latency: N = {n}, run {run + 1} of {count}: {s:.3f} s",
                file=sys.stderr,
                flush=True,
            )
        # Equal files would mean the density change never reached the bake.
        if len(hashes) != count:
            sys.exit(f"N = {n}: {count} densities gave {len(hashes)} distinct frame-{n} files")
        points.append(
            {
                "frame": n,
                "runs_s": runs_s,
                "median_s": statistics.median(runs_s),
                # The last run's: how long bake_all returns after frame n's file.
                "return_after_write_s": gap,
            }
        )
    load_after = os.getloadavg()[0]
    flow.density = base
    d.cache_frame_end = end
    return {
        "solver": "mantaflow",
        "resolution": d.resolution_max,
        "load_before": load_before,
        "load_after": load_after,
        "blender": bpy.app.version_string,
        "points": points,
    }


def int_list(args: list[str], flag: str) -> list[int]:
    return [int(v) for v in args[args.index(flag) + 1].split(",")]


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :]
    scene_json, out_dir = args[0], os.path.abspath(args[1])
    with open(scene_json) as fh:
        sc = json.load(fh)
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    domain = build(sc, cache_dir)
    if "--no-bake" in args[2:]:
        return
    if "--latency" in args[2:]:
        flow = next(
            o
            for o in bpy.data.objects
            if o.modifiers.get("Fluid") and o.modifiers["Fluid"].fluid_type == "FLOW"
        )
        out = latency(
            domain,
            flow.modifiers["Fluid"].flow_settings,
            cache_dir,
            int_list(args, "--latency"),
            int_list(args, "--runs"),
        )
        with open(os.path.join(out_dir, "latency.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        return
    frames = bake(domain, cache_dir, sc["frames"])
    with open(os.path.join(out_dir, "timings.json"), "w") as fh:
        json.dump({"frames": frames, "blender": bpy.app.version_string}, fh, indent=1)


main()
