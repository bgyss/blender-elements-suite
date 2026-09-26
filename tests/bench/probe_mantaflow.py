# SPDX-License-Identifier: GPL-3.0-or-later
"""Bake a tiny gas domain headlessly and print what the cache contains.

Run:
  Blender --background --factory-startup --python-exit-code 1 \
      --python tests/bench/probe_mantaflow.py -- OUT_DIR RES FRAMES [OPTION=VALUE ...]

Options, all optional, for the experiments in docs/bench/mantaflow-notes.md:
  flow=0|1          add the sphere flow object (default 1)
  fill=1            make the flow a cube filling the whole domain instead
  behavior=B        the flow's behaviour (default INFLOW; GEOMETRY fills once)
  volume=V          the flow's volume emission (Blender's default, 0, emits a shell)
  clipping=C        the domain's VDB clipping threshold
  init_vel=X,Y,Z    give the flow an initial velocity (m/s)
  density=D         the flow's density (default Blender's, 1)
  temperature=T     the flow's temperature difference (default Blender's, 1)
  absolute=0|1      the flow's use_absolute (default Blender's, 1: hold the value)
  surface=S         the flow's surface_distance, in cells (default Blender's, 1.0)
  alpha=A beta=B    the domain's buoyancy coefficients
  vorticity=V       the domain's vorticity
  wind=S            add a WIND force field of strength S blowing along +x
  wind_flow=F       that field's flow: drag towards the wind (default 0, off)
  closed=0|1        close all six domain walls (default: Blender's default)
  bench=1           close the sides and floor and leave the top open, as Ember does
  open=S1,S2,...    then open these borders (left, right, front, back, bottom, top)
  heat_fill=T       add a second flow filling the domain: temperature T, no density
  script=0|1        also export Blender's generated Mantaflow script
  fire=1            make the flow a FIRE flow (fuel, no smoke)
  fuel=F            the flow's fuel_amount
  burning=B         the domain's burning_rate
  flame_smoke=S     the domain's flame_smoke
  flame_vorticity=V the domain's flame_vorticity
  ignition=T maxtemp=T  the domain's flame_ignition and flame_max_temp
  resumable=1       make the cache resumable, which also saves fuel and react
"""

import json
import math
import os
import sys
import time

import bpy


def floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split(","))


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :]
    out_dir, res, frames = args[:3]
    opts = dict(a.split("=", 1) for a in args[3:])
    res, frames = int(res), int(frames)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, frames
    scene.render.fps = 24

    # A 2 m domain from (0, 0, 0) to (2, 2, 2), matching Ember's convention.
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(1.0, 1.0, 1.0))
    domain = bpy.context.active_object
    fluid = domain.modifiers.new("Fluid", "FLUID")
    fluid.fluid_type = "DOMAIN"
    s = fluid.domain_settings
    s.domain_type = "GAS"
    s.resolution_max = res
    s.use_adaptive_domain = False
    s.use_noise = False
    s.use_adaptive_timesteps = False
    s.timesteps_min = s.timesteps_max = 1
    s.cache_directory = out_dir
    s.cache_type = "ALL"
    s.cache_data_format = "OPENVDB"
    s.openvdb_cache_compress_type = "NONE"
    s.openvdb_data_depth = "32"
    s.cache_frame_start, s.cache_frame_end = 1, frames
    for key in ("alpha", "beta", "vorticity", "clipping"):
        if key in opts:
            setattr(s, key, float(opts[key]))
    if opts.get("closed") == "1":
        for side in ("front", "back", "right", "left", "top", "bottom"):
            setattr(s, f"use_collision_border_{side}", True)
    if opts.get("bench") == "1":
        for side in ("front", "back", "right", "left", "bottom"):
            setattr(s, f"use_collision_border_{side}", True)
        s.use_collision_border_top = False
    for key, prop in (
        ("burning", "burning_rate"),
        ("flame_smoke", "flame_smoke"),
        ("flame_vorticity", "flame_vorticity"),
        ("ignition", "flame_ignition"),
        ("maxtemp", "flame_max_temp"),
    ):
        if key in opts:
            setattr(s, prop, float(opts[key]))
    for side in filter(None, opts.get("open", "").split(",")):
        setattr(s, f"use_collision_border_{side}", False)
    if opts.get("script") == "1":
        s.export_manta_script = True
    if opts.get("resumable") == "1":
        s.cache_resumable = True

    if opts.get("flow", "1") == "1":
        if opts.get("fill") == "1":
            bpy.ops.mesh.primitive_cube_add(size=2.0, location=(1.0, 1.0, 1.0))
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(1.0, 1.0, 0.3))
        flow_obj = bpy.context.active_object
        flow = flow_obj.modifiers.new("Fluid", "FLUID")
        flow.fluid_type = "FLOW"
        f = flow.flow_settings
        f.flow_type = "SMOKE"
        if opts.get("fire") == "1":
            f.flow_type = "FIRE"
            if "fuel" in opts:
                f.fuel_amount = float(opts["fuel"])
        f.flow_behavior = opts.get("behavior", "INFLOW")
        if "volume" in opts:
            f.volume_density = float(opts["volume"])
        if "density" in opts:
            f.density = float(opts["density"])
        if "temperature" in opts:
            f.temperature = float(opts["temperature"])
        if "surface" in opts:
            f.surface_distance = float(opts["surface"])
        if "absolute" in opts:
            f.use_absolute = opts["absolute"] == "1"
        if "init_vel" in opts:
            f.use_initial_velocity = True
            f.velocity_factor = 0.0
            f.velocity_normal = 0.0
            f.velocity_coord = floats(opts["init_vel"])

    if "heat_fill" in opts:
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(1.0, 1.0, 1.0))
        heat = bpy.context.active_object.modifiers.new("Fluid", "FLUID")
        heat.fluid_type = "FLOW"
        h = heat.flow_settings
        h.flow_type = "SMOKE"
        h.flow_behavior = "INFLOW"
        h.volume_density = 1.0
        h.density = 0.0
        h.temperature = float(opts["heat_fill"])

    if "wind" in opts:
        # A WIND field blows along its object's local +z; turn that onto +x.
        bpy.ops.object.effector_add(type="WIND", location=(1.0, 1.0, 1.0))
        wind = bpy.context.active_object
        wind.rotation_euler = (0.0, math.pi / 2, 0.0)
        wind.field.strength = float(opts["wind"])
        wind.field.flow = float(opts.get("wind_flow", "0"))
        # Blender has no "none" falloff: a sphere falloff of power 0 is 1 everywhere
        # (effect.cc, falloff_func), and BOTH keeps both sides of the plane.
        wind.field.falloff_type = "SPHERE"
        wind.field.falloff_power = 0.0
        wind.field.use_min_distance = False
        wind.field.use_max_distance = False
        wind.field.z_direction = "BOTH"
        wind.field.shape = "PLANE"

    keys = (
        "alpha",
        "beta",
        "vorticity",
        "gravity",
        "time_scale",
        "cfl_condition",
        "burning_rate",
        "flame_smoke",
        "flame_vorticity",
        "flame_ignition",
        "flame_max_temp",
    )
    settings = {k: getattr(s, k) for k in keys}
    settings["gravity"] = list(settings["gravity"])
    settings["scene_gravity"] = list(scene.gravity)
    settings["use_scene_gravity"] = scene.use_gravity
    if opts.get("flow", "1") == "1":
        settings["use_absolute"] = f.use_absolute
        settings["fuel_amount"] = f.fuel_amount
    settings["borders"] = {
        side: getattr(s, f"use_collision_border_{side}")
        for side in ("front", "back", "right", "left", "top", "bottom")
    }

    start = time.perf_counter()
    with bpy.context.temp_override(object=domain, active_object=domain):
        result = bpy.ops.fluid.bake_all()
    elapsed = time.perf_counter() - start
    files = []
    for root, _, names in os.walk(out_dir):
        for name in names:
            path = os.path.join(root, name)
            st = os.stat(path)
            files.append({"path": path, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns})
    files.sort(key=lambda f: f["path"])
    report = {
        "result": list(result),
        "seconds": elapsed,
        "start_point": list(s.start_point),
        "cell_size": list(s.cell_size),
        "domain_resolution": list(s.domain_resolution),
        "settings": settings,
        "files": files,
    }
    print("PROBE " + json.dumps(report))


main()
