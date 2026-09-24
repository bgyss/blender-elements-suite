# SPDX-License-Identifier: GPL-3.0-or-later
"""Build and bake the Mantaflow equivalent of an Ember bench scene.

Run:
  Blender --background --factory-startup --python-exit-code 1 \
      --python tests/bench/mantaflow_scene.py -- SCENE_JSON OUT_DIR [--no-bake]

SCENE_JSON is `Scene::mantaflow_json()`, printed by
`cargo run -p elements-ember --example benchmark -- scene-json SCENE RES`.
The script writes the Mantaflow cache to OUT_DIR/cache/ and each frame's data
file modification time to OUT_DIR/timings.json.

Fairness rules (piece 2 spec §5.3): no noise upres, no adaptive domain, fixed
timesteps equal to Ember's substep count, and an uncompressed 32-bit OpenVDB
cache. Every value comes from the scene JSON, and every conversion goes through
mapping.py, which names its sources. Blender settings that are not
conversions are from docs/bench/mantaflow-notes.md, "Parameter mapping".
"""

import json
import os
import sys

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


def build_domain(sc: dict, cache_dir: str) -> bpy.types.Object:
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
            f"substeps = {sc['substeps']}: the Mantaflow mappings hold only for "
            "one step per frame"
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
    return domain


def build_emitter(sc: dict) -> None:
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
    if not any(sc["wind"]):
        return
    size = sc["domain_size"]
    strength, direction = mapping.wind(tuple(sc["wind"]), sc["fps"])
    bpy.ops.object.effector_add(type="WIND", location=(size / 2,) * 3)
    wind = bpy.context.active_object
    f = wind.field
    f.shape = "PLANE"
    f.strength = strength
    f.flow = 0.0  # WIND defaults to 1, which drags towards the smoke's velocity
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
    domain = build_domain(sc, cache_dir)
    build_emitter(sc)
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
    frames = bake(domain, cache_dir, sc["frames"])
    with open(os.path.join(out_dir, "timings.json"), "w") as fh:
        json.dump({"frames": frames, "blender": bpy.app.version_string}, fh, indent=1)


main()
