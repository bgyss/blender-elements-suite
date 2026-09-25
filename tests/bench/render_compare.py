# SPDX-License-Identifier: GPL-3.0-or-later
"""Render Ember's and Mantaflow's density side by side with one Cycles setup.

Run (headless):
  Blender --background --factory-startup --python-exit-code 1 \
      --python tests/bench/render_compare.py -- EMBER_DIR MANTAFLOW_CACHE SCENE RES SIZE OUT_DIR
  Blender ... --python tests/bench/render_compare.py -- --readme OUT_DIR BAKE_DIR SCENE:RES...

EMBER_DIR holds `elements-cli bake`'s `density.NNNN.vdb`; MANTAFLOW_CACHE is
the cache directory of `mantaflow_scene.py`, with `data/fluid_data_NNNN.vdb`;
SIZE is the scene's domain size in metres (its JSON's `domain_size`).
For frames 30, 60 and 90 it builds a scene from factory settings and writes
OUT_DIR/{scene}-{res}-f{frame:03}.png: Ember on the left, Mantaflow on the
right, each Volume object reading its file's `density` grid through the same
Principled Volume material, placed by placement.py (2b-3b spec §3).

The Cycles device is written to EMBER_DIR/render-device, next to the bake it
rendered. `just bench-render` runs the bakes and this script, then the
--readme mode, which writes OUT_DIR/README.md from BAKE_DIR's records: each
bake's `commit`, the render device, and the scene JSON's domain size.

The images are for a person to judge by eye; nothing here measures them.
"""

import datetime
import json
import math
import os
import subprocess
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import placement  # noqa: E402

FRAMES = (30, 60, 90)
WIDTH, HEIGHT = 960, 540
SAMPLES = 64
SEED = 0
# The Principled Volume's density: one multiplier on the grid's value, the
# same for both solvers. Chosen by eye so that neither plume saturates.
DENSITY = 20.0
WORLD_GREY = 0.18
KEY_STRENGTH = 10.0
FILL_STRENGTH = 2.0
GAP = 0.25
SOLVERS = ("Ember", "Mantaflow")


def frame_files(ember_dir: str, manta_cache: str, frame: int) -> tuple[str, str]:
    return (
        os.path.join(ember_dir, f"density.{frame:04d}.vdb"),
        os.path.join(manta_cache, "data", f"fluid_data_{frame:04d}.vdb"),
    )


def use_gpu(scene: bpy.types.Scene) -> str:
    """Render on the Metal GPU when there is one, else the CPU; returns which."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "METAL"
    except TypeError:
        scene.cycles.device = "CPU"
        return "CPU"
    prefs.get_devices()
    gpus = [d for d in prefs.devices if d.type == "METAL"]
    for d in prefs.devices:
        d.use = d.type == "METAL"
    if not gpus:
        scene.cycles.device = "CPU"
        return "CPU"
    scene.cycles.device = "GPU"
    return f"GPU ({', '.join(d.name for d in gpus)})"


def settings(scene: bpy.types.Scene) -> str:
    scene.render.engine = "CYCLES"
    c = scene.cycles
    c.samples = SAMPLES
    c.use_adaptive_sampling = False  # every pixel gets all 64 samples
    c.seed = SEED
    c.use_animated_seed = False
    c.use_denoising = False
    r = scene.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = WIDTH, HEIGHT, 100
    r.film_transparent = False
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    # Standard, not AgX: the grey world and white smoke keep their values.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    return use_gpu(scene)


def node_material(name: str) -> tuple[bpy.types.Material, bpy.types.NodeTree]:
    mat = bpy.data.materials.new(name)
    if hasattr(mat, "use_nodes"):
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    return mat, nt


def smoke_material() -> bpy.types.Material:
    mat, nt = node_material("smoke")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    pv = nt.nodes.new("ShaderNodeVolumePrincipled")
    pv.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    pv.inputs["Density"].default_value = DENSITY
    pv.inputs["Density Attribute"].default_value = "density"
    pv.inputs["Absorption Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    pv.inputs["Emission Strength"].default_value = 0.0
    pv.inputs["Blackbody Intensity"].default_value = 0.0
    nt.links.new(pv.outputs["Volume"], out.inputs["Volume"])
    return mat


def flat_material(name: str, grey: float) -> bpy.types.Material:
    """An unlit colour, for the labels and the domain outlines."""
    mat, nt = node_material(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (grey, grey, grey, 1.0)
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def link(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.scene.collection.objects.link(obj)
    return obj


def unlit(obj: bpy.types.Object) -> None:
    """Seen by the camera only: casts no shadow and scatters nothing."""
    obj.visible_shadow = False
    obj.visible_diffuse = False
    obj.visible_glossy = False
    obj.visible_transmission = False
    obj.visible_volume_scatter = False


def add_volume(name: str, path: str, location, material) -> bpy.types.Object:
    if not os.path.isfile(path):
        sys.exit(f"missing {path}")
    vol = bpy.data.volumes.new(name)
    vol.filepath = path
    vol.is_sequence = False
    vol.render.precision = "FULL"  # the files are 32-bit; do not halve them
    if not vol.grids.load():
        sys.exit(f"{path}: {vol.grids.error_message}")
    names = [g.name for g in vol.grids]
    if "density" not in names:
        sys.exit(f"{path}: no density grid, only {names}")
    vol.materials.append(material)
    obj = link(bpy.data.objects.new(name, vol))
    obj.location = location
    return obj


def add_outline(slot: int, size: float, material) -> None:
    """The domain's edges, so each plume can be judged against its box."""
    x0 = placement.slot_x(slot, size, GAP)
    bpy.ops.mesh.primitive_cube_add(size=size, location=(x0 + size / 2, size / 2, size / 2))
    obj = bpy.context.active_object
    obj.name = f"{SOLVERS[slot]} domain"
    w = obj.modifiers.new("edges", "WIREFRAME")
    w.thickness = 0.004 * size
    w.use_even_offset = True
    obj.data.materials.append(material)
    unlit(obj)


def add_text(body: str, location, size: float, material) -> None:
    cu = bpy.data.curves.new(body, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.materials.append(material)
    obj = link(bpy.data.objects.new(body, cu))
    obj.location = location
    obj.rotation_euler = (math.pi / 2, 0.0, 0.0)  # face the camera, along −y
    unlit(obj)


def add_sun(name: str, strength: float, angle_deg: float, rotation_deg) -> None:
    light = bpy.data.lights.new(name, "SUN")
    light.energy = strength
    light.angle = math.radians(angle_deg)
    obj = link(bpy.data.objects.new(name, light))
    obj.rotation_euler = tuple(math.radians(a) for a in rotation_deg)


def add_world() -> None:
    world = bpy.data.worlds.new("grey")
    bpy.context.scene.world = world
    if hasattr(world, "use_nodes"):
        world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (WORLD_GREY, WORLD_GREY, WORLD_GREY, 1.0)
    bg.inputs["Strength"].default_value = 1.0


def add_camera(size: float) -> None:
    cam = placement.camera_for(size, GAP, WIDTH / HEIGHT)
    data = bpy.data.cameras.new("camera")
    data.type = "ORTHO"
    data.ortho_scale = cam["ortho_scale"]
    data.clip_end = 20 * size
    obj = link(bpy.data.objects.new("camera", data))
    obj.location = cam["location"]
    obj.rotation_euler = cam["rotation"]
    bpy.context.scene.camera = obj


def build(files: tuple[str, str], size: float, dx: float, caption: str) -> str:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    device = settings(scene)
    smoke = smoke_material()
    ink = flat_material("ink", 0.0)
    edge = flat_material("edge", 0.06)
    for slot, (solver, path) in enumerate(zip(SOLVERS, files, strict=True)):
        # Both files centre voxel i at i·dx (placement.py), so both get the
        # half cell.
        add_volume(solver, path, placement.object_location(slot, size, dx, True, GAP), smoke)
        add_outline(slot, size, edge)
        add_text(solver, placement.label_location(slot, size, GAP), 0.09 * size, ink)
    add_text(caption, placement.caption_location(size, GAP), 0.065 * size, ink)
    # Key from the camera's upper left, a soft fill from its lower right.
    add_sun("key", KEY_STRENGTH, 3.0, (50.0, 0.0, -35.0))
    add_sun("fill", FILL_STRENGTH, 40.0, (100.0, 0.0, 40.0))
    add_world()
    add_camera(size)
    return device


def render(
    ember_dir: str, manta_cache: str, scene_name: str, res: int, size: float, out_dir: str
) -> None:
    dx = size / res
    os.makedirs(out_dir, exist_ok=True)
    devices = set()
    for frame in FRAMES:
        files = frame_files(ember_dir, manta_cache, frame)
        device = build(files, size, dx, f"{scene_name}   {res}³   frame {frame}")
        out = os.path.join(out_dir, f"{scene_name}-{res}-f{frame:03d}.png")
        bpy.context.scene.render.filepath = out
        start = datetime.datetime.now()
        bpy.ops.render.render(write_still=True)
        took = (datetime.datetime.now() - start).total_seconds()
        if not os.path.isfile(out) or os.path.getsize(out) == 0:
            sys.exit(f"render wrote no {out}")
        print(f"render_compare: {out} on {device} in {took:.1f} s", flush=True)
        devices.add(device)
    with open(os.path.join(ember_dir, "render-device"), "w") as fh:
        fh.write(", ".join(sorted(devices)) + "\n")


def shell(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


# Per-scene notes, printed under the scene's heading. They describe what the
# solvers do, from the benchmark's records; keep them to what those show.
NOTES = {
    "plume_collider": (
        "The collider sphere is not drawn; it shows only as the faint outline "
        "the smoke leaves around it."
    ),
    "plume_wind": (
        "Both panels show solver behaviour, not the render. Ember's smoke leaves "
        "through the open +x side from about frames 23–34 and is gone by frame 90 "
        "(`docs/bench/results.md`), which is why its frame-90 panel is empty. "
        "Mantaflow's smoke stays in the domain against +x: at frame 60 its mass "
        "is 0.0842 against Ember's 0.0382 (sum of density × dx³ in these bakes). "
        "Mantaflow's wind acts only on cells that hold smoke, while Ember's moves "
        "all the air (`docs/bench/mantaflow-notes.md`, Wind as ambient airflow)."
    ),
}


def record(path: str) -> str:
    """One line a bake or render left, or a note that it left none."""
    if not os.path.isfile(path):
        return "not recorded"
    with open(path) as fh:
        return fh.read().strip()


def readme(out_dir: str, bake_dir: str, cases: list[str]) -> None:
    ember_commits, manta_commits, devices, sizes = set(), set(), set(), set()
    sections = []
    for case in cases:
        scene_name, res = case.split(":")
        ember_commits.add(record(os.path.join(bake_dir, "ember", f"{scene_name}-{res}", "commit")))
        manta_commits.add(
            record(os.path.join(bake_dir, "mantaflow", f"{scene_name}-{res}", "commit"))
        )
        devices.add(record(os.path.join(bake_dir, "ember", f"{scene_name}-{res}", "render-device")))
        with open(os.path.join(bake_dir, f"{scene_name}-{res}.json")) as fh:
            sizes.add(json.load(fh)["domain_size"])
        images = []
        for frame in FRAMES:
            name = f"{scene_name}-{res}-f{frame:03d}.png"
            if not os.path.isfile(os.path.join(out_dir, name)):
                sys.exit(f"missing {name}: render before writing the README")
            images.append(f"![{scene_name} {res}³ frame {frame}]({name})")
        note = f"{NOTES[scene_name]}\n\n" if scene_name in NOTES else ""
        sections.append(f"## `{scene_name}` ({res}³)\n\n{note}" + "\n\n".join(images) + "\n")
    today = datetime.date.today().isoformat()
    domain = " or ".join(f"{v:g}" for v in sorted(sizes))
    text = f"""# Ember and Mantaflow side by side

- Machine: {shell("sysctl", "-n", "machdep.cpu.brand_string")}
- OS: macOS {shell("sw_vers", "-productVersion")}
- Ember bakes' commit: {", ".join(sorted(ember_commits))}
- Mantaflow bakes' commit: {", ".join(sorted(manta_commits))}
- Blender: {bpy.app.version_string}
- Date: {today}

Written by `just bench-render` (`tests/bench/render_compare.py`), 2b-3b spec §3.
Each image shows one frame of one scene: Ember's density on the left and
Mantaflow's on the right, each in its {domain} m domain drawn as a thin box.
Both are baked from the same scene definition
(`Scene` in `crates/elements-ember/src/bench`, through `benchmark document`
and `benchmark scene-json`), Ember by `elements-cli bake` and Mantaflow by
`tests/bench/mantaflow_scene.py`. Mantaflow bakes frames 1–90 once. Ember is
baked once for each of frames 30, 60 and 90, and each bake simulates from
frame 1 and writes only that frame; the simulation is deterministic, and a
rebake gives byte-identical files. The images are for judging by eye;
nothing here measures them.

## Render settings

- Cycles, {SAMPLES} samples a pixel (adaptive sampling off), seed {SEED}, denoiser
  off, {WIDTH}×{HEIGHT}, Standard view transform. Device:
  {", ".join(sorted(devices))}.
- One orthographic camera looking along +y, so both domains are seen through
  the same projection: no perspective, and depth along y is not visible.
- Both Volume objects read their file's `density` grid at full precision and
  share one Principled Volume material: colour white, **density {DENSITY:g}**
  (a multiplier on the grid value), absorption colour black, no emission, no
  blackbody.
- A sun key light (strength {KEY_STRENGTH:g}, 3° wide) from the camera's upper
  left, a soft sun fill (strength {FILL_STRENGTH:g}, 40° wide) from its lower
  right, and a uniform world of grey {WORLD_GREY:g} at strength 1.
- Placement: both files put voxel i's centre at i·dx, half a cell below the
  cell's centre, so each object is moved by +0.5·dx on every axis
  (`tests/bench/placement.py`). Mantaflow's domain is 1.25 domains to the right
  of Ember's.

**One density scale for both is fair**: while both solvers are still emitting
and the plumes have not yet diverged, the emitted masses match: over frames
12–24, Ember's mass is 0.99–1.04× Mantaflow's across every benchmark run
(`docs/bench/results.md`, Notes, Heat). The scale is not normalised per
solver, so a difference in brightness or opacity is a difference in density.
At density {DENSITY:g}, though, a plume's core is optically thick (τ ≈ 6: the
multiplier times grid values near 1 over a core about 0.3 m across), so the
images compare shape and extent rather than peak density.
Known differences carry through (`results.md`, `mantaflow-notes.md`):
Mantaflow's emitter heat is held at Ember's frame-24 value rather than added
at Ember's rate, so its plume rises faster; Mantaflow's open top boundary layer
is a density sink; and Mantaflow clamps density to [0, 1] at the emitter while
Ember does not, so Ember's densest cells can exceed 1 (behind an optically
thick core, that changes little in the image).

{placement.VERDICT_MARK}

{chr(10).join(sections)}"""
    path = os.path.join(out_dir, "README.md")
    text = text.replace(placement.VERDICT_MARK, placement.kept_verdict(path))
    with open(path, "w") as fh:
        fh.write(text)


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :]
    if args[0] == "--readme":
        readme(args[1], args[2], args[3:])
        return
    ember_dir, manta_cache, scene_name, res, size, out_dir = args
    render(ember_dir, manta_cache, scene_name, int(res), float(size), out_dir)


main()
