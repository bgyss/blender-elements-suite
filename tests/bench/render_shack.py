# SPDX-License-Identifier: GPL-3.0-or-later
"""Render one baked shack frame: beauty, fuel and char stills (FT5 spec §3.5).

Run (headless):
  Blender --background --factory-startup --python-exit-code 1 \
      --python tests/bench/render_shack.py -- BAKE_DIR NAME FRAME SCENE_JSON OUT_DIR

BAKE_DIR holds `elements-cli bake --name NAME`'s `NAME.NNNN.vdb`, one file with
every grid of the frame; SCENE_JSON is the scene it was baked from, for the
domain and the shack mesh. Each pass builds a scene from factory settings and
writes OUT_DIR/shack-{pass}-f{FRAME:03}.png:

- `beauty`: smoke from `density`, emission `flame` × FLAME_STRENGTH coloured
  by a blackbody at the Kelvin the `temperature` grid maps to, lit by a key
  and a fill sun over a grey world, with the shack's planks as a grey mesh.
- `fuel`, `char`: the grid alone as white emission (× its PASS_STRENGTH) on
  black, with no lights and no shack, so the still shows only the grid.
  Cycles surface shaders cannot sample a volume grid, so char cannot shade the
  planks here; that is FT6.

The Cycles device is written to OUT_DIR/render-device. The script exits
nonzero on a missing file, a missing grid (names printed), a render that
writes nothing, or a failed image check:

- `beauty` has at least MIN_WARM_PIXELS warm pixels (red over blue): the
  flame rendered, and not just a few stray voxels of it. It proves the flame
  emission, not the temperature map: that is covered by `kelvin`'s test and
  by eye.
- `fuel` and `char` each light some pixels.
- `char`'s lit columns lie within the shack's projected x range, give or take
  CHAR_TOLERANCE_VOXELS. This catches char lit far from the shack, as a wrong
  or swapped grid would be; it does not check where within the shack's width
  char sits, and in practice only the left bound can fail (char is at the
  shack's front face, far from its right side).
- `fuel`'s lit box is at least FUEL_WIDTH_RATIO times as wide as `char`'s and
  not inside it: fuel fills the jet, char only the shack's face. This is the
  check that fails if the fuel pass draws char (the grids' names swapped).
"""

import array
import datetime
import json
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import placement  # noqa: E402
import render_compare as rc  # noqa: E402
import shack_layout as sl  # noqa: E402

PASSES = ("beauty", "fuel", "char")
# Emission per unit of grid value in the fuel and char stills. Chosen from the
# frame-45 bake's measured pixels (FT5 Task 8 report): char is two dim voxels
# (max 0.19) at the shack's face, so it needs far more than fuel (max 1.8, in
# the jet) to clear the lit threshold by a clear margin without fuel saturating.
PASS_STRENGTH = {"fuel": 5.0, "char": 100.0}
# How far outside the shack's x range char may sit, in voxels. The frame-45
# char sits at voxel x 72-74, just in front of the shack's 74-112. The value is
# fitted to that one measurement, not derived from the solver.
CHAR_TOLERANCE_VOXELS = 4
# Measured at frame 45: fuel's lit box spans 423 columns (121-543), char's 27
# (526-552); with the fuel pass reading char it spans 23. Twice char's width
# leaves the real render an 8x margin and a misnamed fuel pass none.
FUEL_WIDTH_RATIO = 2.0
# Measured at frame 45: 8066 warm pixels; with flame strength 0 or a misnamed
# flame attribute, 0. A floor of 1000 (about 1/8 of the measured count) fails a
# flame that renders only in scattered voxels, without tracking the exact count.
MIN_WARM_PIXELS = 1000
# The planks' grey; Principled diffuse, so the flame's light falls on them.
SHACK_GREY = 0.1


def beauty_material() -> bpy.types.Material:
    """Smoke from `density`; emission `flame` × FLAME_STRENGTH, coloured by a
    blackbody at the Kelvin the real `temperature` grid maps to."""
    mat, nt = rc.node_material("beauty")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    pv = nt.nodes.new("ShaderNodeVolumePrincipled")
    pv.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    pv.inputs["Density"].default_value = sl.DENSITY
    pv.inputs["Density Attribute"].default_value = "density"
    pv.inputs["Absorption Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    pv.inputs["Blackbody Intensity"].default_value = 0.0
    flame = nt.nodes.new("ShaderNodeAttribute")
    flame.attribute_name = "flame"
    strength = nt.nodes.new("ShaderNodeMath")
    strength.operation = "MULTIPLY"
    strength.inputs[1].default_value = sl.FLAME_STRENGTH
    temp = nt.nodes.new("ShaderNodeAttribute")
    temp.attribute_name = "temperature"
    kelvin = nt.nodes.new("ShaderNodeMapRange")
    kelvin.clamp = True
    kelvin.inputs["From Min"].default_value = sl.TEMPERATURE_RANGE[0]
    kelvin.inputs["From Max"].default_value = sl.TEMPERATURE_RANGE[1]
    kelvin.inputs["To Min"].default_value = sl.KELVIN_RANGE[0]
    kelvin.inputs["To Max"].default_value = sl.KELVIN_RANGE[1]
    body = nt.nodes.new("ShaderNodeBlackbody")
    nt.links.new(flame.outputs["Fac"], strength.inputs[0])
    nt.links.new(strength.outputs["Value"], pv.inputs["Emission Strength"])
    nt.links.new(temp.outputs["Fac"], kelvin.inputs["Value"])
    nt.links.new(kelvin.outputs["Result"], body.inputs["Temperature"])
    nt.links.new(body.outputs["Color"], pv.inputs["Emission Color"])
    nt.links.new(pv.outputs["Volume"], out.inputs["Volume"])
    return mat


def pass_material(grid: str) -> bpy.types.Material:
    """Emission only: `grid` × PASS_STRENGTH[grid], white, no density."""
    mat, nt = rc.node_material(f"pass-{grid}")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    pv = nt.nodes.new("ShaderNodeVolumePrincipled")
    pv.inputs["Density"].default_value = 0.0
    pv.inputs["Blackbody Intensity"].default_value = 0.0
    pv.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = grid
    scale = nt.nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = PASS_STRENGTH[grid]
    nt.links.new(attr.outputs["Fac"], scale.inputs[0])
    nt.links.new(scale.outputs["Value"], pv.inputs["Emission Strength"])
    nt.links.new(pv.outputs["Volume"], out.inputs["Volume"])
    return mat


def shack_material() -> bpy.types.Material:
    mat, nt = rc.node_material("shack")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (SHACK_GREY, SHACK_GREY, SHACK_GREY, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def triples(values: list) -> list[tuple]:
    """A flat list of numbers, or a list of 3-lists, as a list of 3-tuples."""
    if values and isinstance(values[0], (list, tuple)):
        return [tuple(v) for v in values]
    if len(values) % 3:
        sys.exit(f"mesh list of {len(values)} numbers is not a list of triples")
    return [tuple(values[i : i + 3]) for i in range(0, len(values), 3)]


def add_shack(doc: dict) -> None:
    """The planks the solver collides with, from the scene's mesh collider.

    The collider's `offset` inflates its distance field and is not drawn."""
    node = next((n for n in doc["nodes"] if n["kind"] == "ember.mesh_collider"), None)
    if node is None:
        sys.exit("scene has no ember.mesh_collider node")
    params = node["params"]
    keys = params["transform"]["keys"]
    if len(keys) != 1:
        sys.exit(f"the shack is drawn at one pose, but its transform has {len(keys)} keys")
    tx, ty, tz = keys[0]["translate"]
    verts = [(x + tx, y + ty, z + tz) for x, y, z in triples(params["mesh"]["positions"])]
    faces = triples(params["mesh"]["indices"])
    mesh = bpy.data.meshes.new("shack")
    mesh.from_pydata(verts, [], faces)
    if mesh.validate():
        sys.exit("the shack mesh from the scene is invalid")
    mesh.materials.append(shack_material())
    rc.link(bpy.data.objects.new("shack", mesh))


def add_volume(path: str, location, material) -> None:
    """The frame's one file as one Volume object, after checking its grids."""
    vol = bpy.data.volumes.new("shack-frame")
    vol.filepath = path
    vol.is_sequence = False
    vol.render.precision = "FULL"  # the files are 32-bit; do not halve them
    if not vol.grids.load():
        sys.exit(f"{path}: {vol.grids.error_message}")
    names = [g.name for g in vol.grids]
    missing = sl.missing_grids(names)
    if missing:
        sys.exit(f"{path}: missing grids {missing}, only {names}")
    # Blender resolves the `velocity` prefix to velocity_x/_y/_z (FT5 Task 7).
    vol.velocity_grid = "velocity"
    vol.materials.append(material)
    obj = rc.link(bpy.data.objects.new("shack-frame", vol))
    obj.location = location


def add_black_world() -> None:
    world = bpy.data.worlds.new("black")
    bpy.context.scene.world = world
    # Older Blender needs use_nodes for a world node tree; 5.x deprecates it.
    if hasattr(world, "use_nodes"):
        world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0


def add_camera(cam: dict, extent) -> None:
    data = bpy.data.cameras.new("camera")
    data.type = "ORTHO"
    data.ortho_scale = cam["ortho_scale"]
    data.clip_end = 20 * max(extent)
    obj = rc.link(bpy.data.objects.new("camera", data))
    obj.location = cam["location"]
    obj.rotation_euler = cam["rotation"]
    bpy.context.scene.camera = obj


def domain(doc: dict) -> tuple[tuple[float, float, float], float]:
    """The domain's extent in metres and its cell size."""
    dims = doc["dims"]
    dx = doc["domain_size"] / max(dims)
    return tuple(d * dx for d in dims), dx


def build(pass_name: str, path: str, doc: dict, cam: dict) -> str:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    device = rc.settings(bpy.context.scene)
    extent, dx = domain(doc)
    # The writer centres voxel i at i·dx; the half cell puts it on its cell.
    off = placement.cell_offset(dx, True)
    material = beauty_material() if pass_name == "beauty" else pass_material(pass_name)
    add_volume(path, (off, off, off), material)
    if pass_name == "beauty":
        add_shack(doc)
        # As render_compare: key from the camera's upper left, fill from its lower right.
        rc.add_sun("key", rc.KEY_STRENGTH, 3.0, (50.0, 0.0, -35.0))
        rc.add_sun("fill", rc.FILL_STRENGTH, 40.0, (100.0, 0.0, 40.0))
        rc.add_world()
    else:
        add_black_world()
    add_camera(cam, extent)
    return device


def pixels(path: str) -> tuple[array.array, int, int]:
    img = bpy.data.images.load(path)
    w, h = img.size
    buf = array.array("f", bytes(4 * w * h * 4))
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf, w, h


def lit_stats(px: array.array, threshold: float = sl.LIT_THRESHOLD) -> str:
    peak = max(max(px[i], px[i + 1], px[i + 2]) for i in range(0, len(px), 4))
    lit = sum(1 for i in range(0, len(px), 4) if max(px[i], px[i + 1], px[i + 2]) > threshold)
    return f"peak {peak:.4f}, {lit} pixels over {threshold:g}"


def render_pass(pass_name: str, path: str, doc: dict, cam: dict, out_dir: str, frame: int):
    """Render one still and run its own check; returns (device, lit box)."""
    device = build(pass_name, path, doc, cam)
    out = os.path.join(out_dir, sl.still_name(pass_name, frame))
    bpy.context.scene.render.filepath = out
    start = datetime.datetime.now()
    bpy.ops.render.render(write_still=True)
    took = (datetime.datetime.now() - start).total_seconds()
    if not os.path.isfile(out) or os.path.getsize(out) == 0:
        sys.exit(f"render wrote no {out}")
    print(f"render_shack: {out} on {device} in {took:.1f} s", flush=True)
    px, w, h = pixels(out)
    bbox = sl.lit_bbox(px, w, h)
    print(f"render_shack: {pass_name}: {lit_stats(px)}, lit box {bbox}", flush=True)
    if pass_name == "beauty":
        warm = sl.warm_pixels(px, w, h)
        print(f"render_shack: beauty: {warm} warm pixels", flush=True)
        if warm < MIN_WARM_PIXELS:
            sys.exit(
                f"beauty still has {warm} warm pixels, under {MIN_WARM_PIXELS}: "
                "the flame did not render"
            )
    elif bbox is None:
        sys.exit(f"{pass_name} still lights no pixels")
    return device, bbox


def check_char(char: tuple, fuel: tuple, cam: dict, dx: float, width: int) -> None:
    pad = CHAR_TOLERANCE_VOXELS * dx
    lo = sl.project_x(sl.SHACK_AT[0] - sl.SHACK_SIZE[0] / 2 - pad, cam, width)
    hi = sl.project_x(sl.SHACK_AT[0] + sl.SHACK_SIZE[0] / 2 + pad, cam, width)
    cols = f"{lo:.1f}-{hi:.1f} (±{CHAR_TOLERANCE_VOXELS} voxels)"
    print(f"render_shack: shack columns {cols}", flush=True)
    if not (lo <= char[0] and char[2] <= hi):
        sys.exit(f"char box {char} lies outside the shack's columns {lo:.1f}-{hi:.1f}")
    # Boxes are (min_x, min_y, max_x, max_y); fuel ⊆ char means fuel reaches nowhere char does not.
    if fuel[0] >= char[0] and fuel[1] >= char[1] and fuel[2] <= char[2] and fuel[3] <= char[3]:
        sys.exit(f"fuel box {fuel} lies inside char box {char}: are the grids swapped?")
    fuel_w, char_w = fuel[2] - fuel[0] + 1, char[2] - char[0] + 1
    print(f"render_shack: fuel box {fuel_w} columns wide, char {char_w}", flush=True)
    if fuel_w < FUEL_WIDTH_RATIO * char_w:
        sys.exit(
            f"fuel box {fuel} is {fuel_w} columns wide, under {FUEL_WIDTH_RATIO:g}x char's "
            f"{char_w} ({char}): are the grids swapped?"
        )


def main() -> None:
    bake_dir, name, frame_arg, scene_json, out_dir = sys.argv[sys.argv.index("--") + 1 :]
    frame = int(frame_arg)
    path = sl.bake_file(bake_dir, name, frame)
    if not os.path.isfile(path):
        sys.exit(f"missing {path}")
    with open(scene_json) as fh:
        doc = json.load(fh)
    extent, dx = domain(doc)
    cam = sl.camera(extent, rc.WIDTH / rc.HEIGHT)
    os.makedirs(out_dir, exist_ok=True)
    devices, boxes = set(), {}
    for pass_name in PASSES:
        device, boxes[pass_name] = render_pass(pass_name, path, doc, cam, out_dir, frame)
        devices.add(device)
    check_char(boxes["char"], boxes["fuel"], cam, dx, rc.WIDTH)
    with open(os.path.join(out_dir, "render-device"), "w") as fh:
        fh.write(", ".join(sorted(devices)) + "\n")


if __name__ == "__main__":
    main()
