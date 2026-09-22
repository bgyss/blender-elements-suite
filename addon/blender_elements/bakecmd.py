"""The engine CLI's bake command line, built without `bpy`.

Kept apart from `handlers.py`, which imports `bpy`, so the contract test can
exercise it outside Blender.
"""

import os


def bake_command(cli: str, graph_path: str, out_dir: str, frame: int) -> tuple[list[str], str]:
    """Return the argv that bakes `frame` of `graph_path`, and the file it writes.

    Blender frames may be negative and the engine's may not, so negative frames
    clamp to 0. The engine clamps again, up to the document's start frame.
    The CLI zero-pads frame numbers to at least four digits, which `:04d`
    matches exactly, including past frame 9999.
    """
    frame = max(0, int(frame))
    args = [
        cli,
        "bake",
        graph_path,
        "--out",
        out_dir,
        "--frames",
        str(frame),
        "--name",
        "density",
    ]
    return args, os.path.join(out_dir, f"density.{frame:04d}.vdb")
