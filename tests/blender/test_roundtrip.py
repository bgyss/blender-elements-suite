"""Run inside Blender:
`blender --background --python tests/blender/test_roundtrip.py -- <zip> <daemon>`.

Installs the built extension, starts the engine, renders one frame, and asserts
the volume data reached Blender. Exits non-zero with a message on failure.
"""

import os
import pathlib
import sys
import tempfile

import bpy

GRAPH = """{
  "version": 1,
  "dims": [8, 8, 8],
  "nodes": [
    { "id": 0, "kind": "core.noise_field", "params": { "seed": 7 } },
    { "id": 1, "kind": "core.output", "params": {} }
  ],
  "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
  "output": 1
}"""


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    zip_path, daemon_path = argv[0], argv[1]

    bpy.ops.extensions.package_install_files(
        filepath=zip_path, repo="user_default", enable_on_install=True
    )

    # The extension imports lazily on first use (e.g. the first operator
    # call below), so there is nothing meaningful to assert about
    # bl_ext.user_default.blender_elements being in sys.modules yet.

    tmp = pathlib.Path(tempfile.mkdtemp())
    graph = tmp / "noise.elements"
    graph.write_text(GRAPH)

    scene = bpy.context.scene
    scene.elements.graph_path = str(graph)
    scene.elements.daemon_path = daemon_path
    scene.elements.endpoint = str(tmp / "control.sock")
    scene.elements.channel_path = str(tmp / "frame.bin")

    result = bpy.ops.elements.start_engine()
    assert result == {"FINISHED"}, f"start_engine returned {result}: {scene.elements.status}"

    result = bpy.ops.elements.render_frame()
    assert result == {"FINISHED"}, f"render_frame returned {result}: {scene.elements.status}"

    volumes = [o for o in bpy.data.objects if o.type == "VOLUME"]
    assert volumes, "no Volume object was created"
    assert "density" in {g.name for g in volumes[0].data.grids}, "no density grid"

    bpy.ops.elements.stop_engine()
    print("blender roundtrip ok")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - must surface through Blender's exit code
        print(f"blender roundtrip FAILED: {e}", file=sys.stderr)
        os._exit(1)
