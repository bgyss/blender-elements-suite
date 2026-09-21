"""Run inside Blender:
`blender --background --python tests/blender/test_roundtrip.py -- <zip> <daemon>`.

Installs the built extension, starts the engine, renders one frame, and asserts
the volume data reached Blender. Exits non-zero with a message on failure.
"""

import contextlib
import pathlib
import shutil
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
    try:
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

        # The broken-graph DoD item: a malformed .elements file must fail
        # start_engine with a document error, without leaving no daemon
        # behind -- ops.py's contract is that a bad graph is the document's
        # fault, not the engine's, so the process this call itself launched
        # (and successfully hello'd) must stay alive rather than being torn
        # down, so the user can fix the typo and retry without restarting.
        # start_engine() unconditionally calls shutdown_engine() first, so
        # this is necessarily a fresh daemon process, not the one from the
        # roundtrip above; what matters is that *this* call's daemon survives
        # its own document error.
        bad_graph = tmp / "broken.elements"
        bad_graph.write_text("{ this is not valid json")
        scene.elements.graph_path = str(bad_graph)

        # Blender's operator-call convention: when execute() reports an
        # {'ERROR'} message and returns {'CANCELLED'}, calling the operator
        # from bpy.ops (rather than through the UI) raises a RuntimeError of
        # "Error: <report message>" instead of just returning {'CANCELLED'}
        # -- so the assertions below are on the exception, not the return
        # value.
        try:
            result = bpy.ops.elements.start_engine()
        except RuntimeError as e:
            assert "document" in str(e).lower(), f"exception does not name a document error: {e}"
        else:
            raise AssertionError(
                f"start_engine on a broken graph did not raise; returned {result}"
            )
        assert "document" in scene.elements.status.lower(), (
            f"status does not name a document error: {scene.elements.status!r}"
        )

        ops_module = sys.modules["bl_ext.user_default.blender_elements.ops"]
        process = ops_module._state["process"]
        assert process is not None, "the daemon process must still be tracked"
        assert process.poll() is None, "the daemon process must still be alive"

        print("blender roundtrip ok")
    finally:
        # This finally must run on BOTH the happy path and any exception, so
        # the engine daemon and its temp dir never outlive this process. An
        # `os._exit()` below would skip this block entirely, which is exactly
        # how a prior run orphaned a daemon holding a GPU device for hours --
        # so cleanup happens here, before any exit call, not after it.
        with contextlib.suppress(Exception):
            bpy.ops.elements.stop_engine()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - must surface through Blender's exit code
        print(f"blender roundtrip FAILED: {e}", file=sys.stderr)
        # sys.exit (not os._exit): main()'s finally has already run cleanup,
        # so there is no more state to bypass. sys.exit still raises
        # SystemExit through Blender's --python entry point with a non-zero
        # code, which is all `cargo test` needs to detect the failure; using
        # the harder os._exit here would buy nothing and risks skipping any
        # future cleanup added above this line.
        sys.exit(1)
