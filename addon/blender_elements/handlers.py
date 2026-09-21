"""Getting engine frames into Blender's volume data.

Blender's Python API cannot build an OpenVDB grid in memory, so the frame is
written to a temporary .vdb through the engine's own writer path and imported.
Core v1 takes the simple, correct route; a direct in-memory path is an Ember
optimisation, not a Core v1 requirement.
"""

import os
import subprocess
import tempfile

import bpy

from .client import ElementsError

VOLUME_NAME = "ElementsVolume"


def _ensure_volume(context) -> bpy.types.Object:
    obj = bpy.data.objects.get(VOLUME_NAME)
    if obj is None or obj.type != "VOLUME":
        data = bpy.data.volumes.new(VOLUME_NAME)
        obj = bpy.data.objects.new(VOLUME_NAME, data)
        context.collection.objects.link(obj)
    return obj


def push_frame_to_volume(context, values, dims) -> bpy.types.Object:
    """Point the Volume object at a freshly baked .vdb of the current graph.

    KNOWN GAP — the shared-memory frame is NOT the source of the geometry.

    `values` and `dims` arrive over the memory-mapped frame channel and are used
    here for the status readout and a length sanity check, and then DISCARDED.
    What Blender actually renders comes from `_bake_current_graph_to()`, which
    shells out to the `elements` CLI — a third process that creates its own GPU
    device and re-evaluates the graph from the `.elements` file on disk.

    The workaround is legitimate: bpy cannot build an OpenVDB grid in memory,
    and the add-on must never reimplement the file format in Python. But the
    consequence is real and is recorded rather than glossed:

    * The design spec's data-flow diagram (section 3.6), which shows the draw
      handler copying the frame into the Volume grid, describes an intent, not
      what ships. Do not cite it as achieved.
    * The frame channel therefore has no production consumer. It is exercised
      end to end only by the Rust test suite and `tests/python/contract.py`,
      both of which are our own model of a reader rather than a real one.
    * The daemon renders the graph it loaded at Start Engine; the CLI bakes the
      graph file as it exists NOW. Edit the document after starting the engine
      and the status line reports the engine's dims while the viewport shows a
      different evaluation. Nothing checks that the two agree.

    Closing this needs an in-memory volume path, which arrives with Ember.
    """
    obj = _ensure_volume(context)

    expected = dims[0] * dims[1] * dims[2]
    if len(values) != expected:
        raise ElementsError("io", f"frame has {len(values)} values, expected {expected}")

    path = os.path.join(tempfile.gettempdir(), f"elements-frame-{os.getpid()}.vdb")
    _bake_current_graph_to(path)

    obj.data.filepath = path
    # Volume has no `.reload()` (unlike Image); force the grid cache to
    # re-read the new file by unloading and reloading it explicitly.
    obj.data.grids.unload()
    obj.data.grids.load()
    return obj


def _bake_current_graph_to(path) -> None:
    """Ask the engine CLI to bake the loaded graph to `path`.

    The CLI lives beside the daemon, since both are built into the same
    cargo target directory and shipped together.
    """
    settings = bpy.context.scene.elements
    cli = os.path.join(os.path.dirname(bpy.path.abspath(settings.daemon_path)), "elements")
    if os.name == "nt":
        cli += ".exe"

    # Bake into a pid-qualified subdirectory of the shared temp dir, not the
    # temp dir itself: two Blender instances baking concurrently would
    # otherwise both write density.0001.vdb into the same directory and
    # collide before either side gets to its pid-qualified os.replace target.
    out_dir = os.path.join(tempfile.gettempdir(), f"elements-bake-{os.getpid()}")
    os.makedirs(out_dir, exist_ok=True)
    # 60s: a Core v1 bake is one frame of a small graph -- generous enough to
    # absorb a slow disk, but short enough that a hung/pathological graph
    # does not block Blender's UI thread indefinitely (execute() runs on it,
    # and there is no way to cancel from the panel while it is blocked).
    try:
        result = subprocess.run(
            [
                cli,
                "bake",
                bpy.path.abspath(settings.graph_path),
                "--out",
                out_dir,
                "--frames",
                "1",
                "--name",
                "density",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as e:
        # subprocess.run() kills the child (and, on POSIX, waits on it) when
        # the timeout fires, so nothing is left running.
        raise ElementsError("io", f"bake timed out after {e.timeout:.0f}s") from e
    if result.returncode != 0:
        raise ElementsError("io", f"bake failed: {result.stderr.strip()}")

    os.replace(os.path.join(out_dir, "density.0001.vdb"), path)


_draw_handle = None
_last_seq = -1


def _on_draw() -> None:
    """Runs on every 3D viewport redraw while `live` is enabled.

    Must never raise into Blender's draw loop: an exception here is reported
    once in the status line and then swallowed, degrading to a stale volume
    rather than a broken viewport.

    UNVERIFIED by the automated test suite: `blender --background` never
    redraws a viewport, so this function is unreachable from
    `tests/blender/test_roundtrip.py`. It must be confirmed by hand in an
    interactive session (open Blender, Start Engine, enable Live, watch the
    volume update, then disable Live and confirm updates stop).
    """
    global _last_seq

    settings = None
    try:
        scene = bpy.context.scene
        settings = getattr(scene, "elements", None)
        if settings is None or not settings.live:
            return

        from . import ops

        reader = ops._state.get("reader")
        client = ops._state.get("client")
        if reader is None or client is None:
            return

        try:
            frame = client.render(scene.frame_current)
        except ElementsError as e:
            if e.kind == "device_lost":
                ops.shutdown_engine()
            raise

        try:
            seq, values = reader.read_latest()
        except ElementsError:
            # The channel was reallocated with different dims; reopen and
            # retry once rather than treating this as fatal.
            from .client import FrameReader

            reader.close()
            reader = FrameReader(bpy.path.abspath(settings.channel_path))
            ops._state["reader"] = reader
            seq, values = reader.read_latest()

        if seq != _last_seq:
            push_frame_to_volume(bpy.context, values, frame["dims"])
            _last_seq = seq
            settings.status = f"Live — frame {seq}"
    except Exception as e:  # noqa: BLE001 - see the docstring
        # `settings` may still be None if bpy.context.scene itself (or the
        # elements PropertyGroup lookup) is what raised -- guard the report
        # so this except block itself cannot raise back into the draw loop.
        if settings is not None:
            settings.live = False
            settings.status = f"Live update stopped: {e}"


def register() -> None:
    global _draw_handle
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _on_draw, (), "WINDOW", "POST_PIXEL"
        )


def unregister() -> None:
    global _draw_handle, _last_seq
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        _draw_handle = None
    _last_seq = -1
