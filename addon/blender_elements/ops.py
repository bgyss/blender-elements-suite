"""Operators that drive the engine process."""

import atexit
import contextlib
import os
import subprocess
import tempfile

import bpy

from .client import ControlClient, ElementsError, FrameReader

# Module-level engine state. Blender operators are stateless, and the process
# and its sockets must outlive any single invocation.
_state = {"process": None, "client": None, "reader": None}


def _defaults(settings) -> None:
    """Fill in endpoint and channel paths if the user left them blank."""
    base = tempfile.gettempdir()
    if not settings.endpoint:
        settings.endpoint = (
            f"\\\\.\\pipe\\elements-{os.getpid()}"
            if os.name == "nt"
            else os.path.join(base, f"elements-{os.getpid()}.sock")
        )
    if not settings.channel_path:
        settings.channel_path = os.path.join(base, f"elements-{os.getpid()}.bin")


def shutdown_engine() -> None:
    """Tear down the client, reader and process. Safe to call repeatedly."""
    if _state["reader"] is not None:
        _state["reader"].close()
        _state["reader"] = None
    if _state["client"] is not None:
        with contextlib.suppress(ElementsError):
            _state["client"].shutdown()
        _state["client"].close()
        _state["client"] = None
    if _state["process"] is not None:
        _state["process"].terminate()
        try:
            _state["process"].wait(timeout=5)
        except subprocess.TimeoutExpired:
            _state["process"].kill()
        _state["process"] = None


# Defence in depth, not a complete guarantee: `atexit` runs on a normal
# interpreter shutdown, which covers the case that actually orphaned a
# daemon here -- an exception after start_engine() that skipped Blender's
# unregister() (Blender does not guarantee unregister() runs on quit either).
# It does NOT run on `kill -9`, a hard crash, or `os._exit()`. Do not treat
# this as closing that gap; it only narrows it. shutdown_engine() is
# idempotent, so this is safe even when unregister() also calls it in the
# same session.
atexit.register(shutdown_engine)


class ELEMENTS_OT_start_engine(bpy.types.Operator):
    bl_idname = "elements.start_engine"
    bl_label = "Start Engine"
    bl_description = "Launch the Elements engine and load the current graph"

    def execute(self, context):
        settings = context.scene.elements
        _defaults(settings)
        shutdown_engine()

        daemon = bpy.path.abspath(settings.daemon_path)
        if not daemon or not os.path.exists(daemon):
            settings.status = "Engine executable not found"
            self.report({"ERROR"}, settings.status)
            return {"CANCELLED"}

        # The bake CLI lives beside the daemon and is only needed later, by
        # render_frame -> _bake_current_graph_to. Check it here too so a
        # missing second binary is reported at engine start, not the first
        # time the user clicks Render.
        cli = os.path.join(os.path.dirname(daemon), "elements")
        if os.name == "nt":
            cli += ".exe"
        if not os.path.exists(cli):
            settings.status = f"Engine CLI executable not found: {cli}"
            self.report({"ERROR"}, settings.status)
            return {"CANCELLED"}

        try:
            process = subprocess.Popen(
                [
                    daemon,
                    "--endpoint",
                    settings.endpoint,
                    "--channel",
                    bpy.path.abspath(settings.channel_path),
                ],
                stdout=subprocess.PIPE,
                text=True,
            )
            # The daemon prints "ready <endpoint>" once it is listening.
            line = process.stdout.readline()
            if not line.startswith("ready "):
                raise ElementsError("io", f"engine did not start: {line.strip()}")
            _state["process"] = process

            # Contract: `hello()` must be the first message the daemon sees,
            # or it refuses LoadGraph/Render with ErrorKind::ProtocolVersion.
            client = ControlClient(settings.endpoint)
            client.connect()
            ack = client.hello()
            _state["client"] = client
        except (ElementsError, OSError) as e:
            # The process never reached a usable, hello'd state -- there is
            # nothing worth keeping alive.
            shutdown_engine()
            settings.status = str(e)
            self.report({"ERROR"}, settings.status)
            return {"CANCELLED"}

        try:
            loaded = client.load_graph(bpy.path.abspath(settings.graph_path))
            _state["reader"] = FrameReader(bpy.path.abspath(settings.channel_path))
        except ElementsError as e:
            # A bad graph (or a bad channel path) is the document's fault,
            # not the engine's: the daemon process and control connection are
            # healthy, so the panel shows the typed error but leaves the
            # engine running rather than tearing down a working session over
            # a typo the user is about to fix.
            settings.status = f"{e.kind}: {e.message}"
            self.report({"ERROR"}, settings.status)
            return {"CANCELLED"}

        settings.status = (
            f"Running on {ack['adapter']} — {loaded['dims']}, {loaded['nodes']} nodes"
        )
        return {"FINISHED"}


class ELEMENTS_OT_stop_engine(bpy.types.Operator):
    bl_idname = "elements.stop_engine"
    bl_label = "Stop Engine"

    def execute(self, context):
        shutdown_engine()
        context.scene.elements.status = "Engine stopped"
        return {"FINISHED"}


class ELEMENTS_OT_render_frame(bpy.types.Operator):
    bl_idname = "elements.render_frame"
    bl_label = "Render Frame"
    bl_description = "Evaluate the graph once and push the result into a Volume"

    def execute(self, context):
        from .handlers import push_frame_to_volume

        settings = context.scene.elements
        client, reader = _state["client"], _state["reader"]
        if client is None or reader is None:
            settings.status = "Engine is not running"
            self.report({"ERROR"}, settings.status)
            return {"CANCELLED"}

        try:
            frame = client.render(context.scene.frame_current)
        except ElementsError as e:
            # Contract: device_lost means the GPU device is gone, not merely
            # that this render failed. Retrying against it cannot succeed, so
            # tear the engine down and require an explicit restart. Any other
            # error (a bad graph, a transient GPU error) leaves the engine
            # running -- the user can fix the graph and try again.
            if e.kind == "device_lost":
                shutdown_engine()
                settings.status = f"device_lost: {e.message} — restart the engine"
            else:
                settings.status = f"{e.kind}: {e.message}"
            self.report({"ERROR"}, settings.status)
            return {"CANCELLED"}

        try:
            try:
                seq, values = reader.read_latest()
            except ElementsError:
                # Contract: the daemon reallocates the channel file in place
                # when a loaded graph's dims change. The reader raises rather
                # than fault; reopen it against the (now correctly sized)
                # channel and retry once instead of treating this as fatal.
                reader.close()
                reader = FrameReader(bpy.path.abspath(settings.channel_path))
                _state["reader"] = reader
                seq, values = reader.read_latest()

            push_frame_to_volume(context, values, frame["dims"], context.scene.frame_current)
            settings.status = f"Frame {seq} — {frame['dims']}"
        except ElementsError as e:
            settings.status = f"{e.kind}: {e.message}"
            self.report({"ERROR"}, settings.status)
            return {"CANCELLED"}
        except OSError as e:
            # _bake_current_graph_to shells out to the CLI binary beside the
            # daemon; if it is missing, subprocess.run raises FileNotFoundError
            # (an OSError subclass), not ElementsError. Surface it the same
            # way rather than letting it escape as a raw traceback.
            settings.status = f"io: {e}"
            self.report({"ERROR"}, settings.status)
            return {"CANCELLED"}

        return {"FINISHED"}


CLASSES = (
    ELEMENTS_OT_start_engine,
    ELEMENTS_OT_stop_engine,
    ELEMENTS_OT_render_frame,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    shutdown_engine()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
