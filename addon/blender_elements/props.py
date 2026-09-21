"""Scene-level settings for the Elements add-on."""

import bpy


class ElementsSettings(bpy.types.PropertyGroup):
    graph_path: bpy.props.StringProperty(
        name="Graph",
        description="The .elements document to evaluate",
        subtype="FILE_PATH",
    )
    daemon_path: bpy.props.StringProperty(
        name="Engine",
        description="Path to the elementsd executable",
        subtype="FILE_PATH",
    )
    endpoint: bpy.props.StringProperty(
        name="Endpoint",
        description="Control socket path or named pipe",
    )
    channel_path: bpy.props.StringProperty(
        name="Channel",
        description="Memory-mapped frame channel file",
    )
    status: bpy.props.StringProperty(
        name="Status",
        default="Engine stopped",
    )
    live: bpy.props.BoolProperty(
        name="Live",
        description="Update the viewport volume on every redraw",
        default=False,
    )


def register() -> None:
    bpy.utils.register_class(ElementsSettings)
    bpy.types.Scene.elements = bpy.props.PointerProperty(type=ElementsSettings)


def unregister() -> None:
    del bpy.types.Scene.elements
    bpy.utils.unregister_class(ElementsSettings)
