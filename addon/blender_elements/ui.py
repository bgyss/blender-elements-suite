"""The N-panel."""

import bpy


class VIEW3D_PT_elements(bpy.types.Panel):
    bl_label = "Elements"
    bl_idname = "VIEW3D_PT_elements"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Elements"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.elements

        layout.prop(settings, "graph_path")
        layout.prop(settings, "daemon_path")

        row = layout.row(align=True)
        row.operator("elements.start_engine", icon="PLAY")
        row.operator("elements.stop_engine", icon="PAUSE")

        layout.operator("elements.render_frame", icon="FILE_REFRESH")

        # Live is EXPERIMENTAL in Core v1 and labelled as such in the UI rather
        # than only in the docs. Each redraw spawns a full `elements` CLI
        # process on Blender's UI thread, and updating the volume schedules
        # another redraw, so it can run away on anything but a tiny graph.
        # It becomes practical once Ember provides an in-memory volume path.
        live_row = layout.row(align=True)
        live_row.prop(settings, "live", toggle=True, icon="REC", text="Live (experimental)")

        layout.label(text=settings.status, icon="INFO")


def register() -> None:
    bpy.utils.register_class(VIEW3D_PT_elements)


def unregister() -> None:
    bpy.utils.unregister_class(VIEW3D_PT_elements)
