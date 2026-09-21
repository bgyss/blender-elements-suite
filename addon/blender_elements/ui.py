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
        layout.prop(settings, "live", toggle=True, icon="REC")
        layout.label(text=settings.status, icon="INFO")


def register() -> None:
    bpy.utils.register_class(VIEW3D_PT_elements)


def unregister() -> None:
    bpy.utils.unregister_class(VIEW3D_PT_elements)
