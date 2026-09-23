"""What the add-on tells the user about an engine error.

No bpy import, so the contract test can check it outside Blender.
"""


def describe(kind: str, message: str) -> tuple[str, bool]:
    """Return the status line for an engine error, and whether Live mode must stop.

    Live mode re-renders on every frame change, so an error that will recur on
    the next frame must turn it off rather than repeat.
    """
    if kind == "device_lost":
        return f"device_lost: {message} — restart the engine", True
    if kind == "out_of_memory":
        return "Out of GPU memory: lower the resolution, then try again", True
    return f"{kind}: {message}", False
