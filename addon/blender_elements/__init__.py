"""The Elements Blender add-on package.

`client.py` is the add-on's only engine-facing code: a pure stdlib
reimplementation of the engine daemon's control (NDJSON) and data
(memory-mapped frame channel) contracts. See `client.py` for details.

Relative imports throughout: extensions are imported as
bl_ext.<repo>.<id>, so absolute imports of this package's modules fail.

The other submodules (`props`, `ops`, `ui`, `handlers`) import `bpy` at module
scope, which does not exist outside Blender. The contract test in
`tests/python/contract.py` imports `blender_elements.client` directly, on a
plain stdlib Python, so this top-level `__init__.py` must not import those
submodules eagerly -- only `register`/`unregister`, which Blender calls after
`bpy` is available, pull them in.
"""


def register() -> None:
    from . import handlers, ops, props, ui

    for module in (props, ops, ui, handlers):
        module.register()


def unregister() -> None:
    from . import handlers, ops, props, ui

    for module in reversed((props, ops, ui, handlers)):
        module.unregister()
