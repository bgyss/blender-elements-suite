# SPDX-License-Identifier: GPL-3.0-or-later
"""Layout helpers for the flamethrower-vs-shack render. No Blender needed.

The look constants (temperature and Kelvin ranges, flame strength, density) are
chosen by eye on the frame-45 bake and recorded here; they are not derived from
the solver. The shack constants mirror `crates/elements-ember/tests/jet_shack.rs`
and are guarded against `examples/jet_shack_render.elements` by the tests.
"""

import math
from collections.abc import Iterable, Sequence

# The eight grids `elements-cli bake` writes into each frame's file.
GRIDS = (
    "density",
    "flame",
    "temperature",
    "fuel",
    "char",
    "velocity_x",
    "velocity_y",
    "velocity_z",
)

# Solver temperature is in [0, max_temperature]; mapped to a blackbody range.
TEMPERATURE_RANGE = (0.0, 3.0)
KELVIN_RANGE = (1000.0, 2400.0)
FLAME_STRENGTH = 8.0
DENSITY = 20.0

# The shack's footprint centre on the floor and its size, in metres.
SHACK_AT = (1.45, 0.5, 0.0)
SHACK_SIZE = (0.6, 0.5, 0.5)


def kelvin(temperature: float) -> float:
    """Linear map of solver temperature onto KELVIN_RANGE, clamped."""
    t_lo, t_hi = TEMPERATURE_RANGE
    k_lo, k_hi = KELVIN_RANGE
    t = min(max((temperature - t_lo) / (t_hi - t_lo), 0.0), 1.0)
    return k_lo + t * (k_hi - k_lo)


def bake_file(bake_dir: str, name: str, frame: int) -> str:
    return f"{bake_dir}/{name}.{frame:04d}.vdb"


def still_name(kind: str, frame: int) -> str:
    return f"shack-{kind}-f{frame:03d}.png"


def missing_grids(present: Iterable[str]) -> list[str]:
    have = set(present)
    return [g for g in GRIDS if g not in have]


def camera(extent: tuple[float, float, float], aspect: float, margin: float = 0.1) -> dict:
    """An orthographic camera looking along +y at a domain [0, extent]."""
    scale = 1.0 + 2.0 * margin
    return {
        "location": (extent[0] / 2, -4.0 * max(extent), extent[2] / 2),
        # Blender's camera looks down its local -z; +90 degrees about x turns that to +y.
        "rotation": (math.pi / 2, 0.0, 0.0),
        # Blender's ortho_scale spans the frame's larger side (the width for aspect >= 1).
        "ortho_scale": max(extent[0] * scale, extent[2] * scale * aspect),
    }


def warm_pixels(rgba: Sequence[float], threshold: float = 0.05) -> int:
    """Count pixels whose red exceeds blue by more than `threshold`."""
    return sum(1 for i in range(0, len(rgba), 4) if rgba[i] - rgba[i + 2] > threshold)


def lit_bbox(
    rgba: Sequence[float], width: int, height: int, threshold: float = 0.02
) -> tuple[int, int, int, int] | None:
    """(min_x, min_y, max_x, max_y) of pixels with max(R, G, B) > threshold.

    Blender's pixel rows run bottom-up, so row 0 is the bottom of the image.
    """
    xs: list[int] = []
    ys: list[int] = []
    for y in range(height):
        for x in range(width):
            i = (y * width + x) * 4
            if max(rgba[i], rgba[i + 1], rgba[i + 2]) > threshold:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))
