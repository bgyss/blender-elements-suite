# SPDX-License-Identifier: GPL-3.0-or-later
"""Where render_compare.py puts the two volumes and the camera. No Blender needed.

Both solvers' grids cover the domain [0, L]³ with dx = L / n, and cell i
spans [i·dx, (i+1)·dx] on each axis, so its true centre is (i + 0.5)·dx:

- **Ember.** `elements-cli bake --voxel-size dx` passes dx to
  `elements_io::write_float_grid`, which writes a `UniformScaleMap` (scale dx,
  no translation) and stores the field's index (i, j, k) as VDB voxel
  (i, j, k) with one root child anchored at the origin
  (`crates/elements-io/src/vdb/tree.rs`). So the file puts voxel (i, j, k)'s
  centre at (i·dx, j·dx, k·dx), half a cell below the cell's centre; the
  writer does not include the half cell. (Ember's own cell centres are at
  (i + 0.5)·dx: `Scene::collider_mask` in `crates/elements-ember/src/bench`.)
- **Mantaflow.** The cache's VDB transform is also a `UniformScaleMap` of dx
  with no translation, so voxel centres are at i·dx too
  (`docs/bench/mantaflow-notes.md`, "Index space").

Blender draws a VDB voxel centred on its world position, so both Volume
objects are translated by +0.5·dx on every axis (`cell_offset`) to put each
voxel back on its cell, and each domain then covers its [0, L]³ exactly.

The domains sit side by side along x: slot 0 (Ember) at x offset 0, slot 1
(Mantaflow) at L + gap, gap = gap_frac·L. The camera is orthographic and looks
along +y, so both domains are seen through the same projection, with room
below them for the labels and above them for the caption.
"""

import math

# Space around the two domains, as fractions of L: the labels sit below the
# floor and the caption above the top.
SIDE_MARGIN = 0.1
BELOW = 0.22
ABOVE = 0.14
# How far in front of the domains (along −y) the camera stands, in L.
CAMERA_DISTANCE = 4.0


def cell_offset(dx: float, writer_centres_at_index: bool) -> float:
    """The translation that moves a voxel from where its file puts it to its
    cell's centre: half a cell if the file centres voxel i at i·dx, else 0."""
    return 0.5 * dx if writer_centres_at_index else 0.0


def slot_x(slot: int, domain_size: float, gap_frac: float = 0.25) -> float:
    """The x offset of domain `slot`'s minimum corner."""
    return slot * domain_size * (1.0 + gap_frac)


def object_location(
    slot: int, domain_size: float, dx: float, half_cell: bool, gap_frac: float = 0.25
) -> tuple[float, float, float]:
    """The Volume object's location for domain `slot` (0 Ember, 1 Mantaflow)."""
    off = cell_offset(dx, half_cell)
    return (slot_x(slot, domain_size, gap_frac) + off, off, off)


def camera_for(domain_size: float, gap_frac: float = 0.25, aspect: float = 16 / 9) -> dict:
    """An orthographic camera looking along +y at both domains.

    Returns `location`, `rotation` (XYZ Euler, radians) and `ortho_scale`,
    Blender's width of the view when aspect ≥ 1 (its larger side).
    """
    size = domain_size
    x_lo = -SIDE_MARGIN * size
    x_hi = slot_x(1, size, gap_frac) + size + SIDE_MARGIN * size
    z_lo = -BELOW * size
    z_hi = size + ABOVE * size
    width = x_hi - x_lo
    height = z_hi - z_lo
    # Blender's ortho_scale spans the larger side of the frame.
    scale = max(width, height * aspect) if aspect >= 1 else max(width / aspect, height)
    return {
        "location": (0.5 * (x_lo + x_hi), -CAMERA_DISTANCE * size, 0.5 * (z_lo + z_hi)),
        # Blender's camera looks down its local −z; +90° about x turns that to +y.
        "rotation": (math.pi / 2, 0.0, 0.0),
        "ortho_scale": scale,
    }


def label_location(slot: int, domain_size: float, gap_frac: float = 0.25) -> tuple:
    """Where a domain's label is centred: under the domain, in front of it."""
    size = domain_size
    return (slot_x(slot, size, gap_frac) + 0.5 * size, -0.05 * size, -0.14 * size)


def caption_location(domain_size: float, gap_frac: float = 0.25) -> tuple:
    """Where the caption is centred: above the two domains."""
    size = domain_size
    mid = 0.5 * (slot_x(1, size, gap_frac) + size)
    return (mid, -0.05 * size, size + 0.05 * size)


# The contact sheet's verdict survives regeneration (render_compare --readme).
VERDICT_MARK = "@@VERDICT@@"
PENDING = "Verdict (recorded by the user): _pending_"


def kept_verdict(path: str) -> str:
    """The verdict paragraph already in `path`, so regenerating the README
    never erases a recorded verdict; `PENDING` when there is none."""
    try:
        with open(path) as fh:
            old = fh.read()
    except FileNotFoundError:
        return PENDING
    start = old.find("Verdict (recorded")
    if start < 0:
        return PENDING
    end = old.find("\n## ", start)
    return old[start:end].rstrip() if end >= 0 else old[start:].rstrip()
