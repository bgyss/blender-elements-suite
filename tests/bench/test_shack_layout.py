# SPDX-License-Identifier: GPL-3.0-or-later
"""Checks of shack_layout.py, the shack render's layout helpers. No Blender needed.

Run: python3 tests/bench/test_shack_layout.py (or python3 -m pytest tests/bench).
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shack_layout as sl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "..", "..", "examples", "jet_shack_render.elements")


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def view(cam: dict, aspect: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """The camera's visible x and z ranges, from Blender's rule that
    ortho_scale spans the frame's larger side."""
    x, _, z = cam["location"]
    s = cam["ortho_scale"]
    w, h = (s, s / aspect) if aspect >= 1 else (s * aspect, s)
    return (x - w / 2, x + w / 2), (z - h / 2, z + h / 2)


def test_kelvin_maps_linearly_and_clamps() -> None:
    assert sl.kelvin(0.0) == 1000.0, sl.kelvin(0.0)
    assert sl.kelvin(3.0) == 2400.0, sl.kelvin(3.0)
    assert close(sl.kelvin(1.5), 1700.0), sl.kelvin(1.5)
    assert sl.kelvin(-1.0) == 1000.0, sl.kelvin(-1.0)
    assert sl.kelvin(99.0) == 2400.0, sl.kelvin(99.0)


def test_bake_file_names_the_frame_with_four_digits() -> None:
    assert sl.bake_file("d", "shack", 7) == "d/shack.0007.vdb"
    assert sl.bake_file("d", "shack", 12345) == "d/shack.12345.vdb"


def test_still_name() -> None:
    assert sl.still_name("beauty", 45) == "shack-beauty-f045.png"


def test_missing_grids_lists_the_absent_in_order() -> None:
    assert sl.missing_grids(sl.GRIDS) == []
    assert sl.missing_grids(["density"]) == [g for g in sl.GRIDS if g != "density"]


def test_camera_frames_the_domain_looking_along_plus_y() -> None:
    cam = sl.camera((2.0, 1.0, 1.0), 16 / 9)
    (x_lo, x_hi), (z_lo, z_hi) = view(cam, 16 / 9)
    assert x_lo <= 0 and x_hi >= 2.0, (x_lo, x_hi)
    assert z_lo <= 0 and z_hi >= 1.0, (z_lo, z_hi)
    assert cam["location"][1] < 0, cam["location"]
    assert cam["rotation"] == (math.pi / 2, 0.0, 0.0), cam["rotation"]


def test_warm_pixels_counts_red_over_blue() -> None:
    grey = (0.5, 0.5, 0.5, 1.0)
    orange = (1.0, 0.4, 0.1, 1.0)
    rgba = [c for px in (orange, grey, grey, grey) for c in px]
    assert sl.warm_pixels(rgba, 2, 2) == 1, sl.warm_pixels(rgba, 2, 2)


def test_warm_pixels_threshold_is_strict() -> None:
    # Red minus blue is exactly the threshold in the first pixel (not counted)
    # and above it in the second; both are exact in binary.
    exact = [0.25, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0]
    assert sl.warm_pixels(exact, 2, 1, threshold=0.25) == 1, sl.warm_pixels(exact, 2, 1, 0.25)


def test_pixel_helpers_reject_a_buffer_of_the_wrong_size() -> None:
    for call in (lambda: sl.warm_pixels([0.0] * 12, 2, 2), lambda: sl.lit_bbox([0.0] * 12, 2, 2)):
        try:
            call()
        except AssertionError:
            continue
        raise AssertionError("a 3-pixel buffer passed as 2x2")


def test_project_x_maps_the_view_onto_pixel_columns() -> None:
    cam = sl.camera((2.0, 1.0, 1.0), 16 / 9)
    (x_lo, x_hi), _ = view(cam, 16 / 9)
    assert close(sl.project_x(x_lo, cam, 960), 0.0), sl.project_x(x_lo, cam, 960)
    assert close(sl.project_x(x_hi, cam, 960), 960.0), sl.project_x(x_hi, cam, 960)
    assert close(sl.project_x(1.0, cam, 960), 480.0), sl.project_x(1.0, cam, 960)
    # The domain spans 2 / 2.4 of the width, centred: x = 1.5 is 0.25 of it right.
    assert close(sl.project_x(1.5, cam, 960), 480.0 + 0.5 / 2.4 * 960), sl.project_x(1.5, cam, 960)


def test_lit_bbox_treats_row_0_as_the_bottom() -> None:
    w, h = 4, 3
    img = [0.0] * (w * h * 4)
    for x, y in ((1, 0), (3, 2)):
        i = (y * w + x) * 4
        img[i : i + 4] = [0.5, 0.5, 0.5, 1.0]
    assert sl.lit_bbox(img, w, h) == (1, 0, 3, 2), sl.lit_bbox(img, w, h)
    assert sl.lit_bbox([0.0] * (w * h * 4), w, h) is None


def test_shack_constants_match_the_scene() -> None:
    with open(SCENE) as fh:
        doc = json.load(fh)
    node = next(n for n in doc["nodes"] if n["kind"] == "ember.mesh_collider")
    got = node["params"]["transform"]["keys"][0]["translate"]
    assert all(close(g, e) for g, e in zip(got, (1.45, 0.5, 0.0), strict=True)), got
    assert sl.SHACK_AT == (1.45, 0.5, 0.0), sl.SHACK_AT
    assert sl.SHACK_SIZE == (0.6, 0.5, 0.5), sl.SHACK_SIZE


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} shack layout tests passed")
