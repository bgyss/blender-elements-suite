# SPDX-License-Identifier: GPL-3.0-or-later
"""Checks of placement.py, the render comparison's layout. No Blender needed.

Run: python3 tests/bench/test_placement.py (or python3 -m pytest tests/bench).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import placement  # noqa: E402

L = 2.0
DX = L / 128


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


def view(cam: dict, aspect: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """The camera's visible x and z ranges, from Blender's rule that
    ortho_scale spans the frame's larger side."""
    x, _, z = cam["location"]
    s = cam["ortho_scale"]
    w, h = (s, s / aspect) if aspect >= 1 else (s * aspect, s)
    return (x - w / 2, x + w / 2), (z - h / 2, z + h / 2)


def looks_along(rotation: tuple[float, float, float]) -> tuple[float, float, float]:
    """The camera's local −z turned by an XYZ Euler rotation."""
    ax, ay, az = rotation
    v = (0.0, 0.0, -1.0)
    # X, then Y, then Z, as Blender's XYZ order applies them.
    v = (v[0], v[1] * math.cos(ax) - v[2] * math.sin(ax), v[1] * math.sin(ax) + v[2] * math.cos(ax))
    v = (
        v[0] * math.cos(ay) + v[2] * math.sin(ay),
        v[1],
        -v[0] * math.sin(ay) + v[2] * math.cos(ay),
    )
    v = (v[0] * math.cos(az) - v[1] * math.sin(az), v[0] * math.sin(az) + v[1] * math.cos(az), v[2])
    return v


def test_half_cell_offset_only_when_the_file_centres_voxels_at_the_index() -> None:
    assert close(placement.cell_offset(DX, True), 0.5 * DX), placement.cell_offset(DX, True)
    assert placement.cell_offset(DX, False) == 0.0, placement.cell_offset(DX, False)


def test_slot_0_is_at_the_origin_plus_the_half_cell() -> None:
    got = placement.object_location(0, L, DX, True)
    assert all(close(g, 0.5 * DX) for g in got), got
    assert placement.object_location(0, L, DX, False) == (0.0, 0.0, 0.0)


def test_slot_1_sits_one_domain_and_a_gap_along_x() -> None:
    x, y, z = placement.object_location(1, L, DX, True)
    assert close(x, L + 0.25 * L + 0.5 * DX), x
    assert close(y, 0.5 * DX) and close(z, 0.5 * DX), (y, z)
    x, _, _ = placement.object_location(1, L, DX, False, gap_frac=0.5)
    assert close(x, L + 0.5 * L), x


def test_camera_frames_both_domains_and_their_labels() -> None:
    for aspect in (16 / 9, 1.0, 0.75):
        cam = placement.camera_for(L, aspect=aspect)
        (x_lo, x_hi), (z_lo, z_hi) = view(cam, aspect)
        for slot in (0, 1):
            lo = placement.slot_x(slot, L)
            assert x_lo < lo and lo + L < x_hi, (aspect, slot, (x_lo, x_hi))
            lx, _, lz = placement.label_location(slot, L)
            assert x_lo < lx < x_hi and z_lo < lz < 0, (aspect, slot, lz)
        assert z_lo < 0 < L < z_hi, (aspect, (z_lo, z_hi))
        _, _, cz = placement.caption_location(L)
        assert L < cz < z_hi, (aspect, cz, z_hi)


def test_camera_stands_in_front_and_looks_along_plus_y() -> None:
    cam = placement.camera_for(L)
    assert cam["location"][1] < 0, cam["location"]
    got = looks_along(cam["rotation"])
    assert all(close(g, e, 1e-12) for g, e in zip(got, (0.0, 1.0, 0.0), strict=True)), got


def test_a_recorded_verdict_survives_regeneration() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "README.md")
        assert placement.kept_verdict(path) == placement.PENDING
        with open(path, "w") as fh:
            fh.write("# X\n\nVerdict (recorded by the user, today): fine.\nMore.\n\n## `plume`\n")
        assert placement.kept_verdict(path) == (
            "Verdict (recorded by the user, today): fine.\nMore."
        ), placement.kept_verdict(path)


def test_a_section_verdict_is_kept_apart_from_the_main_one() -> None:
    import tempfile

    label = "Fire verdict"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "README.md")
        with open(path, "w") as fh:
            fh.write(
                "# X\n\nVerdict (recorded by the user): main.\n\n## `plume`\n\n![p](p.png)\n\n"
                "## `fire`\n\n![f](f.png)\n\n"
            )
        assert placement.kept_verdict(path, label, placement.pending(label)) == (
            "Fire verdict: _pending: recorded by the user_"
        )
        with open(path, "a") as fh:
            fh.write("Fire verdict (recorded by the user): flames.\n\n## `later`\n")
        assert placement.kept_verdict(path, label, placement.pending(label)) == (
            "Fire verdict (recorded by the user): flames."
        ), placement.kept_verdict(path, label, placement.pending(label))
        assert placement.kept_verdict(path) == "Verdict (recorded by the user): main."


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} placement tests passed")
