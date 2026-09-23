# SPDX-License-Identifier: GPL-3.0-or-later
"""Checks of mapping.py's Ember -> Mantaflow conversions. No Blender needed.

Run: python3 tests/bench/test_mapping.py (or python3 -m pytest tests/bench).
The expected numbers are the ones docs/bench/mantaflow-notes.md confirmed by
bake.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapping  # noqa: E402


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def rotate(q: tuple[float, float, float, float], v: tuple[float, float, float]):
    """v rotated by the unit quaternion q = (w, x, y, z): q v q*."""
    w, x, y, z = q
    # t = 2 (u x v), v' = v + w t + u x t, with u = (x, y, z).
    t = (
        2 * (y * v[2] - z * v[1]),
        2 * (z * v[0] - x * v[2]),
        2 * (x * v[1] - y * v[0]),
    )
    return (
        v[0] + w * t[0] + (y * t[2] - z * t[1]),
        v[1] + w * t[1] + (z * t[0] - x * t[2]),
        v[2] + w * t[2] + (x * t[1] - y * t[0]),
    )


def test_rotation_to_turns_plus_z_onto_each_axis() -> None:
    for direction in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1)]:
        q = mapping.rotation_to(direction)
        assert close(math.sqrt(sum(c * c for c in q)), 1.0), f"{direction}: |q| = {q}"
        got = rotate(q, (0.0, 0.0, 1.0))
        assert all(close(g, d) for g, d in zip(got, direction, strict=True)), (
            f"+z turned by rotation_to({direction}) is {got}"
        )


def test_rotation_to_normalises_its_direction() -> None:
    got = rotate(mapping.rotation_to((0.0, 3.0, 4.0)), (0.0, 0.0, 1.0))
    assert all(close(g, d) for g, d in zip(got, (0.0, 0.6, 0.8), strict=True)), got


def test_wind_at_half_a_metre_per_second_squared() -> None:
    strength, direction = mapping.wind((0.5, 0.0, 0.0), 24.0)
    assert close(strength, 0.5 / (0.2 * 24.0)), strength
    assert round(strength, 4) == 0.1042, strength
    assert direction == (1.0, 0.0, 0.0), direction


def test_no_wind_is_zero_strength() -> None:
    assert mapping.wind((0.0, 0.0, 0.0), 24.0) == (0.0, (0.0, 0.0, 1.0))


def test_inflow_adds_the_rate_per_frame_and_holds_frame_24s_heat() -> None:
    density, temperature = mapping.inflow(1.0, 1.0, 24.0)
    assert close(density, 1.0 / 24.0), density
    assert close(temperature, 1.0), temperature


def test_inflow_rejects_a_density_mantaflow_would_clamp() -> None:
    try:
        mapping.inflow(48.0, 1.0, 24.0)
    except ValueError:
        return
    raise AssertionError("a density of 2 per frame was accepted")


def test_buoyancy_scales_by_domain_over_gravity_and_flips_density() -> None:
    alpha, beta = mapping.buoyancy(0.5, 1.0, 2.0, 9.81)
    assert close(beta, 2.0 / 9.81), beta
    assert close(alpha, -0.5 * 2.0 / 9.81), alpha


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} mapping tests passed")
