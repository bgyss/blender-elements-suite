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


def test_wind_matches_plume_winds_baked_pair() -> None:
    # plume_wind: 1 m/s along +x at 1/s, 24 fps, a 2 m domain. The bake in
    # mantaflow-notes.md ("Wind as ambient airflow") used this pair.
    strength, flow, direction = mapping.wind((1.0, 0.0, 0.0), 1.0, 24.0, 2.0)
    blend = 1.0 - math.exp(-1.0 / 24.0)
    assert close(flow, 12.5 * blend * 2.0 / 24.0), flow
    assert close(strength, 5.0 * blend), strength
    # The values mantaflow-notes.md gives, rounded to six places. (The fill
    # bakes there were passed wind=0.204054, one in the sixth place high.)
    assert close(strength, 0.204053, 1e-6) and close(flow, 0.042511, 1e-6), (strength, flow)
    assert direction == (1.0, 0.0, 0.0), direction


def test_wind_strength_over_flow_sets_the_target_speed() -> None:
    # One frame adds 0.2 * strength - 0.08 * fps * flow / L * u, which is zero
    # at u = |w| whatever the rate.
    for rate in (0.5, 1.0, 4.0):
        strength, flow, _ = mapping.wind((0.0, 3.0, 4.0), rate, 30.0, 2.5)
        assert close(0.2 * strength / (0.08 * 30.0 * flow / 2.5), 5.0), (rate, strength, flow)


def test_no_wind_rate_is_no_field() -> None:
    assert mapping.wind((1.0, 0.0, 0.0), 0.0, 24.0, 2.0) == (0.0, 0.0, (0.0, 0.0, 1.0))


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


def test_fire_converts_ember_rates_to_mantaflows_per_frame_settings() -> None:
    m = mapping.fire(12.0, 1.875, 12.0, 24.0)
    assert close(m["fuel_amount"], 0.5), m  # fuel added per frame, like density
    assert close(m["burning_rate"], 0.75), m  # x 0.4 s per Mantaflow time unit
    assert close(m["flame_vorticity"], 0.5), m  # / fps, as vorticity's mapping


def test_fire_rejects_a_fuel_blender_would_clamp() -> None:
    # Blender clamps fuel_amount to [0, 10] (RNA), and 2 per frame is inside.
    assert close(mapping.fire(48.0, 1.875, 12.0, 24.0)["fuel_amount"], 2.0)
    try:
        mapping.fire(264.0, 1.875, 12.0, 24.0)
    except ValueError:
        return
    raise AssertionError("fuel of 11 per frame was accepted")


def test_fire_rejects_a_burning_rate_blender_would_clamp() -> None:
    # Blender clamps burning_rate to [0.01, 4]: a probe asking for 0 got 0.01.
    try:
        mapping.fire(24.0, 0.0, 12.0, 24.0)
    except ValueError:
        return
    raise AssertionError("a burning rate of 0 was accepted")


def test_ember_fire_defaults_are_blenders_converted_at_24_fps() -> None:
    d = mapping.EMBER_FIRE_DEFAULTS
    m = mapping.fire(24.0, d["burning_rate"], d["flame_vorticity"], 24.0)
    # Blender's defaults, recorded in mantaflow-notes.md "Fire".
    assert close(m["burning_rate"], 0.75) and close(m["flame_vorticity"], 0.5), m
    assert (d["flame_smoke"], d["ignition_temperature"], d["max_temperature"]) == (1.0, 1.5, 3.0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} mapping tests passed")
