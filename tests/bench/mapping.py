# SPDX-License-Identifier: GPL-3.0-or-later
"""Ember -> Mantaflow parameter conversions for the 2b-3 benchmark.

Each conversion names its source: a file in github.com/blender/blender (read,
never copied) or an experiment recorded in docs/bench/mantaflow-notes.md,
"Parameter mapping". Plain Python with no bpy import, so it can be checked
outside Blender.
"""

import math

# Ember's inflow is matched to Mantaflow's at this frame (the brief's choice).
MATCH_FRAME = 24


def buoyancy(
    ember_density: float, ember_temperature: float, domain_size: float, gravity: float
) -> tuple[float, float]:
    """Return Mantaflow's (alpha, beta) for Ember's buoyancy coefficients.

    `domain_size` is the domain's longest side in metres and `gravity` the
    magnitude of the scene's gravity in m/s² (Blender's default is 9.81).
    """
    # smoke_script.h divides alpha and beta by the domain size; extforces.cpp
    # addBuoyancy adds -gravity * dt * coefficient, so the acceleration per unit
    # value is |g| * coefficient / L. A positive alpha lifts density, the
    # opposite of Ember's sign. Confirmed by the uniform-fill experiments.
    scale = domain_size / abs(gravity)
    return (0.0 - ember_density) * scale, ember_temperature * scale


def inflow(density_rate: float, temperature_rate: float, fps: float) -> tuple[float, float]:
    """Return the flow object's (density, temperature) for Ember's rates.

    Ember adds rate * h each substep, so at the emitter centre it holds
    rate * MATCH_FRAME / fps at frame MATCH_FRAME (0.99 measured at 64³ for
    rate 1 at 24 fps). Mantaflow's INFLOW holds its flow's values in the
    emitter from the first frame. Its density is clamped to [0, 1].
    """
    # fluid.cc apply_inflow_fields: density_in is clamped to [0, 1], and heat is
    # raised to the flow's temperature. Confirmed by the 64³ plume experiment.
    seconds = MATCH_FRAME / fps
    density, temperature = density_rate * seconds, temperature_rate * seconds
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"Mantaflow clamps flow density to [0, 1], got {density}")
    return density, temperature


def vorticity(ember_confinement: float, fps: float) -> float:
    """Return the domain's vorticity setting for Ember's confinement.

    Assumes one solver step per frame.
    """
    # extforces.cpp KnConfForce adds strength * (N x curl) in grid units with
    # no dx, and smoke_script.h scales strength by dt / frame length. Ember
    # adds h * eps * dx * (N x curl) in SI units. dx cancels, leaving eps / fps.
    # Not confirmed by experiment; the bench scenes use eps = 0.
    return ember_confinement / fps


def wind(
    ember_accel: tuple[float, float, float], fps: float
) -> tuple[float, tuple[float, float, float]]:
    """Return a wind field's (strength, direction) for Ember's acceleration.

    Assumes one solver step per frame, falloff off (SPHERE, power 0), and the
    field's `flow` set to 0. Mantaflow applies it only in cells that hold smoke.
    """
    # fluid.cc update_effectors_task_cb scales the field force by 0.2, and
    # effect.cc do_physical_effector divides it by fps (vel_to_sec). With the
    # script's scaleSpeedFrames and addForceField, the acceleration is
    # 0.2 * strength * fps m/s². Confirmed by the domain-filling wind experiment.
    magnitude = math.sqrt(sum(a * a for a in ember_accel))
    if magnitude == 0.0:
        return 0.0, (0.0, 0.0, 1.0)
    direction = (
        ember_accel[0] / magnitude,
        ember_accel[1] / magnitude,
        ember_accel[2] / magnitude,
    )
    return magnitude / (0.2 * fps), direction


def rotation_to(direction: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Return the (w, x, y, z) quaternion that turns a wind field's blowing
    axis onto `direction`."""
    # effect.cc: a WIND field blows along its object's local +z (efd->nor).
    length = math.sqrt(sum(d * d for d in direction))
    dx, dy, dz = (d / length for d in direction)
    if dz < -1.0 + 1e-9:
        return 0.0, 1.0, 0.0, 0.0  # half a turn about x
    # The half-way quaternion: w = 1 + z . d, axis = z x d = (-dy, dx, 0).
    w, x, y, z = 1.0 + dz, -dy, dx, 0.0
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return w / norm, x / norm, y / norm, z / norm
