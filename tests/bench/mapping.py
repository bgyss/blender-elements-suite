# SPDX-License-Identifier: GPL-3.0-or-later
"""Ember -> Mantaflow parameter conversions for the 2b-3 benchmark.

Each conversion names its source: a file in github.com/blender/blender (read,
never copied) or an experiment recorded in docs/bench/mantaflow-notes.md,
"Parameter mapping". Plain Python with no bpy import, so it can be checked
outside Blender.
"""

import math

# Ember's temperature at the emitter is matched to Mantaflow's at this frame.
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

    The flow must be additive (`use_absolute = False`), with
    `surface_distance = 0` and `volume_density = 1`, and the domain must take
    one solver step per frame.

    Density is added each step, so it gets Ember's rate times the step:
    density_rate / fps. Temperature cannot be added up: Mantaflow raises the
    emitter's heat to the flow's temperature and never above it. It is set to
    Ember's emitter-centre value at MATCH_FRAME, temperature_rate *
    MATCH_FRAME / fps.
    """
    # fluid.cc: before each frame's emission the inflow grids are reset to the
    # current grids, then apply_inflow_fields adds density * emission (clamped
    # to [0, 1]) when use_absolute is off, and sets heat with ADD_IF_LOWER,
    # which stops at the flow's temperature. sample_mesh gives emission 1 inside
    # the mesh (volume_density) and a falloff to surface_distance cells
    # outside, so surface_distance = 0 emits Ember's sphere volume. Confirmed at
    # 64³: mass 0.97-1.03 of Ember's over frames 12-60.
    density = density_rate / fps
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"Mantaflow clamps flow density to [0, 1], got {density}")
    return density, temperature_rate * MATCH_FRAME / fps


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
