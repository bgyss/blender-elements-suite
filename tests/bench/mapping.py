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
    wind_velocity: tuple[float, float, float], wind_rate: float, fps: float, domain_size: float
) -> tuple[float, float, tuple[float, float, float]]:
    """Return a wind field's (strength, flow, direction) for Ember's ambient airflow.

    Ember relaxes the air towards `wind_velocity` (m/s) at `wind_rate` (1/s):
    u += (w - u)(1 - exp(-rate / fps)) each step. `domain_size` is the
    domain's longest side in metres. Assumes one solver step per frame and the
    falloff off (SPHERE, power 0). Mantaflow applies it only in cells that
    hold smoke. (0, 0, (0, 0, 1)) means no field.
    """
    # effect.cc do_physical_effector: a WIND field returns strength * nor /
    # fps (vel_to_sec) minus flow * vel, where fluid.cc
    # update_effectors_task_cb passes the cell's velocity as grid velocity
    # times 1 / resolution, that is u * 0.4 / L for u in m/s (the notes' units
    # rule), and scales the result by 0.2. The script's scaleSpeedFrames and
    # addForceField turn a force F into F * fps m/s per frame, so one frame
    # adds 0.2 * strength - 0.08 * fps * flow / L * u. Matching Ember's blend
    # k = 1 - exp(-rate / fps) and target w gives flow = 12.5 k L / fps and
    # strength = 5 k |w|. Confirmed by the domain-filling wind experiments.
    if wind_rate == 0.0:
        return 0.0, 0.0, (0.0, 0.0, 1.0)
    blend = 1.0 - math.exp(-wind_rate / fps)
    flow = 12.5 * blend * domain_size / fps
    speed = math.sqrt(sum(w * w for w in wind_velocity))
    if speed == 0.0:
        return 0.0, flow, (0.0, 0.0, 1.0)  # a pure drag towards rest
    direction = (
        wind_velocity[0] / speed,
        wind_velocity[1] / speed,
        wind_velocity[2] / speed,
    )
    return 5.0 * blend * speed, flow, direction


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


# One Mantaflow time unit in seconds (docs/bench/mantaflow-notes.md, Velocity
# location and units): Blender's frame length is 2.5 / fps time units.
TIME_UNIT_S = 0.4

# Blender's hard ranges (RNA) for the fire settings. Blender clamps a value
# set outside them without an error, so the mapping refuses one instead.
FUEL_AMOUNT_RANGE = (0.0, 10.0)
BURNING_RATE_RANGE = (0.01, 4.0)
FLAME_VORTICITY_RANGE = (0.0, 2.0)


def fire(
    fuel_rate: float, burning_rate: float, flame_vorticity: float, fps: float
) -> dict[str, float]:
    """Return Mantaflow's fire settings for Ember's: the flow's fuel_amount and
    the domain's burning_rate and flame_vorticity.

    `fuel_rate` is fuel per second at full occupancy, `burning_rate` fuel per
    second and `flame_vorticity` 1/s per unit fuel, all Ember's. Assumes one
    solver step per frame and an additive flow (use_absolute off), as `inflow`
    does. Raises ValueError for a value Blender would clamp.
    """
    # Fuel is emitted like density (the "Fire" probe, emission): fluid.cc
    # apply_inflow_fields adds fuel_amount * emission once a frame, 140.000 a
    # frame into 140 emitter cells at fuel_amount 1. So it is Ember's rate
    # times the frame. The cell's fuel is clamped to [0, 10], not [0, 1].
    #
    # fire.cpp KnProcessBurn subtracts burningRate * dt with dt in time units,
    # a frame being 2.5 / fps of them: the burn probe measured 0.078125 a frame
    # at burning_rate 0.75 and 24 fps, 0.75 * 0.104167. Ember's rate per second
    # is that times fps, so burning_rate / 0.4, and Mantaflow's is Ember's * 0.4.
    #
    # smoke_script.h scales flameVorticity * fuel by timestep / frameLength, as
    # it does vorticity, and extforces.cpp KnConfForce adds that per-cell grid
    # to the uniform strength (str += strGrid), with no dx. So it maps as
    # `vorticity` does: V / fps.
    out = {
        "fuel_amount": fuel_rate / fps,
        "burning_rate": burning_rate * TIME_UNIT_S,
        "flame_vorticity": flame_vorticity / fps,
    }
    for key, (low, high) in (
        ("fuel_amount", FUEL_AMOUNT_RANGE),
        ("burning_rate", BURNING_RATE_RANGE),
        ("flame_vorticity", FLAME_VORTICITY_RANGE),
    ):
        if not low <= out[key] <= high:
            raise ValueError(f"Blender clamps {key} to [{low}, {high}], got {out[key]}")
    return out


# Blender's fire defaults (burning_rate 0.75, flame_smoke 1.0, flame_vorticity
# 0.5, flame_ignition 1.5, flame_max_temp 3.0; read back by the "Fire" probe)
# in Ember's units at 24 fps. flame_smoke and the temperatures have no time
# unit: processBurn uses them per unit of fuel burnt and as the heat it writes,
# and a temperature means the same in both solvers (`buoyancy` maps the
# coefficient, not the temperature). The Rust defaults
# in solver.rs copy these numbers.
EMBER_FIRE_DEFAULTS = {
    "burning_rate": 0.75 / TIME_UNIT_S,
    "flame_smoke": 1.0,
    "flame_vorticity": 0.5 * 24.0,
    "ignition_temperature": 1.5,
    "max_temperature": 3.0,
}
