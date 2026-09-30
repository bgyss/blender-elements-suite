# Surface ignition: the front spreads (flamethrower FT4, Task 6)

Question (FT4 spec §4 test 5): under a steady heat source at one end, does a
burnable collider's surface ignite outward, with ignition frames that do not
decrease with distance? The test is emergent, so every scene tried is recorded
here with its outcome, per the spec's rule on honest verdicts.

## Setup

- Test: `the_front_spreads_up_a_wall` in `crates/elements-ember/tests/surface.rs`
  (`cargo nextest run -p elements-ember --test surface --no-capture -- spreads`).
- Machine: Apple M1 Max (Metal). Load average (`uptime`): 12.07, 18.35, 20.91
  before the first run; 14.27, 16.89, 19.73 after the last. Load does not
  affect this result (it is not a timing), only how long the run took.
- Base commit: `9487d23` (Wire the surface into the solver).
- Measurement: at each frame 1..=60, read the solver's `char` output (socket 4)
  and record, for each z row `k`, the first frame at which any cell in the row
  has `char > 0`. Only the wall carries a load, so only wall rows can char.
- Assertions (unchanged from the plan): at least 3 rows ignite; ignition frames
  do not decrease with `k`; the lowest and highest ignited rows differ.

## Scene (fixed across attempts)

- Domain 16×16×32, `domain_size` 4.0, so dx = 0.125 m and the domain is
  2 × 2 × 4 m. 24 fps.
- Wall: `ember.collider`, box `half_extents [0.0625, 0.4, 0.8]` at
  [1.0625, 1.0, 1.0], static, `surface_fuel.load` 4.0. That is one cell thick
  (i = 8; the centre cell has sdf −0.0625), j 5..=10, k 2..=13: 12 rows.
- Heat: `ember.sphere_emitter` at [0.9, 1.0, 0.3], radius 0.15,
  `temperature_rate` 30, density 0, on the wall's −x side at its foot.
- Fuel input 6 connected to a zero-rate emitter (required by input 7).
- Solver: preview preset (one CFL-clamped substep, MGPCG ×4), buoyancy
  temperature 1.0, gas `burning_rate` 1.875, ignition 1.5, max temperature 3.0.
- Nothing is stochastic, so there is no seed.

## Attempts

### Attempt 1: preset defaults (temperature dissipation 0, surface burn rate 2)

Ignition frames by row `(k, frame)`:

```
[(2, 2), (3, 3), (4, 10), (5, 13), (6, 16), (7, 18), (8, 20), (9, 22), (10, 25), (11, 27), (12, 29), (13, 31)]
```

The test passed, but the mutation (surface_gather.wgsl stores 0.0 instead of
the sum) gave **the same list, frame for frame**, and the test still passed.
So the test could not fail: every row was lit by the heat source's own plume,
which never cools (dissipation 0) as it rises along the wall, not by the
surface's fuel. The same mutation did fail the other tests that use it
(`a_full_substep_moves_the_wood_into_the_gas_fuel`,
`gas_above_the_threshold_ignites_only_its_neighbour`), so the shader change
did take effect.

Why the surface's fuel added nothing is inferred from the arithmetic and was
not measured: an interior cell of a wall one cell thick has two fluid neighbours
(edge cells have three or more, so each of theirs receives less), so each
receives `2 · h / 2` ≈ 0.042 fuel per substep, while the gas burns
`1.875 · h` ≈ 0.078 per substep. The fuel is consumed in the substep it
arrives, react drops to 0, and no flame temperature is set.

### Attempt 2: temperature dissipation 8/s, surface burn rate 8/s (kept)

One adjustment, two solver constants changed together. Intended effect (reasoned, not measured separately):

- `temperature_dissipation` 8.0: the heat source's plume cools below
  ignition within a short rise. Burning gas is not affected the same way,
  because the burn resets the temperature to the flame profile (above 1.5)
  wherever fuel and react remain.
- `surface_burn_rate` 8.0: each fluid neighbour of a flat wall face receives
  4 fuel/s, more than the gas burns (1.875/s), so fuel and react persist and
  the flame is hot.

Ignition frames by row `(k, frame)`:

```
[(2, 2), (3, 4), (4, 15), (5, 20), (6, 23), (7, 27), (8, 30), (9, 33), (10, 37), (11, 40), (12, 43), (13, 46)]
```

All 12 wall rows ignite in order from the foot; the front climbs about one row
(0.125 m) every 3–4 frames above row 4, reaching the top row at frame 46.
Rows 2 and 3 light from the heat source directly.

Mutation (surface_gather.wgsl: `textureStore(rate, p, vec4<f32>(0.0, 0.0, 0.0, 0.0))`),
the same as the spec's test 5 mutation (run at commit 1052529; the fix round's
doc-comment line moves the assertion down one line):

```
ignition frames by row: [(2, 2), (3, 4)]
thread 'the_front_spreads_up_a_wall' panicked at crates/elements-ember/tests/surface.rs:887:5:
the front must reach at least 3 wall rows, got [(2, 2), (3, 4)]
test the_front_spreads_up_a_wall ... FAILED
```

Without the gathered fuel only the two rows the heat source reaches ignite.
The shader was restored and `git diff` checked clean.

Margin: the zeroed-gather mutant falls one row short of the 3-row threshold;
the non-decreasing check alone does not discriminate (attempt 1's heater-only
run passed it), so the proof rests on `rows.len() >= 3`, and on another
backend the heater might reach one more row and lose the mutation proof
(llvmpipe has not run).

Attempt 3 and 4 were not needed.

## What this does not measure

- No flame-meets-shack case: FT3b found preview's single substep never brings
  the flamethrower's flame to the shack, so the shot's real behaviour is FT6's.
- No timing.
- Whether both settings are needed: dissipation and surface burn rate were
  changed in one attempt and not tested apart. The zeroed-gather mutant's
  result (rows 2–3 only) shows dissipation alone stops the heater's plume
  above row 3; whether the default surface burn rate of 2 would still spread
  the front with dissipation 8 is unknown.
- One scene, one resolution (16×16×32), one adapter; nothing stochastic, so
  no seed variation. The Linux software-Vulkan CI run has not happened for
  this commit.
- Whether lower rows burn out (char reaching 1) by frame 60 was not checked:
  at 8/s a 4.0 load lasts about 12 frames after ignition, so they probably
  do, but the test only records first ignition.
- The kept scene needs non-default solver constants (dissipation 8, surface
  burn rate 8). With the defaults (attempt 1), fuel from a one-cell wall burns
  off as it arrives and adds no flame; whether the defaults are right for the
  shot is not settled here.
