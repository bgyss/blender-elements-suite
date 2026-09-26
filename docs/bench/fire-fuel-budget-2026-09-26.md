# Fire fuel budget at 128³ and 256³ (2026-09-26)

At frame 60 Ember retains 1.16× and 1.42× Mantaflow's cached fuel at 128³
and 256³. The difference starts on **frame 1**, before a long advection
history: Ember holds 0.03066 and 0.03076 fuel·m³, while Mantaflow holds
0.02314 and 0.01511. Mantaflow's first-frame `fuel_inflow` occupies only
0.02510 and 0.01639 m³, compared with 0.03284 m³ at 64³. A bake with VDB
clipping set to zero gave the same first-frame values. This identifies a
resolution-dependent difference in how the benchmark's mesh flow supplies
fuel, rather than a difference in the mapped per-cell fuel rate. It does not
establish why Blender's flow voxelization changes with resolution.

## Sources and method

The input scene is the checked-in [`fire` benchmark](../../crates/elements-ember/src/bench/mod.rs),
with a sphere centred at (1, 1, 0.3) m and radius 0.2 m. Its Blender twin
uses a mesh `FIRE` inflow, `fuel_amount = 1` per frame, one solver step per
frame, and the default burn of 0.078125 fuel per occupied cell per frame.
The original resumable VDB caches contain `fuel` and `fuel_inflow` through
frame 120. The [budget reader](../../crates/elements-ember/examples/fuel_budget.rs)
reads both grids through the patched VDB reader and reports each frame to 60.
The original cache is in the primary checkout's `target/bench/fire-RES/bake/cache`;
substitute that path for `CACHE_DIR`:

```bash
cargo run --release -p elements-ember --example fuel_budget -- CACHE_DIR 128 60
```

Two additional bakes used the same scene JSON, stopped at frame 60, and set
only Blender's `domain_settings.clipping` to 0 instead of its default 1e-6.
Their output was kept outside the repository. The 128³ bake finished in
46.85 seconds and the 256³ bake in 324.31 seconds on the loaded machine.
These were quality checks; their elapsed times are not benchmark results.

## First-frame source comparison

| Resolution | Mantaflow `fuel_inflow` cells | Mantaflow inflow volume | Mantaflow fuel after burn | Ember fuel after burn |
| --- | ---: | ---: | ---: | ---: |
| 64³ | 1,076 | 0.03284 | 0.03027 | 0.03047 |
| 128³ | 6,580 | 0.02510 | 0.02314 | 0.03066 |
| 256³ | 34,364 | 0.01639 | 0.01511 | 0.03076 |

Volumes and fuel masses are in m³ and fuel·m³ respectively. Every first-frame
Mantaflow inflow cell holds exactly 1 before burn; the reported cell counts
multiply by `(2 / RES)³` to give the volumes. The `clipping = 0` bakes at
128³ and 256³ reproduced these inflow, burn, and fuel totals. Mantaflow's
saved first-frame fuel is therefore about 25% and 51% below Ember's at the
two higher resolutions. The fire source's voxelization is a concrete
benchmark mismatch to examine before changing fuel or burn rates.

## What the 60-frame budget can establish

For cached frame `f`, the reader calculates:

```text
inflow_delta(f) = sum(fuel_inflow[f]) - sum(fuel[f-1])
visible_burn(f) = sum(min(fuel_inflow[f], 0.078125))
postburn_residual(f) = sum(fuel[f]) - sum(fuel_inflow[f]) + visible_burn(f)
```

`fuel[0]` is zero. These terms telescope exactly to cached fuel, but
`inflow_delta` is only an **apparent** emission term and `postburn_residual`
is only an **apparent** advection term. The VDB exporter clips intermediate
grids to smoke. At 128³, frame 3's `fuel_inflow` sum rises from 18,702 in
the default cache to 24,526 with clipping disabled, while the final `fuel`
sum is identical. Calling the difference “advection gain” would be wrong.

| Resolution, cache | Frame-60 fuel | Cumulative inflow delta | Visible burn | Postburn residual |
| --- | ---: | ---: | ---: | ---: |
| 128³, default | 0.51678 | 1.82036 | 0.83172 | −0.47186 |
| 128³, clipping 0 | 0.51678 | 1.91492 | 0.83338 | −0.56475 |
| 256³, default | 0.44330 | 1.52803 | 0.60372 | −0.48101 |
| 256³, clipping 0 | 0.44330 | 1.71920 | 0.61562 | −0.66029 |

All entries use fuel·m³. The final `fuel` sums match between cache settings
at frame 30 and frame 60; at 256³ frame 60 they differ by less than
4×10⁻⁹ fuel·m³. The cumulative intermediate terms shift substantially.
Every saved top-layer `fuel` sum through frame 60 is zero in the original
caches, but that alone cannot separate boundary outflow from advection or
cache effects inside a solver step.

## Conclusion and next evidence

The high-resolution ratio has a source mismatch visible on frame 1. The
stored 60-frame data do not support assigning the remaining difference to a
specific amount of advection loss or outflow. The source voxelization, not
the per-cell `fuel_amount` or burn-rate conversion, should be investigated
first. A solver-side diagnostic before VDB export, or a controlled flow
geometry experiment, is needed for a true emission/advection/outflow budget.
Until then, the frame-60 ratio should remain a measured comparison rather
than a pass/fail mapping verdict.
