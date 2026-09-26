# Fire divergence investigation (2026-09-26)

The frame-60 fire divergence gap is largely pressure convergence under the
one-substep preview preset. Increasing only the MGPCG pressure solve from four
to six cycles reduced Ember's RMS divergence by 50× at 64³, 28× at 128³, and
18× at 256³. At 128³ this reached the recorded Mantaflow level. The default
remains four cycles because fire preview already exceeds its 100 ms budget;
six cycles cost more in a short, loaded-machine timing probe.

## Method

The probe uses the checked-in [`fire` scene](../../crates/elements-ember/src/bench/mod.rs)
and the benchmark's existing per-frame [metrics](../../crates/elements-ember/examples/benchmark.rs).
It stops at frame 60, reads the solver state, and reports the same divergence,
measured-cell count, fuel mass, and flame volume as the full benchmark. The
pressure-cycle comparison changes only `scene.solver.pressure_cycles`; the
projection and advection code are unchanged. The Metal GPU supplied the local
measurements. Mantaflow numbers below are from the prior
[benchmark results](results.md), not a new Mantaflow bake.

To reproduce a pressure comparison without overwriting `docs/bench/results/`:

```bash
cargo run --release -p elements-ember --example benchmark -- fire-probe 128 4
cargo run --release -p elements-ember --example benchmark -- fire-probe 128 6
```

| Resolution | Mantaflow RMS | Ember ×4 RMS | Ember ×6 RMS | Ember ×10 RMS |
| --- | ---: | ---: | ---: | ---: |
| 64³ | 0.00012525 | 0.00049834 | 0.00001001 | 0.00001106 |
| 128³ | 0.00004603 | 0.00118743 | 0.00004176 | 0.00004401 |
| 256³ | 0.00009001 | 0.00249672 | 0.00014209 | 0.00015014 |

At 128³, measured cells were 118,033 with four cycles, 115,436 with six,
and 117,602 with ten. Flame volume was 0.343, 0.339, and 0.334 m³,
respectively. These similar populations and flame volumes make the pressure
comparison more informative than a divergence number alone. At 256³, Ember
with six cycles still has about 1.58× Mantaflow's recorded RMS divergence.

The short 128³ timing probe measured medians of 123.8, 134.2, and 139.2 ms
per frame for four, six, and ten cycles. These are one run each, with a
60-frame scene and no idle-machine guarantee; they establish the local cost
direction, not a preview gate result. The earlier full benchmark measured
103.5 ms for fire preview [under load](presets-fire.md), already above the
100 ms target.

## Other isolated probes

These one-off variants used the same frame-60 metrics path at 64³. They were
removed after measurement; only the pressure-cycle probe remains in the
benchmark example.

| Change from preview | RMS divergence | Measured cells | Flame volume |
| --- | ---: | ---: | ---: |
| None | 0.00049834 | 21,617 | 0.373 m³ |
| Euler backtrace for scalars too | 0.00051624 | 21,343 | 0.490 m³ |
| Flame vorticity set to zero | 0.00005316 | 4,279 | 0.148 m³ |
| Eight substeps, four pressure cycles | 0.00002444 | 42,483 | 0.456 m³ |
| Final preset (eight substeps, ten cycles) | 0.00000272 | 41,749 | 0.485 m³ |

Euler on all grids did not close the gap. Removing flame vorticity reduced
divergence but also shrank the measured smoke region and flame substantially,
so it is not evidence for an equivalent fire. Eight substeps also changed the
flow and costs several solves per frame. Pressure cycles give the clearest
isolated improvement while preserving the scene's approximate size.

## Decision gate

Do not silently change the preview preset to six cycles. First run the idle
machine preview checks for both fire and colliders, then decide whether the
quality gain warrants higher preview latency or a separate fire quality
setting. If six cycles is chosen, run the full `just bench fire` comparison,
update the benchmark and preset records, and review the side-by-side render.
The pressure probe establishes a likely cause of the divergence gap, not a
complete visual or performance acceptance result.
