# Fire divergence: which variant closes the gap (FT3a, 2026-09-30)

In `fire`, Ember's RMS divergence at frame 60 is 4.0×, 26× and 28×
Mantaflow's at 64³, 128³ and 256³ ([results.md](results.md), Fire). In the
smoke scenes Ember usually leaves less than Mantaflow does. This experiment
changes one thing at a time to find what the gap tracks. No solver change is
committed. It repeats and extends the 64³ probes in
[fire-divergence-investigation.md](fire-divergence-investigation.md) (2026-09-26),
adding 128³ and 256³, peak |u| and fuel.

**Result.** The gap tracks the pressure solve's convergence. The Euler backtrace
has little effect on it. Six MGPCG cycles instead of preview's four close the
gap at 64³ and 128³, and leave 1.6× at 256³. Fuel and flame volume stay within
2% of the baseline. Euler for every grid leaves divergence where it was.
`flame_vorticity = 0` lowers divergence by weakening the flame, which does not
fix anything. The `final` preset goes well below Mantaflow at a cost of about
10× preview's frame time, and changes the fire's size.

## Method

- **Machine:** Apple M1 Max (MacBookPro18,2), macOS 27.2, Metal.
- **Base commit:** `0400839` on `worktree-flamethrower-ft0-ft3`.
- **Scene:** `Scene::fire(res)` from `crates/elements-ember/src/bench/mod.rs`,
  run to frame 60 under preview (1 substep, MGPCG ×4, mass correction on,
  fuel clamped to [0, 10], flame vorticity 12/s). Velocity faces use the Euler
  backtrace while fire burns, and scalars use RK2.
- **Harness:** a scratch `fire-variant RES` mode added to the `benchmark`
  example. It reads variants from environment variables, and it writes no
  result files, so `docs/bench/results/` is untouched. For frame 60 it uses
  the benchmark's own `metrics_run`, so divergence, measured cells, fuel mass
  and flame volume are computed exactly as in `results.md`.
- **Peak |u|:** the largest max |u| over all velocity faces in frames 1–60,
  as `.superpowers/sdd/fire-diagnosis.md` measured it.
- **Frame ms:** the median of frames 2–60 from one timed run of `eval_frame`
  plus a blocking wait. `results.md` uses 120 frames and three runs.
- **Scratch branch:** `scratch/fire-divergence-experiments`, left in place.

| # | variant | code | how to run |
|---|---|---|---|
| 1 | baseline (preview) | `2f21cb4` | `benchmark fire-variant RES` |
| 2 | Euler backtrace for every grid while fire burns | `e1b2fae` (`euler_faces: u32::from(c.fire)` in `kernels/mod.rs`) | `benchmark fire-variant RES` |
| 3 | `flame_vorticity = 0` | `2f21cb4` | `FV=0 benchmark fire-variant RES` |
| 4 | `final` preset (8 substeps, MGPCG ×10) | `2f21cb4` | `QUALITY=final benchmark fire-variant RES` |
| 5 | pressure cycles 6 (preview otherwise) | `2f21cb4` | `CYCLES=6 benchmark fire-variant RES` |

Build with `cargo build --release -p elements-ember --example benchmark` at the
listed commit. The binary is `target/release/examples/benchmark`.

On variant 4: the brief described `final` as "8 substeps, RK2". As shipped,
`final` still traces velocity faces with Euler while fire burns, because
`euler_faces` depends only on fire. It also raises pressure cycles from 4 to
10. It is therefore two changes from preview, and it was run as shipped. The
2026-09-26 probe separated the two changes at 64³ only: 8 substeps with ×4
gave 2.44e-5.

Variant 5 is the one extra variant. The 2026-09-26 probe had named pressure
cycles as the likely cause but had recorded no fuel or peak |u| for it. This
run checks whether the fix changes the fire itself.

## Results

Mantaflow's RMS divergence at frame 60 comes from [results.md](results.md):
1.25e-4 at 64³, 4.60e-5 at 128³ and 9.00e-5 at 256³. Mantaflow's peak |u| is
11.7 m/s at 64³ (fire-diagnosis.md). Mantaflow's fuel at frame 60 is 0.696,
0.517 and 0.443, and its flame volume 0.695, 0.327 and 0.205 m³.

| variant | res | div. RMS f60 (1/s) | × Mantaflow | measured cells | fuel f60 (× baseline) | flame m³ f60 | peak \|u\| m/s (frame) | frame ms | 1-min load before → after |
|---|---|---|---|---|---|---|---|---|---|
| 1 baseline | 64³ | 4.98e-4 | 3.98× | 21,617 | 0.610 | 0.373 | 10.8 (16) | 20.9 | 28.2 → 25.7 |
| 1 baseline | 128³ | 1.19e-3 | 25.8× | 118,033 | 0.599 | 0.343 | 12.9 (27) | 106.4 | 25.7 → 21.4 |
| 1 baseline | 256³ | 2.50e-3 | 27.7× | 782,469 | 0.631 | 0.367 | 14.2 (14) | 867.6 | 18.9 → 9.8 |
| 2 Euler every grid | 64³ | 5.16e-4 | 4.13× | 21,343 | 0.513 (0.84×) | 0.490 | 9.6 (19) | 16.3 | 9.0 → 9.0 |
| 2 Euler every grid | 128³ | 1.16e-3 | 25.1× | 154,333 | 0.659 (1.10×) | 0.578 | 12.0 (31) | 74.0 | 9.0 → 7.7 |
| 2 Euler every grid | 256³ | 2.36e-3 | 26.2× | 845,025 | 0.790 (1.25×) | 0.519 | 12.1 (25) | 547.9 | 7.7 → 6.0 |
| 3 flame_vorticity 0 | 64³ | 5.32e-5 | 0.43× | 4,279 | 0.646 (1.06×) | 0.148 | 3.3 (57) | 20.0 | 9.8 → 9.4 |
| 3 flame_vorticity 0 | 128³ | 3.49e-4 | 7.6× | 42,015 | 0.689 (1.15×) | 0.163 | 3.5 (58) | 103.8 | 9.4 → 9.5 |
| 4 `final` | 64³ | 2.72e-6 | 0.022× | 41,749 | 0.543 (0.89×) | 0.485 | 9.4 (41) | 214.7 | 9.5 → 9.0 |
| 4 `final` | 128³ | 8.06e-6 | 0.18× | 258,014 | 0.764 (1.28×) | 0.449 | 10.3 (48) | 1127.0 | 9.0 → 10.3 |
| 5 pressure ×6 | 64³ | 1.00e-5 | 0.080× | 19,428 | 0.599 (0.98×) | 0.373 | 10.2 (24) | 23.2 | 6.4 → 6.6 |
| 5 pressure ×6 | 128³ | 4.18e-5 | 0.91× | 115,436 | 0.590 (0.99×) | 0.339 | 13.0 (27) | 117.0 | 6.6 → 6.7 |
| 5 pressure ×6 | 256³ | 1.42e-4 | 1.58× | 752,560 | 0.645 (1.02×) | 0.363 | 17.5 (14) | 891.1 | 6.7 → 12.0 |

I skipped `flame_vorticity = 0` and `final` at 256³. They show no
resolution trend that a 256³ run would decide, and `final` at 256³ would take
about 20 minutes.

**Baseline reproduces `results.md`.** At all three resolutions, the RMS
divergence, measured cells and frame-60 fuel match `results.md`'s Ember row to
the printed digits: 4.98e-4 / 1.19e-3 / 2.50e-3, and fuel 0.610 / 0.599 /
0.631. Reruns of variants 1 and 2 at 128³ gave bit-identical metrics, as
expected from one backend on one machine.

**Load.** Every run was under load: the 1-minute load average was between 6.0
and 28.2, and other processes were running. Every frame time is an upper bound
and cannot be compared across rows. Divergence, fuel and peak |u| do not
depend on load.

## What the gap tracks

- **Pressure convergence.** This is the main cause. Changing only the cycle
  count from 4 to 6 lowers divergence 50× at 64³, 28× at 128³ and 18× at
  256³. The fire is unchanged: fuel is 0.98–1.02× the baseline, flame volume
  is within 1.2%, and measured cells are within 10%.
  - At 256³ it still leaves 1.58× Mantaflow's divergence. The 2026-09-26
    probe found 1.50e-4 there with ×10, so extra cycles at one substep stop
    helping at that point. The rest of the 256³ gap is something else.
  - Peak |u| at 256³ rises from 14.2 to 17.5 m/s, at frame 14. That is the
    early core spike, not a runaway, but it is higher than Mantaflow's
    11.7–15.3 m/s at lower resolutions (no 256³ Mantaflow peak is recorded).
- **Not the Euler backtrace.** Tracing every grid with Euler changes
  divergence by −6% to +4%, which is noise-sized next to a 26× gap.
  - It lowers peak |u| (9.6 / 12.0 / 12.1 m/s, matching the diagnosis).
  - It changes the fire. The flame grows 1.3–1.7×, which moves it away from
    Mantaflow at 128³ and 256³. Fuel goes 0.84× the baseline at 64³, but
    1.10× and 1.25× at 128³ and 256³.
  - The handoff's suspect, the Euler velocity trace, therefore does not
    explain the divergence. Moving the scalars onto the same trace does not
    help either.
- **`flame_vorticity = 0` does not fix the gap; it removes most of the
  fire.** Divergence falls 9.4× at 64³ and 3.4× at 128³, and still leaves
  7.6× Mantaflow's at 128³. The measured cells shrink to 20–36% of the
  baseline, flame volume to about 0.4–0.5×, and peak |u| to 3.3–3.5 m/s.
  Flame vorticity drives the fast, rotational core that 4 cycles cannot
  project, but turning it off is not a match for Mantaflow.
- **`final` goes well below Mantaflow**: 0.02× at 64³ and 0.18× at 128³.
  - It costs about 10× preview's frame time: 1,127 ms against 106 ms at 128³,
    under load.
  - It changes the fire: 2.2× the measured cells at 128³, fuel 1.28×, and
    flame 1.3×.
  - It is an offline bake setting, not the fix for preview.

## Side finding: the Euler backtrace's cost

Euler for every grid ran faster than the baseline in this harness.

- I ran it paired at 128³, alternating the two binaries twice at load
  8–10. Euler for every grid took 74.0 and 73.7 ms a frame, and the baseline
  106.4 and 106.6 ms.
- The unpaired 64³ and 256³ runs agree: 16.3 against 20.9 ms, and 548
  against 868 ms.
- Euler for every grid drops the RK2 midpoint sample in each scalar pass, but
  a 30% cut is larger than that alone would explain.
- These runs were under load, so this is not a timing result. It should be
  checked on an idle machine before anyone relies on it.
- If it holds, it matters: it would bring fire preview at 128³ under the
  100 ms budget (risk (n)). It could also pay for the extra pressure cycles:
  ×6 cost about 11 ms at 128³ here (117.0 against 106.4 ms, unpaired and at
  different loads).

## Recommendation (the user decides)

1. **Scope a follow-up task: raise fire's pressure solve to 6 MGPCG cycles.**
   This could be a fire-only preview setting or a new preview default. It is
   the one variant that closes the gap at 64³ and 128³ without changing fuel,
   flame or measured cells, and it needs no solver code, only a preset value.
   - Before deciding, idle-machine timing must show its preview cost at 128³:
     roughly +10% here, while fire preview is already over 100 ms under load.
   - Afterwards, rerun `just bench fire` (Ember and Mantaflow, an hour or more)
     and update `results.md`, `presets-fire.md` and the renders.
2. **Do not adopt Euler for every grid, or `flame_vorticity = 0`, as a
   divergence fix.** Neither closes the gap on equal terms. The first leaves it
   as it was, and the second removes most of the flame.
3. **Optional, same follow-up.** Time Euler for every grid together with ×6
   cycles on an idle machine. If the 30% saving is real, the pair could fit
   fire preview inside 100 ms at 128³, with divergence at Mantaflow's level.
   That pair was not tested here. Adopting it would also change fire's flame
   size (1.3–1.7×), and would contradict the 2b-1 spec's RK2 wording, which
   the 2b-4 exception already relaxes for velocity faces.
4. **The 256³ residual** (1.58× with ×6, and no better with ×10) is not
   explained. It is small next to the original 28× and is not blocking.
