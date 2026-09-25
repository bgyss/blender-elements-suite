//! The Mantaflow benchmark's result files (2b-3 spec §6).
//!
//! Each run of either solver writes one `RunSummary` as JSON, with a CSV of
//! its per-frame metrics beside it for plotting. The report mode reads the
//! JSON files back to build the results table.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::metrics::{FLAME_THRESHOLD, FrameMetrics};
use crate::solver::{PressureSolve, PressureSolver, Quality};

/// One solver's run of one scene at one resolution.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RunSummary {
    /// "ember" or "mantaflow".
    pub solver: String,
    pub scene: String,
    pub resolution: u32,
    /// Timed runs; the metrics come from one further, untimed run.
    pub runs: u32,
    /// Median over runs of each run's median frame time, frame 1 excluded.
    pub frame_ms_median: f64,
    pub frame_ms_min: f64,
    pub frame_ms_max: f64,
    /// Peak GPU (Ember) or process (Mantaflow) memory, bytes.
    pub peak_bytes: u64,
    /// 1-minute load average before and after the timed runs.
    pub load_before: f64,
    pub load_after: f64,
    /// Blender's version string for Mantaflow runs; `None` for Ember.
    pub blender: Option<String>,
    /// The Ember commit the run was built from, with `-dirty` for a changed
    /// tree. The report refuses to mix commits within a scene.
    pub commit: String,
    /// One entry per frame, frame 1 first.
    pub frames: Vec<FrameMetrics>,
    /// `drift` from frame 60, one entry per frame from 60 on.
    pub drift: Vec<f64>,
}

/// `{dir}/{solver}-{scene}-{resolution}.json`.
pub fn summary_path(dir: &Path, solver: &str, scene: &str, resolution: u32) -> PathBuf {
    dir.join(format!("{solver}-{scene}-{resolution}.json"))
}

/// Write `s` as JSON, and its per-frame metrics as a CSV with the same stem.
pub fn write_summary(dir: &Path, s: &RunSummary) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = summary_path(dir, &s.solver, &s.scene, s.resolution);
    let json = serde_json::to_string_pretty(s).map_err(std::io::Error::other)?;
    std::fs::write(&path, json + "\n")?;
    std::fs::write(path.with_extension("csv"), frames_csv(&s.frames))
}

/// A header row, then one row per frame (frame 1 first): `frame`, then each
/// `FrameMetrics` field in declaration order, with `None` as an empty cell.
fn frames_csv(frames: &[FrameMetrics]) -> String {
    let opt = |v: Option<f64>| v.map(|v| v.to_string()).unwrap_or_default();
    let mut out = String::from(
        "frame,divergence_max,divergence_rms,kinetic_energy,vorticity,measured_cells,\
         mass,mass_inside,centroid_m,top_m,outflow_rate,fuel_mass,flame_volume\n",
    );
    for (n, f) in frames.iter().enumerate() {
        let _ = writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            n + 1,
            f.divergence_max,
            f.divergence_rms,
            f.kinetic_energy,
            f.vorticity,
            f.measured_cells,
            f.mass,
            f.mass_inside,
            opt(f.centroid_m),
            opt(f.top_m),
            f.outflow_rate,
            opt(f.fuel_mass),
            opt(f.flame_volume),
        );
    }
    out
}

/// The 1-minute load average, from `sysctl -n vm.loadavg` (macOS prints
/// `{ 1.23 1.45 1.67 }`). Negative when it cannot be read, since JSON has
/// no NaN.
pub fn load_average() -> f64 {
    Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .split_whitespace()
                .find_map(|t| t.parse::<f64>().ok())
        })
        .unwrap_or(-1.0)
}

/// Every scene, resolution and solver the results table needs (spec §1;
/// `fire` from the 2b-4 spec §6.2).
pub const SCENES: [&str; 4] = ["plume", "plume_collider", "plume_wind", "fire"];
/// The scenes whose smoke comes from an emitter. In `fire` it comes from
/// burning, so the emitted-mass note leaves `fire` out.
pub const SMOKE_SCENES: [&str; 3] = ["plume", "plume_collider", "plume_wind"];
/// The scenes the latency table needs (2b-3b spec §2), which predates fire.
pub const LATENCY_SCENES: [&str; 3] = SMOKE_SCENES;
pub const RESOLUTIONS: [u32; 3] = [64, 128, 256];
pub const SOLVERS: [&str; 2] = ["ember", "mantaflow"];

/// A run whose load average was above this should be timed again idle.
pub const IDLE_LOAD: f64 = 2.0;

/// The frames the table reports, 1-based: the last emitting frame and the
/// last frame.
const REPORT_FRAMES: [usize; 2] = [60, 120];

/// The frames at which the table reports drift, 1-based. 80 is meant to come
/// before most of the outflow, but in some runs smoke reaches the outflow
/// plane earlier, so the frame-80 cell says when a run's outflow starts and
/// how much of its drift the outflow estimate credits; 120 includes the
/// whole outflow period.
const DRIFT_FRAMES: [usize; 2] = [80, 120];

/// Below this mass (the density sum, in the solvers' units) a frame's domain
/// is treated as empty: its centroid, top and per-cell values are shown as
/// "—", since they would describe a few stray cells rather than a plume.
pub const MASS_FLOOR: f64 = 1e-9;

/// The frames, 1-based, over which the Notes compare the two solvers'
/// emitted mass: after the first few frames' start-up and before the
/// plumes have diverged.
const EMISSION_CHECK_FRAMES: std::ops::RangeInclusive<usize> = 12..=24;

/// The commit every summary was built from, or one line per summary,
/// `{solver}-{scene}-{resolution}: {commit}`, when they differ. A table
/// mixing commits would compare different code.
pub fn single_commit(summaries: &[RunSummary]) -> Result<String, Vec<String>> {
    one_commit(summaries.iter().map(|s| {
        (
            format!("{}-{}-{}", s.solver, s.scene, s.resolution),
            &*s.commit,
        )
    }))
}

/// The commit every labelled item shares, or one `{label}: {commit}` line
/// per item, sorted, when they differ.
fn one_commit<'a>(items: impl Iterator<Item = (String, &'a str)>) -> Result<String, Vec<String>> {
    let items: Vec<(String, &str)> = items.collect();
    let first = items
        .first()
        .map(|(_, c)| c.to_string())
        .unwrap_or_default();
    if items.iter().all(|(_, c)| *c == first) {
        return Ok(first);
    }
    let mut lines: Vec<String> = items.iter().map(|(l, c)| format!("{l}: {c}")).collect();
    lines.sort();
    Err(lines)
}

/// Where and on what the results were produced, for the table's header.
#[derive(Debug, Clone, PartialEq)]
pub struct Context {
    pub machine: String,
    pub os: String,
    pub commit: String,
    pub blender: String,
    pub date: String,
}

/// The commit each scene's summaries were built from: one commit when
/// every scene shares it, or else each commit followed by its scenes, in
/// [`SCENES`] order, e.g. "31cfeb2 (`plume`, `plume_collider`,
/// `plume_wind`); abc1234 (`fire`)". A scene's table compares its two
/// solvers, so each scene must come from one commit, and the lines naming
/// every summary of each scene that mixes commits are the error. Scenes may
/// differ, since a scene added later is run at a later commit.
pub fn scene_commits(summaries: &[RunSummary]) -> Result<String, Vec<String>> {
    let mut scenes: Vec<&str> = SCENES.to_vec();
    for s in summaries {
        if !scenes.contains(&s.scene.as_str()) {
            scenes.push(&s.scene);
        }
    }
    let mut groups: Vec<(String, Vec<&str>)> = Vec::new();
    let mut errors = Vec::new();
    for scene in scenes {
        let of_scene: Vec<&RunSummary> = summaries.iter().filter(|s| s.scene == scene).collect();
        if of_scene.is_empty() {
            continue;
        }
        let labelled = of_scene.iter().map(|s| {
            (
                format!("{}-{}-{}", s.solver, s.scene, s.resolution),
                &*s.commit,
            )
        });
        match one_commit(labelled) {
            Ok(commit) => match groups.iter_mut().find(|(c, _)| *c == commit) {
                Some((_, names)) => names.push(scene),
                None => groups.push((commit, vec![scene])),
            },
            Err(lines) => errors.extend(lines),
        }
    }
    if !errors.is_empty() {
        errors.sort();
        return Err(errors);
    }
    Ok(match groups.as_slice() {
        [] => String::new(),
        [(commit, _)] => commit.clone(),
        _ => groups
            .iter()
            .map(|(commit, names)| {
                let names: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
                format!("{commit} ({})", names.join(", "))
            })
            .collect::<Vec<_>>()
            .join("; "),
    })
}

/// The whole table, or the list of result files that are missing. Never a
/// partial table (spec §7).
///
/// Missing runs are named `{solver}-{scene}-{resolution}`, the stem of the
/// file [`summary_path`] would give.
pub fn results_markdown(summaries: &[RunSummary], ctx: &Context) -> Result<String, Vec<String>> {
    let find = |solver: &str, scene: &str, res: u32| {
        summaries
            .iter()
            .find(|s| s.solver == solver && s.scene == scene && s.resolution == res)
    };
    let mut missing = Vec::new();
    for solver in SOLVERS {
        for scene in SCENES {
            for res in RESOLUTIONS {
                if find(solver, scene, res).is_none() {
                    missing.push(format!("{solver}-{scene}-{res}"));
                }
            }
        }
    }
    if !missing.is_empty() {
        return Err(missing);
    }

    let mut md = String::new();
    let _ = write!(
        md,
        "# Ember against Mantaflow\n\n\
         - Machine: {}\n\
         - OS: {}\n\
         - Ember commit: {}\n\
         - Blender: {}\n\
         - Date: {}\n\n",
        ctx.machine, ctx.os, ctx.commit, ctx.blender, ctx.date
    );
    let loaded: Vec<String> = summaries_in_order(&find)
        .filter(|s| s.load_before > IDLE_LOAD || s.load_after > IDLE_LOAD)
        .map(|s| {
            format!(
                "`{}-{}-{}` ({:.1} → {:.1})",
                s.solver, s.scene, s.resolution, s.load_before, s.load_after
            )
        })
        .collect();
    let _ = writeln!(
        md,
        "Timings from a run whose 1-minute load average was above {IDLE_LOAD} before or \
         after its timed runs should be redone on an idle machine. Such runs: {}.\n",
        if loaded.is_empty() {
            "none".to_owned()
        } else {
            loaded.join(", ")
        }
    );
    md.push_str("## Summary\n\n_to be written by hand from the tables_\n\n");

    for scene in SCENES {
        let _ = write!(
            md,
            "## `{scene}`\n\n\
             | solver | cells | frame ms (median, min–max) | peak MiB \
             | div. RMS 1/s (60 / 120) | measured cells (60 / 120) \
             | kinetic energy m⁵/s² (60 / 120) | KE per cell m⁵/s² (60 / 120) \
             | vorticity m³/s (60 / 120) | vorticity per cell m³/s (60 / 120) \
             | centroid m (60 / 120) | top m (60 / 120) \
             | drift at 80 (% of mass at 60) | drift at 120 (% of mass at 60) |\n\
             |---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        );
        for res in RESOLUTIONS {
            for solver in SOLVERS {
                let s = find(solver, scene, res).expect("checked above");
                md.push_str(&row(s));
            }
        }
        md.push('\n');
    }

    md.push_str(&fire_table(&find));
    md.push_str(&notes(&find));
    Ok(md)
}

/// The fire scene's fuel and flame at frames 30, 60 and 90 (2b-4 spec §6.2).
fn fire_table<'a>(find: &impl Fn(&str, &str, u32) -> Option<&'a RunSummary>) -> String {
    let mut md = String::from(
        "## `fire`: fuel and flame\n\n\
         | solver | cells | fuel mass (30 / 60 / 90) | flame volume m³ (30 / 60 / 90) |\n\
         |---|---|---|---|\n",
    );
    for res in RESOLUTIONS {
        for solver in SOLVERS {
            let Some(s) = find(solver, "fire", res) else {
                continue;
            };
            let at = |f: fn(&FrameMetrics) -> Option<f64>| {
                [30, 60, 90]
                    .map(|n| s.frames.get(n - 1).and_then(f).map_or("—".to_owned(), sig))
                    .join(" / ")
            };
            let _ = writeln!(
                md,
                "| {solver} | {res}³ | {} | {} |",
                at(|m| m.fuel_mass),
                at(|m| m.flame_volume)
            );
        }
    }
    md.push('\n');
    md
}

/// Ember's mass over Mantaflow's, as the least and greatest ratio over
/// `frames` (1-based) and every scene and resolution. `None` if no pair of
/// runs has a frame in range with Mantaflow mass above [`MASS_FLOOR`].
fn mass_ratio_range<'a>(
    find: &impl Fn(&str, &str, u32) -> Option<&'a RunSummary>,
    frames: std::ops::RangeInclusive<usize>,
) -> Option<(f64, f64)> {
    let mut range: Option<(f64, f64)> = None;
    for scene in SMOKE_SCENES {
        for res in RESOLUTIONS {
            let (Some(e), Some(m)) = (find("ember", scene, res), find("mantaflow", scene, res))
            else {
                continue;
            };
            for n in frames.clone() {
                let (Some(fe), Some(fm)) = (e.frames.get(n - 1), m.frames.get(n - 1)) else {
                    continue;
                };
                if fm.mass < MASS_FLOOR {
                    continue;
                }
                let r = fe.mass / fm.mass;
                range = Some(range.map_or((r, r), |(lo, hi)| (lo.min(r), hi.max(r))));
            }
        }
    }
    range
}

/// The summaries in table order: solver, then scene, then resolution.
fn summaries_in_order<'a>(
    find: &'a impl Fn(&str, &str, u32) -> Option<&'a RunSummary>,
) -> impl Iterator<Item = &'a RunSummary> + 'a {
    SOLVERS.into_iter().flat_map(move |solver| {
        SCENES.into_iter().flat_map(move |scene| {
            RESOLUTIONS
                .into_iter()
                .filter_map(move |res| find(solver, scene, res))
        })
    })
}

/// One table row.
fn row(s: &RunSummary) -> String {
    let at = |frame: usize| s.frames.get(frame - 1);
    let pair = |f: &dyn Fn(&FrameMetrics) -> Option<f64>| {
        let [a, b] = REPORT_FRAMES.map(|n| at(n).and_then(f).map_or("—".to_owned(), sig));
        format!("{a} / {b}")
    };
    let runs = if s.runs == 1 {
        "1 run".to_owned()
    } else {
        format!("median of {}", s.runs)
    };
    let cells = {
        let [a, b] =
            REPORT_FRAMES.map(|n| at(n).map_or("—".to_owned(), |f| f.measured_cells.to_string()));
        format!("{a} / {b}")
    };
    // Where the domain is all but empty, the shape and per-cell values
    // describe stray cells, not a plume.
    let smoky = |f: &FrameMetrics| f.mass >= MASS_FLOOR;
    let per_cell = |v: f64, f: &FrameMetrics| {
        (f.measured_cells > 0 && smoky(f)).then(|| v / f.measured_cells as f64)
    };
    // drift[0] is the last emitting frame, and the percentage is of the mass
    // inside the outflow planes then.
    let cut_off = REPORT_FRAMES[0];
    let base = at(cut_off).map_or(0.0, |f| f.mass_inside);
    let drift_at = |frame: usize| match s.drift.get(frame - cut_off) {
        Some(&d) if base > 0.0 => format!("{} ({:+.1}%)", sig(d), 100.0 * d / base),
        Some(&d) => format!("{} (—)", sig(d)),
        None => "—".to_owned(),
    };
    let [mut drift_a, drift_b] = DRIFT_FRAMES.map(drift_at);
    if let Some(note) = early_outflow(s, DRIFT_FRAMES[0]) {
        drift_a.push_str(&note);
    }
    format!(
        "| {} | {}³ | {} ({}–{}), {runs} | {:.1} | {} | {cells} | {} | {} | {} | {} | {} | {} \
         | {drift_a} | {drift_b} |\n",
        s.solver,
        s.resolution,
        sig(s.frame_ms_median),
        sig(s.frame_ms_min),
        sig(s.frame_ms_max),
        s.peak_bytes as f64 / f64::from(1u32 << 20),
        pair(&|f| Some(f.divergence_rms)),
        pair(&|f| Some(f.kinetic_energy)),
        pair(&|f| per_cell(f.kinetic_energy, f)),
        pair(&|f| Some(f.vorticity)),
        pair(&|f| per_cell(f.vorticity, f)),
        pair(&|f| f.centroid_m.filter(|_| smoky(f))),
        pair(&|f| f.top_m.filter(|_| smoky(f))),
    )
}

/// For a run whose outflow rate is non-zero at or before `frame` (1-based):
/// the first such frame, and the share of the frame-60 mass inside the
/// outflow planes that the drift at `frame` credits as outflow, which rests
/// on the frame-resolution outflow estimate. `None` when no outflow has
/// started by then.
fn early_outflow(s: &RunSummary, frame: usize) -> Option<String> {
    let cut_off = REPORT_FRAMES[0];
    let first = s
        .frames
        .iter()
        .take(frame)
        .position(|f| f.outflow_rate != 0.0)?
        + 1;
    let inside = |n: usize| s.frames.get(n - 1).map(|f| f.mass_inside);
    let (Some(d), Some(then), Some(now)) =
        (s.drift.get(frame - cut_off), inside(cut_off), inside(frame))
    else {
        return Some(format!("; outflow from frame {first}"));
    };
    // drift = mass_inside(frame) + outflow − mass_inside(60).
    let outflow = d - (now - then);
    if then > 0.0 {
        Some(format!(
            "; outflow from frame {first} credits {:.1}%",
            100.0 * outflow / then
        ))
    } else {
        Some(format!("; outflow from frame {first}"))
    }
}

/// Three significant figures, in scientific notation outside [1e-3, 1e5).
fn sig(v: f64) -> String {
    let a = v.abs();
    if v == 0.0 || !v.is_finite() {
        return format!("{v}");
    }
    if (1e-3..1e5).contains(&a) {
        let decimals = (2 - a.log10().floor() as i32).max(0) as usize;
        format!("{v:.decimals$}")
    } else {
        format!("{v:.2e}")
    }
}

/// The resolution the latency table reports (2b-3b spec §2).
pub const LATENCY_RESOLUTION: u32 = 128;

/// One latency point: the seconds from a parameter change to frame `frame`
/// ready, for every run. The N = 1 point's median is the time to the first
/// frame.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LatencyPoint {
    pub frame: u32,
    pub runs_s: Vec<f64>,
    pub median_s: f64,
    /// Mantaflow only: in the point's last run, the seconds from frame N's
    /// file being written (its mtime) to `bake_all` returning. Absent from
    /// runs made before it was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_after_write_s: Option<f64>,
}

/// One solver's latency run of one scene (2b-3b spec §2).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LatencySummary {
    /// "ember" or "mantaflow".
    pub solver: String,
    pub scene: String,
    pub resolution: u32,
    pub commit: String,
    /// 1-minute load average before and after the timed runs, both taken
    /// after the untimed warm-up.
    pub load_before: f64,
    pub load_after: f64,
    /// Blender's version string for Mantaflow runs; `None` for Ember.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blender: Option<String>,
    /// Ember only: pipelines first compiled during the timed runs, after
    /// the warm-up. Anything above 0 put a shader compile inside a timing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipelines_compiled: Option<usize>,
    pub points: Vec<LatencyPoint>,
}

/// `{dir}/latency-{solver}-{scene}-{resolution}.json`.
pub fn latency_path(dir: &Path, solver: &str, scene: &str, resolution: u32) -> PathBuf {
    dir.join(format!("latency-{solver}-{scene}-{resolution}.json"))
}

/// Write `s` as JSON at [`latency_path`], returning the path.
pub fn write_latency(dir: &Path, s: &LatencySummary) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = latency_path(dir, &s.solver, &s.scene, s.resolution);
    let json = serde_json::to_string_pretty(s).map_err(std::io::Error::other)?;
    std::fs::write(&path, json + "\n")?;
    Ok(path)
}

/// The Blender version every Mantaflow latency summary names, or one
/// `{label}: {version}` line per summary when they differ: a table mixing
/// Blender versions would compare different Mantaflows.
pub fn single_latency_blender(summaries: &[LatencySummary]) -> Result<String, Vec<String>> {
    let labelled: Vec<(String, &str)> = summaries
        .iter()
        .filter(|s| s.solver == "mantaflow")
        .map(|s| {
            (
                format!("latency-{}-{}-{}", s.solver, s.scene, s.resolution),
                s.blender.as_deref().unwrap_or("unknown"),
            )
        })
        .collect();
    one_commit(labelled.into_iter())
}

/// Like [`single_commit`], for latency summaries.
pub fn single_latency_commit(summaries: &[LatencySummary]) -> Result<String, Vec<String>> {
    one_commit(summaries.iter().map(|s| {
        (
            format!("latency-{}-{}-{}", s.solver, s.scene, s.resolution),
            &*s.commit,
        )
    }))
}

/// `docs/bench/latency.md`: one table per scene at [`LATENCY_RESOLUTION`]
/// of N against both solvers' median seconds and Mantaflow / Ember, or the
/// stems of the missing latency files. Never a partial table.
/// `recipe_load` is `sysctl -n vm.loadavg` from before the benchmark
/// started, when known: the load that did not come from the benchmark.
pub fn latency_markdown(
    summaries: &[LatencySummary],
    ctx: &Context,
    recipe_load: Option<&str>,
) -> Result<String, Vec<String>> {
    let res = LATENCY_RESOLUTION;
    let find = |solver: &str, scene: &str| {
        summaries
            .iter()
            .find(|s| s.solver == solver && s.scene == scene && s.resolution == res)
    };
    let missing: Vec<String> = SOLVERS
        .into_iter()
        .flat_map(|solver| LATENCY_SCENES.into_iter().map(move |scene| (solver, scene)))
        .filter(|(solver, scene)| find(solver, scene).is_none())
        .map(|(solver, scene)| format!("latency-{solver}-{scene}-{res}"))
        .collect();
    if !missing.is_empty() {
        return Err(missing);
    }

    let mut md = String::new();
    let _ = write!(
        md,
        "# Latency: a parameter change to frame N\n\n\
         - Machine: {}\n\
         - OS: {}\n\
         - Ember commit: {}\n\
         - Blender: {}\n\
         - Date: {}\n\n",
        ctx.machine, ctx.os, ctx.commit, ctx.blender, ctx.date
    );
    let _ = write!(
        md,
        "| run | load before | load after | above {IDLE_LOAD} |\n|---|---|---|---|\n"
    );
    let mut loaded = Vec::new();
    for scene in LATENCY_SCENES {
        for solver in SOLVERS {
            let s = find(solver, scene).expect("checked above");
            let high = s.load_before > IDLE_LOAD || s.load_after > IDLE_LOAD;
            let flag = if high { "**yes**" } else { "no" };
            let _ = writeln!(
                md,
                "| `latency-{solver}-{scene}-{res}` | {:.2} | {:.2} | {flag} |",
                s.load_before, s.load_after
            );
            if high {
                loaded.push(format!("`latency-{solver}-{scene}-{res}`"));
            }
        }
    }
    let _ = writeln!(
        md,
        "\nTimings from a run whose 1-minute load average was above {IDLE_LOAD} before or \
         after its timed runs should be redone on an idle machine. Such runs: {}.\n",
        if loaded.is_empty() {
            "none".to_owned()
        } else {
            loaded.join(", ")
        }
    );
    for scene in LATENCY_SCENES {
        let e = find("ember", scene).expect("checked above");
        if let Some(n) = e.pipelines_compiled.filter(|&n| n > 0) {
            let _ = writeln!(
                md,
                "**Warning:** Ember compiled {n} pipelines during `{scene}`'s timed runs, so \
                 some timings include a shader compile.\n"
            );
        }
    }
    for scene in LATENCY_SCENES {
        let e = find("ember", scene).expect("checked above");
        let m = find("mantaflow", scene).expect("checked above");
        let _ = write!(md, "## `{scene}` ({res}³)\n\n");
        match e.points.iter().find(|p| p.frame == 1) {
            Some(first) => {
                let _ = write!(
                    md,
                    "Ember's time to its first frame, graph construction included, is the \
                     N = 1 median: {} s.\n\n",
                    sig(first.median_s)
                );
            }
            None => md.push_str(
                "Ember's file has no N = 1 point, so its time to the first frame is not \
                 reported.\n\n",
            ),
        }
        md.push_str(
            "| N | Ember s (median) | Mantaflow s (median) | Mantaflow / Ember |\n\
             |---|---|---|---|\n",
        );
        for p in &e.points {
            let (manta, ratio) = match m.points.iter().find(|q| q.frame == p.frame) {
                Some(q) => (
                    sig(q.median_s),
                    format!("{}×", sig(q.median_s / p.median_s)),
                ),
                None => ("—".to_owned(), "—".to_owned()),
            };
            let _ = writeln!(
                md,
                "| {} | {} | {manta} | {ratio} |",
                p.frame,
                sig(p.median_s)
            );
        }
        md.push('\n');
    }
    md.push_str(&latency_notes(summaries, recipe_load));
    Ok(md)
}

/// Each solver's runs per N, e.g. "N = 1: 5, N = 24: 5", from the first
/// scene that has that solver; the runs are the same in every scene.
fn runs_per_point(summaries: &[LatencySummary], solver: &str) -> String {
    summaries
        .iter()
        .find(|s| s.solver == solver)
        .map(|s| {
            s.points
                .iter()
                .map(|p| format!("{} at N = {}", p.runs_s.len(), p.frame))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default()
}

/// What the latency table does and does not compare, and why its load
/// readings are not idle ones.
fn latency_notes(summaries: &[LatencySummary], recipe_load: Option<&str>) -> String {
    let res = LATENCY_RESOLUTION;
    let ember_runs = runs_per_point(summaries, "ember");
    let manta_runs = runs_per_point(summaries, "mantaflow");
    let find = |solver: &str, scene: &str| {
        summaries
            .iter()
            .find(|s| s.solver == solver && s.scene == scene && s.resolution == res)
    };
    let longest = summaries
        .iter()
        .filter(|s| s.solver == "mantaflow")
        .flat_map(|s| s.points.iter().flat_map(|p| p.runs_s.iter().copied()))
        .fold(0.0, f64::max);
    // An Ember run's load_before equal to the previous scene's Mantaflow
    // load_after shows the inheritance directly. Ember reads `sysctl`, which
    // prints two decimals, and Mantaflow Python's full-precision
    // `os.getloadavg`, so they are compared at two decimals.
    let inherited: Vec<String> = LATENCY_SCENES
        .windows(2)
        .filter_map(|w| {
            let (m, e) = (find("mantaflow", w[0])?, find("ember", w[1])?);
            ((e.load_before - m.load_after).abs() < 0.005).then(|| {
                format!(
                    "Ember `{}`'s {:.2} is exactly Mantaflow `{}`'s `load_after`",
                    w[1], e.load_before, w[0]
                )
            })
        })
        .collect();
    let inherited = if inherited.is_empty() {
        String::new()
    } else {
        format!(" ({})", inherited.join("; "))
    };
    let outside = match recipe_load {
        Some(l) => format!(
            "There was outside load too: before the recipe's load wait, `vm.loadavg` \
             (1, 5 and 15 minutes) was `{l}`."
        ),
        None => "The load before the recipe's load wait was not recorded.".to_owned(),
    };
    let gaps: Vec<f64> = summaries
        .iter()
        .filter(|s| s.solver == "mantaflow")
        .flat_map(|s| s.points.iter().filter_map(|p| p.return_after_write_s))
        .collect();
    let gap = if gaps.is_empty() {
        "These runs predate recording how long `bake_all` returns after frame N's file is \
         written (`return_after_write_s`), so how much the call's return adds is not known \
         here."
            .to_owned()
    } else {
        let lo = gaps.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = gaps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        format!(
            "`bake_all` returned {}–{} s after frame N's file was written \
             (`return_after_write_s`, each point's last run), so the call's return, not the \
             file, sets the time.",
            sig(lo),
            sig(hi)
        )
    };
    format!(
        "## Notes\n\n\
- **The change.** Every run changes the emitter's density (Ember's `density_rate`, Mantaflow's \
flow density, each ×(1 + 0.01·(run + 1))), so no run can reuse an earlier result. Each is the \
median of its runs ({ember_runs} for Ember; {manta_runs} for Mantaflow).\n\
- **Ember** runs in one warm process, as a live daemon does: one GPU device, registry and \
pipeline cache throughout, warmed by one untimed evaluation of the unchanged scene to frame 24. \
Each run times graph construction plus `eval_frame` for frames 1..N in a fresh field pool and \
state, ending with a blocking GPU wait. Nothing is read back or written to disk during the \
timed interval.\n\
- **Mantaflow's time is a re-bake in an open Blender with its cache freed**, excluding \
Blender's startup and the scene's construction. After one untimed bake of the unchanged scene \
to frame 24, each run frees the cache (`fluid.free_all`, checked to leave no data file), sets \
the cache's last frame to N and times `fluid.bake_all` from just before the call to frame N's \
data file existing: the later of the call returning and the file's modification time. The time \
therefore includes Mantaflow writing N uncompressed {res}³ OpenVDB files, one per frame with \
every grid, as frame times in `results.md` include writing the cache. {gap} A run whose frame \
N file matches another run's, apart from the VDB header's random UUID, is refused.\n\
- **Loads are not idle readings, and every figure here needs an idle rerun.** Each is the \
1-minute load average. Mantaflow's `load_before` is taken right after its own warm-up bake, \
and its `load_after` after up to {longest:.0} s of multi-threaded baking, so both include \
Mantaflow's own work. Each Ember run after the first starts right after the previous scene's \
Blender run, so its `load_before` inherits that load{inherited}. {outside} Mantaflow is \
CPU-bound and Ember mostly GPU-bound, so CPU contention probably slows Mantaflow more than \
Ember and may inflate the ratio in Ember's favour: neither the seconds nor the ratios should be \
read as idle figures.\n"
    )
}

/// A pressure solve as the report names it, e.g. "MGPCG ×4".
pub fn pressure_label(p: PressureSolve) -> String {
    match p {
        PressureSolve::GaussSeidel(n) => format!("Gauss–Seidel ×{n}"),
        PressureSolve::Multigrid(n) => format!("multigrid V-cycles ×{n}"),
        PressureSolve::Mgpcg(n) => format!("MGPCG ×{n}"),
    }
}

/// What every number in the table does and does not compare
/// (`docs/bench/mantaflow-notes.md`). The emitted-mass comparison is
/// computed from the runs, so it cannot go stale when they are redone.
fn notes<'a>(find: &impl Fn(&str, &str, u32) -> Option<&'a RunSummary>) -> String {
    let ratio = |frames: std::ops::RangeInclusive<usize>| {
        mass_ratio_range(find, frames)
            .map_or("—".to_owned(), |(lo, hi)| format!("{lo:.2}–{hi:.2}"))
    };
    let (first, last) = (EMISSION_CHECK_FRAMES.start(), EMISSION_CHECK_FRAMES.end());
    let cut_off = REPORT_FRAMES[0];
    let emitted = format!(
        "Emitted mass matches while both solvers are still emitting and the plumes have not \
         yet diverged: over frames {first}–{last}, Ember's mass is {}× Mantaflow's across \
         every run. By frame {cut_off} the ratio is {}×, since it then also carries what \
         each solver's advection gains or loses (most of all in `plume_wind`).",
        ratio(EMISSION_CHECK_FRAMES),
        ratio(cut_off..=cut_off),
    );
    let floor = MASS_FLOOR;
    let preview = Quality::Preview.params();
    let preview_pressure = pressure_label(preview.pressure());
    let preview_method = if preview.pressure_solver == PressureSolver::Mgpcg {
        "the same method for a fixed count"
    } else {
        "a fixed count"
    };
    let preview_mass = if preview.conserve_mass {
        ", and then its global mass correction on density and temperature"
    } else {
        ", with no mass correction"
    };
    format!(
        "## Notes\n\n\
- **Velocity metrics** (divergence, kinetic energy, vorticity) cover only measured cells: \
those whose whole 3×3×3 neighbourhood is inside the domain, outside any collider and holds \
smoke (density > 1e-6). The rule is the same for both solvers, because Mantaflow's cache \
stores velocity only where there is smoke (spec §4.1). The measured-cell count says how \
much of each field that is.\n\
- **Per-cell values.** The solvers' smoky regions differ in size, so their measured cells \
differ too, and the kinetic energy and vorticity totals partly measure that size. The totals \
divided by the measured-cell count are the fairer comparison.\n\
- **Near-empty domains.** Where a frame's mass is below {floor:e}, its centroid, top and \
per-cell values are shown as —, since they would describe a few stray cells.\n\
- **Wind.** Mantaflow's wind field acts only on cells that hold smoke; Ember's air relaxes \
towards the ambient airflow in every cell. `plume_wind` compares the plume's shape, not a matched force field.\n\
- **Heat.** Mantaflow's emitter heat is held at a set value (Ember's emitter-centre heat at \
frame 24), not added at Ember's rate, so Mantaflow's emitter is hotter before frame 24 and \
cooler after it. Plume centroid and top carry that difference. {emitted}\n\
- **Pressure.** Mantaflow solves with multigrid-preconditioned conjugate gradients to a \
tolerance; Ember runs {preview_method}, the preview preset's {preview_pressure} per \
substep{preview_mass}.\n\
- **Drift** is (mass inside the outflow planes + outflow since frame 60) − that mass at frame \
60, with outflow estimated at frame resolution as the net upwind flux out through a plane two \
cells in from each open face (the top, and in `plume_wind` both x sides), and mass summed \
over the cells inside those planes (2b-3 spec §4.3, 2b-3c spec §6). Emission stops \
after frame 60, so a perfect solver drifts 0. Where no outflow has started by frame 80, drift \
at 80 is mass gained or lost inside the domain. Where it has, the frame-80 cell names the \
first frame whose outflow rate is non-zero and the share of the frame-60 mass that the \
outflow estimate credits by frame 80; that share rests on the frame-resolution estimate, and \
so does that part of the drift. Drift at 120 covers the whole outflow period.\n\
- **Frame times** exclude frame 1. Ember's frame is `eval_frame` plus a blocking GPU wait; \
Mantaflow's is the difference between consecutive cache files' modification times, so it \
includes writing the cache. 256³ is one run of each solver, the other resolutions the median \
of three runs' medians; the min–max range pools every timed frame.\n\
- **Peak memory** is in MiB (2²⁰ bytes), and the two solvers' figures count different \
things. Ember's is the field pool's allocated bytes, with the frame cache off: textures only, \
not buffers, pipelines or the driver. Mantaflow's is Blender's peak resident memory while \
baking, minus the same scene's peak without a bake.\n\
- **Fire.** `fire` emits fuel, not smoke, so its mass, drift, centroid and top describe the \
smoke made by burning, and it is left out of the emitted-mass comparison above. Mantaflow's \
burn clamps density to [0, 1] in every cell on every step; Ember does not clamp it. Fuel mass \
is Σ fuel dV. Mantaflow's cache stores fuel only where it stores density (above 1e-6), so fuel \
in a cell with no smoke is not counted there. Flame volume is the volume of cells whose flame \
is above {flame}; both solvers' flame is √react, with Ember's react clamped to [0, 1].\n",
        flame = FLAME_THRESHOLD,
    )
}
